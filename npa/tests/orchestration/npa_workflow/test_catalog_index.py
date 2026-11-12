"""Unit and catalog-level tests for the workflow catalog indexer."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from npa.orchestration.npa_workflow import catalog_index
from npa.orchestration.npa_workflow.catalog_index import (
    CATALOG_DISPLAY_EXCLUSIONS,
    index_catalog,
    index_workflow,
)

GPU_PROFILE = {
    "cloud": "kubernetes",
    "accelerators": "B200:1",
    "cpus": 8,
    "memory": "64Gi",
}
CPU_PROFILE = {"cloud": "kubernetes", "cpus": 4, "memory": "16Gi"}


def _leaf(
    name: str,
    resources: str = "gpu",
    *,
    terminal: bool = False,
    next_state: str | None = None,
    sharded: bool = False,
) -> dict:
    """One executing state shaped like the checked-in catalog specs.

    Uses a raw ``run:`` command rather than a toolRef so fixtures stay
    independent of the workbench tool catalog.
    """

    state = {
        "description": f"leaf {name}",
        "resources": resources,
        "run": {"shell": "true"},
    }
    if terminal:
        state["terminal"] = True
    elif next_state:
        state["next"] = next_state
    if sharded:
        state["multi_node_mode"] = "sharded"
    return state


def _write_spec(
    directory: Path, file_name: str, spec: dict, sub: str | None = None
) -> Path:
    """Write one catalog-shaped spec file and return its path."""

    target = directory if sub is None else directory / sub
    target.mkdir(parents=True, exist_ok=True)
    path = target / file_name
    path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return path


def _minimal_spec(
    name: str,
    description: str = "demo",
    profiles: dict | None = None,
) -> dict:
    """Single-stage spec; the leaf uses the first declared profile."""

    leaf_resources = "gpu" if profiles else "default"
    if profiles:
        leaf_resources = next(iter(profiles))
    return {
        "apiVersion": "npa.workflow/v0.0.1",
        "kind": "Workflow",
        "metadata": {"name": name, "description": description},
        "config": {"bucket": "example-bucket"},
        "initial": "only",
        "states": {"only": _leaf("only", leaf_resources, terminal=True)},
        **({"resources": profiles} if profiles else {}),
    }


class TestPhaseCount:
    def test_parallel_fanout_collapses_to_one_phase(self, tmp_path: Path) -> None:
        spec = {
            "apiVersion": "npa.workflow/v0.0.1",
            "kind": "Workflow",
            "metadata": {"name": "fanout", "description": "fanout demo"},
            "config": {"bucket": "example-bucket"},
            "resources": {"gpu": GPU_PROFILE, "cpu": CPU_PROFILE},
            "initial": "wave",
            "states": {
                "wave": {
                    "parallel": ["shard-0", "shard-1", "shard-2", "shard-3"],
                    "next": "join",
                },
                **{f"shard-{i}": _leaf(f"shard-{i}") for i in range(4)},
                "join": _leaf("join", "cpu", terminal=True),
            },
        }
        entry = index_workflow(_write_spec(tmp_path, "fanout.yaml", spec))
        assert entry.phases == 2

    def test_sequence_children_count_individually(self, tmp_path: Path) -> None:
        spec = {
            "apiVersion": "npa.workflow/v0.0.1",
            "kind": "Workflow",
            "metadata": {"name": "sequence", "description": "sequence demo"},
            "config": {"bucket": "example-bucket"},
            "resources": {"gpu": GPU_PROFILE},
            "initial": "chain",
            "states": {
                "chain": {"sequence": ["one", "two", "three"]},
                "one": _leaf("one", next_state="two"),
                "two": _leaf("two", next_state="three"),
                "three": _leaf("three", terminal=True),
            },
        }
        entry = index_workflow(_write_spec(tmp_path, "sequence.yaml", spec))
        assert entry.phases == 3

    def test_loops_do_not_multiply(self, tmp_path: Path) -> None:
        spec = {
            "apiVersion": "npa.workflow/v0.0.1",
            "kind": "Workflow",
            "metadata": {"name": "looped", "description": "loop demo"},
            "config": {"bucket": "example-bucket", "iterations": 5},
            "resources": {"gpu": GPU_PROFILE},
            "initial": "refine",
            "states": {
                "refine": {
                    "sequence": ["one", "two"],
                    "loop": {"max": "{{config.iterations}}"},
                    "next": "done",
                },
                "one": _leaf("one"),
                "two": _leaf("two"),
                "done": _leaf("done", terminal=True),
            },
        }
        entry = index_workflow(_write_spec(tmp_path, "looped.yaml", spec))
        assert entry.phases == 3


class TestModality:
    def test_gpu_only(self, tmp_path: Path) -> None:
        entry = index_workflow(
            _write_spec(
                tmp_path,
                "gpu.yaml",
                _minimal_spec("gpu-only", profiles={"gpu": GPU_PROFILE}),
            )
        )
        assert entry.modality == "GPU"

    def test_cpu_only(self, tmp_path: Path) -> None:
        entry = index_workflow(
            _write_spec(
                tmp_path,
                "cpu.yaml",
                _minimal_spec("cpu-only", profiles={"cpu": CPU_PROFILE}),
            )
        )
        assert entry.modality == "CPU"

    def test_mixed_profiles(self, tmp_path: Path) -> None:
        spec = _minimal_spec(
            "mixed", profiles={"gpu": GPU_PROFILE, "cpu": CPU_PROFILE}
        )
        spec["states"]["prepare"] = _leaf("prepare", "cpu", next_state="only")
        spec["initial"] = "prepare"
        entry = index_workflow(_write_spec(tmp_path, "mixed.yaml", spec))
        assert entry.modality == "CPU+GPU"


class TestResourceDerivation:
    def test_accelerator_template_substitutes_config_defaults(
        self, tmp_path: Path
    ) -> None:
        spec = _minimal_spec(
            "templated",
            profiles={
                "gpu": {
                    "cloud": "kubernetes",
                    "accelerators": "{{config.gpu_type}}:{{config.gpu_count}}",
                    "cpus": 8,
                    "memory": "64Gi",
                }
            },
        )
        spec["config"]["gpu_type"] = "B200"
        spec["config"]["gpu_count"] = 2
        entry = index_workflow(_write_spec(tmp_path, "templated.yaml", spec))
        assert entry.gpu_type == "2x B200"

    def test_countless_accelerator_shows_declared_text(self, tmp_path: Path) -> None:
        spec = _minimal_spec(
            "countless",
            profiles={
                "gpu": {
                    "cloud": "kubernetes",
                    # A count-less raw request (count 1 to SkyPilot) passes
                    # validation; the catalog shows it exactly as declared.
                    "accelerators": "RTXPRO6000",
                    "cpus": 8,
                    "memory": "64Gi",
                }
            },
        )
        entry = index_workflow(_write_spec(tmp_path, "countless.yaml", spec))
        assert entry.gpu_type == "RTXPRO6000"

    def test_blank_when_profile_declares_nothing(self, tmp_path: Path) -> None:
        spec = _minimal_spec(
            "bare",
            profiles={"gpu": {"cloud": "kubernetes", "accelerators": "H100:1"}},
        )
        entry = index_workflow(_write_spec(tmp_path, "bare.yaml", spec))
        assert entry.cpus == ""
        assert entry.memory == ""

    def test_max_declared_value_wins(self, tmp_path: Path) -> None:
        spec = _minimal_spec(
            "maxmem",
            profiles={
                "gpu": dict(GPU_PROFILE, memory="96Gi"),
                "big": dict(GPU_PROFILE, accelerators="B200:8", memory="256Gi"),
            },
        )
        spec["states"]["only"]["terminal"] = False
        spec["states"]["only"]["next"] = "heavy"
        spec["states"]["heavy"] = _leaf("heavy", "big", terminal=True)
        entry = index_workflow(_write_spec(tmp_path, "maxmem.yaml", spec))
        assert entry.memory == "256Gi"
        assert entry.gpu_type == "8x B200"

    def test_pvc_profile_flag(self, tmp_path: Path) -> None:
        spec = _minimal_spec(
            "cached",
            profiles={
                "gpu": {
                    "cloud": "kubernetes",
                    "accelerators": "H100:1",
                    "kubernetes": {
                        "pod_config": {
                            "spec": {
                                "volumes": [
                                    {
                                        "name": "cache",
                                        "persistentVolumeClaim": {
                                            "claimName": "demo"
                                        },
                                    }
                                ]
                            }
                        }
                    },
                }
            },
        )
        entry = index_workflow(_write_spec(tmp_path, "cached.yaml", spec))
        assert entry.pvc_cache is True

    def test_empty_dir_volume_is_not_a_pvc(self, tmp_path: Path) -> None:
        spec = _minimal_spec(
            "scratch",
            profiles={
                "gpu": {
                    "cloud": "kubernetes",
                    "accelerators": "H100:1",
                    "kubernetes": {
                        "pod_config": {
                            "spec": {
                                "volumes": [
                                    {"name": "dshm", "emptyDir": {"medium": "Memory"}}
                                ]
                            }
                        }
                    },
                }
            },
        )
        entry = index_workflow(_write_spec(tmp_path, "scratch.yaml", spec))
        assert entry.pvc_cache is False


class TestComplexity:
    def test_heavy_graph_clamps_to_five_stars(self, tmp_path: Path) -> None:
        spec = {
            "apiVersion": "npa.workflow/v0.0.1",
            "kind": "Workflow",
            "metadata": {"name": "heavy", "description": "heavy demo"},
            "config": {"bucket": "example-bucket", "outer": 2, "inner": 3},
            "resources": {
                "gpu": {
                    "cloud": "kubernetes",
                    "accelerators": "B200:8",
                    "cpus": 32,
                    "memory": "256Gi",
                    "num_nodes": 2,
                },
                "cache": {
                    "cloud": "kubernetes",
                    "accelerators": "B200:1",
                    "cpus": 8,
                    "memory": "64Gi",
                    "kubernetes": {
                        "pod_config": {
                            "spec": {
                                "volumes": [
                                    {
                                        "name": "cache",
                                        "persistentVolumeClaim": {
                                            "claimName": "demo"
                                        },
                                    }
                                ]
                            }
                        }
                    },
                },
                "cpu": CPU_PROFILE,
            },
            "initial": "warm",
            "states": {
                "warm": _leaf("warm", "cache", next_state="outer"),
                "outer": {
                    "sequence": ["inner-loop", "gold"],
                    "loop": {"max": "{{config.outer}}"},
                    "next": "publish",
                },
                "inner-loop": {
                    "sequence": ["train", "evaluate"],
                    "loop": {"max": "{{config.inner}}"},
                },
                "train": _leaf("train", "gpu", sharded=True),
                "evaluate": _leaf("evaluate", "gpu", sharded=True),
                "gold": _leaf("gold", "gpu", sharded=True),
                "publish": _leaf("publish", "cpu", terminal=True),
            },
        }
        entry = index_workflow(_write_spec(tmp_path, "heavy.yaml", spec))
        assert entry.complexity == 5
        assert entry.pvc_cache is True
        assert entry.phases == 5

    def test_single_stage_stays_one_star(self, tmp_path: Path) -> None:
        entry = index_workflow(
            _write_spec(
                tmp_path,
                "simple.yaml",
                _minimal_spec("simple", profiles={"gpu": GPU_PROFILE}),
            )
        )
        assert entry.complexity == 1


class TestApplicationLookup:
    def test_word_boundary_keeps_supports_from_ppo(self, tmp_path: Path) -> None:
        spec = _minimal_spec(
            "boundary",
            description="Cosmos3-Nano is public and supports anonymous download.",
            profiles={"gpu": GPU_PROFILE},
        )
        entry = index_workflow(_write_spec(tmp_path, "cosmos3-generate.yaml", spec))
        assert entry.application == "generation"

    def test_sim2real_keyword_requires_workflow_phrase(self, tmp_path: Path) -> None:
        spec = _minimal_spec(
            "artifact",
            description="Publish the reports/sim2real.rrd recording artifact.",
            profiles={"gpu": GPU_PROFILE},
        )
        entry = index_workflow(_write_spec(tmp_path, "artifact.yaml", spec))
        assert entry.application == ""

    def test_unmatched_stays_blank(self, tmp_path: Path) -> None:
        entry = index_workflow(
            _write_spec(
                tmp_path,
                "mystery.yaml",
                _minimal_spec("mystery", profiles={"gpu": GPU_PROFILE}),
            )
        )
        assert entry.application == ""


class TestCatalogIndexing:
    def test_partners_merge_and_exclusions_apply(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        catalog = tmp_path / "workflows"
        for tier in ("main", "testing"):
            _write_spec(
                catalog,
                f"{tier}-demo.yaml",
                _minimal_spec(f"{tier}-demo", profiles={"gpu": GPU_PROFILE}),
                sub=tier,
            )
        _write_spec(
            catalog,
            "encord-demo.yaml",
            _minimal_spec("encord-demo", profiles={"cpu": CPU_PROFILE}),
            sub="partners/encord",
        )
        _write_spec(
            catalog,
            "antioch-demo.yaml",
            _minimal_spec("antioch-demo", profiles={"gpu": GPU_PROFILE}),
            sub="partners/antioch",
        )
        _write_spec(
            catalog,
            "sim2real-two-step-agent.yaml",
            _minimal_spec("sim2real-two-step", profiles={"gpu": GPU_PROFILE}),
            sub="testing",
        )
        _write_spec(
            catalog,
            "not-a-spec.yaml",
            {"apiVersion": "skypilot/v1", "task": "demo"},
            sub="testing",
        )
        monkeypatch.setattr(
            catalog_index,
            "npa_workflow_spec_dirs",
            lambda: (
                catalog / "main",
                catalog / "testing",
                catalog / "partners" / "encord",
                catalog / "partners" / "antioch",
            ),
        )
        entries = index_catalog()
        assert set(entries) == {"main", "testing", "partners"}
        assert [row.name for row in entries["partners"]] == [
            "antioch-demo",
            "encord-demo",
        ]
        assert {row.partner for row in entries["partners"]} == {
            "antioch",
            "encord",
        }
        assert all(row.partner == "" for row in entries["main"])
        assert "sim2real-two-step-agent.yaml" in CATALOG_DISPLAY_EXCLUSIONS
        assert [row.name for row in entries["testing"]] == ["testing-demo"]


class TestRealCatalog:
    """Anchor rows against the checked-in catalog; counts are floors."""

    def test_every_discovered_spec_indexes(self) -> None:
        entries = index_catalog()
        rows = {
            row.name: row for tier_rows in entries.values() for row in tier_rows
        }
        assert set(entries) >= {"main", "testing", "partners", "cosmos-data-factory"}
        assert len(rows) >= 145

    def test_main_tier_anchors(self) -> None:
        rows = {row.name: row for row in index_catalog()["main"]}
        sim2real = rows["sim2real"]
        assert sim2real.phases == 15
        assert sim2real.pvc_cache is True
        assert sim2real.complexity == 5
        assert sim2real.application == "sim2real"
        assert rows["groot-1-7-finetune"].application == "robotics-policy"

    def test_known_rows_across_tiers(self) -> None:
        entries = index_catalog()
        rows = {
            row.name: row for tier_rows in entries.values() for row in tier_rows
        }
        assert rows["xr1-antioch-finetune"].gpu_type == "8x RTXPRO6000"
        assert rows["groot-1-7-finetune"].gpu_type == "2x B200"
        assert rows["policy-public-training"].rel_path == (
            "testing/robot-policy-train-and-serve.yaml"
        )
        assert rows["paidf-event-video-generation"].tier == "cosmos-data-factory"
        assert rows["encord-pull"].tier == "partners"
        assert rows["encord-pull"].partner == "encord"
        assert rows["xr1-antioch-finetune"].partner == "antioch"
        assert rows["sim2real"].partner == ""
        assert "sim2real-two-step-agent" not in rows

    def test_four_pvc_specs_resolve(self) -> None:
        rows = {
            row.name: row
            for tier_rows in index_catalog().values()
            for row in tier_rows
        }
        pvc_names = sorted(name for name, row in rows.items() if row.pvc_cache)
        assert pvc_names == [
            "behavior-challenge-eval",
            "behavior-comet-native-full-training",
            "openpi-pi05-full-droid-finetune",
            "sim2real",
        ]
