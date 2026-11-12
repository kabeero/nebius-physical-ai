"""CLI tests for ``npa workbench workflow filter``."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.cli.workbench.workflow import filter as filter_cli
from npa.orchestration.npa_workflow.catalog_index import CatalogEntry

runner = CliRunner()


def _entry(
    name: str,
    *,
    application: str = "",
    modality: str = "CPU",
    gpu_type: str = "",
    pvc_cache: bool = False,
    tier: str = "main",
    rel_path: str | None = None,
    complexity: int = 1,
) -> CatalogEntry:
    """One synthetic catalog row for hermetic filter tests.

    Args:
        name: Workflow name.
        application: Application category, or "" for uncategorized.
        modality: CPU, GPU, or CPU+GPU.
        gpu_type: Catalog GPU column text, or "" for CPU-only rows.
        pvc_cache: Whether the row requires a PVC.
        tier: Real catalog tier name the row belongs to.
        rel_path: Catalog-relative file path; defaults to ``<tier>/<name>.yaml``.
        complexity: One-to-five star score.
    Returns:
        The synthetic row.
    Raises:
        None.
    """
    return CatalogEntry(
        tier=tier,
        partner="",
        rel_path=rel_path or f"{tier}/{name}.yaml",
        name=name,
        application=application,
        modality=modality,
        cpus="4",
        memory="16Gi",
        gpu_cloud="",
        gpu_type=gpu_type,
        phases=1,
        inputs=False,
        outputs=False,
        pvc_cache=pvc_cache,
        complexity=complexity,
    )


@pytest.fixture()
def synthetic_catalog(monkeypatch: pytest.MonkeyPatch) -> dict[str, list]:
    """Replace the catalog indexer with four fixed main-tier rows.

    Args:
        monkeypatch: pytest patcher for the filter module's indexer.
    Returns:
        The tier-to-rows mapping the command will see.
    Raises:
        None.
    """
    rows = {
        "main": [
            _entry(
                "gpu-train",
                application="robotics-policy",
                modality="GPU",
                gpu_type="2x B200",
                complexity=2,
            ),
            _entry(
                "cached-train",
                application="robotics-policy",
                modality="GPU",
                gpu_type="8x RTXPRO6000",
                pvc_cache=True,
            ),
            _entry("cpu-report", application="data-curation", modality="CPU"),
            _entry("mystery", modality="CPU"),
        ],
    }
    monkeypatch.setattr(filter_cli, "index_catalog", lambda: rows)
    return rows


@pytest.fixture()
def interactive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend the command runs on a terminal that can answer prompts.

    Args:
        monkeypatch: pytest patcher for the filter module's TTY check.
    Returns:
        None.
    Raises:
        None.
    """
    monkeypatch.setattr(filter_cli, "_is_interactive", lambda: True)


class TestGpuNames:
    def test_counts_are_stripped(self) -> None:
        assert filter_cli._gpu_names(_entry("mixed", gpu_type="8x B200, H100")) == {
            "B200",
            "H100",
        }

    def test_single_countless_name(self) -> None:
        assert filter_cli._gpu_names(_entry("one", gpu_type="RTXPRO6000")) == {
            "RTXPRO6000"
        }

    def test_cpu_only_row_has_no_names(self) -> None:
        assert filter_cli._gpu_names(_entry("cpu")) == set()


class TestFlagPaths:
    def test_application_and_no_pvc(self, synthetic_catalog: dict) -> None:
        result = runner.invoke(
            app,
            [
                "workbench",
                "workflow",
                "filter",
                "--application",
                "robotics-policy",
                "--no-pvc",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "gpu-train" in result.output
        assert "cached-train" not in result.output
        assert "cpu-report" not in result.output
        # A blank line separates the result table from anything above it.
        assert "\nMatching workflows" in result.output

    def test_pvc_flag_keeps_pvc_workflows_eligible(
        self, synthetic_catalog: dict
    ) -> None:
        result = runner.invoke(
            app,
            ["workbench", "workflow", "filter", "--application", "data-curation"],
        )
        assert result.exit_code == 0, result.output
        assert "cpu-report" in result.output

    def test_gpu_any_matches_gpu_and_mixed_rows(self, synthetic_catalog: dict) -> None:
        result = runner.invoke(app, ["workbench", "workflow", "filter", "--gpu", "any"])
        assert result.exit_code == 0, result.output
        assert "gpu-train" in result.output
        assert "cached-train" in result.output
        assert "cpu-report" not in result.output
        assert "mystery" not in result.output

    def test_gpu_none_matches_cpu_only_rows(self, synthetic_catalog: dict) -> None:
        result = runner.invoke(
            app, ["workbench", "workflow", "filter", "--gpu", "none"]
        )
        assert result.exit_code == 0, result.output
        assert "cpu-report" in result.output
        assert "mystery" in result.output
        assert "gpu-train" not in result.output

    def test_gpu_name_matches_its_accelerator(self, synthetic_catalog: dict) -> None:
        result = runner.invoke(
            app, ["workbench", "workflow", "filter", "--gpu", "RTXPRO6000"]
        )
        assert result.exit_code == 0, result.output
        assert "cached-train" in result.output
        assert "gpu-train" not in result.output

    def test_uncategorized_matches_blank_application(
        self, synthetic_catalog: dict
    ) -> None:
        result = runner.invoke(
            app, ["workbench", "workflow", "filter", "--application", "uncategorized"]
        )
        assert result.exit_code == 0, result.output
        assert "mystery" in result.output
        assert "gpu-train" not in result.output

    def test_tier_flag_restricts_rows(self, synthetic_catalog: dict) -> None:
        listed = runner.invoke(
            app, ["workbench", "workflow", "filter", "--tier", "main", "--gpu", "any"]
        )
        assert listed.exit_code == 0, listed.output
        assert "gpu-train" in listed.output
        empty = runner.invoke(
            app, ["workbench", "workflow", "filter", "--tier", "testing"]
        )
        assert empty.exit_code == 0, empty.output
        assert "No workflows match" in empty.output

    def test_rows_are_sorted_by_name_then_application(
        self, synthetic_catalog: dict
    ) -> None:
        result = runner.invoke(
            app, ["workbench", "workflow", "filter", "--json", "--gpu", "any"]
        )
        assert result.exit_code == 0, result.output
        document = json.loads(result.output)
        names = [row["name"] for row in document["workflows"]]
        assert names == sorted(names)

    def test_valid_filters_with_no_matches_report_guidance(
        self, synthetic_catalog: dict
    ) -> None:
        result = runner.invoke(
            app,
            ["workbench", "workflow", "filter", "--gpu", "B200", "--no-pvc"],
        )
        assert result.exit_code == 0, result.output
        assert "gpu-train" in result.output
        narrow = runner.invoke(
            app,
            [
                "workbench",
                "workflow",
                "filter",
                "--application",
                "data-curation",
                "--gpu",
                "B200",
            ],
        )
        assert narrow.exit_code == 0, narrow.output
        assert "No workflows match the selected filters." in narrow.output
        assert "npa workbench workflow catalog" in narrow.output
        # Rich styling never leaks markup or ANSI escapes into captured
        # (non-terminal) output.
        assert "[red]" not in narrow.output
        assert "\x1b[" not in narrow.output


class TestFlagValidation:
    def test_unknown_application_is_rejected(self, synthetic_catalog: dict) -> None:
        result = runner.invoke(
            app, ["workbench", "workflow", "filter", "--application", "bogus"]
        )
        assert result.exit_code != 0
        assert "unknown application" in result.output

    def test_unknown_gpu_is_rejected(self, synthetic_catalog: dict) -> None:
        result = runner.invoke(
            app, ["workbench", "workflow", "filter", "--gpu", "bogus"]
        )
        assert result.exit_code != 0
        assert "unknown gpu" in result.output

    def test_unknown_tier_is_rejected(self, synthetic_catalog: dict) -> None:
        result = runner.invoke(
            app, ["workbench", "workflow", "filter", "--tier", "bogus"]
        )
        assert result.exit_code != 0
        assert "unknown tier" in result.output


class TestJsonOutput:
    def test_json_echoes_filters_and_rows(self, synthetic_catalog: dict) -> None:
        result = runner.invoke(
            app,
            [
                "workbench",
                "workflow",
                "filter",
                "--json",
                "--gpu",
                "none",
                "--no-pvc",
            ],
        )
        assert result.exit_code == 0, result.output
        document = json.loads(result.output)
        assert document["filters"] == {
            "application": "",
            "gpu": "none",
            "pvc": False,
            "tier": "",
        }
        names = [row["name"] for row in document["workflows"]]
        assert names == ["cpu-report", "mystery"]

    def test_json_rows_share_the_catalog_row_shape(
        self, synthetic_catalog: dict
    ) -> None:
        result = runner.invoke(
            app, ["workbench", "workflow", "filter", "--json", "--gpu", "any"]
        )
        assert result.exit_code == 0, result.output
        sample = json.loads(result.output)["workflows"][0]
        assert sample.keys() == {
            "tier",
            "partner",
            "file",
            "name",
            "application",
            "modality",
            "cpus",
            "memory",
            "gpu_cloud",
            "gpu_type",
            "phases",
            "inputs",
            "outputs",
            "pvc_cache",
            "complexity",
        }


class TestInteractiveFlow:
    def test_blank_answers_filter_only_pvc(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = {
            "main": [
                _entry(
                    "cached-train",
                    application="robotics-policy",
                    modality="GPU",
                    gpu_type="8x RTXPRO6000",
                    pvc_cache=True,
                ),
                _entry("gpu-train", modality="GPU", gpu_type="2x B200"),
            ]
        }
        monkeypatch.setattr(filter_cli, "index_catalog", lambda: rows)
        monkeypatch.setattr(filter_cli, "_is_interactive", lambda: True)
        result = runner.invoke(
            app, ["workbench", "workflow", "filter"], input="\n\nn\n"
        )
        assert result.exit_code == 0, result.output
        assert "gpu-train" in result.output
        assert "cached-train" not in result.output
        assert "No filters selected" not in result.output

    def test_prompts_ask_the_three_questions(
        self, interactive: None, synthetic_catalog: dict
    ) -> None:
        result = runner.invoke(
            app, ["workbench", "workflow", "filter"], input="\n2\ny\n"
        )
        assert result.exit_code == 0, result.output
        assert "Application?" in result.output
        assert "GPU?" in result.output
        assert "Have you created a PVC" in result.output
        # Prompts render color only on a real terminal; the captured
        # output stays plain text with no markup or escape codes.
        assert "\x1b[" not in result.output
        assert "[bold" not in result.output
        # GPU menu entry 2 is "None (CPU only)".
        assert "cpu-report" in result.output
        assert "mystery" in result.output
        assert "gpu-train" not in result.output

    def test_numbered_application_choice(
        self, interactive: None, synthetic_catalog: dict
    ) -> None:
        # Application menu sorts unique categories: entry 1 is
        # data-curation, entry 2 is robotics-policy.
        result = runner.invoke(
            app, ["workbench", "workflow", "filter"], input="1\n\ny\n"
        )
        assert result.exit_code == 0, result.output
        assert "cpu-report" in result.output
        assert "gpu-train" not in result.output

    def test_invalid_number_reprompts(
        self, interactive: None, synthetic_catalog: dict
    ) -> None:
        # Application prompt reads "9" (invalid), then blank; GPU prompt
        # reads blank; the PVC confirm reads "y".
        result = runner.invoke(
            app, ["workbench", "workflow", "filter"], input="9\n\n\ny\n"
        )
        assert result.exit_code == 0, result.output
        assert "Enter a number from 1 to" in result.output

    def test_blank_everywhere_shows_every_workflow(
        self, interactive: None, synthetic_catalog: dict
    ) -> None:
        result = runner.invoke(
            app, ["workbench", "workflow", "filter"], input="\n\ny\n"
        )
        assert result.exit_code == 0, result.output
        assert "No filters selected" in result.output
        for name in ("gpu-train", "cached-train", "cpu-report", "mystery"):
            assert name in result.output

    def test_flags_suppress_their_prompts(
        self, interactive: None, synthetic_catalog: dict
    ) -> None:
        result = runner.invoke(
            app,
            ["workbench", "workflow", "filter", "--gpu", "any", "--pvc"],
            input="\n",
        )
        assert result.exit_code == 0, result.output
        assert "Application?" in result.output
        assert "GPU?" not in result.output
        assert "Have you created a PVC" not in result.output


class TestNonInteractiveGuard:
    def test_no_filters_without_a_terminal_fails_with_hints(self) -> None:
        result = runner.invoke(app, ["workbench", "workflow", "filter"])
        assert result.exit_code == 1
        assert "no filters were given" in result.output
        assert "--application" in result.output
        assert "npa workbench workflow catalog" in result.output
        assert "Matching workflows" not in result.output

    def test_tier_flag_counts_as_a_filter(self, synthetic_catalog: dict) -> None:
        result = runner.invoke(
            app, ["workbench", "workflow", "filter", "--tier", "main", "--gpu", "any"]
        )
        assert result.exit_code == 0, result.output
        assert "Matching workflows" in result.output


class TestRealCatalog:
    """Anchor the filters against the checked-in catalog."""

    def test_robotics_policy_without_pvc(self) -> None:
        result = runner.invoke(
            app,
            [
                "workbench",
                "workflow",
                "filter",
                "--application",
                "robotics-policy",
                "--no-pvc",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "groot-1-7-finetune" in result.output
        assert "openpi-pi05-full-droid-finetune" not in result.output

    def test_b200_json_rows_name_b200(self) -> None:
        result = runner.invoke(
            app, ["workbench", "workflow", "filter", "--json", "--gpu", "B200"]
        )
        assert result.exit_code == 0, result.output
        document = json.loads(result.output)
        assert document["filters"]["gpu"] == "B200"
        assert document["workflows"]
        assert all("B200" in row["gpu_type"] for row in document["workflows"])
        names = [row["name"] for row in document["workflows"]]
        assert names == sorted(names)

    def test_cpu_only_json_rows(self) -> None:
        result = runner.invoke(
            app, ["workbench", "workflow", "filter", "--json", "--gpu", "none"]
        )
        assert result.exit_code == 0, result.output
        rows = json.loads(result.output)["workflows"]
        assert rows
        assert all(row["modality"] == "CPU" for row in rows)
