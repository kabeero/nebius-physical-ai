"""Index the discovered npa.workflow catalog into operator-facing summary rows.

The catalog command reports the declared contract of every spec found by the
shared blueprint discovery: resource shape, phase count after fan-out
collapse, artifact flags, and a heuristic operating-complexity score. It
describes what a spec declares, never a guarantee that any configuration has
run successfully.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from npa.orchestration.npa_workflow.blueprints import (
    is_npa_workflow_document,
    npa_workflow_spec_dirs,
)
from npa.orchestration.npa_workflow.errors import NpaWorkflowError
from npa.orchestration.npa_workflow.spec import (
    NpaWorkflowSpec,
    StateSpec,
    load_spec_document,
)

#: Specs that discovery finds but the operator-facing catalog hides.
#: ``sim2real-two-step-agent.yaml`` is a pinned golden snapshot of the agent's
#: generated two-step draft. It duplicates ``sim2real-two-step.yaml`` in
#: ``metadata.name`` and default prefix, and exists only for the
#: generator-equality test in ``npa/tests/cli/test_agent_workflow.py``.
CATALOG_DISPLAY_EXCLUSIONS = frozenset({"sim2real-two-step-agent.yaml"})

#: Ordered keyword rules mapping a spec's file name, workflow name, and
#: description onto a semantic application category. The first matching rule
#: wins; specs matching no rule index with a blank application. Keep this
#: table ordered from most specific domain to most generic, and keep keywords
#: long enough to survive the leading word-boundary match below.
APPLICATION_LUT: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("template", ("demo only", "generic byof")),
    ("sim2real", ("sim2real workflow", "sim2real training", "sim-to-real workflow", "sim to real workflow")),
    # Narrow early rule: this spec name collides with the cosmos3 generation
    # family below but is a policy-learning round, not media generation.
    ("robotics-policy", ("policy-model-factory",)),
    ("autonomous-vehicles", ("bdd100k", "av night", "night-scene", "dashcam", "driving")),
    ("vla-inference", ("alpamayo", "vla")),
    (
        "infrastructure",
        (
            "gang-schedule",
            "routing-evidence",
            "routing evidence",
            "source and checkpoint access",
        ),
    ),
    (
        "perception",
        (
            "depth-anything",
            "sam2",
            "segmentation",
            "mask propagation",
            "depth estimation",
        ),
    ),
    ("world-model", ("dreamer", "world-model", "world model")),
    (
        "neural-reconstruction",
        (
            "nurec",
            "colmap",
            "3dgut",
            "gaussian-splatting",
            "gsplat",
            "lyra",
            "open3d",
            "rgbd-scan",
            "scan-to-isaac",
            "reconstruction",
        ),
    ),
    (
        "digital-twin",
        ("digital-twin", "digital twin", "living-lab", "campus", "marble"),
    ),
    ("restoration", ("restoration", "seedvr")),
    ("serving-benchmark", ("serving record", "serving path", "serving sweep", "ray serve", "ray-batch")),
    (
        "synthetic-data",
        (
            "paidf",
            "sdg batch",
            "robot-sdg",
            "token-factory-sdg",
            "augment",
            "anomaly",
            "defect",
            "variant sweep",
            "embodiedgen",
            "prompt-comparison",
            "cosmos transfer",
            "cosmos2",
            "data factory",
            "scenario-gen",
            "rgbd",
            "rigid-object",
            "content-agents",
        ),
    ),
    (
        "generation",
        (
            "text2image",
            "text-to-image",
            "wan2",
            "ltx2",
            "ltx-2",
            "mochi",
            "cogvideo",
            "lingbot",
            "cosmos3",
            "cosmos 3",
            "video generation",
        ),
    ),
    ("vlm-evaluation", ("vlm", "judge", "critique", "triage")),
    ("hosted-inference", ("token-factory", "token factory", "tokenfactory")),
    (
        "robotics-locomotion",
        ("locomotion", "quadruped", "navigation", "rover", "go1", "sonic"),
    ),
    (
        "robotics-physics",
        ("newton", "mujoco", "habitat", "odometry", "slam", "physics"),
    ),
    (
        "robotics-policy",
        (
            "lerobot",
            "policy",
            "finetune",
            "fine-tune",
            "rsl-rl",
            "ppo",
            "openpi",
            "openvla",
            "groot",
            "molmoact",
            "flex-pi",
            "behavior",
            "droid",
            "smolvla",
            "mjlab",
        ),
    ),
    (
        "data-curation",
        (
            "dataset",
            "encord",
            "insights",
            "curate",
            "lancedb",
            "fiftyone",
            "labeling",
        ),
    ),
    (
        "robotics-manipulation",
        (
            "manipulation",
            "pick",
            "grasp",
            "aloha",
            "franka",
            "curobo",
            "openarm",
            "robocasa",
            "robomimic",
            "libero",
            "maniskill",
            "apriltag",
            "retarget",
            "robotwin",
            "gymnasium",
        ),
    ),
)

#: Display glyph for true flags; false flags stay blank.
CHECK_MARK = "\u2713"

#: Display glyph for each complexity point.
STAR = "\u2605"

_TEMPLATE = re.compile(r"\{\{\s*config\.([A-Za-z0-9_]+)\s*\}\}")
_LEADING_INT = re.compile(r"\s*(\d+)")
_ACCELERATOR_PAIR = re.compile(r"\s*([^:,:\s]+)\s*:\s*([+-]?\d+)\s*$")

#: libyaml parses several times faster than PyYAML's pure-Python loader and
#: produces the same document for these flat workflow specs; fall back to the
#: portable loader when the binding is missing.
_YAML_LOADER = (
    yaml.CSafeLoader if getattr(yaml, "__with_libyaml__", False) else yaml.SafeLoader
)


@dataclass(frozen=True)
class CatalogEntry:
    """One indexed workflow spec row.

    ``partner`` names the owning partner directory for partners-tier
    specs and is blank elsewhere. Resource fields report the spec's
    declared values, with simple ``{{config.<key>}}`` templates resolved
    against the spec's own config defaults; unresolved templates
    display their raw text.
    """

    tier: str
    partner: str
    rel_path: str
    name: str
    application: str
    modality: str
    cpus: str
    memory: str
    gpu_cloud: str
    gpu_type: str
    phases: int
    inputs: bool
    outputs: bool
    pvc_cache: bool
    complexity: int

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-ready mapping for this row.

        Args: None.
        Returns: Every field with booleans and the integer complexity score.
        Raises: None.
        """
        return {
            "tier": self.tier,
            "partner": self.partner,
            "file": self.rel_path,
            "name": self.name,
            "application": self.application,
            "modality": self.modality,
            "cpus": self.cpus,
            "memory": self.memory,
            "gpu_cloud": self.gpu_cloud,
            "gpu_type": self.gpu_type,
            "phases": self.phases,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "pvc_cache": self.pvc_cache,
            "complexity": self.complexity,
        }


def index_catalog() -> dict[str, list[CatalogEntry]]:
    """Index every discovered spec, binned by catalog tier.

    Mirrors ``iter_npa_workflow_specs`` discovery (existing directories only,
    de-duplicated by file name) minus the display exclusions, keeping each
    spec's owning tier directory. Each file is parsed exactly once.

    Args: None.
    Returns: Tier name to entries sorted by workflow name. Partner
      directories merge under ``partners``.
    Raises: NpaWorkflowError: A discovered spec failed validation.
    """
    entries: dict[str, list[CatalogEntry]] = {}
    seen: set[str] = set()
    for directory in npa_workflow_spec_dirs():
        for path in sorted(directory.glob("*.yaml")):
            if path.name in seen or path.name in CATALOG_DISPLAY_EXCLUSIONS:
                continue
            try:
                document = _read_document(path)
            except (OSError, yaml.YAMLError):
                continue
            if not is_npa_workflow_document(document):
                continue
            seen.add(path.name)
            entry = _catalog_entry(path, load_spec_document(document or {}))
            entries.setdefault(_tier_name(path.parent), []).append(entry)
    for rows in entries.values():
        rows.sort(key=lambda row: row.name)
    return entries


def index_workflow(path: Path) -> CatalogEntry:
    """Derive the catalog columns for one workflow spec file.

    Args:
        path: Spec file to index; the parent directory decides its tier.
    Returns: The indexed row.
    Raises:
        NpaWorkflowError: The file fails workflow spec validation.
    """
    try:
        document = _read_document(path) or {}
    except yaml.YAMLError as exc:
        raise NpaWorkflowError(f"workflow spec is not valid YAML: {exc}") from exc
    return _catalog_entry(path, load_spec_document(document))


def _read_document(path: Path) -> Any:
    """Parse one YAML file with the fastest available safe loader."""

    return yaml.load(path.read_text(encoding="utf-8"), Loader=_YAML_LOADER)


def _catalog_entry(path: Path, spec: NpaWorkflowSpec) -> CatalogEntry:
    """Assemble one catalog row from a loaded spec.

    Args:
        path: The spec's file, used for tier and display identity.
        spec: The validated workflow spec.
    Returns: The indexed row.
    Raises: None.
    """
    leaf_profiles = _leaf_profiles(spec)
    gpu_profiles = [profile for profile in leaf_profiles if profile.get("accelerators")]
    pvc_cache = _uses_pvc(leaf_profiles)
    phases = _phase_count(spec)
    max_gpus = _max_gpu_count(gpu_profiles, spec.config)
    return CatalogEntry(
        tier=_tier_name(path.parent),
        partner=_partner_name(path.parent),
        rel_path=_rel_path(path),
        name=_sanitize_name(spec.name),
        application=_application(path.stem, spec),
        modality=_modality(leaf_profiles),
        cpus=_max_declared(leaf_profiles, "cpus", spec.config),
        memory=_max_declared(leaf_profiles, "memory", spec.config),
        gpu_cloud=_gpu_cloud(gpu_profiles, spec.config),
        gpu_type=_gpu_type(gpu_profiles, spec.config),
        phases=phases,
        inputs=any(state.inputs for state in spec.states.values()),
        outputs=any(state.outputs for state in spec.states.values()),
        pvc_cache=pvc_cache,
        complexity=_complexity(
            spec, leaf_profiles, phases, max_gpus, pvc_cache
        ),
    )


def _tier_name(directory: Path) -> str:
    """Collapse partner directories to one tier label."""

    return "partners" if directory.parent.name == "partners" else directory.name


def _partner_name(directory: Path) -> str:
    """Owning partner directory for a partners-tier spec, else blank.

    A spec sitting directly in the ``partners/`` root belongs to no
    partner directory and stays blank; the partner name is the spec's
    owning folder (for example ``antioch``), never the spec itself.
    """

    return directory.name if directory.parent.name == "partners" else ""


def _rel_path(path: Path) -> str:
    """Path relative to the catalog root (for example ``testing/foo.yaml``)."""

    try:
        return path.relative_to(path.parent.parent).as_posix()
    except ValueError:
        return path.as_posix()


def _sanitize_name(name: str) -> str:
    """Collapse whitespace inside a declared workflow name to hyphens."""

    return re.sub(r"\s+", "-", name.strip())


def _application(stem: str, spec: NpaWorkflowSpec) -> str:
    """Return the first matching semantic category, or blank.

    Keywords match at a leading word boundary so short keywords like ``ppo``
    stay distinct from ordinary words like ``supports``.

    Args:
        stem: Spec file name without extension.
        spec: Loaded workflow spec.
    Returns: The application category, or ``""`` when no rule matches.
    Raises: None.
    """
    haystack = " ".join(
        (stem, spec.name, str(spec.metadata.get("description", "")))
    ).lower()
    for category, keywords in APPLICATION_LUT:
        for keyword in keywords:
            pattern = rf"(?<![a-z0-9_]){re.escape(keyword)}"
            if re.search(pattern, haystack):
                return category
    return ""


def _is_leaf(state: StateSpec) -> bool:
    """Whether a state executes work itself rather than grouping others."""

    if state.tool_ref:
        return True
    return state.run is not None and not state.run.is_empty()


def _leaf_states(spec: NpaWorkflowSpec) -> list[StateSpec]:
    """States that execute work, in declaration order."""

    return [state for state in spec.states.values() if _is_leaf(state)]


def _leaf_profiles(spec: NpaWorkflowSpec) -> list[Mapping[str, Any]]:
    """Resource profiles referenced by executing states, in phase order."""

    profiles: list[Mapping[str, Any]] = []
    for state in _leaf_states(spec):
        profiles.append(spec.resources.get(state.resources) or {})
    return profiles


def _phase_count(spec: NpaWorkflowSpec) -> int:
    """Count executing states with each ``parallel`` fan-out as one phase.

    Loops never multiply the count. ``sequence`` children count individually.

    Args:
        spec: Loaded workflow spec.
    Returns: The number of distinct phases.
    Raises: None.
    """
    fanout_members: set[str] = set()
    fanout_groups = 0
    for state in spec.states.values():
        if state.parallel:
            fanout_groups += 1
            fanout_members.update(state.parallel)
    executing = [
        state for state in _leaf_states(spec) if state.name not in fanout_members
    ]
    return len(executing) + fanout_groups


def _modality(leaf_profiles: list[Mapping[str, Any]]) -> str:
    """CPU, GPU, or CPU+GPU across the referenced resource profiles."""

    uses_gpu = any(profile.get("accelerators") for profile in leaf_profiles)
    uses_cpu = any(not profile.get("accelerators") for profile in leaf_profiles)
    if uses_gpu and uses_cpu:
        return "CPU+GPU"
    return "GPU" if uses_gpu else "CPU"


def _resolve(value: Any, config: Mapping[str, Any]) -> Any:
    """Resolve a full ``{{config.<key>}}`` template against spec defaults.

    Args:
        value: Declared resource value.
        config: The spec's config defaults.
    Returns: The resolved literal, or ``None`` when the value is not a
      single template or its key is missing or itself templated.
    Raises: None.
    """
    if not isinstance(value, str):
        return None
    match = _TEMPLATE.fullmatch(value.strip())
    if not match:
        return None
    resolved = config.get(match.group(1))
    if isinstance(resolved, str) and "{{" in resolved:
        return None
    return resolved


def _substitute(value: Any, config: Mapping[str, Any]) -> Any:
    """Replace every resolvable ``{{config.<key>}}`` in a declared value.

    Unlike :func:`_resolve`, partial templates such as
    ``{{config.gpu_type}}:{{config.gpu_count}}`` resolve when every
    referenced key has a literal default. Values with any unresolvable
    key stay exactly as declared.

    Args:
        value: Declared resource value.
        config: The spec's config defaults.
    Returns: The substituted value, or the original when a key is missing.
    Raises: None.
    """
    if not isinstance(value, str) or "{{" not in value:
        return value

    def _replace(match: re.Match[str]) -> str:
        resolved = config.get(match.group(1))
        if resolved is None or "{{" in str(resolved):
            return match.group(0)
        return str(resolved)

    substituted = _TEMPLATE.sub(_replace, value)
    return substituted


def _display(value: Any, config: Mapping[str, Any]) -> Any:
    """Declared value with resolvable templates replaced by their defaults."""

    resolved = _resolve(value, config)
    if resolved is not None:
        return resolved
    substituted = _substitute(value, config)
    return substituted


def _leading_int(value: Any) -> int | None:
    """Leading integer of a declared value such as ``16+`` or ``80Gi``."""

    match = _LEADING_INT.match(str(value))
    return int(match.group(1)) if match else None


def _max_declared(
    profiles: list[Mapping[str, Any]], key: str, config: Mapping[str, Any]
) -> str:
    """Display the numerically largest declared value for ``key``.

    Args:
        profiles: Referenced resource profiles.
        key: Resource field to compare (``cpus`` or ``memory``).
        config: The spec's config defaults for template resolution.
    Returns: The displayed value, or ``""`` when no profile declares it.
    Raises: None.
    """
    best_display = ""
    best_number: int | None = None
    for profile in profiles:
        if key not in profile:
            continue
        displayed = str(_display(profile[key], config))
        number = _leading_int(displayed)
        if best_number is None or (number is not None and number > best_number):
            best_display, best_number = displayed, number
    return best_display


def _gpu_cloud(
    gpu_profiles: list[Mapping[str, Any]], config: Mapping[str, Any]
) -> str:
    """Distinct ``cloud`` values across GPU profiles, comma-joined."""

    clouds: list[str] = []
    for profile in gpu_profiles:
        cloud = str(_display(profile.get("cloud", ""), config))
        if cloud and cloud not in clouds:
            clouds.append(cloud)
    return ", ".join(clouds)


def _accelerator_pairs(
    profile: Mapping[str, Any], config: Mapping[str, Any]
) -> list[tuple[str, int | None]]:
    """Normalized ``(gpu name, count)`` pairs for one profile.

    Counts stay ``None`` when the declaration is a template or otherwise not
    a plain ``name:count`` pair; those display as the raw declaration.

    Args:
        profile: One GPU resource profile.
        config: The spec's config defaults for template resolution.
    Returns: One pair per declared accelerator type.
    Raises: None.
    """
    raw = profile.get("accelerators")
    if isinstance(raw, Mapping):
        return [
            (str(name), _leading_int(count)) for name, count in raw.items()
        ]
    displayed = str(_display(raw, config))
    match = _ACCELERATOR_PAIR.match(displayed)
    if match:
        return [(match.group(1), int(match.group(2)))]
    return [(displayed, None)]


def _gpu_type(
    gpu_profiles: list[Mapping[str, Any]], config: Mapping[str, Any]
) -> str:
    """Accelerator summary, ``8x B200`` style, across GPU profiles."""

    counts: dict[str, int | None] = {}
    for profile in gpu_profiles:
        for name, count in _accelerator_pairs(profile, config):
            known = counts.get(name)
            if known is None or (count is not None and count > known):
                counts[name] = count
    parts = []
    for name, count in sorted(counts.items()):
        parts.append(f"{count}x {name}" if count and count > 1 else name)
    return ", ".join(parts)


def _max_gpu_count(
    gpu_profiles: list[Mapping[str, Any]], config: Mapping[str, Any]
) -> int:
    """Largest single-profile accelerator count (0 when none is parseable)."""

    best = 0
    for profile in gpu_profiles:
        for _name, count in _accelerator_pairs(profile, config):
            best = max(best, count or 0)
    return best


def _uses_pvc(profiles: list[Mapping[str, Any]]) -> bool:
    """Whether any referenced profile mounts a persistent volume claim."""

    for profile in profiles:
        pod_spec = ((profile.get("kubernetes") or {}).get("pod_config") or {}).get(
            "spec"
        )
        volumes = (pod_spec or {}).get("volumes") or []
        for volume in volumes:
            if isinstance(volume, Mapping) and "persistentVolumeClaim" in volume:
                return True
    return False


def _loop_points(spec: NpaWorkflowSpec) -> int:
    """One point for any loop, a second for nested loops."""

    looped = [state for state in spec.states.values() if state.loop is not None]
    if not looped:
        return 0
    nested = any(
        member in spec.states and spec.states[member].loop is not None
        for state in looped
        for member in state.sequence
    )
    return 2 if nested else 1


def _max_fanout(spec: NpaWorkflowSpec) -> int:
    """Widest declared ``parallel`` group."""

    return max(
        (len(state.parallel) for state in spec.states.values() if state.parallel),
        default=0,
    )


def _max_nodes(
    leaf_profiles: list[Mapping[str, Any]], config: Mapping[str, Any]
) -> int:
    """Largest declared ``num_nodes`` across referenced profiles."""

    best = 0
    for profile in leaf_profiles:
        if "num_nodes" not in profile:
            continue
        nodes = _leading_int(_display(profile["num_nodes"], config))
        if nodes:
            best = max(best, nodes)
    return best


def _complexity(
    spec: NpaWorkflowSpec,
    leaf_profiles: list[Mapping[str, Any]],
    phases: int,
    max_gpus: int,
    pvc_cache: bool,
) -> int:
    """Heuristic operating-complexity score from one to five points.

    Heavier graphs, gang-scheduled multi-node stages, wide fan-outs, loops,
    persistent caches, and multi-GPU requests each add points.

    Args:
        spec: Loaded workflow spec.
        leaf_profiles: Resource profiles referenced by executing states.
        phases: Already-collapsed phase count.
        max_gpus: Largest single-profile accelerator count.
        pvc_cache: Whether a referenced profile mounts a PVC.
    Returns: The score clamped to ``[1, 5]``.
    Raises: None.
    """
    points = 0
    if phases >= 3:
        points += 1
    if phases >= 10:
        points += 1
    if max_gpus >= 2:
        points += 1
    if max_gpus >= 8:
        points += 1
    points += _loop_points(spec)
    if _max_fanout(spec) >= 4:
        points += 1
    if pvc_cache:
        points += 1
    if _max_nodes(leaf_profiles, spec.config) >= 2:
        points += 1
    return min(5, 1 + points)
