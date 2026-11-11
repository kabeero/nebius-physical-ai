"""Unit tests for shared project/workbench config resolution guards."""

from __future__ import annotations

import pytest
import yaml


def test_ambiguous_project_without_default_errors_with_aliases():
    from npa.clients.config import ConfigError, _resolve_project_section

    yml = {"projects": {"proj-a": {"workbenches": {}}, "proj-b": {"workbenches": {}}}}
    with pytest.raises(ConfigError, match="-p/--project"):
        _resolve_project_section(yml, None)


def test_single_project_without_default_resolves_unambiguously():
    from npa.clients.config import _resolve_project_section

    yml = {"projects": {"only": {"workbenches": {"wb": {"x": 1}}}}}
    assert _resolve_project_section(yml, None) == {"workbenches": {"wb": {"x": 1}}}


def test_ambiguous_workbench_without_default_errors_with_aliases():
    from npa.clients.config import ConfigError, _resolve_workbench_in_project

    proj = {"workbenches": {"wb-a": {"x": 1}, "wb-b": {"x": 2}}}
    with pytest.raises(ConfigError, match="-n/--name"):
        _resolve_workbench_in_project(proj, None, {})


def test_single_workbench_without_default_resolves_unambiguously():
    from npa.clients.config import _resolve_workbench_in_project

    proj = {"workbenches": {"only": {"x": 1}}}
    assert _resolve_workbench_in_project(proj, None, {}) == {"x": 1}


def _write_config(projects: dict, default: str | None = None) -> None:
    from npa.clients import config

    document: dict = {"projects": projects}
    if default is not None:
        document["default_project"] = default
    config.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.CONFIG_PATH.write_text(yaml.safe_dump(document), encoding="utf-8")


def test_resolve_project_environment_names_the_alias_behind_a_pasted_project_id():
    """The ID-for-alias mistake must name the alias it belongs to."""

    from npa.clients.config import ConfigError, resolve_project_environment

    _write_config(
        {
            "hiring-demo": {
                "project_id": "project-u00demo00000test0xg",
                "tenant_id": "tenant-e00demo00000test0xg",
                "region": "us-central1",
            }
        }
    )

    with pytest.raises(ConfigError) as excinfo:
        resolve_project_environment("project-u00demo00000test0xg")

    message = str(excinfo.value)
    assert "project ID of configured alias 'hiring-demo'" in message
    assert "Pass --project hiring-demo" in message
    # A sole configured project is auto-selected, so omission is a remedy too.
    assert "selected automatically when --project is omitted" in message


def test_resolve_project_environment_lists_available_aliases_for_a_typo():
    from npa.clients.config import ConfigError, resolve_project_environment

    _write_config(
        {
            "alpha": {"project_id": "project-a", "region": "eu-north1"},
            "beta": {"project_id": "project-b", "region": "eu-north1"},
        }
    )

    with pytest.raises(ConfigError, match="Available: alpha, beta"):
        resolve_project_environment("alfa")


def test_resolve_project_environment_without_any_configured_projects():
    from npa.clients.config import ConfigError, resolve_project_environment

    _write_config({})

    with pytest.raises(ConfigError, match="no projects are configured"):
        resolve_project_environment("anything")

    assert resolve_project_environment(None) is None


def test_resolve_project_environment_empty_section_does_not_block_tfvars_identity():
    """A configured-but-empty alias contributes nothing but stays usable."""

    from npa.clients.config import resolve_project_environment

    _write_config({"shell": {}})

    assert resolve_project_environment("shell") is None


def test_resolve_project_environment_resolves_and_propagates_ambiguity():
    from npa.clients.config import ConfigError, resolve_project_environment

    _write_config(
        {
            "alpha": {"project_id": "project-a", "region": "eu-north1"},
            "beta": {"project_id": "project-b", "region": "eu-north1"},
        }
    )

    saved = resolve_project_environment("alpha")
    assert saved is not None
    assert (saved.project_id, saved.region) == ("project-a", "eu-north1")

    # Unlike resolve_environment, an ambiguous no-alias lookup must raise
    # instead of quietly resolving to nothing.
    with pytest.raises(ConfigError, match="No project selected"):
        resolve_project_environment(None)
