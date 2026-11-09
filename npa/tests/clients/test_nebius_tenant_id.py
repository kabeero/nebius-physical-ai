"""Tenant resource-ID validation and its protection of profile writes."""

from __future__ import annotations

import pytest

from npa.clients import nebius


@pytest.mark.parametrize(
    "tenant_id",
    [
        "tenant-0123456789abcdefgh",
        "aitenant-0123456789abcdefgh",
        "tractotenant-0123456789abcdefgh",
        "tenant-x",
    ],
)
def test_is_tenant_id_accepts_supported_resource_id_types(tenant_id: str) -> None:
    assert nebius.is_tenant_id(tenant_id)


@pytest.mark.parametrize(
    "value",
    [
        "my-tenant-name",
        "demo-tenant",
        "project-0123456789abcdefgh",
        "",
        "  ",
        "tenant-",
        "Tenant-Abc",
        "tenant--double",
        "tenant-trailing-",
    ],
)
def test_is_tenant_id_rejects_names_and_other_resources(value: str) -> None:
    assert not nebius.is_tenant_id(value)


def _profile_run_store(commands: list[list[str]]):
    """Fake ``nebius._run`` backed by an in-memory profile key/value store."""

    store: dict[str, str] = {}

    def run(command, *_args, **_kwargs):
        commands.append(list(command))
        if command[:2] == ["config", "get"]:
            return store.get(command[2], "")
        if command[:2] == ["config", "set"]:
            store[command[2]] = command[3]
        elif command[:2] == ["config", "unset"]:
            store.pop(command[2], None)
        return ""

    return run


def _written(commands: list[list[str]], key: str) -> list[list[str]]:
    return [command for command in commands if command[:3] == ["config", "set", key]]


def test_set_profile_project_skips_malformed_tenant_annotation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(nebius, "_run", _profile_run_store(commands))

    result = nebius.set_profile_project("project-new", "my-tenant-name")

    assert result is nebius.ProfileMutationResult.UPDATED
    assert _written(commands, "parent-id") == [
        ["config", "set", "parent-id", "project-new"]
    ]
    assert _written(commands, "tenant-id") == []


def test_set_profile_project_writes_validated_tenant_annotation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(nebius, "_run", _profile_run_store(commands))

    result = nebius.set_profile_project("project-new", "tenant-0123456789abcdefgh")

    assert result is nebius.ProfileMutationResult.UPDATED
    assert [
        "config",
        "set",
        "tenant-id",
        "tenant-0123456789abcdefgh",
    ] in _written(commands, "tenant-id")
