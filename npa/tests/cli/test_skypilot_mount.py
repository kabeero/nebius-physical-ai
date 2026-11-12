from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from npa.cli import skypilot as skypilot_cli
from npa.cli.main import app


runner = CliRunner()

_ALIAS = "sky-jobs-controller-53b2a1ad"
_HOSTNAME = "10.240.0.1"
_OLDER_ALIAS = "sky-jobs-controller-older"


def _flat(output: str) -> str:
    """Collapse rich console word-wrapping so substring checks hold."""

    return " ".join(output.split())


def _write_sky_ssh_config(
    sky_ssh_dir: Path,
    alias: str,
    *,
    hostname: str = _HOSTNAME,
    age_seconds: float = 0.0,
) -> Path:
    path = sky_ssh_dir / alias
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"# Added by sky (use `sky stop/down {alias}` to remove)\n"
        f"Host {alias}\n"
        f"  HostName {hostname}\n"
        "  User sky\n",
        encoding="utf-8",
    )
    stamp = time.time() - age_seconds
    os.utime(path, (stamp, stamp))
    return path


def _write_ssh_config(ssh_config: Path, *, include_sky: bool = True) -> Path:
    ssh_config.parent.mkdir(parents=True, exist_ok=True)
    body = "Host github.com\n  User git\n"
    if include_sky:
        body += "Include ~/.sky/generated/ssh/*\n"
    ssh_config.write_text(body, encoding="utf-8")
    return ssh_config


def _patch_mount_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    sshfs: str | None = "/usr/bin/sshfs",
    hosts: tuple[str, ...] = (_ALIAS,),
    include_sky: bool = True,
    mount_state: skypilot_cli.MountState | None = None,
) -> Path:
    """Point the mount commands at tmp_path fakes; returns the mount point."""

    sky_ssh_dir = tmp_path / "sky-ssh"
    for index, alias in enumerate(hosts):
        # Later entries are older so the first candidate is the most recent.
        _write_sky_ssh_config(sky_ssh_dir, alias, age_seconds=index * 3600)
    if include_sky:
        _write_ssh_config(tmp_path / "ssh-config")
    else:
        _write_ssh_config(tmp_path / "ssh-config", include_sky=False)
    mount_point = tmp_path / "sky"
    monkeypatch.setattr(skypilot_cli, "_resolve_sshfs_bin", lambda: sshfs)
    monkeypatch.setattr(skypilot_cli, "_sky_ssh_dir", lambda: sky_ssh_dir)
    monkeypatch.setattr(
        skypilot_cli, "_ssh_config_path", lambda: tmp_path / "ssh-config"
    )
    monkeypatch.setattr(skypilot_cli, "_default_mount_point", lambda: mount_point)
    if mount_state is not None:
        monkeypatch.setattr(
            skypilot_cli,
            "_mount_state",
            lambda _target: mount_state,
        )
    return mount_point


def test_mount_unmount_registered_under_skypilot_help() -> None:
    result = runner.invoke(app, ["skypilot", "--help"])

    assert result.exit_code == 0
    assert "mount" in _flat(result.output)
    assert "unmount" in _flat(result.output)


def test_mount_requires_sshfs_installed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_mount_env(monkeypatch, tmp_path, sshfs=None)

    result = runner.invoke(app, ["skypilot", "mount"])

    assert result.exit_code == 1
    assert "sshfs is not installed" in _flat(result.output)
    assert "sudo apt-get install sshfs" in _flat(result.output)


@pytest.mark.parametrize("ssh_dir_exists", [False, True])
def test_mount_before_submission_fails_with_premature_warning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, ssh_dir_exists: bool
) -> None:
    _patch_mount_env(monkeypatch, tmp_path)
    sky_ssh_dir = tmp_path / "sky-ssh"
    for entry in sky_ssh_dir.iterdir():
        entry.unlink()
    if not ssh_dir_exists:
        sky_ssh_dir.rmdir()

    result = runner.invoke(app, ["skypilot", "mount"])

    assert result.exit_code == 1
    assert f"No SkyPilot SSH configs under {sky_ssh_dir}" in _flat(result.output)
    assert "npa workbench workflow submit" in _flat(result.output)
    assert "npa workbench workflow status" in _flat(result.output)
    assert "--isolated-config-dir" in _flat(result.output)


def test_mount_requires_ssh_config_stanza(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_mount_env(monkeypatch, tmp_path, include_sky=False)

    result = runner.invoke(app, ["skypilot", "mount"])

    assert result.exit_code == 1
    assert "no SkyPilot stanza" in _flat(result.output)
    assert f"'{_ALIAS}'" in _flat(result.output)
    assert "Include ~/.sky/generated/ssh/*" in _flat(result.output)


def test_mount_multiple_hosts_requires_explicit_choice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_mount_env(monkeypatch, tmp_path, hosts=(_ALIAS, _OLDER_ALIAS))

    result = runner.invoke(app, ["skypilot", "mount"])

    assert result.exit_code == 1
    assert "Multiple SkyPilot SSH configs found" in _flat(result.output)
    assert "--host" in _flat(result.output)
    assert _ALIAS in result.output
    assert _OLDER_ALIAS in result.output
    assert "most recent" in _flat(result.output)


def test_mount_unknown_host_fails_fast(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_mount_env(monkeypatch, tmp_path, hosts=(_ALIAS, _OLDER_ALIAS))

    result = runner.invoke(app, ["skypilot", "mount", "--host", "sky-bogus"])

    assert result.exit_code == 1
    assert "unknown --host 'sky-bogus'" in _flat(result.output)
    assert _ALIAS in result.output
    assert _OLDER_ALIAS in result.output


def test_mount_already_mounted_exits_gracefully(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_mount_env(
        monkeypatch,
        tmp_path,
        hosts=(_ALIAS, _OLDER_ALIAS),
        mount_state=skypilot_cli.MountState("sshfs", _ALIAS),
    )

    result = runner.invoke(app, ["skypilot", "mount"])

    assert result.exit_code == 0
    assert "already mounted over sshfs" in _flat(result.output)
    assert f"mounted host: {_ALIAS}" in _flat(result.output)
    assert "(mounted" in _flat(result.output)
    assert "most recent" in _flat(result.output)


def test_mount_already_mounted_suggests_switch_for_other_host(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_mount_env(
        monkeypatch,
        tmp_path,
        hosts=(_ALIAS, _OLDER_ALIAS),
        mount_state=skypilot_cli.MountState("sshfs", _OLDER_ALIAS),
    )

    result = runner.invoke(app, ["skypilot", "mount", "--host", _ALIAS])

    assert result.exit_code == 0
    assert "already mounted over sshfs" in _flat(result.output)
    assert "to switch" in _flat(result.output)
    assert "npa skypilot unmount" in _flat(result.output)


def test_mount_already_mounted_fails_fast_for_unknown_host(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_mount_env(
        monkeypatch,
        tmp_path,
        hosts=(_ALIAS, _OLDER_ALIAS),
        mount_state=skypilot_cli.MountState("sshfs", _ALIAS),
    )

    result = runner.invoke(app, ["skypilot", "mount", "--host", "sky-bogus"])

    assert result.exit_code == 1
    assert "unknown --host 'sky-bogus'" in _flat(result.output)
    assert _ALIAS in _flat(result.output)
    assert _OLDER_ALIAS in _flat(result.output)


def test_mount_refuses_other_filesystem_mount_point(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_mount_env(
        monkeypatch,
        tmp_path,
        mount_state=skypilot_cli.MountState("other", None),
    )

    result = runner.invoke(app, ["skypilot", "mount"])

    assert result.exit_code == 1
    assert "mount point for another filesystem" in _flat(result.output)


def test_mount_refuses_non_empty_mount_point(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mount_point = _patch_mount_env(monkeypatch, tmp_path)
    mount_point.mkdir(parents=True)
    (mount_point / "leftover.txt").write_text("data", encoding="utf-8")

    result = runner.invoke(app, ["skypilot", "mount"])

    assert result.exit_code == 1
    assert "exists and is not empty" in _flat(result.output)
    assert "--mount-point" in _flat(result.output)


def test_mount_success_uses_host_alias_and_sky_logs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mount_point = _patch_mount_env(
        monkeypatch,
        tmp_path,
        mount_state=skypilot_cli.MountState("none", None),
    )
    calls: list[list[str]] = []
    monkeypatch.setattr(
        skypilot_cli,
        "_run_mount_tool",
        lambda argv, *, timeout: (
            calls.append(argv)
            or subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        ),
    )
    monkeypatch.setattr(skypilot_cli, "_is_mounted", lambda _target: True)

    result = runner.invoke(app, ["skypilot", "mount"])

    assert result.exit_code == 0
    assert calls == [
        [
            "/usr/bin/sshfs",
            f"{_ALIAS}:{skypilot_cli.SSHFS_REMOTE_DIR}",
            str(mount_point),
        ]
    ]
    assert f"mounted {_ALIAS}:sky_logs" in _flat(result.output)
    assert "npa skypilot unmount" in _flat(result.output)


def test_mount_honors_explicit_host_and_mount_point(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_mount_env(
        monkeypatch,
        tmp_path,
        hosts=(_ALIAS, _OLDER_ALIAS),
        mount_state=skypilot_cli.MountState("none", None),
    )
    custom = tmp_path / "logs"
    calls: list[list[str]] = []
    monkeypatch.setattr(
        skypilot_cli,
        "_run_mount_tool",
        lambda argv, *, timeout: (
            calls.append(argv)
            or subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        ),
    )
    monkeypatch.setattr(skypilot_cli, "_is_mounted", lambda _target: True)

    result = runner.invoke(
        app,
        ["skypilot", "mount", "--host", _OLDER_ALIAS, "--mount-point", str(custom)],
    )

    assert result.exit_code == 0
    assert calls == [["/usr/bin/sshfs", f"{_OLDER_ALIAS}:sky_logs", str(custom)]]
    assert f"mounted {_OLDER_ALIAS}:sky_logs at {custom}" in _flat(result.output)


def test_mount_sshfs_failure_reports_cluster_guidance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_mount_env(monkeypatch, tmp_path)
    monkeypatch.setattr(
        skypilot_cli,
        "_run_mount_tool",
        lambda argv, *, timeout: subprocess.CompletedProcess(
            argv, 1, stdout="", stderr="fuse: ssh connection refused"
        ),
    )
    monkeypatch.setattr(skypilot_cli, "_is_mounted", lambda _target: False)

    result = runner.invoke(app, ["skypilot", "mount"])

    assert result.exit_code == 1
    assert "sshfs failed to mount" in _flat(result.output)
    assert "ssh connection refused" in _flat(result.output)
    assert "npa workbench workflow status" in _flat(result.output)


def test_unmount_not_mounted_exits_gracefully(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mount_point = _patch_mount_env(
        monkeypatch, tmp_path, mount_state=skypilot_cli.MountState("none", None)
    )

    result = runner.invoke(app, ["skypilot", "unmount"])

    assert result.exit_code == 0
    assert f"{mount_point} is not mounted" in _flat(result.output)


def test_unmount_refuses_other_filesystem_mount_point(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_mount_env(
        monkeypatch,
        tmp_path,
        mount_state=skypilot_cli.MountState("other", None),
    )

    result = runner.invoke(app, ["skypilot", "unmount"])

    assert result.exit_code == 1
    assert "another filesystem" in _flat(result.output)


def test_unmount_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mount_point = _patch_mount_env(
        monkeypatch,
        tmp_path,
        mount_state=skypilot_cli.MountState("sshfs", _ALIAS),
    )
    calls: list[list[str]] = []
    monkeypatch.setattr(
        skypilot_cli,
        "_resolve_unmount_argv",
        lambda target: ["/usr/bin/fusermount3", "-u", str(target)],
    )
    monkeypatch.setattr(
        skypilot_cli,
        "_run_mount_tool",
        lambda argv, *, timeout: (
            calls.append(argv)
            or subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        ),
    )
    monkeypatch.setattr(skypilot_cli, "_is_mounted", lambda _target: False)

    result = runner.invoke(app, ["skypilot", "unmount"])

    assert result.exit_code == 0
    assert calls == [["/usr/bin/fusermount3", "-u", str(mount_point)]]
    assert f"unmounted {mount_point}" in _flat(result.output)
    assert f"host {_ALIAS}" in _flat(result.output)


def test_unmount_busy_mount_suggests_lazy_unmount(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mount_point = _patch_mount_env(
        monkeypatch,
        tmp_path,
        mount_state=skypilot_cli.MountState("sshfs", _ALIAS),
    )
    monkeypatch.setattr(
        skypilot_cli,
        "_resolve_unmount_argv",
        lambda target: ["/usr/bin/fusermount3", "-u", str(target)],
    )
    monkeypatch.setattr(
        skypilot_cli,
        "_run_mount_tool",
        lambda argv, *, timeout: subprocess.CompletedProcess(
            argv, 1, stdout="", stderr="fuse: device is busy"
        ),
    )
    monkeypatch.setattr(skypilot_cli, "_is_mounted", lambda _target: True)

    result = runner.invoke(app, ["skypilot", "unmount"])

    assert result.exit_code == 1
    assert "could not unmount" in _flat(result.output)
    assert "device is busy" in _flat(result.output)
    assert f"close programs using {mount_point}" in _flat(result.output)


def test_parse_sky_ssh_hosts_skips_files_without_directives(
    tmp_path: Path,
) -> None:
    _write_sky_ssh_config(tmp_path, _ALIAS)
    (tmp_path / "unrelated").write_text("garbage\n", encoding="utf-8")

    hosts = skypilot_cli._parse_sky_ssh_hosts(tmp_path)

    assert [host.alias for host in hosts] == [_ALIAS]
    assert hosts[0].hostname == _HOSTNAME


def test_parse_sky_ssh_hosts_missing_dir_returns_empty(tmp_path: Path) -> None:
    assert skypilot_cli._parse_sky_ssh_hosts(tmp_path / "missing") == []


def test_ssh_config_references_host_variants(tmp_path: Path) -> None:
    included = tmp_path / "included"
    included.write_text("Include ~/.sky/generated/ssh/*\n", encoding="utf-8")
    stanza = tmp_path / "stanza"
    stanza.write_text(f"Host {_ALIAS}\n  User sky\n", encoding="utf-8")
    neither = tmp_path / "neither"
    neither.write_text("Host github.com\n  User git\n", encoding="utf-8")

    assert skypilot_cli._ssh_config_references_host(included, _ALIAS)
    assert skypilot_cli._ssh_config_references_host(stanza, _ALIAS)
    assert not skypilot_cli._ssh_config_references_host(neither, _ALIAS)


def test_mount_state_parses_proc_mounts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mount_point = tmp_path / "sky"
    proc_mounts = tmp_path / "proc-mounts"
    proc_mounts.write_text(
        f"{_ALIAS}:sky_logs {mount_point} fuse.sshfs rw,nosuid 0 0\n"
        "/dev/sda1 / ext4 rw 0 0\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(skypilot_cli, "_PROC_MOUNTS", str(proc_mounts))
    monkeypatch.setattr(skypilot_cli, "_is_mounted", lambda _target: True)

    state = skypilot_cli._mount_state(mount_point)

    assert state.status == "sshfs"
    assert state.host == _ALIAS


def test_mount_state_other_filesystem(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mount_point = tmp_path / "sky"
    proc_mounts = tmp_path / "proc-mounts"
    proc_mounts.write_text(f"/dev/sdb1 {mount_point} ext4 rw 0 0\n", encoding="utf-8")
    monkeypatch.setattr(skypilot_cli, "_PROC_MOUNTS", str(proc_mounts))
    monkeypatch.setattr(skypilot_cli, "_is_mounted", lambda _target: True)

    state = skypilot_cli._mount_state(mount_point)

    assert state.status == "other"
    assert state.host is None


def test_mount_state_unmounted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(skypilot_cli, "_is_mounted", lambda _target: False)

    state = skypilot_cli._mount_state(tmp_path / "sky")

    assert state.status == "none"
