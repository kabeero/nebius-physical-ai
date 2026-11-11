"""Verify-payload filtering of upstream sky-check traceback noise."""

from __future__ import annotations

import subprocess

from npa.cli import skypilot as skypilot_cli


DO_TRACEBACK = "\n".join(
    [
        "  AWS: disabled",
        "    Reason [compute, storage]: AWS credentials are not set. Run:",
        "  DO: disabled",
        "    Reason [compute]: Traceback (most recent call last):",
        '  File "/home/example/.npa/skypilot-venv/lib/python3.12/site-packages/sky/check.py", line 141, in check_one_cloud_one_capability',
        "    ok, reason = cloud.check_credentials(capability)",
        "                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^",
        '  File "/home/example/.npa/skypilot-venv/lib/python3.12/site-packages/sky/clouds/do.py", line 304, in _check_compute_credentials',
        "    installed, err_msg = do.check_exceptions_dependencies_installed()",
        "                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^",
        '  File "/home/example/.npa/skypilot-venv/lib/python3.12/site-packages/sky/adaptors/do.py", line 24, in check_exceptions_dependencies_installed',
        "    if not common.can_import_modules(['azure.core.exceptions']):",
        "           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^",
        '  File "/home/example/.npa/skypilot-venv/lib/python3.12/site-packages/sky/adaptors/common.py", line 98, in can_import_modules',
        "    module_spec = importlib_util.find_spec(module_name)",
        "                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^",
        '  File "<frozen importlib.util>", line 91, in find_spec',
        "ModuleNotFoundError: No module named 'azure'",
        "  Fluidstack: disabled",
        "    Reason [compute]: Failed to access FluidStack Cloud with credentials.",
        "  Kubernetes: enabled",
        "🎉 Enabled infra 🎉",
    ]
)


def _check_result(output: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["sky", "check"], returncode=0, stdout=output, stderr=""
    )


def test_verify_payload_collapses_upstream_adaptor_probe_traceback() -> None:
    payload = skypilot_cli._verification_payload(
        _check_result(DO_TRACEBACK), backend="kubernetes", kubernetes_required=True
    )

    assert payload["status"] == "ok"
    assert payload["kubernetes_enabled"] is True
    joined = "\n".join(payload["sky_check_output"])
    assert "Traceback (most recent call last):" not in joined
    assert "sky/adaptors/do.py" not in joined
    assert "ModuleNotFoundError: No module named 'azure'" not in joined
    assert "  DO: disabled" in payload["sky_check_output"]
    assert (
        "missing optional skypilot dependency 'azure' (traceback filtered)"
        in joined
    )
    # The Kubernetes success line and the Fluidstack reason stay verbatim.
    assert "  Kubernetes: enabled" in payload["sky_check_output"]
    assert "Failed to access FluidStack Cloud" in joined


def test_verify_payload_keeps_tracebacks_outside_sky_adaptors() -> None:
    kubernetes_failure = "\n".join(
        [
            "  DO: disabled",
            "    Reason [compute]: Traceback (most recent call last):",
            '  File "/home/example/.npa/skypilot-venv/lib/python3.12/site-packages/sky/clouds/kubernetes.py", line 10, in check_credentials',
            "    ok, reason = cloud.check_credentials(capability)",
            "ModuleNotFoundError: No module named 'kubernetes'",
        ]
    )

    payload = skypilot_cli._verification_payload(
        _check_result(kubernetes_failure),
        backend="kubernetes",
        kubernetes_required=True,
    )

    assert payload["status"] == "failed"
    joined = "\n".join(payload["sky_check_output"])
    assert "Traceback (most recent call last):" in joined
    assert "ModuleNotFoundError: No module named 'kubernetes'" in joined


def test_verify_payload_keeps_non_module_errors_from_adaptors() -> None:
    adaptor_crash = "\n".join(
        [
            "  DO: disabled",
            "    Reason [compute]: Traceback (most recent call last):",
            '  File "/home/example/.npa/skypilot-venv/lib/python3.12/site-packages/sky/adaptors/do.py", line 24, in probe',
            "    raise RuntimeError('probe crashed')",
            "RuntimeError: probe crashed",
        ]
    )

    payload = skypilot_cli._verification_payload(
        _check_result(adaptor_crash),
        backend="kubernetes",
        kubernetes_required=False,
    )

    joined = "\n".join(payload["sky_check_output"])
    assert "Traceback (most recent call last):" in joined
    assert "RuntimeError: probe crashed" in joined


def test_collapse_survives_ansi_wrapped_lines() -> None:
    lines = [
        "\x1b[2m  DO: disabled \x1b[0m\x1b[0m",
        "    Reason [compute]: Traceback (most recent call last):",
        '  File "/venv/lib/python3.12/site-packages/sky/adaptors/do.py", line 24, in probe',
        "    if not common.can_import_modules(['azure.core.exceptions']):",
        "ModuleNotFoundError: No module named 'azure'",
        "  Fluidstack: disabled",
    ]

    collapsed = skypilot_cli._collapse_upstream_adaptor_tracebacks(lines)

    assert len(collapsed) == 3
    assert collapsed[0] == "\x1b[2m  DO: disabled \x1b[0m\x1b[0m"
    assert "missing optional skypilot dependency 'azure'" in collapsed[1]
    assert collapsed[2] == "  Fluidstack: disabled"
