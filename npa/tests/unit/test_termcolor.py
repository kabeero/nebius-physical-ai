"""Tests for npa.termcolor terminal styling and capability detection."""

from unittest.mock import MagicMock

import pytest

from npa.termcolor import (
    BOLD,
    CYAN,
    DIM,
    RED,
    RESET,
    YELLOW,
    format_key_value,
    style,
    supports_color,
)


def test_supports_color_forced() -> None:
    assert supports_color(force=True) is True
    assert supports_color(force=False) is False


def test_supports_color_no_color_env(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = MagicMock()
    stream.isatty.return_value = True
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    assert supports_color(stream) is False


def test_supports_color_ci_env(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = MagicMock()
    stream.isatty.return_value = True
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv("TERM", "xterm-256color")
    assert supports_color(stream) is False


def test_supports_color_dumb_term(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = MagicMock()
    stream.isatty.return_value = True
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setenv("TERM", "dumb")
    assert supports_color(stream) is False


def test_supports_color_non_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = MagicMock()
    stream.isatty.return_value = False
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    assert supports_color(stream) is False


def test_supports_color_tty_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = MagicMock()
    stream.isatty.return_value = True
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    assert supports_color(stream) is True


def test_style_disabled() -> None:
    assert style("hello", color=YELLOW, bold=True, dim=True, enabled=False) == "hello"


def test_style_enabled() -> None:
    styled = style("hello", color=YELLOW, bold=True, enabled=True)
    assert styled == f"{BOLD}{YELLOW}hello{RESET}"


def test_format_key_value_disabled() -> None:
    assert format_key_value("Cluster ID", "123", enabled=False) == "Cluster ID: 123"


def test_format_key_value_enabled() -> None:
    formatted = format_key_value(
        "Cluster ID",
        "123",
        key_color=CYAN,
        key_bold=True,
        delim_dim=True,
        enabled=True,
    )
    assert formatted == f"{BOLD}{CYAN}Cluster ID{RESET}{DIM}:{RESET} 123"


def test_format_key_value_with_value_style() -> None:
    formatted = format_key_value(
        "Primary failure",
        "Error: failed",
        key_color=RED,
        key_bold=True,
        delim_dim=True,
        value_color=RED,
        enabled=True,
    )
    assert formatted == f"{BOLD}{RED}Primary failure{RESET}{DIM}:{RESET} {RED}Error: failed{RESET}"
