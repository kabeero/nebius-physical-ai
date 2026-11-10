"""Terminal color detection and styling helpers for NPA CLI output."""

from __future__ import annotations

import os
import sys
from typing import Any

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"


def supports_color(stream: Any = None, *, force: bool | None = None) -> bool:
    """Return whether colored ANSI output is appropriate for the stream."""
    if force is not None:
        return bool(force)
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("CI"):
        return False
    if os.environ.get("TERM", "").strip().lower() == "dumb":
        return False

    target_stream = stream if stream is not None else sys.stderr
    isatty = getattr(target_stream, "isatty", None)
    if callable(isatty):
        try:
            return bool(isatty())
        except Exception:
            return False
    return False


def style(
    text: str,
    *,
    color: str = "",
    bold: bool = False,
    dim: bool = False,
    enabled: bool = True,
) -> str:
    """Apply ANSI styling to text if enabled is True."""
    if not enabled or not text:
        return str(text)

    prefix_parts = []
    if bold:
        prefix_parts.append(BOLD)
    if dim:
        prefix_parts.append(DIM)
    if color:
        prefix_parts.append(color)

    prefix = "".join(prefix_parts)
    if not prefix:
        return str(text)
    return f"{prefix}{text}{RESET}"


def format_key_value(
    key: str,
    value: str,
    *,
    key_color: str = "",
    key_bold: bool = True,
    delim_dim: bool = True,
    value_color: str = "",
    value_bold: bool = False,
    enabled: bool = True,
) -> str:
    """Format a key: value pair with optional ANSI styling on key, delimiter, and value."""
    if not enabled:
        return f"{key}: {value}"

    styled_key = style(key, color=key_color, bold=key_bold, enabled=True)
    styled_delim = style(":", dim=delim_dim, enabled=True)
    styled_value = (
        style(value, color=value_color, bold=value_bold, enabled=True)
        if (value_color or value_bold)
        else value
    )
    return f"{styled_key}{styled_delim} {styled_value}"
