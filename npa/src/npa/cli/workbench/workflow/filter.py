"""Narrow the workflow catalog to workflows one operator can actually run."""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Mapping, Sequence
from typing import Any, NoReturn

import typer
from rich import box
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table

from npa.cli.workbench.workflow.catalog import TIER_ORDER
from npa.lifecycle_intent import OperationIntent, intent_boundary, json_stdout_contract
from npa.orchestration.npa_workflow.catalog_index import (
    CHECK_MARK,
    STAR,
    CatalogEntry,
    index_catalog,
)
from npa.orchestration.npa_workflow.errors import NpaWorkflowError

#: Rich styles for the interactive questions, menu numbers, and result
#: messages. Rich disables color automatically for piped output and honors
#: NO_COLOR, so styling stays terminal-only.
QUESTION_STYLE = "bold cyan"
NUMBER_STYLE = "bold magenta"

#: Application-filter value matching rows the keyword lookup leaves blank.
UNCATEGORIZED = "uncategorized"

#: GPU-filter menu values that are not one accelerator name.
ANY_GPU = "any"
CPU_ONLY = "none"

#: Leading per-row count in the catalog GPU column, as in ``8x B200``.
_GPU_COUNT_PREFIX = re.compile(r"^\d+x\s+")

_PVC_PROMPT = "Have you created a PVC (with RWX access) this cluster can mount?"


def register(app: typer.Typer) -> None:
    """Register the interactive catalog filter command.

    Args:
        app: Existing workflow command group.
    Returns:
        None.
    Raises:
        None.
    """
    app.command("filter")(filter_cmd)


@intent_boundary(OperationIntent.OBSERVE)
@json_stdout_contract
def filter_cmd(
    application: str = typer.Option(
        "",
        "--application",
        help=(
            "Application category filter, for example robotics-policy; "
            f"'{UNCATEGORIZED}' matches rows with no category."
        ),
    ),
    gpu: str = typer.Option(
        "",
        "--gpu",
        help=(
            "GPU filter: an accelerator name such as B200, 'any', or "
            "'none' for CPU-only workflows."
        ),
    ),
    pvc: bool | None = typer.Option(
        None,
        "--pvc/--no-pvc",
        help=(
            "Whether you have already created an RWX PVC. --no-pvc hides "
            "workflows that require one; --pvc keeps them eligible."
        ),
    ),
    tier: str = typer.Option(
        "", "--tier", help="Restrict matching to one catalog tier."
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Narrow the workflow catalog with guided questions or explicit flags.

    In a terminal the command asks, in order, the Application category,
    the GPU (an accelerator name, any GPU, or CPU only), and whether an
    RWX PVC is already mounted; each question accepts a blank answer for
    no filter. Every question can instead be answered with --application,
    --gpu, --pvc/--no-pvc, or --tier, which also makes the command
    scriptable. Answering that no PVC exists hides workflows that declare
    one, because Kubernetes will not schedule their pods until the named
    claim is bound. The result is a short table (name, application, GPU
    type, PVC, complexity, file) sorted by workflow name; a catalog entry
    remains a declared contract, never a run guarantee.

    Args:
        application: Application category, 'uncategorized', or empty.
        gpu: Accelerator name, 'any', 'none', or empty.
        pvc: None asks in a terminal (no filter when not interactive);
          False hides PVC-requiring workflows; True keeps them eligible.
        tier: One of main, testing, partners, cosmos-data-factory; empty
          searches every tier.
        json_output: Emit one JSON document instead of the table.
    Returns:
        None.
    Raises:
        typer.BadParameter: A tier, application, or GPU value is unknown.
        typer.Exit: Catalog indexing failed, or the command ran without
          filters on a non-interactive terminal.
    """
    if tier and tier not in TIER_ORDER:
        valid = ", ".join(TIER_ORDER)
        raise typer.BadParameter(f"unknown tier {tier!r}; choose one of: {valid}")
    try:
        entries = index_catalog()
    except (NpaWorkflowError, OSError) as exc:
        typer.echo(f"catalog indexing failed: {exc}", err=True)
        raise typer.Exit(1) from None
    rows = _tier_rows(tier, entries)
    application = _validated(application, _application_choices(rows), "application")
    gpu = _validated(gpu, _gpu_choices(rows), "gpu")
    interactive = _is_interactive()
    if not interactive and not (application or gpu or pvc is not None or tier):
        _reject_without_filters()
    if interactive:
        if not application:
            application = _choose("Application?", _application_choices(rows))
        if not gpu:
            gpu = _choose("GPU?", _gpu_choices(rows))
        if pvc is None:
            pvc = Confirm.ask(_PVC_PROMPT, default=False, console=_stderr_console())
    matched = [row for row in rows if _matches(row, application, gpu, pvc)]
    matched.sort(key=lambda row: (row.name, row.application))
    if json_output:
        typer.echo(
            json.dumps(
                _filter_json(matched, application, gpu, pvc, tier),
                indent=2,
                sort_keys=True,
            )
        )
        return
    # A yes PVC answer keeps every row eligible, so it does not narrow the
    # catalog; only a no answer (or another filter) does.
    narrows = bool(application or gpu or pvc is False or tier)
    _print_matches(matched, narrows=narrows)


def _is_interactive() -> bool:
    """Whether a user can answer prompts on this terminal.

    Args:
        None.
    Returns:
        True when stdin is a terminal.
    Raises:
        None.
    """
    return sys.stdin.isatty()


def _reject_without_filters() -> NoReturn:
    """Fail a filterless, non-interactive invocation with usable guidance.

    Args:
        None.
    Returns:
        Never returns.
    Raises:
        typer.Exit: Always, with exit code 1.
    """
    console = _stderr_console()
    console.print("[red]This command is interactive and no filters were given.[/red]")
    console.print("Run it in a terminal to pick filters, or pass them as flags:")
    console.print("  npa workbench workflow filter --application robotics-policy")
    console.print("  npa workbench workflow filter --gpu B200")
    console.print("  npa workbench workflow filter --no-pvc")
    console.print(
        "[dim]Or list every workflow with: npa workbench workflow catalog[/dim]"
    )
    raise typer.Exit(1)


def _tier_rows(
    tier: str, entries: Mapping[str, list[CatalogEntry]]
) -> list[CatalogEntry]:
    """Flatten the selected tiers' rows in the fixed catalog tier order.

    Args:
        tier: One catalog tier name, or empty for every tier.
        entries: Tier name to indexed rows from the catalog indexer.
    Returns:
        The rows of the selected tiers, unsorted.
    Raises:
        None.
    """
    return [
        row
        for name in TIER_ORDER
        if name in entries and (not tier or name == tier)
        for row in entries[name]
    ]


def _application_choices(rows: Sequence[CatalogEntry]) -> list[tuple[str, str]]:
    """Application menu options for the rows on the table.

    Args:
        rows: Rows the filter currently searches.
    Returns:
        Display/value pairs, one per unique application category, with an
        '(uncategorized)' entry when any row has no category.
    Raises:
        None.
    """
    applications = sorted({row.application for row in rows if row.application})
    choices = [(application, application) for application in applications]
    if any(not row.application for row in rows):
        choices.append((f"({UNCATEGORIZED})", UNCATEGORIZED))
    return choices


def _gpu_choices(rows: Sequence[CatalogEntry]) -> list[tuple[str, str]]:
    """GPU menu options for the rows on the table.

    Args:
        rows: Rows the filter currently searches.
    Returns:
        Display/value pairs: any GPU, CPU only, then every unique
        accelerator name the catalog's GPU column mentions.
    Raises:
        None.
    """
    names = sorted({name for row in rows for name in _gpu_names(row)})
    choices = [("Any GPU", ANY_GPU), ("None (CPU only)", CPU_ONLY)]
    choices.extend((name, name) for name in names)
    return choices


def _gpu_names(row: CatalogEntry) -> set[str]:
    """Accelerator names in one row's catalog GPU column, counts stripped.

    The catalog renders ``8x B200``-style summaries, so matching by GPU
    type needs the bare names.

    Args:
        row: One indexed catalog row.
    Returns:
        Every accelerator name the row's GPU column names.
    Raises:
        None.
    """
    return {
        _GPU_COUNT_PREFIX.sub("", part) for part in row.gpu_type.split(", ") if part
    }


def _validated(value: str, choices: Sequence[tuple[str, str]], label: str) -> str:
    """Pass a flag value through the same values the interactive menu offers.

    Args:
        value: Flag value; empty means no filter and always validates.
        choices: Display/value pairs of the matching menu.
        label: Field name for the error message.
    Returns:
        The validated value.
    Raises:
        typer.BadParameter: The value is not one the menu offers.
    """
    if not value:
        return value
    valid = {menu_value for _display, menu_value in choices}
    if value not in valid:
        options = ", ".join(sorted(valid))
        raise typer.BadParameter(f"unknown {label} {value!r}; choose one of: {options}")
    return value


def _choose(label: str, choices: Sequence[tuple[str, str]]) -> str:
    """Ask one numbered-list question on stderr; blank input skips it.

    The question renders in the question style, the option numbers in the
    number style, and the item text plain; color disappears for piped
    output and under NO_COLOR.

    Args:
        label: Question printed above the numbered options.
        choices: Display/value pairs, numbered starting at one.
    Returns:
        The chosen option's value, or "" for the blank answer.
    Raises:
        EOFError: The prompt's input stream closed before an answer.
    """
    console = _stderr_console()
    console.print(f"[{QUESTION_STYLE}]{label}[/{QUESTION_STYLE}]")
    for index, (display, _value) in enumerate(choices, start=1):
        console.print(f"  [{NUMBER_STYLE}]{index}[/{NUMBER_STYLE}]. {display}")
    while True:
        raw = Prompt.ask(
            "Choice (blank = no filter)",
            default="",
            show_default=False,
            console=console,
        ).strip()
        if not raw:
            return ""
        if raw.isdigit() and 1 <= int(raw) <= len(choices):
            return choices[int(raw) - 1][1]
        console.print(
            f"[red]Enter a number from 1 to {len(choices)}, or leave blank.[/red]"
        )


def _matches(row: CatalogEntry, application: str, gpu: str, pvc: bool | None) -> bool:
    """Whether one row survives every active filter.

    Args:
        row: One indexed catalog row.
        application: Application category, 'uncategorized', or "".
        gpu: Accelerator name, 'any', 'none', or "".
        pvc: False hides PVC-requiring rows; True and None keep every row.
    Returns:
        True when the row matches every active filter.
    Raises:
        None.
    """
    if application and not _matches_application(row, application):
        return False
    if gpu and not _matches_gpu(row, gpu):
        return False
    if pvc is False and row.pvc_cache:
        return False
    return True


def _matches_application(row: CatalogEntry, application: str) -> bool:
    """Application-filter predicate for one row.

    Args:
        row: One indexed catalog row.
        application: Category name or 'uncategorized'; never empty here.
    Returns:
        True when the row carries that category.
    Raises:
        None.
    """
    if application == UNCATEGORIZED:
        return not row.application
    return row.application == application


def _matches_gpu(row: CatalogEntry, gpu: str) -> bool:
    """GPU-filter predicate for one row.

    Args:
        row: One indexed catalog row.
        gpu: 'any', 'none', or one accelerator name.
    Returns:
        True when the row's compute matches the GPU filter.
    Raises:
        None.
    """
    if gpu == ANY_GPU:
        return "GPU" in row.modality
    if gpu == CPU_ONLY:
        return row.modality == "CPU"
    return gpu in _gpu_names(row)


def _filter_json(
    rows: Sequence[CatalogEntry],
    application: str,
    gpu: str,
    pvc: bool | None,
    tier: str,
) -> dict[str, Any]:
    """JSON payload: the effective filters and the surviving rows.

    Args:
        rows: Surviving rows in display order.
        application: Effective application filter, or "".
        gpu: Effective GPU filter, or "".
        pvc: Effective PVC answer; None when unconstrained.
        tier: Effective tier filter, or "".
    Returns:
        The JSON document for --json.
    Raises:
        None.
    """
    return {
        "filters": {
            "application": application,
            "gpu": gpu,
            "pvc": pvc,
            "tier": tier,
        },
        "workflows": [row.as_dict() for row in rows],
    }


def _print_matches(rows: Sequence[CatalogEntry], *, narrows: bool) -> None:
    """Render the human result: the short table or the no-match guidance.

    A blank line separates the prompts or the earlier note from the
    result table.

    Args:
        rows: Surviving rows sorted by name, then application.
        narrows: Whether any filter narrowed the catalog.
    Returns:
        None.
    Raises:
        None.
    """
    console = _result_console()
    if not rows:
        console.print("[red]No workflows match the selected filters.[/red]")
        console.print(
            "[dim]Relax a filter, or list every workflow with: "
            "npa workbench workflow catalog[/dim]"
        )
        return
    if not narrows:
        _stderr_console().print(
            "[dim]No filters selected; showing every workflow.[/dim]"
        )
    console.print()
    console.print(_match_table(rows))


def _result_console() -> Console:
    """Console wide enough that result cells never truncate to ellipsis.

    A piped rich console defaults to 80 columns; the floor keeps declared
    GPU names and catalog-relative file paths intact for piping and grep,
    matching the catalog tables' no-truncation contract.

    Args:
        None.
    Returns:
        The console the result table renders into.
    Raises:
        None.
    """
    return Console(width=max(Console().width, 190))


def _stderr_console() -> Console:
    """Console for prompts and side notes, never for result content.

    Auto-highlighting stays off so plain menu text keeps its color: the
    highlighter would recolor fragments such as the ``None`` entry or the
    digits inside a GPU name.

    Args:
        None.
    Returns:
        The console questions and notes render into.
    Raises:
        None.
    """
    return Console(stderr=True, highlighter=False)


def _match_table(rows: Sequence[CatalogEntry]) -> Table:
    """Build the shortened beginner table for the surviving rows.

    Args:
        rows: Surviving rows sorted by name, then application.
    Returns:
        The rich table to print.
    Raises:
        None.
    """
    table = Table(
        title=f"Matching workflows ({len(rows)})",
        title_justify="left",
        title_style="bold cyan",
        header_style="bold",
        border_style="dim",
        box=box.ROUNDED,
    )
    for column in ("Name", "Application", "GPU type", "PVC", "Complexity", "File"):
        table.add_column(column)
    for row in rows:
        table.add_row(
            row.name,
            row.application,
            row.gpu_type or "[dim]CPU[/dim]",
            _flag(row.pvc_cache),
            f"[yellow]{STAR * row.complexity}[/yellow]",
            row.rel_path,
        )
    return table


def _flag(flagged: bool) -> str:
    """Green check mark for a true flag, empty cell for false.

    Args:
        flagged: Whether the row carries the flag.
    Returns:
        The rendered cell.
    Raises:
        None.
    """
    return f"[green]{CHECK_MARK}[/green]" if flagged else ""
