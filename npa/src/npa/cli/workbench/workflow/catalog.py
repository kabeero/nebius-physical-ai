"""List the discovered npa.workflow catalog as tiered operator tables."""

from __future__ import annotations

import json

import typer
from rich import box
from rich.console import Console
from rich.table import Table

from npa.lifecycle_intent import OperationIntent, intent_boundary, json_stdout_contract
from npa.orchestration.npa_workflow.catalog_index import (
    CHECK_MARK,
    STAR,
    CatalogEntry,
    index_catalog,
)
from npa.orchestration.npa_workflow.errors import NpaWorkflowError

#: Fixed table order: principal, reference, partner, factory tiers.
TIER_ORDER = ("main", "testing", "partners", "cosmos-data-factory")

TIER_TITLES = {
    "main": "Main workflows",
    "testing": "Testing and reference workflows",
    "partners": "Partner workflows",
    "cosmos-data-factory": "Cosmos Data Factory workflows",
}

#: Rich styles for the tiered catalog tables. Rich disables color
#: automatically for piped output and honors NO_COLOR, so styling stays
#: terminal-only.
TIER_TITLE_STYLE = "bold cyan"
HEADER_STYLE = "bold"
BORDER_STYLE = "dim"


@intent_boundary(OperationIntent.OBSERVE)
@json_stdout_contract
def catalog_cmd(
    tier: str = typer.Option(
        "", "--tier", help="Restrict the listing to one catalog tier."
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """List every discovered npa.workflow spec as tiered catalog tables.

    Rows report each spec's declared contract: semantic application
    category, compute modality, resource requests, phase count with
    parallel fan-outs collapsed to one phase, artifact flags, and a
    one-to-five star operating-complexity heuristic; partners-tier rows
    also name the owning partner directory. A catalog entry is
    a contract, not a guarantee that every configuration has run
    successfully; read each spec's guide before allocating compute.

    Args:
        tier: One of main, testing, partners, cosmos-data-factory; empty
          lists every tier.
        json_output: Emit one JSON document instead of tables.
    Returns:
        None.
    Raises:
        typer.BadParameter: The selected tier does not exist.
        typer.Exit: A discovered catalog spec failed to load and validate.
    """
    selected = _selected_tiers(tier)
    try:
        entries = index_catalog()
    except (NpaWorkflowError, OSError) as exc:
        typer.echo(f"catalog indexing failed: {exc}", err=True)
        raise typer.Exit(1) from None
    listed = {name: entries[name] for name in selected if name in entries}
    if json_output:
        typer.echo(
            json.dumps({"tiers": _tiers_json(listed)}, indent=2, sort_keys=True)
        )
        return
    console = _catalog_console()
    for name in listed:
        console.print(_tier_table(name, listed[name]))
        console.print()


def register(app: typer.Typer) -> None:
    """Register the catalog listing command.

    Args:
        app: Existing workflow command group.
    Returns:
        None.
    Raises:
        None.
    """
    app.command("catalog")(catalog_cmd)


def _catalog_console() -> Console:
    """Console wide enough that catalog cells never truncate to ellipsis.

    A piped rich console defaults to 80 columns, which a twelve-column
    catalog cannot fit; the floor keeps declared values intact for
    piping and grep instead of shortening them for narrow terminals.

    Args: None.
    Returns:
        The console the catalog tables render into.
    Raises: None.
    """
    return Console(width=max(Console().width, 190))


def _selected_tiers(tier: str) -> tuple[str, ...]:
    """Resolve the ``--tier`` filter against the known catalog tiers."""

    if not tier:
        return TIER_ORDER
    if tier not in TIER_ORDER:
        valid = ", ".join(TIER_ORDER)
        raise typer.BadParameter(
            f"unknown tier {tier!r}; choose one of: {valid}"
        )
    return (tier,)


def _tiers_json(listed: dict[str, list[CatalogEntry]]) -> dict[str, list[dict]]:
    """JSON payload for the selected tiers."""

    return {
        name: [row.as_dict() for row in listed[name]] for name in listed
    }


def _tier_table(tier: str, rows: list[CatalogEntry]) -> Table:
    """Build one tier's styled catalog table.

    Partners-tier rows lead with the owning partner directory and group
    by it (partner name, then workflow name); other tiers keep the
    indexer's name order.

    Args:
        tier: Tier the rows belong to.
        rows: Already-indexed entries for that tier.
    Returns:
        None.
    Raises:
        None.
    """
    table = Table(
        title=f"{TIER_TITLES[tier]} ({len(rows)})",
        title_justify="left",
        title_style=TIER_TITLE_STYLE,
        header_style=HEADER_STYLE,
        border_style=BORDER_STYLE,
        box=box.ROUNDED,
    )
    columns = [
        "Name",
        "Application",
        "Modality",
        "CPUs",
        "Memory",
        "GPU cloud",
        "GPU type",
        "Phases",
        "In",
        "Out",
        "PVC",
        "Complexity",
    ]
    # Only the merged partners tier needs the owning partner directory;
    # every other tier's rows share one owner, so the column would be blank.
    # It leads the table so the merged rows read grouped by partner.
    if tier == "partners":
        columns.insert(0, "Partner")
        rows = sorted(rows, key=lambda row: (row.partner, row.name))
    for column in columns:
        table.add_column(column)
    for row in rows:
        values = [
            _display_name(row),
            row.application,
            row.modality,
            row.cpus,
            row.memory,
            row.gpu_cloud,
            row.gpu_type,
            str(row.phases),
            _flag(row.inputs),
            _flag(row.outputs),
            _flag(row.pvc_cache),
            f"[yellow]{STAR * row.complexity}[/yellow]",
        ]
        if tier == "partners":
            values.insert(0, row.partner)
        table.add_row(*values)
    return table


def _flag(flagged: bool) -> str:
    """Green check mark for true flags, empty cell for false."""

    return f"[green]{CHECK_MARK}[/green]" if flagged else ""


def _display_name(row: CatalogEntry) -> str:
    """Workflow name, with the file name when the two differ."""

    file_name = row.rel_path.rsplit("/", 1)[-1]
    stem = file_name[: -len(".yaml")]
    if stem == row.name:
        return row.name
    return f"{row.name} ({file_name})"
