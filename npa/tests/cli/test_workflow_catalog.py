"""CLI tests for ``npa workbench workflow catalog``."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from npa.cli.main import app

runner = CliRunner()


def test_catalog_lists_every_tier() -> None:
    result = runner.invoke(app, ["workbench", "workflow", "catalog"])
    assert result.exit_code == 0, result.output
    for title in (
        "Main workflows",
        "Testing and reference workflows",
        "Partner workflows",
        "Cosmos Data Factory workflows",
    ):
        assert title in result.output
    assert "sim2real" in result.output
    assert "robot-policy-train-and-serve.yaml" in result.output


def test_catalog_json_emits_one_document() -> None:
    result = runner.invoke(app, ["workbench", "workflow", "catalog", "--json"])
    assert result.exit_code == 0, result.output
    document = json.loads(result.output)
    assert set(document) == {"tiers"}
    tiers = document["tiers"]
    assert set(tiers) == {
        "main",
        "testing",
        "partners",
        "cosmos-data-factory",
    }
    sample = tiers["main"][0]
    assert sample["name"]
    assert "complexity" in sample
    assert "phases" in sample


def test_catalog_tier_filter() -> None:
    result = runner.invoke(
        app, ["workbench", "workflow", "catalog", "--tier", "partners"]
    )
    assert result.exit_code == 0, result.output
    assert "Partner workflows" in result.output
    assert "Testing and reference workflows" not in result.output
    assert "encord-pull" in result.output


def test_catalog_partners_table_names_partner() -> None:
    partners = runner.invoke(
        app, ["workbench", "workflow", "catalog", "--tier", "partners"]
    )
    assert partners.exit_code == 0, partners.output
    assert "│ Partner │ Name" in partners.output
    assert all(
        f"│ {partner} " in partners.output for partner in ("antioch", "encord")
    )
    # Rows group by partner: every antioch row precedes every encord row.
    assert partners.output.index("xr1-antioch-finetune") < partners.output.index(
        "encord-labeling-demo"
    )
    main = runner.invoke(
        app, ["workbench", "workflow", "catalog", "--tier", "main"]
    )
    assert main.exit_code == 0, main.output
    assert "Partner" not in main.output


def test_catalog_json_partners_have_partner() -> None:
    result = runner.invoke(app, ["workbench", "workflow", "catalog", "--json"])
    assert result.exit_code == 0, result.output
    document = json.loads(result.output)
    assert {row["partner"] for row in document["tiers"]["partners"]} == {
        "antioch",
        "encord",
    }
    assert document["tiers"]["main"][0]["partner"] == ""


def test_catalog_rejects_unknown_tier() -> None:
    result = runner.invoke(
        app, ["workbench", "workflow", "catalog", "--tier", "bogus"]
    )
    assert result.exit_code != 0
    assert "unknown tier" in result.output
