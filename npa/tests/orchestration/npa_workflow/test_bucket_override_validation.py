"""Bucket overrides must stay bare names because specs compose ``s3://``."""

from pathlib import Path

import pytest

from npa.orchestration.npa_workflow.errors import NpaWorkflowError
from npa.orchestration.npa_workflow.spec import load_spec
from npa.orchestration.npa_workflow.submit import merge_config_overrides

SHIPPED = (
    Path(__file__).resolve().parents[4]
    / "workflows/testing/token-factory-trigger-watch.yaml"
)


def test_bare_bucket_override_is_accepted() -> None:
    spec = load_spec(SHIPPED)
    merged = merge_config_overrides(spec, {"bucket": "operator-bucket"})
    assert merged.config["bucket"] == "operator-bucket"


@pytest.mark.parametrize(
    "value",
    [
        "s3://operator-bucket",
        "s3://operator-bucket/",
        "operator-bucket/",
    ],
)
def test_decorated_bucket_override_fails_fast(value: str) -> None:
    spec = load_spec(SHIPPED)
    with pytest.raises(NpaWorkflowError, match="bare bucket name"):
        merge_config_overrides(spec, {"bucket": value})


def test_decorated_bucket_override_names_the_fix() -> None:
    spec = load_spec(SHIPPED)
    with pytest.raises(NpaWorkflowError) as excinfo:
        merge_config_overrides(spec, {"bucket": "s3://operator-bucket/"})
    message = str(excinfo.value)
    assert "s3://" in message
    assert "--var bucket=<name>" in message


def test_other_keys_keep_raw_values() -> None:
    spec = load_spec(SHIPPED)
    merged = merge_config_overrides(
        spec, {"images_uri": "s3://operator-bucket/inbox/"}
    )
    assert merged.config["images_uri"] == "s3://operator-bucket/inbox/"
