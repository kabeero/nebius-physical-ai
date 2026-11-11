"""Unit tests for the missing-cluster-identity controller adoption refusal."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from npa.clients import config
from npa import controller_ownership as ownership
from npa.cluster import state as cluster_state
from npa.controller_ownership import (
    ClusterOwnerIdentityMismatchError,
    resolve_controller_candidate,
)


@pytest.fixture
def isolated_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Configure one ``prod`` project with an isolated cluster-state root."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "default_project": "prod",
                "projects": {
                    "prod": {
                        "project_id": "project-a",
                        "tenant_id": "tenant-a",
                        "region": "region-a",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "CONFIG_PATH", config_path)
    monkeypatch.setattr(ownership, "CONFIG_PATH", config_path)
    monkeypatch.setattr(cluster_state, "CLUSTERS_DIR", tmp_path / "clusters")
    monkeypatch.setenv("NPA_OPERATION_JOURNAL_DIR", str(tmp_path / "operations"))
    return "prod"


def _save_cluster(name: str, cluster_id: str) -> None:
    cluster_state.save_cluster_state(
        cluster_state.ClusterState(
            name=name,
            cluster_id=cluster_id,
            project_id="project-a",
            region="region-a",
            node_count=1,
            node_platform="gpu-rtx6000",
            node_preset="1gpu-24vcpu-218gb",
            k8s_version="1.31",
            subnet_id="subnet-a",
            created_at="2026-08-08T00:00:00Z",
        )
    )


def test_missing_identity_without_adopted_clusters_names_remedy(
    isolated_project: str,
) -> None:
    with pytest.raises(ClusterOwnerIdentityMismatchError) as excinfo:
        resolve_controller_candidate(isolated_project, "unseen-context")

    message = str(excinfo.value)
    assert "No NPA cluster identity exists for context 'unseen-context'" in message
    assert "No cluster is adopted locally yet" in message
    assert (
        "npa cluster kubeconfig --cluster-name <provider-cluster-name> "
        "--project prod --context unseen-context" in message
    )
    assert (
        "npa skypilot bind-controller --project prod --context unseen-context"
        in message
    )


def test_missing_identity_lists_adopted_contexts(isolated_project: str) -> None:
    _save_cluster("known-context", "cluster-live")

    with pytest.raises(ClusterOwnerIdentityMismatchError) as excinfo:
        resolve_controller_candidate(isolated_project, "unseen-context")

    message = str(excinfo.value)
    assert "known-context (cluster-live)" in message
    assert "Submit with one of those contexts" in message
    assert "--context unseen-context" in message


def test_missing_identity_survives_malformed_sibling_state(
    isolated_project: str,
) -> None:
    malformed = cluster_state.CLUSTERS_DIR / "broken-context"
    malformed.mkdir(parents=True)
    (malformed / "cluster.json").write_text("{ not json", encoding="utf-8")

    with pytest.raises(ClusterOwnerIdentityMismatchError) as excinfo:
        resolve_controller_candidate(isolated_project, "unseen-context")

    assert "npa cluster kubeconfig" in str(excinfo.value)
