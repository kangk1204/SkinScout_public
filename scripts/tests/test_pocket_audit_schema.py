"""F47: the legacy pocket-leakage audit is not evidence about the current benchmark.

The stored `results/audits/pocket_leakage_audit_202608.json` binds the superseded
`activity_benchmark_202608` universe and the evidence-only
`skinscout.target-cluster-map.v2` map. The current benchmark binds
`skinscout.screenable-target-cluster-map.v2`, so the sidecar audit keeps the old
result as a labeled historical record and refuses to consume it as current
leakage evidence. The separate RCSB contact-pocket audit is untouched.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "eval" / "audit_pocket_leakage.py"
RCSB_AUDIT = ROOT / "eval" / "audit_rcsb_contact_pocket_leakage.py"
LEGACY_ARTIFACT = ROOT / "results" / "audits" / "pocket_leakage_audit_202608.json"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_target_map(tmp_path: Path, schema_version: str) -> tuple[Path, Path]:
    target_csv = tmp_path / "target_clusters.csv"
    target_csv.write_text(
        "uniprot,target_cluster_30,target_cluster_50\nP1,seq30-P1,seq50-P1\n",
        encoding="utf-8",
    )
    target_manifest = tmp_path / "target_clusters.manifest.json"
    target_manifest.write_text(
        json.dumps(
            {
                "schema_version": schema_version,
                "artifact": {
                    "path": str(target_csv.resolve()),
                    "sha256": hashlib.sha256(target_csv.read_bytes()).hexdigest(),
                    "rows": 1,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return target_csv, target_manifest


def test_the_current_screenable_schema_is_the_audit_boundary(tmp_path: Path) -> None:
    module = _load(AUDIT, "pocket_audit_current_schema")
    target_csv, target_manifest = _write_target_map(
        tmp_path, "skinscout.screenable-target-cluster-map.v2"
    )

    frame, manifest = module._validate_target_clusters(target_csv, target_manifest)

    assert module.TARGET_CLUSTER_SCHEMA_VERSION == "skinscout.screenable-target-cluster-map.v2"
    assert manifest["schema_version"] == "skinscout.screenable-target-cluster-map.v2"
    assert list(frame["uniprot"]) == ["P1"]


def test_the_legacy_target_cluster_schema_is_rejected_as_deprecated(tmp_path: Path) -> None:
    module = _load(AUDIT, "pocket_audit_legacy_schema")
    target_csv, target_manifest = _write_target_map(
        tmp_path, "skinscout.target-cluster-map.v2"
    )

    with pytest.raises(SystemExit, match="historical/deprecated"):
        module._validate_target_clusters(target_csv, target_manifest)


def test_a_legacy_payload_cannot_be_consumed_as_current_evidence() -> None:
    module = _load(AUDIT, "pocket_audit_legacy_guard")
    legacy_payload = {
        "schema_version": "skinscout.pocket-leakage-audit.v1",
        "inputs": {
            "target_clusters": {
                "manifest_schema_version": "skinscout.target-cluster-map.v2"
            }
        },
    }

    with pytest.raises(SystemExit, match="historical/deprecated"):
        module.require_current_audit_evidence(legacy_payload)


def test_a_current_payload_passes_the_evidence_guard() -> None:
    module = _load(AUDIT, "pocket_audit_current_guard")
    payload = {
        "schema_version": "skinscout.pocket-leakage-audit.v1",
        "evidence_status": "current",
        "current_evidence": True,
        "inputs": {
            "target_clusters": {
                "manifest_schema_version": "skinscout.screenable-target-cluster-map.v2"
            }
        },
    }

    assert module.require_current_audit_evidence(payload) is payload


@pytest.mark.skipif(
    not LEGACY_ARTIFACT.exists(),
    reason="the historical 202608 audit artifact is not present",
)
def test_the_stored_historical_audit_is_labeled_and_rejected() -> None:
    module = _load(AUDIT, "pocket_audit_stored_artifact")
    payload = json.loads(LEGACY_ARTIFACT.read_text(encoding="utf-8"))

    assert payload["artifact_status"] == "historical_deprecated"
    assert payload["evidence_status"] == "historical_deprecated"
    assert payload["current_evidence"] is False
    assert (
        payload["inputs"]["target_clusters"]["manifest_schema_version"]
        == "skinscout.target-cluster-map.v2"
    )

    with pytest.raises(SystemExit, match="historical/deprecated"):
        module.require_current_audit_evidence(payload)


def test_the_rcsb_contact_audit_keeps_its_own_schema_protection() -> None:
    module = _load(RCSB_AUDIT, "rcsb_contact_audit_schemas")

    assert module.RCSB_FRAGMENT_SCHEMA_VERSION == "skinscout.rcsb-contact-pocket-fragments.v1"
    assert module.PRIOR_FRAGMENT_SCHEMA_VERSION == "skinscout.pocket-fragment-universe.v1"
    assert module.BENCHMARK_SCHEMA_VERSION == "activity_benchmark.v1"
    assert not hasattr(module, "LEGACY_TARGET_CLUSTER_SCHEMA_VERSIONS")
