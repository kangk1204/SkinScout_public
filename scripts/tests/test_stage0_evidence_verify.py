"""Focused checks for large activity-evidence Stage 0 validation."""

from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from stage0_verify import (  # noqa: E402
    chk_bindingdb_evidence_integrity,
    chk_chembl_evidence_integrity,
    chk_chembl_fingerprint_integrity,
    chk_evidence_manifest,
    chk_gtopdb_evidence_integrity,
    chk_parquet_schema,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_chembl_snapshot(root: Path) -> None:
    root.mkdir()
    human = root / "human_activities.parquet"
    evidence = root / "activity_evidence.parquet"
    fingerprints = root / "fp_morgan2_2048.parquet"
    pd.DataFrame([{
        "molecule_chembl_id": "CHEMBL1",
        "smiles": "CCO",
        "uniprot": "P11111",
    }]).to_parquet(human, index=False)
    pd.DataFrame([{
        "molecule_chembl_id": "CHEMBL1",
        "smiles": "CCO",
        "uniprot": "P11111",
    }]).to_parquet(evidence, index=False)
    pd.DataFrame([{
        "molecule_chembl_id": "CHEMBL1",
        "smiles": "CCO",
        "bitvec": ["0"] * 32,
    }]).to_parquet(fingerprints, index=False)
    source_manifest = root / "source_manifest.json"
    source_manifest.write_text(json.dumps({
        "schema_version": "chembl_activity_evidence.v1",
        "source": {
            "name": "ChEMBL",
            "release": "37",
            "license": "CC BY-SA 3.0",
        },
        "output_sha256": {
            human.name: _sha256(human),
            evidence.name: _sha256(evidence),
        },
        "row_counts": {"human_activities": 1, "activity_evidence": 1},
    }))
    (root / "fingerprint_manifest.json").write_text(json.dumps({
        "schema_version": "chembl_fingerprint_snapshot.v1",
        "source_snapshot": {
            "manifest_sha256": _sha256(source_manifest),
            "source_release": "37",
            "source_license": "CC BY-SA 3.0",
        },
        "algorithm": {
            "name": "Morgan ECFP4",
            "radius": 2,
            "n_bits": 2048,
            "use_chirality": False,
        },
        "input": {"sha256": _sha256(human), "rows": 1},
        "artifact": {"sha256": _sha256(fingerprints), "rows": 1},
    }))


def _write_gtopdb_snapshot(root: Path) -> None:
    root.mkdir()
    evidence_dir = root / "evidence_v1"
    evidence_dir.mkdir()
    for name, text in (
        ("interactions.csv", "banner\nheader\nrow\n"),
        ("ligands.csv", "banner\nheader\nrow\n"),
        ("GtP_to_UniProt_mapping.csv", "banner\nheader\nrow\n"),
        ("file_descriptions.txt", "GtoPdb files\n"),
        ("pubmed_esummary.json", '{"schema_version":"skinscout.pubmed-esummary-cache.v1"}\n'),
    ):
        (root / name).write_text(text)
    source_manifest = {
        "schema_version": "skinscout.gtopdb-source.v1",
        "source": {
            "name": "GtoPdb",
            "release": "2026.2",
            "release_date": "2026-06-15",
            "license": "ODbL 1.0; contents CC BY-SA 4.0",
            "license_url": "https://www.guidetopharmacology.org/about.jsp#license",
        },
        "artifacts": {},
    }
    for name in (
        "interactions.csv",
        "ligands.csv",
        "GtP_to_UniProt_mapping.csv",
        "file_descriptions.txt",
        "pubmed_esummary.json",
    ):
        path = root / name
        source_manifest["artifacts"][name] = {
            "path": str(path),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
    source_manifest_path = root / "source_manifest.json"
    source_manifest_path.write_text(json.dumps(source_manifest))

    rows = pd.DataFrame([
        {
            "evidence_id": "GTP1",
            "source_db": "GtoPdb",
            "temporal_split": "pre_cutoff",
        }
    ])
    outputs = {
        "activity_evidence": evidence_dir / "activity_evidence.parquet",
        "pre_cutoff": evidence_dir / "pre_cutoff.parquet",
        "post_cutoff": evidence_dir / "post_cutoff.parquet",
    }
    rows.to_parquet(outputs["activity_evidence"], index=False)
    rows.to_parquet(outputs["pre_cutoff"], index=False)
    rows.iloc[0:0].to_parquet(outputs["post_cutoff"], index=False)
    artifacts = {
        key: {
            "path": str(path),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
            "rows": len(pd.read_parquet(path)),
        }
        for key, path in outputs.items()
    }
    evidence_manifest = {
        "schema_version": "gtopdb_activity_evidence.v1",
        "source": {
            "name": "GtoPdb",
            "release": "2026.2",
            "license": "ODbL 1.0; contents CC BY-SA 4.0",
        },
        "inputs": {
            "source_manifest": {
                "sha256": _sha256(source_manifest_path),
                "schema_version": "skinscout.gtopdb-source.v1",
            },
            "interactions_csv": {"sha256": _sha256(root / "interactions.csv")},
            "ligands_csv": {"sha256": _sha256(root / "ligands.csv")},
            "target_mapping_csv": {
                "sha256": _sha256(root / "GtP_to_UniProt_mapping.csv"),
            },
            "pubmed_json": {"sha256": _sha256(root / "pubmed_esummary.json")},
        },
        "policy": {},
        "row_counts": {
            "activity_evidence": 1,
            "pre_cutoff": 1,
            "post_cutoff": 0,
        },
        "artifacts": artifacts,
        "output_sha256": {
            "activity_evidence.parquet": artifacts["activity_evidence"]["sha256"],
            "pre_cutoff.parquet": artifacts["pre_cutoff"]["sha256"],
            "post_cutoff.parquet": artifacts["post_cutoff"]["sha256"],
        },
    }
    (evidence_dir / "manifest.json").write_text(json.dumps(evidence_manifest))


def _write_bindingdb_snapshot(root: Path) -> None:
    root.mkdir()
    archive = root / "BindingDB_All_202608_tsv.zip"
    extracted = root / "BindingDB_All.tsv"
    archive.write_bytes(b"bindingdb archive fixture")
    extracted.write_text("Ligand SMILES\tKi (nM)\nCCO\t10\n")
    mirror_manifest = {
        "schema_version": 1,
        "source": {
            "name": "BindingDB",
            "license": "CC BY 3.0",
            "license_url": "https://www.bindingdb.org/rwd/bind/info.jsp",
        },
        "release": "2026-08",
        "archive": {"path": archive.name, "sha256": _sha256(archive)},
        "extracted": {"path": extracted.name, "sha256": _sha256(extracted)},
    }
    mirror_path = root / "bindingdb_source_manifest.json"
    mirror_path.write_text(json.dumps(mirror_manifest))

    evidence_dir = root / "evidence_v1"
    evidence_dir.mkdir()
    frame = pd.DataFrame(
        [{"evidence_id": "BDB1", "temporal_split": "pre_cutoff"}]
    )
    outputs = {
        "activity_evidence": evidence_dir / "activity_evidence.parquet",
        "pre_cutoff": evidence_dir / "pre_cutoff.parquet",
        "post_cutoff": evidence_dir / "post_cutoff.parquet",
    }
    frame.to_parquet(outputs["activity_evidence"], index=False)
    frame.to_parquet(outputs["pre_cutoff"], index=False)
    frame.iloc[0:0].to_parquet(outputs["post_cutoff"], index=False)
    evidence_manifest = {
        "schema_version": "bindingdb_activity_evidence.v1",
        "source": {
            "path": str(extracted),
            "sha256": _sha256(extracted),
            "source_db": "BindingDB",
            "source_release": "2026-08",
            "source_license": "CC BY 3.0 (BindingDB-curated)",
            "source_license_url": "https://www.bindingdb.org/rwd/bind/info.jsp",
            "license_policy": {
                "bindingdb_curated": "CC BY 3.0 (BindingDB-curated)",
                "chembl_derived": "CC BY-SA 3.0 (ChEMBL-derived)",
            },
        },
        "inputs": {
            "mirror_source_manifest": {
                "sha256": _sha256(mirror_path),
                "extracted_sha256": _sha256(extracted),
            }
        },
        "policy": {},
        "counts": {
            "activity_evidence": 1,
            "pre_cutoff": 1,
            "post_cutoff": 0,
        },
        "artifacts": {
            key: {
                "path": str(path),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for key, path in outputs.items()
        },
    }
    (evidence_dir / "manifest.json").write_text(json.dumps(evidence_manifest))


def test_large_parquet_check_uses_schema_and_row_metadata(tmp_path: Path) -> None:
    path = tmp_path / "evidence.parquet"
    pd.DataFrame([{"source_db": "ChEMBL", "activity_id": 1}]).to_parquet(
        path,
        index=False,
    )

    passed = chk_parquet_schema(
        path,
        "evidence",
        {"source_db", "activity_id"},
    )
    failed = chk_parquet_schema(
        path,
        "evidence",
        {"source_db", "source_license"},
    )

    assert passed.ok
    assert "rows=1" in passed.detail
    assert not failed.ok
    assert "source_license" in failed.detail


def test_large_parquet_check_preserves_nested_top_level_column_names(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fingerprints.parquet"
    pd.DataFrame([
        {"molecule_chembl_id": "CHEMBL1", "bitvec": ["0"] * 32},
    ]).to_parquet(path, index=False)

    result = chk_parquet_schema(
        path,
        "fingerprints",
        {"molecule_chembl_id", "bitvec"},
    )

    assert result.ok, result.detail


def test_evidence_manifest_check_fails_closed_on_missing_contract(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema_version": "v1", "source": {}}))

    passed = chk_evidence_manifest(
        path,
        "manifest",
        {"schema_version", "source"},
    )
    failed = chk_evidence_manifest(
        path,
        "manifest",
        {"schema_version", "source", "output_sha256"},
    )

    assert passed.ok
    assert not failed.ok
    assert "output_sha256" in failed.detail


def test_chembl_snapshot_integrity_binds_evidence_and_fingerprints(
    tmp_path: Path,
) -> None:
    root = tmp_path / "chembl37"
    _write_chembl_snapshot(root)

    evidence = chk_chembl_evidence_integrity(root)
    fingerprints = chk_chembl_fingerprint_integrity(root)

    assert evidence.ok, evidence.detail
    assert fingerprints.ok, fingerprints.detail


def test_chembl_snapshot_integrity_rejects_tampered_artifact(tmp_path: Path) -> None:
    root = tmp_path / "chembl37"
    _write_chembl_snapshot(root)
    (root / "fp_morgan2_2048.parquet").write_bytes(b"tampered")

    result = chk_chembl_fingerprint_integrity(root)

    assert not result.ok
    assert "fingerprints=sha256_mismatch" in result.detail


def test_bindingdb_snapshot_integrity_binds_mirror_license_and_evidence(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bindingdb"
    _write_bindingdb_snapshot(root)

    passed = chk_bindingdb_evidence_integrity(root)
    assert passed.ok, passed.detail

    mirror_path = root / "bindingdb_source_manifest.json"
    mirror = json.loads(mirror_path.read_text())
    mirror["source"]["license"] = "CC BY 4.0"
    mirror_path.write_text(json.dumps(mirror))
    failed = chk_bindingdb_evidence_integrity(root)
    assert not failed.ok
    assert "mirror.source.license='CC BY 4.0'" in failed.detail


def test_gtopdb_snapshot_integrity_binds_mirror_and_evidence(tmp_path: Path) -> None:
    root = tmp_path / "gtopdb"
    _write_gtopdb_snapshot(root)

    result = chk_gtopdb_evidence_integrity(root)

    assert result.ok, result.detail


def test_gtopdb_snapshot_integrity_rejects_tampered_artifact(tmp_path: Path) -> None:
    root = tmp_path / "gtopdb"
    _write_gtopdb_snapshot(root)
    (root / "pubmed_esummary.json").write_text("{}\n")

    result = chk_gtopdb_evidence_integrity(root)

    assert not result.ok
    assert "pubmed_esummary.json=sha256_mismatch" in result.detail
    assert "pubmed_json=sha256_mismatch" in result.detail
