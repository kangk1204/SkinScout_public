"""Regression tests for Stage 0 ChEMBL fingerprint generation gates."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import json
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def run_fingerprints(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage0_chembl_fingerprints.py"),
            "--chembl-dir", str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def write_source_manifest(tmp_path: Path) -> None:
    (tmp_path / "source_manifest.json").write_text(json.dumps({
        "schema_version": "chembl_activity_evidence.v1",
        "source": {
            "name": "ChEMBL",
            "release": "37",
            "license": "CC BY-SA 3.0",
        },
    }))


def test_chembl_fingerprints_fail_without_human_activities_and_remove_stale(
    tmp_path: Path,
) -> None:
    out = tmp_path / "fp_morgan2_2048.parquet"
    out.write_text("stale\n")

    res = run_fingerprints(tmp_path)

    assert res.returncode != 0
    assert "Missing" in res.stderr
    assert not out.exists()


def test_chembl_fingerprints_fail_without_valid_smiles_and_remove_stale(
    tmp_path: Path,
) -> None:
    out = tmp_path / "fp_morgan2_2048.parquet"
    out.write_text("stale\n")
    pd.DataFrame([
        {"molecule_chembl_id": "CHEMBL_BAD", "smiles": "not_a_smiles"},
    ]).to_parquet(tmp_path / "human_activities.parquet")

    res = run_fingerprints(tmp_path)

    assert res.returncode != 0
    assert "contains invalid SMILES" in res.stderr
    assert not out.exists()


def test_chembl_fingerprints_reject_partially_invalid_smiles_and_remove_stale(
    tmp_path: Path,
) -> None:
    out = tmp_path / "fp_morgan2_2048.parquet"
    out.write_text("stale\n")
    pd.DataFrame([
        {"molecule_chembl_id": "CHEMBL1", "smiles": "CCO"},
        {"molecule_chembl_id": "CHEMBL_BAD", "smiles": "not_a_smiles"},
    ]).to_parquet(tmp_path / "human_activities.parquet")

    res = run_fingerprints(tmp_path)

    assert res.returncode != 0
    assert "contains invalid SMILES" in res.stderr
    assert "CHEMBL_BAD" in res.stderr
    assert not out.exists()


def test_chembl_fingerprints_reject_conflicting_duplicate_molecule_id(
    tmp_path: Path,
) -> None:
    out = tmp_path / "fp_morgan2_2048.parquet"
    out.write_text("stale\n")
    pd.DataFrame([
        {"molecule_chembl_id": "CHEMBL1", "smiles": "CCO"},
        {"molecule_chembl_id": "CHEMBL1", "smiles": "CCN"},
    ]).to_parquet(tmp_path / "human_activities.parquet")

    res = run_fingerprints(tmp_path)

    assert res.returncode != 0
    assert "duplicate molecule_chembl_id values with conflicting SMILES" in res.stderr
    assert not out.exists()


def test_chembl_fingerprints_reject_blank_molecule_id_and_remove_stale(
    tmp_path: Path,
) -> None:
    out = tmp_path / "fp_morgan2_2048.parquet"
    out.write_text("stale\n")
    pd.DataFrame([
        {"molecule_chembl_id": " ", "smiles": "CCO"},
    ]).to_parquet(tmp_path / "human_activities.parquet")

    res = run_fingerprints(tmp_path)

    assert res.returncode != 0
    assert "contains blank molecule_chembl_id values" in res.stderr
    assert not out.exists()


def test_chembl_fingerprints_write_valid_parquet(tmp_path: Path) -> None:
    write_source_manifest(tmp_path)
    pd.DataFrame([
        {"molecule_chembl_id": "CHEMBL1", "smiles": "CCO"},
        {"molecule_chembl_id": "CHEMBL1", "smiles": "CCO"},
    ]).to_parquet(tmp_path / "human_activities.parquet")

    res = run_fingerprints(tmp_path)

    assert res.returncode == 0, res.stderr
    out = pd.read_parquet(tmp_path / "fp_morgan2_2048.parquet")
    assert list(out.columns) == ["molecule_chembl_id", "smiles", "bitvec"]
    assert out["molecule_chembl_id"].tolist() == ["CHEMBL1"]
    assert len(out.iloc[0]["bitvec"]) == 32
    assert all(isinstance(word, str) for word in out.iloc[0]["bitvec"])
    manifest = json.loads((tmp_path / "fingerprint_manifest.json").read_text())
    assert manifest["schema_version"] == "chembl_fingerprint_snapshot.v1"
    assert manifest["source_snapshot"]["source_release"] == "37"
    assert manifest["input"]["rows"] == 2
    assert manifest["artifact"]["rows"] == 1
    assert len(manifest["artifact"]["sha256"]) == 64


def test_reuse_existing_writes_manifest_without_recomputing(tmp_path: Path) -> None:
    write_source_manifest(tmp_path)
    pd.DataFrame([
        {"molecule_chembl_id": "CHEMBL1", "smiles": "CCO"},
    ]).to_parquet(tmp_path / "human_activities.parquet", index=False)
    pd.DataFrame([
        {"molecule_chembl_id": "CHEMBL1", "smiles": "CCO", "bitvec": ["0"] * 32},
    ]).to_parquet(tmp_path / "fp_morgan2_2048.parquet", index=False)

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage0_chembl_fingerprints.py"),
            "--chembl-dir",
            str(tmp_path),
            "--reuse-existing",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode == 0, res.stderr
    manifest = json.loads((tmp_path / "fingerprint_manifest.json").read_text())
    assert manifest["artifact"]["rows"] == 1
