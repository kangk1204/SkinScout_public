"""Batch Daina scoring loads one reference and emits auditable per-case runs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "stage3_daina_batch.py"


def fingerprint_words(smiles: str) -> list[int]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprint = generator.GetFingerprint(Chem.MolFromSmiles(smiles))
    bits = np.fromiter((int(bit) for bit in fingerprint.ToBitString()), dtype=np.uint8)
    return np.packbits(bits).view(np.uint64).tolist()


def write_sdf(path: Path, smiles: str) -> None:
    writer = Chem.SDWriter(str(path))
    writer.write(Chem.MolFromSmiles(smiles))
    writer.close()


def test_batch_scores_multiple_queries_and_writes_reference_manifest(tmp_path: Path) -> None:
    chembl_fp = tmp_path / "fp_morgan2_2048.parquet"
    pd.DataFrame([
        {"molecule_chembl_id": "CHEMBL1", "bitvec": fingerprint_words("CCO")},
        {"molecule_chembl_id": "CHEMBL2", "bitvec": fingerprint_words("c1ccccc1")},
    ]).to_parquet(chembl_fp)
    pd.DataFrame([
        {"molecule_chembl_id": "CHEMBL1", "uniprot": "P11111"},
        {"molecule_chembl_id": "CHEMBL2", "uniprot": "P22222"},
    ]).to_parquet(tmp_path / "human_activities.parquet")
    sdf_a = tmp_path / "a.sdf"
    sdf_b = tmp_path / "b.sdf"
    write_sdf(sdf_a, "CCO")
    write_sdf(sdf_b, "c1ccccc1")
    batch = tmp_path / "batch.csv"
    pd.DataFrame([
        {
            "case_id": "a",
            "ligand_sdf": sdf_a,
            "out_scores": tmp_path / "a.tsv",
            "out_metadata_json": tmp_path / "a.json",
        },
        {
            "case_id": "b",
            "ligand_sdf": sdf_b,
            "out_scores": tmp_path / "b.tsv",
            "out_metadata_json": tmp_path / "b.json",
        },
    ]).to_csv(batch, index=False)
    summary = tmp_path / "summary.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--batch-csv",
            str(batch),
            "--chembl-fp",
            str(chembl_fp),
            "--out-summary-json",
            str(summary),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert pd.read_csv(tmp_path / "a.tsv", sep="\t").iloc[0]["target_id"] == "P11111"
    assert pd.read_csv(tmp_path / "b.tsv", sep="\t").iloc[0]["target_id"] == "P22222"
    payload = json.loads(summary.read_text())
    assert payload["n_cases"] == 2
    assert payload["evidence_mode"] == "retrieval"
    assert payload["reference"]["activities_sha256"]
    assert json.loads((tmp_path / "a.json").read_text())["case_id"] == "a"


def test_batch_failure_removes_all_staged_and_final_outputs(tmp_path: Path) -> None:
    chembl_fp = tmp_path / "fp_morgan2_2048.parquet"
    pd.DataFrame([
        {"molecule_chembl_id": "CHEMBL1", "bitvec": fingerprint_words("CCO")},
    ]).to_parquet(chembl_fp)
    pd.DataFrame([
        {"molecule_chembl_id": "CHEMBL1", "uniprot": "P11111"},
    ]).to_parquet(tmp_path / "human_activities.parquet")
    batch = tmp_path / "batch.csv"
    pd.DataFrame([
        {
            "case_id": "missing",
            "ligand_sdf": tmp_path / "missing.sdf",
            "out_scores": tmp_path / "scores.tsv",
            "out_metadata_json": tmp_path / "metadata.json",
        },
    ]).to_csv(batch, index=False)
    for path in (tmp_path / "scores.tsv", tmp_path / "metadata.json", tmp_path / "summary.json"):
        path.write_text("stale\n")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--batch-csv",
            str(batch),
            "--chembl-fp",
            str(chembl_fp),
            "--out-summary-json",
            str(tmp_path / "summary.json"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "ligand SDF missing" in result.stderr
    assert not (tmp_path / "scores.tsv").exists()
    assert not (tmp_path / "metadata.json").exists()
    assert not (tmp_path / "summary.json").exists()
