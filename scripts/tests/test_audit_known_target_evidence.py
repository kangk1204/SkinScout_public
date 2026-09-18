"""Regression tests for known-target ChEMBL provenance audit."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402
from rdkit.Chem.MolStandardize import rdMolStandardize  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "eval/audit_known_target_evidence.py"


def load_audit():
    spec = importlib.util.spec_from_file_location("audit_known_target_evidence", AUDIT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parent_inchikey(smiles: str) -> str:
    return Chem.MolToInchiKey(rdMolStandardize.FragmentParent(Chem.MolFromSmiles(smiles)))


def write_panel(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "case_id,smiles,known_targets,known_target_labels",
                "ethanol,CCO.O,P11111;P22222,Target A;Target B",
                "benzene,c1ccccc1,P33333,Target C",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def base_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "standard_inchi_key": parent_inchikey("CCO"),
        "uniprot": "P11111",
        "assay_type": "B",
        "target_type": "SINGLE PROTEIN",
        "assay_confidence_score": 9,
        "assay_relationship_type": "D",
        "standard_relation": "=",
        "standard_type": "IC50",
        "standard_value": 100.0,
        "standard_units": "nM",
        "data_validity_comment": "",
        "potential_duplicate": False,
        "document_type": "Publication",
        "document_year": 2020,
        "assay_variant_id": None,
        "assay_variant_accession": None,
        "assay_variant_mutation": None,
        "assay_source_name": "LITERATURE",
        "document_source_name": "LITERATURE",
        "pchembl_value": 7.0,
        "activity_id": "A1",
        "document_id": "D1",
        "document_chembl_id": "CHEMBL_DOC1",
        "pubmed_id": "12345",
        "doi": "10.1000/ok",
    }
    row.update(overrides)
    return row


def write_evidence(path: Path, rows: list[dict[str, object]]) -> None:
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, row_group_size=1)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(path: Path, evidence: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "chembl_activity_evidence.v1",
                "source": {"name": "ChEMBL", "release": "37", "license": "CC BY-SA 3.0"},
                "output_sha256": {"activity_evidence.parquet": file_sha256(evidence)},
                "row_counts": {
                    "activity_evidence": pq.ParquetFile(evidence).metadata.num_rows
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def run_audit(tmp_path: Path, rows: list[dict[str, object]]) -> subprocess.CompletedProcess[str]:
    panel = tmp_path / "panel.csv"
    evidence = tmp_path / "activity_evidence.parquet"
    manifest = tmp_path / "source_manifest.json"
    write_panel(panel)
    write_evidence(evidence, rows)
    write_manifest(manifest, evidence)
    return subprocess.run(
        [
            sys.executable,
            str(AUDIT),
            "--panel-csv",
            str(panel),
            "--evidence-parquet",
            str(evidence),
            "--source-manifest",
            str(manifest),
            "--out-csv",
            str(tmp_path / "pairs.csv"),
            "--out-manifest",
            str(tmp_path / "audit_manifest.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def read_pairs(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_identity_normalization_uses_parent_inchikey_connectivity_block() -> None:
    audit = load_audit()
    parent_key, connectivity = audit.parent_inchikey_connectivity("CCO.O")
    assert parent_key == parent_inchikey("CCO")
    assert connectivity == parent_key.split("-", 1)[0]


def test_audit_preserves_supported_and_unsupported_pairs(tmp_path: Path) -> None:
    res = run_audit(
        tmp_path,
        [
            base_row(activity_id="A1", pchembl_value=7.0, document_year=2020),
            base_row(activity_id="A2", pchembl_value=8.2, document_year=2021, doi="10.1000/better"),
        ],
    )
    assert res.returncode == 0, res.stderr
    rows = read_pairs(tmp_path / "pairs.csv")
    by_pair = {(row["case_id"], row["uniprot"]): row for row in rows}
    assert by_pair[("ethanol", "P11111")]["support_status"] == "supported"
    assert by_pair[("ethanol", "P11111")]["support_count"] == "2"
    assert by_pair[("ethanol", "P11111")]["best_pchembl"] == "8.2"
    assert by_pair[("ethanol", "P11111")]["earliest_year"] == "2020"
    assert by_pair[("ethanol", "P11111")]["latest_year"] == "2021"
    assert by_pair[("ethanol", "P22222")]["support_status"] == "unsupported"
    assert by_pair[("benzene", "P33333")]["support_status"] == "unsupported"


@pytest.mark.parametrize(
    "override",
    [
        {"assay_type": "F"},
        {"target_type": "PROTEIN FAMILY"},
        {"assay_confidence_score": 8},
        {"assay_relationship_type": "H"},
        {"standard_relation": ">"},
        {"standard_type": "Potency"},
        {"standard_units": "uM"},
        {"standard_value": 0.0},
        {"pchembl_value": 4.99},
        {"data_validity_comment": "Outside typical range"},
        {"potential_duplicate": True},
        {"document_type": "Patent"},
        {"document_year": None},
        {"assay_variant_id": 12},
        {"assay_variant_id": -1},
        {"assay_variant_accession": "P11111-2"},
        {"assay_variant_mutation": "A1V"},
        {"document_source_name": "BindingDB"},
        {"assay_source_name": "BindingDB"},
    ],
)
def test_claim_grade_filter_rejections_leave_pair_unsupported(
    tmp_path: Path, override: dict[str, object]
) -> None:
    res = run_audit(tmp_path, [base_row(**override)])
    assert res.returncode == 0, res.stderr
    rows = read_pairs(tmp_path / "pairs.csv")
    row = next(item for item in rows if item["case_id"] == "ethanol" and item["uniprot"] == "P11111")
    assert row["support_status"] == "unsupported"
    assert row["support_count"] == "0"
    assert row["sample_activity_ids"] == ""


def test_manifest_records_hashes_release_license_and_separation_label(tmp_path: Path) -> None:
    res = run_audit(tmp_path, [base_row()])
    assert res.returncode == 0, res.stderr
    manifest = json.loads((tmp_path / "audit_manifest.json").read_text())
    assert manifest["audit_label"] == "retrospective_ground_truth_provenance_audit"
    assert manifest["separation_label"] == "audit_only_never_ranking_input"
    assert "never ranking input" in manifest["separation_policy"]
    assert manifest["source"]["release"] == "37"
    assert manifest["source"]["license"] == "CC BY-SA 3.0"
    assert len(manifest["inputs"]["panel"]["sha256"]) == 64
    assert len(manifest["inputs"]["evidence"]["sha256"]) == 64
    assert len(manifest["inputs"]["source_manifest"]["sha256"]) == 64
    assert len(manifest["outputs"]["pair_csv"]["sha256"]) == 64
    assert manifest["pair_counts"] == {"total": 3, "supported": 1, "unsupported": 2}
    rows = read_pairs(tmp_path / "pairs.csv")
    assert all(row["separation_label"] == "audit_only_never_ranking_input" for row in rows)


def test_fail_closed_on_bad_evidence_schema_and_removes_stale_outputs(tmp_path: Path) -> None:
    panel = tmp_path / "panel.csv"
    evidence = tmp_path / "bad.parquet"
    manifest = tmp_path / "source_manifest.json"
    out_csv = tmp_path / "pairs.csv"
    out_manifest = tmp_path / "audit_manifest.json"
    write_panel(panel)
    pq.write_table(pa.Table.from_pylist([{"standard_inchi_key": parent_inchikey("CCO")}]), evidence)
    write_manifest(manifest, evidence)
    out_csv.write_text("stale\n", encoding="utf-8")
    out_manifest.write_text("stale\n", encoding="utf-8")

    res = subprocess.run(
        [
            sys.executable,
            str(AUDIT),
            "--panel-csv",
            str(panel),
            "--evidence-parquet",
            str(evidence),
            "--source-manifest",
            str(manifest),
            "--out-csv",
            str(out_csv),
            "--out-manifest",
            str(out_manifest),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "missing required columns" in res.stderr
    assert not out_csv.exists()
    assert not out_manifest.exists()


def test_fail_closed_when_manifest_does_not_bind_evidence(tmp_path: Path) -> None:
    panel = tmp_path / "panel.csv"
    evidence = tmp_path / "activity_evidence.parquet"
    manifest = tmp_path / "source_manifest.json"
    write_panel(panel)
    write_evidence(evidence, [base_row()])
    write_manifest(manifest, evidence)
    payload = json.loads(manifest.read_text())
    payload["output_sha256"]["activity_evidence.parquet"] = "0" * 64
    manifest.write_text(json.dumps(payload) + "\n")

    res = subprocess.run(
        [
            sys.executable,
            str(AUDIT),
            "--panel-csv",
            str(panel),
            "--evidence-parquet",
            str(evidence),
            "--source-manifest",
            str(manifest),
            "--out-csv",
            str(tmp_path / "pairs.csv"),
            "--out-manifest",
            str(tmp_path / "audit_manifest.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "do not bind" in res.stderr
    assert not (tmp_path / "pairs.csv").exists()
