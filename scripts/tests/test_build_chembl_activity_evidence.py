"""Regression tests for provenance-preserving ChEMBL activity extraction."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BUILDER = ROOT / "scripts/build_chembl_activity_evidence.py"


def run_builder(db: Path, out_dir: Path, release: str = "37") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--db",
            str(db),
            "--out-dir",
            str(out_dir),
            "--release",
            release,
            "--chunksize",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def load_builder():
    spec = importlib.util.spec_from_file_location("build_chembl_activity_evidence", BUILDER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def create_fixture_db(path: Path) -> None:
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE target_dictionary (
            tid INTEGER PRIMARY KEY,
            chembl_id TEXT,
            organism TEXT,
            target_type TEXT,
            pref_name TEXT
        );
        CREATE TABLE target_components (tid INTEGER, component_id INTEGER);
        CREATE TABLE component_sequences (component_id INTEGER PRIMARY KEY, accession TEXT);
        CREATE TABLE assays (
            assay_id INTEGER PRIMARY KEY,
            tid INTEGER,
            chembl_id TEXT,
            assay_type TEXT,
            assay_test_type TEXT,
            assay_category TEXT,
            confidence_score INTEGER,
            relationship_type TEXT,
            src_id INTEGER,
            variant_id INTEGER
        );
        CREATE TABLE activities (
            activity_id INTEGER PRIMARY KEY,
            assay_id INTEGER,
            doc_id INTEGER,
            molregno INTEGER,
            standard_relation TEXT,
            standard_type TEXT,
            standard_value REAL,
            standard_units TEXT,
            pchembl_value REAL,
            data_validity_comment TEXT,
            potential_duplicate INTEGER,
            activity_comment TEXT,
            action_type TEXT
        );
        CREATE TABLE molecule_dictionary (molregno INTEGER PRIMARY KEY, chembl_id TEXT);
        CREATE TABLE compound_structures (
            molregno INTEGER PRIMARY KEY,
            canonical_smiles TEXT,
            standard_inchi_key TEXT
        );
        CREATE TABLE docs (
            doc_id INTEGER PRIMARY KEY,
            chembl_id TEXT,
            year INTEGER,
            pubmed_id TEXT,
            doi TEXT,
            patent_id TEXT,
            doc_type TEXT,
            src_id INTEGER,
            chembl_release_id INTEGER
        );
        CREATE TABLE source (
            src_id INTEGER PRIMARY KEY,
            src_short_name TEXT
        );
        CREATE TABLE variant_sequences (
            variant_id INTEGER PRIMARY KEY,
            mutation TEXT,
            accession TEXT
        );
        """
    )
    con.execute(
        "INSERT INTO target_dictionary VALUES (1, 'CHEMBL_T1', 'Homo sapiens', 'SINGLE PROTEIN', 'Target 1')"
    )
    con.execute("INSERT INTO target_components VALUES (1, 10)")
    con.execute("INSERT INTO component_sequences VALUES (10, 'P12345')")
    con.execute(
        "INSERT INTO assays VALUES (100, 1, 'CHEMBL_A100', 'B', 'Functional', 'confirmatory', 9, 'D', 1, NULL)"
    )
    con.execute("INSERT INTO molecule_dictionary VALUES (200, 'CHEMBL_M200')")
    con.execute(
        "INSERT INTO compound_structures VALUES (200, 'CCO', 'LFQSCWFLJHTTHZ-UHFFFAOYSA-N')"
    )
    con.execute(
        "INSERT INTO docs VALUES (300, 'CHEMBL_D300', 2020, '123456', '10.1/example', 'US123', 'Publication', 1, 37)"
    )
    con.execute("INSERT INTO source VALUES (1, 'LITERATURE')")
    con.execute(
        """
        INSERT INTO activities VALUES (
            400, 100, 300, 200, '<', 'IC50', 50.0, 'nM', 7.3,
            'Outside typical range', 1, 'questionable but retained', 'INHIBITOR'
        )
        """
    )
    con.execute(
        """
        INSERT INTO activities VALUES (
            401, 100, 300, 200, '=', 'IC50', 100.0, 'nM', NULL,
            'filtered for compatibility', 0, 'no pchembl', 'INHIBITOR'
        )
        """
    )
    con.commit()
    con.close()


def test_builder_writes_legacy_columns_and_rich_provenance(tmp_path: Path) -> None:
    db = tmp_path / "chembl_37.db"
    create_fixture_db(db)

    res = run_builder(db, tmp_path, "37")

    assert res.returncode == 0, res.stderr
    human = pd.read_parquet(tmp_path / "human_activities.parquet")
    evidence = pd.read_parquet(tmp_path / "activity_evidence.parquet")
    builder = load_builder()
    assert list(human.columns)[:8] == builder.LEGACY_COLUMNS
    pd.testing.assert_frame_equal(human, evidence)
    row = human.iloc[0].to_dict()
    assert row["activity_id"] == 400
    assert row["assay_id"] == 100
    assert row["assay_chembl_id"] == "CHEMBL_A100"
    assert row["assay_type"] == "B"
    assert row["assay_test_type"] == "Functional"
    assert row["assay_category"] == "confirmatory"
    assert row["assay_confidence_score"] == 9
    assert row["assay_relationship_type"] == "D"
    assert row["assay_source_id"] == 1
    assert row["assay_source_name"] == "LITERATURE"
    assert pd.isna(row["assay_variant_id"])
    assert row["target_chembl_id"] == "CHEMBL_T1"
    assert row["target_organism"] == "Homo sapiens"
    assert row["target_type"] == "SINGLE PROTEIN"
    assert row["uniprot"] == "P12345"
    assert row["molecule_chembl_id"] == "CHEMBL_M200"
    assert row["smiles"] == "CCO"
    assert row["standard_inchi_key"] == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
    assert row["document_id"] == 300
    assert row["document_year"] == 2020
    assert row["pubmed_id"] == "123456"
    assert row["doi"] == "10.1/example"
    assert row["patent_id"] == "US123"
    assert row["document_type"] == "Publication"
    assert row["document_source_id"] == 1
    assert row["document_source_name"] == "LITERATURE"
    assert row["document_chembl_release_id"] == 37
    assert row["standard_relation"] == "<"
    assert row["data_validity_comment"] == "Outside typical range"
    assert row["potential_duplicate"] is True
    assert row["action_type"] == "INHIBITOR"
    assert row["source_name"] == "ChEMBL"
    assert row["source_db"] == "ChEMBL"
    assert row["source_release"] == "37"
    assert row["source_version"] == "37"
    assert row["source_license"] == "CC BY-SA 3.0"

    manifest = json.loads((tmp_path / "source_manifest.json").read_text())
    assert manifest["schema_version"] == builder.SCHEMA_VERSION
    assert manifest["source"]["source_db"].endswith("chembl_37.db")
    assert manifest["source"]["release"] == "37"
    assert manifest["source"]["license"] == "CC BY-SA 3.0"
    assert manifest["source"]["redistribution"] == "allowed"
    assert manifest["source"]["source_db_bytes"] == db.stat().st_size
    assert manifest["row_counts"]["human_activities"] == 1
    assert manifest["row_counts"]["activity_evidence"] == 1
    assert "activities.data_validity_comment" in manifest["extraction_policy"]["preserved_not_filtered"]
    assert set(manifest["output_sha256"]) == {
        "human_activities.parquet",
        "activity_evidence.parquet",
    }
    assert "stable self-hashing is impossible" in manifest["manifest_sha256_policy"]


def test_builder_fails_closed_on_invalid_schema_and_removes_stale_outputs(tmp_path: Path) -> None:
    db = tmp_path / "bad.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE activities (activity_id INTEGER)")
    con.commit()
    con.close()
    for name in ["human_activities.parquet", "activity_evidence.parquet", "source_manifest.json"]:
        (tmp_path / name).write_text("stale\n")

    res = run_builder(db, tmp_path)

    assert res.returncode != 0
    assert "Invalid ChEMBL SQLite schema" in res.stderr
    assert "missing required table assays" in res.stderr
    assert not (tmp_path / "human_activities.parquet").exists()
    assert not (tmp_path / "activity_evidence.parquet").exists()
    assert not (tmp_path / "source_manifest.json").exists()


def test_builder_atomic_failure_does_not_leave_partial_or_stale_outputs(tmp_path: Path) -> None:
    db = tmp_path / "chembl_37.db"
    create_fixture_db(db)
    for name in ["human_activities.parquet", "activity_evidence.parquet"]:
        (tmp_path / name).write_text("stale\n")

    res = run_builder(db, tmp_path, "37")

    assert res.returncode == 0, res.stderr
    assert not list(tmp_path.glob(".*.tmp"))
    assert pd.read_parquet(tmp_path / "human_activities.parquet")["activity_id"].tolist() == [400]


def test_builder_manifest_records_release_license_and_archive_sha(tmp_path: Path) -> None:
    db = tmp_path / "chembl_99.db"
    archive = tmp_path / "chembl_99_sqlite.tar.gz"
    archive.write_bytes(b"archive")
    create_fixture_db(db)
    res = subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--db",
            str(db),
            "--out-dir",
            str(tmp_path),
            "--release",
            "99",
            "--source-archive",
            str(archive),
            "--chunksize",
            "2",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode == 0, res.stderr
    manifest = json.loads((tmp_path / "source_manifest.json").read_text())
    assert manifest["source"]["release"] == "99"
    assert manifest["source"]["license"] == "CC BY-SA 3.0"
    assert manifest["source"]["source_archive"].endswith("chembl_99_sqlite.tar.gz")
    assert manifest["source"]["source_archive_sha256"] == builder_sha256(archive)
    assert manifest["input_sha256"]["sqlite_tarball"] == builder_sha256(archive)


def builder_sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
