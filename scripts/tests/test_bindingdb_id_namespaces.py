"""F32: compound identity and measurement identity are different namespaces.

BindingDB's Reactant_set_id is a measurement/assay-level id, but the builder's
legacy ``ligand_id`` preferred it, the merge keyed ``molecule_chembl_id`` on it,
and one compound measured many times became many "ligands" (measured: 2,909,127
rows carried 2,685,612 ligand ids but only 1,145,044 InChIKeys).

These tests pin the split: ``source_compound_id`` keys the compound,
``measurement_id`` keys the measurement, ``structure_id`` carries the
standardised structure, and both output and manifest say which namespace each
id came from.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "scripts" / "build_bindingdb_temporal_evidence.py"
MERGE = ROOT / "scripts" / "merge_runtime_activity_evidence.py"
pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")


def run_script(script: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script)] + args,
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )


def _build_args(tsv: Path, out_dir: Path) -> list[str]:
    return [
        "--bindingdb-tsv",
        str(tsv),
        "--out-dir",
        str(out_dir),
        "--cutoff-date",
        "2020-01-01",
        "--source-release",
        "2026-08",
        "--source-license-url",
        "https://bindingdb.org/",
    ]


def test_builder_keeps_one_compound_for_different_measurements(
    tmp_path: Path,
) -> None:
    header = (
        "BindingDB Reactant_set_id\tBindingDB MonomerID\tLigand SMILES\t"
        "Ligand InChI Key\t"
        "Target Source Organism According to Curator or DataSource\t"
        "UniProt (SwissProt) Primary ID of Target Chain 1\t"
        "Ki (nM)\tPublication Date\tArticle DOI\tBindingDB PMID\tSource License\n"
    )
    tsv = tmp_path / "BindingDB_All.tsv"
    tsv.write_text(
        header
        + "RS1\tBDBM1\tCCO\tLFQSCWFLJHTTHZ-UHFFFAOYSA-N\tHomo sapiens\t"
        "P11111\t10\t2019-01-01\t10.1000/first\t1\tBindingDB terms\n"
        + "RS2\tBDBM1\tCCO\tLFQSCWFLJHTTHZ-UHFFFAOYSA-N\tHomo sapiens\t"
        "P22222\t20\t2019-01-01\t10.1000/second\t2\tBindingDB terms\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"

    res = run_script(BUILD, _build_args(tsv, out_dir))

    assert res.returncode == 0, res.stderr
    activity = pd.read_parquet(out_dir / "activity_evidence.parquet")
    assert len(activity) == 2
    # The compound collapses; the measurements stay distinct.
    assert set(activity["source_compound_id"]) == {"BDBM1"}
    assert set(activity["measurement_id"]) == {"RS1", "RS2"}
    assert set(activity["structure_id"]) == {
        "inchikey:LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
    }
    # The legacy column keeps its old, measurement-first meaning.
    assert set(activity["ligand_id"]) == {"RS1", "RS2"}
    assert set(activity["source_compound_id_basis"]) == {"BindingDB MonomerID"}
    assert set(activity["measurement_id_basis"]) == {"BindingDB Reactant_set_id"}

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    counts = manifest["counts"]
    assert counts["source_compound_id_basis_counts"] == {"BindingDB MonomerID": 2}
    assert counts["measurement_id_basis_counts"] == {
        "BindingDB Reactant_set_id": 2
    }
    assert counts["source_compound_id_differs_from_measurement_id_rows"] == 2
    namespaces = manifest["id_namespaces"]
    assert namespaces["ligand_id"]["may_be_measurement_id"] is True
    assert namespaces["source_compound_id"]["is_measurement_id"] is False
    assert namespaces["measurement_id"]["is_compound_id"] is False
    assert "inchikey:" in namespaces["structure_id"]["basis"]


def test_builder_falls_back_and_records_the_basis(tmp_path: Path) -> None:
    header = (
        "BindingDB Reactant_set_id\tLigand SMILES\tLigand InChI Key\t"
        "Target Source Organism According to Curator or DataSource\t"
        "UniProt (SwissProt) Primary ID of Target Chain 1\t"
        "IC50 (nM)\tPublication Date\tBindingDB PMID\tSource License\n"
    )
    tsv = tmp_path / "BindingDB_All.tsv"
    tsv.write_text(
        header
        + "RS9\tCCO\tLFQSCWFLJHTTHZ-UHFFFAOYSA-N\tHomo sapiens\t"
        "P11111\t10\t2019-01-01\t1\tBindingDB terms\n"
        + "\tCCO\tLFQSCWFLJHTTHZ-UHFFFAOYSA-N\tHomo sapiens\t"
        "P11111\t20\t2019-01-01\t1\tBindingDB terms\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"

    res = run_script(BUILD, _build_args(tsv, out_dir))

    assert res.returncode == 0, res.stderr
    activity = pd.read_parquet(out_dir / "activity_evidence.parquet")
    assert len(activity) == 2
    by_basis = {
        row["source_compound_id_basis"]: row for _, row in activity.iterrows()
    }
    assert by_basis["legacy_ligand_id_fallback"]["source_compound_id"] == "RS9"
    assert (
        by_basis["standard_inchikey_fallback"]["source_compound_id"]
        == "inchikey:LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
    )

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["source_compound_id_basis_counts"] == {
        "legacy_ligand_id_fallback": 1,
        "standard_inchikey_fallback": 1,
    }


def _chembl_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "chembl"
    directory.mkdir()
    base = pd.DataFrame(
        [
            {
                "molecule_chembl_id": "CHEMBL1",
                "uniprot": "P11111",
                "smiles": "CCO",
                "standard_inchi_key": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
                "act_type": "IC50",
                "act_value": 100.0,
                "act_units": "nM",
                "pchembl": 7.0,
                "pubmed_id": "123",
            }
        ]
    )
    base.to_parquet(directory / "human_activities.parquet", index=False)
    pd.DataFrame(
        [{"molecule_chembl_id": "CHEMBL1", "smiles": "CCO", "bitvec": ["0"] * 32}]
    ).to_parquet(directory / "fp_morgan2_2048.parquet", index=False)
    return directory


def _row(**overrides) -> dict:
    row = {
        "ligand_id": "RS1",
        "source_compound_id": "MONO1",
        "source_compound_id_basis": "BindingDB MonomerID",
        "measurement_id": "RS1",
        "measurement_id_basis": "BindingDB Reactant_set_id",
        "structure_id": "inchikey:XLYOFNOQVPJJNP-UHFFFAOYSA-N",
        "ligand_inchikey": "XLYOFNOQVPJJNP-UHFFFAOYSA-N",
        "ligand_smiles": "CCCO",
        "uniprot": "Q22222",
        "affinity_type": "IC50",
        "affinity_value": "250.0",
        "affinity_unit": "nM",
        "relation": "=",
        "censor": "False",
        "source_pmid": "123",
        "evidence_date": "2020-01-01",
        "source_db": "BindingDB",
        "source_release": "2026-08",
        "source_license": "CC BY 4.0 (BindingDB-curated)",
        "chembl_derived_license_flag": False,
    }
    row.update(overrides)
    return row


def _run_merge(tmp_path: Path, chembl: Path, bindingdb: Path):
    out = tmp_path / "merged"
    result = subprocess.run(
        [
            sys.executable,
            str(MERGE),
            "--chembl-dir",
            str(chembl),
            "--bindingdb",
            str(bindingdb),
            "--gtopdb",
            str(tmp_path / "absent.parquet"),
            "--out-dir",
            str(out),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result, out


def test_merge_keys_one_compound_for_repeated_measurements(tmp_path: Path) -> None:
    chembl = _chembl_dir(tmp_path)
    bindingdb = tmp_path / "bindingdb.parquet"
    pd.DataFrame(
        [
            _row(),
            _row(
                ligand_id="RS2",
                measurement_id="RS2",
                uniprot="Q33333",
                affinity_value="300.0",
                source_pmid="456",
            ),
        ]
    ).to_parquet(bindingdb, index=False)

    result, out = _run_merge(tmp_path, chembl, bindingdb)

    assert result.returncode == 0, result.stderr
    merged = pd.read_parquet(out / "human_activities.parquet")
    added = merged[merged["source_db"] == "BindingDB"]
    assert len(added) == 2, "both measurements must survive"
    assert set(added["molecule_chembl_id"]) == {"BDB:MONO1"}, (
        "different measurements of one compound must share one run-path key"
    )

    fingerprints = pd.read_parquet(out / "fp_morgan2_2048.parquet")
    assert (fingerprints["molecule_chembl_id"] == "BDB:MONO1").sum() == 1

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    counts = manifest["counts"]
    assert counts["ligand_keys_by_identity_basis"] == {"source_compound_id": 2}
    assert counts["distinct_molecule_keys_added"] == 1
    assert counts["distinct_measurement_ids_added"] == 2
    assert counts["distinct_structure_ids_added"] == 1
    assert manifest["inputs"]["bindingdb"]["id_namespaces_present"] == {
        "source_compound_id": True,
        "measurement_id": True,
        "structure_id": True,
    }
    assert "source_compound_id" in manifest["policy"]["ligand_key_namespaces"]["BDB"]


def test_merge_records_when_it_had_to_use_the_legacy_measurement_id(
    tmp_path: Path,
) -> None:
    chembl = _chembl_dir(tmp_path)
    legacy = {
        key: value
        for key, value in _row().items()
        if key
        not in {
            "source_compound_id",
            "source_compound_id_basis",
            "measurement_id",
            "measurement_id_basis",
            "structure_id",
        }
    }
    bindingdb = tmp_path / "bindingdb.parquet"
    pd.DataFrame([legacy]).to_parquet(bindingdb, index=False)

    result, out = _run_merge(tmp_path, chembl, bindingdb)

    assert result.returncode == 0, result.stderr
    merged = pd.read_parquet(out / "human_activities.parquet")
    assert set(merged["molecule_chembl_id"]) == {"CHEMBL1", "BDB:RS1"}
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["ligand_keys_by_identity_basis"] == {
        "legacy_ligand_id": 1
    }
    assert manifest["inputs"]["bindingdb"]["id_namespaces_present"] == {
        "source_compound_id": False,
        "measurement_id": False,
        "structure_id": False,
    }


def test_the_manifest_hash_covers_the_identity_columns(tmp_path: Path) -> None:
    """The written table must carry the columns the manifest says it does."""
    chembl = _chembl_dir(tmp_path)
    bindingdb = tmp_path / "bindingdb.parquet"
    pd.DataFrame([_row()]).to_parquet(bindingdb, index=False)

    result, out = _run_merge(tmp_path, chembl, bindingdb)

    assert result.returncode == 0, result.stderr
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["inputs"]["bindingdb"]["sha256"] == hashlib.sha256(
        bindingdb.read_bytes()
    ).hexdigest()
