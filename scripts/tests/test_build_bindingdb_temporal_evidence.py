from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "build_bindingdb_temporal_evidence.py"
pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")


def run_script(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT)] + args,
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )


def _write_tsv(path: Path, rows: list[str], *, header: str | None = None) -> Path:
    header = header or (
        "Ligand SMILES\tLigand InChI Key\tBindingDB Reactant_set_id\t"
        "Target Source Organism According to Curator or DataSource\t"
        "UniProt (SwissProt) Primary ID of Target Chain 1\t"
        "Ki (nM)\tIC50 (nM)\tPublication Date\tBindingDB Curation Date\t"
        "BindingDB Entry DOI\tBindingDB PMID\tSource License\n"
    )
    path.write_text(header + "".join(rows), encoding="utf-8")
    return path


def _base_args(tsv: Path, out_dir: Path) -> list[str]:
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


def _write_source_manifest(
    path: Path,
    tsv: Path,
    *,
    license_text: str = "CC BY 3.0",
) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": {
                    "name": "BindingDB",
                    "license": license_text,
                    "license_url": "https://bindingdb.org/",
                },
                "release": "2026-08",
                "extracted": {
                    "path": tsv.name,
                    "sha256": hashlib.sha256(tsv.read_bytes()).hexdigest(),
                },
            }
        )
        + "\n"
    )
    return path


def test_normalizer_emits_activity_and_temporal_splits(tmp_path: Path) -> None:
    tsv = _write_tsv(
        tmp_path / "BindingDB_All.tsv",
        [
            "CCO\tBSYNRYMUTXBXSQ-UHFFFAOYSA-N\tBDBM1\tHomo sapiens\tP11111\t<10\t50\t2019-12-31\t2024-01-01\t10.1000/pre\t123\tBindingDB terms\n",
            "CCN\tBSYNRYMUTXBXSQ-UHFFFAOYSA-M\tCHEMBL42\tHomo sapiens\tP22222\t\t100\t2021-02-03\t2024-01-01\t10.1000/post\t456\tChEMBL-derived CC BY-SA\n",
            "CCC\t\tBDBM3\tMus musculus\tP33333\t1\t\t2019-01-01\t2024-01-01\t10.1000/mouse\t789\tBindingDB terms\n",
        ],
    )
    out_dir = tmp_path / "out"

    res = run_script(_base_args(tsv, out_dir))

    assert res.returncode == 0, res.stderr
    activity = pd.read_parquet(out_dir / "activity_evidence.parquet")
    pre = pd.read_parquet(out_dir / "pre_cutoff.parquet")
    post = pd.read_parquet(out_dir / "post_cutoff.parquet")
    assert activity.shape[0] == 3
    assert activity["input_row_number"].tolist() == [1, 1, 2]
    assert activity["target_chain_count"].tolist() == [1, 1, 1]
    assert activity["single_chain_target"].tolist() == [True, True, True]
    assert activity["evidence_date_source"].tolist() == [
        "publication",
        "publication",
        "publication",
    ]
    assert pre.shape[0] == 2
    assert post.shape[0] == 1
    assert set(activity["affinity_type"]) == {"Ki", "IC50"}
    assert activity.loc[activity["affinity_type"] == "Ki", "relation"].iloc[0] == "<"
    assert bool(activity.loc[activity["affinity_type"] == "Ki", "censor"].iloc[0])
    assert "P33333" not in set(activity["uniprot"])
    chembl = activity.loc[activity["ligand_id"] == "CHEMBL42"].iloc[0]
    assert bool(chembl["chembl_derived_license_flag"])
    curated = activity.loc[activity["ligand_id"] == "BDBM1"].iloc[0]
    assert curated["source_release"] == "2026-08"
    assert curated["source_license"] == "CC BY 3.0 (BindingDB-curated)"
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["policy"]["date_source"].endswith("file mtime never used")
    assert "disk-backed SQLite" in manifest["policy"]["duplicate_policy"]
    assert manifest["counts"]["skipped_non_human"] == 1
    assert manifest["artifacts"]["activity_evidence"]["sha256"]


def test_normalizer_binds_and_validates_mirror_source_manifest(tmp_path: Path) -> None:
    tsv = _write_tsv(
        tmp_path / "BindingDB_All.tsv",
        [
            "CCO\tBSYNRYMUTXBXSQ-UHFFFAOYSA-N\tBDBM1\tHomo sapiens\tP11111\t"
            "10\t\t2019-12-31\t2024-01-01\t10.1000/pre\t123\tBindingDB terms\n"
        ],
    )
    source_manifest = _write_source_manifest(tmp_path / "bindingdb_source_manifest.json", tsv)
    out_dir = tmp_path / "out"

    res = run_script(
        _base_args(tsv, out_dir) + ["--source-manifest", str(source_manifest)]
    )

    assert res.returncode == 0, res.stderr
    manifest = json.loads((out_dir / "manifest.json").read_text())
    bound = manifest["inputs"]["mirror_source_manifest"]
    assert bound["sha256"] == hashlib.sha256(source_manifest.read_bytes()).hexdigest()
    assert bound["extracted_sha256"] == hashlib.sha256(tsv.read_bytes()).hexdigest()

    _write_source_manifest(source_manifest, tsv, license_text="CC BY 4.0")
    rejected = run_script(
        _base_args(tsv, tmp_path / "bad")
        + ["--source-manifest", str(source_manifest)]
    )
    assert rejected.returncode != 0
    assert "source.license='CC BY 4.0'" in rejected.stderr


def test_normalizer_keeps_mixed_assay_types_separate_and_marks_duplicates(
    tmp_path: Path,
) -> None:
    duplicate = (
        "CCO\tBSYNRYMUTXBXSQ-UHFFFAOYSA-N\tBDBM1\tHomo sapiens\tP11111\t10\t50\t"
        "2019\t2024-01-01\t10.1000/dup\t123\tBindingDB terms\n"
    )
    tsv = _write_tsv(tmp_path / "BindingDB_All.tsv", [duplicate, duplicate])
    out_dir = tmp_path / "out"

    res = run_script(_base_args(tsv, out_dir))

    assert res.returncode == 0, res.stderr
    activity = pd.read_parquet(out_dir / "activity_evidence.parquet")
    assert activity.shape[0] == 4
    assert activity.groupby("affinity_type").size().to_dict() == {"IC50": 2, "Ki": 2}
    assert activity["duplicate_evidence"].all()
    assert set(activity["duplicate_ordinal"]) == {1, 2}


def test_normalizer_reads_bindingdb_zip(tmp_path: Path) -> None:
    tsv = _write_tsv(
        tmp_path / "BindingDB_All.tsv",
        [
            "CCO\tBSYNRYMUTXBXSQ-UHFFFAOYSA-N\tBDBM1\tHomo sapiens\tP11111\t10\t\t2019-01-01\t2024-01-01\t10.1000/zip\t123\tBindingDB terms\n",
        ],
    )
    zip_path = tmp_path / "bindingdb.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(tsv, "BindingDB_All.tsv")
    out_dir = tmp_path / "out"

    res = run_script(_base_args(zip_path, out_dir))

    assert res.returncode == 0, res.stderr
    activity = pd.read_parquet(out_dir / "activity_evidence.parquet")
    assert activity.shape[0] == 1
    assert activity.loc[0, "source_release"] == "2026-08"


def test_normalizer_accepts_current_bindingdb_header_names_and_conservative_year(
    tmp_path: Path,
) -> None:
    header = (
        "BindingDB Reactant_set_id\tLigand SMILES\tLigand InChI Key\t"
        "Target Source Organism According to Curator or DataSource\t"
        "Ki (nM)\tCuration/DataSource\tArticle DOI\tPMID\t"
        "Date of publication\tDate in BindingDB\t"
        "UniProt (SwissProt) Primary ID of Target Chain 1\n"
    )
    tsv = _write_tsv(
        tmp_path / "BindingDB_All.tsv",
        [
            "BDBM1\tCCO\tLFQSCWFLJHTTHZ-UHFFFAOYSA-N\tHomo sapiens\t"
            "10\tChEMBL 37\t10.1000/current\t123\t2020\t10/18/2021\tP11111-2\n"
        ],
        header=header,
    )
    out_dir = tmp_path / "out"

    res = run_script(_base_args(tsv, out_dir))

    assert res.returncode == 0, res.stderr
    activity = pd.read_parquet(out_dir / "activity_evidence.parquet")
    assert activity.loc[0, "publication_date"] == "2020-12-31"
    assert activity.loc[0, "source_origin"] == "ChEMBL 37"
    assert activity.loc[0, "uniprot"] == "P11111"
    assert bool(activity.loc[0, "chembl_derived_license_flag"])
    assert "CC BY-SA 3.0" in activity.loc[0, "source_license"]


def test_normalizer_records_multichain_target_and_date_source(
    tmp_path: Path,
) -> None:
    header = (
        "Ligand SMILES\tLigand InChI Key\tBindingDB Reactant_set_id\t"
        "Target Source Organism According to Curator or DataSource\t"
        "Number of Protein Chains in Target (>1 implies a multichain complex)\t"
        "UniProt (SwissProt) Primary ID of Target Chain 1\t"
        "UniProt (SwissProt) Secondary ID(s) of Target Chain 1\t"
        "UniProt (SwissProt) Primary ID of Target Chain 2\t"
        "Ki (nM)\tPublication Date\tBindingDB Curation Date\t"
        "Article DOI\tBindingDB Entry DOI\tBindingDB PMID\tSource License\n"
    )
    tsv = _write_tsv(
        tmp_path / "BindingDB_All.tsv",
        [
            "CCO\tBSYNRYMUTXBXSQ-UHFFFAOYSA-N\tBDBM1\tHomo sapiens\t"
            "2\tP11111\tQ99999\tP22222-2\t10\t2019-06\t2024-01-01\t"
            "10.1000/article\t10.7270/entry\t123\tBindingDB terms\n",
            "CCN\tBSYNRYMUTXBXSQ-UHFFFAOYSA-M\tBDBM2\tHomo sapiens\t"
            "1\tP33333\tQ88888\t\t20\t\t2021-02-03\t"
            "\t10.7270/curation-entry\t456\tBindingDB terms\n",
        ],
        header=header,
    )
    out_dir = tmp_path / "out"

    res = run_script(_base_args(tsv, out_dir))

    assert res.returncode == 0, res.stderr
    activity = pd.read_parquet(out_dir / "activity_evidence.parquet")
    first = activity.loc[activity["ligand_id"] == "BDBM1"].sort_values("uniprot")
    assert first["uniprot"].tolist() == ["P11111", "P22222"]
    assert first["target_chain_count"].tolist() == [2, 2]
    assert first["single_chain_target"].tolist() == [False, False]
    assert "Q99999" not in set(activity["uniprot"])
    assert first["publication_date"].tolist() == ["2019-06-30", "2019-06-30"]
    assert first["evidence_date_source"].tolist() == ["publication", "publication"]
    assert first["source_doi"].tolist() == ["10.1000/article", "10.1000/article"]
    assert first["source_article_id"].tolist() == ["10.7270/entry", "10.7270/entry"]
    fallback = activity.loc[activity["ligand_id"] == "BDBM2"].iloc[0]
    assert fallback["uniprot"] == "P33333"
    assert fallback["target_chain_count"] == 1
    assert bool(fallback["single_chain_target"])
    assert fallback["publication_date"] == ""
    assert fallback["evidence_date"] == "2021-02-03"
    assert fallback["evidence_date_source"] == "curation"


def test_normalizer_keeps_original_row_number_after_skipped_rows(
    tmp_path: Path,
) -> None:
    tsv = _write_tsv(
        tmp_path / "BindingDB_All.tsv",
        [
            "CCC\t\tBDBM0\tMus musculus\tP00001\t5\t\t2018-01-01\t2024-01-01\t10.1000/mouse\t1\tBindingDB terms\n",
            "CCO\tBSYNRYMUTXBXSQ-UHFFFAOYSA-N\tBDBM1\tHomo sapiens\tP11111\t10\t\t2019-01-01\t2024-01-01\t10.1000/row\t123\tBindingDB terms\n",
        ],
    )
    out_dir = tmp_path / "out"

    res = run_script(_base_args(tsv, out_dir))

    assert res.returncode == 0, res.stderr
    activity = pd.read_parquet(out_dir / "activity_evidence.parquet")
    assert activity["input_row_number"].tolist() == [2]
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["counts"]["input_rows_seen"] == 2
    assert manifest["counts"]["skipped_non_human"] == 1


def test_normalizer_fails_closed_without_date_or_evidence_headers(
    tmp_path: Path,
) -> None:
    stale = tmp_path / "out" / "activity_evidence.parquet"
    stale.parent.mkdir()
    stale.write_text("stale\n")
    tsv = _write_tsv(
        tmp_path / "BindingDB_All.tsv",
        ["CCO\tHomo sapiens\tP11111\n"],
        header=(
            "Ligand SMILES\tTarget Source Organism According to Curator or DataSource\t"
            "UniProt (SwissProt) Primary ID of Target Chain 1\n"
        ),
    )

    res = run_script(_base_args(tsv, stale.parent))

    assert res.returncode != 0
    assert "publication/curation date column" in res.stderr
    assert "Ki/IC50/Kd/EC50 evidence column" in res.stderr
    assert not stale.exists()
