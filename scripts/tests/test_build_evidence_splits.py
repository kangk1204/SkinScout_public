from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "build_evidence_splits.py"
pytest.importorskip("pyarrow")


def run_builder(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT)] + args,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _run(evidence: list[tuple[Path, str]], out_dir: Path, cutoff: str = "2020-01-01") -> subprocess.CompletedProcess[str]:
    args: list[str] = []
    for path, label in evidence:
        args.extend(["--evidence", f"{path}={label}"])
    args.extend(["--cutoff-date", cutoff, "--out-dir", str(out_dir)])
    return run_builder(args)


def test_exact_year_and_boundary_dates_split_conservatively(tmp_path: Path) -> None:
    chembl = pd.DataFrame(
        [
            {
                "source_name": "ChEMBL",
                "source_release": "37",
                "source_license": "CC BY-SA 3.0",
                "uniprot": "P11111",
                "molecule_chembl_id": "CHEMBL1",
                "smiles": "CCO",
                "standard_inchi_key": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
                "activity_id": 1,
                "standard_type": "IC50",
                "standard_value": 10.0,
                "assay_type": "B",
                "evidence_date": "2025",
                "publication_date": "2019-12-31",
                "document_year": "2025",
            },
            {
                "source_name": "ChEMBL",
                "source_release": "37",
                "source_license": "CC BY-SA 3.0",
                "uniprot": "P22222",
                "molecule_chembl_id": "CHEMBL2",
                "smiles": "CCN",
                "standard_inchi_key": "WGKFWAGZGLNEIN-UHFFFAOYSA-N",
                "activity_id": 2,
                "standard_type": "Ki",
                "standard_value": 20.0,
                "assay_type": "F",
                "publication_date": "2020-01-01",
            },
            {
                "source_name": "ChEMBL",
                "source_release": "37",
                "source_license": "CC BY-SA 3.0",
                "uniprot": "P33333",
                "molecule_chembl_id": "CHEMBL3",
                "smiles": "CCC",
                "standard_inchi_key": "ATUOYWHBWRKTHZ-UHFFFAOYSA-N",
                "activity_id": 3,
                "standard_type": "Kd",
                "standard_value": 30.0,
                "assay_type": "B",
                "publication_date": "",
                "document_year": "2020",
            },
            {
                "source_name": "ChEMBL",
                "source_release": "37",
                "source_license": "CC BY-SA 3.0",
                "uniprot": "P44444",
                "molecule_chembl_id": "CHEMBL4",
                "smiles": "CCCC",
                "standard_inchi_key": "IJDNQMDRQITEOD-UHFFFAOYSA-N",
                "activity_id": 4,
                "standard_type": "EC50",
                "standard_value": 40.0,
                "assay_type": "A",
                "publication_date": "",
                "document_year": "",
            },
        ]
    )
    path = tmp_path / "chembl.parquet"
    chembl.to_parquet(path, index=False)

    res = _run([(path, "chembl")], tmp_path / "out")

    assert res.returncode == 0, res.stderr
    pre = pd.read_parquet(tmp_path / "out" / "pre_cutoff.parquet")
    post = pd.read_parquet(tmp_path / "out" / "post_cutoff.parquet")
    undated = pd.read_parquet(tmp_path / "out" / "undated.parquet")
    assert pre["activity_id"].tolist() == [1, 2]
    assert pre.loc[pre["activity_id"].eq(1), "evidence_date_source"].iloc[0] == "publication_date"
    assert post["activity_id"].tolist() == [3]
    assert post["evidence_date"].tolist() == ["2020-12-31"]
    assert post["evidence_date_precision"].tolist() == ["year_conservative_dec31"]
    assert undated["activity_id"].tolist() == [4]
    assert set(pre["assay_type"]) == {"B", "F"}
    manifest = json.loads((tmp_path / "out" / "split_manifest.json").read_text())
    assert manifest["counts"] == {
        "all_rows": 4,
        "post_cutoff": 1,
        "post_cutoff_novel_pairs": 1,
        "pre_cutoff": 2,
        "undated": 1,
    }
    assert manifest["input_sha256"]["chembl"]
    assert set(manifest["output_sha256"]) == {
        "pre_cutoff.parquet",
        "post_cutoff.parquet",
        "undated.parquet",
        "post_cutoff_novel_pairs.parquet",
    }
    assert manifest["manifest_sha256_policy"].startswith("split_manifest.json is excluded")
    manifest_path = tmp_path / "out" / "split_manifest.json"
    assert manifest["outputs"]["split_manifest.json"]["bytes"] == manifest_path.stat().st_size


def test_cold_flags_and_novel_pair_exclusion(tmp_path: Path) -> None:
    rows = pd.DataFrame(
        [
            {
                "source_db": "BindingDB",
                "source_release": "2026-08",
                "source_license": "BindingDB terms",
                "uniprot": "P11111",
                "ligand_id": "L1",
                "ligand_smiles": "CCO",
                "ligand_inchikey": "AAA",
                "evidence_id": "pre-pair",
                "affinity_type": "Ki",
                "affinity_value": 10.0,
                "evidence_date": "2019-01-01",
            },
            {
                "source_db": "BindingDB",
                "source_release": "2026-08",
                "source_license": "BindingDB terms",
                "uniprot": "P11111",
                "ligand_id": "L1",
                "ligand_smiles": "CCO",
                "ligand_inchikey": "AAA",
                "evidence_id": "post-seen-pair",
                "affinity_type": "IC50",
                "affinity_value": 20.0,
                "evidence_date": "2021-01-01",
            },
            {
                "source_db": "BindingDB",
                "source_release": "2026-08",
                "source_license": "BindingDB terms",
                "uniprot": "P22222",
                "ligand_id": "L1",
                "ligand_smiles": "CCO",
                "ligand_inchikey": "AAA",
                "evidence_id": "post-seen-molecule",
                "affinity_type": "Kd",
                "affinity_value": 30.0,
                "evidence_date": "2021-01-02",
            },
            {
                "source_db": "BindingDB",
                "source_release": "2026-08",
                "source_license": "BindingDB terms",
                "uniprot": "P11111",
                "ligand_id": "L2",
                "ligand_smiles": "CCN",
                "ligand_inchikey": "BBB",
                "evidence_id": "post-seen-target",
                "affinity_type": "EC50",
                "affinity_value": 40.0,
                "evidence_date": "2021-01-03",
            },
        ]
    )
    path = tmp_path / "bindingdb.parquet"
    rows.to_parquet(path, index=False)

    res = _run([(path, "bindingdb")], tmp_path / "out")

    assert res.returncode == 0, res.stderr
    post = pd.read_parquet(tmp_path / "out" / "post_cutoff.parquet")
    flags = post.set_index("activity_identity")[
        ["pair_seen_pre_cutoff", "molecule_seen_pre_cutoff", "target_seen_pre_cutoff"]
    ].to_dict("index")
    assert flags["post-seen-pair"] == {
        "pair_seen_pre_cutoff": True,
        "molecule_seen_pre_cutoff": True,
        "target_seen_pre_cutoff": True,
    }
    assert flags["post-seen-molecule"] == {
        "pair_seen_pre_cutoff": False,
        "molecule_seen_pre_cutoff": True,
        "target_seen_pre_cutoff": False,
    }
    assert flags["post-seen-target"] == {
        "pair_seen_pre_cutoff": False,
        "molecule_seen_pre_cutoff": False,
        "target_seen_pre_cutoff": True,
    }
    novel = pd.read_parquet(tmp_path / "out" / "post_cutoff_novel_pairs.parquet")
    assert novel["activity_identity"].tolist() == ["post-seen-molecule", "post-seen-target"]


def test_mixed_inputs_preserve_rows_and_deterministic_ids(tmp_path: Path) -> None:
    chembl = pd.DataFrame(
        [
            {
                "source_name": "ChEMBL",
                "source_release": "37",
                "source_license": "CC BY-SA 3.0",
                "uniprot": "P11111",
                "molecule_chembl_id": "CHEMBL1",
                "standard_inchi_key": "AAA",
                "activity_id": 10,
                "standard_type": "IC50",
                "standard_value": 1.0,
                "document_year": 2018,
            }
        ]
    )
    bindingdb = pd.DataFrame(
        [
            {
                "source_db": "BindingDB",
                "source_release": "2026-08",
                "source_license": "BindingDB terms",
                "uniprot": "P22222",
                "ligand_id": "BDBM1",
                "ligand_inchikey": "BBB",
                "evidence_id": "bdb-e1",
                "input_row_number": 99,
                "affinity_type": "Ki",
                "affinity_value": 2.0,
                "evidence_date": "2022-01-01",
            }
        ]
    )
    chembl_path = tmp_path / "chembl.parquet"
    bindingdb_path = tmp_path / "bindingdb.parquet"
    chembl.to_parquet(chembl_path, index=False)
    bindingdb.to_parquet(bindingdb_path, index=False)

    out_dir = tmp_path / "out"
    first = _run([(bindingdb_path, "bindingdb"), (chembl_path, "chembl")], out_dir)
    assert first.returncode == 0, first.stderr
    first_ids = {
        name: pd.read_parquet(out_dir / name)["evidence_id"].tolist()
        for name in ["pre_cutoff.parquet", "post_cutoff.parquet"]
    }
    second = _run([(bindingdb_path, "bindingdb"), (chembl_path, "chembl")], out_dir)
    assert second.returncode == 0, second.stderr
    second_ids = {
        name: pd.read_parquet(out_dir / name)["evidence_id"].tolist()
        for name in ["pre_cutoff.parquet", "post_cutoff.parquet"]
    }
    assert first_ids == second_ids
    post = pd.read_parquet(out_dir / "post_cutoff.parquet")
    assert post.loc[post["input_label"].eq("bindingdb"), "input_row_number"].tolist() == [99]
    all_rows = sum(len(pd.read_parquet(out_dir / name)) for name in ["pre_cutoff.parquet", "post_cutoff.parquet"])
    assert all_rows == 2


def test_missing_schema_fails_closed_and_removes_stale_outputs(tmp_path: Path) -> None:
    bad = pd.DataFrame(
        [
            {
                "source_db": "BindingDB",
                "source_release": "2026-08",
                "source_license": "BindingDB terms",
                "ligand_id": "BDBM1",
                "evidence_id": "bad",
                "affinity_type": "Ki",
                "affinity_value": 2.0,
                "evidence_date": "2022-01-01",
            }
        ]
    )
    bad_path = tmp_path / "bad.parquet"
    bad.to_parquet(bad_path, index=False)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    for name in [
        "pre_cutoff.parquet",
        "post_cutoff.parquet",
        "undated.parquet",
        "post_cutoff_novel_pairs.parquet",
        "split_manifest.json",
    ]:
        (out_dir / name).write_text("stale\n")

    res = _run([(bad_path, "bad")], out_dir)

    assert res.returncode != 0
    assert "target_uniprot" in res.stderr
    assert not any((out_dir / name).exists() for name in [
        "pre_cutoff.parquet",
        "post_cutoff.parquet",
        "undated.parquet",
        "post_cutoff_novel_pairs.parquet",
        "split_manifest.json",
    ])
