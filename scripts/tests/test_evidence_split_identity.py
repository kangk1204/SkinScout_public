"""Regression tests for Stage 0 evidence-split structure identity (F45).

The split key used to be chosen per row from InChIKey -> source molecule ID ->
raw SMILES, so the same structure could receive different keys depending on
which fields a source happened to fill. These tests pin the standardized parent
InChIKey as the split primary key, source IDs as aliases, salt/parent grouping,
and the explicit unresolved policy.
"""

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

ACETIC_ACID_INCHIKEY = "QTBSBXVTEAMEQO-UHFFFAOYSA-N"
ETHANOL_INCHIKEY = "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"


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


def _chembl_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "source_name": "ChEMBL",
        "source_release": "37",
        "source_license": "CC BY-SA 3.0",
        "uniprot": "P11111",
        "molecule_chembl_id": "CHEMBL1",
        "activity_id": 1,
        "standard_type": "IC50",
        "standard_value": 10.0,
        "evidence_date": "2019-01-01",
    }
    row.update(overrides)
    return row


def _bindingdb_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "source_db": "BindingDB",
        "source_release": "2026-08",
        "source_license": "BindingDB terms",
        "uniprot": "P11111",
        "ligand_id": "BDBM1",
        "evidence_id": "bdb-1",
        "affinity_type": "Ki",
        "affinity_value": 5.0,
        "evidence_date": "2021-01-01",
    }
    row.update(overrides)
    return row


def test_same_structure_different_source_keys_share_a_split_group(tmp_path: Path) -> None:
    chembl = pd.DataFrame([
        _chembl_row(standard_inchi_key=ACETIC_ACID_INCHIKEY)
    ])
    bindingdb = pd.DataFrame([
        _bindingdb_row(ligand_smiles="CC(=O)[O-].[Na+]")
    ])
    chembl_path = tmp_path / "chembl.parquet"
    bindingdb_path = tmp_path / "bindingdb.parquet"
    chembl.to_parquet(chembl_path, index=False)
    bindingdb.to_parquet(bindingdb_path, index=False)

    res = _run(
        [(chembl_path, "chembl"), (bindingdb_path, "bindingdb")],
        tmp_path / "out",
    )

    assert res.returncode == 0, res.stderr
    pre = pd.read_parquet(tmp_path / "out" / "pre_cutoff.parquet")
    post = pd.read_parquet(tmp_path / "out" / "post_cutoff.parquet")
    assert pre["molecule_key"].tolist() == post["molecule_key"].tolist()
    assert post["molecule_key_source"].tolist() == ["standardized_parent_inchikey"]
    # The source supplied only an InChIKey; the other source supplied the salt,
    # and both resolve to the same parent structure, so the pair is warm.
    assert bool(post["molecule_seen_pre_cutoff"].iloc[0]) is True
    assert bool(post["pair_seen_pre_cutoff"].iloc[0]) is True


def test_salt_and_parent_smiles_from_one_source_group_together(tmp_path: Path) -> None:
    rows = pd.DataFrame([
        _chembl_row(
            molecule_chembl_id="CHEMBL1",
            smiles="CC(=O)O",
            standard_inchi_key="",
            evidence_date="2019-01-01",
        ),
        _chembl_row(
            molecule_chembl_id="CHEMBL2",
            smiles="CC(=O)[O-].[Na+]",
            standard_inchi_key="",
            activity_id=2,
            evidence_date="2021-01-01",
        ),
    ])
    path = tmp_path / "chembl.parquet"
    rows.to_parquet(path, index=False)

    res = _run([(path, "chembl")], tmp_path / "out")

    assert res.returncode == 0, res.stderr
    pre = pd.read_parquet(tmp_path / "out" / "pre_cutoff.parquet")
    post = pd.read_parquet(tmp_path / "out" / "post_cutoff.parquet")
    parent_key = f"inchikey:{ACETIC_ACID_INCHIKEY}"
    assert pre["molecule_key"].tolist() == [parent_key]
    assert post["molecule_key"].tolist() == [parent_key]
    assert bool(post["molecule_seen_pre_cutoff"].iloc[0]) is True
    assert bool(post["pair_seen_pre_cutoff"].iloc[0]) is True


def test_distinct_structures_do_not_share_a_group(tmp_path: Path) -> None:
    rows = pd.DataFrame([
        _chembl_row(
            molecule_chembl_id="CHEMBL1",
            smiles="CC(=O)O",
            standard_inchi_key="",
            evidence_date="2019-01-01",
        ),
        _chembl_row(
            molecule_chembl_id="CHEMBL2",
            smiles="CCO",
            standard_inchi_key="",
            activity_id=2,
            evidence_date="2021-01-01",
        ),
    ])
    path = tmp_path / "chembl.parquet"
    rows.to_parquet(path, index=False)

    res = _run([(path, "chembl")], tmp_path / "out")

    assert res.returncode == 0, res.stderr
    pre = pd.read_parquet(tmp_path / "out" / "pre_cutoff.parquet")
    post = pd.read_parquet(tmp_path / "out" / "post_cutoff.parquet")
    assert pre["molecule_key"].tolist() == [
        f"inchikey:{ACETIC_ACID_INCHIKEY}"
    ]
    assert post["molecule_key"].tolist() == [
        f"inchikey:{ETHANOL_INCHIKEY}"
    ]
    assert bool(post["molecule_seen_pre_cutoff"].iloc[0]) is False


def test_unresolved_structures_get_an_explicit_alias_policy(tmp_path: Path) -> None:
    rows = pd.DataFrame([
        _chembl_row(
            molecule_chembl_id="CHEMBL1",
            smiles="",
            standard_inchi_key="",
        ),
        _chembl_row(
            molecule_chembl_id="",
            smiles="this-is-not-a-smiles",
            standard_inchi_key="",
            activity_id=2,
        ),
    ])
    path = tmp_path / "chembl.parquet"
    rows.to_parquet(path, index=False)

    res = _run([(path, "chembl")], tmp_path / "out")

    assert res.returncode == 0, res.stderr
    pre = pd.read_parquet(tmp_path / "out" / "pre_cutoff.parquet").sort_values(
        "activity_id"
    )
    assert pre["molecule_key_source"].tolist() == [
        "unresolved_source_molecule_id",
        "unresolved_raw_smiles",
    ]
    assert pre["molecule_key"].str.startswith("unresolved").all()
    manifest = json.loads((tmp_path / "out" / "split_manifest.json").read_text())
    assert manifest["counts"]["unresolved_molecule_identities"] == 2
    policy = manifest["policy"]["molecule_identity"]
    assert "standardized parent InChIKey" in policy["primary_key"]
    assert "unresolved" in policy["unresolved_policy"]
    assert "molecule_key_source" in manifest["schema"]["normalized_columns"]
