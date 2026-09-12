"""Regression tests for Stage 0 reference mirror placeholder gates."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def load_stage0_skin_kg():
    spec = importlib.util.spec_from_file_location(
        "stage0_skin_kg",
        ROOT / "scripts/stage0_skin_kg.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_stage0_cosing():
    spec = importlib.util.spec_from_file_location(
        "stage0_cosing",
        ROOT / "scripts/stage0_cosing.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_bindingdb_zip(path: Path, tsv: str, member: str = "BindingDB_All.tsv") -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member, tsv)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_drugbank_mirror_fails_without_xml_by_default(tmp_path: Path) -> None:
    res = subprocess.run(
        ["bash", str(ROOT / "scripts/stage0_mirror_drugbank.sh"), str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "not found" in res.stderr
    assert not (tmp_path / "drugbank_polypharm.parquet").exists()


def test_drugbank_placeholder_requires_explicit_flag(tmp_path: Path) -> None:
    res = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/stage0_mirror_drugbank.sh"),
            str(tmp_path),
            "--allow-placeholder",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode == 0, res.stderr
    assert (tmp_path / "drugbank_polypharm.parquet").exists()
    assert (tmp_path / "drugbank_approved.parquet").exists()


def test_drugbank_optional_missing_writes_empty_slice(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    res = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/stage0_mirror_drugbank.sh"),
            str(tmp_path),
            "--allow-missing-optional",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode == 0, res.stderr
    assert "optional licensed enrichment absent" in res.stderr
    approved = pd.read_parquet(tmp_path / "drugbank_approved.parquet")
    polypharm = pd.read_parquet(tmp_path / "drugbank_polypharm.parquet")
    assert approved.empty
    assert list(approved.columns) == ["drug_id", "name", "smiles"]
    assert polypharm.empty
    assert list(polypharm.columns) == ["drugbank_id", "drug_name", "uniprot"]


def test_drugbank_mirror_writes_approved_drug_subset(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    (tmp_path / "drugbank_full_database.xml").write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<drugbank xmlns="http://www.drugbank.ca">
  <drug>
    <drugbank-id primary="true">DB0001</drugbank-id>
    <name>Example Approved</name>
    <groups><group>approved</group></groups>
    <calculated-properties>
      <property><kind>SMILES</kind><value>CCO</value></property>
    </calculated-properties>
    <targets>
      <target>
        <polypeptide>
          <external-identifiers>
            <external-identifier>
              <resource>UniProtKB</resource>
              <identifier>P12345</identifier>
            </external-identifier>
          </external-identifiers>
        </polypeptide>
      </target>
    </targets>
  </drug>
  <drug>
    <drugbank-id primary="true">DB0002</drugbank-id>
    <name>Example Experimental</name>
    <groups><group>experimental</group></groups>
    <calculated-properties>
      <property><kind>SMILES</kind><value>CCN</value></property>
    </calculated-properties>
  </drug>
</drugbank>
"""
    )

    res = subprocess.run(
        ["bash", str(ROOT / "scripts/stage0_mirror_drugbank.sh"), str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode == 0, res.stderr
    approved = pd.read_parquet(tmp_path / "drugbank_approved.parquet")
    assert approved.to_dict("records") == [
        {"drug_id": "DB0001", "name": "Example Approved", "smiles": "CCO"}
    ]
    polypharm = pd.read_parquet(tmp_path / "drugbank_polypharm.parquet")
    assert polypharm.to_dict("records") == [
        {"drugbank_id": "DB0001", "drug_name": "Example Approved", "uniprot": "P12345"}
    ]


def test_bindingdb_no_pdb_slice_fails_without_placeholder_flag(tmp_path: Path) -> None:
    _write_bindingdb_zip(
        tmp_path / "BindingDB_All_202405_tsv.zip",
        "Ligand SMILES\tTarget Name\nCCO\tX\n",
    )

    res = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/stage0_mirror_bindingdb.sh"),
            str(tmp_path),
            "--release",
            "2024-05",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "no PDB-linked BindingDB rows" in res.stderr
    assert not (tmp_path / "bindingdb_with_pdb.parquet").exists()


def test_bindingdb_requires_pinned_release(tmp_path: Path) -> None:
    (tmp_path / "BindingDB_All.tsv.zip").write_bytes(b"unpinned")
    (tmp_path / "BindingDB_All.tsv").write_text(
        "Ligand SMILES\tTarget Name\nCCO\tX\n"
    )

    res = subprocess.run(
        ["bash", str(ROOT / "scripts/stage0_mirror_bindingdb.sh"), str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "--release YYYY-MM is required" in res.stderr
    assert not (tmp_path / "bindingdb_source_manifest.json").exists()
    assert not (tmp_path / "bindingdb_with_pdb.parquet").exists()


def test_bindingdb_release_downloads_exact_archive_and_writes_source_manifest(
    tmp_path: Path,
) -> None:
    pd = pytest.importorskip("pandas")
    fixture = tmp_path / "fixture.zip"
    tsv = (
        "Ligand SMILES\tTarget Name\tPDB ID(s) for Ligand-Target Complex\n"
        "CCO\tTarget A\t1ABC\n"
    )
    _write_bindingdb_zip(fixture, tsv)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    wget = fake_bin / "wget"
    wget.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "out=''\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = '-O' ]; then out=\"$2\"; shift 2; else shift; fi\n"
        "done\n"
        "cp \"$FAKE_BINDINGDB_ZIP\" \"$out\"\n"
    )
    wget.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["FAKE_BINDINGDB_ZIP"] = str(fixture)
    out_dir = tmp_path / "mirror"

    res = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/stage0_mirror_bindingdb.sh"),
            str(out_dir),
            "--release",
            "2024-05",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert res.returncode == 0, res.stderr
    archive = out_dir / "BindingDB_All_202405_tsv.zip"
    extracted = out_dir / "BindingDB_All.tsv"
    manifest = json.loads((out_dir / "bindingdb_source_manifest.json").read_text())
    assert archive.exists()
    assert extracted.read_text() == tsv
    assert manifest["release"] == "2024-05"
    assert manifest["url"].endswith("/BindingDB_All_202405_tsv.zip")
    assert manifest["archive"]["path"] == "BindingDB_All_202405_tsv.zip"
    assert manifest["archive"]["sha256"] == _sha256(archive)
    assert manifest["extracted"]["path"] == "BindingDB_All.tsv"
    assert manifest["extracted"]["sha256"] == _sha256(extracted)
    assert manifest["source"]["name"] == "BindingDB"
    assert manifest["source"]["license"] == "CC BY 3.0"
    assert manifest["source"]["redistribution"] == "allowed"
    assert manifest["release_date"] == "2024-05-01"
    assert manifest["archive"]["bytes"] == archive.stat().st_size
    assert manifest["extracted"]["bytes"] == extracted.stat().st_size
    sliced = pd.read_parquet(out_dir / "bindingdb_with_pdb.parquet")
    assert sliced["PDB ID(s) for Ligand-Target Complex"].tolist() == ["1ABC"]


def test_bindingdb_release_reuses_release_specific_archive_without_network(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pandas")
    tsv = (
        "Ligand SMILES\tTarget Name\tPDB ID(s) for Ligand-Target Complex\n"
        "CCN\tTarget B\t2DEF\n"
    )
    _write_bindingdb_zip(tmp_path / "BindingDB_All_202312_tsv.zip", tsv)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    wget = fake_bin / "wget"
    wget.write_text("#!/usr/bin/env bash\nexit 99\n")
    wget.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    res = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/stage0_mirror_bindingdb.sh"),
            str(tmp_path),
            "--release",
            "2023-12",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert res.returncode == 0, res.stderr
    assert "reusing BindingDB_All_202312_tsv.zip" in res.stdout
    assert (tmp_path / "BindingDB_All.tsv").read_text() == tsv
    manifest = json.loads((tmp_path / "bindingdb_source_manifest.json").read_text())
    assert manifest["release"] == "2023-12"


def test_bindingdb_release_invalid_zip_removes_stale_claim_artifacts(
    tmp_path: Path,
) -> None:
    (tmp_path / "BindingDB_All_202405_tsv.zip").write_bytes(b"not a zip")
    (tmp_path / "BindingDB_All.tsv").write_text("stale\n")
    (tmp_path / "bindingdb_with_pdb.parquet").write_text("stale\n")
    (tmp_path / "bindingdb_source_manifest.json").write_text('{"stale": true}\n')

    res = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/stage0_mirror_bindingdb.sh"),
            str(tmp_path),
            "--release",
            "2024-05",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "not a valid ZIP" in res.stderr
    assert not (tmp_path / "BindingDB_All.tsv").exists()
    assert not (tmp_path / "bindingdb_with_pdb.parquet").exists()
    assert not (tmp_path / "bindingdb_source_manifest.json").exists()


def test_bindingdb_release_slice_failure_removes_extracted_claim_manifest(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pandas")
    _write_bindingdb_zip(
        tmp_path / "BindingDB_All_202405_tsv.zip",
        "Ligand SMILES\tTarget Name\nCCO\tTarget A\n",
    )

    res = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/stage0_mirror_bindingdb.sh"),
            str(tmp_path),
            "--release",
            "2024-05",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "no PDB-linked BindingDB rows" in res.stderr
    assert not (tmp_path / "BindingDB_All.tsv").exists()
    assert not (tmp_path / "bindingdb_with_pdb.parquet").exists()
    assert not (tmp_path / "bindingdb_source_manifest.json").exists()


def test_bindingdb_no_pdb_slice_placeholder_requires_explicit_flag(tmp_path: Path) -> None:
    _write_bindingdb_zip(
        tmp_path / "BindingDB_All_202405_tsv.zip",
        "Ligand SMILES\tTarget Name\nCCO\tX\n",
    )

    res = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/stage0_mirror_bindingdb.sh"),
            str(tmp_path),
            "--release",
            "2024-05",
            "--allow-placeholder",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode == 0, res.stderr
    assert (tmp_path / "bindingdb_with_pdb.parquet").exists()


def test_bindingdb_pdb_slice_handles_mixed_type_columns(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    _write_bindingdb_zip(
        tmp_path / "BindingDB_All_202405_tsv.zip",
        "Ligand SMILES\tTarget Name\tPDB ID(s) for Ligand-Target Complex\tkon (M-1-s-1)\n"
        "CCO\tTarget A\t1ABC\t 200000\n"
        "CCN\tTarget B\t2DEF\tnot determined\n",
    )

    res = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/stage0_mirror_bindingdb.sh"),
            str(tmp_path),
            "--release",
            "2024-05",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode == 0, res.stderr
    sliced = pd.read_parquet(tmp_path / "bindingdb_with_pdb.parquet")
    assert len(sliced) == 2
    assert sliced["kon (M-1-s-1)"].astype(str).tolist() == [
        " 200000",
        "not determined",
    ]


def test_cosing_ingest_requires_source_csv_without_dry_run(tmp_path: Path) -> None:
    (tmp_path / "cosing.parquet").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage0_cosing.py"),
            "--out-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "CosIng CSV is required for full ingest" in res.stderr
    assert not (tmp_path / "cosing.parquet").exists()


def test_cosing_ingest_rejects_missing_inci_name_column(tmp_path: Path) -> None:
    (tmp_path / "cosing.csv").write_text("cas,function\n98-92-0,Skin Conditioning\n")
    (tmp_path / "cosing.parquet").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage0_cosing.py"),
            "--out-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "CosIng CSV must include an INCI name column" in res.stderr
    assert not (tmp_path / "cosing.parquet").exists()


def test_cosing_ingest_rejects_blank_inci_names(tmp_path: Path) -> None:
    (tmp_path / "cosing.csv").write_text("inci_name,cas,function\n,98-92-0,Skin Conditioning\n")
    (tmp_path / "cosing.parquet").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage0_cosing.py"),
            "--out-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "CosIng CSV column 'inci_name' contains blank values" in res.stderr
    assert not (tmp_path / "cosing.parquet").exists()


def test_cosing_ingest_rejects_partially_blank_inci_names(tmp_path: Path) -> None:
    (tmp_path / "cosing.csv").write_text(
        "inci_name,cas,function\n"
        "Niacinamide,98-92-0,Skin Conditioning\n"
        ",56-81-5,Humectant\n"
    )
    (tmp_path / "cosing.parquet").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage0_cosing.py"),
            "--out-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "CosIng CSV column 'inci_name' contains blank values" in res.stderr
    assert "row index 1" in res.stderr
    assert not (tmp_path / "cosing.parquet").exists()


def test_cosing_ingest_parses_semicolon_delimited_csv(tmp_path: Path) -> None:
    stage0_cosing = load_stage0_cosing()
    csv_path = tmp_path / "cosing.csv"
    csv_path.write_text(
        "INCI name;CAS;Function\n"
        'Water;7732-18-5;"Solvent;Humectant"\n'
    )

    rows = stage0_cosing.parse_input(csv_path)

    assert len(rows) == 1
    assert rows[0].inci_name == "Water"
    assert rows[0].cas == "7732-18-5"
    assert rows[0].functions == ["Solvent", "Humectant"]


def test_cosing_ingest_parses_quoted_newlines(tmp_path: Path) -> None:
    stage0_cosing = load_stage0_cosing()
    csv_path = tmp_path / "cosing.csv"
    csv_path.write_text(
        'INCI name,CAS,Function\n'
        '"Alpha\nBeta",123-45-6,"Skin Conditioning;Humectant"\n',
        encoding="utf-8",
    )

    rows = stage0_cosing.parse_input(csv_path)

    assert len(rows) == 1
    assert rows[0].inci_name == "Alpha\nBeta"
    assert rows[0].cas == "123-45-6"
    assert rows[0].functions == ["Skin Conditioning", "Humectant"]


def test_cosing_ingest_uses_cas_before_inci_name() -> None:
    stage0_cosing = load_stage0_cosing()
    row = stage0_cosing.CosingRow(
        inci_name="Long Regulatory Name",
        cas="11-11-1;22-22-2",
        einecs=None,
        functions=[],
        smiles=None,
    )

    assert stage0_cosing._identifier_candidates(row) == [
        "11-11-1",
        "22-22-2",
        "Long Regulatory Name",
    ]


def test_cosing_ingest_placeholder_requires_dry_run(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage0_cosing.py"),
            "--out-dir",
            str(tmp_path),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode == 0, res.stderr
    assert (tmp_path / "cosing.parquet").exists()
    out = pd.read_parquet(tmp_path / "cosing.parquet")
    assert len(out.iloc[0]["ecfp4"]) == 32
    assert all(isinstance(word, str) for word in out.iloc[0]["ecfp4"])


def test_cosing_ingest_dry_run_ignores_malformed_source_csv(tmp_path: Path) -> None:
    (tmp_path / "cosing.csv").write_text("wrong\nvalue\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage0_cosing.py"),
            "--out-dir",
            str(tmp_path),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode == 0, res.stderr
    assert (tmp_path / "cosing.parquet").exists()


def test_cosing_ingest_rejects_missing_pubchem_smiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage0_cosing = load_stage0_cosing()
    (tmp_path / "cosing.csv").write_text(
        "inci_name,cas,function\nNiacinamide,98-92-0,Skin Conditioning\n"
    )
    (tmp_path / "cosing.parquet").write_text("stale\n")
    monkeypatch.setattr(stage0_cosing, "_fetch_smiles_from_identifier", lambda *_: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["stage0_cosing.py", "--out-dir", str(tmp_path)],
    )

    with pytest.raises(SystemExit, match="returned no SMILES"):
        stage0_cosing.main()

    assert not (tmp_path / "cosing.parquet").exists()


def test_cosing_ingest_rejects_invalid_pubchem_smiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage0_cosing = load_stage0_cosing()
    (tmp_path / "cosing.csv").write_text(
        "inci_name,cas,function\nNiacinamide,98-92-0,Skin Conditioning\n"
    )
    (tmp_path / "cosing.parquet").write_text("stale\n")
    monkeypatch.setattr(
        stage0_cosing,
        "_fetch_smiles_from_identifier",
        lambda *_: "not_a_smiles",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["stage0_cosing.py", "--out-dir", str(tmp_path)],
    )

    with pytest.raises(SystemExit, match="returned invalid SMILES"):
        stage0_cosing.main()

    assert not (tmp_path / "cosing.parquet").exists()


def test_cosing_ingest_allows_partial_pubchem_resolution_above_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pd = pytest.importorskip("pandas")
    stage0_cosing = load_stage0_cosing()
    (tmp_path / "cosing.csv").write_text(
        "inci_name,cas,function\n"
        "Resolved Ingredient,11-11-1,Skin Conditioning\n"
        "Polymer Extract,22-22-2,Film Forming\n"
    )
    monkeypatch.setattr(
        stage0_cosing,
        "_fetch_smiles_from_identifier",
        lambda identifier: "CCO" if identifier == "11-11-1" else None,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage0_cosing.py",
            "--out-dir",
            str(tmp_path),
            "--min-resolution-fraction",
            "0.5",
            "--min-resolved-count",
            "1",
        ],
    )

    stage0_cosing.main()

    out = pd.read_parquet(tmp_path / "cosing.parquet")
    assert out["inci_name"].tolist() == ["Resolved Ingredient"]
    manifest = (tmp_path / "cosing_ingest_manifest.json").read_text()
    assert '"unresolved_row_count": 1' in manifest


def test_drug_avoidance_ingest_requires_real_reference_without_dry_run(
    tmp_path: Path,
) -> None:
    (tmp_path / "drugs.parquet").write_text("stale\n")
    (tmp_path / "scaffolds.parquet").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage0_drug_avoidance.py"),
            "--out-dir",
            str(tmp_path),
            "--chembl-dir",
            str(tmp_path / "missing_chembl"),
            "--orange-book",
            str(tmp_path / "missing_orange_book.csv"),
            "--drugbank-parquet",
            str(tmp_path / "missing_drugbank.parquet"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "full ingest found no approved-drug rows" in res.stderr
    assert not (tmp_path / "drugs.parquet").exists()
    assert not (tmp_path / "scaffolds.parquet").exists()


def test_drug_avoidance_ingest_rejects_bad_orange_book_schema(
    tmp_path: Path,
) -> None:
    orange_book = tmp_path / "orange_book.csv"
    orange_book.write_text("ingredient,cas\nAspirin,50-78-2\n")
    (tmp_path / "drugs.parquet").write_text("stale\n")
    (tmp_path / "scaffolds.parquet").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage0_drug_avoidance.py"),
            "--out-dir",
            str(tmp_path),
            "--chembl-dir",
            str(tmp_path / "missing_chembl"),
            "--orange-book",
            str(orange_book),
            "--drugbank-parquet",
            str(tmp_path / "missing_drugbank.parquet"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "Orange Book reference must include" in res.stderr
    assert not (tmp_path / "drugs.parquet").exists()
    assert not (tmp_path / "scaffolds.parquet").exists()


def test_drug_avoidance_ingest_rejects_bad_drugbank_schema(
    tmp_path: Path,
) -> None:
    drugbank = tmp_path / "drugbank.parquet"
    pd = pytest.importorskip("pandas")
    pd.DataFrame([{"drug_id": "DB0001", "name": "Example"}]).to_parquet(
        drugbank,
        index=False,
    )
    (tmp_path / "drugs.parquet").write_text("stale\n")
    (tmp_path / "scaffolds.parquet").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage0_drug_avoidance.py"),
            "--out-dir",
            str(tmp_path),
            "--chembl-dir",
            str(tmp_path / "missing_chembl"),
            "--orange-book",
            str(tmp_path / "missing_orange_book.csv"),
            "--drugbank-parquet",
            str(drugbank),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "DrugBank approved parquet is missing required columns" in res.stderr
    assert not (tmp_path / "drugs.parquet").exists()
    assert not (tmp_path / "scaffolds.parquet").exists()


def test_drug_avoidance_ingest_rejects_bad_chembl_schema(
    tmp_path: Path,
) -> None:
    chembl = tmp_path / "chembl"
    chembl.mkdir()
    pd = pytest.importorskip("pandas")
    pd.DataFrame([{"molecule_chembl_id": "CHEMBL1", "canonical_smiles": "CCO"}]).to_parquet(
        chembl / "human_activities.parquet",
        index=False,
    )
    (tmp_path / "drugs.parquet").write_text("stale\n")
    (tmp_path / "scaffolds.parquet").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage0_drug_avoidance.py"),
            "--out-dir",
            str(tmp_path),
            "--chembl-dir",
            str(chembl),
            "--orange-book",
            str(tmp_path / "missing_orange_book.csv"),
            "--drugbank-parquet",
            str(tmp_path / "missing_drugbank.parquet"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "ChEMBL approved parquet is missing required columns" in res.stderr
    assert not (tmp_path / "drugs.parquet").exists()
    assert not (tmp_path / "scaffolds.parquet").exists()


def test_drug_avoidance_ingest_rejects_blank_required_record_fields(
    tmp_path: Path,
) -> None:
    orange_book = tmp_path / "orange_book.csv"
    orange_book.write_text("ingredient,smiles\nAspirin,\n")
    (tmp_path / "drugs.parquet").write_text("stale\n")
    (tmp_path / "scaffolds.parquet").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage0_drug_avoidance.py"),
            "--out-dir",
            str(tmp_path),
            "--chembl-dir",
            str(tmp_path / "missing_chembl"),
            "--orange-book",
            str(orange_book),
            "--drugbank-parquet",
            str(tmp_path / "missing_drugbank.parquet"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "column 'smiles' contains blank values" in res.stderr
    assert not (tmp_path / "drugs.parquet").exists()
    assert not (tmp_path / "scaffolds.parquet").exists()


def test_drug_avoidance_ingest_rejects_invalid_smiles(
    tmp_path: Path,
) -> None:
    orange_book = tmp_path / "orange_book.csv"
    orange_book.write_text("ingredient,smiles\nAspirin,not_a_smiles\n")
    (tmp_path / "drugs.parquet").write_text("stale\n")
    (tmp_path / "scaffolds.parquet").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage0_drug_avoidance.py"),
            "--out-dir",
            str(tmp_path),
            "--chembl-dir",
            str(tmp_path / "missing_chembl"),
            "--orange-book",
            str(orange_book),
            "--drugbank-parquet",
            str(tmp_path / "missing_drugbank.parquet"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "column 'smiles' contains invalid SMILES" in res.stderr
    assert not (tmp_path / "drugs.parquet").exists()
    assert not (tmp_path / "scaffolds.parquet").exists()


def test_drug_avoidance_ingest_rejects_duplicate_chembl_molecule_id(
    tmp_path: Path,
) -> None:
    chembl = tmp_path / "chembl"
    chembl.mkdir()
    pd = pytest.importorskip("pandas")
    pd.DataFrame([
        {"molecule_chembl_id": "CHEMBL1", "smiles": "CCO"},
        {"molecule_chembl_id": "CHEMBL1", "smiles": "CCN"},
    ]).to_parquet(chembl / "human_activities.parquet", index=False)
    (tmp_path / "drugs.parquet").write_text("stale\n")
    (tmp_path / "scaffolds.parquet").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage0_drug_avoidance.py"),
            "--out-dir",
            str(tmp_path),
            "--chembl-dir",
            str(chembl),
            "--orange-book",
            str(tmp_path / "missing_orange_book.csv"),
            "--drugbank-parquet",
            str(tmp_path / "missing_drugbank.parquet"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "duplicate molecule_chembl_id values" in res.stderr
    assert not (tmp_path / "drugs.parquet").exists()
    assert not (tmp_path / "scaffolds.parquet").exists()


def test_drug_avoidance_ingest_placeholder_requires_dry_run(
    tmp_path: Path,
) -> None:
    pd = pytest.importorskip("pandas")
    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage0_drug_avoidance.py"),
            "--out-dir",
            str(tmp_path),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode == 0, res.stderr
    assert "DEPRECATION WARNING" not in res.stderr
    assert (tmp_path / "drugs.parquet").exists()
    assert (tmp_path / "scaffolds.parquet").exists()
    out = pd.read_parquet(tmp_path / "drugs.parquet")
    assert len(out.iloc[0]["ecfp4"]) == 32
    assert all(isinstance(word, str) for word in out.iloc[0]["ecfp4"])


def test_skin_kg_requires_pubtator_evidence_without_seed_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage0_skin_kg = load_stage0_skin_kg()
    monkeypatch.setattr(stage0_skin_kg, "query_pubtator", lambda _: [])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage0_skin_kg.py",
            "--out-dir",
            str(tmp_path),
        ],
    )

    with pytest.raises(SystemExit, match="PubTator enrichment is required"):
        stage0_skin_kg.main()

    assert not (tmp_path / "skin_efficacy.graphml").exists()


def test_skin_kg_pubtator_http_error_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage0_skin_kg = load_stage0_skin_kg()

    class Response:
        status_code = 503
        text = "service unavailable"

        def json(self):
            return {"results": []}

    monkeypatch.setattr(stage0_skin_kg.requests, "get", lambda *_, **__: Response())

    with pytest.raises(SystemExit, match="PubTator query failed for 'melanogenesis'"):
        stage0_skin_kg.query_pubtator("melanogenesis")


def test_skin_kg_pubtator_gene_mentions_create_uniprot_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage0_skin_kg = load_stage0_skin_kg()

    def fake_query(keyword: str):
        if keyword != "melanogenesis":
            return []
        return [
            {
                "pmid": "123",
                "text_hl": "melanogenesis @GENE_FAKE1 @GENE_7299",
            },
            {
                "pmid": "456",
                "text_hl": "melanogenesis @GENE_TYR",
            },
        ]

    def fake_map(symbol: str, **_kwargs):
        return {"FAKE1": "Q99999", "TYR": "P14679"}.get(symbol)

    monkeypatch.setattr(stage0_skin_kg, "query_pubtator", fake_query)
    monkeypatch.setattr(stage0_skin_kg, "map_gene_symbol_to_uniprot", fake_map)

    graph = stage0_skin_kg.enrich_with_pubtator(
        stage0_skin_kg.build_seed_graph(),
        uniprot_cache={},
        request_sleep_s=0,
    )

    assert graph.has_edge("gene:Q99999", "category:whitening")
    assert not any(str(node).startswith("gene:7299") for node in graph.nodes)
    assert graph.graph["pubtator_new_gene_edges"] == 1
    assert graph.nodes["category:whitening"]["pmid_count"] == 2


def test_skin_kg_seed_graph_requires_explicit_skip_pubtator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage0_skin_kg = load_stage0_skin_kg()
    monkeypatch.setattr(stage0_skin_kg, "query_pubtator", lambda _: [])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage0_skin_kg.py",
            "--out-dir",
            str(tmp_path),
            "--skip-pubtator",
        ],
    )

    stage0_skin_kg.main()

    assert (tmp_path / "skin_efficacy.graphml").exists()
