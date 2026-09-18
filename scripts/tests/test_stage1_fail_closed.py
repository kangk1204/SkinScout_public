"""Regression tests for Stage 1 fail-closed chemistry fallbacks."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import importlib.util
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parents[2]


def load_script_module(script: str):
    spec = importlib.util.spec_from_file_location(script, ROOT / "scripts" / script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_input_sdf(path: Path) -> None:
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()


def write_smiles_meta(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "input_type": "smiles",
                "input_smiles": "C(C)O",
                "input_canonical_smiles": "CCO",
                "canonical_smiles": "CCO",
                "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
            }
        )
    )


def write_sdf_meta(path: Path, input_sdf: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "input_type": "sdf",
                "input_sdf": str(input_sdf),
                "canonical_smiles": "CCO",
                "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
            }
        )
    )


def run_script(
    script: str,
    args: list[str],
    tmp_path: Path,
    *,
    empty_path: bool = False,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if empty_path:
        empty = tmp_path / "empty_path"
        empty.mkdir(exist_ok=True)
        env["PATH"] = str(empty)
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_standardize_smiles_records_input_provenance(tmp_path: Path) -> None:
    res = run_script(
        "stage1_standardize.py",
        [
            "--smiles",
            "C(C)O",
            "--out-sdf",
            str(tmp_path / "out.sdf"),
            "--out-meta",
            str(tmp_path / "out.json"),
        ],
        tmp_path,
    )

    assert res.returncode == 0, res.stderr
    payload = json.loads((tmp_path / "out.json").read_text())
    assert payload["input_type"] == "smiles"
    assert payload["input_smiles"] == "C(C)O"
    assert payload["input_canonical_smiles"] == "CCO"
    assert payload["canonical_smiles"] == "CCO"
    assert payload["inchikey"] == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"


def test_standardize_sdf_records_input_provenance(tmp_path: Path) -> None:
    input_sdf = tmp_path / "in.sdf"
    write_input_sdf(input_sdf)

    res = run_script(
        "stage1_standardize.py",
        [
            "--sdf-in",
            str(input_sdf),
            "--out-sdf",
            str(tmp_path / "out.sdf"),
            "--out-meta",
            str(tmp_path / "out.json"),
        ],
        tmp_path,
    )

    assert res.returncode == 0, res.stderr
    payload = json.loads((tmp_path / "out.json").read_text())
    assert payload["input_type"] == "sdf"
    assert payload["input_sdf"] == str(input_sdf)
    assert "input_smiles" not in payload
    assert "input_canonical_smiles" not in payload
    assert payload["canonical_smiles"] == "CCO"
    assert payload["inchikey"] == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"


def assert_xtb_identity_metadata(tmp_path: Path) -> None:
    meta = json.loads((tmp_path / "out.json").read_text())
    assert meta["canonical_smiles"] == "CCO"
    assert "[H]" not in meta["canonical_smiles"]
    assert meta["inchikey"] == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
    assert (tmp_path / "out.inchikey").read_text().strip() == meta["inchikey"]


def test_xtb_missing_fails_without_explicit_fallback(tmp_path: Path) -> None:
    input_sdf = tmp_path / "in.sdf"
    input_meta = tmp_path / "in.json"
    write_input_sdf(input_sdf)
    write_smiles_meta(input_meta)
    (tmp_path / "out.sdf").write_text("stale\n")
    (tmp_path / "out.inchikey").write_text("stale\n")
    (tmp_path / "out.json").write_text("stale\n")

    res = run_script(
        "stage1_xtb_opt.py",
        [
            "--in-sdf",
            str(input_sdf),
            "--in-meta",
            str(input_meta),
            "--out-sdf",
            str(tmp_path / "out.sdf"),
            "--out-inchikey",
            str(tmp_path / "out.inchikey"),
            "--out-meta",
            str(tmp_path / "out.json"),
        ],
        tmp_path,
        empty_path=True,
    )

    assert res.returncode != 0
    assert "xtb is required" in res.stderr
    assert not (tmp_path / "out.sdf").exists()
    assert not (tmp_path / "out.inchikey").exists()
    assert not (tmp_path / "out.json").exists()


def test_xtb_missing_fallback_is_explicit_and_recorded(tmp_path: Path) -> None:
    input_sdf = tmp_path / "in.sdf"
    input_meta = tmp_path / "in.json"
    write_input_sdf(input_sdf)
    write_smiles_meta(input_meta)

    res = run_script(
        "stage1_xtb_opt.py",
        [
            "--in-sdf",
            str(input_sdf),
            "--in-meta",
            str(input_meta),
            "--out-sdf",
            str(tmp_path / "out.sdf"),
            "--out-inchikey",
            str(tmp_path / "out.inchikey"),
            "--out-meta",
            str(tmp_path / "out.json"),
            "--allow-mmff-fallback",
        ],
        tmp_path,
        empty_path=True,
    )

    assert res.returncode == 0, res.stderr
    meta = json.loads((tmp_path / "out.json").read_text())
    assert meta["xtb_status"] == "fallback_mmff"
    assert meta["xtb_error_tail"] == "xtb_not_found"
    assert meta["input_type"] == "smiles"
    assert meta["input_smiles"] == "C(C)O"
    assert meta["input_canonical_smiles"] == "CCO"
    assert "input_sdf" not in meta
    assert_xtb_identity_metadata(tmp_path)


def test_xtb_missing_fallback_preserves_sdf_input_provenance(tmp_path: Path) -> None:
    input_sdf = tmp_path / "in.sdf"
    input_meta = tmp_path / "in.json"
    write_input_sdf(input_sdf)
    write_sdf_meta(input_meta, input_sdf)

    res = run_script(
        "stage1_xtb_opt.py",
        [
            "--in-sdf",
            str(input_sdf),
            "--in-meta",
            str(input_meta),
            "--out-sdf",
            str(tmp_path / "out.sdf"),
            "--out-inchikey",
            str(tmp_path / "out.inchikey"),
            "--out-meta",
            str(tmp_path / "out.json"),
            "--allow-mmff-fallback",
        ],
        tmp_path,
        empty_path=True,
    )

    assert res.returncode == 0, res.stderr
    meta = json.loads((tmp_path / "out.json").read_text())
    assert meta["input_type"] == "sdf"
    assert meta["input_sdf"] == str(input_sdf)
    assert "input_smiles" not in meta
    assert "input_canonical_smiles" not in meta
    assert_xtb_identity_metadata(tmp_path)


def test_xtb_rejects_missing_input_type_metadata(tmp_path: Path) -> None:
    input_sdf = tmp_path / "in.sdf"
    input_meta = tmp_path / "in.json"
    write_input_sdf(input_sdf)
    input_meta.write_text(json.dumps({"canonical_smiles": "CCO"}))
    for name in ("out.sdf", "out.inchikey", "out.json"):
        (tmp_path / name).write_text("stale\n")

    res = run_script(
        "stage1_xtb_opt.py",
        [
            "--in-sdf",
            str(input_sdf),
            "--in-meta",
            str(input_meta),
            "--out-sdf",
            str(tmp_path / "out.sdf"),
            "--out-inchikey",
            str(tmp_path / "out.inchikey"),
            "--out-meta",
            str(tmp_path / "out.json"),
            "--allow-mmff-fallback",
        ],
        tmp_path,
        empty_path=True,
    )

    assert res.returncode != 0
    assert "Input metadata missing non-empty input_type" in res.stderr
    assert not (tmp_path / "out.sdf").exists()
    assert not (tmp_path / "out.inchikey").exists()
    assert not (tmp_path / "out.json").exists()


def test_xtb_rejects_conflicting_sdf_input_provenance(tmp_path: Path) -> None:
    input_sdf = tmp_path / "in.sdf"
    input_meta = tmp_path / "in.json"
    write_input_sdf(input_sdf)
    input_meta.write_text(
        json.dumps(
            {
                "input_type": "sdf",
                "input_sdf": str(input_sdf),
                "input_smiles": "CCO",
                "canonical_smiles": "CCO",
            }
        )
    )

    res = run_script(
        "stage1_xtb_opt.py",
        [
            "--in-sdf",
            str(input_sdf),
            "--in-meta",
            str(input_meta),
            "--out-sdf",
            str(tmp_path / "out.sdf"),
            "--out-inchikey",
            str(tmp_path / "out.inchikey"),
            "--out-meta",
            str(tmp_path / "out.json"),
            "--allow-mmff-fallback",
        ],
        tmp_path,
        empty_path=True,
    )

    assert res.returncode != 0
    assert "input_type 'sdf' cannot include input_smiles" in res.stderr
    assert not (tmp_path / "out.sdf").exists()
    assert not (tmp_path / "out.inchikey").exists()
    assert not (tmp_path / "out.json").exists()


def test_xtb_rejects_mismatched_smiles_input_canonical(tmp_path: Path) -> None:
    input_sdf = tmp_path / "in.sdf"
    input_meta = tmp_path / "in.json"
    write_input_sdf(input_sdf)
    input_meta.write_text(
        json.dumps(
            {
                "input_type": "smiles",
                "input_smiles": "C(C)O",
                "input_canonical_smiles": "CCN",
                "canonical_smiles": "CCO",
            }
        )
    )

    res = run_script(
        "stage1_xtb_opt.py",
        [
            "--in-sdf",
            str(input_sdf),
            "--in-meta",
            str(input_meta),
            "--out-sdf",
            str(tmp_path / "out.sdf"),
            "--out-inchikey",
            str(tmp_path / "out.inchikey"),
            "--out-meta",
            str(tmp_path / "out.json"),
            "--allow-mmff-fallback",
        ],
        tmp_path,
        empty_path=True,
    )

    assert res.returncode != 0
    assert "input_canonical_smiles does not match input_smiles" in res.stderr
    assert not (tmp_path / "out.sdf").exists()
    assert not (tmp_path / "out.inchikey").exists()
    assert not (tmp_path / "out.json").exists()


def test_etkdg_retains_nonconverged_mmff_conformers(tmp_path: Path) -> None:
    input_sdf = tmp_path / "in.sdf"
    output_sdf = tmp_path / "out.sdf"
    write_input_sdf(input_sdf)

    res = run_script(
        "stage1_etkdg.py",
        [
            "--in-sdf",
            str(input_sdf),
            "--out-sdf",
            str(output_sdf),
            "--n-conf",
            "3",
            "--max-iters",
            "0",
        ],
        tmp_path,
    )

    assert res.returncode == 0, res.stderr
    supplier = Chem.SDMolSupplier(str(output_sdf), removeHs=False)
    mols = [mol for mol in supplier if mol is not None]
    assert mols
    assert {mol.GetProp("mmff_status") for mol in mols} == {"not_converged"}


def test_etkdg_skips_conformers_without_forcefield(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script_module("stage1_etkdg.py")
    input_sdf = tmp_path / "in.sdf"
    output_sdf = tmp_path / "out.sdf"
    write_input_sdf(input_sdf)
    output_sdf.write_text("stale\n")
    monkeypatch.setattr(
        module.AllChem,
        "MMFFGetMoleculeForceField",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        module.AllChem,
        "UFFGetMoleculeForceField",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage1_etkdg.py",
            "--in-sdf",
            str(input_sdf),
            "--out-sdf",
            str(output_sdf),
            "--n-conf",
            "1",
        ],
    )

    with pytest.raises(SystemExit, match="MMFF and UFF failed to optimize"):
        module.main()

    assert not output_sdf.exists()


def test_etkdg_falls_back_to_uff_when_mmff_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script_module("stage1_etkdg.py")
    input_sdf = tmp_path / "in.sdf"
    output_sdf = tmp_path / "out.sdf"
    write_input_sdf(input_sdf)
    output_sdf.write_text("stale\n")
    monkeypatch.setattr(
        module.AllChem,
        "MMFFGetMoleculeForceField",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage1_etkdg.py",
            "--in-sdf",
            str(input_sdf),
            "--out-sdf",
            str(output_sdf),
            "--n-conf",
            "2",
        ],
    )

    module.main()

    mols = [mol for mol in Chem.SDMolSupplier(str(output_sdf), removeHs=False) if mol]
    assert mols
    assert {mol.GetProp("ff_kind") for mol in mols} == {"UFF"}
    assert all(mol.HasProp("ff_status") for mol in mols)


def test_xtb_empty_opt_xyz_fails_without_explicit_fallback(tmp_path: Path) -> None:
    input_sdf = tmp_path / "in.sdf"
    input_meta = tmp_path / "in.json"
    write_input_sdf(input_sdf)
    write_smiles_meta(input_meta)
    for name in ("out.sdf", "out.inchikey", "out.json"):
        (tmp_path / name).write_text("stale\n")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_xtb = fake_bin / "xtb"
    fake_xtb.write_text("#!/bin/sh\n: > xtbopt.xyz\nexit 0\n")
    fake_xtb.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = str(fake_bin)

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage1_xtb_opt.py"),
            "--in-sdf", str(input_sdf),
            "--in-meta", str(input_meta),
            "--out-sdf", str(tmp_path / "out.sdf"),
            "--out-inchikey", str(tmp_path / "out.inchikey"),
            "--out-meta", str(tmp_path / "out.json"),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "produced no valid xtbopt.xyz" in res.stderr
    assert not (tmp_path / "out.sdf").exists()
    assert not (tmp_path / "out.inchikey").exists()
    assert not (tmp_path / "out.json").exists()


def test_xtb_empty_opt_xyz_fallback_is_explicit_and_recorded(tmp_path: Path) -> None:
    input_sdf = tmp_path / "in.sdf"
    input_meta = tmp_path / "in.json"
    write_input_sdf(input_sdf)
    write_smiles_meta(input_meta)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_xtb = fake_bin / "xtb"
    fake_xtb.write_text("#!/bin/sh\n: > xtbopt.xyz\nexit 0\n")
    fake_xtb.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = str(fake_bin)

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage1_xtb_opt.py"),
            "--in-sdf", str(input_sdf),
            "--in-meta", str(input_meta),
            "--out-sdf", str(tmp_path / "out.sdf"),
            "--out-inchikey", str(tmp_path / "out.inchikey"),
            "--out-meta", str(tmp_path / "out.json"),
            "--allow-mmff-fallback",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode == 0, res.stderr
    meta = json.loads((tmp_path / "out.json").read_text())
    assert meta["xtb_status"] == "fallback_mmff"
    assert meta["xtb_error_tail"] == "invalid_xtbopt_xyz"
    assert_xtb_identity_metadata(tmp_path)


def test_protonation_missing_dependency_fails_without_fallback(tmp_path: Path) -> None:
    input_sdf = tmp_path / "in.sdf"
    write_input_sdf(input_sdf)
    (tmp_path / "out.sdf").write_text("stale\n")

    shim_dir = tmp_path / "shim"
    shim_dir.mkdir()
    (shim_dir / "dimorphite_dl.py").write_text("raise ImportError('forced missing')\n")
    env_pythonpath = os.environ.get("PYTHONPATH", "")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{shim_dir}{os.pathsep}{env_pythonpath}" if env_pythonpath else str(shim_dir)
    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage1_protonate.py"),
            "--in-sdf",
            str(input_sdf),
            "--out-sdf",
            str(tmp_path / "out.sdf"),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "dimorphite-dl is required" in res.stderr
    assert not (tmp_path / "out.sdf").exists()


def test_standardize_invalid_smiles_removes_stale_outputs(tmp_path: Path) -> None:
    (tmp_path / "out.sdf").write_text("stale\n")
    (tmp_path / "out.json").write_text("stale\n")

    res = run_script(
        "stage1_standardize.py",
        [
            "--smiles",
            "not-a-smiles",
            "--out-sdf",
            str(tmp_path / "out.sdf"),
            "--out-meta",
            str(tmp_path / "out.json"),
        ],
        tmp_path,
    )

    assert res.returncode != 0
    assert "Could not parse SMILES" in res.stderr
    assert not (tmp_path / "out.sdf").exists()
    assert not (tmp_path / "out.json").exists()


def test_etkdg_unparseable_input_removes_stale_output(tmp_path: Path) -> None:
    input_sdf = tmp_path / "in.sdf"
    input_sdf.write_text("not an sdf\n")
    (tmp_path / "out.sdf").write_text("stale\n")

    res = run_script(
        "stage1_etkdg.py",
        [
            "--in-sdf",
            str(input_sdf),
            "--out-sdf",
            str(tmp_path / "out.sdf"),
        ],
        tmp_path,
    )

    assert res.returncode != 0
    assert "No parseable mol" in res.stderr
    assert not (tmp_path / "out.sdf").exists()


def test_protonation_missing_dependency_fallback_is_recorded(tmp_path: Path) -> None:
    input_sdf = tmp_path / "in.sdf"
    write_input_sdf(input_sdf)

    shim_dir = tmp_path / "shim"
    shim_dir.mkdir()
    (shim_dir / "dimorphite_dl.py").write_text("raise ImportError('forced missing')\n")
    env_pythonpath = os.environ.get("PYTHONPATH", "")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{shim_dir}{os.pathsep}{env_pythonpath}" if env_pythonpath else str(shim_dir)
    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage1_protonate.py"),
            "--in-sdf",
            str(input_sdf),
            "--out-sdf",
            str(tmp_path / "out.sdf"),
            "--allow-unprotonated-fallback",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode == 0, res.stderr
    mol = next(m for m in Chem.SDMolSupplier(str(tmp_path / "out.sdf"), removeHs=False) if m)
    assert mol.GetProp("protonation_method") == "fallback_unprotonated"
    assert mol.GetProp("protonation_fallback") == "yes"
    assert mol.GetProp("protonation_fallback_reason") == "dimorphite_dl_unavailable"


def test_protonation_rejects_implausible_amide_n_anion(tmp_path: Path) -> None:
    input_sdf = tmp_path / "niacinamide.sdf"
    mol = Chem.MolFromSmiles("NC(=O)c1cccnc1")
    AllChem.Compute2DCoords(mol)
    writer = Chem.SDWriter(str(input_sdf))
    writer.write(mol)
    writer.close()

    shim_dir = tmp_path / "shim"
    shim_dir.mkdir()
    (shim_dir / "dimorphite_dl.py").write_text(
        "def protonate_smiles(*args, **kwargs):\n"
        "    return ['[NH-]C(=O)c1cccnc1', 'NC(=O)c1cccnc1']\n"
    )
    env_pythonpath = os.environ.get("PYTHONPATH", "")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{shim_dir}{os.pathsep}{env_pythonpath}" if env_pythonpath else str(shim_dir)
    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage1_protonate.py"),
            "--in-sdf",
            str(input_sdf),
            "--out-sdf",
            str(tmp_path / "out.sdf"),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode == 0, res.stderr
    out_mol = next(
        m for m in Chem.SDMolSupplier(str(tmp_path / "out.sdf"), removeHs=False) if m
    )
    assert out_mol.GetProp("protonated_smiles") == "NC(=O)c1cccnc1"
    assert "[N-]" not in Chem.MolToSmiles(out_mol, canonical=True)


def test_standardize_preserves_tautomer_identity_matching_the_index() -> None:
    """인덱스·참조는 토토머를 별도 구조로 유지한다. stage1이 합치면 신원이 갈린다.

    ilomastat 하이드록삼산: 인덱스 키는 C(=O)NO 형태(-DYVFJYSZSA-), 토토머
    정규화를 거치면 C(O)=NO 형태(-GUYCJALGSA-)가 되어 실행 경로만 구조가 달라진다.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_activity_retrieval_index import _standardize_mol
    from stage1_standardize import standardize as stage1_standardize

    ilomastat = "CNC(=O)[C@H](Cc1c[nH]c2ccccc12)NC(=O)[C@@H](CC(=O)NO)CC(C)C"
    index_key = Chem.MolToInchiKey(_standardize_mol(ilomastat))
    staged = stage1_standardize(Chem.MolFromSmiles(ilomastat))

    assert index_key == "NITYDPDXAAFEIT-DYVFJYSZSA-N"
    assert Chem.MolToInchiKey(staged) == index_key


def test_etkdg_falls_back_to_random_coordinates(monkeypatch) -> None:
    """기본 ETKDG가 실패해도 카이랄리티를 건드리지 않고 좌표 방식만 바꿔 재시도한다."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import stage1_etkdg as mod

    calls = {"n": 0}

    def fake(mol, numConfs, params):  # noqa: ANN001
        calls["n"] += 1
        if calls["n"] == 1:
            return []
        assert params.useRandomCoords is True
        return [0]

    monkeypatch.setattr(mod.AllChem, "EmbedMultipleConfs", fake)
    cids, method = mod._embed(Chem.AddHs(Chem.MolFromSmiles("CCO")), mod.AllChem.ETKDGv3(), 5)

    assert cids == [0]
    assert method == "etkdgv3_random_coords"
    assert calls["n"] == 2
