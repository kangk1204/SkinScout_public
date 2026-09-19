"""Regression tests for fail-closed downstream scientific stages."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]


def write_valid_ligand_sdf(path: Path) -> None:
    """3D 좌표와 **수소**를 가진 리간드. MD 가 받는 것과 같은 모양이다.

    Stage 1 의 xTB 최적화 산출물에는 수소가 있고, MD 준비는 그것을 요구한다.
    수소 없는 SDF 로도 MD 는 끝까지 돌지만 MM-GBSA 가 몇 시간 뒤 "energy terms
    undefined" 로 멈추기 때문이다. 대역이 수소를 빼면 그 검사를 시험할 수 없다.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    assert mol is not None
    AllChem.EmbedMolecule(mol, randomSeed=7)
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()


def trajectory_index_row(
    target_dir: Path,
    *,
    target_id: str,
    replica: int = 1,
    trajectory_body: str = "trajectory\n",
) -> dict[str, object]:
    trajectory = target_dir / f"prod_r{replica}.xtc"
    tpr = target_dir / f"prod_r{replica}.tpr"
    topology = target_dir / "topol.top"
    trajectory.write_text(trajectory_body)
    tpr.write_text("tpr\n")
    topology.write_text("topology\n")
    return {
        "target_id": target_id,
        "replica": replica,
        "trajectory_xtc": str(trajectory),
        "trajectory_xtc_sha256": hashlib.sha256(trajectory.read_bytes()).hexdigest(),
        "tpr": str(tpr),
        "tpr_sha256": hashlib.sha256(tpr.read_bytes()).hexdigest(),
        "topology_top": str(topology),
        "topology_top_sha256": hashlib.sha256(topology.read_bytes()).hexdigest(),
        "status": "ok",
    }


def ready_md_manifest_row(target_dir: Path, *, target_id: str) -> dict[str, str]:
    paths = {
        "system_gro": target_dir / "system.gro",
        "topology_top": target_dir / "topol.top",
        "tpr": target_dir / "prod.tpr",
        "npt_gro": target_dir / "npt.gro",
        "prod_mdp": target_dir / "prod.mdp",
    }
    defaults = {
        "system_gro": "system\n",
        "topology_top": "topology\n",
        "tpr": "tpr\n",
        "npt_gro": "equilibrated\n",
        "prod_mdp": "nsteps = 10\ngen_vel = no\n",
    }
    for name, path in paths.items():
        if not path.exists():
            path.write_text(defaults[name])
    return {
        "target_id": target_id,
        **{name: str(path) for name, path in paths.items()},
        "status": "ready",
        **{
            f"{name}_sha256": hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in paths.items()
        },
    }


def run_script(script: str, args: list[str], tmp_path: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PATH"] = str(tmp_path / "empty_path")
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def load_script_module(script: str):
    spec = importlib.util.spec_from_file_location(
        script.removesuffix(".py"),
        ROOT / "scripts" / script,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_stage4_fails_on_empty_target_list(tmp_path: Path) -> None:
    top = tmp_path / "top.csv"
    pd.DataFrame(columns=["target_id", "rrf_score"]).to_csv(top, index=False)
    res = run_script(
        "stage4_prepare_structures.py",
        [
            "--top-csv", str(top),
            "--clean-dir", str(tmp_path / "clean"),
            "--pocket-dir", str(tmp_path / "pockets"),
            "--cutoff-date", "2023-10-01",
            "--out-dir", str(tmp_path / "out"),
            "--out-manifest", str(tmp_path / "manifest.tsv"),
        ],
        tmp_path,
    )
    assert res.returncode != 0
    assert "No Stage 3 targets" in res.stderr


def test_stage4_fails_on_empty_top_csv_without_stale_manifest(tmp_path: Path) -> None:
    top = tmp_path / "top.csv"
    top.write_text("")
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("stale\n")

    res = run_script(
        "stage4_prepare_structures.py",
        [
            "--top-csv", str(top),
            "--clean-dir", str(tmp_path / "clean"),
            "--pocket-dir", str(tmp_path / "pockets"),
            "--cutoff-date", "2023-10-01",
            "--out-dir", str(tmp_path / "out"),
            "--out-manifest", str(manifest),
        ],
        tmp_path,
    )

    assert res.returncode != 0
    assert "Stage 4 top target CSV is missing or empty" in res.stderr
    assert not manifest.exists()


def test_stage4_rejects_top_csv_missing_target_id(tmp_path: Path) -> None:
    top = tmp_path / "top.csv"
    pd.DataFrame([{"wrong_id": "P1"}]).to_csv(top, index=False)
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("stale\n")

    res = run_script(
        "stage4_prepare_structures.py",
        [
            "--top-csv", str(top),
            "--clean-dir", str(tmp_path / "clean"),
            "--pocket-dir", str(tmp_path / "pockets"),
            "--cutoff-date", "2023-10-01",
            "--out-dir", str(tmp_path / "out"),
            "--out-manifest", str(manifest),
        ],
        tmp_path,
    )

    assert res.returncode != 0
    assert "Stage 4 top target CSV missing required column: target_id" in res.stderr
    assert not manifest.exists()


def test_stage4_rejects_blank_target_ids_without_stale_manifest(tmp_path: Path) -> None:
    top = tmp_path / "top.csv"
    pd.DataFrame([{"target_id": "P1"}, {"target_id": "   "}]).to_csv(
        top,
        index=False,
    )
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("stale\n")

    res = run_script(
        "stage4_prepare_structures.py",
        [
            "--top-csv", str(top),
            "--clean-dir", str(tmp_path / "clean"),
            "--pocket-dir", str(tmp_path / "pockets"),
            "--cutoff-date", "2023-10-01",
            "--out-dir", str(tmp_path / "out"),
            "--out-manifest", str(manifest),
        ],
        tmp_path,
    )

    assert res.returncode != 0
    assert "Stage 4 top target CSV column 'target_id' contains blank values" in res.stderr
    assert not manifest.exists()


def test_stage4_rejects_duplicate_target_ids_without_stale_manifest(
    tmp_path: Path,
) -> None:
    top = tmp_path / "top.csv"
    pd.DataFrame([{"target_id": "P1"}, {"target_id": "P1"}]).to_csv(
        top,
        index=False,
    )
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("stale\n")

    res = run_script(
        "stage4_prepare_structures.py",
        [
            "--top-csv", str(top),
            "--clean-dir", str(tmp_path / "clean"),
            "--pocket-dir", str(tmp_path / "pockets"),
            "--cutoff-date", "2023-10-01",
            "--out-dir", str(tmp_path / "out"),
            "--out-manifest", str(manifest),
        ],
        tmp_path,
    )

    assert res.returncode != 0
    assert (
        "Stage 4 top target CSV contains duplicate target_id values: P1"
        in res.stderr
    )
    assert not manifest.exists()


def test_stage4_rejects_empty_cleaned_structure_without_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = load_script_module("stage4_prepare_structures.py")
    monkeypatch.setattr(module, "fetch_holo_candidates", lambda *_args, **_kwargs: [])

    top = tmp_path / "top.csv"
    pd.DataFrame([{"target_id": "P1", "rrf_score": 1.0}]).to_csv(top, index=False)
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "P1_clean.pdb").write_text("")
    pockets = tmp_path / "pockets"
    pockets.mkdir()
    (pockets / "P1.pockets.json").write_text('{"pockets": []}\n')
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("stale\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage4_prepare_structures.py",
            "--top-csv", str(top),
            "--clean-dir", str(clean),
            "--pocket-dir", str(pockets),
            "--cutoff-date", "2023-10-01",
            "--out-dir", str(tmp_path / "out"),
            "--out-manifest", str(manifest),
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        module.main()

    assert "Missing or empty cleaned AlphaFold structure for P1" in str(excinfo.value)
    assert not manifest.exists()
    assert not (tmp_path / "out" / "P1" / "P1_input.pdb").exists()


def test_stage4_rejects_empty_pocket_manifest_without_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = load_script_module("stage4_prepare_structures.py")
    monkeypatch.setattr(module, "fetch_holo_candidates", lambda *_args, **_kwargs: [])

    top = tmp_path / "top.csv"
    pd.DataFrame([{"target_id": "P1", "rrf_score": 1.0}]).to_csv(top, index=False)
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "P1_clean.pdb").write_text(
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 90.00           C\n"
        "END\n"
    )
    pockets = tmp_path / "pockets"
    pockets.mkdir()
    (pockets / "P1.pockets.json").write_text("")
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("stale\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage4_prepare_structures.py",
            "--top-csv", str(top),
            "--clean-dir", str(clean),
            "--pocket-dir", str(pockets),
            "--cutoff-date", "2023-10-01",
            "--out-dir", str(tmp_path / "out"),
            "--out-manifest", str(manifest),
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        module.main()

    assert "Missing or empty pocket manifest for P1" in str(excinfo.value)
    assert not manifest.exists()
    assert not (tmp_path / "out" / "P1" / "P1_input.pdb").exists()
    assert not (tmp_path / "out" / "P1" / "pocket_box.json").exists()


def test_stage4_rejects_invalid_pocket_manifest_json_without_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = load_script_module("stage4_prepare_structures.py")
    monkeypatch.setattr(module, "fetch_holo_candidates", lambda *_args, **_kwargs: [])

    top = tmp_path / "top.csv"
    pd.DataFrame([{"target_id": "P1", "rrf_score": 1.0}]).to_csv(top, index=False)
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "P1_clean.pdb").write_text(
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 90.00           C\n"
        "END\n"
    )
    pockets = tmp_path / "pockets"
    pockets.mkdir()
    (pockets / "P1.pockets.json").write_text("{not-json\n")
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("stale\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage4_prepare_structures.py",
            "--top-csv", str(top),
            "--clean-dir", str(clean),
            "--pocket-dir", str(pockets),
            "--cutoff-date", "2023-10-01",
            "--out-dir", str(tmp_path / "out"),
            "--out-manifest", str(manifest),
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        module.main()

    assert "Invalid pocket manifest JSON for P1" in str(excinfo.value)
    assert not manifest.exists()
    assert not (tmp_path / "out" / "P1" / "P1_input.pdb").exists()
    assert not (tmp_path / "out" / "P1" / "pocket_box.json").exists()


def test_stage4_partial_prepare_failure_does_not_leave_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = load_script_module("stage4_prepare_structures.py")
    monkeypatch.setattr(module, "fetch_holo_candidates", lambda *_args, **_kwargs: [])

    top = tmp_path / "top.csv"
    pd.DataFrame(
        [
            {"target_id": "P1", "rrf_score": 1.0},
            {"target_id": "P2", "rrf_score": 0.9},
        ]
    ).to_csv(top, index=False)
    clean = tmp_path / "clean"
    clean.mkdir()
    pockets = tmp_path / "pockets"
    pockets.mkdir()
    for uid in ("P1", "P2"):
        (clean / f"{uid}_clean.pdb").write_text(
            "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 90.00           C\n"
            "END\n"
        )
    (pockets / "P1.pockets.json").write_text('{"pockets": []}\n')
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("stale\n")
    stale_target = tmp_path / "out" / "P1"
    stale_target.mkdir(parents=True)
    (stale_target / "P1_input.pdb").write_text("STALE\n")
    (stale_target / "pocket_box.json").write_text('{"stale": true}\n')
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage4_prepare_structures.py",
            "--top-csv", str(top),
            "--clean-dir", str(clean),
            "--pocket-dir", str(pockets),
            "--cutoff-date", "2023-10-01",
            "--out-dir", str(tmp_path / "out"),
            "--out-manifest", str(manifest),
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        module.main()

    assert "Missing or empty pocket manifest for P2" in str(excinfo.value)
    assert not manifest.exists()
    assert not (tmp_path / "out" / "P1" / "P1_input.pdb").exists()
    assert not (tmp_path / "out" / "P1" / "pocket_box.json").exists()
    assert not (tmp_path / "out" / "P2" / "P2_input.pdb").exists()
    assert not (tmp_path / "out" / "P2" / "pocket_box.json").exists()


def test_stage5_fails_when_boltz_missing(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.tsv"
    pd.DataFrame(
        [{"target_id": "P1", "source": "alphafold_cleaned", "input_pdb": "x.pdb"}]
    ).to_csv(manifest, sep="\t", index=False)
    ligand = tmp_path / "ligand.sdf"
    ligand.write_text("")
    res = run_script(
        "stage5_boltz2.py",
        [
            "--manifest", str(manifest),
            "--ligand-sdf", str(ligand),
            "--struct-dir", str(tmp_path),
            "--out-dir", str(tmp_path / "out"),
            "--out-report", str(tmp_path / "report.tsv"),
        ],
        tmp_path,
    )
    assert res.returncode != 0
    assert "boltz is not available" in res.stderr


def test_stage5_rejects_empty_manifest_without_stale_report(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("")
    ligand = tmp_path / "ligand.sdf"
    write_valid_ligand_sdf(ligand)
    report = tmp_path / "report.tsv"
    report.write_text("stale\n")
    tmp_report = report.with_suffix(report.suffix + ".tmp")
    tmp_report.write_text("stale temporary report\n")

    res = run_script(
        "stage5_boltz2.py",
        [
            "--manifest", str(manifest),
            "--ligand-sdf", str(ligand),
            "--struct-dir", str(tmp_path),
            "--out-dir", str(tmp_path / "out"),
            "--out-report", str(report),
        ],
        tmp_path,
    )

    assert res.returncode != 0
    assert "Stage 4 manifest is required and must be non-empty" in res.stderr
    assert not report.exists()
    assert not tmp_report.exists()


def test_stage5_rejects_manifest_missing_required_columns(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.tsv"
    pd.DataFrame([{"target_id": "P1", "input_pdb": "x.pdb"}]).to_csv(
        manifest, sep="\t", index=False
    )
    ligand = tmp_path / "ligand.sdf"
    write_valid_ligand_sdf(ligand)
    report = tmp_path / "report.tsv"
    report.write_text("stale\n")

    res = run_script(
        "stage5_boltz2.py",
        [
            "--manifest", str(manifest),
            "--ligand-sdf", str(ligand),
            "--struct-dir", str(tmp_path),
            "--out-dir", str(tmp_path / "out"),
            "--out-report", str(report),
        ],
        tmp_path,
    )

    assert res.returncode != 0
    assert "Stage 4 manifest missing required columns" in res.stderr
    assert not report.exists()


def test_stage5_rejects_manifest_without_rows(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.tsv"
    pd.DataFrame(columns=["target_id", "source", "input_pdb"]).to_csv(
        manifest, sep="\t", index=False
    )
    ligand = tmp_path / "ligand.sdf"
    write_valid_ligand_sdf(ligand)
    report = tmp_path / "report.tsv"
    report.write_text("stale\n")

    res = run_script(
        "stage5_boltz2.py",
        [
            "--manifest", str(manifest),
            "--ligand-sdf", str(ligand),
            "--struct-dir", str(tmp_path),
            "--out-dir", str(tmp_path / "out"),
            "--out-report", str(report),
        ],
        tmp_path,
    )

    assert res.returncode != 0
    assert "Stage 4 manifest contains no rows" in res.stderr
    assert not report.exists()


def test_stage5_rejects_manifest_blank_required_fields(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.tsv"
    pd.DataFrame([
        {"target_id": "P1", "source": "alphafold_cleaned", "input_pdb": "x.pdb"},
        {"target_id": None, "source": "alphafold_cleaned", "input_pdb": "y.pdb"},
    ]).to_csv(manifest, sep="\t", index=False)
    ligand = tmp_path / "ligand.sdf"
    write_valid_ligand_sdf(ligand)
    report = tmp_path / "report.tsv"
    report.write_text("stale\n")

    res = run_script(
        "stage5_boltz2.py",
        [
            "--manifest", str(manifest),
            "--ligand-sdf", str(ligand),
            "--struct-dir", str(tmp_path),
            "--out-dir", str(tmp_path / "out"),
            "--out-report", str(report),
        ],
        tmp_path,
    )

    assert res.returncode != 0
    assert "Stage 4 manifest column 'target_id' contains blank values" in res.stderr
    assert not report.exists()


def test_stage5_rejects_manifest_duplicate_target_ids(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.tsv"
    pd.DataFrame([
        {"target_id": "P1", "source": "alphafold_cleaned", "input_pdb": "x.pdb"},
        {"target_id": "P1", "source": "alphafold_cleaned", "input_pdb": "y.pdb"},
    ]).to_csv(manifest, sep="\t", index=False)
    ligand = tmp_path / "ligand.sdf"
    write_valid_ligand_sdf(ligand)
    report = tmp_path / "report.tsv"
    report.write_text("stale\n")

    res = run_script(
        "stage5_boltz2.py",
        [
            "--manifest", str(manifest),
            "--ligand-sdf", str(ligand),
            "--struct-dir", str(tmp_path),
            "--out-dir", str(tmp_path / "out"),
            "--out-report", str(report),
        ],
        tmp_path,
    )

    assert res.returncode != 0
    assert "Stage 4 manifest contains duplicate target_id values: P1" in res.stderr
    assert not report.exists()


def test_stage5_rejects_manifest_duplicate_input_pdbs(tmp_path: Path) -> None:
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text(
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 90.00           C\n"
        "END\n"
    )
    manifest = tmp_path / "manifest.tsv"
    pd.DataFrame(
        [
            {
                "target_id": "P1",
                "source": "alphafold_cleaned",
                "input_pdb": str(receptor),
            },
            {
                "target_id": "P2",
                "source": "alphafold_cleaned",
                "input_pdb": str(receptor),
            },
        ]
    ).to_csv(manifest, sep="\t", index=False)
    ligand = tmp_path / "ligand.sdf"
    write_valid_ligand_sdf(ligand)
    report = tmp_path / "report.tsv"
    report.write_text("stale\n")

    res = run_script(
        "stage5_boltz2.py",
        [
            "--manifest", str(manifest),
            "--ligand-sdf", str(ligand),
            "--struct-dir", str(tmp_path),
            "--out-dir", str(tmp_path / "out"),
            "--out-report", str(report),
        ],
        tmp_path,
    )

    assert res.returncode != 0
    assert "Stage 4 manifest contains duplicate input_pdb values" in res.stderr
    assert str(receptor) in res.stderr
    assert not report.exists()


def test_stage5_rejects_manifest_canonical_duplicate_input_pdbs(
    tmp_path: Path,
) -> None:
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text(
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 90.00           C\n"
        "END\n"
    )
    manifest = tmp_path / "manifest.tsv"
    pd.DataFrame(
        [
            {
                "target_id": "P1",
                "source": "alphafold_cleaned",
                "input_pdb": str(receptor),
            },
            {
                "target_id": "P2",
                "source": "alphafold_cleaned",
                "input_pdb": f"{receptor.parent}/./{receptor.name}",
            },
        ]
    ).to_csv(manifest, sep="\t", index=False)
    ligand = tmp_path / "ligand.sdf"
    write_valid_ligand_sdf(ligand)
    report = tmp_path / "report.tsv"
    report.write_text("stale\n")

    res = run_script(
        "stage5_boltz2.py",
        [
            "--manifest", str(manifest),
            "--ligand-sdf", str(ligand),
            "--struct-dir", str(tmp_path),
            "--out-dir", str(tmp_path / "out"),
            "--out-report", str(report),
        ],
        tmp_path,
    )

    assert res.returncode != 0
    assert "Stage 4 manifest contains duplicate input_pdb values" in res.stderr
    assert str(receptor.resolve()) in res.stderr
    assert not report.exists()


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        ("--num-seeds", "0", "--num-seeds must be >= 1: 0"),
        ("--num-recycles", "0", "--num-recycles must be >= 1: 0"),
        ("--max-residues", "0", "--max-residues must be >= 1: 0"),
        ("--crop-radius", "0", "--crop-radius must be a finite value > 0: 0"),
        ("--crop-radius", "nan", "--crop-radius must be a finite value > 0: nan"),
        ("--iptm-threshold", "-0.1", "--iptm-threshold must be between 0 and 1: -0.1"),
        ("--plddt-threshold", "101", "--plddt-threshold must be between 0 and 100: 101"),
    ],
)
def test_stage5_rejects_invalid_numeric_args_without_report(
    tmp_path: Path,
    flag: str,
    value: str,
    message: str,
) -> None:
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text(
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 90.00           C\n"
        "END\n"
    )
    manifest = tmp_path / "manifest.tsv"
    pd.DataFrame(
        [{"target_id": "P1", "source": "alphafold_cleaned", "input_pdb": str(receptor)}]
    ).to_csv(manifest, sep="\t", index=False)
    ligand = tmp_path / "ligand.sdf"
    write_valid_ligand_sdf(ligand)
    report = tmp_path / "report.tsv"
    report.write_text("stale\n")
    tmp_report = report.with_suffix(report.suffix + ".tmp")
    tmp_report.write_text("stale temporary report\n")

    res = run_script(
        "stage5_boltz2.py",
        [
            "--manifest", str(manifest),
            "--ligand-sdf", str(ligand),
            "--struct-dir", str(tmp_path),
            "--out-dir", str(tmp_path / "out"),
            "--out-report", str(report),
            flag, value,
        ],
        tmp_path,
    )

    assert res.returncode != 0
    assert message in res.stderr
    assert not report.exists()
    assert not tmp_report.exists()


def test_stage5_failed_boltz_runs_do_not_leave_report(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_boltz = fake_bin / "boltz"
    fake_boltz.write_text("#!/bin/sh\nexit 0\n")
    fake_boltz.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', os.defpath)}"

    receptor = tmp_path / "receptor.pdb"
    receptor.write_text(
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 90.00           C\n"
        "END\n"
    )
    manifest = tmp_path / "manifest.tsv"
    pd.DataFrame(
        [{"target_id": "P1", "source": "alphafold_cleaned", "input_pdb": str(receptor)}]
    ).to_csv(manifest, sep="\t", index=False)
    ligand = tmp_path / "ligand.sdf"
    write_valid_ligand_sdf(ligand)
    report = tmp_path / "report.tsv"

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage5_boltz2.py"),
            "--manifest", str(manifest),
            "--ligand-sdf", str(ligand),
            "--struct-dir", str(tmp_path),
            "--out-dir", str(tmp_path / "out"),
            "--out-report", str(report),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "Boltz-2 produced no successful cofolding reports" in res.stderr
    assert not report.exists()


def test_stage5_rejects_metrics_without_top_ranked_complex_pdb(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_boltz = fake_bin / "boltz"
    fake_boltz.write_text(
        "#!/bin/sh\n"
        "out=''\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = '--out_dir' ]; then shift; out=\"$1\"; fi\n"
        "  shift\n"
        "done\n"
        "mkdir -p \"$out\"\n"
        "printf '{\"iptm\":0.8,\"complex_plddt\":90,\"affinity_pred_value\":1.2}' > \"$out/report.json\"\n"
    )
    fake_boltz.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', os.defpath)}"

    receptor = tmp_path / "receptor.pdb"
    receptor.write_text(
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 90.00           C\n"
        "END\n"
    )
    manifest = tmp_path / "manifest.tsv"
    pd.DataFrame([
        {"target_id": "P1", "source": "alphafold_cleaned", "input_pdb": str(receptor)}
    ]).to_csv(manifest, sep="\t", index=False)
    ligand = tmp_path / "ligand.sdf"
    write_valid_ligand_sdf(ligand)
    report = tmp_path / "report.tsv"
    stale_complex = tmp_path / "out" / "P1" / "complex.pdb"
    stale_complex.parent.mkdir(parents=True)
    stale_complex.write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage5_boltz2.py"),
            "--manifest", str(manifest),
            "--ligand-sdf", str(ligand),
            "--struct-dir", str(tmp_path),
            "--out-dir", str(tmp_path / "out"),
            "--out-report", str(report),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "Boltz-2 produced no successful cofolding reports" in res.stderr
    assert not report.exists()
    assert not stale_complex.exists()


def test_stage5_empty_ligand_does_not_leave_report(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_boltz = fake_bin / "boltz"
    fake_boltz.write_text(
        "#!/bin/sh\n"
        "out=''\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = '--out_dir' ]; then shift; out=\"$1\"; fi\n"
        "  shift\n"
        "done\n"
        "mkdir -p \"$out\"\n"
        "printf '{\"iptm\":0.8,\"pocket_plddt\":90,\"affinity_log_uM\":1.2,\"posebusters_valid\":true,\"n_residues\":120}' > \"$out/report.json\"\n"
        "mkdir -p \"$out/predictions/input\"\n"
        "printf 'ATOM protein\\nHETATM ligand\\n' > \"$out/predictions/input/input_model_0.pdb\"\n"
    )
    fake_boltz.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', os.defpath)}"

    receptor = tmp_path / "receptor.pdb"
    receptor.write_text(
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 90.00           C\n"
        "END\n"
    )
    manifest = tmp_path / "manifest.tsv"
    pd.DataFrame(
        [{"target_id": "P1", "source": "alphafold_cleaned", "input_pdb": str(receptor)}]
    ).to_csv(manifest, sep="\t", index=False)
    ligand = tmp_path / "ligand.sdf"
    ligand.write_text("")
    report = tmp_path / "report.tsv"
    report.write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage5_boltz2.py"),
            "--manifest", str(manifest),
            "--ligand-sdf", str(ligand),
            "--struct-dir", str(tmp_path),
            "--out-dir", str(tmp_path / "out"),
            "--out-report", str(report),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "Ligand SDF is missing or empty for Boltz-2" in res.stderr
    assert not report.exists()


def test_stage5_empty_boltz_report_does_not_leave_report(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_boltz = fake_bin / "boltz"
    fake_boltz.write_text(
        "#!/bin/sh\n"
        "out=''\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = '--out_dir' ]; then shift; out=\"$1\"; fi\n"
        "  shift\n"
        "done\n"
        "mkdir -p \"$out\"\n"
        ": > \"$out/report.json\"\n"
    )
    fake_boltz.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', os.defpath)}"

    receptor = tmp_path / "receptor.pdb"
    receptor.write_text(
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 90.00           C\n"
        "END\n"
    )
    manifest = tmp_path / "manifest.tsv"
    pd.DataFrame(
        [{"target_id": "P1", "source": "alphafold_cleaned", "input_pdb": str(receptor)}]
    ).to_csv(manifest, sep="\t", index=False)
    ligand = tmp_path / "ligand.sdf"
    write_valid_ligand_sdf(ligand)
    report = tmp_path / "report.tsv"
    report.write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage5_boltz2.py"),
            "--manifest", str(manifest),
            "--ligand-sdf", str(ligand),
            "--struct-dir", str(tmp_path),
            "--out-dir", str(tmp_path / "out"),
            "--out-report", str(report),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "Boltz-2 produced no successful cofolding reports" in res.stderr
    assert not report.exists()


def test_stage5_invalid_boltz_report_does_not_leave_report(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_boltz = fake_bin / "boltz"
    fake_boltz.write_text(
        "#!/bin/sh\n"
        "out=''\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = '--out_dir' ]; then shift; out=\"$1\"; fi\n"
        "  shift\n"
        "done\n"
        "mkdir -p \"$out\"\n"
        "printf 'not-json\\n' > \"$out/report.json\"\n"
    )
    fake_boltz.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', os.defpath)}"

    receptor = tmp_path / "receptor.pdb"
    receptor.write_text(
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 90.00           C\n"
        "END\n"
    )
    manifest = tmp_path / "manifest.tsv"
    pd.DataFrame(
        [{"target_id": "P1", "source": "alphafold_cleaned", "input_pdb": str(receptor)}]
    ).to_csv(manifest, sep="\t", index=False)
    ligand = tmp_path / "ligand.sdf"
    write_valid_ligand_sdf(ligand)
    report = tmp_path / "report.tsv"
    report.write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage5_boltz2.py"),
            "--manifest", str(manifest),
            "--ligand-sdf", str(ligand),
            "--struct-dir", str(tmp_path),
            "--out-dir", str(tmp_path / "out"),
            "--out-report", str(report),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "Boltz-2 produced no successful cofolding reports" in res.stderr
    assert not report.exists()


def test_stage5_nonfinite_boltz_metric_does_not_leave_report(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_boltz = fake_bin / "boltz"
    fake_boltz.write_text(
        "#!/bin/sh\n"
        "out=''\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = '--out_dir' ]; then shift; out=\"$1\"; fi\n"
        "  shift\n"
        "done\n"
        "mkdir -p \"$out\"\n"
        "printf '{\"iptm\":Infinity,\"pocket_plddt\":90,\"affinity_log_uM\":1.2,\"posebusters_valid\":true,\"n_residues\":120}' > \"$out/report.json\"\n"
        "mkdir -p \"$out/predictions/input\"\n"
        "printf 'ATOM protein\\nHETATM ligand\\n' > \"$out/predictions/input/input_model_0.pdb\"\n"
    )
    fake_boltz.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', os.defpath)}"

    receptor = tmp_path / "receptor.pdb"
    receptor.write_text(
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 90.00           C\n"
        "END\n"
    )
    manifest = tmp_path / "manifest.tsv"
    pd.DataFrame(
        [{"target_id": "P1", "source": "alphafold_cleaned", "input_pdb": str(receptor)}]
    ).to_csv(manifest, sep="\t", index=False)
    ligand = tmp_path / "ligand.sdf"
    write_valid_ligand_sdf(ligand)
    report = tmp_path / "report.tsv"
    report.write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage5_boltz2.py"),
            "--manifest", str(manifest),
            "--ligand-sdf", str(ligand),
            "--struct-dir", str(tmp_path),
            "--out-dir", str(tmp_path / "out"),
            "--out-report", str(report),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "Boltz-2 report for P1 metric 'iptm' must be finite" in res.stderr
    assert not report.exists()


def test_stage5_nonboolean_posebusters_flag_does_not_leave_report(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_boltz = fake_bin / "boltz"
    fake_boltz.write_text(
        "#!/bin/sh\n"
        "out=''\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = '--out_dir' ]; then shift; out=\"$1\"; fi\n"
        "  shift\n"
        "done\n"
        "mkdir -p \"$out\"\n"
        "printf '{\"iptm\":0.8,\"pocket_plddt\":90,\"affinity_log_uM\":1.2,\"posebusters_valid\":\"false\",\"n_residues\":120}' > \"$out/report.json\"\n"
        "mkdir -p \"$out/predictions/input\"\n"
        "printf 'ATOM protein\\nHETATM ligand\\n' > \"$out/predictions/input/input_model_0.pdb\"\n"
    )
    fake_boltz.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', os.defpath)}"

    receptor = tmp_path / "receptor.pdb"
    receptor.write_text(
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 90.00           C\n"
        "END\n"
    )
    manifest = tmp_path / "manifest.tsv"
    pd.DataFrame(
        [{"target_id": "P1", "source": "alphafold_cleaned", "input_pdb": str(receptor)}]
    ).to_csv(manifest, sep="\t", index=False)
    ligand = tmp_path / "ligand.sdf"
    write_valid_ligand_sdf(ligand)
    report = tmp_path / "report.tsv"
    report.write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage5_boltz2.py"),
            "--manifest", str(manifest),
            "--ligand-sdf", str(ligand),
            "--struct-dir", str(tmp_path),
            "--out-dir", str(tmp_path / "out"),
            "--out-report", str(report),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "Boltz-2 report for P1 metric 'posebusters_valid' must be boolean" in res.stderr
    assert not report.exists()


def test_stage5_derives_n_residues_when_runner_omits_it(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_boltz = fake_bin / "boltz"
    fake_boltz.write_text(
        "#!/bin/sh\n"
        "out=''\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = '--out_dir' ]; then shift; out=\"$1\"; fi\n"
        "  shift\n"
        "done\n"
        "mkdir -p \"$out\"\n"
        "printf '{\"iptm\":0.8,\"pocket_plddt\":90,\"affinity_log_uM\":1.2,\"posebusters_valid\":true}' > \"$out/report.json\"\n"
        "mkdir -p \"$out/predictions/input\"\n"
        "printf 'ATOM protein\\nHETATM ligand\\n' > \"$out/predictions/input/input_model_0.pdb\"\n"
    )
    fake_boltz.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', os.defpath)}"

    receptor = tmp_path / "receptor.pdb"
    receptor.write_text(
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 90.00           C\n"
        "END\n"
    )
    manifest = tmp_path / "manifest.tsv"
    pd.DataFrame(
        [{"target_id": "P1", "source": "alphafold_cleaned", "input_pdb": str(receptor)}]
    ).to_csv(manifest, sep="\t", index=False)
    ligand = tmp_path / "ligand.sdf"
    write_valid_ligand_sdf(ligand)
    report = tmp_path / "report.tsv"

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage5_boltz2.py"),
            "--manifest", str(manifest),
            "--ligand-sdf", str(ligand),
            "--struct-dir", str(tmp_path),
            "--out-dir", str(tmp_path / "out"),
            "--out-report", str(report),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode == 0, res.stderr
    rows = pd.read_csv(report, sep="\t")
    assert rows["n_residues"].tolist() == [1]


def test_stage5_accepts_standard_boltz_aliases_without_posebusters(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_boltz = fake_bin / "boltz"
    fake_boltz.write_text(
        "#!/bin/sh\n"
        "out=''\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = '--out_dir' ]; then shift; out=\"$1\"; fi\n"
        "  shift\n"
        "done\n"
        "mkdir -p \"$out\"\n"
        "printf '{\"iptm\":0.8,\"complex_plddt\":90}' > \"$out/report.json\"\n"
        "printf '{\"affinity_pred_value\":1.2}' > \"$out/affinity.json\"\n"
        "mkdir -p \"$out/predictions/input\"\n"
        "printf 'ATOM protein\\nHETATM ligand\\n' > \"$out/predictions/input/input_model_0.pdb\"\n"
    )
    fake_boltz.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', os.defpath)}"

    receptor = tmp_path / "receptor.pdb"
    receptor.write_text(
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 90.00           C\n"
        "END\n"
    )
    manifest = tmp_path / "manifest.tsv"
    pd.DataFrame(
        [{"target_id": "P1", "source": "alphafold_cleaned", "input_pdb": str(receptor)}]
    ).to_csv(manifest, sep="\t", index=False)
    ligand = tmp_path / "ligand.sdf"
    write_valid_ligand_sdf(ligand)
    report = tmp_path / "report.tsv"

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage5_boltz2.py"),
            "--manifest", str(manifest),
            "--ligand-sdf", str(ligand),
            "--struct-dir", str(tmp_path),
            "--out-dir", str(tmp_path / "out"),
            "--out-report", str(report),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode == 0, res.stderr
    rows = pd.read_csv(report, sep="\t")
    assert rows["target_id"].tolist() == ["P1"]
    assert rows["complex_plddt"].tolist() == [90.0]
    assert rows["pb_valid"].tolist() == ["not_available"]
    assert rows["kept"].tolist() == ["yes"]
    complex_pdb = Path(rows.loc[0, "complex_pdb"])
    assert complex_pdb.name == "complex.pdb"
    assert complex_pdb.is_file()


def test_stage5_partial_boltz_failure_does_not_leave_report(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_boltz = fake_bin / "boltz"
    fake_boltz.write_text(
        "#!/bin/sh\n"
        "out=''\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = '--out_dir' ]; then shift; out=\"$1\"; fi\n"
        "  shift\n"
        "done\n"
        "mkdir -p \"$out\"\n"
        "case \"$out\" in\n"
        "  *P1*) mkdir -p \"$out/predictions/input\"; printf '{\"iptm\":0.8,\"pocket_plddt\":90,\"affinity_log_uM\":1.2,\"posebusters_valid\":true,\"n_residues\":120}' > \"$out/report.json\"; printf 'ATOM protein\\nHETATM ligand\\n' > \"$out/predictions/input/input_model_0.pdb\" ;;\n"
        "  *) exit 0 ;;\n"
        "esac\n"
    )
    fake_boltz.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', os.defpath)}"

    receptors = []
    for uid in ("P1", "P2"):
        receptor = tmp_path / f"{uid}.pdb"
        receptor.write_text(
            "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 90.00           C\n"
            "END\n"
        )
        receptors.append({"target_id": uid, "source": "alphafold_cleaned", "input_pdb": str(receptor)})
    manifest = tmp_path / "manifest.tsv"
    pd.DataFrame(receptors).to_csv(manifest, sep="\t", index=False)
    ligand = tmp_path / "ligand.sdf"
    write_valid_ligand_sdf(ligand)
    report = tmp_path / "report.tsv"
    report.write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage5_boltz2.py"),
            "--manifest", str(manifest),
            "--ligand-sdf", str(ligand),
            "--struct-dir", str(tmp_path),
            "--out-dir", str(tmp_path / "out"),
            "--out-report", str(report),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "Boltz-2 failed for receptors P2" in res.stderr
    assert not report.exists()


def test_stage5_boltz_partial_report_requires_explicit_flag(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_boltz = fake_bin / "boltz"
    fake_boltz.write_text(
        "#!/bin/sh\n"
        "out=''\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = '--out_dir' ]; then shift; out=\"$1\"; fi\n"
        "  shift\n"
        "done\n"
        "mkdir -p \"$out\"\n"
        "case \"$out\" in\n"
        "  *P1*) mkdir -p \"$out/predictions/input\"; printf '{\"iptm\":0.8,\"pocket_plddt\":90,\"affinity_log_uM\":1.2,\"posebusters_valid\":true,\"n_residues\":120}' > \"$out/report.json\"; printf 'ATOM protein\\nHETATM ligand\\n' > \"$out/predictions/input/input_model_0.pdb\" ;;\n"
        "  *) exit 0 ;;\n"
        "esac\n"
    )
    fake_boltz.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', os.defpath)}"

    receptors = []
    for uid in ("P1", "P2"):
        receptor = tmp_path / f"{uid}.pdb"
        receptor.write_text(
            "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 90.00           C\n"
            "END\n"
        )
        receptors.append({"target_id": uid, "source": "alphafold_cleaned", "input_pdb": str(receptor)})
    manifest = tmp_path / "manifest.tsv"
    pd.DataFrame(receptors).to_csv(manifest, sep="\t", index=False)
    ligand = tmp_path / "ligand.sdf"
    write_valid_ligand_sdf(ligand)
    report = tmp_path / "report.tsv"

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage5_boltz2.py"),
            "--manifest", str(manifest),
            "--ligand-sdf", str(ligand),
            "--struct-dir", str(tmp_path),
            "--out-dir", str(tmp_path / "out"),
            "--out-report", str(report),
            "--allow-partial-output",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode == 0, res.stderr
    rows = pd.read_csv(report, sep="\t")
    assert rows["target_id"].tolist() == ["P1"]
    assert "skipped" not in rows["kept"].astype(str).tolist()


def test_stage6_bioemu_fails_without_kept_boltz_rows(tmp_path: Path) -> None:
    report = tmp_path / "boltz.tsv"
    pd.DataFrame([{"target_id": "P1", "kept": "no"}]).to_csv(report, sep="\t", index=False)
    res = run_script(
        "stage6_bioemu.py",
        [
            "--boltz-report", str(report),
            "--struct-dir", str(tmp_path),
            "--out-dir", str(tmp_path / "out"),
            "--out-manifest", str(tmp_path / "manifest.tsv"),
        ],
        tmp_path,
    )
    assert res.returncode != 0
    assert "No Boltz-2 kept targets" in res.stderr


def test_stage6_bioemu_rejects_empty_boltz_report_without_stale_manifest(
    tmp_path: Path,
) -> None:
    report = tmp_path / "boltz.tsv"
    report.write_text("")
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("stale\n")

    res = run_script(
        "stage6_bioemu.py",
        [
            "--boltz-report", str(report),
            "--struct-dir", str(tmp_path),
            "--out-dir", str(tmp_path / "out"),
            "--out-manifest", str(manifest),
        ],
        tmp_path,
    )

    assert res.returncode != 0
    assert "Boltz-2 report is required and must be non-empty" in res.stderr
    assert not manifest.exists()


def test_stage6_bioemu_rejects_boltz_report_missing_required_columns(
    tmp_path: Path,
) -> None:
    report = tmp_path / "boltz.tsv"
    pd.DataFrame([{"target_id": "P1"}]).to_csv(report, sep="\t", index=False)
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("stale\n")

    res = run_script(
        "stage6_bioemu.py",
        [
            "--boltz-report", str(report),
            "--struct-dir", str(tmp_path),
            "--out-dir", str(tmp_path / "out"),
            "--out-manifest", str(manifest),
        ],
        tmp_path,
    )

    assert res.returncode != 0
    assert "Boltz-2 report missing required columns" in res.stderr
    assert not manifest.exists()


def test_stage6_bioemu_rejects_boltz_report_without_rows(
    tmp_path: Path,
) -> None:
    report = tmp_path / "boltz.tsv"
    pd.DataFrame(columns=["target_id", "kept"]).to_csv(report, sep="\t", index=False)
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("stale\n")

    res = run_script(
        "stage6_bioemu.py",
        [
            "--boltz-report", str(report),
            "--struct-dir", str(tmp_path),
            "--out-dir", str(tmp_path / "out"),
            "--out-manifest", str(manifest),
        ],
        tmp_path,
    )

    assert res.returncode != 0
    assert "Boltz-2 report contains no rows" in res.stderr
    assert not manifest.exists()


def test_stage6_bioemu_rejects_boltz_report_blank_required_fields(
    tmp_path: Path,
) -> None:
    report = tmp_path / "boltz.tsv"
    pd.DataFrame([
        {"target_id": "P1", "kept": "yes"},
        {"target_id": None, "kept": "yes"},
    ]).to_csv(report, sep="\t", index=False)
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("stale\n")

    res = run_script(
        "stage6_bioemu.py",
        [
            "--boltz-report", str(report),
            "--struct-dir", str(tmp_path),
            "--out-dir", str(tmp_path / "out"),
            "--out-manifest", str(manifest),
        ],
        tmp_path,
    )

    assert res.returncode != 0
    assert "Boltz-2 report column 'target_id' contains blank values" in res.stderr
    assert not manifest.exists()


def test_stage6_bioemu_rejects_invalid_kept_values(
    tmp_path: Path,
) -> None:
    report = tmp_path / "boltz.tsv"
    pd.DataFrame([{"target_id": "P1", "kept": "maybe"}]).to_csv(
        report, sep="\t", index=False
    )
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("stale\n")

    res = run_script(
        "stage6_bioemu.py",
        [
            "--boltz-report", str(report),
            "--struct-dir", str(tmp_path),
            "--out-dir", str(tmp_path / "out"),
            "--out-manifest", str(manifest),
        ],
        tmp_path,
    )

    assert res.returncode != 0
    assert "Boltz-2 report column 'kept' contains invalid values: maybe" in res.stderr
    assert not manifest.exists()


def test_stage6_bioemu_rejects_duplicate_boltz_target_ids(
    tmp_path: Path,
) -> None:
    report = tmp_path / "boltz.tsv"
    pd.DataFrame([
        {"target_id": "P1", "kept": "yes"},
        {"target_id": "P1", "kept": "no"},
    ]).to_csv(report, sep="\t", index=False)
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("stale\n")

    res = run_script(
        "stage6_bioemu.py",
        [
            "--boltz-report", str(report),
            "--struct-dir", str(tmp_path),
            "--out-dir", str(tmp_path / "out"),
            "--out-manifest", str(manifest),
        ],
        tmp_path,
    )

    assert res.returncode != 0
    assert "Boltz-2 report contains duplicate target_id values: P1" in res.stderr
    assert not manifest.exists()


def test_stage6_bioemu_rejects_kept_rows_missing_quality_evidence(
    tmp_path: Path,
) -> None:
    report = tmp_path / "boltz.tsv"
    pd.DataFrame([{"target_id": "P1", "kept": "yes"}]).to_csv(
        report, sep="\t", index=False
    )
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("stale\n")

    res = run_script(
        "stage6_bioemu.py",
        [
            "--boltz-report", str(report),
            "--struct-dir", str(tmp_path),
            "--out-dir", str(tmp_path / "out"),
            "--out-manifest", str(manifest),
        ],
        tmp_path,
    )

    assert res.returncode != 0
    assert "Boltz-2 kept rows missing required quality evidence columns" in res.stderr
    assert not manifest.exists()


def test_stage6_bioemu_rejects_nonfinite_kept_quality_metric(
    tmp_path: Path,
) -> None:
    report = tmp_path / "boltz.tsv"
    pd.DataFrame([{
        "target_id": "P1",
        "source": "alphafold_cleaned",
        "n_residues": 120,
        "iptm": float("inf"),
        "pocket_plddt": 90,
        "affinity_log_uM": 1.2,
        "pb_valid": "yes",
        "kept": "yes",
    }]).to_csv(report, sep="\t", index=False)
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("stale\n")

    res = run_script(
        "stage6_bioemu.py",
        [
            "--boltz-report", str(report),
            "--struct-dir", str(tmp_path),
            "--out-dir", str(tmp_path / "out"),
            "--out-manifest", str(manifest),
        ],
        tmp_path,
    )

    assert res.returncode != 0
    assert "Boltz-2 kept rows column 'iptm' must be finite" in res.stderr
    assert not manifest.exists()


def test_stage6_bioemu_accepts_kept_rows_without_posebusters_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    stage6_bioemu = load_script_module("stage6_bioemu.py")
    report = tmp_path / "boltz.tsv"
    pd.DataFrame([{
        "target_id": "P1",
        "source": "alphafold_cleaned",
        "n_residues": 120,
        "iptm": 0.8,
        "pocket_plddt": 90,
        "affinity_log_uM": 1.2,
        "pb_valid": "not_available",
        "kept": "yes",
    }]).to_csv(report, sep="\t", index=False)

    parsed = stage6_bioemu.read_boltz_report(report)

    assert parsed["target_id"].tolist() == ["P1"]


def test_stage6_bioemu_rejects_boolean_kept_quality_metric(
    tmp_path: Path,
) -> None:
    report = tmp_path / "boltz.tsv"
    report.write_text(
        "target_id\tsource\tn_residues\tiptm\tpocket_plddt\t"
        "affinity_log_uM\tpb_valid\tkept\n"
        "P1\talphafold_cleaned\t120\tTrue\t90\t1.2\tyes\tyes\n"
    )
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("stale\n")

    res = run_script(
        "stage6_bioemu.py",
        [
            "--boltz-report", str(report),
            "--struct-dir", str(tmp_path),
            "--out-dir", str(tmp_path / "out"),
            "--out-manifest", str(manifest),
        ],
        tmp_path,
    )

    assert res.returncode != 0
    assert "Boltz-2 kept rows column 'iptm' must be numeric" in res.stderr
    assert not manifest.exists()


def test_stage6_bioemu_rejects_invalid_kept_source(
    tmp_path: Path,
) -> None:
    report = tmp_path / "boltz.tsv"
    pd.DataFrame([{
        "target_id": "P1",
        "source": "manual_patch",
        "n_residues": 120,
        "iptm": 0.8,
        "pocket_plddt": 90,
        "affinity_log_uM": 1.2,
        "pb_valid": "yes",
        "kept": "yes",
    }]).to_csv(report, sep="\t", index=False)
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("stale\n")

    res = run_script(
        "stage6_bioemu.py",
        [
            "--boltz-report", str(report),
            "--struct-dir", str(tmp_path),
            "--out-dir", str(tmp_path / "out"),
            "--out-manifest", str(manifest),
        ],
        tmp_path,
    )

    assert res.returncode != 0
    assert "Boltz-2 kept rows column 'source' contains invalid values" in res.stderr
    assert "manual_patch" in res.stderr
    assert "alphafold_cleaned" in res.stderr
    assert "rcsb_holo_post_cutoff" in res.stderr
    assert not manifest.exists()


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        ("--top-n", "0", "--top-n must be >= 1: 0"),
        ("--num-conformers", "0", "--num-conformers must be >= 1: 0"),
        ("--kmeans-k", "0", "--kmeans-k must be >= 1: 0"),
    ],
)
def test_stage6_bioemu_rejects_nonpositive_numeric_args_without_manifest(
    tmp_path: Path,
    flag: str,
    value: str,
    message: str,
) -> None:
    report = tmp_path / "boltz.tsv"
    pd.DataFrame([{
        "target_id": "P1",
        "source": "alphafold_cleaned",
        "n_residues": 120,
        "iptm": 0.8,
        "pocket_plddt": 90,
        "affinity_log_uM": 1.2,
        "pb_valid": "yes",
        "kept": "yes",
    }]).to_csv(report, sep="\t", index=False)
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("stale\n")

    res = run_script(
        "stage6_bioemu.py",
        [
            "--boltz-report", str(report),
            "--struct-dir", str(tmp_path),
            "--out-dir", str(tmp_path / "out"),
            "--out-manifest", str(manifest),
            flag, value,
        ],
        tmp_path,
    )

    assert res.returncode != 0
    assert message in res.stderr
    assert not manifest.exists()


def test_stage6_failed_bioemu_does_not_leave_partial_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage6_bioemu = load_script_module("stage6_bioemu.py")
    report = tmp_path / "boltz.tsv"
    pd.DataFrame([
        {
            "target_id": "P1",
            "source": "alphafold_cleaned",
            "n_residues": 120,
            "iptm": 0.8,
            "pocket_plddt": 90,
            "affinity_log_uM": 1.2,
            "pb_valid": "yes",
            "kept": "yes",
        },
        {
            "target_id": "P2",
            "source": "alphafold_cleaned",
            "n_residues": 121,
            "iptm": 0.9,
            "pocket_plddt": 91,
            "affinity_log_uM": 1.1,
            "pb_valid": "yes",
            "kept": "yes",
        },
    ]).to_csv(report, sep="\t", index=False)
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("stale\n")
    residue_mapping = load_script_module("residue_mapping.py")
    for uid in ("P1", "P2"):
        receptor = tmp_path / uid / f"{uid}_input.pdb"
        receptor.parent.mkdir()
        receptor.write_text(
            "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 90.00           C\nEND\n"
        )
        record = residue_mapping.build_mapping_record(
            receptor,
            target_id=uid,
            source="alphafold_cleaned",
            canonical_sequence="A",
        )
        residue_mapping.write_mapping_record(
            receptor,
            record,
            residue_mapping.mapping_path_for_pdb(receptor),
        )

    # BioEmu 는 콘솔 스크립트를 만들지 않는다. 있고 없고는 PATH 가 아니라
    # 이 인터프리터가 import 할 수 있느냐로 갈린다.
    monkeypatch.setattr(stage6_bioemu, "bioemu_available", lambda: True)
    monkeypatch.setattr(stage6_bioemu, "run_bioemu", lambda *args, **kwargs: True)

    def fake_cluster(conformer_dir: Path, _k: int) -> list[Path]:
        if conformer_dir.name == "P2":
            return []
        medoid = conformer_dir / "medoid.pdb"
        medoid.write_text(
            "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 90.00           C\nEND\n"
        )
        return [medoid]

    monkeypatch.setattr(stage6_bioemu, "cluster_conformers", fake_cluster)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage6_bioemu.py",
            "--boltz-report", str(report),
            "--struct-dir", str(tmp_path),
            "--out-dir", str(tmp_path / "out"),
            "--out-manifest", str(manifest),
        ],
    )

    with pytest.raises(SystemExit, match="BioEmu clustering produced no medoids for P2"):
        stage6_bioemu.main()

    assert not manifest.exists()


def test_stage6_bioemu_rejects_empty_receptor_pdb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage6_bioemu = load_script_module("stage6_bioemu.py")
    report = tmp_path / "boltz.tsv"
    pd.DataFrame([{
        "target_id": "P1",
        "source": "alphafold_cleaned",
        "n_residues": 120,
        "iptm": 0.8,
        "pocket_plddt": 90,
        "affinity_log_uM": 1.2,
        "pb_valid": "yes",
        "kept": "yes",
    }]).to_csv(report, sep="\t", index=False)
    receptor = tmp_path / "P1" / "P1_input.pdb"
    receptor.parent.mkdir()
    receptor.write_text("")
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("stale\n")
    # BioEmu 는 콘솔 스크립트를 만들지 않는다. 있고 없고는 PATH 가 아니라
    # 이 인터프리터가 import 할 수 있느냐로 갈린다.
    monkeypatch.setattr(stage6_bioemu, "bioemu_available", lambda: True)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage6_bioemu.py",
            "--boltz-report", str(report),
            "--struct-dir", str(tmp_path),
            "--out-dir", str(tmp_path / "out"),
            "--out-manifest", str(manifest),
        ],
    )

    with pytest.raises(SystemExit, match="BioEmu receptor PDB is missing or empty for P1"):
        stage6_bioemu.main()
    assert not manifest.exists()


def run_stage6_ensemble(
    tmp_path: Path,
    manifest: Path,
    *,
    with_tools: bool = True,
) -> subprocess.CompletedProcess[str]:
    ligand = tmp_path / "ligand.sdf"
    write_valid_ligand_sdf(ligand)
    anchors = _stage6_anchor_fixture(tmp_path, manifest)
    env = os.environ.copy()
    if with_tools:
        fake_bin = tmp_path / "bin"
        fake_bin.mkdir(exist_ok=True)
        fake_gnina = fake_bin / "gnina"
        fake_gnina.write_text(_fake_stage6_gnina("1.5"))
        fake_gnina.chmod(0o755)
        fake_pkg = tmp_path / "fake_pkg"
        rtmscore_pkg = fake_pkg / "rtmscore"
        rtmscore_pkg.mkdir(parents=True, exist_ok=True)
        (rtmscore_pkg / "__init__.py").write_text(
            "def predict_affinity(receptor, ligand, **kwargs):\n    return 2.5\n"
        )
        env["PATH"] = str(fake_bin)
        env["SKINSCOUT_GNINA_BINARY"] = str(fake_gnina)
        env["PYTHONPATH"] = str(fake_pkg)
    else:
        env["PATH"] = str(tmp_path / "empty_path")
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage6_ensemble_dock.py"),
            "--manifest",
            str(manifest),
            "--anchor-manifest",
            str(anchors),
            "--ligand-sdf",
            str(ligand),
            "--out-dir",
            str(tmp_path / "out"),
            "--out-consensus",
            str(tmp_path / "ensemble.tsv"),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def _pdb_atom(record: str, serial: int, name: str, residue: str, residue_id: int,
              x: float, y: float, z: float) -> str:
    element = name[0]
    return (
        f"{record:<6}{serial:5d} {name:<4s} {residue:>3s} A{residue_id:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 90.00           {element:>2s}"
    )


def _stage6_anchor_fixture(tmp_path: Path, manifest: Path) -> Path:
    try:
        frame = pd.read_csv(manifest, sep="\t")
        targets = [str(value) for value in frame.get("target_id", []) if pd.notna(value)]
        medoids = [Path(str(value)) for value in frame.get("medoid_pdb", []) if pd.notna(value)]
    except Exception:
        targets, medoids = [], []
    receptor = "\n".join([
        _pdb_atom("ATOM", 1, "CA", "ALA", 1, 0.0, 0.0, 0.0),
        _pdb_atom("ATOM", 2, "CA", "GLY", 2, 2.0, 0.0, 0.0),
        _pdb_atom("ATOM", 3, "CA", "ALA", 3, 0.0, 2.0, 0.0),
    ])
    for medoid in medoids:
        if medoid.exists() and medoid.stat().st_size > 0:
            medoid.write_text(receptor + "\nEND\n")
    rows = []
    for target in dict.fromkeys(targets):
        anchor = tmp_path / f"{target}_anchor.pdb"
        anchor.write_text(
            receptor + "\n"
            + _pdb_atom("HETATM", 4, "C1", "LIG", 1, 1.0, 1.0, 1.0) + "\n"
            + _pdb_atom("HETATM", 5, "C2", "LIG", 1, 2.0, 1.0, 1.0) + "\n"
            + _pdb_atom("HETATM", 6, "O1", "LIG", 1, 3.0, 1.0, 1.0)
            + "\nEND\n"
        )
        rows.append({"target_id": target, "complex_pdb": anchor, "kept": "yes"})
    path = tmp_path / "anchors.tsv"
    pd.DataFrame(rows, columns=["target_id", "complex_pdb", "kept"]).to_csv(
        path, sep="\t", index=False
    )
    return path


def _fake_stage6_gnina(score: str) -> str:
    return (
        "#!/bin/sh\nligand=''\nout=''\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  case \"$1\" in --ligand|-l) shift; ligand=\"$1\" ;; --out) shift; out=\"$1\" ;; esac\n"
        "  shift\n"
        "done\n"
        "if [ -n \"$out\" ]; then /bin/cp \"$ligand\" \"$out\"; fi\n"
        f"printf 'CNNaffinity: {score}\\n'\n"
    )


def test_stage6_ensemble_rejects_manifest_missing_required_columns(tmp_path: Path) -> None:
    manifest = tmp_path / "ensemble_manifest.tsv"
    pd.DataFrame([{"target_id": "P1"}]).to_csv(manifest, sep="\t", index=False)

    res = run_stage6_ensemble(tmp_path, manifest)

    assert res.returncode != 0
    assert "BioEmu medoid manifest missing required columns" in res.stderr
    assert not (tmp_path / "ensemble.tsv").exists()


def test_stage6_ensemble_rejects_empty_manifest_without_stale_consensus(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "ensemble_manifest.tsv"
    manifest.write_text("")
    consensus = tmp_path / "ensemble.tsv"
    consensus.write_text("stale\n")

    res = run_stage6_ensemble(tmp_path, manifest)

    assert res.returncode != 0
    assert "BioEmu medoid manifest is required and must be non-empty" in res.stderr
    assert not consensus.exists()


def test_stage6_ensemble_rejects_manifest_blank_required_fields(tmp_path: Path) -> None:
    medoid = tmp_path / "P1_medoid.pdb"
    medoid.write_text(
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 90.00           C\nEND\n"
    )
    manifest = tmp_path / "ensemble_manifest.tsv"
    pd.DataFrame([
        {"target_id": "P1", "medoid_pdb": str(medoid)},
        {"target_id": None, "medoid_pdb": str(medoid)},
    ]).to_csv(manifest, sep="\t", index=False)

    res = run_stage6_ensemble(tmp_path, manifest)

    assert res.returncode != 0
    assert "BioEmu medoid manifest column 'target_id' contains blank values" in res.stderr
    assert not (tmp_path / "ensemble.tsv").exists()


def test_stage6_ensemble_rejects_duplicate_medoid_paths(tmp_path: Path) -> None:
    medoid = tmp_path / "P1_medoid.pdb"
    medoid.write_text(
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 90.00           C\nEND\n"
    )
    manifest = tmp_path / "ensemble_manifest.tsv"
    stale = tmp_path / "ensemble.tsv"
    stale.write_text("stale\n")
    pd.DataFrame(
        [
            {"target_id": "P1", "medoid_pdb": str(medoid)},
            {"target_id": "P2", "medoid_pdb": str(medoid)},
        ]
    ).to_csv(manifest, sep="\t", index=False)

    res = run_stage6_ensemble(tmp_path, manifest)

    assert res.returncode != 0
    assert "BioEmu medoid manifest contains duplicate medoid_pdb values" in res.stderr
    assert str(medoid) in res.stderr
    assert not stale.exists()


def test_stage6_ensemble_rejects_canonical_duplicate_medoid_paths(tmp_path: Path) -> None:
    medoid = tmp_path / "P1_medoid.pdb"
    medoid.write_text(
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 90.00           C\nEND\n"
    )
    manifest = tmp_path / "ensemble_manifest.tsv"
    stale = tmp_path / "ensemble.tsv"
    stale.write_text("stale\n")
    pd.DataFrame(
        [
            {"target_id": "P1", "medoid_pdb": str(medoid)},
            {"target_id": "P2", "medoid_pdb": f"{tmp_path}/./{medoid.name}"},
        ]
    ).to_csv(manifest, sep="\t", index=False)

    res = run_stage6_ensemble(tmp_path, manifest)

    assert res.returncode != 0
    assert "BioEmu medoid manifest contains duplicate medoid_pdb values" in res.stderr
    assert str(medoid.resolve()) in res.stderr
    assert not stale.exists()


def test_stage6_ensemble_fails_on_missing_medoid_instead_of_partial_consensus(tmp_path: Path) -> None:
    medoid = tmp_path / "P1_medoid.pdb"
    medoid.write_text("ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 90.00           C\nEND\n")
    manifest = tmp_path / "ensemble_manifest.tsv"
    stale = tmp_path / "ensemble.tsv"
    stale.write_text("stale\n")
    pd.DataFrame(
        [
            {"target_id": "P1", "medoid_pdb": str(medoid)},
            {"target_id": "P2", "medoid_pdb": str(tmp_path / "missing_medoid.pdb")},
        ]
    ).to_csv(manifest, sep="\t", index=False)

    res = run_stage6_ensemble(tmp_path, manifest)

    assert res.returncode != 0
    assert "BioEmu medoid PDB is missing or empty for P2" in res.stderr
    assert not (tmp_path / "ensemble.tsv").exists()


def test_stage6_ensemble_fails_on_empty_medoid_instead_of_partial_consensus(tmp_path: Path) -> None:
    medoid = tmp_path / "empty_medoid.pdb"
    medoid.write_text("")
    manifest = tmp_path / "ensemble_manifest.tsv"
    stale = tmp_path / "ensemble.tsv"
    stale.write_text("stale\n")
    pd.DataFrame([{"target_id": "P1", "medoid_pdb": str(medoid)}]).to_csv(
        manifest, sep="\t", index=False
    )
    res = run_stage6_ensemble(tmp_path, manifest)

    assert res.returncode != 0
    assert "BioEmu medoid PDB is missing or empty for P1" in res.stderr
    assert not (tmp_path / "ensemble.tsv").exists()


def test_stage6_ensemble_rejects_empty_ligand_sdf(tmp_path: Path) -> None:
    medoid = tmp_path / "P1_medoid.pdb"
    medoid.write_text(
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 90.00           C\nEND\n"
    )
    ligand = tmp_path / "ligand.sdf"
    ligand.write_text("")
    manifest = tmp_path / "ensemble_manifest.tsv"
    pd.DataFrame([{"target_id": "P1", "medoid_pdb": str(medoid)}]).to_csv(
        manifest, sep="\t", index=False
    )
    anchors = _stage6_anchor_fixture(tmp_path, manifest)
    consensus = tmp_path / "ensemble.tsv"
    consensus.write_text("stale\n")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_gnina = fake_bin / "gnina"
    fake_gnina.write_text(_fake_stage6_gnina("1.5"))
    fake_gnina.chmod(0o755)
    fake_pkg = tmp_path / "fake_pkg"
    rtmscore_pkg = fake_pkg / "rtmscore"
    rtmscore_pkg.mkdir(parents=True)
    (rtmscore_pkg / "__init__.py").write_text(
        "def predict_affinity(receptor, ligand):\n    return 2.5\n"
    )
    env = os.environ.copy()
    env["PATH"] = str(fake_bin)
    env["SKINSCOUT_GNINA_BINARY"] = str(fake_gnina)
    env["PYTHONPATH"] = str(fake_pkg)

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage6_ensemble_dock.py"),
            "--manifest", str(manifest),
            "--anchor-manifest", str(anchors),
            "--ligand-sdf", str(ligand),
            "--out-dir", str(tmp_path / "out"),
            "--out-consensus", str(consensus),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "Ligand SDF is missing or empty for ensemble docking" in res.stderr
    assert not consensus.exists()


def test_stage6_ensemble_writes_consensus_when_all_medoids_score(tmp_path: Path) -> None:
    medoid = tmp_path / "P1_medoid.pdb"
    medoid.write_text("ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 90.00           C\nEND\n")
    manifest = tmp_path / "ensemble_manifest.tsv"
    pd.DataFrame([{"target_id": "P1", "medoid_pdb": str(medoid)}]).to_csv(
        manifest, sep="\t", index=False
    )

    res = run_stage6_ensemble(tmp_path, manifest)

    assert res.returncode == 0, res.stderr
    out = pd.read_csv(tmp_path / "ensemble.tsv", sep="\t")
    assert out["target_id"].tolist() == ["P1"]
    assert out["consensus_score"].tolist() == [1.0]
    assert out["gnina_rank_score"].tolist() == [1.0]
    assert out["rtm_rank_score"].tolist() == [1.0]
    row = out.iloc[0]
    assert "--score_only" not in row["docking_command"]
    assert "--center_x" in row["docking_command"]
    for path_col, digest_col in (
        ("best_medoid_pdb", "best_medoid_sha256"),
        ("best_receptor_pdb", "best_receptor_sha256"),
        ("best_pose_sdf", "best_pose_sha256"),
        ("best_complex_pose", "best_complex_pose_sha256"),
    ):
        path = Path(row[path_col])
        assert path.is_file() and path.stat().st_size > 0
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row[digest_col]


def test_stage6_ensemble_rejects_nonfinite_scorer_output(tmp_path: Path) -> None:
    medoid = tmp_path / "P1_medoid.pdb"
    medoid.write_text(
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 90.00           C\nEND\n"
    )
    ligand = tmp_path / "ligand.sdf"
    write_valid_ligand_sdf(ligand)
    manifest = tmp_path / "ensemble_manifest.tsv"
    pd.DataFrame([{"target_id": "P1", "medoid_pdb": str(medoid)}]).to_csv(
        manifest, sep="\t", index=False
    )
    anchors = _stage6_anchor_fixture(tmp_path, manifest)
    consensus = tmp_path / "ensemble.tsv"
    consensus.write_text("stale\n")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_gnina = fake_bin / "gnina"
    fake_gnina.write_text(_fake_stage6_gnina("1.5"))
    fake_gnina.chmod(0o755)
    fake_pkg = tmp_path / "fake_pkg"
    rtmscore_pkg = fake_pkg / "rtmscore"
    rtmscore_pkg.mkdir(parents=True)
    (rtmscore_pkg / "__init__.py").write_text(
        "def predict_affinity(receptor, ligand, **kwargs):\n    return float('inf')\n"
    )
    env = os.environ.copy()
    env["PATH"] = str(fake_bin)
    env["SKINSCOUT_GNINA_BINARY"] = str(fake_gnina)
    env["PYTHONPATH"] = str(fake_pkg)

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage6_ensemble_dock.py"),
            "--manifest", str(manifest),
            "--anchor-manifest", str(anchors),
            "--ligand-sdf", str(ligand),
            "--out-dir", str(tmp_path / "out"),
            "--out-consensus", str(consensus),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "RTMScore output for" in res.stderr and "must be finite" in res.stderr
    assert not consensus.exists()


def test_stage6_ensemble_rejects_boolean_scorer_output(tmp_path: Path) -> None:
    medoid = tmp_path / "P1_medoid.pdb"
    medoid.write_text(
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 90.00           C\nEND\n"
    )
    ligand = tmp_path / "ligand.sdf"
    write_valid_ligand_sdf(ligand)
    manifest = tmp_path / "ensemble_manifest.tsv"
    pd.DataFrame([{"target_id": "P1", "medoid_pdb": str(medoid)}]).to_csv(
        manifest, sep="\t", index=False
    )
    anchors = _stage6_anchor_fixture(tmp_path, manifest)
    consensus = tmp_path / "ensemble.tsv"
    consensus.write_text("stale\n")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_gnina = fake_bin / "gnina"
    fake_gnina.write_text(_fake_stage6_gnina("1.5"))
    fake_gnina.chmod(0o755)
    fake_pkg = tmp_path / "fake_pkg"
    rtmscore_pkg = fake_pkg / "rtmscore"
    rtmscore_pkg.mkdir(parents=True)
    (rtmscore_pkg / "__init__.py").write_text(
        "def predict_affinity(receptor, ligand, **kwargs):\n    return True\n"
    )
    env = os.environ.copy()
    env["PATH"] = str(fake_bin)
    env["SKINSCOUT_GNINA_BINARY"] = str(fake_gnina)
    env["PYTHONPATH"] = str(fake_pkg)

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage6_ensemble_dock.py"),
            "--manifest", str(manifest),
            "--anchor-manifest", str(anchors),
            "--ligand-sdf", str(ligand),
            "--out-dir", str(tmp_path / "out"),
            "--out-consensus", str(consensus),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "RTMScore output for" in res.stderr and "must be numeric" in res.stderr
    assert not consensus.exists()


def test_stage6_ensemble_rejects_nonpositive_scorer_output(tmp_path: Path) -> None:
    medoid = tmp_path / "P1_medoid.pdb"
    medoid.write_text(
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 90.00           C\nEND\n"
    )
    ligand = tmp_path / "ligand.sdf"
    write_valid_ligand_sdf(ligand)
    manifest = tmp_path / "ensemble_manifest.tsv"
    pd.DataFrame([{"target_id": "P1", "medoid_pdb": str(medoid)}]).to_csv(
        manifest, sep="\t", index=False
    )
    anchors = _stage6_anchor_fixture(tmp_path, manifest)
    consensus = tmp_path / "ensemble.tsv"
    consensus.write_text("stale\n")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_gnina = fake_bin / "gnina"
    fake_gnina.write_text(_fake_stage6_gnina("0.0"))
    fake_gnina.chmod(0o755)
    fake_pkg = tmp_path / "fake_pkg"
    rtmscore_pkg = fake_pkg / "rtmscore"
    rtmscore_pkg.mkdir(parents=True)
    (rtmscore_pkg / "__init__.py").write_text(
        "def predict_affinity(receptor, ligand, **kwargs):\n    return -1.0\n"
    )
    env = os.environ.copy()
    env["PATH"] = str(fake_bin)
    env["SKINSCOUT_GNINA_BINARY"] = str(fake_gnina)
    env["PYTHONPATH"] = str(fake_pkg)

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage6_ensemble_dock.py"),
            "--manifest", str(manifest),
            "--anchor-manifest", str(anchors),
            "--ligand-sdf", str(ligand),
            "--out-dir", str(tmp_path / "out"),
            "--out-consensus", str(consensus),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "GNINA score for P1 must be > 0" in res.stderr
    assert not consensus.exists()


def test_stage7_run_fails_without_ready_systems(tmp_path: Path) -> None:
    manifest = tmp_path / "md.tsv"
    system = tmp_path / "out" / "P1" / "system.gro"
    topology = tmp_path / "out" / "P1" / "topol.top"
    tpr = tmp_path / "out" / "P1" / "prod.tpr"
    system.parent.mkdir(parents=True)
    system.write_text("system\n")
    topology.write_text("topology\n")
    tpr.write_text("tpr\n")
    pd.DataFrame([{
        "target_id": "P1",
        "system_gro": str(system),
        "topology_top": str(topology),
        "tpr": str(tpr),
        "status": "missing_tools",
    }]).to_csv(manifest, sep="\t", index=False)
    res = run_script(
        "stage7_gromacs_run.py",
        [
            "--manifest", str(manifest),
            "--out-dir", str(tmp_path / "out"),
            "--out-index", str(tmp_path / "traj.tsv"),
        ],
        tmp_path,
    )
    assert res.returncode != 0
    assert "MD manifest column 'status' contains invalid values: missing_tools" in res.stderr


def test_stage7_run_rejects_empty_manifest_without_stale_index(tmp_path: Path) -> None:
    manifest = tmp_path / "md.tsv"
    manifest.write_text("")
    out_index = tmp_path / "traj.tsv"
    out_index.write_text("stale\n")

    res = run_script(
        "stage7_gromacs_run.py",
        [
            "--manifest", str(manifest),
            "--out-dir", str(tmp_path / "out"),
            "--out-index", str(out_index),
        ],
        tmp_path,
    )

    assert res.returncode != 0
    assert "MD manifest is required and must be non-empty" in res.stderr
    assert not out_index.exists()


def test_stage7_run_rejects_manifest_missing_required_columns(tmp_path: Path) -> None:
    manifest = tmp_path / "md.tsv"
    pd.DataFrame([{"target_id": "P1", "status": "ready"}]).to_csv(
        manifest, sep="\t", index=False
    )
    res = run_script(
        "stage7_gromacs_run.py",
        [
            "--manifest", str(manifest),
            "--out-dir", str(tmp_path / "out"),
            "--out-index", str(tmp_path / "traj.tsv"),
        ],
        tmp_path,
    )
    assert res.returncode != 0
    assert "MD manifest missing required columns" in res.stderr
    assert not (tmp_path / "traj.tsv").exists()


def test_stage7_run_rejects_manifest_blank_required_fields(tmp_path: Path) -> None:
    manifest = tmp_path / "md.tsv"
    pd.DataFrame([{
        "target_id": None,
        "system_gro": "system.gro",
        "topology_top": "topol.top",
        "tpr": "prod.tpr",
        "status": "ready",
    }]).to_csv(manifest, sep="\t", index=False)
    out_index = tmp_path / "traj.tsv"
    out_index.write_text("stale\n")

    res = run_script(
        "stage7_gromacs_run.py",
        [
            "--manifest", str(manifest),
            "--out-dir", str(tmp_path / "out"),
            "--out-index", str(out_index),
        ],
        tmp_path,
    )

    assert res.returncode != 0
    assert "MD manifest column 'target_id' contains blank values" in res.stderr
    assert not out_index.exists()


def test_stage7_run_rejects_duplicate_target_ids(tmp_path: Path) -> None:
    manifest = tmp_path / "md.tsv"
    pd.DataFrame([
        {
            "target_id": "P1",
            "system_gro": "system1.gro",
            "topology_top": "topol1.top",
            "tpr": "prod1.tpr",
            "status": "ready",
        },
        {
            "target_id": "P1",
            "system_gro": "system2.gro",
            "topology_top": "topol2.top",
            "tpr": "prod2.tpr",
            "status": "ready",
        },
    ]).to_csv(manifest, sep="\t", index=False)
    out_index = tmp_path / "traj.tsv"
    out_index.write_text("stale\n")

    res = run_script(
        "stage7_gromacs_run.py",
        [
            "--manifest", str(manifest),
            "--out-dir", str(tmp_path / "out"),
            "--out-index", str(out_index),
        ],
        tmp_path,
    )

    assert res.returncode != 0
    assert "MD manifest contains duplicate target_id values: P1" in res.stderr
    assert not out_index.exists()


def test_stage7_run_rejects_duplicate_ready_system_paths(tmp_path: Path) -> None:
    system = tmp_path / "P1" / "system.gro"
    topology = tmp_path / "P1" / "topol.top"
    tpr = tmp_path / "P1" / "prod.tpr"
    system.parent.mkdir()
    system.write_text("system\n")
    topology.write_text("topology\n")
    tpr.write_text("tpr\n")
    manifest = tmp_path / "md.tsv"
    pd.DataFrame([
        {
            "target_id": "P1",
            "system_gro": str(system),
            "topology_top": str(topology),
            "tpr": str(tpr),
            "status": "ready",
        },
        {
            "target_id": "P2",
            "system_gro": str(system),
            "topology_top": str(topology),
            "tpr": str(tpr),
            "status": "ready",
        },
    ]).to_csv(manifest, sep="\t", index=False)
    out_index = tmp_path / "traj.tsv"
    out_index.write_text("stale\n")

    res = run_script(
        "stage7_gromacs_run.py",
        [
            "--manifest", str(manifest),
            "--out-dir", str(tmp_path / "out"),
            "--out-index", str(out_index),
        ],
        tmp_path,
    )

    assert res.returncode != 0
    assert "MD manifest contains duplicate system_gro values" in res.stderr
    assert str(system) in res.stderr
    assert not out_index.exists()


def test_stage7_run_rejects_canonical_duplicate_ready_system_paths(
    tmp_path: Path,
) -> None:
    system = tmp_path / "P1" / "system.gro"
    topology = tmp_path / "P1" / "topol.top"
    tpr = tmp_path / "P1" / "prod.tpr"
    system.parent.mkdir()
    system.write_text("system\n")
    topology.write_text("topology\n")
    tpr.write_text("tpr\n")
    manifest = tmp_path / "md.tsv"
    pd.DataFrame([
        {
            "target_id": "P1",
            "system_gro": str(system),
            "topology_top": str(topology),
            "tpr": str(tpr),
            "status": "ready",
        },
        {
            "target_id": "P2",
            "system_gro": f"{system.parent}/./{system.name}",
            "topology_top": f"{topology.parent}/./{topology.name}",
            "tpr": f"{tpr.parent}/./{tpr.name}",
            "status": "ready",
        },
    ]).to_csv(manifest, sep="\t", index=False)
    out_index = tmp_path / "traj.tsv"
    out_index.write_text("stale\n")

    res = run_script(
        "stage7_gromacs_run.py",
        [
            "--manifest", str(manifest),
            "--out-dir", str(tmp_path / "out"),
            "--out-index", str(out_index),
        ],
        tmp_path,
    )

    assert res.returncode != 0
    assert "MD manifest contains duplicate system_gro values" in res.stderr
    assert str(system.resolve()) in res.stderr
    assert not out_index.exists()


def test_stage7_run_rejects_zero_replicas_without_stale_index(tmp_path: Path) -> None:
    manifest = tmp_path / "md.tsv"
    pd.DataFrame([{
        "target_id": "P1",
        "system_gro": "system.gro",
        "topology_top": "topol.top",
        "tpr": "prod.tpr",
        "status": "ready",
    }]).to_csv(manifest, sep="\t", index=False)
    out_index = tmp_path / "traj.tsv"
    out_index.write_text("stale\n")

    res = run_script(
        "stage7_gromacs_run.py",
        [
            "--manifest", str(manifest),
            "--out-dir", str(tmp_path / "out"),
            "--replicas", "0",
            "--out-index", str(out_index),
        ],
        tmp_path,
    )

    assert res.returncode != 0
    assert "--replicas must be >= 1: 0" in res.stderr
    assert not out_index.exists()


def test_stage7_run_rejects_ready_system_with_missing_files(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_gmx = fake_bin / "gmx"
    fake_gmx.write_text("#!/bin/sh\nexit 0\n")
    fake_gmx.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = str(fake_bin)

    manifest = tmp_path / "md.tsv"
    pd.DataFrame([{
        "target_id": "P1",
        "system_gro": str(tmp_path / "out" / "P1" / "system.gro"),
        "topology_top": str(tmp_path / "out" / "P1" / "topol.top"),
        "tpr": str(tmp_path / "out" / "P1" / "prod.tpr"),
        "npt_gro": str(tmp_path / "out" / "P1" / "npt.gro"),
        "prod_mdp": str(tmp_path / "out" / "P1" / "prod.mdp"),
        "status": "ready",
        "system_gro_sha256": "0" * 64,
        "topology_top_sha256": "0" * 64,
        "tpr_sha256": "0" * 64,
        "npt_gro_sha256": "0" * 64,
        "prod_mdp_sha256": "0" * 64,
    }]).to_csv(manifest, sep="\t", index=False)

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage7_gromacs_run.py"),
            "--manifest", str(manifest),
            "--out-dir", str(tmp_path / "out"),
            "--out-index", str(tmp_path / "traj.tsv"),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "Ready MD system file missing or empty for P1" in res.stderr
    assert not (tmp_path / "traj.tsv").exists()


def test_stage7_run_rejects_ready_system_with_empty_files(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_gmx = fake_bin / "gmx"
    fake_gmx.write_text("#!/bin/sh\nexit 0\n")
    fake_gmx.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = str(fake_bin)

    target_dir = tmp_path / "out" / "P1"
    target_dir.mkdir(parents=True)
    system = target_dir / "system.gro"
    topology = target_dir / "topol.top"
    tpr = target_dir / "prod.tpr"
    system.touch()
    topology.touch()
    tpr.touch()
    npt = target_dir / "npt.gro"
    prod_mdp = target_dir / "prod.mdp"
    npt.touch()
    prod_mdp.touch()
    manifest = tmp_path / "md.tsv"
    pd.DataFrame([{
        "target_id": "P1",
        "system_gro": str(system),
        "topology_top": str(topology),
        "tpr": str(tpr),
        "npt_gro": str(npt),
        "prod_mdp": str(prod_mdp),
        "status": "ready",
        "system_gro_sha256": "0" * 64,
        "topology_top_sha256": "0" * 64,
        "tpr_sha256": "0" * 64,
        "npt_gro_sha256": "0" * 64,
        "prod_mdp_sha256": "0" * 64,
    }]).to_csv(manifest, sep="\t", index=False)
    out_index = tmp_path / "traj.tsv"
    out_index.write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage7_gromacs_run.py"),
            "--manifest", str(manifest),
            "--out-dir", str(tmp_path / "out"),
            "--out-index", str(out_index),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "Ready MD system file missing or empty for P1" in res.stderr
    assert not out_index.exists()


def test_stage7_run_rejects_system_paths_outside_out_dir(tmp_path: Path) -> None:
    target_dir = tmp_path / "outside" / "P1"
    target_dir.mkdir(parents=True)
    system = target_dir / "system.gro"
    topology = target_dir / "topol.top"
    tpr = target_dir / "prod.tpr"
    system.write_text("system\n")
    topology.write_text("topology\n")
    tpr.write_text("tpr\n")
    # 복제별 tpr 은 prod.mdp 에서 만들어진다. 준비 단계가 늘 남기는 파일이다.
    prod_mdp = target_dir / "prod.mdp"
    prod_mdp.write_text("nsteps = 10\ngen_vel = no\n")
    npt = target_dir / "npt.gro"
    npt.write_text("equilibrated\n")
    manifest = tmp_path / "md.tsv"
    pd.DataFrame([{
        "target_id": "P1",
        "system_gro": str(system),
        "topology_top": str(topology),
        "tpr": str(tpr),
        "npt_gro": str(npt),
        "prod_mdp": str(prod_mdp),
        "status": "ready",
        "system_gro_sha256": hashlib.sha256(system.read_bytes()).hexdigest(),
        "topology_top_sha256": hashlib.sha256(topology.read_bytes()).hexdigest(),
        "tpr_sha256": hashlib.sha256(tpr.read_bytes()).hexdigest(),
        "npt_gro_sha256": hashlib.sha256(npt.read_bytes()).hexdigest(),
        "prod_mdp_sha256": hashlib.sha256(prod_mdp.read_bytes()).hexdigest(),
    }]).to_csv(manifest, sep="\t", index=False)
    out_index = tmp_path / "traj.tsv"
    out_index.write_text("stale\n")

    res = run_script(
        "stage7_gromacs_run.py",
        [
            "--manifest", str(manifest),
            "--out-dir", str(tmp_path / "out"),
            "--out-index", str(out_index),
        ],
        tmp_path,
    )

    assert res.returncode != 0
    assert "MD manifest system_gro for P1 must be" in res.stderr
    assert not out_index.exists()


def test_stage7_partial_prep_failure_does_not_leave_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    spec = importlib.util.spec_from_file_location(
        "stage7_gromacs_prep_under_test", ROOT / "scripts/stage7_gromacs_prep.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    consensus = tmp_path / "ensemble.tsv"
    pd.DataFrame(
        [
            {"target_id": "P1", "consensus_score": 2.0},
            {"target_id": "P2", "consensus_score": 1.0},
        ]
    ).to_csv(consensus, sep="\t", index=False)
    ligand = tmp_path / "ligand.sdf"
    write_valid_ligand_sdf(ligand)
    receptor_manifest = tmp_path / "receptors.tsv"
    pd.DataFrame(
        [
            {"target_id": "P1", "medoid_pdb": str(tmp_path / "P1.pdb")},
            {"target_id": "P2", "medoid_pdb": str(tmp_path / "P2.pdb")},
        ]
    ).to_csv(receptor_manifest, sep="\t", index=False)
    (tmp_path / "P1.pdb").write_text("receptor\n")
    (tmp_path / "P2.pdb").write_text("receptor\n")
    manifest = tmp_path / "md.tsv"
    manifest.write_text("stale\n")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage7_gromacs_prep.py",
            "--ensemble-consensus",
            str(consensus),
            "--receptor-manifest",
            str(receptor_manifest),
            "--ligand-sdf",
            str(ligand),
            "--out-dir",
            str(tmp_path / "out"),
            "--out-manifest",
            str(manifest),
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        module.main()

    assert "Stage7 GROMACS preparation is fail-closed" in str(excinfo.value)
    assert not manifest.exists()


def test_stage7_prep_partial_manifest_requires_explicit_flag(
    tmp_path: Path,
    monkeypatch,
) -> None:
    spec = importlib.util.spec_from_file_location(
        "stage7_gromacs_prep_under_test_partial",
        ROOT / "scripts/stage7_gromacs_prep.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    consensus = tmp_path / "ensemble.tsv"
    pd.DataFrame(
        [
            {"target_id": "P1", "consensus_score": 2.0},
            {"target_id": "P2", "consensus_score": 1.0},
        ]
    ).to_csv(consensus, sep="\t", index=False)
    ligand = tmp_path / "ligand.sdf"
    write_valid_ligand_sdf(ligand)
    receptor_manifest = tmp_path / "receptors.tsv"
    pd.DataFrame(
        [
            {"target_id": "P1", "medoid_pdb": str(tmp_path / "P1.pdb")},
            {"target_id": "P2", "medoid_pdb": str(tmp_path / "P2.pdb")},
        ]
    ).to_csv(receptor_manifest, sep="\t", index=False)
    (tmp_path / "P1.pdb").write_text("receptor\n")
    (tmp_path / "P2.pdb").write_text("receptor\n")
    manifest = tmp_path / "md.tsv"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage7_gromacs_prep.py",
            "--ensemble-consensus",
            str(consensus),
            "--receptor-manifest",
            str(receptor_manifest),
            "--ligand-sdf",
            str(ligand),
            "--out-dir",
            str(tmp_path / "out"),
            "--out-manifest",
            str(manifest),
            "--allow-partial-output",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        module.main()

    assert "Stage7 GROMACS preparation is fail-closed" in str(excinfo.value)
    assert not manifest.exists()


def test_stage7_prep_rejects_empty_consensus_without_manifest(tmp_path: Path) -> None:
    consensus = tmp_path / "ensemble.tsv"
    consensus.write_text("")
    ligand = tmp_path / "ligand.sdf"
    write_valid_ligand_sdf(ligand)
    manifest = tmp_path / "md.tsv"
    manifest.write_text("stale\n")

    res = run_script(
        "stage7_gromacs_prep.py",
        [
            "--ensemble-consensus", str(consensus),
            "--receptor-manifest", str(tmp_path / "receptors.tsv"),
            "--ligand-sdf", str(ligand),
            "--out-dir", str(tmp_path / "out"),
            "--out-manifest", str(manifest),
        ],
        tmp_path,
    )

    assert res.returncode != 0
    assert "Ensemble consensus is required and must be non-empty" in res.stderr
    assert not manifest.exists()


def test_stage7_prep_rejects_consensus_missing_required_columns(tmp_path: Path) -> None:
    consensus = tmp_path / "ensemble.tsv"
    pd.DataFrame([{"target_id": "P1"}]).to_csv(consensus, sep="\t", index=False)
    ligand = tmp_path / "ligand.sdf"
    write_valid_ligand_sdf(ligand)
    manifest = tmp_path / "md.tsv"
    manifest.write_text("stale\n")

    res = run_script(
        "stage7_gromacs_prep.py",
        [
            "--ensemble-consensus", str(consensus),
            "--receptor-manifest", str(tmp_path / "receptors.tsv"),
            "--ligand-sdf", str(ligand),
            "--out-dir", str(tmp_path / "out"),
            "--out-manifest", str(manifest),
        ],
        tmp_path,
    )

    assert res.returncode != 0
    assert "Ensemble consensus missing required columns" in res.stderr
    assert not manifest.exists()


def test_stage7_prep_rejects_consensus_blank_target_ids(tmp_path: Path) -> None:
    consensus = tmp_path / "ensemble.tsv"
    pd.DataFrame([
        {"target_id": "P1", "consensus_score": 2.0},
        {"target_id": None, "consensus_score": 1.0},
    ]).to_csv(consensus, sep="\t", index=False)
    ligand = tmp_path / "ligand.sdf"
    write_valid_ligand_sdf(ligand)
    manifest = tmp_path / "md.tsv"
    manifest.write_text("stale\n")

    res = run_script(
        "stage7_gromacs_prep.py",
        [
            "--ensemble-consensus", str(consensus),
            "--receptor-manifest", str(tmp_path / "receptors.tsv"),
            "--ligand-sdf", str(ligand),
            "--out-dir", str(tmp_path / "out"),
            "--out-manifest", str(manifest),
        ],
        tmp_path,
    )

    assert res.returncode != 0
    assert "Ensemble consensus column 'target_id' contains blank values" in res.stderr
    assert not manifest.exists()


def test_stage7_prep_rejects_consensus_non_numeric_scores(tmp_path: Path) -> None:
    consensus = tmp_path / "ensemble.tsv"
    pd.DataFrame([{"target_id": "P1", "consensus_score": "bad"}]).to_csv(
        consensus, sep="\t", index=False
    )
    ligand = tmp_path / "ligand.sdf"
    write_valid_ligand_sdf(ligand)
    manifest = tmp_path / "md.tsv"
    manifest.write_text("stale\n")

    res = run_script(
        "stage7_gromacs_prep.py",
        [
            "--ensemble-consensus", str(consensus),
            "--receptor-manifest", str(tmp_path / "receptors.tsv"),
            "--ligand-sdf", str(ligand),
            "--out-dir", str(tmp_path / "out"),
            "--out-manifest", str(manifest),
        ],
        tmp_path,
    )

    assert res.returncode != 0
    assert "Ensemble consensus column 'consensus_score' contains non-numeric values" in res.stderr
    assert not manifest.exists()


def test_stage7_prep_rejects_boolean_consensus_scores(tmp_path: Path) -> None:
    consensus = tmp_path / "ensemble.tsv"
    consensus.write_text("target_id\tconsensus_score\nP1\tTrue\n")
    ligand = tmp_path / "ligand.sdf"
    write_valid_ligand_sdf(ligand)
    manifest = tmp_path / "md.tsv"
    manifest.write_text("stale\n")

    res = run_script(
        "stage7_gromacs_prep.py",
        [
            "--ensemble-consensus", str(consensus),
            "--receptor-manifest", str(tmp_path / "receptors.tsv"),
            "--ligand-sdf", str(ligand),
            "--out-dir", str(tmp_path / "out"),
            "--out-manifest", str(manifest),
        ],
        tmp_path,
    )

    assert res.returncode != 0
    assert (
        "Ensemble consensus column 'consensus_score' contains non-numeric values"
        in res.stderr
    )
    assert not manifest.exists()


def test_stage7_prep_rejects_consensus_non_finite_scores(tmp_path: Path) -> None:
    consensus = tmp_path / "ensemble.tsv"
    consensus.write_text("target_id\tconsensus_score\nP1\tinf\n")
    ligand = tmp_path / "ligand.sdf"
    write_valid_ligand_sdf(ligand)
    manifest = tmp_path / "md.tsv"
    manifest.write_text("stale\n")

    res = run_script(
        "stage7_gromacs_prep.py",
        [
            "--ensemble-consensus", str(consensus),
            "--receptor-manifest", str(tmp_path / "receptors.tsv"),
            "--ligand-sdf", str(ligand),
            "--out-dir", str(tmp_path / "out"),
            "--out-manifest", str(manifest),
        ],
        tmp_path,
    )

    assert res.returncode != 0
    assert "Ensemble consensus column 'consensus_score' contains non-finite values" in res.stderr
    assert not manifest.exists()


def test_stage7_prep_rejects_non_positive_consensus_scores(tmp_path: Path) -> None:
    consensus = tmp_path / "ensemble.tsv"
    pd.DataFrame([
        {"target_id": "P1", "consensus_score": 0.0},
        {"target_id": "P2", "consensus_score": -1.0},
    ]).to_csv(consensus, sep="\t", index=False)
    ligand = tmp_path / "ligand.sdf"
    write_valid_ligand_sdf(ligand)
    manifest = tmp_path / "md.tsv"
    manifest.write_text("stale\n")

    res = run_script(
        "stage7_gromacs_prep.py",
        [
            "--ensemble-consensus", str(consensus),
            "--receptor-manifest", str(tmp_path / "receptors.tsv"),
            "--ligand-sdf", str(ligand),
            "--out-dir", str(tmp_path / "out"),
            "--out-manifest", str(manifest),
        ],
        tmp_path,
    )

    assert res.returncode != 0
    assert "Ensemble consensus column 'consensus_score' must be > 0" in res.stderr
    assert "row index(es) 0, 1" in res.stderr
    assert not manifest.exists()


def test_stage7_prep_rejects_duplicate_consensus_target_ids(tmp_path: Path) -> None:
    consensus = tmp_path / "ensemble.tsv"
    pd.DataFrame([
        {"target_id": "P1", "consensus_score": 2.0},
        {"target_id": "P1", "consensus_score": 1.0},
    ]).to_csv(consensus, sep="\t", index=False)
    ligand = tmp_path / "ligand.sdf"
    write_valid_ligand_sdf(ligand)
    manifest = tmp_path / "md.tsv"
    manifest.write_text("stale\n")

    res = run_script(
        "stage7_gromacs_prep.py",
        [
            "--ensemble-consensus", str(consensus),
            "--receptor-manifest", str(tmp_path / "receptors.tsv"),
            "--ligand-sdf", str(ligand),
            "--out-dir", str(tmp_path / "out"),
            "--out-manifest", str(manifest),
        ],
        tmp_path,
    )

    assert res.returncode != 0
    assert "Ensemble consensus contains duplicate target_id values: P1" in res.stderr
    assert not manifest.exists()


@pytest.mark.parametrize("top_n", ["0", "-1"])
def test_stage7_prep_rejects_invalid_top_n_without_manifest(
    tmp_path: Path,
    top_n: str,
) -> None:
    consensus = tmp_path / "ensemble.tsv"
    pd.DataFrame([{"target_id": "P1", "consensus_score": 2.0}]).to_csv(
        consensus, sep="\t", index=False
    )
    ligand = tmp_path / "ligand.sdf"
    write_valid_ligand_sdf(ligand)
    manifest = tmp_path / "md.tsv"
    manifest.write_text("stale\n")

    res = run_script(
        "stage7_gromacs_prep.py",
        [
            "--ensemble-consensus", str(consensus),
            "--receptor-manifest", str(tmp_path / "receptors.tsv"),
            "--ligand-sdf", str(ligand),
            "--out-dir", str(tmp_path / "out"),
            "--top-n", top_n,
            "--out-manifest", str(manifest),
        ],
        tmp_path,
    )

    assert res.returncode != 0
    assert f"--top-n must be >= 1: {top_n}" in res.stderr
    assert not manifest.exists()


def test_stage7_prep_rejects_empty_ligand_without_manifest(tmp_path: Path) -> None:
    consensus = tmp_path / "ensemble.tsv"
    pd.DataFrame([{"target_id": "P1", "consensus_score": 2.0}]).to_csv(
        consensus, sep="\t", index=False
    )
    ligand = tmp_path / "ligand.sdf"
    ligand.write_text("")
    receptor = tmp_path / "P1.pdb"
    receptor.write_text("receptor\n")
    receptor_manifest = tmp_path / "receptors.tsv"
    pd.DataFrame([{"target_id": "P1", "medoid_pdb": str(receptor)}]).to_csv(
        receptor_manifest, sep="\t", index=False
    )
    manifest = tmp_path / "md.tsv"
    manifest.write_text("stale\n")

    res = run_script(
        "stage7_gromacs_prep.py",
        [
            "--ensemble-consensus", str(consensus),
            "--receptor-manifest", str(receptor_manifest),
            "--ligand-sdf", str(ligand),
            "--out-dir", str(tmp_path / "out"),
            "--out-manifest", str(manifest),
        ],
        tmp_path,
    )

    assert res.returncode != 0
    assert "Ligand SDF is missing or empty for GROMACS prep" in res.stderr
    assert not manifest.exists()


def test_stage7_prep_rejects_empty_prepared_system_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    spec = importlib.util.spec_from_file_location(
        "stage7_gromacs_prep_under_test_empty",
        ROOT / "scripts/stage7_gromacs_prep.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    consensus = tmp_path / "ensemble.tsv"
    pd.DataFrame([{"target_id": "P1", "consensus_score": 2.0}]).to_csv(
        consensus, sep="\t", index=False
    )
    ligand = tmp_path / "ligand.sdf"
    write_valid_ligand_sdf(ligand)
    receptor = tmp_path / "P1.pdb"
    receptor.write_text("receptor\n")
    receptor_manifest = tmp_path / "receptors.tsv"
    pd.DataFrame([{"target_id": "P1", "medoid_pdb": str(receptor)}]).to_csv(
        receptor_manifest, sep="\t", index=False
    )
    manifest = tmp_path / "md.tsv"
    manifest.write_text("stale\n")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage7_gromacs_prep.py",
            "--ensemble-consensus",
            str(consensus),
            "--receptor-manifest",
            str(receptor_manifest),
            "--ligand-sdf",
            str(ligand),
            "--out-dir",
            str(tmp_path / "out"),
            "--out-manifest",
            str(manifest),
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        module.main()

    assert "Stage7 GROMACS preparation is fail-closed" in str(excinfo.value)
    assert not manifest.exists()


def test_stage7_prep_fails_closed_without_complex_pose_contract(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log = tmp_path / "commands.log"
    fake_acpype = fake_bin / "acpype"
    fake_acpype.write_text(
        "#!/bin/sh\n"
        "printf 'acpype %s\\n' \"$*\" >> '" + str(log) + "'\n"
    )
    fake_acpype.chmod(0o755)
    fake_gmx = fake_bin / "gmx"
    fake_gmx.write_text(
        "#!/bin/sh\n"
        "printf 'gmx %s\\n' \"$*\" >> '" + str(log) + "'\n"
        "sub=\"$1\"\n"
        "shift\n"
        "out=''\n"
        "top=''\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = '-o' ]; then shift; out=\"$1\"; fi\n"
        "  if [ \"$1\" = '-p' ]; then shift; top=\"$1\"; fi\n"
        "  shift\n"
        "done\n"
        "if [ \"$sub\" = 'pdb2gmx' ]; then\n"
        "  printf 'fresh system\\n' > \"$out\"\n"
        "  printf 'fresh topology\\n' > \"$top\"\n"
        "fi\n"
        "if [ \"$sub\" = 'grompp' ]; then printf 'fresh tpr\\n' > \"$out\"; fi\n"
    )
    fake_gmx.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = str(fake_bin)

    consensus = tmp_path / "ensemble.tsv"
    pd.DataFrame([{"target_id": "P1", "consensus_score": 2.0}]).to_csv(
        consensus, sep="\t", index=False
    )
    receptor = tmp_path / "P1.pdb"
    receptor.write_text("receptor\n")
    receptor_manifest = tmp_path / "receptors.tsv"
    pd.DataFrame([{"target_id": "P1", "medoid_pdb": str(receptor)}]).to_csv(
        receptor_manifest, sep="\t", index=False
    )
    ligand = tmp_path / "ligand.sdf"
    write_valid_ligand_sdf(ligand)
    target_dir = tmp_path / "out" / "P1"
    target_dir.mkdir(parents=True)
    (target_dir / "system.gro").write_text("stale system\n")
    (target_dir / "topol.top").write_text("stale topology\n")
    (target_dir / "prod.tpr").write_text("stale tpr\n")
    manifest = tmp_path / "md.tsv"

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage7_gromacs_prep.py"),
            "--ensemble-consensus",
            str(consensus),
            "--receptor-manifest",
            str(receptor_manifest),
            "--ligand-sdf",
            str(ligand),
            "--out-dir",
            str(tmp_path / "out"),
            "--out-manifest",
            str(manifest),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "Stage7 GROMACS preparation is fail-closed" in res.stderr
    assert not manifest.exists()
    assert (target_dir / "system.gro").read_text() == "stale system\n"
    assert (target_dir / "topol.top").read_text() == "stale topology\n"
    assert (target_dir / "prod.tpr").read_text() == "stale tpr\n"
    assert not log.exists()


def test_stage7_failed_mdrun_does_not_leave_index(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_gmx = fake_bin / "gmx"
    fake_gmx.write_text("#!/bin/sh\nexit 1\n")
    fake_gmx.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = str(fake_bin)

    target_dir = tmp_path / "out" / "P1"
    target_dir.mkdir(parents=True)
    system = target_dir / "system.gro"
    topology = target_dir / "topol.top"
    tpr = target_dir / "prod.tpr"
    system.write_text("system\n")
    topology.write_text("topology\n")
    tpr.write_text("tpr\n")
    # 복제별 tpr 은 prod.mdp 에서 만들어진다. 준비 단계가 늘 남기는 파일이다.
    (target_dir / "prod.mdp").write_text("nsteps = 10\ngen_vel = no\n")
    manifest = tmp_path / "md.tsv"
    pd.DataFrame([ready_md_manifest_row(target_dir, target_id="P1")]).to_csv(
        manifest, sep="\t", index=False
    )
    out_index = tmp_path / "traj.tsv"
    out_index.write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage7_gromacs_run.py"),
            "--manifest", str(manifest),
            "--out-dir", str(tmp_path / "out"),
            "--replicas", "1",
            "--out-index", str(out_index),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "GROMACS completed 0/1" in res.stderr
    assert not out_index.exists()


def test_stage7_partial_mdrun_removes_all_trajectories(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_gmx = fake_bin / "gmx"
    fake_gmx.write_text(
        "#!/bin/sh\n"
        # 복제별 tpr 을 만드는 grompp 는 통과시킨다. 이 테스트가 재려는 것은
        # mdrun 이 일부만 성공했을 때 궤적을 남기지 않는지다.
        "if [ \"$1\" = 'grompp' ]; then\n"
        "  shift\n"
        "  o=''\n"
        "  while [ \"$#\" -gt 0 ]; do\n"
        "    if [ \"$1\" = '-o' ]; then shift; o=\"$1\"; fi\n"
        "    shift\n"
        "  done\n"
        "  printf 'tpr\\n' > \"$o\"\n"
        "  exit 0\n"
        "fi\n"
        "out=''\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = '-deffnm' ]; then shift; out=\"$1\"; fi\n"
        "  shift\n"
        "done\n"
        "case \"$out\" in\n"
        "  */P1/prod_r1) printf 'trajectory\\n' > \"${out}.xtc\"; exit 0 ;;\n"
        "  *) exit 1 ;;\n"
        "esac\n"
    )
    fake_gmx.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = str(fake_bin)

    rows = []
    trajectories = []
    for uid in ("P1", "P2"):
        target_dir = tmp_path / "out" / uid
        target_dir.mkdir(parents=True)
        system = target_dir / "system.gro"
        topology = target_dir / "topol.top"
        tpr = target_dir / "prod.tpr"
        system.write_text("system\n")
        topology.write_text("topology\n")
        tpr.write_text("tpr\n")
        (target_dir / "prod.mdp").write_text("nsteps = 10\ngen_vel = no\n")
        trajectories.append(target_dir / "prod_r1.xtc")
        rows.append(ready_md_manifest_row(target_dir, target_id=uid))
    manifest = tmp_path / "md.tsv"
    pd.DataFrame(rows).to_csv(manifest, sep="\t", index=False)
    out_index = tmp_path / "traj.tsv"

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage7_gromacs_run.py"),
            "--manifest", str(manifest),
            "--out-dir", str(tmp_path / "out"),
            "--replicas", "1",
            "--out-index", str(out_index),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "GROMACS completed 1/2" in res.stderr
    assert all(not trajectory.exists() for trajectory in trajectories)
    assert not out_index.exists()


def test_stage7_mdrun_removes_stale_xtc_before_success_check(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_gmx = fake_bin / "gmx"
    fake_gmx.write_text("#!/bin/sh\nexit 0\n")
    fake_gmx.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = str(fake_bin)

    target_dir = tmp_path / "out" / "P1"
    target_dir.mkdir(parents=True)
    system = target_dir / "system.gro"
    topology = target_dir / "topol.top"
    tpr = target_dir / "prod.tpr"
    stale_xtc = target_dir / "prod_r1.xtc"
    system.write_text("system\n")
    topology.write_text("topology\n")
    tpr.write_text("tpr\n")
    stale_xtc.write_text("stale trajectory\n")
    manifest = tmp_path / "md.tsv"
    pd.DataFrame([ready_md_manifest_row(target_dir, target_id="P1")]).to_csv(
        manifest, sep="\t", index=False
    )
    out_index = tmp_path / "traj.tsv"

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage7_gromacs_run.py"),
            "--manifest", str(manifest),
            "--out-dir", str(tmp_path / "out"),
            "--replicas", "1",
            "--out-index", str(out_index),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "GROMACS completed 0/1" in res.stderr
    assert not stale_xtc.exists()
    assert not out_index.exists()


def test_stage7_mdrun_exit_zero_without_xtc_fails(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_gmx = fake_bin / "gmx"
    fake_gmx.write_text("#!/bin/sh\nexit 0\n")
    fake_gmx.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = str(fake_bin)

    target_dir = tmp_path / "out" / "P1"
    target_dir.mkdir(parents=True)
    system = target_dir / "system.gro"
    topology = target_dir / "topol.top"
    tpr = target_dir / "prod.tpr"
    system.write_text("system\n")
    topology.write_text("topology\n")
    tpr.write_text("tpr\n")
    # 복제별 tpr 은 prod.mdp 에서 만들어진다. 준비 단계가 늘 남기는 파일이다.
    (target_dir / "prod.mdp").write_text("nsteps = 10\ngen_vel = no\n")
    manifest = tmp_path / "md.tsv"
    pd.DataFrame([ready_md_manifest_row(target_dir, target_id="P1")]).to_csv(
        manifest, sep="\t", index=False
    )
    out_index = tmp_path / "traj.tsv"

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage7_gromacs_run.py"),
            "--manifest", str(manifest),
            "--out-dir", str(tmp_path / "out"),
            "--replicas", "1",
            "--out-index", str(out_index),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "GROMACS completed 0/1" in res.stderr
    assert not out_index.exists()


def test_stage7_mdrun_exit_zero_with_empty_xtc_fails(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_gmx = fake_bin / "gmx"
    fake_gmx.write_text(
        "#!/bin/sh\n"
        "out=''\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = '-deffnm' ]; then shift; out=\"$1\"; fi\n"
        "  shift\n"
        "done\n"
        ": > \"${out}.xtc\"\n"
    )
    fake_gmx.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = str(fake_bin)

    target_dir = tmp_path / "out" / "P1"
    target_dir.mkdir(parents=True)
    system = target_dir / "system.gro"
    topology = target_dir / "topol.top"
    tpr = target_dir / "prod.tpr"
    system.write_text("system\n")
    topology.write_text("topology\n")
    tpr.write_text("tpr\n")
    # 복제별 tpr 은 prod.mdp 에서 만들어진다. 준비 단계가 늘 남기는 파일이다.
    (target_dir / "prod.mdp").write_text("nsteps = 10\ngen_vel = no\n")
    manifest = tmp_path / "md.tsv"
    pd.DataFrame([ready_md_manifest_row(target_dir, target_id="P1")]).to_csv(
        manifest, sep="\t", index=False
    )
    out_index = tmp_path / "traj.tsv"

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage7_gromacs_run.py"),
            "--manifest", str(manifest),
            "--out-dir", str(tmp_path / "out"),
            "--replicas", "1",
            "--out-index", str(out_index),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "GROMACS completed 0/1" in res.stderr
    assert not out_index.exists()


def test_stage7_successful_mdrun_requires_and_records_xtc(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_gmx = fake_bin / "gmx"
    log = tmp_path / "mdrun.log"
    fake_gmx.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = 'grompp' ]; then\n"
        # 복제마다 자기 tpr 을 만드는 단계. 실제 gmx 처럼 -o 를 써 준다.
        "  shift\n"
        "  o=''\n"
        "  while [ \"$#\" -gt 0 ]; do\n"
        "    if [ \"$1\" = '-o' ]; then shift; o=\"$1\"; fi\n"
        "    shift\n"
        "  done\n"
        "  printf 'tpr\\n' > \"$o\"\n"
        "  exit 0\n"
        "fi\n"
        "out=''\n"
        "tpr=''\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = '-s' ]; then shift; tpr=\"$1\"; fi\n"
        "  if [ \"$1\" = '-deffnm' ]; then shift; out=\"$1\"; fi\n"
        "  shift\n"
        "done\n"
        "printf 'tpr=%s\\n' \"$tpr\" > '" + str(log) + "'\n"
        "if [ ! -s \"$tpr\" ]; then exit 2; fi\n"
        "printf 'trajectory\\n' > \"${out}.xtc\"\n"
    )
    fake_gmx.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = str(fake_bin)

    target_dir = tmp_path / "out" / "P1"
    target_dir.mkdir(parents=True)
    system = target_dir / "system.gro"
    topology = target_dir / "topol.top"
    tpr = target_dir / "prod.tpr"
    system.write_text("system\n")
    topology.write_text("topology\n")
    tpr.write_text("tpr\n")
    # 복제별 tpr 은 prod.mdp 에서 만들어진다. 준비 단계가 늘 남기는 파일이다.
    (target_dir / "prod.mdp").write_text("nsteps = 10\ngen_vel = no\n")
    manifest = tmp_path / "md.tsv"
    pd.DataFrame([ready_md_manifest_row(target_dir, target_id="P1")]).to_csv(
        manifest, sep="\t", index=False
    )
    out_index = tmp_path / "traj.tsv"

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage7_gromacs_run.py"),
            "--manifest", str(manifest),
            "--out-dir", str(tmp_path / "out"),
            "--replicas", "1",
            "--out-index", str(out_index),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode == 0, res.stderr
    # 복제는 자기 tpr 로 돈다. 공유 tpr 을 쓰면 두 실행의 시작 속도와 열욕 씨앗이
    # 같아져, 독립 표본이 아닌 것을 복제라고 부르게 된다.
    assert log.read_text() == f"tpr={tpr.parent / 'prod_r1.tpr'}\n"
    out = pd.read_csv(out_index, sep="\t")
    assert out["target_id"].tolist() == ["P1"]
    assert Path(out.iloc[0]["trajectory_xtc"]).exists()


def test_stage7_failed_mmgbsa_does_not_leave_report(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_mmpbsa = fake_bin / "gmx_MMPBSA"
    fake_mmpbsa.write_text("#!/bin/sh\nexit 0\n")
    fake_mmpbsa.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = str(fake_bin)

    target_dir = tmp_path / "P1"
    target_dir.mkdir()
    traj_index = tmp_path / "traj.tsv"
    pd.DataFrame([trajectory_index_row(target_dir, target_id="P1")]).to_csv(
        traj_index, sep="\t", index=False
    )
    report = tmp_path / "mmgbsa.tsv"
    report.write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage7_mmgbsa.py"),
            "--trajectory-index", str(traj_index),
            "--out-report", str(report),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "MM-GBSA produced no successful target energies" in res.stderr
    assert not report.exists()


def test_stage7_nonfinite_mmgbsa_energy_does_not_leave_report(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_mmpbsa = fake_bin / "gmx_MMPBSA"
    fake_mmpbsa.write_text(
        "#!/bin/sh\nprintf 'DELTA TOTAL    inf\\n' > FINAL_RESULTS_MMPBSA.dat\n"
    )
    fake_mmpbsa.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = str(fake_bin)

    target_dir = tmp_path / "P1"
    target_dir.mkdir()
    traj_index = tmp_path / "traj.tsv"
    pd.DataFrame([trajectory_index_row(target_dir, target_id="P1")]).to_csv(
        traj_index, sep="\t", index=False
    )
    report = tmp_path / "mmgbsa.tsv"
    report.write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage7_mmgbsa.py"),
            "--trajectory-index", str(traj_index),
            "--out-report", str(report),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "MM-GBSA produced no successful target energies" in res.stderr
    assert not report.exists()


def test_stage7_mmgbsa_rejects_empty_trajectory_index_without_report(
    tmp_path: Path,
) -> None:
    traj_index = tmp_path / "traj.tsv"
    traj_index.write_text("")
    report = tmp_path / "mmgbsa.tsv"
    report.write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage7_mmgbsa.py"),
            "--trajectory-index", str(traj_index),
            "--out-report", str(report),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "Trajectory index is required and must be non-empty" in res.stderr
    assert not report.exists()


def test_stage7_mmgbsa_rejects_trajectory_index_missing_required_columns(
    tmp_path: Path,
) -> None:
    traj_index = tmp_path / "traj.tsv"
    pd.DataFrame([{"target_id": "P1", "status": "ok"}]).to_csv(
        traj_index, sep="\t", index=False
    )
    report = tmp_path / "mmgbsa.tsv"
    report.write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage7_mmgbsa.py"),
            "--trajectory-index", str(traj_index),
            "--out-report", str(report),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "Trajectory index missing required columns" in res.stderr
    assert "trajectory_xtc" in res.stderr
    assert not report.exists()


def test_stage7_mmgbsa_rejects_blank_trajectory_target_id(tmp_path: Path) -> None:
    traj_index = tmp_path / "traj.tsv"
    pd.DataFrame(
        [{"target_id": " ", "trajectory_xtc": str(tmp_path / "prod.xtc"), "status": "ok"}]
    ).to_csv(traj_index, sep="\t", index=False)
    report = tmp_path / "mmgbsa.tsv"
    report.write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage7_mmgbsa.py"),
            "--trajectory-index", str(traj_index),
            "--out-report", str(report),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "Trajectory index column 'target_id' contains blank values" in res.stderr
    assert not report.exists()


def test_stage7_mmgbsa_rejects_invalid_trajectory_status(tmp_path: Path) -> None:
    traj_index = tmp_path / "traj.tsv"
    pd.DataFrame(
        [
            {
                "target_id": "P1",
                "trajectory_xtc": str(tmp_path / "prod.xtc"),
                "status": "failed",
            }
        ]
    ).to_csv(traj_index, sep="\t", index=False)
    report = tmp_path / "mmgbsa.tsv"
    report.write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage7_mmgbsa.py"),
            "--trajectory-index", str(traj_index),
            "--out-report", str(report),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "Trajectory index column 'status' contains invalid values" in res.stderr
    assert "failed" in res.stderr
    assert not report.exists()


def test_stage7_mmgbsa_rejects_duplicate_trajectory_paths(tmp_path: Path) -> None:
    trajectory = tmp_path / "P1" / "prod_r1.xtc"
    trajectory.parent.mkdir()
    trajectory.write_text("trajectory\n")
    traj_index = tmp_path / "traj.tsv"
    pd.DataFrame(
        [
            {"target_id": "P1", "trajectory_xtc": str(trajectory), "status": "ok"},
            {"target_id": "P2", "trajectory_xtc": str(trajectory), "status": "ok"},
        ]
    ).to_csv(traj_index, sep="\t", index=False)
    report = tmp_path / "mmgbsa.tsv"
    report.write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage7_mmgbsa.py"),
            "--trajectory-index", str(traj_index),
            "--out-report", str(report),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "Trajectory index contains duplicate trajectory_xtc values" in res.stderr
    assert str(trajectory) in res.stderr
    assert not report.exists()


def test_stage7_mmgbsa_rejects_canonical_duplicate_trajectory_paths(
    tmp_path: Path,
) -> None:
    trajectory = tmp_path / "P1" / "prod_r1.xtc"
    trajectory.parent.mkdir()
    trajectory.write_text("trajectory\n")
    traj_index = tmp_path / "traj.tsv"
    pd.DataFrame(
        [
            {"target_id": "P1", "trajectory_xtc": str(trajectory), "status": "ok"},
            {
                "target_id": "P2",
                "trajectory_xtc": f"{trajectory.parent}/./{trajectory.name}",
                "status": "ok",
            },
        ]
    ).to_csv(traj_index, sep="\t", index=False)
    report = tmp_path / "mmgbsa.tsv"
    report.write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage7_mmgbsa.py"),
            "--trajectory-index", str(traj_index),
            "--out-report", str(report),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "Trajectory index contains duplicate trajectory_xtc values" in res.stderr
    assert str(trajectory.resolve()) in res.stderr
    assert not report.exists()


def test_stage7_empty_trajectory_does_not_leave_mmgbsa_report(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_mmpbsa = fake_bin / "gmx_MMPBSA"
    fake_mmpbsa.write_text(
        "#!/bin/sh\n"
        "printf 'DELTA TOTAL    -12.345\\n' > FINAL_RESULTS_MMPBSA.dat\n"
    )
    fake_mmpbsa.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = str(fake_bin)

    target_dir = tmp_path / "P1"
    target_dir.mkdir()
    traj_index = tmp_path / "traj.tsv"
    pd.DataFrame([trajectory_index_row(
        target_dir, target_id="P1", trajectory_body=""
    )]).to_csv(traj_index, sep="\t", index=False)
    report = tmp_path / "mmgbsa.tsv"
    report.write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage7_mmgbsa.py"),
            "--trajectory-index", str(traj_index),
            "--out-report", str(report),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "trajectory_xtc is missing or empty" in res.stderr
    assert not report.exists()


def test_stage7_partial_mmgbsa_failure_does_not_leave_report(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_mmpbsa = fake_bin / "gmx_MMPBSA"
    fake_mmpbsa.write_text(
        "#!/bin/sh\n"
        # 실제 도구는 --create_input 으로 표준 입력 파일을 만들어 준다. 손으로
        # 적은 네임리스트는 파서가 받아 주지 않으므로 코드가 이 경로를 쓴다.
        "if [ \"$1\" = '--create_input' ]; then\n"
        "  printf '&general\\n  sys_name = \"\"\\n  startframe = 1\\n/\\n"
        "&gb\\n  igb = 5\\n  saltcon = 0.0\\n/\\n' > mmpbsa.in\n"
        "  exit 0\n"
        "fi\n"
        "case \"${PWD##*/}\" in\n"
        "  P1) printf 'DELTA TOTAL    -12.345\\n' > FINAL_RESULTS_MMPBSA.dat ;;\n"
        "  *) exit 0 ;;\n"
        "esac\n"
    )
    fake_mmpbsa.chmod(0o755)
    # MM-GBSA 는 인덱스 파일과 프레임 수를 gmx 로 얻는다. 대역에 없으면
    # 실행이 FileNotFoundError 로 죽어, 무엇을 재려던 테스트인지 사라진다.
    fake_gmx = fake_bin / "gmx"
    fake_gmx.write_text(
        "#!/bin/sh\n"
        "sub=\"$1\"; shift\n"
        "out=''\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = '-o' ]; then shift; out=\"$1\"; fi\n"
        "  shift\n"
        "done\n"
        "case \"$sub\" in\n"
        "  make_ndx) printf '[ Protein ]\\n 1 2 3\\n[ Other ]\\n 4 5\\n' > \"$out\" ;;\n"
        "  check) printf 'Step  10  1000\\n' >&2 ;;\n"
        # MM-GBSA 는 궤적을 넘기기 전에 trjconv 로 주기 경계를 제거한다.
        # 대역에 없으면 그 단계가 실패해, 테스트가 재려던 실패 대신 다른
        # 실패를 보게 된다.
        "  trjconv) printf 'whole trajectory\\n' > \"$out\" ;;\n"
        "esac\n"
    )
    fake_gmx.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = str(fake_bin)

    rows = []
    for uid in ("P1", "P2"):
        target_dir = tmp_path / uid
        target_dir.mkdir()
        # 실제 스테이지 7 출력 디렉터리에는 궤적 옆에 구조와 토폴로지가 함께
        # 있다. 그것이 없는 픽스처는 실제보다 좁아서, MM-GBSA 가 정말로 계산에
        # 실패한 경우와 입력이 없어 시작도 못 한 경우를 구분하지 못한다.
        rows.append(trajectory_index_row(target_dir, target_id=uid))
    traj_index = tmp_path / "traj.tsv"
    pd.DataFrame(rows).to_csv(traj_index, sep="\t", index=False)
    report = tmp_path / "mmgbsa.tsv"
    report.write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage7_mmgbsa.py"),
            "--trajectory-index", str(traj_index),
            "--out-report", str(report),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "MM-GBSA failed for final targets P2" in res.stderr
    assert not report.exists()


def test_stage7_mmgbsa_partial_report_requires_explicit_flag(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_mmpbsa = fake_bin / "gmx_MMPBSA"
    fake_mmpbsa.write_text(
        "#!/bin/sh\n"
        # 실제 도구는 --create_input 으로 표준 입력 파일을 만들어 준다. 손으로
        # 적은 네임리스트는 파서가 받아 주지 않으므로 코드가 이 경로를 쓴다.
        "if [ \"$1\" = '--create_input' ]; then\n"
        "  printf '&general\\n  sys_name = \"\"\\n  startframe = 1\\n/\\n"
        "&gb\\n  igb = 5\\n  saltcon = 0.0\\n/\\n' > mmpbsa.in\n"
        "  exit 0\n"
        "fi\n"
        "case \"${PWD##*/}\" in\n"
        "  P1) printf 'DELTA TOTAL    -12.345\\n' > FINAL_RESULTS_MMPBSA.dat ;;\n"
        "  *) exit 0 ;;\n"
        "esac\n"
    )
    fake_mmpbsa.chmod(0o755)
    # MM-GBSA 는 인덱스 파일과 프레임 수를 gmx 로 얻는다. 대역에 없으면
    # 실행이 FileNotFoundError 로 죽어, 무엇을 재려던 테스트인지 사라진다.
    fake_gmx = fake_bin / "gmx"
    fake_gmx.write_text(
        "#!/bin/sh\n"
        "sub=\"$1\"; shift\n"
        "out=''\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = '-o' ]; then shift; out=\"$1\"; fi\n"
        "  shift\n"
        "done\n"
        "case \"$sub\" in\n"
        "  make_ndx) printf '[ Protein ]\\n 1 2 3\\n[ Other ]\\n 4 5\\n' > \"$out\" ;;\n"
        "  check) printf 'Step  10  1000\\n' >&2 ;;\n"
        # MM-GBSA 는 궤적을 넘기기 전에 trjconv 로 주기 경계를 제거한다.
        # 대역에 없으면 그 단계가 실패해, 테스트가 재려던 실패 대신 다른
        # 실패를 보게 된다.
        "  trjconv) printf 'whole trajectory\\n' > \"$out\" ;;\n"
        "esac\n"
    )
    fake_gmx.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = str(fake_bin)

    rows = []
    for uid in ("P1", "P2"):
        target_dir = tmp_path / uid
        target_dir.mkdir()
        # 실제 스테이지 7 출력 디렉터리에는 궤적 옆에 구조와 토폴로지가 함께
        # 있다. 그것이 없는 픽스처는 실제보다 좁아서, MM-GBSA 가 정말로 계산에
        # 실패한 경우와 입력이 없어 시작도 못 한 경우를 구분하지 못한다.
        rows.append(trajectory_index_row(target_dir, target_id=uid))
    traj_index = tmp_path / "traj.tsv"
    pd.DataFrame(rows).to_csv(traj_index, sep="\t", index=False)
    report = tmp_path / "mmgbsa.tsv"

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage7_mmgbsa.py"),
            "--trajectory-index", str(traj_index),
            "--out-report", str(report),
            "--allow-partial-output",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode == 0, res.stderr
    out = pd.read_csv(report, sep="\t")
    assert out["target_id"].tolist() == ["P1"]
    assert out["status"].tolist() == ["ok"]


def test_stage8_crest_rejects_invalid_mmgbsa_status_without_manifest(
    tmp_path: Path,
) -> None:
    report = tmp_path / "mmgbsa.tsv"
    pd.DataFrame(
        [{"target_id": "P1", "mmgbsa_dg_kcal_mol": "", "status": "skipped"}]
    ).to_csv(report, sep="\t", index=False)
    ligand = tmp_path / "ligand.sdf"
    write_valid_ligand_sdf(ligand)
    manifest = tmp_path / "crest.tsv"
    manifest.write_text("stale\n")
    res = run_script(
        "stage8_crest.py",
        [
            "--mmgbsa-report", str(report),
            "--ligand-sdf", str(ligand),
            "--out-dir", str(tmp_path / "out"),
            "--out-manifest", str(manifest),
        ],
        tmp_path,
    )
    assert res.returncode != 0
    assert "MM-GBSA report column 'status' contains invalid values" in res.stderr
    assert not manifest.exists()


def test_stage8_crest_rejects_empty_mmgbsa_report_without_manifest(
    tmp_path: Path,
) -> None:
    report = tmp_path / "mmgbsa.tsv"
    report.write_text("")
    ligand = tmp_path / "ligand.sdf"
    ligand.write_text("")
    manifest = tmp_path / "crest.tsv"
    manifest.write_text("stale\n")

    res = run_script(
        "stage8_crest.py",
        [
            "--mmgbsa-report", str(report),
            "--ligand-sdf", str(ligand),
            "--out-dir", str(tmp_path / "out"),
            "--out-manifest", str(manifest),
        ],
        tmp_path,
    )

    assert res.returncode != 0
    assert "MM-GBSA report is required and must be non-empty" in res.stderr
    assert not manifest.exists()


def test_stage8_crest_rejects_mmgbsa_report_missing_required_columns(
    tmp_path: Path,
) -> None:
    report = tmp_path / "mmgbsa.tsv"
    pd.DataFrame([{"target_id": "P1", "status": "ok"}]).to_csv(
        report, sep="\t", index=False
    )
    ligand = tmp_path / "ligand.sdf"
    ligand.write_text("")
    manifest = tmp_path / "crest.tsv"
    manifest.write_text("stale\n")

    res = run_script(
        "stage8_crest.py",
        [
            "--mmgbsa-report", str(report),
            "--ligand-sdf", str(ligand),
            "--out-dir", str(tmp_path / "out"),
            "--out-manifest", str(manifest),
        ],
        tmp_path,
    )

    assert res.returncode != 0
    assert "MM-GBSA report missing required columns" in res.stderr
    assert "mmgbsa_dg_kcal_mol" in res.stderr
    assert not manifest.exists()


def test_stage8_crest_rejects_blank_mmgbsa_target_id_without_manifest(
    tmp_path: Path,
) -> None:
    report = tmp_path / "mmgbsa.tsv"
    pd.DataFrame(
        [{"target_id": " ", "mmgbsa_dg_kcal_mol": -1.0, "status": "ok"}]
    ).to_csv(report, sep="\t", index=False)
    ligand = tmp_path / "ligand.sdf"
    write_valid_ligand_sdf(ligand)
    manifest = tmp_path / "crest.tsv"
    manifest.write_text("stale\n")

    res = run_script(
        "stage8_crest.py",
        [
            "--mmgbsa-report", str(report),
            "--ligand-sdf", str(ligand),
            "--out-dir", str(tmp_path / "out"),
            "--out-manifest", str(manifest),
        ],
        tmp_path,
    )

    assert res.returncode != 0
    assert "MM-GBSA report column 'target_id' contains blank values" in res.stderr
    assert not manifest.exists()


def test_stage8_crest_rejects_duplicate_mmgbsa_target_id_without_manifest(
    tmp_path: Path,
) -> None:
    report = tmp_path / "mmgbsa.tsv"
    pd.DataFrame(
        [
            {"target_id": "P1", "mmgbsa_dg_kcal_mol": -1.0, "status": "ok"},
            {"target_id": "P1", "mmgbsa_dg_kcal_mol": -2.0, "status": "ok"},
        ]
    ).to_csv(report, sep="\t", index=False)
    manifest = tmp_path / "crest.tsv"
    manifest.write_text("stale\n")

    res = run_script(
        "stage8_crest.py",
        [
            "--mmgbsa-report", str(report),
            "--ligand-sdf", str(tmp_path / "ligand.sdf"),
            "--out-dir", str(tmp_path / "crest"),
            "--out-manifest", str(manifest),
        ],
        tmp_path,
    )

    assert res.returncode != 0
    assert (
        "MM-GBSA report contains duplicate target_id values: P1"
        in res.stderr
    )
    assert not manifest.exists()


def test_stage8_crest_rejects_nonnumeric_mmgbsa_energy_without_manifest(
    tmp_path: Path,
) -> None:
    report = tmp_path / "mmgbsa.tsv"
    pd.DataFrame(
        [{"target_id": "P1", "mmgbsa_dg_kcal_mol": "bad", "status": "ok"}]
    ).to_csv(report, sep="\t", index=False)
    ligand = tmp_path / "ligand.sdf"
    write_valid_ligand_sdf(ligand)
    manifest = tmp_path / "crest.tsv"
    manifest.write_text("stale\n")

    res = run_script(
        "stage8_crest.py",
        [
            "--mmgbsa-report", str(report),
            "--ligand-sdf", str(ligand),
            "--out-dir", str(tmp_path / "out"),
            "--out-manifest", str(manifest),
        ],
        tmp_path,
    )

    assert res.returncode != 0
    assert "MM-GBSA report column 'mmgbsa_dg_kcal_mol' must be numeric" in res.stderr
    assert not manifest.exists()


def test_stage8_crest_rejects_boolean_mmgbsa_energy_without_manifest(
    tmp_path: Path,
) -> None:
    report = tmp_path / "mmgbsa.tsv"
    report.write_text("target_id\tmmgbsa_dg_kcal_mol\tstatus\nP1\tTrue\tok\n")
    ligand = tmp_path / "ligand.sdf"
    ligand.write_text("")
    manifest = tmp_path / "crest.tsv"
    manifest.write_text("stale\n")

    res = run_script(
        "stage8_crest.py",
        [
            "--mmgbsa-report", str(report),
            "--ligand-sdf", str(ligand),
            "--out-dir", str(tmp_path / "out"),
            "--out-manifest", str(manifest),
        ],
        tmp_path,
    )

    assert res.returncode != 0
    assert "MM-GBSA report column 'mmgbsa_dg_kcal_mol' must be numeric" in res.stderr
    assert not manifest.exists()


def test_stage8_crest_rejects_nonfinite_mmgbsa_energy_without_manifest(
    tmp_path: Path,
) -> None:
    report = tmp_path / "mmgbsa.tsv"
    report.write_text("target_id\tmmgbsa_dg_kcal_mol\tstatus\nP1\tinf\tok\n")
    ligand = tmp_path / "ligand.sdf"
    ligand.write_text("")
    manifest = tmp_path / "crest.tsv"
    manifest.write_text("stale\n")

    res = run_script(
        "stage8_crest.py",
        [
            "--mmgbsa-report", str(report),
            "--ligand-sdf", str(ligand),
            "--out-dir", str(tmp_path / "out"),
            "--out-manifest", str(manifest),
        ],
        tmp_path,
    )

    assert res.returncode != 0
    assert (
        "MM-GBSA report column 'mmgbsa_dg_kcal_mol' must be finite"
        in res.stderr
    )
    assert not manifest.exists()


def test_stage8_crest_reports_no_selection_for_nonnegative_mmgbsa(
    tmp_path: Path,
) -> None:
    """0/positive ΔTOTAL are valid results; there is simply nothing to refine."""
    report = tmp_path / "mmgbsa.tsv"
    pd.DataFrame(
        [
            {"target_id": "P1", "mmgbsa_dg_kcal_mol": 0.0, "status": "ok"},
            {"target_id": "P2", "mmgbsa_dg_kcal_mol": 1.0, "status": "ok"},
        ]
    ).to_csv(report, sep="\t", index=False)
    ligand = tmp_path / "ligand.sdf"
    ligand.write_text("")
    manifest = tmp_path / "crest.tsv"
    manifest.write_text("stale\n")

    res = run_script(
        "stage8_crest.py",
        [
            "--mmgbsa-report", str(report),
            "--ligand-sdf", str(ligand),
            "--out-dir", str(tmp_path / "out"),
            "--out-manifest", str(manifest),
        ],
        tmp_path,
    )

    assert res.returncode != 0
    assert "No MM-GBSA target has a favorable (negative) delta TOTAL" in res.stderr
    assert "not selected for QM refinement" in res.stderr
    assert "must be < 0" not in res.stderr
    assert not manifest.exists()


def test_stage8_crest_rejects_nonpositive_top_n_without_manifest(
    tmp_path: Path,
) -> None:
    report = tmp_path / "mmgbsa.tsv"
    pd.DataFrame(
        [{"target_id": "P1", "mmgbsa_dg_kcal_mol": -1.0, "status": "ok"}]
    ).to_csv(report, sep="\t", index=False)
    ligand = tmp_path / "ligand.sdf"
    ligand.write_text("")
    manifest = tmp_path / "crest.tsv"
    manifest.write_text("stale\n")

    res = run_script(
        "stage8_crest.py",
        [
            "--mmgbsa-report", str(report),
            "--ligand-sdf", str(ligand),
            "--out-dir", str(tmp_path / "out"),
            "--top-n", "0",
            "--out-manifest", str(manifest),
        ],
        tmp_path,
    )

    assert res.returncode != 0
    assert "--top-n must be at least 1" in res.stderr
    assert not manifest.exists()


def test_stage8_failed_crest_does_not_leave_manifest(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_obabel = fake_bin / "obabel"
    fake_obabel.write_text(
        "#!/bin/sh\n"
        "out=''\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = '-O' ]; then shift; out=\"$1\"; fi\n"
        "  shift\n"
        "done\n"
        "printf '1\\nligand\\nH 0 0 0\\n' > \"$out\"\n"
        "exit 0\n"
    )
    fake_obabel.chmod(0o755)
    fake_crest = fake_bin / "crest"
    fake_crest.write_text("#!/bin/sh\nexit 0\n")
    fake_crest.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = str(fake_bin)

    report = tmp_path / "mmgbsa.tsv"
    pd.DataFrame(
        [{"target_id": "P1", "mmgbsa_dg_kcal_mol": -1.0, "status": "ok"}]
    ).to_csv(report, sep="\t", index=False)
    ligand = tmp_path / "ligand.sdf"
    write_valid_ligand_sdf(ligand)
    manifest = tmp_path / "crest.tsv"

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage8_crest.py"),
            "--mmgbsa-report", str(report),
            "--ligand-sdf", str(ligand),
            "--out-dir", str(tmp_path / "out"),
            "--out-manifest", str(manifest),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "CREST produced no conformer ensembles" in res.stderr
    assert not manifest.exists()


def test_stage8_empty_ligand_xyz_does_not_leave_crest_manifest(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_obabel = fake_bin / "obabel"
    fake_obabel.write_text(
        "#!/bin/sh\n"
        "out=''\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = '-O' ]; then shift; out=\"$1\"; fi\n"
        "  shift\n"
        "done\n"
        ": > \"$out\"\n"
        "exit 0\n"
    )
    fake_obabel.chmod(0o755)
    fake_crest = fake_bin / "crest"
    fake_crest.write_text(
        "#!/bin/sh\n"
        "printf '1\\nconf\\nH 0 0 0\\n' > crest_conformers.xyz\n"
        "exit 0\n"
    )
    fake_crest.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = str(fake_bin)

    report = tmp_path / "mmgbsa.tsv"
    pd.DataFrame(
        [{"target_id": "P1", "mmgbsa_dg_kcal_mol": -1.0, "status": "ok"}]
    ).to_csv(report, sep="\t", index=False)
    ligand = tmp_path / "ligand.sdf"
    ligand.write_text("")
    manifest = tmp_path / "crest.tsv"
    manifest.write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage8_crest.py"),
            "--mmgbsa-report", str(report),
            "--ligand-sdf", str(ligand),
            "--out-dir", str(tmp_path / "out"),
            "--out-manifest", str(manifest),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "Failed to convert ligand SDF to XYZ for CREST" in res.stderr
    assert not manifest.exists()


def test_stage8_partial_crest_failure_does_not_leave_manifest(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_obabel = fake_bin / "obabel"
    fake_obabel.write_text(
        "#!/bin/sh\n"
        "out=''\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = '-O' ]; then shift; out=\"$1\"; fi\n"
        "  shift\n"
        "done\n"
        "printf '1\\nligand\\nH 0 0 0\\n' > \"$out\"\n"
        "exit 0\n"
    )
    fake_obabel.chmod(0o755)
    fake_crest = fake_bin / "crest"
    # CREST 는 자유 리간드에 대해 **한 번** 돈다. 표적이 무엇이든 같은 분자라
    # 답도 같기 때문이다. 그래서 표적별 부분 실패라는 상태가 없다 - 전부
    # 성공하거나 전부 실패한다.
    fake_crest.write_text(
        "#!/bin/sh\n"
        "exit 1\n"
    )
    fake_crest.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = str(fake_bin)

    report = tmp_path / "mmgbsa.tsv"
    pd.DataFrame(
        [
            {"target_id": "P1", "mmgbsa_dg_kcal_mol": -2.0, "status": "ok"},
            {"target_id": "P2", "mmgbsa_dg_kcal_mol": -1.0, "status": "ok"},
        ]
    ).to_csv(report, sep="\t", index=False)
    ligand = tmp_path / "ligand.sdf"
    write_valid_ligand_sdf(ligand)
    manifest = tmp_path / "crest.tsv"
    manifest.write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage8_crest.py"),
            "--mmgbsa-report", str(report),
            "--ligand-sdf", str(ligand),
            "--out-dir", str(tmp_path / "out"),
            "--out-manifest", str(manifest),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "CREST produced no conformer ensembles" in res.stderr
    assert not manifest.exists()


def test_stage8_crest_runs_once_and_every_target_shares_the_ensemble(
    tmp_path: Path,
) -> None:
    """CREST 는 자유 리간드를 본다. 표적이 달라도 같은 분자이므로 답도 같다.

    예전에는 표적마다 한 번씩 돌렸다. 아다팔렌 실측으로 한 번에 2시간 20분이라
    `top_n_for_qm: 3` 이면 같은 답을 얻는 데 7시간을 썼다. 매니페스트의
    `conformer_scope` 는 그때도 이미 "free_ligand" 라고 적고 있었다.
    """
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_obabel = fake_bin / "obabel"
    fake_obabel.write_text(
        "#!/bin/sh\n"
        "out=''\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = '-O' ]; then shift; out=\"$1\"; fi\n"
        "  shift\n"
        "done\n"
        "printf '1\\nligand\\nH 0 0 0\\n' > \"$out\"\n"
        "exit 0\n"
    )
    fake_obabel.chmod(0o755)
    fake_crest = fake_bin / "crest"
    # CREST 는 자유 리간드에 대해 **한 번** 돈다. 표적이 무엇이든 같은 분자라
    # 답도 같기 때문이다. 그래서 표적별 부분 실패라는 상태가 없다 - 전부
    # 성공하거나 전부 실패한다.
    # 몇 번 불렸는지 센다. 한 번이어야 한다.
    counter = tmp_path / "crest_calls"
    fake_crest.write_text(
        "#!/bin/sh\n"
        f"printf 'x' >> {counter}\n"
        "printf '1\\nconf\\nH 0 0 0\\n' > crest_conformers.xyz\n"
    )
    fake_crest.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = str(fake_bin)

    report = tmp_path / "mmgbsa.tsv"
    pd.DataFrame(
        [
            {"target_id": "P1", "mmgbsa_dg_kcal_mol": -2.0, "status": "ok"},
            {"target_id": "P2", "mmgbsa_dg_kcal_mol": -1.0, "status": "ok"},
        ]
    ).to_csv(report, sep="\t", index=False)
    ligand = tmp_path / "ligand.sdf"
    write_valid_ligand_sdf(ligand)
    manifest = tmp_path / "crest.tsv"

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage8_crest.py"),
            "--mmgbsa-report", str(report),
            "--ligand-sdf", str(ligand),
            "--out-dir", str(tmp_path / "out"),
            "--out-manifest", str(manifest),
            "--allow-partial-output",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode == 0, res.stderr
    assert counter.read_text() == "x", (
        f"CREST 가 {len(counter.read_text())}번 불렸습니다. 자유 리간드는 한 번이면 됩니다"
    )
    out = pd.read_csv(manifest, sep="\t")
    assert out["target_id"].tolist() == ["P1", "P2"], "모든 표적이 행을 가져야 합니다"
    assert out["status"].tolist() == ["ok", "ok"]
    assert out["conformer_scope"].tolist() == ["free_ligand", "free_ligand"]
    # 같은 앙상블을 가리킨다.
    assert len(set(out["conformer_xyz"])) == 1
    assert out["charge"].tolist() == [0, 0]
    assert out["spin"].tolist() == [0, 0]


def test_stage8_xtb_failed_singlepoint_does_not_leave_manifest(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_xtb = fake_bin / "xtb"
    fake_xtb.write_text("#!/bin/sh\nexit 0\n")
    fake_xtb.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = str(fake_bin)

    xyz = tmp_path / "conf.xyz"
    xyz.write_text("1\nconf\nH 0 0 0\n")
    crest = tmp_path / "crest.tsv"
    pd.DataFrame([{
        "target_id": "P1",
        "conformer_xyz": str(xyz),
        "status": "ok",
    }]).to_csv(crest, sep="\t", index=False)
    cluster = tmp_path / "clusters.tsv"
    cluster.write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage8_xtb_cluster.py"),
            "--crest-manifest", str(crest),
            "--out-dir", str(tmp_path / "out"),
            "--out-cluster-manifest", str(cluster),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "xTB single-point failed for P1" in res.stderr
    assert not cluster.exists()


def test_stage8_xtb_passes_charge_spin_and_writes_manifest(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    args_log = tmp_path / "xtb_args.txt"
    fake_xtb = fake_bin / "xtb"
    fake_xtb.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$@\" > \"$XTB_ARGS_LOG\"\n"
        "printf ':: TOTAL ENERGY -1.234567\\n'\n"
        "exit 0\n"
    )
    fake_xtb.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = str(fake_bin)
    env["XTB_ARGS_LOG"] = str(args_log)

    xyz = tmp_path / "conf.xyz"
    xyz.write_text("1\nconf\nH 0 0 0\n")
    crest = tmp_path / "crest.tsv"
    pd.DataFrame([{
        "target_id": "P1",
        "conformer_xyz": str(xyz),
        "charge": 1,
        "spin": 2,
        "status": "ok",
    }]).to_csv(crest, sep="\t", index=False)
    cluster = tmp_path / "clusters.tsv"

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage8_xtb_cluster.py"),
            "--crest-manifest", str(crest),
            "--out-dir", str(tmp_path / "out"),
            "--out-cluster-manifest", str(cluster),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode == 0, res.stderr
    assert "--chrg\n1\n--uhf\n2\n" in args_log.read_text()
    out = pd.read_csv(cluster, sep="\t")
    assert out["charge"].tolist() == [1]
    assert out["spin"].tolist() == [2]


def test_stage8_xtb_rejects_nonfinite_energy_without_cluster(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_xtb = fake_bin / "xtb"
    fake_xtb.write_text(
        "#!/bin/sh\n"
        "printf 'TOTAL ENERGY       inf\\n'\n"
        "exit 0\n"
    )
    fake_xtb.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = str(fake_bin)

    xyz = tmp_path / "conf.xyz"
    xyz.write_text("1\nconf\nH 0 0 0\n")
    crest = tmp_path / "crest.tsv"
    pd.DataFrame([{
        "target_id": "P1",
        "conformer_xyz": str(xyz),
        "status": "ok",
    }]).to_csv(crest, sep="\t", index=False)
    cluster = tmp_path / "clusters.tsv"
    cluster.write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage8_xtb_cluster.py"),
            "--crest-manifest", str(crest),
            "--out-dir", str(tmp_path / "out"),
            "--out-cluster-manifest", str(cluster),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "xTB single-point failed for P1" in res.stderr
    assert not cluster.exists()


def test_stage8_xtb_rejects_manifest_missing_required_columns(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_xtb = fake_bin / "xtb"
    fake_xtb.write_text("#!/bin/sh\nexit 0\n")
    fake_xtb.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = str(fake_bin)

    crest = tmp_path / "crest.tsv"
    pd.DataFrame([{"target_id": "P1", "status": "ok"}]).to_csv(
        crest, sep="\t", index=False
    )
    cluster = tmp_path / "clusters.tsv"

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage8_xtb_cluster.py"),
            "--crest-manifest", str(crest),
            "--out-dir", str(tmp_path / "out"),
            "--out-cluster-manifest", str(cluster),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "CREST manifest missing required columns" in res.stderr
    assert not cluster.exists()


def test_stage8_xtb_rejects_empty_crest_manifest_without_cluster(
    tmp_path: Path,
) -> None:
    crest = tmp_path / "crest.tsv"
    crest.write_text("")
    cluster = tmp_path / "clusters.tsv"
    cluster.write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage8_xtb_cluster.py"),
            "--crest-manifest", str(crest),
            "--out-dir", str(tmp_path / "out"),
            "--out-cluster-manifest", str(cluster),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "CREST manifest is required and must be non-empty" in res.stderr
    assert not cluster.exists()


def test_stage8_xtb_rejects_blank_crest_target_id_without_cluster(
    tmp_path: Path,
) -> None:
    crest = tmp_path / "crest.tsv"
    pd.DataFrame([{
        "target_id": " ",
        "conformer_xyz": str(tmp_path / "conf.xyz"),
        "status": "ok",
    }]).to_csv(crest, sep="\t", index=False)
    cluster = tmp_path / "clusters.tsv"
    cluster.write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage8_xtb_cluster.py"),
            "--crest-manifest", str(crest),
            "--out-dir", str(tmp_path / "out"),
            "--out-cluster-manifest", str(cluster),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "CREST manifest column 'target_id' contains blank values" in res.stderr
    assert not cluster.exists()


def test_stage8_xtb_rejects_duplicate_crest_target_id_without_cluster(
    tmp_path: Path,
) -> None:
    crest = tmp_path / "crest.tsv"
    pd.DataFrame(
        [
            {
                "target_id": "P1",
                "conformer_xyz": str(tmp_path / "conf1.xyz"),
                "status": "ok",
            },
            {
                "target_id": "P1",
                "conformer_xyz": str(tmp_path / "conf2.xyz"),
                "status": "ok",
            },
        ]
    ).to_csv(crest, sep="\t", index=False)
    cluster = tmp_path / "clusters.tsv"
    cluster.write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage8_xtb_cluster.py"),
            "--crest-manifest", str(crest),
            "--out-dir", str(tmp_path / "out"),
            "--out-cluster-manifest", str(cluster),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "CREST manifest contains duplicate target_id values: P1" in res.stderr
    assert not cluster.exists()


def test_stage8_xtb_shares_one_free_ligand_ensemble_across_targets(
    tmp_path: Path,
) -> None:
    conf = tmp_path / "conf.xyz"
    # 실패 경로만 재던 시절에는 빈 파일이어도 됐다. 이제 성공 경로를 타므로
    # 실제 XYZ 여야 한다.
    conf.write_text("1\nfree ligand\nC 0.000 0.000 0.000\n")
    crest = tmp_path / "crest.tsv"
    pd.DataFrame(
        [
            {
                "target_id": "P1",
                "conformer_xyz": str(conf),
                "status": "ok",
            },
            {
                "target_id": "P2",
                "conformer_xyz": str(conf),
                "status": "ok",
            },
        ]
    ).to_csv(crest, sep="\t", index=False)
    cluster = tmp_path / "clusters.tsv"
    cluster.write_text("stale\n")

    # 이제 성공 경로를 타므로 xtb 대역이 필요하다. 표적마다 같은 파일을 주면
    # 계산은 한 번만 돌아야 하므로, 몇 번 불렸는지도 센다.
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    counter = tmp_path / "xtb_calls"
    fake_xtb = fake_bin / "xtb"
    fake_xtb.write_text(
        "#!/bin/sh\n"
        f"printf 'x' >> {counter}\n"
        # 실제 xTB 는 표 안에 넣어 출력한다. 파서가 split()[3] 을 읽으므로
        # 열 위치가 실제와 같아야 대역이 의미를 가진다.
        "printf '          | TOTAL ENERGY  -0.39259238 Eh   |\\n'\n"
    )
    fake_xtb.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = str(fake_bin)

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage8_xtb_cluster.py"),
            "--crest-manifest", str(crest),
            "--out-dir", str(tmp_path / "out"),
            "--out-cluster-manifest", str(cluster),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    # 자유 리간드 앙상블은 표적과 무관한 같은 분자의 것이다. 모든 행이 같은
    # 파일을 가리키는 것이 옳은 상태이고, 매니페스트의 `conformer_scope` 도
    # 그렇게 적고 있다. 예전에는 표적마다 CREST 를 돌려 경로가 달랐고 이 검사는
    # 그 시절의 것이었다 - 중복을 금지하면서 동시에 같은 것이기를 요구했다.
    assert res.returncode == 0, res.stderr
    out = pd.read_csv(cluster, sep="\t")
    assert out["target_id"].tolist() == ["P1", "P2"]
    assert len(set(out["cluster_xyz"])) == 1, "같은 앙상블을 가리켜야 합니다"
    assert len(set(out["xtb_energy_hartree"])) == 1, "같은 분자면 같은 에너지다"
    assert counter.read_text() == "x", (
        f"xTB 가 {len(counter.read_text())}번 불렸습니다. 같은 앙상블은 한 번이면 됩니다"
    )


def test_stage8_xtb_refuses_targets_pointing_at_different_ensembles(
    tmp_path: Path,
) -> None:
    conf = tmp_path / "conf.xyz"
    conf.write_text("1\nconf\nH 0 0 0\n")
    crest = tmp_path / "crest.tsv"
    pd.DataFrame(
        [
            {
                "target_id": "P1",
                "conformer_xyz": str(conf),
                "status": "ok",
            },
            {
                "target_id": "P2",
                # 같은 분자인데 다른 파일을 가리킨다 - 자유 리간드 앙상블이
                # 표적마다 다르게 나왔다는 뜻이므로 그것이 이상한 상태다.
                "conformer_xyz": str(conf.parent / "other.xyz"),
                "status": "ok",
            },
        ]
    ).to_csv(crest, sep="\t", index=False)
    cluster = tmp_path / "clusters.tsv"
    cluster.write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage8_xtb_cluster.py"),
            "--crest-manifest", str(crest),
            "--out-dir", str(tmp_path / "out"),
            "--out-cluster-manifest", str(cluster),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    # 자유 리간드 앙상블은 표적과 무관한 같은 분자의 것이다. 모든 행이 같은
    # 파일을 가리키는 것이 옳은 상태이고, 매니페스트의 `conformer_scope` 도
    # 그렇게 적고 있다. 예전에는 표적마다 CREST 를 한 번씩 돌려 경로가 달랐고
    # 이 검사는 그 시절의 것이었다 - 중복을 금지하면서 동시에 같은 것이기를
    # 요구하는 모순이었다. 아다팔렌 실측으로 한 번에 2시간 24분이라
    # `top_n_for_qm: 3` 이면 같은 답에 7시간을 쓴다.
    # 같은 분자인데 표적마다 다른 앙상블이 나왔다는 뜻이므로 그것이 이상하다.
    assert res.returncode != 0
    assert "different ensembles" in res.stderr
    assert not cluster.exists()


def test_stage8_xtb_rejects_blank_conformer_path_without_cluster(
    tmp_path: Path,
) -> None:
    crest = tmp_path / "crest.tsv"
    pd.DataFrame([{
        "target_id": "P1",
        "conformer_xyz": " ",
        "status": "ok",
    }]).to_csv(crest, sep="\t", index=False)
    cluster = tmp_path / "clusters.tsv"
    cluster.write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage8_xtb_cluster.py"),
            "--crest-manifest", str(crest),
            "--out-dir", str(tmp_path / "out"),
            "--out-cluster-manifest", str(cluster),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "CREST manifest column 'conformer_xyz' contains blank values" in res.stderr
    assert not cluster.exists()


def test_stage8_xtb_rejects_invalid_crest_status_without_cluster(
    tmp_path: Path,
) -> None:
    crest = tmp_path / "crest.tsv"
    pd.DataFrame([{
        "target_id": "P1",
        "conformer_xyz": str(tmp_path / "conf.xyz"),
        "status": "failed",
    }]).to_csv(crest, sep="\t", index=False)
    cluster = tmp_path / "clusters.tsv"
    cluster.write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage8_xtb_cluster.py"),
            "--crest-manifest", str(crest),
            "--out-dir", str(tmp_path / "out"),
            "--out-cluster-manifest", str(cluster),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "CREST manifest column 'status' contains invalid values" in res.stderr
    assert "failed" in res.stderr
    assert not cluster.exists()


def test_stage8_xtb_rejects_empty_conformer_file(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_xtb = fake_bin / "xtb"
    fake_xtb.write_text(
        "#!/bin/sh\n"
        "printf 'TOTAL ENERGY       -1.234567\\n'\n"
        "exit 0\n"
    )
    fake_xtb.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = str(fake_bin)

    xyz = tmp_path / "empty.xyz"
    xyz.write_text("")
    crest = tmp_path / "crest.tsv"
    pd.DataFrame([{
        "target_id": "P1",
        "conformer_xyz": str(xyz),
        "status": "ok",
    }]).to_csv(crest, sep="\t", index=False)
    cluster = tmp_path / "clusters.tsv"
    cluster.write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage8_xtb_cluster.py"),
            "--crest-manifest", str(crest),
            "--out-dir", str(tmp_path / "out"),
            "--out-cluster-manifest", str(cluster),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "Missing or empty CREST conformer file for P1" in res.stderr
    assert not cluster.exists()


def test_stage8_dft_rejects_cluster_manifest_missing_required_columns(tmp_path: Path) -> None:
    fake_pkg = tmp_path / "fake_pkg"
    pyscf = fake_pkg / "pyscf"
    pyscf.mkdir(parents=True)
    (pyscf / "__init__.py").write_text("")

    clusters = tmp_path / "clusters.tsv"
    pd.DataFrame([{"target_id": "P1", "cluster_xyz": str(tmp_path / "bad.xyz")}]).to_csv(
        clusters, sep="\t", index=False
    )
    report = tmp_path / "dft.tsv"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(fake_pkg)

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage8_dft.py"),
            "--cluster-manifest", str(clusters),
            "--out-dir", str(tmp_path / "out"),
            "--out-report", str(report),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "xTB cluster manifest missing required columns" in res.stderr
    assert not report.exists()


def test_stage8_dft_rejects_empty_cluster_manifest_without_report(
    tmp_path: Path,
) -> None:
    clusters = tmp_path / "clusters.tsv"
    clusters.write_text("")
    report = tmp_path / "dft.tsv"
    report.write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage8_dft.py"),
            "--cluster-manifest", str(clusters),
            "--out-dir", str(tmp_path / "out"),
            "--out-report", str(report),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "xTB cluster manifest is required and must be non-empty" in res.stderr
    assert not report.exists()


def test_stage8_dft_rejects_blank_cluster_target_id_without_report(
    tmp_path: Path,
) -> None:
    clusters = tmp_path / "clusters.tsv"
    pd.DataFrame([{
        "target_id": " ",
        "cluster_xyz": str(tmp_path / "cluster.xyz"),
        "xtb_energy_hartree": -1.0,
    }]).to_csv(clusters, sep="\t", index=False)
    report = tmp_path / "dft.tsv"
    report.write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage8_dft.py"),
            "--cluster-manifest", str(clusters),
            "--out-dir", str(tmp_path / "out"),
            "--out-report", str(report),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "xTB cluster manifest column 'target_id' contains blank values" in res.stderr
    assert not report.exists()


def test_stage8_dft_rejects_duplicate_cluster_target_id_without_report(
    tmp_path: Path,
) -> None:
    clusters = tmp_path / "clusters.tsv"
    pd.DataFrame(
        [
            {
                "target_id": "P1",
                "cluster_xyz": str(tmp_path / "cluster1.xyz"),
                "xtb_energy_hartree": -1.0,
            },
            {
                "target_id": "P1",
                "cluster_xyz": str(tmp_path / "cluster2.xyz"),
                "xtb_energy_hartree": -2.0,
            },
        ]
    ).to_csv(clusters, sep="\t", index=False)
    report = tmp_path / "dft.tsv"
    report.write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage8_dft.py"),
            "--cluster-manifest", str(clusters),
            "--out-dir", str(tmp_path / "out"),
            "--out-report", str(report),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert (
        "xTB cluster manifest contains duplicate target_id values: P1"
        in res.stderr
    )
    assert not report.exists()


def test_stage8_dft_rejects_duplicate_cluster_xyz_without_report(
    tmp_path: Path,
) -> None:
    cluster_xyz = tmp_path / "cluster.xyz"
    clusters = tmp_path / "clusters.tsv"
    pd.DataFrame(
        [
            {
                "target_id": "P1",
                "cluster_xyz": str(cluster_xyz),
                "xtb_energy_hartree": -1.0,
            },
            {
                "target_id": "P2",
                "cluster_xyz": str(cluster_xyz),
                "xtb_energy_hartree": -2.0,
            },
        ]
    ).to_csv(clusters, sep="\t", index=False)
    report = tmp_path / "dft.tsv"
    report.write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage8_dft.py"),
            "--cluster-manifest", str(clusters),
            "--out-dir", str(tmp_path / "out"),
            "--out-report", str(report),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "xTB cluster manifest contains duplicate cluster_xyz values" in res.stderr
    assert str(cluster_xyz) in res.stderr
    assert not report.exists()


def test_stage8_dft_rejects_canonical_duplicate_cluster_xyz_without_report(
    tmp_path: Path,
) -> None:
    cluster_xyz = tmp_path / "cluster.xyz"
    cluster_xyz.write_text("1\ncluster\nH 0 0 0\n")
    clusters = tmp_path / "clusters.tsv"
    pd.DataFrame(
        [
            {
                "target_id": "P1",
                "cluster_xyz": str(cluster_xyz),
                "xtb_energy_hartree": -1.0,
            },
            {
                "target_id": "P2",
                "cluster_xyz": f"{cluster_xyz.parent}/./{cluster_xyz.name}",
                "xtb_energy_hartree": -2.0,
            },
        ]
    ).to_csv(clusters, sep="\t", index=False)
    report = tmp_path / "dft.tsv"
    report.write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage8_dft.py"),
            "--cluster-manifest", str(clusters),
            "--out-dir", str(tmp_path / "out"),
            "--out-report", str(report),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "xTB cluster manifest contains duplicate cluster_xyz values" in res.stderr
    assert str(cluster_xyz.resolve()) in res.stderr
    assert not report.exists()


def test_stage8_dft_rejects_blank_cluster_xyz_without_report(
    tmp_path: Path,
) -> None:
    clusters = tmp_path / "clusters.tsv"
    pd.DataFrame([{
        "target_id": "P1",
        "cluster_xyz": " ",
        "xtb_energy_hartree": -1.0,
    }]).to_csv(clusters, sep="\t", index=False)
    report = tmp_path / "dft.tsv"
    report.write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage8_dft.py"),
            "--cluster-manifest", str(clusters),
            "--out-dir", str(tmp_path / "out"),
            "--out-report", str(report),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "xTB cluster manifest column 'cluster_xyz' contains blank values" in res.stderr
    assert not report.exists()


def test_stage8_dft_rejects_nonnumeric_xtb_energy_without_report(
    tmp_path: Path,
) -> None:
    clusters = tmp_path / "clusters.tsv"
    pd.DataFrame([{
        "target_id": "P1",
        "cluster_xyz": str(tmp_path / "cluster.xyz"),
        "xtb_energy_hartree": "bad",
    }]).to_csv(clusters, sep="\t", index=False)
    report = tmp_path / "dft.tsv"
    report.write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage8_dft.py"),
            "--cluster-manifest", str(clusters),
            "--out-dir", str(tmp_path / "out"),
            "--out-report", str(report),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert (
        "xTB cluster manifest column 'xtb_energy_hartree' must be numeric"
        in res.stderr
    )
    assert not report.exists()


def test_stage8_dft_rejects_boolean_xtb_energy_without_report(
    tmp_path: Path,
) -> None:
    clusters = tmp_path / "clusters.tsv"
    clusters.write_text(
        "target_id\tcluster_xyz\txtb_energy_hartree\n"
        f"P1\t{tmp_path / 'cluster.xyz'}\tTrue\n"
    )
    report = tmp_path / "dft.tsv"
    report.write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage8_dft.py"),
            "--cluster-manifest", str(clusters),
            "--out-dir", str(tmp_path / "out"),
            "--out-report", str(report),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert (
        "xTB cluster manifest column 'xtb_energy_hartree' must be numeric"
        in res.stderr
    )
    assert not report.exists()


def test_stage8_dft_rejects_nonfinite_xtb_energy_without_report(
    tmp_path: Path,
) -> None:
    clusters = tmp_path / "clusters.tsv"
    clusters.write_text(
        "target_id\tcluster_xyz\txtb_energy_hartree\n"
        f"P1\t{tmp_path / 'cluster.xyz'}\tinf\n"
    )
    report = tmp_path / "dft.tsv"
    report.write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage8_dft.py"),
            "--cluster-manifest", str(clusters),
            "--out-dir", str(tmp_path / "out"),
            "--out-report", str(report),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert (
        "xTB cluster manifest column 'xtb_energy_hartree' must be finite"
        in res.stderr
    )
    assert not report.exists()


def test_stage8_dft_rejects_empty_cluster_xyz(tmp_path: Path) -> None:
    fake_pkg = tmp_path / "fake_pkg"
    pyscf = fake_pkg / "pyscf"
    pyscf.mkdir(parents=True)
    (pyscf / "__init__.py").write_text("")

    xyz = tmp_path / "empty.xyz"
    xyz.write_text("")
    clusters = tmp_path / "clusters.tsv"
    pd.DataFrame([{
        "target_id": "P1",
        "cluster_xyz": str(xyz),
        "xtb_energy_hartree": -1.0,
    }]).to_csv(clusters, sep="\t", index=False)
    report = tmp_path / "dft.tsv"
    report.write_text("stale\n")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(fake_pkg)

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage8_dft.py"),
            "--cluster-manifest", str(clusters),
            "--out-dir", str(tmp_path / "out"),
            "--out-report", str(report),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "Missing or empty cluster XYZ for P1" in res.stderr
    assert not report.exists()


def test_stage8_failed_dft_does_not_leave_report(tmp_path: Path) -> None:
    fake_pkg = tmp_path / "fake_pkg"
    pyscf = fake_pkg / "pyscf"
    pyscf.mkdir(parents=True)
    (pyscf / "__init__.py").write_text("")

    xyz = tmp_path / "bad.xyz"
    xyz.write_text("not an xyz\n")
    clusters = tmp_path / "clusters.tsv"
    pd.DataFrame([{
        "target_id": "P1",
        "cluster_xyz": str(xyz),
        "xtb_energy_hartree": -1.0,
        "charge": 0,
        "spin": 0,
    }]).to_csv(
        clusters, sep="\t", index=False
    )
    report = tmp_path / "dft.tsv"
    report.write_text("stale\n")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(fake_pkg)

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage8_dft.py"),
            "--cluster-manifest", str(clusters),
            "--out-dir", str(tmp_path / "out"),
            "--out-report", str(report),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "DFT produced no converged energies" in res.stderr
    assert not report.exists()


def test_stage8_nonfinite_dft_energy_does_not_leave_report(tmp_path: Path) -> None:
    fake_pkg = tmp_path / "fake_pkg"
    pyscf = fake_pkg / "pyscf"
    pyscf.mkdir(parents=True)
    (pyscf / "__init__.py").write_text("")
    (pyscf / "gto.py").write_text("def M(**kwargs):\n    return kwargs\n")
    (pyscf / "dft.py").write_text(
        "class RKS:\n"
        "    def __init__(self, mol):\n"
        "        self.mol = mol\n"
        "        self.converged = True\n"
        "        self.xc = ''\n"
        "        self.max_cycle = 0\n"
        "    def kernel(self):\n"
        "        return float('inf')\n"
    )

    xyz = tmp_path / "cluster.xyz"
    xyz.write_text("1\nok\nH 0 0 0\n")
    clusters = tmp_path / "clusters.tsv"
    pd.DataFrame([{
        "target_id": "P1",
        "cluster_xyz": str(xyz),
        "xtb_energy_hartree": -1.0,
        "charge": 0,
        "spin": 0,
    }]).to_csv(clusters, sep="\t", index=False)
    report = tmp_path / "dft.tsv"
    report.write_text("stale\n")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(fake_pkg)

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage8_dft.py"),
            "--cluster-manifest", str(clusters),
            "--out-dir", str(tmp_path / "out"),
            "--out-report", str(report),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "DFT produced no converged energies" in res.stderr
    assert not report.exists()


def test_stage8_nonnegative_dft_energy_does_not_leave_report(tmp_path: Path) -> None:
    fake_pkg = tmp_path / "fake_pkg"
    pyscf = fake_pkg / "pyscf"
    pyscf.mkdir(parents=True)
    (pyscf / "__init__.py").write_text("")
    (pyscf / "gto.py").write_text("def M(**kwargs):\n    return kwargs\n")
    (pyscf / "dft.py").write_text(
        "class RKS:\n"
        "    def __init__(self, mol):\n"
        "        self.mol = mol\n"
        "        self.converged = True\n"
        "        self.xc = ''\n"
        "        self.max_cycle = 0\n"
        "    def kernel(self):\n"
        "        return 0.0\n"
    )

    xyz = tmp_path / "cluster.xyz"
    xyz.write_text("1\nok\nH 0 0 0\n")
    clusters = tmp_path / "clusters.tsv"
    pd.DataFrame([{
        "target_id": "P1",
        "cluster_xyz": str(xyz),
        "xtb_energy_hartree": -1.0,
        "charge": 0,
        "spin": 0,
    }]).to_csv(clusters, sep="\t", index=False)
    report = tmp_path / "dft.tsv"
    report.write_text("stale\n")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(fake_pkg)

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage8_dft.py"),
            "--cluster-manifest", str(clusters),
            "--out-dir", str(tmp_path / "out"),
            "--out-report", str(report),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "DFT produced no converged energies" in res.stderr
    assert not report.exists()


def test_stage8_partial_dft_failure_does_not_leave_report(tmp_path: Path) -> None:
    fake_pkg = tmp_path / "fake_pkg"
    pyscf = fake_pkg / "pyscf"
    pyscf.mkdir(parents=True)
    (pyscf / "__init__.py").write_text("")
    (pyscf / "gto.py").write_text("def M(**kwargs):\n    return kwargs\n")
    (pyscf / "dft.py").write_text(
        "class RKS:\n"
        "    def __init__(self, mol):\n"
        "        self.mol = mol\n"
        "        self.converged = True\n"
        "        self.xc = ''\n"
        "        self.max_cycle = 0\n"
        "    def kernel(self):\n"
        "        return -123.456789\n"
    )

    good_xyz = tmp_path / "good.xyz"
    good_xyz.write_text("1\nok\nH 0 0 0\n")
    bad_xyz = tmp_path / "bad.xyz"
    bad_xyz.write_text("not an xyz\n")
    clusters = tmp_path / "clusters.tsv"
    pd.DataFrame(
        [
                {
                    "target_id": "P1",
                    "cluster_xyz": str(good_xyz),
                    "xtb_energy_hartree": -2.0,
                    "charge": 0,
                    "spin": 0,
                },
                {
                    "target_id": "P2",
                    "cluster_xyz": str(bad_xyz),
                    "xtb_energy_hartree": -1.0,
                    "charge": 0,
                    "spin": 0,
                },
        ]
    ).to_csv(clusters, sep="\t", index=False)
    report = tmp_path / "dft.tsv"
    report.write_text("stale\n")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(fake_pkg)

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage8_dft.py"),
            "--cluster-manifest", str(clusters),
            "--out-dir", str(tmp_path / "out"),
            "--out-report", str(report),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "DFT failed for final candidates P2" in res.stderr
    assert not report.exists()


def test_stage8_dft_partial_report_requires_explicit_flag(tmp_path: Path) -> None:
    fake_pkg = tmp_path / "fake_pkg"
    pyscf = fake_pkg / "pyscf"
    pyscf.mkdir(parents=True)
    (pyscf / "__init__.py").write_text("")
    (pyscf / "gto.py").write_text("def M(**kwargs):\n    return kwargs\n")
    (pyscf / "dft.py").write_text(
        "class RKS:\n"
        "    def __init__(self, mol):\n"
        "        self.mol = mol\n"
        "        self.converged = True\n"
        "        self.xc = ''\n"
        "        self.max_cycle = 0\n"
        "    def kernel(self):\n"
        "        return -123.456789\n"
    )

    good_xyz = tmp_path / "good.xyz"
    good_xyz.write_text("1\nok\nH 0 0 0\n")
    bad_xyz = tmp_path / "bad.xyz"
    bad_xyz.write_text("not an xyz\n")
    clusters = tmp_path / "clusters.tsv"
    pd.DataFrame(
        [
                {
                    "target_id": "P1",
                    "cluster_xyz": str(good_xyz),
                    "xtb_energy_hartree": -2.0,
                    "charge": 0,
                    "spin": 0,
                },
                {
                    "target_id": "P2",
                    "cluster_xyz": str(bad_xyz),
                    "xtb_energy_hartree": -1.0,
                    "charge": 0,
                    "spin": 0,
                },
        ]
    ).to_csv(clusters, sep="\t", index=False)
    report = tmp_path / "dft.tsv"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(fake_pkg)

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage8_dft.py"),
            "--cluster-manifest", str(clusters),
            "--out-dir", str(tmp_path / "out"),
            "--out-report", str(report),
            "--allow-partial-output",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode == 0, res.stderr
    out = pd.read_csv(report, sep="\t")
    assert out["target_id"].tolist() == ["P1"]
    assert out["status"].tolist() == ["ok"]
    assert out["charge"].tolist() == [0]
    assert out["spin"].tolist() == [0]
