from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import stage4_prepare_structures as stage4  # noqa: E402
import stage5_5_plip as plip  # noqa: E402
import stage5_5_prolif as prolif  # noqa: E402
import stage6_bioemu as stage6  # noqa: E402
import stage6_ensemble_dock as ensemble  # noqa: E402
import stage7_gromacs_prep as stage7_prep  # noqa: E402
import stage7_mmgbsa as stage7_mmgbsa  # noqa: E402
import stage8_crest as crest  # noqa: E402
import stage8_dft as dft  # noqa: E402
import stage8_xtb_cluster as xtb  # noqa: E402


def test_crest_command_uses_declared_charge_and_spin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ligand = tmp_path / "ligand.xyz"
    ligand.write_text("1\nligand\nH 0 0 0\n")
    captured: list[str] = []

    def fake_run(command, **kwargs):
        captured.extend(command)
        (tmp_path / "crest_conformers.xyz").write_text("1\nconf\nH 0 0 0\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(crest.shutil, "which", lambda _: "/bin/tool")
    monkeypatch.setattr(crest.subprocess, "run", fake_run)
    assert crest.run_crest(ligand, tmp_path, charge=-1, spin=2)
    assert captured[-4:] == ["--chrg", "-1", "--uhf", "2"]


def test_xtb_scores_every_crest_frame_and_selects_lowest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ensemble_xyz = tmp_path / "crest.xyz"
    ensemble_xyz.write_text(
        "1\nfirst\nH 0 0 0\n"
        "1\nsecond\nH 1 0 0\n"
        "1\nthird\nH 2 0 0\n"
    )
    energies = iter([-1.0, -3.0, -2.0])
    called: list[Path] = []

    def fake_singlepoint(path, *_args, **_kwargs):
        called.append(path)
        return next(energies)

    monkeypatch.setattr(xtb, "xtb_singlepoint", fake_singlepoint)
    selected, energy = xtb.select_lowest_energy_conformer(
        ensemble_xyz, tmp_path / "out", "gfn2", charge=0, spin=0
    )
    assert len(called) == 3
    assert selected.name == "frame_000001.xyz"
    assert energy == -3.0


@pytest.mark.parametrize(
    "body",
    [
        "2\nbad\nH 0 0 0\nBROKEN\n",
        "2\ntruncated\nH 0 0 0\n",
        "1\nnonfinite\nH nan 0 0\n",
        "1\nextra\nH 0 0 0\nH 1 0 0\n",
        "0\nempty\n",
    ],
)
def test_dft_xyz_parser_rejects_malformed_documents(tmp_path: Path, body: str) -> None:
    path = tmp_path / "bad.xyz"
    path.write_text(body)
    assert dft.parse_xyz(path) == ([], [])


def test_dft_accepts_shared_free_ligand_representative(tmp_path: Path) -> None:
    xyz = tmp_path / "selected.xyz"
    xyz.write_text("1\nselected\nH 0 0 0\n")
    manifest = tmp_path / "xtb.tsv"
    pd.DataFrame([
        {"target_id": "P1", "cluster_xyz": xyz, "xtb_energy_hartree": -1.0},
        {"target_id": "P2", "cluster_xyz": xyz, "xtb_energy_hartree": -1.0},
    ]).to_csv(manifest, sep="\t", index=False)
    assert dft.read_cluster_manifest(manifest)["target_id"].tolist() == ["P1", "P2"]


def test_bioemu_coordinates_are_invariant_to_rigid_motion() -> None:
    reference = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    angle = math.pi / 3
    rotation = np.array([
        [math.cos(angle), -math.sin(angle), 0.0],
        [math.sin(angle), math.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    moved = reference @ rotation + np.array([17.0, -9.0, 4.0])
    aligned = stage6.rigid_invariant_coordinates(np.stack([reference, moved]))
    assert np.allclose(aligned[0], aligned[1], atol=1e-10)


def test_ensemble_manifest_rejects_residual_sidechain_strain(tmp_path: Path) -> None:
    medoid = tmp_path / "medoid.pdb"
    medoid.write_text("ATOM\n")
    manifest = tmp_path / "manifest.tsv"
    pd.DataFrame([{
        "target_id": "P1",
        "medoid_pdb": str(medoid),
        "reconstruction_status": "pdbfixer_sidechains_openmm_relaxed_with_strain",
    }]).to_csv(manifest, sep="\t", index=False)
    with pytest.raises(SystemExit, match="ineligible side-chain"):
        ensemble.read_bioemu_manifest(manifest)


def test_stage7_accepts_boltz_complex_pdb_as_pose_manifest(tmp_path: Path) -> None:
    complex_pdb = tmp_path / "complex.pdb"
    complex_pdb.write_text("ATOM\nHETATM\n")
    manifest = tmp_path / "boltz.tsv"
    pd.DataFrame([{"target_id": "P1", "complex_pdb": str(complex_pdb)}]).to_csv(
        manifest, sep="\t", index=False
    )
    assert stage7_prep.read_complex_pose_manifest(manifest) == {"P1": complex_pdb}


def test_stage6_accepts_alphafold_fallback_provenance(tmp_path: Path) -> None:
    report = pd.DataFrame([{
        "target_id": "P1",
        "kept": "yes",
        "source": "alphafold_cleaned_holo_download_failed",
        "n_residues": 100,
        "iptm": 0.8,
        "complex_plddt": 90.0,
        "affinity_log_uM": 1.0,
        "pb_valid": "yes",
    }])
    path = tmp_path / "boltz.tsv"
    report.to_csv(path, sep="\t", index=False)
    assert stage6.read_boltz_report(path).iloc[0]["source"].endswith("download_failed")


@pytest.mark.parametrize("reader", [plip.kept_complexes, prolif.kept_complexes])
def test_stage55_uses_only_quality_approved_complexes(tmp_path: Path, reader) -> None:
    kept = tmp_path / "kept.pdb"
    rejected = tmp_path / "rejected.pdb"
    kept.write_text("KEPT\n")
    rejected.write_text("REJECTED\n")
    report = tmp_path / "boltz.tsv"
    pd.DataFrame([
        {"target_id": "P1", "complex_pdb": kept, "kept": "yes"},
        {"target_id": "P2", "complex_pdb": rejected, "kept": "no"},
    ]).to_csv(report, sep="\t", index=False)
    assert reader(report) == [("P1", kept)]


def test_mmgbsa_removes_stale_result_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trajectory = tmp_path / "prod_r1.xtc"
    trajectory.write_text("trajectory")
    (tmp_path / "prod.tpr").write_text("structure")
    (tmp_path / "topol.top").write_text("topology")
    stale = tmp_path / "FINAL_RESULTS_MMPBSA.dat"
    stale.write_text("DELTA TOTAL -99\n")
    monkeypatch.setattr(stage7_mmgbsa.shutil, "which", lambda _: "/bin/tool")
    monkeypatch.setattr(stage7_mmgbsa, "trajectory_frame_count", lambda *_: 10)
    monkeypatch.setattr(stage7_mmgbsa, "write_mmpbsa_input", lambda *_: tmp_path / "in")
    monkeypatch.setattr(stage7_mmgbsa, "write_index_file", lambda *_: tmp_path / "index")
    monkeypatch.setattr(stage7_mmgbsa, "index_group_numbers", lambda *_: (1, 2))
    # MM-GBSA 는 궤적을 넘기기 전에 trjconv 로 주기 경계를 제거한다. 그 단계도
    # subprocess 를 쓰므로, 대체하지 않으면 아래 fake_run 이 gmx_MMPBSA 대신
    # trjconv 호출을 먼저 받아 이 테스트가 재려던 것을 재지 못한다.
    monkeypatch.setattr(stage7_mmgbsa, "strip_pbc",
                        lambda trajectory, *_a, **_k: trajectory)

    def fake_run(*_args, **_kwargs):
        assert not stale.exists()
        return SimpleNamespace(returncode=1, stdout="failed", stderr="failed")

    monkeypatch.setattr(stage7_mmgbsa.subprocess, "run", fake_run)
    assert stage7_mmgbsa.run_mmgbsa(tmp_path, [trajectory]) is None


def test_stage4_global_alignment_handles_one_residue_offset() -> None:
    assert stage4._sequence_identity("MABCDEFG", "ABCDEFG") == pytest.approx(1.0)


def test_workflow_wires_quality_report_and_default_stage7_pose() -> None:
    pharmacophore_rule = (ROOT / "workflow/rules/stage5_5_pharmacophore.smk").read_text()
    stage7_rule = (ROOT / "workflow/rules/stage7_md.smk").read_text()
    plip_rule = pharmacophore_rule.split("rule plip_run:", 1)[1].split(
        "rule prolif_fingerprint:", 1
    )[0]
    prolif_rule = pharmacophore_rule.split("rule prolif_fingerprint:", 1)[1].split(
        "rule pharmacophore_consensus:", 1
    )[0]
    assert "--boltz-report {input.boltz:q}" in plip_rule
    assert "--boltz-report {input.boltz:q}" in prolif_rule
    assert "boltz_poses = rules.boltz2_cofold.output.report" in stage7_rule
    assert 'or input.boltz_poses' in stage7_rule


def test_mmgbsa_parses_the_uncertainty_next_to_the_value() -> None:
    """평균만 꺼내 쓰면 그 값이 얼마나 흔들리는 값인지가 사라진다.

    gmx_MMPBSA 의 열 순서는 `Average SD(Prop.) SD SEM(Prop.) SEM` 이다.
    """
    line = "ΔTOTAL                  -19.64          1.00       1.65         0.32       0.52"
    assert stage7_mmgbsa.parse_delta_total(line) == (-19.64, 0.52)


def test_mmgbsa_report_carries_replica_counts_and_uncertainty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """복제 하나가 빠져도 평균은 평범한 숫자로 나온다. 읽는 쪽이 그것을 알아야 한다."""
    import subprocess as sp

    target_dir = tmp_path / "P1"
    target_dir.mkdir()
    trajectory = target_dir / "prod_r1.xtc"
    trajectory.write_text("trajectory")
    index = tmp_path / "traj.tsv"
    pd.DataFrame([{
        "target_id": "P1", "replica": 1, "trajectory_xtc": str(trajectory),
        "status": "ok",
    }]).to_csv(index, sep="\t", index=False)
    report = tmp_path / "mmgbsa.tsv"

    monkeypatch.setattr(stage7_mmgbsa.shutil, "which", lambda _: "/bin/tool")
    monkeypatch.setattr(stage7_mmgbsa, "run_mmgbsa", lambda *_a, **_k: {
        "dg_kcal_mol": -12.8904, "uncertainty_kcal_mol": 0.82,
        "spread_kcal_mol": 1.64, "n_replicas_ok": 2,
        "n_replicas_requested": 2, "n_frames": 10, "per_replica": [-12.07, -13.71],
    })
    argv = ["stage7_mmgbsa.py", "--trajectory-index", str(index),
            "--out-report", str(report)]
    monkeypatch.setattr(sys, "argv", argv)
    stage7_mmgbsa.main()

    out = pd.read_csv(report, sep="\t")
    row = out.iloc[0]
    # 세 자리는 이 표집이 뒷받침하지 못한다. 프레임 간 SD 가 1.9~2.8 kcal/mol 이다.
    assert str(row["mmgbsa_dg_kcal_mol"]) == "-12.9"
    # 불확실성도 같은 자릿수로 적는다. 값보다 정밀한 오차는 뜻이 없다.
    assert row["mmgbsa_uncertainty_kcal_mol"] == pytest.approx(0.8)
    assert row["n_replicas_ok"] == 2 and row["n_replicas_requested"] == 2
    assert row["status"] == "ok"
    del sp


def test_a_dropped_replica_does_not_pass_as_a_complete_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """복제 하나가 PBC 처리에서 빠졌는데 status 가 ok 면 온전한 값과 구분되지 않는다."""
    target_dir = tmp_path / "P1"
    target_dir.mkdir()
    trajectory = target_dir / "prod_r1.xtc"
    trajectory.write_text("trajectory")
    index = tmp_path / "traj.tsv"
    pd.DataFrame([{
        "target_id": "P1", "replica": 1, "trajectory_xtc": str(trajectory),
        "status": "ok",
    }]).to_csv(index, sep="\t", index=False)
    report = tmp_path / "mmgbsa.tsv"
    monkeypatch.setattr(stage7_mmgbsa.shutil, "which", lambda _: "/bin/tool")
    monkeypatch.setattr(stage7_mmgbsa, "run_mmgbsa", lambda *_a, **_k: {
        "dg_kcal_mol": -19.64, "uncertainty_kcal_mol": 0.52,
        "spread_kcal_mol": None, "n_replicas_ok": 1,
        "n_replicas_requested": 2, "n_frames": 10, "per_replica": [-19.64],
    })
    monkeypatch.setattr(sys, "argv", [
        "stage7_mmgbsa.py", "--trajectory-index", str(index),
        "--out-report", str(report)])
    with pytest.raises(SystemExit):
        stage7_mmgbsa.main()

    # 명시적으로 열어 주면 쓰되, 온전한 행과 구분되는 status 를 단다.
    monkeypatch.setattr(sys, "argv", [
        "stage7_mmgbsa.py", "--trajectory-index", str(index),
        "--out-report", str(report), "--allow-partial-output"])
    stage7_mmgbsa.main()
    out = pd.read_csv(report, sep="\t")
    assert out.iloc[0]["status"] == "ok_partial"
    assert out.iloc[0]["n_replicas_ok"] == 1
    assert out.iloc[0]["n_replicas_requested"] == 2


def test_a_single_replica_reports_no_uncertainty_rather_than_a_tiny_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """프레임 SEM 을 불확실성으로 내보내면 20배 낙관적인 숫자가 나간다.

    실측(5 ns, 402 프레임): 프레임 SEM 0.13~0.15 인데 복제 간 폭은 2.30~3.22 다.
    10 ps 간격 프레임이 서로 상관돼 있어 SEM 의 N 이 허수이기 때문이다.
    """
    target_dir = tmp_path / "P1"
    target_dir.mkdir()
    trajectory = target_dir / "prod_r1.xtc"
    trajectory.write_text("trajectory")
    (target_dir / "prod.tpr").write_text("structure")
    (target_dir / "topol.top").write_text("topology")

    monkeypatch.setattr(stage7_mmgbsa.shutil, "which", lambda _: "/bin/tool")
    monkeypatch.setattr(stage7_mmgbsa, "trajectory_frame_count", lambda *_: 501)
    monkeypatch.setattr(stage7_mmgbsa, "write_mmpbsa_input", lambda *_: tmp_path / "in")
    monkeypatch.setattr(stage7_mmgbsa, "write_index_file", lambda *_: tmp_path / "ndx")
    monkeypatch.setattr(stage7_mmgbsa, "index_group_numbers", lambda *_: (1, 2))
    monkeypatch.setattr(stage7_mmgbsa, "_run_one_trajectory",
                        lambda *_a, **_k: (-14.44, 0.13))

    out = stage7_mmgbsa.run_mmgbsa(target_dir, [trajectory])
    assert out is not None
    assert out["dg_kcal_mol"] == pytest.approx(-14.44)
    assert out["uncertainty_kcal_mol"] is None, (
        "복제가 하나뿐인데 프레임 SEM 을 불확실성으로 내보냈습니다"
    )


def test_two_replicas_report_half_the_spread_as_the_uncertainty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """복제가 둘이면 그 폭이 유일하게 실제로 잰 불확실성이다."""
    target_dir = tmp_path / "P1"
    target_dir.mkdir()
    trajectories = []
    for replica in (1, 2):
        path = target_dir / f"prod_r{replica}.xtc"
        path.write_text("trajectory")
        trajectories.append(path)
    (target_dir / "prod.tpr").write_text("structure")
    (target_dir / "topol.top").write_text("topology")

    values = iter([(-14.44, 0.13), (-16.74, 0.15)])
    monkeypatch.setattr(stage7_mmgbsa.shutil, "which", lambda _: "/bin/tool")
    monkeypatch.setattr(stage7_mmgbsa, "trajectory_frame_count", lambda *_: 501)
    monkeypatch.setattr(stage7_mmgbsa, "write_mmpbsa_input", lambda *_: tmp_path / "in")
    monkeypatch.setattr(stage7_mmgbsa, "write_index_file", lambda *_: tmp_path / "ndx")
    monkeypatch.setattr(stage7_mmgbsa, "index_group_numbers", lambda *_: (1, 2))
    monkeypatch.setattr(stage7_mmgbsa, "_run_one_trajectory",
                        lambda *_a, **_k: next(values))

    out = stage7_mmgbsa.run_mmgbsa(target_dir, trajectories)
    assert out["dg_kcal_mol"] == pytest.approx(-15.59)
    assert out["spread_kcal_mol"] == pytest.approx(2.30)
    # 프레임 SEM(0.15)이 아니라 폭의 절반(1.15)이다.
    assert out["uncertainty_kcal_mol"] == pytest.approx(1.15)
