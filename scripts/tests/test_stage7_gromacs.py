from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _write_methane_sdf(path: Path) -> None:
    """3D 좌표를 가진 최소 분자. 중원자 1개라 대역 토폴로지와 짝이 맞는다."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles("C"))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()


def write_fake_stage7_tools(fake_bin: Path, log: Path) -> None:
    fake_bin.mkdir()
    # D04 검증이 설치본 GROMACS의 share/top에서 force field 존재를 확인한다.
    # 대역 gmx도 같은 배치를 갖추도록 top 디렉터리를 만든다.
    (fake_bin.parent / "share" / "gromacs" / "top" / "amber99sb-ildn.ff").mkdir(
        parents=True, exist_ok=True
    )
    fake_acpype = fake_bin / "acpype"
    fake_acpype.write_text(
        "#!/bin/sh\n"
        f"printf 'acpype %s\\n' \"$*\" >> '{log}'\n"
        "mkdir -p ligand.acpype\n"
        # 실제 ACPYPE 는 수소까지 포함한 전체 원자를 선언한다. 중원자만 적는
        # 대역은 실제보다 좁아서, 포즈와 토폴로지의 원자 수가 맞는지 확인하는
        # 코드가 있는지 없는지를 구분하지 못한다. 메탄(C + H 4개)에 맞춘다.
        "printf '[ moleculetype ]\\nLIG 3\\n[ atoms ]\\n"
        "1 c3 1 LIG C1 1 -0.1 12.01\\n"
        "2 hc 1 LIG H1 2 0.025 1.008\\n"
        "3 hc 1 LIG H2 3 0.025 1.008\\n"
        "4 hc 1 LIG H3 4 0.025 1.008\\n"
        "5 hc 1 LIG H4 5 0.025 1.008\\n' > ligand.acpype/ligand_GMX.itp\n"
        "exit 0\n"
    )
    fake_acpype.chmod(0o755)
    fake_gmx = fake_bin / "gmx"
    fake_gmx.write_text(
        "#!/bin/sh\n"
        f"printf 'gmx %s\\n' \"$*\" >> '{log}'\n"
        "sub=\"$1\"\n"
        "shift\n"
        "out=''\n"
        "top=''\n"
        "deffnm=''\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = '-o' ]; then shift; out=\"$1\"; fi\n"
        "  if [ \"$1\" = '-p' ]; then shift; top=\"$1\"; fi\n"
        "  if [ \"$1\" = '-deffnm' ]; then shift; deffnm=\"$1\"; fi\n"
        "  shift\n"
        "done\n"
        "case \"$sub\" in\n"
        "  pdb2gmx)\n"
        "    printf 'processed protein\\n1\\n    1PROT    N    1   0.000   0.000   0.000\\n   0.00000   0.00000   0.00000\\n' > \"$out\"\n"
        # 실제 pdb2gmx 는 힘장 include 와 단백질 [ moleculetype ] 을 반드시 쓴다.
        # 그것이 없는 대역은 실제보다 좁아서, 리간드 [ atomtypes ] 를 첫
        # [ moleculetype ] 앞에 넣어야 한다는 GROMACS 의 요구를 표현하지 못한다.
        "    printf '#include \"amber99sb-ildn.ff/forcefield.itp\"\\n\\n"
        "[ moleculetype ]\\nProtein 3\\n\\n[ atoms ]\\n   1 N 1 ALA N 1 -0.4 14.01\\n\\n"
        "[ system ]\\nProtein ligand\\n[ molecules ]\\nProtein 1\\n' > \"$top\"\n"
        "    ;;\n"
        "  editconf|solvate|genion)\n"
        "    printf '%s output\\n' \"$sub\" > \"$out\"\n"
        "    ;;\n"
        "  grompp)\n"
        "    printf 'tpr for %s\\n' \"$out\" > \"$out\"\n"
        "    ;;\n"
        "  mdrun)\n"
        "    case \"$deffnm\" in\n"
        "      *prod_r*) printf 'trajectory for %s\\n' \"$deffnm\" > \"${deffnm}.xtc\" ;;\n"
        "      *) printf 'gro for %s\\n' \"$deffnm\" > \"${deffnm}.gro\" ;;\n"
        "    esac\n"
        "    ;;\n"
        "esac\n"
    )
    fake_gmx.chmod(0o755)


def env_with_fake_tools(fake_bin: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = str(fake_bin) + ":" + os.environ.get("PATH", "")
    return env


def test_stage7_prep_builds_ready_manifest_with_fake_gmx_and_acpype(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    log = tmp_path / "commands.log"
    write_fake_stage7_tools(fake_bin, log)

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
    # 실제 SDF 여야 한다. 포즈를 토폴로지에 맞춰 다시 쓸 때 이 분자를 참조로
    # 삼아 부분구조 매칭을 하기 때문이다. `"ligand\n"` 같은 대역은 실제보다
    # 좁아서, 원자 대응을 확인하는 코드가 있는지 없는지를 구분하지 못한다.
    # 대역 acpype 가 원자 1개(C1)짜리 토폴로지를 내므로 메탄으로 맞춘다.
    ligand = tmp_path / "ligand.sdf"
    _write_methane_sdf(ligand)
    complex_pose = tmp_path / "complex.pdb"
    complex_pose.write_text(
        "ATOM      1  N   PRO A   1       0.000   0.000   0.000  1.00 20.00           N\n"
        "HETATM    2  C1  LIG B   2       1.000   1.000   1.000  1.00 20.00           C\n"
        "END\n"
    )
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
            "--complex-pose",
            str(complex_pose),
            "--out-dir",
            str(tmp_path / "out"),
            "--out-manifest",
            str(manifest),
        ],
        capture_output=True,
        text=True,
        env=env_with_fake_tools(fake_bin),
        check=False,
    )

    assert res.returncode == 0, res.stderr
    rows = pd.read_csv(manifest, sep="\t")
    assert rows["status"].tolist() == ["ready"]
    assert Path(rows.iloc[0]["system_gro"]).name == "system.gro"
    assert Path(rows.iloc[0]["topology_top"]).name == "topol.top"
    assert Path(rows.iloc[0]["tpr"]).name == "prod.tpr"
    assert rows.iloc[0]["system_gro_sha256"]
    assert rows.iloc[0]["topology_top_sha256"]
    assert rows.iloc[0]["tpr_sha256"]
    topol = Path(rows.iloc[0]["topology_top"]).read_text()
    # 리간드 토폴로지는 두 조각으로 들어간다. GROMACS 는 모든 [ atomtypes ] 가
    # 모든 [ moleculetype ] 보다 앞에 오기를 요구하는데, ACPYPE 는 둘을 한 파일에
    # 담아 주기 때문이다. 한 덩어리로 넣으면 "Invalid order for directive
    # atomtypes" 로 멈춘다.
    assert '#include "ligand_GMX_atomtypes.itp"' in topol
    assert '#include "ligand_GMX_moleculetype.itp"' in topol
    lines = topol.splitlines()
    atomtypes_at = next(
        i for i, line in enumerate(lines) if "ligand_GMX_atomtypes.itp" in line
    )
    first_moleculetype = next(
        i for i, line in enumerate(lines) if line.strip().lower() == "[ moleculetype ]"
    )
    assert atomtypes_at < first_moleculetype
    assert "[ molecules ]" in topol
    assert "LIG" in topol
    commands = log.read_text()
    assert "acpype -i" in commands
    assert "gmx pdb2gmx" in commands
    assert "gmx editconf" in commands
    assert "gmx solvate" in commands
    assert "gmx genion" in commands
    # F19: the explicit solvent must carry the same background salt as the
    # 0.15 M saltcon used by the implicit-solvent MM-GBSA evaluation, and the
    # equilibration must actually restrain (define = -DPOSRES).
    assert "-neutral" in commands
    assert "-conc 0.15" in commands
    assert "define = -DPOSRES" in (tmp_path / "out" / "P1" / "nvt.mdp").read_text()
    assert "define = -DPOSRES" in (tmp_path / "out" / "P1" / "npt.mdp").read_text()
    assert rows.iloc[0]["equilibration_restraints"] == "POSRES"
    assert rows.iloc[0]["ion_concentration_molar"] == pytest.approx(0.15)
    assert "gmx grompp" in commands
    assert (tmp_path / "out" / "P1" / "prod.mdp").read_text().count("nsteps = 25000000") == 1


def test_stage7_prep_fails_closed_without_explicit_complex_pose(tmp_path: Path) -> None:
    consensus = tmp_path / "ensemble.tsv"
    pd.DataFrame([{"target_id": "P1", "consensus_score": 2.0}]).to_csv(
        consensus, sep="\t", index=False
    )
    receptor_manifest = tmp_path / "receptors.tsv"
    pd.DataFrame([{"target_id": "P1", "medoid_pdb": str(tmp_path / "P1.pdb")}]).to_csv(
        receptor_manifest, sep="\t", index=False
    )
    # 실제 SDF 여야 한다. 포즈를 토폴로지에 맞춰 다시 쓸 때 이 분자를 참조로
    # 삼아 부분구조 매칭을 하기 때문이다. `"ligand\n"` 같은 대역은 실제보다
    # 좁아서, 원자 대응을 확인하는 코드가 있는지 없는지를 구분하지 못한다.
    # 대역 acpype 가 원자 1개(C1)짜리 토폴로지를 내므로 메탄으로 맞춘다.
    ligand = tmp_path / "ligand.sdf"
    _write_methane_sdf(ligand)
    manifest = tmp_path / "md.tsv"
    manifest.write_text("stale\n")

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
        check=False,
    )

    assert res.returncode != 0
    assert "--complex-pose or --complex-pose-manifest is required" in res.stderr
    assert not manifest.exists()


def test_stage7_prep_writes_only_explicit_diagnostic_empty_status(tmp_path: Path) -> None:
    manifest = tmp_path / "md.tsv"
    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage7_gromacs_prep.py"),
            "--ensemble-consensus",
            str(tmp_path / "missing.tsv"),
            "--receptor-manifest",
            str(tmp_path / "missing_receptors.tsv"),
            "--ligand-sdf",
            str(tmp_path / "missing.sdf"),
            "--out-dir",
            str(tmp_path / "out"),
            "--out-manifest",
            str(manifest),
            "--diagnostic-empty-status",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode == 0, res.stderr
    rows = pd.read_csv(manifest, sep="\t")
    assert rows["status"].tolist() == ["diagnostic_empty"]


def test_stage7_run_defaults_three_replicas_50ns_and_records_checksums(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    log = tmp_path / "commands.log"
    write_fake_stage7_tools(fake_bin, log)
    target_dir = tmp_path / "out" / "P1"
    target_dir.mkdir(parents=True)
    system = target_dir / "system.gro"
    topology = target_dir / "topol.top"
    tpr = target_dir / "prod.tpr"
    system.write_text("system\n")
    topology.write_text("topology\n")
    tpr.write_text("tpr\n")
    npt = target_dir / "npt.gro"
    npt.write_text("equilibrated\n")
    # 복제별 tpr 은 prod.mdp 를 고쳐 만든다. 준비 단계가 늘 남기는 파일이다.
    prod_mdp = target_dir / "prod.mdp"
    prod_mdp.write_text("nsteps = 10\ngen_vel = no\n")
    manifest = tmp_path / "md.tsv"
    pd.DataFrame(
        [
            {
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
            }
        ]
    ).to_csv(manifest, sep="\t", index=False)
    out_index = tmp_path / "traj.tsv"

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage7_gromacs_run.py"),
            "--manifest",
            str(manifest),
            "--out-dir",
            str(tmp_path / "out"),
            "--out-index",
            str(out_index),
        ],
        capture_output=True,
        text=True,
        env=env_with_fake_tools(fake_bin),
        check=False,
    )

    assert res.returncode == 0, res.stderr
    rows = pd.read_csv(out_index, sep="\t")
    assert rows["replica"].tolist() == [1, 2, 3]
    assert rows["status"].tolist() == ["ok", "ok", "ok"]
    assert rows["seed"].tolist() == [17391, 17392, 17393]
    assert rows["duration_ns"].tolist() == [50, 50, 50]
    assert rows["nsteps"].tolist() == [25000000, 25000000, 25000000]
    assert rows["trajectory_xtc_sha256"].astype(str).str.len().tolist() == [64, 64, 64]
    assert rows["tpr_sha256"].astype(str).str.len().tolist() == [64, 64, 64]
    # 복제마다 다른 tpr 을 썼으니 체크섬도 달라야 한다. 같으면 같은 계산이다.
    assert len(set(rows["tpr_sha256"].tolist())) == 3
    commands = log.read_text()
    # `-reseed` 는 replica exchange 의 씨앗이라 `-replex` 없이는 아무 일도 하지
    # 않는다. 그것에 기대면 세 복제가 같은 tpr·같은 시작 속도·같은 열욕 씨앗으로
    # 돌면서 이름만 복제가 된다(실측: 세 로그의 ld-seed 가 모두 같았다).
    assert "-reseed" not in commands
    # 복제마다 자기 tpr 을 만들고, 그 tpr 로 돈다.
    for replica, seed in ((1, 17391), (2, 17392), (3, 17393)):
        replica_tpr = target_dir / f"prod_r{replica}.tpr"
        replica_mdp = target_dir / f"prod_r{replica}.mdp"
        assert f"-f {replica_mdp}" in commands, f"복제 {replica} 의 mdp 를 만들지 않았습니다"
        assert f"-s {replica_tpr}" in commands, f"복제 {replica} 가 공유 tpr 로 돌았습니다"
        text = replica_mdp.read_text()
        assert "gen_vel = yes" in text, "속도를 새로 뽑지 않으면 같은 표본입니다"
        assert f"gen_seed = {seed}" in text
        assert f"ld_seed = {seed}" in text
    # 평형 전 좌표에 300 K 속도를 얹으면 계가 터진다(실측: mdrun 이 SIGSEGV).
    assert f"-c {target_dir / 'npt.gro'}" in commands or \
        f"-c {system}" in commands


def test_stage7_run_preserves_preexisting_outputs_when_validation_fails(tmp_path: Path) -> None:
    manifest = tmp_path / "missing.tsv"
    out_index = tmp_path / "traj.tsv"
    out_index.write_text("stale index\n")
    stale_xtc = tmp_path / "out" / "P1" / "prod_r1.xtc"
    stale_xtc.parent.mkdir(parents=True)
    stale_xtc.write_text("stale trajectory\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage7_gromacs_run.py"),
            "--manifest",
            str(manifest),
            "--out-dir",
            str(tmp_path / "out"),
            "--out-index",
            str(out_index),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "MD manifest is required and must be non-empty" in res.stderr
    assert out_index.read_text() == "stale index\n"
    assert stale_xtc.read_text() == "stale trajectory\n"


def test_the_workflows_own_duration_value_is_accepted(tmp_path: Path) -> None:
    """The default `report` run died in argparse before MD started.

    Snakemake coerces md.duration_ns with config_float, so it always arrives as
    "50.0", and both CLIs declared type=int.
    """
    for script in ("stage7_gromacs_prep.py", "stage7_gromacs_run.py"):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), "--duration-ns", "50.0"],
            capture_output=True, text=True, cwd=ROOT,
        )
        assert "invalid int value" not in result.stderr, (
            f"{script} still rejects the value the workflow passes: {result.stderr}"
        )


def test_a_zero_length_simulation_is_still_refused(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "stage7_gromacs_run.py"),
         "--duration-ns", "0"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert result.returncode != 0


def test_mdrun_keeps_the_whole_step_on_the_gpu(tmp_path: Path) -> None:
    """좌표 갱신과 결합항을 CPU 에 두면 매 스텝 GPU-CPU 왕복이 생긴다.

    실측(P52788, 89,184 원자): 기본 7.1 ns/day → -update gpu 18.0 →
    -update gpu -bonded gpu 68.6. 켜지 않으면 50 ns 여섯 번에 41일이 걸린다.
    """
    fake_bin = tmp_path / "bin"
    log = tmp_path / "commands.log"
    write_fake_stage7_tools(fake_bin, log)
    target_dir = tmp_path / "P1"
    target_dir.mkdir()
    tpr = target_dir / "prod_r1.tpr"
    tpr.write_text("tpr\n")

    sys.path.insert(0, str(ROOT / "scripts"))
    import stage7_gromacs_run as runner

    env = os.environ.copy()
    env["PATH"] = str(fake_bin)
    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = str(fake_bin)
    try:
        runner.mdrun(target_dir, tpr, 0.01, 1, 17391, "gmx")
    finally:
        os.environ["PATH"] = old_path

    commands = log.read_text()
    assert "-update gpu" in commands
    assert "-bonded gpu" in commands


def test_mdrun_falls_back_when_the_gpu_cannot_do_the_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GPU 갱신을 거부하는 계도 있다. 거기서 멈추면 표적 하나를 통째로 잃는다."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import stage7_gromacs_run as runner

    target_dir = tmp_path / "P1"
    target_dir.mkdir()
    tpr = target_dir / "prod_r1.tpr"
    tpr.write_text("tpr\n")
    seen: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        seen.append(list(cmd))
        if "-update" in cmd:
            return SimpleNamespace(
                returncode=1, stdout="",
                stderr="Update task on the GPU was required, but is not supported",
            )
        (target_dir / "prod_r1.xtc").write_text("trajectory")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    assert runner.mdrun(target_dir, tpr, 0.01, 1, 17391, "gmx") is True
    assert len(seen) == 2, "GPU 갱신이 거부됐는데 물러서지 않았습니다"
    assert "-update" in seen[0] and "-update" not in seen[1]
