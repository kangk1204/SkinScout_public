"""Regression tests for the Stage 7 ligand pose and topology plumbing.

Stage 7 had never run. Four things stopped it, each only visible by running it:

* the ensemble receptor manifest legitimately holds several medoids per target
  and Stage 7 rejected it as "duplicate target_id". The ensemble is the point
  of Stage 6, so the duplicate is not the error — the missing information is:
  Stage 6.5 knew which medoid won and threw the path away.
* co-folding and docking write the ligand as heavy atoms only (31 for
  adapalene) while the ACPYPE topology, built from the Stage 1 SDF, declares
  every atom (58). The mismatch was caught, which is right, but nothing
  bridged it.
* ACPYPE puts ``[ atomtypes ]`` and ``[ moleculetype ]`` in one file, and
  GROMACS requires every ``[ atomtypes ]`` to precede every
  ``[ moleculetype ]``. Included after the protein, it fails with "Invalid
  order for directive atomtypes".
* ``nsteps`` was written as ``duration_ns * 500_000`` — a float. GROMACS
  refuses ``50000.0``, and it refuses it at the *production* grompp, after
  minimisation, NVT and NPT have already run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import stage7_gromacs_prep as prep  # noqa: E402


# ------------------------------------------------------------- nsteps


@pytest.mark.parametrize(
    ("duration_ns", "expected"),
    [(0.1, 50_000), (1.0, 500_000), (50.0, 25_000_000), (0.002, 1_000)],
)
def test_nsteps_is_written_as_an_integer(
    tmp_path: Path, duration_ns: float, expected: int
) -> None:
    paths = prep.write_mdp_templates(tmp_path, duration_ns)
    body = paths["prod"].read_text()
    assert f"nsteps = {expected}" in body, body
    assert "nsteps = " in body and ".0" not in body.split("nsteps = ")[1].split("\n")[0]


def test_a_duration_below_one_step_is_refused(tmp_path: Path) -> None:
    """0 걸음짜리 MD 를 조용히 준비하면 빈 궤적이 결과로 나간다."""
    with pytest.raises(SystemExit, match="at least one"):
        prep.write_mdp_templates(tmp_path, 0.0)


def test_a_partial_step_rounds_up_rather_than_down(tmp_path: Path) -> None:
    """내림하면 요청보다 짧게 돈다. 요청한 시간은 하한이어야 한다."""
    paths = prep.write_mdp_templates(tmp_path, 0.0000021)
    assert "nsteps = 2" in paths["prod"].read_text()


# --------------------------------------------------- atomtypes 순서


def _acpype_itp(path: Path) -> Path:
    path.write_text(
        "; ligand\n"
        "[ atomtypes ]\n"
        ";name  bond_type   mass  charge  ptype  sigma  epsilon\n"
        " ca  ca  0.0  0.0  A  3.4e-01  3.6e-01\n"
        "\n"
        "[ moleculetype ]\n"
        "; name  nrexcl\n"
        " LIG  3\n"
        "\n"
        "[ atoms ]\n"
        ";  nr type resi res atom cgnr charge mass\n"
        "   1   ca    1 LIG   C1    1  -0.1  12.01\n"
        "   2   ca    1 LIG   O1    2  -0.2  16.00\n"
        "\n"
        "[ bonds ]\n"
        "  1  2  1\n"
    )
    return path


def _main_topology(path: Path) -> Path:
    path.write_text(
        '#include "amber99sb-ildn.ff/forcefield.itp"\n'
        "\n"
        "[ moleculetype ]\n"
        " Protein  3\n"
        "\n"
        "[ atoms ]\n"
        "   1  N  1  ALA  N  1  -0.4  14.01\n"
        "\n"
        "[ system ]\n"
        " protein\n"
        "\n"
        "[ molecules ]\n"
        "Protein  1\n"
    )
    return path


def test_atomtypes_are_included_before_the_first_moleculetype(tmp_path: Path) -> None:
    ligand = _acpype_itp(tmp_path / "ligand_GMX.itp")
    topology = _main_topology(tmp_path / "topol.top")
    prep.append_ligand_include(topology, ligand)

    lines = topology.read_text().splitlines()
    atomtypes_at = next(
        i for i, line in enumerate(lines)
        if line.strip().startswith("#include") and "atomtypes" in line
    )
    first_moleculetype = next(
        i for i, line in enumerate(lines) if line.strip().lower() == "[ moleculetype ]"
    )
    assert atomtypes_at < first_moleculetype, (
        "GROMACS 는 [ atomtypes ] 가 모든 [ moleculetype ] 보다 앞에 오기를 요구합니다"
    )


def test_the_split_keeps_every_section(tmp_path: Path) -> None:
    """가르면서 잃어버리는 섹션이 있으면 토폴로지가 조용히 불완전해진다."""
    ligand = _acpype_itp(tmp_path / "ligand_GMX.itp")
    atomtypes_itp, rest_itp = prep._split_ligand_atomtypes(ligand)

    assert "[ atomtypes ]" in atomtypes_itp.read_text()
    assert "[ atomtypes ]" not in rest_itp.read_text()
    rest = rest_itp.read_text()
    for section in ("[ moleculetype ]", "[ atoms ]", "[ bonds ]"):
        assert section in rest, section
    assert " ca  ca  0.0" in atomtypes_itp.read_text()


def test_a_topology_without_a_forcefield_include_is_refused(tmp_path: Path) -> None:
    """넣을 자리를 못 찾으면 아무 데나 넣는 대신 말한다."""
    ligand = _acpype_itp(tmp_path / "ligand_GMX.itp")
    topology = tmp_path / "topol.top"
    topology.write_text("[ moleculetype ]\n Protein 3\n\n[ system ]\n x\n")
    with pytest.raises(SystemExit, match="forcefield include"):
        prep.append_ligand_include(topology, ligand)


# ------------------------------------------------- 토폴로지 원자 이름


def test_topology_atom_names_are_read_in_declaration_order(tmp_path: Path) -> None:
    """직접 지어낸 이름은 grompp 가 원자 수만큼 경고하고 -maxwarn 을 넘긴다."""
    ligand = _acpype_itp(tmp_path / "ligand_GMX.itp")
    assert prep._ligand_topology_atom_names(ligand) == ["C1", "O1"]


def test_comment_and_blank_lines_do_not_become_atom_names(tmp_path: Path) -> None:
    ligand = tmp_path / "l.itp"
    ligand.write_text(
        "[ atoms ]\n"
        "; nr type resi res atom cgnr charge\n"
        "\n"
        "   1   ca    1 LIG   C1    1  -0.1\n"
        "; trailing comment\n"
        "[ bonds ]\n"
        "   1 2 1\n"
    )
    assert prep._ligand_topology_atom_names(ligand) == ["C1"]


# ------------------------------------- 앙상블 매니페스트의 medoid 선택


def _receptor_manifest(path: Path, rows: list[tuple[str, Path]]) -> Path:
    lines = ["target_id\tmedoid_pdb"]
    lines += [f"{uid}\t{pdb}" for uid, pdb in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_the_consensus_choice_selects_among_several_medoids(tmp_path: Path) -> None:
    """표적당 medoid 가 여럿인 것은 앙상블의 정상 상태다."""
    first = tmp_path / "medoid_00.pdb"
    second = tmp_path / "medoid_01.pdb"
    for pdb in (first, second):
        pdb.write_text("ATOM      1  CA  ALA A   1       0.0   0.0   0.0\n")
    manifest = _receptor_manifest(
        tmp_path / "receptors.tsv", [("P00918", first), ("P00918", second)]
    )
    chosen = prep.read_receptor_manifest(manifest, {"P00918": str(second)})
    assert chosen == {"P00918": second}


def test_several_medoids_without_a_recorded_choice_is_refused(tmp_path: Path) -> None:
    """임의로 하나를 집으면 아무도 고르지 않은 구조로 MD 를 돌리게 된다."""
    first = tmp_path / "medoid_00.pdb"
    second = tmp_path / "medoid_01.pdb"
    for pdb in (first, second):
        pdb.write_text("ATOM      1  CA  ALA A   1       0.0   0.0   0.0\n")
    manifest = _receptor_manifest(
        tmp_path / "receptors.tsv", [("P00918", first), ("P00918", second)]
    )
    with pytest.raises(SystemExit, match="best_medoid_pdb"):
        prep.read_receptor_manifest(manifest, None)


def test_a_single_medoid_per_target_still_works(tmp_path: Path) -> None:
    only = tmp_path / "medoid_00.pdb"
    only.write_text("ATOM      1  CA  ALA A   1       0.0   0.0   0.0\n")
    manifest = _receptor_manifest(tmp_path / "receptors.tsv", [("P00918", only)])
    assert prep.read_receptor_manifest(manifest, None) == {"P00918": only}


def test_a_consensus_naming_an_absent_receptor_is_refused(tmp_path: Path) -> None:
    """합의가 가리킨 구조가 매니페스트에 없으면 조용히 다른 것을 쓰면 안 된다."""
    only = tmp_path / "medoid_00.pdb"
    only.write_text("ATOM      1  CA  ALA A   1       0.0   0.0   0.0\n")
    manifest = _receptor_manifest(
        tmp_path / "receptors.tsv", [("P00918", only), ("P00918", only)]
    )
    with pytest.raises(SystemExit, match="not in the receptor manifest"):
        prep.read_receptor_manifest(manifest, {"P00918": str(tmp_path / "absent.pdb")})


# ------------------------------------------------- 프로덕션 궤적 출력


def test_the_production_mdp_asks_for_a_trajectory(tmp_path: Path) -> None:
    """궤적 주기가 없으면 GROMACS 는 궤적을 쓰지 않는다.

    `nstxout-compressed` 의 기본값은 0 이다. mdrun 은 종료코드 0 으로 끝나고
    .gro/.edr/.log/.cpt 만 남는다 - 궤적을 만들려고 있는 단계가 궤적을 만들지
    않고, 그것을 먹는 MM-GBSA 는 프레임 없이 시작할 수 없다. 실패가 "0/2 복제
    완료"로만 보여서 원인이 mdrun 쪽에 있는 것처럼 읽혔다.
    """
    body = prep.write_mdp_templates(tmp_path, 0.1)["prod"].read_text()
    assert "nstxout-compressed" in body
    stride = int(
        next(line for line in body.splitlines()
             if line.startswith("nstxout-compressed")).split("=")[1]
    )
    assert stride > 0, "0 이면 궤적을 쓰지 않는다는 뜻입니다"


@pytest.mark.parametrize("duration_ns", [0.001, 0.01, 0.1, 1.0, 50.0])
def test_every_run_length_produces_at_least_one_frame(
    tmp_path: Path, duration_ns: float
) -> None:
    """짧은 시험 실행에서 프레임이 0개면 그 실행으로는 아무것도 확인할 수 없다."""
    body = prep.write_mdp_templates(tmp_path, duration_ns)["prod"].read_text()
    values = {
        line.split("=")[0].strip(): line.split("=")[1].strip()
        for line in body.splitlines() if "=" in line and not line.startswith(";")
    }
    nsteps = int(values["nsteps"])
    stride = int(values["nstxout-compressed"])
    assert nsteps // stride >= 1, (
        f"{duration_ns} ns 에서 프레임이 {nsteps // stride}개입니다"
    )


def test_a_long_run_does_not_write_a_frame_every_step(tmp_path: Path) -> None:
    """매 걸음을 남기면 50 ns 에 수백 GB 가 된다."""
    body = prep.write_mdp_templates(tmp_path, 50.0)["prod"].read_text()
    stride = int(
        next(line for line in body.splitlines()
             if line.startswith("nstxout-compressed")).split("=")[1]
    )
    assert stride == prep.PROD_TRAJECTORY_STRIDE_STEPS
    assert stride >= 1000


# ------------------------------------- 포즈와 토폴로지의 원자 대응


def test_the_pose_is_realigned_even_when_the_atom_counts_match() -> None:
    """개수가 같다고 같은 원자가 같은 자리에 있는 것은 아니다.

    공동 접힘은 리간드 원자에 `C22`, `N19` 같은 이름을 붙이고 ACPYPE 는
    `C1`, `N1` 을 붙인다. 카페인은 양쪽 다 14개라 개수만 보는 조건은 이 경우를
    건너뛰는데, 그러면 좌표와 토폴로지가 **파일 순서로만** 짝지어진다. 두 도구가
    원자를 같은 순서로 나열할 이유는 없다.

    GROMACS 는 여기서 멈추지 않는다:

        WARNING: 14 non-matching atom names
        atom names from topol.top will be used
        atom names from solvated.gro will be ignored

    경고 하나를 남기고 계속 가므로, 탄소 자리에 질소 좌표가 들어간 계가 그대로
    평형화되고 MD 를 돈다. 결과는 정상적인 숫자로 나온다.
    """
    source = (ROOT / "scripts" / "stage7_gromacs_prep.py").read_text()
    call = source[source.index("align_pose_to_topology(\n") - 400:]
    head = call[: call.index("align_pose_to_topology(")]
    assert "if pose_atoms != ligand_atom_count" not in head, (
        "개수가 다를 때만 정렬하면, 개수가 같고 이름이 다른 경우가 새어 나갑니다"
    )


def test_alignment_is_verified_against_the_topology_count() -> None:
    """정렬 뒤에도 개수가 안 맞으면 멈춰야 한다. 맞춰 놓고 확인하지 않으면
    맞췄다는 사실을 아무도 모른다."""
    source = (ROOT / "scripts" / "stage7_gromacs_prep.py").read_text()
    assert "if written != ligand_atom_count" in source


def test_a_ligand_without_hydrogens_does_not_crash_the_minimiser() -> None:
    """참조 분자에 수소가 없으면 고정할 것이 전부라 움직일 원자가 없다.

    RDKit 의 BFGS 는 그 상태에서 "Failed Expression: status >= 0" 으로 죽는다.
    카페인의 Stage 1 SDF 가 정확히 그렇다(중원자 14, 수소 0).
    """
    source = (ROOT / "scripts" / "stage7_gromacs_prep.py").read_text()
    body = source[source.index("def align_pose_to_topology"):]
    assert "movable" in body[: body.index("return len(rows)")], (
        "움직일 원자가 없을 때 최적화를 건너뛰는 경로가 있어야 합니다"
    )


# --------------------------------------------------- 리간드 수소 검사


def test_a_ligand_without_hydrogens_is_refused_before_any_md(tmp_path: Path) -> None:
    """수소 없는 리간드로도 MD 는 끝까지 돌고, MM-GBSA 만 나중에 멈춘다.

    Stage 1 은 SDF 를 둘 남긴다 - `standardized.sdf`(수소 없음)와 xTB 최적화를
    거친 `compound_canonical.sdf`(수소 포함). 전자를 주면 ACPYPE 가 중원자만의
    토폴로지를 만들고 GROMACS 가 용매화·평형화·프로덕션 MD 를 아무 불평 없이
    끝낸다. 실측으로 3표적 × 2복제 = 6개 궤적을 두 시간에 걸쳐 만들었다.
    그러고 나서야 MM-GBSA 가 "Some energy terms are undefined" 로 멈춘다 -
    원인을 말하지 않는 메시지다.
    """
    from rdkit import Chem

    no_h = tmp_path / "no_hydrogens.sdf"
    mol = Chem.MolFromSmiles("Cn1c(=O)c2c(ncn2C)n(C)c1=O")  # 카페인
    writer = Chem.SDWriter(str(no_h))
    writer.write(mol)
    writer.close()

    with pytest.raises(SystemExit, match="no explicit hydrogens"):
        prep.require_explicit_hydrogens(no_h)


def test_a_hydrogen_bearing_ligand_passes(tmp_path: Path) -> None:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    with_h = tmp_path / "with_hydrogens.sdf"
    mol = Chem.AddHs(Chem.MolFromSmiles("Cn1c(=O)c2c(ncn2C)n(C)c1=O"))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    writer = Chem.SDWriter(str(with_h))
    writer.write(mol)
    writer.close()

    prep.require_explicit_hydrogens(with_h)  # 예외가 없어야 한다


def test_a_single_atom_ligand_is_not_falsely_refused(tmp_path: Path) -> None:
    """단원자 이온에는 붙일 수소가 없다. 그것을 결함으로 읽으면 안 된다."""
    from rdkit import Chem

    ion = tmp_path / "ion.sdf"
    mol = Chem.MolFromSmiles("[Na+]")
    writer = Chem.SDWriter(str(ion))
    writer.write(mol)
    writer.close()

    prep.require_explicit_hydrogens(ion)


def test_the_check_runs_before_the_first_gromacs_call() -> None:
    """확인이 늦으면 몇 시간 뒤에야 드러난다."""
    source = (ROOT / "scripts" / "stage7_gromacs_prep.py").read_text()
    body = source[source.index("def main()"):]
    check_at = body.index("require_explicit_hydrogens(")
    # main() 은 표적마다 prepare_target() 을 부르고, 그 안에서 acpype 와 GROMACS 가
    # 돈다. 검사는 그 루프에 들어가기 전에 끝나야 한다.
    first_work = body.index("prepare_target(")
    assert check_at < first_work, (
        "수소 검사가 표적 준비보다 먼저여야 합니다"
    )
    # 다만 인자 검증보다 앞서면 안 된다. 그러면 빈 합의 파일 같은 상류 문제까지
    # 이 검사가 가로채, 실행이 실제 원인을 말하지 못한다.
    nonempty_at = body.index('"Ligand SDF is missing or empty')
    assert nonempty_at < check_at, (
        "수소 검사가 기본 입력 검증보다 뒤에 와야 합니다"
    )


# ----------------------------------------------- MM-GBSA 복제 처리


def test_mmgbsa_runs_one_trajectory_at_a_time() -> None:
    """복제를 `-ct` 에 함께 넘기면 gmx_MMPBSA 가 결과를 쓰다가 멈춘다.

    실측(P11086, 카페인): 궤적 하나면 ΔTOTAL -19.53 kcal/mol 이 나오고, 같은 계에
    복제 둘을 함께 주면 "Some energy terms are undefined" 로 실패한다. 계산 자체는
    완전하다 - complex/receptor/ligand 세 성분이 모두 21프레임 최소화를 끝내고,
    결과를 쓰는 자리에서만 멈춘다. 그래서 원인이 계나 리간드에 있는 것처럼 읽힌다.

    따로 도는 편이 과학적으로도 맞다. 복제는 독립 표본이라 평균과 편차를 낼 수
    있는데, 궤적을 이어 붙이면 그 정보가 사라진다.
    """
    import re

    source = (ROOT / "scripts" / "stage7_mmgbsa.py").read_text()
    body = source[source.index("def _run_one_trajectory"):]
    ct = re.search(r'"-ct",\s*([^\]]+?)\]', body, re.S)
    assert ct, "-ct 인자를 찾지 못했습니다"
    assert "*" not in ct.group(1), (
        "궤적을 펼쳐서 한 번에 넘기고 있습니다: " + ct.group(1).strip()
    )


def test_mmgbsa_averages_the_replicas_and_reports_their_spread() -> None:
    """복제를 돌려 놓고 하나만 쓰면 나머지 계산이 버려진다."""
    source = (ROOT / "scripts" / "stage7_mmgbsa.py").read_text()
    assert "sum(values) / len(values)" in source, "복제 평균을 내야 합니다"
    assert "폭" in source or "spread" in source, (
        "복제 간 편차를 로그에 남겨야 합니다 - 평균만으로는 흩어짐을 알 수 없습니다"
    )


def test_each_replica_keeps_its_own_results_file() -> None:
    """결과 파일 하나만 두면 마지막 복제가 앞의 것을 덮어쓴다."""
    source = (ROOT / "scripts" / "stage7_mmgbsa.py").read_text()
    assert "FINAL_RESULTS_MMPBSA_r" in source, (
        "복제별 결과 파일이 없으면 무엇을 평균했는지 확인할 수 없습니다"
    )


# ------------------------------------------------- MM-GBSA 주기 경계


def test_mmgbsa_removes_pbc_before_running() -> None:
    """gmx_MMPBSA 의 `-ct` 도움말이 요구한다: "pbc have been removed".

    하지 않으면 상자를 가로지른 분자가 끊어진 채로 들어간다. 실측으로 백본 결합
    하나가 74.6 A 가 됐고, sander 의 BOND 항이 9,480만 kcal/mol 로 폭발했다.
    그 값은 출력 서식(13자리)을 넘겨 `*************` 로 찍히고, gmx_MMPBSA 는
    읽지 못해 "Some energy terms are undefined" 로 멈춘다 - 원인을 짐작할 수 없는
    메시지다.

    정육면체 상자에서는 우연히 드러나지 않았다. 부피를 29% 줄이려고 십이면체로
    바꾸면서 삼사정계가 되자 세 표적 모두에서 재현됐다.
    """
    import re

    source = (ROOT / "scripts" / "stage7_mmgbsa.py").read_text()
    assert "def strip_pbc(" in source, "PBC 제거 단계가 있어야 합니다"
    body = source[source.index("def _run_one_trajectory"):]
    strip_at = body.index("strip_pbc(")
    cmd_at = body.index('"gmx_MMPBSA"')
    assert strip_at < cmd_at, "PBC 제거가 gmx_MMPBSA 호출보다 먼저여야 합니다"
    # 원본이 아니라 처리된 궤적을 넘겨야 한다.
    ct = re.search(r'"-ct",\s*str\((\w+)\)', body)
    assert ct and ct.group(1) == "whole", (
        f"-ct 에 처리된 궤적을 넘겨야 합니다: {ct.group(1) if ct else '없음'}"
    )


def test_pbc_removal_keeps_the_complex_together() -> None:
    """`-pbc whole -center` 는 단백질만 옮겨 리간드를 상자 반대편에 남긴다.

    실측으로 리간드가 수용체에서 72.6 A 떨어졌고 ΔG 가 전부 -0.00 으로 나왔다.
    그 0 은 "결합하지 않는다"로 읽히지만 실제로는 계를 잘못 자른 것이다 -
    ΔVDWAALS 까지 0.00 이면 접촉 자체가 없었다는 뜻이다.
    """
    source = (ROOT / "scripts" / "stage7_mmgbsa.py").read_text()
    body = source[source.index("def strip_pbc"):source.index("def _run_one_trajectory")]
    assert '"-pbc", "mol"' in body, (
        "분자 단위로 다뤄야 복합체가 함께 남습니다"
    )
    assert '"-ur", "compact"' in body, (
        "compact 표현이라야 용질 주위로 모입니다"
    )
