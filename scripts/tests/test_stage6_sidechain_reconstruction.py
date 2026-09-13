"""Regression tests for Stage 6's side-chain reconstruction and its gate.

BioEmu samples backbones. Its output carries N, CA, C, O and CB and nothing
else — measured on a 257-residue receptor, 785 side-chain heavy atoms were
missing. Handing that to Stage 6.5 split two ways:

* RTMScore refused it ("The graph of pocket cannot be generated"), which is
  the honest outcome;
* GNINA scored it without complaining. A pocket with no side chains is far
  more open than the real one, so that number is not comparable to anything,
  and it renders as an ordinary score. The silent answer is the dangerous one.

Reconstruction is two steps, and each step failed in its own way before this
was pinned:

* ``pdbfixer`` places default rotamers and does not resolve clashes. PHE 16
  and TRP 12 ended up 1.73 Å apart; RDKit reads that distance as a bond and
  gives a carbon five of them.
* One OpenMM minimisation pass is not always enough. Cut at 500 iterations,
  a carboxylate stayed at 1.59 Å (2.2 Å is normal) on one medoid but not the
  other — one strained residue is enough to make RTMScore discard the whole
  receptor, so "it worked on the frame I checked" is not evidence.

So the geometry of the written file is measured rather than assumed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import stage6_bioemu as stage6  # noqa: E402
import stage6_ensemble_dock as dock  # noqa: E402


def _atom(serial: int, name: str, resname: str, resseq: int,
          x: float, y: float, z: float, element: str = "") -> str:
    element = element or name[0]
    return (
        f"ATOM  {serial:5d} {name:<4s} {resname:>3s} A{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {element:>2s}"
    )


def _write(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\nEND\n")
    return path


# ------------------------------------------------- 뒤틀린 기하 검사


def test_a_collapsed_carboxylate_is_reported(tmp_path: Path) -> None:
    """ASP 의 산소 두 개가 1.59 A 면 RDKit 이 산소에 결합 3개를 매긴다."""
    pdb = _write(tmp_path / "bad.pdb", [
        _atom(1, "N", "ASP", 170, 20.180, 0.552, 5.504),
        _atom(2, "CA", "ASP", 170, 19.955, 1.954, 5.884),
        _atom(3, "C", "ASP", 170, 19.478, 2.791, 4.683),
        _atom(4, "CB", "ASP", 170, 21.231, 2.561, 6.487),
        _atom(5, "O", "ASP", 170, 19.907, 2.560, 3.547),
        _atom(6, "CG", "ASP", 170, 21.715, 1.677, 7.679),
        _atom(7, "OD1", "ASP", 170, 22.383, 1.233, 8.646),
        _atom(8, "OD2", "ASP", 170, 22.643, 2.524, 7.753),
    ])
    strained = stage6.strained_residues(pdb)
    assert len(strained) == 1
    assert "ASP170" in strained[0]
    assert "1.59" in strained[0], "실측 거리를 메시지에 남겨야 진단이 됩니다"


def test_a_normal_carboxylate_is_not_reported(tmp_path: Path) -> None:
    pdb = _write(tmp_path / "ok.pdb", [
        _atom(1, "CG", "ASP", 170, 0.0, 0.0, 0.0),
        _atom(2, "OD1", "ASP", 170, 1.25, 0.0, 0.0),
        _atom(3, "OD2", "ASP", 170, -0.63, 1.09, 0.0),
    ])
    assert stage6.strained_residues(pdb) == []


@pytest.mark.parametrize(
    ("resname", "first", "second"),
    [
        ("GLU", "OE1", "OE2"),
        ("ASN", "OD1", "ND2"),
        ("GLN", "OE1", "NE2"),
        ("ARG", "NH1", "NH2"),
        ("VAL", "CG1", "CG2"),
        ("LEU", "CD1", "CD2"),
        ("PHE", "CD1", "CD2"),
    ],
)
def test_every_covered_residue_type_is_actually_checked(
    tmp_path: Path, resname: str, first: str, second: str
) -> None:
    """표에 적어 놓고 검사하지 않으면 표가 있다는 사실이 오해를 만든다."""
    pdb = _write(tmp_path / f"{resname}.pdb", [
        _atom(1, first, resname, 5, 0.0, 0.0, 0.0),
        _atom(2, second, resname, 5, 0.8, 0.0, 0.0),
    ])
    strained = stage6.strained_residues(pdb)
    assert strained, f"{resname} 의 {first}-{second} 0.8 A 를 잡지 못했습니다"


def test_a_residue_missing_the_pair_is_not_a_false_alarm(tmp_path: Path) -> None:
    """곁사슬이 아직 없는 잔기를 뒤틀렸다고 하면 안 된다. 그것은 다른 문제다."""
    pdb = _write(tmp_path / "backbone.pdb", [
        _atom(1, "N", "ASP", 170, 0.0, 0.0, 0.0),
        _atom(2, "CA", "ASP", 170, 1.5, 0.0, 0.0),
        _atom(3, "C", "ASP", 170, 2.0, 1.4, 0.0),
        _atom(4, "O", "ASP", 170, 1.3, 2.4, 0.0),
        _atom(5, "CB", "ASP", 170, 2.1, -1.2, 0.0),
    ])
    assert stage6.strained_residues(pdb) == []


# ------------------------------------------------- 곁사슬 유무 게이트


def test_a_backbone_only_receptor_is_refused_before_docking(tmp_path: Path) -> None:
    """GNINA 는 이것을 채점한다. 그 숫자가 화면에 나가면 안 된다."""
    lines = []
    serial = 1
    for i in range(1, 21):
        for name in ("N", "CA", "C", "O", "CB"):
            lines.append(_atom(serial, name, "VAL", i, float(i), 0.0, 0.0))
            serial += 1
    pdb = _write(tmp_path / "backbone_only.pdb", lines)
    assert dock.sidechain_fraction(pdb) == 0.0


def test_a_full_atom_receptor_passes_the_gate(tmp_path: Path) -> None:
    lines = []
    serial = 1
    for i in range(1, 21):
        for name in ("N", "CA", "C", "O", "CB", "CG1", "CG2"):
            lines.append(_atom(serial, name, "VAL", i, float(i), 0.0, 0.0))
            serial += 1
    pdb = _write(tmp_path / "full.pdb", lines)
    assert dock.sidechain_fraction(pdb) == 1.0


def test_glycine_only_chains_do_not_divide_by_zero(tmp_path: Path) -> None:
    """글라이신과 알라닌에는 셀 곁사슬이 없다. 0/0 을 실패로 읽으면 안 된다."""
    lines = [
        _atom(1, "N", "GLY", 1, 0.0, 0.0, 0.0),
        _atom(2, "CA", "GLY", 1, 1.5, 0.0, 0.0),
        _atom(3, "C", "GLY", 1, 2.0, 1.4, 0.0),
        _atom(4, "O", "GLY", 1, 1.3, 2.4, 0.0),
    ]
    pdb = _write(tmp_path / "gly.pdb", lines)
    assert dock.sidechain_fraction(pdb) == 1.0


def test_the_gate_threshold_is_a_fraction() -> None:
    assert 0.0 < dock.MIN_SIDECHAIN_FRACTION <= 1.0


def test_the_minimiser_runs_to_convergence_rather_than_a_fixed_count() -> None:
    """500회로 끊었을 때 프레임에 따라 국소 최소에 갇혔다.

    OpenMM 에서 maxIterations=0 은 "수렴할 때까지"이고, 실측으로 1초 미만이다.
    """
    assert stage6.SIDECHAIN_MIN_ITERATIONS == 0
    assert stage6.MAX_MINIMISATION_ATTEMPTS >= 2, (
        "한 번 내려가서 안 되면 다시 시도할 수 있어야 합니다"
    )


def test_a_missing_pdbfixer_is_reported_rather_than_silently_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """복원하지 못한 채 통과시키면 하류가 백본만 있는 수용체를 채점한다."""
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name in {"pdbfixer", "openmm"}:
            raise ImportError(f"no {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    pdb = _write(tmp_path / "x.pdb", [_atom(1, "CA", "GLY", 1, 0.0, 0.0, 0.0)])
    status = stage6.reconstruct_sidechains(pdb)
    assert status == "backbone_only_pdbfixer_missing"


# ------------------------------------------- 서브프로세스 출력 인코딩


def test_subprocess_output_encoding_is_pinned_not_inherited() -> None:
    """도구 출력의 인코딩이 실행 환경의 로케일에 달려 있으면 안 된다.

    `subprocess.run(..., text=True)` 는 로케일의 기본 인코딩으로 디코딩한다.
    대화형 셸에서는 UTF-8 이라 잘 돌다가, 백그라운드로 띄운 같은 명령이 ASCII
    로케일을 물려받으면 `UnicodeDecodeError: 'ascii' codec can't decode byte
    0xe2` 로 죽는다 - BioEmu 의 진행 표시줄에 있는 화살표 한 글자가 스테이지
    전체를 무너뜨렸다. 30분치 샘플링을 마친 뒤 결과를 읽는 자리에서 터지므로
    그 시간이 통째로 버려진다.

    도구가 무엇을 뱉든 우리가 어떻게 읽을지는 우리가 정한다.
    """
    import re

    scripts = ROOT / "scripts"
    offenders = []
    for name in (
        "stage6_bioemu.py", "stage6_ensemble_dock.py",
        "stage7_gromacs_prep.py", "stage7_gromacs_run.py", "stage7_mmgbsa.py",
        "stage8_crest.py", "stage8_xtb_cluster.py", "stage8_dft.py",
        "stage4_prepare_structures.py",
    ):
        path = scripts / name
        if not path.exists():
            continue
        body = path.read_text()
        for match in re.finditer(r"text=True(?!\s*,\s*encoding)", body):
            line = body[: match.start()].count("\n") + 1
            offenders.append(f"{name}:{line}")
    assert not offenders, (
        "text=True 에 encoding 을 지정하지 않은 곳: " + ", ".join(offenders)
    )


def test_the_two_restraint_forces_do_not_share_a_parameter_name() -> None:
    """OpenMM 의 전역 매개변수 이름은 계 안에서 유일해야 한다.

    곁사슬 완화는 두 개의 CustomExternalForce 를 쓴다 - 백본을 붙드는 것과,
    마지막 시도에서 뒤틀린 잔기만 남기고 나머지를 붙드는 것이다. 둘이 같은
    이름(`k`)을 쓰면 OpenMM 이 계를 만들지 않는다:

        Two Forces define different default values for the parameter 'k'

    그러면 잔기 하나를 고치려던 시도가 **구조 전체의 복원을 실패시킨다.**
    실측으로 Q14393 의 medoid 하나가 그렇게 곁사슬 0% 로 나갔고, 그 하나 때문에
    stage 6.5 가 표적 전체를 거부했다.
    """
    import re

    source = (ROOT / "scripts" / "stage6_bioemu.py").read_text()
    names = re.findall(r'addGlobalParameter\(\s*\n?\s*"([^"]+)"', source)
    assert len(names) >= 2, f"구속 힘이 둘 이상이어야 합니다: {names}"
    assert len(names) == len(set(names)), (
        f"전역 매개변수 이름이 겹칩니다: {names}"
    )


def test_a_structure_is_kept_even_when_a_residue_stays_strained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """잔기 하나를 피하려고 복원한 곁사슬 수백 개를 버리면 안 된다.

    버리면 남는 것은 백본만 있는 파일인데, 그것이야말로 stage 6.5 의 게이트가
    막는 상태다. 작은 문제를 완전한 실패로 바꾸는 교환이다.
    """
    source = (ROOT / "scripts" / "stage6_bioemu.py").read_text()
    assert "pdbfixer_sidechains_openmm_relaxed_with_strain" in source, (
        "뒤틀림이 남아도 구조를 유지하는 상태가 있어야 합니다"
    )
    # 그 경로에서 tmp 를 지우지 않고 옮기는지 확인한다.
    tail = source.split("완화 후에도 뒤틀린 잔기")[1]
    assert "tmp.replace(pdb)" in tail[:600], "구조를 유지하지 않고 버리고 있습니다"


def test_a_collapsed_pair_is_pulled_apart_before_minimising() -> None:
    """최소화만으로는 1.55 A 짜리 말단 쌍이 안 풀린다.

    그 자리가 국소 최소이고, 정상 거리(실측 2.14~2.26 A)로 가려면 각도 장벽을
    넘어야 하는데 내리막만 따라가는 최소화는 그것을 못 넘는다. 실측으로 GLU16 은
    구속을 풀어 세 번 다시 내려가도 1.59 A 에 머물렀고, 그 잔기 하나 때문에
    RDKit 이 산소에 결합 3개를 매겨 RTMScore 가 수용체 전체를 버렸다.

    벌려 놓고 출발하면 최소화가 제 일을 한다 - 같은 구조에서 뒤틀림 1개가
    0개가 됐다.
    """
    source = (ROOT / "scripts" / "stage6_bioemu.py").read_text()
    assert "_pull_apart" in source
    # 벌리는 목표 거리가 실측 범위 안에 있어야 한다.
    assert 2.1 <= stage6.IDEAL_TERMINAL_PAIR_ANGSTROM <= 2.3, (
        f"실제 구조는 2.14~2.26 A 다: {stage6.IDEAL_TERMINAL_PAIR_ANGSTROM}"
    )
    # 벌린 뒤에 최소화해야 의미가 있다. 순서를 못 박는다.
    body = source[source.index("def _relax_only"):]
    pull_at = body.index("_pull_apart(")
    minimise_at = body.index("minimizeEnergy")
    assert pull_at < minimise_at, "벌리기가 최소화보다 먼저여야 합니다"


def test_the_pull_apart_table_covers_every_checked_residue() -> None:
    """검사하는 잔기를 고치지 못하면 검사가 경고만 남기고 끝난다."""
    import re

    source = (ROOT / "scripts" / "stage6_bioemu.py").read_text()
    checked = set(re.findall(r'\("([A-Z]{3})",\s*"[A-Z0-9]+",\s*"[A-Z0-9]+",\s*[0-9.]+\)', source))
    pull = source[source.index("def _pull_apart"):source.index("def _relax_only")]
    fixable = set(re.findall(r'"([A-Z]{3})": \(', pull))
    missing = checked - fixable
    assert not missing, f"검사만 하고 고치지 못하는 잔기: {sorted(missing)}"


def _residue(num: int, resname: str,
             atoms: dict[str, tuple[float, float, float]]) -> list[str]:
    lines = []
    for i, (name, (x, y, z)) in enumerate(atoms.items(), start=1):
        lines.append(_atom(i, name, resname, num, x, y, z))
    return lines


def test_strain_check_sees_residues_the_old_table_never_listed(tmp_path: Path) -> None:
    """옛 검사는 11쌍짜리 표였다. LYS 는 그 표에 없어 아무리 겹쳐도 통과했다."""
    pdb = tmp_path / "lys.pdb"
    pdb.write_text("\n".join(_residue(10, "LYS", {
        "N": (0.0, 0.0, 0.0), "CA": (1.458, 0.0, 0.0), "CB": (2.0, 1.4, 0.0),
        "CG": (3.5, 1.4, 0.0), "CD": (4.1, 2.8, 0.0), "CE": (5.6, 2.8, 0.0),
        "NZ": (4.6, 3.6, 0.0),
    })) + "\nEND\n")
    strained = stage6.strained_residues(pdb)
    assert strained, "LYS 의 CD-NZ 겹침을 잡지 못했습니다"
    assert "LYS10" in strained[0]


def test_strain_check_sees_atoms_from_a_different_residue(tmp_path: Path) -> None:
    """잔기 안만 보면 다른 잔기의 원자가 파고든 것은 영원히 보이지 않는다."""
    pdb = tmp_path / "clash.pdb"
    lines = _residue(10, "ALA", {
        "N": (0.0, 0.0, 0.0), "CA": (1.458, 0.0, 0.0),
        "C": (2.0, -1.2, 0.0), "CB": (2.0, 1.4, 0.0),
    })
    lines += _residue(50, "SER", {
        "N": (20.0, 0.0, 0.0), "CA": (21.4, 0.0, 0.0),
        "CB": (2.0, 1.4, 1.5), "OG": (2.0, 1.4, 3.0),
    })
    pdb.write_text("\n".join(lines) + "\nEND\n")
    strained = stage6.strained_residues(pdb)
    assert any("SER50" in token for token in strained), (
        f"잔기를 넘는 1.50 A 충돌을 잡지 못했습니다: {strained}"
    )


def test_a_normal_peptide_bond_is_not_reported_as_a_clash(tmp_path: Path) -> None:
    """C(i)-N(i+1) 은 1.33 A 로 정상이다. 결합인 줄 모르면 매 잔기가 지적된다."""
    pdb = tmp_path / "chain.pdb"
    lines = _residue(1, "ALA", {
        "N": (0.0, 0.0, 0.0), "CA": (1.458, 0.0, 0.0),
        "C": (2.0, 1.42, 0.0), "O": (1.35, 2.46, 0.0), "CB": (2.0, -0.7, 1.2),
    })
    lines += _residue(2, "ALA", {
        "N": (3.33, 1.42, 0.0), "CA": (4.05, 2.68, 0.0),
        "C": (5.55, 2.5, 0.0), "O": (6.2, 3.5, 0.3), "CB": (3.7, 3.5, 1.24),
    })
    pdb.write_text("\n".join(lines) + "\nEND\n")
    assert stage6.strained_residues(pdb) == []


def test_reconstruction_pins_every_source_of_randomness() -> None:
    """씨앗이 없으면 같은 파일이 실행마다 다른 판정을 낸다 - 재현이 불가능하다.

    난수는 네 군데서 들어온다. 하나라도 놓치면 나머지를 고정한 보람이 없다:
    실측으로 pdbfixer 만 씨앗을 줬을 때 원자 하나가 여전히 1.74 A 옮겨졌다.
    """
    source = (ROOT / "scripts/stage6_bioemu.py").read_text()
    for pinned, why in (
        ("integrator.setRandomNumberSeed(RECONSTRUCTION_SEED)",
         "본 최소화의 적분기"),
        ("300 * unit.kelvin, RECONSTRUCTION_SEED + attempt",
         "재시도 때 데우는 속도"),
        ("fixer.addMissingAtoms(seed=RECONSTRUCTION_SEED)",
         "pdbfixer 가 새 원자를 다듬는 적분기"),
        ("random.seed(RECONSTRUCTION_SEED)",
         "Modeller 이 수소를 놓는 난수 위치"),
        ('fixer.platform.setPropertyDefaultValue("Threads", "1")',
         "pdbfixer 최소화의 축약 순서"),
        ('"CUDA": {"DeterministicForces": "true"}',
         "본 최소화의 축약 순서"),
    ):
        assert pinned in source, f"{why} 가 고정되어 있지 않습니다"
