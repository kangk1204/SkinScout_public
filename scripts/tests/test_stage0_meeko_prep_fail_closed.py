"""Regression tests for Stage 0 receptor PDBQT preparation gates."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from stage0_meeko_prep import best_pocket  # noqa: E402


def write_receptor_inputs(root: Path, *targets: str) -> tuple[Path, Path, Path]:
    clean_dir = root / "clean"
    pocket_dir = root / "pockets"
    clean_dir.mkdir()
    pocket_dir.mkdir()
    for target in targets:
        # Four non-collinear atoms so the pocket has a measurable extent; the
        # box is now derived from these rather than from a constant radius.
        (clean_dir / f"{target}_clean.pdb").write_text(
            "".join(
                f"ATOM  {serial:>5}  CA  ALA A{serial:>4}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 90.00           C\n"
                for serial, (x, y, z) in enumerate(
                    [(-5.0, -5.0, -5.0), (5.0, 5.0, 5.0), (-5.0, 5.0, -5.0), (5.0, -5.0, 5.0)],
                    start=1,
                )
            )
        )
        (pocket_dir / f"{target}_clean.pdb_predictions.csv").write_text(
            "name     ,  rank,   score, probability, sas_points, surf_atoms,"
            "   center_x,   center_y,   center_z, residue_ids, surf_atom_ids\n"
            "pocket1  ,     1,   23.53,       0.841,        127,          4,"
            "     0.0,     0.0,    0.0, A_1, 1 2 3 4\n"
        )
        (pocket_dir / f"{target}.pockets.json").write_text(
            json.dumps({
                "pockets": [{
                    "center": [0.0, 0.0, 0.0],
                    "radius": 10.0,
                    "druggability": 0.8,
                }]
            })
        )
    no_pocket = root / "no_pocket_targets.list"
    no_pocket.write_text("")
    return clean_dir, pocket_dir, no_pocket


def run_meeko_prep(
    tmp_path: Path,
    extra: list[str] | None = None,
    stale_outputs: bool = False,
) -> subprocess.CompletedProcess[str]:
    clean_dir, pocket_dir, no_pocket = write_receptor_inputs(tmp_path, "P1")
    if stale_outputs:
        (tmp_path / "pdbqt").mkdir()
        (tmp_path / "boxes").mkdir()
        (tmp_path / "pdbqt/P1.pdbqt").write_text("STALE\n")
        (tmp_path / "boxes/P1.box.txt").write_text("STALE\n")
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage0_meeko_prep.py"),
            "--clean-dir", str(clean_dir),
            "--pocket-dir", str(pocket_dir),
            "--no-pocket-list", str(no_pocket),
            "--out-pdbqt-dir", str(tmp_path / "pdbqt"),
            "--out-box-dir", str(tmp_path / "boxes"),
            "--workers", "1",
            *(extra or []),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": str(tmp_path / "empty-path")},
        check=False,
    )


def test_meeko_prep_fails_when_no_receptor_jobs_are_queued(tmp_path: Path) -> None:
    clean_dir = tmp_path / "clean"
    pocket_dir = tmp_path / "pockets"
    clean_dir.mkdir()
    pocket_dir.mkdir()
    no_pocket = tmp_path / "no_pocket_targets.list"
    no_pocket.write_text("")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage0_meeko_prep.py"),
            "--clean-dir", str(clean_dir),
            "--pocket-dir", str(pocket_dir),
            "--no-pocket-list", str(no_pocket),
            "--out-pdbqt-dir", str(tmp_path / "pdbqt"),
            "--out-box-dir", str(tmp_path / "boxes"),
            "--workers", "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "No receptor PDBQT jobs were queued" in res.stderr


def test_best_pocket_rejects_pocket_without_score(tmp_path: Path) -> None:
    pocket_json = tmp_path / "P1.pockets.json"
    pocket_json.write_text(
        json.dumps({"pockets": [{"center": [0.0, 0.0, 0.0], "radius": 10.0}]})
    )

    with pytest.raises(SystemExit, match="missing required score field"):
        best_pocket(pocket_json)


def test_best_pocket_rejects_nonfinite_center(tmp_path: Path) -> None:
    pocket_json = tmp_path / "P1.pockets.json"
    pocket_json.write_text(
        json.dumps({
            "pockets": [{
                "center": ["nan", 0.0, 0.0],
                "radius": 10.0,
                "druggability": 0.8,
            }]
        })
    )

    with pytest.raises(SystemExit, match="field 'center_x' must be finite"):
        best_pocket(pocket_json)


def test_best_pocket_rejects_nonpositive_radius(tmp_path: Path) -> None:
    pocket_json = tmp_path / "P1.pockets.json"
    pocket_json.write_text(
        json.dumps({
            "pockets": [{
                "center": [0.0, 0.0, 0.0],
                "radius": 0.0,
                "druggability": 0.8,
            }]
        })
    )

    with pytest.raises(SystemExit, match="field 'radius' must be positive"):
        best_pocket(pocket_json)


def test_meeko_prep_fails_when_all_receptors_fail(tmp_path: Path) -> None:
    res = run_meeko_prep(tmp_path, stale_outputs=True)

    assert res.returncode != 0
    assert "did not meet quality gate: 0/1 succeeded" in res.stderr
    assert not list((tmp_path / "pdbqt").glob("*.pdbqt"))
    assert not list((tmp_path / "boxes").glob("*.box.txt"))


def test_meeko_prep_success_gate_can_be_relaxed_for_diagnostics(tmp_path: Path) -> None:
    res = run_meeko_prep(
        tmp_path,
        ["--min-success-count", "0", "--min-success-fraction", "0.0"],
    )

    assert res.returncode == 0, res.stderr


def test_write_box_derives_the_edge_from_pocket_surface_atoms(tmp_path: Path) -> None:
    """The emitted box must reflect pocket geometry, not a constant radius."""
    from stage0_meeko_prep import write_box

    clean_dir, pocket_dir, _ = write_receptor_inputs(tmp_path, "P1")
    box_path = tmp_path / "P1.box.txt"

    write_box(
        box_path,
        {"center": (0.0, 0.0, 0.0), "radius": 12.0},
        prediction=pocket_dir / "P1_clean.pdb_predictions.csv",
        clean_pdb=clean_dir / "P1_clean.pdb",
    )

    fields = dict(
        line.split(" = ") for line in box_path.read_text().splitlines() if " = " in line
    )
    # surface atoms reach 5 A from the centre, plus the 4 A default headroom
    assert float(fields["size_x"]) == pytest.approx(18.0)
    assert float(fields["size_y"]) == pytest.approx(18.0)
    assert float(fields["size_z"]) == pytest.approx(18.0)
    # the constant-radius rule would have written the 30 A cap regardless
    assert float(fields["size_x"]) != pytest.approx(30.0)


def test_write_box_refuses_to_guess_when_geometry_is_missing(tmp_path: Path) -> None:
    from stage0_meeko_prep import write_box

    with pytest.raises(SystemExit, match="requires the P2Rank prediction"):
        write_box(tmp_path / "P1.box.txt", {"center": (0.0, 0.0, 0.0), "radius": 12.0})

    assert not (tmp_path / "P1.box.txt").exists()


def test_write_box_legacy_cube_is_an_explicit_opt_in(tmp_path: Path) -> None:
    from stage0_meeko_prep import write_box

    box_path = tmp_path / "P1.box.txt"
    write_box(
        box_path,
        {"center": (1.0, 2.0, 3.0), "radius": 12.0},
        legacy_cube_edge=30.0,
    )

    text = box_path.read_text()
    assert "center_x = 1.000" in text
    assert "size_x = 30.000" in text

    with pytest.raises(SystemExit, match="must be positive"):
        write_box(box_path, {"center": (0.0, 0.0, 0.0)}, legacy_cube_edge=0.0)


def test_jobs_skip_targets_without_a_p2rank_prediction(tmp_path: Path) -> None:
    """A target with no measurable extent must not receive a default box."""
    from stage0_meeko_prep import build_jobs

    clean_dir, pocket_dir, _ = write_receptor_inputs(tmp_path, "P1", "P2")
    (pocket_dir / "P2_clean.pdb_predictions.csv").unlink()

    jobs = build_jobs(clean_dir, pocket_dir, tmp_path / "pdbqt", tmp_path / "boxes", set())

    assert [job.uniprot for job in jobs] == ["P1"]


def _pdbqt(charges: list[float]) -> str:
    """전하만 다르게 준 최소 PDBQT. 71-76 열이 전하다."""
    lines = []
    for i, q in enumerate(charges, start=1):
        lines.append(
            f"ATOM  {i:5d}  CA  ALA A{i:4d}    "
            f"{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00    {q:+6.3f} C "
        )
    return "\n".join(lines) + "\n"


def test_a_pdbqt_without_partial_charges_is_not_accepted(tmp_path: Path) -> None:
    """전하가 0 인 PDBQT 는 파일로는 멀쩡하지만 autogrid4 가 거부한다.

    실측: 15,038 개 중 1,699 개(11.3%)가 이렇게 만들어졌고, 그 표적들은 맵을
    못 얻어 도킹 자체가 되지 않았다 - 순위표에 등장할 기회가 없었다.
    obabel 은 이 파일을 쓰고도 종료코드 0 을 준다.
    """
    from stage0_meeko_prep import (
        MIN_NONZERO_CHARGE_FRACTION,
        nonzero_charge_fraction,
    )

    empty = tmp_path / "empty.pdbqt"
    empty.write_text(_pdbqt([0.0] * 20))
    assert nonzero_charge_fraction(empty) == 0.0
    assert nonzero_charge_fraction(empty) < MIN_NONZERO_CHARGE_FRACTION

    # 제대로 만든 파일은 실측으로 95~99% 가 0 이 아니다.
    charged = tmp_path / "charged.pdbqt"
    charged.write_text(_pdbqt([0.242, -0.273, 0.0, 0.181] * 5))
    assert nonzero_charge_fraction(charged) == pytest.approx(0.75)
    assert nonzero_charge_fraction(charged) >= MIN_NONZERO_CHARGE_FRACTION


def test_meeko_retries_with_allow_bad_res_before_giving_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """엄격 모드가 거절해도 OpenBabel 로 넘어가면 안 된다.

    obabel 수용체는 극성 수소가 없다 - 실측(A0A087WXS9): meeko HD 553 / N 414 /
    NA 8 vs obabel HD 0 / N 11 / NA 414. HD 가 없으면 AutoDock 의 수소결합 항이
    죽고, 그 수용체를 meeko 수용체 13,339 개와 같은 순위표에 넣으면 점수가
    체계적으로 얕게 나온다. `-a` 로 다시 시도하면 타이핑이 같은 수용체가 나온다.
    """
    import stage0_meeko_prep as prep
    from types import SimpleNamespace

    out = tmp_path / "out.pdbqt"
    job = SimpleNamespace(
        uniprot="P00001",
        clean_pdb=tmp_path / "in.pdb",
        out_pdbqt=out,
        out_box=tmp_path / "out.box.txt",
        pocket={"center": [0.0, 0.0, 0.0], "size": [20.0, 20.0, 20.0]},
        pocket_prediction=None,
    )
    job.clean_pdb.write_text("ATOM\n")
    seen: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        seen.append(list(cmd))
        if "-a" not in cmd:                      # 엄격 모드는 거절한다
            return SimpleNamespace(returncode=1, stdout="", stderr="missing atoms")
        out.write_text(_pdbqt([0.24, -0.27, 0.18, 0.0] * 5))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(prep.shutil, "which", lambda _: "/bin/mk_prepare_receptor.py")
    monkeypatch.setattr(prep.subprocess, "run", fake_run)
    monkeypatch.setattr(prep, "write_box", lambda *a, **k: job.out_box.write_text("box\n"))

    uniprot, ok, detail = prep.run_meeko(job)
    assert (uniprot, ok) == ("P00001", True)
    assert detail == "meeko_allow_bad_res"
    assert len(seen) == 2, "엄격 → -a 순으로 두 번 시도해야 합니다"
    assert "-a" not in seen[0] and "-a" in seen[1]
    assert all("obabel" not in " ".join(c) for c in seen), (
        "meeko 로 살릴 수 있는 수용체를 OpenBabel 로 만들었습니다"
    )


def test_obabel_fallback_is_off_unless_explicitly_asked_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """두 번 다 실패하면 실패로 둔다. 비교할 수 없는 수용체를 만들지 않는다."""
    import stage0_meeko_prep as prep
    from types import SimpleNamespace

    job = SimpleNamespace(
        uniprot="P00002",
        clean_pdb=tmp_path / "in.pdb",
        out_pdbqt=tmp_path / "out.pdbqt",
        out_box=tmp_path / "out.box.txt",
        pocket={"center": [0.0, 0.0, 0.0], "size": [20.0, 20.0, 20.0]},
        pocket_prediction=None,
    )
    job.clean_pdb.write_text("ATOM\n")
    monkeypatch.setattr(prep.shutil, "which", lambda _: "/bin/mk_prepare_receptor.py")
    monkeypatch.setattr(prep.subprocess, "run",
                        lambda *_a, **_k: SimpleNamespace(returncode=1, stdout="", stderr="no"))
    monkeypatch.setattr(prep, "ALLOW_OBABEL_FALLBACK", False)

    uniprot, ok, detail = prep.run_meeko(job)
    assert (uniprot, ok) == ("P00002", False)
    assert "-a 재시도도 실패" in detail
    assert not job.out_pdbqt.exists(), "실패한 수용체 파일을 남기면 검증기가 못 잡습니다"


def test_restrict_leaves_the_other_receptors_alone(tmp_path: Path) -> None:
    """일부만 다시 만들려는 재실행이 나머지 13,339 개를 지우면 안 된다."""
    source = (ROOT / "scripts/stage0_meeko_prep.py").read_text()
    assert '"--restrict"' in source
    # 삭제 루프가 restrict 로 걸러진 뒤에 와야 한다.
    cut = source.index("jobs = [job for job in jobs if job.uniprot in wanted]")
    purge = source.index("remove_outputs(job.out_pdbqt, job.out_box)", cut)
    assert purge > cut, "삭제가 restrict 필터보다 먼저면 나머지가 날아갑니다"


def test_obabel_fallback_asks_for_partial_charges(tmp_path: Path) -> None:
    """`--partialcharge gasteiger` 가 없으면 전하가 전부 0 으로 나온다."""
    source = (ROOT / "scripts/stage0_meeko_prep.py").read_text()
    assert '"--partialcharge", "gasteiger"' in source, (
        "obabel 폴백이 전하 모델을 지정하지 않으면 autogrid4 가 결과를 거부합니다"
    )


def test_obabel_fallback_checks_the_file_it_wrote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """종료코드 0 을 믿으면 안 된다. 전하 없는 파일도 0 으로 끝난다."""
    import stage0_meeko_prep as prep
    from types import SimpleNamespace

    out = tmp_path / "out.pdbqt"
    job = SimpleNamespace(
        uniprot="P00001",
        clean_pdb=tmp_path / "in.pdb",
        out_pdbqt=out,
        out_box=tmp_path / "out.box.txt",
        pocket={"center": [0.0, 0.0, 0.0], "size": [20.0, 20.0, 20.0]},
        pocket_prediction=None,
    )
    job.clean_pdb.write_text("ATOM\n")

    def fake_run(_cmd, **_kwargs):
        out.write_text(_pdbqt([0.0] * 12))   # obabel 이 전하 없이 쓴 상황
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(prep.shutil, "which", lambda _: "/bin/obabel")
    monkeypatch.setattr(prep.subprocess, "run", fake_run)
    uniprot, ok, detail = prep._obabel_fallback(job)
    assert uniprot == "P00001"
    assert ok is False, "전하가 0 인 파일을 성공으로 처리했습니다"
    assert "부분전하" in detail
    assert not job.out_box.exists(), "실패한 수용체의 박스를 써 두면 안 됩니다"
