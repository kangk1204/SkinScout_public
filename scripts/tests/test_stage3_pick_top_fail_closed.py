"""Regression tests for Stage 3 top-pick output handling."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def run_pick_top(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage3_pick_top.py"),
            *args,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def write_autodock(path: Path, rows: str = "P0\t3.0\n") -> None:
    path.write_text("target_id\tneg_vina_score\n" + rows)


def test_pick_top_removes_stale_output_when_scores_missing(tmp_path: Path) -> None:
    out = tmp_path / "top.csv"
    out.write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage3_pick_top.py"),
            "--autodock-scores",
            str(tmp_path / "missing_autodock.tsv"),
            "--blind-scores",
            str(tmp_path / "missing_blind.tsv"),
            "--out-csv",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "autodock score file is required and must be non-empty" in res.stderr
    assert not out.exists()


def test_pick_top_removes_stale_output_when_scores_empty(tmp_path: Path) -> None:
    out = tmp_path / "top.csv"
    out.write_text("stale\n")
    autodock = tmp_path / "autodock.tsv"
    blind = tmp_path / "blind.tsv"
    autodock.write_text("")
    blind.write_text("")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage3_pick_top.py"),
            "--autodock-scores",
            str(autodock),
            "--blind-scores",
            str(blind),
            "--out-csv",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "autodock score file is required and must be non-empty" in res.stderr
    assert not out.exists()


def test_pick_top_rejects_score_file_missing_required_columns(tmp_path: Path) -> None:
    out = tmp_path / "top.csv"
    out.write_text("stale\n")
    autodock = tmp_path / "autodock.tsv"
    autodock.write_text("target_id\tother\nP1\t1.0\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage3_pick_top.py"),
            "--autodock-scores",
            str(autodock),
            "--blind-scores",
            str(tmp_path / "missing_blind.tsv"),
            "--out-csv",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "autodock score file missing required columns: neg_vina_score" in res.stderr
    assert not out.exists()


def test_pick_top_rejects_partially_invalid_autodock_scores(tmp_path: Path) -> None:
    out = tmp_path / "top.csv"
    out.write_text("stale\n")
    autodock = tmp_path / "autodock.tsv"
    autodock.write_text("target_id\tneg_vina_score\nP1\t3.0\nP2\tnot-a-number\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage3_pick_top.py"),
            "--autodock-scores",
            str(autodock),
            "--blind-scores",
            str(tmp_path / "missing_blind.tsv"),
            "--out-csv",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "autodock score file column 'neg_vina_score' contains invalid values" in res.stderr
    assert not out.exists()


def test_pick_top_rejects_boolean_autodock_scores(tmp_path: Path) -> None:
    out = tmp_path / "top.csv"
    out.write_text("stale\n")
    autodock = tmp_path / "autodock.tsv"
    autodock.write_text("target_id\tneg_vina_score\nP1\tTrue\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage3_pick_top.py"),
            "--autodock-scores",
            str(autodock),
            "--blind-scores",
            str(tmp_path / "missing_blind.tsv"),
            "--out-csv",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "autodock score file column 'neg_vina_score' contains invalid values" in res.stderr
    assert not out.exists()


def test_pick_top_rejects_blank_blind_target_ids(tmp_path: Path) -> None:
    out = tmp_path / "top.csv"
    out.write_text("stale\n")
    autodock = tmp_path / "autodock.tsv"
    write_autodock(autodock)
    blind = tmp_path / "blind.tsv"
    blind.write_text("target_id\tscore\nP1\t2.5\n \t2.0\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage3_pick_top.py"),
            "--autodock-scores",
            str(autodock),
            "--blind-scores",
            str(blind),
            "--out-csv",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "diffdock_blind score file column 'target_id' contains blank values" in res.stderr
    assert not out.exists()


def test_pick_top_rejects_duplicate_target_ids_without_output(tmp_path: Path) -> None:
    out = tmp_path / "top.csv"
    out.write_text("stale\n")
    autodock = tmp_path / "autodock.tsv"
    autodock.write_text("target_id\tneg_vina_score\nP1\t3.0\nP1\t2.0\n")

    res = run_pick_top(
        [
            "--autodock-scores",
            str(autodock),
            "--blind-scores",
            str(tmp_path / "missing_blind.tsv"),
            "--out-csv",
            str(out),
        ],
    )

    assert res.returncode != 0
    assert (
        "Top-pick score files contain duplicate target_id values: P1"
        in res.stderr
    )
    assert not out.exists()


def test_pick_top_rejects_cross_source_duplicate_target_ids_without_output(
    tmp_path: Path,
) -> None:
    out = tmp_path / "top.csv"
    out.write_text("stale\n")
    autodock = tmp_path / "autodock.tsv"
    blind = tmp_path / "blind.tsv"
    autodock.write_text("target_id\tneg_vina_score\nP1\t3.0\n")
    blind.write_text("target_id\tscore\nP1\t2.5\n")

    res = run_pick_top(
        [
            "--autodock-scores",
            str(autodock),
            "--blind-scores",
            str(blind),
            "--out-csv",
            str(out),
        ],
    )

    assert res.returncode != 0
    assert (
        "Top-pick score files contain duplicate target_id values: P1"
        in res.stderr
    )
    assert not out.exists()


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("0", "--top-pct must be a finite value in (0, 1]: 0"),
        ("-0.1", "--top-pct must be a finite value in (0, 1]: -0.1"),
        ("nan", "--top-pct must be a finite value in (0, 1]: nan"),
        ("1.1", "--top-pct must be a finite value in (0, 1]: 1.1"),
    ],
)
def test_pick_top_rejects_invalid_top_pct_without_output(
    tmp_path: Path,
    value: str,
    message: str,
) -> None:
    out = tmp_path / "top.csv"
    out.write_text("stale\n")

    res = run_pick_top(
        [
            "--autodock-scores",
            str(tmp_path / "missing_autodock.tsv"),
            "--blind-scores",
            str(tmp_path / "missing_blind.tsv"),
            "--top-pct",
            value,
            "--out-csv",
            str(out),
        ],
    )

    assert res.returncode != 0
    assert message in res.stderr
    assert not out.exists()


def test_pick_top_accepts_blind_score_column_alias(tmp_path: Path) -> None:
    out = tmp_path / "top.csv"
    autodock = tmp_path / "autodock.tsv"
    write_autodock(autodock)
    blind = tmp_path / "blind.tsv"
    blind.write_text("target_id\tscore\nP1\t2.5\n")

    res = subprocess.run(
        [
                sys.executable,
                str(ROOT / "scripts/stage3_pick_top.py"),
                "--autodock-scores",
                str(autodock),
                "--blind-scores",
                str(blind),
            "--out-csv",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode == 0, res.stderr
    assert out.read_text().startswith("target_id,score,raw_score,score_direction,source\n")
    assert "P1,1.0,2.5,higher_is_better,diffdock_blind" in out.read_text()


def test_pick_top_preserves_equal_raw_scores_as_equal_ranks(tmp_path: Path) -> None:
    out = tmp_path / "top.csv"
    autodock = tmp_path / "autodock.tsv"
    autodock.write_text("target_id\tneg_vina_score\nP1\t3.0\nP2\t3.0\n")

    res = run_pick_top(
        [
            "--autodock-scores",
            str(autodock),
            "--blind-scores",
            str(tmp_path / "missing_blind.tsv"),
            "--out-csv",
            str(out),
        ],
    )

    assert res.returncode == 0, res.stderr
    rows = out.read_text().splitlines()
    assert "P1,1.0,3.0,higher_is_better,autodock" in rows
    assert "P2,1.0,3.0,higher_is_better,autodock" in rows
