"""Regression tests for DiffDock blind no-pocket fail-closed behavior."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def run_diffdock(tmp_path: Path, no_pocket: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PATH"] = str(tmp_path / "empty_path")
    (tmp_path / "empty_path").mkdir(exist_ok=True)
    (tmp_path / "diffdock.tsv").write_text("stale\n")
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage3_diffdock_blind.py"),
            "--ligand", str(tmp_path / "ligand.pdbqt"),
            "--no-pocket-list", str(no_pocket),
            "--clean-dir", str(tmp_path / "clean"),
            "--out-scores", str(tmp_path / "diffdock.tsv"),
        ],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


def test_missing_no_pocket_list_fails_without_output(tmp_path: Path) -> None:
    res = run_diffdock(tmp_path, tmp_path / "missing.list")

    assert res.returncode != 0
    assert "No-pocket target list is required" in res.stderr
    assert not (tmp_path / "diffdock.tsv").exists()


def test_empty_no_pocket_list_writes_header_only(tmp_path: Path) -> None:
    no_pocket = tmp_path / "no_pocket.list"
    no_pocket.write_text("")

    res = run_diffdock(tmp_path, no_pocket)

    assert res.returncode == 0, res.stderr
    assert (tmp_path / "diffdock.tsv").read_text() == (
        "target_id\tdiffdock_confidence\tscore\tneg_vina_score\n"
    )


def test_nonempty_no_pocket_list_requires_successful_diffdock_score(tmp_path: Path) -> None:
    no_pocket = tmp_path / "no_pocket.list"
    no_pocket.write_text("P1\n")
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "P1_clean.pdb").write_text("ATOM      1  CA  ALA A   1       0.0 0.0 0.0\nEND\n")

    res = run_diffdock(tmp_path, no_pocket)

    assert res.returncode != 0
    assert "produced no blind docking scores" in res.stderr
    assert not (tmp_path / "diffdock.tsv").exists()
