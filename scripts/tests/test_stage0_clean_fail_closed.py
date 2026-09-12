"""Regression tests for Stage 0 AlphaFold cleaning quality gates."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def run_clean(tmp_path: Path, extra: list[str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage0_clean_alphafold.py"),
            "--af-dir", str(tmp_path / "af"),
            "--out-dir", str(tmp_path / "clean"),
            "--workers", "1",
            *(extra or []),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_clean_fails_when_no_alphafold_inputs_exist(tmp_path: Path) -> None:
    (tmp_path / "af").mkdir()

    res = run_clean(tmp_path)

    assert res.returncode != 0
    assert "No AlphaFold PDB inputs were found" in res.stderr


def test_clean_fails_when_input_has_no_parseable_residues(tmp_path: Path) -> None:
    af = tmp_path / "af"
    af.mkdir()
    (af / "AF-P12345-F1-model_v4.pdb").write_text("HEADER empty\nEND\n")
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "P12345_clean.pdb").write_text("STALE\n")
    (clean / "P12345_softmask.json").write_text('{"stale": true}\n')

    res = run_clean(tmp_path)

    assert res.returncode != 0
    assert "0/1 unique cleaned receptors succeeded" in res.stderr
    assert not (tmp_path / "clean" / "P12345_clean.pdb").exists()
    assert not (tmp_path / "clean" / "P12345_softmask.json").exists()


def test_clean_fails_when_min_success_count_is_not_met(tmp_path: Path) -> None:
    af = tmp_path / "af"
    af.mkdir()
    (af / "AF-P12345-F1-model_v4.pdb").write_text(
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 90.00           N\n"
    )
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "P12345_clean.pdb").write_text("STALE\n")
    (clean / "P12345_softmask.json").write_text('{"stale": true}\n')

    res = run_clean(tmp_path, ["--min-success-count", "2"])

    assert res.returncode != 0
    assert "1/1 unique cleaned receptors succeeded; required at least 2" in res.stderr
    assert not (tmp_path / "clean" / "P12345_clean.pdb").exists()
    assert not (tmp_path / "clean" / "P12345_softmask.json").exists()


def test_clean_uses_one_representative_fragment_per_uniprot(tmp_path: Path) -> None:
    af = tmp_path / "af"
    af.mkdir()
    (af / "AF-P12345-F1-model_v4.pdb").write_text(
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 90.00           N\n"
    )
    (af / "AF-P12345-F2-model_v4.pdb").write_text(
        "ATOM      1  N   GLY A   1       1.000   1.000   1.000  1.00 90.00           N\n"
    )

    res = run_clean(tmp_path, ["--min-success-count", "2"])

    assert res.returncode != 0
    assert "1/1 unique cleaned receptors succeeded; required at least 2" in res.stderr


def test_clean_succeeds_with_parseable_residue(tmp_path: Path) -> None:
    af = tmp_path / "af"
    af.mkdir()
    (af / "AF-P12345-F1-model_v4.pdb").write_text(
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 90.00           N\n"
    )

    res = run_clean(tmp_path)

    assert res.returncode == 0, res.stderr
    assert (tmp_path / "clean" / "P12345_clean.pdb").exists()
    assert (tmp_path / "clean" / "P12345_softmask.json").exists()
