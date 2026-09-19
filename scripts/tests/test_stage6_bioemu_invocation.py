"""Regression tests for how Stage 6 invokes BioEmu.

``run_bioemu`` used to look for a ``bioemu`` executable on PATH and call it as
``bioemu sample --receptor <pdb> --num-conformers <n> --out <dir>``. Both
halves were wrong against the real package:

* BioEmu ships no console script at all, so ``shutil.which("bioemu")`` was
  always None and the stage exited with "bioemu is not available on PATH"
  however carefully the environment had been built.
* ``bioemu.sample`` takes three positional arguments - sequence, sample count,
  output directory - and none of those flags exist. BioEmu samples from a
  sequence, not from a receptor structure.

Neither mistake could surface from reading the code, and nothing exercised the
command shape, so Stage 6 has never run. These tests pin the shape against the
documented interface so a future change has to stay honest about it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import stage6_bioemu as stage6  # noqa: E402


def test_bioemu_is_run_as_a_module_not_a_console_script() -> None:
    """The package installs no ``bioemu`` binary."""
    cmd = stage6.bioemu_command("MKVLA", Path("/out"), 20)
    assert cmd[0] == sys.executable
    assert cmd[1:3] == ["-m", "bioemu.sample"]
    assert "bioemu" not in cmd[3:], "실행파일 이름을 인자로 넘기면 안 됩니다"


def test_the_three_arguments_are_positional_and_in_order() -> None:
    cmd = stage6.bioemu_command("MKVLA", Path("/out/P00918"), 20)
    assert cmd[3:] == ["MKVLA", "20", "/out/P00918"]


@pytest.mark.parametrize("flag", ["--receptor", "--num-conformers", "--out"])
def test_the_old_flags_are_gone(flag: str) -> None:
    """These flags do not exist in ``bioemu.sample`` and never did."""
    assert flag not in stage6.bioemu_command("MKVLA", Path("/out"), 20)


def test_a_receptor_with_no_sequence_is_refused_rather_than_sampled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ligand-only or empty PDB has nothing for BioEmu to sample."""
    receptor = tmp_path / "P00918_input.pdb"
    receptor.write_text("HETATM    1  C1  LIG A   1       0.0   0.0   0.0\n")
    monkeypatch.setattr(stage6, "bioemu_available", lambda: True)

    def fail(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("서열이 없는데 BioEmu 를 부르면 안 됩니다")

    monkeypatch.setattr(stage6.subprocess, "run", fail)
    assert stage6.run_bioemu(receptor, tmp_path / "out", 20) is False


def test_a_missing_receptor_is_refused(tmp_path: Path) -> None:
    assert stage6.run_bioemu(tmp_path / "absent.pdb", tmp_path / "out", 20) is False


def test_availability_is_decided_by_import_not_by_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PATH says nothing about whether this interpreter can import BioEmu."""
    monkeypatch.setattr(stage6.importlib.util, "find_spec", lambda name: None)
    assert stage6.bioemu_available() is False
    monkeypatch.setattr(stage6.importlib.util, "find_spec", lambda name: object())
    assert stage6.bioemu_available() is True
