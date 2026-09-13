"""Regression tests for Boltz-2 per-target failure reporting.

A capped comprehensive run scored 44 of 50 targets and left no record of what
happened to the other six - every failure path returned None in silence. Three
of those six were the top three of the consensus built from the other scorers,
so the gap was not in the tail.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import stage3_boltz2_affinity as boltz  # noqa: E402


@pytest.fixture
def inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    receptor = tmp_path / "P1_clean.pdb"
    receptor.write_text("ATOM\n")
    ligand = tmp_path / "ligand.sdf"
    ligand.write_text("mol\n")
    return receptor, ligand, tmp_path / "work"


def test_a_missing_boltz_command_is_reported(inputs, monkeypatch, caplog) -> None:
    receptor, ligand, work = inputs
    monkeypatch.setattr(boltz.shutil, "which", lambda name: None)

    with caplog.at_level("WARNING"):
        assert boltz.call_boltz(receptor, ligand, work, 400, 12.0) is None
    assert "not on PATH" in caplog.text


def test_a_nonzero_exit_reports_the_target_and_the_detail(
    inputs, monkeypatch, caplog
) -> None:
    receptor, ligand, work = inputs
    monkeypatch.setattr(boltz.shutil, "which", lambda name: "/usr/bin/boltz")
    monkeypatch.setattr(boltz, "write_affinity_yaml", lambda *a, **k: None)
    monkeypatch.setattr(
        boltz,
        "run_boltz_predict",
        lambda *a, **k: subprocess.CompletedProcess(
            ["boltz"], 1, "", "FileNotFoundError: pre_affinity_input.npz"
        ),
    )

    with caplog.at_level("WARNING"):
        assert boltz.call_boltz(receptor, ligand, work, 400, 12.0) is None
    assert "P1_clean.pdb" in caplog.text
    assert "pre_affinity_input.npz" in caplog.text


def test_a_reported_success_without_a_payload_is_reported(
    inputs, monkeypatch, caplog
) -> None:
    receptor, ligand, work = inputs
    monkeypatch.setattr(boltz.shutil, "which", lambda name: "/usr/bin/boltz")
    monkeypatch.setattr(boltz, "write_affinity_yaml", lambda *a, **k: None)
    monkeypatch.setattr(
        boltz,
        "run_boltz_predict",
        lambda *a, **k: subprocess.CompletedProcess(["boltz"], 0, "", ""),
    )
    monkeypatch.setattr(boltz, "load_affinity_payload", lambda *a, **k: None)

    with caplog.at_level("WARNING"):
        assert boltz.call_boltz(receptor, ligand, work, 400, 12.0) is None
    assert "no affinity payload" in caplog.text


def test_an_unbuildable_input_is_reported(inputs, monkeypatch, caplog) -> None:
    receptor, ligand, work = inputs
    monkeypatch.setattr(boltz.shutil, "which", lambda name: "/usr/bin/boltz")

    def _raise(*args, **kwargs):
        raise ValueError("no pocket residues within crop radius")

    monkeypatch.setattr(boltz, "write_affinity_yaml", _raise)

    with caplog.at_level("WARNING"):
        assert boltz.call_boltz(receptor, ligand, work, 400, 12.0) is None
    assert "could not build input YAML" in caplog.text
    assert "crop radius" in caplog.text


def test_a_missing_receptor_is_reported(tmp_path: Path, caplog) -> None:
    ligand = tmp_path / "ligand.sdf"
    ligand.write_text("mol\n")

    with caplog.at_level("WARNING"):
        assert (
            boltz.call_boltz(tmp_path / "absent.pdb", ligand, tmp_path, 400, 12.0)
            is None
        )
    assert "missing/empty" in caplog.text
