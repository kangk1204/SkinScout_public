"""Regression tests for the AutoDock-GPU local-search flag and failure paths.

AutoDock-GPU v1.6 rejects an unknown ``-lsmet`` token during job setup but
still exits 0, so an unvalidated value produced zero docked poses on every
receptor with no warning at all.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import stage3_autodock_run as adr  # noqa: E402


@pytest.mark.parametrize("token", adr.AUTODOCK_LS_METHODS)
def test_binary_accepted_tokens_pass_through_unchanged(token: str) -> None:
    assert adr.canonical_ls_method(token) == token


def test_the_configured_method_name_maps_to_the_token_the_binary_accepts() -> None:
    assert adr.canonical_ls_method("adadelta") == "ad"
    assert adr.canonical_ls_method("ADADELTA") == "ad"


def test_an_unknown_method_is_rejected_instead_of_silently_docking_nothing() -> None:
    with pytest.raises(SystemExit) as excinfo:
        adr.canonical_ls_method("gradient-descent")
    assert "gradient-descent" in str(excinfo.value)


def test_the_shipped_default_is_a_token_the_binary_accepts() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ls-method", default="ad")
    default = parser.parse_args([]).ls_method
    assert adr.canonical_ls_method(default) == default
    assert default in adr.AUTODOCK_LS_METHODS


def _stub_run(tmp_path: Path, monkeypatch, stdout: str, returncode: int = 0):
    ligand = tmp_path / "ligand.pdbqt"
    ligand.write_text("ATOM\n")
    maps = tmp_path / "T.maps.fld"
    maps.write_text("field\n")
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, returncode, stdout, "")

    monkeypatch.setattr(adr.subprocess, "run", fake_run)
    return ligand, maps, calls


def test_a_setup_failure_reported_with_exit_code_zero_is_still_a_failure(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    ligand, maps, _ = _stub_run(
        tmp_path, monkeypatch, "Error in setup of Job #1\nThe job was not successful.\n"
    )
    with caplog.at_level("WARNING"):
        result = adr.run_autodock_gpu_one(
            ligand, maps, 4, "ad", tmp_path / "work", "autodock_gpu_128wi"
        )
    assert result is None
    assert "AutoDock-GPU failed" in caplog.text


def test_a_reported_success_without_a_readable_energy_is_warned_about(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    ligand, maps, _ = _stub_run(
        tmp_path, monkeypatch, "All jobs (1) ran without errors.\n"
    )
    with caplog.at_level("WARNING"):
        result = adr.run_autodock_gpu_one(
            ligand, maps, 4, "ad", tmp_path / "work", "autodock_gpu_128wi"
        )
    assert result is None
    assert "no binding energy" in caplog.text


def test_the_configured_alias_reaches_the_binary_as_the_accepted_token(
    tmp_path: Path, monkeypatch
) -> None:
    ligand, maps, calls = _stub_run(
        tmp_path, monkeypatch, "All jobs (1) ran without errors.\n"
    )
    adr.run_autodock_gpu_one(
        ligand, maps, 4, "adadelta", tmp_path / "work", "autodock_gpu_128wi"
    )
    assert calls, "the binary was never invoked"
    cmd = calls[0]
    assert cmd[cmd.index("-lsmet") + 1] == "ad"
