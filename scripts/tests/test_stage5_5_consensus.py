"""Unit tests for PLIP/ProLIF pose-supported interaction atom consensus."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage5_5_consensus import (  # noqa: E402
    confirm_atoms,
    consensus_for_all,
    targets_with_all_sources,
)


def test_two_of_two_passes() -> None:
    votes = {
        "plip": {1, 2, 3},
        "prolif": {1, 2},
    }

    assert confirm_atoms(votes, min_votes=2) == {1, 2}


def test_one_of_two_dropped_at_default_threshold() -> None:
    votes = {
        "plip": {7},
        "prolif": set(),
    }

    assert confirm_atoms(votes) == set()


def test_min_votes_one_relaxes_threshold_for_explicit_diagnostics() -> None:
    votes = {"plip": {1}, "prolif": set()}

    assert confirm_atoms(votes, min_votes=1) == {1}


def test_empty_inputs_yield_empty_consensus() -> None:
    votes = {"plip": set(), "prolif": set()}

    assert confirm_atoms(votes, min_votes=1) == set()


def test_consensus_for_all_combines_per_target_and_labels_evidence() -> None:
    payload = consensus_for_all(
        plip={"T1": {1, 2}, "T2": {5}},
        prolif={"T1": {1}, "T2": {5, 6}},
    )

    assert payload["T1"]["confirmed_atoms"] == [1]
    assert payload["T1"]["pose_supported_interaction_atoms"] == [1]
    assert payload["T1"]["evidence_label"] == "pose-supported interaction atoms"
    assert payload["T1"]["degraded"] is False
    assert payload["T2"]["confirmed_atoms"] == [5]
    assert set(payload["T1"]["votes"]) == {"plip", "prolif"}


def test_targets_with_all_sources_requires_same_target_coverage() -> None:
    source_payloads = {
        "plip": {"T1": {1}, "T2": {2}},
        "prolif": {"T1": {1}},
    }
    assert targets_with_all_sources(source_payloads) == {"T1"}

    source_payloads["prolif"]["T2"] = {2}
    assert targets_with_all_sources(source_payloads) == {"T1", "T2"}


def test_cli_fails_when_required_source_is_missing(tmp_path: Path) -> None:
    plip = tmp_path / "plip.xml"
    prolif = tmp_path / "missing.csv"
    out_json = tmp_path / "consensus.json"
    plip.write_text("<root><target id='T1'><ligand_atom idx='1'/></target></root>\n")
    out_json.write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "stage5_5_consensus.py"),
            "--plip",
            str(plip),
            "--prolif",
            str(prolif),
            "--out-json",
            str(out_json),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert res.returncode != 0
    assert "Required pose-supported interaction atom source file(s) missing" in res.stderr
    assert not out_json.exists()


def test_cli_writes_pose_supported_interaction_atoms(tmp_path: Path) -> None:
    plip = tmp_path / "plip.xml"
    prolif = tmp_path / "prolif.csv"
    out_json = tmp_path / "consensus.json"
    plip.write_text("<root><target id='T1'><ligand_atom idx='1'/></target></root>\n")
    prolif.write_text("target_id,atom_idx\nT1,1\n")

    res = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "stage5_5_consensus.py"),
            "--plip",
            str(plip),
            "--prolif",
            str(prolif),
            "--out-json",
            str(out_json),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert res.returncode == 0, res.stderr
    payload = json.loads(out_json.read_text())
    assert payload["T1"]["confirmed_atoms"] == [1]
    assert payload["T1"]["pose_supported_interaction_atoms"] == [1]
    assert payload["T1"]["evidence_label"] == "pose-supported interaction atoms"
    assert payload["T1"]["coordinate_system"] == "boltz_complex_ligand_atom_order_0_based"
    assert payload["T1"]["degraded"] is False
    assert payload["T1"]["claim_eligible"] is True


def test_cli_diagnostic_override_marks_consensus_unclaimable(tmp_path: Path) -> None:
    plip = tmp_path / "plip.xml"
    prolif = tmp_path / "prolif.csv"
    out_json = tmp_path / "consensus.json"
    plip.write_text("<root><target id='T1'><ligand_atom idx='1'/></target></root>\n")
    prolif.write_text("target_id,atom_idx\nT1,1\n")

    res = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "stage5_5_consensus.py"),
            "--plip",
            str(plip),
            "--prolif",
            str(prolif),
            "--out-json",
            str(out_json),
            "--allow-empty-sources",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert res.returncode == 0, res.stderr
    payload = json.loads(out_json.read_text())
    assert payload["T1"]["degraded"] is True
    assert payload["T1"]["claim_eligible"] is False
