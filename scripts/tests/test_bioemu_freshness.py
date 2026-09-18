"""BioEmu freshness regressions (audit F16).

Stage 6 used to run BioEmu in the live target directory and treat returncode 0
as success. A run that exited 0 without writing anything therefore read the
previous run's ensemble as current. These tests pin the fresh-staging and
validate-before-publish behaviour.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import residue_mapping as rm  # noqa: E402
import stage6_bioemu as stage6  # noqa: E402

from test_residue_gap_mapping import continuous_pdb  # noqa: E402


def make_receptor(tmp_path: Path) -> Path:
    receptor = continuous_pdb(tmp_path / "P00001_input.pdb", n=6)
    record = rm.build_mapping_record(
        receptor,
        target_id="P00001",
        source="alphafold_cleaned",
        canonical_sequence="GGGGGG",
    )
    rm.write_mapping_record(receptor, record, rm.mapping_path_for_pdb(receptor))
    return receptor


def write_ensemble(directory: Path, *, frames: int = 2, write_topology: bool = True) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    if write_topology:
        (directory / "topology.pdb").write_text(
            "\n".join(
                f"ATOM  {index:5d}   CA GLY A{index:4d}    "
                f"{float(index):8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 10.00           C"
                for index in range(1, 7)
            )
            + "\nEND\n"
        )
    if frames > 0:
        models = []
        for frame in range(frames):
            models.append(f"MODEL     {frame + 1:4d}")
            models.append(
                "ATOM      1   CA GLY A   1       0.000   0.000   0.000  "
                "1.00 10.00           C"
            )
            models.append("ENDMDL")
        (directory / "samples_0.pdb").write_text("\n".join(models) + "\n")


def completed(returncode: int = 0) -> types.SimpleNamespace:
    return types.SimpleNamespace(returncode=returncode, stdout="", stderr="")


def test_stale_files_with_rc0_and_no_new_output_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receptor = make_receptor(tmp_path)
    target = tmp_path / "06_bioemu" / "P00001"
    write_ensemble(target, frames=2)
    (target / "medoid_00.pdb").write_text("stale medoid\n")
    stale = {
        path.name: path.read_bytes()
        for path in target.iterdir()
        if path.is_file()
    }
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return completed(0)

    monkeypatch.setattr(stage6, "bioemu_available", lambda: True)
    monkeypatch.setattr(stage6.subprocess, "run", fake_run)

    assert stage6.run_bioemu(receptor, target, 20) is False

    assert calls, "sanity: BioEmu must still have been invoked"
    staging_dir = Path(calls[0][-1])
    assert staging_dir != target
    assert staging_dir.parent == target.parent
    assert not staging_dir.exists(), "staging directory was not cleaned up"
    for name, body in stale.items():
        assert (target / name).read_bytes() == body, name
    assert not (target / stage6.FRESHNESS_FILENAME).exists()
    assert not [p for p in target.parent.glob(".P00001.bioemu_*")]


def test_fresh_validated_ensemble_is_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receptor = make_receptor(tmp_path)
    target = tmp_path / "06_bioemu" / "P00001"
    target.mkdir(parents=True)
    (target / "stale.txt").write_text("old run\n")
    received: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        received["sequence"] = cmd[3]
        received["num"] = cmd[4]
        write_ensemble(Path(cmd[-1]), frames=3)
        return completed(0)

    monkeypatch.setattr(stage6, "bioemu_available", lambda: True)
    monkeypatch.setattr(stage6.subprocess, "run", fake_run)

    assert stage6.run_bioemu(receptor, target, 20) is True

    assert received["sequence"] == "GGGGGG"
    assert received["num"] == "20"
    assert not (target / "stale.txt").exists(), "stale output was not replaced"
    assert (target / "topology.pdb").is_file()
    assert (target / "samples_0.pdb").is_file()
    assert stage6.verify_freshness_manifest(target) == (True, "")
    record = stage6.json.loads((target / stage6.FRESHNESS_FILENAME).read_text())
    assert record["schema_version"] == stage6.FRESHNESS_SCHEMA
    assert record["total_frames"] == 3
    assert record["n_conf_requested"] == 20
    assert record["sequence_sha256"] == rm._sha256_text("GGGGGG")


def test_zero_frame_output_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receptor = make_receptor(tmp_path)
    target = tmp_path / "06_bioemu" / "P00001"
    target.mkdir(parents=True)

    def fake_run(cmd, **kwargs):
        write_ensemble(Path(cmd[-1]), frames=0)
        return completed(0)

    monkeypatch.setattr(stage6, "bioemu_available", lambda: True)
    monkeypatch.setattr(stage6.subprocess, "run", fake_run)

    assert stage6.run_bioemu(receptor, target, 20) is False
    assert not (target / stage6.FRESHNESS_FILENAME).exists()
    assert not [p for p in target.iterdir()]


def test_freshness_manifest_detects_later_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receptor = make_receptor(tmp_path)
    target = tmp_path / "06_bioemu" / "P00001"
    target.mkdir(parents=True)

    def fake_run(cmd, **kwargs):
        write_ensemble(Path(cmd[-1]), frames=2)
        return completed(0)

    monkeypatch.setattr(stage6, "bioemu_available", lambda: True)
    monkeypatch.setattr(stage6.subprocess, "run", fake_run)
    assert stage6.run_bioemu(receptor, target, 20) is True
    assert stage6.verify_freshness_manifest(target) == (True, "")

    (target / "samples_0.pdb").write_text("replaced after validation\n")

    ok, reason = stage6.verify_freshness_manifest(target)
    assert ok is False
    assert "hash changed" in reason
