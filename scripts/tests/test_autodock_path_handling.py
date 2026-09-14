"""Regression tests for path handling in the AutoDock stage.

Both defects here only appear in a realistic configuration and cost a full
proteome run to surface: 13,339 receptors docked, every one failing, and the
publish step dying afterwards.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import stage3_autodock_run as adr  # noqa: E402


def test_the_ligand_path_is_absolute_when_the_binary_runs(
    tmp_path: Path, monkeypatch
) -> None:
    """The subprocess runs from the map bundle, so a relative path cannot open.

    AutoDock-GPU reported "Can't open ligand data file" for all 13,339
    receptors of a full run because the caller's relative path was passed
    through unchanged.
    """
    maps_dir = tmp_path / "maps"
    maps_dir.mkdir()
    map_file = maps_dir / "T.maps.fld"
    map_file.write_text("field\n")
    ligand = tmp_path / "ligand.pdbqt"
    ligand.write_text("ATOM\n")

    seen: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["cwd"] = kwargs.get("cwd")
        return subprocess.CompletedProcess(cmd, 0, "All jobs (1) ran without errors.", "")

    monkeypatch.setattr(adr.subprocess, "run", fake_run)
    monkeypatch.chdir(tmp_path)

    adr.run_autodock_gpu_one(
        Path("ligand.pdbqt"), map_file, 4, "ad", tmp_path / "work", "autodock_gpu_128wi"
    )

    cmd = seen["cmd"]
    ligand_arg = Path(cmd[cmd.index("-L") + 1])
    resnam_arg = Path(cmd[cmd.index("--resnam") + 1])
    assert ligand_arg.is_absolute(), ligand_arg
    assert resnam_arg.is_absolute(), resnam_arg
    assert ligand_arg == ligand.resolve()
    # The working directory really is elsewhere, which is what breaks relatives.
    assert Path(str(seen["cwd"])).resolve() == maps_dir.resolve()


def test_poses_are_staged_beside_their_destination(monkeypatch) -> None:
    """os.replace cannot move a directory across filesystems.

    Staging in the default temp root put the poses on tmpfs while the run
    directory was on disk, so publishing raised EXDEV after every pose had
    already been written.
    """
    source = (ROOT / "scripts" / "stage3_autodock_run.py").read_text()
    assert "tempfile.TemporaryDirectory(dir=staging_parent)" in source
    assert "args.out_pose_dir.parent if args.out_pose_dir is not None" in source


def test_a_cross_device_publish_is_what_the_staging_choice_prevents(
    tmp_path: Path,
) -> None:
    """Pin the failure mode itself, so the reason for the choice stays visible."""
    destination = tmp_path / "poses"
    with tempfile.TemporaryDirectory() as far_away:
        staged = Path(far_away) / "poses"
        staged.mkdir()
        if Path(far_away).stat().st_dev == tmp_path.stat().st_dev:
            pytest.skip("temp root shares a filesystem with tmp_path here")
        with pytest.raises(OSError):
            staged.replace(destination)

    near = tmp_path / "staging"
    near.mkdir()
    (near / "pose.sdf").write_text("x\n")
    near.replace(destination)
    assert (destination / "pose.sdf").exists()
