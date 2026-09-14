"""Regression tests for the per-receptor docking background builder."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_docking_backgrounds import (  # noqa: E402
    BackgroundBuildError,
    load_ligand_panel,
    main,
    read_box,
)


def _panel(path: Path, n: int = 40) -> Path:
    path.write_text(
        json.dumps({f"L{i:03d}": f"ROOT\nATOM  {i}\nENDROOT\n" for i in range(n)})
    )
    return path


def _shards(directory: Path, receptors: dict[str, list[float]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for target_id, scores in receptors.items():
        (directory / f"{target_id}.json").write_text(
            json.dumps({f"L{i:03d}": s for i, s in enumerate(scores)})
        )


def _summarise(tmp_path: Path, *extra: str) -> int:
    return main(
        [
            "--receptor-list", str(tmp_path / "receptors.txt"),
            "--ligand-panel", str(tmp_path / "panel.json"),
            "--receptor-dir", str(tmp_path),
            "--box-dir", str(tmp_path),
            "--out-dir", str(tmp_path / "shards"),
            "--out-backgrounds", str(tmp_path / "backgrounds.json"),
            "--out-manifest", str(tmp_path / "manifest.json"),
            "--summarise-only",
            *extra,
        ]
    )


def _scores(center: float, spread: float, n: int = 40) -> list[float]:
    rng = np.random.default_rng(0)
    return (center + spread * rng.standard_normal(n)).tolist()


def test_summarise_turns_shards_into_backgrounds_and_a_manifest(tmp_path: Path) -> None:
    _panel(tmp_path / "panel.json")
    (tmp_path / "receptors.txt").write_text("P1\nP2\n")
    _shards(tmp_path / "shards", {"P1": _scores(-6.0, 0.8), "P2": _scores(-9.0, 0.6)})

    assert _summarise(tmp_path) == 0

    backgrounds = json.loads((tmp_path / "backgrounds.json").read_text())
    assert set(backgrounds) == {"P1", "P2"}
    assert backgrounds["P1"]["center"] == pytest.approx(-6.0, abs=0.3)
    assert backgrounds["P2"]["center"] == pytest.approx(-9.0, abs=0.3)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["run_schema_version"] == "skinscout.docking_background_run.v1"
    assert manifest["receptors"] == 2
    assert manifest["panel_ligands"] == 40
    assert "not a probability" in manifest["semantics"]


def test_summarise_records_receptors_it_had_to_reject(tmp_path: Path) -> None:
    """A receptor whose panel is thin or flat cannot define a scale."""
    _panel(tmp_path / "panel.json")
    (tmp_path / "receptors.txt").write_text("P1\nP2\nP3\n")
    _shards(
        tmp_path / "shards",
        {
            "P1": _scores(-6.0, 0.8),
            "P2": [-7.0] * 40,          # no spread
            "P3": _scores(-6.0, 0.8, 5),  # too few ligands
        },
    )

    assert _summarise(tmp_path) == 0

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["receptors"] == 1
    assert manifest["receptors_rejected"] == 2
    assert "no usable spread" in manifest["rejection_reasons"]["P2"]
    assert "at least" in manifest["rejection_reasons"]["P3"]
    assert set(json.loads((tmp_path / "backgrounds.json").read_text())) == {"P1"}


def test_summarise_fails_when_no_receptor_is_usable(tmp_path: Path) -> None:
    _panel(tmp_path / "panel.json")
    (tmp_path / "receptors.txt").write_text("P1\n")
    _shards(tmp_path / "shards", {"P1": [-7.0] * 40})

    assert _summarise(tmp_path) == 1


def test_panel_must_be_non_empty_and_carry_atoms(tmp_path: Path) -> None:
    empty = tmp_path / "empty.json"
    empty.write_text("{}")
    with pytest.raises(BackgroundBuildError, match="non-empty"):
        load_ligand_panel(empty)

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"L1": "no atoms here"}))
    with pytest.raises(BackgroundBuildError, match="no PDBQT atoms"):
        load_ligand_panel(bad)

    with pytest.raises(BackgroundBuildError, match="required"):
        load_ligand_panel(tmp_path / "missing.json")


def test_read_box_requires_every_field(tmp_path: Path) -> None:
    path = tmp_path / "P1.box.txt"
    path.write_text("center_x = 1.0\nsize_x = 20.0\n")

    with pytest.raises(BackgroundBuildError, match="box missing"):
        read_box(path)

    path.write_text(
        "center_x = 1.0\ncenter_y = 2.0\ncenter_z = 3.0\n"
        "size_x = 20.0\nsize_y = 20.0\nsize_z = 20.0\n"
    )
    center, size = read_box(path)
    assert center == [1.0, 2.0, 3.0]
    assert size == [20.0, 20.0, 20.0]


def test_empty_receptor_list_is_refused(tmp_path: Path) -> None:
    _panel(tmp_path / "panel.json")
    (tmp_path / "receptors.txt").write_text("\n")
    (tmp_path / "shards").mkdir()

    assert _summarise(tmp_path) == 1
