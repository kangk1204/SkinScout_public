"""Regression tests for pocket-extent-derived docking boxes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from derive_docking_boxes import main as derive_main  # noqa: E402
from pocket_box import (  # noqa: E402
    PocketBoxError,
    box_for_target,
    derive_box,
    format_box_text,
    read_atom_coordinates,
    read_top_pocket,
)

PRED_HEADER = (
    "name     ,  rank,   score, probability, sas_points, surf_atoms,"
    "   center_x,   center_y,   center_z, residue_ids, surf_atom_ids\n"
)


def _prediction(path: Path, center=(0.0, 0.0, 0.0), serials=(1, 2, 3, 4), rank=1) -> Path:
    path.write_text(
        PRED_HEADER
        + f"pocket1  ,     {rank},   23.53,       0.841,        127,         49,"
        f"     {center[0]},     {center[1]},    {center[2]}, A_10,"
        f" {' '.join(str(s) for s in serials)}\n"
    )
    return path


def _pdb(path: Path, coords: dict[int, tuple[float, float, float]]) -> Path:
    lines = [
        f"ATOM  {serial:>5}  CA  ALA A{serial:>4}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 50.00           C"
        for serial, (x, y, z) in sorted(coords.items())
    ]
    path.write_text("\n".join([*lines, "END", ""]))
    return path


def _cube(radius: float) -> np.ndarray:
    return np.array(
        [[sx * radius, sy * radius, sz * radius]
         for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
        dtype=float,
    )


def test_box_reaches_the_outermost_pocket_atom_plus_headroom() -> None:
    box = derive_box((0.0, 0.0, 0.0), _cube(6.0), headroom=4.0)

    # half-width 6 A + 4 A headroom -> 20 A edge on every axis
    assert box.size == pytest.approx((20.0, 20.0, 20.0))
    assert box.half_width == pytest.approx((6.0, 6.0, 6.0))
    assert box.surface_atoms == 8
    assert box.clamped_low == () and box.clamped_high == ()


def test_box_is_anisotropic_when_the_pocket_is() -> None:
    coords = np.array(
        [[-9.0, -2.0, -2.0], [9.0, 2.0, 2.0], [0.0, -2.0, 2.0], [0.0, 2.0, -2.0]]
    )

    box = derive_box((0.0, 0.0, 0.0), coords, headroom=3.0)

    assert box.size[0] == pytest.approx(24.0)
    assert box.size[1] == pytest.approx(16.0)
    assert box.size[2] == pytest.approx(16.0)


def test_box_stays_centred_on_the_predicted_pocket_centre() -> None:
    """The centre is P2Rank's prediction; only the extent is derived."""
    box = derive_box((5.0, -3.0, 2.0), _cube(6.0) + np.array([5.0, -3.0, 2.0]))

    assert box.center == pytest.approx((5.0, -3.0, 2.0))


def test_an_offset_pocket_widens_the_box_rather_than_moving_it() -> None:
    coords = _cube(2.0) + np.array([8.0, 0.0, 0.0])

    box = derive_box((0.0, 0.0, 0.0), coords, headroom=0.0, min_edge=1.0)

    assert box.center == pytest.approx((0.0, 0.0, 0.0))
    assert box.size[0] == pytest.approx(20.0)


def test_box_clamps_are_reported_per_axis() -> None:
    tiny = derive_box((0.0, 0.0, 0.0), _cube(0.5), headroom=0.0, min_edge=16.0)
    assert tiny.size == pytest.approx((16.0, 16.0, 16.0))
    assert set(tiny.clamped_low) == {"x", "y", "z"}

    huge = derive_box((0.0, 0.0, 0.0), _cube(40.0), headroom=0.0, max_edge=40.0)
    assert huge.size == pytest.approx((40.0, 40.0, 40.0))
    assert set(huge.clamped_high) == {"x", "y", "z"}


def test_derive_box_needs_enough_atoms_to_bound_a_volume() -> None:
    with pytest.raises(PocketBoxError, match="bound a volume"):
        derive_box((0.0, 0.0, 0.0), np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"headroom": -1.0}, "headroom"),
        ({"min_edge": 0.0}, "min_edge"),
        ({"max_edge": float("nan")}, "max_edge"),
        ({"min_edge": 30.0, "max_edge": 20.0}, "exceeds"),
    ],
)
def test_derive_box_rejects_invalid_parameters(kwargs: dict, match: str) -> None:
    with pytest.raises(PocketBoxError, match=match):
        derive_box((0.0, 0.0, 0.0), _cube(5.0), **kwargs)


def test_reading_a_pocket_requires_a_rank_one_row(tmp_path: Path) -> None:
    path = _prediction(tmp_path / "p.csv", rank=2)

    with pytest.raises(PocketBoxError, match="no rank-1 pocket"):
        read_top_pocket(path)


def test_reading_a_pocket_requires_surface_atoms(tmp_path: Path) -> None:
    path = tmp_path / "p.csv"
    path.write_text(PRED_HEADER + "pocket1  ,     1,  1.0, 0.5, 1, 1,  0.0, 0.0, 0.0, A_1, \n")

    with pytest.raises(PocketBoxError, match="no surface atoms"):
        read_top_pocket(path)


def test_a_missing_surface_atom_is_an_error_not_a_smaller_box(tmp_path: Path) -> None:
    """Dropping an unresolved atom would silently shrink the search volume."""
    pdb = _pdb(tmp_path / "s.pdb", {1: (0.0, 0.0, 0.0), 2: (1.0, 1.0, 1.0)})

    with pytest.raises(PocketBoxError, match="absent from"):
        read_atom_coordinates(pdb, {1, 2, 99})


def test_end_to_end_target_box(tmp_path: Path) -> None:
    _prediction(tmp_path / "P1_clean.pdb_predictions.csv", serials=(1, 2, 3, 4))
    _pdb(
        tmp_path / "P1_clean.pdb",
        {1: (-5.0, -5.0, -5.0), 2: (5.0, 5.0, 5.0), 3: (-5.0, 5.0, -5.0), 4: (5.0, -5.0, 5.0)},
    )

    box = box_for_target(
        tmp_path / "P1_clean.pdb_predictions.csv",
        tmp_path / "P1_clean.pdb",
        headroom=4.0,
    )

    assert box.size == pytest.approx((18.0, 18.0, 18.0))
    text = format_box_text(box)
    assert "center_x = 0.000" in text
    assert "size_z = 18.000" in text


def _tree(tmp_path: Path) -> tuple[Path, Path]:
    pockets = tmp_path / "pockets"; pockets.mkdir()
    clean = tmp_path / "clean"; clean.mkdir()
    coords = {1: (-6.0, -6.0, -6.0), 2: (6.0, 6.0, 6.0), 3: (-6.0, 6.0, -6.0), 4: (6.0, -6.0, 6.0)}
    for uid in ("P1", "P2"):
        _prediction(pockets / f"{uid}_clean.pdb_predictions.csv")
        _pdb(clean / f"{uid}_clean.pdb", coords)
    # P3 has a pocket but no structure, P4 has no rank-1 pocket
    _prediction(pockets / "P3_clean.pdb_predictions.csv")
    _prediction(pockets / "P4_clean.pdb_predictions.csv", rank=3)
    _pdb(clean / "P4_clean.pdb", coords)
    return pockets, clean


def test_cli_writes_boxes_and_records_every_exclusion(tmp_path: Path) -> None:
    pockets, clean = _tree(tmp_path)

    code = derive_main(
        [
            "--pocket-dir", str(pockets),
            "--clean-dir", str(clean),
            "--out-dir", str(tmp_path / "boxes"),
            "--out-manifest", str(tmp_path / "manifest.json"),
            "--out-exclusions", str(tmp_path / "exclusions.csv"),
        ]
    )

    assert code == 0
    assert sorted(p.name for p in (tmp_path / "boxes").glob("*.box.txt")) == [
        "P1.box.txt",
        "P2.box.txt",
    ]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["counts"]["boxes_written"] == 2
    assert manifest["counts"]["excluded"] == 2
    assert manifest["exclusion_reasons"]["P3"] == "missing_clean_structure"
    assert "no rank-1 pocket" in manifest["exclusion_reasons"]["P4"]
    assert manifest["edge_angstrom"]["p50"] == pytest.approx(20.0)
    exclusions = (tmp_path / "exclusions.csv").read_text().splitlines()
    assert exclusions[0] == "uniprot,reason"
    assert any(line.startswith("P3,") for line in exclusions)


def test_cli_reports_boxes_over_the_vina_search_volume_advisory(tmp_path: Path) -> None:
    """Vina needs raised exhaustiveness above 27000 A^3; the count must be visible."""
    pockets, clean = _tree(tmp_path)

    code = derive_main(
        [
            "--pocket-dir", str(pockets),
            "--clean-dir", str(clean),
            "--out-dir", str(tmp_path / "boxes"),
            "--out-manifest", str(tmp_path / "manifest.json"),
            # the fixture pocket yields a 20 A cube = 8000 A^3
            "--advisory-volume", "5000",
        ]
    )

    assert code == 0
    advisory = json.loads((tmp_path / "manifest.json").read_text())["search_volume_advisory"]
    assert advisory["threshold_cubic_angstrom"] == 5000.0
    assert advisory["boxes_above_threshold"] == 2
    assert advisory["fraction_above_threshold"] == pytest.approx(1.0)
    assert "exhaustiveness" in advisory["guidance"]


def test_cli_counts_no_box_over_a_generous_advisory(tmp_path: Path) -> None:
    pockets, clean = _tree(tmp_path)

    derive_main(
        [
            "--pocket-dir", str(pockets),
            "--clean-dir", str(clean),
            "--out-dir", str(tmp_path / "boxes"),
            "--out-manifest", str(tmp_path / "manifest.json"),
            "--advisory-volume", "100000",
        ]
    )

    advisory = json.loads((tmp_path / "manifest.json").read_text())["search_volume_advisory"]
    assert advisory["boxes_above_threshold"] == 0
    assert advisory["fraction_above_threshold"] == pytest.approx(0.0)


def test_cli_fails_when_no_target_yields_a_box(tmp_path: Path) -> None:
    pockets = tmp_path / "pockets"; pockets.mkdir()
    (tmp_path / "clean").mkdir()
    _prediction(pockets / "P1_clean.pdb_predictions.csv")

    assert derive_main(
        [
            "--pocket-dir", str(pockets),
            "--clean-dir", str(tmp_path / "clean"),
            "--out-dir", str(tmp_path / "boxes"),
            "--out-manifest", str(tmp_path / "manifest.json"),
        ]
    ) == 1
