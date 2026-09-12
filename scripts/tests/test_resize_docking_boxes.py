"""Regression tests for docking-box resizing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from resize_docking_boxes import (  # noqa: E402
    BoxResizeError,
    grid_points,
    map_bytes_per_receptor,
    parse_args,
    read_box,
    resize_boxes,
)


def _box(path: Path, edge: float = 30.0, center=(1.0, 2.0, 3.0)) -> Path:
    path.write_text(
        "\n".join(
            [
                f"center_x = {center[0]:.3f}",
                f"center_y = {center[1]:.3f}",
                f"center_z = {center[2]:.3f}",
                f"size_x = {edge:.3f}",
                f"size_y = {edge:.3f}",
                f"size_z = {edge:.3f}",
                "",
            ]
        )
    )
    return path


def _run(tmp_path: Path, *extra: str):
    box_dir = tmp_path / "boxes"
    return resize_boxes(
        parse_args(
            [
                "--box-dir", str(box_dir),
                "--out-dir", str(tmp_path / "out"),
                "--out-manifest", str(tmp_path / "manifest.json"),
                *extra,
            ]
        )
    )


def test_resize_keeps_centres_and_shrinks_every_edge(tmp_path: Path) -> None:
    box_dir = tmp_path / "boxes"
    box_dir.mkdir()
    _box(box_dir / "P1.box.txt", center=(1.5, -2.5, 3.25))
    _box(box_dir / "P2.box.txt", center=(0.0, 0.0, 0.0))

    manifest = _run(tmp_path, "--edge", "20")

    assert manifest["boxes"] == 2
    assert manifest["edge_angstrom"] == 20.0
    assert manifest["previous_edge_angstrom"] == [30.0]
    resized = read_box(tmp_path / "out" / "P1.box.txt")
    assert resized["center_x"] == pytest.approx(1.5)
    assert resized["center_y"] == pytest.approx(-2.5)
    assert resized["center_z"] == pytest.approx(3.25)
    assert resized["size_x"] == pytest.approx(20.0)
    assert resized["size_z"] == pytest.approx(20.0)


def test_resize_records_the_grid_and_cache_consequence(tmp_path: Path) -> None:
    box_dir = tmp_path / "boxes"
    box_dir.mkdir()
    _box(box_dir / "P1.box.txt")

    manifest = _run(tmp_path, "--edge", "20")

    grid = manifest["grid"]
    assert grid["npts_per_axis"] == 54
    assert grid["grid_points"] == 55**3
    assert grid["previous_npts_per_axis"] == {"30": 80}
    estimate = manifest["autogrid_cache_estimate"]
    assert estimate["bytes_per_receptor"] == 8 * 55**3 * 10
    assert "narrows the searched volume" in manifest["claim_limit"]


def test_resize_refuses_to_enlarge_a_box(tmp_path: Path) -> None:
    box_dir = tmp_path / "boxes"
    box_dir.mkdir()
    _box(box_dir / "P1.box.txt", edge=16.0)

    with pytest.raises(BoxResizeError, match="refusing to enlarge"):
        _run(tmp_path, "--edge", "20")


def test_resize_rejects_a_box_missing_a_field(tmp_path: Path) -> None:
    box_dir = tmp_path / "boxes"
    box_dir.mkdir()
    (box_dir / "P1.box.txt").write_text("center_x = 1.0\nsize_x = 30.0\n")

    with pytest.raises(BoxResizeError, match="missing field"):
        _run(tmp_path, "--edge", "20")


def test_resize_rejects_a_nonpositive_size(tmp_path: Path) -> None:
    box_dir = tmp_path / "boxes"
    box_dir.mkdir()
    _box(box_dir / "P1.box.txt", edge=0.0)

    with pytest.raises(BoxResizeError, match="must be positive"):
        _run(tmp_path, "--edge", "20")


def test_resize_rejects_an_empty_box_directory(tmp_path: Path) -> None:
    (tmp_path / "boxes").mkdir()

    with pytest.raises(BoxResizeError, match="no \\*.box.txt"):
        _run(tmp_path, "--edge", "20")


@pytest.mark.parametrize(
    ("edge", "spacing", "expected"),
    [(20.0, 0.375, 54), (30.0, 0.375, 80), (24.0, 0.375, 64), (1.0, 0.375, 4)],
)
def test_grid_points_rounds_up_to_an_even_npts(
    edge: float, spacing: float, expected: int
) -> None:
    assert grid_points(edge, spacing) == expected


def test_grid_points_rejects_a_box_autogrid_cannot_grid() -> None:
    with pytest.raises(BoxResizeError, match="even npts"):
        grid_points(100.0, 0.375)
    with pytest.raises(BoxResizeError, match="positive"):
        grid_points(-1.0, 0.375)


def test_cache_estimate_scales_cubically() -> None:
    small = map_bytes_per_receptor(54, maps=8, bytes_per_point=10)
    large = map_bytes_per_receptor(80, maps=8, bytes_per_point=10)

    assert large / small == pytest.approx((81**3) / (55**3))


def test_manifest_is_written_to_disk(tmp_path: Path) -> None:
    box_dir = tmp_path / "boxes"
    box_dir.mkdir()
    _box(box_dir / "P1.box.txt")

    _run(tmp_path, "--edge", "20")

    payload = json.loads((tmp_path / "manifest.json").read_text())
    assert payload["schema_version"] == "skinscout.docking-box-resize.v1"
    assert payload["centres"] == "unchanged"
