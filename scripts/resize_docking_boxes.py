#!/usr/bin/env python3
"""Re-emit AutoDock docking boxes at a smaller cubic edge.

The committed boxes are a uniform 30 A cube. That is not a pocket measurement:
`stage0_meeko_prep.write_box` computes `min(2 * (max(radius, 8) + 4), 30)` and
every `*.pockets.json` carries the constant fallback `radius = 12.0`, so the
expression collapses to the 30 A cap for all 15,038 receptors.

Edge length drives the AutoGrid map cache cubically. At 0.375 A spacing a 30 A
box is 81^3 points, roughly 42 MB of maps per receptor and about 639 GB across
the proteome — past the 300 GB the README asks of a user machine. A 20 A box is
55^3 points, roughly 13 MB, about 200 GB.

Shrinking is a real narrowing of the search volume, so it is recorded rather
than applied silently: the manifest carries the old and new edge, the grid-point
and cache-size implications, and the box centres are left untouched.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "skinscout.docking-box-resize.v1"
FIELD_RE = re.compile(r"(\w+)\s*=\s*(-?[0-9.]+)")
CENTER_FIELDS = ("center_x", "center_y", "center_z")
SIZE_FIELDS = ("size_x", "size_y", "size_z")
# AutoGrid rejects odd or out-of-range npts; stage3_autogrid_maps enforces the
# same window, so a resize that cannot be gridded must fail here instead.
MIN_NPTS = 2
MAX_NPTS = 126


class BoxResizeError(ValueError):
    """Raised when a box file or requested edge violates its contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_box(path: Path) -> dict[str, float]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BoxResizeError(f"docking box could not be read: {path}: {exc}") from exc
    fields = {key: float(value) for key, value in FIELD_RE.findall(text)}
    missing = sorted({*CENTER_FIELDS, *SIZE_FIELDS} - set(fields))
    if missing:
        raise BoxResizeError(
            f"docking box missing field(s) {', '.join(missing)}: {path}"
        )
    for key, value in fields.items():
        if not math.isfinite(value):
            raise BoxResizeError(f"docking box field {key} is not finite: {path}")
    for key in SIZE_FIELDS:
        if fields[key] <= 0.0:
            raise BoxResizeError(f"docking box {key} must be positive: {path}")
    return fields


def grid_points(edge: float, spacing: float) -> int:
    """AutoGrid npts for one axis, validated against the engine's window."""
    if not math.isfinite(edge) or edge <= 0.0:
        raise BoxResizeError(f"box edge must be positive and finite: {edge!r}")
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise BoxResizeError(f"grid spacing must be positive and finite: {spacing!r}")
    npts = int(math.ceil(edge / spacing))
    if npts % 2:
        npts += 1
    if not MIN_NPTS <= npts <= MAX_NPTS:
        raise BoxResizeError(
            f"AutoGrid requires an even npts in [{MIN_NPTS}, {MAX_NPTS}]; "
            f"edge={edge:g} spacing={spacing:g} gives npts={npts}"
        )
    return npts


def map_bytes_per_receptor(npts: int, *, maps: int, bytes_per_point: int) -> int:
    return int(maps * (npts + 1) ** 3 * bytes_per_point)


def resize_text(fields: dict[str, float], edge: float) -> str:
    lines = [f"{key} = {fields[key]:.3f}" for key in CENTER_FIELDS]
    lines += [f"{key} = {edge:.3f}" for key in SIZE_FIELDS]
    return "\n".join([*lines, ""])


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def resize_boxes(args: argparse.Namespace) -> dict[str, Any]:
    if not args.box_dir.is_dir():
        raise BoxResizeError(f"--box-dir is not a directory: {args.box_dir}")
    edge = float(args.edge)
    npts = grid_points(edge, args.spacing)
    sources = sorted(args.box_dir.glob("*.box.txt"))
    if not sources:
        raise BoxResizeError(f"no *.box.txt files under {args.box_dir}")

    previous_edges: set[float] = set()
    written = 0
    for source in sources:
        fields = read_box(source)
        observed = {round(fields[key], 3) for key in SIZE_FIELDS}
        previous_edges.update(observed)
        if min(observed) < edge:
            raise BoxResizeError(
                f"{source.name} is already smaller than the requested edge "
                f"({min(observed):g} < {edge:g}); refusing to enlarge a box"
            )
        _write_atomic(args.out_dir / source.name, resize_text(fields, edge))
        written += 1

    # JSON coerces numeric keys to strings, so emit them that way from the start
    # and keep the in-memory manifest identical to the file on disk.
    previous_npts = {
        f"{value:g}": grid_points(value, args.spacing)
        for value in sorted(previous_edges)
    }
    per_receptor = map_bytes_per_receptor(
        npts, maps=args.map_count, bytes_per_point=args.bytes_per_point
    )
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "source_box_dir": str(args.box_dir),
        "out_box_dir": str(args.out_dir),
        "boxes": written,
        "edge_angstrom": edge,
        "previous_edge_angstrom": sorted(previous_edges),
        "grid": {
            "spacing_angstrom": float(args.spacing),
            "npts_per_axis": npts,
            "grid_points": int((npts + 1) ** 3),
            "previous_npts_per_axis": previous_npts,
        },
        "autogrid_cache_estimate": {
            "map_count": int(args.map_count),
            "bytes_per_point": int(args.bytes_per_point),
            "bytes_per_receptor": per_receptor,
            "megabytes_per_receptor": round(per_receptor / 1e6, 2),
            "gigabytes_for_all_boxes": round(per_receptor * written / 1e9, 1),
        },
        "centres": "unchanged",
        "claim_limit": (
            "a smaller box narrows the searched volume around the P2Rank pocket "
            "centre; poses outside the new edge cannot be found"
        ),
    }
    _write_atomic(args.out_manifest, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--box-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--out-manifest", required=True, type=Path)
    parser.add_argument("--edge", type=float, default=20.0)
    parser.add_argument("--spacing", type=float, default=0.375)
    parser.add_argument(
        "--map-count",
        type=int,
        default=8,
        help="AutoGrid map files per receptor used for the cache estimate",
    )
    parser.add_argument("--bytes-per-point", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        manifest = resize_boxes(parse_args(argv))
    except BoxResizeError as exc:
        print(str(exc))
        return 1
    estimate = manifest["autogrid_cache_estimate"]
    print(
        f"[resize-boxes] wrote {manifest['boxes']} boxes at {manifest['edge_angstrom']:g} A "
        f"(npts {manifest['grid']['npts_per_axis']}); "
        f"AutoGrid cache ~{estimate['megabytes_per_receptor']:g} MB/receptor, "
        f"~{estimate['gigabytes_for_all_boxes']:g} GB total"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
