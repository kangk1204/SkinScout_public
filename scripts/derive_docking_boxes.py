#!/usr/bin/env python3
"""Re-emit every docking box from its P2Rank pocket's measured extent.

Replaces the constant-radius cube written by `stage0_meeko_prep` with a box
sized from the rank-1 pocket's surface atoms. See `scripts/pocket_box.py` for
why the previous size carried no geometric information.

Targets whose pocket cannot yield a trustworthy extent are recorded as
exclusions rather than given a default box, so a missing pocket never turns into
a silently plausible search volume.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from pocket_box import (
    DEFAULT_HEADROOM,
    DEFAULT_MAX_EDGE,
    DEFAULT_MIN_EDGE,
    SCHEMA_VERSION,
    PocketBoxError,
    box_for_target,
    format_box_text,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _percentiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        f"p{q:02d}": round(float(np.percentile(array, q)), 3)
        for q in (5, 25, 50, 75, 95, 99)
    }


def derive_all(args: argparse.Namespace) -> dict[str, Any]:
    targets = sorted(
        path.name.replace("_clean.pdb_predictions.csv", "")
        for path in args.pocket_dir.glob("*_clean.pdb_predictions.csv")
    )
    if args.target_list is not None:
        wanted = {
            line.strip()
            for line in args.target_list.read_text().splitlines()
            if line.strip()
        }
        targets = [uid for uid in targets if uid in wanted]
    if not targets:
        raise PocketBoxError(f"no pocket predictions found under {args.pocket_dir}")

    written = 0
    exclusions: dict[str, str] = {}
    edges: list[float] = []
    volumes: list[float] = []
    clamped_low = clamped_high = 0
    for uid in targets:
        prediction = args.pocket_dir / f"{uid}_clean.pdb_predictions.csv"
        structure = args.clean_dir / f"{uid}_clean.pdb"
        if not structure.exists():
            exclusions[uid] = "missing_clean_structure"
            continue
        try:
            box = box_for_target(
                prediction,
                structure,
                headroom=args.headroom,
                min_edge=args.min_edge,
                max_edge=args.max_edge,
            )
        except PocketBoxError as exc:
            exclusions[uid] = str(exc)
            continue
        _write_atomic(args.out_dir / f"{uid}.box.txt", format_box_text(box))
        written += 1
        edges.extend(box.size)
        volumes.append(float(np.prod(box.size)))
        clamped_low += bool(box.clamped_low)
        clamped_high += bool(box.clamped_high)

    if not written:
        raise PocketBoxError(
            f"no target produced a box; {len(exclusions)} excluded, "
            f"first: {next(iter(exclusions.items()), None)}"
        )
    reference = float(args.reference_edge) ** 3
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "definition": (
            "box centred on the P2Rank rank-1 pocket centre and sized per axis to "
            "reach the outermost pocket surface atom plus headroom"
        ),
        "inputs": {
            "pocket_dir": str(args.pocket_dir),
            "clean_dir": str(args.clean_dir),
            "target_list": str(args.target_list) if args.target_list else None,
        },
        "out_box_dir": str(args.out_dir),
        "parameters": {
            "headroom_angstrom": float(args.headroom),
            "min_edge_angstrom": float(args.min_edge),
            "max_edge_angstrom": float(args.max_edge),
        },
        "counts": {
            "targets_considered": len(targets),
            "boxes_written": written,
            "excluded": len(exclusions),
            "clamped_to_min_edge": clamped_low,
            "clamped_to_max_edge": clamped_high,
        },
        "edge_angstrom": _percentiles(edges),
        "search_volume_advisory": {
            # Vina prints "Search space volume is greater than 27000 Angstrom^3"
            # and needs raised exhaustiveness above it. Measured on one receptor,
            # a 40,700 A^3 box returned a clashing +13.46 kcal/mol pose at
            # exhaustiveness 8 and a clean -5.24 at 32, better than the 30 A cube.
            "threshold_cubic_angstrom": float(args.advisory_volume),
            "boxes_above_threshold": int(sum(1 for v in volumes if v > args.advisory_volume)),
            "fraction_above_threshold": round(
                float(np.mean(np.asarray(volumes) > args.advisory_volume)), 3
            ),
            "guidance": (
                "raise --exhaustiveness for boxes above the threshold; the former "
                "uniform 30 A cube sat exactly at it for every receptor"
            ),
        },
        "volume_vs_reference_cube": {
            "reference_edge_angstrom": float(args.reference_edge),
            "median_ratio": round(float(np.median(volumes)) / reference, 3),
            "fraction_smaller": round(
                float(np.mean(np.asarray(volumes) < reference)), 3
            ),
        },
        "exclusion_reasons": dict(sorted(exclusions.items())[: args.max_reported_exclusions]),
    }
    _write_atomic(args.out_manifest, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if args.out_exclusions is not None:
        rows = ["uniprot,reason"]
        rows += [f"{uid},{reason.replace(',', ';')}" for uid, reason in sorted(exclusions.items())]
        _write_atomic(args.out_exclusions, "\n".join([*rows, ""]))
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pocket-dir", required=True, type=Path)
    parser.add_argument("--clean-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--out-manifest", required=True, type=Path)
    parser.add_argument("--out-exclusions", type=Path)
    parser.add_argument("--target-list", type=Path)
    parser.add_argument("--headroom", type=float, default=DEFAULT_HEADROOM)
    parser.add_argument("--min-edge", type=float, default=DEFAULT_MIN_EDGE)
    parser.add_argument("--max-edge", type=float, default=DEFAULT_MAX_EDGE)
    parser.add_argument(
        "--reference-edge",
        type=float,
        default=30.0,
        help="cube edge the derived volumes are compared against in the manifest",
    )
    parser.add_argument(
        "--advisory-volume",
        type=float,
        default=27000.0,
        help=(
            "search volume above which AutoDock Vina warns that default "
            "exhaustiveness is insufficient (27000 A^3 = a 30 A cube)"
        ),
    )
    parser.add_argument("--max-reported-exclusions", type=int, default=50)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not math.isfinite(args.reference_edge) or args.reference_edge <= 0:
        print("--reference-edge must be finite and > 0")
        return 1
    try:
        manifest = derive_all(args)
    except PocketBoxError as exc:
        print(str(exc))
        return 1
    counts = manifest["counts"]
    ratio = manifest["volume_vs_reference_cube"]
    print(
        f"[derive-boxes] wrote {counts['boxes_written']} boxes "
        f"({counts['excluded']} excluded); median edge "
        f"{manifest['edge_angstrom']['p50']:g} A; median volume "
        f"{ratio['median_ratio']:g}x the {ratio['reference_edge_angstrom']:g} A cube; "
        f"clamped low/high {counts['clamped_to_min_edge']}/{counts['clamped_to_max_edge']}; "
        f"{manifest['search_volume_advisory']['boxes_above_threshold']} above the "
        f"{manifest['search_volume_advisory']['threshold_cubic_angstrom']:g} A^3 "
        "search-volume advisory"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
