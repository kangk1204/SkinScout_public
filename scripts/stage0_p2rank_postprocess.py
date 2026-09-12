#!/usr/bin/env python3
"""stage0_p2rank_postprocess.py

Convert P2Rank's per-PDB output CSVs into per-UniProt JSON manifests, and emit
a list of receptors with no usable pocket (which feed into DiffDock-L blind
docking at Stage 3 per INSTRUCTIONS.md §6.4).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path

LOG = logging.getLogger("stage0.p2rank.post")

PRED_FNAME_RE = re.compile(r"(?P<uid>[A-Z0-9]+)_clean\.pdb_predictions\.csv$")
INPUT_FNAME_RE = re.compile(r"(?P<uid>[A-Z0-9]+)_clean\.pdb$")


@dataclass(frozen=True)
class Pocket:
    rank: int
    score: float
    druggability: float
    center: tuple[float, float, float]
    radius: float

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "Pocket":
        return cls(
            rank=_positive_int(row.get("rank") or row.get("pocket_rank") or 0, "rank"),
            score=_nonnegative_float(row.get("score", "0") or 0, "score"),
            druggability=_finite_float(
                row.get("druggability_score", row.get("score", "0")) or 0,
                "druggability_score",
            ),
            center=(
                _finite_float(row.get("center_x", "0") or 0, "center_x"),
                _finite_float(row.get("center_y", "0") or 0, "center_y"),
                _finite_float(row.get("center_z", "0") or 0, "center_z"),
            ),
            radius=_positive_float(
                row.get("radius", row.get("pocket_radius", "12")) or 12,
                "radius",
            ),
        )


def _is_bool_like(value: object) -> bool:
    return (
        isinstance(value, bool)
        or type(value).__name__ == "bool_"
        or (isinstance(value, str) and value.strip().lower() in {"true", "false"})
    )


def _finite_float(value: object, label: str) -> float:
    if _is_bool_like(value):
        raise ValueError(f"{label} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    return parsed


def _nonnegative_float(value: object, label: str) -> float:
    parsed = _finite_float(value, label)
    if parsed < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return parsed


def _positive_float(value: object, label: str) -> float:
    parsed = _finite_float(value, label)
    if parsed <= 0.0:
        raise ValueError(f"{label} must be positive")
    return parsed


def _positive_int(value: object, label: str) -> int:
    parsed = _finite_float(value, label)
    if parsed < 1 or not parsed.is_integer():
        raise ValueError(f"{label} must be a positive integer")
    return int(parsed)


def parse_pred_csv(csv_path: Path) -> list[Pocket]:
    # P2Rank predictions.csv pads both column headers AND values with leading
    # spaces (e.g. "   score", "   14.93"). Strip keys and values so .get()
    # lookups in Pocket.from_row resolve correctly — otherwise every pocket is
    # silently dropped and every protein is misclassified as "no-pocket".
    pockets: list[Pocket] = []
    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row_number, raw in enumerate(reader, start=1):
            row = {
                (k.strip() if k else k): (v.strip() if isinstance(v, str) else v)
                for k, v in raw.items()
            }
            try:
                pockets.append(Pocket.from_row(row))
            except (ValueError, KeyError) as exc:
                raise SystemExit(
                    f"Invalid P2Rank prediction row {row_number} in "
                    f"{csv_path.name}: {exc}"
                ) from exc
    return pockets


def remove_outputs(*paths: Path) -> None:
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text)
    tmp.replace(path)


def write_json_atomic(path: Path, payload: dict) -> None:
    write_text_atomic(path, json.dumps(payload))


def expected_uids_from_input_list(path: Path) -> set[str]:
    expected: set[str] = set()
    with path.open() as fh:
        for line_number, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            m = INPUT_FNAME_RE.search(Path(text).name)
            if not m:
                raise SystemExit(
                    f"Invalid P2Rank input list row {line_number}: {text!r}"
                )
            expected.add(m.group("uid"))
    if not expected:
        raise SystemExit(f"P2Rank input list is empty: {path}")
    return expected


def prediction_csvs_by_uid(pocket_dir: Path) -> dict[str, list[Path]]:
    by_uid: dict[str, list[Path]] = {}
    for csv_path in sorted(pocket_dir.rglob("*_predictions.csv")):
        m = PRED_FNAME_RE.search(csv_path.name)
        if not m:
            continue
        by_uid.setdefault(m.group("uid"), []).append(csv_path)
    return by_uid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pocket-dir", required=True, type=Path)
    parser.add_argument("--no-pocket-out", required=True, type=Path)
    parser.add_argument("--min-score", type=float, default=1.0,
                        help="Score below which the pocket is treated as no-pocket.")
    parser.add_argument("--top-n", type=int, default=3,
                        help="Retain the top-N pockets per receptor.")
    parser.add_argument("--min-processed-count", type=int, default=1)
    parser.add_argument("--min-usable-fraction", type=float, default=0.5)
    parser.add_argument(
        "--expected-input-list",
        type=Path,
        help="P2Rank proteins.ds used for the run; all listed receptors must have CSVs.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    no_pocket: list[str] = []
    manifests: list[tuple[Path, dict]] = []
    processed_uids: set[str] = set()
    n_with = n_without = 0
    n_processed = 0
    remove_outputs(args.no_pocket_out)
    csvs_by_uid = prediction_csvs_by_uid(args.pocket_dir)
    duplicate_uids = {
        uid: paths for uid, paths in csvs_by_uid.items() if len(paths) > 1
    }
    if duplicate_uids:
        for uid in duplicate_uids:
            remove_outputs(args.pocket_dir / f"{uid}.pockets.json")
        preview = "; ".join(
            f"{uid}: {', '.join(str(path) for path in paths[:3])}"
            for uid, paths in sorted(duplicate_uids.items())[:3]
        )
        raise SystemExit(
            "Duplicate P2Rank prediction CSVs for receptor UID(s): "
            f"{preview}"
        )
    if not csvs_by_uid:
        raise SystemExit("No P2Rank prediction CSV files were processed")
    for uid, paths in sorted(csvs_by_uid.items()):
        csv_path = paths[0]
        processed_uids.add(uid)
        n_processed += 1
        out = args.pocket_dir / f"{uid}.pockets.json"
        remove_outputs(out)
        pockets = parse_pred_csv(csv_path)
        usable = [p for p in pockets if p.score >= args.min_score]
        usable = sorted(usable, key=lambda p: -p.score)[: args.top_n]
        manifests.append((
            out,
            {
                "uniprot": uid,
                "pockets": [p.__dict__ for p in usable],
            },
        ))

        if usable:
            n_with += 1
        else:
            n_without += 1
            no_pocket.append(uid)

    if args.expected_input_list is not None:
        expected_uids = expected_uids_from_input_list(args.expected_input_list)
        for uid in expected_uids:
            if uid not in processed_uids:
                remove_outputs(args.pocket_dir / f"{uid}.pockets.json")
        missing = sorted(expected_uids - processed_uids)
        extras = sorted(processed_uids - expected_uids)
        if missing or extras:
            raise SystemExit(
                "P2Rank postprocess did not match input receptor list: "
                f"{len(missing)} missing, {len(extras)} unexpected"
            )
    if n_processed < args.min_processed_count:
        raise SystemExit(
            "P2Rank postprocess did not meet processed-count gate: "
            f"{n_processed} processed; required at least {args.min_processed_count}"
        )
    usable_fraction = n_with / n_processed
    if usable_fraction < args.min_usable_fraction:
        raise SystemExit(
            "P2Rank postprocess did not meet usable-pocket gate: "
            f"{n_with}/{n_processed} with usable pockets "
            f"({usable_fraction:.3f}); required fraction >= "
            f"{args.min_usable_fraction:.3f}"
        )

    for out, payload in manifests:
        write_json_atomic(out, payload)
    write_text_atomic(args.no_pocket_out, "\n".join(sorted(no_pocket)) + "\n")
    LOG.info("P2Rank prediction CSVs processed: %d", n_processed)
    LOG.info("Receptors with usable pocket: %d", n_with)
    LOG.info("Receptors flagged for blind docking: %d → %s",
             n_without, args.no_pocket_out)


if __name__ == "__main__":
    main()
