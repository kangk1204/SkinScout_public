#!/usr/bin/env python3
"""eval/disagreement_eval.py — DTI vs Docking disagreement per protein class.

Splits agreement / disagreement sets per protein class:
    Kinase / GPCR / NR / ion-channel / protease / orphan.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

LOG = logging.getLogger("eval.disagreement")


def _write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def read_disagreement_payload(path: Path) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"Disagreement JSON is required and must be non-empty: {path}")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Disagreement JSON failed to parse: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Disagreement JSON must be an object: {path}")
    required = {"docking_only", "dti_only", "both_top"}
    missing = sorted(required - set(payload))
    if missing:
        raise SystemExit(f"Disagreement JSON missing required bucket(s) {missing}: {path}")
    n_targets = 0
    seen_targets: dict[str, str] = {}
    for bucket in required:
        targets = payload[bucket]
        if not isinstance(targets, list):
            raise SystemExit(
                f"Disagreement JSON bucket '{bucket}' must be a list: {path}"
            )
        invalid = [
            idx for idx, target_id in enumerate(targets)
            if not isinstance(target_id, str) or not target_id.strip()
        ]
        if invalid:
            shown = ", ".join(str(idx) for idx in invalid[:10])
            suffix = "..." if len(invalid) > 10 else ""
            raise SystemExit(
                "Disagreement JSON bucket target IDs must be non-empty strings: "
                f"{path} bucket={bucket!r} index(es)={shown}{suffix}"
            )
        for target_id in (str(target).strip() for target in targets):
            previous_bucket = seen_targets.get(target_id)
            if previous_bucket is not None:
                raise SystemExit(
                    "Disagreement JSON target IDs must be unique across buckets: "
                    f"{target_id} appears in {previous_bucket!r} and {bucket!r}: {path}"
                )
            seen_targets[target_id] = bucket
        n_targets += len(targets)
    if n_targets == 0:
        raise SystemExit(f"Disagreement JSON contains no target IDs: {path}")
    return payload


def read_class_map(path: Path) -> dict[str, str]:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(
            f"Target class reference is required and must be non-empty: {path}"
        )
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        raise SystemExit(f"Target class reference failed to parse: {path}: {exc}") from exc
    required = {"uniprot", "class"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(
            f"Target class reference missing required columns {missing}: {path}"
        )
    if df.empty:
        raise SystemExit(f"Target class reference contains no rows: {path}")
    for col in sorted(required):
        normalized = df[col].fillna("").astype(str).str.strip()
        blank_indexes = normalized[normalized == ""].index.tolist()
        if blank_indexes:
            shown = ", ".join(str(idx) for idx in blank_indexes[:10])
            suffix = "..." if len(blank_indexes) > 10 else ""
            raise SystemExit(
                f"Target class reference column '{col}' contains blank values "
                f"at row index(es) {shown}{suffix}: {path}"
            )
        df[col] = normalized
    duplicate_ids = df["uniprot"][df["uniprot"].duplicated()].tolist()
    if duplicate_ids:
        shown = ", ".join(str(value) for value in duplicate_ids[:10])
        suffix = "..." if len(duplicate_ids) > 10 else ""
        raise SystemExit(
            f"Target class reference contains duplicate uniprot values: "
            f"{shown}{suffix}: {path}"
        )
    return dict(zip(df["uniprot"].astype(str), df["class"].astype(str), strict=True))


def by_class(target_ids: set[str], cls_map: dict[str, str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for t in target_ids:
        c = cls_map.get(t, "__missing_target_class__")
        out[c] = out.get(c, 0) + 1
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disagreement-json", required=True, type=Path,
                        help="output of stage3_disagreement.py")
    parser.add_argument("--target-classes", default=Path("data/chembl37/target_classes.parquet"),
                        type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.out_csv.exists():
        args.out_csv.unlink()
    payload = read_disagreement_payload(args.disagreement_json)
    cls_map = read_class_map(args.target_classes)

    rows: list[dict] = []
    for bucket in ("docking_only", "dti_only", "both_top"):
        targets = set(payload[bucket])
        for cls, count in by_class(targets, cls_map).items():
            rows.append({"bucket": bucket, "class": cls, "count": count})
    if not rows:
        raise SystemExit(
            f"Disagreement evaluation produced no class rows: {args.disagreement_json}"
        )
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv_atomic(pd.DataFrame(rows), args.out_csv)
    LOG.info("Wrote %s (rows=%d)", args.out_csv, len(rows))


if __name__ == "__main__":
    main()
