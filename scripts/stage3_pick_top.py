#!/usr/bin/env python3
"""stage3_pick_top.py — Pick top-X% of receptors after AutoDock-GPU 전수.

Merges AutoDock-GPU scores (vina/Vinardo energy, lower=better) with the
DiffDock-L blind scores for the no-pocket subset, then returns the top fraction
by a source-normalized rank score (higher=better).
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import pandas as pd

LOG = logging.getLogger("stage3.pick_top")


def _remove_outputs(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def _nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _invalid_row_message(source: str, column: str, rows: list[int], detail: str) -> str:
    shown = ", ".join(str(idx) for idx in rows[:10])
    suffix = "..." if len(rows) > 10 else ""
    return (
        f"{source} score file column '{column}' contains {detail} at row "
        f"index(es) {shown}{suffix}"
    )


def _duplicate_target_message(values: list[str]) -> str:
    shown = ", ".join(values[:10])
    suffix = "..." if len(values) > 10 else ""
    return f"Top-pick score files contain duplicate target_id values: {shown}{suffix}"


def _bool_like_indexes(series: pd.Series) -> list[int]:
    return [
        int(idx)
        for idx, value in series.items()
        if (
            isinstance(value, bool)
            or type(value).__name__ == "bool_"
            or (
                isinstance(value, str)
                and value.strip().lower() in {"true", "false"}
            )
        )
    ]


def _validate_top_pct(top_pct: float) -> None:
    if not math.isfinite(top_pct) or not 0.0 < top_pct <= 1.0:
        raise SystemExit(f"--top-pct must be a finite value in (0, 1]: {top_pct:g}")


def _read_score_file(path: Path, *, sep: str, source: str) -> pd.DataFrame | None:
    if not _nonempty(path):
        if source == "autodock":
            raise SystemExit(f"{source} score file is required and must be non-empty: {path}")
        return None

    df = pd.read_csv(path, sep=sep)
    score_col = _score_column(df, source)
    missing = {"target_id", score_col} - set(df.columns)
    if missing:
        raise SystemExit(
            f"{source} score file missing required columns: {', '.join(sorted(missing))}"
        )
    if df.empty and source == "diffdock_blind":
        return None
    if df.empty:
        raise SystemExit(f"{source} score file contains no usable rows: {path}")
    blank_targets = [
        int(idx)
        for idx, value in df["target_id"].items()
        if pd.isna(value) or not str(value).strip()
    ]
    if blank_targets:
        raise SystemExit(
            _invalid_row_message(source, "target_id", blank_targets, "blank values")
        )
    bool_scores = _bool_like_indexes(df[score_col])
    if bool_scores:
        raise SystemExit(
            _invalid_row_message(
                source,
                score_col,
                bool_scores,
                "invalid values",
            )
        )
    scores = pd.to_numeric(df[score_col], errors="coerce")
    invalid_scores = [
        int(idx)
        for idx, value in scores.items()
        if pd.isna(value) or not math.isfinite(float(value))
    ]
    if invalid_scores:
        raise SystemExit(
            _invalid_row_message(
                source,
                score_col,
                invalid_scores,
                "invalid values",
            )
        )
    df["target_id"] = df["target_id"].astype(str).str.strip()
    df["raw_score"] = scores.astype(float)
    lower_is_better = source == "autodock" and score_col != "neg_vina_score"
    df["score_direction"] = "lower_is_better" if lower_is_better else "higher_is_better"
    ranks = df["raw_score"].rank(
        method="min",
        ascending=lower_is_better,
    )
    denom = max(len(df) - 1, 1)
    df["score"] = 1.0 - ((ranks - 1) / denom)
    df["source"] = source
    return df[["target_id", "score", "raw_score", "score_direction", "source"]]


def _score_column(df: pd.DataFrame, source: str) -> str:
    if source == "autodock":
        for column in ("vina_score", "binding_energy", "affinity", "energy"):
            if column in df.columns:
                return column
        if "neg_vina_score" in df.columns:
            return "neg_vina_score"
        if "score" in df.columns:
            return "score"
    else:
        for column in ("confidence", "diffdock_confidence", "score", "neg_vina_score"):
            if column in df.columns:
                return column
    return "neg_vina_score"


def _write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--autodock-scores", required=True, type=Path)
    parser.add_argument("--blind-scores", required=True, type=Path)
    parser.add_argument("--top-pct", type=float, default=0.02)
    parser.add_argument("--out-csv", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    _remove_outputs(args.out_csv)
    _validate_top_pct(args.top_pct)
    frames: list[pd.DataFrame] = []
    a = _read_score_file(args.autodock_scores, sep="\t", source="autodock")
    if a is not None:
        frames.append(a)
    b = _read_score_file(args.blind_scores, sep="\t", source="diffdock_blind")
    if b is not None:
        frames.append(b)
    if not frames:
        raise SystemExit("No score files present")

    merged = pd.concat(frames, ignore_index=True)
    if merged.empty:
        raise SystemExit("Score files are present but contain no usable rows")
    duplicate_targets = merged["target_id"][merged["target_id"].duplicated()].tolist()
    if duplicate_targets:
        raise SystemExit(_duplicate_target_message(duplicate_targets))
    merged = merged.sort_values(["score", "target_id"], ascending=[False, True]).reset_index(drop=True)

    n_keep = min(len(merged), max(50, int(round(len(merged) * args.top_pct))))
    out = merged.head(n_keep)
    _write_csv_atomic(out, args.out_csv)
    LOG.info("Picked top %d / %d (=%.2f%%) → %s",
             n_keep, len(merged), 100 * n_keep / len(merged), args.out_csv)


if __name__ == "__main__":
    main()
