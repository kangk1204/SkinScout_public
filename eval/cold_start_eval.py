#!/usr/bin/env python3
"""eval/cold_start_eval.py — Cold-start protein evaluation (INSTRUCTIONS §14.2 F).

Compares COMPREHENSIVE vs FAST vs DTI-only target recovery on ChEMBL targets
with ≤ 5 known activities each.
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import pandas as pd

LOG = logging.getLogger("eval.cold_start")
RANK_COLUMNS = ("final_rank", "rank", "rank_skin", "target_rank", "global_rank")
SCORE_COLUMNS = (
    "final_skin_weighted",
    "final_score",
    "skin_weighted_score",
    "rrf_score",
    "score",
    "docking_rrf",
    "psichic_score",
    "pred",
    "predicted_affinity",
)


def _write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _read_required_csv(path: Path, required_cols: set[str], label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"{label} is required and must be non-empty: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise SystemExit(f"{label} failed to parse: {path}: {exc}") from exc
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise SystemExit(f"{label} missing required columns {missing}: {path}")
    if df.empty:
        raise SystemExit(f"{label} contains no rows: {path}")
    return df


def _read_required_parquet(path: Path, required_cols: set[str], label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"{label} is required and must be non-empty: {path}")
    try:
        df = pd.read_parquet(path, columns=list(required_cols))
    except Exception as exc:
        raise SystemExit(f"{label} failed to parse: {path}: {exc}") from exc
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise SystemExit(f"{label} missing required columns {missing}: {path}")
    if df.empty:
        raise SystemExit(f"{label} contains no rows: {path}")
    return df


def _validate_nonempty_strings(df: pd.DataFrame, column: str, label: str) -> None:
    invalid = [
        int(idx) for idx, value in df[column].items()
        if pd.isna(value) or not str(value).strip()
    ]
    if invalid:
        shown = ", ".join(str(idx) for idx in invalid[:10])
        suffix = "..." if len(invalid) > 10 else ""
        raise SystemExit(
            f"{label} column '{column}' contains blank values at row index(es) "
            f"{shown}{suffix}"
        )
    df[column] = df[column].astype(str).str.strip()


def _validate_unique_strings(df: pd.DataFrame, column: str, label: str) -> None:
    duplicate_ids = df[column][df[column].duplicated()].tolist()
    if duplicate_ids:
        shown = ", ".join(str(value) for value in duplicate_ids[:10])
        suffix = "..." if len(duplicate_ids) > 10 else ""
        raise SystemExit(
            f"{label} contains duplicate {column} values: {shown}{suffix}"
        )


def _validate_unique_pairs(
    df: pd.DataFrame,
    columns: tuple[str, str],
    label: str,
) -> None:
    duplicates = df[df.duplicated(list(columns), keep=False)]
    if duplicates.empty:
        return
    duplicate_pairs = duplicates.drop_duplicates(list(columns))
    pair_strings = [
        f"({row[columns[0]]}, {row[columns[1]]})"
        for _, row in duplicate_pairs.head(10).iterrows()
    ]
    suffix = "..." if len(duplicate_pairs) > 10 else ""
    raise SystemExit(
        f"{label} contains duplicate ({columns[0]}, {columns[1]}) pair(s): "
        f"{', '.join(pair_strings)}{suffix}"
    )


def _validate_fraction_threshold(value: float, label: str) -> None:
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise SystemExit(
            f"{label} must be a finite value in [0, 1]: {value}"
        )


def _numeric_series(df: pd.DataFrame, column: str, label: str, path: Path) -> pd.Series:
    if pd.api.types.is_bool_dtype(df[column].dtype) or df[column].map(
        lambda value: isinstance(value, bool)
    ).any():
        raise SystemExit(
            f"{label} column '{column}' must not contain boolean values: {path}"
        )
    values = pd.to_numeric(df[column], errors="coerce")
    invalid = values.isna() | ~values.map(math.isfinite)
    if invalid.any():
        first = int(invalid[invalid].index[0])
        raise SystemExit(
            f"{label} column '{column}' contains non-finite numeric value "
            f"at row index {first}: {path}"
        )
    return values


def _ordered_ranking(ranking: pd.DataFrame, path: Path, label: str) -> pd.DataFrame:
    for column in RANK_COLUMNS:
        if column not in ranking.columns:
            continue
        values = _numeric_series(ranking, column, label, path)
        if ((values % 1) != 0).any() or (values < 1).any():
            raise SystemExit(f"{label} column '{column}' must contain positive integer ranks: {path}")
        if values.duplicated().any():
            duplicate = values[values.duplicated()].iloc[0]
            raise SystemExit(f"{label} column '{column}' contains duplicate rank {int(duplicate)}: {path}")
        ranking = ranking.copy()
        ranking[column] = values.astype(int)
        return ranking.sort_values([column, "target_id"], ascending=[True, True]).reset_index(drop=True)
    for column in SCORE_COLUMNS:
        if column not in ranking.columns:
            continue
        values = _numeric_series(ranking, column, label, path)
        ranking = ranking.copy()
        ranking[column] = values
        return ranking.sort_values([column, "target_id"], ascending=[False, True]).reset_index(drop=True)
    raise SystemExit(
        f"{label} missing ranking order column; expected one of "
        f"{list(RANK_COLUMNS)} or score column {list(SCORE_COLUMNS)}: {path}"
    )


def cold_start_targets(chembl_dir: Path, max_actives: int = 5) -> list[str]:
    act_path = chembl_dir / "human_activities.parquet"
    df = _read_required_parquet(
        act_path,
        {"uniprot", "molecule_chembl_id"},
        "ChEMBL human activity reference",
    )
    _validate_nonempty_strings(df, "uniprot", "ChEMBL human activity reference")
    _validate_nonempty_strings(
        df,
        "molecule_chembl_id",
        "ChEMBL human activity reference",
    )
    _validate_unique_pairs(
        df,
        ("uniprot", "molecule_chembl_id"),
        "ChEMBL human activity reference",
    )
    counts = df.groupby("uniprot").size().reset_index(name="n_actives")
    cold = counts[counts["n_actives"] <= max_actives]["uniprot"].astype(str).tolist()
    if not cold:
        raise SystemExit(
            "ChEMBL human activity reference contains no cold-start targets "
            f"with <= {max_actives} known activities: {act_path}"
        )
    return cold


def topk_recovery(ranking: pd.DataFrame, ground_truth: set[str],
                  ks: tuple[int, ...] = (1, 5, 10, 50)) -> dict[int, float]:
    out: dict[int, float] = {}
    declared_rank = next(
        (column for column in RANK_COLUMNS if column in ranking.columns),
        None,
    )
    for k in ks:
        top_rows = (
            ranking[pd.to_numeric(ranking[declared_rank], errors="raise") <= k]
            if declared_rank is not None
            else ranking.head(k)
        )
        topk = set(top_rows["target_id"].astype(str))
        out[k] = len(topk & ground_truth) / max(1, len(ground_truth))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True,
                        choices=("comprehensive", "fast", "dti_only"))
    parser.add_argument("--ranking-csv", required=True, type=Path)
    parser.add_argument("--ground-truth-csv", required=True, type=Path,
                        help="CSV with column 'uniprot' giving the known targets")
    parser.add_argument("--chembl-dir", default=Path("data/chembl37"), type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--min-recall-at-50", type=float, default=0.01)
    parser.add_argument("--allow-threshold-failure", action="store_true",
                        help="write failing recovery metrics only for "
                             "explicit diagnostics")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.out_csv.exists():
        args.out_csv.unlink()
    _validate_fraction_threshold(args.min_recall_at_50, "--min-recall-at-50")
    ranking = _read_required_csv(args.ranking_csv, {"target_id"}, "Cold-start ranking")
    _validate_nonempty_strings(ranking, "target_id", "Cold-start ranking")
    _validate_unique_strings(ranking, "target_id", "Cold-start ranking")
    ranking = _ordered_ranking(ranking, args.ranking_csv, "Cold-start ranking")
    truth_df = _read_required_csv(
        args.ground_truth_csv,
        {"uniprot"},
        "Cold-start ground truth",
    )
    _validate_nonempty_strings(truth_df, "uniprot", "Cold-start ground truth")
    _validate_unique_strings(truth_df, "uniprot", "Cold-start ground truth")
    truth = truth_df["uniprot"].astype(str).tolist()
    cold = set(cold_start_targets(args.chembl_dir))
    truth_set = set(truth) & cold
    if not truth_set:
        raise SystemExit(
            "Cold-start ground truth has no overlap with ChEMBL cold-start "
            f"targets: {args.ground_truth_csv}"
        )

    recall = topk_recovery(ranking, truth_set)
    failures: list[str] = []
    recall_at_50 = float(recall[50])
    if recall_at_50 < args.min_recall_at_50:
        failures.append(f"recall@50={recall_at_50:.3f} < {args.min_recall_at_50:.3f}")
    if failures and not args.allow_threshold_failure:
        raise SystemExit(
            "Cold-start recovery failed claim threshold(s): "
            + "; ".join(failures)
            + "; pass --allow-threshold-failure only for explicit diagnostics"
        )
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv_atomic(pd.DataFrame([{
        "mode": args.mode,
        "n_cold_targets": len(cold),
        "n_truth_in_cold": len(truth_set),
        "passes_threshold": not failures,
        **{f"recall@{k}": v for k, v in recall.items()},
    }]), args.out_csv)
    LOG.info("Cold-start recall: %s", recall)


if __name__ == "__main__":
    main()
