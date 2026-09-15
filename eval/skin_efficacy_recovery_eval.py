#!/usr/bin/env python3
"""eval/skin_efficacy_recovery_eval.py — Compare KG efficacy labels against
manually curated ground truth (INSTRUCTIONS §18.3).
"""

from __future__ import annotations

import argparse
import logging
import math
import re
from pathlib import Path

import pandas as pd

LOG = logging.getLogger("eval.skin_efficacy")
EFFICACY_EVIDENCE_RE = re.compile(r"^(.+?)\s+\((\d+)\s+papers?\)$")

DEFAULT_GT = pd.DataFrame([
    {"inci_name": "Retinol",       "true_efficacies": "anti_aging;retinoid"},
    {"inci_name": "Niacinamide",   "true_efficacies": "whitening;anti_inflammatory"},
    {"inci_name": "Kojic_acid",    "true_efficacies": "whitening"},
    {"inci_name": "alpha-Arbutin", "true_efficacies": "whitening"},
    {"inci_name": "Salicylic_acid","true_efficacies": "acne"},
    {"inci_name": "Resveratrol",   "true_efficacies": "antioxidant;anti_aging"},
])


def precision_recall(predicted: set[str], truth: set[str]) -> tuple[float, float]:
    if not predicted or not truth:
        return 0.0, 0.0
    tp = len(predicted & truth)
    return tp / len(predicted), tp / len(truth)


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


def _validate_fraction_threshold(value: float, label: str) -> None:
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise SystemExit(
            f"{label} must be a finite value in [0, 1]: {value}"
        )


def _parse_semicolon_list(value: object, label: str, case_id: str) -> set[str]:
    tokens = [token.strip() for token in str(value).split(";")]
    if any(not token for token in tokens):
        raise SystemExit(
            f"{label} contains empty ';'-separated value(s): {case_id}"
        )
    duplicates = sorted({token for token in tokens if tokens.count(token) > 1})
    if duplicates:
        shown = ", ".join(duplicates[:10])
        suffix = "..." if len(duplicates) > 10 else ""
        raise SystemExit(
            f"{label} contains duplicate ';'-separated value(s): "
            f"{case_id}: {shown}{suffix}"
        )
    return set(tokens)


def _parse_efficacy_evidence(
    value: object,
    ranked_path: Path,
    row_idx: int,
    column: str,
) -> str:
    if pd.isna(value) or not str(value).strip():
        return ""
    text = str(value).strip()
    match = EFFICACY_EVIDENCE_RE.fullmatch(text)
    if not match:
        raise SystemExit(
            "Ranking efficacy_top* values must include paper-count evidence "
            f"as '<label> (N papers)' at row index {row_idx}, column '{column}': "
            f"{ranked_path}"
        )
    label = match.group(1).strip()
    n_papers = int(match.group(2))
    if not label or n_papers <= 0:
        raise SystemExit(
            "Ranking efficacy_top* values must include a non-empty label and "
            f"positive paper-count evidence at row index {row_idx}, column "
            f"'{column}': {ranked_path}"
        )
    return label


def _validate_efficacy_evidence(
    df: pd.DataFrame,
    efficacy_cols: list[str],
    ranked_path: Path,
) -> None:
    missing_rows: list[int] = []
    for idx, row in df[efficacy_cols].iterrows():
        labels = [
            _parse_efficacy_evidence(value, ranked_path, int(idx), str(column))
            for column, value in row.items()
        ]
        labels = [label for label in labels if label]
        if not labels:
            missing_rows.append(int(idx))
            continue
        duplicates = sorted({label for label in labels if labels.count(label) > 1})
        if duplicates:
            shown = ", ".join(duplicates[:10])
            suffix = "..." if len(duplicates) > 10 else ""
            raise SystemExit(
                "Ranking top-10 rows contain duplicate efficacy_top* "
                f"label(s) at row index {int(idx)}: {shown}{suffix}: {ranked_path}"
            )
    if missing_rows:
        shown = ", ".join(str(idx) for idx in missing_rows[:10])
        suffix = "..." if len(missing_rows) > 10 else ""
        raise SystemExit(
            "Ranking top-10 rows must each contain at least one non-empty "
            f"efficacy_top* value required for KG recovery; missing row "
            f"index(es) {shown}{suffix}: {ranked_path}"
        )


def _top_ranked_rows(df: pd.DataFrame, ranked_path: Path) -> pd.DataFrame:
    """Select top ten from declared ranking semantics when available."""
    if "rank" in df.columns:
        ranks = pd.to_numeric(df["rank"], errors="coerce")
        if ranks.isna().any() or (ranks < 1).any() or ranks.duplicated().any():
            raise SystemExit(f"Ranking column 'rank' must contain unique positive numbers: {ranked_path}")
        return df.assign(_rank=ranks).sort_values(
            ["_rank", "target_id"], kind="mergesort"
        ).drop(columns="_rank").head(10)
    if "final_score" in df.columns:
        scores = pd.to_numeric(df["final_score"], errors="coerce")
        if scores.isna().any() or not scores.map(math.isfinite).all():
            raise SystemExit(f"Ranking final_score must be finite numeric values: {ranked_path}")
        return df.assign(_score=scores).sort_values(
            ["_score", "target_id"], ascending=[False, True], kind="mergesort"
        ).drop(columns="_score").head(10)
    # Old diagnostic fixtures predate explicit rank/score columns. Production
    # artifacts always carry final_score; retain positional handling only for
    # those legacy diagnostics.
    return df.head(10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranked-dir", required=True, type=Path,
                        help="Dir with <inci>__ranked_targets_v3_with_efficacy.csv")
    parser.add_argument("--ground-truth-csv", type=Path, default=None)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--min-mean-precision", type=float, default=0.01)
    parser.add_argument("--min-mean-recall", type=float, default=0.01)
    parser.add_argument(
        "--allow-missing-rankings",
        action="store_true",
        help="write diagnostic no_ranking rows instead of failing on missing rankings",
    )
    parser.add_argument(
        "--allow-threshold-failure",
        action="store_true",
        help="write failing recovery metrics only for explicit diagnostics",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.out_csv.exists():
        args.out_csv.unlink()
    _validate_fraction_threshold(args.min_mean_precision, "--min-mean-precision")
    _validate_fraction_threshold(args.min_mean_recall, "--min-mean-recall")
    required = {"inci_name", "true_efficacies"}
    gt = (
        _read_required_csv(args.ground_truth_csv, required, "Ground truth")
        if args.ground_truth_csv
        else DEFAULT_GT.copy()
    )
    _validate_nonempty_strings(gt, "inci_name", "Ground truth")
    _validate_nonempty_strings(gt, "true_efficacies", "Ground truth")
    _validate_unique_strings(gt, "inci_name", "Ground truth")
    rows: list[dict] = []
    missing_rankings: list[Path] = []
    for _, g in gt.iterrows():
        inci = str(g["inci_name"]).strip()
        truth = _parse_semicolon_list(
            g["true_efficacies"],
            "Ground truth column 'true_efficacies'",
            inci,
        )
        if not truth:
            raise SystemExit(f"Ground truth case has no true efficacies: {inci}")
        ranked_path = args.ranked_dir / f"{inci}__ranked_targets_v3_with_efficacy.csv"
        if not ranked_path.exists():
            missing_rankings.append(ranked_path)
            rows.append({"inci_name": inci, "status": "no_ranking",
                         "precision": 0.0, "recall": 0.0})
            continue
        df = _read_required_csv(ranked_path, {"target_id"}, "Ranking")
        _validate_nonempty_strings(df, "target_id", "Ranking")
        _validate_unique_strings(df, "target_id", "Ranking")
        df = _top_ranked_rows(df, ranked_path)
        efficacy_cols = [col for col in df.columns if col.startswith("efficacy_top")]
        if not efficacy_cols:
            raise SystemExit(
                f"Ranking missing efficacy_top* columns required for KG recovery: {ranked_path}"
            )
        _validate_efficacy_evidence(df, efficacy_cols, ranked_path)
        predicted: set[str] = set()
        for row_idx, row in df[efficacy_cols].iterrows():
            for col, cell in row.items():
                name = _parse_efficacy_evidence(
                    cell,
                    ranked_path,
                    int(row_idx),
                    str(col),
                )
                if name:
                    predicted.add(name)
        prec, rec = precision_recall(predicted, truth)
        rows.append({"inci_name": inci, "status": "evaluated",
                     "precision": prec, "recall": rec})
    if missing_rankings and not args.allow_missing_rankings:
        preview = ", ".join(str(p) for p in missing_rankings[:5])
        raise SystemExit(
            f"Missing {len(missing_rankings)} ranking file(s); "
            f"pass --allow-missing-rankings for diagnostics only: {preview}"
        )
    df_out = pd.DataFrame(rows)
    mean_precision = float(df_out["precision"].mean())
    mean_recall = float(df_out["recall"].mean())
    failures: list[str] = []
    if mean_precision < args.min_mean_precision:
        failures.append(f"mean_precision={mean_precision:.3f} < {args.min_mean_precision:.3f}")
    if mean_recall < args.min_mean_recall:
        failures.append(f"mean_recall={mean_recall:.3f} < {args.min_mean_recall:.3f}")
    if failures and not args.allow_threshold_failure:
        raise SystemExit(
            "Skin-efficacy recovery failed claim threshold(s): "
            + "; ".join(failures)
            + "; pass --allow-threshold-failure only for explicit diagnostics"
        )
    df_out["passes_threshold"] = not failures
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = args.out_csv.with_suffix(args.out_csv.suffix + ".tmp")
    df_out.to_csv(tmp_csv, index=False)
    tmp_csv.replace(args.out_csv)
    LOG.info("Skin-efficacy recovery: mean precision=%.2f, recall=%.2f",
             df_out["precision"].mean(), df_out["recall"].mean())


if __name__ == "__main__":
    main()
