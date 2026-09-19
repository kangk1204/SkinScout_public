#!/usr/bin/env python3
"""eval/cosmetic_retrospective_eval.py — Did the pipeline recover the known
targets of well-known cosmetic ingredients? (INSTRUCTIONS §18.2)

Inputs:
  cases.csv — at minimum {inci_name, smiles, known_targets} where
              `known_targets` is `;`-separated UniProt accessions.
  rankings/ — directory of stage3 v3 rankings per case, named
              <inci_slug>__ranked_targets_v3.csv

Emits Top-1 / Top-5 / Top-10 recovery + a per-ingredient TSV.
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import pandas as pd
from rdkit import Chem

LOG = logging.getLogger("eval.cosmetic_retro")
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


DEFAULT_CASES = pd.DataFrame([
    {
        "inci_name": "Retinol",
        "smiles": "CC(C)=CCCC(C)=CCCC(C)=CCO",
        "known_targets": "P10276;P10826;P13631",
    },  # RARA/B/G
    {
        "inci_name": "Niacinamide",
        "smiles": "NC(=O)c1cccnc1",
        "known_targets": "P40261;Q96EB6;P09874",
    },  # NNMT / SIRT1 / PARP1
    {
        "inci_name": "Ascorbic_acid",
        "smiles": "O=C1OC(O)C(O)C1O",
        "known_targets": "P13674;Q02809",
    },  # P4HA1, P4HB
    {
        "inci_name": "alpha-Arbutin",
        "smiles": "OC[C@H]1O[C@@H](Oc2ccc(O)cc2)[C@H](O)[C@@H](O)[C@@H]1O",
        "known_targets": "P14679",
    },
    {
        "inci_name": "Kojic_acid",
        "smiles": "O=C1C(O)=C(CO)CO1",
        "known_targets": "P14679",
    },
    {
        "inci_name": "Salicylic_acid",
        "smiles": "O=C(O)c1ccccc1O",
        "known_targets": "P23219;P35354",
    },
    {
        "inci_name": "Resveratrol",
        "smiles": "Oc1ccc(/C=C/c2cc(O)cc(O)c2)cc1",
        "known_targets": "Q96EB6;Q16236",
    },
    {
        "inci_name": "EGCG",
        "smiles": (
            "O=C(O[C@H]1Cc2c(O)cc(O)c(O)c2O[C@@H]1c1ccc(O)c(O)c1)"
            "c1cc(O)c(O)c(O)c1"
        ),
        "known_targets": "P14780;P09874;P14679",
    },
    {
        "inci_name": "Caffeine",
        "smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
        "known_targets": "P29274;P29275",
    },
    {
        "inci_name": "Adenosine",
        "smiles": "Nc1ncnc2c1ncn2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O",
        "known_targets": "P30542;P29274;P29275",
    },
])


def _read_cases(path: Path | None) -> pd.DataFrame:
    if path is None:
        return DEFAULT_CASES.copy()
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(
            f"Cosmetic retrospective cases CSV is required and must be non-empty: {path}"
        )
    try:
        cases = pd.read_csv(path)
    except Exception as exc:
        raise SystemExit(
            f"Cosmetic retrospective cases CSV failed to parse: {path}: {exc}"
        ) from exc
    required = {"inci_name", "smiles", "known_targets"}
    missing = sorted(required - set(cases.columns))
    if missing:
        raise SystemExit(
            f"Cosmetic retrospective cases CSV missing required columns {missing}: {path}"
        )
    if cases.empty:
        raise SystemExit(f"Cosmetic retrospective cases CSV contains no rows: {path}")
    _validate_nonempty_strings(cases, "inci_name", "Cosmetic retrospective cases CSV")
    _validate_nonempty_strings(cases, "smiles", "Cosmetic retrospective cases CSV")
    _validate_nonempty_strings(cases, "known_targets", "Cosmetic retrospective cases CSV")
    _validate_unique_strings(cases, "inci_name", "Cosmetic retrospective cases CSV")
    _validate_smiles(cases, "smiles", "Cosmetic retrospective cases CSV")
    return cases


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


def _validate_unique_strings(df: pd.DataFrame, column: str, label: str) -> None:
    normalized = df[column].astype(str).str.strip()
    duplicate_ids = normalized[normalized.duplicated()].tolist()
    if duplicate_ids:
        shown = ", ".join(str(value) for value in duplicate_ids[:10])
        suffix = "..." if len(duplicate_ids) > 10 else ""
        raise SystemExit(
            f"{label} contains duplicate {column} values: {shown}{suffix}"
        )
    df[column] = normalized


def _validate_smiles(df: pd.DataFrame, column: str, label: str) -> None:
    invalid: list[int] = []
    normalized: list[str] = []
    for idx, value in df[column].items():
        text = str(value).strip()
        if Chem.MolFromSmiles(text) is None:
            invalid.append(int(idx))
        normalized.append(text)
    if invalid:
        shown = ", ".join(str(idx) for idx in invalid[:10])
        suffix = "..." if len(invalid) > 10 else ""
        raise SystemExit(
            f"{label} column '{column}' contains invalid SMILES at row index(es) "
            f"{shown}{suffix}"
        )
    df[column] = normalized


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


def _read_ranking(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(
            f"Cosmetic retrospective ranking is required and must be non-empty: {path}"
        )
    try:
        ranking = pd.read_csv(path)
    except Exception as exc:
        raise SystemExit(
            f"Cosmetic retrospective ranking failed to parse: {path}: {exc}"
        ) from exc
    if "target_id" not in ranking.columns:
        raise SystemExit(
            f"Cosmetic retrospective ranking missing required column 'target_id': {path}"
        )
    if ranking.empty:
        raise SystemExit(f"Cosmetic retrospective ranking contains no rows: {path}")
    _validate_nonempty_strings(
        ranking,
        "target_id",
        "Cosmetic retrospective ranking",
    )
    normalized_targets = ranking["target_id"].astype(str).str.strip()
    duplicate_ids = normalized_targets[normalized_targets.duplicated()].tolist()
    if duplicate_ids:
        shown = ", ".join(duplicate_ids[:10])
        suffix = "..." if len(duplicate_ids) > 10 else ""
        raise SystemExit(
            "Cosmetic retrospective ranking contains duplicate target_id "
            f"values: {shown}{suffix}: {path}"
        )
    ranking["target_id"] = normalized_targets
    return _ordered_ranking(ranking, path, "Cosmetic retrospective ranking")


def top_k_recovery(ranking: pd.DataFrame, known: set[str],
                   ks: tuple[int, ...] = (1, 5, 10)) -> dict[int, int]:
    out: dict[int, int] = {}
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
        out[k] = 1 if known & topk else 0
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-csv", type=Path, default=None,
                        help="CSV with inci_name, smiles, and known_targets; "
                             "falls back to the built-in list when absent.")
    parser.add_argument("--rankings-dir", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--min-mean-top10", type=float, default=0.01)
    parser.add_argument("--allow-missing-rankings", action="store_true",
                        help="write diagnostic no_ranking rows instead of "
                             "failing on missing rankings")
    parser.add_argument("--allow-threshold-failure", action="store_true",
                        help="write failing recovery metrics only for "
                             "explicit diagnostics")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.out_csv.exists():
        args.out_csv.unlink()
    _validate_fraction_threshold(args.min_mean_top10, "--min-mean-top10")
    cases = _read_cases(args.cases_csv)

    rows: list[dict] = []
    missing_rankings: list[Path] = []
    for _, c in cases.iterrows():
        inci = str(c["inci_name"]).strip()
        truth = _parse_semicolon_list(
            c["known_targets"],
            "Cosmetic retrospective cases CSV column 'known_targets'",
            inci,
        )
        if not truth:
            raise SystemExit(f"Cosmetic retrospective case has no known targets: {inci}")
        ranking_path = args.rankings_dir / f"{inci}__ranked_targets_v3.csv"
        if not ranking_path.exists():
            missing_rankings.append(ranking_path)
            rows.append({"inci_name": inci, "status": "no_ranking",
                         "top1": 0, "top5": 0, "top10": 0})
            continue
        ranking = _read_ranking(ranking_path)
        recovery = top_k_recovery(ranking, truth)
        rows.append({"inci_name": inci, "status": "evaluated",
                     "top1": recovery[1], "top5": recovery[5],
                     "top10": recovery[10],
                     "n_known": len(truth)})
    if missing_rankings and not args.allow_missing_rankings:
        preview = ", ".join(str(p) for p in missing_rankings[:5])
        raise SystemExit(
            f"Missing {len(missing_rankings)} cosmetic retrospective ranking file(s); "
            f"pass --allow-missing-rankings for diagnostics only: {preview}"
        )
    df = pd.DataFrame(rows)
    mean_top10 = float(df["top10"].mean())
    failures: list[str] = []
    if mean_top10 < args.min_mean_top10:
        failures.append(f"mean_top10={mean_top10:.3f} < {args.min_mean_top10:.3f}")
    if failures and not args.allow_threshold_failure:
        raise SystemExit(
            "Cosmetic retrospective failed claim threshold(s): "
            + "; ".join(failures)
            + "; pass --allow-threshold-failure only for explicit diagnostics"
        )
    df["passes_threshold"] = not failures
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv_atomic(df, args.out_csv)
    LOG.info("Cosmetic retrospective: Top-1=%.2f  Top-5=%.2f  Top-10=%.2f",
             df["top1"].mean(), df["top5"].mean(), df["top10"].mean())


if __name__ == "__main__":
    main()
