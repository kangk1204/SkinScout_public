#!/usr/bin/env python3
"""stage3_skin_weighting.py — Final target score with skin-expression and known-target priors.

Formula (INSTRUCTIONS.md §7.2):
    final = w_dock × normalize(docking_rrf) + w_skin × skin_score
            + w_prior × normalize(known_target_prior)
        where w_dock = 1 - w_skin, default w_skin = 0.30
    when w_prior > 0, w_dock = 1 - w_skin - w_prior
    if skin_score < min_threshold: final *= 0.30  (heavy penalty)

Prior scores can be filtered or re-shaped before normalization:
    - --known-target-prior-min-score drops prior evidence below a floor.
    - --known-target-prior-power applies prior ** power to smooth/peak evidence.

The pure `apply_skin_weight()` helper is unit-tested.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import pandas as pd

from cosmetic_drug_contract import (
    CosmeticDrugContractError,
    parse_decision_and_policy,
    validate_decision_matches_drug_warnings,
)

LOG = logging.getLogger("stage3.skin_weight")
DOCKING_SCORE_COLUMNS = ("rrf_score", "final_score", "score")
KNOWN_PRIOR_COLUMNS = ("target_id", "prior_score")


def _remove_outputs(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def _nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _read_json_object(path: Path, label: str) -> dict:
    if not _nonempty(path):
        raise SystemExit(f"{label} is required and must be non-empty: {path}")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} must be a JSON object: {path}")
    return payload


def _write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo < 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def _normalize_non_constant_zero(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo < 1e-12:
        if hi <= 0.0:
            return [0.0 for _ in values]
        return [1.0 if value > 0 else 0.0 for value in values]
    return [(v - lo) / (hi - lo) for v in values]


def _preserved_rank_order(top: pd.DataFrame) -> pd.DataFrame:
    """Validate and order an upstream reader-facing ranking by ``final_rank``."""
    if "final_rank" not in top.columns:
        raise SystemExit(
            "--preserve-primary-ranking requires upstream column 'final_rank'"
        )
    bool_ranks = _bool_like_indexes(top["final_rank"])
    ranks = pd.to_numeric(top["final_rank"], errors="coerce")
    invalid = [
        int(idx)
        for idx, value in ranks.items()
        if (
            int(idx) in bool_ranks
            or pd.isna(value)
            or not math.isfinite(float(value))
            or not float(value).is_integer()
        )
    ]
    if invalid:
        raise SystemExit(
            _invalid_row_message(
                "Top-target CSV", "final_rank", invalid, "invalid values"
            )
        )
    integer_ranks = ranks.astype(int)
    expected = list(range(1, len(top) + 1))
    if sorted(integer_ranks.tolist()) != expected:
        raise SystemExit(
            "Top-target CSV column 'final_rank' must contain each dense rank "
            f"from 1 through {len(top)} exactly once"
        )
    ordered = top.assign(final_rank=integer_ranks).sort_values(
        "final_rank", kind="mergesort"
    )
    return ordered.reset_index(drop=True)


def _prepare_prior_scores(
    prior_scores: dict[str, float],
    min_score: float,
    power: float,
) -> dict[str, float]:
    if not prior_scores:
        return {}
    filtered = prior_scores
    if min_score > 0.0:
        filtered = {
            target: score
            for target, score in filtered.items()
            if score >= min_score
        }
    if power == 1.0:
        return filtered
    shaped: dict[str, float] = {}
    for target, score in filtered.items():
        value = float(score)
        if not math.isfinite(value):
            raise SystemExit(
                f"Known-target prior score must be finite: {target}={score!r}"
            )
        # Clamp before exponentiating: a negative base with a fractional power
        # returns a complex number in Python, which then fails comparison.
        base = min(1.0, max(0.0, value))
        shaped[target] = min(1.0, max(0.0, base ** power))
    return shaped


def _invalid_row_message(source: str, column: str, rows: list[int], detail: str) -> str:
    shown = ", ".join(str(idx) for idx in rows[:10])
    suffix = "..." if len(rows) > 10 else ""
    return (
        f"{source} column '{column}' contains {detail} at row "
        f"index(es) {shown}{suffix}"
    )


def _duplicate_value_message(source: str, column: str, values: list[str]) -> str:
    shown = ", ".join(values[:10])
    suffix = "..." if len(values) > 10 else ""
    return f"{source} column '{column}' contains duplicate values: {shown}{suffix}"


def _source_labels(value: object, source: str, row_idx: int) -> list[str]:
    if pd.isna(value) or not str(value).strip():
        raise SystemExit(
            _invalid_row_message(source, "sources", [row_idx], "blank values")
        )
    labels = [part.strip() for part in str(value).split(";")]
    if any(label == "" for label in labels):
        raise SystemExit(
            _invalid_row_message(
                source,
                "sources",
                [row_idx],
                "empty source labels",
            )
        )
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise SystemExit(
            f"{source} column 'sources' contains duplicate labels at row "
            f"index {row_idx}: {', '.join(duplicates[:10])}"
        )
    return labels


def _validate_source_rationale(df: pd.DataFrame, source: str) -> None:
    if "source_count" not in df.columns:
        return
    if "sources" not in df.columns:
        raise SystemExit(
            f"{source} missing required column 'sources' when source_count is present"
        )
    for idx, row in df.iterrows():
        labels = _source_labels(row["sources"], source, int(idx))
        source_count = int(row["source_count"])
        if source_count != len(labels):
            raise SystemExit(
                f"{source} source_count={source_count} but sources lists "
                f"{len(labels)} label(s) at row index {int(idx)}"
            )


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


def validate_parameters(
    cosmetic_downweight_factor: float,
    skin_weight: float,
    known_target_prior_weight: float,
    known_target_prior_min_score: float,
    known_target_prior_power: float,
    min_threshold: float,
    min_source_count: int,
) -> None:
    for name, value in (
        ("--skin-weight", skin_weight),
        ("--min-threshold", min_threshold),
    ):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise SystemExit(f"{name} must be a finite value in [0, 1]: {value:g}")
    if (
        not math.isfinite(cosmetic_downweight_factor)
        or not 0.0 < cosmetic_downweight_factor <= 1.0
    ):
        raise SystemExit(
            "--cosmetic-downweight-factor must be a finite value in (0, 1]: "
            f"{cosmetic_downweight_factor:g}"
        )
    if (
        not math.isfinite(known_target_prior_weight)
        or not 0.0 <= known_target_prior_weight <= 1.0
    ):
        raise SystemExit(
            "--known-target-prior-weight must be a finite value in [0, 1]: "
            f"{known_target_prior_weight:g}"
        )
    if skin_weight + known_target_prior_weight > 1.0:
        raise SystemExit(
            "The sum of --skin-weight and --known-target-prior-weight "
            "must be <= 1.0"
        )
    if (
        not math.isfinite(known_target_prior_min_score)
        or not 0.0 <= known_target_prior_min_score <= 1.0
    ):
        raise SystemExit(
            "--known-target-prior-min-score must be a finite value in [0, 1]: "
            f"{known_target_prior_min_score:g}"
        )
    if (
        not math.isfinite(known_target_prior_power)
        or known_target_prior_power <= 0.0
    ):
        raise SystemExit(
            "--known-target-prior-power must be a finite value > 0: "
            f"{known_target_prior_power:g}"
        )
    if min_source_count < 1:
        raise SystemExit(f"--min-source-count must be >= 1: {min_source_count}")


def apply_skin_weight(
    docking_scores: dict[str, float],
    skin_scores: dict[str, float],
    skin_weight: float = 0.30,
    known_target_prior_scores: dict[str, float] | None = None,
    known_target_prior_weight: float = 0.0,
    known_target_prior_min_score: float = 0.0,
    known_target_prior_power: float = 1.0,
    min_threshold: float = 0.05,
    penalty_factor: float = 0.30,
) -> dict[str, float]:
    """Pure helper — unit-tested.

    Args:
        docking_scores: target_id → RRF (any positive scale).
        skin_scores:    target_id → skin_score in [0, 1] (default 0 for missing).
        skin_weight:    fraction allocated to skin (0.0–1.0).
        min_threshold:  below this skin_score the final value is multiplied by
                        penalty_factor.
        penalty_factor: scalar applied when skin_score < min_threshold.

    Returns:
        target_id → final score (descending = better).
    """
    if not docking_scores:
        return {}
    ids = list(docking_scores.keys())
    vals = [docking_scores[t] for t in ids]
    norm = _minmax(vals)
    prior_scores = _prepare_prior_scores(
        known_target_prior_scores or {},
        min_score=known_target_prior_min_score,
        power=known_target_prior_power,
    )
    prior_norm = _normalize_non_constant_zero(
        [prior_scores.get(t, 0.0) for t in ids]
    )
    w_dock = 1.0 - skin_weight - known_target_prior_weight
    out: dict[str, float] = {}
    for n_dock, n_prior, t in zip(norm, prior_norm, ids, strict=True):
        skin = float(skin_scores.get(t, 0.0))
        final = (
            w_dock * n_dock
            + skin_weight * skin
            + known_target_prior_weight * n_prior
        )
        if skin < min_threshold:
            final *= penalty_factor
        out[t] = final
    return out


def read_decision(path: Path | None, drug_path: Path | None = None) -> str:
    if path is None:
        raise SystemExit("Cosmetic/drug decision file is required")
    if drug_path is None:
        raise SystemExit("Drug-avoidance warnings JSON is required")
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(
            f"Cosmetic/drug decision file is required and must be non-empty: {path}"
        )
    try:
        decision, policy = parse_decision_and_policy(
            path.read_text(),
            label="Cosmetic/drug decision",
        )
        validate_decision_matches_drug_warnings(
            decision=decision,
            policy=policy,
            drug_warnings=_read_json_object(drug_path, "Drug-avoidance warnings"),
        )
    except CosmeticDrugContractError as exc:
        raise SystemExit(str(exc)) from exc
    return decision


def read_skin_scores(path: Path) -> pd.DataFrame:
    if not _nonempty(path):
        raise SystemExit(f"Skin-expression score table is required and must be non-empty: {path}")
    skin = pd.read_csv(path, sep="\t")
    required = {"uniprot", "skin_score"}
    missing = sorted(required - set(skin.columns))
    if missing:
        raise SystemExit(
            "Skin-expression score table missing required column(s): "
            + ", ".join(missing)
        )
    if skin.empty:
        raise SystemExit(f"Skin-expression score table contains no rows: {path}")
    blank_uniprots = [
        int(idx)
        for idx, value in skin["uniprot"].items()
        if pd.isna(value) or not str(value).strip()
    ]
    if blank_uniprots:
        raise SystemExit(
            _invalid_row_message(
                "Skin-expression score table",
                "uniprot",
                blank_uniprots,
                "blank values",
            )
        )
    bool_scores = _bool_like_indexes(skin["skin_score"])
    if bool_scores:
        raise SystemExit(
            _invalid_row_message(
                "Skin-expression score table",
                "skin_score",
                bool_scores,
                "invalid values",
            )
        )
    scores = pd.to_numeric(skin["skin_score"], errors="coerce")
    invalid_scores = [
        int(idx)
        for idx, value in scores.items()
        if pd.isna(value) or not math.isfinite(float(value))
    ]
    if invalid_scores:
        raise SystemExit(
            _invalid_row_message(
                "Skin-expression score table",
                "skin_score",
                invalid_scores,
                "invalid values",
            )
        )
    skin["uniprot"] = skin["uniprot"].astype(str).str.strip()
    duplicate_uniprots = skin["uniprot"][skin["uniprot"].duplicated()].tolist()
    if duplicate_uniprots:
        raise SystemExit(
            _duplicate_value_message(
                "Skin-expression score table",
                "uniprot",
                duplicate_uniprots,
            )
        )
    skin["skin_score"] = scores.astype(float)
    out_of_range_scores = [
        int(idx)
        for idx, value in skin["skin_score"].items()
        if not 0.0 <= float(value) <= 1.0
    ]
    if out_of_range_scores:
        raise SystemExit(
            _invalid_row_message(
                "Skin-expression score table",
                "skin_score",
                out_of_range_scores,
                "values outside [0, 1]",
            )
        )
    if "cell_type_preferred" in skin.columns:
        cell_types = skin["cell_type_preferred"].fillna("").astype(str).str.strip()
        blank_cell_types = cell_types.index[cell_types == ""].tolist()
        if blank_cell_types:
            raise SystemExit(
                _invalid_row_message(
                    "Skin-expression score table",
                    "cell_type_preferred",
                    [int(index) for index in blank_cell_types],
                    "blank values",
                )
            )
        skin["cell_type_preferred"] = cell_types
    else:
        skin["cell_type_preferred"] = "unknown"
    return skin


def read_known_prior_scores(path: Path) -> dict[str, float]:
    if not path.exists():
        raise SystemExit(f"Known-target prior CSV is required and must be non-empty: {path}")
    prior = pd.read_csv(path)
    if prior.empty:
        return {}
    missing = sorted(set(KNOWN_PRIOR_COLUMNS) - set(prior.columns))
    if missing:
        raise SystemExit(
            "Known-target prior CSV missing required column(s): "
            + ", ".join(missing)
        )
    prior["target_id"] = prior["target_id"].astype(str).str.strip()
    blank = [
        int(idx)
        for idx, value in prior["target_id"].items()
        if pd.isna(value) or not str(value).strip()
    ]
    if blank:
        shown = ", ".join(str(idx) for idx in blank[:10])
        suffix = "..." if len(blank) > 10 else ""
        raise SystemExit(
            "Known-target prior CSV contains blank target_id at row index "
            f"{shown}{suffix}"
        )
    duplicate_targets = prior["target_id"][prior["target_id"].duplicated()].tolist()
    if duplicate_targets:
        shown = ", ".join(sorted(set(duplicate_targets[:10])))
        suffix = "..." if len(duplicate_targets) > 10 else ""
        raise SystemExit(
            "Known-target prior CSV contains duplicate target_id values: "
            f"{shown}{suffix}"
        )
    scores = pd.to_numeric(prior["prior_score"], errors="coerce")
    invalid_scores = [
        int(idx)
        for idx, value in scores.items()
        if pd.isna(value) or not math.isfinite(float(value))
    ]
    if invalid_scores:
        shown = ", ".join(str(idx) for idx in invalid_scores[:10])
        suffix = "..." if len(invalid_scores) > 10 else ""
        raise SystemExit(
            "Known-target prior CSV contains invalid prior_score at row index "
            f"{shown}{suffix}"
        )
    out_of_range_scores = [
        int(idx)
        for idx, value in scores.items()
        if not 0.0 <= float(value) <= 1.0
    ]
    if out_of_range_scores:
        shown = ", ".join(str(idx) for idx in out_of_range_scores[:10])
        suffix = "..." if len(out_of_range_scores) > 10 else ""
        raise SystemExit(
            "Known-target prior CSV contains prior_score outside [0, 1] at row index "
            f"{shown}{suffix}"
        )
    if (prior["prior_score"].astype(float) != scores).any():
        raise SystemExit(
            "Known-target prior CSV contains non-finite prior_score values"
        )
    prior["prior_score"] = scores.astype(float)
    if (prior["prior_score"] < 0.0).any() or (prior["prior_score"] > 1.0).any():
        out_of_range_scores = [
            int(idx)
            for idx, value in prior["prior_score"].items()
            if not 0.0 <= float(value) <= 1.0
        ]
        shown = ", ".join(str(idx) for idx in out_of_range_scores[:10])
        suffix = "..." if len(out_of_range_scores) > 10 else ""
        raise SystemExit(
            "Known-target prior CSV contains prior_score outside [0, 1] at row index "
            f"{shown}{suffix}"
        )
    return dict(zip(prior["target_id"], prior["prior_score"], strict=True))


def read_top_targets(path: Path, min_source_count: int) -> tuple[pd.DataFrame, str]:
    if not _nonempty(path):
        raise SystemExit(f"Top-target CSV is required and must be non-empty: {path}")
    top = pd.read_csv(path)
    if "target_id" not in top.columns:
        raise SystemExit(f"Top-target CSV missing required column 'target_id': {path}")
    if top.empty:
        raise SystemExit(f"Top-target CSV contains no rows: {path}")
    blank_targets = [
        int(idx)
        for idx, value in top["target_id"].items()
        if pd.isna(value) or not str(value).strip()
    ]
    if blank_targets:
        raise SystemExit(
            _invalid_row_message(
                "Top-target CSV",
                "target_id",
                blank_targets,
                "blank values",
            )
        )
    docking_col = next(
        (col for col in DOCKING_SCORE_COLUMNS if col in top.columns),
        "",
    )
    if not docking_col:
        raise SystemExit(f"Top-target CSV missing docking score column: {path}")
    bool_scores = _bool_like_indexes(top[docking_col])
    if bool_scores:
        raise SystemExit(
            _invalid_row_message(
                "Top-target CSV",
                docking_col,
                bool_scores,
                "invalid values",
            )
        )
    scores = pd.to_numeric(top[docking_col], errors="coerce")
    invalid_scores = [
        int(idx)
        for idx, value in scores.items()
        if pd.isna(value) or not math.isfinite(float(value))
    ]
    if invalid_scores:
        raise SystemExit(
            _invalid_row_message(
                "Top-target CSV",
                docking_col,
                invalid_scores,
                "invalid values",
            )
        )
    top["target_id"] = top["target_id"].astype(str).str.strip()
    duplicate_targets = top["target_id"][top["target_id"].duplicated()].tolist()
    if duplicate_targets:
        raise SystemExit(
            _duplicate_value_message("Top-target CSV", "target_id", duplicate_targets)
        )
    top[docking_col] = scores.astype(float)
    if min_source_count > 1 and "source_count" not in top.columns:
        raise SystemExit(
            "Top-target CSV missing required column 'source_count' for "
            f"--min-source-count {min_source_count}: {path}"
        )
    if "source_count" in top.columns:
        bool_source_counts = _bool_like_indexes(top["source_count"])
        if bool_source_counts:
            raise SystemExit(
                _invalid_row_message(
                    "Top-target CSV",
                    "source_count",
                    bool_source_counts,
                    "invalid values",
                )
            )
        source_counts = pd.to_numeric(top["source_count"], errors="coerce")
        invalid_source_counts = [
            int(idx)
            for idx, value in source_counts.items()
            if pd.isna(value) or not math.isfinite(float(value)) or float(value) < 1
        ]
        if invalid_source_counts:
            raise SystemExit(
                _invalid_row_message(
                    "Top-target CSV",
                    "source_count",
                    invalid_source_counts,
                    "invalid values",
                )
            )
        noninteger_source_counts = [
            int(idx)
            for idx, value in source_counts.items()
            if not float(value).is_integer()
        ]
        if noninteger_source_counts:
            raise SystemExit(
                _invalid_row_message(
                    "Top-target CSV",
                    "source_count",
                    noninteger_source_counts,
                    "non-integer values",
                )
            )
        top["source_count"] = source_counts.astype(int)
        insufficient = [
            int(idx)
            for idx, value in top["source_count"].items()
            if int(value) < min_source_count
        ]
        if insufficient:
            raise SystemExit(
                _invalid_row_message(
                    "Top-target CSV",
                    "source_count",
                    insufficient,
                    f"values below --min-source-count {min_source_count}",
                )
            )
    if "sources" in top.columns:
        blank_sources = [
            int(idx)
            for idx, value in top["sources"].items()
            if pd.isna(value) or not str(value).strip()
        ]
        if blank_sources:
            raise SystemExit(
                _invalid_row_message(
                    "Top-target CSV",
                    "sources",
                    blank_sources,
                    "blank values",
                )
            )
        top["sources"] = top["sources"].astype(str).str.strip()
    _validate_source_rationale(top, "Top-target CSV")
    return top, docking_col


def warn_missing_skin_score_coverage(top: pd.DataFrame, skin: pd.DataFrame) -> None:
    target_ids = set(top["target_id"].astype(str))
    skin_ids = set(skin["uniprot"].astype(str))
    missing = sorted(target_ids - skin_ids)
    if missing:
        shown = ", ".join(missing[:10])
        suffix = "..." if len(missing) > 10 else ""
        LOG.warning(
            "Skin-expression score table is missing %d top-target id(s); "
            "using skin_score=0.0 and skin_tier=very_low for: %s%s",
            len(missing),
            shown,
            suffix,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-csv", required=True, type=Path)
    parser.add_argument("--skin-tsv", required=True, type=Path)
    parser.add_argument("--known-target-prior-csv", type=Path)
    parser.add_argument("--cosmetic-decision", type=Path, default=None)
    parser.add_argument("--drug-json", required=True, type=Path)
    parser.add_argument("--cosmetic-downweight-factor", type=float, default=0.50)
    parser.add_argument("--skin-weight", type=float, default=0.30)
    parser.add_argument("--known-target-prior-weight", type=float, default=0.0)
    parser.add_argument("--known-target-prior-min-score", type=float, default=0.0)
    parser.add_argument("--known-target-prior-power", type=float, default=1.0)
    parser.add_argument("--min-threshold", type=float, default=0.05)
    parser.add_argument("--min-source-count", type=int, default=1)
    parser.add_argument(
        "--preserve-primary-ranking",
        action="store_true",
        help=(
            "Annotation-only mode for fast runs. Skin and known-target evidence "
            "are retained without changing the upstream final_rank ordering."
        ),
    )
    parser.add_argument("--out-csv", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    _remove_outputs(args.out_csv)
    validate_parameters(
        args.cosmetic_downweight_factor,
        args.skin_weight,
        args.known_target_prior_weight,
        args.known_target_prior_min_score,
        args.known_target_prior_power,
        args.min_threshold,
        args.min_source_count,
    )
    top, docking_col = read_top_targets(args.top_csv, args.min_source_count)
    skin = read_skin_scores(args.skin_tsv)
    warn_missing_skin_score_coverage(top, skin)
    prior_scores = (
        read_known_prior_scores(args.known_target_prior_csv)
        if args.known_target_prior_csv is not None
        else {}
    )
    skin_map = dict(zip(skin["uniprot"].astype(str),
                        skin["skin_score"].astype(float), strict=True))
    tier_map = dict(zip(skin["uniprot"].astype(str),
                        skin["tier"].astype(str), strict=True)) if "tier" in skin.columns else {}
    cell_type_map = dict(zip(
        skin["uniprot"].astype(str),
        skin["cell_type_preferred"].astype(str),
        strict=True,
    ))

    docking = dict(zip(top["target_id"].astype(str),
                       top[docking_col].astype(float), strict=True))
    finals = apply_skin_weight(docking, skin_map,
                               skin_weight=args.skin_weight,
                               known_target_prior_scores=prior_scores,
                               known_target_prior_weight=args.known_target_prior_weight,
                               known_target_prior_min_score=args.known_target_prior_min_score,
                               known_target_prior_power=args.known_target_prior_power,
                               min_threshold=args.min_threshold)
    prepared_prior_scores = _prepare_prior_scores(
        prior_scores,
        min_score=args.known_target_prior_min_score,
        power=args.known_target_prior_power,
    )
    prior_norm = _normalize_non_constant_zero(
        [prepared_prior_scores.get(t, 0.0) for t in finals.keys()]
    )
    prior_norm_by_target = dict(zip(finals.keys(), prior_norm, strict=True))
    cosmetic_decision = read_decision(args.cosmetic_decision, args.drug_json)
    if cosmetic_decision == "HALT":
        raise SystemExit("cosmetic/drug decision is HALT")
    # A DOWNWEIGHT decision scales every row by the same factor, so it changes
    # the absolute final_score but never the within-run ordering. It is recorded
    # per row so downstream claim gates can see the applied factor.
    downweight = args.cosmetic_downweight_factor if cosmetic_decision == "DOWNWEIGHT" else 1.0
    if args.preserve_primary_ranking:
        preserved = _preserved_rank_order(top)
        preserved["docking_rrf"] = preserved[docking_col].astype(float)
        # The upstream band re-ranker defines the reader-facing order.  Its
        # scores are deliberately incommensurate outside the 11-50 band, so a
        # Daina score cannot stand in as the final score after rows have moved.
        # Publish an explicit ordinal score instead, while keeping rrf_score,
        # daina_rank and band_rrf_score untouched as provenance/evidence.
        row_count = len(preserved)
        preserved["final_score"] = [
            downweight * (row_count - rank + 1) / row_count
            for rank in preserved["final_rank"]
        ]
        preserved["final_score_semantics"] = "ordinal_rank"
        if "skin_score" not in preserved.columns:
            preserved["skin_score"] = [
                skin_map.get(target, 0.0) for target in preserved["target_id"]
            ]
        if "skin_tier" not in preserved.columns:
            preserved["skin_tier"] = [
                tier_map.get(target, "very_low") for target in preserved["target_id"]
            ]
        if "cell_type_preferred" not in preserved.columns:
            preserved["cell_type_preferred"] = [
                cell_type_map.get(target, "unknown")
                for target in preserved["target_id"]
            ]
        preserved["cosmetic_drug_decision"] = cosmetic_decision
        preserved["cosmetic_downweight_factor"] = downweight
        if "known_target_prior" not in preserved.columns:
            preserved["known_target_prior"] = [
                prepared_prior_scores.get(target, 0.0)
                for target in preserved["target_id"]
            ]
        if "known_target_prior_norm" not in preserved.columns:
            preserved["known_target_prior_norm"] = [
                prior_norm_by_target[target] for target in preserved["target_id"]
            ]
        _write_csv_atomic(preserved, args.out_csv)
        LOG.info(
            "Wrote %s (n=%d; upstream final_rank preserved)",
            args.out_csv,
            len(preserved),
        )
        return
    if downweight != 1.0:
        finals = {target: score * downweight for target, score in finals.items()}
    rows = {
        "target_id": list(finals.keys()),
        "docking_rrf": [docking[t] for t in finals.keys()],
        "skin_score":  [skin_map.get(t, 0.0) for t in finals.keys()],
        "skin_tier":   [tier_map.get(t, "very_low") for t in finals.keys()],
        "cell_type_preferred": [cell_type_map.get(t, "unknown") for t in finals.keys()],
        "cosmetic_drug_decision": [cosmetic_decision for _ in finals.keys()],
        "cosmetic_downweight_factor": [downweight for _ in finals.keys()],
        "known_target_prior": [prepared_prior_scores.get(t, 0.0) for t in finals.keys()],
        "known_target_prior_norm": prior_norm,
        "final_score": list(finals.values()),
    }
    for col in ("source_count", "sources"):
        if col in top.columns:
            values = dict(zip(top["target_id"], top[col], strict=True))
            rows[col] = [values[t] for t in finals.keys()]
    df = pd.DataFrame(rows).sort_values("final_score", ascending=False)

    _write_csv_atomic(df, args.out_csv)
    LOG.info("Wrote %s (n=%d)", args.out_csv, len(df))


if __name__ == "__main__":
    main()
