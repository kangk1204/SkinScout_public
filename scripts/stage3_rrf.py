#!/usr/bin/env python3
"""stage3_rrf.py — Reciprocal Rank Fusion across N score lists.

RRF(t) = Σ_score 1 / (k + rank_score(t))

Inputs: a comma-separated list of `path=label` pairs, each pointing at a TSV
or CSV that contains at minimum {target_id, score}. Higher score → better rank.
Targets missing from a given score list contribute 0 to their RRF.
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import pandas as pd

LOG = logging.getLogger("stage3.rrf")


def _remove_outputs(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def _write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[str]],
    k: int = 60,
    source_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Pure function (unit-tested).

    Args:
        ranked_lists: { source_label: [target_id, ...] } sorted best-first.
        k: RRF constant (Cormack et al. 2009).
        source_weights: Optional { source_label: positive finite weight }.
            Omitted uses the exact equal-RRF behavior.

    Returns:
        { target_id: rrf_score }, larger = better consensus.
    """
    weights = _source_weights_for_ranked_lists(ranked_lists, source_weights)
    rrf: dict[str, float] = {}
    for label, ordered in ranked_lists.items():
        weight = weights[label]
        for rank, target in enumerate(_unique_targets(ordered), start=1):
            rrf[target] = rrf.get(target, 0.0) + weight / (k + rank)
    return rrf


def _source_weights_for_ranked_lists(
    ranked_lists: dict[str, list[str]],
    source_weights: dict[str, float] | None,
) -> dict[str, float]:
    if source_weights is None:
        return {label: 1.0 for label in ranked_lists}
    missing = set(ranked_lists) - set(source_weights)
    extra = set(source_weights) - set(ranked_lists)
    if missing or extra:
        detail = []
        if missing:
            detail.append(f"missing labels: {sorted(missing)}")
        if extra:
            detail.append(f"extra labels: {sorted(extra)}")
        raise ValueError(
            "source_weights labels must match ranked_lists exactly; "
            + "; ".join(detail)
        )
    weights: dict[str, float] = {}
    for label, weight in source_weights.items():
        if isinstance(weight, bool):
            raise ValueError(f"source weight for {label!r} must not be boolean")
        try:
            value = float(weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"source weight for {label!r} must be numeric") from exc
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"source weight for {label!r} must be a positive finite value"
            )
        weights[label] = value
    return weights


def _unique_targets(ordered: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for target in ordered:
        if target in seen:
            continue
        seen.add(target)
        unique.append(target)
    return unique


def source_support(ranked_lists: dict[str, list[str]]) -> dict[str, set[str]]:
    """Return target_id -> source labels that supplied a rank."""
    support: dict[str, set[str]] = {}
    for label, ordered in ranked_lists.items():
        for target in _unique_targets(ordered):
            support.setdefault(target, set()).add(label)
    return support


def source_ranks(ranked_lists: dict[str, list[str]]) -> dict[str, dict[str, int]]:
    """Return target_id -> source label -> one-based rank."""
    ranks: dict[str, dict[str, int]] = {}
    for label, ordered in ranked_lists.items():
        for rank, target in enumerate(_unique_targets(ordered), start=1):
            ranks.setdefault(target, {})[label] = rank
    return ranks


def source_contributions(
    ranked_lists: dict[str, list[str]],
    k: int = 60,
    source_weights: dict[str, float] | None = None,
) -> dict[str, dict[str, float]]:
    """Return target_id -> source label -> weighted RRF contribution."""
    weights = _source_weights_for_ranked_lists(ranked_lists, source_weights)
    contributions: dict[str, dict[str, float]] = {}
    for label, ordered in ranked_lists.items():
        weight = weights[label]
        for rank, target in enumerate(_unique_targets(ordered), start=1):
            contributions.setdefault(target, {})[label] = weight / (k + rank)
    return contributions


def _invalid_row_message(path: Path, column: str, rows: list[int], detail: str) -> str:
    shown = ", ".join(str(idx) for idx in rows[:10])
    suffix = "..." if len(rows) > 10 else ""
    return (
        f"{path} RRF column '{column}' contains {detail} at row "
        f"index(es) {shown}{suffix}"
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


def _validate_cli_args(
    *,
    rrf_k: int,
    top_n: int | None,
    top_pct: float | None,
    min_sources_per_target: int,
) -> None:
    if rrf_k < 1:
        raise SystemExit(f"--rrf-k must be >= 1: {rrf_k}")
    if min_sources_per_target < 1:
        raise SystemExit(
            f"--min-sources-per-target must be >= 1: {min_sources_per_target}"
        )
    if top_n is not None and top_n < 1:
        raise SystemExit(f"--top-n must be >= 1: {top_n}")
    if top_pct is not None and (
        not math.isfinite(top_pct) or not 0.0 < top_pct <= 1.0
    ):
        raise SystemExit(f"--top-pct must be a finite value in (0, 1]: {top_pct:g}")


def _parse_source_weights(
    source_weights: str | None,
    labels: list[str],
) -> dict[str, float] | None:
    if source_weights is None:
        return None
    weights: dict[str, float] = {}
    for entry in source_weights.split(","):
        if "=" not in entry:
            raise SystemExit(
                "RRF --source-weights entries must use label=weight format: "
                f"{entry!r}"
            )
        label, weight_s = entry.split("=", 1)
        label = label.strip()
        weight_s = weight_s.strip()
        if not label:
            raise SystemExit(
                "RRF --source-weights entries must provide a non-empty label: "
                f"{entry!r}"
            )
        if label in weights:
            raise SystemExit(f"RRF --source-weights contains duplicate label: {label}")
        if weight_s.lower() in {"true", "false"}:
            raise SystemExit(
                f"RRF --source-weights label {label!r} must be a positive finite weight"
            )
        try:
            weight = float(weight_s)
        except ValueError as exc:
            raise SystemExit(
                f"RRF --source-weights label {label!r} must be a positive finite weight"
            ) from exc
        if not math.isfinite(weight) or weight <= 0.0:
            raise SystemExit(
                f"RRF --source-weights label {label!r} must be a positive finite weight"
            )
        weights[label] = weight
    missing = set(labels) - set(weights)
    extra = set(weights) - set(labels)
    if missing or extra:
        detail = []
        if missing:
            detail.append(f"missing labels: {sorted(missing)}")
        if extra:
            detail.append(f"extra labels: {sorted(extra)}")
        raise SystemExit(
            "RRF --source-weights labels must exactly match --inputs labels; "
            + "; ".join(detail)
        )
    return weights


def _rank_from_table(path: Path) -> list[str]:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, sep="\t")
    if "target_id" not in df.columns:
        for cand in ("uniprot", "uniprot_id", "target", "receptor"):
            if cand in df.columns:
                df = df.rename(columns={cand: "target_id"})
                break
    if "score" not in df.columns:
        for cand in ("vina_score", "vinardo", "affinity", "pred",
                     "predicted_affinity", "rtm_score", "cnn_score"):
            if cand in df.columns:
                df = df.rename(columns={cand: "score"})
                break
    missing = {"target_id", "score"} - set(df.columns)
    if missing:
        raise SystemExit(
            f"{path} is missing required RRF column(s): {sorted(missing)}"
        )
    if df.empty:
        raise SystemExit(f"{path} RRF input contains no rows")
    blank_targets = [
        int(idx)
        for idx, value in df["target_id"].items()
        if pd.isna(value) or not str(value).strip()
    ]
    if blank_targets:
        raise SystemExit(
            _invalid_row_message(path, "target_id", blank_targets, "blank values")
        )
    bool_scores = _bool_like_indexes(df["score"])
    if bool_scores:
        raise SystemExit(
            _invalid_row_message(path, "score", bool_scores, "invalid values")
        )
    scores = pd.to_numeric(df["score"], errors="coerce")
    invalid_scores = [
        int(idx)
        for idx, value in scores.items()
        if pd.isna(value) or not math.isfinite(float(value))
    ]
    if invalid_scores:
        raise SystemExit(
            _invalid_row_message(path, "score", invalid_scores, "invalid values")
        )
    df["target_id"] = df["target_id"].astype(str).str.strip()
    duplicate_targets = df["target_id"][df["target_id"].duplicated()].tolist()
    if duplicate_targets:
        shown = ", ".join(str(target) for target in duplicate_targets[:10])
        suffix = "..." if len(duplicate_targets) > 10 else ""
        raise SystemExit(
            f"{path} RRF input contains duplicate target_id values: "
            f"{shown}{suffix}"
        )
    df["score"] = scores.astype(float)
    # Higher score = better. For energies/IC50 (lower better), input file must
    # already invert sign upstream (see stage3 callers).
    #
    # Ties break on target_id with a stable sort, matching stage3_pick_top and
    # stage3_select_daina. Ties are the normal case here rather than an edge
    # one: stage3_pick_top normalises equal raw scores to equal ranks on
    # purpose, and an unstable sort turned that into a rank spread of tens of
    # positions - and a different one on a rerun of the same inputs.
    df = df.sort_values(["score", "target_id"], ascending=[False, True], kind="mergesort")
    return df["target_id"].astype(str).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True,
                        help="Comma-separated path=label pairs.")
    parser.add_argument("--rrf-k", type=int, default=60)
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--top-n", type=int, default=None)
    g.add_argument("--top-pct", type=float, default=None)
    parser.add_argument("--min-sources-per-target", type=int, default=1,
                        help="Discard targets supported by fewer scorer lists.")
    parser.add_argument(
        "--source-weights",
        default=None,
        help="Comma-separated label=positive_finite_weight entries.",
    )
    parser.add_argument("--recipe-id", default=None)
    parser.add_argument("--out-csv", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    _remove_outputs(args.out_csv)
    _validate_cli_args(
        rrf_k=args.rrf_k,
        top_n=args.top_n,
        top_pct=args.top_pct,
        min_sources_per_target=args.min_sources_per_target,
    )
    ranked: dict[str, list[str]] = {}
    for pair in args.inputs.split(","):
        if "=" not in pair:
            raise SystemExit(
                "RRF --inputs entries must use path=label format: "
                f"{pair!r}"
            )
        path_s, label = pair.split("=", 1)
        label = label.strip()
        if not label:
            raise SystemExit(
                "RRF --inputs entries must provide a non-empty label: "
                f"{pair!r}"
            )
        if label in ranked:
            raise SystemExit(f"RRF --inputs contains duplicate RRF input label: {label}")
        ranked[label] = _rank_from_table(Path(path_s))
        LOG.info("Loaded %d targets from %s (%s)", len(ranked[label]), path_s, label)

    weights = _parse_source_weights(args.source_weights, list(ranked))
    rrf = reciprocal_rank_fusion(ranked, k=args.rrf_k, source_weights=weights)
    support = source_support(ranked)
    ranks = source_ranks(ranked)
    contributions = source_contributions(
        ranked,
        k=args.rrf_k,
        source_weights=weights,
    )
    df = pd.DataFrame(
        [
            {
                "target_id": target,
                "rrf_score": score,
                "source_count": len(support.get(target, set())),
                "sources": ";".join(sorted(support.get(target, set()))),
                **{
                    f"{label}_rank": ranks.get(target, {}).get(label)
                    for label in ranked
                },
                **{
                    f"{label}_rrf_contribution": contributions.get(target, {}).get(
                        label, 0.0
                    )
                    for label in ranked
                },
            }
            for target, score in sorted(rrf.items(), key=lambda kv: -kv[1])
        ],
    )
    if df.empty:
        raise SystemExit("RRF produced no ranked targets")
    df = df[df["source_count"] >= args.min_sources_per_target].copy()
    if df.empty:
        raise SystemExit(
            "RRF produced no targets meeting "
            f"--min-sources-per-target={args.min_sources_per_target}"
        )
    # 동점은 흔하다. target_id를 2차 키로 두고 안정 정렬해 실행 간 순서를 고정한다.
    df = df.sort_values(
        ["rrf_score", "target_id"], ascending=[False, True], kind="mergesort"
    )
    if args.top_pct is not None:
        n_keep = max(1, int(round(args.top_pct * len(df))))
        df = df.head(n_keep)
    elif args.top_n is not None:
        df = df.head(args.top_n)
    if args.recipe_id is not None:
        df.insert(0, "recipe_id", args.recipe_id)

    _write_csv_atomic(df, args.out_csv)
    LOG.info("RRF top → %s (%d rows)", args.out_csv, len(df))


if __name__ == "__main__":
    main()
