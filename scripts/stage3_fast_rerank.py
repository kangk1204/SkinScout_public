#!/usr/bin/env python3
"""stage3_fast_rerank.py — top-1 % AutoDock candidates → GNINA + RTM + Boltz-2
rerank, RRF-fused to a final top-50 list for MODE-FAST.
"""

from __future__ import annotations

import argparse
import logging
import math
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from stage3_boltz2_affinity import call_boltz
from stage3_gnina_rescore import gnina_score
from stage3_rrf import reciprocal_rank_fusion, source_support
from stage3_rtmscore import rtmscore

LOG = logging.getLogger("stage3.fast_rerank")


def _remove_outputs(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def _nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _read_efficacy_kg_targets(path: Path) -> set[str]:
    if not _nonempty(path):
        raise SystemExit(f"Skin-efficacy KG GraphML is required and must be non-empty: {path}")
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise SystemExit(f"Skin-efficacy KG GraphML failed to parse: {path}: {exc}") from exc
    targets: set[str] = set()
    for node in root.iter():
        if not node.tag.endswith("node"):
            continue
        node_id = str(node.attrib.get("id", "")).strip()
        if node_id.startswith("gene:") and len(node_id) > len("gene:"):
            targets.add(node_id[len("gene:"):])
    if not targets:
        raise SystemExit(f"Skin-efficacy KG GraphML contains no gene:* nodes: {path}")
    return targets


def _invalid_row_message(label: str, column: str, rows: list[int], detail: str) -> str:
    shown = ", ".join(str(idx) for idx in rows[:10])
    suffix = "..." if len(rows) > 10 else ""
    return (
        f"{label} column '{column}' contains {detail} at row "
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
    min_sources_per_target: int,
    top_n: int,
    rerank_candidates: int,
    rtm_top_n: int,
    boltz_top_n: int,
    max_residues: int,
    crop_radius: float,
) -> None:
    if rrf_k < 1:
        raise SystemExit(f"--rrf-k must be >= 1: {rrf_k}")
    if min_sources_per_target < 1:
        raise SystemExit(
            f"--min-sources-per-target must be >= 1: {min_sources_per_target}"
        )
    if top_n < 1:
        raise SystemExit(f"--top-n must be >= 1: {top_n}")
    if rerank_candidates < 0:
        raise SystemExit(f"--rerank-candidates must be >= 0: {rerank_candidates}")
    if rtm_top_n < 0:
        raise SystemExit(f"--rtm-top-n must be >= 0: {rtm_top_n}")
    if boltz_top_n < 0:
        raise SystemExit(f"--boltz-top-n must be >= 0: {boltz_top_n}")
    if max_residues < 1:
        raise SystemExit(f"--max-residues must be >= 1: {max_residues}")
    if not math.isfinite(crop_radius) or crop_radius <= 0:
        raise SystemExit(f"--crop-radius must be a finite value > 0: {crop_radius:g}")


def _read_autodock_scores(path: Path) -> pd.DataFrame:
    if not _nonempty(path):
        raise SystemExit(
            f"AutoDock score file is missing or empty for fast rerank: {path}"
        )
    autodock = pd.read_csv(path, sep="\t")
    missing = {"target_id", "neg_vina_score"} - set(autodock.columns)
    if missing:
        raise SystemExit(
            "AutoDock score file missing required columns for fast rerank: "
            + ", ".join(sorted(missing))
        )
    if autodock.empty:
        raise SystemExit("AutoDock score file contains no usable fast rerank rows")
    blank_targets = [
        int(idx)
        for idx, value in autodock["target_id"].items()
        if pd.isna(value) or not str(value).strip()
    ]
    if blank_targets:
        raise SystemExit(
            _invalid_row_message(
                "AutoDock score file",
                "target_id",
                blank_targets,
                "blank values",
            )
        )
    duplicate_targets = autodock["target_id"].astype(str).str.strip()
    duplicate_ids = duplicate_targets[duplicate_targets.duplicated()].tolist()
    if duplicate_ids:
        shown = ", ".join(duplicate_ids[:10])
        suffix = "..." if len(duplicate_ids) > 10 else ""
        raise SystemExit(
            "AutoDock score file contains duplicate target_id values for fast "
            f"rerank: {shown}{suffix}"
        )
    bool_scores = _bool_like_indexes(autodock["neg_vina_score"])
    if bool_scores:
        raise SystemExit(
            _invalid_row_message(
                "AutoDock score file",
                "neg_vina_score",
                bool_scores,
                "invalid values",
            )
        )
    scores = pd.to_numeric(autodock["neg_vina_score"], errors="coerce")
    invalid_scores = [
        int(idx)
        for idx, value in scores.items()
        if pd.isna(value) or not math.isfinite(float(value))
    ]
    if invalid_scores:
        raise SystemExit(
            _invalid_row_message(
                "AutoDock score file",
                "neg_vina_score",
                invalid_scores,
                "invalid values",
            )
        )
    autodock["target_id"] = autodock["target_id"].astype(str).str.strip()
    autodock["neg_vina_score"] = scores.astype(float)
    return autodock


def _checked_scorer_score(
    value: object,
    *,
    label: str,
    target_id: str,
) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or type(value).__name__ == "bool_"
        or (
            isinstance(value, str)
            and value.strip().lower() in {"true", "false"}
        )
    ):
        raise SystemExit(
            f"{label} returned non-numeric score for {target_id}: {value!r}"
        )
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            f"{label} returned non-numeric score for {target_id}: {value!r}"
        ) from exc
    if not math.isfinite(score):
        raise SystemExit(
            f"{label} returned non-finite score for {target_id}: {value!r}"
        )
    return score


def _effective_candidate_count(
    *,
    autodock_count: int,
    rerank_candidates: int,
    top_n: int,
) -> tuple[int, int]:
    """Return the requested and effective fast-rerank candidate budgets."""
    default_candidates = max(50, math.ceil(0.01 * autodock_count))
    requested_candidates = rerank_candidates or default_candidates
    return requested_candidates, max(top_n, requested_candidates)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--autodock-scores", required=True, type=Path)
    parser.add_argument("--ligand-sdf", required=True, type=Path)
    parser.add_argument("--clean-dir", required=True, type=Path)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--min-sources-per-target", type=int, default=2)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument(
        "--rerank-candidates",
        type=int,
        default=0,
        help="Number of AutoDock-ranked candidates to rerank; 0 keeps the default top 1%/min 50.",
    )
    parser.add_argument(
        "--rtm-top-n",
        type=int,
        default=50,
        help="Run RTMScore only for the first N fast rerank candidates; 0 skips RTMScore.",
    )
    parser.add_argument(
        "--boltz-top-n",
        type=int,
        default=50,
        help="Run Boltz only for the first N fast rerank candidates; 0 skips Boltz.",
    )
    parser.add_argument(
        "--require-efficacy-kg",
        type=Path,
        default=None,
        help="Restrict fast rerank candidates to targets present as gene:* nodes in this GraphML.",
    )
    parser.add_argument("--max-residues", type=int, default=700)
    parser.add_argument("--crop-radius", type=float, default=20.0)
    parser.add_argument("--out-csv", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    _remove_outputs(args.out_csv)
    _validate_cli_args(
        rrf_k=args.rrf_k,
        min_sources_per_target=args.min_sources_per_target,
        top_n=args.top_n,
        rerank_candidates=args.rerank_candidates,
        rtm_top_n=args.rtm_top_n,
        boltz_top_n=args.boltz_top_n,
        max_residues=args.max_residues,
        crop_radius=args.crop_radius,
    )
    autodock = _read_autodock_scores(args.autodock_scores)
    if not _nonempty(args.ligand_sdf):
        raise SystemExit(
            f"Ligand SDF is missing or empty for fast rerank: {args.ligand_sdf}"
        )
    autodock = autodock.sort_values("neg_vina_score", ascending=False)
    candidate_pool = autodock
    if args.require_efficacy_kg is not None:
        kg_targets = _read_efficacy_kg_targets(args.require_efficacy_kg)
        candidate_pool = autodock[autodock["target_id"].isin(kg_targets)].copy()
        LOG.info(
            "Fast-mode efficacy KG filter: %d/%d AutoDock candidates",
            len(candidate_pool),
            len(autodock),
        )
        if candidate_pool.empty:
            raise SystemExit(
                "No AutoDock candidates overlap skin-efficacy KG gene:* targets"
            )
    requested_candidates, candidate_count = _effective_candidate_count(
        autodock_count=len(autodock),
        rerank_candidates=args.rerank_candidates,
        top_n=args.top_n,
    )
    top1pct = candidate_pool.head(candidate_count)
    LOG.info(
        "Fast-mode rerank candidates: %d (requested=%d, effective=%d)",
        len(top1pct),
        requested_candidates,
        candidate_count,
    )
    boltz_limit = min(args.boltz_top_n, len(top1pct))
    rtm_limit = min(args.rtm_top_n, len(top1pct))
    LOG.info("Fast-mode RTMScore candidates: %d/%d", rtm_limit, len(top1pct))
    LOG.info("Fast-mode Boltz candidates: %d/%d", boltz_limit, len(top1pct))

    gnina_rows: list[tuple[str, float]] = []
    rtm_rows: list[tuple[str, float]] = []
    boltz_rows: list[tuple[str, float]] = []
    with tempfile.TemporaryDirectory() as tmp:
        for idx, uid in enumerate(top1pct["target_id"].astype(str), start=1):
            receptor = args.clean_dir / f"{uid}_clean.pdb"
            if not receptor.exists():
                if idx % 5 == 0 or idx == len(top1pct):
                    LOG.info(
                        "Fast rerank progress %d/%d targets "
                        "(gnina=%d, rtm=%d, boltz=%d)",
                        idx,
                        len(top1pct),
                        len(gnina_rows),
                        len(rtm_rows),
                        len(boltz_rows),
                    )
                continue
            g = _checked_scorer_score(
                gnina_score(receptor, args.ligand_sdf),
                label="GNINA",
                target_id=uid,
            )
            if g is not None:
                gnina_rows.append((uid, g))
            if idx <= rtm_limit:
                r = _checked_scorer_score(
                    rtmscore(receptor, args.ligand_sdf),
                    label="RTMScore",
                    target_id=uid,
                )
                if r is not None:
                    rtm_rows.append((uid, r))
            if idx <= boltz_limit:
                b = _checked_scorer_score(
                    call_boltz(
                        receptor,
                        args.ligand_sdf,
                        Path(tmp) / uid,
                        args.max_residues,
                        args.crop_radius,
                    ),
                    label="Boltz-2",
                    target_id=uid,
                )
                if b is not None:
                    boltz_rows.append((uid, b))
            if idx % 5 == 0 or idx == len(top1pct):
                LOG.info(
                    "Fast rerank progress %d/%d targets "
                    "(gnina=%d, rtm=%d, boltz=%d)",
                    idx,
                    len(top1pct),
                    len(gnina_rows),
                    len(rtm_rows),
                    len(boltz_rows),
                )

    def to_ranked(rows: list[tuple[str, float]]) -> list[str]:
        return [u for u, _ in sorted(rows, key=lambda x: -x[1])]

    ranked = {
        "autodock": top1pct["target_id"].astype(str).tolist(),
        "gnina":    to_ranked(gnina_rows),
        "rtm":      to_ranked(rtm_rows),
        "boltz":    to_ranked(boltz_rows),
    }
    rrf = reciprocal_rank_fusion(ranked, k=args.rrf_k)
    support = source_support(ranked)
    df = pd.DataFrame(
        [
            {
                "target_id": target,
                "rrf_score": score,
                "source_count": len(support.get(target, set())),
                "sources": ";".join(sorted(support.get(target, set()))),
            }
            for target, score in sorted(rrf.items(), key=lambda kv: -kv[1])
        ]
    )
    df = df[df["source_count"] >= args.min_sources_per_target]
    if df.empty:
        raise SystemExit(
            "Fast rerank produced no targets meeting "
            f"--min-sources-per-target={args.min_sources_per_target}"
        )
    df = df.sort_values("rrf_score", ascending=False).head(args.top_n)

    _write_csv_atomic(df, args.out_csv)
    LOG.info("Fast-mode top50 → %s", args.out_csv)


if __name__ == "__main__":
    main()
