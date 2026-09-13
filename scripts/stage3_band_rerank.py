#!/usr/bin/env python3
"""stage3_band_rerank.py — reorder only the band where docking actually helps.

Measured on the 15-compound panel, leave-query-out, 28 known pairs
(docs/RERANK_EXPERIMENT_20260828.md):

    ordering                          mean rank   top-30
    Daina similarity alone                 43.2    14/28
    RRF over the whole list                51.2    17/28
    Daina head kept + ranks 11-50 by RRF   38.5    20/28   (Wilcoxon p = 0.0096)

These are the shipped band-only docking results.  The separate full-256 docking
arm reached mean rank 37.5 (p = 0.0068) but is not the production fast path.

Re-ranking everything is worse than not re-ranking at all. Where similarity
already puts a target in the top 10 the docking scores only add noise - that
went the wrong way in 8 of 8 pairs, with Capsaicin-TRPV1 falling from 1st to
41st. In the 11-50 band, where similarity is genuinely uncertain, it helped in
11 of 13.

So the head is left exactly as it was and only the band is re-ordered. Every
row keeps `daina_rank`, so what moved is always visible.

    python scripts/stage3_band_rerank.py \
        --in-csv daina_structural_targets.csv \
        --out-csv daina_band_reranked_targets.csv \
        --out-top50 top50.csv
"""

from __future__ import annotations

import argparse
import os
import uuid
from pathlib import Path

import pandas as pd

# The head a reader treats as "the shortlist"; below it similarity is uncertain
# enough for structure to be worth consulting.
DEFAULT_KEEP = 10
DEFAULT_BAND = 50
# Reciprocal Rank Fusion constant, unchanged from the comprehensive path.
RRF_K = 60
BAND_RANKING_BASIS = "daina_head_then_band_rrf"
UNCHANGED_RANKING_BASIS = "daina_max_tanimoto"

REQUIRED_COLUMNS = ("target_id", "daina_rank", "ranking_basis")
# Columns the band ordering may consult. Each is optional: a run that did not
# produce one simply contributes no opinion, rather than failing.
SCORE_COLUMNS = (
    ("autodock_energy_kcal_mol", True),   # lower is better
    ("gnina_cnn_affinity", False),        # higher is better
)


def band_rerank(
    frame: pd.DataFrame,
    *,
    keep: int = DEFAULT_KEEP,
    band: int = DEFAULT_BAND,
) -> pd.DataFrame:
    """Return `frame` reordered, with the head untouched.

    `keep >= band` disables the re-ordering entirely and returns the input
    order, so a caller can turn this off without a separate code path.
    """
    for column in REQUIRED_COLUMNS:
        if column not in frame.columns:
            raise SystemExit(f"입력에 필수 열이 없습니다: {column}")
    if keep < 0 or band < 0:
        raise SystemExit("keep과 band는 0 이상이어야 합니다")

    result = frame.copy()
    result["daina_rank"] = pd.to_numeric(result["daina_rank"], errors="coerce")
    if result["daina_rank"].isna().any():
        raise SystemExit("daina_rank에 숫자가 아닌 값이 있습니다")

    in_band = (result["daina_rank"] > keep) & (result["daina_rank"] <= band)
    result["rerank_band"] = in_band
    if keep >= band or not in_band.any():
        # Nothing to reorder; say so rather than claiming a basis we did not use.
        # Keep whatever basis the overlay recorded: under recipe scoring it is
        # not the similarity, and overwriting it here would reintroduce the
        # claim the overlay just stopped making.
        if "ranking_basis" not in result.columns:
            result["ranking_basis"] = UNCHANGED_RANKING_BASIS
        ordered = result.sort_values("daina_rank").reset_index(drop=True)
        ordered["final_rank"] = range(1, len(ordered) + 1)
        return ordered

    # Rank each available scorer within the band only. Docking the rest would
    # change nothing: the head is fixed and the tail keeps its similarity order.
    contributions = [1.0 / (RRF_K + result.loc[in_band, "daina_rank"])]
    used = ["daina"]
    band_size = int(in_band.sum())
    for column, lower_is_better in SCORE_COLUMNS:
        if column not in result.columns:
            continue
        values = pd.to_numeric(result.loc[in_band, column], errors="coerce")
        if values.notna().sum() == 0:
            continue
        ranks = values.rank(method="min", ascending=lower_is_better)
        # A target the scorer could not reach sits behind every one it could.
        contributions.append(1.0 / (RRF_K + ranks.fillna(band_size + 1)))
        used.append(column)

    fused = pd.Series(0.0, index=result.index[in_band])
    for contribution in contributions:
        fused = fused.add(contribution, fill_value=0.0)
    result.loc[in_band, "band_rrf_score"] = fused
    result["band_rrf_sources"] = ""
    result.loc[in_band, "band_rrf_sources"] = ";".join(used)

    head = result[result["daina_rank"] <= keep].sort_values("daina_rank")
    middle = result[in_band].sort_values(
        ["band_rrf_score", "daina_rank"], ascending=[False, True]
    )
    tail = result[result["daina_rank"] > band].sort_values("daina_rank")
    ordered = pd.concat([head, middle, tail]).reset_index(drop=True)
    ordered["final_rank"] = range(1, len(ordered) + 1)
    # Only the band's order came from the fusion, and the basis says exactly that.
    ordered["ranking_basis"] = BAND_RANKING_BASIS
    return ordered


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    try:
        frame.to_csv(temp, index=False)
        os.replace(temp, path)
    except OSError:
        temp.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-csv", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--out-top50", type=Path)
    parser.add_argument("--keep", type=int, default=DEFAULT_KEEP,
                        help="이 순위까지는 유사도 순서를 그대로 둡니다.")
    parser.add_argument("--band", type=int, default=DEFAULT_BAND,
                        help="keep 다음부터 이 순위까지만 재정렬합니다. keep 이하면 비활성.")
    args = parser.parse_args()

    frame = pd.read_csv(args.in_csv)
    ordered = band_rerank(frame, keep=args.keep, band=args.band)
    _write_csv_atomic(ordered, args.out_csv)
    if args.out_top50 is not None:
        _write_csv_atomic(ordered.head(50), args.out_top50)

    moved = int((ordered["final_rank"] != ordered["daina_rank"]).sum())
    print(
        f"밴드 재정렬 완료: {len(ordered)}개 중 {int(ordered['rerank_band'].sum())}개가 "
        f"재정렬 대상({args.keep + 1}-{args.band}위), 순위가 바뀐 표적 {moved}개 → {args.out_csv}"
    )


if __name__ == "__main__":
    main()
