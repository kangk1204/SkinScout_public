#!/usr/bin/env python3
"""Optionally reorder a bounded band using archived or run-local docking scores.

The workflow disables this experimental policy by default. The historical
nearest-Tanimoto experiment does not validate the current recipe; the current
diagnostic retains all 32 truth pairs and reports incomplete docking coverage
(docs/CURRENT_RERANK_POLICY_REVIEW_20260914.md). Every row records the policy,
its bounds, its ranking basis, and the original `daina_rank`.

The CLI below explicitly runs the experimental 11-50 policy. Use --band 10
with the default --keep 10 to preserve the complete original recipe order.

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

# Historical exploratory bounds; they do not establish a validated optimum.
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
    # standalone 호출도 중복/결번/공백 입력을 거부한다(상류 검증에 의존하지 않는다).
    # 정규화(trim)를 검사보다 먼저 적용해야 " T1"/"T1" 같은 표기 차이가
    # 유일성 검사를 빠져나가지 못한다.
    target_ids = frame["target_id"].astype(str).str.strip()
    if target_ids.eq("").any():
        raise SystemExit("target_id가 비어 있습니다")
    if target_ids.duplicated().any():
        raise SystemExit("target_id가 중복되었습니다(trim 후 기준)")

    result = frame.copy()
    result["target_id"] = target_ids
    result["rerank_policy_enabled"] = keep < band
    result["rerank_policy_status"] = "experimental" if keep < band else "disabled"
    result["rerank_keep"] = keep
    result["rerank_band_end"] = band
    # int cast가 1.1을 1로 받아들이던 경로를 막는다: 유한한 정수만 허용한다.
    ranks = pd.to_numeric(result["daina_rank"], errors="coerce")
    if ranks.isna().any() or ranks.abs().eq(float("inf")).any():
        raise SystemExit("daina_rank에 유한한 숫자가 아닌 값이 있습니다")
    if (ranks % 1 != 0).any():
        raise SystemExit("daina_rank는 정수여야 합니다")
    result["daina_rank"] = ranks
    if ranks.duplicated().any():
        raise SystemExit("daina_rank가 중복되었습니다")
    if sorted(ranks.astype(int).tolist()) != list(range(1, len(result) + 1)):
        raise SystemExit("daina_rank가 1..n 연속이 아닙니다")

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


def rerank_policy_error(delivered: pd.DataFrame, canonical: pd.DataFrame) -> str | None:
    """Validate the declared ordering policy against its canonical projection."""
    fields = {"rerank_policy_enabled", "rerank_policy_status", "rerank_keep", "rerank_band_end",
              "target_id", "daina_rank", "final_rank", "ranking_basis", "rerank_band"}
    missing = fields - set(delivered.columns)
    if missing:
        return "missing rerank policy provenance: " + ", ".join(sorted(missing))
    if delivered.empty or canonical.empty:
        return "rerank policy requires nonempty delivered and canonical targets"
    if set(REQUIRED_COLUMNS) - set(canonical.columns):
        return "canonical targets lack the ordering inputs"
    delivered = delivered.copy()
    for field in ("daina_rank", "final_rank"):
        values = pd.to_numeric(delivered[field], errors="coerce")
        if values.isna().any() or (values < 1).any() or (values % 1 != 0).any():
            return f"{field} must contain finite positive integers"
        delivered[field] = values
    enabled_values = delivered.rerank_policy_enabled.astype(str).str.lower()
    if enabled_values.nunique() != 1 or enabled_values.iloc[0] not in {"true", "false"}:
        return "rerank_policy_enabled must be one explicit boolean across all rows"
    enabled = enabled_values.iloc[0] == "true"
    status = "experimental" if enabled else "disabled"
    if not delivered.rerank_policy_status.eq(status).all():
        return "rerank policy status does not match enabled flag"
    bounds = {}
    for field in ("rerank_keep", "rerank_band_end"):
        values = pd.to_numeric(delivered[field], errors="coerce")
        if values.isna().any() or values.nunique() != 1 or (values < 0).any() or (values % 1 != 0).any():
            return f"{field} must be one finite nonnegative integer"
        bounds[field] = int(values.iloc[0])
    if enabled != (bounds["rerank_keep"] < bounds["rerank_band_end"]):
        return "rerank policy bounds do not match enabled flag"
    flags = delivered.rerank_band.astype(str).str.lower()
    if not flags.isin(["true", "false"]).all():
        return "rerank_band must contain explicit booleans"
    if not enabled:
        if delivered.target_id.astype(str).tolist() != canonical.target_id.astype(str).tolist():
            return "disabled reranking must preserve the complete canonical target order"
        if flags.ne("false").any():
            return "disabled reranking must have no rerank_band rows"
        if not pd.to_numeric(delivered.final_rank, errors="coerce").eq(pd.to_numeric(delivered.daina_rank, errors="coerce")).all():
            return "disabled reranking must preserve final_rank == daina_rank"
        if delivered.ranking_basis.astype(str).tolist() != canonical.ranking_basis.astype(str).tolist():
            return "disabled reranking must preserve canonical ranking_basis"
    else:
        expected = delivered.daina_rank.gt(bounds["rerank_keep"]) & delivered.daina_rank.le(bounds["rerank_band_end"])
        if not flags.eq("true").eq(expected).all():
            return "rerank_band does not match declared bounds"
        if not expected.any():
            if delivered.target_id.astype(str).tolist() != canonical.target_id.astype(str).tolist():
                return "a policy with no selected band rows must preserve canonical order"
            if delivered.ranking_basis.astype(str).tolist() != canonical.ranking_basis.astype(str).tolist():
                return "a policy with no selected band rows must preserve canonical ranking_basis"
        elif not delivered.ranking_basis.eq(BAND_RANKING_BASIS).all():
            return "experimental reranking must declare its ranking_basis"
    expected_result = band_rerank(canonical, keep=bounds["rerank_keep"], band=bounds["rerank_band_end"])
    columns = ["target_id", "daina_rank", "final_rank", "ranking_basis", "rerank_band"]
    for column in ("band_rrf_score", "band_rrf_sources"):
        if column in expected_result:
            if column not in delivered:
                return f"missing derived policy column: {column}"
            columns.append(column)
    actual_result = delivered[columns].reset_index(drop=True).copy()
    actual_result["rerank_band"] = flags.eq("true").to_numpy()
    if "band_rrf_sources" in columns:
        actual_result["band_rrf_sources"] = actual_result["band_rrf_sources"].fillna("")
    try:
        pd.testing.assert_frame_equal(
            actual_result, expected_result[columns].reset_index(drop=True),
            check_dtype=False, rtol=1e-12, atol=1e-12,
        )
    except AssertionError:
        return "delivered ordering or derived scores do not reproduce the declared policy from canonical inputs"
    return None


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
                        help="이 순위까지는 원래 검색 순서를 그대로 둡니다.")
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
