#!/usr/bin/env python3
"""measure_pocket_transfer.py — does pocket-cluster transfer earn its coverage?

Coverage route B lends a scorable target's retrieval score to its Foldseek
pocket-cluster mates, which is how the older analysis reported +2,082 newly
scorable targets. Whether the targets it reaches are then *ranked* usefully was
never measured on a panel the transfer could not have memorised.

This scores the pocket-cold panel twice - once as the model ranks it, once with
transfer applied - and reports where the held-out truth target lands. Both runs
use the same reference index and the same leave-query-out exclusion, so the
only difference is the transfer.

    python eval/measure_pocket_transfer.py --view strict --level tm40
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from activity_retrieval_model import (  # noqa: E402
    BASELINE,
    apply_recipe,
    load_ranking_panel,
    load_reference_index,
    query_features,
    score_query_features,
    scorable_target_mask,
)
from evidence_transfer import load_cluster_map, transfer_scores  # noqa: E402

DEFAULT_PANELS = ROOT / "data" / "pocket_cold_panel_202608"
DEFAULT_INDEX = ROOT / "data" / "activity_retrieval_202608"
DEFAULT_CLUSTERS = ROOT / "data" / "evidence_splits" / "screenable_target_clusters_2026_02.csv"
DEFAULT_POCKETS = (
    ROOT / "data" / "pocket_clusters_screenable_202608" / "pocket_cluster_map.csv"
)


def _ranks(scores: np.ndarray) -> np.ndarray:
    """Competition ranks, best first, ties sharing the best position."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    position = 1
    index = 0
    while index < len(order):
        stop = index
        while stop + 1 < len(order) and sorted_scores[stop + 1] == sorted_scores[index]:
            stop += 1
        ranks[order[index:stop + 1]] = position
        position += stop - index + 1
        index = stop + 1
    return ranks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", type=Path, default=DEFAULT_PANELS)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--clusters", type=Path, default=DEFAULT_CLUSTERS)
    parser.add_argument("--pockets", type=Path, default=DEFAULT_POCKETS)
    parser.add_argument("--view", default="strict", choices=("strict", "extended"))
    parser.add_argument("--level", default="tm40", choices=("tm40", "tm50", "tm60"))
    parser.add_argument("--discount", type=float, default=0.5)
    parser.add_argument("--exclude-reference-similarity", type=float, default=0.85)
    parser.add_argument("--limit", type=int, default=0, help="0이면 전체")
    parser.add_argument("--out-json", type=Path)
    parser.add_argument(
        "--panel-path",
        type=Path,
        help="랭킹 패널을 직접 지정합니다. 주지 않으면 pocket-cold 패널을 씁니다.",
    )
    args = parser.parse_args()

    panel_path = args.panel_path or (
        args.panels / f"pocket_cold_{args.level}_{args.view}_ranking.parquet"
    )
    panel = load_ranking_panel(panel_path, "test")
    if args.limit:
        panel = panel.head(args.limit)

    reference, _index_manifest, _targets = load_reference_index(
        ligands_path=args.index / "ligands.parquet",
        edges_path=args.index / "edges.parquet",
        index_manifest_path=args.index / "manifest.json",
        target_csv=args.clusters,
        target_manifest_path=args.clusters.with_suffix(".manifest.json"),
    )
    target_ids = reference.target_ids.tolist()
    scorable = scorable_target_mask(reference)
    cluster_of = load_cluster_map(args.pockets, f"pocket_cluster_{args.level}")
    print(
        f"패널 {len(panel)} 질의 | 표적 우주 {len(target_ids):,} "
        f"(직접 채점 가능 {int(scorable.sum()):,}) | 포켓 클러스터 맵 {len(cluster_of):,}",
        flush=True,
    )

    rows = []
    for position, record in enumerate(panel.to_dict("records"), 1):
        fingerprint = query_features(record["canonical_smiles"])[0]
        features, _stats = score_query_features(
            reference,
            fingerprint,
            exclude_reference_similarity=args.exclude_reference_similarity,
        )
        base = apply_recipe(features, BASELINE)
        moved, records = transfer_scores(
            target_ids, base, scorable, cluster_of,
            discount=args.discount, cluster_level=f"pocket_cluster_{args.level}",
        )
        base_rank = _ranks(base)
        moved_rank = _ranks(moved)
        index_of = {tid: i for i, tid in enumerate(target_ids)}
        for truth in record["truth_targets"]:
            slot = index_of.get(str(truth))
            if slot is None:
                continue
            rows.append({
                "query_id": record["query_id"],
                "target": str(truth),
                "directly_scorable": bool(scorable[slot]),
                "base_score": float(base[slot]),
                "transfer_score": float(moved[slot]),
                "base_rank": float(base_rank[slot]),
                "transfer_rank": float(moved_rank[slot]),
                "transfers_made": len(records),
            })
        if position % 50 == 0:
            print(f"  {position}/{len(panel)}", flush=True)

    table = pd.DataFrame(rows)
    if table.empty:
        raise SystemExit("패널 표적이 표적 우주에 하나도 없습니다")

    universe = len(target_ids)
    summary = {
        "schema_version": "skinscout.pocket-transfer-measurement.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "view": args.view,
        "level": args.level,
        "discount": args.discount,
        "exclude_reference_similarity": args.exclude_reference_similarity,
        "pairs": int(len(table)),
        "target_universe": universe,
        "chance_rank": universe / 2,
        "directly_scorable_pairs": int(table.directly_scorable.sum()),
        "base": _describe(table.base_rank, universe),
        "transfer": _describe(table.transfer_rank, universe),
        "improved": int((table.transfer_rank < table.base_rank).sum()),
        "worsened": int((table.transfer_rank > table.base_rank).sum()),
        "unchanged": int((table.transfer_rank == table.base_rank).sum()),
    }

    print(f"\n표적 우주 {universe:,} (무작위 기대 순위 {universe/2:,.0f})")
    print(f"쌍 {len(table)} | 그중 직접 채점 가능 {int(table.directly_scorable.sum())}")
    print(f"\n{'':10s} {'평균':>9s} {'중앙값':>9s} {'top-10':>8s} {'top-100':>8s} {'top-1%':>8s}")
    for label, column in (("전달 없음", "base_rank"), ("전달 적용", "transfer_rank")):
        stats = _describe(table[column], universe)
        print(
            f"{label:10s} {stats['mean']:9.1f} {stats['median']:9.1f} "
            f"{stats['top10']:6d}/{len(table):<3d} {stats['top100']:6d}/{len(table):<3d} "
            f"{stats['top1pct']:6d}/{len(table):<3d}"
        )
    print(f"\n개선 {summary['improved']} / 악화 {summary['worsened']} / 동일 {summary['unchanged']}")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        table.to_csv(args.out_json.with_suffix(".pairs.csv"), index=False)
        print(f"\n요약 → {args.out_json}")


def _describe(series: pd.Series, target_universe: int) -> dict[str, float]:
    if target_universe < 1:
        raise ValueError("target_universe must be positive")
    top_one_percent = max(1, math.ceil(target_universe * 0.01))
    return {
        "mean": float(series.mean()),
        "median": float(series.median()),
        "top10": int((series <= 10).sum()),
        "top100": int((series <= 100).sum()),
        "top1pct": int((series <= top_one_percent).sum()),
        "top1pct_rank_cutoff": top_one_percent,
    }


if __name__ == "__main__":
    main()
