#!/usr/bin/env python3
"""Emit the nearest-analogue similarity and rank behind the README's band table.

The band table - ">=0.6 recovers, <0.4 does not" - is the first thing a
researcher reads and the colour the Workbench paints on every row. It was
computed once by hand and left as a CSV with no generator, so nothing could
re-derive it when the index changed and nothing could check the UI against it.

For each (compound, known target) pair it writes the leave-query-out max
Tanimoto to a measured ligand of that target, and the rank the promoted recipe
gives that target. Bands are cut at the lower edge inclusively, matching
`similarityCell` in workbench/static/app.js.

Run:
    python scripts/measure_panel_similarity_bands.py \\
        --production-index data/activity_retrieval_runtime_merged_202608
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

PANEL = ROOT / "data" / "validation" / "skin_known_target_panel.csv"
CLUSTERS = ROOT / "data" / "evidence_splits" / "screenable_target_clusters_2026_02.csv"
CLUSTER_MANIFEST = (
    ROOT / "data" / "evidence_splits" / "screenable_target_clusters_2026_02.manifest.json"
)
RECIPE = (
    ROOT / "results" / "eval" / "activity_retrieval_202608" / "dev_selection" / "recipe.json"
)
BANDS = ((0.6, "high"), (0.4, "mid"), (0.0, "low"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-index", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, default=RECIPE)
    parser.add_argument("--exclude-reference-similarity", type=float, default=0.85)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from rdkit import DataStructs

    from activity_retrieval_scoring import (
        load_reference_index,
        query_features,
        score_query_features,
    )
    from stage3_recipe_scoring import load_promoted_recipe
    from activity_retrieval_scoring import apply_recipe

    recipe = load_promoted_recipe(args.recipe)
    reference, _, _ = load_reference_index(
        ligands_path=args.production_index / "ligands.parquet",
        edges_path=args.production_index / "edges.parquet",
        index_manifest_path=args.production_index / "manifest.json",
        target_csv=CLUSTERS,
        target_manifest_path=CLUSTER_MANIFEST,
        allow_production=True,
    )
    target_position = {str(t): i for i, t in enumerate(reference.target_ids)}

    rows = []
    panel = pd.read_csv(PANEL)
    for record in panel.itertuples(index=False):
        query_fp, _, _, _ = query_features(record.smiles)
        features, _ = score_query_features(
            reference, query_fp, exclude_reference_similarity=args.exclude_reference_similarity
        )
        scores = apply_recipe(features, recipe)
        order = np.argsort(-scores, kind="mergesort")
        rank_of = {int(index): position + 1 for position, index in enumerate(order)}

        similarities = np.asarray(
            DataStructs.BulkTanimotoSimilarity(query_fp, reference.fingerprints),
            dtype=np.float64,
        )
        eligible = similarities < args.exclude_reference_similarity
        edge_sim = np.where(
            eligible[reference.edge_ligand], similarities[reference.edge_ligand], -1.0
        )
        best = np.zeros(len(reference.target_ids))
        np.maximum.at(best, reference.edge_target, edge_sim)

        for target in (t.strip() for t in str(record.known_targets).split(";") if t.strip()):
            index = target_position.get(target)
            if index is None:
                # Outside the screenable universe: it has no rank to report, and
                # silently dropping it would shrink the panel without saying so.
                rows.append({"case_id": record.case_id, "target_id": target,
                             "sim": float("nan"), "rank": float("inf")})
                continue
            rows.append({
                "case_id": record.case_id,
                "target_id": target,
                "sim": round(float(max(best[index], 0.0)), 6),
                "rank": float(rank_of[index]),
            })
        print(f"  {record.case_id:22s} done", flush=True)

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame[["sim", "rank"]].to_csv(args.out, index=False)
    frame.to_csv(args.out.with_name(args.out.stem + "_detail.csv"), index=False)

    print(f"\n{'band':10s} {'n':>4s} {'Top30':>8s}  {'median rank':>12s}")
    for low, name in BANDS:
        high = 1.01 if low == 0.6 else (0.6 if low == 0.4 else 0.4)
        band = frame[(frame["sim"] >= low) & (frame["sim"] < high)]
        hits = int((band["rank"] <= 30).sum())
        median = band["rank"].median() if len(band) else float("nan")
        print(f"{name:10s} {len(band):>4d} {hits:>4d}/{len(band):<3d} {median:>12.1f}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
