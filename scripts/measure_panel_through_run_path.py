#!/usr/bin/env python3
"""Measure the known-skin panel the way a researcher's run actually scores it.

The evaluation harness measures recipes against the retrieval index. That is the
right question for choosing a recipe and the wrong one for deciding whether to
switch Stage 3, because Stage 3 scores differently and against a different table.
This asks the question the switch actually poses:

    what does a researcher get today, and what would they get after?

    today   max-similarity over the ChEMBL mirror (data/chembl37)
    after   the promoted recipe over a production-role retrieval index

Both arms use leave-query-out, so a panel compound cannot retrieve itself. The
comparison is per compound-target pair, with exact McNemar on the discordant
pairs - the same test the panel arbitration note uses, because a difference that
cannot reach significance should not decide a production change.

Run:
    python scripts/measure_panel_through_run_path.py \\
        --production-index data/activity_retrieval_production_202608
"""

from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

PANEL = ROOT / "data" / "validation" / "skin_known_target_panel.csv"
CHEMBL_DIR = ROOT / "data" / "chembl37"
CLUSTERS = ROOT / "data" / "evidence_splits" / "screenable_target_clusters_2026_02.csv"
CLUSTER_MANIFEST = (
    ROOT / "data" / "evidence_splits" / "screenable_target_clusters_2026_02.manifest.json"
)
RECIPE = (
    ROOT / "results" / "eval" / "activity_retrieval_202608" / "dev_selection" / "recipe.json"
)
RANKS = (10, 30)


def exact_mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = min(b, c)
    return min(1.0, 2 * sum(comb(n, k) for k in range(tail + 1)) / 2**n)


def _rank_of(frame: pd.DataFrame, target: str) -> float:
    """1-based rank of a target, or inf when the arm cannot score it at all."""
    hit = frame.index[frame["target_id"] == target]
    return float(hit[0] + 1) if len(hit) else float("inf")


_CHEMBL_CACHE: dict[str, object] = {}


def _chembl_reference():
    """Load the 2.6M-row mirror once, not once per panel compound."""
    if not _CHEMBL_CACHE:
        from stage3_daina_zoete import (
            apply_quality_policy,
            load_activities,
            load_fingerprints,
        )

        print("  loading the ChEMBL mirror once...", flush=True)
        fingerprints = load_fingerprints(CHEMBL_DIR / "fp_morgan2_2048.parquet")
        activities = load_activities(
            CHEMBL_DIR / "human_activities.parquet",
            quality_policy="legacy",
            evidence_mode="leave-query-out",
        )
        activities, _ = apply_quality_policy(activities, "legacy")
        _CHEMBL_CACHE["fingerprints"] = fingerprints
        _CHEMBL_CACHE["activities"] = activities
    return _CHEMBL_CACHE["fingerprints"], _CHEMBL_CACHE["activities"]


def score_max_similarity(smiles: str, exclude: float) -> pd.DataFrame:
    """The arm that runs today."""
    from stage3_daina_zoete import score_query_against_reference
    from activity_retrieval_scoring import query_features

    fingerprints, activities = _chembl_reference()
    query_fp, connectivity, _, _ = query_features(smiles)
    frame, _ = score_query_against_reference(
        qfp=query_fp,
        query_connectivity_key=connectivity,
        fp_df=fingerprints,
        act=activities,
        evidence_mode="leave-query-out",
        exclude_reference_similarity=exclude,
        scoring_method="max-similarity",
    )
    return frame.reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-index", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, default=RECIPE)
    parser.add_argument(
        "--also-recipes",
        default="",
        help=(
            "comma-separated candidate recipe ids to measure alongside the "
            "promoted one, as a diagnostic. These have not passed any gate; "
            "loading the mirror and index dominates the cost, so comparing them "
            "in one pass is nearly free"
        ),
    )
    parser.add_argument("--exclude-reference-similarity", type=float, default=0.85)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    from stage3_recipe_scoring import load_promoted_recipe, score_query_with_recipe

    # score_query_with_recipe reloads the index per call; wrap it so the panel
    # pays that cost once instead of 22 times.
    _index_cache: dict[str, object] = {}

    def _cached_recipe_scores(**kwargs):
        from activity_retrieval_scoring import (
            apply_recipe,
            load_reference_index,
            query_features,
            score_query_features,
        )

        if "reference" not in _index_cache:
            print("  loading the production index once...", flush=True)
            reference, manifest, _ = load_reference_index(
                ligands_path=kwargs["ligands_path"],
                edges_path=kwargs["edges_path"],
                index_manifest_path=kwargs["index_manifest_path"],
                target_csv=kwargs["target_csv"],
                target_manifest_path=kwargs["target_manifest_path"],
                allow_production=True,
            )
            _index_cache["reference"] = reference
            _index_cache["manifest"] = manifest
        reference = _index_cache["reference"]
        query_fp, _, _, _ = query_features(kwargs["query_smiles"])
        features, _ = score_query_features(
            reference,
            query_fp,
            exclude_reference_similarity=kwargs["exclude_reference_similarity"],
        )
        scores = apply_recipe(features, kwargs["recipe"])
        keep = scores > 0.0
        frame = pd.DataFrame(
            {"target_id": reference.target_ids[keep], "score": scores[keep]}
        )
        return (
            frame.sort_values(
                ["score", "target_id"], ascending=[False, True], kind="mergesort"
            ).reset_index(drop=True),
            {},
        )

    recipe = load_promoted_recipe(args.recipe)
    from activity_retrieval_scoring import RECIPES

    by_id = {item.recipe_id: item for item in RECIPES}
    extra = []
    for name in (part.strip() for part in args.also_recipes.split(",")):
        if not name:
            continue
        if name not in by_id:
            raise SystemExit(f"unknown recipe id {name!r}; have {sorted(by_id)}")
        extra.append(by_id[name])
    arms = [(recipe, True)] + [(item, False) for item in extra]
    panel = pd.read_csv(PANEL)
    print(f"panel: {len(panel)} compounds, recipe: {recipe.recipe_id}")
    print("scoring both arms with leave-query-out; this takes a few minutes\n")

    rows = []
    for record in panel.itertuples(index=False):
        targets = [t.strip() for t in str(record.known_targets).split(";") if t.strip()]
        today = score_max_similarity(record.smiles, args.exclude_reference_similarity)
        after, _ = _cached_recipe_scores(
            query_smiles=record.smiles,
            recipe=recipe,
            ligands_path=args.production_index / "ligands.parquet",
            edges_path=args.production_index / "edges.parquet",
            index_manifest_path=args.production_index / "manifest.json",
            target_csv=CLUSTERS,
            target_manifest_path=CLUSTER_MANIFEST,
            exclude_reference_similarity=args.exclude_reference_similarity,
        )
        others = {}
        for item, _ in arms[1:]:
            frame_other, _ = _cached_recipe_scores(
                query_smiles=record.smiles,
                recipe=item,
                ligands_path=args.production_index / "ligands.parquet",
                edges_path=args.production_index / "edges.parquet",
                index_manifest_path=args.production_index / "manifest.json",
                target_csv=CLUSTERS,
                target_manifest_path=CLUSTER_MANIFEST,
                exclude_reference_similarity=args.exclude_reference_similarity,
            )
            others[item.recipe_id] = frame_other
        for target in targets:
            row = {
                "case_id": record.case_id,
                "target_id": target,
                "rank_today": _rank_of(today, target),
                "rank_after": _rank_of(after, target),
            }
            for name, frame_other in others.items():
                row[f"rank_{name}"] = _rank_of(frame_other, target)
            rows.append(row)
        print(f"  {record.case_id:22s} {len(targets)} pair(s)", flush=True)

    frame = pd.DataFrame(rows)
    print(f"\n=== {len(frame)} compound-target pairs ===")
    summary = {"pairs": int(len(frame)), "recipe_id": recipe.recipe_id, "ranks": {}}
    for k in RANKS:
        today_hit = frame["rank_today"] <= k
        after_hit = frame["rank_after"] <= k
        b = int((today_hit & ~after_hit).sum())
        c = int((~today_hit & after_hit).sum())
        p = exact_mcnemar(b, c)
        print(f"\nTop{k}")
        print(f"  today (max-similarity, ChEMBL) : {int(today_hit.sum()):>3} / {len(frame)}"
              f"  ({today_hit.mean():.3f})")
        print(f"  after (recipe, production idx) : {int(after_hit.sum()):>3} / {len(frame)}"
              f"  ({after_hit.mean():.3f})")
        print(f"  discordant: today-only {b}, after-only {c}")
        print(f"  exact McNemar p = {p:.4f}"
              f"   -> {'resolves' if p <= 0.05 else 'does not resolve'}")
        summary["ranks"][f"top{k}"] = {
            "today": int(today_hit.sum()),
            "after": int(after_hit.sum()),
            "today_only": b,
            "after_only": c,
            "p": p,
        }

    for item, _ in arms[1:]:
        column = f"rank_{item.recipe_id}"
        print(f"\n--- diagnostic arm: {item.recipe_id} (not gated) ---")
        for k in RANKS:
            today_hit = frame["rank_today"] <= k
            other_hit = frame[column] <= k
            b = int((today_hit & ~other_hit).sum())
            c = int((~today_hit & other_hit).sum())
            print(
                f"  Top{k}: today {int(today_hit.sum()):>3}/{len(frame)} -> "
                f"{int(other_hit.sum()):>3}/{len(frame)}   "
                f"discordant {b}:{c}   p = {exact_mcnemar(b, c):.4f}"
            )
        lost = frame[(frame["rank_today"] <= 30) & (frame[column] > 30)]
        if len(lost):
            print(f"  loses at Top30: "
                  + ", ".join(f"{r.case_id}->{r.target_id}" for r in lost.itertuples()))
        else:
            print("  loses at Top30: nothing")
        summary.setdefault("diagnostic_arms", {})[item.recipe_id] = {
            f"top{k}": int((frame[column] <= k).sum()) for k in RANKS
        }

    unreachable = frame[np.isinf(frame["rank_after"]) & ~np.isinf(frame["rank_today"])]
    if len(unreachable):
        print(f"\n{len(unreachable)} pair(s) reachable today and not after:")
        for record in unreachable.itertuples(index=False):
            print(f"  {record.case_id} -> {record.target_id}")
    summary["lost_pairs"] = unreachable[["case_id", "target_id"]].to_dict(orient="records")

    if args.json:
        args.json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        frame.to_csv(args.json.with_suffix(".csv"), index=False)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
