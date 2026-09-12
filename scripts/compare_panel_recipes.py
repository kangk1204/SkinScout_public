#!/usr/bin/env python3
"""Compare two retrieval recipes on the known-skin panel, with a significance test.

The evaluator reports each recipe's Top10/Top30 but not whether the difference
between them means anything. That gap is how a four-pair difference on 32 pairs
came to block a change: nobody had asked exact McNemar whether four pairs could
say that.

Run it on an `evaluate` output directory:

    python scripts/compare_panel_recipes.py \
        results/eval/activity_retrieval_202608/panel_v3_diagnostic
"""

import sys
from math import comb
from pathlib import Path

import pandas as pd

S = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
BASE, CAND = "chembl_p5_max", "union_p6_consensus"


def exact_mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = min(b, c)
    return min(1.0, 2 * sum(comb(n, k) for k in range(tail + 1)) / 2**n)


targets = pd.read_csv(S / "known_target_metrics.csv")
print("target metrics columns:", list(targets.columns))
print()

pivot_cols = [c for c in targets.columns if "top" in c.lower() or "rank" in c.lower()]
print(targets.head(4).to_string(index=False)[:400])
print()

# Pair-level: one row per (case, truth target) per recipe.
key = [c for c in ("query_id", "case_id") if c in targets.columns]
key += [c for c in ("uniprot", "target_id", "truth_target") if c in targets.columns]
print("pair key:", key)

for k in (10, 30):
    col = next((c for c in targets.columns if c.endswith(f"top{k}")), None)
    if col is None:
        continue
    wide = targets.pivot_table(index=key, columns="recipe_id", values=col)
    if BASE not in wide or CAND not in wide:
        continue
    wide = wide.dropna()
    base_hits = int(wide[BASE].sum())
    cand_hits = int(wide[CAND].sum())
    b = int(((wide[BASE] == 1) & (wide[CAND] == 0)).sum())  # baseline only
    c = int(((wide[BASE] == 0) & (wide[CAND] == 1)).sum())  # candidate only
    p = exact_mcnemar(b, c)
    n = len(wide)
    print()
    print(f"=== Top{k} on {n} pairs ===")
    print(f"  {BASE:20s} {base_hits:>3} / {n}  ({base_hits/n:.3f})")
    print(f"  {CAND:20s} {cand_hits:>3} / {n}  ({cand_hits/n:.3f})")
    print(f"  difference            {cand_hits - base_hits:+3d} pairs")
    print(f"  discordant: baseline-only {b}, candidate-only {c}")
    print(f"  exact McNemar p = {p:.4f}   -> {'RESOLVES' if p <= 0.05 else 'does not resolve'}")
