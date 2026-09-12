#!/usr/bin/env python3
"""Can the 15-compound skin panel decide what it is being used to decide?

The activity-retrieval recipe `union_p6_consensus` beats the frozen production
baseline by +12.7 points of Top30 on the 256-query dev panel. It was not promoted
because the 15-compound known-skin panel regressed by 0.125 of Top30. Stage 3
ran `chembl_p5_max` until 2026-08-31, when the panel was extended to 22
compounds and the regression failed to reproduce.

That panel holds 32 compound-target pairs, so 0.625 -> 0.500 is a difference of
four pairs. This asks whether four pairs out of 32 can support the decision they
are making, and how large the panel would have to be before it could.

Run:
    python scripts/analyse_panel_arbitration_power.py
"""

from __future__ import annotations

import argparse
from math import comb
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "validation" / "skin_known_target_panel.csv"

# docs/SKIN_KNOWN_TARGET_VALIDATION.md, the table that blocked promotion.
BASELINE_TOP30 = 0.625
SELECTED_TOP30 = 0.500
DEV_BASELINE_TOP30 = 0.618012
DEV_SELECTED_TOP30 = 0.745342


def exact_mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact McNemar on the discordant pairs only."""
    n = b + c
    if n == 0:
        return 1.0
    tail = min(b, c)
    p = sum(comb(n, k) for k in range(0, tail + 1)) / 2**n
    return min(1.0, 2 * p)


def best_case_p(difference: int) -> float:
    """The most significant result a difference of this size could ever give.

    Every discordant pair falling the same way is the arrangement most favourable
    to significance; anything else is less significant. If even this is above
    0.05, the comparison cannot resolve at that panel size, whatever happened.
    """
    return exact_mcnemar_p(difference, 0)


def pairs_needed(effect: float, alpha: float = 0.05) -> int:
    """Smallest all-one-way discordance that could clear alpha."""
    discordant = 1
    while best_case_p(discordant) > alpha:
        discordant += 1
        if discordant > 200:
            break
    # The observed effect is a fraction of the panel, so back out the panel size.
    return int(round(discordant / effect))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    panel = pd.read_csv(PANEL)
    compounds = len(panel)
    pairs = sum(
        len(str(value).split(";")) for value in panel["known_targets"] if pd.notna(value)
    )
    effect = BASELINE_TOP30 - SELECTED_TOP30
    difference = round(effect * pairs)

    print("=== the panel that decides ===")
    print(f"  compounds                 : {compounds}")
    print(f"  compound-target pairs     : {pairs}")
    print(f"  Top30 baseline / selected : {BASELINE_TOP30:.3f} / {SELECTED_TOP30:.3f}")
    print(f"  that is a difference of   : {difference} pairs")

    print()
    print("=== can four pairs decide it ===")
    ceiling = best_case_p(difference)
    print(f"  best case exact McNemar p : {ceiling:.4f}")
    print(f"  resolves at alpha={args.alpha}      : {'yes' if ceiling <= args.alpha else 'NO'}")
    if ceiling > args.alpha:
        print("  -> no arrangement of these pairs can reach significance")

    print()
    print("=== what the other panel says ===")
    dev = DEV_SELECTED_TOP30 - DEV_BASELINE_TOP30
    print(f"  dev panel (256 queries) Top30: {DEV_BASELINE_TOP30:.3f} -> {DEV_SELECTED_TOP30:.3f}"
          f"  ({dev:+.3f})")
    print(f"  skin panel ({pairs} pairs)   Top30: {BASELINE_TOP30:.3f} -> {SELECTED_TOP30:.3f}"
          f"  ({-effect:+.3f})")
    print("  the smaller panel is overriding the larger one")

    print()
    print("=== how big would it have to be ===")
    needed = pairs_needed(effect, args.alpha)
    print(f"  pairs to resolve a {effect:.3f} effect: about {needed}")
    print(f"  today: {pairs}  ->  roughly {needed / pairs:.0f}x more")
    print(f"  at ~{pairs / compounds:.1f} pairs per compound, about {round(needed * compounds / pairs)} compounds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
