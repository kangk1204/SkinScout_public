#!/usr/bin/env python3
"""analyse_receptor_background.py — does per-receptor background help recovery?

Route C in docs/COVERAGE_ROUTES_20260824.md established that a receptor's own
docking background carries real bias and that subtracting it removes 71% of
that bias, then sized the production build at about eight days and left the
question open: does removing the bias make known targets rank better?

It does not. Measured over the committed panel score tables, background
normalisation moves 9 pairs up and 10 down (Wilcoxon p = 0.83). The reason is
that AutoDock dG is already near chance for recovery, and de-biasing a
near-chance signal leaves a near-chance signal.

    python scripts/analyse_receptor_background.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORES = ROOT / "data" / "validation" / "rerank_band_20260828"
# A background built from too few compounds is noise, not a background.
DEFAULT_MIN_BACKGROUND = 6
MIN_COMMON_RECEPTORS = 30


def load(scores_dir: Path) -> tuple[dict[str, pd.Series], dict[str, list[str]]]:
    truth = pd.read_csv(scores_dir / "known_targets.csv")
    known: dict[str, list[str]] = {}
    for record in truth.to_dict("records"):
        known.setdefault(record["case_id"], []).append(record["target_id"])
    scores: dict[str, pd.Series] = {}
    for directory in sorted(scores_dir.iterdir()):
        if not directory.is_dir():
            continue
        table = pd.read_csv(directory / "autodock.tsv", sep="\t").dropna(
            subset=["vina_score"]
        )
        scores[directory.name] = table.set_index("target_id")["vina_score"]
    return scores, known


def compare(
    scores: dict[str, pd.Series],
    known: dict[str, list[str]],
    *,
    min_background: int = DEFAULT_MIN_BACKGROUND,
) -> pd.DataFrame:
    rows = []
    for case, mine in scores.items():
        if case not in known:
            continue
        # Built from the other compounds only, so the query never sees itself.
        others = [series for name, series in scores.items() if name != case]
        if not others:
            continue
        stacked = pd.concat(others, axis=1)
        background = stacked.mean(axis=1)
        depth = stacked.notna().sum(axis=1)
        usable = depth[depth >= min_background].index
        common = mine.index.intersection(usable)
        if len(common) < MIN_COMMON_RECEPTORS:
            continue
        raw = mine.loc[common]
        normalised = raw - background.loc[common]
        raw_rank = raw.rank(method="min")
        normalised_rank = normalised.rank(method="min")
        for target_id in known[case]:
            if target_id not in common:
                continue
            rows.append({
                "compound": case,
                "target": target_id,
                "raw": int(raw_rank.loc[target_id]),
                "normalised": int(normalised_rank.loc[target_id]),
                "candidates": len(common),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--min-background", type=int, default=DEFAULT_MIN_BACKGROUND)
    args = parser.parse_args()

    scores, known = load(args.scores)
    table = compare(scores, known, min_background=args.min_background)
    if table.empty:
        raise SystemExit("배경을 만들 수 있는 쌍이 없습니다")

    print(table.to_string(index=False))
    print(
        f"\n쌍 {len(table)}개 · 후보 평균 {table.candidates.mean():.0f}개 "
        f"(무작위 기대 {table.candidates.mean() / 2:.0f})"
    )
    print(f"  원시 ΔG      평균 {table.raw.mean():6.1f}  중앙값 {table.raw.median():5.1f}")
    print(
        f"  배경 정규화   평균 {table.normalised.mean():6.1f}  "
        f"중앙값 {table.normalised.median():5.1f}"
    )
    better = int((table.normalised < table.raw).sum())
    worse = int((table.normalised > table.raw).sum())
    print(f"  개선 {better} / 악화 {worse}")
    try:
        from scipy.stats import wilcoxon

        if (table.raw != table.normalised).any():
            print(f"  Wilcoxon p = {wilcoxon(table.raw, table.normalised)[1]:.4f}")
    except ImportError:  # pragma: no cover
        print("  (scipy 없음 - 유의성 생략)")


if __name__ == "__main__":
    main()
