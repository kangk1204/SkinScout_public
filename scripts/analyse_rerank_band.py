#!/usr/bin/env python3
"""analyse_rerank_band.py — does docking re-ranking improve the Daina order?

Reproduces every number in docs/RERANK_EXPERIMENT_20260828.md from the score
tables committed under data/validation/rerank_band_20260828/. The measured
answer is that re-ranking the whole list does not help, but re-ranking only the
middle band does: keeping Daina's top 10 untouched and re-ordering ranks 11-50
by RRF moves top-30 recovery from 14/28 to 20/28 (Wilcoxon p = 0.0068).

    python scripts/analyse_rerank_band.py
    python scripts/analyse_rerank_band.py --keep 10 --band 50 --sweep
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORES = ROOT / "data" / "validation" / "rerank_band_20260828"
# Tracked alongside the score tables so every number in the report can be
# recomputed from the repository alone; results/eval/ is gitignored.
DEFAULT_TRUTH = DEFAULT_SCORES / "known_targets.csv"
RRF_K = 60


def load_cases(scores_dir: Path, truth_path: Path) -> tuple[dict, dict]:
    truth = pd.read_csv(truth_path)
    known: dict[str, list[tuple[str, str]]] = {}
    for record in truth.to_dict("records"):
        known.setdefault(record["case_id"], []).append(
            (record["target_id"], record["target_label"])
        )
    frames: dict[str, pd.DataFrame] = {}
    for directory in sorted(scores_dir.iterdir()):
        if not directory.is_dir() or directory.name not in known:
            continue
        selected = pd.read_csv(directory / "selected.csv")
        autodock = pd.read_csv(directory / "autodock.tsv", sep="\t")
        gnina = pd.read_csv(directory / "gnina.tsv", sep="\t")
        frame = selected.merge(
            autodock[["target_id", "vina_score"]], on="target_id", how="left"
        ).merge(gnina[["target_id", "cnn_affinity"]], on="target_id", how="left")
        frame["dg_rank"] = frame["vina_score"].rank(method="min")
        frame["gnina_rank"] = frame["cnn_affinity"].rank(method="min", ascending=False)
        missing = len(frame) + 1
        frame["rrf"] = (
            1.0 / (RRF_K + frame["daina_rank"])
            + 1.0 / (RRF_K + frame["dg_rank"].fillna(missing))
            + 1.0 / (RRF_K + frame["gnina_rank"].fillna(missing))
        )
        frame["rrf_rank"] = frame["rrf"].rank(method="min", ascending=False)
        frames[directory.name] = frame
    return frames, known


def hybrid_order(frame: pd.DataFrame, keep: int, band: int) -> pd.DataFrame:
    """Daina's head is left alone; only the middle band is re-ordered."""
    head = frame[frame.daina_rank <= keep].sort_values("daina_rank")
    middle = frame[
        (frame.daina_rank > keep) & (frame.daina_rank <= band)
    ].sort_values("rrf", ascending=False)
    tail = frame[frame.daina_rank > band].sort_values("daina_rank")
    ordered = pd.concat([head, middle, tail]).reset_index(drop=True)
    ordered["hybrid_rank"] = np.arange(1, len(ordered) + 1)
    return ordered


def collect(frames: dict, known: dict, keep: int, band: int) -> pd.DataFrame:
    rows = []
    for case, frame in frames.items():
        indexed = hybrid_order(frame, keep, band).set_index("target_id")
        for target_id, label in known[case]:
            if target_id not in indexed.index:
                continue
            record = indexed.loc[target_id]
            rows.append({
                "compound": case,
                "target": label,
                "daina": float(record["daina_rank"]),
                "dg": float(record["dg_rank"]),
                "gnina": float(record["gnina_rank"]),
                "rrf": float(record["rrf_rank"]),
                "hybrid": float(record["hybrid_rank"]),
            })
    return pd.DataFrame(rows)


def _wilcoxon(left: pd.Series, right: pd.Series) -> float:
    from scipy.stats import wilcoxon

    if (left != right).sum() == 0:
        return 1.0
    return float(wilcoxon(left, right)[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--keep", type=int, default=10)
    parser.add_argument("--band", type=int, default=50)
    parser.add_argument("--sweep", action="store_true",
                        help="Show whether the thresholds are load-bearing.")
    args = parser.parse_args()

    if not args.truth.exists():
        raise SystemExit(f"정답 파일이 없습니다: {args.truth}")
    frames, known = load_cases(args.scores, args.truth)
    if not frames:
        raise SystemExit(f"점수 표를 찾지 못했습니다: {args.scores}")
    table = collect(frames, known, args.keep, args.band)

    print(f"화합물 {table.compound.nunique()}개 / {len(table)}쌍 "
          f"(무작위 기대 순위 ≈ {len(frames[next(iter(frames))]) / 2:.0f})")
    print(f"\n{'방법':10s} {'평균':>7s} {'중앙값':>7s} {'top-10':>8s} {'top-30':>8s}")
    for column, label in (("daina", "Daina"), ("dg", "ΔG"), ("gnina", "GNINA"),
                          ("rrf", "RRF 전면"), ("hybrid", f"Hybrid {args.keep}/{args.band}")):
        values = table[column].dropna()
        print(f"{label:10s} {values.mean():7.1f} {values.median():7.1f} "
              f"{int((values <= 10).sum()):5d}/{len(values):<2d} "
              f"{int((values <= 30).sum()):5d}/{len(values):<2d}")

    better = int((table.hybrid < table.daina).sum())
    worse = int((table.hybrid > table.daina).sum())
    print(f"\nHybrid vs Daina: 개선 {better} / 악화 {worse} / "
          f"동일 {len(table) - better - worse}, "
          f"Wilcoxon p = {_wilcoxon(table.daina, table.hybrid):.4f}")

    print("\n=== Daina 순위 구간별 전면 RRF 효과 ===")
    for low, high, label in ((1, 10, "1-10위"), (11, 50, "11-50위"), (51, 10**6, "51위 이하")):
        subset = table[(table.daina >= low) & (table.daina <= high)]
        if subset.empty:
            continue
        print(f"  {label:10s} n={len(subset):2d}  중앙값 변화 "
              f"{(subset.rrf - subset.daina).median():+6.0f}  "
              f"개선 {int((subset.rrf < subset.daina).sum())} / "
              f"악화 {int((subset.rrf > subset.daina).sum())}")

    if args.sweep:
        print("\n=== 임계값 민감도 ===")
        print(f"{'keep':>5s} {'band':>5s} {'평균':>7s} {'top-30':>8s} {'p':>8s}")
        for keep in (5, 8, 10, 12, 15, 20):
            for band in (30, 40, 50, 60, 80, 100):
                if band <= keep:
                    continue
                swept = collect(frames, known, keep, band)
                probability = _wilcoxon(swept.daina, swept.hybrid)
                print(f"{keep:5d} {band:5d} {swept.hybrid.mean():7.1f} "
                      f"{int((swept.hybrid <= 30).sum()):5d}/{len(swept):<2d} "
                      f"{probability:8.4f}{' *' if probability < 0.05 else ''}")

        print("\n=== 화합물 하나씩 제외 ===")
        for case in sorted(table.compound.unique()):
            subset = table[table.compound != case]
            probability = _wilcoxon(subset.daina, subset.hybrid)
            print(f"  {case:22s} n={len(subset):2d}  "
                  f"top-30 {int((subset.daina <= 30).sum())}→"
                  f"{int((subset.hybrid <= 30).sum())}  p={probability:.4f}"
                  f"{' *' if probability < 0.05 else ''}")


if __name__ == "__main__":
    main()
