#!/usr/bin/env python3
"""analyse_skin_sens_benchmark.py — how good is the skin-sensitisation screen?

Until 2026-08-30 nothing in this repository had ever compared the safety layer
against measured data, so a reader was shown HALT/FLAG_HIGH/PASS with no basis
for believing any of it. This scores the shipped rule against public
LLNA-derived labels.

The shipped rule is >= 2 of 3 positive -> HALT, 1 -> FLAG_HIGH, 0 -> PASS.
On 100 stratified compounds it issues PASS three times and is right all three;
it flags the other 97. It is a conservative screen that almost never clears
anything, not a discriminator.

    python scripts/analyse_skin_sens_benchmark.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "data" / "validation" / "skin_sens_benchmark_20260830"
MODELS = ("husspred", "stoptox", "pred_skin")
HALT_MIN_VOTES = 2


def load(directory: Path) -> pd.DataFrame:
    frame = pd.read_csv(directory / "predictions.csv")
    for model in MODELS:
        frame[f"{model}_pos"] = (
            frame[f"{model}_call"].astype(str).str.lower() == "positive"
        ).astype(int)
    frame["votes"] = frame[[f"{model}_pos" for model in MODELS]].sum(axis=1)
    frame["decision"] = np.where(
        frame.votes >= HALT_MIN_VOTES,
        "HALT",
        np.where(frame.votes == 1, "FLAG_HIGH", "PASS"),
    )
    return frame


def binary_metrics(predicted: pd.Series, truth: pd.Series) -> dict[str, float]:
    tp = int(((predicted == 1) & (truth == 1)).sum())
    tn = int(((predicted == 0) & (truth == 0)).sum())
    fp = int(((predicted == 1) & (truth == 0)).sum())
    fn = int(((predicted == 0) & (truth == 1)).sum())
    total = tp + tn + fp + fn
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    return {
        "n": total,
        "accuracy": (tp + tn) / total if total else float("nan"),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced": (sensitivity + specificity) / 2,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()

    frame = load(args.dir)
    print(
        f"표본 {len(frame)} (감작원 {int(frame.truth.sum())} / "
        f"비감작원 {int((1 - frame.truth).sum())}), 기저율 {frame.truth.mean():.1%}\n"
    )

    print("=== 출하된 규칙 (>=2 HALT / 1 FLAG_HIGH / 0 PASS) ===")
    print(f"{'판정':11s} {'건수':>5s} {'실제 감작원':>11s} {'비율':>7s}")
    for decision in ("HALT", "FLAG_HIGH", "PASS"):
        subset = frame[frame.decision == decision]
        if subset.empty:
            continue
        print(
            f"{decision:11s} {len(subset):5d} {int(subset.truth.sum()):11d} "
            f"{subset.truth.mean():7.1%}"
        )
    cleared = frame[frame.decision == "PASS"]
    print(
        f"\nPASS로 통과한 {len(cleared)}건 중 실제 감작원 "
        f"{int(cleared.truth.sum())}건 — 이 규칙은 거의 아무것도 통과시키지 않는다"
    )

    print("\n=== 모델별 ===")
    print(f"{'모델':12s} {'정확도':>7s} {'민감도':>7s} {'특이도':>7s} {'균형':>7s} {'AUC':>6s}")
    for model in MODELS:
        metrics = binary_metrics(frame[f"{model}_pos"], frame.truth)
        auc = _auc(frame.truth, frame[f"{model}_prob"])
        print(
            f"{model:12s} {metrics['accuracy']:7.1%} {metrics['sensitivity']:7.1%} "
            f"{metrics['specificity']:7.1%} {metrics['balanced']:7.1%} "
            f"{auc if isinstance(auc, str) else f'{auc:6.3f}'}"
        )
    vote_auc = _auc(frame.truth, frame.votes)
    print(f"{'투표수':12s} {'':7s} {'':7s} {'':7s} {'':7s} "
          f"{vote_auc if isinstance(vote_auc, str) else f'{vote_auc:6.3f}'}")

    print(
        "\n주의: 이 모델들이 이 공개 데이터로 학습됐는지 알 수 없다. "
        "높은 점수를 일반화 성능으로 읽으면 안 된다 (data/validation/"
        "skin_sens_benchmark_20260830/SOURCE.md)."
    )


def _auc(truth: pd.Series, score: pd.Series):
    """Rank-based AUC, computed here so the finding never depends on an extra
    package being installed. Ties get their average rank, which is what makes
    this agree with scikit-learn on StopTox's eleven discrete levels."""
    usable = score.notna()
    labels = truth[usable]
    values = score[usable]
    if labels.nunique() < 2:
        return "   n/a"
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    ranks = values.rank(method="average")
    rank_sum = float(ranks[labels == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


if __name__ == "__main__":
    main()
