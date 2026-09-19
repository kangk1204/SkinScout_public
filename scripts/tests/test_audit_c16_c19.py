"""감사 보고서 C16–C19 회귀: 가중치 유한성, tie 안정성, band 사전조건."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


def test_candidate_score_rejects_non_finite_weights() -> None:
    import candidate_score

    frame = pd.DataFrame({"target_id": ["T1"], "structure": [0.5], "evidence": [0.5], "safety": [0.5]})
    for bad in (float("nan"), float("inf"), -0.1, True):
        with pytest.raises(ValueError):
            candidate_score.score_candidates(frame, weights={"structure": bad})


def _band_frame(rows: list[list[object]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=[
            "target_id",
            "daina_rank",
            "daina_score",
            "ranking_basis",
            "autodock_energy_kcal_mol",
            "gnina_cnn_affinity",
        ],
    )


def test_band_rerank_rejects_duplicate_targets_and_ranks() -> None:
    from stage3_band_rerank import band_rerank

    duplicate_target = _band_frame(
        [
            ["T1", 1, 0.5, "daina_max_tanimoto", -7.0, 5.0],
            ["T1", 2, 0.4, "daina_max_tanimoto", -6.0, 4.5],
        ]
    )
    with pytest.raises(SystemExit, match="target_id가 중복"):
        band_rerank(duplicate_target, keep=5, band=10)
    duplicate_rank = _band_frame(
        [
            ["T1", 1, 0.5, "daina_max_tanimoto", -7.0, 5.0],
            ["T2", 1, 0.4, "daina_max_tanimoto", -6.0, 4.5],
        ]
    )
    with pytest.raises(SystemExit, match="daina_rank가 중복"):
        band_rerank(duplicate_rank, keep=5, band=10)
    gap_rank = _band_frame(
        [
            ["T1", 1, 0.5, "daina_max_tanimoto", -7.0, 5.0],
            ["T2", 3, 0.4, "daina_max_tanimoto", -6.0, 4.5],
        ]
    )
    with pytest.raises(SystemExit, match="daina_rank가 1..n 연속이 아닙니다"):
        band_rerank(gap_rank, keep=5, band=10)


def test_band_rerank_accepts_a_valid_contiguous_control() -> None:
    from stage3_band_rerank import band_rerank

    valid = _band_frame(
        [
            [f"T{rank}", rank, 0.5 - rank / 100, "daina_max_tanimoto", -7.0 + rank / 10, 5.0 - rank / 10]
            for rank in range(1, 13)
        ]
    )
    ordered = band_rerank(valid, keep=5, band=10)

    assert ordered["target_id"].tolist() == [f"T{rank}" for rank in range(1, 13)]
    assert ordered["daina_rank"].tolist() == list(range(1, 13))
    assert ordered["final_rank"].tolist() == list(range(1, 13))
    assert int(ordered["rerank_band"].sum()) == 5
