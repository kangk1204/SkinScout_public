"""후보 순위 점수의 계약.

이 점수는 확률이 아니라 줄 세우기다. docs/RESEARCHER_GUIDE.md §133 이
"신뢰도 등급을 합성하지 않는다"고 못 박은 것과 어긋나지 않도록, 여기서
검증하는 것은 **순서가 뜻대로 나오는가**이지 값이 보정됐는가가 아니다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from candidate_score import (  # noqa: E402
    DEFAULT_WEIGHTS,
    axis_score,
    score_candidates,
    score_explanation,
)


def _frame(**columns) -> pd.DataFrame:
    n = len(next(iter(columns.values())))
    base = {"inchikey": [f"KEY{i:03d}-UHFFFAOYSA-N" for i in range(n)]}
    base.update(columns)
    return pd.DataFrame(base)


def test_a_structurally_closer_candidate_ranks_higher() -> None:
    frame = _frame(core_coverage=[0.2, 0.9], similarity=[0.1, 0.8],
                   pharm_similarity=[0.2, 0.7])
    out = score_candidates(frame)
    assert out.loc[1, "score_total"] > out.loc[0, "score_total"]
    assert out.loc[1, "score_rank"] == 1


def test_a_riskier_candidate_ranks_lower_when_structure_is_equal() -> None:
    """안전 항목은 뒤집어 넣는다. 위험이 높으면 순위가 내려가야 한다."""
    frame = _frame(core_coverage=[0.8, 0.8], similarity=[0.5, 0.5],
                   Skin_Reaction=[0.9, 0.1], AMES=[0.8, 0.1])
    out = score_candidates(frame)
    assert out.loc[1, "score_safety"] > out.loc[0, "score_safety"]
    assert out.loc[1, "score_total"] > out.loc[0, "score_total"]


def test_a_missing_admet_value_neither_helps_nor_hurts() -> None:
    """없는 값을 0이나 1로 두면 '안전하다'/'위험하다'로 읽힌다. 중앙에 둔다."""
    frame = _frame(core_coverage=[0.5, 0.5], Skin_Reaction=[np.nan, np.nan])
    out = score_candidates(frame)
    assert out["score_safety"].tolist() == [0.5, 0.5]


def test_an_absent_column_does_not_crash_or_dominate() -> None:
    """ADMET 캐시가 없는 배포에서도 구조 점수만으로 돌아야 한다."""
    frame = _frame(core_coverage=[0.1, 0.9])
    out = score_candidates(frame)
    assert out["score_safety"].tolist() == [0.5, 0.5]
    assert out.loc[1, "score_rank"] == 1


def test_ranks_are_deterministic_for_tied_scores() -> None:
    """동점이 실행마다 다른 순서로 나오면 화면이 흔들린다."""
    frame = _frame(core_coverage=[0.5, 0.5, 0.5])
    first = score_candidates(frame)["score_rank"].tolist()
    second = score_candidates(frame.iloc[::-1].reset_index(drop=True))
    assert first == sorted(first)
    assert sorted(second["score_rank"].tolist()) == [1, 2, 3]


def test_weights_can_be_overridden_and_are_normalised() -> None:
    frame = _frame(core_coverage=[0.1, 0.9], Skin_Reaction=[0.1, 0.9])
    structure_heavy = score_candidates(frame, {"structure": 1.0, "evidence": 0.0,
                                               "safety": 0.0})
    safety_heavy = score_candidates(frame, {"structure": 0.0, "evidence": 0.0,
                                            "safety": 1.0})
    # 구조만 보면 1번이 위, 안전만 보면 0번이 위여야 한다.
    assert structure_heavy.loc[1, "score_rank"] == 1
    assert safety_heavy.loc[0, "score_rank"] == 1


def test_zero_total_weight_is_refused() -> None:
    frame = _frame(core_coverage=[0.5])
    with pytest.raises(ValueError, match="가중치"):
        score_candidates(frame, {"structure": 0.0, "evidence": 0.0, "safety": 0.0})


def test_an_empty_frame_still_returns_the_score_columns() -> None:
    out = score_candidates(pd.DataFrame())
    for column in ("score_structure", "score_evidence", "score_safety",
                   "score_total", "score_rank"):
        assert column in out.columns


def test_the_explanation_says_it_is_not_a_probability() -> None:
    """화면에 뜨는 문장이 이 값을 확률로 읽히게 두면 안 된다."""
    text = score_explanation()
    assert "확률이 아니라" in text
    assert "82%" in text, "구체적인 오독 예를 들어야 읽는 쪽이 알아차립니다"
    assert "보정한 값이 아닙니다" in text
    assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)


def test_axis_score_flips_a_lower_is_better_term() -> None:
    frame = _frame(AMES=[0.1, 0.9])
    flipped = axis_score(frame, {"AMES": False})
    assert flipped[0] > flipped[1]


def test_an_imputed_axis_is_marked_so_it_is_not_read_as_measured() -> None:
    """0.5로 채운 것과 0.5로 잰 것은 화면에서 같아 보이면 안 된다.

    -1 같은 값으로 채우면 "없는 것"이 "가장 나쁜 것"이 되어 정렬에서 맨 아래로
    밀린다. 그건 판정이지 측정이 아니다. 중앙값으로 채우되, 채웠다는 사실을
    행마다 남긴다.
    """
    frame = _frame(core_coverage=[0.5, 0.5],
                   Skin_Reaction=[0.3, np.nan], AMES=[0.2, np.nan])
    out = score_candidates(frame)
    assert out.loc[0, "score_safety_basis"] == "measured"
    assert out.loc[1, "score_safety_basis"] == "imputed"
    assert out.loc[1, "score_safety"] == 0.5


def test_a_partially_present_axis_still_counts_as_measured() -> None:
    """항목 하나만 있어도 잰 것이다. 전부 없을 때만 채운 것이다."""
    frame = _frame(core_coverage=[0.5], Skin_Reaction=[0.4], AMES=[np.nan])
    out = score_candidates(frame)
    assert out.loc[0, "score_safety_basis"] == "measured"


def test_the_explanation_tells_the_reader_what_imputed_means() -> None:
    text = score_explanation()
    assert "추정" in text
    assert "낮게 잰 것이 아닙니다" in text
