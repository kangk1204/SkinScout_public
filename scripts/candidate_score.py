#!/usr/bin/env python3
"""candidate_score.py — 대체소재 후보를 한 줄로 세우는 순위 점수.

**이것은 신뢰도가 아니다.** `docs/RESEARCHER_GUIDE.md` §133 이 못 박고 있듯, 이
도구의 정답표는 실질 표본이 화합물 7~8개 수준이라 확률로 보정할 수 없다. 그래서
여기서 만드는 값은 "이 후보가 대체재일 확률"이 아니라 **여러 기준을 하나의
줄 세우기로 합친 것**이다. 0.82 가 0.41 보다 위라는 뜻이지, 82% 라는 뜻이 아니다.

합치는 방법은 이미 이 저장소가 쓰는 것과 같은 계열이다 - 각 기준을 백분위 순위로
바꾼 뒤 가중 평균한다(`alternative_ingredients.merged_rank_score`). 원시 단위가
서로 다른 값(Tanimoto 0~1, 분자량 100~800, 위험도 0~1)을 그대로 더하면 단위가 큰
쪽이 순위를 지배하므로, 순위로 바꾼 뒤에 합친다.

구성 요소는 셋이고, 각 후보 행에 그대로 남는다. 어느 것이 순위를 끌었는지
보이지 않으면 합친 값은 읽을 수 없다.

  구조   구조가 얼마나 남았는가          core_coverage · similarity · pharm_similarity
  근거   실제로 측정된 활성이 있는가      measured_evidence · measured_target_count
  안전   피부 적용에서 걸릴 것이 있는가   Skin_Reaction · AMES · hERG · DILI · 발암

안전 항목은 **뒤집어서** 넣는다(위험이 낮을수록 순위가 높다). ADMET 값이 없는
후보는 그 축에서 중앙(0.5)으로 두어, 없는 것이 유리해지지도 불리해지지도 않게
한다 - 없는 것을 0으로 두면 "안전하다"로, 1로 두면 "위험하다"로 읽힌다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 순위에 쓰는 기준과 방향. True 면 클수록 좋고, False 면 작을수록 좋다.
STRUCTURE_TERMS: dict[str, bool] = {
    "core_coverage": True,
    "similarity": True,
    "pharm_similarity": True,
}
EVIDENCE_TERMS: dict[str, bool] = {
    "measured_target_count": True,
    "measured_best_pactivity": True,
}
SAFETY_TERMS: dict[str, bool] = {
    "Skin_Reaction": False,
    "AMES": False,
    "hERG": False,
    "DILI": False,
    "Carcinogens_Lagunin": False,
}

# 축 가중치. 구조를 가장 무겁게 둔다 - 이 도구가 실제로 재는 것이 구조이고,
# 근거와 안전은 걸러 내는 역할이다. 이 값은 측정으로 보정한 것이 아니라
# 고른 것이므로, 바꿀 수 있게 인자로 노출한다.
DEFAULT_WEIGHTS = {"structure": 0.55, "evidence": 0.25, "safety": 0.20}


def _percentile(values: pd.Series) -> np.ndarray:
    """동점을 평균 순위로 처리한 0~1 백분위. 값이 없으면 0.5(중앙)."""
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() == 0:
        return np.full(len(numeric), 0.5, dtype=float)
    ranked = numeric.rank(method="average", pct=True)
    return ranked.fillna(0.5).to_numpy(dtype=float)


def axis_score(frame: pd.DataFrame, terms: dict[str, bool]) -> np.ndarray:
    """한 축의 점수. 쓸 수 있는 항목만 평균한다."""
    parts: list[np.ndarray] = []
    for column, higher_is_better in terms.items():
        if column not in frame.columns:
            continue
        ranked = _percentile(frame[column])
        parts.append(ranked if higher_is_better else 1.0 - ranked)
    if not parts:
        return np.full(len(frame), 0.5, dtype=float)
    return np.mean(parts, axis=0)


def axis_basis(frame: pd.DataFrame, terms: dict[str, bool]) -> np.ndarray:
    """행마다 그 축이 실제 값으로 계산됐는지("measured") 채워졌는지("imputed").

    없는 값을 중앙(0.5)으로 두는 것은 옳지만, **어느 행이 채워진 값인지 보이지
    않으면** 그 0.5 가 잰 값처럼 읽힌다. 채워 넣은 사실 자체가 정보다.
    """
    present = np.zeros(len(frame), dtype=bool)
    for column in terms:
        if column in frame.columns:
            present |= pd.to_numeric(frame[column], errors="coerce").notna().to_numpy()
    return np.where(present, "measured", "imputed")


def score_candidates(frame: pd.DataFrame,
                     weights: dict[str, float] | None = None) -> pd.DataFrame:
    """후보 표에 축별 점수와 합친 순위 점수를 붙인다.

    입력 프레임은 바꾸지 않고 새 프레임을 돌려준다. 붙는 열:
      score_structure · score_evidence · score_safety · score_total · score_rank
      score_structure_basis · score_safety_basis · score_evidence_basis

    `*_basis` 가 "imputed" 인 행은 그 축에 쓸 값이 하나도 없어 중앙(0.5)으로
    채운 것이다. 잰 값이 아니므로 화면에서 구분해 보여야 한다.
    """
    if frame.empty:
        out = frame.copy()
        for column in ("score_structure", "score_evidence", "score_safety",
                       "score_total", "score_rank"):
            out[column] = pd.Series(dtype=float)
        for column in ("score_structure_basis", "score_evidence_basis",
                       "score_safety_basis"):
            out[column] = pd.Series(dtype=object)
        return out

    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update({k: float(v) for k, v in weights.items() if k in w})
    total_weight = sum(w.values())
    if total_weight <= 0:
        raise ValueError(f"가중치 합이 0 이하입니다: {w}")

    out = frame.copy()
    out["score_structure"] = np.round(axis_score(frame, STRUCTURE_TERMS), 4)
    out["score_evidence"] = np.round(axis_score(frame, EVIDENCE_TERMS), 4)
    out["score_safety"] = np.round(axis_score(frame, SAFETY_TERMS), 4)
    # 구조 축도 비어 있을 수 있다. 예산 소진·MCS 실패·입체 불일치로 판정하지
    # 못한 후보는 `core_coverage`가 None 이고, 그러면 이 축이 중앙값으로 채워진다.
    # 다른 두 축과 똑같이 그 사실을 행에 남긴다 - 남기지 않으면 0.50 이 잰
    # 값처럼 읽힌다.
    out["score_structure_basis"] = axis_basis(frame, STRUCTURE_TERMS)
    out["score_evidence_basis"] = axis_basis(frame, EVIDENCE_TERMS)
    out["score_safety_basis"] = axis_basis(frame, SAFETY_TERMS)
    combined = (
        out["score_structure"] * w["structure"]
        + out["score_evidence"] * w["evidence"]
        + out["score_safety"] * w["safety"]
    ) / total_weight
    out["score_total"] = np.round(combined, 4)
    # 동점은 InChIKey 로 갈라 순위가 실행마다 달라지지 않게 한다.
    tiebreak = out["inchikey"].astype(str) if "inchikey" in out.columns else out.index.astype(str)
    order = pd.DataFrame({"score": out["score_total"], "key": tiebreak})
    out["score_rank"] = (
        order.sort_values(["score", "key"], ascending=[False, True])
        .assign(rank=range(1, len(order) + 1))
        .sort_index()["rank"]
    )
    return out


def score_explanation() -> str:
    """화면에 그대로 띄울 한 문단. 무엇이 아닌지부터 말한다."""
    return (
        "종합 점수는 **확률이 아니라 줄 세우기**입니다. 구조(55%)·근거(25%)·"
        "안전(20%) 세 축을 각각 백분위 순위로 바꾼 뒤 가중 평균했습니다. "
        "0.82가 0.41보다 위라는 뜻이지 82%라는 뜻이 아닙니다. 축 점수가 함께 "
        "나오므로 무엇이 순위를 끌었는지 보고 판단하세요. 가중치는 고른 값이지 "
        "측정으로 보정한 값이 아닙니다. 축 옆의 '추정'은 그 축에 쓸 값이 없어 "
        "중앙값으로 채웠다는 뜻이며, 낮게 잰 것이 아닙니다."
    )
