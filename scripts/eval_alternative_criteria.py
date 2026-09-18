#!/usr/bin/env python3
"""대체소재 순위 기준 세 가지를, 정답표를 두고 맞대 본다.

`alternative_ingredients`는 후보를 세 가지로 줄 세울 수 있다. 구조 유지율(MCS),
파마코포어 특징 유사도(Gobbi 2D), 그리고 분자 전체 유사도(ECFP4 Tanimoto)다. 셋은
서로 다른 것을 재고 실제로 크게 어긋난다 - 등재 원료 507종에서 구조와 파마코포어의
순위 상관은 질의에 따라 0.09까지 내려간다. 그러면 **어느 것을 위에 두어야 하는가**가
남는데, 그 질문은 정답표 없이는 답이 없다.

이 파일이 하는 일은 그 정답표를 두고 세 기준을 재는 것이고, 하지 않는 일은 승자를
고르는 것이다. 정답표가 무엇을 대신하는지에 따라 답이 달라지기 때문이다.

## 정답표

**A. 신고 배합목적** (`--truth functions`)
CosIng에 등재할 때 신고된 배합목적이 같으면 같은 자리에 쓸 수 있다고 본다. 507종 중
355종에 신고가 있고, 38개 목적 중 13개가 5종 이상이다.

  주의: 이것은 "용도가 같다"이지 "대체 가능하다"가 아니다. 그리고 **용도 분류가 곧
  화학 분류인 경우가 많다** - HAIR DYEING 93종은 대부분 방향족 아민이라, 구조 기준이
  잘 맞히는 것이 부분적으로는 동어반복이다. 이 편향은 세 기준에 똑같이 걸리므로 기준
  사이의 *비교*는 여전히 공정하지만, 절대 AUC를 능력으로 읽으면 안 된다. 그래서
  목적별 결과를 따로 내고, 대형 분류(모집단의 10% 이상을 덮는 것 — 지금 라이브러리
  에서는 PERFUMING 23.9%, SKIN CONDITIONING 22.1%)를 뺀 결과를 함께 낸다. 이 목록은
  이름으로 박아 두지 않고 모집단에서 뽑는다.

**B. UV 필터 참조표** (`--truth uv_filters`)
`data/validation/uv_filter_reference.csv`. 출처가 적힌 10종의 좁고 깨끗한 분류.

**C. 큐레이션된 대체 쌍** (`--truth curated`)
`data/validation/substitute_pairs.csv`가 있을 때만. 행마다 출처와 인용문이 붙어 있고,
출처를 실제로 읽어 확인한 행만 평가에 쓴다.

## 지표

질의마다 ROC AUC를 낸다 - 그 질의의 양성(같은 분류의 다른 원료)을 음성보다 위에
올리는 비율이다. 우연은 0.5다. 양성 비율이 질의마다 크게 다르므로 적중 수가 아니라
AUC를 쓴다. 기준 사이의 비교는 질의를 짝지어 Wilcoxon 부호순위 검정으로 한다 -
같은 질의에 대한 세 값이므로 독립 표본 검정은 틀린다.

    python scripts/eval_alternative_criteria.py --truth functions
    python scripts/eval_alternative_criteria.py --truth functions --exclude-large-classes
    python scripts/eval_alternative_criteria.py --truth uv_filters --out-csv results.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from rdkit import Chem, RDLogger  # noqa: E402

from alternative_ingredients import (  # noqa: E402
    IngredientLibrary,
    McsBudget,
    _packed,
    _split_list,
    _tanimoto,
    core_retention,
    DEFAULT_3D_CONFORMERS,
    judged_coverage,
    load_ingredient_library,
    merged_rank_score,
    pharmacophore_match,
    pharmacophore_profile,
    prepare_query,
)
from alternative_ingredients import prepare_query_3d, three_d_match  # noqa: E402
from build_activity_retrieval_index import MORGAN_GENERATOR  # noqa: E402

RDLogger.DisableLog("rdApp.*")

UV_FILTER_REFERENCE = ROOT / "data" / "validation" / "uv_filter_reference.csv"
CURATED_PAIRS = ROOT / "data" / "validation" / "substitute_pairs.csv"

# 한 분류가 모집단의 이만큼을 덮으면 "대형"으로 본다. 함께 두면 어떤 결과든
# 그 분류의 결과가 되기 때문이다.
#
# 예전에는 이름을 박아 두었다 - ("PERFUMING", "HAIR DYEING", "FRAGRANCE"). 507종
# 중 272종이 거기 속하던 시절에는 맞았지만, 라이브러리가 7,484종이 되면서 조용히
# 틀렸다: 두 번째로 큰 SKIN CONDITIONING(1,651종·22.1%)이 목록에 없어 대형으로
# 치지 않았고, HAIR DYEING(333종·4.4%)은 이제 대형이 아니다. 그 상태로
# `--exclude-large-classes`를 돌리면 "대형 분류를 빼도 결론이 유지되는가"에
# **빼지 않은 채로** 답한다. 피부 도구에서 SKIN CONDITIONING 을 남겨 두는 것은
# 그중에서도 나쁜 누락이다 - caveat 이 경고하는 AUC 부풀림을 지금은 그 분류가
# 끌고 있기 때문이다.
#
# 그래서 이름이 아니라 모집단에서 뽑는다. 라이브러리가 다시 바뀌어도 따라간다.
LARGE_CLASS_SHARE = 0.10


def large_classes(library: IngredientLibrary) -> tuple[str, ...]:
    """모집단의 `LARGE_CLASS_SHARE` 이상을 덮는 배합목적. 큰 것부터."""
    counts: dict[str, int] = {}
    for value in library.frame["functions"]:
        for function in _split_list(value):
            counts[function] = counts.get(function, 0) + 1
    total = max(1, len(library.frame))
    big = [(n, f) for f, n in counts.items() if n / total >= LARGE_CLASS_SHARE]
    return tuple(f for _, f in sorted(big, key=lambda pair: (-pair[0], pair[1])))


# 질의당 양성이 하나뿐이면 AUC는 그 하나의 순위에 불과하다. 셀 수는 있지만, 몇 개인지
# 따로 세어 보고할 만큼은 다른 이야기다.
MIN_POSITIVES = 1

# 3D는 라이브러리 전수에 쌍당 0.2-1.0초가 든다. 질의 하나에 507종이면 2-9분이고
# 351개 질의는 며칠이다. 그래서 기본으로 끄고, 작은 정답표에서만 켠다.
THREE_D_KEY = "three_d"


@dataclass(frozen=True)
class Criterion:
    key: str
    label_ko: str
    describe: str


CRITERIA = (
    Criterion("structural", "구조 유지율", "최대 공통 부분구조 / 질의 중원자 (MCS coverage)"),
    Criterion("pharmacophore", "파마코포어", "Gobbi 2D 파마코포어 지문 Tanimoto"),
    Criterion("tanimoto", "분자 유사도", "ECFP4 Tanimoto"),
    # 구조를 전혀 모르는 기준선. 이것이 없으면 AUC 0.65를 보고 "잘 맞힌다"고 읽게
    # 되는데, 신고 목적이 같은 원료들은 크기부터 비슷해서 분자량 차이만으로도 0.61이
    # 나온다. 세 기준의 값은 우연(0.5)이 아니라 **이 값**과 견주어야 한다.
    Criterion("size_baseline", "크기 기준선", "분자량 차이 (구조 정보 없음, 귀무 기준)"),
    # 극성 표면적도 구조를 "이해"하지 않는 기준선이다. 분자량 하나만 두면 그 하나를
    # 이겼다는 사실이 실제보다 크게 들린다.
    Criterion("polarity_baseline", "극성 기준선", "TPSA 차이 (구조 정보 없음, 귀무 기준)"),
    # 세 기준을 순위평균으로 합친 것. 처음에는 넣지 않았는데, 넣지 않은 채로
    # "합치지 말아야 한다"고 쓴 것이 틀렸다 - 이 정답표에서 합친 쪽이 셋 각각을
    # 모두 이긴다(0.685 대 0.657/0.645/0.666). 재 보지 않고 주장한 것이 잘못이다.
    Criterion("merged_rank_average", "순위평균(합침)", "세 기준의 백분위 순위 평균"),
    # 도구가 정상인지 보는 눈금. 0.5에서 벗어나면 평가 자체가 틀린 것이다.
    Criterion("random_control", "난수 대조", "질의와 무관한 난수 (0.5가 나와야 정상)"),
)

# --with-3d 로 켤 때만 목록에 들어간다. 비용 때문에 기본에서 뺀 것이지, 재 볼 가치가
# 없어서가 아니다.
THREE_D_CRITERION = Criterion(
    THREE_D_KEY, "3D 정렬", "컨포머 정렬 후 특징족 재현율 (ETKDGv3 + MMFF94s)"
)

# "방법"과 그것을 재는 자를 가른다.
METHOD_KEYS = ("structural", "pharmacophore", "tanimoto", "merged_rank_average")
SINGLE_METHOD_KEYS = ("structural", "pharmacophore", "tanimoto")
BASELINE_KEYS = ("size_baseline", "polarity_baseline")
BASELINE_KEY = "size_baseline"          # 대표 귀무 기준(하위 호환)
CONTROL_KEY = "random_control"


def roc_auc(scores: np.ndarray, positive: np.ndarray) -> float | None:
    """동점을 절반으로 세는 표준 AUC. 양성이나 음성이 없으면 정의되지 않는다.

    `scipy` 없이 순위합으로 낸다 - 이 저장소의 실행 환경에 scipy가 있다는 보장이
    없고, 동점 처리를 눈에 보이게 두는 편이 낫다.
    """
    scores = np.asarray(scores, dtype=float)
    positive = np.asarray(positive)
    if scores.ndim != 1 or positive.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("AUC requires equal-length finite one-dimensional inputs")
    if not np.isin(positive, [0, 1]).all():
        raise ValueError("AUC labels must be binary")
    positive = positive.astype(bool)
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    # 동점은 평균 순위로. 이것을 빼먹으면 같은 점수 덩어리가 입력 순서대로 줄을 서서
    # AUC가 우연히 높아지거나 낮아진다.
    sorted_scores = scores[order]
    start = 0
    for index in range(1, len(sorted_scores) + 1):
        if index == len(sorted_scores) or sorted_scores[index] != sorted_scores[start]:
            if index - start > 1:
                block = order[start:index]
                ranks[block] = ranks[block].mean()
            start = index
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def wilcoxon_signed_rank(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """짝지은 두 기준의 차이가 0인지. 정규근사, 동점은 평균 순위.

    같은 질의에 대한 두 값이므로 독립 표본 검정은 쓸 수 없다. 표본이 20쌍 아래면
    정규근사가 거칠어지므로 그 사실을 함께 돌려준다.
    """
    difference = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if not np.isfinite(difference).all():
        raise ValueError("paired differences must be finite")
    difference = difference[difference != 0]
    n = len(difference)
    if n == 0:
        return {"n": 0, "statistic": None, "p_value": None, "approximation_reliable": False}
    magnitude = np.abs(difference)
    order = np.argsort(magnitude, kind="stable")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    sorted_magnitude = magnitude[order]
    start = 0
    for index in range(1, n + 1):
        if index == n or sorted_magnitude[index] != sorted_magnitude[start]:
            if index - start > 1:
                block = order[start:index]
                ranks[block] = ranks[block].mean()
            start = index
    w_plus = float(ranks[difference > 0].sum())
    w_minus = float(ranks[difference < 0].sum())
    statistic = min(w_plus, w_minus)
    mean = n * (n + 1) / 4.0
    # Conditional sign variance of the actual midranks, including ties.
    variance = float(np.square(ranks).sum()) / 4.0
    if variance <= 0:
        return {"n": n, "statistic": statistic, "p_value": None, "approximation_reliable": False}
    z = max(0.0, abs(statistic - mean) - 0.5) / np.sqrt(variance)
    from math import erfc, sqrt

    # `1 + erf(z/√2)`는 z가 -8 아래로 가면 배정밀도에서 정확히 0으로 상쇄된다.
    # PERFUMING 층(n=179)에서 실제로 터져 p=0.0이 찍혔다 - 참값은 2.9e-22다.
    # erfc(|z|/√2)는 같은 값을 꼬리까지 정확히 낸다.
    p_value = erfc(abs(z) / sqrt(2.0))
    return {
        "n": n,
        "statistic": statistic,
        "w_plus": w_plus,
        "w_minus": w_minus,
        "p_value": float(min(1.0, max(0.0, p_value))),
        "approximation_reliable": n >= 20,
    }


# --- 정답표 ------------------------------------------------------------------


@dataclass
class GroundTruth:
    """어떤 원료끼리 "같은 자리에 쓸 수 있다"고 볼 것인가."""

    name: str
    describe: str
    # 라이브러리 행 위치 -> 그 원료와 같은 분류인 다른 행 위치들
    positives: dict[int, set[int]]
    # 질의를 나눠 볼 이름표 (배합목적 이름 등). 뭉친 숫자 하나로 보고하지 않기 위해.
    strata: dict[int, tuple[str, ...]]
    caveat: str
    # 이 정답표를 만들 때 무엇을 "대형 분류"로 봤는지. 모집단에서 뽑으므로
    # 라이브러리마다 달라지고, 결과를 읽을 때 그 목록을 알아야 한다.
    large_classes: tuple[str, ...] = ()


def functions_ground_truth(
    library: IngredientLibrary,
    exclude_large: bool = False,
    subset_queries_only: bool = False,
) -> GroundTruth:
    """`exclude_large`와 `subset_queries_only`는 다른 것을 한다. 섞으면 안 된다.

    `exclude_large`는 **모든** 원료에서 대형 분류 라벨을 떼므로 양성 집합 자체가
    바뀐다. 지금 라이브러리(7,484종)에서 질의당 중앙 양성이 1,650에서 438로,
    기저 비율이 0.220에서 0.059로 떨어진다(507종 시절에는 178→13, 0.35→0.03이었다). 즉 다른 검색 과제이며, 그 결과를 앞 실행의 재검정으로 읽으면
    안 된다 - 처음에 그렇게 읽고 "효과가 사라졌다"고 썼는데, 실제로는 효과 크기가
    그대로이고 n만 351에서 103으로 준 것이었다(같은 351개에서 103개를 무작위로
    뽑으면 18.5%에서만 p<0.05가 나온다).

    `subset_queries_only`는 정답표를 그대로 두고 **질의만** 고른다. 이쪽이
    "대형 분류 질의를 빼면 결론이 유지되는가"에 답하는 검정이다.
    """
    big = set(large_classes(library))
    excluded = big if exclude_large else set()
    declared = [set(_split_list(value)) - excluded for value in library.frame["functions"]]
    members: dict[str, set[int]] = {}
    for index, functions in enumerate(declared):
        for function in functions:
            members.setdefault(function, set()).add(index)

    positives: dict[int, set[int]] = {}
    strata: dict[int, tuple[str, ...]] = {}
    for index, functions in enumerate(declared):
        if not functions:
            continue
        if subset_queries_only and (functions & big):
            continue
        same = set().union(*(members[f] for f in functions)) - {index}
        if len(same) >= MIN_POSITIVES:
            positives[index] = same
            strata[index] = tuple(sorted(functions))
    suffix = "_small_classes" if exclude_large else ("_subset_queries" if subset_queries_only else "")
    return GroundTruth(
        name="functions" + suffix,
        describe=(
            "CosIng 신고 배합목적이 하나라도 겹치면 같은 분류"
            # 어느 분류를 뺐는지 이름으로 적는다. "대형 분류"라고만 쓰면 모집단이
            # 바뀌어 목록이 달라져도 설명이 그대로라, 읽는 쪽이 알 수 없다.
            + (f"(모든 원료에서 {'·'.join(sorted(big))} 라벨 제거 — 정답표가 달라짐)"
               if exclude_large else "")
            + (f"({'·'.join(sorted(big))}에 속한 질의만 제외, 정답표는 그대로)"
               if subset_queries_only else "")
        ),
        positives=positives,
        strata=strata,
        caveat=(
            "신고된 용도가 같다는 것이지 대체 가능하다는 뜻이 아니다. 용도 분류가 곧 화학 "
            "분류인 경우가 많아 절대 AUC는 부풀려져 있다 - 기준 사이의 비교에만 쓸 것."
        ),
        large_classes=tuple(sorted(big)),
    )


def uv_filter_ground_truth(library: IngredientLibrary) -> GroundTruth:
    if not UV_FILTER_REFERENCE.is_file():
        raise SystemExit(f"UV 필터 참조표가 없습니다: {UV_FILTER_REFERENCE}")
    reference = pd.read_csv(UV_FILTER_REFERENCE)
    wanted = {str(v).strip()[:14] for v in reference["inchikey_connectivity"] if str(v).strip()}
    library_skeletons = library.frame["inchikey"].astype(str).str[:14]
    rows = {index for index, skeleton in enumerate(library_skeletons) if skeleton in wanted}
    positives = {index: rows - {index} for index in rows if len(rows - {index}) >= MIN_POSITIVES}
    return GroundTruth(
        name="uv_filters",
        describe=f"data/validation/uv_filter_reference.csv 의 {len(reference)}종 중 라이브러리에 있는 {len(rows)}종",
        positives=positives,
        strata={index: ("UV_FILTER",) for index in positives},
        caveat="좁고 깨끗한 분류지만 질의 수가 매우 적다. 검정력이 낮다.",
    )


def curated_ground_truth(library: IngredientLibrary) -> GroundTruth:
    if not CURATED_PAIRS.is_file():
        raise SystemExit(
            f"큐레이션 대체 쌍 표가 없습니다: {CURATED_PAIRS}. "
            "scripts/build_substitute_pairs.py 로 먼저 만드세요."
        )
    pairs = pd.read_csv(CURATED_PAIRS)
    # 출처를 실제로 읽어 확인한 행만 쓴다. 확인하지 못한 행은 표에 남겨 두되 평가에는
    # 넣지 않는다 - 확인되지 않은 정답으로 방법을 판정하면 판정 자체가 근거를 잃는다.
    usable = pairs[pairs["verified"].astype(str).str.lower().isin({"true", "1", "yes"})]
    skeleton_to_row: dict[str, int] = {}
    for index, key in enumerate(library.frame["inchikey"].astype(str)):
        skeleton_to_row.setdefault(key[:14], index)

    positives: dict[int, set[int]] = {}
    strata: dict[int, list[str]] = {}
    dropped = 0
    for row in usable.itertuples(index=False):
        a = skeleton_to_row.get(str(row.a_inchikey)[:14])
        b = skeleton_to_row.get(str(row.b_inchikey)[:14])
        if a is None or b is None or a == b:
            dropped += 1
            continue
        # 대체 관계는 대칭으로 본다. A 자리에 B를 쓸 수 있으면 그 반대도 후보다.
        positives.setdefault(a, set()).add(b)
        positives.setdefault(b, set()).add(a)
        for side in (a, b):
            strata.setdefault(side, []).append(str(row.relationship))
    return GroundTruth(
        name="curated",
        describe=(
            f"{CURATED_PAIRS.name}: 전체 {len(pairs)}쌍 중 출처 확인된 {len(usable)}쌍, "
            f"라이브러리에서 양쪽을 찾지 못해 버린 것 {dropped}쌍"
        ),
        positives={k: v for k, v in positives.items() if len(v) >= MIN_POSITIVES},
        strata={k: tuple(sorted(set(v))) for k, v in strata.items()},
        caveat="출처가 붙은 쌍만 들어 있지만 수가 적고, 큐레이션 범위가 곧 편향이다.",
    )


# --- 점수 -------------------------------------------------------------------


def score_all_criteria(
    library: IngredientLibrary, query_index: int, *, with_three_d: bool = False
) -> dict[str, np.ndarray]:
    """한 질의에 대해 라이브러리 전체를 세 기준으로 점수 낸다.

    질의도 라이브러리의 한 원료다. 자기 자신은 호출자가 뺀다 - 여기서 빼면 배열
    길이가 달라져 호출자의 색인이 어긋난다.
    """
    query = library.molecules[query_index]
    prepared = prepare_query(query)
    budget = McsBudget()

    packed = _packed(MORGAN_GENERATOR.GetFingerprint(query))
    tanimoto = _tanimoto(library.fingerprints, library.popcounts, packed)

    from rdkit.Chem import Descriptors

    weights = np.array([Descriptors.MolWt(m) for m in library.molecules], dtype=float)
    polarity = np.array([Descriptors.TPSA(m) for m in library.molecules], dtype=float)
    # 질의와 분자량이 가까울수록 위. "닮았다"의 가장 싸구려 대용이며, 구조를
    # 하나도 보지 않는다.
    size_baseline = -np.abs(weights - weights[query_index])
    # 씨앗을 질의 색인에 묶어 재현 가능하게 한다. 실행마다 달라지면 대조가 아니다.
    identities = library.frame["inchikey"].astype(str).tolist()
    control = np.array([
        int.from_bytes(hashlib.sha256(f"{identities[query_index]}:{key}".encode()).digest()[:8], "big") / 2**64
        for key in identities
    ])
    structural = np.full(len(library.molecules), np.nan, dtype=float)
    for i in sorted(range(len(library.molecules)), key=lambda i: identities[i]):
        retained = core_retention(query, library.molecules[i], budget, prepared)
        # 화면이 쓰는 것과 같은 규칙(`judged_coverage`)으로 거른다. 여기서 따로
        # 조건을 적으면 둘이 갈리고, 그때 재는 것은 화면이 내는 순위가 아니다.
        judged = judged_coverage(retained)
        if judged is not None:
            structural[i] = judged

    profiles = library.pharmacophore_profiles
    query_profile = profiles[query_index]
    # `usable`을 버리면 안 된다. pharmacophore_match 는 Gobbi 지문이 비어 있는
    # 쌍에 similarity 0.0 과 usable=False 를 함께 돌려주는데, 그 0.0 은
    # "특징이 하나도 안 겹친다"가 아니라 **잴 수 없다**는 뜻이다. 507종 중 39종의
    # 지문이 비어 있고, 그중 20종이 `functions` 정답표의 질의다. 그 질의는 점수
    # 507칸이 전부 0.0 이 되어 전부 동점이 되고, 동점 평균 AUC 가 정확히 0.500 -
    # 이 도구가 "우연"이라고 인쇄하는 바로 그 값 - 으로 나온다. 그 0.500 이
    # 평균에 섞이고 Wilcoxon 의 관측 하나로 들어가면, 재지 못한 것이 잰 것과
    # 구분되지 않는다. 3D 기준이 바로 아래에서 지키는 규칙과 같은 규칙이다.
    pharm_matches = [pharmacophore_match(query_profile, p) for p in profiles]
    # 보고용: 잴 수 없었던 자리는 NaN. 호출자가 그 자리를 빼고 AUC 를 낸다.
    pharmacophore = np.array(
        [(m.similarity if m.usable else np.nan) for m in pharm_matches], dtype=float
    )
    # 합침용: 원값 그대로. 세 기준의 백분위를 평균하려면 후보마다 값이 있어야 하고,
    # 한 신호가 없다고 그 후보를 순위에서 통째로 빼면 나머지 두 신호로는 멀쩡한
    # 후보가 사라진다. 질의 자신의 지문이 비어 있으면 이 열이 상수가 되고, 그때
    # 합친 순위는 자연히 나머지 두 기준의 평균으로 줄어든다.
    # 화면과 같은 규칙이다. `usable=False` 의 0.0 은 "특징이 하나도 안 겹친다"가
    # 아니라 지문을 만들지 못해 **잴 수 없다**는 뜻이고, 그것을 0 으로 합치면
    # 재지 못한 후보가 조용히 최하위표를 던진다. 실측으로 등재 원료의 12.8% 가
    # 여기 해당한다(특징이 너무 많은 큰 유연 지질·펩타이드).
    pharmacophore_for_merge = np.array(
        [m.similarity if m.usable else np.nan for m in pharm_matches], dtype=float
    )
    scores = {
        "structural": structural,
        "pharmacophore": pharmacophore,
        "tanimoto": tanimoto,
        "size_baseline": size_baseline,
        "polarity_baseline": -np.abs(polarity - polarity[query_index]),
        "random_control": control,
    }
    # 화면이 쓰는 것과 **같은 함수**로 합친다. 여기서 따로 구현하면 잰 순위와
    # 화면이 내는 순위가 달라지고, 그 차이는 화면을 봐서는 알 수 없다.
    #
    # NaN을 0으로 메우지 않는다. 예전에는 `np.nan_to_num(structural)`을 썼는데,
    # 그것이 화면과 같아지는 길이었기 때문이다 - 화면 쪽이 판정 못 한 자리를
    # 0으로 채우고 있었다. 이제 양쪽 다 NaN으로 두고 `merged_rank_score`가 잰
    # 기준들로만 평균한다. 0으로 메우면 판정 못 한 것이 "구조가 가장 다름"으로
    # 들어가, 재지 못한 쌍이 조용히 최하위표를 던진다.
    scores["merged_rank_average"] = merged_rank_score(
        dict(scores, pharmacophore=pharmacophore_for_merge), SINGLE_METHOD_KEYS
    )

    if with_three_d:
        # 판정되지 않은 쌍(컨포머 실패, 공통 구조 3원자 미만)은 0으로 두지 않는다.
        # 0은 "3D에서 전혀 안 맞았다"는 뜻이고, 실제로는 재지 못한 것이다. AUC에서
        # 이 둘을 같게 세면 재지 못한 쌍이 조용히 최하위표를 던진다. NaN으로 두고
        # 호출자가 그 질의를 통째로 버리게 한다.
        prepared = prepare_query_3d(query, conformers=DEFAULT_3D_CONFORMERS)
        values = np.full(len(library.molecules), np.nan, dtype=float)
        for position, candidate in enumerate(library.molecules):
            match = three_d_match(query, candidate, prepared, conformers=DEFAULT_3D_CONFORMERS)
            # recall_only 는 정렬이 실제로 돈 결과다. RMSD 만 정의되지 않았을 뿐
            # 회수율은 잰 값이므로, 버리면 가장 강한 부정 근거를 버리는 셈이다.
            if match.status in {"ok", "recall_only"} and match.feature_recall is not None:
                values[position] = match.feature_recall
        scores[THREE_D_KEY] = values
    return scores


# 워커가 fork 뒤에 읽는 자리. 라이브러리와 파마코포어 프로필을 fork **전에** 만들어
# 두면 리눅스의 copy-on-write로 자식들이 복사 없이 나눠 쓴다. 자식마다 다시 만들면
# 워커 하나당 4초와 수백 MB가 더 든다.
_WORKER: dict[str, Any] = {}


def _score_one_query(task: tuple[int, bool]) -> tuple[int, dict[str, Any]]:
    """질의 하나의 AUC 한 줄. 직렬·병렬이 같은 함수를 쓰므로 결과가 갈릴 수 없다."""
    query_index, with_three_d = task
    library: IngredientLibrary = _WORKER["library"]
    truth: GroundTruth = _WORKER["truth"]
    names: list[str] = _WORKER["names"]
    total = len(library.molecules)

    scores = score_all_criteria(library, query_index, with_three_d=with_three_d)
    keep = np.ones(total, dtype=bool)
    keep[query_index] = False           # 자기 자신은 후보가 아니다
    positive = np.zeros(total, dtype=bool)
    positive[list(truth.positives[query_index])] = True

    row: dict[str, Any] = {
        "query_index": query_index,
        "query": names[query_index],
        "strata": ";".join(truth.strata.get(query_index, ())),
        "positives": int(positive[keep].sum()),
        "pool": int(keep.sum()),
    }
    criteria = list(CRITERIA) + ([THREE_D_CRITERION] if with_three_d else [])
    for criterion in criteria:
        values = scores[criterion.key]
        usable = keep & np.isfinite(values)
        # 3D는 판정 못 한 후보가 생긴다. 그 후보를 최하위로 넣지 않고 뺀 뒤,
        # 몇 개를 뺐는지 함께 남긴다.
        row[criterion.key] = roc_auc(values[usable], positive[usable])
        if criterion.key == THREE_D_KEY:
            row["three_d_unjudged"] = int((keep & np.isnan(values)).sum())
        elif criterion.key == "pharmacophore":
            # Gobbi 지문이 빈 후보. 질의 자신의 지문이 비면 이 수가 후보 전체가
            # 되고 AUC 는 정의되지 않는다 - 예전에는 그 경우가 정확히 0.500 으로
            # 나와, 이 도구가 "우연"이라 인쇄하는 값과 구분되지 않았다.
            row["pharmacophore_unjudged"] = int((keep & np.isnan(values)).sum())
    # Comparisons must use the same candidate population, not merely the same
    # query IDs. Keep descriptive per-method AUCs above, and compute separate
    # shared pools for 2D and optional 3D so enabling 3D cannot change 2D results.
    common = keep.copy()
    for criterion in CRITERIA:
        common &= np.isfinite(scores[criterion.key])
    row["paired_2d_pool"] = int(common.sum())
    for criterion in CRITERIA:
        row[f"paired_2d__{criterion.key}"] = roc_auc(scores[criterion.key][common], positive[common])
    if with_three_d:
        common &= np.isfinite(scores[THREE_D_KEY])
        row["paired_3d_pool"] = int(common.sum())
        for criterion in criteria:
            row[f"paired_3d__{criterion.key}"] = roc_auc(scores[criterion.key][common], positive[common])
    return query_index, row


def evaluate(
    library: IngredientLibrary,
    truth: GroundTruth,
    *,
    with_three_d: bool = False,
    workers: int = 1,
) -> pd.DataFrame:
    """질의마다 세 기준의 AUC를 낸다.

    질의끼리는 완전히 독립이므로 `workers`를 올리면 그대로 나뉜다. 3D는 질의 하나에
    2분 넘게 걸려 20질의가 40분 이상이었다. 다만 **병렬 결과가 직렬과 한 자리도
    달라서는 안 된다** - 달라지면 측정을 바꾼 것이다. 두 경로가 같은
    `_score_one_query`를 부르고, 결과는 질의 색인으로 다시 정렬한다.
    """
    from alternative_ingredients import ensure_pharmacophore_profiles

    ensure_pharmacophore_profiles(library)
    names = library.frame["inci_name"].astype(str).tolist()
    _WORKER["library"] = library
    _WORKER["truth"] = truth
    _WORKER["names"] = names

    ordered = sorted(truth.positives)
    tasks = [(index, with_three_d) for index in ordered]
    started = time.monotonic()
    step = 1 if with_three_d else max(1, min(25, len(ordered) // 10 or 1))
    collected: dict[int, dict[str, Any]] = {}

    def note(done: int) -> None:
        if done % step and done != len(ordered):
            return
        elapsed = time.monotonic() - started
        remaining = (elapsed / done) * (len(ordered) - done)
        print(
            f"  ... {done}/{len(ordered)} 질의 · 경과 {elapsed/60:.1f}분 "
            f"· 남은 예상 {remaining/60:.1f}분",
            file=sys.stderr, flush=True,
        )

    if workers > 1 and len(tasks) > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        # fork여야 자식이 위에서 만든 라이브러리를 그대로 물려받는다. spawn이면
        # 자식이 빈 모듈로 시작해 `_WORKER`가 비어 있다.
        context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            for done, (query_index, row) in enumerate(
                pool.map(_score_one_query, tasks, chunksize=1), 1
            ):
                collected[query_index] = row
                note(done)
    else:
        for done, task in enumerate(tasks, 1):
            query_index, row = _score_one_query(task)
            collected[query_index] = row
            note(done)

    rows = [collected[index] for index in ordered]
    return pd.DataFrame(rows)


def _paired_population(frame: pd.DataFrame, *, three_d: bool = False) -> tuple[pd.DataFrame, bool]:
    prefix = "paired_3d__" if three_d else "paired_2d__"
    criteria = list(CRITERIA) + ([THREE_D_CRITERION] if three_d else [])
    columns = [prefix + c.key for c in criteria]
    if not all(c in frame for c in columns):
        # Old aggregate CSVs cannot establish which candidates each method saw.
        return frame, False
    result = frame.copy()
    for column in columns:
        result[column.removeprefix(prefix)] = frame[column]
    return result.dropna(subset=[c.removeprefix(prefix) for c in columns]), True


def _adjust_summary_tests(summary: dict[str, Any]) -> None:
    family = [(test, "p_value") for name in ("paired_tests", "vs_baseline") for test in summary[name].values()]
    for stratum in summary["within_run_split"].values():
        for test in stratum.values():
            if isinstance(test, dict):
                family.extend([(test, "inside_p"), (test, "outside_p")])
    observed = sorted([(float(test[key]), i) for i, (test, key) in enumerate(family) if test.get(key) is not None])
    for test, key in family:
        test[key + "_adjusted"] = None
    previous = 0.0
    for rank, (p_value, index) in enumerate(observed):
        previous = max(previous, min(1.0, p_value * (len(observed) - rank)))
        test, key = family[index]
        test[key + "_adjusted"] = previous
    summary["multiplicity"] = {"method": "Holm", "family_size": len(observed), "family": "all pairwise, baseline and within-run stratum tests"}


def summarise(frame: pd.DataFrame, truth: GroundTruth) -> dict[str, Any]:
    present = [c for c in list(CRITERIA) + [THREE_D_CRITERION] if c.key in frame.columns]
    # 2D 기준의 요약은 **2D 기준만** 완전한 행으로 낸다. 예전에는 3D 열까지
    # 넣어 dropna 를 걸어서, 3D AUC 하나가 정의되지 않은 질의가 멀쩡한
    # 구조·파마코포어·유사도 수치를 함께 데리고 나갔다. 그러면 `--with-3d` 를
    # 켜는 것만으로 2D 결론이 달라지고, 유일한 흔적인
    # `queries_dropped_undefined_auc` 는 그것이 3D 때문이라고 말하지 않는다.
    two_d = [c for c in CRITERIA if c.key in frame.columns]
    usable = frame.dropna(subset=[c.key for c in two_d])
    # 3D 는 자기가 잴 수 있었던 행에서, 자기 n 과 함께 따로 낸다.
    three_d_usable = (
        frame.dropna(subset=[c.key for c in present])
        if len(present) > len(two_d)
        else usable
    )
    summary: dict[str, Any] = {
        "ground_truth": truth.name,
        "describe": truth.describe,
        "caveat": truth.caveat,
        "queries_evaluated": int(len(usable)),
        "queries_evaluated_three_d": int(len(three_d_usable)),
        "queries_dropped_undefined_auc": int(len(frame) - len(usable)),
        "queries_dropped_undefined_three_d": int(len(usable) - len(three_d_usable)),
        "median_positives": int(usable["positives"].median()) if len(usable) else 0,
        "per_criterion": {},
        "per_criterion_note": "Descriptive AUCs may use different candidate pools; use paired_tests for comparisons.",
        "paired_tests": {},
    }
    for criterion in present:
        # 3D 는 자기가 잴 수 있었던 행에서만 낸다. 2D 기준의 행 집합에 섞으면
        # NaN 이 평균을 오염시키고, 반대로 2D 를 3D 의 행 집합으로 줄이면 3D 를
        # 켰다는 이유만으로 2D 숫자가 바뀐다.
        rows = three_d_usable if criterion.key == THREE_D_KEY else usable
        values = rows[criterion.key].to_numpy(dtype=float)
        summary["per_criterion"][criterion.key] = {
            "label_ko": criterion.label_ko,
            "describe": criterion.describe,
            # 이 수치가 몇 개의 질의에서 나온 것인지. 기준마다 다를 수 있다.
            "n_queries": int(len(values)),
            "mean_auc": None if not len(values) else round(float(values.mean()), 4),
            "median_auc": None if not len(values) else round(float(np.median(values)), 4),
            # 우연(0.5)보다 나은 질의의 비율. 평균 하나로는 분포가 안 보인다.
            "above_chance_fraction": None if not len(values) else round(float((values > 0.5).mean()), 4),
        }
    methods = [c for c in present if c.key in METHOD_KEYS or c.key == THREE_D_KEY]
    paired_2d, verified_2d = _paired_population(usable)
    paired_3d, verified_3d = _paired_population(three_d_usable, three_d=True)
    for first in range(len(methods)):
        for second in range(first + 1, len(methods)):
            a, b = methods[first], methods[second]
            # 3D 가 한쪽에 있으면 **양쪽을** 3D 가 잴 수 있었던 행에서 잰다.
            # 3D 는 원래 자기가 판정한 후보만으로 AUC 를 내므로, 상대를 전체
            # 후보에서 잰 값과 짝지으면 서로 다른 질문의 답을 견주게 된다.
            # 빠지는 후보는 무작위가 아니라 MCS 겹침과 컨포머 성공률이 정한다.
            use_3d = THREE_D_KEY in {a.key, b.key}
            rows = paired_3d if use_3d else paired_2d
            test = wilcoxon_signed_rank(rows[a.key].to_numpy(), rows[b.key].to_numpy())
            test["candidate_pool_verified"] = verified_3d if use_3d else verified_2d
            if not test["candidate_pool_verified"]:
                test["p_value"] = None
            test["n"] = int(len(rows))
            test["mean_difference"] = round(
                float(rows[a.key].mean() - rows[b.key].mean()), 4
            ) if len(rows) else None
            summary["paired_tests"][f"{a.key}_vs_{b.key}"] = test

    # 방법이 기준선을 이기는지가 방법끼리 비교하는 것보다 먼저다. 이기지 못하면
    # 어느 쪽이 더 나은지는 물을 필요가 없는 질문이 된다.
    summary["vs_baseline"] = {}
    for method in methods:
        rows = paired_3d if method.key == THREE_D_KEY else paired_2d
        baseline = rows[list(BASELINE_KEYS)].max(axis=1)
        test = wilcoxon_signed_rank(
            rows[method.key].to_numpy(), baseline.to_numpy()
        )
        test["comparator"] = "per-query maximum of size_baseline and polarity_baseline"
        test["candidate_pool_verified"] = verified_3d if method.key == THREE_D_KEY else verified_2d
        if not test["candidate_pool_verified"]:
            test["p_value"] = None
        test["n"] = int(len(rows))
        test["mean_difference"] = round(
            float(rows[method.key].mean() - baseline.mean()), 4
        ) if len(rows) else None
        test["query_win_fraction"] = round(
            float((rows[method.key] > baseline).mean()), 4
        ) if len(rows) else None
        summary["vs_baseline"][method.key] = test

    # 같은 실행 안에서 한 층의 안과 밖을 갈라 본다. 두 실행의 p값을 견주면 표본
    # 크기 변화를 효과 소멸로 읽게 된다 - 처음에 그렇게 틀렸다.
    summary["within_run_split"] = {}
    summary["large_classes"] = list(truth.large_classes)
    for stratum in truth.large_classes:
        inside = paired_2d[paired_2d["strata"].str.split(";").apply(lambda v: stratum in v)]
        outside = paired_2d[~paired_2d["strata"].str.split(";").apply(lambda v: stratum in v)]
        if len(inside) < 5 or len(outside) < 5:
            continue
        entry: dict[str, Any] = {"inside_n": int(len(inside)), "outside_n": int(len(outside))}
        for first in range(len(SINGLE_METHOD_KEYS)):
            for second in range(first + 1, len(SINGLE_METHOD_KEYS)):
                a, b = SINGLE_METHOD_KEYS[first], SINGLE_METHOD_KEYS[second]
                inside_test = wilcoxon_signed_rank(inside[a].to_numpy(), inside[b].to_numpy())
                outside_test = wilcoxon_signed_rank(outside[a].to_numpy(), outside[b].to_numpy())
                inside_diff = float(inside[a].mean() - inside[b].mean())
                outside_diff = float(outside[a].mean() - outside[b].mean())
                entry[f"{a}_vs_{b}"] = {
                    "inside_difference": round(inside_diff, 4),
                    "inside_p": inside_test["p_value"] if verified_2d else None,
                    "outside_difference": round(outside_diff, 4),
                    "outside_p": outside_test["p_value"] if verified_2d else None,
                    # Direction reversal is descriptive, not a direct interaction test.
                    "sign_flip": bool(inside_diff * outside_diff < 0),
                }
        summary["within_run_split"][stratum] = entry

    control = usable[CONTROL_KEY].to_numpy()
    summary["control_sanity"] = {
        "mean_auc": None if not len(control) else round(float(control.mean()), 4),
        "healthy": None if not len(control) else bool(abs(float(control.mean()) - 0.5) < 0.05),
        "note": "난수 대조가 0.5에서 벗어나면 이 평가 전체를 믿을 수 없다.",
    }
    _adjust_summary_tests(summary)
    return summary


def stratified(frame: pd.DataFrame, truth: "GroundTruth | None" = None) -> pd.DataFrame:
    """분류별로 나눠 본다. 뭉친 숫자 하나는 가장 큰 분류의 숫자일 뿐이다.

    한 가지 주의가 표에 함께 나가야 한다. 질의는 여러 목적을 신고할 수 있고, 양성
    집합은 그 **모든** 목적의 합집합이다. 그래서 COLORANT 행의 점수는 콜로런트만을
    상대로 낸 것이 아니다. `off_class_positive_fraction`이 그 비율을 말한다.

    같은 질의 집합이 이름만 다르게 두 행으로 나오는 경우도 있다(UV FILTER와
    LIGHT STABILIZER는 같은 7종이다). `duplicate_of`가 그것을 표시한다.
    """
    records: list[dict[str, Any]] = []
    frame, pool_verified = _paired_population(frame)
    exploded = frame.assign(stratum=frame["strata"].str.split(";")).explode("stratum")
    # 질의 집합을 **먼저 전부** 모은다. 채우면서 조회하면 먼저 처리된 층은 아직
    # 없는 짝을 못 찾아, 중복 표시가 한쪽에만 붙는다.
    query_sets: dict[str, frozenset] = {
        stratum: frozenset(group["query_index"])
        for stratum, group in exploded.groupby("stratum")
        if stratum
    }
    for stratum, group in exploded.groupby("stratum"):
        if not stratum:
            continue
        usable = group.dropna(subset=[c.key for c in CRITERIA if c.key in group.columns])
        if usable.empty:
            continue
        record: dict[str, Any] = {
            "stratum": stratum, "queries": int(len(usable)),
            "candidate_pool_verified": pool_verified,
        }
        for criterion in list(CRITERIA) + [THREE_D_CRITERION]:
            if criterion.key == CONTROL_KEY or criterion.key not in usable.columns:
                continue
            record[criterion.key] = round(float(usable[criterion.key].mean()), 4)
        # 이 분류에서 구조 정보가 실제로 값을 더하는가. 귀무 기준 중 **가장 센
        # 것**과 견준다. 하나만 두고 이겼다고 하면 실제보다 크게 들린다.
        best_null = max(record[k] for k in BASELINE_KEYS)
        record["best_null"] = round(best_null, 4)
        record["best_minus_null"] = round(
            max(record[k] for k in SINGLE_METHOD_KEYS) - best_null, 4
        )
        if truth is not None:
            # 이 층 질의들의 양성 중, 이 층에 속하지 않는 것의 비율.
            off = total = 0
            for query_index in usable["query_index"]:
                for positive in truth.positives.get(int(query_index), ()):  # type: ignore[union-attr]
                    total += 1
                    if stratum not in truth.strata.get(positive, ()):       # type: ignore[union-attr]
                        off += 1
            record["off_class_positive_fraction"] = round(off / total, 3) if total else None
        mine = query_sets[stratum]
        twin = [other for other, keys in query_sets.items() if other != stratum and keys == mine]
        record["duplicate_of"] = ";".join(sorted(twin))
        records.append(record)
    if not records:
        return pd.DataFrame(columns=["stratum", "queries"])
    return pd.DataFrame(records).sort_values("queries", ascending=False).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--truth", choices=["functions", "uv_filters", "curated"], default="functions")
    parser.add_argument(
        "--exclude-large-classes",
        action="store_true",
        help="모든 원료에서 대형 분류 라벨을 뗀다. 정답표 자체가 달라지므로 "
             "기본 실행의 재검정이 아니다.",
    )
    parser.add_argument(
        "--subset-queries-only",
        action="store_true",
        help="정답표는 그대로 두고 대형 분류에 속한 질의만 뺀다. "
             "'대형 분류를 빼면 결론이 유지되는가'에 답하는 쪽은 이것이다.",
    )
    parser.add_argument("--out-csv", type=Path, help="질의별 AUC를 저장")
    parser.add_argument("--out-json", type=Path, help="요약을 저장")
    parser.add_argument("--limit-queries", type=int, help="시험용. 앞의 N개 질의만.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="질의를 나눠 돌릴 프로세스 수. 질의끼리 독립이라 결과는 같아야 하고, "
             "테스트가 그것을 지킨다.",
    )
    parser.add_argument(
        "--with-3d",
        action="store_true",
        help="3D 정렬(컨포머 기반)도 함께 잰다. 질의당 라이브러리 전수에 2-9분 걸리므로 "
             "작은 정답표에서만 쓸 것.",
    )
    args = parser.parse_args()
    if args.limit_queries is not None and args.limit_queries < 1:
        parser.error("--limit-queries must be at least 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    library = load_ingredient_library()
    if args.exclude_large_classes and args.subset_queries_only:
        parser.error("--exclude-large-classes 와 --subset-queries-only 는 서로 다른 검정입니다. 하나만 쓰세요.")
    if args.truth == "functions":
        truth = functions_ground_truth(
            library,
            exclude_large=args.exclude_large_classes,
            subset_queries_only=args.subset_queries_only,
        )
    elif args.truth == "uv_filters":
        truth = uv_filter_ground_truth(library)
    else:
        truth = curated_ground_truth(library)

    if args.limit_queries:
        kept = dict(sorted(truth.positives.items())[: args.limit_queries])
        truth = GroundTruth(truth.name, truth.describe, kept,
                            {k: truth.strata[k] for k in kept}, truth.caveat,
                            truth.large_classes)

    print(f"정답표: {truth.describe}")
    positives_median = (
        int(np.median([len(v) for v in truth.positives.values()])) if truth.positives else 0
    )
    print(f"평가할 질의: {len(truth.positives)}종 · 모집단 {len(library.frame)}종 "
          f"· 질의당 중앙 양성 {positives_median}종")
    if not truth.positives:
        print("평가할 질의가 없습니다.")
        return 0

    frame = evaluate(library, truth, with_three_d=args.with_3d, workers=max(1, args.workers))
    summary = summarise(frame, truth)

    control = summary["control_sanity"]
    print(f"\n난수 대조 AUC {control['mean_auc']} — "
          + ("정상 (0.5 근처)" if control["healthy"] else "★ 0.5에서 벗어남. 평가를 믿지 마세요"))

    print("\n기준별 기술통계: 판정 가능한 후보 집합이 다를 수 있으므로 이 평균만으로 우열을 비교하지 마세요.")
    print(f"{'기준':14} {'평균 AUC':>9} {'중앙 AUC':>9} {'우연 초과 비율':>12}")
    for criterion in list(CRITERIA) + [THREE_D_CRITERION]:
        entry = summary["per_criterion"].get(criterion.key)
        if entry is None:
            continue
        print(f"{entry['label_ko']:14} {_display_number(entry['mean_auc'], '.4f'):>9} "
              f"{_display_number(entry['median_auc'], '.4f'):>9} "
              f"{_display_number(entry['above_chance_fraction'], '.3f'):>12}")
    if "three_d_unjudged" in frame.columns:
        print(f"  3D가 판정하지 못한 후보: 질의당 중앙 {int(frame['three_d_unjudged'].median())}종 "
              f"({int(frame['pool'].median())}종 중). 최하위로 세지 않고 뺐습니다.")

    print("\n기준선(구조 정보 없음) 대비 — 이것이 먼저 답해야 할 질문:")
    for key, test in summary["vs_baseline"].items():
        label = summary["per_criterion"][key]["label_ko"]
        p = _display_number(test["p_value_adjusted"], ".2e")
        verdict = "보정 후 차이 관찰" if (test["p_value_adjusted"] is not None and test["p_value_adjusted"] < 0.05
                                      and (test["mean_difference"] or 0) > 0) else "구분 안 됨"
        print(f"  {label:14} 차이 {_display_number(test['mean_difference'], '+.4f')}  이긴 질의 "
              f"{_display_number(test['query_win_fraction'], '.3f')}  Holm p={p}  → {verdict}")

    print("\n방법끼리 짝지은 비교 (Wilcoxon 부호순위):")
    for key, test in summary["paired_tests"].items():
        note = "" if test["approximation_reliable"] else "  ※ 표본 20쌍 미만, 근사 거칢"
        p = _display_number(test["p_value_adjusted"], ".2e")
        print(f"  {key:34} n={test['n']:>4}  평균차 {_display_number(test['mean_difference'], '+.4f')}  Holm p={p}{note}")

    for stratum, entry in summary.get("within_run_split", {}).items():
        flips = {k: v for k, v in entry.items()
                 if isinstance(v, dict) and v.get("sign_flip")}
        if not flips:
            continue
        print(f"\n{stratum} 안팎에서 부호가 뒤집히는 비교 "
              f"(안 {entry['inside_n']}건 / 밖 {entry['outside_n']}건):")
        for name, v in flips.items():
            print(f"  {name:34} 안 {v['inside_difference']:+.4f} (Holm p={_display_number(v['inside_p_adjusted'], '.1e')}) · "
                  f"밖 {v['outside_difference']:+.4f} (Holm p={_display_number(v['outside_p_adjusted'], '.1e')})")
        print("  → 층별 차이의 방향이 다릅니다. 교호작용을 직접 검정한 결과는 아닙니다.")

    table = stratified(frame, truth)
    if not table.empty:
        print(f"\n분류별 평균 AUC (질의 5개 이상):")
        shown = table[table["queries"] >= 5]
        print(shown.to_string(index=False) if not shown.empty else "  (해당 없음)")

    print(f"\n주의: {truth.caveat}")

    if args.out_csv:
        frame.to_csv(args.out_csv, index=False)
        print(f"wrote {args.out_csv}")
    if args.out_json:
        args.out_json.write_text(
            json.dumps({"summary": summary, "by_stratum": table.to_dict("records")},
                       indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.out_json}")
    return 0


def _display_number(value: float | None, spec: str) -> str:
    return "-" if value is None else format(value, spec)


if __name__ == "__main__":
    raise SystemExit(main())
