#!/usr/bin/env python3
"""활성 핵심구조를 유지한 대체소재 후보를 찾는다.

과제 제목이 그대로 이 파일의 일이다: *활성 핵심구조 유지형 대체소재 발굴*.
저장소의 다른 경로는 "이 화합물이 어떤 표적에 붙을까"를 묻는다. 여기서 묻는 것은
그 반대다 - **활성이 알려진 화합물을 주면, 그 핵심구조를 유지하면서 화장품 원료로
쓸 수 있는 다른 분자를 고른다.**

## 왜 유사도만으로는 안 되는가

Tanimoto 0.7은 곁사슬만 닮았을 때도, 핵심 고리가 통째로 같을 때도 나온다. 앞의
것은 대체소재가 아니다. 반대 방향의 오류가 더 나쁘다: 코직산의 파이라논 유도체는
핵심 고리를 80% 유지하는데 Tanimoto는 0.209다. 유사도 문턱을 0.25에 두었더니
정작 찾으려던 것이 먼저 버려졌다.

그래서 유사도는 같은 등급 안에서 줄 세우는 데만 쓰고, 등급은 구조 판정이
결정한다. 등재 원료가 507종뿐이라 전수 대조에 0.1-0.3초면 되므로, 미리 거를
이유도 없다.

## 판정은 두 숫자로 한다

* `coverage` = 최대 공통 부분구조 / **질의** 중원자 - 내 핵심이 남았는가
* `share`    = 최대 공통 부분구조 / **후보** 중원자 - 후보가 그 핵심으로 되어 있는가

둘 다 필요하다. coverage만 보면 나이아신아마이드(중원자 9개)의 상위 후보가
펜타닐 유도체가 된다 - 피리딘 고리와 아미드만 있으면 78%가 나오기 때문이다.
share가 그것을 걸러 낸다. Murcko 골격 일치는 세 번째 참고 지표로만 쓴다. 살리실산의
Murcko 골격은 벤젠 고리 하나뿐이라, 그것만 보면 자일레놀도 "골격 동일"이 된다.

## 두 개의 라이브러리

* **CosIng 등재 원료** - 37,071건 중 단일 저분자로 구조가 확인된 681건, 같은 구조에
  붙은 이름을 합쳐 507종. `대체소재`라는 말이 가리키는 대상이 이쪽이다. INCI
  명칭과 신고된 배합목적이 함께 나온다.
* **활성 측정 라이브러리 1,061,686종** (`similar_compounds`) - 무엇이 무엇에
  붙는지 측정된 것. 등재 원료는 아니지만, CosIng 후보가 여기에도 있으면 그
  후보의 측정값을 붙여 준다.

## 이 파일이 말하지 않는 것

핵심구조가 같다는 것은 활성이 같다는 뜻이 아니다. coverage와 share는 원자 수준의
구조 비교이며 측정된 약리 정보가 아니고, 등급 경계값(1.0/0.75/0.5/0.25)은 고른
값이지 보정된 값이 아니다. CosIng의 배합목적은 등재 시 신고된 항목이지 이 도구가
확인한 것이 아니다. 그 구분이 화면에서 사라지지 않도록 모든 반환값이 근거를
함께 들고 다닌다.

    python scripts/alternative_ingredients.py --smiles 'NC(=O)c1cccnc1'
    python scripts/alternative_ingredients.py --input-csv actives.csv --out-csv candidates.csv
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from rdkit import Chem, DataStructs, RDLogger  # noqa: E402
from rdkit.Chem import rdFMCS  # noqa: E402
from rdkit.Chem.Scaffolds import MurckoScaffold  # noqa: E402

from build_activity_retrieval_index import MORGAN_GENERATOR, _standardize_mol  # noqa: E402
from similar_compounds import _POPCOUNT  # noqa: E402

RDLogger.DisableLog("rdApp.*")

DEFAULT_COSING = ROOT / "data" / "cosing" / "cosing.parquet"
DEFAULT_COSING_MANIFEST = ROOT / "data" / "cosing" / "cosing_ingest_manifest.json"

# 질의 분자가 후보 안에 얼마나 남아 있는지. 경계값은 임의 선택이며 측정으로
# 보정된 값이 아니다 - 그래서 등급과 함께 실제 비율(coverage)을 항상 돌려준다.
CONTAINS_COVERAGE = 1.0
RETAINED_COVERAGE = 0.75
PARTIAL_COVERAGE = 0.5
WEAK_COVERAGE = 0.25

# 후보 분자에서 그 공통 구조가 차지하는 비중. 이것이 없으면 작은 질의는 무엇이든
# 통과시킨다: 나이아신아마이드는 중원자 9개라, 피리딘 고리와 아미드만 있으면
# 아무 분자나 유지율 78%가 나온다 - 실제로 펜타닐 유도체와 살균제가 상위에
# 올라왔다. 핵심이 남아 있는지(coverage)와 후보가 그 핵심으로 이루어져 있는지
# (share)는 다른 질문이고, 대체소재는 둘 다 필요하다.
SHARE_MIN = 0.5
PARTIAL_SHARE_MIN = 0.35

# 정렬은 유사도가 아니라 이 순서가 먼저다. 과제 제목이 요구하는 것이 유사도가
# 아니라 핵심구조 유지이기 때문이다.
GRADE_ORDER = {
    "identical": 0,
    "contains": 1,
    "retained": 2,
    "partial": 3,
    "embedded": 4,
    "weak": 5,
    "different": 6,
    "unknown": 7,
}

# 대체소재로 내놓을 수 있는 등급. "일부 유지"부터는 핵심이 이미 바뀐 것이므로
# 후보 목록에는 남기되 유지된 것으로 세지 않는다.
CORE_KEPT_GRADES = ("identical", "contains", "retained")

GRADE_LABEL_KO = {
    "identical": "입력과 같은 구조",
    "contains": "핵심구조 그대로",
    "retained": "핵심구조 유지",
    "partial": "일부 유지",
    "embedded": "핵심 포함, 다른 분자",
    "weak": "거의 다름",
    "different": "다른 구조",
    "unknown": "판정 불가",
}

# 등재 원료 507종 전수에 MCS를 돌리므로 한 건이 오래 끌면 전체가 끌린다. 실측은
# 질의당 0.08-0.16초이고, 이 상한은 병적인 입력이 그것을 초 단위로 늘리는 것을
# 막는다. 상한에 걸린 결과는 `basis`에 `mcs_timeout`으로 남아, 확정된 판정이
# 아니라 하한이라는 사실이 화면까지 간다.
MCS_TIMEOUT_SECONDS = 2

# 한 건이 아니라 한 요청 전체의 상한. 예산을 다 쓰면 남은 후보는 판정하지 않고
# `budget_exhausted`로 남긴다 - 조용히 "다른 구조"로 적어 두면 안 본 것을 봤다고
# 말하는 셈이 된다.
#
# 15초였고, 그 값은 라이브러리가 507종이던 때 정해졌다. 지금은 7,700종이고 예산이
# 실제로 물린다: 우르솔산(펜타사이클릭 트리테르펜, 흔한 원료)은 15초에 1,303종만
# 판정하고 **6,419종(83%)을 안 본 채** 끝났다. 안 본 후보는 coverage 0.0으로
# 남아 순위 계산에 "구조가 전혀 다름"으로 들어간다 - 순위가 조용히 틀린다.
#
# 비용은 쌍당 중앙값 0.14ms인데 상위 1%가 533ms다(축합 다환끼리의 MCS). 즉 예산을
# 올려도 보통 질의는 그대로 1-3초이고, 병적인 질의만 오래 끈다. 전수 실측은
# 우르솔산 148초, 토코페롤 2.7초, 세라마이드NP 2.6초. 600초는 최악 관측의 4배다.
# 쌍당 상한(MCS_TIMEOUT_SECONDS)이 따로 있으므로 한 건이 무한히 끌지는 않는다.
MCS_BUDGET_SECONDS = float(os.environ.get("SKINSCOUT_MCS_BUDGET_SECONDS", "600"))

# 방향족 고리와 포화 고리를 같은 것으로 보지 않는다. 기본 BondCompare.CompareOrder
# 는 고리 결합 질의를 "방향족 또는 단일"로 내보내서, 살리실산과 그 완전 포화
# 유사체가 유지율 100%로 나왔다. 벤젠과 사이클로헥산은 활성 핵심구조로서 같은
# 것이 아니다. Exact로 바꿔도 실제 후보들의 값은 그대로다(3-하이드록시벤조산
# 0.90, 에틸헥실살리실레이트 1.00, 레티닐아세테이트 1.00).
MCS_BOND_COMPARE = rdFMCS.BondCompare.CompareOrderExact

# 이 도구가 다루는 것은 화장품 원료다. 등재 원료 중 가장 큰 것도 중원자 100개
# 아래이고, 그보다 훨씬 큰 입력은 대체소재를 찾는 질문이 아니다. 상한을 두지
# 않으면 골격 계산 하나가 초 단위로 늘어나 요청이 워커를 붙잡는다.
MAX_QUERY_HEAVY_ATOMS = 200


@dataclass(frozen=True)
class CoreRetention:
    """질의의 활성 핵심구조가 후보 안에 살아 있는지에 대한 판정 하나.

    세 가지를 따로 잰다. `coverage`는 질의 분자의 중원자 중 몇 %가 후보 안에서
    그대로 이어지는지, `share`는 반대로 후보 분자가 그 공통 구조로 얼마나
    이루어져 있는지, `scaffold_match`는 질의의 Bemis-Murcko 고리 골격이 후보
    안에 통째로 들어 있는지다.

    셋이 다 필요한 이유는 하나씩으로는 전부 거짓말이 되기 때문이다. 살리실산의
    Murcko 골격은 벤젠 고리 하나뿐이므로, 골격만 보면 2,3-자일레놀도 "골격
    동일"이 된다 - 벤젠을 가진 모든 분자가 그렇게 된다. coverage만 보면
    나이아신아마이드(중원자 9개)의 상위 후보가 펜타닐 유도체가 된다 - 피리딘과
    아미드만 있으면 78%가 나오기 때문이다. share가 그 둘을 갈라 준다.
    """

    grade: str
    coverage: float          # MCS / 질의 중원자 - 내 핵심이 남았는가
    share: float             # MCS / 후보 중원자 - 후보가 그 핵심으로 이루어졌는가
    matched_atoms: int
    query_atoms: int
    candidate_atoms: int
    scaffold_match: bool
    scaffold_smiles: str
    basis: str

    def public(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["label_ko"] = GRADE_LABEL_KO.get(self.grade, self.grade)
        return payload


def murcko_core(molecule: Chem.Mol) -> tuple[Chem.Mol | None, str]:
    """Bemis-Murcko 고리 골격. 고리가 없으면 골격도 없으므로 그 사실을 돌려준다.

    등재 원료 507종 중 130종이 비고리형이다. 이때 빈 골격을 상대로 부분구조 검사를
    하면 무엇이든 일치로 나온다 - 그 함정을 호출자가 아니라 여기서 닫는다.
    """
    if molecule is None:
        return None, "no_molecule"
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    except (RuntimeError, ValueError):
        return None, "scaffold_failed"
    if scaffold is None or scaffold.GetNumHeavyAtoms() == 0:
        return None, "acyclic_query"
    return scaffold, "murcko_scaffold"


class McsBudget:
    """한 요청이 MCS에 쓸 수 있는 총 시간. 다 쓰면 남은 후보는 판정하지 않는다."""

    def __init__(self, seconds: float | None = None) -> None:
        # 기본값을 인자 자리에 두면 모듈 상수를 바꿔도 반영되지 않는다(기본 인자는
        # 정의 시점에 한 번만 평가된다). 상한을 조정하거나 시험할 수 있어야 한다.
        limit = MCS_BUDGET_SECONDS if seconds is None else seconds
        self.deadline = time.monotonic() + max(0.0, limit)

    def exhausted(self) -> bool:
        return time.monotonic() >= self.deadline


@dataclass(frozen=True)
class PreparedQuery:
    """후보마다 다시 계산할 이유가 없는 질의 쪽 값들.

    이것이 없으면 `core_retention`이 후보 507개 각각에 대해 질의의 정규화 SMILES와
    Murcko 골격을 다시 만든다. 작은 분자에서는 티가 안 나지만 중원자 1,200개짜리
    입력에서는 골격 계산 하나가 1.7초라 요청 하나가 841초가 된다.
    """

    mol: Chem.Mol
    canonical: str
    heavy_atoms: int
    scaffold: Chem.Mol | None
    scaffold_smiles: str
    scaffold_basis: str


def prepare_query(query: Chem.Mol) -> PreparedQuery:
    scaffold, basis = murcko_core(query)
    return PreparedQuery(
        mol=query,
        canonical=Chem.MolToSmiles(query),
        heavy_atoms=query.GetNumHeavyAtoms(),
        scaffold=scaffold,
        scaffold_smiles=Chem.MolToSmiles(scaffold) if scaffold is not None else "",
        scaffold_basis=basis,
    )


def core_retention(
    query: Chem.Mol,
    candidate: Chem.Mol,
    budget: "McsBudget | None" = None,
    prepared: "PreparedQuery | None" = None,
) -> CoreRetention:
    """질의의 핵심구조가 후보 분자 안에 남아 있는가.

    Murcko 골격만으로 판정하지 않는다. 골격은 곁사슬을 전부 떼어내므로, 단고리
    분자에서는 "고리가 벤젠이다" 이상의 정보가 남지 않는다. 대신 질의 분자
    **전체**를 기준으로 최대 공통 부분구조가 몇 %를 덮는지를 잰다.

    사다리는 세 단이다. 정규화 SMILES가 같으면 `identical`, 질의 분자가 후보의
    부분구조로 통째로 들어 있으면 `contains`(에스터화·아실화된 유도체가 여기
    걸린다), 둘 다 아니면 MCS로 남은 비율을 잰다. 고리를 쪼개는 부분 일치는
    세지 않는다(`completeRingsOnly`) - 벤젠의 절반은 벤젠이 아니다.

    어느 단에서든 `share`가 낮으면 등급을 내린다. 핵심이 통째로 들어 있어도 후보가
    그보다 훨씬 큰 분자라면 그것은 대체소재가 아니라 "그 조각을 가진 다른
    물질"이고, 그 경우를 `embedded`로 따로 부른다.
    """
    if query is None or candidate is None:
        return CoreRetention("unknown", 0.0, 0.0, 0, 0, 0, False, "", "no_molecule")

    if prepared is None:
        prepared = prepare_query(query)
    query_atoms = prepared.heavy_atoms
    if query_atoms == 0:
        return CoreRetention("unknown", 0.0, 0.0, 0, 0, 0, False, "", "empty_query")

    scaffold, scaffold_basis = prepared.scaffold, prepared.scaffold_basis
    scaffold_smiles = prepared.scaffold_smiles
    scaffold_match = False
    if scaffold is not None:
        try:
            scaffold_match = bool(candidate.HasSubstructMatch(scaffold, useChirality=True))
        except (RuntimeError, ValueError):
            scaffold_match = False

    candidate_atoms = max(1, candidate.GetNumHeavyAtoms())
    if prepared.canonical == Chem.MolToSmiles(candidate):
        return CoreRetention(
            "identical", 1.0, 1.0, query_atoms, query_atoms, candidate_atoms,
            scaffold_match, scaffold_smiles, "same_structure",
        )

    try:
        if candidate.HasSubstructMatch(query, useChirality=True):
            share = query_atoms / candidate_atoms
            return CoreRetention(
                "contains" if share >= SHARE_MIN else "embedded",
                1.0, round(share, 3), query_atoms, query_atoms, candidate_atoms,
                scaffold_match, scaffold_smiles, "query_is_substructure",
            )
        if candidate.HasSubstructMatch(query, useChirality=False):
            # Connectivity is identical but the specified stereochemistry is
            # opposite or absent. It cannot establish retention of the core.
            return CoreRetention(
                "unknown", 1.0, round(query_atoms / candidate_atoms, 3),
                query_atoms, query_atoms, candidate_atoms, scaffold_match,
                scaffold_smiles, "stereochemistry_mismatch_or_unspecified",
            )
    except (RuntimeError, ValueError):
        pass

    if budget is not None and budget.exhausted():
        return CoreRetention(
            "unknown", 0.0, 0.0, 0, query_atoms, candidate_atoms,
            scaffold_match, scaffold_smiles, "budget_exhausted",
        )

    try:
        result = rdFMCS.FindMCS(
            [query, candidate],
            timeout=MCS_TIMEOUT_SECONDS,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
            matchChiralTag=True,
            bondCompare=MCS_BOND_COMPARE,
        )
    # MemoryError 는 **일부러 잡지 않는다.** `FindMCS` 는 병렬로 여럿 돌릴 때
    # 메모리 압력에 따라 이것을 던지는데, 크기로 예측되지 않는다 - 가장 큰 질의
    # 12개 × 후보 100개 = 1,200쌍을 단독으로 돌리면 한 번도 나지 않는다. 즉
    # 같은 쌍이 부하에 따라 나기도 하고 안 나기도 한다.
    #
    # 그것을 잡아서 `mcs_failed` 로 남기면 **직렬과 병렬의 결과가 달라진다.**
    # 실제로 잡아 봤더니 `test_parallel_evaluation_matches_serial_bit_for_bit` 가
    # "structural 열이 다릅니다"로 깨졌다. 3D 경로에서 벽시계 상한 때문에 이미
    # 겪은 것과 같은 종류다 - 재는 값이 기계 부하에 따라 달라지면 그것은 측정이
    # 아니다. 터지면 시끄럽게 터지는 편이 낫고, 그때는 워커 수를 줄이면 된다.
    except (RuntimeError, ValueError):
        return CoreRetention(
            "unknown", 0.0, 0.0, 0, query_atoms, candidate_atoms,
            scaffold_match, scaffold_smiles, "mcs_failed",
        )

    matched = int(result.numAtoms or 0)
    coverage = matched / query_atoms
    share = matched / candidate_atoms
    # 시간이 다 되어 끊긴 MCS는 하한이지 답이 아니다. 등급을 그대로 쓰되
    # 근거에 남겨, 화면이 확정된 판정처럼 읽히지 않게 한다.
    basis = "mcs_timeout" if getattr(result, "canceled", False) else "mcs"
    if scaffold_basis == "acyclic_query":
        basis = f"{basis}_acyclic_query"

    if coverage >= CONTAINS_COVERAGE and share >= SHARE_MIN:
        grade = "contains"
    elif coverage >= RETAINED_COVERAGE and share >= SHARE_MIN:
        grade = "retained"
    elif coverage >= PARTIAL_COVERAGE and share >= PARTIAL_SHARE_MIN:
        grade = "partial"
    elif coverage >= PARTIAL_COVERAGE:
        # 핵심은 통째로 들어 있는데 후보가 그보다 훨씬 큰 분자다. 대체소재가
        # 아니라 "그 조각을 가진 다른 물질"이므로 등급을 따로 둔다.
        grade = "embedded"
    elif coverage >= WEAK_COVERAGE:
        grade = "weak"
    else:
        grade = "different"
    return CoreRetention(
        grade, round(coverage, 3), round(share, 3), matched, query_atoms, candidate_atoms,
        scaffold_match, scaffold_smiles, basis,
    )


# --- 파마코포어 신호 -----------------------------------------------------------
#
# 구조 판정과는 **다른 것을 재는** 두 번째 신호다. 앞의 것은 원자와 결합을 보고,
# 이것은 수소결합 주개/받개·방향족·소수성 같은 상호작용 특징을 본다. 실제로 크게
# 어긋난다: 살리실산에 대해 구조 1위는 3-HYDROXYBENZOIC ACID(입력과 같은 표적
# 5종에서 측정됨)인데 파마코포어 1위는 3-METHYL-3-HYDROXYBUTYRIC ACID(구조 등급은
# '거의 다름')이다.
#
# 둘을 한 점수로 합치지 않는 이유는 **합치면 순위가 나빠져서가 아니다.** 처음에는
# 그렇게 적었는데 틀렸다 - 재 보니 세 기준의 순위평균이 셋 각각을 모두 이긴다
# (AUC 0.685 대 0.657/0.645/0.666, docs/ALTERNATIVE_CRITERIA_EVAL.md). 그 측정은
# 등재 원료 507종·질의 351개일 때 낸 것이고, 지금 라이브러리는 7,484종·질의
# 6,268개다 - 일이 270배라 아직 다시 재지 못했다. 방향의 근거로는 유효하지만
# 절대값을 지금 모집단의 성능으로 읽으면 안 된다. 이유는
# 어긋남의 **부호가 화합물 분류에 따라 뒤집히고 그 뒤집힘이 유의하기** 때문이다.
# 향료에서는 구조 기준이 파마코포어보다 낫고(+0.054, p=7e-08), 그 밖에서는 반대다
# (-0.009, p=3e-02). 숫자 하나만 보여 주면 읽는 사람은 자기 화합물이 어느 쪽인지
# 알 수 없다. 합친 순위를 선택지로 더하는 것은 이 측정이 지지하는 방향이며, 아직
# 넣지 않았다.
#
# 계산은 Stage 5.6b(`discover_substitutes.py`)의 것을 그대로 쓴다. 두 화면이
# "파마코포어"라는 같은 말로 다른 것을 재면 안 되기 때문이며, 두 모듈이 같은 값을
# 내는지는 테스트가 지킨다.
#
# 5.6b의 통과 기준(pharmacophore_score >= 0.55, feature_recall >= 0.60)은 가져오지
# 않는다. 그 값은 같은 표적에 활성이 보고된 화합물 풀에 맞춰진 것이고, 등재 원료
# 507종에 그대로 적용하면 질의당 0-1종만 통과한다. 여기서는 판정하지 않고 측정값만
# 보여 준다.
PHARMACOPHORE_FAMILIES = (
    "Donor", "Acceptor", "Aromatic", "Hydrophobe", "PosIonizable", "NegIonizable",
)


@dataclass(frozen=True)
class PharmacophoreMatch:
    """입력과 후보의 상호작용 특징이 얼마나 겹치는가.

    `recall`과 `precision`은 구조 쪽의 `coverage`/`share`와 같은 구조다 - 앞의 것은
    "내 특징이 남았나", 뒤의 것은 "후보가 그 특징으로 이루어졌나"를 묻는다.
    `similarity`는 Gobbi 2D 파마코포어 지문의 Tanimoto로, 특징의 개수뿐 아니라
    특징 사이의 거리 관계까지 본다.
    """

    similarity: float
    recall: float
    precision: float
    query_features: dict[str, int]
    candidate_features: dict[str, int]
    usable: bool

    def public(self) -> dict[str, Any]:
        return {
            "similarity": self.similarity,
            "recall": self.recall,
            "precision": self.precision,
            "query_features": dict(self.query_features),
            "candidate_features": dict(self.candidate_features),
            "usable": self.usable,
        }


def _pharmacophore_backend():
    """5.6b의 계산을 그대로 가져온다. 무거운 자원이라 한 번만 만든다."""
    global _PHARM_BACKEND
    if _PHARM_BACKEND is None:
        from pathlib import Path as _Path

        from rdkit import RDConfig
        from rdkit.Chem import ChemicalFeatures
        from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D

        from discover_substitutes import _feature_counts, _feature_overlap_scores

        factory = ChemicalFeatures.BuildFeatureFactory(
            str(_Path(RDConfig.RDDataDir) / "BaseFeatures.fdef")
        )
        _PHARM_BACKEND = (factory, Generate, Gobbi_Pharm2D, _feature_counts, _feature_overlap_scores)
    return _PHARM_BACKEND


_PHARM_BACKEND: Any = None


# Gobbi 2D 지문은 특징 삼중항을 전수 열거하므로 비용이 특징 수에 따라 폭발한다.
# 상한이 없어서 큰 유연 분자 하나가 화면 전체를 멈춰 세웠다 - `파마코포어 특징도
# 함께 보기`와 `세 기준 합친 순위` 정렬이 실제 라이브러리에서 180초에도 돌아오지
# 않았고, 표본 성능 평가는 병렬 풀에 도달조차 못 했다.
#
# 실측(표본 700종, 분자당 3초에서 끊음):
#
#   특징  0~30개   633종   3초 초과 0종   최대 0.31초
#   특징 30~40개    44종   3초 초과 1종
#   특징 40개 이상  23종   3초 초과 10종
#
# 끊긴 것들은 DECAPEPTIDE-12(중원자 100), DIPENTAERYTHRITYL HEXAHYDROXYSTEARATE
# (137), TRIBEHENIN 같은 큰 유연 지질·펩타이드다. 저분자 질의의 대체 후보로는
# 어차피 의미가 없는 것들이다.
#
# **시간이 아니라 특징 수로 끊는다.** 벽시계로 끊으면 CPU 경합만으로 같은 분자가
# "잰 값"과 "판정 불가" 사이를 오가고, 병렬과 직렬의 결과가 달라진다 - 3D 경로에서
# 이미 겪은 문제다. 특징 수는 기계와 부하에 무관하다.
MAX_PHARMACOPHORE_FEATURES = 30


def pharmacophore_profile(molecule: Chem.Mol) -> tuple[Any, dict[str, int]]:
    """한 분자의 Gobbi 2D 지문과 특징족 개수. 후보 쪽은 라이브러리마다 한 번만.

    특징이 너무 많으면 지문을 **비워서** 돌려준다. 빈 지문은
    `pharmacophore_match` 에서 `usable=False` 가 되고, 그것은 "특징이 하나도 안
    겹친다"가 아니라 **잴 수 없다**는 뜻으로 이미 처리돼 있다.
    """
    from rdkit import DataStructs

    factory, generate, gobbi, feature_counts, _ = _pharmacophore_backend()
    counts = dict(feature_counts(molecule, factory))
    if sum(counts.values()) > MAX_PHARMACOPHORE_FEATURES:
        return DataStructs.ExplicitBitVect(gobbi.factory.GetSigSize()), counts
    return generate.Gen2DFingerprint(molecule, gobbi.factory), counts


def pharmacophore_match(
    query_profile: tuple[Any, dict[str, int]],
    candidate_profile: tuple[Any, dict[str, int]],
) -> PharmacophoreMatch:
    from collections import Counter

    from rdkit import DataStructs

    _, _, _, _, overlap_scores = _pharmacophore_backend()
    query_fp, query_features = query_profile
    candidate_fp, candidate_features = candidate_profile
    usable = bool(query_fp.GetNumOnBits()) and bool(candidate_fp.GetNumOnBits())
    similarity = float(DataStructs.TanimotoSimilarity(query_fp, candidate_fp)) if usable else 0.0
    recall, precision, _f1 = overlap_scores(Counter(query_features), Counter(candidate_features))
    return PharmacophoreMatch(
        similarity=round(similarity, 3),
        recall=round(recall, 3),
        precision=round(precision, 3),
        query_features=query_features,
        candidate_features=candidate_features,
        usable=usable,
    )


@dataclass
class IngredientLibrary:
    """화장품 원료로 등재된 분자들. 대체소재 후보가 나올 수 있는 유일한 모집단."""

    frame: pd.DataFrame
    molecules: list[Chem.Mol]
    fingerprints: np.ndarray   # (n, 256) uint8, packed bits
    popcounts: np.ndarray      # (n,) int32
    source: Path
    unparsed: int              # 구조가 없어 후보가 될 수 없었던 등재 항목 수
    # CosIng 원본 등재 건수. 모집단이 681종인 것은 필터가 엄해서가 아니라
    # 나머지가 추출물·혼합물·고분자여서 단일 구조가 없기 때문이다. 화면이 "681종"
    # 만 보여 주면 작은 라이브러리로 오해받고, 그 오해는 후보가 안 나올 때
    # 도구를 의심하게 만든다.
    registered_entries: int = 0
    resolved_entries: int = 0   # 그중 단일 구조로 해석된 등재 건수(구조 중복 포함)
    # 등재 원료 전체의 Gobbi 지문과 특징 계수. 7,484종에 실측 118초이므로(507종
    # 시절 3.7초에서 늘었다) 파마코포어를 처음 쓸 때만
    # 만들고, 그 뒤로는 질의당 0.02초다.
    pharmacophore_profiles: list[Any] | None = None


def ensure_pharmacophore_profiles(library: "IngredientLibrary") -> list[Any]:
    if library.pharmacophore_profiles is None:
        library.pharmacophore_profiles = [pharmacophore_profile(m) for m in library.molecules]
    return library.pharmacophore_profiles


def _split_list(value: Any) -> list[str]:
    """CosIng는 CAS와 배합목적을 세미콜론이나 슬래시로 이어 붙여 둔다."""
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return []
    parts: list[str] = []
    for chunk in text.replace("/", ";").split(";"):
        chunk = " ".join(chunk.split())
        if chunk:
            parts.append(chunk)
    return parts


def _extend_unique(target: list[str], values: list[str]) -> None:
    for value in values:
        if value and value not in target:
            target.append(value)


def _name_sort_key(name: str) -> tuple[int, int, str]:
    """읽는 사람이 알아볼 이름을 앞으로.

    한 구조에 붙은 이름 중에는 줄바꿈이 섞인 IUPAC 나열도 있고 `RETINYL ACETATE`
    같은 INCI 명칭도 있다. 줄바꿈이 없는 짧은 쪽이 거의 언제나 INCI 명칭이다.
    """
    multiline = 1 if ("\n" in name or "\r" in name) else 0
    return (multiline, len(name), name)


def _packed(fingerprint: DataStructs.ExplicitBitVect) -> np.ndarray:
    array = np.zeros(2048, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fingerprint, array)
    return np.packbits(array)


def load_ingredient_library(
    path: Path = DEFAULT_COSING, manifest: Path = DEFAULT_COSING_MANIFEST
) -> IngredientLibrary:
    """CosIng 표를 읽되 지문은 다시 계산한다.

    저장된 `ecfp4` 열을 그대로 쓰지 않는 이유가 있다. 그 열은
    `stage0_cosing.py`가 표준화 없이 만든 것이고, 질의 쪽은
    `build_activity_retrieval_index._standardize_mol`을 거친다. 두 지문을 맞대면
    같은 분자끼리도 유사도가 1이 아니게 나온다 - 이 저장소에서 이미 한 번 겪은
    종류의 오차다. 681종을 다시 표준화하는 데 0.3초면 되므로, 비교 가능성을
    포기할 이유가 없다.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"CosIng 원료 표가 없습니다: {path}. scripts/stage0_cosing.py 로 먼저 만드세요."
        )
    columns = ["inci_name", "cas", "functions", "smiles", "inchikey"]
    raw = pd.read_parquet(path, columns=columns)

    # 같은 구조가 이름을 여러 개로 등재돼 있다. 레티닐 아세테이트는 INCI 명칭과
    # IUPAC 명칭으로 두 줄이고, 그대로 두면 상위 20칸 중 두 칸을 같은 분자가
    # 차지한다. 구조로 묶고 이름을 합친다.
    by_structure: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    molecule_of: dict[str, Chem.Mol] = {}
    unparsed = 0
    for record in raw.to_dict("records"):
        try:
            mol = _standardize_mol(str(record.get("smiles") or "").strip())
        except (ValueError, RuntimeError):
            unparsed += 1
            continue
        if mol is None or mol.GetNumHeavyAtoms() == 0:
            unparsed += 1
            continue
        key = Chem.MolToInchiKey(mol) or ""
        if not key:
            unparsed += 1
            continue
        entry = by_structure.get(key)
        if entry is None:
            entry = {
                "names": [],
                "cas": [],
                "functions": [],
                "canonical_smiles": Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
                "inchikey": key,
                "skeleton": key[:14],
                "entries": 0,
            }
            by_structure[key] = entry
            order.append(key)
            molecule_of[key] = mol
        entry["entries"] += 1
        _extend_unique(entry["names"], [str(record.get("inci_name") or "").strip()])
        _extend_unique(entry["cas"], _split_list(record.get("cas")))
        _extend_unique(entry["functions"], _split_list(record.get("functions")))

    if not by_structure:
        raise ValueError(f"CosIng 표에서 구조를 하나도 읽지 못했습니다: {path}")

    rows: list[dict[str, Any]] = []
    molecules: list[Chem.Mol] = []
    packed: list[np.ndarray] = []
    for key in order:
        entry = by_structure[key]
        names = sorted((n for n in entry["names"] if n), key=_name_sort_key)
        rows.append(
            {
                "inci_name": names[0] if names else "",
                "synonyms": names[1:],
                "cas": "; ".join(entry["cas"]),
                "functions": "; ".join(entry["functions"]),
                "canonical_smiles": entry["canonical_smiles"],
                "inchikey": entry["inchikey"],
                "skeleton": entry["skeleton"],
                "registered_names": entry["entries"],
            }
        )
        molecules.append(molecule_of[key])
        packed.append(_packed(MORGAN_GENERATOR.GetFingerprint(molecule_of[key])))

    registered = resolved = 0
    if manifest.is_file():
        try:
            ingest = json.loads(manifest.read_text(encoding="utf-8"))
            registered = int(ingest.get("input_rows") or 0)
            resolved = int(ingest.get("resolved_rows") or 0)
        except (ValueError, OSError):
            registered = resolved = 0

    fingerprints = np.vstack(packed)
    return IngredientLibrary(
        frame=pd.DataFrame(rows),
        molecules=molecules,
        fingerprints=fingerprints,
        popcounts=_POPCOUNT[fingerprints].sum(1, dtype=np.int32),
        source=path,
        unparsed=unparsed,
        registered_entries=registered,
        resolved_entries=resolved,
    )


# 정렬 방식. 기본은 핵심구조 유지 순이고, 합친 순위는 켜야 나온다.
SORT_CORE_FIRST = "core"
SORT_MERGED = "merged"
# 구조를 전혀 보지 않고 극성만 가까운 순. 정답표 평가에서 이것이 향료·방향 분류의
# 질의에서 세 구조 기준 전부를 이겼다(AUC 0.841 대 최고 0.642). 다만 이기는 범위가
# 좁다: 출처 인용 대체 쌍 정답표와 UV 필터에서는 상위 4위 안에도 들지 못한다.
# 대체소재를 잘 찾는 것이 아니라 "휘발성이 비슷한 것"을 잘 찾는 것이고, 향료 분류의
# 소속이 대체로 휘발성으로 정해지기 때문이다.
SORT_POLARITY = "polarity"
SORT_MODES = (SORT_CORE_FIRST, SORT_MERGED, SORT_POLARITY)


def percentile_rank(values: np.ndarray) -> np.ndarray:
    """0-1 백분위 순위. 세 기준을 합칠 때 눈금 차이를 없앤다.

    원점수를 그대로 더하면 눈금이 촘촘한 쪽이 이긴다 - 구조 유지율은 중원자 수로
    나눈 계단값이라 507종에서 서로 같은 값이 수십 개씩 나오는 반면, Tanimoto는
    촘촘한 실수다.

    `scripts/eval_alternative_criteria.py`가 합친 순위를 평가할 때 부르는 것과
    **같은 함수**여야 한다. 다르면 화면이 내는 순위와 측정한 순위가 달라지고,
    그 차이는 화면을 봐서는 알 수 없다.

    동점은 평균 순위로 다룬다. 서수로 매기면 값이 완전히 같은 두 후보의 백분위가
    갈리고, 누가 이기는지를 라이브러리 parquet의 행 순서가 정하게 된다 - 화면의
    세 열이 똑같은 원료가 수십 위 떨어져 나오고, 그 차이를 설명할 근거가 화면
    어디에도 없다. 같은 화면의 신호 불일치 지표도 평균 순위를 쓴다.
    값이 서로 다른 경우의 결과는 서수 방식과 같다.

    NaN은 "재지 못했다"는 뜻이므로 순위에 넣지 않고 NaN으로 내보낸다. 분모도 잰
    것의 개수에서 센다 - 전체 길이로 나누면 못 잰 자리만큼 최고값이 1.0에 닿지
    못해, 기준마다 눈금이 달라진다. NaN이 하나도 없으면 예전과 값이 같다.
    """
    if len(values) == 0:
        return np.zeros(0, dtype=float)
    series = pd.Series(np.asarray(values, dtype=float))
    ranks = series.rank(method="average").to_numpy()
    return (ranks - 1.0) / max(1, int(series.notna().sum()) - 1)


def merged_rank_score(scores: dict[str, np.ndarray], keys: tuple[str, ...]) -> np.ndarray:
    """세 기준의 백분위 순위 평균.

    한 기준을 재지 못한 후보는 **잰 기준들로만** 평균한다. 못 잰 자리를 0으로
    메우면 재지 못한 것이 "가장 나쁨"으로 순위에 들어가고, 그 후보는 다른 두
    기준이 아무리 좋아도 올라오지 못한다. 셋 다 못 쟀으면 NaN이고, 정렬에서
    맨 뒤로 간다 - 판정하지 못한 것을 위로 올리지도 않는다.
    """
    stacked = np.vstack([percentile_rank(scores[key]) for key in keys])
    with warnings.catch_warnings():
        # 셋 다 NaN인 행은 nanmean이 경고를 내지만, 그 NaN이 여기서는 답이다.
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(stacked, axis=0)


# 판정으로 인정하지 않는 `core_retention` 근거. 이 근거로 나온 coverage 는
# 측정값이 아니므로 백분위 순위에 넣지 않는다.
UNJUDGED_BASES = ("budget_exhausted", "mcs_failed", "no_molecule", "empty_query",
                  "stereochemistry_mismatch_or_unspecified")


def judged_coverage(retention: "CoreRetention") -> float | None:
    """실제로 잰 유지율만 돌려준다. 재지 못했으면 None.

    `mcs_timeout`은 값이 있어도 하한이지 측정값이 아니다 - 시간이 더 있었으면 더
    겹쳤을 수 있으므로, 그 값으로 줄을 세우면 오래 걸린 쌍이 체계적으로 밀린다.
    """
    if retention.grade == "unknown" or retention.basis in UNJUDGED_BASES:
        return None
    if str(retention.basis).startswith("mcs_timeout"):
        return None
    return retention.coverage


def _tanimoto(library_fingerprints: np.ndarray, popcounts: np.ndarray, query: np.ndarray) -> np.ndarray:
    query_bits = int(_POPCOUNT[query].sum())
    intersection = _POPCOUNT[np.bitwise_and(library_fingerprints, query)].sum(1, dtype=np.int32)
    union = popcounts + query_bits - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, intersection / union, 0.0)


@dataclass
class AlternativeScan:
    """대조한 것 전부에 대한 결과. 화면에 몇 줄을 보여 줄지와는 별개다.

    `scanned`와 `grade_counts`가 있어야 "아직 안 봤다"와 "다 봤는데 없다"를 구분할
    수 있다. 미백 소재처럼 등재 원료 쪽에 유사 구조가 아예 없는 경우가 실제로
    있고, 그때 빈 표만 내놓으면 읽는 사람은 도구를 의심한다.
    """

    frame: pd.DataFrame
    scanned: int
    grade_counts: dict[str, int]
    kept: int
    library_size: int

    @property
    def unjudged(self) -> int:
        """계산 상한에 걸려 판정하지 못한 후보 수."""
        return self.grade_counts.get("unknown", 0)


def scan_alternatives(
    smiles: str,
    library: IngredientLibrary,
    *,
    min_similarity: float = 0.0,
    require_core: bool = False,
    exclude_self: bool = False,
    with_pharmacophore: bool = False,
    sort_by: str = SORT_CORE_FIRST,
) -> AlternativeScan:
    """등재 원료 전체를 핵심구조 유지 기준으로 줄 세운다.

    Tanimoto로 미리 걸러내지 않는다. 처음에는 유사도 0.25 아래를 버렸는데, 그
    문턱이 정작 찾으려던 것을 먼저 버렸다: 코직산의 파이라논 유도체
    (2-METHYL-3-OXOPROPOXY-PYRAN-4-ONE)는 핵심 고리를 80% 유지하면서 유사도는
    0.209다. 곁사슬이 크게 달라지면 핵심이 그대로여도 Tanimoto는 0.2대로
    떨어지기 때문이다.

    507종 전체에 MCS를 돌려도 0.08-0.16초라서, 거를 이유가 없다. `min_similarity`는
    호출자가 굳이 원할 때만 쓰는 추가 필터로 남긴다.
    """
    if not math.isfinite(min_similarity) or not 0 <= min_similarity <= 1:
        raise ValueError("min_similarity must be finite and between 0 and 1")
    if sort_by not in SORT_MODES:
        raise ValueError(f"sort_by must be one of {SORT_MODES}")
    if sort_by == SORT_MERGED:
        with_pharmacophore = True
    query = _standardize_mol(str(smiles).strip())
    if query is None:
        raise ValueError(
            "SMILES를 해석할 수 없습니다. 유효한 분자 구조인지 확인하세요."
        )
    if query.GetNumHeavyAtoms() > MAX_QUERY_HEAVY_ATOMS:
        raise ValueError(
            f"입력 분자가 너무 큽니다(중원자 {query.GetNumHeavyAtoms()}개). "
            f"이 화면은 중원자 {MAX_QUERY_HEAVY_ATOMS}개 이하의 단일 저분자를 다룹니다."
        )
    query_packed = _packed(MORGAN_GENERATOR.GetFingerprint(query))
    query_key = Chem.MolToInchiKey(query) or ""
    similarity = _tanimoto(library.fingerprints, library.popcounts, query_packed)

    records: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    scanned = 0
    prepared = prepare_query(query)
    library_keys = library.frame["inchikey"].astype(str).tolist()
    # 파마코포어는 구조 판정과 독립된 두 번째 신호다. 등급을 바꾸지 않고 열로만
    # 나간다 - 합치면 두 신호가 어긋나는 것이 화면에서 사라진다.
    query_profile = pharmacophore_profile(query) if with_pharmacophore else None
    library_profiles = ensure_pharmacophore_profiles(library) if with_pharmacophore else None
    # 합친 순위를 내려면 후보 **전체**의 점수가 필요하다. 걸러진 뒤에 다시 계산하면
    # MCS를 두 번 돌리게 되고, 두 번째 호출은 이미 소진된 예산을 만나 coverage 0을
    # 받는다 - 조용히 틀린 순위가 나온다.
    #
    # 판정하지 못한 자리는 0이 아니라 NaN 이다. 0은 "구조가 전혀 다르다"는 **측정
    # 결과**인데, 판정 못 한 자리는 재지 못했다는 뜻이라 뜻이 정반대로 읽힌다.
    # 백분위 순위는 그 0을 실제 값으로 알고 최하위표를 던진다. 판정 못 하는
    # 경로가 셋이다: 예산 소진(`budget_exhausted`), MCS 실패(`mcs_failed`),
    # 그리고 연결성은 같은데 입체가 어긋난 경우(`stereochemistry_mismatch...`,
    # 이쪽은 coverage 1.0 이 들어와 반대로 최상위표를 던진다).
    # `mcs_timeout`도 넣지 않는다 - 끊긴 MCS 가 낸 값은 하한이지 측정값이 아니다.
    # `eval_alternative_criteria.py`가 개별 구조 기준에서 쓰는 규칙과 같은 규칙이다.
    coverage_all = np.full(len(library.molecules), np.nan, dtype=float)
    pharm_all = np.full(len(library.molecules), np.nan, dtype=float)
    # 극성 근접도. 구조 정보가 하나도 들어가지 않는다.
    from rdkit.Chem import Descriptors

    polarity = np.array([Descriptors.TPSA(m) for m in library.molecules], dtype=float)
    polarity_closeness = -np.abs(polarity - Descriptors.TPSA(query))
    # A bounded request must visit the same identities in the same order even
    # when the input table was shuffled. Start the MCS clock after lazy setup.
    budget = McsBudget()
    for position in sorted(range(len(library.molecules)), key=lambda i: library_keys[i]):
        score = float(similarity[position])
        # 같은 분자인지는 지문이 아니라 InChIKey로 판단한다. MORGAN_GENERATOR는
        # includeChirality=False라, Tanimoto 1.0에는 입체이성질체와 호변이성질체가
        # 함께 걸린다 - 그것들은 빼야 할 "자기 자신"이 아니라 별개의 등재 원료다.
        if exclude_self and query_key and library_keys[position] == query_key:
            continue
        retention = core_retention(query, library.molecules[position], budget, prepared)
        if judged_coverage(retention) is not None:
            coverage_all[position] = retention.coverage
        counts[retention.grade] = counts.get(retention.grade, 0) + 1
        scanned += 1
        # 파마코포어도 구조·유사도와 같이 **거르기 전에** 채운다. 아래 필터 뒤에서
        # 채우면 걸러진 후보의 칸만 0으로 남고, 합친 순위가 그 0들을 실제 값으로
        # 알고 백분위를 매긴다 - `require_core`를 켜는 것만으로 남은 후보끼리의
        # 순서가 조용히 바뀐다. 507종 비교에 질의당 0.02초라 아낄 이유도 없다.
        match = None
        if query_profile is not None and library_profiles is not None:
            match = pharmacophore_match(query_profile, library_profiles[position])
            # `usable=False`의 0.0 은 "특징이 하나도 안 겹친다"가 아니라 Gobbi
            # 지문이 비어 잴 수 없다는 뜻이다. 507종 중 39종이 그렇다.
            if match.usable:
                pharm_all[position] = match.similarity
        if score < min_similarity:
            continue
        if require_core and retention.grade not in CORE_KEPT_GRADES:
            continue
        row = library.frame.iloc[position].to_dict()
        row["similarity"] = round(score, 4)
        row["core_grade"] = retention.grade
        row["core_label_ko"] = GRADE_LABEL_KO.get(retention.grade, retention.grade)
        # 판정 못 한 행은 숫자를 내지 않는다. 등급이 "판정 불가"인데 옆 칸에
        # 0.00(또는 입체 불일치의 1.00)이 찍히면 잰 값으로 읽힌다.
        row["core_coverage"] = judged_coverage(retention)
        row["core_share"] = retention.share
        row["core_basis"] = retention.basis
        row["scaffold_match"] = retention.scaffold_match
        row["scaffold_smiles"] = retention.scaffold_smiles
        row["is_query"] = bool(query_key and row.get("inchikey") == query_key)
        row["polarity_closeness"] = round(float(polarity_closeness[position]), 4)
        if match is not None:
            row["pharm_similarity"] = match.similarity
            row["pharm_recall"] = match.recall
            row["pharm_precision"] = match.precision
            row["pharm_usable"] = match.usable
        row["_position"] = position
        records.append(row)

    kept = sum(counts.get(grade, 0) for grade in CORE_KEPT_GRADES)
    if not records:
        return AlternativeScan(_empty_alternatives(), scanned, counts, kept, len(library.frame))

    if query_profile is not None and library_profiles is not None:
        # 백분위 순위는 **후보 전체**를 상대로 매긴다. 남은 행끼리만 매기면
        # `require_core`나 유사도 문턱을 켰을 때 같은 후보의 순위가 달라진다.
        #
        # `exclude_self`로 건너뛴 자리는 0으로 남는데, 그 자리는 어차피 records에
        # 없으므로 결과에 닿지 않는다.
        merged_all = merged_rank_score(
            {"structural": coverage_all, "pharmacophore": pharm_all, "tanimoto": similarity},
            ("structural", "pharmacophore", "tanimoto"),
        )
        for row in records:
            row["merged_score"] = round(float(merged_all[row["_position"]]), 4)
    for row in records:
        row.pop("_position", None)

    frame = pd.DataFrame(records)
    if sort_by == SORT_POLARITY:
        frame = frame.sort_values(["polarity_closeness", "inchikey"], ascending=[False, True], kind="stable")
    elif sort_by == SORT_MERGED and "merged_score" in frame:
        # 구조·파마코포어·유사도의 백분위 순위 평균. 등재 원료 507종을 신고 배합목적
        # 정답표로 재면 이 순서가 세 기준 각각보다 잘 줄 세운다(AUC 0.685 대
        # 0.657/0.645/0.666, docs/ALTERNATIVE_CRITERIA_EVAL.md). 다만 그 정답표는
        # "용도가 같다"이지 "대체 가능하다"가 아니고, 향료 분류에서는 세 기준 모두
        # 크기·극성만 보는 귀무 기준보다 못하다. 그래서 기본값이 아니다.
        frame = frame.sort_values(["merged_score", "inchikey"], ascending=[False, True], kind="stable")
    else:
        frame["_grade_order"] = frame["core_grade"].map(GRADE_ORDER).fillna(len(GRADE_ORDER))
        frame = frame.sort_values(
            ["_grade_order", "similarity", "inchikey"], ascending=[True, False, True], kind="stable"
        ).drop(columns=["_grade_order"])
    frame = frame.reset_index(drop=True)
    frame.insert(0, "rank", np.arange(1, len(frame) + 1))
    return AlternativeScan(frame, scanned, counts, kept, len(library.frame))


def find_alternatives(
    smiles: str,
    library: IngredientLibrary,
    *,
    limit: int = 20,
    min_similarity: float = 0.0,
    require_core: bool = False,
    exclude_self: bool = False,
) -> pd.DataFrame:
    """핵심 골격을 유지한 순으로 정렬한 대체소재 후보 상위 `limit`건.

    유사도는 같은 등급 안에서 줄 세우는 데만 쓰고, 등급을 결정하지는 않는다.
    유사도 0.62에 곁사슬만 닮은 분자보다 유사도 0.41에 핵심 고리가 그대로인
    분자가 대체소재로서 먼저다 - 이것이 이 함수가 `find_similar`과 다른 점의
    전부다.

    `require_core`는 핵심구조가 유지된 것(identical/contains/retained)만 남긴다.
    화면의 기본값은 꺼짐이다 - 비어 있는 표는 "후보가 없다"와 "조건이 좁았다"를
    구분해 주지 않기 때문이다.

    목록만 필요한 호출자를 위한 얇은 편의 함수다. 화면과 CLI는 `scan_alternatives`를
    직접 부른다 - 몇 종을 대조했고 그중 몇 건이 유지 등급이었는지를 함께 말해야
    하는데, 잘라낸 목록만으로는 그 수를 셀 수 없기 때문이다.
    """
    return scan_alternatives(
        smiles, library, min_similarity=min_similarity,
        require_core=require_core, exclude_self=exclude_self,
    ).frame.head(limit).reset_index(drop=True)


def signal_disagreement(frame: pd.DataFrame) -> dict[str, Any]:
    """두 신호가 얼마나 어긋나는지. 어긋남 자체가 읽는 사람에게 필요한 정보다.

    구조 순위와 파마코포어 순위의 Spearman 상관을 낸다. 낮으면 "둘 중 하나가
    틀렸다"가 아니라 "두 질문의 답이 다르다"는 뜻이다 - 원자를 유지하는 것과
    상호작용 특징을 유지하는 것은 같은 일이 아니다.
    """
    if frame.empty or "pharm_similarity" not in frame:
        return {"available": False}
    structural = frame["core_coverage"].rank(ascending=False, method="average")
    pharmacophore = frame["pharm_similarity"].rank(ascending=False, method="average")
    if structural.nunique() < 2 or pharmacophore.nunique() < 2:
        return {"available": False}
    rho = float(structural.corr(pharmacophore, method="spearman"))
    # The caller can display the table sorted by polarity or merged score.
    # Its first row is then not the structural winner.
    top_structural = frame.sort_values(
        ["core_coverage", "inchikey"], ascending=[False, True], kind="stable"
    ).iloc[0]
    top_pharm = frame.sort_values(
        ["pharm_similarity", "inchikey"], ascending=[False, True], kind="stable"
    ).iloc[0]
    return {
        "available": True,
        "spearman": None if rho != rho else round(rho, 3),
        "top_structural": str(top_structural.get("inci_name") or ""),
        "top_pharmacophore": str(top_pharm.get("inci_name") or ""),
        "same_top": str(top_structural.get("inchikey")) == str(top_pharm.get("inchikey")),
    }


def _empty_alternatives() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "rank", "inci_name", "synonyms", "cas", "functions", "canonical_smiles", "inchikey",
            "skeleton", "similarity", "core_grade", "core_label_ko", "core_coverage",
            "core_share", "core_basis", "scaffold_match", "scaffold_smiles", "is_query",
        ]
    )


def annotate_core_retention(frame: pd.DataFrame, query_smiles: str, smiles_column: str = "canonical_smiles") -> pd.DataFrame:
    """이미 뽑아 놓은 표에 골격 판정 열을 붙인다.

    측정 라이브러리 쪽 결과에도 같은 판정이 필요하다. 그쪽은 유사도로 정렬되고
    이쪽은 골격으로 정렬되지만, 두 표가 같은 말을 다르게 쓰면 읽는 사람이
    비교할 수 없다.
    """
    if frame.empty:
        return frame.assign(core_grade=pd.Series(dtype=str), core_label_ko=pd.Series(dtype=str),
                            core_coverage=pd.Series(dtype=float), core_share=pd.Series(dtype=float),
                            core_basis=pd.Series(dtype=str),
                            scaffold_match=pd.Series(dtype=bool))
    query = _standardize_mol(str(query_smiles).strip())
    budget = McsBudget()
    prepared = prepare_query(query)
    candidates = [Chem.MolFromSmiles(str(value or "")) for value in frame[smiles_column]]
    order = sorted(range(len(candidates)), key=lambda i: Chem.MolToSmiles(candidates[i]) if candidates[i] is not None else "")
    evaluated = {i: core_retention(query, candidates[i], budget, prepared) for i in order}
    grades, labels, coverages, shares, bases, scaffolds = [], [], [], [], [], []
    for i in range(len(frame)):
        retention = evaluated[i]
        grades.append(retention.grade)
        labels.append(GRADE_LABEL_KO.get(retention.grade, retention.grade))
        coverages.append(retention.coverage)
        shares.append(retention.share)
        bases.append(retention.basis)
        scaffolds.append(retention.scaffold_match)
    out = frame.copy()
    out["core_grade"] = grades
    out["core_label_ko"] = labels
    out["core_coverage"] = coverages
    out["core_share"] = shares
    out["core_basis"] = bases
    out["scaffold_match"] = scaffolds
    return out


def attach_measured_evidence(frame: pd.DataFrame, index: Any, query_smiles: str = "") -> pd.DataFrame:
    """대체소재 후보 각각에 대해, 그 분자 자체가 측정된 적이 있는지 붙인다.

    붙지 않는 것이 정상이다. 등재 원료 507종 중 이 라이브러리에 기록이 있는 것은
    51종뿐이고, 그 사실 자체가 읽는 사람에게 필요한 정보다 - 핵심구조는 맞는데
    측정 기록을 찾지 못한 후보와, 측정했는데 문턱 아래였던 후보는 완전히 다른
    결정을 부른다. 여기서 "없다"는 이 라이브러리에서 못 찾았다는 뜻이지, 세상
    어디에도 측정이 없다는 뜻이 아니다.

    `query_smiles`를 주면 한 가지를 더 붙인다: 그 후보가 **입력 화합물과 같은
    표적에서** 측정된 적이 있는지다. 핵심구조를 유지하는 이유가 결국 활성을
    유지하려는 것이므로, 같은 단백질에서 실제로 측정된 기록이 있다는 사실은 이
    화면이 낼 수 있는 가장 결정에 가까운 근거다. 없다고 해서 활성이 없다는 뜻은
    아니고, 아무도 그 조합을 재지 않았다는 뜻이다.
    """
    from similar_compounds import evidence_for_inchikeys

    if frame.empty:
        return frame
    keys = [str(k) for k in frame["inchikey"].tolist()]
    query_targets: set[str] = set()
    if query_smiles:
        try:
            query_key = Chem.MolToInchiKey(_standardize_mol(str(query_smiles).strip())) or ""
        except (ValueError, RuntimeError):
            query_key = ""
        if query_key:
            keys = keys + [query_key]
            query_targets = set(
                evidence_for_inchikeys(index, [query_key]).get(query_key, {}).get("targets", [])
            )
    found = evidence_for_inchikeys(index, keys)

    out = frame.copy()
    out["measured_target_count"] = [found.get(k, {}).get("target_count") for k in out["inchikey"]]
    out["measured_best_pactivity"] = [found.get(k, {}).get("best_pactivity") for k in out["inchikey"]]
    out["measured_evidence"] = [found.get(k, {}).get("evidence", "not_measured") for k in out["inchikey"]]
    out["measured_top_targets"] = [found.get(k, {}).get("top_targets", []) for k in out["inchikey"]]
    # 이 분자 자신의 측정값인지, 연결성만 같은 분자의 것인지. 전체 키를 먼저 찾으므로
    # 자기 값이 인덱스에 있으면 반드시 그것이 온다.
    out["measured_match"] = [found.get(k, {}).get("match", "") for k in out["inchikey"]]
    out["shared_targets"] = [
        sorted(query_targets.intersection(found.get(k, {}).get("targets", []))) if query_targets else []
        for k in out["inchikey"]
    ]
    # 입력 화합물 자체에 측정 기록이 없으면 교집합은 언제나 비어 있다. 그것을
    # "겹치는 표적이 없다"로 읽으면 안 되므로 구분해 둔다.
    out.attrs["query_measured"] = bool(query_targets)
    return out


def scan_summary(scan: AlternativeScan) -> str:
    """"다 봤는데 없다"를 "안 봤다"와 구분해 주는 한 줄.

    계산 상한에 걸려 판정하지 못한 후보가 있으면 "전부 대조했다"고 쓰지 않는다.
    안 본 것을 봤다고 말하는 순간 이 문장이 근거로 쓰일 수 없게 된다.
    """
    unjudged = scan.grade_counts.get("unknown", 0)
    judged = scan.scanned - unjudged
    scope = (
        f"{scan.scanned}종을 전부 대조해"
        if not unjudged
        else f"{scan.scanned}종 중 {judged}종을 판정해(나머지 {unjudged}종은 입체화학 또는 계산 제한으로 판정하지 못했습니다)"
    )
    if scan.kept:
        return f"{scope} 핵심구조가 유지된 후보 {scan.kept}건을 찾았습니다."
    return (
        f"{scope} 핵심구조가 유지된 후보를 찾지 못했습니다. "
        "아래는 그중 가장 가까운 것들이며, 등급이 곧 그 사실입니다."
    )


def library_note(library: IngredientLibrary) -> str:
    """모집단이 무엇인지 한 줄로. 후보가 적을 때 도구를 의심하지 않도록."""
    if library.registered_entries:
        resolved = library.resolved_entries or len(library.frame)
        return (
            f"CosIng 등재 {library.registered_entries:,}건 중 단일 저분자로 구조가 확인된 것은 "
            f"{resolved:,}건이고, 같은 구조에 붙은 이름을 합치면 {len(library.frame):,}종입니다. "
            "나머지는 추출물·혼합물·고분자라 구조 비교의 대상이 아닙니다."
        )
    return f"CosIng 원료 {len(library.frame):,}종이 후보 모집단입니다."


def screen_batch(
    queries: list[tuple[str, str]],
    library: IngredientLibrary,
    *,
    limit: int,
    min_similarity: float,
    require_core: bool,
    exclude_self: bool,
    index: Any = None,
) -> pd.DataFrame:
    """출발 화합물 여러 개를 한 번에. 라이브러리와 인덱스는 한 번만 연다.

    후보 목록을 화합물 하나씩 손으로 뽑는 것과 결과가 같아야 하므로, 여기서도
    화면·CLI와 같은 `scan_alternatives`를 부른다. 다른 점은 어떤 질의였는지를
    남기는 두 열과, 후보가 하나도 없거나 읽지 못한 질의도 한 줄로 남긴다는 것뿐이다 -
    결과 CSV에서 빠진 화합물이 "안 돌았다"인지 "후보가 없었다"인지 구분되지 않으면,
    그 표는 다시 확인해야 하는 표가 된다.
    """
    frames: list[pd.DataFrame] = []
    for name, smiles in queries:
        try:
            scan = scan_alternatives(
                smiles, library, min_similarity=min_similarity,
                require_core=require_core, exclude_self=exclude_self,
            )
        except (ValueError, RuntimeError) as exc:
            frames.append(pd.DataFrame([{
                "query_name": name, "query_smiles": smiles, "status": f"읽지 못함: {exc}",
                "core_kept_in_library": 0,
            }]))
            continue
        frame = scan.frame.head(limit).reset_index(drop=True)
        if index is not None and not frame.empty:
            frame = attach_measured_evidence(frame, index, smiles)
        if frame.empty:
            frames.append(pd.DataFrame([{
                "query_name": name, "query_smiles": smiles, "status": "조건에 맞는 후보 없음",
                "core_kept_in_library": scan.kept,
            }]))
            continue
        frame = frame.copy()
        frame.insert(0, "query_smiles", smiles)
        frame.insert(0, "query_name", name)
        # 이 질의에 대해 라이브러리 전체에서 핵심구조가 유지된 것이 몇 건이었는지.
        # 상위 limit만 남은 CSV에서는 이 수가 없으면 "유지 후보가 정말 없었는지"를
        # 되짚을 수 없다.
        frame["core_kept_in_library"] = scan.kept
        frame["status"] = "ok"
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def read_query_csv(path: Path) -> list[tuple[str, str]]:
    """`smiles` 열은 필수, `name`(또는 `inci_name`) 열은 있으면 쓴다."""
    table = pd.read_csv(path)
    columns = {c.lower().strip(): c for c in table.columns}
    smiles_column = columns.get("smiles")
    if smiles_column is None:
        raise SystemExit(f"{path}: `smiles` 열이 필요합니다. 있는 열: {list(table.columns)}")
    name_column = columns.get("name") or columns.get("inci_name") or columns.get("compound")
    queries: list[tuple[str, str]] = []
    for row in table.to_dict("records"):
        smiles = str(row.get(smiles_column) or "").strip()
        if not smiles:
            continue
        name = str(row.get(name_column) or "").strip() if name_column else ""
        queries.append((name or smiles, smiles))
    if not queries:
        raise SystemExit(f"{path}: 읽을 수 있는 SMILES가 없습니다.")
    return queries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smiles", help="활성이 알려진 출발 화합물")
    parser.add_argument(
        "--input-csv",
        type=Path,
        help="여러 화합물을 한 번에. `smiles` 열이 필요하고 `name` 열은 선택입니다.",
    )
    parser.add_argument("--cosing", type=Path, default=DEFAULT_COSING)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.0,
        help="추가 필터. 기본값 0은 등재 원료 전체를 핵심구조 기준으로 줄 세웁니다.",
    )
    parser.add_argument("--require-core", action="store_true", help="핵심구조가 유지된 것만")
    parser.add_argument("--exclude-self", action="store_true")
    parser.add_argument("--with-evidence", action="store_true", help="활성 측정 라이브러리와 대조")
    parser.add_argument("--out-csv", type=Path)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be >= 1")
    if bool(args.smiles) == bool(args.input_csv):
        parser.error("--smiles 또는 --input-csv 중 하나를 주세요.")

    library = load_ingredient_library(args.cosing)

    if args.input_csv:
        if not args.out_csv:
            parser.error("--input-csv 를 쓸 때는 --out-csv 로 저장 위치를 주세요.")
        index = None
        if args.with_evidence:
            from similar_compounds import load_index

            index = load_index()
        queries = read_query_csv(args.input_csv)
        batch = screen_batch(
            queries, library, limit=args.limit, min_similarity=args.min_similarity,
            require_core=args.require_core, exclude_self=args.exclude_self, index=index,
        )
        batch.to_csv(args.out_csv, index=False)
        status = batch["status"] if "status" in batch else pd.Series(dtype=str)
        rows = int((status == "ok").sum())
        kept_column = batch.get("core_kept_in_library")
        with_core = 0
        if kept_column is not None and rows:
            with_core = int(batch.loc[status == "ok"].groupby("query_name")["core_kept_in_library"].max().gt(0).sum())
        print(library_note(library))
        print(
            f"질의 {len(queries)}개 · 후보 행 {rows}개 · 그중 핵심구조가 유지된 후보를 가진 질의 {with_core}개. "
            "읽지 못한 질의는 결과 CSV에 사유가 한 줄씩 남아 있습니다."
        )
        print(f"wrote {args.out_csv}")
        return 0

    scan = scan_alternatives(
        args.smiles,
        library,
        min_similarity=args.min_similarity,
        require_core=args.require_core,
        exclude_self=args.exclude_self,
    )
    frame = scan.frame.head(args.limit).reset_index(drop=True)
    if args.with_evidence and not frame.empty:
        from similar_compounds import load_index

        frame = attach_measured_evidence(frame, load_index(), args.smiles)

    print(library_note(library))
    print(scan_summary(scan))
    if frame.empty:
        return 0
    if args.out_csv:
        frame.to_csv(args.out_csv, index=False)
        print(f"wrote {args.out_csv}")

    print(f"{'순위':>4} {'유사도':>6} {'핵심구조':18} {'유지율':>6} {'비중':>5} {'고리':>4}  INCI")
    for row in frame.itertuples(index=False):
        name = row.inci_name[:56]
        mark = "  ← 입력과 같음" if row.is_query else ""
        ring = "일치" if row.scaffold_match else "-"
        print(
            f"{row.rank:>4} {row.similarity:>6.3f} {row.core_label_ko:18} "
            f"{row.core_coverage:>6.2f} {row.core_share:>5.2f} {ring:>4}  {name}{mark}"
        )
        if getattr(row, "measured_evidence", "not_measured") != "not_measured":
            targets = ", ".join(row.measured_top_targets[:3])
            count = int(row.measured_target_count or 0)
            print(f"       └ 측정됨: 표적 {count}개, 최대 pAct {row.measured_best_pactivity} ({targets})")
            if getattr(row, "measured_match", "exact") == "connectivity":
                print("         └ 주의: 연결성만 같은 분자(입체·염 형태가 다름)의 측정값입니다")
            shared = list(getattr(row, "shared_targets", []) or [])
            if shared:
                print(f"         └ 입력과 같은 표적에서도 측정됨: {', '.join(shared[:5])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# --- 3D 파마코포어 --------------------------------------------------------------
#
# 2D 신호는 특징의 **개수**와 그래프상의 거리를 본다. 3D는 특징이 공간에서 실제로
# 같은 자리에 오는지를 본다. 단백질이 보는 것은 후자이므로, 2D에서 겹쳐 보이던 두
# 분자가 3D에서 갈라지는 일이 생긴다.
#
# 계산은 Stage 5.6b(`discover_substitutes.py`)의 것을 그대로 쓴다. ETKDGv3로 컨포머를
# 만들고 MMFF94s로 다듬은 뒤, 질의와 후보의 MCS 원자맵으로 정렬하고, 정렬된 상태에서
# 특징족의 재현율과 특징 위치의 RMSD를 낸다.
#
# 라이브러리 전수에는 쓸 수 없다. 실측으로 쌍당 0.22-1.03초(컨포머 5-20개)라, 507종에
# 돌리면 114-523초다. 그래서 2D 순위 상위 몇 건만 다시 채점한다 - 5.6b도 같은 이유로
# `max_3d_candidates`를 128로 두고 있다.
#
# 5.6b의 통과 기준(재현율 >= 0.80, RMSD <= 1.50 A)은 2D 때와 마찬가지로 가져오지
# 않는다. 여기서는 판정하지 않고 값만 보여 준다.
MAX_3D_CANDIDATES = 25
DEFAULT_3D_CONFORMERS = 10
# 컨포머 생성은 난수를 쓴다. 씨앗을 고정하지 않으면 같은 질의가 실행마다 다른
# 3D 값을 내고, 그러면 화면의 숫자를 인용할 수 없다.
DEFAULT_3D_SEED = 7
# 한 요청이 쓸 수 있는 3D 총 시간. 컨포머가 잘 생기지 않는 분자에서 한 쌍이 오래
# 끌 수 있어, 건당 상한만으로는 요청 전체가 묶인다.
#
# 이 마감은 `annotate_three_d` 가 질의 앙상블을 만들기 **전에** 시작하고,
# 5.6b 안쪽의 컨포머 쌍 루프까지 내려간다. 그래도 정확히 지켜지지는 않는다:
# 후보 컨포머 임베딩(ETKDG+MMFF)은 RDKit 호출 하나라 도중에 멈출 수 없고,
# 큰 분자에서 6초 남짓 걸린다. 그래서 실제 상한은 "30초 + 마지막으로 시작한
# 후보의 임베딩 시간"이다. 예전에는 마감이 쌍의 **진입**만 막아서, 0.6초 남은
# 예산으로 시작한 쌍이 13.8초를 더 쓰고도 status='ok' 로 나왔다 - 상한을
# 넘긴 것이 아니라 상한이 없었던 것에 가깝다.
MCS_3D_BUDGET_SECONDS = 30.0


@dataclass(frozen=True)
class ThreeDMatch:
    """정렬된 3차원에서 질의의 특징이 후보의 같은 자리에 오는가."""

    status: str                 # ok | unavailable | budget_exhausted
    feature_recall: float | None
    feature_rmsd: float | None  # 옹스트롬. 작을수록 같은 자리
    matched_features: int
    query_features: int
    query_conformers: int
    candidate_conformers: int
    basis: str

    def public(self) -> dict[str, Any]:
        return asdict(self)


def _three_d_backend():
    global _THREE_D_BACKEND
    if _THREE_D_BACKEND is None:
        from pathlib import Path as _Path

        from rdkit import RDConfig
        from rdkit.Chem import ChemicalFeatures

        from discover_substitutes import (
            _conformer_ensemble,
            _parent_feature_records_by_conformer,
            _pharmacophore_3d_comparison,
        )

        factory = ChemicalFeatures.BuildFeatureFactory(
            str(_Path(RDConfig.RDDataDir) / "BaseFeatures.fdef")
        )
        _THREE_D_BACKEND = (
            factory, _conformer_ensemble, _parent_feature_records_by_conformer,
            _pharmacophore_3d_comparison,
        )
    return _THREE_D_BACKEND


_THREE_D_BACKEND: Any = None


def prepare_query_3d(
    query: Chem.Mol, *, seed: int = DEFAULT_3D_SEED, conformers: int = DEFAULT_3D_CONFORMERS
):
    """질의 쪽 컨포머와 특징 좌표. 후보마다 다시 만들 이유가 없다."""
    factory, ensemble_of, features_of, _ = _three_d_backend()
    ensemble = ensemble_of(query, seed=seed, max_conformers=conformers)
    return factory, ensemble, features_of(ensemble, factory)


def three_d_match(
    query: Chem.Mol,
    candidate: Chem.Mol,
    prepared_3d: Any,
    *,
    seed: int = DEFAULT_3D_SEED,
    conformers: int = DEFAULT_3D_CONFORMERS,
    budget: "McsBudget | None" = None,
) -> ThreeDMatch:
    """한 쌍의 3D 파마코포어 비교. 5.6b의 계산을 그대로 부른다."""
    factory, ensemble, cached = prepared_3d
    if budget is not None and budget.exhausted():
        return ThreeDMatch(
            "budget_exhausted", None, None, 0, 0, len(ensemble[1]), 0,
            "etkdg_v3_mmff_ligand_feature_alignment",
        )
    _, _, _, compare = _three_d_backend()
    result = compare(
        query, candidate, factory, ensemble,
        seed=seed, max_conformers=conformers, parent_features_by_conformer=cached,
        # 마감을 안쪽까지 내려보낸다. 진입만 막으면 상한이 지켜지지 않는다:
        # 한 쌍이 후보 임베딩과 컨포머 100쌍 정렬을 끝까지 돌아, 실측으로
        # 0.6초 남은 예산에서 시작한 쌍이 13.8초를 더 썼다.
        deadline=None if budget is None else budget.deadline,
    )
    recall = result.get("feature_family_recall")
    rmsd = result.get("feature_distance_rmsd")
    raw_status = str(result.get("status") or "")
    if raw_status == "budget_exhausted":
        return ThreeDMatch(
            "budget_exhausted", None, None, 0, 0, len(ensemble[1]), 0,
            "etkdg_v3_mmff_ligand_feature_alignment",
        )
    return ThreeDMatch(
        # 입력 분자 쪽 실패는 후보 쪽 실패와 다른 말이다. 하나로 뭉치면 화면이
        # "이 후보를 판정하지 못했다"고 25번 말하는데, 실제로는 후보를 하나도
        # 볼 수 없었던 것이다.
        status=("parent_unavailable" if raw_status == "parent_unavailable"
                else "recall_only" if raw_status == "recall_only"
                else "ok" if raw_status != "unavailable" else "unavailable"),
        feature_recall=None if recall is None else round(float(recall), 3),
        feature_rmsd=None if rmsd is None else round(float(rmsd), 3),
        matched_features=int(result.get("matched_feature_count") or 0),
        query_features=int(result.get("parent_feature_count") or 0),
        query_conformers=int(result.get("parent_conformer_count") or 0),
        candidate_conformers=int(result.get("candidate_conformer_count") or 0),
        basis=str(result.get("basis") or ""),
    )


def annotate_three_d(
    frame: pd.DataFrame,
    query_smiles: str,
    *,
    limit: int = MAX_3D_CANDIDATES,
    conformers: int = DEFAULT_3D_CONFORMERS,
    seed: int = DEFAULT_3D_SEED,
    smiles_column: str = "canonical_smiles",
) -> pd.DataFrame:
    """이미 줄 세운 표의 **상위 몇 건만** 3D로 다시 본다.

    전수에 쓸 수 없어서 상위만 보는 것이고, 그 사실이 표에 남아야 한다. 재채점하지
    않은 행은 `not_rescored`이지 "3D에서 나빴다"가 아니다.
    """
    if frame.empty:
        # 빈 표에도 열은 붙인다. `annotate_core_retention`과 같은 약속이어야 호출자가
        # 열의 존재를 따로 확인하지 않는다.
        return frame.assign(
            three_d_status=pd.Series(dtype=str),
            three_d_recall=pd.Series(dtype=object),
            three_d_rmsd=pd.Series(dtype=object),
        )
    query = _standardize_mol(str(query_smiles).strip())
    # 예산 시계를 질의 앙상블 **전에** 시작한다. 뒤에 두면 이 준비 비용이 예산
    # 밖에 놓이는데, 실측으로 시클로스포린 A(중원자 85개, 상한 200개 안)의
    # 준비만 13.9초였다.
    budget = McsBudget(MCS_3D_BUDGET_SECONDS)
    prepared = prepare_query_3d(query, seed=seed, conformers=conformers)
    _, query_ensemble, _ = prepared
    if query_ensemble[0] is None or not query_ensemble[1]:
        # 결론이 후보를 보기 전에 이미 났다. 후보마다 컨포머를 만들면 정해진
        # 결과에 25행치 비용을 낸다 - 실측 31초, 전부 빈 칸.
        return frame.assign(
            three_d_status=["parent_unavailable"] * len(frame),
            three_d_recall=pd.Series([None] * len(frame), dtype=object, index=frame.index),
            three_d_rmsd=pd.Series([None] * len(frame), dtype=object, index=frame.index),
        )

    statuses: list[str] = []
    recalls: list[float | None] = []
    rmsds: list[float | None] = []
    for position, value in enumerate(frame[smiles_column]):
        if position >= limit:
            statuses.append("not_rescored"); recalls.append(None); rmsds.append(None)
            continue
        candidate = Chem.MolFromSmiles(str(value or ""))
        if candidate is None:
            statuses.append("unavailable"); recalls.append(None); rmsds.append(None)
            continue
        match = three_d_match(query, candidate, prepared, seed=seed,
                              conformers=conformers, budget=budget)
        statuses.append(match.status)
        recalls.append(match.feature_recall)
        rmsds.append(match.feature_rmsd)

    out = frame.copy()
    out["three_d_status"] = statuses
    # object dtype로 넣어야 None이 None으로 남는다. 그냥 대입하면 pandas가 float
    # 열로 만들면서 None을 NaN으로 바꾸고, NaN은 참이라 `value or 기본값`을 그대로
    # 통과한다. 이 저장소는 이 함정에 이미 두 번 물렸다(측정값 칸의 숫자와 문자열).
    out["three_d_recall"] = pd.Series(recalls, index=out.index, dtype=object)
    out["three_d_rmsd"] = pd.Series(rmsds, index=out.index, dtype=object)
    return out
