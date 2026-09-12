#!/usr/bin/env python3
"""평가 도구가 재는 것을 실제로 재는지.

이 도구가 낸 숫자는 "구조 기준과 파마코포어 기준 중 무엇을 위에 둘 것인가"를 정하는
데 쓰인다. 도구가 조용히 틀리면 그 결정도 조용히 틀리고, 화면을 봐서는 알 수 없다.
그래서 손으로 계산되는 경우를 여기 박아 둔다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_alternative_criteria import (  # noqa: E402
    BASELINE_KEY,
    BASELINE_KEYS,
    CONTROL_KEY,
    CRITERIA,
    large_classes,
    METHOD_KEYS,
    SINGLE_METHOD_KEYS,
    functions_ground_truth,
    roc_auc,
    stratified,
    summarise,
    wilcoxon_signed_rank,
)

COSING = ROOT / "data" / "cosing" / "cosing.parquet"


# --------------------------------------------------------------- AUC


def test_perfect_separation_is_one_and_perfect_inversion_is_zero() -> None:
    assert roc_auc(np.array([3.0, 2.0, 1.0, 0.0]), np.array([1, 1, 0, 0])) == 1.0
    assert roc_auc(np.array([0.0, 1.0, 2.0, 3.0]), np.array([1, 1, 0, 0])) == 0.0


def test_all_ties_are_exactly_chance() -> None:
    """동점을 평균 순위로 세지 않으면 입력 순서가 답을 만든다."""
    assert roc_auc(np.array([1.0] * 4), np.array([1, 1, 0, 0])) == 0.5
    # 같은 점수인데 양성이 뒤에 오는 배열. 순진하게 세면 0.5가 아니게 된다.
    assert roc_auc(np.array([1.0] * 4), np.array([0, 0, 1, 1])) == 0.5


def test_a_hand_computed_case_with_mixed_ties() -> None:
    """점수 [2,1,1,0], 양성 = 0번과 2번.

    순위는 4, 2.5, 2.5, 1. U = (4 + 2.5) - 2*3/2 = 3.5, 양성 2 x 음성 2 = 4.
    """
    assert roc_auc(np.array([2.0, 1.0, 1.0, 0.0]), np.array([1, 0, 1, 0])) == 0.875


def test_auc_is_undefined_without_both_classes() -> None:
    """정의되지 않은 것을 0.5로 채우면 그 질의가 조용히 평균을 끌어당긴다."""
    assert roc_auc(np.array([1.0, 2.0]), np.array([0, 0])) is None
    assert roc_auc(np.array([1.0, 2.0]), np.array([1, 1])) is None


def test_auc_does_not_depend_on_input_order() -> None:
    rng = np.random.default_rng(7)
    scores = rng.random(60)
    positive = rng.random(60) < 0.3
    first = roc_auc(scores, positive)
    order = rng.permutation(60)
    assert first == pytest.approx(roc_auc(scores[order], positive[order]))


# --------------------------------------------------- Wilcoxon


def test_a_one_sided_difference_is_detected() -> None:
    result = wilcoxon_signed_rank(np.array([0.9] * 25), np.array([0.5] * 25))
    assert result["n"] == 25
    assert result["p_value"] < 0.001
    assert result["approximation_reliable"] is True


def test_identical_inputs_leave_no_pairs() -> None:
    """모든 차이가 0이면 부호순위는 아무것도 말하지 않는다. 0을 p=1로 꾸미지 않는다."""
    result = wilcoxon_signed_rank(np.array([0.5] * 10), np.array([0.5] * 10))
    assert result["n"] == 0
    assert result["p_value"] is None


def test_small_samples_are_flagged_as_unreliable() -> None:
    """정규근사는 20쌍 아래에서 거칠다. 그 사실이 결과와 함께 나가야 한다."""
    result = wilcoxon_signed_rank(np.arange(6.0), np.arange(6.0) + 1.0)
    assert result["n"] == 6
    assert result["approximation_reliable"] is False


def test_the_test_is_symmetric_in_its_arguments() -> None:
    a = np.array([0.7, 0.2, 0.9, 0.4, 0.55, 0.61, 0.33])
    b = np.array([0.5, 0.6, 0.4, 0.45, 0.50, 0.70, 0.31])
    assert wilcoxon_signed_rank(a, b)["p_value"] == pytest.approx(
        wilcoxon_signed_rank(b, a)["p_value"]
    )


def test_wilcoxon_variance_accounts_for_tied_magnitudes() -> None:
    from scipy.stats import wilcoxon
    a = np.array([1.0] * 14 + [-1.0] * 6 + [2.0] * 8 + [-2.0] * 2)
    expected = wilcoxon(a, method="approx", correction=True).pvalue
    assert wilcoxon_signed_rank(a, np.zeros_like(a))["p_value"] == pytest.approx(expected)


def test_paired_auc_uses_the_same_candidates_for_every_method(monkeypatch) -> None:
    import eval_alternative_criteria as module
    from types import SimpleNamespace

    scores = {c.key: np.array([0.0, 0.9, 0.8, 0.1]) for c in CRITERIA}
    scores["pharmacophore"] = np.array([0.0, np.nan, 0.1, 0.9])
    monkeypatch.setattr(module, "score_all_criteria", lambda *_a, **_k: scores)
    monkeypatch.setattr(module, "_WORKER", {
        "library": SimpleNamespace(molecules=[None] * 4),
        "truth": module.GroundTruth("t", "d", {0: {1, 2}}, {0: ("X",)}, "c"),
        "names": ["query", "a", "b", "c"],
    })
    _, row = module._score_one_query((0, False))
    assert row["paired_2d_pool"] == 2
    assert row["paired_2d__structural"] == 1.0
    assert row["paired_2d__pharmacophore"] == 0.0


def test_baseline_comparison_uses_stronger_null_and_holm_correction() -> None:
    from eval_alternative_criteria import GroundTruth
    frame = pd.DataFrame({
        "query_index": list(range(25)), "query": ["q"] * 25,
        "strata": ["X"] * 25, "positives": [2] * 25, "pool": [5] * 25,
        **{c.key: [0.6] * 25 for c in CRITERIA},
        "size_baseline": [0.4] * 25, "polarity_baseline": [0.9] * 25,
    })
    for c in CRITERIA:
        frame[f"paired_2d__{c.key}"] = frame[c.key]
    summary = summarise(frame, GroundTruth("t", "d", {}, {}, "c"))
    test = summary["vs_baseline"]["structural"]
    assert test["mean_difference"] == -0.3
    assert test["query_win_fraction"] == 0.0
    assert test["p_value_adjusted"] >= test["p_value"]
    assert summary["multiplicity"]["family_size"] > 1


def test_empty_stratified_result_and_undefined_numbers_are_supported() -> None:
    from eval_alternative_criteria import _display_number
    frame = pd.DataFrame(columns=["strata", "query_index", *[c.key for c in CRITERIA]])
    assert stratified(frame).empty
    assert _display_number(None, ".4f") == "-"


def test_stratified_uses_common_candidate_auc_and_partial_binding_is_unverified() -> None:
    from eval_alternative_criteria import CRITERIA, _paired_population

    frame = pd.DataFrame({"query_index": [0], "strata": ["X"]})
    for criterion in CRITERIA:
        frame[criterion.key] = 0.9
        frame[f"paired_2d__{criterion.key}"] = 0.6
    table = stratified(frame)
    assert table.iloc[0]["structural"] == 0.6
    assert bool(table.iloc[0]["candidate_pool_verified"])
    incomplete = frame.drop(columns=["paired_2d__pharmacophore"])
    assert _paired_population(incomplete)[1] is False


# ------------------------------------------------ 기준선과 대조


def test_the_criteria_include_a_null_baseline_and_a_random_control() -> None:
    """세 방법만 있으면 AUC 0.65를 "잘 맞힌다"로 읽게 된다.

    신고 목적이 같은 원료들은 크기부터 비슷해서, 분자량 차이만으로도 0.61이 나온다.
    비교 대상은 우연(0.5)이 아니라 그 값이다.
    """
    keys = {c.key for c in CRITERIA}
    assert BASELINE_KEY in keys
    assert CONTROL_KEY in keys
    assert set(METHOD_KEYS) < keys
    assert BASELINE_KEY not in METHOD_KEYS
    assert CONTROL_KEY not in METHOD_KEYS


def test_the_summary_asks_the_baseline_question_before_the_method_question() -> None:
    frame = pd.DataFrame(
        {
            "query_index": [0, 1, 2],
            "query": ["a", "b", "c"],
            "strata": ["X", "X", "X"],
            "positives": [3, 3, 3],
            "pool": [10, 10, 10],
            "structural": [0.8, 0.7, 0.9],
            "pharmacophore": [0.6, 0.5, 0.7],
            "tanimoto": [0.75, 0.65, 0.85],
            "merged_rank_average": [0.78, 0.68, 0.88],
            "size_baseline": [0.5, 0.5, 0.5],
            "polarity_baseline": [0.5, 0.5, 0.5],
            "random_control": [0.5, 0.5, 0.5],
        }
    )
    from eval_alternative_criteria import GroundTruth

    summary = summarise(frame, GroundTruth("t", "d", {}, {}, "c"))
    assert "vs_baseline" in summary
    for key in METHOD_KEYS:
        assert key in summary["vs_baseline"]
        assert "query_win_fraction" in summary["vs_baseline"][key]
    assert summary["control_sanity"]["healthy"] is True


def test_a_broken_control_is_reported_as_broken() -> None:
    """난수 대조가 0.5를 벗어나면 평가 전체를 믿을 수 없다. 조용히 넘어가면 안 된다."""
    from eval_alternative_criteria import GroundTruth

    frame = pd.DataFrame(
        {
            "query_index": [0, 1],
            "query": ["a", "b"], "strata": ["X", "X"], "positives": [2, 2], "pool": [9, 9],
            "structural": [0.8, 0.8], "pharmacophore": [0.6, 0.6], "tanimoto": [0.7, 0.7],
            "merged_rank_average": [0.75, 0.75],
            "size_baseline": [0.5, 0.5], "polarity_baseline": [0.5, 0.5],
            "random_control": [0.9, 0.9],
        }
    )
    summary = summarise(frame, GroundTruth("t", "d", {}, {}, "c"))
    assert summary["control_sanity"]["healthy"] is False


def test_the_stratified_table_says_whether_structure_beat_the_baseline() -> None:
    """분류마다 답이 뒤집힌다. 향료에서는 크기만 보는 편이 낫다."""
    frame = pd.DataFrame(
        {
            "query_index": [0, 1],
            "query": ["a", "b"], "strata": ["PERFUMING", "HAIR DYEING"],
            "positives": [5, 5], "pool": [20, 20],
            "structural": [0.59, 0.79], "pharmacophore": [0.53, 0.84], "tanimoto": [0.63, 0.75],
            "merged_rank_average": [0.64, 0.82],
            "size_baseline": [0.69, 0.50], "polarity_baseline": [0.84, 0.66],
            "random_control": [0.5, 0.5],
        }
    )
    table = stratified(frame).set_index("stratum")
    assert table.loc["PERFUMING", "best_minus_null"] < 0
    assert table.loc["HAIR DYEING", "best_minus_null"] > 0
    assert CONTROL_KEY not in table.columns


# ------------------------------------------------------- 정답표


@pytest.fixture(scope="module")
def library():
    if not COSING.is_file():
        pytest.skip("CosIng 원료 표가 없습니다")
    from alternative_ingredients import load_ingredient_library

    return load_ingredient_library()


def test_a_query_is_never_its_own_positive(library) -> None:
    """자기 자신을 찾아오면 어떤 기준이든 완벽해 보인다."""
    truth = functions_ground_truth(library)
    for index, positives in truth.positives.items():
        assert index not in positives


def test_excluding_the_large_classes_actually_shrinks_the_set(library) -> None:
    """507종 중 272종이 PERFUMING·HAIR DYEING·FRAGRANCE다.

    빼지 않으면 어떤 결과든 그 세 분류의 결과가 된다.
    """
    full = functions_ground_truth(library)
    small = functions_ground_truth(library, exclude_large=True)
    assert len(small.positives) < len(full.positives)
    # 대형 분류는 모집단에서 뽑는다. 이름을 박아 두면 라이브러리가 커질 때
    # 목록이 낡고, 그때 "대형 분류를 뺐다"는 말만 남는다.
    big = set(large_classes(library))
    assert big == set(small.large_classes)
    for names in small.strata.values():
        assert not (set(names) & big)


def test_every_evaluated_query_has_at_least_one_positive(library) -> None:
    truth = functions_ground_truth(library)
    assert truth.positives
    assert all(len(p) >= 1 for p in truth.positives.values())


# ------------------------------------------- 감사가 잡은 것들에 대한 회귀


def test_the_p_value_does_not_underflow_to_zero_in_the_tail() -> None:
    """`1 + erf(z/√2)`는 z가 -8 아래로 가면 배정밀도에서 정확히 0으로 상쇄된다.

    실제 데이터에서 터졌다 - PERFUMING 층(n=179)에서 p=0.0이 찍혔고 참값은
    2.9e-22였다. p=0은 p값이 아니고, 정확히 그런 숫자가 인용된다.
    """
    strong = wilcoxon_signed_rank(np.linspace(0.9, 1.0, 120), np.linspace(0.1, 0.2, 120))
    assert strong["p_value"] > 0.0
    assert strong["p_value"] < 1e-15


def test_there_is_more_than_one_null_baseline() -> None:
    """귀무 기준이 하나뿐이면 그것을 이겼다는 사실이 실제보다 크게 들린다.

    분자량만 두었을 때는 세 방법이 모두 기준선을 이기는 것으로 보였다. TPSA를
    더하자 향료 분류에서 귀무 기준이 0.84로 올라 결론이 뒤집혔다.
    """
    assert len(BASELINE_KEYS) >= 2
    assert BASELINE_KEY in BASELINE_KEYS
    keys = {c.key for c in CRITERIA}
    for key in BASELINE_KEYS:
        assert key in keys
        assert key not in METHOD_KEYS


def test_the_merged_criterion_exists_so_that_not_merging_can_be_tested() -> None:
    """합친 기준이 없으면 "합치면 안 된다"를 반증할 수 없다.

    처음에 그 기준 없이 "합치지 말아야 한다"고 썼고, 재 보니 틀렸다.
    """
    assert "merged_rank_average" in METHOD_KEYS
    assert "merged_rank_average" not in SINGLE_METHOD_KEYS


def test_subsetting_queries_is_not_the_same_as_rewriting_the_ground_truth(library) -> None:
    """두 모드는 다른 것을 한다. 섞으면 표본 크기 감소를 효과 소멸로 읽는다.

    `--exclude-large-classes`는 모든 원료에서 라벨을 떼므로 질의당 중앙 양성이
    1,650에서 438로 떨어진다 - 다른 검색 과제다. `--subset-queries-only`는 정답표를
    그대로 두고 질의만 고른다.

    배수를 문턱으로 쓰되 넉넉히 잡는다. 이 값은 모집단이 정하고, 모집단은
    라이브러리를 다시 만들 때마다 바뀐다 - 507종일 때 178→13(13.7배)이던 것이
    7,484종에서 1,650→438(3.8배)이다. 예전 문턱(5배)은 그 사실 때문에 깨졌고,
    그때 드러난 진짜 문제는 문턱이 아니라 `LARGE_CLASSES`가 이름으로 박혀 있어
    SKIN CONDITIONING(22.1%)을 대형으로 치지 않던 것이었다.
    """
    from eval_alternative_criteria import functions_ground_truth

    full = functions_ground_truth(library)
    relabelled = functions_ground_truth(library, exclude_large=True)
    subset = functions_ground_truth(library, subset_queries_only=True)

    def median_positives(truth):
        return float(np.median([len(v) for v in truth.positives.values()]))

    # 라벨을 떼면 양성 집합이 무너진다. 실측 3.8배이므로 3배를 문턱으로 둔다.
    assert median_positives(relabelled) < median_positives(full) / 3
    # 배수만으로는 "대형 분류를 실제로 뺐는가"를 못 본다. 뺀 라벨이 정답표에
    # 이름으로 남아 있어야, 어느 분류를 뺀 결과인지 읽는 쪽이 알 수 있다.
    assert relabelled.large_classes, "대형 분류를 하나도 찾지 못했다"
    for name in relabelled.large_classes:
        assert name in relabelled.describe
    # 질의만 빼면 남은 질의의 양성 집합은 그대로여야 한다.
    for index, positives in subset.positives.items():
        assert positives == full.positives[index]
    assert len(subset.positives) < len(full.positives)


def test_the_stratified_table_admits_when_a_row_is_not_measuring_its_own_class(library) -> None:
    """질의는 여러 목적을 신고하고 양성은 그 합집합이다.

    그래서 FRAGRANCE 행이 채점하는 것의 85%는 향료가 아니다. 그 사실이 표에 없으면
    행 이름이 거짓말을 한다.
    """
    from eval_alternative_criteria import GroundTruth

    frame = pd.DataFrame(
        {
            "query_index": [0, 1],
            "query": ["a", "b"], "strata": ["X", "X;Y"],
            "positives": [2, 2], "pool": [9, 9],
            "structural": [0.8, 0.7], "pharmacophore": [0.6, 0.5], "tanimoto": [0.7, 0.6],
            "merged_rank_average": [0.75, 0.65],
            "size_baseline": [0.5, 0.5], "polarity_baseline": [0.55, 0.55],
            "random_control": [0.5, 0.5],
        }
    )
    truth = GroundTruth("t", "d", {0: {1}, 1: {0}}, {0: ("X",), 1: ("X", "Y")}, "c")
    table = stratified(frame, truth).set_index("stratum")
    assert "off_class_positive_fraction" in table.columns
    assert "duplicate_of" in table.columns
    # Y 행의 질의는 1번뿐이고 그 양성은 0번인데, 0번은 Y가 아니다.
    assert table.loc["Y", "off_class_positive_fraction"] == 1.0


def test_identical_strata_are_flagged_as_duplicates() -> None:
    """UV FILTER와 LIGHT STABILIZER는 같은 7종이다. 두 행으로 세면 중복 계수다."""
    frame = pd.DataFrame(
        {
            "query_index": [0, 0],
            "query": ["a", "a"], "strata": ["P;Q", "P;Q"],
            "positives": [2, 2], "pool": [9, 9],
            "structural": [0.8, 0.8], "pharmacophore": [0.6, 0.6], "tanimoto": [0.7, 0.7],
            "merged_rank_average": [0.75, 0.75],
            "size_baseline": [0.5, 0.5], "polarity_baseline": [0.5, 0.5],
            "random_control": [0.5, 0.5],
        }
    )
    table = stratified(frame).set_index("stratum")
    assert table.loc["P", "duplicate_of"] == "Q"
    assert table.loc["Q", "duplicate_of"] == "P"


def test_best_minus_null_uses_the_strongest_null_not_a_convenient_one() -> None:
    """향료에서 TPSA 기준선이 0.84다. 분자량(0.69)만 보면 방법이 이긴 것처럼 보인다."""
    frame = pd.DataFrame(
        {
            "query_index": [0],
            "query": ["a"], "strata": ["PERFUMING"], "positives": [5], "pool": [20],
            "structural": [0.59], "pharmacophore": [0.53], "tanimoto": [0.63],
            "merged_rank_average": [0.64],
            "size_baseline": [0.69], "polarity_baseline": [0.84], "random_control": [0.5],
        }
    )
    row = stratified(frame).iloc[0]
    assert row["best_null"] == 0.84
    assert row["best_minus_null"] < 0


# --------------------------------------------------------------- 병렬 실행


def test_parallel_evaluation_matches_serial_bit_for_bit() -> None:
    """질의를 나눠 돌려도 결과가 한 자리도 달라지면 안 된다.

    나누면서 숫자가 달라지면 측정을 바꾼 것이지 빨라진 것이 아니다. 이 성질은
    **MCS가 벽시계 상한에 닿지 않는 작업에서만** 성립한다 - 3D의
    `rdFMCS.FindMCS(timeout=5)`도, 2D의 `MCS_TIMEOUT_SECONDS`/`McsBudget`도
    시간이 다 되면 그때까지 찾은 부분 일치를 값으로 돌려주므로, `--workers`가
    만드는 CPU 경합만으로 값이 바뀐다(아래
    `test_the_three_d_path_is_not_claimed_to_be_bit_stable`). 그래서 여기서는
    원자 수가 작아 어떤 쌍도 상한에 닿지 않는 합성 라이브러리로 경로 자체만
    시험한다. 전체 CosIng 라이브러리에서는 이 동일성을 주장하지 않는다.
    """
    from rdkit import Chem

    from alternative_ingredients import (
        _POPCOUNT,
        _packed,
        MORGAN_GENERATOR,
        IngredientLibrary,
    )
    from build_activity_retrieval_index import _standardize_mol
    from eval_alternative_criteria import evaluate, functions_ground_truth

    smiles = [
        "CCO", "CCCO", "CCCCO", "CC(C)O", "CC(=O)O", "CC(N)C(=O)O",
        "c1ccccc1", "Cc1ccccc1", "Oc1ccccc1", "Nc1ccccc1",
        "c1ccncc1", "c1cccnc1", "c1cc[nH]c1", "c1ccsc1", "c1ccoc1",
        "OCC(O)CO", "NC(=O)c1cccnc1", "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    ]
    molecules = [_standardize_mol(s) for s in smiles]
    fingerprints = np.vstack([_packed(MORGAN_GENERATOR.GetFingerprint(m)) for m in molecules])
    rows = []
    for s, mol in zip(smiles, molecules, strict=True):
        key = Chem.MolToInchiKey(mol)
        rows.append(
            {
                "inci_name": s,
                "synonyms": [],
                "cas": "",
                "functions": "SKIN CONDITIONING",
                "canonical_smiles": Chem.MolToSmiles(mol),
                "inchikey": key,
                "skeleton": key[:14],
                "registered_names": 1,
            }
        )
    library = IngredientLibrary(
        frame=pd.DataFrame(rows),
        molecules=molecules,
        fingerprints=fingerprints,
        popcounts=_POPCOUNT[fingerprints].sum(1, dtype=np.int32),
        source=Path("synthetic"),
        unparsed=0,
    )
    truth = functions_ground_truth(library)
    assert truth.positives, "합성 라이브러리가 질의를 만들어야 한다"

    serial = evaluate(library, truth, workers=1)
    parallel = evaluate(library, truth, workers=4)

    assert list(serial["query_index"]) == list(parallel["query_index"]), "질의 순서가 다릅니다"
    for column in serial.columns:
        if serial[column].dtype.kind in "fi":
            assert serial[column].equals(parallel[column]), f"{column} 열이 다릅니다"


def test_a_single_query_does_not_take_the_parallel_path(library) -> None:
    """질의가 하나면 프로세스를 띄울 이유가 없다. 띄우면 비용만 늘어난다."""
    from eval_alternative_criteria import GroundTruth, evaluate, functions_ground_truth

    full = functions_ground_truth(library)
    one = dict(sorted(full.positives.items())[:1])
    truth = GroundTruth(
        full.name, full.describe, one, {k: full.strata[k] for k in one}, full.caveat
    )
    frame = evaluate(library, truth, workers=8)
    assert len(frame) == 1


def test_progress_is_reported_for_runs_smaller_than_the_old_fixed_interval(
    library, capsys
) -> None:
    """25질의마다 찍으면 그보다 작은 실행은 끝날 때까지 아무것도 안 나온다.

    3D를 켠 20질의 실행이 40분 넘게 도는 동안 진행을 볼 수 없었다.
    """
    from eval_alternative_criteria import GroundTruth, evaluate, functions_ground_truth

    full = functions_ground_truth(library)
    kept = dict(sorted(full.positives.items())[:6])
    truth = GroundTruth(
        full.name, full.describe, kept, {k: full.strata[k] for k in kept}, full.caveat
    )
    evaluate(library, truth, workers=1)
    assert "질의" in capsys.readouterr().err, "6질의 실행이 진행을 하나도 찍지 않았습니다"


def test_the_three_d_path_is_not_claimed_to_be_bit_stable() -> None:
    """3D 경로의 MCS 는 벽시계 시간 제한을 쓴다. 그 사실이 코드에 남아 있어야 한다.

    `--workers 12` 가 만드는 CPU 경합만으로 5초 절벽 근처의 쌍이 넘어가면, 같은
    입력이 워커 수에 따라 다른 AUC 를 낸다. 위의 비트 단위 동일성 테스트는 3D 를
    켜지 않으므로 그것을 잡지 못한다 - 파일에서 유일하게 비결정적인 경로가 정확히
    그 테스트가 지나가지 않는 경로다.

    시간 제한을 없앨 수는 없다(없애면 한 쌍이 실행을 멈춰 세운다). 대신 주장을
    사실에 맞추고, 누가 timeout 을 지우거나 늘려도 이 테스트가 알려 주게 한다.
    """
    import inspect

    import discover_substitutes

    source = inspect.getsource(discover_substitutes._mcs_atom_map)
    assert "timeout=" in source, (
        "MCS 시간 제한이 사라졌다면 3D 경로의 결정성 주장을 다시 재야 합니다"
    )
    assert "canceled" in source, (
        "시간 초과를 어떻게 다루는지가 결과의 재현성을 정합니다"
    )


def test_an_unmeasurable_pharmacophore_is_not_reported_as_chance(library) -> None:
    """Gobbi 지문이 빈 질의의 AUC 는 정확히 0.500 - 도구가 "우연"이라 인쇄하는 값 -
    으로 나왔고, 그 값이 평균과 Wilcoxon 관측에 그대로 섞였다."""
    import numpy as np

    from alternative_ingredients import pharmacophore_match
    from eval_alternative_criteria import score_all_criteria

    profiles = library.pharmacophore_profiles
    for index in range(len(library.molecules)):
        if pharmacophore_match(profiles[index], profiles[index]).usable:
            continue
        scores = score_all_criteria(library, index, with_three_d=False)
        assert np.isnan(scores["pharmacophore"]).all(), (
            "잴 수 없는 파마코포어가 0.0 으로 나가면 안 됩니다"
        )
        # 합친 순위는 나머지 두 기준으로 계속 나와야 한다.
        assert not np.isnan(scores["merged_rank_average"]).any()
        return
    pytest.skip("이 픽스처에는 지문이 빈 분자가 없습니다")
