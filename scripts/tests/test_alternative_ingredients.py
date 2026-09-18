"""활성 핵심구조가 유지되었는지 판정하고, 그 순서로 후보를 세우는 부분.

과제 제목이 요구하는 것은 "닮은 분자"가 아니라 *활성 핵심구조를 유지한* 대체
소재다. 그 둘은 조용히 갈라진다: Tanimoto 0.62에 곁사슬만 닮은 분자가, 유사도
0.35에 핵심 고리가 통째로 남은 분자보다 앞에 서면 표는 여전히 그럴듯해 보이고
아무도 틀렸다고 말해 주지 않는다. 여기서 못을 박는 것이 그 순서다.

Murcko 골격만으로 판정하던 판이 실제로 한 번 이렇게 무너졌다. 살리실산의 Murcko
골격은 벤젠 고리 하나뿐이라 2,3-자일레놀이 "핵심 골격 동일"로 올라왔다 - 벤젠을
가진 분자면 무엇이든 그렇게 된다. 그 회귀를 막는 시험이 아래에 있다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from rdkit import Chem  # noqa: E402

from alternative_ingredients import (  # noqa: E402
    CORE_KEPT_GRADES,
    GRADE_LABEL_KO,
    GRADE_ORDER,
    RETAINED_COVERAGE,
    IngredientLibrary,
    _extend_unique,
    _name_sort_key,
    _packed,
    _split_list,
    attach_measured_evidence,
    core_retention,
    find_alternatives,
    library_note,
    load_ingredient_library,
    MAX_3D_CANDIDATES,
    McsBudget,
    SORT_CORE_FIRST,
    SORT_MERGED,
    SORT_MODES,
    SORT_POLARITY,
    annotate_three_d,
    merged_rank_score,
    murcko_core,
    percentile_rank,
    pharmacophore_match,
    prepare_query_3d,
    pharmacophore_profile,
    scan_alternatives,
    scan_summary,
    signal_disagreement,
    three_d_match,
)
from build_activity_retrieval_index import MORGAN_GENERATOR, _standardize_mol  # noqa: E402
from similar_compounds import _POPCOUNT  # noqa: E402

COSING = ROOT / "data" / "cosing" / "cosing.parquet"
INDEX_DIR = ROOT / "data" / "similarity_index_202609"

SALICYLIC_ACID = "OC(=O)c1ccccc1O"
ETHYLHEXYL_SALICYLATE = "CCCCC(CC)COC(=O)c1ccccc1O"
XYLENOL = "Cc1cccc(O)c1C"          # 2,3-자일레놀
PHENOL = "Oc1ccccc1"
NIACINAMIDE = "NC(=O)c1cccnc1"
CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
PALMITIC_ACID = "CCCCCCCCCCCCCCCC(=O)O"
STEARIC_ACID = "CCCCCCCCCCCCCCCCCC(=O)O"
GLYCERIN = "OCC(O)CO"


def _retention(query: str, candidate: str):
    """화면과 CLI가 지나는 경로 그대로. 표준화 없이 비교하면 값이 달라진다."""
    return core_retention(_standardize_mol(query), _standardize_mol(candidate))


def _library(entries: list[tuple[str, str]]) -> IngredientLibrary:
    """손으로 만든 등재 원료 표.

    `data/cosing`는 gitignore 대상이라 있는 기계에서만 있다. 정렬 규칙처럼 데이터와
    무관한 성질을 그 파일에 매달아 두면, 없는 기계에서는 조용히 시험되지 않는다.
    """
    molecules = [_standardize_mol(smiles) for _, smiles in entries]
    fingerprints = np.vstack(
        [_packed(MORGAN_GENERATOR.GetFingerprint(mol)) for mol in molecules]
    )
    rows = []
    for (name, _), mol in zip(entries, molecules, strict=True):
        key = Chem.MolToInchiKey(mol)
        rows.append(
            {
                "inci_name": name,
                "synonyms": [],
                "cas": "",
                "functions": "",
                "canonical_smiles": Chem.MolToSmiles(mol),
                "inchikey": key,
                "skeleton": key[:14],
                "registered_names": 1,
            }
        )
    return IngredientLibrary(
        frame=pd.DataFrame(rows),
        molecules=molecules,
        fingerprints=fingerprints,
        popcounts=_POPCOUNT[fingerprints].sum(1, dtype=np.int32),
        source=Path("synthetic"),
        unparsed=0,
    )


# ---------------------------------------------------------------- core_retention


def test_the_same_structure_is_graded_identical() -> None:
    retention = _retention(SALICYLIC_ACID, SALICYLIC_ACID)
    assert retention.grade == "identical"
    assert retention.coverage == 1.0
    assert retention.basis == "same_structure"


@pytest.mark.parametrize("query,candidate", [
    ("C[C@H](O)C(=O)O", "C[C@@H](O)C(=O)O"),
    ("C[C@H](O)C(=O)O", "CC(O)C(=O)O"),
    ("N[C@H](Cc1ccccc1)C(=O)O", "N[C@@H](Cc1ccccc1)C(=O)O"),
    ("F/C=C/F", "F/C=C\\F"),
])
def test_changed_or_missing_specified_stereochemistry_is_not_core_retention(query, candidate) -> None:
    retention = _retention(query, candidate)
    assert retention.grade not in CORE_KEPT_GRADES
    assert retention.basis == "stereochemistry_mismatch_or_unspecified"


def test_budgeted_scan_is_independent_of_library_row_order(monkeypatch) -> None:
    import alternative_ingredients as module

    entries = [("ETHANOL", "CCO"), ("AMINE", "CCN"), ("ACID", "CC(=O)O")]
    visits = []
    original = module.core_retention

    def record(query, candidate, budget, prepared):
        visits.append(Chem.MolToInchiKey(candidate))
        return original(query, candidate, budget, prepared)

    monkeypatch.setattr(module, "core_retention", record)
    first = scan_alternatives("c1ccccc1", _library(entries)).frame
    initial_order = visits[:]
    visits.clear()
    second = scan_alternatives("c1ccccc1", _library(list(reversed(entries)))).frame
    assert visits == initial_order
    pd.testing.assert_frame_equal(first, second)


def test_a_derivative_carrying_the_whole_query_is_graded_contains() -> None:
    """에틸헥실살리실레이트는 살리실산을 에스터화한 것이다.

    대체소재 탐색이 실제로 찾아야 하는 모양이 이것이다 - 곁사슬은 달라졌는데 활성
    핵심은 원자 하나 빠지지 않고 남아 있다. Tanimoto는 0.35에 불과하다.
    """
    retention = _retention(SALICYLIC_ACID, ETHYLHEXYL_SALICYLATE)
    assert retention.grade == "contains"
    assert retention.coverage == 1.0
    assert retention.basis == "query_is_substructure"
    assert retention.grade in CORE_KEPT_GRADES


def test_an_unrelated_molecule_is_not_offered_as_a_substitute() -> None:
    retention = _retention(SALICYLIC_ACID, CAFFEINE)
    assert retention.grade == "different"
    assert retention.grade not in CORE_KEPT_GRADES


def test_a_shared_benzene_ring_is_not_a_shared_pyridine_core() -> None:
    """피리딘 질의에 벤젠만 겹치는 후보가 "핵심구조 그대로"로 올라오면 안 된다.

    나이아신아마이드의 핵심은 피리딘이고, 자일레놀에는 그 고리가 없다. 고리 안의
    원자가 다르면 같은 고리가 아니다.
    """
    retention = _retention(NIACINAMIDE, XYLENOL)
    assert retention.grade not in ("identical", "contains")
    assert retention.grade not in CORE_KEPT_GRADES
    assert not retention.scaffold_match


def test_a_murcko_match_on_its_own_does_not_decide_the_grade() -> None:
    """실제로 났던 오판: 살리실산의 골격은 벤젠 하나라 자일레놀이 "골격 동일"이었다.

    골격 일치는 지금도 참이다(그리고 참인 채로 함께 돌려준다). 등급을 정하는 것이
    골격이 아니라 질의 분자 전체를 덮는 비율이라는 사실을 여기서 고정한다.
    """
    retention = _retention(SALICYLIC_ACID, XYLENOL)
    assert retention.scaffold_match is True
    assert retention.grade not in ("identical", "contains")


def test_an_acyclic_query_carries_that_fact_in_its_basis() -> None:
    """고리가 없으면 골격도 없다. 그 사실이 판정 근거에서 사라지면 안 된다."""
    retention = _retention(PALMITIC_ACID, CAFFEINE)
    assert "acyclic_query" in retention.basis
    assert not retention.scaffold_match


def test_an_acyclic_query_does_not_grade_everything_as_retaining() -> None:
    """빈 골격을 상대로 부분구조 검사를 하면 무엇이든 일치로 나온다.

    글리세린은 팔미트산의 대체소재가 아니다. 골격 검사에 기대는 판정이 되살아나면
    이 줄이 먼저 깨진다.
    """
    retention = _retention(PALMITIC_ACID, GLYCERIN)
    assert retention.grade not in CORE_KEPT_GRADES
    assert retention.coverage < RETAINED_COVERAGE
    assert not retention.scaffold_match


def test_an_acyclic_query_still_finds_its_own_homolog() -> None:
    """비고리형이라고 전부 떨어뜨리는 것도 틀린 답이다."""
    assert _retention(PALMITIC_ACID, STEARIC_ACID).grade == "contains"


def test_every_judgement_states_what_it_was_based_on() -> None:
    """등급만 화면에 남고 근거가 사라지면, 구조 판정이 측정처럼 읽힌다."""
    pairs = [
        (SALICYLIC_ACID, ETHYLHEXYL_SALICYLATE),
        (SALICYLIC_ACID, CAFFEINE),
        (NIACINAMIDE, XYLENOL),
        (PALMITIC_ACID, GLYCERIN),
    ]
    for query, candidate in pairs:
        retention = _retention(query, candidate)
        assert retention.basis.strip()
        assert 0.0 <= retention.coverage <= 1.0
        assert 0.0 <= retention.share <= 1.0
        assert retention.public()["label_ko"] == GRADE_LABEL_KO[retention.grade]


def test_an_unreadable_candidate_is_reported_as_unjudged() -> None:
    """판정하지 못한 것을 "다른 구조"로 적으면 안 본 것을 봤다고 말하는 셈이다."""
    retention = core_retention(_standardize_mol(SALICYLIC_ACID), None)
    assert retention.grade == "unknown"
    assert retention.basis == "no_molecule"


# ------------------------------------------------------------------ murcko_core


def test_murcko_core_says_acyclic_instead_of_returning_an_empty_scaffold() -> None:
    scaffold, basis = murcko_core(_standardize_mol(PALMITIC_ACID))
    assert scaffold is None
    assert basis == "acyclic_query"


def test_murcko_core_returns_a_scaffold_when_there_is_a_ring() -> None:
    scaffold, basis = murcko_core(_standardize_mol(SALICYLIC_ACID))
    assert basis == "murcko_scaffold"
    assert scaffold is not None and scaffold.GetNumHeavyAtoms() > 0


# ----------------------------------------------------------------------- 등급표


def test_every_grade_has_both_an_order_and_a_korean_label() -> None:
    """정렬 키와 화면 문구가 갈라지면, 새 등급이 표 맨 아래에 이름 없이 쌓인다."""
    assert set(GRADE_ORDER) == set(GRADE_LABEL_KO)
    assert len(set(GRADE_LABEL_KO.values())) == len(GRADE_LABEL_KO)
    assert all(label.strip() for label in GRADE_LABEL_KO.values())


def test_the_kept_grades_are_the_top_of_the_order() -> None:
    """"유지되었다"고 세는 등급이 정렬에서 뒤로 밀리면 두 숫자가 서로를 배반한다."""
    assert set(CORE_KEPT_GRADES) <= set(GRADE_ORDER)
    worst_kept = max(GRADE_ORDER[grade] for grade in CORE_KEPT_GRADES)
    others = set(GRADE_ORDER) - set(CORE_KEPT_GRADES)
    assert worst_kept < min(GRADE_ORDER[grade] for grade in others)


# -------------------------------------------------------------- CosIng 필드 정리


def test_the_glued_cosing_fields_are_split_on_both_separators() -> None:
    assert _split_list("110-15-6; 141-82-2 / 50-21-5") == ["110-15-6", "141-82-2", "50-21-5"]
    assert _split_list("SKIN CONDITIONING") == ["SKIN CONDITIONING"]


def test_an_empty_cosing_cell_yields_no_entries() -> None:
    """비어 있는 칸은 pandas를 지나면 문자열 "nan"이 된다."""
    assert _split_list(None) == []
    assert _split_list(float("nan")) == []
    assert _split_list("   ") == []


def test_merging_names_keeps_the_first_spelling_and_drops_repeats() -> None:
    target = ["NIACINAMIDE"]
    _extend_unique(target, ["NIACINAMIDE", "", "VITAMIN B3"])
    assert target == ["NIACINAMIDE", "VITAMIN B3"]


def test_the_inci_name_sorts_ahead_of_the_iupac_listing() -> None:
    """같은 구조에 이름이 여럿 붙는다. 읽는 사람이 알아보는 쪽이 앞이어야 한다."""
    iupac = (
        "[(2E,4E,6E,8E)-3,7-dimethyl-9-(2,6,6-trimethylcyclohexen-1-yl)\n"
        "nona-2,4,6,8-tetraenyl] acetate"
    )
    assert sorted([iupac, "RETINYL ACETATE"], key=_name_sort_key)[0] == "RETINYL ACETATE"
    assert _name_sort_key("RETINYL ACETATE")[0] == 0
    assert _name_sort_key(iupac)[0] == 1


# ------------------------------------------------------------- find_alternatives


def test_a_core_retaining_candidate_outranks_a_more_similar_one() -> None:
    """이 저장소가 `find_similar`과 다른 이유의 전부.

    페놀은 유사도가 더 높지만(0.381) 카복실기가 없어 핵심이 일부만 남는다.
    에틸헥실살리실레이트는 유사도가 낮지만(0.350) 살리실산을 통째로 품고 있다.
    유사도 정렬로 되돌아가면 페놀이 1위가 되고, 표는 여전히 멀쩡해 보인다.
    """
    library = _library(
        [
            ("PHENOL", PHENOL),
            ("ETHYLHEXYL SALICYLATE", ETHYLHEXYL_SALICYLATE),
            ("CAFFEINE", CAFFEINE),
        ]
    )
    frame = find_alternatives(SALICYLIC_ACID, library, limit=10)
    assert list(frame["inci_name"]) == ["ETHYLHEXYL SALICYLATE", "PHENOL", "CAFFEINE"]
    assert frame.loc[0, "core_grade"] == "contains"
    # 순위를 뒤집은 것이 유사도가 아니라 등급이라는 사실을 함께 못박는다.
    assert frame.loc[0, "similarity"] < frame.loc[1, "similarity"]
    assert list(frame["rank"]) == [1, 2, 3]


def test_similarity_still_orders_candidates_inside_one_grade() -> None:
    """등급이 같으면 그때는 유사도가 결정한다."""
    library = _library(
        [
            ("ETHYLHEXYL SALICYLATE", ETHYLHEXYL_SALICYLATE),
            ("METHYL SALICYLATE", "COC(=O)c1ccccc1O"),
        ]
    )
    frame = find_alternatives(SALICYLIC_ACID, library, limit=10)
    assert set(frame["core_grade"]) == {"contains"}
    assert list(frame["inci_name"]) == ["METHYL SALICYLATE", "ETHYLHEXYL SALICYLATE"]


def test_require_core_drops_everything_below_retained() -> None:
    """"일부 유지"부터는 핵심이 이미 바뀐 것이다."""
    library = _library(
        [
            ("PHENOL", PHENOL),                       # partial
            ("ETHYLHEXYL SALICYLATE", ETHYLHEXYL_SALICYLATE),  # contains
            ("CAFFEINE", CAFFEINE),                   # different
            ("GLYCERIN", GLYCERIN),                   # weak/different
        ]
    )
    frame = find_alternatives(SALICYLIC_ACID, library, limit=10, require_core=True)
    assert list(frame["inci_name"]) == ["ETHYLHEXYL SALICYLATE"]
    assert set(frame["core_grade"]) <= set(CORE_KEPT_GRADES)


def test_the_default_keeps_the_near_misses_visible() -> None:
    """빈 표는 "후보가 없다"와 "조건이 좁았다"를 구분해 주지 않는다."""
    library = _library([("PHENOL", PHENOL), ("CAFFEINE", CAFFEINE)])
    assert find_alternatives(SALICYLIC_ACID, library, limit=10, require_core=True).empty
    relaxed = find_alternatives(SALICYLIC_ACID, library, limit=10)
    assert len(relaxed) == 2
    assert set(relaxed["core_grade"]).isdisjoint(CORE_KEPT_GRADES)


def test_limit_cuts_the_list_but_not_the_count_of_what_was_compared() -> None:
    """상위 몇 줄만 보여 주더라도, 몇 종을 봤는지는 따로 세어져 있어야 한다."""
    library = _library(
        [
            ("PHENOL", PHENOL),
            ("ETHYLHEXYL SALICYLATE", ETHYLHEXYL_SALICYLATE),
            ("CAFFEINE", CAFFEINE),
        ]
    )
    assert len(find_alternatives(SALICYLIC_ACID, library, limit=1)) == 1
    scan = scan_alternatives(SALICYLIC_ACID, library)
    assert scan.scanned == 3
    assert scan.kept == 1
    assert sum(scan.grade_counts.values()) == 3
    assert "3종을 전부 대조해" in scan_summary(scan)


def test_a_query_that_is_itself_registered_is_marked_not_hidden() -> None:
    """자기 자신이 1위로 나오는 것은 결과가 아니라 조회다. 그 구분이 표에 남는다."""
    library = _library([("SALICYLIC ACID", SALICYLIC_ACID), ("PHENOL", PHENOL)])
    frame = find_alternatives(SALICYLIC_ACID, library, limit=10)
    assert bool(frame.loc[0, "is_query"]) is True
    assert frame.loc[0, "core_grade"] == "identical"
    excluded = find_alternatives(SALICYLIC_ACID, library, limit=10, exclude_self=True)
    assert list(excluded["inci_name"]) == ["PHENOL"]


def test_an_oversized_input_is_refused_with_a_readable_reason() -> None:
    """대체소재를 찾는 질문이 아닌 입력 하나가 워커를 붙잡으면 안 된다."""
    library = _library([("PHENOL", PHENOL)])
    with pytest.raises(ValueError) as failure:
        find_alternatives("C" * 400, library)
    assert "중원자" in str(failure.value)


# ------------------------------------------------------ 측정 근거를 붙이는 자리


def _fake_index(records: list[dict[str, object]]):
    """`similar_compounds` 인덱스 260MB를 읽지 않고 조인만 시험한다."""
    from similar_compounds import SimilarityIndex

    return SimilarityIndex(
        fingerprints=np.zeros((max(1, len(records)), 256), dtype=np.uint8),
        popcounts=np.zeros(max(1, len(records)), dtype=np.int32),
        ligands=pd.DataFrame(records),
        manifest={},
    )


def test_an_unmeasured_candidate_reads_as_unmeasured_not_as_inactive() -> None:
    """측정이 없다는 것과 측정해서 활성이 없었다는 것은 다른 결정을 부른다.

    화면이 이 둘을 같은 칸에 적는 순간, 재 본 적 없는 후보가 탈락한 후보로 읽힌다.
    """
    library = _library(
        [("ETHYLHEXYL SALICYLATE", ETHYLHEXYL_SALICYLATE), ("PHENOL", PHENOL)]
    )
    frame = find_alternatives(SALICYLIC_ACID, library, limit=10)
    measured_key = Chem.MolToInchiKey(_standardize_mol(ETHYLHEXYL_SALICYLATE))
    index = _fake_index(
        [
            {
                "ligand_index": 0,
                "standard_inchikey": measured_key,
                "canonical_smiles": ETHYLHEXYL_SALICYLATE,
                "target_count": 2,
                "best_pactivity": 5.12,
                "top_targets": "P00001;P00001;P00002",
                "all_targets": "P00001;P00002",
                "evidence": "below_threshold",
            }
        ]
    )
    out = attach_measured_evidence(frame, index, SALICYLIC_ACID)
    by_name = out.set_index("inci_name")
    assert by_name.loc["PHENOL", "measured_evidence"] == "not_measured"
    # 붙지 않은 칸은 pandas를 지나며 None이 아니라 NaN이 된다. NaN은 참이라
    # `값 or 0`으로 받으면 int()가 터진다 - 화면 쪽 변환기가 NaN을 따로 다루는
    # 이유가 이 한 줄이다.
    assert pd.isna(by_name.loc["PHENOL", "measured_target_count"])
    assert by_name.loc["ETHYLHEXYL SALICYLATE", "measured_evidence"] == "below_threshold"
    assert by_name.loc["ETHYLHEXYL SALICYLATE", "measured_match"] == "exact"
    # 같은 UniProt가 세 번 실려 있던 v1 인덱스의 흔적은 읽는 쪽에서도 막는다.
    assert by_name.loc["ETHYLHEXYL SALICYLATE", "measured_top_targets"] == ["P00001", "P00002"]


def test_no_shared_targets_is_not_reported_as_a_disagreement() -> None:
    """입력에 측정 기록이 없으면 교집합은 언제나 빈다 - 그것은 결과가 아니다."""
    library = _library([("PHENOL", PHENOL)])
    frame = find_alternatives(SALICYLIC_ACID, library, limit=10)
    unrelated = _fake_index(
        [
            {
                "ligand_index": 0,
                "standard_inchikey": "ZZZZZZZZZZZZZZ-UHFFFAOYSA-N",
                "canonical_smiles": CAFFEINE,
                "target_count": 1,
                "best_pactivity": 7.0,
                "top_targets": "P00009",
                "all_targets": "P00009",
                "evidence": "above_threshold",
            }
        ]
    )
    out = attach_measured_evidence(frame, unrelated, SALICYLIC_ACID)
    assert list(out["shared_targets"]) == [[]]
    assert out.attrs["query_measured"] is False


def test_a_connectivity_only_match_says_so() -> None:
    """자기 값이 없으면 앞 14자로 내려가되, 내려갔다는 사실을 적어야 한다.

    붙이는 것 자체는 의도된 동작이다. 등재 구조와 측정 기록은 입체배치나 염 형태가
    다르게 적혀 있는 경우가 흔해서, 전체 키로만 맞추면 붙어야 할 근거가 붙지 않는다.
    적지 않는 것이 문제다.
    """
    library = _library([("MENTHOL", "C[C@@H]1CC[C@H](C(C)C)[C@@H](O)C1")])
    frame = find_alternatives("C[C@@H]1CC[C@H](C(C)C)[C@@H](O)C1", library, limit=10)
    skeleton = str(frame.loc[0, "skeleton"])
    index = _fake_index(
        [
            {
                "ligand_index": 0,
                "standard_inchikey": f"{skeleton}-XXXXXXXXXX-N",
                "canonical_smiles": "CC1CCC(C(C)C)C(O)C1",
                "target_count": 1,
                "best_pactivity": 6.0,
                "top_targets": "P00003",
                "all_targets": "P00003",
                "evidence": "above_threshold",
            }
        ]
    )
    out = attach_measured_evidence(frame, index, "")
    assert out.loc[0, "measured_match"] == "connectivity"


def test_the_molecules_own_measurement_wins_over_a_better_measured_isomer() -> None:
    """연결성이 같은 행이 여럿이면 "표적이 가장 많은 행"이 뽑힌다.

    전체 InChIKey를 먼저 찾지 않으면, 자기 측정값이 인덱스에 있는데도 더 잘 측정된
    이성질체의 값이 대신 화면에 찍힌다 - 3-Methylfentanyl이 자기 pAct 8.16 대신
    다른 입체이성질체의 10.7로 나오던 것이 그 경우다.
    """
    library = _library([("MENTHOL", "C[C@@H]1CC[C@H](C(C)C)[C@@H](O)C1")])
    frame = find_alternatives("C[C@@H]1CC[C@H](C(C)C)[C@@H](O)C1", library, limit=10)
    own_key = str(frame.loc[0, "inchikey"])
    skeleton = own_key[:14]
    index = _fake_index(
        [
            # 이성질체 쪽이 표적을 더 많이 갖고 있어, 연결성으로만 고르면 이쪽이 이긴다.
            {
                "ligand_index": 0,
                "standard_inchikey": f"{skeleton}-XXXXXXXXXX-N",
                "canonical_smiles": "CC1CCC(C(C)C)C(O)C1",
                "target_count": 9,
                "best_pactivity": 10.7,
                "top_targets": "P00001;P00002;P00003",
                "all_targets": "P00001;P00002;P00003",
                "evidence": "above_threshold",
            },
            {
                "ligand_index": 1,
                "standard_inchikey": own_key,
                "canonical_smiles": "C[C@@H]1CC[C@H](C(C)C)[C@@H](O)C1",
                "target_count": 1,
                "best_pactivity": 8.16,
                "top_targets": "P00009",
                "all_targets": "P00009",
                "evidence": "above_threshold",
            },
        ]
    )
    out = attach_measured_evidence(frame, index, "")
    assert out.loc[0, "measured_match"] == "exact"
    assert out.loc[0, "measured_best_pactivity"] == 8.16
    assert out.loc[0, "measured_top_targets"] == ["P00009"]


# ---------------------------------------------------- 실제 등재 원료 표가 있을 때


@pytest.fixture(scope="module")
def cosing_library() -> IngredientLibrary:
    if not COSING.is_file():
        pytest.skip("CosIng ingredient table has not been built")
    return load_ingredient_library()


def test_a_missing_ingredient_table_fails_closed_with_the_way_to_build_it(
    tmp_path: Path,
) -> None:
    """빈 표를 조용히 내놓으면 "후보가 없다"로 읽힌다."""
    with pytest.raises(FileNotFoundError) as failure:
        load_ingredient_library(tmp_path / "absent.parquet")
    message = str(failure.value)
    assert "stage0_cosing.py" in message
    assert "CosIng" in message


def test_the_shipped_library_holds_one_row_per_structure(
    cosing_library: IngredientLibrary,
) -> None:
    """같은 분자가 이름만 다르게 두 칸을 차지하면 상위 20칸이 그만큼 줄어든다."""
    frame = cosing_library.frame
    assert not frame["inchikey"].duplicated().any()
    assert len(frame) < int(frame["registered_names"].sum())
    assert (frame["inci_name"].str.strip() != "").all()


def test_the_library_note_says_what_the_population_actually_is(
    cosing_library: IngredientLibrary,
) -> None:
    """후보가 적을 때 도구를 의심하지 않으려면 모집단이 화면에 있어야 한다."""
    note = library_note(cosing_library)
    assert f"{len(cosing_library.frame):,}종" in note
    if cosing_library.registered_entries:
        assert f"{cosing_library.registered_entries:,}건" in note


def test_the_recomputed_fingerprints_are_comparable_with_a_standardised_query(
    cosing_library: IngredientLibrary,
) -> None:
    """저장된 `ecfp4` 열은 표준화 전에 만들어진 것이라 질의 쪽과 맞대면 안 된다.

    그대로 썼다면 같은 분자를 다시 물어봐도 유사도가 1이 아니게 나온다. 여기서
    확인하는 것이 정확히 그 지점이다.
    """
    row = cosing_library.frame.iloc[0]
    frame = find_alternatives(str(row["canonical_smiles"]), cosing_library, limit=1)
    assert frame.loc[0, "inchikey"] == row["inchikey"]
    assert frame.loc[0, "similarity"] == 1.0
    assert frame.loc[0, "core_grade"] == "identical"


def test_a_real_query_ranks_core_retaining_ingredients_first(
    cosing_library: IngredientLibrary,
) -> None:
    """507종 전수 대조에서도 정렬 규칙이 그대로여야 의미가 있다."""
    scan = scan_alternatives(SALICYLIC_ACID, cosing_library, exclude_self=True)
    assert scan.scanned >= len(cosing_library.frame) - 1
    orders = [GRADE_ORDER[grade] for grade in scan.frame["core_grade"]]
    assert orders == sorted(orders)
    assert scan.kept > 0, "살리실산의 핵심을 유지한 등재 원료가 하나도 없을 수는 없다"


# ------------------------------------------- 측정 라이브러리 인덱스까지 있을 때


@pytest.fixture(scope="module")
def measured_index():
    if not (INDEX_DIR / "manifest.json").is_file():
        pytest.skip("similarity index has not been built")
    from similar_compounds import load_index

    return load_index(INDEX_DIR)


def test_measured_evidence_joins_onto_the_shipped_library(
    cosing_library: IngredientLibrary, measured_index
) -> None:
    """조인이 실제 데이터에서도 붙는지, 그리고 안 붙은 칸이 무엇으로 읽히는지."""
    frame = find_alternatives(SALICYLIC_ACID, cosing_library, limit=20)
    out = attach_measured_evidence(frame, measured_index, SALICYLIC_ACID)
    evidence = list(out["measured_evidence"])
    assert len(evidence) == len(frame)
    # 측정이 없다는 뜻의 칸은 오직 이 한 가지 문자열이어야 한다. "inactive"로
    # 새는 순간 화면이 재 본 적 없는 후보를 탈락한 후보로 말하게 된다.
    assert "inactive" not in evidence
    assert set(evidence) - {"not_measured"}, "20건 전부에 측정 기록이 없을 수는 없다"
    for value in out["measured_top_targets"]:
        assert len(value) == len(set(value))


# ------------------------------------------- 파마코포어는 구조와 다른 것을 잰다


def test_the_two_screens_measure_pharmacophore_identically() -> None:
    """대체소재 검색과 Stage 5.6b가 같은 말로 다른 것을 재면 안 된다.

    이 화면은 `discover_substitutes`의 함수를 그대로 부른다. 나중에 한쪽만 고치면
    "파마코포어 유지"라는 같은 라벨이 두 화면에서 다른 뜻이 되는데, 그 어긋남은
    화면을 봐서는 알 수 없다. 여기서 두 경로의 숫자를 직접 맞대 둔다.
    """
    from collections import Counter
    from pathlib import Path as _Path

    from rdkit import DataStructs, RDConfig
    from rdkit.Chem import ChemicalFeatures
    from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D

    import discover_substitutes as reference

    query = _standardize_mol(SALICYLIC_ACID)
    candidate = _standardize_mol("OC(=O)c1cccc(O)c1")

    # 5.6b가 자기 안에서 하는 계산을 그대로 재현한다.
    factory = ChemicalFeatures.BuildFeatureFactory(
        str(_Path(RDConfig.RDDataDir) / "BaseFeatures.fdef")
    )
    expected_similarity = DataStructs.TanimotoSimilarity(
        Generate.Gen2DFingerprint(query, Gobbi_Pharm2D.factory),
        Generate.Gen2DFingerprint(candidate, Gobbi_Pharm2D.factory),
    )
    expected_recall, expected_precision, _ = reference._feature_overlap_scores(
        reference._feature_counts(query, factory),
        reference._feature_counts(candidate, factory),
    )

    match = pharmacophore_match(pharmacophore_profile(query), pharmacophore_profile(candidate))
    assert match.similarity == round(float(expected_similarity), 3)
    assert match.recall == round(expected_recall, 3)
    assert match.precision == round(expected_precision, 3)


def test_the_feature_families_are_the_ones_stage_5_6b_uses() -> None:
    """특징족 목록이 갈라지면 두 화면의 재현율이 조용히 달라진다."""
    import discover_substitutes as reference

    from alternative_ingredients import PHARMACOPHORE_FAMILIES

    assert tuple(PHARMACOPHORE_FAMILIES) == tuple(reference.FEATURE_FAMILIES)


def test_recall_and_precision_answer_different_questions() -> None:
    """재현율은 "내 특징이 남았나", 정밀도는 "후보가 그 특징으로 되어 있나"다.

    구조 쪽의 coverage/share와 같은 구조이며, 큰 후보에서 갈린다.
    """
    query = _standardize_mol("Oc1ccccc1")                    # 페놀
    big = _standardize_mol("Oc1ccccc1CCCCCCCCCCCCN")         # 페놀 + 긴 아민 사슬
    match = pharmacophore_match(pharmacophore_profile(query), pharmacophore_profile(big))
    assert match.recall > match.precision, (
        "질의의 특징은 다 들어 있는데 후보가 훨씬 크면 정밀도가 낮아야 한다"
    )


def test_a_query_with_no_pharmacophore_features_is_reported_unusable() -> None:
    """특징이 하나도 안 잡히는 분자가 0.0으로 조용히 내려가면 안 된다."""
    tiny = Chem.MolFromSmiles("C")
    match = pharmacophore_match(pharmacophore_profile(tiny), pharmacophore_profile(tiny))
    assert match.usable is False
    assert match.similarity == 0.0


def test_the_pharmacophore_columns_are_absent_unless_asked_for() -> None:
    """켜지 않았는데 0이 찍히면 "특징이 하나도 안 겹친다"로 읽힌다.

    실제로는 재 보지 않은 것이고, 그 둘은 다른 말이다.
    """
    library = _library([("PHENOL", "Oc1ccccc1"), ("ANISOLE", "COc1ccccc1")])
    off = scan_alternatives(SALICYLIC_ACID, library)
    assert "pharm_similarity" not in off.frame.columns
    on = scan_alternatives(SALICYLIC_ACID, library, with_pharmacophore=True)
    assert "pharm_similarity" in on.frame.columns
    assert on.frame["pharm_similarity"].notna().all()


def test_the_pharmacophore_signal_does_not_change_the_structural_grade() -> None:
    """두 신호는 독립이어야 한다. 하나가 다른 하나의 등급을 흔들면 합친 것이다."""
    library = _library([("PHENOL", "Oc1ccccc1"), ("ANISOLE", "COc1ccccc1"),
                        ("CYCLOHEXANOL", "OC1CCCCC1")])
    off = scan_alternatives(SALICYLIC_ACID, library)
    on = scan_alternatives(SALICYLIC_ACID, library, with_pharmacophore=True)
    assert list(off.frame["core_grade"]) == list(on.frame["core_grade"])
    assert list(off.frame["inchikey"]) == list(on.frame["inchikey"])
    assert off.kept == on.kept


def test_the_disagreement_between_the_two_signals_is_reported() -> None:
    """나란히 놓기만 하면 서로를 뒷받침하는 것처럼 읽힌다.

    이 저장소는 "네 방법이 점수를 냈다"를 "네 방법이 동의했다"로 쓴 적이 있고,
    같은 실수를 여기서 반복하지 않는다.
    """
    library = _library([
        ("PHENOL", "Oc1ccccc1"),
        ("ANISOLE", "COc1ccccc1"),
        ("CYCLOHEXANOL", "OC1CCCCC1"),
        ("BENZOIC ACID", "OC(=O)c1ccccc1"),
    ])
    scan = scan_alternatives(SALICYLIC_ACID, library, with_pharmacophore=True)
    agreement = signal_disagreement(scan.frame)
    assert agreement["available"] is True
    assert "spearman" in agreement
    assert isinstance(agreement["same_top"], bool)
    # 켜지 않았으면 상관을 낼 근거가 없다. 없는데 있다고 말하면 안 된다.
    assert signal_disagreement(scan_alternatives(SALICYLIC_ACID, library).frame) == {
        "available": False
    }


def test_stage_5_6b_gate_values_are_not_imported(cosing_library: IngredientLibrary) -> None:
    """5.6b의 통과 기준을 이 모집단에 그대로 쓰면 범주 오류다.

    그 값(score>=0.55, recall>=0.60)은 같은 표적에 활성이 보고된 화합물 풀에
    맞춰진 것이고, 등재 원료 507종에 적용하면 질의당 0-1종만 통과한다. 이 화면은
    판정하지 않고 측정값만 보여 준다 - 모듈 어디에도 그 문턱이 없어야 한다.
    """
    import ast

    source = (ROOT / "scripts" / "alternative_ingredients.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    # 주석과 docstring이 아니라 **실행되는 코드**만 본다. 왜 안 가져왔는지 적어 둔
    # 문장까지 금지하면, 이 결정을 설명하는 것 자체가 막힌다.
    names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    } | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    } | {
        node.arg for node in ast.walk(tree) if isinstance(node, ast.arg)
    }
    for forbidden in ("min_pharmacophore", "min_feature_recall", "pharmacophore_score"):
        assert forbidden not in names, f"5.6b의 판정 기준이 새어 들어왔습니다: {forbidden}"

    constants = {
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, float)
    }
    for forbidden in (0.55, 0.65):
        assert forbidden not in constants, f"5.6b의 가중치·문턱이 상수로 들어왔습니다: {forbidden}"


# ------------------------------------------------ 세 기준을 합친 순위


def test_percentile_rank_removes_the_scale_difference() -> None:
    """원점수를 그냥 더하면 눈금이 촘촘한 쪽이 이긴다.

    구조 유지율은 중원자 수로 나눈 계단값이라 같은 값이 수십 개씩 나오고,
    Tanimoto는 촘촘한 실수다.
    """
    coarse = np.array([0.5, 0.5, 0.5, 1.0])
    fine = np.array([0.11, 0.12, 0.13, 0.14])
    assert percentile_rank(coarse).max() == 1.0
    assert percentile_rank(fine).max() == 1.0
    # 순서만 남고 간격은 사라진다.
    assert list(percentile_rank(fine)) == [0.0, pytest.approx(1 / 3), pytest.approx(2 / 3), 1.0]


def test_the_merged_score_is_the_average_of_three_percentile_ranks() -> None:
    """손으로 계산한 값 전부와 맞춘다.

      a = [1, 2, 3] -> 백분위 [0.0, 0.5, 1.0]
      b = [3, 2, 1] -> 백분위 [1.0, 0.5, 0.0]
      c = [1, 3, 2] -> 백분위 [0.0, 1.0, 0.5]
      평균          -> [1/3, 2/3, 0.5]
    """
    scores = {
        "a": np.array([1.0, 2.0, 3.0]),
        "b": np.array([3.0, 2.0, 1.0]),
        "c": np.array([1.0, 3.0, 2.0]),
    }
    merged = merged_rank_score(scores, ("a", "b", "c"))
    assert merged.shape == (3,)
    assert merged[0] == pytest.approx(1 / 3)
    assert merged[1] == pytest.approx(2 / 3)
    assert merged[2] == pytest.approx(0.5)


def test_a_candidate_middling_on_all_three_lands_in_the_middle() -> None:
    scores = {
        "a": np.array([1.0, 2.0, 3.0]),
        "b": np.array([3.0, 2.0, 1.0]),
        "c": np.array([10.0, 20.0, 30.0]),
    }
    assert merged_rank_score(scores, ("a", "b", "c"))[1] == pytest.approx(0.5)


def test_sorting_by_merged_changes_the_order_but_not_the_candidate_set() -> None:
    """정렬은 순서만 바꾼다. 후보 집합이나 등급 집계가 달라지면 다른 것을 센 것이다."""
    library = _library([
        ("PHENOL", "Oc1ccccc1"), ("ANISOLE", "COc1ccccc1"),
        ("CYCLOHEXANOL", "OC1CCCCC1"), ("BENZOIC ACID", "OC(=O)c1ccccc1"),
        ("ANILINE", "Nc1ccccc1"),
    ])
    core = scan_alternatives(SALICYLIC_ACID, library, with_pharmacophore=True,
                             sort_by=SORT_CORE_FIRST)
    merged = scan_alternatives(SALICYLIC_ACID, library, with_pharmacophore=True,
                               sort_by=SORT_MERGED)
    assert set(core.frame["inchikey"]) == set(merged.frame["inchikey"])
    assert core.grade_counts == merged.grade_counts
    assert core.kept == merged.kept
    # 합친 순위는 내림차순이어야 한다.
    assert list(merged.frame["merged_score"]) == sorted(merged.frame["merged_score"], reverse=True)


def test_the_merged_score_is_absent_unless_the_pharmacophore_signal_is_on() -> None:
    """세 기준이 다 있어야 합칠 수 있다. 없는데 값이 있으면 무언가를 지어낸 것이다."""
    library = _library([("PHENOL", "Oc1ccccc1"), ("ANISOLE", "COc1ccccc1")])
    off = scan_alternatives(SALICYLIC_ACID, library)
    assert "merged_score" not in off.frame.columns


def _merged_library():
    """걸러지는 분자가 세 신호 중 어느 것에서도 최하위가 아닌 라이브러리.

    이 조건이 없으면 이 성질을 어겨도 테스트가 통과한다. 걸러지는 분자의 점수가
    이미 최하위면 그 자리를 0으로 덮어써도 남은 후보의 순서가 안 바뀌기 때문이다.
    실제로 파마코포어만 필터 뒤에서 채워지던 시기에, 4종짜리 옛 픽스처는 이
    결함을 그대로 통과시켰다.
    """
    return _library([
        ("PHENOL", "Oc1ccccc1"), ("ANISOLE", "COc1ccccc1"),
        ("CYCLOHEXANOL", "OC1CCCCC1"), ("BENZOIC ACID", "OC(=O)c1ccccc1"),
        ("GLYCERIN", "OCC(O)CO"), ("4-HYDROXYBENZOIC ACID", "OC(=O)c1ccc(O)cc1"),
        ("GALLIC ACID", "OC(=O)c1cc(O)c(O)c(O)c1"), ("UREA", "NC(N)=O"),
        ("CATECHOL", "Oc1ccccc1O"), ("ACETIC ACID", "CC(O)=O"),
    ])


@pytest.mark.parametrize(
    ("label", "kwargs"),
    [
        ("유사도 문턱", {"min_similarity": 0.2}),
        ("핵심구조만", {"require_core": True}),
    ],
)
def test_the_merged_rank_is_taken_over_the_whole_library_not_the_filtered_rows(
    label: str, kwargs: dict
) -> None:
    """걸러진 뒤에 순위를 매기면 필터를 켰다 껐다 할 때 같은 후보의 순위가 바뀐다.

    세 신호가 모두 필터 **이전에** 채워져야 성립한다. 하나만 뒤에서 채워지면
    걸러진 후보의 그 칸이 0으로 남고, 합친 순위가 그 0을 실제 값으로 알고
    백분위를 매긴다.
    """
    library = _merged_library()
    everything = scan_alternatives(SALICYLIC_ACID, library, with_pharmacophore=True,
                                   sort_by=SORT_MERGED)
    narrowed = scan_alternatives(SALICYLIC_ACID, library, with_pharmacophore=True,
                                 sort_by=SORT_MERGED, **kwargs)
    assert not narrowed.frame.empty
    assert len(narrowed.frame) < len(everything.frame), (
        f"{label} 필터가 아무것도 거르지 않으면 이 테스트는 아무것도 재지 않는다"
    )
    wide = dict(zip(everything.frame["inchikey"], everything.frame["merged_score"], strict=True))
    for key, score in zip(narrowed.frame["inchikey"], narrowed.frame["merged_score"], strict=True):
        assert wide[key] == score, f"{label}을 켰다고 같은 후보의 합친 점수가 달라지면 안 됩니다"


def test_candidates_with_identical_signals_get_identical_merged_scores() -> None:
    """세 신호가 완전히 같은 두 후보가 수십 위 떨어져 나오면 설명할 근거가 없다.

    서수 순위로 매기면 동점의 승자를 라이브러리 parquet 행 순서가 정한다.
    """
    values = np.array([0.5, 0.5, 0.5, 0.9, 0.1])
    ranked = percentile_rank(values)
    assert ranked[0] == ranked[1] == ranked[2]
    assert ranked[3] > ranked[0] > ranked[4]


def test_percentile_rank_is_unchanged_when_nothing_is_tied() -> None:
    """동점 처리를 바꾼 것이 값이 다른 경우까지 건드리면 안 된다."""
    values = np.array([3.0, 1.0, 2.0, 4.0])
    assert list(percentile_rank(values)) == pytest.approx([2 / 3, 0.0, 1 / 3, 1.0])


def test_the_evaluation_harness_ranks_with_the_product_function() -> None:
    """평가가 잰 순위와 화면이 내는 순위가 달라지면, 그 차이는 화면을 봐서는 모른다."""
    import eval_alternative_criteria as harness

    assert harness.merged_rank_score is merged_rank_score


# ---------------------------------------------------------- 3D 파마코포어


@pytest.fixture(scope="module")
def three_d_query():
    query = _standardize_mol(SALICYLIC_ACID)
    return query, prepare_query_3d(query)


def test_a_pair_with_almost_no_common_substructure_is_unavailable_not_zero(three_d_query) -> None:
    """정렬할 공통 원자가 3개 미만이면 잴 수 없다.

    0으로 두면 "3D에서 전혀 안 맞았다"로 읽히는데, 실제로는 재지 못한 것이다.
    """
    query, prepared = three_d_query
    match = three_d_match(query, _standardize_mol("C"), prepared)
    assert match.status == "unavailable"
    assert match.feature_recall is None
    assert match.feature_rmsd is None


def test_an_exhausted_budget_is_reported_as_such_not_as_a_bad_score(three_d_query) -> None:
    query, prepared = three_d_query
    match = three_d_match(query, _standardize_mol("CCO"), prepared, budget=McsBudget(0.0))
    assert match.status == "budget_exhausted"
    assert match.feature_recall is None


def test_three_d_is_reproducible_because_the_seed_is_fixed(three_d_query) -> None:
    """컨포머 생성은 난수를 쓴다. 실행마다 값이 달라지면 화면의 숫자를 인용할 수 없다."""
    query, _ = three_d_query
    candidate = _standardize_mol("CCCCC(CC)COC(=O)c1ccccc1O")
    first = three_d_match(query, candidate, prepare_query_3d(query))
    second = three_d_match(query, candidate, prepare_query_3d(query))
    assert first.feature_recall == second.feature_recall
    assert first.feature_rmsd == second.feature_rmsd


def test_only_the_top_rows_are_rescored_and_the_rest_say_so() -> None:
    """3D는 전수에 못 쓴다. 보지 않은 행을 빈칸으로 두면 나쁜 점수로 읽힌다."""
    library = _library([
        ("PHENOL", "Oc1ccccc1"), ("ANISOLE", "COc1ccccc1"),
        ("CYCLOHEXANOL", "OC1CCCCC1"), ("BENZOIC ACID", "OC(=O)c1ccccc1"),
    ])
    frame = scan_alternatives(SALICYLIC_ACID, library).frame
    out = annotate_three_d(frame, SALICYLIC_ACID, limit=2)
    assert list(out["three_d_status"])[2:] == ["not_rescored"] * (len(out) - 2)
    for value in list(out["three_d_recall"])[2:]:
        assert value is None


def test_the_three_d_columns_hold_none_not_nan() -> None:
    """pandas가 float 열로 만들면 None이 NaN이 되고, NaN은 참이라 조용히 새어 나간다.

    이 저장소는 같은 함정에 이미 두 번 물렸다.
    """
    import json
    import math

    library = _library([("PHENOL", "Oc1ccccc1"), ("ANISOLE", "COc1ccccc1")])
    frame = scan_alternatives(SALICYLIC_ACID, library).frame
    out = annotate_three_d(frame, SALICYLIC_ACID, limit=1)
    values = list(out["three_d_recall"]) + list(out["three_d_rmsd"])
    assert not any(isinstance(v, float) and math.isnan(v) for v in values)
    assert any(v is None for v in values)
    json.dumps([None if v is None else float(v) for v in values], allow_nan=False)


def test_an_empty_frame_still_gets_the_three_d_columns() -> None:
    """호출자가 열의 존재를 따로 확인하지 않아도 되게, 빈 표에도 붙인다."""
    out = annotate_three_d(pd.DataFrame(columns=["canonical_smiles"]), SALICYLIC_ACID)
    for column in ("three_d_status", "three_d_recall", "three_d_rmsd"):
        assert column in out.columns


def test_an_unreadable_candidate_does_not_stop_the_rescoring() -> None:
    library = _library([("PHENOL", "Oc1ccccc1"), ("ANISOLE", "COc1ccccc1")])
    frame = scan_alternatives(SALICYLIC_ACID, library).frame.copy()
    frame.loc[0, "canonical_smiles"] = "NOT_A_SMILES"
    out = annotate_three_d(frame, SALICYLIC_ACID, limit=len(frame))
    assert out.loc[0, "three_d_status"] == "unavailable"
    assert "ok" in set(out["three_d_status"])


# ------------------------------------------------------- 극성 근접 정렬


def test_polarity_is_a_declared_sort_mode() -> None:
    assert SORT_POLARITY in SORT_MODES


def test_polarity_ordering_ignores_structure_entirely() -> None:
    """구조를 보지 않는다는 것이 이 정렬의 정의다. 다른 순서가 나와야 한다.

    측정에서 이 순서가 이긴 곳은 향료·방향 분류 하나뿐이고, 출처 인용 정답표에서는
    상위 4위 안에도 들지 못했다. 그래서 기본값이 아니다.
    """
    library = _library([
        ("PHENOL", "Oc1ccccc1"), ("ANISOLE", "COc1ccccc1"),
        ("BENZOIC ACID", "OC(=O)c1ccccc1"), ("DECANE", "CCCCCCCCCC"),
    ])
    core = scan_alternatives(SALICYLIC_ACID, library, sort_by=SORT_CORE_FIRST).frame
    polar = scan_alternatives(SALICYLIC_ACID, library, sort_by=SORT_POLARITY).frame
    assert set(core["inchikey"]) == set(polar["inchikey"])
    assert list(polar["polarity_closeness"]) == sorted(polar["polarity_closeness"], reverse=True)


def test_every_sort_mode_keeps_the_same_candidate_set_and_grade_counts() -> None:
    """정렬은 순서만 바꾼다. 집계가 달라지면 다른 것을 센 것이다."""
    library = _library([
        ("PHENOL", "Oc1ccccc1"), ("ANISOLE", "COc1ccccc1"),
        ("CYCLOHEXANOL", "OC1CCCCC1"), ("BENZOIC ACID", "OC(=O)c1ccccc1"),
    ])
    scans = {
        mode: scan_alternatives(SALICYLIC_ACID, library, with_pharmacophore=True, sort_by=mode)
        for mode in SORT_MODES
    }
    reference = scans[SORT_CORE_FIRST]
    for mode, scan in scans.items():
        assert set(scan.frame["inchikey"]) == set(reference.frame["inchikey"]), mode
        assert scan.grade_counts == reference.grade_counts, mode
        assert scan.kept == reference.kept, mode
