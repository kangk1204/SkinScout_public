"""Contract tests for the alternative-ingredient endpoint of the Workbench.

This screen answers the one question the project is funded to answer - which
*registered cosmetic ingredients* keep the active core structure of a starting
compound - so two properties matter more than the ranking itself:

* it must survive missing measurements. Most CosIng structures have never been
  assayed, and pandas hands those back as NaN. NaN is truthy, so `value or 0`
  let a NaN reach `int()` and killed the entire request with "cannot convert
  float NaN to integer" - one unmeasured candidate in the list was enough.
* it must never turn "nobody measured this" into "this is inactive", and it
  must never turn "the measured library is missing" into "nobody measured
  this". Those are three different sentences and the screen has to keep them
  apart. The same NaN trap sits in the label columns - `str(nan or "x")` is
  `"nan"`, a category no badge and no CSV column knows - so the string path is
  pinned here as well as the numeric one.

The fixtures build a three-molecule ingredient library in memory rather than
reading the real artifacts: `data/cosing/cosing.parquet` and the 272 MB
similarity index are gitignored, and a test that skips when they are absent
would stop guarding the NaN regression exactly where it is least observed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from workbench import server  # noqa: E402

from alternative_ingredients import (  # noqa: E402
    MORGAN_GENERATOR,
    IngredientLibrary,
    _packed,
)
from build_activity_retrieval_index import _standardize_mol  # noqa: E402
from rdkit import Chem  # noqa: E402


KOJIC_ACID = "OCC1=CC(=O)C(O)=CO1"

# Kojic acid plus one pyranone relative and one unrelated polyol. The relative
# is the case the ranking exists for: it keeps the core ring while Tanimoto
# similarity stays low, so it must appear even with no similarity floor.
_LIBRARY_ENTRIES = (
    ("KOJIC ACID", KOJIC_ACID, "Skin conditioning; Bleaching"),
    ("MALTOL", "Cc1occc(=O)c1O", "Masking"),
    ("GLYCERIN", "OCC(O)CO", "Humectant; Solvent"),
)


def _fake_library() -> IngredientLibrary:
    """A real `IngredientLibrary` over three molecules, built without parquet."""
    rows: list[dict[str, object]] = []
    molecules: list[Chem.Mol] = []
    packed: list[np.ndarray] = []
    for name, smiles, functions in _LIBRARY_ENTRIES:
        molecule = _standardize_mol(smiles)
        inchikey = Chem.MolToInchiKey(molecule)
        rows.append(
            {
                "inci_name": name,
                "synonyms": [],
                "cas": "",
                "functions": functions,
                "canonical_smiles": Chem.MolToSmiles(molecule),
                "inchikey": inchikey,
                "skeleton": inchikey[:14],
                "registered_names": 1,
            }
        )
        molecules.append(molecule)
        packed.append(_packed(MORGAN_GENERATOR.GetFingerprint(molecule)))
    fingerprints = np.vstack(packed)
    return IngredientLibrary(
        frame=pd.DataFrame(rows),
        molecules=molecules,
        fingerprints=fingerprints,
        popcounts=np.unpackbits(fingerprints, axis=1).sum(1, dtype=np.int32),
        source=Path("in-memory-test-library"),
        unparsed=0,
        registered_entries=len(_LIBRARY_ENTRIES),
        resolved_entries=len(_LIBRARY_ENTRIES),
    )


_INDEX_MISSING = (
    "활성 측정 라이브러리가 준비되지 않았습니다. "
    "scripts/build_similarity_index.py 를 먼저 실행하세요."
)
_LIBRARY_MISSING = (
    "화장품 원료 라이브러리가 준비되지 않았습니다. "
    "scripts/stage0_cosing.py 로 data/cosing/cosing.parquet 을 먼저 만드세요."
)


def _raise_index_missing() -> object:
    raise ValueError(_INDEX_MISSING)


def _raise_library_missing() -> object:
    raise ValueError(_LIBRARY_MISSING)


@pytest.fixture
def offline_libraries(monkeypatch: pytest.MonkeyPatch) -> None:
    """No 272 MB index, no parquet: the in-memory library and nothing else."""
    monkeypatch.setattr(server, "_ingredient_library", _fake_library)
    monkeypatch.setattr(server, "_similarity_index", _raise_index_missing)


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        (float("nan"), 0),
        (None, 0),
        ("abc", 0),
        (3.7, 3),
        (True, 1),
        (np.int64(4), 4),
        (2, 2),
    ),
)
def test_int_or_zero_absorbs_every_shape_pandas_can_hand_back(
    value: object,
    expected: int,
) -> None:
    """NaN is the regression: it is truthy, so `value or 0` passed it to int()."""
    result = server._int_or_zero(value)

    assert result == expected
    assert isinstance(result, int)


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        (float("nan"), 0.0),
        (None, 0.0),
        ("abc", 0.0),
        (3.7, 3.7),
        (True, 1.0),
        (np.float64(0.25), 0.25),
    ),
)
def test_float_or_zero_absorbs_every_shape_pandas_can_hand_back(
    value: object,
    expected: float,
) -> None:
    result = server._float_or_zero(value)

    assert result == pytest.approx(expected)
    assert isinstance(result, float)
    assert result == result  # never NaN, or json.dumps(allow_nan=False) breaks


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        (20, 20),
        (0, 1),
        (999, 50),
        (-4, 1),
        (3.9, 3),
        ("7", 7),
        ("abc", 20),
        (None, 20),
        (float("nan"), 20),
        (float("inf"), 50),
        (float("-inf"), 1),
    ),
)
def test_bounded_int_clamps_and_falls_back_on_garbage(
    value: object,
    expected: int,
) -> None:
    """`{"limit": Infinity}` is legal JSON and int() raises on it.

    Left uncaught it aborts the request before any response is written, so the
    screen shows a hang rather than a table.
    """
    assert server._bounded_int(value, 20, 1, server.MAX_ALTERNATIVE_ROWS) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        (0.35, 0.35),
        (-1.0, 0.0),
        (2.5, 1.0),
        ("0.5", 0.5),
        ("abc", 0.0),
        (None, 0.0),
        (float("nan"), 0.0),
        (float("inf"), 1.0),
        (float("-inf"), 0.0),
    ),
)
def test_bounded_float_clamps_and_falls_back_on_garbage(
    value: object,
    expected: float,
) -> None:
    assert server._bounded_float(value, 0.0, 0.0, 1.0) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        (float("nan"), None),
        (None, None),
        (7.4567, 7.46),
        (np.float64(6.1), 6.1),
        (0, 0.0),
    ),
)
def test_round_or_none_keeps_missing_measurements_out_of_the_payload(
    value: object,
    expected: float | None,
) -> None:
    """A missing pActivity has to leave as `null`, not as NaN or as 0.

    NaN would break `json.dumps(allow_nan=False)`; 0 would read as "measured,
    and inactive".
    """
    assert server._round_or_none(value) == expected


def _unmeasured_row(**overrides: object) -> dict[str, object]:
    """One CosIng candidate the activity index has never seen - the common case."""
    row: dict[str, object] = {
        "rank": np.int64(1),
        "inci_name": "MALTOL",
        "synonyms": [],
        "cas": "118-71-8",
        "functions": "A; B; C",
        "canonical_smiles": "Cc1occc(=O)c1O",
        "inchikey": "MMJDYCPRGHFVRT-UHFFFAOYSA-N",
        "similarity": np.float64(0.209),
        "core_grade": "retained",
        "core_label_ko": "핵심 유지",
        "core_coverage": 0.8,
        "core_share": 0.75,
        "core_basis": "mcs",
        "scaffold_match": np.bool_(True),
        "is_query": False,
        # `attach_measured_evidence` builds these from a list of ints and Nones,
        # and pandas turns that into a float64 column - so an unmeasured
        # candidate arrives as NaN, not as None.
        "measured_target_count": float("nan"),
        "measured_best_pactivity": float("nan"),
        "measured_top_targets": float("nan"),
        "shared_targets": float("nan"),
        "measured_match": float("nan"),
    }
    row.update(overrides)
    return row


def test_alternative_row_stays_json_serialisable_when_nothing_was_measured() -> None:
    row = server._alternative_row(_unmeasured_row())

    # allow_nan=False is the point: the stdlib default would emit bare `NaN`,
    # which is not JSON and which the browser refuses to parse.
    assert json.loads(json.dumps(row, allow_nan=False)) == row
    assert row["rank"] == 1
    assert row["functions"] == ["A", "B", "C"]
    assert row["measured_target_count"] == 0
    assert row["measured_best_pactivity"] is None
    assert row["measured_top_targets"] == []
    assert row["shared_targets"] == []
    # "이 라이브러리에서 못 찾았다"이지, "재 봤는데 활성이 없다"가 아니다.
    assert row["measured_evidence"] == "not_measured"


def test_alternative_row_labels_a_nan_evidence_column_as_not_measured() -> None:
    """The NaN trap is in the string columns too, and `or` does not close it.

    `str(value or "not_measured")` returns `"nan"` for a NaN, because NaN is
    truthy - and `"nan"` matches no label the screen or the CSV knows, so the
    row shows a blank badge instead of "측정 기록 없음".
    """
    row = server._alternative_row(
        _unmeasured_row(
            measured_evidence=float("nan"),
            core_grade=float("nan"),
            core_label_ko=float("nan"),
            inci_name=float("nan"),
        )
    )

    assert row["measured_evidence"] == "not_measured"
    assert row["core_grade"] == "unknown"
    assert row["core_label_ko"] == ""
    assert row["inci_name"] == ""
    assert row["measured_match"] == ""
    assert "nan" not in json.dumps(row, allow_nan=False, ensure_ascii=False)


def test_alternative_row_separates_unmeasured_from_an_absent_index() -> None:
    """Without the index every row *looks* unmeasured; that is not a finding."""
    row = server._alternative_row(_unmeasured_row(), evidence_available=False)

    assert row["measured_evidence"] == "evidence_unavailable"
    assert row["measured_evidence"] != "not_measured"
    json.dumps(row, allow_nan=False)


@pytest.mark.parametrize(
    ("smiles", "message"),
    (
        ("", "SMILES를 입력하세요"),
        ("   ", "SMILES를 입력하세요"),
        ("C" * (server.MAX_SMILES_LENGTH + 1), "너무 깁니다"),
        ("not a molecule", "RDKit이 이 SMILES를 읽지 못했습니다"),
        ("c1ccccc", "RDKit이 이 SMILES를 읽지 못했습니다"),
    ),
)
def test_compound_alternatives_rejects_bad_input_in_korean(
    smiles: str,
    message: str,
    offline_libraries: None,
) -> None:
    """A wet-lab reader gets a sentence they can act on, not a stack trace."""
    with pytest.raises(ValueError, match=message):
        server.compound_alternatives({"smiles": smiles})


def test_compound_alternatives_rejects_input_beyond_the_screen_scope(
    offline_libraries: None,
) -> None:
    """The size gate fires before the scaffold work, or a reject costs minutes."""
    from alternative_ingredients import MAX_QUERY_HEAVY_ATOMS

    with pytest.raises(ValueError, match="너무 큽니다"):
        server.compound_alternatives({"smiles": "C" * (MAX_QUERY_HEAVY_ATOMS + 1)})


def test_alternatives_answer_from_cosing_alone_when_the_index_is_missing(
    offline_libraries: None,
) -> None:
    """A missing activity index must not delete the ingredient list too.

    The two libraries fail independently; erasing the half that is ready would
    leave the reader with nothing to act on and no reason why.
    """
    result = server.compound_alternatives({"smiles": KOJIC_ACID})

    assert "ingredients" in result
    ingredients = result["ingredients"]
    assert ingredients["rows"], "the CosIng half was ready and must still render"
    assert ingredients["scanned"] == len(_LIBRARY_ENTRIES)
    assert ingredients["library_size"] == len(_LIBRARY_ENTRIES)
    assert "unavailable" not in ingredients
    # "다 봤는데 없다"와 "안 봤다"를 가르는 문장이 비어 있으면 안 된다.
    assert ingredients["summary"]
    assert ingredients["note"]
    assert ingredients["query_measured"] is False

    # The reason belongs in the measured half, and the request stands.
    assert result["measured"]["rows"] == []
    assert _INDEX_MISSING in result["measured"]["unavailable"]
    assert ingredients["evidence_available"] is False
    assert _INDEX_MISSING in ingredients["evidence_unavailable_reason"]
    for row in ingredients["rows"]:
        assert row["measured_evidence"] == "evidence_unavailable"

    # The whole envelope has to survive the same serialisation the handler uses.
    json.dumps(result, allow_nan=False)


def test_alternatives_rank_by_core_retention_and_flag_the_query_itself(
    offline_libraries: None,
) -> None:
    """Ranking is by retention grade first - similarity only breaks ties.

    A known ingredient retrieves itself, so the row that *is* the query is
    marked rather than silently counted as a discovery.
    """
    result = server.compound_alternatives({"smiles": KOJIC_ACID})
    rows = result["ingredients"]["rows"]

    assert [row["rank"] for row in rows] == list(range(1, len(rows) + 1))
    assert rows[0]["core_grade"] == "identical"
    assert rows[0]["is_query"] is True
    assert result["query"]["heavy_atoms"] == 10
    assert result["query"]["scaffold_smiles"]

    excluded = server.compound_alternatives({"smiles": KOJIC_ACID, "exclude_self": True})
    assert all(not row["is_query"] for row in excluded["ingredients"]["rows"])


def test_alternatives_honour_the_row_ceiling(offline_libraries: None) -> None:
    """A per-candidate MCS runs on every row, so an unbounded limit is a stall."""
    result = server.compound_alternatives({"smiles": KOJIC_ACID, "limit": 1})

    assert len(result["ingredients"]["rows"]) == 1
    # The count of everything compared survives the truncation, so "we looked
    # at three and showed one" stays readable.
    assert result["ingredients"]["scanned"] == len(_LIBRARY_ENTRIES)


def test_alternatives_fail_closed_when_neither_library_is_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No silent empty table: with nothing to search, say so and refuse."""
    monkeypatch.setattr(server, "_ingredient_library", _raise_library_missing)
    monkeypatch.setattr(server, "_similarity_index", _raise_index_missing)

    with pytest.raises(ValueError, match="화장품 원료 라이브러리가 준비되지 않았습니다"):
        server.compound_alternatives({"smiles": KOJIC_ACID})


def test_the_pharmacophore_fields_are_none_when_the_signal_was_not_asked_for() -> None:
    """켜지 않았는데 0.0이 나가면 "특징이 하나도 안 겹친다"로 읽힌다.

    실제로는 재 보지 않은 것이고, 그 둘은 다른 말이다. 다른 결측 칸을 0으로
    채우지 않은 것과 같은 이유다.
    """
    row = server._alternative_row(_unmeasured_row())
    assert row["pharm_similarity"] is None
    assert row["pharm_recall"] is None
    assert row["pharm_precision"] is None


def test_the_pharmacophore_fields_survive_serialisation_when_present() -> None:
    payload = dict(_unmeasured_row())
    payload.update({"pharm_similarity": 0.42, "pharm_recall": 1.0, "pharm_precision": 0.5})
    row = server._alternative_row(payload)
    assert json.loads(json.dumps(row, allow_nan=False))["pharm_similarity"] == 0.42
    assert row["pharm_recall"] == 1.0


def test_the_flag_parser_does_not_turn_the_signal_on_for_the_string_false() -> None:
    """`"false"`도 문자열이라 bool()에서는 참이다. 3.7초짜리 계산이 조용히 켜진다."""
    assert server._flag("false") is False
    assert server._flag("0") is False
    assert server._flag("") is False
    assert server._flag("true") is True
    assert server._flag(True) is True


# ------------------------------------------------- 파마코포어 백엔드가 없을 때


@pytest.mark.parametrize(
    ("label", "error"),
    (
        ("fdef 파일 없음", OSError("BaseFeatures.fdef를 열 수 없습니다")),
        ("5.6b 모듈 없음", ImportError("No module named 'discover_substitutes'")),
        ("백엔드가 None", AttributeError("'NoneType' object has no attribute 'GetFeatures'")),
    ),
)
def test_a_broken_pharmacophore_backend_does_not_erase_the_whole_screen(
    offline_libraries: None, monkeypatch: pytest.MonkeyPatch, label: str, error: Exception
) -> None:
    """세 번째 신호도 index/library 와 같이 따로 실패해야 한다.

    정렬을 '세 기준 합친 순위'로 바꾸면 서버가 파마코포어를 강제로 켠다. 그
    백엔드가 열리지 않을 때 요청 전체가 죽으면, 파마코포어가 전혀 필요 없는
    측정 근거 표까지 함께 사라진다 - 사용자는 드롭다운만 건드렸는데 두 표가
    다 비고 상태줄에 영어 오류가 남았다.
    """
    import alternative_ingredients as ai

    def boom(*args: object, **kwargs: object) -> object:
        raise error

    monkeypatch.setattr(ai, "_pharmacophore_backend", boom)
    result = server.compound_alternatives(
        {"smiles": KOJIC_ACID, "limit": 5, "sort_by": "merged"}
    )

    ingredients = result["ingredients"]
    assert ingredients["rows"], f"{label}: 후보 표가 비면 안 됩니다"
    # 요청한 정렬을 낼 수 없으면 되돌리되, 되돌렸다는 사실을 남긴다.
    assert ingredients["pharmacophore"] is False
    assert ingredients["sort_by"] == "core"
    assert ingredients["pharmacophore_unavailable"], "사유가 비면 화면이 설명할 수 없습니다"
    assert json.loads(json.dumps(result, allow_nan=False))


def test_a_working_pharmacophore_run_reports_no_failure_reason(
    offline_libraries: None,
) -> None:
    """정상일 때 사유 칸이 채워져 있으면 화면이 없는 고장을 알린다."""
    result = server.compound_alternatives(
        {"smiles": KOJIC_ACID, "limit": 5, "sort_by": "merged"}
    )
    assert result["ingredients"]["pharmacophore"] is True
    assert result["ingredients"]["sort_by"] == "merged"
    assert result["ingredients"]["pharmacophore_unavailable"] is None


def test_a_failure_without_the_pharmacophore_asked_for_is_not_swallowed(
    offline_libraries: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """파마코포어를 켜지 않은 요청의 실패까지 삼키면 진짜 고장이 숨는다."""
    import alternative_ingredients as ai

    def boom(*args: object, **kwargs: object) -> object:
        raise ValueError("스캔이 깨졌습니다")

    monkeypatch.setattr(ai, "core_retention", boom)
    with pytest.raises(ValueError, match="스캔이 깨졌습니다"):
        server.compound_alternatives({"smiles": KOJIC_ACID, "limit": 5})


# ------------------------------------------------- 배합목적이 신고되지 않은 물질
#
# CosIng 은 화장품 원료만 담은 목록이 아니다. 금지·제한 물질도 규제 목적으로
# 등재돼 있고, 그런 항목은 신고 배합목적이 비어 있다 - 실측으로 7,484종 중
# 1,215종(16.2%)이고 그중에는 메타조신, 헥사클로로 살충제, 아세틸콜린이 있다.
# 그것들을 대체소재 후보로 점수와 함께 내놓으면 읽는 쪽은 쓸 수 있는 원료로 읽는다.
#
# 실제로 그랬다: 트리클로산의 2위가 `Nitrofen`(금지 제초제), 글리콜산의 2위가
# `Methoxyacetic acid`(생식독성)였다.

_UNDECLARED_ENTRIES = _LIBRARY_ENTRIES + (
    # 코직산과 같은 피라논 고리라 구조로는 반드시 올라온다. 배합목적은 비어 있다.
    ("ALLOMALTOL", "Cc1cc(=O)c(O)co1", ""),
)


def _library_with_an_undeclared_entry() -> IngredientLibrary:
    """`_fake_library` 가 읽는 전역을 잠깐 바꿔 4번째 항목을 넣는다."""
    module = sys.modules[__name__]
    original = module._LIBRARY_ENTRIES
    module._LIBRARY_ENTRIES = _UNDECLARED_ENTRIES
    try:
        return _fake_library()
    finally:
        module._LIBRARY_ENTRIES = original


@pytest.fixture()
def library_with_undeclared(monkeypatch: pytest.MonkeyPatch) -> None:
    built = _library_with_an_undeclared_entry()
    monkeypatch.setattr(server, "_ingredient_library", lambda: built)
    monkeypatch.setattr(server, "_similarity_index", _raise_index_missing)


def _names(result: dict) -> list[str]:
    return [row["inci_name"] for row in result["ingredients"]["rows"]]


def test_a_substance_with_no_declared_cosmetic_purpose_is_hidden_by_default(
    library_with_undeclared: None,
) -> None:
    result = server.compound_alternatives({"smiles": KOJIC_ACID, "limit": 10})

    assert "ALLOMALTOL" not in _names(result)
    assert result["ingredients"]["undeclared_hidden"] == 1
    assert result["ingredients"]["include_undeclared"] is False


def test_the_hidden_substance_comes_back_when_asked_for(
    library_with_undeclared: None,
) -> None:
    """숨기는 것이 지우는 것은 아니다. 구조가 금지 물질과 닮았다는 사실도 정보다."""
    result = server.compound_alternatives(
        {"smiles": KOJIC_ACID, "limit": 10, "include_undeclared": True}
    )

    assert "ALLOMALTOL" in _names(result)
    assert result["ingredients"]["undeclared_hidden"] == 0
    assert result["ingredients"]["include_undeclared"] is True


def test_every_row_says_whether_a_cosmetic_purpose_was_declared(
    library_with_undeclared: None,
) -> None:
    """빈 칸은 "아직 안 적혔나 보다"로 읽힌다. 값으로 나가야 화면이 구분해 그린다."""
    result = server.compound_alternatives(
        {"smiles": KOJIC_ACID, "limit": 10, "include_undeclared": True}
    )
    declared = {row["inci_name"]: row["cosmetic_use_declared"] for row in result["ingredients"]["rows"]}

    assert declared["ALLOMALTOL"] is False
    assert all(value for name, value in declared.items() if name != "ALLOMALTOL")


def test_hiding_them_does_not_move_the_scores_of_the_candidates_that_remain(
    library_with_undeclared: None,
) -> None:
    """거르기는 점수를 매긴 **뒤에** 해야 한다.

    앞에서 거르면 백분위의 기준 집합이 바뀌어, 같은 후보가 이 체크박스 하나로
    다른 종합 점수를 받는다. 화면의 숫자가 무엇에 대한 값인지 말할 수 없게 된다.
    """
    hidden = server.compound_alternatives({"smiles": KOJIC_ACID, "limit": 10})
    shown = server.compound_alternatives(
        {"smiles": KOJIC_ACID, "limit": 10, "include_undeclared": True}
    )
    hidden_scores = {r["inci_name"]: r["score_total"] for r in hidden["ingredients"]["rows"]}
    shown_scores = {r["inci_name"]: r["score_total"] for r in shown["ingredients"]["rows"]}

    common = set(hidden_scores) & set(shown_scores)
    assert common, "비교할 공통 후보가 없습니다"
    for name in common:
        assert hidden_scores[name] == shown_scores[name], name


# --------------------------------------------------------------- 번호와 순위
#
# 화면의 번호는 "이 표에서 몇 번째"이고, 스캔이 매긴 순위는 "핵심구조 유지 순으로
# 몇 위"다. 둘을 한 칸에 합치면 둘 다 틀린 값이 된다:
#  - 미신고 물질을 걸러내면 번호에 구멍이 난다(1, 3, 4, ...)
#  - 종합 점수나 안전 순으로 정렬하면 핵심구조 순위가 그대로 찍힌다.
#    실측으로 안전 순 첫 화면이 1977·5773·5663... 이었다.


def test_the_number_on_screen_counts_from_one_even_after_filtering(
    library_with_undeclared: None,
) -> None:
    result = server.compound_alternatives({"smiles": KOJIC_ACID, "limit": 10})
    ranks = [row["rank"] for row in result["ingredients"]["rows"]]

    assert ranks == list(range(1, len(ranks) + 1)), "걸러낸 자리에 구멍이 남았습니다"


def test_the_scan_rank_survives_next_to_it(library_with_undeclared: None) -> None:
    """구멍을 메우느라 원래 순위를 버리면 "핵심구조로는 몇 위였나"를 잃는다."""
    result = server.compound_alternatives({"smiles": KOJIC_ACID, "limit": 10})

    for row in result["ingredients"]["rows"]:
        assert isinstance(row["original_rank"], int)
        assert row["original_rank"] >= 1


def test_sorting_by_a_post_scan_criterion_does_not_print_the_structure_rank(
    offline_libraries: None,
) -> None:
    """정렬을 바꾸면 화면 번호는 1부터, 핵심구조 순위는 그 순서를 따르지 않는다."""
    result = server.compound_alternatives(
        {"smiles": KOJIC_ACID, "limit": 10, "sort_by": "safety"}
    )
    rows = result["ingredients"]["rows"]
    assert [row["rank"] for row in rows] == list(range(1, len(rows) + 1))
