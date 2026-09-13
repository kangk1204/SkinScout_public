"""The promoted recipe is wired and deliberately not switched on.

Measured through the run path, it improves the panel in aggregate (+15 points of
Top10 and Top30) and loses tyrosinase on both whitening compounds: alpha-arbutin
2 -> 391, kojic acid 1 -> 87. That is not incidental. `union_p6_consensus`
assigns zero weight to `max_union5`, so it discards evidence below pActivity 6,
and tyrosinase evidence is exactly there - 181 edges, 28 of them at 6 or above,
median max pActivity 5.11. Cosmetic actives are weak by nature; a recipe selected
on a drug-like benchmark systematically demotes them.

These tests keep the switch off and keep the reason attached to it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

NOTE = ROOT / "docs" / "RECIPE_RUNPATH_MEASURED_20260831.md"


def _promoted_recipe_path() -> Path:
    """Whatever the config actually points at.

    Hardcoding dev_selection/recipe.json meant these tests kept checking the
    dev-score-selected artifact after the run path was pointed at the
    skin-panel-selected one.
    """
    config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text(encoding="utf-8"))
    return ROOT / config["docking"]["daina_recipe_path"]


RECIPE = _promoted_recipe_path()


def test_recipe_scoring_is_on_against_a_production_index() -> None:
    """Both preconditions were met before this was switched on.

    The panel was re-measured through this exact path (24/46 -> 34/46 at Top30,
    p=0.0063) and it reads a production-role index, not the train-only
    evaluation one.
    """
    config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text(encoding="utf-8"))
    docking = config["docking"]

    assert docking["daina_recipe_scoring"] is True

    # The invariant is the index's declared role, not its directory name: an
    # evaluation index is train-only and would narrow what a researcher sees.
    index = ROOT / docking["daina_recipe_index_dir"] / "manifest.json"
    if index.exists():
        manifest = json.loads(index.read_text(encoding="utf-8"))
        assert manifest["index_role"] == "production", index

    raw = (ROOT / "workflow" / "config.yaml").read_text(encoding="utf-8")
    # The whole recipe block, not a character window: the comments grew when the
    # licence decision and the wiring correction were written down, and a window
    # would have started failing for the wrong reason.
    section = raw.split("daina_evidence_dir", 1)[1].split("min_sources_comprehensive", 1)[0]
    assert "false" in section, "how to revert must be stated"
    # Why it does not read the benchmark index - the mistake that cost arbutin.
    assert "CHEMBL1760462" in section
    # And the licence tier it deliberately stops at.
    assert "share-alike" in section and "GtoPdb" in section


@pytest.mark.skipif(not RECIPE.exists(), reason="no promoted recipe artifact")
def test_the_promoted_recipe_now_uses_threshold_free_evidence() -> None:
    """What changed: the promoted recipe reads evidence below pActivity 6.

    Every earlier candidate cut there, and tyrosinase evidence sits below it -
    which is why they all lost kojic acid. If a future promotion drops this
    weight back to zero, the panel needs re-measuring before it ships.
    """
    from stage3_recipe_scoring import load_promoted_recipe

    recipe = load_promoted_recipe(RECIPE)
    assert recipe.recipe_id == "union_discount_light"
    # The decision to switch rests on the recipe seeing sub-6 evidence. It still
    # does - max_union_nonneg keeps everything max_union_any did except edges
    # whose only measurements are at or below the activity threshold.
    assert recipe.max_union_nonneg > 0.0
    # Both are weighted on purpose: that is the discount that keeps EGCG -> MMP2.
    assert recipe.max_union_any > recipe.max_union_nonneg > 0.0


def test_the_note_records_what_is_lost_not_only_the_aggregate() -> None:
    text = NOTE.read_text(encoding="utf-8")

    # The aggregate that would have justified switching.
    assert "0.522" in text and "0.674" in text
    # And the specific loss that overrode it.
    assert "티로시나제" in text
    assert "391" in text and "87" in text
    # With the mechanism, so it is not read as bad luck.
    assert "max_union5" in text
    assert "5.11" in text


def test_the_note_corrects_the_earlier_artefact_claim() -> None:
    """`PANEL_ARBITRATION_20260831.md` called the v2 block an artefact. The
    statistics and structures were genuinely wrong, but the signal underneath
    looks real, and extending the panel diluted it from 3/15 to 3/22 TYR."""
    text = NOTE.read_text(encoding="utf-8")

    assert "앞선 결론을 정정한다" in text
    assert "3/15" in text and "3/22" in text
    assert "희석" in text


def test_the_note_names_the_next_thing_to_try() -> None:
    """A decision to not ship should say what would change it.

    union_p5_max was the obvious candidate - the only one weighting sub-6
    evidence - and it was measured rather than assumed. It is better than the
    promoted recipe and still not enough, which is why the next step is a
    feature change rather than another choice among these six.
    """
    text = NOTE.read_text(encoding="utf-8")
    assert "union_p5_max" in text
    assert "다음에 할 일" in text
    assert "선정을 다시 돌린다" in text
    # And the ordering fix: dev as the filter, the skin panel as the selector.
    assert "자격 필터" in text and "선정자" in text


ORIGINAL_SIX = (
    "chembl_p5_max",
    "union_p5_max",
    "union_p6_max",
    "union_p6_quality",
    "union_p6_consensus",
    "union_p6_contrast",
)


def test_the_original_six_all_applied_a_potency_threshold() -> None:
    """The diagnosis, asserted against the definitions that produced it.

    Cosmetic actives are weak: kojic acid's nearest tyrosinase analogues sit at
    pActivity 4.3-4.7 once leave-query-out removes kojic acid itself. Every
    feature the first six recipes weight cuts at 5 or 6, so the evidence that
    makes the current run path rank TYR first was invisible to all of them.
    """
    from activity_retrieval_scoring import BASELINE, RECIPES

    for recipe in RECIPES:
        if recipe.recipe_id not in ORIGINAL_SIX:
            continue
        assert recipe.max_union_any == 0.0, recipe.recipe_id
        if recipe.recipe_id == BASELINE.recipe_id:
            continue  # the baseline is special-cased to max_chembl5
        weighted = {
            name
            for name in (
                "max_union5",
                "max_union6",
                "quality_union6",
                "source_consensus6",
                "support_union6",
            )
            if getattr(recipe, name) > 0.0
        }
        assert weighted, recipe.recipe_id
        assert all(name.endswith(("5", "6")) for name in weighted), recipe.recipe_id


def test_the_note_reports_the_full_six_way_comparison() -> None:
    text = NOTE.read_text(encoding="utf-8")

    # union_p5_max is the best on aggregate and the only significant one...
    assert "33/46" in text and "0.035" in text
    # ...and it still loses kojic acid, which is why nothing was switched.
    assert "113" in text
    assert "4.34" in text, "the sub-5 evidence that explains it"
    assert "max_union_any" in text, "the missing feature is named"


def test_the_threshold_free_feature_leaves_older_recipes_untouched() -> None:
    """Adding max_union_any must not move any recipe that predates it.

    All six original recipes carry weight 0.0 on it, so their scores have to be
    bit-identical. If this ever fails, the earlier measurements stop describing
    the recipes they were made on.
    """
    import numpy as np

    from activity_retrieval_scoring import RECIPES, apply_recipe

    rng = np.random.default_rng(20260831)
    features = {
        name: rng.random(64)
        for name in (
            "baseline",
            "max_union_any",
            "max_union_nonneg",
            "max_union5",
            "max_union6",
            "quality_union6",
            "source_consensus6",
            "support_union6",
            "negative_contrast",
        )
    }
    for name in ("max_union_any", "max_union_nonneg"):
        without = dict(features, **{name: np.zeros(64)})
        for recipe in RECIPES:
            if getattr(recipe, name) > 0.0:
                continue
            np.testing.assert_array_equal(
                apply_recipe(features, recipe),
                apply_recipe(without, recipe),
                err_msg=f"{recipe.recipe_id} moved when {name} changed",
            )


def test_the_new_candidates_actually_use_the_new_feature() -> None:
    """Otherwise the addition is inert and nothing was fixed."""
    from activity_retrieval_scoring import RECIPES

    using = [r.recipe_id for r in RECIPES if r.max_union_any > 0.0]
    assert using == [
        "union_any_max",
        "union_any_p6",
        "union_any_consensus",
        # The discount recipes weight it too, at a lower rate than the
        # negative-aware variant - that difference is the discount.
        "union_discount_light",
        "union_discount_even",
        "union_discount_heavy",
    ]


def test_the_frozen_baseline_definition_matches_the_dataclass() -> None:
    """The gate compares them field-for-field; a new field must reach both."""
    import sys
    from dataclasses import asdict

    sys.path.insert(0, str(ROOT / "scripts"))
    from activity_retrieval_scoring import BASELINE
    from validate_activity_retrieval_gate import FROZEN_BASELINE_RECIPE

    assert asdict(BASELINE) == FROZEN_BASELINE_RECIPE


def test_the_note_decomposes_the_arbutin_regression() -> None:
    """The one pair the switch cost has a cause, and it is mostly not the recipe.

    Three effects compose: the production index is missing CHEMBL1760462, the
    0.690-similar tyrosinase inhibitor the ChEMBL mirror has (2 -> ~26); the
    threshold-free feature lifts sixteen other targets past it (26 -> 42); and
    the consensus blend weights >=6 features where arbutin has no potent
    analogue (42 -> 70). Reverting the recipe would not fix the largest one and
    would reintroduce the kojic acid loss.
    """
    text = NOTE.read_text(encoding="utf-8")

    assert "CHEMBL1760462" in text
    assert "0.690" in text and "0.471" in text
    # The three steps, so nobody reads it as a single cause. Matched as a whole
    # decomposition rather than as bare numbers: "26" also occurs inside the
    # 2026-08-31 in the header and "70" inside 0.4706, so the loose version of
    # this assertion passed no matter what the document said.
    assert "2위에서 70위로" in text, "the loss being decomposed"
    assert "순위는 26 → 42다" in text, "step 2: the threshold-free feature"
    assert "42위가 70위가 된다" in text, "step 3: the consensus blend dilutes it"
    # And the conclusion that the fix is coverage, not scoring.
    assert "채점 문제가 아니라 인덱스 커버리지 문제" in text
    assert "레시피를 되돌리는 것은 답이 아니다" in text


def test_the_note_records_that_the_regression_was_fixed_not_accepted() -> None:
    """The arbutin loss was diagnosed to index coverage and then removed.

    Rebuilding the index from the run-path evidence brings back CHEMBL1760462,
    which restores arbutin to rank 2 and - because that analogue is at pActivity
    6.11 - collapses the blend-dilution effect too. On all 46 pairs the final
    discordance is 13:0 and 11:0: nothing is lost.
    """
    text = NOTE.read_text(encoding="utf-8")

    assert "실행 예" not in text  # guard against a stray placeholder
    assert "4,659" in text, "the ChEMBL-only index target count"
    assert "13:0" in text and "11:0" in text, "the zero-loss discordance"
    assert "27/46" in text and "35/46" in text
    assert "원인 1을 고치니 2와 3이 따라 풀렸다" in text
    # And the bug found while building it, since it silently dropped 43% of rows.
    assert "798,900" in text


def test_the_note_records_the_bindingdb_merge_and_the_licence_it_refused() -> None:
    """Adding data is a licence decision as much as a coverage one. GtoPdb's 33
    extra targets would put ODbL / CC BY-SA on every run by default, and
    LICENSE_POLICY.md admits CC-BY only."""
    text = NOTE.read_text(encoding="utf-8")

    assert "4,873" in text and "214" in text
    assert "417,804" in text
    # The pair it costs, named rather than buried in an aggregate.
    assert "0:2" in text and "1:0" in text
    assert "하이드로퀴논 → 티로시나제가 23위에서 33위로" in text
    # And the double-counting that had to be fixed to get an honest number.
    assert "151,552" in text and "95.4%" in text
    assert "859,200" in text, "why the InChIKey comparison matched nothing"
    assert "share-alike" in text and "ODbL" in text
    assert "33개" in text, "the size of what is not in the default"
    # Not declined - built, and one config line away, because the licence call
    # belongs to whoever ships the result.
    assert "activity_retrieval_runtime_gtopdb_202608" in text
    assert "두 티어를 다 만들어 뒀다" in text
    # And the drop that had to be fixed to get there.
    assert "567,348" in text and "876" in text


def test_the_note_records_that_the_flag_was_on_while_nothing_read_it() -> None:
    """The second time this exact failure appeared in one session. A config key
    no code reads is indistinguishable from no promotion at all."""
    text = NOTE.read_text(encoding="utf-8")

    assert "최근접 유사도로 순위를 매기고 있었고" in text
    assert "test_stage3_recipe_wiring.py" in text
    assert "--recipe-index-dir" in text


def test_the_note_records_the_query_fingerprint_mismatch() -> None:
    """The largest defect found here: every real run's similarities were capped
    around 0.19 because the query carried explicit hydrogens and no reference
    did. Evaluation could not see it - the harness scores from SMILES."""
    text = NOTE.read_text(encoding="utf-8")

    assert "0.186" in text and "1.000" in text
    assert "removeHs=False" in text
    assert "0.1905" in text, "the ceiling observed in real run outputs"
    assert "에탄올은 ChEMBL에 있다" in text
    assert "test_daina_query_fingerprint.py" in text


def test_the_index_a_run_actually_reads_carries_no_share_alike_licence() -> None:
    """Not "a directory named permissive says permissive" - the configured one.

    Both tiers are built and they differ by one config line. A test that reads
    `data/runtime_evidence_permissive_202608/manifest.json` confirms only that
    the permissive directory is permissive; flipping
    `docking.daina_recipe_index_dir` to the gtopdb index would put ODbL /
    CC BY-SA on every run with the whole suite green.
    """
    import json

    import yaml

    config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text(encoding="utf-8"))
    docking = config["docking"]
    if not docking.get("daina_recipe_scoring"):
        pytest.skip("recipe scoring is off")

    manifest_path = ROOT / docking["daina_recipe_index_dir"] / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("the configured index is not built here")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    licensing = manifest.get("evidence_licensing") or {}
    licences = licensing.get("licences") or []
    assert licences, f"the index must record what it was built from: {manifest_path}"
    for licence in licences:
        assert "SA" not in licence and "ODbL" not in licence, (
            f"the default run path would inherit a share-alike obligation: {licence}"
        )
    # And the sources actually in the edges, not only the declared tier.
    assert "GtoPdb" not in (manifest.get("source_counts") or {}), manifest_path


def test_the_negative_evidence_concern_was_measured_not_dismissed() -> None:
    """max_union_any pays full similarity for an analogue measured as inactive.

    Real - 9.9% of index pairs are negative-only - and the obvious fix measures
    worse: excluding them loses six Top10 pairs and gains none (p=0.031). The
    note has to carry both halves, because "we looked and the fix lost" is the
    only thing that stops it being re-proposed.
    """
    text = NOTE.read_text(encoding="utf-8")

    assert "max_union_nonneg" in text
    assert "9.9%" in text and "18.1%" in text, "gray and negative-only are different"
    assert "6:0" in text and "0.031" in text, "the measured cost of the fix"
    # And the direction that flips, which is the actually interesting finding.
    assert "33위에서 13위로" in text
    assert "용도에 따라 방향이 갈린다" in text


def test_the_negative_aware_candidates_exist_and_only_one_is_promoted() -> None:
    """All three are in the code so the measurement is reproducible; the one a
    run scores with was chosen on the skin panel, and only after the dev gate
    qualified it."""
    import json

    from activity_retrieval_scoring import RECIPES

    using = [r.recipe_id for r in RECIPES if r.max_union_nonneg > 0.0]
    assert using == [
        "union_nonneg_max",
        "union_nonneg_consensus",
        "union_nonneg_contrast",
        "union_discount_light",
        "union_discount_even",
        "union_discount_heavy",
    ]

    # Weighting both is a discount, not double counting: max_union_nonneg
    # maximises over a subset of max_union_any's edges, so a negative-only edge
    # earns the max_union_any weight alone while a non-negative one earns both.
    # An earlier version of this test forbade the combination outright, which
    # would have ruled out the only recipes that keep EGCG -> MMP2.
    for recipe in RECIPES:
        combined = recipe.max_union_any + recipe.max_union_nonneg
        assert combined <= 1.0 + 1e-9, f"{recipe.recipe_id} over-weights similarity"

    if RECIPE.exists():
        promoted = json.loads(RECIPE.read_text(encoding="utf-8"))["selected_recipe"]
        assert promoted["recipe_id"] == "union_discount_light"
        assert promoted.get("max_union_nonneg", 0.0) > 0.0


def test_the_note_records_the_whitening_promotion_and_what_it_costs() -> None:
    """This is the first promotion here that loses a pair outright. If the note
    only carried the tyrosinase gain it would be advocacy, not a record."""
    text = NOTE.read_text(encoding="utf-8")

    assert "union_nonneg_consensus" in text
    # Qualified by the gate, chosen on the panel - both halves.
    assert "0.4376" in text and "0.4607" in text
    assert "operator_choice_among_dev_qualified" in text
    assert "게이트 우회가 아니라" in text
    # What it buys.
    assert "하이드로퀴논 | 37위 | 33위 | **26위**" in text
    assert "2개에서 3개" in text
    # And what it costs, named.
    assert "EGCG → MMP2 (P08253)" in text
    assert "24위 → 56위" in text
    assert "처음으로 순수한 손실이 있는 경우" in text
    assert "EGCG의 티로시나제를 올리고 EGCG의 MMP2를 내린다" in text


def test_a_gate_failing_recipe_cannot_be_promoted_by_preference() -> None:
    """The escape hatch this opened, closed. union_nonneg_max scored best on the
    whitening pairs of all three and failed the dev gate; preference must not
    reach it."""
    source = (ROOT / "eval" / "activity_retrieval_model.py").read_text(encoding="utf-8")

    assert "def _choose_recipe" in source
    block = source.split("def _choose_recipe", 1)[1].split("\ndef ", 1)[0]
    assert "eligible = {recipe.recipe_id: recipe for _, recipe in passing}" in block
    assert "not a ranking to override" in block


def test_the_note_records_that_the_discount_beat_the_exclusion() -> None:
    """Section 8 promoted an exclusion that cost EGCG -> MMP2; section 9 replaced
    it with a discount that costs nothing. The note has to carry the diagnosis,
    or the exclusion looks like the only way to protect tyrosinase."""
    text = NOTE.read_text(encoding="utf-8")

    assert "union_discount_light" in text
    # Why MMP2 collapsed: the analogues are similar and measured under the cut.
    assert "0.727" in text and "4.06" in text
    assert "0.319" in text
    assert "문턱이지 생물학이 아니다" in text
    # And that EGCG's tyrosinase gain in section 8 was not its own edge.
    assert "다른 표적들이 내려간" in text
    # The assertion that blocked the answer, named as mine.
    assert "내가 만든 장애물" in text
    assert "double-counts the same similarity" in text
    assert "틀린 단언이었다" in text
