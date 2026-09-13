"""The 15-compound panel is deciding something it cannot resolve.

`union_p6_consensus` beats the frozen production baseline by +12.7 points of
Top30 on the 256-query dev panel. It is not promoted because the known-skin panel
regressed by 0.125 of Top30 - which, on 32 compound-target pairs, is a difference
of four pairs. Exact McNemar cannot put four pairs below p=0.05 under any
arrangement of the discordance, so the blocking comparison is not a measurement.

These tests pin the arithmetic and the panel size, so that if the panel grows the
claim gets rechecked rather than quietly inherited.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

PANEL = ROOT / "data" / "validation" / "skin_known_target_panel.csv"
NOTE = ROOT / "docs" / "PANEL_ARBITRATION_20260831.md"
VALIDATION = ROOT / "docs" / "SKIN_KNOWN_TARGET_VALIDATION.md"
DEV_METRICS = (
    ROOT / "results" / "eval" / "activity_retrieval_202608" / "dev_selection"
    / "dev_recipe_metrics.csv"
)


@pytest.fixture(scope="module")
def module():
    import analyse_panel_arbitration_power

    return analyse_panel_arbitration_power


def test_the_panel_now_holds_forty_six_pairs() -> None:
    """v3: 22 compounds. v2's 32 pairs could not resolve the 0.125 effect."""
    panel = pd.read_csv(PANEL)
    pairs = sum(
        len(str(value).split(";")) for value in panel["known_targets"] if pd.notna(value)
    )
    assert len(panel) == 22
    assert pairs == 46


def test_four_pairs_could_not_reach_significance(module) -> None:
    """Why v2 was replaced: on 32 pairs the effect was four pairs, and no
    arrangement of four discordant pairs clears alpha."""
    assert module.best_case_p(4) == pytest.approx(0.125)
    assert module.best_case_p(4) > 0.05

    difference = round((module.BASELINE_TOP30 - module.SELECTED_TOP30) * 32)
    assert difference == 4


def test_forty_six_pairs_can(module) -> None:
    """v3 crosses the threshold: six pairs, and six can reach p=0.031."""
    difference = round((module.BASELINE_TOP30 - module.SELECTED_TOP30) * 46)
    assert difference == 6
    assert module.best_case_p(6) <= 0.05

    # 44 is where it flips; 42 still cannot.
    assert module.best_case_p(round(0.125 * 42)) > 0.05
    assert module.best_case_p(round(0.125 * 44)) <= 0.05


def test_the_dev_panel_disagrees_by_the_same_magnitude(module) -> None:
    """+0.127 on 256 queries against -0.125 on 32 pairs, and the small one wins."""
    dev_gain = module.DEV_SELECTED_TOP30 - module.DEV_BASELINE_TOP30
    skin_loss = module.BASELINE_TOP30 - module.SELECTED_TOP30

    assert dev_gain == pytest.approx(0.127, abs=0.001)
    assert skin_loss == pytest.approx(0.125, abs=0.001)


@pytest.mark.skipif(not DEV_METRICS.exists(), reason="dev selection results not present")
def test_the_frozen_baseline_really_is_the_worst_recipe_on_dev() -> None:
    """If the frozen recipe were merely conservative this would not matter much.

    It is last of six on every metric the dev panel reports.
    """
    metrics = pd.read_csv(DEV_METRICS).set_index("recipe_id")
    frozen = metrics.loc["chembl_p5_max"]
    for column in ("ranking_top10", "ranking_top30", "ranking_mrr"):
        assert frozen[column] == metrics[column].min(), column


def test_the_note_states_the_remedy_and_its_size() -> None:
    text = NOTE.read_text(encoding="utf-8")
    assert "화합물 7개" in text
    assert "p = 0.125" in text
    assert "+12.7%p" in text
    # And it must not oversell the outcome either way.
    assert "선정 레시피가 이긴 것은 아니다" in text


def test_the_note_reports_what_the_v3_measurement_actually_showed() -> None:
    """The blocking regression did not reproduce, and the new difference is null.

    Both halves have to stay on the page. Reporting only the first would read as
    "the candidate is better"; reporting only the second would hide that the
    stated reason for blocking it is gone.
    """
    text = NOTE.read_text(encoding="utf-8")

    # The regression that blocked promotion is gone.
    assert "0.674 → 0.696" in text
    assert "재현되지 않는다" in text
    # And the new difference does not resolve either.
    assert "p = 1.000" in text
    assert "구별되지 않는다" in text
    # Including the part where this document's own prediction was half wrong.
    assert "내 예측도 틀렸다" in text
    assert "273쌍" in text
    # Promotion stays a human decision.
    assert "사람이 내려야 한다" in text


def test_the_validation_doc_still_records_the_block() -> None:
    """If the block is ever lifted, this test should fail and be rewritten."""
    text = VALIDATION.read_text(encoding="utf-8")
    assert "chembl_p5_max" in text
    assert "0.125" in text


def test_the_note_records_the_stereoisomer_mistake() -> None:
    """The cis/trans error has to stay on the page, not be quietly fixed.

    The first v3 commit had tranexamic acid as the cis isomer - a valid molecule
    RDKit parses without complaint, and not the compound the case_id names. That
    is exactly what made v1 wrong. It was caught before anyone quoted a number
    from it, by checking each added structure against the ChEMBL mirror rather
    than against the panel file it guards.
    """
    text = NOTE.read_text(encoding="utf-8")

    assert "cis 이성질체" in text
    assert "CHEMBL877" in text
    assert "v1 패널이 틀렸던 것과 정확히 같은 유형이다" in text
    # And the retraction of the claim that came with it.
    assert '"ChEMBL에 없는 어려운 케이스"라고\n적었는데 **틀렸다.**' in text
    # The superseded first-v3 digest is named so its artifacts stay identifiable.
    assert "62bedced" in text


def test_the_note_says_the_correction_changed_no_number() -> None:
    """Honest either way: the fix mattered, and it moved nothing.

    Retrieval fingerprints are built with includeChirality=False, so cis and
    trans tranexamic acid are the same bit vector and the case ranked 25/46 both
    times. Reporting that without also saying why the fix was still required
    would read as "the check was unnecessary".
    """
    text = NOTE.read_text(encoding="utf-8")

    assert "includeChirality=False" in text
    assert "한 자리도 달라지지 않았다" in text
    # And the reason it still had to be fixed.
    assert "고칠 필요가 없었던 건 아니다" in text


def test_the_note_carries_the_follow_up_correction() -> None:
    """Measuring through the run path showed the v2 block had a real signal.

    This document concluded the block was an artefact of wrong structures and a
    small sample. Both were true and neither was the whole story: the recipe
    discards evidence below pActivity 6, which is where tyrosinase lives, and
    extending the panel diluted the TYR share that had been detecting it.
    """
    text = NOTE.read_text(encoding="utf-8")

    assert "후속 정정" in text
    assert "절반만 맞았다" in text
    assert "희석" in text
    assert "RECIPE_RUNPATH_MEASURED_20260831.md" in text
