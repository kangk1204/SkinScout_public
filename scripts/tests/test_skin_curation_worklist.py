"""Skin expression is not druggability, and the worklist has to say so.

Priority 2 of the expansion note started as "curate the 91 skin-expressed targets
a run cannot reach". Pulling the list showed what those 91 actually are: 30 are
keratins, ribosomal proteins, mitochondrially encoded complex I subunits,
collagens and histones - abundant in skin, and not things a cosmetic ingredient
is screened against. 46 more are outside any druggable protein class. Fifteen are
real, and two of those are TYRP1 and DCT, which means the melanin pathway is
currently indexed for TYR alone.

These tests keep the second axis in place. Without it the worklist silently
becomes a list of structural proteins again.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

WORKLIST = ROOT / "data" / "curation" / "skin_target_worklist.csv"
SCRIPT = ROOT / "scripts" / "build_skin_curation_worklist.py"
NOTE = ROOT / "docs" / "DATA_EXPANSION_20260831.md"

pytestmark = pytest.mark.skipif(
    not WORKLIST.exists(), reason="curation worklist is not generated here"
)


@pytest.fixture(scope="module")
def worklist() -> pd.DataFrame:
    return pd.read_csv(WORKLIST)


def test_the_high_band_gap_is_91_targets(worklist: pd.DataFrame) -> None:
    """Measured against the run path, not the benchmark index."""
    assert len(worklist) == 91


def test_only_fifteen_of_them_are_worth_a_curators_time(worklist: pd.DataFrame) -> None:
    """The correction that changed the size of the proposal by 6x."""
    assert int(worklist["worth_curating"].sum()) == 15

    counts = worklist["work_needed"].value_counts().to_dict()
    assert counts["structural protein - skip"] == 30
    assert counts["not in a druggable class - low priority"] == 46
    assert counts["literature or in-house data"] == 15


def test_the_melanin_pathway_gap_is_named(worklist: pd.DataFrame) -> None:
    """TYR is indexed; TYRP1 and DCT are not. For a whitening ingredient that is
    the difference between one hit and the pathway."""
    worth = worklist[worklist["worth_curating"]]
    genes = set(worth["gene"])
    assert "TYRP1" in genes
    assert "DCT" in genes


def test_structural_families_are_excluded_not_merely_ranked_low(
    worklist: pd.DataFrame,
) -> None:
    structural = worklist[worklist["structural_family"]]
    assert len(structural) == 30
    assert not structural["worth_curating"].any()
    # The families that dominate the skin score by abundance.
    genes = " ".join(structural["gene"].astype(str))
    for family in ("KRT", "MT-ND", "RPL"):
        assert family in genes, family


def test_a_target_reachable_from_another_source_is_distinguished(
    worklist: pd.DataFrame,
) -> None:
    """Two different jobs: merge a source we already downloaded, or go find data
    that does not exist in any public source."""
    assert int(worklist["evidence_exists_elsewhere"].sum()) == 9


def test_the_note_reports_fifteen_not_eighty_nine() -> None:
    text = NOTE.read_text(encoding="utf-8")
    assert "**15**" in text
    assert "TYRP1" in text and "DCT" in text
    # The retractions stay visible; the same class of mistake five times.
    assert "## 틀렸던 것" in text


def test_the_builder_states_why_the_second_axis_exists() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "DRUGGABLE_CLASSES" in source
    assert "STRUCTURAL_PREFIXES" in source
    assert "abundance in skin" in source


def test_the_guide_explains_why_a_whitening_ingredient_only_hits_tyr() -> None:
    """A reader will notice TYR alone and read it as a negative result.

    It is not. TYRP1 and DCT have no public activity data at all, so they cannot
    be candidates whatever the ingredient does. That belongs next to the coverage
    number, where a reader meets the question.
    """
    guide = (ROOT / "docs" / "RESEARCHER_GUIDE.md").read_text(encoding="utf-8")
    section = guide.split("## 5. 커버리지", 1)[1].split("\n## ", 1)[0]

    assert "TYRP1" in section and "DCT" in section
    assert "60.4%" in section, "the skin-band number, not the proteome-wide one"
    assert "안 붙어서가 아니다" in section
    assert "skin_target_worklist.csv" in section
