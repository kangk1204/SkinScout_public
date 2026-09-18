"""Unit tests for skin-sens consensus rule (INSTRUCTIONS.md §5)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage2_consensus import skin_sens_decision  # noqa: E402


def test_all_negative_passes() -> None:
    assert skin_sens_decision(["negative", "negative", "negative"]) == "PASS"


def test_two_of_three_halts() -> None:
    assert skin_sens_decision(["positive", "positive", "negative"]) == "HALT"
    assert skin_sens_decision(["positive", "negative", "positive"]) == "HALT"
    assert skin_sens_decision(["positive", "positive", "positive"]) == "HALT"


def test_single_positive_flags() -> None:
    assert skin_sens_decision(["positive", "negative", "negative"]) == "FLAG_HIGH"


def test_all_unavailable_flags_for_review() -> None:
    assert skin_sens_decision([None, None, None]) == "FLAG_HIGH"


def test_two_negatives_one_missing_passes() -> None:
    assert skin_sens_decision(["negative", "negative", None]) == "PASS"


def test_one_positive_one_unknown_one_negative_flags() -> None:
    assert skin_sens_decision(["positive", None, "negative"]) == "FLAG_HIGH"


def test_custom_halt_threshold() -> None:
    # halt_min_votes=1 makes a single positive HALT instead of FLAG_HIGH
    assert (
        skin_sens_decision(["positive", "negative", "negative"], halt_min_votes=1)
        == "HALT"
    )


def test_halt_threshold_cannot_exceed_configured_model_count() -> None:
    with pytest.raises(ValueError, match="between 1 and 3"):
        skin_sens_decision(["positive", "positive", "positive"], halt_min_votes=4)
