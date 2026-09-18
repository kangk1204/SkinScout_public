"""Unit tests for skin-sens consensus rule (INSTRUCTIONS.md §5)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage2_consensus import (  # noqa: E402
    skin_sens_decision,
    unavailable_skin_sens_models,
)


def test_outside_negative_is_available_evidence_not_a_missing_model() -> None:
    """F30: applicability limits availability must not be conflated."""
    payloads = {
        "husspred": {
            "status": "ok",
            "skin_sens_call": "negative",
            "applicability_domain": {"status": "outside"},
        },
        "stoptox": {"status": "ok", "skin_sens_call": "negative"},
        "pred_skin": {"status": "ok", "consensus_call": "negative"},
    }
    assert unavailable_skin_sens_models(payloads) == []


def test_missing_and_error_statuses_are_unavailable() -> None:
    payloads = {
        "husspred": {"status": "unavailable"},
        "stoptox": {"status": "error", "skin_sens_call": "negative"},
        "pred_skin": {"status": "ok", "consensus_call": "negative"},
    }
    assert unavailable_skin_sens_models(payloads) == ["husspred", "stoptox"]


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
