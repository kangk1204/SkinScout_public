"""F20 regression: degraded safety evidence keeps its reasons end-to-end.

A missing model, an applicability-limited negative, and a real risk positive
used to collapse into a single PASS/FLAG_HIGH string. The tests here pin the
separate availability / applicability / decision states: required-model-missing,
partial-positive, and fully-negative fixtures produce distinct reasons and
eligibility, and a degraded result can never be recorded as a safety PASS.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analog_funnel"))

import safety_states  # noqa: E402


def test_fully_negative_is_claimable_and_docking_eligible() -> None:
    state = safety_states.safety_state("PASS")

    assert state["decision"] == "PASS"
    assert state["availability"] == safety_states.AVAILABILITY_AVAILABLE
    assert state["applicability"] == safety_states.APPLICABILITY_WITHIN_DOMAIN
    assert state["claimable"] is True
    assert state["review_required"] is False
    assert state["docking_eligible"] is True
    assert state["reason"] == "decision:PASS"


def test_partial_positive_is_review_required_but_docking_may_proceed() -> None:
    state = safety_states.safety_state("FLAG_HIGH")

    assert state["decision"] == "FLAG_HIGH"
    assert state["availability"] == safety_states.AVAILABILITY_AVAILABLE
    assert state["claimable"] is False
    assert state["review_required"] is True
    assert state["docking_eligible"] is True
    assert state["reason"] == "decision:FLAG_HIGH"


def test_required_model_missing_is_degraded_and_never_pass() -> None:
    state = safety_states.safety_state("PASS", missing_models=["husspred"])

    assert state["decision"] == "FLAG_HIGH"
    assert state["raw_decision"] == "PASS"
    assert state["availability"] == safety_states.AVAILABILITY_DEGRADED
    assert state["degraded"] is True
    assert state["claimable"] is False
    assert state["docking_eligible"] is True
    assert "missing_models:husspred" in state["reason"]
    assert "degraded_not_claimable:degraded_missing_models" in state["reason"]


def test_three_fixtures_produce_distinct_eligibility() -> None:
    negative = safety_states.safety_state("PASS")
    partial = safety_states.safety_state("FLAG_HIGH")
    missing = safety_states.safety_state("PASS", missing_models=["pred_skin"])

    assert negative["claimable"] and not partial["claimable"] and not missing["claimable"]
    assert negative["reason"] != partial["reason"] != missing["reason"]
    assert missing["availability"] != partial["availability"]
    assert missing["decision"] == partial["decision"] == "FLAG_HIGH"


def test_applicability_limited_stays_non_claimable_and_visible() -> None:
    state = safety_states.safety_state(
        "PASS", applicability_limited_models=["husspred"]
    )

    assert state["applicability"] == safety_states.APPLICABILITY_LIMITED
    assert state["claimable"] is False
    assert state["docking_eligible"] is True
    assert "applicability_limited:husspred" in state["reason"]


def test_csv_string_flags_and_model_lists_are_parsed() -> None:
    undegraded = safety_states.safety_state(
        "PASS",
        missing_models="",
        applicability_limited_models="",
        degraded="false",
    )
    assert undegraded["availability"] == safety_states.AVAILABILITY_AVAILABLE
    assert undegraded["degraded"] is False
    assert undegraded["claimable"] is True

    listed = safety_states.safety_state(
        "FLAG_HIGH", missing_models="husspred, pred_skin", degraded="false"
    )
    assert listed["missing_models"] == ["husspred", "pred_skin"]
    assert listed["degraded"] is True


def test_halt_and_unknown_are_not_docking_eligible() -> None:
    for decision in ("HALT", "", "UNKNOWN"):
        state = safety_states.safety_state(decision)
        assert state["docking_eligible"] is False
        assert state["claimable"] is False
        assert state["reason"]


def test_missing_models_are_read_from_nested_skin_sens() -> None:
    payload = {
        "missing_models": ["top_level_must_be_ignored"],
        "skin_sens": {
            "decision": "FLAG_HIGH",
            "missing_models": ["husspred", "pred_skin"],
            "applicability_limited_models": ["stoptox"],
        },
    }

    assert safety_states.missing_models_from_report(payload) == [
        "husspred",
        "pred_skin",
    ]
    state = safety_states.safety_state(
        safety_states.nested_skin_sens(payload).get("decision"),
        missing_models=safety_states.missing_models_from_report(payload),
        applicability_limited_models=safety_states.nested_skin_sens(payload).get(
            "applicability_limited_models"
        ),
    )
    assert "missing_models:husspred" in state["reason"]
    assert "applicability_limited:stoptox" in state["reason"]
    assert "top_level_must_be_ignored" not in state["reason"]


def test_validate_generated_reads_nested_missing_models(tmp_path: Path) -> None:
    module = importlib.import_module("validate_generated")
    run_dir = tmp_path / "results" / "runs" / "R1"
    admet_dir = run_dir / "02_admet"
    admet_dir.mkdir(parents=True)
    (admet_dir / "skin_sens_decision.txt").write_text("PASS\n", encoding="utf-8")
    (admet_dir / "admet_report.json").write_text(
        json.dumps(
            {
                "missing_models": ["top_level_must_be_ignored"],
                "skin_sens": {
                    "decision": "PASS",
                    "degraded": True,
                    "missing_models": ["husspred"],
                    "applicability_limited_models": [],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    skin, state = module.safety_state_for_run(run_dir)

    assert skin == "PASS"
    assert state["decision"] == "FLAG_HIGH"
    assert state["availability"] == safety_states.AVAILABILITY_DEGRADED
    assert state["missing_models"] == ["husspred"]
    assert state["claimable"] is False


def test_run_safety_degraded_reads_nested_report(tmp_path: Path) -> None:
    module = importlib.import_module("run_safety_degraded")
    report = tmp_path / "admet_report.json"
    report.write_text(
        json.dumps(
            {
                "missing_models": ["top_level_must_be_ignored"],
                "skin_sens": {
                    "decision": "PASS",
                    "degraded": True,
                    "missing_models": ["pred_skin"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    state = module.report_state("PASS", report)

    assert state["decision"] == "FLAG_HIGH"
    assert state["missing_models"] == ["pred_skin"]
    assert state["claimable"] is False
    assert "missing_models:pred_skin" in state["reason"]


def test_funnel_safe_decisions_still_match_the_legacy_gate() -> None:
    for name in ("pair_docking", "validate_generated"):
        module = importlib.import_module(name)
        assert module.SAFE_DECISIONS == {"PASS", "FLAG_HIGH"}
