from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import os

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import target_intent as ti

WORKBOOK = Path(os.environ.get("SKINSCOUT_BIOMARKER_XLSX", "").strip())
GENE_MAP = Path(os.environ.get("SKINSCOUT_GENE_MAP_PATH", "").strip())


@pytest.fixture(scope="module")
def intents() -> list[dict]:
    if not (WORKBOOK.is_file() and GENE_MAP.is_file()):
        pytest.skip(
            "private workbook not provided "
            "(SKINSCOUT_BIOMARKER_XLSX / SKINSCOUT_GENE_MAP_PATH)"
        )
    return ti.build_target_intents(WORKBOOK, GENE_MAP)


def _intent(intents: list[dict], intent_id: str) -> dict:
    return next(row for row in intents if row["intent_id"] == intent_id)


def _priority_summary(**overrides: object) -> dict:
    result = {
        "structure_valid": True,
        "route_supported": True,
        "analysis_applicability": "in_scope",
        "analysis_scope_supported": True,
        "direction_relation": "supports",
        "material_policy": "eligible",
        "required_information_complete": True,
        "minimum_evidence_met": True,
        "functional_evidence": True,
    }
    result.update(overrides)
    return result


def _pass_safety(**overrides: object) -> dict:
    result = {
        "run_valid": True,
        "safety_consensus_valid": True,
        "full_analysis_complete": True,
        "decision": "PASS",
        "applicability_domain": "inside",
    }
    result.update(overrides)
    return result


def test_real_workbook_expands_31_source_rows_to_33_intents(
    intents: list[dict],
) -> None:
    assert len(intents) == 33
    assert len({row["source_row_id"] for row in intents}) == 31
    assert [row["source_row_number"] for row in intents][:2] == [4, 5]
    assert intents[-1]["source_row_number"] == 34
    assert all(set(row) == set(ti.INTENT_KEYS) for row in intents)

    pdgf = [row for row in intents if row["biomarker"] == "PDGF"]
    assert [(row["gene_symbol"], row["uniprot_id"]) for row in pdgf] == [
        ("PDGFA", "P04085"),
        ("PDGFB", "P01127"),
    ]
    scf_kit = [row for row in intents if row["biomarker"] == "SCF / c-KIT"]
    assert [(row["gene_symbol"], row["uniprot_id"]) for row in scf_kit] == [
        ("KITLG", "P21583"),
        ("KIT", "P10721"),
    ]


def test_source_lineage_contains_sealed_inputs(intents: list[dict]) -> None:
    source = intents[0]["source_evidence"]
    assert source["row_number"] == 4
    assert source["sheet"] == "Sheet1"
    assert len(source["workbook_sha256"]) == 64
    assert len(source["gene_map_sha256"]) == 64
    assert len(source["registry_sha256"]) == 64


def test_endpoints_never_receive_fake_protein_or_docking_ids(
    intents: list[dict],
) -> None:
    endpoints = [row for row in intents if row["route"] == "endpoint"]
    assert {row["biomarker"] for row in endpoints} == {
        "Hyaluronan (HA)",
        "8-OHdG",
        "Ceramide",
        "TEWL",
    }
    assert all(
        row["uniprot_id"] is None and row["target_taxid"] is None for row in endpoints
    )
    assert all(row["docking_eligible"] is False for row in endpoints)


def test_ll37_keeps_mature_peptide_distinct_from_precursor(intents: list[dict]) -> None:
    ll37 = _intent(intents, "TI-029-LL37")
    assert ll37["gene_symbol"] == "CAMP"
    assert ll37["uniprot_id"] == "P49913"
    assert ll37["entity_type"] == "mature_peptide"
    assert "134-170" in ll37["protein_form"]
    assert "precursor" in ll37["component_of"]
    assert ll37["desired_effect"] == "decrease_endpoint"
    assert ll37["docking_eligible"] is False


def test_expression_and_function_intents_are_not_interchangeable(
    intents: list[dict],
) -> None:
    mitf = _intent(intents, "TI-021-MITF")
    ar = _intent(intents, "TI-007-AR")
    assert mitf["desired_effect"] == "decrease_expression"
    assert ar["desired_effect"] == "inhibit_function"
    assert (
        ti.classify_evidence(
            mitf,
            {
                "evidence_axis": "protein_function",
                "observed_effect": "inhibit_function",
            },
        )["direction_relation"]
        == "unknown"
    )
    assert (
        ti.classify_evidence(
            mitf,
            {"evidence_axis": "expression", "observed_effect": "decrease_expression"},
        )["direction_relation"]
        == "unknown"
    )
    supported = ti.classify_evidence(
        mitf,
        {
            "evidence_axis": "expression",
            "observed_effect": "decrease_expression",
            "target_uniprot_id": "O75030",
            "target_taxid": 9606,
        },
    )
    assert supported["direction_relation"] == "supports"


def test_schema_validates_every_built_intent(intents: list[dict]) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(
        (ROOT / "schemas" / "target_intent_v1.json").read_text(encoding="utf-8")
    )
    validator = jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.FormatChecker()
    )
    for intent in intents:
        validator.validate(intent)


def test_validation_rejects_endpoint_with_uniprot(intents: list[dict]) -> None:
    invalid = copy.deepcopy(_intent(intents, "TI-027-TEWL"))
    invalid["uniprot_id"] = "P99999"
    with pytest.raises(ValueError, match="endpoint intents"):
        ti.validate_intent(invalid)


def test_ar_dht_functional_agonism_is_opposed(intents: list[dict]) -> None:
    ar = _intent(intents, "TI-007-AR")
    result = ti.classify_evidence(
        ar,
        {
            "evidence_axis": "protein_function",
            "directness": "direct_supported",
            "action_type": "AGONIST",
            "target_uniprot_id": "P10275",
            "target_taxid": 9606,
            "model_level": "engineered_cell",
            "assay_material_taxid": 10090,
        },
    )
    assert result["direction_relation"] == "opposes"
    assert result["directness"] == "direct_supported"
    assert "engineered_cell" in result["reason"]


@pytest.mark.parametrize("axis", ["binding", "cooccurrence"])
def test_binding_or_cooccurrence_never_establishes_function_direction(
    intents: list[dict], axis: str
) -> None:
    result = ti.classify_evidence(
        _intent(intents, "TI-007-AR"),
        {
            "evidence_axis": axis,
            "directness": "direct_supported",
            "action_type": "ANTAGONIST",
        },
    )
    assert result["direction_relation"] == "unknown"
    if axis == "cooccurrence":
        assert result["directness"] == "unknown"


@pytest.mark.parametrize(
    "action", ["partial agonist", "inverse agonist", "positive allosteric modulator"]
)
def test_context_dependent_action_types_remain_unknown(
    intents: list[dict], action: str
) -> None:
    result = ti.classify_evidence(
        _intent(intents, "TI-007-AR"),
        {
            "evidence_axis": "protein_function",
            "action_type": action,
            "target_uniprot_id": "P10275",
            "target_taxid": 9606,
        },
    )
    assert result["direction_relation"] == "unknown"
    assert "context" in result["reason"]


def test_wrong_target_identity_cannot_support_direction(intents: list[dict]) -> None:
    result = ti.classify_evidence(
        _intent(intents, "TI-020-TYR"),
        {
            "evidence_axis": "protein_function",
            "observed_effect": "inhibit_function",
            "target_uniprot_id": "P10275",
        },
    )
    assert result["direction_relation"] == "unknown"
    assert "mismatched" in result["reason"]


@pytest.mark.parametrize(
    "evidence",
    [
        {"evidence_axis": "protein_function", "action_type": "ANTAGONIST"},
        {
            "evidence_axis": "protein_function",
            "action_type": "ANTAGONIST",
            "target_uniprot_id": "P10275",
        },
        {
            "evidence_axis": "protein_function",
            "action_type": "ANTAGONIST",
            "target_taxid": 9606,
        },
    ],
)
def test_protein_direction_requires_exact_uniprot_and_taxid(
    intents: list[dict], evidence: dict
) -> None:
    result = ti.classify_evidence(_intent(intents, "TI-007-AR"), evidence)
    assert result["direction_relation"] == "unknown"
    assert "missing or mismatched" in result["reason"]


def test_endpoint_direction_requires_exact_intent_and_no_fake_protein_id(
    intents: list[dict],
) -> None:
    tewl = _intent(intents, "TI-027-TEWL")
    base = {
        "evidence_axis": "phenotype",
        "observed_effect": "decrease_endpoint",
        "model_level": "human_study",
    }
    missing = ti.classify_evidence(tewl, base)
    wrong = ti.classify_evidence(tewl, {**base, "target_intent_id": "TI-025-CERAMIDE"})
    fake = ti.classify_evidence(
        tewl,
        {**base, "target_intent_id": tewl["intent_id"], "target_uniprot_id": "P99999"},
    )
    exact = ti.classify_evidence(tewl, {**base, "target_intent_id": tewl["intent_id"]})
    assert missing["direction_relation"] == "unknown"
    assert wrong["direction_relation"] == "unknown"
    assert fake["direction_relation"] == "unknown"
    assert exact["direction_relation"] == "supports"


def test_decision_precedence_is_fail_closed(intents: list[dict]) -> None:
    ar = _intent(intents, "TI-007-AR")
    unsupported = ti.decide_candidate(
        ar,
        _priority_summary(structure_valid=False, direction_relation="opposes"),
        _pass_safety(decision="HALT"),
    )
    assert unsupported["decision"] == "unsupported"

    halted = ti.decide_candidate(ar, _priority_summary(), _pass_safety(decision="HALT"))
    assert halted["decision"] == "exclude"

    partial_halt = ti.decide_candidate(
        ar,
        _priority_summary(),
        _pass_safety(
            decision="HALT",
            run_valid=False,
            safety_consensus_valid=True,
            full_analysis_complete=False,
        ),
    )
    assert partial_halt["decision"] == "exclude"

    unverified_halt = ti.decide_candidate(
        ar,
        _priority_summary(),
        _pass_safety(
            decision="HALT",
            run_valid=False,
            safety_consensus_valid=False,
            full_analysis_complete=False,
        ),
    )
    assert unverified_halt["decision"] == "needs_evidence"

    opposed = ti.decide_candidate(
        ar, _priority_summary(direction_relation="opposes"), _pass_safety()
    )
    assert opposed["decision"] == "exclude"


@pytest.mark.parametrize(
    ("summary_change", "safety_change", "reason_fragment"),
    [
        ({"direction_relation": "unknown"}, {}, "direction"),
        ({"direction_relation": "conflict"}, {}, "conflicting"),
        ({"functional_evidence": False}, {}, "functional"),
        ({"required_information_complete": False}, {}, "incomplete"),
        ({}, {"decision": "FLAG_HIGH"}, "FLAG_HIGH"),
        ({}, {"applicability_domain": "review"}, "applicability"),
        ({}, {"applicability_domain": "outside"}, "applicability"),
        ({}, {"run_valid": False}, "valid current"),
    ],
)
def test_uncertainty_requires_evidence(
    intents: list[dict],
    summary_change: dict,
    safety_change: dict,
    reason_fragment: str,
) -> None:
    result = ti.decide_candidate(
        _intent(intents, "TI-007-AR"),
        _priority_summary(**summary_change),
        _pass_safety(**safety_change),
    )
    assert result["decision"] == "needs_evidence"
    assert any(reason_fragment in reason for reason in result["reasons"])


def test_priority_requires_all_policy_conditions_and_does_not_claim_safety(
    intents: list[dict],
) -> None:
    result = ti.decide_candidate(
        _intent(intents, "TI-020-TYR"),
        _priority_summary(),
        _pass_safety(),
    )
    assert result["decision"] == "experiment_priority"
    assert any(
        "does not establish biological safety" in reason for reason in result["reasons"]
    )


def test_missing_summary_and_safety_fields_fail_closed(intents: list[dict]) -> None:
    result = ti.decide_candidate(_intent(intents, "TI-007-AR"), {}, {})
    assert result["decision"] == "unsupported"
    assert "exact compound structure is not validated" in result["reasons"]
    assert set(ti.DECISION_SUMMARY_KEYS) == set(_priority_summary())
    assert set(ti.DECISION_SAFETY_KEYS) == set(_pass_safety())


def _synthetic_priority_intent() -> dict:
    return {
        "schema_version": ti.SCHEMA_VERSION,
        "intent_id": "TI-000-SYN",
        "source_row_id": "1",
        "source_sheet": "test",
        "source_row_number": 1,
        "category": "test",
        "biomarker": "GENE",
        "full_name": "GENE",
        "marker_type": "protein",
        "role_summary": "test",
        "source_direction": "test",
        "route": "direct_target",
        "entity_type": "protein",
        "entity_name": "GENE",
        "gene_symbol": "GENE",
        "uniprot_id": "P00001",
        "protein_form": None,
        "component_of": None,
        "desired_effect": "inhibit_function",
        "target_scope": "skin",
        "skin_compartment": "dermis",
        "readout": "enzymatic activity",
        "source_evidence": {
            "workbook_path": "test.xlsx",
            "workbook_sha256": "a" * 64,
            "sheet": "test",
            "row_number": 1,
            "gene_map_path": "gene_map.csv",
            "gene_map_sha256": "b" * 64,
            "registry_path": "registry.csv",
            "registry_sha256": "c" * 64,
            "curated_on": "2026-01-01",
        },
        "target_taxid": 9606,
        "role": "material_candidate",
        "docking_eligible": True,
    }


def test_review_applicability_requires_evidence_but_is_not_exclusion() -> None:
    result = ti.decide_candidate(
        _synthetic_priority_intent(),
        _priority_summary(
            analysis_applicability="review", analysis_scope_supported=True
        ),
        _pass_safety(),
    )

    assert result["decision"] == "needs_evidence"
    assert any("requires review" in reason for reason in result["reasons"])
    assert result["decision"] != "unsupported"
    assert result["decision"] != "exclude"


def test_out_of_scope_applicability_is_unsupported() -> None:
    result = ti.decide_candidate(
        _synthetic_priority_intent(),
        _priority_summary(
            analysis_applicability="out_of_scope", analysis_scope_supported=False
        ),
        _pass_safety(),
    )

    assert result["decision"] == "unsupported"
    assert any("outside the supported scope" in reason for reason in result["reasons"])


def test_in_scope_without_the_support_flag_requires_evidence() -> None:
    result = ti.decide_candidate(
        _synthetic_priority_intent(),
        _priority_summary(analysis_scope_supported=False),
        _pass_safety(),
    )

    assert result["decision"] == "needs_evidence"
    assert any("support is not established" in reason for reason in result["reasons"])


def test_invalid_scope_support_flag_fails_closed() -> None:
    with pytest.raises(ValueError, match="analysis_scope_supported"):
        ti.decide_candidate(
            _synthetic_priority_intent(),
            _priority_summary(analysis_scope_supported="yes"),
            _pass_safety(),
        )
