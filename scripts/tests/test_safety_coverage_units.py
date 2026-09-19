"""C20 regression: safety coverage keeps record, model attempt, valid
consensus (including partial two-positive HALT), and completion separate."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import build_target_lab_package as package
import target_intent as ti
from compound_applicability import assess
from target_intent import build_target_intents


@pytest.fixture(scope="module")
def intents():
    workbook = os.environ.get("SKINSCOUT_BIOMARKER_XLSX", "").strip()
    gene_map = os.environ.get("SKINSCOUT_GENE_MAP_PATH", "").strip()
    if not workbook or not gene_map:
        pytest.skip(
            "private workbook not provided "
            "(SKINSCOUT_BIOMARKER_XLSX / SKINSCOUT_GENE_MAP_PATH)"
        )
    return build_target_intents(Path(workbook), Path(gene_map))


def _source(intent_id: str, smiles: str) -> dict:
    return {
        **package.identity(smiles),
        "intent_id": intent_id,
        "evidence_count": 1,
        "unique_publication_count": 1,
        "direction_relations_json": "[]",
        "molecule_pref_name": "coverage fixture",
    }


def _record(
    row: dict,
    decision: str,
    *,
    consensus: bool,
    complete: bool,
    run_valid: bool,
    attempts: int,
) -> dict:
    scope = assess(row["canonical_isomeric_smiles"])
    return {
        "compound_id": row["compound_id"],
        "canonical_smiles": row["canonical_isomeric_smiles"],
        "applicability": scope,
        "decision": decision,
        "run_valid": run_valid,
        "safety_consensus_valid": consensus,
        "full_analysis_complete": complete,
        "applicability_domain": "outside" if scope["verdict"] == "out_of_scope" else "inside",
        "model_results": {
            f"model{index}": {"execution": "live_refresh"} for index in range(attempts)
        },
        "skin_sens": {"degraded": False},
        "admet_status": "ok",
    }


def _matrix(intents):
    rows = [
        _source("TI-020-TYR", "CC(O)C(=O)O"),
        _source("TI-015-MMP1", "OC(=O)c1ccccc1O"),
        _source("TI-016-MMP3", "CCCC(=O)O"),
        _source("TI-007-AR", "CCCCCCCCCCCCOS(=O)(=O)[O-].[Na+]"),
        _source("TI-030-KLK5", "CCO"),
    ]
    safety_rows = {
        rows[0]["compound_id"]: _record(
            rows[0], "PASS", consensus=True, complete=True, run_valid=True, attempts=3
        ),
        rows[1]["compound_id"]: _record(
            rows[1], "FLAG_HIGH", consensus=True, complete=True, run_valid=True, attempts=3
        ),
        rows[2]["compound_id"]: _record(
            rows[2], "HALT", consensus=True, complete=False, run_valid=False, attempts=2
        ),
        rows[3]["compound_id"]: _record(
            rows[3], "UNAVAILABLE", consensus=False, complete=False, run_valid=False, attempts=0
        ),
    }
    return package.build_matrix(
        intents,
        pd.DataFrame(rows),
        pd.DataFrame(),
        [],
        safety_rows,
    )


def test_pair_and_compound_denominators_are_separated(intents) -> None:
    matrix, _compounds = _matrix(intents)
    metrics = package.safety_coverage_metrics(matrix)
    assert metrics["safety_record_count"]["pairs"] == 4
    assert metrics["safety_record_count"]["compounds"] == 4
    assert metrics["web_model_attempted_count"]["pairs"] == 3
    assert metrics["web_model_attempted_count"]["compounds"] == 3
    assert metrics["safety_consensus_valid_count"]["pairs"] == 3
    assert metrics["safety_consensus_valid_count"]["compounds"] == 3
    assert metrics["full_analysis_complete_count"]["pairs"] == 2
    assert metrics["full_analysis_complete_count"]["compounds"] == 2
    for metric in metrics.values():
        assert metric["unit"] == "intent-compound pairs / exact compounds"


def test_partial_two_positive_halt_counts_as_valid_consensus_only(intents) -> None:
    matrix, _compounds = _matrix(intents)
    coverage = package.coverage_table(intents, matrix)
    row = coverage[coverage.intent_id.eq("TI-016-MMP3")].iloc[0]
    assert row.safety_record_count == 1
    assert row.web_model_attempted_count == 1
    assert row.safety_consensus_valid_count == 1
    assert row.full_analysis_complete_count == 0


def test_out_of_scope_unavailable_has_record_without_attempts(intents) -> None:
    matrix, _compounds = _matrix(intents)
    coverage = package.coverage_table(intents, matrix)
    row = coverage[coverage.intent_id.eq("TI-007-AR")].iloc[0]
    assert row.safety_record_count == 1
    assert row.web_model_attempted_count == 0
    assert row.safety_consensus_valid_count == 0
    assert row.full_analysis_complete_count == 0


def test_no_record_contributes_zero_to_every_metric(intents) -> None:
    matrix, _compounds = _matrix(intents)
    coverage = package.coverage_table(intents, matrix)
    row = coverage[coverage.intent_id.eq("TI-030-KLK5")].iloc[0]
    assert row.safety_record_count == 0
    assert row.web_model_attempted_count == 0
    assert row.safety_consensus_valid_count == 0
    assert row.full_analysis_complete_count == 0


def _priority_summary() -> dict:
    return {
        "structure_valid": True,
        "route_supported": True,
        "analysis_applicability": "in_scope",
        "direction_relation": "supports",
        "material_policy": "eligible",
        "required_information_complete": True,
        "minimum_evidence_met": True,
        "functional_evidence": True,
    }


def test_consensus_validity_is_not_inferred_from_run_valid(intents) -> None:
    intent = next(row for row in intents if row["intent_id"] == "TI-007-AR")
    result = ti.decide_candidate(
        intent,
        _priority_summary(),
        {"run_valid": True, "decision": "HALT", "applicability_domain": "inside"},
    )
    assert result["decision"] == "needs_evidence"


def test_partial_two_positive_halt_still_excludes_with_explicit_consensus(
    intents,
) -> None:
    intent = next(row for row in intents if row["intent_id"] == "TI-007-AR")
    result = ti.decide_candidate(
        intent,
        _priority_summary(),
        {
            "run_valid": False,
            "safety_consensus_valid": True,
            "full_analysis_complete": False,
            "decision": "HALT",
            "applicability_domain": "inside",
        },
    )
    assert result["decision"] == "exclude"
