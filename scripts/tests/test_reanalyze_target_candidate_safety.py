from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from reanalyze_target_candidate_safety import (
    canonical_identity,
    consensus_from_payloads,
    fetch_with_retry,
    model_evidence_rank,
    reconcile_safety_batches,
    reparse_historical_model,
    safety_consensus_is_valid,
    sha256_file,
)


def _payload(call: str) -> dict[str, object]:
    return {"smiles": "CCO", "status": "ok", "skin_sens_call": call}


def test_exact_compound_identity_keeps_stereochemistry() -> None:
    _, first = canonical_identity("C[C@H](O)C(=O)O")
    _, second = canonical_identity("C[C@@H](O)C(=O)O")
    assert first != second


def test_husspred_outside_negative_cannot_create_pass() -> None:
    husspred = _payload("negative")
    husspred["applicability_domain"] = {"status": "outside"}
    report = consensus_from_payloads(
        {
            "husspred": husspred,
            "stoptox": _payload("negative"),
            "pred_skin": {
                "smiles": "CCO",
                "status": "ok",
                "consensus_call": "negative",
            },
        }
    )
    assert report["calls"]["husspred"] is None
    assert report["decision"] == "FLAG_HIGH"
    assert report["missing_models"] == ["husspred"]


def test_positive_husspred_votes_even_when_outside_domain() -> None:
    husspred = _payload("positive")
    husspred["applicability_domain"] = {"status": "outside"}
    report = consensus_from_payloads(
        {
            "husspred": husspred,
            "stoptox": _payload("positive"),
            "pred_skin": {
                "smiles": "CCO",
                "status": "ok",
                "consensus_call": "negative",
            },
        }
    )
    assert report["calls"]["husspred"] == "positive"
    assert report["decision"] == "HALT"


def test_fetch_failure_is_explicit_after_bounded_attempts() -> None:
    calls = 0

    def broken_fetcher(name: str, smiles: str) -> dict[str, object]:
        nonlocal calls
        calls += 1
        raise TimeoutError(f"offline: {name} {smiles}")

    result = fetch_with_retry("stoptox", "CCO", attempts=2, fetcher=broken_fetcher)
    assert calls == 2
    assert result["payload"]["status"] == "unavailable"
    assert result["payload"]["degraded_reason"] == "stoptox_unavailable"
    assert "TimeoutError" in result["error"]


def test_two_positive_votes_produce_halt_despite_third_model_failure() -> None:
    husspred = _payload("positive")
    husspred["applicability_domain"] = {"status": "inside"}
    report = consensus_from_payloads(
        {
            "husspred": husspred,
            "stoptox": _payload("positive"),
            "pred_skin": {"smiles": "CCO", "status": "unavailable"},
        }
    )
    assert report["decision"] == "HALT"
    assert report["missing_models"] == ["pred_skin"]
    assert safety_consensus_is_valid(report, scope="in_scope", web_complete=False)


def test_incomplete_non_halt_consensus_is_not_valid() -> None:
    report = {
        "decision": "FLAG_HIGH",
        "calls": {"husspred": None, "stoptox": "negative", "pred_skin": "negative"},
    }
    assert not safety_consensus_is_valid(report, scope="in_scope", web_complete=False)


def test_historical_reparse_rejects_unbound_input(tmp_path: Path) -> None:
    record = {
        "canonical_smiles": "CCO",
        "canonical_smiles_sha256": "unused",
        "historical": {"run_id": "old", "input_identity_bound": False},
    }
    assert reparse_historical_model("stoptox", record, tmp_path) is None


def _manifested_batch(parent: Path, name: str, row: dict) -> Path:
    directory = parent / name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "candidate_safety.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "skinscout.target-candidate-safety.v1",
                "candidate_count": 1,
                "artifacts": {
                    "candidate_safety_jsonl": {
                        "path": str(path),
                        "sha256": sha256_file(path),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_reconciliation_prefers_complete_base_metadata(tmp_path: Path) -> None:
    base = {
        "compound_id": "structure-sha256:abc",
        "canonical_smiles": "CCO",
        "decision": "FLAG_HIGH",
        "safety_consensus_valid": False,
        "full_analysis_complete": False,
        "run_valid": False,
        "provenance_complete": True,
        "model_results": {},
        "applicability": {"verdict": "in_scope"},
        "admet_status": "unavailable",
    }
    first = _manifested_batch(tmp_path, "first", base)
    complete = {
        **base,
        "safety_consensus_valid": True,
        "full_analysis_complete": True,
        "run_valid": True,
    }
    second = _manifested_batch(tmp_path, "second", complete)
    rows, comparisons = reconcile_safety_batches([first, second])
    assert rows[0]["decision"] == "FLAG_HIGH"
    assert comparisons[0]["base_row_from"] == str(second)


def test_reconciliation_rejects_conflicting_decisions(tmp_path: Path) -> None:
    base = {"compound_id": "structure-sha256:abc", "canonical_smiles": "CCO"}
    first = _manifested_batch(tmp_path, "first", {**base, "decision": "HALT"})
    second = _manifested_batch(tmp_path, "second", {**base, "decision": "FLAG_HIGH"})
    try:
        reconcile_safety_batches([first, second])
    except SystemExit as exc:
        assert "conflicting safety decisions" in str(exc)
    else:
        raise AssertionError("conflicting decisions must fail closed")


def test_known_fresh_retrieval_beats_newer_reparse_with_unknown_retrieval() -> None:
    payload = {"status": "ok"}
    historical = {
        "execution": "historical_raw_reparsed",
        "original_service_retrieval_timestamp": None,
        "reparsed_at": "2099-01-01T00:00:00+00:00",
    }
    fresh = {
        "execution": "live_refresh",
        "retrieved_at": "2026-09-15T00:00:00+00:00",
    }
    assert model_evidence_rank(fresh, payload, 0) > model_evidence_rank(
        historical, payload, 1
    )
