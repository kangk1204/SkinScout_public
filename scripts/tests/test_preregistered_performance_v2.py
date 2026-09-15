"""Regression tests for the additive performance-v2 preregistration gate."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "eval" / "preregistered_performance_v2.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("preregistered_performance_v2", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["preregistered_performance_v2"] = module
    spec.loader.exec_module(module)
    return module


v2 = _load_module()


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contract(tmp_path: Path) -> dict[str, Any]:
    artifacts: dict[str, dict[str, str]] = {}
    for name in sorted(v2.REQUIRED_ARTIFACTS):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"artifact": name}) + "\n", encoding="utf-8")
        artifacts[name] = {"path": str(path), "sha256": _sha(path)}
    models = {}
    for row in v2.REQUIRED_CANDIDATES:
        candidate_id = row["candidate_id"]
        models[candidate_id] = {
            "model_id": f"model-{candidate_id}",
            "version": "frozen-test",
            "license": "test-only",
            "license_profile": "research",
            "commercial_fallback_id": "ecfp-esm2-head",
            "model_sha256": _digest(candidate_id),
        }
    return {
        "schema_version": v2.SCHEMA_VERSION,
        "candidates": v2.REQUIRED_CANDIDATES,
        "training_window": v2.TRAINING_WINDOW,
        "selection_window": v2.SELECTION_WINDOW,
        "artifacts": artifacts,
        "models": models,
        "seeds": v2.SEEDS,
        "label_policy": v2.LABEL_POLICY,
        "budget": v2.BUDGET,
        "dev_gates": v2.DEV_GATES,
    }


def _seal(tmp_path: Path) -> tuple[dict[str, Any], Path]:
    payload = v2.seal_contract(_contract(tmp_path))
    path = tmp_path / "performance_v2_preregistration.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload, path


def _rows(preregistration: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = {
        "P0": (0.20, 0.40, 0.30, 0.20, 0.40, 0.04, 4, 2, 4.0, 0.0, 0, 5.0),
        "P1": (0.21, 0.41, 0.31, 0.19, 0.39, 0.06, 6, 4, 6.0, 8.0, 9_000_000, 8.0),
        "P2": (0.22, 0.42, 0.32, 0.18, 0.38, 0.07, 7, 4, 7.0, 9.0, 9_500_000, 11.0),
        "P3": (0.23, 0.43, 0.33, 0.17, 0.37, 0.08, 8, 5, 8.0, 11.0, 9_900_000, 15.0),
    }
    rows = []
    for candidate_id, values in metrics.items():
        (
            top10,
            top30,
            mrr,
            brier,
            log_loss,
            dual_top30,
            truth_hits,
            unseen_clusters,
            effective_targets,
            peak_vram,
            trainable_parameters,
            latency,
        ) = values
        rows.append(
            {
                "candidate_id": candidate_id,
                "run_id": f"fixture-{candidate_id}",
                "model_sha256": preregistration["models"][candidate_id]["model_sha256"],
                "training_data_sha256": preregistration["artifacts"]["training_data"]["sha256"],
                "target_universe_sha256": preregistration["artifacts"]["target_universe"]["sha256"],
                "dev_benchmark_sha256": preregistration["artifacts"]["dev_ranking_queries"]["sha256"],
                "seeds_sha256": v2.canonical_sha256(v2.SEEDS),
                "top10": top10,
                "top30": top30,
                "mrr": mrr,
                "brier": brier,
                "log_loss": log_loss,
                "dual_cold_target_macro_top30": dual_top30,
                "dual_cold_truth_hits": truth_hits,
                "dual_cold_unseen_target_clusters": unseen_clusters,
                "effective_target_count": effective_targets,
                "peak_vram_gib": peak_vram,
                "trainable_parameters": trainable_parameters,
                "cold_p95_minutes": latency,
            }
        )
    return rows


def _write_results(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(v2.REQUIRED_RESULT_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def test_contract_is_hash_bound_to_artifacts_and_active_validator(tmp_path: Path) -> None:
    preregistration, preregistration_path = _seal(tmp_path)

    assert v2.validate_sealed_contract(preregistration) == preregistration

    training_path = Path(preregistration["artifacts"]["training_data"]["path"])
    training_path.write_text('{"artifact":"drifted"}\n', encoding="utf-8")
    with pytest.raises(v2.PerformanceV2ContractError, match="drift"):
        v2.validate_sealed_contract(json.loads(preregistration_path.read_text(encoding="utf-8")))


def test_selects_best_eligible_candidate_and_validates_binding(tmp_path: Path) -> None:
    _, preregistration_path = _seal(tmp_path)
    preregistration = json.loads(preregistration_path.read_text(encoding="utf-8"))
    results_path = tmp_path / "results.csv"
    _write_results(results_path, _rows(preregistration))

    selection = v2.evaluate_results(preregistration_path, results_path)

    assert selection["status"] == "select_candidate"
    assert selection["selected"]["candidate_id"] == "P3"
    assert v2.validate_selection(
        selection,
        preregistration_path=preregistration_path,
        results_path=results_path,
    ) == selection


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("top10", 0.20, "strict_top10_improvement"),
        ("top30", 0.40, "strict_top30_improvement"),
        ("mrr", 0.30, "strict_mrr_improvement"),
        ("brier", 0.20, "strict_brier_improvement"),
        ("log_loss", 0.40, "strict_log_loss_improvement"),
        ("dual_cold_truth_hits", 4, "dual_cold_truth_hits"),
        ("effective_target_count", 4.99, "effective_target_count"),
        ("peak_vram_gib", 11.51, "peak_vram_budget"),
        ("trainable_parameters", 10_000_001, "trainable_parameter_budget"),
    ],
)
def test_candidate_must_clear_strict_quality_and_resource_gates(
    tmp_path: Path,
    field: str,
    value: Any,
    reason: str,
) -> None:
    preregistration, preregistration_path = _seal(tmp_path)
    rows = _rows(preregistration)
    for row in rows:
        if row["candidate_id"] == "P3":
            row[field] = value
    results_path = tmp_path / "results.csv"
    _write_results(results_path, rows)

    selection = v2.evaluate_results(preregistration_path, results_path)
    evaluated = {row["candidate_id"]: row for row in selection["evaluated"]}

    assert evaluated["P3"]["eligible"] is False
    assert reason in evaluated["P3"]["reasons"]


def test_result_provenance_mismatch_fails_closed(tmp_path: Path) -> None:
    preregistration, preregistration_path = _seal(tmp_path)
    rows = _rows(preregistration)
    rows[1]["training_data_sha256"] = _digest("wrong-training-data")
    results_path = tmp_path / "results.csv"
    _write_results(results_path, rows)

    with pytest.raises(v2.PerformanceV2ContractError, match="training_data_sha256"):
        v2.evaluate_results(preregistration_path, results_path)


def test_duplicate_result_header_cannot_shadow_metric(tmp_path: Path) -> None:
    preregistration, preregistration_path = _seal(tmp_path)
    rows = _rows(preregistration)
    results_path = tmp_path / "results.csv"
    fieldnames = [*v2.REQUIRED_RESULT_COLUMNS, "top10"]
    with results_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        for row in rows:
            writer.writerow([*[row[field] for field in v2.REQUIRED_RESULT_COLUMNS], 0.99])

    with pytest.raises(v2.PerformanceV2ContractError, match="columns"):
        v2.evaluate_results(preregistration_path, results_path)


def test_selection_manifest_tamper_fails_closed(tmp_path: Path) -> None:
    preregistration, preregistration_path = _seal(tmp_path)
    results_path = tmp_path / "results.csv"
    _write_results(results_path, _rows(preregistration))
    selection = v2.evaluate_results(preregistration_path, results_path)
    selection["selected"]["top30"] = 1.0

    with pytest.raises(v2.PerformanceV2ContractError, match="binding_sha256"):
        v2.validate_selection(selection)


def test_invalid_p0_cannot_be_retained_as_fallback(tmp_path: Path) -> None:
    preregistration, preregistration_path = _seal(tmp_path)
    rows = _rows(preregistration)
    rows[0]["peak_vram_gib"] = 11.51
    results_path = tmp_path / "results.csv"
    _write_results(results_path, rows)

    with pytest.raises(v2.PerformanceV2ContractError, match="P0 fallback"):
        v2.evaluate_results(preregistration_path, results_path)


def test_duplicate_candidate_run_ids_fail_closed(tmp_path: Path) -> None:
    preregistration, preregistration_path = _seal(tmp_path)
    rows = _rows(preregistration)
    rows[1]["run_id"] = rows[0]["run_id"]
    results_path = tmp_path / "results.csv"
    _write_results(results_path, rows)

    with pytest.raises(v2.PerformanceV2ContractError, match="run_id"):
        v2.evaluate_results(preregistration_path, results_path)


def test_selection_cannot_be_reused_after_results_drift(tmp_path: Path) -> None:
    preregistration, preregistration_path = _seal(tmp_path)
    rows = _rows(preregistration)
    results_path = tmp_path / "results.csv"
    _write_results(results_path, rows)
    selection = v2.evaluate_results(preregistration_path, results_path)
    rows[1]["cold_p95_minutes"] = 999
    _write_results(results_path, rows)

    with pytest.raises(v2.PerformanceV2ContractError, match="results input is stale"):
        v2.validate_selection(
            selection,
            preregistration_path=preregistration_path,
            results_path=results_path,
        )
