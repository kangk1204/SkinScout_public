"""Tests for the preregistered E0-E3 candidate matrix contract."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "eval" / "preregistered_candidate_matrix.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("preregistered_candidate_matrix", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["preregistered_candidate_matrix"] = module
    spec.loader.exec_module(module)
    return module


matrix = _load_module()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    training = tmp_path / "training.json"
    target = tmp_path / "targets.json"
    bench = tmp_path / "benchmark.json"
    training.write_text('{"cutoff":"2023-12-31"}\n', encoding="utf-8")
    target.write_text('{"targets":["T1"]}\n', encoding="utf-8")
    bench.write_text('{"benchmark":["B1"]}\n', encoding="utf-8")
    return training, target, bench


def _contract(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    training, target, bench = _write_artifacts(tmp_path)
    payload: dict[str, Any] = {
        "schema_version": matrix.SCHEMA_VERSION,
        "candidates": matrix.REQUIRED_CANDIDATES,
        "training_window": "train_through_2023",
        "selection_window": "select_2024_dev_only",
        "artifacts": {
            "training_data": {"path": str(training), "sha256": _sha(training)},
            "target_universe": {"path": str(target), "sha256": _sha(target)},
            "benchmark": {"path": str(bench), "sha256": _sha(bench)},
        },
        "leakage_thresholds": {"seq": 0.30, "ligand": 0.50, "pocket": 0.50},
        "comparators": {
            "E0": {"id": "current-baseline", "license": "internal", "version": "frozen", "model_sha256": _digest("E0")},
            "E1": {"id": "public-dual-encoder", "license": "MIT", "version": "1.0", "model_sha256": _digest("E1")},
            "E2": {"id": "uni-mol", "license": "MIT", "version": "frozen", "model_sha256": _digest("E2")},
            "E3": {"id": "diffdock", "license": "MIT", "version": "frozen", "model_sha256": _digest("E3")},
        },
        "seeds": [3, 5, 7],
        "grid": {"dock": [64, 128, 256], "rerank": [10, 20, 40], "diffdock": [0, 5]},
        "objective_order": [
            {"metric": "mrr", "direction": "max"},
            {"metric": "cold_p95_minutes", "direction": "min"},
            {"metric": "bundle_gb", "direction": "min"},
        ],
        "guardrails": {
            "top30_regression_max": 0.01,
            "brier_regression_max": 0.005,
            "log_loss_regression_max": 0.005,
        },
        "run_budget": {"max_runs": 37, "max_gpu_hours": 1},
    }
    payload.update(overrides)
    return payload


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(matrix.REQUIRED_RESULT_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def _result_rows(prereg: dict[str, Any], **metric_overrides: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in matrix.expected_matrix():
        config_id = item["config_id"]
        row = {
            **item,
            "run_id": f"run-{config_id}",
            "model_sha256": prereg["comparators"][item["candidate_id"]]["model_sha256"],
            "data_sha256": prereg["artifacts"]["training_data"]["sha256"],
            "target_universe_sha256": prereg["artifacts"]["target_universe"]["sha256"],
            "benchmark_sha256": prereg["artifacts"]["benchmark"]["sha256"],
            "seeds_sha256": matrix.canonical_sha256(prereg["seeds"]),
            "top30": 0.50,
            "top10": 0.20,
            "mrr": 0.40,
            "brier": 0.20,
            "log_loss": 0.30,
            "cold_p95_minutes": 12.0,
            "bundle_gb": 4.0,
        }
        if item["candidate_id"] == "E0":
            row.update({"top30": 0.50, "mrr": 0.40, "cold_p95_minutes": 10.0, "bundle_gb": 2.0})
        row.update(metric_overrides.get(config_id, {}))
        rows.append(row)
    return rows


def _seal(tmp_path: Path) -> tuple[dict[str, Any], Path]:
    prereg = matrix.seal_preregistration_contract(_contract(tmp_path))
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(prereg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return prereg, path


def test_exact_matrix_cardinality_and_expansion() -> None:
    expanded = matrix.expected_matrix()
    assert len(expanded) == 37
    assert expanded[0] == {"candidate_id": "E0", "config_id": "E0_baseline", "dock": 0, "rerank": 0, "diffdock": 0}
    assert len([r for r in expanded if r["candidate_id"] == "E1"]) == 9
    assert len([r for r in expanded if r["candidate_id"] == "E2"]) == 9
    assert len([r for r in expanded if r["candidate_id"] == "E3"]) == 18
    assert len([r for r in expanded if r["candidate_id"] == "E3" and r["diffdock"] == 0]) == 9
    assert len([r for r in expanded if r["candidate_id"] == "E3" and r["diffdock"] == 5]) == 9


def test_successful_selection_and_lexicographic_tie_break(tmp_path: Path) -> None:
    prereg, prereg_path = _seal(tmp_path)
    csv_path = tmp_path / "results.csv"
    _write_csv(
        csv_path,
        _result_rows(
            prereg,
            E1_dock64_rerank10_diffdock0={"mrr": 0.70, "cold_p95_minutes": 5, "bundle_gb": 3},
            E2_dock64_rerank10_diffdock0={"mrr": 0.70, "cold_p95_minutes": 5, "bundle_gb": 3},
            E3_dock64_rerank10_diffdock5={"mrr": 0.70, "cold_p95_minutes": 5, "bundle_gb": 4},
        ),
    )

    selection = matrix.evaluate_results(prereg_path, csv_path)

    assert selection["status"] == "select_candidate"
    assert selection["selected"]["config_id"] == "E1_dock64_rerank10_diffdock0"
    matrix.validate_selection_manifest(selection, prereg_json=prereg_path, results_csv=csv_path)


@pytest.mark.parametrize(
    ("config_id", "override", "reason"),
    [
        ("E1_dock64_rerank10_diffdock0", {"top30": 0.489}, "top30_regression_guardrail"),
        ("E1_dock64_rerank10_diffdock0", {"brier": 0.206}, "brier_regression_guardrail"),
        ("E1_dock64_rerank10_diffdock0", {"log_loss": 0.306}, "log_loss_regression_guardrail"),
    ],
)
def test_guardrails_reject_candidate(tmp_path: Path, config_id: str, override: dict[str, Any], reason: str) -> None:
    prereg, prereg_path = _seal(tmp_path)
    csv_path = tmp_path / "results.csv"
    _write_csv(csv_path, _result_rows(prereg, **{config_id: override}))

    selection = matrix.evaluate_results(prereg_path, csv_path)

    rejected = {row["config_id"]: row["reasons"] for row in selection["rejected"]}
    assert reason in rejected[config_id]


def test_no_eligible_retains_e0(tmp_path: Path) -> None:
    prereg, prereg_path = _seal(tmp_path)
    csv_path = tmp_path / "results.csv"
    bad = {row["config_id"]: {"top30": 0.0} for row in matrix.expected_matrix() if row["candidate_id"] != "E0"}
    _write_csv(csv_path, _result_rows(prereg, **bad))

    selection = matrix.evaluate_results(prereg_path, csv_path)

    assert selection["status"] == "retain_E0"
    assert selection["selected"] is None


def test_candidates_worse_than_e0_retain_e0(tmp_path: Path) -> None:
    prereg, prereg_path = _seal(tmp_path)
    csv_path = tmp_path / "results.csv"
    rows = _result_rows(prereg)
    for row in rows:
        if row["candidate_id"] != "E0":
            row["mrr"] = 0.39
    _write_csv(csv_path, rows)

    selection = matrix.evaluate_results(prereg_path, csv_path)

    assert selection["status"] == "retain_E0"
    assert selection["selected"] is None


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("top30", 1.01),
        ("top10", -0.01),
        ("mrr", 1.01),
        ("brier", -0.01),
        ("log_loss", -0.01),
        ("cold_p95_minutes", -0.01),
        ("bundle_gb", -0.01),
    ],
)
def test_metric_ranges_fail_closed(tmp_path: Path, column: str, value: float) -> None:
    prereg, prereg_path = _seal(tmp_path)
    csv_path = tmp_path / "results.csv"
    _write_csv(
        csv_path,
        _result_rows(prereg, E1_dock64_rerank10_diffdock0={column: value}),
    )

    with pytest.raises(matrix.CandidateMatrixError, match=column):
        matrix.evaluate_results(prereg_path, csv_path)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("data_sha256", _digest("wrong-data"), "data_sha256"),
        ("model_sha256", _digest("wrong-model"), "model_sha256"),
        ("seeds_sha256", _digest("wrong-seeds"), "seeds_sha256"),
    ],
)
def test_result_provenance_must_match_preregistration(
    tmp_path: Path,
    column: str,
    value: str,
    message: str,
) -> None:
    prereg, prereg_path = _seal(tmp_path)
    csv_path = tmp_path / "results.csv"
    _write_csv(
        csv_path,
        _result_rows(prereg, E1_dock64_rerank10_diffdock0={column: value}),
    )

    with pytest.raises(matrix.CandidateMatrixError, match=message):
        matrix.evaluate_results(prereg_path, csv_path)


def test_duplicate_run_id_fails_closed(tmp_path: Path) -> None:
    prereg, prereg_path = _seal(tmp_path)
    csv_path = tmp_path / "results.csv"
    rows = _result_rows(prereg)
    rows[1]["run_id"] = rows[0]["run_id"]
    _write_csv(csv_path, rows)

    with pytest.raises(matrix.CandidateMatrixError, match="run_id must be unique"):
        matrix.evaluate_results(prereg_path, csv_path)


def test_missing_extra_and_duplicate_config_fail_closed(tmp_path: Path) -> None:
    prereg, prereg_path = _seal(tmp_path)
    csv_path = tmp_path / "results.csv"
    rows = _result_rows(prereg)
    _write_csv(csv_path, rows[:-1])
    with pytest.raises(matrix.CandidateMatrixError, match="exactly 37 rows"):
        matrix.evaluate_results(prereg_path, csv_path)

    extra = dict(rows[-1], candidate_id="E9", config_id="E9_extra")
    _write_csv(csv_path, rows[:-1] + [extra])
    with pytest.raises(matrix.CandidateMatrixError, match="unexpected or invalid config"):
        matrix.evaluate_results(prereg_path, csv_path)

    rows[1] = dict(rows[0])
    _write_csv(csv_path, rows)
    with pytest.raises(matrix.CandidateMatrixError, match="duplicate config|missing preregistered"):
        matrix.evaluate_results(prereg_path, csv_path)


def test_dev_cutoff_drift_rejected(tmp_path: Path) -> None:
    bad = _contract(tmp_path, selection_window="select_2025_dev_only")
    with pytest.raises(matrix.CandidateMatrixError, match="selection_window"):
        matrix.seal_preregistration_contract(bad)


def test_target_and_benchmark_hash_drift_rejected(tmp_path: Path) -> None:
    payload = _contract(tmp_path)
    target = Path(payload["artifacts"]["target_universe"]["path"])
    target.write_text("changed\n", encoding="utf-8")
    with pytest.raises(matrix.CandidateMatrixError, match="target_universe.*drift"):
        matrix.seal_preregistration_contract(payload)

    payload = _contract(tmp_path)
    bench = Path(payload["artifacts"]["benchmark"]["path"])
    bench.write_text("changed\n", encoding="utf-8")
    with pytest.raises(matrix.CandidateMatrixError, match="benchmark.*drift"):
        matrix.seal_preregistration_contract(payload)

    payload = _contract(tmp_path)
    training = Path(payload["artifacts"]["training_data"]["path"])
    training.write_text("changed\n", encoding="utf-8")
    with pytest.raises(matrix.CandidateMatrixError, match="training_data.*drift"):
        matrix.seal_preregistration_contract(payload)


def test_prereg_tamper_rejected(tmp_path: Path) -> None:
    prereg, _ = _seal(tmp_path)
    prereg["seeds"] = [3, 5, 9]
    with pytest.raises(matrix.CandidateMatrixError, match="tamper|mismatch"):
        matrix.validate_preregistration_contract(prereg)


def test_nonfinite_metrics_fail_closed(tmp_path: Path) -> None:
    prereg, prereg_path = _seal(tmp_path)
    csv_path = tmp_path / "results.csv"
    _write_csv(csv_path, _result_rows(prereg, E1_dock64_rerank10_diffdock0={"mrr": "nan"}))
    with pytest.raises(matrix.CandidateMatrixError, match="finite"):
        matrix.evaluate_results(prereg_path, csv_path)


def test_stale_output_cleanup_on_cli_failure(tmp_path: Path) -> None:
    out = tmp_path / "selection.json"
    out.write_text("stale\n", encoding="utf-8")
    res = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "evaluate",
            "--prereg-json",
            str(tmp_path / "missing.json"),
            "--results-csv",
            str(tmp_path / "missing.csv"),
            "--out-json",
            str(out),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode != 0
    assert not out.exists()


def test_selection_manifest_tamper_rejected(tmp_path: Path) -> None:
    prereg, prereg_path = _seal(tmp_path)
    csv_path = tmp_path / "results.csv"
    _write_csv(csv_path, _result_rows(prereg))
    selection = matrix.evaluate_results(prereg_path, csv_path)
    selection["evaluated"][0]["mrr"] = 0.99
    with pytest.raises(matrix.CandidateMatrixError, match="binding_sha256"):
        matrix.validate_selection_manifest(selection)


def test_deterministic_output(tmp_path: Path) -> None:
    prereg, prereg_path = _seal(tmp_path)
    csv_path = tmp_path / "results.csv"
    _write_csv(csv_path, _result_rows(prereg, E3_dock128_rerank20_diffdock5={"mrr": 0.9}))

    first = matrix.evaluate_results(prereg_path, csv_path)
    second = matrix.evaluate_results(prereg_path, csv_path)

    assert first == second
    assert matrix.canonical_sha256({k: v for k, v in first.items() if k != "binding_sha256"}) == first["binding_sha256"]


def test_cli_create_and_evaluate(tmp_path: Path) -> None:
    contract = tmp_path / "contract.json"
    prereg_path = tmp_path / "prereg.json"
    selection_path = tmp_path / "selection.json"
    contract.write_text(json.dumps(_contract(tmp_path), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    create = subprocess.run(
        [sys.executable, str(MODULE_PATH), "create", "--contract-json", str(contract), "--out-json", str(prereg_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert create.returncode == 0, create.stderr
    prereg = _json(prereg_path)
    matrix.validate_preregistration_contract(prereg)
    csv_path = tmp_path / "results.csv"
    _write_csv(csv_path, _result_rows(prereg))
    evaluate = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "evaluate",
            "--prereg-json",
            str(prereg_path),
            "--results-csv",
            str(csv_path),
            "--out-json",
            str(selection_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert evaluate.returncode == 0, evaluate.stderr
    matrix.validate_selection_manifest(_json(selection_path), prereg_json=prereg_path, results_csv=csv_path)
