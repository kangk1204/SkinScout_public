#!/usr/bin/env python3
"""Evaluate preregistered prospective Discovery promotion gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd


SCHEMA_VERSION = "skinscout.prospective-discovery-promotion.v1"
PROSPECTIVE_AFTER = date(2026, 8, 12)
MIN_COMPOUNDS = 100
MIN_TARGETS = 30
MIN_POSITIVES = 200
MIN_TARGET_MACRO_TOP30 = 0.05
MIN_LOWER_DELTA_TOP30 = 0.0
MIN_LOWER_DELTA_TOP10 = -0.01
MIN_LOWER_DELTA_MRR = -0.01
MAX_UPPER_DELTA_BRIER = 0.005
MAX_UPPER_DELTA_LOG_LOSS = 0.005
MIN_UNSEEN_POSITIVES = 5
MIN_UNSEEN_TARGET_CLUSTERS = 3
EXPECTED_GATE_NAMES = (
    "sealed_row_level_recomputation",
    "first_eligible_post_plan_release",
    "minimum_compounds",
    "minimum_targets",
    "minimum_positives",
    "target_macro_top30",
    "paired_bootstrap_lower_delta_top30",
    "paired_bootstrap_lower_delta_top10",
    "paired_bootstrap_lower_delta_mrr",
    "brier_upper_delta",
    "log_loss_upper_delta",
    "unseen_cluster_positives",
    "unseen_target_clusters",
)
EXPECTED_DECISION_SCOPE = {
    "software_release": "independent",
    "model_promotion": "prospective_gate",
    "sota_claim": "not_evaluated_by_this_gate",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _sha256_text(value: Any, label: str) -> str:
    text = str(value).strip().lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return text


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _require_nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"Prospective promotion metrics are required: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise SystemExit(f"Prospective promotion metrics failed to parse: {path}: {exc}") from exc
    if df.empty:
        raise SystemExit(f"Prospective promotion metrics contain no rows: {path}")
    return df


def _one_row(df: pd.DataFrame) -> pd.Series:
    if len(df) != 1:
        raise SystemExit("Prospective promotion metrics must contain exactly one preregistered row")
    return df.iloc[0]


def _text(row: pd.Series, column: str) -> str:
    if column not in row.index:
        raise SystemExit(f"Prospective promotion metrics missing column '{column}'")
    value = "" if pd.isna(row[column]) else str(row[column]).strip()
    if not value:
        raise SystemExit(f"Prospective promotion metrics column '{column}' is blank")
    return value


def _finite(row: pd.Series, column: str) -> float:
    if column not in row.index:
        raise SystemExit(f"Prospective promotion metrics missing column '{column}'")
    value = row[column]
    if isinstance(value, bool) or type(value).__name__ == "bool_":
        raise SystemExit(f"Prospective promotion metrics column '{column}' must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Prospective promotion metrics column '{column}' must be numeric") from exc
    if not math.isfinite(parsed):
        raise SystemExit(f"Prospective promotion metrics column '{column}' must be finite")
    return parsed


def _nonnegative_int(row: pd.Series, column: str) -> int:
    parsed = _finite(row, column)
    if parsed < 0 or not parsed.is_integer():
        raise SystemExit(f"Prospective promotion metrics column '{column}' must be a non-negative integer")
    return int(parsed)


def _release_date(row: pd.Series) -> date:
    text = _text(row, "prospective_release_date")
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise SystemExit("prospective_release_date must be an ISO date") from exc
    return parsed


def evaluate(path: Path) -> dict[str, Any]:
    df = _read_csv(path)
    row = _one_row(df)
    gates = [
        {
            "name": "sealed_row_level_recomputation",
            "observed": False,
            "threshold": True,
            "passed": False,
            "reason": (
                "aggregate metrics cannot authorize model promotion; a future evaluator "
                "must recompute every gate from sealed row-level truth and predictions"
            ),
        },
        {
            "name": "first_eligible_post_plan_release",
            "observed": _release_date(row).isoformat(),
            "threshold": f">{PROSPECTIVE_AFTER.isoformat()}",
            "passed": _release_date(row) > PROSPECTIVE_AFTER,
        },
        {
            "name": "minimum_compounds",
            "observed": _nonnegative_int(row, "n_compounds"),
            "threshold": MIN_COMPOUNDS,
            "passed": _nonnegative_int(row, "n_compounds") >= MIN_COMPOUNDS,
        },
        {
            "name": "minimum_targets",
            "observed": _nonnegative_int(row, "n_targets"),
            "threshold": MIN_TARGETS,
            "passed": _nonnegative_int(row, "n_targets") >= MIN_TARGETS,
        },
        {
            "name": "minimum_positives",
            "observed": _nonnegative_int(row, "n_positives"),
            "threshold": MIN_POSITIVES,
            "passed": _nonnegative_int(row, "n_positives") >= MIN_POSITIVES,
        },
        {
            "name": "target_macro_top30",
            "observed": _finite(row, "target_macro_top30"),
            "threshold": MIN_TARGET_MACRO_TOP30,
            "passed": _finite(row, "target_macro_top30") >= MIN_TARGET_MACRO_TOP30,
        },
        {
            "name": "paired_bootstrap_lower_delta_top30",
            "observed": _finite(row, "paired_bootstrap_lower_delta_top30"),
            "threshold": f">{MIN_LOWER_DELTA_TOP30}",
            "passed": _finite(row, "paired_bootstrap_lower_delta_top30") > MIN_LOWER_DELTA_TOP30,
        },
        {
            "name": "paired_bootstrap_lower_delta_top10",
            "observed": _finite(row, "paired_bootstrap_lower_delta_top10"),
            "threshold": MIN_LOWER_DELTA_TOP10,
            "passed": _finite(row, "paired_bootstrap_lower_delta_top10") >= MIN_LOWER_DELTA_TOP10,
        },
        {
            "name": "paired_bootstrap_lower_delta_mrr",
            "observed": _finite(row, "paired_bootstrap_lower_delta_mrr"),
            "threshold": MIN_LOWER_DELTA_MRR,
            "passed": _finite(row, "paired_bootstrap_lower_delta_mrr") >= MIN_LOWER_DELTA_MRR,
        },
        {
            "name": "brier_upper_delta",
            "observed": _finite(row, "brier_upper_delta"),
            "threshold": MAX_UPPER_DELTA_BRIER,
            "passed": _finite(row, "brier_upper_delta") <= MAX_UPPER_DELTA_BRIER,
        },
        {
            "name": "log_loss_upper_delta",
            "observed": _finite(row, "log_loss_upper_delta"),
            "threshold": MAX_UPPER_DELTA_LOG_LOSS,
            "passed": _finite(row, "log_loss_upper_delta") <= MAX_UPPER_DELTA_LOG_LOSS,
        },
        {
            "name": "unseen_cluster_positives",
            "observed": _nonnegative_int(row, "unseen_cluster_positives"),
            "threshold": MIN_UNSEEN_POSITIVES,
            "passed": _nonnegative_int(row, "unseen_cluster_positives") >= MIN_UNSEEN_POSITIVES,
        },
        {
            "name": "unseen_target_clusters",
            "observed": _nonnegative_int(row, "unseen_target_clusters"),
            "threshold": MIN_UNSEEN_TARGET_CLUSTERS,
            "passed": _nonnegative_int(row, "unseen_target_clusters") >= MIN_UNSEEN_TARGET_CLUSTERS,
        },
    ]
    for gate in gates:
        gate["status"] = "passed" if gate["passed"] else "failed"
    required_text = {
        "candidate_model_id": _text(row, "candidate_model_id"),
        "previous_signed_model_id": _text(row, "previous_signed_model_id"),
        "preregistration_sha256": _text(row, "preregistration_sha256"),
        "data_cutoff": _text(row, "data_cutoff"),
        "training_window": _text(row, "training_window"),
        "dev_selection_window": _text(row, "dev_selection_window"),
        "prospective_source_releases": _text(row, "prospective_source_releases"),
    }
    if required_text["training_window"] != "through_2023":
        raise SystemExit("training_window must be through_2023")
    if required_text["dev_selection_window"] != "2024_dev_only":
        raise SystemExit("dev_selection_window must be 2024_dev_only")
    digest = required_text["preregistration_sha256"].lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise SystemExit("preregistration_sha256 must be a SHA-256 digest")
    passed = all(bool(gate["passed"]) for gate in gates)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "promote" if passed else "retain_previous_signed_model",
        "decision_scope": dict(EXPECTED_DECISION_SCOPE),
        "evidence_scope": "aggregate_diagnostics_only_not_promotion_authority",
        "input": {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        },
        "metadata": required_text,
        "thresholds": {
            "target_macro_top30_min": MIN_TARGET_MACRO_TOP30,
            "paired_bootstrap_lower_delta_top30": f">{MIN_LOWER_DELTA_TOP30}",
            "paired_bootstrap_lower_delta_top10_min": MIN_LOWER_DELTA_TOP10,
            "paired_bootstrap_lower_delta_mrr_min": MIN_LOWER_DELTA_MRR,
            "brier_upper_delta_max": MAX_UPPER_DELTA_BRIER,
            "log_loss_upper_delta_max": MAX_UPPER_DELTA_LOG_LOSS,
            "minimum_unseen_cluster_positives": MIN_UNSEEN_POSITIVES,
            "minimum_unseen_target_clusters": MIN_UNSEEN_TARGET_CLUSTERS,
        },
        "gates": gates,
        "failed_gates": [gate for gate in gates if not gate["passed"]],
    }
    validate_manifest(payload, metrics_csv=path)
    return payload


def validate_manifest(
    payload: dict[str, Any],
    *,
    metrics_csv: Path | None = None,
) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    status = _require_nonempty_text(payload.get("status"), "status")
    if status not in {"promote", "retain_previous_signed_model"}:
        raise ValueError(f"status has invalid value: {status}")
    if payload.get("decision_scope") != EXPECTED_DECISION_SCOPE:
        raise ValueError("decision_scope must match prospective promotion contract")
    if payload.get("evidence_scope") != "aggregate_diagnostics_only_not_promotion_authority":
        raise ValueError("aggregate prospective evidence scope cannot authorize promotion")
    input_meta = _require_object(payload.get("input"), "input")
    input_path = Path(_require_nonempty_text(input_meta.get("path"), "input.path"))
    _require_positive_int(input_meta.get("bytes"), "input.bytes")
    _sha256_text(input_meta.get("sha256"), "input.sha256")
    active_input_path = metrics_csv if metrics_csv is not None else input_path
    if metrics_csv is not None and input_path != metrics_csv:
        raise ValueError("input.path must match the active metrics CSV")
    if not active_input_path.exists() or not active_input_path.is_file():
        raise ValueError("input.path must exist for prospective manifest validation")
    if active_input_path.stat().st_size == 0:
        raise ValueError("active metrics CSV must exist and be non-empty")
    if input_meta["bytes"] != active_input_path.stat().st_size:
        raise ValueError("input.bytes must match the active metrics CSV")
    if str(input_meta["sha256"]).lower() != _sha256(active_input_path):
        raise ValueError("input.sha256 must match the active metrics CSV")
    metadata = _require_object(payload.get("metadata"), "metadata")
    for key in (
        "candidate_model_id",
        "previous_signed_model_id",
        "preregistration_sha256",
        "data_cutoff",
        "training_window",
        "dev_selection_window",
        "prospective_source_releases",
    ):
        _require_nonempty_text(metadata.get(key), f"metadata.{key}")
    if metadata["training_window"] != "through_2023":
        raise ValueError("metadata.training_window must be through_2023")
    if metadata["dev_selection_window"] != "2024_dev_only":
        raise ValueError("metadata.dev_selection_window must be 2024_dev_only")
    _sha256_text(metadata.get("preregistration_sha256"), "metadata.preregistration_sha256")
    thresholds = _require_object(payload.get("thresholds"), "thresholds")
    expected_thresholds = {
        "target_macro_top30_min": MIN_TARGET_MACRO_TOP30,
        "paired_bootstrap_lower_delta_top30": f">{MIN_LOWER_DELTA_TOP30}",
        "paired_bootstrap_lower_delta_top10_min": MIN_LOWER_DELTA_TOP10,
        "paired_bootstrap_lower_delta_mrr_min": MIN_LOWER_DELTA_MRR,
        "brier_upper_delta_max": MAX_UPPER_DELTA_BRIER,
        "log_loss_upper_delta_max": MAX_UPPER_DELTA_LOG_LOSS,
        "minimum_unseen_cluster_positives": MIN_UNSEEN_POSITIVES,
        "minimum_unseen_target_clusters": MIN_UNSEEN_TARGET_CLUSTERS,
    }
    if thresholds != expected_thresholds:
        raise ValueError("thresholds must match preregistered prospective contract")
    gates = payload.get("gates")
    if not isinstance(gates, list) or len(gates) != len(EXPECTED_GATE_NAMES):
        raise ValueError("gates must contain every preregistered gate exactly once")
    gate_names = []
    failed_from_gates = []
    for idx, gate in enumerate(gates):
        gate_obj = _require_object(gate, f"gates[{idx}]")
        name = _require_nonempty_text(gate_obj.get("name"), f"gates[{idx}].name")
        gate_names.append(name)
        if "observed" not in gate_obj or "threshold" not in gate_obj:
            raise ValueError(f"gates[{idx}] missing observed/threshold")
        passed = gate_obj.get("passed")
        if not isinstance(passed, bool):
            raise ValueError(f"gates[{idx}].passed must be boolean")
        expected_gate_status = "passed" if passed else "failed"
        if gate_obj.get("status") != expected_gate_status:
            raise ValueError(f"gates[{idx}].status must match passed")
        if not passed:
            failed_from_gates.append(gate_obj)
    if tuple(gate_names) != EXPECTED_GATE_NAMES:
        raise ValueError("gate names/order must match preregistered prospective contract")
    aggregate_gate = gates[0]
    if (
        aggregate_gate.get("observed") is not False
        or aggregate_gate.get("threshold") is not True
        or aggregate_gate.get("passed") is not False
    ):
        raise ValueError("aggregate metrics must fail the sealed row-level recomputation gate")
    failed_gates = payload.get("failed_gates")
    if not isinstance(failed_gates, list):
        raise ValueError("failed_gates must be a list")
    failed_names = [
        _require_nonempty_text(
            _require_object(gate, f"failed_gates[{idx}]").get("name"),
            f"failed_gates[{idx}].name",
        )
        for idx, gate in enumerate(failed_gates)
    ]
    if failed_names != [str(gate["name"]) for gate in failed_from_gates]:
        raise ValueError("failed_gates must exactly match failed gate results")
    if status == "promote" and failed_gates:
        raise ValueError("status=promote requires no failed_gates")
    if status == "retain_previous_signed_model" and not failed_gates:
        raise ValueError("retain_previous_signed_model requires failed_gates")


def validate_manifest_against_metrics(
    payload: dict[str, Any],
    *,
    metrics_csv: Path,
) -> None:
    """Require an exact, fresh recomputation from the active prospective metrics."""
    validate_manifest(payload, metrics_csv=metrics_csv)
    expected = evaluate(metrics_csv)
    if payload != expected:
        raise ValueError(
            "manifest does not exactly match a fresh evaluation of the active "
            "prospective metrics CSV"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument(
        "--allow-failed-promotion",
        action="store_true",
        help="Write a retain-previous-model manifest instead of exiting nonzero.",
    )
    args = parser.parse_args()
    if args.out_json.exists():
        args.out_json.unlink()
    payload = evaluate(args.metrics_csv)
    _write_json_atomic(args.out_json, payload)
    if payload["status"] != "promote" and not args.allow_failed_promotion:
        raise SystemExit(
            "Prospective Discovery model promotion failed; retain previous signed model"
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
