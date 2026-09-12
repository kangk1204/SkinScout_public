#!/usr/bin/env python3
"""Fail-closed preregistered E0-E3 candidate matrix contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "skinscout.preregistered-candidate-matrix.v1"
SELECTION_SCHEMA_VERSION = "skinscout.candidate-selection.v1"
TRAINING_WINDOW = "train_through_2023"
SELECTION_WINDOW = "select_2024_dev_only"
DOCK_GRID = [64, 128, 256]
RERANK_GRID = [10, 20, 40]
DIFFDOCK_GRID = [0, 5]
LEAKAGE_THRESHOLDS = {"seq": 0.30, "ligand": 0.50, "pocket": 0.50}
OBJECTIVE_ORDER = [
    {"metric": "mrr", "direction": "max"},
    {"metric": "cold_p95_minutes", "direction": "min"},
    {"metric": "bundle_gb", "direction": "min"},
]
GUARDRAILS = {
    "top30_regression_max": 0.01,
    "brier_regression_max": 0.005,
    "log_loss_regression_max": 0.005,
}
FLOAT_TOLERANCE = 1e-12
UNSEALED_CONTRACT_KEYS = {
    "schema_version",
    "candidates",
    "training_window",
    "selection_window",
    "artifacts",
    "leakage_thresholds",
    "comparators",
    "seeds",
    "grid",
    "objective_order",
    "guardrails",
    "run_budget",
}
SEALED_CONTRACT_KEYS = UNSEALED_CONTRACT_KEYS | {
    "sealed",
    "builder_code_sha256",
    "preregistration_sha256",
}
REQUIRED_CANDIDATES = [
    {"candidate_id": "E0", "definition": "current baseline"},
    {"candidate_id": "E1", "definition": "frozen public dual encoder"},
    {"candidate_id": "E2", "definition": "E1+MIT Uni-Mol rerank"},
    {"candidate_id": "E3", "definition": "E2+MIT DiffDock tie-break"},
]
REQUIRED_RESULT_COLUMNS = {
    "candidate_id",
    "config_id",
    "run_id",
    "dock",
    "rerank",
    "diffdock",
    "model_sha256",
    "data_sha256",
    "target_universe_sha256",
    "benchmark_sha256",
    "seeds_sha256",
    "top30",
    "top10",
    "mrr",
    "brier",
    "log_loss",
    "cold_p95_minutes",
    "bundle_gb",
}


class CandidateMatrixError(ValueError):
    """Raised when a preregistration or selection contract fails validation."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def builder_code_sha256() -> str:
    return sha256_file(Path(__file__).resolve())


def expected_matrix() -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = [
        {"candidate_id": "E0", "config_id": "E0_baseline", "dock": 0, "rerank": 0, "diffdock": 0}
    ]
    for candidate_id, diffdock_values in (
        ("E1", [0]),
        ("E2", [0]),
        ("E3", DIFFDOCK_GRID),
    ):
        for dock in DOCK_GRID:
            for rerank in RERANK_GRID:
                for diffdock in diffdock_values:
                    rows.append(
                        {
                            "candidate_id": candidate_id,
                            "config_id": f"{candidate_id}_dock{dock}_rerank{rerank}_diffdock{diffdock}",
                            "dock": dock,
                            "rerank": rerank,
                            "diffdock": diffdock,
                        }
                    )
    return rows


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(ch in "0123456789abcdef" for ch in value)
        and len(set(value)) > 1
    )


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
        raise CandidateMatrixError(f"{label} is missing, empty, or unsafe: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CandidateMatrixError(f"{label} is invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CandidateMatrixError(f"{label} must be a JSON object: {path}")
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _artifact_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise CandidateMatrixError(f"{field}.path must be a non-empty string")
    path = Path(value)
    if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
        raise CandidateMatrixError(f"{field}.path is missing, empty, or unsafe: {value}")
    return path


def _validate_artifact(artifact: Any, field: str) -> dict[str, Any]:
    if not isinstance(artifact, dict):
        raise CandidateMatrixError(f"{field} must be an object")
    path = _artifact_path(artifact.get("path"), field)
    declared = artifact.get("sha256")
    if not _is_sha256(declared):
        raise CandidateMatrixError(f"{field}.sha256 must be a SHA-256 digest")
    active = sha256_file(path)
    if active != declared:
        raise CandidateMatrixError(f"{field}.sha256 drift: expected {declared}, active {active}")
    return {"path": str(artifact["path"]), "sha256": declared}


def _validate_comparators(value: Any) -> dict[str, dict[str, str]]:
    if not isinstance(value, dict):
        raise CandidateMatrixError("comparators must be an object")
    if set(value) != {"E0", "E1", "E2", "E3"}:
        raise CandidateMatrixError("comparators must contain exactly E0, E1, E2, E3")
    sealed: dict[str, dict[str, str]] = {}
    for candidate_id in ("E0", "E1", "E2", "E3"):
        row = value[candidate_id]
        if not isinstance(row, dict):
            raise CandidateMatrixError(f"comparators.{candidate_id} must be an object")
        if set(row) != {"id", "license", "version", "model_sha256"}:
            raise CandidateMatrixError(
                f"comparators.{candidate_id} must contain exactly id, license, "
                "version, model_sha256"
            )
        sealed[candidate_id] = {}
        for key in ("id", "license", "version"):
            item = row.get(key)
            if not isinstance(item, str) or not item.strip():
                raise CandidateMatrixError(f"comparators.{candidate_id}.{key} must be non-empty")
            sealed[candidate_id][key] = item.strip()
        model_sha256 = row.get("model_sha256")
        if not _is_sha256(model_sha256):
            raise CandidateMatrixError(
                f"comparators.{candidate_id}.model_sha256 must be a non-placeholder SHA-256 digest"
            )
        sealed[candidate_id]["model_sha256"] = model_sha256
    return sealed


def _validate_seeds(value: Any) -> list[int]:
    if not isinstance(value, list) or not value:
        raise CandidateMatrixError("seeds must be a non-empty list")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in value):
        raise CandidateMatrixError("seeds must contain only integers")
    if len(set(value)) != len(value):
        raise CandidateMatrixError("seeds must be unique")
    return list(value)


def _validate_run_budget(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"max_runs", "max_gpu_hours"}:
        raise CandidateMatrixError("run_budget must contain exactly max_runs and max_gpu_hours")
    sealed = dict(value)
    for key, item in sealed.items():
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)) or float(item) <= 0:
            raise CandidateMatrixError(f"run_budget.{key} must be a positive finite number")
    if not float(sealed["max_runs"]).is_integer() or int(sealed["max_runs"]) < len(expected_matrix()):
        raise CandidateMatrixError(
            f"run_budget.max_runs must be an integer >= {len(expected_matrix())}"
        )
    sealed["max_runs"] = int(sealed["max_runs"])
    return sealed


def validate_contract_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize an unsealed preregistration contract payload."""
    if set(payload) != UNSEALED_CONTRACT_KEYS:
        missing = sorted(UNSEALED_CONTRACT_KEYS - set(payload))
        extra = sorted(set(payload) - UNSEALED_CONTRACT_KEYS)
        raise CandidateMatrixError(f"contract keys mismatch missing={missing} extra={extra}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise CandidateMatrixError(f"schema_version must be {SCHEMA_VERSION}")
    if payload.get("candidates") != REQUIRED_CANDIDATES:
        raise CandidateMatrixError("candidates must exactly match preregistered E0-E3 definitions")
    if payload.get("training_window") != TRAINING_WINDOW:
        raise CandidateMatrixError(f"training_window must be {TRAINING_WINDOW}")
    if payload.get("selection_window") != SELECTION_WINDOW:
        raise CandidateMatrixError(f"selection_window must be {SELECTION_WINDOW}")
    if payload.get("leakage_thresholds") != LEAKAGE_THRESHOLDS:
        raise CandidateMatrixError("leakage_thresholds must be seq=.30 ligand=.50 pocket=.50")
    grid = payload.get("grid")
    if grid != {"dock": DOCK_GRID, "rerank": RERANK_GRID, "diffdock": DIFFDOCK_GRID}:
        raise CandidateMatrixError("grid must be exactly dock [64,128,256], rerank [10,20,40], diffdock [0,5]")
    if payload.get("objective_order") != OBJECTIVE_ORDER:
        raise CandidateMatrixError("objective_order must be MRR max, cold_p95_minutes min, bundle_gb min")
    if payload.get("guardrails") != GUARDRAILS:
        raise CandidateMatrixError("guardrails must match preregistered regression thresholds")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise CandidateMatrixError("artifacts must be an object")
    if set(artifacts) != {"training_data", "target_universe", "benchmark"}:
        raise CandidateMatrixError(
            "artifacts must contain exactly training_data, target_universe, benchmark"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "candidates": REQUIRED_CANDIDATES,
        "training_window": TRAINING_WINDOW,
        "selection_window": SELECTION_WINDOW,
        "artifacts": {
            "training_data": _validate_artifact(
                artifacts.get("training_data"), "artifacts.training_data"
            ),
            "target_universe": _validate_artifact(artifacts.get("target_universe"), "artifacts.target_universe"),
            "benchmark": _validate_artifact(artifacts.get("benchmark"), "artifacts.benchmark"),
        },
        "leakage_thresholds": LEAKAGE_THRESHOLDS,
        "comparators": _validate_comparators(payload.get("comparators")),
        "seeds": _validate_seeds(payload.get("seeds")),
        "grid": {"dock": DOCK_GRID, "rerank": RERANK_GRID, "diffdock": DIFFDOCK_GRID},
        "objective_order": OBJECTIVE_ORDER,
        "guardrails": GUARDRAILS,
        "run_budget": _validate_run_budget(payload.get("run_budget")),
    }


def seal_preregistration_contract(payload: dict[str, Any]) -> dict[str, Any]:
    sealed = validate_contract_payload(payload)
    sealed["sealed"] = True
    sealed["builder_code_sha256"] = builder_code_sha256()
    sealed["preregistration_sha256"] = canonical_sha256(sealed)
    return sealed


def validate_preregistration_contract(payload: dict[str, Any]) -> dict[str, Any]:
    """Public validator for an already sealed preregistration manifest."""
    if set(payload) != SEALED_CONTRACT_KEYS:
        missing = sorted(SEALED_CONTRACT_KEYS - set(payload))
        extra = sorted(set(payload) - SEALED_CONTRACT_KEYS)
        raise CandidateMatrixError(f"sealed preregistration keys mismatch missing={missing} extra={extra}")
    if payload.get("sealed") is not True:
        raise CandidateMatrixError("sealed must be true")
    if payload.get("builder_code_sha256") != builder_code_sha256():
        raise CandidateMatrixError("builder_code_sha256 does not match active validator")
    expected = seal_preregistration_contract(
        {k: v for k, v in payload.items() if k not in {"sealed", "builder_code_sha256", "preregistration_sha256"}}
    )
    if payload != expected:
        raise CandidateMatrixError("preregistration_sha256 mismatch or preregistration payload tamper")
    return expected


def _read_prereg(path: Path) -> dict[str, Any]:
    return validate_preregistration_contract(_read_json_object(path, "preregistration"))


def _finite_metric(
    row: dict[str, str],
    column: str,
    *,
    minimum: float,
    maximum: float | None = None,
) -> float:
    text = row.get(column, "")
    try:
        value = float(text)
    except (TypeError, ValueError) as exc:
        raise CandidateMatrixError(f"{column} must be numeric for {row.get('config_id', '<unknown>')}") from exc
    if not math.isfinite(value):
        raise CandidateMatrixError(f"{column} must be finite for {row.get('config_id', '<unknown>')}")
    if value < minimum or (maximum is not None and value > maximum):
        bound = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise CandidateMatrixError(
            f"{column} must be {bound} for {row.get('config_id', '<unknown>')}"
        )
    return value


def _int_field(row: dict[str, str], column: str) -> int:
    text = row.get(column, "")
    try:
        value = int(text)
    except (TypeError, ValueError) as exc:
        raise CandidateMatrixError(f"{column} must be an integer for {row.get('config_id', '<unknown>')}") from exc
    return value


def _read_results_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
        raise CandidateMatrixError(f"results CSV is missing, empty, or unsafe: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise CandidateMatrixError("results CSV is missing a header")
        columns = set(reader.fieldnames)
        if columns != REQUIRED_RESULT_COLUMNS:
            missing = sorted(REQUIRED_RESULT_COLUMNS - columns)
            extra = sorted(columns - REQUIRED_RESULT_COLUMNS)
            raise CandidateMatrixError(f"results CSV columns mismatch missing={missing} extra={extra}")
        raw_rows = list(reader)
    if len(raw_rows) != len(expected_matrix()):
        raise CandidateMatrixError(f"results CSV must contain exactly {len(expected_matrix())} rows")
    expected = {(row["candidate_id"], row["config_id"], row["dock"], row["rerank"], row["diffdock"]): row for row in expected_matrix()}
    seen: set[tuple[str, str, int, int, int]] = set()
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        parsed = {
            "candidate_id": raw["candidate_id"],
            "config_id": raw["config_id"],
            "run_id": raw["run_id"],
            "dock": _int_field(raw, "dock"),
            "rerank": _int_field(raw, "rerank"),
            "diffdock": _int_field(raw, "diffdock"),
            "model_sha256": raw["model_sha256"],
            "data_sha256": raw["data_sha256"],
            "target_universe_sha256": raw["target_universe_sha256"],
            "benchmark_sha256": raw["benchmark_sha256"],
            "seeds_sha256": raw["seeds_sha256"],
        }
        for key in ("run_id", "config_id"):
            if not isinstance(parsed[key], str) or not parsed[key].strip():
                raise CandidateMatrixError(f"{key} must be non-empty")
        for key in (
            "model_sha256",
            "data_sha256",
            "target_universe_sha256",
            "benchmark_sha256",
            "seeds_sha256",
        ):
            if not _is_sha256(parsed[key]):
                raise CandidateMatrixError(f"{key} must be a SHA-256 digest for {parsed['config_id']}")
        for metric in ("top30", "top10", "mrr", "brier"):
            parsed[metric] = _finite_metric(raw, metric, minimum=0.0, maximum=1.0)
        for metric in ("log_loss", "cold_p95_minutes", "bundle_gb"):
            parsed[metric] = _finite_metric(raw, metric, minimum=0.0)
        key = (parsed["candidate_id"], parsed["config_id"], parsed["dock"], parsed["rerank"], parsed["diffdock"])
        if key in seen:
            raise CandidateMatrixError(f"duplicate config: {parsed['config_id']}")
        if key not in expected:
            raise CandidateMatrixError(f"unexpected or invalid config: {parsed['config_id']}")
        seen.add(key)
        rows.append(parsed)
    missing = sorted(expected.keys() - seen)
    if missing:
        raise CandidateMatrixError(f"missing preregistered configs: {missing[:3]}")
    run_ids = [row["run_id"] for row in rows]
    if len(set(run_ids)) != len(run_ids):
        raise CandidateMatrixError("run_id must be unique for every preregistered config")
    return sorted(rows, key=lambda row: expected_matrix().index({
        "candidate_id": row["candidate_id"],
        "config_id": row["config_id"],
        "dock": row["dock"],
        "rerank": row["rerank"],
        "diffdock": row["diffdock"],
    }))


def evaluate_results(prereg_json: Path, results_csv: Path) -> dict[str, Any]:
    prereg = _read_prereg(prereg_json)
    rows = _read_results_csv(results_csv)
    target_hash = prereg["artifacts"]["target_universe"]["sha256"]
    benchmark_hash = prereg["artifacts"]["benchmark"]["sha256"]
    training_data_hash = prereg["artifacts"]["training_data"]["sha256"]
    seeds_hash = canonical_sha256(prereg["seeds"])
    if any(row["target_universe_sha256"] != target_hash for row in rows):
        raise CandidateMatrixError("target_universe_sha256 provenance does not match preregistration")
    if any(row["benchmark_sha256"] != benchmark_hash for row in rows):
        raise CandidateMatrixError("benchmark_sha256 provenance does not match preregistration")
    if any(row["data_sha256"] != training_data_hash for row in rows):
        raise CandidateMatrixError("data_sha256 provenance does not match preregistration")
    if any(row["seeds_sha256"] != seeds_hash for row in rows):
        raise CandidateMatrixError("seeds_sha256 provenance does not match preregistration")
    for row in rows:
        expected_model_hash = prereg["comparators"][row["candidate_id"]]["model_sha256"]
        if row["model_sha256"] != expected_model_hash:
            raise CandidateMatrixError(
                f"model_sha256 provenance does not match preregistration for {row['candidate_id']}"
            )
    e0_rows = [row for row in rows if row["candidate_id"] == "E0"]
    if len(e0_rows) != 1:
        raise CandidateMatrixError("results CSV must contain exactly one E0 baseline row")
    baseline = e0_rows[0]
    evaluated: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        enriched["deltas_vs_E0"] = {
            "top30": row["top30"] - baseline["top30"],
            "top10": row["top10"] - baseline["top10"],
            "mrr": row["mrr"] - baseline["mrr"],
            "brier": row["brier"] - baseline["brier"],
            "log_loss": row["log_loss"] - baseline["log_loss"],
            "cold_p95_minutes": row["cold_p95_minutes"] - baseline["cold_p95_minutes"],
            "bundle_gb": row["bundle_gb"] - baseline["bundle_gb"],
        }
        reasons = []
        if row["candidate_id"] == "E0":
            reasons.append("baseline_not_selectable")
        if (
            enriched["deltas_vs_E0"]["top30"]
            < -GUARDRAILS["top30_regression_max"] - FLOAT_TOLERANCE
        ):
            reasons.append("top30_regression_guardrail")
        if (
            enriched["deltas_vs_E0"]["brier"]
            > GUARDRAILS["brier_regression_max"] + FLOAT_TOLERANCE
        ):
            reasons.append("brier_regression_guardrail")
        if (
            enriched["deltas_vs_E0"]["log_loss"]
            > GUARDRAILS["log_loss_regression_max"] + FLOAT_TOLERANCE
        ):
            reasons.append("log_loss_regression_guardrail")
        enriched["reasons"] = reasons
        enriched["eligible"] = not reasons
        evaluated.append(enriched)
        if enriched["eligible"]:
            eligible.append(enriched)
        else:
            rejected.append(enriched)
    selected = None
    status = "retain_E0"
    objective_key = lambda row: (
        -row["mrr"],
        row["cold_p95_minutes"],
        row["bundle_gb"],
        row["candidate_id"],
        row["config_id"],
    )
    if eligible:
        best_candidate = sorted(eligible, key=objective_key)[0]
        if objective_key(best_candidate) < objective_key(baseline):
            selected = best_candidate
            status = "select_candidate"
    manifest: dict[str, Any] = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "status": status,
        "selected": selected,
        "evaluated": evaluated,
        "eligible": eligible,
        "rejected": rejected,
        "input_hashes": {
            "prereg_json": sha256_file(prereg_json),
            "results_csv": sha256_file(results_csv),
            "target_universe": target_hash,
            "benchmark": benchmark_hash,
            "training_data": training_data_hash,
            "seeds": seeds_hash,
            "preregistration_sha256": prereg["preregistration_sha256"],
        },
    }
    manifest["binding_sha256"] = canonical_sha256(manifest)
    return manifest


def validate_selection_manifest(
    payload: dict[str, Any],
    *,
    prereg_json: Path | None = None,
    results_csv: Path | None = None,
) -> dict[str, Any]:
    """Public validator for a selection manifest, optionally against live inputs."""
    if payload.get("schema_version") != SELECTION_SCHEMA_VERSION:
        raise CandidateMatrixError(f"schema_version must be {SELECTION_SCHEMA_VERSION}")
    binding = payload.get("binding_sha256")
    if not _is_sha256(binding):
        raise CandidateMatrixError("binding_sha256 must be a SHA-256 digest")
    unsigned = {k: v for k, v in payload.items() if k != "binding_sha256"}
    if canonical_sha256(unsigned) != binding:
        raise CandidateMatrixError("binding_sha256 mismatch or selection manifest tamper")
    if prereg_json is not None and results_csv is not None:
        fresh = evaluate_results(prereg_json, results_csv)
        if fresh != payload:
            raise CandidateMatrixError("selection manifest does not match fresh evaluation")
    return payload


def _run_create(args: argparse.Namespace) -> int:
    out = Path(args.out_json)
    try:
        payload = seal_preregistration_contract(_read_json_object(Path(args.contract_json), "contract"))
        _write_json_atomic(out, payload)
        return 0
    except Exception:
        out.unlink(missing_ok=True)
        raise


def _run_evaluate(args: argparse.Namespace) -> int:
    out = Path(args.out_json)
    try:
        payload = evaluate_results(Path(args.prereg_json), Path(args.results_csv))
        _write_json_atomic(out, payload)
        validate_selection_manifest(payload, prereg_json=Path(args.prereg_json), results_csv=Path(args.results_csv))
        return 0
    except Exception:
        out.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create", help="validate and seal a preregistration contract")
    create.add_argument("--contract-json", required=True)
    create.add_argument("--out-json", required=True)
    create.set_defaults(func=_run_create)
    evaluate = sub.add_parser("evaluate", help="evaluate a sealed preregistration against result metrics")
    evaluate.add_argument("--prereg-json", required=True)
    evaluate.add_argument("--results-csv", required=True)
    evaluate.add_argument("--out-json", required=True)
    evaluate.set_defaults(func=_run_evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except CandidateMatrixError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    raise SystemExit(main())
