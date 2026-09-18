#!/usr/bin/env python3
"""Seal and evaluate the additive SkinScout performance-v2 candidate contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "skinscout.performance-v2-preregistration.v2"
SELECTION_SCHEMA_VERSION = "skinscout.performance-v2-selection.v1"
SEEDS = [17, 42, 73]
TRAINING_WINDOW = "supervised_evidence_through_2023-12-31"
SELECTION_WINDOW = "2024_dev_only"
REQUIRED_CANDIDATES = [
    {"candidate_id": "P0", "definition": "frozen chembl_p5_max E0 fallback"},
    {
        "candidate_id": "P1",
        "definition": "frozen MoLFormer-XL plus ECFP and ESM2-150M pair retriever",
    },
    {
        "candidate_id": "P2",
        "definition": "P1 plus confidence-gated biochemical reranker top512",
    },
    {
        "candidate_id": "P3",
        "definition": "P2 plus trusted-pocket structure expert top64 Boltz top3",
    },
]
LABEL_POLICY = {
    "positive": "explicit measured pActivity >= 6.0 only",
    "negative": "explicit measured pActivity <= 5.0 only",
    "gray": "measured 5.0 < pActivity < 6.0 or conflicting evidence; excluded from BCE and calibration",
    "unmeasured": "PU-unlabeled; ranking-only and never a probability negative",
    "positive_threshold": 6.0,
    "negative_threshold": 5.0,
    "calibration_endpoint_families": ["direct_binding", "functional"],
}
BUDGET = {
    "peak_vram_gib_max": 11.5,
    "trainable_parameters_max": 10_000_000,
    "rerank_top_k": 512,
    "structure_top_k": 64,
    "boltz_top_k": 3,
    "boltz_max_residues": 700,
}
DEV_GATES = {
    "strict_top10_improvement": True,
    "strict_top30_improvement": True,
    "strict_mrr_improvement": True,
    "strict_brier_improvement": True,
    "strict_log_loss_improvement": True,
    "dual_cold_target_macro_top30_min": 0.05,
    "dual_cold_truth_hits_min": 5,
    "dual_cold_unseen_target_clusters_min": 3,
    "effective_target_count_min": 5.0,
}
REQUIRED_ARTIFACTS = {
    "training_data",
    "target_universe",
    "evaluation_panel_manifest",
    "dev_ranking_queries",
    "dev_calibration_pairs",
    "dual_cold_ranking_queries",
}
REQUIRED_RESULT_COLUMNS = (
    "candidate_id",
    "run_id",
    "model_sha256",
    "training_data_sha256",
    "target_universe_sha256",
    "dev_benchmark_sha256",
    "seeds_sha256",
    "top10",
    "top30",
    "mrr",
    "brier",
    "log_loss",
    "dual_cold_target_macro_top30",
    "dual_cold_truth_hits",
    "dual_cold_unseen_target_clusters",
    "effective_target_count",
    "peak_vram_gib",
    "trainable_parameters",
    "cold_p95_minutes",
)


class PerformanceV2ContractError(ValueError):
    """Raised when a v2 preregistration or result fails closed."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(char in "0123456789abcdef" for char in value)
        and len(set(value)) > 1
    )


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
        raise PerformanceV2ContractError(f"{label} is missing, empty, or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PerformanceV2ContractError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise PerformanceV2ContractError(f"{label} must be a JSON object")
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _artifact(value: object, field: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise PerformanceV2ContractError(f"{field} must contain exactly path and sha256")
    path_value = value.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise PerformanceV2ContractError(f"{field}.path must be non-empty")
    path = Path(path_value)
    if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
        raise PerformanceV2ContractError(f"{field}.path is missing, empty, or unsafe: {path}")
    declared = value.get("sha256")
    if not _is_sha256(declared):
        raise PerformanceV2ContractError(f"{field}.sha256 must be a SHA-256 digest")
    active = sha256_file(path)
    if active != declared:
        raise PerformanceV2ContractError(f"{field}.sha256 drift: expected {declared}, active {active}")
    return {"path": path_value, "sha256": declared}


def _models(value: object) -> dict[str, dict[str, str]]:
    if not isinstance(value, dict) or set(value) != {row["candidate_id"] for row in REQUIRED_CANDIDATES}:
        raise PerformanceV2ContractError("models must contain exactly P0, P1, P2, and P3")
    output: dict[str, dict[str, str]] = {}
    required = {
        "model_id",
        "version",
        "license",
        "license_profile",
        "commercial_fallback_id",
        "model_sha256",
    }
    for candidate_id in ("P0", "P1", "P2", "P3"):
        row = value[candidate_id]
        if not isinstance(row, dict) or set(row) != required:
            raise PerformanceV2ContractError(
                f"models.{candidate_id} must contain exactly {sorted(required)}"
            )
        normalized: dict[str, str] = {}
        for key in required - {"model_sha256"}:
            item = row.get(key)
            if not isinstance(item, str) or not item.strip():
                raise PerformanceV2ContractError(f"models.{candidate_id}.{key} must be non-empty")
            normalized[key] = item.strip()
        digest = row.get("model_sha256")
        if not _is_sha256(digest):
            raise PerformanceV2ContractError(
                f"models.{candidate_id}.model_sha256 must be a SHA-256 digest"
            )
        normalized["model_sha256"] = digest
        output[candidate_id] = normalized
    return output


def validate_contract_payload(payload: dict[str, Any]) -> dict[str, Any]:
    required_keys = {
        "schema_version",
        "candidates",
        "training_window",
        "selection_window",
        "artifacts",
        "models",
        "seeds",
        "label_policy",
        "budget",
        "dev_gates",
    }
    if set(payload) != required_keys:
        raise PerformanceV2ContractError(
            f"contract keys mismatch missing={sorted(required_keys - set(payload))} "
            f"extra={sorted(set(payload) - required_keys)}"
        )
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise PerformanceV2ContractError(f"schema_version must be {SCHEMA_VERSION}")
    if payload.get("candidates") != REQUIRED_CANDIDATES:
        raise PerformanceV2ContractError("candidate definitions changed")
    if payload.get("training_window") != TRAINING_WINDOW:
        raise PerformanceV2ContractError("training_window changed")
    if payload.get("selection_window") != SELECTION_WINDOW:
        raise PerformanceV2ContractError("selection_window changed")
    if payload.get("seeds") != SEEDS:
        raise PerformanceV2ContractError(f"seeds must be exactly {SEEDS}")
    if payload.get("label_policy") != LABEL_POLICY:
        raise PerformanceV2ContractError("label_policy changed")
    if payload.get("budget") != BUDGET:
        raise PerformanceV2ContractError("budget changed")
    if payload.get("dev_gates") != DEV_GATES:
        raise PerformanceV2ContractError("dev_gates changed")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != REQUIRED_ARTIFACTS:
        raise PerformanceV2ContractError(
            f"artifacts must contain exactly {sorted(REQUIRED_ARTIFACTS)}"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "candidates": REQUIRED_CANDIDATES,
        "training_window": TRAINING_WINDOW,
        "selection_window": SELECTION_WINDOW,
        "artifacts": {
            key: _artifact(artifacts[key], f"artifacts.{key}")
            for key in sorted(REQUIRED_ARTIFACTS)
        },
        "models": _models(payload.get("models")),
        "seeds": SEEDS,
        "label_policy": LABEL_POLICY,
        "budget": BUDGET,
        "dev_gates": DEV_GATES,
    }


def seal_contract(payload: dict[str, Any]) -> dict[str, Any]:
    sealed = validate_contract_payload(payload)
    sealed["sealed"] = True
    sealed["validator_sha256"] = sha256_file(Path(__file__).resolve())
    sealed["preregistration_sha256"] = canonical_sha256(sealed)
    return sealed


def validate_sealed_contract(payload: dict[str, Any]) -> dict[str, Any]:
    unsealed_keys = {
        "schema_version",
        "candidates",
        "training_window",
        "selection_window",
        "artifacts",
        "models",
        "seeds",
        "label_policy",
        "budget",
        "dev_gates",
    }
    if set(payload) != unsealed_keys | {"sealed", "validator_sha256", "preregistration_sha256"}:
        raise PerformanceV2ContractError("sealed contract keys changed")
    if payload.get("sealed") is not True:
        raise PerformanceV2ContractError("sealed must be true")
    if payload.get("validator_sha256") != sha256_file(Path(__file__).resolve()):
        raise PerformanceV2ContractError("validator_sha256 does not match active code")
    expected = seal_contract({key: payload[key] for key in unsealed_keys})
    if payload != expected:
        raise PerformanceV2ContractError("preregistration hash mismatch or contract tamper")
    return expected


def _finite(row: dict[str, str], field: str, *, maximum: float | None = None) -> float:
    try:
        value = float(row.get(field, ""))
    except (TypeError, ValueError) as exc:
        raise PerformanceV2ContractError(f"{field} must be numeric") from exc
    if not math.isfinite(value) or value < 0 or (maximum is not None and value > maximum):
        raise PerformanceV2ContractError(f"{field} is outside its valid range")
    return value


def _integer(row: dict[str, str], field: str) -> int:
    try:
        value = int(row.get(field, ""))
    except (TypeError, ValueError) as exc:
        raise PerformanceV2ContractError(f"{field} must be an integer") from exc
    if value < 0:
        raise PerformanceV2ContractError(f"{field} must be non-negative")
    return value


def _result_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
        raise PerformanceV2ContractError(f"results CSV is missing, empty, or unsafe: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(REQUIRED_RESULT_COLUMNS):
            raise PerformanceV2ContractError("results CSV columns do not match the v2 contract")
        raw_rows = list(reader)
    if len(raw_rows) != len(REQUIRED_CANDIDATES):
        raise PerformanceV2ContractError("results CSV must contain exactly one row per candidate")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_rows:
        candidate_id = str(raw.get("candidate_id", "")).strip()
        if candidate_id not in {row["candidate_id"] for row in REQUIRED_CANDIDATES} or candidate_id in seen:
            raise PerformanceV2ContractError(f"invalid or duplicate candidate_id: {candidate_id}")
        seen.add(candidate_id)
        parsed: dict[str, Any] = {
            "candidate_id": candidate_id,
            "run_id": str(raw.get("run_id", "")).strip(),
        }
        if not parsed["run_id"]:
            raise PerformanceV2ContractError("run_id must be non-empty")
        for field in (
            "model_sha256",
            "training_data_sha256",
            "target_universe_sha256",
            "dev_benchmark_sha256",
            "seeds_sha256",
        ):
            digest = raw.get(field)
            if not _is_sha256(digest):
                raise PerformanceV2ContractError(f"{field} must be a SHA-256 digest")
            parsed[field] = digest
        for field in (
            "top10",
            "top30",
            "mrr",
            "brier",
            "dual_cold_target_macro_top30",
        ):
            parsed[field] = _finite(raw, field, maximum=1.0)
        for field in ("log_loss", "effective_target_count", "peak_vram_gib", "cold_p95_minutes"):
            parsed[field] = _finite(raw, field)
        for field in (
            "dual_cold_truth_hits",
            "dual_cold_unseen_target_clusters",
            "trainable_parameters",
        ):
            parsed[field] = _integer(raw, field)
        rows.append(parsed)
    return sorted(rows, key=lambda row: row["candidate_id"])


def evaluate_results(preregistration_path: Path, results_path: Path) -> dict[str, Any]:
    contract = validate_sealed_contract(_read_json(preregistration_path, "preregistration"))
    rows = _result_rows(results_path)
    expected_hashes = {
        "training_data_sha256": contract["artifacts"]["training_data"]["sha256"],
        "target_universe_sha256": contract["artifacts"]["target_universe"]["sha256"],
        "dev_benchmark_sha256": contract["artifacts"]["dev_ranking_queries"]["sha256"],
        "seeds_sha256": canonical_sha256(SEEDS),
    }
    for row in rows:
        if row["model_sha256"] != contract["models"][row["candidate_id"]]["model_sha256"]:
            raise PerformanceV2ContractError(f"model_sha256 mismatch for {row['candidate_id']}")
        for field, expected in expected_hashes.items():
            if row[field] != expected:
                raise PerformanceV2ContractError(f"{field} mismatch for {row['candidate_id']}")
    baseline = next(row for row in rows if row["candidate_id"] == "P0")
    if baseline["peak_vram_gib"] > BUDGET["peak_vram_gib_max"]:
        raise PerformanceV2ContractError("P0 fallback exceeds the peak VRAM budget")
    if baseline["trainable_parameters"] > BUDGET["trainable_parameters_max"]:
        raise PerformanceV2ContractError("P0 fallback exceeds the trainable parameter budget")
    run_ids = [row["run_id"] for row in rows]
    if len(set(run_ids)) != len(run_ids):
        raise PerformanceV2ContractError("candidate run_id values must be unique")
    evaluated: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for row in rows:
        reasons: list[str] = []
        if row["candidate_id"] != "P0":
            for metric in ("top10", "top30", "mrr"):
                if row[metric] <= baseline[metric]:
                    reasons.append(f"strict_{metric}_improvement")
            for metric in ("brier", "log_loss"):
                if row[metric] >= baseline[metric]:
                    reasons.append(f"strict_{metric}_improvement")
            if row["dual_cold_target_macro_top30"] < DEV_GATES["dual_cold_target_macro_top30_min"]:
                reasons.append("dual_cold_target_macro_top30")
            if row["dual_cold_truth_hits"] < DEV_GATES["dual_cold_truth_hits_min"]:
                reasons.append("dual_cold_truth_hits")
            if row["dual_cold_unseen_target_clusters"] < DEV_GATES["dual_cold_unseen_target_clusters_min"]:
                reasons.append("dual_cold_unseen_target_clusters")
            if row["effective_target_count"] < DEV_GATES["effective_target_count_min"]:
                reasons.append("effective_target_count")
        if row["peak_vram_gib"] > BUDGET["peak_vram_gib_max"]:
            reasons.append("peak_vram_budget")
        if row["trainable_parameters"] > BUDGET["trainable_parameters_max"]:
            reasons.append("trainable_parameter_budget")
        enriched = {**row, "eligible": row["candidate_id"] != "P0" and not reasons, "reasons": reasons}
        evaluated.append(enriched)
        if enriched["eligible"]:
            eligible.append(enriched)
    selected = None
    status = "retain_P0"
    if eligible:
        selected = sorted(
            eligible,
            key=lambda row: (
                -row["dual_cold_target_macro_top30"],
                -row["top30"],
                -row["mrr"],
                row["cold_p95_minutes"],
                row["candidate_id"],
            ),
        )[0]
        status = "select_candidate"
    output: dict[str, Any] = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "status": status,
        "selected": selected,
        "evaluated": evaluated,
        "input_hashes": {
            "preregistration": sha256_file(preregistration_path),
            "results": sha256_file(results_path),
            "preregistration_sha256": contract["preregistration_sha256"],
        },
    }
    output["binding_sha256"] = canonical_sha256(output)
    return output


def validate_selection(
    payload: dict[str, Any],
    *,
    preregistration_path: Path | None = None,
    results_path: Path | None = None,
) -> dict[str, Any]:
    if payload.get("schema_version") != SELECTION_SCHEMA_VERSION:
        raise PerformanceV2ContractError("selection schema_version changed")
    binding = payload.get("binding_sha256")
    if not _is_sha256(binding):
        raise PerformanceV2ContractError("selection binding_sha256 is invalid")
    unsigned = {key: value for key, value in payload.items() if key != "binding_sha256"}
    if canonical_sha256(unsigned) != binding:
        raise PerformanceV2ContractError("selection binding_sha256 mismatch")
    if (preregistration_path is None) != (results_path is None):
        raise PerformanceV2ContractError(
            "selection validation requires both preregistration_path and results_path"
        )
    if preregistration_path is not None and results_path is not None:
        inputs = payload.get("input_hashes")
        if not isinstance(inputs, dict):
            raise PerformanceV2ContractError("selection input_hashes is missing")
        if inputs.get("preregistration") != sha256_file(preregistration_path):
            raise PerformanceV2ContractError("selection preregistration input is stale")
        if inputs.get("results") != sha256_file(results_path):
            raise PerformanceV2ContractError("selection results input is stale")
        expected = evaluate_results(preregistration_path, results_path)
        if payload != expected:
            raise PerformanceV2ContractError("selection no longer matches active inputs")
    return payload


def _create(args: argparse.Namespace) -> int:
    output = Path(args.out_json)
    try:
        _write_json_atomic(output, seal_contract(_read_json(Path(args.contract_json), "contract")))
        return 0
    except Exception:
        output.unlink(missing_ok=True)
        raise


def _evaluate(args: argparse.Namespace) -> int:
    output = Path(args.out_json)
    try:
        payload = evaluate_results(Path(args.prereg_json), Path(args.results_csv))
        validate_selection(
            payload,
            preregistration_path=Path(args.prereg_json),
            results_path=Path(args.results_csv),
        )
        _write_json_atomic(output, payload)
        return 0
    except Exception:
        output.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--contract-json", required=True)
    create.add_argument("--out-json", required=True)
    create.set_defaults(func=_create)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--prereg-json", required=True)
    evaluate.add_argument("--results-csv", required=True)
    evaluate.add_argument("--out-json", required=True)
    evaluate.set_defaults(func=_evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except PerformanceV2ContractError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    raise SystemExit(main())
