#!/usr/bin/env python3
"""Evaluate compact, preregistered SkinScout performance-v2 evidence."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

try:
    from performance_v2_model import ContractError, append_only_outputs
except ModuleNotFoundError:  # pragma: no cover - direct script execution.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from performance_v2_model import ContractError, append_only_outputs

try:
    from preregistered_performance_v2 import (
        LABEL_POLICY,
        REQUIRED_CANDIDATES,
        REQUIRED_RESULT_COLUMNS,
        SEEDS,
        PerformanceV2ContractError,
        canonical_sha256,
        sha256_file,
        validate_sealed_contract,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from preregistered_performance_v2 import (
        LABEL_POLICY,
        REQUIRED_CANDIDATES,
        REQUIRED_RESULT_COLUMNS,
        SEEDS,
        PerformanceV2ContractError,
        canonical_sha256,
        sha256_file,
        validate_sealed_contract,
    )


EVIDENCE_SCHEMA_VERSION = "skinscout.performance-v2-evidence.v1"
DENOMINATOR_SCHEMA_VERSION = "skinscout.performance-v2-denominators.v1"
RECOVERY_PANEL_SCHEMA_VERSION = "skinscout.activity-recovery-panels.v5"
TIE_POLICY = "average"
CANDIDATES = tuple(row["candidate_id"] for row in REQUIRED_CANDIDATES)
CALIBRATION_ENDPOINT_FAMILIES = tuple(LABEL_POLICY["calibration_endpoint_families"])
DUAL_COLD_FLAGS = (
    "absent_pair_from_prior_splits",
    "absent_pair_from_train",
    "absent_publication_from_prior_splits",
    "absent_publication_from_train",
    "absent_scaffold_from_prior_splits",
    "absent_scaffold_from_train",
    "absent_target_cluster_30_from_prior_splits",
    "absent_target_cluster_30_from_train",
    "absent_target_cluster_50_from_prior_splits",
    "absent_target_cluster_50_from_train",
    "claimable",
)
RANK_EVIDENCE_COLUMNS = (
    "candidate_id",
    "query_id",
    "target_id",
    "rank",
    "target_universe_size",
    "tie_policy",
)
CALIBRATION_EVIDENCE_COLUMNS = (
    "candidate_id",
    "pair_id",
    "query_id",
    "target_id",
    "endpoint_family",
    "calibrated_measured_event_probability",
    "calibrator_sha256",
)
RESULT_COLUMNS = list(REQUIRED_RESULT_COLUMNS)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
        raise PerformanceV2ContractError(f"{label} is missing, empty, or unsafe: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PerformanceV2ContractError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise PerformanceV2ContractError(f"{label} must be a JSON object")
    return payload


def _read_table(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
        raise PerformanceV2ContractError(f"{label} is missing, empty, or unsafe: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif suffix in {".csv", ".tsv"}:
        frame = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
            sep="\t" if suffix == ".tsv" else ",",
        )
    else:
        raise PerformanceV2ContractError(f"{label} must be CSV, TSV, or parquet: {path}")
    if frame.empty:
        raise PerformanceV2ContractError(f"{label} is empty: {path}")
    if frame.columns.duplicated().any():
        raise PerformanceV2ContractError(f"{label} contains duplicate columns")
    return frame


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _finite_float(
    value: object,
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    try:
        parsed = float(_as_text(value))
    except ValueError as exc:
        raise PerformanceV2ContractError(f"{field} must be numeric") from exc
    if not math.isfinite(parsed):
        raise PerformanceV2ContractError(f"{field} must be finite")
    if minimum is not None and parsed < minimum:
        raise PerformanceV2ContractError(f"{field} is below its valid range")
    if maximum is not None and parsed > maximum:
        raise PerformanceV2ContractError(f"{field} is above its valid range")
    return parsed


def _nonnegative_integer(value: object, field: str) -> int:
    try:
        parsed = int(_as_text(value))
    except ValueError as exc:
        raise PerformanceV2ContractError(f"{field} must be an integer") from exc
    if parsed < 0:
        raise PerformanceV2ContractError(f"{field} must be non-negative")
    return parsed


def _is_sha256(value: object) -> bool:
    text = _as_text(value)
    return (
        len(text) == 64
        and text == text.lower()
        and all(character in "0123456789abcdef" for character in text)
    )


def _bool(value: object, field: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = _as_text(value).lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise PerformanceV2ContractError(f"{field} must be boolean")


def _measurement_label(pactivity: object) -> str:
    value = _finite_float(pactivity, "pactivity")
    if value >= float(LABEL_POLICY["positive_threshold"]):
        return "positive"
    if value <= float(LABEL_POLICY["negative_threshold"]):
        return "negative"
    return "gray"


def _string_list(value: object, field: str) -> list[str]:
    if isinstance(value, (list, tuple)):
        raw = list(value)
    elif hasattr(value, "tolist") and not isinstance(value, str):
        raw = value.tolist()
    else:
        text = _as_text(value)
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            try:
                raw = ast.literal_eval(text)
            except (ValueError, SyntaxError) as exc:
                raise PerformanceV2ContractError(f"{field} must be a list") from exc
    if not isinstance(raw, list) or not raw:
        raise PerformanceV2ContractError(f"{field} must be a nonempty list")
    values = [_as_text(item) for item in raw]
    if any(not item for item in values) or len(values) != len(set(values)):
        raise PerformanceV2ContractError(f"{field} contains blank or duplicate values")
    return sorted(values)


def _target_universe(path: Path) -> dict[str, str]:
    frame = _read_table(path, "target universe")
    if "target_id" not in frame.columns:
        if "uniprot" not in frame.columns:
            raise PerformanceV2ContractError(
                "target universe must contain target_id or uniprot"
            )
        frame = frame.rename(columns={"uniprot": "target_id"})
    required = {"target_id", "target_cluster_30"}
    missing = required - set(frame.columns)
    if missing:
        raise PerformanceV2ContractError(
            f"target universe missing required columns: {sorted(missing)}"
        )
    output: dict[str, str] = {}
    for row in frame[["target_id", "target_cluster_30"]].itertuples(index=False):
        target_id = _as_text(row.target_id)
        cluster = _as_text(row.target_cluster_30)
        if not target_id or not cluster:
            raise PerformanceV2ContractError("target universe contains blank identifiers")
        if target_id in output:
            raise PerformanceV2ContractError(f"target universe contains duplicate target: {target_id}")
        output[target_id] = cluster
    return output


def _ranking_panel(
    path: Path,
    label: str,
    *,
    dual_cold: bool,
    target_ids: set[str],
) -> dict[str, tuple[str, ...]]:
    frame = _read_table(path, label)
    required = {"query_id", "truth_targets", "n_truth_targets", "split"}
    if dual_cold:
        required.update(DUAL_COLD_FLAGS)
    missing = required - set(frame.columns)
    if missing:
        raise PerformanceV2ContractError(f"{label} missing required columns: {sorted(missing)}")
    if frame["query_id"].map(_as_text).duplicated().any():
        raise PerformanceV2ContractError(f"{label} contains duplicate query_id values")
    expected_split = "test" if dual_cold else "dev"
    output: dict[str, tuple[str, ...]] = {}
    for index, row in frame.iterrows():
        query_id = _as_text(row["query_id"])
        if not query_id:
            raise PerformanceV2ContractError(f"{label}.query_id contains blanks")
        if _as_text(row["split"]) != expected_split:
            raise PerformanceV2ContractError(
                f"{label}.split must be exactly {expected_split!r}"
            )
        if dual_cold:
            for flag in DUAL_COLD_FLAGS:
                if not _bool(row[flag], f"{label}.{flag}"):
                    raise PerformanceV2ContractError(
                        f"{label} is not claimable dual-cold evidence: {flag}=false at row {index}"
                    )
        truths = _string_list(row["truth_targets"], f"{label}.truth_targets")
        declared = _nonnegative_integer(row["n_truth_targets"], f"{label}.n_truth_targets")
        if declared != len(truths):
            raise PerformanceV2ContractError(f"{label}.n_truth_targets mismatch for {query_id}")
        missing_targets = sorted(set(truths) - target_ids)
        if missing_targets:
            raise PerformanceV2ContractError(
                f"{label} truth targets are absent from target universe: {missing_targets[:5]}"
            )
        output[query_id] = tuple(truths)
    return output


def _calibration_panel(path: Path, target_ids: set[str]) -> dict[str, dict[str, Any]]:
    frame = _read_table(path, "dev calibration panel")
    if "target_id" not in frame.columns:
        if "uniprot" not in frame.columns:
            raise PerformanceV2ContractError(
                "dev calibration panel must contain target_id or uniprot"
            )
        frame = frame.rename(columns={"uniprot": "target_id"})
    required = {
        "pair_id",
        "query_id",
        "target_id",
        "endpoint_family",
        "label",
        "sample_weight",
        "split",
        "pactivity_min",
        "pactivity_max",
    }
    missing = required - set(frame.columns)
    if missing:
        raise PerformanceV2ContractError(
            f"dev calibration panel missing required columns: {sorted(missing)}"
        )
    if frame["pair_id"].map(_as_text).duplicated().any():
        raise PerformanceV2ContractError("dev calibration panel contains duplicate pair_id values")
    output: dict[str, dict[str, Any]] = {}
    for _, row in frame.iterrows():
        pair_id = _as_text(row["pair_id"])
        query_id = _as_text(row["query_id"])
        target_id = _as_text(row["target_id"])
        endpoint = _as_text(row["endpoint_family"])
        if not pair_id or not query_id or not target_id:
            raise PerformanceV2ContractError("dev calibration panel contains blank identifiers")
        if target_id not in target_ids:
            raise PerformanceV2ContractError(
                f"dev calibration target absent from target universe: {target_id}"
            )
        if endpoint not in CALIBRATION_ENDPOINT_FAMILIES:
            raise PerformanceV2ContractError(
                f"unsupported calibration endpoint_family: {endpoint}"
            )
        if _as_text(row["split"]) != "dev":
            raise PerformanceV2ContractError("dev calibration panel split must be 'dev'")
        label = _nonnegative_integer(row["label"], "dev calibration panel.label")
        if label not in {0, 1}:
            raise PerformanceV2ContractError("dev calibration panel.label must be 0 or 1")
        minimum = _finite_float(row["pactivity_min"], "pactivity_min")
        maximum = _finite_float(row["pactivity_max"], "pactivity_max")
        if minimum > maximum:
            raise PerformanceV2ContractError("calibration pactivity_min exceeds pactivity_max")
        if label == 1 and minimum < float(LABEL_POLICY["positive_threshold"]):
            raise PerformanceV2ContractError("positive calibration row violates pActivity threshold")
        if label == 0 and maximum > float(LABEL_POLICY["negative_threshold"]):
            raise PerformanceV2ContractError("negative calibration row violates pActivity threshold")
        output[pair_id] = {
            "query_id": query_id,
            "target_id": target_id,
            "endpoint_family": endpoint,
            "label": label,
            "sample_weight": _finite_float(
                row["sample_weight"], "sample_weight", minimum=1e-15
            ),
        }
    return output


def _panel_manifest(path: Path, contract: dict[str, Any]) -> dict[str, Any]:
    payload = _read_json(path, "evaluation panel manifest")
    if payload.get("schema_version") != RECOVERY_PANEL_SCHEMA_VERSION:
        raise PerformanceV2ContractError(
            f"evaluation panel manifest must use {RECOVERY_PANEL_SCHEMA_VERSION}"
        )
    algorithm = payload.get("algorithm")
    if not isinstance(algorithm, dict):
        raise PerformanceV2ContractError("evaluation panel manifest algorithm is missing")
    if algorithm.get("positive_threshold") != LABEL_POLICY["positive_threshold"]:
        raise PerformanceV2ContractError("positive label threshold differs from preregistration")
    if algorithm.get("negative_threshold") != LABEL_POLICY["negative_threshold"]:
        raise PerformanceV2ContractError("negative label threshold differs from preregistration")
    endpoint_families = set((algorithm.get("endpoint_families") or {}).values())
    if endpoint_families != set(CALIBRATION_ENDPOINT_FAMILIES):
        raise PerformanceV2ContractError("calibration endpoint-family policy changed")
    if payload.get("passes_panel_adequacy_gate") is not True:
        raise PerformanceV2ContractError("evaluation panel adequacy gate did not pass")
    dual = (((payload.get("selection") or {}).get("ranking_queries") or {}).get("dual_cold") or {})
    if ((dual.get("adequacy") or {}).get("passes")) is not True:
        raise PerformanceV2ContractError("dual-cold panel adequacy gate did not pass")
    outputs = payload.get("outputs")
    if not isinstance(outputs, dict):
        raise PerformanceV2ContractError("evaluation panel manifest outputs are missing")
    bindings = {
        "dev_ranking_queries": "dev_ranking_queries.parquet",
        "dev_calibration_pairs": "dev_calibration_pairs.parquet",
        "dual_cold_ranking_queries": "dual_cold_ranking_queries.parquet",
    }
    for artifact_name, output_name in bindings.items():
        record = outputs.get(output_name)
        expected = contract["artifacts"][artifact_name]["sha256"]
        if not isinstance(record, dict) or record.get("sha256") != expected:
            raise PerformanceV2ContractError(
                f"evaluation panel manifest does not bind {artifact_name}"
            )
        _nonnegative_integer(record.get("rows"), f"outputs.{output_name}.rows")
    return payload


def _read_exact_csv(path: Path, columns: tuple[str, ...], label: str) -> Iterable[dict[str, str]]:
    if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
        raise PerformanceV2ContractError(f"{label} is missing, empty, or unsafe: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(columns):
            raise PerformanceV2ContractError(f"{label} columns do not match the v2 evidence contract")
        yield from reader


def _rank_evidence(
    path: Path,
    expected_pairs: set[tuple[str, str]],
    universe_count: int,
) -> dict[tuple[str, str, str], float]:
    output: dict[tuple[str, str, str], float] = {}
    observed_candidates: set[str] = set()
    for row in _read_exact_csv(path, RANK_EVIDENCE_COLUMNS, "rank evidence"):
        candidate = _as_text(row["candidate_id"])
        query_id = _as_text(row["query_id"])
        target_id = _as_text(row["target_id"])
        if candidate not in CANDIDATES:
            raise PerformanceV2ContractError(f"invalid rank evidence candidate_id: {candidate}")
        if (query_id, target_id) not in expected_pairs:
            raise PerformanceV2ContractError("rank evidence contains a non-panel query-target pair")
        if _nonnegative_integer(row["target_universe_size"], "target_universe_size") != universe_count:
            raise PerformanceV2ContractError("rank evidence target_universe_size mismatch")
        if _as_text(row["tie_policy"]) != TIE_POLICY:
            raise PerformanceV2ContractError(f"rank evidence tie_policy must be {TIE_POLICY}")
        rank = _finite_float(row["rank"], "rank", minimum=1.0, maximum=float(universe_count))
        key = (candidate, query_id, target_id)
        if key in output:
            raise PerformanceV2ContractError("rank evidence contains duplicate rows")
        output[key] = rank
        observed_candidates.add(candidate)
    if observed_candidates != set(CANDIDATES):
        raise PerformanceV2ContractError("rank evidence candidate coverage must be exactly P0-P3")
    expected_count = len(CANDIDATES) * len(expected_pairs)
    if len(output) != expected_count:
        raise PerformanceV2ContractError(
            f"rank evidence coverage mismatch expected={expected_count} observed={len(output)}"
        )
    return output


def _calibration_evidence(
    path: Path,
    calibration_panel: dict[str, dict[str, Any]],
    calibrators: dict[str, dict[str, str]],
) -> dict[tuple[str, str], float]:
    output: dict[tuple[str, str], float] = {}
    for row in _read_exact_csv(
        path, CALIBRATION_EVIDENCE_COLUMNS, "calibration evidence"
    ):
        candidate = _as_text(row["candidate_id"])
        pair_id = _as_text(row["pair_id"])
        if candidate not in CANDIDATES or pair_id not in calibration_panel:
            raise PerformanceV2ContractError("calibration evidence contains an invalid candidate or pair")
        truth = calibration_panel[pair_id]
        for field in ("query_id", "target_id", "endpoint_family"):
            if _as_text(row[field]) != truth[field]:
                raise PerformanceV2ContractError(
                    f"calibration evidence {field} does not match panel pair {pair_id}"
                )
        digest = _as_text(row["calibrator_sha256"])
        if digest != calibrators[candidate][truth["endpoint_family"]]:
            raise PerformanceV2ContractError(
                f"calibrator_sha256 mismatch for {candidate}/{truth['endpoint_family']}"
            )
        key = (candidate, pair_id)
        if key in output:
            raise PerformanceV2ContractError("calibration evidence contains duplicate rows")
        output[key] = _finite_float(
            row["calibrated_measured_event_probability"],
            "calibrated_measured_event_probability",
            minimum=0.0,
            maximum=1.0,
        )
    expected_count = len(CANDIDATES) * len(calibration_panel)
    if len(output) != expected_count:
        raise PerformanceV2ContractError(
            f"calibration evidence coverage mismatch expected={expected_count} observed={len(output)}"
        )
    return output


def _evidence_manifest(
    path: Path,
    *,
    rank_evidence_path: Path,
    calibration_evidence_path: Path,
    contract: dict[str, Any],
    universe_count: int,
) -> dict[str, dict[str, str]]:
    payload = _read_json(path, "performance-v2 evidence manifest")
    required = {
        "schema_version",
        "candidates",
        "artifacts",
        "bindings",
        "model_sha256",
        "target_universe_count",
        "tie_policy",
        "probability_contract",
        "calibrators",
        "generator_sha256",
        "binding_sha256",
    }
    if set(payload) != required:
        raise PerformanceV2ContractError("performance-v2 evidence manifest keys changed")
    binding = payload.get("binding_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "binding_sha256"}
    if not _is_sha256(binding) or canonical_sha256(unsigned) != binding:
        raise PerformanceV2ContractError("performance-v2 evidence manifest binding mismatch")
    if payload.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise PerformanceV2ContractError("performance-v2 evidence schema changed")
    if payload.get("candidates") != list(CANDIDATES):
        raise PerformanceV2ContractError("performance-v2 evidence candidate order changed")
    artifacts = payload.get("artifacts")
    evidence_paths = {
        "rank_evidence": rank_evidence_path,
        "calibration_evidence": calibration_evidence_path,
    }
    if not isinstance(artifacts, dict) or set(artifacts) != set(evidence_paths):
        raise PerformanceV2ContractError("performance-v2 evidence artifact set changed")
    for name, active_path in evidence_paths.items():
        record = artifacts.get(name)
        if not isinstance(record, dict) or set(record) != {"sha256", "rows"}:
            raise PerformanceV2ContractError(f"invalid evidence artifact record: {name}")
        if record.get("sha256") != sha256_file(active_path):
            raise PerformanceV2ContractError(f"evidence artifact hash mismatch: {name}")
        _nonnegative_integer(record.get("rows"), f"artifacts.{name}.rows")
    expected_bindings = {
        "target_universe_sha256": contract["artifacts"]["target_universe"]["sha256"],
        "dev_ranking_queries_sha256": contract["artifacts"]["dev_ranking_queries"]["sha256"],
        "dev_calibration_pairs_sha256": contract["artifacts"]["dev_calibration_pairs"]["sha256"],
        "dual_cold_ranking_queries_sha256": contract["artifacts"]["dual_cold_ranking_queries"]["sha256"],
        "evaluation_panel_manifest_sha256": contract["artifacts"]["evaluation_panel_manifest"]["sha256"],
    }
    if payload.get("bindings") != expected_bindings:
        raise PerformanceV2ContractError("performance-v2 evidence panel bindings changed")
    expected_models = {
        candidate: contract["models"][candidate]["model_sha256"] for candidate in CANDIDATES
    }
    if payload.get("model_sha256") != expected_models:
        raise PerformanceV2ContractError("performance-v2 evidence model bindings changed")
    if payload.get("target_universe_count") != universe_count:
        raise PerformanceV2ContractError("performance-v2 evidence target universe count changed")
    if payload.get("tie_policy") != TIE_POLICY:
        raise PerformanceV2ContractError("performance-v2 evidence tie policy changed")
    if payload.get("probability_contract") != {
        "calibrated_only": True,
        "ontology_specific": True,
        "endpoint_families": list(CALIBRATION_ENDPOINT_FAMILIES),
    }:
        raise PerformanceV2ContractError("performance-v2 probability contract changed")
    if not _is_sha256(payload.get("generator_sha256")):
        raise PerformanceV2ContractError("performance-v2 evidence generator hash is invalid")
    calibrators = payload.get("calibrators")
    if not isinstance(calibrators, dict) or set(calibrators) != set(CANDIDATES):
        raise PerformanceV2ContractError("performance-v2 calibrator coverage changed")
    normalized: dict[str, dict[str, str]] = {}
    for candidate in CANDIDATES:
        row = calibrators[candidate]
        if not isinstance(row, dict) or set(row) != set(CALIBRATION_ENDPOINT_FAMILIES):
            raise PerformanceV2ContractError(f"calibrator ontology coverage changed for {candidate}")
        if any(not _is_sha256(value) for value in row.values()):
            raise PerformanceV2ContractError(f"invalid calibrator hash for {candidate}")
        normalized[candidate] = dict(row)
    return normalized


def _resources(path: Path, contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if path.suffix.lower() == ".json":
        payload = _read_json(path, "resources")
        if isinstance(payload.get("candidates"), list):
            frame = pd.DataFrame(payload["candidates"])
        elif all(candidate in payload for candidate in CANDIDATES):
            frame = pd.DataFrame(
                [{**payload[candidate], "candidate_id": candidate} for candidate in CANDIDATES]
            )
        else:
            raise PerformanceV2ContractError("resources JSON candidate layout is invalid")
    else:
        frame = _read_table(path, "resources")
    frame = frame.rename(
        columns={
            "model_hash": "model_sha256",
            "peak_vram": "peak_vram_gib",
            "params": "trainable_parameters",
            "cold_latency": "cold_p95_minutes",
            "cold_latency_minutes": "cold_p95_minutes",
        }
    )
    required = {
        "candidate_id",
        "run_id",
        "model_sha256",
        "peak_vram_gib",
        "trainable_parameters",
        "cold_p95_minutes",
    }
    missing = required - set(frame.columns)
    if missing:
        raise PerformanceV2ContractError(f"resources missing columns: {sorted(missing)}")
    output: dict[str, dict[str, Any]] = {}
    for _, row in frame.iterrows():
        candidate = _as_text(row["candidate_id"])
        if candidate not in CANDIDATES or candidate in output:
            raise PerformanceV2ContractError(f"invalid or duplicate resource candidate: {candidate}")
        model_hash = _as_text(row["model_sha256"])
        if model_hash != contract["models"][candidate]["model_sha256"]:
            raise PerformanceV2ContractError(f"resources model_sha256 mismatch for {candidate}")
        run_id = _as_text(row["run_id"])
        if not run_id:
            raise PerformanceV2ContractError("resources.run_id must be nonempty")
        output[candidate] = {
            "run_id": run_id,
            "model_sha256": model_hash,
            "peak_vram_gib": _finite_float(row["peak_vram_gib"], "peak_vram_gib", minimum=0.0),
            "trainable_parameters": _nonnegative_integer(
                row["trainable_parameters"], "trainable_parameters"
            ),
            "cold_p95_minutes": _finite_float(
                row["cold_p95_minutes"], "cold_p95_minutes", minimum=0.0
            ),
        }
    if set(output) != set(CANDIDATES):
        raise PerformanceV2ContractError("resources candidate coverage must be exactly P0-P3")
    return output


def _ranking_metrics(
    ranks: dict[tuple[str, str, str], float],
    panel: dict[str, tuple[str, ...]],
) -> dict[str, dict[str, float]]:
    pairs = [(query, target) for query, targets in panel.items() for target in targets]
    if not pairs:
        raise PerformanceV2ContractError("ranking panel has no truth pairs")
    output: dict[str, dict[str, float]] = {}
    for candidate in CANDIDATES:
        values = [ranks[(candidate, query, target)] for query, target in pairs]
        output[candidate] = {
            "top10": sum(rank <= 10.0 for rank in values) / len(values),
            "top30": sum(rank <= 30.0 for rank in values) / len(values),
            # MRR: reciprocal rank of the BEST (lowest-rank) truth target per
            # query, averaged over queries.  The previous per-pair averaging
            # deflated MRR for multi-target queries and inflated it for
            # single-target queries ranked low.
            "mrr": _mrr_per_query(ranks, candidate, panel),
        }
    return output


def _mrr_per_query(
    ranks: dict[tuple[str, str, str], float],
    candidate: str,
    panel: dict[str, tuple[str, ...]],
) -> float:
    """Mean Reciprocal Rank computed correctly: best rank per query, averaged."""
    if not panel:
        return 0.0
    total = 0.0
    for query, targets in panel.items():
        best = min(ranks[(candidate, query, target)] for target in targets)
        total += 1.0 / best
    return total / len(panel)


def _calibration_metrics(
    probabilities: dict[tuple[str, str], float],
    panel: dict[str, dict[str, Any]],
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for candidate in CANDIDATES:
        total_weight = sum(row["sample_weight"] for row in panel.values())
        brier = 0.0
        log_loss = 0.0
        for pair_id, truth in panel.items():
            probability = probabilities[(candidate, pair_id)]
            label = float(truth["label"])
            weight = float(truth["sample_weight"])
            brier += weight * (probability - label) ** 2
            clipped = min(max(probability, 1e-15), 1.0 - 1e-15)
            log_loss += -weight * (
                label * math.log(clipped) + (1.0 - label) * math.log(1.0 - clipped)
            )
        output[candidate] = {
            "brier": brier / total_weight,
            "log_loss": log_loss / total_weight,
        }
    return output


def _dual_cold_metrics(
    ranks: dict[tuple[str, str, str], float],
    panel: dict[str, tuple[str, ...]],
    target_clusters: dict[str, str],
) -> tuple[dict[str, dict[str, float | int]], dict[str, float | int]]:
    pairs = [(query, target) for query, targets in panel.items() for target in targets]
    counts = Counter(target for _, target in pairs)
    effective_count = (sum(counts.values()) ** 2) / sum(value * value for value in counts.values())
    output: dict[str, dict[str, float | int]] = {}
    for candidate in CANDIDATES:
        hits_by_target: dict[str, list[bool]] = defaultdict(list)
        hit_pairs: list[tuple[str, str]] = []
        for query, target in pairs:
            hit = ranks[(candidate, query, target)] <= 30.0
            hits_by_target[target].append(hit)
            if hit:
                hit_pairs.append((query, target))
        output[candidate] = {
            "dual_cold_target_macro_top30": sum(
                sum(values) / len(values) for values in hits_by_target.values()
            )
            / len(hits_by_target),
            "dual_cold_truth_hits": len(hit_pairs),
            "dual_cold_unseen_target_clusters": len(
                {target_clusters[target] for _, target in hit_pairs}
            ),
            "effective_target_count": effective_count,
        }
    return output, {
        "dual_cold_truth_target_count": len(counts),
        "effective_target_count": effective_count,
    }


def _denominators(
    *,
    contract: dict[str, Any],
    panel_manifest: dict[str, Any],
    target_universe_count: int,
    dev_panel: dict[str, tuple[str, ...]],
    dual_panel: dict[str, tuple[str, ...]],
    calibration_panel: dict[str, dict[str, Any]],
    dual_denominators: dict[str, float | int],
    evidence_manifest_path: Path,
) -> dict[str, Any]:
    calibration_counts = Counter(
        (row["endpoint_family"], int(row["label"])) for row in calibration_panel.values()
    )
    selection = panel_manifest["selection"]["calibration_pairs"]["dev"]
    conflicts = selection.get("conflicts") or {}
    payload: dict[str, Any] = {
        "schema_version": DENOMINATOR_SCHEMA_VERSION,
        "input_hashes": {
            "preregistration_sha256": contract["preregistration_sha256"],
            "evidence_manifest_sha256": sha256_file(evidence_manifest_path),
            "target_universe_sha256": contract["artifacts"]["target_universe"]["sha256"],
            "dev_ranking_queries_sha256": contract["artifacts"]["dev_ranking_queries"]["sha256"],
            "dev_calibration_pairs_sha256": contract["artifacts"]["dev_calibration_pairs"]["sha256"],
            "dual_cold_ranking_queries_sha256": contract["artifacts"]["dual_cold_ranking_queries"]["sha256"],
        },
        "target_universe_count": target_universe_count,
        "dev_query_count": len(dev_panel),
        "dev_truth_pair_count": sum(len(targets) for targets in dev_panel.values()),
        "calibration_pair_count": len(calibration_panel),
        "calibration_positive_count": sum(row["label"] == 1 for row in calibration_panel.values()),
        "calibration_negative_count": sum(row["label"] == 0 for row in calibration_panel.values()),
        "calibration_by_endpoint_and_label": {
            f"{endpoint}|label={label}": count
            for (endpoint, label), count in sorted(calibration_counts.items())
        },
        "gray_measurement_groups_excluded": int(selection.get("gray_excluded", 0)),
        "conflicting_measurement_groups_excluded": int(
            conflicts.get("conflicting_measurement_groups_excluded", 0)
        ),
        "conflicting_measurement_rows_excluded": int(
            conflicts.get("conflicting_measurement_rows_excluded", 0)
        ),
        "dual_cold_query_count": len(dual_panel),
        "dual_cold_truth_pair_count": sum(len(targets) for targets in dual_panel.values()),
        **dual_denominators,
    }
    payload["binding_sha256"] = canonical_sha256(payload)
    return payload


def evaluate(
    *,
    rank_evidence_csv: Path,
    calibration_evidence_csv: Path,
    evidence_manifest_path: Path,
    prereg_json: Path,
    resources_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    contract = validate_sealed_contract(_read_json(prereg_json, "preregistration"))
    artifacts = contract["artifacts"]
    target_path = Path(artifacts["target_universe"]["path"])
    panel_manifest_path = Path(artifacts["evaluation_panel_manifest"]["path"])
    dev_path = Path(artifacts["dev_ranking_queries"]["path"])
    calibration_path = Path(artifacts["dev_calibration_pairs"]["path"])
    dual_path = Path(artifacts["dual_cold_ranking_queries"]["path"])

    target_clusters = _target_universe(target_path)
    target_ids = set(target_clusters)
    panel_manifest = _panel_manifest(panel_manifest_path, contract)
    dev_panel = _ranking_panel(
        dev_path, "dev ranking panel", dual_cold=False, target_ids=target_ids
    )
    dual_panel = _ranking_panel(
        dual_path, "dual-cold ranking panel", dual_cold=True, target_ids=target_ids
    )
    calibration_panel = _calibration_panel(calibration_path, target_ids)
    # Validate that dev and dual-cold panels share no queries so rank evidence
    # is not counted twice in separate evaluations.
    dev_queries = set(dev_panel.keys())
    dual_queries = set(dual_panel.keys())
    overlap = dev_queries & dual_queries
    if overlap:
        raise PerformanceV2ContractError(
            f"dev and dual-cold panels share {len(overlap)} query(ies): "
            f"{sorted(overlap)[:5]}"
        )
    expected_pairs = {
        (query, target)
        for panel in (dev_panel, dual_panel)
        for query, targets in panel.items()
        for target in targets
    }
    calibrators = _evidence_manifest(
        evidence_manifest_path,
        rank_evidence_path=rank_evidence_csv,
        calibration_evidence_path=calibration_evidence_csv,
        contract=contract,
        universe_count=len(target_ids),
    )
    ranks = _rank_evidence(rank_evidence_csv, expected_pairs, len(target_ids))
    probabilities = _calibration_evidence(
        calibration_evidence_csv, calibration_panel, calibrators
    )
    resources = _resources(resources_path, contract)
    ranking_metrics = _ranking_metrics(ranks, dev_panel)
    calibration_metrics = _calibration_metrics(probabilities, calibration_panel)
    dual_metrics, dual_denominators = _dual_cold_metrics(
        ranks, dual_panel, target_clusters
    )

    rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        rows.append(
            {
                "candidate_id": candidate,
                **resources[candidate],
                "training_data_sha256": artifacts["training_data"]["sha256"],
                "target_universe_sha256": artifacts["target_universe"]["sha256"],
                "dev_benchmark_sha256": artifacts["dev_ranking_queries"]["sha256"],
                "seeds_sha256": canonical_sha256(SEEDS),
                **ranking_metrics[candidate],
                **calibration_metrics[candidate],
                **dual_metrics[candidate],
            }
        )
    ordered = [{column: row[column] for column in RESULT_COLUMNS} for row in rows]
    denominators = _denominators(
        contract=contract,
        panel_manifest=panel_manifest,
        target_universe_count=len(target_ids),
        dev_panel=dev_panel,
        dual_panel=dual_panel,
        calibration_panel=calibration_panel,
        dual_denominators=dual_denominators,
        evidence_manifest_path=evidence_manifest_path,
    )
    return ordered, denominators


def _write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank-evidence-csv", required=True, type=Path)
    parser.add_argument("--calibration-evidence-csv", required=True, type=Path)
    parser.add_argument("--evidence-manifest", required=True, type=Path)
    parser.add_argument("--prereg-json", required=True, type=Path)
    parser.add_argument("--resources", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--out-denominators-json", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = (args.out_csv, args.out_denominators_json)
    try:
        with append_only_outputs(outputs):
            rows, denominators = evaluate(
                rank_evidence_csv=args.rank_evidence_csv,
                calibration_evidence_csv=args.calibration_evidence_csv,
                evidence_manifest_path=args.evidence_manifest,
                prereg_json=args.prereg_json,
                resources_path=args.resources,
            )
            _write_csv_atomic(args.out_csv, rows)
            _write_json_atomic(args.out_denominators_json, denominators)
    except ContractError as exc:
        raise PerformanceV2ContractError(str(exc)) from exc
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PerformanceV2ContractError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
