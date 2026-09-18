#!/usr/bin/env python3
"""Create and revalidate the fail-closed activity-retrieval production gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


GATE_SCHEMA = "skinscout.activity-retrieval-production-gate.v1"
OPERATIONAL_GATE_SCHEMA = "skinscout.activity-retrieval-operational-gate.v1"
SOURCE_TARGET_EXCLUSION_SCHEMA = "skinscout.activity-source-target-exclusions.v1"
RECIPE_SCHEMA = "skinscout.activity-retrieval-recipe.v1"
RUNTIME_INDEX_SCHEMA = "skinscout.activity-retrieval-index.production.v1"
# Compared field-for-field against asdict(BASELINE), so a new Recipe field has
# to be added here too. max_union_any arrived 2026-08-31; it is 0.0 on the
# baseline, which keeps the frozen definition numerically identical.
FROZEN_BASELINE_RECIPE = {
    "recipe_id": "chembl_p5_max",
    "max_union5": 0.0,
    "max_union6": 0.0,
    "quality_union6": 0.0,
    "source_consensus6": 0.0,
    "support_union6": 0.0,
    "negative_contrast": 0.0,
    "max_union_any": 0.0,
    "max_union_nonneg": 0.0,
}
COLD_ADEQUACY_FLOORS = {
    "min_queries": 100,
    "min_unique_truth_targets": 20,
    "min_unique_source_documents": 20,
    "max_truth_pair_target_fraction": 0.20,
    "min_effective_target_count": 10.0,
}
MANIFEST_SCHEMAS = {
    "benchmark_manifest": "activity_benchmark.v1",
    "index_manifest": "skinscout.activity-retrieval-index.v4",
    "panels_manifest": "skinscout.activity-recovery-panels.v5",
    "selection_manifest": "skinscout.activity-retrieval-selection.v1",
    "final_evaluation_manifest": "skinscout.activity-retrieval-evaluation.v1",
    "pocket_leakage_manifest": "skinscout.rcsb-contact-pocket-leakage-audit.v1",
}
RCSB_PANEL_SCHEMA = "skinscout.rcsb-holo-direct-contact-panel.v1"
RCSB_FRAGMENT_SCHEMA = "skinscout.rcsb-contact-pocket-fragments.v1"
PRIOR_FRAGMENT_SCHEMA = "skinscout.pocket-fragment-universe.v1"
POCKET_TM_THRESHOLDS = (0.4, 0.5, 0.6)
POCKET_THRESHOLD_COLUMNS = {
    "0.4": "leakage_tm40",
    "0.5": "leakage_tm50",
    "0.6": "leakage_tm60",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_digest(root: Path) -> str:
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"required activity retrieval directory is missing: {root}")
    digest = hashlib.sha256()
    for path in sorted(
        item
        for item in root.rglob("*")
        if item.is_file() and item.name != ".snakemake_timestamp"
    ):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _load_json(path: Path, schema: str) -> dict[str, Any]:
    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"required activity retrieval artifact is missing or empty: {path}")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid activity retrieval JSON artifact: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != schema:
        raise SystemExit(f"{path} schema_version must be {schema}")
    return payload


def _required_mapping(payload: dict[str, Any], key: str, source: Path) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise SystemExit(f"{source} must contain a mapping at {key}")
    return value


def _required_record(
    payload: dict[str, Any], section: str, key: str, source: Path
) -> dict[str, Any]:
    mapping = _required_mapping(payload, section, source)
    record = mapping.get(key)
    if not isinstance(record, dict):
        raise SystemExit(f"{source} must contain {section}.{key}")
    path = record.get("path")
    digest = record.get("sha256")
    if not isinstance(path, str) or not path.strip():
        raise SystemExit(f"{source} {section}.{key}.path must be non-empty")
    if not isinstance(digest, str) or len(digest) != 64:
        raise SystemExit(f"{source} {section}.{key}.sha256 must be a SHA-256 digest")
    return record


def _assert_digest(actual_path: Path, expected: object, context: str) -> None:
    if not actual_path.exists() or not actual_path.is_file():
        raise SystemExit(f"{context} artifact is missing: {actual_path}")
    actual = _sha256(actual_path)
    if actual != expected:
        raise SystemExit(
            f"{context} is stale: expected sha256={expected}, actual sha256={actual}"
        )


def _artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _resolve_path(value: object, source: Path, context: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"{context} path must be non-empty")
    path = Path(value)
    if not path.is_absolute():
        path = source.parent / path
    return path.resolve()


def _validate_file_record(
    record: object,
    source: Path,
    context: str,
    *,
    require_rows: bool = False,
) -> tuple[Path, int | None]:
    if not isinstance(record, dict):
        raise SystemExit(f"{context} record is missing")
    path = _resolve_path(record.get("path"), source, context)
    digest = record.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise SystemExit(f"{context} sha256 must be a SHA-256 digest")
    _assert_digest(path, digest, context)
    rows = record.get("rows")
    if require_rows and (
        not isinstance(rows, int) or isinstance(rows, bool) or rows < 0
    ):
        raise SystemExit(f"{context} rows must be a non-negative integer")
    return path, rows if isinstance(rows, int) and not isinstance(rows, bool) else None


def _nonnegative_int(value: object, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SystemExit(f"{context} must be a non-negative integer")
    return value


def _assert_rate(value: object, numerator: int, denominator: int, context: str) -> None:
    expected = None if denominator == 0 else numerator / denominator
    if expected is None:
        if value is not None:
            raise SystemExit(f"{context} must be null for a zero denominator")
        return
    if isinstance(value, bool):
        raise SystemExit(f"{context} must be a finite rate")
    try:
        actual = float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{context} must be a finite rate") from exc
    if not math.isfinite(actual) or not 0.0 <= actual <= 1.0:
        raise SystemExit(f"{context} must be a finite rate in [0,1]")
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise SystemExit(f"{context} is stale")


def _read_csv_records(path: Path, context: str) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise SystemExit(f"{context} must contain a CSV header")
            rows = [dict(row) for row in reader]
    except (OSError, csv.Error) as exc:
        raise SystemExit(f"unable to parse {context}: {path}") from exc
    return list(reader.fieldnames), rows


def _validate_source_target_exclusions(
    benchmark: dict[str, Any], benchmark_manifest: Path
) -> Path:
    filters = _required_mapping(benchmark, "filters", benchmark_manifest)
    contract = filters.get("source_target_exclusions")
    if not isinstance(contract, dict):
        raise SystemExit(
            "activity retrieval benchmark must contain filters.source_target_exclusions"
        )
    if contract.get("applied") is not True:
        raise SystemExit("activity retrieval benchmark source-target exclusions were not applied")
    if contract.get("schema_version") != SOURCE_TARGET_EXCLUSION_SCHEMA:
        raise SystemExit("activity retrieval benchmark source-target exclusion schema is invalid")
    if contract.get("all_rules_matched") is not True:
        raise SystemExit("activity retrieval benchmark has unmatched source-target exclusions")
    rule_count = contract.get("rule_count")
    filtered_rows = contract.get("total_filtered_rows")
    rules = contract.get("rules")
    if (
        not isinstance(rule_count, int)
        or isinstance(rule_count, bool)
        or rule_count < 1
        or not isinstance(filtered_rows, int)
        or isinstance(filtered_rows, bool)
        or filtered_rows < rule_count
        or not isinstance(rules, list)
        or len(rules) != rule_count
        or any(
            not isinstance(rule, dict)
            or not isinstance(rule.get("matched_rows"), int)
            or isinstance(rule.get("matched_rows"), bool)
            or rule["matched_rows"] < 1
            for rule in rules
        )
    ):
        raise SystemExit("activity retrieval benchmark source-target exclusion audit is invalid")
    artifact = contract.get("artifact")
    if not isinstance(artifact, dict):
        raise SystemExit("activity retrieval benchmark source-target exclusion artifact is missing")
    path_text = artifact.get("path")
    digest = artifact.get("sha256")
    rows = artifact.get("rows")
    if (
        not isinstance(path_text, str)
        or not path_text.strip()
        or not isinstance(digest, str)
        or len(digest) != 64
        or rows != rule_count
    ):
        raise SystemExit("activity retrieval benchmark source-target exclusion artifact is invalid")
    input_hashes = _required_mapping(benchmark, "input_sha256", benchmark_manifest)
    if input_hashes.get("source_target_exclusions") != digest:
        raise SystemExit(
            "activity retrieval benchmark source-target exclusion input binding is stale"
        )
    artifact_path = Path(path_text)
    _assert_digest(
        artifact_path,
        digest,
        "activity retrieval source-target exclusion contract",
    )
    return artifact_path


def _validate_cold_panel_adequacy(panels: dict[str, Any], source: Path) -> None:
    if panels.get("passes_panel_adequacy_gate") is not True:
        raise SystemExit("activity retrieval dual-cold panel adequacy gate did not pass")
    selection = _required_mapping(panels, "selection", source)
    ranking = _required_mapping(selection, "ranking_queries", source)
    dual = ranking.get("dual_cold")
    adequacy = dual.get("adequacy") if isinstance(dual, dict) else None
    if not isinstance(adequacy, dict) or adequacy.get("passes") is not True:
        raise SystemExit("activity retrieval dual-cold adequacy audit is invalid")
    criteria = adequacy.get("criteria")
    observed = adequacy.get("observed")
    checks = adequacy.get("checks")
    if not all(isinstance(value, dict) for value in (criteria, observed, checks)):
        raise SystemExit("activity retrieval dual-cold adequacy audit is incomplete")
    try:
        numeric_criteria = {
            key: float(criteria[key]) for key in COLD_ADEQUACY_FLOORS
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit("activity retrieval dual-cold adequacy criteria are invalid") from exc
    if any(
        isinstance(criteria[key], bool) or not math.isfinite(value)
        for key, value in numeric_criteria.items()
    ):
        raise SystemExit("activity retrieval dual-cold adequacy criteria are invalid")
    for key in (
        "min_queries",
        "min_unique_truth_targets",
        "min_unique_source_documents",
        "min_effective_target_count",
    ):
        if numeric_criteria[key] < COLD_ADEQUACY_FLOORS[key]:
            raise SystemExit(f"activity retrieval dual-cold adequacy criterion is too weak: {key}")
    if numeric_criteria["max_truth_pair_target_fraction"] > (
        COLD_ADEQUACY_FLOORS["max_truth_pair_target_fraction"]
    ):
        raise SystemExit(
            "activity retrieval dual-cold adequacy criterion is too weak: "
            "max_truth_pair_target_fraction"
        )

    target_counts = observed.get("truth_pairs_by_target")
    documents = observed.get("source_documents")
    if not isinstance(target_counts, dict) or not target_counts:
        raise SystemExit("activity retrieval dual-cold target incidence audit is invalid")
    if not isinstance(documents, list) or not documents:
        raise SystemExit("activity retrieval dual-cold source-document audit is invalid")
    if any(
        not isinstance(target, str)
        or not target.strip()
        or not isinstance(count, int)
        or isinstance(count, bool)
        or count < 1
        for target, count in target_counts.items()
    ):
        raise SystemExit("activity retrieval dual-cold target incidence audit is invalid")
    document_keys: set[tuple[str, str]] = set()
    for document in documents:
        if not isinstance(document, dict):
            raise SystemExit("activity retrieval dual-cold source-document audit is invalid")
        source_db = document.get("source_db")
        publication_key = document.get("publication_key")
        if (
            not isinstance(source_db, str)
            or not source_db.strip()
            or not isinstance(publication_key, str)
            or not publication_key.strip()
        ):
            raise SystemExit("activity retrieval dual-cold source-document audit is invalid")
        document_keys.add((source_db, publication_key))
    if len(document_keys) != len(documents):
        raise SystemExit("activity retrieval dual-cold source-document audit has duplicates")

    truth_pairs = sum(target_counts.values())
    fractions = [count / truth_pairs for count in target_counts.values()]
    recalculated_max_fraction = max(fractions)
    recalculated_effective_targets = 1.0 / sum(value * value for value in fractions)
    expected_observed = {
        "truth_pairs": truth_pairs,
        "unique_truth_targets": len(target_counts),
        "unique_source_documents": len(document_keys),
        "source_databases": sorted({key[0] for key in document_keys}),
    }
    for key, expected in expected_observed.items():
        if observed.get(key) != expected:
            raise SystemExit(f"activity retrieval dual-cold adequacy {key} is stale")
    try:
        recorded_max_fraction = float(observed["max_truth_pair_target_fraction"])
        recorded_effective_targets = float(observed["effective_target_count"])
        observed_queries = int(observed["queries"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit("activity retrieval dual-cold adequacy observations are invalid") from exc
    if (
        isinstance(observed.get("queries"), bool)
        or not math.isclose(recorded_max_fraction, recalculated_max_fraction)
        or not math.isclose(recorded_effective_targets, recalculated_effective_targets)
    ):
        raise SystemExit("activity retrieval dual-cold adequacy observations are stale")
    expected_checks = {
        "min_queries": observed_queries >= numeric_criteria["min_queries"],
        "min_unique_truth_targets": (
            len(target_counts) >= numeric_criteria["min_unique_truth_targets"]
        ),
        "min_unique_source_documents": (
            len(document_keys) >= numeric_criteria["min_unique_source_documents"]
        ),
        "max_truth_pair_target_fraction": (
            recalculated_max_fraction
            <= numeric_criteria["max_truth_pair_target_fraction"]
        ),
        "min_effective_target_count": (
            recalculated_effective_targets
            >= numeric_criteria["min_effective_target_count"]
        ),
    }
    if checks != expected_checks or any(value is not True for value in checks.values()):
        raise SystemExit("activity retrieval dual-cold adequacy checks are stale or failed")
    dual_output = _required_record(
        panels, "outputs", "dual_cold_ranking_queries.parquet", source
    )
    if int(dual_output.get("rows", -1)) != observed_queries:
        raise SystemExit("activity retrieval dual-cold adequacy query count is stale")


def _validate_rcsb_panel_binding(panels: dict[str, Any], source: Path) -> Path:
    inputs = _required_mapping(panels, "inputs", source)
    record = inputs.get("rcsb_holo_direct_contact_panel")
    if not isinstance(record, dict):
        raise SystemExit(
            "activity retrieval panels must bind rcsb_holo_direct_contact_panel"
        )
    path_text = record.get("path")
    digest = record.get("sha256")
    if not isinstance(path_text, str) or not path_text.strip():
        raise SystemExit("activity retrieval RCSB ranking binding path is invalid")
    if not isinstance(digest, str) or len(digest) != 64:
        raise SystemExit("activity retrieval RCSB ranking binding sha256 is invalid")
    ranking_path = Path(path_text)
    _assert_digest(ranking_path, digest, "activity retrieval RCSB ranking panel")

    manifest = record.get("manifest")
    if not isinstance(manifest, dict):
        raise SystemExit("activity retrieval RCSB panel manifest binding is missing")
    manifest_path_text = manifest.get("path")
    manifest_digest = manifest.get("sha256")
    if (
        manifest.get("schema_version") != RCSB_PANEL_SCHEMA
        or not isinstance(manifest_path_text, str)
        or not manifest_path_text.strip()
        or not isinstance(manifest_digest, str)
        or len(manifest_digest) != 64
    ):
        raise SystemExit("activity retrieval RCSB panel manifest binding is invalid")
    manifest_path = Path(manifest_path_text)
    _assert_digest(
        manifest_path,
        manifest_digest,
        "activity retrieval RCSB panel manifest",
    )

    contract = record.get("contract")
    if not isinstance(contract, dict):
        raise SystemExit("activity retrieval RCSB panel contract is missing")
    required_contract = {
        "positive_only": True,
        "no_affinities_or_calibration": True,
        "never_training_or_model_selection": True,
    }
    if any(contract.get(key) is not expected for key, expected in required_contract.items()):
        raise SystemExit(
            "activity retrieval RCSB panel contract must be positive-only, "
            "no-calibration, and never-model-selection"
        )
    return manifest_path


def _validate_pocket_leakage_audit(
    payload: dict[str, Any],
    source: Path,
    benchmark_manifest: Path,
    rcsb_panel_manifest: Path,
) -> dict[str, Path]:
    contract = _required_mapping(payload, "contract", source)
    required_contract = (
        "evaluation_only",
        "never_training",
        "never_calibration",
        "never_model_selection",
        "missing_foldseek_hits_are_uncovered_not_cold",
        "fragment_extraction_failures_are_uncovered_not_cold",
    )
    if any(contract.get(key) is not True for key in required_contract):
        raise SystemExit(
            "activity retrieval pocket leakage contract must remain evaluation-only, "
            "never-training/calibration/model-selection, and fail closed on uncovered pairs"
        )

    thresholds = _required_mapping(payload, "thresholds", source)
    raw_thresholds = thresholds.get("max_directional_tm")
    if (
        not isinstance(raw_thresholds, list)
        or len(raw_thresholds) != len(POCKET_TM_THRESHOLDS)
        or any(isinstance(value, bool) for value in raw_thresholds)
    ):
        raise SystemExit("activity retrieval pocket leakage thresholds are invalid")
    try:
        parsed_thresholds = tuple(float(value) for value in raw_thresholds)
    except (TypeError, ValueError) as exc:
        raise SystemExit("activity retrieval pocket leakage thresholds are invalid") from exc
    if parsed_thresholds != POCKET_TM_THRESHOLDS:
        raise SystemExit("activity retrieval pocket leakage thresholds must be 0.4, 0.5, 0.6")
    if thresholds.get("columns") != POCKET_THRESHOLD_COLUMNS:
        raise SystemExit("activity retrieval pocket leakage threshold columns are invalid")

    inputs = _required_mapping(payload, "inputs", source)
    benchmark_record = inputs.get("activity_benchmark_manifest")
    benchmark_path, _ = _validate_file_record(
        benchmark_record,
        source,
        "activity retrieval pocket leakage benchmark manifest",
    )
    if not isinstance(benchmark_record, dict) or (
        benchmark_record.get("schema_version")
        != MANIFEST_SCHEMAS["benchmark_manifest"]
    ):
        raise SystemExit("activity retrieval pocket leakage benchmark schema is invalid")
    if (
        benchmark_path != benchmark_manifest.resolve()
        or benchmark_record.get("sha256") != _sha256(benchmark_manifest)
    ):
        raise SystemExit(
            "activity retrieval pocket leakage audit is stale relative to benchmark manifest"
        )

    contact = inputs.get("rcsb_contact_fragments")
    if not isinstance(contact, dict):
        raise SystemExit("activity retrieval pocket leakage RCSB fragment binding is missing")
    contact_manifest_record = contact.get("manifest")
    contact_manifest_path, _ = _validate_file_record(
        contact_manifest_record,
        source,
        "activity retrieval RCSB contact fragment manifest",
    )
    if not isinstance(contact_manifest_record, dict) or (
        contact_manifest_record.get("schema_version") != RCSB_FRAGMENT_SCHEMA
    ):
        raise SystemExit("activity retrieval RCSB contact fragment schema is invalid")
    contact_manifest = _load_json(contact_manifest_path, RCSB_FRAGMENT_SCHEMA)
    contact_contract = _required_mapping(
        contact_manifest, "contract", contact_manifest_path
    )
    if any(
        contact_contract.get(key) is not True
        for key in (
            "positive_only",
            "no_inferred_negatives",
            "never_prediction",
            "never_training",
            "never_calibration",
            "never_model_selection",
            "strict_pairs_only",
        )
    ):
        raise SystemExit("activity retrieval RCSB contact fragment contract is invalid")

    panel_binding = _required_record(
        contact_manifest,
        "inputs",
        "panel_manifest",
        contact_manifest_path,
    )
    if panel_binding.get("schema_version") != RCSB_PANEL_SCHEMA:
        raise SystemExit("activity retrieval RCSB contact fragments use the wrong panel schema")
    bound_panel_path = _resolve_path(
        panel_binding["path"], contact_manifest_path, "RCSB contact panel manifest"
    )
    if (
        bound_panel_path != rcsb_panel_manifest.resolve()
        or panel_binding["sha256"] != _sha256(rcsb_panel_manifest)
    ):
        raise SystemExit("activity retrieval RCSB contact fragments use a stale panel")
    panel_payload = _load_json(rcsb_panel_manifest, RCSB_PANEL_SCHEMA)
    panel_pairs = _required_record(
        panel_payload, "outputs", "pairs", rcsb_panel_manifest
    )
    fragment_pairs = _required_record(
        contact_manifest, "inputs", "pairs_parquet", contact_manifest_path
    )
    panel_pairs_path = _resolve_path(
        panel_pairs["path"], rcsb_panel_manifest, "RCSB panel pairs"
    )
    fragment_pairs_path = _resolve_path(
        fragment_pairs["path"], contact_manifest_path, "RCSB fragment pairs"
    )
    _assert_digest(panel_pairs_path, panel_pairs["sha256"], "RCSB panel pairs")
    if (
        fragment_pairs_path != panel_pairs_path
        or fragment_pairs["sha256"] != panel_pairs["sha256"]
    ):
        raise SystemExit("activity retrieval RCSB contact fragment pair binding is stale")

    contact_artifacts = _required_mapping(
        contact_manifest, "artifacts", contact_manifest_path
    )
    index_path, index_rows = _validate_file_record(
        contact_artifacts.get("index_csv"),
        contact_manifest_path,
        "RCSB contact fragment index",
        require_rows=True,
    )
    exclusions_path, exclusions_rows = _validate_file_record(
        contact_artifacts.get("exclusions_csv"),
        contact_manifest_path,
        "RCSB contact fragment exclusions",
        require_rows=True,
    )
    index_fields, index_records = _read_csv_records(
        index_path, "RCSB contact fragment index"
    )
    exclusion_fields, exclusion_records = _read_csv_records(
        exclusions_path, "RCSB contact fragment exclusions"
    )
    if "pair_id" not in index_fields or "pair_id" not in exclusion_fields:
        raise SystemExit("RCSB contact fragment CSVs must contain pair_id")
    if len(index_records) != index_rows or len(exclusion_records) != exclusions_rows:
        raise SystemExit("RCSB contact fragment CSV row counts are stale")
    contact_pair_ids = [row.get("pair_id", "").strip() for row in index_records]
    contact_pair_ids.extend(
        row.get("pair_id", "").strip() for row in exclusion_records
    )
    if any(not pair_id for pair_id in contact_pair_ids) or len(set(contact_pair_ids)) != len(
        contact_pair_ids
    ):
        raise SystemExit("RCSB contact fragment pair accounting is invalid")

    fragment_dir_record = contact_artifacts.get("fragment_dir")
    if not isinstance(fragment_dir_record, dict):
        raise SystemExit("RCSB contact fragment directory record is missing")
    fragment_dir = _resolve_path(
        fragment_dir_record.get("path"), contact_manifest_path, "RCSB fragment directory"
    )
    if fragment_dir_record.get("tree_sha256") != _tree_digest(fragment_dir):
        raise SystemExit("RCSB contact fragment directory is stale")
    if fragment_dir_record.get("files") != index_rows:
        raise SystemExit("RCSB contact fragment file count is stale")

    audit_index_path, audit_index_rows = _validate_file_record(
        contact.get("index"), source, "pocket leakage RCSB fragment index", require_rows=True
    )
    audit_exclusions_path, audit_exclusions_rows = _validate_file_record(
        contact.get("exclusions"),
        source,
        "pocket leakage RCSB fragment exclusions",
        require_rows=True,
    )
    audit_fragment_dir = contact.get("fragment_dir")
    if not isinstance(audit_fragment_dir, dict):
        raise SystemExit("pocket leakage RCSB fragment directory binding is missing")
    if (
        audit_index_path != index_path
        or audit_index_rows != index_rows
        or audit_exclusions_path != exclusions_path
        or audit_exclusions_rows != exclusions_rows
        or _resolve_path(
            audit_fragment_dir.get("path"), source, "pocket leakage RCSB fragment directory"
        )
        != fragment_dir
        or audit_fragment_dir.get("tree_sha256")
        != fragment_dir_record.get("tree_sha256")
        or audit_fragment_dir.get("files") != index_rows
    ):
        raise SystemExit("activity retrieval pocket leakage RCSB fragment binding is stale")

    contact_counts = _required_mapping(contact_manifest, "counts", contact_manifest_path)
    strict_pairs = _nonnegative_int(
        contact_counts.get("strict_dual_cold_pairs"), "RCSB strict pair count"
    )
    indexed = _nonnegative_int(contact_counts.get("indexed"), "RCSB indexed pair count")
    excluded = _nonnegative_int(contact_counts.get("excluded"), "RCSB excluded pair count")
    if indexed != index_rows or excluded != exclusions_rows or strict_pairs != indexed + excluded:
        raise SystemExit("RCSB contact fragment pair counts are stale")
    if fragment_pairs.get("strict_dual_cold_rows") != strict_pairs:
        raise SystemExit("RCSB contact fragment strict pair count is stale")

    prior = inputs.get("prior_p2rank_fragments")
    if not isinstance(prior, dict):
        raise SystemExit("activity retrieval prior pocket fragment binding is missing")
    prior_manifest_record = prior.get("manifest")
    prior_manifest_path, _ = _validate_file_record(
        prior_manifest_record, source, "prior pocket fragment manifest"
    )
    if not isinstance(prior_manifest_record, dict) or (
        prior_manifest_record.get("schema_version") != PRIOR_FRAGMENT_SCHEMA
    ):
        raise SystemExit("activity retrieval prior pocket fragment schema is invalid")
    prior_manifest = _load_json(prior_manifest_path, PRIOR_FRAGMENT_SCHEMA)
    prior_artifacts = _required_mapping(prior_manifest, "artifacts", prior_manifest_path)
    prior_index_path, prior_index_rows = _validate_file_record(
        prior_artifacts.get("index"),
        prior_manifest_path,
        "prior pocket fragment index",
        require_rows=True,
    )
    prior_fields, prior_rows = _read_csv_records(
        prior_index_path, "prior pocket fragment index"
    )
    if "uniprot" not in prior_fields or len(prior_rows) != prior_index_rows:
        raise SystemExit("prior pocket fragment index row count is stale")
    prior_dir = _resolve_path(
        prior_artifacts.get("out_dir"), prior_manifest_path, "prior pocket fragment directory"
    )
    prior_tree_sha = prior_artifacts.get("out_dir_tree_sha256")
    if prior_tree_sha != _tree_digest(prior_dir):
        raise SystemExit("prior pocket fragment directory is stale")
    audit_prior_index_path, audit_prior_index_rows = _validate_file_record(
        prior.get("index"), source, "pocket leakage prior fragment index", require_rows=True
    )
    audit_prior_dir = prior.get("fragment_dir")
    if not isinstance(audit_prior_dir, dict):
        raise SystemExit("pocket leakage prior fragment directory binding is missing")
    if (
        audit_prior_index_path != prior_index_path
        or audit_prior_index_rows != prior_index_rows
        or _resolve_path(
            audit_prior_dir.get("path"), source, "pocket leakage prior fragment directory"
        )
        != prior_dir
        or audit_prior_dir.get("tree_sha256") != prior_tree_sha
        or audit_prior_dir.get("files") != prior_index_rows
    ):
        raise SystemExit("activity retrieval pocket leakage prior fragment binding is stale")

    outputs = _required_mapping(payload, "outputs", source)
    detailed_path, detailed_rows = _validate_file_record(
        outputs.get("detailed_csv"),
        source,
        "activity retrieval pocket leakage detailed CSV",
        require_rows=True,
    )
    output_manifest = outputs.get("manifest")
    if not isinstance(output_manifest, dict) or _resolve_path(
        output_manifest.get("path"), source, "pocket leakage output manifest"
    ) != source.resolve():
        raise SystemExit("activity retrieval pocket leakage output manifest path is stale")
    detailed_fields, rows = _read_csv_records(
        detailed_path, "activity retrieval pocket leakage detailed CSV"
    )
    required_fields = {
        "pair_id",
        "ligand_key",
        "fragment_extracted",
        "foldseek_covered",
        *POCKET_THRESHOLD_COLUMNS.values(),
    }
    if not required_fields.issubset(detailed_fields):
        raise SystemExit("activity retrieval pocket leakage detailed CSV schema is invalid")
    if detailed_rows != len(rows) or detailed_rows != strict_pairs or detailed_rows < 1:
        raise SystemExit("activity retrieval pocket leakage detailed CSV row count is stale")
    detailed_pair_ids = [row["pair_id"].strip() for row in rows]
    if (
        any(not pair_id for pair_id in detailed_pair_ids)
        or len(set(detailed_pair_ids)) != len(detailed_pair_ids)
        or set(detailed_pair_ids) != set(contact_pair_ids)
    ):
        raise SystemExit("activity retrieval pocket leakage pair accounting is stale")

    parsed_rows: list[dict[str, Any]] = []
    for row in rows:
        ligand_key = row["ligand_key"].strip()
        if not ligand_key:
            raise SystemExit("activity retrieval pocket leakage contains a blank ligand key")
        parsed: dict[str, Any] = {"ligand_key": ligand_key}
        for key in (
            "fragment_extracted",
            "foldseek_covered",
            *POCKET_THRESHOLD_COLUMNS.values(),
        ):
            value = row[key].strip().lower()
            if value not in {"true", "false"}:
                raise SystemExit(
                    f"activity retrieval pocket leakage {key} must be true or false"
                )
            parsed[key] = value == "true"
        if parsed["foldseek_covered"] and not parsed["fragment_extracted"]:
            raise SystemExit("an unextracted pocket cannot be Foldseek-covered")
        leak_values = [
            parsed[POCKET_THRESHOLD_COLUMNS[key]] for key in ("0.4", "0.5", "0.6")
        ]
        if any(leak_values) and not parsed["foldseek_covered"]:
            raise SystemExit("an uncovered pocket cannot be marked as leaked")
        if leak_values[2] and not leak_values[1] or leak_values[1] and not leak_values[0]:
            raise SystemExit("activity retrieval pocket leakage thresholds are non-monotonic")
        parsed_rows.append(parsed)

    total = len(parsed_rows)
    extracted = sum(row["fragment_extracted"] for row in parsed_rows)
    covered = sum(row["foldseek_covered"] for row in parsed_rows)
    pair_summary = _required_mapping(
        _required_mapping(payload, "summary", source), "pairs", source
    )
    expected_pair_counts = {
        "total": total,
        "fragment_extracted": extracted,
        "fragment_extraction_excluded": total - extracted,
        "covered": covered,
        "uncovered": total - covered,
    }
    for key, expected in expected_pair_counts.items():
        if _nonnegative_int(pair_summary.get(key), f"pocket leakage pairs.{key}") != expected:
            raise SystemExit(f"activity retrieval pocket leakage pairs.{key} is stale")
    _assert_rate(pair_summary.get("coverage_rate"), covered, total, "pocket coverage rate")

    pair_leakage = pair_summary.get("leakage_by_max_directional_tm_threshold")
    if not isinstance(pair_leakage, dict) or set(pair_leakage) != set(
        POCKET_THRESHOLD_COLUMNS
    ):
        raise SystemExit("activity retrieval pair leakage summary is invalid")
    ligand_groups: dict[str, list[dict[str, Any]]] = {}
    for row in parsed_rows:
        ligand_groups.setdefault(row["ligand_key"], []).append(row)
    ligand_summary = _required_mapping(
        _required_mapping(payload, "summary", source), "ligand_queries", source
    )
    ligand_total = len(ligand_groups)
    ligand_covered = sum(
        any(row["foldseek_covered"] for row in group)
        for group in ligand_groups.values()
    )
    if (
        _nonnegative_int(ligand_summary.get("total"), "pocket leakage ligand total")
        != ligand_total
        or _nonnegative_int(
            ligand_summary.get("covered"), "pocket leakage ligand covered"
        )
        != ligand_covered
    ):
        raise SystemExit("activity retrieval pocket leakage ligand summary is stale")
    ligand_leakage = ligand_summary.get("leakage_by_max_directional_tm_threshold")
    if not isinstance(ligand_leakage, dict) or set(ligand_leakage) != set(
        POCKET_THRESHOLD_COLUMNS
    ):
        raise SystemExit("activity retrieval ligand leakage summary is invalid")
    for threshold, column in POCKET_THRESHOLD_COLUMNS.items():
        pair_record = pair_leakage[threshold]
        ligand_record = ligand_leakage[threshold]
        if not isinstance(pair_record, dict) or not isinstance(ligand_record, dict):
            raise SystemExit("activity retrieval pocket leakage threshold record is invalid")
        pair_count = sum(row[column] for row in parsed_rows)
        if _nonnegative_int(
            pair_record.get("count"), f"pocket leakage pair count {threshold}"
        ) != pair_count:
            raise SystemExit(f"activity retrieval pocket leakage pair count {threshold} is stale")
        _assert_rate(
            pair_record.get("rate_all_pairs"),
            pair_count,
            total,
            f"pocket leakage all-pair rate {threshold}",
        )
        _assert_rate(
            pair_record.get("rate_covered_pairs"),
            pair_count,
            covered,
            f"pocket leakage covered-pair rate {threshold}",
        )
        leaked_ligands = sum(
            any(row[column] for row in group) for group in ligand_groups.values()
        )
        if _nonnegative_int(
            ligand_record.get("leaked_ligand_queries"),
            f"pocket leakage ligand count {threshold}",
        ) != leaked_ligands:
            raise SystemExit(
                f"activity retrieval pocket leakage ligand count {threshold} is stale"
            )
        _assert_rate(
            ligand_record.get("leakage_rate"),
            leaked_ligands,
            ligand_total,
            f"pocket leakage ligand rate {threshold}",
        )

    return {
        "pocket_leakage_rows": detailed_path,
        "rcsb_contact_pocket_fragments": contact_manifest_path,
        "prior_pocket_fragments": prior_manifest_path,
    }


def validate_manifest_contract(
    *,
    benchmark_manifest: Path,
    index_manifest: Path,
    panels_manifest: Path,
    selection_manifest: Path,
    final_evaluation_manifest: Path,
    pocket_leakage_manifest: Path,
    require_promotion: bool = True,
) -> dict[str, Any]:
    paths = {
        "benchmark_manifest": benchmark_manifest,
        "index_manifest": index_manifest,
        "panels_manifest": panels_manifest,
        "selection_manifest": selection_manifest,
        "final_evaluation_manifest": final_evaluation_manifest,
        "pocket_leakage_manifest": pocket_leakage_manifest,
    }
    payloads = {
        key: _load_json(path, MANIFEST_SCHEMAS[key]) for key, path in paths.items()
    }
    benchmark = payloads["benchmark_manifest"]
    index = payloads["index_manifest"]
    panels = payloads["panels_manifest"]
    selection = payloads["selection_manifest"]
    final = payloads["final_evaluation_manifest"]
    pocket_leakage = payloads["pocket_leakage_manifest"]

    # A production index also reads dev and test. Measuring recovery against it
    # would score the model on rows it retrieves from, so the gate refuses it by
    # role as well as by schema - an older manifest with no role is train-only.
    index_role = index.get("index_role", "evaluation")
    if index_role != "evaluation":
        raise SystemExit(
            "activity retrieval evaluation requires a train-only index; "
            f"{index_manifest} declares index_role={index_role!r}"
        )

    source_target_exclusions_path = _validate_source_target_exclusions(
        benchmark, benchmark_manifest
    )
    _validate_cold_panel_adequacy(panels, panels_manifest)
    rcsb_panel_manifest_path = _validate_rcsb_panel_binding(panels, panels_manifest)
    pocket_evidence = _validate_pocket_leakage_audit(
        pocket_leakage,
        pocket_leakage_manifest,
        benchmark_manifest,
        rcsb_panel_manifest_path,
    )

    if selection.get("passes_dev_gate") is not True:
        raise SystemExit("activity retrieval dev-selection gate did not pass")
    if require_promotion and final.get("passes_frozen_test_gate") is not True:
        raise SystemExit("activity retrieval final evaluation gate did not pass")
    final_inputs = _required_mapping(final, "inputs", final_evaluation_manifest)
    if final_inputs.get("panels_schema") != MANIFEST_SCHEMAS["panels_manifest"]:
        raise SystemExit("activity retrieval final evaluation uses the wrong panel schema")

    benchmark_sha = _sha256(benchmark_manifest)
    index_sha = _sha256(index_manifest)
    panels_sha = _sha256(panels_manifest)
    expected_bindings = [
        (
            _required_record(index, "inputs", "benchmark_manifest", index_manifest),
            benchmark_sha,
            "retrieval index to benchmark manifest",
        ),
        (
            _required_record(panels, "inputs", "benchmark_manifest", panels_manifest),
            benchmark_sha,
            "recovery panels to benchmark manifest",
        ),
        (
            _required_record(panels, "inputs", "retrieval_index_manifest", panels_manifest),
            index_sha,
            "recovery panels to retrieval index manifest",
        ),
        (
            _required_record(selection, "inputs", "index_manifest", selection_manifest),
            index_sha,
            "dev selection to retrieval index manifest",
        ),
        (
            _required_record(selection, "inputs", "panels_manifest", selection_manifest),
            panels_sha,
            "dev selection to recovery panels manifest",
        ),
        (
            _required_record(final, "inputs", "index_manifest", final_evaluation_manifest),
            index_sha,
            "final evaluation to retrieval index manifest",
        ),
        (
            _required_record(final, "inputs", "panels_manifest", final_evaluation_manifest),
            panels_sha,
            "final evaluation to recovery panels manifest",
        ),
    ]
    for record, expected, context in expected_bindings:
        if record["sha256"] != expected:
            raise SystemExit(f"{context} is stale")

    selection_target = _required_record(
        selection, "inputs", "target_cluster_manifest", selection_manifest
    )
    final_target = _required_record(
        final, "inputs", "target_cluster_manifest", final_evaluation_manifest
    )
    if final_target["sha256"] != selection_target["sha256"]:
        raise SystemExit("final evaluation target universe differs from dev selection")
    target_path = Path(selection_target["path"])
    _assert_digest(target_path, selection_target["sha256"], "target universe manifest")

    recipe = _required_record(selection, "outputs", "recipe.json", selection_manifest)
    final_recipe = _required_record(final, "inputs", "recipe", final_evaluation_manifest)
    if final_recipe["sha256"] != recipe["sha256"]:
        raise SystemExit(
            "activity retrieval final evaluation recipe does not match dev-selected recipe"
        )
    recipe_path = Path(recipe["path"])
    _assert_digest(recipe_path, recipe["sha256"], "dev-selected recipe")

    panel_known = _required_record(panels, "outputs", "known_panel.csv", panels_manifest)
    final_known = _required_record(final, "inputs", "known_panel", final_evaluation_manifest)
    if final_known["sha256"] != panel_known["sha256"]:
        raise SystemExit("final evaluation known panel differs from frozen recovery panel")
    known_path = Path(panel_known["path"])
    _assert_digest(known_path, panel_known["sha256"], "frozen known-target panel")

    summary = _required_record(final, "outputs", "summary.json", final_evaluation_manifest)
    summary_path = Path(summary["path"])
    _assert_digest(summary_path, summary["sha256"], "final evaluation summary")

    return {
        "manifests": {key: _artifact_record(path) for key, path in paths.items()},
        "recipe": _artifact_record(recipe_path),
        "target_cluster_manifest": _artifact_record(target_path),
        "known_panel": _artifact_record(known_path),
        "final_summary": _artifact_record(summary_path),
        "source_target_exclusions": _artifact_record(source_target_exclusions_path),
        "rcsb_holo_direct_contact_panel": _artifact_record(rcsb_panel_manifest_path),
        **{
            key: _artifact_record(path)
            for key, path in pocket_evidence.items()
        },
    }


def _operational_decision_from_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    recipe_path = Path(evidence["recipe"]["path"])
    recipe = _load_json(recipe_path, RECIPE_SCHEMA)
    if recipe.get("passes_dev_gate") is not True:
        raise SystemExit("activity retrieval operational recipe did not pass dev selection")
    baseline = recipe.get("baseline_recipe")
    selected = recipe.get("selected_recipe")
    if baseline != FROZEN_BASELINE_RECIPE:
        raise SystemExit("activity retrieval frozen baseline recipe definition is invalid")
    if not isinstance(selected, dict):
        raise SystemExit("activity retrieval selected recipe definition is missing")
    selected_recipe_id = selected.get("recipe_id")
    if not isinstance(selected_recipe_id, str) or not selected_recipe_id.strip():
        raise SystemExit("activity retrieval selected recipe identifier is invalid")
    baseline_recipe_id = FROZEN_BASELINE_RECIPE["recipe_id"]

    final_manifest_path = Path(
        evidence["manifests"]["final_evaluation_manifest"]["path"]
    )
    final = _load_json(
        final_manifest_path,
        MANIFEST_SCHEMAS["final_evaluation_manifest"],
    )
    summary_path = Path(evidence["final_summary"]["path"])
    summary = _load_json(
        summary_path,
        MANIFEST_SCHEMAS["final_evaluation_manifest"],
    )
    passes = final.get("passes_frozen_test_gate")
    if not isinstance(passes, bool):
        raise SystemExit("activity retrieval final promotion decision must be boolean")
    expected = {
        "evaluation_completed": True,
        "claim_ready": passes,
        "promotion_decision": (
            "promote_selected" if passes else "retain_frozen_baseline"
        ),
        "operational_recipe_id": (
            selected_recipe_id if passes else baseline_recipe_id
        ),
        "selected_recipe_id": selected_recipe_id,
        "baseline_recipe_id": baseline_recipe_id,
    }
    for label, payload in (("manifest", final), ("summary", summary)):
        if payload.get("passes_frozen_test_gate") is not passes:
            raise SystemExit(
                f"activity retrieval final {label} promotion status is inconsistent"
            )
        for key, value in expected.items():
            if payload.get(key) != value:
                raise SystemExit(
                    f"activity retrieval final {label} {key} is inconsistent"
                )
    return {
        "claim_ready": passes,
        "promotion_decision": expected["promotion_decision"],
        "operational_recipe_id": expected["operational_recipe_id"],
        "selected_recipe_id": selected_recipe_id,
        "baseline_recipe_id": baseline_recipe_id,
    }


def _validate_runtime_operational_inputs(
    *,
    runtime_index_manifest: Path,
    runtime_recipe: Path,
    evidence: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    runtime_index = _load_json(runtime_index_manifest, RUNTIME_INDEX_SCHEMA)
    if runtime_index.get("index_role") != "production":
        raise SystemExit("activity retrieval runtime index must declare index_role=production")
    algorithm = runtime_index.get("algorithm")
    standardization = algorithm.get("standardization") if isinstance(algorithm, dict) else None
    if not isinstance(standardization, list) or "rdMolStandardize.Uncharger" not in standardization:
        raise SystemExit(
            "activity retrieval runtime index uses stale ligand standardization; rebuild it"
        )
    for key in ("ligands", "edges"):
        record = _required_record(runtime_index, "outputs", key, runtime_index_manifest)
        path = _resolve_path(record["path"], runtime_index_manifest, f"runtime index {key}")
        _assert_digest(path, record["sha256"], f"runtime index {key}")
        if not isinstance(record.get("rows"), int) or isinstance(record.get("rows"), bool):
            raise SystemExit(f"runtime index outputs.{key}.rows must be an integer")
    target_record = _required_record(
        runtime_index, "inputs", "target_clusters", runtime_index_manifest
    )
    target_path = _resolve_path(
        target_record["path"], runtime_index_manifest, "runtime target universe"
    )
    _assert_digest(target_path, target_record["sha256"], "runtime target universe")

    expected_recipe = evidence["recipe"]
    recipe_payload = _load_json(runtime_recipe, RECIPE_SCHEMA)
    if _sha256(runtime_recipe) != expected_recipe["sha256"]:
        raise SystemExit(
            "activity retrieval runtime recipe does not match the evaluated dev-selected recipe"
        )
    recipe_section = (
        "selected_recipe"
        if decision["operational_recipe_id"] == decision["selected_recipe_id"]
        else "baseline_recipe"
    )
    operational = recipe_payload.get(recipe_section)
    if not isinstance(operational, dict) or operational.get("recipe_id") != decision[
        "operational_recipe_id"
    ]:
        raise SystemExit("activity retrieval runtime recipe decision is inconsistent")
    return {
        "runtime_index_manifest": _artifact_record(runtime_index_manifest),
        "runtime_recipe": _artifact_record(runtime_recipe),
        "runtime_target_clusters": _artifact_record(target_path),
    }


def _runtime_decision(evaluation_decision: dict[str, Any]) -> dict[str, Any]:
    """Describe what the runtime binding authorizes without borrowing eval claims."""
    return {
        "claim_ready": False,
        "promotion_decision": "diagnostic_runtime_binding",
        "operational_recipe_id": evaluation_decision["operational_recipe_id"],
        "reason": (
            "the production runtime index is bound and integrity-checked but was "
            "not the frozen evaluation index; frozen evaluation metrics do not apply"
        ),
    }


def create_gate(
    *,
    benchmark_manifest: Path,
    index_manifest: Path,
    panels_manifest: Path,
    selection_manifest: Path,
    final_evaluation_manifest: Path,
    pocket_leakage_manifest: Path,
    out_gate: Path,
) -> dict[str, Any]:
    tmp = out_gate.with_suffix(out_gate.suffix + ".tmp")
    if out_gate.exists():
        out_gate.unlink()
    tmp.unlink(missing_ok=True)
    evidence = validate_manifest_contract(
        benchmark_manifest=benchmark_manifest,
        index_manifest=index_manifest,
        panels_manifest=panels_manifest,
        selection_manifest=selection_manifest,
        final_evaluation_manifest=final_evaluation_manifest,
        pocket_leakage_manifest=pocket_leakage_manifest,
    )
    payload = {
        "schema_version": GATE_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        **evidence,
    }
    out_gate.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(out_gate)
    return payload


def create_operational_gate(
    *,
    benchmark_manifest: Path,
    index_manifest: Path,
    panels_manifest: Path,
    selection_manifest: Path,
    final_evaluation_manifest: Path,
    pocket_leakage_manifest: Path,
    runtime_index_manifest: Path,
    runtime_recipe: Path,
    out_gate: Path,
) -> dict[str, Any]:
    tmp = out_gate.with_suffix(out_gate.suffix + ".tmp")
    if out_gate.exists():
        out_gate.unlink()
    tmp.unlink(missing_ok=True)
    evidence = validate_manifest_contract(
        benchmark_manifest=benchmark_manifest,
        index_manifest=index_manifest,
        panels_manifest=panels_manifest,
        selection_manifest=selection_manifest,
        final_evaluation_manifest=final_evaluation_manifest,
        pocket_leakage_manifest=pocket_leakage_manifest,
        require_promotion=False,
    )
    evaluation_decision = _operational_decision_from_evidence(evidence)
    runtime_evidence = _validate_runtime_operational_inputs(
        runtime_index_manifest=runtime_index_manifest,
        runtime_recipe=runtime_recipe,
        evidence=evidence,
        decision=evaluation_decision,
    )
    runtime_decision = _runtime_decision(evaluation_decision)
    payload = {
        "schema_version": OPERATIONAL_GATE_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "claim_gate_required_for_performance_claims": True,
        **runtime_decision,
        "evaluation_decision": evaluation_decision,
        "runtime_decision": runtime_decision,
        **runtime_evidence,
        **evidence,
    }
    out_gate.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(out_gate)
    return payload


def _gate_manifest_paths(
    payload: dict[str, Any], gate: Path, *, label: str
) -> dict[str, Path]:
    records = _required_mapping(payload, "manifests", gate)
    expected_keys = set(MANIFEST_SCHEMAS)
    if set(records) != expected_keys:
        raise SystemExit(f"{label} has an incomplete manifest set")
    paths: dict[str, Path] = {}
    for key in sorted(expected_keys):
        record = records[key]
        if not isinstance(record, dict):
            raise SystemExit(f"{label} manifest record is invalid: {key}")
        path = record.get("path")
        digest = record.get("sha256")
        if not isinstance(path, str) or not isinstance(digest, str):
            raise SystemExit(f"{label} manifest record is invalid: {key}")
        paths[key] = Path(path)
        _assert_digest(paths[key], digest, f"{label} {key}")
    return paths


def _validate_gate_evidence_records(
    payload: dict[str, Any], evidence: dict[str, Any], *, label: str
) -> None:
    for key in (
        "recipe",
        "target_cluster_manifest",
        "known_panel",
        "final_summary",
        "source_target_exclusions",
        "rcsb_holo_direct_contact_panel",
        "pocket_leakage_rows",
        "rcsb_contact_pocket_fragments",
        "prior_pocket_fragments",
    ):
        recorded = payload.get(key)
        current = evidence[key]
        if not isinstance(recorded, dict) or recorded.get("sha256") != current["sha256"]:
            raise SystemExit(f"{label} {key} is stale")


def check_gate(gate: Path) -> dict[str, Any]:
    payload = _load_json(gate, GATE_SCHEMA)
    if payload.get("status") != "pass":
        raise SystemExit("activity retrieval production gate status must be pass")
    paths = _gate_manifest_paths(
        payload,
        gate,
        label="activity retrieval production gate",
    )
    evidence = validate_manifest_contract(**paths)
    _validate_gate_evidence_records(
        payload,
        evidence,
        label="activity retrieval production gate",
    )
    return payload


def check_operational_gate(gate: Path) -> dict[str, Any]:
    payload = _load_json(gate, OPERATIONAL_GATE_SCHEMA)
    if payload.get("status") != "pass":
        raise SystemExit("activity retrieval operational gate status must be pass")
    if payload.get("claim_gate_required_for_performance_claims") is not True:
        raise SystemExit(
            "activity retrieval operational gate must preserve the separate claim gate"
        )
    paths = _gate_manifest_paths(
        payload,
        gate,
        label="activity retrieval operational gate",
    )
    evidence = validate_manifest_contract(**paths, require_promotion=False)
    evaluation_decision = _operational_decision_from_evidence(evidence)
    runtime_decision = _runtime_decision(evaluation_decision)
    if payload.get("evaluation_decision") != evaluation_decision:
        raise SystemExit("activity retrieval operational gate evaluation decision is stale")
    if payload.get("runtime_decision") != runtime_decision:
        raise SystemExit("activity retrieval operational gate runtime decision is stale")
    for key, value in runtime_decision.items():
        if payload.get(key) != value:
            raise SystemExit(f"activity retrieval operational gate {key} is stale")
    runtime_index_record = payload.get("runtime_index_manifest")
    runtime_recipe_record = payload.get("runtime_recipe")
    if not isinstance(runtime_index_record, dict) or not isinstance(runtime_recipe_record, dict):
        raise SystemExit(
            "activity retrieval operational gate lacks runtime index/recipe binding; regenerate it"
        )
    runtime_evidence = _validate_runtime_operational_inputs(
        runtime_index_manifest=Path(str(runtime_index_record.get("path") or "")),
        runtime_recipe=Path(str(runtime_recipe_record.get("path") or "")),
        evidence=evidence,
        decision=evaluation_decision,
    )
    for key, record in runtime_evidence.items():
        if not isinstance(payload.get(key), dict) or payload[key].get("sha256") != record["sha256"]:
            raise SystemExit(f"activity retrieval operational gate {key} is stale")
    _validate_gate_evidence_records(
        payload,
        evidence,
        label="activity retrieval operational gate",
    )
    return payload


def _add_create_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--benchmark-manifest", required=True, type=Path)
    parser.add_argument("--index-manifest", required=True, type=Path)
    parser.add_argument("--panels-manifest", required=True, type=Path)
    parser.add_argument("--selection-manifest", required=True, type=Path)
    parser.add_argument("--final-evaluation-manifest", required=True, type=Path)
    parser.add_argument("--pocket-leakage-manifest", required=True, type=Path)
    parser.add_argument("--out-gate", required=True, type=Path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    _add_create_arguments(create)
    create_operational = subparsers.add_parser("create-operational")
    _add_create_arguments(create_operational)
    create_operational.add_argument("--runtime-index-manifest", required=True, type=Path)
    create_operational.add_argument("--runtime-recipe", required=True, type=Path)
    check = subparsers.add_parser("check")
    check.add_argument("--gate", required=True, type=Path)
    check_operational = subparsers.add_parser("check-operational")
    check_operational.add_argument("--gate", required=True, type=Path)
    args = parser.parse_args()

    if args.command == "create":
        create_gate(
            benchmark_manifest=args.benchmark_manifest,
            index_manifest=args.index_manifest,
            panels_manifest=args.panels_manifest,
            selection_manifest=args.selection_manifest,
            final_evaluation_manifest=args.final_evaluation_manifest,
            pocket_leakage_manifest=args.pocket_leakage_manifest,
            out_gate=args.out_gate,
        )
        print("activity retrieval production claim gate passed")
    elif args.command == "create-operational":
        create_operational_gate(
            benchmark_manifest=args.benchmark_manifest,
            index_manifest=args.index_manifest,
            panels_manifest=args.panels_manifest,
            selection_manifest=args.selection_manifest,
            final_evaluation_manifest=args.final_evaluation_manifest,
            pocket_leakage_manifest=args.pocket_leakage_manifest,
            runtime_index_manifest=args.runtime_index_manifest,
            runtime_recipe=args.runtime_recipe,
            out_gate=args.out_gate,
        )
        print("activity retrieval operational gate passed")
    elif args.command == "check":
        check_gate(args.gate)
        print("activity retrieval production claim gate passed")
    else:
        check_operational_gate(args.gate)
        print("activity retrieval operational gate passed")


if __name__ == "__main__":
    main()
