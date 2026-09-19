#!/usr/bin/env python3
"""Evaluate a dev-selected sequence-transfer recipe as post-hoc diagnostics.

The 2025 temporal, known-skin, and integrated dual-cold panels were inspected
before this sequence model was designed. Results from this script therefore
cannot be presented as prospective confirmation or used to promote the recipe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "scripts"))

from activity_recovery_contracts import FROZEN_KNOWN_PANEL_CASES  # noqa: E402
from activity_retrieval_model import (  # noqa: E402
    Recipe,
    _artifact_record,
    _parse_truth_targets,
    _rank_map,
    _ranking_metrics_by_recipe,
    _read_json,
    _source_stratified_ranking_metrics_by_recipe,
    _validate_index_panel_binding,
    _validate_panel_manifest,
    _validate_query_identity,
    _validate_recipe_provenance,
    _write_csv_atomic,
    _write_json_atomic,
    _write_parquet_atomic,
    apply_platt,
    apply_recipe,
    calibration_metrics,
    load_calibration_panel,
    load_ranking_panel,
    load_reference_index,
    query_features,
    score_query_features,
    validate_frozen_known_panel,
)
from run_sequence_inductive_iteration import (  # noqa: E402
    SCHEMA_VERSION as SELECTION_SCHEMA,
    score_sequence_calibration_panel,
)
from sequence_inductive_retrieval import (  # noqa: E402
    SEQUENCE_BASELINE,
    SEQUENCE_CANDIDATES,
    SequenceTransferRecipe,
    TargetEmbeddingIndex,
    TargetNeighborIndex,
    build_target_neighbor_index,
    load_target_embedding_index,
    sequence_fill_scores,
    train_supported_targets,
)


SCHEMA_VERSION = "skinscout.sequence-inductive-posthoc-evaluation.v1"
EVALUATION_STATUS = "post_hoc_diagnostic_only"
POST_HOC_CONTRACT = {
    "evaluation_status": EVALUATION_STATUS,
    "panels_previously_inspected_before_sequence_model_design": True,
    "panels_used_for_sequence_recipe_selection": False,
    "panel_truth_passed_to_scorer": False,
    "prospective_confirmation_claim_permitted": False,
    "recipe_promotion_permitted_from_this_evaluation": False,
    "development_gate_may_be_retroactively_changed": False,
}
SOURCE_PROVENANCE_COLUMNS = {
    "panel_sources",
    "source_databases",
    "source_documents_json",
    "truth_target_panel_sources_json",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_safe_output_paths(
    *,
    input_paths: tuple[Path, ...],
    output_paths: tuple[Path, ...],
    destructive_dirs: tuple[Path, ...] = (),
) -> None:
    protected = {path.resolve() for path in input_paths}
    outputs = [path.resolve() for path in output_paths]
    destructive = outputs + [
        path.with_suffix(path.suffix + ".tmp").resolve() for path in output_paths
    ]
    if len(set(destructive)) != len(destructive):
        raise SystemExit("sequence post-hoc output paths must be distinct")
    aliases = sorted(str(path) for path in set(destructive) & protected)
    if aliases:
        raise SystemExit(
            "sequence post-hoc output path aliases a protected input: "
            + ", ".join(aliases)
        )
    for directory in (path.resolve() for path in destructive_dirs):
        contained = sorted(
            str(path) for path in protected if path == directory or directory in path.parents
        )
        if contained:
            raise SystemExit(
                "sequence post-hoc output directory contains a protected input: "
                + ", ".join(contained)
            )


def _resolve_record_path(
    record: dict[str, Any], manifest_path: Path, label: str
) -> Path:
    raw = str(record.get("path") or "").strip()
    if not raw:
        raise SystemExit(f"{label} provenance record is missing path")
    path = Path(raw)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _require_bound_record(
    record: object,
    *,
    path: Path,
    manifest_path: Path,
    label: str,
    rows: int | None = None,
) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise SystemExit(f"{label} provenance record is missing")
    if _resolve_record_path(record, manifest_path, label) != path.resolve():
        raise SystemExit(f"{label} path binding mismatch")
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"{label} artifact is missing or empty: {path}")
    if record.get("sha256") != _sha256(path):
        raise SystemExit(f"{label} sha256 mismatch")
    if rows is not None and int(record.get("rows", -1)) != rows:
        raise SystemExit(f"{label} row count mismatch")
    return record


def _sequence_recipe(value: object, label: str) -> SequenceTransferRecipe:
    if not isinstance(value, dict):
        raise SystemExit(f"sequence recipe artifact missing {label}")
    try:
        recipe = SequenceTransferRecipe(**value)
    except TypeError as exc:
        raise SystemExit(f"sequence recipe artifact has invalid {label}") from exc
    if not recipe.recipe_id.strip():
        raise SystemExit(f"sequence recipe artifact {label}.recipe_id must be nonblank")
    return recipe


def _validate_calibrator(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SystemExit(f"sequence recipe artifact missing {label} calibrator")
    try:
        slope = float(value["slope"])
        intercept = float(value["intercept"])
        rows = int(value["n_rows"])
        weight_sum = float(value["weight_sum"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"sequence recipe artifact has invalid {label} calibrator") from exc
    if (
        not math.isfinite(slope)
        or slope < 0.0
        or not math.isfinite(intercept)
        or rows < 2
        or not math.isfinite(weight_sum)
        or weight_sum <= 0.0
    ):
        raise SystemExit(f"sequence recipe artifact has invalid {label} calibrator")
    return value


def validate_sequence_selection(
    *,
    recipe_path: Path,
    selection_manifest_path: Path,
    index_manifest: Path,
    target_cluster_manifest: Path,
    panels_manifest: Path,
    base_recipe: Path,
    dev_cold_manifest: Path,
    dev_cold_calibration: Path,
    embedding_manifest: Path,
) -> tuple[
    dict[str, Any],
    SequenceTransferRecipe,
    SequenceTransferRecipe,
    dict[str, dict[str, dict[str, Any]]],
]:
    payload = _read_json(recipe_path, "sequence-inductive selected recipe")
    if payload.get("schema_version") != SELECTION_SCHEMA:
        raise SystemExit(f"sequence recipe schema must be {SELECTION_SCHEMA}")
    if payload.get("passes_dev_gate") is not True:
        raise SystemExit("sequence recipe did not pass its frozen development gate")
    if payload.get("selection_split") != "full_2024_sequence_target_cluster_cold_dev":
        raise SystemExit("sequence recipe selection split is not the frozen cold dev split")
    if payload.get("score_is_calibrated_probability") is not False:
        raise SystemExit("sequence retrieval score must not be labeled a probability")
    contract = payload.get("contract")
    required_contract = {
        "dev_only_selection": True,
        "test_input_used": False,
        "known_skin_input_used": False,
        "rcsb_input_used": False,
        "truth_target_assistance_to_scorer": False,
        "train_supported_target_scores_unchanged": True,
        "retrieval_score_is_probability": False,
    }
    if not isinstance(contract, dict) or any(
        contract.get(key) is not expected
        for key, expected in required_contract.items()
    ):
        raise SystemExit("sequence recipe no-assistance contract is invalid")

    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise SystemExit("sequence recipe artifact missing provenance")
    for key, path in (
        ("index_manifest", index_manifest),
        ("target_cluster_manifest", target_cluster_manifest),
        ("regular_panels_manifest", panels_manifest),
        ("base_recipe", base_recipe),
        ("sequence_dev_panel_manifest", dev_cold_manifest),
        ("sequence_dev_cold_calibration", dev_cold_calibration),
        ("embedding_manifest", embedding_manifest),
    ):
        _require_bound_record(
            provenance.get(key),
            path=path,
            manifest_path=recipe_path,
            label=f"sequence recipe {key}",
        )

    selected = _sequence_recipe(
        payload.get("selected_sequence_recipe"), "selected_sequence_recipe"
    )
    baseline = _sequence_recipe(
        payload.get("baseline_sequence_recipe"), "baseline_sequence_recipe"
    )
    if selected not in SEQUENCE_CANDIDATES:
        raise SystemExit("sequence recipe selected an unregistered candidate")
    if baseline != SEQUENCE_BASELINE:
        raise SystemExit("sequence recipe changed the frozen sequence baseline")
    calibrators = payload.get("calibrators")
    if not isinstance(calibrators, dict):
        raise SystemExit("sequence recipe artifact missing calibrators")
    validated_calibrators: dict[str, dict[str, dict[str, Any]]] = {}
    for label in ("baseline", "selected"):
        group = calibrators.get(label)
        if not isinstance(group, dict):
            raise SystemExit(f"sequence recipe artifact missing {label} calibrators")
        validated_calibrators[label] = {
            "aggregate": _validate_calibrator(
                group.get("aggregate"), f"{label}.aggregate"
            ),
            "unsupported_target": _validate_calibrator(
                group.get("unsupported_target"), f"{label}.unsupported_target"
            ),
        }

    selection_manifest = _read_json(
        selection_manifest_path, "sequence-inductive selection manifest"
    )
    if selection_manifest.get("schema_version") != SELECTION_SCHEMA:
        raise SystemExit("sequence selection manifest schema mismatch")
    if selection_manifest.get("passes_dev_gate") is not True:
        raise SystemExit("sequence selection manifest records a failed development gate")
    if selection_manifest.get("selected_sequence_recipe") != asdict(selected):
        raise SystemExit("sequence selection manifest selected recipe mismatch")
    manifest_inputs = selection_manifest.get("inputs")
    manifest_outputs = selection_manifest.get("outputs")
    if not isinstance(manifest_inputs, dict) or not isinstance(manifest_outputs, dict):
        raise SystemExit("sequence selection manifest artifact bindings are incomplete")
    for key, path in (
        ("index_manifest", index_manifest),
        ("target_cluster_manifest", target_cluster_manifest),
        ("panels_manifest", panels_manifest),
        ("base_recipe", base_recipe),
        ("dev_cold_manifest", dev_cold_manifest),
        ("dev_cold_calibration", dev_cold_calibration),
        ("embedding_manifest", embedding_manifest),
    ):
        _require_bound_record(
            manifest_inputs.get(key),
            path=path,
            manifest_path=selection_manifest_path,
            label=f"sequence selection {key}",
        )
    _require_bound_record(
        manifest_outputs.get("recipe.json"),
        path=recipe_path,
        manifest_path=selection_manifest_path,
        label="sequence selection recipe",
    )
    return payload, selected, baseline, validated_calibrators


def _provenance_fields(row: dict[str, Any]) -> dict[str, Any]:
    if not SOURCE_PROVENANCE_COLUMNS.issubset(row):
        return {"source_provenance_available": False}
    source_map = json.loads(str(row["truth_target_panel_sources_json"]))
    return {
        "source_provenance_available": True,
        "panel_sources": json.dumps(
            list(row["panel_sources"]), sort_keys=True, separators=(",", ":")
        ),
        "source_databases": json.dumps(
            list(row["source_databases"]), sort_keys=True, separators=(",", ":")
        ),
        "source_documents_json": str(row["source_documents_json"]),
        "truth_target_panel_sources_json": json.dumps(
            source_map, sort_keys=True, separators=(",", ":")
        ),
    }


def score_posthoc_ranking_panel(
    reference: Any,
    embeddings: TargetEmbeddingIndex,
    neighbors: TargetNeighborIndex,
    panel: pd.DataFrame,
    base_recipe: Recipe,
    sequence_recipes: tuple[SequenceTransferRecipe, SequenceTransferRecipe],
    *,
    exclude_reference_similarity: float,
    require_all_truth_targets_unsupported: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_to_index = {
        str(target): index for index, target in enumerate(reference.target_ids.tolist())
    }
    target_set = set(target_to_index)
    query_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    for row in panel.to_dict("records"):
        truth_targets = _parse_truth_targets(row["truth_targets"])
        unknown = sorted(set(truth_targets) - target_set)
        if unknown:
            raise SystemExit(
                f"post-hoc query {row['query_id']} has targets outside universe: {unknown}"
            )
        truth_indexes = np.asarray(
            [target_to_index[target] for target in truth_targets], dtype=np.int64
        )
        truth_support = embeddings.train_supported[truth_indexes]
        if require_all_truth_targets_unsupported and truth_support.any():
            raise SystemExit(
                f"dual-cold query {row['query_id']} contains a train-supported truth target"
            )
        query_fp = _validate_query_identity(row)
        features, stats = score_query_features(
            reference,
            query_fp,
            exclude_reference_similarity=exclude_reference_similarity,
        )
        base_scores = apply_recipe(features, base_recipe)
        provenance = _provenance_fields(row)
        source_map = (
            json.loads(provenance["truth_target_panel_sources_json"])
            if provenance["source_provenance_available"]
            else None
        )
        for sequence_recipe in sequence_recipes:
            scores, diagnostics = sequence_fill_scores(
                base_scores, embeddings, neighbors, sequence_recipe
            )
            ranks = _rank_map(reference.target_ids, scores)
            truth_ranks = [ranks[target] for target in truth_targets]
            query_rows.append(
                {
                    "query_id": row["query_id"],
                    "recipe_id": sequence_recipe.recipe_id,
                    "base_activity_recipe_id": base_recipe.recipe_id,
                    "n_truth_targets": len(truth_targets),
                    "n_supported_truth_targets": int(truth_support.sum()),
                    "n_unsupported_truth_targets": int((~truth_support).sum()),
                    "case_top10": int(any(rank <= 10 for rank in truth_ranks)),
                    "case_top30": int(any(rank <= 30 for rank in truth_ranks)),
                    "best_truth_rank": min(truth_ranks),
                    **stats,
                    **diagnostics,
                    **provenance,
                }
            )
            for target, target_index, supported in zip(
                truth_targets, truth_indexes, truth_support, strict=True
            ):
                rank = ranks[target]
                record = {
                    "query_id": row["query_id"],
                    "recipe_id": sequence_recipe.recipe_id,
                    "base_activity_recipe_id": base_recipe.recipe_id,
                    "target_id": target,
                    "target_train_supported": bool(supported),
                    "rank": rank,
                    "top10": int(rank <= 10),
                    "top30": int(rank <= 30),
                    "reciprocal_rank": 1.0 / rank,
                    "score": float(scores[target_index]),
                    **provenance,
                }
                if source_map is not None:
                    record["target_panel_sources_json"] = json.dumps(
                        source_map[target], sort_keys=True, separators=(",", ":")
                    )
                target_rows.append(record)
    return pd.DataFrame(query_rows), pd.DataFrame(target_rows)


def _write_known_ranking(
    *,
    out_dir: Path,
    case_id: str,
    target_ids: np.ndarray,
    scores: np.ndarray,
    sequence_recipe: SequenceTransferRecipe,
    sequence_recipe_artifact: Path,
    embedding_manifest: Path,
    stats: dict[str, Any],
    diagnostics: dict[str, Any],
    exclude_reference_similarity: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    order = np.lexsort((target_ids.astype(str), -scores))
    ranks = np.asarray(list(_rank_map(target_ids, scores).values()), dtype=float)
    ranking = pd.DataFrame(
        {
            "target_id": target_ids[order],
            "rank": ranks[order],
            "score": scores[order],
            "known_target_prior": np.zeros(len(order), dtype=float),
            "known_target_prior_norm": np.zeros(len(order), dtype=float),
            "scoring_method": sequence_recipe.recipe_id,
            "evidence_mode": "leave-query-out-sequence-transfer",
        }
    )
    ranking_path = out_dir / f"{case_id}.tsv"
    tmp = ranking_path.with_suffix(".tsv.tmp")
    ranking.to_csv(tmp, sep="\t", index=False)
    tmp.replace(ranking_path)
    _write_json_atomic(
        {
            "schema_version": "skinscout.sequence-inductive-known-ranking.v1",
            "created_at_utc": _utc_now(),
            "evaluation_status": EVALUATION_STATUS,
            "prospective_confirmation_claim_permitted": False,
            "case_id": case_id,
            "scoring_method": sequence_recipe.recipe_id,
            "score_is_calibrated_probability": False,
            "known_target_prior_nonzero_count": 0,
            "exclude_reference_similarity": exclude_reference_similarity,
            "ranking_sha256": _sha256(ranking_path),
            "inputs": {
                "sequence_recipe": _artifact_record(sequence_recipe_artifact),
                "embedding_manifest": _artifact_record(embedding_manifest),
            },
            "counts": {**stats, **diagnostics, "ranked_targets": len(ranking)},
        },
        ranking_path.with_suffix(".metadata.json"),
    )


def score_known_posthoc_panel(
    reference: Any,
    embeddings: TargetEmbeddingIndex,
    neighbors: TargetNeighborIndex,
    known_panel: Path,
    base_recipe: Recipe,
    sequence_recipes: tuple[SequenceTransferRecipe, SequenceTransferRecipe],
    *,
    exclude_reference_similarity: float,
    sequence_recipe_artifact: Path,
    embedding_manifest: Path,
    out_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_frozen_known_panel(known_panel)
    cases = pd.read_csv(known_panel)
    required = {"case_id", "smiles", "known_targets"}
    missing = sorted(required - set(cases.columns))
    if missing:
        raise SystemExit(f"known skin panel missing columns {missing}: {known_panel}")
    # The count lives in the panel contract, not here: v2 was 15 cases, v3 is 22.
    # Hardcoding it meant extending the panel failed inside the evaluator with a
    # message that read like the panel was corrupt.
    if len(cases) != len(FROZEN_KNOWN_PANEL_CASES) or cases["case_id"].duplicated().any():
        raise SystemExit(
            "known skin panel must remain the frozen "
            f"{len(FROZEN_KNOWN_PANEL_CASES)} unique cases"
        )
    target_to_index = {
        str(target): index for index, target in enumerate(reference.target_ids.tolist())
    }
    ranking_dirs = {
        recipe.recipe_id: out_root / f"known_rankings_{recipe.recipe_id}"
        for recipe in sequence_recipes
    }
    for directory in ranking_dirs.values():
        shutil.rmtree(directory, ignore_errors=True)
    query_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    for row in cases.to_dict("records"):
        case_id = str(row["case_id"]).strip()
        truth_targets = _parse_truth_targets(row["known_targets"])
        unknown = sorted(set(truth_targets) - set(target_to_index))
        if unknown:
            raise SystemExit(f"known case {case_id} targets outside universe: {unknown}")
        truth_indexes = np.asarray(
            [target_to_index[target] for target in truth_targets], dtype=np.int64
        )
        truth_support = embeddings.train_supported[truth_indexes]
        query_fp, _, _, _ = query_features(str(row["smiles"]))
        features, stats = score_query_features(
            reference,
            query_fp,
            exclude_reference_similarity=exclude_reference_similarity,
        )
        base_scores = apply_recipe(features, base_recipe)
        for sequence_recipe in sequence_recipes:
            scores, diagnostics = sequence_fill_scores(
                base_scores, embeddings, neighbors, sequence_recipe
            )
            ranks = _rank_map(reference.target_ids, scores)
            truth_ranks = [ranks[target] for target in truth_targets]
            query_rows.append(
                {
                    "query_id": case_id,
                    "recipe_id": sequence_recipe.recipe_id,
                    "base_activity_recipe_id": base_recipe.recipe_id,
                    "n_truth_targets": len(truth_targets),
                    "n_supported_truth_targets": int(truth_support.sum()),
                    "n_unsupported_truth_targets": int((~truth_support).sum()),
                    "case_top10": int(any(rank <= 10 for rank in truth_ranks)),
                    "case_top30": int(any(rank <= 30 for rank in truth_ranks)),
                    "best_truth_rank": min(truth_ranks),
                    **stats,
                    **diagnostics,
                }
            )
            for target, target_index, supported in zip(
                truth_targets, truth_indexes, truth_support, strict=True
            ):
                rank = ranks[target]
                target_rows.append(
                    {
                        "query_id": case_id,
                        "recipe_id": sequence_recipe.recipe_id,
                        "base_activity_recipe_id": base_recipe.recipe_id,
                        "target_id": target,
                        "target_train_supported": bool(supported),
                        "rank": rank,
                        "top10": int(rank <= 10),
                        "top30": int(rank <= 30),
                        "reciprocal_rank": 1.0 / rank,
                        "score": float(scores[target_index]),
                    }
                )
            _write_known_ranking(
                out_dir=ranking_dirs[sequence_recipe.recipe_id],
                case_id=case_id,
                target_ids=reference.target_ids,
                scores=scores,
                sequence_recipe=sequence_recipe,
                sequence_recipe_artifact=sequence_recipe_artifact,
                embedding_manifest=embedding_manifest,
                stats=stats,
                diagnostics=diagnostics,
                exclude_reference_similarity=exclude_reference_similarity,
            )
    return pd.DataFrame(query_rows), pd.DataFrame(target_rows)


def _metrics_for_probabilities(
    frame: pd.DataFrame, probabilities: np.ndarray
) -> dict[str, Any]:
    metrics: dict[str, Any] = calibration_metrics(
        probabilities,
        frame["label"].to_numpy(int),
        frame["sample_weight"].to_numpy(float),
    )
    metrics["n_positive"] = int(frame["label"].eq(1).sum())
    metrics["n_negative"] = int(frame["label"].eq(0).sum())
    by_family: dict[str, Any] = {}
    for family, group in frame.groupby("endpoint_family", sort=True):
        indexes = frame.index.get_indexer(group.index)
        by_family[str(family)] = calibration_metrics(
            probabilities[indexes],
            group["label"].to_numpy(int),
            group["sample_weight"].to_numpy(float),
        )
    metrics["by_endpoint_family"] = by_family
    return metrics


def evaluate_test_calibration(
    scored: pd.DataFrame,
    recipes: tuple[SequenceTransferRecipe, SequenceTransferRecipe],
    calibrators: dict[str, dict[str, dict[str, Any]]],
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    if scored.empty:
        raise SystemExit("post-hoc test calibration scores are empty")
    unsupported = scored.loc[~scored["target_train_supported"].astype(bool)].copy()
    if unsupported.empty:
        raise SystemExit("post-hoc test calibration contains no unsupported targets")
    result: dict[str, dict[str, Any]] = {}
    output = scored.copy()
    for label, recipe in zip(("baseline", "selected"), recipes, strict=True):
        score_column = f"score__{recipe.recipe_id}"
        aggregate_probabilities = apply_platt(
            output[score_column].to_numpy(float), calibrators[label]["aggregate"]
        )
        unsupported_probabilities = apply_platt(
            unsupported[score_column].to_numpy(float),
            calibrators[label]["unsupported_target"],
        )
        output[f"probability_aggregate__{recipe.recipe_id}"] = aggregate_probabilities
        cold_column = f"probability_unsupported_target__{recipe.recipe_id}"
        output[cold_column] = np.nan
        output.loc[unsupported.index, cold_column] = unsupported_probabilities
        result[recipe.recipe_id] = {
            "aggregate": _metrics_for_probabilities(output, aggregate_probabilities),
            "unsupported_target": _metrics_for_probabilities(
                unsupported, unsupported_probabilities
            ),
        }
    return result, output


def _ranking_by_support(target_rows: pd.DataFrame) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for recipe_id, recipe_group in target_rows.groupby("recipe_id", sort=True):
        strata: dict[str, Any] = {}
        for supported, label in ((True, "train_supported"), (False, "unsupported")):
            group = recipe_group.loc[
                recipe_group["target_train_supported"].astype(bool).eq(supported)
            ]
            strata[label] = (
                _ranking_metrics_by_recipe(group)[str(recipe_id)]
                if not group.empty
                else None
            )
        result[str(recipe_id)] = strata
    return result


def _ranking_delta(
    baseline: dict[str, Any], selected: dict[str, Any]
) -> dict[str, Any]:
    return {
        metric: float(selected[metric]) - float(baseline[metric])
        for metric in ("top10", "top30", "mrr")
    } | {
        "target_macro": {
            metric: float(selected["target_macro"][metric])
            - float(baseline["target_macro"][metric])
            for metric in ("top10", "top30", "mrr")
        }
    }


def _calibration_delta(
    baseline: dict[str, Any], selected: dict[str, Any]
) -> dict[str, float]:
    return {
        f"{metric}_improvement": float(baseline[metric]) - float(selected[metric])
        for metric in ("brier", "log_loss")
    }


def run(args: argparse.Namespace) -> None:
    out_dir = args.out_dir
    files = {
        "temporal_test_query_metrics.csv": out_dir / "temporal_test_query_metrics.csv",
        "temporal_test_target_metrics.csv": out_dir / "temporal_test_target_metrics.csv",
        "dual_cold_query_metrics.csv": out_dir / "dual_cold_query_metrics.csv",
        "dual_cold_target_metrics.csv": out_dir / "dual_cold_target_metrics.csv",
        "test_calibration_scores.parquet": out_dir / "test_calibration_scores.parquet",
        "known_query_metrics.csv": out_dir / "known_query_metrics.csv",
        "known_target_metrics.csv": out_dir / "known_target_metrics.csv",
        "summary.json": out_dir / "summary.json",
        "manifest.json": out_dir / "manifest.json",
    }
    input_paths = (
        args.ligands,
        args.edges,
        args.index_manifest,
        args.target_clusters,
        args.target_cluster_manifest,
        args.embeddings,
        args.embedding_manifest,
        args.panels_manifest,
        args.base_recipe,
        args.sequence_recipe,
        args.sequence_selection_manifest,
        args.dev_cold_manifest,
        args.dev_cold_calibration,
        args.test_ranking,
        args.dual_cold_ranking,
        args.test_calibration,
        args.known_panel,
    )
    potential_ranking_dirs = tuple(
        out_dir / f"known_rankings_{recipe.recipe_id}"
        for recipe in (SEQUENCE_BASELINE, *SEQUENCE_CANDIDATES)
    )
    _assert_safe_output_paths(
        input_paths=input_paths,
        output_paths=tuple(files.values()),
        destructive_dirs=potential_ranking_dirs,
    )
    ranking_dirs = list(potential_ranking_dirs)
    completed = False
    for path in files.values():
        path.unlink(missing_ok=True)
        path.with_suffix(path.suffix + ".tmp").unlink(missing_ok=True)
    for directory in ranking_dirs:
        shutil.rmtree(directory, ignore_errors=True)
    try:
        panel_manifest = _validate_panel_manifest(
            args.panels_manifest,
            {
                "test_ranking_queries.parquet": args.test_ranking,
                "dual_cold_ranking_queries.parquet": args.dual_cold_ranking,
                "test_calibration_pairs.parquet": args.test_calibration,
                "known_panel.csv": args.known_panel,
            },
        )
        reference, index_manifest, _target_manifest = load_reference_index(
            ligands_path=args.ligands,
            edges_path=args.edges,
            index_manifest_path=args.index_manifest,
            target_csv=args.target_clusters,
            target_manifest_path=args.target_cluster_manifest,
        )
        _validate_index_panel_binding(
            index_manifest_path=args.index_manifest,
            index_manifest=index_manifest,
            panel_manifest_path=args.panels_manifest,
            panel_manifest=panel_manifest,
        )
        base_payload = _read_json(args.base_recipe, "base activity recipe")
        base_recipe, _base_baseline, _base_calibrators = _validate_recipe_provenance(
            base_payload,
            recipe_path=args.base_recipe,
            index_manifest=args.index_manifest,
            target_cluster_manifest=args.target_cluster_manifest,
            panels_manifest=args.panels_manifest,
        )
        exclude_similarity = float(
            base_payload.get("exclude_reference_similarity", math.nan)
        )
        if not math.isfinite(exclude_similarity) or not (
            0.0 < exclude_similarity <= 1.0
        ):
            raise SystemExit("base recipe has invalid exclude_reference_similarity")
        sequence_payload, selected, baseline, calibrators = validate_sequence_selection(
            recipe_path=args.sequence_recipe,
            selection_manifest_path=args.sequence_selection_manifest,
            index_manifest=args.index_manifest,
            target_cluster_manifest=args.target_cluster_manifest,
            panels_manifest=args.panels_manifest,
            base_recipe=args.base_recipe,
            dev_cold_manifest=args.dev_cold_manifest,
            dev_cold_calibration=args.dev_cold_calibration,
            embedding_manifest=args.embedding_manifest,
        )
        if sequence_payload.get("base_activity_recipe") != asdict(base_recipe):
            raise SystemExit("sequence recipe base activity recipe mismatch")
        support = train_supported_targets(
            len(reference.target_ids),
            reference.edge_target,
            reference.edge_positive_count,
        )
        embeddings = load_target_embedding_index(
            embeddings_path=args.embeddings,
            embeddings_manifest_path=args.embedding_manifest,
            target_ids=reference.target_ids,
            target_clusters_path=args.target_clusters,
            target_cluster_manifest_path=args.target_cluster_manifest,
            train_supported=support,
        )
        neighbors = build_target_neighbor_index(
            embeddings, max_neighbors=selected.neighbor_top_k
        )
        recipes = (baseline, selected)
        test_panel = load_ranking_panel(args.test_ranking, "test")
        dual_panel = load_ranking_panel(
            args.dual_cold_ranking, "test", require_source_provenance=True
        )
        calibration_panel = load_calibration_panel(args.test_calibration, "test")
        temporal_source_provenance = SOURCE_PROVENANCE_COLUMNS.issubset(
            test_panel.columns
        )
        test_query, test_target = score_posthoc_ranking_panel(
            reference,
            embeddings,
            neighbors,
            test_panel,
            base_recipe,
            recipes,
            exclude_reference_similarity=exclude_similarity,
            require_all_truth_targets_unsupported=False,
        )
        dual_query, dual_target = score_posthoc_ranking_panel(
            reference,
            embeddings,
            neighbors,
            dual_panel,
            base_recipe,
            recipes,
            exclude_reference_similarity=exclude_similarity,
            require_all_truth_targets_unsupported=True,
        )
        calibration_scores = score_sequence_calibration_panel(
            reference,
            embeddings,
            neighbors,
            calibration_panel,
            base_recipe,
            exclude_reference_similarity=exclude_similarity,
            sequence_recipes=recipes,
        )
        keep_columns = [
            column
            for column in calibration_scores.columns
            if not column.startswith("score__")
            or column in {f"score__{recipe.recipe_id}" for recipe in recipes}
        ]
        calibration_scores = calibration_scores[keep_columns].copy()
        calibration, calibration_scores = evaluate_test_calibration(
            calibration_scores, recipes, calibrators
        )
        known_query, known_target = score_known_posthoc_panel(
            reference,
            embeddings,
            neighbors,
            args.known_panel,
            base_recipe,
            recipes,
            exclude_reference_similarity=exclude_similarity,
            sequence_recipe_artifact=args.sequence_recipe,
            embedding_manifest=args.embedding_manifest,
            out_root=out_dir,
        )
        test_metrics = _ranking_metrics_by_recipe(test_target)
        dual_metrics = _ranking_metrics_by_recipe(dual_target)
        known_metrics = _ranking_metrics_by_recipe(known_target)
        dual_source_metrics = _source_stratified_ranking_metrics_by_recipe(dual_target)
        baseline_id = baseline.recipe_id
        selected_id = selected.recipe_id
        summary = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": _utc_now(),
            "evaluation_status": EVALUATION_STATUS,
            "development_gate_passed": True,
            "selected_sequence_recipe": asdict(selected),
            "baseline_sequence_recipe": asdict(baseline),
            "base_activity_recipe": asdict(base_recipe),
            "ranking": {
                "temporal_test_2025": test_metrics,
                "temporal_test_2025_by_target_support": _ranking_by_support(test_target),
                "dual_cold_integrated": dual_metrics,
                "dual_cold_by_panel_source": dual_source_metrics,
                "known_skin_15_case": known_metrics,
                "known_skin_by_target_support": _ranking_by_support(known_target),
            },
            "calibration": {
                "temporal_test_2025": calibration,
                "labels": "measured binary events only; unmeasured pairs are not negatives",
                "application_boundary": {
                    "aggregate": "2024 regular-dev calibrator applied to all 2025 measured test pairs",
                    "unsupported_target": "2024 sequence-cold calibrator applied only to 2025 measured pairs whose target had no positive train support",
                },
            },
            "diagnostic_deltas": {
                "temporal_test_2025": _ranking_delta(
                    test_metrics[baseline_id], test_metrics[selected_id]
                ),
                "dual_cold_integrated": _ranking_delta(
                    dual_metrics[baseline_id], dual_metrics[selected_id]
                ),
                "known_skin_15_case": _ranking_delta(
                    known_metrics[baseline_id], known_metrics[selected_id]
                ),
                "calibration_aggregate": _calibration_delta(
                    calibration[baseline_id]["aggregate"],
                    calibration[selected_id]["aggregate"],
                ),
                "calibration_unsupported_target": _calibration_delta(
                    calibration[baseline_id]["unsupported_target"],
                    calibration[selected_id]["unsupported_target"],
                ),
            },
            "contract": dict(POST_HOC_CONTRACT),
            "source_provenance": {
                "temporal_test_2025_available": temporal_source_provenance,
                "dual_cold_available": True,
                "known_skin_case_metadata_available": True,
                "temporal_test_note": (
                    "source/document attribution columns are absent from the bound temporal ranking panel"
                    if not temporal_source_provenance
                    else "source/document attribution is carried into ranking metrics"
                ),
            },
            "claim_scope": {
                "retrieval_score": "not a calibrated probability",
                "target_assistance": "none",
                "known_target_prior": "zero for every known-skin ranking",
                "prospective_validation": "requires a data snapshot after 2026-08-05 that was not inspected during model development",
            },
        }
        _write_csv_atomic(test_query, files["temporal_test_query_metrics.csv"])
        _write_csv_atomic(test_target, files["temporal_test_target_metrics.csv"])
        _write_csv_atomic(dual_query, files["dual_cold_query_metrics.csv"])
        _write_csv_atomic(dual_target, files["dual_cold_target_metrics.csv"])
        _write_parquet_atomic(
            calibration_scores, files["test_calibration_scores.parquet"]
        )
        _write_csv_atomic(known_query, files["known_query_metrics.csv"])
        _write_csv_atomic(known_target, files["known_target_metrics.csv"])
        _write_json_atomic(summary, files["summary.json"])
        _write_json_atomic(
            {
                "schema_version": SCHEMA_VERSION,
                "created_at_utc": summary["created_at_utc"],
                "evaluation_status": EVALUATION_STATUS,
                "contract": dict(POST_HOC_CONTRACT),
                "source_provenance": summary["source_provenance"],
                "inputs": {
                    "sequence_recipe": _artifact_record(args.sequence_recipe),
                    "sequence_selection_manifest": _artifact_record(
                        args.sequence_selection_manifest
                    ),
                    "base_recipe": _artifact_record(args.base_recipe),
                    "index_manifest": _artifact_record(args.index_manifest),
                    "target_cluster_manifest": _artifact_record(
                        args.target_cluster_manifest
                    ),
                    "panels_manifest": _artifact_record(args.panels_manifest),
                    "embedding_manifest": _artifact_record(args.embedding_manifest),
                    "dev_cold_manifest": _artifact_record(args.dev_cold_manifest),
                    "dev_cold_calibration": _artifact_record(
                        args.dev_cold_calibration,
                        int(pq.ParquetFile(args.dev_cold_calibration).metadata.num_rows),
                    ),
                    "test_ranking": _artifact_record(args.test_ranking, len(test_panel)),
                    "dual_cold_ranking": _artifact_record(
                        args.dual_cold_ranking, len(dual_panel)
                    ),
                    "test_calibration": _artifact_record(
                        args.test_calibration, len(calibration_panel)
                    ),
                    "known_panel": _artifact_record(args.known_panel, 15),
                },
                "outputs": {
                    name: _artifact_record(
                        path,
                        (
                            int(pq.ParquetFile(path).metadata.num_rows)
                            if path.suffix == ".parquet"
                            else (
                                len(pd.read_csv(path))
                                if path.suffix == ".csv"
                                else None
                            )
                        ),
                    )
                    for name, path in files.items()
                    if name != "manifest.json"
                },
            },
            files["manifest.json"],
        )
        completed = True
    except BaseException:
        if not completed:
            for path in files.values():
                path.unlink(missing_ok=True)
                path.with_suffix(path.suffix + ".tmp").unlink(missing_ok=True)
            for directory in ranking_dirs:
                shutil.rmtree(directory, ignore_errors=True)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ligands", required=True, type=Path)
    parser.add_argument("--edges", required=True, type=Path)
    parser.add_argument("--index-manifest", required=True, type=Path)
    parser.add_argument("--target-clusters", required=True, type=Path)
    parser.add_argument("--target-cluster-manifest", required=True, type=Path)
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--embedding-manifest", required=True, type=Path)
    parser.add_argument("--panels-manifest", required=True, type=Path)
    parser.add_argument("--base-recipe", required=True, type=Path)
    parser.add_argument("--sequence-recipe", required=True, type=Path)
    parser.add_argument("--sequence-selection-manifest", required=True, type=Path)
    parser.add_argument("--dev-cold-manifest", required=True, type=Path)
    parser.add_argument("--dev-cold-calibration", required=True, type=Path)
    parser.add_argument("--test-ranking", required=True, type=Path)
    parser.add_argument("--dual-cold-ranking", required=True, type=Path)
    parser.add_argument("--test-calibration", required=True, type=Path)
    parser.add_argument("--known-panel", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
