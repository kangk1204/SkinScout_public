#!/usr/bin/env python3
"""Select an ESM-2 cold-target transfer recipe on bound 2024 dev data only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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

from activity_retrieval_model import (  # noqa: E402
    Recipe,
    _calibration_recipe_metrics,
    _calibration_split,
    _parse_truth_targets,
    _rank_map,
    _ranking_summary,
    _read_json,
    _validate_index_panel_binding,
    _validate_panel_manifest,
    _validate_query_identity,
    _validate_recipe_provenance,
    _write_csv_atomic,
    _write_json_atomic,
    _write_parquet_atomic,
    apply_recipe,
    load_calibration_panel,
    load_ranking_panel,
    load_reference_index,
    score_query_features,
)
from build_sequence_inductive_dev_panel import SCHEMA_VERSION as DEV_PANEL_SCHEMA  # noqa: E402
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


SCHEMA_VERSION = "skinscout.sequence-inductive-selection.v1"
MIN_UNSUPPORTED_CALIBRATION_ROWS = 20
MIN_UNSUPPORTED_CALIBRATION_PER_CLASS = 5
SEQUENCE_RECIPES = (SEQUENCE_BASELINE, *SEQUENCE_CANDIDATES)
REQUIRED_DEV_CONTRACT = {
    "split": "dev",
    "dev_only_model_selection": True,
    "no_test_input": True,
    "no_known_skin_input": True,
    "no_rcsb_input": True,
    "no_truth_assistance_artifact_input": True,
    "no_target_assistance_to_scorer": True,
    "truth_targets_collected_only_after_ligand_selection": True,
    "source_provenance_required": True,
    "all_cold_flags_true": True,
    "all_claimable_true": True,
    "cold_calibration_full_population": True,
    "cold_calibration_measured_labels_only": True,
}


def _contract_value_matches(actual: object, expected: object) -> bool:
    if isinstance(expected, bool):
        return actual is expected
    return actual == expected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
    }
    if rows is not None:
        record["rows"] = int(rows)
    return record


def _assert_safe_output_files(
    *, input_paths: tuple[Path, ...], output_paths: tuple[Path, ...]
) -> None:
    protected = {path.resolve() for path in input_paths}
    outputs = [path.resolve() for path in output_paths]
    destructive = outputs + [
        path.with_suffix(path.suffix + ".tmp").resolve() for path in output_paths
    ]
    if len(set(destructive)) != len(destructive):
        raise SystemExit("sequence selection output paths must be distinct")
    aliases = sorted(str(path) for path in set(destructive) & protected)
    if aliases:
        raise SystemExit(
            "sequence selection output path aliases a protected input: "
            + ", ".join(aliases)
        )


def _bound_path(record: dict[str, Any], manifest_path: Path, label: str) -> Path:
    raw = str(record.get("path") or "").strip()
    if not raw:
        raise SystemExit(f"{label} is missing path")
    path = Path(raw)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _require_bound_artifact(
    record: object,
    *,
    path: Path,
    manifest_path: Path,
    label: str,
    rows: int | None = None,
) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise SystemExit(f"{label} provenance record is missing")
    if _bound_path(record, manifest_path, label) != path.resolve():
        raise SystemExit(f"{label} path binding mismatch")
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"{label} artifact is missing or empty: {path}")
    if record.get("sha256") != _sha256(path):
        raise SystemExit(f"{label} sha256 mismatch")
    if rows is not None and int(record.get("rows", -1)) != rows:
        raise SystemExit(f"{label} row count mismatch")
    return record


def validate_sequence_dev_panel(
    *,
    panel_path: Path,
    calibration_path: Path,
    panel_manifest_path: Path,
    index_manifest_path: Path,
    index_manifest: dict[str, Any],
) -> dict[str, Any]:
    manifest = _read_json(panel_manifest_path, "sequence-inductive dev panel manifest")
    if manifest.get("schema_version") != DEV_PANEL_SCHEMA:
        raise SystemExit(f"sequence-inductive dev panel schema must be {DEV_PANEL_SCHEMA}")
    contract = manifest.get("contracts")
    if not isinstance(contract, dict) or any(
        not _contract_value_matches(contract.get(key), expected)
        for key, expected in REQUIRED_DEV_CONTRACT.items()
    ):
        raise SystemExit("sequence-inductive dev panel contract is invalid")
    selection = manifest.get("selection")
    if not isinstance(selection, dict):
        raise SystemExit("sequence-inductive dev panel selection metadata is missing")
    if int(selection.get("selected_ligands", -1)) != int(
        selection.get("candidate_positive_ligands", -2)
    ):
        raise SystemExit("sequence-inductive dev panel must contain the full eligible ligand set")

    inputs = manifest.get("inputs")
    outputs = manifest.get("outputs")
    index_inputs = index_manifest.get("inputs")
    if not isinstance(inputs, dict) or not isinstance(outputs, dict) or not isinstance(
        index_inputs, dict
    ):
        raise SystemExit("sequence panel and retrieval index manifests require inputs/outputs")
    benchmark_record = inputs.get("activity_benchmark_manifest")
    index_benchmark_record = index_inputs.get("benchmark_manifest")
    if not isinstance(benchmark_record, dict) or not isinstance(index_benchmark_record, dict):
        raise SystemExit("benchmark provenance is missing from sequence panel or index")
    benchmark_path = _bound_path(
        benchmark_record,
        panel_manifest_path,
        "sequence panel benchmark manifest",
    )
    index_benchmark_path = _bound_path(
        index_benchmark_record,
        index_manifest_path,
        "retrieval index benchmark manifest",
    )
    if benchmark_path != index_benchmark_path or benchmark_record.get(
        "sha256"
    ) != index_benchmark_record.get("sha256"):
        raise SystemExit("sequence dev panel and retrieval index use different benchmarks")
    _require_bound_artifact(
        benchmark_record,
        path=benchmark_path,
        manifest_path=panel_manifest_path,
        label="sequence panel benchmark manifest",
    )
    benchmark = _read_json(benchmark_path, "bound activity benchmark manifest")
    if benchmark.get("schema_version") != "activity_benchmark.v1":
        raise SystemExit("bound activity benchmark schema changed")

    train_path = benchmark_path.parent / "train.parquet"
    dev_path = benchmark_path.parent / "dev.parquet"
    split_counts = benchmark.get("splits")
    counts = split_counts.get("counts") if isinstance(split_counts, dict) else None
    hashes = benchmark.get("output_sha256")
    if not isinstance(counts, dict) or not isinstance(hashes, dict):
        raise SystemExit("bound activity benchmark split provenance is incomplete")
    train_rows = int(counts.get("train", -1))
    dev_rows = int(counts.get("dev", -1))
    train_record = _require_bound_artifact(
        inputs.get("train.parquet"),
        path=train_path,
        manifest_path=panel_manifest_path,
        label="sequence panel train parquet",
        rows=train_rows,
    )
    _require_bound_artifact(
        inputs.get("dev.parquet"),
        path=dev_path,
        manifest_path=panel_manifest_path,
        label="sequence panel dev parquet",
        rows=dev_rows,
    )
    if train_record.get("sha256") != hashes.get("train.parquet") or (
        inputs["dev.parquet"].get("sha256") != hashes.get("dev.parquet")
    ):
        raise SystemExit("sequence panel train/dev hashes differ from benchmark manifest")
    index_train = index_inputs.get("train_parquet")
    if not isinstance(index_train, dict) or (
        _bound_path(index_train, index_manifest_path, "retrieval index train parquet")
        != train_path.resolve()
        or index_train.get("sha256") != train_record.get("sha256")
        or int(index_train.get("rows", -1)) != train_rows
    ):
        raise SystemExit("sequence panel and retrieval index use different train artifacts")

    panel_rows = int(pq.ParquetFile(panel_path).metadata.num_rows)
    _require_bound_artifact(
        outputs.get("ranking.parquet"),
        path=panel_path,
        manifest_path=panel_manifest_path,
        label="sequence-inductive dev ranking parquet",
        rows=panel_rows,
    )
    if panel_rows != int(selection.get("selected_ligands", -1)):
        raise SystemExit("sequence dev panel rows differ from selected ligand count")
    calibration_rows = int(pq.ParquetFile(calibration_path).metadata.num_rows)
    _require_bound_artifact(
        outputs.get("calibration.parquet"),
        path=calibration_path,
        manifest_path=panel_manifest_path,
        label="sequence-inductive dev calibration parquet",
        rows=calibration_rows,
    )
    calibration_selection = selection.get("calibration")
    if not isinstance(calibration_selection, dict) or (
        calibration_selection.get("full_cold_population") is not True
        or int(calibration_selection.get("rows", -1)) != calibration_rows
        or int(calibration_selection.get("positive_rows", -1)) < 1
        or int(calibration_selection.get("negative_rows", -1)) < 1
    ):
        raise SystemExit("sequence-inductive cold calibration metadata is invalid")
    return manifest


def score_sequence_ranking_panel(
    reference: Any,
    embeddings: TargetEmbeddingIndex,
    neighbors: TargetNeighborIndex,
    panel: pd.DataFrame,
    base_recipe: Recipe,
    *,
    exclude_reference_similarity: float,
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
                f"sequence dev query {row['query_id']} has targets outside universe: {unknown}"
            )
        truth_indexes = np.asarray([target_to_index[target] for target in truth_targets])
        if embeddings.train_supported[truth_indexes].any():
            raise SystemExit(
                f"sequence dev query {row['query_id']} contains a train-supported truth target"
            )
        query_fp = _validate_query_identity(row)
        target_source_map = json.loads(str(row["truth_target_panel_sources_json"]))
        query_provenance = {
            "panel_sources": json.dumps(
                list(row["panel_sources"]), sort_keys=True, separators=(",", ":")
            ),
            "source_databases": json.dumps(
                list(row["source_databases"]), sort_keys=True, separators=(",", ":")
            ),
            "source_documents_json": str(row["source_documents_json"]),
            "truth_target_panel_sources_json": json.dumps(
                target_source_map, sort_keys=True, separators=(",", ":")
            ),
        }
        features, stats = score_query_features(
            reference,
            query_fp,
            exclude_reference_similarity=exclude_reference_similarity,
        )
        base_scores = apply_recipe(features, base_recipe)
        for sequence_recipe in SEQUENCE_RECIPES:
            scores, diagnostics = sequence_fill_scores(
                base_scores,
                embeddings,
                neighbors,
                sequence_recipe,
            )
            ranks = _rank_map(reference.target_ids, scores)
            truth_ranks = [ranks[target] for target in truth_targets]
            query_rows.append(
                {
                    "query_id": row["query_id"],
                    "recipe_id": sequence_recipe.recipe_id,
                    "n_truth_targets": len(truth_targets),
                    "case_top10": int(any(rank <= 10 for rank in truth_ranks)),
                    "case_top30": int(any(rank <= 30 for rank in truth_ranks)),
                    "best_truth_rank": min(truth_ranks),
                    **stats,
                    **diagnostics,
                    **query_provenance,
                }
            )
            for target, target_index in zip(truth_targets, truth_indexes, strict=True):
                rank = ranks[target]
                target_rows.append(
                    {
                        "query_id": row["query_id"],
                        "recipe_id": sequence_recipe.recipe_id,
                        "target_id": target,
                        "rank": rank,
                        "top10": int(rank <= 10),
                        "top30": int(rank <= 30),
                        "reciprocal_rank": 1.0 / rank,
                        "score": float(scores[target_index]),
                        **query_provenance,
                        "target_panel_sources_json": json.dumps(
                            target_source_map[target],
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    }
                )
    return pd.DataFrame(query_rows), pd.DataFrame(target_rows)


def score_sequence_calibration_panel(
    reference: Any,
    embeddings: TargetEmbeddingIndex,
    neighbors: TargetNeighborIndex,
    panel: pd.DataFrame,
    base_recipe: Recipe,
    *,
    exclude_reference_similarity: float,
    sequence_recipes: tuple[SequenceTransferRecipe, ...] = SEQUENCE_RECIPES,
) -> pd.DataFrame:
    target_to_index = {
        str(target): index for index, target in enumerate(reference.target_ids.tolist())
    }
    unknown = sorted(set(panel["uniprot"].astype(str)) - set(target_to_index))
    if unknown:
        raise SystemExit(f"sequence calibration targets outside universe: {unknown[:10]}")
    identities = (
        panel[
            [
                "query_id",
                "ligand_key",
                "standard_inchikey",
                "connectivity_key",
                "canonical_smiles",
                "standardization_route",
            ]
        ]
        .drop_duplicates()
        .sort_values("query_id", kind="mergesort")
    )
    if identities["query_id"].duplicated().any():
        raise SystemExit("calibration query_id maps to conflicting ligand identities")
    pair_groups = {key: group for key, group in panel.groupby("query_id", sort=True)}
    rows: list[dict[str, Any]] = []
    for identity in identities.to_dict("records"):
        query_id = identity["query_id"]
        query_fp = _validate_query_identity(identity)
        features, stats = score_query_features(
            reference,
            query_fp,
            exclude_reference_similarity=exclude_reference_similarity,
        )
        base_scores = apply_recipe(features, base_recipe)
        score_by_recipe = {
            recipe.recipe_id: sequence_fill_scores(
                base_scores, embeddings, neighbors, recipe
            )[0]
            for recipe in sequence_recipes
        }
        for pair in pair_groups[query_id].to_dict("records"):
            target_index = target_to_index[str(pair["uniprot"])]
            result = {
                "pair_id": pair["pair_id"],
                "query_id": query_id,
                "uniprot": pair["uniprot"],
                "endpoint_family": pair["endpoint_family"],
                "label": int(pair["label"]),
                "sample_weight": float(pair["sample_weight"]),
                "target_train_supported": bool(embeddings.train_supported[target_index]),
                **stats,
            }
            for recipe in sequence_recipes:
                result[f"score__{recipe.recipe_id}"] = float(
                    score_by_recipe[recipe.recipe_id][target_index]
                )
            rows.append(result)
    return pd.DataFrame(rows).sort_values("pair_id", kind="mergesort").reset_index(drop=True)


def sequence_metrics(
    target_rows: pd.DataFrame,
    calibration_scores: pd.DataFrame,
    cold_calibration_scores: pd.DataFrame,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    ranking: dict[str, dict[str, Any]] = {}
    for recipe_id, group in target_rows.groupby("recipe_id", sort=True):
        ranking[str(recipe_id)] = _ranking_summary(group)
    fit, evaluation = _calibration_split(calibration_scores)
    cold_fit, cold_evaluation = _calibration_split(cold_calibration_scores)
    if cold_calibration_scores["target_train_supported"].any():
        raise SystemExit("sequence-cold calibration contains a train-supported target")
    calibration: dict[str, dict[str, Any]] = {}
    calibrators: dict[str, dict[str, Any]] = {}
    for sequence_recipe in SEQUENCE_RECIPES:
        metrics, calibrator = _calibration_recipe_metrics(
            calibration_scores,
            Recipe(sequence_recipe.recipe_id),
            fit,
            evaluation,
        )
        cold_metrics, cold_calibrator = _calibration_recipe_metrics(
            cold_calibration_scores,
            Recipe(sequence_recipe.recipe_id),
            cold_fit,
            cold_evaluation,
        )
        cold_metrics["n_positive"] = int(cold_evaluation["label"].eq(1).sum())
        cold_metrics["n_negative"] = int(cold_evaluation["label"].eq(0).sum())
        metrics["unsupported_target"] = cold_metrics
        calibration[sequence_recipe.recipe_id] = metrics
        calibrators[sequence_recipe.recipe_id] = {
            "aggregate": calibrator,
            "unsupported_target": cold_calibrator,
        }
    return ranking, calibration, calibrators


def sequence_gate(
    baseline_ranking: dict[str, Any],
    candidate_ranking: dict[str, Any],
    baseline_calibration: dict[str, Any],
    candidate_calibration: dict[str, Any],
    *,
    tolerance: float = 1e-12,
) -> tuple[bool, list[str], float]:
    failures: list[str] = []
    gains: list[float] = []
    unsupported_baseline = baseline_calibration["unsupported_target"]
    for key, minimum in (
        ("n_rows", MIN_UNSUPPORTED_CALIBRATION_ROWS),
        ("n_positive", MIN_UNSUPPORTED_CALIBRATION_PER_CLASS),
        ("n_negative", MIN_UNSUPPORTED_CALIBRATION_PER_CLASS),
    ):
        observed = int(unsupported_baseline.get(key, -1))
        if observed < minimum:
            failures.append(
                f"unsupported_calibration_{key}={observed} must be >= {minimum}"
            )
    for scope, baseline, candidate in (
        ("micro", baseline_ranking, candidate_ranking),
        ("target_macro", baseline_ranking["target_macro"], candidate_ranking["target_macro"]),
    ):
        for metric in ("top10", "top30", "mrr"):
            gain = float(candidate[metric]) - float(baseline[metric])
            gains.append(gain)
            if gain <= tolerance:
                failures.append(
                    f"{scope}_{metric}_gain={gain:.12g} must be > {tolerance:g}"
                )
    for metric in ("brier", "log_loss"):
        gain = float(baseline_calibration[metric]) - float(candidate_calibration[metric])
        gains.append(gain)
        if gain <= tolerance:
            failures.append(f"{metric}_improvement={gain:.12g} must be > {tolerance:g}")
        unsupported_gain = float(
            baseline_calibration["unsupported_target"][metric]
        ) - float(candidate_calibration["unsupported_target"][metric])
        gains.append(unsupported_gain)
        if unsupported_gain <= tolerance:
            failures.append(
                f"unsupported_{metric}_improvement={unsupported_gain:.12g} "
                f"must be > {tolerance:g}"
            )
    return not failures, failures, float(sum(gains))


def _flatten_metric_row(
    recipe: SequenceTransferRecipe,
    ranking: dict[str, Any],
    calibration: dict[str, Any],
    *,
    passes: bool,
    score: float,
    failures: list[str],
) -> dict[str, Any]:
    return {
        **asdict(recipe),
        "ranking_top10": ranking["top10"],
        "ranking_top30": ranking["top30"],
        "ranking_mrr": ranking["mrr"],
        "target_macro_top10": ranking["target_macro"]["top10"],
        "target_macro_top30": ranking["target_macro"]["top30"],
        "target_macro_mrr": ranking["target_macro"]["mrr"],
        "calibration_brier": calibration["brier"],
        "calibration_log_loss": calibration["log_loss"],
        "calibration_ece10": calibration["ece10"],
        "unsupported_calibration_brier": calibration["unsupported_target"]["brier"],
        "unsupported_calibration_log_loss": calibration["unsupported_target"][
            "log_loss"
        ],
        "passes_dev_gate": passes,
        "dev_gate_score": score,
        "dev_gate_failures": ";".join(failures),
    }


def run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metric_paths = {
        "dev_query_metrics.csv": args.out_dir / "dev_query_metrics.csv",
        "dev_target_metrics.csv": args.out_dir / "dev_target_metrics.csv",
        "dev_calibration_scores.parquet": args.out_dir / "dev_calibration_scores.parquet",
        "dev_cold_calibration_scores.parquet": (
            args.out_dir / "dev_cold_calibration_scores.parquet"
        ),
        "dev_recipe_metrics.csv": args.out_dir / "dev_recipe_metrics.csv",
    }
    recipe_path = args.out_dir / "recipe.json"
    diagnostic_path = args.out_dir / "failed_recipe_diagnostic.json"
    manifest_path = args.out_dir / "manifest.json"
    cleanup = [*metric_paths.values(), recipe_path, diagnostic_path, manifest_path]
    _assert_safe_output_files(
        input_paths=(
            args.ligands,
            args.edges,
            args.index_manifest,
            args.target_clusters,
            args.target_cluster_manifest,
            args.embeddings,
            args.embedding_manifest,
            args.panels_manifest,
            args.dev_calibration,
            args.base_recipe,
            args.dev_cold_ranking,
            args.dev_cold_calibration,
            args.dev_cold_manifest,
        ),
        output_paths=tuple(cleanup),
    )
    preserve_diagnostics = False
    for path in cleanup:
        path.unlink(missing_ok=True)
        path.with_suffix(path.suffix + ".tmp").unlink(missing_ok=True)
    try:
        reference, index_manifest, _target_manifest = load_reference_index(
            ligands_path=args.ligands,
            edges_path=args.edges,
            index_manifest_path=args.index_manifest,
            target_csv=args.target_clusters,
            target_manifest_path=args.target_cluster_manifest,
        )
        regular_panel_manifest = _validate_panel_manifest(
            args.panels_manifest,
            {"dev_calibration_pairs.parquet": args.dev_calibration},
        )
        _validate_index_panel_binding(
            index_manifest_path=args.index_manifest,
            index_manifest=index_manifest,
            panel_manifest_path=args.panels_manifest,
            panel_manifest=regular_panel_manifest,
        )
        sequence_panel_manifest = validate_sequence_dev_panel(
            panel_path=args.dev_cold_ranking,
            calibration_path=args.dev_cold_calibration,
            panel_manifest_path=args.dev_cold_manifest,
            index_manifest_path=args.index_manifest,
            index_manifest=index_manifest,
        )
        base_payload = _read_json(args.base_recipe, "base activity recipe")
        base_recipe, _baseline_recipe, _base_calibrators = _validate_recipe_provenance(
            base_payload,
            recipe_path=args.base_recipe,
            index_manifest=args.index_manifest,
            target_cluster_manifest=args.target_cluster_manifest,
            panels_manifest=args.panels_manifest,
        )
        exclude_similarity = float(base_payload.get("exclude_reference_similarity", math.nan))
        if not math.isfinite(exclude_similarity) or not (0.0 < exclude_similarity <= 1.0):
            raise SystemExit("base recipe has invalid exclude_reference_similarity")
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
            embeddings,
            max_neighbors=max(recipe.neighbor_top_k for recipe in SEQUENCE_RECIPES),
        )
        ranking_panel = load_ranking_panel(
            args.dev_cold_ranking,
            "dev",
            require_source_provenance=True,
        )
        calibration_panel = load_calibration_panel(args.dev_calibration, "dev")
        cold_calibration_panel = load_calibration_panel(
            args.dev_cold_calibration,
            "dev",
        )
        query_rows, target_rows = score_sequence_ranking_panel(
            reference,
            embeddings,
            neighbors,
            ranking_panel,
            base_recipe,
            exclude_reference_similarity=exclude_similarity,
        )
        calibration_scores = score_sequence_calibration_panel(
            reference,
            embeddings,
            neighbors,
            calibration_panel,
            base_recipe,
            exclude_reference_similarity=exclude_similarity,
        )
        cold_calibration_scores = score_sequence_calibration_panel(
            reference,
            embeddings,
            neighbors,
            cold_calibration_panel,
            base_recipe,
            exclude_reference_similarity=exclude_similarity,
        )
        ranking_metrics, calibration_metrics, calibrators = sequence_metrics(
            target_rows,
            calibration_scores,
            cold_calibration_scores,
        )
        baseline_ranking = ranking_metrics[SEQUENCE_BASELINE.recipe_id]
        baseline_calibration = calibration_metrics[SEQUENCE_BASELINE.recipe_id]
        comparisons: list[dict[str, Any]] = []
        passing: list[tuple[float, SequenceTransferRecipe]] = []
        for candidate in SEQUENCE_CANDIDATES:
            passes, failures, score = sequence_gate(
                baseline_ranking,
                ranking_metrics[candidate.recipe_id],
                baseline_calibration,
                calibration_metrics[candidate.recipe_id],
            )
            comparisons.append(
                _flatten_metric_row(
                    candidate,
                    ranking_metrics[candidate.recipe_id],
                    calibration_metrics[candidate.recipe_id],
                    passes=passes,
                    score=score,
                    failures=failures,
                )
            )
            if passes:
                passing.append((score, candidate))
        selected = (
            sorted(passing, key=lambda item: (-item[0], item[1].recipe_id))[0][1]
            if passing
            else max(
                SEQUENCE_CANDIDATES,
                key=lambda candidate: next(
                    row["dev_gate_score"]
                    for row in comparisons
                    if row["recipe_id"] == candidate.recipe_id
                ),
            )
        )
        passes_dev_gate = bool(passing)
        baseline_row = _flatten_metric_row(
            SEQUENCE_BASELINE,
            baseline_ranking,
            baseline_calibration,
            passes=True,
            score=0.0,
            failures=["baseline"],
        )
        _write_csv_atomic(query_rows, metric_paths["dev_query_metrics.csv"])
        _write_csv_atomic(target_rows, metric_paths["dev_target_metrics.csv"])
        _write_parquet_atomic(
            calibration_scores,
            metric_paths["dev_calibration_scores.parquet"],
        )
        _write_parquet_atomic(
            cold_calibration_scores,
            metric_paths["dev_cold_calibration_scores.parquet"],
        )
        _write_csv_atomic(
            pd.DataFrame([baseline_row, *comparisons]),
            metric_paths["dev_recipe_metrics.csv"],
        )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "selection_split": "full_2024_sequence_target_cluster_cold_dev",
            "passes_dev_gate": passes_dev_gate,
            "base_activity_recipe": asdict(base_recipe),
            "selected_sequence_recipe": asdict(selected),
            "baseline_sequence_recipe": asdict(SEQUENCE_BASELINE),
            "exclude_reference_similarity": exclude_similarity,
            "score_is_calibrated_probability": False,
            "gate": {
                "ranking": "strict micro and target-macro Top10, Top30, and MRR improvement",
                "calibration": (
                    "strict aggregate and unsupported-target Brier and log-loss improvement"
                ),
                "unsupported_calibration_adequacy": {
                    "min_rows": MIN_UNSUPPORTED_CALIBRATION_ROWS,
                    "min_positive": MIN_UNSUPPORTED_CALIBRATION_PER_CLASS,
                    "min_negative": MIN_UNSUPPORTED_CALIBRATION_PER_CLASS,
                },
                "tolerance": 1e-12,
            },
            "calibrators": {
                "selected": calibrators[selected.recipe_id],
                "baseline": calibrators[SEQUENCE_BASELINE.recipe_id],
            },
            "dev_metrics": {
                "ranking": ranking_metrics,
                "calibration": calibration_metrics,
            },
            "contract": {
                "dev_only_selection": True,
                "test_input_used": False,
                "known_skin_input_used": False,
                "rcsb_input_used": False,
                "truth_target_assistance_to_scorer": False,
                "train_supported_target_scores_unchanged": True,
                "retrieval_score_is_probability": False,
            },
            "provenance": {
                "index_manifest": _artifact(args.index_manifest),
                "target_cluster_manifest": _artifact(args.target_cluster_manifest),
                "regular_panels_manifest": _artifact(args.panels_manifest),
                "base_recipe": _artifact(args.base_recipe),
                "sequence_dev_panel_manifest": _artifact(args.dev_cold_manifest),
                "sequence_dev_cold_calibration": _artifact(args.dev_cold_calibration),
                "embedding_manifest": _artifact(args.embedding_manifest),
                "sequence_dev_panel_schema": sequence_panel_manifest["schema_version"],
            },
        }
        output_recipe = recipe_path if passes_dev_gate else diagnostic_path
        _write_json_atomic(payload, output_recipe)
        output_records = {
            name: _artifact(
                path,
                rows=(
                    int(pq.ParquetFile(path).metadata.num_rows)
                    if path.suffix == ".parquet"
                    else max(0, sum(1 for _ in path.open("rb")) - 1)
                ),
            )
            for name, path in metric_paths.items()
        }
        output_records[output_recipe.name] = _artifact(output_recipe)
        _write_json_atomic(
            {
                "schema_version": SCHEMA_VERSION,
                "created_at_utc": payload["created_at_utc"],
                "passes_dev_gate": passes_dev_gate,
                "selected_sequence_recipe": asdict(selected),
                "inputs": {
                    "index_manifest": _artifact(args.index_manifest),
                    "target_cluster_manifest": _artifact(args.target_cluster_manifest),
                    "panels_manifest": _artifact(args.panels_manifest),
                    "base_recipe": _artifact(args.base_recipe),
                    "dev_cold_manifest": _artifact(args.dev_cold_manifest),
                    "dev_cold_calibration": _artifact(args.dev_cold_calibration),
                    "embedding_manifest": _artifact(args.embedding_manifest),
                },
                "outputs": output_records,
            },
            manifest_path,
        )
        if not passes_dev_gate and not args.allow_no_improvement:
            preserve_diagnostics = True
            raise SystemExit(
                "No sequence-transfer candidate strictly improved all registered dev metrics; "
                "diagnostic artifacts were written"
            )
    except BaseException:
        if not preserve_diagnostics:
            for path in cleanup:
                path.unlink(missing_ok=True)
                path.with_suffix(path.suffix + ".tmp").unlink(missing_ok=True)
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
    parser.add_argument("--dev-calibration", required=True, type=Path)
    parser.add_argument("--base-recipe", required=True, type=Path)
    parser.add_argument("--dev-cold-ranking", required=True, type=Path)
    parser.add_argument("--dev-cold-calibration", required=True, type=Path)
    parser.add_argument("--dev-cold-manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--allow-no-improvement", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
