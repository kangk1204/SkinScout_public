#!/usr/bin/env python3
"""Select and evaluate a train-only compound-to-target retrieval model.

The scorer never receives query truth targets. It standardizes the query,
removes every training reference at or above the registered similarity cutoff,
and ranks the complete provenance-bound target universe from pre-2024 measured
activity evidence. Calibration uses measured dev/test pairs only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem, DataStructs
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_activity_retrieval_index import (  # noqa: E402
    MORGAN_GENERATOR,
    _standardize_mol,
    _structure_ligand_key,
)
from activity_recovery_contracts import (  # noqa: E402
    ALPHAFOLD_HUMAN_V4_SOURCE,
    EXPECTED_ALPHAFOLD_BASE_TARGET_COUNT,
    EXPECTED_SCREENABLE_TARGET_COUNT,
    FROZEN_KNOWN_PANEL_CASES,
    validate_frozen_known_panel,
)
from activity_retrieval_scoring import (  # noqa: E402  (moved out of this file)
    BASELINE,
    CANDIDATES,
    RECIPES,
    ReferenceIndex,
    Recipe,
    _aggregate_edge_pairs,
    _load_target_universe,
    _read_json,
    _require_file,
    _target_max,
    _validate_registered_artifact,
    apply_recipe,
    average_tie_ranks as _average_tie_ranks,
    load_reference_index,
    query_features,
    scorable_target_mask,
    score_query_features,
)
from stage3_daina_zoete import fp_from_uint64_row  # noqa: E402


INDEX_SCHEMA = "skinscout.activity-retrieval-index.v4"
TARGET_SCHEMA = "skinscout.screenable-target-cluster-map.v2"
PANEL_SCHEMA = "skinscout.activity-recovery-panels.v5"
RECIPE_SCHEMA = "skinscout.activity-retrieval-recipe.v1"
SELECTION_SCHEMA = "skinscout.activity-retrieval-selection.v1"
EVALUATION_SCHEMA = "skinscout.activity-retrieval-evaluation.v1"
RANKING_KS = (10, 30)
SOURCE_STRATIFIED_PANEL_SOURCES = (
    "activity_quantitative",
    "rcsb_holo_direct_contact",
)




def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def _artifact_record(path: Path, rows: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
    }
    if rows is not None:
        record["rows"] = int(rows)
    return record


def _parse_truth_targets(value: object) -> list[str]:
    if isinstance(value, np.ndarray):
        values = value.tolist()
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    elif pd.isna(value):
        values = []
    else:
        text = str(value).strip()
        values = [part.strip() for part in text.split(";") if part.strip()]
    normalized = sorted({str(item).strip() for item in values if str(item).strip()})
    if not normalized:
        raise SystemExit("ranking query contains no truth targets")
    return normalized


def _parse_string_list(value: object, label: str) -> list[str]:
    if isinstance(value, np.ndarray):
        values = value.tolist()
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    elif pd.isna(value):
        values = []
    else:
        text = str(value).strip()
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{label} must be a list of strings") from exc
            if not isinstance(parsed, list):
                raise SystemExit(f"{label} must be a list of strings")
            values = parsed
        else:
            values = [part.strip() for part in text.split(";") if part.strip()]
    normalized = sorted({str(item).strip() for item in values if str(item).strip()})
    if not normalized:
        raise SystemExit(f"{label} must contain at least one source")
    return normalized


def _parse_json_value(value: object, label: str) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise SystemExit(f"{label} must be nonblank JSON")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{label} must be valid JSON") from exc
    if value is None or (isinstance(value, float) and pd.isna(value)):
        raise SystemExit(f"{label} must be nonblank JSON")
    return value


def _parse_truth_target_panel_sources(
    value: object,
    truth_targets: list[str],
    panel_sources: list[str],
    *,
    query_id: str,
) -> dict[str, list[str]]:
    parsed = _parse_json_value(
        value,
        f"ranking query {query_id} truth_target_panel_sources_json",
    )
    if not isinstance(parsed, dict):
        raise SystemExit(
            f"ranking query {query_id} truth_target_panel_sources_json must be an object"
        )
    truth_set = set(truth_targets)
    parsed_by_target = {str(key).strip(): value for key, value in parsed.items()}
    keys = set(parsed_by_target)
    if keys != truth_set:
        raise SystemExit(
            f"ranking query {query_id} truth_target_panel_sources_json must match truth_targets"
        )
    panel_source_set = set(panel_sources)
    allowed = set(SOURCE_STRATIFIED_PANEL_SOURCES)
    normalized: dict[str, list[str]] = {}
    for target in truth_targets:
        sources = _parse_string_list(
            parsed_by_target[target],
            f"ranking query {query_id} attribution for {target}",
        )
        unknown_panel_sources = sorted(set(sources) - panel_source_set)
        if unknown_panel_sources:
            raise SystemExit(
                f"ranking query {query_id} attribution for {target} is not in panel_sources: "
                f"{unknown_panel_sources}"
            )
        unsupported_sources = sorted(set(sources) - allowed)
        if unsupported_sources:
            raise SystemExit(
                f"ranking query {query_id} attribution for {target} has unsupported sources: "
                f"{unsupported_sources}"
            )
        normalized[target] = sources
    return normalized


def load_ranking_panel(
    path: Path,
    expected_split: str,
    *,
    require_source_provenance: bool = False,
) -> pd.DataFrame:
    _require_file(path, "ranking query panel")
    frame = pd.read_parquet(path)
    required = {
        "query_id",
        "ligand_key",
        "standard_inchikey",
        "connectivity_key",
        "canonical_smiles",
        "standardization_route",
        "truth_targets",
        "split",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"ranking query panel missing columns {missing}: {path}")
    if frame.empty:
        raise SystemExit(f"ranking query panel is empty: {path}")
    for column in (
        "query_id",
        "ligand_key",
        "standard_inchikey",
        "connectivity_key",
        "canonical_smiles",
        "standardization_route",
    ):
        frame[column] = frame[column].fillna("").astype(str).str.strip()
        if frame[column].eq("").any():
            raise SystemExit(f"ranking query panel contains blank {column}: {path}")
    if frame["query_id"].duplicated().any():
        raise SystemExit(f"ranking query panel contains duplicate query_id: {path}")
    if not frame["split"].astype(str).eq(expected_split).all():
        raise SystemExit(
            f"ranking query panel must contain only split={expected_split}: {path}"
        )
    frame["truth_targets"] = frame["truth_targets"].map(_parse_truth_targets)
    if "n_truth_targets" in frame.columns:
        declared = pd.to_numeric(frame["n_truth_targets"], errors="coerce")
        actual = frame["truth_targets"].map(len)
        if declared.isna().any() or not declared.astype(int).eq(actual).all():
            raise SystemExit(f"ranking query panel n_truth_targets mismatch: {path}")
    provenance_columns = {
        "panel_sources",
        "source_databases",
        "source_documents_json",
        "truth_target_panel_sources_json",
    }
    present_provenance = provenance_columns & set(frame.columns)
    if require_source_provenance and present_provenance != provenance_columns:
        missing = sorted(provenance_columns - set(frame.columns))
        raise SystemExit(
            f"ranking query panel missing source provenance columns {missing}: {path}"
        )
    if present_provenance:
        if present_provenance != provenance_columns:
            missing = sorted(provenance_columns - present_provenance)
            raise SystemExit(
                f"ranking query panel has incomplete source provenance columns {missing}: {path}"
            )
        normalized_panel_sources: list[list[str]] = []
        normalized_source_databases: list[list[str]] = []
        normalized_source_documents: list[Any] = []
        normalized_attribution: list[dict[str, list[str]]] = []
        for row in frame.to_dict("records"):
            query_id = str(row["query_id"])
            panel_sources = _parse_string_list(
                row["panel_sources"],
                f"ranking query {query_id} panel_sources",
            )
            unsupported = sorted(
                set(panel_sources) - set(SOURCE_STRATIFIED_PANEL_SOURCES)
            )
            if unsupported:
                raise SystemExit(
                    f"ranking query {query_id} has unsupported panel_sources: {unsupported}"
                )
            source_databases = _parse_string_list(
                row["source_databases"],
                f"ranking query {query_id} source_databases",
            )
            source_documents = _parse_json_value(
                row["source_documents_json"],
                f"ranking query {query_id} source_documents_json",
            )
            attribution = _parse_truth_target_panel_sources(
                row["truth_target_panel_sources_json"],
                row["truth_targets"],
                panel_sources,
                query_id=query_id,
            )
            normalized_panel_sources.append(panel_sources)
            normalized_source_databases.append(source_databases)
            normalized_source_documents.append(source_documents)
            normalized_attribution.append(attribution)
        frame["panel_sources"] = normalized_panel_sources
        frame["source_databases"] = normalized_source_databases
        frame["source_documents_json"] = [
            json.dumps(value, sort_keys=True, separators=(",", ":"))
            for value in normalized_source_documents
        ]
        frame["truth_target_panel_sources_json"] = [
            json.dumps(value, sort_keys=True, separators=(",", ":"))
            for value in normalized_attribution
        ]
    return frame.sort_values("query_id", kind="mergesort").reset_index(drop=True)


def load_calibration_panel(path: Path, expected_split: str) -> pd.DataFrame:
    _require_file(path, "calibration pair panel")
    frame = pd.read_parquet(path)
    required = {
        "pair_id",
        "query_id",
        "ligand_key",
        "standard_inchikey",
        "connectivity_key",
        "canonical_smiles",
        "standardization_route",
        "uniprot",
        "endpoint_family",
        "label",
        "sample_weight",
        "split",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"calibration panel missing columns {missing}: {path}")
    if frame.empty:
        raise SystemExit(f"calibration panel is empty: {path}")
    for column in (
        "pair_id",
        "query_id",
        "ligand_key",
        "standard_inchikey",
        "connectivity_key",
        "canonical_smiles",
        "standardization_route",
        "uniprot",
        "endpoint_family",
    ):
        frame[column] = frame[column].fillna("").astype(str).str.strip()
        if frame[column].eq("").any():
            raise SystemExit(f"calibration panel contains blank {column}: {path}")
    if frame["pair_id"].duplicated().any():
        raise SystemExit(f"calibration panel contains duplicate pair_id: {path}")
    if not frame["split"].astype(str).eq(expected_split).all():
        raise SystemExit(
            f"calibration panel must contain only split={expected_split}: {path}"
        )
    labels = pd.to_numeric(frame["label"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise SystemExit("calibration labels must be measured binary values 0 or 1")
    weights = pd.to_numeric(frame["sample_weight"], errors="coerce")
    if weights.isna().any() or not np.isfinite(weights.to_numpy(float)).all() or (
        weights <= 0
    ).any():
        raise SystemExit("calibration sample_weight must be finite and positive")
    allowed_families = {"direct_binding", "functional"}
    if not set(frame["endpoint_family"]).issubset(allowed_families):
        raise SystemExit("calibration endpoint_family must be direct_binding or functional")
    frame["label"] = labels.astype(int)
    frame["sample_weight"] = weights.astype(float)
    return frame.sort_values("pair_id", kind="mergesort").reset_index(drop=True)


def _validate_query_identity(row: dict[str, Any]) -> DataStructs.ExplicitBitVect:
    qfp, standard_inchikey, connectivity_key, canonical = query_features(
        row["canonical_smiles"]
    )
    expected_ligand_key, expected_route = _structure_ligand_key(
        standard_inchikey, canonical
    )
    if str(row["standard_inchikey"]) != standard_inchikey:
        raise SystemExit(
            f"query standard_inchikey does not match standardized SMILES for {row['query_id']}"
        )
    if str(row["ligand_key"]) != expected_ligand_key:
        raise SystemExit(
            f"query ligand_key does not match standardized SMILES for {row['query_id']}"
        )
    if str(row["connectivity_key"]) != connectivity_key:
        raise SystemExit(
            f"query connectivity_key does not match standardized SMILES for {row['query_id']}"
        )
    if str(row["canonical_smiles"]) != canonical:
        raise SystemExit(
            f"query canonical_smiles is not canonical for {row['query_id']}: "
            f"{row['canonical_smiles']!r} != {canonical!r}"
        )
    if str(row["standardization_route"]) != expected_route:
        raise SystemExit(
            f"query standardization_route does not match structure-key policy for {row['query_id']}"
        )
    return qfp


def _rank_map(target_ids: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    if len(target_ids) != len(scores):
        raise ValueError("target IDs and ranking scores must have equal length")
    ranks = _average_tie_ranks(scores)
    return {
        str(target): float(rank)
        for target, rank in zip(target_ids, ranks, strict=True)
    }


def _coverage_summary(target_rows: pd.DataFrame) -> dict[str, Any]:
    """Separate what the model could not score from what it scored and missed.

    Without this a truth pair the model never had evidence for reads exactly
    like one it ranked last, and top10=0 cannot be attributed to either.
    """
    if "in_scorable_universe" not in target_rows.columns:
        return {
            "measured": False,
            "reason": "ranking rows predate scorable-universe accounting",
        }
    in_universe = target_rows["in_scorable_universe"].astype(bool)
    covered = target_rows.loc[in_universe]
    summary: dict[str, Any] = {
        "measured": True,
        "truth_pairs": int(len(target_rows)),
        "truth_pairs_in_scorable_universe": int(in_universe.sum()),
        "truth_pairs_outside_scorable_universe": int((~in_universe).sum()),
        "truth_pair_coverage": float(in_universe.mean()) if len(target_rows) else 0.0,
        # All-truth metrics are the headline numbers, but with zero scorable
        # truth they describe the tied zero-score tail, not retrieval ability.
        "all_truth": {
            "top10": float((target_rows["rank"] <= 10).mean()),
            "top30": float((target_rows["rank"] <= 30).mean()),
            "mrr": float((1.0 / target_rows["rank"].astype(float)).mean()),
        },
    }
    if "scorable_universe_size" in target_rows.columns:
        sizes = sorted({int(value) for value in target_rows["scorable_universe_size"]})
        summary["scorable_universe_size"] = sizes[0] if len(sizes) == 1 else sizes
    if "target_universe_size" in target_rows.columns:
        sizes = sorted({int(value) for value in target_rows["target_universe_size"]})
        summary["target_universe_size"] = sizes[0] if len(sizes) == 1 else sizes
    # Ranking quality restricted to pairs the model could actually score.
    summary["in_universe"] = (
        {
            "top10": float((covered["rank"] <= 10).mean()),
            "top30": float((covered["rank"] <= 30).mean()),
            "mrr": float((1.0 / covered["rank"].astype(float)).mean()),
        }
        if not covered.empty
        else {"top10": None, "top30": None, "mrr": None}
    )
    return summary


def _ranking_summary(target_rows: pd.DataFrame) -> dict[str, Any]:
    if target_rows.empty:
        raise ValueError("ranking target rows must be nonempty")
    target_metrics = target_rows.groupby("target_id", sort=True).agg(
        top10=("top10", "mean"),
        top30=("top30", "mean"),
        mrr=("reciprocal_rank", "mean"),
        truth_pairs=("query_id", "size"),
    )
    target_fractions = target_metrics["truth_pairs"] / len(target_rows)
    return {
        "coverage": _coverage_summary(target_rows),
        "n_queries": int(target_rows["query_id"].nunique()),
        "n_truth_pairs": int(len(target_rows)),
        "n_truth_targets": int(len(target_metrics)),
        "top10": float((target_rows["rank"] <= 10).mean()),
        "top30": float((target_rows["rank"] <= 30).mean()),
        "mrr": float((1.0 / target_rows["rank"].astype(float)).mean()),
        "max_truth_pair_target_fraction": float(target_fractions.max()),
        "effective_target_count": float(
            1.0 / float((target_fractions * target_fractions).sum())
        ),
        "target_macro": {
            "top10": float(target_metrics["top10"].mean()),
            "top30": float(target_metrics["top30"].mean()),
            "mrr": float(target_metrics["mrr"].mean()),
        },
    }


def _cold_start_adequacy(panel_manifest: dict[str, Any]) -> dict[str, Any]:
    selection = panel_manifest.get("selection")
    ranking = selection.get("ranking_queries") if isinstance(selection, dict) else None
    dual = ranking.get("dual_cold") if isinstance(ranking, dict) else None
    adequacy = dual.get("adequacy") if isinstance(dual, dict) else None
    if not isinstance(adequacy, dict) or not isinstance(adequacy.get("checks"), dict):
        raise SystemExit("activity recovery panel manifest missing dual-cold adequacy audit")
    return adequacy


def _cold_start_adequacy_record(panel_manifest: dict[str, Any]) -> dict[str, Any]:
    """Build the dual-cold panel adequacy record.

    Recipe selection deliberately does not gate on this audit; the frozen test
    gate does. Emitting the same record from both entry points keeps
    ``used_for_recipe_selection=False`` auditable from the dev artifacts
    instead of computing the verdict there and dropping it.
    """
    adequacy = _cold_start_adequacy(panel_manifest)
    passes = bool(
        panel_manifest.get("passes_panel_adequacy_gate") is True
        and adequacy.get("passes") is True
        and all(value is True for value in adequacy["checks"].values())
    )
    return {
        "passes_panel_adequacy_gate": passes,
        "adequacy": adequacy,
        "used_for_recipe_selection": False,
    }


def _cold_start_claim_record(
    adequacy_record: dict[str, Any],
    coverage: dict[str, Any],
) -> dict[str, Any]:
    """Combine panel adequacy with truth scoreability for the cold-start claim.

    Adequacy counts queries, truth targets and source documents, but it cannot
    see whether the retrieval model carries an activity edge for any frozen
    truth target. With zero scorable truth the reported ranks come from the
    tied zero-score tail, so a large query/target count must not make the
    cold-start scope claimable.
    """
    adequacy_passes = adequacy_record.get("passes_panel_adequacy_gate") is True
    measured = coverage.get("measured") is True
    truth_pairs = coverage.get("truth_pairs")
    in_universe_pairs = coverage.get("truth_pairs_in_scorable_universe")
    in_universe_metrics = coverage.get("in_universe")
    counts_valid = (
        isinstance(truth_pairs, int)
        and not isinstance(truth_pairs, bool)
        and truth_pairs >= 1
        and isinstance(in_universe_pairs, int)
        and not isinstance(in_universe_pairs, bool)
        and 1 <= in_universe_pairs <= truth_pairs
    )
    metrics_valid = isinstance(in_universe_metrics, dict) and all(
        in_universe_metrics.get(key) is not None
        for key in ("top10", "top30", "mrr")
    )
    coverage_passes = bool(measured and counts_valid and metrics_valid)
    return {
        "claimable": bool(adequacy_passes and coverage_passes),
        "passes_panel_adequacy_gate": adequacy_passes,
        "passes_truth_universe_coverage": coverage_passes,
        "coverage": coverage,
        "policy": (
            "The cold-start claim additionally requires at least one frozen truth "
            "pair inside the retrieval model's scorable universe with reported "
            "in-universe ranking metrics; sample size alone is not sufficient."
        ),
    }


def score_ranking_panel(
    reference: ReferenceIndex,
    panel: pd.DataFrame,
    recipes: Iterable[Recipe],
    *,
    exclude_reference_similarity: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_set = set(reference.target_ids.tolist())
    scorable = scorable_target_mask(reference)
    scorable_ids = set(reference.target_ids[scorable].tolist())
    scorable_universe_size = int(scorable.sum())
    target_universe_size = int(len(reference.target_ids))
    recipe_list = list(recipes)
    query_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    provenance_columns = {
        "panel_sources",
        "source_databases",
        "source_documents_json",
        "truth_target_panel_sources_json",
    }
    has_source_provenance = provenance_columns.issubset(set(panel.columns))
    for row in panel.to_dict("records"):
        truth_targets = _parse_truth_targets(row["truth_targets"])
        query_provenance: dict[str, Any] = {}
        target_source_map: dict[str, list[str]] = {}
        if has_source_provenance:
            target_source_map = json.loads(str(row["truth_target_panel_sources_json"]))
            query_provenance = {
                "panel_sources": json.dumps(
                    list(row["panel_sources"]),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "source_databases": json.dumps(
                    list(row["source_databases"]),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "source_documents_json": str(row["source_documents_json"]),
                "truth_target_panel_sources_json": json.dumps(
                    target_source_map,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        unknown = sorted(set(truth_targets) - target_set)
        if unknown:
            raise SystemExit(
                f"ranking query {row['query_id']} truth targets outside universe: {unknown}"
            )
        qfp = _validate_query_identity(row)
        features, stats = score_query_features(
            reference,
            qfp,
            exclude_reference_similarity=exclude_reference_similarity,
        )
        for recipe in recipe_list:
            scores = apply_recipe(features, recipe)
            ranks = _rank_map(reference.target_ids, scores)
            truth_ranks = [ranks[target] for target in truth_targets]
            query_rows.append(
                {
                    "query_id": row["query_id"],
                    "recipe_id": recipe.recipe_id,
                    "n_truth_targets": len(truth_targets),
                    "case_top10": int(any(rank <= 10 for rank in truth_ranks)),
                    "case_top30": int(any(rank <= 30 for rank in truth_ranks)),
                    "best_truth_rank": min(truth_ranks),
                    "n_truth_targets_in_scorable_universe": int(
                        sum(target in scorable_ids for target in truth_targets)
                    ),
                    "scorable_universe_size": scorable_universe_size,
                    "target_universe_size": target_universe_size,
                    **stats,
                    **query_provenance,
                }
            )
            for target in truth_targets:
                rank = ranks[target]
                target_provenance = (
                    {
                        **query_provenance,
                        "target_panel_sources_json": json.dumps(
                            target_source_map[target],
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    }
                    if has_source_provenance
                    else {}
                )
                target_rows.append(
                    {
                        "query_id": row["query_id"],
                        "recipe_id": recipe.recipe_id,
                        "target_id": target,
                        "rank": rank,
                        "top10": int(rank <= 10),
                        "top30": int(rank <= 30),
                        "reciprocal_rank": 1.0 / rank,
                        "score": float(scores[
                            int(np.searchsorted(reference.target_ids, target))
                        ]),
                        "in_scorable_universe": bool(target in scorable_ids),
                        "scorable_universe_size": scorable_universe_size,
                        "target_universe_size": target_universe_size,
                        **target_provenance,
                    }
                )
    return pd.DataFrame(query_rows), pd.DataFrame(target_rows)


def score_calibration_panel(
    reference: ReferenceIndex,
    panel: pd.DataFrame,
    recipes: Iterable[Recipe],
    *,
    exclude_reference_similarity: float,
) -> pd.DataFrame:
    recipe_list = list(recipes)
    target_to_index = {
        target: index for index, target in enumerate(reference.target_ids.tolist())
    }
    unknown = sorted(set(panel["uniprot"]) - set(target_to_index))
    if unknown:
        raise SystemExit(f"calibration targets outside target universe: {unknown[:10]}")
    rows: list[dict[str, Any]] = []
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
    identity_lookup = identities.set_index("query_id").to_dict("index")
    pair_groups = {key: group for key, group in panel.groupby("query_id", sort=True)}
    for query_id, identity in identity_lookup.items():
        qrow = {"query_id": query_id, **identity}
        qfp = _validate_query_identity(qrow)
        features, stats = score_query_features(
            reference,
            qfp,
            exclude_reference_similarity=exclude_reference_similarity,
        )
        score_by_recipe = {
            recipe.recipe_id: apply_recipe(features, recipe) for recipe in recipe_list
        }
        for pair in pair_groups[query_id].to_dict("records"):
            target_index = target_to_index[pair["uniprot"]]
            result = {
                "pair_id": pair["pair_id"],
                "query_id": query_id,
                "uniprot": pair["uniprot"],
                "endpoint_family": pair["endpoint_family"],
                "label": int(pair["label"]),
                "sample_weight": float(pair["sample_weight"]),
                **stats,
            }
            for recipe in recipe_list:
                result[f"score__{recipe.recipe_id}"] = float(
                    score_by_recipe[recipe.recipe_id][target_index]
                )
            rows.append(result)
    return pd.DataFrame(rows).sort_values("pair_id", kind="mergesort").reset_index(drop=True)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_platt(
    scores: np.ndarray, labels: np.ndarray, weights: np.ndarray
) -> dict[str, float]:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if scores.size == 0 or not (scores.size == labels.size == weights.size):
        raise ValueError("Platt inputs must be nonempty and have equal lengths")
    if not set(np.unique(labels)).issubset({0.0, 1.0}) or len(np.unique(labels)) != 2:
        raise ValueError("Platt fitting requires measured examples from both classes")
    if not np.isfinite(scores).all() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Platt scores/weights must be finite and weights positive")
    prevalence = float(np.average(labels, weights=weights))
    intercept = math.log(prevalence / (1.0 - prevalence))

    def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
        slope, bias = float(params[0]), float(params[1])
        probabilities = _sigmoid(slope * scores + bias)
        eps = 1e-12
        loss = -np.average(
            labels * np.log(np.clip(probabilities, eps, 1.0))
            + (1.0 - labels) * np.log(np.clip(1.0 - probabilities, eps, 1.0)),
            weights=weights,
        )
        loss += 1e-6 * slope * slope
        residual = probabilities - labels
        norm = weights.sum()
        gradient = np.array(
            [
                float(np.sum(weights * residual * scores) / norm + 2e-6 * slope),
                float(np.sum(weights * residual) / norm),
            ]
        )
        return float(loss), gradient

    result = minimize(
        lambda params: objective(params),
        np.array([1.0, intercept], dtype=float),
        method="L-BFGS-B",
        jac=True,
        bounds=[(0.0, 100.0), (-50.0, 50.0)],
    )
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError(f"Platt optimization failed: {result.message}")
    return {
        "slope": float(result.x[0]),
        "intercept": float(result.x[1]),
        "weighted_prevalence": prevalence,
        "n_rows": int(scores.size),
        "weight_sum": float(weights.sum()),
    }


def apply_platt(scores: np.ndarray, calibrator: dict[str, Any]) -> np.ndarray:
    return _sigmoid(
        float(calibrator["slope"]) * np.asarray(scores, dtype=float)
        + float(calibrator["intercept"])
    )


def calibration_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    bins: int = 10,
) -> dict[str, float | int]:
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if probabilities.size == 0 or not (
        probabilities.size == labels.size == weights.size
    ):
        raise ValueError("calibration metric inputs must be nonempty and aligned")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (
        probabilities > 1
    ).any():
        raise ValueError("calibrated probabilities must be finite and in [0,1]")
    eps = 1e-12
    brier = float(np.average((probabilities - labels) ** 2, weights=weights))
    log_loss = float(
        -np.average(
            labels * np.log(np.clip(probabilities, eps, 1.0))
            + (1.0 - labels) * np.log(np.clip(1.0 - probabilities, eps, 1.0)),
            weights=weights,
        )
    )
    bin_ids = np.minimum((probabilities * bins).astype(int), bins - 1)
    ece = 0.0
    total_weight = float(weights.sum())
    for bin_id in range(bins):
        mask = bin_ids == bin_id
        if not mask.any():
            continue
        bin_weight = float(weights[mask].sum())
        confidence = float(np.average(probabilities[mask], weights=weights[mask]))
        accuracy = float(np.average(labels[mask], weights=weights[mask]))
        ece += bin_weight / total_weight * abs(confidence - accuracy)
    return {
        "n_rows": int(probabilities.size),
        "weight_sum": total_weight,
        "brier": brier,
        "log_loss": log_loss,
        "ece10": float(ece),
    }


def _calibration_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit_indexes: list[int] = []
    eval_indexes: list[int] = []
    for _, group in frame.groupby(["endpoint_family", "label"], sort=True):
        ordered = group.assign(
            _hash=group["pair_id"].map(
                lambda value: hashlib.sha256(str(value).encode()).hexdigest()
            )
        ).sort_values(["_hash", "pair_id"], kind="mergesort")
        for offset, index in enumerate(ordered.index):
            (fit_indexes if offset % 2 == 0 else eval_indexes).append(int(index))
    fit = frame.loc[sorted(fit_indexes)].copy()
    evaluation = frame.loc[sorted(eval_indexes)].copy()
    for label, subset in (("fit", fit), ("evaluation", evaluation)):
        if len(subset) < 2 or set(subset["label"]) != {0, 1}:
            raise SystemExit(f"deterministic calibration {label} split lacks both classes")
    return fit, evaluation


def _calibration_recipe_metrics(
    scored: pd.DataFrame, recipe: Recipe, fit: pd.DataFrame, evaluation: pd.DataFrame
) -> tuple[dict[str, Any], dict[str, Any]]:
    column = f"score__{recipe.recipe_id}"
    calibrator = fit_platt(
        fit[column].to_numpy(float),
        fit["label"].to_numpy(int),
        fit["sample_weight"].to_numpy(float),
    )
    probabilities = apply_platt(evaluation[column].to_numpy(float), calibrator)
    metrics: dict[str, Any] = calibration_metrics(
        probabilities,
        evaluation["label"].to_numpy(int),
        evaluation["sample_weight"].to_numpy(float),
    )
    by_family: dict[str, Any] = {}
    for family, group in evaluation.groupby("endpoint_family", sort=True):
        indexes = group.index
        family_probabilities = apply_platt(
            evaluation.loc[indexes, column].to_numpy(float), calibrator
        )
        by_family[str(family)] = calibration_metrics(
            family_probabilities,
            evaluation.loc[indexes, "label"].to_numpy(int),
            evaluation.loc[indexes, "sample_weight"].to_numpy(float),
        )
    metrics["by_endpoint_family"] = by_family
    final_calibrator = fit_platt(
        scored[column].to_numpy(float),
        scored["label"].to_numpy(int),
        scored["sample_weight"].to_numpy(float),
    )
    return metrics, final_calibrator


def _validate_panel_manifest(
    manifest_path: Path,
    artifacts: dict[str, Path],
) -> dict[str, Any]:
    manifest = _read_json(manifest_path, "activity recovery panel manifest")
    if manifest.get("schema_version") != PANEL_SCHEMA:
        raise SystemExit("activity recovery panel manifest schema mismatch")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise SystemExit("activity recovery panel manifest missing outputs")
    for name, path in artifacts.items():
        record = outputs.get(name)
        if not isinstance(record, dict):
            raise SystemExit(f"panel manifest missing output {name}")
        if record.get("sha256") != _sha256(path):
            raise SystemExit(f"panel artifact sha256 mismatch for {name}: {path}")
        if name.endswith(".parquet"):
            rows = int(pq.ParquetFile(path).metadata.num_rows)
        else:
            rows = max(0, sum(1 for _ in path.open("rb")) - 1)
        if int(record.get("rows", -1)) != rows:
            raise SystemExit(f"panel artifact row count mismatch for {name}: {path}")
        if name == "known_panel.csv":
            validate_frozen_known_panel(path)
    return manifest


def _bound_path(record: dict[str, Any], manifest_path: Path, label: str) -> Path:
    value = record.get("path")
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"{label} missing path")
    path = Path(value)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _validate_index_panel_binding(
    *,
    index_manifest_path: Path,
    index_manifest: dict[str, Any],
    panel_manifest_path: Path,
    panel_manifest: dict[str, Any],
) -> None:
    """Revalidate the train/index/panel provenance chain at every gate."""

    panel_inputs = panel_manifest.get("inputs")
    index_inputs = index_manifest.get("inputs")
    if not isinstance(panel_inputs, dict) or not isinstance(index_inputs, dict):
        raise SystemExit("index/panel manifests require inputs objects")

    panel_index = panel_inputs.get("retrieval_index_manifest")
    if not isinstance(panel_index, dict):
        raise SystemExit("panel manifest missing bound retrieval index manifest")
    if _bound_path(
        panel_index, panel_manifest_path, "panel retrieval index manifest"
    ) != index_manifest_path.resolve() or panel_index.get("sha256") != _sha256(
        index_manifest_path
    ):
        raise SystemExit("panel manifest is not bound to the supplied retrieval index")

    panel_benchmark = panel_inputs.get("benchmark_manifest")
    index_benchmark = index_inputs.get("benchmark_manifest")
    panel_train = panel_inputs.get("train.parquet")
    index_train = index_inputs.get("train_parquet")
    for label, value in (
        ("panel benchmark", panel_benchmark),
        ("index benchmark", index_benchmark),
        ("panel train", panel_train),
        ("index train", index_train),
    ):
        if not isinstance(value, dict):
            raise SystemExit(f"missing {label} provenance record")

    assert isinstance(panel_benchmark, dict)
    assert isinstance(index_benchmark, dict)
    assert isinstance(panel_train, dict)
    assert isinstance(index_train, dict)
    panel_benchmark_path = _bound_path(
        panel_benchmark, panel_manifest_path, "panel benchmark manifest"
    )
    index_benchmark_path = _bound_path(
        index_benchmark, index_manifest_path, "index benchmark manifest"
    )
    if panel_benchmark_path != index_benchmark_path or (
        panel_benchmark.get("sha256") != index_benchmark.get("sha256")
    ):
        raise SystemExit("retrieval index and recovery panels use different benchmarks")
    _require_file(panel_benchmark_path, "bound activity benchmark manifest")
    benchmark_sha = _sha256(panel_benchmark_path)
    if benchmark_sha != panel_benchmark.get("sha256"):
        raise SystemExit("bound activity benchmark manifest sha256 changed")
    benchmark = _read_json(panel_benchmark_path, "bound activity benchmark manifest")
    if benchmark.get("schema_version") != "activity_benchmark.v1":
        raise SystemExit("bound activity benchmark manifest schema changed")

    panel_train_path = _bound_path(panel_train, panel_manifest_path, "panel train parquet")
    index_train_path = _bound_path(index_train, index_manifest_path, "index train parquet")
    if panel_train_path != index_train_path or panel_train_path != (
        panel_benchmark_path.parent / "train.parquet"
    ).resolve():
        raise SystemExit("retrieval index and recovery panels use different train parquet files")
    try:
        panel_train_rows = int(panel_train.get("rows", -1))
        index_train_rows = int(index_train.get("rows", -1))
    except (TypeError, ValueError) as exc:
        raise SystemExit("bound train row count is invalid") from exc
    output_hashes = benchmark.get("output_sha256")
    split_counts = benchmark.get("splits")
    split_count_values = (
        split_counts.get("counts") if isinstance(split_counts, dict) else None
    )
    expected_train_sha = (
        output_hashes.get("train.parquet") if isinstance(output_hashes, dict) else None
    )
    try:
        expected_train_rows = int(
            split_count_values.get("train", -1)
            if isinstance(split_count_values, dict)
            else -1
        )
    except (TypeError, ValueError) as exc:
        raise SystemExit("benchmark train row count is invalid") from exc
    if (
        not isinstance(expected_train_sha, str)
        or panel_train.get("sha256") != expected_train_sha
        or index_train.get("sha256") != expected_train_sha
        or panel_train_rows != expected_train_rows
        or index_train_rows != expected_train_rows
    ):
        raise SystemExit("bound train provenance differs across benchmark/index/panels")
    _require_file(panel_train_path, "bound train parquet")
    if _sha256(panel_train_path) != expected_train_sha or int(
        pq.ParquetFile(panel_train_path).metadata.num_rows
    ) != expected_train_rows:
        raise SystemExit("bound train parquet no longer matches benchmark provenance")


def _ranking_metrics_by_recipe(target_rows: pd.DataFrame) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for recipe_id, group in target_rows.groupby("recipe_id", sort=True):
        metrics[str(recipe_id)] = _ranking_summary(group)
    return metrics


def _empty_ranking_summary() -> dict[str, Any]:
    return {
        "coverage": {
            "measured": True,
            "truth_pairs": 0,
            "truth_pairs_in_scorable_universe": 0,
            "truth_pairs_outside_scorable_universe": 0,
            "truth_pair_coverage": 0.0,
            "all_truth": {"top10": None, "top30": None, "mrr": None},
            "in_universe": {"top10": None, "top30": None, "mrr": None},
        },
        "n_queries": 0,
        "n_truth_pairs": 0,
        "n_truth_targets": 0,
        "top10": None,
        "top30": None,
        "mrr": None,
        "max_truth_pair_target_fraction": None,
        "effective_target_count": None,
        "target_macro": {
            "top10": None,
            "top30": None,
            "mrr": None,
        },
    }


def _source_stratified_ranking_metrics_by_recipe(
    target_rows: pd.DataFrame,
) -> dict[str, dict[str, dict[str, Any]]]:
    source_column = (
        "target_panel_sources_json"
        if "target_panel_sources_json" in target_rows.columns
        else "truth_target_panel_sources_json"
    )
    if source_column not in target_rows.columns:
        return {}
    expanded = target_rows.copy()
    expanded["_panel_sources"] = expanded[source_column].map(
        lambda value: _parse_string_list(value, "target metric panel source attribution")
    )
    metrics: dict[str, dict[str, dict[str, Any]]] = {}
    for recipe_id, recipe_group in expanded.groupby("recipe_id", sort=True):
        by_source: dict[str, dict[str, Any]] = {}
        for source in SOURCE_STRATIFIED_PANEL_SOURCES:
            mask = recipe_group["_panel_sources"].map(lambda values: source in values)
            source_group = recipe_group.loc[mask].drop(columns=["_panel_sources"])
            by_source[source] = (
                _ranking_summary(source_group)
                if not source_group.empty
                else _empty_ranking_summary()
            )
        metrics[str(recipe_id)] = by_source
    return metrics


def _calibration_metrics_by_recipe(
    scored: pd.DataFrame,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    fit, evaluation = _calibration_split(scored)
    metrics: dict[str, dict[str, Any]] = {}
    calibrators: dict[str, dict[str, Any]] = {}
    for recipe in RECIPES:
        recipe_metrics, calibrator = _calibration_recipe_metrics(
            scored, recipe, fit, evaluation
        )
        metrics[recipe.recipe_id] = recipe_metrics
        calibrators[recipe.recipe_id] = calibrator
    return metrics, calibrators


def _strict_improvement(
    baseline_ranking: dict[str, Any],
    candidate_ranking: dict[str, Any],
    baseline_calibration: dict[str, Any],
    candidate_calibration: dict[str, Any],
    *,
    tolerance: float = 1e-12,
) -> tuple[bool, list[str], float]:
    ranking_passes, failures, ranking_score = _strict_ranking_improvement(
        baseline_ranking,
        candidate_ranking,
        tolerance=tolerance,
    )
    gains: list[float] = [ranking_score]
    for metric in ("brier", "log_loss"):
        gain = float(baseline_calibration[metric]) - float(candidate_calibration[metric])
        gains.append(gain)
        if gain <= tolerance:
            failures.append(f"{metric}_improvement={gain:.12g} must be > {tolerance:g}")
    # ECE is reported as a calibration diagnostic but is not a proper scoring rule.
    score = float(sum(gains))
    return ranking_passes and not failures, failures, score


def _strict_ranking_improvement(
    baseline_ranking: dict[str, Any],
    candidate_ranking: dict[str, Any],
    *,
    tolerance: float = 1e-12,
) -> tuple[bool, list[str], float]:
    failures: list[str] = []
    gains: list[float] = []
    for metric in ("top10", "top30", "mrr"):
        gain = float(candidate_ranking[metric]) - float(baseline_ranking[metric])
        gains.append(gain)
        if gain <= tolerance:
            failures.append(f"{metric}_gain={gain:.12g} must be > {tolerance:g}")
    return not failures, failures, float(sum(gains))


def _write_selection_recipe_artifact(
    payload: dict[str, Any], out_dir: Path, *, passes_dev_gate: bool
) -> Path:
    recipe_path = out_dir / "recipe.json"
    diagnostic_path = out_dir / "failed_recipe_diagnostic.json"
    recipe_path.unlink(missing_ok=True)
    diagnostic_path.unlink(missing_ok=True)
    target = recipe_path if passes_dev_gate else diagnostic_path
    _write_json_atomic(payload, target)
    return target


def _evaluation_decision(
    *,
    passes_frozen_test_gate: bool,
    selected_recipe_id: str,
    baseline_recipe_id: str,
) -> dict[str, Any]:
    if not selected_recipe_id.strip() or not baseline_recipe_id.strip():
        raise ValueError("evaluation recipe identifiers must be nonblank")
    promoted = bool(passes_frozen_test_gate)
    return {
        "evaluation_completed": True,
        "claim_ready": promoted,
        "promotion_decision": (
            "promote_selected" if promoted else "retain_frozen_baseline"
        ),
        "operational_recipe_id": (
            selected_recipe_id if promoted else baseline_recipe_id
        ),
    }


def _choose_recipe(
    passing: list[tuple[float, "Recipe"]],
    comparison_rows: list[dict[str, Any]],
    preferred: str | None,
) -> "Recipe":
    """Pick among the recipes the dev gate qualified.

    Without `preferred` this is the highest dev-gate score, which is a
    benchmark-wide criterion. That is not always the right selector for a
    cosmetics tool - the 22-compound skin panel is - so an operator may name one
    of the qualifying recipes instead, and the artifact records that they did.

    The gate is a qualification, never a ranking to override: naming a recipe
    that failed it raises rather than promoting it.
    """
    if preferred:
        eligible = {recipe.recipe_id: recipe for _, recipe in passing}
        if preferred not in eligible:
            row = next((r for r in comparison_rows if r["recipe_id"] == preferred), None)
            reason = (
                f"it failed the dev gate: {row['dev_gate_failures']}"
                if row
                else "it is not a known candidate"
            )
            raise SystemExit(
                f"--prefer-recipe {preferred} cannot be selected because {reason}. "
                "The dev gate is a qualification, not a ranking to override."
            )
        return eligible[preferred]
    if passing:
        return sorted(passing, key=lambda item: (-item[0], item[1].recipe_id))[0][1]
    return max(
        CANDIDATES,
        key=lambda recipe: next(
            row["dev_gate_score"]
            for row in comparison_rows
            if row["recipe_id"] == recipe.recipe_id
        ),
    )


def select_recipe(args: argparse.Namespace) -> None:
    metric_paths = [
        args.out_dir / "dev_query_metrics.csv",
        args.out_dir / "dev_target_metrics.csv",
        args.out_dir / "dev_calibration_scores.parquet",
        args.out_dir / "dev_recipe_metrics.csv",
    ]
    cleanup_paths = [
        *metric_paths,
        args.out_dir / "recipe.json",
        args.out_dir / "failed_recipe_diagnostic.json",
        args.out_dir / "manifest.json",
    ]
    try:
        for path in cleanup_paths:
            path.unlink(missing_ok=True)
            path.with_suffix(path.suffix + ".tmp").unlink(missing_ok=True)
        panel_manifest = _validate_panel_manifest(
            args.panels_manifest,
            {
                "dev_ranking_queries.parquet": args.ranking_panel,
                "dev_calibration_pairs.parquet": args.calibration_panel,
            },
        )
        cold_adequacy_record = _cold_start_adequacy_record(panel_manifest)
        reference, index_manifest, target_manifest = load_reference_index(
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
        ranking_panel = load_ranking_panel(args.ranking_panel, "dev")
        calibration_panel = load_calibration_panel(args.calibration_panel, "dev")
        query_metrics, target_metrics = score_ranking_panel(
            reference,
            ranking_panel,
            RECIPES,
            exclude_reference_similarity=args.exclude_reference_similarity,
        )
        calibration_scores = score_calibration_panel(
            reference,
            calibration_panel,
            RECIPES,
            exclude_reference_similarity=args.exclude_reference_similarity,
        )
        ranking_metrics = _ranking_metrics_by_recipe(target_metrics)
        calibration_metrics_by_recipe, calibrators = _calibration_metrics_by_recipe(
            calibration_scores
        )
        baseline_ranking = ranking_metrics[BASELINE.recipe_id]
        baseline_calibration = calibration_metrics_by_recipe[BASELINE.recipe_id]
        comparison_rows: list[dict[str, Any]] = []
        passing: list[tuple[float, Recipe]] = []
        for recipe in CANDIDATES:
            passes, failures, score = _strict_improvement(
                baseline_ranking,
                ranking_metrics[recipe.recipe_id],
                baseline_calibration,
                calibration_metrics_by_recipe[recipe.recipe_id],
            )
            comparison_rows.append(
                {
                    "recipe_id": recipe.recipe_id,
                    **{f"ranking_{key}": value for key, value in ranking_metrics[recipe.recipe_id].items()},
                    **{
                        f"calibration_{key}": value
                        for key, value in calibration_metrics_by_recipe[recipe.recipe_id].items()
                        if key != "by_endpoint_family"
                    },
                    "passes_dev_gate": passes,
                    "dev_gate_score": score,
                    "dev_gate_failures": ";".join(failures),
                }
            )
            if passes:
                passing.append((score, recipe))
        selected = _choose_recipe(
            passing, comparison_rows, getattr(args, "prefer_recipe", None)
        )
        passes_dev_gate = bool(passing)
        _write_csv_atomic(query_metrics, metric_paths[0])
        _write_csv_atomic(target_metrics, metric_paths[1])
        _write_parquet_atomic(calibration_scores, metric_paths[2])
        baseline_row = {
            "recipe_id": BASELINE.recipe_id,
            **{f"ranking_{key}": value for key, value in baseline_ranking.items()},
            **{
                f"calibration_{key}": value
                for key, value in baseline_calibration.items()
                if key != "by_endpoint_family"
            },
            "passes_dev_gate": True,
            "dev_gate_score": 0.0,
            "dev_gate_failures": "baseline",
        }
        recipe_metrics_frame = pd.DataFrame([baseline_row, *comparison_rows])
        _write_csv_atomic(recipe_metrics_frame, metric_paths[3])
        recipe_payload = {
            "schema_version": RECIPE_SCHEMA,
            "created_at_utc": _utc_now(),
            "selection_split": "dev_2024",
            "passes_dev_gate": passes_dev_gate,
            "selection_basis": (
                "operator_choice_among_dev_qualified"
                if getattr(args, "prefer_recipe", None)
                else "highest_dev_gate_score"
            ),
            "selection_rationale": getattr(args, "selection_rationale", "") or "",
            "selected_recipe": asdict(selected),
            "baseline_recipe": asdict(BASELINE),
            "exclude_reference_similarity": args.exclude_reference_similarity,
            "score_is_calibrated_probability": False,
            "target_assistance": "none; scorer signature has no truth-target argument",
            "label_policy": {
                "positive": "measured pActivity >= 6",
                "negative": "measured pActivity <= 5",
                "gray": "5 < measured pActivity < 6 excluded",
                "unmeasured": "unlabeled, never treated as negative",
            },
            "calibrators": {
                "selected": calibrators[selected.recipe_id],
                "baseline": calibrators[BASELINE.recipe_id],
            },
            "dev_metrics": {
                "ranking": ranking_metrics,
                "calibration": calibration_metrics_by_recipe,
            },
            "gates": {"dual_cold": cold_adequacy_record},
            "provenance": {
                "index_manifest": _artifact_record(args.index_manifest),
                "target_cluster_manifest": _artifact_record(args.target_cluster_manifest),
                "panels_manifest": _artifact_record(args.panels_manifest),
                "index_schema": index_manifest["schema_version"],
                "target_schema": target_manifest["schema_version"],
                "panels_schema": panel_manifest["schema_version"],
            },
        }
        recipe_artifact = _write_selection_recipe_artifact(
            recipe_payload,
            args.out_dir,
            passes_dev_gate=passes_dev_gate,
        )
        manifest = {
            "schema_version": SELECTION_SCHEMA,
            "created_at_utc": _utc_now(),
            "passes_dev_gate": passes_dev_gate,
            "gates": {"dual_cold": cold_adequacy_record},
            "selected_recipe_id": selected.recipe_id,
            "inputs": {
                "ligands": _artifact_record(args.ligands, len(reference.fingerprints)),
                "edges": _artifact_record(args.edges),
                "index_manifest": _artifact_record(args.index_manifest),
                "target_clusters": _artifact_record(args.target_clusters, len(reference.target_ids)),
                "target_cluster_manifest": _artifact_record(args.target_cluster_manifest),
                "ranking_panel": _artifact_record(args.ranking_panel, len(ranking_panel)),
                "calibration_panel": _artifact_record(args.calibration_panel, len(calibration_panel)),
                "panels_manifest": _artifact_record(args.panels_manifest),
            },
            "outputs": {
                path.name: _artifact_record(
                    path,
                    len(pd.read_parquet(path)) if path.suffix == ".parquet" else (
                        len(pd.read_csv(path)) if path.suffix == ".csv" else None
                    ),
                )
                for path in [*metric_paths, recipe_artifact]
            },
        }
        _write_json_atomic(manifest, args.out_dir / "manifest.json")
        if not passes_dev_gate and not args.allow_no_improvement:
            raise SystemExit(
                "No candidate recipe strictly improved dev Top10, Top30, MRR, "
                "Brier, and log loss; diagnostic artifacts were written"
            )
    except BaseException:
        # Preserve complete diagnostics for a scientifically valid no-improvement result.
        if not (args.out_dir / "dev_recipe_metrics.csv").exists():
            for path in cleanup_paths:
                path.unlink(missing_ok=True)
        raise


def _recipe_from_payload(value: object, label: str) -> Recipe:
    if not isinstance(value, dict):
        raise SystemExit(f"recipe artifact missing {label}")
    try:
        recipe = Recipe(**value)
    except TypeError as exc:
        raise SystemExit(f"recipe artifact has invalid {label}") from exc
    if not recipe.recipe_id.strip():
        raise SystemExit(f"recipe artifact {label}.recipe_id must be nonblank")
    return recipe


def _validate_recipe_provenance(
    payload: dict[str, Any],
    *,
    recipe_path: Path,
    index_manifest: Path,
    target_cluster_manifest: Path,
    panels_manifest: Path,
) -> tuple[Recipe, Recipe, dict[str, Any]]:
    if payload.get("schema_version") != RECIPE_SCHEMA:
        raise SystemExit(f"recipe schema must be {RECIPE_SCHEMA}: {recipe_path}")
    if payload.get("passes_dev_gate") is not True:
        raise SystemExit("recipe did not pass frozen 2024 dev selection gate")
    if payload.get("score_is_calibrated_probability") is not False:
        raise SystemExit("ranking recipe score must not be labeled a probability")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise SystemExit("recipe artifact missing provenance")
    for key, path in (
        ("index_manifest", index_manifest),
        ("target_cluster_manifest", target_cluster_manifest),
        ("panels_manifest", panels_manifest),
    ):
        record = provenance.get(key)
        if not isinstance(record, dict) or record.get("sha256") != _sha256(path):
            raise SystemExit(f"recipe provenance mismatch for {key}: {path}")
    calibrators = payload.get("calibrators")
    if not isinstance(calibrators, dict) or not isinstance(
        calibrators.get("selected"), dict
    ) or not isinstance(calibrators.get("baseline"), dict):
        raise SystemExit("recipe artifact missing selected/baseline calibrators")
    selected = _recipe_from_payload(payload.get("selected_recipe"), "selected_recipe")
    baseline = _recipe_from_payload(payload.get("baseline_recipe"), "baseline_recipe")
    if baseline != BASELINE:
        raise SystemExit("recipe artifact changed the frozen baseline definition")
    if selected not in CANDIDATES:
        raise SystemExit("recipe artifact selected an unregistered candidate definition")
    return selected, baseline, calibrators


def _test_calibration_metrics(
    scored: pd.DataFrame,
    recipe: Recipe,
    calibrator: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    probabilities = apply_platt(
        scored[f"score__{recipe.recipe_id}"].to_numpy(float), calibrator
    )
    metrics: dict[str, Any] = calibration_metrics(
        probabilities,
        scored["label"].to_numpy(int),
        scored["sample_weight"].to_numpy(float),
    )
    by_family: dict[str, Any] = {}
    for family, group in scored.groupby("endpoint_family", sort=True):
        indexes = group.index.to_numpy()
        by_family[str(family)] = calibration_metrics(
            probabilities[indexes],
            group["label"].to_numpy(int),
            group["sample_weight"].to_numpy(float),
        )
    metrics["by_endpoint_family"] = by_family
    return metrics, probabilities


def _write_known_ranking(
    *,
    out_dir: Path,
    case_id: str,
    target_ids: np.ndarray,
    scores: np.ndarray,
    recipe: Recipe,
    recipe_artifact: Path,
    stats: dict[str, int],
    exclude_reference_similarity: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    order = np.lexsort((target_ids.astype(str), -scores))
    ranks = _average_tie_ranks(scores)
    ranking = pd.DataFrame(
        {
            "target_id": target_ids[order],
            "rank": ranks[order],
            "score": scores[order],
            "known_target_prior": np.zeros(len(order), dtype=float),
            "known_target_prior_norm": np.zeros(len(order), dtype=float),
            "scoring_method": recipe.recipe_id,
            "evidence_mode": "leave-query-out",
        }
    )
    ranking_path = out_dir / f"{case_id}.tsv"
    tmp = ranking_path.with_suffix(".tsv.tmp")
    ranking.to_csv(tmp, sep="\t", index=False)
    tmp.replace(ranking_path)
    metadata = {
        "schema_version": "skinscout.daina-run.v1",
        "created_at_utc": _utc_now(),
        "case_id": case_id,
        "evidence_mode": "leave-query-out",
        "evidence_snapshot_id": _sha256(recipe_artifact),
        "cutoff_date": None,
        "exclude_reference_similarity": exclude_reference_similarity,
        "quality_policy": "claim-grade multi-source train-only",
        "scoring_method": recipe.recipe_id,
        "score_is_calibrated_probability": False,
        "rank_policy": (
            "average rank across each exact-score tie group; TSV ties sort by target_id"
        ),
        "known_target_prior_nonzero_count": 0,
        "ranking_sha256": _sha256(ranking_path),
        "inputs": {
            "recipe_artifact": str(recipe_artifact.resolve()),
            "recipe_artifact_sha256": _sha256(recipe_artifact),
        },
        "counts": {**stats, "ranked_targets": len(ranking)},
    }
    _write_json_atomic(metadata, ranking_path.with_suffix(".metadata.json"))


def score_known_panel(
    reference: ReferenceIndex,
    known_panel: Path,
    recipes: tuple[Recipe, Recipe],
    *,
    exclude_reference_similarity: float,
    recipe_artifact: Path,
    out_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
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
    target_set = set(reference.target_ids.tolist())
    validated_cases: list[tuple[dict[str, Any], str, list[str]]] = []
    for row in cases.to_dict("records"):
        case_id = str(row["case_id"]).strip()
        targets = _parse_truth_targets(row["known_targets"])
        unknown = sorted(set(targets) - target_set)
        if unknown:
            raise SystemExit(f"known case {case_id} targets outside universe: {unknown}")
        validated_cases.append((row, case_id, targets))
    query_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    ranking_dirs = {
        recipe.recipe_id: out_root / f"known_rankings_{recipe.recipe_id}"
        for recipe in recipes
    }
    for directory in ranking_dirs.values():
        shutil.rmtree(directory, ignore_errors=True)
    scorable = scorable_target_mask(reference)
    scorable_ids = set(reference.target_ids[scorable].tolist())
    scorable_universe_size = int(scorable.sum())
    target_universe_size = int(len(reference.target_ids))
    for row, case_id, targets in validated_cases:
        qfp, _, _, _ = query_features(str(row["smiles"]))
        features, stats = score_query_features(
            reference,
            qfp,
            exclude_reference_similarity=exclude_reference_similarity,
        )
        for recipe in recipes:
            scores = apply_recipe(features, recipe)
            ranks = _rank_map(reference.target_ids, scores)
            truth_ranks = [ranks[target] for target in targets]
            query_rows.append(
                {
                    "query_id": case_id,
                    "recipe_id": recipe.recipe_id,
                    "n_truth_targets": len(targets),
                    "case_top10": int(any(rank <= 10 for rank in truth_ranks)),
                    "case_top30": int(any(rank <= 30 for rank in truth_ranks)),
                    "best_truth_rank": min(truth_ranks),
                    "n_truth_targets_in_scorable_universe": int(
                        sum(target in scorable_ids for target in targets)
                    ),
                    "scorable_universe_size": scorable_universe_size,
                    "target_universe_size": target_universe_size,
                    **stats,
                }
            )
            for target in targets:
                rank = ranks[target]
                target_rows.append(
                    {
                        "query_id": case_id,
                        "recipe_id": recipe.recipe_id,
                        "target_id": target,
                        "rank": rank,
                        "top10": int(rank <= 10),
                        "top30": int(rank <= 30),
                        "reciprocal_rank": 1.0 / rank,
                        "in_scorable_universe": bool(target in scorable_ids),
                        "scorable_universe_size": scorable_universe_size,
                        "target_universe_size": target_universe_size,
                    }
                )
            _write_known_ranking(
                out_dir=ranking_dirs[recipe.recipe_id],
                case_id=case_id,
                target_ids=reference.target_ids,
                scores=scores,
                recipe=recipe,
                recipe_artifact=recipe_artifact,
                stats=stats,
                exclude_reference_similarity=exclude_reference_similarity,
            )
    target_frame = pd.DataFrame(target_rows)
    return (
        pd.DataFrame(query_rows),
        target_frame,
        _ranking_metrics_by_recipe(target_frame),
    )


def evaluate_recipe(args: argparse.Namespace) -> None:
    out_dir = args.out_dir
    files = {
        "test_query_metrics.csv": out_dir / "test_query_metrics.csv",
        "test_target_metrics.csv": out_dir / "test_target_metrics.csv",
        "dual_cold_query_metrics.csv": out_dir / "dual_cold_query_metrics.csv",
        "dual_cold_target_metrics.csv": out_dir / "dual_cold_target_metrics.csv",
        "test_calibration_scores.parquet": out_dir / "test_calibration_scores.parquet",
        "known_query_metrics.csv": out_dir / "known_query_metrics.csv",
        "known_target_metrics.csv": out_dir / "known_target_metrics.csv",
        "summary.json": out_dir / "summary.json",
        "manifest.json": out_dir / "manifest.json",
    }
    ranking_dirs: list[Path] = []
    try:
        for directory in sorted(out_dir.glob("known_rankings_*")):
            if directory.is_dir():
                shutil.rmtree(directory)
        for path in files.values():
            path.unlink(missing_ok=True)
            path.with_suffix(path.suffix + ".tmp").unlink(missing_ok=True)
        panel_manifest = _validate_panel_manifest(
            args.panels_manifest,
            {
                "test_ranking_queries.parquet": args.ranking_panel,
                "dual_cold_ranking_queries.parquet": args.dual_cold_panel,
                "test_calibration_pairs.parquet": args.calibration_panel,
                "known_panel.csv": args.known_panel,
            },
        )
        cold_adequacy_record = _cold_start_adequacy_record(panel_manifest)
        recipe_payload = _read_json(args.recipe, "frozen activity retrieval recipe")
        if not math.isclose(
            float(recipe_payload.get("exclude_reference_similarity", float("nan"))),
            args.exclude_reference_similarity,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise SystemExit(
                "evaluation similarity exclusion threshold differs from frozen recipe"
            )
        selected, baseline, calibrators = _validate_recipe_provenance(
            recipe_payload,
            recipe_path=args.recipe,
            index_manifest=args.index_manifest,
            target_cluster_manifest=args.target_cluster_manifest,
            panels_manifest=args.panels_manifest,
        )
        reference, index_manifest, _ = load_reference_index(
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
        recipes = (baseline, selected)
        ranking_dirs = [
            out_dir / f"known_rankings_{recipe.recipe_id}" for recipe in recipes
        ]
        test_panel = load_ranking_panel(args.ranking_panel, "test")
        dual_panel = load_ranking_panel(
            args.dual_cold_panel,
            "test",
            require_source_provenance=True,
        )
        calibration_panel = load_calibration_panel(args.calibration_panel, "test")
        test_query, test_target = score_ranking_panel(
            reference,
            test_panel,
            recipes,
            exclude_reference_similarity=args.exclude_reference_similarity,
        )
        dual_query, dual_target = score_ranking_panel(
            reference,
            dual_panel,
            recipes,
            exclude_reference_similarity=args.exclude_reference_similarity,
        )
        calibration_scores = score_calibration_panel(
            reference,
            calibration_panel,
            recipes,
            exclude_reference_similarity=args.exclude_reference_similarity,
        )
        baseline_calibration, baseline_probabilities = _test_calibration_metrics(
            calibration_scores, baseline, calibrators["baseline"]
        )
        selected_calibration, selected_probabilities = _test_calibration_metrics(
            calibration_scores, selected, calibrators["selected"]
        )
        calibration_scores[f"probability__{baseline.recipe_id}"] = baseline_probabilities
        calibration_scores[f"probability__{selected.recipe_id}"] = selected_probabilities
        known_query, known_target, known_metrics = score_known_panel(
            reference,
            args.known_panel,
            recipes,
            exclude_reference_similarity=args.exclude_reference_similarity,
            recipe_artifact=args.recipe,
            out_root=out_dir,
        )
        test_metrics = _ranking_metrics_by_recipe(test_target)
        dual_metrics = _ranking_metrics_by_recipe(dual_target)
        dual_source_metrics = _source_stratified_ranking_metrics_by_recipe(dual_target)
        cold_gate_record = {
            **_cold_start_claim_record(
                cold_adequacy_record,
                dual_metrics[selected.recipe_id]["coverage"],
            ),
            "metrics_include_target_macro": True,
            "source_stratified_metrics": list(SOURCE_STRATIFIED_PANEL_SOURCES),
        }
        cold_start_claimable = bool(cold_gate_record["claimable"])
        test_passes, test_failures, _ = _strict_improvement(
            test_metrics[baseline.recipe_id],
            test_metrics[selected.recipe_id],
            baseline_calibration,
            selected_calibration,
        )
        known_passes, known_failures, _ = _strict_ranking_improvement(
            known_metrics[baseline.recipe_id],
            known_metrics[selected.recipe_id],
        )
        passes = test_passes and known_passes and cold_start_claimable
        evaluation_decision = _evaluation_decision(
            passes_frozen_test_gate=passes,
            selected_recipe_id=selected.recipe_id,
            baseline_recipe_id=baseline.recipe_id,
        )
        _write_csv_atomic(test_query, files["test_query_metrics.csv"])
        _write_csv_atomic(test_target, files["test_target_metrics.csv"])
        _write_csv_atomic(dual_query, files["dual_cold_query_metrics.csv"])
        _write_csv_atomic(dual_target, files["dual_cold_target_metrics.csv"])
        _write_parquet_atomic(
            calibration_scores, files["test_calibration_scores.parquet"]
        )
        _write_csv_atomic(known_query, files["known_query_metrics.csv"])
        _write_csv_atomic(known_target, files["known_target_metrics.csv"])
        summary = {
            "schema_version": EVALUATION_SCHEMA,
            "created_at_utc": _utc_now(),
            "passes_frozen_test_gate": passes,
            "selected_recipe_id": selected.recipe_id,
            "baseline_recipe_id": baseline.recipe_id,
            **evaluation_decision,
            "ranking": {
                "temporal_test_2025": test_metrics,
                "dual_cold": dual_metrics,
                "dual_cold_by_panel_source": dual_source_metrics,
                "full_known_skin_panel_15_cases": known_metrics,
            },
            "calibration": {
                "baseline": baseline_calibration,
                "selected": selected_calibration,
                "labels": "measured only; no unmeasured negatives",
            },
            "gates": {
                "temporal_test_and_calibration": {
                    "passes": test_passes,
                    "failures": test_failures,
                },
                "full_known_skin_panel": {
                    "passes": known_passes,
                    "failures": known_failures,
                },
                "dual_cold": cold_gate_record,
            },
            "claim_scope": {
                "ranking_score": "not a calibrated probability",
                "calibrated_probability": "operational potent-interaction event only",
                "target_assistance": "none",
                "known_panel": (
                    f"all {len(FROZEN_KNOWN_PANEL_CASES)} cases and all declared "
                    "target pairs unchanged"
                ),
                "cold_start": (
                    "claimable"
                    if cold_start_claimable
                    else (
                        "diagnostic only; panel adequacy or scorable truth "
                        "coverage insufficient"
                    )
                ),
                "reference_filter": f"all train ligands with Tanimoto >= {args.exclude_reference_similarity:g} excluded globally per query",
            },
        }
        _write_json_atomic(summary, files["summary.json"])
        manifest = {
            "schema_version": EVALUATION_SCHEMA,
            "created_at_utc": _utc_now(),
            "passes_frozen_test_gate": passes,
            "selected_recipe_id": selected.recipe_id,
            "baseline_recipe_id": baseline.recipe_id,
            **evaluation_decision,
            "gates": {"dual_cold": cold_gate_record},
            "inputs": {
                "recipe": _artifact_record(args.recipe),
                "index_manifest": _artifact_record(args.index_manifest),
                "target_cluster_manifest": _artifact_record(args.target_cluster_manifest),
                "panels_manifest": _artifact_record(args.panels_manifest),
                "panels_schema": panel_manifest["schema_version"],
                "known_panel": _artifact_record(args.known_panel, 15),
            },
            "outputs": {
                name: _artifact_record(
                    path,
                    len(pd.read_parquet(path)) if path.suffix == ".parquet" else (
                        len(pd.read_csv(path)) if path.suffix == ".csv" else None
                    ),
                )
                for name, path in files.items()
                if name != "manifest.json"
            },
            "metrics": {
                "dual_cold_by_panel_source": dual_source_metrics,
                "source_attribution_input_column": "truth_target_panel_sources_json",
                "source_attribution_metric_column": "target_panel_sources_json",
                "source_stratified_panel_sources": list(SOURCE_STRATIFIED_PANEL_SOURCES),
                "rcsb_holo_direct_contact_calibration_policy": (
                    "ranking-only; never treated as probability or calibration data"
                ),
            },
        }
        _write_json_atomic(manifest, files["manifest.json"])
        if not passes and not args.allow_no_improvement:
            raise SystemExit(
                "Frozen 2025/full-known-panel evaluation did not strictly improve "
                "Top10, Top30, MRR and measured-only proper calibration scores, or "
                "the dual-cold panel lacked scorable truth coverage"
            )
    except BaseException:
        if not files["summary.json"].exists():
            for path in files.values():
                path.unlink(missing_ok=True)
            for directory in ranking_dirs:
                shutil.rmtree(directory, ignore_errors=True)
        raise


def _add_reference_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ligands", required=True, type=Path)
    parser.add_argument("--edges", required=True, type=Path)
    parser.add_argument("--index-manifest", required=True, type=Path)
    parser.add_argument("--target-clusters", required=True, type=Path)
    parser.add_argument("--target-cluster-manifest", required=True, type=Path)
    parser.add_argument("--panels-manifest", required=True, type=Path)
    parser.add_argument("--exclude-reference-similarity", type=float, default=0.85)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--allow-no-improvement",
        action="store_true",
        help="retain complete diagnostic artifacts but exit successfully when a gate fails",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser(
        "select", help="select and calibrate a recipe on frozen 2024 dev panels"
    )
    _add_reference_arguments(select)
    select.add_argument("--ranking-panel", required=True, type=Path)
    select.add_argument("--calibration-panel", required=True, type=Path)
    select.add_argument(
        "--prefer-recipe",
        help=(
            "select this recipe instead of the highest dev-gate score. It must "
            "still pass the dev gate; this chooses among qualifiers for a "
            "criterion the benchmark does not express, such as the skin panel"
        ),
    )
    select.add_argument(
        "--selection-rationale",
        default="",
        help="recorded in the artifact next to --prefer-recipe",
    )
    select.set_defaults(func=select_recipe)

    evaluate = subparsers.add_parser(
        "evaluate", help="evaluate a frozen dev-selected recipe on 2025 and known panels"
    )
    _add_reference_arguments(evaluate)
    evaluate.add_argument("--recipe", required=True, type=Path)
    evaluate.add_argument("--ranking-panel", required=True, type=Path)
    evaluate.add_argument("--dual-cold-panel", required=True, type=Path)
    evaluate.add_argument("--calibration-panel", required=True, type=Path)
    evaluate.add_argument("--known-panel", required=True, type=Path)
    evaluate.set_defaults(func=evaluate_recipe)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not math.isfinite(args.exclude_reference_similarity) or not (
        0.0 < args.exclude_reference_similarity <= 1.0
    ):
        raise SystemExit(
            "--exclude-reference-similarity must be finite and in (0, 1]"
        )
    args.func(args)


if __name__ == "__main__":
    main()
