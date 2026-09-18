#!/usr/bin/env python3
"""Score a query with the promoted retrieval recipe, for Stage 3.

The operational gate has recorded `operational_recipe_id=union_p6_consensus`
since 2026-08-31, but nothing read it: Stage 3 scored with `max-similarity`
against the ChEMBL mirror, and the recipe weights existed only inside the
evaluation harness. This is the missing connection.

It produces the same per-target frame Stage 3 already emits - `target_id`,
`max_tanimoto`, `evidence_count`, `supporting_molecule_id`, `score` - so the
downstream overlay, band re-ranking and report code need no changes. What
differs is `score`: a weighted combination of six features rather than the
single nearest-neighbour similarity.

Two things a reader should know before switching Stage 3 onto this:

* It scores against the retrieval index, not the ChEMBL mirror. The evaluation
  index is built from the benchmark train split (<=2023-12-31), so pointing a
  production run at it would *narrow* what a researcher sees. A production run
  needs an index built with `--index-role production`.
* The recovery numbers in `docs/RESEARCHER_GUIDE.md` were measured on the
  max-similarity path. They do not describe this one until the panel is re-run
  through it.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from activity_retrieval_scoring import (
    Recipe,
    apply_recipe,
    load_reference_index,
    score_query_features,
)

RECIPE_SCHEMA = "skinscout.activity-retrieval-recipe.v1"


def load_operational_recipe(path: Path, recipe_id: str) -> Recipe:
    """Read the selected or frozen-baseline recipe named by the operational gate."""
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"recipe artifact is missing or empty: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"unable to parse recipe artifact: {path}") from error
    if payload.get("schema_version") != RECIPE_SCHEMA:
        raise SystemExit(f"recipe schema must be {RECIPE_SCHEMA}: {path}")
    if payload.get("passes_dev_gate") is not True:
        raise SystemExit(f"recipe did not pass dev selection: {path}")
    candidates = [payload.get("selected_recipe"), payload.get("baseline_recipe")]
    matches = [item for item in candidates if isinstance(item, dict) and item.get("recipe_id") == recipe_id]
    if len(matches) != 1:
        raise SystemExit(f"recipe artifact does not define operational recipe {recipe_id!r}: {path}")
    selected = matches[0]

    known = {field.name for field in fields(Recipe)}
    unknown = sorted(set(selected) - known)
    if unknown:
        raise SystemExit(
            f"recipe artifact carries weights this scorer does not implement "
            f"{unknown}; scoring with it would silently ignore them: {path}"
        )
    missing = sorted(known - set(selected) - {"recipe_id"})
    values = {name: selected[name] for name in known if name in selected}
    if "recipe_id" not in values:
        raise SystemExit(f"recipe artifact has no recipe_id: {path}")
    for name in missing:
        values[name] = 0.0
    return Recipe(**values)


def load_promoted_recipe(path: Path) -> Recipe:
    """Read the recipe the dev selection promoted, failing closed on anything else."""
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"recipe artifact is missing or empty: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"unable to parse recipe artifact: {path}") from error
    if payload.get("schema_version") != RECIPE_SCHEMA:
        raise SystemExit(f"recipe schema must be {RECIPE_SCHEMA}: {path}")
    if payload.get("passes_dev_gate") is not True:
        raise SystemExit(f"recipe did not pass dev selection: {path}")
    selected = payload.get("selected_recipe")
    if not isinstance(selected, dict) or not str(selected.get("recipe_id") or "").strip():
        raise SystemExit(f"recipe artifact has no selected_recipe: {path}")
    return load_operational_recipe(path, str(selected["recipe_id"]))


def score_query_with_recipe(
    *,
    query_smiles: str | None = None,
    query_fp: Any = None,
    recipe: Recipe,
    ligands_path: Path,
    edges_path: Path,
    index_manifest_path: Path,
    target_csv: Path,
    target_manifest_path: Path,
    exclude_reference_similarity: float | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return (per_target, stats) in the shape Stage 3 already writes.

    Pass `query_fp` when the caller already has a fingerprint - Stage 3 builds
    one from the run's SDF, and re-deriving it from SMILES would put a second
    standardisation between the structure that was docked and the one that was
    scored.
    """
    from activity_retrieval_scoring import query_features

    if (query_smiles is None) == (query_fp is None):
        raise ValueError("pass exactly one of query_smiles or query_fp")

    reference, index_manifest, _ = load_reference_index(
        ligands_path=ligands_path,
        edges_path=edges_path,
        index_manifest_path=index_manifest_path,
        target_csv=target_csv,
        target_manifest_path=target_manifest_path,
        # A researcher's own compound gains nothing from the temporal split, and
        # a production run against the train-only index would see less.
        allow_production=True,
    )
    if query_fp is None:
        query_fp, _, _, _ = query_features(query_smiles)
    features, feature_stats = score_query_features(
        reference,
        query_fp,
        exclude_reference_similarity=exclude_reference_similarity,
    )
    scores = apply_recipe(features, recipe)

    # The nearest measured analogue per target, and which ligand it was: the
    # reader-facing columns are about evidence, not about the weighted score.
    from rdkit import DataStructs

    similarities = np.asarray(
        DataStructs.BulkTanimotoSimilarity(query_fp, reference.fingerprints),
        dtype=np.float64,
    )
    eligible = (
        np.ones(len(similarities), dtype=bool)
        if exclude_reference_similarity is None
        else similarities < exclude_reference_similarity
    )
    edge_sim = np.where(eligible[reference.edge_ligand], similarities[reference.edge_ligand], -1.0)

    n_targets = len(reference.target_ids)
    best_sim = np.full(n_targets, -1.0)
    best_ligand = np.full(n_targets, -1, dtype=np.int64)
    best_edge = np.full(n_targets, -1, dtype=np.int64)
    counts = np.zeros(n_targets, dtype=np.int64)
    for edge_index in range(len(reference.edge_target)):
        target = reference.edge_target[edge_index]
        counts[target] += 1
        value = edge_sim[edge_index]
        if value > best_sim[target]:
            best_sim[target] = value
            best_ligand[target] = reference.edge_ligand[edge_index]
            best_edge[target] = edge_index

    # What that nearest analogue's measurement actually said. max_union_any
    # weights similarity without regard to the label, so a target can rank on an
    # analogue that was measured *not* to bind; 9.9% of index edges are
    # negative-only. Excluding them scores worse on the panel
    # (docs/RECIPE_RUNPATH_MEASURED_20260831.md, section 7), so the ranking keeps
    # them and the reader is told instead.
    # Named for the index's label policy, not for biology. alpha-arbutin really
    # does inhibit tyrosinase, and its own measurement still lands under the
    # negative threshold - calling that "inactive" would be a stronger claim than
    # the number supports.
    def _label(edge_index: int) -> str:
        if edge_index < 0:
            return "none"
        if reference.edge_positive_count[edge_index] > 0:
            return "at_or_above_threshold"
        if reference.edge_negative_count[edge_index] > 0:
            return "below_threshold"
        return "between_thresholds"

    keep = scores > 0.0
    frame = pd.DataFrame(
        {
            "target_id": reference.target_ids[keep],
            "max_tanimoto": np.where(best_sim[keep] < 0.0, 0.0, best_sim[keep]),
            "evidence_count": counts[keep],
            # The index keys ligands by structure - "<InChIKey>#SMILES-<hash>" -
            # because it deduplicates across sources that have no ChEMBL id. The
            # mirror path put a ChEMBL accession here and a reader looks this up
            # by hand, so hand back the standard InChIKey rather than an internal
            # key with a hash glued to it.
            "supporting_molecule_id": [
                str(reference.ligand_keys[index]).split("#", 1)[0] if index >= 0 else ""
                for index in best_ligand[keep]
            ],
            "score": scores[keep],
            "supporting_evidence": [_label(int(index)) for index in best_edge[keep]],
        }
    )
    frame = frame.sort_values(
        ["score", "target_id"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)

    stats = {
        **feature_stats,
        "recipe_id": recipe.recipe_id,
        "index_schema": index_manifest.get("schema_version"),
        "index_role": index_manifest.get("index_role", "evaluation"),
        "exclude_reference_similarity": exclude_reference_similarity,
        "scored_targets": int(len(frame)),
        "target_universe": int(n_targets),
    }
    return frame, stats
