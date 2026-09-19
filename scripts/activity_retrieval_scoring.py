#!/usr/bin/env python3
"""Recipe scoring for activity retrieval, shared by the run path and evaluation.

This used to live only in `eval/activity_retrieval_model.py`, which made it
unreachable from Stage 3: eval imports `stage3_daina_zoete` for
`fp_from_uint64_row`, so importing eval back would be a cycle. That is why the
promoted recipe changed the recorded decision and nothing a researcher saw.

Nothing here is evaluation-specific. `load_reference_index` reads the retrieval
index, `score_query_features` turns one query fingerprint into the per-target
features, and `apply_recipe` weights them. The evaluation harness re-exports
these names, so its behaviour is unchanged by the move.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# The standardizer the index was built with. Scoring a query with a different
# one would compare fingerprints of differently-normalised molecules.
from activity_recovery_contracts import (  # noqa: E402
    ALPHAFOLD_HUMAN_V4_SOURCE,
    EXPECTED_ALPHAFOLD_BASE_TARGET_COUNT,
    EXPECTED_SCREENABLE_TARGET_COUNT,
)
from build_activity_retrieval_index import (  # noqa: E402
    MORGAN_GENERATOR,
    _standardize_mol,
    _structure_ligand_key,
)
from rdkit import Chem, DataStructs
from stage3_daina_zoete import fp_from_uint64_row  # noqa: E402

INDEX_SCHEMA = "skinscout.activity-retrieval-index.v4"
PRODUCTION_INDEX_SCHEMA = "skinscout.activity-retrieval-index.production.v1"
TARGET_SCHEMA = "skinscout.screenable-target-cluster-map.v2"
RECIPE_SCHEMA = "skinscout.activity-retrieval-recipe.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# --- recipes -------------------------------------------------------------
@dataclass(frozen=True)
class Recipe:
    recipe_id: str
    max_union5: float = 0.0
    max_union6: float = 0.0
    quality_union6: float = 0.0
    source_consensus6: float = 0.0
    support_union6: float = 0.0
    negative_contrast: float = 0.0
    # No potency threshold. Every other feature cuts at pActivity 5 or 6, which
    # is above where cosmetic actives live: kojic acid's nearest tyrosinase
    # analogues sit at 4.3-4.7, so all six original recipes lost a target the
    # run path ranks first. Defaults to 0.0, so recipes that predate it score
    # exactly as before.
    max_union_any: float = 0.0
    # Same as max_union_any but blind to edges whose only measurements say the
    # compound does not bind. Defaults to 0.0 so every earlier recipe is
    # bit-identical.
    max_union_nonneg: float = 0.0


BASELINE = Recipe("chembl_p5_max")
CANDIDATES = (
    Recipe("union_p5_max", max_union5=1.0),
    Recipe("union_p6_max", max_union6=1.0),
    Recipe(
        "union_p6_quality",
        max_union6=0.72,
        quality_union6=0.23,
        support_union6=0.05,
    ),
    Recipe(
        "union_p6_consensus",
        max_union6=0.68,
        quality_union6=0.17,
        source_consensus6=0.10,
        support_union6=0.05,
    ),
    Recipe(
        "union_p6_contrast",
        max_union6=0.68,
        quality_union6=0.17,
        source_consensus6=0.08,
        support_union6=0.07,
        negative_contrast=0.08,
    ),
)
# Added 2026-08-31 after every earlier candidate lost tyrosinase on the run path.
# Three structurally distinct points rather than weights tuned against the skin
# panel - the dev selection picks, as it does for the others.
# max_union_any weights an analogue's similarity with no regard to what its
# measurement said. That is deliberate - cosmetic actives are weak, and every
# potency-thresholded recipe lost tyrosinase - but it does not separate "measured
# and weakly active" from "measured and inactive". On the shipped index 18.1% of
# pairs are gray-only (between the thresholds) and 9.9% are negative-only. These
# keep the first and drop the second.
NEGATIVE_AWARE_CANDIDATES = (
    Recipe("union_nonneg_max", max_union_nonneg=1.0),
    Recipe(
        "union_nonneg_consensus",
        max_union_nonneg=0.45,
        max_union6=0.30,
        quality_union6=0.15,
        source_consensus6=0.10,
    ),
    Recipe(
        "union_nonneg_contrast",
        max_union_nonneg=0.45,
        max_union6=0.30,
        quality_union6=0.15,
        negative_contrast=0.10,
    ),
)

# Excluding negative-only edges outright is blunt, because "negative" here means
# "at or below the index's threshold", not "does not bind". EGCG's nearest MMP2
# analogues sit at similarity 0.727 with pActivity 4.06-4.66: real catechin
# measurements, labelled negative by the 5.0 cut, and dropping them takes
# max_union_nonneg from 0.727 to 0.319 and EGCG -> MMP2 from rank 8 to 56.
#
# max_union_nonneg <= max_union_any always, since it maximises over a subset. So
# weighting both is not double counting - it is a discount: a negative-only edge
# earns the max_union_any weight alone, while a non-negative edge earns both.
DISCOUNT_CANDIDATES = (
    Recipe("union_discount_light", max_union_any=0.30, max_union_nonneg=0.15,
           max_union6=0.30, quality_union6=0.15, source_consensus6=0.10),
    Recipe("union_discount_even", max_union_any=0.225, max_union_nonneg=0.225,
           max_union6=0.30, quality_union6=0.15, source_consensus6=0.10),
    Recipe("union_discount_heavy", max_union_any=0.15, max_union_nonneg=0.30,
           max_union6=0.30, quality_union6=0.15, source_consensus6=0.10),
)

THRESHOLD_FREE_CANDIDATES = (
    Recipe("union_any_max", max_union_any=1.0),
    Recipe("union_any_p6", max_union_any=0.5, max_union6=0.5),
    Recipe(
        "union_any_consensus",
        max_union_any=0.45,
        max_union6=0.30,
        quality_union6=0.15,
        source_consensus6=0.10,
    ),
)
CANDIDATES = (
    *CANDIDATES,
    *THRESHOLD_FREE_CANDIDATES,
    *NEGATIVE_AWARE_CANDIDATES,
    *DISCOUNT_CANDIDATES,
)
RECIPES = (BASELINE, *CANDIDATES)

# --- json helpers --------------------------------------------------------
def _require_file(path: Path, label: str) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"{label} is required and must be non-empty: {path}")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _require_file(path, label)
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Unable to parse {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} root must be an object: {path}")
    return payload



# --- scorer --------------------------------------------------------------
def _validate_registered_artifact(
    manifest: dict[str, Any], key: str, path: Path, rows: int
) -> None:
    outputs = manifest.get("outputs")
    record = outputs.get(key) if isinstance(outputs, dict) else None
    if not isinstance(record, dict):
        raise SystemExit(f"Index manifest missing outputs.{key}")
    if record.get("sha256") != _sha256(path):
        raise SystemExit(f"Index manifest outputs.{key}.sha256 mismatch: {path}")
    if int(record.get("rows", -1)) != rows:
        raise SystemExit(f"Index manifest outputs.{key}.rows mismatch: {path}")


@dataclass
class ReferenceIndex:
    target_ids: np.ndarray
    fingerprints: list[DataStructs.ExplicitBitVect]
    ligand_keys: np.ndarray
    connectivity_keys: np.ndarray
    edge_ligand: np.ndarray
    edge_target: np.ndarray
    edge_max_pactivity: np.ndarray
    edge_chembl_max: np.ndarray
    edge_bindingdb_max: np.ndarray
    edge_gtopdb_max: np.ndarray
    edge_positive_count: np.ndarray
    edge_negative_count: np.ndarray
    edge_measurement_count: np.ndarray


def _load_target_universe(
    target_csv: Path, target_manifest_path: Path
) -> tuple[np.ndarray, dict[str, Any]]:
    manifest = _read_json(target_manifest_path, "target cluster manifest")
    if manifest.get("schema_version") != TARGET_SCHEMA:
        raise SystemExit(
            f"target cluster manifest schema must be {TARGET_SCHEMA}: {target_manifest_path}"
        )
    policy = manifest.get("universe_policy")
    if not isinstance(policy, dict):
        raise SystemExit("screenable target manifest missing universe_policy")
    if policy.get("evaluation_panel_used") is not False:
        raise SystemExit("screenable target universe must not use an evaluation panel")
    if policy.get("known_target_assistance") is not False:
        raise SystemExit("screenable target universe must not use known-target assistance")
    contract = manifest.get("production_contract")
    if not isinstance(contract, dict) or contract.get("passes") is not True:
        raise SystemExit("screenable target universe did not pass its production contract")
    if contract.get("fixture_mode") is not False:
        raise SystemExit("fixture target universes cannot be used for model selection or evaluation")
    if int(contract.get("expected_base_target_count", -1)) != (
        EXPECTED_ALPHAFOLD_BASE_TARGET_COUNT
    ) or int(contract.get("expected_union_target_count", -1)) != (
        EXPECTED_SCREENABLE_TARGET_COUNT
    ):
        raise SystemExit("screenable target universe production count contract changed")
    sources = manifest.get("sources")
    independent_base = (
        sources.get("independent_base") if isinstance(sources, dict) else None
    )
    if independent_base != ALPHAFOLD_HUMAN_V4_SOURCE:
        raise SystemExit("screenable target universe AlphaFold v4 provenance changed")
    evidence_source = (
        sources.get("evidence_sequence_source") if isinstance(sources, dict) else None
    )
    if not isinstance(evidence_source, dict) or (
        evidence_source.get("name") != "UniProtKB"
        or not str(evidence_source.get("release") or "").strip()
        or evidence_source.get("license") != "CC BY 4.0"
    ):
        raise SystemExit("screenable target universe UniProt provenance is invalid")
    _require_file(target_csv, "target cluster CSV")
    frame = pd.read_csv(target_csv)
    if "uniprot" not in frame.columns:
        raise SystemExit(f"target cluster CSV missing uniprot: {target_csv}")
    ids = frame["uniprot"].fillna("").astype(str).str.strip()
    if ids.eq("").any() or ids.duplicated().any():
        raise SystemExit("target cluster CSV requires unique nonblank uniprot values")
    artifact = manifest.get("artifact")
    if not isinstance(artifact, dict):
        raise SystemExit("target cluster manifest missing artifact")
    if artifact.get("sha256") != _sha256(target_csv) or int(
        artifact.get("rows", -1)
    ) != len(frame):
        raise SystemExit("target cluster CSV does not match target cluster manifest")
    if int(policy.get("union_target_count", -1)) != len(frame):
        raise SystemExit("screenable target universe count does not match its policy")
    base_count = int(policy.get("base_target_count", -1))
    evidence_count = int(policy.get("evidence_target_count", -1))
    supplemental_count = int(policy.get("supplemental_target_count", -1))
    if (
        base_count != EXPECTED_ALPHAFOLD_BASE_TARGET_COUNT
        or evidence_count < 1
        or supplemental_count < 0
        or base_count + supplemental_count != len(frame)
        or supplemental_count > evidence_count
        or len(frame) != EXPECTED_SCREENABLE_TARGET_COUNT
    ):
        raise SystemExit("screenable target universe policy counts are inconsistent")
    return np.array(sorted(ids.tolist()), dtype=object), manifest


def _aggregate_edge_pairs(edges: pd.DataFrame, target_to_index: dict[str, int]) -> pd.DataFrame:
    required = {
        "ligand_index",
        "uniprot",
        "source_db",
        "endpoint_family",
        "measurement_count",
        "positive_measurement_count",
        "gray_measurement_count",
        "negative_measurement_count",
        "max_pactivity",
    }
    missing = sorted(required - set(edges.columns))
    if missing:
        raise SystemExit(f"retrieval edges missing required columns: {missing}")
    work = edges.loc[:, sorted(required)].copy()
    work["uniprot"] = work["uniprot"].fillna("").astype(str).str.strip()
    unknown = sorted(set(work["uniprot"]) - set(target_to_index))
    if unknown:
        raise SystemExit(f"retrieval edges contain targets outside universe: {unknown[:10]}")
    work["source_normalized"] = work["source_db"].astype(str).str.strip().str.lower()
    allowed_sources = {"chembl", "bindingdb", "gtopdb"}
    bad_sources = sorted(set(work["source_normalized"]) - allowed_sources)
    if bad_sources:
        raise SystemExit(f"retrieval edges contain unsupported sources: {bad_sources}")
    numeric_columns = [
        "ligand_index",
        "measurement_count",
        "positive_measurement_count",
        "gray_measurement_count",
        "negative_measurement_count",
        "max_pactivity",
    ]
    for column in numeric_columns:
        if work[column].map(lambda value: isinstance(value, (bool, np.bool_))).any():
            raise SystemExit(f"retrieval edges contain invalid numeric {column}: boolean")
        work[column] = pd.to_numeric(work[column], errors="coerce")
        if work[column].isna().any() or not np.isfinite(work[column].to_numpy(float)).all():
            raise SystemExit(f"retrieval edges contain invalid numeric {column}")
        if column != "max_pactivity":
            values = work[column]
            if (values % 1 != 0).any():
                raise SystemExit(f"retrieval edges require integer {column}")
            if not values.empty and int(values.max()) > np.iinfo(np.int64).max:
                raise SystemExit(f"retrieval edges contain out-of-range {column}")
            if (values < 0).any():
                raise SystemExit(f"retrieval edges contain negative {column}")
            work[column] = values.astype(np.int64)
    counts = work[
        [
            "measurement_count",
            "positive_measurement_count",
            "gray_measurement_count",
            "negative_measurement_count",
        ]
    ]
    if (counts < 0).any().any():
        raise SystemExit("retrieval edges contain negative measurement counts")
    if not (
        counts["positive_measurement_count"]
        + counts["gray_measurement_count"]
        + counts["negative_measurement_count"]
        == counts["measurement_count"]
    ).all():
        raise SystemExit("retrieval edge label counts do not sum to measurement_count")
    work["chembl_max"] = work["max_pactivity"].where(
        work["source_normalized"].eq("chembl"), -np.inf
    )
    work["bindingdb_max"] = work["max_pactivity"].where(
        work["source_normalized"].eq("bindingdb"), -np.inf
    )
    work["gtopdb_max"] = work["max_pactivity"].where(
        work["source_normalized"].eq("gtopdb"), -np.inf
    )
    aggregated = (
        work.groupby(["ligand_index", "uniprot"], sort=True, as_index=False)
        .agg(
            max_pactivity=("max_pactivity", "max"),
            chembl_max=("chembl_max", "max"),
            bindingdb_max=("bindingdb_max", "max"),
            gtopdb_max=("gtopdb_max", "max"),
            positive_count=("positive_measurement_count", "sum"),
            negative_count=("negative_measurement_count", "sum"),
            measurement_count=("measurement_count", "sum"),
        )
        .sort_values(["ligand_index", "uniprot"], kind="mergesort")
        .reset_index(drop=True)
    )
    aggregated["target_index"] = aggregated["uniprot"].map(target_to_index)
    return aggregated


def load_reference_index(
    *,
    ligands_path: Path,
    edges_path: Path,
    index_manifest_path: Path,
    target_csv: Path,
    target_manifest_path: Path,
    allow_production: bool = False,
) -> tuple[ReferenceIndex, dict[str, Any], dict[str, Any]]:
    """Load a retrieval index.

    A production index also holds dev and test, so measuring recovery against one
    would score the model on rows it retrieves from. Evaluation therefore leaves
    `allow_production` false and gets the evaluation index or nothing; the run
    path passes true, because for a researcher's own compound the temporal split
    is a constraint with no purpose.
    """
    index_manifest = _read_json(index_manifest_path, "activity retrieval index manifest")
    schema = index_manifest.get("schema_version")
    permitted = {INDEX_SCHEMA}
    if allow_production:
        permitted.add(PRODUCTION_INDEX_SCHEMA)
    if schema not in permitted:
        raise SystemExit(
            f"retrieval index schema must be one of {sorted(permitted)}: got {schema!r}"
        )
    if not allow_production and index_manifest.get("index_role", "evaluation") != "evaluation":
        raise SystemExit(
            "this caller may only read an evaluation index; "
            f"{index_manifest_path} declares index_role="
            f"{index_manifest.get('index_role')!r}"
        )
    target_ids, target_manifest = _load_target_universe(target_csv, target_manifest_path)
    ligands = pd.read_parquet(ligands_path)
    edges = pd.read_parquet(edges_path)
    _validate_registered_artifact(index_manifest, "ligands", ligands_path, len(ligands))
    _validate_registered_artifact(index_manifest, "edges", edges_path, len(edges))
    required_ligands = {
        "ligand_index",
        "ligand_key",
        "standard_inchikey",
        "connectivity_key",
        "canonical_smiles",
        "standardization_route",
        "bitvec",
    }
    missing = sorted(required_ligands - set(ligands.columns))
    if missing:
        raise SystemExit(f"retrieval ligands missing required columns: {missing}")
    ligand_indexes = pd.to_numeric(ligands["ligand_index"], errors="coerce")
    if ligand_indexes.isna().any() or ligand_indexes.tolist() != list(range(len(ligands))):
        raise SystemExit("retrieval ligand_index must be contiguous, sorted, and zero-based")
    for column in (
        "ligand_key",
        "standard_inchikey",
        "connectivity_key",
        "canonical_smiles",
        "standardization_route",
    ):
        values = ligands[column].fillna("").astype(str).str.strip()
        if values.eq("").any():
            raise SystemExit(f"retrieval ligands contain blank {column}")
        ligands[column] = values
    if ligands["ligand_key"].duplicated().any():
        raise SystemExit("retrieval ligands contain duplicate ligand_key values")
    expected_keys_and_routes = [
        _structure_ligand_key(str(inchikey), str(smiles))
        for inchikey, smiles in zip(
            ligands["standard_inchikey"], ligands["canonical_smiles"], strict=True
        )
    ]
    if ligands["ligand_key"].tolist() != [value[0] for value in expected_keys_and_routes]:
        raise SystemExit("retrieval ligand_key does not match registered structure-key policy")
    if ligands["standardization_route"].tolist() != [
        value[1] for value in expected_keys_and_routes
    ]:
        raise SystemExit("retrieval standardization_route does not match structure-key policy")
    if not ligands["connectivity_key"].eq(ligands["standard_inchikey"].str[:14]).all():
        raise SystemExit("retrieval connectivity_key does not match standard_inchikey")
    fingerprints: list[DataStructs.ExplicitBitVect] = []
    for ligand_key, words in zip(ligands["ligand_key"], ligands["bitvec"], strict=True):
        try:
            fingerprints.append(fp_from_uint64_row(list(words)))
        except (TypeError, ValueError, OverflowError) as exc:
            raise SystemExit(f"invalid fingerprint for retrieval ligand {ligand_key}") from exc
    target_to_index = {target: idx for idx, target in enumerate(target_ids.tolist())}
    pair_edges = _aggregate_edge_pairs(edges, target_to_index)
    edge_ligand = pair_edges["ligand_index"].to_numpy(np.int64)
    if edge_ligand.size and (
        edge_ligand.min() < 0 or edge_ligand.max() >= len(ligands)
    ):
        raise SystemExit("retrieval edges reference out-of-range ligand_index")
    return (
        ReferenceIndex(
            target_ids=target_ids,
            fingerprints=fingerprints,
            ligand_keys=ligands["ligand_key"].to_numpy(object),
            connectivity_keys=ligands["connectivity_key"].to_numpy(object),
            edge_ligand=edge_ligand,
            edge_target=pair_edges["target_index"].to_numpy(np.int64),
            edge_max_pactivity=pair_edges["max_pactivity"].to_numpy(float),
            edge_chembl_max=pair_edges["chembl_max"].to_numpy(float),
            edge_bindingdb_max=pair_edges["bindingdb_max"].to_numpy(float),
            edge_gtopdb_max=pair_edges["gtopdb_max"].to_numpy(float),
            edge_positive_count=pair_edges["positive_count"].to_numpy(float),
            edge_negative_count=pair_edges["negative_count"].to_numpy(float),
            edge_measurement_count=pair_edges["measurement_count"].to_numpy(float),
        ),
        index_manifest,
        target_manifest,
    )


def query_features(smiles: str) -> tuple[DataStructs.ExplicitBitVect, str, str, str]:
    try:
        mol = _standardize_mol(str(smiles).strip())
    except ValueError as exc:
        raise SystemExit(f"invalid query SMILES: {smiles!r}") from exc
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    ligand_key = Chem.MolToInchiKey(mol)
    if not ligand_key or len(ligand_key) != 27:
        raise SystemExit(f"unable to derive query InChIKey: {smiles!r}")
    return MORGAN_GENERATOR.GetFingerprint(mol), ligand_key, ligand_key[:14], canonical


def _target_max(
    target_count: int,
    target_indexes: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    output = np.zeros(target_count, dtype=np.float64)
    if mask.any():
        np.maximum.at(output, target_indexes[mask], values[mask])
    return output


def score_query_features(
    reference: ReferenceIndex,
    query_fp: DataStructs.ExplicitBitVect,
    *,
    exclude_reference_similarity: float | None,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """`None` excludes nothing, which is what Stage 3's `retrieval` mode means.

    Every evaluation caller passes a threshold, because a known compound that
    retrieves itself measures lookup rather than prediction. A production run is
    the opposite case: the researcher wants to be told their compound is already
    known to hit a target. Folding that into the threshold - by passing 1.0 -
    would silently drop the exact self-match, so it is a separate state.
    """
    if exclude_reference_similarity is not None and (
        not math.isfinite(exclude_reference_similarity)
        or not 0.0 < exclude_reference_similarity <= 1.0
    ):
        raise ValueError("exclude_reference_similarity must be finite and in (0, 1]")
    similarities = np.asarray(
        DataStructs.BulkTanimotoSimilarity(query_fp, reference.fingerprints),
        dtype=np.float64,
    )
    eligible_ligand = (
        np.ones(len(similarities), dtype=bool)
        if exclude_reference_similarity is None
        else similarities < exclude_reference_similarity
    )
    edge_similarity = similarities[reference.edge_ligand]
    eligible_edge = eligible_ligand[reference.edge_ligand]
    targets = reference.edge_target
    n_targets = len(reference.target_ids)

    union5 = eligible_edge & (reference.edge_max_pactivity >= 5.0)
    union6 = eligible_edge & (reference.edge_positive_count > 0.0)
    chembl5 = eligible_edge & (reference.edge_chembl_max >= 5.0)
    chembl6 = eligible_edge & (reference.edge_chembl_max >= 6.0)
    binding6 = eligible_edge & (reference.edge_bindingdb_max >= 6.0)
    gtopdb6 = eligible_edge & (reference.edge_gtopdb_max >= 6.0)
    negative_only = (
        eligible_edge
        & (reference.edge_positive_count == 0.0)
        & (reference.edge_negative_count > 0.0)
    )

    max_union_any = _target_max(n_targets, targets, edge_similarity, eligible_edge)
    max_union_nonneg = _target_max(
        n_targets, targets, edge_similarity, eligible_edge & ~negative_only
    )
    max_union5 = _target_max(n_targets, targets, edge_similarity, union5)
    max_union6 = _target_max(n_targets, targets, edge_similarity, union6)
    max_chembl5 = _target_max(n_targets, targets, edge_similarity, chembl5)
    max_chembl6 = _target_max(n_targets, targets, edge_similarity, chembl6)
    max_binding6 = _target_max(n_targets, targets, edge_similarity, binding6)
    max_gtopdb6 = _target_max(n_targets, targets, edge_similarity, gtopdb6)
    source_consensus6 = np.sort(
        np.vstack((max_chembl6, max_binding6, max_gtopdb6)), axis=0
    )[-2]

    potency = np.clip((reference.edge_max_pactivity - 6.0) / 3.0, 0.0, 1.0)
    replication = np.clip(np.log1p(reference.edge_positive_count) / np.log(5.0), 0.0, 1.0)
    quality_weight = 0.78 + 0.17 * potency + 0.05 * replication
    quality_union6 = _target_max(
        n_targets,
        targets,
        edge_similarity * quality_weight,
        union6,
    )
    support_signal = np.zeros(n_targets, dtype=np.float64)
    support_mask = union6 & (edge_similarity >= 0.30)
    if support_mask.any():
        support_raw = np.bincount(
            targets[support_mask],
            weights=(
                edge_similarity[support_mask] ** 4
                * np.log1p(reference.edge_positive_count[support_mask])
            ),
            minlength=n_targets,
        )
        support_signal = np.tanh(support_raw / 4.0)
    negative_max = _target_max(
        n_targets,
        targets,
        edge_similarity,
        negative_only,
    )

    features = {
        "baseline": max_chembl5,
        "max_union_any": max_union_any,
        "max_union_nonneg": max_union_nonneg,
        "max_union5": max_union5,
        "max_union6": max_union6,
        "quality_union6": quality_union6,
        "source_consensus6": source_consensus6,
        "support_union6": support_signal,
        "negative_contrast": np.maximum(negative_max - max_union6, 0.0),
    }
    return features, {
        "reference_ligands": len(reference.fingerprints),
        "excluded_reference_ligands": int((~eligible_ligand).sum()),
        "eligible_reference_ligands": int(eligible_ligand.sum()),
    }


def apply_recipe(features: dict[str, np.ndarray], recipe: Recipe) -> np.ndarray:
    if recipe.recipe_id == BASELINE.recipe_id:
        return features["baseline"].copy()
    score = (
        recipe.max_union_any * features["max_union_any"]
        + recipe.max_union_nonneg * features["max_union_nonneg"]
        + recipe.max_union5 * features["max_union5"]
        + recipe.max_union6 * features["max_union6"]
        + recipe.quality_union6 * features["quality_union6"]
        + recipe.source_consensus6 * features["source_consensus6"]
        + recipe.support_union6 * features["support_union6"]
        - recipe.negative_contrast * features["negative_contrast"]
    )
    return np.maximum(score, 0.0)


def average_tie_ranks(scores: np.ndarray) -> np.ndarray:
    """Average rank within each exact-score tie group.

    The frozen activity-retrieval evaluator and every ranking artifact it emits
    use this tie policy; callers outside `eval/` must use it too rather than
    `np.argsort`, which invents a distinct ordinal rank for every zero-score
    target and makes the metric depend on array order.
    """
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("ranking scores must be a finite one-dimensional array")
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        score = float(values[order[start]])
        while end < len(order) and float(values[order[end]]) == score:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def scorable_target_mask(reference: ReferenceIndex) -> np.ndarray:
    """Targets the retrieval model can score at all.

    ``_target_max`` zero-initialises every target and only fills the ones that
    carry an activity edge, and ``apply_recipe`` clamps at zero. A target with
    no edge is therefore indistinguishable from one the model scored and
    rejected: both read 0.0 and land in the tied tail. This mask separates the
    two so a ranking can say which targets were never in its competence.
    """
    mask = np.zeros(len(reference.target_ids), dtype=bool)
    if len(reference.edge_target):
        mask[np.asarray(reference.edge_target, dtype=np.int64)] = True
    return mask


