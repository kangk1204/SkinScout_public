#!/usr/bin/env python3
"""Target-sequence transfer primitives for diagnostic cold-target retrieval.

This module does not use evaluation truth or a known-target prior. It transfers
train-supported ligand-analogy scores into targets without positive training
evidence through provenance-bound protein embeddings. The existing score is
never changed for train-supported targets.
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


EMBEDDING_SCHEMA = "skinscout.target-sequence-embeddings.v1"


@dataclass(frozen=True)
class SequenceTransferRecipe:
    recipe_id: str
    neighbor_top_k: int
    similarity_power: float
    fill_weight: float


SEQUENCE_BASELINE = SequenceTransferRecipe(
    recipe_id="analog_only",
    neighbor_top_k=32,
    similarity_power=2.0,
    fill_weight=0.0,
)
SEQUENCE_CANDIDATES = (
    SequenceTransferRecipe("esm6_knn_k8_p2_w050", 8, 2.0, 0.50),
    SequenceTransferRecipe("esm6_knn_k16_p2_w075", 16, 2.0, 0.75),
    SequenceTransferRecipe("esm6_knn_k32_p2_w075", 32, 2.0, 0.75),
    SequenceTransferRecipe("esm6_knn_k32_p4_w075", 32, 4.0, 0.75),
    SequenceTransferRecipe("esm6_knn_k64_p4_w100", 64, 4.0, 1.00),
)


@dataclass(frozen=True)
class TargetEmbeddingIndex:
    target_ids: np.ndarray
    matrix: np.ndarray
    train_supported: np.ndarray
    manifest: dict[str, Any]


@dataclass(frozen=True)
class TargetNeighborIndex:
    target_ids: np.ndarray
    source_indices: np.ndarray
    cosine_similarities: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"{label} is required and must be non-empty: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Unable to parse {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} root must be an object: {path}")
    return payload


def _resolve_bound_path(
    raw_path: object,
    *,
    manifest_path: Path,
    expected_path: Path,
    label: str,
) -> None:
    text = str(raw_path or "").strip()
    if not text:
        raise SystemExit(f"target embedding manifest missing {label}.path")
    recorded = Path(text)
    if not recorded.is_absolute():
        recorded = manifest_path.parent / recorded
    if recorded.resolve() != expected_path.resolve():
        raise SystemExit(f"target embedding manifest {label}.path mismatch")


def _embedding_matrix(frame: pd.DataFrame) -> np.ndarray:
    if list(frame.columns) != ["uniprot", "embedding"]:
        raise SystemExit("target embedding parquet columns must be uniprot, embedding")
    ids = frame["uniprot"].fillna("").astype(str).str.strip()
    if ids.eq("").any() or ids.duplicated().any():
        raise SystemExit("target embedding parquet requires unique nonblank uniprot values")
    rows: list[np.ndarray] = []
    dimension: int | None = None
    for index, value in enumerate(frame["embedding"]):
        try:
            row = np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"invalid target embedding at row {index}") from exc
        if row.ndim != 1 or row.size == 0:
            raise SystemExit(f"target embedding at row {index} must be one-dimensional")
        if dimension is None:
            dimension = int(row.size)
        elif row.size != dimension:
            raise SystemExit("target embedding dimensions are inconsistent")
        if not np.isfinite(row).all() or float(np.linalg.norm(row)) <= 0.0:
            raise SystemExit(f"target embedding at row {index} must be finite and nonzero")
        rows.append(row)
    if not rows:
        raise SystemExit("target embedding parquet is empty")
    return np.stack(rows).astype(np.float32, copy=False)


def prepare_embedding_matrix(
    raw_matrix: np.ndarray,
    train_supported: np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(raw_matrix, dtype=np.float32)
    supported = np.asarray(train_supported, dtype=bool)
    if matrix.ndim != 2 or matrix.shape[0] != supported.size or matrix.shape[1] < 1:
        raise ValueError("embedding matrix and train-supported mask are not aligned")
    if not np.isfinite(matrix).all():
        raise ValueError("embedding matrix must be finite")
    if not supported.any() or supported.all():
        raise ValueError("sequence transfer requires supported and unsupported targets")
    centered = matrix - matrix[supported].mean(axis=0, dtype=np.float64).astype(
        np.float32
    )
    norms = np.linalg.norm(centered, axis=1)
    if not np.isfinite(norms).all() or (norms <= 0.0).any():
        raise ValueError("centered target embeddings must have finite nonzero norms")
    return (centered / norms[:, None]).astype(np.float32, copy=False)


def load_target_embedding_index(
    *,
    embeddings_path: Path,
    embeddings_manifest_path: Path,
    target_ids: np.ndarray,
    target_clusters_path: Path,
    target_cluster_manifest_path: Path,
    train_supported: np.ndarray,
) -> TargetEmbeddingIndex:
    manifest = _read_json(
        embeddings_manifest_path,
        "target sequence embedding manifest",
    )
    if manifest.get("schema_version") != EMBEDDING_SCHEMA:
        raise SystemExit(
            f"target embedding manifest schema must be {EMBEDDING_SCHEMA}"
        )
    contract = manifest.get("contract")
    required_contract = {
        "evaluation_panel_used": False,
        "known_target_assistance": False,
        "training_labels_used": False,
    }
    if not isinstance(contract, dict) or any(
        contract.get(key) is not expected
        for key, expected in required_contract.items()
    ):
        raise SystemExit("target embedding manifest usage contract is invalid")
    model = manifest.get("model")
    if not isinstance(model, dict):
        raise SystemExit("target embedding manifest model provenance is missing")
    if model.get("name") != "esm2_t30_150M_UR50D":
        raise SystemExit("target embedding manifest model name is not preregistered ESM-2")
    if int(model.get("representation_layer", -1)) != 30:
        raise SystemExit("target embedding manifest representation layer must be 30")
    checkpoint_sha256 = str(model.get("checkpoint_sha256") or "")
    if len(checkpoint_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in checkpoint_sha256.lower()
    ):
        raise SystemExit("target embedding manifest checkpoint SHA-256 is invalid")
    for version_key in ("fair_esm_version", "torch_version"):
        if not str(model.get(version_key) or "").strip():
            raise SystemExit(f"target embedding manifest {version_key} is missing")
    embedding_policy = manifest.get("embedding")
    if not isinstance(embedding_policy, dict):
        raise SystemExit("target embedding manifest embedding policy is missing")
    if embedding_policy.get("dtype") != "float32" or embedding_policy.get("pooling") != (
        "mean residue pooling weighted by chunk residue counts"
    ):
        raise SystemExit("target embedding manifest dtype/pooling policy is invalid")
    chunk_policy = embedding_policy.get("chunk_policy")
    if (
        not isinstance(chunk_policy, dict)
        or int(chunk_policy.get("max_chunk_residues", 0)) < 1
        or int(chunk_policy.get("max_model_residues_per_chunk", -1)) != 1022
        or int(chunk_policy.get("overlap", -1)) != 0
        or chunk_policy.get("deterministic_non_overlapping") is not True
        or int(chunk_policy.get("tokens_per_batch", 0)) < 1
    ):
        raise SystemExit("target embedding manifest chunk policy is invalid")
    universe_policy = manifest.get("universe_policy")
    if (
        not isinstance(universe_policy, dict)
        or universe_policy.get("evaluation_panel_used") is not False
        or universe_policy.get("known_target_assistance") is not False
        or universe_policy.get("production_contract_passes") is not True
    ):
        raise SystemExit("target embedding manifest universe policy is invalid")
    inputs = manifest.get("inputs")
    outputs = manifest.get("outputs")
    target_record = inputs.get("target_clusters") if isinstance(inputs, dict) else None
    target_manifest_record = (
        inputs.get("target_cluster_manifest") if isinstance(inputs, dict) else None
    )
    embedding_record = outputs.get("embeddings") if isinstance(outputs, dict) else None
    if not all(
        isinstance(record, dict)
        for record in (target_record, target_manifest_record, embedding_record)
    ):
        raise SystemExit("target embedding manifest artifact bindings are incomplete")
    _resolve_bound_path(
        target_record.get("path"),
        manifest_path=embeddings_manifest_path,
        expected_path=target_clusters_path,
        label="inputs.target_clusters",
    )
    _resolve_bound_path(
        target_manifest_record.get("path"),
        manifest_path=embeddings_manifest_path,
        expected_path=target_cluster_manifest_path,
        label="inputs.target_cluster_manifest",
    )
    _resolve_bound_path(
        embedding_record.get("path"),
        manifest_path=embeddings_manifest_path,
        expected_path=embeddings_path,
        label="outputs.embeddings",
    )
    expected_ids = np.asarray(target_ids, dtype=object)
    expected_count = int(expected_ids.size)
    bindings = (
        (target_record, target_clusters_path, expected_count, "target clusters"),
        (target_manifest_record, target_cluster_manifest_path, None, "target manifest"),
        (embedding_record, embeddings_path, expected_count, "target embeddings"),
    )
    for record, path, rows, label in bindings:
        if not path.exists() or path.stat().st_size == 0:
            raise SystemExit(f"{label} artifact is missing or empty: {path}")
        if record.get("sha256") != _sha256(path):
            raise SystemExit(f"target embedding manifest {label} sha256 mismatch")
        if rows is not None and int(record.get("rows", -1)) != rows:
            raise SystemExit(f"target embedding manifest {label} row count mismatch")
    frame = pd.read_parquet(embeddings_path)
    raw_matrix = _embedding_matrix(frame)
    observed_ids = frame["uniprot"].astype(str).to_numpy(object)
    if not np.array_equal(observed_ids, expected_ids):
        raise SystemExit("target embedding order does not match the target universe")
    if int(embedding_record.get("dimension", -1)) != raw_matrix.shape[1]:
        raise SystemExit("target embedding manifest dimension mismatch")
    if int(embedding_policy.get("dimension", -1)) != raw_matrix.shape[1]:
        raise SystemExit("target embedding policy dimension mismatch")
    prepared = prepare_embedding_matrix(raw_matrix, train_supported)
    return TargetEmbeddingIndex(
        target_ids=expected_ids,
        matrix=prepared,
        train_supported=np.asarray(train_supported, dtype=bool),
        manifest=manifest,
    )


def build_target_neighbor_index(
    embeddings: TargetEmbeddingIndex,
    *,
    max_neighbors: int,
    block_size: int = 512,
) -> TargetNeighborIndex:
    if max_neighbors < 1 or block_size < 1:
        raise ValueError("max_neighbors and block_size must be positive")
    supported_indices = np.flatnonzero(embeddings.train_supported)
    if supported_indices.size < max_neighbors:
        raise ValueError("max_neighbors exceeds the number of train-supported targets")
    source_matrix = embeddings.matrix[supported_indices]
    source_ids = embeddings.target_ids[supported_indices].astype(str)
    n_targets = embeddings.target_ids.size
    neighbor_indices = np.empty((n_targets, max_neighbors), dtype=np.int32)
    neighbor_cosines = np.empty((n_targets, max_neighbors), dtype=np.float32)
    for start in range(0, n_targets, block_size):
        stop = min(start + block_size, n_targets)
        cosine = embeddings.matrix[start:stop] @ source_matrix.T
        selected = np.argpartition(cosine, -max_neighbors, axis=1)[:, -max_neighbors:]
        for row_offset in range(stop - start):
            local = selected[row_offset]
            values = cosine[row_offset, local]
            order = np.lexsort((source_ids[local], -values))
            ordered_local = local[order]
            neighbor_indices[start + row_offset] = supported_indices[ordered_local]
            neighbor_cosines[start + row_offset] = np.clip(
                cosine[row_offset, ordered_local], 0.0, 1.0
            )
    return TargetNeighborIndex(
        target_ids=embeddings.target_ids.copy(),
        source_indices=neighbor_indices,
        cosine_similarities=neighbor_cosines,
    )


def sequence_fill_scores(
    base_scores: np.ndarray,
    embeddings: TargetEmbeddingIndex,
    neighbors: TargetNeighborIndex,
    recipe: SequenceTransferRecipe,
) -> tuple[np.ndarray, dict[str, Any]]:
    scores = np.asarray(base_scores, dtype=np.float64)
    if scores.ndim != 1 or scores.size != embeddings.target_ids.size:
        raise ValueError("base scores must align with the target embedding universe")
    if not np.isfinite(scores).all() or (scores < 0.0).any():
        raise ValueError("base scores must be finite and nonnegative")
    if recipe.neighbor_top_k < 1:
        raise ValueError("neighbor_top_k must be positive")
    if not math.isfinite(recipe.similarity_power) or recipe.similarity_power <= 0.0:
        raise ValueError("similarity_power must be finite and positive")
    if not math.isfinite(recipe.fill_weight) or recipe.fill_weight < 0.0:
        raise ValueError("fill_weight must be finite and nonnegative")
    if recipe.fill_weight == 0.0:
        return scores.copy(), {
            "source_targets": 0,
            "filled_targets": 0,
            "max_sequence_fill": 0.0,
        }
    if not np.array_equal(neighbors.target_ids, embeddings.target_ids):
        raise ValueError("target neighbor index is not aligned with target embeddings")
    if neighbors.source_indices.shape != neighbors.cosine_similarities.shape or (
        neighbors.source_indices.ndim != 2
        or neighbors.source_indices.shape[0] != scores.size
    ):
        raise ValueError("target neighbor arrays are not aligned")
    if recipe.neighbor_top_k > neighbors.source_indices.shape[1]:
        raise ValueError("neighbor_top_k is outside the precomputed neighbor index")
    source_count = int((embeddings.train_supported & (scores > 0.0)).sum())
    if source_count == 0:
        return scores.copy(), {
            "source_targets": 0,
            "filled_targets": 0,
            "max_sequence_fill": 0.0,
        }
    source_indices = neighbors.source_indices[:, : recipe.neighbor_top_k]
    cosine = neighbors.cosine_similarities[:, : recipe.neighbor_top_k].astype(
        np.float64, copy=False
    )
    if source_indices.size and (
        source_indices.min() < 0 or source_indices.max() >= scores.size
    ):
        raise ValueError("target neighbor index contains out-of-range sources")
    if not embeddings.train_supported[source_indices].all():
        raise ValueError("target neighbor index contains unsupported source targets")
    transfer = np.max(
        scores[source_indices] * np.power(cosine, recipe.similarity_power),
        axis=1,
    ) * recipe.fill_weight
    transfer[embeddings.train_supported] = 0.0
    augmented = np.maximum(scores, transfer)
    filled = (~embeddings.train_supported) & (augmented > scores)
    return augmented, {
        "source_targets": source_count,
        "filled_targets": int(filled.sum()),
        "max_sequence_fill": float(transfer.max(initial=0.0)),
    }


def train_supported_targets(
    target_count: int,
    edge_target: np.ndarray,
    edge_positive_count: np.ndarray,
) -> np.ndarray:
    indexes = np.asarray(edge_target, dtype=np.int64)
    positives = np.asarray(edge_positive_count, dtype=np.float64)
    if indexes.shape != positives.shape:
        raise ValueError("edge target and positive-count arrays must align")
    if target_count < 1 or (indexes.size and (indexes.min() < 0 or indexes.max() >= target_count)):
        raise ValueError("edge target index is outside the target universe")
    if not np.isfinite(positives).all() or (positives < 0.0).any():
        raise ValueError("edge positive counts must be finite and nonnegative")
    supported = np.zeros(target_count, dtype=bool)
    supported[indexes[positives > 0.0]] = True
    return supported
