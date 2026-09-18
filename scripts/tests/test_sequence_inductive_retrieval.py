from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eval.sequence_inductive_retrieval import (
    SEQUENCE_BASELINE,
    SequenceTransferRecipe,
    TargetEmbeddingIndex,
    build_target_neighbor_index,
    load_target_embedding_index,
    prepare_embedding_matrix,
    sequence_fill_scores,
    train_supported_targets,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _index() -> TargetEmbeddingIndex:
    raw = np.asarray(
        [
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.9, 0.1, 0.2],
            [0.1, 1.9, -0.2],
        ],
        dtype=np.float32,
    )
    supported = np.asarray([True, True, False, False])
    return TargetEmbeddingIndex(
        target_ids=np.asarray(["P1", "P2", "P3", "P4"], dtype=object),
        matrix=prepare_embedding_matrix(raw, supported),
        train_supported=supported,
        manifest={},
    )


def test_sequence_fill_changes_only_unsupported_targets() -> None:
    index = _index()
    neighbors = build_target_neighbor_index(index, max_neighbors=2, block_size=2)
    base = np.asarray([0.8, 0.1, 0.0, 0.0])
    recipe = SequenceTransferRecipe("test", 1, 2.0, 1.0)

    augmented, diagnostics = sequence_fill_scores(base, index, neighbors, recipe)

    assert augmented[:2].tolist() == base[:2].tolist()
    assert augmented[2] > augmented[3]
    assert augmented[2] > 0.0
    assert diagnostics["source_targets"] == 2
    assert diagnostics["filled_targets"] >= 1


def test_sequence_baseline_is_exact_noop() -> None:
    base = np.asarray([0.8, 0.1, 0.0, 0.0])
    index = _index()
    neighbors = build_target_neighbor_index(index, max_neighbors=2, block_size=2)
    augmented, diagnostics = sequence_fill_scores(
        base, index, neighbors, SEQUENCE_BASELINE
    )

    assert np.array_equal(augmented, base)
    assert diagnostics == {
        "source_targets": 0,
        "filled_targets": 0,
        "max_sequence_fill": 0.0,
    }


def test_sequence_fill_is_deterministic_across_score_ties() -> None:
    index = _index()
    neighbors = build_target_neighbor_index(index, max_neighbors=2, block_size=2)
    base = np.asarray([0.7, 0.7, 0.0, 0.0])
    recipe = SequenceTransferRecipe("test", 1, 2.0, 1.0)

    first, _ = sequence_fill_scores(base, index, neighbors, recipe)
    second, _ = sequence_fill_scores(base, index, neighbors, recipe)

    assert np.array_equal(first, second)
    assert first[2] == pytest.approx(first[3])


def test_sequence_fill_weighting_is_stable_for_tiny_scores() -> None:
    base = np.asarray([1e-80, 5e-81, 0.0, 0.0])
    recipe = SequenceTransferRecipe("tiny", 2, 8.0, 1.0)
    index = _index()
    neighbors = build_target_neighbor_index(index, max_neighbors=2, block_size=2)

    augmented, diagnostics = sequence_fill_scores(base, index, neighbors, recipe)

    assert np.isfinite(augmented).all()
    assert diagnostics["source_targets"] == 2


def test_prepare_embeddings_requires_both_support_classes() -> None:
    raw = np.eye(2, dtype=np.float32)
    with pytest.raises(ValueError, match="supported and unsupported"):
        prepare_embedding_matrix(raw, np.asarray([True, True]))


def test_train_supported_targets_uses_positive_measurements_only() -> None:
    supported = train_supported_targets(
        4,
        np.asarray([0, 1, 1, 3]),
        np.asarray([1.0, 0.0, 2.0, 0.0]),
    )
    assert supported.tolist() == [True, True, False, False]


def _write_embedding_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, np.ndarray]:
    targets = tmp_path / "targets.csv"
    pd.DataFrame(
        {
            "uniprot": ["P1", "P2", "P3", "P4"],
            "target_cluster_30": ["C1", "C2", "C3", "C4"],
            "target_cluster_50": ["D1", "D2", "D3", "D4"],
        }
    ).to_csv(targets, index=False)
    target_manifest = tmp_path / "targets.manifest.json"
    target_manifest.write_text('{"schema_version":"fixture"}\n')
    embeddings = tmp_path / "embeddings.parquet"
    pd.DataFrame(
        {
            "uniprot": ["P1", "P2", "P3", "P4"],
            "embedding": [
                np.asarray([2.0, 0.0, 0.0], dtype=np.float32),
                np.asarray([0.0, 2.0, 0.0], dtype=np.float32),
                np.asarray([1.9, 0.1, 0.2], dtype=np.float32),
                np.asarray([0.1, 1.9, -0.2], dtype=np.float32),
            ],
        }
    ).to_parquet(embeddings, index=False)
    manifest = tmp_path / "embeddings.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "skinscout.target-sequence-embeddings.v1",
                "contract": {
                    "evaluation_panel_used": False,
                    "known_target_assistance": False,
                    "training_labels_used": False,
                },
                "model": {
                    "name": "esm2_t30_150M_UR50D",
                    "representation_layer": 30,
                    "checkpoint_sha256": "a" * 64,
                    "fair_esm_version": "fixture",
                    "torch_version": "fixture",
                },
                "embedding": {
                    "dimension": 3,
                    "dtype": "float32",
                    "pooling": "mean residue pooling weighted by chunk residue counts",
                    "chunk_policy": {
                        "max_chunk_residues": 1022,
                        "max_model_residues_per_chunk": 1022,
                        "overlap": 0,
                        "deterministic_non_overlapping": True,
                        "tokens_per_batch": 4096,
                    },
                },
                "universe_policy": {
                    "evaluation_panel_used": False,
                    "known_target_assistance": False,
                    "production_contract_passes": True,
                },
                "inputs": {
                    "target_clusters": {
                        "path": str(targets.resolve()),
                        "sha256": _sha256(targets),
                        "rows": 4,
                    },
                    "target_cluster_manifest": {
                        "path": str(target_manifest.resolve()),
                        "sha256": _sha256(target_manifest),
                    },
                },
                "outputs": {
                    "embeddings": {
                        "path": str(embeddings.resolve()),
                        "sha256": _sha256(embeddings),
                        "rows": 4,
                        "dimension": 3,
                    }
                },
            }
        )
        + "\n"
    )
    return targets, target_manifest, embeddings, manifest, np.asarray(
        ["P1", "P2", "P3", "P4"], dtype=object
    )


def test_load_embedding_index_checks_bound_artifacts(tmp_path: Path) -> None:
    targets, target_manifest, embeddings, manifest, target_ids = (
        _write_embedding_fixture(tmp_path)
    )
    result = load_target_embedding_index(
        embeddings_path=embeddings,
        embeddings_manifest_path=manifest,
        target_ids=target_ids,
        target_clusters_path=targets,
        target_cluster_manifest_path=target_manifest,
        train_supported=np.asarray([True, True, False, False]),
    )

    assert result.matrix.shape == (4, 3)
    assert np.allclose(np.linalg.norm(result.matrix, axis=1), 1.0)


def test_load_embedding_index_rejects_missing_model_provenance(tmp_path: Path) -> None:
    targets, target_manifest, embeddings, manifest, target_ids = _write_embedding_fixture(
        tmp_path
    )
    payload = json.loads(manifest.read_text())
    payload.pop("model")
    manifest.write_text(json.dumps(payload) + "\n")

    with pytest.raises(SystemExit, match="model provenance is missing"):
        load_target_embedding_index(
            embeddings_path=embeddings,
            embeddings_manifest_path=manifest,
            target_ids=target_ids,
            target_clusters_path=targets,
            target_cluster_manifest_path=target_manifest,
            train_supported=np.asarray([True, True, False, False]),
        )


def test_load_embedding_index_rejects_stale_parquet(tmp_path: Path) -> None:
    targets, target_manifest, embeddings, manifest, target_ids = (
        _write_embedding_fixture(tmp_path)
    )
    frame = pd.read_parquet(embeddings)
    frame.loc[0, "uniprot"] = "STALE"
    frame.to_parquet(embeddings, index=False)

    with pytest.raises(SystemExit, match="sha256 mismatch"):
        load_target_embedding_index(
            embeddings_path=embeddings,
            embeddings_manifest_path=manifest,
            target_ids=target_ids,
            target_clusters_path=targets,
            target_cluster_manifest_path=target_manifest,
            train_supported=np.asarray([True, True, False, False]),
        )


@pytest.mark.parametrize("contract_key", ["evaluation_panel_used", "known_target_assistance"])
def test_load_embedding_index_rejects_assistance_contracts(
    tmp_path: Path, contract_key: str
) -> None:
    targets, target_manifest, embeddings, manifest, target_ids = _write_embedding_fixture(
        tmp_path
    )
    payload = json.loads(manifest.read_text())
    payload["contract"][contract_key] = True
    manifest.write_text(json.dumps(payload) + "\n")

    with pytest.raises(SystemExit, match="usage contract is invalid"):
        load_target_embedding_index(
            embeddings_path=embeddings,
            embeddings_manifest_path=manifest,
            target_ids=target_ids,
            target_clusters_path=targets,
            target_cluster_manifest_path=target_manifest,
            train_supported=np.asarray([True, True, False, False]),
        )


def test_load_embedding_index_rejects_stale_target_manifest(tmp_path: Path) -> None:
    targets, target_manifest, embeddings, manifest, target_ids = _write_embedding_fixture(
        tmp_path
    )
    target_manifest.write_text('{"schema_version":"changed"}\n')

    with pytest.raises(SystemExit, match="target manifest sha256 mismatch"):
        load_target_embedding_index(
            embeddings_path=embeddings,
            embeddings_manifest_path=manifest,
            target_ids=target_ids,
            target_clusters_path=targets,
            target_cluster_manifest_path=target_manifest,
            train_supported=np.asarray([True, True, False, False]),
        )


def test_load_embedding_index_rejects_row_count_mismatch(tmp_path: Path) -> None:
    targets, target_manifest, embeddings, manifest, target_ids = _write_embedding_fixture(
        tmp_path
    )
    payload = json.loads(manifest.read_text())
    payload["outputs"]["embeddings"]["rows"] = 3
    manifest.write_text(json.dumps(payload) + "\n")

    with pytest.raises(SystemExit, match="target embeddings row count mismatch"):
        load_target_embedding_index(
            embeddings_path=embeddings,
            embeddings_manifest_path=manifest,
            target_ids=target_ids,
            target_clusters_path=targets,
            target_cluster_manifest_path=target_manifest,
            train_supported=np.asarray([True, True, False, False]),
        )


def test_load_embedding_index_rejects_target_order_mismatch(tmp_path: Path) -> None:
    targets, target_manifest, embeddings, manifest, target_ids = _write_embedding_fixture(
        tmp_path
    )
    target_ids[[0, 1]] = target_ids[[1, 0]]

    with pytest.raises(SystemExit, match="target embedding order"):
        load_target_embedding_index(
            embeddings_path=embeddings,
            embeddings_manifest_path=manifest,
            target_ids=target_ids,
            target_clusters_path=targets,
            target_cluster_manifest_path=target_manifest,
            train_supported=np.asarray([True, True, False, False]),
        )
