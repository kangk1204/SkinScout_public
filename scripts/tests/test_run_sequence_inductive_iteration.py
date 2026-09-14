from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import eval.run_sequence_inductive_iteration as iteration
from eval.activity_retrieval_model import Recipe
from eval.sequence_inductive_retrieval import TargetEmbeddingIndex, TargetNeighborIndex


sequence_gate = iteration.sequence_gate


def _ranking(top10: float, top30: float, mrr: float) -> dict[str, object]:
    return {
        "top10": top10,
        "top30": top30,
        "mrr": mrr,
        "target_macro": {"top10": top10, "top30": top30, "mrr": mrr},
    }


def _calibration(brier: float, log_loss: float) -> dict[str, object]:
    return {
        "brier": brier,
        "log_loss": log_loss,
        "unsupported_target": {
            "brier": brier,
            "log_loss": log_loss,
            "n_rows": 20,
            "n_positive": 10,
            "n_negative": 10,
        },
    }


def test_sequence_gate_requires_strict_improvement_everywhere() -> None:
    passes, failures, score = sequence_gate(
        _ranking(0.1, 0.2, 0.05),
        _ranking(0.2, 0.3, 0.08),
        _calibration(0.10, 0.30),
        _calibration(0.09, 0.29),
    )

    assert passes is True
    assert failures == []
    assert score > 0.0


def test_sequence_gate_rejects_tied_or_regressed_metric() -> None:
    passes, failures, _score = sequence_gate(
        _ranking(0.1, 0.2, 0.05),
        _ranking(0.1, 0.3, 0.08),
        _calibration(0.10, 0.30),
        _calibration(0.09, 0.31),
    )

    assert passes is False
    assert any("micro_top10_gain=0" in failure for failure in failures)
    assert any("target_macro_top10_gain=0" in failure for failure in failures)
    assert any("log_loss_improvement=" in failure for failure in failures)
    assert any("unsupported_log_loss_improvement=" in failure for failure in failures)


def _ranking_panel(target: str = "P2") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "query_id": "Q1",
                "truth_targets": [target],
                "panel_sources": ["activity_quantitative"],
                "source_databases": ["chembl"],
                "source_documents_json": '[{"source_db":"chembl","publication_key":"pmid:1"}]',
                "truth_target_panel_sources_json": json.dumps(
                    {target: ["activity_quantitative"]}
                ),
            }
        ]
    )


def _embedding_fixture() -> tuple[TargetEmbeddingIndex, TargetNeighborIndex]:
    target_ids = np.asarray(["P1", "P2"], dtype=object)
    embeddings = TargetEmbeddingIndex(
        target_ids=target_ids,
        matrix=np.eye(2, dtype=np.float32),
        train_supported=np.asarray([True, False]),
        manifest={},
    )
    neighbors = TargetNeighborIndex(
        target_ids=target_ids,
        source_indices=np.zeros((2, 1), dtype=np.int32),
        cosine_similarities=np.ones((2, 1), dtype=np.float32),
    )
    return embeddings, neighbors


def test_ranking_scores_preserve_source_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    embeddings, neighbors = _embedding_fixture()
    reference = type("Reference", (), {"target_ids": embeddings.target_ids})()
    monkeypatch.setattr(iteration, "_validate_query_identity", lambda _row: object())
    monkeypatch.setattr(iteration, "score_query_features", lambda *_args, **_kwargs: ({}, {}))
    monkeypatch.setattr(iteration, "apply_recipe", lambda *_args: np.asarray([0.8, 0.0]))
    monkeypatch.setattr(
        iteration,
        "sequence_fill_scores",
        lambda base, *_args: (base.copy(), {"source_targets": 1, "filled_targets": 0, "max_sequence_fill": 0.0}),
    )

    query_rows, target_rows = iteration.score_sequence_ranking_panel(
        reference,
        embeddings,
        neighbors,
        _ranking_panel(),
        Recipe("base"),
        exclude_reference_similarity=0.85,
    )

    assert set(query_rows["panel_sources"]) == {'["activity_quantitative"]'}
    assert set(target_rows["source_databases"]) == {'["chembl"]'}
    assert set(target_rows["target_panel_sources_json"]) == {
        '["activity_quantitative"]'
    }


def test_ranking_scores_reject_unknown_truth_target() -> None:
    embeddings, neighbors = _embedding_fixture()
    reference = type("Reference", (), {"target_ids": embeddings.target_ids})()

    with pytest.raises(SystemExit, match="targets outside universe"):
        iteration.score_sequence_ranking_panel(
            reference,
            embeddings,
            neighbors,
            _ranking_panel("P9"),
            Recipe("base"),
            exclude_reference_similarity=0.85,
        )


def test_ranking_scores_reject_train_supported_truth_target() -> None:
    embeddings, neighbors = _embedding_fixture()
    reference = type("Reference", (), {"target_ids": embeddings.target_ids})()

    with pytest.raises(SystemExit, match="contains a train-supported truth target"):
        iteration.score_sequence_ranking_panel(
            reference,
            embeddings,
            neighbors,
            _ranking_panel("P1"),
            Recipe("base"),
            exclude_reference_similarity=0.85,
        )


def _calibration_panel(targets: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pair_id": f"pair-{index}",
                "query_id": "Q1",
                "ligand_key": "L1",
                "standard_inchikey": "I1",
                "connectivity_key": "C1",
                "canonical_smiles": "C",
                "standardization_route": "standard_inchikey",
                "uniprot": target,
                "endpoint_family": "direct_binding",
                "label": index % 2,
                "sample_weight": 1.0,
            }
            for index, target in enumerate(targets)
        ]
    )


def test_calibration_scores_reject_unknown_target() -> None:
    embeddings, neighbors = _embedding_fixture()
    reference = type("Reference", (), {"target_ids": embeddings.target_ids})()

    with pytest.raises(SystemExit, match="targets outside universe"):
        iteration.score_sequence_calibration_panel(
            reference,
            embeddings,
            neighbors,
            _calibration_panel(["P9"]),
            Recipe("base"),
            exclude_reference_similarity=0.85,
        )


def test_calibration_scores_preserve_supported_target_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embeddings, neighbors = _embedding_fixture()
    reference = type("Reference", (), {"target_ids": embeddings.target_ids})()
    monkeypatch.setattr(iteration, "_validate_query_identity", lambda _row: object())
    monkeypatch.setattr(
        iteration, "score_query_features", lambda *_args, **_kwargs: ({}, {})
    )
    monkeypatch.setattr(
        iteration, "apply_recipe", lambda *_args: np.asarray([0.8, 0.0])
    )
    monkeypatch.setattr(
        iteration,
        "sequence_fill_scores",
        lambda base, *_args: (
            base.copy(),
            {"source_targets": 1, "filled_targets": 0, "max_sequence_fill": 0.0},
        ),
    )

    scores = iteration.score_sequence_calibration_panel(
        reference,
        embeddings,
        neighbors,
        _calibration_panel(["P1", "P2"]),
        Recipe("base"),
        exclude_reference_similarity=0.85,
        sequence_recipes=(iteration.SEQUENCE_BASELINE,),
    )

    assert scores.set_index("uniprot")["target_train_supported"].to_dict() == {
        "P1": True,
        "P2": False,
    }


@pytest.mark.parametrize(
    ("key", "value", "failure"),
    [
        ("n_rows", 19, "unsupported_calibration_n_rows=19"),
        ("n_positive", 4, "unsupported_calibration_n_positive=4"),
        ("n_negative", 4, "unsupported_calibration_n_negative=4"),
    ],
)
def test_sequence_gate_requires_cold_calibration_adequacy(
    key: str, value: int, failure: str
) -> None:
    baseline = _calibration(0.10, 0.30)
    candidate = _calibration(0.09, 0.29)
    baseline["unsupported_target"][key] = value

    passes, failures, _score = sequence_gate(
        _ranking(0.1, 0.2, 0.05),
        _ranking(0.2, 0.3, 0.08),
        baseline,
        candidate,
    )

    assert passes is False
    assert any(failure in observed for observed in failures)


def test_output_alias_guard_preserves_input(tmp_path: Path) -> None:
    protected = tmp_path / "dev-ranking.parquet"
    protected.write_bytes(b"bound input")

    with pytest.raises(SystemExit, match="aliases a protected input"):
        iteration._assert_safe_output_files(
            input_paths=(protected,),
            output_paths=(protected,),
        )

    assert protected.read_bytes() == b"bound input"
