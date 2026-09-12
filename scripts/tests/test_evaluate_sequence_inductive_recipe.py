from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import eval.evaluate_sequence_inductive_recipe as evaluator
from eval.activity_retrieval_model import Recipe
from eval.sequence_inductive_retrieval import (
    SEQUENCE_BASELINE,
    SEQUENCE_CANDIDATES,
    SequenceTransferRecipe,
    TargetEmbeddingIndex,
    TargetNeighborIndex,
)


SELECTED = SequenceTransferRecipe("selected", 1, 2.0, 0.75)


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


def _panel(targets: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "query_id": "Q1",
                "truth_targets": targets,
                "canonical_smiles": "C",
                "panel_sources": ["activity_quantitative"],
                "source_databases": ["chembl"],
                "source_documents_json": '[{"publication_key":"pmid:1"}]',
                "truth_target_panel_sources_json": json.dumps(
                    {target: ["activity_quantitative"] for target in targets}
                ),
            }
        ]
    )


def _patch_scoring(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluator, "_validate_query_identity", lambda _row: object())
    monkeypatch.setattr(
        evaluator, "score_query_features", lambda *_args, **_kwargs: ({}, {"eligible": 1})
    )
    monkeypatch.setattr(
        evaluator, "apply_recipe", lambda *_args: np.asarray([0.8, 0.0])
    )

    def fill(
        base: np.ndarray,
        _embeddings: TargetEmbeddingIndex,
        _neighbors: TargetNeighborIndex,
        recipe: SequenceTransferRecipe,
    ) -> tuple[np.ndarray, dict[str, float]]:
        scores = base.copy()
        if recipe.fill_weight:
            scores[1] = 0.9
        return scores, {
            "source_targets": 1,
            "filled_targets": int(recipe.fill_weight > 0),
            "max_sequence_fill": float(scores[1]),
        }

    monkeypatch.setattr(evaluator, "sequence_fill_scores", fill)


def test_posthoc_ranking_allows_mixed_temporal_truth_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_scoring(monkeypatch)
    embeddings, neighbors = _embedding_fixture()
    reference = type("Reference", (), {"target_ids": embeddings.target_ids})()

    query, targets = evaluator.score_posthoc_ranking_panel(
        reference,
        embeddings,
        neighbors,
        _panel(["P1", "P2"]),
        Recipe("base"),
        (SEQUENCE_BASELINE, SELECTED),
        exclude_reference_similarity=0.85,
        require_all_truth_targets_unsupported=False,
    )

    assert len(query) == 2
    assert set(query["n_supported_truth_targets"]) == {1}
    assert set(query["n_unsupported_truth_targets"]) == {1}
    assert set(targets.loc[targets["target_id"].eq("P1"), "target_train_supported"]) == {
        True
    }
    assert set(targets.loc[targets["target_id"].eq("P2"), "target_train_supported"]) == {
        False
    }
    assert set(targets["target_panel_sources_json"]) == {
        '["activity_quantitative"]'
    }


def test_posthoc_dual_cold_rejects_supported_truth_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_scoring(monkeypatch)
    embeddings, neighbors = _embedding_fixture()
    reference = type("Reference", (), {"target_ids": embeddings.target_ids})()

    with pytest.raises(SystemExit, match="contains a train-supported truth target"):
        evaluator.score_posthoc_ranking_panel(
            reference,
            embeddings,
            neighbors,
            _panel(["P1"]),
            Recipe("base"),
            (SEQUENCE_BASELINE, SELECTED),
            exclude_reference_similarity=0.85,
            require_all_truth_targets_unsupported=True,
        )


def test_posthoc_ranking_rejects_unknown_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_scoring(monkeypatch)
    embeddings, neighbors = _embedding_fixture()
    reference = type("Reference", (), {"target_ids": embeddings.target_ids})()

    with pytest.raises(SystemExit, match="targets outside universe"):
        evaluator.score_posthoc_ranking_panel(
            reference,
            embeddings,
            neighbors,
            _panel(["P9"]),
            Recipe("base"),
            (SEQUENCE_BASELINE, SELECTED),
            exclude_reference_similarity=0.85,
            require_all_truth_targets_unsupported=False,
        )


def test_posthoc_ranking_marks_missing_source_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_scoring(monkeypatch)
    embeddings, neighbors = _embedding_fixture()
    reference = type("Reference", (), {"target_ids": embeddings.target_ids})()
    panel = _panel(["P2"]).drop(columns=sorted(evaluator.SOURCE_PROVENANCE_COLUMNS))

    query, targets = evaluator.score_posthoc_ranking_panel(
        reference,
        embeddings,
        neighbors,
        panel,
        Recipe("base"),
        (SEQUENCE_BASELINE, SELECTED),
        exclude_reference_similarity=0.85,
        require_all_truth_targets_unsupported=False,
    )

    assert set(query["source_provenance_available"]) == {False}
    assert set(targets["source_provenance_available"]) == {False}
    assert "target_panel_sources_json" not in targets


def test_test_calibration_keeps_aggregate_and_cold_calibrator_boundaries() -> None:
    scored = pd.DataFrame(
        {
            "pair_id": ["a", "b", "c", "d"],
            "endpoint_family": ["direct_binding"] * 4,
            "label": [1, 0, 1, 0],
            "sample_weight": [1.0] * 4,
            "target_train_supported": [True, True, False, False],
            "score__analog_only": [0.9, 0.1, 0.8, 0.2],
            "score__selected": [0.9, 0.1, 0.85, 0.15],
        }
    )
    calibrator = lambda slope, intercept: {  # noqa: E731
        "slope": slope,
        "intercept": intercept,
        "n_rows": 4,
        "weight_sum": 4.0,
    }
    calibrators = {
        "baseline": {
            "aggregate": calibrator(1.0, 0.0),
            "unsupported_target": calibrator(2.0, -1.0),
        },
        "selected": {
            "aggregate": calibrator(3.0, -1.0),
            "unsupported_target": calibrator(4.0, -2.0),
        },
    }

    metrics, output = evaluator.evaluate_test_calibration(
        scored, (SEQUENCE_BASELINE, SELECTED), calibrators
    )

    assert metrics["analog_only"]["aggregate"]["n_rows"] == 4
    assert metrics["analog_only"]["unsupported_target"]["n_rows"] == 2
    assert output.loc[:1, "probability_unsupported_target__analog_only"].isna().all()
    assert output.loc[2:, "probability_unsupported_target__analog_only"].notna().all()
    assert not np.allclose(
        output["probability_aggregate__selected"],
        output["probability_aggregate__analog_only"],
    )


def test_posthoc_contract_forbids_prospective_claim_and_recipe_promotion() -> None:
    assert evaluator.POST_HOC_CONTRACT == {
        "evaluation_status": "post_hoc_diagnostic_only",
        "panels_previously_inspected_before_sequence_model_design": True,
        "panels_used_for_sequence_recipe_selection": False,
        "panel_truth_passed_to_scorer": False,
        "prospective_confirmation_claim_permitted": False,
        "recipe_promotion_permitted_from_this_evaluation": False,
        "development_gate_may_be_retroactively_changed": False,
    }


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _selection_fixture(tmp_path: Path) -> dict[str, Path]:
    paths = {
        name: tmp_path / name
        for name in (
            "index.json",
            "targets.json",
            "panels.json",
            "base_recipe.json",
            "dev_cold_manifest.json",
            "dev_cold_calibration.parquet",
            "embedding_manifest.json",
        )
    }
    for path in paths.values():
        path.write_bytes(b"bound-artifact")
    selected = SEQUENCE_CANDIDATES[0]
    calibrator = {
        "slope": 1.0,
        "intercept": 0.0,
        "n_rows": 20,
        "weight_sum": 20.0,
    }
    provenance_keys = {
        "index_manifest": "index.json",
        "target_cluster_manifest": "targets.json",
        "regular_panels_manifest": "panels.json",
        "base_recipe": "base_recipe.json",
        "sequence_dev_panel_manifest": "dev_cold_manifest.json",
        "sequence_dev_cold_calibration": "dev_cold_calibration.parquet",
        "embedding_manifest": "embedding_manifest.json",
    }
    recipe_path = tmp_path / "recipe.json"
    recipe_payload = {
        "schema_version": evaluator.SELECTION_SCHEMA,
        "passes_dev_gate": True,
        "selection_split": "full_2024_sequence_target_cluster_cold_dev",
        "score_is_calibrated_probability": False,
        "selected_sequence_recipe": asdict(selected),
        "baseline_sequence_recipe": asdict(SEQUENCE_BASELINE),
        "contract": {
            "dev_only_selection": True,
            "test_input_used": False,
            "known_skin_input_used": False,
            "rcsb_input_used": False,
            "truth_target_assistance_to_scorer": False,
            "train_supported_target_scores_unchanged": True,
            "retrieval_score_is_probability": False,
        },
        "calibrators": {
            label: {
                "aggregate": dict(calibrator),
                "unsupported_target": dict(calibrator),
            }
            for label in ("baseline", "selected")
        },
        "provenance": {
            key: _record(paths[path_name])
            for key, path_name in provenance_keys.items()
        },
    }
    recipe_path.write_text(json.dumps(recipe_payload))
    selection_manifest_path = tmp_path / "selection_manifest.json"
    selection_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": evaluator.SELECTION_SCHEMA,
                "passes_dev_gate": True,
                "selected_sequence_recipe": asdict(selected),
                "inputs": {
                    "index_manifest": _record(paths["index.json"]),
                    "target_cluster_manifest": _record(paths["targets.json"]),
                    "panels_manifest": _record(paths["panels.json"]),
                    "base_recipe": _record(paths["base_recipe.json"]),
                    "dev_cold_manifest": _record(paths["dev_cold_manifest.json"]),
                    "dev_cold_calibration": _record(
                        paths["dev_cold_calibration.parquet"]
                    ),
                    "embedding_manifest": _record(paths["embedding_manifest.json"]),
                },
                "outputs": {"recipe.json": _record(recipe_path)},
            }
        )
    )
    paths["recipe"] = recipe_path
    paths["selection_manifest"] = selection_manifest_path
    return paths


def _validate_fixture(paths: dict[str, Path]) -> None:
    evaluator.validate_sequence_selection(
        recipe_path=paths["recipe"],
        selection_manifest_path=paths["selection_manifest"],
        index_manifest=paths["index.json"],
        target_cluster_manifest=paths["targets.json"],
        panels_manifest=paths["panels.json"],
        base_recipe=paths["base_recipe.json"],
        dev_cold_manifest=paths["dev_cold_manifest.json"],
        dev_cold_calibration=paths["dev_cold_calibration.parquet"],
        embedding_manifest=paths["embedding_manifest.json"],
    )


def test_sequence_selection_requires_bound_recipe_and_no_assistance_contract(
    tmp_path: Path,
) -> None:
    paths = _selection_fixture(tmp_path)
    _validate_fixture(paths)

    payload = json.loads(paths["recipe"].read_text())
    payload["contract"]["test_input_used"] = True
    paths["recipe"].write_text(json.dumps(payload))
    with pytest.raises(SystemExit, match="no-assistance contract is invalid"):
        _validate_fixture(paths)


def test_sequence_selection_rejects_stale_recipe_output_hash(tmp_path: Path) -> None:
    paths = _selection_fixture(tmp_path)
    manifest = json.loads(paths["selection_manifest"].read_text())
    manifest["outputs"]["recipe.json"]["sha256"] = "0" * 64
    paths["selection_manifest"].write_text(json.dumps(manifest))

    with pytest.raises(SystemExit, match="sequence selection recipe sha256 mismatch"):
        _validate_fixture(paths)


def test_output_guards_preserve_protected_inputs(tmp_path: Path) -> None:
    protected = tmp_path / "input.csv"
    protected.write_bytes(b"bound input")

    with pytest.raises(SystemExit, match="aliases a protected input"):
        evaluator._assert_safe_output_paths(
            input_paths=(protected,),
            output_paths=(protected,),
        )
    assert protected.read_bytes() == b"bound input"

    ranking_dir = tmp_path / "known_rankings"
    nested_input = ranking_dir / "bound.json"
    ranking_dir.mkdir()
    nested_input.write_bytes(b"nested input")
    with pytest.raises(SystemExit, match="contains a protected input"):
        evaluator._assert_safe_output_paths(
            input_paths=(nested_input,),
            output_paths=(tmp_path / "summary.json",),
            destructive_dirs=(ranking_dir,),
        )
    assert nested_input.read_bytes() == b"nested input"


def test_late_manifest_failure_removes_all_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_names = (
        "ligands.parquet",
        "edges.parquet",
        "index.json",
        "targets.csv",
        "targets.json",
        "embeddings.parquet",
        "embeddings.json",
        "panels.json",
        "base_recipe.json",
        "sequence_recipe.json",
        "selection.json",
        "dev_cold.json",
        "dev_cold_calibration.parquet",
        "test_ranking.parquet",
        "dual_ranking.parquet",
        "test_calibration.parquet",
        "known.csv",
    )
    inputs = {name: tmp_path / name for name in input_names}
    for name, path in inputs.items():
        if name == "dev_cold_calibration.parquet":
            pd.DataFrame({"pair_id": ["dev-pair"]}).to_parquet(path, index=False)
        else:
            path.write_bytes(b"bound input")

    args = SimpleNamespace(
        ligands=inputs["ligands.parquet"],
        edges=inputs["edges.parquet"],
        index_manifest=inputs["index.json"],
        target_clusters=inputs["targets.csv"],
        target_cluster_manifest=inputs["targets.json"],
        embeddings=inputs["embeddings.parquet"],
        embedding_manifest=inputs["embeddings.json"],
        panels_manifest=inputs["panels.json"],
        base_recipe=inputs["base_recipe.json"],
        sequence_recipe=inputs["sequence_recipe.json"],
        sequence_selection_manifest=inputs["selection.json"],
        dev_cold_manifest=inputs["dev_cold.json"],
        dev_cold_calibration=inputs["dev_cold_calibration.parquet"],
        test_ranking=inputs["test_ranking.parquet"],
        dual_cold_ranking=inputs["dual_ranking.parquet"],
        test_calibration=inputs["test_calibration.parquet"],
        known_panel=inputs["known.csv"],
        out_dir=tmp_path / "out",
    )
    base_recipe = Recipe("base")
    baseline = SEQUENCE_BASELINE
    selected = SEQUENCE_CANDIDATES[0]
    reference = SimpleNamespace(
        target_ids=np.asarray(["P1", "P2"], dtype=object),
        edge_target=np.asarray([0], dtype=np.int32),
        edge_positive_count=np.asarray([1], dtype=np.int32),
    )
    embeddings, neighbors = _embedding_fixture()

    monkeypatch.setattr(evaluator, "_validate_panel_manifest", lambda *_a, **_k: {})
    monkeypatch.setattr(
        evaluator,
        "load_reference_index",
        lambda **_kwargs: (reference, {}, {}),
    )
    monkeypatch.setattr(evaluator, "_validate_index_panel_binding", lambda **_kwargs: None)
    monkeypatch.setattr(
        evaluator,
        "_read_json",
        lambda *_args: {"exclude_reference_similarity": 0.85},
    )
    monkeypatch.setattr(
        evaluator,
        "_validate_recipe_provenance",
        lambda *_args, **_kwargs: (base_recipe, base_recipe, {}),
    )
    monkeypatch.setattr(
        evaluator,
        "validate_sequence_selection",
        lambda **_kwargs: (
            {"base_activity_recipe": asdict(base_recipe)},
            selected,
            baseline,
            {},
        ),
    )
    monkeypatch.setattr(
        evaluator,
        "train_supported_targets",
        lambda *_args: np.asarray([True, False]),
    )
    monkeypatch.setattr(
        evaluator, "load_target_embedding_index", lambda **_kwargs: embeddings
    )
    monkeypatch.setattr(
        evaluator, "build_target_neighbor_index", lambda *_args, **_kwargs: neighbors
    )
    monkeypatch.setattr(
        evaluator,
        "load_ranking_panel",
        lambda path, *_args, **_kwargs: (
            _panel(["P2"]).drop(
                columns=sorted(evaluator.SOURCE_PROVENANCE_COLUMNS)
            )
            if path == args.test_ranking
            else _panel(["P2"])
        ),
    )
    monkeypatch.setattr(
        evaluator,
        "load_calibration_panel",
        lambda *_args, **_kwargs: pd.DataFrame({"pair_id": ["test-pair"]}),
    )

    def fake_ranking(*_args: object, **_kwargs: object) -> tuple[pd.DataFrame, pd.DataFrame]:
        query = pd.DataFrame(
            {"query_id": ["Q1", "Q1"], "recipe_id": [baseline.recipe_id, selected.recipe_id]}
        )
        target = pd.DataFrame(
            {
                "query_id": ["Q1", "Q1"],
                "target_id": ["P2", "P2"],
                "recipe_id": [baseline.recipe_id, selected.recipe_id],
                "target_train_supported": [False, False],
            }
        )
        return query, target

    monkeypatch.setattr(evaluator, "score_posthoc_ranking_panel", fake_ranking)
    monkeypatch.setattr(
        evaluator,
        "score_sequence_calibration_panel",
        lambda *_args, **_kwargs: pd.DataFrame(
            {"pair_id": ["test-pair"], "label": [1], "sample_weight": [1.0]}
        ),
    )

    calibration_metrics = {
        baseline.recipe_id: {
            "aggregate": {"brier": 0.2, "log_loss": 0.5},
            "unsupported_target": {"brier": 0.2, "log_loss": 0.5},
        },
        selected.recipe_id: {
            "aggregate": {"brier": 0.1, "log_loss": 0.4},
            "unsupported_target": {"brier": 0.1, "log_loss": 0.4},
        },
    }
    monkeypatch.setattr(
        evaluator,
        "evaluate_test_calibration",
        lambda frame, *_args: (calibration_metrics, frame),
    )

    def fake_known(*_args: object, **kwargs: object) -> tuple[pd.DataFrame, pd.DataFrame]:
        out_root = Path(kwargs["out_root"])
        for recipe in (baseline, selected):
            directory = out_root / f"known_rankings_{recipe.recipe_id}"
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "case.tsv").write_text("rank\n1\n")
        return fake_ranking()

    monkeypatch.setattr(evaluator, "score_known_posthoc_panel", fake_known)
    ranking_metrics = {
        recipe.recipe_id: {
            "top10": value,
            "top30": value,
            "mrr": value,
            "target_macro": {"top10": value, "top30": value, "mrr": value},
        }
        for recipe, value in ((baseline, 0.1), (selected, 0.2))
    }
    monkeypatch.setattr(
        evaluator, "_ranking_metrics_by_recipe", lambda _frame: ranking_metrics
    )
    monkeypatch.setattr(
        evaluator,
        "_source_stratified_ranking_metrics_by_recipe",
        lambda _frame: {"activity_quantitative": ranking_metrics},
    )
    monkeypatch.setattr(
        evaluator, "_ranking_by_support", lambda _frame: {"all": ranking_metrics}
    )
    real_write_json = evaluator._write_json_atomic
    written_payloads: dict[str, dict[str, object]] = {}

    def fail_manifest(payload: dict[str, object], path: Path) -> None:
        written_payloads[path.name] = payload
        if path.name == "manifest.json":
            raise RuntimeError("manifest write failed")
        real_write_json(payload, path)

    monkeypatch.setattr(evaluator, "_write_json_atomic", fail_manifest)
    for recipe in (SEQUENCE_BASELINE, *SEQUENCE_CANDIDATES):
        stale_dir = args.out_dir / f"known_rankings_{recipe.recipe_id}"
        stale_dir.mkdir(parents=True, exist_ok=True)
        (stale_dir / "stale.tsv").write_text("stale\n")

    with pytest.raises(RuntimeError, match="manifest write failed"):
        evaluator.run(args)

    expected_provenance = {
        "temporal_test_2025_available": False,
        "dual_cold_available": True,
        "known_skin_case_metadata_available": True,
        "temporal_test_note": (
            "source/document attribution columns are absent from the bound "
            "temporal ranking panel"
        ),
    }
    assert written_payloads["summary.json"]["source_provenance"] == expected_provenance
    assert written_payloads["manifest.json"]["source_provenance"] == expected_provenance
    assert args.out_dir.exists()
    assert not list(args.out_dir.rglob("*"))
