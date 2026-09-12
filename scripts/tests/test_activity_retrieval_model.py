from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "scripts"))

from activity_retrieval_model import (  # noqa: E402
    BASELINE,
    CANDIDATES,
    PANEL_SCHEMA,
    ReferenceIndex,
    SOURCE_STRATIFIED_PANEL_SOURCES,
    _calibration_split,
    _aggregate_edge_pairs,
    _cold_start_adequacy_record,
    _coverage_summary,
    _average_tie_ranks,
    _evaluation_decision,
    _rank_map,
    _ranking_summary,
    _source_stratified_ranking_metrics_by_recipe,
    _strict_improvement,
    _strict_ranking_improvement,
    _load_target_universe,
    _validate_index_panel_binding,
    _validate_query_identity,
    _write_selection_recipe_artifact,
    apply_platt,
    apply_recipe,
    calibration_metrics,
    fit_platt,
    load_ranking_panel,
    query_features,
    score_query_features,
    score_ranking_panel,
    scorable_target_mask,
)
from activity_recovery_contracts import (  # noqa: E402
    ALPHAFOLD_HUMAN_V4_SOURCE,
    validate_frozen_known_panel,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reference() -> ReferenceIndex:
    fingerprints = [query_features(smiles)[0] for smiles in ["CCO", "CCN", "c1ccccc1O"]]
    return ReferenceIndex(
        target_ids=np.array(["P00001", "P00002", "P00003"], dtype=object),
        fingerprints=fingerprints,
        ligand_keys=np.array(["l0", "l1", "l2"], dtype=object),
        connectivity_keys=np.array(["c0", "c1", "c2"], dtype=object),
        edge_ligand=np.array([0, 1, 2], dtype=np.int64),
        edge_target=np.array([0, 1, 2], dtype=np.int64),
        edge_max_pactivity=np.array([8.0, 7.0, 4.0]),
        edge_chembl_max=np.array([8.0, -np.inf, 4.0]),
        edge_bindingdb_max=np.array([-np.inf, 7.0, -np.inf]),
        edge_gtopdb_max=np.array([-np.inf, 7.2, -np.inf]),
        edge_positive_count=np.array([1.0, 2.0, 0.0]),
        edge_negative_count=np.array([0.0, 0.0, 1.0]),
        edge_measurement_count=np.array([1.0, 2.0, 1.0]),
    )


def _ranking_panel_row(
    *,
    query_id: str,
    smiles: str,
    truth_targets: list[str],
    split: str = "test",
) -> dict[str, object]:
    _, standard_inchikey, connectivity_key, canonical_smiles = query_features(smiles)
    smiles_hash = hashlib.sha256(canonical_smiles.encode()).hexdigest()[:12]
    ligand_key = f"{standard_inchikey}#SMILES-{smiles_hash}"
    return {
        "query_id": query_id,
        "ligand_key": ligand_key,
        "standard_inchikey": standard_inchikey,
        "connectivity_key": connectivity_key,
        "canonical_smiles": canonical_smiles,
        "standardization_route": "fragment_parent_canonical_smiles_key",
        "truth_targets": truth_targets,
        "n_truth_targets": len(truth_targets),
        "split": split,
    }


def test_global_similarity_exclusion_removes_exact_reference_before_target_scoring() -> None:
    reference = _reference()
    query_fp = query_features("CCO")[0]

    features, stats = score_query_features(
        reference,
        query_fp,
        exclude_reference_similarity=0.85,
    )

    assert stats["excluded_reference_ligands"] == 1
    assert features["baseline"][0] == 0.0
    assert features["max_union6"][0] == 0.0
    assert features["max_union6"][1] > 0.0
    assert features["source_consensus6"][1] > 0.0


def test_edge_aggregation_accepts_curated_gtopdb_source() -> None:
    edges = pd.DataFrame(
        [
            {
                "ligand_index": 0,
                "uniprot": "P00001",
                "source_db": "GtoPdb",
                "endpoint_family": "direct_binding",
                "measurement_count": 1,
                "positive_measurement_count": 1,
                "gray_measurement_count": 0,
                "negative_measurement_count": 0,
                "max_pactivity": 7.0,
            }
        ]
    )

    aggregated = _aggregate_edge_pairs(edges, {"P00001": 0})

    assert aggregated.loc[0, "gtopdb_max"] == pytest.approx(7.0)
    assert np.isneginf(aggregated.loc[0, "chembl_max"])
    assert np.isneginf(aggregated.loc[0, "bindingdb_max"])


def test_ranking_summary_reports_target_macro_and_dominance() -> None:
    rows = pd.DataFrame(
        {
            "query_id": ["q1", "q2", "q3"],
            "target_id": ["dominant", "dominant", "rare"],
            "rank": [1, 100, 20],
            "top10": [1, 0, 0],
            "top30": [1, 0, 1],
            "reciprocal_rank": [1.0, 0.01, 0.05],
        }
    )

    summary = _ranking_summary(rows)

    assert summary["top10"] == pytest.approx(1 / 3)
    assert summary["target_macro"]["top10"] == pytest.approx(0.25)
    assert summary["target_macro"]["top30"] == pytest.approx(0.75)
    assert summary["max_truth_pair_target_fraction"] == pytest.approx(2 / 3)
    assert summary["effective_target_count"] == pytest.approx(1.8)


def test_dual_cold_panel_rejects_malformed_truth_target_attribution(
    tmp_path: Path,
) -> None:
    row = _ranking_panel_row(
        query_id="q1",
        smiles="CCO",
        truth_targets=["P00001", "P00002"],
    )
    row.update(
        {
            "panel_sources": list(SOURCE_STRATIFIED_PANEL_SOURCES),
            "source_databases": ["ChEMBL", "RCSB"],
            "source_documents_json": json.dumps({"rcsb": ["1ABC"]}),
            "truth_target_panel_sources_json": json.dumps(
                {"P00001": ["activity_quantitative"]}
            ),
        }
    )
    path = tmp_path / "dual.parquet"
    pd.DataFrame([row]).to_parquet(path, index=False)

    with pytest.raises(SystemExit, match="must match truth_targets"):
        load_ranking_panel(path, "test", require_source_provenance=True)


def test_ranking_panel_without_source_provenance_keeps_dev_test_behavior(
    tmp_path: Path,
) -> None:
    path = tmp_path / "test.parquet"
    pd.DataFrame(
        [
            _ranking_panel_row(
                query_id="q1",
                smiles="CCO",
                truth_targets=["P00001"],
            )
        ]
    ).to_parquet(path, index=False)

    loaded = load_ranking_panel(path, "test")

    assert loaded["query_id"].tolist() == ["q1"]
    assert "panel_sources" not in loaded.columns


def test_dual_cold_mixed_source_query_preserves_provenance_and_counts_metrics(
    tmp_path: Path,
) -> None:
    rows = []
    first = _ranking_panel_row(
        query_id="q1",
        smiles="CCO",
        truth_targets=["P00001", "P00002"],
    )
    first.update(
        {
            "panel_sources": list(SOURCE_STRATIFIED_PANEL_SOURCES),
            "source_databases": ["ChEMBL", "RCSB"],
            "source_documents_json": json.dumps({"rcsb": ["1ABC"], "activity": ["doc1"]}),
            "truth_target_panel_sources_json": json.dumps(
                {
                    "P00001": ["activity_quantitative"],
                    "P00002": ["rcsb_holo_direct_contact"],
                }
            ),
        }
    )
    rows.append(first)
    second = _ranking_panel_row(
        query_id="q2",
        smiles="CCN",
        truth_targets=["P00003"],
    )
    second.update(
        {
            "panel_sources": list(SOURCE_STRATIFIED_PANEL_SOURCES),
            "source_databases": ["ChEMBL", "RCSB"],
            "source_documents_json": json.dumps({"rcsb": ["2DEF"], "activity": ["doc2"]}),
            "truth_target_panel_sources_json": json.dumps(
                {
                    "P00003": [
                        "activity_quantitative",
                        "rcsb_holo_direct_contact",
                    ],
                }
            ),
        }
    )
    rows.append(second)
    path = tmp_path / "dual.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)

    loaded = load_ranking_panel(path, "test", require_source_provenance=True)
    query_metrics, target_metrics = score_ranking_panel(
        _reference(),
        loaded,
        (BASELINE,),
        exclude_reference_similarity=1.0,
    )

    assert {
        "panel_sources",
        "source_databases",
        "source_documents_json",
        "truth_target_panel_sources_json",
    }.issubset(query_metrics.columns)
    assert "target_panel_sources_json" in target_metrics.columns

    deterministic_rows = pd.DataFrame(
        [
            {
                "query_id": "q1",
                "recipe_id": "r",
                "target_id": "P00001",
                "rank": 1,
                "top10": 1,
                "top30": 1,
                "reciprocal_rank": 1.0,
                "target_panel_sources_json": json.dumps(["activity_quantitative"]),
            },
            {
                "query_id": "q1",
                "recipe_id": "r",
                "target_id": "P00002",
                "rank": 40,
                "top10": 0,
                "top30": 0,
                "reciprocal_rank": 1.0 / 40.0,
                "target_panel_sources_json": json.dumps(["rcsb_holo_direct_contact"]),
            },
            {
                "query_id": "q2",
                "recipe_id": "r",
                "target_id": "P00003",
                "rank": 20,
                "top10": 0,
                "top30": 1,
                "reciprocal_rank": 1.0 / 20.0,
                "target_panel_sources_json": json.dumps(
                    ["activity_quantitative", "rcsb_holo_direct_contact"]
                ),
            },
        ]
    )

    metrics = _source_stratified_ranking_metrics_by_recipe(deterministic_rows)["r"]

    activity = metrics["activity_quantitative"]
    assert activity["n_queries"] == 2
    assert activity["n_truth_pairs"] == 2
    assert activity["top10"] == pytest.approx(0.5)
    assert activity["top30"] == pytest.approx(1.0)
    assert activity["mrr"] == pytest.approx(0.525)

    rcsb = metrics["rcsb_holo_direct_contact"]
    assert rcsb["n_queries"] == 2
    assert rcsb["n_truth_pairs"] == 2
    assert rcsb["top10"] == pytest.approx(0.0)
    assert rcsb["top30"] == pytest.approx(0.5)
    assert rcsb["mrr"] == pytest.approx(0.0375)


def test_query_identity_requires_dataset_independent_structure_key() -> None:
    _, standard_inchikey, connectivity_key, canonical_smiles = query_features("CCO")
    row = {
        "query_id": "ethanol",
        "ligand_key": (
            "LFQSCWFLJHTTHZ-UHFFFAOYSA-N#SMILES-"
            "ab1de819ede9"
        ),
        "standard_inchikey": standard_inchikey,
        "connectivity_key": connectivity_key,
        "canonical_smiles": canonical_smiles,
        "standardization_route": "fragment_parent_canonical_smiles_key",
    }

    _validate_query_identity(row)

    invalid = {**row, "ligand_key": standard_inchikey}
    with pytest.raises(SystemExit, match="ligand_key does not match"):
        _validate_query_identity(invalid)


def test_recipes_rank_complete_target_universe_and_never_emit_assisted_prior() -> None:
    features = {
        "baseline": np.array([0.2, 0.7, 0.0]),
        "max_union_any": np.array([0.9, 0.8, 0.0]),
        "max_union_nonneg": np.array([0.9, 0.8, 0.0]),
        "max_union5": np.array([0.8, 0.7, 0.0]),
        "max_union6": np.array([0.8, 0.7, 0.0]),
        "quality_union6": np.array([0.75, 0.65, 0.0]),
        "source_consensus6": np.array([0.5, 0.0, 0.0]),
        "support_union6": np.array([0.4, 0.1, 0.0]),
        "negative_contrast": np.array([0.0, 0.2, 0.0]),
    }
    targets = np.array(["P00001", "P00002", "P00003"], dtype=object)

    # By name, not by position: CANDIDATES[-1] silently became a different
    # recipe when negative-aware candidates were appended, and the test kept
    # passing against whatever happened to be last.
    promoted = next(r for r in CANDIDATES if r.recipe_id == "union_any_consensus")
    baseline = apply_recipe(features, BASELINE)
    candidate = apply_recipe(features, promoted)
    baseline_ranks = _rank_map(targets, baseline)
    candidate_ranks = _rank_map(targets, candidate)

    assert set(baseline_ranks) == set(targets)
    assert set(candidate_ranks) == set(targets)
    assert candidate_ranks["P00001"] < baseline_ranks["P00001"]
    assert "prior" not in " ".join(features)


def test_equal_scores_use_average_rank_without_accession_order_bias() -> None:
    targets = np.array(["Z_LAST", "A_FIRST", "M_MIDDLE", "B_SECOND"], dtype=object)
    scores = np.array([0.0, 1.0, 0.0, 1.0])

    ranks = _average_tie_ranks(scores)
    rank_map = _rank_map(targets, scores)

    assert ranks.tolist() == [3.5, 1.5, 3.5, 1.5]
    assert rank_map == {
        "Z_LAST": 3.5,
        "A_FIRST": 1.5,
        "M_MIDDLE": 3.5,
        "B_SECOND": 1.5,
    }


def test_all_zero_target_scores_cannot_create_false_top30_hits() -> None:
    target_count = 20_204
    ranks = _average_tie_ranks(np.zeros(target_count, dtype=float))

    assert np.unique(ranks).tolist() == [10_102.5]
    assert not bool((ranks <= 30).any())


def test_failed_promotion_retains_only_the_frozen_baseline_for_operations() -> None:
    decision = _evaluation_decision(
        passes_frozen_test_gate=False,
        selected_recipe_id="union_p6_consensus",
        baseline_recipe_id="chembl_p5_max",
    )

    assert decision == {
        "evaluation_completed": True,
        "claim_ready": False,
        "promotion_decision": "retain_frozen_baseline",
        "operational_recipe_id": "chembl_p5_max",
    }


def test_platt_is_monotone_and_reports_weighted_proper_scores() -> None:
    scores = np.array([0.05, 0.15, 0.75, 0.95])
    labels = np.array([0, 0, 1, 1])
    weights = np.array([3.0, 1.0, 1.0, 3.0])

    calibrator = fit_platt(scores, labels, weights)
    probabilities = apply_platt(scores, calibrator)
    metrics = calibration_metrics(probabilities, labels, weights)

    assert calibrator["slope"] >= 0.0
    assert np.all(np.diff(probabilities) >= 0.0)
    assert 0.0 <= metrics["brier"] < 0.25
    assert 0.0 <= metrics["log_loss"] < np.log(2.0)
    assert metrics["weight_sum"] == 8.0


def test_calibration_split_is_deterministic_stratified_and_keeps_both_classes() -> None:
    frame = pd.DataFrame(
        [
            {
                "pair_id": f"p{family}{label}{index}",
                "endpoint_family": family,
                "label": label,
            }
            for family in ("direct_binding", "functional")
            for label in (0, 1)
            for index in range(4)
        ]
    )

    first_fit, first_eval = _calibration_split(frame)
    second_fit, second_eval = _calibration_split(frame.sample(frac=1.0, random_state=3))

    assert set(first_fit["pair_id"]) == set(second_fit["pair_id"])
    assert set(first_eval["pair_id"]) == set(second_eval["pair_id"])
    assert set(first_fit["label"]) == {0, 1}
    assert set(first_eval["label"]) == {0, 1}


def test_strict_gate_rejects_candidate_when_any_required_metric_does_not_improve() -> None:
    baseline_rank = {"top10": 0.2, "top30": 0.4, "mrr": 0.1}
    candidate_rank = {"top10": 0.3, "top30": 0.4, "mrr": 0.2}
    baseline_cal = {"brier": 0.20, "log_loss": 0.60}
    candidate_cal = {"brier": 0.19, "log_loss": 0.59}

    passes, failures, _ = _strict_improvement(
        baseline_rank,
        candidate_rank,
        baseline_cal,
        candidate_cal,
    )

    assert not passes
    assert any("top30_gain" in failure for failure in failures)


def test_known_panel_ranking_gate_does_not_invent_calibration_scores() -> None:
    baseline = {"top10": 0.2, "top30": 0.4, "mrr": 0.1}
    candidate = {"top10": 0.3, "top30": 0.5, "mrr": 0.2}

    passes, failures, score = _strict_ranking_improvement(baseline, candidate)

    assert passes
    assert failures == []
    assert score == pytest.approx(0.3)


def test_fit_platt_rejects_unmeasured_or_single_class_surrogates() -> None:
    with pytest.raises(ValueError, match="both classes"):
        fit_platt(np.array([0.1, 0.2]), np.array([1, 1]), np.ones(2))


def test_frozen_known_panel_contract_rejects_any_changed_target(tmp_path: Path) -> None:
    source = ROOT / "data/validation/skin_known_target_panel.csv"
    panel = tmp_path / "known_panel.csv"
    panel.write_bytes(source.read_bytes())
    contract = validate_frozen_known_panel(panel)
    assert contract["rows"] == 22
    assert contract["target_pair_count"] == 46

    panel.write_text(
        panel.read_text().replace("P10276;P10826;P13631", "P10276"),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="sha256 mismatch"):
        validate_frozen_known_panel(panel)


def test_failed_selection_never_leaves_recipe_json(tmp_path: Path) -> None:
    stale = tmp_path / "recipe.json"
    stale.write_text("stale\n")

    written = _write_selection_recipe_artifact(
        {"schema_version": "diagnostic", "passes_dev_gate": False},
        tmp_path,
        passes_dev_gate=False,
    )

    assert written == tmp_path / "failed_recipe_diagnostic.json"
    assert written.exists()
    assert not stale.exists()


def test_selection_revalidates_benchmark_index_panel_binding(tmp_path: Path) -> None:
    train = tmp_path / "train.parquet"
    pd.DataFrame({"split": ["train"]}).to_parquet(train, index=False)
    benchmark_path = tmp_path / "manifest.json"
    benchmark_path.write_text(
        json.dumps(
            {
                "schema_version": "activity_benchmark.v1",
                "output_sha256": {"train.parquet": _sha256(train)},
                "splits": {"counts": {"train": 1}},
            }
        )
        + "\n"
    )
    index_path = tmp_path / "index.json"
    index_payload = {
        "schema_version": "skinscout.activity-retrieval-index.v4",
        "inputs": {
            "benchmark_manifest": {
                "path": str(benchmark_path.resolve()),
                "sha256": _sha256(benchmark_path),
            },
            "train_parquet": {
                "path": str(train.resolve()),
                "sha256": _sha256(train),
                "rows": 1,
            },
        },
    }
    index_path.write_text(json.dumps(index_payload) + "\n")
    panel_path = tmp_path / "panels.json"
    panel_payload = {
        "schema_version": PANEL_SCHEMA,
        "inputs": {
            "retrieval_index_manifest": {
                "path": str(index_path.resolve()),
                "sha256": _sha256(index_path),
            },
            "benchmark_manifest": {
                "path": str(benchmark_path.resolve()),
                "sha256": _sha256(benchmark_path),
            },
            "train.parquet": {
                "path": str(train.resolve()),
                "sha256": _sha256(train),
                "rows": 1,
            },
        },
    }
    panel_path.write_text(json.dumps(panel_payload) + "\n")

    _validate_index_panel_binding(
        index_manifest_path=index_path,
        index_manifest=index_payload,
        panel_manifest_path=panel_path,
        panel_manifest=panel_payload,
    )

    index_payload["inputs"]["benchmark_manifest"]["sha256"] = "0" * 64
    index_path.write_text(json.dumps(index_payload) + "\n")
    panel_payload["inputs"]["retrieval_index_manifest"]["sha256"] = _sha256(
        index_path
    )
    with pytest.raises(SystemExit, match="different benchmarks"):
        _validate_index_panel_binding(
            index_manifest_path=index_path,
            index_manifest=index_payload,
            panel_manifest_path=panel_path,
            panel_manifest=panel_payload,
        )


def test_model_rejects_fixture_or_reduced_target_universe(tmp_path: Path) -> None:
    target_csv = tmp_path / "targets.csv"
    pd.DataFrame(
        {
            "uniprot": ["P1", "P2", "P3"],
            "target_cluster_30": ["P1", "P1", "P3"],
            "target_cluster_50": ["P1", "P2", "P3"],
        }
    ).to_csv(target_csv, index=False)
    manifest_path = tmp_path / "targets.manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "skinscout.screenable-target-cluster-map.v2",
                "universe_policy": {
                    "evaluation_panel_used": False,
                    "known_target_assistance": False,
                    "base_target_count": 2,
                    "evidence_target_count": 2,
                    "supplemental_target_count": 1,
                    "union_target_count": 3,
                },
                "production_contract": {
                    "fixture_mode": True,
                    "expected_base_target_count": 20171,
                    "expected_union_target_count": 20204,
                    "passes": False,
                },
                "sources": {
                    "independent_base": ALPHAFOLD_HUMAN_V4_SOURCE,
                    "evidence_sequence_source": {
                        "name": "UniProtKB",
                        "release": "2026_02",
                        "license": "CC BY 4.0",
                    },
                },
                "artifact": {"sha256": _sha256(target_csv), "rows": 3},
            }
        )
        + "\n"
    )

    with pytest.raises(SystemExit, match="did not pass its production contract"):
        _load_target_universe(target_csv, manifest_path)


def test_cold_start_adequacy_record_is_emitted_not_discarded() -> None:
    """Both entry points must publish the same dual-cold adequacy record."""
    passing = {
        "passes_panel_adequacy_gate": True,
        "selection": {
            "ranking_queries": {
                "dual_cold": {
                    "adequacy": {"passes": True, "checks": {"min_queries": True}}
                }
            }
        },
    }

    record = _cold_start_adequacy_record(passing)

    assert record["passes_panel_adequacy_gate"] is True
    assert record["used_for_recipe_selection"] is False
    assert record["adequacy"]["checks"] == {"min_queries": True}


def test_cold_start_adequacy_record_fails_on_any_failed_check() -> None:
    base = {
        "passes_panel_adequacy_gate": True,
        "selection": {
            "ranking_queries": {
                "dual_cold": {
                    "adequacy": {
                        "passes": True,
                        "checks": {"min_queries": True, "min_unique_truth_targets": False},
                    }
                }
            }
        },
    }
    assert _cold_start_adequacy_record(base)["passes_panel_adequacy_gate"] is False

    outer_failed = json.loads(json.dumps(base))
    outer_failed["passes_panel_adequacy_gate"] = False
    outer_failed["selection"]["ranking_queries"]["dual_cold"]["adequacy"]["checks"] = {
        "min_queries": True
    }
    assert _cold_start_adequacy_record(outer_failed)["passes_panel_adequacy_gate"] is False


def test_cold_start_adequacy_record_requires_the_audit_block() -> None:
    with pytest.raises(SystemExit, match="dual-cold adequacy audit"):
        _cold_start_adequacy_record({"passes_panel_adequacy_gate": True})


def _reference_with_unscorable_target() -> ReferenceIndex:
    """Three targets, but P00004 carries no activity edge at all."""
    fingerprints = [query_features(smiles)[0] for smiles in ["CCO", "CCN", "c1ccccc1O"]]
    return ReferenceIndex(
        target_ids=np.array(["P00001", "P00002", "P00003", "P00004"], dtype=object),
        fingerprints=fingerprints,
        ligand_keys=np.array(["l0", "l1", "l2"], dtype=object),
        connectivity_keys=np.array(["c0", "c1", "c2"], dtype=object),
        edge_ligand=np.array([0, 1, 2], dtype=np.int64),
        edge_target=np.array([0, 1, 2], dtype=np.int64),
        edge_max_pactivity=np.array([8.0, 7.0, 4.0]),
        edge_chembl_max=np.array([8.0, -np.inf, 4.0]),
        edge_bindingdb_max=np.array([-np.inf, 7.0, -np.inf]),
        edge_gtopdb_max=np.array([-np.inf, 7.2, -np.inf]),
        edge_positive_count=np.array([1.0, 2.0, 0.0]),
        edge_negative_count=np.array([0.0, 0.0, 1.0]),
        edge_measurement_count=np.array([1.0, 2.0, 1.0]),
    )


def test_scorable_target_mask_separates_edgeless_targets() -> None:
    mask = scorable_target_mask(_reference_with_unscorable_target())

    assert mask.tolist() == [True, True, True, False]
    assert scorable_target_mask(_reference()).all()


def test_ranking_rows_flag_pairs_the_model_could_never_score(tmp_path: Path) -> None:
    """A truth pair with no activity edge must not read like a scored miss.

    Without the flag P00004 lands in the tied zero block and is reported as a
    rank like any other, so top10=0 cannot be attributed to bad ranking rather
    than absent evidence.
    """
    path = tmp_path / "panel.parquet"
    pd.DataFrame(
        [
            _ranking_panel_row(query_id="q1", smiles="CCO", truth_targets=["P00001"]),
            _ranking_panel_row(query_id="q2", smiles="CCN", truth_targets=["P00004"]),
        ]
    ).to_parquet(path, index=False)
    loaded = load_ranking_panel(path, "test")

    _, target_metrics = score_ranking_panel(
        _reference_with_unscorable_target(),
        loaded,
        (BASELINE,),
        exclude_reference_similarity=1.0,
    )

    flags = dict(
        zip(
            target_metrics["target_id"],
            target_metrics["in_scorable_universe"],
            strict=True,
        )
    )
    assert flags == {"P00001": True, "P00004": False}
    assert set(target_metrics["scorable_universe_size"]) == {3}
    assert set(target_metrics["target_universe_size"]) == {4}

    summary = _ranking_summary(target_metrics)
    coverage = summary["coverage"]
    assert coverage["measured"] is True
    assert coverage["truth_pairs"] == 2
    assert coverage["truth_pairs_in_scorable_universe"] == 1
    assert coverage["truth_pairs_outside_scorable_universe"] == 1
    assert coverage["truth_pair_coverage"] == pytest.approx(0.5)
    assert coverage["scorable_universe_size"] == 3
    # in_universe metrics describe only the pair the model could score.
    assert coverage["in_universe"]["top10"] == pytest.approx(1.0)


def test_coverage_summary_reports_when_accounting_is_absent() -> None:
    legacy = pd.DataFrame([{"query_id": "q1", "target_id": "P1", "rank": 3.0}])

    coverage = _coverage_summary(legacy)

    assert coverage["measured"] is False
    assert "predate" in coverage["reason"]
