"""Stage 3 can now score with the promoted recipe, and must agree with eval.

The operational gate recorded `operational_recipe_id=union_p6_consensus` while
nothing read it: the recipe weights lived only inside the evaluation harness, so
promotion changed a decision and not a single run. `stage3_recipe_scoring` is the
connection, and the thing that matters about it is that it produces the *same*
ranking the evaluation measured - otherwise the recovery numbers that justified
promoting the recipe would not describe what runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "eval"))

INDEX = ROOT / "data" / "activity_retrieval_202608"
CLUSTERS = ROOT / "data" / "evidence_splits" / "screenable_target_clusters_2026_02.csv"
CLUSTER_MANIFEST = (
    ROOT / "data" / "evidence_splits" / "screenable_target_clusters_2026_02.manifest.json"
)
RECIPE = ROOT / "results" / "eval" / "activity_retrieval_202608" / "dev_selection" / "recipe.json"

CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"

pytestmark = pytest.mark.skipif(
    not (INDEX / "edges.parquet").exists() or not CLUSTERS.exists(),
    reason="retrieval index is not provisioned here",
)


def test_the_recipe_artifact_is_read_fail_closed(tmp_path: Path) -> None:
    from stage3_recipe_scoring import load_promoted_recipe

    missing = tmp_path / "absent.json"
    with pytest.raises(SystemExit, match="missing or empty"):
        load_promoted_recipe(missing)

    wrong_schema = tmp_path / "wrong.json"
    wrong_schema.write_text(json.dumps({"schema_version": "nope"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="schema must be"):
        load_promoted_recipe(wrong_schema)

    not_promoted = tmp_path / "failed.json"
    not_promoted.write_text(
        json.dumps(
            {
                "schema_version": "skinscout.activity-retrieval-recipe.v1",
                "passes_dev_gate": False,
                "selected_recipe": {"recipe_id": "x"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="did not pass dev selection"):
        load_promoted_recipe(not_promoted)


def test_a_weight_this_scorer_cannot_apply_is_refused(tmp_path: Path) -> None:
    """Silently ignoring an unknown weight would score with a different recipe
    than the one the gate promoted, while reporting the promoted id."""
    from stage3_recipe_scoring import load_promoted_recipe

    path = tmp_path / "future.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "skinscout.activity-retrieval-recipe.v1",
                "passes_dev_gate": True,
                "selected_recipe": {
                    "recipe_id": "union_p7_future",
                    "max_union6": 0.6,
                    "some_new_feature": 0.4,
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="weights this scorer does not implement"):
        load_promoted_recipe(path)


@pytest.mark.skipif(not RECIPE.exists(), reason="no promoted recipe artifact")
def test_the_run_path_reproduces_the_evaluation_ranking() -> None:
    """The proof the wiring is correct: same query, same index, same recipe,
    same scores as the harness that measured them."""
    from activity_retrieval_scoring import (
        apply_recipe,
        load_reference_index,
        query_features,
        score_query_features,
    )
    from stage3_recipe_scoring import load_promoted_recipe, score_query_with_recipe

    recipe = load_promoted_recipe(RECIPE)
    # Whatever the dev selection promoted; pinned by name in
    # test_recipe_runpath_decision.py rather than duplicated here.
    assert recipe.recipe_id.startswith("union_")

    frame, stats = score_query_with_recipe(
        query_smiles=CAFFEINE,
        recipe=recipe,
        ligands_path=INDEX / "ligands.parquet",
        edges_path=INDEX / "edges.parquet",
        index_manifest_path=INDEX / "manifest.json",
        target_csv=CLUSTERS,
        target_manifest_path=CLUSTER_MANIFEST,
        exclude_reference_similarity=0.85,
    )

    reference, _, _ = load_reference_index(
        ligands_path=INDEX / "ligands.parquet",
        edges_path=INDEX / "edges.parquet",
        index_manifest_path=INDEX / "manifest.json",
        target_csv=CLUSTERS,
        target_manifest_path=CLUSTER_MANIFEST,
    )
    query_fp, _, _, _ = query_features(CAFFEINE)
    features, _ = score_query_features(
        reference, query_fp, exclude_reference_similarity=0.85
    )
    expected = apply_recipe(features, recipe)

    by_target = dict(zip(frame["target_id"], frame["score"]))
    for index, target in enumerate(reference.target_ids):
        want = float(expected[index])
        if want > 0.0:
            assert target in by_target, target
            assert by_target[target] == pytest.approx(want), target
        else:
            assert target not in by_target, target

    # And the frame is in the shape Stage 3 already writes downstream.
    # `supporting_evidence` says what the nearest analogue's own measurement was,
    # which the ranking deliberately does not use.
    assert list(frame.columns) == [
        "target_id",
        "max_tanimoto",
        "evidence_count",
        "supporting_molecule_id",
        "score",
        "supporting_evidence",
    ]
    assert frame["score"].is_monotonic_decreasing
    assert (frame["max_tanimoto"] <= 1.0).all()
    assert stats["recipe_id"] == recipe.recipe_id


@pytest.mark.skipif(not RECIPE.exists(), reason="no promoted recipe artifact")
def test_the_recipe_ranks_differently_from_max_similarity() -> None:
    """If it ranked identically the wiring would be pointless - and a sign the
    weights were not actually being applied."""
    from stage3_recipe_scoring import load_promoted_recipe, score_query_with_recipe

    frame, _ = score_query_with_recipe(
        query_smiles=CAFFEINE,
        recipe=load_promoted_recipe(RECIPE),
        ligands_path=INDEX / "ligands.parquet",
        edges_path=INDEX / "edges.parquet",
        index_manifest_path=INDEX / "manifest.json",
        target_csv=CLUSTERS,
        target_manifest_path=CLUSTER_MANIFEST,
        exclude_reference_similarity=0.85,
    )

    by_similarity = frame.sort_values(
        ["max_tanimoto", "target_id"], ascending=[False, True], kind="mergesort"
    )
    assert list(by_similarity["target_id"][:30]) != list(frame["target_id"][:30])


@pytest.mark.skipif(not RECIPE.exists(), reason="no promoted recipe artifact")
def test_the_scorer_reports_which_index_role_it_read() -> None:
    """A production run against the train-only evaluation index would narrow what
    a researcher sees rather than widen it, so the role has to be visible."""
    from stage3_recipe_scoring import load_promoted_recipe, score_query_with_recipe

    _, stats = score_query_with_recipe(
        query_smiles=CAFFEINE,
        recipe=load_promoted_recipe(RECIPE),
        ligands_path=INDEX / "ligands.parquet",
        edges_path=INDEX / "edges.parquet",
        index_manifest_path=INDEX / "manifest.json",
        target_csv=CLUSTERS,
        target_manifest_path=CLUSTER_MANIFEST,
        exclude_reference_similarity=0.85,
    )
    assert stats["index_role"] == "evaluation"
    assert stats["scored_targets"] > 0
    assert stats["scored_targets"] <= stats["target_universe"]


def test_recipe_scoring_reads_a_production_index() -> None:
    """Switched on 2026-08-31. The index it reads is the part that can go wrong.

    The evaluation index is train-only (<=2023-12-31); pointing production at it
    would narrow a researcher's results to pre-2024 evidence - a regression
    dressed as an improvement. The state of the switch itself is asserted in
    test_recipe_runpath_decision.py along with the measurement behind it.
    """
    import yaml

    config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text(encoding="utf-8"))
    docking = config["docking"]

    assert docking["daina_recipe_path"].endswith("recipe.json")
    if docking["daina_recipe_scoring"]:
        index = ROOT / docking["daina_recipe_index_dir"] / "manifest.json"
        if index.exists():
            manifest = json.loads(index.read_text(encoding="utf-8"))
            assert manifest["index_role"] == "production", (
                "recipe scoring must not read the train-only evaluation index"
            )
