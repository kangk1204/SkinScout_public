"""Regression tests for the similarity-band tie/scoreability contract (F44).

The auxiliary band table used `np.argsort`, which gives every zero-score target
a distinct ordinal rank and lets an unscored truth become a Top30 hit whenever
fewer than 30 positive targets exist. These tests pin the shared average-tie
contract, the scorable gate, and permutation invariance.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "eval"))

from activity_retrieval_scoring import (  # noqa: E402
    ReferenceIndex,
    average_tie_ranks,
    scorable_target_mask,
)


def load_band_module():
    spec = importlib.util.spec_from_file_location(
        "measure_panel_similarity_bands_for_ties",
        ROOT / "scripts" / "measure_panel_similarity_bands.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_band_module_reuses_the_frozen_evaluator_tie_contract() -> None:
    import activity_retrieval_model

    assert activity_retrieval_model._average_tie_ranks is average_tie_ranks
    assert activity_retrieval_model.scorable_target_mask is scorable_target_mask


def test_average_tie_ranks_average_each_exact_score_group() -> None:
    scores = np.array([0.5, 0.5, 0.1, 0.1, 0.1])
    assert np.allclose(average_tie_ranks(scores), [1.5, 1.5, 4.0, 4.0, 4.0])


def test_unscored_zero_score_truth_with_29_positive_targets_is_not_a_hit() -> None:
    module = load_band_module()
    scores = np.concatenate([np.linspace(1.0, 0.7, 29), [0.0]])
    scorable = np.concatenate([np.ones(29, dtype=bool), [False]])

    ranks = module.rank_targets(scores, scorable)

    assert ranks[-1] == np.inf
    assert not bool(ranks[-1] <= 30)
    assert bool((ranks[:-1] <= 30).all())
    # The ungated ordinal ranking is exactly the bug: the zero-score tail got
    # position 30 and would have counted as a Top30 recovery.
    assert bool(average_tie_ranks(scores)[-1] <= 30)


def test_ranks_are_invariant_under_target_array_permutation() -> None:
    module = load_band_module()
    rng = np.random.default_rng(20260919)
    scores = rng.integers(0, 3, size=250).astype(float)
    scorable = rng.random(250) < 0.7

    original = module.rank_targets(scores, scorable)
    permutation = rng.permutation(len(scores))
    permuted = module.rank_targets(scores[permutation], scorable[permutation])
    restored = np.empty_like(permuted)
    restored[permutation] = permuted

    assert np.array_equal(original, restored)


def test_rank_targets_rejects_mismatched_shapes() -> None:
    module = load_band_module()
    with pytest.raises(ValueError):
        module.rank_targets(np.zeros(3), np.zeros(2, dtype=bool))


def test_scorable_target_mask_flags_targets_without_activity_edges() -> None:
    reference = ReferenceIndex(
        target_ids=np.array(["T1", "T2", "T3"]),
        fingerprints=[],
        ligand_keys=np.array([]),
        connectivity_keys=np.array([]),
        edge_ligand=np.array([0, 1], dtype=np.int64),
        edge_target=np.array([0, 0], dtype=np.int64),
        edge_max_pactivity=np.array([6.0, 5.0]),
        edge_chembl_max=np.array([6.0, 0.0]),
        edge_bindingdb_max=np.array([0.0, 5.0]),
        edge_gtopdb_max=np.array([0.0, 0.0]),
        edge_positive_count=np.array([1.0, 1.0]),
        edge_negative_count=np.array([0.0, 0.0]),
        edge_measurement_count=np.array([1.0, 1.0]),
    )

    assert scorable_target_mask(reference).tolist() == [True, False, False]
