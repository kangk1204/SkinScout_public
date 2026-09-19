"""Regression tests for performance-v2 calibration split groups (F46).

The calibration holdout key used to be ligand+target+ontology even though the
model features are ligand+target only. Two ontology rows of one pair could
therefore cross fit/holdout and leak the same feature. These tests pin the
ligand-target pair as the split unit.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def load_model():
    spec = importlib.util.spec_from_file_location(
        "performance_v2_model_for_split_groups",
        ROOT / "eval" / "performance_v2_model.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frame() -> pd.DataFrame:
    return pd.DataFrame([
        # T1: two label-1 pairs, each with two ontology variants.
        {
            "ligand_id": "L1", "target_id": "T1",
            "ontology": "direct_binding_reversible", "measured_label": 1,
        },
        {
            "ligand_id": "L1", "target_id": "T1",
            "ontology": "functional_modulation", "measured_label": 1,
        },
        {
            "ligand_id": "L2", "target_id": "T1",
            "ontology": "direct_binding_reversible", "measured_label": 1,
        },
        {
            "ligand_id": "L2", "target_id": "T1",
            "ontology": "functional_modulation", "measured_label": 1,
        },
        # T2: two label-0 pairs, each with two ontology variants.
        {
            "ligand_id": "L3", "target_id": "T2",
            "ontology": "functional_modulation", "measured_label": 0,
        },
        {
            "ligand_id": "L3", "target_id": "T2",
            "ontology": "unknown_mixed", "measured_label": 0,
        },
        {
            "ligand_id": "L4", "target_id": "T2",
            "ontology": "functional_modulation", "measured_label": 0,
        },
        {
            "ligand_id": "L4", "target_id": "T2",
            "ontology": "unknown_mixed", "measured_label": 0,
        },
        # T3: one pair whose two ontologies disagree; it must stay atomic.
        {
            "ligand_id": "L5", "target_id": "T3",
            "ontology": "direct_binding_reversible", "measured_label": 1,
        },
        {
            "ligand_id": "L5", "target_id": "T3",
            "ontology": "functional_modulation", "measured_label": 0,
        },
    ]).assign(
        evidence_state=lambda frame: frame["measured_label"].map(
            {1: "measured_positive", 0: "measured_negative"}
        )
    )


def _observed_sides(
    fit: pd.DataFrame, holdout: pd.DataFrame
) -> dict[tuple[str, str], set[str]]:
    sides: dict[tuple[str, str], set[str]] = {}
    for frame, label in ((fit, "fit"), (holdout, "holdout")):
        for pair in zip(frame["ligand_id"], frame["target_id"], strict=True):
            sides.setdefault(pair, set()).add(label)
    return sides


def test_ontology_variants_of_a_pair_never_cross_fit_and_holdout() -> None:
    module = load_model()
    frame = _frame()

    fit, holdout, evidence = module.calibration_holdout_split(frame, fraction=0.2)

    fit_pairs = set(zip(fit["ligand_id"], fit["target_id"], strict=True))
    holdout_pairs = set(zip(holdout["ligand_id"], holdout["target_id"], strict=True))
    assert fit_pairs.isdisjoint(holdout_pairs)
    cross = fit_pairs & holdout_pairs
    assert len(cross) == 0
    assert len(fit) + len(holdout) == len(frame)
    assert evidence["disjoint_pair_hashes"] is True
    assert evidence["split_unit"] == (
        "ligand_id+target_id (feature identity; ontology variants never split)"
    )
    assert evidence["strategy"] == module.CALIBRATION_SPLIT_STRATEGY


def test_every_ontology_row_of_a_pair_lands_on_one_side() -> None:
    module = load_model()
    frame = _frame()

    fit, holdout, _ = module.calibration_holdout_split(frame, fraction=0.2)
    sides = _observed_sides(fit, holdout)

    assert set(sides) == set(
        zip(frame["ligand_id"], frame["target_id"], strict=True)
    )
    assert all(len(observed) == 1 for observed in sides.values())
    for (ligand_id, target_id), observed in sides.items():
        group = frame.loc[
            frame["ligand_id"].eq(ligand_id) & frame["target_id"].eq(target_id)
        ]
        assigned = fit if observed == {"fit"} else holdout
        rows = assigned.loc[
            assigned["ligand_id"].eq(ligand_id)
            & assigned["target_id"].eq(target_id)
        ]
        assert len(rows) == len(group)


def test_mixed_label_pair_is_not_rejected_and_stays_atomic() -> None:
    module = load_model()
    frame = _frame()

    fit, holdout, _ = module.calibration_holdout_split(frame, fraction=0.2)

    mixed_fit = fit.loc[
        fit["ligand_id"].eq("L5") & fit["target_id"].eq("T3")
    ]
    mixed_holdout = holdout.loc[
        holdout["ligand_id"].eq("L5") & holdout["target_id"].eq("T3")
    ]
    assert (len(mixed_fit) == 0) != (len(mixed_holdout) == 0)
    assert len(mixed_fit) + len(mixed_holdout) == 2


def test_split_is_deterministic_under_row_permutation() -> None:
    module = load_model()
    frame = _frame()

    _, _, evidence = module.calibration_holdout_split(frame, fraction=0.2)
    shuffled = frame.sample(frac=1.0, random_state=17).reset_index(drop=True)
    _, _, shuffled_evidence = module.calibration_holdout_split(shuffled, fraction=0.2)

    assert shuffled_evidence == evidence
