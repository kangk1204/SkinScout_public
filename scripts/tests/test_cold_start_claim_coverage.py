"""Regression tests for the dual-cold scorable-truth claim contract (F09).

A frozen dual-cold panel can pass the counts/diversity adequacy gate while the
retrieval model has no activity edge for any of its truth targets. In that case
the reported ranks come from the tied zero-score tail, so the cold-start scope
must stay unclaimable no matter how large the panel is.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "scripts"))

from activity_retrieval_model import (  # noqa: E402
    BASELINE,
    _cold_start_adequacy_record,
    _cold_start_claim_record,
    _ranking_summary,
)


# Average rank of the all-zero tied block for a ~20k target universe, the
# scale at which the F09 tied-tail MRR was measured.
ZERO_TIE_RANK = 10102.5


def _adequacy_panel(*, queries: int, targets: int) -> dict[str, object]:
    checks = {
        "min_queries": True,
        "min_unique_truth_targets": True,
        "min_unique_source_documents": True,
        "max_truth_pair_target_fraction": True,
        "min_effective_target_count": True,
    }
    return {
        "passes_panel_adequacy_gate": True,
        "selection": {
            "ranking_queries": {
                "dual_cold": {
                    "adequacy": {"passes": True, "checks": checks},
                }
            }
        },
    }


def _target_rows(
    *,
    n_outside: int,
    n_inside: int,
) -> pd.DataFrame:
    rows = []
    for index in range(n_outside + n_inside):
        inside = index < n_inside
        rank = 1.0 if inside else ZERO_TIE_RANK
        rows.append(
            {
                "query_id": f"q{index}",
                "recipe_id": BASELINE.recipe_id,
                "target_id": f"P{index:05d}",
                "rank": rank,
                "top10": int(rank <= 10),
                "top30": int(rank <= 30),
                "reciprocal_rank": 1.0 / rank,
                "in_scorable_universe": inside,
                "scorable_universe_size": max(n_inside, 1),
                "target_universe_size": n_outside + n_inside,
            }
        )
    return pd.DataFrame(rows)


def test_zero_scorable_truth_with_large_panel_stays_unclaimable() -> None:
    """115 queries / 98 targets / 163 pairs reproduced as a contract case."""
    target_rows = _target_rows(n_outside=163, n_inside=0)
    summary = _ranking_summary(target_rows)
    coverage = summary["coverage"]

    # In-universe metrics must be null, not a tied-tail artifact.
    assert coverage["measured"] is True
    assert coverage["truth_pairs"] == 163
    assert coverage["truth_pairs_in_scorable_universe"] == 0
    assert coverage["truth_pair_coverage"] == 0.0
    assert coverage["in_universe"] == {"top10": None, "top30": None, "mrr": None}
    # All-truth metrics are still reported, but they are the zero-tie tail.
    assert coverage["all_truth"]["top10"] == 0.0
    assert coverage["all_truth"]["top30"] == 0.0
    assert coverage["all_truth"]["mrr"] < 1e-3

    adequacy_record = _cold_start_adequacy_record(
        _adequacy_panel(queries=115, targets=98)
    )
    assert adequacy_record["passes_panel_adequacy_gate"] is True
    record = _cold_start_claim_record(adequacy_record, coverage)

    assert record["passes_panel_adequacy_gate"] is True
    assert record["passes_truth_universe_coverage"] is False
    assert record["claimable"] is False


def test_one_scorable_truth_pair_with_adequate_panel_is_claimable() -> None:
    target_rows = _target_rows(n_outside=162, n_inside=1)
    coverage = _ranking_summary(target_rows)["coverage"]

    assert coverage["truth_pairs_in_scorable_universe"] == 1
    assert coverage["in_universe"] == {
        "top10": pytest.approx(1.0),
        "top30": pytest.approx(1.0),
        "mrr": pytest.approx(1.0),
    }

    record = _cold_start_claim_record(
        _cold_start_adequacy_record(_adequacy_panel(queries=115, targets=98)),
        coverage,
    )

    assert record["passes_truth_universe_coverage"] is True
    assert record["claimable"] is True


def test_inadequate_panel_cannot_claim_even_with_scorable_truth() -> None:
    coverage = _ranking_summary(_target_rows(n_outside=0, n_inside=1))["coverage"]

    record = _cold_start_claim_record(
        {
            "passes_panel_adequacy_gate": False,
            "adequacy": {"passes": False, "checks": {"min_queries": False}},
        },
        coverage,
    )

    assert record["passes_truth_universe_coverage"] is True
    assert record["claimable"] is False


def test_unmeasured_legacy_coverage_cannot_claim() -> None:
    legacy = pd.DataFrame(
        [
            {
                "query_id": "q1",
                "target_id": "P1",
                "rank": 3.0,
                "top10": 1,
                "top30": 1,
                "reciprocal_rank": 1.0 / 3.0,
            }
        ]
    )
    coverage = _ranking_summary(legacy)["coverage"]

    assert coverage["measured"] is False
    record = _cold_start_claim_record(
        _cold_start_adequacy_record(_adequacy_panel(queries=115, targets=98)),
        coverage,
    )
    assert record["passes_truth_universe_coverage"] is False
    assert record["claimable"] is False
