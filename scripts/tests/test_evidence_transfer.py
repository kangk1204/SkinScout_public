"""Regression tests for cross-cluster activity-evidence transfer."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "eval"))

from evidence_transfer import (  # noqa: E402
    EvidenceTransferError,
    TransferRecord,
    load_cluster_map,
    transfer_manifest,
    transfer_scores,
)


def _cluster_csv(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "clusters.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_transfer_lends_a_donor_score_to_its_unscorable_cluster_mates() -> None:
    scores, records = transfer_scores(
        target_ids=["P1", "P2", "P3"],
        scores=[0.8, 0.0, 0.0],
        scorable=[True, False, False],
        cluster_of={"P1": "c1", "P2": "c1", "P3": "c2"},
        discount=0.5,
        cluster_level="target_cluster_30",
    )

    assert scores.tolist() == [0.8, 0.4, 0.0]
    assert len(records) == 1
    assert records[0].target_id == "P2"
    assert records[0].donor_target_id == "P1"
    assert records[0].cluster_id == "c1"
    assert records[0].transferred_score == pytest.approx(0.4)


def test_transfer_never_overwrites_a_directly_scored_target() -> None:
    """A weak direct measurement outranks nothing, but it is still a measurement."""
    scores, records = transfer_scores(
        target_ids=["P1", "P2"],
        scores=[0.9, 0.1],
        scorable=[True, True],
        cluster_of={"P1": "c1", "P2": "c1"},
        discount=1.0,
        cluster_level="target_cluster_30",
    )

    assert scores.tolist() == [0.9, 0.1]
    assert records == []


def test_transfer_picks_the_best_donor_and_breaks_ties_deterministically() -> None:
    first, _ = transfer_scores(
        target_ids=["A", "B", "Z"],
        scores=[0.5, 0.5, 0.0],
        scorable=[True, True, False],
        cluster_of={"A": "c", "B": "c", "Z": "c"},
        discount=1.0,
        cluster_level="target_cluster_50",
    )
    _, records = transfer_scores(
        target_ids=["B", "A", "Z"],
        scores=[0.5, 0.5, 0.0],
        scorable=[True, True, False],
        cluster_of={"A": "c", "B": "c", "Z": "c"},
        discount=1.0,
        cluster_level="target_cluster_50",
    )

    assert first.tolist() == [0.5, 0.5, 0.5]
    # Same cluster, same scores, different row order -> same donor.
    assert [record.donor_target_id for record in records] == ["B"]


def test_transfer_ignores_targets_with_no_cluster() -> None:
    scores, records = transfer_scores(
        target_ids=["P1", "P2"],
        scores=[0.8, 0.0],
        scorable=[True, False],
        cluster_of={"P1": "c1"},
        discount=0.5,
        cluster_level="pocket_cluster_tm50",
    )

    assert scores.tolist() == [0.8, 0.0]
    assert records == []


def test_transfer_respects_the_donor_floor() -> None:
    _, records = transfer_scores(
        target_ids=["P1", "P2"],
        scores=[0.05, 0.0],
        scorable=[True, False],
        cluster_of={"P1": "c1", "P2": "c1"},
        discount=1.0,
        cluster_level="target_cluster_30",
        min_donor_score=0.1,
    )

    assert records == []


@pytest.mark.parametrize("discount", [0.0, -0.1, 1.5, float("nan")])
def test_transfer_rejects_an_out_of_range_discount(discount: float) -> None:
    with pytest.raises(EvidenceTransferError, match="discount"):
        transfer_scores(
            target_ids=["P1"],
            scores=[0.5],
            scorable=[True],
            cluster_of={},
            discount=discount,
            cluster_level="target_cluster_30",
        )


def test_transfer_rejects_mismatched_input_lengths() -> None:
    with pytest.raises(EvidenceTransferError, match="equal length"):
        transfer_scores(
            target_ids=["P1", "P2"],
            scores=[0.5],
            scorable=[True, False],
            cluster_of={},
            discount=0.5,
            cluster_level="target_cluster_30",
        )


def test_transfer_rejects_non_finite_scores() -> None:
    with pytest.raises(EvidenceTransferError, match="finite"):
        transfer_scores(
            target_ids=["P1"],
            scores=[np.nan],
            scorable=[True],
            cluster_of={},
            discount=0.5,
            cluster_level="target_cluster_30",
        )


def test_transfer_rejects_an_unknown_cluster_level() -> None:
    with pytest.raises(EvidenceTransferError, match="cluster level"):
        transfer_scores(
            target_ids=["P1"],
            scores=[0.5],
            scorable=[True],
            cluster_of={},
            discount=0.5,
            cluster_level="made_up_level",
        )


def test_load_cluster_map_skips_unclustered_targets(tmp_path: Path) -> None:
    """Blank cluster cells must not collapse into one giant shared cluster."""
    path = _cluster_csv(
        tmp_path,
        [
            {"uniprot": "P1", "target_cluster_30": "c1"},
            {"uniprot": "P2", "target_cluster_30": ""},
            {"uniprot": "P3", "target_cluster_30": "c1"},
        ],
    )

    mapping = load_cluster_map(path, "target_cluster_30")

    assert mapping == {"P1": "c1", "P3": "c1"}


def test_load_cluster_map_rejects_duplicate_targets(tmp_path: Path) -> None:
    path = _cluster_csv(
        tmp_path,
        [
            {"uniprot": "P1", "target_cluster_30": "c1"},
            {"uniprot": "P1", "target_cluster_30": "c2"},
        ],
    )

    with pytest.raises(EvidenceTransferError, match="duplicate uniprot"):
        load_cluster_map(path, "target_cluster_30")


def test_load_cluster_map_rejects_a_missing_level(tmp_path: Path) -> None:
    path = _cluster_csv(tmp_path, [{"uniprot": "P1", "target_cluster_30": "c1"}])

    with pytest.raises(EvidenceTransferError, match="missing required column"):
        load_cluster_map(path, "target_cluster_50")


def test_transfer_manifest_reports_the_coverage_delta(tmp_path: Path) -> None:
    records = [
        TransferRecord("P2", "P1", "c1", "target_cluster_30", 0.8, 0.5, 0.4),
        TransferRecord("P3", "P1", "c1", "target_cluster_30", 0.8, 0.5, 0.4),
    ]

    manifest = transfer_manifest(
        records,
        cluster_level="target_cluster_30",
        discount=0.5,
        cluster_map_path=tmp_path / "clusters.csv",
        scorable_before=10,
        universe_size=100,
    )

    assert manifest["schema_version"] == "skinscout.evidence_transfer.v1"
    assert manifest["transferred_targets"] == 2
    assert manifest["distinct_donors"] == 1
    assert manifest["scorable_after"] == 12
    assert manifest["coverage_before"] == pytest.approx(0.10)
    assert manifest["coverage_after"] == pytest.approx(0.12)
    assert "not a direct measurement" in manifest["evidence_route"]
