#!/usr/bin/env python3
"""Borrow activity evidence across target clusters.

The retrieval model can only score a target that carries an activity edge. On
the committed 2026-08 index that is 4,510 of 20,204 screenable targets, so 78%
of the universe is unscorable no matter how good the ranking is.

This module fills part of that hole by lending a scorable target's score to its
cluster mates at a discount. The cluster map is a parameter, so the same
mechanism serves sequence clusters (MMseqs ``target_cluster_30``/``_50``) and
pocket clusters (Foldseek ``pocket_cluster_tm40``/``_50``/``_60``).

Every borrowed score keeps its donor, so a transferred hit is never
indistinguishable from a directly measured one.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


SCHEMA_VERSION = "skinscout.evidence_transfer.v1"
SEQUENCE_CLUSTER_LEVELS = ("target_cluster_30", "target_cluster_50")
POCKET_CLUSTER_LEVELS = (
    "pocket_cluster_tm40",
    "pocket_cluster_tm50",
    "pocket_cluster_tm60",
)
CLUSTER_LEVELS = SEQUENCE_CLUSTER_LEVELS + POCKET_CLUSTER_LEVELS


class EvidenceTransferError(ValueError):
    """Raised when a transfer input violates its contract."""


@dataclass(frozen=True, slots=True)
class TransferRecord:
    """One borrowed score, with the donor that supplied it."""

    target_id: str
    donor_target_id: str
    cluster_id: str
    cluster_level: str
    donor_score: float
    discount: float
    transferred_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_discount(discount: float) -> float:
    value = float(discount)
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise EvidenceTransferError(
            f"transfer discount must be a finite value in (0, 1]: {discount!r}"
        )
    return value


def load_cluster_map(path: Path, level: str) -> dict[str, str]:
    """Read ``uniprot -> cluster`` for one clustering level.

    Blank cluster cells mean the target was never clustered (no structure, no
    pocket); those targets are omitted rather than silently grouped together
    under an empty cluster id, which would make every one of them a donor for
    every other.
    """
    if level not in CLUSTER_LEVELS:
        allowed = ", ".join(CLUSTER_LEVELS)
        raise EvidenceTransferError(
            f"cluster level must be one of: {allowed}; got {level!r}"
        )
    if not path.exists() or path.stat().st_size == 0:
        raise EvidenceTransferError(
            f"cluster map is required and must be non-empty: {path}"
        )
    frame = pd.read_csv(path)
    for column in ("uniprot", level):
        if column not in frame.columns:
            raise EvidenceTransferError(
                f"cluster map missing required column {column!r}: {path}"
            )
    uniprot = frame["uniprot"].astype("string").str.strip()
    if uniprot.isna().any() or uniprot.eq("").any():
        raise EvidenceTransferError(f"cluster map contains a blank uniprot: {path}")
    duplicates = uniprot[uniprot.duplicated()].tolist()
    if duplicates:
        shown = ", ".join(sorted(set(duplicates))[:10])
        raise EvidenceTransferError(
            f"cluster map contains duplicate uniprot values: {shown}: {path}"
        )
    cluster = frame[level].astype("string").str.strip()
    keep = cluster.notna() & cluster.ne("")
    return dict(zip(uniprot[keep], cluster[keep], strict=True))


def transfer_scores(
    target_ids: Sequence[str],
    scores: Sequence[float],
    scorable: Sequence[bool],
    cluster_of: Mapping[str, str],
    *,
    discount: float,
    cluster_level: str,
    min_donor_score: float = 0.0,
) -> tuple[np.ndarray, list[TransferRecord]]:
    """Lend each scorable target's score to its unscorable cluster mates.

    Args:
        target_ids: ranking universe, one entry per score.
        scores: model scores aligned to ``target_ids``.
        scorable: whether the model had direct evidence for that target.
        cluster_of: ``uniprot -> cluster id`` for one clustering level.
        discount: multiplier applied to a borrowed score, in (0, 1].
        cluster_level: recorded on every transfer for provenance.
        min_donor_score: donors at or below this contribute nothing.

    Returns:
        ``(scores, records)`` where scores is a copy with borrowed values filled
        in for previously unscorable targets. Directly scored targets are never
        modified.
    """
    ids = [str(value) for value in target_ids]
    values = np.asarray(scores, dtype=float)
    covered = np.asarray(scorable, dtype=bool)
    if not (len(ids) == len(values) == len(covered)):
        raise EvidenceTransferError(
            "target_ids, scores and scorable must have equal length"
        )
    if values.size and not np.isfinite(values).all():
        raise EvidenceTransferError("transfer input scores must all be finite")
    factor = _require_discount(discount)
    if cluster_level not in CLUSTER_LEVELS:
        raise EvidenceTransferError(f"unknown cluster level: {cluster_level!r}")
    floor = float(min_donor_score)
    if not math.isfinite(floor) or floor < 0.0:
        raise EvidenceTransferError("min_donor_score must be finite and >= 0")

    # Best donor per cluster; ties break on target_id so the record is stable.
    best: dict[str, tuple[float, str]] = {}
    for identifier, score, is_covered in zip(ids, values, covered, strict=True):
        if not is_covered or score <= floor:
            continue
        cluster = cluster_of.get(identifier)
        if cluster is None:
            continue
        current = best.get(cluster)
        if current is None or (score, identifier) > (current[0], current[1]):
            best[cluster] = (float(score), identifier)

    output = values.copy()
    records: list[TransferRecord] = []
    for index, identifier in enumerate(ids):
        if covered[index]:
            continue
        cluster = cluster_of.get(identifier)
        if cluster is None:
            continue
        donor = best.get(cluster)
        if donor is None or donor[1] == identifier:
            continue
        transferred = donor[0] * factor
        if transferred <= output[index]:
            continue
        output[index] = transferred
        records.append(
            TransferRecord(
                target_id=identifier,
                donor_target_id=donor[1],
                cluster_id=cluster,
                cluster_level=cluster_level,
                donor_score=donor[0],
                discount=factor,
                transferred_score=float(transferred),
            )
        )
    return output, records


def transfer_manifest(
    records: Sequence[TransferRecord],
    *,
    cluster_level: str,
    discount: float,
    cluster_map_path: Path,
    scorable_before: int,
    universe_size: int,
) -> dict[str, Any]:
    """Summarise one transfer pass for the run manifest."""
    donors = sorted({record.donor_target_id for record in records})
    return {
        "schema_version": SCHEMA_VERSION,
        "cluster_level": cluster_level,
        "cluster_map": str(cluster_map_path),
        "discount": _require_discount(discount),
        "universe_size": int(universe_size),
        "scorable_before": int(scorable_before),
        "scorable_after": int(scorable_before + len(records)),
        "transferred_targets": int(len(records)),
        "distinct_donors": int(len(donors)),
        "coverage_before": (
            float(scorable_before / universe_size) if universe_size else 0.0
        ),
        "coverage_after": (
            float((scorable_before + len(records)) / universe_size)
            if universe_size
            else 0.0
        ),
        "evidence_route": "cluster_transfer; not a direct measurement",
    }
