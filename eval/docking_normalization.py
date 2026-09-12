#!/usr/bin/env python3
"""Make docking scores comparable across receptors.

A Vina-family score is an energy, not a probability, and its scale moves with
the pocket. Large, buried, hydrophobic sites return more negative energies for
almost any ligand, so ranking one compound against thousands of different
pockets by raw dG mostly ranks pockets by size. `results/RETROSPECTIVE.md`
records the same effect on the ligand axis and warns that magnitudes are not
comparable across compounds; the receptor axis has the identical problem and it
is the one a proteome-wide screen depends on.

The fix is to score each receptor against its own background. A fixed panel of
diverse ligands is docked into every receptor once, giving that receptor an
empirical score distribution. A query is then reported as its displacement from
that distribution rather than as a raw energy. The background is independent of
the query, so it is computed once per receptor and reused for every future
compound.

Robust statistics are the default: docking panels contain occasional gross
outliers (failed poses, degenerate boxes) and a single one distorts a mean/sd
normalisation far more than a median/MAD one.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "skinscout.docking_background.v1"
MIN_BACKGROUND_LIGANDS = 30
MAD_TO_SIGMA = 1.4826  # scale factor making MAD consistent with sd for normals
NORMALIZATION_METHODS = ("robust", "gaussian")


class DockingNormalizationError(ValueError):
    """Raised when a background or score violates its contract."""


@dataclass(frozen=True, slots=True)
class ReceptorBackground:
    """One receptor's empirical score distribution over the background panel."""

    target_id: str
    n_ligands: int
    center: float
    scale: float
    method: str
    best: float
    worst: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_array(values: Sequence[float], label: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise DockingNormalizationError(f"{label} must be one-dimensional")
    if array.size and not np.isfinite(array).all():
        raise DockingNormalizationError(f"{label} must contain only finite values")
    return array


def build_receptor_background(
    target_id: str,
    scores: Sequence[float],
    *,
    method: str = "robust",
    min_ligands: int = MIN_BACKGROUND_LIGANDS,
) -> ReceptorBackground:
    """Summarise a receptor's background docking scores.

    Args:
        target_id: receptor accession.
        scores: raw docking energies for the background panel, lower = better.
        method: ``robust`` (median/MAD) or ``gaussian`` (mean/sd).
        min_ligands: refuse a background thinner than this.
    """
    identifier = str(target_id).strip()
    if not identifier:
        raise DockingNormalizationError("receptor background requires a target_id")
    if method not in NORMALIZATION_METHODS:
        allowed = ", ".join(NORMALIZATION_METHODS)
        raise DockingNormalizationError(
            f"normalization method must be one of: {allowed}; got {method!r}"
        )
    if min_ligands < 2:
        raise DockingNormalizationError("min_ligands must be >= 2")
    array = _finite_array(scores, f"background scores for {identifier}")
    if array.size < min_ligands:
        raise DockingNormalizationError(
            f"{identifier} background has {array.size} ligands; "
            f"at least {min_ligands} are required"
        )
    if method == "robust":
        center = float(np.median(array))
        scale = float(MAD_TO_SIGMA * np.median(np.abs(array - center)))
    else:
        center = float(np.mean(array))
        scale = float(np.std(array, ddof=1))
    if not math.isfinite(scale) or scale <= 0.0:
        # A degenerate spread means every background ligand scored identically:
        # the receptor cannot discriminate and must not be silently normalised.
        raise DockingNormalizationError(
            f"{identifier} background has no usable spread (scale={scale!r}); "
            "the receptor cannot rank a query against itself"
        )
    return ReceptorBackground(
        target_id=identifier,
        n_ligands=int(array.size),
        center=center,
        scale=scale,
        method=method,
        best=float(array.min()),
        worst=float(array.max()),
    )


def normalized_score(raw_score: float, background: ReceptorBackground) -> float:
    """Displacement of a query below its receptor's background, higher = better.

    Docking energies are negative-is-better, so the sign is flipped: a query
    scoring well below the receptor's own typical ligand returns a positive
    value, and one that merely matches the background returns ~0.
    """
    value = float(raw_score)
    if not math.isfinite(value):
        raise DockingNormalizationError(
            f"raw docking score for {background.target_id} must be finite"
        )
    return float((background.center - value) / background.scale)


def normalize_screen(
    raw_scores: Mapping[str, float],
    backgrounds: Mapping[str, ReceptorBackground],
    *,
    require_background: bool = True,
) -> tuple[dict[str, float], list[str]]:
    """Normalise a whole screen, reporting receptors that had no background.

    A receptor without a background is not scored at zero — that would rank it
    against receptors measured on a different scale. It is returned separately
    so the caller can abstain on it explicitly.
    """
    normalized: dict[str, float] = {}
    missing: list[str] = []
    for target_id, raw in raw_scores.items():
        background = backgrounds.get(str(target_id))
        if background is None:
            missing.append(str(target_id))
            continue
        normalized[str(target_id)] = normalized_score(raw, background)
    if missing and require_background:
        shown = ", ".join(sorted(missing)[:10])
        raise DockingNormalizationError(
            f"{len(missing)} receptor(s) have no background distribution: {shown}"
        )
    return normalized, sorted(missing)


def background_manifest(
    backgrounds: Iterable[ReceptorBackground],
    *,
    panel_id: str,
    panel_sha256: str,
    engine: str,
    engine_version: str,
    method: str = "robust",
) -> dict[str, Any]:
    """Provenance record for one background build."""
    records = list(backgrounds)
    if not records:
        raise DockingNormalizationError("background manifest requires >= 1 receptor")
    methods = {record.method for record in records}
    if methods != {method}:
        raise DockingNormalizationError(
            f"background manifest method {method!r} does not match records: {sorted(methods)}"
        )
    sizes = [record.n_ligands for record in records]
    scales = [record.scale for record in records]
    return {
        "schema_version": SCHEMA_VERSION,
        "panel_id": str(panel_id),
        "panel_sha256": str(panel_sha256),
        "engine": str(engine),
        "engine_version": str(engine_version),
        "method": method,
        "receptors": int(len(records)),
        "background_ligands_min": int(min(sizes)),
        "background_ligands_max": int(max(sizes)),
        "scale_min": float(min(scales)),
        "scale_median": float(np.median(np.asarray(scales, dtype=float))),
        "scale_max": float(max(scales)),
        "semantics": (
            "score is displacement below the receptor's own background in robust "
            "sigma units; higher is better; not a probability and not an affinity"
        ),
    }
