"""Regression tests for cross-receptor docking score normalization."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "eval"))

from docking_normalization import (  # noqa: E402
    DockingNormalizationError,
    ReceptorBackground,
    background_manifest,
    build_receptor_background,
    normalize_screen,
    normalized_score,
)


def _scores(center: float, spread: float, n: int = 60) -> list[float]:
    rng = np.random.default_rng(0)
    return (center + spread * rng.standard_normal(n)).tolist()


def test_background_summarises_a_receptor_robustly() -> None:
    background = build_receptor_background("P1", _scores(-6.0, 0.8))

    assert background.target_id == "P1"
    assert background.n_ligands == 60
    assert background.center == pytest.approx(-6.0, abs=0.3)
    assert background.scale == pytest.approx(0.8, abs=0.3)
    assert background.method == "robust"


def test_robust_background_resists_an_outlier_that_moves_the_mean() -> None:
    clean = _scores(-6.0, 0.5)
    contaminated = [*clean, -60.0]

    robust = build_receptor_background("P1", contaminated, min_ligands=30)
    gaussian = build_receptor_background(
        "P1", contaminated, method="gaussian", min_ligands=30
    )

    assert robust.center == pytest.approx(-6.0, abs=0.3)
    assert abs(gaussian.center + 6.0) > abs(robust.center + 6.0)


def test_normalization_removes_a_pure_receptor_offset() -> None:
    """Two receptors differing only by a constant offset must rank a query equally.

    This is the bias that makes a raw-dG proteome screen rank pockets by size.
    """
    deep = build_receptor_background("deep", _scores(-9.0, 0.7))
    shallow = build_receptor_background("shallow", _scores(-5.0, 0.7))

    # A query one background-sigma better than each receptor's own typical ligand.
    deep_z = normalized_score(deep.center - deep.scale, deep)
    shallow_z = normalized_score(shallow.center - shallow.scale, shallow)

    assert deep_z == pytest.approx(1.0)
    assert shallow_z == pytest.approx(1.0)
    # Raw energies would have ranked the deep pocket far ahead.
    assert (deep.center - deep.scale) < (shallow.center - shallow.scale) - 3.0


def test_normalized_score_is_zero_at_the_background_centre() -> None:
    background = build_receptor_background("P1", _scores(-7.0, 1.0))

    assert normalized_score(background.center, background) == pytest.approx(0.0)


def test_background_rejects_a_thin_panel() -> None:
    with pytest.raises(DockingNormalizationError, match="at least"):
        build_receptor_background("P1", [-6.0, -6.1, -5.9])


def test_background_rejects_a_receptor_that_cannot_discriminate() -> None:
    with pytest.raises(DockingNormalizationError, match="no usable spread"):
        build_receptor_background("P1", [-6.0] * 40)


def test_background_rejects_non_finite_scores() -> None:
    with pytest.raises(DockingNormalizationError, match="finite"):
        build_receptor_background("P1", [*_scores(-6.0, 1.0), float("inf")])


def test_background_rejects_an_unknown_method() -> None:
    with pytest.raises(DockingNormalizationError, match="method"):
        build_receptor_background("P1", _scores(-6.0, 1.0), method="zscore")


def test_screen_refuses_to_score_a_receptor_with_no_background() -> None:
    backgrounds = {"P1": build_receptor_background("P1", _scores(-6.0, 1.0))}

    with pytest.raises(DockingNormalizationError, match="no background"):
        normalize_screen({"P1": -7.0, "P2": -8.0}, backgrounds)

    scored, missing = normalize_screen(
        {"P1": -7.0, "P2": -8.0}, backgrounds, require_background=False
    )
    assert set(scored) == {"P1"}
    assert missing == ["P2"]


def test_background_manifest_records_provenance_and_semantics() -> None:
    records = [
        build_receptor_background("P1", _scores(-6.0, 1.0)),
        build_receptor_background("P2", _scores(-8.0, 0.5)),
    ]

    manifest = background_manifest(
        records,
        panel_id="diverse-64",
        panel_sha256="a" * 64,
        engine="AutoDock Vina",
        engine_version="1.2.5",
    )

    assert manifest["schema_version"] == "skinscout.docking_background.v1"
    assert manifest["receptors"] == 2
    assert manifest["background_ligands_min"] == 60
    assert "not a probability" in manifest["semantics"]


def test_background_manifest_rejects_a_method_mismatch() -> None:
    record = ReceptorBackground("P1", 60, -6.0, 1.0, "robust", -8.0, -4.0)

    with pytest.raises(DockingNormalizationError, match="does not match"):
        background_manifest(
            [record],
            panel_id="p",
            panel_sha256="a" * 64,
            engine="Vina",
            engine_version="1.2.5",
            method="gaussian",
        )
