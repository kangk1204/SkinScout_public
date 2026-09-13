"""Regression tests for volume-scaled docking search effort."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from stage3_autodock_run import (  # noqa: E402
    DEFAULT_HIGH_VOLUME_MIN_EFFORT,
    DEFAULT_SEARCH_VOLUME_ADVISORY,
    box_volume,
    sampling_effort,
)


def test_a_box_within_the_advisory_keeps_the_configured_effort() -> None:
    assert sampling_effort(8, 11_555.0) == 8
    assert sampling_effort(4, 27_000.0) == 4
    assert sampling_effort(20, 26_999.0) == 20


def test_a_box_over_the_advisory_reaches_the_measured_floor() -> None:
    """4, 8 and 16 all left a clashing pose at 1.5-2.1x the advisory; 32 did not."""
    # the Q8N434 box that produced the +13.46 kcal/mol clash
    assert sampling_effort(8, 40_726.0) == DEFAULT_HIGH_VOLUME_MIN_EFFORT
    # fast mode configures 4, below Vina's own default, and must still reach it
    assert sampling_effort(4, 40_726.0) == DEFAULT_HIGH_VOLUME_MIN_EFFORT


def test_a_far_larger_box_scales_past_the_floor() -> None:
    # a high comprehensive base on the largest observed box keeps scaling
    assert sampling_effort(20, 64_000.0) == 48
    assert sampling_effort(8, 64_000.0) == DEFAULT_HIGH_VOLUME_MIN_EFFORT


def test_effort_never_drops_below_the_configured_base() -> None:
    for volume in (1_000.0, 27_000.0, 30_000.0, 64_000.0):
        assert sampling_effort(64, volume) >= 64


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"base": 0, "volume": 1000.0}, "effort must be >= 1"),
        ({"base": 8, "volume": 0.0}, "volume must be positive"),
        ({"base": 8, "volume": float("nan")}, "volume must be positive"),
    ],
)
def test_sampling_effort_rejects_invalid_input(kwargs: dict, match: str) -> None:
    with pytest.raises(SystemExit, match=match):
        sampling_effort(**kwargs)


def test_sampling_effort_rejects_invalid_thresholds() -> None:
    with pytest.raises(SystemExit, match="advisory must be positive"):
        sampling_effort(8, 1000.0, advisory=0.0)
    with pytest.raises(SystemExit, match="floor must be >= 1"):
        sampling_effort(8, 1000.0, high_volume_floor=0)


def test_box_volume_multiplies_the_three_edges() -> None:
    assert box_volume((38.3, 29.0, 36.7)) == pytest.approx(38.3 * 29.0 * 36.7)
    assert box_volume((30.0, 30.0, 30.0)) == pytest.approx(DEFAULT_SEARCH_VOLUME_ADVISORY)


def test_workflow_rules_pass_the_volume_scaling_knobs() -> None:
    """Both docking rules must forward the thresholds, not silently use defaults."""
    import yaml

    config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())
    docking = config["docking"]
    assert docking["search_volume_advisory"] == 27000
    assert docking["high_volume_min_runs"] == DEFAULT_HIGH_VOLUME_MIN_EFFORT

    for rule_file in ("stage3a_comprehensive.smk", "stage3b_fast.smk"):
        text = (ROOT / "workflow" / "rules" / rule_file).read_text()
        assert "--search-volume-advisory {params.volume_advisory:q}" in text, rule_file
        assert "--high-volume-min-nrun {params.high_volume_runs:q}" in text, rule_file
        assert 'config["docking"]["search_volume_advisory"]' in text, rule_file
        assert 'config["docking"]["high_volume_min_runs"]' in text, rule_file


def test_a_missing_box_leaves_the_gpu_path_runnable(tmp_path: Path) -> None:
    """AutoDock-GPU encodes the box in its maps and must not need the file.

    Scaling the effort by box volume must not turn an optional input into a
    required one for the engine that never read it.
    """
    import subprocess

    script = ROOT / "scripts" / "stage3_autodock_run.py"
    text = script.read_text()
    # the GPU branch only parses the box opportunistically
    assert 'box_geometry = parse_box(box) if args.engine == "vina" else None' in text
    assert "unscaled_effort_targets += 1" in text
    # and reports when it could not scale
    assert "was not scaled to the box volume" in text

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--search-volume-advisory" in result.stdout
    assert "--high-volume-min-nrun" in result.stdout
