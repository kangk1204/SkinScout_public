"""Run the gate the pipeline runs, against the file the pipeline reads.

`rule daina_zoete` starts with

    python scripts/validate_activity_retrieval_gate.py check-operational --gate ...

so if that exits non-zero the default MODE-FAST path cannot start at all. On
2026-09-01 it did exit 1 while the whole suite was green, because every existing
gate test builds its own fixture and none of them opened
`data/manifests/activity_retrieval_operational_gate.flag`.

What broke it: adding a field to `Recipe` made FROZEN_BASELINE_RECIPE nine
fields while the recipe artifact the flag pointed at still had eight, and the
flag pointed into a scratch directory at a recipe that was no longer promoted.
Both are invisible to a fixture.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
GATE = ROOT / "data" / "manifests" / "activity_retrieval_operational_gate.flag"
VALIDATOR = ROOT / "scripts" / "validate_activity_retrieval_gate.py"


@pytest.mark.skipif(not GATE.exists(), reason="the operational gate flag is not built here")
def test_the_gate_the_run_path_checks_actually_passes() -> None:
    result = subprocess.run(
        [sys.executable, str(VALIDATOR), "check-operational", "--gate", str(GATE)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        "MODE-FAST cannot start: " + (result.stdout + result.stderr).strip()
    )


@pytest.mark.skipif(not GATE.exists(), reason="the operational gate flag is not built here")
def test_the_gate_and_the_config_name_the_same_recipe() -> None:
    """A gate that passes for a recipe the run does not use proves nothing."""
    flag = json.loads(GATE.read_text(encoding="utf-8"))
    docking = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text(encoding="utf-8"))[
        "docking"
    ]
    recipe_path = ROOT / docking["daina_recipe_path"]
    if not recipe_path.exists():
        pytest.skip("the configured recipe artifact is not built here")

    configured = json.loads(recipe_path.read_text(encoding="utf-8"))["selected_recipe"]["recipe_id"]
    assert flag.get("operational_recipe_id") == configured


@pytest.mark.skipif(not GATE.exists(), reason="the operational gate flag is not built here")
def test_the_gate_does_not_reference_a_scratch_directory() -> None:
    """It pointed at /tmp/.../scratchpad/dev_anyfeature/recipe.json, which is
    both outside the project and not the promoted recipe."""
    text = GATE.read_text(encoding="utf-8")

    assert "/tmp/" not in text, "the gate must bind artifacts that survive a reboot"


def test_the_frozen_baseline_matches_the_dataclass() -> None:
    """The specific mismatch that broke it, checked without needing the flag."""
    from dataclasses import asdict

    sys.path.insert(0, str(ROOT / "scripts"))
    from activity_retrieval_scoring import BASELINE
    from validate_activity_retrieval_gate import FROZEN_BASELINE_RECIPE

    assert asdict(BASELINE) == FROZEN_BASELINE_RECIPE
