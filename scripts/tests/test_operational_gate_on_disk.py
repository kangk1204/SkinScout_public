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
    # F24: the stored runtime index predates the per-source release/license
    # contract, so the gate must refuse it until the separate data-regeneration
    # job rebuilds the evidence manifest and runtime index. Once that job has
    # run this test asserts the normal pass again.
    docking = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text(encoding="utf-8"))[
        "docking"
    ]
    index_manifest = ROOT / docking["daina_recipe_index_dir"] / "manifest.json"
    if index_manifest.exists():
        licensing = (
            json.loads(index_manifest.read_text(encoding="utf-8")).get(
                "evidence_licensing"
            )
            or {}
        )
        if not licensing.get("source_metadata"):
            result = subprocess.run(
                [sys.executable, str(VALIDATOR), "check-operational", "--gate", str(GATE)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            assert result.returncode != 0
            assert "per-source release/license/source-manifest metadata" in result.stderr
            pytest.skip("F24 data regeneration pending for the stored runtime index")

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
    """A gate that passes for a recipe the run does not use proves nothing.

    `stage3_daina_zoete` loads `--recipe <config daina_recipe_path>` and then
    applies the gate's `operational_recipe_id` section from it. So the gate must
    bind that configured file, the id must exist in it, and the id must follow
    the evaluation decision: `promote_selected` ships the dev-selected recipe
    while `retain_frozen_baseline` falls back to the baseline section.
    """
    flag = json.loads(GATE.read_text(encoding="utf-8"))
    docking = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text(encoding="utf-8"))[
        "docking"
    ]
    recipe_path = ROOT / docking["daina_recipe_path"]
    if not recipe_path.exists():
        pytest.skip("the configured recipe artifact is not built here")

    runtime_recipe = flag.get("runtime_recipe") or {}
    assert Path(str(runtime_recipe.get("path"))).resolve() == recipe_path.resolve(), (
        "the gate must bind the recipe file the run loads"
    )
    payload = json.loads(recipe_path.read_text(encoding="utf-8"))
    selected = payload["selected_recipe"]["recipe_id"]
    baseline = payload["baseline_recipe"]["recipe_id"]
    operational = flag.get("operational_recipe_id")
    assert operational in {selected, baseline}, (
        "the operational recipe must be a section of the configured recipe file"
    )
    decision = flag.get("evaluation_decision") or {}
    if decision.get("promotion_decision") == "promote_selected":
        assert operational == selected
    elif decision.get("promotion_decision") == "retain_frozen_baseline":
        assert operational == baseline


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
