"""R02: the rendered status must reflect the official operational-gate verdict.

`render_current_status.py` used to read `evaluation_decision` and
`claim_gate_status` from the gate flag and print them as the current state. When
the runtime index manifest was regenerated without rebinding the gate (the R01
situation), `validate_activity_retrieval_gate.py check-operational` failed while
the status block still showed PASS and `--check` still said "up to date".

These tests build one full valid chain + operational gate fixture. In the valid
case the official checker and the block must both report verification; in the
refreshed-manifest case they must both report failed/unverified, with the stored
evaluation numbers kept only as history.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "tests"))

from render_current_status import (
    current_state,
    document_with_block,
    render_block,
)
from test_validate_activity_retrieval_gate import (
    _contract,
    _operational_args,
    _record,
    _write_json,
)
from validate_activity_retrieval_gate import (
    check_operational_gate,
    create_operational_gate,
)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "root"
    (root / "workflow").mkdir(parents=True)
    (root / "data" / "manifests").mkdir(parents=True)

    contract = _contract(tmp_path)
    runtime = _operational_args(contract, tmp_path)
    recipe_path = Path(runtime["runtime_recipe"])
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    selected = recipe["selected_recipe"]["recipe_id"]
    baseline = recipe["baseline_recipe"]["recipe_id"]

    final_manifest_path = contract["final_evaluation_manifest"]
    final_manifest = json.loads(final_manifest_path.read_text(encoding="utf-8"))
    summary_path = Path(final_manifest["outputs"]["summary.json"]["path"])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["ranking"] = {
        "full_known_skin_panel_15_cases": {
            selected: {
                "n_truth_pairs": 46,
                "top10": 26 / 46,
                "top30": 34 / 46,
                "mrr": 0.56187,
            },
            baseline: {
                "n_truth_pairs": 46,
                "top10": 18 / 46,
                "top30": 31 / 46,
                "mrr": 0.43224,
            },
        },
        "temporal_test_2025": {
            selected: {
                "n_truth_pairs": 301,
                "top10": 222 / 301,
                "top30": 237 / 301,
                "mrr": 0.56187,
            },
            baseline: {
                "n_truth_pairs": 301,
                "top10": 187 / 301,
                "top30": 216 / 301,
                "mrr": 0.43224,
            },
        },
        "dual_cold": {
            selected: {"mrr": 0.00008095},
            baseline: {"mrr": 0.00008293},
        },
    }
    _write_json(summary_path, summary)
    final_manifest["outputs"]["summary.json"] = _record(summary_path)
    _write_json(final_manifest_path, final_manifest)

    gate_path = root / "data" / "manifests" / "activity_retrieval_operational_gate.flag"
    create_operational_gate(**contract, **runtime, out_gate=gate_path)

    config = {
        "docking": {
            "daina_recipe_scoring": True,
            "daina_recipe_path": str(recipe_path),
            "daina_recipe_index_dir": str(
                Path(runtime["runtime_index_manifest"]).parent
            ),
            "fast_mode_rerank_enabled": False,
        }
    }
    (root / "workflow" / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    return root, gate_path, Path(runtime["runtime_index_manifest"])


def test_valid_chain_and_gate_render_verified(tmp_path: Path) -> None:
    root, gate_path, _ = _fixture(tmp_path)
    check_operational_gate(gate_path)

    state = current_state(root)
    assert state["operational_validation"]["status"] == "pass"
    block = render_block(state)
    assert "현재 운영 검증: **통과**" in block
    assert "미검증" not in block
    assert "역사 기록" not in block
    assert "Top10 26/46" in block
    assert "Top30 34/46" in block


def test_refreshed_runtime_index_without_rebinding_is_unverified(
    tmp_path: Path,
) -> None:
    root, gate_path, runtime_manifest = _fixture(tmp_path)

    # R01: the runtime index manifest is regenerated (same semantics, new bytes)
    # and the operational gate is not rebound to it.
    payload = json.loads(runtime_manifest.read_text(encoding="utf-8"))
    runtime_manifest.write_text(
        json.dumps(payload, indent=4) + "\n", encoding="utf-8"
    )

    with pytest.raises(SystemExit) as excinfo:
        check_operational_gate(gate_path)
    assert "runtime_index_manifest is stale" in str(excinfo.value)

    state = current_state(root)
    assert state["operational_validation"]["status"] == "fail"
    assert (
        "runtime_index_manifest is stale"
        in state["operational_validation"]["detail"]
    )

    block = render_block(state)
    assert "**실패(미검증)**" in block
    assert "runtime_index_manifest is stale" in block
    assert "기본 MODE-FAST 표적 분석은 차단된다" in block
    # The stored evaluation numbers survive, explicitly as history.
    assert "Top10 26/46" in block
    assert "Top30 34/46" in block
    assert "(역사 기록)" in block

    # A document still carrying the old PASS block can no longer pass --check.
    document = (
        "intro\n"
        "<!-- BEGIN GENERATED: current-operational-state -->\n"
        "old pass block\n"
        "<!-- END GENERATED: current-operational-state -->\n"
    )
    assert document_with_block(document, block) != document
