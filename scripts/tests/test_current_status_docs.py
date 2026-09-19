"""Pin the current-state statements to the active gate, recipe, and summary.

The status document used to be hand-written, so it kept claiming an operational
PASS and a `union_any_consensus` recipe with Top30 35/46 after the active gate
had moved to `union_discount_light` and a diagnostic runtime binding, and the
hash-bound summary said 34/46. The current-state block is generated, not typed:
these tests regenerate it and fail if the document, the config rationale, or the
history labels drift.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from render_current_status import (  # noqa: E402
    GATE,
    STATUS_DOC,
    current_state,
    document_with_block,
    render_block,
)

CONFIG = ROOT / "workflow" / "config.yaml"
GUIDE = ROOT / "docs" / "RESEARCHER_GUIDE.md"


def _skip_reason() -> str | None:
    if not GATE.is_file():
        return f"active operational gate is not built here: {GATE}"
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    summary = gate.get("final_summary")
    if not isinstance(summary, dict) or not summary.get("path"):
        return "active operational gate does not bind a summary"
    if not Path(str(summary["path"])).is_file():
        return f"bound summary is not built here: {summary['path']}"
    docking = yaml.safe_load(CONFIG.read_text(encoding="utf-8")).get("docking", {})
    recipe = ROOT / str(docking.get("daina_recipe_path") or "")
    if not recipe.is_file():
        return f"active recipe is not built here: {recipe}"
    return None


pytestmark = pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")


@pytest.fixture(scope="module")
def state() -> dict:
    return current_state()


def test_the_generated_block_matches_the_active_artifacts(state: dict) -> None:
    document = STATUS_DOC.read_text(encoding="utf-8")
    assert document_with_block(document, render_block(state)) == document, (
        "docs/IMPLEMENTATION_STATUS.md current-state block is stale; run "
        "python scripts/render_current_status.py --write"
    )


def test_current_claims_name_the_active_recipe_and_diagnostic_binding(
    state: dict,
) -> None:
    document = STATUS_DOC.read_text(encoding="utf-8")
    assert "`union_discount_light`" in document
    assert "diagnostic_runtime_binding" in document
    assert "claim_ready=false" in document
    assert state["recipe_id"] == "union_discount_light"
    assert state["runtime_decision"]["promotion_decision"] == "diagnostic_runtime_binding"
    assert state["runtime_decision"]["claim_ready"] is False


def test_current_known_panel_numbers_are_generated_from_the_summary(
    state: dict,
) -> None:
    document = STATUS_DOC.read_text(encoding="utf-8")
    selected = state["known_selected"]
    assert selected["top10"] == 26 and selected["top30"] == 34 and selected["pairs"] == 46
    assert "Top10 26/46" in document
    assert "Top30 34/46" in document
    # The superseded 35/46 has no current-state home.
    assert "35/46" not in document


def test_config_rationale_quotes_the_hash_bound_figure() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    rationale = config["evaluation"]["activity_retrieval"]["prefer_recipe_rationale"]
    assert "34/46" in rationale
    assert "35/46" in rationale, "the stale figure must be named as superseded"
    assert "ChEMBL-only" in rationale


def test_historical_promotion_is_not_presented_as_current() -> None:
    document = STATUS_DOC.read_text(encoding="utf-8")
    generated_end = document.index("<!-- END GENERATED: current-operational-state -->")
    assert "union_any_consensus" not in document[:generated_end]
    for line in document.splitlines():
        if "union_any_consensus" in line:
            assert "역사" in line, (
                "the replaced recipe may only appear in an explicitly historical sentence"
            )
    assert "union_discount_light" in document[:generated_end]


def test_researcher_guide_band_policy_matches_the_disabled_config() -> None:
    guide = GUIDE.read_text(encoding="utf-8")
    docking = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["docking"]
    assert docking["fast_mode_rerank_enabled"] is False
    assert "밴드 정책은 현재 파이프라인에 반영돼 있다" not in guide
    assert "밴드 재정렬은 현재 기본 설정에서 꺼져 있다" in guide
    # The historical 35/46 table stays, clearly labeled.
    assert "## 4. 과거 평가 스냅샷 (현재 성능 주장에 사용 금지)" in guide
    assert "역사적" in guide
