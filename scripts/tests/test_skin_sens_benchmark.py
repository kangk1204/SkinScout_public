"""The first measurement of the safety layer, pinned to its data.

Until 2026-08-30 nothing compared HALT/FLAG_HIGH/PASS against measured
sensitisation. These assertions keep the published numbers tied to the
predictions they came from.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "analyse_skin_sens_benchmark.py"
DATA = ROOT / "data" / "validation" / "skin_sens_benchmark_20260830"
REPORT = ROOT / "docs" / "SKIN_SENS_BENCHMARK_20260830.md"

pytestmark = pytest.mark.skipif(not DATA.is_dir(), reason="benchmark data absent")


def _module():
    spec = importlib.util.spec_from_file_location("sens_benchmark_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def frame():
    module = _module()
    return module, module.load(DATA)


def test_the_shipped_rule_almost_never_clears_anything(frame) -> None:
    """PASS three times in a hundred is a screen, not a discriminator."""
    _, table = frame

    assert len(table) == 100
    counts = table.decision.value_counts().to_dict()
    assert counts["HALT"] == 70
    assert counts["FLAG_HIGH"] == 27
    assert counts["PASS"] == 3
    # No false negatives - which is the right direction, and the reason the
    # rule is defensible even though it flags 97%.
    assert int(table[table.decision == "PASS"].truth.sum()) == 0


def test_one_model_is_worse_than_chance_and_still_gets_a_vote(frame) -> None:
    module, table = frame

    metrics = module.binary_metrics(table.pred_skin_pos, table.truth)
    assert metrics["balanced"] < 0.5, "Pred-Skin was above chance; re-read the report"
    # And it carries the same weight as the others in the shipped rule.
    rule = (ROOT / "scripts" / "stage2_consensus.py").read_text(encoding="utf-8")
    assert "halt_min_votes" in rule


def test_the_consensus_is_worse_than_its_best_member(frame) -> None:
    """Averaging a strong model with two weak ones loses the strong one."""
    module, table = frame

    best = module._auc(table.truth, table.stoptox_prob)
    consensus = module._auc(table.truth, table.votes)
    assert isinstance(best, float) and isinstance(consensus, float)
    assert best == pytest.approx(0.969, abs=0.002)
    assert consensus == pytest.approx(0.758, abs=0.002)
    assert consensus < best


def test_the_report_refuses_to_promote_a_model_on_this_evidence() -> None:
    """The benchmark is public and predates the models; contamination is not
    ruled out, so a high score cannot be read as generalisation."""
    report = REPORT.read_text(encoding="utf-8")
    source = (DATA / "SOURCE.md").read_text(encoding="utf-8")

    assert "학습됐는지 알 수 없다" in report
    assert "구분할 수 없다" in report
    assert "학습됐는지 알 수 없다" in source
    # ADMET-AI is trained on TDC, so it must be excluded from a TDC benchmark.
    assert "순환" in source


def test_the_guide_carries_the_measured_numbers() -> None:
    guide = (ROOT / "docs" / "RESEARCHER_GUIDE.md").read_text(encoding="utf-8")

    assert "47.3%" in guide and "0.969" in guide
    assert "97%를 flag" in guide
    assert "triage" in guide
