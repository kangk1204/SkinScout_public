"""The re-ranking experiment, pinned to the data committed with it.

Every number in docs/RERANK_EXPERIMENT_20260828.md is recomputed here from
data/validation/rerank_band_20260828/, so the report cannot drift away from
its evidence the way the retrospective figures did.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "analyse_rerank_band.py"
SCORES = ROOT / "data" / "validation" / "rerank_band_20260828"
REPORT = ROOT / "docs" / "RERANK_EXPERIMENT_20260828.md"

pytest.importorskip("scipy")


def _module():
    spec = importlib.util.spec_from_file_location("rerank_band_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def table():
    module = _module()
    frames, known = module.load_cases(SCORES, SCORES / "known_targets.csv")
    return module, module.collect(frames, known, 10, 50), frames, known


def test_the_evidence_is_committed_with_the_report() -> None:
    assert (SCORES / "known_targets.csv").is_file()
    compounds = [p for p in SCORES.iterdir() if p.is_dir()]
    assert len(compounds) == 15
    for directory in compounds:
        for name in ("selected.csv", "autodock.tsv", "gnina.tsv"):
            assert (directory / name).is_file(), f"{directory.name}/{name}"


def test_docking_alone_does_not_beat_similarity(table) -> None:
    """The headline the report leads with."""
    _, rows, _, _ = table

    assert len(rows) == 28
    assert rows.daina.mean() == pytest.approx(43.2, abs=0.1)
    assert rows.dg.dropna().mean() == pytest.approx(100.2, abs=0.1)
    assert rows.gnina.dropna().mean() == pytest.approx(79.6, abs=0.1)
    # Re-ranking the whole list is not an improvement.
    assert rows.rrf.mean() > rows.daina.mean()


def test_holding_the_head_and_reranking_the_band_is_an_improvement(table) -> None:
    module, rows, _, _ = table

    assert rows.hybrid.mean() == pytest.approx(37.5, abs=0.1)
    assert int((rows.daina <= 30).sum()) == 14
    assert int((rows.hybrid <= 30).sum()) == 20
    assert module._wilcoxon(rows.daina, rows.hybrid) == pytest.approx(0.0068, abs=0.0005)


def test_reranking_only_hurts_where_similarity_is_already_confident(table) -> None:
    """0 of 8 top-10 pairs improved; 11 of 13 in the 11-50 band did."""
    _, rows, _, _ = table

    head = rows[rows.daina <= 10]
    band = rows[(rows.daina > 10) & (rows.daina <= 50)]

    assert len(head) == 8
    assert int((head.rrf < head.daina).sum()) == 0
    assert len(band) == 13
    assert int((band.rrf < band.daina).sum()) == 11


def test_the_thresholds_are_not_load_bearing(table) -> None:
    """The structure carries the effect, not the exact cut points."""
    module, _, frames, known = table

    significant = 0
    total = 0
    for keep in (5, 8, 10, 12, 15, 20):
        for band in (40, 50, 60, 80):
            rows = module.collect(frames, known, keep, band)
            total += 1
            if module._wilcoxon(rows.daina, rows.hybrid) < 0.05:
                significant += 1
    assert significant >= total - 2, f"only {significant}/{total} combinations held"


def test_no_single_compound_carries_the_result(table) -> None:
    module, rows, _, _ = table

    for compound in sorted(rows.compound.unique()):
        subset = rows[rows.compound != compound]
        probability = module._wilcoxon(subset.daina, subset.hybrid)
        assert probability < 0.05, f"dropping {compound} breaks it (p={probability:.4f})"


def test_the_report_quotes_the_numbers_it_computed(table) -> None:
    _, rows, _, _ = table
    report = REPORT.read_text(encoding="utf-8")

    assert f"{rows.daina.mean():.1f}" in report
    assert f"{rows.hybrid.mean():.1f}" in report
    assert "p = 0.0068" in report
    assert re.search(r"14/28", report) and re.search(r"\*\*20/28\*\*", report)
