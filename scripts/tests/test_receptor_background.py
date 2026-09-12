"""Route C: the bias is real, but removing it does not improve recovery.

docs/COVERAGE_ROUTES_20260824.md sized the production background build at about
eight days and left this untested. It is testable from the panel score tables
already committed, and the answer is negative - which is what retires the eight
days rather than spending them.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "analyse_receptor_background.py"
SCORES = ROOT / "data" / "validation" / "rerank_band_20260828"
REPORT = ROOT / "docs" / "COVERAGE_ROUTE_C_RETIRED_20260829.md"

pytestmark = pytest.mark.skipif(not SCORES.is_dir(), reason="panel score tables absent")


def _module():
    spec = importlib.util.spec_from_file_location("background_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def comparison():
    module = _module()
    scores, known = module.load(SCORES)
    return module, module.compare(scores, known)


def test_normalisation_does_not_improve_recovery(comparison) -> None:
    module, table = comparison
    pytest.importorskip("scipy")
    from scipy.stats import wilcoxon

    assert len(table) == 19
    assert table.raw.mean() == pytest.approx(38.9, abs=0.1)
    assert table.normalised.mean() == pytest.approx(38.8, abs=0.1)
    better = int((table.normalised < table.raw).sum())
    worse = int((table.normalised > table.raw).sum())
    assert (better, worse) == (9, 10)
    assert wilcoxon(table.raw, table.normalised)[1] > 0.5


def test_the_background_never_contains_the_query(comparison) -> None:
    """Leaking the query into its own background would fabricate the result."""
    module, _ = comparison
    source = SCRIPT.read_text(encoding="utf-8")

    body = source.split("def compare(", 1)[1]
    assert "for name, series in scores.items() if name != case" in body


def test_a_thin_background_is_not_used(comparison) -> None:
    """A mean over two compounds is noise, not a receptor's background."""
    module, _ = comparison
    scores, known = module.load(SCORES)

    thin = module.compare(scores, known, min_background=14)
    deep = module.compare(scores, known, min_background=2)

    assert len(thin) <= len(deep)
    assert module.DEFAULT_MIN_BACKGROUND >= 6


def test_the_report_quotes_what_the_script_computes(comparison) -> None:
    _, table = comparison
    report = REPORT.read_text(encoding="utf-8")

    assert "p = 0.83" in report
    assert "개선 9쌍 / 악화 10쌍" in report
    assert f"{len(table)}쌍" in report
    # And it must not overclaim a negative.
    assert "개선한다는 증거가 없다" in report
    assert "증명" in report


def test_the_pocket_cluster_figures_the_report_quotes_are_real() -> None:
    """The older coverage document is quoted, so its numbers were re-derived.

    Infrastructure counts check out exactly; the coverage gains it reports do
    not have artifacts in the repository and are labelled as unreproduced.
    """
    pd = pytest.importorskip("pandas")
    root = ROOT / "data"
    evidence = root / "pocket_clusters_202608" / "pocket_cluster_map.csv"
    screenable = root / "pocket_clusters_screenable_202608" / "pocket_cluster_map.csv"
    if not (evidence.is_file() and screenable.is_file()):
        pytest.skip("pocket cluster maps have not been built")

    left = pd.read_csv(evidence)
    right = pd.read_csv(screenable)

    def available(frame):
        column = frame["pocket_available"]
        if column.dtype == object:
            return int(column.astype(str).str.lower().isin({"true", "1"}).sum())
        return int(column.sum())

    assert available(left) == 4071
    assert available(right) == 15421
    assert left["pocket_cluster_tm50"].nunique() == 2840
    assert right["pocket_cluster_tm50"].nunique() == 10890

    report = REPORT.read_text(encoding="utf-8")
    assert "재현 못 함" in report, "unreproduced figures must be labelled"
    assert "낡음" in report, "the stale Vina cost must be labelled"
