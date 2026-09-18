"""Route B measured: it widens coverage and barely ranks.

The first attempt measured a pocket-cold panel and found every pair got worse.
That was not a result - transfer needs a donor in the target's cluster, a cold
cluster has no training member, and the retrieval index is train-only, so cold
targets are unreachable by construction. These assertions pin both the
correction and the measurement that replaced it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BUILDER = ROOT / "eval" / "build_transfer_reachable_panel.py"
PANEL = ROOT / "data" / "transfer_reachable_panel_202608"
REPORT = ROOT / "docs" / "ROUTE_B_MEASURED_20260830.md"

pytestmark = pytest.mark.skipif(not PANEL.is_dir(), reason="reachable panel absent")


@pytest.fixture(scope="module")
def manifest():
    return json.loads((PANEL / "manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def measurement():
    path = PANEL / "measurement_tm40.json"
    if not path.is_file():
        pytest.skip("measurement has not been run")
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_reachable_population_matches_the_older_analysis(manifest) -> None:
    """+2,082 / +1,917 / +1,726 were quoted figures; they reproduce exactly."""
    assert manifest["populations"]["tm40"]["reachable"] == 2082
    assert manifest["populations"]["tm50"]["reachable"] == 1917
    assert manifest["populations"]["tm60"]["reachable"] == 1726


def test_most_evidence_free_targets_stay_out_of_reach(manifest) -> None:
    """Transfer closes about a fifth of the hole, not the hole."""
    tm40 = manifest["populations"]["tm40"]
    assert tm40["unreachable"] > 4 * tm40["reachable"]


def test_transfer_creates_ranks_where_there_were_none(measurement) -> None:
    """Without a donor every truth target scores zero and ties; the baseline
    therefore never places one anywhere useful."""
    assert measurement["base"]["top100"] == 0
    assert measurement["base"]["top10"] == 0
    assert measurement["transfer"]["top100"] > 0


def test_transfer_is_not_an_improvement_in_aggregate(measurement) -> None:
    """It is a coin flip that slightly worsens the typical case, which is why
    it belongs beside the ranking rather than inside it."""
    assert measurement["transfer"]["median"] > measurement["base"]["median"]
    assert measurement["worsened"] > measurement["improved"]


def test_the_report_explains_why_the_cold_panel_could_not_measure_this() -> None:
    report = REPORT.read_text(encoding="utf-8")

    assert "정의에 의해" in report
    assert "train만으로" in report
    # And it must own the correction rather than quietly restating it.
    assert "재현 못 함" in report and "틀렸다" in report


def test_the_builder_records_why_a_cold_panel_is_the_wrong_instrument() -> None:
    source = BUILDER.read_text(encoding="utf-8")
    assert "structurally unable" in source
    assert "by construction" in source


@pytest.mark.parametrize("level", ["tm40", "tm50", "tm60"])
def test_every_level_was_measured(level: str) -> None:
    path = PANEL / f"measurement_{level}.json"
    assert path.is_file(), f"{level} has not been measured"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["base"]["top100"] == 0
    assert payload["pairs"] > 500


def test_tighter_clusters_transfer_better() -> None:
    """Borrowing from a more structurally similar pocket is more justified,
    and the measurement follows that: tm40 is significantly worse than not
    transferring, tm60 significantly better."""
    means = {}
    for level in ("tm40", "tm50", "tm60"):
        payload = json.loads((PANEL / f"measurement_{level}.json").read_text(encoding="utf-8"))
        means[level] = payload["transfer"]["mean"]

    assert means["tm40"] > means["tm50"] > means["tm60"]
    # tm40 is the only level that ends up worse than the baseline it replaces.
    tm40 = json.loads((PANEL / "measurement_tm40.json").read_text(encoding="utf-8"))
    tm60 = json.loads((PANEL / "measurement_tm60.json").read_text(encoding="utf-8"))
    assert tm40["transfer"]["mean"] > tm40["base"]["mean"]
    assert tm60["transfer"]["mean"] < tm60["base"]["mean"]
    assert tm60["improved"] > tm60["worsened"]


def test_the_report_names_the_level_to_use_and_the_one_to_avoid() -> None:
    report = REPORT.read_text(encoding="utf-8")

    assert "tm40은 쓰지 말아야 한다" in report
    assert "p=0.0022" in report
    # Better is not the same as usable, and the report has to say so.
    assert "덜 틀렸다" in report
