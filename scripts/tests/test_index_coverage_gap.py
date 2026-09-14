"""Coverage claims have to be measured against the table a run actually reads.

The expansion note was wrong three times, each time because it measured the
wrong thing:

1. "+764 targets from dropping the train-only restriction" - only 133 of those
   come from the split; 631 were removed by the benchmark's quality filters.
2. Skin-band coverage quoted against the evidence pool rather than the index,
   which read 3-4 points high.
3. Worst: the retrieval index is an evaluation artefact. Stage 3 retrieves
   against `data/chembl37/activity_evidence.parquet`, so the whole "AQP3 is
   missing" story was about a file no user run opens. AQP3 is reachable today.

These tests pin the corrected numbers to the data, so the next time the evidence
moves the test fails before the document becomes wrong again.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

BENCHMARK = ROOT / "data" / "activity_benchmark_202608"
RUNTIME_EVIDENCE = ROOT / "data" / "chembl37" / "activity_evidence.parquet"
NOTE = ROOT / "docs" / "DATA_EXPANSION_20260831.md"

pytestmark = pytest.mark.skipif(
    not (BENCHMARK / "train.parquet").exists(),
    reason="activity benchmark splits are not provisioned here",
)


def _targets(path: Path) -> set[str]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["uniprot"])
    return {value for value in table.column("uniprot").to_pylist() if value}


@pytest.fixture(scope="module")
def split_targets() -> dict[str, set[str]]:
    return {name: _targets(BENCHMARK / f"{name}.parquet") for name in ("train", "dev", "test")}


@pytest.fixture(scope="module")
def runtime_targets() -> set[str]:
    if not RUNTIME_EVIDENCE.exists():
        pytest.skip("ChEMBL mirror is not provisioned here")
    return _targets(RUNTIME_EVIDENCE)


@pytest.fixture(scope="module")
def note_text() -> str:
    return NOTE.read_text(encoding="utf-8")


def test_a_run_reaches_4727_targets_not_4510(runtime_targets: set[str]) -> None:
    """The number a researcher experiences, from the table Stage 3 opens."""
    assert len(runtime_targets) == 4_727


def test_aqp3_is_already_reachable_by_a_real_run(runtime_targets: set[str]) -> None:
    """The retraction that mattered most.

    Two drafts of the note led with "AQP3 is missing, so a moisturising
    ingredient cannot surface it". It was measured against the benchmark index.
    """
    assert "Q92482" in runtime_targets


def test_the_melanin_pathway_really_is_half_missing(runtime_targets: set[str]) -> None:
    """TYR is reachable; TYRP1 and DCT are in no public source at all.

    This one survived every correction, and it is what priority 2 is for.
    """
    assert "P14679" in runtime_targets, "TYR"
    assert "P17643" not in runtime_targets, "TYRP1"
    assert "P40126" not in runtime_targets, "DCT"


def test_the_index_is_built_from_the_train_split_only(split_targets) -> None:
    assert len(split_targets["train"]) == 4_510


def test_dropping_the_split_restriction_adds_133_targets_not_764(split_targets) -> None:
    everything = split_targets["train"] | split_targets["dev"] | split_targets["test"]

    assert len(everything - split_targets["train"]) == 133
    assert len(everything) == 4_643


def test_the_note_leads_with_the_runtime_table(note_text: str) -> None:
    assert "data/chembl37/activity_evidence.parquet" in note_text
    assert "4,727" in note_text
    assert "+547" in note_text
    # And the three retractions stay on the page.
    assert "## 틀렸던 것" in note_text
    assert "AQP3가 빠져 있다" in note_text and "이미 있다" in note_text


def test_the_note_no_longer_proposes_rebuilding_the_evaluation_index(
    note_text: str,
) -> None:
    """It has no consumer; proposing it again would cost a rebuild for nothing."""
    assert "소비자가 없다" in note_text
    assert "+764 표적을 얻는다" not in note_text


def test_no_document_still_quotes_the_pool_based_skin_coverage() -> None:
    """64.3% is the number after merging every source, not the number today.

    A reader deciding whether their target is covered is running the tool, and a
    run reaches 60.4% of the high band.
    """
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "발현되는 것만 보면 64%" not in readme
    assert "발현되는 것만 보면 60%" in readme


def test_the_analysis_script_names_the_runtime_table() -> None:
    source = (ROOT / "scripts" / "analyse_index_coverage_gap.py").read_text(encoding="utf-8")
    assert "RUNTIME_EVIDENCE" in source
    assert "chembl37" in source
    assert "skin_score.tsv" in source
