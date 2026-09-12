"""Pin the researcher guide's figures to the artifacts they were measured from.

The guide tells a wet-lab reader when to believe the tool, so a number that
silently goes stale is worse than no number. Each test re-derives one figure
from committed output and fails if the document and the artifacts disagree.
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / "docs" / "RESEARCHER_GUIDE.md"
LQO_DIR = (
    ROOT
    / "results"
    / "eval"
    / "skin_known_full_20260826"
    / "daina_leave_query_out"
)


# results/eval/ is gitignored, so a fresh clone has no evaluation output to
# compare the guide against. Skip rather than fail: the figures are only
# checkable where the run that produced them is present.
pytestmark = pytest.mark.skipif(
    not (LQO_DIR / "skin_known_targets.csv").exists(),
    reason=f"evaluation output not present: {LQO_DIR}",
)


# Named in prose as run outputs; they live under a run directory, not at the
# repository root, so their absence here is not a broken reference.
RUN_ARTIFACT_NAMES = {
    "run_summary.md",
    "top50_4way_consensus.csv",
    # The similarity-ordered artifact the band re-ranking is compared against.
    "daina_structural_targets.csv",
    "top50_band_reranked.csv",
    "ranked_targets_v3.csv",
    "ranked_targets_v3_with_efficacy.csv",
}


@pytest.fixture(scope="module")
def guide_text() -> str:
    return GUIDE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def lqo_pairs_with_case(lqo_pairs: pd.DataFrame) -> pd.DataFrame:
    return lqo_pairs


@pytest.fixture(scope="module")
def lqo_pairs() -> pd.DataFrame:
    targets = pd.read_csv(LQO_DIR / "skin_known_targets.csv")
    rows = []
    for _, record in targets.iterrows():
        ranking = LQO_DIR / "rankings" / f"{record.case_id}__ranked_targets_v3.csv"
        if not ranking.exists():
            continue
        frame = pd.read_csv(ranking)
        hit = frame[frame["target_id"] == record["target_id"]]
        if hit.empty:
            continue
        rank = pd.to_numeric(record["rank"], errors="coerce")
        if pd.isna(rank):
            continue
        rows.append({"case": record.case_id, "rank": float(rank), "similarity": float(hit["score"].iloc[0])})
    return pd.DataFrame(rows)


def test_the_similarity_bins_match_the_evaluation(lqo_pairs: pd.DataFrame) -> None:
    """The table the guide's "when to believe it" section rests on."""
    expected = {
        ">=0.6": {"n": 12, "median": 8, "top30": 11},
        "0.4-0.6": {"n": 13, "median": 45, "top30": 2},
        "<0.4": {"n": 6, "median": 316.5, "top30": 1},
    }
    bounds = {">=0.6": (0.6, 1.01), "0.4-0.6": (0.4, 0.6), "<0.4": (0.0, 0.4)}
    for label, (low, high) in bounds.items():
        group = lqo_pairs[
            (lqo_pairs["similarity"] >= low) & (lqo_pairs["similarity"] < high)
        ]
        assert len(group) == expected[label]["n"], label
        assert group["rank"].median() == expected[label]["median"], label
        assert int((group["rank"] <= 30).sum()) == expected[label]["top30"], label


def test_the_two_lower_bins_measure_the_same_thing(lqo_pairs: pd.DataFrame) -> None:
    """Why the guide offers no three-step confidence tier.

    Top-30 recovery below 0.6 does not separate into two bands, so the lower
    two steps of any three-step badge would be reporting the same number.
    """
    middle = lqo_pairs[(lqo_pairs["similarity"] >= 0.4) & (lqo_pairs["similarity"] < 0.6)]
    lowest = lqo_pairs[lqo_pairs["similarity"] < 0.4]
    assert (middle["rank"] <= 30).mean() == pytest.approx(
        (lowest["rank"] <= 30).mean(), abs=0.03
    )


def test_the_similarity_boundary_the_guide_names_is_where_recovery_breaks(
    lqo_pairs: pd.DataFrame,
) -> None:
    """0.6 is quoted as the practical boundary, so it has to be one."""
    above = lqo_pairs[lqo_pairs["similarity"] >= 0.6]
    below = lqo_pairs[lqo_pairs["similarity"] < 0.6]
    assert (above["rank"] <= 30).mean() > 0.9
    assert (below["rank"] <= 30).mean() < 0.2


def test_pairs_within_a_compound_are_not_independent_observations(
    lqo_pairs_with_case: pd.DataFrame,
) -> None:
    """The guide quotes compound counts beside pair counts for a reason.

    One compound contributes several target pairs, so 31 pairs are nowhere near
    31 independent measurements.
    """
    for low, high, pairs, compounds in (
        (0.6, 1.01, 12, 7),
        (0.4, 0.6, 13, 8),
        (0.0, 0.4, 6, 4),
    ):
        group = lqo_pairs_with_case[
            (lqo_pairs_with_case["similarity"] >= low)
            & (lqo_pairs_with_case["similarity"] < high)
        ]
        assert len(group) == pairs
        assert group["case"].nunique() == compounds


def test_no_committed_run_has_ever_passed(guide_text: str) -> None:
    decisions: dict[str, int] = {}
    claimable = set()
    for path in glob.glob(str(ROOT / "results" / "runs" / "*" / "run_summary.json")):
        overall = json.loads(Path(path).read_text(encoding="utf-8")).get(
            "overall_decision", {}
        )
        decisions[overall.get("decision")] = decisions.get(overall.get("decision"), 0) + 1
        claimable.add(bool(overall.get("claimable")))
    assert decisions.get("PASS") is None
    assert claimable == {False}
    assert f"{decisions['FLAG_HIGH']}건 FLAG_HIGH" in guide_text
    assert f"{decisions['HALT']}건 HALT" in guide_text
    assert f"커밋된 {sum(decisions.values())}개 실행" in guide_text


def test_the_panel_property_range_matches_the_gate() -> None:
    """The guide quotes the bounds the applicability gate actually enforces."""
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from compound_applicability import (
        PANEL_HEAVY_ATOMS,
        PANEL_MOLECULAR_WEIGHT,
        PANEL_ROTATABLE_BONDS,
    )

    text = GUIDE.read_text(encoding="utf-8")
    assert f"{PANEL_HEAVY_ATOMS[0]}–{PANEL_HEAVY_ATOMS[1]}개" in text
    assert f"{PANEL_ROTATABLE_BONDS[0]}–{PANEL_ROTATABLE_BONDS[1]}개" in text
    assert (
        f"{PANEL_MOLECULAR_WEIGHT[0]:.0f}–{PANEL_MOLECULAR_WEIGHT[1]:.0f} Da" in text
    )


def test_every_source_the_appendix_names_exists(guide_text: str) -> None:
    """Only repository paths are checked.

    The guide also names run artifacts by bare filename - those live under a
    run directory, not at the repository root, so a name without a directory
    separator is prose rather than a source reference.
    """
    for match in re.findall(r"`([A-Za-z0-9_./*-]+/[A-Za-z0-9_./*-]+\.(?:md|json|csv))`", guide_text):
        if "*" in match:
            assert glob.glob(str(ROOT / match)), match
        else:
            assert (ROOT / match).exists(), match
    for match in re.findall(r"`([A-Za-z0-9_-]+\.(?:md|json|csv))`", guide_text):
        assert (ROOT / match).exists() or match in RUN_ARTIFACT_NAMES, match


def test_the_readme_orients_a_wet_lab_reader_before_anything_technical() -> None:
    """The collaborator opening this repo is not a computational scientist.

    The first thing after the title used to be a spec reference; it now says
    what the tool does, the three steps, and where the interpretation limits
    are written down.
    """
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    intro = readme.split("## Quick Start", 1)[0]
    assert "실험 연구자라면 여기부터" in intro
    assert "docs/RESEARCHER_GUIDE.md" in intro
    # The two things most likely to be misread, named up front.
    assert "자기 자신을 찾아옵니다" in intro
    assert "도킹 점수를 신뢰할 수 없습니다" in intro


def test_the_quick_start_covers_viewing_and_remote_access() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    quick = readme.split("## Quick Start", 1)[1].split("\n## ", 1)[0]
    assert "viewer/index.html" in quick, "the viewer path a reader opens"
    assert "ssh -L 8080:localhost:8080" in quick, "reaching a lab GPU server"
    # A remote bind is supported now, but only with a login. The Quick Start has
    # to say so where it shows the command, and has to name where the token is.
    assert "--host 0.0.0.0" in quick
    assert "로그인이 자동으로 켜집니다" in quick
    assert "workbench-access.token" in quick
    # HTTP is unencrypted; that limit belongs next to the instruction.
    assert "HTTPS" in quick


def test_the_quick_start_covers_the_whole_wet_lab_path() -> None:
    """Install, input, run, read, view, download, report - each has a section."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    quick = readme.split("## Quick Start", 1)[1].split("\n## ", 1)[0]
    for heading in (
        "자동 설치",
        "무엇을 넣을 수 있나",
        "첫 분석",
        "결과 읽기",
        "3D로 보기",
        "결과 내려받기",
        "GPU 서버에 두고 노트북에서 쓰기",
    ):
        assert heading in quick, heading


def test_the_input_section_says_where_a_smiles_comes_from() -> None:
    """A wet-lab reader has a compound name, not a SMILES string."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    section = readme.split("무엇을 넣을 수 있나", 1)[1].split("\n### ", 1)[0]
    assert "pubchem" in section.lower()
    assert "Canonical SMILES" in section
    # The refusals a reader will actually hit, with the reason.
    for excluded in ("혼합물", "펩타이드", "고분자", "계면활성제", "자외선차단"):
        assert excluded in section, excluded


def test_the_download_section_explains_sharing_a_result() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    section = readme.split("### 8. 결과 내려받기", 1)[1].split("\n### ", 1)[0]
    assert "run_summary.md" in section
    assert "viewer/summary.csv" in section
    assert "zip -r" in section, "handing the viewer to a colleague"
    assert "scp -r" in section, "pulling it off a GPU server"


def test_the_coverage_section_says_absent_not_zero(guide_text: str) -> None:
    """An unscored target is missing from the output, not scored zero.

    The distinction decides how a reader acts: "0점" reads as a low score,
    when the target was never scored at all. Verified against the largest
    committed Daina proteome output, 4,727 rows of 20,204 targets.
    """
    section = guide_text.split("## 5. 커버리지", 1)[1].split("\n## ", 1)[0]

    assert "정확히 0점" not in section
    assert "결과에 아예 나타나지 않는다" in section
    assert "채점 시도조차" in section


def test_no_document_still_calls_unscored_targets_zero() -> None:
    for path in sorted((ROOT / "docs").glob("*.md")) + [ROOT / "README.md"]:
        text = path.read_text(encoding="utf-8")
        assert "정확히 0점" not in text, path.name


def test_the_readme_states_what_it_cannot_answer_before_install() -> None:
    """A first-time collaborator should learn the limits before spending an
    hour installing, not after their first run."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    intro = readme.split("## Quick Start", 1)[0]

    # The heading wording is free to change; what must survive is that the
    # limits are stated as questions a reader would actually ask.
    assert "뭘 물어볼 수 있나요" in intro
    assert "이 성분 안전한가?" in intro and "아니요" in intro
    # The measured similarity bands, which decide whether a row is worth acting on.
    assert "0.6 이상" in intro and "0.4 미만" in intro
    assert "근거 유사도" in intro
    # PASS has never occurred; a reader must not read REVIEW as a problem.
    assert "한 번도 내지 않았습니다" in intro or "한 번도 나온 적 없습니다" in readme


def test_the_readme_keeps_operator_material_out_of_the_researcher_path() -> None:
    """654 lines of Stage 0 and CLI reference sat between a researcher and the
    rest of the document."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    operator = ROOT / "docs" / "OPERATOR_CLI.md"

    assert operator.is_file()
    assert "docs/OPERATOR_CLI.md" in readme
    assert "볼 필요가 없습니다" in readme
    # 900이던 상한을 930으로 올렸다. 표적 우선 조회(explore_target)라는 연구자용
    # 진입점이 짧은 포인터로 들어왔고, 공개 스냅샷은 여기에 "이 저장소는
    # 공개본입니다" 안내가 11줄 더 붙는다. 상세 사용법은
    # docs/EXPLORE_TARGET.md로 보냈으니 이 상한의 목적 - 운영자용 Stage 0/CLI
    # 설명이 README로 돌아오지 않는 것 - 은 그대로다.
    assert len(readme.splitlines()) < 930, "README is drifting back toward a manual"


def test_the_readme_says_how_to_share_a_result() -> None:
    """First delivery to a collaborator: they will be asked to show someone."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    quick = readme.split("## Quick Start", 1)[1].split("\n## ", 1)[0]

    assert "동료에게 보여주기" in quick
    assert "인터넷 없이 열립니다" in quick
    # And a caveat that travels with the result.
    assert "실험 전 가설" in quick


def test_the_readme_troubleshooting_speaks_to_a_wet_lab_reader() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    quick = readme.split("## Quick Start", 1)[1].split("\n## ", 1)[0]
    section = quick.split("### 10.", 1)[1].split("### 11.", 1)[0]

    # The two symptoms most likely to be misread as failures.
    assert "오류가 아닐 가능성이 높습니다" in section
    assert "정상입니다" in section


def test_the_readme_shows_the_screens_in_order_before_the_install() -> None:
    """A collaborator should see what the tool looks like before spending an hour.

    The screenshots are captured from the running application, so they cannot
    drift into showing a UI that no longer exists without the capture script
    also being run.
    """
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    intro = readme.split("## Quick Start", 1)[0]

    assert "어떻게 생겼나 — 실제 화면" in intro
    positions = [intro.index(f"wb-{n}-") for n in range(1, 7)]
    assert positions == sorted(positions), "screenshots must appear in use order"

    for name in (
        "wb-1-readiness.png",
        "wb-2-input.png",
        "wb-3-choices.png",
        "wb-4-runs.png",
        "wb-5-results.png",
        "wb-6-viewer.png",
    ):
        image = ROOT / "docs" / "images" / name
        assert image.is_file(), name
        assert image.stat().st_size > 40_000, name
        assert image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"), name


def test_the_readme_says_what_works_and_what_does_not_up_front() -> None:
    """The collaborator's first question is "how far along is this"."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    intro = readme.split("## Quick Start", 1)[0]

    assert "지금 되는 것 / 아직 안 되는 것" in intro
    assert "앞으로 할 것" in intro
    # The unfinished thing has to be named, not hidden among the finished ones.
    assert "한 번도 완주된 적 없습니다" in intro
    # And the thing that is out of scope, so nobody waits for it.
    assert "계획 없습니다" in intro
    # The roadmap points at the measured evidence rather than asserting.
    assert "DATA_EXPANSION_20260831.md" in intro
    assert "TYRP1" in intro
