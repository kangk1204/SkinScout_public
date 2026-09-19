"""Make the "green with 55 tests missing" case visible, and fail it on demand.

Most data gates in this suite point at gitignored artifacts. On this working
tree they are present and nothing skips; on a fresh clone or in CI whole modules
vanish - `test_researcher_guide_numbers.py` (20 tests), `test_index_coverage_gap.py`
(9), `test_stage3_recipe_scoring.py` (6), `test_pocket_cold_panel.py` (9) and
more - and pytest still exits 0. Every test that checks the promoted recipe
reaches a run is in that set.

Two things happen here:

* a run that skipped anything data-gated says so at the end, with the reasons,
  instead of a bare "3 skipped"
* `SKINSCOUT_REQUIRE_DATA=1` turns those skips into a failing run, which is what
  CI should use once it provisions the artifacts

Environment gates (no torch, no snakemake, no meeko) are reported separately and
never fail the run: they say something about the machine, not about the repo.
"""

from __future__ import annotations

import os

import pytest

# MCS 예산은 자원 한도이지 판정 규칙이 아니다. 운영 기본값은 600초인데(라이브러리
# 7,484종에서 우르솔산 같은 축합 다환 질의가 전수 148초까지 걸리고, 잘리면 라이브
# 러리의 83%를 안 본 채 답한다), 그 값을 시험에도 쓰면 실제 라이브러리를 읽는
# 시험 하나가 40분을 넘긴다 - `test_parallel_evaluation_matches_serial_bit_for_bit`
# 는 12질의를 직렬·병렬로 두 번 전수 스캔한다.
#
# 시험이 확인하는 것은 로직(직렬과 병렬이 같은가, 순위가 뜻대로 나오는가)이지
# 화학 처리량이 아니므로 여기서 한도를 묶는다. 예산 동작 자체를 보는 시험은
# `McsBudget(0.0)`처럼 값을 직접 넘기므로 이 설정에 가리지 않는다.
# 밖에서 지정했으면 그것을 존중한다.
os.environ.setdefault("SKINSCOUT_MCS_BUDGET_SECONDS", "15")

# Reasons that mean "a repository artifact is missing", written the way the
# existing gates phrase it. A new data gate should use `requires_artifact`
# below rather than adding a phrase here.
_ARTIFACT_MARKERS = (
    "not built",
    "has not been built",
    "is not built",
    "not provisioned",
    "absent",
    "not present",
    "not been run",
    "are not present",
    "is not built here",
    "no promoted recipe artifact",
    "no merged evidence table",
)
_ENVIRONMENT_MARKERS = ("could not import", "on PATH", "is required", "shares a filesystem")

_DATA_SKIPS: list[tuple[str, str]] = []
_ENV_SKIPS: list[tuple[str, str]] = []


def requires_artifact(path, what: str) -> None:
    """Skip because a repository artifact is missing, in a form this file counts.

    Prefer this to a bare `pytest.skip` so the run's final summary knows the
    difference between "this machine lacks torch" and "this checkout lacks the
    retrieval index".
    """
    if not os.path.exists(path):
        pytest.skip(f"artifact is not built here: {what} ({path})")


def _reason(report) -> str:
    longrepr = getattr(report, "longrepr", None)
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        return str(longrepr[2]).removeprefix("Skipped: ")
    return str(longrepr or "")


def pytest_runtest_logreport(report) -> None:
    if not report.skipped or report.when != "setup":
        return
    reason = _reason(report)
    lowered = reason.lower()
    if any(marker in lowered for marker in _ENVIRONMENT_MARKERS):
        _ENV_SKIPS.append((report.nodeid, reason))
    elif any(marker in lowered for marker in _ARTIFACT_MARKERS):
        _DATA_SKIPS.append((report.nodeid, reason))


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    if not _DATA_SKIPS and not _ENV_SKIPS:
        return
    write = terminalreporter.write_line
    if _ENV_SKIPS:
        write("")
        write(f"{len(_ENV_SKIPS)} test(s) skipped for this machine, not this checkout:")
        for nodeid, reason in _ENV_SKIPS[:10]:
            write(f"  {nodeid}  -  {reason}")
    if _DATA_SKIPS:
        write("")
        write(
            f"{len(_DATA_SKIPS)} test(s) did not run because repository artifacts "
            "are missing. A green run does not cover them:",
            red=True,
        )
        seen: dict[str, int] = {}
        for _, reason in _DATA_SKIPS:
            seen[reason] = seen.get(reason, 0) + 1
        for reason, count in sorted(seen.items(), key=lambda item: -item[1]):
            write(f"  {count:>3d}  {reason}")
        if not os.environ.get("SKINSCOUT_REQUIRE_DATA"):
            write("  (set SKINSCOUT_REQUIRE_DATA=1 to make this a failure)")


def pytest_sessionfinish(session, exitstatus) -> None:
    if _DATA_SKIPS and os.environ.get("SKINSCOUT_REQUIRE_DATA"):
        session.exitstatus = 1
