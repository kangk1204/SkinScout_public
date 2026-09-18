"""Static contract tests for the Workbench "표적 검색" view.

The screen answers the question the CLI `explore_target` answers, from the
browser: give it a protein and it lists compounds that have a measured record
against it. The failure mode is the same as the alternatives view - a silent
empty table because an id, a column or a router entry drifted - so the wiring is
pinned here, together with the one honesty rule: "measured" must not read as
"potent".
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "workbench" / "static"

# Ids the target-search code binds to. Kept here as well as derived from app.js
# so the test fails in both directions.
TARGET_IDS = (
    "targets-badge",
    "targets-readiness",
    "targets-query",
    "targets-mode",
    "targets-limit",
    "targets-run",
    "targets-status",
    "targets-candidates",
    "targets-surface",
    "targets-count",
    "targets-target-note",
    "targets-body",
    "targets-bulk-file",
    "targets-bulk",
    "targets-bulk-top",
    "targets-bulk-run",
    "targets-bulk-status",
    "targets-bulk-surface",
    "targets-bulk-count",
    "targets-bulk-summary",
    "targets-bulk-coverage-body",
    "targets-bulk-candidates-body",
)


def _read(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


def _element_ids(markup: str) -> set[str]:
    return set(re.findall(r'\bid="([A-Za-z0-9_:-]+)"', markup))


def _referenced_ids(javascript: str) -> set[str]:
    found = set(re.findall(r'\$\("#([A-Za-z0-9_:-]+)"\)', javascript))
    found |= set(re.findall(r'getElementById\("([A-Za-z0-9_:-]+)"\)', javascript))
    return found


def _panel(html: str, view: str) -> str:
    anchor = html.index(f'data-view-panel="{view}"')
    start = html.rindex("<section", 0, anchor)
    depth = 0
    for match in re.finditer(r"</?section\b", html[start:]):
        depth += 1 if match.group(0) == "<section" else -1
        if depth == 0:
            return html[start : start + match.end()]
    raise AssertionError(f"the {view} panel is never closed")


def _thead_columns(panel: str, body_id: str) -> int:
    before = panel[: panel.index(f'id="{body_id}"')]
    head_open = before.rindex("<thead")
    head = before[head_open : before.index("</thead>", head_open)]
    return len(re.findall(r"<th\b", head))


def _function_body(javascript: str, name: str) -> str:
    start = javascript.index(f"function {name}(")
    tail = javascript[start:]
    depth = 0
    for index, char in enumerate(tail):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return tail[: index + 1]
    raise AssertionError(f"{name} is never closed")


def _visible_text(markup: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", markup))


def test_the_target_view_is_reachable_from_the_nav_and_the_router() -> None:
    html = _read("index.html")
    javascript = _read("app.js")

    assert 'data-view="targets"' in html, "no nav entry opens the view"
    assert re.search(
        r'<button[^>]*class="[^"]*\bnav-item\b[^"]*"[^>]*data-view="targets"', html
    ), "the target entry is not a primary nav button"
    assert 'data-view-panel="targets"' in html

    valid = javascript.split("VALID_VIEWS", 1)[1].split("]", 1)[0]
    assert '"targets"' in valid, "the router drops the view back to setup"
    # 결과 표의 '이 후보 분석'이 분석 화면으로 넘긴다.
    assert 'navigate("analyze")' in javascript


def test_every_id_the_target_code_reaches_for_exists_in_the_markup() -> None:
    html = _read("index.html")
    javascript = _read("app.js")

    markup_ids = _element_ids(html)
    referenced = _referenced_ids(javascript)
    view_ids = {name for name in referenced if name.startswith("targets")}
    assert view_ids, "the target code looks up no ids at all"

    missing = sorted(name for name in view_ids if name not in markup_ids)
    assert not missing, f"app.js reads ids that index.html never defines: {missing}"

    for name in TARGET_IDS:
        assert name in markup_ids, f"index.html lost #{name}"
        assert name in referenced, f"app.js no longer binds #{name}"


def test_empty_state_colspan_matches_the_real_column_count() -> None:
    html = _read("index.html")
    javascript = _read("app.js")

    panel = _panel(html, "targets")
    columns = _thead_columns(panel, "targets-body")
    body = _function_body(javascript, "targetRowsHtml")
    assert f'colspan="{columns}"' in body, (
        "the empty-state colspan disagrees with the header, so the message "
        "renders as a stray fragment"
    )

    # 리스트 발굴 표도 같은 규칙으로 확인한다.
    coverage_columns = _thead_columns(panel, "targets-bulk-coverage-body")
    coverage_body = _function_body(javascript, "targetBulkCoverageHtml")
    assert f'colspan="{coverage_columns}"' in coverage_body

    candidate_columns = _thead_columns(panel, "targets-bulk-candidates-body")
    candidate_body = _function_body(javascript, "targetBulkCandidateHtml")
    assert f'colspan="{candidate_columns}"' in candidate_body


def test_the_screen_says_measured_is_not_potent() -> None:
    """문턱 아래의 측정을 '알려진 결합'으로 읽으면 안 된다."""
    panel = _panel(_read("index.html"), "targets")
    text = _visible_text(panel)

    assert re.search(r'"측정됨"은\s*"세다"와\s*다[르릅]', text), "measured/potent distinction is gone"
    assert "양성 문턱" in text
    # 문턱 라벨이 세 갈래로 나뉜다: 양성 / 경계 / 문턱 아래.
    javascript = _read("app.js")
    threshold = javascript.split("const TARGET_THRESHOLD", 1)[1].split("};", 1)[0]
    for label in ("양성", "경계", "문턱 아래"):
        assert label in threshold, f"threshold label {label} is missing"


def test_missing_index_is_announced_before_the_button_is_pressed() -> None:
    javascript = _read("app.js")
    body = _function_body(javascript, "renderTargetsReadiness")
    assert "검색 인덱스가 없습니다" in body, "a data-less install must say so up front"
    assert "run.disabled = true" in body, "the button must not invite a doomed request"


def test_the_nav_numbering_no_longer_collides() -> None:
    """04 표적 검색이 들어오면서 뒤 번호를 하나씩 밀었다."""
    html = _read("index.html")
    nav = html.split("<nav", 1)[1].split("</nav>", 1)[0]
    numbers = re.findall(r'class="nav-icon"[^>]*>(\d{2})<', nav)
    assert len(numbers) == len(set(numbers)), f"duplicate nav numbers: {numbers}"
    assert numbers == [f"{index:02d}" for index in range(1, len(numbers) + 1)]
