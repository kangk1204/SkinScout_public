"""Static contract tests for the "활성 핵심구조 유지 대체소재" view.

This screen is the one that answers the funded project's title directly: give it
a compound whose activity is known and it picks already-listed cosmetic
ingredients that keep the active core. It is also the screen with the shortest
path from "a structure looks similar" to "so it works", so the assertions here
are of two kinds:

* wiring - every id, class and column count the code reaches for actually
  exists, because the failure mode is a silent empty table rather than an error
* honesty - the copy that separates structural retention from measured activity,
  and "no measurement" from "inactive", has to survive a copy rewrite

The honesty checks match on meaning through a set of alternative patterns. An
earlier test in this suite pinned one exact sentence and had to be edited every
time the wording changed, which trains a reader to edit the test rather than
read it.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "workbench" / "static"

# The ids the alternatives code binds to. Kept here as well as derived from
# `app.js` so the test fails in both directions: a renamed id in the markup, and
# an id that quietly stopped being used by the code.
ALTERNATIVES_IDS = (
    "alternatives-smiles",
    "alternatives-run",
    "alternatives-csv",
    "alternatives-status",
    "alternatives-badge",
    "alternatives-limit",
    "alternatives-require-core",
    "alternatives-exclude-self",
    "alternatives-query",
    "alternatives-ingredients-surface",
    "alternatives-ingredients-body",
    "alternatives-ingredients-count",
    "alternatives-library-note",
    "alternatives-measured-surface",
    "alternatives-measured-body",
    "alternatives-measured-count",
    "similar-jump",
    "similar-jump-button",
)

VOID_ELEMENTS = frozenset(
    {
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr",
    }
)


def _read(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


def _element_ids(markup: str) -> set[str]:
    return set(re.findall(r'\bid="([A-Za-z0-9_:-]+)"', markup))


def _referenced_ids(javascript: str) -> set[str]:
    """Ids the code looks up, from either accessor it uses."""
    found = set(re.findall(r'\$\("#([A-Za-z0-9_:-]+)"\)', javascript))
    found |= set(re.findall(r'getElementById\("([A-Za-z0-9_:-]+)"\)', javascript))
    return found


def _alternatives_panel(html: str) -> str:
    """The `<section data-view-panel="alternatives">` element, tags balanced.

    Slicing to "the next panel" would break the day this view moves last, so the
    nesting is counted instead.
    """
    anchor = html.index('data-view-panel="alternatives"')
    start = html.rindex("<section", 0, anchor)
    depth = 0
    for match in re.finditer(r"</?section\b", html[start:]):
        depth += 1 if match.group(0) == "<section" else -1
        if depth == 0:
            return html[start : start + match.end() + len("</section>") - len(match.group(0))]
    raise AssertionError("the alternatives panel is never closed")


def _thead_columns(panel: str, body_id: str) -> int:
    """Column count of the table whose <tbody> carries `body_id`."""
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
    """Markup with tags dropped, so a phrase split across <strong> still reads."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", markup))


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


class _BalanceParser(HTMLParser):
    """Fail on a stray or crossed close tag, and on anything left open."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.open_tags: list[tuple[str, tuple[int, int]]] = []
        self.errors: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001 - stdlib signature
        if tag not in VOID_ELEMENTS:
            self.open_tags.append((tag, self.getpos()))

    def handle_startendtag(self, tag: str, attrs) -> None:  # noqa: ANN001 - stdlib signature
        return

    def handle_endtag(self, tag: str) -> None:
        if tag in VOID_ELEMENTS:
            return
        if not self.open_tags:
            self.errors.append(f"line {self.getpos()[0]}: </{tag}> with nothing open")
            return
        expected, opened_at = self.open_tags[-1]
        if expected != tag:
            self.errors.append(
                f"line {self.getpos()[0]}: </{tag}> closes <{expected}> opened on line {opened_at[0]}"
            )
            return
        self.open_tags.pop()


def test_the_alternatives_view_is_reachable_from_the_nav_and_the_router() -> None:
    html = _read("index.html")
    javascript = _read("app.js")

    assert 'data-view="alternatives"' in html, "no nav entry opens the view"
    assert re.search(
        r'<button[^>]*class="[^"]*\bnav-item\b[^"]*"[^>]*data-view="alternatives"', html
    ), "the alternatives entry is not a primary nav button"
    assert 'data-view-panel="alternatives"' in html

    valid = javascript.split("VALID_VIEWS", 1)[1].split("]", 1)[0]
    assert '"alternatives"' in valid, "the router drops the view back to setup"
    # The analyze screen hands a SMILES over to this view; that jump is the only
    # reason a first-time reader finds the screen at all.
    assert 'navigate("alternatives")' in javascript


def test_every_id_the_alternatives_code_reaches_for_exists_in_the_markup() -> None:
    html = _read("index.html")
    javascript = _read("app.js")

    markup_ids = _element_ids(html)
    referenced = _referenced_ids(javascript)
    view_ids = {
        name
        for name in referenced
        if name.startswith("alternatives") or name.startswith("similar-jump")
    }
    assert view_ids, "the alternatives code looks up no ids at all"

    missing = sorted(name for name in view_ids if name not in markup_ids)
    assert not missing, f"app.js reads ids that index.html never defines: {missing}"

    # And the reverse direction, so a rename cannot pass by making both sides
    # agree on a name nothing renders.
    for name in ALTERNATIVES_IDS:
        assert name in markup_ids, f"index.html lost #{name}"
        assert name in referenced, f"app.js no longer binds #{name}"


def test_the_replaced_similar_compounds_panel_left_nothing_behind() -> None:
    """Half a removed panel is worse than none: the ids resolve, the loader does
    not, and the table sits empty with no message."""
    stale = ("similar-table-body", "similar-run", "similar-wrap", "loadSimilarCompounds")
    for asset in sorted(STATIC.glob("*")):
        if not asset.is_file():
            continue
        text = asset.read_text(encoding="utf-8", errors="ignore")
        for token in stale:
            assert token not in text, f"{asset.name} still carries {token}"


def test_empty_state_colspans_match_the_real_column_counts() -> None:
    """The empty state is the first thing a new reader sees.

    A colspan that disagrees with the header collapses that row into one narrow
    cell, so the "no candidates, try turning the filter off" message - and the
    "library unavailable" message, which must never look like "no candidates" -
    render as a stray fragment instead of a sentence.
    """
    panel = _alternatives_panel(_read("index.html"))
    javascript = _read("app.js")

    columns = {
        "alternatives-ingredients-body": _thead_columns(panel, "alternatives-ingredients-body"),
        "alternatives-measured-body": _thead_columns(panel, "alternatives-measured-body"),
    }
    # 등재 원료 표에는 조건부로 나타나는 열이 넷 있다. 파마코포어와 3D 는
    # 사용자가 켤 때, 종합 점수와 ADMET 은 서버가 그 값을 실제로 붙였을 때만
    # 보인다(각각 score_available · admet_available). 없는데 열이 보이면 빈 칸이
    # "위험 없음"으로 읽히므로, 마크업에 표시를 달아 두고 app.js 가 토글한다.
    optional = (panel.count("data-pharm-col") + panel.count("data-3d-col")
                + panel.count("data-score-col") + panel.count("data-admet-col"))
    assert optional == 6, f"선택적 열 표시가 {optional}개입니다"
    assert columns["alternatives-ingredients-body"] == 14, columns  # 8 고정 + 6 선택
    assert columns["alternatives-measured-body"] == 7, columns

    # 가변 열을 가진 표는 상수를 쓰면 안 되고, 고정 열만 가진 표는 상수라도 된다.
    ingredient_body = _function_body(javascript, "ingredientRowsHtml")
    assert "ingredientColumnCount()" in ingredient_body, (
        "ingredientRowsHtml이 colspan을 계산하지 않습니다"
    )
    assert not re.findall(r'colspan="\d+"', ingredient_body), (
        "등재 원료 표가 colspan을 상수로 박고 있습니다"
    )
    counter = _function_body(javascript, "ingredientColumnCount")
    assert "8" in counter, f"고정 열 수가 계산에 없습니다: {counter}"
    assert "pharmacophore" in counter and "threeD" in counter, (
        f"두 선택 열을 모두 세지 않습니다: {counter}"
    )

    measured_spans = [
        int(value) for value in re.findall(r'colspan="(\d+)"', _function_body(javascript, "measuredRowsHtml"))
    ]
    assert measured_spans, "measuredRowsHtml renders no full-width row"
    assert set(measured_spans) == {columns["alternatives-measured-body"]}, measured_spans

    # 실패 메시지는 헬퍼 밖, 로더에서 직접 본문에 쓰인다. 이쪽도 같은 규칙을 따라야
    # 한다 - "라이브러리 없음"이 좁은 칸으로 찌그러지면 "후보 없음"처럼 보인다.
    loader = _function_body(javascript, "loadAlternatives")
    for hit in re.finditer(re.escape("#alternatives-ingredients-body"), loader):
        window = loader[hit.end() : hit.end() + 400]
        assert not re.search(r'colspan="\d+"', window), (
            "등재 원료 표에 쓰이는 실패 메시지가 colspan을 상수로 박고 있습니다"
        )
    for hit in re.finditer(re.escape("#alternatives-measured-body"), loader):
        window = loader[hit.end() : hit.end() + 400]
        span = re.search(r'colspan="(\d+)"', window)
        if span is not None:
            assert int(span.group(1)) == columns["alternatives-measured-body"], span.group(1)


def test_index_html_parses_with_balanced_tags() -> None:
    parser = _BalanceParser()
    parser.feed(_read("index.html"))
    assert not parser.errors, parser.errors
    assert not parser.open_tags, [
        f"<{tag}> opened on line {position[0]} is never closed" for tag, position in parser.open_tags
    ]


def test_every_class_the_alternatives_markup_uses_is_styled_or_a_javascript_hook() -> None:
    html = _read("index.html")
    css = _read("styles.css")
    javascript = _read("app.js")

    for name in (
        "link-button",
        "similar-jump",
        "alternatives-examples",
        "alternatives-options",
        "check-inline",
        "alternatives-query",
    ):
        assert re.search(rf"\.{re.escape(name)}\b", css), f"styles.css defines no .{name}"

    panel = _alternatives_panel(html)
    used: set[str] = set()
    for value in re.findall(r'class="([^"]+)"', panel):
        used.update(value.split())
    styled = set(re.findall(r"\.([A-Za-z0-9_-]+)", css))
    # A class can legitimately carry no styling if it exists only as a selector
    # the code binds to - `alternatives-example` is one. Anything else is a typo.
    orphans = sorted(
        name
        for name in used - styled
        if f".{name}" not in javascript
    )
    assert not orphans, f"classes in the alternatives markup with no style and no handler: {orphans}"


def test_the_view_states_that_structural_retention_is_not_measured_activity() -> None:
    """House rule: the screen must never overclaim.

    Every column on it is an atom-count comparison. Matched on meaning through
    alternatives, because this paragraph gets rewritten often and an exact-string
    assertion would only teach the next author to delete the test.
    """
    text = _visible_text(_alternatives_panel(_read("index.html")))

    retention_is_not_activity = (
        r"활성이\s*같지는\s*않",
        r"활성을?\s*(잰|측정한)\s*값이\s*아",
        r"구조\s*비교이며",
        r"활성이\s*같다는\s*뜻이?\s*아",
        r"활성을?\s*보장하지\s*않",
    )
    assert _matches_any(text, retention_is_not_activity), (
        "the alternatives view no longer says structural retention is not measured activity"
    )

    # And it has to name what the number actually is, not only what it is not.
    # 유지율 is a heavy-atom ratio, which is why a nine-atom query reaches 70%
    # off one shared ring - the reader needs that stated, not just the caveat.
    assert _matches_any(text, (r"중원자", r"원자\s*수준의\s*구조\s*비교")), (
        "the view no longer explains that the retention figure is an atom count"
    )


def test_an_absent_measurement_is_never_presented_as_inactivity() -> None:
    """House rule: "no measurement exists" must never render as "inactive"."""
    html = _read("index.html")
    javascript = _read("app.js")
    panel = _alternatives_panel(html)
    text = _visible_text(panel)

    absence_is_not_inactivity = (
        # "…없다는 뜻도 … 아닙니다" - the disclaimer, however its clauses are ordered.
        r"없다는\s*뜻(도|이)?[^.]{0,80}아[닙니]",
        # "없는 것과 없다고 밝혀진 것은 다릅니다" - absence of a record vs. a measured negative.
        r"없\S*\s*것과\s*없다고\s*밝혀진\s*것은\s*다",
        r"비활성[^.]{0,20}(뜻하지|의미하지)\s*않",
    )
    assert _matches_any(text, absence_is_not_inactivity), (
        "the alternatives view no longer separates 'not measured' from 'inactive'"
    )

    # The copy above is undone if the same screen asserts inactivity anywhere
    # else. These patterns are the assertion, not the disclaimer: the honest
    # sentence reads "활성이 없다는 뜻도 아닙니다" and must keep passing.
    for pattern in (r"비활성", r"불활성", r"활성\s*없음", r"활성이\s*없(습니다|다\.|다는\s*것)"):
        found = re.search(pattern, text)
        assert found is None, f"the alternatives panel asserts inactivity: {found.group(0)!r}"

    evidence_map = javascript.split("const MEASURED_EVIDENCE", 1)[1].split("};", 1)[0]
    for state in ("none", "not_measured", "evidence_unavailable"):
        assert state in evidence_map, f"MEASURED_EVIDENCE lost the {state} state"
    for word in ("비활성", "불활성", "활성 없음"):
        assert word not in evidence_map, f"an evidence label reads as {word}"

    # Fail closed: a library that could not be opened is not a molecule with no
    # record. Sharing one label would erase the difference on screen.
    labels = dict(re.findall(r'(\w+):\s*\["[^"]*",\s*"([^"]*)"\]', evidence_map))
    assert labels.get("evidence_unavailable"), "evidence_unavailable has no label of its own"
    assert labels.get("not_measured"), "not_measured has no label of its own"
    assert labels["evidence_unavailable"] != labels["not_measured"], labels


def test_the_pharmacophore_column_is_optional_and_the_colspans_follow_it() -> None:
    """파마코포어를 켜면 표가 한 칸 넓어진다.

    colspan을 상수로 박아 두면 빈 상태와 구분선이 조용히 어긋나는데, 첫 사용자가
    보는 것이 바로 그 빈 상태다.
    """
    html = _read("index.html")
    javascript = _read("app.js")

    assert 'id="alternatives-pharmacophore"' in html
    assert "data-pharm-col" in html, "파마코포어 열에 토글용 표시가 없습니다"
    assert "with_pharmacophore" in javascript, "요청에 옵션이 실리지 않습니다"

    # 빈 상태와 구분선은 계산된 열 수를 써야 한다. 검사 범위는 대체소재 함수로
    # 좁힌다 - app.js 전체를 훑으면 열 수가 고정된 결과 화면의 표까지 걸린다.
    assert "ingredientColumnCount()" in javascript
    body = _function_body(javascript, "ingredientRowsHtml")
    assert not re.findall(r'colspan="\d+"', body), (
        "등재 원료 표가 colspan을 상수로 박고 있습니다"
    )


def test_the_view_says_the_two_signals_are_not_one_score() -> None:
    """두 열을 나란히 놓기만 하면 서로를 뒷받침하는 것처럼 읽힌다."""
    javascript = _read("app.js")
    html = _read("index.html")
    assert 'id="alternatives-disagreement"' in html
    assert "renderSignalAgreement" in javascript

    # 어긋남을 말하는 문구가 있어야 한다. 문장이 다시 쓰일 수 있으므로 뜻으로 본다.
    combined = javascript + html
    meaning = (
        r"하나의?\s*점수로\s*합치지\s*않",
        r"같은\s*일이\s*아니",
        r"순위\s*상관",
    )
    assert _matches_any(combined, meaning), (
        "두 신호가 하나가 아니라는 사실이 화면에서 사라졌습니다"
    )


def test_the_merged_ordering_is_offered_and_explained() -> None:
    """합친 순위는 측정이 지지하는 선택지다. 다만 어느 정답표에서 지지되는지,
    그리고 어디서 그 정답표가 못 미더운지를 함께 말해야 한다."""
    html = _read("index.html")
    javascript = _read("app.js")

    assert 'id="alternatives-sort"' in html
    assert 'value="merged"' in html
    assert 'id="alternatives-sort-note"' in html
    assert "sort_by" in javascript, "정렬 선택이 요청에 실리지 않습니다"
    assert "renderSortNote" in javascript

    note = _function_body(javascript, "renderSortNote")
    # 근거를 대되, 그 근거의 한계도 같은 자리에서 말해야 한다.
    assert _matches_any(note, (r"0\.697", r"AUC")), "합친 순위가 나은 근거가 없습니다"
    assert _matches_any(note, (r"향료", r"기준선")), "그 근거가 어디서 무너지는지가 없습니다"
    assert _matches_any(note, (r"용도가\s*같다", r"대체\s*가능하다가\s*아니")), (
        "정답표가 무엇이었는지 말하지 않습니다"
    )


def test_the_divider_is_not_drawn_when_the_list_is_not_grade_ordered() -> None:
    """구분선은 등급 순일 때만 뜻이 있다.

    합친 순위에서는 유지 등급이 섞여 들어오므로, 한 곳에 줄을 그으면 없는 경계를
    있다고 말하는 것이 된다.
    """
    javascript = _read("app.js")
    body = _function_body(javascript, "ingredientRowsHtml")
    assert "sortBy" in body, "정렬 방식을 보지 않고 구분선을 그립니다"
    # 등급 순이 아닌 정렬이 둘(합친 순위, 극성 순)이므로 하나를 이름으로 집지 않는다.
    assert _matches_any(body, (r'sortBy\s*!==\s*"core"',)), (
        "등급 순일 때만 구분선을 그리도록 되어 있지 않습니다"
    )


def test_the_three_d_column_is_optional_and_the_column_count_follows_both_toggles() -> None:
    """열이 둘 다 켜지면 표는 10칸이다.

    colspan을 파마코포어 하나만 보고 계산하면 3D를 켰을 때 빈 상태와 구분선이
    조용히 어긋난다.
    """
    html = _read("index.html")
    javascript = _read("app.js")

    assert 'id="alternatives-three-d"' in html
    assert "data-3d-col" in html
    assert "with_three_d" in javascript

    counter = _function_body(javascript, "ingredientColumnCount")
    assert "pharmacophore" in counter and "threeD" in counter, (
        "열 수 계산이 두 토글을 모두 보지 않습니다"
    )


def test_a_row_that_was_not_rescored_does_not_read_as_a_bad_3d_score() -> None:
    """3D는 상위 몇 건만 본다. 보지 않은 것과 나빴던 것은 다른 말이다."""
    javascript = _read("app.js")
    labels = javascript.split("const THREE_D_LABEL", 1)[1].split("};", 1)[0]
    for state in ("not_rescored", "unavailable", "budget_exhausted"):
        assert state in labels, f"{state} 상태를 구분하지 않습니다"
    assert _matches_any(labels, (r"재채점", r"보지 않", r"판정 불가", r"시간 상한")), (
        "재채점하지 않은 행이 무엇인지 말하지 않습니다"
    )
    cell = _function_body(javascript, "threeDCell")
    assert "THREE_D_LABEL" in cell, "셀이 상태 표를 쓰지 않습니다"


def test_the_polarity_ordering_declares_that_it_ignores_structure() -> None:
    """구조를 보지 않는 순서를 구조 기준 옆에 두면, 말하지 않는 한 구분되지 않는다.

    측정에서 이 순서가 이긴 곳은 향료 분류 하나뿐이었고, 출처가 인용된 정답표에서는
    상위 4위 안에도 들지 못했다. 그 범위가 화면에 있어야 한다.
    """
    html = _read("index.html")
    javascript = _read("app.js")

    assert 'value="polarity"' in html
    assert _matches_any(html, (r"구조\s*안\s*봄", r"구조를\s*보지\s*않")), (
        "선택지 이름이 구조를 보지 않는다는 사실을 말하지 않습니다"
    )
    note = _function_body(javascript, "renderSortNote")
    assert "polarity" in note
    assert _matches_any(note, (r"향료", r"방향\s*분류")), "어디서 이겼는지가 없습니다"
    assert _matches_any(note, (r"4위\s*안에도", r"들지\s*못", r"이긴 곳은")), (
        "어디서 못 이겼는지가 없습니다"
    )


def test_the_divider_is_only_drawn_for_the_grade_ordering() -> None:
    """구분선은 등급 순일 때만 뜻이 있다. 합친 순위와 극성 순 모두 등급 순이 아니다."""
    javascript = _read("app.js")
    body = _function_body(javascript, "ingredientRowsHtml")
    assert _matches_any(body, (r'sortBy\s*!==\s*"core"',)), (
        "등급 순이 아닐 때 구분선을 억제하지 않습니다"
    )
