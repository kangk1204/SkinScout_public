"""The downloaded verdict file has to say what the screen said.

`run_summary.md` is what the README sends a wet-lab reader to for "판정과 다음에
할 일", and what the Workbench hands over as `판정 보고서 (.md)`. Before this it
was entirely English, and its decision word `FLAG_HIGH` appears in no Korean
surface - the Workbench renders that same value as `REVIEW`. A reader comparing
the screen with the file would have found two different words for one verdict.

These tests pin the two catalogues together so a change on one side cannot
silently leave the other behind.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import messages_ko  # noqa: E402

APP_JS = (ROOT / "workbench" / "static" / "app.js").read_text(encoding="utf-8")


def _js_object(function_name: str) -> dict[str, str]:
    """Pull the `const labels = {...}` / `const exact = {...}` literal out of a function."""
    body = APP_JS.split(f"function {function_name}(", 1)[1]
    literal = body.split("{", 1)[1]
    depth = 1
    for index, char in enumerate(literal):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                literal = literal[:index]
                break
    # Keys appear both quoted ("skin toxicity is HALT") and bare (skin_expression_only).
    pairs = re.findall(r'(?:"([^"]+)"|([A-Za-z_][\w]*))\s*:\s*"([^"]*)"', literal)
    return {quoted or bare: value for quoted, bare, value in pairs}


def test_the_reason_wording_matches_the_screen() -> None:
    js_reasons = _js_object("reasonLabel")
    assert js_reasons, "could not read reasonLabel from app.js"
    assert messages_ko.REASONS == js_reasons


def test_the_skin_context_wording_matches_the_screen() -> None:
    js_context = _js_object("skinContextLabel")
    assert js_context, "could not read skinContextLabel from app.js"
    assert messages_ko.SKIN_CONTEXT == js_context


def test_flag_high_is_called_review_the_way_every_korean_surface_calls_it() -> None:
    """The wire value and the word a reader sees are not the same string."""
    badge, meaning = messages_ko.decision_label("FLAG_HIGH")
    assert badge == "REVIEW"
    assert meaning
    # And app.js has to agree, or the file and the screen disagree again.
    assert '"REVIEW", title: "사람 검토 필요"' in APP_JS.replace("label: ", "")

    assert messages_ko.decision_label("PASS")[0] == "PASS"
    assert messages_ko.decision_label("HALT")[0] == "HALT"


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        (
            "top predicted target is RARA (P10276) among 10 ranked targets",
            "상위 예측 표적은 RARA (P10276)이며, 총 10개 후보가 순위화되었습니다.",
        ),
        ("high ADMET risk endpoints: DILI,Skin_Reaction", "높은 ADMET 위험 항목: DILI,Skin_Reaction."),
        (
            "skin-specialized binding context requires review: skin_efficacy_literature_only",
            "피부 특화 결합 맥락을 검토해야 합니다: 피부 효능 문헌만 지원.",
        ),
    ],
)
def test_the_patterned_reasons_render_in_korean(reason: str, expected: str) -> None:
    assert messages_ko.reason_label(reason) == expected


def test_an_unknown_reason_survives_untranslated() -> None:
    """A new reason must reach the reader, not be swallowed by the lookup."""
    assert messages_ko.reason_label("some brand new reason") == "some brand new reason"


def test_the_summary_opens_with_the_korean_verdict() -> None:
    import summarize_run_outputs

    summary = json.loads(
        (ROOT / "results" / "runs" / "deepdiag_Tretinoin" / "run_summary.json").read_text(
            encoding="utf-8"
        )
    )
    markdown = summarize_run_outputs.render_markdown_summary(summary)
    head = markdown.split("## Run", 1)[0]

    assert "한눈에 보기" in head
    assert "**판정: REVIEW**" in head
    assert "FLAG_HIGH" not in head, "the wire value must not reach the reader"
    assert "실험 전 가설" in head
    # The reasons a reader acts on, in the same words the Workbench uses.
    assert "피부 감작성 종합 판정에 사람 검토가 필요합니다." in head
    assert "docs/RESEARCHER_GUIDE.md" in head
    # And the machine-readable body is untouched below it.
    assert "## Overall Decision" in markdown
    assert "- Decision: FLAG_HIGH" in markdown


def test_the_action_enum_matches_the_summary_generator() -> None:
    """`stop`/`proceed_with_caution` never matched the generated wire values,
    so every HALT/PASS Markdown silently dropped its next-action line."""
    import summarize_run_outputs

    assert messages_ko.ACTIONS_BY_DECISION == {
        "PASS": "proceed",
        "FLAG_HIGH": "review_before_claim",
        "HALT": "stop_before_claim",
    }
    assert set(messages_ko.ACTIONS) == set(messages_ko.ACTIONS_BY_DECISION.values())
    for action in messages_ko.ACTIONS_BY_DECISION.values():
        assert messages_ko.action_label(action)
    # The generator imports the same enum instead of keeping a second copy.
    assert summarize_run_outputs.ACTIONS_BY_DECISION is messages_ko.ACTIONS_BY_DECISION
    # The screen maps the same wire values; its keys have to agree.
    assert set(_js_object("actionLabel")) == set(messages_ko.ACTIONS)


def _minimal_summary(decision: str, action: str) -> dict[str, object]:
    return {
        "run_id": "action_enum_fixture",
        "preset": "report",
        "mode": "fast",
        "compound": {"canonical_smiles": "CCO", "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"},
        "safety": {},
        "skin_toxicity": {},
        "cosmetic_drug": {},
        "artifacts": {},
        "overall_decision": {
            "decision": decision,
            "recommended_action": action,
            "claimable": decision == "PASS",
            "requires_human_review": decision != "PASS",
            "reasons": [],
        },
    }


def test_halt_and_pass_markdown_render_the_action_line() -> None:
    import summarize_run_outputs

    halt = summarize_run_outputs.render_markdown_summary(
        _minimal_summary("HALT", "stop_before_claim")
    )
    assert f"**다음에 할 일:** {messages_ko.ACTIONS['stop_before_claim']}" in halt

    passed = summarize_run_outputs.render_markdown_summary(
        _minimal_summary("PASS", "proceed")
    )
    assert f"**다음에 할 일:** {messages_ko.ACTIONS['proceed']}" in passed
