"""Korean labels for the run summary.

The Workbench renders the verdict in Korean; `run_summary.md` did not. A wet-lab
reader who follows the README to "판정 보고서 (.md)" opened a file whose decision
word (`FLAG_HIGH`) does not appear anywhere in the Korean documentation, and whose
reasons are English sentences.

The strings here mirror `workbench/static/app.js` (`reasonLabel`,
`decisionPresentation`). `scripts/tests/test_messages_ko.py` fails if the two
drift apart, so the file a reader downloads says what the screen said.
"""

from __future__ import annotations

import re

# Overall decision -> (badge shown to a reader, one line on what it means).
# `FLAG_HIGH` is the wire value; every Korean surface calls it REVIEW.
DECISIONS: dict[str, tuple[str, str]] = {
    "PASS": ("PASS", "필요한 계산과 파일 검증을 전부 통과했습니다."),
    "FLAG_HIGH": ("REVIEW", "사람이 근거를 확인해야 합니다."),
    "HALT": ("HALT", "이 결과를 근거로 주장하면 안 됩니다."),
}

# Overall decision -> canonical wire action produced by
# `summarize_run_outputs._overall_decision`. ACTIONS's keys must be exactly
# these values: the old `stop`/`proceed_with_caution` spelling rendered an
# empty action line for every HALT and PASS report.
ACTIONS_BY_DECISION: dict[str, str] = {
    "PASS": "proceed",
    "FLAG_HIGH": "review_before_claim",
    "HALT": "stop_before_claim",
}

ACTIONS: dict[str, str] = {
    "review_before_claim": "아래 사유를 먼저 확인한 뒤에 결론을 내리세요.",
    "proceed": "다음 단계로 진행해도 됩니다. 근거를 함께 기록하세요.",
    "stop_before_claim": "이 결과를 근거로 주장하지 말고, 원인을 먼저 해결하세요.",
}

REASONS: dict[str, str] = {
    "skin sensitization consensus is HALT": "피부 감작성 종합 판정이 HALT입니다.",
    "skin sensitization consensus is FLAG_HIGH": "피부 감작성 종합 판정에 사람 검토가 필요합니다.",
    "skin toxicity is HALT": "피부 독성 판정이 HALT입니다.",
    "skin toxicity requires review": "피부 독성 결과를 사람이 검토해야 합니다.",
    "safety evidence is degraded": "일부 안전성 근거가 제한된 상태입니다.",
    "target prediction summary is missing": "표적 예측 요약이 없습니다.",
    "target prediction produced no top target": "상위 표적 후보가 생성되지 않았습니다.",
    "skin-specialized binding summary is missing": "피부 특화 결합 요약이 없습니다.",
    "skin efficacy evidence is automatic literature co-occurrence and requires review": (
        "피부 효능 근거가 자동 문헌 공출현이라 사람이 검토해야 합니다."
    ),
    "safety and cosmetic/drug gates passed": "안전성과 화장품·의약품 관련 필수 검사를 통과했습니다.",
}

SKIN_CONTEXT: dict[str, str] = {
    "skin_context_supported": "피부 맥락 근거 지원",
    "skin_expression_only": "피부 발현 근거만 지원",
    "skin_efficacy_literature_only": "피부 효능 문헌만 지원",
    "insufficient_skin_context": "피부 맥락 근거 부족",
}


def skin_context_label(decision: object) -> str:
    text = str(decision or "").strip()
    return SKIN_CONTEXT.get(text, text or "확인 필요")


_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"^top predicted target is (.+) among ([0-9,]+) ranked targets$"),
        "상위 예측 표적은 {0}이며, 총 {1}개 후보가 순위화되었습니다.",
    ),
    (
        re.compile(r"^skin-specialized binding context requires review: (.+)$"),
        "피부 특화 결합 맥락을 검토해야 합니다: {0}.",
    ),
    (
        re.compile(r"^top binding target lacks direct skin context support: (.+)$"),
        "상위 결합 표적 {0}에 직접적인 피부 맥락 근거가 부족합니다.",
    ),
    (re.compile(r"^high ADMET risk endpoints: (.+)$"), "높은 ADMET 위험 항목: {0}."),
    (re.compile(r"^structural alerts present: (.+)$"), "구조 경고가 발견되었습니다: {0}."),
    (re.compile(r"^missing skin-sens models: (.+)$"), "누락된 피부 감작성 모델: {0}."),
    (
        re.compile(r"^drug-avoidance warnings present: (.+)$"),
        "의약품 유사성 경고 {0}건이 있습니다.",
    ),
)


def decision_label(decision: object) -> tuple[str, str]:
    """Return (badge, meaning) for an overall decision, unknown values passed through."""
    text = str(decision or "").strip()
    if text in DECISIONS:
        return DECISIONS[text]
    return (text or "확인 필요", "이 판정에 대한 설명이 아직 없습니다.")


def action_label(action: object) -> str:
    text = str(action or "").strip()
    return ACTIONS.get(text, "")


def reason_label(reason: object) -> str:
    """Translate one decision reason; unknown reasons are returned unchanged."""
    text = str(reason or "").strip()
    if text in REASONS:
        return REASONS[text]
    for pattern, template in _PATTERNS:
        match = pattern.match(text)
        if match:
            groups = list(match.groups())
            if pattern.pattern.startswith("^skin-specialized binding context"):
                groups[0] = skin_context_label(groups[0])
            return template.format(*groups)
    return text
