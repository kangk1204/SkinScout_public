"""Static UI contract tests for the SkinScout Workbench beginner flow."""

from __future__ import annotations

from pathlib import Path

# 아래 `pytest.skip` 이 이것 없이 쓰이고 있었다. 산출물이 있는 이 기계에서는 그
# 줄에 닿지 않아 드러나지 않았지만, 산출물이 없는 곳(새로 깐 공동연구자 기계가
# 그렇다)에서는 건너뛰는 대신 NameError 로 죽는다 - 안전장치가 안전하지 않았다.
import pytest

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "workbench" / "static"


def _read(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


def test_beginner_flow_hides_stage0_action_and_keeps_target_data_readiness() -> None:
    html = _read("index.html")
    javascript = _read("app.js")

    assert "start-stage0" not in html
    assert "/api/setup/stage0" not in javascript
    assert 'check.id !== "stage0"' in javascript
    # The target-prediction card still has to say the run needs prepared target
    # data - just not by naming an internal profile at a wet-lab reader.
    assert "어떤 단백질에 작용하나" in html
    assert "Stage0" not in html.split('name="preset"', 1)[1]
    assert "첫 분석 때 몰래 다운로드하지 않습니다." in html


def test_setup_button_reaches_idempotent_installer_after_demo_is_ready() -> None:
    javascript = _read("app.js")

    assert "install.disabled = !system.ubuntu_ready;" in javascript
    assert 'profile === "full" ? "Full 프로필 확인·확장"' in javascript
    assert '"Demo 프로필 다시 확인"' in javascript
    assert "runtime.base_env && runtime.p2rank);" not in javascript


def test_evidence_is_default_and_discovery_is_guarded() -> None:
    html = _read("index.html")
    javascript = _read("app.js")

    assert 'name="evidence_mode" value="evidence" checked' in html
    assert 'name="evidence_mode" value="discovery" disabled' not in html
    assert 'name="evidence_mode" value="discovery">' in html
    # The gate has to be stated in words a wet-lab reader parses: discovery
    # will not start until the self-match exclusion list is verified.
    assert "그 목록이 검증되지 않으면 실행을 시작하지 않습니다" in html
    assert "evidence_mode: evidenceMode" in javascript
    assert '|| "evidence"' in javascript
    assert 'evidenceMode === "discovery" ? "Discovery" : "Evidence"' in javascript
    assert "${evidenceModeLabel} 모드로 실행됩니다." in javascript


def test_input_accessibility_and_sdf_hash_state_are_present() -> None:
    html = _read("index.html")
    javascript = _read("app.js")
    css = _read("styles.css")

    assert "skip-link" in html
    assert 'href="#main-content"' in html
    assert 'id="main-content" tabindex="-1"' in html
    assert 'aria-pressed="true"' in html
    assert 'aria-describedby="sdf-validation sdf-hash"' in html
    assert "파일 해시 대기" in html
    assert "sha256Hex" in javascript
    assert "SHA256 계산 중" in javascript
    assert ".file-drop:focus-within" in css
    assert ".choice-card:focus-within" in css
    assert ".choice-card.is-selected:focus-within" in css
    assert ".mode-card:focus-within" in css
    assert ".mode-card.is-selected:focus-within" in css
    assert ".segmented-control input:focus-visible + span" in css


def test_submit_and_sdf_upload_have_beginner_guards() -> None:
    javascript = _read("app.js")

    assert "submitting: false" in javascript
    assert "if (state.submitting) return;" in javascript
    assert 'submit.textContent = state.submitting ? "분석 시작 중" : "분석 시작"' in javascript
    assert "const MAX_SDF_BYTES = 10 * 1024 * 1024" in javascript
    assert "if (file.size > MAX_SDF_BYTES)" in javascript
    assert "state.sdfContent = await file.text();" in javascript
    assert javascript.index("if (file.size > MAX_SDF_BYTES)") < javascript.index("state.sdfContent = await file.text();")


def test_substitute_discovery_has_beginner_input_and_interactive_results() -> None:
    html = _read("index.html")
    javascript = _read("app.js")
    css = _read("styles.css")

    assert 'name="preset" value="substitute"' in html
    assert 'id="substitute-target-id"' in html
    assert 'id="substitute-max-candidates"' in html
    assert 'id="substitute-target-conditioned" type="checkbox" checked' in html
    assert "타겟을 비우면 입력 화합물의 1순위 예측 타겟" in html
    assert '"substitute_target_conditioned"' in javascript
    assert 'payload.target_id = $("#substitute-target-id").value.trim().toUpperCase();' in javascript
    assert 'payload.max_candidates = Number($("#substitute-max-candidates").value);' in javascript
    assert 'payload.target_conditioned = $("#substitute-target-conditioned").checked;' in javascript
    assert 'report.target?.interaction_anchor_conditioned !== false' in javascript
    assert "대체소재 발굴은 동일 타겟의 공개 활성 근거를 비교하므로 Evidence 모드로만 실행됩니다." in javascript
    assert "function filteredSubstituteCandidates()" in javascript
    assert 'id="substitute-table-body"' in javascript
    assert "function substituteTrackLabel(track)" in javascript
    assert "target_conditioned_anchor_score" in javascript
    assert "target_conditioned_mapping_ambiguous" in javascript
    assert "다중 대응 보수값" in javascript
    assert "표적 anchor 적용" in javascript
    assert "2D gate 통과" in javascript
    assert "동일 타겟 근거 입장" in javascript
    assert "화장품 원료 트랙" in javascript
    assert "global_priority_rank" in javascript
    assert "feature_family_f1" in javascript
    assert "binding_pactivity_delta" in javascript
    assert "ΔpActivity" in javascript
    assert "구조·물성 triage" in javascript
    assert "안전성 triage" not in javascript
    assert "동일 결합 자세, 동등 효능, 피부 안전성을 뜻하지 않습니다." in javascript
    assert ".preset-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr))" in css
    assert ".substitute-table { min-width: 1080px; }" in css
    assert ".source-chip--track" in css
    assert ".substitute-anchor-option" in css
    assert ".result-grid > * { min-width: 0; }" in css


def test_beginner_run_filter_helpers_are_used_for_dynamic_lists() -> None:
    javascript = _read("app.js")

    assert "const BEGINNER_INTERNAL_RUN_RE" in javascript
    assert "function isBeginnerVisibleRun(run)" in javascript
    assert "function beginnerRuns()" in javascript
    assert "function beginnerCompletedRuns()" in javascript
    assert "const visibleRuns = beginnerRuns();" in javascript
    assert "const completedRuns = beginnerCompletedRuns();" in javascript
    assert "state.runs.filter((run) => run.status === \"completed\")" not in javascript


def test_recovery_and_modal_handlers_are_honest_and_single_bound() -> None:
    javascript = _read("app.js")

    assert "function recoverRun(runId)" in javascript
    assert "function recoverableInput(detail, fallbackRun)" in javascript
    assert "가능한 입력과 실행 설정을 복원했습니다." in javascript
    assert "복원 가능한 입력이 없어 새 분석 입력 화면을 열었습니다." in javascript
    assert 'canRecoverInput ? "입력 복원" : "새 분석"' in javascript
    assert 'container.querySelectorAll(".target-row").forEach' in javascript
    assert '".target-row, .table-action"' not in javascript


def test_target_results_expose_hpa_cell_context_without_overclaiming() -> None:
    javascript = _read("app.js")

    assert "row.cell_type_preferred" in javascript
    assert "우세 HPA 세포 유형" in javascript
    assert "발현 맥락이며 해당 세포에서 선택적으로 작용한다는 뜻이 아닙니다" in javascript


def test_mobile_navigation_and_setup_steps_do_not_hide_actions() -> None:
    css = _read("styles.css")

    assert ".primary-nav { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr))" in css
    assert ".setup-stepper { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr))" in css
    assert ".step-line { display: none; }" in css


def test_the_similarity_bands_match_what_the_panel_actually_measured() -> None:
    """Recompute the bands, do not grep for them.

    The earlier version of this test asserted that three strings appeared in
    both `app.js` and the README. It passed while the UI and the documents
    disagreed: `similarityCell` bands with `value >= min`, the published counts
    had been derived with a strict `>`, and the one pair at exactly sim 0.400 -
    a top-30 miss at rank 61 - was red in the docs and amber on screen.

    So the thresholds and the counts are both read out of the code and checked
    against the measurement, with the operator the code actually uses.
    """
    import csv
    import re

    javascript = _read("app.js")
    panel = (
        ROOT / "results" / "eval" / "activity_retrieval_202608" / "runpath_panel"
        / "band_runtimeidx.csv"
    )
    if not panel.exists():
        pytest.skip("band measurement artifact is not built here")

    block = javascript.split("SIMILARITY_BANDS", 1)[1].split("];", 1)[0]
    thresholds = [float(value) for value in re.findall(r"min:\s*([0-9.]+)", block)]
    assert thresholds == [0.6, 0.4, 0.0], thresholds
    # The operator the labels are counted under; a change here changes the counts.
    assert "value >= entry.min" in javascript

    with panel.open(encoding="utf-8") as handle:
        pairs = [(float(row["sim"]), float(row["rank"])) for row in csv.DictReader(handle)]
    assert len(pairs) == 46, len(pairs)

    counted = []
    for low, high in ((0.6, 1.01), (0.4, 0.6), (0.0, 0.4)):
        band = [(s, r) for s, r in pairs if low <= s < high]
        counted.append((len(band), sum(1 for _, r in band if r <= 30)))

    labels = re.findall(r'label:\s*"([^"]+)"', block)
    assert len(labels) == 3, labels
    for (total, hits), label in zip(counted, labels, strict=True):
        assert str(total) in label, (total, label)
        assert str(hits) in label, (hits, label)

    # And the README's table has to be the same measurement, not a second one.
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for total, hits in counted[1:]:
        assert f"{total}쌍 중 {hits}쌍" in readme, (total, hits)


def test_the_band_comment_records_that_green_was_unreachable_until_the_fix() -> None:
    """Anyone comparing an old run against a new one will see this column jump
    by 4x. The reason has to sit next to the thresholds, not only in a doc."""
    javascript = _read("app.js")

    context = javascript.split("SIMILARITY_BANDS", 1)[0][-800:]
    assert "explicit hydrogens" in context
    assert "2026-08-31" in context


def test_ordinal_scores_and_historical_performance_are_disclosed() -> None:
    javascript = _read("app.js")
    assert "final_score_semantics" in javascript
    assert "ordinal_rank" in javascript
    assert "순위 표시값" in javascript
    assert "결합 확률 100%" in javascript
    assert "현재 코드·데이터 성능은 재검증이 필요합니다" in javascript


def test_the_drawer_warns_when_the_nearest_evidence_is_below_threshold() -> None:
    """The rank does not use the label, so the screen has to.

    alpha-arbutin's top target is tyrosinase on evidence under the threshold.
    The warning must not call that "inactive" - it is a real, weak inhibition,
    and the label is named for the index's threshold policy.
    """
    javascript = _read("app.js")

    assert "EVIDENCE_STRENGTH" in javascript
    for state in ("at_or_above_threshold", "between_thresholds", "below_threshold"):
        assert state in javascript, state
    assert "evidenceStrengthNote" in javascript
    # The warning fires only for the below-threshold case.
    note = javascript.split("function evidenceStrengthNote", 1)[1].split("\n  }", 1)[0]
    assert 'daina_supporting_evidence !== "below_threshold"' in note
    assert "약한 작용일 수도" in note, "below threshold must not be equated to inactive"
    assert "비활성" not in note, "do not overstate what the threshold means"


def test_every_run_choice_discloses_remote_sensitization_calls() -> None:
    html = (ROOT / "workbench" / "static" / "index.html").read_text(encoding="utf-8")

    choice_block = html.split('<div class="preset-grid">', 1)[1].split("</div>", 1)[0]
    for preset in ("safety", "target-id", "substitute", "report"):
        card = choice_block.split(f'value="{preset}"', 1)[1].split("</label>", 1)[0]
        assert "외부 서버" in card
    assert "모든 분석 경로에 포함되는 감작성 예측" in html
