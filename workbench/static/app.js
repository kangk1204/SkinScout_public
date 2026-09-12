(() => {
  "use strict";

  const state = {
    view: "setup",
    status: null,
    setup: null,
    runs: [],
    selectedRunId: null,
    detail: null,
    inputType: "smiles",
    sdfContent: "",
    targetRows: [],
    targetTotal: 0,
    targetSource: null,
    targetError: null,
    evidenceMode: "evidence",
    submitting: false,
    toastTimer: null,
  };
  const VALID_VIEWS = new Set(["setup", "analyze", "alternatives", "runs", "results", "explore", "settings"]);
  const MAX_SDF_BYTES = 10 * 1024 * 1024;
  const UNIPROT_ACCESSION_RE = /^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})(?:-[0-9]+)?$/;
  const BEGINNER_INTERNAL_RUN_RE = /\b(?:stage0|bootstrap|operator)\b|stage0_|_stage0|bootstrap_|_bootstrap|operator_|_operator/i;

  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => Array.from(document.querySelectorAll(selector));

  async function api(path, options = {}) {
    const method = String(options.method || "GET").toUpperCase();
    const { headers: optionHeaders = {}, ...requestOptions } = options;
    const tokenHeader = method === "GET" || !state.status?.csrf_token
      ? {}
      : { "X-SkinScout-Token": state.status.csrf_token };
    const response = await fetch(path, {
      ...requestOptions,
      headers: { "Content-Type": "application/json", ...tokenHeader, ...optionHeaders },
    });
    const payload = await response.json().catch(() => ({}));
    if (response.status === 401) {
      // The session expired or was never opened; show the sign-in screen rather
      // than a stream of failures behind a dead page.
      showLoginGate();
      throw new Error(payload.error || "로그인이 필요합니다.");
    }
    if (!response.ok) throw new Error(payload.error || `요청 실패 (${response.status})`);
    return payload;
  }

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function formatScore(value) {
    if (value === null || value === undefined || value === "") return "-";
    const number = Number(value);
    return Number.isFinite(number) ? number.toFixed(3) : "-";
  }

  function formatCount(value) {
    if (value === null || value === undefined || value === "") return "-";
    return Number(value).toLocaleString("ko-KR");
  }

  function statusLabel(status) {
    const labels = { ready: "준비됨", warning: "주의", partial: "부분 준비", action: "작업 필요", blocked: "차단됨", manual: "수동 준비", available: "가능", unregistered: "등록 필요", missing: "없음", queued: "대기 중", running: "실행 중", cancel_requested: "취소 중", cancelled: "취소됨", completed: "완료", failed: "실패", incomplete: "미완료", stale: "오래된 상태", verification_failed: "검증 실패", unverified: "미검증" };
    return labels[status] || status || "확인 중";
  }

  function beginnerText(value) {
    return String(value ?? "")
      .replaceAll("Stage 0 데이터", "표적 데이터")
      .replaceAll("Stage 0", "표적 데이터 준비")
      .replaceAll("stage0", "target-data");
  }

  function selectedSetupProfile() {
    return document.querySelector('input[name="setup_profile"]:checked')?.value || "demo";
  }

  function isBeginnerVisibleRun(run) {
    if (!run) return false;
    const searchable = [run.run_id, run.preset, run.mode, run.kind, run.profile, run.label]
      .filter((value) => value !== null && value !== undefined)
      .join(" ");
    return !BEGINNER_INTERNAL_RUN_RE.test(searchable);
  }

  function beginnerRuns() {
    return state.runs.filter(isBeginnerVisibleRun);
  }

  function beginnerCompletedRuns() {
    return beginnerRuns().filter((run) => ["completed", "verification_failed", "unverified"].includes(run.status));
  }

  function statusTone(status) {
    if (["ready", "available", "completed"].includes(status)) return "success";
    if (["blocked", "failed", "verification_failed"].includes(status)) return "danger";
    if (["queued", "running", "cancel_requested"].includes(status)) return "info";
    if (["warning", "partial", "action", "manual", "unregistered", "missing", "cancelled", "incomplete", "stale", "unverified"].includes(status)) return "warning";
    return "neutral";
  }

  async function sha256Hex(file) {
    if (!window.crypto?.subtle) return "";
    const buffer = await file.arrayBuffer();
    const hash = await crypto.subtle.digest("SHA-256", buffer);
    return Array.from(new Uint8Array(hash)).map((byte) => byte.toString(16).padStart(2, "0")).join("");
  }

  function actionLabel(action, claimable) {
    if (action === "proceed" && claimable !== true) return "근거 보완 필요";
    const labels = {
      proceed: "후보 검토 진행",
      review_before_claim: "검토 후 판단",
      stop_before_claim: "주장 중지",
    };
    return labels[action] || "결과 검토";
  }

  function skinContextLabel(decision) {
    const labels = {
      skin_context_supported: "피부 맥락 근거 지원",
      skin_expression_only: "피부 발현 근거만 지원",
      skin_efficacy_literature_only: "피부 효능 문헌만 지원",
      insufficient_skin_context: "피부 맥락 근거 부족",
    };
    return labels[decision] || decision || "확인 필요";
  }

  function failureModeLabel(mode) {
  if (mode === "missing_metal_cofactor") return "금속 보조인자 누락";
  if (mode === "gpcr_inactive_state") return "GPCR 비활성 상태로 모델링됨";
  return mode || "알려진 한계";
}

function skinTierLabel(tier) {
    const labels = { very_high: "매우 높음", high: "높음", medium: "보통", low: "낮음", very_low: "매우 낮음", unknown: "미확인" };
    return labels[tier] || tier || "미확인";
  }

  function efficacyLabel(value) {
    const labels = {
      hair_growth: "모발 성장",
      acne: "여드름",
      anti_aging: "피부 노화",
      anti_inflammatory: "항염",
      whitening: "색소 완화",
      retinoid: "레티노이드",
    };
    const text = String(value || "");
    const match = text.match(/^([A-Za-z0-9_-]+)(.*)$/);
    if (!match) return text;
    return `${labels[match[1]] || match[1]}${match[2].replace("papers", "문헌")}`;
  }

  function reasonLabel(reason) {
    const text = String(reason || "");
    const exact = {
      "skin sensitization consensus is HALT": "피부 감작성 종합 판정이 HALT입니다.",
      "skin sensitization consensus is FLAG_HIGH": "피부 감작성 종합 판정에 사람 검토가 필요합니다.",
      "skin toxicity is HALT": "피부 독성 판정이 HALT입니다.",
      "skin toxicity requires review": "피부 독성 결과를 사람이 검토해야 합니다.",
      "safety evidence is degraded": "일부 안전성 근거가 제한된 상태입니다.",
      "target prediction summary is missing": "표적 예측 요약이 없습니다.",
      "target prediction produced no top target": "상위 표적 후보가 생성되지 않았습니다.",
      "skin-specialized binding summary is missing": "피부 특화 결합 요약이 없습니다.",
      "safety and cosmetic/drug gates passed": "안전성과 화장품·의약품 관련 필수 검사를 통과했습니다.",
    };
    if (exact[text]) return exact[text];
    let match = text.match(/^top predicted target is (.+) among ([0-9,]+) ranked targets$/);
    if (match) return `상위 예측 표적은 ${match[1]}이며, 총 ${match[2]}개 후보가 순위화되었습니다.`;
    match = text.match(/^skin-specialized binding context requires review: (.+)$/);
    if (match) return `피부 특화 결합 맥락을 검토해야 합니다: ${skinContextLabel(match[1])}.`;
    match = text.match(/^top binding target lacks direct skin context support: (.+)$/);
    if (match) return `상위 결합 표적 ${match[1]}에 직접적인 피부 맥락 근거가 부족합니다.`;
    match = text.match(/^high ADMET risk endpoints: (.+)$/);
    if (match) return `높은 ADMET 위험 항목: ${match[1]}.`;
    match = text.match(/^structural alerts present: (.+)$/);
    if (match) return `구조 경고가 발견되었습니다: ${match[1]}.`;
    match = text.match(/^missing skin-sens models: (.+)$/);
    if (match) return `누락된 피부 감작성 모델: ${match[1]}.`;
    match = text.match(/^drug-avoidance warnings present: (.+)$/);
    if (match) return `의약품 유사성 경고 ${match[1]}건이 있습니다.`;
    return text;
  }

  function cosmeticDecisionLabel(decision) {
    const labels = { PROCEED: "통과", DOWNWEIGHT: "주의", HALT: "중지" };
    return labels[decision] || decision || "확인 필요";
  }

  function substituteTierLabel(tier) {
    const labels = {
      direct_retained: "동일 endpoint/source 활성 유지 범위",
      direct_activity: "동일 타겟 활성 근거",
      direct_reduced: "활성 저하 관찰",
      proxy_only: "Ligand-feature 유사성만",
    };
    return labels[tier] || tier || "근거 미확인";
  }

  function substitutePriorityLabel(priority) {
    const labels = {
      priority_validation: "우선 실험 검증",
      target_evidence_review: "타겟 근거 검토",
      cosmetic_proxy_review: "화장품 후보 검토",
      research_hypothesis: "연구 가설",
    };
    return labels[priority] || priority || "검토 필요";
  }

  function substituteTrackLabel(track) {
    const labels = {
      target_activity: "표적 근거 트랙",
      cosmetic_material: "화장품 원료 트랙",
      feature_proxy: "Feature proxy 트랙",
    };
    return labels[track] || track || "트랙 미확인";
  }

  function decisionInfo(decision, action, claimable) {
    if (claimable === true) return { label: "PASS", title: "검토 통과", text: "계약상 주요 검증을 통과했습니다. 실험적 효능의 증명은 아닙니다.", tone: "success" };
    // A run that never produced a summary has no verdict to report. Falling
    // through to DIAGNOSTIC told the reader a failed or still-queued run was
    // "usable for exploration" when there was no result at all. The server
    // always sends claimable as a boolean, so absence shows up as a missing
    // decision and recommended_action.
    if (!decision && !action) {
      return { label: "—", title: "판정 없음", text: "아직 판정할 결과가 없습니다.", tone: "neutral" };
    }
    if (decision === "HALT" || action === "stop_before_claim") return { label: "HALT", title: "주장 중지", text: "현재 결과로 효능 또는 결합을 주장할 수 없습니다.", tone: "danger" };
    if (decision === "FLAG_HIGH" || action === "review_before_claim") return { label: "REVIEW", title: "사람 검토 필요", text: "안전성, 피부 맥락, 또는 검증 상태를 확인한 뒤 다음 결정을 내려야 합니다.", tone: "warning" };
    if (decision === "PASS") return { label: "DIAGNOSTIC", title: "검증 제한 결과", text: "생물학적 판정은 PASS지만 전체 주장 조건을 충족하지 못했습니다.", tone: "warning" };
    return { label: "DIAGNOSTIC", title: "진단용 결과", text: "탐색에는 사용할 수 있지만 claimable 결과가 아닙니다.", tone: "info" };
  }

  function analysisReadinessKey(preset, mode) {
    if (preset === "safety") return "safety";
    if (preset === "substitute") {
      return $("#substitute-target-conditioned")?.checked
        ? "substitute_target_conditioned"
        : "substitute";
    }
    if (preset === "report") return "report";
    return mode === "fast" ? "target_fast" : "target_comprehensive";
  }

  function nextActionInfo(decision, action, claimable) {
    if (claimable === true) return "상위 표적과 근거를 검토한 뒤, 결합·세포 반응·용량 반응 실험의 우선순위를 정하세요.";
    if (decision === "HALT" || action === "stop_before_claim") return "이 결과를 효능 결론으로 사용하지 마세요. 안전성 경고와 실패 사유를 먼저 해결해야 합니다.";
    return "아래 제한 사유와 원본 근거를 검토하세요. 확인 전에는 효능 또는 직접 결합을 주장하지 마세요.";
  }

  function showToast(message) {
    const toast = $("#toast");
    toast.textContent = message;
    toast.classList.add("is-visible");
    clearTimeout(state.toastTimer);
    state.toastTimer = setTimeout(() => toast.classList.remove("is-visible"), 3600);
  }

  function navigate(view) {
    if (!VALID_VIEWS.has(view)) view = "setup";
    state.view = view;
    if (window.location.hash !== `#${view}`) window.history.replaceState(null, "", `#${view}`);
    $$("[data-view-panel]").forEach((panel) => panel.classList.toggle("is-visible", panel.dataset.viewPanel === view));
    $$(".nav-item").forEach((item) => item.classList.toggle("is-active", item.dataset.view === view));
    if (view === "setup") renderSetup();
    if (view === "runs") renderRuns();
    if (view === "results") renderResults();
    if (view === "explore") renderExplorer();
    if (view === "settings") renderSettings();
    if (view === "alternatives") renderAlternativesReadiness();
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function renderTopbar() {
    if (!state.status) return;
    const system = state.status.system || {};
    $("#topbar-os").textContent = system.distro || system.platform || "OS 확인 중";
    $("#topbar-checked").textContent = state.status.checked_at ? `확인 ${new Date(state.status.checked_at).toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit" })}` : "검사 대기";
    const analysis = state.status.analysis_readiness || {};
    const safetyReady = analysis.safety?.ready === true;
    const targetReady = analysis.target_fast?.ready === true;
    const sidebar = $("#sidebar-status");
    const label = targetReady ? "분석 환경 준비됨" : safetyReady ? "안전성 분석 가능" : "준비 상태 확인 필요";
    sidebar.innerHTML = `<span class="status-dot ${safetyReady || targetReady ? "status-dot--success" : "status-dot--warning"}" aria-hidden="true"></span><span>${escapeHtml(label)}</span>`;
  }

  function renderReadiness() {
    const list = $("#readiness-list");
    if (!state.status || !Array.isArray(state.status.checks)) {
      list.innerHTML = `<div class="empty-state"><span>환경 정보를 읽는 중입니다.</span></div>`;
      return;
    }
    list.innerHTML = state.status.checks.filter((check) => check.id !== "stage0").map((check) => `
      <div class="readiness-row">
        <div class="readiness-label">${escapeHtml(beginnerText(check.label))}</div>
        <div class="readiness-detail">${escapeHtml(beginnerText(check.detail))}</div>
        <div class="state-label state-label--${escapeHtml(check.status)}">${escapeHtml(statusLabel(check.status))}</div>
        ${check.action ? `<div class="readiness-action">${escapeHtml(beginnerText(check.action))}</div>` : ""}
      </div>
    `).join("");
    $("#readiness-time").textContent = state.status.checked_at ? new Date(state.status.checked_at).toLocaleString("ko-KR") : "-";
  }

  function renderSetup() {
    if (!state.status) return;
    renderReadiness();
    const stage = state.status.stage0 || {};
    const runtime = state.status.runtime || {};
    const system = state.status.system || {};
    const analysis = state.status.analysis_readiness || {};
    const safetyReady = analysis.safety?.ready === true;
    const targetReady = analysis.target_fast?.ready === true;
    let overall = "확인 필요";
    let tone = "neutral";
    if (!system.ubuntu_ready) { overall = "Ubuntu 준비 필요"; tone = "warning"; }
    else if (!runtime.base_env) { overall = "SkinScout 설치 필요"; tone = "warning"; }
    else if (targetReady) { overall = "분석 가능"; tone = "success"; }
    else if (safetyReady) { overall = "안전성 분석 가능"; tone = "success"; }
    else { overall = "추가 준비 필요"; tone = "warning"; }
    const badge = $("#setup-overall");
    badge.textContent = overall;
    badge.className = `status-badge status-badge--${tone}`;
    $("#setup-lede").textContent = beginnerText(state.status.next_action || "현재 컴퓨터에서 분석을 시작할 수 있는지 확인합니다.");
    $("#safe-boundary").textContent = state.status.safe_boundary || "OS 디스크를 포맷하거나 부트 장치를 수정하지 않습니다.";

    const guidance = state.setup?.os_guidance || {};
    const steps = (guidance.steps || []).map((step) => `<li>${escapeHtml(beginnerText(step))}</li>`).join("");
    const command = guidance.command ? `<div class="install-command"><code>${escapeHtml(guidance.command)}</code><button class="button button--quiet button--small copy-command" data-command="${escapeHtml(guidance.command)}" title="설치 명령 복사">복사</button></div>` : "";
    $("#os-guidance").innerHTML = `<h3>${escapeHtml(beginnerText(guidance.title || "환경 확인 중"))}</h3><p>${escapeHtml(beginnerText(guidance.body || ""))}</p>${steps ? `<ol>${steps}</ol>` : ""}${command}`;
    const install = $("#install-runtime");
    const profile = selectedSetupProfile();
    const baseInstalled = runtime.base_env && runtime.p2rank;
    install.disabled = !system.ubuntu_ready;
    install.textContent = baseInstalled
      ? profile === "full" ? "Full 프로필 확인·확장" : "Demo 프로필 다시 확인"
      : profile === "full" ? "Full 프로필 설치" : "Demo 프로필 설치";
    $$(".setup-step").forEach((step) => step.classList.remove("is-current", "is-complete"));
    const stepState = [
      ["environment", system.ubuntu_ready],
      ["runtime", runtime.base_env],
      ["data", stage.status === "ready" || safetyReady],
      ["ready", safetyReady || targetReady],
    ];
    let foundCurrent = false;
    stepState.forEach(([id, complete]) => {
      const step = $(`[data-setup-step="${id}"]`);
      if (complete) step.classList.add("is-complete");
      else if (!foundCurrent) { step.classList.add("is-current"); foundCurrent = true; }
    });
  }

  function renderRuns() {
    const list = $("#run-list");
    const monitor = $("#run-monitor");
    const visibleRuns = beginnerRuns();
    $("#run-count").textContent = `${visibleRuns.length}개`;
    const monitoredRun = isBeginnerVisibleRun(state.detail?.run) ? state.detail.run : null;
    const activeStatuses = new Set(["queued", "running", "cancel_requested"]);
    const monitorActive = monitoredRun && activeStatuses.has(monitoredRun.status);
    if (monitorActive) {
      const log = state.detail.logs || "로그가 아직 없습니다.";
      const job = state.detail.job || {};
      // Stage state comes from artifacts the run actually wrote. The previous
      // version pinned every running job to stage 1 for its whole life, which
      // on a multi-hour run reads as "stuck" and misleads the cancel decision.
      const progress = monitoredRun.progress || {};
      const stageState = Array.isArray(progress.stages) ? progress.stages : [];
      const determinate = progress.determinate === true && stageState.length > 0;
      const stages = determinate
        ? stageState.map((stage) => stage.label)
        : monitoredRun.preset === "substitute"
          ? ["입력 확인", "안전성", "Target", "Pharmacophore", "후보 순위", "검증"]
          : ["입력 확인", "안전성", "Target", "구조/물리", "보고서", "검증"];
      const doneFlags = determinate ? stageState.map((stage) => stage.done === true) : [];
      // "Current" is the first stage with no output yet, never a fixed guess.
      const currentStage = determinate ? doneFlags.indexOf(false) : -1;
      const progressNote = determinate
        ? `산출물 기준 ${progress.completed}/${stageState.length}단계 완료`
        : "아직 단계를 판정할 산출물이 없습니다. 아래 로그를 보세요.";
      monitor.classList.remove("is-hidden");
      const canCancel = monitoredRun.status !== "cancel_requested";
      monitor.innerHTML = `<div class="monitor-header"><div><p class="eyebrow">LIVE RUN MONITOR</p><h2 id="run-monitor-title">${escapeHtml(monitoredRun.run_id)}</h2><p class="muted">${escapeHtml(monitoredRun.preset || "-")} · ${escapeHtml(monitoredRun.mode || "-")} · ${escapeHtml(statusLabel(job.status || monitoredRun.status || "running"))}</p></div><button class="button button--quiet button--small cancel-run" data-run-id="${escapeHtml(monitoredRun.run_id)}" ${canCancel ? "" : "disabled"} title="실행 중인 분석에 취소 요청 보내기">실행 취소</button></div><div class="monitor-progress">${stages.map((stage, index) => `<div class="monitor-stage ${doneFlags[index] ? "is-done" : ""} ${index === currentStage ? "is-current" : ""}"><strong>0${index + 1}</strong><span>${escapeHtml(stage)}</span></div>`).join("")}</div><p class="muted">${escapeHtml(progressNote)}</p><pre class="monitor-log">${escapeHtml(log)}</pre><div class="monitor-footer"><span class="muted">${monitoredRun.status === "cancel_requested" ? "취소 요청을 보냈습니다. 작업이 안전하게 멈출 때까지 기다립니다." : "분석은 백그라운드에서 계속됩니다. 상태는 자동 갱신됩니다."}</span><button class="button button--secondary button--small" data-view="results">결과 화면</button></div>`;
    } else {
      monitor.classList.add("is-hidden");
      monitor.innerHTML = "";
    }
    if (!visibleRuns.length) {
      list.innerHTML = `<div class="empty-state"><strong>아직 실행 결과가 없습니다.</strong><span>새 분석에서 첫 번째 화합물을 입력하세요.</span><button class="button button--secondary" data-view="analyze">새 분석</button></div>`;
      return;
    }
    list.innerHTML = visibleRuns.map((run) => {
      const info = decisionInfo(run.decision, run.recommended_action, run.claimable);
      const target = run.top_target || {};
      const running = ["queued", "running", "cancel_requested"].includes(run.status);
      const recoverable = ["failed", "cancelled", "incomplete"].includes(run.status);
      const canRecoverInput = Boolean(run.input_smiles || run.canonical_smiles);
      const statusText = run.status === "completed" ? (run.updated_at || "완료") : statusLabel(run.status);
      const verification = run.verification || {};
      const verificationNote = run.status === "verification_failed"
        ? `<p class="muted">검증기가 이 실행을 통과시키지 않았습니다${verification.error ? ` — ${escapeHtml(verification.error)}` : ""}. 결과는 볼 수 있지만 근거로 인용하지 마세요.</p>`
        : run.status === "unverified"
          ? `<p class="muted">검증 기록이 없습니다. 이 실행은 확인되지 않았습니다.</p>`
          : "";
      const targetTitle = run.preset === "substitute" && run.substitute_count !== null && run.substitute_count !== undefined
        ? `${formatCount(run.substitute_count)}개 대체 후보`
        : target.gene_symbol || "top target 없음";
      return `<article class="run-card">
        <div class="run-card-main"><div class="run-id">${escapeHtml(run.run_id)}</div><div class="run-meta"><span>${escapeHtml(run.preset || "-")}</span><span>${escapeHtml(run.mode || "-")}</span><span>${escapeHtml(statusText)}</span></div>${recoverable ? `<div class="run-reason">${canRecoverInput ? "입력을 복원해 새 분석으로 재시도할 수 있습니다." : "복원 가능한 입력이 없어 로그를 확인한 뒤 새 분석을 시작하세요."}</div>` : ""}${verificationNote}</div>
        <div><span class="status-badge status-badge--${info.tone}">${escapeHtml(info.label)}</span></div>
        <div class="run-target"><strong>${escapeHtml(targetTitle)}</strong><span>${escapeHtml(target.protein_name || target.target_id || (running ? "실행 중" : "결과 확인 필요"))}</span></div>
        <div class="run-card-actions"><button class="button button--quiet button--small open-run" data-run-id="${escapeHtml(run.run_id)}">${running ? "모니터" : recoverable ? "로그" : "결과"}</button>${running ? `<button class="button button--quiet button--small cancel-run" data-run-id="${escapeHtml(run.run_id)}" title="실행 취소 요청">취소</button>` : ""}${recoverable ? `<button class="button button--secondary button--small recover-run" data-run-id="${escapeHtml(run.run_id)}" title="${canRecoverInput ? "가능한 입력과 실행 설정을 복원" : "복원 가능한 입력 없음"}">${canRecoverInput ? "입력 복원" : "새 분석"}</button>` : ""}</div>
      </article>`;
    }).join("");
  }

  function renderDecision(summary) {
    const overall = summary?.overall_decision || {};
    const reasons = Array.isArray(overall.reasons) ? overall.reasons : [];
    // run_summary.json is written before the verifier runs, so its own
    // `claimable` is a pipeline verdict, not a verified one. The run list
    // already ANDs it with the verifier; this screen used the raw value and
    // could show a green PASS for a run whose verification failed.
    const verification = state.detail?.verification || null;
    const verified = verification ? verification.ok === true : true;
    const claimable = overall.claimable === true && verified;
    const info = decisionInfo(overall.decision, overall.recommended_action, claimable);
    const nextAction = nextActionInfo(overall.decision, overall.recommended_action, claimable);
    const verificationNote = verification && verification.ok !== true
      ? `<div class="notice notice--warn">검증에 실패한 실행입니다. 이 판정을 근거로 인용하지 마세요.</div>`
      : "";
    return `<div class="decision-banner decision-banner--${info.tone}">
      <div><div class="decision-label">${escapeHtml(info.label)}</div><div class="muted">${claimable ? "검증 기반 주장 가능" : "과학적 주장 불가"}</div></div>
      <div class="decision-copy"><strong>${escapeHtml(info.title)}</strong><p>${escapeHtml(info.text)}</p>${reasons.length ? `<ul class="decision-reasons">${reasons.slice(0, 5).map((reason) => `<li title="${escapeHtml(reason)}">${escapeHtml(reasonLabel(reason))}</li>`).join("")}</ul>` : ""}</div>
      <span class="status-badge status-badge--${info.tone}">${escapeHtml(actionLabel(overall.recommended_action, claimable))}</span>
    </div>${verificationNote}<div class="next-action"><strong>다음에 할 일</strong><span>${escapeHtml(nextAction)}</span></div>`;
  }

  // Measured on the 22-compound / 46-pair panel (2026-08-31): at nearest-analog
  // similarity >= 0.6 all 24 pairs recovered their known target inside the top
  // 30, median rank 3.5; below 0.4 only 2 of 8 did. It is the single number that
  // predicts whether a row is worth acting on, so it belongs in the table rather
  // than one click away.
  //
  // Boundaries are inclusive at the lower edge (`value >= min`), and the counts
  // above are computed the same way. The docs once used a strict `>`, which put
  // the single pair at exactly 0.400 - a top-30 miss at rank 61 - in the red
  // band while this code painted it amber.
  //
  // Reachable in a real run only since 2026-08-31. Stage 3 fingerprinted the
  // query with explicit hydrogens while every reference had none, which capped
  // this column near 0.2 - the green band could not be hit no matter what a
  // researcher submitted.
  const SIMILARITY_BANDS = [
    { min: 0.6, tone: "success", label: "2026-08-31 과거 패널: 24쌍 전부 상위 30위 (중앙 3.5위). 현재 코드·데이터 성능은 재검증이 필요합니다" },
    { min: 0.4, tone: "warning", label: "2026-08-31 과거 패널: 14쌍 중 9쌍이 상위 30위. 현재 성능 보장이 아니며 문헌으로 교차확인하세요" },
    { min: 0, tone: "danger", label: "2026-08-31 과거 패널: 8쌍 중 2쌍이 상위 30위. 현재 성능 보장이 아니며 근거가 부족한 구간입니다" },
  ];

  // The ranking weights how similar the nearest measured analogue is, not how
  // active it turned out to be. On the shipped index 9.9% of ligand-target pairs
  // have only measurements at or below the activity threshold, so a target can
  // sit near the top on an analogue whose own number was weak. Excluding those
  // from the score measured worse on the panel, so the reader is told instead.
  const EVIDENCE_STRENGTH = {
    at_or_above_threshold: "활성 문턱 이상 (pActivity ≥ 6)",
    between_thresholds: "문턱 사이 — 약한 활성 (pActivity 5~6)",
    below_threshold: "활성 문턱 아래 (pActivity ≤ 5)",
    none: "측정값 없음",
    unknown: "확인 불가",
  };

  function evidenceStrengthLabel(value) {
    return EVIDENCE_STRENGTH[value] || EVIDENCE_STRENGTH.unknown;
  }

  function evidenceStrengthNote(row) {
    if (row.daina_supporting_evidence !== "below_threshold") return "";
    return `<div class="notice notice--warn">이 표적을 뒷받침하는 <strong>가장 가까운 근거의 측정값이 활성 문턱 아래</strong>입니다. 순위 레시피는 유사도와 여러 근거를 조합하므로 높은 순위가 강한 활성을 뜻하지 않습니다. 이 줄은 문헌으로 먼저 확인하세요. 약한 작용일 수도 있고 실제로 결합하지 않는 것일 수도 있습니다.</div>`;
  }

  function similarityCell(row) {
    const value = row.daina_max_tanimoto;
    if (value === null || value === undefined) return `<td class="muted">—</td>`;
    if (row.daina_is_self_match) {
      return `<td><span class="status-badge status-badge--info" title="입력과 지문이 같은 참조 리간드가 이 표적에서 이미 측정돼 있습니다. 새 예측이 아니라 조회입니다.">조회</span></td>`;
    }
    const band = SIMILARITY_BANDS.find((entry) => value >= entry.min) || SIMILARITY_BANDS[2];
    return `<td><span class="status-badge status-badge--${band.tone}" title="${escapeHtml(band.label)}">${formatScore(value)}</span></td>`;
  }

  function targetRowsHtml(rows, compact = false) {
    if (!rows || !rows.length) {
      const reason = state.targetError
        ? `${escapeHtml(state.targetError)} 결과가 비어 있는 것이 아니라 파일을 읽지 못한 것입니다.`
        : "표적 ranking이 없습니다.";
      return `<tr><td colspan="8" class="muted">${reason}</td></tr>`;
    }
    return rows.map((row, index) => {
      const rank = row.original_rank || row.rank || index + 1;
      const position = row.filtered_position;
      // A filtered view still reports each target's place in the full ranking;
      // its position on screen is shown separately so the two never merge.
      const rankCell = position && position !== rank
        ? `${escapeHtml(rank)}<br><span class="muted">이 목록 ${escapeHtml(position)}번째</span>`
        : escapeHtml(rank);
      const efficacy = Array.isArray(row.efficacy) ? row.efficacy : [];
      const sources = Array.isArray(row.sources) ? row.sources : [];
      return `<tr class="target-row" data-target-index="${index}">
        <td class="rank-cell">${rankCell}</td>
        <td class="gene-cell"><strong>${escapeHtml(row.gene_symbol || row.target_id || "-")}</strong><span>${escapeHtml(row.protein_name || row.target_id || "")}</span></td>
        ${similarityCell(row)}
        <td class="score-cell" title="${row.final_score_semantics === 'ordinal_rank' ? '순위 표시값이며 결합 확률이나 효능 점수가 아닙니다' : '순위 산정 점수이며 결합 확률이 아닙니다'}">${formatScore(row.final_score)}</td>
        <td><span class="tier-cell">${escapeHtml(skinTierLabel(row.skin_tier))}</span><br><span class="muted">${formatScore(row.skin_score)}</span><br><span class="muted">${escapeHtml(row.cell_type_preferred || "unknown")}</span></td>
        <td><div class="source-chips">${sources.map((source) => `<span class="source-chip">${escapeHtml(source)}</span>`).join("") || "-"}</div></td>
        <td>${efficacy.slice(0, compact ? 1 : 3).map((item) => `<span class="source-chip" title="${escapeHtml(item)}">${escapeHtml(efficacyLabel(item))}</span>`).join(" ") || "-"}</td>
        <td><button class="table-action" data-target-index="${index}">상세</button></td>
      </tr>`;
    }).join("");
  }

  function renderSubstituteDecision(report) {
    const count = Array.isArray(report?.candidates) ? report.candidates.length : 0;
    const targetId = report?.target?.target_id || "선택 타겟";
    const anchorConditioned = report?.target?.interaction_anchor_conditioned === true;
    const evidenceText = anchorConditioned
      ? "엄격한 2D pharmacophore, parent 3D pose에서 확인된 표적 anchor 보존 proxy, 동일 타겟 활성 근거를 분리해 비교했습니다. 후보의 실제 결합 pose와 affinity는 검증되지 않았습니다."
      : "엄격한 2D pharmacophore와 정확한 동일 타겟 활성 근거를 분리해 비교했습니다.";
    return `<div class="decision-banner decision-banner--info">
      <div><div class="decision-label">HYPOTHESIS</div><div class="muted">과학적 주장 불가</div></div>
      <div class="decision-copy"><strong>대체소재 실험 우선순위 ${formatCount(count)}개</strong><p>${escapeHtml(targetId)}에 대해 ${escapeHtml(evidenceText)} 동등한 결합력, 효능, 안전성을 확정한 결과가 아닙니다.</p></div>
      <span class="status-badge status-badge--info">Wet-lab 필수</span>
    </div><div class="next-action"><strong>다음에 할 일</strong><span>상위 후보를 동일 assay 조건에서 비교하고, 피부 투과·세포독성·감작성·제형 안정성을 순서대로 검증하세요.</span></div>`;
  }

  function substituteRowsHtml(rows) {
    if (!rows.length) return `<tr><td colspan="7" class="muted">현재 필터에 맞는 대체 후보가 없습니다.</td></tr>`;
    return rows.map((row) => {
      const sources = Array.isArray(row.candidate_sources) ? row.candidate_sources : [];
      const tracks = Array.isArray(row.selection_tracks) ? row.selection_tracks : [];
      const activity = row.target_activity_median_pactivity === null || row.target_activity_median_pactivity === undefined
        ? "직접값 없음"
        : `pActivity ${formatScore(row.target_activity_median_pactivity)}`;
      const bindingDelta = typeof row.binding_pactivity_delta === "number" && Number.isFinite(row.binding_pactivity_delta)
        ? ` · ΔpActivity ${row.binding_pactivity_delta >= 0 ? "+" : ""}${formatScore(row.binding_pactivity_delta)}`
        : "";
      const pharmacophoreGate = row.pharmacophore_gate_passed === true
        ? "2D gate 통과"
        : "동일 타겟 근거 입장";
      const anchorEvidence = typeof row.target_conditioned_anchor_score === "number" && Number.isFinite(row.target_conditioned_anchor_score)
        ? ` · 표적 anchor ${formatScore(row.target_conditioned_anchor_score)} (${formatCount(row.target_conditioned_preserved_anchor_count)}/${formatCount(row.target_conditioned_anchor_count)})${row.target_conditioned_mapping_ambiguous === true ? " · 다중 대응 보수값" : ""}`
        : "";
      return `<tr class="substitute-row">
        <td class="rank-cell">${escapeHtml(row.rank || "-")}<br><span class="muted">전체 #${escapeHtml(row.global_priority_rank || "-")}</span></td>
        <td class="gene-cell"><strong>${escapeHtml(row.name || row.candidate_id || "-")}</strong><span>${escapeHtml(row.smiles || "")}</span></td>
        <td class="score-cell">${formatScore(row.pharmacophore_preservation_score)}<br><span class="muted">${escapeHtml(pharmacophoreGate)} · recall ${formatScore(row.feature_family_recall)} · F1 ${formatScore(row.feature_family_f1)}${escapeHtml(anchorEvidence)}</span></td>
        <td><strong>${escapeHtml(substituteTierLabel(row.evidence_tier))}</strong><br><span class="muted">${escapeHtml(activity)} · ${formatCount(row.activity_evidence_count)}건${escapeHtml(bindingDelta)}</span></td>
        <td class="score-cell">${formatScore(row.safety_triage_score)}<br><span class="muted">경고 ${formatCount(row.structural_alert_count)}건</span></td>
        <td class="score-cell">${formatScore(row.routeability_proxy)}</td>
        <td><div class="source-chips">${tracks.map((track) => `<span class="source-chip source-chip--track">${escapeHtml(substituteTrackLabel(track))}</span>`).join("")}</div><div class="source-chips">${sources.map((source) => `<span class="source-chip">${escapeHtml(source)}</span>`).join("") || "-"}</div><span class="priority-label">${escapeHtml(substitutePriorityLabel(row.priority_label))}</span></td>
      </tr>`;
    }).join("");
  }

  function filteredSubstituteCandidates() {
    const report = state.detail?.substitutes;
    const candidates = Array.isArray(report?.candidates) ? report.candidates : [];
    const query = $("#substitute-query")?.value.trim().toLowerCase() || "";
    const tier = $("#substitute-tier")?.value || "all";
    const source = $("#substitute-source")?.value || "all";
    return candidates.filter((candidate) => {
      const searchable = [
        candidate.name,
        candidate.smiles,
        ...(Array.isArray(candidate.alternate_names) ? candidate.alternate_names : []),
        ...(Array.isArray(candidate.candidate_sources) ? candidate.candidate_sources : []),
        ...(Array.isArray(candidate.selection_tracks) ? candidate.selection_tracks : []),
        ...(Array.isArray(candidate.admission_bases) ? candidate.admission_bases : []),
        ...(Array.isArray(candidate.cosmetic_functions) ? candidate.cosmetic_functions : []),
      ].join(" ").toLowerCase();
      const sourceMatch = source === "all"
        || (source === "cosing" && candidate.cosing_reference === true)
        || (source === "activity" && candidate.evidence_tier !== "proxy_only");
      return (!query || searchable.includes(query))
        && (tier === "all" || candidate.evidence_tier === tier)
        && sourceMatch;
    });
  }

  function updateSubstituteTable() {
    const body = $("#substitute-table-body");
    if (!body) return;
    const rows = filteredSubstituteCandidates();
    body.innerHTML = substituteRowsHtml(rows);
    const visible = $("#substitute-visible-count");
    if (visible) visible.textContent = `${formatCount(rows.length)}개 표시`;
  }

  function renderSubstituteSection(report) {
    if (!report) return "";
    if (report.status === "blocked") {
      return `<section class="surface result-section substitute-section"><div class="surface-heading"><div><p class="eyebrow">SUBSTITUTE DISCOVERY</p><h2>대체소재 보고서 차단</h2></div><span class="status-badge status-badge--danger">계약 위반</span></div><div class="limitations">${escapeHtml(report.error || "보고서 계약을 검증하지 못했습니다.")}</div></section>`;
    }
    const candidates = Array.isArray(report.candidates) ? report.candidates : [];
    const summary = report.summary || {};
    const strategy = report.selection_strategy || {};
    const targetId = report.target?.target_id || "-";
    const anchorConditioned = report.target?.interaction_anchor_conditioned === true;
    return `<section class="surface result-section substitute-section" aria-labelledby="substitute-result-title">
      <div class="surface-heading"><div><p class="eyebrow">SUBSTITUTE DISCOVERY</p><h2 id="substitute-result-title">Pharmacophore 대체소재 후보</h2></div><span class="status-badge status-badge--info">Hypothesis only</span></div>
      <div class="substitute-summary">
        <div><span>선택 타겟</span><strong>${escapeHtml(targetId)}</strong></div>
        <div><span>전체 후보</span><strong>${formatCount(candidates.length)}</strong></div>
        <div><span>동일 타겟 활성</span><strong>${formatCount(summary.direct_activity_candidates)}</strong></div>
        <div><span>활성 유지 범위</span><strong>${formatCount(summary.direct_retained_candidates)}</strong></div>
        <div><span>엄격한 2D gate</span><strong>${formatCount(summary.strict_pharmacophore_candidates)}</strong></div>
        <div><span>표적 anchor 적용</span><strong>${anchorConditioned ? formatCount(summary.target_conditioned_anchor_candidates) : "미사용"}</strong></div>
        <div><span>화장품 원료 트랙</span><strong>${formatCount(summary.cosing_candidates)}</strong></div>
      </div>
      <p class="claim-note">후보 바스켓은 엄격한 2D pharmacophore, 표적 활성 근거, CosIng 원료를 분리해 구성하며, 2D gate ${formatCount(strategy.reserved_pharmacophore_slots)}자리와 원료 트랙 ${formatCount(strategy.reserved_material_slots)}자리를 예약했습니다. ${anchorConditioned ? "표적 anchor는 parent 3D pose에서 확인된 원자를 2D MCS/feature로 옮기며, 여러 대응이 있으면 최소 보존값을 사용하는 순위 proxy입니다. 후보 pose는 검증하지 않습니다. " : ""}“활성 유지 범위”는 같은 endpoint와 source의 보수적 pActivity 비교이며 동일 결합 자세, 동등 효능, 피부 안전성을 뜻하지 않습니다.</p>
      <div class="substitute-filters">
        <label><span class="field-label">후보 검색</span><input id="substitute-query" class="text-input" type="search" placeholder="이름, SMILES, 기능"></label>
        <label><span class="field-label">근거 단계</span><select id="substitute-tier" class="select-input"><option value="all">전체</option><option value="direct_retained">활성 유지 범위</option><option value="direct_activity">동일 타겟 활성</option><option value="direct_reduced">활성 저하</option><option value="proxy_only">Feature proxy만</option></select></label>
        <label><span class="field-label">후보 트랙</span><select id="substitute-source" class="select-input"><option value="all">전체</option><option value="cosing">화장품 원료</option><option value="activity">표적 활성 근거</option></select></label>
        <span class="substitute-visible" id="substitute-visible-count">${formatCount(candidates.length)}개 표시</span>
      </div>
      <div class="table-wrap"><table class="data-table substitute-table"><caption class="sr-only">Pharmacophore 유지형 대체소재 후보 순위</caption><thead><tr><th>순위</th><th>후보 / 구조</th><th>Pharmacophore</th><th>동일 타겟 근거</th><th>구조·물성 triage</th><th>합성 용이성 proxy</th><th>출처 / 우선순위</th></tr></thead><tbody id="substitute-table-body">${substituteRowsHtml(candidates)}</tbody></table></div>
    </section>`;
  }

  function renderViewerSection(runId, viewerState) {
    const section = $("#result-viewer");
    const mount = $("#result-viewer-mount");
    const note = $("#result-viewer-note");
    const open = $("#result-viewer-open");
    if (!section) return;
    section.classList.remove("is-hidden");
    if (!viewerState.available || !runId) {
      mount.innerHTML = "";
      state.viewerFrameRun = null;
      open.classList.add("is-hidden");
      note.textContent = viewerState.reason
        ? `3D 뷰어를 만들지 못했습니다: ${viewerState.reason}`
        : "이 실행에는 3D 뷰어가 없습니다.";
      return;
    }
    const src = `/api/runs/${encodeURIComponent(runId)}/viewer/index.html`;
    open.classList.remove("is-hidden");
    open.href = src;
    note.textContent = "순위표에서 유전자명을 누르면 수용체와 도킹된 리간드가 함께 표시됩니다. 이 폴더는 그대로 압축해 보내도 상대 컴퓨터에서 열립니다.";
    // Only build the frame when the run changes. This container is outside the
    // subtree renderResults rewrites, so the five-second poll never reloads the
    // 3D view or discards where the reader navigated to.
    if (state.viewerFrameRun === runId && mount.firstElementChild) return;
    mount.innerHTML = "";
    const frame = document.createElement("iframe");
    frame.className = "viewer-frame";
    frame.title = "표적 예측 결과 3D 뷰어";
    frame.src = src;
    mount.appendChild(frame);
    state.viewerFrameRun = runId;
  }

  // A skin researcher thinks in 미백 / 주름 / 진정, not in UniProt accessions.
  // The literature labels are already attached to each target; this rolls them
  // up so the first thing on screen is what the compound is reported to do.
  const EFFICACY_LABELS = {
    anti_aging: "주름 · 노화",
    whitening: "미백 · 색소",
    pigmentation: "미백 · 색소",
    hair_growth: "모발",
    acne: "여드름",
    barrier: "장벽 · 보습",
    moisturizing: "장벽 · 보습",
    soothing: "진정 · 항염",
    anti_inflammatory: "진정 · 항염",
    wound_healing: "상처 치유",
    uv_protection: "자외선",
    antioxidant: "항산화",
  };

  function efficacySummaryHtml(rows) {
    const tally = new Map();
    (rows || []).forEach((row) => {
      (Array.isArray(row.efficacy) ? row.efficacy : []).forEach((entry) => {
        const key = String(entry).split(" (")[0].trim();
        if (!key) return;
        const label = EFFICACY_LABELS[key] || key.replace(/_/g, " ");
        const bucket = tally.get(label) || { count: 0, genes: new Set() };
        bucket.count += 1;
        if (row.gene_symbol) bucket.genes.add(row.gene_symbol);
        tally.set(label, bucket);
      });
    });
    if (!tally.size) {
      return `<section class="surface"><div class="surface-heading"><div><p class="eyebrow">SKIN EFFECT</p><h2>피부에서 보고된 작용</h2></div></div><p class="muted">상위 표적에 연결된 피부 효능 문헌이 없습니다. 이 화합물에 대해 이 도구가 말할 수 있는 피부 맥락이 없다는 뜻이며, 작용이 없다는 뜻은 아닙니다.</p></section>`;
    }
    const chips = Array.from(tally.entries())
      .sort((a, b) => b[1].count - a[1].count)
      .map(([label, bucket]) => `<span class="effect-chip"><strong>${escapeHtml(label)}</strong><span>${Array.from(bucket.genes).slice(0, 4).map(escapeHtml).join(", ")}</span></span>`)
      .join("");
    return `<section class="surface"><div class="surface-heading"><div><p class="eyebrow">SKIN EFFECT</p><h2>피부에서 보고된 작용</h2></div><span class="muted">상위 표적에 연결된 문헌 기준</span></div><div class="effect-chips">${chips}</div><p class="muted">문헌이 그 표적을 이 효능과 연결했다는 뜻이며, <strong>이 화합물이 그 효능을 낸다는 근거는 아닙니다.</strong></p></section>`;
  }

  function renderResults() {
    const empty = $("#result-empty");
    const content = $("#result-content");
    if (!state.detail || !state.detail.summary) {
      empty.classList.remove("is-hidden");
      content.classList.add("is-hidden");
      // The viewer lives outside #result-content, so it has to be hidden here.
      $("#result-viewer").classList.add("is-hidden");
      $("#result-viewer-mount").innerHTML = "";
      state.viewerFrameRun = null;
      return;
    }
    empty.classList.add("is-hidden");
    content.classList.remove("is-hidden");
    const summary = state.detail.summary;
    const compound = summary.compound || {};
    const safety = summary.safety || {};
    const toxicity = summary.skin_toxicity || {};
    const cosmetic = summary.cosmetic_drug || {};
    const binding = summary.skin_specialized_binding || {};
    const target = summary.target_prediction || {};
    const top = Array.isArray(target.top_targets) ? target.top_targets.slice(0, 10) : [];
    const admet = safety.admet_risk_assessment || {};
    const reasons = Array.isArray(summary.overall_decision?.reasons) ? summary.overall_decision.reasons : [];
    const artifacts = Array.isArray(state.detail.artifacts) ? state.detail.artifacts : [];
    const substitutes = state.detail.substitutes;
    const substituteRun = state.detail.run?.preset === "substitute" || Boolean(substitutes);
    const artifactCount = artifacts.length;
    const artifactRows = artifacts.map((artifact) => {
      const href = typeof artifact.download_url === "string" ? artifact.download_url : "";
      const status = artifact.status === "available" ? "available" : artifact.status;
      const tone = statusTone(status);
      const citation = `${state.selectedRunId || "-"}:${artifact.path}`;
      return `<div class="artifact-row"><span>${escapeHtml(artifact.label)}<small>${escapeHtml(citation)}</small></span><code>${escapeHtml(artifact.path)}</code><span class="artifact-actions"><span class="status-badge status-badge--${tone}">${escapeHtml(statusLabel(status))}</span>${artifact.status === "available" && href ? `<a href="${escapeHtml(href)}" target="_blank" rel="noreferrer" title="새 탭에서 산출물 열기">열기</a>` : ""}</span></div>`;
    }).join("");
    const runId = state.selectedRunId || "";
    const viewerState = state.detail.run?.viewer || {};
    // The 3D viewer is written beside every finished run and was reachable only
    // by finding the file on disk; the screen that started the run never
    // mentioned it. Same-origin iframe, so the offline copy stays byte-identical
    // to the one a colleague receives.
    const downloadBar = runId
      ? `<div class="download-bar"><a class="button button--secondary button--small" href="/api/runs/${encodeURIComponent(runId)}/bundle.zip">결과 전체 내려받기 (.zip)</a>${viewerState.available ? `<a class="button button--quiet button--small" href="/api/runs/${encodeURIComponent(runId)}/file?path=viewer%2Fsummary.csv">순위표 CSV (엑셀)</a>` : ""}<a class="button button--quiet button--small" href="/api/runs/${encodeURIComponent(runId)}/file?path=run_summary.md">판정 보고서 (.md)</a></div>`
      : "";
    content.innerHTML = `${substituteRun && substitutes?.status !== "blocked" ? renderSubstituteDecision(substitutes) : renderDecision(summary)}
      ${substituteRun ? "" : efficacySummaryHtml(top)}
      <div class="metric-grid">
        <div class="metric"><span class="metric-label">화합물</span><strong class="metric-value metric-value--small">${escapeHtml(compound.inchikey || compound.canonical_smiles || "-")}</strong></div>
        <div class="metric"><span class="metric-label">피부 반응 위험 점수</span><strong class="metric-value">${formatScore(safety.admet_metrics?.Skin_Reaction)}</strong></div>
        <div class="metric"><span class="metric-label">피부 독성</span><strong class="metric-value metric-value--small">${escapeHtml(toxicity.decision || "-")}</strong></div>
        <div class="metric"><span class="metric-label">표적 후보</span><strong class="metric-value">${formatCount(target.n_targets)}</strong></div>
        <div class="metric"><span class="metric-label">피부 맥락</span><strong class="metric-value metric-value--small">${escapeHtml(skinContextLabel(binding.skin_context_decision))}</strong></div>
      </div>
      <details class="surface interpretation-guide">
        <summary>결과 용어를 쉽게 설명해 주세요</summary>
        <dl class="glossary-list"><div><dt>PASS</dt><dd>필수 검증을 통과했다는 뜻입니다. 실제 효능이 입증됐다는 뜻은 아닙니다.</dd></div><div><dt>REVIEW</dt><dd>누락 근거나 피부 맥락을 사람이 확인해야 합니다. 내려받은 파일에는 <code>FLAG_HIGH</code>로 적혀 있는데 같은 판정입니다.</dd></div><div><dt>HALT</dt><dd>현재 결과를 효능이나 직접 결합 결론으로 사용하면 안 됩니다.</dd></div><div><dt>피부 반응 위험 점수</dt><dd>ADMET 모델의 피부 반응 예측값입니다. 판정과 함께 읽고 단독으로 안전성을 결론 내리지 마세요.</dd></div><div><dt>종합 예측 점수</dt><dd>같은 실행 안에서만 유효한 상대 순위입니다. 다른 실행과 절대값을 직접 비교하지 마세요. 승격된 레시피로 채점할 때는 <strong>근거 유사도와 다른 값</strong>입니다 — 가장 닮은 분자와의 거리뿐 아니라 그 측정이 얼마나 강했는지, 몇 개의 데이터베이스가 뒷받침하는지까지 함께 반영합니다. 순위를 재현하려면 이 열로 정렬하세요. 도킹·피부 점수는 순위에 반영되지 않는 주석입니다.</dd></div><div><dt>도킹 순위</dt><dd>여러 도킹·예측 방법이 매긴 순위를 하나로 합친 값입니다. 결합 실험값이 아닙니다. 유사도로만 순위를 매기는 기본 경로에서는 도킹이 순위에 반영되지 않아 종합 예측 점수와 같은 값이 됩니다.</dd></div><div><dt>조회 / 예측</dt><dd>“조회”는 입력 화합물과 지문이 같은 참조 리간드가 이 표적에서 이미 측정돼 있다는 뜻입니다. 즉 새로 예측한 것이 아니라 알려진 상호작용을 찾아온 것입니다. “예측”은 유사 화합물로부터 외삽한 결과입니다.</dd></div><div><dt>최근접 유사도</dt><dd>이 표적의 측정된 리간드 중 입력 화합물과 가장 가까운 것의 Tanimoto 값입니다. 1.0이면 조회입니다.</dd></div><div><dt>피부 점수</dt><dd>피부 발현과 피부 관련 근거를 반영한 가중치입니다. 데이터가 있는 축만으로 재정규화되므로, 실제로 기여한 축은 <code>skin_score.tsv.axes.json</code>에서 확인하세요.</dd></div><div><dt>연결된 피부 문헌</dt><dd>지식 그래프의 문헌 연결 수입니다. 일부 항목은 실제 집계가 아니라 큐레이션 seed 상수이며 (<code>n_papers_basis</code>로 구분), 검증된 PMID 수는 <code>n_papers_counted</code>입니다. 이 수를 문헌 근거의 강도로 읽지 마세요.</dd></div></dl>
      </details>
      <div class="result-grid">
        <div>
          <section class="surface result-section"><div class="surface-heading"><div><p class="eyebrow">COMPOUND</p><h2>입력과 안전성</h2></div></div><dl class="compact-list"><div><dt>입력 SMILES</dt><dd><code>${escapeHtml(compound.input_smiles || "-")}</code></dd></div><div><dt>표준화 SMILES</dt><dd><code>${escapeHtml(compound.canonical_smiles || "-")}</code></dd></div><div><dt>높은 ADMET 위험</dt><dd>${escapeHtml((admet.high_risk_endpoints || []).join(", ") || "없음")}</dd></div><div><dt>중간 ADMET 위험</dt><dd>${escapeHtml((admet.moderate_risk_endpoints || []).join(", ") || "없음")}</dd></div><div><dt>화장품·의약품 검사</dt><dd>${escapeHtml(cosmeticDecisionLabel(cosmetic.decision))} · 경고 ${formatCount(cosmetic.n_warnings)}건</dd></div><div><dt>구조 경고</dt><dd>${escapeHtml((toxicity.structural_alert_flags || []).join(", ") || "없음")}</dd></div></dl></section>
          <section class="surface result-section"><div class="surface-heading"><div><p class="eyebrow">TARGET EVIDENCE</p><h2>상위 표적 후보</h2></div><button class="button button--secondary button--small" data-view="explore">전체 탐색</button></div><div class="table-wrap"><table class="data-table"><thead><tr><th>순위</th><th>Gene / Protein</th><th title="이 표적의 측정된 리간드 중 입력과 가장 가까운 것의 Tanimoto. 회수율을 결정하는 값입니다.">근거 유사도</th><th>종합 예측</th><th>피부 관련성</th><th>계산 근거</th><th>연결 문헌</th><th><span class="sr-only">상세</span></th></tr></thead><tbody id="result-target-body">${targetRowsHtml(top, true)}</tbody></table></div></section>
        </div>
        <div>
          <section class="surface result-section"><div class="surface-heading"><div><p class="eyebrow">SKIN CONTEXT</p><h2>피부 맥락</h2></div></div><div class="status-row"><span>피부 발현 지원</span><span class="status-badge status-badge--${binding.skin_expression_supported ? "success" : "warning"}">${binding.skin_expression_supported ? "지원" : "부족"}</span></div><div class="status-row"><span>피부 효능 문헌</span><span class="status-badge status-badge--${binding.skin_efficacy_supported ? "success" : "warning"}">${binding.skin_efficacy_supported ? "지원" : "부족"}</span></div><div class="status-row"><span>상위 표적 맥락</span><span class="status-badge status-badge--${binding.top_target_skin_context_supported ? "success" : "warning"}">${binding.top_target_skin_context_supported ? "지원" : "검토"}</span></div><div class="status-row"><span>가장 피부 관련 표적</span><strong>${escapeHtml(binding.most_skin_relevant_target?.gene_symbol || binding.top_target_gene_symbol || "-")}</strong></div></section>
          <section class="surface result-section"><div class="surface-heading"><div><p class="eyebrow">LIMITATIONS</p><h2>해석 전 확인</h2></div></div><div class="limitations">${reasons.length ? `<ul class="reason-list">${reasons.slice(0, 7).map((reason) => `<li title="${escapeHtml(reason)}">${escapeHtml(reasonLabel(reason))}</li>`).join("")}</ul>` : "현재 요약에 추가 제한 사유가 없습니다."}<p style="margin:12px 0 0">결과는 계산 기반 표적 가설이며, 세포·조직 효능과 임상 안전성을 대체하지 않습니다.</p></div></section>
        </div>
      </div>
      ${renderSubstituteSection(substitutes)}
      <section class="surface"><div class="surface-heading"><div><p class="eyebrow">PROVENANCE</p><h2>검증과 산출물</h2></div><span class="muted">${escapeHtml(state.selectedRunId || "-")}</span></div>${downloadBar}<details class="artifact-details"><summary class="advanced-summary">개별 산출물 파일 ${artifactCount}개 <span class="muted">(원본 수치를 직접 볼 때만 필요합니다)</span></summary><div class="artifact-list">${artifactRows || `<span class="muted">artifact 정보가 없습니다.</span>`}</div></details></section>`;
    renderViewerSection(runId, viewerState);
    bindTargetRows($("#result-target-body"), top);
    [$("#substitute-query"), $("#substitute-tier"), $("#substitute-source")].filter(Boolean).forEach((element) => {
      element.addEventListener(element.tagName === "INPUT" ? "input" : "change", updateSubstituteTable);
    });
  }

  function bindTargetRows(container, rows) {
    if (!container) return;
    container.querySelectorAll(".target-row").forEach((row) => row.addEventListener("click", (event) => {
      const target = event.target.closest("[data-target-index]");
      const index = Number(target?.dataset.targetIndex ?? row.dataset.targetIndex);
      openTargetDrawer(rows[index]);
    }));
  }

  function openTargetDrawer(row) {
    if (!row) return;
    const drawer = $("#target-drawer");
    $("#drawer-title").textContent = row.gene_symbol || row.target_id || "표적 상세";
    const sources = Array.isArray(row.sources) ? row.sources : [];
    const efficacy = Array.isArray(row.efficacy) ? row.efficacy : [];
    const selfMatch = row.daina_is_self_match;
    const nearest = row.daina_max_tanimoto;
    let evidenceBlock;
    if (selfMatch === null || selfMatch === undefined || nearest === null || nearest === undefined) {
      evidenceBlock = "<p>이 실행에는 검색 근거 정보가 기록되지 않았습니다.</p>";
    } else {
      const reference = escapeHtml(row.daina_supporting_molecule_id || "-");
      const ligands = row.daina_known_ligand_count ?? "-";
      const kind = selfMatch
        ? `<strong>조회</strong> — 입력 화합물과 지문이 같은 참조 리간드(${reference})가 이 표적에서 이미 측정돼 있습니다. 새로 예측한 결과가 아닙니다.`
        : `<strong>예측</strong> — 가장 가까운 측정 리간드(${reference})로부터 외삽한 결과입니다.`;
      evidenceBlock = `<p>${kind}</p>${evidenceStrengthNote(row)}<div class="compact-list"><div><dt>최근접 유사도</dt><dd>${formatScore(nearest)}</dd></div><div><dt>참조 분자</dt><dd>${reference}</dd></div><div><dt>가장 가까운 근거의 측정값</dt><dd>${escapeHtml(evidenceStrengthLabel(row.daina_supporting_evidence))}</dd></div><div><dt>이 표적의 측정된 리간드 수</dt><dd>${escapeHtml(String(ligands))}</dd></div></div>`;
    }
    if (row.final_score_semantics === "ordinal_rank") {
      evidenceBlock = `<p>최종 점수는 순위 표시값입니다. 1.0도 결합 확률 100%나 높은 효능을 뜻하지 않습니다.</p>${evidenceBlock}`;
    }
    const failureModes = Array.isArray(row.failure_modes) ? row.failure_modes : [];
    const limitationBlock = failureModes.length
      ? `<div class="drawer-section"><h3>이 수용체의 알려진 한계</h3><p>이 표적에서는 도킹 점수가 낮게 나와도 결합이 약하다는 근거가 되지 않습니다. 검증 패널의 도킹 실패는 모두 화합물이 아니라 아래 한계에서 비롯됐습니다.</p><ul>${failureModes
          .map((mode) => `<li>${escapeHtml(failureModeLabel(mode.failure_mode))} — ${escapeHtml(mode.detail || "")}</li>`)
          .join("")}</ul></div>`
      : "";
    $("#drawer-content").innerHTML = `<div class="drawer-score-grid"><div class="drawer-score"><span>${row.final_score_semantics === 'ordinal_rank' ? '순위 표시값' : '순위 산정 점수'}</span><strong>${formatScore(row.final_score)}</strong></div><div class="drawer-score"><span>${row.daina_max_tanimoto != null ? '원본 검색 점수' : '도킹 융합 점수'}</span><strong>${formatScore(row.docking_rrf)}</strong></div><div class="drawer-score"><span>피부 관련성</span><strong>${formatScore(row.skin_score)}</strong></div></div><div class="compact-list"><div><dt>UniProt ID</dt><dd>${escapeHtml(row.target_id || "-")}</dd></div><div><dt>단백질 이름</dt><dd>${escapeHtml(row.protein_name || "-")}</dd></div><div><dt>피부 근거 단계</dt><dd>${escapeHtml(skinTierLabel(row.skin_tier))}</dd></div><div><dt>우세 HPA 세포 유형</dt><dd>${escapeHtml(row.cell_type_preferred || "unknown")}</dd></div><div><dt>점수를 낸 방법 수</dt><dd>${escapeHtml(row.source_count ?? "-")}</dd></div></div>${limitationBlock}<div class="drawer-section"><h3>이 표적이 나온 근거</h3>${evidenceBlock}</div><div class="drawer-section"><h3>계산 근거</h3><div class="source-chips">${sources.map((source) => `<span class="source-chip">${escapeHtml(source)}</span>`).join("") || "-"}</div></div><div class="drawer-section"><h3>연결된 피부 문헌</h3><p>${efficacy.map((item) => `<span title="${escapeHtml(item)}">${escapeHtml(efficacyLabel(item))}</span>`).join("<br>") || "연결된 문헌 정보 없음"}</p></div><div class="drawer-section"><p>우세 HPA 세포 유형은 발현 맥락이며 해당 세포에서 선택적으로 작용한다는 뜻이 아닙니다. 이 값은 실험 우선순위를 정하기 위한 계산 가설로, 직접 결합, 인과적 효능, 임상 안전성을 의미하지 않습니다.</p></div>`;
    if (typeof drawer.showModal === "function") drawer.showModal();
    else drawer.setAttribute("open", "open");
  }

  function renderExplorer() {
    const select = $("#explorer-run");
    const current = state.selectedRunId || "";
    const completedRuns = beginnerCompletedRuns();
    select.innerHTML = `<option value="">실행을 선택하세요</option>` + completedRuns.map((run) => `<option value="${escapeHtml(run.run_id)}">${escapeHtml(run.run_id)} · ${escapeHtml(run.decision || "결과")}</option>`).join("");
    if (current) select.value = current;
    [$("#compare-run-a"), $("#compare-run-b")].forEach((element) => { element.innerHTML = `<option value="">선택</option>` + completedRuns.map((run) => `<option value="${escapeHtml(run.run_id)}">${escapeHtml(run.run_id)}</option>`).join(""); });
    renderTargetTable();
  }

  function renderTargetTable() {
    const body = $("#target-table-body");
    if (!state.selectedRunId) {
      body.innerHTML = `<tr><td colspan="8" class="muted">실행을 선택하세요.</td></tr>`;
      $("#explorer-source").textContent = "ranking source: -";
      $("#explorer-total").textContent = "0개";
      return;
    }
    body.innerHTML = targetRowsHtml(state.targetRows);
    $("#explorer-source").textContent = `ranking source: ${state.targetSource || "-"}`;
    $("#explorer-total").textContent = `${formatCount(state.targetTotal)}개`;
    bindTargetRows(body, state.targetRows);
  }

  function renderSettings() {
    const status = state.status;
    if (!status) return;
    const system = status.system || {};
    const runtime = status.runtime || {};
    const gpu = status.gpu || {};
    const disk = status.disk || {};
    $("#settings-system").innerHTML = [["운영체제", system.distro || system.platform], ["Ubuntu 상태", system.ubuntu_ready ? "준비됨" : "준비 필요"], ["WSL", system.is_wsl ? "감지됨" : "아님"], ["Python", `${runtime.python_version || "-"} · ${runtime.python || "-"}`], ["micromamba", runtime.micromamba || "없음"], ["GPU", gpu.detail || "확인 필요"], ["여유 디스크", `${disk.free_gb ?? "-"} GB`], ["표적 데이터", beginnerText(status.stage0?.detail || "-")], ["확인 시각", status.checked_at || "-"]].map(([label, value]) => `<div class="setting-cell"><span>${escapeHtml(label)}</span><strong>${escapeHtml(beginnerText(value))}</strong></div>`).join("");
  }

  async function loadStatus(force = false) {
    state.status = await api(`/api/status${force ? "?force=1" : ""}`);
    state.setup = await api("/api/setup");
    renderTopbar();
    renderSetup();
    renderSettings();
    updateReview();
  }

  async function loadRuns() {
    const response = await api("/api/runs");
    state.runs = response.runs || [];
    renderRuns();
    renderExplorer();
    if (state.selectedRunId && state.runs.some((run) => run.run_id === state.selectedRunId)) await loadRun(state.selectedRunId, false);
  }

  async function loadRun(runId, move = true) {
    state.selectedRunId = runId;
    state.detail = await api(`/api/runs/${encodeURIComponent(runId)}`);
    // The five-second poll used to drop the reader's filter and sort back to
    // the default rows while the controls still showed their selection, so the
    // table silently stopped matching what the screen said it was showing.
    if (targetQueryIsActive()) {
      await loadTargets();
    } else {
      state.targetRows = state.detail.targets?.rows || [];
      state.targetTotal = state.detail.targets?.total || 0;
      state.targetSource = state.detail.targets?.source_path || null;
      // The server distinguishes "no targets" from "could not read the ranking".
      // Dropping this made an unreadable CSV look like an empty result.
      state.targetError = state.detail.targets?.error || null;
    }
    if (move) navigate(["completed", "verification_failed", "unverified"].includes(state.detail.run.status) ? "results" : "runs");
    else { renderResults(); renderExplorer(); renderRuns(); }
  }

  function targetQueryIsActive() {
    const query = $("#target-query");
    const tier = $("#target-tier");
    const sort = $("#target-sort");
    if (!query || !tier || !sort) return false;
    return Boolean(query.value.trim()) || tier.value !== "all" || sort.value !== "final_score";
  }

  async function loadTargets() {
    if (!state.selectedRunId) return;
    const params = new URLSearchParams({ q: $("#target-query").value, skin_tier: $("#target-tier").value, sort: $("#target-sort").value, limit: "100" });
    const response = await api(`/api/runs/${encodeURIComponent(state.selectedRunId)}/targets?${params.toString()}`);
    state.targetRows = response.rows || [];
    state.targetTotal = response.total || 0;
    state.targetSource = response.source_path || null;
    state.targetError = response.error || null;
    renderTargetTable();
  }

  function updateReview() {
    const input = state.inputType === "smiles" ? $("#smiles-input").value.trim() : ($("#sdf-file").files[0]?.name || "SDF 파일 없음");
    $("#review-input").textContent = input || "-";
    const preset = document.querySelector('input[name="preset"]:checked')?.value || "safety";
    const substituteRun = preset === "substitute";
    if (preset === "safety" || substituteRun) document.querySelector('input[name="mode"][value="fast"]').checked = true;
    const discoveryInput = $("#evidence-discovery");
    if (substituteRun) document.querySelector('input[name="evidence_mode"][value="evidence"]').checked = true;
    discoveryInput.disabled = substituteRun;
    discoveryInput.closest(".mode-card")?.classList.toggle("mode-card--locked", substituteRun);
    $$(".mode-card").forEach((card) => card.classList.toggle("is-selected", card.querySelector("input")?.checked));
    const mode = document.querySelector('input[name="mode"]:checked')?.value || "fast";
    const labels = { safety: "안전성 / 피부 반응", "target-id": "단백질 표적 예측", substitute: "Pharmacophore 대체소재", report: "전체 증거 보고서" };
    const outputs = { safety: "안전성 요약 + 검증 상태", "target-id": "표적 ranking + 피부 맥락", substitute: "대체 후보 ranking + 3D SDF + interactive report", report: "전체 증거 산출물" };
    $("#review-preset").textContent = labels[preset] || preset;
    $("#review-output").textContent = outputs[preset] || "분석 산출물";
    $("#mode-row").classList.toggle("is-hidden", preset === "safety" || substituteRun);
    $("#substitute-options").classList.toggle("is-hidden", !substituteRun);
    $("#evidence-mode-note").textContent = substituteRun
      ? "대체소재 발굴은 동일 타겟의 공개 활성 근거를 비교하므로 Evidence 모드로만 실행됩니다."
      : "Discovery는 아직 알려지지 않은 표적을 찾아보고 싶을 때 쓰는 모드입니다. 입력 성분과 같은 분자를 근거에서 빼야 하는데, 그 목록이 검증되지 않으면 실행을 시작하지 않습니다.";
    $("#review-claim-note").textContent = substituteRun
      ? "대체 후보는 모두 실험 전 가설입니다. 3D 표적 anchor 옵션도 후보 pose나 동등한 결합력, 효능, 안전성을 증명하지 않습니다."
      : "표적 순위는 실험 전 가설입니다. PASS라도 실험적 효능이나 결합을 증명하지 않습니다.";
    const ubuntuReady = state.status?.system?.ubuntu_ready === true;
    const capability = state.status?.analysis_readiness?.[analysisReadinessKey(preset, mode)] || {};
    const evidenceMode = document.querySelector('input[name="evidence_mode"]:checked')?.value || "evidence";
    const evidenceModeLabel = evidenceMode === "discovery" ? "Discovery" : "Evidence";
    const missing = Array.isArray(capability.missing) ? capability.missing : [];
    const targetId = $("#substitute-target-id").value.trim().toUpperCase();
    const maxCandidates = Number($("#substitute-max-candidates").value);
    const targetValid = !substituteRun || !targetId || UNIPROT_ACCESSION_RE.test(targetId);
    const countValid = !substituteRun || (Number.isInteger(maxCandidates) && maxCandidates >= 1 && maxCandidates <= 200);
    const configurationIssue = !targetValid
      ? "선택 타겟을 P14679와 같은 UniProt accession으로 입력하세요."
      : !countValid
        ? "후보 수를 1에서 200 사이의 정수로 입력하세요."
        : "";
    // An out-of-scope compound is refused by the server anyway; refusing it here
    // means the reader sees why before waiting for a rejection.
    const preview = state.preview;
    const previewBlocks = state.inputType === "smiles" && preview && preview.can_start === false;
    const canStart = ubuntuReady && capability.ready === true && !configurationIssue && !previewBlocks;
    const readinessBadge = $("#review-readiness");
    readinessBadge.textContent = !ubuntuReady ? "Ubuntu 준비 필요" : previewBlocks ? "입력 범위 밖" : configurationIssue ? "입력 확인" : canStart ? "분석 가능" : "준비 필요";
    readinessBadge.className = `status-badge status-badge--${canStart ? "success" : "warning"}`;
    const readinessText = canStart
      ? `필요한 프로그램과 데이터가 준비되었습니다. ${evidenceModeLabel} 모드로 실행됩니다.`
      : `첫 분석 때 자동 다운로드하지 않습니다. 먼저 준비하세요: ${beginnerText(missing.join(", ") || "기본 실행 환경")}`;
    $("#analysis-readiness-detail").textContent = previewBlocks
      ? (preview.message || "이 입력은 분석 범위 밖입니다.")
      : configurationIssue || readinessText;
    const submit = $("#start-analysis");
    submit.disabled = state.submitting || !canStart;
    submit.textContent = state.submitting ? "분석 시작 중" : "분석 시작";
  }

  // A wet-lab reader knows the ingredient name, not the SMILES. The index is
  // built offline from the cosmetic ingredient list, so this never leaves the
  // machine.
  let nameSearchTimer = null;
  // 같은 사전을 두 화면이 쓴다. 분석 화면과 대체소재 화면 각각에 이름 검색이
  // 있어야 하는데, 대체소재 화면에서 "이름은 저쪽 화면에서 찾으세요"라고 보내는
  // 것은 왕복 두 번이고 그 사이에 SMILES를 손으로 옮겨야 한다.
  const NAME_TARGETS = {
    analyze: { list: "#name-results", hint: "#name-search-hint", query: "#name-search" },
    alternatives: { list: "#alternatives-name-results", hint: "#alternatives-name-hint", query: "#alternatives-name" },
  };

  function scheduleNameSearch(query, where = "analyze") {
    if (nameSearchTimer) clearTimeout(nameSearchTimer);
    if (!query || query.trim().length < 2) {
      renderNameResults(null, where);
      return;
    }
    nameSearchTimer = setTimeout(() => runNameSearch(query.trim(), where), 250);
  }

  async function runNameSearch(query, where = "analyze") {
    const token = Symbol("name");
    state.nameToken = token;
    try {
      const result = await api(`/api/compound/search?q=${encodeURIComponent(query)}`);
      if (state.nameToken !== token) return;
      renderNameResults(result, where);
    } catch (error) {
      if (state.nameToken !== token) return;
      renderNameResults({ matches: [], available: false, error: error.message }, where);
    }
  }

  function renderNameResults(result, where = "analyze") {
    const target = NAME_TARGETS[where] || NAME_TARGETS.analyze;
    const list = $(target.list);
    const hint = $(target.hint);
    if (!list) return;
    if (!result) {
      list.classList.add("is-hidden");
      list.innerHTML = "";
      return;
    }
    if (result.available === false) {
      list.classList.add("is-hidden");
      hint.textContent = "이름 사전이 준비되지 않았습니다. scripts/build_compound_name_index.py 를 먼저 실행하세요.";
      return;
    }
    if (!result.matches.length) {
      list.classList.remove("is-hidden");
      list.innerHTML = `<li class="name-result name-result--empty">이 이름으로는 찾지 못했습니다. SMILES를 직접 입력하세요.</li>`;
      return;
    }
    list.classList.remove("is-hidden");
    list.innerHTML = result.matches.map((match) => `
      <li><button type="button" class="name-result" data-smiles="${escapeHtml(match.smiles)}" data-name="${escapeHtml(match.name)}">
        <strong>${escapeHtml(match.name)}</strong>${match.exact ? `<span class="name-result-tag">정확히 일치</span>` : ""}
        <code>${escapeHtml(match.smiles.length > 60 ? `${match.smiles.slice(0, 60)}…` : match.smiles)}</code>
        <small>${escapeHtml(match.source || "")}</small>
      </button></li>`).join("");
  }

  // A typo that still parses used to survive until the run finished. The
  // structure, the canonical form and the applicability verdict are shown
  // before anything starts.
  let previewTimer = null;
  function schedulePreview(smiles) {
    if (previewTimer) clearTimeout(previewTimer);
    if (!smiles) {
      state.preview = null;
      renderStructurePreview();
      return;
    }
    previewTimer = setTimeout(() => runPreview(smiles), 350);
  }

  async function runPreview(smiles) {
    const token = Symbol("preview");
    state.previewToken = token;
    try {
      const result = await api("/api/compound/preview", { method: "POST", body: JSON.stringify({ smiles }) });
      if (state.previewToken !== token) return;
      state.preview = result;
    } catch (error) {
      if (state.previewToken !== token) return;
      state.preview = { valid: false, verdict: "invalid", can_start: false, verdict_label: "구조를 읽지 못했습니다", message: error.message || String(error) };
    }
    renderStructurePreview();
    updateReview();
  }

  // 활성 핵심구조를 유지한 대체소재 후보.
  //
  // 표적 예측 경로가 "이 분자가 어디에 붙나"를 묻는다면, 이 화면은 그 반대를
  // 묻는다: 활성이 알려진 분자를 주고 그 핵심구조를 유지한 다른 원료를 고른다.
  // 과제 제목이 가리키는 기능이 이것이다.
  //
  // 정렬은 유사도가 아니라 핵심구조 유지 판정이 먼저다. Tanimoto 0.7은 곁사슬만
  // 닮았을 때도 나오고, 그런 분자는 대체소재가 아니다.
  const CORE_TONE = {
    identical: "info",
    contains: "success",
    retained: "success",
    partial: "warning",
    embedded: "warning",
    weak: "danger",
    different: "danger",
    unknown: "neutral",
  };
  const MEASURED_EVIDENCE = {
    at_or_above_threshold: ["success", "활성 문턱 이상으로 측정됨"],
    between_thresholds: ["warning", "문턱 사이 · 약하게 측정됨"],
    below_threshold: ["danger", "문턱 아래로 측정됨"],
    none: ["neutral", "기록은 있으나 쓸 수 있는 측정값 없음"],
    not_measured: ["neutral", "측정된 적 없음"],
    // 라이브러리를 못 연 것은 측정이 없다는 뜻이 아니다. 같은 칸에 같은 모양으로
    // 찍으면 그 구분이 사라진다.
    evidence_unavailable: ["neutral", "활성 측정 라이브러리를 열 수 없어 대조하지 못했습니다"],
  };
  const alternativesState = {
    ingredients: [], measured: [], query: null,
    pharmacophore: false, threeD: false, sortBy: "core",
    scoreAvailable: false, admetAvailable: false, scoreNote: "",
  };

  // 서버를 다시 부르지 않고 화면에서 다시 줄 세운다. 검색은 라이브러리 7천여
  // 종을 전수 대조하므로 몇 초가 걸리는데, 정렬 기준을 바꿀 때마다 그것을
  // 되풀이할 이유가 없다. 이미 받은 행에 있는 값만 쓴다.
  //
  // [키, 큰 값이 위인가]. 없는 값은 항상 아래로 보낸다 - 위로 올리면 재지
  // 않은 후보가 1위가 된다.
  const CLIENT_SORTS = {
    score: ["score_total", true],
    similarity: ["similarity", true],
    safety: ["score_safety", true],
    evidence: ["score_evidence", true],
    skin: ["Skin_Reaction", false],
    logp: ["logP", false],
  };

  function sortIngredientsClientSide(rows, mode) {
    const spec = CLIENT_SORTS[mode];
    if (!spec) return rows;
    const [key, descending] = spec;
    const withIndex = rows.map((row, index) => ({ row, index }));
    withIndex.sort((a, b) => {
      const av = a.row[key];
      const bv = b.row[key];
      const aMissing = av === null || av === undefined || Number.isNaN(Number(av));
      const bMissing = bv === null || bv === undefined || Number.isNaN(Number(bv));
      if (aMissing && bMissing) return a.index - b.index;
      if (aMissing) return 1;
      if (bMissing) return -1;
      const diff = Number(av) - Number(bv);
      if (diff !== 0) return descending ? -diff : diff;
      return a.index - b.index;          // 동점은 받은 순서를 지킨다
    });
    return withIndex.map((entry) => entry.row);
  }

  // CosIng의 IUPAC 명칭은 한 줄이 200자를 넘는다. 그대로 두면 표의 한 행이
  // 화면 높이의 절반을 먹는다.
  function shortName(name) {
    const first = String(name || "").split("\n")[0].trim().replace(/;\s*$/, "");
    return first.length > 58 ? `${first.slice(0, 58)}…` : first;
  }

  function pct(value) {
    // 재지 못한 값은 0%가 아니다. 0%는 "하나도 안 겹친다"는 판정이고, null 은
    // 판정을 못 했다는 뜻이라 화면에서 갈라 보여야 한다.
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
    return `${Math.round(Number(value) * 100)}%`;
  }

  // 서버가 보내는 상태 문자열로 표를 조회할 때는 상속된 이름을 피해야 한다.
  // status 가 "constructor" 나 "__proto__" 면 obj[status] 가 참인 값을 돌려주고,
  // 구조분해에서 TypeError 가 나면서 한 칸이 표 전체와 옆 표까지 지운다.
  function lookup(table, key, fallbackKey) {
    return Object.prototype.hasOwnProperty.call(table, key)
      ? table[key]
      : table[fallbackKey];
  }

  function coreBadge(grade, label, coverage, share, basis) {
    const tone = lookup(CORE_TONE, grade, "__absent__") || "neutral";
    // MCS가 시간 안에 끝나지 못했으면 그 값은 하한이지 판정이 아니다.
    const truncated = String(basis || "").startsWith("mcs_timeout");
    const hint = truncated ? " · 계산이 상한에서 끊겨 하한값입니다" : "";
    const title = `입력 분자의 ${pct(coverage)}가 이 후보 안에 남아 있고, 이 후보의 ${pct(share)}가 그 공통 구조입니다${hint}`;
    return `<span class="status-badge status-badge--${tone}" title="${escapeHtml(title)}">${escapeHtml(label || grade)}${truncated ? " *" : ""}</span>`;
  }

  function measuredEvidenceCell(evidence, activity, count, targets, shared, match) {
    const [tone, label] = lookup(MEASURED_EVIDENCE, evidence, "not_measured");
    if (evidence === "not_measured") {
      return `<span class="muted" title="${escapeHtml(label)}">기록 없음</span>`;
    }
    if (evidence === "evidence_unavailable") {
      return `<span class="muted" title="${escapeHtml(label)}">대조 못 함</span>`;
    }
    const value = typeof activity === "number" ? activity.toFixed(2) : "-";
    const list = (targets || []).slice(0, 3).join(", ");
    // 입력과 같은 단백질에서 측정된 기록이 있으면 그것이 이 표에서 가장 결정에
    // 가까운 한 줄이다. 다른 무엇보다 눈에 띄어야 한다.
    const same = (shared || []).length
      ? `<br><span class="status-badge status-badge--success" title="${escapeHtml(`입력 화합물과 이 후보가 모두 측정된 표적: ${(shared || []).join(", ")}`)}">입력과 같은 표적 ${shared.length}개</span>`
      : "";
    // 전체 InChIKey를 먼저 찾으므로 이 표시가 붙는 것은 자기 값이 인덱스에 없어
    // 연결성으로 내려간 경우뿐이다(51종 중 4종). 같은 분자의 측정값과 나란히
    // 놓으면서 그 사실을 적지 않으면 표가 틀린 말을 한다.
    const stereo = match === "connectivity"
      ? `<br><span class="muted" title="이 분자 자신의 측정 기록은 없습니다. InChIKey의 연결성 부분만 일치하는 분자(입체배치나 염 형태가 다름)의 값입니다.">연결성만 같은 분자의 값</span>`
      : "";
    return `<span class="status-badge status-badge--${tone}" title="${escapeHtml(label)}">pAct ${escapeHtml(value)}</span>`
      + `<br><span class="muted">표적 ${escapeHtml(String(count || 0))}개${list ? ` · ${escapeHtml(list)}` : ""}</span>${stereo}${same}`;
  }

  const KEPT_GRADES = ["identical", "contains", "retained"];

  // 파마코포어 열이 켜지면 표가 한 칸 넓어진다. colspan을 8로 고정해 두면 빈
  // 상태와 구분선이 조용히 어긋난다 - 첫 사용자가 보는 것이 바로 그 빈 상태다.
  function ingredientColumnCount() {
    return 8 + (alternativesState.scoreAvailable ? 2 : 0)
      + (alternativesState.admetAvailable ? 2 : 0)
      + (alternativesState.pharmacophore ? 1 : 0) + (alternativesState.threeD ? 1 : 0);
  }

  function ingredientRowsHtml(rows) {
    if (!rows.length) {
      return `<tr><td colspan="${ingredientColumnCount()}" class="muted">이 조건으로는 등재 원료 후보가 없습니다. "핵심구조가 유지된 것만"을 꺼 보세요.</td></tr>`;
    }
    // 구분선은 등급 순으로 줄을 세웠을 때만 뜻이 있다. 합친 순위에서는 유지 등급이
    // 섞여 들어오므로, 한 곳에 줄을 그으면 없는 경계를 있다고 말하는 것이 된다.
    if (alternativesState.sortBy !== "core") {
      return rows.map(ingredientRowHtml).join("");
    }
    const boundary = rows.findIndex((row) => !KEPT_GRADES.includes(row.core_grade));
    const divider = `<tr class="table-divider"><td colspan="${ingredientColumnCount()}">여기부터는 핵심구조가 유지되지 않은 것들입니다. 가까운 순서일 뿐, 대체소재 후보로 제안하는 것이 아닙니다.</td></tr>`;
    return rows.map((row, index) => {
      const lead = index === boundary && boundary > 0 ? divider : "";
      return lead + ingredientRowHtml(row);
    }).join("");
  }

  // 파마코포어를 켜지 않았으면 열 자체가 없다. 0%로 채우면 "특징이 하나도 안
  // 겹친다"로 읽히는데, 그것은 재 보지 않았다는 뜻이다.
  function pharmCell(row) {
    if (!alternativesState.pharmacophore) return "";
    if (row.pharm_similarity === null || row.pharm_similarity === undefined) {
      return `<td><span class="muted">재지 않음</span></td>`;
    }
    // 지문이 비어 잴 수 없었던 후보를 0.00 으로 그리면 "특징이 안 겹친다"로 읽힌다.
    if (row.pharm_usable === false) {
      return `<td><span class="muted" title="이 분자는 특징이 너무 많아 Gobbi 지문을 만들지 못했습니다(큰 유연 지질·펩타이드). 겹치지 않는다는 뜻이 아니라 재지 못했다는 뜻입니다.">잴 수 없음</span></td>`;
    }
    return `<td>${row.pharm_similarity.toFixed(2)}<br><span class="muted">특징 ${pct(row.pharm_recall)}</span></td>`;
  }

  // 3D는 상위 몇 건만 본다. 재채점하지 않은 행을 빈칸으로 두면 "3D에서 나빴다"로
  // 읽히는데, 실제로는 보지 않은 것이다. 두 상태를 다른 말로 적는다.
  const THREE_D_LABEL = {
    not_rescored: ["muted", "재채점 안 함", "3D는 상위 몇 건만 다시 봅니다. 이 행은 보지 않았습니다."],
    unavailable: ["muted", "판정 불가", "컨포머를 만들지 못했거나 공통 구조가 3원자 미만이라 정렬할 수 없습니다."],
    budget_exhausted: ["muted", "시간 상한", "이 요청의 3D 계산 시간을 다 써서 판정하지 못했습니다."],
    // 후보 쪽 실패가 아니라 입력 쪽 실패다. 하나로 뭉치면 화면이 "이 후보를
    // 판정하지 못했다"고 모든 행에서 말하는데, 실제로는 후보를 하나도 볼 수
    // 없었던 것이다.
    recall_only: ["muted", "회수율만",
      "특징이 3개 미만으로 맞아 RMSD 는 정의되지 않지만, 회수율은 실제로 잰 값입니다."],
    parent_unavailable: ["muted", "입력 분자 3D 실패",
      "입력 분자의 3D 컨포머를 만들지 못해 어떤 후보도 3D로 비교하지 못했습니다."],
  };

  function threeDCell(row) {
    if (!alternativesState.threeD) return "";
    const status = row.three_d_status || "not_rescored";
    if (status !== "ok") {
      const [tone, label, hint] = lookup(THREE_D_LABEL, status, "not_rescored");
      return `<td><span class="${tone}" title="${escapeHtml(hint)}">${escapeHtml(label)}</span></td>`;
    }
    const recall = typeof row.three_d_recall === "number" ? pct(row.three_d_recall) : "-";
    const rmsd = typeof row.three_d_rmsd === "number" ? `${row.three_d_rmsd.toFixed(2)} Å` : "-";
    return `<td>${escapeHtml(recall)}<br><span class="muted">RMSD ${escapeHtml(rmsd)}</span></td>`;
  }

  // 종합 점수와 축별 점수. `*_basis` 가 "imputed" 면 그 축에 쓸 값이 없어
  // 중앙값으로 채운 것이다. 같은 숫자로만 보여 주면 채운 값이 잰 값처럼 읽힌다.
  function axisText(value, basis) {
    if (value === null || value === undefined) return `<span class="muted">-</span>`;
    const shown = Number(value).toFixed(2);
    return basis === "imputed"
      ? `<span class="muted" title="이 축에 쓸 값이 없어 중앙값으로 채웠습니다. 낮게 잰 것이 아닙니다.">${shown}<sup>추정</sup></span>`
      : shown;
  }

  function scoreCells(row) {
    if (!alternativesState.scoreAvailable) return "";
    const total = row.score_total === null || row.score_total === undefined
      ? `<span class="muted">-</span>`
      : `<strong>${Number(row.score_total).toFixed(2)}</strong>`;
    return `<td>${total}</td>
      <td class="score-axes">${axisText(row.score_structure, row.score_structure_basis)}
        / ${axisText(row.score_evidence, row.score_evidence_basis)}
        / ${axisText(row.score_safety, row.score_safety_basis)}</td>`;
  }

  // ADMET 캐시가 없으면 열 자체를 내지 않는다. 0.00 으로 채우면 "위험 없음"
  // 으로 읽히는데, 그것은 재 보지 않았다는 뜻이다.
  function admetCells(row) {
    if (!alternativesState.admetAvailable) return "";
    const skin = row.Skin_Reaction === null || row.Skin_Reaction === undefined
      ? `<span class="muted">예측 없음</span>` : Number(row.Skin_Reaction).toFixed(2);
    const logp = row.logP === null || row.logP === undefined
      ? `<span class="muted">-</span>` : Number(row.logP).toFixed(2);
    return `<td>${skin}</td><td>${logp}</td>`;
  }

  // Mol* 인스턴스는 하나만 만든다. 후보를 눌러 볼 때마다 새로 만들면 GPU
  // 컨텍스트가 쌓여 몇 번 만에 화면이 멈춘다.
  let candidateViewer = null;
  let candidateViewerBusy = false;

  async function ensureCandidateViewer() {
    if (candidateViewer) return candidateViewer;
    const mount = $("#candidate-3d-mount");
    if (!mount || typeof molstar === "undefined") return null;
    candidateViewer = await molstar.Viewer.create(mount, {
      layoutIsExpanded: false,
      layoutShowControls: false,
      layoutShowSequence: false,
      layoutShowLog: false,
      viewportShowExpand: true,
      viewportShowSelectionMode: false,
    });
    return candidateViewer;
  }

  async function showCandidate3d(smiles, name) {
    const section = $("#candidate-3d");
    const note = $("#candidate-3d-note");
    if (!section) return;
    section.classList.remove("is-hidden");
    $("#candidate-3d-name").textContent = name || "후보";
    if (candidateViewerBusy) return;
    candidateViewerBusy = true;
    note.textContent = "3차원 좌표를 만드는 중입니다…";
    try {
      const result = await api("/api/compound/structure3d", {
        method: "POST", body: JSON.stringify({ smiles }),
      });
      if (!result || result.ok !== true) {
        note.textContent = `3차원 구조를 만들지 못했습니다: ${result?.reason || "알 수 없는 이유"}`;
        return;
      }
      const viewer = await ensureCandidateViewer();
      if (!viewer) {
        note.textContent = "3D 뷰어를 불러오지 못했습니다. 페이지를 새로 고쳐 보세요.";
        return;
      }
      await viewer.plugin.clear();
      await viewer.loadStructureFromData(result.data, "mol", false);
      // 힘장이 수렴하지 않았으면 그것도 적는다. 좌표는 쓸 만하지만 완전히
      // 이완된 배좌는 아니다.
      const relaxed = result.force_field === "none"
        ? "힘장 최적화를 하지 못했습니다"
        : `${result.force_field}${result.converged ? " 수렴" : " (반복 한도 도달, 미수렴)"}`;
      note.textContent = `${result.note} · 원자 ${result.atoms}개 · ${relaxed}`;
    } catch (error) {
      note.textContent = `3차원 구조를 만들지 못했습니다: ${error}`;
    } finally {
      candidateViewerBusy = false;
    }
  }

  // 화면의 번호와 스캔이 매긴 순위는 다른 것이다. 정렬을 종합 점수나 안전 순으로
  // 바꾸면 스캔 순위는 뒤죽박죽이 되고(실측: 안전 순 첫 줄이 핵심구조 1977위),
  // 미신고 물질을 걸러내면 구멍이 난다. 둘을 한 칸에 합치면 둘 다 틀린 값이 된다.
  // 표적 표가 쓰는 것과 같은 방식으로, 다를 때만 아래에 작게 덧붙인다.
  function ingredientRankCell(row) {
    const shown = row.rank;
    const scanned = row.original_rank;
    if (!scanned || scanned === shown) return escapeHtml(String(shown ?? "-"));
    return `${escapeHtml(String(shown))}<br><span class="muted" title="이 표의 정렬과 무관하게, 핵심구조 유지 순으로 줄 세웠을 때의 자리입니다.">핵심구조 순 ${escapeHtml(String(scanned))}위</span>`;
  }

  function ingredientRowHtml(row) {
    const functions = (row.functions || []).slice(0, 4).join(", ");
    const self = row.is_query ? ` <span class="status-badge status-badge--info">입력과 같음</span>` : "";
    // 질의에 고리가 없으면 비교할 골격 자체가 없다. 그때 모든 행에 "고리 골격은
    // 다름"을 붙이면 없는 차이를 있다고 말하는 것이 된다.
    const ring = alternativesState.query?.acyclic || row.scaffold_match
      ? "" : `<br><span class="muted">고리 골격은 다름</span>`;
    // 같은 구조가 INCI 명칭과 IUPAC 명칭으로 따로 등재된 경우가 흔하다. 구조로
    // 묶었으므로, 합쳐진 다른 이름은 여기서만 보여 준다.
    const synonyms = (row.synonyms || []).length
      ? `<br><span class="muted">다른 등재 명칭: ${escapeHtml(row.synonyms.map(shortName).slice(0, 2).join(" · "))}</span>`
      : "";
    return `<tr>
      <td>${ingredientRankCell(row)}</td>
      <td>${coreBadge(row.core_grade, row.core_label_ko, row.core_coverage, row.core_share, row.core_basis)}${ring}</td>
      <td>${pct(row.core_coverage)}<br><span class="muted">비중 ${pct(row.core_share)}</span></td>
      <td>${(Number(row.similarity) || 0).toFixed(3)}</td>
      ${pharmCell(row)}${threeDCell(row)}${scoreCells(row)}${admetCells(row)}
      <td><strong title="${escapeHtml(row.inci_name || "")}">${escapeHtml(shortName(row.inci_name) || "-")}</strong>${self}${synonyms}${row.cas ? `<br><span class="muted">CAS ${escapeHtml(row.cas)}</span>` : ""}</td>
      <td>${functions
        ? escapeHtml(functions)
        : `<span class="status-badge status-badge--warning" title="CosIng 에 항목은 있지만 화장품 배합목적이 신고돼 있지 않습니다. 금지·제한 물질이 규제 목적으로 등재된 경우가 여기 들어갑니다 - 화장품 원료로 검증된 것이 아닙니다.">배합목적 미신고</span>`}</td>
      <td>${measuredEvidenceCell(row.measured_evidence, row.measured_best_pactivity, row.measured_target_count, row.measured_top_targets, row.shared_targets, row.measured_match)}</td>
      <td><button type="button" class="button button--quiet button--small candidate-3d-open" data-smiles="${escapeHtml(row.smiles)}" data-name="${escapeHtml(row.inci_name || "")}" title="계산으로 만든 컨포머를 3차원으로 봅니다. 실험 구조나 결합 자세가 아닙니다.">3D</button>
      <button type="button" class="button button--quiet button--small alternatives-analyse" data-smiles="${escapeHtml(row.smiles)}" title="이 후보를 새 분석 화면으로 보냅니다">이 후보 분석</button></td>
    </tr>`;
  }

  function measuredRowsHtml(rows) {
    if (!rows.length) {
      return `<tr><td colspan="7" class="muted">이 문턱 이상으로 닮은 측정 분자가 없습니다.</td></tr>`;
    }
    return rows.map((row) => {
      const [tone, label] = lookup(MEASURED_EVIDENCE, row.evidence, "none");
      const activity = typeof row.best_pactivity === "number" ? row.best_pactivity.toFixed(2) : "-";
      const targets = (row.top_targets || []).join(", ");
      return `<tr>
        <td>${escapeHtml(String(row.rank))}</td>
        <td>${coreBadge(row.core_grade, row.core_label_ko, row.core_coverage, row.core_share, row.core_basis)}</td>
        <td>${pct(row.core_coverage)}<br><span class="muted">비중 ${pct(row.core_share)}</span></td>
        <td>${(Number(row.similarity) || 0).toFixed(3)}</td>
        <td><code>${escapeHtml(row.inchikey)}</code></td>
        <td>${escapeHtml(String(row.target_count))}${targets ? `<br><span class="muted">${escapeHtml(targets)}</span>` : ""}</td>
        <td><span class="status-badge status-badge--${tone}" title="${escapeHtml(label)}">${escapeHtml(activity)}</span></td>
      </tr>`;
    }).join("");
  }

  function renderAlternativesQuery(query) {
    const panel = $("#alternatives-query");
    if (!panel) return;
    if (!query) { panel.classList.add("is-hidden"); panel.innerHTML = ""; return; }
    // 비고리형 입력은 Murcko 골격이 없다. 그 사실을 숨기면 "고리 골격 다름"이
    // 왜 전부 비어 있는지 읽는 사람이 알 수 없다.
    const core = query.acyclic
      ? `<dt>핵심구조 기준</dt><dd>고리가 없는 분자라 분자 전체를 핵심으로 봅니다.</dd>`
      : `<dt>고리 골격</dt><dd><code>${escapeHtml(query.scaffold_smiles || "-")}</code></dd>`;
    const note = query.heavy_atom_note
      ? ` <span class="muted">${escapeHtml(query.heavy_atom_note)}</span>` : "";
    panel.innerHTML = `<dt>정규화된 구조</dt><dd><code>${escapeHtml(query.canonical_smiles)}</code></dd>`
      + `<dt>InChIKey</dt><dd><code>${escapeHtml(query.inchikey)}</code></dd>`
      + `<dt>중원자 수</dt><dd>${escapeHtml(String(query.heavy_atoms))}${note}</dd>` + core;
    panel.classList.remove("is-hidden");
  }

  let alternativesToken = null;

  function clearAlternatives() {
    alternativesState.ingredients = [];
    alternativesState.measured = [];
    alternativesState.query = null;
    renderAlternativesQuery(null);
    $("#alternatives-ingredients-surface")?.classList.add("is-hidden");
    $("#alternatives-measured-surface")?.classList.add("is-hidden");
    const csv = $("#alternatives-csv");
    if (csv) csv.disabled = true;
  }

  // 두 신호가 어긋난다는 사실 자체가 정보다. 나란히 놓기만 하면 서로를 뒷받침하는
  // 것처럼 읽히는데, 등재 원료 507종에서 두 순위의 Spearman은 질의에 따라 0.09까지
  // 내려간다. 이 저장소는 "네 방법이 점수를 냈다"를 "네 방법이 동의했다"로 쓴 적이
  // 있고, 같은 실수를 여기서 반복하지 않는다.
  //
  // 어긋남을 드러내는 것이 합친 점수보다 순위가 좋아서는 아니다 - 재 보면 합친
  // 쪽이 더 잘 줄 세운다(docs/ALTERNATIVE_CRITERIA_EVAL.md). 드러내는 이유는
  // 어긋남의 부호가 화합물 분류에 따라 뒤집히기 때문이다. 향료에서는 구조 기준이
  // 낫고 염모제에서는 파마코포어가 낫다. 숫자 하나만 주면 읽는 사람은 자기
  // 화합물이 어느 쪽인지 알 수 없다.
  function renderSignalAgreement(agreement) {
    const panel = $("#alternatives-disagreement");
    if (!panel) return;
    if (!agreement || agreement.available !== true) {
      panel.classList.add("is-hidden");
      panel.textContent = "";
      return;
    }
    const rho = agreement.spearman;
    const strength = rho === null || rho === undefined ? "재지 못했습니다"
      : rho < 0.3 ? "거의 관계가 없습니다"
      : rho < 0.6 ? "약하게만 같이 움직입니다"
      : "대체로 같이 움직입니다";
    const tops = agreement.same_top
      ? "두 기준의 1위는 같은 후보입니다."
      : `구조 1위는 ${agreement.top_structural}, 파마코포어 1위는 ${agreement.top_pharmacophore}로 다릅니다.`;
    panel.textContent = `이 질의에서 두 기준의 순위 상관은 ρ=${rho ?? "-"}로 ${strength}. ${tops} `
      + "원자를 유지하는 것과 상호작용 특징을 유지하는 것은 같은 일이 아니므로, 둘을 하나의 점수로 합치지 않습니다.";
    panel.classList.remove("is-hidden");
  }

  // caption 은 sr-only 라 눈으로는 안 보이지만 표의 접근성 이름이다. 정렬을 바꿔도
  // 옛 문장에 묶여 있으면, 스크린리더 사용자만 화면 위 정렬 안내와 정반대되는
  // 설명을 듣게 된다.
  const SORT_CAPTION = {
    core: "핵심구조 유지 순으로 정렬한 화장품 원료 후보",
    merged: "세 기준을 합친 순위로 정렬한 화장품 원료 후보",
    polarity: "극성이 가까운 순으로 정렬한 화장품 원료 후보",
    score: "종합 점수 순으로 정렬한 화장품 원료 후보",
    similarity: "구조 유사도(Tanimoto) 순으로 정렬한 화장품 원료 후보",
    safety: "안전 축 점수 순으로 정렬한 화장품 원료 후보",
    evidence: "활성 측정 근거 순으로 정렬한 화장품 원료 후보",
    skin: "피부반응 예측이 낮은 순으로 정렬한 화장품 원료 후보",
    logp: "logP가 낮은 순으로 정렬한 화장품 원료 후보",
  };

  // 각 정렬이 무엇을 가정하는지. 문장이 없으면 "모르는 기준"으로 나가고, 읽는
  // 사람은 무엇으로 줄 세운 표를 보고 있는지 알 수 없다.
  const SORT_NOTE = {
    score:
      "구조(55%)·근거(25%)·안전(20%) 세 축의 백분위 순위를 가중평균한 순서입니다. "
      + "확률이 아니라 줄 세우기이고, 가중치는 측정으로 보정한 값이 아니라 고른 "
      + "값입니다. 백분위는 대조한 후보 전체를 기준으로 매깁니다. 축 옆의 '추정'은 "
      + "그 축에 쓸 값이 없어 중앙값으로 채웠다는 뜻입니다.",
    similarity:
      "Morgan 지문의 Tanimoto 순서입니다. 곁사슬이 크게 달라지면 핵심 고리가 그대로여도 "
      + "0.2대로 떨어지므로, 이 순서만으로 대체 가능성을 판단하지 마세요.",
    safety:
      "ADMET-AI 예측(피부반응·AMES·hERG·DILI·발암)을 뒤집어 백분위 평균한 순서입니다. "
      + "예측이지 측정이 아니고, 예측값이 없는 후보는 중앙값으로 채워집니다.",
    evidence:
      "이 라이브러리에서 활성이 측정된 표적 수와 최고 활성값 순서입니다. 기록이 없다는 "
      + "것은 활성이 없다는 뜻이 아니라 이 라이브러리에서 못 찾았다는 뜻입니다.",
    skin:
      "ADMET-AI의 피부반응 예측이 낮은 순서입니다. 단일 예측 하나로 줄 세운 것이라 "
      + "구조도 활성 근거도 보지 않습니다. 예측값이 없는 후보는 맨 뒤로 갑니다.",
    logp:
      "예측 logP가 낮은(친수성이 큰) 순서입니다. 구조도 활성도 보지 않는 물성 하나이며, "
      + "제형 적합성 판단의 출발점이지 대체 가능성의 근거가 아닙니다.",
  };

  // 이 설치가 무엇을 갖추지 못했는지. 행마다 붙는 `추정` 배지는 그 행 이야기이고,
  // 축이 **통째로** 비었다는 것은 다른 사실이다 - 데이터 번들 없이 깐 설치가
  // 정확히 그 상태이고, 그때 종합 점수는 사실상 구조 점수 하나가 된다
  // (0.55×구조 + 0.225 로 눌려 0.225~0.775 사이에서만 움직인다).
  // 서버는 `analysis_readiness.alternatives` 로 무엇이 없는지 이미 알려 준다.
  // 예전에는 그 값을 아무도 읽지 않아서, 데이터가 없어도 실행 단추가 눌리고
  // 누른 뒤에야 글로 에러가 났다. 더 나쁜 것은 인덱스만 없는 경우로, 요청이
  // 성공하고 표까지 나오지만 근거 축이 통째로 추정값이다.
  function renderAlternativesReadiness() {
    const panel = $("#alternatives-axis-warning");
    if (!panel) return;
    const ready = state.status?.analysis_readiness?.alternatives;
    if (!ready || ready.status === "ready") return;   // 결과가 있으면 그쪽이 덮어쓴다
    const missing = Array.isArray(ready.missing) ? ready.missing : [];
    if (!missing.length) return;
    panel.innerHTML = "이 설치에 없는 데이터가 있어 결과가 반쪽입니다: <strong>"
      + missing.map(escapeHtml).join("</strong>, <strong>") + "</strong>."
      + "<br>담당자에게 데이터 번들을 받아 <code>python3 scripts/fetch_analog_bundle.py "
      + "--from &lt;번들&gt;</code> 로 채우세요.";
    panel.classList.remove("is-hidden");
  }

  function renderAxisWarning(ingredients) {
    const panel = $("#alternatives-axis-warning");
    if (!panel) return;
    const rows = ingredients?.axis_measured_rows;
    const notes = [];
    if (ingredients?.admet_available === false) {
      notes.push("ADMET 예측이 이 설치에 없어 <strong>안전 축이 전부 추정값</strong>입니다"
        + "(<code>data/cosing/admet_cache.parquet</code> 없음)");
    }
    if (ingredients?.evidence_available === false) {
      notes.push("활성 측정 라이브러리를 열지 못해 <strong>근거 축이 전부 추정값</strong>입니다"
        + "(<code>data/similarity_index_202609/</code> 없음)");
    }
    // 핵심구조를 유지한 후보가 하나도 없으면, 이 표는 "가장 덜 다른 것"이지
    // 대체 후보가 아니다. 그런데 종합 점수는 그대로 나온다 - 바쿠치올이 그 예로,
    // 유지후보 0건인데 상위 셋이 0.80 으로 뜨고 리날룰 계열이 올라온다.
    if (ingredients?.core_kept === 0) {
      notes.push("입력의 <strong>핵심구조를 유지한 후보가 한 건도 없습니다</strong>. "
        + "아래는 '가장 덜 다른 것'이지 대체 후보가 아닙니다 - 종합 점수가 높아도 "
        + "그렇습니다.");
    }
    if (ingredients?.undeclared_hidden > 0) {
      notes.push(`구조는 가깝지만 <strong>화장품 배합목적이 신고되지 않은 물질 `
        + `${ingredients.undeclared_hidden}건</strong>을 이 표에서 뺐습니다. `
        + `CosIng 에는 금지·제한 물질도 규제 목적으로 등재돼 있습니다. `
        + `보려면 '배합목적 미신고 물질도 보기'를 켜세요.`);
    }
    if (!notes.length && rows && rows.total) {
      // 갖춰진 설치에서도 근거는 대부분 비어 있다 - 등재 원료 7,484종 중 활성
      // 측정 기록이 있는 것은 1,035종(13.8%)뿐이다. 그것은 결함이 아니라 사실이고,
      // 기저율을 말해 두지 않으면 '추정' 배지가 이 도구의 흠으로 읽힌다.
      if (rows.evidence < rows.total) {
        notes.push(`이 표 ${rows.total}건 중 활성 측정 기록이 있는 것은 `
          + `<strong>${rows.evidence}건</strong>입니다. 나머지는 근거 축이 중앙값으로 `
          + `채워집니다 - 등재 원료 전체에서도 기록이 있는 것은 13.8%입니다.`);
      }
    }
    if (!notes.length) { panel.classList.add("is-hidden"); panel.innerHTML = ""; return; }
    panel.innerHTML = notes.join("<br>");
    panel.classList.remove("is-hidden");
  }

  function renderIngredientsCaption(sortBy) {
    const caption = $("#alternatives-ingredients-caption");
    if (!caption) return;
    caption.textContent = Object.prototype.hasOwnProperty.call(SORT_CAPTION, sortBy)
      ? SORT_CAPTION[sortBy]
      : `알 수 없는 기준(${sortBy})으로 정렬한 화장품 원료 후보`;
  }

  // 어느 정렬을 골랐는지가 곧 어떤 가정을 받아들였는지다. 화면이 그 말을 해야 한다.
  function renderSortNote(sortBy, pharmacophoreUnavailable) {
    const panel = $("#alternatives-sort-note");
    if (!panel) return;
    if (pharmacophoreUnavailable) {
      // 서버가 파마코포어 없이 되돌아온 경우. 순서가 요청한 것과 다르므로
      // 말하지 않으면 읽는 사람은 합친 순위를 보고 있다고 믿는다.
      panel.textContent =
        "파마코포어를 열지 못해 핵심구조 유지 순으로 보여 줍니다. "
        + `사유: ${pharmacophoreUnavailable}`;
      panel.classList.remove("is-hidden");
      return;
    }
    if (sortBy === "polarity") {
      panel.textContent =
        "극성(TPSA)이 가까운 순서입니다. 구조를 전혀 보지 않습니다. 이 순서가 이긴 곳은 "
        + "향료·방향 분류 하나뿐입니다(AUC 0.841로 세 구조 기준 전부를 이겼고, 그 분류에서는 "
        + "구조를 보는 편이 오히려 나빴습니다). 반대로 출처가 인용된 대체 쌍 정답표와 UV 필터"
        + "에서는 상위 4위 안에도 들지 못했습니다. 대체소재를 잘 찾는 것이 아니라 휘발성이 "
        + "비슷한 것을 잘 찾는 것이고, 향료 분류의 소속이 대체로 휘발성으로 정해지기 "
        + "때문입니다. 향료 소재를 다룰 때만 쓰세요.";
      panel.classList.remove("is-hidden");
      return;
    }
    if (sortBy === "core") { panel.classList.add("is-hidden"); panel.textContent = ""; return; }
    if (Object.prototype.hasOwnProperty.call(SORT_NOTE, sortBy)) {
      panel.textContent = SORT_NOTE[sortBy];
      panel.classList.remove("is-hidden");
      return;
    }
    if (sortBy !== "merged") {
      // 이 화면이 모르는 정렬 기준. 조용히 숨기면 기본 정렬로 읽히므로 밝힌다.
      panel.textContent =
        `이 화면이 모르는 정렬 기준입니다(${sortBy}). 순서를 설명할 수 없습니다.`;
      panel.classList.remove("is-hidden");
      return;
    }
    panel.textContent =
      "세 기준(구조 유지율·파마코포어·분자 유사도)의 백분위 순위를 평균한 순서입니다. "
      + "이 순서가 세 기준 각각보다 잘 줄 세운다는 측정이 있습니다(AUC 0.685 대 "
      + "0.657/0.645/0.666). 다만 그 측정은 등재 원료가 **507종이던 때** 낸 것이고, "
      + "지금 이 화면이 훑는 것은 7,484종입니다 - 모집단이 15배 커져 재평가에 며칠이 "
      + "걸리므로 아직 다시 재지 못했습니다. 순서를 고르는 근거로는 쓰되, 그 수치를 "
      + "지금 모집단의 성능으로 읽지 마세요. 그 정답표는 \u0027용도가 같다\u0027이지 "
      + "\u0027대체 가능하다\u0027가 아니고, 향료 분류에서는 세 기준 모두 크기·극성만 보는 기준선보다 "
      + "못했습니다. 합치면 어느 기준이 그 후보를 밀어 올렸는지도 보이지 않으므로, "
      + "옆의 세 열을 함께 보세요.";
    panel.classList.remove("is-hidden");
  }

  // 얼마나 걸릴지 미리 말해 준다. 이 화면은 라이브러리 7,484종 전부에 최대공통
  // 부분구조를 돌리는데, 비용이 고르지 않다 - 쌍당 중앙값은 0.14ms 인데 축합
  // 다환끼리는 500ms 를 넘는다. 실측으로 나이아신아마이드·살리실산류는 2초,
  // 우르솔산(펜타사이클릭 트리테르펜)은 153초다. 그 차이를 말하지 않으면 읽는
  // 사람은 화면이 멈춘 줄 안다.
  //
  // 고리 수는 SMILES 의 고리 닫힘 표시로 센다. 두 자리 이상은 `%nn` 이므로 그것을
  // 먼저 떼고, 남은 숫자를 센 뒤 2로 나눈다. 대괄호 안의 숫자(전하·동위원소)는
  // 고리 표시가 아니므로 함께 뗀다.
  function ringClosureCount(smiles) {
    const withoutBrackets = String(smiles || "").replace(/\[[^\]]*\]/g, "");
    const twoDigit = (withoutBrackets.match(/%\d\d/g) || []).length;
    const singleDigit = (withoutBrackets.replace(/%\d\d/g, "").match(/\d/g) || []).length;
    return Math.floor((twoDigit + singleDigit) / 2);
  }

  function searchingMessage(smiles) {
    const base = "찾는 중… 활성 측정 라이브러리를 처음 여는 요청은 몇 초 걸립니다.";
    return ringClosureCount(smiles) >= 4
      ? base + " 고리가 여럿 붙은 구조라 2~3분 걸릴 수 있습니다 - 라이브러리 전체를 끝까지 대조합니다."
      : base;
  }

  async function loadAlternatives() {
    const smiles = ($("#alternatives-smiles")?.value || "").trim();
    const status = $("#alternatives-status");
    const badge = $("#alternatives-badge");
    if (!smiles) {
      if (status) status.textContent = "출발 화합물의 SMILES를 입력하세요.";
      return;
    }
    // 앞선 요청이 늦게 도착해 새 결과를 덮어쓰는 것을 막는다. 인덱스를 처음 여는
    // 요청은 몇 초 걸리므로 실제로 일어난다.
    const token = Symbol("alternatives");
    alternativesToken = token;
    const runButton = $("#alternatives-run");
    if (runButton) runButton.disabled = true;
    if (status) status.textContent = searchingMessage(smiles);
    if (badge) { badge.textContent = "계산 중"; badge.className = "status-badge status-badge--neutral"; }
    try {
      const result = await api("/api/compound/alternatives", {
        method: "POST",
        body: JSON.stringify({
          smiles,
          limit: Number($("#alternatives-limit")?.value || 20),
          require_core: $("#alternatives-require-core")?.checked === true,
          exclude_self: $("#alternatives-exclude-self")?.checked === true,
          with_pharmacophore: $("#alternatives-pharmacophore")?.checked === true,
          with_three_d: $("#alternatives-three-d")?.checked === true,
          include_undeclared: $("#alternatives-include-undeclared")?.checked === true,
          sort_by: $("#alternatives-sort")?.value || "core",
        }),
      });
      if (alternativesToken !== token) return;
      alternativesState.query = result.query || null;
      alternativesState.pharmacophore = result.ingredients?.pharmacophore === true;
      alternativesState.sortBy = result.ingredients?.sort_by || "core";
      // 서버가 실제로 붙였는지 그대로 받는다. 붙지 않았는데 열을 띄우면
      // 빈 칸이 "위험 없음"으로 읽힌다.
      alternativesState.scoreAvailable = Boolean(result.ingredients?.score_available);
      alternativesState.admetAvailable = Boolean(result.ingredients?.admet_available);
      alternativesState.scoreNote = result.ingredients?.score_note || "";
      // 합친 순위는 파마코포어가 있어야 낼 수 있어서 서버가 켠다. 체크박스가 꺼진
      // 채로 열이 나타나면 사용자는 자기가 켠 적 없는 것을 보게 된다.
      const pharmBox = $("#alternatives-pharmacophore");
      if (pharmBox && alternativesState.pharmacophore) pharmBox.checked = true;
      alternativesState.threeD = result.ingredients?.three_d === true;
      $$("[data-pharm-col]").forEach((cell) => cell.classList.toggle("is-hidden", !alternativesState.pharmacophore));
      $$("[data-score-col]").forEach((cell) => cell.classList.toggle("is-hidden", !alternativesState.scoreAvailable));
      $$("[data-admet-col]").forEach((cell) => cell.classList.toggle("is-hidden", !alternativesState.admetAvailable));
      $$("[data-3d-col]").forEach((cell) => cell.classList.toggle("is-hidden", !alternativesState.threeD));
      alternativesState.ingredients = result.ingredients?.rows || [];
      alternativesState.measured = result.measured?.rows || [];
      renderAlternativesQuery(alternativesState.query);

      // 원료 라이브러리만 없을 수도 있다. 그때 빈 표를 내놓으면 "후보가 없다"로
      // 읽히므로, 이유를 그 자리에 적는다.
      const ingredientsUnavailable = result.ingredients?.unavailable;
      $("#alternatives-ingredients-body").innerHTML = ingredientsUnavailable
        ? `<tr><td colspan="${ingredientColumnCount()}" class="muted">${escapeHtml(ingredientsUnavailable)}</td></tr>`
        : ingredientRowsHtml(alternativesState.ingredients);
      $("#alternatives-ingredients-surface").classList.remove("is-hidden");
      $("#alternatives-library-note").textContent = result.ingredients?.note || "";
      // 서버가 라이브러리 전체를 대조하고 센 값이다. 화면에 온 상위 몇 줄만 세면
      // "유지 후보 0건"과 "상위 20줄 안에 없음"을 구분할 수 없다.
      const kept = Number(result.ingredients?.core_kept || 0);
      const scanned = Number(result.ingredients?.scanned || 0);
      const unjudged = Number(result.ingredients?.unjudged || 0);
      $("#alternatives-ingredients-count").textContent = ingredientsUnavailable
        ? "원료 라이브러리 없음"
        : `${scanned.toLocaleString("ko-KR")}종 대조${unjudged ? `(${unjudged}종 미판정)` : ""} · 핵심구조 유지 ${kept}건 · 아래 ${alternativesState.ingredients.length}건 표시`;
      $("#alternatives-summary").textContent = result.ingredients?.summary || "";
      renderSignalAgreement(result.ingredients?.signal_agreement);
      renderSortNote(alternativesState.sortBy,
                     result.ingredients?.pharmacophore_unavailable || null);
      renderIngredientsCaption(alternativesState.sortBy);
      renderAxisWarning(result.ingredients);
      // "같은 표적" 칸이 전부 비어 있는 이유가 둘이다: 겹치는 표적이 정말 없거나,
      // 입력 화합물 자체가 측정된 적이 없거나. 화면이 그것을 말하지 않으면 읽는
      // 사람은 앞쪽으로 읽는다.
      const evidenceNote = $("#alternatives-evidence-note");
      if (evidenceNote) {
        if (result.ingredients?.evidence_available === false) {
          evidenceNote.textContent = result.ingredients.evidence_unavailable_reason
            || "활성 측정 라이브러리를 열 수 없어 측정 기록을 대조하지 못했습니다.";
        } else if (result.ingredients?.query_measured === false) {
          evidenceNote.textContent = "입력한 화합물 자체에 활성 측정 기록이 없어, \u0027입력과 같은 표적\u0027 표시는 나오지 않습니다.";
        } else {
          evidenceNote.textContent = "";
        }
      }

      const measuredSurface = $("#alternatives-measured-surface");
      if (result.measured?.unavailable) {
        measuredSurface.classList.remove("is-hidden");
        $("#alternatives-measured-body").innerHTML = `<tr><td colspan="7" class="muted">${escapeHtml(result.measured.unavailable)}</td></tr>`;
        $("#alternatives-measured-count").textContent = "";
      } else {
        measuredSurface.classList.remove("is-hidden");
        $("#alternatives-measured-body").innerHTML = measuredRowsHtml(alternativesState.measured);
        $("#alternatives-measured-count").textContent = `${alternativesState.measured.length}건 · 측정 분자 ${(result.measured?.library_size || 0).toLocaleString("ko-KR")}개 중`;
      }

      $("#alternatives-csv").disabled = alternativesState.ingredients.length === 0 && alternativesState.measured.length === 0;
      if (status) status.textContent = "";
      if (badge) {
        badge.textContent = ingredientsUnavailable
          ? "원료 라이브러리 없음"
          : (kept > 0 ? `핵심구조 유지 후보 ${kept}건` : "핵심구조 유지 후보 없음");
        badge.className = `status-badge status-badge--${ingredientsUnavailable ? "danger" : (kept > 0 ? "success" : "warning")}`;
      }
    } catch (error) {
      if (alternativesToken !== token) return;
      // 옛 질의의 표를 남겨 두면 새 입력의 결과로 읽힌다.
      clearAlternatives();
      renderSignalAgreement(null);
      renderSortNote("core");
      if (status) status.textContent = error.message || String(error);
      if (badge) { badge.textContent = "찾지 못했습니다"; badge.className = "status-badge status-badge--danger"; }
    } finally {
      if (alternativesToken === token && runButton) runButton.disabled = false;
    }
  }

  function alternativesCsv() {
    const escape = (value) => {
      const text = String(value ?? "");
      return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
    };
    // 열 정의를 한 곳에 모은다. 헤더와 두 행 종류를 따로 적던 구조에서는 화면에
    // 열이 늘어도 파일이 따라오지 않았다 - 파마코포어·3D·합친점수가 통째로 빠져,
    // 내려받은 파일만 보고는 rank 가 어느 기준의 순위인지 알 수 없었다.
    // 열 하나에 두 가지를 쓰지 않는다. 예전에는 `functions` 칸에 CosIng 배합목적과
    // UniProt 접근번호가 행 종류에 따라 번갈아 들어갔다.
    const COLUMNS = [
      ["library", () => "cosing", () => "measured"],
      ["rank", (r) => r.rank, (r) => r.rank],
    ["structure_rank", (r) => r.original_rank, () => ""],
      ["name_or_key", (r) => r.inci_name, (r) => r.inchikey],
      ["cas", (r) => r.cas, () => ""],
      ["declared_functions", (r) => (r.functions || []).join(";"), () => ""],
      ["measured_targets", (r) => (r.measured_top_targets || []).join(";"),
        (r) => (r.top_targets || []).join(";")],
      ["smiles", (r) => r.smiles, (r) => r.smiles],
      ["similarity", (r) => r.similarity, (r) => r.similarity],
      ["core_grade", (r) => r.core_grade, (r) => r.core_grade],
      ["core_coverage", (r) => r.core_coverage, (r) => r.core_coverage],
      ["core_share", (r) => r.core_share, (r) => r.core_share],
      ["core_basis", (r) => r.core_basis, (r) => r.core_basis],
      ["scaffold_match", (r) => r.scaffold_match, (r) => r.scaffold_match],
      // 화면에 있는 두 번째 신호와 3D 재채점. 재는 데 요청당 최대 30초를 쓰고도
      // 파일로 나가지 못하면 그 계산은 화면 밖에서 사라진다.
      ["pharm_similarity", (r) => r.pharm_similarity ?? "", () => ""],
      ["pharm_recall", (r) => r.pharm_recall ?? "", () => ""],
      ["three_d_status", (r) => r.three_d_status ?? "", () => ""],
      ["three_d_recall", (r) => r.three_d_recall ?? "", () => ""],
      ["three_d_rmsd", (r) => r.three_d_rmsd ?? "", () => ""],
      ["merged_score", (r) => r.merged_score ?? "", () => ""],
      ["polarity_closeness", (r) => r.polarity_closeness ?? "", () => ""],
      ["measured_evidence", (r) => r.measured_evidence, (r) => r.evidence],
      ["measured_match", (r) => r.measured_match, () => ""],
      ["measured_best_pactivity", (r) => r.measured_best_pactivity ?? "",
        (r) => r.best_pactivity ?? ""],
      ["measured_target_count", (r) => r.measured_target_count, (r) => r.target_count],
      ["shared_targets_with_query", (r) => (r.shared_targets || []).join(";"), () => ""],
    ];
    const lines = [];
    // 어느 질의를 어느 기준으로 세운 순위인지 파일 안에서 읽혀야 한다. 정렬만
    // 다르게 두 번 내려받으면 열 값은 같고 행 순서만 다른 두 파일이 된다.
    lines.push(["# query_smiles", alternativesState.query?.canonical_smiles
      || alternativesState.query?.smiles || ""].map(escape).join(","));
    lines.push(["# sort_by", alternativesState.sortBy || "core"].map(escape).join(","));
    lines.push(COLUMNS.map((c) => c[0]).join(","));
    alternativesState.ingredients.forEach((row) =>
      lines.push(COLUMNS.map((c) => escape(c[1](row))).join(",")));
    alternativesState.measured.forEach((row) =>
      lines.push(COLUMNS.map((c) => escape(c[2](row))).join(",")));
    return lines.join("\n");
  }

  function renderStructurePreview() {
    const panel = $("#structure-preview");
    const preview = state.preview;
    if (!panel) return;
    if (!preview || state.inputType !== "smiles") {
      panel.classList.add("is-hidden");
      return;
    }
    panel.classList.remove("is-hidden");
    $("#structure-preview-art").innerHTML = preview.svg || `<span class="muted">구조를 그릴 수 없습니다.</span>`;
    const tone = preview.verdict === "in_scope" ? "success" : preview.verdict === "review" ? "warning" : "danger";
    const badge = $("#preview-verdict");
    badge.textContent = preview.verdict_label || "-";
    badge.className = `status-badge status-badge--${tone}`;
    $("#preview-formula").textContent = preview.formula || "";
    $("#preview-identity").innerHTML = preview.valid
      ? `<dt>정규 SMILES</dt><dd><code>${escapeHtml(preview.canonical_smiles || "-")}</code></dd><dt>InChIKey</dt><dd><code>${escapeHtml(preview.inchikey || "-")}</code></dd>`
      : "";
    const exclusions = Array.isArray(preview.exclusions) ? preview.exclusions : [];
    const warnings = Array.isArray(preview.warnings) ? preview.warnings : [];
    const notes = [
      ...exclusions.map((item) => `<p class="preview-note preview-note--block">${escapeHtml(item.detail)}<small>${escapeHtml(item.basis || "")}</small></p>`),
      ...warnings.map((item) => `<p class="preview-note">${escapeHtml(item.detail)}<small>${escapeHtml(item.basis || "")}</small></p>`),
    ];
    if (!preview.valid && preview.message) notes.unshift(`<p class="preview-note preview-note--block">${escapeHtml(preview.message)}</p>`);
    $("#preview-notes").innerHTML = notes.join("");
    const properties = Array.isArray(preview.properties) ? preview.properties : [];
    $("#preview-properties-wrap").classList.toggle("is-hidden", properties.length === 0);
    $("#preview-properties").innerHTML = properties
      .map((item) => `<dt>${escapeHtml(item.label)}</dt><dd>${escapeHtml(formatScore(item.value))}</dd>`)
      .join("");
  }

  // The editor is a GWT bundle that compiles at load time and therefore needs
  // eval; keeping it in its own frame means this page never relaxes script-src.
  function mountSketcher() {
    const mount = $("#sketcher-mount");
    if (!mount || mount.firstElementChild) return;
    const frame = document.createElement("iframe");
    frame.className = "sketcher-frame";
    frame.title = "분자 구조 편집기";
    frame.src = "/assets/vendor/jsme/editor.html";
    mount.appendChild(frame);
  }

  window.addEventListener("message", (event) => {
    if (event.origin !== window.location.origin) return;
    const data = event.data;
    if (!data || data.source !== "skinscout-sketcher" || typeof data.smiles !== "string") return;
    // A drawn structure goes through exactly the same validation a typed one
    // does, so the preview and the applicability gate still apply.
    setInputType("smiles");
    $("#smiles-input").value = data.smiles;
    $("#smiles-input").dispatchEvent(new Event("input"));
    showToast("그린 구조를 가져왔습니다.");
  });

  const BATCH_ITEM_LABELS = {
    queued: "대기", running: "실행 중", completed: "완료",
    failed: "실패", skipped: "건너뜀", cancelled: "취소됨",
  };

  function renderBatchStatus(batch) {
    const panel = $("#batch-status");
    if (!panel) return;
    if (!batch) {
      panel.classList.add("is-hidden");
      panel.innerHTML = "";
      return;
    }
    panel.classList.remove("is-hidden");
    const rows = batch.items.map((item) => {
      const tone = item.status === "completed" ? "success"
        : item.status === "running" ? "info"
        : item.status === "failed" ? "danger" : "warning";
      const link = item.run_id
        ? `<button class="button button--quiet button--small open-run" data-run-id="${escapeHtml(item.run_id)}">결과</button>`
        : "";
      return `<tr><td>${escapeHtml(item.label)}</td>
        <td><span class="status-badge status-badge--${tone}">${escapeHtml(BATCH_ITEM_LABELS[item.status] || item.status)}</span></td>
        <td class="muted">${escapeHtml(item.error || "")}</td>
        <td>${link}</td></tr>`;
    }).join("");
    panel.innerHTML = `<div class="batch-head"><strong>${escapeHtml(batch.batch_id)}</strong>
      <span class="muted">${batch.finished} / ${batch.total} 완료</span>
      ${batch.status === "running" ? `<button class="button button--quiet button--small" id="batch-cancel" data-batch-id="${escapeHtml(batch.batch_id)}">전체 취소</button>` : ""}</div>
      <table class="batch-table"><tbody>${rows}</tbody></table>`;
  }

  async function refreshBatch() {
    if (!state.batchId) return;
    try {
      const response = await api(`/api/batches/${encodeURIComponent(state.batchId)}`);
      renderBatchStatus(response.batch);
      if (response.batch.status !== "running") state.batchId = null;
    } catch (_) { /* the batch may have been dropped by a restart */ }
  }

  function setInputType(type) {
    state.inputType = type;
    $$("[data-input-type]").forEach((button) => {
      const active = button.dataset.inputType === type;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    $("#smiles-input-panel").classList.toggle("is-hidden", type !== "smiles");
    $("#sdf-input-panel").classList.toggle("is-hidden", type !== "sdf");
    $("#batch-input-panel").classList.toggle("is-hidden", type !== "batch");
    $("#draw-input-panel").classList.toggle("is-hidden", type !== "draw");
    if (type === "draw") mountSketcher();
    renderStructurePreview();
    updateReview();
  }

  async function submitAnalysis(event) {
    event.preventDefault();
    if (state.submitting) return;
    const error = $("#analysis-error");
    error.classList.add("is-hidden");
    const preset = document.querySelector('input[name="preset"]:checked')?.value || "safety";
    const mode = document.querySelector('input[name="mode"]:checked')?.value || "fast";
    const evidenceMode = document.querySelector('input[name="evidence_mode"]:checked')?.value || "evidence";
    const payload = { input_type: state.inputType, preset, mode, evidence_mode: evidenceMode, cores: 4 };
    if (preset === "substitute") {
      payload.target_id = $("#substitute-target-id").value.trim().toUpperCase();
      payload.max_candidates = Number($("#substitute-max-candidates").value);
      payload.target_conditioned = $("#substitute-target-conditioned").checked;
    }
    if (state.inputType === "smiles") payload.smiles = $("#smiles-input").value.trim();
    else if (state.inputType === "batch") payload.compounds = $("#batch-input").value;
    else payload.sdf_content = state.sdfContent;
    state.submitting = true;
    updateReview();
    try {
      if (state.inputType === "batch") {
        const response = await api("/api/batches", { method: "POST", body: JSON.stringify(payload) });
        state.batchId = response.batch.batch_id;
        renderBatchStatus(response.batch);
        showToast(`${response.batch.total}개 화합물을 순서대로 실행합니다.`);
        await loadRuns();
        return;
      }
      const response = await api("/api/runs", { method: "POST", body: JSON.stringify(payload) });
      state.selectedRunId = response.job.run_id;
      showToast("분석을 시작했습니다.");
      await loadRuns();
      navigate("runs");
    } catch (requestError) {
      error.textContent = requestError.message;
      error.classList.remove("is-hidden");
    } finally {
      state.submitting = false;
      updateReview();
    }
  }

  function selectRadio(name, value, fallback) {
    const preferred = document.querySelector(`input[name="${name}"][value="${value}"]:not(:disabled)`);
    const input = preferred || document.querySelector(`input[name="${name}"][value="${fallback}"]:not(:disabled)`);
    if (input) {
      input.checked = true;
      input.dispatchEvent(new Event("change", { bubbles: true }));
    }
  }

  function recoverableInput(detail, fallbackRun) {
    const summary = detail?.summary || {};
    const compound = summary.compound || {};
    const run = detail?.run || fallbackRun || {};
    return compound.input_smiles || run.input_smiles || compound.canonical_smiles || run.canonical_smiles || "";
  }

  async function recoverRun(runId) {
    const fallbackRun = state.runs.find((run) => run.run_id === runId) || {};
    let detail = null;
    try {
      detail = await api(`/api/runs/${encodeURIComponent(runId)}`);
    } catch (_) {
      detail = null;
    }
    const run = detail?.run || fallbackRun;
    const summary = detail?.summary || {};
    const input = recoverableInput(detail, fallbackRun);
    const preset = run.preset || summary.preset || "safety";
    const mode = run.mode || summary.mode || "fast";
    const evidenceMode = run.evidence_mode || summary.evidence_mode || "evidence";
    selectRadio("preset", preset, "safety");
    selectRadio("mode", mode, "fast");
    selectRadio("evidence_mode", evidenceMode, "evidence");
    if (preset === "substitute") {
      const report = detail?.substitutes || {};
      $("#substitute-target-id").value = report.target?.selection_basis === "user_selected_ranked_target"
        ? report.target?.target_id || ""
        : "";
      $("#substitute-max-candidates").value = String(report.thresholds?.max_candidates || 50);
      $("#substitute-target-conditioned").checked = report.target?.interaction_anchor_conditioned !== false;
    }
    if (input) {
      setInputType("smiles");
      $("#smiles-input").value = input;
      $("#smiles-input").dispatchEvent(new Event("input", { bubbles: true }));
    }
    navigate("analyze");
    showToast(input ? "가능한 입력과 실행 설정을 복원했습니다." : "복원 가능한 입력이 없어 새 분석 입력 화면을 열었습니다.");
  }

  async function startSetup(path, payload = {}) {
    try {
      const response = await api(path, { method: "POST", body: JSON.stringify(payload) });
      showToast("작업을 시작했습니다. 로그를 확인할 수 있습니다.");
      renderJob(response.job);
      pollJob(response.job.job_id);
    } catch (error) { showToast(error.message); }
  }

  function renderJob(job) {
    const panel = $("#setup-job-panel");
    panel.classList.remove("is-hidden");
    const phaseText = job.kind === "setup"
      ? "설치 단계는 재실행해도 이미 끝난 항목을 재사용합니다. 실패하면 로그의 마지막 원인을 해결한 뒤 같은 프로필로 다시 실행하세요."
      : "표적 데이터 준비는 큰 다운로드와 변환을 포함합니다. 실패하면 완료된 파일을 재사용해 이어서 실행합니다.";
    panel.innerHTML = `<div class="status-row"><strong>${escapeHtml(job.kind === "setup" ? "환경 준비" : "표적 데이터 준비")}</strong><span class="status-badge status-badge--${statusTone(job.status)}">${escapeHtml(statusLabel(job.status))}</span></div><p class="muted">${escapeHtml(phaseText)}</p><pre id="setup-job-log">${escapeHtml(beginnerText(job.log_tail || job.detail || "작업을 시작하고 있습니다."))}</pre>`;
  }

  async function pollJob(jobId) {
    try {
      const job = await api(`/api/jobs/${encodeURIComponent(jobId)}`);
      renderJob(job);
      if (["queued", "running"].includes(job.status)) {
        setTimeout(() => pollJob(jobId), 1800);
      } else {
        await loadStatus(true);
        await loadRuns();
        showToast(job.status === "completed" ? "작업이 완료되었습니다." : "작업이 종료되었습니다. 로그를 확인하세요.");
      }
    } catch (error) { showToast(error.message); }
  }

  async function compareRuns() {
    const a = $("#compare-run-a").value;
    const b = $("#compare-run-b").value;
    const target = $("#compare-result");
    if (!a || !b) { target.innerHTML = `<span class="muted">두 실행을 선택하세요.</span>`; return; }
    try {
      const [left, right] = await Promise.all([api(`/api/runs/${encodeURIComponent(a)}`), api(`/api/runs/${encodeURIComponent(b)}`)]);
      const rows = [["판정", decisionInfo(left.run.decision, left.run.recommended_action, left.run.claimable).label, decisionInfo(right.run.decision, right.run.recommended_action, right.run.claimable).label], ["과학적 주장", left.run.claimable ? "검증 통과" : "불가", right.run.claimable ? "검증 통과" : "불가"], ["상위 표적", left.run.top_target?.gene_symbol || "-", right.run.top_target?.gene_symbol || "-"], ["피부 맥락", left.run.skin_context_supported ? "지원" : "검토 필요", right.run.skin_context_supported ? "지원" : "검토 필요"]];
      target.innerHTML = `<table class="compare-table"><thead><tr><th>항목</th><th>${escapeHtml(a)}</th><th>${escapeHtml(b)}</th></tr></thead><tbody>${rows.map((row) => `<tr><th>${escapeHtml(row[0])}</th><td>${escapeHtml(row[1])}</td><td>${escapeHtml(row[2])}</td></tr>`).join("")}</tbody></table>`;
    } catch (error) { target.textContent = error.message; }
  }

  function bindEvents() {
    document.addEventListener("click", async (event) => {
      const example = event.target.closest(".alternatives-example[data-smiles]");
      if (example) {
        if ($("#alternatives-smiles")) $("#alternatives-smiles").value = example.dataset.smiles;
        loadAlternatives();
        return;
      }
      // 후보를 골라 바로 표적 예측으로 넘기는 고리. 이것이 없으면 사용자가
      // SMILES를 손으로 옮겨 적어야 하고, 그 자리에서 오타가 난다.
      const open3d = event.target.closest(".candidate-3d-open[data-smiles]");
      if (open3d) {
        showCandidate3d(open3d.dataset.smiles, open3d.dataset.name);
        return;
      }
      const analyse = event.target.closest(".alternatives-analyse[data-smiles]");
      if (analyse) {
        // SDF·그리기·여러 개 탭이 열려 있으면 SMILES 칸은 숨어 있다. 값만 넣고
        // 화면을 넘기면 사용자는 빈 화면을 보게 되므로, 입력 형식부터 되돌린다.
        setInputType("smiles");
        const input = $("#smiles-input");
        if (input) {
          input.value = analyse.dataset.smiles;
          input.dispatchEvent(new Event("input", { bubbles: true }));
        }
        navigate("analyze");
        showToast("후보를 새 분석 입력에 넣었습니다.");
        return;
      }
      const viewButton = event.target.closest("[data-view]");
      if (viewButton) { navigate(viewButton.dataset.view); return; }
      const batchCancel = event.target.closest("#batch-cancel[data-batch-id]");
      if (batchCancel) {
        try {
          const response = await api(`/api/batches/${encodeURIComponent(batchCancel.dataset.batchId)}/cancel`, { method: "POST", body: "{}" });
          renderBatchStatus(response.batch);
          showToast("배치를 취소했습니다.");
        } catch (error) { showToast(error.message); }
        return;
      }
      const inputButton = event.target.closest("[data-input-type]");
      if (inputButton) { setInputType(inputButton.dataset.inputType); return; }
      const open = event.target.closest(".open-run");
      if (open) { await loadRun(open.dataset.runId); return; }
      const cancel = event.target.closest(".cancel-run");
      if (cancel) { try { await api(`/api/runs/${encodeURIComponent(cancel.dataset.runId)}/cancel`, { method: "POST", body: "{}" }); showToast("취소 요청을 보냈습니다."); await loadRuns(); } catch (error) { showToast(error.message); } return; }
      const recover = event.target.closest(".recover-run");
      if (recover) { await recoverRun(recover.dataset.runId); return; }
      const copy = event.target.closest(".copy-command");
      if (copy) { await navigator.clipboard?.writeText(copy.dataset.command); showToast("명령을 복사했습니다."); return; }
      if (event.target.id === "close-drawer") { $("#target-drawer").close(); return; }
    });
    $("#analysis-form").addEventListener("submit", submitAnalysis);
    $("#smiles-input").addEventListener("input", () => {
      const value = $("#smiles-input").value.trim();
      $("#preview-value").textContent = value || "아직 입력되지 않았습니다.";
      $("#smiles-validation").textContent = value ? "구조를 확인하는 중입니다." : "입력 후 화학 구조를 확인합니다.";
      schedulePreview(value);
      $("#similar-jump")?.classList.toggle("is-hidden", !value);
      updateReview();
    });
    $("#similar-jump-button")?.addEventListener("click", () => {
      const value = ($("#smiles-input")?.value || "").trim();
      if ($("#alternatives-smiles")) $("#alternatives-smiles").value = value;
      navigate("alternatives");
      if (value) loadAlternatives();
    });
    $("#alternatives-run")?.addEventListener("click", loadAlternatives);
    $("#alternatives-smiles")?.addEventListener("keydown", (event) => {
      if (event.key === "Enter") { event.preventDefault(); loadAlternatives(); }
    });
    $("#alternatives-csv")?.addEventListener("click", () => {
      // 한글·기호가 섞인 INCI 명칭이 Excel에서 깨지지 않도록 BOM을 앞에 둔다.
      const blob = new Blob(["\uFEFF", alternativesCsv()], { type: "text/csv;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "skinscout_alternatives.csv";
      document.body.appendChild(link);
      link.click();
      link.remove();
      // 같은 틱에서 회수하면 브라우저가 저장을 시작하기 전에 URL이 사라진다.
      setTimeout(() => URL.revokeObjectURL(url), 10_000);
    });
    $("#name-search").addEventListener("input", (event) => scheduleNameSearch(event.target.value, "analyze"));
    $("#name-results").addEventListener("click", (event) => {
      const button = event.target.closest(".name-result[data-smiles]");
      if (!button) return;
      $("#smiles-input").value = button.dataset.smiles;
      $("#smiles-input").dispatchEvent(new Event("input"));
      $("#name-search").value = button.dataset.name;
      renderNameResults(null, "analyze");
      showToast(`${button.dataset.name} 구조를 넣었습니다.`);
    });
    $("#alternatives-name")?.addEventListener("input", (event) => scheduleNameSearch(event.target.value, "alternatives"));
    $("#alternatives-name-results")?.addEventListener("click", (event) => {
      const button = event.target.closest(".name-result[data-smiles]");
      if (!button) return;
      if ($("#alternatives-smiles")) $("#alternatives-smiles").value = button.dataset.smiles;
      $("#alternatives-name").value = button.dataset.name;
      renderNameResults(null, "alternatives");
      loadAlternatives();
    });
    $("#use-example").addEventListener("click", () => { $("#smiles-input").value = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"; $("#smiles-input").dispatchEvent(new Event("input")); });
    $("#sdf-file").addEventListener("change", async () => {
      const file = $("#sdf-file").files[0];
      if (!file) return;
      if (file.size > MAX_SDF_BYTES) {
        state.sdfContent = "";
        $("#sdf-file").value = "";
        $("#preview-value").textContent = "아직 입력되지 않았습니다.";
        $("#sdf-validation").textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(1)} MB · 10 MiB 이하 파일만 사용할 수 있습니다.`;
        $("#sdf-hash").textContent = "파일 크기 초과로 읽지 않음";
        updateReview();
        return;
      }
      $("#sdf-validation").textContent = `${file.name} · ${(file.size / 1024).toFixed(1)} KB · 읽는 중`;
      $("#sdf-hash").textContent = "SHA256 계산 중";
      state.sdfContent = await file.text();
      const digest = await sha256Hex(file);
      $("#preview-value").textContent = file.name;
      $("#sdf-validation").textContent = `${file.name} · ${(file.size / 1024).toFixed(1)} KB`;
      $("#sdf-hash").textContent = digest ? `SHA256 ${digest}` : "SHA256: 이 브라우저에서 계산할 수 없음";
      updateReview();
    });
    $$('input[name="preset"]').forEach((input) => input.addEventListener("change", () => { $$(".choice-card").forEach((card) => card.classList.toggle("is-selected", card.querySelector("input")?.checked)); updateReview(); }));
    $$('input[name="mode"]').forEach((input) => input.addEventListener("change", updateReview));
    [$("#substitute-target-id"), $("#substitute-max-candidates")].forEach((input) => input.addEventListener("input", updateReview));
    $("#candidate-3d-close")?.addEventListener("click", () => {
      $("#candidate-3d")?.classList.add("is-hidden");
    });
    // 정렬을 바꾸면 **서버를 다시 부른다.** 예전에는 받아 온 20행만 화면에서
    // 다시 줄 세웠는데, 그러면 "종합 점수 순"이 실제로는 "핵심구조 상위 20을
    // 점수로 재배열"이었다 - 실측으로 점수 상위 20 중 15건이 그 20행 밖에 있어
    // 영영 보이지 않았다. 다시 부르는 데 몇 초가 들지만, 보이지 않는 후보가
    // 있는 것보다 낫다. 기다리는 동안에는 지금 표를 그 기준으로 정렬해 둔다.
    $("#alternatives-sort")?.addEventListener("change", (event) => {
      const mode = event.target.value;
      if (!alternativesState.ingredients.length) return;
      alternativesState.sortBy = mode;
      if (CLIENT_SORTS[mode]) {
        alternativesState.ingredients = sortIngredientsClientSide(
          alternativesState.ingredients, mode);
        alternativesState.ingredients.forEach((row, index) => { row.rank = index + 1; });
        const body = $("#alternatives-ingredients-body");
        if (body) body.innerHTML = ingredientRowsHtml(alternativesState.ingredients);
      }
      loadAlternatives();
    });
    $("#substitute-target-conditioned").addEventListener("change", updateReview);
    $$('input[name="evidence_mode"]').forEach((input) => input.addEventListener("change", () => {
      state.evidenceMode = document.querySelector('input[name="evidence_mode"]:checked')?.value || "evidence";
      $$(".mode-card").forEach((card) => card.classList.toggle("is-selected", card.querySelector("input")?.checked));
      updateReview();
    }));
    $("#refresh-status").addEventListener("click", async () => { await loadStatus(true); showToast("환경 상태를 갱신했습니다."); });
    $("#settings-refresh").addEventListener("click", async () => { await loadStatus(true); showToast("환경 상태를 갱신했습니다."); });
    $("#refresh-runs").addEventListener("click", async () => { await loadRuns(); showToast("실행 목록을 갱신했습니다."); });
    $("#result-refresh").addEventListener("click", async () => { if (state.selectedRunId) await loadRun(state.selectedRunId, false); });
    $$('input[name="setup_profile"]').forEach((input) => input.addEventListener("change", () => {
      $$(".profile-option").forEach((card) => card.classList.toggle("is-selected", card.querySelector("input")?.checked));
      renderSetup();
    }));
    $("#install-runtime").addEventListener("click", () => startSetup("/api/setup/install", { profile: selectedSetupProfile() }));
    $("#explorer-run").addEventListener("change", async (event) => { state.selectedRunId = event.target.value || null; if (state.selectedRunId) await loadTargets(); else { state.targetRows = []; state.targetTotal = 0; renderTargetTable(); } });
    [$("#target-query"), $("#target-tier"), $("#target-sort")].forEach((element) => element.addEventListener("change", loadTargets));
    $("#target-query").addEventListener("keydown", (event) => { if (event.key === "Enter") { event.preventDefault(); loadTargets(); } });
    $("#compare-runs").addEventListener("click", compareRuns);
  }

  function showLoginGate(message) {
    const gate = $("#login-gate");
    if (!gate) return;
    gate.classList.remove("is-hidden");
    if (message) $("#login-message").textContent = message;
    $("#login-token").focus();
  }

  async function ensureSession() {
    // Auth is off for a local single-user server, so this is a no-op there.
    let session;
    try {
      const response = await fetch("/api/session");
      session = await response.json();
    } catch (_) {
      return true;
    }
    if (!session.required || session.authenticated) return true;
    showLoginGate();
    return false;
  }

  async function submitLogin(event) {
    event.preventDefault();
    const button = $("#login-submit");
    button.disabled = true;
    try {
      await api("/api/session", {
        method: "POST",
        body: JSON.stringify({ token: $("#login-token").value.trim() }),
      });
      $("#login-gate").classList.add("is-hidden");
      $("#login-token").value = "";
      await init();
    } catch (error) {
      $("#login-message").textContent = error.message;
    } finally {
      button.disabled = false;
    }
  }

  let eventsBound = false;

  async function init() {
    if (!eventsBound) {
      bindEvents();
      $("#login-form").addEventListener("submit", submitLogin);
      eventsBound = true;
    }
    if (!(await ensureSession())) return;
    try {
      await loadStatus();
      await loadRuns();
      const initialRunId = new URLSearchParams(window.location.search).get("run_id");
      if (initialRunId) await loadRun(initialRunId, false);
    } catch (error) { showToast(error.message); }
    const initialView = window.location.hash.slice(1);
    navigate(VALID_VIEWS.has(initialView) ? initialView : "setup");
    window.addEventListener("hashchange", () => navigate(window.location.hash.slice(1)));
    if (!state.pollTimer) {
      state.pollTimer = setInterval(async () => {
        try { await loadRuns(); await refreshBatch(); } catch (_) { /* local server may be restarting */ }
      }, 5000);
    }
  }

  init();
})();
