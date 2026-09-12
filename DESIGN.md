# SkinScout Workbench Design Source of Truth

## Source of truth

- Status: Active
- Last refreshed: 2026-08-12
- Primary product surfaces: Ubuntu installer, local Workbench, run results,
  target explorer, beginner README
- Evidence reviewed: `README.md`, `install_skinscout.sh`,
  `scripts/bootstrap_runtime.sh`, `scripts/start_workbench.py`,
  `workbench/server.py`, `workbench/static/*`, pipeline readiness contracts,
  `.omx/plans/fresh-ubuntu-realtime-target-3d-sota-20260812T043137Z.md`

This document is the canonical product and interaction specification for the
local SkinScout Workbench. The backend pipeline remains the source of truth
for scientific results, verification status, provenance, and claimability.
The Workbench must never reinterpret a failed gate as a positive result.

The product is a browser-based local application served from the SkinScout
checkout. It is designed for wet-lab researchers who should not need to know
Ubuntu, Snakemake, conda, file paths, or command-line flags.

## Brand

- Product name: SkinScout Workbench
- Personality: calm, precise, transparent, laboratory-grade
- Primary promise: move from compound input to evidence-aware target
  hypotheses without hiding operational or scientific limits
- Audience language: Korean by default, with scientific names, gene symbols,
  file names, and status tokens preserved in English where that prevents
  ambiguity

## Product goals

1. Guide a first-time user through environment readiness and SkinScout setup.
2. Let a user submit one SMILES or an SDF file without touching a terminal.
3. Show live workflow progress and actionable failure recovery.
4. Present safety, skin context, ranked targets, evidence sources, and
   claimability in one readable result view.
5. Support filtering, comparison, target inspection, and artifact access for
   interactive exploration.
6. Make every important claim traceable to a generated artifact and its
   verification status.
7. Reduce first installation to one copyable terminal block, then provide an
   Ubuntu app-menu launcher so routine use requires no terminal.

## Non-goals and safety boundary

- The application must not silently repartition disks, overwrite a boot
  device, or install an operating system without an explicit operator step.
- Native Ubuntu Desktop is the Release 1 primary platform. WSL2 remains
  guidance-only unless a later release explicitly qualifies it.
- On Windows, the Workbench may provide a copyable WSL2 installation guide and
  launch a user-confirmed installer flow, but must warn about administrator
  rights and reboot requirements.
- On macOS or other systems, the Workbench provides VM or bootable-USB guidance
  and does not perform disk operations.
- Stage 0 is a large, long-running data bootstrap. It is not exposed in the
  beginner flow for Release 1; operator or developer documentation may cover
  it separately with disk/time estimates and explicit confirmation.

## Personas and jobs

### Wet-lab researcher

Job: paste a compound, choose a skin question, and understand whether the
result is useful enough to discuss with a scientist.

Needs: plain-language labels, visible progress, no terminal, clear warnings,
exportable evidence, and an explanation of what is not proven.

### Computational collaborator

Job: inspect provenance, rerun a case, compare runs, and retrieve raw tables.

Needs: run IDs, mode/preset visibility, artifact links, verification details,
filterable target tables, and reproducibility metadata.

### Platform operator

Job: prepare Ubuntu, GPU/model environments, Stage 0 data, and recover failed
runs.

Needs: environment checks, exact next actions, logs, safe install boundaries,
disk estimates, and no ambiguous green states.

## Information architecture

The app has a persistent left rail on desktop and a compact top navigation on
small screens:

1. **시작하기**
   - 환경 확인
   - Ubuntu readiness guidance
   - SkinScout runtime bootstrap
   - data/model readiness
2. **새 분석**
   - compound input
   - analysis question and depth
   - run review and submit
3. **실행 중**
   - active run timeline
   - current stage, elapsed time, live log, cancel action
4. **결과**
   - decision banner
   - compound and safety summary
   - target ranking
   - skin-context evidence
   - provenance and artifact links
5. **탐색**
   - run comparison
   - target search and filtering
   - target detail drawer
6. **환경**
   - runtime checks
   - data/model readiness
   - safe maintenance actions

The first screen is the current setup or analysis workspace, never a marketing
landing page.

## Primary flows

### First launch

1. The native Ubuntu bootstrap installs missing basic tools, prepares the
   runtime, verifies GPU and pipeline readiness, creates an app-menu launcher,
   and opens the Workbench. Existing installations are reused.
2. Detect operating system, distribution, Python/runtime availability, GPU,
   free disk, repository, and data/model artifacts.
3. Explain the current state in one sentence and show the next safe action.
4. If Ubuntu is ready, offer `기본 프로그램 자동 설치`; this action must also
   bootstrap micromamba when it is absent.
5. If native Ubuntu is not ready, offer guided remediation with explicit
   administrator/reboot/disk boundaries. WSL2 and prebuilt-image paths are
   documented as non-primary alternatives, not beginner defaults.
6. After runtime bootstrap, show safety, fast target, comprehensive target,
   and report readiness separately. A user can run a
   safety-only analysis before the full Stage 0 target infrastructure is
   complete, when the backend contract permits it.

### New analysis

1. Choose `SMILES 붙여넣기` or `SDF 파일 불러오기`.
2. Show normalized compound preview, canonical SMILES when available, and
   validation errors before submission.
3. Choose an analysis question:
   - `안전성/피부 반응 먼저 보기` -> `safety`
   - `가능한 단백질 표적 찾기` -> `target-id`
   - `전체 증거 보고서 만들기` -> `report`
4. Choose `빠른 분석`, `종합 분석`, or `둘 다` only when that choice is
   relevant to the selected preset.
5. Show a review screen with estimated compute, required readiness, and an
   explicit statement that target ranking is a hypothesis, not experimental
   proof.
6. Submit and route to the run monitor.

### Results

The result page always starts with the backend's final decision:

- `PASS / proceed`: claimable only when all contract checks pass.
- `FLAG_HIGH / review_before_claim`: human review is required.
- `HALT / stop_before_claim`: stop before making a scientific claim.
- `diagnostic`: useful for debugging or exploration, not claimable.

The page then follows this order: what was input, safety and skin risk,
target evidence, skin context, limitations, provenance, and raw artifacts.

## Design principles

1. **Evidence before enthusiasm**: a green visual never overrides a failed
   verifier or missing source.
2. **One decision, one explanation**: every status has a short reason and a
   next action.
3. **Progressive disclosure**: wet-lab users see the conclusion first;
   collaborators can expand the technical details.
4. **Operational density**: use tables, timelines, and compact panels instead
   of decorative cards or oversized hero sections.
5. **Reproducibility is visible**: run ID, preset, mode, timestamp, and
   artifact paths remain easy to find.
6. **Safe defaults**: no degraded safety mode, unsafe readiness skip, or Stage
   0 build is hidden behind a normal Run button.

## Visual language

- Canvas: warm near-white `#f7f8f6` with white work surfaces.
- Ink: near-black `#17201d`; muted text `#5f6b65`.
- Primary accent: deep teal `#0b6b62` for navigation and confirmed actions.
- Secondary accent: blue `#3568b8` for information and technical links.
- Warning: amber `#a86700`; halt/error: red `#b23a3a`; success: green `#19734a`.
- Borders: cool neutral `#d9dfdb`; focus ring: `#2c79d0`.
- Typography: system sans stack; tabular numerals for scores and counts.
- Radius: 6px for controls and 8px for framed tools; no nested cards.
- Shadows: minimal, used only for drawers and modal surfaces.
- Status is never color-only; include text, icon/shape, and a reason.

## Components

- App shell: rail, breadcrumb, page title, environment status chip.
- Setup stepper: numbered stages with completed/current/blocked states.
- Readiness row: check label, status, detail, and action.
- Compound input: SMILES textarea, SDF drop zone, validation message,
  structure preview placeholder, local SHA256 file hash, and example selector.
- Analysis selector: radio/segmented controls with compute and evidence
  descriptions.
- Run review panel: input, preset, mode, readiness blockers, and submit.
- Run timeline: Stage 1 through Stage 11, current stage, duration, log toggle.
- Decision banner: decision, claimability, reason list, and next action.
- Metric strip: compact safety values and skin context support flags.
- Target table: rank, gene, protein, final score, docking RRF, skin tier,
  source count, efficacy labels, and inspect action.
- Target detail drawer: score composition, source labels, skin evidence,
  efficacy labels, and artifact links.
- Explorer toolbar: run selector, query, skin tier filter, source filter,
  score sort, and comparison toggle.
- Artifact list: human label, relative path, type, verified/unverified state,
  canonical citation, preview/download action.
- Beginner bootstrap: idempotent Ubuntu script, user-local micromamba install,
  app-menu launcher, automatic browser start, dry-run mode, and clear automatic
  versus licensed/manual download boundaries.
- Interpretation guide: plain-language PASS/REVIEW/HALT meanings, score
  glossary, and decision-specific next action.
- Toast/live region: non-blocking run and validation messages.

## Accessibility

- Korean text is plain and concrete; scientific identifiers stay copyable.
- All controls have visible labels, keyboard focus, and 44px minimum hit area.
- Tables have real headers, captions, and row-level keyboard actions.
- Status colors meet contrast requirements and include text labels.
- Live run updates use an `aria-live` region without stealing focus.
- File input supports keyboard selection and drag-and-drop.
- Do not rely on hover to reveal essential information; drawers also open by
  click and keyboard.
- Respect `prefers-reduced-motion`; progress animations are optional.

## Responsive behavior

- Desktop: 248px rail plus a centered content column capped at 1440px.
- Tablet: rail collapses to an icon/text top bar; two-column panels stack when
  either column would be narrower than 360px.
- Mobile: single column, sticky bottom primary action, horizontal table scroll
  with the rank and gene columns pinned where supported.
- Result banners and long compound names wrap; no fixed-height text containers.
- Target detail becomes a full-height bottom sheet on mobile.
- Stable controls use explicit min-height, aspect ratio, or table layout so
  live labels cannot shift surrounding content.

## Interaction states

Every major view must support loading, ready, empty, blocked, running,
completed, failed, and stale states.

- Loading: skeleton or text status, never a blank panel.
- Blocked: reason, affected action, and one recommended recovery.
- Failed: preserve the run ID/log link and show retry only when safe.
- Stale: mark cached readiness or old artifacts with the checked timestamp.
- Empty explorer: suggest loading an existing verified run or starting an
  analysis; do not fabricate target rows.
- Offline: local results remain browseable; new runs and public safety calls
  explain the network dependency.

## Content voice

- Use direct Korean: `분석 시작`, `검토 필요`, `주장 불가`, `환경 준비`.
- Replace unexplained UI terms such as `claimable`, `final score`, and
  `source count` with `검증 기반 주장 가능`, `종합 예측 점수`, and
  `계산 근거 수`; preserve the machine token only in technical artifacts.
- Avoid claims such as `효과가 있습니다`; use `가능성`, `예측`, `근거`,
  `검증 상태`.
- Explain technical terms on first use: `피부 맥락 지원(skin context support)`.
- Keep actionable guidance to one or two sentences.
- Never hide degraded, diagnostic, or missing-evidence states behind vague
  labels such as `완료`.

## Implementation constraints

- Keep the first implementation dependency-light: Python standard-library
  local HTTP server plus HTML/CSS/JavaScript.
- Bind to loopback by default. Do not expose a local research workstation to
  the network without an explicit operator option.
- Run commands with argument arrays, never shell interpolation.
- Require a per-server token, same-origin browser request, and JSON content
  type for every state-changing local API call.
- Constrain artifact access to `results/runs/<run_id>` and reject traversal.
- Reuse `scripts/run_skinscout.py` as the execution contract; do not duplicate
  pipeline logic in the UI.
- Read `run_summary.json` and `run_verification.json` as authoritative.
- Treat CSV/TSV ranking files as exploratory data and preserve their source
  path in the UI.
- Keep SDF uploads local to the browser/workstation. Record the local SHA256
  hash for provenance and never send SDF files to external object storage in
  Release 1.
- The server must degrade gracefully when optional Python packages or model
  environments are not available.
- Resolve `MAMBA_ROOT_PREFIX` per environment because the base and GPU
  environments may live under different user-local roots.
- Show a green `PASS` only when backend `claimable` is exactly true; a
  biological `PASS` with incomplete verification remains diagnostic.
- Installer scripts must be Ubuntu-only, idempotent, support `--dry-run`, keep
  environments under the user's home directory, reuse existing downloads, and
  never start Stage 0 without a separate explicit confirmation.
- Licensed or account-gated sources must be shown as manual/optional and must
  never be described as automatically downloaded.
- Use canonical citation labels from generated manifests or backend metadata.
  Do not invent institution-specific display names in the Workbench.
- Beginner UI must not expose or start Stage 0. Stage 0 controls, if present,
  belong in operator/developer surfaces outside the Release 1 beginner path.

## Release 1 decisions

- UI shell: browser-local Workbench remains the Release 1 interface. Do not
  introduce Tauri or Electron for Release 1.
- Platform: native Ubuntu Desktop is the primary supported operator path.
  WSL2, prebuilt workstation images, and Ubuntu 26.04 qualification are
  follow-up compatibility work.
- Compound files: SDF upload stays local to the browser/workstation. Store and
  display local SHA256 hashes for provenance; do not add institution object
  storage in Release 1.
- Citations: show canonical citation labels from backend artifacts and
  manifests. Institution-specific citation templates are deferred.
- Beginner data flow: Stage 0 is not visible in the beginner Workbench path.
  Beginners see readiness, repair, and data/model availability, not Stage 0
  controls.
- Shared GPU policy: shared-GPU administration is deferred. Release 1 targets
  a dedicated local Ubuntu workstation with coordinator-owned GPU leasing.
