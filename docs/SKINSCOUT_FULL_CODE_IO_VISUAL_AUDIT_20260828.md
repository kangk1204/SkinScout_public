# SkinScout 전 코드·I/O·시각화 초정밀 감사 보고서

- 감사일: 2026-08-28 (Asia/Seoul)
- 최종 기준 커밋: `6646dc5e0c13bbb14e93d3658496d55d73af4a49`
- 최초 관찰 커밋: `4f4efe69` — 감사 도중 외부 작업이 4개 변경 파일을 `6646dc5e`로 커밋하여 최종 판정 기준을 새 HEAD로 재고정함
- 판정: **REQUEST CHANGES / RELEASE BLOCK**
- 독립 판정: code-reviewer **REQUEST CHANGES**, architect **BLOCK**
- 변경 범위: 이 보고서만 신규 생성. SkinScout 소스·설정·테스트는 수정하지 않음

## 1. 결론

SkinScout는 방대한 fail-closed validator, 원자적 artifact promotion, 입력 quoting, 다수의 회귀 테스트 등 좋은 하부 계약을 갖고 있다. 그러나 현재 구현은 **실행 의도, 준비성, 실제 DAG, 검증, 산출물 게시, Workbench 상태, 과학적 claim**을 하나의 권위 있는 계약으로 연결하지 못한다. 그 결과 다음과 같은 수용 차단 문제가 실제로 성립한다.

1. verifier가 실패한 실행도 `run_summary.json`이 있으면 Workbench에서 `completed` 및 `claimable`처럼 보일 수 있다.
2. prospective model promotion은 원시 예측·정답·bootstrap·실제 preregistration을 재검산하지 않고 자체 보고 CSV만으로 `promote`를 낼 수 있다.
3. 기본 `report` 실행의 MD 인자는 `50.0`을 정수 CLI로 넘겨 시작 즉시 실패한다.
4. 광고된 `report-fast`와 실제 readiness 요구사항이 모순되고, comprehensive에 필수인 DiffDock는 readiness에서 빠져 있다.
5. provenance manifest가 일반 실행에서는 없어도 verifier가 `ok`를 반환할 수 있다.
6. 오프라인 Mol* viewer는 현재 API 오류로 깨지고, 그 한 줄을 바꿔도 `file://` CORS 때문에 구조가 로드되지 않는다.
7. Stage 11의 “publication figures”는 과학 plot 대신 파일 수·행 수·키 수를 그리며, 과거 회고 그림과 수치 주장은 현재 저장소만으로 재생성되지 않는다.
8. KG seed 우선순위 상수가 실제 논문 수 및 피부 효능 지원처럼 소비되고, Boltz 단위·crop, Stage 8 전자상태·conformer 처리에도 과학적 의미 오류가 있다.

따라서 **개별 로컬 연구 단계의 탐색적 사용은 가능하지만**, Workbench 성공 상태, 자동 보고서, 재현 가능한 비교, model promotion, publication/claim 산출물을 신뢰하는 릴리스는 차단해야 한다.

### 심각도 집계

| 등급 | 건수 | 의미 |
|---|---:|---|
| Critical | 2 | 잘못된 성공/승격 판정으로 핵심 의사결정을 직접 오염 |
| High | 30 | 핵심 기능 실패, 안전·과학·재현성·보안 또는 릴리스 계약 위반 |
| Medium | 24 | 특정 입력/환경/동시성에서 잘못된 동작 또는 중요한 운영·UX 결함 |
| Low | 6 | 유지보수·문서·관측성·성능·접근성 부채 |
| **합계** | **62** | 중복 원인을 하나의 결함으로 병합한 수치 |

## 2. 검토 범위와 방법

### 2.1 저장소 인벤토리

| 범위 | 파일 | LOC/비고 |
|---|---:|---:|
| 전체 tracked | 418 | 코드, 설정, 문서, assets 포함 |
| 코드/설정으로 분류한 tracked | 375 | 전 파일 정적 inventory 및 패턴 스캔 |
| Production Python | 176 | 107,747 LOC |
| Test Python | 132 | 81,779 LOC |
| Snakemake/Snakefile | 23 | 약 5,314 LOC |
| Shell | 17 | 2,420 LOC |
| Workbench 비벤더 HTML/CSS/JS | 3–5 | 분류 기준에 따라 1,740–2,101 LOC |
| JSON/YAML 계약·설정 | 21 | 약 1,260 LOC |
| Mol* vendored/minified assets | 4 | 약 7,907 LOC |
| 코드성 텍스트 합계 | — | 약 207,010 LOC |

“line by line” 요구에 따라 375개 코드/설정 파일을 inventory 및 정적 line scan 대상으로 삼고, first-party 실행 경로는 producer→consumer→validator→UI까지 의미 검토했다. 다만 minified Mol* vendor 내부는 사람이 의미 단위로 재감사하는 대신 **파일 존재·무결성·로딩 API·SkinScout 호출 경계**를 검토했다. 이 경계를 숨기고 “모든 vendor 코드를 인간이 한 줄씩 재검증했다”고 주장하지 않는다.

### 2.2 독립 검토 레인

| 레인 | 독립 역할 | 범위 | 판정 |
|---|---|---|---|
| 전체 코드 | code-reviewer | 전 저장소, 변경 스냅샷, false-positive 반증 | REQUEST CHANGES |
| 아키텍처 | architect | profile/readiness/DAG/state/provenance/artifact | BLOCK |
| I/O 계약 | executor | CLI, JSON/CSV/TSV/SDF, 경로, HTTP, atomic write | 다수 High |
| 시각화 | designer | Workbench, Stage 9/11, Mol*, PNG, 접근성 | 수용 차단 |
| 파이프라인·평가 | code-reviewer | Stage 0–8, `eval/`, 과학적 의미 | REQUEST CHANGES |
| 테스트·운영 | test-engineer | 전체 suite, 설치, 공급망, Compose, CI | NEEDS ATTENTION |

### 2.3 증거 등급

- **재현**: 실제 명령, 브라우저 또는 최소 입력으로 현상 확인.
- **직접 코드**: producer와 consumer 경로가 결정적으로 결함을 만든다는 정적 증거.
- **추론**: 코드상 가능한 장애이나 장시간/GPU/재시작 실험은 수행하지 않음.
- 각 finding은 이 구분과 trigger, impact, 수정 방향을 포함한다.

### 2.4 감사 중 스냅샷 변경

감사 시작 시 `README.md`, `scripts/model_readiness.py`, `scripts/tests/test_model_readiness.py`, `scripts/tests/test_researcher_guide_numbers.py`가 수정 상태였다. 외부 작업이 이를 다음 커밋으로 만들었다.

```text
6646dc5e fix: stop readiness blocking runs over tools that are installed
4 files changed, 232 insertions(+), 24 deletions(-)
```

최종 보고서는 `6646dc5e`를 기준으로 해당 4개 파일을 다시 검토하고 관련 테스트 31개를 재실행했다. 기존 untracked 문서 `docs/SKINSCOUT_COMPREHENSIVE_READONLY_AUDIT_20260826.md`는 사용자 자산으로 보존했으며 수정하지 않았다.

## 3. 실패가 전파되는 현재 구조

```text
사용자 입력
  │
  ├─ RunProfile ──X── readiness별 별도 요구 목록
  │                    └─X─ 실제 Snakemake rule/env와 불일치
  ▼
Snakemake 계산
  ▼
run_summary/viewer 생성
  ▼
output verifier
  │
  ├─ 실패해도 summary가 남음
  └─ manifest/sealed package가 일반 실행에서 필수가 아님
  ▼
Workbench
  ├─ summary 존재를 completed의 대리값으로 사용
  ├─ summary의 claimable을 그대로 노출
  └─ 정상 로컬 실행은 durable artifact promotion을 우회
```

근본 원인은 **좋은 계약들이 서로 연결되지 않은 “계약 섬”**이라는 점이다. `RunProfile`, model readiness, Snakefile, verifier, coordinator, report package, Stage 11 claim manifest가 각각 다른 성공 정의를 갖는다.

## 4. Critical findings

### C-01. 검증 실패 실행이 `completed` 및 `claimable`처럼 표시됨

- 근거: `scripts/run_skinscout.py:2790-2803`, `workbench/server.py:1918-1921`, `workbench/server.py:1969-1981`, `workbench/server.py:2247-2263`
- 실제 fixture: `results/runs/failed_verify_case/run_verification.json:36,73`
- 증거: **직접 코드 + 저장된 실패 사례**
- Trigger: Snakemake는 끝났지만 output verification 또는 후단 report/viewer 검증이 실패하고 summary 파일은 남은 실행.
- 현재 동작: 실행기는 summary/viewer를 먼저 만들고 verifier를 나중에 실행한다. Workbench는 active job이 없고 summary가 있으면 verifier 상태 대신 `completed`를 반환하며 UI용 `claimable`도 summary 값을 사용한다.
- Impact: 사용자는 검증 실패 산출물을 성공 결과나 claim 가능한 결과로 오인할 수 있다. 계산 완료, 검증 완료, 시각화 준비, artifact 게시, publication claim이 한 상태로 축약된다.
- 수정: `queued → running → compute_succeeded → verification_succeeded → artifacts_published → claim_ready` 상태 머신을 도입하고, Workbench 상태는 coordinator+verification+artifact registry의 권위 있는 기록으로만 산출한다. summary 존재를 성공 판정에 사용하지 않는다.
- 회귀 테스트: verifier 실패 후 list/detail/card 모두 `verification_failed`; summary가 있어도 `completed` 금지; claim manifest가 없으면 `claim_ready=false`.

### C-02. Prospective model promotion이 자체 보고 CSV만으로 가능

- 근거: `eval/prospective_promotion_eval.py:91-106,149-269,272-379`, `eval/run_iteration.py:397-425,3141-3154`
- 현재 테스트: `scripts/tests/test_prospective_promotion_eval.py:17-58`, `scripts/tests/test_eval_run_iteration.py:3110-3161`
- 증거: **직접 코드 + 최소 CSV 재현**
- Trigger: 임계값을 만족하는 한 행 CSV와 형식만 맞는 임의 64자리 `preregistration_sha256` 제공.
- 현재 동작: raw prediction, ground truth, bootstrap samples, candidate experiment, model artifact, 실제 preregistration 파일을 읽어 지표를 재계산하지 않는다. “fresh recomputation”도 동일 집계 CSV를 다시 판정한다. 후보 실험이 `not_run`이어도 promotion 가능하다.
- Impact: 조작·오계산·stale 집계로 모델이 `promote`될 수 있어 이후 모든 순위와 과학적 결론을 오염한다.
- 수정: 봉인된 pair/query-level 예측과 정답에서 metric 및 bootstrap을 내부 재계산하고 model/data/preregistration 파일의 실제 hash를 검증한다. candidate experiment `not_run` 상태에서는 promotion을 금지한다.
- 회귀 테스트: 임의 집계값, 존재하지 않는 preregistration hash, mismatch model hash, 누락 raw rows, `not_run` candidate를 모두 fail-closed.

## 5. High findings

### H-01. `report-fast` 계약과 실제 readiness 요구가 모순

- 근거: `skinscout/contracts/run_profile.py:52-60,134-150`, `scripts/run_skinscout.py:424-443`, `workbench/server.py:2385-2409`, `README.md:79-87,930-941`
- 증거: **직접 코드 + 동적 요구 목록 확인**
- `report/fast`가 현재 요구하는 항목: `autodock_gpu`, `autodock_gpu_env`, `autogrid`, `bioemu_env`, `gnina`, `gpu`, `md_env`, `meeko_env`, `qm_env`.
- Impact: Demo가 지원한다고 광고하는 report-fast가 Full/Advanced 환경 없이는 시작되지 않는다.
- 수정: `analysis_profile → stages/tools/envs/artifacts/publication eligibility` capability matrix를 하나의 계약으로 만들고 launcher, Workbench, Snakefile이 함께 소비한다.

### H-02. Durable coordinator가 실제 OS 프로세스를 durable하게 소유하지 않음

- 근거: `workbench/coordinator.py:175-283,608-673`, `workbench/server.py:1627-1659,1719-1808,1769-1777,2519-2551`
- 증거: **직접 코드; restart/kill 실험은 미수행한 추론 포함**
- Trigger: 장시간 child 실행 중 Workbench 서버 재시작, lease 만료 또는 cancel.
- Impact: PID/process group/attempt token이 메모리에만 있어 orphan child, 동일 job 재queue 후 중복 실행, 실제 프로세스가 살아 있는 상태의 논리적 cancel이 가능하다.
- 수정: 전용 durable worker가 process를 소유하거나 PID, start-time, process group, attempt ownership을 영속화하고 재시작 때 reattach/kill/reconcile한다.

### H-03. 일반 실행에서 provenance manifest가 없어도 verifier `ok` 가능

- 근거: `scripts/verify_run_outputs.py:262-281,4861-4964`, `scripts/run_skinscout.py:1081-1141`, `skinscout/contracts/run_profile.py:271-344`
- 관찰: local run 34개가 manifest 없이 verification `ok`; 예 `results/runs/deepdiag_Tapinarof/run_verification.json:284-288`.
- 증거: **직접 코드 + 저장 산출물 조사**
- Impact: 실제 code commit/dirty state, resolved workflow config, rules, tool/env/model versions, 소비한 data/CAS bundle을 하나의 실행 identity로 증명할 수 없다.
- 수정: 모든 non-stage0 verifier에서 manifest를 필수화하고 git/config/rule/env/model/data/input hash를 결속한다.

### H-04. 정상 Workbench 실행이 atomic artifact promotion과 sealed report 검증을 우회

- 근거: `workbench/coordinator.py:373-544`, `workbench/server.py:1719-1808,2007-2086`, `workbench/static/app.js:553-558`, `scripts/report_package_contract.py:141-405`, `scripts/verify_run_outputs.py:4828-4858,4931-4932`
- 증거: **직접 코드**
- Impact: 강한 registry/promotion 구현이 있어도 일반 실행 산출물은 `unregistered`, hash/download URL 없음. completed verifier는 strict report package, browser seal, immutable package identity를 검사하지 않는다.
- 수정: attempt workspace에서 생성→verification→atomic promotion을 모든 실행에 강제하고, flat HTML이 아니라 sealed package validator 결과를 완료 조건으로 연결한다.

### H-05. Publication DAG가 mode와 optional-stage 계약을 역으로 강제

- 근거: `workflow/rules/stage11_publication.smk:11-26,122-169`, `workflow/Snakefile:1429-1455`
- 증거: **직접 코드**
- Trigger: fast 또는 analog/retrosynthesis/MD 비활성 조합에서 publication flag 사용.
- Impact: 비활성 optional stage와 comprehensive consensus가 암묵적으로 다시 필수화되어 설정은 수락되지만 DAG가 성립하지 않는다.
- 수정: publication 지원 profile을 명시적으로 제한하거나 profile별 input graph를 생성하고 DAG 전에 조합을 검증한다.

### H-06. 기본 `report`의 Stage 7 MD가 CLI parsing에서 실패하며 설정 일부가 dead

- 근거: `workflow/Snakefile:1167-1185`, `workflow/rules/stage7_md.smk:26-58,73-88`, `scripts/stage7_gromacs_prep.py:503-511,556-584,858`, `scripts/stage7_gromacs_run.py:31-56,172`, `workflow/config.yaml:153-165`
- 증거: **직접 재현**
- 재현 결과: 기본 `duration_ns: 50`은 `config_float()`로 `50.0`이 되어 두 `type=int` CLI 모두 `invalid int value: '50.0'`로 종료.
- 추가 문제: `ligand_ff`는 ACPYPE topology 생성에 사용되지 않고, `timestep_fs`는 전달되지 않으며 실행은 `dt=0.002`, `duration*500000`을 고정 가정한다.
- Impact: 기본 report DAG의 MD가 시작 전 실패하고, 선언 설정과 실제 시뮬레이션 provenance가 다르다.
- 수정: duration/timestep 타입과 계산을 전 구간 통일하고 ligand force field를 실제 실행에 연결하거나 미지원 설정을 제거한다.

### H-07. Comprehensive 필수 DiffDock가 launcher/Workbench readiness에서 누락

- 근거: `workflow/rules/stage3a_comprehensive.smk:157-185`, `scripts/run_skinscout.py:413-421`, `workbench/server.py:118,1118`
- 증거: **직접 코드**
- Impact: DiffDock가 없는 환경을 준비 완료로 표시한 뒤 DAG 중간에서 실패한다.
- 수정: comprehensive/both의 required models와 Workbench mapping/label에 `diffdock`를 연결하고 preflight 통합 테스트를 추가한다.

### H-08. 임의의 Conda env에 동일 명령이 있으면 readiness가 녹색

- 근거: `scripts/model_readiness.py:140-185`, `scripts/tests/test_model_readiness.py:429-451`
- 증거: **직접 코드; 최신 커밋 회귀**
- Impact: Snakemake rule이 선언한 특정 env에는 도구가 없어도 unrelated env의 executable 때문에 ready가 되어 본 실행에서 실패한다. 현재 테스트는 잘못된 계약을 성공으로 고정한다.
- 수정: tool→rule/env mapping을 만들고 정확한 env 내부 version/smoke를 실행한다. wrong-env 및 `CONDA_PREFIX` sibling 탐색 테스트를 추가한다.

### H-09. verifier/summary가 symlink artifact를 따라 run 외부 데이터를 검증

- 근거: `scripts/verify_run_outputs.py:193-239`, `scripts/summarize_run_outputs.py:158-160,177-227`, `scripts/run_skinscout.py:200-216`
- 증거: **직접 코드**
- Trigger: required artifact를 run 밖 파일을 가리키는 symlink로 교체.
- Impact: mutable 외부 데이터가 검증·fingerprint·summary에 포함되어 run 격리와 provenance가 깨진다.
- 수정: 모든 path component의 symlink를 거부하고 resolved containment, `O_NOFOLLOW`, open 후 `fstat` 검증을 사용한다.

### H-10. launcher와 Workbench의 결과 root 계약이 분리됨

- 근거: `scripts/run_skinscout.py:77-82`, `workbench/server.py:74-77,1048-1051,2453-2465`
- 증거: **직접 코드**
- Trigger: `SKINSCOUT_RESULTS_ROOT`를 사용자 지정한 Workbench 실행.
- Impact: child는 custom root에 결과를 쓰지만 UI는 repository-relative `results/runs`만 읽어 성공 결과가 사라진 것처럼 보인다.
- 수정: 공통 path contract를 사용하고 server, child, logs, artifact registry에 동일 resolved root를 명시적으로 전달한다.

### H-11. Qualification output이 input과 같으면 입력을 먼저 삭제

- 근거: `scripts/qualification_manifest.py:891-905,1113-1117,1136-1143`
- 증거: **직접 코드**
- Trigger: `--out-json`을 input manifest/report 경로 또는 동일 inode의 hardlink/symlink로 지정.
- Impact: 평가 전에 원본 입력이 삭제되는 파괴적 I/O alias.
- 수정: 모든 input/output resolved path와 inode alias를 사전 거부하고, 입력을 모두 읽은 뒤 unique temp→fsync→replace한다. output 사전 unlink를 제거한다.

### H-12. KG seed 상수가 실제 논문 수와 피부 효능 지원으로 소비됨

- 근거: `scripts/stage0_skin_kg.py:55-73,232-244,286-323`, `scripts/stage3_kg_efficacy_label.py:81-123,199,251-258,378-399,642`, `eval/skin_efficacy_recovery_eval.py:17,100-124`, `scripts/summarize_run_outputs.py:1058,1147`, `workbench/static/app.js:578`
- 증거: **직접 실행 + 코드**
- 예: counted PMID가 0인 TYR whitening seed가 `562 papers`처럼 반환될 수 있다.
- Impact: curated prior가 사실적 문헌 건수 및 `skin_efficacy_supported`/`skin_context_supported`로 승격되어 과학적 결과를 과장한다.
- 수정: `curation_prior`와 unique verified PMID count를 타입 수준에서 분리하고, 실제 문헌 지원 판정에는 counted evidence만 사용한다.

### H-13. STopTox 주 API 실패가 광범위 fallback 성공에 묻힘

- 근거: `scripts/stage2_stoptox.py:60-64,213`
- 증거: **직접 코드**
- Impact: 응답 계약 변경이나 코드 오류까지 `except Exception`으로 HTML scraping에 전환되고 최종 `status=ok`; 주 실패 원인이 사라져 안전성 모델 degradation을 감지할 수 없다.
- 수정: 예상 network/parse 예외만 포착하고 `primary_source`, `fallback_source`, `fallback_reason`, 원 오류를 결과에 보존한다.

### H-14. Cold-start와 cosmetic Top-K가 선언된 global rank를 무시

- 근거: `eval/cold_start_eval.py:133-152,187-196`, `eval/cosmetic_retrospective_eval.py:195-214,273-282`; 정상 대조 `eval/skin_known_target_recovery_eval.py:345-364,488-496`
- 증거: **직접 재현**
- 재현: rank `[100,101]`인 2행 subset에서 truth가 첫 행이면 두 evaluator 모두 Top1 hit.
- Impact: truncated ranking이 실제 global rank보다 과도하게 좋은 Top-K를 얻어 threshold를 통과한다.
- 수정: rank 열이 있으면 `rank <= k`; score-only 입력은 명시된 tie 정책과 full-universe coverage manifest로 rank를 생성한다.

### H-15. Boltz binder probability가 `neg_log_uM` 열에 기록됨

- 근거: `scripts/boltz2_runner.py:169-180`, `scripts/stage3_boltz2_affinity.py:96-103,176-187`, `scripts/tests/test_stage3_scorer_outputs_fail_closed.py:1198-1243`
- 증거: **직접 재현**
- 재현: probability `0.81`, affinity value `1.5`에서 `boltz2_neg_log_uM=0.81`.
- Impact: 확률과 affinity가 같은 단위 열·ranking 정책에 혼합되어 score 비교와 해석이 무효다.
- 수정: `binder_probability`와 명시적 affinity quantity/unit을 분리하고 schema version과 ranking transform을 기록한다.

### H-16. Boltz `max_residues`/`crop_radius`가 검증만 되고 입력에 적용되지 않음

- 근거: `scripts/stage3_boltz2_affinity.py:54-60,106-147,155-184`, `scripts/stage5_boltz2.py:81-106,270-326`, `scripts/boltz2_runner.py:28-41,59-77`, `workflow/config.yaml:40-45`
- 증거: **직접 코드**
- Impact: 1,000 residue receptor에 max 100을 줘도 전체 sequence가 YAML에 들어가 GPU OOM 및 target-selective 누락으로 consensus 편향 가능.
- 수정: pocket-aware crop+residue mapping provenance를 구현하거나 상한 초과를 명시적으로 거부한다.

### H-17. 감작성 HALT 투표 임계값 상한이 없어 안전 gate 무력화 가능

- 근거: `workflow/Snakefile:1071-1076`, `scripts/stage2_consensus.py:141-160,163-201`
- 증거: **직접 재현**
- 재현: 3개 모델 모두 positive, `halt_min_votes=4`이면 `HALT`가 아닌 `FLAG_HIGH`.
- 수정: CLI와 config 모두 `1 <= threshold <= configured_model_count`를 강제한다.

### H-18. Charged/open-shell ligand도 CREST를 neutral/singlet로 실행

- 근거: `scripts/stage8_crest.py:118-147,178-198`, `scripts/tests/test_downstream_fail_closed.py:4175-4234`
- 증거: **직접 코드 + [CREST 공식 keyword 계약](https://crest-lab.github.io/crest-docs/page/documentation/keywords.html)**
- Impact: 계산한 charge/spin을 command에 넘기지 않아 CREST와 후속 xTB/DFT가 서로 다른 전자상태를 다룬다.
- 수정: `--chrg`와 `--uhf`를 전달하고 command/version/electronic-state provenance를 저장한다.

### H-19. “xTB cluster”가 ensemble을 분리·클러스터링하지 않고 DFT는 첫 frame만 사용

- 근거: `scripts/stage8_xtb_cluster.py:104-133,149-184`, `scripts/stage8_dft.py:117-138,232-259`, `scripts/stage8_crest.py:123-128`
- 증거: **직접 재현 + [CREST coordinate 계약](https://crest-lab.github.io/crest-docs/page/documentation/coords.html)**
- Impact: multi-structure `crest_conformers.xyz`에서 첫 energy/첫 structure만 소비해 conformer reranking과 cluster 선택이 사실상 구현되지 않는다.
- 수정: ensemble을 frame별 분리, 각 conformer xTB 계산, cluster/선택 정책 기록, 선택된 단일 XYZ만 DFT에 전달한다.

### H-20. 기본 오프라인 Mol* viewer가 실제 브라우저에서 동작하지 않음

- 근거: `scripts/make_results_viewer.py:2-12,91-101,433-459`, `scripts/run_skinscout.py:1499-1526`, `README.md:187-204`; 정상 API 대조 `scripts/report_assets/fast_report.js:33-38`
- 증거: **실제 생성 + headless Chrome 재현**
- 재현 1: `viewer.clear is not a function`.
- 재현 2: 임시 HTML에서 `viewer.plugin.clear()`로만 교체해도 `file://`에서 receptor PDB와 pose SDF fetch가 CORS로 차단되어 blank canvas.
- 추가: 빈 `mode_comprehensive` 디렉터리도 존재만으로 선택하여 실제 fast run viewer 생성이 consensus CSV 없음으로 실패.
- 수정: 데이터가 실제 있는 mode를 선택하고 구조를 embed/`loadStructureFromData`하거나 지원되는 로컬 HTTP server 경로를 제공한다. 실제 browser E2E를 completion gate로 둔다.

### H-21. Viewer target ID로 output directory를 탈출하고 stale HTML을 성공처럼 재사용

- 근거: `scripts/make_results_viewer.py:121-163,467-523`
- 증거: **직접 재현**
- 재현: target ID `../../escaped`가 지정 output 밖 `/tmp/.../escaped.html`을 생성.
- Impact: 공격자/오염 CSV가 임의 상위 경로의 `.html` 파일을 덮어쓸 수 있고, 개별 target 생성 실패 뒤 이전 HTML이 남아 현재 결과처럼 보일 수 있다.
- 수정: target ID를 slug로 정규화하고 resolved containment를 검증하며, 전체 output을 unique staging directory에 생성한 뒤 원자 교체한다.

### H-22. Stage 11 “publication figures”가 과학 plot 대신 artifact count를 그림

- 근거: `scripts/stage11_make_figures.py:37-78,145,2050-2105,2149-2172`
- 증거: **직접 코드 + 산출물 시각 확인**
- 현재 caption은 artifact-count diagnostic임을 일부 공개하지만 파일명·제목·status는 target landscape, pharmacophore, affinity scatter, RMSD/RMSF 같은 과학 figure를 계속 암시한다.
- Impact: publication package에서 실제 분석이 구현된 것으로 오인될 수 있다.
- 수정: figure별 데이터 변환·단위·누락 정책·plot을 구현한다. 전까지는 `diagnostic_artifact_count`로 이름과 status를 낮추고 publication readiness를 false로 둔다.

### H-23. Stage 9 생성 HTML에 stored HTML/script injection 가능

- 근거: `scripts/stage9_report.py:146-168,2766-2767,3016-3026,3032-3040`
- 증거: **직접 코드**
- Impact: raw run ID, JSON, CosIng INCI/functions가 escape 없이 `<pre>`/`<p>`에 삽입된다. Workbench HTTP CSP와 달리 다운로드한 local HTML에는 응답 CSP가 없어 content spoofing 또는 script 실행 가능.
- 수정: 모든 문자열과 JSON dump를 `html.escape()`하고 CSP meta, injection fixture, direct-file browser test를 추가한다.

### H-24. Fast report가 좌표계가 다른 free ligand를 receptor 위에 “input ligand”로 overlay

- 근거: `scripts/report_assets/fast_report.js:103-133`, `scripts/stage9_report.py:2081-2088,2433-2438`
- 증거: **직접 코드**
- Impact: docking/alignment되지 않은 free ligand coordinates를 receptor와 같은 complex처럼 표시해 결합 위치를 과학적으로 오해시킨다.
- 수정: ligand-only view로 분리하거나 검증된 pose/alignment만 overlay하고 provenance label을 표시한다.

### H-25. Mol* browser seal이 실제 구조·원자 identity를 증명하지 못함

- 근거: `scripts/report_assets/fast_report.js:47-89`, `scripts/verify_molstar_report.py:70-79,139-209`
- 증거: **직접 코드**
- Impact: screenshot의 색 분포/픽셀 차이만으로 통과할 수 있어 빈/잘못된 구조나 다른 receptor/ligand identity를 false-pass할 수 있으며 screenshot hash도 결과에 영속 결속되지 않는다.
- 수정: plugin state의 loaded model count, atom count, source hash, ligand component identity를 검사하고 screenshot 및 verification hash를 package manifest에 기록한다.

### H-26. 현재 retrospective/manuscript 수치와 그림이 저장소만으로 재현되지 않음

- 근거: `scripts/make_retrospective_figures.py:30-49,135-195,231-275`, `results/MANUSCRIPT.md:1-6`, `results/RETROSPECTIVE.md:47,58-69`, `docs/RESEARCHER_GUIDE.md:141-176,236-275`
- 증거: **재생성 실패 + tracked artifact 조사 + PNG 직접 확인**
- 재현: figure generator가 `results/runs/retinol_demo/03_targets/demo_ranked_targets.csv` 부재로 종료. `results/runs/`는 ignored이며 tracked 결과는 3 PNG, 두 Markdown, 제한된 validation CSV뿐이다.
- 문제: manuscript는 per-run CSV가 committed됐다고 주장한다. 회고는 Vina `-10`을 “nM regime”로 직접 대응하고 모든 실패를 receptor 구조 하나의 원인으로 단정한다. 가이드는 v2 구조 오류를 뒤에서 인정하면서 앞에서는 44개 “committed runs” 및 단일 원인 서술을 계속 사용한다.
- Impact: 현재 코드/데이터로 검증 불가능한 역사 수치와 과도한 인과 해석이 README에서 연결된다.
- 수정: 입력 CSV/manifest/hash를 추적 가능한 snapshot으로 봉인하고 그림을 재생성한다. docking score를 affinity 단위로 직접 변환하거나 단일 원인으로 단정하지 않는다. v1 산출물은 명확히 withdrawn/stale 표기한다.

### H-27. 설치 신뢰 사슬의 최상단이 mutable `main`

- 근거: `README.md:57`, `install_skinscout.sh:6,178-181`, `scripts/tests/test_beginner_bootstrap.py:75-80`
- 증거: **직접 코드**
- Impact: 이후 binary SHA가 강해도 설치기와 hash 목록을 정의하는 코드를 `curl | bash`와 default branch clone으로 받아 신규 설치 신뢰 사슬이 끊긴다.
- 수정: release tag/commit SHA URL, clone 후 expected commit+signature/attestation, installer 자체 checksum 검증을 제공한다.

### H-28. Runtime env/model checkout이 lock되지 않고 stale env를 재사용

- 근거: `scripts/install_runtime.py:404-420,1061-1098`, `envs/base.yml:7-32`, `envs/autodock_gpu.yml:10-16`, `envs/dti.yml:11-21`, `envs/viz.yml:11-18`, `envs/bioemu.yml:18`, `scripts/model_readiness.py:292-342`, `README.md:1111-1188`
- 증거: **직접 코드 + env inventory**
- 관찰: 10개 YAML에 무버전 conda dependency 54개, 무버전 pip dependency 1개. 기존 env는 일부 import만 되면 YAML 변경과 무관하게 재사용된다. RTMScore/PSICHIC는 checkout commit/model hash 없이 파일 존재만 검사한다.
- Impact: 같은 Git commit도 solver 시점과 기존 머신 상태에 따라 다른 계산을 수행한다.
- 수정: platform별 explicit/conda lock과 pip hash lock을 실제 installer가 사용하고, env YAML/lock hash 및 external model/checkout hash를 qualification/run manifest에 결속한다.

### H-29. Stage 0 cutoff와 대형 mirror 무결성 계약이 불충분

- 근거: `scripts/stage0_fetch_training_cutoff.py:171-194`, `scripts/stage0_verify.py:235-246`, `workflow/rules/stage0_infra.smk:528-554`, `scripts/stage0_download_alphafold.sh:44-51`, `scripts/stage0_download_hpa.sh:26-39`, `scripts/stage0_mirror_chembl.sh:21-30`, `scripts/stage0_mirror_bindingdb.sh:144-176`
- 증거: **직접 코드**
- 문제: 기존 FASTA는 row count만 맞으면 다른 cutoff/provenance여도 재사용하고 새 manifest를 붙일 수 있다. AlphaFold는 4 GiB 크기, HPA/ChEMBL은 존재, BindingDB는 다운로드 후 계산한 TOFU hash 위주다.
- Impact: leakage audit universe, archive, extracted DB가 요청 release/cutoff와 다르면서 complete marker를 받을 수 있다.
- 수정: 기대 release hash/size/schema/row와 cutoff/source/output hash manifest를 DAG 필수 input/output으로 만들고 temp download→검증→atomic promotion한다.

### H-30. 자동 CI·coverage·static release gate가 없음

- 근거: `.github/workflows` 부재, `pyproject.toml`/`pytest.ini`/`.coveragerc`/tox 부재, coverage/ruff/mypy/bandit/shellcheck/eslint/stylelint 미설치
- 증거: **저장소·환경 조사**
- Impact: 2,700여 테스트가 로컬에서 통과해도 PR/release마다 수행된다는 보장이 없고 line/branch coverage도 증명할 수 없다.
- 수정: lock된 지원 환경의 CI에 pytest, workflow parse, compileall, shellcheck/lint/type/security, coverage threshold, browser E2E, Compose policy를 필수 gate로 추가한다.

## 6. Medium findings

| ID | 문제 및 근거 | Trigger / Impact | 수정 방향 |
|---|---|---|---|
| M-01 | AutoGrid 실행은 `AD4_parameters.dat` **또는** `AD4.1_bound.dat`를 수용하지만 readiness는 primary만 요구. `scripts/model_readiness.py:92-137`, `scripts/stage3_autogrid_maps.py:45-50,267-281`, `scripts/tests/test_model_readiness.py:145-181,463-475` | bound-only 정상 설치가 실행 가능해도 preflight에서 차단 | `primary or bound`를 실제 선택 파일로 반환하고 bound-only 행동 테스트 추가 |
| M-02 | JSON/숫자 경계에서 `NaN`/`Infinity` 허용. `skinscout/contracts/run_profile.py:243-250,656-693`, `scripts/summarize_run_outputs.py:978-985,1556-1560`, `workbench/server.py:2692-2709` | 비표준 JSON과 무한 score가 validation/sort/consumer에 전파 | `parse_constant` 거부, `math.isfinite`, `allow_nan=False` 전 구간 적용 |
| M-03 | 여러 writer가 예측 가능한 고정 `.tmp` 사용. `scripts/run_skinscout.py:189-197`, `scripts/summarize_run_outputs.py:1556-1560,1910-1914`, `scripts/verify_run_outputs.py:4993-4998`, `scripts/data_readiness.py:694-699`, `scripts/pipeline_readiness.py:1489-1494` | 동시 run/잔여 temp가 충돌·덮어쓰기 | 같은 디렉터리의 unique temp, fsync, atomic replace, cleanup |
| M-04 | workflow log가 configurable result root와 무관하게 `results/logs` 고정. `workflow/Snakefile:253-257`, `workflow/rules/stage2_admet.smk:17-18,42-43`, `workflow/rules/stage7_md.smk:22-23,67-68` | custom root에서 결과와 log/provenance 분리 | 공통 resolved root로 모든 log/output 생성 |
| M-05 | SDF upload가 lease/GPU/Popen 실패 시 남고 0600/0700 계약 없음. `workbench/server.py:1719-1808,2416-2421,2483-2503` | 동일 run 동시 요청, 시작 실패, 민감 구조 파일 잔류 | start 전체를 try/finally, 실패 cleanup, 디렉터리 0700·파일 0600, unique staging |
| M-06 | oversized body와 내부 I/O 오류를 주로 400으로 평탄화하고 raw error 노출. `workbench/server.py:2700-2701,3133-3140,3198-3200` | client가 413/409/500을 구분 못하고 내부 경로/사유 노출 가능 | typed exception→정확한 HTTP status, 외부용 안전 메시지와 내부 log 분리 |
| M-07 | 진행 표시가 실제 stage와 무관하게 running이면 고정 stage. `workbench/static/app.js:357-363` | 장시간 run에서 사용자 판단/취소 시점 오도 | coordinator event/stage artifact 기반 progress만 표시; 모르면 indeterminate |
| M-08 | Start 가능 여부가 empty/invalid input을 충분히 반영하지 않고 “구조 확인”은 SMILES/파일명만 표시. `workbench/static/app.js:700-789,907-928`, `workbench/static/index.html:162,174-175` | 잘못된 입력이 submit 시점까지 지연되고 분자 구조 확인 불가 | client+server 동일 schema validation, RDKit 기반 2D preview 또는 명칭 수정 |
| M-09 | 5초 polling의 `loadRuns()`가 target filter/sort control은 유지하면서 row를 기본값으로 초기화. `workbench/static/app.js:672-697,954-965` | 사용자가 보고 있던 정렬/필터가 자동으로 풀림 | UI state를 source of truth로 두고 refresh 후 동일 query 재적용 |
| M-10 | filter/sort 후 `display_rank`를 1부터 재번호화하여 원래 global rank를 “순위”로 오표시. `workbench/server.py:2124-2147,2184-2193`, `workbench/static/app.js:403-417` | 원래 2·5위가 화면에서 1위로 보임 | original rank와 filtered position을 별도 열/label로 표시 |
| M-11 | target CSV parsing 오류를 빈 목록과 동일 취급하고 UI가 오류를 버림 | malformed/missing column이 “검색 결과 없음”으로 보임 | typed parse error와 empty result 분리, UI error banner와 artifact line 표시 |
| M-12 | missing numeric score를 `Infinity`로 치환해 descending sort 시 상단에 올 수 있고 viewer async load race 존재. `workbench/static/app.js`, `scripts/report_assets/fast_report.js:103-140,198-207` | score 누락 행이 최고값처럼 보이거나 빠른 target 전환 시 이전 구조가 덮음 | missing-last comparator, request generation token/abort controller |
| M-13 | viewer metric tuple의 direction을 무시하고 source-column 문자열을 표시하며 receptor-missing flag를 계산만 하고 무시. `scripts/make_results_viewer.py:350-360,436,467-493`; `_read_tsv` `54-60`은 오류를 삼킴 | lower/higher 및 Boltz transform이 틀리고 missing receptor가 blank viewer로 진행 | typed metric schema, unit/direction formatter, required receptor fail, parse error 노출 |
| M-14 | Stage 9 계약의 histogram/scatter/radar가 표와 raw JSON으로 대체. `scripts/stage9_report.py:4,1716-1719,2990-3094` | 분포·불일치 관계를 시각적으로 검토할 수 없음 | 실제 numeric plot 구현 또는 panel/문서 명칭을 현재 출력에 맞게 축소 |
| M-15 | README의 “실제” quick-start PNG는 magic/size 존재만 테스트되고 source run/hash와 결속되지 않음. `README.md:92,160,182`, `scripts/tests/test_beginner_bootstrap.py:75-86` | stale/수동 편집 screenshot이 최신 UI 증거처럼 남음 | capture script, fixture run ID, commit/hash, browser viewport를 manifest로 저장 |
| M-16 | SkinScore는 축 누락 시 남은 축으로 재정규화하여 tier 의미가 snapshot마다 달라짐. `scripts/stage0_skin_score.py:38-44,361-409,412-495,537-552`, `scripts/stage0_verify.py:1543-1544,1695-1699` | 현재 일부 data가 header-only이고 axes sidecar도 없어 동일 단백질 score 비교 불안정 | axis schema별 calibration/version과 axes manifest를 verifier/downstream provenance에 필수화 |
| M-17 | PSICHIC 긴 단백질은 window 최대값만 사용. `scripts/stage3_psichic.py:268-319`, `scripts/tests/test_stage3_psichic_fail_closed.py:273-281` | window 수가 많은 장단백질이 극값 편향을 받을 가능성 | 검증된 long-sequence aggregation, window count 출력, 길이 strata 평가 |
| M-18 | `score_performance_v2 --device cpu`가 SMILES embedding에는 적용되지 않음. `scripts/score_performance_v2.py:55-74,118-125,148-179`, `eval/performance_v2_model.py:1321-1345,1399-1416,2106-2139` | visible GPU에서 query embedding만 CUDA 사용, OOM/placement/provenance 불일치 | requested device를 embedding까지 전달하고 effective device/fallback 기록 |
| M-19 | cold-start target set을 training cutoff가 아닌 최신 전체 ChEMBL snapshot으로 결정. `eval/cold_start_eval.py:159-184,219-239` | post-cutoff activity가 denominator/panel 포함 여부를 바꾸는 selection leakage | cutoff 이전 target universe를 봉인하고 최신 activity는 진단값으로만 사용 |
| M-20 | diagnostic allow-missing에서 coverage가 불완전해도 `passes_threshold=true`. `eval/skin_known_target_recovery_eval.py:563-580,687-728,890-905` | JSON 단독 consumer가 incomplete panel을 pass로 오해 | `status=diagnostic_incomplete`, `passes_threshold=false`, minimum coverage gate |
| M-21 | P2Rank archive만 hash 검증하고 기존 설치 tree/launcher는 재사용. `scripts/setup_p2rank.sh:33-75`, `scripts/tests/test_setup_tool_pinning.py:20-28` | 설치 tree가 변경·오염돼도 검증된 archive 존재와 무관하게 계속 실행 | 안전 재추출 후 atomic replace 또는 설치 tree hash manifest 검증 |
| M-22 | Compose는 placeholder image로 의도적 fail-closed이나 설치/prereq/env 계약이 자기완결적이지 않음. `compose/skinscout.compose.yaml:5,27,45`, `compose/image-policy.json:8-26`, `scripts/bootstrap_runtime.sh:137-178`, `install_skinscout.sh:159-162,225` | 현재 Compose는 실행 불가이고 missing Docker/NVIDIA/cosign을 installer가 해결하지 않으며 reboot-required도 완료처럼 보일 수 있음 | 실제 release digest/OIDC/SBOM, pending 상태 exit contract, 공통 `SKINSCOUT_RESULTS_ROOT` 적용 |
| M-23 | 운영 스크립트가 과거 절대경로·고정 수신자를 사용하고 SMTP 실패도 성공처럼 종료. `scripts/monitor_stage0.sh:11-14`, `scripts/report_email.sh:6,8-11,31-60` | 다른 repo를 감시/보고, host/IP/hardware 외부 전송, msmtp 실패 후 “sent”/exit 0 | repo root/config 기반 경로, recipient/diagnostics opt-in, CRLF 거부, 명시적 pipeline 실패 처리 |
| M-24 | 다운로드 resume/timeout/retry 계약이 불완전. `scripts/stage0_mirror_chembl.sh:21`, `scripts/install_runtime.py:482`, `scripts/setup_p2rank.sh:38` | 부분 ChEMBL 파일은 존재만으로 재개를 건너뛰고 network stall 가능 | `.part`에 `-c`, timeout/backoff, checksum+archive integrity 후 atomic promote |

## 7. Low findings

| ID | 문제 | 근거 | 권고 |
|---|---|---|---|
| L-01 | README는 `--host 0.0.0.0` 가능성을 설명하지만 서버는 non-loopback을 거부 | `README.md:287`, `workbench/server.py:3204` | “미지원”으로 수정하거나 인증된 remote mode 구현 |
| L-02 | tab/dialog/sort table의 keyboard/ARIA semantics가 불완전 | `workbench/static/index.html`, `workbench/static/app.js`, `workbench/static/styles.css` | `aria-current`, tab roles/selection, dialog label/focus trap, keyboard sortable header 추가 |
| L-03 | HTTP Range 요청마다 전체 artifact SHA-256을 다시 계산 | `workbench/server.py:2785-2819` | promotion 때 검증한 immutable inode/size/mtime 또는 cached digest로 serve 전 확인 |
| L-04 | 핵심 함수가 매우 크고 coordinator에 return 뒤 도달 불가능 fsync가 있음 | `scripts/verify_run_outputs.py:2400`, `eval/run_iteration.py:2546`, `workbench/server.py:396`, `workbench/coordinator.py:1094` | 입력 검증/변환/판정/직렬화를 순수 함수로 분리하고 fsync 순서 수정 |
| L-05 | 아키텍처 문서가 Stage10/v2에 머물고 100개 이상 schema-like ID에 중앙 registry/migration 표가 없음 | `docs/ARCHITECTURE.md:1-47`, `schemas/`, 전 코드의 `skinscout.*.vN` 식별자 | producer→consumer→validator→migration registry와 현재 DAG 기반 문서 자동생성 |
| L-06 | Snakemake lint가 환경/log/긴 run block 등 다수 경고로 exit 1 | 예 `workflow/rules/stage0_infra.smk:41,60`, `workflow/Snakefile:1473` | rule별 env/container/log 선언, hardcoded prefix 제거, 긴 logic을 script/module로 이동 |

## 8. 모든 Input 검토

### 8.1 입력 경계 매트릭스

| 입력 종류 | 주요 producer/entry | 현재 강점 | 결함/위험 |
|---|---|---|---|
| CLI/config | `run_skinscout.py`, Snakefile, stage scripts | 다수 choice/min/max, shell `{...:q}`, `shlex.quote` 사용 | profile/readiness/DAG 불일치(H-01/H-07), float→int MD(H-06), safety 상한 누락(H-17), dead config |
| SMILES/SDF upload | Workbench, `compound_applicability.py` | multi-record·malformed trailing record·curated UV filter를 fail-closed | start failure orphan/permission(M-05), client preview/validation(M-08) |
| JSON | manifests, readiness, summary, API | 다수 schema ID와 명시 validator | NaN/Infinity(M-02), manifest optional(H-03), raw HTML insertion(H-23) |
| CSV/TSV ranking | Stage 0–11, eval, viewer | duplicate/blank/required column을 많은 경로에서 거부 | global rank 무시(H-14), parse error→empty(M-11), viewer target/path/metric(H-21/M-13) |
| Filesystem path | run root, report package, artifact API | static/artifact resolver와 report package는 traversal/symlink 방어가 강함 | verifier symlink(H-09), viewer output traversal(H-21), root split(H-10), output=input deletion(H-11) |
| Network data | RCSB, AlphaFold, HPA, ChEMBL, BindingDB | 일부 installer binary는 expected SHA+temp+atomic replace | cutoff/release/archive 결속 부족(H-29), retry/resume(M-24) |
| Runtime/tool | conda env, external checkout, GPU | 일부 smoke/version readiness, AutoDock source commit/dirty 확인 | wrong-env fallback(H-08), env/model unlock(H-28), DiffDock 누락(H-07) |
| Browser data | report JSON, PDB/SDF, API polling | Workbench text는 대체로 `escapeHtml()` 사용 | Stage9 injection(H-23), file CORS(H-20), async stale(M-12), unaligned overlay(H-24) |
| Evaluation input | ranking, truth, preregistration, metrics | activity benchmark의 tie rank와 coverage는 비교적 명시적 | self-reported promotion(C-02), subset rank(H-14), snapshot leakage(M-19) |

### 8.2 입력별 필수 수정 순서

1. 단일 `RunProfileCapability` schema로 CLI/config/readiness/DAG 요구를 생성한다.
2. 모든 path는 resolved containment와 symlink policy를 명시하고 input/output inode alias를 거부한다.
3. JSON parser/serializer에 finite-number 정책을 일괄 적용한다.
4. run/eval/data manifest에 실제 소비한 파일·code·config·env·model hash를 묶는다.
5. scientific quantity는 `value + unit + direction + provenance + schema_version`을 강제한다.

## 9. 모든 Output 검토

### 9.1 출력 경계 매트릭스

| 출력 | 현재 생성/검증 | 판정 |
|---|---|---|
| `run_summary.json` | 계산 뒤 verifier 전에 생성; 다수 semantic validation | **성공 상태의 권위로 사용 금지**. C-01 |
| `run_verification.json` | content check가 강한 부분 존재 | manifest와 sealed package가 일반 실행에서 필수가 아님. H-03/H-04 |
| coordinator artifact | workspace validation+atomic promotion 우수 | 정상 local Workbench가 우회. H-04 |
| CSV/TSV score | 다수 required column/finiteness 검사 | Boltz unit 혼합, rank 의미 상실, KG seed count. H-12/H-14/H-15 |
| MD/QM output | fail-closed parser/test 존재 | 기본 MD 시작 실패, config dead, 전자상태/conformer 불일치. H-06/H-18/H-19 |
| HTML report | sealed package validator 별도 존재 | flat marker만 completion 검사, stored injection, chart 미구현. H-04/H-23/M-14 |
| Mol* viewer | HTML과 per-target page 생성 | 현재 전면 실패, path traversal, metric/identity 오류. H-20/H-21/H-25/M-13 |
| Stage11 figure | PNG/SVG 및 caption manifest | 과학 plot 대신 artifact counts. H-22 |
| retrospective PNG | 저장소에 3개 tracked | 입력 run CSV 미추적, 재생성 실패, 과도한 caption. H-26 |
| API status/progress | list/detail/card/progress | verification truth와 분리, 가짜 progress, rank 재번호화. C-01/M-07/M-10 |
| email/ops | shell output과 exit status | 전송 실패를 성공으로 보고 가능. M-23 |

### 9.2 Atomicity·동시성·무결성

- 긍정: `workbench/coordinator.py`의 attempt artifact promotion과 `report_package_contract.py`의 package validation은 path, symlink, manifest, checksum, undeclared file을 강하게 방어한다.
- 문제: 그 강한 경로가 일반 Workbench 성공 흐름에서 사용되지 않는다(H-04).
- 고정 `.tmp` writer는 동일 output에 대한 동시 실행 계약이 없다(M-03).
- `run_summary`/viewer를 verification 이전에 게시하는 순서는 partial success를 terminal success로 노출한다(C-01).
- Range serve는 무결성 재검사 비용이 파일 크기에 선형으로 반복된다(L-03).

## 10. 시각화 구현 전수 검토

### 10.1 구현별 판정

| 시각화 | 실제 상태 | 사용자/과학 영향 | 판정 |
|---|---|---|---|
| Workbench readiness/start | 기본 UI는 읽기 쉬우며 text escaping 양호 | profile 요구가 실제 DAG와 다르고 입력 preview가 구조가 아님 | BLOCKED by H-01/H-07/M-08 |
| Run cards/status | polling과 card 구현 존재 | failed verification도 completed/claimable 가능 | CRITICAL C-01 |
| Progress rail | stage UI 존재 | 실제 stage가 아니라 고정 진행 | MISLEADING M-07 |
| Target table | 검색/필터/정렬 구현 | refresh가 state를 풀고 original rank를 재번호화 | WRONG STATE/RANK M-09/M-10 |
| Fast report Mol* | receptor/ligand load 코드 존재 | free ligand를 unaligned overlay, async race, seal 약함 | UNSAFE H-24/H-25/M-12 |
| Offline results viewer | per-target Mol* page 생성 | API 오류, file CORS, path traversal, metric direction 오류 | BROKEN H-20/H-21/M-13 |
| Stage 9 A2/A3 등 | HTML table/JSON 제공 | histogram/scatter/radar라는 계약 미충족 | INCOMPLETE M-14 |
| Stage 11 figures | source-count diagnostic plot | scientific filename/title/status와 내용 불일치 | NOT PUBLICATION READY H-22 |
| Retrospective fig1 | red/green demotion panel | 색각 접근성 부족, “systemic target” 추론이 score 비교만으로 도출 | STALE/OVERCLAIM H-26 |
| Retrospective fig2 | target score labels | label overlap을 실제 PNG에서 확인 | QUALITY DEFECT H-26/L-02 |
| Retrospective fig3 | histogram | zero-heavy log 시각화, source data 미추적 | NOT REPRODUCIBLE H-26 |
| README quick-start screenshots | 실제 크기의 PNG | result compound identifier overflow/wrap; source run/hash 미봉인 | STALE RISK M-15 |

### 10.2 브라우저 재현 증거

1. 실제 receptor/pose를 사용해 one-target viewer를 임시 디렉터리에 생성했다.
2. headless Chrome에서 canvas는 생성됐지만 `viewer.clear is not a function`이 발생하고 fallback이 표시됐다.
3. repo 파일을 수정하지 않고 임시 HTML에서 호출만 `viewer.plugin.clear()`로 바꿨다.
4. 이후 `file://` PDB/SDF fetch가 CORS로 차단되어 Mol* canvas가 blank였고, 이 오류는 fallback 상태와도 정확히 연결되지 않았다.
5. target ID `../../escaped` 입력으로 지정 output 밖 HTML 생성이 재현됐다.

따라서 단순 DOM 존재/문자열 검사나 canvas 색상 비교는 “구조가 올바르게 로드·표현됐다”는 완료 증거가 아니다.

### 10.3 시각화 수용 기준

- 실제 supported browser E2E에서 receptor/ligand source hash, atom/component count, screenshot hash를 함께 검증.
- `file://` offline을 지원하려면 data embedding 또는 browser가 허용하는 API를 사용. 그렇지 않으면 “offline” 명칭을 제거하고 localhost serve 명령 제공.
- 모든 chart는 데이터 unit, direction, missing policy, sample count, source hash를 figure manifest에 기록.
- original global rank와 filtered/display position을 구분.
- publication figure는 artifact-count diagnostic을 넘어서 실제 scientific variables를 그린 경우에만 `source_backed/publication_ready` 허용.
- 적록 단독 encoding 금지, label collision test, keyboard/ARIA 및 reduced-motion 기준 추가.

## 11. 검증 결과

### 11.1 실행한 검증

| 검증 | 결과 |
|---|---|
| 독립 test-engineer가 PATH를 지원 `cosmax-base/bin`으로 명시해 실행한 전체 `scripts/tests` | **2714 passed, 3 skipped, 0 failed** |
| 같은 Python만 사용하고 PATH를 보정하지 않은 전체 suite | **2681 passed, 36 skipped, 0 failed** |
| 최신 HEAD의 변경 관련 readiness/guide 테스트 | **31 passed** |
| architecture 계약 선별 | **75 passed** |
| visual/UI 선별 | **51 passed** |
| pipeline/eval 선별 | **327 passed** |
| Stage 7/9/11/KG/Workbench 추가 선별 | **319 passed** |
| Production Python `compileall` | PASS |
| Shell 17개 `bash -n` | PASS |
| JSON/YAML parse | PASS |
| Snakemake lint | workflow parse 후 다수 lint warning으로 exit 1 |
| Compose default doctor | exit 2; placeholder images 3개를 의도대로 fail-closed |
| MD 기본 `50.0` CLI | prep/run 양쪽 parsing 실패 재현 |
| Retrospective figure 재생성 | 필수 run CSV 부재로 실패 재현 |
| 실제 run viewer generation | comprehensive consensus 부재로 실패 재현 |
| one-target Chrome viewer | API 오류 및 CORS blank 재현 |
| viewer target traversal | output root 탈출 HTML 생성 재현 |

PATH 차이는 중요하다. `scripts/tests/test_workflow_config_fail_closed.py:20-30`이 Snakemake를 찾지 못하면 33개 workflow 검증을 skip한다. 따라서 릴리스 canonical test command는 지원 env의 `bin`을 PATH에 명시하거나 필수 tool 부재를 실패로 처리해야 한다.

### 11.2 테스트가 많이 통과해도 판정이 BLOCK인 이유

현재 suite는 개별 parser/validator와 기존 동작 회귀에 강하다. 이번 Critical/High의 상당수는 다음과 같은 **합성 경계**에 있다.

- profile contract는 통과하지만 readiness와 DAG가 다름.
- coordinator promotion test는 통과하지만 정상 Workbench가 그 경로를 안 씀.
- viewer HTML test는 통과하지만 실제 Chrome에서 구조가 안 뜸.
- promotion CSV test는 통과하지만 원시 데이터를 재검산하지 않는 현재 동작을 고정함.
- model readiness test는 통과하지만 unrelated conda env를 ready로 인정하는 잘못된 계약을 고정함.
- Stage 8 test는 single-frame fixture만 써 multi-conformer 손실을 발견하지 못함.

## 12. 검증하지 못한 범위와 잔여 위험

- 약 200 GB Stage 0 전체 재다운로드·재구축은 실행하지 않았다.
- GPU comprehensive/Boltz/DiffDock/AutoDock-GPU end-to-end, 장시간 BioEmu/MD/QM는 실행하지 않았다.
- 실제 서버 kill/restart를 통한 orphan/duplicate process 실험은 하지 않았다. H-02는 process ownership 코드에 근거한 높은 확신의 추론이다.
- released Compose image가 없으므로 real Docker/GPU/Cosign/SBOM E2E는 불가능했다. 현재 placeholder는 정상적으로 차단된다.
- `node`가 설치되어 있지 않아 독립 `node --check`는 실행하지 못했다. 브라우저 JS는 targeted test와 headless Chrome 실행으로 일부 대체했다.
- LSP/pyright/mypy/ruff/bandit/shellcheck/eslint/stylelint가 설치되어 있지 않아 해당 정적 분석은 수행하지 못했다. `compileall`은 type/LSP 대체가 아니다.
- coverage 도구와 기준이 없어 line/branch coverage percentage는 증명할 수 없다.
- vendor Mol* minified 내부 알고리즘은 upstream audit 범위이며 SkinScout integration boundary만 검토했다.

## 13. 긍정적으로 확인된 방어와 기각한 오탐

다음은 실제로 양호하거나 과거 문제가 해소되어 finding에서 제외했다.

- production 경로에서 `shell=True`, `os.system`, 동적 `eval/exec` 명령 실행은 발견되지 않았다.
- 주요 shell/Snakemake 인자는 배열, `{...:q}`, `shlex.quote`를 대체로 올바르게 사용한다.
- Workbench static/artifact path guard와 일반 UI `escapeHtml()` 사용은 대체로 강하다.
- `report_package_contract.py`는 pointer traversal, symlink, checksum, identity, undeclared file을 강하게 거부한다. 문제는 일반 실행이 이 계약을 완료 조건으로 사용하지 않는다는 점이다.
- multi-record SDF, malformed trailing record, curated UV filter는 `compound_applicability.py`에서 fail-closed다.
- `install_runtime.py`의 일부 binary download는 expected SHA-256, temp file, atomic replace를 사용한다.
- AutoGrid archive extract는 path escape와 symlink/hardlink를 사전 거부한다.
- AutoDock-GPU source는 full commit, resolved HEAD, dirty/untracked 상태를 검증한다.
- Compose placeholder digest는 숨겨진 mutable-image 문제가 아니라 문서화·테스트된 release blocker다.
- CosIng의 공개 SPA search key는 repository 주석과 사용 형태상 비밀키 유출로 판정하지 않았다.
- comprehensive GNINA/RTMScore는 현재 AutoDock pose manifest/directory를 소비하여 과거 free-ligand lineage 문제는 재현되지 않았다.
- 과거 validation panel의 잘못된 receptor 구조는 현재 수정됐으며, PAINS `drop` no-op도 config에서 `warn`만 허용하도록 제한됐다.

## 14. 수정 우선순위와 완료 조건

### P0 — 릴리스 전 반드시 완료

1. 권위 있는 상태 머신을 도입해 C-01을 제거한다.
2. prospective promotion을 raw sealed evidence 재계산으로 바꿔 C-02를 제거한다.
3. `RunProfileCapability` 단일 계약으로 H-01/H-05/H-07/H-08을 함께 해결한다.
4. 모든 실행 manifest와 sealed artifact promotion을 필수화해 H-03/H-04/H-09를 해결한다.
5. 기본 report MD, Boltz unit/crop, safety threshold, CREST/conformer 처리를 수정한다.
6. viewer의 offline contract, traversal, identity verification을 실제 브라우저 E2E로 잠근다.
7. KG seed/evidence 타입과 publication/retrospective 산출물을 과학적으로 교정한다.

### P1 — 재현 가능한 배포 전 완료

1. env/data/model/installer를 hash-locked trust chain으로 만든다.
2. Stage 0 source cutoff/archive manifest를 mandatory DAG artifact로 만든다.
3. result/log/upload/temp path를 공통 root와 unique atomic write 계약으로 통일한다.
4. CI에 canonical PATH, full pytest, browser E2E, workflow lint, static checks, coverage gate를 추가한다.

### 수용 가능한 최종 상태

- verification 실패 run이 어떤 API/UI에서도 completed/claimable로 보이지 않음.
- 각 profile의 readiness와 실제 DAG required rules/env가 자동 비교되어 불일치 0.
- 기본 YAML의 report-fast 및 지원되는 report profile이 fresh environment에서 E2E 통과.
- 모든 성공 run이 code/config/data/env/model/input hash가 포함된 mandatory manifest와 registered artifact를 가짐.
- viewer/report가 실제 browser에서 source identity와 atom/component를 검증하고 package manifest에 seal됨.
- prospective promotion metric이 raw sealed data에서 재계산되고 preregistration/model hash mismatch를 거부.
- Stage11 figure와 retrospective가 tracked source bundle에서 byte-reproducible하게 재생성됨.
- 지원 PATH의 전체 suite, lint/type/security/browser/coverage/Compose release gates가 CI에서 통과.

## 15. 최종 판정

**REQUEST CHANGES / RELEASE BLOCK**

현재 상태는 “많은 개별 검증기가 있는 연구용 파이프라인”으로서는 강점이 크지만, **성공 상태·재현성·과학적 단위·시각적 증거·model promotion을 end-to-end로 신뢰할 수 있는 제품 계약**에는 도달하지 못했다. 특히 C-01과 C-02가 남아 있는 동안 결과를 사용자 성공, claim-ready, publication-ready 또는 model promotion-ready로 표현해서는 안 된다.
