# SkinScout 전 코드·논리·결과·시각화·해석 정밀 읽기 전용 감사

**감사일:** 2026-08-26 (Asia/Seoul)  
**최초 감사 기준:** `190ac2ba6c79bfebc9f36370622f672c75b32c84`  
**보고 시점 HEAD:** `53b63364cdfe5b7091f85a54da2e837c0367df60`  
**감사 성격:** 코드·설정·문서·결과·시각화에 대한 읽기 전용 독립 검토  
**변경 범위:** 이 보고서만 추가했다. 소스, 설정, 데이터, 결과, 그림, 테스트는 수정하지 않았다.

---

## 1. 결론

SkinScout는 단순한 프로토타입보다 훨씬 강한 실패 폐쇄(fail-closed), 해시 결속, 입력·경로 검증, 결과 비주장(non-claimable) 통제를 갖췄다. 그러나 현재 상태를 과학적 성능, 출판 품질, 재현 가능한 릴리스로 승인할 수는 없다.

가장 중요한 이유는 다음 네 가지다.

1. **검증 도구 자체가 잘못되어 있었다.** 최초 기준 v1 알려진 표적 패널은 Retinol, Ascorbic acid, Kojic acid, EGCG의 구조가 이름과 달랐고, 32개 화합물–표적 쌍 중 11개가 영향을 받았다. 감사 중 외부 동시 작업이 v2 구조·계약·검증 문서를 `53b63364…`로 커밋했다. 그러나 새 평가 결과는 여전히 Git ignored 상태이고 activity recovery, ChEMBL pair audit, 2026-08-22 bundle 등 하류 증거는 v1을 참조한다.
2. **Comprehensive의 구조 점수 의미가 구현과 다르다.** AutoDock가 실제 pose를 내보내지만 GNINA와 RTMScore 규칙은 그 pose가 아니라 자유 ligand SDF를 다시 사용한다. 따라서 현재 `top50_4way_consensus.csv`를 “동일 AutoDock pose에 대한 4-way 구조 합의”로 해석할 수 없다.
3. **주장과 그림의 계보가 끊긴 부분이 있다.** 커밋된 원고·회고 그림은 현 파이프라인보다 오래된 Vina 세대이고, 원본 demo CSV는 Git에 없어 재생성되지 않는다. Stage 11의 “source-backed” 그림은 캡션이 약속한 과학량이 아니라 행 수·키 수·바이트 수를 그린다.
4. **최신 완료된 v1 과학 게이트가 명시적으로 실패한다.** `claim_ready=false`, `passes_frozen_test_gate=false`다. temporal 성능은 개선되지만 v1 known-skin Top30은 감소했고 dual-cold Top10/Top30은 모두 0이다. v2 전체 activity gate는 미재실행이라 Unknown이며, v2 Daina LQO도 자체 threshold를 통과하지 못했다. v1 운영용 baseline 유지 PASS는 성능·임상·SOTA 승인과 무관하며 v2 현행 PASS도 아니다.

### 1.1 최종 판정표

| 영역 | 판정 | 근거 수준 | 핵심 이유 |
|---|---|---|---|
| 엔지니어링 릴리스 | **REQUEST CHANGES / 조건부** | Evidence | 정적 검사는 통과했지만 Comprehensive pose 배선, 입력 적용 범위, 구성–실행 불일치, 준비성 검사와 inference device 계약 결함이 남았다. |
| 과학적 주장 | **FAIL** | Evidence + Unknown | 최신 완료 v1 summary는 `claim_ready=false`, `passes_frozen_test_gate=false`이고 v2 Daina LQO도 threshold FAIL이다. v2 전체 activity 비교는 미재실행이므로 Unknown이며, 이를 PASS로 간주할 근거가 없다. |
| 결과 산술 | **부분 VERIFIED** | Evidence | 아래 명시한 v1 표의 분자·분모·MRR은 원시 CSV에서 독립 재계산과 일치한다. 그러나 잘못된 구조가 포함된 이름의 v1 값은 해당 실제 화합물의 성능 근거가 아니다. |
| 라벨·근거 계보 | **INCOMPLETE** | Evidence + Unknown | 패널은 구조화 DOI/PMID 없이 자유문장 `evidence_note`를 사용한다. 기존 v1 ChEMBL 37 감사는 32쌍 중 7쌍만 직접 지지했지만, 구조가 정정된 v2의 직접 지지 수는 재감사 전까지 Unknown이다. |
| 시각화·출판 | **FAIL** | Evidence | Stage 11 그림의 내용이 캡션과 다르고, 커밋 그림은 오래된 세대이며 원자료가 추적되지 않는다. |
| 재현성 | **FAIL (보고 시점 스냅샷)** | Evidence | 작업 트리에 별도 applicability 보완 초안과 이 보고서가 남아 있고, v2 전체 suite의 한 실행은 `2,600 passed / 39 skipped / 1 failed`였다. 동일 실패는 이후 CPU-hidden 및 최신 GPU-visible targeted 재실행에서 모두 통과했으나, 메모리 상태 의존 OOM을 허용하는 inference device-contract 공백과 깨끗한 릴리스 재검증 부재는 남는다. |
| 보안·개인정보·라이선스 | **조건부 / 고위험** | Evidence + Unknown | 로컬 Workbench 통제는 강하지만 외부 안전성 서비스에 SMILES를 전송하고 약관이 미확정이며, 프로젝트 루트 LICENSE가 없고 설치·Compose 릴리스가 미고정이다. |
| v1 운영 baseline 유지 게이트 | **HISTORICAL PASS / v2 Unknown** | Evidence + Unknown | v1에서는 `chembl_p5_max`를 유지하도록 정상 작동했다. 이는 실패한 후보를 승격하지 않은 과거 운영 통제이며, v2 현행 게이트나 과학적 PASS가 아니다. |

### 1.2 사용 금지 해석

현재 산출물을 다음으로 표현하면 안 된다.

- 임상적 안전성 또는 효능 증명
- 실제 결합 여부 또는 결합 자유에너지의 확정
- 표적 확률, 피부 선택성 또는 작용기전의 확정
- SOTA, 검증 완료, 출판 준비 완료
- ADMET endpoint 음성을 근거로 한 안전성 허가
- HPA 발현을 근거로 한 피부 특이적 작용 또는 세포 선택적 작용

허용 가능한 최대 표현은 **“공개·계산 근거를 결합한 실험 우선순위용 표적 가설”**이다. 이 문장은 `README.md:117-119,394-400`의 보수적 경계와 일치한다.

---

## 2. 감사 기준, 증거 규칙, 한계

### 2.1 스냅샷과 동시 변경 분리

| 시점 | 상태 | 감사에서의 취급 |
|---|---|---|
| 최초 기준 | HEAD `190ac2b…`, clean, 추적 파일 403개 | 본 감사의 1차 커밋 기준 |
| 중간 외부 변경 | 패널 digest가 `cce7…`로 바뀌었으나 코드는 v1 `1e0d…`를 기대 | 전체 테스트 10건 실패의 원인. 실패 폐쇄가 drift를 포착한 긍정 증거 |
| v2 확정 커밋 | HEAD `53b63364…`, panel digest `e76a265d…`, 추적 파일 405개 | 구조·계약·정정 문서는 커밋됨. 새 ignored 결과 및 v1 하류 성능과 구분 |
| 보고 시점 외부 초안 | applicability/UI 관련 추적 파일 10개 수정, 이 보고서 미추적 | 미확정 작업 트리 evidence로만 검토. `53b63364…`의 확정 동작과 혼합하지 않음 |

v2 정정은 `53b63364cdfe5b7091f85a54da2e837c0367df60`(`fix: correct four wrong structures in the preregistered validation panel`)에 포함됐다. 보고 시점에 이 감사가 만들지 않은 **미커밋** 변경은 다음과 같다.

- 수정: `scripts/compound_applicability.py`
- 수정: `scripts/run_skinscout.py`
- 수정: `scripts/summarize_run_outputs.py`
- 수정: `scripts/tests/test_compound_applicability.py`
- 수정: `scripts/tests/test_pipeline_readiness.py`
- 수정: `scripts/tests/test_run_skinscout.py`
- 수정: `scripts/tests/test_verify_run_outputs.py`
- 수정: `workbench/server.py`
- 수정: `workbench/static/app.js`
- 수정: `workbench/static/index.html`
- ignored 결과: `results/eval/skin_known_full_20260826/`

이 보고서 자체는 미추적 신규 파일이다. 따라서 보고 시점 작업 트리는 **깨끗하고 불변인 릴리스 스냅샷이 아니다.** 이 감사는 외부 초안을 자체 수정으로 간주하지 않는다. v2 패널의 Git 정체성은 확정됐지만, 새 결과가 커밋 또는 content-addressed release로 봉인되고 모든 v1 하류 증거가 이관되기 전까지 “현 패널의 전체 성능 검증이 확정됐다”고 판정하지 않는다.

### 2.2 증거 표기

- **Evidence:** 저장소 파일, 해시, 실행 출력, 원시 결과를 직접 확인했다.
- **Inference:** 직접 증거에서 합리적으로 도출했으나 별도 실험으로 입증하지 않았다.
- **Unknown:** 필요한 외부 데이터, GPU 실행, 서비스 응답, 라이선스 계약 또는 깨끗한 환경 검증이 없다.

### 2.3 심각도

| 등급 | 의미 |
|---|---|
| Critical | 핵심 결과의 정체성·의미·출판 주장을 무효화하거나 잘못된 결론을 직접 만들 수 있음 |
| High | 일반 실행·재현성·사용자 해석을 크게 훼손하거나 릴리스를 차단해야 함 |
| Medium | 특정 경로에서 오동작·혼동·통제 공백을 만들지만 즉시 전체 결과를 무효화하지는 않음 |
| Low | 유지보수성·가독성·표현 정확도의 제한적 문제 |

### 2.4 범위와 정직한 제한

- 추적 파일 **403/403개를 유형·역할·주장 영향별로 disposition/inventory**했다.
- Python 301개는 AST/compile 대상으로 확인했고, shell 16개, JSON 9개, YAML 12개, 주요 JavaScript를 정적으로 검사했다.
- 심층 의미 검토는 모든 동작·주장·결과·시각화에 영향을 주는 핵심 표면을 대상으로 했다. vendored/minified 코드와 모든 테스트 보조 행을 줄 단위 의미 검토했다고 주장하지 않는다.
- GPU 모델, 외부 웹 endpoint, 123 GB 데이터 전체의 의미 정확성, 실제 새 컴퓨터 설치, 전체 E2E 과학 실행은 재수행하지 않았다.
- 의료·규제·법률 감사를 수행한 것이 아니다.

---

## 3. 우선순위 발견사항

| ID | 심각도 | 발견 | 상태/확신 |
|---|---|---|---|
| F-01 | Critical | v1 known-target 패널 4개 화합물 구조 오류, 11/32쌍 영향 | Evidence / 높음 |
| F-02 | Critical | Comprehensive GNINA·RTMScore가 exported AutoDock pose가 아닌 자유 ligand SDF 사용 | Evidence / 높음 |
| F-03 | Critical | KG `n_papers` seed 상수가 실제 문헌 수처럼 노출되고 label만으로 efficacy-supported 가능 | Evidence / 높음 |
| F-04 | Critical | Stage 11의 8개 “source-backed” 그림이 과학량 대신 artifact count를 그림 | Evidence / 높음 |
| F-05 | Critical | 커밋 원고·회고·그림이 오래된 Vina 세대이고 재생성 원자료가 Git에 없음 | Evidence / 높음 |
| F-06 | High | 최신 완료 v1 claim gate 실패; v2 Daina threshold 실패, v2 전체 activity gate는 미재실행 | Evidence + Unknown / 높음 |
| F-07 | High | v1 알려진 표적 32쌍 중 엄격 ChEMBL 37 직접 지지는 7쌍; v2 직접 지지 수는 미재감사 | Evidence + Unknown / 높음 |
| F-08 | High | 최초 기준의 CLI·SDF·Workbench gate 불일치는 미커밋 초안에서 부분 보완됐지만 UV-filter, 비정상 복수 SDF, 예외 cleanup·통합 검증 공백이 남음 | Evidence + Inference / 높음 |
| F-09 | High | `pains_action=drop`을 허용하지만 실제 drop 경로가 없음 | Evidence / 높음 |
| F-10 | High | MD 설정의 OpenFF Sage·timestep이 실제 ACPYPE·고정 2 fs 실행과 결속되지 않음 | Evidence / 높음 |
| F-11 | High | DiffDock가 Comprehensive DAG에 필수지만 readiness/env에서 완결되지 않음 | Evidence / 높음 |
| F-12 | High | SkinScore가 현재 사실상 HPA tissue/cell 두 축만으로 재정규화됨 | Evidence / 높음 |
| F-13 | High | 설치·Compose·환경·CI·라이선스가 배포 재현성 기준에 미달 | Evidence / 중간~높음 |
| F-14 | High | Stage 9 “histogram/scatter”가 표와 잘린 JSON으로 렌더링됨 | Evidence / 높음 |
| F-15 | Medium | Boltz crop/max-residues와 `msa_source`가 실행 의미에 반영되지 않음 | Evidence / 높음 |
| F-16 | Medium | `mode=both` 준비성 검사가 실제 두 경로 요구를 완전히 반영하지 않음 | Evidence / 중간 |
| F-17 | Medium | sensitization threshold가 모델 수 3보다 큰 값을 허용 | Evidence / 높음 |
| F-18 | Medium | verification fingerprint가 모든 주장 원시 근거를 해시하지 않음 | Evidence / 중간 |
| F-19 | Medium | Workbench가 내부 fixture/진단 summary를 “완료” 목록에 포함할 수 있음 | Evidence / 높음 |
| F-20 | Medium | Demo 카드의 “Daina + docking으로 순위화”가 annotation-only 실제 동작과 충돌 | Evidence / 높음 |
| F-21 | High | CPU-trained performance-v2 scoring에 device 선택권이 없어 CUDA를 자동 선택하며 메모리 상태 의존 OOM이 가능 | Evidence / 높음 |

---

## 4. 핵심 발견 상세

### F-01. v1 검증 패널의 화합물 정체성 오류

**Evidence.** 기준 커밋의 패널은 다음 이름–구조 불일치를 포함했다. 보고 시점 v2 정정 기록은 `docs/PANEL_STRUCTURE_CORRECTION_20260826.md:1-57`, v2 고정 계약은 `scripts/activity_recovery_contracts.py:12-60`에 있다.

| case | v1 실제 구조 | 의도 구조 | 영향 표적쌍 |
|---|---|---|---:|
| Retinol | farnesol, InChIKey prefix `CRDAMVZIKSXKFV` | all-trans-retinol, `FPIPGXGPPPQFEQ` | 3 |
| Ascorbic_acid | 다른 포화 lactone, `ATLYZOSHQRMQRP` | L-ascorbic acid, `CIWBSHSKHKDKBQ` | 3 |
| Kojic_acid | 5원 furanone, `JTQXALQLNOBOPA` | kojic acid 6원 pyranone, `BEJNERDRQOWKJM` | 1 |
| EGCG | 동일 분자식의 다른 이성질체, `IDJGBBBMZRXGSW` | (−)-EGCG, `WMBWREPUVVBILR` | 4 |

v1 SHA-256은 `1e0dd6763ea7f245839002bbe027a3a0cefb50137cf2959f178ee41d307a5017`, 현재 v2는 `e76a265db2e8c56554378091b0dd7d06de0e678076e64ed93923803e572bf8ec`다. v2는 case와 target을 바꾸지 않고 구조만 정정한다.

**영향.** 11/32쌍의 v1 순위는 이름 붙은 실제 화합물을 평가한 값이 아니다. 예를 들어 v2 재실행에서 Retinol→RARA는 467→25로 좋아졌지만 Ascorbic acid→P4HB는 169→703, EGCG→DNMT1은 52→506으로 나빠졌다. 방향이 혼재하므로 결과를 좋게 만들기 위한 단순 cherry-pick이라고 단정할 근거는 없다. 그러나 **정정 전 화합물별 성능 해석은 무효**다.

**긍정 통제.** 중간 `cce7…` panel이 들어왔을 때 v1 SHA 계약이 10개 테스트를 실패시켰다. 이는 계약이 drift를 fail-closed로 잡았다는 강한 증거다.

**추가 검증.** 네 정정 구조는 로컬 ChEMBL 37과 RDKit InChIKey connectivity 대조에 일치했다. 새 batch의 22/22 manifest hash가 검증됐고, 수정하지 않은 11개 화합물의 21쌍 순위와 해당 화합물 출력은 전후 동일했다. 이는 재실행 noise보다 구조 정정이 변화 원인이라는 강한 내부 증거다.

**한계와 남은 Unknown.** 새 `scripts/tests/test_panel_structures.py`는 InChIKey connectivity block만 고정하므로 stereochemistry 전체를 pin하지 않는다. v2의 독립 외부 curator 승인, 모든 v1 하류 산출물의 재생성·서명·커밋은 아직 없다. `results/audits/known_target_pair_evidence_chembl37.csv`, `data/activity_recovery_panels_202608/`, activity/operational bundle, 2026-08-22 bundle은 v1이다. 정정된 panel을 사용한 union-vs-baseline activity 비교는 **Unknown**이다.

### F-02. Comprehensive의 pose 계보 단절

**Evidence.** `workflow/rules/stage3a_comprehensive.smk:57-94`는 AutoDock pose 디렉터리를 만든다. 그러나 GNINA 입력은 `rules.xtb_optimize.output.sdf`다(`workflow/rules/stage3a_comprehensive.smk:142-162`). RTMScore도 동일한 자유 ligand SDF를 쓴다(`workflow/rules/stage3a_comprehensive.smk:166-186`). 내보낸 `autodock_all_target_poses/`는 두 규칙의 input에 없다.

GNINA 스크립트 자체는 pose manifest 모드를 구현하고 pose SHA와 target parity를 검증한다(`scripts/stage3_gnina_rescore.py:100-147,218-242,280-323`). 문제는 Comprehensive 규칙이 그 모드를 호출하지 않는다는 것이다.

**Inference.** AutoDock, GNINA, RTMScore, Boltz 네 열을 RRF로 합치는 연산 자체는 수행될 수 있다. 그러나 GNINA·RTMScore를 “AutoDock가 생성한 동일 pose의 rescore”로 부르는 의미는 성립하지 않는다. 따라서 `top50_4way_consensus.csv`의 4-way semantics는 현재 배선 기준으로 잘못되었다. Fast 경로는 별도 pose manifest와 GNINA exact-pose 경로를 갖기 때문에 이 결론을 Fast에 일반화하지 않는다.

### F-03. KG 문헌 수와 지지 플래그의 의미 오류

**Evidence.** `scripts/stage0_skin_kg.py:55-73`은 22~640과 같은 `n_papers`를 하드코딩한 fallback seed로 만든다. PubTator 병합은 기존 수와 실제 PMID 수의 최대값을 유지한다(`scripts/stage0_skin_kg.py:284-300`). 예컨대 저장된 AR hair-growth edge는 `n_papers=250`이어도 sample PMID는 훨씬 적을 수 있다. Stage 0 verifier는 전역 형태·개수 검사를 중심으로 하며 각 edge의 `n_papers == 검증된 고유 PMID 수`를 입증하지 않는다(`scripts/stage0_verify.py:1432-1454`).

`scripts/summarize_run_outputs.py:1063-1083,1148-1191`은 KG efficacy label 존재만으로 `efficacy_supported`를 참으로 만들 수 있다. UI는 이 수를 문헌 수로 노출한다(`workbench/static/app.js:141-153,578`).

**판정.** `n_papers`는 현재 실제 문헌 건수로 표시하면 안 된다. seed 우선순위 상수 또는 미검증 count로 명시해야 한다. KG label은 분류 annotation이며 기전·효능 근거가 아니다.

### F-04. Stage 11 “source-backed” 그림의 내용 불일치

**Evidence.** 8개 캡션은 target landscape, pharmacophore, analog affinity, MD RMSD/RMSF, retrosynthesis, 평가, case study를 약속한다(`scripts/stage11_make_figures.py:37-78`). 실제 `_source_metric`은 CSV/TSV 행 수, JSON key 수 또는 artifact 규모를 반환한다(`scripts/stage11_make_figures.py:145-160`). PNG는 이 값을 `artifact evidence count` 막대로 그린다(`scripts/stage11_make_figures.py:2075-2097`). SVG도 파일명과 count 텍스트를 쓴다(`scripts/stage11_make_figures.py:2050-2072`). 그럼에도 manifest status는 `source_backed`다(`scripts/stage11_make_figures.py:2149-2163`).

**판정.** source file에 해시로 연결되어 있다는 점은 사실이지만, 그림이 캡션에 적힌 과학량을 시각화하지 않는다. 출판용 그림으로는 FAIL이다.

### F-05. 커밋된 원고·회고와 현재 구현의 세대 불일치

**Evidence.** `results/MANUSCRIPT.md:1-30`은 “validated”, 24개쌍, Vina 및 retinol #1/#3/#4를 주장한다. 현 Demo는 Daina 순서를 고정하고 구조 점수를 annotation으로만 붙인다(`README.md:308-321,919-929`). 현 validation 문서는 이전 subset이 구현/SOTA/임상 증명이 아니라고 명시한다(`docs/SKIN_KNOWN_TARGET_VALIDATION.md:5-8,145-174`).

`results/RETROSPECTIVE.md:47`은 Vina ΔG≈−10 kcal/mol을 “nM regime”으로 연결하고, `results/MANUSCRIPT.md:30`은 여러 실패를 “single cause”로 환원한다. 이 두 해석은 각각 docking score의 정량 affinity 전환과 복합 실패 원인의 단정으로 과도하다.

커밋된 결과 파일은 `MANUSCRIPT.md`, `RETROSPECTIVE.md`, PNG 3개 등 5개뿐이다. 원고가 재현 원자료로 지칭하는 run CSV는 추적되지 않는다. `scripts/make_retrospective_figures.py:135,155-184`가 요구하는 demo CSV가 없어 새 임시 출력 디렉터리로의 재생성이 실패한다.

**판정.** 이 자료는 “2026-06 historical Vina retrospective”로 격리해 읽어야 하며, 현재 endpoint 또는 현재 Demo의 검증 결과가 아니다.

### F-06. 최신 완료 v1 과학 게이트 실패와 v2 미완결

**Evidence.** 현재 로컬에 존재하지만 Git에서 ignored인 평가 summary는 다음을 기록한다.

- `claim_ready=false`
- `passes_frozen_test_gate=false`
- `promotion_decision=retain_frozen_baseline`
- `operational_recipe_id=chembl_p5_max`

출처: `results/eval/activity_retrieval_202608/test_evaluation/summary.json:52,685-686`, `README.md:347,363-394`.

v1 평가에서 selected는 temporal을 개선하지만 known-skin Top30은 0.625→0.500으로 0.125 감소한다. dual-cold 163 truth pair의 baseline과 selected Top10/Top30은 모두 0이다. 따라서 operational gate가 기존 recipe를 계속 쓰게 한 것은 올바른 보수 동작이지만, 모델 성능의 PASS가 아니다. 구조 정정 후 v2 전체 activity 비교는 아직 없으므로 이 v1 수치를 현 canonical panel의 최종 성능으로 사용할 수 없다. 다만 현재 v2 Daina LQO도 자체 Top10/Top30 threshold를 통과하지 못하므로 과학적 승인 상태는 여전히 FAIL이다.

### F-07. known-target 라벨의 직접 근거 부족

**Evidence.** 패널은 15 case, 32 pair이며 각 row에는 자유문장 `evidence_note`만 있고 구조화 DOI/PMID가 없다. ChEMBL 37의 strict human single-protein/activity 규칙으로 v1을 독립 대조한 `results/audits/known_target_pair_evidence_chembl37.csv`는 7/32쌍을 지지하고 25/32쌍은 지지하지 못한다. 이 audit artifact는 v1 구조에 결속되어 있으므로 v2의 직접 지지 분모·분자는 재실행 전까지 **Unknown**이다.

**중요한 경계.** 여기서 “unsupported”는 “생물학적으로 거짓”이 아니다. 해당 ChEMBL 규칙에서 직접 근거를 찾지 못했다는 뜻이다. 다른 문헌의 진위와 품질은 **Unknown**이다. 따라서 32쌍 전체를 “문헌 검증 ground truth”라고 부르려면 pair별 DOI/PMID, assay, 종, endpoint, 방향성의 구조화 검토가 필요하다.

### F-08. 입력 적용 범위 gate 불일치와 보고 시점 부분 보완

**최초 기준 Evidence.** `scripts/compound_applicability.py:2-14`는 mixture, peptide, surfactant, polymer, UV filter를 범위 밖이라고 선언한다. 구현 분기 `scripts/compound_applicability.py:76-120`에는 mixture/peptide/polymer/surfactant 검사가 있지만 UV filter 판정은 없다. CLI command builder는 SMILES canonicalization만 하고 applicability를 호출하지 않는다(`scripts/run_skinscout.py:2454-2474`). Workbench는 SMILES에만 gate를 적용하고 SDF 업로드는 적용하지 않는다(`workbench/server.py:2468-2495`). Stage 1은 multi-record SDF의 첫 parseable molecule만 조용히 고른다(`scripts/stage1_standardize.py:44-55`).

Summary validator는 applicability를 계산하지만(`scripts/summarize_run_outputs.py:307-314`) 최종 `compound` 객체를 재구성할 때 해당 필드를 전달하지 않아(`scripts/summarize_run_outputs.py:1508-1525`) renderer가 기대하는 경고(`scripts/summarize_run_outputs.py:1618-1636`)가 사라질 수 있다.

**재현 Evidence.** tripeptide는 checker에서 거부되지만 `run_skinscout.py --print-command` 경로에서 명령 생성이 가능했다. 이는 데이터 입력점별 정책이 동일하지 않음을 보여준다.

**보고 시점 미커밋 델타.** 외부 초안은 `assess_sdf`와 공통 refusal formatter를 추가하고, CLI의 SMILES/SDF, Workbench의 SMILES/SDF에 같은 gate를 연결하며, summary의 `compound.applicability`를 보존한다. 둘 이상의 **parseable** molecule이 있는 SDF도 명시적으로 거부한다. 새 함수·summary 회귀 테스트의 targeted 실행은 **21 passed in 2.92 s**였고, 같은 tripeptide를 `run_skinscout.py --print-command`에 다시 넣은 직접 probe도 exit 1과 적용 범위 거부문을 반환했다. 따라서 최초 기준의 주요 경로 불일치와 summary 유실은 부분 해소됐다.

**잔여 Evidence/Inference.** 이 변경은 아직 미커밋이고, 선언된 **UV-filter 배제 규칙은 판정 로직과 테스트에 여전히 없다.** 직접 probe한 oxybenzone, avobenzone, octinoxate 3종은 모두 exclusion 없이 `in_scope`였다. 또한 유효 ethanol record 뒤에 불량 record를 붙인 SDF는 거부가 아니라 `review`로 수락됐다. 이는 `assess_sdf`가 supplier의 `None` record를 버린 뒤 유효 molecule 수만 세기 때문이다. Workbench는 upload를 쓴 뒤 정상적인 refusal 결과에는 unlink하지만, `assess_sdf` 자체가 예외를 던지는 경로의 cleanup 보장은 없다. 뒤이어 추가된 CLI SMILES/SDF 거부·정상 허용 회귀 3건과 readiness SDF 회귀 1건은 **4 passed in 5.33 s**로 CLI 배선을 고정한다. 그러나 Workbench 배선을 직접 고정하는 서버 통합 회귀와 실제 SDF 진입점 E2E는 여전히 없다. 그러므로 F-08은 **부분 해소, 미해결 유지**로 판정한다.

### F-09. PAINS `drop` 옵션 미집행

`workflow/config.yaml:86-92`와 `workflow/Snakefile:1071-1075`는 `pains_action`의 `warn|drop`을 허용한다. 그러나 `scripts/stage2_pains_brenk.py:38-68`은 alert JSON만 쓰며 action을 받지 않는다. Stage 3 진입은 skin sensitization과 cosmetic/drug HALT만 확인한다(`workflow/rules/stage3a_comprehensive.smk:37-53`). 따라서 `drop`은 유효 설정처럼 보이지만 실행 의미가 없다. **Evidence / High.**

### F-10. MD 설정–실행 불일치

설정은 `ligand_ff: openff_sage_2.2`, `timestep_fs: 2.0`을 선언한다(`workflow/config.yaml:149-161`). 실제 준비는 ACPYPE를 호출하고(`scripts/stage7_gromacs_prep.py:503-556`), run script의 timestep은 고정 2 fs다(`scripts/stage7_gromacs_run.py:31`). Snakefile은 duration/replica/timestep 값을 검증하지만(`workflow/Snakefile:1157-1187`) 선언된 ligand force field가 실제 topology provenance와 동등하다는 보장은 없다. **Evidence.** MD 결과를 OpenFF Sage 실행이라고 쓰면 안 된다.

### F-11. DiffDock 준비성 공백

`docking.diffdock_for_no_pocket` 설정은 `workflow/config.yaml:127`에 있으나 Comprehensive 규칙은 이를 조건으로 사용하지 않고 DiffDock rule을 mandatory dependency로 둔다(`workflow/rules/stage3a_comprehensive.smk:97-125`). `scripts/stage3_diffdock_blind.py:35,77`은 실행 파일/모듈 부재를 치명적으로 처리한다. 반면 launcher와 Workbench readiness 목록에는 이 요구가 명확히 포함되지 않는다(`scripts/run_skinscout.py:396`, `workbench/server.py:128`, `scripts/model_readiness.py:40`). 제공 env가 DiffDock 완결 설치를 입증하지도 않는다. **Evidence.**

### F-12. 현재 SkinScore는 사실상 두 HPA 축

공식 식은 HPA tissue 0.30, HPA cell 0.25, proteome 0.20, GTEx 0.10, scRNA 0.15다(`scripts/stage0_skin_score.py:2-43`). 사용 가능한 축만으로 가중치를 재정규화한다(`scripts/stage0_skin_score.py:459-471`). 보고 시점 `data/skin_proteome/skin_proteome.tsv`와 `data/gtex_v10/skin_tpm.tsv`는 header 1줄뿐이고 scRNA 실데이터도 없다. 결과적으로 현재 score는 HPA tissue 약 54.5%, HPA cell 약 45.5%다.

**판정.** “5-source composite absolute skin score”가 아니라 **가용 HPA 축 내 상대 순위**다. HPA cell은 HPA tissue 양성 단백질에만 허용하는 보수 통제가 있다(`scripts/stage0_skin_score.py:418-424`). 이 점은 긍정적이나, 발현은 결합·기전·피부 특이성 증거가 아니다.

### F-13. 재현 배포·보안·라이선스 공백

- Quick Start는 `main` branch의 원격 설치 스크립트를 즉시 shell에 전달한다(`README.md:26`). commit/digest pin이 없다.
- installer는 Ubuntu 24.04 x86_64만 허용한다(`scripts/bootstrap_runtime.sh:77-82`). 감사 host Ubuntu 26.04에서는 실제 설치 경로가 실패하므로 fresh-machine claim을 검증하지 못했다.
- Compose image digest는 `000…`, `111…`, `222…` placeholder라 실행 가능한 릴리스가 아니다(`compose/skinscout.compose.yaml:5,27,45`).
- conda YAML은 상위 package pin을 일부 갖지만 전체 transitive lock/hash가 아니다. Stage 11 environment capture도 모든 외부 binary/model/container digest를 완전히 봉인하지 않는다.
- 프로젝트 루트에 배포 조건을 정하는 `LICENSE`가 없다. `LICENSE_POLICY.md:21-23`은 STopTox/HuSSPred/Pred-Skin 계열 약관 재감사를 요구하므로 “모든 dependency permissive” 또는 “license-clean”은 확정되지 않는다.
- 외부 safety 서비스에 입력 SMILES가 전달될 수 있다. 개인정보라 단정할 수는 없지만 미공개 화합물 구조의 기밀성·서비스 보존정책·상업적 약관은 **Unknown**이다.
- CI, coverage threshold, 정적 보안 scanner를 릴리스 gate로 강제하는 추적 workflow가 없다.
- P2Rank 등 archive 설치 경로는 해시 검증이 있으나 extraction hardening을 외부 hostile archive 관점에서 완전 검증한 증거는 부족하다.

반대로 설치 artifact SHA, CAS/Cosign policy scaffolding, tar path traversal 방어, Workbench secret file 권한은 긍정 통제다. 문제는 **실제 서명 릴리스 artifact가 placeholder**라는 점이다.

### F-14. Stage 9 패널의 표현 불일치

파일 주석은 Panel A2를 histogram, A3를 scatter라고 부른다(`scripts/stage9_report.py:2-8`). 실제 A2는 상위 50행 HTML table이고(`scripts/stage9_report.py:1716-1719,2990-2992`), A3는 JSON을 6,000자로 잘라 `<pre>`로 표시한다(`scripts/stage9_report.py:3016-3019`). 이는 데이터 손실 여부와 별개로 시각화 설명과 구현이 불일치한다. **Evidence / High.**

### F-15~F-20. 중간 위험 공백

- **Boltz crop/max-residues:** 인자를 검증하고 함수에 전달하지만 `call_boltz`는 실제 crop을 하지 않고 full receptor YAML을 쓴다(`scripts/stage3_boltz2_affinity.py:54-59,106-129`). Stage 5도 인자를 전달하지만 crop 구현 없이 예측 호출로 이어진다(`scripts/stage5_boltz2.py:83-103,287-326`).
- **Boltz `msa_source`:** `workflow/config.yaml:134-140`에 있지만 rule/CLI에 전달되지 않는다.
- **mode=both readiness:** UI는 선택을 제공하지만 준비성 집계는 comprehensive 중심이며 실제 DAG는 fast와 comprehensive 산출물을 모두 요구한다(`workbench/static/app.js:229`, `workbench/server.py:1339,2395`, `workflow/Snakefile:1413-1418`).
- **sensitization threshold:** 모델은 정확히 3개인데 `skin_sens_halt_min_votes`는 최소 1만 검증하고 최대 3을 제한하지 않는다(`workflow/Snakefile:1064-1070`, `scripts/stage2_consensus.py:141-160,189-201`). 4 이상이면 HALT가 불가능하다.
- **verification coverage:** run summary/log/report와 일부 핵심 파일은 해시되지만 모든 주장 원시 evidence 파일을 최종 verification record가 직접 결속하지 않는다(`scripts/run_skinscout.py:153,227,1254,1914`).
- **Workbench fixture 노출:** 서버는 summary가 있으면 내부/fixture 실행도 completed로 읽을 수 있고(`workbench/server.py:1940-1976`), beginner filter는 일부 prefix만 제외한다(`workbench/static/app.js:23,79-92`). all summaries가 `claimable=false`인 것은 완화책이지만 목록 의미는 혼동될 수 있다.
- **Demo 문구:** 최초 기준 카드의 “Daina + 도킹 근거로 표적 순위화”(`workbench/static/index.html:213`)는 Daina 순서를 보존하는 annotation-only 구현(`README.md:318-321`)보다 강한 표현이다. 보고 시점 미커밋 UI 초안은 이를 “리간드 유사도로 표적 순위화, 도킹은 주석”으로 고치고 glossary도 Daina 1차 경로의 종합 점수가 최근접 Tanimoto임을 명시해 의미 불일치를 코드상 해소한다. 아직 커밋·UI 회귀 검증 전이므로 최초 기준 결함과 별도 표시한다.
- **효능 confidence:** `high/medium/low`와 uncertainty `0.1/0.5/0.8/0.9`는 구조화 metadata 완성도에 따른 상수다(`scripts/stage3_kg_efficacy_label.py:328-350`). 예측 신뢰도의 통계적 calibration이 아니다.
- **promotion epsilon:** 활동 검색 gate는 Top10/Top30/MRR의 point estimate가 `1e-12`보다 모두 커야 한다(`eval/activity_retrieval_model.py:1587-1600`). 이 gate 자체에는 표본 상관, confidence interval, significance가 없다. 별도 prospective bootstrap gate가 존재하므로 두 계약을 혼동하면 안 된다.
- **retrieval 기본값:** Daina 기본 `retrieval`은 직접 evidence를 허용하는 retrospective lookup이다(`workflow/config.yaml:115`, `scripts/stage3_daina_zoete.py:595-616`). 신규 발견 성능에는 `discovery`, leave-query-out 또는 temporal 통제가 필요하다.

### F-21. Performance-v2 scoring의 device 계약 결함

**Evidence.** v2 전체 suite의 유일한 실패는 `scripts/tests/test_performance_v2.py::test_torch_cpu_minibatched_train_and_score_is_deterministic`였다. 테스트는 training에만 `--cpu`를 전달하고(`scripts/tests/test_performance_v2.py:1255-1269`) CPU budget을 검사하지만(`scripts/tests/test_performance_v2.py:1285-1288`), 두 score 호출에는 device 인자가 없다(`scripts/tests/test_performance_v2.py:1302-1311,1322-1331`). Train CLI는 `--cpu`를 지원하지만(`scripts/train_performance_v2.py:44-58,90-100`) score CLI에는 device 선택자가 없다(`scripts/score_performance_v2.py:55-66`). 공통 scorer는 `torch.cuda.is_available()`만으로 CUDA를 선택한다(`eval/performance_v2_model.py:2207-2208`). 모델을 device로 옮기는 단계의 OOM은 forward loop의 batch-halving 복구보다 먼저 발생한다(`eval/performance_v2_model.py:2241-2256`). query embedding도 CUDA를 자동 선택한다(`eval/performance_v2_model.py:1320-1344`).

**실행 결과.** v2 전체 suite 실행 당시 GPU-visible 환경의 model `.to(device)`에서 CUDA OOM이 났고, 같은 테스트를 `CUDA_VISIBLE_DEVICES=''`로 실행하면 **1 passed in 7.41 s**였다. 이후 GPU-visible targeted 재실행도 **1 passed in 18.98 s**였다. 따라서 OOM은 현재 항상 재현되는 산술 결함이 아니라 **GPU 메모리 상태 의존적·간헐적 실패**다. 다만 score CLI가 명시적 device를 받지 않고, CPU budget 테스트가 inference device를 고정하지 않으며, model 이동 OOM이 복구 범위 밖이라는 코드 계약 공백은 재실행 성공과 무관하게 남는다. model manifest의 training execution mode는 inference policy가 아니며 ranking manifest도 inference device를 기록하지 않는다.

---

## 5. 아키텍처와 Stage별 논리 감사

### 5.1 전체 데이터 흐름

Public Demo/Fast의 실제 핵심은 다음과 같다.

`SMILES/SDF → Stage 1 표준화 → Stage 2 ADMET/피부감작성 → Stage 2.5 CosIng/약물 회피 → Daina 고정 순위 → top 256 구조 주석(AutoGrid/AutoDock pose/GNINA exact pose) → 피부/KG annotation → sealed report-fast/Mol*`

Comprehensive는 별도 경로다.

`표준화 ligand → 전 단백질 AutoDock → top percentile → GNINA/RTM/Boltz → RRF → skin weighting/KG → 구조·Boltz·BioEmu·MD·QM → Stage 9 report → opt-in Stage 11 publication scaffold`

### 5.2 Stage별 판정

| Stage | 구현 역할 | 감사 판정 |
|---|---|---|
| 0 Infrastructure | AlphaFold 구조 정리, P2Rank pocket, PDBQT/box, ChEMBL/BindingDB/GtoPdb, HPA/SkinScore, CosIng, drug/KG, MMseqs | 대규모 provenance·해시·완전성 검사가 강점. 다만 KG seed count 의미, HPA-only score, 빈 보조 발현원, P2Rank extraction/외부 데이터 신뢰 경계가 남음. |
| 1 Compound prep | RDKit 표준화, salt/neutralization, Dimorphite, ETKDG/MMFF, xTB | fallback fail-closed는 긍정적. SDF 다중 record 첫 분자 선택과 공통 applicability gate 부재는 High. |
| 2 ADMET/sensitization | ADMET-AI, STopTox, HuSSPred, Pred-Skin, PAINS/Brenk/NIH, 3-model consensus | source evidence와 decision 일치 검증이 강함. 외부 서비스 privacy/ToS, PAINS drop 무효, threshold 상한 부재. endpoint screen이지 안전성 허가가 아님. |
| 2.5 Cosmetic/drug | CosIng exact/similar/analog, 승인약·scaffold 경고, HALT/DOWNWEIGHT/PROCEED | missing reference fail-closed는 강함. policy source의 법적·생물학적 의미는 별도. |
| 3a Comprehensive | AutoDock 전수, DiffDock no-pocket, GNINA/RTM/Boltz, weighted RRF | Critical pose 계보 단절, DiffDock readiness 공백. 4-way 점수는 서로 다른 입력 구조 의미를 섞음. |
| 3b Fast | Daina top 256, query-specific AutoGrid, AutoDock pose export, GNINA exact-pose, immutable Daina overlay | 구현 계약이 가장 명확함. Daina order 보존·nonprobability annotation이 긍정적. 기본 retrieval은 discovery 성능 증거가 아님. |
| 3c Disagreement | PSICHIC 대 docking 비교 | 진단적 사용은 가능. 700-aa window max aggregation은 외부 검증되지 않음. |
| 3 v3 | docking RRF + SkinScore, KG efficacy label | Comprehensive 전용으로 Demo 순위를 바꾸지 않는 점은 명확. SkinScore와 efficacy label 의미는 과해석 금지. |
| Activity/performance | temporal, known-skin, dual-cold, calibration, leakage gate | manifest/hash/failed promotion 보존은 강함. 현재 scientific gate FAIL. v1 구조 오류로 일부 지표 재평가 필요. |
| 4 Structure prep | top target 구조와 pocket 준비, RCSB holo/AlphaFold fallback | 구조 출처 분리는 긍정적. AlphaFold 단일 예측 상태와 cofactor 결손은 기전 원인 단정 불가. |
| 5 Boltz-2 | cofold/affinity, iPTM/pLDDT/PoseBusters gate | partial output fail-closed는 강함. crop/max-residues와 MSA config 미결속. binary binder score와 continuous affinity 구분 필요. |
| 5.5 Interaction atoms | PLIP + ProLIF, atom mapping | 2/2 consensus·isomorphism·hash binding은 강함. pose-supported contact이지 causal pharmacophore가 아님. |
| 5.6 Analog | manifest-driven REINVENT 및 2D substitute baseline | 외부 model/plugin pin 요구와 `analog_pose_verified=false`가 보수적. 실제 pinned production artifact 부재. |
| 6 BioEmu | ensemble 생성·ensemble docking | rule/code 존재. 실제 end-to-end 품질 실행과 불확실성 검증은 Unknown. |
| 7 MD | GROMACS prep/production/MMGBSA | deterministic contract는 있으나 force-field/timestep 설정 의미가 실제 실행과 불일치. 실제 장기 trajectory QA는 Unknown. |
| 7.5 Retrosynthesis | AiZynthFinder route 및 score | opt-in, 외부 manifest gate는 타당. 실제 model/stock 실행 증거는 없음. |
| 8 QM | CREST/xTB/DFT | scaffold 존재. 현재 결과에서 과학적 검증 evidence 없음. |
| 9 Report | Mol* 통합 HTML, run evidence, fast package link | 입력·pose·bundle 해시와 보수적 제한문은 강함. A2/A3 시각화 명칭 불일치. |
| 10 | 번호가 붙은 Stage 10 rule 없음 | 누락 구현으로 단정하지 않는다. 현재 DAG는 Stage 9에서 Stage 11로 간다. |
| 11 Publication | figures, data availability, repro pack, manuscript skeleton | upstream 누락 시 fail-closed는 긍정적. 실제 그림 내용은 출판 요구와 불일치하므로 FAIL. |

### 5.3 Launcher·Workbench·claimability

- `scripts/run_skinscout.py`는 preset별 target을 계산하고 기존 run manifest의 hash mismatch를 거부한다.
- degraded/diagnostic 경로는 claimable 결과로 승격되지 않는다(`scripts/run_skinscout.py:1499,2004`).
- clean-run 삭제는 결과 루트와 run dir 관계를 검사한다(`workflow/Snakefile:1453-1463`).
- Workbench는 loopback bind, Host/Origin, CSRF, bearer-like worker token, 0600 secret, path containment, CSP를 구현한다(`workbench/server.py:254-325,2666-2715,3093-3124,3197-3207`).
- 그러나 claimability가 false라는 사실이 UI의 “완료” 분류 자체를 항상 명확하게 바꾸지는 않는다.

---

## 6. 결과·세대·계보 감사

### 6.1 결과 디렉터리 총괄

| 항목 | 수량 |
|---|---:|
| `results/runs` run directory | 279 |
| `run_manifest` | 21 |
| `run_summary` | 44 |
| verification record | 45 |
| verification `ok` / `failed` | 38 / 7 |
| summary `FLAG_HIGH` / `HALT` | 32 / 12 |
| summary `claimable=true` | **0 / 44** |

presence 조합은 summary/manifest/verification이 고르게 존재하지 않는다: 222개는 셋 다 없고, 35개는 summary+verification, 11개는 manifest만, 1개는 manifest+verification, 9개만 셋 다 있다. 이는 run directory 수를 완료·재현 가능한 실험 수로 사용할 수 없음을 뜻한다.

### 6.2 세대별 계보

| 세대 | 대표 성격 | 판정 |
|---|---|---|
| 2026-06 historical Vina | 24-pair/300-receptor 회고, 커밋 원고·그림 | 현 endpoint 아님; 구조 오류·과장·원자료 부재로 출판 근거 부적격 |
| 2026-06 deepdiag | relaxed/degraded 진단 | 명시적 diagnostic; 성능 주장 불가 |
| 2026-07 zero-eval | 0 또는 불완전 평가 산출물 | claim 근거 아님 |
| 2026-08-04 variants | 다수 실패/부분 recipe | 실패 기록 보존은 유용; 승격 근거 아님 |
| PSICHIC partial | 부분 실행 | 통합 비교 불완전 |
| stale pocket audit | staging path 90개가 stale | 재현 계보 결함; 최신 gate에 혼합 금지 |
| 2026-08-21 activity gate | 최종 gate 준비 세대 | `claim_ready=false` |
| 2026-08-22 full methods | Daina/PSICHIC/RRF 등 full 평가 | 모든 핵심 과학 gate 실패; v1 구조 오류 영향 포함 |
| 2026-08-26 v2 Daina LQO | 커밋된 구조 정정 후 ignored 결과 재실행 | manifest/hash는 강함; 결과는 Git 미추적, threshold FAIL, 전체 계보 이관 미완료 |
| single caffeine scratch | 단일 화합물 진단 | 일반 성능 근거 아님 |

### 6.3 독립 재계산된 v1/기존 지표

다음 값은 원시 CSV의 분자·분모와 reciprocal rank에서 재계산해 저장 값과 일치했다.

| 평가 | recipe | Top10 | Top30 | MRR |
|---|---|---:|---:|---:|
| 2025 temporal, 298 truth pairs | baseline | 179/298 = **0.60067114** | 212/298 = **0.71140940** | **0.41727273** |
| 2025 temporal, 298 truth pairs | selected | 212/298 = **0.71140940** | 232/298 = **0.77852349** | **0.54393692** |
| known-skin v1, 32 pairs | baseline | 6/32 = **0.1875** | 20/32 = **0.625** | **0.13622827** |
| known-skin v1, 32 pairs | selected | 7/32 = **0.21875** | 16/32 = **0.500** | **0.14740439** |
| dual-cold, 163 pairs | baseline | 0/163 = **0** | 0/163 = **0** | **0.00008293** |
| dual-cold, 163 pairs | selected | 0/163 = **0** | 0/163 = **0** | **0.00008416** |
| Daina LQO v1, 32 pairs | Daina | 8/32 = **0.25** | 11/32 = **0.34375** | **0.16856095** |
| PSICHIC, 32 pairs | PSICHIC | 0/32 = **0** | 확인 범위에서 0 | **0.00380224** |
| RRF LQO v1, 32 pairs | RRF | 8/32 = **0.25** | 13/32 = **0.40625** | **0.12048655** |

**구조 오류 경고:** known-skin/Daina/PSICHIC/RRF 표에서 Retinol, Ascorbic acid, Kojic acid, EGCG와 연관된 v1 값은 이름 붙은 실제 화합물의 값이 아니다. 전체 집계도 11/32쌍의 영향을 받으므로 v2 집계와 직접 비교해 모델 개선을 주장하면 안 된다.

### 6.4 현재 Git ignored v2 결과

`results/eval/skin_known_full_20260826/daina_leave_query_out/manifest.json`은 panel SHA `e76a…`, ChEMBL37 activity/fingerprint SHA, 각 ranking SHA, 실행 argv를 기록한다. 22/22 manifest-bound hash가 일치했고, 수정하지 않은 11개 화합물의 21쌍과 출력은 전후 동일했다. 이 범위의 내부 provenance는 강하다. 다만 새 batch에는 절대 staging path 45개가 남아 있어 이식성은 불완전하다.

| v2 Daina leave-query-out | 값 |
|---|---:|
| case coverage | 15/15 = 1.0 |
| ranked target coverage | 31/32 = 0.96875 |
| case Top10 | 7/15 = 0.46666667 |
| target Top10 | 8/32 = 0.25 |
| target Top30 | 14/32 = 0.4375 |
| target-pair MRR | 0.19239941 |
| finite-rank median / mean | 32 / 181.80645 |
| threshold result | **false** |

실행 threshold는 case Top10≥0.8, target Top10≥0.5, target Top30≥0.6이며 세 항목 모두 실패한다. 이 결과는 v2 구조 정정의 영향 확인에는 유용하지만 다음 이유로 승격 근거가 아니다.

1. 구조·계약·정정 문서는 `53b63364…`로 커밋됐지만 결과 82개는 Git ignored 상태다.
2. v1 기반 ChEMBL pair audit와 activity recovery panels가 아직 갱신되지 않았다.
3. v2 전체 suite의 한 실행은 device-contract 공백에서 촉발된 간헐적 CUDA OOM 1건을 포함했다. 이후 CPU-hidden·GPU-visible targeted 재실행은 각각 통과했지만 명시적 device 계약은 없다.
4. panel/code commit과 ignored 결과·ChEMBL snapshot을 하나의 release manifest로 봉인하지 않았다.

v1 Daina LQO와의 기술적 차이는 Top30 11→14/32, MRR 0.16856095→0.19239941, finite-rank median 42→32로 개선됐지만 mean은 154.68→181.81로 악화됐다. 구조가 바뀐 평가끼리의 비교이므로 이를 모델 개선으로 부르면 안 된다.

v2 similarity bin은 다음과 같다.

| 최근접 similarity | pair / 고유 화합물 | median rank | Top10 | Top30 |
|---|---:|---:|---:|---:|
| ≥0.6 | 12 / 7 | 8 | 6/12 | 11/12 |
| 0.4–0.6 | 13 / 8 | 45 | 1/13 | 2/13 |
| <0.4 | 6 / 4 | 316.5 | 1/6 | 1/6 |

`docs/PANEL_STRUCTURE_CORRECTION_20260826.md`의 “양쪽 각각 7–8개 화합물” 설명은 정확하지 않다. `<0.6` union은 10개 화합물이며 bin 간 화합물이 겹친다. pair가 화합물 내 상관된다는 핵심 경고는 타당하지만 실질 표본수 설명은 수정이 필요하다.

### 6.5 수치 해석

- temporal 개선은 직접 확인되지만 같은 recipe가 known-skin Top30을 악화시킨다. 도메인 간 일관된 개선이 아니다.
- dual-cold Top10/Top30 0은 미지원 표적 일반화가 사실상 해결되지 않았음을 보여준다.
- Daina, RRF, PSICHIC score는 서로 다른 scale과 의미를 가지므로 확률처럼 비교하면 안 된다.
- gate의 strict epsilon은 point estimate 비교다. 실질적 효과 크기, 화합물 단위 상관, 반복, 신뢰구간을 대신하지 않는다.

---

## 7. 시각화·UI 감사

### 7.1 asset census

| 형식 | 파일 수 |
|---|---:|
| PML | 20,188 |
| CXC | 20,173 |
| PNG | 562 |
| GIF | 245 |
| HTML | 1,248 |
| HTM | 184 |
| SVG | 159 |
| PDF | 117 |
| JPG | 50 |

추적 PNG 6개는 원본 해상도로 직접 확인했다. 나머지 대량 PML/CXC/HTML은 생성 규칙과 대표 표본을 중심으로 감사했다.

### 7.2 발견사항

1. **Stage 11 내용–캡션 불일치:** F-04 참조.
2. **Stage 9 가짜 histogram/scatter:** F-14 참조.
3. **PyMOL empty-pocket crash 위험:** `data/human_pockets/visualizations/A0A024R1R8_clean.pdb_pymol.pml:43-45`는 `stored.list[-1]`을 빈 목록 검사 없이 사용한다. pocket point가 없으면 IndexError 가능성이 있다.
4. **Figure 1 과해석:** `scripts/make_retrospective_figures.py:234-276`은 rank demotion을 빨간색 “systemic target”으로 라벨한다. rank 이동만으로 systemic expression 원인을 입증하지 않는다.
5. **분모 불투명:** 화합물–표적 map은 25개쌍을 하드코딩하지만 누락 데이터를 조용히 drop하고 실제 제목은 24개일 수 있다(`scripts/make_retrospective_figures.py:30-42,231-242`).
6. **불확실성 부재:** 커밋 그림은 반복, 신뢰구간, 효과 크기, 화합물 cluster 상관을 표시하지 않는다.
7. **접근성:** red/green 의미 부호가 색각 접근성에 취약하고 Figure 2 label이 중첩된다. PML/CXC 기본 뷰에는 범례·단위·confidence·고정 camera 계약이 부족하다.
8. **재생성 실패:** 필요한 demo CSV가 추적되지 않아 retrospective figure generator가 입력 없음으로 실패한다.

### 7.3 긍정 UI 증거

- Workbench 결과 화면은 hypothesis/비임상 경계를 반복한다(`workbench/static/app.js:391-400,560-579`).
- 입력 파일 크기 제한과 browser-side SHA 표시가 있다(`workbench/static/app.js:21,916-927`).
- loopback/Origin/CSRF/CSP/path containment는 로컬 도구로서 강한 기본값이다.
- HTML은 의미 있는 label, keyboard 가능한 native control, status text를 사용한다. 다만 표의 score 단위·범위 설명이 접힌 glossary에만 있는 경우가 있어 결과 옆 인라인 경계가 더 필요하다.

---

## 8. 과학적 해석 경계

아래는 코드가 선택하는 field와 1차 자료를 함께 확인한 해석 상한이다.

### 8.1 Boltz-2

`scripts/boltz2_runner.py:169-180`은 `affinity_probability_binary`를 우선 score로 선택한다. Boltz 공식 문서는 이를 binder-vs-decoy binary probability로 설명하고, `affinity_pred_value`를 log10(IC50, μM) 연속값으로 구분한다.

- 공식 저장소: https://github.com/jwohlwend/boltz#binding-affinity-prediction
- preprint: https://doi.org/10.1101/2025.06.14.659707

따라서 binary field를 절대 affinity, potency, calibrated target probability로 부르면 안 된다. Stage 5가 별도 continuous field를 읽는 경로와 Stage 3의 binary score를 문서·열 이름에서 명시적으로 분리해야 한다.

### 8.2 PSICHIC

`scripts/stage3_psichic.py:296-319`은 700 aa보다 긴 단백질을 window로 나눈 뒤 최대 score를 취한다. PSICHIC upstream이 이 max-window aggregation을 단백질 전체 score로 검증했다는 근거는 확인하지 못했다.

- 공식 저장소: https://github.com/huankoh/PSICHIC#byo-psichic-with-annotated-sequence-data
- 논문/preprint: https://doi.org/10.1101/2023.09.17.558145

이는 긴 서열을 처리하기 위한 구현 추론이며, calibration을 보존한다고 가정하면 안 된다.

### 8.3 Docking·GNINA·AutoDock-GPU

Docking score는 제한된 sampling과 scoring function의 근사치다. Vina FAQ와 원 논문은 결과를 실험 binding/free energy로 동일시할 수 없음을 뒷받침한다. GNINA도 pose quality와 affinity 모델의 출력 의미를 구분한다.

- Vina FAQ: https://autodock-vina.readthedocs.io/en/stable/faq.html
- AutoDock Vina: https://doi.org/10.1002/jcc.21334
- GNINA: https://doi.org/10.1186/s13321-021-00522-2
- GNINA 공식 저장소: https://github.com/gnina/gnina#cnn-scoring
- AutoDock-GPU: https://github.com/ccsb-scripps/AutoDock-GPU

따라서 docking/GNINA는 결합 가설과 구조 우선순위를 만들 뿐, 실제 결합·기전·효능을 증명하지 않는다. Vina ΔG를 곧바로 nM로 환산하는 회고 문장은 제거 대상 주장이다.

### 8.4 AlphaFold 구조

AlphaFold는 cofactor/ligand가 빠질 수 있고 단일 예측 구조가 모든 기능 상태를 나타내지 않는다.

- FAQ: https://alphafold.ebi.ac.uk/faq
- EBI limitations: https://www.ebi.ac.uk/training/online/courses/alphafold/an-introductory-guide-to-its-strengths-and-limitations/strengths-and-limitations-of-alphafold/

이 자료는 “missing cofactor와 상태 불확실성을 고려해야 한다”는 경계를 지지하지만, 모든 GPCR가 inactive이거나 여러 실패가 하나의 원인 때문이라는 단정을 지지하지 않는다.

### 8.5 RRF

RRF `k=60`은 순위를 결합하는 정보검색 기법이다.

- 원 논문: https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf
- DOI: https://doi.org/10.1145/1571941.1572114

구현은 입력 검증과 source contribution을 잘 기록한다(`scripts/stage3_rrf.py:139-321`). 그러나 RRF score는 생물학적 확률, calibration, 독립적 검증이 아니다. 가중 RRF는 이 저장소의 확장 recipe다.

### 8.6 Daina와 SwissTargetPrediction

Daina가 ligand-similarity/Zoete 계열 아이디어를 사용한다고 해서 SwissTargetPrediction의 성능을 상속하지 않는다.

- SwissTargetPrediction 2019: https://doi.org/10.1093/nar/gkz382
- 2024 target prediction 연구: https://doi.org/10.1038/s42004-024-01179-2

현재 Daina 구현·데이터 snapshot·평가 panel을 독립적으로 검증해야 한다. “Daina-Zoete”는 계보 라벨이지 등가 구현 증명은 아니다.

### 8.7 HPA와 SkinScore

- HPA skin: https://www.proteinatlas.org/humanproteome/tissue/skin
- HPA about: https://www.proteinatlas.org/about

발현은 조직·세포 맥락 prior다. 화합물이 그 단백질에 결합하거나 해당 피부 세포에서 선택적으로 작동한다는 증거가 아니다. min-max와 가용축 재정규화 score는 dataset 내부 상대량이며 외부 절대 척도가 아니다.

### 8.8 ADMET·피부감작성

- ADMET-AI: https://doi.org/10.1093/bioinformatics/btae399
- 공식 저장소: https://github.com/swansonk14/admet_ai
- 피부감작성 관련: https://doi.org/10.1021/acs.chemrestox.0c00186
- 독성 endpoint/평가 문맥: https://doi.org/10.1289/EHP9341
- 모델 검토: https://pmc.ncbi.nlm.nih.gov/articles/PMC11598222/

각 도구는 endpoint별 screen이다. 모델 음성, 3-vote PASS, PAINS 음성은 사람 대상 안전성 clearance가 아니다. 적용 도메인, assay 정의, 서비스 version, threshold calibration이 함께 기록돼야 한다.

### 8.9 Calibration

- neural calibration: https://proceedings.mlr.press/v70/guo17a.html
- chemical model applicability/calibration 문맥: https://doi.org/10.1039/C7SC02664A

확률로 부르려면 독립 calibration set, Brier/log-loss/ECE, calibration plot, 시간·scaffold/target 이동에서의 안정성이 필요하다. metadata completeness 상수와 RRF/Daina/docking score는 이 조건을 만족하지 않는다.

---

## 9. 테스트·정적 검증·재현성

### 9.1 실행한 정적 검증

| 검증 | 결과 |
|---|---|
| Python `compileall` — `scripts eval skinscout workbench` | PASS |
| 추적 shell 16개 `bash -n` | 16/16 PASS |
| 추적 JSON parse | 9/9 PASS |
| 추적 YAML parse | 12/12 PASS |
| `node --check` — Workbench app 및 Mol* viewer | PASS |
| 추적 symlink/special file | 없음 |
| 추적 secret pattern scan | 명백한 credential 없음; placeholder·token 경로만 확인 |

정적 PASS는 외부 binary, GPU inference, network service, 대규모 데이터 의미 또는 과학 결과의 정확성을 입증하지 않는다.

### 9.2 pytest 시간선

1. **중간 이동 작업 트리 전체 suite:** 2,571 passed, 35 skipped, 10 failed, 1,386.18 s. 10건은 모두 panel digest `cce7…`와 코드가 기대한 v1 `1e0d…`의 불일치였다.
2. **외부 재실행, v2 계약 반영 전:** 2,590 passed, 39 skipped, 10 failed. 같은 panel contract mismatch 계열이었다.
3. **v2 계약 영향 대상 suite:** `test_activity_retrieval_model.py`, `test_build_activity_recovery_panels.py`, `test_panel_structures.py`를 함께 실행해 **62 passed**.
4. **v2 전체 suite:** **2,600 passed, 39 skipped, 1 failed in 532.41 s**. 유일한 실패는 performance-v2 CPU 학습/score 결정성 테스트가 score 단계에서 CUDA를 자동 선택해 model `.to(device)`에서 OOM 난 것이다.
5. **유일 실패의 CPU-hidden 재실행:** 같은 환경에서 `CUDA_VISIBLE_DEVICES=''`로 해당 테스트를 재실행해 **1 passed in 7.41 s**.
6. **유일 실패의 최신 GPU-visible targeted 재실행:** 동일 테스트가 **1 passed in 18.98 s**. 즉 OOM은 GPU 메모리 상태에 의존해 간헐적으로 나타나며 항상 재현되지는 않는다.
7. **보고 시점 applicability 초안 targeted 검증:** `test_compound_applicability.py`와 신규 summary 보존/Markdown rendering 2건을 실행해 **21 passed in 2.92 s**.
8. **보고 시점 Workbench 정적 UI 검증:** `test_workbench_static_ui.py` **10 passed**. 바뀐 표현이 정적 UI 계약을 깨지 않았다는 좁은 증거다.
9. **보고 시점 CLI/readiness 통합 회귀:** CLI의 out-of-scope SMILES/SDF 거부, in-scope 허용, readiness의 실제 SDF provenance 4건이 **4 passed in 5.33 s**.

따라서 실패했던 전체 suite 실행을 사후 targeted pass만으로 “green”이라고 바꾸지 않는다. 두 targeted pass는 모델 산술 오류나 지속적 GPU 고장의 증거가 없음을 좁게 보여준다. 반면 명시적 device 선택 부재, model-load OOM 복구 공백, CPU 테스트 격리 결함은 코드에서 직접 확인된다. 또한 현재 미커밋 applicability 초안까지 포함한 전체 suite는 재실행되지 않았다.

### 9.3 환경·coverage 공백

- skip 35~39개가 무엇을 생략했는지 GPU/외부 서비스/대형 데이터별로 release note에 분해되지 않았다.
- coverage percentage와 최소 threshold가 없다.
- 실제 Ubuntu 24.04 + NVIDIA fresh install, Demo qualification, Full physics E2E를 이번 감사에서 실행하지 못했다.
- host Ubuntu 26.04는 installer가 의도적으로 지원하지 않는다.
- 결과 경로 329 MB와 데이터 123 GB가 있어 동일 byte snapshot 없이 “재현”을 주장할 수 없다.

---

## 10. 긍정 통제와 잘 구현된 부분

1. **RRF:** duplicate/blank/score validation, deterministic rank, source contribution 기록이 좋다(`scripts/stage3_rrf.py:139-321`).
2. **Daina structural overlay:** 원 순서를 유지하고 구조 score를 probability라고 부르지 않는다(`scripts/stage3_daina_structural_overlay.py:21,426-498`).
3. **AutoGrid cache:** 입력 receptor/box/ligand와 artifact를 강한 hash로 결속한다(`scripts/stage3_autogrid_maps.py:288-492`).
4. **Degraded 경로:** degraded/diagnostic을 claimable completion으로 승격하지 않는다.
5. **Workbench 보안:** loopback-only, Host/Origin, CSRF, worker token, 0600 secret, path containment, CSP, artifact SHA 검사가 있다.
6. **삭제 안전:** clean-run 경로가 results root와 정확한 run id 아래인지 확인한다.
7. **CAS/서명 계약:** image/data/qualification trust policy가 Cosign identity/issuer와 digest를 요구하는 구조는 올바르다. 현재 placeholder라 실행 준비가 안 된 점과 구분해야 한다.
8. **패널 fail-closed:** SHA 계약이 중간 drift를 실제 테스트 실패로 포착했다.
9. **결과 보수성:** 44/44 summary가 claimable false이며, 실패한 후보를 baseline으로 되돌리는 gate가 작동한다.
10. **문서의 좋은 경계:** README의 자세한 절은 no SOTA/clinical/probability, expression≠action, diagnostic≠claim을 명시한다.

---

## 11. 릴리스·과학·출판 사용 전 필수 조건

이 절은 수정 작업이 아니라 감사 권고다.

### P0 — 결과 정체성과 주장 차단

1. `53b63364…`에 커밋된 v2 panel을 독립 curator가 입체화학까지 검토하고, connectivity-only 회귀 테스트를 전체 identity 계약으로 보강하며, v1을 명시적으로 superseded artifact로 보존한다.
2. 11개 영향 pair가 들어간 모든 Daina/PSICHIC/RRF/activity recovery/문서/그림을 동일 v2 snapshot으로 재생성한다.
3. v2 panel, 코드 commit, ChEMBL snapshot, 모든 하류 artifact hash를 하나의 release manifest로 결속한다.
4. Comprehensive GNINA·RTMScore를 실제 exported AutoDock pose에 연결하거나, 현재 계산을 pose rescore라고 부르지 않도록 contract를 분리한다.
5. `claim_ready=false` 동안 SOTA·validated·clinical·publication-ready 문구와 Stage 11 publication artifact 공개를 차단한다.
6. KG seed `n_papers`를 실제 검증 PMID count와 분리하고 label-only support를 “문헌 지지”로 표시하지 않는다.

### P1 — 재현성·검증·시각화

1. v2 clean checkout에서 전체 suite, GPU qualification, 핵심 E2E를 실행하고 skip 이유를 공개한다.
2. 원시 CSV/manifest/env/model/binary/container digest를 포함한 재현 bundle을 추적 또는 content-addressed release로 제공한다.
3. Stage 11 각 그림이 캡션의 실제 변수·분모·불확실성을 그리게 하고 source hash와 plotting code를 봉인한다.
4. 커밋된 historical 원고·그림을 명확히 archive 처리하고 재생 가능한 현 세대 artifact와 분리한다.
5. known-target pair마다 DOI/PMID, species, target, assay endpoint, activity direction, curator decision을 구조화한다.
6. 보고 시점 applicability 초안의 CLI/Workbench/SMILES/SDF 공통 gate와 multi-record SDF 정책을 정식 검증·확정하고, 빠진 UV-filter 판정을 구현 계약과 일치시킨다.
7. PAINS action, sensitization threshold, `mode=both`, DiffDock readiness, Boltz crop/MSA, MD force-field/timestep 설정을 실제 실행 contract에 결속한다.
8. performance-v2 score CLI에 명시적 inference device 정책을 추가하고 model/ranking manifest에 실제 device를 기록하며, CUDA OOM이 model load에서 발생해도 CPU fallback 또는 명확한 fail-closed 진단을 제공한다.

### P2 — 운영 품질

1. Quick Start installer를 immutable commit/digest와 검증 hash로 pin한다.
2. 실제 Cosign-signed non-placeholder image/CAS bundle을 발행하고 fresh Ubuntu 24.04에서 재검증한다.
3. 루트 LICENSE와 제3자 약관 매트릭스를 확정하고 외부 endpoint 데이터 보존/상업 사용 조건을 명시한다.
4. CI에 lint/type/static security/coverage/targeted E2E gate를 추가한다.
5. score 옆에 단위·범위·“probability 아님”·evidence mode를 인라인 표시하고 색상 외 시각 부호를 제공한다.

---

## 12. 부록 A — 파일·artifact 범위

### A.1 기준 식별자

- 최초 HEAD: `190ac2ba6c79bfebc9f36370622f672c75b32c84`
- 보고 시점 HEAD: `53b63364cdfe5b7091f85a54da2e837c0367df60`
- 최초 / 보고 시점 추적 파일 수: 403 / 405
- 최초 `git ls-files -s` digest: `6a2b1c2ec120aa591189f71b7a914e5ac68e3523a00ff3cd0d12ed42b54bce91`
- 최초 tracked content aggregate: `6905773b3b916d3295c688fe5cb154e6fb16ecbcc87d5a4472dc0357c5a6fbdb`
- 보고 시점 `git ls-files -s` digest: `f4024585b183fe3513279f5aea6a2abcede1df6f96361b742a441fc04378937f`
- 보고 시점 `git ls-tree -r --full-tree HEAD` digest: `267460adb9d18edfd4549e93bd4af6a389b9d855c440cb6a4b3d41daa44d3c4e`
- 주요 추적 유형: Python 301, Snakemake 22, Markdown 17, shell 16, YAML 12, JSON 9, PNG 6

### A.2 저장량

| 경로 | 파일 수 | 크기 | 해석 |
|---|---:|---:|---|
| `results/` | 3,669 | 약 329 MB | 5개만 추적. 최초 기준보다 82개 늘어난 파일은 감사 중 생성된 ignored v2 결과이며, 총 3,664개가 비추적/ignored 상태 |
| `data/` | 293,883 | 123 GB | 대규모 외부/파생 데이터; Git commit만으로 재현 불가 |
| `.snakemake/` | 395,409 | 38 GB | 로컬 workflow cache/state; 릴리스 증거 아님 |

### A.3 최근 기준 변경

기존 감사 계획의 오래된 baseline 이후 현재 HEAD까지 AutoDock-GPU silent-fail 방어, Daina first-run 문서, predicted-target evidence 표시, input-scope, researcher guide, guide figure pin test 등이 추가됐다. 따라서 과거 baseline 결론을 그대로 재사용하지 않고 현재 HEAD에서 파일·테스트·결과를 다시 읽었다.

---

## 13. 부록 B — OMX 권한 진단

**Evidence.** 설치된 OMX는 `0.20.5`, Codex CLI는 `0.149.1`이다. `omx doctor`는 **19 pass / 1 legacy warning / 0 fail**이었다. 경고는 오래된 `features.multi_agent`, `agents.max_threads/max_depth` 보존 설정이며 이번 감사 권한 문제의 원인이 아니다.

`omx ralplan preflight --json`의 `unsupported_documented_leader_proof`는 손상된 설치나 사용자 승인 부족이 아니라 v0.20.5의 adapted-authority 경계 설계다. 로컬 구현은 문서화된 leader proof를 host가 검증할 수 없으면 의도적으로 실패한다. 이를 config 변경, 재설치, token 위조로 통과시키는 것은 지원되는 해결이 아니다.

공식 근거:

- https://github.com/Yeachan-Heo/oh-my-codex/blob/v0.20.5/docs/adr/3194-codex-01445-documented-leader-proof.md
- https://github.com/Yeachan-Heo/oh-my-codex/blob/v0.20.5/docs/adr/3212-same-user-native-child-auth-boundary.md
- https://github.com/Yeachan-Heo/oh-my-codex/blob/v0.20.5/docs/contracts/ralplan-consensus-gate.md
- https://github.com/Yeachan-Heo/oh-my-codex/releases/tag/v0.20.5

이번 감사는 gate를 spoof하거나 OMX core/config를 패치하지 않았다. 지원되는 ordinary native explicit-role agent 경로로 읽기 전용 병렬 감사를 수행했다. 이것이 권한 검증 문제의 안전한 운영상 해소다.

---

## 14. 부록 C — 감사 종료 조건

이 감사의 종료 조건은 다음이었다.

- 코드·핵심 논리·결과·시각화·해석의 claim-bearing surface를 모두 추적할 것
- Evidence/Inference/Unknown을 분리할 것
- 수치 재계산과 정적 검증 결과를 기록할 것
- 외부 동시 변경을 감사 자체 변경과 분리할 것
- 소스·데이터·결과를 수정하지 않고 단일 Markdown 보고서만 남길 것

이 조건은 충족했다. v2 전체 suite에서 기록된 간헐적 OOM 1건, 현 device-contract 공백, Git ignored v2 결과, v1 하류 계보, 보고 시점 미커밋 applicability 초안 때문에 릴리스 검증은 미완료다. 이후 CPU-hidden·GPU-visible targeted 재실행 통과는 함께 기록했으며, 보고서의 PASS 항목은 각 검사가 직접 입증한 좁은 범위에만 적용된다.
