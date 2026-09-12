# SkinScout 전 코드 초정밀 전문가 감사 보고서

> 감사일: 2026-09-04 KST
> 기준 브랜치: main
> 기준 HEAD: 9e632c97c6ef1ba185aeff87be3ac13c4e4b2fce
> 기준 커밋: fix: 서브프로세스 출력의 인코딩을 로케일에 맡기지 않는다
> 비교 기준: fc0fbf4b39fb71f12a485e93b8296a2e523ea48b
> 요청된 전문 패스: petr-expert-agent, coding-lord, coding-expert, biostatinfo-expert
> 최종 판정: **BLOCK / REQUEST CHANGES**
> 코드 수정: **없음**
> 감사 산출물: **이 Markdown 파일 1개만 생성**

## 1. 최종 결론

현재 SkinScout을 운영 배포, 재현 가능한 과학적 claim 생성, 또는 전체 기능 완료 상태로 승인할 수 없다.

현재 HEAD만으로도 다음 결함들이 서로 독립적인 차단 사유다.

1. Stage 8 생산자는 여러 target에 동일한 conformer 경로를 기록하고, 바로 다음 소비자는 그 중복을 거부한다. Stage 8을 기본 top-N 값 3으로 호출하면 xTB 계산 전에 확정적으로 중단된다.
2. Stage 7의 환경 정의는 pip의 --no-deps를 requirement로 기록하여 환경 생성 단계에서 실패한다. Stage 7 rule의 기본 config도 소비자가 필수로 요구하는 complex pose 입력을 제공하지 않는다.
3. 피부 효능 KG의 curated seed 숫자가 실제로 센 논문 수처럼 n_papers에 들어가고 사용자 출력에서 “562 papers”로 표시될 수 있다.
4. prospective promotion은 원시 예측과 정답 없이 작성자가 만든 aggregate CSV 한 행만으로 promote 판정을 낼 수 있다.
5. pLDDT가 낮은 내부 잔기를 제거한 clean PDB 잔기열을 canonical sequence처럼 연결해 MMseqs, ESM, PSICHIC에 재사용한다. 따라서 sequence-cold 및 sequence-model 관련 claim의 입력 의미가 깨진다.
6. public snapshot helper는 사용자가 준 출력 경로를 안전성 검증 없이 재귀 삭제한다. 잘못된 인자로 기존 임의 디렉터리를 영구 삭제할 수 있다.

작업 트리에는 감사 시작 전부터 scripts/stage6_bioemu.py의 미커밋 변경 65 additions / 6 deletions가 있었다. 이 변경은 residual strain이 남은 구조를 보존한 뒤 상태를 반환하지만 호출자가 그 상태를 버리고 정상 medoid로 전달하는 추가 HIGH 결함을 포함한다. 본 감사는 이 사용자 변경을 수정하거나 되돌리지 않았다.

중요한 범위 구분:

- workflow/config.yaml:11의 저장소 기본 mode는 comprehensive다. workflow/Snakefile:1450-1491의 rule all은 Stage 7 또는 Stage 8 output을 직접 열거하지 않고 Stage 9 report를 목표로 한다.
- 그러나 workflow/rules/stage9_report.smk:16-23,74-98은 MODE가 comprehensive 또는 both이면 molstar_report의 physics_reports에 mmgbsa와 dft_pyscf를 넣는다. 따라서 comprehensive/both의 rule all은 Stage 7/8을 **간접적으로 실제 요구**하며 현재 계약 결함의 영향을 받는다.
- 따라서 저장소 기본 comprehensive rule all은 Stage 7/8 결함으로 runtime 완료할 수 없다. MODE를 fast로 명시적으로 바꾸면 physics_reports가 비므로 이 결함만으로 fast ranking/report 경로가 차단되지는 않는다.
- performance-v2 neural branch는 workflow/config.yaml:334-359에서 비활성이다. 아래 neural finding은 현재 E0 경로의 즉시 회귀가 아니라, 이 branch의 승격 차단 조건이다.
- 탐색적 로컬 사용은 결과를 비임상·비확증적 연구 보조물로 명시하고 아래 결함의 영향을 수동 확인하는 조건에서만 제한적으로 가능하다.

## 2. 판정 기준과 증거 표기

심각도:

| 등급 | 의미 |
|---|---|
| BLOCKER | 선언된 경로가 기본 계약으로 실행 불가하거나 승인 결정을 직접 차단 |
| CRITICAL | 잘못된 과학적 claim, 다른 분자/구조 결과의 오귀속, 또는 비가역 데이터 손실 가능 |
| HIGH | 결과 타당성, 보안 경계, 재현성, 상태 전이를 중대하게 훼손 |
| MEDIUM | 특정 조건에서 편향·오작동·운영 불안정을 만들지만 즉시 전면 차단은 아님 |
| LOW | 문서, 유지보수, 제한적 일관성 또는 방어 심층성 결함 |

증거 상태:

| 표기 | 의미 |
|---|---|
| 동적 입증 | 격리된 무해한 재현 또는 실제 테스트로 현상이 관찰됨 |
| 소스 입증 | 제어 흐름과 데이터 흐름이 조건 없이 결함을 구성함 |
| 조건부 | 실제 영향이 입력·배포 형태·외부 도구에 의존함 |
| 검증 공백 | 필요한 E2E 또는 운영 환경이 없어 결론 범위를 제한함 |

## 3. 감사 스냅샷과 무결성

### 3.1 저장소 상태

- 감사 기준 HEAD: 9e632c97c6ef1ba185aeff87be3ac13c4e4b2fce
- 브랜치: main
- 이전 전수 감사 기준: fc0fbf4b39fb71f12a485e93b8296a2e523ea48b
- 기준 이후 변경: 19 commits, 55 files, 10,966 insertions, 334 deletions
- 기존 사용자 변경: scripts/stage6_bioemu.py, 65 insertions, 6 deletions
- 기존 변경 diff SHA-256: b6b4e95548ed73ddd380611ce1ac51d7eb96764b3b25301af02b5c6f51cbc643

### 3.2 전 행 검토 범위

| 영역 | 파일 수 | 행 수 | 검토 방식 |
|---|---:|---:|---|
| scripts production, tests/report_assets 제외 | 171 | 82,551 | 전 행, 호출부, subprocess, artifact 계약, 실제 표본 |
| eval 및 workflow | 64 | 36,983 | 전 행, rule I/O/params, gate, split, claim provenance |
| scripts/tests | 163 | 91,428 | 전 행, fixture, assertion 강도, skip, 격리·순서 의존성 |
| Workbench, contracts, compose, schemas, envs, adapters | 29 | 11,377 | 전 행, 신뢰 경계, 상태 전이, schema parity, 공급망 |
| docs/design | 3 | 3,177 | 전 행, 제품/정적 mockup 경계 |
| 루트 installer | 1 | 228 | 전 행, bootstrap 및 공급망 계약 |
| **합계** | **431** | **225,744** | first-party 코드·테스트·운영 계약 |

이번 “line by line” 검토는 다음 두 층으로 수행했다.

1. 이전 보고서의 415개 파일, 216,392행 전수 검토를 기준선으로 삼았다.
2. 그 이후 변경된 55개 파일 전체와 현재 dirty diff 전체를 다시 전 행 검토하고, 변경되지 않은 producer/consumer·schema·test 호출부까지 역추적했다.

따라서 단순 diff review가 아니다. 기존 전수 기준선과 현재 delta를 결합하고, 신규 경로가 접촉하는 이전 코드까지 다시 읽은 누적 전수 감사다.

### 3.3 제외 또는 경계 검토 범위

- workbench/static/vendor/** 및 scripts/report_assets/**의 minified/generated 제3자 내부 구현은 토큰 단위 전수 대상에서 제외했다.
- bundled scripts/report_assets/molstar/** 4개 파일 7,907행을 포함한 report asset 내부 6개 파일 8,268행은 로딩, CSP, iframe, 라이선스, 주입 경계만 검토했다.
- 대형 binary/raw data의 원자료 진위 자체는 코드 감사 범위가 아니다. manifest, hash, row count, provenance 및 소비 계약은 검토했다.
- 실제 CREST, xTB, PySCF DFT, 장시간 MD, full GPU model inference, 외부 API, clean OS installer E2E는 실행하지 않았다.
- 외부 제3자 모델 리뷰는 호출하지 않았다.

## 4. 사용한 전문가 워크플로

### 4.1 petr-expert-agent

파이프라인을 파일 목록이 아니라 artifact state transformation graph로 재정식화했다. 핵심 불변식은 다음과 같다.

각 artifact 상태는 최소한 다음 의미를 보존해야 한다.

    Artifact = (
        semantic_role,
        artifact_id,
        schema_version,
        input_sha256,
        output_sha256,
        code_revision,
        environment_digest,
        validation_predicate,
        validation_status
    )

각 edge는 이전 상태를 단조롭게 정제해야 하며, 의미를 약화하거나 다른 상태를 같은 ID로 합치면 안 된다. 이 관점에서 확인된 대표 위반은 다음과 같다.

- ligand-level free conformer artifact를 target별 row로 복제한 뒤 중복으로 거부한다.
- curated prior와 counted papers라는 다른 의미를 n_papers 하나로 합친다.
- residual strain status를 path-only manifest로 투영하며 검증 상태를 잃는다.
- gate가 검증한 index와 실제 production index가 다른데 같은 승인 상태처럼 사용한다.
- rigid transform으로 동치인 구조를 raw Cartesian 좌표에서 다른 conformation으로 취급한다.
- canonical sequence와 structure-observed residue sequence를 같은 sequence artifact로 취급한다.

Petr 판정: **REJECT**. artifact 의미와 동치 관계를 먼저 고정하지 않으면 국소 패치로는 같은 결함이 재발한다.

### 4.2 coding-expert 고정 5-lens panel

고정 context packet:

- 모드: review-only
- 대상: 현재 first-party 구현 전체와 기존 dirty Stage 6 변경
- 산출물: 단일 Markdown 보고서
- 금지: production/config/schema/test 수정, 외부 서비스 상태 변경
- 수용 기준: 파일:행 근거, evidence/inference 구분, 이전 감사 delta, 정적·동적 검증, 남은 공백 명시

| Lens | 적용 결과 | 판정 |
|---|---|---|
| Ken | status를 반환하는 sidechain 복원 API와 status를 버리는 호출자의 stream contract 불일치. Stage 7/8 optional/required 인자 계약도 shell 조합 뒤에야 깨짐 | REJECT |
| Petr | artifact 역할·동치·provenance를 명시한 단조 상태 전이로 환원해야 함. 현재 shared-path, pseudo-count, pseudo-sequence가 의미를 합침 | REJECT |
| Demis | 승격 실험은 sealed row-level truth/prediction과 preregistration을 재생해야 하나 aggregate self-report만 재평가함 | REJECT |
| Donald | conformer clustering은 rigid transform quotient 위에서 정의되어야 하나 raw 좌표 KMeans를 사용. 정확성 불변식 부재 | BLOCK |
| Gennady | duplicate path, malformed XYZ, row permutation, global time budget, tie boundary 같은 적대 입력에서 결과가 실패하거나 순서 의존 | REJECT |

패널 간 실질적 이견은 없었다. Ken은 최소 인터페이스 복구, Petr은 전역 artifact algebra, Demis는 결정적 실험, Donald/Gennady는 수학적 불변식과 적대 경계를 강조했지만 모두 현재 상태 승인을 거부했다.

### 4.3 biostatinfo-expert

다음 통계·생물정보학 축을 전수 검토했다.

- estimand와 comparator 정의
- evidence provenance와 독립 측정 보존
- temporal/sequence/structure leakage
- calibration, abstention, cold split
- multiple testing 및 model selection
- missingness와 candidate-pool comparability
- endpoint ontology, batch/target weighting
- tie, Top-K, row-order invariance

최종 과학 판정: **BLOCK**. 탐색적 사용은 가능하지만 현재 artifact만으로 확증적 성능, 문헌 근거 수, sequence-cold 일반화, prospective promotion을 주장하면 안 된다.

### 4.4 coding-lord

설치된 expert roster를 discovery helper로 검색해 20개 후보를 확인하고, 전체 코드 위험과 직접 관련된 14개를 선택했다. 상세 roster와 pass ledger는 12절에 기록했다.

## 5. 테스트와 정적 검증

### 5.1 실행 결과

| 검증 | 결과 | 해석 |
|---|---|---|
| 전체 test collection | **3,273 collected** | 전체 수집 경로는 성공 |
| 전체 pytest, 단일 uninterrupted run | **3,270 passed, 3 skipped, 0 failed / 33m00s** | 전체 green이지만 아래 composition·provenance·isolation gap은 assertion 대상이 아님 |
| Stage 4/6/7 표적 테스트 | **38 passed** | 개별 계약은 green이나 합성 결함 잔존 |
| Stage 8 상충 계약 테스트 | **2 passed** | 서로 반대 계약을 각각 승인하는 test-design defect |
| Stage 6 BioEmu/sidechain | **34 passed, 132 deselected** | rigid invariance와 strain propagation 미검증 |
| Workbench coordinator/server 선택 | **38 passed, 95 deselected** | ghost commit/restart child ownership 미검증 |
| Workbench durability 최소 선택 | **3 passed, 130 deselected** | coordinator re-claim만 검증 |
| KG/promotion 관련 | **42 passed, 32 deselected** | 두 CRITICAL provenance 결함이 녹색 suite에 남음 |
| security/UI 선택 | **13 passed, 126 deselected** | 저장형 HTML 경계 결함 잔존 |
| sequence 관련 scoped suite | **88 passed, 2 skipped** | standalone import와 의미 계약 미검증 |
| sequence PYTHONPATH 명시 실행 | **18 passed** | 외부 path 주입이 있으면 성공 |
| performance-v2, Torch 환경 | **87 passed** | 아래 학습 의미 결함은 assertion 대상 아님 |
| performance-v2, base 환경 | **31 passed, 2 skipped** | Torch 부재로 production CPU 학습 test skip |
| PSICHIC/RTMScore 선택 | **25 passed, 65 deselected** | upstream model E2E 아님 |
| activity operational gate | **PASS** | 이전 B-01은 해결 |

### 5.2 정적·workflow 검증

| 검증 | 결과 |
|---|---|
| Python compileall: scripts eval workbench skinscout psichic rtmscore | PASS |
| bash -n | PASS, shell 18개 |
| JSON parse | PASS, 16개 |
| YAML parse | PASS, 12개 |
| JSON Schema metaschema | PASS, 3개 |
| 현재 working diff git diff --check | PASS |
| Snakemake 9.23 isolated dry-run | PASS, 35 jobs. DAG 구성만 성공했으며 script/env/runtime contract 성공을 의미하지 않음 |
| Snakemake lint | FAIL, 64 lint sections / 90 warning bullets |
| JavaScript runtime syntax | 미실행: node, nodejs, deno, bun 없음 |
| ruff, mypy, bandit, shellcheck, coverage | 미설치, 결과 UNKNOWN |

Snakemake lint의 주된 항목은 hardcoded output prefix 44건, log 부재 8건, function/rule 혼합 7건, conda/container 부재 6건 등이다. 이는 이번 BLOCK 판정의 직접 원인은 아니지만 유지보수와 재현성 위험을 높인다.

### 5.3 의도적으로 확인한 실패

- python eval/evaluate_sequence_inductive_recipe.py --help는 ModuleNotFoundError로 종료 코드 1.
- 해당 test file만 격리 수집하면 종료 코드 2.
- PYTHONPATH에 repository scripts를 명시하면 관련 10개 test는 통과.
- envs/md.yml의 pip subsection을 requirements parser와 같은 방식으로 dry-run하면 --no-deps가 invalid requirement로 거부됨.
- 현재 base 환경에는 crest, xtb, gmx_MMPBSA, gmx, vina가 없었다. obabel, gnina, snakemake, nvidia-smi는 존재했다.

전체 pytest의 skip 3건:

1. scripts/tests/test_autodock_path_handling.py:84 — 현재 tmp root와 tmp_path가 같은 filesystem이라 해당 분기 skip.
2. scripts/tests/test_performance_v2.py:441 — base environment에 Torch 없음.
3. scripts/tests/test_performance_v2.py:1251 — base environment에 Torch 없음.

### 5.4 테스트 신뢰도 자체의 결함

- 74개 test file이 process-global sys.path.insert를 사용한다.
- 25개 test file이 sys.modules에 직접 값을 넣는다.
- random-order/randomly 실행 설정이 없다.
- artifact 관련 skip decorator 19개, bare pytest.skip 27개가 있으며 기본 run은 데이터 누락을 exit 0으로 허용할 수 있다.
- scripts/tests/conftest.py:45-105의 SKINSCOUT_REQUIRE_DATA strict mode는 skip reason 문자열 휴리스틱에 의존한다.
- 전체 suite가 green이어도 격리 import와 clean-clone artifact coverage를 증명하지 못한다.

필수 CI matrix는 default, provisioned SKINSCOUT_REQUIRE_DATA=1, isolated core files, randomized order, clean clone을 포함해야 한다.

## 6. BLOCKER 및 CRITICAL 발견

### B-01. Stage 8 producer와 consumer의 직접 계약 모순

**상태:** 신규 회귀, 동적 입증, Stage 8 경로 BLOCKER

근거:

- scripts/stage8_crest.py:192-205는 free ligand CREST를 한 번 실행하고 top-N target 모든 행에 같은 conformer_xyz를 기록한다.
- scripts/stage8_xtb_cluster.py:57-68은 canonical conformer path가 두 번 나오면 즉시 거부한다.
- workflow/rules/stage8_qm.smk:8-51은 두 단계를 직접 연결한다.
- workflow/config.yaml:247-253의 기본 top_n_for_qm은 3이다.
- scripts/tests/test_downstream_fail_closed.py:4237-4313은 경로 공유를 요구하고, 같은 파일 :4578-4616은 경로 공유를 거부하도록 요구한다.
- 두 테스트를 함께 실행한 결과는 2 passed다.

트리거:

- Stage 8을 target 두 개 이상으로 호출.

영향:

- xTB 시작 전에 CREST manifest reader에서 확정 실패.
- 분리된 unit test의 green 상태가 pipeline composition을 전혀 증명하지 못함.

최소 수용 계약:

- free-ligand ensemble을 ligand-level artifact 하나로 정의한 뒤 target association만 fan-out하거나, consumer가 명시적 free_ligand alias를 한 번만 계산하도록 해야 한다.
- producer가 만든 2개 이상 target manifest를 수정 없이 xTB와 DFT까지 전달하는 contract integration test가 필요하다.

주의:

- rule all은 Stage 8 output을 직접 열거하지 않지만 workflow/rules/stage9_report.smk:16-23,74-98을 통해 comprehensive/both에서 dft_pyscf를 간접 요구한다. workflow/config.yaml:11의 기본값이 comprehensive이므로 이 finding은 명시적 Stage 8 target뿐 아니라 저장소 기본 rule all도 차단한다.
- MODE fast의 Stage 9 physics_reports는 비어 있으므로 이 결함 하나가 fast 경로를 차단하지는 않는다.

### B-02. Stage 7 환경과 기본 rule contract가 완결되지 않음

**상태:** 신규, 소스 및 safe parser 입증, Stage 7 경로 BLOCKER

환경 결함:

- envs/md.yml:38-42의 pip requirements list에 "--no-deps"가 requirement처럼 들어 있다.
- pip requirements parser는 이를 option으로 허용하지 않고 Invalid requirement로 종료했다.
- gmx-mmpbsa 설치 전에 환경 생성이 실패할 수 있다.

workflow 결함:

- workflow/config.yaml:233-242에서 complex_pose와 complex_pose_manifest 기본값이 모두 빈 문자열이다.
- workflow/rules/stage7_md.smk:33-58은 빈 optional flag를 명령에서 생략한다.
- scripts/stage7_gromacs_prep.py:1165-1179는 두 인자가 모두 없으면 반드시 fail-closed한다.
- 관련 테스트는 explicit pose 성공과 no-pose 실패를 따로 검증하지만 기본 config와 rule command의 합성을 검증하지 않는다.

영향:

- 문서화된 Stage 7 환경을 clean build할 수 없고, 기본 Stage 7 rule은 필수 pose를 생성·전달하지 않는다.
- workflow/rules/stage9_report.smk:16-23의 comprehensive/both report가 mmgbsa를 요구하므로 이 결함은 명시적 Stage 7 target뿐 아니라 comprehensive/both rule all에도 전파된다.

최소 수용 계약:

- 지원되는 설치 방식으로 dependency suppression 정책을 정의하고 clean micromamba create와 gmx_MMPBSA/GROMACS smoke를 통과해야 한다.
- upstream pose artifact를 rule input으로 연결하거나 config에서 명시적으로 제공해야 한다.
- default config → rendered rule command → prep consumer를 잇는 composition test가 필요하다.

### C-01. Curated seed가 counted paper 수로 표시됨

**상태:** 기존에 식별되지 않은 CRITICAL scientific provenance 결함, 동적 입증

근거:

- scripts/stage0_skin_kg.py:55-73은 TYR whitening curated seed를 562로 둔다.
- 같은 파일 :237-243은 n_papers=562, n_papers_counted=0, n_papers_basis=curated_seed를 함께 기록한다.
- :296-307은 역사적 maximum을 유지한다.
- scripts/stage3_kg_efficacy_label.py:81-123,198-216은 basis/count를 보지 않고 n_papers로 정렬·판정한다.
- 같은 파일 :643-646은 사용자에게 “562 papers”로 표시한다.
- scripts/stage9_report.py:1099-1117은 label 존재를 literature support로 사용한다.

fresh 재현:

    n_papers=562
    n_papers_counted=0
    n_papers_basis=curated_seed
    output=[("whitening", 562)]

로컬 ignored runtime graph에서도 P14679에 n_papers=562와 표본 PMID 5개만 있었고 counted/basis field는 없었다. 이는 HEAD artifact가 아니라 ambient artifact 보조 증거로만 취급했다.

영향:

- 전문가 prior 또는 seed weight가 실제 문헌 건수로 오인된다.
- 효능 정렬, 라벨, 보고서의 literature support claim이 과장될 수 있다.

최소 수용 계약:

- prior_weight와 counted_publications를 별도 field·type으로 분리한다.
- counted paper는 검증된 고유 PMID/DOI 집합에서만 계산한다.
- seed-only edge가 paper count나 literature-supported claim을 만들지 못하는 회귀 테스트가 필요하다.

### C-02. Aggregate self-report만으로 prospective promotion 가능

**상태:** 이전 H-02 지속, CRITICAL, 동적 입증

근거:

- eval/prospective_promotion_eval.py:91-151은 raw prediction/truth가 아닌 aggregate CSV 한 행을 읽는다.
- :228-243은 preregistration SHA가 64자리 hex 형식인지 중심으로 검사한다.
- :244-268은 그 행의 수치로 promote를 판정한다.
- :367-379의 fresh recomputation도 같은 aggregate CSV를 다시 평가한다.
- eval/run_iteration.py:3146-3153에서는 candidate experiment가 not_run이어도 promotion-ready 경로가 가능하다.
- scripts/tests/test_prospective_promotion_eval.py:43-180도 작성된 promotion_row를 정상 승격 fixture로 사용한다.

동적 재현:

    status=promote
    candidate=FABRICATED
    provenance=made_up
    failed_gates=0

영향:

- 원시 관찰 없이 표본 수, Top-K, confidence lower bound, calibration delta를 자기보고해 모델을 승격할 수 있다.
- preregistration, model binary, evaluation release, raw records의 독립성이 결속되지 않는다.

최소 수용 계약:

- sealed row-level truth와 prediction에서 evaluator가 metric, CI/bootstrap, calibration을 직접 재계산해야 한다.
- preregistration 내용 hash, model artifact hash, dataset release membership, sample independence를 검증해야 한다.
- aggregate CSV는 계산 결과이지 신뢰 root가 되어서는 안 된다.

### C-03. pLDDT-trimmed pseudo-sequence가 canonical sequence 역할을 함

**상태:** 신규 claim-blocking representation 결함, 소스 및 데이터 scan 입증

근거:

- scripts/stage0_clean_alphafold.py:111-118,147-159는 pLDDT cutoff 아래 잔기를 내부 위치까지 삭제한다.
- scripts/stage0_mmseqs_build.sh:33-68은 clean PDB에 남은 CA 잔기를 순서대로 이어 FASTA를 만든다.
- data/validation/alphafold_human_v4_sequence_source.json:12-16도 human_clean.fasta가 pLDDT-trimmed PDB에서 재구축됐다고 기록한다.
- eval/build_screenable_target_cluster_map.py:108-167,320-350은 이 계열 FASTA를 cold cluster에 사용한다.
- scripts/build_target_sequence_embeddings.py:231-357은 이 서열을 ESM 입력으로 사용한다.
- scripts/stage3_psichic.py:143-163,241-306과 psichic/__init__.py:97-112도 clean PDB 잔기열을 사용한다.

fresh scan:

- clean PDB 20,171개 중 14,539개에 내부 residue-number gap 존재.
- 내부 누락 위치 합계 1,648,276개, retained CA 7,645,944개.
- Q86TB3은 raw 2,170 residues 중 clean/FASTA 441 residues만 연결되고 내부 1,685 위치가 빠진다.

영향:

- 원래 인접하지 않은 잔기가 인접한 peptide처럼 ESM과 PSICHIC에 입력된다.
- 구조 confidence filtering이 sequence identity와 MMseqs cluster를 바꾸어 sequence-cold split 정의까지 오염시킨다.
- 기존 sequence-derived artifact와 그 평가 지표를 canonical sequence 기반 결과로 해석할 수 없다.

최소 수용 계약:

- accession별 canonical full sequence를 MMseqs, ESM, PSICHIC의 공통 source로 사용한다.
- pLDDT mask는 structure computation에만 적용하고 sequence artifact는 바꾸지 않는다.
- sequence hash/length parity를 검증하고 cluster, embedding, cold split, 모든 관련 metric을 재생성한다.

### C-04. Public snapshot helper가 임의 경로를 재귀 삭제

**상태:** 신규 CRITICAL safety 결함, 소스 입증, 위험한 실제 재현은 의도적으로 미실행

근거:

- scripts/build_public_snapshot.sh:16-18은 사용자가 제공한 OUT을 그대로 받는다.
- :32-34는 canonicalization, ownership, sentinel, symlink, ancestor 검증 없이 rm -rf "$OUT"을 수행한다.
- script 자체 위치가 아니라 호출 당시 cwd에서 git archive HEAD를 실행한다.
- embedded redactor도 :59-64에서 Path.cwd()를 root로 사용한다.
- scripts/tests/test_public_snapshot.py:24-31은 새 temporary output만 사용하며 unsafe target을 검증하지 않는다.

영향:

- typo, 절대경로, symlink, repository root, unrelated existing directory를 인자로 주면 복구 불가능한 삭제가 가능하다.
- 외부 cwd에서 호출하면 OUT을 먼저 삭제하고 archive/redaction이 실패하거나 다른 repository를 snapshot할 수 있다.

최소 수용 계약:

- script directory에서 canonical repository root를 결정한다.
- /, home, repository root/ancestor, symlink, existing nonempty unrelated directory를 거부한다.
- fresh 또는 명시적으로 소유권이 입증된 output만 허용한다.
- 각 unsafe target에서 sentinel이 보존되는 비파괴 test가 필요하다.

### W-01. 미커밋 Stage 6 변경이 residual strain을 정상 medoid로 전달

**상태:** 기존 dirty worktree 한정, HIGH, 동적 입증

근거:

- scripts/stage6_bioemu.py:455-472는 완화 후 strain이 남아도 구조 파일을 보존하고 pdbfixer_sidechains_openmm_relaxed_with_strain을 반환한다.
- :545-548의 caller는 return status를 버리고 medoid path를 결과에 추가한다.
- :602-615의 manifest는 target_id와 path만 기록한다.
- scripts/stage6_ensemble_dock.py:113-128,159-175는 sidechain 존재 비율만 확인하고 strain status를 받지 않는다.
- scripts/stage6_bioemu.py:293-311의 표시 label과 :326-335의 relaxation matching은 chain identity를 잃는다.

동적 재현:

- strain status를 반환하도록 한 격리 재현에서 cluster_conformers가 해당 medoid를 정상 출력으로 유지했다.
- A/B chain의 같은 ASP residue 번호는 같은 label로 축약됐다.

영향:

- 내부 검증이 부적합하다고 판정한 receptor가 docking과 후속 scoring으로 진입한다.
- 다중 chain에서는 다른 chain의 같은 residue 번호까지 함께 완화될 수 있다.

최소 수용 계약:

- exact success status만 eligible medoid로 수용한다.
- strain artifact는 진단 경로로 격리하고 manifest/downstream validator가 상태를 fail-closed한다.
- residue identity에 model, chain, residue number, insertion code를 포함한다.

## 7. HIGH 발견 — 구조·물리·화학 경로

| ID | 발견 및 근거 | 영향 | 최소 수용·검증 |
|---|---|---|---|
| H-01 | scripts/stage8_crest.py:118-147,190-205는 charge/spin을 계산·기록하지만 실제 CREST 명령에 --chrg/--uhf를 넣지 않음. 명령 capture로 입증 | charged/open-shell ligand를 다른 전자 상태로 sampling하면서 manifest는 올바른 값을 주장 | anion/open-shell별 실제 command와 output provenance test |
| H-02 | scripts/stage8_xtb_cluster.py:104-144,165-196은 multi-frame XYZ 전체에 xTB를 한 번 호출하고 첫 TOTAL ENERGY 하나만 읽음. frame score, cluster, representative select 없음 | stage 명칭과 달리 conformer reranking이 없고 downstream DFT는 첫 frame 사용 | frame split → per-frame score → deterministic cluster/select E2E |
| H-03 | scripts/stage8_dft.py:117-138은 malformed row를 건너뛰고 declared atom count와 실제 count를 비교하지 않음. 2 atoms 선언 + 1 valid + 1 malformed가 H 한 원자로 반환됨 | 잘린 다른 분자의 DFT energy가 원 ligand에 귀속 가능 | n>=1, 정확히 n개, valid element, finite coordinates, extra/truncated row 거부 |
| H-04 | scripts/stage6_bioemu.py:530-546은 CA 좌표를 superposition 없이 flatten해 KMeans | rigid rotation/translation으로 동치인 구조가 서로 다른 conformation으로 분리 | Kabsch-aligned RMSD, distance/internal-coordinate representation, rigid invariance test |
| H-05 | scripts/stage7_mmgbsa.py:186-245는 실행 전 FINAL_RESULTS_MMPBSA.dat를 격리하지 않고 nonzero여도 기존 파일이 있으면 parse. stale -99 또는 -42.5 반환 재현 | 이전 run의 ΔG가 현재 target/trajectory 결과로 기록 | fresh attempt directory 또는 pre-delete, output hash/mtime, recognized analyzer-only failure만 허용 |
| H-06 | scripts/stage4_prepare_structures.py:422-474는 chain_id를 받지만 같은 CCD ligand의 다른 chain/model copy까지 하나의 pocket에 합침. A/B distant LIG 재현 center 약 55.5, radius 약 54.5 | 두 binding site나 assembly 전체를 덮는 무의미한 docking box | 선택 chain의 ligand instance 또는 nearest unambiguous instance, ambiguity fail-closed |
| H-07 | scripts/stage4_prepare_structures.py:315-357은 alignment 없이 zip identity를 계산하고 :516-525는 identity threshold 없이 coverage만 검사. 1-aa offset true target identity 0.0, unrelated equal-length 0.1도 coverage 통과 | 실제 target chain 거부 또는 unrelated chain 선택 | global/local alignment, prespecified identity+coverage, RCSB entity mapping, indel/fusion tests |
| H-08 | scripts/stage3_psichic.py:143-163은 insertion code를 residue key에서 빼 100A/100B 중 하나를 잃음 | sequence-only 표현 손실 | canonical FASTA primary, PDB parity check, insertion-code fixture |
| H-09 | scripts/alternative_ingredients.py:197-207,725,742-750의 MCS global deadline은 row 순서에 따라 뒤쪽 후보의 평가 기회를 바꿈 | 같은 후보 집합도 입력 순서에 따라 core retention 결과 변경 | per-pair budget 또는 deterministic scheduling, row permutation invariance |
| H-10 | scripts/alternative_ingredients.py:269-309은 substructure와 MCS에서 chirality를 강제하지 않음. lactic acid·phenylalanine enantiomer가 coverage/share 1.0 contains로 재현 | 반대 입체화학을 완전한 core retention으로 표시 | chirality-aware policy 또는 stereo_mismatch/unresolved 상태, enantiomer/diastereomer tests |

## 8. HIGH 발견 — 데이터·모델·통계 경로

### 8.1 핵심 목록

| ID | 발견 및 근거 | 영향 | 최소 수용·검증 |
|---|---|---|---|
| H-11 | index 표준화 scripts/build_activity_retrieval_index.py:225-254는 FragmentParent만, query scripts/stage3_daina_zoete.py:97-134는 Uncharger까지 적용. charged acid self similarity 0.64 재현 | self-match, identity exclusion, nearest rank 왜곡 | 단일 canonicalization 함수, charged/zwitterion exact-self=1 tests, index 재생성 |
| H-12 | operational gate는 evaluation index를 승인하지만 workflow/config.yaml:164,189의 production index는 data/activity_retrieval_runtime_merged_202608. workflow/rules/stage3b_fast.smk:58-61,81-118은 둘을 별도로 전달 | gate PASS가 실제 production index를 승인했다는 보장 없음 | gate artifact에 runtime recipe/index/universe digest 결속, exact-match tamper test |
| H-13 | scripts/merge_runtime_activity_evidence.py:365-384는 canonical structure-target pair가 ChEMBL에 있으면 다른 source의 publication/endpoint/relation/value를 제거 | 독립 반복·상충 evidence와 measurement 수가 사라짐 | publication/assay/endpoint/relation/value provenance 단위 dedup |
| H-14 | eval/skin_efficacy_recovery_eval.py:196-220은 rank 검증 없이 read_csv(...).head(10) 사용 | 행 순서만 바꿔 metric/gate 변화 | explicit rank sort, uniqueness/continuity 검증, permutation test |
| H-15 | scripts/eval_alternative_criteria.py:131-136,590-605의 vs_baseline은 항상 size_baseline만 사용하나 docs는 best null 대비로 기술. synthetic method=.6, size=.4, polarity=.9에서도 +.2/win=1 | 방법이 더 강한 null에 지는데 이겼다고 보고 | comparator 사전 지정, 두 baseline 모두 보고 또는 per-query conservative max |
| H-16 | scripts/eval_alternative_criteria.py:430-465,530-588은 criterion마다 다른 candidate mask에서 AUC를 만든 뒤 scalar를 paired 비교 | chemistry-dependent missingness 아래 서로 다른 ranking task를 같은 estimand처럼 비교 | method-pair별 공통 candidate mask에서 AUC 재계산, coverage와 sensitivity 기록 |
| H-17 | scripts/eval_alternative_criteria.py:166-209,573-631,776-800은 다수 criterion/stratum 검정을 p<.05로 해석하며 family correction 없음; Wilcoxon tie variance도 보정하지 않음 | 선택적 false positive와 과소/과대 분산 | family 정의, Holm/FDR 또는 simultaneous CI, exact/tie-aware test |
| H-18 | eval/run_sequence_inductive_iteration.py:481-527,689-723은 5개 candidate의 여러 point metric을 1e-12 차이로 비교하고 최대 합계로 선택 | model-selection multiplicity와 작은 target 집중을 일반화로 오인 | target bootstrap CI, nested/held-out selection, matched sequence identity baseline |
| H-19 | scripts/build_target_sequence_embeddings.py:400-470,519-530은 model 의미를 기록하지만 eval/sequence_inductive_retrieval.py:158-228은 model/embedding/universe_policy를 강제하지 않음. metadata 없는 fixture도 통과 | 임의 벡터가 provenance-bound ESM2로 해석 가능 | model ID, immutable revision/hash, layer, pooling, chunk policy fail-closed |
| H-20 | eval/performance_v2_model.py:1806-1811,1871-1900은 독립 seed 비선형 head를 hidden alignment 없이 parameter 평균. 기능적으로 동치인 permutation 모델 평균이 출력 최대 1.2407366 변화 | robustness ensemble이 다른 함수의 model soup이 됨 | per-seed prediction/rank ensemble 또는 weight matching; frozen dev/dual-cold ablation |
| H-21 | eval/performance_v2_model.py:795-813,1251-1279,1732-1741,1836-1886은 target-balanced weight를 batch마다 다시 정규화하고 OOM 시 batch size 변경 | batch/shard/OOM에 따라 최적화 목적함수 변경. 예시 global 5.0 대 batch-step 3.3333 | global normalization 또는 unbiased sampler, full-vs-minibatch gradient equality |
| H-22 | scripts/build_performance_v2_embeddings.py:555-560,621-735은 ligand max token length를 manifest에 기록하지 않고 scripts/score_performance_v2.py:55-80,119-124는 독립 query length 사용 | train/serve tokenizer truncation mismatch가 계약 통과 | tokenizer revision, truncation, max length, count 결속; offline/online parity |
| H-23 | prereg cutoff 2023-12-31은 eval/preregistered_performance_v2.py:17-19에 선언되지만 scripts/prepare_performance_v2_inputs.py:26-35,208-246이 split/date/release를 강제하지 않음 | future/test rows도 training adapter가 수용 가능 | adapter fail-closed와 upstream split manifest hash. 현재 실제 train snapshot에서는 미래 leakage 미관찰 |
| H-24 | scripts/prepare_performance_v2_inputs.py:356-380은 ontology별 head 전에 ligand-target pair로 endpoint를 합쳐 conflict를 unknown_mixed로 제거 | direct binding과 functional endpoint supervision 소실 | ligand-target-ontology 단위 보존, ontology 내부 conflict만 해결 |
| H-25 | scripts/stage3_psichic.py:295-319은 max window score를 4 decimals로 저장하고 scripts/stage3_disagreement.py:83-103,259-272는 score-only top-N. eval/run_psichic_panel_benchmark.py:150-165는 score,target_id라는 다른 tie rule 사용 | 길수록 max 기회 증가; 15개 table 중 10개에서 top-50 boundary tie, membership 순서 의존 | length/window calibration, full precision, 공통 tie rule, permutation/cutoff tie test |

### 8.2 biostatinfo-expert의 estimand 판정

| 분석 대상 | 현재 estimand 문제 | 판정 |
|---|---|---|
| KG literature support | curated prior와 observed publication count 혼합 | CRITICAL |
| Prospective promotion | raw observation이 아니라 self-reported aggregate를 estimand로 사용 | CRITICAL |
| Activity dedup | independent measurement unit이 pair 하나로 붕괴 | HIGH |
| Alternative criteria baseline | “best null” claim과 size-only comparator 불일치 | HIGH |
| Alternative criteria AUC | method별 다른 candidate population | HIGH |
| Sequence recipe selection | 같은 dev에서 다중 recipe·다중 metric 선택 후 point estimate 보고 | HIGH |
| Skin efficacy Top-10 | rank가 아니라 input order를 estimand에 포함 | HIGH |
| Performance ontology | endpoint ontology를 학습 전에 pair 하나로 합침 | MEDIUM-HIGH |
| Weighted BCE | batch 구성에 따라 target-balanced estimand 변화 | HIGH |
| Temporal cutoff | 선언되지만 adapter에서 강제되지 않음 | MEDIUM; 현재 snapshot leakage는 미관찰 |

### 8.3 확인된 통계·ML 강점

- eval/performance_v2_model.py:877-953의 calibration holdout은 train-only이며 pair-hash disjoint다.
- :686-714는 gray/unmeasured를 BCE negative로 사용하지 않는다.
- :988-1034는 measured positive/negative가 없는 ontology에서 probability를 생성하지 않는다.
- :2302-2325는 dual-cold/structure-abstained row에서 probability를 비운다.
- eval/build_activity_benchmark.py:1391-1439,1525-1545는 cold flag를 명시한다.
- eval/build_activity_recovery_panels.py:1011-1167은 canonical pair와 inverse weights를 사용한다.
- scripts/build_target_sequence_embeddings.py:181-272는 FASTA/target coverage와 artifact hash를 엄격히 검사한다.
- ESM inference는 eval/no-grad와 residue-only pooling을 사용한다.

이 강점들은 개별 내부 통제를 입증하지만 입력 의미, promotion trust root, comparator 정의 결함을 상쇄하지 않는다.

## 9. HIGH 발견 — Workbench·보안·운영 경로

| ID | 발견 및 근거 | 영향 | 최소 수용·검증 |
|---|---|---|---|
| H-26 | scripts/make_results_viewer.py:1110-1112는 raw JSON을 script element에 넣고 scripts/stage9_report.py:3019,3025-3026은 raw JSON을 pre HTML에 삽입. 종료 tag가 context-aware escape되지 않음 | 악성/오염 PDB·SDF·metadata가 저장형 script/HTML injection 가능 | script-safe JSON serialization, HTML escape, CSP/sandbox, browser fixture |
| H-27 | eval/performance_v2_model.py:2223-2226은 torch.load(weights_only=...) TypeError 시 제한 없는 torch.load로 재시도 | 구버전 Torch에서 untrusted checkpoint pickle 실행 가능 | safe format/state_dict, trusted hash, 버전 gate; fallback 제거 |
| H-28 | workbench/server.py:4665-4671은 access token 자체를 출력하고 scripts/start_workbench.py:98-124는 results/logs/workbench.log로 redirect. 일반 append 생성 mode가 0664일 수 있음 | bearer token이 terminal/log/group-readable 파일에 노출 | token 값 미출력, restrictive umask/0600, rotation/redaction |
| H-29 | workbench/server.py:3853-3894,421-428,4521-4525,4624-4665는 remote bearer/cookie auth를 지원하지만 transport는 HTTP 가능, cookie Secure 강제 없음 | 원격 배포 시 token/cookie 탈취 가능. 단 로컬 개인 사용에서는 조건부 | non-loopback TLS 강제, Secure cookie, reverse-proxy trust contract |
| H-30 | workbench/server.py:1856-1932는 child를 start_new_session으로 실행하지만 PID/PGID/birth identity를 durable state에 기록하지 않고 shutdown :4673-4679도 child를 조정하지 않음; workbench/coordinator.py:1195-1238은 lease 후 requeue | server restart 뒤 기존 child와 retry가 중복 실행되거나 cancel 불능 | process identity 영속화, adopt/terminate/quarantine 정책, restart-mid-process E2E |
| H-31 | workbench/coordinator.py:1096-1139은 caller가 30초 timeout을 받아도 queued mutation을 취소하지 않음. 이후 create_job이 실제 commit되는 ghost commit 재현 | caller 재시도와 늦은 원 transaction이 중복 mutation | queue deadline/cancel, 실행 전 expiry check, timeout 후 no-mutation test |
| H-32 | workbench/coordinator.py:687-705는 LIMIT 후 status filtering. 1,000개 선행 row 밖의 running job이 목록에서 사라지는 재현 | admission/monitoring이 active job을 놓침 | WHERE status predicate를 LIMIT 전에 DB query에 적용 |
| H-33 | workbench/coordinator.py:400-439,816-824는 artifact identity를 content와 job association 양쪽에 사용. 같은 content를 다른 job에 등록하면 invalid transition | dedup과 ownership 의미 충돌 | content object ID와 job binding ID 분리 |
| H-34 | workbench/server.py:3509-3520은 verification error가 있어도 item status completed, batch도 completed로 종료 | 실패한 검증이 성공 batch처럼 표시 | completed/failed/partial 상태를 verification과 일치시킴 |
| H-35 | workbench/server.py:2254-2261,3633-3696의 local job은 artifact registry promotion 없이 direct result path로 제공 | hash/verification/immutability 우회 | local·remote 동일 registry promotion gate |
| H-36 | workbench/server.py:1621-1627은 safety job이 GPU 없이 가능하다고 표시하지만 :1844-1852,1871-1879,3683-3696의 process 기본 resource는 gpu:0 | CPU-only safety 작업이 불필요한 GPU admission에 묶임 | job type별 explicit resource contract |
| H-37 | workbench/server.py:1960-1967의 tail은 매 요청 전체 log read 후 48KB slice. downloads/bundles도 :2289-2338,3896-3923,4080-4109에서 큰 파일을 통째로 읽음 | 큰 artifact polling에서 메모리·I/O 급증 | seek-based bounded read와 streaming response |
| H-38 | workbench/server.py:233,1925-1926,3382,3471-3473의 _JOBS와 _BATCHES 전역 map은 완료 후 TTL/LRU 제거가 없음 | 장시간 서버에서 object graph 누적 | durable completion 뒤 bounded archival/eviction |
| H-39 | scripts/build_similarity_index.py:124-154가 여러 file을 transactional set으로 publish하지 않고 scripts/similar_compounds.py:129-152는 schema/row count만 검사 | 중단 시 서로 다른 세대 arrays/table 혼합 | staging dir + manifest hashes + atomic directory switch |
| H-40 | workbench/server.py:2849 부근의 similarity map build는 process once-lock이 없음. key-map load 실측 RSS 약 1.29GB, map 후 약 1.35GB였고 동시 build가 가능 | Workbench 동시 요청에서 메모리·latency spike | process-level once lock, compact index, measured limit |
| H-41 | scripts/report_email.sh:10-11,23-58은 고정 수신자와 hostname/network/hardware metadata를 전송 | 다른 사용 환경에서 의도하지 않은 외부 전송. 개인 전용이면 심각도 하향 가능 | explicit recipient, opt-in fields, redaction |

보안상 확인된 강점:

- Workbench는 compare_digest, CSRF 검사, path containment, request size 제한을 여러 경로에 적용한다.
- public snapshot은 report_email과 raw review dump를 제외하고 redaction을 시도한다.
- 다만 위의 token logging, transport, destructive output, HTML context 경계가 남아 있어 배포 승인에는 충분하지 않다.

## 10. HIGH·MEDIUM 발견 — config, provenance, 재현성

| ID | 등급 | 발견 및 근거 |
|---|---|---|
| P-01 | HIGH | scripts/run_skinscout.py:648-710은 extra config에 reserved key를 허용하고 :2534-2567에서 canonical config 뒤에 append. run-id/mode/smiles override를 주면 launcher가 기록한 값과 Snakemake 실제 값이 갈리는 split-brain 재현 |
| P-02 | HIGH | scripts/run_skinscout.py:1112-1153 및 skinscout/contracts/run_profile.py:302-323의 run manifest는 전체 workflow config가 아니라 축약 payload hash를 저장하고 image는 null, data hash는 ambient manifest set |
| P-03 | HIGH | workflow/config.yaml:225-230의 tICA_lag_steps는 validation되지만 workflow/rules/stage6_bioemu.smk:19-35와 scripts/stage6_bioemu.py:554-575에 전달되지 않음. 실제 구현은 raw KMeans |
| P-04 | HIGH | scripts/build_runtime_retrieval_index.py:171-215는 threshold finite/order, workers>=1, max_unusable>=0를 강제하지 않음 |
| P-05 | MEDIUM-HIGH | scripts/stage2_consensus.py:141-201 및 Snakefile:1092-1097은 halt_min_votes 최소 1만 강제. 4를 주면 3/3 positive도 HALT 불가 |
| P-06 | HIGH | README.md installer 안내와 install_skinscout.sh:164-190은 mutable main/default branch를 checksum/signature 없이 내려받아 실행 |
| P-07 | MEDIUM | eval/run_all.sh:33-59는 malformed evaluation config를 default로 대체해 claim threshold 오류를 숨김 |
| P-08 | MEDIUM | workflow/rules/stage3b_fast.smk:30의 문자열 bool 처리에서 "false"가 truthy가 될 수 있음 |
| P-09 | MEDIUM | workflow/rules/stage3b_fast.smk:49-61,107-118과 scripts/stage3_daina_zoete.py:844-862는 recipe mode도 쓰지 않는 ChEMBL mirror input을 강제해 production-index 환경을 불필요하게 차단 |
| P-10 | MEDIUM | eval/build_transfer_reachable_panel.py:72-100 및 eval/build_pocket_cold_panel.py:161-223은 upstream snapshot/index hash를 충분히 결속하지 않음 |
| P-11 | MEDIUM | workflow/config.yaml:232-247 및 eval/build_activity_benchmark.py:85-92,495-503의 common cutoff와 activity train boundary가 2023-10-02~12-31 구간에서 다르게 정의됨 |
| P-12 | MEDIUM | schemas/target_fast_v2.json:57-67 및 skinscout/contracts/run_profile.py:105-115,662 사이 JSON Schema와 Python enum/API export가 drift. structure_not_requested는 schema 허용, Python validator 거부 |
| P-13 | MEDIUM | scripts/monitor_stage0.sh, scripts/stage0_download_hpa.sh, eval/measure_panel_similarity_bands.py의 monitor/HPA/band artifact가 obsolete path, mutable URL, existence-only reuse 또는 hash 없는 manifest에 의존 |
| P-14 | MEDIUM | envs/**, scripts/install_runtime.py:389-450, docs/LICENSE_POLICY.md의 pin/solver/license 목록과 실제 설치 version 사이 drift; lockfile/SBOM이 완전하지 않음 |
| P-15 | LOW | docs/ALTERNATIVE_CRITERIA_EVAL.md는 같은 문서 안에 53/20과 57/24 표본 설명이 공존. 실제 substitute_pairs.csv는 57 data rows |
| P-16 | LOW | workbench/static/index.html:411의 stereo fallback 9/51 문구는 current docs의 4/51 및 exact-first 구현과 불일치 |
| P-17 | LOW | eval/measure_pocket_transfer.py:146-193의 Pocket transfer Top-1% cutoff가 universe와 무관하게 202로 고정 |
| P-18 | LOW | eval/disagreement_eval.py:112-116은 missing target class와 실제 orphan class를 같은 label로 합침 |
| P-19 | LOW | workflow/Snakefile:1399-1400은 Snakemake parse 중 read-only 명령에서도 directory를 만들 수 있음 |
| P-20 | LOW | workflow/rules/stage3a_comprehensive.smk:50의 dirname command substitution 결과가 shell-safe하게 quote되지 않음 |
| P-21 | LOW | docs/ARCHITECTURE.md:1-47과 scripts/pipeline_readiness.py:1218-1252의 architecture/readiness naming이 v2에 머물러 current contracts와 혼동 가능 |

## 11. 이전 감사 해결 상태

이 표의 ID는 docs/SKINSCOUT_LINE_BY_LINE_AUDIT_20260901.md 기준이다.

| 이전 ID | 현재 상태 | 현재 근거 |
|---|---|---|
| B-01 | **RESOLVED** | 현재 activity operational gate 실제 PASS |
| B-02 | **PARTIAL** | recipe ID/digest 결속 개선. production index는 gate가 승인한 evaluation index와 다름 |
| H-01 | **PERSISTENT** | index FragmentParent 대 query Uncharger 불일치 |
| H-02 | **PERSISTENT / CRITICAL** | aggregate one-row promotion trust 지속 |
| H-03 | **PARTIAL** | 0은 차단됐으나 상한 3 없음 |
| H-04 | **RESOLVED** | server/UI가 status==ok 및 verifier_status==ok를 사용 |
| H-05 | **PERSISTENT** | script/pre HTML context escape 미해결 |
| H-06 | **PARTIAL** | fallback source는 raw payload에 보이나 primary failure reason/degraded 상태 유실 |
| H-07 | **RESOLVED** | curated failure-mode CSV 누락·오류 fail-closed |
| H-08 | **PERSISTENT** | CREST charge/spin 명령 미전달 |
| H-09 | **REGRESSED TO BLOCKER** | xTB cluster 미구현에 producer/consumer duplicate path 모순 추가 |
| H-10 | **PERSISTENT** | pair-level dedup 지속 |
| H-11 | **PERSISTENT** | hardcoded email/host metadata. 개인 전용 환경이면 MEDIUM |
| H-12 | **PERSISTENT** | runtime index threshold/workers/max-unusable 검증 부재 |
| H-13 | **PERSISTENT** | .head(10) 지속 |
| H-14 | **PERSISTENT** | PID/PGID 없는 process recovery |
| H-15 | **PARTIAL / PERSISTENT** | 반환 log는 bounded지만 전체 read, timeout callback 취소 없음 |
| H-16 | **PERSISTENT** | scripts/stage2_5_drug_avoidance.py의 os._exit 유지 |
| H-17 | **PERSISTENT** | mutable main installer 유지 |

이전 MEDIUM/LOW:

- Persistent: M-01~07, M-09, M-11, M-14, M-16~20, L-01~04, L-06.
- Partial: M-08, M-10, M-12, M-13, M-15, L-05.
- M-10은 rank sort가 추가됐지만 positional subset Top-K 의미는 남아 있다.
- M-14 schema/Python enum drift는 남아 있다.

## 12. Coding Lord roster와 expert pass ledger

### 12.1 발견된 expert roster 20개

| Expert | 선택 | 이유 |
|---|---|---|
| biostatinfo-expert | 선택 | scientific estimand, leakage, multiplicity, calibration, evidence |
| citing-expert | 제외 | citation-only live source verification이 현재 code-correctness 핵심이 아니며 외부 근거 요청 없음 |
| debugging-expert | 선택 | runtime failure, stale artifact, fail-open, resource lifecycle |
| demis-expert-agent | 선택 | decisive experiment 및 promotion mechanism |
| dennis-expert-agent | 제외 | first-party C/ABI 구현 없음 |
| donald-expert-agent | 선택 | invariants, correctness, equivalence, complexity |
| env-mixture-expert | 제외 | BKMR/qgcomp/WQS/ToxPi/DML mixture analysis 없음 |
| gennady-expert-agent | 선택 | adversarial edge cases, ordering, bounds |
| geoffrey-expert-agent | 선택 | neural training, gradient objective, parameter averaging |
| grace-expert-agent | 선택 | config/CLI/workflow language와 deterministic semantics |
| innovator-expert | 제외 | cross-domain mechanism invention 요청이 아님 |
| jeff-expert-agent | 선택 | Workbench queue, concurrency, state, tail latency |
| ken-expert-agent | 선택 | composable stream/interface and narrow contracts |
| kevin-mitnick-expert | 선택 | auth, bearer token, operational human/security boundary |
| linus-expert-agent | 선택 | regression, compatibility, lifecycle, hot path |
| novelty-expert | 제외 | scientific novelty/prior-art 검색이 요청 범위가 아님 |
| petr-expert-agent | 선택 | representation reformulation and equivalence |
| tavis-expert-agent | 선택 | untrusted input, parser, harmless local repro, destructive boundary |
| yann-expert-agent | 제외 | first-party CV/image/video training surface 없음 |
| yoshua-expert-agent | 선택 | sequence, attention/embedding, tokenization, generalization |

### 12.2 선택된 14개 pass ledger

| Expert | Path | Lens applied | Findings | Concrete recommendation | Verification required | Decision | Notes |
|---|---|---|---|---|---|---|---|
| biostatinfo | KG, promotion, retrieval evidence, alternative eval, sequence/performance eval | estimand·bias·multiplicity·calibration | seed=count 혼합, self-report promotion, pair dedup, comparator/population mismatch | observation unit과 comparator를 사전 정의하고 raw-level 재계산 | sealed raw records, bootstrap, family correction, permutation/cold checks | ACCEPT / BLOCK | exploratory use만 조건부 |
| debugging | Stage 6-8, Workbench | failure path·stale output·lifecycle | Stage8 contract conflict, stale MMGBSA, partial XYZ, ghost commit, orphan child | attempt-scoped outputs, fail-closed parser, durable process identity | producer→consumer E2E, forced failure, restart-mid-process | ACCEPT | 동적 재현 다수 |
| demis | prospective promotion | decisive experiment | not_run 또는 fabricated aggregate도 promotion-ready | sealed preregistered experiment replay | raw truth/prediction recomputation and future release | ACCEPT / REJECT PROMOTION | current claim invalid |
| donald | BioEmu, Stage8 | invariant·proof·complexity | rigid transform invariance 부재, Stage8 state contract 모순 | quotient-space distance와 명시적 state machine | rotation/translation proof test, exact transition test | ACCEPT / BLOCK | correctness first |
| gennady | parsers, Top-K, time budget, ties | adversarial boundary | duplicate path, malformed XYZ, row-order MCS, tie cutoff | deterministic total order, local budgets, strict cardinality | fuzz/property/permutation/boundary tests | ACCEPT | worst-case 중심 |
| geoffrey | performance-v2 | gradient flow·objective·representation | unaligned parameter averaging, batch-dependent weighted BCE, dynamics observability 부족 | prediction ensemble, unbiased objective, training diagnostics | gradient equality, per-seed ablation, tiny-set overfit | ACCEPT / P1 BLOCK | branch currently disabled |
| grace | config, CLI, Snakemake | language semantics·diagnostics | extra config reserved-key override, ghost config, Stage7 required/optional mismatch | one typed schema and rendered-command contract | unknown/reserved mutation, default composition test | ACCEPT | config split-brain |
| jeff | Workbench, similarity index | concurrency·durability·tail latency | orphan/requeue, ghost commit, post-LIMIT filtering, large reads/maps | durable identities, cancellation, transactional publish, bounded IO | restart/load/concurrent build tests | ACCEPT | single-host도 해당 |
| ken | shell/API boundaries | composability·narrow interface | status/path 의미 손실, destructive OUT, optional flag late failure | explicit result type, no ambient cwd, narrow stream contract | status propagation and unsafe-target tests | ACCEPT | 최소 repair 방향 |
| kevin-mitnick | Workbench auth, report helper | defensive auth·human factors | token logging, HTTP cookie risk, implicit email metadata | secret redaction, TLS boundary, explicit recipient | log scan without secret, non-loopback transport tests | ACCEPT / CONDITIONAL | 개인 localhost면 일부 하향 |
| linus | delta compatibility | regression·lifetime·performance | green tests가 interface regressions를 못 잡고 map/full-read 비용 큼 | composition tests, stable API/manifest, measured hot path | old/new artifact compatibility and memory benchmarks | ACCEPT | 55-file delta 전수 |
| petr | entire DAG | representation·equivalence | artifact role conflation, pseudo-sequence, raw-coordinate equivalence error | content-addressed typed artifact algebra | monotonicity, identity, equivalence property tests | ACCEPT / REJECT | explicit requested skill |
| tavis | HTML, XYZ/PDB/SDF, snapshot path, checkpoint | untrusted input·safe repro | stored injection, partial molecule, arbitrary delete, unsafe torch fallback | strict parsers, context escape, path allowlist, safe serialization | malicious local fixtures, sentinel preservation, no code execution | ACCEPT / BLOCK | 실제 destructive repro 미실행 |
| yoshua | ESM, MoLFormer, PSICHIC, sequence-cold | sequence representation·tokenization·generalization | pLDDT pseudo-sequence, weak embedding contract, max-window/tie, standalone evaluator | canonical sequence, bound embedding policy, deterministic aggregation | regenerate clusters/embeddings/metrics; isolation and tie tests | ACCEPT / BLOCK | claim-blocking |

### 12.3 Coding Lord 합성

가장 높은 가치의 공통 결론은 “각 단계가 green인 것”과 “단계 간 의미 계약이 보존되는 것”이 다르다는 점이다.

- Stage 8은 양쪽 unit test가 모두 green인데 합성 경로는 불가능하다.
- KG와 promotion 관련 42개 test가 green인데 핵심 provenance가 틀리다.
- sequence suite가 green인데 standalone evaluator는 import조차 못 하고 canonical sequence 의미가 깨져 있다.
- Workbench durability test가 green인데 실제 child ownership을 검증하지 않는다.

따라서 단순 test count는 승인 근거가 아니다. release gate는 artifact 의미, producer-consumer composition, raw evidence recomputation, clean-process isolation을 직접 검증해야 한다.

## 13. Test-design gap 상세

다음 회귀 검증은 현재 없거나 충분하지 않다.

1. Stage8 CREST 실제 output manifest를 수정 없이 xTB reader와 DFT까지 연결하는 top_n=3 test.
2. Stage7 default config와 rendered rule command를 GROMACS prep consumer에 전달하는 test.
3. clean envs/md.yml build와 gmx_MMPBSA/GROMACS/RDKit smoke.
4. raw row-level prospective truth/prediction으로 metric과 CI를 재계산하고 prereg/model hash를 검증하는 test.
5. seed-only KG edge가 counted paper 또는 literature claim을 만들지 못하는 test.
6. canonical full sequence와 structure mask가 분리되며 internal structure deletion이 sequence artifact를 바꾸지 않는 test.
7. BioEmu rigid rotation/translation invariance test.
8. 동일 residue number를 가진 다중 chain strain selection test.
9. residual strain status가 manifest와 ensemble docking에서 fail-closed되는 test.
10. Stage4 duplicate ligand chain, N-terminal truncation, internal gap, unrelated equal-length chain, fusion test.
11. stale FINAL_RESULTS_MMPBSA.dat + early subprocess failure test.
12. truncated, malformed, nonfinite, extra-row XYZ 거부 test.
13. charged/open-shell CREST command provenance test.
14. charged/zwitterion index-query exact-self=1 test.
15. operational gate와 runtime recipe/index exact digest binding tamper test.
16. alternative criteria의 stronger-null, common candidate mask, multiplicity, missingness sensitivity test.
17. PSICHIC row permutation, full precision, boundary tie, window-count calibration test.
18. embedding manifest model/revision/layer/pooling/chunk/max-token mutation test.
19. Workbench restart 중 실제 child survival, adopt/terminate, cancel, retry policy test.
20. coordinator timeout 뒤 mutation이 절대 commit되지 않는 short-deadline test.
21. batch item verification failure가 batch success가 되지 않는 test.
22. public snapshot unsafe target와 outside-cwd sentinel preservation test.
23. script/pre context의 closing-tag injection browser test.
24. isolated core module imports와 randomized suite.
25. provisioned artifact strict-mode 및 clean-clone CI.

## 14. 확인된 구현 강점

- 3,273개 test를 수집하는 넓은 회귀 표면이 있다.
- 여러 artifact writer가 temp file 후 atomic replace를 사용한다.
- retrieval/evaluation 여러 경로가 schema, input SHA, target universe, role을 fail-closed한다.
- 현재 activity operational gate는 실제로 PASS한다.
- Workbench server/UI verification status 역전 문제는 해결됐다.
- curated receptor failure-mode file 누락은 현재 fail-closed한다.
- subprocess text encoding을 locale에 맡기지 않는 개선이 HEAD에 포함됐다.
- host doctor는 placeholder container image를 숨기지 않고 차단한다.
- compose policy와 여러 외부 binary에 version/hash 검증 경로가 있다.
- performance-v2는 gray/unmeasured를 negative로 오인하지 않고, calibration pair split과 abstention을 비교적 엄격히 다룬다.
- sequence embedding producer는 coverage, hash, finite output, resolved model metadata를 기록한다.
- public snapshot은 민감 helper 제외와 문자열 redaction 의도를 갖는다.
- Snakemake 현재 config는 isolated dry-run에서 DAG를 생성했다.

이 강점은 구현의 방어적 의도와 상당한 테스트 투자를 보여준다. 다만 상위 의미 계약과 test composition이 깨져 있어 BLOCK 판정을 바꾸지는 못한다.

## 15. 최소 수용 순서

### P0 — release/claim 차단 해제

1. Stage 8 artifact cardinality와 역할을 하나로 정하고 CREST → per-frame xTB → deterministic select → strict DFT E2E를 만든다.
2. Stage 7 환경 정의를 clean build 가능하게 만들고 pose artifact를 workflow input으로 연결한다.
3. KG prior와 counted publication을 분리하고 사용자 claim은 검증된 publication set에서만 만든다.
4. prospective promotion을 sealed raw record 재계산으로 전환한다.
5. canonical protein sequence를 구조 mask에서 분리하고 MMseqs/ESM/PSICHIC artifact와 지표를 모두 재생성한다.
6. public snapshot output path의 비가역 삭제를 안전 계약으로 감싼다.
7. dirty Stage 6의 residual-strain status를 downstream까지 보존하고 fail-closed한다.

### P1 — 과학 타당성

1. query/index chemical standardization을 단일 함수로 통일하고 index를 재생성한다.
2. pair dedup을 독립 measurement provenance 단위로 바꾼다.
3. alternative evaluator의 comparator, common candidate pool, multiplicity를 사전 정의한다.
4. sequence model selection에 target bootstrap, matched identity baseline, future snapshot을 추가한다.
5. performance-v2는 parameter ensemble과 weighted objective를 교정한 뒤 승격 검증한다.
6. 모든 Top-K 경로에 rank schema, stable tie policy, permutation invariance를 적용한다.

### P2 — 운영·보안·재현성

1. Workbench process identity, shutdown/restart ownership, queue cancellation, artifact registry를 통합한다.
2. token logging과 non-TLS remote auth를 차단하고 HTML context escape를 적용한다.
3. config override를 typed single source of truth로 만들고 full resolved config hash를 manifest에 결속한다.
4. similarity index를 transactional set으로 publish하고 memory/once-lock을 검증한다.
5. mutable installer를 release digest/signature 기반으로 전환한다.
6. lint, typecheck, security scan, coverage, random-order, clean-clone strict artifact CI를 추가한다.

## 16. 승인 게이트

다음이 모두 fresh evidence로 통과하기 전에는 release/claim 승인을 권고하지 않는다.

- 전체 pytest failure 0, skip 사유와 artifact strict-mode 결과 명시.
- isolated core test와 direct CLI entrypoint 모두 clean process에서 성공.
- Stage 7 clean environment와 default composition smoke 성공.
- Stage 8 top_n=3 actual producer-consumer E2E 성공.
- KG 표시 paper count와 고유 검증 publication 집합 exact match.
- prospective promotion metric이 sealed raw records에서 재계산되고 artifact hashes가 일치.
- canonical sequence 기반 cluster/embedding/PSICHIC 재생성 및 cold metrics 재검증.
- Stage 6 rigid invariance와 residual-strain fail-closed test 통과.
- production gate의 recipe/index/universe/code/env exact binding.
- Workbench restart-mid-process에서 orphan·duplicate·ghost mutation 없음.
- HTML injection, unsafe snapshot path, unsafe checkpoint fixture가 모두 fail-closed.
- clean clone에서 required artifacts와 environments를 재현 가능.

## 17. Report-only 구현 계약

사용자 지시에 따라 본 작업은 구현이 아니라 감사 보고에 한정했다.

- 선택한 접근: 기존 전수 감사 기준선 + 현재 55-file delta + dirty diff + 호출 경계 재추적 + 정적/동적 검증.
- 보존 불변식: production code, config, schema, test, data, environment file을 수정하지 않는다.
- 허용된 유일한 write: 이 Markdown 보고서.
- 배제한 대안: 발견 즉시 patch, test 보강, dependency 설치, destructive snapshot 재현, 외부 API/third-party model 호출.
- 재검토 trigger: HEAD 변경, dirty Stage 6 변경, Stage 7/8 contract 수정, KG/promotion/sequence artifact 재생성, Workbench lifecycle 수정.
- stop condition: 보고서가 현재 snapshot, evidence, risk, acceptance test를 자급적으로 기록하고 working tree에 이 보고서 외 신규 변경이 없을 때.

## 18. 감사 무결성 선언

- 소스, config, schema, test, data artifact를 수정하지 않았다.
- 기존 scripts/stage6_bioemu.py 변경을 그대로 보존했다.
- 위험한 rm -rf 재현, 실제 외부 전송, production service 조작을 하지 않았다.
- 테스트와 최소 재현은 temporary directory, pytest tmp path, read-only data scan 또는 mock subprocess를 사용했다.
- access token이나 credential file 내용을 읽거나 보고서에 기록하지 않았다.
- 이 보고서의 결론은 현재 HEAD와 명시된 dirty diff에만 유효하다.
- 최종 판정은 **BLOCK / REQUEST CHANGES**다.
