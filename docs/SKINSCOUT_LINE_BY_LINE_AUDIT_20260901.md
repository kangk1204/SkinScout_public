# SkinScout 초정밀 코드·테스트 감사 보고서

> 감사 기간: 2026-08-31 ~ 2026-09-01 (KST)  
> 최종 감사 스냅샷: `fc0fbf4b39fb71f12a485e93b8296a2e523ea48b`  
> 브랜치: `harden-workflow-gates`  
> 최종 판정: **REQUEST CHANGES**  
> 아키텍처 판정: **BLOCK**  
> 코드 수정: **없음** — 본 보고서 파일만 생성

## 1. 결론

SkinScout은 대규모 테스트와 다수의 fail-closed 검증을 갖춘 연구용 파이프라인이지만, 현재 상태를 운영·배포 또는 과학적 claim 생성에 승인할 수 없다.

가장 직접적인 차단 사유는 다음 두 가지다.

1. 기본 MODE-FAST 경로가 현재 operational activity-retrieval gate에서 실제로 종료 코드 1로 실패한다.
2. gate가 승인한 evaluation recipe와 Stage 3가 실제로 읽는 production recipe/index가 하나의 content-addressed 계약으로 묶여 있지 않다. 현재 config recipe는 gitignored 로컬 artifact이고, workflow가 재생성하는 recipe와도 다르다.

그 밖에도 화학 표준화 불일치, 자기보고 집계 CSV만으로 가능한 모델 승격, 안전 HALT 임계값 우회, Workbench 검증 표시 역전, 저장형 script injection, Stage 8 전자상태·conformer 처리 결함 등 다수의 HIGH 결함이 확인됐다.

독립 코드 리뷰 권고는 `REQUEST CHANGES`, 독립 아키텍처 리뷰는 `BLOCK`이었다. 두 판정을 결정론적으로 합성한 최종 권고도 `REQUEST CHANGES`다.

## 2. 감사 대상과 스냅샷

감사 시작 후 같은 공유 작업 트리에서 다른 세션이 여러 차례 커밋을 추가했다. 감사 중 관찰한 주요 HEAD 이동은 다음과 같다.

`10ba9b9f` → `fb2263f5` → `37da8d2a` → `5cbea180` → `590697fc` → `9a4d5da2` → `fc0fbf4b`

따라서 중간 실행 결과와 최종 스냅샷을 혼동하지 않도록 다음 원칙을 적용했다.

- 최종 코드 판정과 행 번호는 `fc0fbf4b` 기준이다.
- 전체 pytest는 최종 구현 부모 `590697fc`에서 실행했다. 그 뒤 추가된 것은 README, 정적 디자인 문서/이미지, 그리고 5개 README 전달 테스트뿐이다.
- 최종 delta 5개 테스트와 JavaScript 구문 검사는 `fc0fbf4b`에서 별도로 통과했다.
- 전체 실행에서 실패한 1개 테스트는 최종 HEAD에서 단독 재실행해 통과 여부와 로그를 교차확인했다.
- 보고서 생성 직전 작업 트리는 clean이었다.

## 3. 검토 범위

### 3.1 전 행 검토 범위

| 영역 | 파일 | 행 수 | 검토 방식 |
|---|---:|---:|---|
| `scripts/**` production (`tests`, 생성 vendor 제외) | 166 | 78,460 | 전 행 + 호출 계약 + 실제 데이터 표본 |
| `eval/**`, `workflow/**` | 64 | 36,978 | 전 행 + rule input/output/params + gate/provenance 추적 |
| `scripts/tests/**` | 152 | 87,409 | 전 행 + 수집/skip/fixture/검증 강도 감사 |
| Workbench·contracts·compose·schemas·envs·adapters | 29 | 10,140 | 전 행 + 신뢰 경계·상태 전이·스키마 parity |
| `docs/design/**` 정적 mockup | 3 | 3,177 | 전 행 + 브라우저 smoke + production 여부 구분 |
| 루트 installer | 1 | 228 | 전 행 + dry-run/공급망 경계 |
| **합계** | **415** | **216,392** | first-party 구현·테스트·운영 계약 |

추가로 독립 코드 리뷰어가 약 408개 파일/212,275행을 광역 재검토했고, 아키텍처 리뷰어가 evaluation gate, production scorer, run manifest, Workbench, installer 경계를 별도로 추적했다.

### 3.2 제외 또는 경계 검토만 수행한 범위

- `workbench/static/vendor/**`, `scripts/report_assets/**`의 minified/generated 제3자 코드 내부는 토큰 단위 전수 검토에서 제외했다.
- 해당 vendor의 로딩 방식, CSP, iframe 격리, 라이선스, 결과 HTML 경계는 검토했다.
- 대형 binary/data artifact 자체의 과학적 원자료 진위는 코드 감사 범위를 벗어난다. 다만 manifest/hash/row-count/provenance 결속은 검토했다.
- GPU 모델, 외부 웹 API, 실제 CREST/xTB/DFT, clean Ubuntu 설치, signed container pull은 환경 제약으로 E2E 실행하지 못했다.

## 4. 테스트 및 검증 결과

### 4.1 테스트 결과 요약

| 검증 | 결과 | 해석 |
|---|---|---|
| 최종 HEAD test collection | **3,010 collected** | 수집 오류 없음 |
| 전체 pytest (`590697fc`) | **3,001 passed, 1 failed, 3 skipped** / 32m10s | source 구현 전체. 1건은 native teardown flake |
| 실패 테스트 단독 재실행 (`fc0fbf4b`) | **1 passed** / 6.00s | 순서·native teardown 의존으로 판단 |
| 최종 delta README 테스트 | **5 passed** / 0.03s | `590697fc..fc0fbf4b` 추가 테스트 전부 통과 |
| 최신 retrieval/UI/workflow 영향 범위 | **168 passed** / 3m03s | recipe/operator/gate/UI/config 관련 |
| Python `compileall` | PASS | `scripts eval workbench skinscout psichic rtmscore` |
| Shell `bash -n` | PASS | first-party shell 17개 |
| Node syntax | PASS | `workbench/static/app.js`, `docs/design/support.js` |
| JSON/YAML parse | PASS | 저장소의 구조화 파일 파싱 |
| JSON Schema metaschema | PASS | `schemas/*.json` 3개 |
| `git diff --check` | PASS | whitespace 오류 없음 |
| Snakemake 9.23 dry-run | PASS | 최종 config, `compound_smiles=CCO`, DAG 생성 |
| Snakemake lint | FAIL | 종료 1, 56개 rule에 lint section; hardcoded prefix/log/긴 run 등 |
| activity operational gate | **FAIL** | `activity retrieval frozen baseline recipe definition is invalid` |
| host CLI doctor | **FAIL** | app/target/physics 이미지 placeholder 3개 차단 |

전체 pytest는 Snakemake가 실제 PATH에 포함되도록 실행했다. 이를 생략하면 관련 33개 workflow 테스트가 도구 부재로 조용히 skip될 수 있다.

### 4.2 전체 pytest의 단일 실패

실패:

`scripts/tests/test_eval_quality_fail_closed.py::test_eval_run_all_requires_collected_run_provenance_for_claims`

관찰 결과:

- 예상 실패는 `missing collected run provenance manifest`였다.
- 실제로는 세 번째 `cold_start_eval.py --mode dti_only` 종료 시 `SIGABRT`, return code 134가 먼저 발생했다.
- 로그 마지막 줄은 정상 metric 출력 뒤 `terminate called without an active exception`이었다.
- 같은 테스트를 최종 HEAD에서 단독 재실행하면 `1 passed in 6.00s`였다.

판정: 기능 assertion의 결정적 실패는 아니지만, pandas/Parquet/native runtime teardown이 전체 실행 순서에서 프로세스를 비결정적으로 abort할 수 있다는 운영 안정성 결함이다. 녹색 단독 재실행만으로 원인이 해결된 것은 아니다.

### 4.3 Skip 3건

1. `test_autodock_path_handling.py`: 임시 경로가 같은 filesystem을 공유해 해당 분기 skip.
2. `test_performance_v2.py` 2건: 현재 환경에 `torch`가 없어 skip.

### 4.4 실행하지 못한 품질 도구

환경에 `ruff`, `mypy`, `bandit`, `shellcheck`, `coverage`가 없었다. 따라서 정량 line/branch coverage는 **UNKNOWN**이며, lint/type/security 수치가 없다는 사실 자체를 품질 공백으로 기록한다.

## 5. 동적 재현 핵심 증거

| 항목 | 재현 결과 |
|---|---|
| query/index 표준화 | 같은 charged compound가 index canonical anion, query canonical neutral acid; Morgan Tanimoto **0.64** |
| 안전 vote threshold | 3개 모델 모두 positive인데 `halt_min_votes=4` → `FLAG_HIGH`, `HALT` 아님 |
| prospective promotion | 임의로 만든 1행 aggregate CSV → `status=promote`, failed gates 0; preregistration 파일은 존재하지 않음 |
| recipe tampering | promoted JSON weight를 임시 변경해도 `load_promoted_recipe()`가 같은 recipe ID로 수용 |
| viewer injection | PDB 문자열의 `</script><script ...>`가 생성 HTML에 그대로 존재 |
| Workbench verification | raw artifact는 `status=ok`, `verifier_status=ok`, `ok` 필드 없음; UI 식은 false, 서버 정규화는 true |
| operational gate | 종료 코드 1, frozen baseline recipe definition invalid |

## 6. Release Blocker

### B-01. 현재 기본 MODE-FAST가 operational gate에서 즉시 실패 — 확정

근거:

- `scripts/validate_activity_retrieval_gate.py:20-33`의 frozen baseline에는 `max_union_nonneg`가 포함된다.
- 현재 gate가 가리키는 기존 recipe artifact의 baseline에는 이 필드가 없다.
- `scripts/validate_activity_retrieval_gate.py:1004-1012`는 baseline dict 전체 일치를 요구한다.
- `workflow/rules/stage3b_fast.smk:93-118`은 Stage 3 scorer 실행 전에 gate check를 항상 수행한다.
- 실제 명령은 `activity retrieval frozen baseline recipe definition is invalid`로 종료 코드 1을 반환했다.

영향: 테스트가 대부분 통과해도 기본 fast 실행은 현재 artifact 조합으로 완료될 수 없다.

필요 조치: recipe schema/version migration을 명시하고 recipe, dev/final manifests, operational gate를 같은 revision에서 재생성해야 한다. 완료 기준은 현재 checkout의 `check-operational` 성공이다.

### B-02. 승인된 gate와 실제 production recipe/index가 하나의 계약이 아님 — 확정

근거:

- gate는 `union_any_consensus`, SHA-256 `714e9b...`를 기록한다: `data/manifests/activity_retrieval_operational_gate.flag:48,70-76`.
- Stage 3 config는 별도 `dev_selection_skin/recipe.json`을 읽는다: `workflow/config.yaml:143-164`.
- 해당 파일은 `.gitignore:45`의 `results/eval/` 아래에 있고 git이 추적하지 않는다. 현재 로컬 SHA-256은 `a50229...`다.
- workflow가 생성하는 recipe는 `workflow/rules/activity_retrieval.smk:377-411`의 `dev_selection/recipe.json`이며, config의 `dev_selection_skin` artifact를 만드는 rule이나 `--prefer-recipe` 전달은 없다.
- `workflow/rules/stage3b_fast.smk:58-61,81-118`은 gate와 recipe를 별도 input/argument로 전달하고 둘의 ID/SHA를 비교하지 않는다.
- `scripts/stage3_recipe_scoring.py:46-75`는 schema와 `passes_dev_gate`만 확인한다. 변조된 임시 recipe도 동적 재현에서 수용됐다.
- gate의 recipe path는 저장소가 아니라 `/tmp/.../recipe.json`이다: gate line 72.

영향:

- clean clone에서 현재 runtime recipe를 재생성하거나 검증할 수 없다.
- gate를 통과시킨 recipe와 실제 scoring recipe가 달라질 수 있다.
- production index, recipe, target universe, code/env identity가 한 claim contract로 결속되지 않는다.

필요 조치: evaluation decision, selected recipe SHA, production index manifest/SHA, target-universe SHA, label policy, code/env identity를 하나의 immutable production scoring contract로 묶고 Stage 3에는 이 contract 하나만 전달해야 한다.

## 7. HIGH 발견

### H-01. Index와 query가 서로 다른 화학 표준화를 사용 — 확정

- Query: `scripts/stage3_daina_zoete.py:97-133` — `Uncharger(FragmentParent(...))`.
- Index: `scripts/build_activity_retrieval_index.py:225-254` — `FragmentParent(...)`만 사용.
- production scoring도 index builder의 `_standardize_mol`을 import하지만 run query fingerprint는 별도 query path에서 생성한다.

재현 compound `N#Cc1ccc(C(=O)[O-])cc1`은 index에서 anion, query에서 neutral acid가 됐고 self-similarity가 1.0이 아닌 0.64였다. 실제 index 조사에서도 bracket charge를 가진 SMILES가 다수 존재했다.

영향: self-match, leave-query-out identity exclusion, nearest similarity, target rank가 charged/zwitterionic compound에서 잘못될 수 있다.

### H-02. Prospective promotion이 자기보고 1행 CSV를 신뢰 — 확정

`eval/prospective_promotion_eval.py:91-149,149-268,367-379`

원시 예측·정답·bootstrap sample을 읽지 않고 CSV 한 행의 표본 수, Top-K, confidence lower bound, calibration delta를 그대로 사용한다. `preregistration_sha256`은 64자리 hex 형식만 확인하며 실파일 존재/hash 일치를 검사하지 않는다. “fresh recomputation”도 같은 aggregate CSV를 다시 평가한다.

동적 재현에서 임의 숫자와 존재하지 않는 preregistration hash만으로 `status=promote`가 생성됐다.

### H-03. 안전 HALT 임계값이 설정으로 무력화 가능 — 확정

`scripts/stage2_consensus.py:141-160,163-201`, `workflow/Snakefile:1092-1097`

`halt_min_votes`에 `1 <= threshold <= 3` 제약이 없다. 세 모델이 모두 positive여도 threshold 4이면 `FLAG_HIGH`다. threshold 0도 즉시 HALT가 된다.

영향: 안전 판단이 잘못된 config/CLI override 하나로 fail-open 또는 과잉 차단될 수 있다.

### H-04. Workbench 검증 표시가 정상/누락 양쪽에서 역전 — 확정

`workbench/static/app.js:425-444`, `workbench/server.py:2096-2129,2559-2569`

- 상세 API는 raw `run_verification.json`을 반환한다.
- raw schema는 `status`와 `verifier_status`를 사용하며 `ok`가 없다.
- UI는 `verification.ok === true`를 검사한다.
- verification이 아예 `null`이면 UI는 오히려 `verified=true`로 둔다.

결과적으로 정상 검증은 claimable false로, 검증 누락은 summary 값에 따라 claimable true로 표시될 수 있다.

### H-05. 결과 HTML에 저장형 script/HTML injection 경로 — 확정

1. `scripts/make_results_viewer.py:1110-1112`: PDB/SDF 전체 텍스트를 `json.dumps` 후 `<script type="application/json">`에 직접 삽입한다. JSON encoding은 `<\/script>`를 자동 방어하지 않는다.
2. `scripts/stage9_report.py:3016-3036`: disagreement/ADMET/alert JSON과 CosIng level/INCI/functions를 HTML escape 없이 삽입한다.
3. legacy report/page에는 이를 보완할 충분한 CSP가 없다.

동적 재현에서 구조 문자열의 `</script><script id=...>`가 출력 HTML에 그대로 들어갔다.

### H-06. STopTox fallback이 primary 실패를 정상 증거처럼 숨김 — 확정

`scripts/stage2_stoptox.py:60-69,213-237`

REST 경로의 광범위한 `except Exception`이 HTML fallback으로 전환한다. fallback 성공 시 최종 status는 `ok`이고 원래 오류·primary source 실패가 provenance에 남지 않는다. 네트워크 오류뿐 아니라 parser/programming defect도 정상 안전 evidence로 마스킹될 수 있다.

### H-07. 필수 receptor failure-mode 데이터 누락 시 경고가 조용히 사라짐 — 확정

`workbench/server.py:54,64-74,2473-2476`, `scripts/target_failure_modes.py:30-55`

원 loader는 누락/손상 CSV에 fail-closed하지만 Workbench는 `SystemExit`을 잡아 빈 dict를 캐시한다. 금속/cofactor/GPCR 상태 등 알려진 docking 한계가 아무 degraded 표시 없이 UI에서 사라질 수 있다.

### H-08. CREST에 계산한 charge/spin을 전달하지 않음 — 확정

`scripts/stage8_crest.py:118-147,174-198`

charge와 spin을 계산·manifest에 기록하지만 실제 명령은 `crest ligand.xyz -gfn2 -T 8 -niceprint`이며 `--chrg`, `--uhf`가 없다. charged/open-shell ligand가 잘못된 전자 상태로 conformer sampling될 수 있다.

### H-09. `xTB cluster`가 conformer별 score/cluster/select를 하지 않음 — 확정

`scripts/stage8_xtb_cluster.py:117-185`, `scripts/stage8_dft.py:117-138,223-259`

multi-frame `crest_conformers.xyz` 전체를 xTB에 한 번 넘기고, conformer 분리·개별 에너지·clustering·대표 선택이 없다. DFT XYZ parser는 첫 frame만 읽는다. 단계 명칭과 달리 ensemble reranking이 수행되지 않는다.

### H-10. Pair-level dedup이 독립 측정 evidence까지 제거 — 확정

`scripts/merge_runtime_activity_evidence.py:364-386`

canonical `(structure, target)`가 ChEMBL에 있으면 BindingDB/GtoPdb 행을 publication, endpoint, relation, value와 무관하게 모두 제거한다. 실제 parquet 교차검사에서 같은 pair이지만 ChEMBL에 없는 publication을 가진 행이 수천 건 확인됐다.

영향: 독립 문헌 측정, 상충 측정, endpoint별 evidence가 사라져 support/consensus/label이 왜곡될 수 있다.

### H-11. Email helper가 고정된 제3자 주소로 내부 정보를 전송 — 확정

`scripts/report_email.sh:10-11,23-58`

수신자·발신자가 `<개인 주소>`으로 고정되어 있고 hostname, Tailscale/LAN IP, CPU/GPU/RAM, disk, commit을 자동 첨부한다. msmtp가 설정된 다른 사용자가 실행하면 보고서와 내부 네트워크 정보가 명시적 recipient 선택 없이 전송된다.

### H-12. Production retrieval index builder의 핵심 입력 검증 누락 — 확정

`scripts/build_runtime_retrieval_index.py:94-111,171-215`

다음 검증이 없다.

- threshold finite 여부
- `negative < positive` 순서
- `workers >= 1`
- `max_unusable_ligands >= 0`

NaN 또는 역전 threshold가 대규모 label 오분류 index를 만들 수 있다.

### H-13. Skin efficacy Top-10이 rank가 아니라 CSV 행 순서 — 확정

`eval/skin_efficacy_recovery_eval.py:196-220`

ranking을 정렬하거나 rank 연속성/중복을 확인하지 않고 `read_csv(...).head(10)`을 사용한다. 같은 데이터의 행 순서만 바꿔도 precision/recall과 gate 결과가 달라진다.

### H-14. Workbench 재시작 후 child process가 고아·중복 상태가 될 수 있음 — 확정

`workbench/server.py:1827-1871,4205-4210`, `workbench/coordinator.py:1195-1238`

local job은 `start_new_session=True`로 실행하지만 PID/process-group identity가 durable record에 없다. 서버 shutdown은 coordinator만 닫고 child를 재연결·종료하지 않는다. lease recovery는 record를 requeue할 수 있지만 기존 local process와 조정하지 못한다.

### H-15. Coordinator timeout 후 뒤늦은 ghost commit 및 무제한 log read — 확정

`workbench/coordinator.py:1115-1126`, `workbench/server.py:1929-1935`

- `_submit()`은 30초 후 caller에게 실패를 반환하지만 queued callback을 취소하지 않아 나중에 mutation이 commit될 수 있다.
- log tail은 매 poll마다 `read_bytes()`로 전체 로그를 메모리에 읽은 뒤 끝부분만 사용한다.

장시간 hash/대형 로그에서 호출자 상태와 durable state가 어긋나거나 서버 메모리가 고갈될 수 있다.

### H-16. RDKit teardown crash를 `os._exit()`로 숨기는 경로 — 확정

`scripts/stage2_5_drug_avoidance.py:260-290`

결과 파일을 쓴 뒤 정상 Python teardown/atexit/destructor를 모두 건너뛴다. 알려진 native abort를 격리·해결하지 않고 process lifecycle에서 숨기므로 향후 flush/cleanup failure 또는 native corruption을 관찰할 수 없다. 전체 pytest에서 별도 native teardown abort가 실제로 한 번 발생했다는 점도 위험을 뒷받침한다.

### H-17. Installer bootstrap이 mutable `main`을 즉시 실행 — 확정

`README.md:216`, `install_skinscout.sh:178-181`

README는 GitHub `main`의 installer를 curl로 받아 즉시 실행하고 installer는 기본 branch를 plain clone한다. release tag/commit, checksum, signature/attestation을 검증하지 않는다.

## 8. MEDIUM 발견

| ID | 발견 | 근거 | 영향 |
|---|---|---|---|
| M-01 | malformed evaluation config를 조용히 default로 대체 | `eval/run_all.sh:33-59` | YAML/import/key/type 오류가 claim threshold 기본값으로 은폐됨 |
| M-02 | 문자열 `"false"`가 recipe scoring을 켬 | `workflow/rules/stage3b_fast.smk:30` | `bool("false") == True`; CLI override 의미 왜곡 |
| M-03 | recipe mode도 사용하지 않는 ChEMBL mirror를 input으로 강제 | `stage3b_fast.smk:49-61,107-118`, `stage3_daina_zoete.py:844-862` | clean production-index 환경에서 불필요하게 DAG 차단 |
| M-04 | recipe cross-field 오류가 DAG parse에서 차단되지 않음 | `workflow/Snakefile:866-887`, `stage3_daina_zoete.py:807-827` | 실행 후반에 실패; provenance snapshot도 실제 index와 다를 수 있음 |
| M-05 | transfer-reachable가 dev+test를 합쳐 모두 test로 표시 | `eval/build_transfer_reachable_panel.py:76-103,121` | dev 관측치가 test 결과처럼 보고될 수 있음 |
| M-06 | transfer/pocket panel이 upstream snapshot hash를 충분히 결속하지 않음 | `build_transfer_reachable_panel.py:72-100`, `build_pocket_cold_panel.py:161-223` | 서로 다른 benchmark/index 조합을 artifact만으로 식별 불가 |
| M-07 | common cutoff와 activity train boundary 불일치 | `workflow/config.yaml:232-247`, `eval/build_activity_benchmark.py:85-92,495-503` | 2023-10-02~12-31 evidence가 경로마다 train/post-cutoff로 달라짐 |
| M-08 | 검증되는 config 중 실행 명령에 반영되지 않는 값 존재 | `diffdock_for_no_pocket`, `msa_source`, `tICA_lag_steps`, `ligand_ff`, `timestep_fs` 관련 rules | 사용자는 config가 실행을 바꾼다고 오인 |
| M-09 | 여러 claim threshold가 0.01 수준 | `workflow/config.yaml` evaluation thresholds | chance/null baseline·CI·최소효과크기 없이 거의 무성능도 통과 가능 |
| M-10 | cold/cosmetic evaluator가 subset의 행 위치를 global Top-K처럼 사용 | `eval/cold_start_eval.py:133-152,187-196`, `cosmetic_retrospective_eval.py:195-214,273-282` | rank 100/101 subset도 positional Top-1로 계산 |
| M-11 | target universe·licensing provenance fail-open | `build_runtime_retrieval_index.py:94-111,193-199` | universe 파일 누락/manifest 부재에도 잘못된 index 생성 가능 |
| M-12 | Workbench remote transport·tenant·cache 경계가 약함 | `server.py:329-412,3418-3426,3633-3664,4164-4207` | shared-token 전 사용자 공간, HTTP/non-Secure cookie, authenticated artifact public cache |
| M-13 | 실제 run/batch 입력 제한과 SDF 권한 미흡 | `server.py:2812-2816,2922-3211` | preview만 4096 제한; 12 MiB 요청/argv DoS, 기본 0644 입력 노출 가능 |
| M-14 | JSON Schema와 Python enum/API export drift | `schemas/target_fast_v2.json:57-67`, `run_profile.py:105-115,662`, `contracts/__init__.py` | 같은 record가 schema PASS/Python FAIL; v3 helpers가 package API에 없음 |
| M-15 | readiness와 artifact fallback이 실제 신뢰를 과장 | `server.py:1332-1377,2223-2255,3914-3928` | path/flag 존재만으로 ready, legacy result route는 registry hash/verification 우회 |
| M-16 | 환경·라이선스 재현성 drift | `envs/**`, `install_runtime.py:389-450`, `LICENSE_POLICY.md` | solver 시점 의존; 감사 버전과 Dimorphite/Meeko/Boltz 실제 버전 불일치; vendored JSME 표 누락 |
| M-17 | monitor/HPA/band artifact provenance 취약 | `monitor_stage0.sh`, `stage0_download_hpa.sh`, `measure_panel_similarity_bands.py` | obsolete 경로로도 exit 0, mutable URL/existence-only reuse, panel/index hash manifest 없음 |
| M-18 | 테스트 skip/coverage/native teardown 신뢰 공백 | `scripts/tests/conftest.py`, 전체 pytest 결과 | call-phase data skip을 strict mode가 놓칠 수 있고 coverage 수치 없음; native abort 비결정성 |
| M-19 | 정적 mockup이 standalone에서 실제 제품처럼 보임 | `docs/design/*.dc.html`, `README.md:91-114` | HTML 자체에 지속 MOCKUP 표식 없음; mockup은 CHEMBL38, 운영 문서는 ChEMBL37 |
| M-20 | README 핵심 운영 주장 테스트가 local artifact 없으면 생략 | `scripts/tests/test_readme_delivery.py:82-89` | clean CI에서 recipe ID/4,873 target/source composition 주장이 고정 검증되지 않음 |

## 9. LOW 발견

| ID | 발견 | 근거 |
|---|---|---|
| L-01 | Pocket transfer Top-1% cutoff가 target universe와 무관하게 202로 고정 | `eval/measure_pocket_transfer.py:146-193` |
| L-02 | class metadata 누락 target을 실제 `orphan` class와 합침 | `eval/disagreement_eval.py:112-116` |
| L-03 | Snakemake parse가 read-only 명령에서도 directory 생성 가능 | `workflow/Snakefile:1399-1400` |
| L-04 | `dirname` command substitution 결과가 shell-safe하게 quote되지 않음 | `workflow/rules/stage3a_comprehensive.smk:50` |
| L-05 | unreachable fsync와 중복 hashchange listener 가능 | `workbench/coordinator.py:1084-1094`, `workbench/static/app.js` 초기화부 |
| L-06 | architecture/readiness naming이 v2에 머무름 | `docs/ARCHITECTURE.md:1-47`, `pipeline_readiness.py:1218-1252` |

## 10. 테스트 설계 공백

1. `stage2_pains_brenk.py` 직접 테스트가 없다.
2. Stage 4 핵심 orchestration은 구현 규모에 비해 직접 테스트가 매우 적다.
3. `stage3_meeko_ligand.py` 직접 테스트가 없다.
4. charged acid/base/zwitterion/permanent ion에 대한 index-query exact-self=1 회귀 테스트가 없다.
5. operational gate와 runtime recipe SHA tamper 테스트가 없다.
6. CREST `--chrg/--uhf`, multi-frame conformer cluster/selection E2E 테스트가 없다.
7. 악성 PDB/SDF/JSON/CosIng HTML injection 브라우저 테스트가 없다.
8. AlphaFold downloader 재사용 테스트는 실제 hash/content보다 file count 중심이다.
9. source 문자열 존재만 검사하는 테스트와 여러 동작을 한 함수에 묶은 mega-test 비중이 높다.
10. `sys.path`를 테스트 파일에서 직접 수정하는 패턴이 광범위해 수집 순서/import cache 영향을 받을 수 있다.
11. 실제 브라우저 E2E는 전체 UI 규모에 비해 적다.
12. `.github/workflows`, 프로젝트-level coverage/lint/type/security gate가 없다.

## 11. 확인된 강점

- 3,010개 테스트가 수집되고 대부분 통과하는 넓은 회귀 표면이 있다.
- 다수 artifact writer가 임시 파일 후 atomic replace 패턴을 사용한다.
- evaluation index role, manifest schema, input SHA, target universe 등의 fail-closed 검사가 상당수 구현되어 있다.
- host doctor가 placeholder container image를 숨기지 않고 명시적으로 차단한다.
- compose policy와 여러 외부 binary는 version/hash 검증 경로를 갖는다.
- 최종 Snakemake DAG는 현재 config에서 성공적으로 생성된다.
- UI band 문구는 실제 artifact 재계산 결과 `24/23`, `14/9`, `8/2`와 현재 일치한다.
- 새 정적 design 문서는 production API를 호출하지 않는 mockup이며, 브라우저 smoke에서 page/request error가 없었다.

이 강점은 회귀 방어와 운영 의도를 보여주지만, 위 blocker처럼 “평가된 과학적 정책과 실제 실행 input이 동일하다”는 상위 불변식을 대신하지는 못한다.

## 12. 우선순위 권고

### P0 — 실행·claim 차단 해소 전 필수

1. recipe/gate schema를 명시적으로 migration하고 current artifact를 재생성한다.
2. gate, runtime recipe, production index, target universe, code/env를 단일 signed/content-addressed contract로 결속한다.
3. query/index 표준화를 공유 함수 하나로 통일하고 index를 재구축한다.
4. `halt_min_votes`를 모델 수 범위로 제한한다.
5. Workbench verification payload를 정규화하고 missing verification을 fail-closed 처리한다.
6. 모든 생성 HTML의 dynamic content를 context-aware escape하고 CSP/sandbox를 적용한다.

### P1 — 과학적 타당성

1. prospective promotion metric을 sealed row-level predictions/truth에서 내부 재계산한다.
2. CREST charge/UHF와 conformer별 xTB→cluster→representative 계약을 구현한다.
3. activity evidence dedup을 publication/endpoint/relation/value provenance 단위로 바꾼다.
4. skin efficacy와 모든 Top-K evaluator에 explicit rank schema·정렬·연속성 검증을 적용한다.
5. fallback/degraded evidence를 정상 evidence와 구분하고 원래 실패 원인을 보존한다.

### P2 — 운영·재현성

1. Workbench process identity, shutdown/recovery, coordinator timeout 취소, bounded tail read를 보강한다.
2. lockfile/SBOM/license audit를 실제 설치 version과 자동 결속한다.
3. mutable installer/main bootstrap을 release digest/signature 기반으로 바꾼다.
4. CI에 lint, typecheck, security scan, coverage, strict artifact-skip gate를 추가한다.

## 13. 최종 승인 조건

다음이 모두 충족되기 전에는 release/claim 승인하지 않는 것을 권고한다.

- `check-operational`이 현재 checkout에서 성공한다.
- Stage 3가 gate가 승인한 exact recipe/index contract만 읽는 tamper test가 통과한다.
- charged/self-match 표준화 회귀가 1.0을 보장한다.
- 전체 pytest가 단일 uninterrupted run에서 failure 0이며 skip 3건의 환경 사유가 명시된다.
- XSS fixture와 verification-missing fixture가 브라우저/UI E2E에서 fail-closed한다.
- Stage 8 charged/open-shell 및 multi-conformer E2E가 실제 도구로 검증된다.
- clean clone에서 runtime recipe/index/gate artifact를 재생성할 수 있다.

## 14. 감사 무결성 선언

- 본 감사에서는 production code, config, schema, test, data artifact를 수정하지 않았다.
- 실행한 테스트는 임시 디렉터리와 pytest temp 영역을 사용했다.
- 저장소에 의도적으로 추가한 유일한 파일은 이 보고서다.
- 외부 동시 세션이 감사 중 추가한 커밋은 위 스냅샷 이력에 명시했고, 최종 delta를 별도로 검토했다.

