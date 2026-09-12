# SkinScout 기능·수정·실행 준비도 감사 — 2026-09-05

## 결론

SkinScout는 단일 저분자의 안전성 탐색, 사람 단백질 표적 우선순위화, 피부 맥락 주석과
화장품 대체소재 후보 검색을 연결하는 연구용 플랫폼이다. 실험적 결합·효능·안전성을
입증하거나 사용 농도·제형을 정하는 도구는 아니다.

준비·검증된 서버에서의 입력, 실행, 다운로드와 탐색적 해석은 README로 안내할 수 있다.
깨끗한 컴퓨터에서 wet-lab 연구자가 혼자 설치·전체 데이터 구축·고급 모델 준비·장시간
계산 복구까지 끝낼 수 있는 검증된 배포판은 아니다. 이번에는 코드와 문서의 구체적인
결함을 수정했으며, 새 코드에 맞는 생산 데이터 재생성과 전체 GPU 완주는 별도 상태다.

## 입력과 출력

| 경로 | 입력 | 출력과 해석 |
|---|---|---|
| 구조 입력 | 이름, SMILES, 구조 그리기, 단일 분자 SDF/MOL; 배치 이름/SMILES 최대 50개 | 표준화 구조, 적용 범위 경고, 2D 미리보기 |
| 안전성 탐색 | 구조와 준비된 ADMET/감작성 환경 | ADMET, 감작성 합의, 구조 경고, CosIng·의약품 대조, 판정과 사유 |
| Fast 표적 탐색 | 같은 화합물, 활성 검색 인덱스·수용체·도킹 도구 | 원본 검색 순위, 11–50위 밴드 재정렬, 유사 활성분자 근거, 구조·피부·KG 주석 |
| 로컬 핵심구조 대체소재 | 기준 화합물 또는 `name,smiles` CSV | CosIng 단일 저분자 후보, 입체화학을 고려한 핵심구조 등급, 선택적 파마코포어·활성 근거 |
| Pharmacophore 대체소재 | 검증된 부모 표적 분석, 선택적 표적과 3D anchor | 같은 표적의 활성 근거를 이용한 후보 CSV/SDF/HTML/JSON |
| Comprehensive/전체 보고서 | 고급 외부 모델·환경과 같은 입력 | PSICHIC/RTMScore/Boltz-2, 상호작용, BioEmu, MD/MMGBSA, QM 등의 확장 산출물 |

분석 실행은 `results/runs/<run_id>/`에 저장된다. 먼저 `run_summary.md`와
`run_verification.json`을 읽고, `viewer/index.html`, `viewer/summary.csv`, 표적 CSV와
원본 근거를 확인한다. 실패·HALT·구조 미요청 실행에는 일부 산출물이 없을 수 있다.
`PASS`는 프로그램의 계약을 통과했다는 뜻이다. `REVIEW`는 위험 확률이 아니며,
`HYPOTHESIS` 후보의 실제 활성 보존 여부는 별도 검증이 필요하다.

안전성, 표적, report 및 새 부모 분석을 만드는 substitute 경로는 모두 외부 감작성
서비스에 구조를 보낼 수 있다. 표적 경로를 고르는 것만으로 비공개 성분이 로컬에
남는 것은 아니다. 로컬 원료표 검색과 오프라인 결과 열기는 이 감작성 호출과 구분한다.

## 검토 방법과 범위의 한계

- 기존 작업 트리 수정과 외부에서 추가된 커밋을 보존했다. 감사 에이전트는 커밋하지 않았다.
- Stage 0–3, Stage 4–8, 평가/모델, Workbench/보고서, 설치/실행 무결성을 나누어 검토했다.
- 독립 code-reviewer와 architect가 수정 후 호출 경계를 다시 검토했고, 새로 발견한
  계약 불일치도 수정 대상으로 되돌렸다.
- 독립 architect의 마지막 판정은 배포 준비도 `BLOCK`, 코드 통합 `WATCH`였다.
  이는 저장소 전체 무결점 승인과 다르다. code-reviewer가 종료 시 미확인으로 남긴
  상대 SDF 결함은 후속 독립 verifier의 현행 코드·회귀 검사로 종료했다.
- production 소스의 직접 독해, 회귀 재현, 전체 테스트, Python/Shell/JavaScript 문법
  검사를 구분했다. 전체 테스트 통과를 전체 행 수동 독해의 증거로 취급하지 않았다.
- Workbench/report lane은 22,392행을 직접 검토했다. Stage 4–8 lane의 검토 목록은
  production/workflow/environment 31파일, 8,546행이며, 이후 Stage 11·대체소재의
  9파일, 7,839행을 추가 검토했다. 평가 lane은 초기 15파일, 8,501행과 추가
  7파일, 3,620행으로 총 22파일, 12,121행을 전체 독해했다. 나머지 평가 소스는
  부분 독해·패턴 검사·컴파일·테스트 대상이며 전체 행 수동 검토로 계산하지 않았다.
- 검색 lane은 10개 production 파일을 전체 독해하고 관련 workflow rule 블록을
  검토했다. 대형 검색·gate·서열 파일 7개는 함수·호출 경로 중심 부분 독해다.
  84파일/37,223행의 기계적 검사를 전체 행 수동 독해로 계산하지 않았다.
- vendor/minified 제3자 내부 구현, 모델 가중치, 전체 외부 데이터의 과학적 진실성은
  이번 수정으로 증명하지 않았다. 따라서 “전체 저장소의 모든 행을 동일 깊이로 수동
  검토했고 모든 버그가 0개”라고 주장하지 않는다.

## 수정한 주요 결함

| 영역 | 결함과 수정 | 대표 파일 |
|---|---|---|
| 실행 설정 | 사용자 추가 설정이 run/input/mode를 덮어쓰는 문제 차단; 기본 workflow 설정도 fingerprint에 포함 | `scripts/run_skinscout.py` |
| 환경 | named env 존재만으로 재사용하지 않고 환경 정의 digest와 import 확인; 실제 Snakemake env 사전 구축 | `scripts/install_runtime.py` |
| 무-Conda 실행 | 다른 비활성 환경에서 readiness가 통과하는 문제 차단; 실제 실행 Python으로 검사 | `scripts/model_readiness.py`, `scripts/run_skinscout.py` |
| Stage 0 모니터 | 과거 절대경로 제거; 다른 프로세스를 현재 단계로 오인하지 않음; 두 준비도 검증을 통과할 때만 완료로 표시 | `scripts/monitor_stage0.sh` |
| 컨테이너 | 사용자 data/results 경로가 실제 bind mount에 전달되도록 수정 | `scripts/host_cli.py`, `compose/skinscout.compose.yaml` |
| 실행 증명 | run identity, verifier 종료코드·상태, 산출물 경로·바이트·SHA 및 소비되는 표적·밴드 결과 변경 검증 강화; 상위 symlink 경로 우회 차단 | `scripts/verify_goal_contract.py`, `scripts/run_skinscout.py` |
| 부모 분석 재사용 | 완료 여부만 보지 않고 전체 검증 기록·분자 identity·입력·산출물을 확인; 상대 SDF 경로를 한 번 절대화하여 부모·anchor·provenance가 같은 파일 사용 | `scripts/run_substitute_discovery.py` |
| 서열 | pLDDT로 잘린 구조 유래 서열 대신 원본 AFDB v4 조각을 병합한 전체 길이 서열 사용; CRC64·중복·gap·coverage 검증 | `scripts/build_canonical_sequences.py`, `scripts/stage0_verify.py` |
| 검색 근거 | 중성화 규칙 일치, 측정 provenance 기반 중복 처리, 런타임 인덱스·레시피 gate binding | `scripts/build_runtime_retrieval_index.py`, `scripts/merge_runtime_activity_evidence.py` |
| 실제 전달 순위 | 계산만 하고 버려지던 밴드 순위를 skin→KG→summary/report/viewer에 전달; 원본 Daina 순위 보존; final score를 명시적 서수로 처리 | `workflow/rules/stage3*.smk`, `scripts/stage3_skin_weighting.py` |
| KG | 큐레이션 seed를 논문 수로 세는 문제 수정; Fast의 미확인 효능을 양성 근거로 만들거나 표적에서 제거하지 않음 | `scripts/stage0_skin_kg.py`, `scripts/stage3_kg_efficacy_label.py` |
| 구조·상호작용 | 서열 offset을 고려한 chain 정렬, 선택 chain 근처 ligand instance 사용, `kept=yes` Boltz 복합체만 후속 분석 | `scripts/stage4_prepare_structures.py`, `scripts/stage5_5_plip.py`, `scripts/stage5_5_prolif.py` |
| BioEmu/MD | rigid-body 이동에 의존하던 clustering 수정, 구현되지 않은 tICA 옵션 거부, chain별 잔기 identity, strained medoid 거부, MD pose 연결, stale MMGBSA 격리 | `scripts/stage6_bioemu.py`, `workflow/Snakefile`, `workflow/rules/stage7_md.smk`, `scripts/stage7_mmgbsa.py` |
| QM | CREST charge/spin 전달, 다중 XYZ 엄격 파싱, 각 conformer의 xTB 계산과 최저 에너지 선택, 공유 free-ligand DFT 경로 | `scripts/stage8_crest.py`, `scripts/stage8_xtb_cluster.py`, `scripts/stage8_dft.py` |
| 대체소재 | 반대/미지 입체화학을 완전 보존으로 판정하는 문제, 행 순서에 따른 MCS 예산·동점 순위 의존성 수정 | `scripts/alternative_ingredients.py` |
| 대체소재 근거 | 부모/후보 안전성 차이를 같은 penalty로 계산; 상대 anchor 경로는 실제 bytes/SHA에 일치하는 파일로 확인 | `scripts/discover_substitutes.py`, `scripts/interaction_anchor.py` |
| 통계 | 방법 간 공통 후보 모집단, 동점 Wilcoxon 분산, 더 강한 기준선, Holm 보정; 교호작용 미검정 사실 명시 | `scripts/eval_alternative_criteria.py` |
| 학습·평가 | 집계 CSV만으로 prospective 승격 차단, 학습/추론 tokenizer·feature binding, 안전한 checkpoint load, 독립 prediction ensemble, BCE 정규화, 날짜·ontology 보존 | `eval/performance_v2_model.py`, `eval/prospective_promotion_eval.py`, `scripts/prepare_performance_v2_inputs.py` |
| 평가 순위 | `final_rank`를 원본 rank보다 우선하며 boolean 순위 거부; 파일 행 순서가 아닌 실제 순위로 Top-N 선정; 명시한 FASTA가 없거나 비면 실패 | `eval/collect_run_outputs.py`, `eval/cold_start_eval.py`, `eval/cosmetic_retrospective_eval.py` |
| Workbench | timeout 뒤 ghost commit, 잔존 child process, PID 재사용, batch 실패 전파, 상태 필터와 읽기 비용 수정 | `workbench/coordinator.py`, `workbench/server.py` |
| 출력 보안 | HTML/script 삽입 escape, token 로그 노출 제거, private log/PID 파일, 공개 snapshot 경로·symlink·백업 보호 | `scripts/stage9_report.py`, `scripts/start_workbench.py`, `scripts/build_public_snapshot.sh` |
| 출판 산출물 | 단순 artifact 개수 그림의 `source_backed` 오표기를 `diagnostic_only`로 수정; 그림·caption·원고 및 재현 provenance를 claim에 결속 | `scripts/stage11_make_figures.py`, `scripts/stage11_claim_manifest.py`, `workflow/rules/stage11_publication.smk` |
| 전송·문서 | 명시적 수신자 없이는 report 이메일 미전송; 외부 구조 전송·설치 한계·낡은 성능표 정정 | `scripts/report_email.sh`, `README.md`, `docs/RESEARCHER_GUIDE.md` |

## 검증 기록

아래 수는 중복되는 테스트 묶음이므로 서로 더하지 않는다.

| 검사 | 이번 실행에서 확인한 결과 |
|---|---|
| 마지막 전체 `scripts/tests` | 3,453 passed / 4 failed / 3 skipped, 1,072.26초; 총 3,460 cases |
| 마지막 실패 후속 검증 | REINVENT의 변경된 오류 문구 기대값을 갱신하고 생성 산출물 부재 assertion을 추가; 관련 72 passed. 남은 실패 3건은 기존 운영 gate/recipe 결속 |
| 첫 전체 `scripts/tests` | 3,395 passed / 10 failed / 6 skipped, 981.01초; 수정 중 수집된 이전 fixture와 새 코드가 섞인 검증도 포함 |
| 초기 launcher/profile/snapshot | 179 passed |
| 대체소재 및 평가 회귀 | 100 passed; 공통 모집단/입력 검사 추가 후 관련 workflow 포함 106 passed |
| Stage 4–8 | scoped 158 passed; fail-closed 선택 149 passed |
| Stage 11·대체소재 후속 검토 | 197 passed; 진단용 그림은 생성 가능하나 publication claim은 차단 |
| Workbench/report | scoped 386 passed; 별도 verifier/package/viewer 등 204 passed |
| 마지막 런타임·gate 회귀 | 202 passed; AlphaFold fixture 2건 수정 후 gate/다운로드 등 28 passed. 별도 기존 데이터 통합 테스트 2건은 실패 유지 |
| 상대 SDF 독립 종료 검증 | 서로 다른 디렉터리의 동명 파일 회귀 1 passed; 독립 verifier가 해당 HIGH finding을 CLOSE로 판정 |
| Stage 0 모니터 | strict/claim-quality/activity 및 operational gate, 호출 cwd를 검사하는 4개 회귀 통과 |
| 실제 Torch CPU 신경망 경로 | 73 passed; base env에서 생략되는 경로를 별도 환경으로 검사 |
| 파일시스템 경계 | 전체 실행에서 같은 tmp 파일시스템 때문에 생략된 cross-device 검사를 `/var/tmp`와 `/tmp`로 분리해 실행: 1 passed |
| canonical FASTA + claim-quality | 65 passed; 실제 전체 서열 checksum/coverage 검사 통과 |
| Python 문법 | `scripts eval skinscout workbench psichic rtmscore` compileall 통과; 후속 372개 Python parse/compile 통과 |
| Shell/JS 문법 | scripts/eval Shell 17파일 `bash -n`, Workbench 및 fast-report JavaScript `node --check` 통과 |
| 린터/정적 분석 도구 | Ruff/Mypy/Bandit/ShellCheck 미설치. 문법 검사를 이 도구들의 실행으로 표현하지 않음 |

전체 실행을 모두 녹색으로 보고하지 않는다. 마지막 전체 실행의 실패 4건 중 1건은
변조된 anchor를 정상 거부하면서 바뀐 오류 문구를 테스트가 따라가지 못한 경우였고,
후속 72개 회귀에서 수정 확인했다. 나머지는 다음 세 운영 데이터 통합 검사다.

- `test_operational_gate_on_disk::test_the_gate_the_run_path_checks_actually_passes`
- `test_stage3_recipe_wiring::test_a_real_run_produces_what_the_next_rule_consumes`
- `test_stage3_recipe_wiring::test_the_run_says_whether_its_nearest_evidence_was_active`

전체 JUnit 기록은 `/tmp/skinscout-full-audit-final-20260905.xml`, anchor 후속 기록은
`/tmp/skinscout-anchor-final-20260905.xml`이다. 최초 실패 4건의 최종 재검사는
`/tmp/skinscout-final-failure-audit-20260905.xml`에 기록했다.

전체 서열 실측: 23,391 압축 mmCIF 조각 → 20,504 canonical 서열, 현재 cleaned receptor
20,171종 모두 포함. 각 서열 CRC64 독립 재계산 통과. FASTA SHA256:
`a20547aa81da01a42ddc8c905a5113bd78f05fb2346897f3fa9e8c7e056ca54d`.
검증 산출물은 `/tmp/skinscout-canonical-final.liDNYt/`이며 생산 데이터 경로를 덮어쓰지 않았다.

로컬 retinol 후보 CLI가 종료코드 0으로 실행되어 507종에서 핵심구조 유지 후보 2건을
출력했다. 수정된 curated 평가도 24질의/57쌍을 실행했다. 공통 후보·질의별 더 강한
기준선·Holm 보정 후 네 방법 모두 우월하다는 이전 결론은 지지되지 않았다.
이 제한적인 큐레이션 결과를 임상·활성 보존 검증으로 읽지 않는다.

## 아직 남은 운영·과학적 검증 조건

1. 생산 경로의 canonical FASTA, MMseqs/sequence embeddings, 검색 인덱스, 모델,
   panel 및 gate는 새 계약에 맞게 재생성해야 한다. 기존 산출물에 새 manifest만
   붙여 현재 평가로 가장해서는 안 된다.
2. 실제 on-disk operational gate는 초기 검사에서 runtime binding 누락으로 실패했고,
   최종 코드에서는 `activity retrieval operational gate evaluation decision is stale`
   로 거부된다. 이를 감추기 위해 해당 테스트나 gate를 완화하지 않았다.
   또한 현재 runtime 설정의 `dev_selection_skin/recipe.json`
   (`union_discount_light`)과 frozen evaluation의 `dev_selection/recipe.json`
   (`union_any_consensus`)이 다르다. 이 상태에서는 gate 파일만 다시 만드는 것으로
   해결되지 않는다. 의도한 runtime recipe와 같은 recipe를 사용한 평가 근거를
   재생성·검증하거나, 검증된 recipe를 실제 실행 설정으로 선택하는 명시적 결정이 필요하다.
3. 검사 시 GPU는 다른 계산이 사용 중이었다. 그 작업을 중단하지 않았으며, 전체
   target/report GPU 실행과 clean-Ubuntu 설치를 완료했다고 주장하지 않는다.
4. 현재 공유 base 환경의 `pip check`는 Conda와 ruamel.yaml 버전 충돌을 발견했다.
   새 설치 정의에 기존 transitive dependency의 호환 범위를 고정했지만, 사용 중인
   공유 환경의 패키지를 임의로 교체하지 않았다.
5. 내장 Workbench는 HTTPS/개별 계정/프로젝트 권한 격리를 제공하지 않는다.
   기본 loopback과 SSH 터널을 사용한다. 대규모 다중 사용자 운영은 별도 설계·검증 대상이다.
6. 새 sealed row-level prospective 근거가 없는 집계 보고서는 진단 전용이다.
   sequence recipe 후보 선택의 다중성, 작은 표적 수, 학습 데이터 오염과 시간 cutoff
   정의는 성능 일반화 주장을 위한 추가 연구 조건이다.
   구조 평가의 `2023-10-01`과 supervised activity 학습의 `2023-12-31`은 서로 다른
   경계를 뜻하므로 임의로 같은 날짜로 바꾸지 않았다. 두 축을 결합한 시간 일반화
   주장은 별도의 cutoff 의미·manifest 결속 검토가 필요하다.
7. Stage 11의 현재 그림은 artifact 개수 진단이며 실제 논문 그림의 물리·생물학적 양을
   그리지 않는다. 이 경로는 이제 출판 가능으로 승인되지 않는다. 그림별 실제 데이터
   schema와 수량을 사용하는 renderer가 있어야 publication-ready 패키지를 만들 수 있다.
8. bootstrap의 기본 branch clone, 버전 없는 HPA 다운로드 URL, 환경 전체의 불완전한
   lock/SBOM은 배포 재현성의 남은 조건이다. 이번 로컬 수정은 서명된 release 배포나
   외부 데이터 release 고정을 대신하지 않는다. 사용자 지정 환경·제3자 내부 구현까지
   안전성과 재현성을 보증하는 공급망 감사도 완료했다고 주장하지 않는다.

설치·재생성 안내는 [Stage 0 runbook](STAGE0_RUNBOOK.md), 연구자 해석은
[연구자 가이드](RESEARCHER_GUIDE.md), 시작 안내는 [README](../README.md)를 따른다.
