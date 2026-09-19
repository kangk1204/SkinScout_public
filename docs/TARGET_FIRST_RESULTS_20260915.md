# SkinScout 실험팀 인계 결과 — 2026-09-15

승인된 표적 목적 기반 분석 기능과 1차 인계본을 구현·실행·검증했다.
**엄격한 실험 우선 조건을 모두 충족한 후보는 0개다.** 문헌 후보와 대조물질은
추가 확인할 근거·실험안과 함께 제공하며, 실제 효능·안전성을 확정하지 않는다.

## 최종 파일

- [전달용 ZIP, 약 29.2 MB](../results/target_first_20260915/SkinScout_experiment_handoff_20260915.zip)
- [실험팀 엑셀](../results/target_first_20260915/handoff/SkinScout_experiment_handoff.xlsx)
- [읽기용 요약](../results/target_first_20260915/handoff/START_HERE.html)
- [전체 표적–화합물 관계 CSV](../results/target_first_20260915/handoff/target_candidate_matrix.csv)
- [33개 목적별 실험안 CSV](../results/target_first_20260915/handoff/assay_plan.csv)
- [입력·코드·산출물 해시와 개수](../results/target_first_20260915/handoff/decision_manifest.json)

ZIP SHA-256: `08003405a714c37888c8c8386e7b7b2a213ebbb9688273fc896da7dad3b8c6d4`.
`audit/initial_handoff_build*`는 파일명 호환성 수정 전 보존본이며 전달 대상은 위 ZIP이다.

## 분석 범위와 결과

| 구분 | 확인한 결과 |
|---|---|
| 원본 입력 | 31개 행 → 33개 목적, 29개 단백질 ID와 4개 비단백질 지표 |
| 저장된 원천 자료 조회 | 25개 단백질 ID에서 근거 확보, 전체 33개 목적의 상태 보존 |
| 원천 근거 | 66,636행; ChEMBL 활성 29,715행, mechanism 141행, BindingDB 36,780행 |
| 전체 관계 | 29,557개 표적–화합물 관계, 27,855개 화합물 ID; 이 중 56개 구조 미확정 |
| 과거 입력 보존 | 176개 행, 173개 정확 구조, 174개 고유 목적–화합물 관계, 176개 계보 연결 |
| 최신 코드 안전성 재평가 (2026-09-15 정책) | 안전성 record 198쌍/182종; 웹 모델 실행 시도 126종; 합의 유효(부분 2-positive HALT 포함) 65쌍/53종; 완전 분석 44쌍/37종; PASS 1, FLAG_HIGH 92, HALT 33, UNAVAILABLE 56 |
| 안전성 record 없음 | 27,673개 고유 화합물 ID에 record 없음; 세 웹 모델이 실행되지 않은 고유 화합물은 27,729종; 전체 원천 구조에 모델을 실행한 것은 아님 |
| 문헌 작업 목록 | 10종, 11개 목적–화합물 관계, 12개 큐레이션 근거 행 |
| 엄격한 실험 우선 후보 | 0개; 조건을 충족하지 못한 후보를 수량 확보용으로 승격하지 않음 |
| 실제 지정 표적 도킹 | 인간 KLK5의 실측 활성/약한 결합 대조물질 2종 실행 완료 |

문헌 작업 목록의 Dexpanthenol, Ectoine, Asiaticoside, 4-n-Butylresorcinol,
Glabridin은 추가 근거가 필요한 상태다. Niacinamide는 현재 모델 합의 정책에서 HALT이며,
이를 실험으로 확인된 위해성으로 해석하지 않는다. DHT, Ilomastat, Sulforaphane,
Thiamidol은 이 인계본의 대조물질이다. 유일한 계산상 PASS인 DHT도 AR 억제 목적과
작용 방향이 반대이므로 목적 후보에서 제외했다. 각 판단의 원저·시험계·구조는
[작업 목록](../results/target_first_20260915/handoff/working_candidates.csv)과
[큐레이션 원장](../results/target_first_20260915/handoff/curated_evidence.csv)에 연결되어 있다.

안전성 모델의 실시간 조회와 저장 응답의 현재 코드 재해석을 구분했다. STopTox 장애로
새 응답을 확보하지 못한 경우를 결측으로 남겼다. 정확 구조에 연결된 과거 구조화 응답
61건은 재해석했고, 최초 조회 시각이 없는 기록에 파일 수정 시각을 대신 넣지 않았다.
원천 InChIKey 불일치 16,090개 근거 행도 숨기거나 같은 입체구조로 자동 병합하지 않았다.

## 구현과 수정

| 파일 | 역할 |
|---|---|
| [target_intent.py](../scripts/target_intent.py), [schema](../schemas/target_intent_v1.json), [목적 레지스트리](../data/curation/target_intents_20260915.csv) | 표적·성숙 단백질·원하는 작용 방향·비단백질 지표 계약과 후보 판정 |
| [build_target_candidate_evidence.py](../scripts/build_target_candidate_evidence.py) | 상위 10개 절단 없이 전체 원천 관계 조회; 종·시험·부등호·입체정보 보존 |
| [reanalyze_target_candidate_safety.py](../scripts/reanalyze_target_candidate_safety.py) | 현재 코드 재평가, 모델별 실패·과거 응답·실제 시각 보존, 중복 구조 결과의 명시적 조합 |
| [screen_target_candidates.py](../scripts/screen_target_candidates.py), [독립 실행 규칙](../workflow/rules/target_candidate_screen.smk) | 지정 표적과 화합물 쌍의 구조 분석, 필수 잔기·포켓·안전성·캐시 해시 검증 |
| [build_target_lab_package.py](../scripts/build_target_lab_package.py) | 원장에 연결된 CSV·XLSX·HTML·SDF·ZIP, 독립 근거 검토 확인, 파일명 호환성과 해시 검증 |
| [문헌 큐레이션](../data/curation/target_candidate_literature_20260915.json) | 10종의 정확 구조와 검증 출처·제한 연결 |

결합 수치가 기능 방향으로 바뀌는 오류, 사람 표적이 사람 세포 시험으로 바뀌는 오류,
요약의 방향 표지만으로 후보를 승격하는 오류를 차단했다. 기존 적용 범위·안전성 모듈을
재사용하며 별도 종합 효능 점수나 새 학습 모델을 추가하지 않았다.
재현 명령과 판정 의미는 [사용 안내](TARGET_FIRST_HANDOFF.md)에 있다.

## 검증 증거와 남은 한계

- 관련 고유 테스트 **309개 통과**, 실패·건너뜀 0개. 기존 회귀 포함 307개와 최종
  인계 기능 23개 재검증의 합집합이며 중복 21개를 제외한 수다.
- 변경된 10개 Python 소스/테스트의 Ruff와 구문 분석, `git diff --check` 통과.
- 독립 Codex 검토자가 큐레이션 12행을 출처와 대조해 제한 조건을 명시했다.
  이는 실제 피부 연구자 또는 해당 논문 저자의 검증을 뜻하지 않는다.
- KLK5 6QFE/J08 재도킹 RMSD **1.0964 Å**로 사전 기준 2 Å 통과.
  대조물질의 CNN affinity·CNN score 순서는 실측 결합력과 반대였다. 도킹 순위 성능
  개선은 입증되지 않았고, 도킹 점수로 소재 후보를 승격하지 않았다.
- ZIP 460개 파일의 CRC·파일별 해시·폴더 일치, Windows 금지 문자·대소문자 경로 충돌,
  엑셀 8개 시트, HTML 내부 링크, SDF 10개 구조를 확인했다.
  `portable_path_map.json`의 432개 대응 관계로 원래 경로와 변경된 파일명을 추적한다.

[테스트·정적 검사](../results/target_first_20260915/audit/static_checks.json)와
[최종 인계본 검증](../results/target_first_20260915/audit/final_package_validation.json)에
기계 판독 가능한 결과가 있다.
별도 검토자의 [최종 수신자 감사](../results/target_first_20260915/audit/independent_recipient_audit_final.md)도
**CLEAR**다. 1,877개 확인 항목에서 원장·CSV·Parquet·엑셀의 수치와 판정, 입력·출력 해시,
ZIP의 파일별 내용 일치를 확인했으며 추가 수정 사항은 없었다.

TYR 구리·MMP 아연의 구조 처리는 검증된 지원이 없어 보류했다. HA·8-OHdG·Ceramide·TEWL에는
단백질 도킹을 만들지 않았다. 문헌 검토는 작업 목록에 한정되며 전 세계 문헌의 망라적
검토가 아니다. 실제 실험, 구매·합성, 제형별 노출 검증, Workbench 확대, MD, 실험 데이터
수집 기능과 재학습은 이번 1차 인계 범위 밖이다. 실험팀에는 현재 후보 상태와 대조군,
농도반응·세포독성·측정 간섭·별도 원리 확인 시험을 함께 검토할 자료를 제공한다.
