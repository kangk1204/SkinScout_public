# 독립 감사(2026-08-26) 대응 기록

`docs/SKINSCOUT_COMPREHENSIVE_READONLY_AUDIT_20260826.md`의 발견을 **코드에서 직접
재확인한 뒤** 처리한 내역이다. 감사 보고서를 그대로 받아들이지 않고 항목마다 근거를
다시 읽었다.

## 재확인 결과: 검증한 항목은 전부 사실이었다

| ID | 재확인 방법 | 결과 |
|---|---|---|
| F-01 | 패널 15개 InChIKey 연결성 대조 | 사실. 4개 오류(감사 보고 전 독립 발견) |
| F-02 | `stage3a_comprehensive.smk:142-186` 직접 읽기 | 사실. GNINA·RTMScore 입력이 `rules.xtb_optimize.output.sdf` |
| F-03 | `stage0_skin_kg.py:57-73, 294` | 사실. seed 상수가 실제 PMID 수보다 크면 그것이 이김 |
| F-04 | `stage11_make_figures.py:_source_metric` 반환값 | 사실. `("rows", len(records))`, 축 라벨 `artifact evidence count` |
| F-08 | `summarize_run_outputs.py` compound 재조립부 | 사실. `applicability`가 요약에서 탈락해 보고서 절이 렌더링된 적 없음 |
| F-09 | `pains_action` 전체 grep | 사실. `stage2_pains_brenk.py`가 인자를 받지 않음 |
| F-12 | `skin_proteome.tsv`, `skin_tpm.tsv` 줄 수 | 사실. 둘 다 헤더 1줄 |
| F-20 | `index.html:213` | 사실. "Daina + 도킹 근거로 표적 순위화" |
| F-21 | 테스트 실패 로그 역추적 | 사실. **flake가 아니라 실제 결함**이었다 |

F-21은 특기할 만하다. `test_torch_cpu_minibatched_train_and_score_is_deterministic`가
간헐적으로 실패해 환경적 flake로 넘겼는데, 감사 지적을 받고 로그를 되짚으니
`_score_torch_vector`가 CUDA를 하드코딩해 GPU가 바쁠 때 죽는 것이었다. **감사가 없었으면
계속 flake로 처리했을 것이다.**

## 수정한 것

### 적용 범위 게이트 (F-08, F-20)
- CLI(`run_skinscout.py`)와 Workbench가 **SMILES·SDF 양쪽 모두** 게이트를 통과하도록 연결.
  이전에는 Workbench SMILES 입력 한 곳뿐이라 CLI에 펩타이드를 넣으면 그대로 실행됐다
- 선언만 있고 구현이 없던 **UV 필터 판정**을 큐레이션 목록 10종으로 구현
  (`data/validation/uv_filter_reference.csv`). oxybenzone·avobenzone·octinoxate가
  전부 `in_scope`로 통과하던 상태였다. 목록은 완전하지 않음을 명시
- **해석 불가 레코드가 섞인 SDF** 거부. RDKit은 읽지 못한 레코드에 `None`을 내지만
  레코드로 인식조차 못 한 뒤쪽 내용은 조용히 버리므로, `$$$$` 구분자를 세어 대조
- 거부된 업로드를 예외 경로에서도 삭제
- `applicability`를 요약 객체까지 전달 (이것이 없어 보고서 절이 렌더링된 적 없음)
- Demo 카드 문구를 실제 동작(주석 계층)에 맞게 정정

### 채점 device (F-21)
- 채점이 GPU가 보이기만 하면 CUDA를 강제하고 CPU를 요청할 방법이 없었다.
  명시적 `--device`, CUDA 배치 실패 시 CPU 폴백(경고 동반), 명시적 CUDA 요청은
  조용히 강등하지 않음. fixture 채점기에서는 무의미한 인자를 제거

### 정직한 라벨링 (F-03, F-04, F-09, F-12)
- KG `n_papers`를 `n_papers_counted`(검증된 PMID 수) / `n_papers_seed`(큐레이션 상수) /
  `n_papers_basis`로 분해. 기존 `n_papers`는 호환을 위해 유지
- Stage 11 그림 캡션이 실제 그리는 것(소스별 artifact 개수)을 말하도록 정정하고,
  의도한 과학량은 `intended_caption`으로 분리 보존
- `pains_action`에서 구현되지 않은 `drop`을 제거. 없는 안전 통제로 읽히던 설정이다
- SkinScore가 실제로 사용한 축과 재정규화된 가중치를 `*.axes.json`에 기록.
  선언된 5축 공식과 실제(HPA 2축)가 달랐다

## 수정하지 않고 기록만 한 것

### F-02 pose 계보 — 이후 GNINA는 재배선 완료

`gnina_rescore_top`을 pose 모드로 재배선하고 실제 pose로 검증했다
(`docs/F02_GNINA_POSE_REWIRE_20260826.md`). 실측 결과 이전 배선은 무해한 근사가 아니었다 —
같은 50개 표적에서 **0/50이 동일**했고 두 모드의 Spearman은 **−0.16**이었다.

RTMScore도 이후 같은 pose 모드로 확장했다. 확장 과정에서 **RTMScore가 이 머신에서 한 번도
점수를 낸 적이 없었다**는 것이 드러났다 — F-21과 같은 유형으로, CPU 전용 DGL에 CUDA를
강제하고 있었고 모든 예외가 `LOG.debug`로 삼켜지고 있었다.

재배선 전후로 3-way 합의를 만들어 비교하면 **상위 10개 중 2개만 겹친다**. 이전 배선은
무해한 근사가 아니었다.

Boltz-2는 pose를 보지 않는 독립 예측기라 대상이 아니다. 전체 comprehensive DAG 실행은
이 환경에 `snakemake`·`xtb`·DiffDock이 없어 여전히 불가능하다.

### 그 밖의 High 항목
F-05(원고 세대 불일치), F-06(과학 게이트 FAIL), F-07(known-target 직접 근거 부족),
F-10(MD 설정 불일치), F-11(DiffDock 준비성), F-13(배포 재현성), F-14(Stage 9 표현)은
재확인은 했으나 이번 범위에서 다루지 않았다. 각각 별도 결정이 필요하다.

## 감사가 지적한 F-01 잔여 사항

- `scripts/tests/test_panel_structures.py`는 InChIKey **연결성 블록**만 고정한다.
  입체화학 전체는 고정하지 않는다. 검색이 쓰는 Morgan 지문이 기본적으로 입체를 보지
  않으므로 의도한 선택이지만, 한계로 명시해 둔다
- v1 기준 하류 산출물(`results/audits/known_target_pair_evidence_chembl37.csv`,
  `data/activity_recovery_panels_202608/`)은 아직 v1이다. v2 재생성은 미완
