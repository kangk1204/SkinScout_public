# pocket-cold 패널 구축 — Route B를 순환논증 없이 측정하기 위한 전제

- 작성일: 2026-08-30 (Asia/Seoul)
- 산출물: `data/pocket_cold_panel_202608/`
- 재현: `python eval/build_pocket_cold_panel.py`

## 왜 필요했나

커버리지 경로 B(포켓 클러스터 전달)는 **Foldseek 포켓 클러스터를 통해 리간드 근거를
빌려온다**. 그러면 평가 패널의 표적이 학습에 등장한 클러스터에 속할 경우, 답을 미리
넘겨준 표적으로 채점하는 셈이 된다 — 순환이다.

기존 평가 뷰는 **서열 클러스터와 스캐폴드**를 홀드아웃하지만 **포켓 클러스터는 하지
않는다.** 직접 측정한 결과:

| 분할 | 행 | pocket-cold(tm40) 비율 |
|---|---:|---:|
| dev | 77,683 | **2.16%** |
| test | 67,629 | 1.56% |
| dual_cold | 93 | 94.6% (그러나 총 93행·13표적) |

즉 Route B를 측정할 자리가 없었다. `COVERAGE_ROUTES_20260824.md`가 남긴
"적정 규모의 pocket-cold 패널을 먼저 만들어야 한다"가 이 문서의 대상이다.

## 무엇을 만들었나

벤치마크 자체의 규약("test 행 중 pair·publication·<대상>이 train/dev에 없는 것")을
그대로 따라 두 뷰를 만들었다.

| 뷰 | 평가 | 참조 | claimable |
|---|---|---|---|
| **strict** | test | train + dev | 예 — 기존 `target_cold30/50`·`dual_cold`와 직접 비교 가능 |
| **extended** | dev + test | train | **아니오** — dev로 평가하므로 헤드라인 수치로 인용 불가 |

각 뷰를 tm40 / tm50 / tm60 세 수준으로 냈다.

| 패널 | 행 | 표적 | 화합물 | 랭킹 질의 |
|---|---:|---:|---:|---:|
| tm40 strict | 400 | **25** | 396 | 396 |
| tm50 strict | 402 | 27 | 398 | 398 |
| tm60 strict | 440 | 28 | 434 | 434 |
| tm40 extended | 2,733 | **87** | 2,275 | 2,302 |
| tm50 extended | 2,735 | 89 | 2,277 | 2,304 |
| tm60 extended | 2,797 | 92 | 2,337 | 2,364 |

비교: `dual_cold`는 93행 / **13표적**이다. strict가 1.9배, extended가 6.7배다.

## 검증한 것

| 성질 | 결과 |
|---|---|
| 패널 표적의 포켓 클러스터가 참조에 없는가 | **누출 0행** (두 뷰 × 세 수준 전부) |
| 포켓이 없는 표적이 "cold"로 세어졌는가 | 아니오 — 포켓 없음은 커버리지 문제이지 홀드아웃이 아니므로 제외 |
| 파이프라인이 채점할 수 있는 표적인가 | strict 25개 중 수용체 24개, 도킹 박스 25개 |
| 기존 채점기가 패널을 받는가 | `load_ranking_panel`이 396개 질의로 수용 |
| pair 키 정의가 벤치마크 빌더와 같은가 | 같음 (`ligand_inchikey → ligand_id → ligand_smiles`, 대문자) |

## 형식

각 뷰가 두 가지 모양으로 나온다.

- `pocket_cold_<level>_<view>.parquet` — **간선 중심**(화합물–표적 쌍 한 줄). 감사용.
  `absent_pair_from_reference`, `absent_publication_from_reference`,
  `absent_pocket_<level>_from_reference`, `pocket_cluster_<level>` 열을 갖는다
- `pocket_cold_<level>_<view>_ranking.parquet` — **화합물 중심**
  (`query_id`, `truth_targets`, …). `eval/activity_retrieval_model.load_ranking_panel`이
  그대로 읽는다

## 이 패널이 답할 수 있는 질문과 없는 질문

**답할 수 있다**: "포켓 클러스터 전달이, 그 클러스터를 학습에서 본 적 없는 표적에 대해
알려진 표적의 순위를 올리는가."

**답할 수 없다**:
- 활성/비활성 분류 — 이 벤치마크는 **전량 양성**이다(train/dev/test 모두 binary_label=1).
  회수 평가용이지 분류 평가용이 아니다
- 커버리지 자체 — 패널은 이미 근거가 있는 표적들로 만들어졌다. 근거가 아예 없는
  표적(약 77%)에 무엇이 되는지는 별개 문제다

## 한계

- **strict가 25표적으로 작다.** 신뢰구간이 넓다. extended(87표적)가 크지만 dev로
  평가하므로 claimable이 아니다
- 화합물당 정답 표적이 평균 1.01개다. 다중 표적 화합물이 거의 없어, 순위 지표가
  단일 표적 회수에 가깝다
- tm40/50/60 세 수준의 패널이 크게 겹친다(표적 25 / 27 / 28). 독립된 세 측정이 아니다

## 다음 단계

이 패널로 Route B를 실제 측정하는 것이 남았다. `eval/evidence_transfer.transfer_scores`가
전달 메커니즘이고 `eval/activity_retrieval_model.score_ranking_panel`이 채점기이므로,
전달 있음/없음의 순위를 같은 패널에서 비교하면 된다.

측정 전에는 **Route B가 커버리지를 올린다는 것도, 순위를 유지한다는 것도 주장할 수 없다.**
원 문서의 "+2,082 newly scorable / dual-cold 16 / 98"은 이 저장소에서 재현되지 않은
인용 수치다(`docs/COVERAGE_ROUTE_C_RETIRED_20260829.md` 검증 표 참조).
