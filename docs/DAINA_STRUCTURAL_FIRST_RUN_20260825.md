# Daina 구조 오버레이 경로 최초 실행 (2026-08-25)

`daina_structural_targets.csv`가 이 저장소에서 처음으로 생성되었다. 커밋된 44개 실행은
모두 `sources = autodock;gnina` 도킹 경로였고, Daina 1차 + 구조 주석 경로는 한 번도
끝까지 돈 적이 없었다.

- 질의 화합물: caffeine (`Cn1c(=O)c2c(ncn2C)n(C)c1=O`)
- 실행 디렉터리: 스크래치 (커밋하지 않음). 산출물 수치만 여기에 고정한다.
- 박스: `data/docking_boxes_derived` (P2Rank pocket 범위에서 유도)
- known-target prior: `workflow/config.yaml` 동결값 그대로 `enabled: false` → 헤더만
  있는 CSV. 정답 주입 없음.

## 실행 결과

| 단계 | 결과 | 소요 |
|---|---|---|
| Daina-Zoete 프로테옴 검색 | 4,727 표적 채점 | — |
| top-256 선별 | 256행 | — |
| AutoGrid 맵 | 210 map_ready / 33 structure_failed_map / 13 no_pocket | — |
| AutoDock-GPU | **210/210** 에너지 산출 | 72초 |
| GNINA CNN 재채점 (CPU) | **210/210** 성공 | 8분 |
| 오버레이 | 256행 × 39열 | 1초 미만 |

박스 부피가 27,000 Å³를 넘는 39/210 수용체는 `-nrun`이 4 → 32로 자동 상향되었다.
AutoGrid 캐시는 4.5 GB.

### 두 개의 실제 장애를 통과해야 했다

1. **`-lsmet adadelta`** — AutoDock-GPU v1.6이 받지 않는 값. 바이너리가 job setup에서
   거부한 뒤 **exit 0**으로 끝나 210개 수용체가 경고 한 줄 없이 `written=0`으로 끝났다.
   커밋 `fa33ec1d`에서 수정.
2. **GNINA GPU** — `Invalid handle. Cannot load symbol cublasLtGetVersion`로 210/210
   실패. `--use-gpu` 없이 CPU로 돌리면 210/210 정상. 코드 문제가 아니라 CUDA 라이브러리
   불일치다.

## 핵심 발견: 구조 계층은 순위에 전혀 기여하지 않는다

210개 표적이 `structure_supported`(AutoDock ΔG + GNINA CNN 둘 다 보유) 상태인데도:

```
final_score  == daina_score  전행 일치
docking_rrf  == daina_score  전행 일치
score        == daina_score  전행 일치
```

`stage3_daina_structural_overlay.py:469-473`에 의도가 명시돼 있다 —
`# Compatibility fields retain the Daina primary score/order.` 즉 오버레이는 설계상
**주석 계층**이며 재랭킹을 하지 않는다. 버그가 아니다.

다만 `docking_rrf`라는 컬럼명은 도킹 정보를 전혀 담지 않으므로 오해를 부른다.
`final_score`도 융합 점수가 아니라 원시 max Tanimoto다.

## 신호는 어디에 있나 (단일 화합물 관찰, n=4)

카페인의 알려진 표적 중 구조 근거가 붙은 4개(ADORA1/2A/2B/3)의 210개 내 순위:

| 지표 | ADORA2A | ADORA2B | ADORA1 | ADORA3 | 평균순위 |
|---|---:|---:|---:|---:|---:|
| Tanimoto | 1 | 1 | 1 | 1 | **7.0** |
| AutoDock ΔG | 113 | 150 | 138 | 86 | **122.1** |
| GNINA CNNaffinity | 7 | 86 | 4 | 1 | **24.5** |

우연 기대값은 105.5다. 따라서:

- **AutoDock ΔG는 무작위와 구분되지 않는다** (122.1). ΔG 상위 10개(Q04771, Q9NYV8,
  P15121 …)는 Tanimoto 순위 42~226이고, 진짜 표적 4개는 ΔG로는 86~150위다. ΔG로
  재랭킹하면 정답이 오히려 아래로 밀린다.
- **GNINA CNN은 실제 신호를 담는다** (24.5). ADORA3 1위, ADORA1 4위, ADORA2A 7위.

**단, Tanimoto 결과는 동어반복이다.** 카페인은 이 4개 표적의 참조 리간드 집합에 그
자신이 들어 있어 Tanimoto가 정확히 1.0이다. 즉 "회수"가 아니라 자기 자신을 찾은 것이다.
일반화 성능을 보려면 leave-query-out이 필요하다. 화합물 1개·표적 4개 관찰이므로
검증된 결론이 아니라 **다음 검증의 가설**로만 다룬다.

또한 알려진 표적 5개 중 PDE3A(Q14432)는 Tanimoto 0.407로 58위이고 맵 생성에
실패(`structure_failed_map`)해 구조 근거가 아예 없다.

## 산출물이 노출하지 않는 것

- `daina_top256.csv`는 `daina_max_tanimoto`와 `daina_known_ligand_count`를 갖고 있지만
  **오버레이가 둘 다 버린다.** 최종 CSV 39열 어디에도 없다.
- `target_name`은 `stage3_daina_structural_overlay.py:437`에서 `""`로 하드코딩돼 있어
  전 행이 비어 있다. 사람이 읽을 유전자명이 없다.
- 256개 중 Tanimoto ≥ 0.5는 31개뿐, < 0.3이 151개다. 상위 256에 들었다는 사실 자체가
  근거의 강도를 뜻하지 않는다.

## 테스트

`scripts/tests` 전량: 2,543 passed / 39 skipped / 0 failed.

(GNINA CPU 재채점과 동시 실행한 회차에서 `test_prepare_performance_v2_inputs.py`와
`test_stage2_5.py`가 각각 SIGABRT(`terminate called without an active exception`)로
실패했으나, 서브프로세스는 올바른 출력을 낸 뒤 종료 중 크래시한 것이고 부하 없는
재실행에서 전량 통과했다.)
