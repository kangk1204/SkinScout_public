# F-02 pose 재배선과 검증 (2026-08-26)

Comprehensive의 `gnina_rescore_top`과 `rtmscore_top`이 AutoDock가 내보낸 pose가 아니라
xTB로 최적화한 **자유 리간드**를 채점하고 있었다. `top50_4way_consensus.csv`를 "같은 pose에 대한 4중
재채점"으로 읽을 수 없다는 감사 지적(F-02)에 대한 수정과 실측 검증이다.

## 무엇을 바꿨나

`workflow/rules/stage3a_comprehensive.smk`

```
- ligand_sdf = rules.xtb_optimize.output.sdf
+ poses      = rules.autodock_gpu_all.output.poses
```

호출도 pose 모드로 바뀐다 — `--pose-manifest`, `--pose-dir`, `--out-status-manifest`.
GNINA는 매니페스트의 SHA-256으로 각 pose를 검증한 뒤 채점한다.

### 부분집합 재채점을 명시적으로 허용해야 했다

pose 모드는 원래 top 목록과 pose 매니페스트의 **정확한 집합 일치**를 요구했다. Fast
경로는 도킹한 표적을 전부 재채점하므로 문제가 없었지만, comprehensive는 프로테옴 전체를
도킹하고 상위 비율만 재채점한다. 여분 pose 때문에 즉시 실패한다.

`--allow-unscored-poses`를 추가했다. **채점할 표적에 pose가 없는 경우는 여전히 치명적**이고,
여분 pose만 명시적 요청이 있을 때 허용한다. 기본값은 그대로 정확 일치다.

## 검증

**환경 제약을 먼저 밝힌다.** 이 머신에는 `snakemake`, `xtb`, DiffDock이 없어 **전체
comprehensive DAG는 실행할 수 없다.** 대신 재배선이 실제로 건드리는 체인을 실제
스크립트·실제 AutoDock pose로 돌렸다.

입력: 2026-08-25 카페인 실행의 실제 산출물 — AutoDock 210개 표적 점수와 210개 pose
(SHA 매니페스트 포함).

| 단계 | 방법 | 결과 |
|---|---|---|
| top 선별 | 실제 `stage3_pick_top.py` | 210개 중 **50개** 부분집합 |
| 음성 대조 | 플래그 없이 pose 모드 | **거부** — 여분 pose를 지적하고 플래그를 안내 |
| GNINA 재배선 | `--allow-unscored-poses` | **50/50 채점**, 110초 |
| GNINA 이전 배선 | `--ligand-sdf` 자유 리간드 | 50/50 채점, 114초 |
| RTMScore 재배선 | pose 모드 | **50/50 채점**, 13분 23초 |
| RTMScore 이전 배선 | 자유 리간드 | 50/50 채점, 13분 50초 |
| RRF 3-way | `min_sources=3` | **정상** — 50행, `autodock;gnina;rtm` |

### 이전 배선은 무해한 근사가 아니었다

같은 50개 표적에 대한 두 모드의 CNN affinity 비교:

| 지표 | 값 |
|---|---|
| 동일한 값이 나온 표적 | **0 / 50** |
| 차이 &#124;Δ&#124; 중앙값 | 1.234 |
| 차이 &#124;Δ&#124; 최댓값 | 3.91 |
| 두 모드의 Spearman | **−0.16** |

예:

| 표적 | pose 채점 | 자유 리간드 채점 |
|---|---:|---:|
| Q04771 | 3.31 | 2.11 |
| Q9NYV8 | 3.85 | 2.01 |
| P15121 | 4.21 | 1.70 |

**두 값은 사실상 무관하다.** 이전 GNINA 컬럼은 자기가 재채점한다고 표시된 pose와
상관이 없었고, 그 컬럼이 4-way 합의 순위에 들어가고 있었다. 자유 리간드 점수가
일관되게 낮은 것도 예상과 맞는다 — 포켓에 배치되지 않은 구조를 채점한 값이다.

### 산출물에 계보가 남는다

pose 모드 출력은 `pose_file`, `pose_sha256`, `scored_actual_docked_pose` 컬럼을 갖는다.
자유 리간드 경로에는 없던 값이다. 상태 매니페스트(`skinscout.gnina-pose-status.v1`)에
50건이 기록됐다.

## RTMScore — 확장했고, 그전에 왜 한 번도 안 돌았는지가 드러났다

`scripts/stage3_rtmscore.py`에도 같은 pose 모드를 붙였다(`--pose-manifest`, `--pose-dir`,
`--allow-unscored-poses`, `--out-status-manifest`). 매니페스트 읽기와 parity 검사는
`scripts/docking_pose_manifest.py`로 뽑아 GNINA와 **한 벌만 유지**한다.

확장하려고 보니 **RTMScore는 이 머신에서 한 번도 점수를 낸 적이 없었다.** 원인은 F-21과
같은 유형이다 — 설치된 DGL이 CPU 전용인데 `rtmscore/__init__.py`가 `torch.cuda.is_available()`만
보고 CUDA를 골라, DGL이 `Device API cuda is not enabled`로 죽었다. 그리고
`stage3_rtmscore.py`가 모든 예외를 `LOG.debug`로 삼켜서, 환경 결함이 "RTMScore produced
no usable rescores" 한 줄로만 보였다.

- device 선택이 DGL이 실제로 CUDA를 쓸 수 있는지 확인하고 안 되면 CPU로 내려간다(경고 동반)
- 명시적 `--device`를 받는다
- 개별 실패를 `LOG.warning`으로 올려 원인이 보이게 했다

### RTMScore 두 모드 비교 (같은 50개 표적)

| 지표 | 값 |
|---|---|
| 동일한 값이 나온 표적 | **0 / 50** |
| &#124;Δ&#124; 중앙값 | 5.65 (점수 범위 0.76–21.46) |
| 두 모드의 Spearman | **0.22** |

## 합의 순위가 실제로 바뀐다

같은 50개 표적으로 `min_sources=3` 3-way 합의(AutoDock + GNINA + RTMScore)를 두 배선으로
각각 만들어 비교했다.

| 지표 | 값 |
|---|---|
| 순위가 그대로인 표적 | **1 / 50** |
| &#124;순위 변동&#124; 중앙값 | 13 |
| &#124;순위 변동&#124; 최댓값 | 34 |
| Spearman | **0.29** |
| **상위 10개 중 겹치는 수** | **2 / 10** |

이전 배선은 무해한 근사가 아니라 **실질적으로 다른 상위 표적 목록**을 내고 있었다.

따라서 `top50_4way_consensus.csv`는 이제 **네 열 중 셋이 AutoDock pose를 설명한다** —
AutoDock는 자기가 만든 pose를, GNINA와 RTMScore는 그 같은 pose를 SHA 검증 매니페스트로
재채점한다. Boltz-2는 pose를 보지 않는 독립 구조 예측기이므로 여전히 예외다.

## 아직 하지 않은 것

전체 comprehensive 실행은 이 환경에서 불가능하다(snakemake·xtb·DiffDock 부재). Boltz-2
열도 이 검증에 포함하지 못했다 — pose를 쓰지 않는 독립 예측기라 이번 수정 대상이
아니지만, 실제 4-way 합의는 확인하지 못했다는 뜻이다.

프로테옴 규모의 비용은 실제 실행에서 재야 한다. 이 머신 기준 실측으로 외삽하면
상위 2%(~400개) 재채점에 GNINA 약 15분, **RTMScore 약 1시간 50분**(50개에 13분 23초)이다.

RTMScore 비용을 CUDA DGL로 줄일 수 있는지 별도로 설치해 실측했다 — **6.6% 개선에
그쳤다.** 병목이 신경망이 아니라 CPU 쪽 그래프 구성이기 때문이다. 도입하지 않았다.
`docs/RTMSCORE_CUDA_DGL_20260826.md`.
