# Comprehensive 경로 차단 해소와 DAG 검증 (2026-08-26)

"DiffDock을 설치하고 comprehensive를 실행해 달라"에서 출발했다. 실제로는 **DiffDock이
병목이 아니었고**, 그보다 앞에 두 개의 구조적 결함이 있었다.

## 결함 1: Snakemake 워크플로가 아예 파싱되지 않았다

```
WorkflowError in workflow/rules/stage3b_fast.smk, line 220:
Rule known_target_prior is not defined in this workflow.
```

`stage3b_fast.smk`가 `rules.known_target_prior`를 참조하는데, 그 규칙을 정의하는
`stage3_v3_skin_weight.smk`가 **뒤에** include되어 있었다. Snakemake는 rule input을 파싱
시점에 평가하므로 실패한다.

**fast·comprehensive 어느 모드에서도 워크플로가 로드되지 않았다.** 변경 사항을
`git stash`한 상태에서도 같은 오류가 재현되므로 이번 작업이 만든 문제가 아니다.

이 머신에 `snakemake`가 설치된 적이 없어(환경 자체가 없었다) 아무도 부딪히지 않은 것으로
보인다.

**수정**: `stage3_v3_skin_weight.smk`를 stage 3 경로들보다 먼저 include한다. 이 파일은
stage2_5 규칙에만 의존하므로 안전하다.

그러자 순환이 드러났다 — `S3C_DIR`/`S3F_DIR`이 `stage3a_comprehensive.smk`에 정의돼 있는데
`stage3_v3_skin_weight.smk`가 그걸 쓴다. 두 상수는 `RUN_DIR`에서 파생되는 단순 경로이므로
**Snakefile의 include 이전으로 올렸다.**

## 결함 2: comprehensive에 AutoGrid 규칙이 없었다

AutoGrid는 **fast 경로에만** 존재했다(`fast_autogrid_maps`). Comprehensive의
`autodock_gpu_all`은 맵 매니페스트를 받지 않고 수용체 옆의 `.maps.fld`를 찾는데,
`data/human_pdbqt/`에는 `.fld`가 **0개**다. 실증:

```
INFO Docking 15038 receptors
INFO AutoDock-GPU map coverage: 0/15038 selected targets
AutoDock-GPU map coverage incomplete: 0/15038 receptors have .maps.fld inputs
```

**수정**: `comprehensive_receptor_list`(수용체 목록 생성)와
`comprehensive_autogrid_maps`(맵 생성) 규칙을 추가하고 `autodock_gpu_all`에
`--map-manifest`를 연결했다.

`stage3_autogrid_maps.py`는 Daina 선별 CSV(연속 순위·점수)를 요구했다. Comprehensive는
순위가 없으므로 가짜 순위를 만들어 넣는 대신 **`--receptor-list` 모드**를 추가했다 —
순위 없는 표적 ID 목록을 받고, 매니페스트의 `daina_score`는 null로 둔다. 어느 입력이
쓰였는지는 provenance에 각각의 해시와 함께 기록된다.

## 검증: DAG가 끝까지 해석된다

```
snakemake -s workflow/Snakefile --cores 16 --use-conda \
  kg_efficacy_label disagreement_analysis --resources gpu=1 -n \
  --config run_id=comp_dryrun mode=comprehensive compound_smiles=... 
→ rc=0, Job stats: total 47
```

stage 3 체인이 올바른 순서로 잡힌다:

```
meeko_ligand → comprehensive_receptor_list → comprehensive_autogrid_maps
→ autodock_gpu_all → autodock_pick_top_pct
→ {gnina_rescore_top, rtmscore_top, boltz2_affinity_top}
→ rrf_4way_consensus → skin_weight_apply → kg_efficacy_label → disagreement_analysis
```

## 준비한 환경

이 머신에는 파이프라인 자체 환경이 **하나도 없었다**. 생성한 것:

| 환경 | 확인 |
|---|---|
| `cosmax-base` | snakemake 9.25.2 |
| `cosmax-qm` | xtb 6.7.1 |
| `cosmax-meeko` | mk_prepare_ligand.py |
| `cosmax-autodock-gpu` | 생성됨 |
| `skinscout-diffdock` | torch 2.4.1+cu121, pyg 2.8.0, e3nn 0.6.0, torch_cluster 1.6.3 |

DiffDock 체크아웃(`~/.local/opt/DiffDock`)과 어댑터(`~/.local/bin/diffdock`)도 만들었다.
어댑터는 README에서 경고했던 두 가지 차이를 흡수한다 — 업스트림은 `--ligand`가 아니라
`--ligand_description`을 받고, 결과를 `<out_dir>/<complex_name>/`에 쓴다.

## 축소 실행 (수용체 300개)

`docking.comprehensive_max_receptors`로 수용체를 300개로 제한하고 comprehensive stage 3
체인을 **실제로 끝까지** 돌렸다. Snakemake로 돌리면 Stage 0 미러(ChEMBL, RCSB 스냅샷)와
activity-retrieval 평가까지 함께 걸리므로, 도킹 체인만 같은 스크립트·같은 인자로 실행했다.

| 단계 | 결과 | 시간 |
|---|---|---|
| AutoGrid | **292 map_ready / 8 structure_failed_map** | 11분 42초 |
| AutoDock-GPU | **292 docked**, ΔG −5.85 ~ −2.80 | 1분 40초 |
| pick_top (2%) | 상위 **50개** (하한 50 적용) | 즉시 |
| GNINA (pose 모드) | **50/50**, `scored_actual_docked_pose=true` | 1분 29초 |
| RTMScore (pose 모드) | **50/50**, `scored_actual_docked_pose=true` | 3분 35초 |
| Boltz-2 | **44/50**, 점수 0.119 ~ 0.589 | 71분 48초 |
| RRF 4-way (`min_sources=3`) | **50행** — 44행 4소스, 6행 3소스 | 즉시 |

도킹 체인(Boltz 제외)이 **18분 30초**, Boltz-2까지 포함해 **90분**에 끝났다. 상위 8개:

```
target_id   rrf_score  src  autodock  gnina  rtm  boltz
A0A0C4DH69   0.054627    4        20     19    5     11
A0A0C4DH25   0.053356    4        39     31    3      1
A0A4W9AIG4   0.053287    4        18      3    9     39
A0A1B0GV85   0.053265    4        12     35   13      6
```

AutoDock·GNINA·RTMScore 세 열이 **같은 AutoDock pose**를 설명한다 — 이번 세션의 F-02
수정이 실제 실행에서 확인된 것이다. Boltz-2는 pose를 보지 않는 독립 예측기다.

### Boltz-2: 실측 86초/표적, 그리고 조용한 실패 6건

Boltz-2는 `--use_msa_server`로 원격 MSA 서버를 조회하므로 GPU 사용률이 0%다. 실행 초반
진행이 느려 22분/표적으로 추정했으나 **틀렸다** — 50개를 71분 48초에 끝냈으므로
**86초/표적**이다. 초반 지연은 서버 대기였던 것으로 보인다.

50개 중 **44개만 채점**됐고 로그에는 경고가 **한 줄도 없었다.** `call_boltz`의 모든 실패
경로가 이유 없이 None을 반환하고 있었다. 이번 세션에서 GNINA·RTMScore에 고친 것과 같은
결함이다.

사유를 노출하도록 고친 뒤 실패한 표적을 다시 돌리자 원인이 드러났다:

```
WARNING Boltz-2 failed for A1A5B4_clean.pdb: exit 1:
  FileNotFoundError: .../predictions/input/pre_affinity_input.npz
```

Boltz 내부에서 affinity 입력을 만들지 못한 것이다. **중요한 것은 이 6개가 꼬리가 아니라는
점이다** — A1A5B4, A0A7I2V3D3, A0PJE2는 나머지 세 채점기로 만든 3-way 합의의 **상위 3위**였다.
조용히 빠졌다면 4-way 합의에서 상위권이 통째로 달라진 것을 아무도 몰랐을 것이다.

### AutoGrid 비용 추정을 정정한다

수용체 5개 표본에서 7.6초/수용체로 추정했으나, 300개 실측은 **2.3초/수용체**였다.
표본이 작았고 수용체 크기 분포도 달랐다. 프로테옴 전체(15,038개) 추정은
**9.6시간 ~ 32시간** 범위로 봐야 하며, 정확한 값은 전체 실행에서만 알 수 있다.
디스크는 292개 맵에 3.1 GB였으므로(10.9 MB/수용체) 전체는 약 160 GB다.

## 실행하지 않은 이유와 비용

사용자 선택에 따라 **긴 실행은 걸지 않았다.** 실측 기반 예상 비용:

| 단계 | 비용 |
|---|---|
| AutoGrid 15,038개 | ~32시간, 235 GB (7.6초·16 MB per 수용체 실측) |
| AutoDock-GPU | ~85분 |
| DiffDock (no-pocket 5,133개) | 미측정 |
| GNINA / RTMScore (상위 2% ~300개) | 11분 / 80분 |
| Boltz-2 (~300개) | 미측정 |

DiffDock 모델 가중치는 아직 받지 않았다(`workdir/v1.1/` 부재). dry-run은 규칙을 실행하지
않으므로 DAG 검증에는 필요하지 않았다.
