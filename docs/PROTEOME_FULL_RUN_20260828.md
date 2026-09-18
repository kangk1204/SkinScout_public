# 인간 프로테옴 전체 실행 (2026-08-26 ~ 08-28)

Comprehensive 경로를 프로테옴 전체로 처음 완주했다. 입력은 카페인
(`Cn1c(=O)c2c(ncn2C)n(C)c1=O`), 총 **42시간 21분**.

## 단계별 실측

| 단계 | 결과 | 소요 |
|---|---|---|
| AutoGrid | 15,038 중 **13,339 map_ready** / 1,699 실패 | 19h 46m |
| AutoDock-GPU | **13,339 도킹**, 경고 0건, ΔG −6.69 ~ 8.15 (중앙 −4.48) | 1h 21m |
| DiffDock-L (no-pocket) | **4,961 blind 점수**, confidence −3.23 ~ 1.02 | 6h 23m |
| pick_top (2%) | **366개** = AutoDock 269 + DiffDock 97 | 즉시 |
| GNINA (pose) | 269 채점 / 97 pose 없음(기록) | 10m |
| RTMScore (pose) | 269 채점 / 97 pose 없음(기록) | 60m |
| Boltz-2 | **316 / 366** | 11h 44m |
| RRF 4-way | **50행** — 44행 4소스, 6행 3소스 | 즉시 |

디스크 205 GB(대부분 AutoGrid 맵, 중간 산출물이라 삭제 가능).

## 상위 5개

| 순위 | 유전자 | 단백질 | ΔG | GNINA | RTM | Boltz |
|---:|---|---|---:|---:|---:|---:|
| 1 | PNMT | Phenylethanolamine N-methyltransferase | −6.24 | 4.40 | 20.50 | 0.32 |
| 2 | C17orf99 | Chromosome 17 open reading frame 99 | −6.63 | 4.01 | 14.86 | 0.49 |
| 3 | GAS6 | Growth arrest specific 6 | −6.29 | 4.37 | 20.69 | 0.24 |
| 4 | SMS | Spermine synthase | −6.69 | 4.37 | 15.30 | 0.35 |
| 5 | METTL2A | Methyltransferase 2A, methylcytidine | −6.21 | 4.58 | 19.96 | 0.21 |

## 가장 중요한 결과: 같은 pose를 보는 둘만 서로 동의한다

네 방법 모두 점수가 있는 **220개 표적**(상위 50 선택 이전이라 범위 제한 없음)의
Spearman 상관:

| 쌍 | ρ |
|---|---:|
| **GNINA vs RTMScore** | **+0.450** |
| RTMScore vs Boltz-2 | +0.118 |
| AutoDock(−ΔG) vs RTMScore | +0.104 |
| AutoDock vs GNINA | +0.042 |
| AutoDock vs Boltz-2 | −0.015 |
| GNINA vs Boltz-2 | −0.154 |

**서로 동의하는 쌍은 GNINA와 RTMScore 하나뿐이고, 그 둘이 정확히 같은 AutoDock pose를
재채점하는 쌍이다.** 나머지는 전부 0 근처다 — AutoDock의 자체 ΔG도, pose를 보지 않는
Boltz-2도 다른 어느 것과도 상관되지 않는다.

이번 세션의 F-02 수정(GNINA·RTMScore를 자유 리간드가 아니라 도킹 pose로 재배선)이
프로테옴 규모에서 확인된 셈이다. 같은 기하를 보면 서로 동의하고, 안 보면 동의하지 않는다.

## 카페인의 알려진 표적은 상위 50에 없다

아데노신 수용체 4종(ADORA1/2A/2B/3)도 PDE3A도 들어오지 않았다. **버그가 아니라 이미
측정된 성질이다** — `docs/DAINA_STRUCTURAL_FIRST_RUN_20260825.md`에서 AutoDock ΔG의
알려진 표적 회수는 평균순위 122/210(우연 105.5)로 우연과 구분되지 않았다.

따라서 이 산출물은 **"카페인의 표적을 찾았다"가 아니라 "프로테옴 전체에 구조 계산을
수행했고 그 순위는 이렇다"**로 읽어야 한다. 실제로 알려진 표적을 회수하는 것은 리간드
유사도 검색(fast 경로)이며, 거기서는 아데노신 수용체 4종이 상위 8위 안에 들어온다 —
다만 카페인이 그 표적들의 참조 리간드에 자기 자신으로 들어 있어 뷰어가 `조회`로 표시한다.

## 이 실행이 드러낸 결함

축소 실행에서는 나타나지 않고 전체 규모에서만 나온 것들이다.

| 결함 | 증상 | 수정 |
|---|---|---|
| 상대 경로 리간드 | AutoDock가 맵 디렉터리로 `cd`한 뒤 `Can't open ligand data file`, **13,339개 전부 실패** | `3fac05da` |
| cross-device 발행 | pose를 tmpfs에 스테이징 후 `os.replace`로 실디스크 이동 실패(EXDEV). **모든 pose를 쓴 뒤** 죽음 | `3fac05da` |
| pose parity | top 목록에 DiffDock 전용 97개가 섞여 GNINA·RTMScore가 즉시 실패 | `--allow-unposed-targets` |
| Boltz-2 무음 실패 | 50개 중 6개가 경고 없이 빠짐 | `92713533` |

첫 두 개는 **이번 세션에 추가한 경고 덕분에** 찾을 수 있었다 — 같은 실패가 이전에는
무음의 0행이었는데, 이번에는 13,339건의 경고로 남았다.

## 산출물

```
results/runs/proteome_full_20260826/
├── 03_targets/mode_comprehensive/
│   ├── top50_4way_consensus.csv      최종 순위 50개
│   ├── autodock_all_targets.tsv      13,339개 도킹 에너지
│   ├── diffdock_blind_full.tsv       4,961개 blind 점수
│   ├── autodock_all_target_poses/    13,339개 pose (56 MB)
│   └── autogrid_maps/                205 GB (삭제 가능)
└── viewer/index.html                 오프라인 3D 뷰어 (18 MB)
```
