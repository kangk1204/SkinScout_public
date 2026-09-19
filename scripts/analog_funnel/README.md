# analog_funnel — 후보 발굴 펀넬 오케스트레이션

피부 표적 후보를 만들고(생성) → 걸러내고(게이트·유사도·물성) → 안전성·도킹으로 검증하는
세션 드라이버 모음입니다. 결과 데이터(후보 SMILES·ID·점수)는 이 저장소에 넣지 않습니다
(공개 스냅샷에는 `scripts/tests/test_public_snapshot.py`의 가드 테스트가 강제).

## 트랙

- **A (기지유사체)**: `discover_substitutes.py`(레포) 결과를 `select_candidates.py`로 게이트+MMR 선별
- **B (BRICS 준신규)**: `brics_generate.py`로 단편 재조합 → `validate_generated.py`로 검증
- **C (linkinvent 신규)**: REINVENT4 LinkInvent 샘플링 후 `linkinvent_filter.py`로 필터

## 실행 순서 (기본)

1. 생성
   - B: `brics_generate.py` (입력: `/tmp/opencode/genb/...` 관례 경로)
   - C: REINVENT4 `sampling`(prior=`linkinvent.prior`, warheads 파일) → `linkinvent_filter.py`
2. 선별
   - A: `select_candidates.py`
3. 검증(공통)
   - 안전성: `run_safety_chain.py` → 실패분 `run_safety_degraded.py`
   - 도킹: `pair_docking.py` → 실패분 `pair_docking_repair.py`
   - 생성 후보: `validate_generated.py`

## 환경 변수

- `MICROMAMBA_BIN` (기본 `micromamba`)
- `FUNNEL_DIR` (기본 `/tmp/opencode/analog_funnel`) — 작업 디렉터리
- `FUNNEL_HOME` — 비공개 워크스페이스 루트(비공개 데이터가 있는 디렉터리). `target_pipeline_20260913/queue_top10.csv`, `analog_funnel_20260915/analog_candidates.csv`를 참조합니다.
- `FUNNEL_SEED_SMILES` — `linkinvent_filter.py`가 기준으로 삼을 시드 SMILES
- 그 외 작업 경로는 각 스크립트 상단 상수에서 조정합니다.

## 전제

- 레포 데이터: `data/human_pdbqt`, `data/docking_boxes_derived`, `results/runs/*`
- conda env: `cosmax-base`(본체), `cosmax-autodock-gpu`(도킹), `cosmax-reinvent`(C 생성)
- REINVENT4는 레포 밖(`~/.local/opt/REINVENT4`)에 설치하고 prior는 공개 배포판을 사용합니다.

## 알려진 한계(다음 작업)

- 안전성 결정 파일이 비어 있으면 통과로 처리되는 fail-open 경로가 남아 있습니다.
- Daina top-256 밖 표적 쌍 도킹은 `--selected-csv` 대신 `stage3_autogrid_maps.py --receptor-list`
  (비랭킹) 경로를 사용해야 하며, 일부 드라이버는 아직 랭크를 합성합니다.
- BRICS `BRICSBuild`는 `seed` 인자를 지원하지 않아 열거 순서가 실행마다 달라질 수 있습니다
  (`maxDepth` 고정으로 영향 축소).
- B/C 트랙에는 MMR 다양성 선택이 아직 없습니다.
- 락 파일은 원자적 생성이 아니므로 동시 실행을 피하세요.
