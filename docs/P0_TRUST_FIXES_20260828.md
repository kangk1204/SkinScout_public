# P0 신뢰성 수정 — 화면이 사실과 다르게 말하던 것들

- 작성일: 2026-08-28 (Asia/Seoul)
- 기준 커밋: `c8d66d5f`
- 근거 감사: `docs/SKINSCOUT_FULL_CODE_IO_VISUAL_AUDIT_20260828.md` (62건, RELEASE BLOCK)
- 범위: 연구자가 보는 화면이 **거짓을 말하던 결함**. 기능 추가가 아니라 정정.

## 1. 오프라인 3D 뷰어가 브라우저에서 열리지 않았다

README:189는 `viewer/index.html`을 **더블클릭하라**고 안내한다. 그 경로로 열면
구조가 뜨지 않았다.

### 재현 (수정 전 코드, 저장소 안에서 실행)

```
$ python scripts/make_results_viewer.py --run-dir results/runs/proteome_full_20260826 --out-dir /tmp/vold
뷰어 생성 완료: /tmp/vold/index.html  (3개 표적, 채점 autodock, gnina, rtm, boltz)

$ (headless Chrome, file:// 로 targets/P11086.html 열기)
{"fallbackVisible": true,
 "fallbackText": "3D 뷰어를 열 수 없습니다: viewer.clear is not a function ...",
 "canvases": 1}
```

두 개의 독립된 원인이 겹쳐 있었다.

1. `make_results_viewer.py:435`가 `viewer.clear()`를 불렀다. Mol* `Viewer`에는 그런
   메서드가 없고 `viewer.plugin.clear()`가 정상 API다 — 같은 저장소의
   `scripts/report_assets/fast_report.js:33-38`은 가드까지 두고 올바르게 쓰고 있었다.
2. 그 한 줄을 고쳐도 `loadStructureFromUrl`이 `file://`에서 CORS로 차단된다.
   브라우저는 `file://` 페이지가 **같은 폴더의 형제 파일조차** fetch하지 못하게 한다.

### 수정

- `viewer.plugin.clear()` + 없으면 명시적 실패
- `loadStructureFromUrl` → **`loadStructureFromData`**. 수용체 PDB와 pose SDF 텍스트를
  페이지 안에 `<script type="application/json">`으로 심어 fetch를 아예 없앴다.
  내려받기 링크는 `structures/` 파일을 그대로 가리킨다.

### 수정 후 (같은 검사)

```
{"pageErrors": [], "consoleErrors": [], "fallbackVisible": false,
 "canvases": 1, "molstar": {"trajectories": 2, "atoms": 4218}}
```

수용체 2,085원자 + pose 24원자가 각각 한 번씩 로드된다.

### 완료 게이트

`scripts/tests/test_results_viewer.py::test_the_viewer_renders_in_a_real_browser_from_a_file_url`
가 Playwright + 로컬 Chrome으로 `file://` 페이지를 열어 콘솔 에러 0, trajectory 2개,
원자 수 > 0을 단언한다. **이 게이트는 단위 테스트가 잡지 못한다** — 수정 전 코드는
이 파일의 다른 모든 테스트를 통과하면서 브라우저에서는 아무것도 그리지 않았다.

playwright는 `cosmax-base`에 설치했다(`pip install playwright`, 1.62.0).
Chrome은 `/usr/bin/google-chrome` 149.0.7827.155를 쓴다.

## 2. "근거 수"가 합의를 뜻한다고 설명했다

`make_results_viewer.py:327`의 원문:

> 근거 수 — 이 표적을 지지한 채점 방법의 개수입니다. **클수록 여러 방법이 동의했다는 뜻입니다.**

`docs/RESEARCHER_GUIDE.md:230-232`가 금지한 바로 그 독법이다. 실측:

| 측정 | 값 |
|---|---|
| 프로테옴 실행 220표적에서 상관이 있는 방법 쌍 | GNINA–RTMScore (ρ=+0.45) **하나뿐** |
| 나머지 모든 쌍 | ≈ 0 |
| 상위 50개 방법 간 순위 편차(max−min) 중앙값 | **201위** |

예: 1위 PNMT는 편차 101위, 3위 GAS6는 **209위**.

### 수정

- 열 이름 `근거 수` → **`점수를 낸 방법 수`**
- 설명을 "동의했다는 뜻이 **아닙니다**"로 바꾸고 측정된 ρ를 함께 적었다
- **`순위 편차` 열 신설.** `:147`에서 수집만 하고 버리던 `{key}_rank` 4개를
  실제로 렌더한다 — 표, 표적 상세(`방법별 순위` 블록), CSV 모두
- 도움말에 이 실행의 편차 중앙값을 계산해 넣는다(다른 실행 수치를 박아두지 않는다)

## 3. AutoDock 전 행이 불완전 격자 기반인데 표시가 없었다

```
autodock_all_targets.tsv: 13,339행 전부 map_coverage_complete=false
커버리지: 13,339 / 15,038 = 88.7%
```

이 열은 어디에도 렌더되지 않았다. 이제 index 상단 배너로 나온다:

> **AutoDock 격자가 불완전합니다.** 수용체 15,038개 중 13,339개(88.7%)에만 도킹 격자가
> 만들어졌습니다. 격자가 없는 표적은 낮은 점수가 아니라 *점수 없음*이며, 이 표에 오르지
> 못한 표적 중에 실제 결합자가 있을 수 있습니다.

## 4. DiffDock 경로가 통째로 보이지 않았다

재채점 후보 366개 중 **97개가 `diffdock_blind` 경로**로 들어왔다(269개는 autodock).
`SCORE_FILES`에 diffdock 키가 없어 어느 화면에도 나타나지 않았다.

이제 `진입 경로` 열과 배너가 나오고, 이 실행에서 확인된 사실도 함께 적는다 —
**상위 50개에는 DiffDock 경로로 들어온 표적이 하나도 없다.**

## 5. 표적 ID로 출력 디렉터리를 벗어날 수 있었다

`target_id`는 CSV에서 오는 신뢰할 수 없는 입력인데 그대로 파일명이 됐다.
`../../escaped`가 지정한 폴더 밖에 파일을 만들었다.

- slug 정규화(`[^A-Za-z0-9_-]` → `_`, 점 불허) + `resolve()` containment 검증
- 전체 출력을 형제 staging 디렉터리에 만든 뒤 `os.replace`로 원자 교체.
  실패하면 이전 뷰어가 그대로 남는다(반쯤 쓰인 뷰어가 최신 결과처럼 보이지 않게)
- staging을 같은 부모에 두는 이유는 `/tmp`가 별도 파일시스템이라 디렉터리를
  건너 옮길 수 없기 때문이다(프로테옴 실행에서 겪은 EXDEV와 같은 원인)

## 6. 단위 설명 자리에 컬럼명이 찍혔다

`:353`이 `SCORE_FILES[key][2]`(컬럼명)를 읽었다. 방향 힌트는 인덱스 3이었다.
실제 출력:

```html
<span>AutoDock ΔG</span><strong>-6.240</strong><span>autodock_energy_kcal_mol</span>
<span>RTMScore</span><strong>20.500</strong><span>None</span>
```

정성껏 쓴 `"낮을수록 좋음 (kcal/mol)"`은 **한 번도 렌더된 적이 없었다.**
`SCORE_FILES` 튜플을 이름있는 `Scorer` dataclass로 바꿔 이 부류의 버그를 구조적으로 없앴다.

## 7. 검증에 실패한 실행이 "완료"·"claimable"로 보였다

`run_summary.json`은 검증기보다 먼저 쓰이는데, Workbench는 그 파일의 존재만으로
`completed`를 붙였다.

커밋된 44개 실행 실측:

| verification | 실행 수 |
|---|---:|
| `status=ok, verifier=ok` | 38 |
| `status=failed, verifier=ok` | 3 |
| `status=failed, verifier=failed` | 3 |

**6개가 실패했는데 전부 "완료"로 표시되고 있었다.** 그중 `ethanol_target_fast_full`은
테스트 픽스처가 아닌 실제 실행이다.

### 수정

- `_verification_state()` 신설 — 검증기 판정을 일급 사실로 다룬다
- 상태: 검증 실패 → **`verification_failed`**, 검증 기록 없음 → **`unverified`**
- `claimable`은 파이프라인 판정 **그리고** 검증 통과를 모두 요구한다
  (현재 44개 중 claimable=True는 **0개**)
- 결과 화면은 계속 열 수 있다. 판정만 정직해진다

## 8. 진행률이 가짜였다

`app.js:360` 원문:

```js
const currentStage = job.status === "queued" ? 0 : job.status === "cancel_requested" ? 5 : 1;
```

실행 중이면 **무조건 6단계 중 2번째**에 고정. 42시간짜리 실행에서 이건 "멈춤"으로 읽힌다.

이제 실제 산출물로 판정한다(`01_input`, `02_admet`, `03_targets`, `04~08_*`,
`09_report`/`run_summary.json`, `run_verification.json`). 판정할 산출물이 없으면
"아직 단계를 판정할 산출물이 없습니다"라고 말하고 아무 단계도 칠하지 않는다.

## 9. 필터를 걸면 전체 순위를 잃었다

- **M-10**: 필터 후 `display_rank`를 1부터 재번호화해 전체 5위가 화면에서 1위로 보였다.
  `rank`(전체 순위)와 `filtered_position`(이 목록 안 위치)을 분리했다
- **M-09**: 5초 폴링이 정렬·필터 결과를 기본값으로 되돌리면서 컨트롤은 선택 상태를
  유지해, 표가 화면 설명과 달라졌다. 필터가 걸려 있으면 같은 질의를 다시 적용한다

## 10. DiffDock이 준비성 검사에서 빠져 green이었다

comprehensive DAG는 `autodock_pick_top_pct`가 `diffdock_blind_no_pocket`의 출력을
**필수 입력**으로 받는다. 그런데:

- `model_readiness.py`의 `STATUS_REQUIREMENTS`에는 `diffdock`이 있었지만
  argparse `choices`에는 없어 **`--require diffdock`이 argparse 단계에서 거부**됐다
- `run_skinscout._required_models`도, `workbench/server.py`의 요구사항 맵도 몰랐다
  (`grep -n diffdock workbench/` → 0건)

DiffDock 없는 기계가 green을 받고 DAG 중간에 죽었다.

- `choices`를 `STATUS_REQUIREMENTS ∪ SPECIAL_REQUIREMENTS`에서 **파생**시켜 드리프트를 차단
- comprehensive/both에 `diffdock` 요구 추가 (fast에는 추가하지 않음)
- Workbench 요구사항 맵에 추가

## 11. 기본 `report` 실행이 MD 인자에서 즉시 죽었다

`Snakefile:1168`이 `config_float("md.duration_ns", 50)`으로 `50.0`을 만들고,
`stage7_gromacs_prep.py:858`과 `stage7_gromacs_run.py:172`가 `type=int`로 받았다 →
`invalid int value: '50.0'`.

시뮬레이션 길이는 분수일 수 있는 물리량이므로 CLI를 `type=float`로 바꾸고,
하한을 `>= 1`에서 `> 0`으로 맞췄다.

## 12. 동작하는 AutoGrid 설치가 차단될 수 있었다

`stage3_autogrid_maps._resolve_parameter_file`은 5개 후보 중 **아무거나** 받는데,
readiness는 `AD4_parameters.dat`만 인정했다. bound-only 설치는 실행 가능한데 차단됐다.

readiness가 같은 후보 집합을 받도록 하고 `resolved_parameter_file`로 실제 쓰일 파일을
보고한다. 소스 텍스트를 단언하던 기존 테스트는 **행동 기반**으로 교체했다.

---

## 검증

| 검사 | 결과 |
|---|---|
| 브라우저 E2E (`file://`, Playwright + Chrome) | 통과 — 콘솔 에러 0, trajectory 2, 원자 4,218 |
| 수정 전 코드로 같은 검사 | `viewer.clear is not a function`으로 실패 |
| `scripts/tests/test_results_viewer.py` | 22 통과 (신규 10) |
| `scripts/tests/test_model_readiness.py` | 21 통과 |
| `scripts/tests/test_workbench_server.py` | 신규 11 통과 |
| `scripts/tests/test_stage7_gromacs.py` | 7 통과 |
| 전체 스위트 | 2,724 통과 / 3 skip |
| 실데이터 재생성 | `proteome_full_20260826` 뷰어 재생성, 88.7%·편차 201·DiffDock 97건이 화면에 나옴 |

## 남은 것

이 문서는 **P0(거짓말 제거)만** 다룬다. 감사 62건 중 나머지, 그리고
Workbench 3D 내장·다운로드·차트·입력 업그레이드·인증은 후속 단계다.
계획은 승인된 업그레이드 계획을 따른다.

특히 아직 손대지 않은 것:
- `09_report`(stage9 9패널)은 여전히 **한 번도 생성된 적이 없다**. README:264가
  이 파일을 안내하고 있으므로 문서 정정이나 경로 복구 중 하나가 필요하다
- 안전성 계층(ADMET 49엔드포인트, 감작 3모델 합의)은 산출되지만
  `results/eval/` 어디에도 **실측 대조 자료가 없다**
