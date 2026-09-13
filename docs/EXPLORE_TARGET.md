# 표적 단백질부터 시작하기

보통은 성분을 넣고 표적을 물어본다. 반대로 단백질을 먼저 정해 두고, 그 표적에
측정 기록이 있는 화합물을 찾을 수도 있다. 검색 인덱스(ChEMBL 37 + BindingDB,
표적 4,873개)를 뒤집어 보는 도구가 `scripts/explore_target.py`다.

```bash
python scripts/explore_target.py P14679          # 티로시나아제 (UniProt)
python scripts/explore_target.py tyrosinase      # 이름으로도 찾습니다
python scripts/explore_target.py --search melan  # 이름 후보부터 훑기
```

표에는 `최대 pActivity`와 양성 측정 수가 함께 나온다. 양성 문턱(6.0) 아래인 줄은
측정은 됐어도 세다고 보기는 어려우니 근거로 쓰지 마세요. 화합물을 파이프라인에
그대로 넣어 그 표적의 근거와 판정을 보려면:

```bash
python scripts/explore_target.py tyrosinase --smiles-out ligands.txt --print-run-cmds
python scripts/run_skinscout.py --smiles '<SMILES>' --mode fast
```

여기서 "알려진 화합물"은 그 표적과 화합물 사이에 측정 기록이 있는 것을 말한다.
이 도구가 결합을 새로 예측한 목록은 아니다. 이미 아는 성분을 넣으면 결과에
`조회`로 표시된다(README의 "시작 전에 알아두실 것").

## 어떻게 동작하나

- 이름 해석: 저장소 안의 큐레이션 표(워크리스트·known_failure_modes·rerank/패널
  라벨)와 잘 알려진 피부 단백질 별칭 사전을 쓴다. UniProt 계정번호·유전자 기호·
  흔한 이름을 받는다.
- 조회: `workflow/config.yaml`의 `daina_recipe_index_dir`가 가리키는 생산 인덱스를
  읽는다(파이프라인과 같은 판). 평가용 인덱스는 거부하고, 인덱스가 없으면
  Stage 0 빌드를 안내한 뒤 멈춘다.
- 정렬: `--mode balanced|potency|evidence`(기본 balanced). `--top`(기본 20)까지
  보여주고, `--print-run-cmds`는 최대 10개 명령을 낸다.

## Workbench 화면

터미널 대신 브라우저를 쓰는 연구자는 왼쪽 `04 표적 검색` 화면에서 같은 조회를
한다. 서버가 이 문서의 CLI와 같은 생산 인덱스를 읽고, 결과 표의 각 행에는
`이 후보 분석` 버튼이 있어 그 화합물을 `새 분석` 입력으로 바로 넘긴다.

표적이 여러 개라면 표를 그대로 `scripts/discover_from_targets.py`에 넘겨 한 번에
발굴한다([`docs/TARGET_LIST_DISCOVERY.md`](./TARGET_LIST_DISCOVERY.md)).

- 생산 인덱스가 없는 설치(예: `--profile analog`)에서는 화면이 그 사실과 빠진
  항목을 버튼 위에 먼저 알린다. 결과를 조용히 빈 표로 보여주지 않는다.
- 표의 `문턱` 배지는 양성(6.0 이상) / 경계(5.0~6.0) / 문턱 아래를 구분한다.
  문턱 아래는 "측정됐다"는 뜻일 뿐 "세다"가 아니다.

