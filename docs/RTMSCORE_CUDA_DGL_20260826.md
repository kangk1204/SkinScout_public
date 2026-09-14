# RTMScore용 CUDA DGL: 설치했고, 권장하지 않는다 (2026-08-26)

RTMScore가 CPU로만 돌아 comprehensive 상위 2%(~400 표적) 재채점에 약 1시간 50분이
걸린다는 추정에서 출발해 CUDA 지원 DGL을 설치하고 실측했다.

**결론: 점수는 완전히 동일하고 속도는 6.6%만 빨라진다. 도입할 이유가 없다.**

앞서 이 문서 이전에 "GPU면 훨씬 빠를 것"이라고 쓴 추정은 측정 없이 한 말이었고 틀렸다.

## 왜 그냥 설치가 안 되는가

| 제약 | 내용 |
|---|---|
| DGL 공식 휠 | **cu118·cu121만** 제공, 최신 2.5.0 (2024) |
| 설치된 torch | 2.12.1+**cu130** (`envs/boltz2.yml`이 Boltz-2 때문에 고정) |
| DGL CUDA 휠이 요구하는 torch | 2.1–2.4 |

한 환경에서 양립 불가다. Boltz-2를 깨지 않으려면 **별도 환경**이 필요하다.

또한 인덱스에 올라 있는 휠이 다 받아지지도 않는다 — `dgl-2.5.0+cu121-cp311`과
`dgl-2.2.1+cu121-cp311`은 HTTP 403이고, **2.4.0+cu121-cp311만 실제로 서빙된다.**

## 만든 환경

```bash
micromamba create -y -n skinscout-rtmscore -c conda-forge \
  python=3.11 rdkit=2025.03 openbabel=3.1.1 prody=2.6.1 \
  mdanalysis numpy=1.26 pandas scipy scikit-learn joblib
PY=~/.local/share/mamba/envs/skinscout-rtmscore/bin/python
$PY -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
$PY -m pip install dgl==2.4.0 -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html
$PY -m pip install dgllife==0.3.2
```

`dgl.graph(([0],[0])).to('cuda')`가 성공한다 — 기존 환경에서
`Device API cuda is not enabled`로 죽던 바로 그 연산이다.

## 업스트림 버그가 하나 더 있었다

환경을 갖춰도 여전히 실패했다:

```
RuntimeError: indices should be either on cpu or on the same device as the indexed tensor (cpu)
  RTMScore/model/model2.py:527 in forward
```

`model2.py:526`이 `C_batch = th.tensor(range(B))`로 **CPU에** 텐서를 만든 뒤 CUDA에 있는
`C_mask`로 인덱싱한다. GPU에서만 터지는 결함이라 아무도 보고한 적이 없다. 한 줄 수정:

```python
C_batch = th.tensor(range(B), device=C_mask.device).unsqueeze(-1).unsqueeze(-1)
```

CPU 실행에서는 `C_mask.device`가 cpu이므로 동작이 완전히 같다.

## 실측 (같은 50개 표적, 같은 AutoDock pose)

| | 시간 | 표적당 | GPU 사용률 |
|---|---:|---:|---:|
| CPU (`cosmax-boltz2`, dgl 1.1.3) | 13분 23초 | 16.1초 | — |
| GPU (`skinscout-rtmscore`, dgl 2.4.0+cu121) | 12분 30초 | 15.0초 | 0–11% |

**개선 6.6%.** 점수는 50/50 전부 **최대 |Δ| = 0.0**으로 완전히 동일하다.

GPU 사용률이 0~11%에 머문다. 병목은 신경망 forward가 아니라 MDAnalysis의 PDB 파싱과
분자 그래프 구성 같은 **CPU 쪽 전처리**다. 그래서 GPU를 붙여도 거의 달라지지 않는다.

## 그래서 어떻게 했나

- **파이프라인에 배선하지 않았다.** `rtmscore_top` 규칙은 그대로 `envs/boltz2.yml`을 쓴다
- `rtmscore/__init__.py`의 device 선택 수정은 **유지한다.** 이건 속도가 아니라 정상 동작
  문제였다 — 이 수정 전에는 CPU 전용 DGL에 CUDA를 강제해 RTMScore가 이 머신에서
  **한 번도 점수를 낸 적이 없었다**
- 업스트림 `model2.py` 패치도 유지한다. 백업은
  `~/.local/opt/RTMScore/RTMScore/model/model2.py.skinscout-backup`
- `skinscout-rtmscore` 환경은 디스크에 남아 있다. 필요 없으면
  `micromamba env remove -n skinscout-rtmscore`

## 정말 빠르게 하려면

GPU가 아니라 전처리를 봐야 한다. 수용체 그래프는 표적마다 한 번만 만들면 되는데
현재는 매 호출마다 다시 만든다. 같은 수용체를 여러 pose로 재채점하는 구조라면
캐싱이 GPU보다 훨씬 큰 효과를 낼 것이다. 측정하지 않은 가설이다.
