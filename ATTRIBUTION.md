# 번들 데이터의 출처와 약관

이 저장소의 **코드**는 Apache License 2.0입니다 (`LICENSE`).

`data/` 아래 파일은 사정이 다릅니다. 공개 데이터베이스에서 가져온 것이라 각자의
약관이 그대로 따라오고, 코드 라이선스가 이걸 덮지 못합니다. 무엇이 어디서 왔고 어떤
조건이 붙는지 아래에 적었습니다.

## 성분명 색인 — `data/compound_names/name_index.csv`

이름이나 INCI로 성분을 찾을 때 쓰는 표입니다. 17,765행이고 출처는 섞여 있습니다.

| 출처 | 행 | 라이선스 | 조건 |
|---|---:|---|---|
| ChEMBL | 14,225 | CC BY-SA 3.0 | 출처 표시 + **동일조건변경허락** |
| GtoPdb (IUPHAR/BPS) | 1,460 | ODbL 1.0 / 내용 CC BY-SA 4.0 | 출처 표시 + **동일조건변경허락** |
| BindingDB | 1,087 | CC BY 4.0 | 출처 표시 |
| PubChem | 914 | 퍼블릭 도메인 | 없음 |
| 직접 큐레이션 | 79 | 이 저장소, Apache-2.0 | — |

ChEMBL과 GtoPdb 행이 들어 있어서, **이 파일에서 파생된 것을 배포하실 때는
동일조건변경허락(share-alike)이 따라옵니다.** 상업적으로 쓰는 것 자체는 막히지
않습니다. 다만 파생물을 남에게 배포한다면 같은 조건을 붙여야 합니다.

share-alike가 곤란하시면 `source` 열로 걸러 쓰시면 됩니다. BindingDB와 PubChem
행 2,001개는 출처만 밝히면 됩니다.

## 검증·큐레이션 패널 — `data/validation/`, `data/curation/`

문헌과 ChEMBL에서 뽑아 손으로 확인한 표입니다. 행마다 근거가 붙어 있고, 그 근거가
된 원 데이터의 약관을 따릅니다. 어떤 표적을 왜 골랐는지 하는 큐레이션 판단 자체는 이
저장소 것이라 Apache-2.0입니다.

`data/curation/pubchem_probe_cache*.json`은 PubChem 응답 캐시로 퍼블릭 도메인입니다.

## 구조 패널 — `data/pocket_cold_panel_202608/`, `data/transfer_reachable_panel_202608/`

AlphaFold DB(CC BY 4.0)와 RCSB PDB에서 파생된 포켓·표적 목록입니다.
**AlphaFold DB 재배포에는 출처 표시가 필요합니다:**

> Jumper et al. *Highly accurate protein structure prediction with AlphaFold.*
> Nature 596, 583–589 (2021).
> Varadi et al. *AlphaFold Protein Structure Database.* NAR 50, D439–D444 (2022).

`data/holo_transplant/P14679_with_Cu.pdb`는 RCSB PDB 유래 구조입니다.

## 파이프라인이 내려받아 쓰는 것 (이 저장소에 없음)

설치 시 받아오는 것들이고 여기에 포함돼 있지 않습니다. 각자의 약관을 따릅니다.

| | 라이선스 |
|---|---|
| ChEMBL 37 | CC BY-SA 3.0 |
| BindingDB | CC BY 4.0 (자체 큐레이션분) |
| GtoPdb | ODbL 1.0 / 내용 CC BY-SA 4.0 |
| AlphaFold DB v4 | CC BY 4.0 |
| Human Protein Atlas | CC BY-SA 3.0 |
| CosIng | 유럽집행위원회 공개 자료 |

## 도구 라이선스

`LICENSE_POLICY.md`에 스테이지별로 정리돼 있습니다. GPL·LGPL 의존성(OpenBabel,
GROMACS, xTB, Meeko 등)은 별도 conda 환경으로 격리해 두어 나머지 코드베이스가
permissive로 남습니다.

감작성 예측 3종(HuSSPred · Pred-Skin · STopTox)은 **입력 구조를 외부 연구기관
서버로 보냅니다.** 미공개 성분이면 돌리기 전에 한 번 확인하시는 게 좋습니다. 화면과
README에도 같은 경고를 띄워 뒀습니다.
