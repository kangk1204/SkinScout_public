# 표적 리스트로 한 번에 발굴하기

`scripts/discover_from_targets.py`는 표적(바이오마커·유전자·UniProt) 목록을 받아,
표적마다 측정 기록이 있는 화합물을 꺼내고 그중 화장품 등재 원료(CosIng)와 구조가
같은 것을 표시한다. 사용자가 관심 표적 표를 그대로 주면 되는 입구다.

## 쓰는 법

```bash
python scripts/discover_from_targets.py \
  --targets targets.xlsx \
  --target-column target --category-column category \
  --gene-map gene_uniprot.csv \
  --top 10 --mode balanced \
  --out-dir results/discovery/20260912
```

- 입력: CSV/TSV/XLSX. 열 이름은 자동으로 찾고, 애매하면 `--target-column`,
  `--category-column`, `--direction-column`으로 지정한다.
- 표적 해석 순서: ① UniProt 계정번호면 그대로 ② 저장소 사전(`explore_target`)
  ③ `--gene-map` 파일(gene,uniprot) ④ `--resolve-online`(UniProt REST, 결과는
  `--out-dir/uniprot_cache.json`에 캐시).
- 조회: `workflow/config.yaml`의 `daina_recipe_index_dir`(파이프라인과 같은 생산
  인덱스). 인덱스 밖 표적은 "측정 데이터 없음"으로 리포트에 남는다.
- `--smiles-out candidates.txt`로 상위 후보 SMILES를 저장하면
  `python scripts/run_skinscout.py --smiles '<SMILES>' --mode fast`로 바로 이어갈 수 있다.

## 나가는 것

| 파일 | 내용 |
|---|---|
| `summary.md` | 카테고리별 해석·조회 가능 수, 등재 원료 일치 후보, 다음 단계 |
| `discovery_candidates.csv` | 표적×후보 긴 표 (pActivity·문헌 수·문턱·등재 일치) |
| `target_coverage.csv` | 표적별 해석 여부·인덱스 포함·사유 |
| `unresolved.csv` | 해석 실패/인덱스 밖 목록 |

## 읽을 때 주의

- "알려진"은 ChEMBL·BindingDB에 **측정 기록이 있다**는 뜻이다. `max_pactivity`와
  `threshold`(양성 6.0, 경계 5.0)를 함께 보라. 문턱 아래는 "측정됐다"일 뿐이다.
- 등재 일치(`cosing_match=exact`)는 InChIKey 전체가 같은 경우, `connectivity`는
  연결성(앞 14자)만 같은 경우다. 후자는 입체배치가 다를 수 있다.
- 이 리포트는 실험 전 가설이다. 상위 후보는 안전성/표적 분석을 한 번 더 돌리고,
  assay로 확인하라.
