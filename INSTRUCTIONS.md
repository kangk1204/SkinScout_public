# Claim-Quality Cosmetic Ingredient Discovery Platform (2026)
## v3 — Compound-First Cosmetic Material Discovery with Skin-Expression-Weighted Targeting and Fail-Closed Downstream Design

**문서 버전**: v3.0 (2026-05-24, v2.0의 확장)
**도메인**: 화장품(cosmetic) 소재 발굴 플랫폼
**대상 시스템**: 16 GB GPU + 32+ core CPU + ≥ 4 TB SSD
**라이선스 정책**: 상업적 사용 가능성을 검토한 도구와 데이터만 claim-quality 경로에 사용
**최종 목표**: (1) 화합물 → 피부 발현 타겟 단백질 후보 제시, (2) bound-pose interaction atom 근거 식별, (3) 향후 안전·합성 가능 analog 생성 후보 제시, (4) ADMET·화장품 적합성·의약품 회피 검증, (5) 향후 합성 경로 제시, (6) computational evidence limits와 구현 차단점을 명시한 **methodology paper 게재**

### v2 → v3 주요 추가

| 영역 | v3 신설/변경 |
|---|---|
| **도메인 specialization** | 일반 약물 → **화장품 소재** 맥락 통합 |
| **Stage 0 확장** | Skin expression DB (HPA + skin.science + GTEx + scRNA-seq), **CosIng INCI DB**, **DrugBank avoidance DB**, **PubMed 기반 skin efficacy KG** (PubTator 3.0) 추가 |
| **Stage 2.5 신설** | Cosmetic ingredient annotation + Drug similarity warning |
| **Stage 3 가중치** | Target ranking에 **skin expression score 가중치** 추가 (HPA + skin proteome 기반) |
| **Stage 5.5 신설** | **Pose-supported interaction atom 식별** (bound Boltz complex ligand atoms에서 PLIP + ProLIF agreement, 2 of 2; original SDF atom index가 아님) |
| **Stage 5.6 설계** | **현재 fail-closed**: REINVENT 4 custom scoring plugin, prior/agent model, 유효한 config, bound-complex↔generator atom map 구현 후 활성화 |
| **Stage 7.5 신설** | **Retrosynthesis route planning** (AiZynthFinder MCTS) |
| **Stage 9 보강** | INCI 매핑, drug 회피 warning, skin efficacy 추론, synthesis tree 시각화, 생성된 analog list |
| **Stage 11 신설** | Publication-ready output (figure auto-gen + config archive + 재현성 패키지) |
| **평가 확장** | Cosmetic ingredient retrospective (retinol → RAR, niacinamide → NNMT 등) + Skin efficacy literature recovery |

---

## 1. 핵심 도구 스택 (v3, 라이선스 검증)

### 1.1 v2 유지 도구
(Boltz-2, BioEmu, GROMACS, ADMET-AI, HuSSPred, STopTox, PSICHIC, AutoDock-GPU, GNINA, P2Rank, Meeko, RTMScore, Mol\*, ZINC22 등 — v2 표 참고)

### 1.2 v3 신규 도구

| 도메인 | 도구 | 버전 | 라이선스 | 역할 |
|---|---|---|---|---|
| **Skin expression DB** | **Human Protein Atlas (HPA)** v25 | 2025 release | CC-BY-SA 3.0 | tissue 단위 발현 (skin sun-exposed/not), cell-type (keratinocyte, fibroblast, melanocyte 등) |
|  | **Skin proteome atlas** (Dyring-Andersen 2020) | skin.science | 학술 무료 | 10,701 proteins cell-type resolved 정량 |
|  | **GTEx v10** | 2024 | open | bulk RNA-seq, skin sun-exposed/not, dbGaP 비공개 부분 제외 |
|  | **Tabula Sapiens / scRNA-seq skin (GSE 시리즈)** | various | open | single-cell, cell-type-specific |
| **Cosmetic DB** | **CosIng (EU)** | rolling | EU 공공 데이터 | INCI name + function + restrictions + SCCS opinion |
|  | (보조) **CIR (Cosmetic Ingredient Review)** | rolling | open (학술) | 미국 화장품 성분 안전성 평가 |
|  | (보조) **PubChem CID → INCI 매핑** | rolling | open | CAS / EINECS 기반 cross-reference |
| **Drug avoidance** | **DrugBank** (open subset) | 5.x | CC-BY-NC-4.0 (비상업) / Pro license (상업) | approved + investigational drug 식별 |
|  | **ChEMBL Drug Indication** | 34 | CC-BY-SA 3.0 | approved drug + indication |
|  | **FDA Orange Book** | rolling | public domain | 미국 승인 의약품 |
| **Literature mining** | **PubTator 3.0** (NLM) | 2024-04 | 무료 API (NLM 정책) | 36M PubMed + 6M PMC, 6 entity + 12 relation |
|  | **OpenTargets Platform** | 24.x | CC-BY-4.0 | gene-disease-drug evidence aggregator |
|  | (옵션) BioREx, BioBERT 자체 호스팅 | - | MIT/Apache-2.0 | 보조 NER |
| **Pharmacophore / interaction** | **PLIP** (Protein-Ligand Interaction Profiler) | 2.4 | GPL-2.0 | 잔기 수준 H-bond, π-stack, salt bridge 검출 |
|  | **ProLIF** | 2.0 | Apache-2.0 | interaction fingerprint |
|  | **Pharmer** | 1.1.3 | GPL-2.0 | 3D pharmacophore 추출/검색 |
|  | **fpocket-pharmacophore** | - | open | pocket 기반 pharmacophore |
| **Analog generation** | **REINVENT 4** ⭐ | 4.x | Apache-2.0 | scaffold hopping, R-group, linker, Mol2Mol RL |
|  | (보조) **STONED-SELFIES** | latest | MIT | SELFIES 기반 string mutation |
|  | (보조) **MolMIM-style local search** | - | MIT-compatible | latent space optimization |
| **Synthesizability** | **AiZynthFinder 4** ⭐ | 4.x | MIT | MCTS retrosynthesis, USPTO trained |
|  | RAscore (Thakkar 2021) | latest | MIT | retrosynthetic accessibility prediction |
|  | SA score (RDKit/Ertl) | RDKit | BSD | synthetic accessibility |
|  | SCScore | latest | MIT | complexity score |
| **Knowledge graph** | NetworkX + Neo4j (Community) | latest | BSD / GPL-3 | skin-target-efficacy KG 구축 |
| **NLP enrichment** | (옵션) BioBERT, PubMedBERT | - | MIT/Apache-2.0 | abstract embedding, 의미 검색 |

**라이선스 주의 (v3 신규)**
- **DrugBank**: 상업 사용 시 별도 라이선스 필요. 비상업/학술이면 CC-BY-NC-4.0 OK. → **상업 배포 시 ChEMBL + FDA Orange Book + RxNorm 등으로 대체** 가능.
- **HPA**: CC-BY-SA 3.0 — share-alike 조항으로 인해 derivative DB도 동일 라이선스로 공유 필요. 산출물 보고서는 OK.
- **CosIng**: EU 공공 데이터, 자유 사용. 단 "informative purpose, no legal value" 주의.
- **PubTator 3.0**: NLM 정책, 학술/상업 모두 사용 가능하나 rate limit 있음 (대규모 mining 시 NLM에 사전 요청 권장).

---

## 2. 파이프라인 전체 아키텍처 (v3)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 0 (ONE-TIME, ~3-5 days, ~300 GB):                                     │
│   • AlphaFold DB human proteome → pLDDT trim → P2Rank pockets → PDBQT      │
│   • ChEMBL / BindingDB 미러                                                  │
│   • ★ NEW: Skin expression DB (HPA + skin.science + GTEx + scRNA-seq)       │
│   • ★ NEW: CosIng INCI database + cosmetic function 분류                     │
│   • ★ NEW: DrugBank/ChEMBL Drug Indication/FDA Orange Book                  │
│   • ★ NEW: PubMed 기반 skin efficacy KG (PubTator 3.0 query → Neo4j)         │
│   • MMseqs2 sequence DB + Foldseek (leakage 체크)                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ INPUT: cosmetic candidate compound (SMILES / SDF / MOL)                     │
└──────────────┬──────────────────────────────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────┐
│ STAGE 1: Compound Preprocessing                     │
└──────────────┬──────────────────────────────────────┘
               ▼
┌────────────────────────────────────────────────────┐
│ STAGE 2: ADMET Gate (HARD FILTER)                   │
│   • Skin sens 3-model consensus → PASS/FLAG/HALT    │
└──────────────┬──────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ★ STAGE 2.5 (NEW): Cosmetic Annotation + Drug Avoidance Check               │
│   • CosIng INCI 매칭: Tanimoto ≥ 0.85 or 동일 → annotation                   │
│     "이미 알려진 화장품 성분: [INCI name], function: [emollient/...]"          │
│   • DrugBank/Orange Book 매칭: Tanimoto ≥ 0.85 → WARNING_DRUG_LIKE           │
│     scaffold-level Bemis-Murcko 동일  STRICT_WARNING                       │
│   • 의약품 회피 정책에 따라 HALT / DOWNWEIGHT / 통과                          │
└──────────────┬──────────────────────────────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ★ STAGE 3 (v3): Skin-Expression-Weighted Target Identification              │
│                                                                              │
│  MODE-COMPREHENSIVE (default) | MODE-FAST (option)                          │
│    [v2 dual-mode 그대로]                                                      │
│                                                                              │
│  ★ NEW: Skin Expression Weighting                                            │
│    Final score = 0.7 × RRF(docking_consensus) + 0.3 × SkinScore             │
│    SkinScore = α·HPA_tissue_level + β·HPA_cell_type_level                   │
│              + γ·SkinProteome_abundance + δ·Skin_relevance_KG                │
│    (cell-type level: keratinocyte, fibroblast, melanocyte, immune, endothelial)│
│                                                                              │
│  ★ NEW: Skin-Efficacy KG Cross-reference                                     │
│    각 후보 target에 대해 PubMed-mined efficacy 라벨 추가                       │
│    "TYR → 미백 (whitening)", "MMP1 → 주름/광노화 (anti-wrinkle)",             │
│    "FLG → 보습/아토피", "RARα/β/γ → anti-aging/acne" 등                       │
│                                                                              │
│  출력: ranked_targets_with_skin_priority.csv                                  │
└──────────────┬──────────────────────────────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────┐
│ STAGE 4: Target Structure Preparation (top 30-50)   │
└──────────────┬──────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────┐
│ STAGE 5: Boltz-2 Co-folding + Affinity              │
└──────────────┬──────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ★ STAGE 5.5 (NEW): Bound-pose interaction atom 식별                         │
│   • PLIP: H-bond, π-stack, salt bridge, hydrophobic 잔기 단위 검출            │
│   • ProLIF interaction fingerprint                                           │
│   • PLIP + ProLIF가 같은 bound Boltz complex ligand atom을 지목한 경우만     │
│     pose-supported interaction atom으로 기록                                  │
│   • causal pharmacophore validation 또는 experimental efficacy로 해석하지 않음 │
│   출력: PLIP/ProLIF source evidence + agreement summary                      │
└──────────────┬──────────────────────────────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ★ STAGE 5.6: REINVENT 4 integration gate                                    │
│   • 현재 claim-capable 실행은 fail-closed                                    │
│   • custom scoring plugin + prior/agent + upstream-compatible config 필요    │
│   • bound-complex ligand atom order를 generator 표현으로 매핑하는 계약 필요   │
│   • diagnostic mode는 blocked status + empty SMILES만 기록                   │
└──────────────┬──────────────────────────────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────┐
│ STAGE 6: BioEmu Ensemble (top 5-10 candidates)       │
└──────────────┬──────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────┐
│ STAGE 7: MD Refinement (fail-closed until valid complex setup is implemented) │
└──────────────┬──────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ★ STAGE 7.5 (NEW): Retrosynthesis Route Planning (AiZynthFinder)             │
│   • MCTS, USPTO-trained policy network                                       │
│   • Stock: eMolecules + ZINC commercial (옵션)                               │
│   • 최대 depth 6, time limit 120 s/molecule                                  │
│   • 출: top-3 routes per analog, step count, stock fraction                │
│   • RAscore, SA score, SCScore 동시 계산                                     │
│   • 합성 불가 (no route found) 인 후보는 자동 deprioritize                    │
└──────────────┬──────────────────────────────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────┐
│ STAGE 8: QM / QM-MM (final 1-3 candidates)           │
└──────────────┬──────────────────────────────────────┘
               ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ ★ STAGE 9 (v3): Cosmetic Discovery Report (Mol*)                             │
│   Panel A: ranked targets w/ skin expression + efficacy KG                  │
│   Panel B: 3D complex + PLIP/ProLIF pose-interaction atom 표시               │
│   Panel C: ADMET radar + skin-sens 3-consensus + cosmetic suitability       │
│   ★ Panel D: INCI annotation + Drug avoidance warning                        │
│   ★ Panel E: Stage 5.6 blocked status 또는 향후 validated analog table        │
│   Panel F: MD/system-preparation status and fail-closed blockers              │
│   ★ Panel G: Synthesis tree (AiZynthFinder SVG) + step list                  │
│   ★ Panel H: Skin efficacy 추론 ("이 화합물이 결합하는 X target은            │
│              PubMed 보고서에 따라 [acne/wrinkle/whitening] 관련")             │
│   Panel I: QM interaction decomposition                                      │
└──────────────┬──────────────────────────────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────┐
│ STAGE 10: (Optional) ZINC22 Forward Expansion        │
└──────────────┬──────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ★ STAGE 11 (NEW): Publication-Ready Output                                   │
│   • Methodology figure 자동 생성 (matplotlib + Mol* PNG export)              │
│   • Data availability statement template (each tool's dataset URL)          │
│   • Reproducibility package: tool versions, random seeds, config hash       │
│   • Supplementary: benchmark결과, leakage report, cold-start performance    │
│   • LaTeX/Markdown 초안 (Introduction, Methods, Results) 자동 생성 옵션      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Stage 0 (v3) — Infrastructure 확장

v2의 인프라에 4가지 새 DB를 추가합니다.

### 3.1 Skin Expression Database 구축

**(a) Human Protein Atlas 다운로드**
```bash
# HPA full data (XML / TSV)
wget https://www.proteinatlas.org/download/proteinatlas.tsv.zip
wget https://www.proteinatlas.org/download/rna_tissue_consensus.tsv.zip
wget https://www.proteinatlas.org/download/normal_tissue.tsv.zip
wget https://www.proteinatlas.org/download/rna_single_cell_type.tsv.zip
unzip *.zip -d data/hpa/
```

**(b) Skin-relevant tissue / cell types 필터링**
- Tissue: skin (sun-exposed, not sun-exposed), 부속 기관 (hair follicle 추가 분석 시).
- Cell type: keratinocyte (basal, suprabasal), fibroblast, melanocyte, Langerhans cell, endothelial.
- HPA TSV → pandas → 각 UniProt에 대해:
  - `skin_tissue_score`: nTPM in skin tissue.
  - `cell_type_specificity`: 어떤 skin cell에 specific한지.
  - `skin_enriched_flag`: tissue-enriched/group-enriched 여부.

**(c) Skin proteome (Dyring-Andersen 2020)**
- 10,701 단백질의 LFQ 정량값, skin layer + cell-type resolved.
- TSV 다운로드, UniProt ID 키 정렬, layer/cell별 abundance 정규화.

**(d) GTEx v10 skin RNA-seq**
- bulk-level, 두 가지 skin sample (sun-exposed lower leg / not sun-exposed suprapubic).
- TPM 값으로 cross-check.

**(e) scRNA-seq skin atlas (옵션)**
- GSE130973 (healthy keratinocyte), GSE150672 (fibroblast spatial), GSE173706 (melanocyte) 등.
- Tabula Sapiens skin compartment.
- Seurat / Scanpy 처리 → cell-type별 marker gene expression matrix.

**(f) 통합 SkinScore 계산**
```python
# scripts/stage0_skin_score.py
def skin_score(uniprot_id, hpa, skin_proteome, gtex, sc_data):
    s_hpa_tissue   = hpa.get(uniprot_id, 'skin_nTPM_log')         # weight α=0.30
    s_hpa_cell     = hpa.get(uniprot_id, 'best_skin_celltype_nTPM') # β=0.25
    s_proteome     = skin_proteome.get(uniprot_id, 'log_LFQ')      # γ=0.20
    s_gtex         = gtex.get(uniprot_id, 'skin_log_TPM')          # ε=0.10
    s_sc           = sc_data.get(uniprot_id, 'max_cell_type_score') # ζ=0.15
    return weighted_sum(...)  # normalize to [0, 1]
```
출력: `data/skin_expression/skin_score.tsv` (UniProt | gene | SkinScore | cell_type_preferred | tier(very_high/high/medium/low/very_low))

### 3.2 CosIng + Cosmetic Function Database

```bash
# EU CosIng - data.europa.eu에서 CSV 다운로드
wget https://data.europa.eu/data/datasets/cosmetic-ingredient-database-... -O cosing.zip
unzip cosing.zip -d data/cosing/
```
- 컬럼: INCI name, CAS, EINECS, function (Antioxidant, Humectant, Skin Conditioning - Emollient, etc.), restriction.
- **Compound 매칭 전략**:
  - CAS / EINECS lookup → 직접 매칭.
  - INCI → PubChem CID resolver (PUG REST API) → SMILES 변환.
  - Fingerprint 사전 계산: ECFP4, ECFP6 → CosIng 전체 fingerprint DB.

### 3.3 Drug Avoidance Database

- DrugBank approved + investigational (~14,000) — 상업 라이선스 시 ChEMBL drug subset + Orange Book + 위키데이터로 대체 가능.
- ChEMBL37 phase-4 (approved) 약 4,000개 SMILES.
- FDA Orange Book SMILES 매핑 (DrugCentral / PubChem 활용).
- 사전 계산: ECFP4 fingerprint, Bemis-Murcko scaffold.

### 3.4 PubMed 기반 Skin Efficacy Knowledge Graph

이 부분이 v3에서 가장 새로 정보 자산입니다. 사용자 요구사항 9번 그대로 구현.

**키워드 set (한·영 혼합)**

| 효능 카테고리 | 영어 키워드 | 한국어 키워드 | 관련 타겟 (예시) |
|---|---|---|---|
| 미백 / Hyperpigmentation | tyrosinase inhibitor, melanogenesis, depigmenting, skin lightening | 미백, 색소침착, 멜라닌 | TYR, TYRP1, DCT, MITF, MC1R |
| 항노화 / Anti-aging | anti-wrinkle, photoaging, collagen, elastin, MMP inhibitor | 주름, 항노화, 광노화, 콜라겐 | MMP1, MMP3, MMP9, ELN, COL1A1, SIRT1 |
| 보습 / Hydration | moisturization, aquaporin, hyaluronic acid synthase, filaggrin | 보습, 히알루론산, 필라그린 | AQP3, HAS1/2/3, FLG, LOR |
| 여드름 / Acne | acne, sebum, sebocyte, Cutibacterium acnes | 여드름, 피지 | SREBP1, PPARG, IGF1R, AR, SRD5A1, TLR2 |
| 아토피 / AD | atopic dermatitis, barrier function, Th2, IL-13, IL-31 | 아토피, 피부장벽 | FLG, TSLP, IL4R, IL13, IL31, JAK1/2 |
| 항염증 / Anti-inflammatory | NF-κB, COX-2, TLR4, inflammasome | 항염증 | NFKB1, PTGS2, TLR4, NLRP3 |
| 항산화 / Antioxidant | NRF2, KEAP1, oxidative stress | 항산화 | NFE2L2, KEAP1, SOD1, CAT, GPX1 |
| 광노화 / Photoaging | UV-induced, AhR, photodamage | 광노화, 자외선 | AHR, NRF2, FOXO3, MMP1 |
| 비듬·지루성 | seborrheic, Malassezia | 비듬 | LCN2, S100A8/9 |
| 발모 / Hair growth | hair follicle, Wnt/β-catenin, DKK1 | 발모, 모발 | CTNNB1, WNT3A, DKK1, AR, SRD5A2 |
| 레티놀 관련 | retinoid, RAR, RXR | 레티놀, 비타민 A | RARA, RARB, RARG, RXRA, RXRB, ALDH1A2 |

**구축 절차**
```python
# scripts/stage0_skin_kg.py 개요
import requests

PUBTATOR_API = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api"

for category, kw_set in KEYWORDS.items():
    for kw in kw_set:
        # 1) PubTator 3.0 search
        r = requests.get(f"{PUBTATOR_API}/search/?text={kw}&format=json")
        pmids = extract_pmids(r.json())
        
        # 2) annotation 가져오기 (gene/disease/chemical)
        for pmid_batch in chunked(pmids, 100):
            ann = requests.get(
                f"{PUBTATOR_API}/publications/export/biocjson",
                params={"pmids": ",".join(pmid_batch)}
            )
            for paper in ann.json()['PubTator3']:
                genes = extract_entities(paper, "Gene")
                diseases = extract_entities(paper, "Disease")
                chemicals = extract_entities(paper, "Chemical")
                # 3) co-occurrence + relation extraction
                for gene_id in genes:
                    add_edge(graph, 
                             source=gene_id, 
                             target=category,
                             evidence_pmid=paper['pmid'],
                             relation=infer_relation(paper))
        
        # 4) (옵션) BioBERT/PubMedBERT으로 abstract embedding
        #    → 의미 유사도 기반 추가 paper 회수

# 결과를 Neo4j에 저장
save_to_neo4j(graph, "data/skin_efficacy_kg/")
```

**KG 스키마 (Neo4j)**
```
(:Gene {uniprot, gene_symbol, skin_score})
  -[:ASSOCIATED_WITH {pmid_list, n_papers, score}]->
(:EfficacyCategory {name, korean, type})

(:Compound {smiles, inchikey, inci_name})
  -[:USED_FOR {pmid_list}]->
(:EfficacyCategory)

(:Gene)
  -[:KNOWN_TARGET_OF]->
(:Compound)
```

→ 후속 Stage 3에서 cross-reference: 후보 target → 어떤 efficacy category에 강하게 묶여 있는지 자동 라벨.

### 3.5 검증 체크리스트 (v3 확장)
- [ ] HPA 다운로드 OK, skin tissue 매핑 OK
- [ ] CosIng SMILES 변환률 ≥ 70% (일부는 polymer/extract라 변환 불가)
- [ ] DrugBank approved + ChEMBL phase-4 dedup 완료
- [ ] PubTator 3.0 API 응답 OK, KG node ≥ 5,000 / edge ≥ 50,000
- [ ] SkinScore 계산값 분포 sanity (TYR, FLG, MMP1 등 known skin 단백질이 상위에 위치)
- [ ] `stage0_v3_complete.flag` 생성

---

## 4. Stage 1 — Compound Preprocessing
(v2와 동일)

---

## 5. Stage 2 — ADMET 1차 게이트
(v2와 동일, 화장품 맥락에서 **skin sensitization** strict 정책 지)

---

## 6. Stage 2.5 (v3 신설) — Cosmetic Annotation + Drug Avoidance

이 단계가 사용자 요구사항 5번과 6번에 해당합니다.

### 6.1 CosIng INCI 매칭

```python
def cosing_match(input_inchikey, input_ecfp4, input_cas=None):
    # Layer 1: 직접 매칭 (CAS / InChIKey)
    direct = cosing_db.lookup(cas=input_cas) or cosing_db.lookup(inchikey=input_inchikey)
    if direct:
        return MatchResult(level="EXACT", inci=direct.inci_name, functions=direct.functions)
    
    # Layer 2: Tanimoto 유사도
    sims = []
    for cosing_entry in cosing_db.all_with_smiles():
        sim = tanimoto(input_ecfp4, cosing_entry.ecfp4)
        sims.append((sim, cosing_entry))
    sims.sort(reverse=True)
    
    if sims[0][0] >= 0.85:
        return MatchResult(level="SIMILAR", inci=sims[0][1].inci_name, 
                           functions=sims[0][1].functions, tanimoto=sims[0][0])
    elif sims[0][0] >= 0.65:
        return MatchResult(level="ANALOG", inci=sims[0][1].inci_name, 
                           functions=sims[0][1].functions, tanimoto=sims[0][0])
    return MatchResult(level="NEW")
```

화면 표시:
- EXACT: "**기존 화장품 성분과 동일**: INCI = [Niacinamide], function = Skin Conditioning, Smoothing"
- SIMILAR: "기존 화장품 성분과 매우 유사 (Tanimoto=0.92): INCI = [Retinol]"
- ANALOG: "기존 성분의 analog (Tanimoto=0.71)"
- NEW: "신규 chemical space"

### 6.2 Drug Avoidance

```python
def drug_avoidance(input_ecfp4, input_scaffold):
    warnings = []
    
    # Layer 1: approved drug Tanimoto
    for drug in drug_db.approved:
        if tanimoto(input_ecfp4, drug.ecfp4) >= 0.85:
            warnings.append(("STRICT_WARNING", drug.name, drug.tanimoto))
    
    # Layer 2: scaffold match (Bemis-Murcko)
    if input_scaffold in drug_db.scaffolds_approved:
        warnings.append(("SCAFFOLD_MATCH", drug_db.scaffolds_approved[input_scaffold]))
    
    # Layer 3: 약한 유사 (analog) — 경고만
    for drug in drug_db.approved:
        if 0.65 <= tanimoto(input_ecfp4, drug.ecfp4) < 0.85:
            warnings.append(("SOFT_WARNING", drug.name))
    
    return warnings
```

**정책 옵션**:
- `--drug-policy=strict`: STRICT_WARNING 시 HALT (의약품 회피 엄격)
- `--drug-policy=moderate` (권장 default): WARNING 라벨링만, 진행
- `--drug-policy=lenient`: 경고 표시 없이 진행 (의약품 재포지셔닝 의도일 때)

산출물: `02b_cosmetic_drug/cosing_match.json`, `drug_warnings.json`

### 6.3 의미

이 단계는 단순 필터가 아닙니다:
- **EXACT/SIMILAR INCI 매칭이 나오면** → 이 화합물은 새로운 발견이 아니라 기존 성분의 새 target 발견 (repositioning 측면에서 가치). 논문에서는 "we found a new mode of action for the existing cosmetic ingredient niacinamide" 같은 framing.
- **NEW**이면서 drug 유사도 낮으면** → 가장 가치 있는 신소재 후보.
- **Drug scaffold match가 나오면** → 의약품 영역에 들어갈 위험. 현재는 경고와
  우선순위 조정까지만 claim 가능하며, 향후 Stage 5.6 계약 구현 후 scaffold
  hopping을 검토한다.

---

## 7. Stage 3 (v3) — Skin-Expression-Weighted Target Identification

### 7.1 기본 흐름은 v2 dual-mode 유지

MODE-COMPREHENSIVE는 AutoDock-GPU claim path이며, 선택된 모든 receptor에
대해 precomputed AutoGrid4 `.maps.fld` / `.fld` coverage가 있어야 한다.
이 repo에는 AutoGrid map 생성 stage/dependency가 없으므로 coverage가 없으면
claim-quality 실행은 fail-closed 처리한다. MODE-FAST의 explicit Vina 경로는
degraded diagnostic이며 target/report claim 근거로 사용하지 않는다.

### 7.2 ★ 새로운 가중치 융합

```python
def final_target_score(uniprot):
    docking_rrf = rrf_consensus(autodock_gpu, gnina_cnn, rtmscore, boltz2_aff)
    skin_score = skin_db.score(uniprot)            # [0, 1]
    
    # 사용자 시나리오에 따라 가중치 조정 가능
    # 기본: 결합 70%, 발현 30%
    final = 0.7 * normalize(docking_rrf) + 0.3 * skin_score
    
    # 단, skin_score=0 (피부에 거의 발현 안 됨) target는 hard-cap:
    if skin_score < 0.05:
        final *= 0.3  # heavy penalty
    
    return final
```

가중치는 config로 조절 가능. 화장품 맥락이 아닌 일반 약물 발굴이라면 0.7/0.3 → 1.0/0.0로.

### 7.3 ★ Skin Efficacy KG cross-reference

각 top target에 대해:
```python
def efficacy_label(uniprot):
    edges = skin_kg.query(f"MATCH (g:Gene {{uniprot:'{uniprot}'}})-[r:ASSOCIATED_WITH]->(e) RETURN e, r")
    if not edges:
        return "no_known_skin_efficacy"
    return sorted(edges, key=lambda x: x.r.n_papers, reverse=True)[:3]
    # 예: [("wrinkle", 142 papers), ("photoaging", 89), ("collagen", 67)]
```

리포트에 표시:
> "Target TYR (Tyrosinase): SkinScore=0.94 (very high, melanocyte-specific), 
>  Predicted efficacy from PubMed KG: **whitening (562 papers)**, melanoma (89 papers)."

### 7.4 산출물

`03_targets/ranked_targets_v3.csv` 컬럼:
| uniprot | gene | docking_rrf | psichic | skin_score | skin_tier | efficacy_top1 | efficacy_top2 | final_score |

---

## 8. Stage 4–5
(v2 유지)

---

## 9. Stage 5.5 (v3 신설) — Pose-supported interaction atoms

사용자 요구사항 2번 구현.

### 9.1 PLIP + ProLIF agreement

**(a) PLIP** (Protein-Ligand Interaction Profiler)
- 입력: Boltz-2 top 복합체 PDB.
- 출력: hydrogen bonds (donor/acceptor 잔기-원자 pair), π-stacking, π-cation, salt bridge, hydrophobic contact, halogen bond, water bridge.

**(b) ProLIF** (Interaction fingerprint)
- bit fingerprint 또는 count fingerprint 형태로 정량화.
- 현재 검증된 executable evidence는 bound Boltz complex pose의 ligand atom에
  대한 PLIP + ProLIF agreement이다.

→ **2 of 2 agreement 평가**: PLIP와 ProLIF가 같은 ligand atom을 지목하면
"pose-supported interaction atom"으로 기록한다. 인덱스 좌표계는 각 Boltz
complex의 ligand HETATM/AtomGroup 순서에 대응하는 0-based index다. original
xTB SDF 또는 REINVENT representation의 atom index가 아니며 causal
pharmacophore validation 또는 experimental efficacy evidence로 해석하지 않는다.

### 9.2 Functional group 라벨링 (향후 구현 조건)

아래 SMARTS 기반 라벨링은 설계 후보이며 현재 executable DAG에서 제거했다.
Bound-complex ligand atom과 original SDF atom 사이의 target-specific,
검증 가능한 mapping sidecar가 먼저 구현되어야 한다.

| 작용기 | SMARTS | 화장품 맥락 |
|---|---|---|
| Phenol (페놀) | `c1ccc(O)cc1` | 항산화 (resveratrol, EGCG) |
| Catechol | `c1cc(O)c(O)cc1` | 항산화 (caffeic acid) |
| Carboxylic acid | `C(=O)[OH]` | AHA (acne, exfoliation) |
| Hydroxy acid | `C(O)C(=O)[OH]` | AHA, BHA |
| Retinoid skeleton | `CC1=CCCCC1` (β-ionone ring) + polyene | RAR/RXR binding |
| Niacinamide / amide | `C(=O)N` | NAD+ precursor |
| Vitamin C scaffold | (specific SMARTS) | collagen synthesis |
| Polyethylene glycol | `OCCO` 반복 | 보습 (humectant) |
| Glycerol-like | `OCC(O)CO` | humectant |
| Quaternary ammonium | `[N+](C)(C)C` | conditioning agent (특히 hair) |

→ mapping 검증 후에만 입력 화합물 작용기와 pose-supported atom의 겹침을 분석한다.

### 9.3 3D Pharmacophore extraction (향후 구현 조건)

- 현재 Pharmer query 생성은 executable DAG에 포함하지 않는다.
- 향후 bound pose 자체에서 feature를 추출하고 target/pose/atom-map provenance를
  함께 기록하는 구현이 필요하다.
- original SDF 좌표에 bound-pose atom index를 직접 적용하는 방식은 금지한다.

### 9.4 산출물

현재 문서에서 claim 가능한 산출물은 PLIP source evidence, ProLIF source
evidence, 그리고 두 source가 같은 bound-pose ligand atom에 동의한 agreement
summary이다. PLIP/ProLIF 외 atom-evidence source와 multi-source consensus
산출물은 현재 verified architecture의 claim 근거로 문서화하지 않는다.

리포트 허용 표현:
> "PLIP와 ProLIF가 bound complex ligand atom indices 4-7을 interaction
>  support로 함께 지목했다. 이 인덱스는 해당 complex 내부 좌표계이며 작용기,
>  causal pharmacophore 또는 효능 검증을 의미하지 않는다."

---

## 10. Stage 5.6 — REINVENT 4 integration gate

현재 구현은 claim-capable analog generation을 수행하지 않는다. 이전 TOML은
REINVENT 4 upstream schema/CLI와 맞지 않았고, 기재된 7개 endpoint도 built-in
component가 아니며 repo-local custom plugin이 없었다. 따라서 잘못된 reward로
분자를 생성하는 대신 기본 실행을 중단한다.

### 10.1 활성화 전 필수 계약

1. REINVENT 4 버전에 고정된 staged-learning config와 CLI invocation.
2. `reinvent_plugins/components/comp_*` 또는 검증된 ExternalProcess/REST scoring.
3. 재현 가능한 prior/agent model artifact와 checksum provenance.
4. PLIP/ProLIF bound-complex atom order를 generator representation으로 변환하는
   target-specific atom-map sidecar와 graph/isomorphism 검증.
5. ADMET, affinity, drug/cosmetic similarity, synthesis score 각각의 입력·출력
   schema, 방향성, 결측치 및 실패 정책.
6. 생성 결과의 비어 있음, 중복, invalid SMILES, lineage, 모델/seed provenance를
   검증하는 회귀 및 통합 테스트.

### 10.2 현재 산출물

기본 실행은 stale output을 제거한 뒤 실패한다. 명시적 diagnostic mode만 아래를
기록하며 downstream funnel은 empty SMILES를 다시 거부한다.

```
05_6_analogs/
├── reinvent4_contract_status.json     # blocked, claim_eligible=false
└── all_generated.smi                  # empty diagnostic artifact
```

---

## 11. Stage 6 — BioEmu Ensemble (top 5-10 analogs)
(v2와 동일)

---

## 12. Stage 7 — MD Refinement (fail-closed)

현재 verified architecture는 MD end-to-end readiness를 claim하지 않는다. Stage 7은
scientifically valid complex topology/system preparation이 구현될 때까지
fail-closed이다. 필요한 누락 항목:

- protein + ligand topology merge
- solvation
- ion placement / neutralization
- minimization
- restrained equilibration

이 준비가 구현되고 검증되기 전까지 MD stability, MM-GBSA, 또는 downstream
physics evidence는 target/report claim 근거로 사용하지 않는다.

---

## 13. Stage 7.5 (v3 신설) — Retrosynthesis (AiZynthFinder)

사용자 요구사항 7번 구현.

### 13.1 실행

```bash
# 각 top analog에 대해
aizynthcli --config configs/aizynth.yml \
           --smiles "{ANALOG_SMILES}" \
           --output retro/{analog_id}.json
```

**aizynth.yml 핵심 설정**
```yaml
policy:
  files:
    uspto: data/aizynth/uspto_model.onnx       # MIT, USPTO 공개 데이터 학습
filter:
  files:
    uspto_filter: data/aizynth/uspto_filter_model.onnx
stock:
  files:
    zinc: data/aizynth/zinc_stock.hdf5         # commercial small molecule stock
    emolecules: data/aizynth/emolecules_stock.hdf5  # (옵션, 별도 다운로드)
properties:
  iteration_limit: 100
  return_first: false
  time_limit: 120
  max_transforms: 6
```

### 13.2 평가 메트릭 자동 계산

각 후보에 대해:
- `n_routes_found`: 발견된 경로 수.
- `min_steps`: 최단 경로의 step 수.
- `best_route_score`: AiZynthFinder 종합 score.
- `stock_fraction`: 시판 가능한 출발물질의 비율.
- `RAscore`, `SA_score`, `SCScore`.

**의사결정**:
- `n_routes_found == 0` → 합성 불가, 제외 또는 deprioritize.
- `min_steps > 8` → 비용/시간 부담, downweight.
- `stock_fraction < 0.5` → 일부 출발물질 미가용, 검토.

### 13.3 산출물

```
07_5_retrosynthesis/
├── {analog_id}/
│   ├── routes.json
│   ├── tree.svg                       # 시각화
│   ├── route_summary.csv
│   └── reagent_list.csv
└── synthesis_priority_ranking.csv     # 모든 analog 통합 랭킹
```

리포트:
> "Top analog #3 (SMILES: ...) 합성 경로: **4단계**, 시판 출발물질 비율 100%, 
> 핵심 단계: Mitsunobu coupling → Suzuki → deprotection → recrystallization. 
> RAscore = 0.94, SA score = 2.3 (rotinely synthesizable)."

---

## 14. Stage 8 — QM / QM-MM (final 1-3)
(v2와 동일)

---

## 15. Stage 9 (v3) — Cosmetic Discovery Report (Mol*)

v2의 패널에 4개를 추가합니다 (Panel D, E, G, H).

### 15.1 추가 패널 상세

**Panel D — INCI Annotation + Drug Avoidance**
- 입력 화합물의 CosIng 매칭 상태 (EXACT/SIMILAR/ANALOG/NEW) 배지.
- 매칭된 INCI 이름과 functions 표시 (예: "Niacinamide — Skin Conditioning, Smoothing").
- DrugBank 매칭 warning: red/yellow/green badge + 가장 유사한 drug 이름 + Tanimoto.

**Panel E — Generated Analogs Table**
- 현재는 Stage 5.6 contract status를 표시하고 analog 성공을 주장하지 않는다.
- Stage 5.6 계약 구현 후에만 top analog 구조와 검증된 속성을 표시한다.

**Panel G — Synthesis Tree**
- AiZynthFinder SVG export embed.
- Step-by-step text description (자동 생성).
- 출발물질 → CAS/공급자 링크 (옵션, eMolecules/Sigma-Aldrich).

**Panel H — Skin Efficacy Inference**
- Top target에 대한 PubMed KG cross-reference 결과.
- 예: "이 화합물의 주요 결합 target은 **Tyrosinase (TYR)**이며, PubMed mining 결과 562건의 논문이 TYR을 **미백/whitening** 효능과 연결합니다. 보조 target MITF (89 papers)도 같은 경로."
- 단순 keyword 매칭이 아닌 PubTator 3.0 relation extraction 활용.

### 15.2 단일 HTML 파일 구성

Mol\* + Vega-Lite (chart) + Plotly (3D) + RDKit Web (2D structure) 통합 — 모두 client-side rendering, offline 재현 가능.

---

## 16. Stage 10 — ZINC22 Forward Expansion
(v2와 동일, **단 화장품 도메인에서는 ZINC22 대신 또는 함께 cosmetic-specific compound library 사용 가능**)
- 보조 라이브러리: Cosmetic Ingredient Database SMILES + natural product DB (COCONUT, NPAtlas) — 천연물 유래 화장품 소재 발굴 시.

---

## 17. Stage 11 (v3 신설) — Publication-Ready Output

사용자 요구사항 10번 구현.

### 17.1 자동 figure 생성

스크립트 `stage11_make_figures.py`:

| Figure | 내용 |
|---|---|
| Fig 1 | Workflow schematic (이 문서 Section 2의 다이어그램을 SVG로 export) |
| Fig 2 | Stage 3 결과: top 50 target의 docking score vs PSICHIC + skin expression heatmap |
| Fig 3 | Stage 5.5 pose-supported interaction atoms: PLIP/ProLIF agreement over bound-pose ligand atoms |
| Fig 4 | Stage 5.6 status; validated analog artifact가 없으면 non-placeholder package 차단 |
| Fig 5 | Stage 7 fail-closed MD/system-preparation status; no MD stability claim until valid complex setup exists |
| Fig 6 | Stage 7.5: 최종 top-3 analog의 retrosynthesis tree |
| Fig 7 | Evaluation: PoseBusters / PLINDER / cold-start / DTI vs Docking disagreement bar charts |
| Fig 8 | Case study: known cosmetic ingredients (retinol, niacinamide, ...)에서 우리 파이프라인이 known target을 recover하는 능력 |

### 17.2 데이터 가용성 statement (template)

```
DATA AVAILABILITY
- All training data sources are publicly available:
  * AlphaFold DB human proteome v4: https://alphafold.ebi.ac.uk (CC-BY-4.0)
  * Human Protein Atlas v25: https://www.proteinatlas.org (CC-BY-SA 3.0)
  * CosIng database: https://ec.europa.eu/growth/tools-databases/cosing
  * ChEMBL37, BindingDB, ZINC22, PoseBusters benchmark v2
- All software is open-source and listed in Supplementary Table S1
- The skin-efficacy knowledge graph constructed in this study is released at
  Zenodo (DOI: 10.5281/zenodo.XXXXXXX) under CC-BY-4.0
- Pipeline code: github.com/{user}/cosmetic-discovery-pipeline (MIT)
```

### 17.3 재현성 패키지

`results/runs/{run_id}/reproducibility/`:
- `tool_versions.lock` (모든 도구 정확한 버전)
- `config_hash.txt` (입력 설정의 SHA256)
- `random_seeds.json`
- `git_commit.txt`
- `runtime_log.txt` (GPU 모델, CPU, OS, 소요 시간)
- `input_compound.sdf` + `input_compound.fingerprint`

### 17.4 (옵션) LaTeX/Markdown 초안 자동 생성

각 figure에 대한 caption + Methods 섹션 boilerplate + bibliography 항목을 자동 생성. 사용자가 검토 후 수정.

```
results/runs/{run_id}/manuscript_draft/
├── 00_abstract.md       # 자동 placeholder
├── 01_introduction.md   # boilerplate
├── 02_methods.md        # 자동 생성 (stage별 method 텍스트)
├── 03_results.md        # 결과 표/figure 참조 자동 삽입
├── 04_discussion.md     # placeholder
├── 05_references.bib    # 자동 BibTeX
└── figures/
```

---

## 18. 평가 프로토콜 (v3) — Cosmetic 특화 추가

### 18.1 v2 평가 그대로 유지
- PoseBusters Benchmark v2 (post-2023-10)
- PLINDER-PL50
- Cold-start protein 평가
- DTI vs Docking disagreement 분석
- Leakage 자동 점검

### 18.2 ★ v3 신규: Cosmetic Ingredient Retrospective Evaluation

**목적**: 우리 파이프라인이 알려진 화장품 성분의 알려진 target을 recover하는가?

**테스트 케이스 (예시)**

| Cosmetic ingredient (INCI) | Known target | Known efficacy | 우리 파이프라인 recovery 여부 |
|---|---|---|---|
| **Retinol** | RAR α/β/γ, RXR α/β/γ | anti-wrinkle, photoaging, acne | top-K? |
| **Niacinamide** | NNMT, SIRT1, PARP | hyperpigmentation, redness | top-K? |
| **Ascorbic acid** | various (collagen prolyl hydroxylase 등) | antioxidant, collagen synthesis | top-K? |
| **α-Arbutin** | TYR | whitening | top-K? |
| **Kojic acid** | TYR | whitening | top-K? |
| **Salicylic acid** | (다중) | BHA, exfoliation, anti-acne | top-K? |
| **Glycolic acid** | (다중) | AHA, exfoliation | top-K? |
| **Hyaluronic acid (small MW)** | CD44, HAS receptors | hydration | challenging — polymer |
| **Resveratrol** | SIRT1, NRF2 | antioxidant, anti-aging | top-K? |
| **EGCG (Epigallocatechin gallate)** | MMP9, NF-κB, TYR | antioxidant, anti-aging | top-K? |
| **Bakuchiol** | RAR-like (retinol-mimetic) | anti-wrinkle (retinoid alternative) | challenging |
| **Caffeine** | adenosine receptors, PDE | anti-cellulite, eye-bag | top-K? |
| **Adenosine** | A1/A2A/A2B/A3 receptors | anti-wrinkle | top-K? |
| **Centella asiatica components (asiaticoside)** | TGF-β, collagen | wound healing, anti-aging | challenging — complex |

→ 각 ingredient에 대해 Top-1, Top-5, Top-10 recovery 계산. 보고서에 표로.

**중요**: 일부 성분(retinol 등)은 Boltz-2 학습 데이터에 포함되어 있을 수 있음 — **leakage check 필수** (Sec. 14.3 v2 프로토콜).

### 18.3 ★ v3 신규: Skin Efficacy Literature Recovery

- Stage 3 top-10 target에 대해 Stage 0의 efficacy KG가 라벨한 효능 카테고리가 **알려진 사실**과 일치하는지 측정.
- 예: 입력 = retinol → top target = RAR → KG 라벨 = "anti-wrinkle, acne, photoaging" — 정답.
- 메트릭: 라벨 정확도 (manually curated ground truth 대비).

### 18.4 ★ v3 신규: Generated Analog Quality 평가

Stage 5.6 계약이 구현된 뒤 활성화할 미래 평가다. 현재 claim iteration에서는
REINVENT analog가 생성되지 않으므로 이 평가를 통과로 기록할 수 없다.

### 18.5 ★ v3 신규: Interaction-Atom Conservation Test (현재 차단)

- Stage 5.5 index는 bound-complex ligand order이고 parent SDF index가 아니다.
- target-specific atom-map sidecar가 구현·검증되기 전에는 보존율을 계산하지
  않으며 `--allow-threshold-failure`도 이 계약을 우회할 수 없다.

---

## 19. 16 GB GPU 운용 가이드 (v3 갱신)

v2 표 +:

| 단계 | VRAM | 시간 | 비고 |
|---|---|---|---|
| Stage 0 KG 구축 (1회) | 0 (CPU + API) | 1-3일 | PubTator API rate limit 주의 |
| Stage 2.5 fingerprint 매칭 | 0 (CPU) | <1분 | 미리 인덱싱된 fingerprint |
| Stage 5.5 interaction atoms | 2-4 GB | 10-30분 | PLIP/ProLIF agreement evidence; causal pharmacophore claim 아님 |
| **Stage 5.6 REINVENT 4** | 미측정 | 미측정 | integration contract 구현 전 fail-closed |
| Mini-validation Boltz-2 (250개) | 미측정 | 미측정 | Stage 5.6 활성화 후 벤치마크 필요 |
| Stage 7.5 AiZynthFinder | 0 (CPU+ONNX) | 30s-2분/분자 | CPU 32-core 권장 |
| Stage 11 figure 생성 | 0 | 10-30분 | matplotlib + Mol* PNG |

전체 downstream 실행 시간은 Stage 5.6/7 계약 구현 후 실제 벤치마크로 다시
측정해야 하며 현재 문서에서 end-to-end 시간을 claim하지 않는다.

---

## 20. 디렉터리 구조 (v3)

v2에 다음 추가:
```
data/
├── ... (v2와 동일)
├── hpa/                              # Human Protein Atlas
├── skin_proteome/                    # skin.science
├── gtex_v10/
├── scrnaseq_skin/
├── skin_expression/                  # 통합 SkinScore TSV
├── cosing/                           # CosIng INCI database
├── drug_avoidance/                   # DrugBank + ChEMBL phase-4 + FDA OB
├── skin_efficacy_kg/                 # Neo4j dump + Cypher script
└── pharmacophore_smarts/             # cosmetic-relevant SMARTS library

workflow/rules/
├── ... (v2와 동일)
├── stage2_5_cosmetic_drug.smk        # v3
├── stage5_5_pharmacophore.smk        # v3
├── stage5_6_analog_gen.smk           # v3
├── stage7_5_retrosynthesis.smk       # v3
└── stage11_publication.smk           # v3

eval/
├── ... (v2와 동일)
├── cosmetic_retrospective/           # v3
├── skin_efficacy_recovery/           # v3
├── analog_quality/                   # v3
└── pharmacophore_conservation/       # v3
```

---

## 21. 한계 및 주의사항 (v3 보강)

**v2 한계 + 화장품 specialization 한계**:

1. **Boltz-2 affinity는 ranking proxy**. 절대값 해석 금지.
2. **DTI bias** — Stage 3 disagreement 분석으로 진단.
3. **BioEmu**는 ligand·온도·pH·membrane 없음.
4. **co-folding은 allosteric/cryptic site 약함**.
5. **Skin sensitization in silico**는 in vitro/regulatory 대체 불가.
6. **CosIng 매칭의 한계**: extract/polymer/peptide는 정확한 SMILES 매핑 불가 (전체 약 20-30%). 이런 경우 INCI 매칭 confidence ↓.
7. **DrugBank 라이선스**: 상업 사용 시 별도 계약. 비상업 또는 ChEMBL + Orange Book 대체 운영.
8. **HPA CC-BY-SA**: 파생 DB 공개 시 동일 라이선스 요구.
9. **PubTator NER 정확도**: F1 ~85% 정도 (gene/disease/chemical), 일부 잘못된 관계 추출 가능. KG 결과는 hint 수준, 최종 해석은 사용자.
10. **REINVENT 4 현재 차단**: custom scoring plugin, prior/agent artifact,
    upstream-compatible config, atom-map 계약이 없어 claim-capable 생성은
    fail-closed. 향후 활성화 후에도 novelty와 train-set leakage 측정이 필수.
11. **AiZynthFinder 한계**: USPTO 기반이라 academic chemistry에 강함, 일부 industrial process나 catalyst-specific reaction은 미반영. "합성 가능"이라는 예측은 chemistry 자체의 retrosynthetic feasibility 평가 — 실제 yield/scalability는 별개.
12. **화장품 규제**: in silico 결과는 EU Cosmetics Regulation 1223/2009 또는 한 화장품법의 안전성 평가를 대체할 수 없음. 본 플랫폼은 **R&D 우선순위 결정 도구**이며, 최종 product 출시 전 in vitro (DPRA, KeratinoSens, h-CLAT, EpiSkin 등) + 패치 테스트가 필수.
13. **알레르기 정보**: EU 80 mandatory allergen 목록 별도 cross-check 권장 (Annex III).
14. **천연물 추출물 / 펩타이드**: 단일 분자가 아닌 mixture는 본 파이프라인이 직접 처리 어려움. 주요 성분 분리 후 입력.
15. **계면활성제, 폴리머, UV filter**: 일부 화장품 성분은 단백질 binding 메커니즘이 아닌 물리화학적 작용 (occlusion, dispersion 등) — 본 파이프라인의 적용 범위 밖.

---

## 22. 빠른 시작 (One-page Cheatsheet, v3)

```bash
# (최초 1회) 전체 인프라 구축 — v3 새 DB 포함
mamba env create -f envs/master.yml
snakemake -s workflow/rules/stage0_infra.smk --cores 32 --resources gpu=1
# → 3-5일, ~300 GB

# 현재 지원되는 target-identification 경로
python scripts/run_skinscout.py \
  --smiles "OC1=CC(=O)NC(=N1)..." \
  --run-id novel_skin_brightener \
  --preset target-id \
  --mode fast \
  --cores 16

# 결과
xdg-open results/runs/novel_skin_brightener/09_report/index.html
# Stage 5.6/7/7.5와 non-placeholder publication package는 현재 fail-closed

# Retrospective 평가 (cosmetic ingredient recovery)
bash eval/run_cosmetic_retrospective.sh
```

---

## 23. 출판 전략 (Publication Roadmap)

사용자 요구사항 10번에 대한 구체적 제안.

### 23.1 가능한 논문 1편 — Methodology

**제목 후보**:
- "An Open-Source Claim-Quality Pipeline for Cosmetic Ingredient Discovery: Skin-Expression-Weighted Target Identification, Pose-Supported Analog Constraints, and Retrosynthesis"
- "From SMILES to Synthesis: An Integrated Bias-Aware Platform for In Silico Cosmetic Material Design"

**Target journals (open access 가능)**:
- *Journal of Cheminformatics* — methodology focus, open access, IF ~7
- *Briefings in Bioinformatics* — review/method
- *npj Computational Materials* / *Communications Chemistry* — broad methodology
- *Journal of Cosmetic Dermatology* — domain-specific
- *PLOS Computational Biology* — methodology + KG

**핵심 figure** (위 Stage 11 자동 생성):
- Workflow + dual-mode + skin-weighted ranking
- DTI vs Docking disagreement per-protein-class — **bias 정량화가 임팩트 큰 단독 figure**
- Cosmetic retrospective recovery (retinol → RAR, kojic acid → TYR 등)
- 향후 계약 구현 후 analog novelty + synthesizability scatter
- 현재 검증된 범위의 case study + 명시적 downstream blocker

### 23.2 가능한 논문 2편 — Knowledge Graph 자체

**제목**: "A Curated Knowledge Graph Linking Human Proteins, Cosmetic Ingredients, and Skin Efficacy from 36 Million PubMed Articles"

- Stage 0의 skin-efficacy KG 자체를 데이터 publication.
- Zenodo + GitHub publish, *Scientific Data* 또는 *Database* journal.

### 23.3 가능한 논문 3편 — Case study

- 특정 신규 후보의 target-prioritization 결과 → 실험 검증 (in vitro
  tyrosinase assay 등). Downstream analog/MD 단계는 계약 구현 후 별도 검증.

### 23.4 Pre-print 전략

- bioRxiv / ChemRxiv 동시 업로드, 코드는 GitHub MIT.
- Zenodo로 데이터/모델 weights archive (DOI 부여).

### 23.5 사용자 (Keunsoo 교수) 컨텍스트 기반 권장

- 단국대 미생물학과 + 다중오믹스 통합 분석 + AI drug discovery 트랙 → 화장품 영역은 **새로운 application**이지만 **기존 platform 기술의 확장**으로 framing 가능.
- BK21 grant 또는 NRF 산업협력 프로젝트와 연계 가능.
- 화장품 회사와의 공동 연구 시: in silico 후보 → 회사가 in vitro 검증 → 공동 publication 모델.

---

## 24. 참고 핵심 문헌 (v3 추가)

v2 문헌 +:

- **AiZynthFinder**: Genheden et al., 2020, *J. Cheminform.* 12:70 + v4.0 update Saigiridharan et al., 2024.
- **REINVENT 4**: Loeffler et al., 2024, *J. Cheminform.* 16:20.
- **RAscore**: Thakkar et al., 2021, *Chem. Sci.* 12:3339–3349.
- **PLIP**: Salentin et al., 2015 + v2 update Adasme et al., 2021.
- **ProLIF**: Bouysset & Fiorucci, 2021, *J. Cheminform.*
- **Pharmer**: Koes & Camacho, 2011, *J. Chem. Inf. Model.*
- **PubTator 3.0**: Wei et al., 2024, *Nucleic Acids Res.* 52(W1):W540–W546.
- **Human Protein Atlas v25**: Uhlén et al., latest update 2024.
- **Skin proteome atlas**: Dyring-Andersen et al., 2020, *Nat. Commun.* 11:5587.
- **CosIng**: European Commission, https://ec.europa.eu/growth/tools-databases/cosing.
- **DrugBank**: Wishart et al., 2018 + ongoing.
- **OpenTargets Platform**: Ochoa et al., 2023, *Nucleic Acids Res.*
- **GTEx v10**: 2024 release.
- **SHARP** (synthesizable hierarchical RL): 2025 (대안 generative).

---

## 25. v2 → v3 마이그레이션 노트

- 인프라 (Stage 0)에 4개 DB 추가 필요 (~ 1-2일 추가).
- Stage 2.5, 5.5, 7.5, 11 코드와 Stage 5.6 계약 게이트가 추가됨. Stage 5.6과
  이에 의존하는 end-to-end 경로는 아직 활성화되지 않았다.
- 평가 결과의 비교: v2 결과는 v3에 그대로 유효, v3 신규 평가는 별도 보고.
- 시간: Stage 5.6/7 계약 구현 전에는 v3 full downstream runtime을 claim하지
  않는다. 각 단계 활성화 후 실제 host benchmark로 갱신한다.

---

*문서 작성: 2026-05-24. 모든 도구 버전·라이선스는 작성 시점 확인. 실제 배포 전 각 도구의 LICENSE 재검토 필수. 화장품 R&D 우선순위 결정 도구이며, in vitro/regulatory 검증을 대체하지 않음.*
