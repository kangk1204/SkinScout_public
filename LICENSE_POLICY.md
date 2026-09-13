# License Policy & Per-Tool Audit

> Source: INSTRUCTIONS.md §1 (2026-05-24).
> All entries verified at design-time. Re-audit before each public deployment.
> Last re-audit: 2026-09-09 — pandas/NumPy/pyarrow added (they were in every
> code path but missing from the table), and §2-B added for the analog-search
> handoff build.

## 1. Policy

Only tools with **commercial-use-permissive** licenses (MIT, Apache-2.0, BSD-2/3, LGPL-2.1+, GPL-2.0/3.0 with linking exceptions, CC-BY) are integrated into the pipeline. Tools whose output cannot be commercially redistributed (AlphaFold3 weights, Chai-1 weights) are excluded; Schrödinger / Gaussian / AMBER pmemd.cuda / DEREK / ChemTunes are excluded for licensing or cost.

Note: GPL-licensed dependencies (OpenBabel, gmx_MMPBSA, Foldseek, PLIP) and LGPL dependencies (Meeko, GROMACS, xTB, CREST) are isolated to their own conda environments so the rest of the codebase remains under permissive licensing. Outputs (numerical results, structures) are not derivative works of the GPL code under U.S. fair-use copyright theory, but redistribution of pipeline source must respect the strongest copyleft requirement of any combined binary.

## 2. Verdict table

| Stage | Tool | Version | License | Commercial? | Notes |
|-------|------|---------|---------|-------------|-------|
| 1 preprocess | RDKit | 2025.03 | BSD-3 | ✅ | |
| 1 preprocess | Dimorphite-DL | 1.3 | Apache-2.0 | ✅ | |
| 1 preprocess | OpenBabel | 3.1.1 | GPL-2.0 | ⚠ GPL | Isolate in env, treat as separate process boundary |
| 1 preprocess | Meeko | 0.6 | LGPL-2.1 | ✅ LGPL | Dynamic linking only |
| 2 ADMET | ADMET-AI | 1.4 | MIT | ✅ | |
| 2 ADMET | STopTox | 1.0 | public web app / CLI | ⚠ terms | Public web footer says research/educational use; re-confirm commercial terms before public deployment |
| 2 ADMET | HuSSPred | 2024-11 | public web + GitHub code | ⚠ terms | Public endpoint supported; re-audit redistribution/commercial terms before deployment |
| 2 ADMET | Pred-Skin | 2026 web app | public web | ⚠ terms | Public endpoint supported; site advertises commercial offering, so do not assume commercial redistribution rights |
| 2 ADMET | Skin Doctor CP | 0.0.12 | NERDD web service | ❌ default | Excluded from default consensus because NERDD states free use for non-commercial and academic research only |
| 2 ADMET | PAINS/Brenk/NIH filters | RDKit | BSD-3 | ✅ | |
| 0 infra | AlphaFold DB human proteome v4 | v4 | **CC-BY-4.0** | ✅ + attribution | Attribution required in publications |
| 0 infra | P2Rank | 2.5 | Apache-2.0 | ✅ | |
| 0 infra | MMseqs2 | 15 | BSD-3 | ✅ | |
| 0 infra | Foldseek (옵션) | 9 | GPL-3 | ⚠ GPL | Optional; isolate |
| 3 docking | AutoDock-GPU | 1.6 | Apache-2.0 | ✅ | |
| 3 docking | GNINA | 1.3 | Apache-2.0 | ✅ | |
| 3 docking | DiffDock-L | 1.1 | MIT | ✅ | |
| 3 docking | AutoDock Vina | 1.2.7 | Apache-2.0 | ✅ | |
| 3 docking | RTMScore | 1.0 | MIT | ✅ | |
| 3 DTI | PSICHIC | 1.0 | MIT | ✅ | |
| 3 DTI | ConPLex, DrugBAN | latest | MIT | ✅ | optional ensemble |
| 5 fold | ESMFold | 1.0.3 | MIT | ✅ | |
| 5 fold | Boltz-2 | 1.x (2025-06) | **MIT** | ✅ | confirmed by Passaro et al. 2025 |
| 5 fold | NeuralPLexer3 | 3.x | MIT | ✅ | cross-check |
| 5 fold | RoseTTAFold-All-Atom | 1.0 | BSD-3 | ✅ | |
| 5 fold | PoseBusters | 0.4 | BSD-3 | ✅ | |
| 6 ensemble | BioEmu | 1.0 (2025-07) | MIT | ✅ | |
| 6 ensemble | AlphaFlow / ESMFlow | - | MIT | ✅ | |
| 7 MD | GROMACS | 2024.4 | LGPL-2.1 | ✅ LGPL | linker boundary |
| 7 MD | OpenMM | 8.1 | MIT | ✅ | alt engine |
| 7 MD | OpenFF Toolkit + Sage | 2.2 | MIT | ✅ | |
| 7 MD | ACPYPE | - | GPL-2.0 | ⚠ GPL | isolate |
| 7 MD | gmx_MMPBSA | 1.6 | GPL-3 | ⚠ GPL | isolate, run as subprocess |
| 7 MD | alchemlyb / pmx | - | BSD/MIT | ✅ | optional FEP |
| 8 QM | xTB / GFN2-xTB | 6.7 | LGPL-3 | ✅ LGPL | |
| 8 QM | CREST | 3.0 | LGPL-3 | ✅ LGPL | |
| 8 QM | PySCF + GPU4PySCF | 2.7 | Apache-2.0 | ✅ | |
| 9 viz | Mol* | 4.x | Apache-2.0 | ✅ | |
| 9 viz | PyMOL Open-Source | 3.x | BSD-style | ✅ | |
| 9 viz | ProLIF | latest | Apache-2.0 | ✅ | |
| 9 viz | PLIP | latest | GPL-2.0 | ⚠ GPL | isolate, subprocess |
| 10 expand | ZINC22 | rolling | open | ✅ | per-tranche T&C still apply |
| workflow | pandas | 2.x / 3.x | BSD-3 | ✅ | Used by every path including the analog search |
| workflow | NumPy | 2.x | BSD-3 | ✅ | Used by every path including the analog search |
| workflow | pyarrow (parquet) | 17+ | Apache-2.0 | ✅ | Library and cache storage |
| workflow | Snakemake | 8.x | MIT | ✅ | |
| workflow | DVC + git | - | Apache-2.0 + GPL-2.0 | mixed | DVC permissive; git is for source control only |

## 2-B. 대체소재 검색 경로 (공동연구자 전달본)

`--profile analog` 로 깔면 대체소재 검색만 돈다. 그 경로가 실제로 부르는 것은
아래 넷뿐이고 **전부 상업 이용이 열려 있다.** 위 표에서 ⚠/❌ 가 붙은 것들은
이 경로에 하나도 들어오지 않는다.

| 무엇에 | 도구 | 라이선스 |
|---|---|---|
| 구조 판정·지문·정규화 | RDKit | BSD-3 |
| ADMET 예측(안전 축) | ADMET-AI | MIT |
| 화면의 3D | Mol\* | Apache-2.0 |
| 표·배열·저장 | pandas · NumPy · pyarrow | BSD-3 / BSD-3 / Apache-2.0 |

특히 **감작성 예측 4종(STopTox · HuSSPred · Pred-Skin · Skin Doctor CP)은
이 경로에 없다.** 그것들은 Stage 2 안전성 화면의 것이고, 외부 연구기관 서버로
구조를 전송하며 연구·교육 한정 조건이 붙어 있다. 대체소재 검색은 이 컴퓨터
안에서만 돌고 아무것도 밖으로 보내지 않는다.

## 3. Excluded (re-stated for safety)

- ❌ AlphaFold3 weights — academic-only license, not redistributable.
- ❌ Chai-1 weights — non-commercial; web inference allowed for spot-check only.
- ❌ Schrödinger Glide / FEP+ — commercial.
- ❌ Gaussian — commercial.
- ❌ AMBER `pmemd.cuda` — commercial (Amber Tools is GPL-3 and OK; the CUDA accelerator is not).
- ❌ DEREK Nexus, ChemTunes — commercial.

## 4. Attribution boilerplate

Any artifact published from this pipeline must include:

> Structures derived from the **AlphaFold Protein Structure Database** (Varadi et al., *Nucleic Acids Res.*, 2024), used under CC-BY-4.0.
> Docking performed with **AutoDock-GPU** (Santos-Martins et al., *JCTC* 17:1060-1073, 2021).
> Affinity ranking via **Boltz-2** (Passaro et al., 2025).
> Ensemble sampling via **BioEmu** (Lewis et al., *Science* 389:eadv9817, 2025).

Per §17.8 of INSTRUCTIONS.md, redistribution of derived structures must reproduce the AlphaFold DB attribution.
