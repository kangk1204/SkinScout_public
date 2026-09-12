> **Archived unrelated reference:** This NanoForge document is not a SkinScout
> specification and must not be used as implementation guidance for this repository.
> `INSTRUCTIONS.md` is the authoritative SkinScout specification.

# NanoForge v3 — Automated Nanobody Design & Refinement Pipeline

**Authoritative build specification for Codex CLI.**

**Target environment**: H200 single-GPU workstation, Ubuntu 22.04+, CUDA 12.x, Python 3.11.

**Scope**: Target protein sequence → conformational-ensemble-aware epitope mapping → ranked, MD/NNP-refined, in-silico-matured VHH candidates → FTO-checked experimental brief.

**Hard constraint**: every dependency must be verified commercially usable. No PyRosetta, no AlphaFold3 weights, no NetMHCIIpan, no FoldX, no Schrödinger, no Gaussian, no ORCA, no AMBER pmemd.cuda, no MODELLER, no AbNatiV, no Ibex weights, no nanoBERT, no Llamanade, no NABP-BERT.

**Authority**: this document supersedes `NANOBODY_PIPELINE_SPEC_v2.md`, `NANOBODY_PIPELINE_SPEC_v2.1_NNP_PATCH.md`, `NANOBODY_PIPELINE_SPEC_v2.2_VHH_TOOLS_PATCH.md`, and `NANOBODY_PIPELINE_SPEC_v2.3_CORRECTION.md`. Treat v3 as the single source of truth.

---

# 0. License register (every dependency verified)

CI runs `pip-licenses` + `scancode-toolkit` on every PR. Any new dependency without an entry below fails the build.

## 0.1 Models (weights + code)

| Tool | License | Commercial OK | Verification | Obligations |
|---|---|---|---|---|
| Boltz-2 | MIT | ✅ | repo LICENSE | cite Passaro 2025 |
| Chai-1 | Apache 2.0 (code + weights) | ✅ | repo LICENSE | attribution |
| ESM-2 / ESMFold | MIT | ✅ | repo LICENSE | cite Lin 2023 |
| RFantibody | MIT | ✅ | repo LICENSE + Baker Lab Nov 2025 announcement | cite Bennett 2025 |
| ProteinMPNN | MIT | ✅ | repo LICENSE | cite Dauparas 2022 |
| LigandMPNN | MIT | ✅ | repo LICENSE | cite Dauparas 2023 |
| FreeBindCraft | MIT | ✅ | repo LICENSE | cite Pacesa 2024; no PyRosetta |
| NanoBodyBuilder2 / ImmuneBuilder | BSD-3 | ✅ | repo LICENSE | attribution |
| IgFold | Apache 2.0 | ✅ | repo LICENSE | attribution |
| ANARCI | BSD-3 | ✅ | repo LICENSE | attribution |
| AbLang2 | BSD-3 | ✅ | repo LICENSE | attribution; **base model for VHH-nativeness head** |
| Paragraph | BSD-3 | ✅ | setup.py inspection | attribution; **used for paratope sanity** |
| BioPhi / Sapiens | MIT | ✅ | repo LICENSE | cite Prihoda 2022; **used for humanization** |
| AlphaFlow / ESMFlow | MIT (code + weights) | ✅ | repo LICENSE + README | cite Jing 2024; **used for conformational ensemble** |
| OpenFold | Apache 2.0 | ✅ | repo LICENSE | AlphaFlow dependency |
| MACE-OFF / MACE-MP-0 | MIT | ✅ | repo LICENSE | cite Kovács 2025 / Batatia 2024 |
| TorchANI / ANI-2x | MIT | ✅ | repo LICENSE | cite Devereux 2020 |
| AIMNet2 | MIT | ✅ | repo LICENSE | cite Anstine 2024 |
| MHCflurry | Apache 2.0 | ✅ | repo LICENSE | MHC-I only; cite O'Donnell 2020 |

## 0.2 MD / QM / scientific computing

| Tool | License | Commercial OK | Notes |
|---|---|---|---|
| OpenMM | MIT | ✅ | primary classical MD engine |
| GROMACS | LGPL-2.1 | ✅ (dynamic link) | alternative MD + FEP via pmx |
| pmx | LGPL | ✅ | FEP / alchemistry |
| alchemlyb | BSD-3 | ✅ | FEP analysis |
| PySCF | Apache 2.0 | ✅ | full QM (DFT/HF/MP2) |
| Psi4 | LGPL | ✅ (dynamic link) | alternative QM |
| xtb / GFN-xTB | LGPL | ✅ | semi-empirical QM |
| DFTB+ | LGPL | ✅ | optional QM alternative |
| ASE | LGPL | ✅ | universal NNP-MD driver |
| openmm-torch | MIT | ✅ | OpenMM-NNP bridge |
| FreeSASA | LGPL | ✅ (link only) | surface area |
| propka | LGPL | ✅ | pKa assignment |
| MMseqs2 / ColabFold-search | MIT | ✅ | MSA, BLAST replacement |
| P2Rank | Apache 2.0 | ✅ | pocket detection |
| fpocket | MIT | ✅ | alternative pocket detection |

## 0.3 Infrastructure

| Tool | License | Commercial OK |
|---|---|---|
| Nextflow DSL2 | Apache 2.0 | ✅ |
| Apptainer | BSD-3 | ✅ |
| PostgreSQL 16 | PostgreSQL License | ✅ |
| FastAPI | MIT | ✅ |
| Streamlit | Apache 2.0 | ✅ |
| pydantic v2 | MIT | ✅ |
| SQLAlchemy 2.x | MIT | ✅ |
| Alembic | MIT | ✅ |

## 0.4 Data sources

| Source | License | Commercial OK | Notes |
|---|---|---|---|
| OAS (Observed Antibody Space) | CC BY 4.0 | ✅ | attribution; **primary VHH-nativeness training data (camelid subset)** |
| INDI | CC BY 4.0 | ✅ | attribution; alternative training data |
| SAbDab structures | OPIG free for download | ✅ at runtime fetch | do NOT bundle in repo; runtime cache |
| PDB | public domain | ✅ | always cite source structures |
| UniProt | CC BY 4.0 | ✅ | attribution |
| IEDB | CC BY 4.0 | ✅ | for MHC-II proxy |
| USPTO patent bulk | public domain | ✅ | for FTO module |
| EPO Open Patent Services | free with registration | ✅ | for FTO module |
| NbThermo | open scientific data | ✅ | for thermostability head training |
| AVIDa-hIL6 | CC BY 4.0 (NeurIPS 2023 dataset convention) | ✅ at adopt-verify | for affinity head auxiliary training |
| NbBench | open benchmark | ✅ at adopt-verify | for CI validation |

## 0.5 Explicitly excluded (DO NOT add to dependencies)

| Tool | Reason |
|---|---|
| AlphaFold3 weights | CC BY-NC-SA 4.0 — non-commercial |
| AlphaProteo | not released |
| PyRosetta / Rosetta | commercial license required |
| NetMHCIIpan, NetMHCpan | academic only |
| FoldX | commercial license required |
| HADDOCK web | academic only |
| Schrödinger Suite (Maestro, FEP+, Glide, Jaguar) | commercial license required |
| Gaussian | commercial license required |
| ORCA | commercial license required |
| AMBER pmemd.cuda | restricted; use OpenMM/GROMACS |
| MODELLER | academic only; **blocks Llamanade** |
| AbNatiV | license unclear (Sormanni lab); defer |
| Ibex weights | Genentech Apache 2.0 Non-Commercial |
| NbForge | license unclear; defer |
| **nanoBERT (NaturalAntibody)** | CC BY-NC-SA 4.0 on HuggingFace model card |
| **Llamanade** | requires MODELLER (academic) — pipeline cannot run commercially |
| **NABP-BERT** | no LICENSE file → All Rights Reserved; also uses Sanofi-trademarked "Nanobody®" name |
| TANGO | academic only; reimplement scoring with published parameters |
| HuNb / GeoFlow-V2-ab / IgGM / tFold-Ab / DiffAb / ANTIPASTI / NbX | unverified licenses; defer |

---

# 1. Application profiles

Select at run-time via `application_profile`. Each profile pre-configures defaults.

| Profile | Use case | Defaults |
|---|---|---|
| `general` | Soluble globular target, full pipeline | All modules on; ESMFlow ensemble |
| `aav_capsid` | AAV serotype engineering, BBB receptor binders | Multi-conformation on; CNS-relevant immunogenicity; glycan-full |
| `gpcr_allosteric` | GPCR with cryo-EM structure | Multi-conformation on; AlphaFlow-PDB; ECL-restricted epitopes |
| `ion_channel` | Selectivity filter / pore region | Conformational ensemble; pore-region whitelist |
| `soluble_research` | Research tool, no humanization | Humanization skipped; affinity priority |
| `therapeutic_strict` | Therapeutic candidate | All filters strict; QM verification on; FTO mandatory; FEP enabled |

**Profile gates downstream behavior.** A `therapeutic_strict` run that fails FTO check halts before brief generation; a `soluble_research` run skips humanization entirely.

---

# 2. Component stack (final, locked)

| Role | Tool | License |
|---|---|---|
| Complex structure + affinity | Boltz-2 | MIT |
| Alt. validator | Chai-1 | Apache 2.0 |
| Fast monomer | ESMFold | MIT |
| PLM embeddings | ESM-2 | MIT |
| Conformational ensemble (default) | **ESMFlow-MD+Templates 12l-distilled** | MIT |
| Conformational ensemble (alt) | AlphaFlow-MD+Templates | MIT |
| Conformational ensemble (physics) | NNP-MD (MACE-OFF) | MIT |
| De novo backbone+seq+verify | **RFantibody** | MIT |
| Sequence design (fallback) | LigandMPNN / ProteinMPNN | MIT |
| Binder pipeline (ablation) | FreeBindCraft | MIT |
| Nanobody structure (monomer) | NanoBodyBuilder2 / ImmuneBuilder | BSD-3 |
| Numbering | ANARCI | BSD-3 |
| Antibody LM (base for VHH head) | AbLang2 | BSD-3 |
| Paratope sanity | **Paragraph** | BSD-3 |
| Humanization | BioPhi + VHH post-filter | MIT |
| Pocket detection | P2Rank, fpocket | Apache 2.0 / MIT |
| Classical MD | OpenMM 8.x | MIT |
| Classical MD (alt) | GROMACS 2024 | LGPL |
| FEP | pmx + GROMACS | LGPL |
| FEP analysis | alchemlyb | BSD-3 |
| Primary NNP | MACE-OFF | MIT |
| NNP (charge-aware) | AIMNet2 | MIT |
| NNP (speed-critical) | ANI-2x | MIT |
| NNP (universal elements) | MACE-MP-0 | MIT |
| Semi-empirical QM | xtb (GFN2) | LGPL |
| Full QM | PySCF | Apache 2.0 |
| ASE bridge | ASE | LGPL |
| MHC-I prediction | MHCflurry | Apache 2.0 |
| MHC-II proxy | IEDB-identity scan (custom) | n/a |
| MSA / sequence search | MMseqs2 | MIT |
| Patent FTO | USPTO/EPO bulk + MMseqs2 | public + MIT |
| Workflow | Nextflow DSL2 | Apache 2.0 |
| Containers | Apptainer | BSD-3 |
| API | FastAPI | MIT |
| UI | Streamlit | Apache 2.0 |
| DB | PostgreSQL 16 | PostgreSQL License |

---

# 3. High-level architecture

```
                              INPUT
                                ▼
┌──────────────────────────────────────────────────────────┐
│ Module 0: FTO precheck (target patent landscape)         │ ← warns before compute
└──────────────────────────────┬───────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────┐
│ Module 1: Target prep                                    │
│   structure + epitope + conformational ensemble          │
│   (ESMFlow / AlphaFlow / NNP-MD / AF-cluster / off)      │
└──────────────────────────────┬───────────────────────────┘
                               │
            ┌──────────────────┴─────────────────┐
            ▼                                    ▼
┌──────────────────────────┐         ┌──────────────────────────┐
│ Module 2a: Library route │         │ Module 2b: De novo route │
│   VHH lib × Boltz-2      │         │   RFantibody (Ig-aware)  │
└──────────┬───────────────┘         └──────────┬───────────────┘
           └─────────────────┬──────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────┐
│ Module 3: Complex validation                             │
│   Boltz-2 + Chai-1, multi-seed ensemble                  │
└──────────────────────────────┬───────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────┐
│ Module 4: Developability + humanness                     │
│   Liability scan + ΔΔG + AbLang2-OAS VHH-nativeness      │
│   + BioPhi humanization with VHH post-filter             │
└──────────────────────────────┬───────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────┐
│ Module 4.5: Paratope sanity (Paragraph)                  │
│   Orthogonal paratope vs Boltz-2 interface concordance   │
└──────────────────────────────┬───────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────┐
│ Module 5: CDR refinement (NNP-primary)                   │
│   classical equilibration → NNP-MD production            │
│   → NNP ΔE_bind ± MM/GBSA cross-check                    │
│   → (auto) QM verification if polarization-sensitive     │
└──────────────────────────────┬───────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────┐
│ Module 6: In silico affinity maturation                  │
│   Tier 1: ProteinMPNN/ESM-IF (fast)                      │
│   Tier 2: NNP-MD ΔΔG                                     │
│   Tier 3: FEP via pmx+GROMACS (opt-in)                   │
└──────────────────────────────┬───────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────┐
│ Module 7: Cross-reactivity screen                        │
│   paralog BLAST + Boltz-2 spot check + PSR proxy         │
└──────────────────────────────┬───────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────┐
│ Module 8: Pareto ranking + diversity clustering          │
└──────────────────────────────┬───────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────┐
│ Module 9: FTO post-check (CDR3 + full seq patent BLAST)  │
└──────────────────────────────┬───────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────┐
│ Module 10: Experimental brief (tier-coded, DNA, SPR)     │
└──────────────────────────────┬───────────────────────────┘
                              OUTPUT
```

---

# 4. Project layout

```
nanoforge/
├── README.md
├── LICENSE                                # MIT for wrapper code
├── LICENSES/
│   ├── REGISTER.md                        # full license register
│   └── deps/                              # per-dep license texts
├── pyproject.toml
├── nextflow.config
├── main.nf                                # entry workflow
├── configs/
│   ├── default.yaml
│   ├── profiles/
│   │   ├── general.yaml
│   │   ├── aav_capsid.yaml
│   │   ├── gpcr_allosteric.yaml
│   │   ├── ion_channel.yaml
│   │   ├── soluble_research.yaml
│   │   └── therapeutic_strict.yaml
│   ├── targets/                           # per-target overrides
│   └── benchmarks/                        # calibration targets
├── containers/                            # Apptainer .def files
│   ├── base_cuda.def
│   ├── boltz2.def
│   ├── chai1.def
│   ├── rfantibody.def
│   ├── alphaflow.def
│   ├── mace.def
│   ├── ani.def
│   ├── aimnet2.def
│   ├── xtb_pyscf.def
│   ├── gromacs_pmx.def
│   ├── ablang2.def
│   ├── paragraph.def
│   ├── biophi.def
│   ├── immunebuilder.def
│   └── freebindcraft.def
├── nanoforge/
│   ├── __init__.py
│   ├── cli.py                             # `nanoforge run --target ...`
│   ├── api/                               # FastAPI service
│   ├── ui/                                # Streamlit app
│   ├── pipeline/
│   │   ├── fto_pre.py                     # Module 0
│   │   ├── target.py                      # Module 1
│   │   ├── epitope.py
│   │   ├── ensemble.py                    # AlphaFlow/ESMFlow/NNP-MD dispatch
│   │   ├── library_route.py               # Module 2a
│   │   ├── denovo_route.py                # Module 2b
│   │   ├── validate.py                    # Module 3
│   │   ├── developability.py              # Module 4
│   │   ├── humanness.py
│   │   ├── humanization.py                # BioPhi + VHH post-filter
│   │   ├── paratope_sanity.py             # Module 4.5
│   │   ├── refinement/                    # Module 5
│   │   │   ├── md_refine.py
│   │   │   ├── nnp_refine.py
│   │   │   ├── nnp_dg_bind.py
│   │   │   └── qm_verify.py
│   │   ├── maturation/                    # Module 6
│   │   │   ├── saturation_mutgen.py
│   │   │   ├── fast_scoring.py
│   │   │   ├── nnp_ddg.py
│   │   │   └── fep.py
│   │   ├── crossreact.py                  # Module 7
│   │   ├── rank.py                        # Module 8
│   │   ├── fto_post.py                    # Module 9
│   │   └── brief.py                       # Module 10
│   ├── adapters/
│   │   ├── boltz2.py
│   │   ├── chai1.py
│   │   ├── esm.py
│   │   ├── esmfold.py
│   │   ├── alphaflow.py                   # AlphaFlow + ESMFlow unified
│   │   ├── rfantibody.py
│   │   ├── proteinmpnn.py
│   │   ├── ligandmpnn.py
│   │   ├── freebindcraft.py
│   │   ├── immunebuilder.py
│   │   ├── biophi.py
│   │   ├── ablang2.py
│   │   ├── paragraph.py
│   │   ├── anarci.py
│   │   ├── openmm_md.py
│   │   ├── gromacs_fep.py
│   │   ├── pmx_fep.py
│   │   ├── mace.py
│   │   ├── ani.py
│   │   ├── aimnet2.py
│   │   ├── xtb.py
│   │   ├── pyscf_qm.py
│   │   ├── mhcflurry.py
│   │   ├── p2rank.py
│   │   └── patent_blast.py
│   ├── data/
│   │   ├── vhh_library_builder.py
│   │   ├── sabdab_fetch.py
│   │   ├── pdb_fetch.py
│   │   ├── uspto_index.py
│   │   ├── epo_ops.py
│   │   ├── iedb_loader.py
│   │   ├── oas_camelid_loader.py
│   │   ├── nbthermo_loader.py
│   │   ├── avida_hil6_loader.py
│   │   └── nbbench_loader.py
│   ├── scoring/
│   │   ├── liabilities.py
│   │   ├── interface.py
│   │   ├── ddg_proxy.py
│   │   ├── vhh_nativeness.py               # AbLang2-OAS head
│   │   ├── thermostability.py              # NbThermo head
│   │   ├── psr_proxy.py
│   │   ├── mmgbsa.py
│   │   ├── nnp_energy.py
│   │   └── tier_classifier.py
│   ├── training/                           # one-time training scripts
│   │   ├── ablang2_vhh_finetune.py
│   │   ├── ablang2_tm_regressor.py
│   │   └── calibration_thresholds.py
│   ├── db/
│   │   ├── models.py
│   │   └── migrations/                     # Alembic
│   └── utils/
│       ├── logging.py
│       ├── numbering.py                    # ANARCI wrapper
│       ├── codon_optimize.py
│       └── adr.py
├── adrs/                                   # Architecture Decision Records
│   ├── ADR-0001-use-boltz2-not-af3.md
│   ├── ADR-0002-no-pyrosetta.md
│   ├── ADR-0003-mhc-ii-iedb-proxy.md
│   ├── ADR-0004-tier-coded-output.md
│   ├── ADR-0005-nnp-primary-refinement.md
│   ├── ADR-0006-fep-opt-in.md
│   ├── ADR-0007-uspto-patent-index.md
│   ├── ADR-0008-defer-abnativ.md
│   ├── ADR-0009-wet-lab-bottleneck.md
│   ├── ADR-0010-application-profiles.md
│   ├── ADR-0011-use-rfantibody.md
│   ├── ADR-0013-paragraph-paratope-sanity.md
│   ├── ADR-0014-exclude-ibex-nc-weights.md
│   ├── ADR-0015-exclude-nanobert-nc.md
│   ├── ADR-0016-exclude-llamanade-modeller.md
│   ├── ADR-0017-exclude-nabp-bert-no-license.md
│   ├── ADR-0018-add-alphaflow-esmflow.md
│   └── ADR-0019-ablang2-oas-vhh-nativeness.md
│   # (ADR-0012 superseded by ADR-0019; numbering preserved for history)
├── workflows/
│   ├── target_prep.nf
│   ├── ensemble.nf
│   ├── library_route.nf
│   ├── denovo_route.nf
│   ├── validate.nf
│   ├── paratope_sanity.nf
│   ├── refine.nf
│   ├── mature.nf
│   ├── crossreact.nf
│   ├── fto_pre.nf
│   ├── fto_post.nf
│   ├── filter_rank.nf
│   └── brief.nf
├── tests/
│   ├── benchmarks/                         # calibration targets + NbBench
│   ├── unit/
│   └── integration/
└── scripts/
    ├── install_models.sh
    ├── build_vhh_library.sh
    ├── index_patents.sh
    ├── train_vhh_nativeness.sh
    ├── calibrate_thresholds.py
    └── license_audit.py
```

---

# 5. Module specifications

## Module 0 — Patent FTO precheck

**Purpose**: warn before launching ~70-hour compute if target's therapeutic class is heavily encumbered.

**Steps**:
1. Resolve target → known disease indication via UniProt + DisGeNET/OpenTargets
2. Query patent landscape:
   - USPTO bulk patent data (pre-indexed locally, ~500 GB)
   - EPO Open Patent Services (OPS) REST API
3. Identify: active patents on target; known anti-target nanobody patents (sequence claims); therapeutic-class patents
4. Output: `fto_landscape.json`

**Gate**:
- `therapeutic_strict`: FTO returns > 5 active relevant patents → require explicit user acknowledgment to proceed
- Other profiles: warning only

**Disclaimer prominent in output**: FTO module is screening only, NOT legal advice. External IP attorney review required before clinical/commercial action.

## Module 1 — Target preprocessor

**Steps**:

1. **Sequence resolution**: FASTA or UniProt ID → fetch sequence + functional annotations + oligomeric state from UniProt + ComplexPortal
2. **Single-state structure**: ESMFold (fast pass) → Boltz-2 (high-confidence pass) with MSA from ColabFold server or local UniRef30
3. **Conformational ensemble** (controlled by `target.ensemble`):
   - `off`: skip
   - `esmflow`: ESMFlow-MD+Templates 12l-distilled, default 50 samples, ~15 min wallclock
   - `alphaflow`: AlphaFlow-MD+Templates 12l-distilled, default 50 samples, ~50 min wallclock
   - `af_cluster`: MSA subsampling on ESMFold, 20 subsamples
   - `nnp_md`: 50 ns MACE-OFF MD with target alone, cluster by Cα RMSD (DBSCAN ε=2 Å)
4. **PTM prediction**:
   - N-glycosylation: scan `N-X-[ST]` (X≠P), weight by SASA × pLDDT
   - O-glycosylation: S/T in disordered regions (IUPred-equivalent via pLDDT < 70 proxy)
   - Disulfides: extract from predicted structure (Cα-Cα < 3 Å between Cys)
5. **Membrane protein detection**: TM helices via hydrophobicity windows + DeepTMHMM-equivalent proxy → flag ECD-only mode if positive
6. **Glycosylation modeling** (`target.glycan_modeling`):
   - `off`: skip
   - `minimal`: exclude epitopes within 8 Å of solvent-accessible NxS/T sites
   - `full`: attach representative Man5/Man9 glycan via GlycoSHIELD-style open implementation, re-evaluate epitope accessibility
7. **Surface analysis**: SASA via FreeSASA; pocket detection via P2Rank (primary) + fpocket (fallback)
8. **Conservation**: BLAST UniRef90 → MUSCLE → per-residue conservation score
9. **Epitope candidate ranking** (composite):
   - SASA > 30%
   - Conservation (tier configurable)
   - pLDDT > 80
   - NOT glycan-shielded
   - User override available

**Output**: `target_ensemble.json` (per-conformer epitope maps):
```json
{
  "uniprot_id": "P04626",
  "sequence": "...",
  "assembly": "monomer",
  "ensemble_method": "esmflow",
  "conformers": [
    {
      "conformer_id": "c1",
      "structure_path": "target_c1.pdb",
      "plddt_mean": 87.2,
      "epitopes": [
        {"id": "epi_01", "residues": [310,311,312,...], "score": 0.91, "rationale": "..."}
      ]
    }
  ]
}
```

## Module 2a — Library route

**One-time library construction**:
1. Download OAS camelid heavy chains + INDI nanobody sequences
2. Filter: length 110–135 aa; hallmark V37/G44/L45/W47 (Kabat) present; Cys22–Cys92 disulfide intact
3. Cluster at 90% CDR3 identity → expect 50k–500k unique scaffolds
4. Optional: enrich with literature-validated affinity-matured sequences for benchmarking

**Screening (per target)**:
1. Sample N candidates (default 10,000)
2. For each conformer × candidate: Boltz-2 multimer prediction
3. Stage 1 (fast): single seed, MSA recycling off → top 1000 by ipTM
4. Stage 2 (HQ): 3 seeds, full MSA recycling → top 200
5. Epitope filter: ≥ 70% interface contacts within designated epitope

**Output**: `library_hits.parquet` with columns `[vhh_id, sequence, conformer_id, ipTM, pAE_iface, plddt_iface, epitope_overlap, source]`

## Module 2b — De novo route (RFantibody)

**Per epitope, per conformer**:

1. **RFantibody backbone diffusion**:
   - Input: target conformer PDB, hotspot residues = epitope, framework template (Vincke universal scaffold default)
   - Settings: 1000 backbones (configurable up to 9000+ for high-rigor); CDR length from natural VHH distribution (built-in to RFantibody)
2. **Antibody-specific MPNN sequence design** (RFantibody's wrapper):
   - 8 sequences per backbone
   - Hallmark residue preservation native
   - Post-hoc validation: V37/E44/R45/G47 present (defensive check)
3. **RoseTTAFold2 antibody-finetuned fold-back validation** (RFantibody's RF2):
   - Filter candidates with intended-backbone RMSD > 2 Å
4. **CDR3 length filter** (defensive): retain 13–24 aa
5. **Free Cys scanner**: reject any extra free Cys beyond canonical Cys22-Cys92

**Output**: `denovo_hits.parquet` (same schema as library hits + `backbone_id`, `rfantibody_run_id`)

**Fallback**: FreeBindCraft can be enabled as ablation/comparison route only via `denovo.enable_freebindcraft_ablation: true`.

## Module 3 — Complex validation

**Input**: merged candidates from 2a + 2b (typical 200–500 candidates).

**Steps**:
1. **Boltz-2 ensemble**: 5 seeds per candidate, multimer mode with target MSA, no MSA for VHH
2. **Chai-1 cross-validation** on top 50 by Boltz-2 → require ipTM concordance > 0.7 between predictors
3. **Per-candidate metrics**:
   - `iptm_mean`, `iptm_std` across seeds
   - `pae_interface` (mean PAE target epitope ↔ VHH CDR atoms)
   - `plddt_interface`
   - **Paratope check** (preliminary): % VHH→target contacts from CDR (≥70% required)
   - **Epitope concordance**: % target contacts within designated epitope (≥80%)
   - **Mode stability**: ≥3/5 seeds agree on epitope (Jaccard > 0.6)
4. **Short OpenMM minimization** for top 50: 5000 steps Amber14 + GBN2 implicit solvent; drop candidates with energy > +20 kcal/mol vs starting model

**Output**: `validated.parquet` + per-candidate complex PDB files

## Module 4 — Developability + humanness

### 4.1 Liability scans (deterministic rules)

| Liability | Pattern | Region | Action |
|---|---|---|---|
| N-glyc | `N[^P][ST]` | CDR | HIGH flag |
| N-glyc | `N[^P][ST]` | framework | MEDIUM flag |
| Deamidation | `N[GS]` | CDR | MEDIUM flag |
| Isomerization | `D[GS]` | CDR | MEDIUM flag |
| Met oxidation | `M` | CDR | LOW flag |
| Trp oxidation | `W` | CDR | LOW flag |
| Free Cys (extra) | `C` not paired | anywhere | HIGH (reject) |
| Acid cleavage | `DP`, `DK` | CDR | MEDIUM flag |

### 4.2 Biophysical scores

- Net charge at pH 7.0 (Henderson-Hasselbalch)
- pI (Bjellqvist)
- GRAVY hydropathy
- Aggregation: reimplement TANGO scoring with published parameters (TANGO is academic; algorithm is public)
- ΔΔG proxy: ESM-IF likelihood ratio (no FoldX)
- Net charge patches: 5-residue sliding window, flag |Q| > 3
- Hydrophobic patches: surface-exposed F/W/Y/L/I/V cluster ≥ 50 Å²

### 4.3 VHH-nativeness via AbLang2 fine-tuned on OAS camelid

**Training (one-time, Phase 2 deliverable)**:
- Base: AbLang2 BSD-3 model
- Fine-tuning data: OAS camelid subset (CC BY 4.0)
- Method: continued masked-language-modeling on VHH sequences with ANARCI-aware position embeddings
- Compute: ~6–12 hours on H200
- Output: `models/ablang2_vhh_head/` checkpoint

**Inference**:
- Per-position pseudo-likelihood under fine-tuned model
- Geometric mean → VHH-nativeness score in [0, 1]
- Decomposable into framework vs CDR scores via ANARCI numbering

**Calibration**:
- Reference set: 18 therapeutic VHHs (Caplacizumab, Envafolimab, Ozoralizumab, Sonelokimab, ciltacabtagene heads, etc.) — these define Tier A range
- Random scrambled antibody sequences define bottom tier
- Tier thresholds set empirically; recalibrated quarterly

### 4.4 Humanization via BioPhi + VHH post-filter

**Path 1 (default)**: BioPhi/Sapiens with VHH-aware constraints
- BioPhi/Sapiens suggests humanization edits
- Post-filter (custom ~200 LOC):
  - **Freeze Kabat 37, 44, 45, 47** (hallmark)
  - **Freeze Cys22, Cys92**
  - **Block CDR1/2/3 edits** (CDR-graft style)
  - Re-score candidates with VHH-nativeness head; reject edits dropping nativeness > 0.1

**Path 2 (advanced, opt-in)**: Clean-room re-implementation of Llamanade algorithm from *Structure* 2022 paper. Algorithm rules (Vincke scaffold + per-position substitution table) are public scientific facts.

### 4.5 Thermostability (NbThermo-trained head)

- Linear regression / shallow MLP on AbLang2 embeddings → NbThermo Tm regression
- Output: `predicted_tm_celsius`
- Calibration: held-out NbThermo subset; require Pearson r > 0.6

### 4.6 Polyreactivity proxy

- Sequence-based PSR (polyspecificity reagent) reactivity prediction
- CDR3 chemistry: % aromatic, % positive charge, hydrophobicity patches
- Flag candidates with PSR-positive sequence features

### 4.7 T-cell epitope (MHC-II proxy)

NetMHCIIpan is excluded (license). Use:
- MHCflurry for MHC-I (Apache 2.0)
- For MHC-II: BLAST 15-mer windows against IEDB experimental positive set; flag > 50% identity windows
- **Explicit limitation noted in user-facing brief**

## Module 4.5 — Paratope sanity check (Paragraph)

**Purpose**: orthogonal validation that the predicted paratope is consistent with VHH-presented surface, independent of Boltz-2 complex prediction.

**Steps**:
1. Run **Paragraph** on VHH structure alone (heavy-chain-only mode)
2. Threshold P > 0.5 → `P_pred` = predicted paratope residue set
3. From Module 3 complex prediction: `P_obs` = VHH residues with target contact < 5 Å
4. Compute Jaccard: `concordance = |P_pred ∩ P_obs| / |P_pred ∪ P_obs|`
5. Compute CDR contact fraction and framework contact fraction
6. **Gate**:
   - PASS: concordance ≥ 0.5
   - PASS with warning: 0.3 ≤ concordance < 0.5 AND CDR contact ≥ 70%
   - FAIL: concordance < 0.3 OR framework contact ≥ 30%

**Output stored per candidate**:
```json
{
  "paratope_predicted_residues": [33, 50, 52, 99, 100, 103, 104, 105],
  "paratope_observed_residues": [33, 50, 99, 100, 103, 104, 105],
  "paratope_concordance_jaccard": 0.78,
  "observed_cdr_fraction": 0.86,
  "observed_framework_fraction": 0.14,
  "paratope_sanity_pass": true
}
```

**Calibration note**: Paragraph was trained primarily on mAb data. Phase 2 deliverable validates VHH heavy-only performance on SAbDab-nano. If false-rejection rate > 40%, reduce paratope_concordance weight in Module 8 ranking rather than disabling the module.

## Module 5 — CDR refinement (NNP-primary)

### 5.1 Production refinement

For each candidate passing Module 4 + 4.5 (~50 candidates):

**Equilibration (classical, fast)**:
- Solvate TIP3P/OPC, 10 Å padding
- 0.15 M NaCl; propka-assigned protonation at pH 7.0
- Histidine tautomers HID/HIE/HIP per local environment
- Cys22-Cys92 explicit disulfide bond
- Minimization → NVT 100 ps → NPT 1 ns with gradual restraint release
- Engine: OpenMM with Amber ff14SB; classical chosen for speed

**Production (NNP-MD)**:
- Engine: OpenMM + openmm-torch
- NNP auto-selected via decision tree below
- Duration: 20 ns screening tier, 50 ns refinement tier (top 10 promoted)
- Timestep: 1 fs (no SHAKE with NNP)
- Integrator: Langevin γ=1 ps⁻¹
- Ensemble: NPT (Monte Carlo barostat, 1 atm, 300 K)
- Coordinates saved every 100 ps

**NNP selection decision tree**:
```
1. Element check:
   If non-{H,C,N,O,F,P,S,Cl,Br,I} present → MACE-MP-0 (universal)
2. Interface character:
   If >2 salt bridges or >3 charged pairs → AIMNet2 (explicit charges)
   If aromatic π-stacking dominant → MACE-OFF (best aromatic/dispersion)
   If speed-critical (screening) → ANI-2x (fastest)
   Else → MACE-OFF (default)
3. System size:
   If NNP-region atoms > 8000 → hybrid (NNP interface + classical bulk)
```

**Trajectory analysis**:
- Interface RMSD (heavy atom, target-aligned)
- Per-residue contact persistence (heavy atom < 5 Å)
- Hydrogen bond persistence (Baker-Hubbard criteria)
- CDR3 conformational clustering (Cα RMSD, DBSCAN)

### 5.2 NNP-based binding energy (replaces classical MM/GBSA at top tier)

**Method**:
- 100 evenly-spaced frames from NNP-MD trajectory
- Per frame: E(complex), E(target alone, same coords), E(binder alone, same coords) via NNP
- `ΔE_bind = E(complex) − E(target) − E(binder)`
- Average over 100 frames

**Entropy correction**:
- Conformational: quasi-harmonic / NMA on representative frames
- Solvation: classical GBN2 surface term

**Cross-check with classical MM/GBSA on same trajectory** (sanity check):
- Flag candidates where NNP-ΔE and classical MM/GBSA rank differ by > 5 positions
- These are "polarization-sensitive" → trigger 5.3

### 5.3 QM/QM-MM verification (auto-triggered or opt-in)

**Triggers**:
- Automatic for polarization-sensitive candidates (per 5.2)
- Manual via `config.refinement.qm_enabled: true`

**Method**:
- First pass: GFN2-xTB via xtb-python on hotspot region (~1000 atoms)
- Critical case: ωB97X-D / def2-TZVP via PySCF on smaller region (< 100 atoms)
- Outputs: H-bond network energetics, polarization contribution, charge transfer, π-π / cation-π quantification

### 5.4 Validation against benchmarks (quarterly)

- PDBbind subset: 20 protein-protein complexes, NNP rank vs experimental KD; required Spearman ρ > 0.5
- Antibody-antigen SAbDab subset: required ρ > 0.4
- VHH-specific: caplacizumab-vWF, anti-HEWL VHH-HEWL; rank correctly vs published KD-mutant series

Failure → fall back to classical MM/GBSA + open issue.

**Output**: `refinement.parquet` per candidate with MD summary, NNP ΔE_bind, hotspots, persistent contacts.

## Module 6 — In silico affinity maturation

For top 5 candidates (configurable) from Module 5:

### 6.1 Saturation mutagenesis

- Positions: CDR1, CDR2, CDR3 residues; optional Vernier zone (Kabat 2, 27, 28, 29, 30, 48, 49, 67, 69, 71, 73, 78, 93, 94)
- Skip: hallmark V37/E44/R45/G47, conserved Cys22/Cys92
- Generate: N positions × 19 amino acids ≈ 700 candidates per parent

### 6.2 Tier 1 — Fast scoring

- ProteinMPNN log-likelihood ratio: P(mut | structure) / P(WT | structure)
- ESM-IF likelihood ratio
- Combined score: average of ranks
- Output: top 50 per parent

### 6.3 Tier 2 — NNP-MD ΔΔG (REPLACES classical MM/GBSA)

For top 50 mutations per parent:
- Apply mutation via OpenMM Modeller.add with rotamer library (open implementation; no MODELLER dependency)
- 1 ns classical equilibration
- **5 ns NNP-MD on mutant complex** (same NNP as Module 5.1)
- ΔE_bind via §5.2 method
- `ΔΔG_bind = ΔE_bind(mutant) − ΔE_bind(WT)`
- Always cross-check with classical MM/GBSA on same trajectory

**Cost**: ~5 hours/mutation on H200 → ~25 hrs for 50 mutations (parallelizable to ~6 hrs with 4 jobs)

### 6.4 Tier 3 — FEP (opt-in)

For top 10 mutations from Tier 2:
- Alchemical FEP via pmx + GROMACS
- 20 lambda windows, 5 ns/window, dt 2 fs with H-bond constraints
- BAR analysis via alchemlyb
- Charged mutations: counter-charge correction documented in code
- Cost: ~12 hrs/mutation; opt-in only

### 6.5 Combinatorial design

- Top 5 × top 5 = 25 double mutants
- Re-score with Tier 1 + Tier 2
- Optional Bayesian optimization over mutation space

**Output**: `matured_variants.parquet`: `parent_id, variant_id, mutations, ddg_proteinmpnn, ddg_nnp_mmgbsa, ddg_fep, predicted_kd_fold`

## Module 7 — Cross-reactivity / off-target screen

1. **Paralog identification**: BLAST target vs human SwissProt; identify > 30% identity in epitope region
2. **Structural similarity**: AlphaFold DB or ESMFold for top paralogs; structural alignment; rank by epitope-region similarity
3. **Boltz-2 spot check**: candidate VHH against top 5 paralogs; cross-reactivity score = max(ipTM_paralog) / ipTM_target
4. **Polyspecificity proxy**: CDR3 chemistry analysis; PSR-positive flag

**Output**: `cross_reactivity.json` per candidate (top paralogs, scores, tier)

## Module 8 — Pareto ranking + diversity

Objectives (all normalized to [0,1], maximize):
1. `iptm_mean`
2. `epitope_overlap`
3. `humanness_oasis_score` (BioPhi)
4. `vhh_nativeness` (AbLang2-OAS head)
5. `−liability_count`
6. `−ddg_proxy_esmif`
7. `mode_stability`
8. `nnp_de_bind` (from Module 5)
9. `−cross_reactivity_max_iptm_ratio` (Module 7)
10. `contact_persistence_md` (Module 5)
11. `paratope_concordance` (Module 4.5)
12. `predicted_tm_celsius` (Module 4.5 thermostability head)

**Algorithm**: NSGA-II non-dominated sorting → keep Pareto frontier rank 0 + 1 → diversity clustering on CDR3 (Levenshtein, threshold 0.7) → top 20 final.

## Module 9 — FTO post-check

For final ranked candidates (~20):
1. CDR3 BLAST against USPTO/EPO patent sequence corpus (monthly-updated local index, MMseqs2-backed)
2. Full VHH BLAST at lower stringency
3. Risk classification:
   - HIGH: identical CDR3 to claimed sequence
   - MEDIUM: > 85% CDR3 identity
   - LOW: > 70% full sequence identity
4. Report: patent ID, applicant, expiry, jurisdiction, claim excerpt
5. **Gate**: HIGH risk candidates removed from top recommendations (kept in DB with flag); user override possible

**Prominent disclaimer**: not legal advice; external IP attorney for final opinion.

## Module 10 — Experimental brief

Tier-coded scores (Tier A/B/C; raw scores in DB but not in user-facing markdown):

```yaml
candidate_id: NF_HER2_001
tier: A
sequence_aa: QVQLVES...
length: 124
parent_candidate: null
mutations_from_parent: []

provenance:
  source: denovo
  backbone_id: rfantibody_0287
  application_profile: therapeutic_strict
  generation_timestamp: 2026-05-24T12:30:00Z
  git_commit: a3f2d91...
  container_digests: {boltz2: "sha256:...", rfantibody: "sha256:...", ...}

scores_summary:
  binding_tier: A
  developability_tier: A
  humanness_tier: B
  vhh_nativeness_tier: A
  paratope_sanity_tier: A
  thermostability_tier: A
  cross_reactivity_tier: A
  fto_tier: A
  overall_tier: A

target_conformation:
  ensemble_method: esmflow
  binds_conformers: [c1, c3]              # robustness signal
  binds_all_conformers: false

refinement:
  md_engine: openmm
  nnp_model: mace_off
  md_simulation_ns: 50
  interface_rmsd_mean_angstrom: 1.42
  cdr3_dominant_cluster_occupancy: 0.78
  hotspot_residues:
    - {residue: "Y32", contribution_kcal: -3.2}
    - {residue: "W104", contribution_kcal: -2.8}
    - {residue: "R97", contribution_kcal: -2.1}
  nnp_de_bind_kcal: -41.2                  # NNP energy (ranking only)
  mmgbsa_dg_crosscheck_kcal: -38.7        # classical MM/GBSA sanity
  persistent_contacts:
    - {vhh: "Y32", target: "E312", type: "hbond", persistence: 0.91}
    - {vhh: "W104", target: "F315", type: "pi_pi", persistence: 0.87}

paratope_sanity:
  paragraph_concordance: 0.84
  cdr_contact_fraction: 0.86
  pass: true

maturation_suggestions:
  - {mutation: "S52T", ddg_nnp_mmgbsa: -0.8, ddg_fep: null, evidence: "MM/GBSA + ProteinMPNN agree"}
  - {mutation: "G55A", ddg_nnp_mmgbsa: -0.6, ddg_fep: null, evidence: "stabilizes CDR2 conformation"}

cross_reactivity:
  paralog_check:
    - {target: "HER3", risk: "low", iptm_ratio: 0.31}
    - {target: "HER4", risk: "low", iptm_ratio: 0.18}
  polyspecificity_proxy: 0.22

fto:
  cdr3_match: null
  full_seq_top_match:
    patent_id: "US10000000A1"
    identity_pct: 67
    risk: "low"
  external_legal_review_required: false

experimental_brief:
  expression:
    host_primary: "E. coli BL21(DE3)"
    vector: "pET-28a(+) (His6, T7 promoter)"
    dna_codon_optimized: ATG...
    expected_yield_mg_per_l: "5-20"
  characterization:
    affinity_assay: "BLI (Octet) with biotinylated target"
    counterscreens:
      - {name: "Anti-GFP VHH", purpose: "negative control"}
      - {name: "HER3 ECD", purpose: "paralog selectivity"}
      - {name: "PSR panel", purpose: "polyspecificity"}
    stability_assay: "DSF (NanoDSF preferred)"
    target_tm_celsius: 68  # predicted; verify experimentally
  if_validates:
    - "Test top 3 affinity-matured variants"
    - "Build bivalent format (VHH-15GS-VHH) for avidity"
    - "If therapeutic: anti-HSA VHH fusion for PK extension"
```

Output in both JSON (machine-readable) and Markdown (lab-ready).

---

# 6. GPU resource planning (H200, 141 GB HBM3e)

| Stage | VRAM peak | Time per target |
|---|---|---|
| Module 0: FTO precheck | < 5 GB | 1–5 min |
| Module 1: Target single-state | 50 GB | 15–30 min |
| Module 1: Ensemble (ESMFlow-MD+Templates 12l-distilled, 50 samples) | 30 GB | ~15 min |
| Module 1: Ensemble (NNP-MD 50 ns) | 30 GB | 3–6 hrs |
| Module 2a: Library screen (10k → 200) | 80–120 GB | 10–15 hrs |
| Module 2b: De novo (RFantibody, 1000 backbones) | 30–40 GB | 8–10 hrs |
| Module 3: Validation (Boltz-2 5-seed + Chai-1 top 50) | 80–120 GB | 7–10 hrs |
| Module 4: Developability + humanness | 20 GB | 1–2 hrs |
| Module 4.5: Paratope sanity (Paragraph) | < 5 GB | < 30 min |
| Module 5.1: NNP-MD production (50 × 20 ns) | 30 GB | ~25 hrs serialized; ~8 hrs with 4 parallel jobs |
| Module 5.1: Refinement tier (10 × 50 ns) | 30 GB | ~10 hrs |
| Module 5.2: NNP ΔE_bind ensemble | 20 GB | included in 5.1 |
| Module 5.3: QM verification (auto-triggered, ~10%) | 20 GB | 2–8 hrs |
| Module 6 Tier 1 (fast scoring 700×5) | 10 GB | < 1 hr |
| Module 6 Tier 2 (NNP-MD top 50 × 5 ns × 5 parents) | 30 GB | ~50 hrs serialized; ~15 hrs parallelized |
| Module 6 Tier 3 FEP (opt-in, ~10 muts) | 20 GB | ~120 hrs (off by default) |
| Module 7: Cross-reactivity | 60 GB | 2–4 hrs |
| Module 8: Ranking | < 5 GB | minutes |
| Module 9: FTO post-check | < 5 GB | 10–30 min |
| Module 10: Brief generation | < 5 GB | minutes |
| **Total default (no QM/FEP)** | | **~75–90 hrs per target** |
| **Total fast_mode** | | **~30 hrs per target** |
| **Total therapeutic_strict (with QM + FEP)** | | **~250 hrs per target** |

**Fast_mode** (`config.fast_mode: true`):
- Skip ensemble (Module 1)
- Library 2000 candidates
- RFantibody 300 backbones
- Validation 3 seeds
- Module 5 MD 10 ns (no NNP refinement tier)
- Module 6 Tier 2 NNP-MD 2 ns, skip Tier 3
- Module 7 sequence-only paralog (no Boltz-2 spot check)

**Wet-lab capacity matching**: `max_candidates_refined: 50` default; only top 5 promoted to Module 6. Configurable.

---

# 7. Implementation phases (17 weeks)

## Phase 0 — Foundation (week 1)
- Project skeleton, license register, CI with `pip-licenses` + `scancode-toolkit`
- Base Apptainer container (`base_cuda.def`)
- PostgreSQL schema initialized via Alembic
- Nextflow scaffolding with module stubs
- ADR template + first 5 ADRs (0001–0005)

## Phase 1 — Target prep + ensemble + library route (weeks 2–4)
- Module 0, Module 1 (with ESMFlow/AlphaFlow adapter as primary ensemble path)
- Build VHH library from OAS + INDI
- Boltz-2 adapter; Module 2a
- Smoke test: HEWL target → recover known anti-HEWL VHH
- Calibration benchmark: HER2 ECD → 2Rs15d/11A4 retrieval in top 200

## Phase 2 — Validation + filters + humanness training (weeks 5–8)
- Module 3 with Boltz-2 ensemble + Chai-1
- Module 4 with liability scans + biophysical scoring + thermostability head training
- **AbLang2 fine-tuning on OAS camelid** (~12 hours one-time)
- VHH-nativeness head calibration against 18 therapeutic VHH set
- BioPhi/Sapiens humanization integration + VHH post-filter
- Module 4.5 Paragraph integration + SAbDab-nano calibration
- Module 7 sequence-only paralog screen (Boltz-2 spot check deferred to Phase 6)

## Phase 3 — De novo route (RFantibody) (weeks 9–11)
- RFantibody adapter (Apptainer container + Python wrapper)
- Module 2b end-to-end
- Compare against legacy RFdiffusion+ProteinMPNN (FreeBindCraft) on benchmarks
- Calibrate `n_backbones` for production use (start 1000, scale per success rate)

## Phase 4 — CDR refinement (NNP-primary) (weeks 12–14)
- Module 5.1 with OpenMM equilibration + MACE-OFF/AIMNet2/ANI-2x NNP-MD
- openmm-torch integration
- PDBbind subset benchmark calibration (Spearman > 0.5 required)
- Module 5.2 NNP ΔE_bind ± classical MM/GBSA cross-check
- Module 5.3 QM verification path (xtb + PySCF)
- Gate: lysozyme-HEWL benchmark must reproduce published hotspots within 50% energy contribution

## Phase 5 — Maturation (weeks 15–16)
- Module 6 Tier 1 (ProteinMPNN + ESM-IF)
- Module 6 Tier 2 (NNP-MD ΔΔG with cross-check)
- Module 6 Tier 3 FEP via pmx + GROMACS (opt-in, configurable)
- Lysozyme T4 stability mutant benchmark for FEP

## Phase 6 — FTO + brief + UI (week 17)
- USPTO patent bulk indexing (one-time ~500 GB)
- EPO OPS API integration
- Module 0 + Module 9
- Module 7 Boltz-2 spot check enabled
- Module 10 brief generation with tier coding
- FastAPI + Streamlit minimal UI
- End-to-end integration test on full benchmark suite

## Phase 7+ — Operations & feedback (ongoing)
- Wet-lab feedback ingestion API
- Quarterly retrospective: model scores vs experimental outcome
- License audit per release
- Quarterly SOTA model review (no auto-upgrade; explicit migration with regression benchmark)
- Fine-tuning on accumulated wet-lab data when >100 results available

---

# 8. Configuration

## 8.1 `configs/default.yaml`

```yaml
run:
  name: null
  output_dir: ./runs
  random_seed: 42
  gpu: 0
  fast_mode: false
  application_profile: general

target:
  source: fasta
  fasta_path: null
  uniprot_id: null
  assembly_state: auto
  user_epitope_residues: null
  ensemble: esmflow                                # off | esmflow | alphaflow | af_cluster | nnp_md
  flow_model: esmflow_md_templates_12l_distilled
  flow_n_samples: 50
  flow_tmax: 1.0
  flow_steps: 10
  flow_use_template: true
  glycan_modeling: minimal                         # off | minimal | full
  conservation_filter: medium

library:
  vhh_db_path: ./data/vhh_library.parquet
  n_candidates: 10000
  cdr3_length_range: [13, 24]
  cluster_identity_threshold: 0.9
  prefilter_engine: none                           # none | boltz2_fast_screen

denovo:
  enabled: true
  engine: rfantibody                               # rfantibody (default) | rfdiffusion_generic (legacy ablation)
  n_backbones: 1000
  n_sequences_per_backbone: 8
  cdr_length_distribution: natural_vhh
  framework_template: vincke_universal
  rosettafold2_filter: true
  hallmark_postcheck: true
  enable_freebindcraft_ablation: false

validation:
  boltz2_seeds: 5
  chai1_topk: 50
  iptm_threshold: 0.7
  pae_interface_threshold: 6.0
  paratope_cdr_min: 0.7
  epitope_overlap_min: 0.8
  mode_stability_min: 3

developability:
  reject_on: [free_cys]
  flag_only:
    - deamidation_NG_cdr
    - isomerization_DG_cdr
    - met_oxidation_cdr
    - trp_oxidation_cdr
    - n_glyc_cdr
  net_charge_range: [4.0, 9.5]
  patch_size_max: 50
  thermostability_engine: ablang2_nbthermo_head
  tm_min_celsius: 60

humanness:
  oasis_threshold: 0.6
  vhh_nativeness_engine: ablang2_oas_finetuned
  vhh_nativeness_model_path: ./models/ablang2_vhh_head/
  vhh_nativeness_threshold: 0.5

humanization:
  enabled: true
  engines: [biophi_sapiens_with_vhh_constraints]
  preserve_hallmark: true
  preserve_cdr: true
  combine_strategy: pareto_humanness_vs_nativeness

paratope_sanity:
  enabled: true
  predictor: paragraph
  concordance_min: 0.5
  cdr_contact_fraction_min: 0.7
  framework_contact_fraction_max: 0.3
  reject_on_fail: true

refinement:
  enabled: true
  max_candidates: 50
  production_engine: nnp                           # nnp | classical | hybrid
  production_ns: 20
  refinement_tier_ns: 50
  refinement_tier_n: 10
  nnp_model: auto                                  # auto | mace_off | mace_mp_0 | ani2x | aimnet2
  nnp_timestep_fs: 1.0
  nnp_max_atoms_full: 8000
  nnp_interface_radius_angstrom: 5.0
  equilibration_engine: classical
  equilibration_forcefield: amber14sb
  equilibration_water: tip3p
  binding_energy_method: nnp_primary               # nnp_primary | mmgbsa_primary | both_consensus
  bind_e_n_frames: 100
  always_run_mmgbsa_crosscheck: true
  polarization_sensitivity_threshold_kcal: 5.0
  qm_enabled: false                                # auto-trigger on polarization-sensitive
  qm_method: xtb

maturation:
  enabled: true
  top_n_to_mature: 5
  mutation_positions: cdr_all
  max_mutations_per_candidate: 50
  tier1_top: 50
  tier2_top: 20
  tier3_fep: false
  tier2_method: nnp_mmgbsa_hybrid
  tier2_nnp_ns: 5
  combinatorial: true
  combinatorial_pairs: 25

cross_reactivity:
  enabled: true
  paralog_identity_threshold: 0.3
  boltz_spot_check: true
  spot_check_top_paralogs: 5

fto:
  enabled: true
  pre_check_indication_lookup: true
  post_check_cdr3_blast: true
  cdr3_match_threshold: 0.85
  full_seq_match_threshold: 0.70
  patent_db: ./data/patents
  external_legal_review_warning: true

ranking:
  pareto_ranks_kept: [0, 1]
  diversity_clustering_threshold: 0.7
  top_n_final: 20

brief:
  expression_hosts: [ecoli_bl21, pichia_gs115]
  tier_coding: true
  generate_markdown: true
  generate_json: true
```

## 8.2 Profile override examples

`configs/profiles/therapeutic_strict.yaml`:
```yaml
fto:
  external_legal_review_warning: true
  pre_check_blocking: true

cross_reactivity:
  paralog_identity_threshold: 0.25

humanness:
  oasis_threshold: 0.75
  vhh_nativeness_threshold: 0.65

refinement:
  qm_enabled: true
  refinement_tier_ns: 100

maturation:
  tier3_fep: true
```

`configs/profiles/aav_capsid.yaml`:
```yaml
target:
  ensemble: nnp_md                                 # receptors with conformational states
  glycan_modeling: full

library:
  cdr3_length_range: [12, 20]                      # shorter for transcytosis

cross_reactivity:
  custom_paralogs: [TfR1, LRP1, LRP8, BCAM]

brief:
  custom_assays:
    - "in vitro BBB transcytosis (hCMEC/D3 monolayer)"
    - "in vivo BBB transit (mouse IV + brain ELISA)"
```

---

# 9. Database schema (PostgreSQL)

```sql
-- Core entities
CREATE TABLE targets (
    target_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    uniprot_id TEXT,
    sequence TEXT NOT NULL,
    structure_path TEXT,
    plddt_mean FLOAT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE target_conformers (
    conformer_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    target_id UUID REFERENCES targets,
    structure_path TEXT NOT NULL,
    ensemble_method TEXT NOT NULL,                  -- esmflow | alphaflow | af_cluster | nnp_md
    cluster_representative INT,
    cluster_size INT,
    plddt_mean FLOAT,
    epitope_map JSONB
);

CREATE TABLE runs (
    run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    target_id UUID REFERENCES targets,
    config JSONB NOT NULL,
    status TEXT NOT NULL,
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    git_commit TEXT,
    container_digests JSONB
);

CREATE TABLE candidates (
    candidate_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID REFERENCES runs,
    source TEXT NOT NULL,                           -- library | denovo_rfantibody | freebindcraft_ablation
    sequence TEXT NOT NULL,
    cdr1 TEXT, cdr2 TEXT, cdr3 TEXT,
    structure_path TEXT,
    complex_path TEXT,
    scores JSONB NOT NULL,
    liabilities JSONB,
    risk_tier TEXT,
    pareto_rank INT,
    final_rank INT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Module-specific results
CREATE TABLE paratope_sanity_results (
    paratope_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    candidate_id UUID REFERENCES candidates,
    paragraph_predicted_residues INT[],
    observed_residues INT[],
    concordance_jaccard FLOAT,
    cdr_contact_fraction FLOAT,
    framework_contact_fraction FLOAT,
    sanity_pass BOOLEAN
);

CREATE TABLE refinement_results (
    refinement_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    candidate_id UUID REFERENCES candidates,
    md_engine TEXT,
    md_ns FLOAT,
    nnp_model TEXT,
    interface_rmsd_mean FLOAT,
    interface_rmsd_std FLOAT,
    cdr3_cluster_occupancy FLOAT,
    nnp_de_bind_kcal FLOAT,
    mmgbsa_dg_kcal FLOAT,
    polarization_sensitive BOOLEAN,
    persistent_contacts JSONB,
    hotspot_residues JSONB,
    qm_used BOOLEAN DEFAULT FALSE,
    qm_method TEXT,
    trajectory_path TEXT
);

CREATE TABLE matured_variants (
    variant_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    parent_candidate_id UUID REFERENCES candidates,
    mutations TEXT NOT NULL,
    sequence TEXT NOT NULL,
    ddg_proteinmpnn FLOAT,
    ddg_nnp_mmgbsa FLOAT,
    ddg_classical_mmgbsa_crosscheck FLOAT,
    ddg_fep FLOAT,
    ddg_fep_error FLOAT,
    predicted_kd_fold FLOAT,
    tier TEXT
);

CREATE TABLE cross_reactivity_results (
    crossreact_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    candidate_id UUID REFERENCES candidates,
    paralog_results JSONB,
    psr_proxy_score FLOAT,
    cross_reactivity_tier TEXT
);

CREATE TABLE fto_results (
    fto_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    candidate_id UUID REFERENCES candidates,
    cdr3_match JSONB,
    full_seq_match JSONB,
    risk_tier TEXT,
    external_review_required BOOLEAN,
    checked_at TIMESTAMPTZ DEFAULT now()
);

-- Wet-lab feedback
CREATE TABLE experimental_results (
    result_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    candidate_id UUID REFERENCES candidates,
    expression_yield_mg_per_l FLOAT,
    binding_observed BOOLEAN,
    kd_nm FLOAT,
    kon FLOAT,
    koff FLOAT,
    tm_celsius FLOAT,
    notes TEXT,
    submitted_at TIMESTAMPTZ DEFAULT now(),
    submitted_by TEXT
);

-- Audit
CREATE TABLE license_register (
    dep_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    version TEXT,
    license_spdx TEXT NOT NULL,
    commercial_ok BOOLEAN NOT NULL,
    verification_method TEXT,                       -- "repo LICENSE inspected" | "model card inspected" | etc.
    obligations TEXT,
    last_audited TIMESTAMPTZ NOT NULL DEFAULT now(),
    auditor TEXT
);

-- Indexes
CREATE INDEX idx_candidates_run ON candidates(run_id);
CREATE INDEX idx_candidates_rank ON candidates(run_id, final_rank);
CREATE INDEX idx_candidates_source ON candidates(source);
CREATE INDEX idx_conformers_target ON target_conformers(target_id);
CREATE INDEX idx_matured_parent ON matured_variants(parent_candidate_id);
```

---

# 10. Critical implementation notes

## 10.1 General

- Adapter pattern strictly: each external model wrapped in `adapters/<name>.py` with typed pydantic interface. No model-specific code outside `adapters/`.
- Schemas first: define JSON schema for every inter-module artifact before writing modules.
- Every adapter supports `dry_run=True` for testing without GPU.
- Reproducibility: pin all container digests by SHA256; record git commit + container digests + seeds + resolved config in every run.

## 10.2 NNP integration gotchas

- **Element coverage**: MACE-OFF {H,C,N,O,F,P,S,Cl,Br,I}; AIMNet2 wider with explicit charges; ANI-2x narrowest {H,C,N,O,S,F,Cl}. Pipeline element-checks first.
- **Timestep**: 1 fs default (NNP-MD); SHAKE not safe with non-harmonic potentials. Drop to 0.5 fs if energy drift > 0.5 kcal/mol/ns.
- **Thermostat**: Langevin γ=1 ps⁻¹ safe. **NOT Berendsen** (wrong ensemble). Test Nosé-Hoover before use.
- **Barostat**: Monte Carlo (no volume-derivative needed).
- **Hybrid MD boundary**: NVE 1 ns drift test required before production; drift < 2 kcal/mol/ns expected.
- **Mixed precision**: bf16 inference OK on H200; check NaN in forces; fall back to fp32 if encountered.
- **NNP energy comparison**: same model across all calculations within a candidate; never mix MACE-OFF vs AIMNet2 outputs.
- **Checkpoint pinning**: NNP weights pinned by SHA256 in container manifest.

## 10.3 MD setup gotchas

- Protonation at pH 7.0 default; adjust per biological context (lysosomal pH 5, etc.)
- Histidine tautomers HID/HIE/HIP explicitly per local environment
- Cys22-Cys92 disulfide explicit in topology
- 0.15 M NaCl (not zero); affects surface electrostatics
- Termini: ACE/NME capping unless free termini biologically relevant
- Trajectory storage: save every 100 ps + final state; on-demand replay for analysis

## 10.4 FEP gotchas (Module 6.4)

- pmx free-energy topologies (state A + state B)
- 20 lambda windows minimum, 5 ns/window
- Convergence check: BAR error < 0.5 kcal/mol; reject windows with sub-1% acceptance
- Charged mutations: counter-charge correction documented

## 10.5 RFantibody specifics

- Framework hallmark preservation native; post-hoc validation defensive
- CDR3 length from natural VHH distribution built-in
- RoseTTAFold2 fold-back validation integrated
- Container based on official Docker; pin to release tag by git SHA
- Framework template: Vincke universal scaffold default; allow user override

## 10.6 AlphaFlow/ESMFlow specifics

- Distilled 12l models recommended default (2.5x faster, small accuracy loss)
- ESMFlow no MSA needed (orphan proteins OK)
- AlphaFlow needs MSA (use ColabFold-search MMseqs2)
- Templates: provide static structure prediction as anchor for MD+Templates variants
- tmax=1.0 full diversity; lower for precision-mode
- 10 steps default; fewer for speed

## 10.7 Patent FTO gotchas

- USPTO bulk ~500 GB indexed; monthly cron update
- Sequence claims in supplementary XML; parse carefully
- Claim scope parsing NOT reliable; output informational only with disclaimer
- For Korean filings: KIPRIS integration deferred to Phase 7+
- Never make legal conclusions; surface for human review

## 10.8 Tier classifier calibration

- Define empirically from benchmark target distribution
- Initial: Tier A top 20%, B next 40%, C bottom 40%
- Quarterly recalibration as wet-lab data accumulates
- NEVER expose raw numerical scores in user-facing brief

## 10.9 Boltz-2 batching

- Pre-compute target MSA once; cache it
- VHH single-domain — no MSA needed for binder
- On H200: 4–6 parallel sequences saturate HBM

## 10.10 Conformational ensemble downstream

- Each conformer becomes separate epitope-mapping target
- Downstream tracks `conformer_id` provenance
- Candidates binding multiple conformers preferred (robustness signal recorded in DB)

---

# 11. Testing strategy

## 11.1 Unit tests
- Every adapter: mock outputs, validate parsing
- `scoring/liabilities.py`: parametrized synthetic sequences
- `numbering.py`: ANARCI on known VHHs
- `codon_optimize.py`: round-trip + CAI > 0.8

## 11.2 Integration tests
- Smoke test on synthetic 50-aa target → pipeline completes < 30 min
- Lysozyme (HEWL) end-to-end with limited library → recover cAbBcII10-like hits

## 11.3 Benchmark suite (must pass for release)

`tests/benchmarks/` contains target + known-binder ground truth:

| Target | Known binders | Pass criterion |
|---|---|---|
| HEWL | cAbBcII10 | rank top 50 from 10k library |
| HER2 ECD | 2Rs15d, 11A4 | rank ≥1 in top 50 |
| BCMA | cilta-cel parent | rank top 50 |
| vWF A1 | caplacizumab | rank top 50 |
| TNF-α | ozoralizumab heads | rank top 50 |
| GFP | multiple in SAbDab | rank top 50 |
| SARS-CoV-2 RBD | Ty1, etc. | ensemble-target test |

De novo: produce ≥ 5 candidates with ipTM > 0.8 and canonical-epitope paratope.

Refinement: identify known hotspot residues in top 5 contributors.

## 11.4 NbBench validation (Phase 2 gate)
- VHH-specific classification tasks
- AbLang2-OAS head must outperform vanilla ESM-2 on VHH tasks

## 11.5 NNP benchmarks (Phase 4 gate)
- PDBbind protein-protein subset: Spearman ρ > 0.5
- Antibody-antigen SAbDab: ρ > 0.4
- Failure → fall back to classical MM/GBSA, open issue

## 11.6 FEP validation (Phase 5 gate, if enabled)
- Lysozyme T4 stability mutant: published ΔΔG recovered within 1 kcal/mol

## 11.7 Cross-validation
- Hold out known binder; verify re-discovery
- Random CDR3 scrambling: WT > mutants in ranking

## 11.8 Reproducibility tests
- Same input + seeds + containers → bit-identical scores (CPU) / numerically-close (GPU)
- Regression alert on deviation

## 11.9 Continuous benchmarks (CI)
- License audit: `pip-licenses` + `scancode-toolkit`
- NbBench on every PR affecting Module 4 / 4.5
- PDBbind benchmark on every PR affecting Module 5

---

# 12. Architecture Decision Records (summary index)

Full ADR texts in `adrs/` directory. Summary:

| ID | Title | Status |
|---|---|---|
| 0001 | Use Boltz-2 + Chai-1 instead of AlphaFold3 | accepted (license) |
| 0002 | Exclude PyRosetta; use FreeBindCraft fork | accepted |
| 0003 | IEDB-identity proxy instead of NetMHCIIpan | accepted (license) |
| 0004 | Tier-coded output (Tier A/B/C) instead of raw scores | accepted |
| 0005 | NNP as primary CDR refinement engine | accepted (supersedes v2's classical-MD plan) |
| 0006 | FEP gated behind `tier3_fep: true` (cost) | accepted |
| 0007 | USPTO patent index for FTO (vs commercial DB) | accepted |
| 0008 | Defer AbNatiV until license clarified | accepted |
| 0009 | Wet-lab artificial bottleneck (top 5 for refinement) | accepted |
| 0010 | Application profile system | accepted |
| 0011 | Use RFantibody as primary de novo engine | accepted (Nov 2025 MIT release) |
| 0012 | (SUPERSEDED by 0019) | superseded |
| 0013 | Paragraph for orthogonal paratope sanity check | accepted |
| 0014 | Exclude Ibex (NC weights) | accepted (license) |
| 0015 | Exclude nanoBERT (CC BY-NC-SA 4.0) | accepted (license) |
| 0016 | Exclude Llamanade (MODELLER dependency) | accepted (transitive license) |
| 0017 | Exclude NABP-BERT (no LICENSE file) | accepted (license) |
| 0018 | Add AlphaFlow / ESMFlow for conformational ensemble | accepted |
| 0019 | AbLang2 + OAS camelid for VHH-nativeness (replaces 0012) | accepted |

---

# 13. Known limitations (user-facing)

State prominently in README:

1. **Affinity values are relative, not absolute.** All affinity scores (Boltz-2, NNP ΔE, MM/GBSA, FEP) provide ranking only. Absolute KD prediction unreliable.
2. **De novo wet-lab success rate ~5–15%.** RFantibody specifically targets rigorous quality over hit count: expect ~1–2% per design, but cryo-EM-validated.
3. **MHC-II / T-cell epitope is approximate** (IEDB-identity proxy, not NetMHCIIpan). For therapeutic IND-enabling, license NetMHCIIpan or EpiVax separately.
4. **Target preprocessing limitations**: monomeric soluble globular targets work best. Membrane/glycosylated/large multimers see degraded accuracy.
5. **VHH-nativeness in-house calibrated** (AbLang2 + OAS camelid); not equivalent to commercial alternatives.
6. **FTO module screening only, NOT legal advice**: external IP attorney required.
7. **MD refinement single binding pose**: alternative poses not explored.
8. **FEP costs scale with mutation count**: Tier 3 opt-in only.
9. **Cross-reactivity limited to known paralogs**: orphan off-targets not predicted.
10. **GPU memory pressure**: targets > 500 aa may OOM in Module 1 + 3; auto-fallback to ESMFold reduces accuracy.
11. **Model version drift**: containers pinned by SHA256; manual upgrade with regression benchmark required.
12. **Paragraph mAb-trained**: VHH heavy-only mode validated in Phase 2; weight adjusted if false-rejection > 40%.
13. **NNP element coverage**: standard biomolecular + halogens; metal-containing targets need MACE-MP-0 with reduced validation.
14. **NNP ΔE is energy, not free energy**: entropy correction via classical proxy.
15. **RFantibody benefits from ~9000 designs**: smaller batches reduce success rate.
16. **AlphaFlow/ESMFlow distilled models trade accuracy for speed**: switch to full models for critical targets.

---

# 14. Codex CLI execution guidance

## 14.1 Implementation order discipline

1. **Containers first, code second**: get all model containers reproducible before pipeline logic.
2. **Adapter pattern strictly**: no model-specific code outside `adapters/`.
3. **Schemas first**: pydantic v2 for every inter-module artifact.
4. **Test with mini-benchmark before scaling**: implement Phase 1 fully with lysozyme target before Phase 2.
5. **Every adapter supports `dry_run=True`** for testing without GPU.

## 14.2 License audit policy

- Every new dep requires entry in `LICENSES/REGISTER.md`
- Every entry specifies verification method ("repo LICENSE inspected at SHA X" or "model card inspected at HF revision Y")
- CI runs `pip-licenses` + `scancode-toolkit` on every PR
- Quarterly re-verification of license status (licenses can change)

## 14.3 Critical "do NOT" list

1. Do NOT implement substitutes for excluded tools (NetMHCIIpan etc.) by scraping or reverse engineering.
2. Do NOT add a new dep without LICENSE register entry.
3. Do NOT modify ADRs retroactively; supersede with new ADR.
4. Do NOT trust NNP energies without classical MM/GBSA cross-check.
5. Do NOT expose raw scores in user-facing brief (Tier coding only).
6. Do NOT auto-upgrade model versions; explicit migration with benchmark.
7. Do NOT make legal conclusions in FTO module output.
8. Do NOT use SHAKE constraints with NNP-MD.
9. Do NOT mix NNP models within a single candidate analysis.
10. Do NOT skip Cys scanner (free Cys = HIGH reject).

## 14.4 Phase gate checks

Each phase has gating benchmarks (Section 11). **Do not proceed to next phase before previous gate passes.** Use git tags `phase-{N}-passed` to mark progress.

## 14.5 Code generation discipline

When Codex generates new code:
- Write draft ADR for any deviation from this spec
- Add tests before merging (especially for scoring/numbering)
- Profile GPU usage; flag if SM utilization < 70% under expected load
- Document `dry_run` behavior in adapter docstring

---

# 15. References

(citations for code attribution)

- Boltz-2: Passaro et al. 2025
- Chai-1: Chai Discovery 2024
- RFantibody: Bennett et al. 2025 (Nature)
- RFdiffusion: Watson et al. 2023 (Nature)
- ProteinMPNN: Dauparas et al. 2022 (Science)
- LigandMPNN: Dauparas et al. 2023
- AlphaFlow / ESMFlow: Jing et al. 2024 (ICML)
- OpenFold: Ahdritz et al. 2024 (Nat Methods)
- BindCraft: Pacesa et al. 2024 (Nature)
- ImmuneBuilder / NanoBodyBuilder2: Abanades et al. 2023 (Comm. Biol.)
- IgFold: Ruffolo et al. 2023 (Nat Commun)
- AbLang2: Olsen et al. 2024
- ESM-2 / ESMFold: Lin et al. 2023 (Science)
- BioPhi / Sapiens: Prihoda et al. 2022 (MAbs)
- Paragraph: Chinery et al. 2023 (Bioinformatics)
- MACE: Batatia et al. 2023 (NeurIPS); Kovács et al. 2025 (MACE-OFF)
- ANI-2x: Devereux et al. 2020 (JCTC)
- AIMNet2: Anstine et al. 2024 (JACS Au)
- GFN2-xTB: Bannwarth et al. 2019 (JCTC)
- pmx: Gapsys et al. 2015 (JCC)
- MM/GBSA: Genheden & Ryde 2015 (Expert Opin DD)
- propka: Olsson et al. 2011 (JCTC)
- ColabFold-search: Mirdita et al. 2022 (Nat Methods)
- ANARCI: Dunbar & Deane 2016
- OAS: Olsen et al. 2022 (Protein Sci.)
- INDI: Deszynski et al. 2022
- NbThermo: Pereira et al. (NbThermo paper)
- Vincke universal scaffold: Vincke et al. 2009 (JBC)
- Hamers-Casterman discovery: Hamers-Casterman et al. 1993 (Nature)
- Caplacizumab: Peyvandi et al. 2016 (NEJM)
- Cilta-cel: Berdeja et al. 2021 (Lancet)

---

End of NanoForge v3 spec.
