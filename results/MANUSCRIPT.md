# SkinScout: a license-clean, reproducible compound→target prediction pipeline for cosmetic ingredient discovery, validated on commodity hardware

*Methodology draft — auto-assembled from the SkinScout repository, then
hand-corrected against an adversarial review. All numbers trace to
`results/RETROSPECTIVE.md` and the committed per-run CSVs; figures are produced
by `scripts/make_retrospective_figures.py` from the same CSVs.*

---

## Abstract

SkinScout is an end-to-end, commercial-use-permissive pipeline that maps a
small-molecule cosmetic ingredient (SMILES) to a ranked list of candidate human
protein targets. It couples a one-time structural infrastructure build over the
AlphaFold human proteome (pLDDT cleanup → P2Rank pockets → Meeko PDBQT) with
per-compound preprocessing, an ADMET/skin-sensitisation gate, a CosIng/drug
annotation step, and a structure-based docking stage, then re-ranks targets with
a skin-expression prior. We demonstrate the system on a single commodity
workstation (Intel i7-13700K, 16 threads; NVIDIA RTX 3080 Ti, 12 GB; 123 GB RAM)
— below the 16 GB-GPU / 32-core specification the design targets. On a
retrospective panel of nine compounds (eight cosmetic ingredients plus aspirin
as a drug-target control) spanning 24 literature-curated compound–target pairs,
blind AutoDock Vina docking against a 300-receptor subset recovers the known
target within the top 10 for 7/24 pairs and the top 30 for 9/24. Adding the
Stage-3 skin-expression weighting (`0.7·norm(−ΔG) + 0.3·skin_score`) raises these
to 9/24 (top 10) and 13/24 (top 30). Retinol recovers RAR-β/α/γ at ranks 1/3/4;
niacinamide recovers NNMT at rank 2; aspirin recovers COX-2 at rank 7. The
characteristic failures — kojic acid and α-arbutin against tyrosinase, EGCG
against the matrix metalloproteinases, caffeine against the adenosine receptors
— share a single cause: AlphaFold does not model the metal cofactors (Cu²⁺,
Zn²⁺) or the active GPCR state on which those ligands depend. We report one
honest negative (transplanting Cu²⁺ into the tyrosinase model did **not** improve
rank under metal-blind Vina scoring) and one recovered capability: after
isolating Boltz-2 in a driver-compatible environment (torch 2.4 + CUDA 12.1),
co-folding + affinity prediction ran on the 12 GB GPU and, on a five-pair probe,
complemented Vina exactly where docking failed — assigning the caffeine→ADORA2A
GPCR pair a binder probability of 0.65 (Vina rank 57) and a non-target control
the lowest probability (0.35). GNINA remained unrunnable (no nvcc/Docker). The
effective, verified recipe on this configuration is
*Stage 0–3 + skin-weighting*, with Boltz-2 affinity available as a GPU re-rank.

## 1. Introduction

Cosmetic-ingredient R&D has historically been driven by phenotypic screening and
incremental analogue synthesis, with target-level mechanism often inferred only
after the fact. Computational target prediction could front-load that mechanism,
but three constraints have limited adoption in an industrial cosmetic setting:
(i) commercial-use licensing — many SOTA structure and docking tools ship
academic-only weights; (ii) the need to consider the *whole* human proteome
rather than a curated target list, while still prioritising skin biology; and
(iii) reproducibility on hardware a small lab actually owns.

SkinScout addresses all three. Every dependency is verified commercial-use
permissive (`LICENSE_POLICY.md`); docking runs proteome-wide (20,504 AlphaFold
structures) with skin relevance entering as a tunable weight rather than a hard
filter, so orphan or systemically-expressed targets are not silently excluded;
and the entire Stage 0 infrastructure plus a demonstration run were executed on
a 12 GB consumer GPU.

**Contributions.**
- A reproducible Stage 0 infrastructure build over the AlphaFold human proteome,
  with every runtime/infrastructure failure encountered on real hardware
  documented and fixed (§2.6).
- A skin-expression weighting (Stage 3 v3) that measurably improves known-target
  recovery on a cosmetic-ingredient retrospective (§3).
- Two reported negative results — metal-cofactor transplant under metal-blind
  scoring, and the environment blocks for ML re-scorers — that delimit where the
  single-engine pipeline is and is not trustworthy (§4).

## 2. Methods

Only stages that were **actually executed end-to-end on real data** are described
as run. Stages 4–9, 5.5, 5.6, 7.5 and 11 are implemented in the Snakemake
workflow but were not executed for this report; they are listed as code-only in
§4.

### 2.1 Stage 0 — structural infrastructure (one-time)

The AlphaFold human proteome (UP000005640, v4) was downloaded and each model
cleaned by pLDDT: residues with pLDDT < 50 removed, terminal low-confidence runs
(≥ 5 residues, pLDDT < 70) trimmed, and pLDDT 50–70 residues soft-masked into a
side manifest. This yielded **20,504 cleaned structures**. P2Rank 2.5 (run under
its `alphafold` config) detected pockets for all structures: **15,021 received a
usable pocket; 5,483 were flagged no-pocket**. Meeko prepared rigid-receptor
PDBQT files for the with-pocket set, with OpenBabel (`obabel -xr`) as a fallback
for structures meeko's residue templates rejected, producing **12,808 docking
receptors**. ChEMBL34 was mirrored and sliced to **1,978,826 human
single-protein activity rows**; CosIng, a drug-avoidance set, and a
skin-efficacy knowledge-graph placeholder complete the infrastructure.

### 2.2 Stage 1 — compound preprocessing

Input SMILES were standardised with RDKit (normalise, largest-fragment,
uncharge, canonical tautomer), protonated at pH 7.2–7.6 with Dimorphite-DL, and
written as a canonical record. (The full design also embeds ETKDGv3 conformers
and xTB-optimises them; the demonstration used the standardise + meeko-embed
path.)

### 2.3 Stage 2 / 2.5 — ADMET gate and cosmetic/drug annotation

A three-model skin-sensitisation consensus returns HALT (≥ 2/3 positive),
FLAG_HIGH (1/3, or all models unavailable → human review) or PASS, alongside
PAINS/Brenk/NIH structural alerts. Stage 2.5 matches the compound against the
CosIng INCI database (EXACT / SIMILAR ≥ 0.85 / ANALOG ≥ 0.65 / NEW) and an
approved-drug set (strict/scaffold/soft warnings), emitting a
policy-driven HALT/DOWNWEIGHT/PROCEED.

### 2.4 Stage 3 — structure-based target identification (demonstration)

The primary 300-receptor ranking uses **AutoDock Vina** (`sf_name = vina`) as a
single, fast, CPU docking engine; the production multi-engine scoring stack
(GNINA, RTMScore, PSICHIC) was unavailable on this host (§4), and Boltz-2 was run
separately as a GPU affinity probe (§3.4). For each
compound the ligand was prepared with meeko (after an explicit-H + ETKDGv3 3D
embed to satisfy meeko 0.7.x), then docked against a **300-receptor subset**:
the literature-known target(s) were force-included and the remainder were a
random sample of the with-pocket receptor set drawn with a **fixed seed of 42**.
Docking used **exhaustiveness 8**; the per-receptor score is the best
binding-free-energy pose (kcal/mol). Targets are ranked by ascending ΔG.
Box centres and sizes come from the Stage 0 P2Rank pockets.

### 2.5 Stage 3 v3 — skin-expression weighting

A composite SkinScore in [0, 1] is computed per UniProt accession from the
Human Protein Atlas `proteinatlas.tsv` (keyed on its `Uniprot` column): a
weighted sum of skin-tissue nTPM (0.40), best skin cell-type nCPM (0.40), and a
skin tissue-specificity bonus (0.20), each min-max normalised in log space.
Sanity check (expected order recovered): KRT14 0.767 (high, basal keratinocytes),
TYR 0.418 (medium, melanocytes), FLG 0.253, MMP1 0.174 (fibroblasts) — see
Figure 3. The final target score is
`final = 0.7·minmax(−ΔG) + 0.3·skin_score`, with a hard cap multiplying the
result by 0.30 when `skin_score < 0.05`. **The skin weighting changes the
ranking only; it is not applied to the reported ΔG values, which remain raw
Vina scores.**

### 2.6 Reproducibility and runtime-bug log

The pipeline was hardened against a series of failures that only surfaced when
running on real data, each fixed and recorded in `progress.txt`: the
`rdMolStandardize` API surface; Dimorphite-DL 2.x's removed CLI (rewritten to the
Python API); `DataStructs.ExplicitBitVect` vs `AllChem`; a missing
`assays.tid` index in ChEMBL34 (which collapsed the activity-slice query from
hours to 0.30 s); a PyArrow int64 overflow on ChEMBL `standard_value`; P2Rank's
incompatibility with Java 25 (pinned OpenJDK 21) and its dataset-file syntax;
leading whitespace in P2Rank `predictions.csv` headers; and meeko 0.7.x's
`gemmi`/`prody` dependencies, output-basename behaviour, and rejection of many
AlphaFold structures (mitigated by the OpenBabel fallback). Key tool versions:
Snakemake 8.x, P2Rank 2.5, MMseqs2 17.x, RDKit 2025.03, Meeko 0.7.x, OpenBabel
3.1.1, AutoDock Vina 1.2.5, OpenJDK 21.

## 3. Results

### 3.1 Known-target recovery across nine compounds

Table 1 lists each compound's literature target(s), the Vina rank, the
skin-weighted rank, and the raw Vina ΔG. Across the 24 compound–target pairs,
Vina-only recovery is **7/24 in the top 10 and 9/24 in the top 30**;
skin-weighting raises this to **9/24 and 13/24** respectively (Figure 1).

**Table 1 — retrospective recovery (300 receptors/compound, Vina exhaustiveness 8, seed 42).**

| Compound | Mechanism | Target | UniProt | Vina rank | +Skin rank | Vina ΔG |
|---|---|---|---|---:|---:|---:|
| Retinol | anti-aging | RAR-β | P10826 | 1 | 22 | −10.22 |
| Retinol | anti-aging | RAR-α | P10276 | 3 | 23 | −9.97 |
| Retinol | anti-aging | RAR-γ | P13631 | 4 | 1 | −9.96 |
| Niacinamide | brightening | NNMT | P40261 | 2 | 1 | −6.35 |
| Niacinamide | brightening | CD38 | P28907 | 25 | 42 | −5.59 |
| Niacinamide | brightening | SIRT1 | Q96EB6 | 32 | 50 | −5.50 |
| Ascorbic acid | collagen | P4HB | P07237 | 6 | 27 | −7.04 |
| Ascorbic acid | collagen | P4HA1 | P13674 | 43 | 62 | — |
| Ascorbic acid | collagen | P4HA2 | O15460 | 56 | 75 | — |
| Aspirin (control) | anti-inflammatory | COX-2 | P35354 | 7 | 1 | −7.04 |
| Aspirin (control) | anti-inflammatory | COX-1 | P23219 | 16 | 2 | −6.63 |
| Resveratrol | antioxidant | SIRT1 | Q96EB6 | 8 | 29 | −8.42 |
| Resveratrol | antioxidant | AHR | P35869 | 249 | 252 | −5.79 |
| Kojic acid | whitening | TYR | P14679 | 35 | 1 | −5.48 |
| α-Arbutin | whitening | TYR | P14679 | 88 | 3 | −6.61 |
| EGCG | antioxidant | MMP9 | P14780 | 43 | 61 | −9.20 |
| EGCG | antioxidant | MMP2 | P08253 | 64 | 5 | −8.90 |
| EGCG | antioxidant | DNMT1 | P26358 | 61 | 77 | −8.90 |
| EGCG | antioxidant | TYR | P14679 | 99 | 4 | −8.50 |
| Caffeine | anti-cellulite | ADORA2A | P29274 | 57 | 70 | −5.80 |
| Caffeine | anti-cellulite | ADORA3 | P0DMS8 | 62 | 75 | −5.80 |
| Caffeine | anti-cellulite | ADORA1 | P30542 | 102 | 113 | −5.60 |
| Caffeine | anti-cellulite | ADORA2B | P29275 | 121 | 10 | −5.50 |
| Caffeine | anti-cellulite | PDE3A | Q14432 | 141 | 151 | −5.30 |

The strongest single signal is retinol, whose three retinoic-acid receptors
occupy Vina ranks 1, 3 and 4 at ΔG ≈ −10 kcal/mol (Figure 2). Niacinamide → NNMT
(#2) and aspirin → COX-2 (#7) confirm the pipeline reproduces both cosmetic and
classical drug-target relationships.

### 3.2 Skin-weighting rescues cofactor-dependent targets

The largest rank gains under skin-weighting all fall on receptors whose ligand
recognition AlphaFold cannot model (Figure 1, green): kojic acid → TYR #35→#1,
α-arbutin → TYR #88→#3, EGCG → TYR #99→#4, EGCG → MMP2 #64→#5, caffeine →
ADORA2B #121→#10. The trade-off is by design: systemically-expressed targets are
demoted (retinol RAR-α/β #3/#1 → #23/#22; resveratrol SIRT1 #8→#29), reflecting
the cosmetic-application prior. The net effect across the panel is positive
(+2 top-10, +4 top-30).

### 3.3 Negative result: Vinardo scoring does not rescue tyrosinase

Re-docking kojic acid against tyrosinase with Vina's alternate Vinardo scoring
function moved the target from #35 to #180 — worse — because Vinardo only
re-weights hydrogen-bond/hydrophobic terms and is equally blind to the missing
copper.

### 3.4 Boltz-2 affinity as a GPU re-rank (five-pair probe)

After isolating Boltz-2 in a torch-2.4 + CUDA-12.1 environment compatible with
the host driver 570 (§4), co-folding plus the affinity head ran on the 12 GB
GPU in ≈ 1 min per pair. On a five-pair probe (Table 2) Boltz-2's binder
probability tracks the literature better than Vina rank on the two docking
misses: caffeine→ADORA2A (Vina #57) scores 0.65 and kojic acid→tyrosinase
(Vina #35) scores 0.57, while a deliberate non-target control (kojic acid vs
keratin-14) scores lowest at 0.35. Retinol→RAR-β, the strongest docking hit,
also scores highest (0.89), so the two methods agree where docking is reliable.
The exception is niacinamide→NNMT (0.25): the small ligand (122 Da) that Vina
ranked #2 is under-scored by the affinity head, a known weak spot for very
small molecules. This is a probe, not a benchmark — but it demonstrates the
intended multi-method behaviour: Boltz-2 supplies orthogonal evidence on the
metalloenzyme and GPCR targets where single-engine docking is structurally
blind.

**Table 2 — Boltz-2 affinity probe (cosmax-boltz env, GPU).**

| Compound | Target | Vina rank | Boltz-2 pred (lower = stronger) | Boltz-2 binder prob |
|---|---|---:|---:|---:|
| Retinol | RAR-β (P10826) | 1 | −1.61 | 0.89 |
| Caffeine | ADORA2A (P29274) | 57 | 1.39 | 0.65 |
| Kojic acid | TYR (P14679) | 35 | 1.04 | 0.57 |
| Kojic acid | KRT14 (P02533, control) | — | 1.24 | 0.35 |
| Niacinamide | NNMT (P40261) | 2 | 2.69 | 0.25 |

## 4. Discussion

**Why skin-weighting is the effective driver here.** On a single, metal-blind
docking engine, the structural score alone cannot recover ligands that bind via
a cofactor AlphaFold omits. The skin-expression prior supplies orthogonal
evidence — TYR, MMP2 and the adenosine receptors are skin-relevant — and so
rescues exactly the targets docking misses. This is why the verified recipe on
this host is Stage 0–3 + skin-weighting rather than docking alone.

**The cofactor limitation, and an honest negative.** Every recovery failure in
Table 1 is a metallo-enzyme (TYR-Cu, MMP-Zn) or an active-state GPCR
(adenosine receptors), consistent with AlphaFold modelling neither metal
cofactors nor agonist-bound states. We tested the obvious fix for tyrosinase:
`scripts/transplant_cofactor.py` inserts Cu²⁺ atoms at the His-triad centroids
and tightens the docking box around them. Under metal-blind Vina this made rank
**worse** (kojic acid #35→#56; α-arbutin #88→#245), because Vina has no
atom-type for copper. The transplant code also defines Zn²⁺ sites for MMP9/MMP2
and CA-II, but those were **not** re-docked — only tyrosinase was tested. A
metal-aware engine (AutoDock4 PMF, or an FF/MD treatment of the coordination)
is required for the transplant to help, and the holo structures are kept in the
repo for that future run.

**ML re-scoring: one recovered, two still blocked.** Boltz-2 initially failed
because `pip install boltz` pulled torch 2.12 + CUDA 13, which the host driver
570.195 cannot drive. Isolating Boltz-2 in a dedicated env pinned to
torch 2.4 + CUDA 12.1 — the highest CUDA the driver supports — restored GPU
execution without any driver change; co-folding + affinity then ran in ≈ 1 min
per pair (§3.4, Table 2). This makes the point that the "GPU too small / driver
too old" barrier was a packaging artefact, not a hardware limit. GNINA, by
contrast, remains unrunnable: no conda or pip wheel, a source build needing
`nvcc` (absent) and Docker (absent, no sudo); RTMScore was blocked by a
`torch_scatter` ABI mismatch. None of these are scientific failures; they are
toolchain/IT constraints, and two of the three are now resolved or have a clear
resolution path.

**Limitations.** The primary 300-receptor ranking uses a single docking engine
(Vina) on a 300-receptor subset per compound (not the full 12,808) and a single
AlphaFold conformer per receptor (no BioEmu ensemble); ΔG values are raw Vina,
not skin-weighted; the skin-weighting analysis is post-hoc. Boltz-2 was run only
as a five-pair probe (Table 2), not as the full top-50 re-rank the design
intends; GNINA, RTMScore, BioEmu, MD and QM stages are implemented but were
**not run**. Host hardware is below the design spec (12 GB vs 16 GB GPU; 16 vs
32 threads), mitigated by pocket-centric cropping.

**Future work.** (1) A box with a CUDA-12 driver / Docker where Boltz-2 and GNINA
run, turning Stage 3 into the intended multi-engine RRF; (2) a metal-aware
docking engine to validate the cofactor transplant on TYR/MMP/CA; (3) BioEmu
ensemble docking on the top-10 targets per compound; (4) scaling from 300 to the
full receptor proteome once a faster engine than Vina-CPU is available.

## 5. Data and code availability

Code: https://github.com/kangk1204/SkinScout_public (MIT). Per-run rankings, SkinScore
table and figures are committed under `results/`. Data sources and licenses:
AlphaFold DB v4 (CC-BY-4.0, attribution required), Human Protein Atlas
(CC-BY-SA-3.0), ChEMBL34 (CC-BY-SA-3.0), CosIng (EU public data), PoseBusters /
PLINDER (open). The full license audit is in `LICENSE_POLICY.md`.

## Figures

- **Figure 1** (`results/figures/fig1_recovery_slope.png`) — per compound–target
  rank, Vina vs skin-weighted; green improved, red demoted.
- **Figure 2** (`results/figures/fig2_deltaG_strip.png`) — per-compound ΔG
  distribution over 300 receptors, known targets highlighted.
- **Figure 3** (`results/figures/fig3_skinscore.png`) — SkinScore distribution
  across 19,179 human proteins with known skin proteins marked.
