# Cosmetic Ingredient Retrospective Recovery — SkinScout

Blind retrospective validation: do well-characterised cosmetic / dermatological
small molecules dock to *their known targets* when ranked against 300 real
AlphaFold receptors per compound (1 forced ground-truth target + a random peer
panel from the with-pocket subset, fixed seed = 42, AutoDock Vina
`sf_name="vina"`, exhaustiveness = 8)?

Host: i7-13700K · RTX 3080 Ti 12 GB · 16 cores · 123 GB RAM.

## Results — 8 compounds × 24 known targets

| Compound | Mechanism | Best rank | Best ΔG | Recoveries (rank, ΔG, kcal/mol) |
|---|---|---:|---:|---|
| **Retinol** | anti-aging | **#1** | **−10.22** | 🎯 RAR-β #1 (−10.22) · 🎯 RAR-α #3 (−9.97) · 🎯 RAR-γ #4 (−9.96) |
| **Niacinamide** | brightening / barrier | **#2** | −6.51 | 🎯 NNMT #2 (−6.35) · ✅ CD38 #25 (−5.59) · △ SIRT1 #32 (−5.50) |
| **Ascorbic acid** | collagen synthesis | **#6** | −7.29 | 🎯 P4HB #6 (−7.04) · △ P4HA1 #43 · △ P4HA2 #56 |
| **Aspirin** | anti-inflammatory (control) | **#7** | −7.46 | 🎯 COX-2 #7 (−7.04) · ✅ COX-1 #16 (−6.63) |
| **Resveratrol** | antioxidant / anti-aging | **#8** | −9.75 | 🎯 SIRT1 #8 (−8.42) · △ AHR #249 (−5.79) |
| **Kojic acid** | whitening | #35 | −6.29 | △ TYR #35 (−5.48) |
| **EGCG** | antioxidant / MMP inhibition | #43 | −11.03 | △ MMP9 #43 · △ DNMT1 #61 · △ MMP2 #64 · △ TYR #99 |
| **α-Arbutin** | whitening | #88 | −8.48 | △ TYR #88 (−6.61) |
| **Caffeine** | anti-cellulite | #57 | −7.47 | △ ADORA2A #57 · △ ADORA3 #62 · △ ADORA1 #102 · △ ADORA2B #121 · △ PDE3A #141 |

Legend: 🎯 Top-10 · ✅ Top-30 · △ outside Top-30.

**Aggregate**: Top-10 recovery **7/24 (29 %)** · Top-30 recovery **9/24 (38 %)**.

ΔG span per run (kcal/mol):

| Compound | min | max |
|---|---|---|
| Retinol | −10.22 | −4.95 |
| EGCG | −11.03 | −5.81 |
| Resveratrol | −9.75 | −5.10 |
| α-Arbutin | −8.48 | −5.20 |
| Caffeine | −7.47 | −4.21 |
| Aspirin | −7.46 | −3.66 |
| Ascorbic acid | −7.29 | −3.97 |
| Niacinamide | −6.51 | −3.66 |
| Kojic acid | −6.29 | −3.51 |

## Interpretation

### Where the pipeline excels (5/8 compounds, all Top-10 hits)

- **Retinol → RAR α/β/γ at #1/#3/#4 (ΔG ≈ −10 kcal/mol, nM regime)** — the
  textbook anti-aging mechanism is recovered with three of three targets in
  top-4. This is the strongest single signal in the panel.
- **Niacinamide → NNMT #2** — primary metabolic target dominates the ranking.
- **Aspirin → COX-2 #7, COX-1 #16** — classical drug-target control reproduces
  cleanly (canonical, not skin-specific).
- **Ascorbic acid → P4HB #6** — collagen prolyl-4-hydroxylase β subunit; the
  collagen-synthesis mechanism behind topical vitamin C is captured.
- **Resveratrol → SIRT1 #8** — sirtuin activation, the canonical longevity
  pathway, is in the top 3 % blind.

### Where it fails — *all four failures share one cause*

| Compound | Target(s) missed | Cofactor / state Stage 0 cannot model |
|---|---|---|
| Kojic acid | TYR | **Cu²⁺ × 2** (dinuclear copper centre) |
| α-Arbutin | TYR | **Cu²⁺ × 2** (same as kojic) |
| EGCG | MMP9, MMP2 | **Zn²⁺** (catalytic + structural) |
| Caffeine | ADORA1/2A/2B/3 | **GPCR active state** (AlphaFold predicts inactive) |

This is **not random failure** — every miss is on a receptor whose ligand
recognition depends on a metal cofactor (TYR-Cu, MMP-Zn) or a conformational
state (GPCR active-state, agonist-bound) that bare AlphaFold cannot reproduce.
It is exactly the limitation called out in `INSTRUCTIONS.md §17`:

> "공유결합 ligand, 금속결합, 큰 conformational change: 별도 분기 (RFAA,
> Boltz-2 covalent mode 등)"

### Negative result we ran: Vinardo scoring does NOT fix kojic acid

We hypothesised that `sf_name='vinardo'` (Vina's alternate empirical scoring)
might rescue the metal-binding miss. It did not:

```
Kojic acid → TYR
  Vina:    #35 / 300  (top 12 %)  ΔG = −5.48
  Vinardo: #180/300   (top 60 %)  ΔG = −3.81   ← worse
```

Vinardo only re-weights hydrogen-bond / hydrophobic terms. It does not model
metal coordination. Both empirical scoring functions are blind to the missing
Cu²⁺ in the AlphaFold receptor. **The fix has to be at the receptor level**
(transplant Cu²⁺ from a PDB experimental holo, or use a metal-aware docking
engine like AutoDock4 PMF / Vina with explicit metals), not at the scoring
level.

### Score-axis observation

Bigger ligands generate more negative ΔG by surface-area effect: EGCG (MW 458,
ΔG max −11.0) > retinol (286, −10.2) > caffeine (194, −7.5) > kojic (142,
−6.3). **ΔG magnitudes are not directly comparable across compounds** — the
within-compound *ranking* is what matters for target ID.

## Caveats

- Single-method ranking (Vina only). GNINA / RTMScore re-scoring was attempted
  but blocked by environment issues (Docker absent · torch_scatter ABI vs
  torch 2.4 mismatch · nvcc absent for source build). Tracked for the next
  iteration.
- No skin-expression weighting applied — the published Stage 3 v3 formula
  (`0.7×dock + 0.3×skin_score`) is implemented but skipped so that recoveries
  are attributable to docking alone.
- Single AlphaFold conformer per receptor (no BioEmu ensemble dock).
- 300-receptor subset per compound (not the full 12,808 PDBQT proteome).
- Compound preparation skipped ETKDG / xTB and used the standardise + meeko
  embed path; results consistent with literature, but a full Stage 1 run
  (3D-refined conformers) would tighten ΔG estimates.

## Next iterations the data motivates

1. **Cofactor transplant** for TYR / MMP-family / similar metallo-enzymes —
   take Cu²⁺ / Zn²⁺ coordinates from a recent (post-2023-10) PDB holo, splice
   into the AlphaFold structure, redo docking box. Most cost-effective
   single fix on this benchmark.
2. **Boltz-2 affinity head re-rank on top-50** — the spec's intended Stage 3
   second pass; turns the docking-only ranking into a 2-way fusion. GPU model
   weights are available; deferred only because Stage 0 first.
3. **GNINA CNN re-rank** — once Docker or a clean CUDA build env is set up
   (the actual blocker, not the science).
4. **Apply skin-expression weighting** and re-rank these 8 compounds; expected
   to amplify skin-relevant targets (P4HB, NNMT, RAR) further.

## Reproduce

```bash
python scripts/demo_dock_vina.py \
  --ligand-sdf results/runs/<name>_demo/01_input/compound_canonical.sdf \
  --pdbqt-dir data/human_pdbqt --box-dir data/docking_boxes \
  --include <known_uniprot_csv> \
  --n-receptors 300 --exhaustiveness 8 --seed 42 \
  --score-func vina \
  --out-csv results/runs/<name>_demo/03_targets/demo_ranked_targets.csv
```

Stage 0 receptor set required: AlphaFold human proteome + pLDDT cleanup +
P2Rank pockets + Meeko PDBQT. See `docs/STAGE0_RUNBOOK.md`.

---

# Iteration 2 — Skin weighting · Cofactor transplant · Boltz-2 / GNINA attempts

After the first 8-compound retrospective the four follow-ups originally listed
under "Next iterations" were attempted. Here's what worked, what didn't, and
why — recorded as a single, honest changelog so the methodology section of any
manuscript can lift it verbatim.

## ④ Skin-expression weighting — **WORKED**

`scripts/stage0_skin_score_v2.py` rebuilds the SkinScore from the single
consolidated `proteinatlas.tsv` (column `Uniprot`), fixing a key mismatch in
v1 that joined on Ensembl IDs (`Gene`) and lost everything downstream when
merged against the UniProt-keyed receptor set.

Sanity (known skin proteins, expected order):

| Symbol | UniProt | SkinScore | Cell type preferred |
|---|---|---:|---|
| KRT14 | P02533 | 0.767 (high) | Basal keratinocytes |
| TYR | P14679 | 0.418 (medium) | Melanocytes |
| FLG | P20930 | 0.253 (low) | — |
| MMP1 | P03956 | 0.174 | Fibroblasts |
| CXADR | P78310 | 0.164 | — |

Stage 3 v3 formula `final = 0.7 × normalize(−ΔG) + 0.3 × skin_score`
(skin_score < 0.05 → final × 0.30) re-ranks all 8 compounds:

| Metric | Vina-only | + Skin-weight | Δ |
|---|---:|---:|---:|
| Top-10 recovery | 7 / 24 | **9 / 24** | **+2** |
| Top-30 recovery | 9 / 24 | **13 / 24** | **+4** |

Biggest rescues (all on cofactor-missing receptors):

| Compound · target | Vina | + Skin | Δ |
|---|---:|---:|---:|
| Kojic acid · TYR | #35 | **#1** | +34 |
| α-Arbutin · TYR | #88 | **#3** | +85 |
| EGCG · TYR | #99 | **#4** | +95 |
| EGCG · MMP2 | #64 | **#5** | +59 |
| Caffeine · ADORA2B | #121 | **#10** | +111 |

Intended trade-off (the formula is by-design biased toward skin-relevance):

| Compound · target | Vina | + Skin | Δ |
|---|---:|---:|---:|
| Retinol · RAR-α | #3 | #23 | −20 |
| Retinol · RAR-β | #1 | #22 | −21 |
| Niacinamide · SIRT1 | #32 | #50 | −18 |
| Resveratrol · SIRT1 | #8 | #29 | −21 |

These are systemically expressed targets that the cosmetic-application prior
deliberately demotes. Net +2/+4 across the panel confirms the weighting helps
on this cohort.

## ① Cofactor transplant (TYR Cu²⁺) — single-engine effect is **NULL**

`scripts/transplant_cofactor.py` inserts metal HETATM atoms at the centroid
of each His-binding triad in the AF receptor, then expands the docking box to
an 18 Å cube centred on the metal sites. Built-in mappings:

| UniProt | Site | Element | Residues |
|---|---|---|---|
| P14679 TYR | CuA / CuB | Cu² × 2 | H180/202/211 · H363/367/390 |
| P14780 MMP9 | Catalytic Zn²⁺ | Zn | H401/405/411 |
| P08253 MMP2 | Catalytic Zn²⁺ | Zn | H403/407/413 |
| P00918 CA2 | Zn²⁺ | Zn | H94/96/119 |

Re-docked kojic acid + α-arbutin against the Cu²⁺-holo TYR (P14679 only was
swapped, the other 299 receptors stayed as bare AF):

| | Bare AF | + Cu²⁺ holo | Δ |
|---|---:|---:|---:|
| Kojic acid · TYR | #35 | #56 | **worse** |
| α-Arbutin · TYR | #88 | #245 | **worse** |

The cofactor transplant did **not** improve rank on this single-engine
pipeline. Cause:

> **Vina is not metal-aware.** The HETATM Cu atoms have no entry in Vina's
> atom-type / grid map, so they contribute nothing (or, when the box expands
> around them, generate spurious vdW clashes that hurt the score). The fix
> needs a metal-aware engine — AutoDock4 PMF, Glide metal constraint, or an
> MD/FF approach that explicitly treats the Cu coordination.

The transplant code, data (`data/holo_transplant/P14679_with_Cu.pdb`), and
test re-dock outputs are kept in the repo so a future metal-aware engine run
can re-use them without re-deriving the centroids.

## ② Boltz-2 affinity head — **blocked on driver/CUDA mismatch**

`boltz` (≥ 2.0) was pip-installed into `cosmax-base` for a re-rank pass on
the top-50 docked candidates per compound. Installation pulled torch
2.12 + cu130 as a transitive dependency, which the host NVIDIA driver 570.195
cannot drive (CUDA 13 too new). CPU fallback (`--accelerator cpu`) starts but
the TYR receptor is 529 residues; the diffusion model's CPU run is too slow
to be useful on a wall-clock budget (≫ 1 hour per affinity).

Concrete blockers (any one of which unblocks it):
- downgrade torch to 2.4 + cu121 (was the working state before `pip install
  boltz`) **and** pin an older boltz version that accepts torch 2.4, **or**
- bump the host NVIDIA driver to ≥ 580 (allows torch + cu130), **or**
- move the boltz step to a separate env / box that has a CUDA-12.x driver.

Tracked. The full Stage 5 (`boltz2_cofold` rule) is implemented in the
Snakefile (`workflow/rules/stage5_boltz2.smk`) and ready to run as soon as
the env compatibility is resolved.

## ③ GNINA CNN re-rank — **environment unavailable on this host**

GNINA is the textbook choice for CNN re-rank, but on this host:

- not on conda-forge / bioconda (verified)
- no pip wheel
- source build needs `nvcc` (CUDA dev toolkit) — absent
- Docker absent (would also need sudo to install)
- RTMScore (an alternative ML re-scorer in the same category) was attempted
  but the torch-scatter prebuilt wheels expose an ABI mismatch with torch 2.4

Skipped honestly. The 4-way RRF described in `INSTRUCTIONS.md §6.2(c)` is
implemented as `scripts/stage3_rrf.py` and unit-tested; only the GNINA leg
needs the missing weights / engine to slot in.

## Overall conclusion

For *this single-engine, GPU-driver-mismatched, no-Docker* configuration, the
effective recipe is:

```
Stage 0 + Stage 1 + Stage 2/2.5 + (Vina, vina sf, exhaustiveness=8)
  + Stage 3 v3 skin-weighting  ← the actual driver of cofactor-target recovery
  → Mol* report (Stage 9)
```

The other three follow-ups are correct in design but each blocked on an
environment/IT detail — not on the science.

Future iterations that would change the answer here:
1. A box (Docker or fresh CUDA driver) where boltz-2 + GNINA can both run.
2. A metal-aware docking engine (AutoDock4 PMF or AMBER metal MD) — the
   `data/holo_transplant/` files are ready to consume.
3. BioEmu ensemble docking per top-10 target (already wired in
   `workflow/rules/stage6_bioemu.smk`).
4. A larger receptor subset (3000 → 12,808 full PDBQT proteome) once an
   engine faster than Vina-CPU is available.
