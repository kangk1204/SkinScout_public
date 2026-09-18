# Architecture (v2)

> Concise restatement of `INSTRUCTIONS.md` §2. Read alongside, not instead of, the spec.

## 1. Dataflow

```
            ┌────────────── ONE-TIME ──────────────┐
            │   STAGE 0: Infrastructure (≈200 GB)  │
            │   AF human proteome + pLDDT trim +   │
            │   P2Rank pockets + Meeko PDBQT +     │
            │   ChEMBL/BindingDB + optional DrugBank│
            │   + MMseqs2                           │
            └──────────────┬───────────────────────┘
                           ▼ (cached, reused per compound)

  SMILES/SDF  ──►  Stage 1 preprocess  ──►  Stage 2 ADMET gate
                                              │
                                              ▼ (PASS / FLAG_HIGH / HALT)
                       ┌─────────── Stage 3 dual-mode ───────────┐
                       │                                          │
                MODE-COMPREHENSIVE                          MODE-FAST
                AutoDock-GPU 전수 20.4k                     PSICHIC + Daina-Zoete
                → top 1-2% → GNINA + RTM + Boltz-2          → top 25% → AutoDock 5k
                → 4-way RRF → DTI sanity ◄──────────────────┘ → top 1% → rerank
                                              │
                                              ▼
                              Stage 4 struct prep (top 50)
                                              │
                                              ▼
                              Stage 5 Boltz-2 co-fold + affinity
                                              │
                                              ▼
                              Stage 6 BioEmu ensemble + dock (top 10)
                                              │
                                              ▼
                              Stage 7 GROMACS MD + MM-GBSA (top 3-5)
                                              │
                                              ▼
                              Stage 8 QM / xTB / DFT (final 1-3)
                                              │
                                              ▼
                              Stage 9 Mol* HTML report
                                              │
                                              ▼ optional
                              Stage 10 ZINC22 expansion
```

## 2. Mode selection (§6.1)

| Scenario | Mode |
|---|---|
| Novel chemotype / orphan target probability / new scaffold | **COMPREHENSIVE** (default) |
| Known chemotype + known target class, fast triage | **FAST** |
| Methodology paper / disagreement-as-data | **BOTH** |
| Analog repeat-screen | **FAST** (warm infrastructure cache) |

`workflow/config.yaml: mode` controls this. `run_dti_sanity: true` always runs the PSICHIC sanity-check leg in COMPREHENSIVE.

## 3. Scoring fusion

4-way Reciprocal Rank Fusion at Stage 3 ranks top candidates:

```
RRF(t) = Σ_score 1/(60 + rank_score(t))
score ∈ { AutoDock-GPU Vinardo, GNINA-CNN, RTMScore, Boltz-2 affinity }
```

Implemented in `scripts/stage3_rrf.py` and unit-tested with three rank lists per candidate.

## 4. Data-leakage discipline (§14)

| Axis | DB | Threshold |
|------|----|-----------|
| Sequence | MMseqs2 vs. training-cutoff FASTA | seq_id ≥ 30 % → leak |
| Ligand | Morgan FP Tanimoto vs. training ligand DB | t ≥ 0.5 → leak |
| Pocket | PLINDER SuCOS vs. training holo pockets | s ≥ 0.5 → leak |

All eval scripts write a paired `*_leakage_report.tsv` next to each main result.

## 5. Hardware constraints (current host)

- 12 GB GPU forces pocket-centric cropping (±20 Å) in Stage 5/6 for proteins > 500 AA.
- 16 cores → P2Rank batch in §6.1 of STAGE0_RUNBOOK takes ≈ 24-40 h (CPU-bound, no GPU).
- AutoDock-GPU stays under 6 GB → can co-schedule with PSICHIC (DTI) via separate CUDA streams.

## 6. Failure modes & recovery

- **Pocket not detected** (P2Rank score < threshold) → DiffDock-L blind branch in Stage 3.
- **Boltz-2 ipTM < 0.6** or pocket pLDDT < 75 → drop candidate, escalate next on list.
- **PoseBusters PB-invalid pose** → re-dock with denser sampling or skip.
- **MD blowup** → halve timestep, switch to OpenMM engine, retry once.
