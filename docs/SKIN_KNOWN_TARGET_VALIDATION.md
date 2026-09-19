# Skin Known-Target Validation

Date: 2026-08-21

This document records the current frozen known-skin target validation status.
It replaces earlier retrospective demo wording. Do not use the older 8-compound
subset as evidence for implementation PASS, SOTA performance, clinical utility,
or calibrated target-retrieval probability.

## Scope

The validation asks whether SkinScout ranks known skin-beneficial or
skin-adverse compound targets near the top without using a known-target prior.
It is separate from assisted production workflow, where an operator may add
licensed data, manual curation, or follow-up review before deciding what to
test experimentally.

The tracked fixed panel remains
`data/validation/skin_known_target_panel.csv`, containing 15 skin-relevant
small molecules and their UniProt target annotations.

## Current Frozen Results

| Evaluation | Baseline | Selected | Gate |
|---|---:|---:|---|
| 2025 temporal frozen Top10 | 0.60067 | 0.71141 | PASS |
| 2025 temporal frozen Top30 | 0.71141 | 0.77852 | PASS |
| 2025 temporal frozen MRR | 0.41727 | 0.54394 | PASS |
| 2025 temporal frozen Brier | 0.07031 | 0.06673 | PASS |
| 2025 temporal frozen log-loss | 0.26835 | 0.24916 | PASS |
| Full 15-case known-skin retrospective Top10 | 0.1875 | 0.21875 | FAIL overall |
| Full 15-case known-skin retrospective Top30 | 0.625 | 0.500 | FAIL: -0.125 |
| Full 15-case known-skin retrospective MRR | 0.13623 | 0.14740 | FAIL overall |

The 2025 temporal frozen test passes after selection. The full 15-case
known-skin no-known-target-prior retrospective panel fails because selected
Top30 regresses by 0.125 from the baseline. Small Top10 and MRR improvements do
not override that failure.

The candidate `union_p6_consensus` is therefore not promoted. The operational
gate passes by retaining the exact frozen `chembl_p5_max` baseline for Stage 3
exploratory runs. The separate scientific claim gate fails and continues to
block Stage 11 publication and all comparative-performance or SOTA claims.

## Dual-Cold Integrated Panel

The true ligand/scaffold + target-sequence dual-cold integrated panel contains:

| Item | Count |
|---|---:|
| Queries | 115 |
| Truth pairs | 163 |
| Targets | 98 |

Results:

| Metric | Baseline | Selected |
|---|---:|---:|
| Top10 | 0 | 0 |
| Top30 | 0 | 0 |
| MRR | 0.00008293 | 0.00008416 |

These results do not support a cold-start recovery claim.

## ESM2 Sequence-Transfer Research Iteration

This iteration adds provenance-bound ESM2 (`esm2_t6_8M_UR50D`, layer 6)
embeddings for the complete 20,204-target screenable universe. The scorer uses
only pre-2024 positive training support and protein-sequence neighbors. It does
not receive query truth targets and does not alter scores for train-supported
targets.

The registered 2024 target-cluster-cold development panel contains 672 queries,
677 truth pairs, and 27 targets. The selected `esm6_knn_k16_p2_w075` recipe
passes every registered development comparison against the `analog_only`
sequence baseline:

| Development metric | Analog only | ESM2 transfer |
|---|---:|---:|
| Micro Top10 | 0 | 0.001477 |
| Micro Top30 | 0 | 0.001477 |
| Micro MRR | 0.00008413 | 0.00050222 |
| Target-macro Top10 | 0 | 0.03704 |
| Target-macro Top30 | 0 | 0.03704 |
| Target-macro MRR | 0.00008413 | 0.00586188 |
| Regular-dev Brier | 0.079268 | 0.078688 |
| Regular-dev log-loss | 0.285251 | 0.285222 |
| Unsupported-target Brier | 0.090423 | 0.088640 |
| Unsupported-target log-loss | 0.326244 | 0.316154 |

The absolute ranking effect remains small: one of 677 development truth pairs
enters the top 30. One target contributes 58.1% of all truth pairs and the
effective target count is 2.71. Aggregate ECE worsens from 0.01946 to 0.02587,
and unsupported-target ECE worsens from 0.00027 to 0.07734. Passing this
development gate therefore demonstrates a narrow reproducible signal, not
practical or broadly distributed cold-target recovery.

### Previously inspected post-hoc panels

The table below belongs to the earlier evaluation snapshot inspected before the
sequence model was designed. It is retained as historical rejection evidence,
not as the current benchmark result. These post-hoc diagnostics cannot provide
prospective confirmation or change the development gate.

| Post-hoc metric | Analog only | ESM2 transfer | Result |
|---|---:|---:|---|
| 2025 temporal Top10 | 0.72241 | 0.71906 | regresses |
| 2025 temporal Top30 | 0.78595 | 0.78595 | tied |
| 2025 temporal MRR | 0.54864 | 0.54759 | regresses |
| 2025 aggregate Brier | 0.06670 | 0.06659 | improves slightly |
| 2025 aggregate log-loss | 0.24814 | 0.24873 | regresses |
| 2025 unsupported-target Brier | 0.21676 | 0.22686 | regresses |
| 2025 unsupported-target log-loss | 0.67289 | 0.75698 | regresses |
| Known-skin Top10 | 0.21875 | 0.21875 | tied |
| Known-skin Top30 | 0.5000 | 0.4375 | regresses |
| Known-skin MRR | 0.14716 | 0.14462 | regresses |
| Dual-cold Top10 | 0 | 0 | no recovery |
| Dual-cold Top30 | 0 | 0 | no recovery |
| Dual-cold MRR | 0.00008416 | 0.00025753 | improves, still impractical |

The transfer fills part of the 16,636-target unsupported search space from
3,568 train-supported targets. This raises many unsupported candidates at once:
cold-target reciprocal rank can improve while supported truth targets are
displaced in the unified 20,204-target ranking. The observed post-hoc
regressions reject production promotion of this recipe.

No untouched evaluation panel remains for this iteration. Prospective
confirmation requires a data snapshot after 2026-08-05 that was not inspected
during model development.

## Coordinate Pocket Audit

The coordinate pocket audit is diagnostic leakage evidence, not a pocket-cold
benchmark.

| Audit item | Count |
|---|---:|
| Compound-target pairs | 149 |
| Pairs covered by coordinate audit | 119 |
| Leakage max directional TM >= 0.4 | 83 / 149 |
| Leakage max directional TM >= 0.5 | 44 / 149 |
| Leakage max directional TM >= 0.6 | 19 / 149 |

Because the audit detects coordinate similarity in many evaluated pairs, it
should be reported as leakage diagnostics. It does not demonstrate performance
on pocket-cold targets.

## Claims Allowed

Allowed:

- The 2025 temporal frozen test selected model passes the configured temporal
  gate.
- The full 15-case known-skin no-known-target-prior retrospective panel fails
  overall because Top30 regresses.
- The dual-cold integrated panel shows no practical Top10 or Top30 recovery.
- The coordinate pocket audit is diagnostic leakage evidence.
- The ESM2 sequence-transfer candidate passes its registered 2024 development
  gate, with the small absolute effect size reported alongside the relative
  improvement.
- The ESM2 2025/known-skin/dual-cold results are previously inspected post-hoc
  diagnostics that reject promotion.
- Assisted production workflow is a separate operational setting from the
  unassisted frozen benchmark.
- The operational gate retains the frozen `chembl_p5_max` baseline for Stage 3
  exploratory use; it is not a scientific performance gate.

Not allowed:

- Claiming overall frozen validation PASS.
- Claiming implementation PASS from the current known-skin validation.
- Claiming SOTA target identification.
- Claiming clinical efficacy, clinical safety, or diagnostic utility.
- Treating `final_score`, `docking_rrf`, `skin_score`, or other retrieval
  ranking scores as probabilities.
- Claiming cold-start recovery from the dual-cold panel.
- Claiming that the ESM2 candidate passed an untouched test or is production
  validated.
- Using the previously inspected post-hoc panels to tune or promote the ESM2
  recipe.
- Treating the coordinate pocket audit as a pocket-cold benchmark.

Only calibrated potent-interaction event metrics, such as Brier score and
log-loss in the temporal frozen test, are probabilistic. Retrieval scores are
relative ranking signals within a run.
