# Line-by-line code review — 2026-08-23

Scope: the repository at `569dfed` (`harden-workflow-gates` branch).
Reviewer environment: `cosmax-boltz2` (pytest 9.1.1, pandas 2.3.3, numpy 1.26.4),
`jsonschema` supplied out-of-tree because it is absent from that env.

## Method

- Full read of the new and load-bearing logic: `stage3_daina_structural_overlay`,
  `stage3_select_daina`, `stage3_skin_weighting`, `stage3_rrf`, `stage3_daina_zoete`,
  the statistical core of `performance_v2_model`, the gate paths of
  `activity_retrieval_model`, `skin_known_target_recovery_eval`,
  `skinscout/contracts/run_profile`, the verdict logic of `verify_run_outputs`,
  and the request surface of `workbench/server`.
- Whole-tree static analysis (`ruff --select=F,E9,B,ARG,RUF,PLE,PLW`).
- Full test suite before and after.
- Every quantitative claim in `README.md` checked against the committed result JSON.

## Results verification

All README numbers reproduce exactly from
`results/eval/activity_retrieval_202608/test_evaluation/summary.json`:

| Claim | README | Committed artifact |
|---|---|---|
| temporal 2025 Top10 (selected vs baseline) | 0.71141 / 0.60067 | 0.7114093959731543 / 0.6006711409395973 |
| temporal 2025 Top30 | 0.77852 / 0.71141 | 0.7785234899328859 / 0.7114093959731543 |
| temporal 2025 MRR | 0.54394 / 0.41727 | 0.543936923423584 / 0.4172727331586758 |
| temporal 2025 Brier | 0.06673 / 0.07031 | 0.06672800336569214 / 0.07030835074396907 |
| temporal 2025 log-loss | 0.24916 / 0.26835 | 0.24916065020689446 / 0.2683491401899628 |
| known-skin 15-case Top10 | 0.21875 / 0.1875 | identical |
| known-skin 15-case Top30 | 0.500 / 0.625 | identical (drop = 0.125) |
| known-skin 15-case MRR | 0.14740 / 0.13623 | identical |
| dual-cold Top10/Top30 | 0 / 0 | identical |
| dual-cold MRR | 8.416e-5 / 8.293e-5 | identical |

No overstated claim was found. The RRF implementation matches Cormack et al.
exactly; the substitute `priority_score` weights sum to 1.0 in both the
anchor-present and anchor-absent branches; the calibration holdout is
pair-hash-stratified and train-only; PU labelling refuses to mark unmeasured
evidence negative.

## Defects found and fixed

| # | Location | Defect | Commit |
|---|---|---|---|
| 1 | `stage3_daina_structural_overlay.py` | JSON `null` became the literal string `"None"` in the published CSV and in `structure_annotation_json`; `pose_sha256="None"` would defeat the hash audit | `56ed43d` |
| 2 | `stage3_daina_structural_overlay.py` | Structural status was never cross-checked against score presence: a `structure_failed_docking` row could publish an AutoDock energy, and a `docked` target with a missing score row was relabelled `structure_failed_gnina` | `56ed43d` |
| 3 | `stage3_daina_structural_overlay.py` | Non-numeric skin/prior cells raised a bare `ValueError` instead of the module's `SystemExit` | `56ed43d` |
| 4 | `stage3_select_daina.py` | `daina_max_tanimoto` silently fell back to the Daina score, which under `quality-hybrid` is `0.7*max_tanimoto + 0.3*quality` | `56ed43d` |
| 5 | `skin_known_target_recovery_eval.py` | `target_top{k}` averaged over evaluated rows while MRR/coverage used all rows — two panel sizes in one summary block | `6800c0b` |
| 6 | `skin_known_target_recovery_eval.py` | `_target_rank_map` renumbered by row position, discarding the validated rank column; a `global_rank` slice would report better ranks than were produced | `6800c0b` |
| 7 | `activity_retrieval_model.py` | `select_recipe` computed the dual-cold adequacy verdict and neither used nor recorded it | `bdb4623` |
| 8 | `performance_v2_model.py` | `effective_target_count` returned NaN when any target's weights summed to zero | `b0a91f3` |
| 9 | `stage3_skin_weighting.py` | Negative prior + fractional `--known-target-prior-power` produced a complex number and an unrelated `TypeError` | `b0a91f3` |
| 10 | `performance_v2_model.py` | `smoothed_prior_logit` error message contradicted its own check | `b0a91f3` |
| 11 | `test_workflow_config_fail_closed.py`, `test_run_skinscout.py` | 37 cases failed hard when `snakemake` was absent instead of skipping | `d866eab` |
| 12 | 46 sites | `zip()` over parallel sequences without `strict=`; six unused imports | `c854d99` |

## Behaviour preservation

Replaying both committed 2026-08-22 recovery panels (`daina_retrieval` and
`daina_leave_query_out`) against the fixed evaluator reproduces **all 127
published metric values exactly**. All 205 committed ranking CSVs carry
contiguous ranks, so defect 6 was latent rather than active.

## Test suite

| | Before | After |
|---|---|---|
| passed | 2430 | 2445 |
| failed | 37 (all `snakemake` absent) | 0 |
| skipped | 2 | 39 |

The 37 previously-failing cases now skip with a reason naming the environment
that can run them (`cosmax-base`). 15 regression tests were added covering the
fixes above.

## Not changed

- `cold_start_eval` and `cosmetic_retrospective_eval` score top-k positionally
  by design. That assumption is now stated at the point it is made rather than
  altered.
- `verify_run_outputs` keeps one deliberate `zip(..., strict=False)`: an
  over-long `top_targets` list is already recorded as an invalid check, and that
  loop must keep collecting findings rather than raise.
- `cosmetic_downweight_factor` scales every row by the same constant, so it
  changes the absolute `final_score` but never the within-run ordering. It is
  recorded per row for the claim gates; the code now says so.
