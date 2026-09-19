# SkinScout Claim-Quality Execution Plan

This document turns the current SkinScout workflow into an iteration loop for
computational cosmetic target discovery. The operating rule is fail-closed:
scientific outputs that are empty, skipped, degraded, or backed by missing
tools are not allowed to look like successful evidence.

## Success Criteria

- Stage 2.5 cosmetic and drug-avoidance gates are enforced before docking.
- Stage 2.5 claim-quality gates require non-empty CosIng and approved-drug
  reference assets; missing references may only emit degraded diagnostics.
- Stage 0 claim-quality verification rejects placeholder CosIng, approved-drug,
  skin-score, scaffold, and skin-efficacy KG references.
- Stage 0 skin-expression ingestion uses HPA as the default SkinScore base and
  fails closed on empty SkinScore output; optional GTEx/proteome enrichment
  fails closed when configured as required or when provided but malformed.
- Comprehensive mode ranks targets with at least 3 independent scorer families.
- Fast mode ranks targets with at least 2 independent scorer families.
- Stage 4 prefers RCSB post-cutoff holo structures with real bound ligands and
  records the PDB id, deposit date, ligand ids, and resolution in the manifest.
- Stage 1 protonation and xTB refinement fail closed by default; neutral/MMFF
  fallbacks must be explicitly configured and recorded as degraded evidence.
- ADMET-AI must be installed for claim-quality Stage 2 reports; unavailable
  records are allowed only in explicitly configured degraded runs.
- HuSSPred, STopTox, and Pred-Skin/Skin Doctor II must each provide usable
  skin-sensitization evidence for claim-quality Stage 2 consensus.
- PSICHIC must be installed and produce receptor scores when DTI sanity or fast
  mode is part of the evidence; header-only DTI tables are degraded diagnostics.
- Daina-Zoete fast-mode ligand-similarity scoring requires non-empty ChEMBL
  fingerprint and molecule-target activity references.
- DiffDock-L no-pocket blind docking treats a missing no-pocket target list or
  non-empty list with zero successful blind scores as a blocker.
- Stage 5.5 claimable executable evidence is PLIP + ProLIF agreement over
  ligand atoms in bound Boltz complex poses (2 of 2). The indices remain in
  the bound-complex ligand order and are not original-SDF indices, SMARTS,
  causal pharmacophore validation, or experimental efficacy evidence.
- Stage 5.6 has manifest-driven fine-tune/generation adapters, deterministic
  seeds, output lineage, and graph-isomorphism atom-mapping utilities. The
  opt-in `analog_gen.finetune` rule validates a frozen train/holdout split and
  passes a hash-pinned model manifest into generation. It remains fail-closed
  until the operator supplies and pins compatible REINVENT 4
  executable/plugins, prior/agent artifacts, staged-learning config, and a
  validated target-specific input/output mapping.
- Boltz-2 success reports include only targets with successful cofolding
  payloads and fail closed if no target passes quality thresholds.
- Leakage audit reports `ok` only when sequence, ligand, and pocket axes are
  all backed by implemented checks and reference data.
- Analog and skin-efficacy recovery evaluations fail closed on missing or empty
  claim inputs. Interaction-atom conservation evaluation remains blocked until
  a validated target-specific atom-map sidecar exists.
- Direct evaluator threshold arguments must be finite values in `[0, 1]`; case
  identifiers and semicolon-delimited ground-truth lists must be unique and
  free of empty tokens.
- Iteration manifests validate direct evaluator row statuses and never treat
  `no_ranking` diagnostics as passing claim evidence.
- Stage 11 publication figures/manuscript fail closed on missing claim
  artifacts or scaffold text unless placeholder output is explicitly enabled
  for drafts.
- Stage 11 reproducibility packs validate the claim-ready eval manifest and
  include its SHA-256 digest alongside source artifact digests.
- Stage 11 manuscript drafts require source-backed captions, reproducibility
  artifact manifests, and DOI-backed data availability unless explicitly
  generated as placeholder drafts.
- Stage 0 verification covers both v2 structural infrastructure and v3 cosmetic
  data assets.
- Every claim-quality iteration records input compound, data snapshot, tool versions,
  leakage status, ranking metrics, and top target rationale.
- Claim-quality known skin-compound validation uses a fixed 15-case panel, records
  skin context/effect direction/evidence grade per case, and gates claims at
  case Top-10 >= 0.80, target-pair Top-10 >= 0.50, and target-pair Top-30 >=
  0.60 unless an explicit diagnostic threshold override is set.
- Target-sequence transfer selection must use a complete target-cluster-cold
  development panel, report both micro and target-macro Top10/Top30/MRR, and
  improve regular plus unsupported-target Brier and log-loss. A relative gain
  with negligible absolute Top10/Top30 remains development evidence only.
- A target-sequence candidate is not promotable when previously inspected
  temporal, known-skin, or dual-cold diagnostics regress. Prospective promotion
  requires a later snapshot that was neither inspected nor used for model or
  parameter decisions.

## Iteration Loop

1. Freeze inputs.
   - Record compound SMILES/SDF, run_id, config hash, git commit, and data
     manifest checksums.
   - Run `scripts/stage0_verify.py --strict --claim-quality` before accepting
     any claim-quality run.

2. Generate candidates.
   - Treat missing Dimorphite-DL or xTB as Stage 1 blockers unless an explicitly
     named degraded-mode experiment enables the corresponding fallback config.
   - Treat missing ADMET-AI as a Stage 2 blocker for claim-quality reports.
   - Treat missing skin-sens model evidence as a Stage 2 blocker unless an
     explicitly named degraded-mode experiment enables unavailable models.
   - Treat missing PSICHIC or zero receptor DTI scores as a Stage 3 blocker when
     DTI sanity or fast mode is enabled.
   - Treat missing or empty ChEMBL fingerprint/activity references as fast-mode
     blockers.
   - Run comprehensive mode first for primary evidence.
   - Run fast mode only as a throughput shortcut or disagreement probe.
   - Treat missing AutoDock-GPU, GNINA, RTMScore, or Boltz as run blockers
     unless an explicitly named degraded-mode experiment is being performed.
   - Treat incomplete precomputed AutoGrid4 map/field coverage (`.maps.fld` /
     `.fld`) as an AutoDock-GPU claim-path blocker. This repository checks map
     coverage but does not implement a map-generation stage or dependency.
   - Treat explicit Vina output as degraded diagnostics only; Vina cannot
     support target/report claims.
   - Treat Stage 5.6 itself as blocked until its REINVENT plugin/model/config
     assets and target-specific ligand atom-mapping validation are supplied;
     the repository adapters must never synthesize missing assets or infer an
     upstream REINVENT CLI contract.

3. Apply safety and cosmetic gates.
   - HALT from skin sensitization or cosmetic/drug avoidance stops docking.
   - Treat missing or empty CosIng, approved-drug, or approved-drug scaffold
     references as blockers unless a degraded diagnostic run is explicitly
     configured.
   - DOWNWEIGHT from drug avoidance remains visible in
     `ranked_targets_v3.csv` and scales the final score.

4. Fuse and rank.
   - Require scorer coverage using `source_count` and `sources` columns.
   - Inspect rank disagreement between AutoDock, GNINA, RTMScore, Boltz, and
     DTI sanity checks before promoting a target.

5. Evaluate leakage and retrospective validity.
   - Run `eval/collect_run_outputs.py` to normalize one or more
     `results/runs/<run_id>` directories into the evaluation harness layout.
   - Run `eval/run_iteration.py` to execute all available benchmark inputs and
     write `results/eval/iteration_manifest.json`.
   - Run `eval/leakage_check.py` without `--allow-incomplete` for claims.
   - `--allow-incomplete` is allowed only for diagnostic reports and must be
     labeled incomplete.
   - `eval/run_all.sh` keeps partial-input diagnostics
     (`EVAL_ALLOW_PARTIAL=1`) separate from threshold-failure diagnostics
     (`EVAL_ALLOW_THRESHOLD_FAILURE=1`); do not use partial mode to mask
     failing claim metrics.
   - Track top-k enrichment, rank correlation across scorers, novelty, and
     held-out target/ligand similarity.

6. Escalate top targets.
   - Promote only targets with strong skin expression, compatible cosmetic
     efficacy category, adequate pocket quality, and multi-scorer support.
   - Run Boltz/BioEmu/QM readiness stages for promoted targets and record any
     skipped stage as a blocker, not as neutral evidence.
   - Use the Stage 7 GROMACS preparation/run contracts only with target-specific
     complex poses and validated ligand topologies. Keep MD claim evidence
     fail-closed until protein+ligand topology merge, solvation, ions,
     minimization, restrained equilibration, production replicas, and output
     QA all complete successfully.
   - Treat missing or empty PLIP or ProLIF evidence as a Stage 5.5 blocker.
     Do not apply bound-complex atom indices to the original SDF or generator
     representation without a validated target-specific atom map.

7. Review and iterate.
   - Compare failures against the previous run.
   - Fix one bottleneck class at a time: data quality, missing tool, scoring
     calibration, leakage reference, or downstream physics validation.
   - Commit code/config changes separately from run artifacts.

## Minimum Commands

```bash
python -m pytest scripts/tests
python -m compileall -q scripts eval
python scripts/stage0_verify.py --strict --claim-quality
python eval/collect_run_outputs.py --run-dirs results/runs/<run_id>
python eval/skin_known_target_recovery_eval.py \
  --rankings-dir results/eval/rankings/skin_known_target \
  --out-csv results/eval/skin_known_target_recovery.csv \
  --out-target-csv results/eval/skin_known_target_recovery_targets.csv \
  --out-summary-json results/eval/skin_known_target_recovery_summary.json
python eval/run_iteration.py \
  --collected-runs-manifest results/eval/collected_runs.json \
  --skin-known-run-ledger-csv results/eval/skin_known_target_run_ledger.csv \
  --sota-baselines-csv results/eval/sota_baselines.csv \
  --sota-ablations-csv results/eval/sota_ablations.csv
snakemake -s workflow/Snakefile -n --cores 1 \
  --config compound_smiles='CC(=O)Oc1ccccc1C(=O)O' run_id=sota_dryrun
```

For a claim-quality run, use the same commit and data snapshot for all stages:

```bash
snakemake -s workflow/Snakefile --cores 16 --use-conda \
  --config compound_smiles='<SMILES>' run_id='<run_id>' mode=comprehensive
```

For the user-facing launcher, `target-id-sota` and `report-sota` are aliases
that keep the existing target-id/report output contracts while enabling strict
claim thresholds and context-profile provenance:

```bash
python scripts/run_skinscout.py '<SMILES>' \
  --preset target-id-sota \
  --mode comprehensive \
  --context-profile auto
```

When `skin_known_target_recovery` is part of the claim, the iteration manifest
is claim-ready only if three same-run freeze artifacts are present and valid:
`skin_known_target_run_ledger.csv` maps every panel case to the collected run
and copied ranking artifact, `sota_baselines.csv` freezes comparator fairness
under the active leakage thresholds, and `sota_ablations.csv` freezes the
required ablation families against the active panel checksum. `eval/run_all.sh`
passes these as `SKIN_KNOWN_RUN_LEDGER`, `SOTA_BASELINES`, and `SOTA_ABLATIONS`
and fails closed in strict mode when known-target rankings exist without them.

## Current Known Gaps

- Sequence leakage uses MMseqs2 `easy-search` when available and a validated
  cutoff FASTA fallback when the executable is unavailable; evaluation rows
  must carry a `sequence` column for that axis to be claim-ready. Pocket
  leakage consumes precomputed SuCOS CSV/TSV/JSON references and still fails
  closed when only raw holo archives are provided.
- Stage 4-8 wrappers now fail closed for empty/skipped outputs and omit failed
  target rows from successful reports, but target/report claims still require
  installed Boltz, BioEmu, GNINA, RTMScore, CREST, xTB, PySCF, complete
  AutoGrid4 map coverage for AutoDock-GPU evidence, and real full-run evidence.
  MD/GROMACS preparation and run contracts are implemented, but evidence is
  not claimable until target-specific complex poses, ligand parameters, real
  GROMACS systems, production replicas, and trajectory QA are verified.
- Placeholder cosmetic data is acceptable for DAG validation only. Target/report claims
  require full CosIng, claim-quality approved-drug references, HPA-backed
  SkinScore, and KG inputs. The full CosIng source is mirrored from the public
  EC API by default when no manual `data/cosing/cosing.csv` is present; set
  `cosing.auto_mirror_public_api: false` only when a curated operator export is
  mandatory. Skin proteome and GTEx strengthen SkinScore when supplied, and can
  be promoted to required evidence with the `skin_expression.require_*_source`
  flags. DrugBank is optional licensed enrichment when ChEMBL/Orange Book
  approved-drug coverage passes the same gates.
