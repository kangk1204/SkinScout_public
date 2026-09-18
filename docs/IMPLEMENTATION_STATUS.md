# 구현 상태 — 스테이지별

README에 있던 표를 옮긴 것입니다. 각 스테이지가 어디까지 돌고 무엇이 근거인지
적혀 있습니다. 파이프라인을 손보실 분을 위한 문서이고, 결과를 읽기만 하실 때는
볼 필요가 없습니다.

## Stage status

Host: i7-13700K (16 threads) · RTX 3080 Ti 12 GB · 123 GB RAM · 5.2 TB SSD.

| Stage | Status | Evidence |
|---|---|---|
| 0 — Infrastructure | ✅ | 20,171 cleaned receptors · 20,171 P2Rank pocket manifests (15,038 with-pocket; 5,133 no-pocket) · **15,038 PDBQT receptors** · CosIng mirror 36,832 rows / claim-quality parquet 672 rows · drug-avoidance 811,894 drugs / 286,034 scaffolds · MMseqs cutoff DB 662,415 protein sequences from 190,466 PDB entries (≤2021-09-30); claim-quality verifier rejects placeholder cosmetic/drug/KG references |
| 1 — Compound prep | ✅ | Aspirin: standardize → pH 7.4 deprotonate (C(=O)O→C(=O)[O⁻]) → conformer → xTB canonical; Dimorphite-DL/xTB fallbacks are fail-closed unless explicitly enabled |
| 2 — ADMET + skin-sens | ✅ | ADMET-AI + current HuSSPred/STopTox/Pred-Skin adapters, launcher safety-readiness gate, 3-model consensus (HALT/FLAG_HIGH/PASS), PAINS/Brenk/NIH; post-run verifier requires core ADMET endpoints and all 3 skin-sens source evidence files |
| 2.5 — Cosmetic + drug | ✅ | CosIng EXACT/SIMILAR/ANALOG/NEW + drug strict/scaffold/soft warnings + 3-policy decision; missing/empty references are fail-closed |
| 3 — Target ID | code + contract tests ready | MODE-FAST freezes the Daina top 256, generates target/query-specific AutoGrid4 maps, exports real AutoDock-GPU poses, and scores those exact poses with GNINA. `daina_structural_targets.csv` is schema-validated and keeps Daina rank unchanged; structure/KG/skin evidence is annotation-only. Scientific performance claims still require frozen-panel and fresh-machine qualification. |
| 3 v3 — skin weighting | comprehensive path | Comprehensive mode retains its independently versioned skin-weighting/consensus contract; it is not used to reorder Demo Daina output. |
| 4–9 — Boltz-2/BioEmu/MD/QM/Mol\* | code+rules ready, physics claim gaps remain | Stage 4 uses RCSB post-cutoff holo lookup with AlphaFold fallback; launcher preflights Boltz-2/GNINA/RTMScore/PSICHIC plus AutoDock-GPU/Meeko/BioEmu/MD/QM env readiness before `report` runs. Stage 7 now has deterministic GROMACS preparation/run contracts for target-specific complex poses, ligand topology, solvation/ions, minimization, equilibration, and production replicas; it remains fail-closed until real systems and trajectories pass QA. Mol\* HTML includes input SMILES/SDF provenance, input canonical SMILES when available, explicit overall decision/action, claimable status, high/moderate ADMET risk endpoints, drug warnings, skin-toxicity status from `Skin_Reaction`, skin-sens 3-model call/probability evidence, structural-alert/degraded-evidence markers, screened target candidate/stage counts, aggregate skin-context support, top binding target gene/protein labels when available, top binding target skin-context support, top-target binding score/source support, and most skin-relevant target gene/protein plus score/source support |
| 5.5 — Interaction atoms | executable evidence | PLIP + ProLIF agreement over ligand atoms in the same bound Boltz complex pose (2 of 2). A fail-closed graph-isomorphism sidecar maps complex indices to parent SDF order and then to canonical-SMILES atom order; ambiguous/symmetric mappings are rejected. These are parent pose-supported interaction anchors, not causal efficacy evidence. |
| 5.6 — REINVENT analog | manifest-driven, externally gated | The fine-tune and generation adapters, output lineage, seed/model/plugin/config manifests, and fail-closed diagnostics are implemented. Generation now requires a hash-bound interaction-anchor map and requires the plugin command to receive it. Setting `analog_gen.finetune.enabled=true` wires the frozen train/holdout split through the DAG and feeds its hashed model manifest into generation. Claim-capable execution still requires operator-supplied pinned REINVENT 4 executable/plugins, prior/agent artifacts, staged-learning config, and a validated input/output run. |
| 5.6b — Pharmacophore substitute baseline | executable, hypothesis-only | Verified parent `target-id` run followed by RDKit Gobbi 2D ligand-feature/feature-family comparison over CosIng and same-target ChEMBL, BindingDB, and GtoPdb ligands. When a matching Stage 5.5 sidecar is supplied or present in the run, ranking also uses target-conditioned parent-anchor MCS/feature preservation (10% weight) while explicitly keeping `analog_pose_verified=false`. Full InChIKey mismatches, including stereo/detail-layer mismatches, are rejected from direct evidence after SMILES standardization. `direct_retained` requires matching activity endpoint and source and uses the most conservative matched-stratum pActivity delta. Outputs include JSON/CSV/3D SDF/interactive HTML/Markdown and always require prospective target, permeation, safety, and formulation assays. |
| 5.7 — Core-retention ingredient retrieval | executable, retrieval-only | The project title - *활성 핵심구조 유지형 대체소재 발굴* - asks the opposite question from the rest of the pipeline, so it has its own entry point: the Workbench `대체소재 검색` view (nav `03`, `POST /api/compound/alternatives`) and `scripts/alternative_ingredients.py`. Population is the CosIng registered library: 37,071 registered entries -> 10,120 with a resolved structure -> 7,485 distinct InChIKeys after merging the names, CAS, and declared functions that share a structure (verified against `scripts/alternative_ingredients.load_ingredient_library()`). Fingerprints are recomputed with `build_activity_retrieval_index._standardize_mol` + `MORGAN_GENERATOR`, because the stored `ecfp4` column was built without standardisation and is not comparable to the retrieval index. Ranking is by core-retention grade first and Tanimoto only inside a grade: MCS coverage over the *query* heavy atoms says the core survived, MCS share over the *candidate* says the candidate is made of that core, and Bemis-Murcko scaffold identity is a third, weaker signal. The grade boundaries (coverage 1.0/0.75/0.5/0.25, share 0.5/0.35) are chosen, not calibrated, so every row carries the raw coverage and share beside its grade. `--with-evidence` joins each candidate to `data/similarity_index_202609` on the full InChIKey first and only falls back to the 14-character connectivity block when the molecule's own row is absent, flagging that row as a connectivity match: matching on connectivity alone let a better-measured stereoisomer stand in for a molecule whose own measurement was present (3-Methylfentanyl showed pAct 10.7 instead of its own 8.16), and 52,845 connectivity blocks in the index cover more than one full key, 4,226 of them spanning at least two log units. Exact-first cut the connectivity-only attachments from 9 of 51 to 4; most registered ingredients have none, and absence is rendered as "not measured", never as inactive. Structural retention is not measured activity and the CosIng function field is a self-declared registration entry, so no efficacy claim is made or claimable here. One query scans the whole 7,485-structure library with MCS; the first `--with-evidence` call additionally opens the 272 MB index. No target prediction is involved. When the query itself is in the activity library, each candidate also reports the targets *both* were measured against - the closest thing this view has to decision-grade evidence, and the reason the compact index stores every target per ligand (`all_targets`, schema `skinscout.similarity-index.v2`) rather than only the best three. Queries above 200 heavy atoms are refused: no registered ingredient is that large, and the Murcko computation alone costs 1.7 s at 1,200 atoms, which one request multiplied by every candidate before the query-side work was hoisted out of the loop. MCS runs with `BondCompare.CompareOrderExact`, because the default treats an aromatic ring bond as matching a single bond and graded a fully saturated salicylic acid analogue at 100% retention. A per-request MCS budget (default 600 s; the test suite lowers it) bounds pathological inputs, and any candidate it cuts short is reported as unjudged rather than as "different" - the summary line then says how many of the 7,485 were actually compared. `scripts/alternative_ingredients.py --input-csv` screens a list of compounds in one pass, keeping one row per query for the ones that could not be read or had no candidate. `data/cosing/` and `data/similarity_index_*/` are build products absent from a fresh clone (both gitignored); rebuild with `scripts/stage0_cosing.py --out-dir data/cosing` and `scripts/build_similarity_index.py --index-dir data/activity_retrieval_runtime_merged_202608 --out-dir data/similarity_index_202609`. An optional second signal reads pharmacophore features - Gobbi 2D fingerprint Tanimoto plus feature-family recall over Donor/Acceptor/Aromatic/Hydrophobe/PosIonizable/NegIonizable - by calling `discover_substitutes`' own functions, so the two screens cannot drift into meaning different things by the same name (a contract test asserts they agree). It is reported beside the structural grade and never folded into it: over the earlier 507-structure library snapshot the Spearman correlation between the two rankings ran from 0.09 (niacinamide) to 0.52 (salicylic acid), and the two top-ranked candidates differ in both cases, so the screen computes and states that correlation per query rather than presenting a combined score. 5.6b's gate values (pharmacophore_score >= 0.55, feature_recall >= 0.60) are deliberately NOT imported: they are tuned for a pool of compounds with reported activity against the same target, and applied to the registered-ingredient library they passed 0-1 candidates per query on the earlier snapshot, so this view measures without adjudicating. Building the library's fingerprints costs 3.7 s once, then 0.02 s per query, which is why it is off by default. Distinct from 5.6b, which requires a parent target-id run first and widens the pool to same-target bioactivity references, and from 5.6, which generates new structures. |
| 7.5 — AiZynthFinder retro | manifest-driven, externally gated | Route scoring, executable/config/model/stock manifest validation, and provenance are implemented. The end-to-end analog path remains gated on generated candidates plus pinned AiZynthFinder model, stock, config, executable, and non-empty route outputs. |
| 11 — Publication output | scaffold implemented | Source-backed figures, repro pack, and manuscript skeleton are implemented; a non-placeholder claim package is blocked whenever required upstream analog/MD/retrosynthesis or eval evidence is unavailable. |
| Eval harness | runtime binding diagnostic; frozen evaluation claim gate pass | PoseBusters, PLINDER, cold-start, cosmetic retrospective, skin known-target, KG, and leakage gates are implemented. The activity-retrieval recipe `union_discount_light` is applied by Stage 3: `--recipe`/`--recipe-index-dir` reach `scripts/stage3_daina_zoete.py` from `workflow/rules/stage3b_fast.smk`, scoring against a production-role index built from run-path evidence (ChEMBL 37 + BindingDB-curated, 4,873 targets). The active gate decisions and the measured known-panel figures are generated in "현재 운영 상태" below, not written by hand: the runtime operational binding is diagnostic (`claim_ready=false`) because the production index is not the frozen evaluation index, while the frozen evaluation decision passes with `union_discount_light`. The discount keeps analogues measured at or below the activity threshold in the score rather than excluding them, which keeps the tyrosinase pairs inside Top30 without costing EGCG -> MMP2. The promotion of `union_p6_consensus` described in older snapshots is history; no run-path code applied recipe weights at the time. Stage 11 and all performance/SOTA claims remain blocked by the stricter claim gate. The pharmacophore evaluator now computes target-conditioned parent-anchor retention only from a validated atom map and emits per-analog/per-target details, but remains non-claimable because analog poses are not verified. |
| Beginner release delivery | pre-production | Conda 설치, Workbench 런처, readiness gate는 로컬 회귀검증 대상이다. Compose/CAS 계약과 fail-closed doctor는 구현됐지만 실제 digest-pinned Cosign-signed 이미지, 공개 CAS bundle, fresh Ubuntu 24.04 + RTX 4090 qualification은 아직 외부 release gate다. |

The README does not pin a helper-test count; use the current project test
environment or CI output for live status.

The similarity index at `data/similarity_index_202609` was rebuilt on 2026-09-01
after `scripts/build_similarity_index.py` gained a
`drop_duplicates(["ligand_index", "uniprot"])` before its per-ligand `head(3)`.
Without it, 35,264 rows spent all three of their top-target slots on the same
UniProt, so a candidate's "measured against" list understated how many distinct
targets it had actually been tested on.

Goal-level proof for the current SMILES-to-target, ADMET, skin-toxicity, and
skin-specialized binding scope is tracked in
[`docs/GOAL_EVIDENCE.md`](./docs/GOAL_EVIDENCE.md).
Use `scripts/verify_goal_contract.py --run-dir results/runs/<run_id>` to
machine-check that a completed run satisfies that goal contract.
Known skin-beneficial/adverse compound target recovery is tracked separately in
[`docs/SKIN_KNOWN_TARGET_VALIDATION.md`](./docs/SKIN_KNOWN_TARGET_VALIDATION.md).

## 현재 운영 상태 (자동 생성)

아래 블록은 활성 게이트·recipe·해시로 결속된 summary에서 생성한다. 숫자를 손으로
고치지 말고 `python scripts/render_current_status.py --write`로 다시 만든다.

<!-- BEGIN GENERATED: current-operational-state -->
- 활성 recipe: `union_discount_light` — `results/eval/activity_retrieval_20260914_identity/dev_selection/recipe.json` (sha256 `93ed753559b1f45eeca3aa820abc114626dd37642a407e6462a20996722d81c9`)
- 런타임 검색 인덱스: `data/activity_retrieval_runtime_merged_202608` (운영 게이트가 sha256으로 결속)
- operational gate `data/manifests/activity_retrieval_operational_gate.flag` (sha256 `513c7b261d0ac81047414f4944d7ab76a2aa6990a23310d457ec3aa01986feb3`): runtime decision `diagnostic_runtime_binding`, `claim_ready=false` — 런타임 인덱스는 동결 평가 인덱스가 아니므로 동결 평가 수치를 운영 성능 주장으로 쓸 수 없다.
- 같은 게이트의 evaluation decision: `promote_selected`, operational recipe `union_discount_light`, `claim_ready=true`; claim gate `status=pass`.
- 해시로 결속된 `results/eval/activity_retrieval_20260914_identity/test_evaluation/summary.json` (sha256 `0dad81685a2343eb1237bded0371e0d90b103559e8467fb1472dcabfb6266817`)의 46쌍 known panel: baseline `chembl_p5_max` Top10 18/46 · Top30 31/46; selected `union_discount_light` Top10 26/46 · Top30 34/46.
- temporal 2025: baseline Top10 187/301 · Top30 216/301 · MRR 0.43224; selected Top10 222/301 · Top30 237/301 · MRR 0.56187.
- dual-cold: baseline MRR 0.00008293, selected MRR 0.00008095; 회수 주장 없음.
- 밴드 재정렬은 현재 기본값에서 꺼져 있다. (구조 주석은 11–50위 밴드를 그대로 사용)
<!-- END GENERATED: current-operational-state -->

> **데이터 재생성 대기 (F24·F25):** 저장된 runtime index manifest는 per-source
> release/license/source-manifest metadata가 추가되기 전 버전이다. 별도 재생성
> 작업이 끝날 때까지 `validate_activity_retrieval_gate.py check-operational`은
> fail-closed로 거부하며, `scripts/tests/test_operational_gate_on_disk.py`도 그
> 상태를 검증한 뒤 재생성 대기로 skip한다. F25가 지적한 ChEMBL evidence의 PMID
> `.0` 문자열 정리(F05)도 같은 재생성 작업에 포함된다.

## 검증 상태

Current validation status is deliberately conservative. 현재 gate·recipe·수치는 위
자동 생성 블록이 유일한 출처이며, 아래 항목은 현재 산출물에 결속되지 않는 보관
실험을 역사로 표시한 것이다.

- **ESM2 target-sequence transfer development experiment (보관됨; 현재
  performance_v2 산출물 없음):** the preregistered
  `esm6_knn_k16_p2_w075` candidate passes the full 2024 target-cluster-cold
  development gate over 672 queries and 677 truth pairs. It changes Top10 and
  Top30 from 0 to 0.001477 and MRR from 0.00008413 to 0.00050222; target-macro
  Top10/Top30 change from 0 to 0.03704. Both regular-dev and dedicated
  unsupported-target calibration Brier/log-loss also improve. This is a weak,
  concentrated signal: one target accounts for 58.1% of truth pairs, the
  effective target count is 2.71, only one truth pair enters the top 30, and
  ECE worsens.
- **Archived ESM2 post-hoc diagnostics, not a new frozen test:** these numbers
  belong to the earlier evaluation snapshot inspected before the current
  rerun. In that snapshot, temporal Top10 fell from 0.72241 to 0.71906,
  known-skin Top30 fell from 0.5000 to 0.4375, and dual-cold Top10/Top30 stayed
  at 0 despite MRR increasing from 0.00008416 to 0.00025753. Unsupported-target
  calibration also worsened. They remain rejection evidence, not current
  benchmark evidence or a basis for promotion.
- **Coordinate pocket audit:** diagnostic leakage audit only, not a
  pocket-cold benchmark; 149 pairs, 119 covered, leakage max directional TM
  >= 0.4 for 83/149, >= 0.5 for 44/149, and >= 0.6 for 19/149.

The project therefore makes no SOTA, clinical, or probability claim for target
retrieval. Retrieval scores rank candidates within a run; they are not
probabilities. Only calibrated potent-interaction event metrics such as Brier
and log-loss are probabilistic. Assisted production workflow, where an operator
may add licensed data, manual curation, or follow-up review, must be reported
separately from the unassisted frozen benchmark. No untouched evaluation panel
remains for the ESM2 iteration; prospective confirmation requires a data
snapshot after 2026-08-05 that was not inspected during model development.

활성 operational gate의 결정과 현재 수치는 위 "현재 운영 상태" 자동 생성 블록에
있다. 아래 문단은 2026-08-31 `union_any_consensus` 승격 경위의 역사 기록이며, 그
recipe는 이후 `union_discount_light`로 대체됐다.

### 역사: 2026-08-31 승격 경위 (현재 상태 아님)

Getting there took two corrections. The frozen `chembl_p5_max` baseline had been
retained because the 15-compound panel regressed at Top30 - a four-pair
difference exact McNemar cannot resolve, measured on the superseded v1 panel
whose structures were corrected five days later. But the signal underneath was
real: every candidate recipe weighted only evidence above pActivity 5 or 6, and
cosmetic actives sit below that, so all of them lost tyrosinase. Adding a
threshold-free feature (`max_union_any`) produced a recipe that wins on both the
256-query dev panel and the 46-pair skin panel, measured through the run path
itself: Top30 24/46 -> 34/46, p=0.0063.

It costs one pair - alpha-arbutin to tyrosinase, rank 2 to 70 - which is stated
in `docs/RECIPE_RUNPATH_MEASURED_20260831.md` and beside the config switch.
Setting `docking.daina_recipe_scoring` to false returns to nearest-neighbour
scoring. The gate still does not authorize publication, comparative performance,
SOTA, clinical, or probability claims; those require the strict claim gate.

