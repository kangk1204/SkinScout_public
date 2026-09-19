# 운영자용 CLI 참고

> 이 문서는 **파이프라인을 운영하는 사람**을 위한 것입니다. 브라우저에서 분석만
> 하는 연구자는 볼 필요가 없습니다 — [`README.md`](../README.md)의 Quick Start를
> 보세요.

Stage 0 인프라 구축, 준비성 점검, CLI 실행, 외부 도구 설치(RTMScore·PSICHIC·
DiffDock)를 다룹니다.

---

일반 사용자는 위의 `처음 사용하는 분` 절과 Workbench만 사용하면 됩니다.
서버를 브라우저 자동 열기 없이 직접 실행하려면 다음 명령을 사용합니다.

```bash
python3 scripts/run_workbench.py
```

아래 절은 Stage 0와 고급 모델 환경을 직접 관리하는 운영자용입니다.
명령줄에서 직접 실행할 때는 설치기가 만든 기본 환경을 활성화하거나 같은
환경의 Python을 사용합니다. 이 저장소에서 검증한 예시는
`cosmax-base` 환경 기준입니다.

### 1. First-time infrastructure (Stage 0)

```bash
# Base env (snakemake, p2rank deps, mmseqs2, rdkit, openjdk-21, ...)
mamba env create -f envs/base.yml
conda activate cosmax-base
pip install dimorphite-dl   # Stage 1 extra for --no-use-conda operator runs

# P2Rank 2.5 (not on conda — GitHub tarball + wrapper)
bash scripts/setup_p2rank.sh 2.5
```

Full-data Stage 0 mirrors the public EC CosIng search API into
`data/cosing/cosing.csv` when that file is absent, then ingests it into the
claim-quality CosIng reference parquet. To force a manually curated CosIng
export instead, set `cosing.auto_mirror_public_api: false` and import
`cosing.csv` before the build.

HPA is downloaded by Stage 0 and is the default SkinScore base. The skin
proteome LFQ table and GTEx skin TPM matrix are optional enrichment by default;
provide `raw_lfq.tsv` and `gtex_v10_gene_tpm.gct` (or `.gct.gz`) when you want
those axes included, or set `skin_expression.require_proteome_source` /
`skin_expression.require_gtex_source` to `true` to make them blockers.
The HPA single-cell table is pan-tissue, so it never establishes skin relevance
by itself. Its score and `cell_type_preferred` label are used only for proteins
with positive expression in the HPA skin-tissue table. For those proteins, the
label reports the highest positive HPA expression among the configured
skin-related cell categories; ties are deterministic, and absent, zero, or
skin-tissue-unsupported evidence remains `unknown`. This is an indirect
expression-context label, not proof of skin specificity or cell-selective drug
action.
Licensed DrugBank XML is also optional enrichment, not a default blocker. Provide
`drugbank_full_database.xml` or `drugbank_full_database.xml.zip` when you want
DrugBank polypharmacology/approved-drug slices; otherwise approved-drug
avoidance can be satisfied by ChEMBL/Orange Book outputs if claim-quality gates
pass.

Put any manual/optional source files in one staging directory, then import them
into the canonical `data/...` locations and record SHA-256 provenance:

```bash
python scripts/import_stage0_sources.py --source-dir /path/to/stage0_sources
python scripts/data_readiness.py --preset stage0
```

The importer accepts either flat filenames or canonical subfolders such as
`cosing/cosing.csv`, `skin_proteome/raw_lfq.tsv`, and
`gtex_v10/gtex_v10_gene_tpm.gct`.
It also recursively recognizes common download names such as
`CosIng - Glossary of Ingredients.csv`,
`drugbank_all_full_database.xml.zip`, `*skin*proteome*lfq*.tsv.gz`, and
`*gtex*v10*gene*tpm*.gct.gz`; if multiple candidates match, pass the explicit
`--cosing-csv`, `--drugbank-xml`, `--skin-proteome-tsv`, or `--gtex-gct`
argument. The GTEx file must already expose skin-labelled TPM columns because
the Stage 0 slicer does not infer tissue labels from a separate sample
annotation table.
It writes `data/manifests/stage0_source_manifest.json` with source/target
paths, byte counts, SHA-256 digests, and minimal format-validation details.

Then start the long infrastructure build:

```bash
# Build infrastructure (≈ ½ – 1 day on this host)
python scripts/run_skinscout.py \
  --preset stage0 \
  --run-id stage0_bootstrap \
  --cores 16 \
  --allow-stage0-build
```

Outputs land under `data/` (AlphaFold proteome cleaned, P2Rank pockets, Meeko
PDBQT, fixed ChEMBL 37, BindingDB 2026-08, GtoPdb 2026.2, PubChem 2026-08-01
exact-alias data, optional DrugBank slices, CosIng + approved-drug +
skin-efficacy KG). These generated products and imported operator sources are
local-only and gitignored; the repository intentionally does not ship
placeholder CosIng, approved-drug, skin-proteome, skin-expression, or KG
references for full-data runs.

### 2. Check local data readiness

Stage 0 products are not committed to Git. Before a normal compound run, check
that this machine has the receptor, docking-box, skin-expression, CosIng, drug,
and skin-efficacy artifacts required by the selected preset:

```bash
python scripts/data_readiness.py --preset stage0
python scripts/data_readiness.py --preset target-id --mode comprehensive
python scripts/data_readiness.py --preset report --mode comprehensive --json
```

The command exits non-zero until the required artifacts exist, are non-empty,
and pass claim-quality checks that reject placeholder CosIng/drug/KG references
and undersized skin-expression tables. It is the standalone form of the same
data preflight used by `scripts/run_skinscout.py`.

### 3. One-shot run readiness

Before spending GPU time, aggregate the exact SMILES input, Stage 0 data/source
state, safety runtime wiring, and model/tool readiness into one report:

```bash
python scripts/pipeline_readiness.py \
  --smiles "CC(=O)Oc1ccccc1C(=O)O" \
  --run-id aspirin_demo \
  --preset target-id \
  --mode comprehensive
```

The command does not run Snakemake. It exits non-zero while blockers remain,
prints the Snakemake command that would run, the expected run directory, the
completed-run `run_summary.json` / `run_summary.md` paths, persistent
`run_verification.json` / `run_verification.log` paths, the verifier command,
the required summary fields, compound-matched ADMET-AI / structural-alert /
skin-sens source evidence artifacts and invariants, target-ranking / skin-context
evidence invariants, report markers, the final
decision gate, and next actions such as optional Stage 0 source import or
infrastructure bootstrap. For `target-id`/`report` runs whose base data
readiness already passes, it also runs the claim-quality Stage 0 verifier
equivalent to `python scripts/stage0_verify.py --strict --claim-quality`. A run
is claimable only when the workflow succeeds, the verifier exits 0, Stage 0
claim-quality status is ok, `overall_decision.claimable` is true,
`overall_decision.recommended_action` is `proceed`, no diagnostic
non-claimable reasons exist from skipped readiness checks or degraded safety
evidence, and `target-id`/`report` runs have
`skin_specialized_binding.skin_context_supported` true and
`skin_specialized_binding.top_target_skin_context_supported` true;
`review_before_claim` requires human review and `stop_before_claim` is not
claimable.

### 4. Run a compound, MODE-COMPREHENSIVE (cosmetic default)

The SMILES can be provided as a positional argument or with `--smiles`.
`--run-id` is optional for SMILES runs. If omitted, the launcher derives a
deterministic safe output ID from the canonical SMILES, so a one-SMILES
invocation is enough:

```bash
python scripts/run_skinscout.py \
  "CC(=O)Oc1ccccc1C(=O)O" \
  --preset target-id \
  --mode fast \
  --evidence-mode evidence
```

Use `--run-id` when you want a human-readable output directory:

```bash
python scripts/run_skinscout.py \
  --smiles "CC(=O)Oc1ccccc1C(=O)O" \
  --run-id aspirin_demo \
  --preset target-id \
  --mode comprehensive \
  --evidence-mode evidence \
  --cores 16

# The launcher automatically writes run_summary.json / run_summary.md, verifies
# completed safety/target-id/report runs, and prints the final decision/action
# plus claimable status, ADMET risk, drug warnings, skin toxicity, ranked
# top-target proteins with score/source/skin-efficacy evidence, and the
# skin-context summary.
# Re-run the same summary + verification + final-result contract for archived
# output without rerunning Snakemake:
python scripts/run_skinscout.py \
  --smiles "CC(=O)Oc1ccccc1C(=O)O" \
  --run-id aspirin_demo \
  --preset target-id \
  --mode comprehensive \
  --evidence-mode evidence \
  --verify-existing-run
```

Launcher presets:

- `safety`: Stage 1–2.5 only; standardization, ADMET, skin sensitization,
  CosIng, and drug-avoidance gates.
- `target-id --mode fast`: safety gates plus a fixed Daina top 256, per-query
  AutoGrid4 map generation/cache, real AutoDock-GPU pose export, GNINA scoring
  of those poses, and annotation-only skin-expression/KG evidence.
- `target-id --mode comprehensive`: the separately versioned multi-model
  consensus path; Full profile dependencies are required.
- `report`: full `rule all`, including downstream structure/report stages.

Evidence-mode selector:

- `--evidence-mode evidence` is the default for known ingredients or positive
  controls. It allows source-backed direct records to support a run, while the
  normal claim-quality gates still apply.
- `--evidence-mode discovery` is for new-candidate screening. Before Snakemake
  starts, the launcher checks `data/discovery_aliases/direct_exact_reference.smi`
  and `data/discovery_aliases/manifest.json`. If the input is an exact direct
  reference match, the run stops and tells the operator to use Evidence mode.
  If the sealed alias package or the legacy Discovery leakage audit is missing,
  Discovery fails closed.

### 4A. Pharmacophore 유지형 대체소재 발굴

Workbench에서는 **새 분석 → Pharmacophore 대체소재**를 선택하면 됩니다.
CLI에서는 먼저 검증된 빠른 Target 분석을 실행한 뒤 대체 후보를 자동으로
생성하는 wrapper를 사용합니다.

```bash
python scripts/run_substitute_discovery.py \
  --smiles "CC(C)=CCCC(C)=CCCC(C)=CCO" \
  --run-id retinol_substitutes \
  --mode fast \
  --evidence-mode evidence \
  --build-interaction-anchors \
  --max-candidates 50
```

알려진 피부 유효성분으로 기능을 점검하려면 all-trans Tretinoin과 RARG를
사용할 수 있습니다. 아래 SMILES는 입체정보가 지정되어 있으므로 그대로
입력합니다.

```bash
python scripts/run_substitute_discovery.py \
  --smiles 'CC1=C(/C=C/C(C)=C/C=C/C(C)=C/C(=O)O)C(C)(C)CCC1' \
  --target-id P13631 \
  --run-id tretinoin_rarg_validation \
  --mode fast \
  --evidence-mode evidence \
  --max-candidates 50
```

타겟을 지정하지 않으면 parent Target ranking의 1순위를 사용합니다. 특정
UniProt accession을 사용하려면 그 타겟이 parent ranking에 포함된 경우에만
`--target-id P13631`처럼 지정할 수 있습니다. 이 검사는 사용자가 임의 타겟을
끼워 넣어 같은 실행의 근거 연결이 끊기는 것을 막습니다.

`--build-interaction-anchors`를 사용하면 wrapper가 parent Target 분석 뒤
Boltz-2, PLIP, ProLIF 및 원자 매핑 Stage 5.5를 실행하며, 검증된 sidecar가
생성되지 않으면 2D-only 순위로 후퇴하지 않고 중단합니다. 이미 동일
화합물·타겟에 대해 Stage 5.5가 만든 검증된
`interaction_anchor_map.json`이 있으면 표적별 3D parent interaction anchor를
후보 순위에 추가할 수 있습니다. 같은 run 폴더의
`05_pharmacophore/interaction_anchor_map.json`은 자동으로 사용되며, 다른
위치라면 명시적으로 전달합니다.

```bash
python scripts/run_substitute_discovery.py \
  --smiles 'CC1=C(/C=C/C(C)=C/C=C/C(C)=C/C(=O)O)C(C)(C)CCC1' \
  --target-id P13631 \
  --run-id tretinoin_rarg_anchor_validation \
  --interaction-anchor-map results/runs/reference_run/05_pharmacophore/interaction_anchor_map.json
```

sidecar의 parent InChIKey와 선택 타겟이 입력과 다르거나, 복합체→parent
SDF→canonical SMILES 원자 매핑이 유일하지 않으면 실행은 중단됩니다. 또한
parent SDF, consensus, Boltz report, 복합체 PDB의 기록된 크기·SHA-256이 현재
원본과 다르면 해당 anchor를 사용하지 않습니다.

후보 근거 단계는 다음과 같습니다.

| 근거 단계 | 의미 | 허용되는 해석 |
|---|---|---|
| `direct_retained` | parent와 후보에 같은 endpoint·source 활성값이 있고 가장 보수적인 pActivity 차이가 설정 범위 안임 | 동일 조건 실험의 우선 후보 |
| `direct_activity` | 후보의 동일 타겟 활성값은 있으나 비교 가능한 parent strata가 없음 | 타겟 활성 근거가 있는 후보 |
| `direct_reduced` | 비교 가능한 공개 활성값에서 감소가 관찰됨 | 저하 원인 또는 용도별 허용 범위 검토 |
| `proxy_only` | ligand-feature pharmacophore 유사성만 있음 | 결합 가설만 있으며 target assay가 우선 |

기본 후보 pool은 CosIng 참조 성분과 동일 타겟 공개 활성 ligand입니다.
결과 바스켓은 **엄격한 2D pharmacophore 트랙**, **표적 활성 근거 트랙**,
**CosIng 화장품 원료 트랙**을 분리합니다. 기본값은 엄격한 pharmacophore
후보에 20%, 원료 후보에 40%를 예약하며 실제 후보가 부족한 자리는 다른
트랙으로 채웁니다. `rank`는 이 균형 바스켓의 순위이고
`global_priority_rank`는 트랙 예약 전 전체 점수 순위입니다. 따라서 특정
트랙 후보를 포함하면서도 전체 점수 순위를 숨기지 않습니다.

Pharmacophore 점수는 RDKit Gobbi 2D feature-pair fingerprint와 parent
feature-family recall을 함께 사용합니다. feature-pair fingerprint가 비어
있는 저-feature 분자는 feature-family precision/recall의 F1을 fallback으로
사용하고 recall minimum gate를 별도로 적용합니다. 그래도 이는 ligand 기반
feature proxy이며 단백질 결합 자세의 pharmacophore를 증명하지 않습니다.
`pharmacophore_gate_passed=true`인 후보만 설정된 2D pharmacophore와 recall
gate를 모두 통과한 것입니다. 정확한 full InChIKey의 동일 타겟 공개 활성이
있고 feature-family recall만 통과한 후보는
`admission_bases=["same_target_activity_feature_family"]`로 별도 입장할 수
있지만 `pharmacophore_gate_passed=false`를 유지합니다. 이 후보를
“pharmacophore 유지”로 해석하면 안 됩니다.

검증된 interaction anchor가 있으면 우선순위 점수의 기존 2D
pharmacophore 30%를 **2D 20% + 표적 anchor 10%**로 나눕니다. anchor 점수는
parent의 Boltz 복합체에서 PLIP·ProLIF가 함께 지지한 원자가 후보의 exact-bond,
chirality-aware MCS 안에 남고 같은 RDKit feature family를 유지하는 비율입니다.
가능한 MCS 원자 대응이 여러 개면 가장 높은 값이 아니라 **모든 열거 대응 중
최소 보존값**을 사용하며, 대응 열거가 상한에 닿으면 0점으로 처리합니다.
이는 target-conditioned 구조 가설을 강화하지만 후보를 단백질에 다시 docking한
값은 아닙니다. 따라서 보고서의 `analog_pose_verified=false`와
`claimable=false`는 유지되며, “동일 결합 자세·결합력 유지”로 표현하면 안
됩니다.

구조 novelty는 입체정보를 포함한 ECFP4를 사용합니다. 따라서 all-trans,
9-cis, 13-cis처럼 정의된 이성질체를 같은 구조로 잘못 제거하지 않습니다.
반대로 parent가 입체적으로 정의됐는데 같은 connectivity의 후보가 입체정보를
덜 가진 경우에는 모호한 대체물로 보고 제외합니다. ChEMBL/GtoPdb 후보명도
full InChIKey가 정확히 일치할 때만 보강합니다.

합성 용이성 값은 분자 복잡도 proxy이며 실제 retrosynthesis route가
아닙니다. 더 넓은 생성 공간과 실제 합성 경로가 필요하면 별도로 검증된
REINVENT/AiZynthFinder 자산을 사용하는 고급 Stage 5.6/7.5 경로가 필요합니다.
후보의 `safety_triage_score`는 호환성을 위해 유지한 필드명이며 구조 경고와
일반 물성만 반영합니다. 피부 안전성, 감작성 또는 독성 예측값이 아닙니다.

주요 산출물은 다음 폴더에 생성됩니다.

```text
results/runs/retinol_substitutes/05_6_substitutes/
├── substitute_report.json
├── substitute_candidates.csv
├── substitute_candidates_3d.sdf
├── substitute_report.html
├── substitute_report.md
└── substitute_run_manifest.json
```

`substitute_run_manifest.json`은 검증된 parent summary/verification, 사용한
interaction-anchor sidecar의 fingerprint, 다섯 대체 산출물의 SHA-256을
연결합니다. Workbench는 manifest가 없거나 어느 한
파일이라도 크기·SHA-256이 달라지면 후보 보고서를 표시하지 않습니다. 공개
DB의 endpoint·source가 같아도 assay 조건은 서로 다를 수 있으므로 결과를
“결합력 유지 확정”으로 표현하면 안 됩니다.

To create the interactive 3D report, run the `report` preset:

```bash
python scripts/run_skinscout.py \
  --smiles "CC(=O)Oc1ccccc1C(=O)O" \
  --run-id aspirin_report \
  --preset report \
  --mode fast \
  --evidence-mode evidence
```

The main report path is `results/runs/aspirin_report/09_report/index.html`.
The same run also writes `run_summary.md`, `run_summary.json`,
`run_verification.json`, and `run_verification.log`.

For `target-id` and `report`, the launcher checks the Stage 0 receptor/skin data
prepared by the public installer before starting, so a normal compound run does
not unexpectedly trigger a large AlphaFold/P2Rank/PDBQT rebuild. Operators who
manage the pipeline without the public installer can run the Stage 0 command
above first, or add `--allow-stage0-build` when they explicitly want Snakemake
to generate missing infrastructure during the compound run. This flag only bypasses missing
generated outputs; placeholder, too-small, empty, or invalid references still
fail closed and must be replaced or rebuilt. The flag also requires the manual
Stage 0 source import to pass first when those sources are configured as
required; otherwise the launcher stops before the long Snakemake build. When
generated Stage 0 artifacts already exist, `target-id`/`report` launcher runs
also require `stage0_verify.py --strict --claim-quality` before Snakemake starts.
When `--allow-stage0-build` is used, the same claim-quality check runs after
Snakemake succeeds and before completed-run summary/report verification.

Readiness skip flags are guarded. The launcher accepts
`--skip-data-readiness`, `--skip-safety-readiness`, and
`--skip-model-readiness` for `--dry-run`; real Snakemake execution requires the
explicit `--allow-unsafe-readiness-skip` diagnostic override, and those runs
should not be treated as claimable prediction outputs. The aggregate
readiness-only report also accepts `--skip-stage0-source-readiness` for
diagnostics. Readiness reports record skipped preflights in
`diagnostic_nonclaimable_reasons` and list a `next_actions` item to rerun
without the diagnostic skip flags before the result is treated as claimable.
The same override is required for `--skip-output-verification`; when it is
used, the launcher prints a diagnostic non-claimable result plus
`diagnostic_next_action` lines instead of a claimable completed-run summary.

After a successful `safety`, `target-id`, or `report` run, the launcher writes
`results/runs/<run_id>/run_summary.json` and a human-readable
`results/runs/<run_id>/run_summary.md` with input SMILES/SDF provenance, input
canonical SMILES when a SMILES was provided, standardized canonical SMILES/InChIKey,
skin-sens call/probability evidence, ADMET highlights plus endpoint risk levels, an explicit skin-toxicity
PASS/REVIEW/HALT summary from skin-sens + `Skin_Reaction`, cosmetic/drug
decision plus the `strict`/`moderate`/`lenient` drug policy used to derive it,
an overall PASS/FLAG_HIGH/HALT decision with structured `claimable` status and
reasons, screened target candidate counts from mode-specific DTI/docking
intermediates, top target predictions with gene/protein annotations when
metadata is available, a
skin-specialized material-protein binding summary built from `skin_score`,
`skin_tier`, top-target gene/protein annotation,
`final_score`/`docking_rrf`/source support, top-binding target
skin-expression/skin-efficacy/context support, the most skin-relevant top
target's score/source/skin-efficacy evidence, and KG efficacy labels,
an explicit skin-context decision
(`skin_context_supported`, `skin_expression_only`,
`skin_efficacy_literature_only`, or `insufficient_skin_context`) to prevent
overclaiming when expression evidence is weak. For `target-id` and `report`,
the overall decision remains non-claimable unless the top binding target itself
has direct skin-expression and skin-efficacy context support, even when a
lower-ranked top target supplies the aggregate skin context. The summary also
records relative paths to the source artifacts, including ADMET-AI, structural
alerts, HuSSPred, STopTox, Pred-Skin, and mode-specific target prediction
intermediate evidence files. Summary generation itself fails closed before
writing these files when
compound metadata has an invalid or non-canonical RDKit canonical SMILES,
missing InChIKey, or InChIKey that does not match the canonical SMILES,
ADMET-AI core endpoints / structural-alert flags / skin-sens model calls are
missing or invalid outside an explicit degraded run, ADMET-AI /
structural-alert / HuSSPred / STopTox / Pred-Skin source evidence disagrees
with the consolidated ADMET report, or when the target ranking has
blank/duplicate targets, invalid score ranges, mismatched
`source_count`/`sources`, or non-descending `final_score`. It then calls
`scripts/verify_run_outputs.py` automatically. The verifier checks the completed
run directory against the selected preset: canonical input metadata, ADMET /
skin-sens report plus ADMET-AI/structural-alert/HuSSPred/STopTox/Pred-Skin
source evidence,
CosIng/drug decision-policy-source agreement, mode-specific Stage 3
DTI/docking/rescore intermediates, skin-weighted target ranking with KG
efficacy evidence, `run_summary.json` /
`run_summary.md` consistency including skin-toxicity and skin-specialized
binding fields, and the final HTML report for `report` runs. Stage 9 report
generation also requires the same Stage 2 safety source-evidence contract and
Stage 3 ranking/screening-count contracts before writing HTML:
`skin_sens_decision.txt` /
ADMET-report decision agreement, ADMET-AI core endpoints,
structural-alert flags/matches, HuSSPred/STopTox/Pred-Skin calls and
probabilities, report/source agreement, `final_score`/`skin_score`,
`source_count`/`sources`, KG efficacy columns, minimum source support, and
descending `final_score` plus mode-specific screening artifact row counts. The
report HTML
must include input SMILES/SDF provenance, input canonical SMILES when a SMILES
was provided, standardized canonical SMILES/InChIKey, the overall decision,
recommended action, claimable status, claim status,
cosmetic/drug decision, high/moderate ADMET risk endpoints, full summary ADMET
metrics, drug-avoidance warning count, skin-toxicity decision,
Skin_Reaction value/risk, skin-sens 3-model call/probability evidence,
structural-alert/degraded-evidence markers, skin-context/expression/efficacy
support markers, screened target candidate count, mode-specific screening stage
counts, and top binding target
gene/protein labels when available plus
score/source/skin-score/skin-efficacy/skin-context support.
It also includes the most skin-relevant top target plus its gene/protein labels,
score, source, skin score/tier, and skin-efficacy evidence.
Skipping this post-run verification is guarded
by the same explicit diagnostic override.
After verification, the launcher prints a compact completed-run result with the
summary paths, persistent `run_verification.json` / `run_verification.log`
records containing the structured verifier `checks` and verified artifact
fingerprints, verification check/artifact counts, run ID, preset/mode, input
SMILES/SDF provenance, input canonical SMILES when a SMILES was provided,
standardized canonical SMILES/InChIKey,
`overall_decision`, `recommended_action`, `requires_human_review`,
`claimable` status, decision reasons,
skin-toxicity decision/level, Skin_Reaction value/risk, structural-alert flags,
degraded or missing skin-sens evidence, high and moderate ADMET risk endpoints,
skin-sens model call/probability evidence, full summary ADMET metrics, cosmetic/drug decision, drug policy,
drug-warning count, final ranked target count, screened target candidate count,
mode-specific screening stage counts, top target, ranked top-target list with
gene/protein labels, per-rank score and rank provenance, source count/source
labels, skin score/tier, efficacy evidence, skin-context / skin-expression /
skin-efficacy support, top-target
gene/protein annotation, final score, source count/source labels,
skin score/tier,
skin-efficacy evidence counts, the top-target skin-context support, the most
skin-relevant top target plus its gene/protein labels and score/source/skin-efficacy
evidence, source
artifact counts, and each verified source artifact path.
Runs that used unsafe readiness-skip overrides or explicitly degraded
ADMET/skin-sens evidence are printed as diagnostic and non-claimable even if the
biological decision summary says `proceed`.

`--mode fast` runs Daina over the available human activity evidence, freezes
the first 256 targets, and performs bounded structure annotation on that exact
set. AutoGrid4 maps are generated and hash-cached for each receptor, ligand,
box, spacing, parameter library, and engine version. AutoDock-GPU must export a
real pose and GNINA must score that exact pose for `structure_supported` status.
Missing receptors, pockets, maps, docking, or GNINA results remain explicit
statuses and never change the Daina order.

### 5. Safety runtime readiness

For `safety`, `target-id`, and `report`, the launcher validates Stage 2 ADMET /
skin-sens wiring before Snakemake starts. The default check verifies the
`envs/dti.yml` dependency manifest and that the Stage 2 scripts point to the
current HuSSPred, STopTox, and Pred-Skin public endpoints. Add
`--online-safety-readiness` when you also want live web-service probes.

```bash
python scripts/safety_readiness.py --runtime-mode conda
python scripts/safety_readiness.py --runtime-mode conda --online
```

Degraded safety output is disabled by default. For dry/demo runs only, pass
`--allow-safety-degraded`; readiness marks that run non-claimable and includes
a `next_actions` item to rerun without the degraded-safety override. The
launcher then forwards:
`admet.allow_admet_ai_unavailable=true` and
`admet.allow_skin_sens_unavailable=true`.

### 6. Model runtime readiness

Boltz-2, RTMScore, and PSICHIC run in `envs/boltz2.yml` for Full/Advanced.
GNINA is installed as a pinned CUDA executable for Demo and must be on `PATH`.
AutoDock-GPU itself is expected as `autodock_gpu_128wi` on `PATH`,
`AUTODOCK_GPU_BIN`, or `tools/autodock_gpu/bin/`; conda-forge/bioconda do not
publish that binary. Demo generates AutoGrid4 map/field inputs for each selected
receptor and records every map path/hash/status in an explicit manifest. Meeko
receptor/ligand prep is isolated in `envs/meeko.yml`,
while `envs/autodock_gpu.yml` carries the Python/Vina/OpenBabel support stack
for docking-adjacent rules and degraded Vina diagnostics. `report` runs also verify BioEmu, GROMACS /
gmx_MMPBSA, and QM/CREST environment manifests (`envs/bioemu.yml`,
`envs/md.yml`, `envs/qm.yml`) before Snakemake starts. RTMScore and PSICHIC use
upstream checkouts at `~/.local/opt/RTMScore` and `~/.local/opt/PSICHIC` unless
`RTMSCORE_ROOT` / `PSICHIC_ROOT` override them.

#### Comprehensive 모드가 추가로 요구하는 것 (설치기가 준비하지 않음)

`--profile full`은 conda 환경(`qm`=xTB, `boltz2`, `bioemu`, `md`)만 만든다. 아래 세 가지는
**어떤 설치 경로에도 포함되어 있지 않으므로 직접 준비해야 한다.** 하나라도 없으면
comprehensive DAG는 시작하지 못한다.

먼저 무엇이 빠졌는지부터 확인한다. 아래가 세 항목을 모두 검사한다.

```bash
python scripts/model_readiness.py | python -c "
import json,sys; d=json.load(sys.stdin)
for k in ('diffdock','xtb','obabel'): print(k, d['tools'][k]['status'])
for k in ('rtmscore','psichic'): print(k, d['imports'][k]['status'])"
```

### (1) RTMScore

`rtmscore/__init__.py`가 `RTMSCORE_ROOT`(기본 `~/.local/opt/RTMScore`)에 위임한다.
readiness가 요구하는 파일은 `example/rtmscore.py`와
`trained_models/rtmscore_model1.pth` 두 개다.

```bash
mkdir -p ~/.local/opt
git clone https://github.com/sc8668/RTMScore ~/.local/opt/RTMScore
# 모델 가중치는 업스트림 README의 안내에 따라 trained_models/ 에 배치한다
```

**업스트림 버그를 하나 고쳐야 한다.** `RTMScore/model/model2.py:526`이
`C_batch = th.tensor(range(B))`로 CPU에 텐서를 만든 뒤 CUDA에 있는 마스크로 인덱싱해서
GPU에서만 실패한다. `device=C_mask.device`를 넣으면 된다.

```bash
sed -i 's/C_batch = th.tensor(range(B))/C_batch = th.tensor(range(B), device=C_mask.device)/'   ~/.local/opt/RTMScore/RTMScore/model/model2.py
```

CPU 실행에서는 `C_mask.device`가 cpu이므로 동작이 완전히 같다. CUDA 지원 DGL은
**권장하지 않는다** — 점수는 동일하고 속도는 6.6%만 빨라진다
(`docs/RTMSCORE_CUDA_DGL_20260826.md`).

### (2) PSICHIC

`psichic/__init__.py`가 `PSICHIC_ROOT`(기본 `~/.local/opt/PSICHIC`)에 위임한다.
readiness는 `PSICHIC-prod/inference.py`와
`trained_weights/$PSICHIC_MODEL/model.pt`(기본 `PDBv2020_PSICHIC`)를 요구한다.

```bash
git clone https://github.com/huankoh/PSICHIC ~/.local/opt/PSICHIC
# 학습 가중치는 업스트림 안내에 따라 trained_weights/PDBv2020_PSICHIC/ 에 배치한다
```

### (3) DiffDock

`autodock_pick_top_pct`가 `diffdock_blind_no_pocket`의 출력을 **필수 입력**으로 받으므로,
없으면 comprehensive DAG가 시작하지 못한다.

`scripts/stage3_diffdock_blind.py`는 PATH에서 `diffdock` **명령**을 찾고 다음 형식으로
호출한다. 업스트림은 `inference.py`만 제공하므로 GNINA와 같은 방식으로 래퍼를 만든다.

```
diffdock --protein_path <pdb> --ligand <sdf> --out_dir <dir>          --samples_per_complex 5 --inference_steps 20
```

출력은 `<out_dir>/rank1_confidence*.sdf` 형식이어야 하고, 파일명의 숫자를 신뢰도로 읽는다.

업스트림 `inference.py`는 이 계약과 두 군데가 다르다 — 리간드 인자가
`--ligand_description`이고, 결과를 `<out_dir>/<complex_name>/`에 쓴다. 래퍼가 그 차이를
흡수한다.

```bash
git clone --depth 1 https://github.com/gcorso/DiffDock ~/.local/opt/DiffDock

# 환경 (업스트림은 torch 1.13을 쓰지만 2.4.1+cu121로 동작을 확인했다)
micromamba create -y -n skinscout-diffdock -c conda-forge \
  python=3.11 rdkit prody scipy scikit-learn networkx pandas biopython pyyaml
PY=~/.local/share/mamba/envs/skinscout-diffdock/bin/python
$PY -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
$PY -m pip install torch_geometric e3nn fair-esm==2.0.0
$PY -m pip install torch_cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

# 모델 가중치 134 MB. releases/latest는 404이므로 v1.1을 직접 받는다.
cd ~/.local/opt/DiffDock && $PY -c "
import sys; sys.path.insert(0,'.')
from utils.download import download_and_extract
download_and_extract('https://github.com/gcorso/DiffDock/releases/download/v1.1/diffdock_models.zip','workdir')"

# 압축은 workdir/{score,confidence}_model로 풀리는데 설정은 workdir/v1.1/ 아래를 본다
mkdir -p workdir/v1.1 && mv workdir/score_model workdir/confidence_model workdir/v1.1/

# 어댑터 설치
cd -; install -Dm755 scripts/diffdock_adapter.sh ~/.local/bin/diffdock
```

확인:

```bash
python scripts/model_readiness.py | python -c "
import json,sys; print('diffdock', json.load(sys.stdin)['tools']['diffdock']['status'])"
```

> 이 절차는 2026-08-26에 실제로 실행해 확인했다 — 어댑터가
> `rank1_confidence-1.55.sdf`를 호출자가 찾는 위치에 만들었고,
> `stage3_diffdock_blind.py`가 그 값을 읽어 `diffdock_confidence` 열을 산출했다.
> 업스트림 버전이 바뀌면 인자 이름이 달라질 수 있으므로 실패 시 체크아웃의
> `python inference.py --help`와 대조한다.

```bash
bash scripts/setup_autodock_gpu.sh

micromamba run -n cosmax-boltz2 python scripts/model_readiness.py \
  --require autodock_gpu --require autodock_gpu_env --require meeko_env \
  --require bioemu_env --require md_env --require qm_env \
  --require boltz --require gpu --require gnina --require rtmscore --require psichic
```

### 7. Optional legacy CPU Vina diagnostic

This is not the public Demo pipeline. A CPU Vina driver remains available for
developer diagnostics over a small receptor subset. Its output is degraded
diagnostic evidence only and cannot support target/report claims:

```bash
python scripts/demo_dock_vina.py \
  --ligand-sdf results/runs/aspirin_demo/01_input/compound_canonical.sdf \
  --pdbqt-dir data/human_pdbqt --box-dir data/docking_boxes \
  --include P23219,P35354,P11473,P00918 \
  --n-receptors 300 --exhaustiveness 8 \
  --out-csv results/runs/aspirin_demo/03_targets/demo_ranked_targets.csv
```

Aspirin demo result (recoveries vs. literature):

| Known target | Rank | ΔG (kcal/mol) |
|---|---|---|
| COX-2 (PTGS2, P35354) | **#7 / 300** | −7.04 |
| COX-1 (PTGS1, P23219) | **#16 / 300** | −6.63 |
| VDR (P11473) | #36 | −6.40 |
| CA-II (P00918) | #131 | −5.65 |

## Analog bundle manifest 신뢰 (README에서 이동)

매니페스트를 번들과 같은 곳에서 받으면 그 자체로는 출처를 믿지 않습니다. 원격 번들에는
담당자가 **다른 채널로** 알려 준 매니페스트 SHA256(`--analog-bundle-manifest-sha256`)이 함께 필요합니다.

## Workbench Host 허용 정책 (README에서 이동)

- `--allowed-host <공개 이름>`: reverse proxy가 전달하는 공개 Host만 허용합니다. 목록에 없는
  Host로 오는 GET/HEAD/POST는 모두 거부됩니다(`SKINSCOUT_ALLOWED_HOSTS`로도 지정 가능).
- 비-loopback 평문 HTTP 직접 노출은 거부됩니다. proxy가 별도 컴퓨터라면 backend 구간을
  신뢰된 사설망으로 제한하고 `--host 0.0.0.0 --require-auth --behind-https-proxy --allowed-host <공개 이름>`을 명시하세요.
