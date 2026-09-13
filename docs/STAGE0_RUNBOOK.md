# Stage 0 Runbook

> Versioned infrastructure build. Output: roughly 150–200 GB under `data/`.
> Reuse is allowed only while the current strict verifier and activity-retrieval
> operational gate accept the artifacts; it is not an indefinitely compatible cache.

## 0. Pre-flight

Stage 0 is the large first-time data build. The public Ubuntu installer runs
this build for both profiles and then performs strict verification. This runbook is for
operators who need to inspect, repair, or rebuild it manually. Plan for 150-200 GB of local disk
use and several hours to more than a day depending on network, CPU, and GPU
availability. It is safe to rerun most steps because the download scripts reuse
some existing files, but the current verifiers decide whether they are usable.
Do not start it on a nearly full disk.

The supported completion condition has **two artifacts**:

- `data/manifests/stage0_complete.flag`
- `data/manifests/activity_retrieval_operational_gate.flag`

The first records strict Stage 0 and claim-quality checks. The second binds the
evaluated recipe to the production runtime index and its manifest hashes. A
flag from an older schema, standardization policy, sequence source, recipe, or
index is stale; the launcher and verifier must fail closed instead of silently
accepting it.

> **Repository artifact status checked 2026-09-05:** the existing on-disk
> operational gate predates runtime index/recipe binding, and the current
> canonical FASTA contract has not been published into the production `data/`
> tree. Source-level tests pass, but this checkout's old data artifacts are not
> current-ready. Rebuild Stage 0 and both gates before a target or report run.

```bash
# Check the filesystem that will hold ./data.
df -h .

# CUDA toolkit visible
nvidia-smi

# Optional/manual source files staged locally, then imported with provenance.
# The staging directory may be flat or use canonical subfolders.
# CosIng mirrors from the public EC API by default when cosing.csv is absent.
# Optional enrichment: manual CosIng override, DrugBank XML, skin proteome
# raw_lfq.tsv, GTEx gtex_v10_gene_tpm.gct.
python scripts/import_stage0_sources.py --source-dir /path/to/stage0_sources
python scripts/data_readiness.py --preset stage0
# manifest: data/manifests/stage0_source_manifest.json
# imported sources and generated Stage 0 products remain local-only/gitignored
```

`import_stage0_sources.py` also searches recursively for common download names
such as `CosIng - Glossary of Ingredients.csv`,
`drugbank_all_full_database.xml.zip`, `*skin*proteome*lfq*.tsv.gz`, and
`*gtex*v10*gene*tpm*.gct.gz`. If a directory contains multiple matches for one
source, pass the explicit `--cosing-csv`, `--drugbank-xml`,
`--skin-proteome-tsv`, or `--gtex-gct` argument. The GTEx GCT must already
contain skin-labelled TPM columns; a raw sample-ID-only GTEx matrix needs
preprocessing before import.

To require a manually curated CosIng CSV instead of the public API mirror, set
`cosing.auto_mirror_public_api: false`; `scripts/data_readiness.py --preset
stage0` will then block until `data/cosing/cosing.csv` is imported and passes
format validation.

HPA is downloaded by Stage 0 and is the default SkinScore base. Set
`skin_expression.require_proteome_source` or
`skin_expression.require_gtex_source` to `true` only when those optional
enrichment axes must be present before the build starts. The HPA single-cell
table is pan-tissue, so its axis contributes only for proteins with positive HPA
skin-tissue expression. For those proteins SkinScore records the highest
positive HPA expression label among configured skin-related cell categories as
`cell_type_preferred`; `unknown` means absent, zero, or skin-tissue-unsupported
cell evidence. Treat the label as indirect expression context, not skin
specificity or cell-selective activity.

The public Stage 0 snapshots are pinned in `workflow/config.yaml`:

| Source | Snapshot used by current workflow | Handling |
|---|---|---|
| ChEMBL | 37, release date 2026-05-01 | Downloaded by `mirror_chembl` |
| BindingDB | 2026-08, release date 2026-08-01 | Downloaded by `mirror_bindingdb` |
| GtoPdb | 2026.2, release date 2026-06-15 | Downloaded by `mirror_gtopdb` with pinned SHA-256 files |
| PubChem exact aliases | 2026-08-01 `CID-SMILES.gz` | Downloaded by `mirror_pubchem_alias_source` with pinned MD5 |
| DrugBank | operator-supplied full XML | Optional manual licensed enrichment; never auto-downloaded |

DrugBank is optional by default (`stage0.require_drugbank_source: false`). If
`data/drugbank/drugbank_full_database.xml` is absent, the workflow writes empty
DrugBank parquet slices through the optional-missing path and relies on the
other approved-drug references for claim-quality drug-avoidance checks. Set
`stage0.require_drugbank_source: true` only when licensed DrugBank enrichment is
required for the run.

## 1. Step-by-step

### Step 1 — Download AlphaFold human proteome v4 (~4.8 GB tar; extracted ~50 GB)

```bash
bash scripts/stage0_download_alphafold.sh
# when downloaded, the tar lands at:
# data/alphafold_human_v4/UP000005640_9606_HUMAN_v4.tar
# resumable: wget -c, no re-download on rerun
```

The extracted directory must retain the raw `AF-*-model_v4.cif.gz` files as
well as the PDB structures. The canonical sequence builder reads the mmCIF
metadata. A PDB-only extraction is incomplete for the current workflow and must
not be treated as reusable Stage 0 data.

### Step 2 — pLDDT trim & cleanup (≈ 6-10 h, CPU)

```bash
snakemake -s workflow/Snakefile --use-conda --cores 16 clean_pLDDT_trim \
  --config run_id=stage0_bootstrap compound_smiles=C
# output: data/human_clean/{uniprot}_clean.pdb (~20k unique receptors;
#         20,171 on the 2026-06-21 verified host snapshot)
#         data/human_clean/{uniprot}_softmask.json
```

Trim policy (§3.2 (b)):
- Per-residue pLDDT < 50 → remove
- Run of 5+ residues with pLDDT < 70 at N- or C-terminus → trim
- Soft-mask (low confidence loops): record but keep residues

### Step 3 — P2Rank pocket batch (≈ 24-40 h on 16 cores)

```bash
snakemake -s workflow/Snakefile --cores 16 p2rank_batch \
  --config run_id=stage0_bootstrap compound_smiles=C
# output: data/human_pockets/{uniprot}.pockets.json
#         data/no_pocket_targets.list   # proteins routed to blind docking
# verified host snapshot: 20,171 pocket manifests, 5,133 no-pocket targets
```

### Step 4 — Meeko receptor PDBQT prep (≈ 3-5 h)

```bash
snakemake -s workflow/Snakefile --use-conda --cores 16 meeko_prep_receptors \
  --config run_id=stage0_bootstrap compound_smiles=C
# output: data/human_pdbqt/{uniprot}.pdbqt
#         data/docking_boxes/{uniprot}.box.txt
# verified host snapshot: 15,038 PDBQT receptors
```

Each box is built from the highest-druggability P2Rank pocket centroid ± 12 Å. Targets in `no_pocket_targets.list` skip this step and feed into DiffDock-L blind docking at Stage 3 instead.

### Step 5 - public evidence mirrors and optional DrugBank (about 6-8 h, mostly download)

The `mirror_databases` target mirrors ChEMBL 37, BindingDB 2026-08, GtoPdb
2026.2, and the PubChem 2026-08-01 exact-alias source used by Discovery
exclusion. It also handles optional DrugBank. DrugBank full XML is licensed
enrichment and is not downloaded by this repository; if supplied, the importer
installs it at `data/drugbank/drugbank_full_database.xml` and records its
digest.

```bash
snakemake -s workflow/Snakefile --cores 8 mirror_databases \
  --config run_id=stage0_bootstrap compound_smiles=C
# output: data/chembl37/, data/bindingdb/, data/gtopdb/, data/pubchem/,
#         data/discovery_aliases/, data/drugbank/
# ECFP4 pre-computed for ChEMBL ligands -> data/chembl37/fp_morgan2_2048.parquet
# Discovery exact-match exclusion -> data/discovery_aliases/direct_exact_reference.smi
```

### Step 6 — canonical human sequences and MMseqs2 leakage-audit DB

The workflow does not derive the human sequence database from pLDDT-trimmed PDB
coordinates. It reconstructs each full canonical sequence from the pinned raw
AlphaFold DB v4 mmCIF fragment metadata. For each accession the builder requires:

- human taxonomy ID `9606`;
- exact fragment spans with consistent overlaps and gapless coverage from residue 1;
- agreement with the full UniProt CRC64 embedded in the AFDB mmCIF;
- coverage of every cleaned receptor used by the workflow.

It writes `data/mmseqs/human_canonical.fasta` and
`data/mmseqs/human_canonical.fasta.manifest.json`. The manifest records each
sequence SHA-256, UniProt CRC64, source fragment span, and compressed source
SHA-256. This is an offline reconstruction from AFDB v4 metadata, not a separately
downloaded UniProt FASTA snapshot.

```bash
snakemake -s workflow/Snakefile --use-conda --cores 16 mmseqs_build_db \
  --config run_id=stage0_bootstrap compound_smiles=C
# output: data/mmseqs/human_canonical.fasta
#         data/mmseqs/human_canonical.fasta.manifest.json
#         data/mmseqs/human_db        (canonical human sequence DB)
#         data/mmseqs/training_cutoff_db (PDB ≤ 2021-09-30, Boltz-2 cutoff)
# verified host snapshot: 662,415 protein sequences from 190,466 cutoff entries
```

### Step 7 — Build and verify both completion gates

Use the launcher for the complete Stage 0 contract. It requests both the Stage 0
flag and the activity-retrieval operational gate and enables dependency building.

```bash
python scripts/run_skinscout.py \
  --preset stage0 \
  --run-id stage0_bootstrap \
  --allow-stage0-build \
  --cores 16
```

Internally, `stage0_complete_flag` runs `scripts/stage0_verify.py --strict
--skip-stage0-flag --claim-quality --require-activity-evidence` before writing
`stage0_complete.flag`. The separate operational-gate rule verifies benchmark,
selection, final-evaluation, leakage-audit, runtime-index, and recipe manifests
before writing `activity_retrieval_operational_gate.flag`.

Do not create either flag by hand. Downstream target analysis requires the
operational gate and validates it again. If either gate fails after a code,
source, standardization, or recipe update, rebuild the reported dependencies;
do not copy an older flag into place.

## 2. Disk budget

| Artifact | Size |
|---|---|
| `alphafold_human_v4/` (raw tar + extracted) | ~100 GB peak (delete tar after extract → 50 GB) |
| `human_clean/` | ~50 GB |
| `human_pockets/` | ~3 GB |
| `human_pdbqt/` | ~20 GB |
| `chembl37/` + `bindingdb/` + `gtopdb/` + `pubchem/` + `discovery_aliases/` + `drugbank/` | ~12 GB |
| `mmseqs/` | ~5 GB |
| **Total Stage 0** | **~150-200 GB** |
| External ZINC22 drug-like collection (not a shipped Stage 10 workflow) | potentially hundreds of GB; budget separately |

## 3. Verification checklist (§3.4)

```bash
# Run the bundled verifier
python scripts/stage0_verify.py
python scripts/stage0_verify.py \
  --strict \
  --claim-quality \
  --require-activity-evidence
python scripts/validate_activity_retrieval_gate.py check-operational \
  --gate data/manifests/activity_retrieval_operational_gate.flag
# checks include: unique cleaned receptor count ≥ 20000, no empty cleaned PDBs,
# P2Rank JSON non-NaN, PDBQT header sanity, canonical sequence FASTA/manifest,
# MMseqs2 index sanity, Stage 0 flag, claim-quality CosIng/drug/KG references,
# and runtime recipe/index binding.
```

Historical host snapshot, verified 2026-06-21: strict 11/11 and claim-quality 17/17
passed with 20,171 cleaned receptors, 15,038 PDBQT receptors, 36,832 mirrored
CosIng rows, 672 CosIng structure rows, 811,894 drug rows, 286,034 scaffolds,
and 68 KG gene edges.

Those counts predate the current canonical-sequence and operational-gate
contracts. They document the old host snapshot and do not certify current
readiness. Current readiness requires the strict command and operational-gate
check above to pass on the current checkout and artifacts. The plain verifier
is a quick diagnostic, not the completion criterion.

## 4. Updating the infrastructure

- Keep the source release identifiers in `workflow/config.yaml` pinned for a
  reproducible build.
- Treat any source-release update as an explicit migration and rerun the
  affected builders and verifiers.
- Re-run a Stage 0 rule selectively only after identifying the stale dependency;
  then rebuild both completion gates.
- After changing the sequence source, runtime index, standardization policy,
  selected recipe, or evaluation artifacts, rebuild the operational gate.
- Do not advertise a 60-minute full target/report runtime from this runbook.
  That claim requires separate Ubuntu 24.04 + RTX 4090 qualification evidence.
