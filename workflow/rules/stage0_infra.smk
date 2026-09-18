# =============================================================================
# Stage 0 — One-time infrastructure build
# =============================================================================
# Output: data/{alphafold_human_v4, human_clean, human_pockets, human_pdbqt,
#              docking_boxes, chembl37, bindingdb, gtopdb, drugbank, mmseqs}
#         + data/manifests/stage0_complete.flag
# =============================================================================

AF_RAW        = Path(config["paths"]["alphafold_raw"])
AF_CLEAN      = Path(config["paths"]["alphafold_clean"])
POCKETS       = Path(config["paths"]["pockets"])
PDBQT         = Path(config["paths"]["pdbqt"])
BOXES         = Path(config["paths"]["docking_boxes"])
CHEMBL        = Path(config["paths"]["chembl"])
BINDINGDB     = Path(config["paths"]["bindingdb"])
GTOPDB        = Path(config["paths"]["gtopdb"])
PUBCHEM       = Path(config["paths"]["pubchem"])
DISCOVERY_ALIASES = Path(config["paths"]["discovery_aliases"])
DISCOVERY_ALIAS_SOURCES = DISCOVERY_ALIASES / "sources"
EVIDENCE_SPLITS = Path(config["paths"]["evidence_splits"])
DRUGBANK      = Path(config["paths"]["drugbank"])
MMSEQS        = Path(config["paths"]["mmseqs"])
CANONICAL_HUMAN_FASTA = Path(config["paths"]["uniprot_human_fasta"])
CANONICAL_HUMAN_FASTA_MANIFEST = Path(
    f"{CANONICAL_HUMAN_FASTA}.manifest.json"
)
NO_POCKET     = Path(config["paths"]["no_pocket_list"])
S0            = config["stage0"]
# Kept in sync with scripts/stage0_manifest.MANIFEST_FILENAME: the producer,
# readiness and the verifier must agree on one file name.
MANIFEST_FILENAME = "receptor_prep_manifest.json"

# stage0_verify.py is a subprocess inside the DAG, so it cannot see Snakemake's
# in-memory config. Re-state the resolved paths (and the producer policy) as
# --extra-config overrides for the same effective-config resolver readiness
# uses, instead of letting the verifier fall back to repo/data.
_STAGE0_VERIFY_PATH_KEYS = (
    "alphafold_raw",
    "alphafold_clean",
    "pockets",
    "pdbqt",
    "docking_boxes",
    "no_pocket_list",
    "drugbank",
    "chembl",
    "bindingdb",
    "gtopdb",
    "mmseqs",
    "uniprot_human_fasta",
    "manifests",
    "evidence_splits",
)
_STAGE0_VERIFY_V3_PATH_KEYS = (
    "hpa",
    "skin_proteome",
    "gtex",
    "scrnaseq",
    "skin_expression",
    "cosing",
    "drug_avoidance",
    "skin_kg",
)
_STAGE0_VERIFY_OVERRIDE_PARTS: list[str] = []
for _key in _STAGE0_VERIFY_PATH_KEYS:
    _STAGE0_VERIFY_OVERRIDE_PARTS.extend(
        ["--extra-config", f"paths.{_key}={config['paths'][_key]}"]
    )
for _key in _STAGE0_VERIFY_V3_PATH_KEYS:
    _STAGE0_VERIFY_OVERRIDE_PARTS.extend(
        ["--extra-config", f"paths_v3.{_key}={config['paths_v3'][_key]}"]
    )
STAGE0_VERIFY_OVERRIDES = " ".join(
    shlex.quote(part) for part in _STAGE0_VERIFY_OVERRIDE_PARTS
)
CHEMBL_SQLITE = (
    CHEMBL / f"chembl_{S0['chembl_release']}" /
    f"chembl_{S0['chembl_release']}_sqlite" /
    f"chembl_{S0['chembl_release']}.db"
)
DRUGBANK_REQUIRED = config_bool(S0.get("require_drugbank_source"), False)
DRUGBANK_SOURCE_INPUT = (
    []
    if (
        config_bool(S0.get("allow_drugbank_placeholder"), False)
        or not DRUGBANK_REQUIRED
    )
    else [DRUGBANK / "drugbank_full_database.xml"]
)


rule download_alphafold_human:
    output:
        marker  = AF_RAW / ".download_complete"
    log:
        "results/logs/stage0_download_alphafold.log"
    params:
        url   = S0["alphafold_url"],
        outdir = str(AF_RAW),
        min_existing_models = S0["expected_protein_count"]
    shell:
        r"""
        mkdir -p {params.outdir:q}
        MIN_EXISTING_MODELS={params.min_existing_models:q} \
        bash scripts/stage0_download_alphafold.sh \
            {params.url:q} {params.outdir:q} > {log:q} 2>&1
        touch {output.marker:q}
        """


rule extract_alphafold:
    input:
        ancient(rules.download_alphafold_human.output.marker)
    output:
        marker = AF_RAW / ".extracted"
    params:
        outdir = str(AF_RAW),
        tar = str(AF_RAW / Path(S0["alphafold_url"]).name),
        min_existing_models = S0["expected_protein_count"]
    threads: 4
    shell:
        r"""
        # A validated extraction may predate this run.  Keep its marker and
        # downstream timestamps intact so an infrastructure bootstrap does
        # not redownload or reprocess an already materialized proteome.
        if [ -f {output.marker:q} ]; then
            echo "[stage0] existing AlphaFold extraction detected; reusing it"
            exit 0
        fi
        tar -xf {params.tar:q} -C {params.outdir:q} --strip-components=0
        # AF v4 archive yields AF-*-F1-model_v4.pdb.gz; decompress in place.
        find {params.outdir:q} -name "*.pdb.gz" -print0 \
            | xargs -0 -n 64 -P {threads:q} gunzip -f
        PDB_MODELS=$(find {params.outdir:q} -maxdepth 1 -type f \
            -name 'AF-*-F1-model_v4.pdb' -printf '.' | wc -c)
        CIF_MODELS=$(find {params.outdir:q} -maxdepth 1 -type f \
            -name 'AF-*-F1-model_v4.cif.gz' -printf '.' | wc -c)
        if [ "$PDB_MODELS" -lt {params.min_existing_models:q} ] \
           || [ "$CIF_MODELS" -lt {params.min_existing_models:q} ]; then
            echo "[stage0][FATAL] extracted AFDB v4 archive is incomplete: "\
                 "$PDB_MODELS PDB and $CIF_MODELS canonical mmCIF files; "\
                 "need at least {params.min_existing_models:q} of each" >&2
            exit 2
        fi
        touch {output.marker:q}
        """


rule clean_pLDDT_trim:
    """pLDDT trim + soft-mask cleanup → data/human_clean/{uniprot}_clean.pdb."""
    input:
        rules.extract_alphafold.output.marker
    output:
        marker = AF_CLEAN / ".clean_complete"
    log:
        "results/logs/stage0_clean.log"
    # 12 (not all 16) so the single-threaded ChEMBL slice + mmseqs build can run
    # concurrently instead of serialising behind this 6-10 h step.
    threads: 12
    conda:
        "../../envs/base.yml"
    params:
        af_dir   = str(AF_RAW),
        out_dir  = str(AF_CLEAN),
        rm_cut   = S0["plddt_remove_cutoff"],
        soft_cut = S0["plddt_softmask_cutoff"],
        run_len  = S0["trim_run_length"],
        min_trim = S0["trim_min_length"],
        min_success_count = S0["expected_protein_count"]
    shell:
        r"""
        mkdir -p {params.out_dir:q}
        python scripts/stage0_clean_alphafold.py \
            --af-dir {params.af_dir:q} \
            --out-dir {params.out_dir:q} \
            --remove-cutoff {params.rm_cut:q} \
            --softmask-cutoff {params.soft_cut:q} \
            --run-length {params.run_len:q} \
            --min-trim {params.min_trim:q} \
            --workers {threads:q} \
            --min-success-count {params.min_success_count:q} > {log:q} 2>&1
        touch {output.marker:q}
        """


rule p2rank_batch:
    """Batch pocket detection on every cleaned PDB."""
    input:
        rules.clean_pLDDT_trim.output.marker
    output:
        pocket_marker = POCKETS / ".p2rank_complete",
        no_pocket_list = str(NO_POCKET)
    log:
        "results/logs/stage0_p2rank.log"
    # 14 (not 16) so P2Rank — the 24-40 h critical path — can start immediately
    # alongside the single-threaded ChEMBL slice instead of waiting for it.
    threads: 14
    conda:
        "../../envs/base.yml"
    params:
        clean_dir   = str(AF_CLEAN),
        pocket_dir  = str(POCKETS),
        min_processed_count = S0["expected_protein_count"],
        min_usable_fraction = S0["p2rank_min_usable_fraction"]
    shell:
        r"""
        mkdir -p {params.pocket_dir:q}
        bash scripts/stage0_p2rank_batch.sh \
            {params.clean_dir:q} {params.pocket_dir:q} {threads:q} > {log:q} 2>&1
        # post-process: any protein with empty/low pockets → no_pocket_list
        python scripts/stage0_p2rank_postprocess.py \
            --pocket-dir {params.pocket_dir:q} \
            --no-pocket-out {output.no_pocket_list:q} \
            --expected-input-list {params.pocket_dir:q}/proteins.ds \
            --min-processed-count {params.min_processed_count:q} \
            --min-usable-fraction {params.min_usable_fraction:q} >> {log:q} 2>&1
        touch {output.pocket_marker:q}
        """


rule meeko_prep_receptors:
    """Convert cleaned PDB → AutoDock-GPU PDBQT for every with-pocket receptor."""
    input:
        rules.p2rank_batch.output.pocket_marker
    output:
        marker = PDBQT / ".pdbqt_complete",
        manifest = PDBQT / MANIFEST_FILENAME
    log:
        "results/logs/stage0_meeko.log"
    threads: 8
    conda:
        "../../envs/meeko.yml"
    params:
        clean_dir  = str(AF_CLEAN),
        pocket_dir = str(POCKETS),
        out_dir    = str(PDBQT),
        box_dir    = str(BOXES),
        no_pocket  = str(NO_POCKET),
        min_success_fraction = config["stage0"]["pdbqt_min_success_fraction"],
        min_success_count = config["stage0"]["pdbqt_min_success_count"]
    shell:
        r"""
        mkdir -p {params.out_dir:q} {params.box_dir:q}
        python scripts/stage0_meeko_prep.py \
            --clean-dir {params.clean_dir:q} \
            --pocket-dir {params.pocket_dir:q} \
            --no-pocket-list {params.no_pocket:q} \
            --out-pdbqt-dir {params.out_dir:q} \
            --out-box-dir {params.box_dir:q} \
            --workers {threads:q} \
            --manifest-path {output.manifest:q} \
            --min-success-fraction {params.min_success_fraction:q} \
            --min-success-count {params.min_success_count:q} > {log:q} 2>&1
        touch {output.marker:q}
        """


rule mirror_chembl:
    output:
        flag = CHEMBL / ".mirror_complete",
        sqlite_db = CHEMBL_SQLITE,
        activities = CHEMBL / "human_activities.parquet",
        evidence = CHEMBL / "activity_evidence.parquet",
        manifest = CHEMBL / "source_manifest.json",
        fingerprints = CHEMBL / "fp_morgan2_2048.parquet",
        fingerprint_manifest = CHEMBL / "fingerprint_manifest.json"
    log:
        "results/logs/stage0_chembl.log"
    conda:
        "../../envs/base.yml"
    params:
        outdir = str(CHEMBL),
        release = S0["chembl_release"],
        release_date = S0["chembl_release_date"]
    shell:
        r"""
        mkdir -p {params.outdir:q}
        bash scripts/stage0_mirror_chembl.sh \
            {params.outdir:q} {params.release:q} {params.release_date:q} \
            > {log:q} 2>&1
        python scripts/stage0_chembl_fingerprints.py \
            --chembl-dir {params.outdir:q} >> {log:q} 2>&1
        touch {output.flag:q}
        """


rule mirror_bindingdb:
    output:
        flag = BINDINGDB / ".mirror_complete",
        raw = BINDINGDB / "BindingDB_All.tsv",
        source_manifest = BINDINGDB / "bindingdb_source_manifest.json",
        pdb = BINDINGDB / "bindingdb_with_pdb.parquet"
    log:
        "results/logs/stage0_bindingdb.log"
    conda:
        "../../envs/base.yml"
    params:
        outdir = str(BINDINGDB),
        release = S0["bindingdb_release"],
        allow_placeholder = (
            "--allow-placeholder"
            if config_bool(S0.get("allow_bindingdb_placeholder"), False)
            else ""
        )
    shell:
        r"""
        mkdir -p {params.outdir:q}
        bash scripts/stage0_mirror_bindingdb.sh {params.outdir:q} \
            --release {params.release:q} \
            {params.allow_placeholder} > {log:q} 2>&1
        touch {output.flag:q}
        """


rule mirror_gtopdb:
    output:
        flag = GTOPDB / ".mirror_complete",
        interactions = GTOPDB / "interactions.csv",
        ligands = GTOPDB / "ligands.csv",
        target_mapping = GTOPDB / "GtP_to_UniProt_mapping.csv",
        file_descriptions = GTOPDB / "file_descriptions.txt",
        pubmed = GTOPDB / "pubmed_esummary.json",
        manifest = GTOPDB / "source_manifest.json"
    log:
        "results/logs/stage0_gtopdb.log"
    conda:
        "../../envs/base.yml"
    params:
        outdir = str(GTOPDB),
        release = S0["gtopdb_release"],
        release_date = S0["gtopdb_release_date"],
        interactions_sha256 = S0["gtopdb_interactions_sha256"],
        ligands_sha256 = S0["gtopdb_ligands_sha256"],
        target_mapping_sha256 = S0["gtopdb_target_mapping_sha256"],
        file_descriptions_sha256 = S0["gtopdb_file_descriptions_sha256"]
    shell:
        r"""
        mkdir -p {params.outdir:q}
        python scripts/mirror_gtopdb.py \
            --out-dir {params.outdir:q} \
            --release {params.release:q} \
            --release-date {params.release_date:q} \
            --interactions-sha256 {params.interactions_sha256:q} \
            --ligands-sha256 {params.ligands_sha256:q} \
            --target-mapping-sha256 {params.target_mapping_sha256:q} \
            --file-descriptions-sha256 {params.file_descriptions_sha256:q} \
            > {log:q} 2>&1
        touch {output.flag:q}
        """


rule mirror_pubchem_alias_source:
    """Mirror the pinned PubChem CID-SMILES snapshot used only for exact aliases."""
    output:
        raw = PUBCHEM / "CID-SMILES.gz",
        manifest = PUBCHEM / "source_manifest.json"
    log:
        "results/logs/stage0_pubchem_alias_source.log"
    conda:
        "../../envs/base.yml"
    params:
        outdir = str(PUBCHEM),
        release_date = S0["pubchem_alias_release_date"],
        expected_md5 = S0["pubchem_alias_smiles_md5"],
        url = S0["pubchem_alias_smiles_url"]
    shell:
        r"""
        python scripts/mirror_pubchem_alias_source.py \
            --out-dir {params.outdir:q} \
            --release-date {params.release_date:q} \
            --expected-md5 {params.expected_md5:q} \
            --url {params.url:q} > {log:q} 2>&1
        """


rule build_discovery_alias_sources:
    """Normalize the four licensed exact-alias sources into sealed parquets."""
    input:
        chembl_db = rules.mirror_chembl.output.sqlite_db,
        chembl_manifest = rules.mirror_chembl.output.manifest,
        bindingdb_tsv = rules.mirror_bindingdb.output.raw,
        bindingdb_manifest = rules.mirror_bindingdb.output.source_manifest,
        gtopdb_ligands = rules.mirror_gtopdb.output.ligands,
        gtopdb_manifest = rules.mirror_gtopdb.output.manifest,
        pubchem_smiles = rules.mirror_pubchem_alias_source.output.raw,
        pubchem_manifest = rules.mirror_pubchem_alias_source.output.manifest
    output:
        chembl = DISCOVERY_ALIAS_SOURCES / "chembl_aliases.parquet",
        bindingdb = DISCOVERY_ALIAS_SOURCES / "bindingdb_aliases.parquet",
        gtopdb = DISCOVERY_ALIAS_SOURCES / "gtopdb_aliases.parquet",
        pubchem = DISCOVERY_ALIAS_SOURCES / "pubchem_aliases.parquet",
        registry = DISCOVERY_ALIAS_SOURCES / "source_registry.json",
        manifest = DISCOVERY_ALIAS_SOURCES / "source_manifest.json"
    log:
        "results/logs/stage0_discovery_alias_sources.log"
    conda:
        "../../envs/base.yml"
    params:
        outdir = str(DISCOVERY_ALIAS_SOURCES),
        chembl_release_date = S0["chembl_release_date"],
        bindingdb_release_date = S0["bindingdb_release_date"]
    shell:
        r"""
        python scripts/build_discovery_alias_sources.py \
            --chembl-db {input.chembl_db:q} \
            --chembl-manifest {input.chembl_manifest:q} \
            --chembl-release-date {params.chembl_release_date:q} \
            --bindingdb-tsv {input.bindingdb_tsv:q} \
            --bindingdb-manifest {input.bindingdb_manifest:q} \
            --bindingdb-release-date {params.bindingdb_release_date:q} \
            --gtopdb-ligands {input.gtopdb_ligands:q} \
            --gtopdb-manifest {input.gtopdb_manifest:q} \
            --pubchem-smiles-gz {input.pubchem_smiles:q} \
            --pubchem-manifest {input.pubchem_manifest:q} \
            --out-dir {params.outdir:q} > {log:q} 2>&1
        """


rule build_discovery_alias_map:
    """Build the canonical exact-alias map and direct-evidence exclusion set."""
    threads: 16
    input:
        registry = rules.build_discovery_alias_sources.output.registry,
        source_manifest = rules.build_discovery_alias_sources.output.manifest
    output:
        aliases = DISCOVERY_ALIASES / "aliases.parquet",
        direct = DISCOVERY_ALIASES / "direct_exact_reference.smi",
        canonicalization_audit = DISCOVERY_ALIASES / "canonicalization_exclusions.jsonl",
        manifest = DISCOVERY_ALIASES / "manifest.json"
    log:
        "results/logs/stage0_discovery_alias_map.log"
    conda:
        "../../envs/base.yml"
    shell:
        r"""
        python scripts/build_discovery_alias_map.py \
            --source-registry {input.registry:q} \
            --source-manifest {input.source_manifest:q} \
            --out-aliases-parquet {output.aliases:q} \
            --out-direct-reference {output.direct:q} \
            --out-manifest {output.manifest:q} \
            --canonicalization-workers {threads:q} > {log:q} 2>&1
        """


rule build_bindingdb_temporal_evidence:
    input:
        mirror = rules.mirror_bindingdb.output.flag,
        raw = rules.mirror_bindingdb.output.raw,
        source_manifest = rules.mirror_bindingdb.output.source_manifest
    output:
        evidence = BINDINGDB / "evidence_v1" / "activity_evidence.parquet",
        pre = BINDINGDB / "evidence_v1" / "pre_cutoff.parquet",
        post = BINDINGDB / "evidence_v1" / "post_cutoff.parquet",
        manifest = BINDINGDB / "evidence_v1" / "manifest.json"
    log:
        "results/logs/stage0_bindingdb_evidence.log"
    conda:
        "../../envs/base.yml"
    params:
        outdir = str(BINDINGDB / "evidence_v1"),
        cutoff = config["evaluation"]["common_cutoff_date"],
        release = S0["bindingdb_release"]
    shell:
        r"""
        python scripts/build_bindingdb_temporal_evidence.py \
            --bindingdb-tsv {input.raw:q} \
            --source-manifest {input.source_manifest:q} \
            --out-dir {params.outdir:q} \
            --cutoff-date {params.cutoff:q} \
            --source-release {params.release:q} > {log:q} 2>&1
        """


rule build_gtopdb_activity_evidence:
    input:
        interactions = rules.mirror_gtopdb.output.interactions,
        ligands = rules.mirror_gtopdb.output.ligands,
        target_mapping = rules.mirror_gtopdb.output.target_mapping,
        file_descriptions = rules.mirror_gtopdb.output.file_descriptions,
        pubmed = rules.mirror_gtopdb.output.pubmed,
        source_manifest = rules.mirror_gtopdb.output.manifest
    output:
        evidence = GTOPDB / "evidence_v1" / "activity_evidence.parquet",
        pre = GTOPDB / "evidence_v1" / "pre_cutoff.parquet",
        post = GTOPDB / "evidence_v1" / "post_cutoff.parquet",
        manifest = GTOPDB / "evidence_v1" / "manifest.json"
    log:
        "results/logs/stage0_gtopdb_evidence.log"
    conda:
        "../../envs/base.yml"
    params:
        outdir = str(GTOPDB / "evidence_v1"),
        cutoff = config["evaluation"]["common_cutoff_date"],
        release = S0["gtopdb_release"]
    shell:
        r"""
        python scripts/build_gtopdb_activity_evidence.py \
            --interactions-csv {input.interactions:q} \
            --ligands-csv {input.ligands:q} \
            --target-mapping-csv {input.target_mapping:q} \
            --pubmed-json {input.pubmed:q} \
            --source-manifest {input.source_manifest:q} \
            --required-release {params.release:q} \
            --cutoff-date {params.cutoff:q} \
            --out-dir {params.outdir:q} > {log:q} 2>&1
        """


rule build_activity_evidence_splits:
    input:
        chembl = rules.mirror_chembl.output.evidence,
        bindingdb = rules.build_bindingdb_temporal_evidence.output.evidence,
        gtopdb = rules.build_gtopdb_activity_evidence.output.evidence
    output:
        pre = EVIDENCE_SPLITS / "pre_cutoff.parquet",
        post = EVIDENCE_SPLITS / "post_cutoff.parquet",
        undated = EVIDENCE_SPLITS / "undated.parquet",
        novel = EVIDENCE_SPLITS / "post_cutoff_novel_pairs.parquet",
        manifest = EVIDENCE_SPLITS / "split_manifest.json"
    log:
        "results/logs/stage0_activity_evidence_splits.log"
    conda:
        "../../envs/base.yml"
    params:
        outdir = str(EVIDENCE_SPLITS),
        cutoff = config["evaluation"]["common_cutoff_date"],
        chembl_spec = str(CHEMBL / "activity_evidence.parquet") + "=chembl",
        bindingdb_spec = (
            str(BINDINGDB / "evidence_v1" / "activity_evidence.parquet")
            + "=bindingdb"
        ),
        gtopdb_spec = str(GTOPDB / "evidence_v1" / "activity_evidence.parquet") + "=gtopdb"
    shell:
        r"""
        python scripts/build_evidence_splits.py \
            --evidence {params.chembl_spec:q} \
            --evidence {params.bindingdb_spec:q} \
            --evidence {params.gtopdb_spec:q} \
            --cutoff-date {params.cutoff:q} \
            --out-dir {params.outdir:q} > {log:q} 2>&1
        """


rule mirror_drugbank:
    input:
        DRUGBANK_SOURCE_INPUT
    output:
        flag = DRUGBANK / ".mirror_complete",
        polypharm = DRUGBANK / "drugbank_polypharm.parquet",
        approved = DRUGBANK / "drugbank_approved.parquet"
    log:
        "results/logs/stage0_drugbank.log"
    conda:
        "../../envs/base.yml"
    params:
        outdir = str(DRUGBANK),
        allow_placeholder = (
            "--allow-placeholder"
            if config_bool(S0.get("allow_drugbank_placeholder"), False)
            else ""
        ),
        allow_missing_optional = (
            "--allow-missing-optional"
            if not DRUGBANK_REQUIRED
            else ""
        )
    shell:
        r"""
        mkdir -p {params.outdir:q}
        bash scripts/stage0_mirror_drugbank.sh {params.outdir:q} \
            {params.allow_placeholder} {params.allow_missing_optional} > {log:q} 2>&1
        touch {output.flag:q}
        """


rule mirror_databases:
    """Aggregate convenience target."""
    input:
        rules.mirror_chembl.output.flag,
        rules.mirror_bindingdb.output.flag,
        rules.mirror_gtopdb.output.flag,
        rules.build_gtopdb_activity_evidence.output.manifest,
        rules.build_discovery_alias_map.output.manifest,
        rules.mirror_drugbank.output.flag,
        evidence_splits = (
            rules.build_activity_evidence_splits.output.manifest
            if not config_bool(S0.get("allow_bindingdb_placeholder"), False)
            else []
        )


rule build_canonical_human_sequences:
    """Assemble full canonical sequences from pinned AFDB v4 mmCIF fragments."""
    input:
        extracted = rules.extract_alphafold.output.marker,
        cleaned = rules.clean_pLDDT_trim.output.marker
    output:
        fasta = str(CANONICAL_HUMAN_FASTA),
        manifest = str(CANONICAL_HUMAN_FASTA_MANIFEST)
    log:
        "results/logs/stage0_canonical_sequences.log"
    conda:
        "../../envs/base.yml"
    params:
        alphafold_dir = str(AF_RAW),
        clean_dir = str(AF_CLEAN),
        min_sequences = S0["expected_protein_count"]
    shell:
        r"""
        python scripts/build_canonical_sequences.py \
            --alphafold-dir {params.alphafold_dir:q} \
            --out-fasta {output.fasta:q} \
            --out-manifest {output.manifest:q} \
            --min-sequences {params.min_sequences:q} \
            --required-receptor-dir {params.clean_dir:q} > {log:q} 2>&1
        """


rule mmseqs_build_db:
    input:
        canonical_fasta = rules.build_canonical_human_sequences.output.fasta,
        canonical_manifest = rules.build_canonical_human_sequences.output.manifest
    output:
        flag = MMSEQS / ".build_complete"
    log:
        "results/logs/stage0_mmseqs.log"
    threads:
        config_int(S0.get("mmseqs_threads"), "stage0.mmseqs_threads", 4, minimum=1)
    conda:
        "../../envs/base.yml"
    params:
        canonical_fasta = str(CANONICAL_HUMAN_FASTA),
        out_dir   = str(MMSEQS),
        allow_missing_training_cutoff = (
            "--allow-missing-training-cutoff"
            if config_bool(S0.get("allow_missing_training_cutoff"), False)
            else ""
        )
    shell:
        r"""
        mkdir -p {params.out_dir:q}
        MMSEQS_THREADS={threads:q} bash scripts/stage0_mmseqs_build.sh \
            {input.canonical_fasta:q} {params.out_dir:q} \
            {params.allow_missing_training_cutoff} > {log:q} 2>&1
        touch {output.flag:q}
        """


rule stage0_complete_flag:
    """v2 + v3 combined infrastructure flag.

    Pulls in BOTH the v2 mirror set (AlphaFold + P2Rank + PDBQT + ChEMBL +
    BindingDB + optional DrugBank + MMseqs2) AND the v3 cosmetic-domain DBs
    (skin_score, CosIng, drug avoidance, skin-efficacy KG). The separate
    activity-retrieval operational gate remains required before Stage 3 target
    scoring can run.
    """
    input:
        # --- v2 ---
        rules.meeko_prep_receptors.output.marker,
        rules.meeko_prep_receptors.output.manifest,
        rules.mirror_chembl.output.flag,
        rules.mirror_bindingdb.output.flag,
        rules.mirror_gtopdb.output.flag,
        rules.build_gtopdb_activity_evidence.output.manifest,
        rules.build_discovery_alias_map.output.manifest,
        rules.build_discovery_alias_map.output.direct,
        rules.mirror_drugbank.output.flag,
        rules.mmseqs_build_db.output.flag,
        rules.build_canonical_human_sequences.output.manifest,
        # --- v3 (paths only — rule objects not yet defined at include time) ---
        str(Path(config["paths_v3"]["skin_expression"]) / "skin_score.tsv"),
        str(
            Path(config["paths_v3"]["skin_expression"])
            / "skin_score.tsv.axes.json"
        ),
        str(Path(config["paths_v3"]["cosing"]) / "cosing.parquet"),
        str(Path(config["paths_v3"]["drug_avoidance"]) / "drugs.parquet"),
        str(Path(config["paths_v3"]["skin_kg"]) / "skin_efficacy.graphml"),
        activity_evidence_splits = (
            rules.build_activity_evidence_splits.output.manifest
            if not config_bool(S0.get("allow_bindingdb_placeholder"), False)
            else []
        )
    output:
        str(MANIFESTS / "stage0_complete.flag")
    params:
        bindingdb_placeholder_arg = (
            "--allow-bindingdb-placeholder"
            if config_bool(S0.get("allow_bindingdb_placeholder"), False)
            else ""
        ),
        verify_paths = STAGE0_VERIFY_OVERRIDES,
        pdbqt_min_success_fraction = S0["pdbqt_min_success_fraction"],
        pdbqt_min_success_count = S0["pdbqt_min_success_count"]
    shell:
        "python scripts/stage0_verify.py --strict --skip-stage0-flag "
        "--claim-quality "
        "--require-activity-evidence "
        "--pdbqt-min-success-fraction {params.pdbqt_min_success_fraction:q} "
        "--pdbqt-min-success-count {params.pdbqt_min_success_count:q} "
        "{params.verify_paths} "
        "{params.bindingdb_placeholder_arg} && touch {output:q}"
