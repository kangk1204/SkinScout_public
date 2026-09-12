# =============================================================================
# Activity retrieval recovery gates
# =============================================================================

AR_CFG = config["evaluation"]["activity_retrieval"]
AR_BENCHMARK_DIR = Path(AR_CFG["benchmark_dir"])
AR_INDEX_DIR = Path(AR_CFG["retrieval_index_dir"])
AR_PANELS_DIR = Path(AR_CFG["panels_dir"])
AR_RCSB_RAW_JSONL_GZ = Path(AR_CFG["rcsb_raw_jsonl_gz"])
AR_RCSB_SNAPSHOT_MANIFEST = Path(AR_CFG["rcsb_snapshot_manifest"])
AR_RCSB_PANEL_PAIRS = Path(AR_CFG["rcsb_panel_pairs"])
AR_RCSB_PANEL_EXCLUSIONS = Path(AR_CFG["rcsb_panel_exclusions"])
AR_RCSB_RANKING_QUERIES = Path(AR_CFG["rcsb_ranking_queries"])
AR_RCSB_PANEL_MANIFEST = Path(AR_CFG["rcsb_panel_manifest"])
AR_RCSB_CONTACT_COORDINATE_DIR = Path(AR_CFG["rcsb_contact_coordinate_dir"])
AR_RCSB_CONTACT_FRAGMENT_DIR = Path(AR_CFG["rcsb_contact_fragment_dir"])
AR_RCSB_CONTACT_FRAGMENT_INDEX = Path(AR_CFG["rcsb_contact_fragment_index"])
AR_RCSB_CONTACT_FRAGMENT_EXCLUSIONS = Path(
    AR_CFG["rcsb_contact_fragment_exclusions"]
)
AR_RCSB_CONTACT_FRAGMENT_MANIFEST = Path(AR_CFG["rcsb_contact_fragment_manifest"])
AR_PRIOR_POCKET_TARGETS = Path(AR_CFG["prior_pocket_targets"])
AR_PRIOR_POCKET_TARGET_MANIFEST = Path(AR_CFG["prior_pocket_target_manifest"])
AR_PRIOR_POCKET_FRAGMENT_DIR = Path(AR_CFG["prior_pocket_fragment_dir"])
AR_PRIOR_POCKET_FRAGMENT_INDEX = Path(AR_CFG["prior_pocket_fragment_index"])
AR_PRIOR_POCKET_FRAGMENT_EXCLUSIONS = Path(
    AR_CFG["prior_pocket_fragment_exclusions"]
)
AR_PRIOR_POCKET_FRAGMENT_MANIFEST = Path(AR_CFG["prior_pocket_fragment_manifest"])
AR_RCSB_POCKET_LEAKAGE_ROWS = Path(AR_CFG["rcsb_contact_pocket_leakage_rows"])
AR_RCSB_POCKET_LEAKAGE_MANIFEST = Path(
    AR_CFG["rcsb_contact_pocket_leakage_manifest"]
)
AR_DEV_SELECTION_DIR = Path(AR_CFG["dev_selection_dir"])
AR_FINAL_EVAL_DIR = Path(AR_CFG["final_evaluation_dir"])
AR_BENCHMARK_TARGET_CLUSTERS = Path(AR_CFG["benchmark_target_clusters"])
AR_BENCHMARK_TARGET_CLUSTER_MANIFEST = Path(
    AR_CFG["benchmark_target_cluster_manifest"]
)
AR_SOURCE_TARGET_EXCLUSIONS = Path(AR_CFG["source_target_exclusions"])
AR_SCREENABLE_TARGET_CLUSTERS = Path(AR_CFG["screenable_target_clusters"])
AR_SCREENABLE_TARGET_CLUSTER_MANIFEST = Path(
    AR_CFG["screenable_target_cluster_manifest"]
)
AR_KNOWN_PANEL = Path(AR_CFG["known_panel"])
AR_ALLOW_NO_IMPROVEMENT_ARG = (
    "--allow-no-improvement"
    if config_bool(AR_CFG.get("allow_no_improvement"), False)
    else ""
)


rule activity_benchmark_partitions:
    input:
        chembl = rules.mirror_chembl.output.evidence,
        chembl_manifest = rules.mirror_chembl.output.manifest,
        bindingdb = rules.build_bindingdb_temporal_evidence.output.evidence,
        bindingdb_manifest = rules.build_bindingdb_temporal_evidence.output.manifest,
        gtopdb = rules.build_gtopdb_activity_evidence.output.evidence,
        gtopdb_manifest = rules.build_gtopdb_activity_evidence.output.manifest,
        target_clusters = AR_BENCHMARK_TARGET_CLUSTERS,
        target_cluster_manifest = AR_BENCHMARK_TARGET_CLUSTER_MANIFEST,
        source_target_exclusions = AR_SOURCE_TARGET_EXCLUSIONS
    output:
        train = AR_BENCHMARK_DIR / "train.parquet",
        dev = AR_BENCHMARK_DIR / "dev.parquet",
        test = AR_BENCHMARK_DIR / "test.parquet",
        temporal_test = AR_BENCHMARK_DIR / "temporal_test.parquet",
        ligand_scaffold_cold = AR_BENCHMARK_DIR / "ligand_scaffold_cold.parquet",
        target_cold30 = AR_BENCHMARK_DIR / "target_cold30.parquet",
        target_cold50 = AR_BENCHMARK_DIR / "target_cold50.parquet",
        dual_cold = AR_BENCHMARK_DIR / "dual_cold.parquet",
        manifest = AR_BENCHMARK_DIR / "manifest.json"
    log:
        "results/logs/activity_retrieval_benchmark.log"
    conda:
        "../../envs/base.yml"
    params:
        outdir = str(AR_BENCHMARK_DIR),
        bindingdb_release = S0["bindingdb_release"],
        gtopdb_release = S0["gtopdb_release"]
    shell:
        r"""
        python eval/build_activity_benchmark.py \
            --chembl-evidence {input.chembl:q} \
            --chembl-manifest {input.chembl_manifest:q} \
            --bindingdb-evidence {input.bindingdb:q} \
            --bindingdb-manifest {input.bindingdb_manifest:q} \
            --bindingdb-required-release {params.bindingdb_release:q} \
            --gtopdb-evidence {input.gtopdb:q} \
            --gtopdb-manifest {input.gtopdb_manifest:q} \
            --gtopdb-required-release {params.gtopdb_release:q} \
            --target-clusters {input.target_clusters:q} \
            --target-cluster-manifest {input.target_cluster_manifest:q} \
            --source-target-exclusions {input.source_target_exclusions:q} \
            --out-dir {params.outdir:q} > {log:q} 2>&1
        """


rule activity_retrieval_index:
    input:
        train = rules.activity_benchmark_partitions.output.train,
        benchmark_manifest = rules.activity_benchmark_partitions.output.manifest
    output:
        ligands = AR_INDEX_DIR / "ligands.parquet",
        edges = AR_INDEX_DIR / "edges.parquet",
        manifest = AR_INDEX_DIR / "manifest.json"
    log:
        "results/logs/activity_retrieval_index.log"
    threads:
        4
    conda:
        "../../envs/base.yml"
    params:
        negative_threshold = AR_CFG["negative_threshold"],
        positive_threshold = AR_CFG["positive_threshold"]
    shell:
        r"""
        python scripts/build_activity_retrieval_index.py \
            --train-parquet {input.train:q} \
            --benchmark-manifest {input.benchmark_manifest:q} \
            --out-ligands {output.ligands:q} \
            --out-edges {output.edges:q} \
            --out-manifest {output.manifest:q} \
            --negative-threshold {params.negative_threshold:q} \
            --positive-threshold {params.positive_threshold:q} \
            --workers {threads} > {log:q} 2>&1
        """


rule rcsb_holo_snapshot:
    output:
        raw_jsonl_gz = AR_RCSB_RAW_JSONL_GZ,
        manifest = AR_RCSB_SNAPSHOT_MANIFEST
    log:
        "results/logs/rcsb_holo_snapshot.log"
    conda:
        "../../envs/base.yml"
    params:
        release_cutoff = AR_CFG["rcsb_release_cutoff"],
        min_nonpolymer_mw = AR_CFG["rcsb_min_nonpolymer_mw"]
    shell:
        r"""
        python scripts/mirror_rcsb_holo_snapshot.py \
            --release-cutoff {params.release_cutoff:q} \
            --min-nonpolymer-mw {params.min_nonpolymer_mw:q} \
            --raw-jsonl-gz {output.raw_jsonl_gz:q} \
            --manifest {output.manifest:q} > {log:q} 2>&1
        """


rule rcsb_holo_direct_contact_panel:
    input:
        raw_jsonl_gz = rules.rcsb_holo_snapshot.output.raw_jsonl_gz,
        source_manifest = rules.rcsb_holo_snapshot.output.manifest,
        train = rules.activity_benchmark_partitions.output.train,
        dev = rules.activity_benchmark_partitions.output.dev,
        benchmark_manifest = rules.activity_benchmark_partitions.output.manifest,
        screenable_target_clusters = AR_SCREENABLE_TARGET_CLUSTERS,
        screenable_target_cluster_manifest = AR_SCREENABLE_TARGET_CLUSTER_MANIFEST
    output:
        pairs = AR_RCSB_PANEL_PAIRS,
        exclusions = AR_RCSB_PANEL_EXCLUSIONS,
        ranking_queries = AR_RCSB_RANKING_QUERIES,
        manifest = AR_RCSB_PANEL_MANIFEST
    log:
        "results/logs/rcsb_holo_direct_contact_panel.log"
    conda:
        "../../envs/base.yml"
    params:
        min_nonpolymer_mw = AR_CFG["rcsb_min_nonpolymer_mw"],
        min_queries = AR_CFG["dual_cold_min_queries"],
        min_targets = AR_CFG["dual_cold_min_targets"],
        min_documents = AR_CFG["dual_cold_min_documents"],
        max_target_fraction = AR_CFG["dual_cold_max_target_fraction"],
        min_effective_targets = AR_CFG["dual_cold_min_effective_targets"]
    shell:
        r"""
        python eval/build_rcsb_holo_panel.py \
            --raw-jsonl-gz {input.raw_jsonl_gz:q} \
            --source-manifest {input.source_manifest:q} \
            --train-parquet {input.train:q} \
            --dev-parquet {input.dev:q} \
            --benchmark-manifest {input.benchmark_manifest:q} \
            --screenable-target-clusters {input.screenable_target_clusters:q} \
            --screenable-target-cluster-manifest {input.screenable_target_cluster_manifest:q} \
            --out-pairs {output.pairs:q} \
            --out-exclusions {output.exclusions:q} \
            --out-ranking-queries {output.ranking_queries:q} \
            --out-manifest {output.manifest:q} \
            --min-mw {params.min_nonpolymer_mw:q} \
            --min-queries {params.min_queries:q} \
            --min-targets {params.min_targets:q} \
            --min-documents {params.min_documents:q} \
            --max-target-fraction {params.max_target_fraction:q} \
            --min-effective-targets {params.min_effective_targets:q} > {log:q} 2>&1
        """


rule activity_prior_pocket_fragments:
    input:
        targets = AR_PRIOR_POCKET_TARGETS,
        target_manifest = AR_PRIOR_POCKET_TARGET_MANIFEST,
        p2rank_complete = rules.p2rank_batch.output.pocket_marker
    output:
        fragments = directory(str(AR_PRIOR_POCKET_FRAGMENT_DIR)),
        index = AR_PRIOR_POCKET_FRAGMENT_INDEX,
        exclusions = AR_PRIOR_POCKET_FRAGMENT_EXCLUSIONS,
        manifest = AR_PRIOR_POCKET_FRAGMENT_MANIFEST
    log:
        "results/logs/activity_prior_pocket_fragments.log"
    conda:
        "../../envs/base.yml"
    params:
        pdb_dir = str(AF_CLEAN),
        pocket_dir = str(POCKETS),
        p2rank_params = AR_CFG["prior_pocket_p2rank_params"],
        p2rank_version = AR_CFG["prior_pocket_p2rank_version"],
        context_residues = AR_CFG["prior_pocket_context_residues"],
        min_fragment_residues = AR_CFG["prior_pocket_min_fragment_residues"]
    shell:
        r"""
        python scripts/build_pocket_fragments.py \
            --targets-csv {input.targets:q} \
            --target-cluster-manifest {input.target_manifest:q} \
            --pdb-dir {params.pdb_dir:q} \
            --pocket-dir {params.pocket_dir:q} \
            --p2rank-params {params.p2rank_params:q} \
            --p2rank-version {params.p2rank_version:q} \
            --context-residues {params.context_residues:q} \
            --min-fragment-residues {params.min_fragment_residues:q} \
            --out-dir {output.fragments:q} \
            --out-index {output.index:q} \
            --out-exclusions {output.exclusions:q} \
            --out-manifest {output.manifest:q} > {log:q} 2>&1
        """


rule rcsb_contact_pocket_fragments:
    input:
        pairs = rules.rcsb_holo_direct_contact_panel.output.pairs,
        panel_manifest = rules.rcsb_holo_direct_contact_panel.output.manifest
    output:
        coordinates = directory(str(AR_RCSB_CONTACT_COORDINATE_DIR)),
        fragments = directory(str(AR_RCSB_CONTACT_FRAGMENT_DIR)),
        index = AR_RCSB_CONTACT_FRAGMENT_INDEX,
        exclusions = AR_RCSB_CONTACT_FRAGMENT_EXCLUSIONS,
        manifest = AR_RCSB_CONTACT_FRAGMENT_MANIFEST
    log:
        "results/logs/rcsb_contact_pocket_fragments.log"
    threads:
        4
    conda:
        "../../envs/base.yml"
    params:
        context_residues = AR_CFG["rcsb_contact_context_residues"],
        retries = AR_CFG["rcsb_contact_download_retries"],
        timeout = AR_CFG["rcsb_contact_download_timeout"]
    shell:
        r"""
        python scripts/build_rcsb_contact_pocket_fragments.py \
            --pairs-parquet {input.pairs:q} \
            --panel-manifest {input.panel_manifest:q} \
            --coordinate-dir {output.coordinates:q} \
            --fragment-dir {output.fragments:q} \
            --out-index {output.index:q} \
            --out-exclusions {output.exclusions:q} \
            --out-manifest {output.manifest:q} \
            --context-residues {params.context_residues:q} \
            --retries {params.retries:q} \
            --timeout {params.timeout:q} \
            --workers {threads} > {log:q} 2>&1
        """


rule rcsb_contact_pocket_leakage_audit:
    input:
        benchmark_manifest = rules.activity_benchmark_partitions.output.manifest,
        train = rules.activity_benchmark_partitions.output.train,
        dev = rules.activity_benchmark_partitions.output.dev,
        rcsb_index = rules.rcsb_contact_pocket_fragments.output.index,
        rcsb_manifest = rules.rcsb_contact_pocket_fragments.output.manifest,
        rcsb_fragments = rules.rcsb_contact_pocket_fragments.output.fragments,
        prior_index = rules.activity_prior_pocket_fragments.output.index,
        prior_manifest = rules.activity_prior_pocket_fragments.output.manifest,
        prior_fragments = rules.activity_prior_pocket_fragments.output.fragments
    output:
        rows = AR_RCSB_POCKET_LEAKAGE_ROWS,
        manifest = AR_RCSB_POCKET_LEAKAGE_MANIFEST
    log:
        "results/logs/rcsb_contact_pocket_leakage_audit.log"
    threads:
        AR_CFG["pocket_leakage_threads"]
    conda:
        "../../envs/base.yml"
    params:
        foldseek = AR_CFG["pocket_leakage_foldseek"]
    shell:
        r"""
        FOLDSEEK_BIN="$(command -v {params.foldseek:q})"
        python eval/audit_rcsb_contact_pocket_leakage.py \
            --foldseek "$FOLDSEEK_BIN" \
            --threads {threads:q} \
            --activity-benchmark-manifest {input.benchmark_manifest:q} \
            --train-parquet {input.train:q} \
            --dev-parquet {input.dev:q} \
            --rcsb-index {input.rcsb_index:q} \
            --rcsb-manifest {input.rcsb_manifest:q} \
            --rcsb-fragment-dir {input.rcsb_fragments:q} \
            --prior-index {input.prior_index:q} \
            --prior-manifest {input.prior_manifest:q} \
            --prior-fragment-dir {input.prior_fragments:q} \
            --out-csv {output.rows:q} \
            --out-manifest {output.manifest:q} > {log:q} 2>&1
        """


rule activity_recovery_panels:
    input:
        dev = rules.activity_benchmark_partitions.output.dev,
        test = rules.activity_benchmark_partitions.output.test,
        dual_cold = rules.activity_benchmark_partitions.output.dual_cold,
        benchmark_manifest = rules.activity_benchmark_partitions.output.manifest,
        retrieval_index_manifest = rules.activity_retrieval_index.output.manifest,
        rcsb_ranking_queries = rules.rcsb_holo_direct_contact_panel.output.ranking_queries,
        rcsb_panel_manifest = rules.rcsb_holo_direct_contact_panel.output.manifest,
        known_panel = AR_KNOWN_PANEL
    output:
        dev_ranking = AR_PANELS_DIR / "dev_ranking_queries.parquet",
        test_ranking = AR_PANELS_DIR / "test_ranking_queries.parquet",
        dual_cold_ranking = AR_PANELS_DIR / "dual_cold_ranking_queries.parquet",
        dev_calibration = AR_PANELS_DIR / "dev_calibration_pairs.parquet",
        test_calibration = AR_PANELS_DIR / "test_calibration_pairs.parquet",
        known_panel = AR_PANELS_DIR / "known_panel.csv",
        manifest = AR_PANELS_DIR / "manifest.json"
    log:
        "results/logs/activity_recovery_panels.log"
    conda:
        "../../envs/base.yml"
    params:
        outdir = str(AR_PANELS_DIR),
        dev_query_count = AR_CFG["dev_query_count"],
        test_query_count = AR_CFG["test_query_count"],
        dual_cold_min_queries = AR_CFG["dual_cold_min_queries"],
        dual_cold_min_targets = AR_CFG["dual_cold_min_targets"],
        dual_cold_min_documents = AR_CFG["dual_cold_min_documents"],
        dual_cold_max_target_fraction = AR_CFG["dual_cold_max_target_fraction"],
        dual_cold_min_effective_targets = AR_CFG["dual_cold_min_effective_targets"],
        calibration_cap = AR_CFG["calibration_cap"],
        negative_threshold = AR_CFG["negative_threshold"],
        positive_threshold = AR_CFG["positive_threshold"]
    shell:
        r"""
        python eval/build_activity_recovery_panels.py \
            --dev-parquet {input.dev:q} \
            --test-parquet {input.test:q} \
            --dual-cold-parquet {input.dual_cold:q} \
            --benchmark-manifest {input.benchmark_manifest:q} \
            --retrieval-index-manifest {input.retrieval_index_manifest:q} \
            --rcsb-ranking-queries {input.rcsb_ranking_queries:q} \
            --rcsb-panel-manifest {input.rcsb_panel_manifest:q} \
            --known-panel {input.known_panel:q} \
            --out-dir {params.outdir:q} \
            --dev-query-count {params.dev_query_count:q} \
            --test-query-count {params.test_query_count:q} \
            --dual-cold-min-queries {params.dual_cold_min_queries:q} \
            --dual-cold-min-targets {params.dual_cold_min_targets:q} \
            --dual-cold-min-documents {params.dual_cold_min_documents:q} \
            --dual-cold-max-target-fraction {params.dual_cold_max_target_fraction:q} \
            --dual-cold-min-effective-targets {params.dual_cold_min_effective_targets:q} \
            --calibration-cap {params.calibration_cap:q} \
            --negative-threshold {params.negative_threshold:q} \
            --positive-threshold {params.positive_threshold:q} > {log:q} 2>&1
        """


rule activity_retrieval_dev_selection:
    input:
        ligands = rules.activity_retrieval_index.output.ligands,
        edges = rules.activity_retrieval_index.output.edges,
        index_manifest = rules.activity_retrieval_index.output.manifest,
        target_clusters = AR_SCREENABLE_TARGET_CLUSTERS,
        target_cluster_manifest = AR_SCREENABLE_TARGET_CLUSTER_MANIFEST,
        panels_manifest = rules.activity_recovery_panels.output.manifest,
        ranking_panel = rules.activity_recovery_panels.output.dev_ranking,
        calibration_panel = rules.activity_recovery_panels.output.dev_calibration
    output:
        recipe = AR_DEV_SELECTION_DIR / "recipe.json",
        manifest = AR_DEV_SELECTION_DIR / "manifest.json"
    log:
        "results/logs/activity_retrieval_dev_selection.log"
    conda:
        "../../envs/base.yml"
    params:
        outdir = str(AR_DEV_SELECTION_DIR),
        exclude_similarity = AR_CFG["exclude_reference_similarity"],
        allow_no_improvement = AR_ALLOW_NO_IMPROVEMENT_ARG
    shell:
        r"""
        python eval/activity_retrieval_model.py select \
            --ligands {input.ligands:q} \
            --edges {input.edges:q} \
            --index-manifest {input.index_manifest:q} \
            --target-clusters {input.target_clusters:q} \
            --target-cluster-manifest {input.target_cluster_manifest:q} \
            --panels-manifest {input.panels_manifest:q} \
            --ranking-panel {input.ranking_panel:q} \
            --calibration-panel {input.calibration_panel:q} \
            --exclude-reference-similarity {params.exclude_similarity:q} \
            --out-dir {params.outdir:q} \
            {params.allow_no_improvement} > {log:q} 2>&1
        """


rule activity_retrieval_final_evaluation:
    input:
        ligands = rules.activity_retrieval_index.output.ligands,
        edges = rules.activity_retrieval_index.output.edges,
        index_manifest = rules.activity_retrieval_index.output.manifest,
        target_clusters = AR_SCREENABLE_TARGET_CLUSTERS,
        target_cluster_manifest = AR_SCREENABLE_TARGET_CLUSTER_MANIFEST,
        panels_manifest = rules.activity_recovery_panels.output.manifest,
        recipe = rules.activity_retrieval_dev_selection.output.recipe,
        ranking_panel = rules.activity_recovery_panels.output.test_ranking,
        dual_cold_panel = rules.activity_recovery_panels.output.dual_cold_ranking,
        calibration_panel = rules.activity_recovery_panels.output.test_calibration,
        known_panel = rules.activity_recovery_panels.output.known_panel
    output:
        summary = AR_FINAL_EVAL_DIR / "summary.json",
        manifest = AR_FINAL_EVAL_DIR / "manifest.json"
    log:
        "results/logs/activity_retrieval_final_evaluation.log"
    conda:
        "../../envs/base.yml"
    params:
        outdir = str(AR_FINAL_EVAL_DIR),
        exclude_similarity = AR_CFG["exclude_reference_similarity"]
    shell:
        r"""
        python eval/activity_retrieval_model.py evaluate \
            --ligands {input.ligands:q} \
            --edges {input.edges:q} \
            --index-manifest {input.index_manifest:q} \
            --target-clusters {input.target_clusters:q} \
            --target-cluster-manifest {input.target_cluster_manifest:q} \
            --panels-manifest {input.panels_manifest:q} \
            --recipe {input.recipe:q} \
            --ranking-panel {input.ranking_panel:q} \
            --dual-cold-panel {input.dual_cold_panel:q} \
            --calibration-panel {input.calibration_panel:q} \
            --known-panel {input.known_panel:q} \
            --exclude-reference-similarity {params.exclude_similarity:q} \
            --out-dir {params.outdir:q} \
            --allow-no-improvement > {log:q} 2>&1
        """


rule activity_retrieval_operational_gate:
    input:
        benchmark = rules.activity_benchmark_partitions.output.manifest,
        index = rules.activity_retrieval_index.output.manifest,
        panels = rules.activity_recovery_panels.output.manifest,
        dev_selection = rules.activity_retrieval_dev_selection.output.manifest,
        final_evaluation = rules.activity_retrieval_final_evaluation.output.manifest,
        pocket_leakage = rules.rcsb_contact_pocket_leakage_audit.output.manifest,
        runtime_index = str(
            Path(config["docking"]["daina_recipe_index_dir"]) / "manifest.json"
        ),
        runtime_recipe = str(Path(config["docking"]["daina_recipe_path"]))
    output:
        str(MANIFESTS / "activity_retrieval_operational_gate.flag")
    log:
        "results/logs/activity_retrieval_operational_gate.log"
    conda:
        "../../envs/base.yml"
    shell:
        r"""
        python scripts/validate_activity_retrieval_gate.py create-operational \
            --benchmark-manifest {input.benchmark:q} \
            --index-manifest {input.index:q} \
            --panels-manifest {input.panels:q} \
            --selection-manifest {input.dev_selection:q} \
            --final-evaluation-manifest {input.final_evaluation:q} \
            --pocket-leakage-manifest {input.pocket_leakage:q} \
            --runtime-index-manifest {input.runtime_index:q} \
            --runtime-recipe {input.runtime_recipe:q} \
            --out-gate {output:q} > {log:q} 2>&1
        """


rule activity_retrieval_final_gate:
    input:
        benchmark = rules.activity_benchmark_partitions.output.manifest,
        index = rules.activity_retrieval_index.output.manifest,
        panels = rules.activity_recovery_panels.output.manifest,
        dev_selection = rules.activity_retrieval_dev_selection.output.manifest,
        final_evaluation = rules.activity_retrieval_final_evaluation.output.manifest,
        pocket_leakage = rules.rcsb_contact_pocket_leakage_audit.output.manifest
    output:
        str(MANIFESTS / "activity_retrieval_final_gate.flag")
    log:
        "results/logs/activity_retrieval_final_gate.log"
    conda:
        "../../envs/base.yml"
    shell:
        r"""
        python scripts/validate_activity_retrieval_gate.py create \
            --benchmark-manifest {input.benchmark:q} \
            --index-manifest {input.index:q} \
            --panels-manifest {input.panels:q} \
            --selection-manifest {input.dev_selection:q} \
            --final-evaluation-manifest {input.final_evaluation:q} \
            --pocket-leakage-manifest {input.pocket_leakage:q} \
            --out-gate {output:q} > {log:q} 2>&1
        """
