# =============================================================================
# Stage 3b — MODE-FAST (Daina head + measured-band structural re-ranking)
# =============================================================================
# Output: results/runs/<run_id>/03_targets/mode_fast/
#           ├── daina_zoete_proteome.tsv
#           ├── daina_top256.csv
#           ├── dti_rrf_top25pct.csv       (legacy compatibility projection)
#           ├── autogrid_map_manifest.json
#           ├── autogrid_maps/
#           ├── autodock_top5k.tsv
#           ├── autodock_top5k_poses/
#           ├── gnina_pose_rescores.tsv
#           ├── daina_structural_targets.csv
#           └── top50.csv                  (Daina-sorted compatibility view)
# =============================================================================


# Retrieval reads the fingerprints and, beside them, human_activities.parquet.
# `daina_evidence_dir` lets that be the merged BindingDB/GtoPdb table instead of
# ChEMBL alone. It stays on data/chembl37 by default: the merged table reaches 248
# more targets, and every recovery number the guide reports was measured without
# them, so switching it invalidates those numbers until the panel is re-run.
DAINA_EVIDENCE_DIR = Path(DOCKING.get("daina_evidence_dir", config["paths"]["chembl"]))
DAINA_USES_MERGED_EVIDENCE = DAINA_EVIDENCE_DIR != Path(config["paths"]["chembl"])

# Ranking by the promoted recipe instead of nearest neighbour. This reads the
# retrieval index rather than the mirror, so both paths must exist before the
# switch does anything - and the index has to declare index_role=production, or
# the run would be scored against the benchmark's train split.
DAINA_RECIPE_SCORING = config_bool(
    DOCKING.get("daina_recipe_scoring"), False
)
DAINA_RECIPE_PATH = Path(DOCKING["daina_recipe_path"]) if DAINA_RECIPE_SCORING else None
DAINA_RECIPE_INDEX = (
    Path(DOCKING["daina_recipe_index_dir"]) if DAINA_RECIPE_SCORING else None
)
DAINA_RECIPE_INPUTS = (
    [
        str(DAINA_RECIPE_PATH),
        str(DAINA_RECIPE_INDEX / "ligands.parquet"),
        str(DAINA_RECIPE_INDEX / "edges.parquet"),
        str(DAINA_RECIPE_INDEX / "manifest.json"),
    ]
    if DAINA_RECIPE_SCORING
    else []
)


rule daina_zoete:
    input:
        sdf = rules.xtb_optimize.output.sdf,
        chembl_fp = (
            []
            if DAINA_RECIPE_SCORING
            else
            str(DAINA_EVIDENCE_DIR / "fp_morgan2_2048.parquet")
            if DAINA_USES_MERGED_EVIDENCE
            else rules.mirror_chembl.output.fingerprints
        ),
        chembl_manifest = (
            [] if DAINA_RECIPE_SCORING else rules.mirror_chembl.output.manifest
        ),
        decision = rules.skin_sens_consensus.output.decision,
        cosmetic_decision = rules.cosmetic_drug_decision.output.decision,
        activity_retrieval_gate = str(
            MANIFESTS / "activity_retrieval_operational_gate.flag"
        ),
        recipe_files = DAINA_RECIPE_INPUTS
    output:
        scores = S3F_DIR / "daina_zoete_proteome.tsv",
        metadata = S3F_DIR / "daina_zoete_proteome.metadata.json"
    log:
        "results/logs/{run_id}/stage3_daina_zoete.log".format(run_id=RUN_ID)
    conda:
        "../../envs/dti.yml"
    threads: 16
    params:
        evidence_mode = DOCKING["daina_evidence_mode"],
        quality_policy = DOCKING["daina_quality_policy"],
        scoring_method = DOCKING["daina_scoring_method"],
        exclude_similarity = DOCKING["daina_exclude_reference_similarity"],
        evidence_snapshot_arg = (
            ""
            if DOCKING["daina_evidence_mode"] == "retrieval"
            else "--evidence-snapshot-id "
            + shlex.quote(str(CHEMBL / "source_manifest.json"))
        ),
        recipe_arg = (
            "--recipe " + shlex.quote(str(DAINA_RECIPE_PATH))
            + " --recipe-index-dir " + shlex.quote(str(DAINA_RECIPE_INDEX))
            + " --operational-gate "
            + shlex.quote(str(MANIFESTS / "activity_retrieval_operational_gate.flag"))
            if DAINA_RECIPE_SCORING
            else ""
        ),
        chembl_arg = (
            ""
            if DAINA_RECIPE_SCORING
            else "--chembl-fp " + shlex.quote(
                str(
                    DAINA_EVIDENCE_DIR / "fp_morgan2_2048.parquet"
                    if DAINA_USES_MERGED_EVIDENCE
                    else rules.mirror_chembl.output.fingerprints
                )
            )
        ),
        cutoff_arg = (
            "--cutoff-date "
            + shlex.quote(str(config["evaluation"]["common_cutoff_date"]))
            if DOCKING["daina_evidence_mode"] == "temporal"
            else ""
        )
    shell:
        r"""
        python scripts/validate_activity_retrieval_gate.py check-operational \
            --gate {input.activity_retrieval_gate:q}
        decision=$(head -n 1 {input.decision:q})
        if [ "$decision" = "HALT" ]; then
            echo "[stage3] skin-sens HALT — refusing fast DTI." >&2
            exit 10
        fi
        cosmetic_decision=$(head -n 1 {input.cosmetic_decision:q})
        if [ "$cosmetic_decision" = "HALT" ]; then
            echo "[stage3] cosmetic/drug HALT — refusing fast DTI." >&2
            exit 11
        fi
        python scripts/stage3_daina_zoete.py \
            --ligand-sdf {input.sdf:q} \
            {params.chembl_arg} \
            --evidence-mode {params.evidence_mode:q} \
            --quality-policy {params.quality_policy:q} \
            --scoring-method {params.scoring_method:q} \
            --exclude-reference-similarity {params.exclude_similarity:q} \
            {params.evidence_snapshot_arg} \
            {params.cutoff_arg} \
            {params.recipe_arg} \
            --out-scores {output.scores:q} \
            --out-metadata-json {output.metadata:q} > {log:q} 2>&1
        """


rule dti_rrf_fast:
    input:
        daina = rules.daina_zoete.output.scores,
        metadata = rules.daina_zoete.output.metadata
    output:
        selected = S3F_DIR / "daina_top256.csv",
        top = S3F_DIR / "dti_rrf_top25pct.csv"
    conda:
        "../../envs/base.yml"
    params:
        top_n = DOCKING["fast_mode_daina_top_n"]
    shell:
        r"""
        python scripts/stage3_select_daina.py \
            --daina-scores {input.daina:q} \
            --daina-metadata {input.metadata:q} \
            --top-n {params.top_n:q} \
            --out-csv {output.selected:q} \
            --out-compat-csv {output.top:q}
        """


rule fast_autogrid_maps:
    input:
        selected = rules.dti_rrf_fast.output.selected,
        ligand = rules.meeko_ligand.output.pdbqt,
        pdbqt_marker = PDBQT / ".pdbqt_complete",
        no_pocket = str(NO_POCKET)
    output:
        maps = directory(S3F_DIR / "autogrid_maps"),
        manifest = S3F_DIR / "autogrid_map_manifest.json"
    log:
        "results/logs/{run_id}/stage3_fast_autogrid.log".format(run_id=RUN_ID)
    conda:
        "../../envs/autodock_gpu.yml"
    params:
        receptor_dir = str(PDBQT),
        box_dir = str(BOXES),
        spacing = DOCKING["fast_mode_autogrid_spacing"],
        cache_dir = DOCKING["fast_mode_autogrid_cache"],
        # Grids outside the re-ranking band are never read, so they are not
        # built. Targets keep a row in the manifest with a status of their own.
        dock_rank_from = (
            DOCKING["fast_mode_rerank_keep"] + 1
            if DOCKING["fast_mode_dock_band_only"] else 0
        ),
        dock_rank_to = (
            DOCKING["fast_mode_rerank_band"]
            if DOCKING["fast_mode_dock_band_only"] else 0
        )
    shell:
        r"""
        python scripts/stage3_autogrid_maps.py \
            --selected-csv {input.selected:q} \
            --ligand-pdbqt {input.ligand:q} \
            --receptor-dir {params.receptor_dir:q} \
            --box-dir {params.box_dir:q} \
            --no-pocket-list {input.no_pocket:q} \
            --spacing {params.spacing:q} \
            --cache-dir {params.cache_dir:q} \
            --dock-rank-from {params.dock_rank_from:q} \
            --dock-rank-to {params.dock_rank_to:q} \
            --out-map-dir {output.maps:q} \
            --out-manifest {output.manifest:q} > {log:q} 2>&1
        """


rule autodock_top5k:
    input:
        ligand = rules.meeko_ligand.output.pdbqt,
        selected = rules.dti_rrf_fast.output.selected,
        maps = rules.fast_autogrid_maps.output.maps,
        map_manifest = rules.fast_autogrid_maps.output.manifest
    output:
        scores = S3F_DIR / "autodock_top5k.tsv",
        poses = directory(S3F_DIR / "autodock_top5k_poses"),
        status = S3F_DIR / "autodock_status_manifest.json"
    log:
        "results/logs/{run_id}/stage3_autodock_top5k.log".format(run_id=RUN_ID)
    conda:
        "../../envs/autodock_gpu.yml"
    threads: config["hardware"]["cpu_cores"]
    resources:
        gpu = 1
    params:
        receptor_dir = str(PDBQT),
        box_dir      = str(BOXES),
        nrun         = config["docking"]["fast_mode_autodock_runs"],
        volume_advisory = config["docking"]["search_volume_advisory"],
        high_volume_runs = config["docking"]["high_volume_min_runs"],
        ls_method    = config["docking"]["autodock_local_search"],
        max_receptors = config["docking"]["fast_mode_autodock_max_receptors"]
    shell:
        r"""
        python scripts/stage3_autodock_run.py \
            --ligand-pdbqt {input.ligand:q} \
            --receptor-dir {params.receptor_dir:q} \
            --box-dir {params.box_dir:q} \
            --map-manifest {input.map_manifest:q} \
            --allow-partial-structure \
            --engine autodock_gpu \
            --nrun {params.nrun:q} \
            --search-volume-advisory {params.volume_advisory:q} \
            --high-volume-min-nrun {params.high_volume_runs:q} \
            --ls-method {params.ls_method:q} \
            --max-receptors {params.max_receptors:q} \
            --vina-cpu {threads:q} \
            --out-scores {output.scores:q} \
            --out-pose-dir {output.poses:q} \
            --out-status-manifest {output.status:q} > {log:q} 2>&1
        """


rule fast_gnina_pose_rescore:
    input:
        autodock = rules.autodock_top5k.output.scores,
        poses = rules.autodock_top5k.output.poses
    output:
        scores = S3F_DIR / "gnina_pose_rescores.tsv",
        status = S3F_DIR / "gnina_pose_status_manifest.json"
    log:
        "results/logs/{run_id}/stage3_fast_gnina_pose.log".format(run_id=RUN_ID)
    conda:
        "../../envs/autodock_gpu.yml"
    resources:
        gpu = 1
    params:
        clean_dir = str(AF_CLEAN)
    shell:
        r"""
        python scripts/stage3_gnina_rescore.py \
            --top-csv {input.autodock:q} \
            --pose-manifest {input.poses:q}/pose_manifest.json \
            --pose-dir {input.poses:q} \
            --clean-dir {params.clean_dir:q} \
            --use-gpu \
            --allow-partial-structure \
            --out-scores {output.scores:q} \
            --out-status-manifest {output.status:q} > {log:q} 2>&1
        """


rule fast_rerank_consensus:
    input:
        selected = rules.dti_rrf_fast.output.selected,
        map_manifest = rules.fast_autogrid_maps.output.manifest,
        autodock = rules.autodock_top5k.output.scores,
        autodock_status = rules.autodock_top5k.output.status,
        gnina = rules.fast_gnina_pose_rescore.output.scores,
        gnina_status = rules.fast_gnina_pose_rescore.output.status,
        skin = SKIN_EXPR_DIR / "skin_score.tsv",
        known_target_priors = rules.known_target_prior.output.csv,
        efficacy_kg = KG_DIR / "skin_efficacy.graphml"
    output:
        canonical = S3F_DIR / "daina_structural_targets.csv",
        top50 = S3F_DIR / "top50.csv"
    log:
        "results/logs/{run_id}/stage3_fast_overlay.log".format(run_id=RUN_ID)
    conda:
        "../../envs/base.yml"
    shell:
        r"""
        python scripts/stage3_daina_structural_overlay.py \
            --selected-csv {input.selected:q} \
            --map-manifest {input.map_manifest:q} \
            --docking-scores {input.autodock:q} \
            --docking-status-manifest {input.autodock_status:q} \
            --gnina-scores {input.gnina:q} \
            --gnina-status-manifest {input.gnina_status:q} \
            --skin-tsv {input.skin:q} \
            --skin-kg {input.efficacy_kg:q} \
            --known-target-priors {input.known_target_priors:q} \
            --out-csv {output.canonical:q} \
            --out-top50 {output.top50:q} > {log:q} 2>&1
        """


# The similarity order is what the canonical artifact carries; this produces the
# reader-facing order beside it. Measured on the 15-compound panel
# (docs/RERANK_EXPERIMENT_20260828.md): holding the head and re-ordering ranks
# 11-50 moves top-30 recovery from 14/28 to 20/28, while re-ranking everything
# is worse than leaving it alone. Additive - daina_structural_targets.csv and
# top50.csv are untouched.
rule fast_band_rerank:
    input:
        canonical = rules.fast_rerank_consensus.output.canonical
    output:
        reranked = S3F_DIR / "daina_band_reranked_targets.csv",
        top50 = S3F_DIR / "top50_band_reranked.csv"
    log:
        "results/logs/{run_id}/stage3_fast_band_rerank.log".format(run_id=RUN_ID)
    conda:
        "../../envs/base.yml"
    params:
        keep = DOCKING["fast_mode_rerank_keep"],
        band = DOCKING["fast_mode_rerank_band"]
    shell:
        r"""
        python scripts/stage3_band_rerank.py \
            --in-csv {input.canonical:q} \
            --keep {params.keep:q} \
            --band {params.band:q} \
            --out-csv {output.reranked:q} \
            --out-top50 {output.top50:q} > {log:q} 2>&1
        """
