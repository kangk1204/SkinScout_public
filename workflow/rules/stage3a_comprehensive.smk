# =============================================================================
# Stage 3a — MODE-COMPREHENSIVE (bias-free, default)
# =============================================================================
# Pre-req: Stage 0 infra + Stage 2 decision != HALT
# Input:   01_input/compound_canonical.sdf
# Output:  03_targets/mode_comprehensive/
#            ├── ligand.pdbqt
#            ├── autodock_all_targets.tsv          (20.4k scores)
#            ├── autodock_all_target_poses/        (<target_id>.sdf docked poses)
#            ├── top_pct_pre_rescore.csv           (top 1-2 %)
#            ├── gnina_rescores.tsv
#            ├── rtmscore_rescores.tsv
#            ├── boltz2_affinity_top.tsv
#            ├── top50_4way_consensus.csv          (final RRF)
#            └── disagreement_analysis.json        (when run_dti_sanity)
# =============================================================================

# S3C_DIR / S3F_DIR are defined in workflow/Snakefile before the includes.


rule meeko_ligand:
    input:
        sdf = rules.xtb_optimize.output.sdf,
        decision = rules.skin_sens_consensus.output.decision,
        cosmetic_decision = rules.cosmetic_drug_decision.output.decision,
        activity_retrieval_gate = str(
            MANIFESTS / "activity_retrieval_operational_gate.flag"
        )
    output:
        pdbqt = S3C_DIR / "ligand.pdbqt"
    log:
        "results/logs/{run_id}/stage3_meeko_ligand.log".format(run_id=RUN_ID)
    conda:
        "../../envs/meeko.yml"
    shell:
        r"""
        python scripts/validate_activity_retrieval_gate.py check-operational \
            --gate {input.activity_retrieval_gate:q}
        # Refuse to proceed on HALT safety gates.
        decision=$(head -n 1 {input.decision:q})
        if [ "$decision" = "HALT" ]; then
            echo "[stage3] skin-sens HALT — refusing to dock." >&2
            exit 10
        fi
        cosmetic_decision=$(head -n 1 {input.cosmetic_decision:q})
        if [ "$cosmetic_decision" = "HALT" ]; then
            echo "[stage3] cosmetic/drug HALT — refusing to dock." >&2
            exit 11
        fi
        mkdir -p "$(dirname {output.pdbqt:q})"
        python scripts/stage3_meeko_ligand.py \
            --in-sdf {input.sdf:q} --out-pdbqt {output.pdbqt:q} > {log:q} 2>&1
        """


# The comprehensive path docks every receptor, so its map inputs come from an
# unranked receptor list rather than a ranked selection.
rule comprehensive_receptor_list:
    input:
        pdbqt_marker = PDBQT / ".pdbqt_complete"
    output:
        listing = S3C_DIR / "receptor_list.txt"
    params:
        receptor_dir = str(PDBQT),
        max_receptors = DOCKING["comprehensive_max_receptors"]
    shell:
        r"""
        find {params.receptor_dir:q} -maxdepth 1 -name '*.pdbqt' -printf '%f\n' \
          | sed 's/\.pdbqt$//' | LC_ALL=C sort > {output.listing:q}.all
        if [ "{params.max_receptors}" -gt 0 ]; then
            head -n {params.max_receptors} {output.listing:q}.all > {output.listing:q}
        else
            mv {output.listing:q}.all {output.listing:q}
        fi
        rm -f {output.listing:q}.all
        """


# AutoGrid existed only in the fast path, so autodock_gpu_all had no .maps.fld
# inputs to find and failed closed at 0/15038 coverage - the comprehensive DAG
# could not run at all, whatever else was installed.
rule comprehensive_autogrid_maps:
    input:
        listing = rules.comprehensive_receptor_list.output.listing,
        ligand = rules.meeko_ligand.output.pdbqt,
        no_pocket = str(NO_POCKET)
    output:
        maps = directory(S3C_DIR / "autogrid_maps"),
        manifest = S3C_DIR / "autogrid_map_manifest.json"
    log:
        "results/logs/{run_id}/stage3_comprehensive_autogrid.log".format(run_id=RUN_ID)
    conda:
        "../../envs/autodock_gpu.yml"
    params:
        receptor_dir = str(PDBQT),
        box_dir = str(BOXES),
        spacing = DOCKING["fast_mode_autogrid_spacing"],
        cache_dir = DOCKING["fast_mode_autogrid_cache"]
    shell:
        r"""
        python scripts/stage3_autogrid_maps.py \
            --receptor-list {input.listing:q} \
            --ligand-pdbqt {input.ligand:q} \
            --receptor-dir {params.receptor_dir:q} \
            --box-dir {params.box_dir:q} \
            --no-pocket-list {input.no_pocket:q} \
            --spacing {params.spacing:q} \
            --cache-dir {params.cache_dir:q} \
            --out-map-dir {output.maps:q} \
            --out-manifest {output.manifest:q} > {log:q} 2>&1
        """


rule autodock_gpu_all:
    input:
        ligand   = rules.meeko_ligand.output.pdbqt,
        pdbqt    = PDBQT / ".pdbqt_complete",      # Stage 0 marker
        maps     = rules.comprehensive_autogrid_maps.output.maps,
        map_manifest = rules.comprehensive_autogrid_maps.output.manifest,
        no_pocket = str(NO_POCKET)
    output:
        scores = S3C_DIR / "autodock_all_targets.tsv",
        poses = directory(S3C_DIR / "autodock_all_target_poses")
    log:
        "results/logs/{run_id}/stage3_autodock_all.log".format(run_id=RUN_ID)
    conda:
        "../../envs/autodock_gpu.yml"
    threads: config["hardware"]["cpu_cores"]
    resources:
        gpu = 1
    params:
        receptor_dir = str(PDBQT),
        box_dir      = str(BOXES),
        nrun         = config["docking"]["autodock_runs"],
        volume_advisory = config["docking"]["search_volume_advisory"],
        high_volume_runs = config["docking"]["high_volume_min_runs"],
        ls_method    = config["docking"]["autodock_local_search"]
    shell:
        r"""
        python scripts/stage3_autodock_run.py \
            --ligand-pdbqt {input.ligand:q} \
            --receptor-dir {params.receptor_dir:q} \
            --box-dir {params.box_dir:q} \
            --no-pocket-list {input.no_pocket:q} \
            --map-manifest {input.map_manifest:q} \
            --engine autodock_gpu \
            --nrun {params.nrun:q} \
            --search-volume-advisory {params.volume_advisory:q} \
            --high-volume-min-nrun {params.high_volume_runs:q} \
            --ls-method {params.ls_method:q} \
            --vina-cpu {threads:q} \
            --out-scores {output.scores:q} \
            --out-pose-dir {output.poses:q} > {log:q} 2>&1
        """


rule diffdock_blind_no_pocket:
    """Blind docking for receptors with no usable P2Rank pocket."""
    input:
        ligand    = rules.meeko_ligand.output.pdbqt,
        no_pocket = str(NO_POCKET)
    output:
        scores = S3C_DIR / "diffdock_no_pocket_scores.tsv"
    log:
        "results/logs/{run_id}/stage3_diffdock_blind.log".format(run_id=RUN_ID)
    conda:
        "../../envs/boltz2.yml"
    resources:
        gpu = 1
    params:
        clean_dir = str(AF_CLEAN)
    shell:
        r"""
        python scripts/stage3_diffdock_blind.py \
            --ligand {input.ligand:q} \
            --no-pocket-list {input.no_pocket:q} \
            --clean-dir {params.clean_dir:q} \
            --out-scores {output.scores:q} > {log:q} 2>&1
        """


rule autodock_pick_top_pct:
    input:
        all_scores = rules.autodock_gpu_all.output.scores,
        blind      = rules.diffdock_blind_no_pocket.output.scores
    output:
        top = S3C_DIR / "top_pct_pre_rescore.csv"
    conda:
        "../../envs/base.yml"
    params:
        top_pct = config["docking"]["top_pct_to_rescore"]
    shell:
        r"""
        python scripts/stage3_pick_top.py \
            --autodock-scores {input.all_scores:q} \
            --blind-scores {input.blind:q} \
            --top-pct {params.top_pct:q} \
            --out-csv {output.top:q}
        """


# Rescores the pose AutoDock produced, not a free ligand conformer. It used to
# read rules.xtb_optimize.output.sdf, so its score described a geometry that had
# nothing to do with the docking it was presented as rescoring. The manifest
# covers every docked target while the score table is a top percentage, hence
# --allow-unscored-poses; a target in the score table with no verified pose is
# still fatal.
rule gnina_rescore_top:
    input:
        poses = rules.autodock_gpu_all.output.poses,
        top   = rules.autodock_pick_top_pct.output.top
    output:
        scores = S3C_DIR / "gnina_rescores.tsv",
        status = S3C_DIR / "gnina_status.json"
    log:
        "results/logs/{run_id}/stage3_gnina.log".format(run_id=RUN_ID)
    conda:
        "../../envs/autodock_gpu.yml"
    resources:
        gpu = 1
    params:
        clean_dir = str(AF_CLEAN)
    shell:
        r"""
        python scripts/stage3_gnina_rescore.py \
            --top-csv {input.top:q} \
            --pose-manifest {input.poses:q}/pose_manifest.json \
            --pose-dir {input.poses:q} \
            --allow-unscored-poses \
            --clean-dir {params.clean_dir:q} \
            --out-scores {output.scores:q} \
            --out-status-manifest {output.status:q} > {log:q} 2>&1
        """


# Same pose as AutoDock and GNINA. This also read the free ligand conformer
# until 2026-08-26; RTMScore had in fact never produced a score on this host,
# because its DGL backend is CPU-only while the adapter selected CUDA whenever
# a GPU was visible, and every failure was swallowed at debug level.
rule rtmscore_top:
    input:
        poses = rules.autodock_gpu_all.output.poses,
        top   = rules.autodock_pick_top_pct.output.top
    output:
        scores = S3C_DIR / "rtmscore_rescores.tsv",
        status = S3C_DIR / "rtmscore_status.json"
    log:
        "results/logs/{run_id}/stage3_rtmscore.log".format(run_id=RUN_ID)
    conda:
        "../../envs/boltz2.yml"
    resources:
        gpu = 1
    params:
        clean_dir = str(AF_CLEAN)
    shell:
        r"""
        python scripts/stage3_rtmscore.py \
            --top-csv {input.top:q} \
            --pose-manifest {input.poses:q}/pose_manifest.json \
            --pose-dir {input.poses:q} \
            --allow-unscored-poses \
            --clean-dir {params.clean_dir:q} \
            --out-scores {output.scores:q} \
            --out-status-manifest {output.status:q} > {log:q} 2>&1
        """


rule boltz2_affinity_top:
    input:
        ligand_sdf = rules.xtb_optimize.output.sdf,
        top        = rules.autodock_pick_top_pct.output.top
    output:
        scores = S3C_DIR / "boltz2_affinity_top.tsv"
    log:
        "results/logs/{run_id}/stage3_boltz2_affinity.log".format(run_id=RUN_ID)
    conda:
        "../../envs/boltz2.yml"
    resources:
        gpu = 1
    params:
        clean_dir = str(AF_CLEAN),
        box_dir   = str(BOXES),
        max_res   = config["hardware"]["boltz2_max_residues"],
        crop_r    = config["hardware"]["pocket_crop_radius"]
    shell:
        r"""
        python scripts/stage3_boltz2_affinity.py \
            --top-csv {input.top:q} \
            --ligand-sdf {input.ligand_sdf:q} \
            --clean-dir {params.clean_dir:q} \
            --box-dir {params.box_dir:q} \
            --max-residues {params.max_res:q} \
            --crop-radius {params.crop_r:q} \
            --out-scores {output.scores:q} > {log:q} 2>&1
        """


# Three of the four columns now describe the AutoDock pose: AutoDock scores the
# pose it produced, and GNINA and RTMScore each rescore that same pose through
# the SHA-verified manifest. Boltz-2 is a separate structure predictor and
# never sees the pose, so this is still a rank fusion over four methods rather
# than four scores of one geometry.
rule rrf_4way_consensus:
    input:
        autodock = rules.autodock_pick_top_pct.output.top,
        gnina    = rules.gnina_rescore_top.output.scores,
        rtm      = rules.rtmscore_top.output.scores,
        boltz    = rules.boltz2_affinity_top.output.scores
    output:
        top50 = S3C_DIR / "top50_4way_consensus.csv"
    conda:
        "../../envs/base.yml"
    params:
        rrf_k = config["docking"]["rrf_k"],
        top_n = config["docking"]["top_n_consensus"],
        min_sources = config["docking"]["min_sources_comprehensive"],
        source_weights = ",".join(
            f"{label}={DOCKING['comprehensive_rrf_source_weights'][label]}"
            for label in ("autodock", "gnina", "rtm", "boltz")
        ),
        recipe_id = DOCKING["comprehensive_rrf_recipe_id"]
    shell:
        r"""
        python scripts/stage3_rrf.py \
            --inputs {input.autodock:q}=autodock,{input.gnina:q}=gnina,{input.rtm:q}=rtm,{input.boltz:q}=boltz \
            --rrf-k {params.rrf_k:q} \
            --source-weights {params.source_weights:q} \
            --recipe-id {params.recipe_id:q} \
            --min-sources-per-target {params.min_sources:q} \
            --top-n {params.top_n:q} \
            --out-csv {output.top50:q}
        """
