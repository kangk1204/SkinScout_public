# =============================================================================
# Stage 6 — BioEmu ensemble + ensemble docking (top 10)
# =============================================================================

S6_DIR = RUN_DIR / "06_bioemu"


rule bioemu_ensemble:
    input:
        boltz_report = rules.boltz2_cofold.output.report
    output:
        cluster_manifest = S6_DIR / "ensemble_manifest.tsv"
    log:
        "results/logs/{run_id}/stage6_bioemu.log".format(run_id=RUN_ID)
    conda:
        "../../envs/bioemu.yml"
    resources:
        gpu = 1
    params:
        struct_dir = str(S4_DIR),
        out_dir = str(S6_DIR),
        n_conf = config["bioemu"]["num_conformers"],
        k = config["bioemu"]["kmeans_k"],
        top_n = config["bioemu"]["ensemble_dock_top_n"]
    shell:
        r"""
        mkdir -p {params.out_dir:q}
        python scripts/stage6_bioemu.py \
            --boltz-report {input.boltz_report:q} \
            --struct-dir {params.struct_dir:q} \
            --out-dir {params.out_dir:q} \
            --top-n {params.top_n:q} \
            --num-conformers {params.n_conf:q} \
            --kmeans-k {params.k:q} \
            --out-manifest {output.cluster_manifest:q} > {log:q} 2>&1
        """


rule ensemble_dock:
    input:
        manifest = rules.bioemu_ensemble.output.cluster_manifest,
        ligand_sdf = rules.xtb_optimize.output.sdf
    output:
        consensus = S6_DIR / "ensemble_consensus.tsv"
    log:
        "results/logs/{run_id}/stage6_ensemble_dock.log".format(run_id=RUN_ID)
    conda:
        "../../envs/autodock_gpu.yml"
    resources:
        gpu = 1
    params:
        out_dir = str(S6_DIR)
    shell:
        r"""
        python scripts/stage6_ensemble_dock.py \
            --manifest {input.manifest:q} \
            --ligand-sdf {input.ligand_sdf:q} \
            --out-dir {params.out_dir:q} \
            --out-consensus {output.consensus:q} > {log:q} 2>&1
        """
