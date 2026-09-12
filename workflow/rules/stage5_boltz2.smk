# =============================================================================
# Stage 5 — Boltz-2 co-folding + affinity (top 50)
# =============================================================================

S5_DIR = RUN_DIR / "05_boltz2"


rule boltz2_cofold:
    input:
        manifest = rules.prepare_top50_structures.output.manifest,
        ligand_sdf = rules.xtb_optimize.output.sdf
    output:
        report = S5_DIR / "boltz2_report.tsv"
    log:
        "results/logs/{run_id}/stage5_boltz2.log".format(run_id=RUN_ID)
    conda:
        "../../envs/boltz2.yml"
    resources:
        gpu = 1
    params:
        struct_dir = str(S4_DIR),
        out_dir    = str(S5_DIR),
        num_seeds  = config["boltz2"]["num_seeds"],
        num_recyc  = config["boltz2"]["num_recycles"],
        iptm_thr   = config["boltz2"]["iptm_threshold"],
        plddt_thr  = config["boltz2"]["pocket_plddt_threshold"],
        max_res    = config["hardware"]["boltz2_max_residues"],
        crop_r     = config["hardware"]["pocket_crop_radius"]
    shell:
        r"""
        mkdir -p {params.out_dir:q}
        python scripts/stage5_boltz2.py \
            --manifest {input.manifest:q} \
            --ligand-sdf {input.ligand_sdf:q} \
            --struct-dir {params.struct_dir:q} \
            --out-dir {params.out_dir:q} \
            --num-seeds {params.num_seeds:q} \
            --num-recycles {params.num_recyc:q} \
            --iptm-threshold {params.iptm_thr:q} \
            --plddt-threshold {params.plddt_thr:q} \
            --max-residues {params.max_res:q} \
            --crop-radius {params.crop_r:q} \
            --out-report {output.report:q} > {log:q} 2>&1
        """
