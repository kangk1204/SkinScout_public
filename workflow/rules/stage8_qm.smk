# =============================================================================
# Stage 8 — QM / QM-MM (final 1-3 candidates)
# =============================================================================

S8_DIR = RUN_DIR / "08_qm"


rule crest_conformer:
    input:
        mmgbsa = rules.mmgbsa.output.report,
        ligand_sdf = rules.xtb_optimize.output.sdf
    output:
        manifest = S8_DIR / "crest_manifest.tsv"
    log:
        "results/logs/{run_id}/stage8_crest.log".format(run_id=RUN_ID)
    conda:
        "../../envs/qm.yml"
    params:
        out_dir = str(S8_DIR),
        top_n   = config["qm"]["top_n_for_qm"]
    shell:
        r"""
        mkdir -p {params.out_dir:q}
        python scripts/stage8_crest.py \
            --mmgbsa-report {input.mmgbsa:q} \
            --ligand-sdf {input.ligand_sdf:q} \
            --out-dir {params.out_dir:q} \
            --top-n {params.top_n:q} \
            --out-manifest {output.manifest:q} > {log:q} 2>&1
        """


rule xtb_binding_site_cluster:
    input:
        manifest = rules.crest_conformer.output.manifest
    output:
        cluster_manifest = S8_DIR / "xtb_clusters.tsv"
    log:
        "results/logs/{run_id}/stage8_xtb_cluster.log".format(run_id=RUN_ID)
    conda:
        "../../envs/qm.yml"
    params:
        out_dir = str(S8_DIR),
        gfn     = config["qm"]["semi_empirical"]
    shell:
        r"""
        python scripts/stage8_xtb_cluster.py \
            --crest-manifest {input.manifest:q} \
            --out-dir {params.out_dir:q} \
            --gfn {params.gfn:q} \
            --out-cluster-manifest {output.cluster_manifest:q} > {log:q} 2>&1
        """


rule dft_pyscf:
    input:
        cluster_manifest = rules.xtb_binding_site_cluster.output.cluster_manifest
    output:
        report = S8_DIR / "dft_report.tsv"
    log:
        "results/logs/{run_id}/stage8_dft.log".format(run_id=RUN_ID)
    conda:
        "../../envs/qm.yml"
    resources:
        gpu = 1
    params:
        out_dir   = str(S8_DIR),
        func      = config["qm"]["dft_functional"],
        basis     = config["qm"]["dft_basis"]
    shell:
        r"""
        python scripts/stage8_dft.py \
            --cluster-manifest {input.cluster_manifest:q} \
            --out-dir {params.out_dir:q} \
            --functional {params.func:q} \
            --basis {params.basis:q} \
            --out-report {output.report:q} > {log:q} 2>&1
        """
