# =============================================================================
# Stage 1 — Compound preprocessing
# =============================================================================
# Input:  config["compound_smiles"]  OR  config["compound_sdf"]
# Output: results/runs/<run_id>/01_input/{compound_canonical.sdf, .inchikey,
#                                          .json}
# Toolchain: RDKit → Dimorphite-DL → ETKDGv3 → MMFF94 → xTB GFN2.
# =============================================================================

S1_DIR = RUN_DIR / "01_input"

rule rdkit_standardize:
    output:
        sdf = S1_DIR / "standardized.sdf",
        meta = S1_DIR / "standardized.json"
    log:
        "results/logs/{run_id}/stage1_standardize.log".format(run_id=RUN_ID)
    conda:
        "../../envs/dti.yml"
    params:
        smiles_arg = (
            f"--smiles {shlex.quote(config['compound_smiles'])}"
            if config["compound_smiles"]
            else ""
        ),
        sdf_arg = (
            f"--sdf-in {shlex.quote(config['compound_sdf'])}"
            if config["compound_sdf"]
            else ""
        )
    shell:
        r"""
        mkdir -p $(dirname {output.sdf:q})
        cmd=(python scripts/stage1_standardize.py
            --out-sdf {output.sdf:q}
            --out-meta {output.meta:q}
            {params.smiles_arg}
            {params.sdf_arg})
        "${{cmd[@]}}" > {log:q} 2>&1
        """


rule dimorphite_protonate:
    input:
        sdf = rules.rdkit_standardize.output.sdf
    output:
        sdf = S1_DIR / "protonated.sdf"
    log:
        "results/logs/{run_id}/stage1_protonate.log".format(run_id=RUN_ID)
    conda:
        "../../envs/dti.yml"
    params:
        fallback = (
            "--allow-unprotonated-fallback"
            if config_bool(
                config.get("stage1", {}).get("allow_unprotonated_fallback"),
                False,
            )
            else ""
        )
    shell:
        r"""
        python scripts/stage1_protonate.py \
            --in-sdf {input.sdf:q} \
            --out-sdf {output.sdf:q} \
            --ph-min 7.2 --ph-max 7.6 \
            {params.fallback} > {log:q} 2>&1
        """


rule etkdgv3_conformer:
    input:
        sdf = rules.dimorphite_protonate.output.sdf
    output:
        sdf = S1_DIR / "conformers_etkdg.sdf"
    log:
        "results/logs/{run_id}/stage1_etkdg.log".format(run_id=RUN_ID)
    conda:
        "../../envs/dti.yml"
    shell:
        r"""
        python scripts/stage1_etkdg.py \
            --in-sdf {input.sdf:q} \
            --out-sdf {output.sdf:q} \
            --n-conf 50 --rmsd-prune 0.5 > {log:q} 2>&1
        """


rule xtb_optimize:
    input:
        sdf = rules.etkdgv3_conformer.output.sdf,
        meta = rules.rdkit_standardize.output.meta
    output:
        sdf      = S1_DIR / "compound_canonical.sdf",
        inchikey = S1_DIR / "compound_canonical.inchikey",
        meta     = S1_DIR / "compound_canonical.json"
    log:
        "results/logs/{run_id}/stage1_xtb.log".format(run_id=RUN_ID)
    conda:
        "../../envs/qm.yml"
    params:
        fallback = (
            "--allow-mmff-fallback"
            if config_bool(config.get("stage1", {}).get("allow_xtb_fallback"), False)
            else ""
        )
    shell:
        r"""
        python scripts/stage1_xtb_opt.py \
            --in-sdf {input.sdf:q} \
            --in-meta {input.meta:q} \
            --out-sdf {output.sdf:q} \
            --out-inchikey {output.inchikey:q} \
            --out-meta {output.meta:q} \
            --gfn 2 \
            {params.fallback} > {log:q} 2>&1
        """
