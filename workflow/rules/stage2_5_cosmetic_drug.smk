# =============================================================================
# Stage 2.5 — Cosmetic Annotation + Drug Avoidance (v3)
# =============================================================================
# Output:
#   02b_cosmetic_drug/cosing_match.json
#                  /drug_warnings.json
#                  /cosmetic_drug_decision.txt   (HALT|DOWNWEIGHT|PROCEED)
# =============================================================================

S25_DIR = RUN_DIR / "02b_cosmetic_drug"


rule cosing_match:
    input:
        sdf = rules.xtb_optimize.output.sdf,
        cosing = COSING_DIR / "cosing.parquet"
    output:
        json = S25_DIR / "cosing_match.json"
    log:
        "results/logs/{run_id}/stage2_5_cosing.log".format(run_id=RUN_ID)
    conda:
        "../../envs/dti.yml"
    params:
        exact = config["cosing"]["exact_threshold"],
        sim   = config["cosing"]["similar_threshold"],
        ana   = config["cosing"]["analog_threshold"],
        allow_missing = (
            "--allow-missing-reference"
            if config_bool(config.get("cosing", {}).get("allow_missing_reference"), False)
            else ""
        )
    shell:
        r"""
        mkdir -p $(dirname {output.json:q})
        python scripts/stage2_5_cosing_match.py \
            --in-sdf {input.sdf:q} \
            --cosing-parquet {input.cosing:q} \
            --similar-threshold {params.sim:q} \
            --analog-threshold {params.ana:q} \
            --out-json {output.json:q} \
            {params.allow_missing} > {log:q} 2>&1
        """


rule drug_avoidance_match:
    input:
        sdf = rules.xtb_optimize.output.sdf,
        drugs = DRUG_DIR / "drugs.parquet",
        scaffolds = DRUG_DIR / "scaffolds.parquet"
    output:
        json = S25_DIR / "drug_warnings.json"
    log:
        "results/logs/{run_id}/stage2_5_drug_avoidance.log".format(run_id=RUN_ID)
    conda:
        "../../envs/dti.yml"
    params:
        strict = config["drug_avoidance"]["strict_threshold"],
        soft   = config["drug_avoidance"]["soft_threshold"],
        allow_missing = (
            "--allow-missing-reference"
            if config_bool(config.get("drug_avoidance", {}).get("allow_missing_reference"), False)
            else ""
        )
    shell:
        r"""
        python scripts/stage2_5_drug_avoidance.py \
            --in-sdf {input.sdf:q} \
            --drugs-parquet {input.drugs:q} \
            --scaffolds-parquet {input.scaffolds:q} \
            --strict-threshold {params.strict:q} \
            --soft-threshold {params.soft:q} \
            --out-json {output.json:q} \
            {params.allow_missing} > {log:q} 2>&1
        """


rule cosmetic_drug_decision:
    input:
        cosing = rules.cosing_match.output.json,
        drugs = rules.drug_avoidance_match.output.json
    output:
        decision = S25_DIR / "cosmetic_drug_decision.txt"
    log:
        "results/logs/{run_id}/stage2_5_decision.log".format(run_id=RUN_ID)
    conda:
        "../../envs/dti.yml"
    params:
        policy = config["drug_avoidance"]["drug_policy"],
        allow_degraded = (
            "--allow-degraded-references"
            if (
                config_bool(config.get("cosing", {}).get("allow_missing_reference"), False)
                or config_bool(config.get("drug_avoidance", {}).get("allow_missing_reference"), False)
            )
            else ""
        )
    shell:
        r"""
        python scripts/stage2_5_decision.py \
            --cosing-json {input.cosing:q} \
            --drug-json {input.drugs:q} \
            --drug-policy {params.policy:q} \
            --out-decision {output.decision:q} \
            {params.allow_degraded} > {log:q} 2>&1
        """
