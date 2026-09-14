# =============================================================================
# Stage 2 — Early ADMET gate (HARD FILTER on skin sensitization)
# =============================================================================
# Input:   01_input/compound_canonical.sdf
# Output:  02_admet/admet_report.json
#          02_admet/skin_sens_decision.txt   (HALT | FLAG_HIGH | PASS)
# =============================================================================

S2_DIR = RUN_DIR / "02_admet"


rule admet_ai:
    input:
        sdf = rules.xtb_optimize.output.sdf
    output:
        json = S2_DIR / "admet_ai.json"
    log:
        "results/logs/{run_id}/stage2_admet_ai.log".format(run_id=RUN_ID)
    conda:
        "../../envs/dti.yml"
    params:
        allow_unavailable = (
            "--allow-unavailable"
            if config_bool(config.get("admet", {}).get("allow_admet_ai_unavailable"), False)
            else ""
        )
    shell:
        r"""
        mkdir -p $(dirname {output.json:q})
        python scripts/stage2_admet_ai.py \
            --in-sdf {input.sdf:q} \
            --out-json {output.json:q} \
            {params.allow_unavailable} > {log:q} 2>&1
        """


rule stoptox:
    input:
        sdf = rules.xtb_optimize.output.sdf
    output:
        json = S2_DIR / "stoptox.json"
    log:
        "results/logs/{run_id}/stage2_stoptox.log".format(run_id=RUN_ID)
    conda:
        "../../envs/dti.yml"
    params:
        allow_unavailable = (
            "--allow-unavailable"
            if config_bool(config.get("admet", {}).get("allow_skin_sens_unavailable"), False)
            else ""
        )
    shell:
        r"""
        python scripts/stage2_stoptox.py \
            --in-sdf {input.sdf:q} --out-json {output.json:q} \
            {params.allow_unavailable} > {log:q} 2>&1
        """


rule husspred:
    input:
        sdf = rules.xtb_optimize.output.sdf
    output:
        json = S2_DIR / "husspred.json"
    log:
        "results/logs/{run_id}/stage2_husspred.log".format(run_id=RUN_ID)
    conda:
        "../../envs/dti.yml"
    params:
        allow_unavailable = (
            "--allow-unavailable"
            if config_bool(config.get("admet", {}).get("allow_skin_sens_unavailable"), False)
            else ""
        )
    shell:
        r"""
        python scripts/stage2_husspred.py \
            --in-sdf {input.sdf:q} --out-json {output.json:q} \
            {params.allow_unavailable} > {log:q} 2>&1
        """


rule pred_skin:
    input:
        sdf = rules.xtb_optimize.output.sdf
    output:
        json = S2_DIR / "pred_skin.json"
    log:
        "results/logs/{run_id}/stage2_pred_skin.log".format(run_id=RUN_ID)
    conda:
        "../../envs/dti.yml"
    params:
        allow_unavailable = (
            "--allow-unavailable"
            if config_bool(config.get("admet", {}).get("allow_skin_sens_unavailable"), False)
            else ""
        )
    shell:
        r"""
        python scripts/stage2_pred_skin.py \
            --in-sdf {input.sdf:q} --out-json {output.json:q} \
            {params.allow_unavailable} > {log:q} 2>&1
        """


rule pains_brenk_filter:
    input:
        sdf = rules.xtb_optimize.output.sdf
    output:
        json = S2_DIR / "structural_alerts.json"
    log:
        "results/logs/{run_id}/stage2_pains.log".format(run_id=RUN_ID)
    conda:
        "../../envs/dti.yml"
    shell:
        r"""
        python scripts/stage2_pains_brenk.py \
            --in-sdf {input.sdf:q} --out-json {output.json:q} > {log:q} 2>&1
        """


rule skin_sens_consensus:
    input:
        admet     = rules.admet_ai.output.json,
        stoptox   = rules.stoptox.output.json,
        husspred  = rules.husspred.output.json,
        pred_skin = rules.pred_skin.output.json,
        alerts    = rules.pains_brenk_filter.output.json
    output:
        report    = S2_DIR / "admet_report.json",
        decision  = S2_DIR / "skin_sens_decision.txt"
    log:
        "results/logs/{run_id}/stage2_consensus.log".format(run_id=RUN_ID)
    conda:
        "../../envs/dti.yml"
    params:
        halt_min = config["admet"]["skin_sens_halt_min_votes"],
        allow_unavailable = (
            "--allow-unavailable-models"
            if config_bool(config.get("admet", {}).get("allow_skin_sens_unavailable"), False)
            else ""
        )
    shell:
        r"""
        python scripts/stage2_consensus.py \
            --admet-ai {input.admet:q} \
            --stoptox {input.stoptox:q} \
            --husspred {input.husspred:q} \
            --pred-skin {input.pred_skin:q} \
            --alerts {input.alerts:q} \
            --halt-min-votes {params.halt_min:q} \
            {params.allow_unavailable} \
            --out-report {output.report:q} \
            --out-decision {output.decision:q} > {log:q} 2>&1
        """
