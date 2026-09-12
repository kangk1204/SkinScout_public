# =============================================================================

import shlex
# Stage 7.5 — AiZynthFinder retrosynthesis (v3)
# =============================================================================
# Per analog from Stage 5.6 (or per top-N candidate from Stage 7), produce a
# retrosynthesis route + summary. Opt-in via config["retrosynthesis"]["enabled"].
# =============================================================================

S75_DIR = RUN_DIR / "07_5_retrosynthesis"


def _retro_optional_flag(flag: str, value: str) -> str:
    value = str(value or "").strip()
    return f"{flag} {shlex.quote(value)}" if value else ""


rule aizynth_run:
    input:
        sdf = rules.mini_validate_funnel.output.sdf
    output:
        manifest = S75_DIR / "routes_manifest.tsv"
    log:
        "results/logs/{run_id}/stage7_5_aizynth.log".format(run_id=RUN_ID)
    conda:
        "../../envs/dti.yml"
    params:
        out_dir = str(S75_DIR),
        iters = config["retrosynthesis"]["iteration_limit"],
        time_limit = config["retrosynthesis"]["time_limit_s"],
        max_xform = config["retrosynthesis"]["max_transforms"],
        executable = config["retrosynthesis"].get("executable", "aizynthcli"),
        config_flag = lambda wildcards, input: _retro_optional_flag(
            "--config", config["retrosynthesis"].get("config", "")
        ),
        model_manifest_flag = lambda wildcards, input: _retro_optional_flag(
            "--model-manifest", config["retrosynthesis"].get("model_manifest", "")
        ),
        stock_manifest_flag = lambda wildcards, input: _retro_optional_flag(
            "--stock-manifest", config["retrosynthesis"].get("stock_manifest", "")
        )
    shell:
        r"""
        mkdir -p {params.out_dir:q}
        python scripts/stage7_5_aizynth.py \
            --in-sdf {input.sdf:q} \
            --out-dir {params.out_dir:q} \
            --executable {params.executable:q} \
            --iteration-limit {params.iters:q} \
            --time-limit {params.time_limit:q} \
            --max-transforms {params.max_xform:q} \
            {params.config_flag} \
            {params.model_manifest_flag} \
            {params.stock_manifest_flag} \
            --out-manifest {output.manifest:q} > {log:q} 2>&1
        """


rule score_routes:
    input:
        manifest = rules.aizynth_run.output.manifest
    output:
        ranking = S75_DIR / "synthesis_priority_ranking.csv"
    log:
        "results/logs/{run_id}/stage7_5_score.log".format(run_id=RUN_ID)
    conda:
        "../../envs/dti.yml"
    shell:
        r"""
        python scripts/stage7_5_score.py \
            --manifest {input.manifest:q} \
            --out-ranking {output.ranking:q} > {log:q} 2>&1
        """
