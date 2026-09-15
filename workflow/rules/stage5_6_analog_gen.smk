# =============================================================================
# Stage 5.6 — REINVENT 4 manifest-driven generation
# =============================================================================
# The external REINVENT4 executable and all model/plugin assets are operator
# supplied and hash-pinned in config.  The adapter never guesses CLI flags.
# =============================================================================

S56_DIR = RUN_DIR / "05_6_analogs"
S56_FINETUNE = ANALOG_GEN.get("finetune", {})
S56_FINETUNE_ENABLED = config_bool(S56_FINETUNE.get("enabled"), False)

# Keep the rule parseable when the opt-in path is disabled.  These sentinel
# inputs are never scheduled unless the corresponding config gate is enabled.
S56_FINETUNE_TRAIN = Path(
    S56_FINETUNE.get("train_smiles")
    or "data/REINVENT4_FINETUNE_TRAIN_SMILES_REQUIRED.smi"
)
S56_FINETUNE_HOLDOUT = Path(
    S56_FINETUNE.get("holdout_smiles")
    or "data/REINVENT4_FINETUNE_HOLDOUT_SMILES_REQUIRED.smi"
)
S56_FINETUNE_PRIOR = Path(
    S56_FINETUNE.get("prior")
    or "data/REINVENT4_FINETUNE_PRIOR_REQUIRED.prior"
)
S56_FINETUNE_MANIFEST = S56_DIR / "reinvent4_finetune_manifest.json"


rule reinvent4_finetune:
    input:
        train = S56_FINETUNE_TRAIN,
        holdout = S56_FINETUNE_HOLDOUT,
        prior = S56_FINETUNE_PRIOR
    output:
        manifest = S56_FINETUNE_MANIFEST
    log:
        "results/logs/{run_id}/stage5_6_finetune.log".format(run_id=RUN_ID)
    conda:
        "../../envs/dti.yml"
    resources:
        gpu = 1
    params:
        agent = str(S56_FINETUNE.get("agent", "")),
        staged_config = str(S56_FINETUNE.get("staged_config", "")),
        plugin_manifest = str(S56_FINETUNE.get("plugin_manifest", "")),
        out_dir = str(S56_DIR / "finetune"),
        command_template = str(S56_FINETUNE.get("command_template", "")),
        seed = ANALOG_GEN.get("reinvent_seed", 49242),
        allow_empty = (
            "--allow-empty-output"
            if config_bool(S56_FINETUNE.get("allow_empty_output"), False)
            else ""
        )
    shell:
        r"""
        mkdir -p {params.out_dir:q}
        python scripts/stage5_6_finetune.py \
            --train-smi {input.train:q} \
            --holdout-smi {input.holdout:q} \
            --prior {input.prior:q} \
            --agent {params.agent:q} \
            --staged-config {params.staged_config:q} \
            --plugin-manifest {params.plugin_manifest:q} \
            --out-dir {params.out_dir:q} \
            --out-manifest {output.manifest:q} \
            --command-template {params.command_template:q} \
            --seed {params.seed:q} \
            {params.allow_empty} > {log:q} 2>&1
        """


if S56_FINETUNE_ENABLED:
    S56_MODEL_MANIFEST = rules.reinvent4_finetune.output.manifest
else:
    S56_MODEL_MANIFEST = Path(
        ANALOG_GEN.get("reinvent_model_manifest")
        or "data/REINVENT4_MODEL_MANIFEST_REQUIRED.json"
    )


rule reinvent4_contract_gate:
    input:
        sdf       = rules.xtb_optimize.output.sdf,
        consensus = rules.pharmacophore_consensus.output.json,
        anchors   = rules.pharmacophore_anchor_map.output.json,
        boltz     = rules.boltz2_cofold.output.report,
        model_manifest = S56_MODEL_MANIFEST
    output:
        smi  = S56_DIR / "all_generated.smi",
        lineage_csv = S56_DIR / "reinvent4_lineage.csv",
        lineage_json = S56_DIR / "reinvent4_lineage.json",
        status = S56_DIR / "reinvent4_status.json",
        stdout = S56_DIR / "reinvent4.stdout.txt",
        stderr = S56_DIR / "reinvent4.stderr.txt"
    log:
        "results/logs/{run_id}/stage5_6_reinvent.log".format(run_id=RUN_ID)
    conda:
        "../../envs/dti.yml"
    resources:
        gpu = 1
    params:
        out_dir = str(S56_DIR),
        executable = str(ANALOG_GEN.get("reinvent_executable", "")),
        reinvent_config = str(ANALOG_GEN.get("reinvent_config", "")),
        plugin_manifest = str(ANALOG_GEN.get("reinvent_plugin_manifest", "")),
        seed = ANALOG_GEN.get("reinvent_seed", 49242),
        timeout = ANALOG_GEN.get("reinvent_timeout_seconds", 3600),
        allow_empty = (
            "--allow-empty-output"
            if config_bool(config.get("analog_gen", {}).get("allow_empty_reinvent_output"), False)
            else ""
        )
    shell:
        r"""
        mkdir -p {params.out_dir:q}
        python scripts/stage5_6_reinvent.py \
            --in-sdf {input.sdf:q} \
            --consensus {input.consensus:q} \
            --interaction-anchors {input.anchors:q} \
            --boltz-report {input.boltz:q} \
            --model-manifest {input.model_manifest:q} \
            --config {params.reinvent_config:q} \
            --plugin-manifest {params.plugin_manifest:q} \
            --executable {params.executable:q} \
            --seed {params.seed:q} \
            --timeout-seconds {params.timeout:q} \
            --out-status {output.status:q} \
            --out-smi {output.smi:q} \
            --out-lineage-csv {output.lineage_csv:q} \
            --out-lineage-json {output.lineage_json:q} \
            --out-stdout {output.stdout:q} \
            --out-stderr {output.stderr:q} \
            {params.allow_empty} > {log:q} 2>&1
        """


rule mini_validate_funnel:
    input:
        smi       = rules.reinvent4_contract_gate.output.smi,
        cosing    = COSING_DIR / "cosing.parquet",
        drugs     = DRUG_DIR / "drugs.parquet"
    output:
        sdf     = S56_DIR / "top30_analogs.sdf",
        csv     = S56_DIR / "top30_with_properties.csv",
        lineage = S56_DIR / "analog_lineage.json",
        funnel  = S56_DIR / "stage_filters_summary.tsv"
    log:
        "results/logs/{run_id}/stage5_6_validate.log".format(run_id=RUN_ID)
    conda:
        "../../envs/dti.yml"
    resources:
        gpu = 1
    params:
        admet  = config["analog_gen"]["funnel_admet"],
        route_proxy = config["analog_gen"]["funnel_aizynth"],
        drug    = config["analog_gen"]["funnel_drug"],
        boltz   = config["analog_gen"]["funnel_boltz"]
    shell:
        r"""
        python scripts/stage5_6_mini_validate.py \
            --in-smi {input.smi:q} \
            --cosing-parquet {input.cosing:q} \
            --drugs-parquet {input.drugs:q} \
            --keep-admet {params.admet:q} \
            --keep-route-proxy {params.route_proxy:q} \
            --keep-drug {params.drug:q} \
            --keep-boltz {params.boltz:q} \
            --out-sdf {output.sdf:q} \
            --out-csv {output.csv:q} \
            --out-lineage {output.lineage:q} \
            --out-funnel {output.funnel:q} > {log:q} 2>&1
        """
