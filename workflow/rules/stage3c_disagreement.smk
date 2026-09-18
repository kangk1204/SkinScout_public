# =============================================================================
# Stage 3c — DTI vs Docking disagreement analysis (sanity check in COMPREHENSIVE,
# decisive in BOTH-mode methodology runs).
# =============================================================================
# Output: 03_targets/mode_comprehensive/{psichic_sanity.tsv,
#                                        disagreement_analysis.json}
# =============================================================================


rule psichic_sanity:
    input:
        sdf = rules.xtb_optimize.output.sdf,
        decision = rules.skin_sens_consensus.output.decision,
        cosmetic_decision = rules.cosmetic_drug_decision.output.decision,
        activity_retrieval_gate = str(
            MANIFESTS / "activity_retrieval_operational_gate.flag"
        ),
        canonical_fasta = str(CANONICAL_HUMAN_FASTA)
    output:
        scores = S3C_DIR / "psichic_sanity.tsv"
    log:
        "results/logs/{run_id}/stage3_psichic_sanity.log".format(run_id=RUN_ID)
    conda:
        "../../envs/boltz2.yml"
    resources:
        gpu = 1
    params:
        clean_dir = str(AF_CLEAN),
        batch_size = config["docking"]["psichic_batch_size"],
        score_batch_size = config["docking"]["psichic_score_batch_size"],
        esm_short_batch_size = config["docking"]["psichic_esm_short_batch_size"],
        esm_medium_batch_size = config["docking"]["psichic_esm_medium_batch_size"],
        esm_long_batch_size = config["docking"]["psichic_esm_long_batch_size"],
        max_sequence_length = config["docking"]["psichic_max_sequence_length"],
        allow_unavailable = (
            "--allow-unavailable"
            if config_bool(config.get("docking", {}).get("allow_psichic_unavailable"), False)
            else ""
        )
    shell:
        r"""
        python scripts/validate_activity_retrieval_gate.py check-operational \
            --gate {input.activity_retrieval_gate:q}
        decision=$(head -n 1 {input.decision:q})
        if [ "$decision" = "HALT" ]; then
            echo "[stage3] skin-sens HALT — refusing DTI sanity." >&2
            exit 10
        fi
        cosmetic_decision=$(head -n 1 {input.cosmetic_decision:q})
        if [ "$cosmetic_decision" = "HALT" ]; then
            echo "[stage3] cosmetic/drug HALT — refusing DTI sanity." >&2
            exit 11
        fi
        PSICHIC_ESM_SHORT_BATCH={params.esm_short_batch_size:q} \
        PSICHIC_ESM_MEDIUM_BATCH={params.esm_medium_batch_size:q} \
        PSICHIC_ESM_LONG_BATCH={params.esm_long_batch_size:q} \
        python scripts/stage3_psichic.py \
            --ligand-sdf {input.sdf:q} \
            --clean-dir {params.clean_dir:q} \
            --sequence-fasta {input.canonical_fasta:q} \
            --batch-size {params.batch_size:q} \
            --score-batch-size {params.score_batch_size:q} \
            --max-sequence-length {params.max_sequence_length:q} \
            --out-scores {output.scores:q} \
            {params.allow_unavailable} > {log:q} 2>&1
        """


rule disagreement_analysis:
    input:
        docking = rules.rrf_4way_consensus.output.top50,
        autodock_full = rules.autodock_gpu_all.output.scores,
        psichic = rules.psichic_sanity.output.scores
    output:
        json = S3C_DIR / "disagreement_analysis.json"
    log:
        "results/logs/{run_id}/stage3_disagreement.log".format(run_id=RUN_ID)
    conda:
        "../../envs/base.yml"
    params:
        chembl = str(CHEMBL)
    shell:
        r"""
        python scripts/stage3_disagreement.py \
            --docking-top {input.docking:q} \
            --autodock-full {input.autodock_full:q} \
            --psichic {input.psichic:q} \
            --chembl-dir {params.chembl:q} \
            --out-json {output.json:q} > {log:q} 2>&1
        """
