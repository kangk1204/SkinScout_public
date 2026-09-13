# =============================================================================
# Stage 3 v3 — Skin-expression weighting + KG efficacy label
# =============================================================================
# Consumes the v2 top-50 CSV emitted by stage3a / stage3b and rewrites it as
# ranked_targets_v3.csv with the final skin-weighted score plus per-target
# efficacy categories pulled from the skin-efficacy KG.
# =============================================================================


def _known_target_prior_path() -> str:
    return str(RUN_DIR / "03_targets" / "known_target_priors.csv")


rule known_target_prior:
    input:
        compound = S1_DIR / "compound_canonical.json"
    output:
        csv = _known_target_prior_path()
    log:
        "results/logs/{run_id}/stage3_known_target_prior.log".format(run_id=RUN_ID)
    conda:
        "../../envs/base.yml"
    params:
        enabled = KNOWN_TARGET_PRIORS["enabled"],
        source_csv = KNOWN_TARGET_PRIORS["source_csv"],
        supplemental_arg = (
            "--supplemental-panels "
            + shlex.quote(",".join(KNOWN_TARGET_PRIORS["supplemental_source_csv"]))
            if KNOWN_TARGET_PRIORS["supplemental_source_csv"]
            else ""
        ),
        similarity_enabled = KNOWN_TARGET_PRIORS["similarity_enabled"],
        similarity_threshold = KNOWN_TARGET_PRIORS["similarity_threshold"],
        morgan_radius = KNOWN_TARGET_PRIORS["morgan_radius"],
        morgan_n_bits = KNOWN_TARGET_PRIORS["morgan_n_bits"],
        prior_combination = KNOWN_TARGET_PRIORS["prior_combination_strategy"],
        similarity_score_power = KNOWN_TARGET_PRIORS["similarity_score_power"],
        max_similarity_candidates = KNOWN_TARGET_PRIORS["max_similarity_candidates"],
    shell:
        r"""
        python scripts/stage3_known_target_prior.py \
            --compound-json {input.compound:q} \
            --known-panel {params.source_csv:q} \
            {params.supplemental_arg} \
            --out-csv {output.csv:q} \
            --enabled {params.enabled:q} \
            --similarity-enabled {params.similarity_enabled:q} \
            --similarity-threshold {params.similarity_threshold:q} \
            --prior-combination {params.prior_combination:q} \
            --similarity-score-power {params.similarity_score_power:q} \
            --morgan-radius {params.morgan_radius:q} \
            --morgan-n-bits {params.morgan_n_bits:q} \
            --max-similarity-candidates {params.max_similarity_candidates:q} \
            > {log:q} 2>&1
        """


def _stage3_v3_input() -> str:
    if MODE == "fast":
        # The measured policy keeps Daina ranks 1-10 fixed and uses docking
        # evidence only within ranks 11-50.  This reader-facing ordering retains
        # daina_rank and the original score columns for provenance. Skin/KG
        # stages annotate this ordering; they do not restore Daina's old order.
        return str(S3F_DIR / "top50_band_reranked.csv")
    return str(S3C_DIR / "top50_4way_consensus.csv")


rule skin_weight_apply:
    input:
        top = _stage3_v3_input(),
        skin = SKIN_EXPR_DIR / "skin_score.tsv",
        known_target_priors = rules.known_target_prior.output.csv,
        cosmetic = rules.cosmetic_drug_decision.output.decision,
        drugs = rules.drug_avoidance_match.output.json
    output:
        csv = RUN_DIR / "03_targets" / "ranked_targets_v3.csv"
    log:
        "results/logs/{run_id}/stage3_skin_weight.log".format(run_id=RUN_ID)
    conda:
        "../../envs/base.yml"
    params:
        skin_weight = config["skin_weight"],
        known_target_prior_weight = KNOWN_TARGET_PRIORS["weight"],
        known_target_prior_min_score = KNOWN_TARGET_PRIORS["min_score"],
        known_target_prior_power = KNOWN_TARGET_PRIORS["power"],
        min_thresh  = config["skin_min_threshold"],
        cosmetic_factor = config["drug_avoidance"]["downweight_factor"],
        min_source_count = (
            1
            if MODE == "fast"
            else config["docking"]["min_sources_comprehensive"]
        ),
        preserve_primary = "--preserve-primary-ranking" if MODE == "fast" else ""
    shell:
        r"""
        python scripts/stage3_skin_weighting.py \
            --top-csv {input.top:q} \
            --skin-tsv {input.skin:q} \
            --known-target-prior-csv {input.known_target_priors:q} \
            --cosmetic-decision {input.cosmetic:q} \
            --drug-json {input.drugs:q} \
            --cosmetic-downweight-factor {params.cosmetic_factor:q} \
            --skin-weight {params.skin_weight:q} \
            --known-target-prior-weight {params.known_target_prior_weight:q} \
            --known-target-prior-min-score {params.known_target_prior_min_score:q} \
            --known-target-prior-power {params.known_target_prior_power:q} \
            --min-threshold {params.min_thresh:q} \
            --min-source-count {params.min_source_count:q} \
            {params.preserve_primary} \
            --out-csv {output.csv:q} > {log:q} 2>&1
        """


rule kg_efficacy_label:
    input:
        ranked = rules.skin_weight_apply.output.csv,
        kg = KG_DIR / "skin_efficacy.graphml"
    output:
        csv = RUN_DIR / "03_targets" / "ranked_targets_v3_with_efficacy.csv"
    log:
        "results/logs/{run_id}/stage3_kg_efficacy.log".format(run_id=RUN_ID)
    conda:
        "../../envs/base.yml"
    params:
        allow_missing = "--allow-missing-efficacy" if MODE == "fast" else ""
    shell:
        r"""
        python scripts/stage3_kg_efficacy_label.py \
            --ranked-csv {input.ranked:q} \
            --kg-graphml {input.kg:q} \
            {params.allow_missing} \
            --out-csv {output.csv:q} > {log:q} 2>&1
        """
