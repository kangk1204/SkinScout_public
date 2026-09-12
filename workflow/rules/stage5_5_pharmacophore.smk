# =============================================================================
# Stage 5.5 — Pose-supported interaction atom identification (v3)
# =============================================================================
# 2-source consensus: PLIP + ProLIF bound-pose interaction atoms.
# An atom is confirmed when both executable evidence sources flag it.
# =============================================================================

S55_DIR = RUN_DIR / "05_pharmacophore"


rule plip_run:
    input:
        boltz = rules.boltz2_cofold.output.report
    output:
        xml = S55_DIR / "plip_interactions.xml"
    log:
        "results/logs/{run_id}/stage5_5_plip.log".format(run_id=RUN_ID)
    conda:
        "../../envs/viz.yml"
    params:
        out_dir = str(S55_DIR),
        boltz_dir = str(S5_DIR),
        allow_empty = (
            "--allow-empty"
            if config_bool(config.get("pharmacophore", {}).get("allow_empty_interaction_sources"), False)
            else ""
        )
    shell:
        r"""
        mkdir -p {params.out_dir:q}
        python scripts/stage5_5_plip.py \
            --boltz-dir {params.boltz_dir:q} \
            --boltz-report {input.boltz:q} \
            --out-xml {output.xml:q} \
            {params.allow_empty} > {log:q} 2>&1
        """


rule prolif_fingerprint:
    input:
        boltz = rules.boltz2_cofold.output.report
    output:
        csv = S55_DIR / "prolif_fingerprint.csv"
    log:
        "results/logs/{run_id}/stage5_5_prolif.log".format(run_id=RUN_ID)
    conda:
        "../../envs/viz.yml"
    params:
        out_dir = str(S55_DIR),
        boltz_dir = str(S5_DIR),
        allow_empty = (
            "--allow-empty"
            if config_bool(config.get("pharmacophore", {}).get("allow_empty_interaction_sources"), False)
            else ""
        )
    shell:
        r"""
        python scripts/stage5_5_prolif.py \
            --boltz-dir {params.boltz_dir:q} \
            --boltz-report {input.boltz:q} \
            --out-csv {output.csv:q} \
            {params.allow_empty} > {log:q} 2>&1
        """


rule pharmacophore_consensus:
    input:
        plip = rules.plip_run.output.xml,
        prolif = rules.prolif_fingerprint.output.csv
    output:
        json = S55_DIR / "consensus_pharmacophore_atoms.json"
    log:
        "results/logs/{run_id}/stage5_5_consensus.log".format(run_id=RUN_ID)
    conda:
        "../../envs/viz.yml"
    params:
        min_votes = config["pharmacophore"]["consensus_min_votes"],
        allow_empty = (
            "--allow-empty-sources"
            if config_bool(config.get("pharmacophore", {}).get("allow_empty_interaction_sources"), False)
            else ""
        )
    shell:
        r"""
        python scripts/stage5_5_consensus.py \
            --plip {input.plip:q} \
            --prolif {input.prolif:q} \
            --min-votes {params.min_votes:q} \
            --out-json {output.json:q} \
            {params.allow_empty} > {log:q} 2>&1
        """


rule pharmacophore_anchor_map:
    input:
        parent_sdf = rules.xtb_optimize.output.sdf,
        boltz = rules.boltz2_cofold.output.report,
        consensus = rules.pharmacophore_consensus.output.json
    output:
        json = S55_DIR / "interaction_anchor_map.json"
    log:
        "results/logs/{run_id}/stage5_5_atom_map.log".format(run_id=RUN_ID)
    conda:
        "../../envs/viz.yml"
    shell:
        r"""
        python scripts/stage5_5_atom_map.py \
            --parent-sdf {input.parent_sdf:q} \
            --boltz-report {input.boltz:q} \
            --consensus-json {input.consensus:q} \
            --out-json {output.json:q} > {log:q} 2>&1
        """
