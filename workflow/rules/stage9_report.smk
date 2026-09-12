# =============================================================================
# Stage 9 — Mol* integrated HTML report
# =============================================================================
# v2 panels added:
#   Panel A2 — AutoDock-GPU 전수 distribution histogram + top-50 highlight
#   Panel A3 — DTI vs Docking disagreement scatter (RRF rank vs PSICHIC rank)
# =============================================================================

import shlex

S9_DIR = RUN_DIR / "09_report"
S9_FAST_DIR = RUN_DIR / "reports" / "fast"
S9_PHYSICS_DIR = RUN_DIR / "reports" / "physics"


def _physics_report_inputs() -> list[str]:
    if MODE in ("comprehensive", "both"):
        return [
            str(rules.boltz2_cofold.output.report),
            str(rules.ensemble_dock.output.consensus),
            str(rules.mmgbsa.output.report),
            str(rules.dft_pyscf.output.report),
        ]
    return []


def _physics_report_args(wildcards, input) -> str:
    reports = list(input.physics_reports)
    if not reports:
        return ""
    boltz, ensemble, mmgbsa, dft = reports
    return (
        f"--boltz-report {shlex.quote(str(boltz))} "
        f"--ensemble {shlex.quote(str(ensemble))} "
        f"--mmgbsa {shlex.quote(str(mmgbsa))} "
        f"--dft {shlex.quote(str(dft))}"
    )


def _stage3_top_input() -> str:
    return str(rules.kg_efficacy_label.output.csv)


def _disagreement_inputs() -> list[str]:
    if MODE in ("comprehensive", "both") and RUN_DTI:
        return [str(S3C_DIR / "disagreement_analysis.json")]
    return []


def _screening_count_inputs() -> list[str]:
    inputs = []
    if MODE in ("fast", "both"):
        inputs.extend([
            str(rules.daina_zoete.output.scores),
            str(rules.dti_rrf_fast.output.selected),
            str(rules.fast_autogrid_maps.output.manifest),
            str(rules.autodock_top5k.output.scores),
            str(rules.fast_gnina_pose_rescore.output.scores),
            str(rules.fast_rerank_consensus.output.canonical),
            str(rules.fast_rerank_consensus.output.top50),
        ])
    if MODE in ("comprehensive", "both"):
        inputs.extend([
            str(rules.autodock_gpu_all.output.scores),
            str(rules.autodock_pick_top_pct.output.top),
            str(rules.gnina_rescore_top.output.scores),
            str(rules.rtmscore_top.output.scores),
            str(rules.boltz2_affinity_top.output.scores),
            str(rules.rrf_4way_consensus.output.top50),
        ])
    return inputs


rule molstar_report:
    input:
        compound_meta = rules.xtb_optimize.output.meta,
        admet_report  = rules.skin_sens_consensus.output.report,
        skin_sens_decision = rules.skin_sens_consensus.output.decision,
        admet_ai      = rules.admet_ai.output.json,
        structural_alerts = rules.pains_brenk_filter.output.json,
        husspred      = rules.husspred.output.json,
        stoptox       = rules.stoptox.output.json,
        pred_skin     = rules.pred_skin.output.json,
        stage3_top    = _stage3_top_input(),
        cosing_json   = rules.cosing_match.output.json,
        drug_warnings = rules.drug_avoidance_match.output.json,
        cosmetic_decision = rules.cosmetic_drug_decision.output.decision,
        kg_efficacy   = rules.kg_efficacy_label.output.csv,
        autodock_full = (str(S3C_DIR / "autodock_all_targets.tsv")
                         if MODE in ("comprehensive", "both") else
                         str(S3F_DIR / "autodock_top5k.tsv")),
        screening_counts = _screening_count_inputs(),
        ligand_sdf    = rules.xtb_optimize.output.sdf,
        ligand_pdbqt  = rules.meeko_ligand.output.pdbqt,
        pose_dir      = (rules.autodock_gpu_all.output.poses
                         if MODE in ("comprehensive", "both") else
                         rules.autodock_top5k.output.poses),
        physics_reports = _physics_report_inputs(),
        disagreement  = _disagreement_inputs()
    output:
        html = S9_DIR / "index.html",
        fast_manifest = S9_DIR / "fast_report_manifest.json",
        physics_manifest = S9_DIR / "physics_report_manifest.json"
    log:
        "results/logs/{run_id}/stage9_report.log".format(run_id=RUN_ID)
    conda:
        "../../envs/viz.yml"
    params:
        run_id = RUN_ID,
        mode = MODE,
        disagreement = lambda wildcards, input: (
            str(input.disagreement[0]) if input.disagreement else ""
        ),
        run_dir = str(RUN_DIR),
        out_dir = str(S9_DIR),
        fast_report_root = str(S9_FAST_DIR),
        physics_report_root = str(S9_PHYSICS_DIR),
        receptor_dir = str(AF_CLEAN),
        physics_report_args = _physics_report_args,
        target_metadata = str(config.get("target_metadata", "data/hpa/proteinatlas.tsv")),
        allow_degraded_safety = (
            "--allow-degraded-safety"
            if (
                config_bool(config.get("admet", {}).get("allow_admet_ai_unavailable"), False)
                or config_bool(config.get("admet", {}).get("allow_skin_sens_unavailable"), False)
            )
            else ""
        )
    shell:
        r"""
        mkdir -p {params.out_dir:q}
        python scripts/stage9_report.py \
            --run-id {params.run_id:q} \
            --mode {params.mode:q} \
            --compound-meta {input.compound_meta:q} \
            --admet-report {input.admet_report:q} \
            --skin-sens-decision {input.skin_sens_decision:q} \
            --admet-ai-json {input.admet_ai:q} \
            --structural-alerts-json {input.structural_alerts:q} \
            --husspred-json {input.husspred:q} \
            --stoptox-json {input.stoptox:q} \
            --pred-skin-json {input.pred_skin:q} \
            {params.allow_degraded_safety} \
            --stage3-top {input.stage3_top:q} \
            --cosmetic-drug-decision {input.cosmetic_decision:q} \
            --autodock-full {input.autodock_full:q} \
            --ligand-sdf {input.ligand_sdf:q} \
            --ligand-pdbqt {input.ligand_pdbqt:q} \
            --pose-dir {input.pose_dir:q} \
            --receptor-dir {params.receptor_dir:q} \
            --fast-report-root {params.fast_report_root:q} \
            --physics-report-root {params.physics_report_root:q} \
            --fast-manifest-output {output.fast_manifest:q} \
            --physics-manifest-output {output.physics_manifest:q} \
            {params.physics_report_args} \
            --disagreement {params.disagreement:q} \
            --target-metadata {params.target_metadata:q} \
            --cosing-json {input.cosing_json:q} \
            --drug-warnings-json {input.drug_warnings:q} \
            --kg-efficacy-csv {input.kg_efficacy:q} \
            --run-dir {params.run_dir:q} \
            --out-html {output.html:q} > {log:q} 2>&1
        """
