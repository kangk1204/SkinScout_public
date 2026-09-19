# =============================================================================

import shlex
# Stage 11 — Publication-ready output (v3)
# =============================================================================
# Emits figures, data-availability statement, reproducibility pack, and an
# optional Markdown manuscript skeleton.
# Opt-in via config["publication"]["build_publication_package"].
# =============================================================================

S11_DIR = RUN_DIR / "publication"
# Stage 11 evaluation inputs default to results/eval but can be redirected so
# the publication package binds the evaluation tree the run actually used.
S11_EVAL_DIR = PROJECT_ROOT / "results" / "eval"
S11_EVAL_DIR_OVERRIDE = str(
    PUBLICATION.get("evaluation_dir", "")
).strip()
if S11_EVAL_DIR_OVERRIDE:
    S11_EVAL_DIR = Path(S11_EVAL_DIR_OVERRIDE)
    if not S11_EVAL_DIR.is_absolute():
        S11_EVAL_DIR = PROJECT_ROOT / S11_EVAL_DIR
S11_ALLOW_PLACEHOLDERS = config_bool(config.get("publication", {}).get("allow_placeholders"), False)
S11_DRAFT_DOI_OK = config_bool(config.get("publication", {}).get("draft_doi_ok"), False)
S11_EMIT_MANUSCRIPT_DRAFT = config_bool(
    PUBLICATION.get("emit_manuscript_draft"), False
)
# A diagnostic bundle completes with claim_ready=false and never writes a
# manuscript; the claim-quality publication path keeps the strict gate.
S11_BUNDLE_MODE = config_choice(
    PUBLICATION.get("bundle_mode"),
    "publication.bundle_mode",
    {"claim_quality", "diagnostic"},
    "claim_quality",
)
S11_DIAGNOSTIC_BUNDLE = S11_BUNDLE_MODE == "diagnostic"
if S11_DIAGNOSTIC_BUNDLE and S11_EMIT_MANUSCRIPT_DRAFT:
    raise ValueError(
        "publication.emit_manuscript_draft cannot be true for a diagnostic "
        "publication bundle: diagnostic bundles never emit a manuscript"
    )
S11_DIRTY_SNAPSHOT_LIMITATION = str(
    config.get("publication", {}).get("dirty_snapshot_limitation", "")
).strip()
S11_REQUIRE_PHYSICS = MODE in ("comprehensive", "both")
S11_FIGURE_INPUTS = {} if S11_ALLOW_PLACEHOLDERS else {
    "report": rules.molstar_report.output.html,
    "target_landscape": rules.kg_efficacy_label.output.csv,
    "pharmacophore": rules.pharmacophore_consensus.output.json,
    "analogs": rules.mini_validate_funnel.output.csv,
    "md_trajectory": rules.gromacs_production.output.traj_index,
    "md_mmgbsa": rules.mmgbsa.output.report,
    "retrosynthesis": rules.score_routes.output.ranking,
    "eval_manifest": S11_EVAL_DIR / "iteration_manifest.json",
    "eval_retrospective": S11_EVAL_DIR / "cosmetic_retrospective.csv",
}
S11_FIGURE_OUTPUTS = {
    "fig01": S11_DIR / "figures" / "fig01_workflow.svg",
    "fig02": S11_DIR / "figures" / "fig02_target_landscape.png",
    "fig03": S11_DIR / "figures" / "fig03_pharmacophore.png",
    "fig04": S11_DIR / "figures" / "fig04_analog_scatter.png",
    "fig05": S11_DIR / "figures" / "fig05_md_rmsd.png",
    "fig06": S11_DIR / "figures" / "fig06_retrosynthesis.png",
    "fig07": S11_DIR / "figures" / "fig07_eval_bars.png",
    "fig08": S11_DIR / "figures" / "fig08_case_study.png",
}
S11_REPRO_OUTPUTS = {
    "config_hash": S11_DIR / "reproducibility" / "config_hash.txt",
    "git_commit": S11_DIR / "reproducibility" / "git_commit.txt",
    "code_snapshot": S11_DIR / "reproducibility" / "code_snapshot.json",
    "effective_config": S11_DIR / "reproducibility" / "effective_config.json",
    "random_seeds": S11_DIR / "reproducibility" / "random_seeds.json",
    "runtime_log": S11_DIR / "reproducibility" / "runtime_log.txt",
}
S11_CLAIM_MANIFEST = S11_DIR / "claim_manifest.json"
S11_CLAIM_ARTIFACTS = (
    ("report", "09_report/index.html"),
    ("fast_report_pointer", "09_report/fast_report_manifest.json"),
    ("physics_report_pointer", "09_report/physics_report_manifest.json"),
    ("target_landscape", "03_targets/ranked_targets_v3_with_efficacy.csv"),
    ("pharmacophore", "05_pharmacophore/consensus_pharmacophore_atoms.json"),
    ("analogs", "05_6_analogs/top30_with_properties.csv"),
    ("md_trajectory", "07_md/trajectory_index.tsv"),
    ("md_mmgbsa", "07_md/mmgbsa.tsv"),
    ("retrosynthesis", "07_5_retrosynthesis/synthesis_priority_ranking.csv"),
    ("reproducibility", "publication/reproducibility/artifact_manifest.json"),
    ("data_availability", "publication/data_availability.md"),
    ("figure_captions", "publication/figures/captions.json"),
    ("fig01_workflow", "publication/figures/fig01_workflow.svg"),
    ("fig02_target_landscape", "publication/figures/fig02_target_landscape.png"),
    ("fig03_pharmacophore", "publication/figures/fig03_pharmacophore.png"),
    ("fig04_analog_scatter", "publication/figures/fig04_analog_scatter.png"),
    ("fig05_md_rmsd", "publication/figures/fig05_md_rmsd.png"),
    ("fig06_retrosynthesis", "publication/figures/fig06_retrosynthesis.png"),
    ("fig07_eval_bars", "publication/figures/fig07_eval_bars.png"),
    ("fig08_case_study", "publication/figures/fig08_case_study.png"),
)
# Manuscript artifacts and dependencies exist only for an emitted draft. The
# claim-quality gate still inspects any emitted manuscript when configured.
if S11_EMIT_MANUSCRIPT_DRAFT:
    S11_CLAIM_ARTIFACTS += (
        ("manuscript_abstract", "publication/manuscript_draft/00_abstract.md"),
        ("manuscript_introduction", "publication/manuscript_draft/01_introduction.md"),
        ("manuscript_methods", "publication/manuscript_draft/02_methods.md"),
        ("manuscript_results", "publication/manuscript_draft/03_results.md"),
        ("manuscript_discussion", "publication/manuscript_draft/04_discussion.md"),
        ("manuscript_references", "publication/manuscript_draft/05_references.bib"),
    )
if config_bool(FINETUNE.get("enabled"), False):
    S11_CLAIM_ARTIFACTS += (
        ("reinvent_finetune", "05_6_analogs/reinvent4_finetune_manifest.json"),
    )
S11_MANUSCRIPT_OUTPUTS = {
    "abstract": S11_DIR / "manuscript_draft" / "00_abstract.md",
    "introduction": S11_DIR / "manuscript_draft" / "01_introduction.md",
    "results": S11_DIR / "manuscript_draft" / "03_results.md",
    "discussion": S11_DIR / "manuscript_draft" / "04_discussion.md",
    "references": S11_DIR / "manuscript_draft" / "05_references.bib",
}


rule make_figures:
    input:
        **S11_FIGURE_INPUTS
    output:
        captions = S11_DIR / "figures" / "captions.json",
        **S11_FIGURE_OUTPUTS
    log:
        "results/logs/{run_id}/stage11_figures.log".format(run_id=RUN_ID)
    conda:
        "../../envs/viz.yml"
    params:
        run_dir = str(RUN_DIR),
        eval_dir = str(S11_EVAL_DIR),
        out_dir = str(S11_DIR / "figures"),
        allow_placeholders = "--allow-placeholders"
            if S11_ALLOW_PLACEHOLDERS
            else ""
    shell:
        r"""
        mkdir -p {params.out_dir:q}
        python scripts/stage11_make_figures.py \
            --run-dir {params.run_dir:q} \
            --eval-dir {params.eval_dir:q} \
            --out-dir {params.out_dir:q} \
            --out-captions {output.captions:q} \
            {params.allow_placeholders} > {log:q} 2>&1
        """


rule write_data_availability:
    output:
        statement = S11_DIR / "data_availability.md"
    log:
        "results/logs/{run_id}/stage11_data_availability.log".format(run_id=RUN_ID)
    conda:
        "../../envs/base.yml"
    params:
        out_dir = str(S11_DIR),
        skin_efficacy_kg_doi = str(config.get("publication", {}).get("skin_efficacy_kg_doi", "pending")),
        pipeline_source_url = str(config.get("publication", {}).get("pipeline_source_url", "https://github.com/kangk1204/SkinScout_public")),
        draft_doi_ok = "--draft-doi-ok" if S11_DRAFT_DOI_OK else ""
    shell:
        r"""
        mkdir -p {params.out_dir:q}
        python scripts/stage11_data_availability.py \
            --out-md {output.statement:q} \
            --skin-efficacy-kg-doi {params.skin_efficacy_kg_doi:q} \
            --pipeline-source-url {params.pipeline_source_url:q} \
            {params.draft_doi_ok} > {log:q} 2>&1
        """


rule write_repro_pack:
    input:
        report = rules.molstar_report.output.html,
        fast_report_pointer = rules.molstar_report.output.fast_manifest,
        physics_report_pointer = rules.molstar_report.output.physics_manifest,
        consensus = rules.rrf_4way_consensus.output.top50,
        ranked_with_efficacy = rules.kg_efficacy_label.output.csv,
        eval_manifest = S11_EVAL_DIR / "iteration_manifest.json"
    output:
        versions = S11_DIR / "reproducibility" / "tool_versions.lock",
        manifest = S11_DIR / "reproducibility" / "artifact_manifest.json",
        **S11_REPRO_OUTPUTS
    log:
        "results/logs/{run_id}/stage11_repro.log".format(run_id=RUN_ID)
    conda:
        "../../envs/base.yml"
    params:
        run_dir = str(RUN_DIR),
        out_dir = str(S11_DIR / "reproducibility"),
        require_physics = "--require-physics-report" if S11_REQUIRE_PHYSICS else "",
        dirty_snapshot_limitation = (
            f"--dirty-snapshot-limitation {shlex.quote(S11_DIRTY_SNAPSHOT_LIMITATION)}"
            if S11_DIRTY_SNAPSHOT_LIMITATION
            else ""
        )
    shell:
        r"""
        mkdir -p {params.out_dir:q}
        python scripts/stage11_repro_pack.py \
            --run-dir {params.run_dir:q} \
            --out-dir {params.out_dir:q} \
            --fast-report-pointer {input.fast_report_pointer:q} \
            --physics-report-pointer {input.physics_report_pointer:q} \
            {params.require_physics} \
            {params.dirty_snapshot_limitation} \
            --eval-manifest {input.eval_manifest:q} > {log:q} 2>&1
        """


rule write_manuscript_draft:
    input:
        captions = rules.make_figures.output.captions,
        repro_manifest = rules.write_repro_pack.output.manifest,
        data_availability = rules.write_data_availability.output.statement
    output:
        methods = S11_DIR / "manuscript_draft" / "02_methods.md",
        **S11_MANUSCRIPT_OUTPUTS
    log:
        "results/logs/{run_id}/stage11_manuscript.log".format(run_id=RUN_ID)
    conda:
        "../../envs/base.yml"
    params:
        run_dir = str(RUN_DIR),
        out_dir = str(S11_DIR / "manuscript_draft"),
        allow_placeholders = "--allow-placeholders"
            if S11_ALLOW_PLACEHOLDERS
            else ""
    shell:
        r"""
        mkdir -p {params.out_dir:q}
        python scripts/stage11_manuscript.py \
            --run-dir {params.run_dir:q} \
            --out-dir {params.out_dir:q} \
            --captions {input.captions:q} \
            --repro-manifest {input.repro_manifest:q} \
            --data-availability {input.data_availability:q} \
            {params.allow_placeholders} > {log:q} 2>&1
        """


S11_CLAIM_INPUTS = {
    "report": rules.molstar_report.output.html,
    "fast_report_pointer": rules.molstar_report.output.fast_manifest,
    "physics_report_pointer": rules.molstar_report.output.physics_manifest,
    "target_landscape": rules.kg_efficacy_label.output.csv,
    "pharmacophore": rules.pharmacophore_consensus.output.json,
    "analogs": rules.mini_validate_funnel.output.csv,
    "md_trajectory": rules.gromacs_production.output.traj_index,
    "md_mmgbsa": rules.mmgbsa.output.report,
    "retrosynthesis": rules.score_routes.output.ranking,
    "reproducibility": rules.write_repro_pack.output.manifest,
    "data_availability": rules.write_data_availability.output.statement,
    "eval_manifest": S11_EVAL_DIR / "iteration_manifest.json",
    "activity_retrieval_gate": rules.activity_retrieval_final_gate.output[0],
    "figure_captions": rules.make_figures.output.captions,
    **S11_FIGURE_OUTPUTS,
}
# An opt-out of the manuscript draft must not make the claim manifest rule wait
# on manuscript outputs that no rule target requested.
if S11_EMIT_MANUSCRIPT_DRAFT:
    S11_CLAIM_INPUTS.update(S11_MANUSCRIPT_OUTPUTS)
    S11_CLAIM_INPUTS["manuscript_methods"] = rules.write_manuscript_draft.output.methods
if config_bool(FINETUNE.get("enabled"), False):
    S11_CLAIM_INPUTS["reinvent_finetune"] = rules.reinvent4_finetune.output.manifest


rule write_claim_manifest:
    input:
        **S11_CLAIM_INPUTS
    output:
        manifest = S11_CLAIM_MANIFEST
    log:
        "results/logs/{run_id}/stage11_claim_manifest.log".format(run_id=RUN_ID)
    conda:
        "../../envs/base.yml"
    params:
        run_dir = str(RUN_DIR),
        require_physics = "--require-physics-report" if S11_REQUIRE_PHYSICS else "",
        allow_diagnostic = "--allow-diagnostic" if S11_DIAGNOSTIC_BUNDLE else "",
        artifacts = " ".join(
            f"--artifact {shlex.quote(label + '=' + relative)}"
            for label, relative in S11_CLAIM_ARTIFACTS
        )
    shell:
        r"""
        mkdir -p {params.run_dir:q}/publication
        python scripts/stage11_claim_manifest.py \
            --run-dir {params.run_dir:q} \
            --out-manifest {output.manifest:q} \
            --evaluation-manifest {input.eval_manifest:q} \
            --activity-retrieval-gate {input.activity_retrieval_gate:q} \
            --fast-report-pointer {input.fast_report_pointer:q} \
            --physics-report-pointer {input.physics_report_pointer:q} \
            {params.require_physics} \
            {params.allow_diagnostic} \
            {params.artifacts} > {log:q} 2>&1
        """
