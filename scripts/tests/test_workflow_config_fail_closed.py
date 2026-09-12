"""Regression tests for fail-closed workflow defaults."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_FILES = [
    ROOT / "workflow" / "Snakefile",
    *sorted((ROOT / "workflow" / "rules").glob("*.smk")),
]


def _snakemake_executable() -> str:
    """Resolve the snakemake CLI, or skip.

    These cases shell out to the real workflow engine, which lives in the
    cosmax-base environment. Without this they raise FileNotFoundError and
    turn an unrelated environment into dozens of red failures that can mask a
    genuine regression.
    """
    executable = shutil.which("snakemake")
    if executable is None:
        pytest.skip("snakemake is not on PATH; run this suite in cosmax-base")
    return executable


def test_cosmetic_reference_mode_defaults_to_full_data() -> None:
    config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())

    assert config["cosmetic_db_mode"] == "full"


def test_stage0_cosmetic_rule_fallback_is_full_and_validated() -> None:
    rule_text = (ROOT / "workflow" / "rules" / "stage0_v3_cosmetic.smk").read_text()

    assert 'config.get("cosmetic_db_mode", "full")' in rule_text
    assert "cosmetic_db_mode must be 'full' or 'placeholder'" in rule_text


def test_stage0_drug_avoidance_rule_uses_configured_references() -> None:
    config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())
    rule_text = (ROOT / "workflow" / "rules" / "stage0_v3_cosmetic.smk").read_text()

    assert config["drug_avoidance"]["chembl_dir"] == "data/chembl37"
    assert config["drug_avoidance"]["orange_book"] == (
        "data/drug_avoidance/orange_book.csv"
    )
    assert config["drug_avoidance"]["drugbank_approved_parquet"] == (
        "data/drugbank/drugbank_approved.parquet"
    )
    assert 'DRUG_CFG.get("chembl_dir", config["paths"]["chembl"])' in rule_text
    assert "--chembl-dir {params.chembl_dir:q}" in rule_text
    assert "--orange-book {params.orange_book:q}" in rule_text
    assert "--drugbank-parquet {params.drugbank_parquet:q}" in rule_text
    rule_block = rule_text.split("rule drug_avoidance_ingest:")[1].split(
        "\n\nrule ",
        1,
    )[0]
    assert "rules.mirror_chembl.output.flag" in rule_block
    assert "rules.mirror_drugbank.output.approved" in rule_block


def test_stage0_operator_sources_are_declared_as_snakemake_inputs() -> None:
    infra_rules = (ROOT / "workflow" / "rules" / "stage0_infra.smk").read_text()
    cosmetic_rules = (ROOT / "workflow" / "rules" / "stage0_v3_cosmetic.smk").read_text()
    skin_rules = (ROOT / "workflow" / "rules" / "stage0_v3_skin.smk").read_text()

    assert "DRUGBANK_REQUIRED = config_bool" in infra_rules
    assert 'DRUGBANK / "drugbank_full_database.xml"' in infra_rules
    drugbank_block = infra_rules.split("rule mirror_drugbank:")[1].split(
        "\n\nrule ",
        1,
    )[0]
    assert "input:\n        DRUGBANK_SOURCE_INPUT" in drugbank_block
    assert "--allow-missing-optional" in drugbank_block

    assert "COSING_AUTO_MIRROR = config_bool" in cosmetic_rules
    assert "rule mirror_cosing_public_api:" in cosmetic_rules
    assert "stage0_mirror_cosing.py" in cosmetic_rules
    assert 'COSING_SOURCE_PATH = COSING_DIR / "cosing.csv"' in cosmetic_rules
    cosing_block = cosmetic_rules.split("rule cosing_ingest:")[1].split(
        "\n\nrule ",
        1,
    )[0]
    assert "input:\n        COSING_SOURCE_INPUT" in cosing_block

    assert 'SKIN_PROTEOME_DIR / "raw_lfq.tsv"' in skin_rules
    assert 'GTEX_DIR / "gtex_v10_gene_tpm.gct"' in skin_rules
    assert "REQUIRE_SKIN_PROTEOME_SOURCE = config_bool" in skin_rules
    assert "REQUIRE_GTEX_SOURCE = config_bool" in skin_rules
    assert "skin_source_input(" in skin_rules
    assert "allow_empty_skin_source(" in skin_rules
    assert 'HPA_DIR / "rna_tissue_consensus.tsv"' in skin_rules
    assert 'HPA_DIR / "rna_single_cell_type.tsv"' in skin_rules
    proteome_block = skin_rules.split("rule ingest_skin_proteome:")[1].split(
        "\n\nrule ",
        1,
    )[0]
    gtex_block = skin_rules.split("rule ingest_gtex_skin:")[1].split(
        "\n\nrule ",
        1,
    )[0]
    score_block = skin_rules.split("rule compute_skin_score:")[1].split(
        "\n\nrule ",
        1,
    )[0]
    assert "input:\n        SKIN_PROTEOME_SOURCE_INPUT" in proteome_block
    assert "input:\n        GTEX_SOURCE_INPUT" in gtex_block
    assert "allow_empty_skin_source(" in proteome_block
    assert "allow_empty_skin_source(" in gtex_block
    assert "proteome = rules.ingest_skin_proteome.output.tsv" in score_block
    assert "gtex = rules.ingest_gtex_skin.output.tsv" in score_block
    assert "hpa_tissue = rules.download_hpa.output.tissue" in score_block
    assert "hpa_cell = rules.download_hpa.output.cell" in score_block


def test_stage0_v3_python_ingests_declare_base_environment() -> None:
    cosmetic_rules = (ROOT / "workflow" / "rules" / "stage0_v3_cosmetic.smk").read_text()
    skin_rules = (ROOT / "workflow" / "rules" / "stage0_v3_skin.smk").read_text()

    for rule_name, rule_text in {
        "mirror_cosing_public_api": cosmetic_rules,
        "cosing_ingest": cosmetic_rules,
        "drug_avoidance_ingest": cosmetic_rules,
        "skin_efficacy_kg": cosmetic_rules,
        "ingest_skin_proteome": skin_rules,
        "ingest_gtex_skin": skin_rules,
    }.items():
        rule_block = rule_text.split(f"rule {rule_name}:")[1].split("\n\nrule ", 1)[0]
        assert "conda:" in rule_block
        assert '"../../envs/base.yml"' in rule_block
        assert "log:" in rule_block


def test_stage1_fallback_flags_parse_strict_booleans() -> None:
    config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())
    rule_text = (ROOT / "workflow" / "rules" / "stage1_preprocess.smk").read_text()

    assert config["stage1"]["allow_unprotonated_fallback"] is False
    assert config["stage1"]["allow_xtb_fallback"] is False
    assert (
        'config_bool(\n'
        '                config.get("stage1", {}).get("allow_unprotonated_fallback"),\n'
        "                False,\n"
        "            )"
    ) in rule_text


def test_stage1_standardize_builds_only_the_selected_input_argument() -> None:
    rule_text = (ROOT / "workflow" / "rules" / "stage1_preprocess.smk").read_text()

    assert "smiles_arg = (" in rule_text
    assert "sdf_arg = (" in rule_text
    assert "shlex.quote(config['compound_smiles'])" in rule_text
    assert "shlex.quote(config['compound_sdf'])" in rule_text
    assert '"${{cmd[@]}}" > {log:q} 2>&1' in rule_text
    assert (
        'config_bool(config.get("stage1", {}).get("allow_xtb_fallback"), False)'
        in rule_text
    )


def test_known_target_prior_omits_empty_supplemental_panel_argument() -> None:
    rule_text = (ROOT / "workflow" / "rules" / "stage3_v3_skin_weight.smk").read_text()

    assert "supplemental_arg = (" in rule_text
    assert "shlex.quote(\",\".join(KNOWN_TARGET_PRIORS[\"supplemental_source_csv\"]))" in rule_text
    assert "{params.supplemental_arg}" in rule_text


def test_workflow_degraded_flags_use_strict_boolean_parser() -> None:
    config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())

    assert config["run_dti_sanity"] is True
    assert config["stage0"]["allow_bindingdb_placeholder"] is False
    assert config["stage0"]["require_drugbank_source"] is False
    assert config["stage0"]["allow_drugbank_placeholder"] is False
    assert config["stage0"]["allow_missing_training_cutoff"] is False
    assert config["skin_expression"]["require_proteome_source"] is False
    assert config["skin_expression"]["require_gtex_source"] is False
    assert config["skin_expression"]["allow_empty_sources"] is False
    assert config["cosing"]["auto_mirror_public_api"] is True
    assert config["cosing"]["allow_missing_reference"] is False
    assert config["drug_avoidance"]["allow_missing_reference"] is False
    assert config["admet"]["allow_admet_ai_unavailable"] is False
    assert config["admet"]["allow_skin_sens_unavailable"] is False
    assert config["docking"]["allow_psichic_unavailable"] is False
    assert config["docking"]["psichic_batch_size"] == 512
    assert config["docking"]["psichic_score_batch_size"] == 8
    assert config["docking"]["psichic_esm_short_batch_size"] == 4
    assert config["docking"]["psichic_esm_medium_batch_size"] == 2
    assert config["docking"]["psichic_esm_long_batch_size"] == 1
    assert config["docking"]["psichic_max_sequence_length"] == 700
    assert config["docking"]["fast_mode_autodock_runs"] == 4
    assert config["docking"]["fast_mode_autodock_max_receptors"] == 256
    assert config["stage0"]["chembl_release"] == 37
    assert config["stage0"]["bindingdb_release"] == "2026-08"
    assert config["stage0"]["gtopdb_release"] == "2026.2"
    assert config["stage0"]["gtopdb_release_date"] == "2026-06-15"
    assert config["paths"]["gtopdb"] == "data/gtopdb"
    assert config["paths"]["chembl"] == "data/chembl37"
    assert config["docking"]["daina_evidence_mode"] == "retrieval"
    assert config["docking"]["daina_quality_policy"] == "legacy"
    assert config["docking"]["daina_scoring_method"] == "max-similarity"
    assert "fast_rrf_source_weights" not in config["docking"]
    assert config["pharmacophore"]["allow_empty_interaction_sources"] is False
    assert "allow_empty_boltz_attention" not in config["pharmacophore"]
    assert config["analog_gen"]["allow_empty_reinvent_output"] is False
    assert config["evaluation"]["sota"]["enabled"] is False
    assert config["evaluation"]["sota"]["default_context_profile"] == "auto"
    assert config["evaluation"]["sota"]["skin_known_min_case_top10"] == 0.80
    assert config["evaluation"]["sota"]["skin_known_min_target_top10"] == 0.50
    assert config["evaluation"]["sota"]["skin_known_min_target_top30"] == 0.60
    assert config["evaluation"]["activity_retrieval"]["required"] is True
    assert config["evaluation"]["activity_retrieval"]["allow_no_improvement"] is False
    assert config["evaluation"]["activity_retrieval"]["benchmark_dir"] == (
        "data/activity_benchmark_202608"
    )
    assert config["evaluation"]["activity_retrieval"]["retrieval_index_dir"] == (
        "data/activity_retrieval_202608"
    )
    assert config["evaluation"]["activity_retrieval"]["panels_dir"] == (
        "data/activity_recovery_panels_202608"
    )
    assert config["evaluation"]["activity_retrieval"]["rcsb_release_cutoff"] == (
        "2021-01-01"
    )
    assert config["evaluation"]["activity_retrieval"]["rcsb_min_nonpolymer_mw"] == 50.0
    assert config["evaluation"]["activity_retrieval"]["rcsb_raw_jsonl_gz"] == (
        "data/rcsb_holo_20210101/holo_contacts.jsonl.gz"
    )
    assert config["evaluation"]["activity_retrieval"]["rcsb_snapshot_manifest"] == (
        "data/rcsb_holo_20210101/manifest.json"
    )
    assert config["evaluation"]["activity_retrieval"]["rcsb_ranking_queries"] == (
        "data/rcsb_holo_20210101/ranking_queries.parquet"
    )
    assert config["evaluation"]["activity_retrieval"]["rcsb_panel_manifest"] == (
        "data/rcsb_holo_20210101/panel_manifest.json"
    )
    assert config["evaluation"]["activity_retrieval"]["rcsb_contact_fragment_manifest"] == (
        "data/rcsb_holo_20210101/contact_pocket_fragments.manifest.json"
    )
    assert config["evaluation"]["activity_retrieval"]["prior_pocket_fragment_manifest"] == (
        "data/pocket_fragments_202608_manifest.json"
    )
    assert config["evaluation"]["activity_retrieval"]["rcsb_contact_pocket_leakage_manifest"] == (
        "results/audits/rcsb_contact_pocket_leakage.manifest.json"
    )
    assert config["evaluation"]["activity_retrieval"]["pocket_leakage_foldseek"] == (
        "foldseek"
    )
    assert config["evaluation"]["activity_retrieval"]["pocket_leakage_threads"] == 16
    assert config["evaluation"]["activity_retrieval"]["benchmark_target_clusters"] == (
        "data/evidence_splits/screenable_target_clusters_2026_02.csv"
    )
    assert config["evaluation"]["activity_retrieval"]["source_target_exclusions"] == (
        "data/validation/activity_source_target_exclusions.csv"
    )
    assert config["evaluation"]["activity_retrieval"]["dual_cold_min_queries"] == 100
    assert config["evaluation"]["activity_retrieval"]["dual_cold_min_targets"] == 20
    assert config["evaluation"]["activity_retrieval"]["dual_cold_min_documents"] == 20
    assert config["evaluation"]["activity_retrieval"]["dual_cold_max_target_fraction"] == 0.20
    assert config["evaluation"]["activity_retrieval"]["dual_cold_min_effective_targets"] == 10.0
    assert config["evaluation"]["activity_retrieval"]["screenable_target_clusters"] == (
        "data/evidence_splits/screenable_target_clusters_2026_02.csv"
    )
    assert config["evaluation"]["activity_retrieval"]["final_evaluation_dir"] == (
        "results/eval/activity_retrieval_202608/test_evaluation"
    )

    offenders: list[str] = []
    for path in WORKFLOW_FILES:
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if "bool(config" in line and "config_bool(" not in line:
                offenders.append(f"{path.relative_to(ROOT)}:{lineno}:{line.strip()}")
            if (
                ('.get("allow_' in line or "S0.get(\"allow_" in line)
                and "False" in line
                and "config_bool(" not in line
            ):
                offenders.append(f"{path.relative_to(ROOT)}:{lineno}:{line.strip()}")

    assert offenders == []


def test_performance_v2_defaults_are_preregistered_and_disabled() -> None:
    config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())
    performance = config["evaluation"]["performance_v2"]

    assert performance["enabled"] is False
    assert performance["mode"] == "production"
    assert performance["target_fasta"] == (
        "data/evidence_splits/target_clusters_aug30_2026_02_all_seqs.fasta"
    )
    assert performance["target_clusters"] == (
        "data/evidence_splits/screenable_target_clusters_2026_02.csv"
    )
    assert performance["evaluation_panel_manifest"] == (
        "data/activity_recovery_panels_202608/manifest.json"
    )
    assert performance["dev_ranking_queries"] == (
        "data/activity_recovery_panels_202608/dev_ranking_queries.parquet"
    )
    assert performance["dev_calibration_pairs"] == (
        "data/activity_recovery_panels_202608/dev_calibration_pairs.parquet"
    )
    assert performance["dual_cold_ranking_queries"] == (
        "data/activity_recovery_panels_202608/dual_cold_ranking_queries.parquet"
    )
    assert performance["molformer_checkpoint"] == (
        "ibm-research/MoLFormer-XL-both-10pct"
    )
    assert performance["molformer_revision"] == "compat-v4"
    assert performance["esm2_checkpoint"] == "esm2_t30_150M_UR50D"
    assert performance["esm2_revision"] == "main"
    assert performance["protein_batch_size"] == 8
    assert performance["seeds"] == [17, 42, 73]
    assert performance["rerank_top_k"] == 512
    assert performance["structure_top_k"] == 64
    assert performance["boltz_top_k"] == 3
    assert performance["boltz_max_residues"] == 700
    assert performance["max_peak_vram_gib"] == 11.5
    assert performance["max_trainable_parameters"] == 10_000_000


def test_performance_v2_is_additive_and_fail_closed_in_workflow() -> None:
    snakefile = (ROOT / "workflow" / "Snakefile").read_text()
    rules = (ROOT / "workflow" / "rules" / "performance_v2.smk").read_text()
    attachment = (
        ROOT / "scripts" / "attach_performance_v2_predictions.py"
    ).read_text()

    assert 'include: "rules/performance_v2.smk"' in snakefile
    assert 'if PERFORMANCE_V2["enabled"]:' in snakefile
    assert "ranked_targets_v3_with_efficacy_and_performance_v2.csv" in snakefile
    assert "rule performance_v2_prepare_inputs:" in rules
    assert "--target-clusters {input.clusters:q}" in rules
    assert "rule performance_v2_embeddings:" in rules
    assert "--protein-batch-size {params.protein_batch}" in rules
    assert "rule performance_v2_train_p1:" in rules
    assert "--seeds {params.seeds:q}" in rules
    assert "rule performance_v2_score_query:" in rules
    assert "rule performance_v2_attach_stage3:" in rules
    assert "ranked = rules.kg_efficacy_label.output.csv" in rules
    assert "--candidate-id P1" in rules
    assert "research_candidate_not_promoted" in attachment


def test_workflow_rejects_performance_v2_contract_drift(tmp_path: Path) -> None:
    cases = [
        (
            "bad_seeds",
            {"seeds": [17, 42]},
            "evaluation.performance_v2.seeds must remain exactly [17, 42, 73]",
        ),
        (
            "excess_vram",
            {"max_peak_vram_gib": 11.6},
            "evaluation.performance_v2.max_peak_vram_gib",
        ),
        (
            "inverted_budgets",
            {"structure_top_k": 2, "boltz_top_k": 3},
            "boltz_top_k <= structure_top_k <= rerank_top_k",
        ),
        (
            "model_revision_drift",
            {"molformer_revision": "main"},
            "evaluation.performance_v2.molformer_revision must remain pinned",
        ),
    ]

    for case_name, override, expected in cases:
        bad_config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())
        bad_config["evaluation"]["performance_v2"].update(override)
        config_path = tmp_path / f"{case_name}.yaml"
        config_path.write_text(yaml.safe_dump(bad_config))
        result = subprocess.run(
            [
                _snakemake_executable(),
                "-s",
                "workflow/Snakefile",
                "--cores",
                "1",
                "--configfile",
                str(config_path),
                "--config",
                f"run_id={case_name}",
                "compound_smiles=CCO",
                "-n",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        assert result.returncode != 0, case_name
        assert expected in result.stdout + result.stderr, case_name


def test_fast_workflow_has_no_heavyweight_rerank_budget_knobs() -> None:
    snakefile = (ROOT / "workflow" / "Snakefile").read_text()

    assert "fast_mode_rerank_candidates" not in snakefile
    assert "fast_mode_rtm_top_n" not in snakefile
    assert "fast_mode_boltz_top_n" not in snakefile


def test_activity_retrieval_gates_are_wired_into_workflow_and_eval() -> None:
    snakefile = (ROOT / "workflow" / "Snakefile").read_text()
    rules = (ROOT / "workflow" / "rules" / "activity_retrieval.smk").read_text()
    fast_rules = (ROOT / "workflow" / "rules" / "stage3b_fast.smk").read_text()
    comprehensive_rules = (
        ROOT / "workflow" / "rules" / "stage3a_comprehensive.smk"
    ).read_text()
    disagreement_rules = (
        ROOT / "workflow" / "rules" / "stage3c_disagreement.smk"
    ).read_text()
    publication_rules = (
        ROOT / "workflow" / "rules" / "stage11_publication.smk"
    ).read_text()
    gate_script = (
        ROOT / "scripts" / "validate_activity_retrieval_gate.py"
    ).read_text()
    eval_runner = (ROOT / "eval" / "run_all.sh").read_text()
    index_rule = rules.split("rule activity_retrieval_index:", 1)[1].split(
        "rule rcsb_holo_snapshot:", 1
    )[0]

    assert 'include: "rules/activity_retrieval.smk"' in snakefile
    assert 'MANIFESTS / "activity_retrieval_operational_gate.flag"' in snakefile
    assert "rule activity_benchmark_partitions:" in rules
    assert "rule activity_retrieval_index:" in rules
    assert "threads:\n        4" in index_rule
    assert "--workers {threads}" in index_rule
    assert "rule rcsb_holo_snapshot:" in rules
    assert "rule rcsb_holo_direct_contact_panel:" in rules
    assert "rule activity_prior_pocket_fragments:" in rules
    assert "rule rcsb_contact_pocket_fragments:" in rules
    assert "rule rcsb_contact_pocket_leakage_audit:" in rules
    assert "rule activity_recovery_panels:" in rules
    assert "rule activity_retrieval_dev_selection:" in rules
    assert "rule activity_retrieval_final_evaluation:" in rules
    assert "rule activity_retrieval_operational_gate:" in rules
    assert "rule activity_retrieval_final_gate:" in rules
    assert "passes_dev_gate" in gate_script
    assert "passes_frozen_test_gate" in gate_script
    assert "skinscout.activity-recovery-panels.v5" in gate_script
    assert "rcsb_holo_direct_contact_panel" in gate_script
    assert "no_affinities_or_calibration" in gate_script
    assert "never_training_or_model_selection" in gate_script
    assert "skinscout.rcsb-contact-pocket-leakage-audit.v1" in gate_script
    assert "missing_foldseek_hits_are_uncovered_not_cold" in gate_script
    assert "fragment_extraction_failures_are_uncovered_not_cold" in gate_script
    assert "pocket_leakage_rows" in gate_script
    assert "passes_panel_adequacy_gate" in gate_script
    assert "does not match dev-selected recipe" in gate_script
    assert "AR_BENCHMARK_TARGET_CLUSTERS" in rules
    assert "AR_SOURCE_TARGET_EXCLUSIONS" in rules
    assert "--source-target-exclusions" in rules
    assert "AR_SCREENABLE_TARGET_CLUSTERS" in rules
    assert "--release-cutoff {params.release_cutoff:q}" in rules
    assert "--min-nonpolymer-mw {params.min_nonpolymer_mw:q}" in rules
    assert "--rcsb-ranking-queries {input.rcsb_ranking_queries:q}" in rules
    assert "--rcsb-panel-manifest {input.rcsb_panel_manifest:q}" in rules
    assert "build_rcsb_contact_pocket_fragments.py" in rules
    assert rules.count("--workers {threads}") >= 2
    assert "audit_rcsb_contact_pocket_leakage.py" in rules
    assert "--pocket-leakage-manifest {input.pocket_leakage:q}" in rules
    assert "validate_activity_retrieval_gate.py create" in rules
    assert "validate_activity_retrieval_gate.py create-operational" in rules
    assert "--allow-no-improvement > {log:q}" in rules
    assert "touch {output:q}" not in rules
    assert fast_rules.count(
        'MANIFESTS / "activity_retrieval_operational_gate.flag"'
    ) >= 1
    assert fast_rules.count(
        "validate_activity_retrieval_gate.py check-operational"
    ) >= 1
    assert 'MANIFESTS / "activity_retrieval_operational_gate.flag"' in comprehensive_rules
    assert "validate_activity_retrieval_gate.py check-operational" in comprehensive_rules
    assert 'MANIFESTS / "activity_retrieval_operational_gate.flag"' in disagreement_rules
    assert "validate_activity_retrieval_gate.py check-operational" in disagreement_rules
    assert 'rules.activity_retrieval_final_gate.output[0]' in publication_rules
    assert "ACTIVITY_RETRIEVAL_REQUIRED" in eval_runner
    assert "activity retrieval final-evaluation manifest" in eval_runner
    assert "activity retrieval RCSB contact-pocket leakage manifest" in eval_runner
    assert "validate_activity_retrieval_gate" in eval_runner
    assert "passes_frozen_test_gate" in eval_runner
    assert "skinscout.activity-recovery-panels.v5" in eval_runner
    assert "rcsb_holo_direct_contact_panel" in eval_runner
    assert "no_affinities_or_calibration" in eval_runner
    assert "never_training_or_model_selection" in eval_runner
    assert "passes_panel_adequacy_gate" in eval_runner


def test_daina_overlay_and_comprehensive_rrf_emit_auditable_provenance() -> None:
    fast_rules = (ROOT / "workflow" / "rules" / "stage3b_fast.smk").read_text()
    comprehensive_rules = (
        ROOT / "workflow" / "rules" / "stage3a_comprehensive.smk"
    ).read_text()
    stage0_rules = (ROOT / "workflow" / "rules" / "stage0_infra.smk").read_text()

    assert 'manifest = CHEMBL / "source_manifest.json"' in stage0_rules
    assert 'fingerprints = CHEMBL / "fp_morgan2_2048.parquet"' in stage0_rules
    assert "{params.release:q}" in stage0_rules
    assert "rule build_bindingdb_temporal_evidence:" in stage0_rules
    assert "rule mirror_gtopdb:" in stage0_rules
    assert "rule build_gtopdb_activity_evidence:" in stage0_rules
    assert "--gtopdb-evidence" in (
        ROOT / "workflow" / "rules" / "activity_retrieval.smk"
    ).read_text()
    assert "rule build_activity_evidence_splits:" in stage0_rules
    assert "--require-activity-evidence" in stage0_rules
    assert "--out-metadata-json {output.metadata:q}" in fast_rules
    assert "--evidence-mode {params.evidence_mode:q}" in fast_rules
    assert "scripts/stage3_select_daina.py" in fast_rules
    assert "--top-n {params.top_n:q}" in fast_rules
    assert "scripts/stage3_daina_structural_overlay.py" in fast_rules
    assert "--gnina-status-manifest {input.gnina_status:q}" in fast_rules
    assert "--out-csv {output.canonical:q}" in fast_rules
    assert "--source-weights {params.source_weights:q}" not in fast_rules
    assert "--recipe-id {params.recipe_id:q}" not in fast_rules
    assert "--source-weights {params.source_weights:q}" in comprehensive_rules
    assert "--recipe-id {params.recipe_id:q}" in comprehensive_rules


def test_psichic_is_comprehensive_only_and_uses_configured_batch_size() -> None:
    fast_rule_text = (ROOT / "workflow" / "rules" / "stage3b_fast.smk").read_text()
    comprehensive_rule_text = (
        ROOT / "workflow" / "rules" / "stage3c_disagreement.smk"
    ).read_text()

    assert "rule psichic_proteome:" not in fast_rule_text
    assert 'batch_size = config["docking"]["psichic_batch_size"]' in comprehensive_rule_text
    assert (
        'score_batch_size = config["docking"]["psichic_score_batch_size"]'
        in comprehensive_rule_text
    )
    assert (
        'esm_short_batch_size = config["docking"]["psichic_esm_short_batch_size"]'
        in comprehensive_rule_text
    )
    assert (
        'esm_medium_batch_size = config["docking"]["psichic_esm_medium_batch_size"]'
        in comprehensive_rule_text
    )
    assert (
        'esm_long_batch_size = config["docking"]["psichic_esm_long_batch_size"]'
        in comprehensive_rule_text
    )
    assert (
        'max_sequence_length = config["docking"]["psichic_max_sequence_length"]'
        in comprehensive_rule_text
    )
    assert "--batch-size {params.batch_size:q}" in comprehensive_rule_text
    assert "--score-batch-size {params.score_batch_size:q}" in comprehensive_rule_text
    assert (
        "PSICHIC_ESM_SHORT_BATCH={params.esm_short_batch_size:q}"
        in comprehensive_rule_text
    )
    assert (
        "PSICHIC_ESM_MEDIUM_BATCH={params.esm_medium_batch_size:q}"
        in comprehensive_rule_text
    )
    assert (
        "PSICHIC_ESM_LONG_BATCH={params.esm_long_batch_size:q}"
        in comprehensive_rule_text
    )
    assert "--max-sequence-length {params.max_sequence_length:q}" in comprehensive_rule_text
    assert "--batch-size 512" not in comprehensive_rule_text
    assert "--score-batch-size 8" not in comprehensive_rule_text


def test_fast_daina_rule_uses_frozen_selection_contract() -> None:
    rule_text = (ROOT / "workflow" / "rules" / "stage3b_fast.smk").read_text()

    assert 'top_n = DOCKING["fast_mode_daina_top_n"]' in rule_text
    assert "--top-n {params.top_n:q}" in rule_text
    assert "--min-sources-per-target" not in rule_text


def test_fast_autodock_rule_uses_fast_limits() -> None:
    fast_rule_text = (ROOT / "workflow" / "rules" / "stage3b_fast.smk").read_text()
    comprehensive_rule_text = (
        ROOT / "workflow" / "rules" / "stage3a_comprehensive.smk"
    ).read_text()

    assert 'nrun         = config["docking"]["fast_mode_autodock_runs"]' in fast_rule_text
    assert (
        'max_receptors = config["docking"]["fast_mode_autodock_max_receptors"]'
        in fast_rule_text
    )
    assert "--max-receptors {params.max_receptors:q}" in fast_rule_text
    assert "--engine autodock_gpu" in fast_rule_text
    assert "--vina-cpu {threads:q}" in fast_rule_text
    assert "--ls-method {params.ls_method:q}" in fast_rule_text
    assert 'nrun         = config["docking"]["autodock_runs"]' in comprehensive_rule_text
    assert "--max-receptors" not in comprehensive_rule_text
    assert "--engine autodock_gpu" in comprehensive_rule_text
    assert "--vina-cpu {threads:q}" in comprehensive_rule_text


def test_stage3_halt_gates_parse_multiline_decision_files() -> None:
    rule_text = (ROOT / "workflow" / "rules" / "stage3a_comprehensive.smk").read_text()

    assert "decision=$(head -n 1 {input.decision:q})" in rule_text
    assert "cosmetic_decision=$(head -n 1 {input.cosmetic_decision:q})" in rule_text
    assert "decision=$(cat {input.decision})" not in rule_text
    assert "cosmetic_decision=$(cat {input.cosmetic_decision})" not in rule_text


def test_stage3_dti_rules_receive_halt_gates() -> None:
    fast_rule_text = (ROOT / "workflow" / "rules" / "stage3b_fast.smk").read_text()
    sanity_rule_text = (
        ROOT / "workflow" / "rules" / "stage3c_disagreement.smk"
    ).read_text()

    assert fast_rule_text.count(
        "decision = rules.skin_sens_consensus.output.decision"
    ) >= 1
    assert fast_rule_text.count(
        "cosmetic_decision = rules.cosmetic_drug_decision.output.decision"
    ) >= 1
    assert fast_rule_text.count("decision=$(head -n 1 {input.decision:q})") >= 1
    assert fast_rule_text.count(
        "cosmetic_decision=$(head -n 1 {input.cosmetic_decision:q})"
    ) >= 1
    assert "refusing fast DTI" in fast_rule_text

    assert "decision = rules.skin_sens_consensus.output.decision" in sanity_rule_text
    assert (
        "cosmetic_decision = rules.cosmetic_drug_decision.output.decision"
        in sanity_rule_text
    )
    assert "decision=$(head -n 1 {input.decision:q})" in sanity_rule_text
    assert (
        "cosmetic_decision=$(head -n 1 {input.cosmetic_decision:q})"
        in sanity_rule_text
    )
    assert "refusing DTI sanity" in sanity_rule_text


def test_stage3_skin_weight_receives_drug_warning_source() -> None:
    rule_text = (ROOT / "workflow" / "rules" / "stage3_v3_skin_weight.smk").read_text()

    assert "drugs = rules.drug_avoidance_match.output.json" in rule_text
    assert "--drug-json {input.drugs:q}" in rule_text


def test_fast_overlay_uses_annotation_only_evidence() -> None:
    rule_text = (ROOT / "workflow" / "rules" / "stage3b_fast.smk").read_text()

    assert 'efficacy_kg = KG_DIR / "skin_efficacy.graphml"' in rule_text
    assert "--skin-kg {input.efficacy_kg:q}" in rule_text
    assert "--docking-status-manifest {input.autodock_status:q}" in rule_text
    assert "--gnina-status-manifest {input.gnina_status:q}" in rule_text
    assert "--rtm-top-n" not in rule_text
    assert "--boltz-top-n" not in rule_text


def test_workflow_rejects_ambiguous_boolean_cli_override() -> None:
    res = subprocess.run(
        [
            _snakemake_executable(),
            "-s",
            "workflow/Snakefile",
            "--cores",
            "1",
            "--config",
            "run_id=bad_boolean_probe",
            "compound_smiles=CCO",
            "run_dti_sanity=maybe",
            "-n",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert res.returncode != 0
    output = res.stdout + res.stderr
    assert "Config boolean must be one of true/false" in output


def test_workflow_rejects_invalid_skin_weight_cli_override() -> None:
    res = subprocess.run(
        [
            _snakemake_executable(),
            "-s",
            "workflow/Snakefile",
            "--cores",
            "1",
            "--config",
            "run_id=bad_skin_weight_probe",
            "compound_smiles=CCO",
            "skin_weight=not-a-number",
            "-n",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert res.returncode != 0
    output = res.stdout + res.stderr
    assert (
        "Config value 'skin_weight' must be a finite number in [0.0, 1.0]"
        in output
    )


def test_workflow_rejects_invalid_skin_min_threshold_cli_override() -> None:
    res = subprocess.run(
        [
            _snakemake_executable(),
            "-s",
            "workflow/Snakefile",
            "--cores",
            "1",
            "--config",
            "run_id=bad_skin_min_probe",
            "compound_smiles=CCO",
            "skin_min_threshold=not-a-number",
            "-n",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert res.returncode != 0
    output = res.stdout + res.stderr
    assert (
        "Config value 'skin_min_threshold' must be a finite number in [0.0, 1.0]"
        in output
    )


def test_workflow_rejects_unsafe_run_id_cli_override() -> None:
    res = subprocess.run(
        [
            _snakemake_executable(),
            "-s",
            "workflow/Snakefile",
            "--cores",
            "1",
            "--config",
            "run_id=../escape",
            "compound_smiles=CCO",
            "-n",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert res.returncode != 0
    output = res.stdout + res.stderr
    assert "Config value 'run_id' must be a safe run identifier" in output


def test_clean_run_uses_checked_python_path_removal() -> None:
    snakefile = (ROOT / "workflow" / "Snakefile").read_text()
    clean_rule = snakefile.split("rule clean_run:", 1)[1]

    assert "rm -rf" not in clean_rule
    assert "run_dir.parent != results_root" in clean_rule
    assert "RUN_DIR.is_symlink()" in clean_rule
    assert "shutil.rmtree(RUN_DIR)" in clean_rule


def test_enabled_analog_generation_targets_validated_funnel_output() -> None:
    snakefile = (ROOT / "workflow" / "Snakefile").read_text()

    assert 'targets.append(str(S56_DIR / "top30_analogs.sdf"))' in snakefile
    assert 'targets.append(str(S56_DIR / "all_generated.smi"))' not in snakefile


def test_reinvent_finetune_is_wired_to_generation_and_claim_provenance() -> None:
    config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())
    analog_rule = (ROOT / "workflow" / "rules" / "stage5_6_analog_gen.smk").read_text()
    publication_rule = (ROOT / "workflow" / "rules" / "stage11_publication.smk").read_text()

    assert config["analog_gen"]["finetune"]["enabled"] is False
    assert "rule reinvent4_finetune:" in analog_rule
    assert "--holdout-smi {input.holdout:q}" in analog_rule
    assert "--out-manifest {output.manifest:q}" in analog_rule
    assert "S56_MODEL_MANIFEST = rules.reinvent4_finetune.output.manifest" in analog_rule
    assert "reinvent4_finetune_manifest.json" in publication_rule
    assert 'S11_CLAIM_INPUTS["reinvent_finetune"]' in publication_rule


def test_pharmacophore_anchor_map_is_required_by_reinvent_generation() -> None:
    pharmacophore_rule = (
        ROOT / "workflow" / "rules" / "stage5_5_pharmacophore.smk"
    ).read_text()
    analog_rule = (
        ROOT / "workflow" / "rules" / "stage5_6_analog_gen.smk"
    ).read_text()

    assert "rule pharmacophore_anchor_map:" in pharmacophore_rule
    assert "python scripts/stage5_5_atom_map.py" in pharmacophore_rule
    assert "--parent-sdf {input.parent_sdf:q}" in pharmacophore_rule
    assert "--consensus-json {input.consensus:q}" in pharmacophore_rule
    assert "anchors   = rules.pharmacophore_anchor_map.output.json" in analog_rule
    assert "--interaction-anchors {input.anchors:q}" in analog_rule


def test_workflow_rejects_missing_compound_input() -> None:
    res = subprocess.run(
        [
            _snakemake_executable(),
            "-s",
            "workflow/Snakefile",
            "--cores",
            "1",
            "--config",
            "run_id=missing_compound_probe",
            "-n",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert res.returncode != 0
    output = res.stdout + res.stderr
    assert "Exactly one of compound_smiles or compound_sdf must be provided" in output


def test_workflow_rejects_duplicate_compound_inputs_cli_override(
    tmp_path: Path,
) -> None:
    sdf_path = tmp_path / "input.sdf"
    sdf_path.write_text("placeholder\n")

    res = subprocess.run(
        [
            _snakemake_executable(),
            "-s",
            "workflow/Snakefile",
            "--cores",
            "1",
            "--config",
            "run_id=duplicate_compound_probe",
            "compound_smiles=CCO",
            f"compound_sdf={sdf_path}",
            "-n",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert res.returncode != 0
    output = res.stdout + res.stderr
    assert "Exactly one of compound_smiles or compound_sdf must be provided" in output


def test_workflow_rejects_blank_path_configfile_override(
    tmp_path: Path,
) -> None:
    bad_config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())
    bad_config["paths_v3"]["skin_kg"] = " "
    config_path = tmp_path / "bad_blank_path.yaml"
    config_path.write_text(yaml.safe_dump(bad_config))

    res = subprocess.run(
        [
            _snakemake_executable(),
            "-s",
            "workflow/Snakefile",
            "--cores",
            "1",
            "--configfile",
            str(config_path),
            "--config",
            "run_id=bad_blank_path_probe",
            "compound_smiles=CCO",
            "-n",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert res.returncode != 0
    output = res.stdout + res.stderr
    assert "Config path 'paths_v3.skin_kg' must be a non-empty path" in output


def test_workflow_rejects_invalid_leakage_threshold_configfile_override(
    tmp_path: Path,
) -> None:
    bad_config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())
    bad_config["evaluation"]["leakage"]["ligand_tanimoto_threshold"] = "wide"
    config_path = tmp_path / "bad_leakage_threshold.yaml"
    config_path.write_text(yaml.safe_dump(bad_config))

    res = subprocess.run(
        [
            _snakemake_executable(),
            "-s",
            "workflow/Snakefile",
            "--cores",
            "1",
            "--configfile",
            str(config_path),
            "--config",
            "run_id=bad_leakage_threshold_probe",
            "compound_smiles=CCO",
            "-n",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert res.returncode != 0
    output = res.stdout + res.stderr
    assert (
        "Config value 'evaluation.leakage.ligand_tanimoto_threshold' "
        "must be a finite number"
    ) in output


def test_workflow_rejects_invalid_eval_metric_threshold_configfile_override(
    tmp_path: Path,
) -> None:
    bad_config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())
    bad_config["evaluation"]["thresholds"]["analog_min_novelty"] = "optimistic"
    config_path = tmp_path / "bad_eval_metric_threshold.yaml"
    config_path.write_text(yaml.safe_dump(bad_config))

    res = subprocess.run(
        [
            _snakemake_executable(),
            "-s",
            "workflow/Snakefile",
            "--cores",
            "1",
            "--configfile",
            str(config_path),
            "--config",
            "run_id=bad_eval_metric_threshold_probe",
            "compound_smiles=CCO",
            "-n",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert res.returncode != 0
    output = res.stdout + res.stderr
    assert (
        "Config value 'evaluation.thresholds.analog_min_novelty' "
        "must be a finite number"
    ) in output


def test_workflow_rejects_invalid_publication_boolean_configfile_override(
    tmp_path: Path,
) -> None:
    bad_config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())
    bad_config["publication"]["build_publication_package"] = "maybe"
    config_path = tmp_path / "bad_publication_boolean.yaml"
    config_path.write_text(yaml.safe_dump(bad_config))

    res = subprocess.run(
        [
            _snakemake_executable(),
            "-s",
            "workflow/Snakefile",
            "--cores",
            "1",
            "--configfile",
            str(config_path),
            "--config",
            "run_id=bad_publication_boolean_probe",
            "compound_smiles=CCO",
            "-n",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert res.returncode != 0
    output = res.stdout + res.stderr
    assert "Config boolean 'publication.build_publication_package' must be" in output


def test_workflow_rejects_invalid_publication_source_url_configfile_override(
    tmp_path: Path,
) -> None:
    bad_config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())
    bad_config["publication"]["pipeline_source_url"] = "ftp://example.org/repo"
    config_path = tmp_path / "bad_publication_url.yaml"
    config_path.write_text(yaml.safe_dump(bad_config))

    res = subprocess.run(
        [
            _snakemake_executable(),
            "-s",
            "workflow/Snakefile",
            "--cores",
            "1",
            "--configfile",
            str(config_path),
            "--config",
            "run_id=bad_publication_url_probe",
            "compound_smiles=CCO",
            "-n",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert res.returncode != 0
    output = res.stdout + res.stderr
    assert "Config URL 'publication.pipeline_source_url' must use http(s)" in output


def test_workflow_rejects_claim_publication_with_pending_doi_configfile_override(
    tmp_path: Path,
) -> None:
    bad_config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())
    bad_config["publication"]["build_publication_package"] = True
    bad_config["publication"]["skin_efficacy_kg_doi"] = "pending"
    bad_config["publication"]["draft_doi_ok"] = False
    config_path = tmp_path / "bad_publication_pending_doi.yaml"
    config_path.write_text(yaml.safe_dump(bad_config))

    res = subprocess.run(
        [
            _snakemake_executable(),
            "-s",
            "workflow/Snakefile",
            "--cores",
            "1",
            "--configfile",
            str(config_path),
            "--config",
            "run_id=bad_publication_pending_doi_probe",
            "compound_smiles=CCO",
            "-n",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert res.returncode != 0
    output = res.stdout + res.stderr
    assert "Config DOI 'publication.skin_efficacy_kg_doi' must be a real DOI" in output


def test_workflow_rejects_claim_publication_with_placeholders_configfile_override(
    tmp_path: Path,
) -> None:
    bad_config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())
    bad_config["publication"]["build_publication_package"] = True
    bad_config["publication"]["allow_placeholders"] = True
    bad_config["publication"]["draft_doi_ok"] = False
    bad_config["publication"]["skin_efficacy_kg_doi"] = "10.5281/zenodo.1234567"
    config_path = tmp_path / "bad_publication_placeholders.yaml"
    config_path.write_text(yaml.safe_dump(bad_config))

    res = subprocess.run(
        [
            _snakemake_executable(),
            "-s",
            "workflow/Snakefile",
            "--cores",
            "1",
            "--configfile",
            str(config_path),
            "--config",
            "run_id=bad_publication_placeholders_probe",
            "compound_smiles=CCO",
            "-n",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert res.returncode != 0
    output = res.stdout + res.stderr
    assert "publication.allow_placeholders cannot be true" in output
    assert "placeholder outputs are only for explicit draft diagnostics" in output


def test_workflow_rejects_claim_publication_with_draft_doi_ok_configfile_override(
    tmp_path: Path,
) -> None:
    bad_config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())
    bad_config["publication"]["build_publication_package"] = True
    bad_config["publication"]["allow_placeholders"] = False
    bad_config["publication"]["draft_doi_ok"] = True
    bad_config["publication"]["skin_efficacy_kg_doi"] = "10.5281/zenodo.1234567"
    config_path = tmp_path / "bad_publication_draft_doi_ok.yaml"
    config_path.write_text(yaml.safe_dump(bad_config))

    res = subprocess.run(
        [
            _snakemake_executable(),
            "-s",
            "workflow/Snakefile",
            "--cores",
            "1",
            "--configfile",
            str(config_path),
            "--config",
            "run_id=bad_publication_draft_doi_ok_probe",
            "compound_smiles=CCO",
            "-n",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert res.returncode != 0
    output = res.stdout + res.stderr
    assert "publication.draft_doi_ok cannot be true" in output
    assert "claim-quality publication output requires a real DOI" in output


def test_workflow_rejects_invalid_docking_top_pct_configfile_override(
    tmp_path: Path,
) -> None:
    bad_config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())
    bad_config["docking"]["top_pct_to_rescore"] = "not-a-number"
    config_path = tmp_path / "bad_docking_top_pct.yaml"
    config_path.write_text(yaml.safe_dump(bad_config))

    res = subprocess.run(
        [
            _snakemake_executable(),
            "-s",
            "workflow/Snakefile",
            "--cores",
            "1",
            "--configfile",
            str(config_path),
            "--config",
            "run_id=bad_docking_top_pct_probe",
            "compound_smiles=CCO",
            "-n",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert res.returncode != 0
    output = res.stdout + res.stderr
    assert (
        "Config value 'docking.top_pct_to_rescore' must be a finite number "
        "in [0.0, 1.0]"
    ) in output


def test_workflow_rejects_invalid_docking_rrf_k_configfile_override(
    tmp_path: Path,
) -> None:
    bad_config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())
    bad_config["docking"]["rrf_k"] = "1.5"
    config_path = tmp_path / "bad_docking_rrf_k.yaml"
    config_path.write_text(yaml.safe_dump(bad_config))

    res = subprocess.run(
        [
            _snakemake_executable(),
            "-s",
            "workflow/Snakefile",
            "--cores",
            "1",
            "--configfile",
            str(config_path),
            "--config",
            "run_id=bad_docking_rrf_k_probe",
            "compound_smiles=CCO",
            "-n",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert res.returncode != 0
    output = res.stdout + res.stderr
    assert "Config value 'docking.rrf_k' must be an integer" in output


def test_workflow_rejects_invalid_fast_autodock_max_receptors_configfile_override(
    tmp_path: Path,
) -> None:
    bad_config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())
    bad_config["docking"]["fast_mode_autodock_max_receptors"] = -1
    config_path = tmp_path / "bad_fast_autodock_max_receptors.yaml"
    config_path.write_text(yaml.safe_dump(bad_config))

    res = subprocess.run(
        [
            _snakemake_executable(),
            "-s",
            "workflow/Snakefile",
            "--cores",
            "1",
            "--configfile",
            str(config_path),
            "--config",
            "run_id=bad_fast_autodock_max_receptors_probe",
            "compound_smiles=CCO",
            "-n",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert res.returncode != 0
    output = res.stdout + res.stderr
    assert (
        "Config value 'docking.fast_mode_autodock_max_receptors' must be an integer"
        in output
    )


def test_workflow_rejects_noncanonical_fast_daina_target_count(
    tmp_path: Path,
) -> None:
    bad_config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())
    bad_config["docking"]["fast_mode_daina_top_n"] = 128
    config_path = tmp_path / "bad_fast_daina_top_n.yaml"
    config_path.write_text(yaml.safe_dump(bad_config))

    res = subprocess.run(
        [
            _snakemake_executable(),
            "-s",
            "workflow/Snakefile",
            "--cores",
            "1",
            "--configfile",
            str(config_path),
            "--config",
            "run_id=bad_fast_daina_top_n_probe",
            "compound_smiles=CCO",
            "-n",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert res.returncode != 0
    assert "fast_mode_daina_top_n is a frozen public contract and must equal 256" in (
        res.stdout + res.stderr
    )


def test_workflow_rejects_invalid_psichic_batch_size_configfile_override(
    tmp_path: Path,
) -> None:
    bad_config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())
    bad_config["docking"]["psichic_batch_size"] = 0
    config_path = tmp_path / "bad_psichic_batch_size.yaml"
    config_path.write_text(yaml.safe_dump(bad_config))

    res = subprocess.run(
        [
            _snakemake_executable(),
            "-s",
            "workflow/Snakefile",
            "--cores",
            "1",
            "--configfile",
            str(config_path),
            "--config",
            "run_id=bad_psichic_batch_size_probe",
            "compound_smiles=CCO",
            "-n",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert res.returncode != 0
    output = res.stdout + res.stderr
    assert "Config value 'docking.psichic_batch_size' must be an integer" in output


def test_workflow_rejects_invalid_psichic_score_batch_size_configfile_override(
    tmp_path: Path,
) -> None:
    bad_config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())
    bad_config["docking"]["psichic_score_batch_size"] = 0
    config_path = tmp_path / "bad_psichic_score_batch_size.yaml"
    config_path.write_text(yaml.safe_dump(bad_config))

    res = subprocess.run(
        [
            _snakemake_executable(),
            "-s",
            "workflow/Snakefile",
            "--cores",
            "1",
            "--configfile",
            str(config_path),
            "--config",
            "run_id=bad_psichic_score_batch_size_probe",
            "compound_smiles=CCO",
            "-n",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert res.returncode != 0
    output = res.stdout + res.stderr
    assert (
        "Config value 'docking.psichic_score_batch_size' must be an integer"
        in output
    )


def test_workflow_rejects_invalid_psichic_esm_batch_size_configfile_override(
    tmp_path: Path,
) -> None:
    bad_config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())
    bad_config["docking"]["psichic_esm_long_batch_size"] = 0
    config_path = tmp_path / "bad_psichic_esm_batch_size.yaml"
    config_path.write_text(yaml.safe_dump(bad_config))

    res = subprocess.run(
        [
            _snakemake_executable(),
            "-s",
            "workflow/Snakefile",
            "--cores",
            "1",
            "--configfile",
            str(config_path),
            "--config",
            "run_id=bad_psichic_esm_batch_size_probe",
            "compound_smiles=CCO",
            "-n",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert res.returncode != 0
    output = res.stdout + res.stderr
    assert (
        "Config value 'docking.psichic_esm_long_batch_size' must be an integer"
        in output
    )


def test_workflow_rejects_invalid_psichic_max_sequence_length_configfile_override(
    tmp_path: Path,
) -> None:
    bad_config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())
    bad_config["docking"]["psichic_max_sequence_length"] = 0
    config_path = tmp_path / "bad_psichic_max_sequence_length.yaml"
    config_path.write_text(yaml.safe_dump(bad_config))

    res = subprocess.run(
        [
            _snakemake_executable(),
            "-s",
            "workflow/Snakefile",
            "--cores",
            "1",
            "--configfile",
            str(config_path),
            "--config",
            "run_id=bad_psichic_max_sequence_length_probe",
            "compound_smiles=CCO",
            "-n",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert res.returncode != 0
    output = res.stdout + res.stderr
    assert (
        "Config value 'docking.psichic_max_sequence_length' must be an integer"
        in output
    )


def test_workflow_rejects_invalid_hardware_crop_radius_configfile_override(
    tmp_path: Path,
) -> None:
    bad_config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())
    bad_config["hardware"]["pocket_crop_radius"] = "not-a-number"
    config_path = tmp_path / "bad_hardware_crop_radius.yaml"
    config_path.write_text(yaml.safe_dump(bad_config))

    res = subprocess.run(
        [
            _snakemake_executable(),
            "-s",
            "workflow/Snakefile",
            "--cores",
            "1",
            "--configfile",
            str(config_path),
            "--config",
            "run_id=bad_hardware_crop_probe",
            "compound_smiles=CCO",
            "-n",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert res.returncode != 0
    output = res.stdout + res.stderr
    assert (
        "Config value 'hardware.pocket_crop_radius' must be a finite number"
        in output
    )


def test_workflow_rejects_invalid_boltz2_num_seeds_configfile_override(
    tmp_path: Path,
) -> None:
    bad_config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())
    bad_config["boltz2"]["num_seeds"] = "1.5"
    config_path = tmp_path / "bad_boltz2_num_seeds.yaml"
    config_path.write_text(yaml.safe_dump(bad_config))

    res = subprocess.run(
        [
            _snakemake_executable(),
            "-s",
            "workflow/Snakefile",
            "--cores",
            "1",
            "--configfile",
            str(config_path),
            "--config",
            "run_id=bad_boltz2_num_seeds_probe",
            "compound_smiles=CCO",
            "-n",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert res.returncode != 0
    output = res.stdout + res.stderr
    assert "Config value 'boltz2.num_seeds' must be an integer" in output


def test_workflow_rejects_invalid_stage0_fraction_configfile_override(
    tmp_path: Path,
) -> None:
    bad_config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())
    bad_config["stage0"]["p2rank_min_usable_fraction"] = "not-a-number"
    config_path = tmp_path / "bad_stage0_fraction.yaml"
    config_path.write_text(yaml.safe_dump(bad_config))

    res = subprocess.run(
        [
            _snakemake_executable(),
            "-s",
            "workflow/Snakefile",
            "--cores",
            "1",
            "--configfile",
            str(config_path),
            "--config",
            "run_id=bad_stage0_fraction_probe",
            "compound_smiles=CCO",
            "-n",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert res.returncode != 0
    output = res.stdout + res.stderr
    assert (
        "Config value 'stage0.p2rank_min_usable_fraction' "
        "must be a finite number"
    ) in output


def test_workflow_rejects_invalid_admet_vote_count_configfile_override(
    tmp_path: Path,
) -> None:
    bad_config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())
    bad_config["admet"]["skin_sens_halt_min_votes"] = "1.5"
    config_path = tmp_path / "bad_admet_vote_count.yaml"
    config_path.write_text(yaml.safe_dump(bad_config))

    res = subprocess.run(
        [
            _snakemake_executable(),
            "-s",
            "workflow/Snakefile",
            "--cores",
            "1",
            "--configfile",
            str(config_path),
            "--config",
            "run_id=bad_admet_vote_count_probe",
            "compound_smiles=CCO",
            "-n",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert res.returncode != 0
    output = res.stdout + res.stderr
    assert "Config value 'admet.skin_sens_halt_min_votes' must be an integer" in output


def test_workflow_rejects_invalid_drug_policy_configfile_override(
    tmp_path: Path,
) -> None:
    bad_config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())
    bad_config["drug_avoidance"]["drug_policy"] = "permissive"
    config_path = tmp_path / "bad_drug_policy.yaml"
    config_path.write_text(yaml.safe_dump(bad_config))

    res = subprocess.run(
        [
            _snakemake_executable(),
            "-s",
            "workflow/Snakefile",
            "--cores",
            "1",
            "--configfile",
            str(config_path),
            "--config",
            "run_id=bad_drug_policy_probe",
            "compound_smiles=CCO",
            "-n",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert res.returncode != 0
    output = res.stdout + res.stderr
    assert "Config value 'drug_avoidance.drug_policy' must be one of" in output


def test_workflow_rejects_unimplemented_bioemu_tica_option(tmp_path: Path) -> None:
    config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())
    config["bioemu"]["tICA_lag_steps"] = 10
    config_path = tmp_path / "unsupported_tica.yaml"
    config_path.write_text(yaml.safe_dump(config))
    result = subprocess.run(
        [
            _snakemake_executable(), "-s", "workflow/Snakefile", "--cores", "1",
            "--configfile", str(config_path), "--config",
            "run_id=unsupported_tica_probe", "compound_smiles=CCO", "-n",
        ],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert result.returncode != 0
    assert "bioemu.tICA_lag_steps is unsupported" in result.stdout + result.stderr


def test_workflow_rejects_invalid_bioemu_num_conformers_configfile_override(
    tmp_path: Path,
) -> None:
    bad_config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())
    bad_config["bioemu"]["num_conformers"] = "100.5"
    config_path = tmp_path / "bad_bioemu_num_conformers.yaml"
    config_path.write_text(yaml.safe_dump(bad_config))

    res = subprocess.run(
        [
            _snakemake_executable(),
            "-s",
            "workflow/Snakefile",
            "--cores",
            "1",
            "--configfile",
            str(config_path),
            "--config",
            "run_id=bad_bioemu_num_conformers_probe",
            "compound_smiles=CCO",
            "-n",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert res.returncode != 0
    output = res.stdout + res.stderr
    assert "Config value 'bioemu.num_conformers' must be an integer" in output


def test_workflow_rejects_invalid_md_duration_configfile_override(
    tmp_path: Path,
) -> None:
    bad_config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())
    bad_config["md"]["duration_ns"] = "not-a-number"
    config_path = tmp_path / "bad_md_duration.yaml"
    config_path.write_text(yaml.safe_dump(bad_config))

    res = subprocess.run(
        [
            _snakemake_executable(),
            "-s",
            "workflow/Snakefile",
            "--cores",
            "1",
            "--configfile",
            str(config_path),
            "--config",
            "run_id=bad_md_duration_probe",
            "compound_smiles=CCO",
            "-n",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert res.returncode != 0
    output = res.stdout + res.stderr
    assert "Config value 'md.duration_ns' must be a finite number" in output


def test_workflow_rejects_invalid_analog_mode_configfile_override(
    tmp_path: Path,
) -> None:
    bad_config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())
    bad_config["analog_gen"]["reinvent_mode"] = "freeform"
    config_path = tmp_path / "bad_analog_mode.yaml"
    config_path.write_text(yaml.safe_dump(bad_config))

    res = subprocess.run(
        [
            _snakemake_executable(),
            "-s",
            "workflow/Snakefile",
            "--cores",
            "1",
            "--configfile",
            str(config_path),
            "--config",
            "run_id=bad_analog_mode_probe",
            "compound_smiles=CCO",
            "-n",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert res.returncode != 0
    output = res.stdout + res.stderr
    assert "Config value 'analog_gen.reinvent_mode' must be one of" in output


def test_workflow_rejects_invalid_retrosynthesis_iterations_configfile_override(
    tmp_path: Path,
) -> None:
    bad_config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())
    bad_config["retrosynthesis"]["iteration_limit"] = "1.5"
    config_path = tmp_path / "bad_retrosynthesis_iterations.yaml"
    config_path.write_text(yaml.safe_dump(bad_config))

    res = subprocess.run(
        [
            _snakemake_executable(),
            "-s",
            "workflow/Snakefile",
            "--cores",
            "1",
            "--configfile",
            str(config_path),
            "--config",
            "run_id=bad_retrosynthesis_iterations_probe",
            "compound_smiles=CCO",
            "-n",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert res.returncode != 0
    output = res.stdout + res.stderr
    assert (
        "Config value 'retrosynthesis.iteration_limit' must be an integer"
        in output
    )


def test_workflow_rejects_invalid_mode_cli_override() -> None:
    res = subprocess.run(
        [
            _snakemake_executable(),
            "-s",
            "workflow/Snakefile",
            "--cores",
            "1",
            "--config",
            "run_id=bad_mode_probe",
            "compound_smiles=CCO",
            "mode=diagnostic",
            "-n",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert res.returncode != 0
    output = res.stdout + res.stderr
    assert "Config value 'mode' must be one of both, comprehensive, fast" in output


def test_stage11_publication_figures_declare_claim_sources() -> None:
    rule_text = (ROOT / "workflow" / "rules" / "stage11_publication.smk").read_text()

    assert '"target_landscape": rules.kg_efficacy_label.output.csv' in rule_text
    assert '"pharmacophore": rules.pharmacophore_consensus.output.json' in rule_text
    assert '"analogs": rules.mini_validate_funnel.output.csv' in rule_text
    assert '"md_trajectory": rules.gromacs_production.output.traj_index' in rule_text
    assert '"retrosynthesis": rules.score_routes.output.ranking' in rule_text
    assert 'S11_EVAL_DIR = PROJECT_ROOT / "results" / "eval"' in rule_text
    assert '"eval_manifest": S11_EVAL_DIR / "iteration_manifest.json"' in rule_text
    assert '"eval_retrospective": S11_EVAL_DIR / "cosmetic_retrospective.csv"' in rule_text
    assert "eval_dir = str(S11_EVAL_DIR)" in rule_text
    assert "--eval-dir {params.eval_dir:q}" in rule_text
    assert "S11_FIGURE_INPUTS = {} if S11_ALLOW_PLACEHOLDERS else" in rule_text
    assert "report = rules.molstar_report.output.html" in rule_text


def test_stage11_repro_pack_declares_required_artifact_dependencies() -> None:
    rule_text = (ROOT / "workflow" / "rules" / "stage11_publication.smk").read_text()

    assert "report = rules.molstar_report.output.html" in rule_text
    assert "fast_report_pointer = rules.molstar_report.output.fast_manifest" in rule_text
    assert "physics_report_pointer = rules.molstar_report.output.physics_manifest" in rule_text
    assert "consensus = rules.rrf_4way_consensus.output.top50" in rule_text
    assert "ranked_with_efficacy = rules.kg_efficacy_label.output.csv" in rule_text
    assert 'eval_manifest = S11_EVAL_DIR / "iteration_manifest.json"' in rule_text
    assert "--fast-report-pointer {input.fast_report_pointer:q}" in rule_text
    assert "--physics-report-pointer {input.physics_report_pointer:q}" in rule_text
    assert 'S11_REQUIRE_PHYSICS = MODE in ("comprehensive", "both")' in rule_text
    assert 'require_physics = "--require-physics-report"' in rule_text
    assert "--eval-manifest {input.eval_manifest:q}" in rule_text


def test_stage4_uses_skin_weighted_efficacy_targets_for_downstream_structure() -> None:
    rule_text = (ROOT / "workflow" / "rules" / "stage4_struct.smk").read_text()

    assert "return str(rules.kg_efficacy_label.output.csv)" in rule_text
    assert 'S3C_DIR / "top50_4way_consensus.csv"' not in rule_text
    assert 'S3F_DIR / "top50.csv"' not in rule_text


def test_stage9_report_receives_skin_aware_target_and_context_inputs() -> None:
    rule_text = (ROOT / "workflow" / "rules" / "stage9_report.smk").read_text()

    assert "return str(rules.kg_efficacy_label.output.csv)" in rule_text
    assert "cosing_json   = rules.cosing_match.output.json" in rule_text
    assert "drug_warnings = rules.drug_avoidance_match.output.json" in rule_text
    assert "cosmetic_decision = rules.cosmetic_drug_decision.output.decision" in rule_text
    assert "kg_efficacy   = rules.kg_efficacy_label.output.csv" in rule_text
    assert "screening_counts = _screening_count_inputs()" in rule_text
    assert "rules.daina_zoete.output.scores" in rule_text
    assert "rules.dti_rrf_fast.output.selected" in rule_text
    assert "rules.fast_autogrid_maps.output.manifest" in rule_text
    assert "rules.autodock_top5k.output.scores" in rule_text
    assert "rules.fast_gnina_pose_rescore.output.scores" in rule_text
    assert "rules.fast_rerank_consensus.output.canonical" in rule_text
    assert "rules.fast_rerank_consensus.output.top50" in rule_text
    assert "rules.autodock_gpu_all.output.scores" in rule_text
    assert "rules.autodock_pick_top_pct.output.top" in rule_text
    assert "rules.gnina_rescore_top.output.scores" in rule_text
    assert "rules.rtmscore_top.output.scores" in rule_text
    assert "rules.boltz2_affinity_top.output.scores" in rule_text
    assert "rules.rrf_4way_consensus.output.top50" in rule_text
    assert "--cosmetic-drug-decision {input.cosmetic_decision:q}" in rule_text
    assert "--cosing-json {input.cosing_json:q}" in rule_text
    assert "--drug-warnings-json {input.drug_warnings:q}" in rule_text
    assert "--kg-efficacy-csv {input.kg_efficacy:q}" in rule_text


def test_stage11_publication_rules_track_all_generated_outputs() -> None:
    rule_text = (ROOT / "workflow" / "rules" / "stage11_publication.smk").read_text()

    for name in (
        "fig01_workflow.svg",
        "fig08_case_study.png",
        "config_hash.txt",
        "random_seeds.json",
        "00_abstract.md",
        "05_references.bib",
    ):
        assert name in rule_text
    assert "**S11_FIGURE_OUTPUTS" in rule_text
    assert "**S11_REPRO_OUTPUTS" in rule_text
    assert "**S11_MANUSCRIPT_OUTPUTS" in rule_text
    assert "captions = rules.make_figures.output.captions" in rule_text
    assert "repro_manifest = rules.write_repro_pack.output.manifest" in rule_text
    assert "data_availability = rules.write_data_availability.output.statement" in rule_text
    assert "--captions {input.captions:q}" in rule_text
    assert "--repro-manifest {input.repro_manifest:q}" in rule_text
    assert "--data-availability {input.data_availability:q}" in rule_text


def test_stage11_publication_package_opt_in_is_all_target() -> None:
    snakefile = (ROOT / "workflow" / "Snakefile").read_text()
    stage11_rule = (ROOT / "workflow" / "rules" / "stage11_publication.smk").read_text()

    assert 'PUBLICATION   = config.get("publication", {})' in snakefile
    assert "def config_bool(value, default=False, name=None):" in snakefile
    assert '"publication.build_publication_package"' in snakefile
    assert '"publication.emit_manuscript_draft"' in snakefile
    assert 'config_bool(PUBLICATION.get("build_publication_package"), False)' in snakefile
    assert 'config_bool(PUBLICATION.get("emit_manuscript_draft"), False)' in snakefile
    assert '"publication" / "figures" / "captions.json"' in snakefile
    assert '"publication" / "data_availability.md"' in snakefile
    assert '"publication" / "reproducibility" / "artifact_manifest.json"' in snakefile
    assert '"publication" / "manuscript_draft" / "02_methods.md"' in snakefile
    assert 'S11_ALLOW_PLACEHOLDERS = config_bool(config.get("publication", {}).get("allow_placeholders"), False)' in stage11_rule
    assert 'S11_DRAFT_DOI_OK = config_bool(config.get("publication", {}).get("draft_doi_ok"), False)' in stage11_rule


def test_stage11_data_availability_requires_configured_doi() -> None:
    config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())
    rule_text = (ROOT / "workflow" / "rules" / "stage11_publication.smk").read_text()

    assert config["publication"]["skin_efficacy_kg_doi"] == "pending"
    assert config["publication"]["pipeline_source_url"] == "https://github.com/kangk1204/SkinScout_public"
    assert config["publication"]["draft_doi_ok"] is False
    assert 'skin_efficacy_kg_doi = str(config.get("publication", {}).get(' in rule_text
    assert 'pipeline_source_url = str(config.get("publication", {}).get(' in rule_text
    assert "--skin-efficacy-kg-doi {params.skin_efficacy_kg_doi:q}" in rule_text
    assert "--pipeline-source-url {params.pipeline_source_url:q}" in rule_text
    assert "--draft-doi-ok" in rule_text


def test_docking_boxes_path_is_the_pocket_derived_set() -> None:
    """The uniform 30 A cube carried no pocket geometry; the workflow must not use it."""
    config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())

    assert config["paths"]["docking_boxes"] == "data/docking_boxes_derived"


def test_no_script_hardcodes_a_docking_box_directory() -> None:
    """A hardcoded path lets one stage run against a different box set than the rest."""
    offenders: list[str] = []
    for directory in ("scripts", "eval", "workbench"):
        for path in sorted((ROOT / directory).rglob("*.py")):
            if "tests" in path.parts:
                continue
            for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if "data/docking_boxes" in line and "paths.docking_boxes" not in line:
                    offenders.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()}")

    assert offenders == [], (
        "resolve the docking box directory through workflow config "
        "paths.docking_boxes instead:\n" + "\n".join(offenders)
    )


def test_qualification_smoke_resolves_boxes_from_the_workflow_config() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from qualification_structural_smoke import docking_box_dir

    config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())
    expected = ROOT / config["paths"]["docking_boxes"]

    assert docking_box_dir() == expected


def test_the_comprehensive_path_builds_its_own_autogrid_maps() -> None:
    """AutoGrid existed only in the fast path.

    autodock_gpu_all had no .maps.fld inputs to find and failed closed at
    0/15038 coverage, so the comprehensive DAG could not run at all.
    """
    text = (ROOT / "workflow" / "rules" / "stage3a_comprehensive.smk").read_text()

    assert "rule comprehensive_receptor_list:" in text
    assert "rule comprehensive_autogrid_maps:" in text
    assert "--receptor-list {input.listing:q}" in text

    autodock = text.split("rule autodock_gpu_all:", 1)[1].split("rule ", 1)[0]
    assert "rules.comprehensive_autogrid_maps.output.manifest" in autodock
    assert "--map-manifest {input.map_manifest:q}" in autodock


def test_both_rescorers_read_the_docked_pose_in_comprehensive() -> None:
    text = (ROOT / "workflow" / "rules" / "stage3a_comprehensive.smk").read_text()
    for rule in ("gnina_rescore_top", "rtmscore_top"):
        body = text.split(f"rule {rule}:", 1)[1].split("\nrule ", 1)[0]
        assert "rules.autodock_gpu_all.output.poses" in body, rule
        assert "--pose-manifest" in body, rule
        assert "--allow-unscored-poses" in body, rule
        assert "xtb_optimize" not in body, rule


def test_skin_weighting_is_included_before_the_stage_three_paths() -> None:
    """stage3b_fast references rules.known_target_prior at parse time.

    With stage3_v3_skin_weight included last, Snakemake could not parse the
    workflow in any mode, so neither fast nor comprehensive could start.
    """
    text = (ROOT / "workflow" / "Snakefile").read_text()
    order = [
        text.index('include: "rules/stage3_v3_skin_weight.smk"'),
        text.index('include: "rules/stage3a_comprehensive.smk"'),
        text.index('include: "rules/stage3b_fast.smk"'),
    ]
    assert order == sorted(order)


def test_the_shared_stage_three_directories_are_defined_before_the_includes() -> None:
    """Moving the skin-weighting include first made the old placement circular."""
    text = (ROOT / "workflow" / "Snakefile").read_text()
    for name in ("S3C_DIR", "S3F_DIR"):
        assert f"\n{name} = RUN_DIR /" in text, name
        assert text.index(f"\n{name} = RUN_DIR /") < text.index(
            'include: "rules/stage3_v3_skin_weight.smk"'
        )
    comprehensive = (
        ROOT / "workflow" / "rules" / "stage3a_comprehensive.smk"
    ).read_text()
    for name in ("S3C_DIR", "S3F_DIR"):
        assert f"\n{name} = " not in comprehensive, f"{name} redefined"


def test_the_comprehensive_receptor_set_can_be_capped() -> None:
    """A full run is ~15k receptors; the cap exercises the same path cheaply."""
    config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())
    assert config["docking"]["comprehensive_max_receptors"] == 0

    snakefile = (ROOT / "workflow" / "Snakefile").read_text()
    assert '"docking.comprehensive_max_receptors"' in snakefile

    rules = (ROOT / "workflow" / "rules" / "stage3a_comprehensive.smk").read_text()
    body = rules.split("rule comprehensive_receptor_list:", 1)[1].split("\nrule ", 1)[0]
    assert "max_receptors = DOCKING[\"comprehensive_max_receptors\"]" in body
    # Zero must mean "every receptor", never "none".
    assert 'if [ "{params.max_receptors}" -gt 0 ]' in body
    assert "head -n {params.max_receptors}" in body
