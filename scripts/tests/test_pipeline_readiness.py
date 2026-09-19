"""Regression tests for aggregate pipeline readiness reporting."""

from __future__ import annotations

# A real single-molecule SDF: the runner now refuses inputs outside the
# documented scope, and an unparseable placeholder is one of them.
CAFFEINE_SDF = """
     RDKit          3D

 14 15  0  0  0  0  0  0  0  0999 V2000
   -1.3276    2.7758    0.3282 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.9139    1.3789    0.1374 N   0  0  0  0  0  0  0  0  0  0  0  0
    0.3962    1.1304    0.0685 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.2433    2.0523    0.1628 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.8165   -0.1834   -0.1118 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.1444   -1.1648   -0.2105 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.5328   -2.3350   -0.3796 N   0  0  0  0  0  0  0  0  0  0  0  0
    1.8531   -2.0769   -0.3838 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.0315   -0.7531   -0.2191 N   0  0  0  0  0  0  0  0  0  0  0  0
    3.2648    0.0113   -0.1560 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.4542   -0.8542   -0.1337 N   0  0  0  0  0  0  0  0  0  0  0  0
   -2.4583   -1.9024   -0.2397 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.8688    0.4349    0.0433 C   0  0  0  0  0  0  0  0  0  0  0  0
   -3.0933    0.7157    0.1139 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  1  0
  3  4  2  0
  3  5  1  0
  5  6  2  0
  6  7  1  0
  7  8  2  0
  8  9  1  0
  9 10  1  0
  6 11  1  0
 11 12  1  0
 11 13  1  0
 13 14  2  0
 13  2  1  0
  9  5  1  0
M  END
$$$$
"""


import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pandas as pd
import yaml
from Bio.SeqUtils.CheckSum import crc64

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipeline_readiness  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
READINESS = ROOT / "scripts/pipeline_readiness.py"


def run_readiness(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(READINESS), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def groups_by_name(payload: dict) -> dict[str, dict]:
    return {group["name"]: group for group in payload["groups"]}


def write_valid_discovery_alias_package(
    alias_dir: Path,
    direct_smiles: str,
) -> None:
    from scripts.tests.test_run_skinscout import (
        write_valid_discovery_alias_package as write_package,
    )

    write_package(alias_dir, direct_smiles)


def write_data_readiness_ok_tree(root: Path) -> None:
    data = root / "data"
    cosing = data / "cosing"
    cosing.mkdir(parents=True)
    pd.DataFrame([
        {
            "inci_name": f"Ingredient {idx}",
            "smiles": "CCO",
            "inchikey": f"KEY{idx}",
            "ecfp4": "fp",
            "scaffold_smiles": "CC",
        }
        for idx in range(100)
    ]).to_parquet(cosing / "cosing.parquet")

    drugs = data / "drug_avoidance"
    drugs.mkdir(parents=True)
    pd.DataFrame([
        {
            "drug_id": f"DRUG{idx}",
            "name": f"Drug {idx}",
            "smiles": "CCO",
            "inchikey": f"DRUGKEY{idx}",
            "ecfp4": "fp",
        }
        for idx in range(100)
    ]).to_parquet(drugs / "drugs.parquet")
    pd.DataFrame([
        {"scaffold_smiles": f"C{idx}", "n_drugs": 1}
        for idx in range(20)
    ]).to_parquet(drugs / "scaffolds.parquet")

    human_clean = data / "human_clean"
    human_clean.mkdir(parents=True)
    (human_clean / ".clean_complete").write_text("")
    human_pdbqt = data / "human_pdbqt"
    human_pdbqt.mkdir(parents=True)
    (human_pdbqt / ".pdbqt_complete").write_text("")
    (human_pdbqt / "P00001.pdbqt").write_text("REMARK ok\n")
    digest = "0" * 64
    (human_pdbqt / "receptor_prep_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "skinscout.stage0-receptor-pdbqt.v1",
                "scope": "full",
                "status": "ok",
                "policy": {
                    "min_success_fraction": 0.80,
                    "min_success_count": 1,
                },
                "counts": {"expected": 1, "success": 1, "failed": 0},
                "targets": [
                    {
                        "uniprot": "P00001",
                        "status": "ok",
                        "method": "meeko",
                        "reason": None,
                        "clean_pdb_sha256": digest,
                        "pocket_json_sha256": digest,
                    }
                ],
            }
        )
        + "\n"
    )
    mmseqs = data / "mmseqs"
    mmseqs.mkdir(parents=True)
    (human_clean / "P00001_clean.pdb").write_text("ATOM\n")
    fasta = mmseqs / "human_canonical.fasta"
    fasta_bytes = b">P00001\nACDEFG\n"
    fasta.write_bytes(fasta_bytes)
    (mmseqs / "human_canonical.fasta.manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "skinscout.afdb-v4-canonical-sequences.v1",
                "provenance": {
                    "source": (
                        "AlphaFold Protein Structure Database human proteome v4 "
                        "raw mmCIF files"
                    ),
                    "derivation": (
                        "Canonical sequences assembled offline from AFDB "
                        "fragment metadata and validated against the full "
                        "UniProt CRC64 embedded by AFDB; this is not an "
                        "independently downloaded UniProt FASTA snapshot."
                    ),
                    "alphafold_directory": str(
                        (data / "alphafold_human_v4").resolve()
                    ),
                    "source_pattern": "*.cif.gz",
                },
                "validation": {
                    "taxonomy_id": "9606",
                    "checksum_algorithm": "Bio.SeqUtils.CheckSum.crc64",
                    "fragment_policy": (
                        "all fragments; exact spans; equal overlaps; "
                        "gapless from residue 1"
                    ),
                    "unknown_residues_allowed": False,
                },
                "counts": {
                    "source_files": 1,
                    "canonical_sequences": 1,
                    "required_receptors": 1,
                    "required_receptors_covered": 1,
                },
                "required_receptor_coverage": {
                    "directory": str(human_clean.resolve()),
                    "missing_accessions": [],
                },
                "artifact": {
                    "path": str(fasta.resolve()),
                    "sha256": hashlib.sha256(fasta_bytes).hexdigest(),
                    "bytes": len(fasta_bytes),
                    "sequence_count": 1,
                },
                "sequences": [
                    {
                        "accession": "P00001",
                        "length": 6,
                        "sequence_sha256": hashlib.sha256(
                            b"ACDEFG"
                        ).hexdigest(),
                        "uniprot_crc64": crc64("ACDEFG").removeprefix("CRC-"),
                        "source_fragments": [
                            {
                                "path": "AF-P00001-F1-model_v4.cif.gz",
                                "compressed_sha256": "0" * 64,
                                "seq_db_align_begin": 1,
                                "seq_db_align_end": 6,
                            }
                        ],
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    # Track the configured directory rather than duplicating its name here; the
    # readiness check resolves it through workflow config paths.docking_boxes.
    boxes = root / yaml.safe_load(
        (ROOT / "workflow" / "config.yaml").read_text()
    )["paths"]["docking_boxes"]
    boxes.mkdir(parents=True)
    (boxes / "P00001_box.txt").write_text("0 0 0 10 10 10\n")
    (data / "no_pocket_targets.list").write_text("P00002\n")

    skin_expression = data / "skin_expression"
    skin_expression.mkdir(parents=True)
    pd.DataFrame([
        {"uniprot": f"P{idx:05d}", "skin_score": 0.5}
        for idx in range(1000)
    ]).to_csv(skin_expression / "skin_score.tsv", sep="\t", index=False)
    (skin_expression / "skin_score.tsv.axes.json").write_text(
        json.dumps(
            {
                "schema_version": "skinscout.skin-score-axes.v2",
                "score_semantics": (
                    "relative_within_build_expression_context"
                ),
                "normalization": (
                    "per-axis min-max across proteins in this build"
                ),
                "declared_weights": {
                    "hpa_tissue": 0.30,
                    "hpa_cell": 0.25,
                    "proteome": 0.20,
                    "gtex": 0.10,
                    "sc": 0.15,
                },
                "effective_weights": {
                    "hpa_tissue": 0.30,
                    "hpa_cell": 0.25,
                    "proteome": 0.20,
                    "gtex": 0.10,
                    "sc": 0.15,
                },
                "axes_absent": [],
                "context_support_threshold": 0.2,
            }
        )
        + "\n"
    )

    graph = nx.Graph()
    for idx in range(50):
        gene = f"gene:P{idx:05d}"
        efficacy = f"efficacy:{idx}"
        graph.add_node(gene, pmid_count=1)
        graph.add_node(efficacy)
        graph.add_edge(gene, efficacy)
    skin_kg = data / "skin_efficacy_kg"
    skin_kg.mkdir(parents=True)
    nx.write_graphml(graph, skin_kg / "skin_efficacy.graphml")


def test_pipeline_readiness_reports_generated_data_blockers_for_target_id(
    tmp_path: Path,
) -> None:
    res = run_readiness(
        [
            "--smiles",
            "CCO",
            "--run-id",
            "ethanol_case",
            "--preset",
            "target-id",
            "--repo-root",
            str(tmp_path),
            "--skip-safety-readiness",
            "--skip-model-readiness",
            "--json",
        ]
    )

    assert res.returncode == 2
    payload = json.loads(res.stdout)
    groups = groups_by_name(payload)

    assert payload["status"] == "failed"
    assert payload["input_type"] == "smiles"
    assert payload["input_smiles"] == "CCO"
    assert payload["input_canonical_smiles"] == "CCO"
    assert payload["input_sdf"] is None
    assert payload["canonical_smiles"] == "CCO"
    assert groups["input"]["input_type"] == "smiles"
    assert groups["input"]["input_smiles"] == "CCO"
    assert groups["input"]["input_canonical_smiles"] == "CCO"
    assert groups["input"]["input_sdf"] is None
    assert groups["input"]["status"] == "ok"
    assert groups["data:target-id"]["status"] == "failed"
    assert groups["stage0_sources"]["status"] == "ok"
    assert any(
        "run_skinscout.py --preset stage0" in action
        for action in payload["next_actions"]
    )
    assert "kg_efficacy_label" in payload["snakemake_command"]
    assert "compound_smiles=CCO" in payload["snakemake_command"]


def test_pipeline_readiness_blocks_on_stage0_claim_quality_after_data_passes(
    tmp_path: Path,
) -> None:
    write_data_readiness_ok_tree(tmp_path)

    res = run_readiness(
        [
            "--smiles",
            "CCO",
            "--run-id",
            "claim_quality_case",
            "--preset",
            "target-id",
            "--repo-root",
            str(tmp_path),
            "--skip-safety-readiness",
            "--skip-model-readiness",
            "--json",
        ]
    )

    assert res.returncode == 2
    payload = json.loads(res.stdout)
    groups = groups_by_name(payload)

    assert payload["status"] == "failed"
    assert groups["data:target-id"]["status"] == "ok"
    claim_quality = groups["stage0_claim_quality"]
    assert claim_quality["status"] == "failed"
    assert claim_quality["blocking"] is True
    blocker_labels = {blocker["label"] for blocker in claim_quality["blockers"]}
    assert "alphafold_count" in blocker_labels
    assert "claim_mmseqs_training_cutoff" in blocker_labels
    assert any(
        action == "python scripts/stage0_verify.py --strict --claim-quality"
        for action in payload["next_actions"]
    )
    assert "stage0_claim_quality status is ok" in (
        payload["output_contract"]["decision_gate"]
    )


def test_pipeline_readiness_autogenerates_run_id_from_smiles() -> None:
    res = run_readiness(
        [
            "--smiles",
            "CCO",
            "--preset",
            "target-id",
            "--skip-data-readiness",
            "--skip-stage0-source-readiness",
            "--skip-safety-readiness",
            "--skip-model-readiness",
            "--json",
        ]
    )

    assert res.returncode == 2, res.stderr
    payload = json.loads(res.stdout)
    assert payload["status"] == "readiness_blocked"
    assert groups_by_name(payload)["readiness_skips"]["blocking"] is True
    assert payload["input_type"] == "smiles"
    assert payload["input_smiles"] == "CCO"
    assert payload["input_canonical_smiles"] == "CCO"
    assert payload["input_sdf"] is None
    assert payload["run_id"].startswith("smiles_")
    input_group = groups_by_name(payload)["input"]
    assert input_group["run_id"] == payload["run_id"]
    assert input_group["input_type"] == "smiles"
    assert input_group["input_smiles"] == "CCO"
    assert input_group["input_canonical_smiles"] == "CCO"
    assert input_group["input_sdf"] is None
    assert f"run_id={payload['run_id']}" in payload["snakemake_command"]
    assert "compound_smiles=CCO" in payload["snakemake_command"]
    contract = payload["output_contract"]
    assert contract["run_dir"].endswith(f"results/runs/{payload['run_id']}")
    assert contract["summary_json"].endswith(f"{payload['run_id']}/run_summary.json")
    assert contract["summary_md"].endswith(f"{payload['run_id']}/run_summary.md")
    assert contract["verification_json"].endswith(
        f"{payload['run_id']}/run_verification.json"
    )
    assert contract["verification_log"].endswith(
        f"{payload['run_id']}/run_verification.log"
    )
    assert contract["html_report"] is None
    verified_artifacts = contract["verified_artifacts"]
    assert [item["name"] for item in verified_artifacts] == [
        "run_summary_json",
        "run_summary_md",
        "run_verification_log",
    ]
    assert all(item["checks"] == ["bytes", "sha256"] for item in verified_artifacts)
    assert all(
        item["recorded_in"] == "run_verification.json verified_artifacts"
        for item in verified_artifacts
    )
    assert "scripts/verify_run_outputs.py" in contract["verify_command"]
    assert "--preset target-id" in contract["verify_command"]
    assert "--target-metadata data/hpa/proteinatlas.tsv" in contract["verify_command"]
    assert "overall_decision.claimable" in contract["decision_gate"]
    assert "overall_decision.decision must be PASS" in contract["decision_gate"]
    assert (
        "overall_decision.recommended_action must be proceed"
        in contract["decision_gate"]
    )
    assert (
        "overall_decision.requires_human_review must be false"
        in contract["decision_gate"]
    )
    assert (
        "skin_specialized_binding.skin_context_supported"
        in contract["decision_gate"]
    )
    assert (
        "skin_specialized_binding.top_target_skin_context_supported"
        in contract["decision_gate"]
    )
    assert "review_before_claim requires human review" in contract["decision_gate"]
    assert "overall_decision.decision is PASS" in contract["claimable_when"]
    assert "overall_decision.claimable is true" in contract["claimable_when"]
    assert (
        "overall_decision.requires_human_review is false"
        in contract["claimable_when"]
    )
    assert (
        "skin_specialized_binding.skin_context_supported is true"
        in contract["claimable_when"]
    )
    assert (
        "skin_specialized_binding.top_target_skin_context_supported is true"
        in contract["claimable_when"]
    )
    assert "overall_decision.recommended_action is proceed" in contract["claimable_when"]
    assert "diagnostic_nonclaimable_reasons empty" in contract["claimable_when"]
    assert "stage0_claim_quality status is ok" in contract["decision_gate"]
    assert "stage0_claim_quality status is ok" in contract["claimable_when"]
    assert contract["diagnostic_nonclaimable_reasons"] == [
        "readiness report skipped preflight checks: data readiness, "
        "Stage 0 source readiness, safety readiness, model readiness"
    ]
    assert payload["next_actions"] == [
        "rerun readiness without diagnostic skip flags before treating this "
        "run as claimable: remove --skip-data-readiness, "
        "--skip-stage0-source-readiness, --skip-safety-readiness, "
        "--skip-model-readiness"
    ]
    assert "skin_toxicity" in contract["required_summary_sections"]
    assert "target_prediction" in contract["required_summary_sections"]
    assert "skin_specialized_binding" in contract["required_summary_sections"]
    fields = contract["required_summary_fields"]
    assert "input_type" in fields["compound"]
    assert "input_smiles" in fields["compound"]
    assert "input_canonical_smiles" in fields["compound"]
    assert "input_sdf" not in fields["compound"]
    assert "claimable" in fields["overall_decision"]
    assert "admet_metrics" in fields["safety"]
    assert "skin_sens_calls.husspred" in fields["safety"]
    assert "skin_sens_calls.stoptox" in fields["safety"]
    assert "skin_sens_calls.pred_skin" in fields["safety"]
    assert "skin_sens_evidence[].probability" in fields["safety"]
    assert "admet_metrics.AMES" in fields["safety"]
    assert "admet_metrics.Skin_Reaction" in fields["safety"]
    assert "admet_metrics.QED" in fields["safety"]
    assert "admet_risk_assessment" in fields["safety"]
    assert "skin_reaction_value" in fields["skin_toxicity"]
    assert "drug_policy" in fields["cosmetic_drug"]
    assert "admet_ai" in fields["artifacts"]
    assert "structural_alerts" in fields["artifacts"]
    assert "husspred" in fields["artifacts"]
    assert "stoptox" in fields["artifacts"]
    assert "pred_skin" in fields["artifacts"]
    assert "target_comprehensive_autodock" in fields["artifacts"]
    assert "target_comprehensive_gnina" in fields["artifacts"]
    assert "target_comprehensive_consensus" in fields["artifacts"]
    assert "screened_target_count" in fields["target_prediction"]
    assert "screening_counts" in fields["target_prediction"]
    assert "top_targets[].target_id" in fields["target_prediction"]
    assert "top_targets[].final_score" in fields["target_prediction"]
    assert "top_targets[].sources" in fields["target_prediction"]
    assert "top_targets[].efficacy" in fields["target_prediction"]
    launcher_fields = contract["required_launcher_result_fields"]
    assert "input_type" in launcher_fields
    assert "input_smiles" in launcher_fields
    assert "input_canonical_smiles" in launcher_fields
    assert "input_sdf" not in launcher_fields
    assert "decision_reasons" in launcher_fields
    assert "requires_human_review" in launcher_fields
    assert "skin_toxicity_level" in launcher_fields
    assert "skin_reaction_value" in launcher_fields
    assert "structural_alerts_present" in launcher_fields
    assert "structural_alert_flags" in launcher_fields
    assert "safety_evidence_degraded" in launcher_fields
    assert "missing_skin_sens_models" in launcher_fields
    assert "skin_sens_<model>_status" in launcher_fields
    assert "top_target" in launcher_fields
    assert "high_admet_risk_endpoints" in launcher_fields
    assert "moderate_admet_risk_endpoints" in launcher_fields
    assert "drug_policy" in launcher_fields
    assert "drug_warnings" in launcher_fields
    assert "screened_target_count" in launcher_fields
    assert "screening_<stage>" in launcher_fields
    assert "ranked_targets" in launcher_fields
    assert "ranked_target_<rank>" in launcher_fields
    assert "ranked_target_<rank>_final_score" in launcher_fields
    assert "ranked_target_<rank>_docking_rrf" in launcher_fields
    assert "ranked_target_<rank>_source_count" in launcher_fields
    assert "ranked_target_<rank>_sources" in launcher_fields
    assert "ranked_target_<rank>_skin_score" in launcher_fields
    assert "ranked_target_<rank>_skin_tier" in launcher_fields
    assert "ranked_target_<rank>_efficacy" in launcher_fields
    assert "skin_context" in launcher_fields
    assert "skin_context_supported" in launcher_fields
    assert "skin_expression_supported" in launcher_fields
    assert "skin_efficacy_supported" in launcher_fields
    assert "top_target_final_score" in launcher_fields
    assert "top_target_docking_rrf" in launcher_fields
    assert "top_target_source_count" in launcher_fields
    assert "top_target_sources" in launcher_fields
    assert "top_target_skin_score" in launcher_fields
    assert "top_target_skin_tier" in launcher_fields
    assert "top_target_gene_symbol" in launcher_fields
    assert "top_target_protein_name" in launcher_fields
    assert "top_target_skin_expression_supported" in launcher_fields
    assert "top_target_skin_efficacy_supported" in launcher_fields
    assert "top_target_skin_context_supported" in launcher_fields
    assert "top_targets_with_skin_efficacy" in launcher_fields
    assert "top_target_skin_efficacy" in launcher_fields
    assert "most_skin_relevant_gene_symbol" in launcher_fields
    assert "most_skin_relevant_protein_name" in launcher_fields
    assert "most_skin_relevant_final_score" in launcher_fields
    assert "most_skin_relevant_docking_rrf" in launcher_fields
    assert "most_skin_relevant_source_count" in launcher_fields
    assert "most_skin_relevant_sources" in launcher_fields
    assert "most_skin_relevant_skin_score" in launcher_fields
    assert "most_skin_relevant_skin_tier" in launcher_fields
    assert "most_skin_relevant_efficacy" in launcher_fields
    assert "verification_checks" in launcher_fields
    assert "verified_artifacts" in launcher_fields
    assert "source_artifacts" in launcher_fields
    assert "target_source_artifacts" in launcher_fields
    assert "artifact_<key>" in launcher_fields
    assert "skin_context_decision" in fields["skin_specialized_binding"]
    assert "top_target_skin_score" in fields["skin_specialized_binding"]
    assert "top_target_final_score" in fields["skin_specialized_binding"]
    assert "top_target_docking_rrf" in fields["skin_specialized_binding"]
    assert "top_target_source_count" in fields["skin_specialized_binding"]
    assert "top_target_sources" in fields["skin_specialized_binding"]
    assert "top_target_gene_symbol" in fields["skin_specialized_binding"]
    assert "top_target_protein_name" in fields["skin_specialized_binding"]
    assert "top_target_skin_expression_supported" in fields["skin_specialized_binding"]
    assert "top_target_skin_efficacy_supported" in fields["skin_specialized_binding"]
    assert "top_target_skin_context_supported" in fields["skin_specialized_binding"]
    assert "most_skin_relevant_target.gene_symbol" in fields["skin_specialized_binding"]
    assert "most_skin_relevant_target.protein_name" in fields["skin_specialized_binding"]
    assert "most_skin_relevant_target.final_score" in fields["skin_specialized_binding"]
    assert "most_skin_relevant_target.sources" in fields["skin_specialized_binding"]
    assert "most_skin_relevant_target.efficacy" in fields["skin_specialized_binding"]
    assert "02_admet/admet_ai.json" in contract["required_source_artifacts"]
    assert "02_admet/husspred.json" in contract["required_source_artifacts"]
    assert "02_admet/stoptox.json" in contract["required_source_artifacts"]
    assert "02_admet/pred_skin.json" in contract["required_source_artifacts"]
    invariants = contract["source_evidence_invariants"]
    assert any(
        "source smiles must canonical-match" in invariant
        and "compound_canonical.json" in invariant
        for invariant in invariants
    )
    assert any(
        "canonical_smiles must equal RDKit canonical SMILES" in invariant
        for invariant in invariants
    )
    assert any(
        "inchikey must equal RDKit InChIKey for canonical_smiles" in invariant
        for invariant in invariants
    )
    assert any(
        "must agree with 02_admet/admet_report.json" in invariant
        for invariant in invariants
    )
    target_invariants = contract["target_evidence_invariants"]
    assert any("nonblank unique target_id" in invariant for invariant in target_invariants)
    assert any("final_score must be sorted descending" in invariant for invariant in target_invariants)
    assert any("source_count must be an integer >= 3" in invariant for invariant in target_invariants)
    assert any("efficacy_top* KG evidence" in invariant for invariant in target_invariants)
    assert any(
        "target_prediction and skin_specialized_binding fields must match"
        in invariant
        for invariant in target_invariants
    )
    assert any(
        "target_prediction.screening_counts" in invariant
        for invariant in target_invariants
    )
    assert any(
        "skin_specialized_binding.skin_context_supported true" in invariant
        for invariant in target_invariants
    )
    assert any(
        "top_target_skin_context_supported true" in invariant
        for invariant in target_invariants
    )
    assert any("four-way consensus evidence" in invariant for invariant in target_invariants)
    assert "03_targets/mode_comprehensive/autodock_all_targets.tsv" in (
        contract["required_source_artifacts"]
    )
    assert "03_targets/mode_comprehensive/top50_4way_consensus.csv" in (
        contract["required_source_artifacts"]
    )
    assert "03_targets/ranked_targets_v3_with_efficacy.csv" in (
        contract["required_source_artifacts"]
    )


def test_pipeline_readiness_stage0_builds_snakemake_command_without_input() -> None:
    res = run_readiness(
        [
            "--preset",
            "stage0",
            "--skip-data-readiness",
            "--skip-stage0-source-readiness",
            "--skip-safety-readiness",
            "--skip-model-readiness",
            "--json",
        ]
    )

    assert res.returncode == 2, res.stderr
    payload = json.loads(res.stdout)
    assert payload["status"] == "readiness_blocked"
    assert payload["run_id"] == "stage0_bootstrap"
    assert payload["input_smiles"] == "C"
    assert "stage0_complete" in payload["snakemake_command"]
    assert "compound_smiles=C" in payload["snakemake_command"]


def test_pipeline_readiness_normalizes_sota_alias_for_snakemake_command() -> None:
    res = run_readiness(
        [
            "--smiles",
            "CCO",
            "--preset",
            "target-id-sota",
            "--context-profile",
            "barrier",
            "--skip-data-readiness",
            "--skip-stage0-source-readiness",
            "--skip-safety-readiness",
            "--skip-model-readiness",
            "--json",
        ]
    )

    assert res.returncode == 2, res.stderr
    payload = json.loads(res.stdout)
    assert payload["status"] == "readiness_blocked"
    assert payload["preset"] == "target-id"
    assert payload["requested_preset"] == "target-id-sota"
    assert payload["sota_claim"] is True
    assert payload["run_profile"] == {
        "requested_preset": "target-id-sota",
        "normalized_preset": "target-id",
        "execution_mode": "comprehensive",
        "analysis_profile": "target_comprehensive",
        "evidence_mode": "evidence",
        "context_profile": "barrier",
        "sota_claim": True,
    }
    assert payload["run_manifest_v2"]["schema_version"] == (
        "skinscout.run_manifest.v2"
    )
    assert payload["run_manifest_v2"]["run_profile"] == payload["run_profile"]
    assert len(payload["run_manifest_v2"]["hashes"]["data_sha256"]) == 64
    assert "kg_efficacy_label" in payload["snakemake_command"]
    assert "evaluation={" in payload["snakemake_command"]
    assert "enabled" in payload["snakemake_command"]
    assert "true" in payload["snakemake_command"]
    assert "default_context_profile" in payload["snakemake_command"]
    assert "barrier" in payload["snakemake_command"]
    assert "skin_weight=0.40" in payload["snakemake_command"]


def test_pipeline_readiness_passes_explicit_sota_claim_to_snakemake_command() -> None:
    res = run_readiness(
        [
            "--smiles",
            "CCO",
            "--preset",
            "report",
            "--sota-claim",
            "--context-profile",
            "pigmentation",
            "--skip-data-readiness",
            "--skip-stage0-source-readiness",
            "--skip-safety-readiness",
            "--skip-model-readiness",
            "--json",
        ]
    )

    assert res.returncode == 2, res.stderr
    payload = json.loads(res.stdout)
    assert payload["preset"] == "report"
    assert payload["requested_preset"] == "report"
    assert payload["sota_claim"] is True
    assert " all " in f" {payload['snakemake_command']} "
    assert "evaluation={" in payload["snakemake_command"]
    assert "enabled" in payload["snakemake_command"]
    assert "true" in payload["snakemake_command"]
    assert "default_context_profile" in payload["snakemake_command"]
    assert "pigmentation" in payload["snakemake_command"]


def test_pipeline_readiness_discovery_mode_fails_closed_without_alias_package(
    tmp_path: Path,
) -> None:
    res = run_readiness(
        [
            "--smiles",
            "CCO",
            "--preset",
            "target-id",
            "--mode",
            "fast",
            "--evidence-mode",
            "discovery",
            "--discovery-alias-dir",
            str(tmp_path / "missing-alias-package"),
            "--skip-data-readiness",
            "--skip-stage0-source-readiness",
            "--skip-safety-readiness",
            "--skip-model-readiness",
            "--json",
        ]
    )

    assert res.returncode == 2
    payload = json.loads(res.stdout)
    assert payload["status"] == "failed"
    group = groups_by_name(payload)["discovery-aliases"]
    assert group["blocking"] is True
    assert group["blockers"][0]["label"] == "discovery_alias_manifest"
    assert payload["run_profile"]["evidence_mode"] == "discovery"


def test_pipeline_readiness_discovery_mode_accepts_verified_alias_package(
    tmp_path: Path,
) -> None:
    alias_dir = tmp_path / "aliases"
    write_valid_discovery_alias_package(alias_dir, "CCC")
    res = run_readiness(
        [
            "--smiles",
            "CCO",
            "--preset",
            "target-id",
            "--mode",
            "fast",
            "--evidence-mode",
            "discovery",
            "--discovery-alias-dir",
            str(alias_dir),
            "--skip-data-readiness",
            "--skip-stage0-source-readiness",
            "--skip-safety-readiness",
            "--skip-model-readiness",
            "--json",
        ]
    )

    assert res.returncode == 2
    payload = json.loads(res.stdout)
    group = groups_by_name(payload)["discovery-aliases"]
    assert group["blocking"] is False
    assert group["status"] == "ok"
    assert groups_by_name(payload)["readiness_skips"]["blocking"] is True


def test_pipeline_readiness_discovery_rejects_stale_upstream_manifest(
    tmp_path: Path,
) -> None:
    alias_dir = tmp_path / "aliases"
    write_valid_discovery_alias_package(alias_dir, "CCC")
    upstream = alias_dir / "upstream" / "chembl_manifest.json"
    upstream.write_text(upstream.read_text() + "\n")

    res = run_readiness([
        "--smiles",
        "CCO",
        "--preset",
        "target-id",
        "--mode",
        "fast",
        "--evidence-mode",
        "discovery",
        "--discovery-alias-dir",
        str(alias_dir),
        "--skip-data-readiness",
        "--skip-stage0-source-readiness",
        "--skip-safety-readiness",
        "--skip-model-readiness",
        "--json",
    ])

    assert res.returncode == 2
    payload = json.loads(res.stdout)
    group = groups_by_name(payload)["discovery-aliases"]
    assert group["blocking"] is True
    assert "upstream manifest SHA-256 drift" in group["blockers"][0]["detail"]


def test_pipeline_readiness_preserves_raw_input_smiles() -> None:
    res = run_readiness(
        [
            "--smiles",
            "C(C)O",
            "--preset",
            "target-id",
            "--skip-data-readiness",
            "--skip-stage0-source-readiness",
            "--skip-safety-readiness",
            "--skip-model-readiness",
            "--json",
        ]
    )

    assert res.returncode == 2, res.stderr
    payload = json.loads(res.stdout)
    input_group = groups_by_name(payload)["input"]
    assert payload["input_type"] == "smiles"
    assert payload["input_smiles"] == "C(C)O"
    assert payload["input_canonical_smiles"] == "CCO"
    assert payload["canonical_smiles"] == "CCO"
    assert input_group["input_smiles"] == "C(C)O"
    assert input_group["input_canonical_smiles"] == "CCO"
    assert "compound_smiles=C(C)O" in payload["snakemake_command"]


def test_pipeline_readiness_accepts_positional_smiles() -> None:
    res = run_readiness(
        [
            "CCO",
            "--preset",
            "target-id",
            "--skip-data-readiness",
            "--skip-stage0-source-readiness",
            "--skip-safety-readiness",
            "--skip-model-readiness",
            "--json",
        ]
    )

    assert res.returncode == 2, res.stderr
    payload = json.loads(res.stdout)
    assert payload["status"] == "readiness_blocked"
    assert payload["input_type"] == "smiles"
    assert payload["input_smiles"] == "CCO"
    assert payload["input_canonical_smiles"] == "CCO"
    assert payload["canonical_smiles"] == "CCO"
    assert payload["run_id"].startswith("smiles_")
    assert "compound_smiles=CCO" in payload["snakemake_command"]
    assert payload["output_contract"]["summary_json"].endswith(
        f"{payload['run_id']}/run_summary.json"
    )


def test_pipeline_readiness_rejects_positional_smiles_with_flagged_input() -> None:
    res = run_readiness(
        [
            "CCO",
            "--smiles",
            "CCC",
            "--skip-data-readiness",
            "--skip-stage0-source-readiness",
            "--skip-safety-readiness",
            "--skip-model-readiness",
            "--json",
        ]
    )

    assert res.returncode == 2
    assert "positional SMILES cannot be combined" in res.stderr


def test_pipeline_readiness_marks_missing_generated_data_buildable_when_sources_ok(
    tmp_path: Path,
) -> None:
    res = run_readiness(
        [
            "--smiles",
            "CCO",
            "--run-id",
            "ethanol_case",
            "--preset",
            "target-id",
            "--repo-root",
            str(tmp_path),
            "--allow-stage0-build",
            "--skip-safety-readiness",
            "--skip-model-readiness",
            "--json",
        ]
    )

    assert res.returncode == 2
    payload = json.loads(res.stdout)
    groups = groups_by_name(payload)

    assert payload["status"] == "readiness_blocked"
    assert groups["data:target-id"]["status"] == "buildable"
    assert groups["data:target-id"]["blocking"] is False
    assert all(
        "[missing]" in blocker["label"]
        for blocker in groups["data:target-id"]["blockers"]
    )
    assert not any(
        "import_stage0_sources.py" in action
        for action in payload["next_actions"]
    )
    assert any(
        "launch with --allow-stage0-build" in action
        for action in payload["next_actions"]
    )
    assert groups["stage0_sources"]["status"] == "ok"


def test_pipeline_readiness_invalid_smiles_keeps_aggregate_json() -> None:
    res = run_readiness(
        [
            "--smiles",
            "not a smiles",
            "--run-id",
            "bad_smiles_case",
            "--skip-data-readiness",
            "--skip-stage0-source-readiness",
            "--skip-safety-readiness",
            "--skip-model-readiness",
            "--json",
        ]
    )

    assert res.returncode == 2
    payload = json.loads(res.stdout)
    groups = groups_by_name(payload)

    assert payload["status"] == "failed"
    assert payload["input_type"] == "smiles"
    assert payload["input_smiles"] == "not a smiles"
    assert payload["input_canonical_smiles"] is None
    assert payload["snakemake_command"] is None
    assert payload["output_contract"] is None
    assert groups["input"]["status"] == "failed"
    assert "Could not parse SMILES" in groups["input"]["blockers"][0]["detail"]


def test_pipeline_readiness_text_output_lists_next_actions(tmp_path: Path) -> None:
    res = run_readiness(
        [
            "--smiles",
            "CCO",
            "--run-id",
            "ethanol_case",
            "--preset",
            "target-id",
            "--repo-root",
            str(tmp_path),
            "--skip-safety-readiness",
            "--skip-model-readiness",
        ]
    )

    assert res.returncode == 2
    assert "SkinScout pipeline readiness: failed" in res.stdout
    assert "input_type=smiles" in res.stdout
    assert "input_smiles=CCO" in res.stdout
    assert "input_canonical_smiles=CCO" in res.stdout
    assert "canonical_smiles=CCO" in res.stdout
    assert "Next actions:" in res.stdout
    assert "Expected completed-run contract:" in res.stdout
    assert "run_summary.json" in res.stdout
    assert "run_verification.json" in res.stdout
    assert "run_verification.log" in res.stdout
    assert "verified artifacts:" in res.stdout
    assert "run_summary_json:" in res.stdout
    assert "sha256" in res.stdout
    assert "remove --skip-safety-readiness, --skip-model-readiness" in res.stdout
    assert "scripts/verify_run_outputs.py" in res.stdout
    assert "decision gate:" in res.stdout
    assert "overall_decision.decision must be PASS" in res.stdout
    assert "overall_decision.recommended_action must be proceed" in res.stdout
    assert "overall_decision.requires_human_review must be false" in res.stdout
    assert "stage0_claim_quality status is ok" in res.stdout
    assert "review_before_claim requires human review" in res.stdout
    assert "claimable when:" in res.stdout
    assert "overall_decision.decision is PASS" in res.stdout
    assert "overall_decision.requires_human_review is false" in res.stdout
    assert "diagnostic non-claimable reasons:" in res.stdout
    assert "readiness report skipped preflight checks: safety readiness, model readiness" in (
        res.stdout
    )
    assert "skin_toxicity" in res.stdout
    assert "required safety fields:" in res.stdout
    assert "admet_metrics" in res.stdout
    assert "skin_sens_calls.husspred" in res.stdout
    assert "skin_sens_evidence[].probability" in res.stdout
    assert "admet_metrics.Skin_Reaction" in res.stdout
    assert "required target_prediction fields:" in res.stdout
    assert "top_targets[].target_id" in res.stdout
    assert "screened_target_count" in res.stdout
    assert "screening_counts" in res.stdout
    assert "required launcher result fields:" in res.stdout
    assert "input_type" in res.stdout
    assert "input_smiles" in res.stdout
    assert "input_canonical_smiles" in res.stdout
    assert "skin_toxicity_level" in res.stdout
    assert "skin_reaction_value" in res.stdout
    assert "structural_alerts_present" in res.stdout
    assert "safety_evidence_degraded" in res.stdout
    assert "skin_sens_<model>_status" in res.stdout
    assert "high_admet_risk_endpoints" in res.stdout
    assert "moderate_admet_risk_endpoints" in res.stdout
    assert "drug_policy" in res.stdout
    assert "screening_<stage>" in res.stdout
    assert "ranked_target_<rank>" in res.stdout
    assert "ranked_target_<rank>_final_score" in res.stdout
    assert "ranked_target_<rank>_docking_rrf" in res.stdout
    assert "ranked_target_<rank>_source_count" in res.stdout
    assert "ranked_target_<rank>_sources" in res.stdout
    assert "ranked_target_<rank>_efficacy" in res.stdout
    assert "skin_context" in res.stdout
    assert "skin_context_supported" in res.stdout
    assert "skin_expression_supported" in res.stdout
    assert "skin_efficacy_supported" in res.stdout
    assert "top_target_final_score" in res.stdout
    assert "top_target_docking_rrf" in res.stdout
    assert "top_target_source_count" in res.stdout
    assert "top_target_sources" in res.stdout
    assert "top_target_skin_score" in res.stdout
    assert "top_target_skin_tier" in res.stdout
    assert "top_target_skin_expression_supported" in res.stdout
    assert "top_target_skin_efficacy_supported" in res.stdout
    assert "top_target_skin_context_supported" in res.stdout
    assert "top_targets_with_skin_efficacy" in res.stdout
    assert "top_target_skin_efficacy" in res.stdout
    assert "most_skin_relevant_gene_symbol" in res.stdout
    assert "most_skin_relevant_protein_name" in res.stdout
    assert "target_source_artifacts" in res.stdout
    assert "source_artifacts" in res.stdout
    assert "artifact_<key>" in res.stdout
    assert "required skin_specialized_binding fields:" in res.stdout
    assert "skin_context_decision" in res.stdout
    assert "top_target_final_score" in res.stdout
    assert "top_target_sources" in res.stdout
    assert "most_skin_relevant_target.gene_symbol" in res.stdout
    assert "most_skin_relevant_target.protein_name" in res.stdout
    assert "required source artifacts:" in res.stdout
    assert "02_admet/admet_ai.json" in res.stdout
    assert "02_admet/husspred.json" in res.stdout
    assert "source evidence invariants:" in res.stdout
    assert "target evidence invariants:" in res.stdout
    assert "canonical_smiles must equal RDKit canonical SMILES" in res.stdout
    assert "inchikey must equal RDKit InChIKey for canonical_smiles" in res.stdout
    assert "source smiles must canonical-match" in res.stdout
    assert "02_admet/admet_report.json" in res.stdout
    assert "nonblank unique target_id" in res.stdout
    assert "source_count must be an integer >= 3" in res.stdout
    assert "target_prediction and skin_specialized_binding fields must match" in (
        res.stdout
    )
    assert "target_prediction.screening_counts" in res.stdout
    assert "top_target_skin_context_supported true" in res.stdout
    assert "03_targets/mode_comprehensive/top50_4way_consensus.csv" in res.stdout
    assert "03_targets/ranked_targets_v3_with_efficacy.csv" in res.stdout
    assert "required report markers:" not in res.stdout
    assert "python scripts/run_skinscout.py --preset stage0" in res.stdout


def test_pipeline_readiness_fast_contract_includes_fast_target_evidence() -> None:
    res = run_readiness(
        [
            "--smiles",
            "CCO",
            "--run-id",
            "ethanol_fast_contract_case",
            "--preset",
            "target-id",
            "--mode",
            "fast",
            "--skip-data-readiness",
            "--skip-stage0-source-readiness",
            "--skip-safety-readiness",
            "--skip-model-readiness",
            "--json",
        ]
    )

    assert res.returncode == 2, res.stderr
    payload = json.loads(res.stdout)
    contract = payload["output_contract"]
    fields = contract["required_summary_fields"]
    assert "target_fast_daina_selected" in fields["artifacts"]
    assert "target_fast_autogrid_manifest" in fields["artifacts"]
    assert "target_fast_gnina_pose" in fields["artifacts"]
    assert "target_fast_daina_structural_targets" in fields["artifacts"]
    assert "target_fast_rerank_consensus" in fields["artifacts"]
    assert "target_comprehensive_consensus" not in fields["artifacts"]
    assert any(
        "source_count >= 1" in invariant
        for invariant in contract["target_evidence_invariants"]
    )
    assert any(
        "fixed Daina primary set" in invariant
        for invariant in contract["target_evidence_invariants"]
    )
    assert "03_targets/mode_fast/daina_structural_targets.csv" in (
        contract["required_source_artifacts"]
    )
    assert "03_targets/mode_fast/top50.csv" in contract["required_source_artifacts"]
    assert "03_targets/mode_comprehensive/top50_4way_consensus.csv" not in (
        contract["required_source_artifacts"]
    )


def test_pipeline_readiness_degraded_safety_is_nonclaimable_contract() -> None:
    res = run_readiness(
        [
            "--smiles",
            "CCO",
            "--run-id",
            "ethanol_degraded_contract_case",
            "--preset",
            "safety",
            "--allow-safety-degraded",
            "--skip-data-readiness",
            "--skip-stage0-source-readiness",
            "--skip-safety-readiness",
            "--skip-model-readiness",
            "--json",
        ]
    )

    assert res.returncode == 2, res.stderr
    payload = json.loads(res.stdout)
    contract = payload["output_contract"]
    assert contract["diagnostic_nonclaimable_reasons"] == [
        "readiness report skipped preflight checks: data readiness, "
        "Stage 0 source readiness, safety readiness, model readiness",
        "explicit degraded ADMET/skin-sens evidence was allowed",
    ]
    assert payload["next_actions"] == [
        "rerun readiness without diagnostic skip flags before treating this "
        "run as claimable: remove --skip-data-readiness, "
        "--skip-stage0-source-readiness, --skip-safety-readiness, "
        "--skip-model-readiness",
        "rerun readiness and launcher without --allow-safety-degraded before "
        "treating ADMET/skin-sens output as claimable",
    ]
    assert "--allow-degraded" in contract["verify_command"]
    assert "overall_decision.decision is PASS" in contract["claimable_when"]
    assert "overall_decision.claimable is true" in contract["claimable_when"]
    assert (
        "overall_decision.requires_human_review is false"
        in contract["claimable_when"]
    )
    assert (
        "skin_specialized_binding.top_target_skin_context_supported"
        not in contract["decision_gate"]
    )
    assert (
        "skin_specialized_binding.top_target_skin_context_supported"
        not in contract["claimable_when"]
    )
    assert contract["target_evidence_invariants"] == []
    assert "diagnostic_nonclaimable_reasons empty" in contract["claimable_when"]


def test_pipeline_readiness_report_contract_includes_html_report() -> None:
    res = run_readiness(
        [
            "--smiles",
            "CCO",
            "--run-id",
            "ethanol_report_case",
            "--preset",
            "report",
            "--skip-data-readiness",
            "--skip-stage0-source-readiness",
            "--skip-safety-readiness",
            "--skip-model-readiness",
            "--json",
        ]
    )

    assert res.returncode == 2, res.stderr
    payload = json.loads(res.stdout)
    contract = payload["output_contract"]
    assert contract["run_dir"].endswith("results/runs/ethanol_report_case")
    assert contract["verification_json"].endswith(
        "ethanol_report_case/run_verification.json"
    )
    assert contract["verification_log"].endswith(
        "ethanol_report_case/run_verification.log"
    )
    assert contract["html_report"].endswith("ethanol_report_case/09_report/index.html")
    verified_artifacts = contract["verified_artifacts"]
    assert [item["name"] for item in verified_artifacts] == [
        "run_summary_json",
        "run_summary_md",
        "run_verification_log",
        "html_report",
    ]
    assert verified_artifacts[-1]["path"].endswith(
        "ethanol_report_case/09_report/index.html"
    )
    assert "html_report" not in contract["required_summary_sections"]
    assert "html_report" in contract["required_summary_fields"]["artifacts"]
    assert "09_report/index.html" in contract["required_source_artifacts"]
    assert "Input type" in contract["required_report_markers"]
    assert "Input SMILES" in contract["required_report_markers"]
    assert "Input canonical SMILES" in contract["required_report_markers"]
    assert "Input SDF" not in contract["required_report_markers"]
    assert "InChIKey" in contract["required_report_markers"]
    assert "canonical SMILES" in contract["required_report_markers"]
    assert "Claimable" in contract["required_report_markers"]
    assert "Claim status" in contract["required_report_markers"]
    assert "Skin_Reaction" in contract["required_report_markers"]
    assert "Structural alerts" in contract["required_report_markers"]
    assert "degraded skin-sens evidence" in contract["required_report_markers"]
    assert "missing skin-sens models" in contract["required_report_markers"]
    assert "Cosmetic/drug decision" in contract["required_report_markers"]
    assert "Moderate ADMET risk endpoints" in contract["required_report_markers"]
    assert "Summary ADMET metrics" in contract["required_report_markers"]
    assert "Skin-sens 3-model evidence" in contract["required_report_markers"]
    assert "Skin expression supported" in contract["required_report_markers"]
    assert "Skin efficacy supported" in contract["required_report_markers"]
    assert "Screened target candidates" in contract["required_report_markers"]
    assert "Screening stage counts" in contract["required_report_markers"]
    assert "top target gene/protein labels when available" in (
        contract["required_report_markers"]
    )
    assert "top target final score" in contract["required_report_markers"]
    assert "top target docking RRF when available" in (
        contract["required_report_markers"]
    )
    assert "top target source count" in contract["required_report_markers"]
    assert "top target sources" in contract["required_report_markers"]
    assert "top target skin score" in contract["required_report_markers"]
    assert "top target skin tier" in contract["required_report_markers"]
    assert "top target skin efficacy" in contract["required_report_markers"]
    assert "top target skin context supported" in contract["required_report_markers"]
    assert "Most skin-relevant top target" in contract["required_report_markers"]
    assert "most skin-relevant gene/protein labels when available" in (
        contract["required_report_markers"]
    )
    assert "most skin-relevant score/source/skin-efficacy evidence" in (
        contract["required_report_markers"]
    )
    assert "top target row details from target_prediction.top_targets" in (
        contract["required_report_markers"]
    )
    assert (
        "skin_specialized_binding.skin_context_supported"
        in contract["decision_gate"]
    )
    assert (
        "skin_specialized_binding.top_target_skin_context_supported"
        in contract["decision_gate"]
    )
    assert "overall_decision.decision must be PASS" in contract["decision_gate"]
    assert (
        "overall_decision.requires_human_review must be false"
        in contract["decision_gate"]
    )
    assert "stage0_claim_quality status is ok" in contract["decision_gate"]
    assert "overall_decision.decision is PASS" in contract["claimable_when"]
    assert (
        "overall_decision.requires_human_review is false"
        in contract["claimable_when"]
    )
    assert (
        "skin_specialized_binding.skin_context_supported is true"
        in contract["claimable_when"]
    )
    assert (
        "skin_specialized_binding.top_target_skin_context_supported is true"
        in contract["claimable_when"]
    )
    assert "stage0_claim_quality status is ok" in contract["claimable_when"]
    assert "--preset report" in contract["verify_command"]


def test_pipeline_readiness_verify_command_uses_custom_target_metadata(
    tmp_path: Path,
) -> None:
    target_metadata = tmp_path / "target_metadata.tsv"

    res = run_readiness(
        [
            "--smiles",
            "CCO",
            "--run-id",
            "custom_metadata_case",
            "--preset",
            "target-id",
            "--target-metadata",
            str(target_metadata),
            "--skip-data-readiness",
            "--skip-stage0-source-readiness",
            "--skip-safety-readiness",
            "--skip-model-readiness",
            "--json",
        ]
    )

    assert res.returncode == 2, res.stderr
    payload = json.loads(res.stdout)
    contract = payload["output_contract"]

    assert f"target_metadata={target_metadata}" in payload["snakemake_command"]
    assert f"--target-metadata {target_metadata}" in contract["verify_command"]


def test_pipeline_readiness_sdf_contract_uses_sdf_provenance(tmp_path: Path) -> None:
    sdf = tmp_path / "source_ligand.sdf"
    sdf.write_text(CAFFEINE_SDF)

    res = run_readiness(
        [
            "--sdf",
            str(sdf),
            "--run-id",
            "sdf_report_case",
            "--preset",
            "report",
            "--skip-data-readiness",
            "--skip-stage0-source-readiness",
            "--skip-safety-readiness",
            "--skip-model-readiness",
            "--json",
        ]
    )

    assert res.returncode == 2, res.stderr
    payload = json.loads(res.stdout)
    contract = payload["output_contract"]
    fields = contract["required_summary_fields"]
    markers = contract["required_report_markers"]
    launcher_fields = contract["required_launcher_result_fields"]

    assert payload["input_type"] == "sdf"
    assert payload["input_smiles"] is None
    assert payload["input_canonical_smiles"] is None
    assert payload["input_sdf"] == str(sdf)
    assert "compound_sdf=" + str(sdf) in payload["snakemake_command"]
    assert "input_type" in fields["compound"]
    assert "input_sdf" in fields["compound"]
    assert "input_smiles" not in fields["compound"]
    assert "input_canonical_smiles" not in fields["compound"]
    assert "Input type" in markers
    assert "Input SDF" in markers
    assert "Input SMILES" not in markers
    assert "Input canonical SMILES" not in markers
    assert "input_sdf" in launcher_fields
    assert "input_smiles" not in launcher_fields
    assert "input_canonical_smiles" not in launcher_fields


def test_pipeline_readiness_report_text_lists_html_report_markers() -> None:
    res = run_readiness(
        [
            "--smiles",
            "CCO",
            "--run-id",
            "ethanol_report_text_case",
            "--preset",
            "report",
            "--skip-data-readiness",
            "--skip-stage0-source-readiness",
            "--skip-safety-readiness",
            "--skip-model-readiness",
        ]
    )

    assert res.returncode == 2, res.stderr
    assert "required report markers:" in res.stdout
    assert "Input type" in res.stdout
    assert "Input SMILES" in res.stdout
    assert "Input canonical SMILES" in res.stdout
    assert "Input SDF" not in res.stdout
    assert "InChIKey" in res.stdout
    assert "canonical SMILES" in res.stdout
    assert "Claim status" in res.stdout
    assert "Skin_Reaction" in res.stdout
    assert "Structural alerts" in res.stdout
    assert "degraded skin-sens evidence" in res.stdout
    assert "missing skin-sens models" in res.stdout
    assert "Moderate ADMET risk endpoints" in res.stdout
    assert "Summary ADMET metrics" in res.stdout
    assert "Cosmetic/drug decision" in res.stdout


    assert "Skin expression supported" in res.stdout
    assert "Skin-sens 3-model evidence" in res.stdout
    assert "Skin efficacy supported" in res.stdout
    assert "Screened target candidates" in res.stdout
    assert "Screening stage counts" in res.stdout
    assert "top target gene/protein labels when available" in res.stdout
    assert "Top binding target" in res.stdout
    assert "top target final score" in res.stdout
    assert "top target docking RRF when available" in res.stdout
    assert "top target source count" in res.stdout
    assert "top target sources" in res.stdout
    assert "top target skin score" in res.stdout
    assert "top target skin tier" in res.stdout
    assert "top target skin efficacy" in res.stdout
    assert "Most skin-relevant top target" in res.stdout
    assert "most skin-relevant gene/protein labels when available" in res.stdout
    assert "most skin-relevant score/source/skin-efficacy evidence" in res.stdout
    assert "required launcher result fields:" in res.stdout
    assert "ranked_target_<rank>" in res.stdout
    assert "ranked_target_<rank>_final_score" in res.stdout
    assert "ranked_target_<rank>_sources" in res.stdout
    assert "most_skin_relevant_gene_symbol" in res.stdout
    assert "most_skin_relevant_protein_name" in res.stdout
    assert "decision gate:" in res.stdout
    assert (
        "skin_specialized_binding.top_target_skin_context_supported"
        in res.stdout
    )
    assert "claimable when:" in res.stdout
    assert "overall_decision.decision is PASS" in res.stdout
    assert "overall_decision.requires_human_review is false" in res.stdout


def test_pipeline_readiness_gpu_mode_fails_closed_on_nvml_mismatch(monkeypatch) -> None:
    def fake_run(command, **_kwargs):
        if command == ["nvidia-smi"]:
            return SimpleNamespace(
                returncode=9,
                stdout="",
                stderr="Failed to initialize NVML: Driver/library version mismatch",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pipeline_readiness.subprocess, "run", fake_run)
    args = pipeline_readiness.parse_args(
        [
            "--smiles",
            "CCO",
            "--preset",
            "target-id",
            "--mode",
            "fast",
            "--runtime-mode",
            "gpu",
            "--skip-data-readiness",
            "--skip-stage0-source-readiness",
            "--skip-safety-readiness",
            "--skip-model-readiness",
        ]
    )

    payload = pipeline_readiness.readiness_payload(args)
    groups = groups_by_name(payload)

    assert payload["status"] == "failed"
    assert groups["runtime:gpu"]["status"] == "failed"
    assert groups["runtime:gpu"]["blocking"] is True
    assert groups["runtime:gpu"]["blockers"][0]["label"] == "nvml_driver_library"


def test_pipeline_readiness_gpu_mode_requires_real_cuda_smoke(monkeypatch) -> None:
    def fake_run(command, **_kwargs):
        if command == ["nvidia-smi"]:
            return SimpleNamespace(returncode=0, stdout="NVIDIA-SMI OK", stderr="")
        return SimpleNamespace(returncode=1, stdout="", stderr="torch cuda unavailable")

    monkeypatch.setattr(pipeline_readiness.subprocess, "run", fake_run)
    args = pipeline_readiness.parse_args(
        [
            "--smiles",
            "CCO",
            "--preset",
            "target-id",
            "--mode",
            "fast",
            "--runtime-mode",
            "gpu",
            "--skip-data-readiness",
            "--skip-stage0-source-readiness",
            "--skip-safety-readiness",
            "--skip-model-readiness",
        ]
    )

    payload = pipeline_readiness.readiness_payload(args)
    blockers = groups_by_name(payload)["runtime:gpu"]["blockers"]

    assert payload["status"] == "failed"
    assert blockers == [
        {
            "label": "real_cuda_tensor_smoke",
            "detail": "GPU mode requires an actual CUDA tensor/kernel smoke test",
        }
    ]
