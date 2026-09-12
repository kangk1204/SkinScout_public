"""Regression tests for completed-run output contract verification."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VERIFY = ROOT / "scripts" / "verify_run_outputs.py"
SUMMARIZE = ROOT / "scripts" / "summarize_run_outputs.py"


def run_verify(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VERIFY), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def run_summarize(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SUMMARIZE), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def load_summarize_module():
    scripts_dir = str(ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(
        "summarize_run_outputs_under_test",
        SUMMARIZE,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_verify_module():
    scripts_dir = str(ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(
        "verify_run_outputs_under_test",
        VERIFY,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_manifest_member_path_rejects_escape_and_accepts_local_file(
    tmp_path: Path,
) -> None:
    verifier = load_verify_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = run_dir / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    local = run_dir / "pose.sdf"
    local.write_text("pose\n", encoding="utf-8")
    outside = tmp_path / "outside.sdf"
    outside.write_text("outside\n", encoding="utf-8")

    assert verifier._manifest_member_path(manifest, "pose.sdf") == local
    assert verifier._manifest_member_path(manifest, "../outside.sdf") is None
    assert verifier._manifest_member_path(manifest, str(outside)) is None
    assert verifier._manifest_member_path(manifest, "") is None


def skin_sens_calls_for_decision(decision: str) -> dict[str, str]:
    if decision == "HALT":
        return {"husspred": "positive", "stoptox": "positive", "pred_skin": "negative"}
    if decision == "FLAG_HIGH":
        return {"husspred": "positive", "stoptox": "negative", "pred_skin": "negative"}
    return {"husspred": "negative", "stoptox": "negative", "pred_skin": "negative"}


def probability_for_call(call: str) -> float:
    return 0.8 if call == "positive" else 0.2


def admet_risk_payload(predictions: dict[str, float]) -> dict[str, object]:
    endpoints = {}
    for endpoint in ("AMES", "ClinTox", "DILI", "Skin_Reaction", "hERG"):
        value = predictions[endpoint]
        if value >= 0.5:
            risk_level = "high"
        elif value >= 0.25:
            risk_level = "moderate"
        else:
            risk_level = "low"
        endpoints[endpoint] = {
            "value": value,
            "risk_level": risk_level,
            "moderate_threshold": 0.25,
            "high_threshold": 0.5,
        }
    return {
        "endpoints": endpoints,
        "high_risk_endpoints": [
            endpoint for endpoint, item in endpoints.items() if item["risk_level"] == "high"
        ],
        "moderate_risk_endpoints": [
            endpoint for endpoint, item in endpoints.items() if item["risk_level"] == "moderate"
        ],
    }


def skin_toxicity_payload(
    safety_decision: str,
    admet_risk: dict[str, object],
) -> dict[str, object]:
    decision = "PASS"
    toxicity_level = "low"
    reasons: list[str] = []

    def review(reason: str, level: str = "moderate") -> None:
        nonlocal decision, toxicity_level
        if decision != "HALT":
            decision = "REVIEW"
        if level == "high" or toxicity_level == "high":
            toxicity_level = "high"
        elif level == "moderate":
            toxicity_level = "moderate"
        reasons.append(reason)

    if safety_decision == "HALT":
        decision = "HALT"
        toxicity_level = "high"
        reasons.append("skin sensitization consensus is HALT")
    elif safety_decision == "FLAG_HIGH":
        review("skin sensitization consensus is FLAG_HIGH", "high")

    endpoints = admet_risk["endpoints"]
    assert isinstance(endpoints, dict)
    skin_reaction = endpoints["Skin_Reaction"]
    assert isinstance(skin_reaction, dict)
    skin_reaction_risk = skin_reaction["risk_level"]
    if skin_reaction_risk == "high":
        review("Skin_Reaction ADMET risk is high", "high")
    elif skin_reaction_risk == "moderate":
        review("Skin_Reaction ADMET risk is moderate", "moderate")

    if not reasons:
        reasons.append("skin sensitization and Skin_Reaction ADMET risk are low")

    return {
        "decision": decision,
        "toxicity_level": toxicity_level,
        "skin_sens_decision": safety_decision,
        "skin_reaction_risk_level": skin_reaction_risk,
        "skin_reaction_value": skin_reaction["value"],
        "structural_alerts_present": False,
        "structural_alert_flags": [],
        "degraded": False,
        "missing_models": [],
        "reasons": reasons,
    }


def overall_decision_payload(safety_decision: str) -> dict[str, object]:
    if safety_decision == "PASS":
        decision = "PASS"
        reasons = ["top predicted target is P1 among 2 ranked targets"]
    elif safety_decision == "HALT":
        decision = "HALT"
        reasons = [
            "skin sensitization consensus is HALT",
            "top predicted target is P1 among 2 ranked targets",
        ]
    else:
        decision = "FLAG_HIGH"
        reasons = [
            "skin sensitization consensus is FLAG_HIGH",
            "top predicted target is P1 among 2 ranked targets",
        ]
    return {
        "decision": decision,
        "requires_human_review": decision != "PASS",
        "recommended_action": {
            "HALT": "stop_before_claim",
            "FLAG_HIGH": "review_before_claim",
            "PASS": "proceed",
        }[decision],
        "claimable": decision == "PASS",
        "reasons": reasons,
        "top_target_id": "P1",
        "target_count": 2,
    }


def write_minimal_run(run_dir: Path, *, safety_decision: str = "PASS") -> None:
    (run_dir / "01_input").mkdir(parents=True)
    (run_dir / "02_admet").mkdir(parents=True)
    (run_dir / "02b_cosmetic_drug").mkdir(parents=True)
    (run_dir / "03_targets").mkdir(parents=True)
    (run_dir / "03_targets" / "mode_fast").mkdir(parents=True)
    (run_dir / "03_targets" / "mode_comprehensive").mkdir(parents=True)
    (run_dir / "09_report").mkdir(parents=True)

    (run_dir / "01_input" / "compound_canonical.json").write_text(json.dumps({
        "input_type": "smiles",
        "input_smiles": "CCO",
        "input_canonical_smiles": "CCO",
        "canonical_smiles": "CCO",
        "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
    }))
    calls = skin_sens_calls_for_decision(safety_decision)
    skin_sens_evidence = [
        {
            "model": model,
            "status": "ok",
            "call": calls[model],
            "probability": probability_for_call(calls[model]),
        }
        for model in ("husspred", "stoptox", "pred_skin")
    ]
    admet_predictions = {
        "molecular_weight": 46.069,
        "logP": -0.0014,
        "QED": 0.4068,
        "tpsa": 20.23,
        "AMES": 0.04,
        "ClinTox": 0.01,
        "DILI": 0.09,
        "Skin_Reaction": 0.21,
        "hERG": 0.01,
        "Caco2_Wang": -4.0,
        "LD50_Zhu": 1.1,
        "Solubility_AqSolDB": 1.2,
    }
    structural_alerts = {
        "smiles": "CCO",
        "matches": {"PAINS_A": [], "PAINS_B": [], "PAINS_C": [], "BRENK": [], "NIH": []},
        "any_pains": False,
        "any_brenk": False,
        "any_nih": False,
    }
    (run_dir / "02_admet" / "admet_report.json").write_text(json.dumps({
        "skin_sens": {
            "husspred": calls["husspred"],
            "stoptox": calls["stoptox"],
            "pred_skin": calls["pred_skin"],
            "decision": safety_decision,
            "degraded": False,
            "missing_models": [],
        },
        "structural_alerts": structural_alerts,
        "admet_ai_predictions": admet_predictions,
    }))
    (run_dir / "02_admet" / "admet_ai.json").write_text(json.dumps({
        "smiles": "CCO",
        "status": "ok",
        "predictions": admet_predictions,
    }))
    (run_dir / "02_admet" / "structural_alerts.json").write_text(json.dumps(
        structural_alerts
    ))
    (run_dir / "02_admet" / "husspred.json").write_text(json.dumps({
        "smiles": "CCO",
        "status": "ok",
        "skin_sens_probability": probability_for_call(calls["husspred"]),
        "skin_sens_call": calls["husspred"],
        "degraded": False,
        "degraded_reason": "",
        "raw": {},
    }))
    (run_dir / "02_admet" / "stoptox.json").write_text(json.dumps({
        "smiles": "CCO",
        "status": "ok",
        "skin_sens_probability": probability_for_call(calls["stoptox"]),
        "skin_sens_call": calls["stoptox"],
        "degraded": False,
        "degraded_reason": "",
        "raw": {},
    }))
    (run_dir / "02_admet" / "pred_skin.json").write_text(json.dumps({
        "smiles": "CCO",
        "status": "ok",
        "consensus_call": calls["pred_skin"],
        "pred_skin": {
            "probability": probability_for_call(calls["pred_skin"]),
            "call": calls["pred_skin"],
            "raw": {},
        },
        "degraded": False,
        "degraded_reason": "",
    }))
    (run_dir / "02_admet" / "skin_sens_decision.txt").write_text(
        safety_decision + "\n"
    )
    (run_dir / "02b_cosmetic_drug" / "cosing_match.json").write_text(json.dumps({
        "level": "NEW",
        "inci": None,
        "functions": [],
        "tanimoto": 0.0,
        "reference_status": "ok",
    }))
    (run_dir / "02b_cosmetic_drug" / "drug_warnings.json").write_text(json.dumps({
        "max_tanimoto_to_approved_drug": 0.1,
        "n_warnings": 0,
        "warnings": [],
        "reference_status": "ok",
    }))
    (run_dir / "02b_cosmetic_drug" / "cosmetic_drug_decision.txt").write_text(
        "PROCEED\n# policy: moderate\n"
    )
    (run_dir / "03_targets" / "ranked_targets_v3_with_efficacy.csv").write_text(
        "target_id,docking_rrf,skin_score,skin_tier,cell_type_preferred,final_score,source_count,sources,efficacy_top1\n"
        "P1,0.9,0.8,high,keratinocytes,0.85,3,autodock;gnina;rtmscore,hydration (3 papers)\n"
        "P2,0.7,0.6,medium,fibroblasts,0.70,3,autodock;gnina;rtmscore,barrier (2 papers)\n"
    )
    fast = run_dir / "03_targets" / "mode_fast"
    (fast / "psichic_proteome.tsv").write_text(
        "target_id\tpsichic_score\tscore\n"
        "P1\t0.8\t0.8\n"
        "P2\t0.7\t0.7\n"
        "P3\t0.6\t0.6\n"
        "P4\t0.5\t0.5\n"
    )
    (fast / "daina_zoete_proteome.tsv").write_text(
        "target_id\tmax_tanimoto\tscore\n"
        "P1\t0.4\t0.4\n"
        "P2\t0.3\t0.3\n"
        "P3\t0.2\t0.2\n"
        "P4\t0.1\t0.1\n"
    )
    (fast / "dti_rrf_top25pct.csv").write_text(
        "target_id,rrf_score,source_count,sources\n"
        "P1,0.9,2,daina;psichic\n"
        "P2,0.8,2,daina;psichic\n"
        "P3,0.7,2,daina;psichic\n"
    )
    (fast / "autodock_top5k.tsv").write_text(
        "target_id\tvina_score\tneg_vina_score\tengine\tmap_coverage_complete\t"
        "map_coverage_numerator\tmap_coverage_denominator\tdegraded\n"
        "P1\t-7.0\t7.0\tautodock_gpu\ttrue\t3\t3\tfalse\n"
        "P2\t-6.0\t6.0\tautodock_gpu\ttrue\t3\t3\tfalse\n"
        "P3\t-5.0\t5.0\tautodock_gpu\ttrue\t3\t3\tfalse\n"
    )
    (fast / "top50.csv").write_text(
        "target_id,rrf_score,source_count,sources\n"
        "P1,0.9,2,autodock;gnina\n"
        "P2,0.8,2,autodock;gnina\n"
    )
    comprehensive = run_dir / "03_targets" / "mode_comprehensive"
    (comprehensive / "ligand.pdbqt").write_text("REMARK test ligand\n")
    (comprehensive / "autodock_all_targets.tsv").write_text(
        "target_id\tvina_score\tneg_vina_score\tengine\tmap_coverage_complete\t"
        "map_coverage_numerator\tmap_coverage_denominator\tdegraded\n"
        "P1\t-7.0\t7.0\tautodock_gpu\ttrue\t4\t4\tfalse\n"
        "P2\t-6.0\t6.0\tautodock_gpu\ttrue\t4\t4\tfalse\n"
        "P3\t-5.0\t5.0\tautodock_gpu\ttrue\t4\t4\tfalse\n"
        "P4\t-4.0\t4.0\tautodock_gpu\ttrue\t4\t4\tfalse\n"
    )
    (comprehensive / "top_pct_pre_rescore.csv").write_text(
        "target_id,score,source\n"
        "P1,7.0,autodock\n"
        "P2,6.0,autodock\n"
        "P3,5.0,autodock\n"
    )
    (comprehensive / "gnina_rescores.tsv").write_text(
        "target_id\tcnn_affinity\tscore\n"
        "P1\t8.0\t8.0\n"
        "P2\t7.0\t7.0\n"
        "P3\t6.0\t6.0\n"
    )
    (comprehensive / "rtmscore_rescores.tsv").write_text(
        "target_id\trtm_score\tscore\n"
        "P1\t0.8\t0.8\n"
        "P2\t0.7\t0.7\n"
        "P3\t0.6\t0.6\n"
    )
    (comprehensive / "boltz2_affinity_top.tsv").write_text(
        "target_id\tboltz2_neg_log_uM\tscore\n"
        "P1\t6.5\t6.5\n"
        "P2\t5.5\t5.5\n"
        "P3\t4.5\t4.5\n"
    )
    (comprehensive / "top50_4way_consensus.csv").write_text(
        "target_id,rrf_score,source_count,sources\n"
        "P1,0.9,4,autodock;boltz;gnina;rtm\n"
        "P2,0.8,4,autodock;boltz;gnina;rtm\n"
    )
    (run_dir / "09_report" / "index.html").write_text(
        "<html><body>"
        "ADMET ranked targets skin-efficacy skin toxicity "
        "Overall decision: PASS "
        "Recommended action: proceed "
        "Claimable: yes "
        "Claim status: proceed "
        "Input type: smiles "
        "Input SMILES: CCO "
        "Input canonical SMILES: CCO "
        "InChIKey: LFQSCWFLJHTTHZ-UHFFFAOYSA-N "
        "canonical SMILES: CCO "
        "Skin toxicity: PASS "
        "Skin_Reaction: 0.210 (low) "
        "Structural alerts: none "
        "degraded skin-sens evidence: False "
        "missing skin-sens models: none "
        "Skin-sens 3-model evidence "
        "Cosmetic/drug decision: PROCEED "
        "Drug policy: moderate "
        "CosIng level: NEW "
        "INCI = — "
        "functions = — "
        "Tanimoto = 0.000 "
        "High ADMET risk endpoints: none "
        "Moderate ADMET risk endpoints: none "
        "Summary ADMET metrics "
        "Drug-avoidance warnings: 0 "
        "Drug-avoidance: max Tanimoto vs approved = 0.100, 0 warning(s). "
        "Screened target candidates: 4 "
        "Screening stage counts "
        "Skin context decision: skin_context_supported "
        "Skin context supported: yes "
        "Skin expression supported: yes "
        "Skin efficacy supported: yes "
        "Top binding target: P1 "
        "top target gene: GENE1 "
        "top target protein: Protein one "
        "top target final score: 0.85 "
        "top target docking RRF: 0.9 "
        "top target source count: 3 "
        "top target sources: autodock, gnina, rtmscore "
        "top target skin score: 0.8 "
        "top target skin tier: high "
        "top target skin efficacy: hydration (3 papers) "
        "top target skin context supported: yes "
        "Most skin-relevant top target: P1 "
        "most skin-relevant gene: GENE1 "
        "most skin-relevant protein: Protein one "
        "most skin-relevant final score: 0.85 "
        "most skin-relevant docking RRF: 0.9 "
        "most skin-relevant source count: 3 "
        "most skin-relevant sources: autodock, gnina, rtmscore "
        "most skin-relevant skin score: 0.8 "
        "most skin-relevant skin tier: high "
        "most skin-relevant efficacy: hydration (3 papers) "
        "<table>"
        "<tr><th>ADMET Metric</th><th>Value</th></tr>"
        "<tr><td>AMES</td><td>0.04</td></tr>"
        "<tr><td>ClinTox</td><td>0.01</td></tr>"
        "<tr><td>DILI</td><td>0.09</td></tr>"
        "<tr><td>Skin_Reaction</td><td>0.21</td></tr>"
        "<tr><td>hERG</td><td>0.01</td></tr>"
        "<tr><td>LD50_Zhu</td><td>1.1</td></tr>"
        "<tr><td>Solubility_AqSolDB</td><td>1.2</td></tr>"
        "<tr><td>logP</td><td>-0.0014</td></tr>"
        "<tr><td>QED</td><td>0.4068</td></tr>"
        "<tr><td>tpsa</td><td>20.23</td></tr>"
        "</table>"
        "<table>"
        "<tr><th>model</th><th>status</th><th>call</th><th>probability</th></tr>"
        "<tr><td>husspred</td><td>ok</td><td>negative</td><td>0.2</td></tr>"
        "<tr><td>stoptox</td><td>ok</td><td>negative</td><td>0.2</td></tr>"
        "<tr><td>pred_skin</td><td>ok</td><td>negative</td><td>0.2</td></tr>"
        "</table>"
        "<table>"
        "<tr><th>Screening stage</th><th>Target count</th></tr>"
        "<tr><td>autodock_rescored_targets</td><td>3</td></tr>"
        "<tr><td>autodock_screened_targets</td><td>4</td></tr>"
        "<tr><td>boltz2_affinity_targets</td><td>3</td></tr>"
        "<tr><td>daina_zoete_targets</td><td>4</td></tr>"
        "<tr><td>dti_rrf_candidates</td><td>3</td></tr>"
        "<tr><td>four_way_consensus_targets</td><td>2</td></tr>"
        "<tr><td>gnina_rescored_targets</td><td>3</td></tr>"
        "<tr><td>pre_rescore_candidates</td><td>3</td></tr>"
        "<tr><td>psichic_proteome_targets</td><td>4</td></tr>"
        "<tr><td>rerank_consensus_targets</td><td>2</td></tr>"
        "<tr><td>rtmscore_rescored_targets</td><td>3</td></tr>"
        "<tr><td>skin_weighted_ranked_targets</td><td>2</td></tr>"
        "</table>"
        "<table>"
        "<tr><td>P1</td><td>GENE1</td><td>Protein one</td><td>0.85</td><td>0.8</td><td>high</td>"
        "<td>0.9</td><td>autodock;gnina;rtmscore</td>"
        "<td>hydration (3 papers)</td></tr>"
        "<tr><td>P2</td><td>GENE2</td><td>Protein two</td><td>0.7</td><td>0.6</td><td>medium</td>"
        "<td>0.7</td><td>autodock;gnina;rtmscore</td>"
        "<td>barrier (2 papers)</td></tr>"
        "</table>"
        "</body></html>\n"
    )
    overall_decision = overall_decision_payload(safety_decision)
    admet_risk = admet_risk_payload(admet_predictions)
    skin_toxicity = skin_toxicity_payload(safety_decision, admet_risk)
    (run_dir / "run_summary.json").write_text(json.dumps({
        "schema_version": "skinscout.run_summary.v1",
        "run_id": run_dir.name,
        "preset": "target-id",
        "mode": "both",
        "compound": {
            "input_type": "smiles",
            "input_smiles": "CCO",
            "input_canonical_smiles": "CCO",
            "canonical_smiles": "CCO",
            "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        },
        "safety": {
            "skin_sens_decision": safety_decision,
            "skin_sens_calls": calls,
            "skin_sens_evidence": skin_sens_evidence,
            "degraded": False,
            "missing_models": [],
            "admet_metrics": {
                key: admet_predictions[key]
                for key in (
                    "AMES",
                    "ClinTox",
                    "DILI",
                    "Skin_Reaction",
                    "hERG",
                    "LD50_Zhu",
                    "Solubility_AqSolDB",
                    "logP",
                    "QED",
                    "tpsa",
                )
            },
            "structural_alert_flags": {
                "any_pains": False,
                "any_brenk": False,
                "any_nih": False,
            },
            "admet_risk_assessment": admet_risk,
        },
        "skin_toxicity": skin_toxicity,
        "cosmetic_drug": {
            "decision": "PROCEED",
            "drug_policy": "moderate",
            "cosing_level": "NEW",
            "inci": None,
            "cosing_functions": [],
            "cosing_tanimoto": 0.0,
            "max_tanimoto_to_approved_drug": 0.1,
            "n_warnings": 0,
        },
        "overall_decision": overall_decision,
        "target_prediction": {
            "ranking_path": "03_targets/ranked_targets_v3_with_efficacy.csv",
            "n_targets": 2,
            "screened_target_count": 4,
            "screening_counts": {
                "autodock_rescored_targets": 3,
                "autodock_screened_targets": 4,
                "boltz2_affinity_targets": 3,
                "daina_zoete_targets": 4,
                "dti_rrf_candidates": 3,
                "four_way_consensus_targets": 2,
                "gnina_rescored_targets": 3,
                "pre_rescore_candidates": 3,
                "psichic_proteome_targets": 4,
                "rerank_consensus_targets": 2,
                "rtmscore_rescored_targets": 3,
                "skin_weighted_ranked_targets": 2,
            },
            "top_n": 2,
            "top_targets": [
                {
                    "target_id": "P1",
                    "gene_symbol": "GENE1",
                    "protein_name": "Protein one",
                    "final_score": 0.85,
                    "skin_score": 0.8,
                    "skin_tier": "high",
                    "cell_type_preferred": "keratinocytes",
                    "docking_rrf": 0.9,
                    "source_count": 3,
                    "sources": ["autodock", "gnina", "rtmscore"],
                    "efficacy": ["hydration (3 papers)"],
                },
                {
                    "target_id": "P2",
                    "gene_symbol": "GENE2",
                    "protein_name": "Protein two",
                    "final_score": 0.70,
                    "skin_score": 0.6,
                    "skin_tier": "medium",
                    "cell_type_preferred": "fibroblasts",
                    "docking_rrf": 0.7,
                    "source_count": 3,
                    "sources": ["autodock", "gnina", "rtmscore"],
                    "efficacy": ["barrier (2 papers)"],
                },
            ],
        },
        "skin_specialized_binding": {
            "context": "skin-specialized material-protein binding",
            "ranking_path": "03_targets/ranked_targets_v3_with_efficacy.csv",
            "skin_context_decision": "skin_context_supported",
            "skin_context_supported": True,
            "skin_expression_supported": True,
            "skin_efficacy_supported": True,
            "skin_context_reasons": [
                "top targets include low-or-higher skin-expression support",
                "top targets include KG skin-efficacy labels",
            ],
            "top_target_id": "P1",
            "top_target_gene_symbol": "GENE1",
            "top_target_protein_name": "Protein one",
            "top_target_final_score": 0.85,
            "top_target_docking_rrf": 0.9,
            "top_target_source_count": 3,
            "top_target_sources": ["autodock", "gnina", "rtmscore"],
            "top_target_skin_score": 0.8,
            "top_target_skin_tier": "high",
            "top_target_cell_type_preferred": "keratinocytes",
            "top_target_efficacy": ["hydration (3 papers)"],
            "top_target_skin_expression_supported": True,
            "top_target_skin_efficacy_supported": True,
            "top_target_skin_context_supported": True,
            "top_targets_with_skin_efficacy": 2,
            "most_skin_relevant_target": {
                "target_id": "P1",
                "gene_symbol": "GENE1",
                "protein_name": "Protein one",
                "final_score": 0.85,
                "skin_score": 0.8,
                "skin_tier": "high",
                "cell_type_preferred": "keratinocytes",
                "docking_rrf": 0.9,
                "source_count": 3,
                "sources": ["autodock", "gnina", "rtmscore"],
                "efficacy": ["hydration (3 papers)"],
            },
        },
        "artifacts": {
            "compound": "01_input/compound_canonical.json",
            "admet_report": "02_admet/admet_report.json",
            "admet_ai": "02_admet/admet_ai.json",
            "structural_alerts": "02_admet/structural_alerts.json",
            "husspred": "02_admet/husspred.json",
            "stoptox": "02_admet/stoptox.json",
            "pred_skin": "02_admet/pred_skin.json",
            "skin_sens_decision": "02_admet/skin_sens_decision.txt",
            "cosing_match": "02b_cosmetic_drug/cosing_match.json",
            "drug_warnings": "02b_cosmetic_drug/drug_warnings.json",
            "cosmetic_drug_decision": "02b_cosmetic_drug/cosmetic_drug_decision.txt",
            "target_ranking": "03_targets/ranked_targets_v3_with_efficacy.csv",
            "target_fast_psichic": "03_targets/mode_fast/psichic_proteome.tsv",
            "target_fast_daina_zoete": "03_targets/mode_fast/daina_zoete_proteome.tsv",
            "target_fast_dti_rrf": "03_targets/mode_fast/dti_rrf_top25pct.csv",
            "target_fast_autodock": "03_targets/mode_fast/autodock_top5k.tsv",
            "target_fast_rerank_consensus": "03_targets/mode_fast/top50.csv",
            "target_comprehensive_ligand": "03_targets/mode_comprehensive/ligand.pdbqt",
            "target_comprehensive_autodock": "03_targets/mode_comprehensive/autodock_all_targets.tsv",
            "target_comprehensive_pre_rescore": "03_targets/mode_comprehensive/top_pct_pre_rescore.csv",
            "target_comprehensive_gnina": "03_targets/mode_comprehensive/gnina_rescores.tsv",
            "target_comprehensive_rtmscore": "03_targets/mode_comprehensive/rtmscore_rescores.tsv",
            "target_comprehensive_boltz2": "03_targets/mode_comprehensive/boltz2_affinity_top.tsv",
            "target_comprehensive_consensus": "03_targets/mode_comprehensive/top50_4way_consensus.csv",
        },
    }))
    (run_dir / "run_summary.md").write_text(
        "# SkinScout Run Summary\n"
        "\n"
        "## Run\n"
        f"- Run: {run_dir.name}\n"
        "- Preset: target-id\n"
        "- Mode: both\n"
        "- Input type: smiles\n"
        "- Input SMILES: `CCO`\n"
        "- Input canonical SMILES: `CCO`\n"
        "- Input SDF: `n/a`\n"
        "- Compound: `CCO`\n"
        "- InChIKey: LFQSCWFLJHTTHZ-UHFFFAOYSA-N\n"
        "\n"
        "## Overall Decision\n"
        f"- Decision: {overall_decision['decision']}\n"
        f"- Recommended action: {overall_decision['recommended_action']}\n"
        f"- Claimable: {'yes' if overall_decision['claimable'] else 'no'}\n"
        "- Requires human review: "
        f"{'yes' if overall_decision['requires_human_review'] else 'no'}\n"
        "\n"
        "## Skin Toxicity\n"
        f"- Decision: {skin_toxicity['decision']}\n"
        f"- Toxicity level: {skin_toxicity['toxicity_level']}\n"
        f"- Skin sensitization: {skin_toxicity['skin_sens_decision']}\n"
        f"- Skin_Reaction risk: {skin_toxicity['skin_reaction_risk_level']}\n"
        f"- Skin_Reaction value: {skin_toxicity['skin_reaction_value']}\n"
        "- Structural alerts present: no\n"
        "- Structural alert flags: none\n"
        "- Degraded evidence: no\n"
        "- Missing skin-sens models: none\n"
        f"- Reason: {skin_toxicity['reasons'][0]}\n"
        "\n"
        "## Safety And ADMET\n"
        f"- Skin sensitization: {safety_decision}\n"
        "- Degraded evidence: no\n"
        "- Missing skin-sens models: none\n"
        "| Skin-sens model | Status | Call | Probability |\n"
        "|---|---|---|---:|\n"
        + "".join(
            "| "
            f"{row['model']} | {row['status']} | {row['call']} | {row['probability']} |\n"
            for row in skin_sens_evidence
        )
        + "| Endpoint | Value | Risk |\n"
        "|---|---:|---|\n"
        + "".join(
            f"| {endpoint} | {item['value']} | {item['risk_level']} |\n"
            for endpoint, item in admet_risk["endpoints"].items()
        )
        + "| ADMET Metric | Value |\n"
        "|---|---:|\n"
        + "".join(
            f"| {metric} | {admet_predictions[metric]} |\n"
            for metric in (
                "AMES",
                "ClinTox",
                "DILI",
                "Skin_Reaction",
                "hERG",
                "LD50_Zhu",
                "Solubility_AqSolDB",
                "logP",
                "QED",
                "tpsa",
            )
        )
        + "- High risk endpoints: none\n"
        "- Moderate risk endpoints: none\n"
        "\n"
        "## Cosmetic And Drug\n"
        "- Decision: PROCEED\n"
        "- Drug policy: moderate\n"
        "- CosIng level: NEW\n"
        "- INCI: n/a\n"
        "- Drug warnings: 0\n"
        "\n"
        "## Top Targets\n"
        "- Ranked targets: 2\n"
        "- Screened target candidates: 4\n"
        "| Rank | Target | Gene | Protein | Final | Skin | Skin Tier | Docking RRF | Sources | Efficacy |\n"
        "|---:|---|---|---|---:|---:|---|---:|---|---|\n"
        "| 1 | P1 | GENE1 | Protein one | 0.85 | 0.8 | high | 0.9 | autodock, gnina, rtmscore | hydration (3 papers) |\n"
        "| 2 | P2 | GENE2 | Protein two | 0.7 | 0.6 | medium | 0.7 | autodock, gnina, rtmscore | barrier (2 papers) |\n"
        "\n"
        "| Screening Stage | Targets |\n"
        "|---|---:|\n"
        "| autodock_rescored_targets | 3 |\n"
        "| autodock_screened_targets | 4 |\n"
        "| boltz2_affinity_targets | 3 |\n"
        "| daina_zoete_targets | 4 |\n"
        "| dti_rrf_candidates | 3 |\n"
        "| four_way_consensus_targets | 2 |\n"
        "| gnina_rescored_targets | 3 |\n"
        "| pre_rescore_candidates | 3 |\n"
        "| psichic_proteome_targets | 4 |\n"
        "| rerank_consensus_targets | 2 |\n"
        "| rtmscore_rescored_targets | 3 |\n"
        "| skin_weighted_ranked_targets | 2 |\n"
        "\n"
        "## Skin-Specialized Binding\n"
        "- Context: skin-specialized material-protein binding\n"
        "- Skin context decision: skin_context_supported\n"
        "- Skin context supported: yes\n"
        "- Skin expression supported: yes\n"
        "- Skin efficacy supported: yes\n"
        "- Top predicted binding target: P1\n"
        "- Top target gene: GENE1\n"
        "- Top target protein: Protein one\n"
        "- Top target final score: 0.85\n"
        "- Top target docking RRF: 0.9\n"
        "- Top target source count: 3\n"
        "- Top target sources: autodock, gnina, rtmscore\n"
        "- Top target skin score: 0.8\n"
        "- Top target skin tier: high\n"
        "- Top target skin efficacy: hydration (3 papers)\n"
        "- Top target skin expression supported: yes\n"
        "- Top target skin efficacy supported: yes\n"
        "- Top target skin context supported: yes\n"
        "- Most skin-relevant top target: GENE1 (P1)\n"
        "- Most skin-relevant gene: GENE1\n"
        "- Most skin-relevant protein: Protein one\n"
        "- Most skin-relevant final score: 0.85\n"
        "- Most skin-relevant docking RRF: 0.9\n"
        "- Most skin-relevant source count: 3\n"
        "- Most skin-relevant sources: autodock, gnina, rtmscore\n"
        "- Most skin-relevant skin score: 0.8\n"
        "- Most skin-relevant skin tier: high\n"
        "- Most skin-relevant efficacy: hydration (3 papers)\n"
        "- Top targets with skin-efficacy evidence: 2\n"
        "- Reason: top targets include low-or-higher skin-expression support\n"
        "- Reason: top targets include KG skin-efficacy labels\n"
        "\n"
        "## Source Artifacts\n"
        "- admet_ai: `02_admet/admet_ai.json`\n"
        "- admet_report: `02_admet/admet_report.json`\n"
        "- compound: `01_input/compound_canonical.json`\n"
        "- cosmetic_drug_decision: `02b_cosmetic_drug/cosmetic_drug_decision.txt`\n"
        "- cosing_match: `02b_cosmetic_drug/cosing_match.json`\n"
        "- drug_warnings: `02b_cosmetic_drug/drug_warnings.json`\n"
        "- husspred: `02_admet/husspred.json`\n"
        "- pred_skin: `02_admet/pred_skin.json`\n"
        "- skin_sens_decision: `02_admet/skin_sens_decision.txt`\n"
        "- stoptox: `02_admet/stoptox.json`\n"
        "- structural_alerts: `02_admet/structural_alerts.json`\n"
        "- target_comprehensive_autodock: `03_targets/mode_comprehensive/autodock_all_targets.tsv`\n"
        "- target_comprehensive_boltz2: `03_targets/mode_comprehensive/boltz2_affinity_top.tsv`\n"
        "- target_comprehensive_consensus: `03_targets/mode_comprehensive/top50_4way_consensus.csv`\n"
        "- target_comprehensive_gnina: `03_targets/mode_comprehensive/gnina_rescores.tsv`\n"
        "- target_comprehensive_ligand: `03_targets/mode_comprehensive/ligand.pdbqt`\n"
        "- target_comprehensive_pre_rescore: `03_targets/mode_comprehensive/top_pct_pre_rescore.csv`\n"
        "- target_comprehensive_rtmscore: `03_targets/mode_comprehensive/rtmscore_rescores.tsv`\n"
        "- target_fast_autodock: `03_targets/mode_fast/autodock_top5k.tsv`\n"
        "- target_fast_daina_zoete: `03_targets/mode_fast/daina_zoete_proteome.tsv`\n"
        "- target_fast_dti_rrf: `03_targets/mode_fast/dti_rrf_top25pct.csv`\n"
        "- target_fast_psichic: `03_targets/mode_fast/psichic_proteome.tsv`\n"
        "- target_fast_rerank_consensus: `03_targets/mode_fast/top50.csv`\n"
        "- target_ranking: `03_targets/ranked_targets_v3_with_efficacy.csv`\n"
    )


def rewrite_minimal_run_as_sdf_input(run_dir: Path, input_sdf: str) -> None:
    compound_path = run_dir / "01_input" / "compound_canonical.json"
    compound = json.loads(compound_path.read_text())
    compound["input_type"] = "sdf"
    compound["input_sdf"] = input_sdf
    compound.pop("input_smiles", None)
    compound.pop("input_canonical_smiles", None)
    compound_path.write_text(json.dumps(compound))

    summary_path = run_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text())
    summary_compound = summary["compound"]
    summary_compound["input_type"] = "sdf"
    summary_compound["input_sdf"] = input_sdf
    summary_compound.pop("input_smiles", None)
    summary_compound.pop("input_canonical_smiles", None)
    summary_path.write_text(json.dumps(summary))

    report_path = run_dir / "09_report" / "index.html"
    report_path.write_text(
        report_path.read_text()
        .replace("Input type: smiles ", "Input type: sdf ")
        .replace("Input SMILES: CCO ", "")
        .replace("Input canonical SMILES: CCO ", f"Input SDF: {input_sdf} ")
    )

    markdown_path = run_dir / "run_summary.md"
    markdown_path.write_text(
        markdown_path.read_text()
        .replace("- Input type: smiles\n", "- Input type: sdf\n")
        .replace("- Input SMILES: `CCO`\n", "- Input SMILES: `n/a`\n")
        .replace(
            "- Input canonical SMILES: `CCO`\n",
            "- Input canonical SMILES: `n/a`\n",
        )
        .replace(
            "- Input SDF: `n/a`\n",
            f"- Input SDF: `{input_sdf}`\n",
        )
    )


def mark_minimal_run_as_report(run_dir: Path) -> None:
    summary_path = run_dir / "run_summary.json"
    payload = json.loads(summary_path.read_text())
    payload["preset"] = "report"
    payload["artifacts"]["html_report"] = "09_report/index.html"
    summary_path.write_text(json.dumps(payload))

    markdown_path = run_dir / "run_summary.md"
    markdown = markdown_path.read_text()
    markdown = markdown.replace("- Preset: target-id\n", "- Preset: report\n")
    if "- html_report: `09_report/index.html`" not in markdown:
        markdown = markdown.replace(
            "## Source Artifacts\n",
            "## Source Artifacts\n- html_report: `09_report/index.html`\n",
            1,
        )
    markdown_path.write_text(markdown)


def write_unsupported_skin_context_ranking(run_dir: Path) -> None:
    (run_dir / "03_targets" / "ranked_targets_v3_with_efficacy.csv").write_text(
        "target_id,docking_rrf,skin_score,skin_tier,final_score,source_count,sources,efficacy_top1\n"
        "P1,0.9,0.0,very_low,0.85,3,autodock;gnina;rtmscore,hydration (3 papers)\n"
        "P2,0.7,0.0,very_low,0.70,3,autodock;gnina;rtmscore,barrier (2 papers)\n"
    )


def test_verify_safety_contract_accepts_halt_decision(tmp_path: Path) -> None:
    run_dir = tmp_path / "halt_case"
    write_minimal_run(run_dir, safety_decision="HALT")

    res = run_verify(["--run-dir", str(run_dir), "--preset", "safety", "--json"])

    assert res.returncode == 0, res.stderr
    payload = json.loads(res.stdout)
    assert payload["schema_version"] == "skinscout.run_output_verification.v1"
    assert payload["status"] == "ok"


def test_verify_target_id_contract_accepts_valid_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "target_case"
    write_minimal_run(run_dir)

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id", "--json"])

    assert res.returncode == 0, res.stderr
    payload = json.loads(res.stdout)
    assert payload["schema_version"] == "skinscout.run_output_verification.v1"
    assert payload["status"] == "ok"
    assert any(
        check["name"] == "skin-weighted target ranking with efficacy"
        for check in payload["checks"]
    )


def test_verify_report_contract_accepts_sdf_input_provenance(tmp_path: Path) -> None:
    run_dir = tmp_path / "report_sdf_input_case"
    write_minimal_run(run_dir)
    input_sdf = str(tmp_path / "source_ligand.sdf")
    rewrite_minimal_run_as_sdf_input(run_dir, input_sdf)
    mark_minimal_run_as_report(run_dir)

    res = run_verify(["--run-dir", str(run_dir), "--preset", "report", "--json"])

    assert res.returncode == 0, res.stderr
    payload = json.loads(res.stdout)
    assert payload["status"] == "ok"


def test_summarize_outputs_writes_user_facing_summary(tmp_path: Path) -> None:
    run_dir = tmp_path / "summary_case"
    write_minimal_run(run_dir)
    (run_dir / "run_summary.json").unlink()
    metadata = tmp_path / "target_metadata.tsv"
    metadata.write_text(
        "Uniprot\tGene\tGene description\n"
        "P1\tGENE1\tProtein one\n"
        "P2\tGENE2\tProtein two\n"
    )

    res = run_summarize([
        "--run-dir",
        str(run_dir),
        "--preset",
        "target-id",
        "--mode",
        "comprehensive",
        "--target-metadata",
        str(metadata),
    ])

    assert res.returncode == 0, res.stderr
    payload = json.loads((run_dir / "run_summary.json").read_text())
    assert payload["schema_version"] == "skinscout.run_summary.v1"
    assert payload["compound"]["input_type"] == "smiles"
    assert payload["compound"]["input_smiles"] == "CCO"
    assert payload["compound"]["input_canonical_smiles"] == "CCO"
    assert payload["compound"]["canonical_smiles"] == "CCO"
    assert payload["safety"]["skin_sens_decision"] == "PASS"
    assert payload["safety"]["skin_sens_evidence"] == [
        {
            "model": "husspred",
            "status": "ok",
            "call": "negative",
            "probability": 0.2,
        },
        {
            "model": "stoptox",
            "status": "ok",
            "call": "negative",
            "probability": 0.2,
        },
        {
            "model": "pred_skin",
            "status": "ok",
            "call": "negative",
            "probability": 0.2,
        },
    ]
    assert payload["safety"]["admet_risk_assessment"]["endpoints"]["Skin_Reaction"]["risk_level"] == "low"
    assert payload["skin_toxicity"]["decision"] == "PASS"
    assert payload["skin_toxicity"]["toxicity_level"] == "low"
    assert payload["skin_toxicity"]["skin_reaction_risk_level"] == "low"
    assert payload["skin_toxicity"]["skin_reaction_value"] == 0.21
    assert payload["skin_toxicity"]["structural_alerts_present"] is False
    assert payload["skin_toxicity"]["reasons"] == [
        "skin sensitization and Skin_Reaction ADMET risk are low"
    ]
    assert payload["overall_decision"]["decision"] == "PASS"
    assert payload["overall_decision"]["requires_human_review"] is False
    assert payload["overall_decision"]["claimable"] is True
    assert "safety and cosmetic/drug gates passed" in payload["overall_decision"]["reasons"]
    assert payload["overall_decision"]["top_target_id"] == "P1"
    assert payload["target_prediction"]["screened_target_count"] == 4
    assert payload["target_prediction"]["screening_counts"] == {
        "autodock_screened_targets": 4,
        "boltz2_affinity_targets": 3,
        "four_way_consensus_targets": 2,
        "gnina_rescored_targets": 3,
        "pre_rescore_candidates": 3,
        "rtmscore_rescored_targets": 3,
        "skin_weighted_ranked_targets": 2,
    }
    assert payload["target_prediction"]["top_targets"][0]["target_id"] == "P1"
    assert payload["target_prediction"]["top_targets"][0]["gene_symbol"] == "GENE1"
    assert payload["target_prediction"]["top_targets"][0]["protein_name"] == "Protein one"
    assert payload["target_prediction"]["top_targets"][0]["skin_tier"] == "high"
    assert payload["target_prediction"]["top_targets"][0]["cell_type_preferred"] == "keratinocytes"
    assert payload["skin_specialized_binding"]["top_target_id"] == "P1"
    assert payload["skin_specialized_binding"]["top_target_gene_symbol"] == "GENE1"
    assert payload["skin_specialized_binding"]["top_target_protein_name"] == "Protein one"
    assert payload["skin_specialized_binding"]["top_target_final_score"] == 0.85
    assert payload["skin_specialized_binding"]["top_target_docking_rrf"] == 0.9
    assert payload["skin_specialized_binding"]["top_target_source_count"] == 3
    assert payload["skin_specialized_binding"]["top_target_sources"] == [
        "autodock",
        "gnina",
        "rtmscore",
    ]
    assert payload["skin_specialized_binding"]["top_target_skin_tier"] == "high"
    assert payload["skin_specialized_binding"]["top_target_cell_type_preferred"] == "keratinocytes"
    assert payload["skin_specialized_binding"]["top_target_efficacy"] == [
        "hydration (3 papers)"
    ]
    assert payload["skin_specialized_binding"]["top_target_skin_expression_supported"] is True
    assert payload["skin_specialized_binding"]["top_target_skin_efficacy_supported"] is True
    assert payload["skin_specialized_binding"]["top_target_skin_context_supported"] is True
    assert payload["skin_specialized_binding"]["top_targets_with_skin_efficacy"] == 2
    assert payload["skin_specialized_binding"]["skin_context_decision"] == "skin_context_supported"
    assert payload["skin_specialized_binding"]["skin_context_supported"] is True
    assert payload["skin_specialized_binding"]["most_skin_relevant_target"]["cell_type_preferred"] == "keratinocytes"
    assert payload["artifacts"]["admet_ai"] == "02_admet/admet_ai.json"
    assert payload["artifacts"]["structural_alerts"] == "02_admet/structural_alerts.json"
    assert payload["artifacts"]["husspred"] == "02_admet/husspred.json"
    assert payload["artifacts"]["stoptox"] == "02_admet/stoptox.json"
    assert payload["artifacts"]["pred_skin"] == "02_admet/pred_skin.json"
    assert payload["artifacts"]["target_comprehensive_autodock"] == (
        "03_targets/mode_comprehensive/autodock_all_targets.tsv"
    )
    assert payload["artifacts"]["target_comprehensive_consensus"] == (
        "03_targets/mode_comprehensive/top50_4way_consensus.csv"
    )
    markdown = (run_dir / "run_summary.md").read_text()
    assert "Overall Decision" in markdown
    assert "Input type: smiles" in markdown
    assert "Input SMILES: `CCO`" in markdown
    assert "Input canonical SMILES: `CCO`" in markdown
    assert "InChIKey: LFQSCWFLJHTTHZ-UHFFFAOYSA-N" in markdown
    assert "Skin Toxicity" in markdown
    assert "Toxicity level: low" in markdown
    assert "Skin-sens model" in markdown
    assert "husspred" in markdown
    assert "pred_skin" in markdown
    assert "0.2" in markdown
    assert "Skin-Specialized Binding" in markdown
    assert "Skin context decision: skin_context_supported" in markdown
    assert "Top target gene: GENE1" in markdown
    assert "Top target protein: Protein one" in markdown
    assert "Top target final score: 0.85" in markdown
    assert "Top target sources: autodock, gnina, rtmscore" in markdown
    assert "Top target HPA cell context: keratinocytes" in markdown
    assert "Top target skin context supported: yes" in markdown
    assert "Most skin-relevant gene: GENE1" in markdown
    assert "Most skin-relevant protein: Protein one" in markdown
    assert "Most skin-relevant final score: 0.85" in markdown
    assert "Most skin-relevant sources: autodock, gnina, rtmscore" in markdown
    assert "Most skin-relevant HPA cell context: keratinocytes" in markdown
    assert "Most skin-relevant efficacy: hydration (3 papers)" in markdown
    assert "HPA cell context is an expression label" in markdown
    assert "Skin_Reaction" in markdown
    assert "LD50_Zhu" in markdown
    assert "Solubility_AqSolDB" in markdown
    assert "QED" in markdown
    assert "tpsa" in markdown
    assert "GENE1" in markdown
    assert "Protein one" in markdown
    assert "Source Artifacts" in markdown
    assert "02_admet/admet_ai.json" in markdown
    assert "03_targets/mode_comprehensive/top50_4way_consensus.csv" in markdown


def test_summarize_safety_omits_stale_target_and_report_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "summary_safety_stale_target_artifacts_case"
    write_minimal_run(run_dir)
    (run_dir / "run_summary.json").unlink()
    (run_dir / "run_summary.md").unlink()

    res = run_summarize(["--run-dir", str(run_dir), "--preset", "safety"])

    assert res.returncode == 0, res.stderr
    payload = json.loads((run_dir / "run_summary.json").read_text())
    assert payload["preset"] == "safety"
    assert "target_prediction" not in payload
    assert "skin_specialized_binding" not in payload
    assert "target_ranking" not in payload["artifacts"]
    assert "target_fast_psichic" not in payload["artifacts"]
    assert "target_comprehensive_consensus" not in payload["artifacts"]
    assert "html_report" not in payload["artifacts"]
    markdown = (run_dir / "run_summary.md").read_text()
    assert "03_targets/" not in markdown
    assert "09_report/" not in markdown


def test_summarize_rejects_invalid_compound_smiles(tmp_path: Path) -> None:
    run_dir = tmp_path / "summary_bad_compound_smiles_case"
    write_minimal_run(run_dir)
    (run_dir / "run_summary.json").unlink()
    payload = json.loads((run_dir / "01_input" / "compound_canonical.json").read_text())
    payload["canonical_smiles"] = "not a smiles"
    (run_dir / "01_input" / "compound_canonical.json").write_text(json.dumps(payload))

    res = run_summarize(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 1
    assert "invalid canonical_smiles" in res.stderr
    assert not (run_dir / "run_summary.json").exists()


def test_summarize_rejects_noncanonical_compound_smiles(tmp_path: Path) -> None:
    run_dir = tmp_path / "summary_noncanonical_compound_smiles_case"
    write_minimal_run(run_dir)
    (run_dir / "run_summary.json").unlink()
    payload = json.loads((run_dir / "01_input" / "compound_canonical.json").read_text())
    payload["canonical_smiles"] = "C(C)O"
    (run_dir / "01_input" / "compound_canonical.json").write_text(json.dumps(payload))

    res = run_summarize(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 1
    assert "canonical_smiles is not canonical" in res.stderr
    assert not (run_dir / "run_summary.json").exists()


def test_summarize_rejects_blank_compound_inchikey(tmp_path: Path) -> None:
    run_dir = tmp_path / "summary_blank_compound_inchikey_case"
    write_minimal_run(run_dir)
    (run_dir / "run_summary.json").unlink()
    payload = json.loads((run_dir / "01_input" / "compound_canonical.json").read_text())
    payload["inchikey"] = " "
    (run_dir / "01_input" / "compound_canonical.json").write_text(json.dumps(payload))

    res = run_summarize(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 1
    assert "missing non-empty inchikey" in res.stderr
    assert not (run_dir / "run_summary.json").exists()


def test_summarize_rejects_compound_inchikey_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "summary_compound_inchikey_mismatch_case"
    write_minimal_run(run_dir)
    (run_dir / "run_summary.json").unlink()
    payload = json.loads((run_dir / "01_input" / "compound_canonical.json").read_text())
    payload["inchikey"] = "WRONG-INCHIKEY"
    (run_dir / "01_input" / "compound_canonical.json").write_text(json.dumps(payload))

    res = run_summarize(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 1
    assert "inchikey does not match canonical_smiles" in res.stderr
    assert not (run_dir / "run_summary.json").exists()


def test_summarize_rejects_smiles_metadata_with_sdf_provenance(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "summary_smiles_with_sdf_provenance_case"
    write_minimal_run(run_dir)
    (run_dir / "run_summary.json").unlink()
    (run_dir / "run_summary.md").unlink()
    payload = json.loads((run_dir / "01_input" / "compound_canonical.json").read_text())
    payload["input_sdf"] = str(tmp_path / "stale.sdf")
    (run_dir / "01_input" / "compound_canonical.json").write_text(json.dumps(payload))

    res = run_summarize(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 1
    assert "input_type 'smiles' cannot include input_sdf" in res.stderr
    assert not (run_dir / "run_summary.json").exists()


def test_summarize_rejects_sdf_metadata_with_smiles_provenance(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "summary_sdf_with_smiles_provenance_case"
    write_minimal_run(run_dir)
    rewrite_minimal_run_as_sdf_input(run_dir, str(tmp_path / "source_ligand.sdf"))
    (run_dir / "run_summary.json").unlink()
    (run_dir / "run_summary.md").unlink()
    payload = json.loads((run_dir / "01_input" / "compound_canonical.json").read_text())
    payload["input_smiles"] = "CCO"
    payload["input_canonical_smiles"] = "CCO"
    (run_dir / "01_input" / "compound_canonical.json").write_text(json.dumps(payload))

    res = run_summarize(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 1
    assert "input_type 'sdf' cannot include input_smiles" in res.stderr
    assert not (run_dir / "run_summary.json").exists()


def test_summarize_rejects_missing_core_admet_endpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "summary_missing_admet_endpoint_case"
    write_minimal_run(run_dir)
    (run_dir / "run_summary.json").unlink()
    payload = json.loads((run_dir / "02_admet" / "admet_report.json").read_text())
    del payload["admet_ai_predictions"]["Skin_Reaction"]
    (run_dir / "02_admet" / "admet_report.json").write_text(json.dumps(payload))

    res = run_summarize(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 1
    assert "missing ADMET-AI endpoints" in res.stderr
    assert "Skin_Reaction" in res.stderr
    assert not (run_dir / "run_summary.json").exists()


def test_summarize_allows_degraded_admet_when_explicit(tmp_path: Path) -> None:
    run_dir = tmp_path / "summary_degraded_admet_allowed_case"
    write_minimal_run(run_dir)
    (run_dir / "run_summary.json").unlink()
    payload = json.loads((run_dir / "02_admet" / "admet_report.json").read_text())
    payload["skin_sens"]["degraded"] = True
    payload["admet_ai_predictions"] = None
    (run_dir / "02_admet" / "admet_report.json").write_text(json.dumps(payload))

    res = run_summarize([
        "--run-dir",
        str(run_dir),
        "--preset",
        "target-id",
        "--allow-degraded",
    ])

    assert res.returncode == 0, res.stderr
    summary = json.loads((run_dir / "run_summary.json").read_text())
    assert summary["safety"]["degraded"] is True
    assert summary["safety"]["admet_metrics"] == {}
    assert summary["skin_toxicity"]["decision"] == "REVIEW"


def test_summarize_rejects_nonnumeric_core_admet_endpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "summary_nonnumeric_core_admet_case"
    write_minimal_run(run_dir)
    (run_dir / "run_summary.json").unlink()
    payload = json.loads((run_dir / "02_admet" / "admet_report.json").read_text())
    payload["admet_ai_predictions"]["Caco2_Wang"] = "not-a-number"
    (run_dir / "02_admet" / "admet_report.json").write_text(json.dumps(payload))

    res = run_summarize(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 1
    assert "admet_ai_predictions.Caco2_Wang must be numeric" in res.stderr
    assert not (run_dir / "run_summary.json").exists()


def test_summarize_rejects_admet_source_report_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "summary_admet_source_mismatch_case"
    write_minimal_run(run_dir)
    (run_dir / "run_summary.json").unlink()
    payload = json.loads((run_dir / "02_admet" / "admet_ai.json").read_text())
    payload["predictions"]["DILI"] = 0.77
    (run_dir / "02_admet" / "admet_ai.json").write_text(json.dumps(payload))

    res = run_summarize(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 1
    assert "source/report mismatch for endpoint DILI" in res.stderr
    assert not (run_dir / "run_summary.json").exists()


def test_summarize_rejects_admet_source_smiles_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "summary_admet_source_smiles_mismatch_case"
    write_minimal_run(run_dir)
    (run_dir / "run_summary.json").unlink()
    payload = json.loads((run_dir / "02_admet" / "admet_ai.json").read_text())
    payload["smiles"] = "CCC"
    (run_dir / "02_admet" / "admet_ai.json").write_text(json.dumps(payload))

    res = run_summarize(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 1
    assert (
        "ADMET-AI source predictions smiles does not match compound canonical_smiles"
        in res.stderr
    )
    assert not (run_dir / "run_summary.json").exists()


def test_summarize_rejects_skin_sens_source_report_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "summary_skin_sens_source_mismatch_case"
    write_minimal_run(run_dir)
    (run_dir / "run_summary.json").unlink()
    payload = json.loads((run_dir / "02_admet" / "husspred.json").read_text())
    payload["skin_sens_call"] = "positive"
    payload["skin_sens_probability"] = 0.8
    (run_dir / "02_admet" / "husspred.json").write_text(json.dumps(payload))

    res = run_summarize(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 1
    assert "source/report mismatch for husspred call" in res.stderr
    assert not (run_dir / "run_summary.json").exists()


def test_summarize_rejects_structural_alert_source_report_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "summary_structural_alert_source_mismatch_case"
    write_minimal_run(run_dir)
    (run_dir / "run_summary.json").unlink()
    payload = json.loads((run_dir / "02_admet" / "structural_alerts.json").read_text())
    payload["matches"]["NIH"] = ["nitro_aromatic"]
    payload["any_nih"] = True
    (run_dir / "02_admet" / "structural_alerts.json").write_text(json.dumps(payload))

    res = run_summarize(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 1
    assert "source/report mismatch for structural-alert flag any_nih" in res.stderr
    assert not (run_dir / "run_summary.json").exists()


def test_summarize_rejects_target_source_count_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "summary_bad_target_sources_case"
    write_minimal_run(run_dir)
    (run_dir / "run_summary.json").unlink()
    (run_dir / "03_targets" / "ranked_targets_v3_with_efficacy.csv").write_text(
        "target_id,docking_rrf,skin_score,skin_tier,final_score,source_count,sources,efficacy_top1\n"
        "P1,0.9,0.8,high,0.85,3,autodock;gnina,hydration (3 papers)\n"
        "P2,0.7,0.6,medium,0.70,3,autodock;gnina;rtmscore,barrier (2 papers)\n"
    )

    res = run_summarize(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 1
    assert "source_count=3 but sources lists 2 label(s)" in res.stderr
    assert not (run_dir / "run_summary.json").exists()


def test_summarize_rejects_unsorted_target_ranking(tmp_path: Path) -> None:
    run_dir = tmp_path / "summary_unsorted_targets_case"
    write_minimal_run(run_dir)
    (run_dir / "run_summary.json").unlink()
    (run_dir / "03_targets" / "ranked_targets_v3_with_efficacy.csv").write_text(
        "target_id,docking_rrf,skin_score,skin_tier,final_score,source_count,sources,efficacy_top1\n"
        "P1,0.9,0.8,high,0.70,3,autodock;gnina;rtmscore,hydration (3 papers)\n"
        "P2,0.7,0.6,medium,0.85,3,autodock;gnina;rtmscore,barrier (2 papers)\n"
    )

    res = run_summarize(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 1
    assert "target ranking final_score must be sorted descending" in res.stderr
    assert not (run_dir / "run_summary.json").exists()


def test_summarize_outputs_flags_unsupported_skin_context_for_review(tmp_path: Path) -> None:
    run_dir = tmp_path / "summary_unsupported_skin_context_case"
    write_minimal_run(run_dir)
    write_unsupported_skin_context_ranking(run_dir)
    (run_dir / "run_summary.json").unlink()

    res = run_summarize(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 0, res.stderr
    payload = json.loads((run_dir / "run_summary.json").read_text())
    expected_reason = (
        "skin-specialized binding context requires review: "
        "skin_efficacy_literature_only"
    )
    assert payload["skin_specialized_binding"]["skin_context_decision"] == "skin_efficacy_literature_only"
    assert payload["skin_specialized_binding"]["skin_context_supported"] is False
    assert payload["skin_specialized_binding"]["skin_expression_supported"] is False
    assert payload["skin_specialized_binding"]["skin_efficacy_supported"] is True
    assert payload["overall_decision"]["decision"] == "FLAG_HIGH"
    assert payload["overall_decision"]["requires_human_review"] is True
    assert payload["overall_decision"]["recommended_action"] == "review_before_claim"
    assert expected_reason in payload["overall_decision"]["reasons"]
    assert expected_reason in (run_dir / "run_summary.md").read_text()
    verify = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])
    assert verify.returncode == 0, verify.stdout + verify.stderr


def test_summarize_distinguishes_top_binding_target_skin_context(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "summary_top_binding_skin_context_case"
    write_minimal_run(run_dir)
    (run_dir / "run_summary.json").unlink()
    (run_dir / "03_targets" / "ranked_targets_v3_with_efficacy.csv").write_text(
        "target_id,docking_rrf,skin_score,skin_tier,final_score,source_count,sources,efficacy_top1\n"
        "P1,0.9,0.0,very_low,0.90,3,autodock;gnina;rtmscore,hydration (3 papers)\n"
        "P2,0.7,0.8,high,0.85,3,autodock;gnina;rtmscore,barrier (2 papers)\n"
    )

    res = run_summarize(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 0, res.stderr
    payload = json.loads((run_dir / "run_summary.json").read_text())
    binding = payload["skin_specialized_binding"]
    assert binding["top_target_id"] == "P1"
    assert binding["top_target_skin_expression_supported"] is False
    assert binding["top_target_skin_efficacy_supported"] is True
    assert binding["top_target_skin_context_supported"] is False
    assert binding["skin_context_decision"] == "skin_context_supported"
    assert binding["skin_context_supported"] is True
    assert binding["most_skin_relevant_target"]["target_id"] == "P2"
    assert payload["overall_decision"]["decision"] == "FLAG_HIGH"
    assert payload["overall_decision"]["requires_human_review"] is True
    assert payload["overall_decision"]["recommended_action"] == "review_before_claim"
    assert any(
        "top binding target lacks direct skin context support" in reason
        for reason in payload["overall_decision"]["reasons"]
    )
    assert "Top target skin context supported: no" in (
        run_dir / "run_summary.md"
    ).read_text()
    verify = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])
    assert verify.returncode == 0, verify.stdout + verify.stderr


def test_verify_rejects_overall_pass_without_top_binding_target_skin_context(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "verify_top_binding_skin_context_case"
    write_minimal_run(run_dir)
    (run_dir / "run_summary.json").unlink()
    (run_dir / "03_targets" / "ranked_targets_v3_with_efficacy.csv").write_text(
        "target_id,docking_rrf,skin_score,skin_tier,final_score,source_count,sources,efficacy_top1\n"
        "P1,0.9,0.0,very_low,0.90,3,autodock;gnina;rtmscore,hydration (3 papers)\n"
        "P2,0.7,0.8,high,0.85,3,autodock;gnina;rtmscore,barrier (2 papers)\n"
    )
    res = run_summarize(["--run-dir", str(run_dir), "--preset", "target-id"])
    assert res.returncode == 0, res.stderr
    summary_path = run_dir / "run_summary.json"
    payload = json.loads(summary_path.read_text())
    payload["overall_decision"].update(
        {
            "decision": "PASS",
            "requires_human_review": False,
            "recommended_action": "proceed",
            "claimable": True,
        }
    )
    summary_path.write_text(json.dumps(payload))

    verify = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert verify.returncode == 2
    assert (
        "overall_decision cannot PASS without top binding target skin context"
        in verify.stdout
    )


def test_summarize_outputs_flags_high_admet_risk(tmp_path: Path) -> None:
    run_dir = tmp_path / "summary_high_admet_case"
    write_minimal_run(run_dir)
    (run_dir / "run_summary.json").unlink()
    report_path = run_dir / "02_admet" / "admet_report.json"
    report = json.loads(report_path.read_text())
    report["admet_ai_predictions"]["DILI"] = 0.72
    report_path.write_text(json.dumps(report))
    admet_ai_path = run_dir / "02_admet" / "admet_ai.json"
    admet_ai = json.loads(admet_ai_path.read_text())
    admet_ai["predictions"]["DILI"] = 0.72
    admet_ai_path.write_text(json.dumps(admet_ai))

    res = run_summarize(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 0, res.stderr
    payload = json.loads((run_dir / "run_summary.json").read_text())
    assert payload["safety"]["admet_risk_assessment"]["endpoints"]["DILI"]["risk_level"] == "high"
    assert payload["safety"]["admet_risk_assessment"]["high_risk_endpoints"] == ["DILI"]
    assert payload["overall_decision"]["decision"] == "FLAG_HIGH"
    assert any("high ADMET risk endpoints: DILI" in reason for reason in payload["overall_decision"]["reasons"])


def test_summarize_outputs_flags_moderate_skin_reaction_for_review(tmp_path: Path) -> None:
    run_dir = tmp_path / "summary_moderate_skin_reaction_case"
    write_minimal_run(run_dir)
    (run_dir / "run_summary.json").unlink()
    report_path = run_dir / "02_admet" / "admet_report.json"
    report = json.loads(report_path.read_text())
    report["admet_ai_predictions"]["Skin_Reaction"] = 0.31
    report_path.write_text(json.dumps(report))
    admet_ai_path = run_dir / "02_admet" / "admet_ai.json"
    admet_ai = json.loads(admet_ai_path.read_text())
    admet_ai["predictions"]["Skin_Reaction"] = 0.31
    admet_ai_path.write_text(json.dumps(admet_ai))

    res = run_summarize(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 0, res.stderr
    payload = json.loads((run_dir / "run_summary.json").read_text())
    assert payload["safety"]["admet_risk_assessment"]["endpoints"]["Skin_Reaction"]["risk_level"] == "moderate"
    assert payload["skin_toxicity"]["decision"] == "REVIEW"
    assert payload["skin_toxicity"]["toxicity_level"] == "moderate"
    assert payload["skin_toxicity"]["skin_reaction_value"] == 0.31
    assert payload["skin_toxicity"]["reasons"] == [
        "Skin_Reaction ADMET risk is moderate"
    ]
    assert payload["overall_decision"]["decision"] == "FLAG_HIGH"
    assert "skin toxicity requires review" in payload["overall_decision"]["reasons"]
    assert any(
        "Skin_Reaction ADMET risk requires review: moderate" in reason
        for reason in payload["overall_decision"]["reasons"]
    )
    verify = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])
    assert verify.returncode == 0, verify.stdout + verify.stderr


def test_overall_decision_directly_gates_skin_toxicity_review() -> None:
    module = load_summarize_module()

    overall = module._overall_decision(
        preset="safety",
        safety={
            "skin_sens_decision": "PASS",
            "degraded": False,
            "missing_models": [],
            "admet_risk_assessment": {
                "high_risk_endpoints": [],
                "endpoints": {},
            },
            "structural_alert_flags": {},
        },
        skin_toxicity={"decision": "REVIEW"},
        cosmetic_drug={"decision": "PROCEED", "n_warnings": 0},
        target_prediction=None,
        skin_specialized_binding=None,
    )

    assert overall["decision"] == "FLAG_HIGH"
    assert overall["requires_human_review"] is True
    assert overall["recommended_action"] == "review_before_claim"
    assert overall["claimable"] is False
    assert "skin toxicity requires review" in overall["reasons"]


def test_verify_rejects_missing_user_facing_summary(tmp_path: Path) -> None:
    run_dir = tmp_path / "missing_summary_case"
    write_minimal_run(run_dir)
    (run_dir / "run_summary.json").unlink()

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert "[missing] user-facing run summary" in res.stdout


def test_verify_rejects_missing_markdown_user_facing_summary(tmp_path: Path) -> None:
    run_dir = tmp_path / "missing_markdown_summary_case"
    write_minimal_run(run_dir)
    (run_dir / "run_summary.md").unlink()

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert "[missing] user-facing markdown run summary" in res.stdout


def test_verify_rejects_stale_markdown_top_target(tmp_path: Path) -> None:
    run_dir = tmp_path / "stale_markdown_summary_case"
    write_minimal_run(run_dir)
    (run_dir / "run_summary.md").write_text(
        "# SkinScout Run Summary\n"
        "\n"
        "Run: stale_markdown_summary_case\n"
        "Compound: CCO\n"
        "Decision: PASS\n"
        "Recommended action: proceed\n"
        "Skin sensitization: PASS\n"
        "Cosmetic decision: PROCEED\n"
        "AMES ClinTox DILI Skin_Reaction hERG low\n"
        "Top target: P9\n"
    )

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert "missing top target 1 target_id: P1" in res.stdout


def test_verify_rejects_stale_markdown_skin_sens_evidence(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "stale_markdown_skin_sens_evidence_case"
    write_minimal_run(run_dir)
    markdown_path = run_dir / "run_summary.md"
    markdown = markdown_path.read_text()
    markdown = markdown.replace("| husspred | ok | negative | 0.2 |\n", "")
    markdown_path.write_text(markdown)

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert "missing skin-sens evidence row 1" in res.stdout
    assert "husspred" in res.stdout
    assert "0.2" in res.stdout


def test_verify_rejects_stale_markdown_target_row_details(tmp_path: Path) -> None:
    run_dir = tmp_path / "stale_markdown_target_row_case"
    write_minimal_run(run_dir)
    markdown_path = run_dir / "run_summary.md"
    markdown = markdown_path.read_text()
    markdown = markdown.replace(
        "| 1 | P1 | GENE1 | Protein one | 0.85 | 0.8 | high | 0.9 | autodock, gnina, rtmscore | hydration (3 papers) |",
        "| 1 | P1 | WRONG | Wrong protein | 0.99 | 0.1 | high | 0.2 | psichic | barrier (9 papers) |",
    )
    markdown_path.write_text(markdown)

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert "missing top target 1 row details" in res.stdout
    assert "0.85" in res.stdout
    assert "autodock, gnina, rtmscore" in res.stdout
    assert "hydration (3 papers)" in res.stdout


def test_verify_rejects_stale_markdown_skin_binding_details(tmp_path: Path) -> None:
    run_dir = tmp_path / "stale_markdown_skin_binding_case"
    write_minimal_run(run_dir)
    markdown_path = run_dir / "run_summary.md"
    markdown = markdown_path.read_text()
    markdown = markdown.replace(
        "- Top target skin score: 0.8\n",
        "- Top target skin score: 0.1\n",
    )
    markdown = markdown.replace(
        "- Top target gene: GENE1\n",
        "- Top target gene: WRONG\n",
    )
    markdown = markdown.replace(
        "- Top target protein: Protein one\n",
        "- Top target protein: Wrong protein\n",
    )
    markdown = markdown.replace(
        "- Top target skin efficacy: hydration (3 papers)\n",
        "- Top target skin efficacy: barrier (9 papers)\n",
    )
    markdown = markdown.replace(
        "- Most skin-relevant final score: 0.85\n",
        "- Most skin-relevant final score: 0.5\n",
    )
    markdown = markdown.replace(
        "- Most skin-relevant gene: GENE1\n",
        "- Most skin-relevant gene: WRONG\n",
    )
    markdown = markdown.replace(
        "- Most skin-relevant protein: Protein one\n",
        "- Most skin-relevant protein: Wrong protein\n",
    )
    markdown = markdown.replace(
        "- Most skin-relevant sources: autodock, gnina, rtmscore\n",
        "- Most skin-relevant sources: psichic\n",
    )
    markdown = markdown.replace(
        "- Most skin-relevant efficacy: hydration (3 papers)\n",
        "- Most skin-relevant efficacy: barrier (9 papers)\n",
    )
    markdown_path.write_text(markdown)

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert "missing skin binding top target gene: GENE1" in res.stdout
    assert "missing skin binding top target protein: Protein one" in res.stdout
    assert "missing skin binding top skin score: 0.8" in res.stdout
    assert "missing skin binding top efficacy: hydration (3 papers)" in res.stdout
    assert "missing most skin-relevant gene: GENE1" in res.stdout
    assert "missing most skin-relevant protein: Protein one" in res.stdout
    assert "missing most skin-relevant final_score: 0.85" in res.stdout
    assert "missing most skin-relevant sources: autodock, gnina, rtmscore" in res.stdout
    assert "missing most skin-relevant efficacy: hydration (3 papers)" in res.stdout


def test_verify_rejects_stale_markdown_artifact_path(tmp_path: Path) -> None:
    run_dir = tmp_path / "stale_markdown_artifact_case"
    write_minimal_run(run_dir)
    markdown_path = run_dir / "run_summary.md"
    markdown = markdown_path.read_text()
    markdown = markdown.replace(
        "03_targets/mode_fast/psichic_proteome.tsv",
        "03_targets/mode_fast/stale_psichic.tsv",
    )
    markdown_path.write_text(markdown)

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id", "--mode", "fast"])

    assert res.returncode == 2
    assert "missing artifact path target_fast_psichic" in res.stdout
    assert "03_targets/mode_fast/psichic_proteome.tsv" in res.stdout


def test_verify_rejects_summary_top_target_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "summary_target_mismatch_case"
    write_minimal_run(run_dir)
    payload = json.loads((run_dir / "run_summary.json").read_text())
    payload["target_prediction"]["top_targets"][0]["target_id"] = "P9"
    (run_dir / "run_summary.json").write_text(json.dumps(payload))

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert "top target_id mismatch" in res.stdout


def test_verify_rejects_summary_screening_count_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "summary_screening_count_mismatch_case"
    write_minimal_run(run_dir)
    payload = json.loads((run_dir / "run_summary.json").read_text())
    payload["target_prediction"]["screened_target_count"] = 999
    payload["target_prediction"]["screening_counts"]["psichic_proteome_targets"] = 999
    (run_dir / "run_summary.json").write_text(json.dumps(payload))

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert "target_prediction.screening_counts mismatch" in res.stdout
    assert "target_prediction.screened_target_count mismatch" in res.stdout


def test_verify_rejects_summary_non_top_target_row_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "summary_non_top_target_mismatch_case"
    write_minimal_run(run_dir)
    payload = json.loads((run_dir / "run_summary.json").read_text())
    payload["target_prediction"]["top_n"] = 1
    payload["target_prediction"]["top_targets"][1]["target_id"] = "P9"
    payload["target_prediction"]["top_targets"][1]["sources"] = ["psichic"]
    payload["target_prediction"]["top_targets"][1]["efficacy"] = []
    (run_dir / "run_summary.json").write_text(json.dumps(payload))

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert "target_prediction.top_n mismatch" in res.stdout
    assert "top_targets[1].target_id mismatch" in res.stdout
    assert "top_targets[1].sources mismatch" in res.stdout
    assert "top_targets[1].efficacy mismatch" in res.stdout


def test_verify_rejects_summary_overall_decision_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "summary_overall_mismatch_case"
    write_minimal_run(run_dir, safety_decision="HALT")
    payload = json.loads((run_dir / "run_summary.json").read_text())
    payload["overall_decision"]["decision"] = "PASS"
    payload["overall_decision"]["requires_human_review"] = False
    (run_dir / "run_summary.json").write_text(json.dumps(payload))

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert "overall_decision must HALT on skin-sens HALT" in res.stdout


def test_verify_rejects_summary_recommended_action_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "summary_recommended_action_mismatch_case"
    write_minimal_run(run_dir)
    payload = json.loads((run_dir / "run_summary.json").read_text())
    payload["overall_decision"]["recommended_action"] = "review_before_claim"
    (run_dir / "run_summary.json").write_text(json.dumps(payload))
    markdown = (run_dir / "run_summary.md").read_text()
    markdown = markdown.replace(
        "Recommended action: proceed",
        "Recommended action: review_before_claim",
    )
    (run_dir / "run_summary.md").write_text(markdown)

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert "overall_decision recommended_action mismatch" in res.stdout


def test_verify_rejects_summary_claimable_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "summary_claimable_mismatch_case"
    write_minimal_run(run_dir)
    payload = json.loads((run_dir / "run_summary.json").read_text())
    payload["overall_decision"]["claimable"] = False
    (run_dir / "run_summary.json").write_text(json.dumps(payload))
    markdown = (run_dir / "run_summary.md").read_text()
    markdown = markdown.replace("Claimable: yes", "Claimable: no")
    (run_dir / "run_summary.md").write_text(markdown)

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert "overall_decision claimable mismatch" in res.stdout


def test_verify_rejects_markdown_missing_requires_human_review(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "markdown_missing_requires_human_review_case"
    write_minimal_run(run_dir)
    markdown_path = run_dir / "run_summary.md"
    markdown_path.write_text(
        markdown_path.read_text().replace("- Requires human review: no\n", "")
    )

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert "missing requires_human_review: no" in res.stdout


def test_verify_rejects_missing_markdown_run_input_sdf(tmp_path: Path) -> None:
    run_dir = tmp_path / "markdown_missing_run_input_sdf_case"
    write_minimal_run(run_dir)
    markdown_path = run_dir / "run_summary.md"
    markdown_path.write_text(
        markdown_path.read_text().replace("- Input SDF: `n/a`\n", "")
    )

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert "missing input_sdf: n/a" in res.stdout


def test_verify_rejects_missing_markdown_overall_claimable(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "markdown_missing_overall_claimable_case"
    write_minimal_run(run_dir)
    markdown_path = run_dir / "run_summary.md"
    markdown_path.write_text(
        markdown_path.read_text().replace("- Claimable: yes\n", "")
    )

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert "missing claimable: yes" in res.stdout


def test_verify_rejects_missing_markdown_safety_missing_models(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "markdown_missing_safety_models_case"
    write_minimal_run(run_dir)
    markdown_path = run_dir / "run_summary.md"
    markdown = markdown_path.read_text()
    markdown = markdown.replace(
        "## Safety And ADMET\n"
        "- Skin sensitization: PASS\n"
        "- Degraded evidence: no\n"
        "- Missing skin-sens models: none\n",
        "## Safety And ADMET\n"
        "- Skin sensitization: PASS\n"
        "- Degraded evidence: no\n",
    )
    markdown_path.write_text(markdown)

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert "missing safety missing skin-sens models: none" in res.stdout


def test_verify_rejects_markdown_missing_skin_reaction_value(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "markdown_missing_skin_reaction_value_case"
    write_minimal_run(run_dir)
    markdown_path = run_dir / "run_summary.md"
    markdown_path.write_text(
        markdown_path.read_text().replace("- Skin_Reaction value: 0.21\n", "")
    )

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert "missing Skin_Reaction value: 0.21" in res.stdout


def test_verify_rejects_markdown_missing_skin_toxicity_degraded_status(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "markdown_missing_skin_toxicity_degraded_case"
    write_minimal_run(run_dir)
    markdown_path = run_dir / "run_summary.md"
    markdown_path.write_text(
        markdown_path.read_text().replace("- Degraded evidence: no\n", "", 1)
    )

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert "missing degraded evidence: no" in res.stdout


def test_verify_rejects_summary_admet_risk_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "summary_admet_risk_mismatch_case"
    write_minimal_run(run_dir)
    payload = json.loads((run_dir / "run_summary.json").read_text())
    payload["safety"]["admet_risk_assessment"]["endpoints"]["AMES"]["risk_level"] = "high"
    payload["safety"]["admet_risk_assessment"]["high_risk_endpoints"] = ["AMES"]
    (run_dir / "run_summary.json").write_text(json.dumps(payload))

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert "ADMET risk level mismatch for AMES" in res.stdout
    assert "ADMET risk high_risk_endpoints mismatch" in res.stdout


def test_verify_rejects_summary_skin_sens_calls_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "summary_skin_sens_calls_mismatch_case"
    write_minimal_run(run_dir)
    payload = json.loads((run_dir / "run_summary.json").read_text())
    payload["safety"]["skin_sens_calls"]["husspred"] = "positive"
    (run_dir / "run_summary.json").write_text(json.dumps(payload))

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert "skin_sens_calls mismatch" in res.stdout


def test_verify_rejects_summary_skin_sens_evidence_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "summary_skin_sens_evidence_mismatch_case"
    write_minimal_run(run_dir)
    payload = json.loads((run_dir / "run_summary.json").read_text())
    payload["safety"]["skin_sens_evidence"][0]["probability"] = 0.99
    (run_dir / "run_summary.json").write_text(json.dumps(payload))

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert "skin_sens_evidence mismatch" in res.stdout


def test_verify_rejects_summary_admet_metric_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "summary_admet_metric_mismatch_case"
    write_minimal_run(run_dir)
    payload = json.loads((run_dir / "run_summary.json").read_text())
    payload["safety"]["admet_metrics"]["Skin_Reaction"] = 0.22
    (run_dir / "run_summary.json").write_text(json.dumps(payload))

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert "ADMET metric mismatch for Skin_Reaction" in res.stdout


def test_verify_rejects_missing_summary_admet_metric(tmp_path: Path) -> None:
    run_dir = tmp_path / "summary_missing_admet_metric_case"
    write_minimal_run(run_dir)
    payload = json.loads((run_dir / "run_summary.json").read_text())
    del payload["safety"]["admet_metrics"]["QED"]
    (run_dir / "run_summary.json").write_text(json.dumps(payload))

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert "missing ADMET metric QED" in res.stdout


def test_verify_rejects_stale_markdown_admet_risk_row(tmp_path: Path) -> None:
    run_dir = tmp_path / "markdown_stale_admet_risk_row_case"
    write_minimal_run(run_dir)
    markdown_path = run_dir / "run_summary.md"
    markdown_path.write_text(
        markdown_path.read_text().replace(
            "| Skin_Reaction | 0.21 | low |\n",
            "| Skin_Reaction | 0.99 | high |\n",
        )
    )

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert "missing ADMET risk row Skin_Reaction: Skin_Reaction | 0.21 | low" in (
        res.stdout
    )


def test_verify_rejects_stale_markdown_admet_metric_row(tmp_path: Path) -> None:
    run_dir = tmp_path / "markdown_stale_admet_metric_row_case"
    write_minimal_run(run_dir)
    markdown_path = run_dir / "run_summary.md"
    markdown_path.write_text(
        markdown_path.read_text().replace("| QED | 0.4068 |\n", "| QED | 0.9 |\n")
    )

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert "missing ADMET metric row QED: QED | 0.4068" in res.stdout


def test_verify_rejects_missing_markdown_high_admet_risk_summary(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "markdown_missing_high_admet_summary_case"
    write_minimal_run(run_dir)
    markdown_path = run_dir / "run_summary.md"
    markdown_path.write_text(
        markdown_path.read_text().replace("- High risk endpoints: none\n", "")
    )

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert "missing high ADMET risk endpoints: none" in res.stdout


def test_verify_rejects_stale_markdown_moderate_admet_risk_summary(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "markdown_stale_moderate_admet_summary_case"
    write_minimal_run(run_dir)
    markdown_path = run_dir / "run_summary.md"
    markdown_path.write_text(
        markdown_path.read_text().replace(
            "- Moderate risk endpoints: none\n",
            "- Moderate risk endpoints: Skin_Reaction\n",
        )
    )

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert "missing moderate ADMET risk endpoints: none" in res.stdout


def test_verify_rejects_missing_markdown_cosmetic_drug_decision(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "markdown_missing_cosmetic_drug_decision_case"
    write_minimal_run(run_dir)
    markdown_path = run_dir / "run_summary.md"
    markdown_path.write_text(
        markdown_path.read_text().replace("- Decision: PROCEED\n", "", 1)
    )

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert "missing cosmetic/drug decision: PROCEED" in res.stdout


def test_verify_rejects_missing_markdown_drug_policy(tmp_path: Path) -> None:
    run_dir = tmp_path / "markdown_missing_drug_policy_case"
    write_minimal_run(run_dir)
    markdown_path = run_dir / "run_summary.md"
    markdown_path.write_text(
        markdown_path.read_text().replace("- Drug policy: moderate\n", "")
    )

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert "missing drug policy: moderate" in res.stdout


def test_verify_rejects_stale_markdown_cosing_level(tmp_path: Path) -> None:
    run_dir = tmp_path / "markdown_stale_cosing_level_case"
    write_minimal_run(run_dir)
    markdown_path = run_dir / "run_summary.md"
    markdown_path.write_text(
        markdown_path.read_text().replace(
            "- CosIng level: NEW\n",
            "- CosIng level: EXACT\n",
        )
    )

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert "missing CosIng level: NEW" in res.stdout


def test_verify_rejects_missing_markdown_inci(tmp_path: Path) -> None:
    run_dir = tmp_path / "markdown_missing_inci_case"
    write_minimal_run(run_dir)
    markdown_path = run_dir / "run_summary.md"
    markdown_path.write_text(markdown_path.read_text().replace("- INCI: n/a\n", ""))

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert "missing INCI: n/a" in res.stdout


def test_verify_rejects_stale_markdown_drug_warning_count(tmp_path: Path) -> None:
    run_dir = tmp_path / "markdown_stale_drug_warning_count_case"
    write_minimal_run(run_dir)
    markdown_path = run_dir / "run_summary.md"
    markdown_path.write_text(
        markdown_path.read_text().replace(
            "- Drug warnings: 0\n",
            "- Drug warnings: 2\n",
        )
    )

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert "missing drug warnings: 0" in res.stdout


def test_verify_rejects_stale_markdown_screening_stage_row(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "markdown_stale_screening_stage_row_case"
    write_minimal_run(run_dir)
    markdown_path = run_dir / "run_summary.md"
    markdown_path.write_text(
        markdown_path.read_text().replace(
            "| psichic_proteome_targets | 4 |\n",
            "| psichic_proteome_targets | 999 |\n",
        )
    )

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert (
        "missing screening stage row psichic_proteome_targets: "
        "psichic_proteome_targets | 4"
    ) in res.stdout


def test_verify_rejects_stale_markdown_skin_binding_efficacy_count(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "markdown_stale_skin_binding_efficacy_count_case"
    write_minimal_run(run_dir)
    markdown_path = run_dir / "run_summary.md"
    markdown_path.write_text(
        markdown_path.read_text().replace(
            "- Top targets with skin-efficacy evidence: 2\n",
            "- Top targets with skin-efficacy evidence: 0\n",
        )
    )

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert "missing skin binding efficacy count: 2" in res.stdout


def test_verify_rejects_summary_compound_inchikey_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "summary_compound_inchikey_mismatch_case"
    write_minimal_run(run_dir)
    payload = json.loads((run_dir / "run_summary.json").read_text())
    payload["compound"]["inchikey"] = "WRONG-INCHIKEY"
    (run_dir / "run_summary.json").write_text(json.dumps(payload))

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert "compound inchikey mismatch" in res.stdout


def test_verify_rejects_markdown_missing_compound_inchikey(tmp_path: Path) -> None:
    run_dir = tmp_path / "markdown_missing_compound_inchikey_case"
    write_minimal_run(run_dir)
    markdown_path = run_dir / "run_summary.md"
    markdown_path.write_text(
        markdown_path.read_text().replace(
            "- InChIKey: LFQSCWFLJHTTHZ-UHFFFAOYSA-N\n",
            "",
        )
    )

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert "missing inchikey: LFQSCWFLJHTTHZ-UHFFFAOYSA-N" in res.stdout


def test_verify_rejects_summary_skin_toxicity_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "summary_skin_toxicity_mismatch_case"
    write_minimal_run(run_dir)
    payload = json.loads((run_dir / "run_summary.json").read_text())
    payload["skin_toxicity"]["decision"] = "REVIEW"
    payload["skin_toxicity"]["toxicity_level"] = "moderate"
    payload["skin_toxicity"]["reasons"] = ["Skin_Reaction ADMET risk is moderate"]
    (run_dir / "run_summary.json").write_text(json.dumps(payload))

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert "skin_toxicity decision mismatch" in res.stdout
    assert "skin_toxicity toxicity_level mismatch" in res.stdout
    assert "skin_toxicity reasons mismatch" in res.stdout


def test_verify_rejects_summary_overall_top_target_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "summary_overall_top_mismatch_case"
    write_minimal_run(run_dir)
    payload = json.loads((run_dir / "run_summary.json").read_text())
    payload["overall_decision"]["top_target_id"] = "P2"
    (run_dir / "run_summary.json").write_text(json.dumps(payload))

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert "overall_decision.top_target_id mismatch" in res.stdout


def test_verify_rejects_skin_specialized_binding_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "summary_skin_binding_mismatch_case"
    write_minimal_run(run_dir)
    payload = json.loads((run_dir / "run_summary.json").read_text())
    payload["skin_specialized_binding"]["top_target_skin_tier"] = "very_low"
    payload["skin_specialized_binding"]["top_target_gene_symbol"] = "WRONG"
    payload["skin_specialized_binding"]["top_target_protein_name"] = "Wrong protein"
    payload["skin_specialized_binding"]["most_skin_relevant_target"]["gene_symbol"] = "WRONG"
    payload["skin_specialized_binding"]["most_skin_relevant_target"]["protein_name"] = "Wrong protein"
    (run_dir / "run_summary.json").write_text(json.dumps(payload))

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert "skin_specialized_binding top skin_tier mismatch" in res.stdout
    assert "skin_specialized_binding.top_target_gene_symbol mismatch" in res.stdout
    assert "skin_specialized_binding.top_target_protein_name mismatch" in res.stdout
    assert "most skin-relevant gene_symbol mismatch" in res.stdout
    assert "most skin-relevant protein_name mismatch" in res.stdout


def test_verify_rejects_skin_specialized_binding_evidence_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "summary_skin_binding_evidence_mismatch_case"
    write_minimal_run(run_dir)
    payload = json.loads((run_dir / "run_summary.json").read_text())
    payload["skin_specialized_binding"]["top_target_final_score"] = 0.5
    payload["skin_specialized_binding"]["top_target_sources"] = ["psichic"]
    payload["skin_specialized_binding"]["most_skin_relevant_target"]["docking_rrf"] = 0.5
    (run_dir / "run_summary.json").write_text(json.dumps(payload))
    with (run_dir / "run_summary.md").open("a") as handle:
        handle.write("\n0.5\npsichic\n")

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert "skin_specialized_binding top final_score mismatch" in res.stdout
    assert "skin_specialized_binding top sources mismatch" in res.stdout
    assert "most skin-relevant docking_rrf mismatch" in res.stdout


def test_verify_rejects_skin_context_decision_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "summary_skin_context_mismatch_case"
    write_minimal_run(run_dir)
    payload = json.loads((run_dir / "run_summary.json").read_text())
    payload["skin_specialized_binding"]["skin_context_decision"] = "skin_efficacy_literature_only"
    payload["skin_specialized_binding"]["skin_context_supported"] = False
    (run_dir / "run_summary.json").write_text(json.dumps(payload))

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert "skin_context_decision mismatch" in res.stdout
    assert "skin_context_supported mismatch" in res.stdout


def test_verify_rejects_overall_pass_without_supported_skin_context(tmp_path: Path) -> None:
    run_dir = tmp_path / "summary_unsupported_skin_context_pass_case"
    write_minimal_run(run_dir)
    write_unsupported_skin_context_ranking(run_dir)
    payload = json.loads((run_dir / "run_summary.json").read_text())
    for target in payload["target_prediction"]["top_targets"]:
        target["skin_score"] = 0.0
        target["skin_tier"] = "very_low"
    payload["skin_specialized_binding"].update({
        "skin_context_decision": "skin_efficacy_literature_only",
        "skin_context_supported": False,
        "skin_expression_supported": False,
        "skin_efficacy_supported": True,
        "skin_context_reasons": [
            "top targets have very-low skin-expression support",
            "top targets include KG skin-efficacy labels",
        ],
        "top_target_skin_score": 0.0,
        "top_target_skin_tier": "very_low",
    })
    payload["skin_specialized_binding"]["most_skin_relevant_target"]["skin_score"] = 0.0
    payload["skin_specialized_binding"]["most_skin_relevant_target"]["skin_tier"] = "very_low"
    payload["overall_decision"].update({
        "decision": "PASS",
        "requires_human_review": False,
        "recommended_action": "proceed",
        "claimable": True,
        "reasons": [
            "safety and cosmetic/drug gates passed",
            "top predicted target is P1 among 2 ranked targets",
        ],
    })
    (run_dir / "run_summary.json").write_text(json.dumps(payload))
    with (run_dir / "run_summary.md").open("a") as handle:
        handle.write(
            "\nvery_low\n"
            "skin_efficacy_literature_only\n"
            "top targets have very-low skin-expression support\n"
        )

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert "overall_decision cannot PASS without supported skin context" in res.stdout


def test_verify_rejects_summary_target_annotation_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "summary_target_annotation_mismatch_case"
    write_minimal_run(run_dir)
    metadata = tmp_path / "target_metadata.tsv"
    metadata.write_text("target_id\tgene_symbol\nP1\tGENE1\n")
    payload = json.loads((run_dir / "run_summary.json").read_text())
    payload["target_prediction"]["top_targets"][0]["gene_symbol"] = "WRONG"
    (run_dir / "run_summary.json").write_text(json.dumps(payload))

    res = run_verify([
        "--run-dir",
        str(run_dir),
        "--preset",
        "target-id",
        "--target-metadata",
        str(metadata),
    ])

    assert res.returncode == 2
    assert "top target gene_symbol mismatch" in res.stdout


def test_verify_rejects_summary_artifact_outside_run_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "summary_external_artifact_case"
    write_minimal_run(run_dir)
    (tmp_path / "outside.txt").write_text("not a run artifact\n")
    payload = json.loads((run_dir / "run_summary.json").read_text())
    payload["artifacts"]["external"] = "../outside.txt"
    (run_dir / "run_summary.json").write_text(json.dumps(payload))

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert "artifact external must be a relative path inside run directory" in res.stdout


def test_verify_rejects_safety_summary_with_target_claims(tmp_path: Path) -> None:
    run_dir = tmp_path / "safety_summary_target_claim_case"
    write_minimal_run(run_dir)
    summary = run_summarize(["--run-dir", str(run_dir), "--preset", "safety"])
    assert summary.returncode == 0, summary.stderr
    payload = json.loads((run_dir / "run_summary.json").read_text())
    payload["target_prediction"] = {"ranking_path": "03_targets/ranked_targets_v3_with_efficacy.csv"}
    payload["skin_specialized_binding"] = {
        "ranking_path": "03_targets/ranked_targets_v3_with_efficacy.csv"
    }
    payload["artifacts"]["target_ranking"] = "03_targets/ranked_targets_v3_with_efficacy.csv"
    (run_dir / "run_summary.json").write_text(json.dumps(payload))

    res = run_verify(["--run-dir", str(run_dir), "--preset", "safety"])

    assert res.returncode == 2
    assert "unexpected artifact keys ['target_ranking']" in res.stdout
    assert (
        "unexpected summary sections ['skin_specialized_binding', 'target_prediction']"
        in res.stdout
    )


def test_verify_rejects_missing_summary_safety_source_artifact(tmp_path: Path) -> None:
    run_dir = tmp_path / "summary_missing_safety_source_artifact_case"
    write_minimal_run(run_dir)
    payload = json.loads((run_dir / "run_summary.json").read_text())
    del payload["artifacts"]["husspred"]
    (run_dir / "run_summary.json").write_text(json.dumps(payload))

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert "missing artifact keys ['husspred']" in res.stdout


def test_verify_rejects_missing_summary_target_source_artifact(tmp_path: Path) -> None:
    run_dir = tmp_path / "summary_missing_target_source_artifact_case"
    write_minimal_run(run_dir)
    payload = json.loads((run_dir / "run_summary.json").read_text())
    del payload["artifacts"]["target_fast_psichic"]
    (run_dir / "run_summary.json").write_text(json.dumps(payload))

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id", "--mode", "fast"])

    assert res.returncode == 2
    assert "missing artifact keys ['target_fast_psichic']" in res.stdout


def test_verify_rejects_summary_cosmetic_drug_source_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "summary_cosmetic_source_mismatch_case"
    write_minimal_run(run_dir)
    drug_warnings = run_dir / "02b_cosmetic_drug" / "drug_warnings.json"
    source_payload = json.loads(drug_warnings.read_text())
    source_payload["n_warnings"] = 2
    source_payload["warnings"] = [{"kind": "approved_drug_similarity", "target": "aspirin"}]
    drug_warnings.write_text(json.dumps(source_payload))

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert "cosmetic_drug n_warnings mismatch" in res.stdout
    assert "overall_decision cannot PASS with source drug warnings" in res.stdout


def test_summarize_rejects_cosmetic_drug_decision_source_mismatch(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "summary_cosmetic_decision_mismatch_case"
    write_minimal_run(run_dir)
    (run_dir / "run_summary.json").unlink()
    drug_warnings = run_dir / "02b_cosmetic_drug" / "drug_warnings.json"
    source_payload = json.loads(drug_warnings.read_text())
    source_payload["warnings"] = [{"tier": "STRICT_WARNING"}]
    source_payload["n_warnings"] = 1
    drug_warnings.write_text(json.dumps(source_payload))

    res = run_summarize(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode != 0
    assert (
        "cosmetic/drug decision PROCEED does not match expected DOWNWEIGHT "
        "for policy moderate"
    ) in res.stderr
    assert not (run_dir / "run_summary.json").exists()


def test_verify_rejects_cosmetic_drug_decision_source_mismatch(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "verify_cosmetic_decision_mismatch_case"
    write_minimal_run(run_dir)
    drug_warnings = run_dir / "02b_cosmetic_drug" / "drug_warnings.json"
    source_payload = json.loads(drug_warnings.read_text())
    source_payload["warnings"] = [{"tier": "STRICT_WARNING"}]
    source_payload["n_warnings"] = 1
    drug_warnings.write_text(json.dumps(source_payload))

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert (
        "cosmetic/drug decision PROCEED does not match expected DOWNWEIGHT "
        "for policy moderate"
    ) in res.stdout


def test_verify_rejects_summary_artifact_wrong_canonical_path(tmp_path: Path) -> None:
    run_dir = tmp_path / "summary_wrong_artifact_path_case"
    write_minimal_run(run_dir)
    payload = json.loads((run_dir / "run_summary.json").read_text())
    payload["artifacts"]["drug_warnings"] = "02b_cosmetic_drug/cosing_match.json"
    (run_dir / "run_summary.json").write_text(json.dumps(payload))

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert (
        "artifact drug_warnings must point to 02b_cosmetic_drug/drug_warnings.json"
        in res.stdout
    )


def test_verify_target_id_contract_accepts_valid_fast_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "target_fast_case"
    write_minimal_run(run_dir)

    res = run_verify([
        "--run-dir",
        str(run_dir),
        "--preset",
        "target-id",
        "--mode",
        "fast",
        "--json",
    ])

    assert res.returncode == 0, res.stderr
    payload = json.loads(res.stdout)
    assert payload["status"] == "ok"
    assert any(check["name"] == "fast AutoDock-GPU provenance" for check in payload["checks"])


def test_verify_target_id_rejects_vina_fast_autodock_claim(tmp_path: Path) -> None:
    run_dir = tmp_path / "target_fast_vina_claim_case"
    write_minimal_run(run_dir)
    (run_dir / "03_targets" / "mode_fast" / "autodock_top5k.tsv").write_text(
        "target_id\tvina_score\tneg_vina_score\tengine\tmap_coverage_complete\t"
        "map_coverage_numerator\tmap_coverage_denominator\tdegraded\n"
        "P1\t-7.0\t7.0\tvina\tfalse\t0\t1\ttrue\n"
    )

    res = run_verify([
        "--run-dir",
        str(run_dir),
        "--preset",
        "target-id",
        "--mode",
        "fast",
    ])

    assert res.returncode == 2
    assert "fast AutoDock-GPU provenance" in res.stdout
    assert "expected engine autodock_gpu" in res.stdout


def test_verify_target_id_rejects_incomplete_autodock_map_coverage(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "target_incomplete_maps_case"
    write_minimal_run(run_dir)
    (run_dir / "03_targets" / "mode_comprehensive" / "autodock_all_targets.tsv").write_text(
        "target_id\tvina_score\tneg_vina_score\tengine\tmap_coverage_complete\t"
        "map_coverage_numerator\tmap_coverage_denominator\tdegraded\n"
        "P1\t-7.0\t7.0\tautodock_gpu\tfalse\t3\t4\tfalse\n"
    )

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert "comprehensive AutoDock-GPU provenance" in res.stdout
    assert "map coverage must be complete" in res.stdout


def test_verify_target_id_rejects_partial_autodock_row_coverage(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "target_partial_docking_case"
    write_minimal_run(run_dir)
    (run_dir / "03_targets" / "mode_comprehensive" / "autodock_all_targets.tsv").write_text(
        "target_id\tvina_score\tneg_vina_score\tengine\tmap_coverage_complete\t"
        "map_coverage_numerator\tmap_coverage_denominator\tdegraded\n"
        "P1\t-7.0\t7.0\tautodock_gpu\ttrue\t2\t2\tfalse\n"
    )

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert "comprehensive AutoDock-GPU provenance" in res.stdout
    assert "row count must equal the complete docking coverage denominator" in res.stdout


def test_verify_target_id_rejects_inconsistent_autodock_score_signs(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "target_bad_docking_sign_case"
    write_minimal_run(run_dir)
    path = run_dir / "03_targets" / "mode_comprehensive" / "autodock_all_targets.tsv"
    text = path.read_text().replace("P1\t-7.0\t7.0", "P1\t-7.0\t6.0", 1)
    path.write_text(text)

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert "vina_score and neg_vina_score must be opposites" in res.stdout


def test_verify_target_id_contract_accepts_valid_both_mode_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "target_both_case"
    write_minimal_run(run_dir)

    res = run_verify([
        "--run-dir",
        str(run_dir),
        "--preset",
        "target-id",
        "--mode",
        "both",
        "--json",
    ])

    assert res.returncode == 0, res.stderr
    payload = json.loads(res.stdout)
    check_names = {check["name"] for check in payload["checks"]}
    assert "fast rerank consensus" in check_names
    assert "comprehensive four-way consensus" in check_names


def test_verify_target_id_rejects_missing_fast_intermediate(tmp_path: Path) -> None:
    run_dir = tmp_path / "missing_fast_intermediate_case"
    write_minimal_run(run_dir)
    (run_dir / "03_targets" / "mode_fast" / "psichic_proteome.tsv").unlink()

    res = run_verify([
        "--run-dir",
        str(run_dir),
        "--preset",
        "target-id",
        "--mode",
        "fast",
    ])

    assert res.returncode == 2
    assert "[missing] fast PSICHIC proteome scores" in res.stdout


def test_verify_target_id_rejects_missing_comprehensive_intermediate(tmp_path: Path) -> None:
    run_dir = tmp_path / "missing_comprehensive_intermediate_case"
    write_minimal_run(run_dir)
    (run_dir / "03_targets" / "mode_comprehensive" / "ligand.pdbqt").unlink()

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert "[missing] comprehensive docking ligand PDBQT" in res.stdout


def test_verify_target_id_rejects_bad_intermediate_target_id(tmp_path: Path) -> None:
    run_dir = tmp_path / "bad_intermediate_target_case"
    write_minimal_run(run_dir)
    (run_dir / "03_targets" / "mode_fast" / "autodock_top5k.tsv").write_text(
        "target_id\tvina_score\tneg_vina_score\tengine\tmap_coverage_complete\t"
        "map_coverage_numerator\tmap_coverage_denominator\tdegraded\n"
        "\t-7.0\t7.0\tautodock_gpu\ttrue\t1\t1\tfalse\n"
    )

    res = run_verify([
        "--run-dir",
        str(run_dir),
        "--preset",
        "target-id",
        "--mode",
        "fast",
    ])

    assert res.returncode == 2
    assert "fast AutoDock-GPU provenance" in res.stdout
    assert "blank target_id value" in res.stdout


def test_verify_target_id_rejects_missing_efficacy_columns(tmp_path: Path) -> None:
    run_dir = tmp_path / "missing_efficacy_case"
    write_minimal_run(run_dir)
    (run_dir / "03_targets" / "ranked_targets_v3_with_efficacy.csv").write_text(
        "target_id,docking_rrf,skin_score,final_score,source_count,sources\n"
        "P1,0.9,0.8,0.85,3,autodock;gnina;rtmscore\n"
    )

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert "missing efficacy_top* columns" in res.stdout


def test_verify_target_id_rejects_halt_gate(tmp_path: Path) -> None:
    run_dir = tmp_path / "halt_target_case"
    write_minimal_run(run_dir, safety_decision="HALT")

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert "skin-sens decision HALT prevents target prediction" in res.stdout


def test_verify_target_id_rejects_source_count_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "source_mismatch_case"
    write_minimal_run(run_dir)
    (run_dir / "03_targets" / "ranked_targets_v3_with_efficacy.csv").write_text(
        "target_id,docking_rrf,skin_score,final_score,source_count,sources,efficacy_top1\n"
        "P1,0.9,0.8,0.85,3,autodock;gnina,hydration (3 papers)\n"
    )

    res = run_verify(["--run-dir", str(run_dir), "--preset", "target-id"])

    assert res.returncode == 2
    assert "source_count does not match sources labels" in res.stdout


def test_verify_report_rejects_placeholder_html(tmp_path: Path) -> None:
    run_dir = tmp_path / "placeholder_report_case"
    write_minimal_run(run_dir)
    (run_dir / "09_report" / "index.html").write_text(
        "<html><body>ADMET ranked targets skin-efficacy placeholder</body></html>\n"
    )

    res = run_verify(["--run-dir", str(run_dir), "--preset", "report"])

    assert res.returncode == 2
    assert "contains placeholder marker" in res.stdout


def test_verify_report_rejects_summary_value_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "report_summary_value_mismatch_case"
    write_minimal_run(run_dir)
    payload = json.loads((run_dir / "run_summary.json").read_text())
    payload["preset"] = "report"
    payload["artifacts"]["html_report"] = "09_report/index.html"
    (run_dir / "run_summary.json").write_text(json.dumps(payload))
    report = run_dir / "09_report" / "index.html"
    report.write_text(
        report.read_text().replace(
            "Drug-avoidance warnings: 0",
            "Drug-avoidance warnings: 9",
        )
    )

    res = run_verify(["--run-dir", str(run_dir), "--preset", "report"])

    assert res.returncode == 2
    assert (
        "missing report value drug warning count: Drug-avoidance warnings: 0"
        in res.stdout
    )


def test_verify_report_rejects_missing_drug_policy(tmp_path: Path) -> None:
    run_dir = tmp_path / "report_missing_drug_policy_case"
    write_minimal_run(run_dir)
    payload = json.loads((run_dir / "run_summary.json").read_text())
    payload["preset"] = "report"
    payload["artifacts"]["html_report"] = "09_report/index.html"
    (run_dir / "run_summary.json").write_text(json.dumps(payload))
    report = run_dir / "09_report" / "index.html"
    report.write_text(report.read_text().replace("Drug policy: moderate ", ""))

    res = run_verify(["--run-dir", str(run_dir), "--preset", "report"])

    assert res.returncode == 2
    assert "missing report value drug policy: Drug policy: moderate" in res.stdout


def test_verify_report_rejects_missing_cosing_annotation_details(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "report_missing_cosing_details_case"
    write_minimal_run(run_dir)
    payload = json.loads((run_dir / "run_summary.json").read_text())
    payload["preset"] = "report"
    payload["artifacts"]["html_report"] = "09_report/index.html"
    (run_dir / "run_summary.json").write_text(json.dumps(payload))
    report = run_dir / "09_report" / "index.html"
    report.write_text(
        report.read_text()
        .replace("CosIng level: NEW ", "")
        .replace("INCI = — ", "")
        .replace("functions = — ", "")
        .replace("Tanimoto = 0.000 ", "")
        .replace("max Tanimoto vs approved = 0.100", "max Tanimoto vs approved = 0.900")
    )

    res = run_verify(["--run-dir", str(run_dir), "--preset", "report"])

    assert res.returncode == 2
    assert "missing report value CosIng level: CosIng level: NEW" in res.stdout
    assert "missing report value INCI: INCI = —" in res.stdout
    assert "missing report value CosIng functions: functions = —" in res.stdout
    assert "missing report value CosIng tanimoto: Tanimoto = 0" in res.stdout
    assert (
        "missing report value drug max Tanimoto: "
        "max Tanimoto vs approved = 0.1"
    ) in res.stdout


def test_verify_report_rejects_missing_screening_count_summary(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "report_missing_screening_count_case"
    write_minimal_run(run_dir)
    mark_minimal_run_as_report(run_dir)
    report = run_dir / "09_report" / "index.html"
    report.write_text(
        report.read_text()
        .replace("Screened target candidates: 4 ", "")
        .replace("<tr><td>psichic_proteome_targets</td><td>4</td></tr>", "")
    )

    res = run_verify(["--run-dir", str(run_dir), "--preset", "report"])

    assert res.returncode == 2
    assert (
        "missing report value screened target candidates: "
        "Screened target candidates: 4"
    ) in res.stdout
    assert (
        "missing report value screening count psichic_proteome_targets"
        in res.stdout
    )


def test_verify_report_rejects_stale_skin_reaction_value(tmp_path: Path) -> None:
    run_dir = tmp_path / "report_stale_skin_reaction_case"
    write_minimal_run(run_dir)
    payload = json.loads((run_dir / "run_summary.json").read_text())
    payload["preset"] = "report"
    payload["artifacts"]["html_report"] = "09_report/index.html"
    (run_dir / "run_summary.json").write_text(json.dumps(payload))
    report = run_dir / "09_report" / "index.html"
    report.write_text(
        report.read_text().replace("Skin_Reaction: 0.210", "Skin_Reaction: 0.990")
    )

    res = run_verify(["--run-dir", str(run_dir), "--preset", "report"])

    assert res.returncode == 2
    assert (
        "missing report value Skin_Reaction value: Skin_Reaction: 0.21"
        in res.stdout
    )


def test_verify_report_rejects_missing_admet_metric_table(tmp_path: Path) -> None:
    run_dir = tmp_path / "report_missing_admet_metric_table_case"
    write_minimal_run(run_dir)
    payload = json.loads((run_dir / "run_summary.json").read_text())
    payload["preset"] = "report"
    payload["artifacts"]["html_report"] = "09_report/index.html"
    (run_dir / "run_summary.json").write_text(json.dumps(payload))
    report = run_dir / "09_report" / "index.html"
    report.write_text(
        report.read_text().replace("<tr><td>QED</td><td>0.4068</td></tr>", "")
    )

    res = run_verify(["--run-dir", str(run_dir), "--preset", "report"])

    assert res.returncode == 2
    assert "missing report value ADMET metric QED: QED: 0.4068" in res.stdout


def test_verify_report_rejects_stale_admet_metric_value(tmp_path: Path) -> None:
    run_dir = tmp_path / "report_stale_admet_metric_case"
    write_minimal_run(run_dir)
    payload = json.loads((run_dir / "run_summary.json").read_text())
    payload["preset"] = "report"
    payload["artifacts"]["html_report"] = "09_report/index.html"
    (run_dir / "run_summary.json").write_text(json.dumps(payload))
    report = run_dir / "09_report" / "index.html"
    report.write_text(
        report.read_text().replace(
            "<tr><td>LD50_Zhu</td><td>1.1</td></tr>",
            "<tr><td>LD50_Zhu</td><td>9.9</td></tr>",
        )
    )

    res = run_verify(["--run-dir", str(run_dir), "--preset", "report"])

    assert res.returncode == 2
    assert "missing report value ADMET metric LD50_Zhu: LD50_Zhu: 1.1" in res.stdout


def test_verify_report_rejects_claimable_value_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "report_claimable_value_mismatch_case"
    write_minimal_run(run_dir, safety_decision="FLAG_HIGH")
    payload = json.loads((run_dir / "run_summary.json").read_text())
    payload["preset"] = "report"
    payload["artifacts"]["html_report"] = "09_report/index.html"
    (run_dir / "run_summary.json").write_text(json.dumps(payload))
    report = run_dir / "09_report" / "index.html"
    report.write_text(report.read_text().replace("Claimable: no", "Claimable: yes"))

    res = run_verify(["--run-dir", str(run_dir), "--preset", "report"])

    assert res.returncode == 2
    assert "missing report value claimable: Claimable: no" in res.stdout


def test_verify_report_rejects_claim_status_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "report_claim_status_mismatch_case"
    write_minimal_run(run_dir)
    payload = json.loads((run_dir / "run_summary.json").read_text())
    payload["preset"] = "report"
    payload["artifacts"]["html_report"] = "09_report/index.html"
    (run_dir / "run_summary.json").write_text(json.dumps(payload))
    report = run_dir / "09_report" / "index.html"
    report.write_text(
        report.read_text().replace(
            "Claim status: proceed",
            "Claim status: human review required before claim",
        )
    )

    res = run_verify(["--run-dir", str(run_dir), "--preset", "report"])

    assert res.returncode == 2
    assert "missing report value claim status: Claim status: proceed" in res.stdout


def test_verify_report_rejects_missing_compound_identity(tmp_path: Path) -> None:
    run_dir = tmp_path / "report_missing_compound_identity_case"
    write_minimal_run(run_dir)
    payload = json.loads((run_dir / "run_summary.json").read_text())
    payload["preset"] = "report"
    payload["artifacts"]["html_report"] = "09_report/index.html"
    (run_dir / "run_summary.json").write_text(json.dumps(payload))
    report = run_dir / "09_report" / "index.html"
    report.write_text(
        report.read_text()
        .replace("Input type: smiles ", "")
        .replace("Input SMILES: CCO ", "")
        .replace("Input canonical SMILES: CCO ", "")
        .replace("InChIKey: LFQSCWFLJHTTHZ-UHFFFAOYSA-N ", "")
        .replace("canonical SMILES: CCO ", "")
    )

    res = run_verify(["--run-dir", str(run_dir), "--preset", "report"])

    assert res.returncode == 2
    assert "missing report value input type: Input type: smiles" in res.stdout
    assert "missing report value input SMILES: Input SMILES: CCO" in res.stdout
    assert (
        "missing report value input canonical SMILES: "
        "Input canonical SMILES: CCO"
    ) in res.stdout
    assert "missing report value canonical SMILES: canonical SMILES: CCO" in res.stdout
    assert "missing report value InChIKey: InChIKey: LFQSCWFLJHTTHZ-UHFFFAOYSA-N" in res.stdout


def test_verify_report_rejects_missing_input_sdf_provenance(tmp_path: Path) -> None:
    run_dir = tmp_path / "report_missing_input_sdf_case"
    write_minimal_run(run_dir)
    input_sdf = str(tmp_path / "source_ligand.sdf")
    rewrite_minimal_run_as_sdf_input(run_dir, input_sdf)
    mark_minimal_run_as_report(run_dir)
    report = run_dir / "09_report" / "index.html"
    report.write_text(report.read_text().replace(f"Input SDF: {input_sdf} ", ""))

    res = run_verify(["--run-dir", str(run_dir), "--preset", "report"])

    assert res.returncode == 2
    assert f"missing report value input SDF: Input SDF: {input_sdf}" in res.stdout


def test_verify_report_rejects_missing_skin_binding_evidence(tmp_path: Path) -> None:
    run_dir = tmp_path / "report_missing_skin_binding_evidence_case"
    write_minimal_run(run_dir)
    payload = json.loads((run_dir / "run_summary.json").read_text())
    payload["preset"] = "report"
    payload["artifacts"]["html_report"] = "09_report/index.html"
    (run_dir / "run_summary.json").write_text(json.dumps(payload))
    report = run_dir / "09_report" / "index.html"
    report.write_text(
        report.read_text()
        .replace("top target final score: 0.85 ", "")
        .replace("top target docking RRF: 0.9 ", "")
        .replace("top target source count: 3 ", "")
        .replace("top target sources: autodock, gnina, rtmscore ", "")
        .replace("top target skin score: 0.8 ", "")
        .replace("top target skin tier: high ", "")
        .replace("top target skin efficacy: hydration (3 papers) ", "")
        .replace("Most skin-relevant top target: P1 ", "")
        .replace("most skin-relevant final score: 0.85 ", "")
        .replace("most skin-relevant docking RRF: 0.9 ", "")
        .replace("most skin-relevant source count: 3 ", "")
        .replace("most skin-relevant sources: autodock, gnina, rtmscore ", "")
        .replace("most skin-relevant skin score: 0.8 ", "")
        .replace("most skin-relevant skin tier: high ", "")
        .replace("most skin-relevant efficacy: hydration (3 papers) ", "")
    )

    res = run_verify(["--run-dir", str(run_dir), "--preset", "report"])

    assert res.returncode == 2
    assert "missing report value top target final score" in res.stdout
    assert "missing report value top target docking RRF" in res.stdout
    assert "missing report value top target source count" in res.stdout
    assert "missing report value top target sources" in res.stdout
    assert "missing report value top target skin score" in res.stdout
    assert "missing report value top target skin tier" in res.stdout
    assert "missing report value top target skin efficacy" in res.stdout
    assert "missing report value most skin-relevant target" in res.stdout
    assert "missing report value most skin-relevant final score" in res.stdout
    assert "missing report value most skin-relevant docking RRF" in res.stdout
    assert "missing report value most skin-relevant source count" in res.stdout
    assert "missing report value most skin-relevant sources" in res.stdout
    assert "missing report value most skin-relevant skin score" in res.stdout
    assert "missing report value most skin-relevant skin tier" in res.stdout
    assert "missing report value most skin-relevant efficacy" in res.stdout


def test_verify_report_rejects_missing_summary_top_target(tmp_path: Path) -> None:
    run_dir = tmp_path / "report_missing_top_target_case"
    write_minimal_run(run_dir)
    payload = json.loads((run_dir / "run_summary.json").read_text())
    payload["preset"] = "report"
    payload["artifacts"]["html_report"] = "09_report/index.html"
    (run_dir / "run_summary.json").write_text(json.dumps(payload))
    report = run_dir / "09_report" / "index.html"
    report.write_text(report.read_text().replace(">P2<", ">P9<"))

    res = run_verify(["--run-dir", str(run_dir), "--preset", "report"])

    assert res.returncode == 2
    assert "missing report value top target 2 target_id: P2" in res.stdout


def test_verify_report_rejects_stale_target_row_details(tmp_path: Path) -> None:
    run_dir = tmp_path / "report_stale_target_row_case"
    write_minimal_run(run_dir)
    mark_minimal_run_as_report(run_dir)
    report = run_dir / "09_report" / "index.html"
    report.write_text(
        report.read_text().replace(
            "<tr><td>P1</td><td>GENE1</td><td>Protein one</td><td>0.85</td><td>0.8</td><td>high</td>"
            "<td>0.9</td><td>autodock;gnina;rtmscore</td>"
            "<td>hydration (3 papers)</td></tr>",
            "<tr><td>P1</td><td>WRONG</td><td>Wrong protein</td><td>0.99</td><td>0.1</td><td>high</td>"
            "<td>0.2</td><td>psichic</td><td>barrier (9 papers)</td></tr>",
        )
    )

    res = run_verify(["--run-dir", str(run_dir), "--preset", "report"])

    assert res.returncode == 2
    assert "missing report value top target 1 row details" in res.stdout
    assert "0.85" in res.stdout
    assert "autodock | gnina | rtmscore" in res.stdout
    assert "hydration (3 papers)" in res.stdout


def test_verify_report_rejects_missing_target_gene_protein_labels(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "report_missing_target_annotation_case"
    write_minimal_run(run_dir)
    mark_minimal_run_as_report(run_dir)
    report = run_dir / "09_report" / "index.html"
    report.write_text(
        report.read_text()
        .replace("top target gene: GENE1 ", "")
        .replace("top target protein: Protein one ", "")
        .replace(
            "<tr><td>P1</td><td>GENE1</td><td>Protein one</td>",
            "<tr><td>P1</td>",
        )
    )

    res = run_verify(["--run-dir", str(run_dir), "--preset", "report"])

    assert res.returncode == 2
    assert "missing report value top target 1 row details" in res.stdout
    assert "GENE1" in res.stdout
    assert "Protein one" in res.stdout


def test_verify_report_rejects_missing_skin_sens_evidence_rows(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "report_missing_skin_sens_evidence_case"
    write_minimal_run(run_dir)
    payload = json.loads((run_dir / "run_summary.json").read_text())
    payload["preset"] = "report"
    payload["artifacts"]["html_report"] = "09_report/index.html"
    (run_dir / "run_summary.json").write_text(json.dumps(payload))
    report = run_dir / "09_report" / "index.html"
    report.write_text(
        report.read_text().replace(
            "<tr><td>husspred</td><td>ok</td><td>negative</td><td>0.2</td></tr>",
            "",
        )
    )

    res = run_verify(["--run-dir", str(run_dir), "--preset", "report"])

    assert res.returncode == 2
    assert "missing report value skin-sens evidence row husspred" in res.stdout
    assert "negative" in res.stdout
    assert "0.2" in res.stdout


def test_verify_report_accepts_html_float_precision_variants(tmp_path: Path) -> None:
    run_dir = tmp_path / "report_float_precision_case"
    write_minimal_run(run_dir)
    ranking = run_dir / "03_targets" / "ranked_targets_v3_with_efficacy.csv"
    ranking.write_text(
        "target_id,docking_rrf,skin_score,skin_tier,final_score,source_count,sources,efficacy_top1\n"
        "P1,0.666666667,0.333333333,high,0.987654321,3,autodock;gnina;rtmscore,hydration (3 papers)\n"
        "P2,0.7,0.6,medium,0.70,3,autodock;gnina;rtmscore,barrier (2 papers)\n"
    )
    summary = run_summarize([
        "--run-dir",
        str(run_dir),
        "--preset",
        "report",
        "--mode",
        "both",
    ])
    assert summary.returncode == 0, summary.stderr
    report = run_dir / "09_report" / "index.html"
    report.write_text(
        (
            report.read_text()
            .replace("top target final score: 0.85", "top target final score: 0.987654")
            .replace("top target docking RRF: 0.9", "top target docking RRF: 0.666667")
            .replace("top target skin score: 0.8", "top target skin score: 0.333333")
            .replace("Most skin-relevant top target: P1", "Most skin-relevant top target: P2")
            .replace("most skin-relevant final score: 0.85", "most skin-relevant final score: 0.7")
            .replace("most skin-relevant docking RRF: 0.9", "most skin-relevant docking RRF: 0.7")
            .replace("most skin-relevant skin score: 0.8", "most skin-relevant skin score: 0.6")
            .replace("most skin-relevant skin tier: high", "most skin-relevant skin tier: medium")
            .replace(
                "most skin-relevant efficacy: hydration (3 papers)",
                "most skin-relevant efficacy: barrier (2 papers)",
            )
            .replace(
                "<tr><td>P1</td><td>GENE1</td><td>Protein one</td><td>0.85</td><td>0.8</td><td>high</td>"
                "<td>0.9</td><td>autodock;gnina;rtmscore</td>"
                "<td>hydration (3 papers)</td></tr>",
                "<tr><td>P1</td><td>0.987654</td><td>0.333333</td><td>high</td>"
                "<td>0.666667</td><td>autodock;gnina;rtmscore</td>"
                "<td>hydration (3 papers)</td></tr>",
            )
        )
    )

    res = run_verify(["--run-dir", str(run_dir), "--preset", "report"])

    assert res.returncode == 0, res.stdout + res.stderr


def test_verify_rejects_missing_core_admet_endpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "missing_admet_endpoint_case"
    write_minimal_run(run_dir)
    payload = json.loads((run_dir / "02_admet" / "admet_report.json").read_text())
    del payload["admet_ai_predictions"]["Skin_Reaction"]
    (run_dir / "02_admet" / "admet_report.json").write_text(json.dumps(payload))

    res = run_verify(["--run-dir", str(run_dir), "--preset", "safety"])

    assert res.returncode == 2
    assert "missing ADMET-AI endpoints" in res.stdout
    assert "Skin_Reaction" in res.stdout


def test_verify_rejects_missing_skin_sens_source_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "missing_skin_source_case"
    write_minimal_run(run_dir)
    (run_dir / "02_admet" / "pred_skin.json").unlink()

    res = run_verify(["--run-dir", str(run_dir), "--preset", "safety"])

    assert res.returncode == 2
    assert "[missing] pred_skin skin-sens source evidence" in res.stdout


def test_verify_rejects_nonnumeric_skin_sens_probability(tmp_path: Path) -> None:
    run_dir = tmp_path / "bad_skin_probability_case"
    write_minimal_run(run_dir)
    payload = json.loads((run_dir / "02_admet" / "stoptox.json").read_text())
    payload["skin_sens_probability"] = "true"
    (run_dir / "02_admet" / "stoptox.json").write_text(json.dumps(payload))

    res = run_verify(["--run-dir", str(run_dir), "--preset", "safety"])

    assert res.returncode == 2
    assert "stoptox skin-sens probability must be numeric" in res.stdout


def test_verify_rejects_admet_source_report_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "admet_source_mismatch_case"
    write_minimal_run(run_dir)
    payload = json.loads((run_dir / "02_admet" / "admet_ai.json").read_text())
    payload["predictions"]["DILI"] = 0.77
    (run_dir / "02_admet" / "admet_ai.json").write_text(json.dumps(payload))

    res = run_verify(["--run-dir", str(run_dir), "--preset", "safety"])

    assert res.returncode == 2
    assert "source/report mismatch for endpoint DILI" in res.stdout


def test_verify_rejects_skin_sens_source_smiles_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "skin_sens_source_smiles_mismatch_case"
    write_minimal_run(run_dir)
    payload = json.loads((run_dir / "02_admet" / "husspred.json").read_text())
    payload["smiles"] = "CCC"
    (run_dir / "02_admet" / "husspred.json").write_text(json.dumps(payload))

    res = run_verify(["--run-dir", str(run_dir), "--preset", "safety"])

    assert res.returncode == 2
    assert "husspred skin-sens source evidence" in res.stdout
    assert "smiles does not match compound canonical_smiles" in res.stdout


def test_verify_rejects_structural_alert_source_flag_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "structural_alert_source_flag_mismatch_case"
    write_minimal_run(run_dir)
    payload = json.loads((run_dir / "02_admet" / "structural_alerts.json").read_text())
    payload["any_brenk"] = True
    (run_dir / "02_admet" / "structural_alerts.json").write_text(json.dumps(payload))

    res = run_verify(["--run-dir", str(run_dir), "--preset", "safety"])

    assert res.returncode == 2
    assert "source/report mismatch for structural-alert flag any_brenk" in res.stdout


def test_verify_rejects_nonbool_structural_alert_report_flags(tmp_path: Path) -> None:
    run_dir = tmp_path / "structural_alert_report_nonbool_case"
    write_minimal_run(run_dir)
    payload = json.loads((run_dir / "02_admet" / "admet_report.json").read_text())
    payload["structural_alerts"]["any_nih"] = None
    (run_dir / "02_admet" / "admet_report.json").write_text(json.dumps(payload))

    res = run_verify(["--run-dir", str(run_dir), "--preset", "safety"])

    assert res.returncode == 2
    assert "structural_alerts.any_nih must be boolean" in res.stdout
    assert "cannot validate safety.structural_alert_flags" in res.stdout


def test_verify_rejects_degraded_admet_by_default(tmp_path: Path) -> None:
    run_dir = tmp_path / "degraded_case"
    write_minimal_run(run_dir)
    payload = json.loads((run_dir / "02_admet" / "admet_report.json").read_text())
    payload["skin_sens"]["degraded"] = True
    payload["admet_ai_predictions"] = None
    (run_dir / "02_admet" / "admet_report.json").write_text(json.dumps(payload))

    res = run_verify(["--run-dir", str(run_dir), "--preset", "safety"])

    assert res.returncode == 2
    assert "degraded skin-sens evidence" in res.stdout
    assert "missing ADMET-AI predictions" in res.stdout


def test_verify_rejects_stale_summary_admet_metrics_when_degraded(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "degraded_stale_summary_metrics_case"
    write_minimal_run(run_dir)
    payload = json.loads((run_dir / "02_admet" / "admet_report.json").read_text())
    payload["skin_sens"]["degraded"] = True
    payload["admet_ai_predictions"] = None
    (run_dir / "02_admet" / "admet_report.json").write_text(json.dumps(payload))

    res = run_verify([
        "--run-dir",
        str(run_dir),
        "--preset",
        "safety",
        "--allow-degraded",
    ])

    assert res.returncode == 2
    assert "unexpected ADMET metrics" in res.stdout
    assert "unexpected ADMET risk endpoints" in res.stdout


def test_verify_allows_degraded_admet_when_explicit(tmp_path: Path) -> None:
    run_dir = tmp_path / "degraded_allowed_case"
    write_minimal_run(run_dir)
    payload = json.loads((run_dir / "02_admet" / "admet_report.json").read_text())
    payload["skin_sens"]["degraded"] = True
    payload["admet_ai_predictions"] = None
    (run_dir / "02_admet" / "admet_report.json").write_text(json.dumps(payload))
    summary = run_summarize([
        "--run-dir",
        str(run_dir),
        "--preset",
        "safety",
        "--allow-degraded",
    ])
    assert summary.returncode == 0, summary.stderr

    res = run_verify([
        "--run-dir",
        str(run_dir),
        "--preset",
        "safety",
        "--allow-degraded",
    ])

    assert res.returncode == 0, res.stdout


def test_verify_rejects_invalid_canonical_smiles(tmp_path: Path) -> None:
    run_dir = tmp_path / "bad_smiles_case"
    write_minimal_run(run_dir)
    payload = json.loads((run_dir / "01_input" / "compound_canonical.json").read_text())
    payload["canonical_smiles"] = "not a smiles"
    (run_dir / "01_input" / "compound_canonical.json").write_text(json.dumps(payload))

    res = run_verify(["--run-dir", str(run_dir), "--preset", "safety"])

    assert res.returncode == 2
    assert "invalid canonical_smiles" in res.stdout


def test_verify_rejects_missing_input_smiles_provenance(tmp_path: Path) -> None:
    run_dir = tmp_path / "missing_input_smiles_provenance_case"
    write_minimal_run(run_dir)
    payload = json.loads((run_dir / "01_input" / "compound_canonical.json").read_text())
    del payload["input_type"]
    del payload["input_smiles"]
    del payload["input_canonical_smiles"]
    (run_dir / "01_input" / "compound_canonical.json").write_text(json.dumps(payload))

    res = run_verify(["--run-dir", str(run_dir), "--preset", "safety"])

    assert res.returncode == 2
    assert "missing non-empty input_type" in res.stdout


def test_verify_rejects_smiles_metadata_with_sdf_provenance(tmp_path: Path) -> None:
    run_dir = tmp_path / "smiles_metadata_with_sdf_provenance_case"
    write_minimal_run(run_dir)
    payload = json.loads((run_dir / "01_input" / "compound_canonical.json").read_text())
    payload["input_sdf"] = str(tmp_path / "stale.sdf")
    (run_dir / "01_input" / "compound_canonical.json").write_text(json.dumps(payload))

    res = run_verify(["--run-dir", str(run_dir), "--preset", "safety"])

    assert res.returncode == 2
    assert "input_type smiles cannot include input_sdf" in res.stdout


def test_verify_rejects_sdf_metadata_with_smiles_provenance(tmp_path: Path) -> None:
    run_dir = tmp_path / "sdf_metadata_with_smiles_provenance_case"
    write_minimal_run(run_dir)
    rewrite_minimal_run_as_sdf_input(run_dir, str(tmp_path / "source_ligand.sdf"))
    payload = json.loads((run_dir / "01_input" / "compound_canonical.json").read_text())
    payload["input_smiles"] = "CCO"
    payload["input_canonical_smiles"] = "CCO"
    (run_dir / "01_input" / "compound_canonical.json").write_text(json.dumps(payload))

    res = run_verify(["--run-dir", str(run_dir), "--preset", "safety"])

    assert res.returncode == 2
    assert "input_type sdf cannot include input_smiles" in res.stdout
    assert "input_type sdf cannot include input_canonical_smiles" in res.stdout


def test_verify_rejects_summary_with_unexpected_input_provenance(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "summary_unexpected_input_provenance_case"
    write_minimal_run(run_dir)
    payload = json.loads((run_dir / "run_summary.json").read_text())
    payload["compound"]["input_sdf"] = str(tmp_path / "stale.sdf")
    (run_dir / "run_summary.json").write_text(json.dumps(payload))

    res = run_verify(["--run-dir", str(run_dir), "--preset", "safety"])

    assert res.returncode == 2
    assert "unexpected compound input_sdf" in res.stdout


def test_verify_rejects_noncanonical_canonical_smiles(tmp_path: Path) -> None:
    run_dir = tmp_path / "noncanonical_smiles_case"
    write_minimal_run(run_dir)
    payload = json.loads((run_dir / "01_input" / "compound_canonical.json").read_text())
    payload["canonical_smiles"] = "C(C)O"
    (run_dir / "01_input" / "compound_canonical.json").write_text(json.dumps(payload))

    res = run_verify(["--run-dir", str(run_dir), "--preset", "safety"])

    assert res.returncode == 2
    assert "canonical_smiles is not canonical" in res.stdout


def test_verify_rejects_compound_inchikey_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "compound_inchikey_mismatch_case"
    write_minimal_run(run_dir)
    payload = json.loads((run_dir / "01_input" / "compound_canonical.json").read_text())
    payload["inchikey"] = "WRONG-INCHIKEY"
    (run_dir / "01_input" / "compound_canonical.json").write_text(json.dumps(payload))

    res = run_verify(["--run-dir", str(run_dir), "--preset", "safety"])

    assert res.returncode == 2
    assert "inchikey does not match canonical_smiles" in res.stdout


def test_verify_rejects_blank_decision_file_without_crashing(tmp_path: Path) -> None:
    run_dir = tmp_path / "blank_decision_case"
    write_minimal_run(run_dir)
    (run_dir / "02b_cosmetic_drug" / "cosmetic_drug_decision.txt").write_text("\n")

    res = run_verify(["--run-dir", str(run_dir), "--preset", "safety"])

    assert res.returncode == 2
    assert "contains no decision line" in res.stdout


def test_verify_rejects_reference_payload_without_status(tmp_path: Path) -> None:
    run_dir = tmp_path / "missing_reference_status_case"
    write_minimal_run(run_dir)
    payload = json.loads((run_dir / "02b_cosmetic_drug" / "cosing_match.json").read_text())
    del payload["reference_status"]
    (run_dir / "02b_cosmetic_drug" / "cosing_match.json").write_text(json.dumps(payload))

    res = run_verify(["--run-dir", str(run_dir), "--preset", "safety"])

    assert res.returncode == 2
    assert "missing non-empty reference_status" in res.stdout


def test_verify_missing_run_dir_reports_clean_failure(tmp_path: Path) -> None:
    missing = tmp_path / "missing_run"

    res = run_verify(["--run-dir", str(missing), "--preset", "target-id"])

    assert res.returncode == 2
    assert "SkinScout run output verification: failed" in res.stdout
    assert "[missing] run directory" in res.stdout
    assert "Traceback" not in res.stderr


def test_the_applicability_record_survives_into_the_published_summary(
    tmp_path: Path,
) -> None:
    """The summary rebuilds `compound` field by field, dropping anything unnamed.

    The applicability verdict was computed and then lost there, so the report
    section that reads it could never render. Ethanol is smaller than anything
    in the validation panel, so it is a warning case rather than a refusal.
    """
    run_dir = tmp_path / "summary_applicability_case"
    write_minimal_run(run_dir)
    (run_dir / "run_summary.json").unlink()
    (run_dir / "run_summary.md").unlink()

    res = run_summarize(["--run-dir", str(run_dir), "--preset", "safety"])

    assert res.returncode == 0, res.stderr
    payload = json.loads((run_dir / "run_summary.json").read_text())
    applicability = payload["compound"]["applicability"]
    assert applicability["verdict"] == "review"
    codes = {item["code"] for item in applicability["warnings"]}
    assert "heavy_atoms_outside_panel" in codes
    assert "molecular_weight_outside_panel" in codes


def test_an_out_of_range_input_is_flagged_in_the_report_a_reader_sees(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "summary_applicability_markdown_case"
    write_minimal_run(run_dir)
    (run_dir / "run_summary.json").unlink()
    (run_dir / "run_summary.md").unlink()

    res = run_summarize(["--run-dir", str(run_dir), "--preset", "safety"])

    assert res.returncode == 0, res.stderr
    markdown = (run_dir / "run_summary.md").read_text()
    assert "### Input applicability" in markdown
    assert "검증 패널 실측 범위" in markdown
