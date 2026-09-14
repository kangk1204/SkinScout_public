"""Regression tests for the SkinScout goal contract verifier."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VERIFY_GOAL = ROOT / "scripts" / "verify_goal_contract.py"


def run_goal(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VERIFY_GOAL), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def artifact(path: Path, name: str) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "name": name,
        "path": str(path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def write_goal_run(
    run_dir: Path,
    *,
    verification_status: str = "ok",
    diagnostic_reasons: list[str] | None = None,
) -> None:
    diagnostic_reasons = diagnostic_reasons or []
    run_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = run_dir / "03_targets/ranked_targets_v3_with_efficacy.csv"
    ranking_path.parent.mkdir()
    ranking_path.write_text("final_rank,target_id,final_score\n1,P10275,0.21\n")
    summary = {
        "schema_version": "skinscout.run_summary.v1",
        "run_id": run_dir.name,
        "preset": "target-id",
        "mode": "fast",
        "artifacts": {
            "target_ranking": "03_targets/ranked_targets_v3_with_efficacy.csv",
        },
        "compound": {
            "input_type": "smiles",
            "input_smiles": "CCO",
            "input_canonical_smiles": "CCO",
            "canonical_smiles": "CCO",
            "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        },
        "overall_decision": {
            "decision": "FLAG_HIGH",
            "recommended_action": "review_before_claim",
            "claimable": False,
            "requires_human_review": True,
            "reasons": ["top binding target lacks direct skin context support: P10275"],
        },
        "safety": {
            "skin_sens_decision": "PASS",
            "skin_sens_evidence": [
                {"model": "husspred", "status": "ok", "call": "negative", "probability": 0.13},
                {"model": "stoptox", "status": "ok", "call": "negative", "probability": 0.90},
                {"model": "pred_skin", "status": "ok", "call": "negative", "probability": 0.04},
            ],
            "admet_metrics": {
                "AMES": 0.04,
                "ClinTox": 0.0004,
                "DILI": 0.09,
                "Skin_Reaction": 0.21,
                "hERG": 0.01,
                "LD50_Zhu": 1.1,
                "Solubility_AqSolDB": 1.2,
                "logP": -0.001,
                "QED": 0.40,
                "tpsa": 20.23,
            },
            "admet_risk_assessment": {
                "high_risk_endpoints": [],
                "moderate_risk_endpoints": [],
            },
            "degraded": False,
            "missing_models": [],
        },
        "skin_toxicity": {
            "decision": "PASS",
            "toxicity_level": "low",
            "skin_sens_decision": "PASS",
            "skin_reaction_risk_level": "low",
            "skin_reaction_value": 0.21,
            "structural_alerts_present": False,
            "structural_alert_flags": [],
            "degraded": False,
            "missing_models": [],
            "reasons": ["skin sensitization and Skin_Reaction ADMET risk are low"],
        },
        "target_prediction": {
            "ranking_path": "03_targets/ranked_targets_v3_with_efficacy.csv",
            "n_targets": 2,
            "screened_target_count": 20171,
            "screening_counts": {
                "psichic_proteome_targets": 20171,
                "daina_zoete_targets": 2943,
                "dti_rrf_candidates": 732,
                "autodock_rescored_targets": 256,
                "rerank_consensus_targets": 10,
                "skin_weighted_ranked_targets": 2,
            },
            "top_n": 2,
            "top_targets": [
                {
                    "target_id": "P10275",
                    "gene_symbol": "AR",
                    "protein_name": "Androgen receptor",
                    "final_score": 0.21,
                    "docking_rrf": 0.031,
                    "source_count": 2,
                    "sources": ["autodock", "gnina"],
                    "skin_score": 0.0,
                    "skin_tier": "very_low",
                    "efficacy": ["hair_growth (250 papers)"],
                }
            ],
        },
        "skin_specialized_binding": {
            "context": "skin-specialized material-protein binding",
            "skin_context_decision": "skin_efficacy_literature_only",
            "skin_context_supported": False,
            "skin_expression_supported": False,
            "skin_efficacy_supported": True,
            "top_target_id": "P10275",
            "top_target_gene_symbol": "AR",
            "top_target_protein_name": "Androgen receptor",
            "top_target_final_score": 0.21,
            "top_target_docking_rrf": 0.031,
            "top_target_source_count": 2,
            "top_target_sources": ["autodock", "gnina"],
            "top_target_skin_score": 0.0,
            "top_target_skin_tier": "very_low",
            "top_target_skin_expression_supported": False,
            "top_target_skin_efficacy_supported": True,
            "top_target_skin_context_supported": False,
            "top_targets_with_skin_efficacy": 1,
            "top_target_efficacy": ["hair_growth (250 papers)"],
            "most_skin_relevant_target": {
                "target_id": "P10275",
                "gene_symbol": "AR",
                "protein_name": "Androgen receptor",
                "final_score": 0.21,
                "docking_rrf": 0.031,
                "source_count": 2,
                "sources": ["autodock", "gnina"],
                "skin_score": 0.0,
                "skin_tier": "very_low",
                "efficacy": ["hair_growth (250 papers)"],
            },
        },
    }
    summary_path = run_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary))
    summary_md = run_dir / "run_summary.md"
    summary_md.write_text("# SkinScout Run Summary\n")
    verification_log = run_dir / "run_verification.log"
    verification_log.write_text("SkinScout run output verification: ok\n")
    verification = {
        "schema_version": "skinscout.run_verification.v1",
        "status": verification_status,
        "returncode": 0,
        "verifier_status": "ok",
        "run_dir": str(run_dir),
        "preset": "target-id",
        "mode": "fast",
        "input_provenance": {
            "input_type": "smiles",
            "input_smiles": "CCO",
            "input_canonical_smiles": "CCO",
            "input_sdf": None,
        },
        "diagnostic_nonclaimable_reasons": diagnostic_reasons,
        "checks": [
            {
                "name": "user-facing run summary",
                "status": "ok" if verification_status == "ok" else "failed",
                "path": str(summary_path),
                "detail": "",
            }
        ],
        "verified_artifacts": [
            artifact(summary_path, "run_summary_json"),
            artifact(summary_md, "run_summary_md"),
            artifact(verification_log, "run_verification_log"),
            artifact(ranking_path, "target_ranking"),
        ],
    }
    (run_dir / "run_verification.json").write_text(json.dumps(verification))


def test_goal_contract_accepts_verified_target_id_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "goal_ok"
    write_goal_run(run_dir)

    res = run_goal(["--run-dir", str(run_dir), "--smiles", "CCO", "--json"])

    assert res.returncode == 0, res.stderr
    payload = json.loads(res.stdout)
    assert payload["schema_version"] == "skinscout.goal_contract.v1"
    assert payload["status"] == "ok"
    checks = {check["name"]: check for check in payload["checks"]}
    assert checks["single SMILES input"]["status"] == "ok"
    assert checks["ADMET and skin-sens evidence"]["status"] == "ok"
    assert checks["skin toxicity prediction"]["status"] == "ok"
    assert checks["protein target prediction"]["status"] == "ok"
    assert checks["skin-specialized binding context"]["status"] == "ok"
    assert checks["run verification status"]["status"] == "ok"


def test_goal_contract_rejects_missing_admet_metric(tmp_path: Path) -> None:
    run_dir = tmp_path / "missing_admet"
    write_goal_run(run_dir)
    summary_path = run_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text())
    del summary["safety"]["admet_metrics"]["Skin_Reaction"]
    summary_path.write_text(json.dumps(summary))

    res = run_goal(["--run-dir", str(run_dir), "--smiles", "CCO", "--json"])

    assert res.returncode == 2
    payload = json.loads(res.stdout)
    checks = {check["name"]: check for check in payload["checks"]}
    assert checks["ADMET and skin-sens evidence"]["status"] == "failed"
    assert "Skin_Reaction" in checks["ADMET and skin-sens evidence"]["detail"]


def test_goal_contract_rejects_failed_run_verification(tmp_path: Path) -> None:
    run_dir = tmp_path / "failed_verification"
    write_goal_run(run_dir, verification_status="failed")

    res = run_goal(["--run-dir", str(run_dir), "--smiles", "CCO", "--json"])

    assert res.returncode == 2
    payload = json.loads(res.stdout)
    checks = {check["name"]: check for check in payload["checks"]}
    assert checks["run verification status"]["status"] == "failed"
    assert checks["underlying verifier checks"]["status"] == "failed"


def test_goal_contract_requires_successful_verifier_execution(tmp_path: Path) -> None:
    cases = (
        ("missing_returncode", lambda record: record.pop("returncode")),
        ("nonzero_returncode", lambda record: record.__setitem__("returncode", 1)),
        ("boolean_returncode", lambda record: record.__setitem__("returncode", False)),
        ("failed_verifier_status", lambda record: record.__setitem__("verifier_status", "failed")),
    )
    for name, mutate in cases:
        run_dir = tmp_path / name
        write_goal_run(run_dir)
        verification_path = run_dir / "run_verification.json"
        verification = json.loads(verification_path.read_text())
        mutate(verification)
        verification_path.write_text(json.dumps(verification))

        res = run_goal(["--run-dir", str(run_dir), "--smiles", "CCO", "--json"])

        assert res.returncode == 2, name
        checks = {check["name"]: check for check in json.loads(res.stdout)["checks"]}
        assert checks["run verifier execution"]["status"] == "failed", name


def test_goal_contract_can_require_claim_ready_skin_context(tmp_path: Path) -> None:
    run_dir = tmp_path / "skin_context_required"
    write_goal_run(run_dir)

    res = run_goal([
        "--run-dir",
        str(run_dir),
        "--smiles",
        "CCO",
        "--require-skin-context-supported",
        "--json",
    ])

    assert res.returncode == 2
    payload = json.loads(res.stdout)
    checks = {check["name"]: check for check in payload["checks"]}
    assert checks["skin-specialized binding context"]["status"] == "failed"
    assert "skin_context_supported is required" in (
        checks["skin-specialized binding context"]["detail"]
    )


def test_goal_contract_recomputes_artifact_fingerprints(tmp_path: Path) -> None:
    run_dir = tmp_path / "tampered_artifact"
    write_goal_run(run_dir)
    (run_dir / "run_verification.log").write_text("tampered after verification\n")

    res = run_goal(["--run-dir", str(run_dir), "--smiles", "CCO", "--json"])

    assert res.returncode == 2
    checks = {check["name"]: check for check in json.loads(res.stdout)["checks"]}
    assert checks["verified artifact fingerprints"]["status"] == "failed"
    assert "run_verification_log" in checks["verified artifact fingerprints"]["detail"]


def test_goal_contract_rejects_artifact_paths_outside_the_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "redirected_artifact"
    write_goal_run(run_dir)
    verification_path = run_dir / "run_verification.json"
    verification = json.loads(verification_path.read_text())
    external = tmp_path / "external.md"
    external.write_text("# external\n")
    verification["verified_artifacts"][1] = artifact(external, "run_summary_md")
    verification_path.write_text(json.dumps(verification))

    res = run_goal(["--run-dir", str(run_dir), "--smiles", "CCO", "--json"])

    assert res.returncode == 2
    checks = {check["name"]: check for check in json.loads(res.stdout)["checks"]}
    assert "run_summary_md.path" in checks["verified artifact fingerprints"]["detail"]


def test_goal_contract_rejects_expected_artifact_through_ancestor_symlink(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "ancestor_symlink"
    write_goal_run(run_dir)
    external = tmp_path / "external_fast"
    external.mkdir()
    band_paths = {
        "target_fast_original_ranking": external / "top50.csv",
        "target_fast_band_ranking": external / "top50_band_reranked.csv",
        "target_fast_band_targets": external / "daina_band_reranked_targets.csv",
    }
    for index, path in enumerate(band_paths.values(), start=1):
        path.write_text(f"rank,target_id\n{index},P10275\n")
    targets_dir = run_dir / "03_targets"
    targets_dir.mkdir(exist_ok=True)
    (targets_dir / "mode_fast").symlink_to(external, target_is_directory=True)
    verification_path = run_dir / "run_verification.json"
    verification = json.loads(verification_path.read_text())
    verification["verified_artifacts"].extend(
        artifact(path, name) for name, path in band_paths.items()
    )
    verification_path.write_text(json.dumps(verification))

    res = run_goal(["--run-dir", str(run_dir), "--smiles", "CCO", "--json"])

    assert res.returncode == 2
    checks = {check["name"]: check for check in json.loads(res.stdout)["checks"]}
    assert "target_fast_band_ranking.path" in (
        checks["verified artifact fingerprints"]["detail"]
    )


def test_goal_contract_binds_summary_and_verification_run_identity(tmp_path: Path) -> None:
    run_dir = tmp_path / "identity_mismatch"
    write_goal_run(run_dir)
    verification_path = run_dir / "run_verification.json"
    verification = json.loads(verification_path.read_text())
    verification["mode"] = "comprehensive"
    verification_path.write_text(json.dumps(verification))

    res = run_goal(["--run-dir", str(run_dir), "--smiles", "CCO", "--json"])

    assert res.returncode == 2
    checks = {check["name"]: check for check in json.loads(res.stdout)["checks"]}
    assert checks["run identity binding"]["status"] == "failed"
    assert "mode" in checks["run identity binding"]["detail"]


def test_goal_contract_validates_optional_fast_band_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "fast_band_artifacts"
    write_goal_run(run_dir)
    fast_dir = run_dir / "03_targets/mode_fast"
    fast_dir.mkdir(parents=True)
    band_paths = {
        "target_fast_original_ranking": fast_dir / "top50.csv",
        "target_fast_band_ranking": fast_dir / "top50_band_reranked.csv",
        "target_fast_band_targets": fast_dir / "daina_band_reranked_targets.csv",
    }
    for index, path in enumerate(band_paths.values(), start=1):
        path.write_text(f"rank,target_id\n{index},P10275\n")
    verification_path = run_dir / "run_verification.json"
    verification = json.loads(verification_path.read_text())
    verification["verified_artifacts"].extend(
        artifact(path, name) for name, path in band_paths.items()
    )
    verification_path.write_text(json.dumps(verification))

    accepted = run_goal(["--run-dir", str(run_dir), "--smiles", "CCO", "--json"])

    assert accepted.returncode == 0, accepted.stderr
    checks = {check["name"]: check for check in json.loads(accepted.stdout)["checks"]}
    assert checks["verified artifact fingerprints"] == {
        "name": "verified artifact fingerprints",
        "status": "ok",
        "detail": "7 required artifacts",
    }

    band_paths["target_fast_band_ranking"].write_text(
        "rank,target_id\n1,Q99999\n"
    )
    tampered = run_goal(["--run-dir", str(run_dir), "--smiles", "CCO", "--json"])

    assert tampered.returncode == 2
    checks = {check["name"]: check for check in json.loads(tampered.stdout)["checks"]}
    assert "target_fast_band_ranking.sha256" in (
        checks["verified artifact fingerprints"]["detail"]
    )
