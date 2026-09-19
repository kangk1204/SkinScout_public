"""Regression tests for the SkinScout goal contract verifier."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


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
    """Write a current-contract target-id run fixture.

    The fixture carries the Daina structural artifact the shared contract marks
    as current, the current Daina screening-count keys, and the fast band
    artifact fingerprints the goal verifier requires. The legacy PSICHIC/DTI
    shape lives in ``write_legacy_goal_run`` and is expected to be rejected.
    """
    diagnostic_reasons = diagnostic_reasons or []
    run_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = run_dir / "03_targets/ranked_targets_v3_with_efficacy.csv"
    ranking_path.parent.mkdir()
    ranking_path.write_text("final_rank,target_id,final_score\n1,P10275,0.21\n")
    fast_dir = run_dir / "03_targets/mode_fast"
    fast_dir.mkdir(parents=True)
    (fast_dir / "daina_structural_targets.csv").write_text("target_id\nP10275\n")
    band_paths = {
        "target_fast_original_ranking": fast_dir / "top50.csv",
        "target_fast_band_ranking": fast_dir / "top50_band_reranked.csv",
        "target_fast_band_targets": fast_dir / "daina_band_reranked_targets.csv",
    }
    for index, path in enumerate(band_paths.values(), start=1):
        path.write_text(f"rank,target_id\n{index},P10275\n")
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
            "screened_target_count": 2943,
            "screening_counts": {
                "daina_zoete_targets": 2943,
                "daina_primary_candidates": 256,
                "autodock_rescored_targets": 256,
                "gnina_pose_rescored_targets": 100,
                "daina_structural_targets": 256,
                "rerank_consensus_targets": 10,
                "band_reranked_targets": 10,
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
            *(
                artifact(path, name)
                for name, path in band_paths.items()
            ),
        ],
    }
    (run_dir / "run_verification.json").write_text(json.dumps(verification))


def rewrite_summary(run_dir: Path, mutate) -> None:
    """Mutate run_summary.json and refresh its recorded fingerprint."""
    summary_path = run_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text())
    mutate(summary)
    summary_path.write_text(json.dumps(summary))
    verification_path = run_dir / "run_verification.json"
    verification = json.loads(verification_path.read_text())
    for index, item in enumerate(verification["verified_artifacts"]):
        if item["name"] == "run_summary_json":
            verification["verified_artifacts"][index] = artifact(
                summary_path, "run_summary_json"
            )
    verification_path.write_text(json.dumps(verification))


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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("AMES", float("nan")),
        ("AMES", float("inf")),
        ("Skin_Reaction", -3.0),
        ("Skin_Reaction", 4.0),
        # A JSON integer too large for a float must fail closed, not raise.
        ("AMES", 10**400),
    ],
)
def test_goal_contract_rejects_nonfinite_or_out_of_range_admet_metric(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    run_dir = tmp_path / "bad_admet_metric"
    write_goal_run(run_dir)

    def mutate(summary: dict) -> None:
        summary["safety"]["admet_metrics"][field] = value

    rewrite_summary(run_dir, mutate)

    res = run_goal(["--run-dir", str(run_dir), "--smiles", "CCO", "--json"])

    assert res.returncode == 2, res.stderr
    payload = json.loads(res.stdout)
    checks = {check["name"]: check for check in payload["checks"]}
    assert checks["verified artifact fingerprints"]["status"] == "ok"
    assert checks["ADMET and skin-sens evidence"]["status"] == "failed"
    assert field in checks["ADMET and skin-sens evidence"]["detail"]


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), -3.0, 4.0],
)
def test_goal_contract_rejects_invalid_skin_sens_probability(
    tmp_path: Path,
    value: float,
) -> None:
    run_dir = tmp_path / "bad_probability"
    write_goal_run(run_dir)

    def mutate(summary: dict) -> None:
        summary["safety"]["skin_sens_evidence"][0]["probability"] = value

    rewrite_summary(run_dir, mutate)

    res = run_goal(["--run-dir", str(run_dir), "--smiles", "CCO", "--json"])

    assert res.returncode == 2, res.stderr
    payload = json.loads(res.stdout)
    checks = {check["name"]: check for check in payload["checks"]}
    assert checks["ADMET and skin-sens evidence"]["status"] == "failed"
    assert "probability" in checks["ADMET and skin-sens evidence"]["detail"]


@pytest.mark.parametrize("malformed", [None, "ok"])
def test_goal_contract_reports_malformed_verifier_checks_as_failure(
    tmp_path: Path,
    malformed: object,
) -> None:
    run_dir = tmp_path / "malformed_checks"
    write_goal_run(run_dir)
    verification_path = run_dir / "run_verification.json"
    verification = json.loads(verification_path.read_text())
    verification["checks"] = [malformed]
    verification_path.write_text(json.dumps(verification))

    res = run_goal(["--run-dir", str(run_dir), "--smiles", "CCO", "--json"])

    assert res.returncode == 2, res.stderr
    assert "Traceback" not in res.stderr
    payload = json.loads(res.stdout)
    assert payload["status"] == "failed"
    checks = {check["name"]: check for check in payload["checks"]}
    assert checks["underlying verifier checks"]["status"] == "failed"
    assert "malformed" in checks["underlying verifier checks"]["detail"]


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
    fast_dir = targets_dir / "mode_fast"
    if fast_dir.is_symlink() or fast_dir.is_file():
        fast_dir.unlink()
    elif fast_dir.exists():
        import shutil

        shutil.rmtree(fast_dir)
    (targets_dir / "mode_fast").symlink_to(external, target_is_directory=True)
    verification_path = run_dir / "run_verification.json"
    verification = json.loads(verification_path.read_text())
    verification["verified_artifacts"] = [
        item
        for item in verification["verified_artifacts"]
        if item["name"] not in band_paths
    ]
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


def test_goal_contract_requires_current_fast_band_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "fast_band_artifacts"
    write_goal_run(run_dir)

    accepted = run_goal(["--run-dir", str(run_dir), "--smiles", "CCO", "--json"])

    assert accepted.returncode == 0, accepted.stderr
    checks = {check["name"]: check for check in json.loads(accepted.stdout)["checks"]}
    assert checks["verified artifact fingerprints"] == {
        "name": "verified artifact fingerprints",
        "status": "ok",
        "detail": "7 required artifacts",
    }

    verification_path = run_dir / "run_verification.json"
    verification = json.loads(verification_path.read_text())
    verification["verified_artifacts"] = [
        item
        for item in verification["verified_artifacts"]
        if not str(item["name"]).startswith("target_fast_band")
        and item["name"] != "target_fast_original_ranking"
    ]
    verification_path.write_text(json.dumps(verification))
    missing = run_goal(["--run-dir", str(run_dir), "--smiles", "CCO", "--json"])

    assert missing.returncode == 2
    checks = {check["name"]: check for check in json.loads(missing.stdout)["checks"]}
    assert "target_fast_band_ranking" in (
        checks["verified artifact fingerprints"]["detail"]
    )

    tampered_dir = tmp_path / "fast_band_tampered"
    write_goal_run(tampered_dir)
    band_path = tampered_dir / "03_targets/mode_fast/top50_band_reranked.csv"
    band_path.write_text("rank,target_id\n1,Q99999\n")
    tampered = run_goal(["--run-dir", str(tampered_dir), "--smiles", "CCO", "--json"])

    assert tampered.returncode == 2
    checks = {check["name"]: check for check in json.loads(tampered.stdout)["checks"]}
    assert "target_fast_band_ranking.sha256" in (
        checks["verified artifact fingerprints"]["detail"]
    )


def test_goal_contract_accepts_minimal_v3_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "v3_manifest"
    write_goal_run(run_dir)
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"schema_version": "skinscout.run_manifest.v3"})
    )

    res = run_goal(["--run-dir", str(run_dir), "--smiles", "CCO", "--json"])

    assert res.returncode == 0, res.stderr
    payload = json.loads(res.stdout)
    assert payload["contract_version"] == "current"
    assert payload["legacy_label"] is None


def test_goal_contract_labels_legacy_manifest_not_currently_verified(
    tmp_path: Path,
) -> None:
    """A v2 (PSICHIC/DTI) run must not be reported as Current-contract success."""
    run_dir = tmp_path / "legacy_contract"
    write_goal_run(run_dir)
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps({"schema_version": "skinscout.run_manifest.v2"}))

    res = run_goal(["--run-dir", str(run_dir), "--smiles", "CCO", "--json"])

    assert res.returncode == 2
    payload = json.loads(res.stdout)
    assert payload["contract_version"] == "legacy/not_currently_verified"
    assert payload["legacy_label"] == "legacy/not_currently_verified"
    checks = {check["name"]: check for check in payload["checks"]}
    assert checks["artifact contract version"]["status"] == "failed"
    assert "legacy/not_currently_verified" in (
        checks["artifact contract version"]["detail"]
    )


def test_goal_contract_rejects_manifest_mode_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "manifest_mode_mismatch"
    write_goal_run(run_dir)
    (run_dir / "run_manifest.json").write_text(
        json.dumps({
            "schema_version": "skinscout.run_manifest.v3",
            "run_profile": {
                "normalized_preset": "target-id",
                "execution_mode": "comprehensive",
            },
        })
    )

    res = run_goal(["--run-dir", str(run_dir), "--smiles", "CCO", "--json"])

    assert res.returncode == 2
    checks = {check["name"]: check for check in json.loads(res.stdout)["checks"]}
    assert checks["artifact contract version"]["status"] == "failed"
    assert "execution_mode" in checks["artifact contract version"]["detail"]


def test_goal_contract_rejects_summary_with_a_different_ranking_binding(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "different_ranking"
    write_goal_run(run_dir)
    summary_path = run_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["target_prediction"]["ranking_path"] = (
        "03_targets/ranked_targets_v3_with_efficacy_other.csv"
    )
    summary_path.write_text(json.dumps(summary))
    # Refresh the summary fingerprint so only the binding inconsistency remains.
    verification_path = run_dir / "run_verification.json"
    verification = json.loads(verification_path.read_text())
    for index, item in enumerate(verification["verified_artifacts"]):
        if item["name"] == "run_summary_json":
            verification["verified_artifacts"][index] = artifact(
                summary_path, "run_summary_json"
            )
    verification_path.write_text(json.dumps(verification))

    res = run_goal(["--run-dir", str(run_dir), "--smiles", "CCO", "--json"])

    assert res.returncode == 2
    checks = {check["name"]: check for check in json.loads(res.stdout)["checks"]}
    assert checks["ranking binding"]["status"] == "failed"
    assert "ranking_path" in checks["ranking binding"]["detail"]


def test_goal_contract_writes_new_verification_without_mutating_old_files(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "new_verification_record"
    write_goal_run(run_dir)
    old_bytes = (run_dir / "run_verification.json").read_bytes()
    out_path = run_dir / "goal_contract_verification.json"

    res = run_goal([
        "--run-dir",
        str(run_dir),
        "--smiles",
        "CCO",
        "--write-verification",
        "--json",
    ])

    assert res.returncode == 0, res.stderr
    assert (run_dir / "run_verification.json").read_bytes() == old_bytes
    record = json.loads(out_path.read_text())
    assert record["schema_version"] == "skinscout.goal_contract_verification.v2"
    assert record["status"] == "ok"
    assert record["contract_version"] == "current"
    record_bytes = out_path.read_bytes()

    again = run_goal([
        "--run-dir",
        str(run_dir),
        "--smiles",
        "CCO",
        "--write-verification",
        "--json",
    ])

    assert again.returncode == 1
    assert "refusing to overwrite" in again.stderr
    assert out_path.read_bytes() == record_bytes
