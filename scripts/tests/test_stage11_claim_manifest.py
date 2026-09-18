"""Tests for Stage 11 publication claim manifest gates."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "stage11_claim_manifest.py"


def run_script(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def write_eval_manifest(path: Path, *, claim_ready: bool = True) -> Path:
    from scripts.tests.test_validate_activity_retrieval_gate import _contract
    from scripts.validate_activity_retrieval_gate import create_gate

    path.parent.mkdir(parents=True, exist_ok=True)
    contract_root = path.parent / "activity-gate-contract"
    contract_root.mkdir(exist_ok=True)
    gate = contract_root / "activity_retrieval_final_gate.flag"
    create_gate(**_contract(contract_root), out_gate=gate)
    blockers = [] if claim_ready else [{"code": "threshold_failures"}]
    return write_text(
        path,
        json.dumps({
            "claim_ready": claim_ready,
            "claim_blockers": blockers,
            "n_steps": 1,
            "threshold_status": {"n_checked": 1, "n_failed": 0},
            "activity_retrieval_gate": {
                "status": "pass",
                "schema_version": (
                    "skinscout.activity-retrieval-production-gate.v1"
                ),
                "path": str(gate.resolve()),
                "bytes": gate.stat().st_size,
                "sha256": hashlib.sha256(gate.read_bytes()).hexdigest(),
            },
        }) + "\n",
    )


def test_claim_manifest_success(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "case"
    report = write_text(run_dir / "09_report" / "index.html", "<html>ok</html>\n")
    status = write_text(
        run_dir / "publication" / "status.json",
        json.dumps({
            "execution_status": "completed",
            "diagnostic_only": False,
            "claim_eligible": True,
        }) + "\n",
    )
    write_text(
        run_dir / "publication" / "reproducibility" / "config_hash.txt",
        "a" * 64 + "\n",
    )
    write_text(run_dir / "00_untrusted" / "config_hash.txt", "wrong\n")
    eval_manifest = write_eval_manifest(tmp_path / "eval" / "iteration_manifest.json")
    out = run_dir / "publication" / "claim_manifest.json"

    res = run_script([
        "--run-dir", str(run_dir),
        "--out-manifest", str(out),
        "--evaluation-manifest", str(eval_manifest),
        "--artifact", f"report={report}",
        "--artifact", f"status={status}",
    ])

    assert res.returncode == 0, res.stderr
    payload = json.loads(out.read_text())
    assert payload["run_id"] == "case"
    assert payload["claim_ready"] is True
    assert payload["blockers"] == []
    assert payload["config_hash"] == "a" * 64
    assert payload["scientific_claim_scope"] == (
        "computational_and_literature_candidate_validation_only"
    )
    artifacts = {item["label"]: item for item in payload["artifacts"]}
    assert artifacts["report"]["bytes"] == report.stat().st_size
    assert artifacts["report"]["sha256"] == hashlib.sha256(
        report.read_bytes()
    ).hexdigest()
    assert artifacts["report"]["relative_path"] == "09_report/index.html"
    assert payload["evaluation_manifest"]["claim_ready"] is True


def test_claim_manifest_binds_named_reproducibility_and_eval_provenance(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "case"
    report = write_text(run_dir / "report.html", "ok\n")
    repro_dir = run_dir / "publication" / "reproducibility"
    git_sidecar = write_text(repro_dir / "git_commit.txt", "b" * 40 + "\n")
    config_sidecar = write_text(repro_dir / "config_hash.txt", "a" * 64 + "\n")
    snapshot_sidecar = write_text(
        repro_dir / "code_snapshot.json",
        json.dumps({
            "schema_version": "skinscout.stage11-code-snapshot.v1",
            "git_commit": "b" * 40,
            "dirty": False,
            "limitation": None,
        }) + "\n",
    )
    effective_sidecar = write_text(
        repro_dir / "effective_config.json",
        json.dumps({
            "schema_version": "skinscout.stage11-effective-config.v1",
            "source": "config_yaml_plus_cli_overrides",
            "base_config_sha256": "a" * 64,
            "effective_config_sha256": "c" * 64,
            "overrides": [],
        }) + "\n",
    )
    metadata = []
    for sidecar in (
        git_sidecar,
        config_sidecar,
        snapshot_sidecar,
        effective_sidecar,
    ):
        metadata.append({
            "relative_path": sidecar.name,
            "bytes": sidecar.stat().st_size,
            "sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
        })
    repro_manifest = write_text(
        repro_dir / "artifact_manifest.json",
        json.dumps({"metadata_files": metadata}) + "\n",
    )
    eval_manifest = write_eval_manifest(tmp_path / "eval.json")
    eval_payload = json.loads(eval_manifest.read_text())
    eval_payload["provenance"] = {
        "git_commit": "b" * 40,
        "config_sha256": "a" * 64,
    }
    eval_manifest.write_text(json.dumps(eval_payload) + "\n")
    out = run_dir / "claim_manifest.json"

    res = run_script([
        "--run-dir", str(run_dir),
        "--out-manifest", str(out),
        "--evaluation-manifest", str(eval_manifest),
        "--artifact", f"report={report}",
        "--artifact", f"reproducibility={repro_manifest}",
    ])

    assert res.returncode == 0, res.stderr
    payload = json.loads(out.read_text())
    assert payload["git_commit"] == "b" * 40
    assert payload["config_hash"] == "a" * 64

    config_sidecar.write_text("c" * 64 + "\n")
    res = run_script([
        "--run-dir", str(run_dir),
        "--out-manifest", str(out),
        "--evaluation-manifest", str(eval_manifest),
        "--artifact", f"report={report}",
        "--artifact", f"reproducibility={repro_manifest}",
    ])
    assert res.returncode != 0
    assert "reproducibility_provenance_binding_mismatch" in res.stderr
    assert not out.exists()


def test_claim_manifest_removes_stale_output_on_blocked_upstream(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "case"
    status = write_text(
        run_dir / "status.json",
        json.dumps({"execution_status": "blocked", "claim_eligible": True}) + "\n",
    )
    eval_manifest = write_eval_manifest(tmp_path / "eval.json")
    out = write_text(run_dir / "claim_manifest.json", '{"stale": true}\n')

    res = run_script([
        "--run-dir", str(run_dir),
        "--out-manifest", str(out),
        "--evaluation-manifest", str(eval_manifest),
        "--artifact", f"status={status}",
    ])

    assert res.returncode != 0
    assert "blocked_execution_status" in res.stderr
    assert not out.exists()


def test_claim_manifest_rejects_stale_activity_retrieval_gate(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "case"
    artifact = write_text(run_dir / "report.html", "ok\n")
    eval_manifest = write_eval_manifest(tmp_path / "eval.json")
    eval_payload = json.loads(eval_manifest.read_text())
    gate = Path(eval_payload["activity_retrieval_gate"]["path"])
    gate.write_text('{"status":"pass"}\n')
    out = run_dir / "claim_manifest.json"

    res = run_script([
        "--run-dir", str(run_dir),
        "--out-manifest", str(out),
        "--evaluation-manifest", str(eval_manifest),
        "--artifact", f"report={artifact}",
    ])

    assert res.returncode != 0
    assert "activity_retrieval_gate_binding_mismatch" in res.stderr
    assert not out.exists()


def test_claim_manifest_removes_stale_output_on_missing_artifact(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "case"
    run_dir.mkdir(parents=True)
    eval_manifest = write_eval_manifest(tmp_path / "eval.json")
    out = write_text(run_dir / "claim_manifest.json", '{"stale": true}\n')

    res = run_script([
        "--run-dir", str(run_dir),
        "--out-manifest", str(out),
        "--evaluation-manifest", str(eval_manifest),
        "--artifact", "report=missing.html",
    ])

    assert res.returncode != 0
    assert "missing_artifact" in res.stderr
    assert not out.exists()


def test_claim_manifest_rejects_duplicate_labels_and_paths(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "case"
    artifact = write_text(run_dir / "report.html", "ok\n")
    other = write_text(run_dir / "other.html", "ok\n")
    eval_manifest = write_eval_manifest(tmp_path / "eval.json")
    out = write_text(run_dir / "claim_manifest.json", '{"stale": true}\n')

    duplicate_label = run_script([
        "--run-dir", str(run_dir),
        "--out-manifest", str(out),
        "--evaluation-manifest", str(eval_manifest),
        "--artifact", f"report={artifact}",
        "--artifact", f"report={other}",
    ])

    assert duplicate_label.returncode != 0
    assert "duplicate_artifact_label" in duplicate_label.stderr
    assert not out.exists()

    out.write_text('{"stale": true}\n')
    duplicate_path = run_script([
        "--run-dir", str(run_dir),
        "--out-manifest", str(out),
        "--evaluation-manifest", str(eval_manifest),
        "--artifact", f"report={artifact}",
        "--artifact", f"copy={artifact}",
    ])

    assert duplicate_path.returncode != 0
    assert "duplicate_artifact_path" in duplicate_path.stderr
    assert not out.exists()


def test_claim_manifest_diagnostic_mode_writes_not_ready_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "case"
    status = write_text(
        run_dir / "status.json",
        json.dumps({"diagnostic_only": True, "claim_eligible": False}) + "\n",
    )
    eval_manifest = write_eval_manifest(tmp_path / "eval.json", claim_ready=False)
    out = run_dir / "claim_manifest.json"

    res = run_script([
        "--run-dir", str(run_dir),
        "--out-manifest", str(out),
        "--evaluation-manifest", str(eval_manifest),
        "--artifact", f"status={status}",
        "--allow-diagnostic",
    ])

    assert res.returncode == 0, res.stderr
    payload = json.loads(out.read_text())
    assert payload["claim_ready"] is False
    codes = {item["code"] for item in payload["blockers"]}
    assert "diagnostic_only_status" in codes
    assert "claim_ineligible_status" in codes
    assert "evaluation_not_claim_ready" in codes
