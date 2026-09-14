"""Regression tests for immutable Stage 9 report publication contracts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CONTRACT = _load_module(
    "report_package_contract",
    SCRIPTS / "report_package_contract.py",
)
REPRO = _load_module(
    "stage11_repro_pack_report_contract_tests",
    SCRIPTS / "stage11_repro_pack.py",
)


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, separators=(",", ": ")) + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _write_activity_bound_eval_manifest(path: Path) -> None:
    from scripts.tests.test_validate_activity_retrieval_gate import _contract
    from scripts.validate_activity_retrieval_gate import create_gate

    contract_root = path.parent / "activity-gate-contract"
    contract_root.mkdir(parents=True, exist_ok=True)
    gate = contract_root / "activity_retrieval_final_gate.flag"
    create_gate(**_contract(contract_root), out_gate=gate)
    _write_bytes(path, _json_bytes({
        "claim_ready": True,
        "claim_blockers": [],
        "activity_retrieval_gate": {
            "status": "pass",
            "schema_version": "skinscout.activity-retrieval-production-gate.v1",
            "path": str(gate.resolve()),
            "bytes": gate.stat().st_size,
            "sha256": hashlib.sha256(gate.read_bytes()).hexdigest(),
        },
    }))


def _artifact_id(checksums: dict[str, Any]) -> str:
    canonical = json.dumps(checksums, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _write_package(
    run_dir: Path,
    *,
    kind: str,
    parent_fast_hash: str | None = None,
) -> tuple[str, Path]:
    if kind == "fast":
        payloads = {
            "index.html": b"<html><body>sealed report</body></html>\n",
            "targets.json": b'{"targets":[{"target_id":"P1"}]}\n',
        }
        manifest_core: dict[str, Any] = {
            "schema_version": "skinscout.report_fast.v1",
            "sealed": True,
            "run_id": run_dir.name,
            "mode": "fast",
            "top_target_id": "P1",
            "target_count": 1,
            "compound": {"inchikey": "TEST"},
            "viewer": {"name": "Mol*", "provenance_verified": True},
            "browser_verification": {"status": "passed"},
            "source_hashes": {"stage3": {"sha256": "a" * 64, "bytes": 1}},
            "checksums_path": "checksums.json",
        }
        checksum_schema = "skinscout.report_fast.checksums.v1"
    else:
        assert parent_fast_hash is not None
        payloads = {
            "summary.json": _json_bytes({
                "schema_version": "skinscout.report_physics.summary.v1",
                "parent_fast_hash": parent_fast_hash,
            }),
        }
        manifest_core = {
            "schema_version": "skinscout.report_physics.v1",
            "sealed": True,
            "run_id": run_dir.name,
            "mode": "comprehensive",
            "parent_fast_hash": parent_fast_hash,
            "source_hashes": {"mmgbsa": {"sha256": "b" * 64, "bytes": 1}},
            "checksums_path": "checksums.json",
        }
        checksum_schema = "skinscout.report_physics.checksums.v1"

    identity = {
        "schema_version": "skinscout.report_identity.v1",
        "kind": kind,
        "manifest_core": manifest_core,
    }
    payloads["identity.json"] = _json_bytes(identity)
    checksums = {
        "schema_version": checksum_schema,
        "files": [
            {
                "path": relative,
                "sha256": _sha256_bytes(payload),
                "bytes": len(payload),
            }
            for relative, payload in sorted(payloads.items())
        ],
    }
    artifact_id = _artifact_id(checksums)
    package_root = run_dir / "reports" / kind / artifact_id
    for relative, payload in payloads.items():
        _write_bytes(package_root / relative, payload)
    _write_bytes(package_root / "checksums.json", _json_bytes(checksums))
    manifest = {
        **manifest_core,
        "artifact_id": artifact_id,
        "identity_path": "identity.json",
        "identity_sha256": _sha256_bytes(payloads["identity.json"]),
    }
    if kind == "fast":
        manifest["parent_linkage"] = {
            "fast_parent_hash": artifact_id,
            "physics_parent_hash": None,
            "physics_parent_status": "not_requested",
            "physics_parent_manifest": None,
        }
    _write_bytes(package_root / "manifest.json", _json_bytes(manifest))
    return artifact_id, package_root


def _write_report_contract(
    run_dir: Path,
    *,
    include_physics: bool,
) -> tuple[Path, Path, Path, Path | None]:
    fast_id, fast_root = _write_package(run_dir, kind="fast")
    report_dir = run_dir / "09_report"
    fast_pointer = report_dir / "fast_report_manifest.json"
    _write_bytes(fast_pointer, _json_bytes({
        "schema_version": "skinscout.report_pointer.v1",
        "kind": "fast",
        "status": "ready",
        "artifact_id": fast_id,
        "sealed": True,
        "package_path": f"reports/fast/{fast_id}",
        "manifest_path": f"reports/fast/{fast_id}/manifest.json",
        "index_path": f"reports/fast/{fast_id}/index.html",
    }))

    physics_pointer = report_dir / "physics_report_manifest.json"
    physics_root: Path | None = None
    if include_physics:
        physics_id, physics_root = _write_package(
            run_dir,
            kind="physics",
            parent_fast_hash=fast_id,
        )
        _write_bytes(physics_pointer, _json_bytes({
            "schema_version": "skinscout.report_pointer.v1",
            "kind": "physics",
            "status": "ready",
            "artifact_id": physics_id,
            "sealed": True,
            "parent_fast_hash": fast_id,
            "parent_manifest_path": f"reports/fast/{fast_id}/manifest.json",
            "package_path": f"reports/physics/{physics_id}",
            "manifest_path": f"reports/physics/{physics_id}/manifest.json",
        }))
    else:
        _write_bytes(physics_pointer, _json_bytes({
            "schema_version": "skinscout.report_pointer.v1",
            "kind": "physics",
            "status": "not_requested",
            "reason": "fast profile",
            "artifact_id": None,
            "sealed": False,
        }))
    return fast_pointer, physics_pointer, fast_root, physics_root


def test_fast_only_report_contract_passes_when_physics_is_optional(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "case"
    fast_pointer, physics_pointer, _, _ = _write_report_contract(
        run_dir,
        include_physics=False,
    )

    result = CONTRACT.validate_report_packages(
        run_dir=run_dir,
        fast_pointer=fast_pointer,
        physics_pointer=physics_pointer,
    )

    assert result["status"] == "passed"
    assert result["fast"]["sealed"] is True
    assert result["physics"]["status"] == "not_requested"


def test_linked_physics_report_contract_passes_when_required(tmp_path: Path) -> None:
    run_dir = tmp_path / "case"
    fast_pointer, physics_pointer, _, _ = _write_report_contract(
        run_dir,
        include_physics=True,
    )

    result = CONTRACT.validate_report_packages(
        run_dir=run_dir,
        fast_pointer=fast_pointer,
        physics_pointer=physics_pointer,
        require_physics=True,
    )

    assert result["physics"]["status"] == "ready"
    assert result["physics"]["package"]["parent_fast_hash"] == (
        result["fast"]["artifact_id"]
    )


def test_report_contract_rejects_missing_required_physics(tmp_path: Path) -> None:
    run_dir = tmp_path / "case"
    fast_pointer, physics_pointer, _, _ = _write_report_contract(
        run_dir,
        include_physics=False,
    )

    with pytest.raises(CONTRACT.ReportContractError, match="Physics report is required"):
        CONTRACT.validate_report_packages(
            run_dir=run_dir,
            fast_pointer=fast_pointer,
            physics_pointer=physics_pointer,
            require_physics=True,
        )


def test_report_contract_rejects_checksum_bound_payload_tampering(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "case"
    fast_pointer, physics_pointer, fast_root, _ = _write_report_contract(
        run_dir,
        include_physics=False,
    )
    (fast_root / "index.html").write_text("<html>tampered</html>\n")

    with pytest.raises(CONTRACT.ReportContractError, match="payload .* mismatch"):
        CONTRACT.validate_report_packages(
            run_dir=run_dir,
            fast_pointer=fast_pointer,
            physics_pointer=physics_pointer,
        )


def test_report_contract_rejects_wrong_physics_parent(tmp_path: Path) -> None:
    run_dir = tmp_path / "case"
    fast_pointer, physics_pointer, _, _ = _write_report_contract(
        run_dir,
        include_physics=True,
    )
    pointer = json.loads(physics_pointer.read_text())
    pointer["parent_fast_hash"] = "f" * 64
    physics_pointer.write_bytes(_json_bytes(pointer))

    with pytest.raises(CONTRACT.ReportContractError, match="parent fast hash mismatch"):
        CONTRACT.validate_report_packages(
            run_dir=run_dir,
            fast_pointer=fast_pointer,
            physics_pointer=physics_pointer,
            require_physics=True,
        )


def test_stage11_repro_gate_rejects_deleted_fast_package(tmp_path: Path) -> None:
    run_dir = tmp_path / "case"
    fast_pointer, physics_pointer, fast_root, _ = _write_report_contract(
        run_dir,
        include_physics=False,
    )
    for path in sorted(fast_root.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        else:
            path.rmdir()
    fast_root.rmdir()

    with pytest.raises(SystemExit, match="immutable report validation failed"):
        REPRO._validate_report_packages_for_publication(
            run_dir=run_dir,
            fast_pointer=fast_pointer,
            physics_pointer=physics_pointer,
            require_physics=False,
        )


def test_stage11_claim_gate_rejects_deleted_fast_package(tmp_path: Path) -> None:
    run_dir = tmp_path / "case"
    fast_pointer, physics_pointer, fast_root, _ = _write_report_contract(
        run_dir,
        include_physics=False,
    )
    artifact = run_dir / "09_report" / "index.html"
    artifact.write_text("<html>compatibility report</html>\n")
    for path in sorted(fast_root.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        else:
            path.rmdir()
    fast_root.rmdir()
    eval_manifest = tmp_path / "eval" / "iteration_manifest.json"
    _write_activity_bound_eval_manifest(eval_manifest)
    out_manifest = run_dir / "publication" / "claim_manifest.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "stage11_claim_manifest.py"),
            "--run-dir",
            str(run_dir),
            "--out-manifest",
            str(out_manifest),
            "--evaluation-manifest",
            str(eval_manifest),
            "--fast-report-pointer",
            str(fast_pointer),
            "--physics-report-pointer",
            str(physics_pointer),
            "--artifact",
            f"report={artifact}",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "invalid_report_package_contract" in result.stderr
    assert not out_manifest.exists()


def test_stage11_claim_manifest_records_validated_report_packages(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "case"
    fast_pointer, physics_pointer, _, _ = _write_report_contract(
        run_dir,
        include_physics=True,
    )
    artifact = run_dir / "09_report" / "index.html"
    artifact.write_text("<html>compatibility report</html>\n")
    eval_manifest = tmp_path / "eval" / "iteration_manifest.json"
    _write_activity_bound_eval_manifest(eval_manifest)
    out_manifest = run_dir / "publication" / "claim_manifest.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "stage11_claim_manifest.py"),
            "--run-dir",
            str(run_dir),
            "--out-manifest",
            str(out_manifest),
            "--evaluation-manifest",
            str(eval_manifest),
            "--fast-report-pointer",
            str(fast_pointer),
            "--physics-report-pointer",
            str(physics_pointer),
            "--require-physics-report",
            "--artifact",
            f"report={artifact}",
            "--artifact",
            f"fast_report_pointer={fast_pointer}",
            "--artifact",
            f"physics_report_pointer={physics_pointer}",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(out_manifest.read_text())
    assert payload["claim_ready"] is True
    assert payload["report_packages"]["status"] == "passed"
    assert payload["report_packages"]["physics"]["status"] == "ready"
    labels = {record["label"] for record in payload["artifacts"]}
    assert {"report", "fast_report_pointer", "physics_report_pointer"} <= labels
