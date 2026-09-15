"""Tests for Release 1 host and 60-minute SLA qualification evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import qualification_manifest as qualification  # noqa: E402


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def environment(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": qualification.ENVIRONMENT_SCHEMA_VERSION,
        "captured_at_utc": "2026-08-14T00:00:00Z",
        "os": {
            "id": "ubuntu",
            "version_id": "24.04",
            "pretty_name": "Ubuntu 24.04.3 LTS",
        },
        "kernel": "6.8.0-79-generic",
        "cpu": {
            "model": "AMD Ryzen 9 7950X 16-Core Processor",
            "logical_cores": 32,
        },
        "memory_bytes": 140_000_000_000,
        "gpu": {
            "name": "NVIDIA GeForce RTX 4090",
            "memory_bytes": 25_000_000_000,
            "count": 1,
        },
        "nvidia": {
            "package": "nvidia-driver-580=580.65.06-0ubuntu0.24.04.1",
            "driver_version": "580.65.06",
        },
        "docker": {"version": "28.3.3", "compose_version": "2.39.1"},
        "nvidia_container_toolkit": {"version": "1.17.8"},
        "storage": {
            "device": "/dev/nvme0n1",
            "technology": "nvme",
            "sequential_read_bytes_per_second": 7_100_000_000,
        },
        "cpu_governor": "performance",
        "commands": [
            {
                "name": name,
                "argv": ["skinscout-qualification", name],
                "exit_code": 0,
                "stdout_sha256": digest(f"command:{name}"),
            }
            for name in sorted(qualification.REQUIRED_COMMAND_NAMES)
        ],
    }
    payload.update(overrides)
    return payload


def artifact_lock() -> dict[str, Any]:
    return {
        "schema_version": qualification.ARTIFACT_LOCK_SCHEMA_VERSION,
        "created_at_utc": "2026-08-14T00:00:00Z",
        "images": [
            {
                "name": name,
                "reference": f"ghcr.io/kangk1204/{name}@sha256:{digest('image:' + name)}",
                "sbom_sha256": digest("sbom:" + name),
                "provenance_sha256": digest("provenance:" + name),
            }
            for name in sorted(qualification.REQUIRED_IMAGE_NAMES)
        ],
        "data": [
            {
                "name": "skinscout-data-core",
                "version": "1.0.0",
                "sha256": digest("data-core"),
            }
        ],
        "models": [
            {
                "name": "skinscout-target-model",
                "version": "1.0.0",
                "sha256": digest("target-model"),
            }
        ],
    }


def make_inputs(
    tmp_path: Path,
    *,
    environment_payload: dict[str, Any] | None = None,
    durations: dict[tuple[str, str, int], float] | None = None,
    outcomes: dict[tuple[str, str, int], str] | None = None,
) -> tuple[Path, Path, Path]:
    env_path = tmp_path / "environment.json"
    lock_path = tmp_path / "artifact-lock.json"
    ledger_path = tmp_path / "sla-ledger.json"
    write_json(env_path, environment_payload or environment())
    write_json(lock_path, artifact_lock())
    env_sha = qualification._sha256(env_path)
    lock_sha = qualification._sha256(lock_path)
    durations = durations or {}
    outcomes = outcomes or {}
    accepted = datetime(2026, 8, 14, tzinfo=timezone.utc)

    panels = {
        "public": {
            "selection_method": "fixed_public",
            "panel_sha256": digest("public-panel"),
            "compound_ids": [f"public-{index:02d}" for index in range(20)],
        },
        "sealed": {
            "selection_method": "sealed_hash_sampled",
            "panel_sha256": digest("sealed-panel"),
            "compound_ids": [f"sealed-{index:02d}" for index in range(20)],
        },
    }
    for panel in panels.values():
        panel["compound_ids_sha256"] = qualification._canonical_sha256(
            {"compound_ids": sorted(panel["compound_ids"])}
        )
    observations: list[dict[str, Any]] = []
    for panel_id, panel in panels.items():
        for compound_id in panel["compound_ids"]:
            for replicate in qualification.REPLICATES:
                key = (panel_id, compound_id, replicate)
                outcome = outcomes.get(key, "ready")
                duration = durations.get(key, 1_800.0)
                ready = accepted + timedelta(seconds=duration)
                report_path = tmp_path / "reports" / panel_id / compound_id / f"{replicate}.json"
                if outcome == "ready":
                    report_path.parent.mkdir(parents=True, exist_ok=True)
                    report_path.write_text(
                        json.dumps({"sealed": True, "key": key}, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                observations.append(
                    {
                        "panel_id": panel_id,
                        "compound_id": compound_id,
                        "replicate": replicate,
                        "outcome": outcome,
                        "accepted_at": accepted.isoformat().replace("+00:00", "Z"),
                        "ready_at": (
                            ready.isoformat().replace("+00:00", "Z")
                            if outcome == "ready"
                            else None
                        ),
                        "failure_reason": None if outcome == "ready" else f"synthetic {outcome}",
                        "stage_elapsed_seconds": {"target_and_report": min(duration, 1_800.0)},
                        "cache_counts": {"hits": 10, "misses": 2},
                        "environment_sha256": env_sha,
                        "artifact_lock_sha256": lock_sha,
                        "run_cache_cleared": True,
                        "sealed_report": (
                            {
                                "path": str(report_path.resolve()),
                                "bytes": report_path.stat().st_size,
                                "sha256": qualification._sha256(report_path),
                            }
                            if outcome == "ready"
                            else None
                        ),
                    }
                )
    ledger = {
        "schema_version": qualification.LEDGER_SCHEMA_VERSION,
        "created_at_utc": "2026-08-14T12:00:00Z",
        "panels": panels,
        "shared_precompute_sha256": digest("shared-precompute"),
        "observations": observations,
    }
    write_json(ledger_path, ledger)
    return env_path, ledger_path, lock_path


def evaluate_paths(paths: tuple[Path, Path, Path]) -> dict[str, Any]:
    env_path, ledger_path, lock_path = paths
    return qualification.evaluate(
        environment_json=env_path,
        sla_ledger_json=ledger_path,
        artifact_lock_json=lock_path,
    )


def test_passing_evidence_stays_unqualified_until_signature(tmp_path: Path) -> None:
    paths = make_inputs(tmp_path)
    payload = evaluate_paths(paths)

    assert payload["qualification_evidence_status"] == "passed"
    assert payload["reference_host_status"] == "reference_candidate"
    assert payload["release_qualified"] is False
    assert payload["failed_gates"] == []
    assert len(payload["ledger"]["observations"]) == 120
    qualification.validate_manifest_against_inputs(
        payload,
        environment_json=paths[0],
        sla_ledger_json=paths[1],
        artifact_lock_json=paths[2],
    )


def test_nearest_rank_p95_uses_observation_57(tmp_path: Path) -> None:
    durations: dict[tuple[str, str, int], float] = {}
    index = 0
    for compound_index in range(20):
        for replicate in qualification.REPLICATES:
            index += 1
            durations[("public", f"public-{compound_index:02d}", replicate)] = float(index)
    payload = evaluate_paths(make_inputs(tmp_path, durations=durations))
    public = payload["panel_summaries"][0]

    assert public["p95_rank"] == 57
    assert public["p95_seconds"] == 57.0


def test_four_timeouts_make_observation_57_infinite(tmp_path: Path) -> None:
    outcomes = {
        ("public", "public-18", 3): "timeout",
        ("public", "public-19", 1): "timeout",
        ("public", "public-19", 2): "timeout",
        ("public", "public-19", 3): "timeout",
    }
    payload = evaluate_paths(make_inputs(tmp_path, outcomes=outcomes))
    public = payload["panel_summaries"][0]

    assert public["p95_is_infinite"] is True
    assert public["p95_seconds"] is None
    assert "public_nearest_rank_p95" in {gate["name"] for gate in payload["failed_gates"]}


def test_two_failures_fail_compound_median_gate(tmp_path: Path) -> None:
    outcomes = {
        ("sealed", "sealed-00", 1): "failure",
        ("sealed", "sealed-00", 2): "failure",
    }
    payload = evaluate_paths(make_inputs(tmp_path, outcomes=outcomes))

    assert "sealed_all_compound_medians" in {
        gate["name"] for gate in payload["failed_gates"]
    }


def test_success_over_75_minutes_fails_maximum_gate(tmp_path: Path) -> None:
    durations = {("public", "public-00", 1): 4_501.0}
    payload = evaluate_paths(make_inputs(tmp_path, durations=durations))

    assert "public_max_successful_duration" in {
        gate["name"] for gate in payload["failed_gates"]
    }


def test_missing_replicate_is_rejected(tmp_path: Path) -> None:
    paths = make_inputs(tmp_path)
    ledger = json.loads(paths[1].read_text(encoding="utf-8"))
    ledger["observations"].pop()
    write_json(paths[1], ledger)

    with pytest.raises(qualification.QualificationError, match="exactly three runs"):
        evaluate_paths(paths)


def test_duplicate_observation_is_rejected(tmp_path: Path) -> None:
    paths = make_inputs(tmp_path)
    ledger = json.loads(paths[1].read_text(encoding="utf-8"))
    ledger["observations"][-1] = copy.deepcopy(ledger["observations"][0])
    write_json(paths[1], ledger)

    with pytest.raises(qualification.QualificationError, match="duplicate"):
        evaluate_paths(paths)


def test_ubuntu_26_is_compatibility_only_not_reference_qualification(tmp_path: Path) -> None:
    env = environment(
        os={
            "id": "ubuntu",
            "version_id": "26.04",
            "pretty_name": "Ubuntu 26.04 LTS",
        }
    )
    payload = evaluate_paths(make_inputs(tmp_path, environment_payload=env))

    assert payload["qualification_evidence_status"] == "failed"
    assert payload["reference_host_status"] == "ubuntu_26_04_compatibility_only"
    assert [gate["name"] for gate in payload["failed_gates"]] == ["reference_os"]


@pytest.mark.parametrize(
    ("field", "value", "failed_gate"),
    [
        ("memory_bytes", qualification.MIN_MEMORY_BYTES - 1, "reference_memory"),
        (
            "gpu",
            {"name": "NVIDIA GeForce RTX 4080", "memory_bytes": 25_000_000_000, "count": 1},
            "reference_gpu",
        ),
        (
            "storage",
            {"device": "/dev/nvme0n1", "technology": "nvme", "sequential_read_bytes_per_second": 6_999_999_999},
            "reference_storage",
        ),
        ("cpu_governor", "powersave", "reference_cpu_governor"),
    ],
)
def test_reference_hardware_boundaries_fail_closed(
    tmp_path: Path,
    field: str,
    value: Any,
    failed_gate: str,
) -> None:
    payload = evaluate_paths(
        make_inputs(tmp_path, environment_payload=environment(**{field: value}))
    )

    assert failed_gate in {gate["name"] for gate in payload["failed_gates"]}


def test_placeholder_artifact_digest_is_rejected(tmp_path: Path) -> None:
    paths = make_inputs(tmp_path)
    lock = json.loads(paths[2].read_text(encoding="utf-8"))
    lock["images"][0]["reference"] = "ghcr.io/example/app@sha256:" + "0" * 64
    write_json(paths[2], lock)

    with pytest.raises(qualification.QualificationError, match="non-placeholder"):
        evaluate_paths(paths)


def test_public_and_sealed_panels_must_be_disjoint(tmp_path: Path) -> None:
    paths = make_inputs(tmp_path)
    ledger = json.loads(paths[1].read_text(encoding="utf-8"))
    ledger["panels"]["sealed"]["compound_ids"] = list(
        ledger["panels"]["public"]["compound_ids"]
    )
    ledger["panels"]["sealed"]["compound_ids_sha256"] = qualification._canonical_sha256(
        {"compound_ids": sorted(ledger["panels"]["sealed"]["compound_ids"])}
    )
    write_json(paths[1], ledger)

    with pytest.raises(qualification.QualificationError, match="must be disjoint"):
        evaluate_paths(paths)


def test_non_nvme_device_fails_reference_storage_gate(tmp_path: Path) -> None:
    env = environment(
        storage={
            "device": "/dev/sda",
            "technology": "ssd",
            "sequential_read_bytes_per_second": 7_100_000_000,
        }
    )
    payload = evaluate_paths(make_inputs(tmp_path, environment_payload=env))

    assert "reference_storage" in {gate["name"] for gate in payload["failed_gates"]}


def test_ready_report_mutation_invalidates_manifest(tmp_path: Path) -> None:
    paths = make_inputs(tmp_path)
    payload = evaluate_paths(paths)
    report_path = Path(payload["ledger"]["observations"][0]["sealed_report"]["path"])
    report_path.write_text('{"mutated": true}\n', encoding="utf-8")

    with pytest.raises(qualification.QualificationError, match="active immutable file"):
        qualification.validate_manifest(payload)


def test_active_input_drift_is_rejected(tmp_path: Path) -> None:
    paths = make_inputs(tmp_path)
    payload = evaluate_paths(paths)
    env = json.loads(paths[0].read_text(encoding="utf-8"))
    env["kernel"] = "6.8.0-80-generic"
    write_json(paths[0], env)

    with pytest.raises(qualification.QualificationError, match="active input"):
        qualification.validate_manifest_against_inputs(
            payload,
            environment_json=paths[0],
            sla_ledger_json=paths[1],
            artifact_lock_json=paths[2],
        )


def test_binding_tamper_is_rejected(tmp_path: Path) -> None:
    payload = evaluate_paths(make_inputs(tmp_path))
    payload["panel_summaries"][0]["p95_seconds"] = 1.0

    with pytest.raises(qualification.QualificationError, match="binding_sha256 mismatch"):
        qualification.validate_manifest(payload)


def test_rebound_metric_forgery_is_rejected_by_internal_recalculation(tmp_path: Path) -> None:
    payload = evaluate_paths(make_inputs(tmp_path))
    payload["panel_summaries"][0]["p95_seconds"] = 1.0
    unsigned = dict(payload)
    unsigned.pop("binding_sha256")
    payload["binding_sha256"] = qualification._canonical_sha256(unsigned)

    with pytest.raises(qualification.QualificationError, match="fresh internal evaluation"):
        qualification.validate_manifest(payload)


def test_release_validation_requires_detached_signature(tmp_path: Path) -> None:
    paths = make_inputs(tmp_path)
    payload = evaluate_paths(paths)
    manifest = tmp_path / "qualification.json"
    write_json(manifest, payload)

    unsigned = qualification.validate_release(
        manifest_path=manifest,
        environment_json=paths[0],
        sla_ledger_json=paths[1],
        artifact_lock_json=paths[2],
        require_signed=False,
    )
    assert unsigned["release_qualified"] is False
    assert unsigned["signature_verified"] is False

    with pytest.raises(qualification.QualificationError, match="detached Cosign bundle"):
        qualification.validate_release(
            manifest_path=manifest,
            environment_json=paths[0],
            sla_ledger_json=paths[1],
            artifact_lock_json=paths[2],
            require_signed=True,
        )


def test_verified_signature_promotes_passing_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = make_inputs(tmp_path)
    payload = evaluate_paths(paths)
    manifest = tmp_path / "qualification.json"
    bundle = tmp_path / "qualification.bundle.json"
    write_json(manifest, payload)
    bundle.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(qualification, "verify_cosign_signature", lambda *args, **kwargs: None)

    result = qualification.validate_release(
        manifest_path=manifest,
        environment_json=paths[0],
        sla_ledger_json=paths[1],
        artifact_lock_json=paths[2],
        bundle_path=bundle,
        require_signed=True,
    )

    assert result["signature_verified"] is True
    assert result["release_qualified"] is True


def test_cli_validate_requires_signature_by_default(tmp_path: Path) -> None:
    paths = make_inputs(tmp_path)
    manifest = tmp_path / "qualification.json"
    write_json(manifest, evaluate_paths(paths))

    result = subprocess.run(
        [
            sys.executable,
            "scripts/qualification_manifest.py",
            "validate",
            "--manifest",
            str(manifest),
            "--environment-json",
            str(paths[0]),
            "--sla-ledger-json",
            str(paths[1]),
            "--artifact-lock-json",
            str(paths[2]),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "detached Cosign bundle" in result.stderr


def test_cli_removes_stale_output_on_malformed_input(tmp_path: Path) -> None:
    paths = make_inputs(tmp_path)
    paths[0].write_text("{}\n", encoding="utf-8")
    output = tmp_path / "qualification.json"
    output.write_text('{"stale": true}\n', encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/qualification_manifest.py",
            "evaluate",
            "--environment-json",
            str(paths[0]),
            "--sla-ledger-json",
            str(paths[1]),
            "--artifact-lock-json",
            str(paths[2]),
            "--out-json",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert not output.exists()
