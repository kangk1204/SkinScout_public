"""F35: qualification outputs must be named/hashed and diagnostic rc must not qualify."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import host_cli  # noqa: E402
import qualification_manifest as qualification  # noqa: E402
from scripts.tests.test_host_cli import (  # noqa: E402
    _completed,
    _qualified_args,
    _rendered_config,
)
from scripts.tests.test_qualification_manifest import (  # noqa: E402
    environment,
    evaluate_paths,
    make_inputs,
    write_json,
)


STRUCTURAL_SCHEMA = "skinscout.structural-qualification.v1"
REPORT_SCHEMA = "skinscout.report-qualification.v1"


def load_install_runtime_module():
    spec = importlib.util.spec_from_file_location(
        "install_runtime_qualification_outputs",
        ROOT / "scripts/install_runtime.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_qualification_status(
    root: Path,
    installer,
    schema: str,
    *,
    mutate=None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    required = sorted(installer.QUALIFICATION_REQUIRED_OUTPUTS[schema])
    outputs: dict[str, dict[str, object]] = {}
    for name in required:
        artifact = root / "run" / f"{name}.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(json.dumps({"name": name}) + "\n", encoding="utf-8")
        outputs[name] = {
            "path": str(artifact.relative_to(root)),
            "bytes": artifact.stat().st_size,
            "sha256": installer.sha256(artifact),
        }
    if mutate is not None:
        mutate(required, outputs)
    status = root / "qualification_status.json"
    status.write_text(
        json.dumps(
            {
                "schema_version": schema,
                "status": "passed",
                "outputs": outputs,
            }
        ),
        encoding="utf-8",
    )
    return status


def test_empty_outputs_do_not_qualify(tmp_path: Path) -> None:
    installer = load_install_runtime_module()
    status = tmp_path / "status.json"
    status.write_text(
        json.dumps(
            {
                "schema_version": STRUCTURAL_SCHEMA,
                "status": "passed",
                "outputs": {},
            }
        ),
        encoding="utf-8",
    )

    assert not installer._qualification_payload_valid(status, STRUCTURAL_SCHEMA)


def test_missing_required_output_does_not_qualify(tmp_path: Path) -> None:
    installer = load_install_runtime_module()

    def drop_one(required, outputs):
        outputs.pop(required[0])

    status = _write_qualification_status(
        tmp_path / "structural",
        installer,
        STRUCTURAL_SCHEMA,
        mutate=drop_one,
    )

    assert not installer._qualification_payload_valid(status, STRUCTURAL_SCHEMA)


def test_named_bytes_and_hash_outputs_qualify(tmp_path: Path) -> None:
    installer = load_install_runtime_module()
    status = _write_qualification_status(
        tmp_path / "structural",
        installer,
        STRUCTURAL_SCHEMA,
    )

    assert installer._qualification_payload_valid(status, STRUCTURAL_SCHEMA)


def test_report_schema_requires_report_outputs(tmp_path: Path) -> None:
    installer = load_install_runtime_module()
    status = _write_qualification_status(
        tmp_path / "structural",
        installer,
        REPORT_SCHEMA,
    )

    assert installer._qualification_payload_valid(status, REPORT_SCHEMA)

    payload = json.loads(status.read_text(encoding="utf-8"))
    payload["outputs"].pop("fast_manifest")
    status.write_text(json.dumps(payload), encoding="utf-8")

    assert not installer._qualification_payload_valid(status, REPORT_SCHEMA)


def test_bytes_mismatch_does_not_qualify(tmp_path: Path) -> None:
    installer = load_install_runtime_module()

    def inflate_bytes(required, outputs):
        outputs[required[0]]["bytes"] = int(outputs[required[0]]["bytes"]) + 1

    status = _write_qualification_status(
        tmp_path / "structural",
        installer,
        STRUCTURAL_SCHEMA,
        mutate=inflate_bytes,
    )

    assert not installer._qualification_payload_valid(status, STRUCTURAL_SCHEMA)


def test_unknown_schema_does_not_qualify(tmp_path: Path) -> None:
    installer = load_install_runtime_module()
    status = tmp_path / "status.json"
    status.write_text(
        json.dumps(
            {
                "schema_version": "qualification.v1",
                "status": "passed",
                "outputs": {"artifact": {"path": "artifact.txt"}},
            }
        ),
        encoding="utf-8",
    )

    assert not installer._qualification_payload_valid(status, "qualification.v1")


def test_validate_release_labels_diagnostic_unverified(tmp_path: Path) -> None:
    paths = make_inputs(tmp_path)
    manifest = tmp_path / "qualification.json"
    write_json(manifest, evaluate_paths(paths))

    result = qualification.validate_release(
        manifest_path=manifest,
        environment_json=paths[0],
        sla_ledger_json=paths[1],
        artifact_lock_json=paths[2],
        require_signed=False,
    )

    assert result["validated"] is True
    assert result["diagnostic"] is True
    assert result["qualified"] is False
    assert result["release_qualified"] is False
    assert result["signature_verified"] is False
    assert result["signature_status"] == "unverified"


def test_cli_unsigned_passing_evidence_is_diagnostic_not_qualified(
    tmp_path: Path,
) -> None:
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
            "--allow-unsigned-evidence",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 3, result.stderr
    payload = json.loads(result.stdout)
    assert payload["diagnostic"] is True
    assert payload["qualified"] is False
    assert payload["release_qualified"] is False
    assert payload["signature_status"] == "unverified"


def test_cli_failed_diagnostic_cannot_be_read_as_qualified(tmp_path: Path) -> None:
    failing_env = environment(
        os={
            "id": "ubuntu",
            "version_id": "26.04",
            "pretty_name": "Ubuntu 26.04 LTS",
        }
    )
    paths = make_inputs(tmp_path, environment_payload=failing_env)
    payload = evaluate_paths(paths)
    assert payload["qualification_evidence_status"] == "failed"
    manifest = tmp_path / "qualification.json"
    write_json(manifest, payload)

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
            "--allow-unsigned-evidence",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2, result.stderr
    parsed = json.loads(result.stdout)
    assert parsed["diagnostic"] is True
    assert parsed["qualified"] is False
    assert parsed["release_qualified"] is False


def test_verified_signature_is_the_only_qualified_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = make_inputs(tmp_path)
    manifest = tmp_path / "qualification.json"
    bundle = tmp_path / "qualification.bundle.json"
    write_json(manifest, evaluate_paths(paths))
    bundle.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        qualification,
        "verify_cosign_signature",
        lambda *args, **kwargs: None,
    )

    result = qualification.validate_release(
        manifest_path=manifest,
        environment_json=paths[0],
        sla_ledger_json=paths[1],
        artifact_lock_json=paths[2],
        bundle_path=bundle,
        require_signed=True,
    )

    assert result["validated"] is True
    assert result["diagnostic"] is False
    assert result["qualified"] is True
    assert result["release_qualified"] is True
    assert result["signature_status"] == "verified"


def _status_args(tmp_path: Path, *argv: str):
    return _qualified_args(tmp_path, "status", *argv)


def test_status_runs_cosign_and_reports_verified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args, references = _status_args(tmp_path)
    events: list[tuple[list[str], bool]] = []

    def fake_run(command, dry_run=False):
        events.append((list(command), dry_run))
        return 0

    def fake_capture(command):
        call = list(command)
        if call[-3:] == ["config", "--format", "json"]:
            return _completed(call, stdout=json.dumps(_rendered_config(references)))
        raise AssertionError(f"unexpected captured command: {call}")

    monkeypatch.setattr(host_cli, "_run", fake_run)
    monkeypatch.setattr(host_cli, "_capture", fake_capture)

    assert host_cli.cmd_status(args) == 0
    assert "signature=verified" in capsys.readouterr().out
    cosign_events = [
        (call, dry) for call, dry in events if call[:2] == ["cosign", "verify"]
    ]
    assert len(cosign_events) == 2
    assert all(dry is False for _call, dry in cosign_events)


def test_status_dry_run_reports_dry_run_signature_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args, _references = _status_args(tmp_path)
    args.dry_run = True
    events: list[tuple[list[str], bool]] = []

    def fake_run(command, dry_run=False):
        events.append((list(command), dry_run))
        return 0

    monkeypatch.setattr(host_cli, "_run", fake_run)
    monkeypatch.setattr(
        host_cli,
        "_capture",
        lambda command: (_ for _ in ()).throw(
            AssertionError("dry-run status must not render Compose")
        ),
    )

    assert host_cli.cmd_status(args) == 0
    assert "signature=dry-run" in capsys.readouterr().out
    assert all(dry is True for _call, dry in events)


def test_status_cosign_failure_reports_unverified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args, references = _status_args(tmp_path)

    def fake_run(command, dry_run=False):
        return 1 if list(command)[:2] == ["cosign", "verify"] else 0

    def fake_capture(command):
        call = list(command)
        if call[-3:] == ["config", "--format", "json"]:
            return _completed(call, stdout=json.dumps(_rendered_config(references)))
        raise AssertionError(f"unexpected captured command: {call}")

    monkeypatch.setattr(host_cli, "_run", fake_run)
    monkeypatch.setattr(host_cli, "_capture", fake_capture)

    assert host_cli.cmd_status(args) == 2
    captured = capsys.readouterr()
    assert "unqualified" in captured.out
    assert "signature: unverified" in captured.out
