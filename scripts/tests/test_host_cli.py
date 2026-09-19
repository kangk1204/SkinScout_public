"""Tests for the beginner host CLI distribution surface."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from scripts import host_cli


def _args(tmp_path: Path, *argv: str):
    return host_cli.build_parser().parse_args(
        [
            "--compose-file",
            str(Path("compose/skinscout.compose.yaml")),
            "--image-policy",
            str(Path("compose/image-policy.json")),
            "--state-dir",
            str(tmp_path / "state"),
            "--data-dir",
            str(tmp_path / "data"),
            "--results-dir",
            str(tmp_path / "results"),
            "--docker",
            "docker",
            *argv,
        ]
    )


def _qualified_distribution(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    references = {
        "app": f"ghcr.io/acme/app@sha256:{hashlib.sha256(b'app').hexdigest()}",
        "target": f"ghcr.io/acme/target@sha256:{hashlib.sha256(b'target').hexdigest()}",
    }
    compose = tmp_path / "compose.yaml"
    compose.write_text(
        "\n".join([
            "services:",
            "  app:",
            f"    image: {references['app']}",
            "    ports:",
            '      - "127.0.0.1:8765:8765"',
            "  target:",
            f"    image: {references['target']}",
        ]),
        encoding="utf-8",
    )
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps({
            "fail_closed": True,
            "verification": "cosign",
            "images": [
                {
                    "name": name,
                    "reference": reference,
                    "certificate_identity": "<개인 주소>",
                    "issuer": "https://token.actions.githubusercontent.com",
                    "sbom_required": True,
                    "provenance_required": True,
                }
                for name, reference in references.items()
            ],
        }),
        encoding="utf-8",
    )
    return compose, policy, references


def _qualified_args(tmp_path: Path, *argv: str):
    compose, policy, references = _qualified_distribution(tmp_path)
    args = host_cli.build_parser().parse_args([
        "--compose-file",
        str(compose),
        "--image-policy",
        str(policy),
        "--state-dir",
        str(tmp_path / "state"),
        "--data-dir",
        str(tmp_path / "data"),
        "--results-dir",
        str(tmp_path / "results"),
        "--docker",
        "docker",
        *argv,
    ])
    return args, references


def _rendered_config(references: dict[str, str], port: object | None = None) -> dict[str, object]:
    app: dict[str, object] = {"image": references["app"]}
    if port is not None:
        app["ports"] = [port]
    return {
        "services": {
            "app": app,
            "target": {"image": references["target"]},
        }
    }


def _completed(
    command: list[str],
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, returncode, stdout, stderr)


def test_host_cli_exposes_beginner_commands() -> None:
    parser = host_cli.build_parser()
    subparsers = [
        action for action in parser._actions if getattr(action, "choices", None)
    ][0]
    assert set(subparsers.choices) == {
        "start",
        "stop",
        "status",
        "doctor",
        "update",
        "repair",
        "rollback",
        "run",
        "qualification",
    }


def test_qualification_dispatches_to_release_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(
        host_cli,
        "_run",
        lambda command, dry_run=False: calls.append(list(command)) or 0,
    )
    args = _args(
        tmp_path,
        "qualification",
        "evaluate",
        "--environment-json",
        "environment.json",
    )

    assert host_cli.cmd_qualification(args) == 0
    assert calls == [[
        host_cli.sys.executable,
        str(host_cli.ROOT / "scripts/qualification_manifest.py"),
        "evaluate",
        "--environment-json",
        "environment.json",
    ]]


def test_start_uses_loopback_compose_without_docker_socket(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(host_cli, "_run", lambda command, dry_run=False: calls.append(list(command)) or 0)
    monkeypatch.setattr(host_cli, "_verify_distribution_policy", lambda args: [])
    args = _args(tmp_path, "start")

    assert host_cli.cmd_start(args) == 0
    assert calls == [
        [
            "env",
            f"SKINSCOUT_HOST_DATA_DIR={(tmp_path / 'data').resolve()}",
            f"SKINSCOUT_HOST_RESULTS_DIR={(tmp_path / 'results').resolve()}",
            "docker",
            "compose",
            "--project-name",
            "skinscout",
            "--file",
            "compose/skinscout.compose.yaml",
            "up",
            "--detach",
        ]
    ]
    status = json.loads((tmp_path / "state/host_runtime.json").read_text())
    assert status["status"] == "running"
    compose = Path("compose/skinscout.compose.yaml").read_text(encoding="utf-8")
    policy = json.loads(Path("compose/image-policy.json").read_text(encoding="utf-8"))
    assert "127.0.0.1:8765:8765" in compose
    assert "SKINSCOUT_HOST_DATA_DIR" in compose
    assert "SKINSCOUT_HOST_RESULTS_DIR" in compose
    assert "docker.sock" not in compose
    assert policy["fail_closed"] is True
    assert policy["verification"] == "cosign"
    assert all("@sha256:" in image["reference"] for image in policy["images"])


def test_run_dispatches_to_container_runner(tmp_path: Path, monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(host_cli, "_run", lambda command, dry_run=False: calls.append(list(command)) or 0)
    monkeypatch.setattr(host_cli, "_verify_distribution_policy", lambda args: [])
    args = _args(tmp_path, "run", "--smiles", "CCO", "--preset", "target-id")

    assert host_cli.cmd_run(args) == 0
    assert calls[0][-6:] == [
        "python",
        "scripts/run_skinscout.py",
        "--smiles",
        "CCO",
        "--preset",
        "target-id",
    ]


def test_doctor_fails_closed_on_compose_policy_violation(tmp_path: Path, monkeypatch) -> None:
    bad_compose = tmp_path / "compose.yaml"
    bad_compose.write_text(
        "services:\n  app:\n    image: skinscout:latest\n    volumes:\n      - /var/run/docker.sock:/var/run/docker.sock\n",
        encoding="utf-8",
    )
    calls: list[list[str]] = []
    monkeypatch.setattr(host_cli, "_run", lambda command, dry_run=False: calls.append(list(command)) or 0)
    args = host_cli.build_parser().parse_args(
        [
            "--compose-file",
            str(bad_compose),
            "--image-policy",
            str(Path("compose/image-policy.json")),
            "--state-dir",
            str(tmp_path / "state"),
            "--data-dir",
            str(tmp_path / "data"),
            "--results-dir",
            str(tmp_path / "results"),
            "doctor",
        ]
    )

    assert host_cli.cmd_doctor(args) == 2


def test_default_compose_policy_is_unqualified_and_blocks_start(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(host_cli, "_run", lambda command, dry_run=False: calls.append(list(command)) or 0)
    args = _args(tmp_path, "start")

    assert host_cli.cmd_start(args) == 2
    status = json.loads((tmp_path / "state/host_runtime.json").read_text())
    assert status["status"] == "unqualified"
    assert calls == []


@pytest.mark.parametrize(
    "argv",
    [
        ("start",),
        ("update",),
        ("repair", "--manifest", "untrusted.json"),
        ("rollback", "--data"),
        ("run", "--smiles", "CCO", "--preset", "target-id"),
        ("doctor",),
    ],
)
def test_every_container_creating_command_is_blocked_before_side_effects(
    tmp_path: Path,
    monkeypatch,
    argv: tuple[str, ...],
) -> None:
    args = _args(tmp_path, *argv)
    monkeypatch.setattr(
        host_cli,
        "_verify_distribution_policy",
        lambda _args: ["distribution is unqualified"],
    )

    def forbidden_run(*_args, **_kwargs):
        raise AssertionError("no Compose or CAS subprocess may run before qualification")

    monkeypatch.setattr(host_cli, "_run", forbidden_run)

    assert args.func(args) == 2
    status = json.loads((tmp_path / "state/host_runtime.json").read_text())
    assert status["status"] == "unqualified"


@pytest.mark.parametrize(
    "unsafe_port",
    [
        {"target": 8765, "published": "8765", "host_ip": "0.0.0.0", "protocol": "tcp"},
        {"target": 8765, "published": "8765", "protocol": "tcp"},
        "8765:8765",
    ],
)
def test_rendered_compose_rejects_public_or_bare_published_ports(
    tmp_path: Path,
    monkeypatch,
    unsafe_port: object,
) -> None:
    args, references = _qualified_args(tmp_path, "start")
    capture_calls: list[list[str]] = []

    def fake_capture(command):
        capture_calls.append(list(command))
        return _completed(list(command), stdout=json.dumps(_rendered_config(references, unsafe_port)))

    monkeypatch.setattr(host_cli, "_capture", fake_capture)
    monkeypatch.setattr(
        host_cli,
        "_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Cosign must not run for an unsafe rendered port")
        ),
    )

    failures = host_cli._verify_distribution_policy(args)

    assert any("must bind exactly 127.0.0.1" in failure for failure in failures)
    assert capture_calls[0][-3:] == ["config", "--format", "json"]


def test_status_reports_unqualified_without_docker_for_placeholder_policy(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(host_cli, "_run", lambda command, dry_run=False: calls.append(list(command)) or 0)
    args = _args(tmp_path, "status")

    assert host_cli.cmd_status(args) == 2
    assert "unqualified" in capsys.readouterr().out
    assert calls == []


def test_doctor_verifies_images_sbom_and_provenance_before_ready(
    tmp_path: Path,
    monkeypatch,
) -> None:
    args, references = _qualified_args(tmp_path, "doctor")
    events: list[list[str]] = []

    def fake_run(command, dry_run=False):
        events.append(list(command))
        return 0

    def fake_capture(command):
        call = list(command)
        events.append(call)
        if call[-3:] == ["config", "--format", "json"]:
            rendered = _rendered_config(
                references,
                {"target": 8765, "published": "8765", "host_ip": "127.0.0.1"},
            )
            return _completed(call, stdout=json.dumps(rendered))
        if call[0] == "nvidia-smi":
            return _completed(call, stdout="550.54.14\n")
        if call[:2] == ["docker", "info"]:
            return _completed(call, stdout=json.dumps({"runc": {}, "nvidia": {}}))
        raise AssertionError(f"unexpected captured command: {call}")

    monkeypatch.setattr(host_cli, "_run", fake_run)
    monkeypatch.setattr(host_cli, "_capture", fake_capture)

    assert host_cli.cmd_doctor(args) == 0
    cosign_calls = [call for call in events if call[:2] == ["cosign", "verify"]]
    attestation_calls = [
        call for call in events if call[:2] == ["cosign", "verify-attestation"]
    ]
    container_calls = [
        call
        for call in events
        if "run" in call and "docker" in call and "compose" in call
    ]
    assert len(cosign_calls) == 2
    assert len(attestation_calls) == 4
    assert any("--type" in call and "spdxjson" in call for call in attestation_calls)
    assert any("--type" in call and "slsaprovenance" in call for call in attestation_calls)
    assert len(container_calls) == 2
    assert any("nvidia-smi" in call for call in container_calls)
    assert any("torch.cuda.is_available" in " ".join(call) for call in container_calls)
    assert not any("pipeline_readiness.py" in " ".join(call) for call in events)
    first_container = min(events.index(call) for call in container_calls)
    assert all(events.index(call) < first_container for call in cosign_calls + attestation_calls)


def test_doctor_rejects_gpu_skip_without_running_checks(tmp_path: Path, monkeypatch, capsys) -> None:
    args = _args(tmp_path, "doctor", "--skip-gpu")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("skip rejection must not be converted into mocked readiness")

    monkeypatch.setattr(host_cli, "_run", forbidden)
    monkeypatch.setattr(host_cli, "_capture", forbidden)

    assert host_cli.cmd_doctor(args) == 2
    assert "cannot skip GPU/NVML/CUDA checks" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("failure_mode", "expected_error"),
    [
        ("host_nvml", "host NVIDIA/NVML check failed"),
        ("docker_runtime", "Docker does not report the NVIDIA container runtime"),
        ("container_nvml", "target container NVIDIA/NVML check failed"),
        ("container_cuda", "target container CUDA tensor smoke test failed"),
    ],
)
def test_doctor_fails_when_any_real_runtime_probe_fails(
    tmp_path: Path,
    monkeypatch,
    capsys,
    failure_mode: str,
    expected_error: str,
) -> None:
    args = _args(tmp_path, "doctor")
    monkeypatch.setattr(host_cli, "_verify_distribution_policy", lambda _args: [])

    def fake_capture(command):
        call = list(command)
        if call[0] == "nvidia-smi":
            return _completed(
                call,
                returncode=1 if failure_mode == "host_nvml" else 0,
                stdout="" if failure_mode == "host_nvml" else "550.54.14\n",
            )
        if call[:2] == ["docker", "info"]:
            runtimes = {"runc": {}}
            if failure_mode != "docker_runtime":
                runtimes["nvidia"] = {}
            return _completed(call, stdout=json.dumps(runtimes))
        raise AssertionError(f"unexpected captured command: {call}")

    container_probe = 0

    def fake_run(command, dry_run=False):
        nonlocal container_probe
        call = list(command)
        assert "run" in call
        container_probe += 1
        if failure_mode == "container_nvml" and container_probe == 1:
            return 1
        if failure_mode == "container_cuda" and container_probe == 2:
            return 1
        return 0

    monkeypatch.setattr(host_cli, "_capture", fake_capture)
    monkeypatch.setattr(host_cli, "_run", fake_run)

    assert host_cli.cmd_doctor(args) == 2
    assert expected_error in capsys.readouterr().err
