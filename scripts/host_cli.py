#!/usr/bin/env python3
"""Beginner host command surface for the SkinScout Compose runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE = Path.home() / ".local/share/skinscout/state"
DEFAULT_DATA = Path.home() / ".local/share/skinscout/data"
DEFAULT_RESULTS = Path.home() / ".local/share/skinscout/results"
DEFAULT_COMPOSE = ROOT / "compose/skinscout.compose.yaml"
DEFAULT_IMAGE_POLICY = ROOT / "compose/image-policy.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _run(command: Sequence[str], *, dry_run: bool = False) -> int:
    if dry_run:
        print("[SkinScout][dry-run] " + shlex.join(list(command)))
        return 0
    return subprocess.run(list(command), cwd=ROOT, check=False).returncode


def _capture(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _valid_digest(reference: str) -> bool:
    if "@sha256:" not in reference:
        return False
    digest = reference.rsplit("@sha256:", 1)[1]
    return bool(SHA256_RE.fullmatch(digest)) and len(set(digest)) > 1


def _policy_args(image: Mapping[str, Any]) -> list[str] | None:
    identity = image.get("certificate_identity")
    issuer = image.get("issuer")
    public_key = image.get("public_key")
    public_key_sha256 = image.get("public_key_sha256")
    if identity and issuer:
        return [
            "--certificate-identity",
            str(identity),
            "--certificate-oidc-issuer",
            str(issuer),
        ]
    if public_key and public_key_sha256:
        key_path = Path(str(public_key))
        if not key_path.is_absolute():
            key_path = ROOT / key_path
        if not key_path.is_file():
            return None
        actual = hashlib.sha256(key_path.read_bytes()).hexdigest()
        if actual != str(public_key_sha256):
            return None
        return ["--key", str(key_path)]
    return None


def _load_image_policy(args: argparse.Namespace) -> tuple[dict[str, Any] | None, list[str]]:
    failures: list[str] = []
    try:
        policy = json.loads(args.image_policy.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"cannot read image signature policy: {exc}"]
    if not isinstance(policy, dict):
        return None, ["image signature policy must be a JSON object"]
    if policy.get("fail_closed") is not True or policy.get("verification") != "cosign":
        failures.append("image signature policy must be fail-closed Cosign")
    images = policy.get("images")
    if not isinstance(images, list) or not images:
        failures.append("image signature policy must list images")
    else:
        names: set[str] = set()
        for image in images:
            if not isinstance(image, dict):
                failures.append("image policy image entry must be an object")
                continue
            name = str(image.get("name", ""))
            if not name or name in names:
                failures.append(f"image policy has missing/duplicate name: {name or '<missing>'}")
            names.add(name)
            reference = str(image.get("reference", ""))
            if not _valid_digest(reference):
                failures.append(f"image policy reference is unqualified or placeholder: {name}")
            if _policy_args(image) is None:
                failures.append(f"image policy lacks usable pinned identity/key: {name}")
            if image.get("sbom_required") is not True or image.get("provenance_required") is not True:
                failures.append(f"image policy lacks SBOM/provenance requirement: {name}")
    return policy, failures


def _render_compose_config(args: argparse.Namespace) -> tuple[dict[str, Any] | None, list[str]]:
    command = _compose_command(args, "config", "--format", "json")
    result = _capture(command)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        return None, [f"cannot render Compose config: {detail}"]
    try:
        rendered = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return None, [f"rendered Compose config is not valid JSON: {exc}"]
    if not isinstance(rendered, dict):
        return None, ["rendered Compose config must be a JSON object"]
    return rendered, []


def _rendered_compose_failures(rendered: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    references: list[str] = []
    services = rendered.get("services")
    if not isinstance(services, Mapping) or not services:
        return references, ["rendered Compose config must declare services"]

    for service_name, raw_service in services.items():
        if not isinstance(raw_service, Mapping):
            failures.append(f"rendered Compose service is invalid: {service_name}")
            continue
        reference = raw_service.get("image")
        if not isinstance(reference, str) or not reference:
            failures.append(f"Compose service must use a released image reference: {service_name}")
        else:
            references.append(reference)

        ports = raw_service.get("ports", [])
        if not isinstance(ports, list):
            failures.append(f"rendered Compose ports must be a list: {service_name}")
        else:
            for port in ports:
                if isinstance(port, Mapping):
                    host_ip = port.get("host_ip")
                elif isinstance(port, str):
                    host_ip = "127.0.0.1" if port.startswith("127.0.0.1:") else None
                else:
                    host_ip = None
                if host_ip != "127.0.0.1":
                    failures.append(
                        f"Compose published port for {service_name} must bind exactly 127.0.0.1"
                    )

        volumes = raw_service.get("volumes", [])
        if isinstance(volumes, list):
            for volume in volumes:
                if isinstance(volume, Mapping):
                    values = (volume.get("source"), volume.get("target"))
                else:
                    values = (volume,)
                if any("docker.sock" in str(value) for value in values if value is not None):
                    failures.append("Compose file must not mount the Docker socket")
                    break
    return references, failures


def _verify_distribution_policy(args: argparse.Namespace, *, dry_run: bool | None = None) -> list[str]:
    failures: list[str] = []
    dry = args.dry_run if dry_run is None else dry_run
    policy, policy_failures = _load_image_policy(args)
    failures.extend(policy_failures)
    if policy is None or policy_failures:
        return failures

    try:
        compose_text = args.compose_file.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"cannot read Compose file: {exc}"]
    if "docker.sock" in compose_text:
        failures.append("Compose file must not mount the Docker socket")
        return failures

    rendered, render_failures = _render_compose_config(args)
    failures.extend(render_failures)
    if rendered is None:
        return failures
    compose_refs, compose_failures = _rendered_compose_failures(rendered)
    failures.extend(compose_failures)
    if not compose_refs:
        failures.append("Compose file must declare image references")
    if any(not _valid_digest(ref) for ref in compose_refs):
        failures.append("Compose images must be digest-pinned with non-placeholder SHA-256 digests")

    policy_images = policy.get("images", [])
    policy_refs = [str(image.get("reference", "")) for image in policy_images]
    if sorted(compose_refs) != sorted(policy_refs):
        failures.append("Compose image references must exactly match image policy references")
    if failures:
        return failures

    for image in policy_images:
        name = str(image["name"])
        reference = str(image["reference"])
        cosign_args = _policy_args(image)
        if cosign_args is None:
            failures.append(f"image policy lacks usable pinned identity/key: {name}")
            continue
        commands = [
            ["cosign", "verify", reference, *cosign_args],
            ["cosign", "verify-attestation", "--type", "spdxjson", reference, *cosign_args],
            ["cosign", "verify-attestation", "--type", "slsaprovenance", reference, *cosign_args],
        ]
        for command in commands:
            code = _run(command, dry_run=dry)
            if code != 0:
                failures.append(f"Cosign verification failed for {name}: {shlex.join(command)}")
                break
    return failures


def _fail_policy(failures: Sequence[str]) -> int:
    for failure in failures:
        print(f"[SkinScout][ERROR] {failure}", file=sys.stderr)
    return 2


def _gate_container_action(args: argparse.Namespace) -> int | None:
    failures = _verify_distribution_policy(args)
    if not failures:
        return None
    if not args.dry_run:
        _write_status(args, "unqualified")
    return _fail_policy(failures)


def _compose_command(args: argparse.Namespace, *extra: str) -> list[str]:
    return [
        "env",
        f"SKINSCOUT_HOST_DATA_DIR={args.data_dir.resolve()}",
        f"SKINSCOUT_HOST_RESULTS_DIR={args.results_dir.resolve()}",
        args.docker,
        "compose",
        "--project-name",
        "skinscout",
        "--file",
        str(args.compose_file),
        *extra,
    ]


def _ensure_host_dirs(args: argparse.Namespace) -> None:
    for directory in (args.state_dir, args.data_dir, args.results_dir):
        directory.mkdir(parents=True, exist_ok=True)


def _write_status(args: argparse.Namespace, status: str) -> None:
    args.state_dir.mkdir(parents=True, exist_ok=True)
    (args.state_dir / "host_runtime.json").write_text(
        json.dumps({"status": status, "compose_file": str(args.compose_file)}, indent=2)
        + "\n",
        encoding="utf-8",
    )


def cmd_start(args: argparse.Namespace) -> int:
    _ensure_host_dirs(args)
    blocked = _gate_container_action(args)
    if blocked is not None:
        return blocked
    code = _run(_compose_command(args, "up", "--detach"), dry_run=args.dry_run)
    if code == 0 and not args.dry_run:
        _write_status(args, "running")
    return code


def cmd_stop(args: argparse.Namespace) -> int:
    code = _run(_compose_command(args, "down"), dry_run=args.dry_run)
    if code == 0 and not args.dry_run:
        _write_status(args, "stopped")
    return code


def cmd_status(args: argparse.Namespace) -> int:
    if args.dry_run:
        failures: list[str] = []
        signature_state = "dry-run"
    else:
        failures = _verify_distribution_policy(args, dry_run=False)
        signature_state = "unverified" if failures else "verified"
    if failures:
        print(f"SkinScout runtime status: unqualified (signature: {signature_state})")
        for failure in failures:
            print(f"[SkinScout][ERROR] {failure}", file=sys.stderr)
        return 2
    print(f"SkinScout runtime status: signature={signature_state}")
    return _run(_compose_command(args, "ps"), dry_run=args.dry_run)


def cmd_update(args: argparse.Namespace) -> int:
    _ensure_host_dirs(args)
    blocked = _gate_container_action(args)
    if blocked is not None:
        return blocked
    for command in (
        _compose_command(args, "pull"),
        _compose_command(args, "up", "--detach"),
    ):
        code = _run(command, dry_run=args.dry_run)
        if code != 0:
            return code
    if not args.dry_run:
        _write_status(args, "running")
    return 0


def cmd_repair(args: argparse.Namespace) -> int:
    blocked = _gate_container_action(args)
    if blocked is not None:
        return blocked
    if args.manifest:
        command = [
            sys.executable,
            str(ROOT / "scripts/data_cas.py"),
            "--store",
            str(args.data_dir),
            "repair",
            str(args.manifest),
        ]
        code = _run(command, dry_run=args.dry_run)
        if code != 0:
            return code
    return _run(_compose_command(args, "up", "--detach", "--force-recreate"), dry_run=args.dry_run)


def cmd_rollback(args: argparse.Namespace) -> int:
    blocked = _gate_container_action(args)
    if blocked is not None:
        return blocked
    if args.data:
        code = _run(
            [
                sys.executable,
                str(ROOT / "scripts/data_cas.py"),
                "--store",
                str(args.data_dir),
                "rollback",
            ],
            dry_run=args.dry_run,
        )
        if code != 0:
            return code
    return _run(_compose_command(args, "up", "--detach"), dry_run=args.dry_run)


def cmd_run(args: argparse.Namespace) -> int:
    runtime_args = []
    for option in ("smiles", "sdf", "preset", "mode", "run_id"):
        value = getattr(args, option, None)
        if value:
            runtime_args.extend([f"--{option.replace('_', '-')}", str(value)])
    runtime_args.extend(args.runtime_args)
    if not runtime_args:
        print("[SkinScout][ERROR] run requires arguments for scripts/run_skinscout.py", file=sys.stderr)
        return 2
    blocked = _gate_container_action(args)
    if blocked is not None:
        return blocked
    command = _compose_command(
        args,
        "run",
        "--rm",
        "app",
        "python",
        "scripts/run_skinscout.py",
        *runtime_args,
    )
    return _run(command, dry_run=args.dry_run)


def _runtime_doctor_failures(args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    host_nvml = _capture([
        "nvidia-smi",
        "--query-gpu=driver_version",
        "--format=csv,noheader",
    ])
    host_output = host_nvml.stdout + host_nvml.stderr
    if (
        host_nvml.returncode != 0
        or not host_nvml.stdout.strip()
        or "Driver/library version mismatch" in host_output
    ):
        failures.append("host NVIDIA/NVML check failed")

    runtime = _capture([args.docker, "info", "--format", "{{json .Runtimes}}"])
    if runtime.returncode != 0:
        failures.append("Docker daemon/runtime check failed")
    else:
        try:
            runtimes = json.loads(runtime.stdout)
        except json.JSONDecodeError:
            runtimes = None
        if not isinstance(runtimes, Mapping) or "nvidia" not in runtimes:
            failures.append("Docker does not report the NVIDIA container runtime")

    container_nvml = _compose_command(
        args,
        "run",
        "--rm",
        "--no-deps",
        "target",
        "nvidia-smi",
        "--query-gpu=driver_version",
        "--format=csv,noheader",
    )
    if _run(container_nvml, dry_run=args.dry_run) != 0:
        failures.append("target container NVIDIA/NVML check failed")

    cuda_code = (
        "import torch; "
        "assert torch.cuda.is_available(), 'torch cuda unavailable'; "
        "x=torch.ones((256,256),device='cuda'); "
        "assert (x @ x).sum().item() > 0"
    )
    container_cuda = _compose_command(
        args,
        "run",
        "--rm",
        "--no-deps",
        "target",
        "python",
        "-c",
        cuda_code,
    )
    if _run(container_cuda, dry_run=args.dry_run) != 0:
        failures.append("target container CUDA tensor smoke test failed")
    return failures


def cmd_doctor(args: argparse.Namespace) -> int:
    if args.skip_gpu:
        return _fail_policy(["runtime doctor cannot skip GPU/NVML/CUDA checks"])
    blocked = _gate_container_action(args)
    if blocked is not None:
        return blocked
    failures: list[str] = []
    if args.dry_run:
        failures.append("runtime doctor cannot establish readiness in dry-run mode")
    else:
        failures.extend(_runtime_doctor_failures(args))
    for failure in failures:
        print(f"[SkinScout][ERROR] {failure}", file=sys.stderr)
    return 2 if failures else 0


def cmd_qualification(args: argparse.Namespace) -> int:
    if not args.qualification_args:
        print(
            "[SkinScout][ERROR] qualification requires evaluate or validate arguments",
            file=sys.stderr,
        )
        return 2
    return _run(
        [
            sys.executable,
            str(ROOT / "scripts/qualification_manifest.py"),
            *args.qualification_args,
        ],
        dry_run=args.dry_run,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compose-file", type=Path, default=DEFAULT_COMPOSE)
    parser.add_argument("--image-policy", type=Path, default=DEFAULT_IMAGE_POLICY)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--docker", default=os.environ.get("SKINSCOUT_DOCKER", "docker"))
    parser.add_argument("--dry-run", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("start").set_defaults(func=cmd_start)
    sub.add_parser("stop").set_defaults(func=cmd_stop)
    sub.add_parser("status").set_defaults(func=cmd_status)
    sub.add_parser("update").set_defaults(func=cmd_update)
    repair = sub.add_parser("repair")
    repair.add_argument("--manifest", type=Path)
    repair.set_defaults(func=cmd_repair)
    rollback_cmd = sub.add_parser("rollback")
    rollback_cmd.add_argument("--data", action="store_true")
    rollback_cmd.set_defaults(func=cmd_rollback)
    run_cmd = sub.add_parser("run")
    run_cmd.add_argument("--smiles")
    run_cmd.add_argument("--sdf")
    run_cmd.add_argument("--preset")
    run_cmd.add_argument("--mode")
    run_cmd.add_argument("--run-id")
    run_cmd.add_argument("runtime_args", nargs=argparse.REMAINDER)
    run_cmd.set_defaults(func=cmd_run)
    doctor = sub.add_parser("doctor")
    doctor.add_argument("--skip-gpu", action="store_true", help=argparse.SUPPRESS)
    doctor.set_defaults(func=cmd_doctor)
    qualification = sub.add_parser(
        "qualification",
        help="Build or validate signed Release 1 qualification evidence.",
    )
    qualification.add_argument("qualification_args", nargs=argparse.REMAINDER)
    qualification.set_defaults(func=cmd_qualification)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
