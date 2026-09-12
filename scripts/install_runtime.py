#!/usr/bin/env python3
"""Idempotent installer runtime for the public SkinScout profiles."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
INSTALLER_VERSION = "2026.08.23.2"
STATE_SCHEMA = "skinscout.install.v2"
PHASES = [
    "host_preflight",
    "host_packages",
    "source_checkout",
    "base_runtime",
    "profile_envs",
    "pinned_tools",
    "fetch_analog_bundle",
    "stage0_sources",
    "stage0_build",
    "stage0_verify",
    "target_smoke",
    "report_smoke",
    "launcher_ready",
]

PUBLIC_PROFILES = {"analog", "demo", "full"}

# `analog` 는 대체소재 검색만 쓰는 설치다. 공동연구자의 용도가 그것 하나일 때
# 전체 Stage 0 을 만들게 할 이유가 없다 - 실측 125 GB(ChEMBL 두 판본 64 GB,
# BindingDB 20 GB, AlphaFold 13 GB, 포켓·PDBQT 17 GB …)를 받아서 그중 대체소재
# 화면이 여는 것은 13 MB 짜리 원료 표 하나다. 나머지 둘(활성 인덱스·ADMET 캐시)은
# 애초에 Stage 0 이 만들지도 않는다.
#
# 그래서 이 프로파일은 무거운 단계를 건너뛰고 대신 만들어진 번들을 받는다.
# 표적 예측·도킹·MD 는 이 설치로 돌지 않는다 - 그 용도면 `--profile full` 이다.
ANALOG_SKIP_PHASES = (
    "stage0_sources",
    "stage0_build",
    "stage0_verify",
    "target_smoke",
    "report_smoke",
)
PROFILE_UPGRADE_PHASES = (
    "profile_envs",
    "target_smoke",
    "report_smoke",
    "launcher_ready",
)
INSTALLER_UPGRADE_PHASES = (
    "base_runtime",
    "profile_envs",
    "pinned_tools",
    "target_smoke",
    "report_smoke",
    "launcher_ready",
)
REVALIDATE_PHASES = {
    "host_preflight",
    "host_packages",
    "source_checkout",
    "base_runtime",
    "profile_envs",
    "pinned_tools",
    "stage0_sources",
    "stage0_verify",
    "launcher_ready",
}
FULL_MODEL_REQUIREMENTS = ("boltz", "rtmscore", "psichic", "diffdock")

ARTIFACTS: dict[str, dict[str, str]] = {
    "micromamba": {
        "version": "2.8.1-0",
        "url": "https://github.com/mamba-org/micromamba-releases/releases/download/2.8.1-0/micromamba-linux-64",
        "sha256": "9689782d863c05a1bf5d2d371ba527104e7a4eb4310c1637d8653b751aed9c82",
    },
    "gnina": {
        "version": "v1.3.3",
        "url": "https://github.com/gnina/gnina/releases/download/v1.3.3/gnina.cuda12.8.static",
        "sha256": "3340c1f49cd3c7c84d8699182a1c6af13c7fa2a22448d1204640446106f72172",
        "destination": "tools/gnina",
    },
    "autodock_gpu_ocl": {
        "version": "v1.6",
        "url": "https://github.com/ccsb-scripps/AutoDock-GPU/releases/download/v1.6/adgpu-v1.6_linux_x64_ocl_128wi",
        "sha256": "8a22804c1fde62a59c030a45e8a9d9d960ff329d22e4927daf7464df393b8047",
        "destination": "tools/AutoDock-GPU/bin/autodock_gpu_128wi",
    },
    "autogrid": {
        "commit": "6d2847beaeac8ff43ca99094707fd74e3ca1ff37",
        "url": "https://github.com/ccsb-scripps/AutoGrid/archive/6d2847beaeac8ff43ca99094707fd74e3ca1ff37.tar.gz",
        "sha256": "4d0bd83a446fd81577f4fc492299e22f131245589e1782e0532aecf3435e772a",
        "destination": "tools/AutoGrid",
    },
    "autodock4_parameters": {
        "sha256": "912b557d6249c8ef74d4bca62acbb82dc456a4c92c07c6fd0cec37530dcc8307",
        "destination": "tools/AutoDock-GPU/AD4_parameters.dat",
    },
    "autodock4_bound_parameters": {
        "url": "https://raw.githubusercontent.com/ccsb-scripps/AutoGrid/6d2847beaeac8ff43ca99094707fd74e3ca1ff37/ad4_shared/AD4.1_bound.dat",
        "sha256": "91a720ec15959727d6a5386bc19d46350f1d4683758900d88109101fccabb869",
        "destination": "tools/AutoGrid/AD4.1_bound.dat",
    },
    "meeko_conda": {
        "version": "0.7.1",
        "build": "pyhd8ed1ab_1",
        "url": "https://conda.anaconda.org/conda-forge/noarch/meeko-0.7.1-pyhd8ed1ab_1.conda",
        "sha256": "f9e1f221be6305c9b3214c832e450c303ff291e88dd6ffc966ff4ef77f468e5d",
    },
    "p2rank": {
        "version": "2.5",
        "url": "https://github.com/rdk/p2rank/releases/download/2.5/p2rank_2.5.tar.gz",
        "sha256": "9c21755967450300f2eb052d059ef128d38021626e52e135561c7c4687891751",
        "destination": "tools/p2rank_2.5/prank",
    },
}


class InstallError(RuntimeError):
    """Installer error with a process exit code."""

    def __init__(self, message: str, code: int = 2) -> None:
        super().__init__(message)
        self.code = code


def log(message: str) -> None:
    print(f"[SkinScout] {message}", flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], *, dry_run: bool, env: dict[str, str] | None = None) -> None:
    if dry_run:
        print("[SkinScout][dry-run] " + " ".join(shlex_quote(part) for part in command))
        return
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def shlex_quote(value: str) -> str:
    import shlex

    return shlex.quote(value)


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        tmp_name = handle.name
    os.replace(tmp_name, path)


@contextlib.contextmanager
def installer_lock(state_dir: Path, *, dry_run: bool) -> Iterable[None]:
    if dry_run:
        yield
        return
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = state_dir / "install.lock"
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        yield


def initial_state(args: argparse.Namespace, install_id: str) -> dict[str, Any]:
    return {
        "schema": STATE_SCHEMA,
        "installer_version": INSTALLER_VERSION,
        "install_id": install_id,
        "root": str(ROOT),
        "profile": args.profile,
        "original_invocation": args.original_invocation,
        "last_invocation": args.original_invocation,
        "profile_history": [args.profile],
        "phases": {phase: {"status": "pending"} for phase in PHASES},
    }


def prepare_state(args: argparse.Namespace, state_path: Path) -> dict[str, Any]:
    install_id = f"{int(time.time())}-{os.getpid()}"
    state = load_json(state_path) or initial_state(args, install_id)
    if state.get("schema") != STATE_SCHEMA:
        raise InstallError(
            f"installer state schema mismatch at {state_path}: {state.get('schema')}"
        )
    existing_version = state.get("installer_version")
    if existing_version != INSTALLER_VERSION:
        state["installer_version"] = INSTALLER_VERSION
        for phase in INSTALLER_UPGRADE_PHASES:
            state.setdefault("phases", {})[phase] = {"status": "pending"}
        log(
            "설치기 버전이 변경되어 도구와 자격시험을 다시 검증합니다: "
            f"{existing_version!r} -> {INSTALLER_VERSION}"
        )
    existing_profile = state.get("profile")
    requested_profile = args.profile
    if existing_profile == "demo" and requested_profile == "full":
        state["profile"] = "full"
        history = state.setdefault("profile_history", ["demo"])
        if isinstance(history, list) and "full" not in history:
            history.append("full")
        for phase in PROFILE_UPGRADE_PHASES:
            state.setdefault("phases", {})[phase] = {"status": "pending"}
        log("기존 Demo 설치를 Full 프로필로 업그레이드합니다.")
    elif existing_profile == "full" and requested_profile == "demo":
        args.profile = "full"
        log("기존 Full 설치가 Demo를 포함하므로 Full 상태를 유지합니다.")
    elif existing_profile != requested_profile:
        raise InstallError(
            "existing install profile is incompatible with this invocation: "
            f"existing={existing_profile!r} requested={requested_profile!r}"
        )
    state["installer_version"] = INSTALLER_VERSION
    state.setdefault("install_id", install_id)
    state.setdefault("original_invocation", args.original_invocation)
    state["last_invocation"] = args.original_invocation
    state.setdefault("profile_history", [state.get("profile")])
    state.setdefault("phases", {})
    for phase in PHASES:
        state["phases"].setdefault(phase, {"status": "pending"})
    return state


def configure_mamba_root() -> Path:
    configured = os.environ.get("MAMBA_ROOT_PREFIX", "").strip()
    if configured:
        root = Path(configured).expanduser()
    else:
        candidates = (
            Path.home() / ".local/share/micromamba",
            Path.home() / ".local/share/mamba",
        )
        root = next(
            (
                candidate
                for candidate in candidates
                if (candidate / "envs/cosmax-base").is_dir()
            ),
            candidates[0],
        )
    root = root.resolve()
    os.environ["MAMBA_ROOT_PREFIX"] = str(root)
    log(f"micromamba root: {root}")
    return root


def environment_prefix(env_name: str) -> Path:
    root = Path(
        os.environ.get(
            "MAMBA_ROOT_PREFIX", Path.home() / ".local/share/micromamba"
        )
    ).expanduser()
    return root / "envs" / env_name


def mark_phase(
    state: dict[str, Any],
    state_path: Path,
    phase: str,
    status: str,
    *,
    dry_run: bool,
    detail: str | None = None,
) -> None:
    state["phases"][phase] = {
        "status": status,
        "updated_at": int(time.time()),
    }
    if detail:
        state["phases"][phase]["detail"] = detail
    if dry_run:
        return
    write_json_atomic(state_path, state)


def phase_complete(state: dict[str, Any], phase: str) -> bool:
    return state["phases"].get(phase, {}).get("status") == "complete"


def require_host_preflight() -> None:
    os_release = Path("/etc/os-release")
    if not os_release.exists():
        raise InstallError("Ubuntu 24.04 x86_64 is required")
    values: dict[str, str] = {}
    for line in os_release.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value.strip().strip('"')
    if values.get("ID") != "ubuntu" or values.get("VERSION_ID") != "24.04":
        raise InstallError("Ubuntu 24.04 x86_64 is required")
    if platform.machine() != "x86_64":
        raise InstallError("Ubuntu 24.04 x86_64 is required")
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        raise InstallError("working nvidia-smi is required; installer will not install or change NVIDIA drivers")
    smi = subprocess.run(
        [
            nvidia_smi,
            "--query-gpu=memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if smi.returncode != 0:
        raise InstallError("nvidia-smi failed; installer will not install or change NVIDIA drivers")
    total_vram = max((int(line.strip()) for line in smi.stdout.splitlines() if line.strip()), default=0)
    if total_vram < 12 * 1024:
        raise InstallError(">=12GB GPU VRAM is required")
    meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
    total_ram_kib = 0
    for line in meminfo.splitlines():
        if line.startswith("MemTotal:"):
            total_ram_kib = int(line.split()[1])
            break
    if total_ram_kib < 32 * 1024 * 1024:
        raise InstallError(">=32GB system RAM is required")
    free_gib = shutil.disk_usage(Path.home()).free // (1024**3)
    if free_gib < 300:
        raise InstallError(">=300GB free disk is required; 500GB is recommended")
    if not shutil.which("clinfo") and not Path("/etc/OpenCL/vendors/nvidia.icd").exists():
        raise InstallError("NVIDIA OpenCL is required")
    if shutil.which("clinfo"):
        clinfo = subprocess.run(
            ["clinfo"],
            capture_output=True,
            text=True,
            check=False,
        )
        if clinfo.returncode != 0 or "NVIDIA" not in (clinfo.stdout + clinfo.stderr):
            raise InstallError("NVIDIA OpenCL is required")


def ensure_micromamba(local_bin: Path, *, dry_run: bool) -> Path:
    artifact = ARTIFACTS["micromamba"]
    override = os.environ.get("SKINSCOUT_MICROMAMBA")
    if override:
        micromamba = Path(override).expanduser()
        if not micromamba.is_file() or not os.access(micromamba, os.X_OK):
            raise InstallError(f"SKINSCOUT_MICROMAMBA is not executable: {micromamba}")
        if sha256(micromamba) != artifact["sha256"]:
            raise InstallError(
                "SKINSCOUT_MICROMAMBA does not match the pinned micromamba "
                f"{artifact['version']} artifact"
            )
        log(f"pinned micromamba 재사용: {micromamba}")
        return micromamba
    micromamba = local_bin / "micromamba"
    found = shutil.which("micromamba")
    if found and sha256(Path(found)) == artifact["sha256"]:
        log(f"pinned micromamba 재사용: {found}")
        return Path(found)
    if micromamba.is_file() and sha256(micromamba) == artifact["sha256"]:
        log(f"pinned micromamba 재사용: {micromamba}")
        return micromamba
    log(f"공식 GitHub 배포처에서 micromamba {artifact['version']}를 준비합니다.")
    download_verified(artifact, micromamba, dry_run=dry_run, executable=True)
    return micromamba


def ensure_mamba_link(local_bin: Path, micromamba: Path, *, dry_run: bool) -> None:
    if shutil.which("mamba"):
        return
    link = local_bin / "mamba"
    if dry_run:
        log(f"Snakemake용 mamba 호환 명령 준비 예정: {link}")
    elif link.is_symlink():
        link.unlink()
        link.symlink_to(micromamba)
    elif not link.exists():
        link.symlink_to(micromamba)


def ensure_env(
    micromamba: Path,
    env_name: str,
    env_file: Path,
    import_check: str,
    *,
    dry_run: bool,
) -> None:
    prefix = environment_prefix(env_name)
    spec_digest = sha256(env_file)
    digest_path = prefix / ".skinscout-environment-spec.sha256"
    if dry_run:
        log(
            f"기본 환경 {env_name} 준비 예정: {env_file.relative_to(ROOT)} "
            f"-> {prefix}"
        )
        return
    check = subprocess.run(
        [
            str(micromamba),
            "run",
            "--prefix",
            str(prefix),
            "python",
            "-c",
            import_check,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    recorded_digest = (
        digest_path.read_text(encoding="utf-8").strip()
        if digest_path.is_file()
        else None
    )
    if check.returncode == 0 and recorded_digest == spec_digest:
        log(f"환경 재사용: {env_name}")
        return
    log(f"환경을 다운로드합니다: {env_name}")
    env_exists = prefix.is_dir()
    action = "update" if env_exists else "create"
    command = [
        str(micromamba),
        "env",
        action,
        "--yes",
        "--file",
        str(env_file),
        "--prefix",
        str(prefix),
    ]
    run(command, dry_run=False)
    verified = subprocess.run(
        [
            str(micromamba),
            "run",
            "--prefix",
            str(prefix),
            "python",
            "-c",
            import_check,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if verified.returncode != 0:
        raise InstallError(f"environment validation failed after {action}: {env_name}")
    digest_path.write_text(spec_digest + "\n", encoding="utf-8")


def validate_activity_retrieval_gate(*, dry_run: bool) -> None:
    run(
        [
            sys.executable,
            str(ROOT / "scripts/validate_activity_retrieval_gate.py"),
            "check-operational",
            "--gate",
            str(ROOT / "data/manifests/activity_retrieval_operational_gate.flag"),
        ],
        dry_run=dry_run,
    )


def validate_full_model_readiness(micromamba: Path, *, dry_run: bool) -> None:
    command = [
        str(micromamba),
        "run",
        "--prefix",
        str(environment_prefix("cosmax-boltz2")),
        "python",
        str(ROOT / "scripts/model_readiness.py"),
    ]
    for requirement in FULL_MODEL_REQUIREMENTS:
        command.extend(["--require", requirement])
    run(command, dry_run=dry_run)


def create_snakemake_environments(
    micromamba: Path,
    profile: str,
    *,
    dry_run: bool,
) -> None:
    mode = "fast" if profile == "demo" else "both"
    run(
        [
            str(micromamba),
            "run",
            "--prefix",
            str(environment_prefix("cosmax-base")),
            "python",
            str(ROOT / "scripts/run_skinscout.py"),
            "--smiles",
            "CCO",
            "--preset",
            "report",
            "--mode",
            mode,
            "--evidence-mode",
            "evidence",
            "--run-id",
            f"installer_envs_{profile}",
            "--cores",
            str(max(1, min(16, os.cpu_count() or 1))),
            "--conda-create-envs-only",
        ],
        dry_run=dry_run,
    )


def download_verified(
    artifact: dict[str, str],
    destination: Path,
    *,
    dry_run: bool,
    executable: bool = False,
) -> None:
    try:
        display_path = destination.relative_to(ROOT)
    except ValueError:
        display_path = destination
    if destination.exists() and sha256(destination) == artifact["sha256"]:
        log(f"pinned artifact 재사용: {display_path}")
        return
    log(
        "pinned artifact 준비: "
        f"{artifact.get('version') or artifact.get('commit') or destination.name} "
        f"sha256={artifact['sha256']}"
    )
    if dry_run:
        log(f"다운로드: {artifact.get('url', '(repo-local artifact)')}")
        return
    url = artifact.get("url")
    if not url:
        raise InstallError(f"missing URL for pinned artifact {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=destination.parent) as handle:
        tmp_path = Path(handle.name)
    try:
        urllib.request.urlretrieve(url, tmp_path)
        observed = sha256(tmp_path)
        if observed != artifact["sha256"]:
            raise InstallError(
                f"checksum mismatch for {url}: expected {artifact['sha256']} got {observed}"
            )
        os.replace(tmp_path, destination)
        destination.chmod(0o755 if executable else 0o644)
    finally:
        tmp_path.unlink(missing_ok=True)


def _extract_tar_safely(archive: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise InstallError(
                    f"archive member escapes extraction directory: {member.name}"
                ) from exc
            if member.issym() or member.islnk():
                raise InstallError(
                    f"archive links are not accepted for pinned AutoGrid: {member.name}"
                )
        handle.extractall(destination, members=members)
    source_dirs = [path for path in destination.iterdir() if path.is_dir()]
    if len(source_dirs) != 1:
        raise InstallError("pinned AutoGrid archive must contain one source directory")
    return source_dirs[0]


def _autogrid_install_valid() -> bool:
    binary = ROOT / "tools/AutoGrid/autogrid4"
    parameter = ROOT / ARTIFACTS["autodock4_parameters"]["destination"]
    bound_parameter = ROOT / ARTIFACTS["autodock4_bound_parameters"]["destination"]
    manifest_path = ROOT / "tools/AutoGrid/source_manifest.json"
    if (
        not binary.is_file()
        or not os.access(binary, os.X_OK)
        or not parameter.is_file()
        or not bound_parameter.is_file()
    ):
        return False
    manifest = load_json(manifest_path)
    return bool(
        isinstance(manifest, dict)
        and manifest.get("schema_version") == "skinscout.autogrid-install.v1"
        and manifest.get("commit") == ARTIFACTS["autogrid"]["commit"]
        and manifest.get("archive_sha256") == ARTIFACTS["autogrid"]["sha256"]
        and manifest.get("binary_sha256") == sha256(binary)
        and manifest.get("parameter_sha256")
        == ARTIFACTS["autodock4_parameters"]["sha256"]
        and manifest.get("bound_parameter_sha256")
        == ARTIFACTS["autodock4_bound_parameters"]["sha256"]
        and sha256(parameter) == ARTIFACTS["autodock4_parameters"]["sha256"]
        and sha256(bound_parameter)
        == ARTIFACTS["autodock4_bound_parameters"]["sha256"]
    )


def ensure_autogrid(micromamba: Path, *, dry_run: bool) -> None:
    artifact = ARTIFACTS["autogrid"]
    archive = ROOT / "tools/downloads" / f"AutoGrid-{artifact['commit']}.tar.gz"
    if dry_run:
        download_verified(artifact, archive, dry_run=True)
        log("AutoGrid 4.2.8 Meson build and AD4 parameter verification 예정")
        return
    if _autogrid_install_valid():
        log("pinned AutoGrid 재사용")
        return
    download_verified(artifact, archive, dry_run=False)
    with tempfile.TemporaryDirectory(prefix="skinscout-autogrid-") as temporary:
        source = _extract_tar_safely(archive, Path(temporary) / "source")
        build = Path(temporary) / "build"
        run(
            [
                str(micromamba),
                "run",
                "--prefix",
                str(environment_prefix("cosmax-base")),
                "meson",
                "setup",
                str(build),
                str(source),
                "--buildtype=release",
            ],
            dry_run=False,
        )
        run(
            [
                str(micromamba),
                "run",
                "--prefix",
                str(environment_prefix("cosmax-base")),
                "meson",
                "compile",
                "-C",
                str(build),
            ],
            dry_run=False,
        )
        built_binary = build / "autogrid4"
        source_parameter = source / "ad4_shared/AD4_parameters.dat"
        source_bound_parameter = source / "ad4_shared/AD4.1_bound.dat"
        if not built_binary.is_file() or built_binary.stat().st_size == 0:
            raise InstallError("AutoGrid build did not produce autogrid4")
        observed_parameter = sha256(source_parameter)
        expected_parameter = ARTIFACTS["autodock4_parameters"]["sha256"]
        if observed_parameter != expected_parameter:
            raise InstallError(
                "AutoGrid AD4 parameter checksum mismatch: "
                f"expected {expected_parameter} got {observed_parameter}"
            )
        observed_bound_parameter = sha256(source_bound_parameter)
        expected_bound_parameter = ARTIFACTS["autodock4_bound_parameters"]["sha256"]
        if observed_bound_parameter != expected_bound_parameter:
            raise InstallError(
                "AutoGrid AD4 bound-parameter checksum mismatch: "
                f"expected {expected_bound_parameter} got {observed_bound_parameter}"
            )
        install_dir = ROOT / "tools/AutoGrid"
        install_dir.mkdir(parents=True, exist_ok=True)
        binary = install_dir / "autogrid4"
        parameter = ROOT / ARTIFACTS["autodock4_parameters"]["destination"]
        bound_parameter = ROOT / ARTIFACTS["autodock4_bound_parameters"]["destination"]
        parameter.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(delete=False, dir=install_dir) as handle:
            binary_tmp = Path(handle.name)
        with tempfile.NamedTemporaryFile(delete=False, dir=parameter.parent) as handle:
            parameter_tmp = Path(handle.name)
        with tempfile.NamedTemporaryFile(
            delete=False, dir=bound_parameter.parent
        ) as handle:
            bound_parameter_tmp = Path(handle.name)
        try:
            shutil.copy2(built_binary, binary_tmp)
            binary_tmp.chmod(0o755)
            shutil.copy2(source_parameter, parameter_tmp)
            parameter_tmp.chmod(0o644)
            shutil.copy2(source_bound_parameter, bound_parameter_tmp)
            bound_parameter_tmp.chmod(0o644)
            os.replace(binary_tmp, binary)
            os.replace(parameter_tmp, parameter)
            os.replace(bound_parameter_tmp, bound_parameter)
        finally:
            binary_tmp.unlink(missing_ok=True)
            parameter_tmp.unlink(missing_ok=True)
            bound_parameter_tmp.unlink(missing_ok=True)
        write_json_atomic(
            install_dir / "source_manifest.json",
            {
                "schema_version": "skinscout.autogrid-install.v1",
                "commit": artifact["commit"],
                "archive_url": artifact["url"],
                "archive_sha256": artifact["sha256"],
                "binary_sha256": sha256(binary),
                "parameter_sha256": sha256(parameter),
                "bound_parameter_sha256": sha256(bound_parameter),
            },
        )
    if not _autogrid_install_valid():
        raise InstallError("pinned AutoGrid installation did not validate")


def _meeko_build_is_exact(micromamba: Path, env_name: str) -> bool:
    result = subprocess.run(
        [
            str(micromamba),
            "list",
            "--prefix",
            str(environment_prefix(env_name)),
            "meeko",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return False
    try:
        records = json.loads(result.stdout)
    except json.JSONDecodeError:
        return False
    return any(
        isinstance(record, dict)
        and record.get("name") == "meeko"
        and str(record.get("version")) == ARTIFACTS["meeko_conda"]["version"]
        and str(record.get("build_string") or record.get("build"))
        == ARTIFACTS["meeko_conda"]["build"]
        for record in records
    )


def ensure_exact_meeko(micromamba: Path, *, dry_run: bool) -> None:
    artifact = ARTIFACTS["meeko_conda"]
    package = ROOT / "tools/downloads" / Path(artifact["url"]).name
    download_verified(artifact, package, dry_run=dry_run)
    for env_name in ("cosmax-autodock-gpu", "cosmax-meeko"):
        if dry_run:
            log(f"{env_name}: exact Meeko {artifact['version']} {artifact['build']} 설치 예정")
            continue
        if _meeko_build_is_exact(micromamba, env_name):
            log(f"exact Meeko 재사용: {env_name}")
            continue
        run(
            [
                str(micromamba),
                "install",
                "--yes",
                "--prefix",
                str(environment_prefix(env_name)),
                str(package),
            ],
            dry_run=False,
        )
        if not _meeko_build_is_exact(micromamba, env_name):
            raise InstallError(f"exact Meeko package validation failed: {env_name}")


def _ensure_tool_link(name: str, target: Path, local_bin: Path, *, dry_run: bool) -> None:
    link = local_bin / name
    if dry_run:
        log(f"도구 명령 링크 준비 예정: {link} -> {target}")
        return
    local_bin.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        link.unlink()
    elif link.exists():
        if link.is_file() and target.is_file() and sha256(link) == sha256(target):
            return
        raise InstallError(
            f"refusing to replace unrelated existing command: {link}; move it and rerun"
        )
    link.symlink_to(target)


def ensure_tool_links(local_bin: Path, *, dry_run: bool) -> None:
    for name, target in {
        "gnina": ROOT / ARTIFACTS["gnina"]["destination"],
        "autogrid4": ROOT / "tools/AutoGrid/autogrid4",
        "autodock_gpu_128wi": ROOT / ARTIFACTS["autodock_gpu_ocl"]["destination"],
    }.items():
        _ensure_tool_link(name, target, local_bin, dry_run=dry_run)


def ensure_pinned_tools(
    micromamba: Path,
    local_bin: Path,
    *,
    dry_run: bool,
) -> None:
    download_verified(
        ARTIFACTS["gnina"],
        ROOT / ARTIFACTS["gnina"]["destination"],
        dry_run=dry_run,
        executable=True,
    )
    download_verified(
        ARTIFACTS["autodock_gpu_ocl"],
        ROOT / ARTIFACTS["autodock_gpu_ocl"]["destination"],
        dry_run=dry_run,
        executable=True,
    )
    ensure_autogrid(micromamba, dry_run=dry_run)
    ensure_exact_meeko(micromamba, dry_run=dry_run)
    env = os.environ.copy()
    env["SKINSCOUT_P2RANK_SHA256"] = ARTIFACTS["p2rank"]["sha256"]
    run(
        [
            str(micromamba),
            "run",
            "--prefix",
            str(environment_prefix("cosmax-base")),
            "bash",
            str(ROOT / "scripts/setup_p2rank.sh"),
            "2.5",
        ],
        dry_run=dry_run,
        env=env,
    )
    ensure_tool_links(local_bin, dry_run=dry_run)


def qualification_dir(runtime_root: Path, state: dict[str, Any]) -> Path:
    return (
        runtime_root
        / "qualification"
        / str(state["installer_version"])
        / str(state["install_id"])
    )


def write_qualification_manifest(
    runtime_root: Path, state: dict[str, Any], *, dry_run: bool
) -> None:
    qdir = qualification_dir(runtime_root, state)
    if dry_run:
        log(f"qualification artifacts 준비 예정: {qdir}")
        return
    qdir.mkdir(parents=True, exist_ok=True)
    structural = load_json(qdir / "structural/qualification_status.json")
    report = load_json(qdir / "structural/report_qualification_status.json")
    if not isinstance(structural, dict) or structural.get("status") != "passed":
        raise InstallError("structural qualification status is not passed")
    if not isinstance(report, dict) or report.get("status") != "passed":
        raise InstallError("report qualification status is not passed")
    observed_artifacts: dict[str, dict[str, object]] = {}
    for name in ("gnina", "autodock_gpu_ocl"):
        artifact = ARTIFACTS[name]
        path = ROOT / artifact["destination"]
        observed_artifacts[name] = {
            "path": str(path),
            "sha256": sha256(path),
            "expected_sha256": artifact["sha256"],
        }
    autogrid = ROOT / "tools/AutoGrid/autogrid4"
    parameter = ROOT / ARTIFACTS["autodock4_parameters"]["destination"]
    bound_parameter = ROOT / ARTIFACTS["autodock4_bound_parameters"]["destination"]
    observed_artifacts["autogrid"] = {
        "path": str(autogrid),
        "sha256": sha256(autogrid),
        "source_commit": ARTIFACTS["autogrid"]["commit"],
        "source_archive_sha256": ARTIFACTS["autogrid"]["sha256"],
    }
    observed_artifacts["autodock4_parameters"] = {
        "path": str(parameter),
        "sha256": sha256(parameter),
        "expected_sha256": ARTIFACTS["autodock4_parameters"]["sha256"],
    }
    observed_artifacts["autodock4_bound_parameters"] = {
        "path": str(bound_parameter),
        "sha256": sha256(bound_parameter),
        "expected_sha256": ARTIFACTS["autodock4_bound_parameters"]["sha256"],
    }
    write_json_atomic(
        qdir / "installer_manifest.json",
        {
            "schema": "skinscout.installer-qualification.v2",
            "installer_version": state["installer_version"],
            "install_id": state["install_id"],
            "profile": state["profile"],
            "root": str(ROOT),
            "mamba_root_prefix": os.environ.get("MAMBA_ROOT_PREFIX"),
            "expected_artifacts": ARTIFACTS,
            "observed_artifacts": observed_artifacts,
            "structural_qualification": structural,
            "report_qualification": report,
        },
    )


def _qualification_payload_valid(path: Path, schema: str) -> bool:
    payload = load_json(path)
    if not isinstance(payload, dict) or payload.get("schema_version") != schema:
        return False
    if payload.get("status") != "passed":
        return False
    outputs = payload.get("outputs")
    if not isinstance(outputs, dict):
        return False
    root = path.parent
    for record in outputs.values():
        if not isinstance(record, dict):
            return False
        raw_path = record.get("path")
        expected_sha = record.get("sha256")
        if not isinstance(raw_path, str) or not isinstance(expected_sha, str):
            return False
        candidate = (root / raw_path).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError:
            return False
        if not candidate.is_file() or sha256(candidate) != expected_sha:
            return False
    return True


def structural_qualification_valid(runtime_root: Path, state: dict[str, Any]) -> bool:
    return _qualification_payload_valid(
        qualification_dir(runtime_root, state)
        / "structural/qualification_status.json",
        "skinscout.structural-qualification.v1",
    )


def report_qualification_valid(runtime_root: Path, state: dict[str, Any]) -> bool:
    return _qualification_payload_valid(
        qualification_dir(runtime_root, state)
        / "structural/report_qualification_status.json",
        "skinscout.report-qualification.v1",
    )


def ensure_playwright_browser(micromamba: Path, *, dry_run: bool) -> Path:
    if dry_run:
        log("Playwright Chromium 다운로드 및 desktop/mobile Mol* 검증 예정")
        return Path("<playwright-chromium>")
    query = (
        "from pathlib import Path; "
        "from playwright.sync_api import sync_playwright; "
        "p=sync_playwright().start(); "
        "print(p.chromium.executable_path); p.stop()"
    )

    def browser_path() -> Path | None:
        result = subprocess.run(
            [
                str(micromamba),
                "run",
                "--prefix",
                str(environment_prefix("cosmax-viz")),
                "python",
                "-c",
                query,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return None
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        candidate = Path(lines[-1]).expanduser() if lines else None
        return (
            candidate
            if candidate is not None
            and candidate.is_file()
            and os.access(candidate, os.X_OK)
            else None
        )

    existing = browser_path()
    if existing is not None:
        log(f"Playwright Chromium 재사용: {existing}")
        return existing
    run(
        [
            str(micromamba),
            "run",
            "--prefix",
            str(environment_prefix("cosmax-viz")),
            "playwright",
            "install",
            "chromium",
        ],
        dry_run=False,
    )
    installed = browser_path()
    if installed is None:
        raise InstallError("Playwright Chromium installation could not be validated")
    return installed


def run_structural_qualification(
    micromamba: Path,
    runtime_root: Path,
    state: dict[str, Any],
    *,
    dry_run: bool,
) -> None:
    output = qualification_dir(runtime_root, state) / "structural"
    run(
        [
            str(micromamba),
            "run",
            "--prefix",
            str(environment_prefix("cosmax-meeko")),
            "python",
            str(ROOT / "scripts/qualification_structural_smoke.py"),
            "--out-dir",
            str(output),
            "--daina-python",
            str(environment_prefix("cosmax-dti") / "bin/python"),
            "--chembl-fp",
            str(ROOT / "data/chembl37/fp_morgan2_2048.parquet"),
        ],
        dry_run=dry_run,
    )
    if not dry_run and not structural_qualification_valid(runtime_root, state):
        raise InstallError("structural qualification outputs failed validation")


def run_report_qualification(
    micromamba: Path,
    browser: Path,
    runtime_root: Path,
    state: dict[str, Any],
    *,
    dry_run: bool,
) -> None:
    output = qualification_dir(runtime_root, state) / "structural"
    run(
        [
            str(micromamba),
            "run",
            "--prefix",
            str(environment_prefix("cosmax-viz")),
            "python",
            str(ROOT / "scripts/qualification_report_smoke.py"),
            "--structural-dir",
            str(output),
            "--browser",
            str(browser),
        ],
        dry_run=dry_run,
    )
    if not dry_run and not report_qualification_valid(runtime_root, state):
        raise InstallError("report qualification outputs failed validation")


def run_phase(
    phase: str,
    args: argparse.Namespace,
    state: dict[str, Any],
    state_path: Path,
    micromamba_holder: dict[str, Path],
) -> None:
    if phase_complete(state, phase) and phase not in REVALIDATE_PHASES:
        if phase == "stage0_build" and not all(
            path.is_file()
            for path in (
                ROOT / "data/manifests/stage0_complete.flag",
                ROOT / "data/manifests/activity_retrieval_operational_gate.flag",
            )
        ):
            log("stage0_build: required completion gate missing; rebuilding")
        elif phase == "target_smoke" and not structural_qualification_valid(
            args.runtime_root, state
        ):
            log("target_smoke: qualification artifacts changed; rerunning")
        elif phase == "report_smoke" and not report_qualification_valid(
            args.runtime_root, state
        ):
            log("report_smoke: qualification artifacts changed; rerunning")
        else:
            log(f"{phase}: resume complete")
            return
    mark_phase(state, state_path, phase, "running", dry_run=args.dry_run)
    local_bin = Path.home() / ".local/bin"
    try:
        if phase == "host_preflight":
            if args.dry_run:
                log("host preflight 예정: Ubuntu 24.04 x86_64, nvidia-smi, NVIDIA OpenCL, >=12GB VRAM, >=32GB RAM, >=300GB free (500GB recommended)")
            else:
                require_host_preflight()
        elif phase == "host_packages":
            required_commands = (
                "git",
                "curl",
                "bzip2",
                "wget",
                "python3",
                "tar",
                "sha256sum",
                "clinfo",
            )
            missing = [cmd for cmd in required_commands if not shutil.which(cmd)]
            if missing and not args.dry_run:
                raise InstallError(
                    f"missing host packages/commands: {', '.join(missing)}"
                )
            log("Ubuntu host packages verified; NVIDIA driver is never installed or changed")
        elif phase == "source_checkout":
            if not (ROOT / "scripts/run_skinscout.py").exists():
                raise InstallError(f"SkinScout source checkout incomplete: {ROOT}")
            log(f"SkinScout source ready: {ROOT}")
        elif phase == "base_runtime":
            micromamba = ensure_micromamba(local_bin, dry_run=args.dry_run)
            micromamba_holder["path"] = micromamba
            ensure_mamba_link(local_bin, micromamba, dry_run=args.dry_run)
            ensure_env(
                micromamba,
                "cosmax-base",
                ROOT / "envs/base.yml",
                (
                    "import shutil, snakemake, rdkit; "
                    "assert all(shutil.which(x) for x in "
                    "('meson','ninja','tcsh','c++'))"
                ),
                dry_run=args.dry_run,
            )
        elif phase == "profile_envs":
            micromamba = micromamba_holder.get("path") or ensure_micromamba(
                local_bin, dry_run=args.dry_run
            )
            for env_name, env_file, import_check in (
                ("cosmax-autodock-gpu", "autodock_gpu.yml", "import meeko, rdkit"),
                ("cosmax-meeko", "meeko.yml", "import meeko, rdkit"),
                (
                    "cosmax-dti",
                    "dti.yml",
                    "import pandas, pyarrow, rdkit",
                ),
                (
                    "cosmax-viz",
                    "viz.yml",
                    "import playwright, PIL, pandas, rdkit",
                ),
            ):
                ensure_env(
                    micromamba,
                    env_name,
                    ROOT / "envs" / env_file,
                    import_check,
                    dry_run=args.dry_run,
                )
            if args.profile == "full":
                for env_name, env_file, import_check in (
                    ("cosmax-qm", "qm.yml", "import rdkit"),
                    ("cosmax-boltz2", "boltz2.yml", "import boltz, torch"),
                    ("cosmax-bioemu", "bioemu.yml", "print('ok')"),
                    ("cosmax-md", "md.yml", "print('ok')"),
                ):
                    ensure_env(
                        micromamba,
                        env_name,
                        ROOT / "envs" / env_file,
                        import_check,
                        dry_run=args.dry_run,
                    )
            else:
                log("demo profile: heavyweight model/physics/QM environments skipped")
            micromamba_holder["browser"] = ensure_playwright_browser(
                micromamba, dry_run=args.dry_run
            )
        elif phase == "pinned_tools":
            micromamba = micromamba_holder.get("path") or ensure_micromamba(
                local_bin, dry_run=args.dry_run
            )
            ensure_pinned_tools(micromamba, local_bin, dry_run=args.dry_run)
        elif phase == "fetch_analog_bundle":
            micromamba = micromamba_holder.get("path") or ensure_micromamba(
                local_bin, dry_run=args.dry_run
            )
            if not args.analog_bundle:
                raise InstallError(
                    "--profile analog 에는 --analog-bundle 이 필요합니다.\n"
                    "대체소재 검색용 데이터(원료 표·활성 인덱스·ADMET 예측)는 "
                    "저장소에 들어 있지 않고 이 설치가 만들지도 않습니다 - "
                    "만들려면 ChEMBL·BindingDB 원본 55 GB 가 있어야 하기 때문입니다.\n"
                    "담당자에게 번들 위치를 받아 넘기세요:\n"
                    "  --analog-bundle <URL 또는 경로>"
                )
            command = [
                str(micromamba), "run", "--prefix", str(environment_prefix("cosmax-base")),
                "python", str(ROOT / "scripts/fetch_analog_bundle.py"),
                "--from", args.analog_bundle,
            ]
            if args.analog_bundle_manifest:
                command += ["--manifest", args.analog_bundle_manifest]
            run(command, dry_run=args.dry_run)
        elif phase == "stage0_sources":
            micromamba = micromamba_holder.get("path") or ensure_micromamba(
                local_bin, dry_run=args.dry_run
            )
            run(
                [
                    str(micromamba),
                    "run",
                    "--prefix",
                    str(environment_prefix("cosmax-base")),
                    "python",
                    str(ROOT / "scripts/data_readiness.py"),
                    "--preset",
                    "stage0",
                    "--allow-stage0-build",
                ],
                dry_run=args.dry_run,
            )
            log("full Stage 0 source readiness checked; no hidden first-analysis downloads")
        elif phase == "stage0_build":
            micromamba = micromamba_holder.get("path") or ensure_micromamba(
                local_bin, dry_run=args.dry_run
            )
            run(
                [
                    str(micromamba),
                    "run",
                    "--prefix",
                    str(environment_prefix("cosmax-base")),
                    "python",
                    str(ROOT / "scripts/run_skinscout.py"),
                    "--preset",
                    "stage0",
                    "--run-id",
                    "stage0_bootstrap",
                    "--allow-stage0-build",
                    "--cores",
                    str(max(1, min(16, os.cpu_count() or 1))),
                ],
                dry_run=args.dry_run,
            )
        elif phase == "stage0_verify":
            micromamba = micromamba_holder.get("path") or ensure_micromamba(
                local_bin, dry_run=args.dry_run
            )
            run(
                [
                    str(micromamba),
                    "run",
                    "--prefix",
                    str(environment_prefix("cosmax-base")),
                    "python",
                    str(ROOT / "scripts/stage0_verify.py"),
                    "--strict",
                    "--claim-quality",
                    "--require-activity-evidence",
                ],
                dry_run=args.dry_run,
            )
            validate_activity_retrieval_gate(dry_run=args.dry_run)
            create_snakemake_environments(
                micromamba,
                args.profile,
                dry_run=args.dry_run,
            )
        elif phase == "target_smoke":
            micromamba = micromamba_holder.get("path") or ensure_micromamba(
                local_bin, dry_run=args.dry_run
            )
            run_structural_qualification(
                micromamba,
                args.runtime_root,
                state,
                dry_run=args.dry_run,
            )
        elif phase == "report_smoke":
            micromamba = micromamba_holder.get("path") or ensure_micromamba(
                local_bin, dry_run=args.dry_run
            )
            browser = micromamba_holder.get("browser") or ensure_playwright_browser(
                micromamba, dry_run=args.dry_run
            )
            run_report_qualification(
                micromamba,
                browser,
                args.runtime_root,
                state,
                dry_run=args.dry_run,
            )
            log("report-fast desktop/mobile Mol* smoke complete")
        elif phase == "launcher_ready":
            validate_activity_retrieval_gate(dry_run=args.dry_run)
            if args.profile == "full":
                micromamba = micromamba_holder.get("path") or ensure_micromamba(
                    local_bin, dry_run=args.dry_run
                )
                validate_full_model_readiness(micromamba, dry_run=args.dry_run)
            write_qualification_manifest(
                args.runtime_root,
                state,
                dry_run=args.dry_run,
            )
            log("launcher_ready complete")
    except BaseException as exc:
        mark_phase(
            state,
            state_path,
            phase,
            "failed",
            dry_run=args.dry_run,
            detail=str(exc)[:1000],
        )
        raise
    mark_phase(state, state_path, phase, "complete", dry_run=args.dry_run)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--runtime-root", type=Path, default=Path.home() / ".local/share/skinscout")
    parser.add_argument("--profile", default="demo", choices=sorted(PUBLIC_PROFILES),
                        help="analog=대체소재 검색만(Stage 0 없음), demo=기본, full=전체")
    parser.add_argument("--analog-bundle", default=None,
                        help="--profile analog 에서 받을 데이터 번들의 URL 또는 경로")
    parser.add_argument("--analog-bundle-manifest", default=None,
                        help="번들 매니페스트. 생략하면 번들 옆에서 찾는다")
    args = parser.parse_args(argv)
    args.original_invocation = ["bash", "scripts/bootstrap_runtime.sh", *argv]
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    args.runtime_root = args.runtime_root.expanduser()
    configure_mamba_root()
    local_bin = Path.home() / ".local/bin"
    os.environ["PATH"] = f"{local_bin}:{os.environ.get('PATH', '')}"
    state_dir = args.runtime_root / "state"
    state_path = state_dir / "install_state.json"
    try:
        with installer_lock(state_dir, dry_run=args.dry_run):
            state = prepare_state(args, state_path)
            if args.dry_run:
                log(
                    f"installer state 기록 예정: {state_path} schema={STATE_SCHEMA} profile={args.profile}"
                )
            else:
                write_json_atomic(state_path, state)
            micromamba_holder: dict[str, Path] = {}
            for phase in PHASES:
                if args.profile == "analog" and phase in ANALOG_SKIP_PHASES:
                    log(f"{phase}: analog 프로파일에서는 건너뜁니다(Stage 0 을 만들지 않습니다)")
                    continue
                if phase == "fetch_analog_bundle" and args.profile != "analog":
                    continue
                run_phase(phase, args, state, state_path, micromamba_holder)
        log("기본 실행 환경 준비가 끝났습니다.")
        log("대용량 Stage 0 데이터는 Workbench가 아니라 installer phase에서 명시적으로 준비됩니다.")
        return 0
    except InstallError as exc:
        print(f"[SkinScout][ERROR] {exc}", file=sys.stderr)
        return exc.code
    except subprocess.CalledProcessError as exc:
        print(f"[SkinScout][ERROR] command failed ({exc.returncode}): {exc.cmd}", file=sys.stderr)
        return exc.returncode or 2
    except Exception as exc:
        print(f"[SkinScout][ERROR] unexpected installer failure: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
