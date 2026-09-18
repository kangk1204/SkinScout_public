"""Runtime bootstrap state-transition tests."""

from __future__ import annotations

import json
import os
import importlib.util
import io
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
INSTALL_RUNTIME = ROOT / "scripts/install_runtime.py"


def load_install_runtime_module():
    spec = importlib.util.spec_from_file_location(
        "install_runtime_under_test",
        INSTALL_RUNTIME,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def test_compose_bootstrap_stays_blocked_when_real_runtime_doctor_fails(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    for command in ("bzip2", "docker", "nvidia-smi", "cosign"):
        _write_executable(fake_bin / command, "#!/usr/bin/env bash\nexit 0\n")

    doctor_args = tmp_path / "doctor-args.txt"
    expected_host_cli = ROOT / "scripts/host_cli.py"
    _write_executable(
        fake_bin / "python3",
        """#!/usr/bin/env bash
if [[ "${1:-}" == "${SKINSCOUT_EXPECTED_HOST_CLI}" ]]; then
    printf '%s\n' "$@" > "${SKINSCOUT_DOCTOR_ARGS_PATH}"
    exit 17
fi
exec "${SKINSCOUT_REAL_PYTHON}" "$@"
""",
    )
    env = os.environ.copy()
    env.update({
        "PATH": f"{fake_bin}:{env['PATH']}",
        "SKINSCOUT_RUNTIME_ROOT": str(tmp_path / "runtime"),
        "SKINSCOUT_EXPECTED_HOST_CLI": str(expected_host_cli),
        "SKINSCOUT_DOCTOR_ARGS_PATH": str(doctor_args),
        "SKINSCOUT_REAL_PYTHON": sys.executable,
    })

    result = subprocess.run(
        ["bash", "scripts/bootstrap_runtime.sh", "--runtime", "compose"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 2
    state = json.loads(
        (tmp_path / "runtime/state/bootstrap_state.json").read_text(encoding="utf-8")
    )
    assert state["state"] == "readiness_blocked"
    assert doctor_args.read_text(encoding="utf-8").splitlines() == [
        str(expected_host_cli),
        "doctor",
    ]
    assert "--skip-gpu" not in doctor_args.read_text(encoding="utf-8")


def test_conda_legacy_bootstrap_dry_run_remains_available() -> None:
    result = subprocess.run(
        ["bash", "scripts/bootstrap_runtime.sh", "--dry-run", "--runtime", "conda"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "cosmax-base" in result.stdout


def test_bootstrap_defaults_to_beginner_conda_runtime() -> None:
    result = subprocess.run(
        ["bash", "scripts/bootstrap_runtime.sh", "--dry-run"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "cosmax-base" in result.stdout
    assert "Compose 실행 환경 준비" not in result.stdout
    assert "--preset stage0" in result.stdout
    assert "--allow-stage0-build" in result.stdout
    assert "--print-command" not in result.stdout
    assert "cosmax-dti" in result.stdout
    assert "qualification_structural_smoke.py" in result.stdout
    assert "--daina-python" in result.stdout
    assert "fp_morgan2_2048.parquet" in result.stdout
    assert "qualification_report_smoke.py" in result.stdout


def test_public_profiles_are_analog_demo_and_full_only() -> None:
    """세 개뿐이고, 그 밖의 이름은 거부한다.

    `analog` 는 공동연구자에게 넘기는 설치다 - 대체소재 검색만 쓰고 Stage 0
    (125 GB)을 만들지 않는다. 이름을 늘리는 것은 계약을 바꾸는 일이므로 여기에
    못 박아 둔다.
    """
    installer = load_install_runtime_module()

    assert installer.PUBLIC_PROFILES == {"analog", "demo", "full"}

    result = subprocess.run(
        ["bash", "scripts/bootstrap_runtime.sh", "--dry-run", "--profile", "base_runtime"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 2
    assert "--profile must be analog, demo or full" in result.stderr


def test_the_analog_profile_refuses_to_start_without_its_data_bundle() -> None:
    """번들이 없으면 시작하지 않는다.

    이 프로파일은 Stage 0 을 만들지 않으므로 번들이 유일한 데이터 출처다. 끝까지
    깔고 나서 "데이터가 없습니다"를 보여 주면 받는 쪽은 무엇이 잘못됐는지 모른다.
    """
    for script in ("install_skinscout.sh", "scripts/bootstrap_runtime.sh"):
        result = subprocess.run(
            ["bash", script, "--dry-run", "--profile", "analog"],
            cwd=ROOT, text=True, capture_output=True, check=False, timeout=60,
        )
        assert result.returncode == 2, script
        assert "--analog-bundle" in result.stderr, script


def test_with_target_models_aliases_full_but_conflicts_with_profile() -> None:
    alias = subprocess.run(
        ["bash", "scripts/bootstrap_runtime.sh", "--dry-run", "--with-target-models"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert alias.returncode == 0, alias.stderr
    assert "profile=full" in alias.stdout
    assert "cosmax-boltz2" in alias.stdout

    conflict = subprocess.run(
        [
            "bash",
            "scripts/bootstrap_runtime.sh",
            "--dry-run",
            "--profile",
            "demo",
            "--with-target-models",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert conflict.returncode == 2
    assert "cannot be combined with --profile" in conflict.stderr


def test_install_runtime_pins_exact_public_artifacts() -> None:
    installer = load_install_runtime_module()

    assert installer.STATE_SCHEMA == "skinscout.install.v2"
    assert installer.PHASES == [
        "host_preflight",
        "host_packages",
        "source_checkout",
        "base_runtime",
        "profile_envs",
        "pinned_tools",
        # 대체소재 검색 전용 데이터 번들. `--profile analog` 에서만 돌고, 그때는
        # 아래 stage0_* 와 두 smoke 를 건너뛴다 - Stage 0 은 125 GB 를 만드는데
        # 대체소재 화면이 여는 것은 그중 원료 표 하나(13 MB)와 번들이 가져오는
        # 둘뿐이기 때문이다.
        "fetch_analog_bundle",
        "stage0_sources",
        "stage0_build",
        "stage0_verify",
        "target_smoke",
        "report_smoke",
        "launcher_ready",
    ]
    assert installer.ARTIFACTS["micromamba"] == {
        "version": "2.8.1-0",
        "url": "https://github.com/mamba-org/micromamba-releases/releases/download/2.8.1-0/micromamba-linux-64",
        "sha256": "9689782d863c05a1bf5d2d371ba527104e7a4eb4310c1637d8653b751aed9c82",
    }
    assert installer.ARTIFACTS["gnina"] == {
        "version": "v1.3.3",
        "url": "https://github.com/gnina/gnina/releases/download/v1.3.3/gnina.cuda12.8.static",
        "sha256": "3340c1f49cd3c7c84d8699182a1c6af13c7fa2a22448d1204640446106f72172",
        "destination": "tools/gnina",
    }
    assert installer.ARTIFACTS["autodock_gpu_ocl"]["sha256"] == (
        "8a22804c1fde62a59c030a45e8a9d9d960ff329d22e4927daf7464df393b8047"
    )
    assert installer.ARTIFACTS["autogrid"]["commit"] == (
        "6d2847beaeac8ff43ca99094707fd74e3ca1ff37"
    )
    assert installer.ARTIFACTS["autogrid"]["sha256"] == (
        "4d0bd83a446fd81577f4fc492299e22f131245589e1782e0532aecf3435e772a"
    )
    assert installer.ARTIFACTS["autodock4_parameters"]["sha256"] == (
        "912b557d6249c8ef74d4bca62acbb82dc456a4c92c07c6fd0cec37530dcc8307"
    )
    assert installer.ARTIFACTS["autodock4_bound_parameters"]["sha256"] == (
        "91a720ec15959727d6a5386bc19d46350f1d4683758900d88109101fccabb869"
    )
    assert installer.ARTIFACTS["autodock4_bound_parameters"]["destination"] == (
        "tools/AutoGrid/AD4.1_bound.dat"
    )
    assert installer.ARTIFACTS["meeko_conda"]["sha256"] == (
        "f9e1f221be6305c9b3214c832e450c303ff291e88dd6ffc966ff4ef77f468e5d"
    )
    assert installer.ARTIFACTS["meeko_conda"]["build"] == "pyhd8ed1ab_1"
    assert installer.ARTIFACTS["meeko_conda"]["url"].endswith(
        "meeko-0.7.1-pyhd8ed1ab_1.conda"
    )
    assert installer.ARTIFACTS["p2rank"]["sha256"] == (
        "9c21755967450300f2eb052d059ef128d38021626e52e135561c7c4687891751"
    )


def test_install_state_preserves_resume_and_supports_demo_to_full_upgrade(
    tmp_path: Path,
) -> None:
    installer = load_install_runtime_module()
    state_path = tmp_path / "state/install_state.json"
    demo_args = installer.parse_args(
        ["--runtime-root", str(tmp_path), "--profile", "demo"]
    )
    state = installer.initial_state(demo_args, "install-1")
    installer.write_json_atomic(state_path, state)

    resumed = installer.prepare_state(demo_args, state_path)
    assert resumed["schema"] == "skinscout.install.v2"
    assert resumed["profile"] == "demo"
    assert resumed["original_invocation"] == [
        "bash",
        "scripts/bootstrap_runtime.sh",
        "--runtime-root",
        str(tmp_path),
        "--profile",
        "demo",
    ]

    full_args = installer.parse_args(
        ["--runtime-root", str(tmp_path), "--profile", "full"]
    )
    upgraded = installer.prepare_state(full_args, state_path)
    assert upgraded["profile"] == "full"
    assert upgraded["profile_history"] == ["demo", "full"]
    assert all(
        upgraded["phases"][phase]["status"] == "pending"
        for phase in installer.PROFILE_UPGRADE_PHASES
    )

    installer.write_json_atomic(state_path, upgraded)
    requested_demo = installer.parse_args(
        ["--runtime-root", str(tmp_path), "--profile", "demo"]
    )
    retained = installer.prepare_state(requested_demo, state_path)
    assert retained["profile"] == "full"
    assert requested_demo.profile == "full"


def test_install_state_revalidates_code_dependent_phases_after_version_upgrade(
    tmp_path: Path,
) -> None:
    installer = load_install_runtime_module()
    args = installer.parse_args(
        ["--runtime-root", str(tmp_path), "--profile", "demo"]
    )
    state_path = tmp_path / "state/install_state.json"
    state = installer.initial_state(args, "install-1")
    state["installer_version"] = "older"
    for phase in installer.PHASES:
        state["phases"][phase] = {"status": "complete"}
    installer.write_json_atomic(state_path, state)

    upgraded = installer.prepare_state(args, state_path)

    assert upgraded["installer_version"] == installer.INSTALLER_VERSION
    assert all(
        upgraded["phases"][phase]["status"] == "pending"
        for phase in installer.INSTALLER_UPGRADE_PHASES
    )
    assert upgraded["phases"]["stage0_build"]["status"] == "complete"


def test_ensure_env_uses_the_managed_mamba_prefix(
    tmp_path: Path, monkeypatch,
) -> None:
    installer = load_install_runtime_module()
    mamba_root = tmp_path / "mamba-root"
    monkeypatch.setenv("MAMBA_ROOT_PREFIX", str(mamba_root))
    command_log = tmp_path / "commands.txt"
    monkeypatch.setenv("FAKE_MAMBA_LOG", str(command_log))
    micromamba = tmp_path / "micromamba"
    _write_executable(
        micromamba,
        """#!/usr/bin/env bash
printf '%s\n' "$*" >> "${FAKE_MAMBA_LOG}"
if [[ "$1" == "run" ]]; then
    [[ -d "$3" ]]
    exit $?
fi
if [[ "$1" == "env" ]]; then
    while (($#)); do
        if [[ "$1" == "--prefix" ]]; then shift; mkdir -p "$1"; exit 0; fi
        shift
    done
fi
exit 1
""",
    )
    env_file = tmp_path / "environment.yml"
    env_file.write_text("name: ignored\ndependencies:\n  - python\n", encoding="utf-8")

    installer.ensure_env(
        micromamba,
        "cosmax-test",
        env_file,
        "print('ok')",
        dry_run=False,
    )

    expected = mamba_root / "envs/cosmax-test"
    commands = command_log.read_text(encoding="utf-8")
    assert expected.is_dir()
    assert f"--prefix {expected}" in commands
    assert " -n " not in f" {commands} "


def test_ensure_env_reuses_only_the_exact_environment_spec(
    tmp_path: Path, monkeypatch,
) -> None:
    installer = load_install_runtime_module()
    mamba_root = tmp_path / "mamba-root"
    monkeypatch.setenv("MAMBA_ROOT_PREFIX", str(mamba_root))
    command_log = tmp_path / "commands.txt"
    monkeypatch.setenv("FAKE_MAMBA_LOG", str(command_log))
    micromamba = tmp_path / "micromamba"
    _write_executable(
        micromamba,
        """#!/usr/bin/env bash
printf '%s\n' "$*" >> "${FAKE_MAMBA_LOG}"
if [[ "$1" == "run" ]]; then [[ -d "$3" ]]; exit $?; fi
if [[ "$1" == "env" ]]; then
    while (($#)); do
        if [[ "$1" == "--prefix" ]]; then shift; mkdir -p "$1"; exit 0; fi
        shift
    done
fi
exit 1
""",
    )
    env_file = tmp_path / "environment.yml"
    env_file.write_text("dependencies:\n  - python=3.12\n", encoding="utf-8")

    installer.ensure_env(micromamba, "cosmax-test", env_file, "print('ok')", dry_run=False)
    installer.ensure_env(micromamba, "cosmax-test", env_file, "print('ok')", dry_run=False)
    env_file.write_text("dependencies:\n  - python=3.12\n  - pandas\n", encoding="utf-8")
    installer.ensure_env(micromamba, "cosmax-test", env_file, "print('ok')", dry_run=False)

    commands = command_log.read_text(encoding="utf-8").splitlines()
    mutations = [line for line in commands if line.startswith(("env create ", "env update "))]
    assert len(mutations) == 2
    digest = mamba_root / "envs/cosmax-test/.skinscout-environment-spec.sha256"
    assert digest.read_text().strip() == installer.sha256(env_file)


def test_runtime_gate_and_full_model_readiness_commands_are_fail_closed(
    tmp_path: Path, monkeypatch,
) -> None:
    installer = load_install_runtime_module()
    calls: list[tuple[list[str], bool]] = []
    monkeypatch.setattr(
        installer,
        "run",
        lambda command, dry_run: calls.append((list(command), dry_run)),
    )

    installer.validate_activity_retrieval_gate(dry_run=False)
    installer.validate_full_model_readiness(tmp_path / "micromamba", dry_run=False)

    gate = calls[0][0]
    assert gate[-3:] == [
        "check-operational",
        "--gate",
        str(installer.ROOT / "data/manifests/activity_retrieval_operational_gate.flag"),
    ]
    model = calls[1][0]
    assert model[:4] == [
        str(tmp_path / "micromamba"),
        "run",
        "--prefix",
        str(installer.environment_prefix("cosmax-boltz2")),
    ]
    assert [model[index + 1] for index, value in enumerate(model) if value == "--require"] == list(
        installer.FULL_MODEL_REQUIREMENTS
    )


def test_installer_precreates_exact_snakemake_environments_for_each_profile(
    tmp_path: Path, monkeypatch,
) -> None:
    installer = load_install_runtime_module()
    calls: list[list[str]] = []
    monkeypatch.setattr(
        installer,
        "run",
        lambda command, dry_run: calls.append(list(command)),
    )

    installer.create_snakemake_environments(
        tmp_path / "micromamba", "demo", dry_run=False
    )
    installer.create_snakemake_environments(
        tmp_path / "micromamba", "full", dry_run=False
    )

    assert len(calls) == 2
    for command, profile, mode in zip(
        calls,
        ("demo", "full"),
        ("fast", "both"),
        strict=True,
    ):
        assert command[command.index("--preset") + 1] == "report"
        assert command[command.index("--mode") + 1] == mode
        assert command[command.index("--run-id") + 1] == f"installer_envs_{profile}"
        assert "--conda-create-envs-only" in command
        assert "--no-use-conda" not in command


def test_safe_tar_extraction_rejects_parent_escape(tmp_path: Path) -> None:
    installer = load_install_runtime_module()
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        member = tarfile.TarInfo("../outside.txt")
        payload = b"unsafe"
        member.size = len(payload)
        handle.addfile(member, io.BytesIO(payload))

    with pytest.raises(installer.InstallError, match="escapes extraction directory"):
        installer._extract_tar_safely(archive, tmp_path / "extract")


def test_qualification_validation_rejects_tamper_and_path_escape(
    tmp_path: Path,
) -> None:
    installer = load_install_runtime_module()
    root = tmp_path / "qualification"
    root.mkdir()
    schema = "skinscout.structural-qualification.v1"
    required = sorted(installer.QUALIFICATION_REQUIRED_OUTPUTS[schema])
    outputs: dict[str, dict[str, object]] = {}
    for name in required:
        artifact = root / f"{name}.txt"
        artifact.write_text(f"{name}\n", encoding="utf-8")
        outputs[name] = {
            "path": artifact.name,
            "bytes": artifact.stat().st_size,
            "sha256": installer.sha256(artifact),
        }
    status = root / "status.json"
    payload: dict[str, object] = {
        "schema_version": schema,
        "status": "passed",
        "outputs": outputs,
    }
    status.write_text(json.dumps(payload), encoding="utf-8")
    assert installer._qualification_payload_valid(status, schema)

    tampered = root / f"{required[0]}.txt"
    tampered.write_text("tampered\n", encoding="utf-8")
    assert not installer._qualification_payload_valid(status, schema)

    tampered.write_text(f"{required[0]}\n", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("qualified\n", encoding="utf-8")
    outputs[required[0]] = {
        "path": "../outside.txt",
        "bytes": outside.stat().st_size,
        "sha256": installer.sha256(outside),
    }
    status.write_text(json.dumps(payload), encoding="utf-8")
    assert not installer._qualification_payload_valid(status, schema)


def test_compose_bootstrap_stays_blocked_when_runtime_doctor_is_missing(
    tmp_path: Path,
) -> None:
    isolated_root = tmp_path / "isolated"
    isolated_scripts = isolated_root / "scripts"
    isolated_scripts.mkdir(parents=True)
    bootstrap = isolated_scripts / "bootstrap_runtime.sh"
    shutil.copy2(ROOT / "scripts/bootstrap_runtime.sh", bootstrap)

    fake_bin = tmp_path / "missing-doctor-bin"
    fake_bin.mkdir()
    for command in ("bzip2", "docker", "nvidia-smi", "cosign"):
        _write_executable(fake_bin / command, "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(
        fake_bin / "python3",
        "#!/usr/bin/env bash\nexec \"${SKINSCOUT_REAL_PYTHON}\" \"$@\"\n",
    )
    env = os.environ.copy()
    env.update({
        "PATH": f"{fake_bin}:{env['PATH']}",
        "SKINSCOUT_RUNTIME_ROOT": str(tmp_path / "missing-doctor-runtime"),
        "SKINSCOUT_REAL_PYTHON": sys.executable,
    })

    result = subprocess.run(
        ["bash", str(bootstrap), "--runtime", "compose"],
        cwd=isolated_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 2
    state = json.loads(
        (tmp_path / "missing-doctor-runtime/state/bootstrap_state.json").read_text(
            encoding="utf-8"
        )
    )
    assert state["state"] == "readiness_blocked"
    assert "doctor" in result.stderr
