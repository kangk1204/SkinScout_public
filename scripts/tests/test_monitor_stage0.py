from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess


SOURCE = Path(__file__).resolve().parents[1] / "monitor_stage0.sh"


def _sandbox(tmp_path: Path, *, verifier_exit: int) -> tuple[Path, dict[str, str]]:
    root = tmp_path / "repo"
    scripts = root / "scripts"
    manifests = root / "data" / "manifests"
    fake_bin = tmp_path / "bin"
    scripts.mkdir(parents=True)
    manifests.mkdir(parents=True)
    fake_bin.mkdir()
    shutil.copy2(SOURCE, scripts / SOURCE.name)
    (scripts / "stage0_verify.py").touch()
    (scripts / "validate_activity_retrieval_gate.py").touch()

    fake_python = fake_bin / "python3"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$*" >> "$MONITOR_CALLS"\n'
        'printf "%s\\n" "$PWD" >> "$MONITOR_CWD"\n'
        f"exit {verifier_exit}\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    calls = tmp_path / "verifier.calls"
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["MONITOR_CALLS"] = str(calls)
    env["MONITOR_CWD"] = str(tmp_path / "verifier.cwd")
    return root, env


def _run(
    root: Path,
    env: dict[str, str],
    cwd: Path,
    pid: str = "99999999",
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(root / "scripts" / "monitor_stage0.sh"), pid, "1"],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )


def test_resolves_repo_root_from_script_when_started_from_arbitrary_cwd(
    tmp_path: Path,
) -> None:
    root, env = _sandbox(tmp_path, verifier_exit=0)
    manifests = root / "data" / "manifests"
    (manifests / "stage0_complete.flag").touch()
    (manifests / "activity_retrieval_operational_gate.flag").touch()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    result = _run(root, env, elsewhere)

    assert result.returncode == 0, result.stderr
    log = (root / "results" / "logs" / "stage0_progress.log").read_text()
    assert "DONE — Stage 0 and operational readiness verified." in log
    calls = (tmp_path / "verifier.calls").read_text().splitlines()
    assert calls == [
        f"{root}/scripts/stage0_verify.py --repo {root} "
        "--strict --claim-quality --require-activity-evidence",
        f"{root}/scripts/validate_activity_retrieval_gate.py check-operational --gate "
        f"{manifests}/activity_retrieval_operational_gate.flag",
    ]
    assert (tmp_path / "verifier.cwd").read_text().splitlines() == [str(root), str(root)]
    assert not (elsewhere / "results").exists()


def test_stale_completion_artifacts_are_not_reported_as_done(
    tmp_path: Path,
) -> None:
    root, env = _sandbox(tmp_path, verifier_exit=1)
    manifests = root / "data" / "manifests"
    (manifests / "stage0_complete.flag").touch()
    (manifests / "activity_retrieval_operational_gate.flag").touch()

    result = _run(root, env, tmp_path)

    assert result.returncode == 0, result.stderr
    log = (root / "results" / "logs" / "stage0_progress.log").read_text()
    assert "DONE" not in log
    assert "strict verification failed" in log
    assert "readiness not verified" in log


def test_stage0_flag_alone_is_only_reported_as_artifact_presence(
    tmp_path: Path,
) -> None:
    root, env = _sandbox(tmp_path, verifier_exit=0)
    (root / "data" / "manifests" / "stage0_complete.flag").touch()

    result = _run(root, env, tmp_path)

    assert result.returncode == 0, result.stderr
    log = (root / "results" / "logs" / "stage0_progress.log").read_text()
    assert "DONE" not in log
    assert "operational gate is missing; readiness not verified" in log
    assert not (tmp_path / "verifier.calls").exists()


def test_rejects_invalid_pid_and_interval_without_creating_log(
    tmp_path: Path,
) -> None:
    root, env = _sandbox(tmp_path, verifier_exit=0)

    bad_pid = _run(root, env, tmp_path, pid="not-a-pid")
    bad_interval = subprocess.run(
        ["bash", str(root / "scripts" / "monitor_stage0.sh"), "123", "0"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )

    assert bad_pid.returncode == 2
    assert "positive integer" in bad_pid.stderr
    assert bad_interval.returncode == 2
    assert "positive integer" in bad_interval.stderr
    assert not (root / "results" / "logs" / "stage0_progress.log").exists()
