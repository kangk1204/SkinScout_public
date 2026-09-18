"""설치 점검기가 빠진 것을 정확히 FAIL/WARN으로 가르는지.

설치기는 "해시가 맞다"까지만 말해 준다. 분석이 실제로 도는지는 다른 질문이고,
공동연구자가 전달받은 자리에서 그 질문에 답할 수 있어야 한다.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import verify_install as verify  # noqa: E402


def _make_env_python(root: Path) -> Path:
    python = root / "envs" / "cosmax-base" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python.chmod(0o755)
    return python


def test_env_probe_accepts_a_probe_that_imports_rdkit(tmp_path, monkeypatch) -> None:
    _make_env_python(tmp_path)
    monkeypatch.setenv("MAMBA_ROOT_PREFIX", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert verify.find_env_python() == tmp_path / "envs" / "cosmax-base" / "bin" / "python"


def test_empty_install_reports_blocking_failures(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    monkeypatch.setattr(verify, "find_env_python", lambda: None)
    checks = verify.run_checks("demo", skip_smoke=True)
    status = {check["id"]: check["status"] for check in checks}

    assert status["runtime"] == verify.FAIL
    assert status["autogrid"] == verify.FAIL
    assert status["gate"] == verify.FAIL
    assert status["stage0"] == verify.WARN  # 막지는 않지만 알려 준다
    assert status["smoke"] == verify.SKIP


def test_analog_profile_skips_gpu_and_checks_its_own_data(tmp_path, monkeypatch) -> None:
    (tmp_path / "data" / "cosing").mkdir(parents=True)
    (tmp_path / "data" / "cosing" / "cosing.parquet").write_bytes(b"x")
    (tmp_path / "data" / "similarity_index_202609").mkdir(parents=True)
    (tmp_path / "data" / "similarity_index_202609" / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    monkeypatch.setattr(verify, "find_env_python", lambda: None)
    checks = verify.run_checks("analog", skip_smoke=True)
    status = {check["id"]: check["status"] for check in checks}

    assert status["gpu"] == verify.SKIP
    assert status["autogrid"] == verify.SKIP
    assert status["analog_data"] == verify.PASS


def test_json_output_is_machine_readable() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "verify_install.py"), "--json", "--skip-smoke"],
        capture_output=True, text=True, timeout=600, check=False,
    )
    payload = json.loads(completed.stdout)
    assert payload["profile"] == "demo"
    assert payload["checks"]
    for check in payload["checks"]:
        assert set(check) == {"id", "label", "status", "detail"}
        assert check["status"] in {verify.PASS, verify.WARN, verify.FAIL, verify.SKIP}


def test_the_readme_points_at_the_verifier() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "verify_install" in readme
