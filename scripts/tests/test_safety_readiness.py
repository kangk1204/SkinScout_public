"""Regression tests for Stage 2 safety readiness preflight."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
READINESS = ROOT / "scripts/safety_readiness.py"


def load_readiness_module():
    spec = importlib.util.spec_from_file_location("safety_readiness_under_test", READINESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_safety_readiness_accepts_current_dti_env_manifest() -> None:
    readiness = load_readiness_module()

    checks = readiness._check_env_manifest(ROOT / "envs/dti.yml")

    assert {check["name"] for check in checks if check["ok"]} >= {
        "env:admet-ai",
        "env:rdkit",
        "env:requests",
    }


def test_safety_readiness_rejects_missing_requests(tmp_path: Path) -> None:
    readiness = load_readiness_module()
    env_yml = tmp_path / "dti.yml"
    env_yml.write_text(
        """
name: probe
dependencies:
  - python=3.11
  - rdkit
  - pip
  - pip:
      - admet-ai==1.4.0
"""
    )

    checks = readiness._check_env_manifest(env_yml)
    failed = {check["name"] for check in checks if not check["ok"]}

    assert "env:requests" in failed


def test_safety_readiness_allows_optional_degraded_dependencies(tmp_path: Path) -> None:
    readiness = load_readiness_module()
    env_yml = tmp_path / "dti.yml"
    env_yml.write_text(
        """
name: probe
dependencies:
  - python=3.11
  - rdkit
"""
    )

    checks = readiness._check_env_manifest(env_yml, allow_degraded=True)
    failed = {check["name"] for check in checks if not check["ok"]}
    warnings = {check["name"] for check in checks if check.get("warning")}

    assert failed == set()
    assert warnings == {"env:admet-ai", "env:requests"}


def test_safety_readiness_stage2_endpoints_are_current() -> None:
    readiness = load_readiness_module()

    checks = readiness._check_stage2_endpoints()

    assert all(check["ok"] for check in checks)


def test_safety_readiness_online_probe_rejects_client_error(monkeypatch) -> None:
    readiness = load_readiness_module()

    def fake_get(url: str, timeout: int, allow_redirects: bool):
        assert timeout == 15
        assert allow_redirects is True
        return SimpleNamespace(status_code=404)

    import requests

    monkeypatch.setattr(requests, "get", fake_get)

    check = readiness._online_probe("missing", "https://example.test/missing")

    assert not check["ok"]
    assert check["status_code"] == 404


def test_safety_readiness_online_probe_accepts_success(monkeypatch) -> None:
    readiness = load_readiness_module()

    def fake_get(url: str, timeout: int, allow_redirects: bool):
        return SimpleNamespace(status_code=204)

    import requests

    monkeypatch.setattr(requests, "get", fake_get)

    check = readiness._online_probe("ok", "https://example.test/ok")

    assert check["ok"]
    assert check["status_code"] == 204
