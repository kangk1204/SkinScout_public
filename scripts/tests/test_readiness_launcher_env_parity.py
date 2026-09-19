"""F34: no-conda readiness and launch must inspect the same environment."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import pipeline_readiness  # noqa: E402
import run_skinscout  # noqa: E402


def _write_tool(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)


def _environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    tool_in_inactive_env: bool,
) -> Path:
    active_bin = tmp_path / "active-bin"
    active_bin.mkdir()
    monkeypatch.setenv("PATH", str(active_bin))
    monkeypatch.delenv("SKINSCOUT_DISABLE_LOCAL_TOOL_PATHS", raising=False)
    if tool_in_inactive_env:
        monkeypatch.setenv("MAMBA_ROOT_PREFIX", str(tmp_path))
        _write_tool(tmp_path / "envs" / "cosmax-qm" / "bin" / "xtb")
    else:
        monkeypatch.delenv("MAMBA_ROOT_PREFIX", raising=False)
        _write_tool(active_bin / "xtb")
    return active_bin


def _model_group(
    monkeypatch: pytest.MonkeyPatch,
    *,
    use_conda: bool,
    readiness_command: str | None = None,
) -> dict:
    monkeypatch.setattr(
        pipeline_readiness,
        "_required_model_readiness",
        lambda *_args, **_kwargs: ["xtb"],
    )
    return pipeline_readiness._model_group(
        preset="safety",
        mode="fast",
        run_dti_sanity=False,
        use_conda=use_conda,
        readiness_command=readiness_command,
    )


def test_readiness_and_launcher_share_one_model_readiness_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[str]] = []

    def fake_run(command, **_kwargs):
        captured.append(list(command))
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"tools": {"xtb": {"status": "available"}}}),
            stderr="",
        )

    monkeypatch.setattr(pipeline_readiness.subprocess, "run", fake_run)
    group = _model_group(monkeypatch, use_conda=False)

    monkeypatch.setattr(run_skinscout.subprocess, "run", fake_run)
    run_skinscout._run_model_readiness(["xtb"], None, use_conda=False)

    expected = run_skinscout.model_readiness_command(["xtb"], None, use_conda=False)
    assert captured[0] == captured[1] == expected
    assert group["command"] == expected


def test_no_conda_model_only_in_inactive_env_fails_readiness_and_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _environment(tmp_path, monkeypatch, tool_in_inactive_env=True)

    group = _model_group(monkeypatch, use_conda=False)

    assert group["status"] == "failed"
    assert group["blocking"] is True
    assert any(blocker["label"] == "xtb" for blocker in group["blockers"])

    with pytest.raises(SystemExit, match="Model readiness preflight failed"):
        run_skinscout._run_model_readiness(["xtb"], None, use_conda=False)
    assert "xtb" in capsys.readouterr().err


def test_active_env_model_passes_readiness_and_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _environment(tmp_path, monkeypatch, tool_in_inactive_env=False)

    group = _model_group(monkeypatch, use_conda=False)

    assert group["status"] == "ok"
    assert group["blockers"] == []
    assert group["command"][0] == sys.executable
    assert "--active-environment-only" in group["command"]

    assert run_skinscout._run_model_readiness(["xtb"], None, use_conda=False) is None


def test_conda_discovery_passes_readiness_and_launch_together(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _environment(tmp_path, monkeypatch, tool_in_inactive_env=True)

    group = _model_group(monkeypatch, use_conda=True)

    assert group["status"] == "ok"
    assert group["command"][0] == sys.executable
    assert "--active-environment-only" not in group["command"]

    assert run_skinscout._run_model_readiness(["xtb"], None, use_conda=True) is None


def test_no_conda_readiness_command_rejection_matches_launcher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _environment(tmp_path, monkeypatch, tool_in_inactive_env=False)

    with pytest.raises(SystemExit, match="current Python environment"):
        _model_group(
            monkeypatch,
            use_conda=False,
            readiness_command="micromamba run -n other python",
        )

    with pytest.raises(SystemExit, match="current Python environment"):
        run_skinscout._run_model_readiness(
            ["xtb"],
            "micromamba run -n other python",
            use_conda=False,
        )
