"""Regression tests for Stage 2 ADMET-AI availability gates."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[2]


def write_input_sdf(path: Path) -> None:
    mol = Chem.MolFromSmiles("CCO")
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()


def run_with_admet_ai_shim(
    tmp_path: Path,
    shim_source: str,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    input_sdf = tmp_path / "in.sdf"
    write_input_sdf(input_sdf)

    shim_dir = tmp_path / "shim"
    shim_dir.mkdir()
    (shim_dir / "admet_ai.py").write_text(shim_source)

    env_pythonpath = os.environ.get("PYTHONPATH", "")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{shim_dir}{os.pathsep}{env_pythonpath}" if env_pythonpath else str(shim_dir)

    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage2_admet_ai.py"),
            "--in-sdf",
            str(input_sdf),
            "--out-json",
            str(tmp_path / "admet.json"),
            *(extra_args or []),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def run_with_missing_admet_ai(
    tmp_path: Path,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return run_with_admet_ai_shim(
        tmp_path,
        "raise ModuleNotFoundError(\"No module named 'admet_ai'\")\n",
        extra_args,
    )


def test_admet_ai_missing_fails_without_explicit_degraded_mode(tmp_path: Path) -> None:
    (tmp_path / "admet.json").write_text("stale\n")
    res = run_with_missing_admet_ai(tmp_path)

    assert res.returncode != 0
    assert "admet-ai is required" in res.stderr
    assert not (tmp_path / "admet.json").exists()


def test_admet_ai_missing_degraded_mode_is_recorded(tmp_path: Path) -> None:
    res = run_with_missing_admet_ai(tmp_path, ["--allow-unavailable"])

    assert res.returncode == 0, res.stderr
    payload = json.loads((tmp_path / "admet.json").read_text())
    assert payload["status"] == "unavailable"
    assert payload["degraded"] is True
    assert payload["degraded_reason"] == "admet_ai_unavailable"
    assert payload["predictions"] == {}


def test_admet_ai_dependency_failure_is_not_degraded(tmp_path: Path) -> None:
    (tmp_path / "admet.json").write_text("stale\n")
    res = run_with_admet_ai_shim(
        tmp_path,
        "raise ModuleNotFoundError(\"No module named 'pkg_resources'\")\n",
        ["--allow-unavailable"],
    )

    assert res.returncode != 0
    assert "one of its dependencies failed to import" in res.stderr
    assert "pkg_resources" in res.stderr
    assert not (tmp_path / "admet.json").exists()


def test_admet_ai_defaults_to_cpu_isolation(tmp_path: Path) -> None:
    res = run_with_admet_ai_shim(
        tmp_path,
        """
import os

class ADMETModel:
    def __init__(self):
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise RuntimeError("GPU was not isolated")

    def predict(self, *, smiles):
        return {"score": 0.5, "smiles": smiles}
""",
    )

    assert res.returncode == 0, res.stderr
    payload = json.loads((tmp_path / "admet.json").read_text())
    assert payload["status"] == "ok"
    assert payload["predictions"]["score"] == 0.5
