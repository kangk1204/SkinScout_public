"""Regression tests for lightweight model readiness probes (C12).

``--active-environment-only`` must not report an adapter as ready just because
the bundled wrapper imports: the same interpreter has to be able to load the
upstream checkout, its core dependencies, and the checkpoint. Full model
initialisation stays a separate smoke test.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_model_readiness_module():
    scripts_dir = str(ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(
        "model_readiness_probe_under_test",
        ROOT / "scripts/model_readiness.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_psichic_checkout(root: Path) -> Path:
    inference = root / "PSICHIC-prod" / "inference.py"
    inference.parent.mkdir(parents=True)
    inference.write_text("x\n")
    weight_dir = root / "trained_weights" / "PDBv2020_PSICHIC"
    weight_dir.mkdir(parents=True)
    for name in ("config.json", "degree.pt", "model.pt"):
        (weight_dir / name).write_text("x\n")
    return weight_dir


def write_rtmscore_checkout(root: Path) -> Path:
    script = root / "example" / "rtmscore.py"
    script.parent.mkdir(parents=True)
    script.write_text("x\n")
    model = root / "trained_models" / "rtmscore_model1.pth"
    model.parent.mkdir(parents=True)
    model.write_text("x\n")
    return model


def find_spec_factory(*, missing: set[str], present: set[str]):
    original = importlib.util.find_spec

    def fake_find_spec(name: str):
        if name in missing:
            return None
        if name in present:
            return importlib.machinery.ModuleSpec(name, loader=None)
        return original(name)

    return fake_find_spec


def test_psichic_probe_reports_missing_torch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    readiness = load_model_readiness_module()
    write_psichic_checkout(tmp_path)
    monkeypatch.setenv("PSICHIC_ROOT", str(tmp_path))
    monkeypatch.delenv("PSICHIC_MODEL", raising=False)
    monkeypatch.setattr(
        readiness.importlib.util,
        "find_spec",
        find_spec_factory(missing={"torch", "torch_geometric"}, present=set()),
    )

    status = readiness.psichic_status(probe_runtime=True)

    assert status["status"] == "missing"
    assert status["reason"] == "missing PSICHIC runtime dependencies"
    assert status["missing_dependencies"] == ["torch", "torch_geometric"]
    assert status["probe"] == "runtime_dependencies;checkpoint_readability"


def test_psichic_probe_reports_missing_checkpoint_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    readiness = load_model_readiness_module()
    weight_dir = write_psichic_checkout(tmp_path)
    (weight_dir / "config.json").unlink()
    monkeypatch.setenv("PSICHIC_ROOT", str(tmp_path))
    monkeypatch.delenv("PSICHIC_MODEL", raising=False)
    monkeypatch.setattr(
        readiness.importlib.util,
        "find_spec",
        find_spec_factory(missing=set(), present={"torch", "torch_geometric"}),
    )

    status = readiness.psichic_status(probe_runtime=True)

    assert status["status"] == "missing"
    assert status["reason"] == "unreadable PSICHIC checkpoint"
    assert any("config.json" in detail for detail in status["unreadable_checkpoints"])


def test_psichic_probe_passes_when_dependencies_and_checkpoint_are_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    readiness = load_model_readiness_module()
    write_psichic_checkout(tmp_path)
    monkeypatch.setenv("PSICHIC_ROOT", str(tmp_path))
    monkeypatch.delenv("PSICHIC_MODEL", raising=False)
    monkeypatch.setattr(
        readiness.importlib.util,
        "find_spec",
        find_spec_factory(missing=set(), present={"torch", "torch_geometric"}),
    )

    status = readiness.psichic_status(probe_runtime=True)

    assert status["status"] == "available"
    assert status["probe"] == "runtime_dependencies;checkpoint_readability"
    assert "missing_dependencies" not in status


def test_rtmscore_probe_reports_missing_dgl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    readiness = load_model_readiness_module()
    write_rtmscore_checkout(tmp_path)
    monkeypatch.setenv("RTMSCORE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        readiness.importlib.util,
        "find_spec",
        find_spec_factory(missing={"torch", "dgl"}, present=set()),
    )

    status = readiness.rtmscore_status(probe_runtime=True)

    assert status["status"] == "missing"
    assert status["reason"] == "missing RTMScore runtime dependencies"
    assert status["missing_dependencies"] == ["torch", "dgl"]


def test_probes_are_off_without_active_environment_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness = load_model_readiness_module()
    seen: dict[str, object] = {}

    def fake_psichic_status(**kwargs):
        seen["psichic"] = kwargs
        return {"status": "available"}

    def fake_rtmscore_status(**kwargs):
        seen["rtmscore"] = kwargs
        return {"status": "available"}

    monkeypatch.setattr(
        readiness,
        "command_status",
        lambda name, discover_inactive_conda_envs=True: {"status": "missing"},
    )
    monkeypatch.setattr(readiness, "gnina_status", lambda: {"status": "missing"})
    monkeypatch.setattr(
        readiness, "autogrid_status", lambda **kwargs: {"status": "missing"}
    )
    monkeypatch.setattr(
        readiness, "diffdock_status", lambda **kwargs: {"status": "missing"}
    )
    monkeypatch.setattr(
        readiness,
        "cuda_status",
        lambda: {"status": "available", "cuda_available": False},
    )
    monkeypatch.setattr(readiness.gpu_admission, "probe", lambda **kwargs: {})
    monkeypatch.setattr(readiness, "psichic_status", fake_psichic_status)
    monkeypatch.setattr(readiness, "rtmscore_status", fake_rtmscore_status)

    args = argparse.Namespace(
        active_environment_only=True,
        boltz_smoke_receptor=None,
        boltz_smoke_ligand=None,
    )
    readiness.readiness(args)
    assert seen["psichic"] == {"probe_runtime": True}
    assert seen["rtmscore"] == {"probe_runtime": True}

    seen.clear()
    args.active_environment_only = False
    readiness.readiness(args)
    assert seen["psichic"] == {"probe_runtime": False}
    assert seen["rtmscore"] == {"probe_runtime": False}
