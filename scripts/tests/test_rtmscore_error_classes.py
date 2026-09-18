"""F33: RTMScore must separate bad structures from a broken batch.

The old wrapper turned every Exception into ``None`` and the batch proceeded
whenever at least one target scored, so a model-init or shape error looked like
a per-target miss. These tests pin the classification, the partial-coverage
manifest, and the torch_scatter shim provenance/parity evidence.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "stage3_rtmscore.py"
ADAPTER_PATH = ROOT / "rtmscore" / "__init__.py"


def _load_stage3(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "stage3_rtmscore_under_test", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_adapter():
    spec = importlib.util.spec_from_file_location(
        "rtmscore_adapter_under_test", ADAPTER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _raise(exc: BaseException):
    def _raiser(*_args, **_kwargs):
        raise exc

    return _raiser


def _pose_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    pose_dir = tmp_path / "poses"
    pose_dir.mkdir()
    clean = tmp_path / "clean"
    clean.mkdir()
    top = tmp_path / "top.csv"
    top.write_text("target_id,score\nP1,1.0\nP2,0.9\n", encoding="utf-8")
    records = []
    for target_id in ("P1", "P2"):
        pose = pose_dir / f"{target_id}.sdf"
        pose.write_text(f"{target_id} ligand\n", encoding="utf-8")
        (clean / f"{target_id}_clean.pdb").write_text(
            "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 90.00           C\nEND\n",
            encoding="utf-8",
        )
        records.append(
            {
                "target_id": target_id,
                "pose_file": f"{target_id}.sdf",
                "pose_sha256": hashlib.sha256(pose.read_bytes()).hexdigest(),
            }
        )
    manifest = pose_dir / "pose_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "skinscout.docking_pose_manifest.v1",
                "target_count": len(records),
                "targets": records,
            }
        ),
        encoding="utf-8",
    )
    return pose_dir, manifest, clean, top


def _fake_package(tmp_path: Path, body: str) -> Path:
    package = tmp_path / "fake_pkg"
    module_dir = package / "rtmscore"
    module_dir.mkdir(parents=True)
    (module_dir / "__init__.py").write_text(body, encoding="utf-8")
    return package


def _run_rtmscore(
    tmp_path: Path,
    pose_dir: Path,
    manifest: Path,
    clean: Path,
    top: Path,
    env: dict[str, str],
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    scores = tmp_path / "rtm.tsv"
    status = tmp_path / "rtm_status.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--top-csv",
            str(top),
            "--pose-manifest",
            str(manifest),
            "--pose-dir",
            str(pose_dir),
            "--clean-dir",
            str(clean),
            "--out-scores",
            str(scores),
            "--out-status-manifest",
            str(status),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    return result, scores, status


PER_STRUCTURE_FAKE = """
class RTMScoreStructureError(RuntimeError):
    rtmscore_structure_error = True


def backend_provenance():
    return {
        "backend": "shim",
        "real_backend_available": False,
        "shim_installed": True,
        "parity_claim": "unverified",
    }


def probe_scatter_parity():
    return {
        "status": "skipped",
        "reason": "real torch_scatter backend is not installed",
        "parity_claim": "unverified",
    }


def predict_affinity(receptor, ligand, **kwargs):
    if "P2" in str(receptor):
        raise RTMScoreStructureError("ligand SDF has no readable molecule")
    return 1.5
"""

MODEL_INIT_FAKE = """
def predict_affinity(receptor, ligand, **kwargs):
    raise RuntimeError(
        "Error(s) in loading state_dict for RTMScore: size mismatch for layer"
    )
"""


def test_only_expected_structure_errors_are_per_structure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_stage3(monkeypatch)

    assert module._is_structure_input_error(
        module.RTMScoreStructureError("bad ligand")
    )
    assert module._is_structure_input_error(FileNotFoundError("pose gone"))

    class FakeUpstreamStructureError(RuntimeError):
        rtmscore_structure_error = True

    assert module._is_structure_input_error(FakeUpstreamStructureError("x"))
    for exc in (
        RuntimeError("Error(s) in loading state_dict for RTMScore"),
        ValueError("shape mismatch in scatter"),
        KeyError("model_state_dict"),
        AttributeError("'NoneType' object has no attribute 'GetAtoms'"),
        IndexError("index out of range"),
    ):
        assert not module._is_structure_input_error(exc), exc


def test_structure_failure_is_partial_and_recorded_in_the_manifest(
    tmp_path: Path,
) -> None:
    pose_dir, manifest, clean, top = _pose_fixture(tmp_path)
    package = _fake_package(tmp_path, PER_STRUCTURE_FAKE)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(package)

    result, scores, status = _run_rtmscore(
        tmp_path, pose_dir, manifest, clean, top, env
    )

    assert result.returncode == 0, result.stderr
    frame = pd.read_csv(scores, sep="\t")
    assert frame["target_id"].tolist() == ["P1"]
    payload = json.loads(status.read_text(encoding="utf-8"))
    assert payload["scored_count"] == 1
    assert payload["input_target_count"] == 2
    assert payload["partial_coverage"] is True
    assert payload["status_counts"]["structure_failed_rtmscore"] == 1
    failures = {record["target_id"]: record for record in payload["failure_reasons"]}
    assert "no readable molecule" in failures["P2"]["detail"]
    assert failures["P2"]["status"] == "structure_failed_rtmscore"
    backend = payload["rtmscore_backend"]
    assert backend["provenance"]["recorded"] is True
    assert backend["provenance"]["backend"] == "shim"
    assert backend["parity"]["status"] == "skipped"


def test_model_init_failure_fails_the_whole_batch(tmp_path: Path) -> None:
    pose_dir, manifest, clean, top = _pose_fixture(tmp_path)
    package = _fake_package(tmp_path, MODEL_INIT_FAKE)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(package)

    result, scores, status = _run_rtmscore(
        tmp_path, pose_dir, manifest, clean, top, env
    )

    assert result.returncode != 0
    assert "RTMScore model/runtime failure" in result.stderr
    assert "RTMScore batch failed" in result.stderr
    assert not scores.exists()
    assert not scores.with_suffix(scores.suffix + ".tmp").exists()
    assert not status.exists()


def test_with_reason_returns_structure_reason_and_reraises_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_stage3(monkeypatch)
    receptor = Path("receptor.pdb")
    ligand = Path("ligand.sdf")

    monkeypatch.setattr(
        module,
        "_predict_score",
        _raise(module.RTMScoreStructureError("ligand has no atoms")),
    )
    score, reason = module.rtmscore_with_reason(receptor, ligand)
    assert score is None
    assert "ligand has no atoms" in reason
    assert module.rtmscore(receptor, ligand) is None

    monkeypatch.setattr(
        module,
        "_predict_score",
        _raise(module.RTMScoreBatchError("model checkpoint unreadable")),
    )
    with pytest.raises(module.RTMScoreBatchError):
        module.rtmscore_with_reason(receptor, ligand)
    with pytest.raises(module.RTMScoreBatchError):
        module.rtmscore(receptor, ligand)


def test_adapter_maps_upstream_structure_errors_but_not_model_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _load_adapter()
    model = tmp_path / "model.pth"
    model.write_bytes(b"x")

    class FakeUpstream:
        @staticmethod
        def scoring(**_kwargs):
            raise ValueError("The graph of pocket cannot be generated")

    monkeypatch.setattr(adapter, "_load_upstream", lambda: FakeUpstream)
    monkeypatch.setattr(adapter, "_usable_device", lambda requested: "cpu")
    with pytest.raises(adapter.RTMScoreStructureError):
        adapter.predict_affinity("prot.pdb", "lig.sdf", model_path=model)

    class BrokenModelUpstream:
        @staticmethod
        def scoring(**_kwargs):
            raise RuntimeError("Error(s) in loading state_dict for RTMScore")

    monkeypatch.setattr(adapter, "_load_upstream", lambda: BrokenModelUpstream)
    with pytest.raises(RuntimeError) as excinfo:
        adapter.predict_affinity("prot.pdb", "lig.sdf", model_path=model)
    assert not isinstance(excinfo.value, adapter.RTMScoreStructureError)

    assert adapter._is_structure_error(FileNotFoundError("gone"))
    assert adapter._is_structure_error(
        ValueError("The graph of pocket cannot be generated")
    )
    assert not adapter._is_structure_error(
        RuntimeError("Error(s) in loading state_dict for RTMScore")
    )


def test_adapter_records_backend_provenance_and_skips_parity_without_backend() -> None:
    adapter = _load_adapter()

    provenance = adapter.backend_provenance()
    assert provenance["backend"] in {"shim", "real", "unavailable"}
    assert provenance["aggregations_used"] == ["scatter_add"]
    assert provenance["parity_claim"] == "unverified"

    probe = adapter.probe_scatter_parity()
    if provenance["real_backend_available"]:
        assert probe["status"] in {"passed", "failed"}
    else:
        assert probe["status"] == "skipped"
        assert probe["parity_claim"] == "unverified"
        assert "no parity is claimed" in probe["reason"]


def test_adapter_flags_an_installed_shim_without_claiming_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _load_adapter()
    fake = types.ModuleType("torch_scatter")
    setattr(fake, adapter.SHIM_FLAG, True)
    fake.__skinscout_shim_reason__ = "test shim"
    monkeypatch.setitem(sys.modules, "torch_scatter", fake)

    provenance = adapter.backend_provenance()
    assert provenance["backend"] == "shim"
    assert provenance["shim_installed"] is True
    assert provenance["real_backend_available"] is False
    assert provenance["shim_reason"] == "test shim"

    probe = adapter.probe_scatter_parity()
    assert probe["status"] == "skipped"
    assert probe["backend"] == "shim"
    assert probe["parity_claim"] == "unverified"
