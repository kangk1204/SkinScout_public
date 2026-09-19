"""Regression tests for Stage 11 code/config reproducibility binding (F12).

The repro pack must seal the executed code bytes (commit + tracked diff +
untracked source list) and bind the effective config, and the claim gate must
refuse a dirty snapshot without an explicit limitation record.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_repro_pack_module():
    spec = importlib.util.spec_from_file_location(
        "stage11_repro_pack_for_binding",
        ROOT / "scripts" / "stage11_repro_pack.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_script(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_effective_config_digest_changes_with_one_override(tmp_path: Path) -> None:
    module = load_repro_pack_module()
    config = tmp_path / "config.yaml"
    config.write_text("seed: 0\npaths:\n  results_root: results/runs\n")

    baseline = module._effective_config_digest(config, [])
    overridden = module._effective_config_digest(config, ["seed=1"])
    other = module._effective_config_digest(config, ["seed=2"])
    nested = module._effective_config_digest(
        config, ["paths.results_root=results/elsewhere"]
    )

    assert baseline != overridden
    assert overridden != other
    assert baseline != nested
    assert module._effective_config_digest(config, []) == baseline


def test_code_snapshot_digest_changes_with_one_byte_code_edit() -> None:
    module = load_repro_pack_module()
    base = module._code_snapshot_record(
        git_commit="a" * 40,
        tracked_diff=b"threshold=0.5\n",
        untracked_files=[],
        limitation=None,
    )
    edited = module._code_snapshot_record(
        git_commit="a" * 40,
        tracked_diff=b"threshold=0.6\n",
        untracked_files=[],
        limitation=None,
    )

    assert base["dirty"] is True
    assert base["snapshot_sha256"] != edited["snapshot_sha256"]
    assert base["snapshot_sha256"] == module._code_snapshot_record(
        git_commit="a" * 40,
        tracked_diff=b"threshold=0.5\n",
        untracked_files=[],
        limitation=None,
    )["snapshot_sha256"]


def test_code_snapshot_digest_changes_with_untracked_source_byte() -> None:
    module = load_repro_pack_module()
    first = module._code_snapshot_record(
        git_commit="a" * 40,
        tracked_diff=b"",
        untracked_files=[
            {"path": "scripts/new_tool.py", "bytes": 5, "sha256": "0" * 64}
        ],
        limitation=None,
    )
    second = module._code_snapshot_record(
        git_commit="a" * 40,
        tracked_diff=b"",
        untracked_files=[
            {"path": "scripts/new_tool.py", "bytes": 6, "sha256": "1" * 64}
        ],
        limitation=None,
    )

    assert first["dirty"] is True
    assert first["snapshot_sha256"] != second["snapshot_sha256"]


def test_effective_config_prefers_sealed_run_manifest(tmp_path: Path) -> None:
    module = load_repro_pack_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"hashes": {"config_sha256": "d" * 64}}) + "\n"
    )
    config = tmp_path / "config.yaml"
    config.write_text("seed: 0\n")

    record = module._effective_config_record(
        config_path=config,
        config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        overrides=["seed=5"],
        run_dir=run_dir,
    )

    assert record["source"] == "run_manifest"
    assert record["effective_config_sha256"] == "d" * 64
    assert record["base_config_sha256"] == hashlib.sha256(
        config.read_bytes()
    ).hexdigest()


def _write_eval_manifest(
    path: Path,
    *,
    git_commit: str = "b" * 40,
    config_sha256: str = "a" * 64,
) -> Path:
    from scripts.tests.test_stage11_claim_manifest import write_eval_manifest

    write_eval_manifest(path)
    payload = json.loads(path.read_text())
    payload["provenance"] = {
        "git_commit": git_commit,
        "config_sha256": config_sha256,
    }
    path.write_text(json.dumps(payload) + "\n")
    return path


def _write_repro_artifact(
    run_dir: Path,
    *,
    dirty: bool,
    limitation: dict[str, str] | None,
    git_commit: str = "b" * 40,
    config_sha256: str = "a" * 64,
    base_config_sha256: str | None = None,
) -> Path:
    repro_dir = run_dir / "publication" / "reproducibility"
    repro_dir.mkdir(parents=True, exist_ok=True)
    sidecars = {
        "git_commit.txt": git_commit + "\n",
        "config_hash.txt": config_sha256 + "\n",
        "code_snapshot.json": json.dumps({
            "schema_version": "skinscout.stage11-code-snapshot.v1",
            "git_commit": git_commit,
            "dirty": dirty,
            "limitation": limitation,
        }) + "\n",
        "effective_config.json": json.dumps({
            "schema_version": "skinscout.stage11-effective-config.v1",
            "source": "config_yaml_plus_cli_overrides",
            "base_config_sha256": base_config_sha256 or config_sha256,
            "effective_config_sha256": "c" * 64,
            "overrides": [],
        }) + "\n",
    }
    metadata = []
    for name, text in sidecars.items():
        sidecar = repro_dir / name
        sidecar.write_text(text)
        metadata.append({
            "relative_path": name,
            "bytes": sidecar.stat().st_size,
            "sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
        })
    manifest = repro_dir / "artifact_manifest.json"
    manifest.write_text(json.dumps({"metadata_files": metadata}) + "\n")
    return manifest


def _run_claim_manifest(run_dir: Path, manifest: Path, eval_manifest: Path) -> subprocess.CompletedProcess[str]:
    report = run_dir / "report.html"
    report.write_text("ok\n")
    return run_script([
        "scripts/stage11_claim_manifest.py",
        "--run-dir", str(run_dir),
        "--out-manifest", str(run_dir / "claim_manifest.json"),
        "--evaluation-manifest", str(eval_manifest),
        "--artifact", f"report={report}",
        "--artifact", f"reproducibility={manifest}",
    ])


def test_claim_manifest_blocks_dirty_snapshot_without_limitation(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = _write_repro_artifact(
        run_dir,
        dirty=True,
        limitation=None,
    )
    eval_manifest = _write_eval_manifest(tmp_path / "eval" / "iteration_manifest.json")

    res = _run_claim_manifest(run_dir, manifest, eval_manifest)

    assert res.returncode != 0
    assert "dirty_code_snapshot_without_limitation" in res.stderr
    assert not (run_dir / "claim_manifest.json").exists()


def test_claim_manifest_accepts_limited_dirty_snapshot(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    limitation = {
        "reason": "hotfix applied before stage11 publication",
        "approved_by": "release-owner",
        "recorded_at_utc": "2026-09-18T00:00:00Z",
    }
    manifest = _write_repro_artifact(
        run_dir,
        dirty=True,
        limitation=limitation,
    )
    eval_manifest = _write_eval_manifest(tmp_path / "eval" / "iteration_manifest.json")

    res = _run_claim_manifest(run_dir, manifest, eval_manifest)

    assert res.returncode == 0, res.stderr
    payload = json.loads((run_dir / "claim_manifest.json").read_text())
    assert payload["claim_ready"] is True
    assert payload["code_snapshot"]["dirty"] is True
    assert payload["code_snapshot"]["limitation"]["approved_by"] == "release-owner"
    assert payload["effective_config"]["effective_config_sha256"] == "c" * 64


def test_claim_manifest_accepts_clean_snapshot_control(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = _write_repro_artifact(
        run_dir,
        dirty=False,
        limitation=None,
    )
    eval_manifest = _write_eval_manifest(tmp_path / "eval" / "iteration_manifest.json")

    res = _run_claim_manifest(run_dir, manifest, eval_manifest)

    assert res.returncode == 0, res.stderr
    payload = json.loads((run_dir / "claim_manifest.json").read_text())
    assert payload["claim_ready"] is True
    assert payload["code_snapshot"]["dirty"] is False


def test_claim_manifest_rejects_effective_config_base_mismatch(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = _write_repro_artifact(
        run_dir,
        dirty=False,
        limitation=None,
        base_config_sha256="e" * 64,
    )
    eval_manifest = _write_eval_manifest(tmp_path / "eval" / "iteration_manifest.json")

    res = _run_claim_manifest(run_dir, manifest, eval_manifest)

    assert res.returncode != 0
    assert "effective_config_base_mismatch" in res.stderr
