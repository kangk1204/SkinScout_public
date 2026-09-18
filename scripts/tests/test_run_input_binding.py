"""F15 regression tests for run input binding and verify-existing ordering.

The target metadata path is not an identity: rewriting the same file with
different gene/protein names changes every displayed summary label while the
path stays the same. These tests pin the content hash into the run config
identity, verify that ``--verify-existing-run`` validates the recorded identity
before regenerating anything, and prove summary regeneration is staged and
atomic.
"""

from __future__ import annotations

import importlib.util
import tempfile
from argparse import Namespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "run_skinscout.py"
_TEST_RESULTS_TMP = tempfile.TemporaryDirectory(prefix="skinscout-input-binding-")
TEST_RESULTS_ROOT = Path(_TEST_RESULTS_TMP.name)
METADATA_V1 = "UniProt\tGene\tProtein\nP12345\tGENEA\tProtein A\n"
METADATA_V2 = "UniProt\tGene\tProtein\nP12345\tGENEB\tProtein B\n"


def load_runner_module():
    spec = importlib.util.spec_from_file_location(
        "run_skinscout_input_binding",
        RUNNER,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.DEFAULT_RESULTS_ROOT = TEST_RESULTS_ROOT
    return module


def _args(
    runner,
    tmp_path: Path,
    metadata: Path,
    run_id: str,
):
    return runner.parse_args(
        [
            "--smiles",
            "CCO",
            "--run-id",
            run_id,
            "--preset",
            "target-id",
            "--mode",
            "fast",
            "--target-metadata",
            str(metadata),
            "--extra-config",
            f"paths.results_root={tmp_path}",
        ]
    )


def test_config_identity_binds_target_metadata_bytes(tmp_path: Path) -> None:
    runner = load_runner_module()
    metadata = tmp_path / "proteinatlas.tsv"
    metadata.write_text(METADATA_V1)
    args = _args(runner, tmp_path, metadata, "binding_case")

    before = runner._run_manifest_payload(args)
    metadata.write_text(METADATA_V2)
    after = runner._run_manifest_payload(args)

    assert before["hashes"]["config_sha256"] != after["hashes"]["config_sha256"]


def test_ensure_run_manifest_rejects_same_path_different_bytes(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    metadata = tmp_path / "proteinatlas.tsv"
    metadata.write_text(METADATA_V1)
    args = _args(runner, tmp_path, metadata, "manifest_bytes_case")

    manifest = runner._ensure_run_manifest(args)
    before = manifest.read_bytes()

    metadata.write_text(METADATA_V2)
    with pytest.raises(SystemExit, match="config_sha256"):
        runner._ensure_run_manifest(args)

    assert manifest.read_bytes() == before


def test_verify_existing_validates_corrupt_manifest_before_regenerating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_runner_module()
    metadata = tmp_path / "proteinatlas.tsv"
    metadata.write_text(METADATA_V1)
    args = _args(runner, tmp_path, metadata, "corrupt_manifest_case")
    run_dir = tmp_path / "corrupt_manifest_case"
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text("{not-json\n")
    (run_dir / "run_summary.json").write_text("prior summary bytes\n")
    (run_dir / "run_verification.json").write_text("prior verification bytes\n")

    monkeypatch.setattr(
        runner,
        "_run_output_summary",
        lambda *_a, **_k: pytest.fail(
            "regenerated a summary before validating the manifest"
        ),
    )
    monkeypatch.setattr(
        runner,
        "_run_output_verification",
        lambda *_a: pytest.fail(
            "reverified outputs before validating the manifest"
        ),
    )

    with pytest.raises(SystemExit, match="could not be read"):
        runner._verify_existing_run(args)

    assert (run_dir / "run_summary.json").read_text() == "prior summary bytes\n"
    assert (
        run_dir / "run_verification.json"
    ).read_text() == "prior verification bytes\n"
    assert (run_dir / "run_manifest.json").read_text() == "{not-json\n"


def test_verify_existing_rejects_changed_metadata_without_rewriting_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_runner_module()
    metadata = tmp_path / "proteinatlas.tsv"
    metadata.write_text(METADATA_V1)
    args = _args(runner, tmp_path, metadata, "changed_metadata_case")
    manifest = runner._ensure_run_manifest(args)
    run_dir = manifest.parent
    prior = {
        "run_summary.json": "prior summary bytes\n",
        "run_summary.md": "prior summary markdown\n",
        "run_verification.json": "prior verification bytes\n",
        "run_verification.log": "prior verification log\n",
        "run_manifest.json": manifest.read_text(),
    }
    for name, text in prior.items():
        (run_dir / name).write_text(text)

    metadata.write_text(METADATA_V2)
    monkeypatch.setattr(
        runner,
        "_run_output_summary",
        lambda *_a, **_k: pytest.fail(
            "regenerated a summary for changed metadata bytes"
        ),
    )
    monkeypatch.setattr(
        runner,
        "_run_output_verification",
        lambda *_a: pytest.fail(
            "reverified outputs for changed metadata bytes"
        ),
    )

    with pytest.raises(SystemExit, match="config_sha256"):
        runner._verify_existing_run(args)

    for name, text in prior.items():
        assert (run_dir / name).read_text() == text


def test_verify_existing_regenerates_only_after_identity_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_runner_module()
    metadata = tmp_path / "proteinatlas.tsv"
    metadata.write_text(METADATA_V1)
    args = _args(runner, tmp_path, metadata, "clean_control_case")
    manifest = runner._ensure_run_manifest(args)
    before = manifest.read_bytes()

    calls: list[object] = []
    original_ensure = runner._ensure_run_manifest

    def recording_ensure(*call_args: object, **call_kwargs: object):
        calls.append("ensure")
        return original_ensure(*call_args, **call_kwargs)

    monkeypatch.setattr(runner, "_ensure_run_manifest", recording_ensure)
    monkeypatch.setattr(
        runner,
        "_run_output_summary",
        lambda *_a, **kwargs: calls.append(("summary", kwargs)),
    )
    monkeypatch.setattr(
        runner,
        "_run_output_verification",
        lambda *_a: calls.append(("verify", {})),
    )
    monkeypatch.setattr(
        runner,
        "_print_completed_run_result",
        lambda *_a: calls.append(("print", {})),
    )

    assert runner._verify_existing_run(args) == 0
    assert calls[0] == "ensure"
    assert calls[1][0] == "summary"
    assert calls[1][1].get("atomic_replace") is True
    assert ["ensure", "summary", "verify", "print"] == [
        call[0] if isinstance(call, tuple) else call for call in calls
    ]
    assert manifest.read_bytes() == before


def test_output_summary_stages_then_replaces_on_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_runner_module()
    metadata = tmp_path / "proteinatlas.tsv"
    metadata.write_text(METADATA_V1)
    args = _args(runner, tmp_path, metadata, "staged_case")
    run_dir = tmp_path / "staged_case"
    run_dir.mkdir(parents=True)
    (run_dir / "run_summary.json").write_text("old json")
    (run_dir / "run_summary.md").write_text("old md")

    def fake_run(cmd: list[str], *_args: object, **_kwargs: object) -> Namespace:
        out_json = Path(cmd[cmd.index("--out-json") + 1])
        out_md = Path(cmd[cmd.index("--out-md") + 1])
        assert out_json.parent != run_dir
        assert out_json.parent.parent == run_dir
        out_json.write_text("new json")
        out_md.write_text("new md")
        return Namespace(returncode=0, stdout="summary ok\n", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(runner, "_build_results_viewer", lambda *_a: None)

    runner._run_output_summary(args, atomic_replace=True)

    assert (run_dir / "run_summary.json").read_text() == "new json"
    assert (run_dir / "run_summary.md").read_text() == "new md"
    assert not [
        path
        for path in run_dir.iterdir()
        if path.name.startswith(".run_summary_stage_")
    ]


def test_output_summary_failure_preserves_previous_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_runner_module()
    metadata = tmp_path / "proteinatlas.tsv"
    metadata.write_text(METADATA_V1)
    args = _args(runner, tmp_path, metadata, "failed_stage_case")
    run_dir = tmp_path / "failed_stage_case"
    run_dir.mkdir(parents=True)
    (run_dir / "run_summary.json").write_text("old json")
    (run_dir / "run_summary.md").write_text("old md")

    def fake_run(cmd: list[str], *_args: object, **_kwargs: object) -> Namespace:
        out_json = Path(cmd[cmd.index("--out-json") + 1])
        out_json.write_text("partial json")
        return Namespace(returncode=2, stdout="", stderr="summary failed\n")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(runner, "_build_results_viewer", lambda *_a: None)

    with pytest.raises(SystemExit, match="Completed run summary generation failed"):
        runner._run_output_summary(args, atomic_replace=True)

    assert (run_dir / "run_summary.json").read_text() == "old json"
    assert (run_dir / "run_summary.md").read_text() == "old md"
    assert not [
        path
        for path in run_dir.iterdir()
        if path.name.startswith(".run_summary_stage_")
    ]


def test_output_summary_missing_staged_output_is_not_a_silent_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_runner_module()
    metadata = tmp_path / "proteinatlas.tsv"
    metadata.write_text(METADATA_V1)
    args = _args(runner, tmp_path, metadata, "empty_stage_case")
    run_dir = tmp_path / "empty_stage_case"
    run_dir.mkdir(parents=True)
    (run_dir / "run_summary.json").write_text("old json")
    (run_dir / "run_summary.md").write_text("old md")

    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_a, **_k: Namespace(returncode=0, stdout="", stderr=""),
    )
    monkeypatch.setattr(runner, "_build_results_viewer", lambda *_a: None)

    with pytest.raises(SystemExit, match="produced no"):
        runner._run_output_summary(args, atomic_replace=True)

    assert (run_dir / "run_summary.json").read_text() == "old json"
    assert (run_dir / "run_summary.md").read_text() == "old md"


def test_config_identity_is_stable_for_unchanged_metadata_bytes(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    metadata = tmp_path / "proteinatlas.tsv"
    metadata.write_text(METADATA_V1)
    args = _args(runner, tmp_path, metadata, "stable_metadata_case")

    before = runner._run_manifest_payload(args)
    after = runner._run_manifest_payload(args)

    assert before["hashes"]["config_sha256"] == after["hashes"]["config_sha256"]


def _verification_record(
    run_dir: Path,
    metadata: Path,
    *,
    metadata_sha256: str | None,
) -> dict[str, object]:
    return {
        "schema_version": "skinscout.run_verification.v1",
        "status": "ok",
        "returncode": 0,
        "verifier_status": "ok",
        "verified_at_utc": "2026-01-01T00:00:00Z",
        "run_dir": str(run_dir),
        "preset": "target-id",
        "mode": "fast",
        "stdout_log": "run_verification.log",
        "command": [
            "python",
            "scripts/verify_run_outputs.py",
            "--run-dir",
            str(run_dir),
            "--target-metadata",
            str(metadata),
        ],
        "target_metadata": {
            "path": str(metadata),
            "sha256": metadata_sha256,
        },
        "diagnostic_nonclaimable_reasons": [],
    }


def test_verification_record_rejects_changed_metadata_identity(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    metadata = tmp_path / "proteinatlas.tsv"
    metadata.write_text(METADATA_V1)
    run_dir = tmp_path / "record_case"
    record = _verification_record(
        run_dir,
        metadata,
        metadata_sha256="0" * 64,
    )

    with pytest.raises(SystemExit, match="target metadata identity"):
        runner._validate_stored_completed_verification_record(
            record,
            run_dir,
            preset="target-id",
            mode="fast",
        )


def test_verification_record_accepts_matching_metadata_identity(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    metadata = tmp_path / "proteinatlas.tsv"
    metadata.write_text(METADATA_V1)
    run_dir = tmp_path / "record_case"
    record = _verification_record(
        run_dir,
        metadata,
        metadata_sha256=runner._target_metadata_sha256(metadata),
    )

    with pytest.raises(SystemExit) as exc:
        runner._validate_stored_completed_verification_record(
            record,
            run_dir,
            preset="target-id",
            mode="fast",
        )

    assert "target metadata identity" not in str(exc.value)
