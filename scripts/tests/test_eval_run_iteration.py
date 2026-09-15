"""Unit tests for the evaluation iteration runner."""

from __future__ import annotations

import json
import importlib.util
import hashlib
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


RUN_ITERATION_MODULE = Path(__file__).resolve().parents[2] / "eval" / "run_iteration.py"
spec = importlib.util.spec_from_file_location("run_iteration_module", RUN_ITERATION_MODULE)
assert spec is not None and spec.loader is not None
run_iteration_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = run_iteration_module
spec.loader.exec_module(run_iteration_module)
run_step = run_iteration_module.run_step
ranking_metrics = run_iteration_module._ranking_metrics
input_status = run_iteration_module._input_status
provenance = run_iteration_module._provenance
skin_known_run_ledger = run_iteration_module._skin_known_run_ledger
sota_baseline_fairness = run_iteration_module._sota_baseline_fairness
sota_ablation_freeze = run_iteration_module._sota_ablation_freeze

CANDIDATE_MATRIX_MODULE = (
    Path(__file__).resolve().parents[2] / "eval" / "preregistered_candidate_matrix.py"
)
candidate_spec = importlib.util.spec_from_file_location(
    "candidate_matrix_module",
    CANDIDATE_MATRIX_MODULE,
)
assert candidate_spec is not None and candidate_spec.loader is not None
candidate_matrix_module = importlib.util.module_from_spec(candidate_spec)
sys.modules[candidate_spec.name] = candidate_matrix_module
candidate_spec.loader.exec_module(candidate_matrix_module)


def write_activity_retrieval_gate(tmp_path: Path) -> Path:
    from scripts.tests.test_validate_activity_retrieval_gate import _contract
    from scripts.validate_activity_retrieval_gate import create_gate

    contract_root = tmp_path / "activity-gate-contract"
    contract_root.mkdir(exist_ok=True)
    gate = contract_root / "activity_retrieval_final_gate.flag"
    create_gate(**_contract(contract_root), out_gate=gate)
    return gate


def run_iteration(tmp_path: Path, extra_args: list[str] | None = None) -> subprocess.CompletedProcess[str]:
    manifest = tmp_path / "iteration.json"
    return subprocess.run(
        [
            sys.executable,
            "eval/run_iteration.py",
            "--eval-dir", str(tmp_path / "eval"),
            "--rankings-dir", str(tmp_path / "rankings"),
            "--out-manifest", str(manifest),
            *(extra_args or []),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_direct_exact_input_resolution_prefers_explicit_then_legacy_then_canonical(
    tmp_path: Path,
) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    project_root = tmp_path / "project"
    explicit = tmp_path / "explicit.smi"
    explicit_manifest = tmp_path / "explicit.json"

    assert run_iteration_module._resolve_direct_exact_inputs(
        eval_dir=eval_dir,
        explicit_reference=explicit,
        explicit_manifest=explicit_manifest,
        project_root=project_root,
    ) == (explicit, explicit_manifest)

    legacy = eval_dir / "direct_exact_reference.smi"
    legacy.write_text("CCO known\n")
    assert run_iteration_module._resolve_direct_exact_inputs(
        eval_dir=eval_dir,
        explicit_reference=None,
        explicit_manifest=None,
        project_root=project_root,
    ) == (legacy, None)

    legacy.unlink()
    canonical = project_root / "data/discovery_aliases"
    assert run_iteration_module._resolve_direct_exact_inputs(
        eval_dir=eval_dir,
        explicit_reference=None,
        explicit_manifest=None,
        project_root=project_root,
    ) == (
        canonical / "direct_exact_reference.smi",
        canonical / "manifest.json",
    )


def test_sealed_leakage_status_forwards_direct_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit_path = tmp_path / "discovery_leakage_audit.json"
    direct_manifest = tmp_path / "manifest.json"
    audit_path.write_text(json.dumps({
        "schema_version": "skinscout.discovery-leakage-audit.v2",
        "status": "ok",
        "binding_sha256": "a" * 64,
        "execution_challenge": "b" * 64,
        "direct_exact_count": 0,
        "incomplete_audit_count": 0,
        "neutral_exclusion_count": 0,
        "counts": {"survivor_rows": 1},
    }))
    captured: dict[str, object] = {}

    def fake_validate(_payload: object, **kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(
        run_iteration_module,
        "validate_sealed_discovery_audit_sources",
        fake_validate,
    )
    status = run_iteration_module._sealed_leakage_status(
        audit_path,
        expected_sources={},
        expected_thresholds=run_iteration_module.LeakageThresholds(
            0.30,
            0.50,
            0.50,
        ),
        expected_execution_challenge="b" * 64,
        direct_exact_manifest=direct_manifest,
    )

    assert status["status"] == "ok"
    assert captured["direct_exact_manifest"] == direct_manifest


def cold_start_metric_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "mode": "comprehensive",
        "n_cold_targets": 10,
        "n_truth_in_cold": 1,
        "passes_threshold": True,
        "recall@1": 1.0,
        "recall@5": 1.0,
        "recall@10": 1.0,
        "recall@50": 1.0,
    }
    row.update(overrides)
    return row


def test_run_iteration_fails_when_no_steps_are_runnable(tmp_path: Path) -> None:
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')
    res = run_iteration(tmp_path)

    assert res.returncode != 0
    assert "No evaluation steps were runnable" in res.stderr
    payload = json.loads(manifest.read_text())
    assert "stale" not in payload
    assert payload["n_steps"] == 0
    assert payload["n_failed"] == 0


def test_run_iteration_allows_empty_manifest_only_when_explicit(tmp_path: Path) -> None:
    manifest = tmp_path / "iteration.json"
    res = run_iteration(tmp_path, ["--allow-empty-iteration"])

    assert res.returncode == 0
    payload = json.loads(manifest.read_text())
    assert payload["n_steps"] == 0
    assert payload["n_failed"] == 0


def test_artifact_snapshot_rejects_duplicate_relative_paths(tmp_path: Path) -> None:
    root_a = tmp_path / "eval_a"
    root_b = tmp_path / "eval_b"
    root_a.mkdir()
    root_b.mkdir()
    (root_a / "metrics.csv").write_text("metric,value\nauc,1.0\n")
    (root_b / "metrics.csv").write_text("metric,value\nauprc,1.0\n")

    with pytest.raises(SystemExit, match="duplicate relative_path 'metrics.csv'"):
        run_iteration_module._artifact_snapshot(
            [root_a, root_b],
            tmp_path / "iteration.json",
        )


def test_run_step_removes_stale_outputs_before_running(tmp_path: Path) -> None:
    output = tmp_path / "stale.csv"
    output.write_text("passes_threshold\nTrue\n")

    step = run_step(
        "stale_output_guard",
        [sys.executable, "-c", "import sys; sys.exit(1)"],
        [output],
    )

    assert step.status == "failed"
    assert not output.exists()


def test_run_step_rejects_empty_success_output(tmp_path: Path) -> None:
    output = tmp_path / "empty.csv"

    step = run_step(
        "empty_output_guard",
        [sys.executable, "-c", f"from pathlib import Path; Path({str(output)!r}).touch()"],
        [output],
    )

    assert step.status == "failed"
    assert output.exists()
    assert output.stat().st_size == 0


def test_run_iteration_rejects_precomputed_threshold_failures(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    pd.DataFrame([cold_start_metric_row(
        passes_threshold=False,
        **{"recall@50": 0.0},
    )]).to_csv(eval_dir / "cold_start_comprehensive.csv", index=False)

    res = run_iteration(tmp_path, ["--allow-empty-iteration"])

    assert res.returncode != 0
    assert "Evaluation metric threshold failure" in res.stderr
    assert "cold_start_comprehensive.csv" in res.stderr
    payload = json.loads((tmp_path / "iteration.json").read_text())
    assert payload["threshold_status"]["status"] == "failed"
    assert payload["threshold_status"]["n_failed"] == 1
    assert not payload["ranking_metrics"]["cold_start_comprehensive"]["passes_threshold"]


def test_run_iteration_threshold_failure_flag_is_diagnostic(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    pd.DataFrame([cold_start_metric_row(
        passes_threshold=False,
        **{"recall@50": 0.0},
    )]).to_csv(eval_dir / "cold_start_comprehensive.csv", index=False)

    res = run_iteration(
        tmp_path,
        ["--allow-empty-iteration", "--allow-threshold-failure"],
    )

    assert res.returncode == 0, res.stderr
    payload = json.loads((tmp_path / "iteration.json").read_text())
    assert payload["threshold_status"]["status"] == "failed"
    assert payload["threshold_status"]["failures"][0]["n_failed_rows"] == 1


def test_run_iteration_rejects_blank_threshold_values(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    pd.DataFrame([cold_start_metric_row(passes_threshold=None)]).to_csv(
        eval_dir / "cold_start_comprehensive.csv",
        index=False,
    )

    res = run_iteration(tmp_path, ["--allow-empty-iteration"])

    assert res.returncode != 0
    assert "Evaluation metric threshold failure" in res.stderr
    payload = json.loads((tmp_path / "iteration.json").read_text())
    failure = payload["threshold_status"]["failures"][0]
    assert failure["n_failed_rows"] == 1
    assert failure["n_invalid_rows"] == 1
    assert not payload["ranking_metrics"]["cold_start_comprehensive"]["passes_threshold"]


def test_run_iteration_rejects_thresholded_metric_without_threshold_column(
    tmp_path: Path,
) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    row = cold_start_metric_row()
    row.pop("passes_threshold")
    pd.DataFrame([row]).to_csv(
        eval_dir / "cold_start_comprehensive.csv",
        index=False,
    )

    res = run_iteration(tmp_path, ["--allow-empty-iteration"])

    assert res.returncode != 0
    assert "Evaluation metric threshold failure" in res.stderr
    payload = json.loads((tmp_path / "iteration.json").read_text())
    assert payload["threshold_status"]["status"] == "failed"
    assert payload["threshold_status"]["n_failed"] == 1
    assert payload["threshold_status"]["n_missing_required"] == 1
    assert payload["threshold_status"]["missing_required"][0]["path"].endswith(
        "cold_start_comprehensive.csv"
    )


def test_run_iteration_rejects_nonfinite_ranking_metric(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')
    pd.DataFrame([cold_start_metric_row(**{"recall@50": float("inf")})]).to_csv(
        eval_dir / "cold_start_comprehensive.csv",
        index=False,
    )

    res = run_iteration(tmp_path, ["--allow-empty-iteration"])

    assert res.returncode != 0
    assert "cold_start_comprehensive.csv column 'recall@50' must be finite" in res.stderr
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_rejects_boolean_ranking_metric(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')
    pd.DataFrame([cold_start_metric_row(**{"recall@50": True})]).to_csv(
        eval_dir / "cold_start_comprehensive.csv",
        index=False,
    )

    res = run_iteration(tmp_path, ["--allow-empty-iteration"])

    assert res.returncode != 0
    assert "cold_start_comprehensive.csv column 'recall@50' must be numeric" in res.stderr
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_rejects_boolean_like_string_ranking_metric(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')
    row = cold_start_metric_row(**{"recall@50": "True"})
    pd.DataFrame([row]).to_csv(eval_dir / "cold_start_comprehensive.csv", index=False)

    res = run_iteration(tmp_path, ["--allow-empty-iteration"])

    assert res.returncode != 0
    assert "cold_start_comprehensive.csv column 'recall@50' must be numeric" in res.stderr
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_rejects_out_of_range_fraction_metric(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')
    pd.DataFrame([cold_start_metric_row(**{"recall@50": 1.2})]).to_csv(
        eval_dir / "cold_start_comprehensive.csv",
        index=False,
    )

    res = run_iteration(tmp_path, ["--allow-empty-iteration"])

    assert res.returncode != 0
    assert "cold_start_comprehensive.csv column 'recall@50' must be in [0, 1]" in res.stderr
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_rejects_cold_start_metric_without_required_columns(
    tmp_path: Path,
) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')
    pd.DataFrame([{
        "mode": "comprehensive",
        "passes_threshold": True,
    }]).to_csv(eval_dir / "cold_start_comprehensive.csv", index=False)

    res = run_iteration(tmp_path, ["--allow-empty-iteration"])

    assert res.returncode != 0
    assert "cold_start_comprehensive.csv missing required metric columns" in res.stderr
    assert "n_cold_targets" in res.stderr
    assert "n_truth_in_cold" in res.stderr
    assert "recall@50" in res.stderr
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_rejects_multirow_cold_start_metric(
    tmp_path: Path,
) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')
    pd.DataFrame([
        cold_start_metric_row(),
        cold_start_metric_row(**{"recall@50": 0.0, "passes_threshold": False}),
    ]).to_csv(eval_dir / "cold_start_comprehensive.csv", index=False)

    res = run_iteration(tmp_path, ["--allow-empty-iteration"])

    assert res.returncode != 0
    assert "cold_start_comprehensive.csv must contain exactly one metric row" in res.stderr
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_rejects_empty_snapshot_artifact(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / "diagnostic.csv").write_text("")
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')

    res = run_iteration(tmp_path, ["--allow-empty-iteration"])

    assert res.returncode != 0
    assert "artifact snapshot contains empty file" in res.stderr
    assert "diagnostic.csv" in res.stderr
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_records_configured_data_snapshot_checksums(
    tmp_path: Path,
) -> None:
    refs = tmp_path / "refs"
    refs.mkdir()
    seq_db = refs / "seq_db"
    seq_db.mkdir()
    (seq_db / "index").write_text("seq-index\n")
    ligands = refs / "training_ligands.smi"
    ligands.write_text("CCO ligand-1\n")
    holo = refs / "training_holo.csv"
    holo.write_text("target_id,pocket_sucos\nP1,0.1\n")
    target_classes = refs / "target_classes.parquet"
    pd.DataFrame([{"uniprot": "P1", "class": "kinase"}]).to_parquet(target_classes)
    chembl_fp = refs / "fp.parquet"
    pd.DataFrame([{"chembl_id": "CHEMBL1", "fp": "101"}]).to_parquet(chembl_fp)

    res = run_iteration(
        tmp_path,
        [
            "--allow-empty-iteration",
            "--allow-incomplete-eval-inputs",
            "--target-classes", str(target_classes),
            "--chembl-fp-parquet", str(chembl_fp),
            "--training-seq-db", str(seq_db),
            "--training-ligands", str(ligands),
            "--training-holo", str(holo),
        ],
    )

    assert res.returncode == 0, res.stderr
    payload = json.loads((tmp_path / "iteration.json").read_text())
    snapshot = payload["data_snapshot"]
    assert snapshot["workflow_config"]["status"] == "present"
    assert snapshot["target_classes"]["sha256"] == hashlib.sha256(
        target_classes.read_bytes()
    ).hexdigest()
    assert snapshot["chembl_fingerprints"]["sha256"] == hashlib.sha256(
        chembl_fp.read_bytes()
    ).hexdigest()
    assert snapshot["training_ligands"]["sha256"] == hashlib.sha256(
        ligands.read_bytes()
    ).hexdigest()
    assert snapshot["training_holo"]["sha256"] == hashlib.sha256(
        holo.read_bytes()
    ).hexdigest()
    assert snapshot["training_sequence_db"]["kind"] == "directory"
    assert snapshot["training_sequence_db"]["entries"][0]["relative_path"] == "index"
    assert snapshot["training_sequence_db"]["entries"][0]["sha256"] == hashlib.sha256(
        (seq_db / "index").read_bytes()
    ).hexdigest()


def test_run_iteration_records_missing_data_snapshot_references(
    tmp_path: Path,
) -> None:
    res = run_iteration(
        tmp_path,
        [
            "--allow-empty-iteration",
            "--chembl-fp-parquet",
            str(tmp_path / "refs" / "missing_fp.parquet"),
            "--training-seq-db",
            str(tmp_path / "refs" / "missing_seq_db"),
            "--training-ligands",
            str(tmp_path / "refs" / "missing_ligands.smi"),
            "--training-holo",
            str(tmp_path / "refs" / "missing_holo.tar"),
        ],
    )

    assert res.returncode == 0, res.stderr
    payload = json.loads((tmp_path / "iteration.json").read_text())
    assert payload["input_runs"] == {
        "status": "missing",
        "manifest": str(tmp_path / "eval" / "collected_runs.json"),
        "n_runs": 0,
        "runs": [],
    }
    assert payload["data_snapshot"]["workflow_config"]["status"] == "present"
    assert payload["data_snapshot"]["target_classes"]["status"] == "missing"
    assert payload["data_snapshot"]["chembl_fingerprints"]["status"] == "missing"
    assert payload["data_snapshot"]["training_sequence_db"]["status"] == "missing"
    assert payload["data_snapshot"]["training_ligands"]["status"] == "missing"
    assert payload["data_snapshot"]["training_holo"]["status"] == "missing"


def test_run_iteration_provenance_requires_workflow_config(tmp_path: Path) -> None:
    missing_config = tmp_path / "missing.yaml"

    with pytest.raises(SystemExit, match="Workflow config is required"):
        provenance(missing_config)


def test_run_iteration_provenance_rejects_unknown_git_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("paths: {}\n")
    monkeypatch.setattr(run_iteration_module, "_git_commit", lambda: "unknown")

    with pytest.raises(SystemExit, match="Git commit provenance must be a 40-hex SHA"):
        provenance(config)


def test_run_iteration_provenance_requires_rdkit_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("paths: {}\n")
    monkeypatch.setattr(run_iteration_module, "_git_commit", lambda: "a" * 40)
    monkeypatch.setattr(
        run_iteration_module,
        "_tool_versions",
        lambda: {
            "python": "3.11.15",
            "platform": "Linux-test",
            "rdkit": "missing",
        },
    )

    with pytest.raises(SystemExit, match="available rdkit version"):
        provenance(config)


def test_run_iteration_rejects_empty_data_snapshot_file(tmp_path: Path) -> None:
    target_classes = tmp_path / "target_classes.parquet"
    target_classes.write_text("")
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')

    res = run_iteration(
        tmp_path,
        ["--allow-empty-iteration", "--target-classes", str(target_classes)],
    )

    assert res.returncode != 0
    assert "Data snapshot contains empty file" in res.stderr
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_rejects_empty_data_snapshot_directory(tmp_path: Path) -> None:
    seq_db = tmp_path / "seq_db"
    seq_db.mkdir()
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')

    res = run_iteration(
        tmp_path,
        ["--allow-empty-iteration", "--training-seq-db", str(seq_db)],
    )

    assert res.returncode != 0
    assert "Data snapshot contains empty directory" in res.stderr
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_rejects_duplicate_data_snapshot_file_paths(
    tmp_path: Path,
) -> None:
    refs = tmp_path / "refs"
    refs.mkdir()
    shared = refs / "shared.parquet"
    pd.DataFrame([{"id": "P1", "value": "101"}]).to_parquet(shared)
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')

    res = run_iteration(
        tmp_path,
        [
            "--allow-empty-iteration",
            "--allow-incomplete-eval-inputs",
            "--target-classes", str(shared),
            "--chembl-fp-parquet", str(shared),
        ],
    )

    assert res.returncode != 0
    assert "Data snapshot contains duplicate file path" in res.stderr
    assert "chembl_fingerprints" in res.stderr
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_rejects_data_snapshot_file_inside_directory(
    tmp_path: Path,
) -> None:
    refs = tmp_path / "refs"
    seq_db = refs / "seq_db"
    seq_db.mkdir(parents=True)
    shared = seq_db / "index"
    shared.write_text("seq-index\n")
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')

    res = run_iteration(
        tmp_path,
        [
            "--allow-empty-iteration",
            "--allow-incomplete-eval-inputs",
            "--target-classes", str(shared),
            "--training-seq-db", str(seq_db),
        ],
    )

    assert res.returncode != 0
    assert "Data snapshot contains duplicate file path" in res.stderr
    assert "training_sequence_db" in res.stderr
    assert "target_classes" in res.stderr
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_records_collected_input_runs(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    collected = eval_dir / "collected_runs.json"
    copied = tmp_path / "copied" / "case__ranked_targets_v3.csv"
    run_dir = tmp_path / "results" / "runs" / "case"
    source = run_dir / "03_targets" / "ranked_targets_v3.csv"
    copied.parent.mkdir(parents=True)
    source.parent.mkdir(parents=True)
    source.write_text("target_id,final_score,source_count,sources\nP1,0.9,2,autodock;gnina\n")
    copied.write_text(source.read_text())
    copied_record = {
        "path": str(copied),
        "source_path": str(source),
        "source_bytes": source.stat().st_size,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "bytes": copied.stat().st_size,
        "sha256": hashlib.sha256(copied.read_bytes()).hexdigest(),
    }
    collected.write_text(json.dumps({
        "out_dir": str(eval_dir),
        "rankings_dir": str(tmp_path / "rankings"),
        "n_runs": 1,
        "runs": [{
            "run_id": "case",
            "run_dir": str(run_dir),
            "canonical_smiles": "CCO",
            "ranked_v3": str(source),
            "leakage_rows": [{"run_id": "case", "uniprot": "P1", "smiles": "CCO"}],
            "copied": [str(copied)],
            "copied_artifacts": [copied_record],
        }],
    }))

    res = run_iteration(tmp_path, ["--allow-empty-iteration"])

    assert res.returncode == 0, res.stderr
    payload = json.loads((tmp_path / "iteration.json").read_text())
    assert payload["input_runs"] == {
        "status": "present",
        "manifest": str(collected),
        "out_dir": str(eval_dir),
        "rankings_dir": str(tmp_path / "rankings"),
        "n_runs": 1,
        "runs": [{
            "run_id": "case",
            "run_dir": str(run_dir),
            "canonical_smiles": "CCO",
            "ranked_v3": str(source),
            "n_leakage_rows": 1,
            "copied": [str(copied)],
            "copied_artifacts": [copied_record],
        }],
    }


def test_run_iteration_rejects_collected_run_without_copied_artifacts(
    tmp_path: Path,
) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')
    copied = tmp_path / "rankings" / "cosmetic_retro" / "case__ranked_targets_v3.csv"
    run_dir = tmp_path / "results" / "runs" / "case"
    source = run_dir / "03_targets" / "ranked_targets_v3.csv"
    copied.parent.mkdir(parents=True)
    source.parent.mkdir(parents=True)
    source.write_text("target_id,final_score,source_count,sources\nP1,0.9,2,autodock;gnina\n")
    copied.write_text(source.read_text())
    (eval_dir / "collected_runs.json").write_text(json.dumps({
        "n_runs": 1,
        "runs": [{
            "run_id": "case",
            "run_dir": str(run_dir),
            "canonical_smiles": "CCO",
            "copied": [str(copied)],
        }],
    }))

    res = run_iteration(tmp_path, ["--allow-empty-iteration"])

    assert res.returncode != 0
    assert "copied_artifacts" in res.stderr
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_rejects_collected_run_duplicate_copied_path(
    tmp_path: Path,
) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')
    copied = tmp_path / "rankings" / "cosmetic_retro" / "case__ranked_targets_v3.csv"
    run_dir = tmp_path / "results" / "runs" / "case"
    source = run_dir / "03_targets" / "ranked_targets_v3.csv"
    copied.parent.mkdir(parents=True)
    source.parent.mkdir(parents=True)
    source.write_text("target_id,final_score,source_count,sources\nP1,0.9,2,autodock;gnina\n")
    copied.write_text(source.read_text())
    copied_record = {
        "path": str(copied),
        "source_path": str(source),
        "source_bytes": source.stat().st_size,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "bytes": copied.stat().st_size,
        "sha256": hashlib.sha256(copied.read_bytes()).hexdigest(),
    }
    (eval_dir / "collected_runs.json").write_text(json.dumps({
        "n_runs": 1,
        "runs": [{
            "run_id": "case",
            "run_dir": str(run_dir),
            "canonical_smiles": "CCO",
            "leakage_rows": [{"run_id": "case", "uniprot": "P1", "smiles": "CCO"}],
            "copied": [str(copied), str(copied)],
            "copied_artifacts": [copied_record, dict(copied_record)],
        }],
    }))

    res = run_iteration(tmp_path, ["--allow-empty-iteration"])

    assert res.returncode != 0
    assert "run field 'copied' contains duplicate paths" in res.stderr
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_rejects_collected_run_canonical_duplicate_copied_path(
    tmp_path: Path,
) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')
    copied = tmp_path / "rankings" / "cosmetic_retro" / "case__ranked_targets_v3.csv"
    copied_alias = f"{copied.parent}/./{copied.name}"
    run_dir = tmp_path / "results" / "runs" / "case"
    source = run_dir / "03_targets" / "ranked_targets_v3.csv"
    copied.parent.mkdir(parents=True)
    source.parent.mkdir(parents=True)
    source.write_text("target_id,final_score,source_count,sources\nP1,0.9,2,autodock;gnina\n")
    copied.write_text(source.read_text())
    copied_record = {
        "path": str(copied),
        "source_path": str(source),
        "source_bytes": source.stat().st_size,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "bytes": copied.stat().st_size,
        "sha256": hashlib.sha256(copied.read_bytes()).hexdigest(),
    }
    copied_alias_record = dict(copied_record)
    copied_alias_record["path"] = copied_alias
    (eval_dir / "collected_runs.json").write_text(json.dumps({
        "out_dir": str(eval_dir),
        "rankings_dir": str(tmp_path / "rankings"),
        "n_runs": 1,
        "runs": [{
            "run_id": "case",
            "run_dir": str(run_dir),
            "canonical_smiles": "CCO",
            "leakage_rows": [{"run_id": "case", "uniprot": "P1", "smiles": "CCO"}],
            "copied": [str(copied), copied_alias],
            "copied_artifacts": [copied_record, copied_alias_record],
        }],
    }))

    res = run_iteration(tmp_path, ["--allow-empty-iteration"])

    assert res.returncode != 0
    assert "run field 'copied' contains duplicate paths" in res.stderr
    assert str(copied.resolve()) in res.stderr
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_rejects_collected_run_copied_artifact_digest_mismatch(
    tmp_path: Path,
) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')
    copied = tmp_path / "rankings" / "cosmetic_retro" / "case__ranked_targets_v3.csv"
    run_dir = tmp_path / "results" / "runs" / "case"
    source = run_dir / "03_targets" / "ranked_targets_v3.csv"
    copied.parent.mkdir(parents=True)
    source.parent.mkdir(parents=True)
    source.write_text("target_id,final_score,source_count,sources\nP1,0.9,2,autodock;gnina\n")
    copied.write_text(source.read_text())
    (eval_dir / "collected_runs.json").write_text(json.dumps({
        "n_runs": 1,
        "runs": [{
            "run_id": "case",
            "run_dir": str(run_dir),
            "canonical_smiles": "CCO",
            "leakage_rows": [{"run_id": "case", "uniprot": "P1", "smiles": "CCO"}],
            "copied": [str(copied)],
            "copied_artifacts": [{
                "path": str(copied),
                "source_path": str(source),
                "source_bytes": source.stat().st_size,
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "bytes": copied.stat().st_size,
                "sha256": "0" * 64,
            }],
        }],
    }))

    res = run_iteration(tmp_path, ["--allow-empty-iteration"])

    assert res.returncode != 0
    assert "copied_artifacts sha256 does not match existing file" in res.stderr
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_rejects_collected_run_source_artifact_digest_mismatch(
    tmp_path: Path,
) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')
    copied = tmp_path / "rankings" / "cosmetic_retro" / "case__ranked_targets_v3.csv"
    run_dir = tmp_path / "results" / "runs" / "case"
    source = run_dir / "03_targets" / "ranked_targets_v3.csv"
    copied.parent.mkdir(parents=True)
    source.parent.mkdir(parents=True)
    source.write_text("target_id,final_score,source_count,sources\nP1,0.9,2,autodock;gnina\n")
    copied.write_text(source.read_text())
    (eval_dir / "collected_runs.json").write_text(json.dumps({
        "n_runs": 1,
        "runs": [{
            "run_id": "case",
            "run_dir": str(run_dir),
            "canonical_smiles": "CCO",
            "leakage_rows": [{"run_id": "case", "uniprot": "P1", "smiles": "CCO"}],
            "copied": [str(copied)],
            "copied_artifacts": [{
                "path": str(copied),
                "source_path": str(source),
                "source_bytes": source.stat().st_size,
                "source_sha256": "0" * 64,
                "bytes": copied.stat().st_size,
                "sha256": hashlib.sha256(copied.read_bytes()).hexdigest(),
            }],
        }],
    }))

    res = run_iteration(tmp_path, ["--allow-empty-iteration"])

    assert res.returncode != 0
    assert "copied_artifacts source_sha256 does not match existing source" in res.stderr
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_rejects_collected_run_source_outside_run_dir(
    tmp_path: Path,
) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')
    copied = tmp_path / "rankings" / "cosmetic_retro" / "case__ranked_targets_v3.csv"
    run_dir = tmp_path / "results" / "runs" / "case"
    source = tmp_path / "outside" / "ranked_targets_v3.csv"
    copied.parent.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    source.parent.mkdir(parents=True)
    source.write_text("target_id,final_score,source_count,sources\nP1,0.9,2,autodock;gnina\n")
    copied.write_text(source.read_text())
    (eval_dir / "collected_runs.json").write_text(json.dumps({
        "n_runs": 1,
        "runs": [{
            "run_id": "case",
            "run_dir": str(run_dir),
            "canonical_smiles": "CCO",
            "leakage_rows": [{"run_id": "case", "uniprot": "P1", "smiles": "CCO"}],
            "copied": [str(copied)],
            "copied_artifacts": [{
                "path": str(copied),
                "source_path": str(source),
                "source_bytes": source.stat().st_size,
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "bytes": copied.stat().st_size,
                "sha256": hashlib.sha256(copied.read_bytes()).hexdigest(),
            }],
        }],
    }))

    res = run_iteration(tmp_path, ["--allow-empty-iteration"])

    assert res.returncode != 0
    assert "source_path must be inside run_dir" in res.stderr
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_rejects_collected_run_missing_copied_artifact(
    tmp_path: Path,
) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')
    missing = tmp_path / "rankings" / "cosmetic_retro" / "case__ranked_targets_v3.csv"
    run_dir = tmp_path / "results" / "runs" / "case"
    run_dir.mkdir(parents=True)
    (eval_dir / "collected_runs.json").write_text(json.dumps({
        "n_runs": 1,
        "runs": [{
            "run_id": "case",
            "run_dir": str(run_dir),
            "canonical_smiles": "CCO",
            "copied": [str(missing)],
            "copied_artifacts": [{
                "path": str(missing),
                "source_path": "results/runs/case/03_targets/ranked_targets_v3.csv",
                "source_bytes": 1,
                "source_sha256": "0" * 64,
                "bytes": 1,
                "sha256": "0" * 64,
            }],
        }],
    }))

    res = run_iteration(tmp_path, ["--allow-empty-iteration"])

    assert res.returncode != 0
    assert "Collected-runs manifest copied artifact is missing" in res.stderr
    assert str(missing) in res.stderr
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_uses_configured_collected_runs_manifest(tmp_path: Path) -> None:
    collected = tmp_path / "custom_collected.json"
    copied = tmp_path / "copied" / "custom__ranked_targets_v3.csv"
    run_dir = tmp_path / "results" / "runs" / "custom"
    source = run_dir / "03_targets" / "ranked_targets_v3.csv"
    copied.parent.mkdir(parents=True)
    source.parent.mkdir(parents=True)
    source.write_text("target_id,final_score,source_count,sources\nP1,0.9,2,autodock;gnina\n")
    copied.write_text(source.read_text())
    collected.write_text(json.dumps({
        "n_runs": 1,
        "runs": [{
            "run_id": "custom",
            "run_dir": str(run_dir),
            "canonical_smiles": "CCN",
            "copied": [str(copied)],
            "copied_artifacts": [{
                "path": str(copied),
                "source_path": str(source),
                "source_bytes": source.stat().st_size,
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "bytes": copied.stat().st_size,
                "sha256": hashlib.sha256(copied.read_bytes()).hexdigest(),
            }],
        }],
    }))

    res = run_iteration(
        tmp_path,
        [
            "--allow-empty-iteration",
            "--collected-runs-manifest", str(collected),
        ],
    )

    assert res.returncode == 0, res.stderr
    payload = json.loads((tmp_path / "iteration.json").read_text())
    assert payload["input_runs"]["manifest"] == str(collected)
    assert payload["input_runs"]["runs"][0]["run_id"] == "custom"
    assert payload["input_runs"]["runs"][0]["canonical_smiles"] == "CCN"


def test_run_iteration_rejects_collected_run_without_run_id(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')
    (eval_dir / "collected_runs.json").write_text(json.dumps({
        "runs": [{"run_id": " ", "canonical_smiles": "CCO"}],
    }))

    res = run_iteration(tmp_path, ["--allow-empty-iteration"])

    assert res.returncode != 0
    assert "missing non-empty 'run_id'" in res.stderr
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_rejects_collected_run_invalid_canonical_smiles(
    tmp_path: Path,
) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    run_dir = tmp_path / "runs" / "case"
    run_dir.mkdir(parents=True)
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')
    (eval_dir / "collected_runs.json").write_text(json.dumps({
        "runs": [{
            "run_id": "case",
            "run_dir": str(run_dir),
            "canonical_smiles": "not-a-smiles",
            "copied": [],
            "copied_artifacts": [],
        }],
    }))

    res = run_iteration(tmp_path, ["--allow-empty-iteration"])

    assert res.returncode != 0
    assert (
        "Collected-runs manifest run canonical_smiles at row index 0 "
        "must be a parseable SMILES: not-a-smiles"
    ) in res.stderr
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_rejects_duplicate_collected_run_ids(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    run_dir_a = tmp_path / "runs" / "case_a"
    run_dir_b = tmp_path / "runs" / "case_b"
    run_dir_a.mkdir(parents=True)
    run_dir_b.mkdir(parents=True)
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')
    (eval_dir / "collected_runs.json").write_text(json.dumps({
        "runs": [
            {
                "run_id": "case",
                "run_dir": str(run_dir_a),
                "canonical_smiles": "CCO",
                "copied": [],
                "copied_artifacts": [],
            },
            {
                "run_id": "case",
                "run_dir": str(run_dir_b),
                "canonical_smiles": "CCN",
                "copied": [],
                "copied_artifacts": [],
            },
        ],
    }))

    res = run_iteration(tmp_path, ["--allow-empty-iteration"])

    assert res.returncode != 0
    assert "duplicate run_id values: case" in res.stderr
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_rejects_duplicate_collected_run_dirs(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    rankings_dir = tmp_path / "rankings"
    eval_dir.mkdir()
    run_dir = tmp_path / "runs" / "case"
    source_dir = run_dir / "03_targets"
    copied_dir = rankings_dir / "cosmetic_retro"
    source_dir.mkdir(parents=True)
    copied_dir.mkdir(parents=True)
    source_a = source_dir / "ranked_targets_v3.csv"
    source_b = source_dir / "ranked_targets_v3_with_efficacy.csv"
    source_a.write_text("target_id,final_score,source_count,sources\nP1,0.9,2,autodock;gnina\n")
    source_b.write_text(
        "target_id,final_score,source_count,sources,efficacy_top1\n"
        "P1,0.9,2,autodock;gnina,barrier\n"
    )
    copied_a = copied_dir / "case_a__ranked_targets_v3.csv"
    copied_b = copied_dir / "case_b__ranked_targets_v3.csv"
    copied_a.write_text(source_a.read_text())
    copied_b.write_text(source_b.read_text())
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')

    def copied_record(copied: Path, source: Path) -> dict[str, object]:
        return {
            "path": str(copied),
            "source_path": str(source),
            "source_bytes": source.stat().st_size,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "bytes": copied.stat().st_size,
            "sha256": hashlib.sha256(copied.read_bytes()).hexdigest(),
        }

    (eval_dir / "collected_runs.json").write_text(json.dumps({
        "out_dir": str(eval_dir),
        "rankings_dir": str(rankings_dir),
        "n_runs": 2,
        "runs": [
            {
                "run_id": "case_a",
                "run_dir": str(run_dir),
                "canonical_smiles": "CCO",
                "leakage_rows": [{"run_id": "case_a", "uniprot": "P1", "smiles": "CCO"}],
                "copied": [str(copied_a)],
                "copied_artifacts": [copied_record(copied_a, source_a)],
            },
            {
                "run_id": "case_b",
                "run_dir": str(run_dir),
                "canonical_smiles": "CCN",
                "leakage_rows": [{"run_id": "case_b", "uniprot": "P1", "smiles": "CCN"}],
                "copied": [str(copied_b)],
                "copied_artifacts": [copied_record(copied_b, source_b)],
            },
        ],
    }))

    res = run_iteration(tmp_path, ["--allow-empty-iteration"])

    assert res.returncode != 0
    assert "duplicate run_dir values" in res.stderr
    assert str(run_dir.resolve()) in res.stderr
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_rejects_collected_run_count_mismatch(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')
    (eval_dir / "collected_runs.json").write_text(json.dumps({
        "n_runs": 2,
        "runs": [{"run_id": "case", "canonical_smiles": "CCO"}],
    }))

    res = run_iteration(tmp_path, ["--allow-empty-iteration"])

    assert res.returncode != 0
    assert "field 'n_runs' does not match runs length" in res.stderr
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_rejects_malformed_eval_csv(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')
    (eval_dir / "analog_quality.csv").write_text('"unterminated\n')

    res = run_iteration(tmp_path, ["--allow-empty-iteration"])

    assert res.returncode != 0
    assert "analog_quality.csv failed to parse" in res.stderr
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_rejects_nonfinite_top_target_rationale_score(
    tmp_path: Path,
) -> None:
    rankings_dir = tmp_path / "rankings"
    rankings_dir.mkdir()
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')
    pd.DataFrame([{
        "target_id": "P1",
        "final_score": float("inf"),
    }]).to_csv(rankings_dir / "cold_start__comprehensive.csv", index=False)

    res = run_iteration(tmp_path, ["--allow-empty-iteration"])

    assert res.returncode != 0
    assert "cold_start__comprehensive.csv column 'final_score' must be finite" in res.stderr
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_rejects_top_target_rationale_unsorted_scores(
    tmp_path: Path,
) -> None:
    rankings_dir = tmp_path / "rankings"
    rankings_dir.mkdir()
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')
    pd.DataFrame([
        {
            "target_id": "P1",
            "final_score": 0.80,
            "source_count": 3,
            "sources": "autodock;gnina;rtmscore",
        },
        {
            "target_id": "P2",
            "final_score": 0.95,
            "source_count": 3,
            "sources": "autodock;gnina;rtmscore",
        },
    ]).to_csv(rankings_dir / "cold_start__comprehensive.csv", index=False)

    res = run_iteration(
        tmp_path,
        ["--allow-empty-iteration", "--allow-incomplete-eval-inputs"],
    )

    assert res.returncode != 0
    assert (
        "cold_start__comprehensive.csv top row must have the best "
        "final_score"
    ) in res.stderr
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_rejects_top_target_rationale_tied_top_scores(
    tmp_path: Path,
) -> None:
    rankings_dir = tmp_path / "rankings"
    rankings_dir.mkdir()
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')
    pd.DataFrame([
        {
            "target_id": "P1",
            "final_score": 0.95,
            "source_count": 3,
            "sources": "autodock;gnina;rtmscore",
        },
        {
            "target_id": "P2",
            "final_score": 0.95,
            "source_count": 3,
            "sources": "autodock;gnina;rtmscore",
        },
    ]).to_csv(rankings_dir / "cold_start__comprehensive.csv", index=False)

    res = run_iteration(
        tmp_path,
        ["--allow-empty-iteration", "--allow-incomplete-eval-inputs"],
    )

    assert res.returncode != 0
    assert (
        "cold_start__comprehensive.csv top row final_score is tied"
    ) in res.stderr
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_rejects_blank_top_target_rationale_id(
    tmp_path: Path,
) -> None:
    rankings_dir = tmp_path / "rankings"
    rankings_dir.mkdir()
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')
    pd.DataFrame([{
        "target_id": " ",
        "final_score": 0.9,
    }]).to_csv(rankings_dir / "cold_start__comprehensive.csv", index=False)

    res = run_iteration(tmp_path, ["--allow-empty-iteration"])

    assert res.returncode != 0
    assert "cold_start__comprehensive.csv column 'target_id' is blank at row index 0" in res.stderr
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_rejects_duplicate_top_target_rationale_ids(
    tmp_path: Path,
) -> None:
    rankings_dir = tmp_path / "rankings"
    rankings_dir.mkdir()
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')
    pd.DataFrame([
        {
            "target_id": "P1",
            "final_score": 0.9,
        },
        {
            "target_id": "P1",
            "final_score": 0.8,
        },
    ]).to_csv(rankings_dir / "cold_start__comprehensive.csv", index=False)

    res = run_iteration(tmp_path, ["--allow-empty-iteration"])

    assert res.returncode != 0
    assert (
        "cold_start__comprehensive.csv contains duplicate target_id values: P1"
        in res.stderr
    )
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_rejects_top_target_rationale_without_target_id(
    tmp_path: Path,
) -> None:
    rankings_dir = tmp_path / "rankings"
    rankings_dir.mkdir()
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')
    pd.DataFrame([{
        "final_score": 0.9,
    }]).to_csv(rankings_dir / "cold_start__comprehensive.csv", index=False)

    res = run_iteration(tmp_path, ["--allow-empty-iteration"])

    assert res.returncode != 0
    assert "cold_start__comprehensive.csv missing required column 'target_id'" in res.stderr
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_rejects_top_target_rationale_without_score(
    tmp_path: Path,
) -> None:
    rankings_dir = tmp_path / "rankings"
    rankings_dir.mkdir()
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')
    pd.DataFrame([{
        "target_id": "P1",
        "sources": "autodock;gnina",
    }]).to_csv(rankings_dir / "cold_start__comprehensive.csv", index=False)

    res = run_iteration(tmp_path, ["--allow-empty-iteration"])

    assert res.returncode != 0
    assert (
        "cold_start__comprehensive.csv top row must include at least one "
        "score column"
    ) in res.stderr
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_rejects_missing_multi_scorer_rationale_columns(
    tmp_path: Path,
) -> None:
    rankings_dir = tmp_path / "rankings"
    rankings_dir.mkdir()
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')
    pd.DataFrame([{
        "target_id": "P1",
        "final_score": 0.9,
    }]).to_csv(rankings_dir / "cold_start__comprehensive.csv", index=False)

    res = run_iteration(tmp_path, ["--allow-empty-iteration"])

    assert res.returncode != 0
    assert (
        "cold_start__comprehensive.csv missing required top-target rationale "
        "columns ['source_count', 'sources']"
    ) in res.stderr
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_rejects_non_top_rationale_source_count_mismatch(
    tmp_path: Path,
) -> None:
    rankings_dir = tmp_path / "rankings"
    rankings_dir.mkdir()
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')
    pd.DataFrame([
        {
            "target_id": "P1",
            "final_score": 0.95,
            "source_count": 3,
            "sources": "autodock;gnina;rtmscore",
        },
        {
            "target_id": "P2",
            "final_score": 0.80,
            "source_count": 3,
            "sources": "autodock;gnina",
        },
    ]).to_csv(rankings_dir / "cold_start__comprehensive.csv", index=False)

    res = run_iteration(
        tmp_path,
        ["--allow-empty-iteration", "--allow-incomplete-eval-inputs"],
    )

    assert res.returncode != 0
    assert (
        "cold_start__comprehensive.csv source_count=3 but sources lists "
        "2 label(s) at row index 1"
    ) in res.stderr
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_allows_dti_only_rationale_without_scorer_coverage(
    tmp_path: Path,
) -> None:
    rankings_dir = tmp_path / "rankings"
    rankings_dir.mkdir()
    pd.DataFrame([{
        "target_id": "P1",
        "psichic_score": 0.9,
    }]).to_csv(rankings_dir / "cold_start__dti_only.csv", index=False)

    res = run_iteration(
        tmp_path,
        ["--allow-empty-iteration", "--allow-incomplete-eval-inputs"],
    )

    assert res.returncode == 0, res.stderr
    payload = json.loads((tmp_path / "iteration.json").read_text())
    assert payload["top_target_rationale"] == [{
        "ranking": str(rankings_dir / "cold_start__dti_only.csv"),
        "target_id": "P1",
        "score": 0.9,
        "rationale": {},
    }]


def test_run_iteration_rejects_blank_top_target_rationale_field(
    tmp_path: Path,
) -> None:
    rankings_dir = tmp_path / "rankings"
    rankings_dir.mkdir()
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')
    pd.DataFrame([{
        "target_id": "P1",
        "final_score": 0.9,
        "source_count": 1,
        "sources": " ",
    }]).to_csv(rankings_dir / "cold_start__comprehensive.csv", index=False)

    res = run_iteration(tmp_path, ["--allow-empty-iteration"])

    assert res.returncode != 0
    assert "cold_start__comprehensive.csv column 'sources' is blank in top row" in res.stderr
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_rejects_invalid_top_target_source_count(
    tmp_path: Path,
) -> None:
    rankings_dir = tmp_path / "rankings"
    rankings_dir.mkdir()
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')
    pd.DataFrame([{
        "target_id": "P1",
        "final_score": 0.9,
        "source_count": 0,
        "sources": "autodock",
    }]).to_csv(rankings_dir / "cold_start__comprehensive.csv", index=False)

    res = run_iteration(tmp_path, ["--allow-empty-iteration"])

    assert res.returncode != 0
    assert (
        "cold_start__comprehensive.csv column 'source_count' "
        "must be a positive integer"
    ) in res.stderr
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_rejects_top_target_source_count_mismatch(
    tmp_path: Path,
) -> None:
    rankings_dir = tmp_path / "rankings"
    rankings_dir.mkdir()
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')
    pd.DataFrame([{
        "target_id": "P1",
        "final_score": 0.9,
        "source_count": 4,
        "sources": "autodock",
    }]).to_csv(rankings_dir / "cold_start__comprehensive.csv", index=False)

    res = run_iteration(
        tmp_path,
        ["--allow-empty-iteration", "--allow-incomplete-eval-inputs"],
    )

    assert res.returncode != 0
    assert (
        "cold_start__comprehensive.csv source_count=4 but sources lists 1 "
        "label(s) in top row"
    ) in res.stderr
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_rejects_comprehensive_top_target_below_min_sources(
    tmp_path: Path,
) -> None:
    rankings_dir = tmp_path / "rankings"
    rankings_dir.mkdir()
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')
    pd.DataFrame([{
        "target_id": "P1",
        "final_score": 0.9,
        "source_count": 2,
        "sources": "autodock;gnina",
    }]).to_csv(rankings_dir / "cold_start__comprehensive.csv", index=False)

    res = run_iteration(
        tmp_path,
        ["--allow-empty-iteration", "--allow-incomplete-eval-inputs"],
    )

    assert res.returncode != 0
    assert (
        "cold_start__comprehensive.csv top row requires source_count >= 3 "
        "for claim-quality rationale"
    ) in res.stderr
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_rejects_cosmetic_top_target_below_min_sources(
    tmp_path: Path,
) -> None:
    retro_dir = tmp_path / "rankings" / "cosmetic_retro"
    retro_dir.mkdir(parents=True)
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')
    pd.DataFrame([{
        "target_id": "P1",
        "final_score": 0.9,
        "source_count": 1,
        "sources": "autodock",
    }]).to_csv(retro_dir / "case__ranked_targets_v3.csv", index=False)

    res = run_iteration(
        tmp_path,
        [
            "--allow-empty-iteration",
            "--allow-incomplete-eval-inputs",
            "--allow-threshold-failure",
        ],
    )

    assert res.returncode != 0
    assert (
        "case__ranked_targets_v3.csv top row requires source_count >= 2 "
        "for claim-quality rationale"
    ) in res.stderr
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_rejects_out_of_range_top_target_skin_score(
    tmp_path: Path,
) -> None:
    rankings_dir = tmp_path / "rankings"
    rankings_dir.mkdir()
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')
    pd.DataFrame([{
        "target_id": "P1",
        "final_score": 0.9,
        "source_count": 3,
        "sources": "autodock;gnina;rtmscore",
        "skin_score": 1.2,
    }]).to_csv(rankings_dir / "cold_start__comprehensive.csv", index=False)

    res = run_iteration(tmp_path, ["--allow-empty-iteration"])

    assert res.returncode != 0
    assert (
        "cold_start__comprehensive.csv column 'skin_score' must be in [0, 1]"
        in res.stderr
    )
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_rejects_empty_top_target_skin_score(
    tmp_path: Path,
) -> None:
    rankings_dir = tmp_path / "rankings"
    rankings_dir.mkdir()
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')
    (rankings_dir / "cold_start__comprehensive.csv").write_text(
        "target_id,final_score,source_count,sources,skin_score\n"
        "P1,0.9,3,autodock;gnina;rtmscore,\n"
    )

    res = run_iteration(
        tmp_path,
        ["--allow-empty-iteration", "--allow-incomplete-eval-inputs"],
    )

    assert res.returncode != 0
    assert (
        "cold_start__comprehensive.csv column 'skin_score' is blank in top row"
        in res.stderr
    )
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_rejects_non_top_out_of_range_skin_score(
    tmp_path: Path,
) -> None:
    rankings_dir = tmp_path / "rankings"
    rankings_dir.mkdir()
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')
    pd.DataFrame([
        {
            "target_id": "P1",
            "final_score": 0.95,
            "source_count": 3,
            "sources": "autodock;gnina;rtmscore",
            "skin_score": 0.8,
        },
        {
            "target_id": "P2",
            "final_score": 0.80,
            "source_count": 3,
            "sources": "autodock;gnina;rtmscore",
            "skin_score": 1.2,
        },
    ]).to_csv(rankings_dir / "cold_start__comprehensive.csv", index=False)

    res = run_iteration(
        tmp_path,
        ["--allow-empty-iteration", "--allow-incomplete-eval-inputs"],
    )

    assert res.returncode != 0
    assert (
        "cold_start__comprehensive.csv column 'skin_score' at row index 1 "
        "must be in [0, 1]"
    ) in res.stderr
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_rejects_non_top_empty_skin_score(
    tmp_path: Path,
) -> None:
    rankings_dir = tmp_path / "rankings"
    rankings_dir.mkdir()
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')
    (rankings_dir / "cold_start__comprehensive.csv").write_text(
        "target_id,final_score,source_count,sources,skin_score\n"
        "P1,0.95,3,autodock;gnina;rtmscore,0.8\n"
        "P2,0.80,3,autodock;gnina;rtmscore,\n"
    )

    res = run_iteration(
        tmp_path,
        ["--allow-empty-iteration", "--allow-incomplete-eval-inputs"],
    )

    assert res.returncode != 0
    assert (
        "cold_start__comprehensive.csv column 'skin_score' is blank "
        "at row index 1"
    ) in res.stderr
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_rejects_non_top_blank_rationale_text_field(
    tmp_path: Path,
) -> None:
    rankings_dir = tmp_path / "rankings"
    rankings_dir.mkdir()
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')
    pd.DataFrame([
        {
            "target_id": "P1",
            "final_score": 0.95,
            "source_count": 3,
            "sources": "autodock;gnina;rtmscore",
            "efficacy_category": "barrier",
        },
        {
            "target_id": "P2",
            "final_score": 0.80,
            "source_count": 3,
            "sources": "autodock;gnina;rtmscore",
            "efficacy_category": " ",
        },
    ]).to_csv(rankings_dir / "cold_start__comprehensive.csv", index=False)

    res = run_iteration(
        tmp_path,
        ["--allow-empty-iteration", "--allow-incomplete-eval-inputs"],
    )

    assert res.returncode != 0
    assert (
        "cold_start__comprehensive.csv column 'efficacy_category' is blank "
        "at row index 1"
    ) in res.stderr
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_rejects_top_empty_rationale_text_field(
    tmp_path: Path,
) -> None:
    rankings_dir = tmp_path / "rankings"
    rankings_dir.mkdir()
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')
    (rankings_dir / "cold_start__comprehensive.csv").write_text(
        "target_id,final_score,source_count,sources,efficacy_category\n"
        "P1,0.95,3,autodock;gnina;rtmscore,\n"
        "P2,0.80,3,autodock;gnina;rtmscore,barrier\n"
    )

    res = run_iteration(
        tmp_path,
        ["--allow-empty-iteration", "--allow-incomplete-eval-inputs"],
    )

    assert res.returncode != 0
    assert (
        "cold_start__comprehensive.csv column 'efficacy_category' is blank "
        "in top row"
    ) in res.stderr
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_rejects_non_top_empty_rationale_text_field(
    tmp_path: Path,
) -> None:
    rankings_dir = tmp_path / "rankings"
    rankings_dir.mkdir()
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')
    (rankings_dir / "cold_start__comprehensive.csv").write_text(
        "target_id,final_score,source_count,sources,efficacy_category\n"
        "P1,0.95,3,autodock;gnina;rtmscore,barrier\n"
        "P2,0.80,3,autodock;gnina;rtmscore,\n"
    )

    res = run_iteration(
        tmp_path,
        ["--allow-empty-iteration", "--allow-incomplete-eval-inputs"],
    )

    assert res.returncode != 0
    assert (
        "cold_start__comprehensive.csv column 'efficacy_category' is blank "
        "at row index 1"
    ) in res.stderr
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_invalid_threshold_values_are_diagnostic_only(
    tmp_path: Path,
) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    pd.DataFrame([cold_start_metric_row(passes_threshold="maybe")]).to_csv(
        eval_dir / "cold_start_comprehensive.csv",
        index=False,
    )

    res = run_iteration(
        tmp_path,
        ["--allow-empty-iteration", "--allow-threshold-failure"],
    )

    assert res.returncode == 0, res.stderr
    payload = json.loads((tmp_path / "iteration.json").read_text())
    checked = payload["threshold_status"]["checked"][0]
    assert checked["passed"] is False
    assert checked["n_invalid_rows"] == 1
    assert payload["threshold_status"]["failures"][0]["n_invalid_rows"] == 1
    assert not payload["ranking_metrics"]["cold_start_comprehensive"]["passes_threshold"]
    assert not payload["ranking_metrics"]["cold_start_comprehensive"]["passes_threshold_valid"]


def test_run_iteration_rejects_analog_metric_without_required_columns(
    tmp_path: Path,
) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')
    pd.DataFrame([{
        "passes_threshold": True,
    }]).to_csv(eval_dir / "analog_quality.csv", index=False)

    res = run_iteration(tmp_path, ["--allow-empty-iteration"])

    assert res.returncode != 0
    assert "analog_quality.csv missing required metric columns" in res.stderr
    assert "recovery_fraction" in res.stderr
    assert "novelty_fraction" in res.stderr
    assert "mean_ra_score" in res.stderr
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_rejects_multirow_analog_metric(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')
    pd.DataFrame([
        {
            "recovery_fraction": 1.0,
            "novelty_fraction": 1.0,
            "mean_ra_score": 1.0,
            "passes_threshold": True,
        },
        {
            "recovery_fraction": 0.0,
            "novelty_fraction": 0.0,
            "mean_ra_score": 0.0,
            "passes_threshold": False,
        },
    ]).to_csv(eval_dir / "analog_quality.csv", index=False)

    res = run_iteration(tmp_path, ["--allow-empty-iteration"])

    assert res.returncode != 0
    assert "analog_quality.csv must contain exactly one metric row" in res.stderr
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_rejects_pharmacophore_metric_without_required_columns(
    tmp_path: Path,
) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')
    pd.DataFrame([{
        "passes_threshold": True,
    }]).to_csv(eval_dir / "pharmacophore_conservation.csv", index=False)

    res = run_iteration(tmp_path, ["--allow-empty-iteration"])

    assert res.returncode != 0
    assert (
        "pharmacophore_conservation.csv missing required metric columns"
        in res.stderr
    )
    assert "preserved_fraction" in res.stderr
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_rejects_multirow_pharmacophore_metric(
    tmp_path: Path,
) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')
    pd.DataFrame([
        {"preserved_fraction": 1.0, "passes_threshold": True},
        {"preserved_fraction": 0.0, "passes_threshold": False},
    ]).to_csv(eval_dir / "pharmacophore_conservation.csv", index=False)

    res = run_iteration(tmp_path, ["--allow-empty-iteration"])

    assert res.returncode != 0
    assert (
        "pharmacophore_conservation.csv must contain exactly one metric row"
        in res.stderr
    )
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_rejects_cosmetic_metric_without_required_columns(
    tmp_path: Path,
) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')
    pd.DataFrame([{
        "passes_threshold": True,
    }]).to_csv(eval_dir / "cosmetic_retrospective.csv", index=False)

    res = run_iteration(tmp_path, ["--allow-empty-iteration"])

    assert res.returncode != 0
    assert "cosmetic_retrospective.csv missing required metric columns" in res.stderr
    assert "top1" in res.stderr
    assert "top5" in res.stderr
    assert "top10" in res.stderr
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_rejects_cosmetic_metric_with_invalid_status(
    tmp_path: Path,
) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')
    pd.DataFrame([{
        "top1": 1,
        "top5": 1,
        "top10": 1,
        "status": "forged",
        "passes_threshold": True,
    }]).to_csv(eval_dir / "cosmetic_retrospective.csv", index=False)

    res = run_iteration(tmp_path, ["--allow-empty-iteration"])

    assert res.returncode != 0
    assert (
        "cosmetic_retrospective.csv contains invalid status values"
        in res.stderr
    )
    assert "forged" in res.stderr
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_rejects_out_of_range_fraction_column_metric(
    tmp_path: Path,
) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')
    pd.DataFrame([{
        "top1": 0,
        "top5": 0,
        "top10": 1.1,
        "passes_threshold": True,
    }]).to_csv(eval_dir / "cosmetic_retrospective.csv", index=False)

    res = run_iteration(tmp_path, ["--allow-empty-iteration"])

    assert res.returncode != 0
    assert (
        "cosmetic_retrospective.csv column 'top10' must be in [0, 1]"
        in res.stderr
    )
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_rejects_boolean_fraction_column_metric(
    tmp_path: Path,
) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')
    pd.DataFrame([{
        "top1": 0,
        "top5": 0,
        "top10": True,
        "status": "evaluated",
        "passes_threshold": True,
    }]).to_csv(eval_dir / "cosmetic_retrospective.csv", index=False)

    res = run_iteration(tmp_path, ["--allow-empty-iteration"])

    assert res.returncode != 0
    assert (
        "cosmetic_retrospective.csv column 'top10' must be numeric"
        in res.stderr
    )
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_rejects_skin_efficacy_metric_without_required_columns(
    tmp_path: Path,
) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')
    pd.DataFrame([{
        "passes_threshold": True,
    }]).to_csv(eval_dir / "skin_efficacy_recovery.csv", index=False)

    res = run_iteration(tmp_path, ["--allow-empty-iteration"])

    assert res.returncode != 0
    assert "skin_efficacy_recovery.csv missing required metric columns" in res.stderr
    assert "precision" in res.stderr
    assert "recall" in res.stderr
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_rejects_skin_metric_passing_non_evaluated_status(
    tmp_path: Path,
) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    manifest = tmp_path / "iteration.json"
    manifest.write_text('{"stale": true}\n')
    pd.DataFrame([{
        "precision": 1,
        "recall": 1,
        "status": "no_ranking",
        "passes_threshold": True,
    }]).to_csv(eval_dir / "skin_efficacy_recovery.csv", index=False)

    res = run_iteration(tmp_path, ["--allow-empty-iteration"])

    assert res.returncode != 0
    assert (
        "skin_efficacy_recovery.csv contains passing threshold rows "
        "with non-evaluated status"
    ) in res.stderr
    assert json.loads(manifest.read_text()) == {"stale": True}


def test_run_iteration_rejects_partial_eval_input_sets(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    pd.DataFrame([{"smiles": "CCO"}]).to_csv(
        eval_dir / "generated_analogs.csv",
        index=False,
    )

    res = run_iteration(tmp_path, ["--allow-empty-iteration"])

    assert res.returncode != 0
    assert "Incomplete evaluation input set" in res.stderr
    payload = json.loads((tmp_path / "iteration.json").read_text())
    assert payload["input_status"]["status"] == "incomplete"
    assert payload["input_status"]["n_incomplete"] == 2


def test_run_iteration_treats_empty_eval_input_artifact_as_missing(
    tmp_path: Path,
) -> None:
    eval_dir = tmp_path / "eval"
    rankings_dir = tmp_path / "rankings"
    eval_dir.mkdir()
    rankings_dir.mkdir()
    (eval_dir / "generated_analogs.csv").touch()

    payload = input_status(
        eval_dir,
        rankings_dir,
        tmp_path / "refs" / "chembl_fp.parquet",
        tmp_path / "refs" / "seq_db",
        tmp_path / "refs" / "training_ligands.smi",
        tmp_path / "refs" / "training_holo.csv",
    )

    assert payload["status"] == "incomplete"
    assert payload["n_checked"] == 2
    assert payload["n_incomplete"] == 2
    assert {
        item["name"] for item in payload["incomplete"]
    } == {"analog_quality", "pharmacophore_conservation"}
    assert all(
        str(eval_dir / "generated_analogs.csv") in item["missing"]
        for item in payload["incomplete"]
    )


def test_run_iteration_incomplete_eval_inputs_flag_is_diagnostic(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    pd.DataFrame([{"smiles": "CCO"}]).to_csv(
        eval_dir / "generated_analogs.csv",
        index=False,
    )

    res = run_iteration(
        tmp_path,
        ["--allow-empty-iteration", "--allow-incomplete-eval-inputs"],
    )

    assert res.returncode == 0, res.stderr
    payload = json.loads((tmp_path / "iteration.json").read_text())
    assert payload["input_status"]["status"] == "incomplete"
    assert {
        item["name"] for item in payload["input_status"]["incomplete"]
    } == {"analog_quality", "pharmacophore_conservation"}


def test_run_iteration_rejects_partial_cold_start_inputs(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    pd.DataFrame([{"uniprot": "P1"}]).to_csv(
        eval_dir / "cold_start_truth.csv",
        index=False,
    )

    res = run_iteration(tmp_path, ["--allow-empty-iteration"])

    assert res.returncode != 0
    assert "Incomplete evaluation input set" in res.stderr
    payload = json.loads((tmp_path / "iteration.json").read_text())
    assert payload["input_status"]["status"] == "incomplete"
    assert payload["input_status"]["incomplete"][0]["name"] == "cold_start"
    assert any(
        path.endswith("cold_start__comprehensive.csv")
        for path in payload["input_status"]["incomplete"][0]["missing"]
    )


def test_run_iteration_uses_configured_leakage_references(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    pd.DataFrame([{
        "run_id": "case",
        "uniprot": "P1",
        "smiles": "CCO",
    }]).to_csv(eval_dir / "eval_targets.csv", index=False)
    seq_db = tmp_path / "refs" / "seq_db"
    ligands = tmp_path / "refs" / "training_ligands.smi"
    holo = tmp_path / "refs" / "training_holo.csv"

    res = run_iteration(
        tmp_path,
        [
            "--training-seq-db", str(seq_db),
            "--training-ligands", str(ligands),
            "--training-holo", str(holo),
            "--allow-empty-iteration",
        ],
    )

    assert res.returncode != 0
    payload = json.loads((tmp_path / "iteration.json").read_text())
    command = payload["steps"][0]["command"]
    assert str(seq_db) in command
    assert str(ligands) in command
    assert str(holo) in command


def test_run_iteration_uses_configured_leakage_thresholds(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    pd.DataFrame([{
        "run_id": "case",
        "uniprot": "P1",
        "smiles": "CCO",
    }]).to_csv(eval_dir / "eval_targets.csv", index=False)
    seq_db = tmp_path / "refs" / "seq_db"
    ligands = tmp_path / "refs" / "training_ligands.smi"
    holo = tmp_path / "refs" / "training_holo.csv"

    res = run_iteration(
        tmp_path,
        [
            "--training-seq-db", str(seq_db),
            "--training-ligands", str(ligands),
            "--training-holo", str(holo),
            "--seq-id-threshold", "0.21",
            "--ligand-tanimoto-threshold", "0.62",
            "--pocket-sucos-threshold", "0.73",
            "--allow-empty-iteration",
        ],
    )

    assert res.returncode != 0
    payload = json.loads((tmp_path / "iteration.json").read_text())
    command = payload["steps"][0]["command"]
    assert "--seq-id-threshold" in command
    assert "0.21" in command
    assert "--ligand-tanimoto-threshold" in command
    assert "0.62" in command
    assert "--pocket-sucos-threshold" in command
    assert "0.73" in command


def test_run_iteration_rejects_invalid_cli_thresholds_before_steps(
    tmp_path: Path,
) -> None:
    res = run_iteration(
        tmp_path,
        [
            "--allow-empty-iteration",
            "--allow-incomplete-eval-inputs",
            "--allow-incomplete-leakage",
            "--allow-missing-input-runs",
            "--cold-start-min-recall-at-50", "1.5",
        ],
    )

    assert res.returncode != 0
    assert (
        "--cold-start-min-recall-at-50 must be a finite value in [0, 1]"
        in res.stderr
    )
    assert not (tmp_path / "iteration.json").exists()


def test_run_iteration_rejects_invalid_leakage_threshold_before_steps(
    tmp_path: Path,
) -> None:
    res = run_iteration(
        tmp_path,
        [
            "--allow-empty-iteration",
            "--allow-incomplete-eval-inputs",
            "--allow-incomplete-leakage",
            "--allow-missing-input-runs",
            "--pocket-sucos-threshold", "-0.1",
        ],
    )

    assert res.returncode != 0
    assert "--pocket-sucos-threshold must be a finite value in [0, 1]" in res.stderr
    assert not (tmp_path / "iteration.json").exists()


def write_disagreement_inputs(tmp_path: Path) -> Path:
    eval_dir = tmp_path / "eval"
    rankings_dir = tmp_path / "rankings"
    eval_dir.mkdir()
    rankings_dir.mkdir()
    (eval_dir / "disagreement_analysis.json").write_text(json.dumps({
        "docking_only": ["P1"],
        "dti_only": [],
        "both_top": [],
    }))
    pd.DataFrame([{
        "target_id": "P1",
        "final_score": 0.95,
        "source_count": 4,
        "sources": "autodock;gnina;rtmscore;boltz",
        "skin_score": 0.8,
    }]).to_csv(rankings_dir / "cold_start__comprehensive.csv", index=False)
    target_classes = tmp_path / "target_classes.parquet"
    pd.DataFrame([{"uniprot": "P1", "class": "kinase"}]).to_parquet(target_classes)
    return target_classes


def write_clean_leakage_inputs(tmp_path: Path, *, leak: bool = False) -> list[str]:
    eval_dir = tmp_path / "eval"
    refs = tmp_path / "leakage_refs"
    eval_dir.mkdir(exist_ok=True)
    refs.mkdir(exist_ok=True)
    pd.DataFrame([{
        "uniprot": "P1",
        "smiles": "CCO",
        "sequence": "AAAA",
    }]).to_csv(eval_dir / "eval_targets.csv", index=False)
    seq = refs / "training_cutoff_seqs.fasta"
    seq.write_text(">train\nAAAA\n" if leak else ">train\nYYYY\n")
    ligands = refs / "training_ligands.smi"
    ligands.write_text("c1ccccc1 ref\n")
    pockets = refs / "training_holo_pockets.csv"
    pockets.write_text("target_id,pocket_sucos\nP1,0.1\n")
    direct = refs / "direct_exact_reference.smi"
    direct.write_text("CCN known_direct\n")
    return [
        "--training-seq-db", str(seq),
        "--training-ligands", str(ligands),
        "--training-holo", str(pockets),
        "--direct-exact-reference", str(direct),
    ]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_candidate_experiment(
    tmp_path: Path,
    *,
    selected: bool = True,
) -> tuple[Path, Path, Path]:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir(exist_ok=True)
    candidate_dir = tmp_path / "candidate_inputs"
    candidate_dir.mkdir(exist_ok=True)
    training_data = candidate_dir / "training.csv"
    target_universe = candidate_dir / "targets.csv"
    benchmark = candidate_dir / "benchmark.csv"
    training_data.write_text("compound_id,smiles\nC1,CCO\n")
    target_universe.write_text("target_id\nP1\n")
    benchmark.write_text("target_id,compound_id,label\nP1,C1,1\n")
    comparators = {}
    for candidate_id in ("E0", "E1", "E2", "E3"):
        model = candidate_dir / f"{candidate_id}.model"
        model.write_text(f"{candidate_id} frozen model\n")
        comparators[candidate_id] = {
            "id": f"{candidate_id}_model",
            "license": "MIT",
            "version": "1.0",
            "model_sha256": file_sha256(model),
        }
    contract = {
        "schema_version": candidate_matrix_module.SCHEMA_VERSION,
        "candidates": candidate_matrix_module.REQUIRED_CANDIDATES,
        "training_window": candidate_matrix_module.TRAINING_WINDOW,
        "selection_window": candidate_matrix_module.SELECTION_WINDOW,
        "artifacts": {
            "training_data": {
                "path": str(training_data),
                "sha256": file_sha256(training_data),
            },
            "target_universe": {
                "path": str(target_universe),
                "sha256": file_sha256(target_universe),
            },
            "benchmark": {
                "path": str(benchmark),
                "sha256": file_sha256(benchmark),
            },
        },
        "leakage_thresholds": candidate_matrix_module.LEAKAGE_THRESHOLDS,
        "comparators": comparators,
        "seeds": [11, 23, 37],
        "grid": {
            "dock": candidate_matrix_module.DOCK_GRID,
            "rerank": candidate_matrix_module.RERANK_GRID,
            "diffdock": candidate_matrix_module.DIFFDOCK_GRID,
        },
        "objective_order": candidate_matrix_module.OBJECTIVE_ORDER,
        "guardrails": candidate_matrix_module.GUARDRAILS,
        "run_budget": {"max_runs": 37, "max_gpu_hours": 1000},
    }
    prereg = eval_dir / "candidate_preregistration.json"
    prereg.write_text(
        json.dumps(
            candidate_matrix_module.seal_preregistration_contract(contract),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    rows = []
    seeds_sha256 = candidate_matrix_module.canonical_sha256(contract["seeds"])
    for idx, row in enumerate(candidate_matrix_module.expected_matrix()):
        is_baseline = row["candidate_id"] == "E0"
        is_selected = selected and row["config_id"] == "E1_dock64_rerank10_diffdock0"
        metrics = {
            "top30": 0.50,
            "top10": 0.40,
            "mrr": 0.50,
            "brier": 0.20,
            "log_loss": 0.30,
            "cold_p95_minutes": 20.0,
            "bundle_gb": 5.0,
        }
        if not is_baseline:
            metrics.update({
                "top30": 0.49,
                "top10": 0.39,
                "mrr": 0.49,
                "brier": 0.20,
                "log_loss": 0.30,
                "cold_p95_minutes": 21.0,
                "bundle_gb": 6.0,
            })
        if is_selected:
            metrics.update({
                "top30": 0.51,
                "top10": 0.42,
                "mrr": 0.60,
                "cold_p95_minutes": 18.0,
                "bundle_gb": 4.0,
            })
        rows.append({
            **row,
            "run_id": f"run_{idx:02d}",
            "model_sha256": comparators[row["candidate_id"]]["model_sha256"],
            "data_sha256": file_sha256(training_data),
            "target_universe_sha256": file_sha256(target_universe),
            "benchmark_sha256": file_sha256(benchmark),
            "seeds_sha256": seeds_sha256,
            **metrics,
        })
    results = eval_dir / "candidate_results.csv"
    pd.DataFrame(rows).to_csv(results, index=False)
    selection = eval_dir / "candidate_selection.json"
    selection.write_text(
        json.dumps(
            candidate_matrix_module.evaluate_results(prereg, results),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return prereg, results, selection


def write_skin_known_claim_inputs(
    tmp_path: Path,
    *,
    write_ledger: bool = True,
    write_baselines: bool = True,
    write_ablations: bool = True,
) -> list[str]:
    eval_dir = tmp_path / "eval"
    rankings_root = tmp_path / "rankings"
    rankings = rankings_root / "skin_known_target"
    run_dir = tmp_path / "results" / "runs" / "Retinol"
    source = run_dir / "03_targets" / "ranked_targets_v3.csv"
    copied = rankings / "Retinol__ranked_targets_v3.csv"
    eval_dir.mkdir(exist_ok=True)
    source.parent.mkdir(parents=True)
    rankings.mkdir(parents=True)

    panel = tmp_path / "known_panel.csv"
    pd.DataFrame([{
        "case_id": "Retinol",
        "inci_name": "Retinol",
        "panel": "beneficial",
        "skin_effect": "anti-aging retinoid",
        "smiles": "CCO",
        "known_targets": "P2",
        "known_target_labels": "Target 2",
        "evidence_note": "Retinoid receptor positive control",
    }]).to_csv(panel, index=False)

    ranking_rows = [{
        "target_id": "P2",
        "final_score": 0.95,
        "source_count": 3,
        "sources": "psichic;gnina;skinscore",
        "known_target_prior": 0.0,
        "known_target_prior_norm": 0.0,
    }]
    pd.DataFrame(ranking_rows).to_csv(source, index=False)
    pd.DataFrame(ranking_rows).to_csv(copied, index=False)

    run_manifest = run_dir / "run_summary.json"
    run_manifest.write_text(json.dumps({
        "run_id": "Retinol",
        "canonical_smiles": "CCO",
        "ranking": str(source),
    }) + "\n")
    copied_record = {
        "path": str(copied),
        "source_path": str(source),
        "source_bytes": source.stat().st_size,
        "source_sha256": file_sha256(source),
        "bytes": copied.stat().st_size,
        "sha256": file_sha256(copied),
    }
    (eval_dir / "collected_runs.json").write_text(json.dumps({
        "out_dir": str(eval_dir),
        "rankings_dir": str(rankings_root),
        "n_runs": 1,
        "runs": [{
            "run_id": "Retinol",
            "run_dir": str(run_dir),
            "canonical_smiles": "CCO",
            "ranked_v3": str(source),
            "leakage_rows": [{
                "run_id": "Retinol",
                "uniprot": "P2",
                "smiles": "CCO",
            }],
            "copied": [str(copied)],
            "copied_artifacts": [copied_record],
        }],
    }))

    ledger = eval_dir / "skin_known_target_run_ledger.csv"
    if write_ledger:
        pd.DataFrame([{
            "case_id": "Retinol",
            "smiles": "CCO",
            "run_id": "Retinol",
            "command": (
                "python scripts/run_skinscout.py CCO "
                "--preset target-id-sota"
            ),
            "source_run_dir": str(run_dir),
            "copied_ranking_path": str(copied),
            "config_sha256": "a" * 64,
            "run_manifest_sha256": file_sha256(run_manifest),
            "run_manifest_path": str(run_manifest),
        }]).to_csv(ledger, index=False)

    panel_sha = file_sha256(panel)
    baselines = eval_dir / "sota_baselines.csv"
    if write_baselines:
        pd.DataFrame([
            {
                "baseline_id": baseline_type,
                "baseline_type": baseline_type,
                "status": "included",
                "panel_sha256": panel_sha,
                "target_universe": "skin_human_stage0_v1",
                "uniprot_mapping": "uniprot_accession_v1",
                "seq_id_threshold": 0.30,
                "ligand_tanimoto_threshold": 0.50,
                "pocket_sucos_threshold": 0.50,
                "input_evidence_status": "same",
                "training_data_status": "disclosed_no_panel_overlap",
                "license_status": "compatible",
            }
            for baseline_type in sorted(
                run_iteration_module.REQUIRED_BASELINE_TYPES,
            )
        ]).to_csv(baselines, index=False)

    ablations = eval_dir / "sota_ablations.csv"
    if write_ablations:
        pd.DataFrame([
            {
                "ablation_id": family,
                "ablation_family": family,
                "status": "included",
                "panel_sha256": panel_sha,
                "target_universe": "skin_human_stage0_v1",
                "case_top10": 1.0,
                "target_top10": 1.0,
                "target_top30": 1.0,
            }
            for family in sorted(
                run_iteration_module.REQUIRED_ABLATION_FAMILIES,
            )
        ]).to_csv(ablations, index=False)

    return [
        "--skin-known-cases-csv", str(panel),
        "--skin-known-rankings-dir", str(rankings),
        "--skin-known-run-ledger-csv", str(ledger),
        "--sota-baselines-csv", str(baselines),
        "--sota-ablations-csv", str(ablations),
        "--skin-known-context-profile", "anti_aging",
        "--skin-known-min-case-top10", "0.50",
        "--skin-known-min-target-top10", "0.50",
        "--skin-known-min-target-top30", "0.50",
    ]


def write_full_retrospective_rankings(tmp_path: Path) -> None:
    retro_dir = tmp_path / "rankings" / "cosmetic_retro"
    retro_dir.mkdir(parents=True)
    cosmetic_targets = {
        "Retinol": "P10276",
        "Niacinamide": "P40261",
        "Ascorbic_acid": "P13674",
        "alpha-Arbutin": "P14679",
        "Kojic_acid": "P14679",
        "Salicylic_acid": "P23219",
        "Resveratrol": "Q96EB6",
        "EGCG": "P14780",
        "Caffeine": "P29274",
        "Adenosine": "P30542",
    }
    efficacy_labels = {
        "Retinol": "anti_aging (10 papers)",
        "Niacinamide": "whitening (10 papers)",
        "Kojic_acid": "whitening (10 papers)",
        "alpha-Arbutin": "whitening (10 papers)",
        "Salicylic_acid": "acne (10 papers)",
        "Resveratrol": "antioxidant (10 papers)",
    }
    for inci, target in cosmetic_targets.items():
        pd.DataFrame([{
            "target_id": target,
            "final_score": 1.0,
            "source_count": 3,
            "sources": "autodock;gnina;rtmscore",
        }]).to_csv(
            retro_dir / f"{inci}__ranked_targets_v3.csv",
            index=False,
        )
        if inci in efficacy_labels:
            pd.DataFrame([{
                "target_id": target,
                "final_score": 1.0,
                "source_count": 3,
                "sources": "autodock;gnina;rtmscore",
                "efficacy_top1": efficacy_labels[inci],
            }]).to_csv(
                retro_dir / f"{inci}__ranked_targets_v3_with_efficacy.csv",
                index=False,
            )


def test_run_iteration_executes_skin_efficacy_when_rankings_are_available(
    tmp_path: Path,
) -> None:
    write_full_retrospective_rankings(tmp_path)

    res = run_iteration(
        tmp_path,
        ["--allow-incomplete-leakage", "--allow-missing-input-runs"],
    )

    assert res.returncode == 0, res.stderr
    payload = json.loads((tmp_path / "iteration.json").read_text())
    assert payload["n_steps"] == 2
    assert payload["n_passed"] == 2
    assert [step["name"] for step in payload["steps"]] == [
        "cosmetic_retrospective",
        "skin_efficacy_recovery",
    ]
    assert payload["ranking_metrics"]["cosmetic_retrospective"]["passes_threshold"]
    assert payload["ranking_metrics"]["skin_efficacy_recovery"]["passes_threshold"]
    assert payload["threshold_status"]["status"] == "ok"
    assert (tmp_path / "eval" / "skin_efficacy_recovery.csv").exists()


def test_run_iteration_executes_skin_known_target_recovery_when_available(
    tmp_path: Path,
) -> None:
    cases = tmp_path / "known_panel.csv"
    rankings = tmp_path / "rankings" / "skin_known_target"
    rankings.mkdir(parents=True)
    pd.DataFrame([{
        "case_id": "Retinol",
        "inci_name": "Retinol",
        "panel": "beneficial",
        "skin_effect": "anti-aging retinoid",
        "smiles": "CCO",
        "known_targets": "P1;P2",
        "known_target_labels": "Target 1;Target 2",
        "evidence_note": "Retinoid receptor positive-control panel member",
    }]).to_csv(cases, index=False)
    pd.DataFrame([
        {
            "target_id": "P2",
            "final_score": 0.95,
            "known_target_prior": 0.0,
            "known_target_prior_norm": 0.0,
        },
        {
            "target_id": "P3",
            "final_score": 0.80,
            "known_target_prior": 0.0,
            "known_target_prior_norm": 0.0,
        },
    ]).to_csv(rankings / "Retinol__ranked_targets_v3.csv", index=False)

    res = run_iteration(
        tmp_path,
        [
            "--skin-known-cases-csv", str(cases),
            "--skin-known-rankings-dir", str(rankings),
            "--skin-known-context-profile", "anti_aging",
            "--skin-known-min-case-top10", "0.50",
            "--skin-known-min-target-top10", "0.50",
            "--skin-known-min-target-top30", "0.50",
            "--allow-incomplete-leakage",
            "--allow-missing-input-runs",
        ],
    )

    assert res.returncode == 0, res.stderr
    payload = json.loads((tmp_path / "iteration.json").read_text())
    assert payload["n_steps"] == 1
    assert payload["steps"][0]["name"] == "skin_known_target_recovery"
    metrics = payload["ranking_metrics"]["skin_known_target_recovery"]
    assert metrics["case_top10"] == 1.0
    assert metrics["target_top10"] == 0.5
    assert metrics["passes_threshold"]
    assert payload["threshold_status"]["status"] == "ok"
    summary = tmp_path / "eval" / "skin_known_target_recovery_summary.json"
    assert json.loads(summary.read_text())["context_profile"] == "anti_aging"


def test_run_iteration_claim_ready_requires_and_records_sota_freeze(
    tmp_path: Path,
) -> None:
    leakage_args = write_clean_leakage_inputs(tmp_path)
    claim_args = write_skin_known_claim_inputs(tmp_path)
    activity_gate = write_activity_retrieval_gate(tmp_path)

    res = run_iteration(tmp_path, [
        *claim_args,
        *leakage_args,
        "--activity-retrieval-gate",
        str(activity_gate),
    ])

    assert res.returncode == 0, res.stderr
    payload = json.loads((tmp_path / "iteration.json").read_text())
    assert payload["claim_ready"] is True
    assert payload["claim_blockers"] == []
    assert payload["n_steps"] == 2
    assert payload["n_passed"] == 2
    assert payload["leakage_status"]["status"] == "ok"
    sealed = payload["leakage_status"]["sealed_audit"]
    assert sealed["status"] == "ok"
    assert sealed["schema_version"] == "skinscout.discovery-leakage-audit.v2"
    assert len(sealed["binding_sha256"]) == 64
    leakage_step = next(step for step in payload["steps"] if step["name"] == "leakage")
    challenge_index = leakage_step["command"].index("--execution-challenge") + 1
    assert leakage_step["command"][challenge_index] == sealed["execution_challenge"]
    assert payload["input_runs"]["status"] == "present"
    freeze = payload["sota_freeze"]
    assert freeze["status"] == "checked"
    assert freeze["skin_known_run_ledger"]["status"] == "ok"
    assert freeze["baseline_fairness"]["status"] == "ok"
    assert freeze["ablation_freeze"]["status"] == "ok"
    assert payload["data_snapshot"]["skin_known_run_ledger"]["status"] == "present"
    assert payload["ranking_metrics"]["skin_known_target_recovery"]["case_top10"] == 1.0
    assert payload["release_governance"]["sota_claim_ready"] is True
    assert payload["release_governance"]["software_release_ready"] is True
    assert payload["release_governance"]["model_promotion_ready"] is False


def test_run_iteration_claim_ready_requires_activity_retrieval_gate(
    tmp_path: Path,
) -> None:
    leakage_args = write_clean_leakage_inputs(tmp_path)
    claim_args = write_skin_known_claim_inputs(tmp_path)

    res = run_iteration(tmp_path, [
        *claim_args,
        *leakage_args,
        "--activity-retrieval-gate",
        str(tmp_path / "missing-activity-gate.flag"),
    ])

    assert res.returncode == 0, res.stderr
    payload = json.loads((tmp_path / "iteration.json").read_text())
    assert payload["claim_ready"] is False
    assert payload["activity_retrieval_gate"]["status"] == "missing"
    assert {item["code"] for item in payload["claim_blockers"]} == {
        "activity_retrieval_gate_not_ready"
    }


def test_run_iteration_rejects_missing_direct_exact_reference(
    tmp_path: Path,
) -> None:
    leakage_args = write_clean_leakage_inputs(tmp_path)
    direct = Path(leakage_args[-1])
    direct.unlink()

    res = run_iteration(
        tmp_path,
        [
            *leakage_args,
            "--allow-incomplete-eval-inputs",
            "--allow-threshold-failure",
            "--allow-missing-input-runs",
        ],
    )

    assert res.returncode != 0
    payload = json.loads((tmp_path / "iteration.json").read_text())
    leakage_step = next(step for step in payload["steps"] if step["name"] == "leakage")
    assert leakage_step["status"] == "failed"
    assert "Direct exact reference is required and missing" in leakage_step["stderr"]
    assert payload["leakage_status"]["sealed_audit"]["status"] == "missing"
    assert payload["claim_ready"] is False


def test_run_iteration_requires_skin_known_run_ledger_for_claim_quality(
    tmp_path: Path,
) -> None:
    leakage_args = write_clean_leakage_inputs(tmp_path)
    claim_args = write_skin_known_claim_inputs(tmp_path, write_ledger=False)

    res = run_iteration(tmp_path, [*claim_args, *leakage_args])

    assert res.returncode != 0
    assert "SOTA freeze artifacts are required" in res.stderr
    assert "skin known-target run ledger status=missing" in res.stderr
    payload = json.loads((tmp_path / "iteration.json").read_text())
    assert payload["sota_freeze"]["skin_known_run_ledger"]["status"] == "missing"
    assert {
        blocker["code"] for blocker in payload["claim_blockers"]
    } == {"sota_freeze_not_claim_ready"}


def test_run_iteration_rejects_sota_baseline_without_required_type(
    tmp_path: Path,
) -> None:
    write_skin_known_claim_inputs(tmp_path)
    panel = tmp_path / "known_panel.csv"
    baselines = tmp_path / "eval" / "sota_baselines.csv"
    df = pd.read_csv(baselines)
    df = df[df["baseline_type"] != "nearest_neighbor"]
    df.to_csv(baselines, index=False)

    with pytest.raises(SystemExit, match="missing included required baseline_type"):
        sota_baseline_fairness(
            baselines,
            panel,
            0.30,
            0.50,
            0.50,
        )


def test_run_iteration_rejects_sota_ablation_without_required_family(
    tmp_path: Path,
) -> None:
    write_skin_known_claim_inputs(tmp_path)
    panel = tmp_path / "known_panel.csv"
    ablations = tmp_path / "eval" / "sota_ablations.csv"
    df = pd.read_csv(ablations)
    df = df[df["ablation_family"] != "dti_only"]
    df.to_csv(ablations, index=False)

    with pytest.raises(SystemExit, match="missing included required ablation_family"):
        sota_ablation_freeze(ablations, panel)


def test_run_iteration_rejects_skin_known_ledger_not_in_collected_artifacts(
    tmp_path: Path,
) -> None:
    write_skin_known_claim_inputs(tmp_path)
    eval_dir = tmp_path / "eval"
    ledger = eval_dir / "skin_known_target_run_ledger.csv"
    input_runs = {
        "status": "present",
        "runs": [{
            "run_id": "Retinol",
            "run_dir": str(tmp_path / "results" / "runs" / "Retinol"),
            "canonical_smiles": "CCO",
            "copied_artifacts": [],
        }],
    }

    with pytest.raises(SystemExit, match="must be in collected run copied_artifacts"):
        skin_known_run_ledger(
            ledger,
            tmp_path / "known_panel.csv",
            tmp_path / "rankings" / "skin_known_target",
            input_runs,
        )


def test_run_iteration_rejects_nonzero_prior_in_sota_ledger_ranking(
    tmp_path: Path,
) -> None:
    write_skin_known_claim_inputs(tmp_path)
    copied = (
        tmp_path
        / "rankings"
        / "skin_known_target"
        / "Retinol__ranked_targets_v3.csv"
    )
    ranking = pd.read_csv(copied)
    ranking["known_target_prior"] = 0.4
    ranking.to_csv(copied, index=False)
    input_runs_doc = json.loads(
        (tmp_path / "eval" / "collected_runs.json").read_text()
    )

    with pytest.raises(
        SystemExit,
        match="zero known-target prior contribution",
    ):
        skin_known_run_ledger(
            tmp_path / "eval" / "skin_known_target_run_ledger.csv",
            tmp_path / "known_panel.csv",
            tmp_path / "rankings" / "skin_known_target",
            {"status": "present", "runs": input_runs_doc["runs"]},
        )


def test_run_iteration_rejects_missing_prior_audit_columns_in_sota_ledger(
    tmp_path: Path,
) -> None:
    write_skin_known_claim_inputs(tmp_path)
    copied = (
        tmp_path
        / "rankings"
        / "skin_known_target"
        / "Retinol__ranked_targets_v3.csv"
    )
    ranking = pd.read_csv(copied).drop(columns=["known_target_prior_norm"])
    ranking.to_csv(copied, index=False)
    input_runs_doc = json.loads(
        (tmp_path / "eval" / "collected_runs.json").read_text()
    )

    with pytest.raises(SystemExit, match="must retain prior audit columns"):
        skin_known_run_ledger(
            tmp_path / "eval" / "skin_known_target_run_ledger.csv",
            tmp_path / "known_panel.csv",
            tmp_path / "rankings" / "skin_known_target",
            {"status": "present", "runs": input_runs_doc["runs"]},
        )


def test_run_iteration_requires_collected_input_runs_for_claim_quality(
    tmp_path: Path,
) -> None:
    write_full_retrospective_rankings(tmp_path)
    leakage_args = write_clean_leakage_inputs(tmp_path)

    res = run_iteration(tmp_path, leakage_args)

    assert res.returncode != 0
    assert "Collected run provenance is required for claim-quality iterations" in res.stderr
    assert "status=missing, n_runs=0" in res.stderr
    payload = json.loads((tmp_path / "iteration.json").read_text())
    assert payload["n_steps"] == 3
    assert payload["n_passed"] == 3
    assert payload["leakage_status"]["status"] == "ok"
    assert payload["threshold_status"]["status"] == "ok"
    assert payload["input_runs"]["status"] == "missing"


def test_run_iteration_requires_collected_run_artifacts_for_claim_quality(
    tmp_path: Path,
) -> None:
    write_full_retrospective_rankings(tmp_path)
    leakage_args = write_clean_leakage_inputs(tmp_path)
    eval_dir = tmp_path / "eval"
    run_dir = tmp_path / "results" / "runs" / "case"
    run_dir.mkdir(parents=True)
    (eval_dir / "collected_runs.json").write_text(json.dumps({
        "n_runs": 1,
        "runs": [{
            "run_id": "case",
            "run_dir": str(run_dir),
            "canonical_smiles": "CCO",
            "copied": [],
            "copied_artifacts": [],
        }],
    }))

    res = run_iteration(tmp_path, leakage_args)

    assert res.returncode != 0
    assert "runs[0].copied must be a non-empty list" in res.stderr
    payload = json.loads((tmp_path / "iteration.json").read_text())
    assert payload["n_steps"] == 3
    assert payload["n_passed"] == 3
    assert payload["input_runs"]["status"] == "present"
    assert payload["input_runs"]["n_runs"] == 1


def test_run_iteration_requires_collected_artifacts_inside_rankings_dir_for_claim_quality(
    tmp_path: Path,
) -> None:
    write_full_retrospective_rankings(tmp_path)
    leakage_args = write_clean_leakage_inputs(tmp_path)
    eval_dir = tmp_path / "eval"
    run_dir = tmp_path / "results" / "runs" / "case"
    source = run_dir / "03_targets" / "ranked_targets_v3.csv"
    copied = tmp_path / "outside_rankings" / "case__ranked_targets_v3.csv"
    source.parent.mkdir(parents=True)
    copied.parent.mkdir(parents=True)
    source.write_text("target_id,final_score,source_count,sources\nP1,0.9,3,autodock;gnina;rtmscore\n")
    copied.write_text(source.read_text())
    (eval_dir / "collected_runs.json").write_text(json.dumps({
        "out_dir": str(eval_dir),
        "rankings_dir": str(tmp_path / "rankings"),
        "n_runs": 1,
        "runs": [{
            "run_id": "case",
            "run_dir": str(run_dir),
            "canonical_smiles": "CCO",
            "leakage_rows": [{"run_id": "case", "uniprot": "P1", "smiles": "CCO"}],
            "copied": [str(copied)],
            "copied_artifacts": [{
                "path": str(copied),
                "source_path": str(source),
                "source_bytes": source.stat().st_size,
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "bytes": copied.stat().st_size,
                "sha256": hashlib.sha256(copied.read_bytes()).hexdigest(),
            }],
        }],
    }))

    res = run_iteration(tmp_path, leakage_args)

    assert res.returncode != 0
    assert "runs[0].copied[0] must be inside rankings_dir" in res.stderr
    payload = json.loads((tmp_path / "iteration.json").read_text())
    assert payload["n_steps"] == 3
    assert payload["n_passed"] == 3
    assert payload["input_runs"]["rankings_dir"] == str(tmp_path / "rankings")


def test_run_iteration_requires_collected_rankings_dir_to_match_active_eval(
    tmp_path: Path,
) -> None:
    write_full_retrospective_rankings(tmp_path)
    leakage_args = write_clean_leakage_inputs(tmp_path)
    eval_dir = tmp_path / "eval"
    run_dir = tmp_path / "results" / "runs" / "case"
    source = run_dir / "03_targets" / "ranked_targets_v3.csv"
    copied = tmp_path / "other_rankings" / "case__ranked_targets_v3.csv"
    source.parent.mkdir(parents=True)
    copied.parent.mkdir(parents=True)
    source.write_text("target_id,final_score,source_count,sources\nP1,0.9,3,autodock;gnina;rtmscore\n")
    copied.write_text(source.read_text())
    (eval_dir / "collected_runs.json").write_text(json.dumps({
        "out_dir": str(eval_dir),
        "rankings_dir": str(copied.parent),
        "n_runs": 1,
        "runs": [{
            "run_id": "case",
            "run_dir": str(run_dir),
            "canonical_smiles": "CCO",
            "leakage_rows": [{"run_id": "case", "uniprot": "P1", "smiles": "CCO"}],
            "copied": [str(copied)],
            "copied_artifacts": [{
                "path": str(copied),
                "source_path": str(source),
                "source_bytes": source.stat().st_size,
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "bytes": copied.stat().st_size,
                "sha256": hashlib.sha256(copied.read_bytes()).hexdigest(),
            }],
        }],
    }))

    res = run_iteration(tmp_path, leakage_args)

    assert res.returncode != 0
    assert "rankings_dir must match active --rankings-dir" in res.stderr
    payload = json.loads((tmp_path / "iteration.json").read_text())
    assert payload["n_steps"] == 3
    assert payload["n_passed"] == 3
    assert payload["input_runs"]["rankings_dir"] == str(copied.parent)


def test_run_iteration_requires_collected_run_leakage_rows_for_claim_quality(
    tmp_path: Path,
) -> None:
    write_full_retrospective_rankings(tmp_path)
    leakage_args = write_clean_leakage_inputs(tmp_path)
    eval_dir = tmp_path / "eval"
    rankings_dir = tmp_path / "rankings"
    run_dir = tmp_path / "results" / "runs" / "case"
    source = run_dir / "03_targets" / "ranked_targets_v3.csv"
    copied = rankings_dir / "cosmetic_retro" / "case__ranked_targets_v3.csv"
    source.parent.mkdir(parents=True)
    copied.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("target_id,final_score,source_count,sources\nP1,0.9,3,autodock;gnina;rtmscore\n")
    copied.write_text(source.read_text())
    (eval_dir / "collected_runs.json").write_text(json.dumps({
        "out_dir": str(eval_dir),
        "rankings_dir": str(rankings_dir),
        "n_runs": 1,
        "runs": [{
            "run_id": "case",
            "run_dir": str(run_dir),
            "canonical_smiles": "CCO",
            "copied": [str(copied)],
            "copied_artifacts": [{
                "path": str(copied),
                "source_path": str(source),
                "source_bytes": source.stat().st_size,
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "bytes": copied.stat().st_size,
                "sha256": hashlib.sha256(copied.read_bytes()).hexdigest(),
            }],
        }],
    }))

    res = run_iteration(tmp_path, leakage_args)

    assert res.returncode != 0
    assert "runs[0].n_leakage_rows must be > 0" in res.stderr
    payload = json.loads((tmp_path / "iteration.json").read_text())
    assert payload["n_steps"] == 3
    assert payload["n_passed"] == 3
    assert payload["input_runs"]["runs"][0]["n_leakage_rows"] == 0


def test_run_iteration_rejects_direct_discovery_source_in_ranking_support(
    tmp_path: Path,
) -> None:
    write_full_retrospective_rankings(tmp_path)
    ranking = tmp_path / "rankings" / "cosmetic_retro" / "Retinol__ranked_targets_v3.csv"
    df = pd.read_csv(ranking)
    df.loc[0, "sources"] = "autodock;direct_exact"
    df.loc[0, "source_count"] = 2
    df.to_csv(ranking, index=False)
    leakage_args = write_clean_leakage_inputs(tmp_path)

    res = run_iteration(
        tmp_path,
        [
            "--allow-missing-input-runs",
            *leakage_args,
        ],
    )

    assert res.returncode != 0
    assert "Discovery direct records cannot be support or ranking features" in res.stderr


def test_run_iteration_records_prospective_model_promotion_separately(
    tmp_path: Path,
) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
    import prospective_promotion_eval as promotion

    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    metrics = eval_dir / "prospective_metrics.csv"
    pd.DataFrame([{
        "candidate_model_id": "E3",
        "previous_signed_model_id": "E0",
        "preregistration_sha256": "a" * 64,
        "data_cutoff": "train_through_2023_select_2024",
        "training_window": "through_2023",
        "dev_selection_window": "2024_dev_only",
        "prospective_release_date": "2026-09-01",
        "prospective_source_releases": "ChEMBL;BindingDB;GtoPdb",
        "n_compounds": 100,
        "n_targets": 30,
        "n_positives": 200,
        "target_macro_top30": 0.05,
        "paired_bootstrap_lower_delta_top30": 0.001,
        "paired_bootstrap_lower_delta_top10": -0.01,
        "paired_bootstrap_lower_delta_mrr": -0.01,
        "brier_upper_delta": 0.005,
        "log_loss_upper_delta": 0.005,
        "unseen_cluster_positives": 5,
        "unseen_target_clusters": 3,
    }]).to_csv(metrics, index=False)
    manifest = eval_dir / "prospective_promotion_manifest.json"
    manifest.write_text(json.dumps(promotion.evaluate(metrics)) + "\n")
    activity_gate = write_activity_retrieval_gate(tmp_path)

    res = run_iteration(
        tmp_path,
        [
            "--allow-empty-iteration",
            "--allow-incomplete-leakage",
            "--allow-incomplete-eval-inputs",
            "--allow-threshold-failure",
            "--allow-missing-input-runs",
            "--activity-retrieval-gate",
            str(activity_gate),
        ],
    )

    assert res.returncode == 0, res.stderr
    payload = json.loads((tmp_path / "iteration.json").read_text())
    assert payload["prospective_promotion"]["model_promotion_ready"] is False
    assert payload["release_governance"]["model_promotion_ready"] is False
    assert payload["release_governance"]["sota_claim_ready"] is False

    changed = pd.read_csv(metrics)
    changed.loc[0, "target_macro_top30"] = 0.01
    changed.to_csv(metrics, index=False)
    stale_res = run_iteration(
        tmp_path,
        [
            "--allow-empty-iteration",
            "--allow-incomplete-leakage",
            "--allow-incomplete-eval-inputs",
            "--allow-threshold-failure",
            "--allow-missing-input-runs",
            "--activity-retrieval-gate",
            str(activity_gate),
        ],
    )

    assert stale_res.returncode != 0
    assert "Prospective promotion manifest is invalid" in stale_res.stderr
    assert "input.sha256" in stale_res.stderr


def test_run_iteration_rejects_minimal_prospective_promotion_manifest(
    tmp_path: Path,
) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / "prospective_promotion_manifest.json").write_text(json.dumps({
        "schema_version": "skinscout.prospective-discovery-promotion.v1",
        "status": "promote",
        "failed_gates": [],
    }) + "\n")

    res = run_iteration(
        tmp_path,
        [
            "--allow-empty-iteration",
            "--allow-incomplete-leakage",
            "--allow-incomplete-eval-inputs",
            "--allow-threshold-failure",
            "--allow-missing-input-runs",
        ],
    )

    assert res.returncode != 0
    assert "Prospective promotion manifest is invalid" in res.stderr


def test_run_iteration_candidate_experiment_all_or_none(tmp_path: Path) -> None:
    prereg, _results, _selection = write_candidate_experiment(tmp_path)
    prereg.unlink()

    res = run_iteration(
        tmp_path,
        [
            "--allow-empty-iteration",
            "--allow-incomplete-leakage",
            "--allow-incomplete-eval-inputs",
            "--allow-threshold-failure",
            "--allow-missing-input-runs",
        ],
    )

    assert res.returncode != 0
    assert "Candidate experiment artifacts are all-or-none" in res.stderr
    assert "candidate_preregistration.json" in res.stderr


def test_run_iteration_rejects_tampered_candidate_preregistration(
    tmp_path: Path,
) -> None:
    prereg, _results, _selection = write_candidate_experiment(tmp_path)
    payload = json.loads(prereg.read_text())
    Path(payload["artifacts"]["training_data"]["path"]).write_text(
        "compound_id,smiles\nC1,CCN\n"
    )

    res = run_iteration(
        tmp_path,
        [
            "--allow-empty-iteration",
            "--allow-incomplete-leakage",
            "--allow-incomplete-eval-inputs",
            "--allow-threshold-failure",
            "--allow-missing-input-runs",
        ],
    )

    assert res.returncode != 0
    assert "Candidate experiment artifact validation failed" in res.stderr
    assert "sha256 drift" in res.stderr


def test_run_iteration_rejects_candidate_selection_recompute_mismatch(
    tmp_path: Path,
) -> None:
    _prereg, results, _selection = write_candidate_experiment(tmp_path)
    changed = pd.read_csv(results)
    changed.loc[changed["config_id"] == "E1_dock64_rerank10_diffdock0", "mrr"] = 0.10
    changed.to_csv(results, index=False)

    res = run_iteration(
        tmp_path,
        [
            "--allow-empty-iteration",
            "--allow-incomplete-leakage",
            "--allow-incomplete-eval-inputs",
            "--allow-threshold-failure",
            "--allow-missing-input-runs",
        ],
    )

    assert res.returncode != 0
    assert "selection manifest does not match fresh evaluation" in res.stderr


def test_run_iteration_binds_successful_candidate_experiment_to_manifest(
    tmp_path: Path,
) -> None:
    prereg, results, selection = write_candidate_experiment(tmp_path)

    res = run_iteration(
        tmp_path,
        [
            "--allow-empty-iteration",
            "--allow-incomplete-leakage",
            "--allow-incomplete-eval-inputs",
            "--allow-threshold-failure",
            "--allow-missing-input-runs",
        ],
    )

    assert res.returncode == 0, res.stderr
    payload = json.loads((tmp_path / "iteration.json").read_text())
    candidate = payload["candidate_experiment"]
    assert candidate["status"] == "validated"
    assert candidate["eligible"] is True
    assert candidate["selected_candidate"] == "E1"
    assert candidate["selected_config"] == "E1_dock64_rerank10_diffdock0"
    assert candidate["preregistration_sha256"] == json.loads(
        prereg.read_text()
    )["preregistration_sha256"]
    assert candidate["paths"]["preregistration"]["sha256"] == file_sha256(prereg)
    assert candidate["paths"]["results"]["sha256"] == file_sha256(results)
    assert candidate["paths"]["selection"]["sha256"] == file_sha256(selection)
    assert candidate["guardrail_status"]["status"] == "passed"
    assert payload["data_snapshot"]["candidate_preregistration"]["status"] == "present"
    assert payload["data_snapshot"]["candidate_results"]["status"] == "present"
    assert payload["data_snapshot"]["candidate_selection"]["status"] == "present"


def test_run_iteration_model_promotion_requires_candidate_eligibility_when_present(
    tmp_path: Path,
) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
    import prospective_promotion_eval as promotion

    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    metrics = eval_dir / "prospective_metrics.csv"
    pd.DataFrame([{
        "candidate_model_id": "E3",
        "previous_signed_model_id": "E0",
        "preregistration_sha256": "b" * 64,
        "data_cutoff": "train_through_2023_select_2024",
        "training_window": "through_2023",
        "dev_selection_window": "2024_dev_only",
        "prospective_release_date": "2026-09-01",
        "prospective_source_releases": "ChEMBL;BindingDB;GtoPdb",
        "n_compounds": 100,
        "n_targets": 30,
        "n_positives": 200,
        "target_macro_top30": 0.05,
        "paired_bootstrap_lower_delta_top30": 0.001,
        "paired_bootstrap_lower_delta_top10": -0.01,
        "paired_bootstrap_lower_delta_mrr": -0.01,
        "brier_upper_delta": 0.005,
        "log_loss_upper_delta": 0.005,
        "unseen_cluster_positives": 5,
        "unseen_target_clusters": 3,
    }]).to_csv(metrics, index=False)
    (eval_dir / "prospective_promotion_manifest.json").write_text(
        json.dumps(promotion.evaluate(metrics)) + "\n"
    )
    write_candidate_experiment(tmp_path, selected=False)

    res = run_iteration(
        tmp_path,
        [
            "--allow-empty-iteration",
            "--allow-incomplete-leakage",
            "--allow-incomplete-eval-inputs",
            "--allow-threshold-failure",
            "--allow-missing-input-runs",
        ],
    )

    assert res.returncode == 0, res.stderr
    payload = json.loads((tmp_path / "iteration.json").read_text())
    assert payload["prospective_promotion"]["model_promotion_ready"] is False
    assert payload["candidate_experiment"]["status"] == "validated"
    assert payload["candidate_experiment"]["eligible"] is False
    assert payload["release_governance"]["model_promotion_ready"] is False


def test_run_iteration_rejects_missing_leakage_audit_by_default(tmp_path: Path) -> None:
    target_classes = write_disagreement_inputs(tmp_path)

    res = run_iteration(
        tmp_path,
        ["--target-classes", str(target_classes), "--allow-incomplete-eval-inputs"],
    )

    assert res.returncode != 0
    assert "Leakage audit is required for claim-quality iterations" in res.stderr
    payload = json.loads((tmp_path / "iteration.json").read_text())
    assert payload["n_steps"] == 1
    assert payload["n_passed"] == 1
    assert payload["leakage_status"]["status"] == "not_run"


def test_run_iteration_ignores_stale_leakage_audit_when_step_did_not_run(tmp_path: Path) -> None:
    target_classes = write_disagreement_inputs(tmp_path)
    (tmp_path / "eval" / "leakage_audit.csv").write_text("audit_status,leak_flag\n")

    res = run_iteration(
        tmp_path,
        ["--target-classes", str(target_classes), "--allow-incomplete-eval-inputs"],
    )

    assert res.returncode != 0
    assert "Leakage audit is required for claim-quality iterations" in res.stderr
    payload = json.loads((tmp_path / "iteration.json").read_text())
    assert payload["leakage_status"]["status"] == "not_run"
    assert payload["leakage_status"]["n_rows"] == 0


def test_run_iteration_parses_string_false_leak_flags(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    pd.DataFrame([{
        "uniprot": "P1",
        "smiles": "CCO",
        "seq_id": 0.1,
        "ligand_tanimoto": 0.1,
        "pocket_sucos": 0.1,
        "leak_flag": "False",
        "audit_status": "ok",
        "missing_axes": "",
    }]).to_csv(eval_dir / "leakage_audit.csv", index=False)

    status = run_iteration_module._leakage_status(eval_dir)
    assert status["status"] == "ok"
    assert status["n_leak_flags"] == 0


def test_run_iteration_rejects_inconsistent_ok_leakage_audit(
    tmp_path: Path,
) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    pd.DataFrame([{
        "uniprot": "P1",
        "smiles": "CCO",
        "seq_id": "",
        "ligand_tanimoto": 0.1,
        "pocket_sucos": 0.1,
        "leak_flag": False,
        "audit_status": "ok",
        "missing_axes": "sequence",
    }]).to_csv(eval_dir / "leakage_audit.csv", index=False)

    status = run_iteration_module._leakage_status(eval_dir)
    assert status["status"] == "invalid"
    assert status["n_invalid_rows"] == 1


def test_run_iteration_rejects_forged_leakage_score_flag_mismatch(
    tmp_path: Path,
) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    pd.DataFrame([{
        "uniprot": "P1",
        "smiles": "CCO",
        "seq_id": 1.0,
        "ligand_tanimoto": 0.1,
        "pocket_sucos": 0.1,
        "leak_flag": False,
        "audit_status": "ok",
        "missing_axes": "",
    }]).to_csv(eval_dir / "leakage_audit.csv", index=False)

    status = run_iteration_module._leakage_status(eval_dir)
    assert status["status"] == "invalid"
    assert status["n_invalid_rows"] == 1


def test_run_iteration_rejects_leakage_audit_missing_identity_columns(
    tmp_path: Path,
) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    pd.DataFrame([{
        "leak_flag": False,
        "audit_status": "ok",
        "missing_axes": "",
    }]).to_csv(eval_dir / "leakage_audit.csv", index=False)

    status = run_iteration_module._leakage_status(eval_dir)
    assert status["status"] == "invalid"
    assert status["missing_columns"] == [
        "ligand_tanimoto",
        "pocket_sucos",
        "seq_id",
        "smiles",
        "uniprot",
    ]


def test_run_iteration_rejects_positive_leak_flags_by_default(tmp_path: Path) -> None:
    target_classes = write_disagreement_inputs(tmp_path)
    leakage_args = write_clean_leakage_inputs(tmp_path, leak=True)

    res = run_iteration(
        tmp_path,
        [
            "--target-classes", str(target_classes),
            "--allow-incomplete-eval-inputs",
            *leakage_args,
        ],
    )

    assert res.returncode != 0
    assert "Detected leakage flags block claim-quality iterations" in res.stderr
    assert "status=leak_detected" in res.stderr
    payload = json.loads((tmp_path / "iteration.json").read_text())
    assert payload["leakage_status"]["status"] == "leak_detected"
    assert payload["leakage_status"]["n_leak_flags"] == 1


def test_run_iteration_rejects_claim_without_checked_thresholds(
    tmp_path: Path,
) -> None:
    target_classes = write_disagreement_inputs(tmp_path)
    leakage_args = write_clean_leakage_inputs(tmp_path)

    res = run_iteration(
        tmp_path,
        [
            "--target-classes", str(target_classes),
            "--allow-incomplete-eval-inputs",
            *leakage_args,
        ],
    )

    assert res.returncode != 0
    assert "No evaluation metric thresholds were checked" in res.stderr
    payload = json.loads((tmp_path / "iteration.json").read_text())
    assert payload["n_passed"] == 2
    assert payload["leakage_status"]["status"] == "ok"
    assert payload["threshold_status"]["n_checked"] == 0


def test_run_iteration_leakage_override_does_not_allow_positive_leak_flags(
    tmp_path: Path,
) -> None:
    target_classes = write_disagreement_inputs(tmp_path)
    leakage_args = write_clean_leakage_inputs(tmp_path, leak=True)

    res = run_iteration(
        tmp_path,
        [
            "--target-classes", str(target_classes),
            "--allow-incomplete-leakage",
            "--allow-incomplete-eval-inputs",
            "--allow-threshold-failure",
            "--allow-missing-input-runs",
            *leakage_args,
        ],
    )

    assert res.returncode != 0
    assert "Detected leakage flags block claim-quality iterations" in res.stderr
    assert "n_leak_flags=1" in res.stderr
    payload = json.loads((tmp_path / "iteration.json").read_text())
    assert payload["n_passed"] == 1
    assert payload["leakage_status"]["status"] == "leak_detected"
    assert payload["leakage_status"]["sealed_audit"]["status"] == "failed"


def test_run_iteration_records_disagreement_step_for_explicit_diagnostics(tmp_path: Path) -> None:
    target_classes = write_disagreement_inputs(tmp_path)

    res = run_iteration(
        tmp_path,
        [
            "--target-classes", str(target_classes),
            "--allow-incomplete-leakage",
            "--allow-incomplete-eval-inputs",
            "--allow-threshold-failure",
            "--allow-missing-input-runs",
        ],
    )

    assert res.returncode == 0, res.stderr
    payload = json.loads((tmp_path / "iteration.json").read_text())
    assert payload["n_steps"] == 1
    assert payload["n_passed"] == 1
    assert payload["steps"][0]["name"] == "disagreement"
    assert payload["provenance"]["git_commit"]
    assert payload["provenance"]["config_sha256"] != "missing"
    assert payload["provenance"]["tool_versions"]["python"]
    assert payload["leakage_status"]["status"] == "not_run"
    assert payload["ranking_metrics"]["disagreement"]["n_classes"] == 1
    assert payload["top_target_rationale"] == [{
        "ranking": str(tmp_path / "rankings" / "cold_start__comprehensive.csv"),
        "target_id": "P1",
        "score": 0.95,
        "rationale": {
            "source_count": 4,
            "sources": "autodock;gnina;rtmscore;boltz",
            "skin_score": 0.8,
        },
    }]
    assert any(
        item["relative_path"] == "disagreement_analysis.json"
        for item in payload["artifact_snapshot"]
    )
    assert (tmp_path / "eval" / "disagreement_per_class.csv").exists()


def test_ranking_metrics_rejects_invalid_disagreement_bucket(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    pd.DataFrame([{
        "bucket": "unknown",
        "class": "kinase",
        "count": 1,
    }]).to_csv(eval_dir / "disagreement_per_class.csv", index=False)

    try:
        ranking_metrics(eval_dir)
    except SystemExit as exc:
        assert "invalid bucket values" in str(exc)
    else:
        raise AssertionError("invalid disagreement bucket was accepted")


def test_ranking_metrics_rejects_duplicate_disagreement_class_rows(
    tmp_path: Path,
) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    pd.DataFrame([
        {"bucket": "docking_only", "class": "kinase", "count": 1},
        {"bucket": "docking_only", "class": "kinase", "count": 2},
    ]).to_csv(eval_dir / "disagreement_per_class.csv", index=False)

    try:
        ranking_metrics(eval_dir)
    except SystemExit as exc:
        assert "duplicate bucket/class rows" in str(exc)
    else:
        raise AssertionError("duplicate disagreement class rows were accepted")


def test_ranking_metrics_summarizes_valid_disagreement_counts(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    pd.DataFrame([
        {"bucket": "docking_only", "class": "kinase", "count": 1},
        {"bucket": "dti_only", "class": "gpcr", "count": 2},
    ]).to_csv(eval_dir / "disagreement_per_class.csv", index=False)

    metrics = ranking_metrics(eval_dir)

    assert metrics["disagreement"] == {
        "n_rows": 2,
        "n_classes": 2,
        "n_targets": 3,
        "buckets": ["docking_only", "dti_only"],
    }
