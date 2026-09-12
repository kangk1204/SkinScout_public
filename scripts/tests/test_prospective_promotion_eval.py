"""Tests for preregistered prospective Discovery promotion governance."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
import prospective_promotion_eval as promotion  # noqa: E402


def promotion_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
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
    }
    row.update(overrides)
    return row


def test_aggregate_metrics_cannot_authorize_prospective_promotion(
    tmp_path: Path,
) -> None:
    metrics = tmp_path / "metrics.csv"
    pd.DataFrame([promotion_row()]).to_csv(metrics, index=False)

    payload = promotion.evaluate(metrics)

    assert payload["status"] == "retain_previous_signed_model"
    assert payload["decision_scope"] == {
        "software_release": "independent",
        "model_promotion": "prospective_gate",
        "sota_claim": "not_evaluated_by_this_gate",
    }
    assert payload["evidence_scope"] == "aggregate_diagnostics_only_not_promotion_authority"
    assert [gate["name"] for gate in payload["failed_gates"]] == [
        "sealed_row_level_recomputation"
    ]
    promotion.validate_manifest(payload, metrics_csv=metrics)


def test_prospective_promotion_retains_previous_model_on_failed_gate(
    tmp_path: Path,
) -> None:
    metrics = tmp_path / "metrics.csv"
    pd.DataFrame([promotion_row(target_macro_top30=0.049)]).to_csv(metrics, index=False)

    payload = promotion.evaluate(metrics)

    assert payload["status"] == "retain_previous_signed_model"
    assert [gate["name"] for gate in payload["failed_gates"]] == [
        "sealed_row_level_recomputation",
        "target_macro_top30",
    ]


def test_prospective_promotion_rejects_non_preregistered_training_window(
    tmp_path: Path,
) -> None:
    metrics = tmp_path / "metrics.csv"
    pd.DataFrame([promotion_row(training_window="through_2024")]).to_csv(metrics, index=False)

    with pytest.raises(SystemExit, match="training_window must be through_2023"):
        promotion.evaluate(metrics)


def test_prospective_promotion_cli_writes_failed_manifest_with_override(
    tmp_path: Path,
) -> None:
    metrics = tmp_path / "metrics.csv"
    out = tmp_path / "manifest.json"
    pd.DataFrame([promotion_row(n_compounds=99)]).to_csv(metrics, index=False)

    res = subprocess.run(
        [
            sys.executable,
            "eval/prospective_promotion_eval.py",
            "--metrics-csv",
            str(metrics),
            "--out-json",
            str(out),
            "--allow-failed-promotion",
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode == 0, res.stderr
    payload = json.loads(out.read_text())
    assert payload["status"] == "retain_previous_signed_model"


def test_prospective_manifest_validator_rejects_minimal_handwritten_payload(
    tmp_path: Path,
) -> None:
    metrics = tmp_path / "metrics.csv"
    pd.DataFrame([promotion_row()]).to_csv(metrics, index=False)

    with pytest.raises(ValueError, match="decision_scope"):
        promotion.validate_manifest(
            {
                "schema_version": "skinscout.prospective-discovery-promotion.v1",
                "status": "promote",
                "failed_gates": [],
            },
            metrics_csv=metrics,
        )


def test_prospective_manifest_validator_rejects_stale_input_hash(
    tmp_path: Path,
) -> None:
    metrics = tmp_path / "metrics.csv"
    pd.DataFrame([promotion_row()]).to_csv(metrics, index=False)
    payload = promotion.evaluate(metrics)
    payload["input"]["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="input.sha256"):
        promotion.validate_manifest(payload, metrics_csv=metrics)


def test_prospective_manifest_validator_rejects_gate_status_mismatch(
    tmp_path: Path,
) -> None:
    metrics = tmp_path / "metrics.csv"
    pd.DataFrame([promotion_row()]).to_csv(metrics, index=False)
    payload = promotion.evaluate(metrics)
    payload["gates"][1]["status"] = "failed"

    with pytest.raises(ValueError, match="status must match passed"):
        promotion.validate_manifest(payload, metrics_csv=metrics)


def test_prospective_manifest_validator_rejects_forged_aggregate_authority(
    tmp_path: Path,
) -> None:
    metrics = tmp_path / "metrics.csv"
    pd.DataFrame([promotion_row()]).to_csv(metrics, index=False)
    payload = promotion.evaluate(metrics)
    payload["gates"][0].update({"observed": True, "passed": True, "status": "passed"})
    payload["failed_gates"] = []
    payload["status"] = "promote"

    with pytest.raises(ValueError, match="must fail the sealed row-level"):
        promotion.validate_manifest(payload, metrics_csv=metrics)


def test_prospective_manifest_rejects_metrics_changed_after_generation(
    tmp_path: Path,
) -> None:
    metrics = tmp_path / "metrics.csv"
    pd.DataFrame([promotion_row()]).to_csv(metrics, index=False)
    payload = promotion.evaluate(metrics)
    pd.DataFrame([promotion_row(target_macro_top30=0.01)]).to_csv(
        metrics,
        index=False,
    )

    with pytest.raises(ValueError, match="input.sha256"):
        promotion.validate_manifest_against_metrics(
            payload,
            metrics_csv=metrics,
        )


def test_prospective_manifest_rejects_forged_observed_gate_value(
    tmp_path: Path,
) -> None:
    metrics = tmp_path / "metrics.csv"
    pd.DataFrame([promotion_row()]).to_csv(metrics, index=False)
    payload = promotion.evaluate(metrics)
    payload["gates"][5]["observed"] = 0.99

    with pytest.raises(ValueError, match="fresh evaluation"):
        promotion.validate_manifest_against_metrics(
            payload,
            metrics_csv=metrics,
        )
