"""Regression tests for Daina benchmark comparison guardrails."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "eval" / "compare_daina_benchmarks.py"


def run_compare(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def write_benchmark(
    root: Path,
    *,
    case_top10: float = 0.8,
    target_top10: float = 0.5,
    target_top30: float = 0.7,
    target_pair_mrr: float = 0.4,
    mean_finite_rank: float = 8.0,
    median_finite_rank: float = 6.0,
    ranked_target_coverage: float = 0.9,
    case_coverage: float = 1.0,
    evidence_snapshot: str = "snapshot-a",
) -> Path:
    root.mkdir(parents=True)
    summary = {
        "schema_version": "skinscout.sota_known_target_recovery.v1",
        "evaluation_mode": "leave-query-out",
        "evaluation_metadata": {
            "mode": "leave-query-out",
            "evidence_snapshot": evidence_snapshot,
            "cutoff_date": "",
        },
        "thresholds": {
            "min_case_top10": 0.75,
            "min_target_top10": 0.5,
            "min_target_top30": 0.6,
        },
        "metrics": {
            "n_cases": 10,
            "n_targets": 20,
            "n_evaluated_cases": 10,
            "n_evaluated_targets": 20,
            "case_top10": case_top10,
            "target_top10": target_top10,
            "target_top30": target_top30,
            "target_pair_mrr": target_pair_mrr,
            "mean_finite_rank": mean_finite_rank,
            "median_finite_rank": median_finite_rank,
            "ranked_target_coverage": ranked_target_coverage,
            "case_coverage": case_coverage,
        },
        "passes_threshold": True,
    }
    (root / "skin_known_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    summary_sha256 = hashlib.sha256(
        (root / "skin_known_summary.json").read_bytes()
    ).hexdigest()
    manifest = {
        "schema_version": "skinscout.daina-panel-benchmark.v1",
        "parameters": {
            "evidence_mode": "leave-query-out",
            "evaluation_mode": "leave-query-out",
            "evidence_snapshot_id": evidence_snapshot,
            "cutoff_date": "",
            "exclude_reference_similarity": 0.85,
            "quality_policy": "claim-grade",
            "scoring_method": "quality-hybrid",
            "thresholds": {
                "min_case_top10": 0.75,
                "min_target_top10": 0.5,
                "min_target_top30": 0.6,
                "allow_threshold_failure": False,
            },
        },
        "inputs": {
            "panel_csv_sha256": "a" * 64,
            "source_manifest_sha256": "b" * 64,
            "fingerprint_manifest_sha256": "c" * 64,
            "chembl_fp_sha256": "d" * 64,
            "human_activities_sha256": "e" * 64,
        },
        "outputs": {
            "summary_path": "skin_known_summary.json",
            "summary_sha256": summary_sha256,
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return root


def test_valid_comparison_writes_absolute_metrics_deltas_and_selection(
    tmp_path: Path,
) -> None:
    baseline = write_benchmark(tmp_path / "baseline", case_top10=0.80, mean_finite_rank=8)
    candidate = write_benchmark(tmp_path / "candidate", case_top10=0.90, mean_finite_rank=5)
    out_csv = tmp_path / "comparison.csv"
    out_json = tmp_path / "comparison.json"

    result = run_compare([
        "--baseline",
        str(baseline),
        "--candidate",
        str(candidate),
        "--out-csv",
        str(out_csv),
        "--out-json",
        str(out_json),
        "--selection-split",
        "development",
        "--select-best",
        "--primary-metric",
        "case_top10",
        "--tie-break-metric",
        "mean_finite_rank",
    ])

    assert result.returncode == 0, result.stderr
    rows = list(csv.DictReader(out_csv.open()))
    assert float(rows[0]["baseline_case_top10"]) == 0.8
    assert float(rows[0]["candidate_case_top10"]) == 0.9
    assert abs(float(rows[0]["delta_case_top10"]) - 0.1) < 1e-12
    assert float(rows[0]["delta_mean_finite_rank"]) == -3.0
    assert float(rows[0]["improvement_mean_finite_rank"]) == 3.0
    payload = json.loads(out_json.read_text())
    assert payload["selection"]["selected_candidate_label"] == "candidate"
    assert payload["baseline"]["recipe_identity"] == {
        "quality_policy": "claim-grade",
        "scoring_method": "quality-hybrid",
    }
    assert payload["metric_directions"]["mean_finite_rank"] == "lower_is_better"
    assert payload["significance_claim"] == "not_assessed"


def test_mismatched_snapshot_rejected_without_outputs(tmp_path: Path) -> None:
    baseline = write_benchmark(tmp_path / "baseline", evidence_snapshot="snapshot-a")
    candidate = write_benchmark(tmp_path / "candidate", evidence_snapshot="snapshot-b")
    out_csv = tmp_path / "comparison.csv"
    out_json = tmp_path / "comparison.json"
    out_csv.write_text("stale\n")
    out_json.write_text("stale\n")

    result = run_compare([
        "--baseline",
        str(baseline),
        "--candidate",
        str(candidate),
        "--out-csv",
        str(out_csv),
        "--out-json",
        str(out_json),
        "--selection-split",
        "development",
    ])

    assert result.returncode != 0
    assert "evidence_snapshot_id differs from baseline" in result.stderr
    assert not out_csv.exists()
    assert not out_json.exists()


def test_final_test_selection_rejected(tmp_path: Path) -> None:
    baseline = write_benchmark(tmp_path / "baseline")
    candidate = write_benchmark(tmp_path / "candidate")

    result = run_compare([
        "--baseline",
        str(baseline),
        "--candidate",
        str(candidate),
        "--out-csv",
        str(tmp_path / "comparison.csv"),
        "--out-json",
        str(tmp_path / "comparison.json"),
        "--selection-split",
        "final-test",
        "--select-best",
        "--primary-metric",
        "case_top10",
        "--tie-break-metric",
        "mean_finite_rank",
    ])

    assert result.returncode != 0
    assert "--select-best is only allowed" in result.stderr
