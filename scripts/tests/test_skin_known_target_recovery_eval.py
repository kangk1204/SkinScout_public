"""Regression tests for skin known-target recovery evaluation."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "eval" / "skin_known_target_recovery_eval.py"


def run_eval(tmp_path: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--cases-csv", str(tmp_path / "cases.csv"),
            "--rankings-dir", str(tmp_path / "rankings"),
            "--out-csv", str(tmp_path / "skin_known_cases.csv"),
            "--out-target-csv", str(tmp_path / "skin_known_targets.csv"),
            *extra,
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def write_case(tmp_path: Path) -> None:
    (tmp_path / "rankings").mkdir()
    pd.DataFrame([{
        "case_id": "Retinol",
        "inci_name": "Retinol",
        "panel": "beneficial",
        "skin_effect": "anti-aging retinoid",
        "smiles": "CCO",
        "known_targets": "P1;P2",
        "known_target_labels": "Target 1;Target 2",
    }]).to_csv(tmp_path / "cases.csv", index=False)


def write_daina_sidecar(
    ranking_path: Path,
    *,
    evidence_mode: str,
    evidence_snapshot: str,
    cutoff_date: str | None,
    case_id: str = "Retinol",
) -> Path:
    sidecar = ranking_path.with_suffix(".metadata.json")
    payload = {
        "schema_version": "skinscout.daina-run.v1",
        "case_id": case_id,
        "evidence_mode": evidence_mode,
        "evidence_snapshot_id": evidence_snapshot,
        "cutoff_date": cutoff_date,
        "score_is_calibrated_probability": False,
        "ranking_sha256": hashlib.sha256(ranking_path.read_bytes()).hexdigest(),
    }
    sidecar.write_text(json.dumps(payload, indent=2) + "\n")
    return sidecar


def test_skin_known_target_default_panel_parses_in_diagnostic_mode(
    tmp_path: Path,
) -> None:
    rankings = tmp_path / "rankings"
    rankings.mkdir()
    out_cases = tmp_path / "panel_cases.csv"
    out_targets = tmp_path / "panel_targets.csv"

    res = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--rankings-dir", str(rankings),
            "--out-csv", str(out_cases),
            "--out-target-csv", str(out_targets),
            "--allow-missing-rankings",
            "--allow-threshold-failure",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert res.returncode == 0, res.stderr
    cases = pd.read_csv(out_cases)
    assert {"Retinol", "Hydroquinone", "Caffeine"} <= set(cases["case_id"])
    assert set(cases["status"]) == {"no_ranking"}
    assert not bool(cases["passes_threshold"].iloc[0])
    targets = pd.read_csv(out_targets)
    assert {"P10276", "P14679", "Q8NER1", "Q7Z2W7"} <= set(targets["target_id"])


def test_skin_known_target_fails_on_missing_ranking_by_default(
    tmp_path: Path,
) -> None:
    write_case(tmp_path)
    (tmp_path / "skin_known_cases.csv").write_text("stale\n")
    (tmp_path / "skin_known_targets.csv").write_text("stale\n")

    res = run_eval(tmp_path)

    assert res.returncode != 0
    assert "Missing 1 skin known-target ranking" in res.stderr
    assert not (tmp_path / "skin_known_cases.csv").exists()
    assert not (tmp_path / "skin_known_targets.csv").exists()


def test_skin_known_target_allow_missing_is_diagnostic(tmp_path: Path) -> None:
    write_case(tmp_path)

    res = run_eval(
        tmp_path,
        "--allow-missing-rankings",
        "--allow-threshold-failure",
    )

    assert res.returncode == 0, res.stderr
    cases = pd.read_csv(tmp_path / "skin_known_cases.csv")
    targets = pd.read_csv(tmp_path / "skin_known_targets.csv")
    assert cases.iloc[0]["status"] == "no_ranking"
    assert set(targets["status"]) == {"no_ranking"}
    assert not bool(cases.iloc[0]["passes_threshold"])


def test_skin_known_target_rejects_label_count_mismatch(tmp_path: Path) -> None:
    write_case(tmp_path)
    (tmp_path / "skin_known_cases.csv").write_text("stale\n")
    pd.DataFrame([{
        "case_id": "Retinol",
        "inci_name": "Retinol",
        "panel": "beneficial",
        "skin_effect": "anti-aging retinoid",
        "smiles": "CCO",
        "known_targets": "P1;P2",
        "known_target_labels": "Target 1",
    }]).to_csv(tmp_path / "cases.csv", index=False)

    res = run_eval(tmp_path)

    assert res.returncode != 0
    assert "known_target_labels count must match known_targets count" in res.stderr
    assert not (tmp_path / "skin_known_cases.csv").exists()


def test_skin_known_target_rejects_duplicate_ranking_target_id(
    tmp_path: Path,
) -> None:
    write_case(tmp_path)
    (tmp_path / "skin_known_cases.csv").write_text("stale\n")
    pd.DataFrame([{"target_id": "P1"}, {"target_id": "P1"}]).to_csv(
        tmp_path / "rankings" / "Retinol__ranked_targets_v3.csv",
        index=False,
    )

    res = run_eval(tmp_path)

    assert res.returncode != 0
    assert "Known-target ranking contains duplicate target_id values: P1" in res.stderr
    assert not (tmp_path / "skin_known_cases.csv").exists()


def test_skin_known_target_writes_case_and_target_recovery(
    tmp_path: Path,
) -> None:
    write_case(tmp_path)
    pd.DataFrame([
        {
            "target_id": "P2",
            "final_score": 0.91,
            "known_target_prior": 0.0,
            "known_target_prior_norm": 0.0,
            "gene_symbol": "GENE2",
            "protein_name": "Target 2 protein",
        },
        {
            "target_id": "P3",
            "final_score": 0.82,
            "known_target_prior": 0.0,
            "known_target_prior_norm": 0.0,
        },
    ]).to_csv(
        tmp_path / "rankings" / "Retinol__ranked_targets_v3_with_efficacy.csv",
        index=False,
    )

    res = run_eval(tmp_path)

    assert res.returncode == 0, res.stderr
    case = pd.read_csv(tmp_path / "skin_known_cases.csv").iloc[0]
    assert case["status"] == "evaluated"
    assert case["case_top1"] == 1
    assert case["target_top1_fraction"] == 0.5
    assert case["target_pair_mrr"] == 0.5
    assert case["mean_finite_rank"] == 1.0
    assert case["median_finite_rank"] == 1.0
    assert case["ranked_target_coverage"] == 0.5
    assert case["case_coverage"] == 1
    assert case["best_known_target_id"] == "P2"
    assert case["known_target_ranks"] == "P1:NA;P2:1"
    assert bool(case["passes_threshold"])
    targets = pd.read_csv(tmp_path / "skin_known_targets.csv")
    missing = targets[targets["target_id"] == "P1"].iloc[0]
    assert missing["has_finite_rank"] == 0
    assert missing["reciprocal_rank"] == 0.0
    recovered = targets[targets["target_id"] == "P2"].iloc[0]
    assert recovered["rank"] == 1
    assert recovered["has_finite_rank"] == 1
    assert recovered["reciprocal_rank"] == 1.0
    assert recovered["target_top1"] == 1
    assert recovered["gene_symbol"] == "GENE2"


def test_skin_known_target_writes_sota_summary_schema(tmp_path: Path) -> None:
    write_case(tmp_path)
    summary = tmp_path / "skin_known_summary.json"
    pd.DataFrame([
        {
            "target_id": "P2",
            "final_score": 0.91,
            "known_target_prior": 0.0,
            "known_target_prior_norm": 0.0,
            "gene_symbol": "GENE2",
            "protein_name": "Target 2 protein",
        },
        {
            "target_id": "P3",
            "final_score": 0.82,
            "known_target_prior": 0.0,
            "known_target_prior_norm": 0.0,
        },
    ]).to_csv(
        tmp_path / "rankings" / "Retinol__ranked_targets_v3_with_efficacy.csv",
        index=False,
    )

    res = run_eval(
        tmp_path,
        "--out-summary-json", str(summary),
        "--context-profile", "anti_aging",
        "--min-case-top10", "0.50",
        "--min-target-top10", "0.50",
        "--min-target-top30", "0.50",
    )

    assert res.returncode == 0, res.stderr
    cases = pd.read_csv(tmp_path / "skin_known_cases.csv")
    targets = pd.read_csv(tmp_path / "skin_known_targets.csv")
    assert set(cases["context_profile"]) == {"anti_aging"}
    assert set(targets["effect_direction"]) == {"beneficial"}
    payload = json.loads(summary.read_text())
    assert payload["schema_version"] == "skinscout.sota_known_target_recovery.v1"
    assert payload["context_profile"] == "anti_aging"
    assert payload["metrics"]["n_cases"] == 1
    assert payload["metrics"]["n_evaluated_targets"] == 2
    assert payload["metrics"]["case_top10"] == 1.0
    assert payload["metrics"]["target_top10"] == 0.5
    assert payload["passes_threshold"] is True
    assert payload["evaluation_mode"] == "retrospective"
    assert payload["evaluation_metadata"] == {
        "mode": "retrospective",
        "evidence_snapshot": "",
        "cutoff_date": "",
    }
    assert payload["metrics"]["target_pair_mrr"] == 0.5
    assert payload["metrics"]["mean_finite_rank"] == 1.0
    assert payload["metrics"]["median_finite_rank"] == 1.0
    assert payload["metrics"]["ranked_target_coverage"] == 0.5
    assert payload["metrics"]["case_coverage"] == 1.0
    assert payload["denominators"]["target_pair_mrr"] == 2
    assert payload["denominators"]["ranked_target_coverage_numerator"] == 1
    assert payload["denominators"]["ranked_target_coverage_denominator"] == 2
    assert payload["denominators"]["case_coverage_numerator"] == 1
    assert payload["denominators"]["case_coverage_denominator"] == 1


def test_skin_known_target_temporal_requires_snapshot_and_cutoff(
    tmp_path: Path,
) -> None:
    write_case(tmp_path)
    (tmp_path / "skin_known_cases.csv").write_text("stale\n")
    (tmp_path / "skin_known_targets.csv").write_text("stale\n")

    res = run_eval(tmp_path, "--evaluation-mode", "temporal")

    assert res.returncode != 0
    assert "temporal requires both --evidence-snapshot and --cutoff-date" in res.stderr
    assert not (tmp_path / "skin_known_cases.csv").exists()
    assert not (tmp_path / "skin_known_targets.csv").exists()


def test_skin_known_target_records_temporal_metadata_when_closed(
    tmp_path: Path,
) -> None:
    write_case(tmp_path)
    summary = tmp_path / "skin_known_summary.json"
    ranking_path = tmp_path / "rankings" / "Retinol__ranked_targets_v3.csv"
    pd.DataFrame([
        {
            "target_id": "P2",
            "final_score": 0.91,
            "known_target_prior": 0.0,
            "known_target_prior_norm": 0.0,
        },
    ]).to_csv(ranking_path, index=False)
    write_daina_sidecar(
        ranking_path,
        evidence_mode="temporal",
        evidence_snapshot="snapshot-2024q1",
        cutoff_date="2024-03-31",
    )

    res = run_eval(
        tmp_path,
        "--out-summary-json", str(summary),
        "--evaluation-mode", "temporal",
        "--evidence-snapshot", "snapshot-2024q1",
        "--cutoff-date", "2024-03-31",
    )

    assert res.returncode == 0, res.stderr
    cases = pd.read_csv(tmp_path / "skin_known_cases.csv")
    targets = pd.read_csv(tmp_path / "skin_known_targets.csv")
    payload = json.loads(summary.read_text())
    assert set(cases["evaluation_mode"]) == {"temporal"}
    assert set(cases["evidence_snapshot"]) == {"snapshot-2024q1"}
    assert set(cases["cutoff_date"]) == {"2024-03-31"}
    assert set(targets["evaluation_mode"]) == {"temporal"}
    assert payload["evaluation_metadata"] == {
        "mode": "temporal",
        "evidence_snapshot": "snapshot-2024q1",
        "cutoff_date": "2024-03-31",
    }
    assert cases.iloc[0]["ranking_evidence_mode"] == "temporal"
    assert cases.iloc[0]["ranking_metadata_sha256"]
    assert payload["ranking_evidence_audit"][0]["ranking_sha256"]


def test_skin_known_target_leave_query_out_requires_ranking_sidecar(
    tmp_path: Path,
) -> None:
    write_case(tmp_path)
    pd.DataFrame([
        {
            "target_id": "P2",
            "final_score": 0.91,
            "known_target_prior": 0.0,
            "known_target_prior_norm": 0.0,
        },
    ]).to_csv(
        tmp_path / "rankings" / "Retinol__ranked_targets_v3.csv",
        index=False,
    )

    res = run_eval(
        tmp_path,
        "--evaluation-mode", "leave-query-out",
        "--evidence-snapshot", "chembl-fixture",
    )

    assert res.returncode != 0
    assert "requires a non-empty Daina ranking metadata sidecar" in res.stderr
    assert not (tmp_path / "skin_known_cases.csv").exists()


def test_skin_known_target_rejects_sidecar_mode_mismatch(tmp_path: Path) -> None:
    write_case(tmp_path)
    ranking_path = tmp_path / "rankings" / "Retinol__ranked_targets_v3.csv"
    pd.DataFrame([
        {
            "target_id": "P2",
            "final_score": 0.91,
            "known_target_prior": 0.0,
            "known_target_prior_norm": 0.0,
        },
    ]).to_csv(ranking_path, index=False)
    write_daina_sidecar(
        ranking_path,
        evidence_mode="retrieval",
        evidence_snapshot="chembl-fixture",
        cutoff_date=None,
    )

    res = run_eval(
        tmp_path,
        "--evaluation-mode", "leave-query-out",
        "--evidence-snapshot", "chembl-fixture",
    )

    assert res.returncode != 0
    assert "evidence_mode does not match evaluator mode" in res.stderr


def test_skin_known_target_rejects_ranking_changed_after_sidecar(tmp_path: Path) -> None:
    write_case(tmp_path)
    ranking_path = tmp_path / "rankings" / "Retinol__ranked_targets_v3.csv"
    ranking = pd.DataFrame([
        {
            "target_id": "P2",
            "final_score": 0.91,
            "known_target_prior": 0.0,
            "known_target_prior_norm": 0.0,
        },
    ])
    ranking.to_csv(ranking_path, index=False)
    write_daina_sidecar(
        ranking_path,
        evidence_mode="leave-query-out",
        evidence_snapshot="chembl-fixture",
        cutoff_date=None,
    )
    ranking.loc[0, "final_score"] = 0.12
    ranking.to_csv(ranking_path, index=False)

    res = run_eval(
        tmp_path,
        "--evaluation-mode", "leave-query-out",
        "--evidence-snapshot", "chembl-fixture",
    )

    assert res.returncode != 0
    assert "ranking SHA-256 does not match" in res.stderr


def test_skin_known_target_rejects_assisted_prior_by_default(
    tmp_path: Path,
) -> None:
    write_case(tmp_path)
    pd.DataFrame([
        {
            "target_id": "P2",
            "final_score": 0.91,
            "known_target_prior": 0.5,
            "known_target_prior_norm": 0.5,
        },
        {
            "target_id": "P3",
            "final_score": 0.82,
            "known_target_prior": 0.0,
            "known_target_prior_norm": 0.0,
        },
    ]).to_csv(
        tmp_path / "rankings" / "Retinol__ranked_targets_v3.csv",
        index=False,
    )

    res = run_eval(tmp_path)

    assert res.returncode != 0
    assert "assisted known-target prior" in res.stderr
    assert not (tmp_path / "skin_known_cases.csv").exists()


def test_skin_known_target_records_assisted_prior_when_explicitly_allowed(
    tmp_path: Path,
) -> None:
    write_case(tmp_path)
    summary = tmp_path / "nested" / "summary.json"
    pd.DataFrame([
        {
            "target_id": "P2",
            "final_score": 0.91,
            "known_target_prior": 1.0,
            "known_target_prior_norm": 1.0,
        },
        {
            "target_id": "P3",
            "final_score": 0.82,
            "known_target_prior": 0.0,
            "known_target_prior_norm": 0.0,
        },
    ]).to_csv(
        tmp_path / "rankings" / "Retinol__ranked_targets_v3.csv",
        index=False,
    )

    res = run_eval(
        tmp_path,
        "--allow-assisted-known-target-prior",
        "--out-summary-json", str(summary),
    )

    assert res.returncode == 0, res.stderr
    cases = pd.read_csv(tmp_path / "skin_known_cases.csv")
    targets = pd.read_csv(tmp_path / "skin_known_targets.csv")
    payload = json.loads(summary.read_text())
    assert set(cases["assisted_by_known_target_prior"]) == {True}
    assert set(targets["assisted_by_known_target_prior"]) == {True}
    assert payload["assisted_by_known_target_prior"] is True


def test_skin_known_target_validates_both_prior_audit_columns(
    tmp_path: Path,
) -> None:
    write_case(tmp_path)
    pd.DataFrame([
        {
            "target_id": "P2",
            "final_score": 0.91,
            "known_target_prior": 1.0,
            "known_target_prior_norm": "not-a-number",
        },
        {
            "target_id": "P3",
            "final_score": 0.82,
            "known_target_prior": 0.0,
            "known_target_prior_norm": 0.0,
        },
    ]).to_csv(
        tmp_path / "rankings" / "Retinol__ranked_targets_v3.csv",
        index=False,
    )

    res = run_eval(tmp_path, "--allow-assisted-known-target-prior")

    assert res.returncode != 0
    assert "known_target_prior_norm" in res.stderr
    assert "non-numeric assisted prior value" in res.stderr
    assert not (tmp_path / "skin_known_cases.csv").exists()


def test_skin_known_target_rejects_missing_prior_audit_columns(
    tmp_path: Path,
) -> None:
    write_case(tmp_path)
    pd.DataFrame([
        {"target_id": "P2", "final_score": 0.91},
        {"target_id": "P3", "final_score": 0.82},
    ]).to_csv(
        tmp_path / "rankings" / "Retinol__ranked_targets_v3.csv",
        index=False,
    )

    res = run_eval(tmp_path)

    assert res.returncode != 0
    assert "missing required prior-audit column" in res.stderr
    assert not (tmp_path / "skin_known_cases.csv").exists()


def test_declared_rank_column_is_not_renumbered_by_row_position(
    tmp_path: Path,
) -> None:
    """A subset export must keep its declared ranks.

    Renumbering by row position would report P1 at rank 1 and P2 at rank 2
    instead of the 7 and 41 the producer actually assigned.
    """
    write_case(tmp_path)
    pd.DataFrame(
        [
            {"rank": 7, "target_id": "P1", "score": 0.9,
             "known_target_prior": 0.0, "known_target_prior_norm": 0.0},
            {"rank": 41, "target_id": "P2", "score": 0.8,
             "known_target_prior": 0.0, "known_target_prior_norm": 0.0},
        ]
    ).to_csv(tmp_path / "rankings" / "Retinol__ranked_targets_v3.csv", index=False)

    res = run_eval(tmp_path, "--allow-threshold-failure")

    assert res.returncode == 0, res.stderr
    targets = pd.read_csv(tmp_path / "skin_known_targets.csv")
    ranks = dict(zip(targets["target_id"], targets["rank"], strict=True))
    assert ranks == {"P1": 7, "P2": 41}
    hits = dict(zip(targets["target_id"], targets["target_top10"], strict=True))
    assert hits == {"P1": 1, "P2": 0}
    top30 = dict(zip(targets["target_id"], targets["target_top30"], strict=True))
    assert top30 == {"P1": 1, "P2": 0}


def test_rank_metrics_use_the_same_denominator_as_top_k(tmp_path: Path) -> None:
    """MRR and top-k must average over the same population.

    The second case has no ranking file, so its two target pairs are
    `no_ranking`. Counting them in the MRR denominator but not in the top-k
    denominator would report two different panel sizes in one summary.
    """
    (tmp_path / "rankings").mkdir()
    pd.DataFrame(
        [
            {
                "case_id": "Retinol",
                "inci_name": "Retinol",
                "panel": "beneficial",
                "skin_effect": "anti-aging retinoid",
                "smiles": "CCO",
                "known_targets": "P1;P2",
                "known_target_labels": "Target 1;Target 2",
            },
            {
                "case_id": "Missing",
                "inci_name": "Missing",
                "panel": "beneficial",
                "skin_effect": "anti-aging retinoid",
                "smiles": "CCC",
                "known_targets": "P3;P4",
                "known_target_labels": "Target 3;Target 4",
            },
        ]
    ).to_csv(tmp_path / "cases.csv", index=False)
    pd.DataFrame(
        [
            {"rank": 1, "target_id": "P1", "score": 0.9,
             "known_target_prior": 0.0, "known_target_prior_norm": 0.0},
            {"rank": 2, "target_id": "P2", "score": 0.8,
             "known_target_prior": 0.0, "known_target_prior_norm": 0.0},
        ]
    ).to_csv(tmp_path / "rankings" / "Retinol__ranked_targets_v3.csv", index=False)

    res = run_eval(
        tmp_path,
        "--allow-missing-rankings",
        "--allow-threshold-failure",
        "--out-summary-json",
        str(tmp_path / "summary.json"),
    )

    assert res.returncode == 0, res.stderr
    summary = json.loads((tmp_path / "summary.json").read_text())
    metrics = summary["metrics"]
    assert metrics["n_targets"] == 4
    assert metrics["n_evaluated_targets"] == 2
    # Both P1 and P2 rank inside the top 10, so every evaluated-row metric is 1.0.
    assert metrics["target_top10"] == 1.0
    assert metrics["target_pair_mrr"] == pytest.approx(0.75)
    assert metrics["ranked_target_coverage"] == 1.0
    assert summary["denominators"]["target_pair_mrr"] == 2
    # case_coverage still reports over every declared case.
    assert summary["denominators"]["case_coverage_denominator"] == 2
    assert metrics["case_coverage"] == 0.5
