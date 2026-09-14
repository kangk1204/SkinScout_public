"""Regression tests for the compact performance-v2 evaluator."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
EVALUATOR_PATH = ROOT / "eval" / "evaluate_performance_v2.py"
PREREG_PATH = ROOT / "eval" / "preregistered_performance_v2.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prereg = _load_module("preregistered_performance_v2", PREREG_PATH)
evaluator = _load_module("evaluate_performance_v2", EVALUATOR_PATH)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    fieldnames = columns or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _ranking_panel_rows(*, dual: bool) -> list[dict[str, Any]]:
    pairs = (
        [("D1", ["T1", "T2"]), ("D2", ["T1"]), ("D3", ["T3"])]
        if dual
        else [("Q1", ["T1", "T2"]), ("Q2", ["T3"])]
    )
    rows = []
    for query_id, truths in pairs:
        row: dict[str, Any] = {
            "query_id": query_id,
            "truth_targets": truths,
            "n_truth_targets": len(truths),
            "split": "test" if dual else "dev",
        }
        if dual:
            row.update({flag: True for flag in evaluator.DUAL_COLD_FLAGS})
        rows.append(row)
    return rows


def _calibration_rows() -> list[dict[str, Any]]:
    return [
        {
            "pair_id": "A",
            "query_id": "C1",
            "uniprot": "T1",
            "endpoint_family": "direct_binding",
            "label": 1,
            "sample_weight": 2.0,
            "split": "dev",
            "pactivity_min": 6.0,
            "pactivity_max": 7.0,
        },
        {
            "pair_id": "B",
            "query_id": "C2",
            "uniprot": "T2",
            "endpoint_family": "direct_binding",
            "label": 0,
            "sample_weight": 1.0,
            "split": "dev",
            "pactivity_min": 4.0,
            "pactivity_max": 5.0,
        },
        {
            "pair_id": "C",
            "query_id": "C3",
            "uniprot": "T3",
            "endpoint_family": "functional",
            "label": 1,
            "sample_weight": 3.0,
            "split": "dev",
            "pactivity_min": 6.1,
            "pactivity_max": 8.0,
        },
        {
            "pair_id": "D",
            "query_id": "C4",
            "uniprot": "T4",
            "endpoint_family": "functional",
            "label": 0,
            "sample_weight": 1.0,
            "split": "dev",
            "pactivity_min": 3.0,
            "pactivity_max": 4.9,
        },
    ]


def _rank(candidate: str, query: str, target: str) -> float:
    if candidate != "P0":
        return 1.0
    return {
        ("Q1", "T1"): 1.5,
        ("Q1", "T2"): 11.0,
        ("Q2", "T3"): 31.0,
        ("D1", "T1"): 2.0,
        ("D1", "T2"): 3.0,
        ("D2", "T1"): 4.0,
        ("D3", "T3"): 5.0,
    }[(query, target)]


def _rank_rows() -> list[dict[str, Any]]:
    pairs = [
        (query, target)
        for rows in (_ranking_panel_rows(dual=False), _ranking_panel_rows(dual=True))
        for query, targets in ((row["query_id"], row["truth_targets"]) for row in rows)
        for target in targets
    ]
    return [
        {
            "candidate_id": candidate,
            "query_id": query,
            "target_id": target,
            "rank": _rank(candidate, query, target),
            "target_universe_size": 40,
            "tie_policy": "average",
        }
        for candidate in evaluator.CANDIDATES
        for query, target in pairs
    ]


def _calibration_evidence_rows(calibrators: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    probabilities = {"A": 0.9, "B": 0.2, "C": 0.8, "D": 0.1}
    panel = {row["pair_id"]: row for row in _calibration_rows()}
    return [
        {
            "candidate_id": candidate,
            "pair_id": pair_id,
            "query_id": row["query_id"],
            "target_id": row["uniprot"],
            "endpoint_family": row["endpoint_family"],
            "calibrated_measured_event_probability": probabilities[pair_id],
            "calibrator_sha256": calibrators[candidate][row["endpoint_family"]],
        }
        for candidate in evaluator.CANDIDATES
        for pair_id, row in panel.items()
    ]


def _resources(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidates": [
            {
                "candidate_id": candidate,
                "run_id": f"run-{candidate}",
                "model_hash": contract["models"][candidate]["model_sha256"],
                "peak_vram": index,
                "params": index * 100,
                "cold_latency": index + 0.5,
            }
            for index, candidate in enumerate(evaluator.CANDIDATES, start=1)
        ]
    }


def _fixture(tmp_path: Path) -> dict[str, Any]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    target_universe = tmp_path / "target_universe.csv"
    _write_csv(
        target_universe,
        [
            {
                "target_id": f"T{index}",
                "target_cluster_30": "C1" if index in {1, 2} else f"C{index}",
            }
            for index in range(1, 41)
        ],
    )
    dev = tmp_path / "dev_ranking_queries.parquet"
    dual = tmp_path / "dual_cold_ranking_queries.parquet"
    calibration = tmp_path / "dev_calibration_pairs.parquet"
    pd.DataFrame(_ranking_panel_rows(dual=False)).to_parquet(dev, index=False)
    pd.DataFrame(_ranking_panel_rows(dual=True)).to_parquet(dual, index=False)
    pd.DataFrame(_calibration_rows()).to_parquet(calibration, index=False)

    panel_manifest = tmp_path / "panel_manifest.json"
    panel_payload = {
        "schema_version": evaluator.RECOVERY_PANEL_SCHEMA_VERSION,
        "algorithm": {
            "positive_threshold": 6.0,
            "negative_threshold": 5.0,
            "endpoint_families": {
                "KD": "direct_binding",
                "KI": "direct_binding",
                "IC50": "functional",
                "EC50": "functional",
            },
        },
        "passes_panel_adequacy_gate": True,
        "outputs": {
            "dev_ranking_queries.parquet": {"sha256": _sha(dev), "rows": 2},
            "dev_calibration_pairs.parquet": {"sha256": _sha(calibration), "rows": 4},
            "dual_cold_ranking_queries.parquet": {"sha256": _sha(dual), "rows": 3},
        },
        "selection": {
            "ranking_queries": {"dual_cold": {"adequacy": {"passes": True}}},
            "calibration_pairs": {
                "dev": {
                    "gray_excluded": 17,
                    "conflicts": {
                        "conflicting_measurement_groups_excluded": 2,
                        "conflicting_measurement_rows_excluded": 5,
                    },
                }
            },
        },
    }
    panel_manifest.write_text(json.dumps(panel_payload, sort_keys=True) + "\n", encoding="utf-8")
    training = tmp_path / "training.json"
    training.write_text('{"training":"fixture"}\n', encoding="utf-8")
    artifact_paths = {
        "training_data": training,
        "target_universe": target_universe,
        "evaluation_panel_manifest": panel_manifest,
        "dev_ranking_queries": dev,
        "dev_calibration_pairs": calibration,
        "dual_cold_ranking_queries": dual,
    }
    raw_contract = {
        "schema_version": prereg.SCHEMA_VERSION,
        "candidates": prereg.REQUIRED_CANDIDATES,
        "training_window": prereg.TRAINING_WINDOW,
        "selection_window": prereg.SELECTION_WINDOW,
        "artifacts": {
            name: {"path": str(path), "sha256": _sha(path)}
            for name, path in artifact_paths.items()
        },
        "models": {
            candidate: {
                "model_id": f"model-{candidate}",
                "version": "fixture",
                "license": "test-only",
                "license_profile": "research",
                "commercial_fallback_id": "fallback",
                "model_sha256": _digest(candidate),
            }
            for candidate in evaluator.CANDIDATES
        },
        "seeds": prereg.SEEDS,
        "label_policy": prereg.LABEL_POLICY,
        "budget": prereg.BUDGET,
        "dev_gates": prereg.DEV_GATES,
    }
    contract = prereg.seal_contract(raw_contract)
    prereg_path = tmp_path / "prereg.json"
    prereg_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rank_evidence = tmp_path / "rank_evidence.csv"
    _write_csv(rank_evidence, _rank_rows(), list(evaluator.RANK_EVIDENCE_COLUMNS))
    calibrators = {
        candidate: {
            endpoint: _digest(f"{candidate}-{endpoint}")
            for endpoint in evaluator.CALIBRATION_ENDPOINT_FAMILIES
        }
        for candidate in evaluator.CANDIDATES
    }
    calibration_evidence = tmp_path / "calibration_evidence.csv"
    _write_csv(
        calibration_evidence,
        _calibration_evidence_rows(calibrators),
        list(evaluator.CALIBRATION_EVIDENCE_COLUMNS),
    )
    evidence_manifest = tmp_path / "evidence_manifest.json"
    evidence_payload: dict[str, Any] = {
        "schema_version": evaluator.EVIDENCE_SCHEMA_VERSION,
        "candidates": list(evaluator.CANDIDATES),
        "artifacts": {
            "rank_evidence": {"sha256": _sha(rank_evidence), "rows": len(_rank_rows())},
            "calibration_evidence": {
                "sha256": _sha(calibration_evidence),
                "rows": len(_calibration_evidence_rows(calibrators)),
            },
        },
        "bindings": {
            "target_universe_sha256": contract["artifacts"]["target_universe"]["sha256"],
            "dev_ranking_queries_sha256": contract["artifacts"]["dev_ranking_queries"]["sha256"],
            "dev_calibration_pairs_sha256": contract["artifacts"]["dev_calibration_pairs"]["sha256"],
            "dual_cold_ranking_queries_sha256": contract["artifacts"]["dual_cold_ranking_queries"]["sha256"],
            "evaluation_panel_manifest_sha256": contract["artifacts"]["evaluation_panel_manifest"]["sha256"],
        },
        "model_sha256": {
            candidate: contract["models"][candidate]["model_sha256"]
            for candidate in evaluator.CANDIDATES
        },
        "target_universe_count": 40,
        "tie_policy": "average",
        "probability_contract": {
            "calibrated_only": True,
            "ontology_specific": True,
            "endpoint_families": list(evaluator.CALIBRATION_ENDPOINT_FAMILIES),
        },
        "calibrators": calibrators,
        "generator_sha256": _digest("fixture-generator"),
    }
    evidence_payload["binding_sha256"] = prereg.canonical_sha256(evidence_payload)
    evidence_manifest.write_text(
        json.dumps(evidence_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    resources = tmp_path / "resources.json"
    resources.write_text(json.dumps(_resources(contract), sort_keys=True) + "\n", encoding="utf-8")
    return {
        "prereg": prereg_path,
        "rank": rank_evidence,
        "calibration": calibration_evidence,
        "manifest": evidence_manifest,
        "resources": resources,
        "contract": contract,
        "target_universe": target_universe,
        "dual": dual,
        "panel_manifest": panel_manifest,
    }


def _run(tmp_path: Path, fixture: dict[str, Any] | None = None):
    fixture = fixture or _fixture(tmp_path)
    out = tmp_path / "results.csv"
    denominators = tmp_path / "denominators.json"
    assert evaluator.main(
        [
            "--rank-evidence-csv",
            str(fixture["rank"]),
            "--calibration-evidence-csv",
            str(fixture["calibration"]),
            "--evidence-manifest",
            str(fixture["manifest"]),
            "--prereg-json",
            str(fixture["prereg"]),
            "--resources",
            str(fixture["resources"]),
            "--out-csv",
            str(out),
            "--out-denominators-json",
            str(denominators),
        ]
    ) == 0
    return list(csv.DictReader(out.open(newline="", encoding="utf-8"))), json.loads(
        denominators.read_text(encoding="utf-8")
    )


def test_known_metrics_weighted_calibration_and_selector_compatibility(tmp_path: Path) -> None:
    rows, denominators = _run(tmp_path)
    p0 = next(row for row in rows if row["candidate_id"] == "P0")
    assert list(rows[0]) == evaluator.RESULT_COLUMNS
    assert float(p0["top10"]) == pytest.approx(1 / 3)
    assert float(p0["top30"]) == pytest.approx(2 / 3)
    # MRR is the best-rank reciprocal per query, averaged over queries.
    # Q1 has truths at 1.5 and 11.0 (best 1.5), Q2 at 31.0 (best 31.0):
    assert float(p0["mrr"]) == pytest.approx(((1 / 1.5) + (1 / 31.0)) / 2)
    assert float(p0["brier"]) == pytest.approx((2 * 0.1**2 + 0.2**2 + 3 * 0.2**2 + 0.1**2) / 7)
    expected_log = -(2 * math.log(0.9) + math.log(0.8) + 3 * math.log(0.8) + math.log(0.9)) / 7
    assert float(p0["log_loss"]) == pytest.approx(expected_log)
    assert denominators["target_universe_count"] == 40
    assert denominators["dev_query_count"] == 2
    assert prereg.evaluate_results(tmp_path / "prereg.json", tmp_path / "results.csv")


def test_real_label_policy_thresholds() -> None:
    assert evaluator._measurement_label(5.0) == "negative"
    assert evaluator._measurement_label(5.5) == "gray"
    assert evaluator._measurement_label(6.0) == "positive"


def test_full_universe_is_loaded_without_product_multiindex(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pd.MultiIndex, "from_product", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError()))
    rows, denominators = _run(tmp_path)
    assert len(rows) == 4
    assert denominators["target_universe_count"] == 40
    assert denominators["dev_truth_pair_count"] == 3


def test_dual_cold_target_denominators_and_clusters(tmp_path: Path) -> None:
    rows, denominators = _run(tmp_path)
    p0 = next(row for row in rows if row["candidate_id"] == "P0")
    assert float(p0["dual_cold_target_macro_top30"]) == pytest.approx(1.0)
    assert int(p0["dual_cold_truth_hits"]) == 4
    assert int(p0["dual_cold_unseen_target_clusters"]) == 2
    assert float(p0["effective_target_count"]) == pytest.approx(16 / 6)
    assert denominators["dual_cold_truth_target_count"] == 3
    assert denominators["dual_cold_truth_pair_count"] == 4


def test_conflict_and_gray_exclusions_are_sealed(tmp_path: Path) -> None:
    _, denominators = _run(tmp_path)
    assert denominators["gray_measurement_groups_excluded"] == 17
    assert denominators["conflicting_measurement_groups_excluded"] == 2
    assert denominators["conflicting_measurement_rows_excluded"] == 5
    binding = denominators.pop("binding_sha256")
    assert binding == prereg.canonical_sha256(denominators)


def test_calibration_requires_matching_ontology_and_calibrator(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    rows = list(csv.DictReader(fixture["calibration"].open(newline="", encoding="utf-8")))
    rows[0]["endpoint_family"] = "functional"
    _write_csv(fixture["calibration"], rows, list(evaluator.CALIBRATION_EVIDENCE_COLUMNS))
    panel = evaluator._calibration_panel(
        Path(fixture["contract"]["artifacts"]["dev_calibration_pairs"]["path"]),
        {f"T{index}" for index in range(1, 41)},
    )
    calibrators = json.loads(fixture["manifest"].read_text(encoding="utf-8"))["calibrators"]
    with pytest.raises(prereg.PerformanceV2ContractError, match="endpoint_family"):
        evaluator._calibration_evidence(fixture["calibration"], panel, calibrators)


def test_uncalibrated_probability_column_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    text = fixture["calibration"].read_text(encoding="utf-8").replace(
        "calibrated_measured_event_probability", "measured_event_probability", 1
    )
    fixture["calibration"].write_text(text, encoding="utf-8")
    with pytest.raises(prereg.PerformanceV2ContractError, match="columns"):
        list(
            evaluator._read_exact_csv(
                fixture["calibration"],
                evaluator.CALIBRATION_EVIDENCE_COLUMNS,
                "calibration evidence",
            )
        )


def test_dual_cold_requires_all_definition_flags(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    frame = pd.read_parquet(fixture["dual"])
    frame.loc[0, "absent_scaffold_from_train"] = False
    frame.to_parquet(fixture["dual"], index=False)
    with pytest.raises(prereg.PerformanceV2ContractError, match="dual-cold|claimable"):
        evaluator._ranking_panel(
            fixture["dual"],
            "dual-cold ranking panel",
            dual_cold=True,
            target_ids={f"T{index}" for index in range(1, 41)},
        )


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "nonfinite"])
def test_rank_evidence_coverage_and_values_fail_closed(tmp_path: Path, mutation: str) -> None:
    fixture = _fixture(tmp_path)
    rows = list(csv.DictReader(fixture["rank"].open(newline="", encoding="utf-8")))
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(dict(rows[0]))
    else:
        rows[0]["rank"] = "nan"
    _write_csv(fixture["rank"], rows, list(evaluator.RANK_EVIDENCE_COLUMNS))
    expected_pairs = {
        (query, target)
        for panel_rows in (_ranking_panel_rows(dual=False), _ranking_panel_rows(dual=True))
        for row in panel_rows
        for query, target in [(row["query_id"], target) for target in row["truth_targets"]]
    }
    with pytest.raises(prereg.PerformanceV2ContractError, match="coverage|duplicate|finite"):
        evaluator._rank_evidence(fixture["rank"], expected_pairs, 40)


def test_evidence_manifest_tamper_and_duplicate_headers_fail_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    manifest = json.loads(fixture["manifest"].read_text(encoding="utf-8"))
    manifest["target_universe_count"] = 3
    fixture["manifest"].write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    with pytest.raises(prereg.PerformanceV2ContractError, match="binding"):
        evaluator.evaluate(
            rank_evidence_csv=fixture["rank"],
            calibration_evidence_csv=fixture["calibration"],
            evidence_manifest_path=fixture["manifest"],
            prereg_json=fixture["prereg"],
            resources_path=fixture["resources"],
        )

    fixture = _fixture(tmp_path / "duplicate")
    lines = fixture["rank"].read_text(encoding="utf-8").splitlines()
    lines[0] += ",rank"
    lines[1:] = [line + ",999" for line in lines[1:]]
    fixture["rank"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(prereg.PerformanceV2ContractError, match="columns"):
        list(
            evaluator._read_exact_csv(
                fixture["rank"], evaluator.RANK_EVIDENCE_COLUMNS, "rank evidence"
            )
        )


def test_stale_outputs_are_preserved_and_overwrite_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    out = tmp_path / "results.csv"
    denominators = tmp_path / "denominators.json"
    out.write_text("stale\n", encoding="utf-8")
    denominators.write_text("stale\n", encoding="utf-8")
    fixture["rank"].write_text("broken\n", encoding="utf-8")
    with pytest.raises(prereg.PerformanceV2ContractError, match="refusing to overwrite"):
        evaluator.main(
            [
                "--rank-evidence-csv",
                str(fixture["rank"]),
                "--calibration-evidence-csv",
                str(fixture["calibration"]),
                "--evidence-manifest",
                str(fixture["manifest"]),
                "--prereg-json",
                str(fixture["prereg"]),
                "--resources",
                str(fixture["resources"]),
                "--out-csv",
                str(out),
                "--out-denominators-json",
                str(denominators),
            ]
        )
    assert out.read_text(encoding="utf-8") == "stale\n"
    assert denominators.read_text(encoding="utf-8") == "stale\n"
