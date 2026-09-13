#!/usr/bin/env python3
"""Compare reproducible Daina panel benchmark directories without significance claims."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MANIFEST_SCHEMA = "skinscout.daina-panel-benchmark.v1"
SUMMARY_SCHEMA = "skinscout.sota_known_target_recovery.v1"
SELECTION_SPLITS = ("development", "final-test", "retrospective-diagnostic")
METRICS = (
    "case_top10",
    "target_top10",
    "target_top30",
    "target_pair_mrr",
    "mean_finite_rank",
    "median_finite_rank",
    "ranked_target_coverage",
    "case_coverage",
)
RANK_METRICS = {"mean_finite_rank", "median_finite_rank"}
COMPATIBILITY_KEYS = (
    ("manifest", "inputs.panel_csv_sha256"),
    ("manifest", "inputs.source_manifest_sha256"),
    ("manifest", "inputs.fingerprint_manifest_sha256"),
    ("manifest", "inputs.chembl_fp_sha256"),
    ("manifest", "inputs.human_activities_sha256"),
    ("manifest", "parameters.evidence_mode"),
    ("manifest", "parameters.evaluation_mode"),
    ("manifest", "parameters.evidence_snapshot_id"),
    ("manifest", "parameters.cutoff_date"),
    ("manifest", "parameters.exclude_reference_similarity"),
    ("manifest", "parameters.thresholds.min_case_top10"),
    ("manifest", "parameters.thresholds.min_target_top10"),
    ("manifest", "parameters.thresholds.min_target_top30"),
    ("manifest", "parameters.thresholds.allow_threshold_failure"),
    ("summary", "evaluation_mode"),
    ("summary", "evaluation_metadata.mode"),
    ("summary", "evaluation_metadata.evidence_snapshot"),
    ("summary", "evaluation_metadata.cutoff_date"),
    ("summary", "thresholds.min_case_top10"),
    ("summary", "thresholds.min_target_top10"),
    ("summary", "thresholds.min_target_top30"),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"{label} is required and must be non-empty: {path}")
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        raise SystemExit(f"Unable to read {label}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} must contain a JSON object: {path}")
    return payload


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        tmp.replace(path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def _write_csv_atomic(rows: list[dict[str, Any]], path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def _get(payload: dict[str, Any], dotted: str, label: str) -> Any:
    current: Any = payload
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            raise SystemExit(f"{label} missing required field: {dotted}")
        current = current[part]
    return current


def _resolve_run_path(run_dir: Path, value: str, label: str) -> Path:
    raw = Path(value)
    path = raw if raw.is_absolute() else run_dir / raw
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"{label} path is required and must be non-empty: {path}")
    return path


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise SystemExit(f"{label} must be numeric, not boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{label} must be numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise SystemExit(f"{label} must be finite: {value!r}")
    return number


def _metric_delta(metric: str, baseline: float, candidate: float) -> tuple[float, float]:
    delta = candidate - baseline
    improvement = -delta if metric in RANK_METRICS else delta
    return delta, improvement


def _metric_direction(metric: str) -> str:
    return "lower_is_better" if metric in RANK_METRICS else "higher_is_better"


def _validate_metric_name(metric: str, label: str) -> None:
    if metric not in METRICS:
        raise SystemExit(
            f"{label} must be one of {', '.join(METRICS)}; got {metric!r}"
        )


def _run_label(path: Path, seen: set[str]) -> str:
    label = path.resolve().name
    if label not in seen:
        seen.add(label)
        return label
    index = 2
    while f"{label}-{index}" in seen:
        index += 1
    label = f"{label}-{index}"
    seen.add(label)
    return label


def _load_run(run_dir: Path, label: str) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "manifest.json"
    summary_path = run_dir / "skin_known_summary.json"
    manifest = _read_json(manifest_path, f"{label} manifest")
    summary = _read_json(summary_path, f"{label} skin known summary")
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise SystemExit(
            f"{label} manifest schema mismatch; expected {MANIFEST_SCHEMA}: {manifest_path}"
        )
    if summary.get("schema_version") != SUMMARY_SCHEMA:
        raise SystemExit(
            f"{label} summary schema mismatch; expected {SUMMARY_SCHEMA}: {summary_path}"
        )

    recorded_summary = _get(manifest, "outputs.summary_path", f"{label} manifest")
    resolved_summary = _resolve_run_path(
        run_dir,
        str(recorded_summary),
        f"{label} manifest outputs.summary_path",
    )
    if resolved_summary.resolve() != summary_path.resolve():
        raise SystemExit(
            f"{label} manifest outputs.summary_path must point to skin_known_summary.json"
        )
    recorded_summary_sha = str(
        _get(manifest, "outputs.summary_sha256", f"{label} manifest")
    ).strip().lower()
    summary_hash = _sha256(summary_path)
    if recorded_summary_sha != summary_hash:
        raise SystemExit(
            f"{label} summary SHA-256 does not match manifest: "
            f"expected={recorded_summary_sha or '<missing>'} actual={summary_hash}"
        )

    params = _get(manifest, "parameters", f"{label} manifest")
    inputs = _get(manifest, "inputs", f"{label} manifest")
    if not isinstance(params, dict) or not isinstance(inputs, dict):
        raise SystemExit(f"{label} manifest parameters and inputs must be objects")
    for key in ("quality_policy", "scoring_method"):
        value = str(params.get(key, "")).strip()
        if not value:
            raise SystemExit(f"{label} manifest missing recipe identity: parameters.{key}")
    if summary.get("evaluation_mode") != _get(params, "evaluation_mode", f"{label} manifest"):
        raise SystemExit(f"{label} summary evaluation_mode does not match manifest")
    metadata = _get(summary, "evaluation_metadata", f"{label} summary")
    if not isinstance(metadata, dict):
        raise SystemExit(f"{label} summary evaluation_metadata must be an object")
    if metadata.get("mode") != params.get("evaluation_mode"):
        raise SystemExit(f"{label} summary evaluation_metadata.mode does not match manifest")
    if metadata.get("evidence_snapshot") != params.get("evidence_snapshot_id"):
        raise SystemExit(
            f"{label} summary evidence snapshot does not match manifest"
        )
    if metadata.get("cutoff_date") != params.get("cutoff_date"):
        raise SystemExit(f"{label} summary cutoff date does not match manifest")

    metrics_payload = _get(summary, "metrics", f"{label} summary")
    if not isinstance(metrics_payload, dict):
        raise SystemExit(f"{label} summary metrics must be an object")
    metrics = {
        metric: _finite_number(metrics_payload.get(metric), f"{label} metric {metric}")
        for metric in METRICS
    }
    return {
        "label": label,
        "run_dir": str(run_dir),
        "manifest_path": str(manifest_path),
        "summary_path": str(summary_path),
        "summary_sha256": summary_hash,
        "manifest": manifest,
        "summary": summary,
        "metrics": metrics,
        "recipe_identity": {
            "quality_policy": params["quality_policy"],
            "scoring_method": params["scoring_method"],
        },
    }


def _compatibility_value(run: dict[str, Any], source: str, dotted: str) -> Any:
    payload = run[source]
    return _get(payload, dotted, f"{run['label']} {source}")


def _assert_compatible(baseline: dict[str, Any], candidates: list[dict[str, Any]]) -> None:
    for candidate in candidates:
        for source, dotted in COMPATIBILITY_KEYS:
            expected = _compatibility_value(baseline, source, dotted)
            observed = _compatibility_value(candidate, source, dotted)
            if observed != expected:
                raise SystemExit(
                    "Benchmark compatibility check failed for "
                    f"{candidate['label']}: {source}.{dotted} differs from baseline "
                    f"(baseline={expected!r}, candidate={observed!r})"
                )


def _comparison_rows(
    baseline: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        row: dict[str, Any] = {
            "baseline_label": baseline["label"],
            "baseline_dir": baseline["run_dir"],
            "baseline_summary_sha256": baseline["summary_sha256"],
            "candidate_label": candidate["label"],
            "candidate_dir": candidate["run_dir"],
            "candidate_summary_sha256": candidate["summary_sha256"],
            "quality_policy": candidate["recipe_identity"]["quality_policy"],
            "scoring_method": candidate["recipe_identity"]["scoring_method"],
        }
        for metric in METRICS:
            baseline_value = baseline["metrics"][metric]
            candidate_value = candidate["metrics"][metric]
            delta, improvement = _metric_delta(metric, baseline_value, candidate_value)
            row[f"baseline_{metric}"] = baseline_value
            row[f"candidate_{metric}"] = candidate_value
            row[f"delta_{metric}"] = delta
            row[f"improvement_{metric}"] = improvement
        rows.append(row)
    return rows


def _select_best(
    rows: list[dict[str, Any]],
    primary_metric: str,
    tie_break_metrics: list[str],
) -> dict[str, Any]:
    sort_metrics = [primary_metric, *tie_break_metrics]

    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        values: list[Any] = []
        for metric in sort_metrics:
            value = row[f"candidate_{metric}"]
            values.append(value if metric in RANK_METRICS else -value)
        values.append(row["candidate_label"])
        values.append(row["candidate_dir"])
        return tuple(values)

    return min(rows, key=key)


def _csv_columns() -> list[str]:
    columns = [
        "baseline_label",
        "baseline_dir",
        "baseline_summary_sha256",
        "candidate_label",
        "candidate_dir",
        "candidate_summary_sha256",
        "quality_policy",
        "scoring_method",
    ]
    for metric in METRICS:
        columns.extend(
            [
                f"baseline_{metric}",
                f"candidate_{metric}",
                f"delta_{metric}",
                f"improvement_{metric}",
            ]
        )
    return columns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", action="append", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--selection-split", required=True, choices=SELECTION_SPLITS)
    parser.add_argument("--select-best", action="store_true")
    parser.add_argument("--primary-metric", default="")
    parser.add_argument("--tie-break-metric", action="append", default=[])
    args = parser.parse_args()

    if args.out_csv.resolve() == args.out_json.resolve():
        raise SystemExit("--out-csv and --out-json must be different paths")
    csv_stage = args.out_csv.with_name(f".{args.out_csv.name}.comparison-stage")
    json_stage = args.out_json.with_name(f".{args.out_json.name}.comparison-stage")
    for path in (args.out_csv, args.out_json, csv_stage, json_stage):
        path.unlink(missing_ok=True)

    if args.select_best and args.selection_split != "development":
        raise SystemExit("--select-best is only allowed with --selection-split=development")
    if args.select_best and not args.primary_metric:
        raise SystemExit("--select-best requires --primary-metric")
    if args.select_best and not args.tie_break_metric:
        raise SystemExit("--select-best requires at least one --tie-break-metric")
    if args.primary_metric:
        _validate_metric_name(args.primary_metric, "--primary-metric")
    for metric in args.tie_break_metric:
        _validate_metric_name(metric, "--tie-break-metric")
    if len(set(args.tie_break_metric)) != len(args.tie_break_metric):
        raise SystemExit("--tie-break-metric values must be unique")
    if args.primary_metric and args.primary_metric in args.tie_break_metric:
        raise SystemExit("--primary-metric must not also appear in --tie-break-metric")

    seen_labels: set[str] = set()
    baseline = _load_run(args.baseline, _run_label(args.baseline, seen_labels))
    candidates = [
        _load_run(path, _run_label(path, seen_labels))
        for path in args.candidate
    ]
    candidate_dirs = [candidate["run_dir"] for candidate in candidates]
    if baseline["run_dir"] in candidate_dirs:
        raise SystemExit("--baseline must not also be supplied as a --candidate")
    if len(set(candidate_dirs)) != len(candidate_dirs):
        raise SystemExit("--candidate directories must be unique")

    _assert_compatible(baseline, candidates)
    rows = _comparison_rows(baseline, candidates)
    selected = (
        _select_best(rows, args.primary_metric, args.tie_break_metric)
        if args.select_best
        else None
    )
    selection = {
        "selection_split": args.selection_split,
        "select_best": bool(args.select_best),
        "primary_metric": args.primary_metric,
        "tie_break_metrics": args.tie_break_metric,
        "selected_candidate_label": selected["candidate_label"] if selected else "",
        "selected_candidate_dir": selected["candidate_dir"] if selected else "",
    }
    payload = {
        "schema_version": "skinscout.daina-benchmark-comparison.v1",
        "created_at_utc": _utc_now(),
        "selection": selection,
        "metric_directions": {
            metric: _metric_direction(metric)
            for metric in METRICS
        },
        "baseline": {
            key: baseline[key]
            for key in (
                "label",
                "run_dir",
                "manifest_path",
                "summary_path",
                "summary_sha256",
                "recipe_identity",
                "metrics",
            )
        },
        "candidates": [
            {
                key: candidate[key]
                for key in (
                    "label",
                    "run_dir",
                    "manifest_path",
                    "summary_path",
                    "summary_sha256",
                    "recipe_identity",
                    "metrics",
                )
            }
            for candidate in candidates
        ],
        "comparisons": rows,
        "compatibility_checks": [
            f"{source}.{dotted}"
            for source, dotted in COMPATIBILITY_KEYS
        ],
        "significance_claim": "not_assessed",
    }
    try:
        _write_csv_atomic(rows, csv_stage, _csv_columns())
        _write_json_atomic(payload, json_stage)
        csv_stage.replace(args.out_csv)
        json_stage.replace(args.out_json)
    except BaseException:
        for path in (args.out_csv, args.out_json, csv_stage, json_stage):
            path.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
