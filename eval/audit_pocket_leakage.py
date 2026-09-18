#!/usr/bin/env python3
"""Audit pocket-cluster leakage across activity benchmark splits.

This is an independent sidecar audit. It validates registered provenance before
reading split semantics and never feeds ranking, training, or candidate
generation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow.parquet as pq


SCHEMA_VERSION = "skinscout.pocket-leakage-audit.v1"
BENCHMARK_SCHEMA_VERSION = "activity_benchmark.v1"
TARGET_CLUSTER_SCHEMA_VERSION = "skinscout.target-cluster-map.v2"
POCKET_CLUSTER_SCHEMA_VERSION = "skinscout.pocket-cluster-map.v1"

SPLIT_FILES = {
    "train": "train.parquet",
    "dev": "dev.parquet",
    "test": "test.parquet",
    "dual_cold": "dual_cold.parquet",
}
EVAL_SPLITS = ("dev", "test", "dual_cold")
THRESHOLDS = ("tm50", "tm40", "tm60")
REQUIRED_BENCHMARK_COLUMNS = {
    "benchmark_id",
    "evidence_id",
    "split",
    "uniprot",
    "target_cluster_30",
    "target_cluster_50",
}
REQUIRED_VIEW_COLUMNS = REQUIRED_BENCHMARK_COLUMNS | {"evaluation_view"}
REQUIRED_TARGET_COLUMNS = {"uniprot", "target_cluster_30", "target_cluster_50"}
REQUIRED_POCKET_COLUMNS = {
    "uniprot",
    "pocket_available",
    "exclusion_reason",
    "pocket_cluster_tm40",
    "pocket_cluster_tm50",
    "pocket_cluster_tm60",
}
ROW_FIELDS = [
    "audit_split",
    "row_index",
    "benchmark_id",
    "evidence_id",
    "source_split",
    "evaluation_view",
    "uniprot",
    "target_cluster_30",
    "target_cluster_50",
    "sequence_cluster_30_seen_in_train",
    "sequence_cluster_30_seen_in_prior_splits",
    "sequence_cold_30_from_train",
    "sequence_cold_30_from_prior_splits",
    "sequence_cluster_50_seen_in_train",
    "sequence_cluster_50_seen_in_prior_splits",
    "sequence_cold_50_from_train",
    "sequence_cold_50_from_prior_splits",
    "pocket_available",
    "pocket_unavailable_reason",
    "pocket_cluster_tm50",
    "pocket_tm50_seen_in_train",
    "pocket_tm50_seen_in_prior_splits",
    "pocket_cold_tm50_from_train",
    "pocket_cold_tm50_from_prior_splits",
    "pocket_cluster_tm40",
    "pocket_tm40_seen_in_train",
    "pocket_tm40_seen_in_prior_splits",
    "pocket_cold_tm40_from_train",
    "pocket_cold_tm40_from_prior_splits",
    "pocket_cluster_tm60",
    "pocket_tm60_seen_in_train",
    "pocket_tm60_seen_in_prior_splits",
    "pocket_cold_tm60_from_train",
    "pocket_cold_tm60_from_prior_splits",
]


def _clean(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.upper() in {"", "NA", "N/A", "NULL", "NAN", "NONE"} else text


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_file(path: Path, label: str) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"{label} is required and must be non-empty: {path}")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _require_file(path, label)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid {label} JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} must contain a JSON object: {path}")
    return payload


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    _require_file(path, label)
    try:
        df = pd.read_csv(path, dtype="string").fillna("")
    except Exception as exc:
        raise SystemExit(f"{label} failed to parse: {path}: {exc}") from exc
    if df.empty:
        raise SystemExit(f"{label} contains no rows: {path}")
    return df


def _tmp_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".tmp")


def _clean_outputs(*paths: Path) -> None:
    for path in paths:
        path.unlink(missing_ok=True)
        _tmp_path(path).unlink(missing_ok=True)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    tmp.unlink(missing_ok=True)
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_csv_atomic(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    tmp.unlink(missing_ok=True)
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def _require_columns(columns: Iterable[str], required: set[str], label: str) -> None:
    missing = sorted(required - set(columns))
    if missing:
        raise SystemExit(f"{label} missing required column(s): {', '.join(missing)}")


def _parquet_meta(path: Path, required: set[str], label: str) -> dict[str, Any]:
    _require_file(path, label)
    try:
        parquet = pq.ParquetFile(path)
    except Exception as exc:
        raise SystemExit(f"{label} failed to open as parquet: {path}: {exc}") from exc
    columns = list(parquet.schema_arrow.names)
    _require_columns(columns, required, label)
    return {"path": str(path.resolve()), "sha256": _sha256(path), "rows": int(parquet.metadata.num_rows), "columns": columns}


def _validate_artifact(
    path: Path,
    rows: int,
    artifact: dict[str, Any],
    label: str,
    *,
    require_path: bool = True,
) -> None:
    if str(artifact.get("sha256") or "").strip() != _sha256(path):
        raise SystemExit(f"{label} sha256 does not match manifest")
    if artifact.get("rows") != rows:
        raise SystemExit(f"{label} row count does not match manifest")
    artifact_path = _clean(artifact.get("path"))
    if require_path and (not artifact_path or Path(artifact_path).resolve() != path.resolve()):
        raise SystemExit(f"{label} path does not match manifest")


def _validate_benchmark_manifest(
    manifest_path: Path,
    split_meta: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    manifest = _read_json(manifest_path, "activity benchmark manifest")
    if manifest.get("schema_version") != BENCHMARK_SCHEMA_VERSION:
        raise SystemExit(f"activity benchmark manifest schema must be {BENCHMARK_SCHEMA_VERSION}")
    output_sha = manifest.get("output_sha256")
    if not isinstance(output_sha, dict):
        raise SystemExit("activity benchmark manifest missing output_sha256 object")
    for split, name in SPLIT_FILES.items():
        expected = _clean(output_sha.get(name))
        if expected != split_meta[split]["sha256"]:
            raise SystemExit(f"activity benchmark {name} sha256 does not match manifest")

    split_counts = manifest.get("splits", {}).get("counts") if isinstance(manifest.get("splits"), dict) else None
    if not isinstance(split_counts, dict):
        raise SystemExit("activity benchmark manifest missing splits.counts object")
    for split in ("train", "dev", "test"):
        if split_counts.get(split) != split_meta[split]["rows"]:
            raise SystemExit(f"activity benchmark {split}.parquet row count does not match manifest")
    view_counts = (
        manifest.get("evaluation_views", {}).get("counts")
        if isinstance(manifest.get("evaluation_views"), dict)
        else None
    )
    if not isinstance(view_counts, dict) or view_counts.get("dual_cold") != split_meta["dual_cold"]["rows"]:
        raise SystemExit("activity benchmark dual_cold.parquet row count does not match manifest")
    return manifest


def _validate_target_clusters(path: Path, manifest_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = _read_csv(path, "target cluster CSV")
    _require_columns(df.columns, REQUIRED_TARGET_COLUMNS, "target cluster CSV")
    targets = [_clean(value) for value in df["uniprot"]]
    if any(not value for value in targets):
        raise SystemExit("target cluster CSV contains blank uniprot values")
    if len(set(targets)) != len(targets):
        raise SystemExit("target cluster CSV contains duplicate uniprot values")
    for column in ("target_cluster_30", "target_cluster_50"):
        if any(not _clean(value) for value in df[column]):
            raise SystemExit(f"target cluster CSV contains blank {column} values")

    manifest = _read_json(manifest_path, "target cluster manifest")
    if manifest.get("schema_version") != TARGET_CLUSTER_SCHEMA_VERSION:
        raise SystemExit(f"target cluster manifest schema must be {TARGET_CLUSTER_SCHEMA_VERSION}")
    artifact = manifest.get("artifact") if isinstance(manifest.get("artifact"), dict) else None
    if not isinstance(artifact, dict):
        raise SystemExit("target cluster manifest missing artifact object")
    _validate_artifact(path, len(df), artifact, "target cluster CSV")
    return df, manifest


def _validate_pocket_clusters(path: Path, manifest_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = _read_csv(path, "pocket cluster CSV")
    _require_columns(df.columns, REQUIRED_POCKET_COLUMNS, "pocket cluster CSV")
    targets = [_clean(value) for value in df["uniprot"]]
    if any(not value for value in targets):
        raise SystemExit("pocket cluster CSV contains blank uniprot values")
    if len(set(targets)) != len(targets):
        raise SystemExit("pocket cluster CSV contains duplicate uniprot values")

    manifest = _read_json(manifest_path, "pocket cluster manifest")
    if manifest.get("schema_version") != POCKET_CLUSTER_SCHEMA_VERSION:
        raise SystemExit(f"pocket cluster manifest schema must be {POCKET_CLUSTER_SCHEMA_VERSION}")
    artifact = manifest.get("artifact") if isinstance(manifest.get("artifact"), dict) else None
    if not isinstance(artifact, dict):
        raise SystemExit("pocket cluster manifest missing artifact object")
    _validate_artifact(path, len(df), artifact, "pocket cluster CSV")

    for idx, row in df.iterrows():
        available = _clean(row["pocket_available"]).casefold()
        if available not in {"true", "false"}:
            raise SystemExit(f"pocket cluster CSV has invalid pocket_available at row {idx}")
        clusters = [_clean(row[f"pocket_cluster_{threshold}"]) for threshold in ("tm40", "tm50", "tm60")]
        reason = _clean(row["exclusion_reason"])
        if available == "true":
            if any(not value for value in clusters):
                raise SystemExit(f"available pocket row has blank cluster assignment at row {idx}")
            if reason:
                raise SystemExit(f"available pocket row has exclusion_reason at row {idx}")
        else:
            if any(clusters):
                raise SystemExit(f"unavailable pocket row has nonblank cluster assignment at row {idx}")
            if not reason:
                raise SystemExit(f"unavailable pocket row has blank exclusion_reason at row {idx}")
    return df, manifest


def _validate_universe_alignment(
    benchmark_manifest: dict[str, Any],
    target_csv: Path,
    target_manifest_path: Path,
    target_df: pd.DataFrame,
    pocket_csv: Path,
    pocket_manifest_path: Path,
    pocket_df: pd.DataFrame,
    pocket_manifest: dict[str, Any],
) -> None:
    required_input = (
        benchmark_manifest.get("target_cluster_policy", {}).get("required_input")
        if isinstance(benchmark_manifest.get("target_cluster_policy"), dict)
        else None
    )
    if not isinstance(required_input, dict):
        raise SystemExit("activity benchmark manifest missing target_cluster_policy.required_input")
    expected_target = {
        "path": str(target_csv.resolve()),
        "sha256": _sha256(target_csv),
        "rows": len(target_df),
        "manifest_path": str(target_manifest_path.resolve()),
        "manifest_sha256": _sha256(target_manifest_path),
    }
    for key, expected in expected_target.items():
        if required_input.get(key) != expected:
            raise SystemExit(f"activity benchmark target cluster provenance mismatch: {key}")

    inputs = pocket_manifest.get("inputs") if isinstance(pocket_manifest.get("inputs"), dict) else {}
    pocket_target = inputs.get("target_clusters") if isinstance(inputs.get("target_clusters"), dict) else None
    pocket_target_manifest = (
        inputs.get("target_cluster_manifest")
        if isinstance(inputs.get("target_cluster_manifest"), dict)
        else None
    )
    if not isinstance(pocket_target, dict) or not isinstance(pocket_target_manifest, dict):
        raise SystemExit("pocket cluster manifest missing target cluster input provenance")
    if pocket_target.get("path") != expected_target["path"] or pocket_target.get("sha256") != expected_target["sha256"] or pocket_target.get("rows") != expected_target["rows"]:
        raise SystemExit("pocket cluster manifest target CSV provenance mismatch")
    if pocket_target_manifest.get("path") != expected_target["manifest_path"] or pocket_target_manifest.get("sha256") != expected_target["manifest_sha256"]:
        raise SystemExit("pocket cluster manifest target manifest provenance mismatch")

    target_universe = set(_clean(value) for value in target_df["uniprot"])
    pocket_universe = set(_clean(value) for value in pocket_df["uniprot"])
    if pocket_universe != target_universe:
        missing = sorted(target_universe - pocket_universe)
        extra = sorted(pocket_universe - target_universe)
        raise SystemExit(
            "target and pocket cluster universes do not align exactly "
            f"(missing={len(missing)} extra={len(extra)})"
        )


def _read_split(path: Path, split_name: str) -> pd.DataFrame:
    columns = REQUIRED_VIEW_COLUMNS if split_name == "dual_cold" else REQUIRED_BENCHMARK_COLUMNS
    meta = _parquet_meta(path, columns, f"{SPLIT_FILES[split_name]}")
    del meta
    df = pd.read_parquet(path, columns=sorted(columns)).fillna("")
    if split_name in {"train", "dev", "test"}:
        bad = sorted(set(_clean(value) for value in df["split"]) - {split_name})
        if bad:
            raise SystemExit(f"{SPLIT_FILES[split_name]} contains unexpected split values: {', '.join(bad)}")
    else:
        if sorted(set(_clean(value) for value in df["split"]) - {"test"}):
            raise SystemExit("dual_cold.parquet must contain only source split 'test'")
        bad_views = sorted(set(_clean(value) for value in df["evaluation_view"]) - {"dual_cold"})
        if bad_views:
            raise SystemExit(f"dual_cold.parquet contains unexpected evaluation_view values: {', '.join(bad_views)}")
    if any(not _clean(value) for value in df["uniprot"]):
        raise SystemExit(f"{SPLIT_FILES[split_name]} contains blank uniprot values")
    return df.reset_index(drop=True)


def _rate(count: int, denominator: int) -> float | None:
    return None if denominator == 0 else count / denominator


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _summarize_rows(rows: list[dict[str, object]]) -> dict[str, Any]:
    total = len(rows)
    unavailable = sum(1 for row in rows if row["pocket_available"] == "false")
    available = total - unavailable
    summary: dict[str, Any] = {
        "rows": total,
        "pocket_available_rows": available,
        "pocket_unavailable_rows": unavailable,
        "pocket_unavailable_rate": _rate(unavailable, total),
        "sequence_cold": {},
        "pocket_cluster_leakage": {},
    }
    for threshold in ("30", "50"):
        for prior_label in ("train", "prior_splits"):
            key = f"sequence_cold_{threshold}_from_{prior_label}"
            count = sum(1 for row in rows if row[key] == "true")
            summary["sequence_cold"][f"target_cluster_{threshold}_from_{prior_label}"] = {
                "count": count,
                "rate": _rate(count, total),
            }
    for threshold in THRESHOLDS:
        threshold_summary: dict[str, Any] = {}
        for prior_label in ("train", "prior_splits"):
            seen_key = f"pocket_{threshold}_seen_in_{prior_label}"
            cold_key = f"pocket_cold_{threshold}_from_{prior_label}"
            seen_count = sum(1 for row in rows if row[seen_key] == "true")
            cold_count = sum(1 for row in rows if row[cold_key] == "true")
            threshold_summary[prior_label] = {
                "seen_count": seen_count,
                "seen_rate_all_rows": _rate(seen_count, total),
                "seen_rate_available_pockets": _rate(seen_count, available),
                "pocket_cold_count": cold_count,
                "pocket_cold_rate_available_pockets": _rate(cold_count, available),
                "unavailable_pocket_count": unavailable,
            }
        summary["pocket_cluster_leakage"][threshold] = threshold_summary
    return summary


def _audit(
    frames: dict[str, pd.DataFrame],
    pocket_df: pd.DataFrame,
) -> tuple[list[dict[str, object]], dict[str, Any]]:
    pocket_lookup = {
        _clean(row["uniprot"]): row
        for _, row in pocket_df.iterrows()
    }
    target_sets = {
        split: {
            column: set(_clean(value) for value in frame[column] if _clean(value))
            for column in ("target_cluster_30", "target_cluster_50")
        }
        for split, frame in frames.items()
    }
    pocket_sets: dict[str, dict[str, set[str]]] = {}
    for split, frame in frames.items():
        split_sets: dict[str, set[str]] = {threshold: set() for threshold in THRESHOLDS}
        for uniprot in frame["uniprot"]:
            pocket = pocket_lookup[_clean(uniprot)]
            if _clean(pocket["pocket_available"]).casefold() != "true":
                continue
            for threshold in THRESHOLDS:
                split_sets[threshold].add(_clean(pocket[f"pocket_cluster_{threshold}"]))
        pocket_sets[split] = split_sets

    row_rows: list[dict[str, object]] = []
    for audit_split in EVAL_SPLITS:
        frame = frames[audit_split]
        prior_split_names = ("train",) if audit_split == "dev" else ("train", "dev")
        prior_targets = {
            column: set().union(*(target_sets[name][column] for name in prior_split_names))
            for column in ("target_cluster_30", "target_cluster_50")
        }
        prior_pockets = {
            threshold: set().union(*(pocket_sets[name][threshold] for name in prior_split_names))
            for threshold in THRESHOLDS
        }
        for row_index, row in frame.iterrows():
            uniprot = _clean(row["uniprot"])
            pocket = pocket_lookup[uniprot]
            available = _clean(pocket["pocket_available"]).casefold() == "true"
            out: dict[str, object] = {
                "audit_split": audit_split,
                "row_index": int(row_index),
                "benchmark_id": _clean(row.get("benchmark_id")),
                "evidence_id": _clean(row.get("evidence_id")),
                "source_split": _clean(row.get("split")),
                "evaluation_view": _clean(row.get("evaluation_view")),
                "uniprot": uniprot,
                "target_cluster_30": _clean(row["target_cluster_30"]),
                "target_cluster_50": _clean(row["target_cluster_50"]),
                "pocket_available": _bool_text(available),
                "pocket_unavailable_reason": "" if available else _clean(pocket["exclusion_reason"]),
            }
            for threshold in ("30", "50"):
                column = f"target_cluster_{threshold}"
                cluster = _clean(row[column])
                seen_train = cluster in target_sets["train"][column]
                seen_prior = cluster in prior_targets[column]
                out[f"sequence_cluster_{threshold}_seen_in_train"] = _bool_text(seen_train)
                out[f"sequence_cluster_{threshold}_seen_in_prior_splits"] = _bool_text(seen_prior)
                out[f"sequence_cold_{threshold}_from_train"] = _bool_text(not seen_train)
                out[f"sequence_cold_{threshold}_from_prior_splits"] = _bool_text(not seen_prior)
            for threshold in THRESHOLDS:
                cluster = _clean(pocket[f"pocket_cluster_{threshold}"]) if available else ""
                seen_train = available and cluster in pocket_sets["train"][threshold]
                seen_prior = available and cluster in prior_pockets[threshold]
                out[f"pocket_cluster_{threshold}"] = cluster
                out[f"pocket_{threshold}_seen_in_train"] = _bool_text(seen_train)
                out[f"pocket_{threshold}_seen_in_prior_splits"] = _bool_text(seen_prior)
                out[f"pocket_cold_{threshold}_from_train"] = _bool_text(available and not seen_train)
                out[f"pocket_cold_{threshold}_from_prior_splits"] = _bool_text(available and not seen_prior)
            row_rows.append(out)

    by_split = {
        split: _summarize_rows([row for row in row_rows if row["audit_split"] == split])
        for split in EVAL_SPLITS
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "audit_scope": {
            "primary_pocket_threshold": "tm50",
            "sensitivity_pocket_thresholds": ["tm40", "tm60"],
            "sequence_cold_definition": "MMseqs target_cluster_30/50 absent from train or prior temporal splits.",
            "pocket_cold_definition": "Foldseek pocket_cluster_tm40/tm50/tm60 absent from train or prior temporal splits among rows with available pockets.",
            "unavailable_pocket_policy": "Unavailable pockets are counted separately and excluded from pocket-cold denominators.",
        },
        "by_split": by_split,
    }
    return row_rows, summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    _clean_outputs(args.out_json, args.out_csv)
    try:
        split_paths = {
            "train": args.train_parquet,
            "dev": args.dev_parquet,
            "test": args.test_parquet,
            "dual_cold": args.dual_cold_parquet,
        }
        split_meta = {
            split: _parquet_meta(
                path,
                REQUIRED_VIEW_COLUMNS if split == "dual_cold" else REQUIRED_BENCHMARK_COLUMNS,
                f"{SPLIT_FILES[split]}",
            )
            for split, path in split_paths.items()
        }
        benchmark_manifest = _validate_benchmark_manifest(args.benchmark_manifest, split_meta)
        target_df, target_manifest = _validate_target_clusters(args.target_clusters, args.target_cluster_manifest)
        pocket_df, pocket_manifest = _validate_pocket_clusters(args.pocket_clusters, args.pocket_cluster_manifest)
        _validate_universe_alignment(
            benchmark_manifest,
            args.target_clusters,
            args.target_cluster_manifest,
            target_df,
            args.pocket_clusters,
            args.pocket_cluster_manifest,
            pocket_df,
            pocket_manifest,
        )

        target_universe = set(_clean(value) for value in target_df["uniprot"])
        frames = {split: _read_split(path, split) for split, path in split_paths.items()}
        observed_targets = {
            split: set(_clean(value) for value in frame["uniprot"])
            for split, frame in frames.items()
        }
        outside = {
            split: sorted(values - target_universe)
            for split, values in observed_targets.items()
            if values - target_universe
        }
        if outside:
            split, values = next(iter(outside.items()))
            raise SystemExit(
                f"{SPLIT_FILES[split]} contains targets outside registered target universe: "
                f"{', '.join(values[:20])}"
            )

        row_rows, summary = _audit(frames, pocket_df)
        payload: dict[str, Any] = {
            **summary,
            "inputs": {
                "activity_benchmark_manifest": {
                    "path": str(args.benchmark_manifest.resolve()),
                    "sha256": _sha256(args.benchmark_manifest),
                    "schema_version": benchmark_manifest["schema_version"],
                },
                "activity_benchmark_parquets": {
                    split: {key: meta[key] for key in ("path", "sha256", "rows")}
                    for split, meta in split_meta.items()
                },
                "target_clusters": {
                    "path": str(args.target_clusters.resolve()),
                    "sha256": _sha256(args.target_clusters),
                    "rows": len(target_df),
                    "manifest_path": str(args.target_cluster_manifest.resolve()),
                    "manifest_sha256": _sha256(args.target_cluster_manifest),
                    "manifest_schema_version": target_manifest["schema_version"],
                },
                "pocket_clusters": {
                    "path": str(args.pocket_clusters.resolve()),
                    "sha256": _sha256(args.pocket_clusters),
                    "rows": len(pocket_df),
                    "manifest_path": str(args.pocket_cluster_manifest.resolve()),
                    "manifest_sha256": _sha256(args.pocket_cluster_manifest),
                    "manifest_schema_version": pocket_manifest["schema_version"],
                },
            },
            "target_universe_alignment": {
                "benchmark_target_cluster_input_matches_supplied_target_map": True,
                "pocket_cluster_input_matches_supplied_target_map": True,
                "pocket_and_target_universes_exactly_equal": True,
                "registered_target_count": len(target_df),
                "observed_activity_targets_by_split": {
                    split: len(values) for split, values in observed_targets.items()
                },
            },
            "outputs": {
                "row_csv": {"path": str(args.out_csv.resolve()), "rows": len(row_rows)},
                "summary_json": {"path": str(args.out_json.resolve())},
            },
        }
        _write_csv_atomic(args.out_csv, row_rows)
        payload["outputs"]["row_csv"]["sha256"] = _sha256(args.out_csv)
        _write_json_atomic(args.out_json, payload)
        print(
            "[pocket-leakage-audit] wrote "
            f"rows={len(row_rows)} json={args.out_json} csv={args.out_csv}"
        )
        return payload
    except BaseException:
        _clean_outputs(args.out_json, args.out_csv)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-manifest", required=True, type=Path)
    parser.add_argument("--train-parquet", required=True, type=Path)
    parser.add_argument("--dev-parquet", required=True, type=Path)
    parser.add_argument("--test-parquet", required=True, type=Path)
    parser.add_argument("--dual-cold-parquet", required=True, type=Path)
    parser.add_argument("--target-clusters", required=True, type=Path)
    parser.add_argument("--target-cluster-manifest", required=True, type=Path)
    parser.add_argument("--pocket-clusters", required=True, type=Path)
    parser.add_argument("--pocket-cluster-manifest", required=True, type=Path)
    parser.add_argument("--out-json", default=Path("results/audits/pocket_leakage_audit.json"), type=Path)
    parser.add_argument("--out-csv", default=Path("results/audits/pocket_leakage_audit_rows.csv"), type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        run(parse_args(argv))
    except SystemExit as exc:
        if isinstance(exc.code, str):
            print(exc.code, file=sys.stderr)
            return 1
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
