#!/usr/bin/env python3
"""Build a preregistration-safe 2024 dev scaffold/target-cold ranking panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from build_activity_recovery_panels import (  # noqa: E402
    BENCHMARK_SCHEMA_VERSION,
    DEV_START,
    DUAL_COLD_FLAGS,
    REQUIRED_ACTIVITY_COLUMNS,
    _activity_dual_provenance,
    _add_standard_ligands,
    _calibration_pairs,
    _parse_iso_date,
    _ranking_columns,
    _require_schema,
    _stable_hash,
    _validate_activity_rows,
)
from build_activity_benchmark import OUTPUT_COLUMNS as BENCHMARK_OUTPUT_COLUMNS  # noqa: E402


SCHEMA_VERSION = "skinscout.sequence-inductive-dev-cold-panel.v1"
OUTPUT_NAMES = ("ranking.parquet", "calibration.parquet", "manifest.json")
COLD_CRITERIA = {
    "absent_pair_from_train": "standardized ligand_key/uniprot pair absent from train.parquet",
    "absent_publication_from_train": "publication_key absent from train.parquet",
    "absent_scaffold_from_train": "scaffold_id absent from train.parquet",
    "absent_target_cluster_30_from_train": "target_cluster_30 absent from train.parquet",
    "absent_target_cluster_50_from_train": "target_cluster_50 absent from train.parquet",
}
REQUIRED_COLUMNS = REQUIRED_ACTIVITY_COLUMNS | {
    "target_cluster_30",
    "target_cluster_50",
    "scaffold_id",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"{label} is missing or empty: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Unable to parse {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} must be a JSON object: {path}")
    return payload


def _parquet_schema_and_rows(path: Path) -> tuple[list[str], dict[str, str], int]:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"Parquet input is missing or empty: {path}")
    try:
        parquet = pq.ParquetFile(path)
    except Exception as exc:
        raise SystemExit(f"Unable to inspect parquet input: {path}") from exc
    schema = parquet.schema_arrow
    return (
        list(schema.names),
        {field.name: str(field.type) for field in schema},
        int(parquet.metadata.num_rows),
    )


def _manifest_output_sha(manifest: dict[str, Any], name: str) -> str:
    hashes = manifest.get("output_sha256")
    if not isinstance(hashes, dict):
        raise SystemExit("Benchmark manifest missing output_sha256 object")
    value = hashes.get(name)
    if not isinstance(value, str) or len(value) != 64:
        raise SystemExit(f"Benchmark manifest missing output_sha256[{name!r}]")
    return value


def _manifest_output_rows(manifest: dict[str, Any], split: str, name: str) -> int:
    splits = manifest.get("splits")
    counts = splits.get("counts") if isinstance(splits, dict) else None
    if not isinstance(counts, dict) or split not in counts:
        raise SystemExit(f"Benchmark manifest missing splits.counts.{split} for {name}")
    try:
        rows = int(counts[split])
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Benchmark manifest has invalid row count for {name}") from exc
    if rows < 0:
        raise SystemExit(f"Benchmark manifest has negative row count for {name}")
    return rows


def _require_columns(columns: Iterable[str], path: Path) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(columns))
    if missing:
        raise SystemExit(f"{path} missing required column(s): {', '.join(missing)}")
    if list(columns) != list(BENCHMARK_OUTPUT_COLUMNS):
        raise SystemExit(f"{path} schema does not match canonical activity benchmark schema")


def _validate_benchmark_parquet(
    path: Path,
    manifest: dict[str, Any],
    *,
    split: str,
    name: str,
) -> dict[str, Any]:
    columns, schema, rows = _parquet_schema_and_rows(path)
    _require_columns(columns, path)
    expected_rows = _manifest_output_rows(manifest, split, name)
    if rows != expected_rows:
        raise SystemExit(
            f"{name} row count does not match benchmark manifest: {rows} != {expected_rows}"
        )
    sha = _sha256(path)
    expected_sha = _manifest_output_sha(manifest, name)
    if sha != expected_sha:
        raise SystemExit(
            f"{name} sha256 does not match benchmark manifest: {sha} != {expected_sha}"
        )
    return {
        "path": str(path.resolve()),
        "sha256": sha,
        "rows": rows,
        "columns": columns,
        "schema": schema,
    }


def _load_activity(path: Path, split: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if split == "train":
        _validate_train_rows(frame, path)
        return frame
    else:
        _validate_activity_rows(frame, path, split)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise SystemExit(f"{path} missing required column(s): {', '.join(missing)}")
    return _add_standard_ligands(frame, path)


def _validate_train_rows(frame: pd.DataFrame, path: Path) -> None:
    missing = sorted(REQUIRED_ACTIVITY_COLUMNS - set(frame.columns))
    if missing:
        raise SystemExit(f"{path} missing required column(s): {', '.join(missing)}")
    if not frame["split"].astype(str).str.strip().eq("train").all():
        raise SystemExit(f"{path} contains rows outside required split=train")
    parsed_dates = frame["evidence_date"].map(
        lambda value: _parse_iso_date(value, "evidence_date")
    )
    if bool(parsed_dates.map(lambda value: value >= DEV_START).any()):
        raise SystemExit(f"{path} contains train evidence_date on or after 2024-01-01")
    pactivity = pd.to_numeric(frame["pactivity"], errors="coerce")
    if pactivity.map(lambda value: bool(pd.isna(value) or not math.isfinite(float(value)))).any():
        raise SystemExit(f"{path} contains non-finite pactivity")
    frame["pactivity"] = pactivity.astype(float)
    for column in [
        "benchmark_id",
        "source_db",
        "evidence_id",
        "publication_key",
        "uniprot",
        "ligand_smiles",
        "scaffold_id",
        "target_cluster_30",
        "target_cluster_50",
    ]:
        if frame[column].map(lambda value: pd.isna(value) or str(value).strip() == "").any():
            raise SystemExit(f"{path} contains blank {column}")


def _clean_set(values: pd.Series) -> set[str]:
    return {str(value).strip() for value in values if str(value).strip()}


def _train_absence_sets(train: pd.DataFrame) -> dict[str, set[Any]]:
    return {
        # An unseen target_cluster_30 necessarily implies the exact target, and
        # therefore every ligand/target pair for it, is absent from training.
        "target": _clean_set(train["uniprot"]),
        "publication": _clean_set(train["publication_key"]),
        "scaffold": _clean_set(train["scaffold_id"]),
        "target_cluster_30": _clean_set(train["target_cluster_30"]),
        "target_cluster_50": _clean_set(train["target_cluster_50"]),
    }


def _eligible_cold_rows(dev: pd.DataFrame, train_sets: dict[str, set[Any]]) -> pd.DataFrame:
    rows = dev[dev["identity_exclusion_reason"].eq("")].copy()
    rows["absent_pair_from_train"] = [
        str(value).strip() not in train_sets["target"] for value in rows["uniprot"]
    ]
    rows["absent_publication_from_train"] = [
        str(value).strip() not in train_sets["publication"] for value in rows["publication_key"]
    ]
    rows["absent_scaffold_from_train"] = [
        str(value).strip() not in train_sets["scaffold"] for value in rows["scaffold_id"]
    ]
    rows["absent_target_cluster_30_from_train"] = [
        str(value).strip() not in train_sets["target_cluster_30"]
        for value in rows["target_cluster_30"]
    ]
    rows["absent_target_cluster_50_from_train"] = [
        str(value).strip() not in train_sets["target_cluster_50"]
        for value in rows["target_cluster_50"]
    ]
    mask = (
        rows["absent_pair_from_train"]
        & rows["absent_publication_from_train"]
        & rows["absent_scaffold_from_train"]
        & rows["absent_target_cluster_30_from_train"]
        & rows["absent_target_cluster_50_from_train"]
    )
    return rows[mask].copy()


def _select_ranking(
    cold_rows: pd.DataFrame,
    *,
    cap: int | None,
    positive_threshold: float,
    seed: str,
) -> pd.DataFrame:
    positive = cold_rows[cold_rows["pactivity"] >= positive_threshold]
    candidates = (
        positive[
            [
                "ligand_key",
                "standard_inchikey",
                "canonical_smiles",
                "standardization_route",
            ]
        ]
        .drop_duplicates()
        .sort_values(["ligand_key", "canonical_smiles"])
        .reset_index(drop=True)
    )
    candidates["selection_hash"] = [
        _stable_hash(SCHEMA_VERSION, seed, row.ligand_key, row.canonical_smiles)
        for row in candidates.itertuples(index=False)
    ]
    ordered = candidates.sort_values(
        ["selection_hash", "ligand_key", "canonical_smiles"], kind="mergesort"
    )
    selected = ordered if cap is None else ordered.head(cap)
    selected_keys = set(selected["ligand_key"])

    truth_rows = positive[positive["ligand_key"].isin(selected_keys)]
    truth_by_key = (
        truth_rows.groupby("ligand_key", sort=True)["uniprot"]
        .agg(lambda values: sorted({str(value).strip() for value in values}))
        .to_dict()
    )
    records: list[dict[str, object]] = []
    for row in selected.itertuples(index=False):
        targets = truth_by_key[str(row.ligand_key)]
        record: dict[str, object] = {
            "query_id": str(row.ligand_key),
            "ligand_key": str(row.ligand_key),
            "standard_inchikey": str(row.standard_inchikey),
            "connectivity_key": str(row.standard_inchikey)[:14],
            "canonical_smiles": str(row.canonical_smiles),
            "standardization_route": str(row.standardization_route),
            "truth_targets": targets,
            "n_truth_targets": len(targets),
            "split": "dev",
        }
        for flag in sorted(DUAL_COLD_FLAGS):
            record[flag] = True
        records.append(record)
    return pd.DataFrame(records, columns=_ranking_columns())


def _diversity(ranking: pd.DataFrame, selected_rows: pd.DataFrame) -> dict[str, Any]:
    target_counts: Counter[str] = Counter()
    for targets in ranking["truth_targets"]:
        target_counts.update(str(target).strip() for target in targets)
    return {
        "queries": int(len(ranking)),
        "truth_pairs": int(sum(target_counts.values())),
        "unique_truth_targets": int(len(target_counts)),
        "unique_scaffolds": int(selected_rows["scaffold_id"].nunique()),
        "unique_target_cluster_30": int(selected_rows["target_cluster_30"].nunique()),
        "unique_target_cluster_50": int(selected_rows["target_cluster_50"].nunique()),
        "truth_pairs_by_target": dict(sorted(target_counts.items())),
    }


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _remove_outputs(*paths: Path) -> None:
    for path in paths:
        path.unlink(missing_ok=True)
        path.with_suffix(path.suffix + ".tmp").unlink(missing_ok=True)


def _assert_safe_output_files(
    *, input_paths: Iterable[Path], output_paths: Iterable[Path]
) -> None:
    protected = {path.resolve() for path in input_paths}
    outputs = tuple(path.resolve() for path in output_paths)
    destructive = outputs + tuple(
        path.with_suffix(path.suffix + ".tmp").resolve() for path in outputs
    )
    if len(set(destructive)) != len(destructive):
        raise SystemExit("sequence dev panel output paths must be distinct")
    aliases = sorted(str(path) for path in set(destructive) & protected)
    if aliases:
        raise SystemExit(
            "sequence dev panel output path aliases a protected input: "
            + ", ".join(aliases)
        )


def build_panel(args: argparse.Namespace) -> None:
    _assert_safe_output_files(
        input_paths=(
            args.train_parquet,
            args.dev_parquet,
            args.benchmark_manifest,
        ),
        output_paths=(
            args.output_parquet,
            args.output_calibration,
            args.output_manifest,
        ),
    )
    try:
        _remove_outputs(
            args.output_parquet,
            args.output_calibration,
            args.output_manifest,
        )
        if args.cap is not None and args.cap < 1:
            raise SystemExit("--cap must be >= 1")
        if not math.isfinite(args.positive_threshold):
            raise SystemExit("--positive-threshold must be finite")
        if not math.isfinite(args.negative_threshold) or (
            args.negative_threshold >= args.positive_threshold
        ):
            raise SystemExit(
                "--negative-threshold must be finite and below --positive-threshold"
            )

        benchmark_manifest = _read_json(args.benchmark_manifest, "Benchmark manifest")
        _require_schema(benchmark_manifest, BENCHMARK_SCHEMA_VERSION, "Benchmark manifest")
        manifest_meta = {
            "path": str(args.benchmark_manifest.resolve()),
            "sha256": _sha256(args.benchmark_manifest),
            "schema_version": benchmark_manifest["schema_version"],
        }
        train_meta = _validate_benchmark_parquet(
            args.train_parquet, benchmark_manifest, split="train", name="train.parquet"
        )
        dev_meta = _validate_benchmark_parquet(
            args.dev_parquet, benchmark_manifest, split="dev", name="dev.parquet"
        )

        train = _load_activity(args.train_parquet, "train")
        dev = _load_activity(args.dev_parquet, "dev")
        if len(train) != train_meta["rows"]:
            raise SystemExit("Loaded train row count changed after validation")
        if len(dev) != dev_meta["rows"]:
            raise SystemExit("Loaded dev row count changed after validation")

        cold_rows = _eligible_cold_rows(dev, _train_absence_sets(train))
        ranking = _select_ranking(
            cold_rows,
            cap=args.cap,
            positive_threshold=args.positive_threshold,
            seed=args.selection_seed,
        )
        if ranking.empty:
            raise SystemExit("sequence-inductive dev-cold ranking panel is empty")
        ranking = _activity_dual_provenance(
            ranking,
            cold_rows,
            positive_threshold=args.positive_threshold,
        )
        calibration, calibration_selection = _calibration_pairs(
            cold_rows,
            panel_name="dev",
            cap=None,
            negative_threshold=args.negative_threshold,
            positive_threshold=args.positive_threshold,
            seed=f"{args.selection_seed}:sequence-cold-calibration",
        )
        if calibration.empty or set(calibration["label"].astype(int)) != {0, 1}:
            raise SystemExit(
                "sequence-inductive dev-cold calibration requires measured pairs from both classes"
            )

        selected_rows = cold_rows[cold_rows["ligand_key"].isin(set(ranking["ligand_key"]))]
        _write_parquet_atomic(ranking, args.output_parquet)
        _write_parquet_atomic(calibration, args.output_calibration)
        output_meta = {
            "ranking.parquet": {
                "path": str(args.output_parquet.resolve()),
                "sha256": _sha256(args.output_parquet),
                "rows": int(len(ranking)),
                "columns": list(ranking.columns),
            },
            "calibration.parquet": {
                "path": str(args.output_calibration.resolve()),
                "sha256": _sha256(args.output_calibration),
                "rows": int(len(calibration)),
                "columns": list(calibration.columns),
            },
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                "activity_benchmark_manifest": manifest_meta,
                "train.parquet": train_meta,
                "dev.parquet": dev_meta,
            },
            "outputs": output_meta,
            "selection": {
                "seed": args.selection_seed,
                "positive_threshold": args.positive_threshold,
                "cap": args.cap,
                "candidate_rows": int(len(cold_rows)),
                "candidate_positive_ligands": int(
                    cold_rows.loc[
                        cold_rows["pactivity"] >= args.positive_threshold, "ligand_key"
                    ].nunique()
                ),
                "selected_ligands": int(len(ranking)),
                "cold_criteria": COLD_CRITERIA,
                "diversity": _diversity(ranking, selected_rows),
                "calibration": {
                    "rows": int(len(calibration)),
                    "positive_rows": int(calibration["label"].eq(1).sum()),
                    "negative_rows": int(calibration["label"].eq(0).sum()),
                    "full_cold_population": True,
                    "negative_threshold": args.negative_threshold,
                    "positive_threshold": args.positive_threshold,
                    **calibration_selection,
                },
            },
            "contracts": {
                "split": "dev",
                "dev_only_model_selection": True,
                "no_test_input": True,
                "no_known_skin_input": True,
                "no_rcsb_input": True,
                "no_truth_assistance_artifact_input": True,
                "no_target_assistance_to_scorer": True,
                "truth_targets_collected_only_after_ligand_selection": True,
                "source_provenance_required": True,
                "cold_calibration_full_population": True,
                "cold_calibration_measured_labels_only": True,
                "positive_truth_definition": "measured pActivity >= positive_threshold in dev.parquet",
                "all_cold_flags_true": True,
                "all_claimable_true": True,
            },
        }
        _write_json_atomic(manifest, args.output_manifest)
    except BaseException:
        _remove_outputs(
            args.output_parquet,
            args.output_calibration,
            args.output_manifest,
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-parquet", type=Path, default=Path("data/activity_benchmark_202608/train.parquet"))
    parser.add_argument("--dev-parquet", type=Path, default=Path("data/activity_benchmark_202608/dev.parquet"))
    parser.add_argument(
        "--benchmark-manifest",
        type=Path,
        default=Path("data/activity_benchmark_202608/manifest.json"),
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=Path("data/activity_benchmark_202608/sequence_inductive_dev_cold_ranking.parquet"),
    )
    parser.add_argument(
        "--output-calibration",
        type=Path,
        default=Path(
            "data/activity_benchmark_202608/sequence_inductive_dev_cold_calibration.parquet"
        ),
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("data/activity_benchmark_202608/sequence_inductive_dev_cold_ranking.manifest.json"),
    )
    parser.add_argument(
        "--cap",
        type=int,
        default=None,
        help="optional deterministic diagnostic cap; model selection defaults to all eligible ligands",
    )
    parser.add_argument("--positive-threshold", type=float, default=6.0)
    parser.add_argument("--negative-threshold", type=float, default=5.0)
    parser.add_argument("--selection-seed", default=SCHEMA_VERSION)
    build_panel(parser.parse_args())


if __name__ == "__main__":
    main()
