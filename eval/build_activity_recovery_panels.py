#!/usr/bin/env python3
"""Build frozen activity-recovery evaluation panels with bound provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Hashable, Iterable

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_activity_retrieval_index import (  # noqa: E402
    _standardize_mol,
    _structure_ligand_key,
)
from activity_recovery_contracts import validate_frozen_known_panel  # noqa: E402


BENCHMARK_SCHEMA_VERSION = "activity_benchmark.v1"
RETRIEVAL_SCHEMA_VERSION = "skinscout.activity-retrieval-index.v4"
RCSB_PANEL_SCHEMA_VERSION = "skinscout.rcsb-holo-direct-contact-panel.v1"
RCSB_SOURCE_SCHEMA_VERSION = "skinscout.rcsb-holo-contact-snapshot.v1"
RCSB_CC0_LICENSE_URL = "https://www.rcsb.org/pages/policies"
SCHEMA_VERSION = "skinscout.activity-recovery-panels.v5"
OUTPUT_NAMES = (
    "dev_ranking_queries.parquet",
    "test_ranking_queries.parquet",
    "dual_cold_ranking_queries.parquet",
    "dev_calibration_pairs.parquet",
    "test_calibration_pairs.parquet",
    "known_panel.csv",
    "manifest.json",
)
DATA_OUTPUT_NAMES = tuple(name for name in OUTPUT_NAMES if name != "manifest.json")
REQUIRED_ACTIVITY_COLUMNS = {
    "benchmark_id",
    "source_db",
    "evidence_id",
    "publication_key",
    "evidence_date",
    "split",
    "uniprot",
    "ligand_smiles",
    "ligand_inchikey",
    "endpoint",
    "pactivity",
}
DUAL_COLD_FLAGS = {
    "claimable",
    "absent_pair_from_train",
    "absent_pair_from_prior_splits",
    "absent_publication_from_train",
    "absent_publication_from_prior_splits",
    "absent_scaffold_from_train",
    "absent_scaffold_from_prior_splits",
    "absent_target_cluster_30_from_train",
    "absent_target_cluster_30_from_prior_splits",
    "absent_target_cluster_50_from_train",
    "absent_target_cluster_50_from_prior_splits",
}
ENDPOINT_FAMILIES = {
    "KI": "direct_binding",
    "KD": "direct_binding",
    "IC50": "functional",
    "EC50": "functional",
}
DEV_START = date(2024, 1, 1)
DEV_END = date(2024, 12, 31)
TEST_START = date(2025, 1, 1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_hash(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
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


def _require_schema(payload: dict[str, Any], expected: str, label: str) -> None:
    if payload.get("schema_version") != expected:
        raise SystemExit(f"{label} schema_version must be {expected}")


def _parquet_row_count(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"Parquet input is missing or empty: {path}")
    try:
        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception as exc:
        raise SystemExit(f"Unable to inspect parquet input: {path}") from exc


def _csv_row_count(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"CSV input is missing or empty: {path}")
    with path.open("rb") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def _manifest_output_sha(manifest: dict[str, Any], name: str) -> str:
    hashes = manifest.get("output_sha256")
    if not isinstance(hashes, dict):
        raise SystemExit("Benchmark manifest missing output_sha256 object")
    value = hashes.get(name)
    if not isinstance(value, str) or len(value) != 64:
        raise SystemExit(f"Benchmark manifest missing output_sha256[{name!r}]")
    return value


def _manifest_output_rows(manifest: dict[str, Any], split: str, name: str) -> int:
    if split == "dual_cold":
        views = manifest.get("evaluation_views")
        counts = views.get("counts") if isinstance(views, dict) else None
    else:
        splits = manifest.get("splits")
        counts = splits.get("counts") if isinstance(splits, dict) else None
    if not isinstance(counts, dict) or split not in counts:
        raise SystemExit(f"Benchmark manifest missing row count for {name}")
    try:
        rows = int(counts[split])
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Benchmark manifest has invalid row count for {name}") from exc
    if rows < 0:
        raise SystemExit(f"Benchmark manifest has negative row count for {name}")
    return rows


def _validate_benchmark_input(
    path: Path,
    manifest: dict[str, Any],
    *,
    split: str,
    name: str,
) -> tuple[str, int]:
    expected_sha = _manifest_output_sha(manifest, name)
    actual_sha = _sha256(path)
    if actual_sha != expected_sha:
        raise SystemExit(
            f"{name} sha256 does not match benchmark manifest: "
            f"{actual_sha} != {expected_sha}"
        )
    actual_rows = _parquet_row_count(path)
    expected_rows = _manifest_output_rows(manifest, split, name)
    if actual_rows != expected_rows:
        raise SystemExit(
            f"{name} row count does not match benchmark manifest: "
            f"{actual_rows} != {expected_rows}"
        )
    return actual_sha, actual_rows


def _resolve_manifest_path(raw_path: object, manifest_path: Path, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise SystemExit(f"{label} missing path")
    resolved = Path(raw_path)
    if not resolved.is_absolute():
        resolved = manifest_path.parent / resolved
    return resolved.resolve()


def _validate_retrieval_manifest(
    path: Path,
    manifest: dict[str, Any],
    *,
    benchmark_manifest_path: Path,
    benchmark_manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require_schema(manifest, RETRIEVAL_SCHEMA_VERSION, "Retrieval index manifest")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise SystemExit("Retrieval index manifest missing inputs object")
    benchmark_record = inputs.get("benchmark_manifest")
    if not isinstance(benchmark_record, dict):
        raise SystemExit("Retrieval index manifest missing inputs.benchmark_manifest")
    recorded_benchmark_path = _resolve_manifest_path(
        benchmark_record.get("path"), path, "Retrieval benchmark manifest"
    )
    expected_benchmark_path = benchmark_manifest_path.resolve()
    expected_benchmark_sha = _sha256(benchmark_manifest_path)
    if recorded_benchmark_path != expected_benchmark_path:
        raise SystemExit(
            "Retrieval index benchmark path differs from panel benchmark: "
            f"{recorded_benchmark_path} != {expected_benchmark_path}"
        )
    if benchmark_record.get("sha256") != expected_benchmark_sha:
        raise SystemExit("Retrieval index benchmark sha256 differs from panel benchmark")

    train_record = inputs.get("train_parquet")
    if not isinstance(train_record, dict):
        raise SystemExit("Retrieval index manifest missing inputs.train_parquet")
    recorded_train_path = _resolve_manifest_path(
        train_record.get("path"), path, "Retrieval train parquet"
    )
    expected_train_path = (benchmark_manifest_path.parent / "train.parquet").resolve()
    if recorded_train_path != expected_train_path:
        raise SystemExit(
            "Retrieval index train path differs from benchmark train.parquet: "
            f"{recorded_train_path} != {expected_train_path}"
        )
    expected_train_sha = _manifest_output_sha(benchmark_manifest, "train.parquet")
    expected_train_rows = _manifest_output_rows(
        benchmark_manifest, "train", "train.parquet"
    )
    if train_record.get("sha256") != expected_train_sha:
        raise SystemExit("Retrieval index train sha256 differs from benchmark train.parquet")
    try:
        recorded_train_rows = int(train_record.get("rows", -1))
    except (TypeError, ValueError) as exc:
        raise SystemExit("Retrieval index train row count is invalid") from exc
    if recorded_train_rows != expected_train_rows:
        raise SystemExit("Retrieval index train row count differs from benchmark train.parquet")
    actual_train_sha = _sha256(recorded_train_path)
    actual_train_rows = _parquet_row_count(recorded_train_path)
    if actual_train_sha != expected_train_sha or actual_train_rows != expected_train_rows:
        raise SystemExit("Retrieval index train artifact no longer matches benchmark manifest")

    validated_outputs: dict[str, Any] = {}
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        raise SystemExit("Retrieval index manifest missing outputs object")
    for key, meta in sorted(outputs.items()):
        if not isinstance(meta, dict):
            raise SystemExit(f"Retrieval index manifest output {key!r} must be an object")
        raw_path = meta.get("path")
        expected_sha = meta.get("sha256")
        expected_rows = meta.get("rows")
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise SystemExit(f"Retrieval index manifest output {key!r} missing sha256")
        try:
            rows = int(expected_rows)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"Retrieval index manifest output {key!r} has invalid rows") from exc
        output_path = _resolve_manifest_path(
            raw_path, path, f"Retrieval index manifest output {key!r}"
        )
        if _sha256(output_path) != expected_sha:
            raise SystemExit(f"Retrieval index output {key!r} sha256 does not match manifest")
        actual_rows = _parquet_row_count(output_path)
        if actual_rows != rows:
            raise SystemExit(f"Retrieval index output {key!r} row count does not match manifest")
        validated_outputs[key] = {
            "path": str(output_path.resolve()),
            "sha256": expected_sha,
            "rows": rows,
        }
    return validated_outputs, {
        "benchmark_manifest": {
            "path": str(expected_benchmark_path),
            "sha256": expected_benchmark_sha,
            "schema_version": benchmark_manifest["schema_version"],
        },
        "train.parquet": {
            "path": str(expected_train_path),
            "sha256": expected_train_sha,
            "rows": expected_train_rows,
        },
    }


def _parse_iso_date(value: object, column: str) -> date:
    text = "" if pd.isna(value) else str(value).strip()
    try:
        return date.fromisoformat(text[:10])
    except ValueError as exc:
        raise SystemExit(f"Invalid {column} value: {value!r}") from exc


def _require_columns(df: pd.DataFrame, path: Path, columns: Iterable[str]) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise SystemExit(f"{path} missing required column(s): {', '.join(missing)}")


def _validate_activity_rows(df: pd.DataFrame, path: Path, split: str) -> None:
    required = set(REQUIRED_ACTIVITY_COLUMNS)
    if split == "dual_cold":
        required |= DUAL_COLD_FLAGS | {"evaluation_view"}
    _require_columns(df, path, required)
    bad_split = df.index[df["split"].astype(str).str.strip() != ("test" if split == "dual_cold" else split)]
    if len(bad_split):
        raise SystemExit(f"{path} contains rows outside required split={split}: {bad_split[:10].tolist()}")
    parsed_dates = df["evidence_date"].map(lambda value: _parse_iso_date(value, "evidence_date"))
    if split == "dev":
        bad_dates = parsed_dates.map(lambda value: not (DEV_START <= value <= DEV_END))
        if bool(bad_dates.any()):
            raise SystemExit(f"{path} contains dev evidence_date outside 2024")
    else:
        bad_dates = parsed_dates.map(lambda value: value < TEST_START)
        if bool(bad_dates.any()):
            raise SystemExit(f"{path} contains test evidence_date before 2025-01-01")
    pactivity = pd.to_numeric(df["pactivity"], errors="coerce")
    bad_pactivity = pactivity.map(
        lambda value: bool(pd.isna(value) or not math.isfinite(float(value)))
    )
    if bool(bad_pactivity.any()):
        raise SystemExit(f"{path} contains non-finite pactivity")
    df["pactivity"] = pactivity.astype(float)
    endpoints = df["endpoint"].astype(str).str.strip().str.upper()
    unsupported = sorted(set(endpoints) - set(ENDPOINT_FAMILIES))
    if unsupported:
        raise SystemExit(f"{path} contains unsupported endpoint(s): {unsupported}")
    df["endpoint"] = endpoints
    for column in ["benchmark_id", "source_db", "evidence_id", "publication_key", "uniprot", "ligand_smiles"]:
        blank = df[column].map(lambda value: pd.isna(value) or str(value).strip() == "")
        if bool(blank.any()):
            raise SystemExit(f"{path} contains blank {column}")
    if split == "dual_cold":
        if not (df["evaluation_view"].astype(str).str.strip() == "dual_cold").all():
            raise SystemExit(f"{path} contains non-dual_cold evaluation_view rows")
        for column in sorted(DUAL_COLD_FLAGS):
            if not df[column].map(lambda value: bool(value) is True).all():
                raise SystemExit(f"{path} contains dual-cold rows with {column}=False")


def _standardize_ligand(raw_smiles: str, ligand_inchikey: object) -> tuple[str, str]:
    del ligand_inchikey
    parent = _standardize_mol(raw_smiles)
    canonical = Chem.MolToSmiles(parent, canonical=True, isomericSmiles=True)
    ligand_key = Chem.MolToInchiKey(parent)
    if not ligand_key:
        raise ValueError(f"unable to derive InChIKey for ligand_smiles: {raw_smiles!r}")
    return ligand_key, canonical


def _add_standard_ligands(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    ligand_cache: dict[tuple[str, str], tuple[str, str, str]] = {}
    standard_inchikeys: list[str] = []
    canonical_smiles: list[str] = []
    exclusion_reasons: list[str] = []
    for raw, inchikey in zip(df["ligand_smiles"], df["ligand_inchikey"], strict=True):
        key = (str(raw).strip(), "" if pd.isna(inchikey) else str(inchikey).strip())
        if key not in ligand_cache:
            try:
                standard_inchikey, canonical = _standardize_ligand(key[0], key[1])
                ligand_cache[key] = (standard_inchikey, canonical, "")
            except ValueError:
                ligand_cache[key] = (
                    "",
                    "",
                    "unable_to_standardize_fragment_parent_identity",
                )
        standard_inchikey, canonical, exclusion_reason = ligand_cache[key]
        standard_inchikeys.append(standard_inchikey)
        canonical_smiles.append(canonical)
        exclusion_reasons.append(exclusion_reason)
    out = df.copy()
    out["standard_inchikey"] = standard_inchikeys
    out["canonical_smiles"] = canonical_smiles
    out["identity_exclusion_reason"] = exclusion_reasons
    structure_keys = [
        _structure_ligand_key(str(standard_inchikey), str(canonical))
        if not exclusion_reason
        else ("", "")
        for standard_inchikey, canonical, exclusion_reason in zip(
            out["standard_inchikey"],
            out["canonical_smiles"],
            out["identity_exclusion_reason"],
            strict=True,
        )
    ]
    out["ligand_key"] = [structure_key[0] for structure_key in structure_keys]
    out["standardization_route"] = [structure_key[1] for structure_key in structure_keys]
    return out


def _load_activity(path: Path, split: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    _validate_activity_rows(df, path, split)
    return _add_standard_ligands(df, path)


def _as_string_list(value: object, label: str) -> list[str]:
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, list, tuple, set)):
        value = value.tolist()
    if not isinstance(value, (list, tuple, set)):
        raise SystemExit(f"{label} must be a list")
    cleaned = sorted({str(item).strip() for item in value if str(item).strip()})
    if not cleaned:
        raise SystemExit(f"{label} must contain at least one nonblank value")
    return cleaned


def _encode_source_documents(documents: Iterable[tuple[str, str]]) -> str:
    records = [
        {"source_db": source_db, "publication_key": publication_key}
        for source_db, publication_key in sorted(set(documents))
    ]
    return json.dumps(records, separators=(",", ":"), sort_keys=True)


def _decode_source_documents(value: object, label: str) -> set[tuple[str, str]]:
    try:
        payload = json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"{label} is not valid JSON") from exc
    if not isinstance(payload, list) or not payload:
        raise SystemExit(f"{label} must contain a nonempty JSON list")
    documents: set[tuple[str, str]] = set()
    for record in payload:
        if not isinstance(record, dict):
            raise SystemExit(f"{label} contains a non-object document")
        source_db = str(record.get("source_db") or "").strip().casefold()
        publication_key = str(record.get("publication_key") or "").strip()
        if not source_db or not publication_key:
            raise SystemExit(f"{label} contains a blank source document")
        documents.add((source_db, publication_key))
    return documents


def _encode_target_sources(mapping: dict[str, set[str]]) -> str:
    payload = {target: sorted(sources) for target, sources in sorted(mapping.items())}
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _decode_target_sources(value: object, label: str) -> dict[str, set[str]]:
    try:
        payload = json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"{label} is not valid JSON") from exc
    if not isinstance(payload, dict) or not payload:
        raise SystemExit(f"{label} must contain a nonempty JSON object")
    mapping: dict[str, set[str]] = {}
    for target, raw_sources in payload.items():
        target_id = str(target).strip()
        if not target_id:
            raise SystemExit(f"{label} contains a blank target")
        mapping[target_id] = set(_as_string_list(raw_sources, f"{label}[{target_id!r}]"))
    return mapping


def _ranking_queries(
    df: pd.DataFrame,
    *,
    panel_name: str,
    cap: int | None,
    positive_threshold: float,
    seed: str,
) -> pd.DataFrame:
    df = df[df["identity_exclusion_reason"].eq("")]
    positive_keys = set(df.loc[df["pactivity"] >= positive_threshold, "ligand_key"])
    candidates = (
        df.loc[
            df["ligand_key"].isin(positive_keys),
            [
                "ligand_key",
                "standard_inchikey",
                "canonical_smiles",
                "standardization_route",
            ],
        ]
        .drop_duplicates()
        .sort_values(["ligand_key", "canonical_smiles"])
        .reset_index(drop=True)
    )
    candidates["selection_hash"] = [
        _stable_hash(SCHEMA_VERSION, panel_name, seed, row.ligand_key, row.canonical_smiles)
        for row in candidates.itertuples(index=False)
    ]
    selected = candidates.sort_values(["selection_hash", "ligand_key", "canonical_smiles"])
    if cap is not None:
        selected = selected.head(cap)
    selected = selected.reset_index(drop=True)
    selected_keys = set(selected["ligand_key"])
    positives = df[
        df["ligand_key"].isin(selected_keys) & (df["pactivity"] >= positive_threshold)
    ]
    truth_by_key = (
        positives.groupby("ligand_key", sort=True)["uniprot"]
        .agg(lambda values: sorted({str(value).strip() for value in values}))
        .to_dict()
    )
    rows = []
    for row in selected.itertuples(index=False):
        truth_targets = truth_by_key[str(row.ligand_key)]
        record: dict[str, object] = {
            "query_id": str(row.ligand_key),
            "ligand_key": str(row.ligand_key),
            "standard_inchikey": str(row.standard_inchikey),
            "connectivity_key": str(row.standard_inchikey)[:14],
            "canonical_smiles": str(row.canonical_smiles),
            "standardization_route": str(row.standardization_route),
            "truth_targets": truth_targets,
            "n_truth_targets": len(truth_targets),
            "split": "test" if panel_name == "dual_cold" else panel_name,
        }
        for flag in sorted(DUAL_COLD_FLAGS):
            record[flag] = bool(panel_name == "dual_cold")
        rows.append(record)
    return pd.DataFrame(rows, columns=_ranking_columns())


def _ranking_columns() -> list[str]:
    return [
        "query_id",
        "ligand_key",
        "standard_inchikey",
        "connectivity_key",
        "canonical_smiles",
        "standardization_route",
        "truth_targets",
        "n_truth_targets",
        "split",
        *sorted(DUAL_COLD_FLAGS),
    ]


DUAL_SOURCE_COLUMNS = [
    "panel_sources",
    "source_databases",
    "source_documents_json",
    "truth_target_panel_sources_json",
]


def _activity_dual_provenance(
    ranking: pd.DataFrame,
    activity: pd.DataFrame,
    *,
    positive_threshold: float,
) -> pd.DataFrame:
    positive = activity[
        activity["identity_exclusion_reason"].eq("")
        & activity["ligand_key"].isin(set(ranking["ligand_key"]))
        & (activity["pactivity"] >= positive_threshold)
    ]
    metadata: dict[str, dict[str, object]] = {}
    for ligand_key, group in positive.groupby("ligand_key", sort=True):
        documents = {
            (str(row.source_db).strip().casefold(), str(row.publication_key).strip())
            for row in group[["source_db", "publication_key"]].itertuples(index=False)
        }
        if any(not source or not document for source, document in documents):
            raise SystemExit("activity dual-cold provenance contains a blank source document")
        target_sources: dict[str, set[str]] = {}
        for target, target_group in group.groupby("uniprot", sort=True):
            target_sources[str(target).strip()] = {
                "activity_quantitative"
                for _source in target_group["source_db"]
            }
        metadata[str(ligand_key)] = {
            "panel_sources": ["activity_quantitative"],
            "source_databases": sorted({source for source, _document in documents}),
            "source_documents_json": _encode_source_documents(documents),
            "truth_target_panel_sources_json": _encode_target_sources(target_sources),
        }
    out = ranking.copy()
    for column in DUAL_SOURCE_COLUMNS:
        out[column] = [metadata[str(key)][column] for key in out["ligand_key"]]
    return out


def _validate_rcsb_panel(
    ranking_path: Path,
    manifest_path: Path,
    *,
    benchmark_manifest_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = _read_json(manifest_path, "RCSB holo panel manifest")
    _require_schema(manifest, RCSB_PANEL_SCHEMA_VERSION, "RCSB holo panel manifest")
    contract = manifest.get("contract")
    required_contract = {
        "positive_only": True,
        "no_inferred_negatives": True,
        "no_affinities_or_calibration": True,
        "never_training_or_model_selection": True,
        "ranking_split": "test",
        "all_ranking_rows_dual_cold": True,
    }
    if not isinstance(contract, dict) or any(
        contract.get(key) != expected for key, expected in required_contract.items()
    ):
        raise SystemExit("RCSB holo panel usage contract is invalid")
    if not isinstance(contract.get("pocket_leakage_audited"), bool):
        raise SystemExit("RCSB holo panel must state whether pocket leakage was audited")

    inputs = manifest.get("inputs")
    benchmark_record = inputs.get("benchmark_manifest") if isinstance(inputs, dict) else None
    if not isinstance(benchmark_record, dict):
        raise SystemExit("RCSB holo panel manifest missing benchmark input binding")
    recorded_benchmark_path = _resolve_manifest_path(
        benchmark_record.get("path"), manifest_path, "RCSB benchmark manifest"
    )
    if recorded_benchmark_path != benchmark_manifest_path.resolve():
        raise SystemExit("RCSB holo panel benchmark manifest path differs from recovery panels")
    benchmark_sha = _sha256(benchmark_manifest_path)
    if benchmark_record.get("sha256") != benchmark_sha:
        raise SystemExit("RCSB holo panel benchmark manifest sha256 is stale")
    benchmark_manifest = _read_json(benchmark_manifest_path, "Benchmark manifest")

    source_record = inputs.get("source_manifest") if isinstance(inputs, dict) else None
    raw_record = inputs.get("raw_jsonl_gz") if isinstance(inputs, dict) else None
    if not isinstance(source_record, dict) or not isinstance(raw_record, dict):
        raise SystemExit("RCSB holo panel manifest missing source snapshot bindings")
    source_path = _resolve_manifest_path(
        source_record.get("path"), manifest_path, "RCSB source manifest"
    )
    if source_record.get("sha256") != _sha256(source_path):
        raise SystemExit("RCSB source manifest sha256 is stale")
    source_manifest = _read_json(source_path, "RCSB source manifest")
    _require_schema(
        source_manifest,
        RCSB_SOURCE_SCHEMA_VERSION,
        "RCSB source manifest",
    )
    source_license = source_manifest.get("source_license")
    source_usage = source_manifest.get("usage_contract")
    if (
        not isinstance(source_license, dict)
        or source_license.get("url") != RCSB_CC0_LICENSE_URL
        or not isinstance(source_usage, dict)
        or source_usage.get("positive_only_direct_contact_evaluation_source") is not True
        or source_usage.get("never_training_or_calibration") is not True
    ):
        raise SystemExit("RCSB source provenance or usage contract is invalid")
    query_hashes = source_manifest.get("query_sha256s")
    if not isinstance(query_hashes, dict) or any(
        not isinstance(query_hashes.get(key), str) or len(query_hashes[key]) != 64
        for key in ("search", "graphql")
    ):
        raise SystemExit("RCSB source query hashes are invalid")

    raw_path = _resolve_manifest_path(
        raw_record.get("path"), manifest_path, "RCSB raw snapshot"
    )
    raw_sha = _sha256(raw_path)
    source_raw = source_manifest.get("raw_jsonl_gz")
    if not isinstance(source_raw, dict):
        raise SystemExit("RCSB source manifest missing raw snapshot record")
    try:
        panel_raw_rows = int(raw_record.get("rows", -1))
        source_raw_rows = int(source_raw.get("rows", -2))
        source_raw_bytes = int(source_raw.get("bytes", -1))
    except (TypeError, ValueError) as exc:
        raise SystemExit("RCSB raw snapshot metadata is invalid") from exc
    if (
        raw_record.get("sha256") != raw_sha
        or _resolve_manifest_path(source_raw.get("path"), source_path, "RCSB source raw snapshot")
        != raw_path
        or source_raw.get("sha256") != raw_sha
        or panel_raw_rows != source_raw_rows
        or source_raw_bytes != raw_path.stat().st_size
    ):
        raise SystemExit("RCSB raw snapshot transitive provenance is stale")

    for split, name in (("train", "train.parquet"), ("dev", "dev.parquet")):
        record = inputs.get(f"{split}_parquet") if isinstance(inputs, dict) else None
        if not isinstance(record, dict):
            raise SystemExit(f"RCSB holo panel manifest missing {split} benchmark binding")
        expected_path = (benchmark_manifest_path.parent / name).resolve()
        try:
            recorded_rows = int(record.get("rows", -1))
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"RCSB holo panel {name} row count is invalid") from exc
        if (
            _resolve_manifest_path(record.get("path"), manifest_path, f"RCSB {name}")
            != expected_path
            or record.get("sha256") != _manifest_output_sha(
                benchmark_manifest, name
            )
            or recorded_rows != _manifest_output_rows(benchmark_manifest, split, name)
        ):
            raise SystemExit(f"RCSB holo panel {name} benchmark binding is stale")

    cluster_record = inputs.get("screenable_target_clusters") if isinstance(inputs, dict) else None
    if not isinstance(cluster_record, dict):
        raise SystemExit("RCSB holo panel manifest missing screenable target binding")
    cluster_path = _resolve_manifest_path(
        cluster_record.get("path"), manifest_path, "RCSB screenable target clusters"
    )
    cluster_manifest_path = _resolve_manifest_path(
        cluster_record.get("manifest_path"),
        manifest_path,
        "RCSB screenable target cluster manifest",
    )
    if (
        cluster_record.get("sha256") != _sha256(cluster_path)
        or cluster_record.get("manifest_sha256") != _sha256(cluster_manifest_path)
    ):
        raise SystemExit("RCSB screenable target provenance is stale")

    outputs = manifest.get("outputs")
    ranking_record = outputs.get("ranking_queries") if isinstance(outputs, dict) else None
    if not isinstance(ranking_record, dict):
        raise SystemExit("RCSB holo panel manifest missing ranking_queries output")
    recorded_ranking_path = _resolve_manifest_path(
        ranking_record.get("path"), manifest_path, "RCSB ranking queries"
    )
    if recorded_ranking_path != ranking_path.resolve():
        raise SystemExit("RCSB ranking query path differs from its panel manifest")
    ranking_sha = _sha256(ranking_path)
    if ranking_record.get("sha256") != ranking_sha:
        raise SystemExit("RCSB ranking query sha256 differs from its panel manifest")
    rows = _parquet_row_count(ranking_path)
    try:
        expected_rows = int(ranking_record.get("rows", -1))
    except (TypeError, ValueError) as exc:
        raise SystemExit("RCSB ranking query row count is invalid") from exc
    if rows != expected_rows:
        raise SystemExit("RCSB ranking query row count differs from its panel manifest")

    ranking = pd.read_parquet(ranking_path)
    _require_columns(
        ranking,
        ranking_path,
        set(_ranking_columns()) | {"source_publication_keys"},
    )
    if ranking.empty:
        raise SystemExit("RCSB ranking query panel is empty")
    if ranking["query_id"].duplicated().any():
        raise SystemExit("RCSB ranking query panel contains duplicate query_id")
    if not ranking["query_id"].astype(str).eq(ranking["ligand_key"].astype(str)).all():
        raise SystemExit("RCSB ranking query_id must equal ligand_key")
    if not ranking["split"].astype(str).eq("test").all():
        raise SystemExit("RCSB ranking query panel must contain only split=test")
    for column in sorted(DUAL_COLD_FLAGS):
        valid = ranking[column].map(
            lambda value: isinstance(value, (bool, np.bool_)) and bool(value)
        )
        if not bool(valid.all()):
            raise SystemExit(f"RCSB ranking query panel contains invalid {column}")
    for idx, row in ranking.iterrows():
        truth_targets = _as_string_list(row["truth_targets"], f"RCSB truth_targets row {idx}")
        if int(row["n_truth_targets"]) != len(truth_targets):
            raise SystemExit("RCSB ranking query n_truth_targets is stale")
        publications = _as_string_list(
            row["source_publication_keys"], f"RCSB source_publication_keys row {idx}"
        )
        for column in (
            "query_id",
            "ligand_key",
            "standard_inchikey",
            "canonical_smiles",
            "standardization_route",
        ):
            if not str(row[column]).strip():
                raise SystemExit(f"RCSB ranking query contains blank {column}")
        ranking.at[idx, "truth_targets"] = truth_targets
        ranking.at[idx, "source_publication_keys"] = publications

    ranking = ranking.copy()
    ranking["panel_sources"] = [["rcsb_holo_direct_contact"] for _ in range(len(ranking))]
    ranking["source_databases"] = [["rcsb_pdb"] for _ in range(len(ranking))]
    ranking["source_documents_json"] = [
        _encode_source_documents(("rcsb_pdb", key) for key in keys)
        for keys in ranking["source_publication_keys"]
    ]
    ranking["truth_target_panel_sources_json"] = [
        _encode_target_sources(
            {target: {"rcsb_holo_direct_contact"} for target in truth_targets}
        )
        for truth_targets in ranking["truth_targets"]
    ]
    return ranking[_ranking_columns() + DUAL_SOURCE_COLUMNS], {
        "path": str(ranking_path.resolve()),
        "sha256": ranking_sha,
        "rows": rows,
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": _sha256(manifest_path),
            "schema_version": manifest["schema_version"],
        },
        "contract": contract,
        "source": {
            "manifest_path": str(source_path),
            "manifest_sha256": _sha256(source_path),
            "raw_path": str(raw_path),
            "raw_sha256": raw_sha,
            "license_url": RCSB_CC0_LICENSE_URL,
            "query_sha256s": query_hashes,
        },
        "passes_standalone_adequacy": manifest.get("passes_adequacy") is True,
    }


def _combine_dual_rankings(
    activity_ranking: pd.DataFrame,
    rcsb_ranking: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    frames = [activity_ranking]
    if rcsb_ranking is not None:
        frames.append(rcsb_ranking)
    combined = pd.concat(frames, ignore_index=True)
    rows: list[dict[str, object]] = []
    overlap_queries = 0
    for ligand_key, group in combined.groupby("ligand_key", sort=True):
        identities: dict[str, str] = {}
        for column in (
            "standard_inchikey",
            "connectivity_key",
            "canonical_smiles",
            "standardization_route",
        ):
            values = sorted({str(value).strip() for value in group[column]})
            if len(values) != 1 or not values[0]:
                raise SystemExit(
                    f"dual-cold sources disagree on {column} for ligand_key={ligand_key}"
                )
            identities[column] = values[0]
        panel_sources = {
            source
            for values in group["panel_sources"]
            for source in _as_string_list(values, "panel_sources")
        }
        if len(panel_sources) > 1:
            overlap_queries += 1
        documents: set[tuple[str, str]] = set()
        target_sources: dict[str, set[str]] = {}
        truth_targets: set[str] = set()
        for idx, row in group.iterrows():
            row_targets = set(_as_string_list(row["truth_targets"], f"truth_targets row {idx}"))
            truth_targets.update(row_targets)
            documents.update(
                _decode_source_documents(row["source_documents_json"], "source_documents_json")
            )
            row_target_sources = _decode_target_sources(
                row["truth_target_panel_sources_json"],
                "truth_target_panel_sources_json",
            )
            if set(row_target_sources) != row_targets:
                raise SystemExit("truth-target source attribution does not match truth_targets")
            for target, sources in row_target_sources.items():
                target_sources.setdefault(target, set()).update(sources)
        record: dict[str, object] = {
            "query_id": str(ligand_key),
            "ligand_key": str(ligand_key),
            **identities,
            "truth_targets": sorted(truth_targets),
            "n_truth_targets": len(truth_targets),
            "split": "test",
            **{flag: True for flag in sorted(DUAL_COLD_FLAGS)},
            "panel_sources": sorted(panel_sources),
            "source_databases": sorted({source for source, _document in documents}),
            "source_documents_json": _encode_source_documents(documents),
            "truth_target_panel_sources_json": _encode_target_sources(target_sources),
        }
        rows.append(record)
    result = pd.DataFrame(rows, columns=_ranking_columns() + DUAL_SOURCE_COLUMNS)
    return result, {
        "activity_quantitative_queries": int(len(activity_ranking)),
        "rcsb_holo_direct_contact_queries": int(len(rcsb_ranking)) if rcsb_ranking is not None else 0,
        "overlap_queries": overlap_queries,
        "combined_queries": int(len(result)),
    }


def _dual_cold_adequacy(
    ranking: pd.DataFrame,
    *,
    min_queries: int,
    min_targets: int,
    min_documents: int,
    max_target_fraction: float,
    min_effective_targets: float,
) -> dict[str, Any]:
    target_counts: Counter[str] = Counter()
    for values in ranking["truth_targets"]:
        target_counts.update(str(value).strip() for value in values)
    truth_pairs = sum(target_counts.values())
    target_fractions = {
        target: count / truth_pairs for target, count in target_counts.items()
    } if truth_pairs else {}
    observed_max_fraction = max(target_fractions.values(), default=1.0)
    effective_targets = (
        1.0 / sum(fraction * fraction for fraction in target_fractions.values())
        if target_fractions
        else 0.0
    )

    documents: set[tuple[str, str]] = set()
    queries_by_panel_source: Counter[str] = Counter()
    truth_pairs_by_panel_source: Counter[str] = Counter()
    for idx, row in ranking.iterrows():
        documents.update(
            _decode_source_documents(
                row["source_documents_json"], f"source_documents_json row {idx}"
            )
        )
        panel_sources = _as_string_list(row["panel_sources"], f"panel_sources row {idx}")
        queries_by_panel_source.update(panel_sources)
        target_sources = _decode_target_sources(
            row["truth_target_panel_sources_json"],
            f"truth_target_panel_sources_json row {idx}",
        )
        truth_targets = set(_as_string_list(row["truth_targets"], f"truth_targets row {idx}"))
        if set(target_sources) != truth_targets:
            raise SystemExit("dual-cold truth-target source attribution is stale")
        for sources in target_sources.values():
            truth_pairs_by_panel_source.update(sources)
    criteria = {
        "min_queries": min_queries,
        "min_unique_truth_targets": min_targets,
        "min_unique_source_documents": min_documents,
        "max_truth_pair_target_fraction": max_target_fraction,
        "min_effective_target_count": min_effective_targets,
    }
    observed = {
        "queries": int(len(ranking)),
        "truth_pairs": int(truth_pairs),
        "unique_truth_targets": len(target_counts),
        "unique_source_documents": len(documents),
        "source_databases": sorted({source for source, _document in documents}),
        "source_documents": [
            {"source_db": source, "publication_key": document}
            for source, document in sorted(documents)
        ],
        "queries_by_panel_source": dict(sorted(queries_by_panel_source.items())),
        "truth_pairs_by_panel_source": dict(sorted(truth_pairs_by_panel_source.items())),
        "max_truth_pair_target_fraction": float(observed_max_fraction),
        "effective_target_count": float(effective_targets),
        "truth_pairs_by_target": dict(
            sorted(target_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
    }
    checks = {
        "min_queries": observed["queries"] >= min_queries,
        "min_unique_truth_targets": observed["unique_truth_targets"] >= min_targets,
        "min_unique_source_documents": (
            observed["unique_source_documents"] >= min_documents
        ),
        "max_truth_pair_target_fraction": (
            observed["max_truth_pair_target_fraction"] <= max_target_fraction
        ),
        "min_effective_target_count": (
            observed["effective_target_count"] >= min_effective_targets
        ),
    }
    return {
        "passes": all(checks.values()),
        "criteria": criteria,
        "observed": observed,
        "checks": checks,
        "claim_requires_scorable_truth_coverage": True,
        "policy": (
            "Cold-start performance is claimable only when the frozen positive ranking panel "
            "has sufficient query, target, and primary-document diversity and no target "
            "dominates truth-pair incidence. Adequacy is necessary but not sufficient: the "
            "frozen evaluation must also report at least one truth pair inside the retrieval "
            "model's scorable universe; sample size alone cannot make the claim."
        ),
    }


def _allocate_stratified_counts(
    total_cap: int,
    stratum_sizes: dict[Hashable, int],
) -> dict[Hashable, int]:
    total = sum(stratum_sizes.values())
    if total_cap >= total:
        return dict(stratum_sizes)
    raw = {
        label: (total_cap * count / total if total else 0.0)
        for label, count in stratum_sizes.items()
    }
    allocated = {
        label: min(count, int(math.floor(raw[label])))
        for label, count in stratum_sizes.items()
    }
    remaining = total_cap - sum(allocated.values())
    order = sorted(
        stratum_sizes,
        key=lambda label: (raw[label] - math.floor(raw[label]), stratum_sizes[label], label),
        reverse=True,
    )
    while remaining > 0:
        progressed = False
        for label in order:
            if allocated[label] >= stratum_sizes[label]:
                continue
            allocated[label] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break
    return allocated


def _calibration_pairs(
    df: pd.DataFrame,
    *,
    panel_name: str,
    cap: int | None,
    negative_threshold: float,
    positive_threshold: float,
    seed: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = df[df["identity_exclusion_reason"].eq("")]
    rows = df[(df["pactivity"] >= positive_threshold) | (df["pactivity"] <= negative_threshold)].copy()
    rows["label"] = (rows["pactivity"] >= positive_threshold).astype("int64")
    rows["endpoint_family"] = rows["endpoint"].map(ENDPOINT_FAMILIES)
    rows["connectivity_key"] = rows["standard_inchikey"].astype(str).str[:14]
    identity_columns = [
        "ligand_key",
        "standard_inchikey",
        "connectivity_key",
        "canonical_smiles",
        "standardization_route",
        "uniprot",
        "endpoint_family",
    ]
    collapsed_rows: list[dict[str, object]] = []
    conflicting_measurement_groups = 0
    conflicting_measurement_rows = 0
    grouped = rows.groupby(identity_columns, sort=True, dropna=False)
    for identity, group in grouped:
        labels = sorted(set(int(value) for value in group["label"]))
        if len(labels) != 1:
            conflicting_measurement_groups += 1
            conflicting_measurement_rows += int(len(group))
            continue
        identity_values = dict(zip(identity_columns, identity, strict=True))
        label = labels[0]
        pair_id = _stable_hash(
            SCHEMA_VERSION,
            "calibration",
            panel_name,
            identity_values["ligand_key"],
            identity_values["uniprot"],
            identity_values["endpoint_family"],
            label,
        )
        collapsed_rows.append(
            {
                "pair_id": pair_id,
                "query_id": identity_values["ligand_key"],
                **identity_values,
                "label": label,
                "split": "test" if panel_name == "dual_cold" else panel_name,
                "n_measurements": int(len(group)),
                "pactivity_min": float(group["pactivity"].min()),
                "pactivity_max": float(group["pactivity"].max()),
                "source_db": sorted({str(value).strip() for value in group["source_db"]}),
                "evidence_id": sorted({str(value).strip() for value in group["evidence_id"]}),
                "benchmark_id": sorted({str(value).strip() for value in group["benchmark_id"]}),
                "publication_key": sorted({str(value).strip() for value in group["publication_key"]}),
                "evidence_date_min": min(str(value).strip() for value in group["evidence_date"]),
                "evidence_date_max": max(str(value).strip() for value in group["evidence_date"]),
            }
        )
    rows = pd.DataFrame(collapsed_rows, columns=_calibration_columns_without_weight())
    if rows.empty:
        rows["sample_weight"] = pd.Series(dtype="float64")
        rows["selection_hash"] = pd.Series(dtype="str")
    else:
        rows["selection_hash"] = [
            _stable_hash(
                SCHEMA_VERSION,
                panel_name,
                seed,
                row.pair_id,
            )
            for row in rows.itertuples(index=False)
        ]
    rows["stratum_key"] = [
        (str(row.endpoint_family), int(row.label)) for row in rows.itertuples(index=False)
    ]
    stratum_sizes = {
        key: int(count)
        for key, count in rows["stratum_key"].value_counts(sort=False).to_dict().items()
    }
    if cap is None or cap >= len(rows):
        included_counts = dict(stratum_sizes)
    else:
        included_counts = _allocate_stratified_counts(cap, stratum_sizes)
    selected_parts = []
    for stratum_key in sorted(stratum_sizes):
        part = rows[rows["stratum_key"] == stratum_key].sort_values(
            ["selection_hash", "pair_id", "ligand_key", "uniprot", "endpoint_family"]
        )
        part = part.head(included_counts[stratum_key])
        selected_parts.append(part)
    selected = (
        pd.concat(selected_parts, ignore_index=True)
        if selected_parts
        else rows.head(0).copy()
    )
    selected = selected.sort_values(["selection_hash", "pair_id", "ligand_key", "uniprot"])
    selected = selected.reset_index(drop=True)
    weight_by_stratum = {
        key: (
            float(stratum_sizes[key] / included_counts[key])
            if included_counts[key] > 0
            else 0.0
        )
        for key in stratum_sizes
    }
    selected["sample_weight"] = selected["stratum_key"].map(weight_by_stratum).astype(float)
    inclusion = {
        f"{endpoint_family}|label={label}": {
            "endpoint_family": endpoint_family,
            "label": label,
            "population": stratum_sizes[(endpoint_family, label)],
            "sample": included_counts[(endpoint_family, label)],
            "inclusion_rate": (
                float(
                    included_counts[(endpoint_family, label)]
                    / stratum_sizes[(endpoint_family, label)]
                )
                if stratum_sizes[(endpoint_family, label)] > 0
                else 0.0
            ),
            "sample_weight": weight_by_stratum[(endpoint_family, label)],
        }
        for endpoint_family, label in sorted(stratum_sizes)
    }
    conflict_meta = {
        "conflicting_measurement_groups_excluded": conflicting_measurement_groups,
        "conflicting_measurement_rows_excluded": conflicting_measurement_rows,
    }
    return selected[
        [
            "pair_id",
            "query_id",
            "ligand_key",
            "standard_inchikey",
            "connectivity_key",
            "canonical_smiles",
            "standardization_route",
            "uniprot",
            "endpoint_family",
            "label",
            "sample_weight",
            "split",
            "n_measurements",
            "pactivity_min",
            "pactivity_max",
            "source_db",
            "evidence_id",
            "benchmark_id",
            "publication_key",
            "evidence_date_min",
            "evidence_date_max",
        ]
    ], {"strata": inclusion, "conflicts": conflict_meta}


def _calibration_columns_without_weight() -> list[str]:
    return [
        "pair_id",
        "query_id",
        "ligand_key",
        "standard_inchikey",
        "connectivity_key",
        "canonical_smiles",
        "standardization_route",
        "uniprot",
        "endpoint_family",
        "label",
        "split",
        "n_measurements",
        "pactivity_min",
        "pactivity_max",
        "source_db",
        "evidence_id",
        "benchmark_id",
        "publication_key",
        "evidence_date_min",
        "evidence_date_max",
    ]


def _write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _copy_atomic(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    shutil.copyfile(src, tmp)
    tmp.replace(dst)


def _identity_exclusion_summary(df: pd.DataFrame) -> dict[str, Any]:
    excluded = df[~df["identity_exclusion_reason"].eq("")]
    examples = (
        excluded[["ligand_smiles", "source_db", "identity_exclusion_reason"]]
        .drop_duplicates()
        .sort_values(["identity_exclusion_reason", "source_db", "ligand_smiles"])
        .head(20)
    )
    return {
        "rows": int(len(excluded)),
        "unique_ligand_smiles": int(excluded["ligand_smiles"].nunique()),
        "by_reason": {
            str(key): int(value)
            for key, value in sorted(
                excluded["identity_exclusion_reason"].value_counts().to_dict().items()
            )
        },
        "by_source_db": {
            str(key): int(value)
            for key, value in sorted(excluded["source_db"].value_counts().to_dict().items())
        },
        "examples": [
            {
                "ligand_smiles_sha256": hashlib.sha256(
                    str(row.ligand_smiles).encode("utf-8")
                ).hexdigest(),
                "source_db": str(row.source_db),
                "reason": str(row.identity_exclusion_reason),
            }
            for row in examples.itertuples(index=False)
        ],
    }


def _output_paths(out_dir: Path) -> list[Path]:
    return [out_dir / name for name in OUTPUT_NAMES]


def _remove_outputs(out_dir: Path) -> None:
    for path in _output_paths(out_dir):
        path.unlink(missing_ok=True)
        path.with_suffix(path.suffix + ".tmp").unlink(missing_ok=True)


def _output_meta(out_dir: Path) -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}
    for name in DATA_OUTPUT_NAMES:
        path = out_dir / name
        if name.endswith(".parquet"):
            rows = _parquet_row_count(path)
        elif name.endswith(".csv"):
            rows = _csv_row_count(path)
        else:
            rows = None
        record: dict[str, Any] = {"path": str(path.resolve()), "sha256": _sha256(path)}
        if rows is not None:
            record["rows"] = rows
        meta[name] = record
    return meta


def build_panels(args: argparse.Namespace) -> None:
    try:
        _remove_outputs(args.out_dir)
        if not math.isfinite(args.negative_threshold):
            raise SystemExit("--negative-threshold must be finite")
        if not math.isfinite(args.positive_threshold):
            raise SystemExit("--positive-threshold must be finite")
        if args.negative_threshold >= args.positive_threshold:
            raise SystemExit("--negative-threshold must be less than --positive-threshold")
        if args.dev_query_count < 0 or args.test_query_count < 0:
            raise SystemExit("query counts must be non-negative")
        if bool(args.rcsb_ranking_queries) != bool(args.rcsb_panel_manifest):
            raise SystemExit(
                "--rcsb-ranking-queries and --rcsb-panel-manifest must be supplied together"
            )
        calibration_cap = None if args.calibration_cap == 0 else args.calibration_cap
        if args.calibration_cap < 0:
            raise SystemExit("--calibration-cap must be non-negative")
        if min(
            args.dual_cold_min_queries,
            args.dual_cold_min_targets,
            args.dual_cold_min_documents,
        ) < 1:
            raise SystemExit("dual-cold minimum counts must be >= 1")
        if not 0.0 < args.dual_cold_max_target_fraction <= 1.0:
            raise SystemExit("--dual-cold-max-target-fraction must be in (0, 1]")
        if (
            not math.isfinite(args.dual_cold_min_effective_targets)
            or args.dual_cold_min_effective_targets <= 0.0
        ):
            raise SystemExit("--dual-cold-min-effective-targets must be finite and > 0")

        benchmark_manifest = _read_json(args.benchmark_manifest, "Benchmark manifest")
        _require_schema(benchmark_manifest, BENCHMARK_SCHEMA_VERSION, "Benchmark manifest")
        retrieval_manifest = _read_json(args.retrieval_index_manifest, "Retrieval index manifest")
        retrieval_outputs, bound_retrieval_inputs = _validate_retrieval_manifest(
            args.retrieval_index_manifest,
            retrieval_manifest,
            benchmark_manifest_path=args.benchmark_manifest,
            benchmark_manifest=benchmark_manifest,
        )

        input_meta = {
            "benchmark_manifest": bound_retrieval_inputs["benchmark_manifest"],
            "train.parquet": bound_retrieval_inputs["train.parquet"],
            "retrieval_index_manifest": {
                "path": str(args.retrieval_index_manifest.resolve()),
                "sha256": _sha256(args.retrieval_index_manifest),
                "schema_version": retrieval_manifest["schema_version"],
                "validated_outputs": retrieval_outputs,
            },
        }
        split_specs = {
            "dev": (args.dev_parquet, "dev.parquet"),
            "test": (args.test_parquet, "test.parquet"),
            "dual_cold": (args.dual_cold_parquet, "dual_cold.parquet"),
        }
        activity_frames: dict[str, pd.DataFrame] = {}
        for split, (path, name) in split_specs.items():
            sha, rows = _validate_benchmark_input(path, benchmark_manifest, split=split, name=name)
            df = _load_activity(path, split)
            if len(df) != rows:
                raise SystemExit(f"Loaded row count changed for {name}: {len(df)} != {rows}")
            input_meta[name] = {"path": str(path.resolve()), "sha256": sha, "rows": rows}
            activity_frames[split] = df

        rcsb_ranking: pd.DataFrame | None = None
        if args.rcsb_ranking_queries is not None:
            rcsb_ranking, rcsb_meta = _validate_rcsb_panel(
                args.rcsb_ranking_queries,
                args.rcsb_panel_manifest,
                benchmark_manifest_path=args.benchmark_manifest,
            )
            input_meta["rcsb_holo_direct_contact_panel"] = rcsb_meta

        known_panel_contract = validate_frozen_known_panel(args.known_panel)
        known_panel_rows = int(known_panel_contract["rows"])
        known_panel_sha = str(known_panel_contract["sha256"])
        input_meta["known_panel.csv"] = {
            "path": str(args.known_panel.resolve()),
            "sha256": known_panel_sha,
            "rows": known_panel_rows,
            "contract": known_panel_contract,
        }

        dev_ranking = _ranking_queries(
            activity_frames["dev"],
            panel_name="dev",
            cap=args.dev_query_count,
            positive_threshold=args.positive_threshold,
            seed=args.selection_seed,
        )
        test_ranking = _ranking_queries(
            activity_frames["test"],
            panel_name="test",
            cap=args.test_query_count,
            positive_threshold=args.positive_threshold,
            seed=args.selection_seed,
        )
        activity_dual_ranking = _ranking_queries(
            activity_frames["dual_cold"],
            panel_name="dual_cold",
            cap=None,
            positive_threshold=args.positive_threshold,
            seed=args.selection_seed,
        )
        activity_dual_ranking = _activity_dual_provenance(
            activity_dual_ranking,
            activity_frames["dual_cold"],
            positive_threshold=args.positive_threshold,
        )
        dual_ranking, dual_source_counts = _combine_dual_rankings(
            activity_dual_ranking,
            rcsb_ranking,
        )
        dual_cold_adequacy = _dual_cold_adequacy(
            dual_ranking,
            min_queries=args.dual_cold_min_queries,
            min_targets=args.dual_cold_min_targets,
            min_documents=args.dual_cold_min_documents,
            max_target_fraction=args.dual_cold_max_target_fraction,
            min_effective_targets=args.dual_cold_min_effective_targets,
        )
        dev_calibration, dev_calibration_inclusion = _calibration_pairs(
            activity_frames["dev"],
            panel_name="dev",
            cap=calibration_cap,
            negative_threshold=args.negative_threshold,
            positive_threshold=args.positive_threshold,
            seed=args.selection_seed,
        )
        test_calibration, test_calibration_inclusion = _calibration_pairs(
            activity_frames["test"],
            panel_name="test",
            cap=calibration_cap,
            negative_threshold=args.negative_threshold,
            positive_threshold=args.positive_threshold,
            seed=args.selection_seed,
        )

        _write_parquet_atomic(dev_ranking, args.out_dir / "dev_ranking_queries.parquet")
        _write_parquet_atomic(test_ranking, args.out_dir / "test_ranking_queries.parquet")
        _write_parquet_atomic(dual_ranking, args.out_dir / "dual_cold_ranking_queries.parquet")
        _write_parquet_atomic(dev_calibration, args.out_dir / "dev_calibration_pairs.parquet")
        _write_parquet_atomic(test_calibration, args.out_dir / "test_calibration_pairs.parquet")
        _copy_atomic(args.known_panel, args.out_dir / "known_panel.csv")
        if _sha256(args.out_dir / "known_panel.csv") != known_panel_sha:
            raise SystemExit("Copied known_panel.csv hash changed")

        selection = {
            "seed": args.selection_seed,
            "identity_exclusions": {
                split: _identity_exclusion_summary(frame)
                for split, frame in sorted(activity_frames.items())
            },
            "ranking_queries": {
                "dev": {
                    "cap": args.dev_query_count,
                    "candidate_ligands_with_positive": int(
                        activity_frames["dev"]
                        .loc[
                            activity_frames["dev"]["identity_exclusion_reason"].eq("")
                            & (activity_frames["dev"]["pactivity"] >= args.positive_threshold),
                            "ligand_key",
                        ]
                        .nunique()
                    ),
                    "selected": int(len(dev_ranking)),
                },
                "test": {
                    "cap": args.test_query_count,
                    "candidate_ligands_with_positive": int(
                        activity_frames["test"]
                        .loc[
                            activity_frames["test"]["identity_exclusion_reason"].eq("")
                            & (activity_frames["test"]["pactivity"] >= args.positive_threshold),
                            "ligand_key",
                        ]
                        .nunique()
                    ),
                    "selected": int(len(test_ranking)),
                },
                "dual_cold": {
                    "cap": "all",
                    "candidate_ligands_with_positive": int(
                        activity_frames["dual_cold"]
                        .loc[
                            activity_frames["dual_cold"]["identity_exclusion_reason"].eq("")
                            & (
                                activity_frames["dual_cold"]["pactivity"]
                                >= args.positive_threshold
                            ),
                            "ligand_key",
                        ]
                        .nunique()
                    ),
                    "selected": int(len(dual_ranking)),
                    "source_query_counts": dual_source_counts,
                    "adequacy": dual_cold_adequacy,
                },
            },
            "calibration_pairs": {
                "cap": args.calibration_cap,
                "cap_policy": (
                    "default cap is 1024 eligible pair rows per split; cap=0 retains all; "
                    "capped panels sample deterministically with proportional allocation across "
                    "(endpoint_family,label) strata and emit inverse-probability sample_weight"
                ),
                "dev": {
                    "selected_positive": int((dev_calibration["label"] == 1).sum()),
                    "selected_negative": int((dev_calibration["label"] == 0).sum()),
                    "inclusion_by_stratum": dev_calibration_inclusion["strata"],
                    "conflicts": dev_calibration_inclusion["conflicts"],
                    "gray_excluded": int(
                        (
                            activity_frames["dev"]["identity_exclusion_reason"].eq("")
                            & (activity_frames["dev"]["pactivity"] > args.negative_threshold)
                            & (activity_frames["dev"]["pactivity"] < args.positive_threshold)
                        ).sum()
                    ),
                },
                "test": {
                    "selected_positive": int((test_calibration["label"] == 1).sum()),
                    "selected_negative": int((test_calibration["label"] == 0).sum()),
                    "inclusion_by_stratum": test_calibration_inclusion["strata"],
                    "conflicts": test_calibration_inclusion["conflicts"],
                    "gray_excluded": int(
                        (
                            activity_frames["test"]["identity_exclusion_reason"].eq("")
                            & (activity_frames["test"]["pactivity"] > args.negative_threshold)
                            & (activity_frames["test"]["pactivity"] < args.positive_threshold)
                        ).sum()
                    ),
                },
            },
        }
        manifest_payload = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "passes_panel_adequacy_gate": dual_cold_adequacy["passes"],
            "inputs": input_meta,
            "outputs": {},
            "algorithm": {
                "positive_threshold": args.positive_threshold,
                "negative_threshold": args.negative_threshold,
                "date_boundaries": {
                    "dev": "2024-01-01..2024-12-31",
                    "test": ">=2025-01-01",
                    "dual_cold": "test split and all dual-cold claim flags true",
                },
                "ranking_policy": (
                    "standardize unique ligand queries; exclude ligands without measured "
                    "positives before SHA256 ordering; collect truth targets only after selection; "
                    "merge independently frozen RCSB direct-contact truths by exact ligand key"
                ),
                "calibration_policy": (
                    "measured rows only; positives >= positive_threshold; negatives <= "
                    "negative_threshold; gray rows excluded; no unmeasured negatives fabricated; "
                    "RCSB positive-only observations are never used for calibration"
                ),
                "rcsb_usage_policy": (
                    "positive-only direct-contact test ranking evidence; never training, model "
                    "selection, negative-label construction, affinity estimation, or calibration"
                ),
                "query_identity_policy": (
                    "require a deterministic RDKit FragmentParent, canonical SMILES, standard "
                    "InChIKey, and dataset-independent structure key; exclude non-reproducible "
                    "source structures without guessing bond order and record every exclusion"
                ),
                "endpoint_families": ENDPOINT_FAMILIES,
                "target_universe_policy": "unchanged; no candidate target universe emitted or assisted",
                "cold_start_claim_policy": (
                    "dual-cold metrics remain diagnostic unless passes_panel_adequacy_gate is true"
                ),
            },
            "selection": selection,
        }
        manifest_payload["outputs"] = _output_meta(args.out_dir)
        _write_json_atomic(manifest_payload, args.out_dir / "manifest.json")
    except BaseException:
        _remove_outputs(args.out_dir)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-parquet", type=Path, default=Path("data/activity_benchmark_202608/dev.parquet"))
    parser.add_argument("--test-parquet", type=Path, default=Path("data/activity_benchmark_202608/test.parquet"))
    parser.add_argument(
        "--dual-cold-parquet",
        type=Path,
        default=Path("data/activity_benchmark_202608/dual_cold.parquet"),
    )
    parser.add_argument(
        "--benchmark-manifest",
        type=Path,
        default=Path("data/activity_benchmark_202608/manifest.json"),
    )
    parser.add_argument("--retrieval-index-manifest", required=True, type=Path)
    parser.add_argument("--rcsb-ranking-queries", type=Path)
    parser.add_argument("--rcsb-panel-manifest", type=Path)
    parser.add_argument(
        "--known-panel",
        type=Path,
        default=Path("data/validation/skin_known_target_panel.csv"),
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--dev-query-count", type=int, default=256)
    parser.add_argument("--test-query-count", type=int, default=256)
    parser.add_argument("--dual-cold-min-queries", type=int, default=100)
    parser.add_argument("--dual-cold-min-targets", type=int, default=20)
    parser.add_argument("--dual-cold-min-documents", type=int, default=20)
    parser.add_argument("--dual-cold-max-target-fraction", type=float, default=0.20)
    parser.add_argument("--dual-cold-min-effective-targets", type=float, default=10.0)
    parser.add_argument(
        "--calibration-cap",
        type=int,
        default=1024,
        help="Total measured positive/negative pair rows to retain per split; 0 keeps all.",
    )
    parser.add_argument("--negative-threshold", type=float, default=5.0)
    parser.add_argument("--positive-threshold", type=float, default=6.0)
    parser.add_argument("--selection-seed", default="skinscout.activity-recovery-panels.v2")
    build_panels(parser.parse_args())


if __name__ == "__main__":
    main()
