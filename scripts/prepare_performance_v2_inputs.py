#!/usr/bin/env python3
"""Prepare production performance-v2 CSV inputs from benchmark parquet files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA_VERSION = "skinscout.performance-v2-inputs.v1"
TRAIN_COLUMNS = ["ligand_id", "target_id", "evidence_state", "measured_label", "ontology"]
LIGAND_COLUMNS = ["ligand_id", "smiles"]
TARGET_COLUMNS = ["target_id", "sequence", "target_cluster_30", "target_cluster_50"]
BENCHMARK_REQUIRED_COLUMNS = {
    "source_db",
    "uniprot",
    "ligand_id",
    "ligand_smiles",
    "ligand_inchikey",
    "endpoint",
    "activity_class",
    "pactivity",
    "evidence_date",
    "split",
}
TRAIN_CUTOFF = date(2023, 12, 31)
RETRIEVAL_LIGAND_REQUIRED_COLUMNS = {
    "ligand_index",
    "ligand_key",
    "standard_inchikey",
    "canonical_smiles",
}
DIRECT_BINDING_ENDPOINTS = {"KD", "KI"}
FUNCTIONAL_ENDPOINTS = {"EC50", "IC50"}
NEGATIVE_THRESHOLD = 5.0
POSITIVE_THRESHOLD = 6.0
POSITIVE_ACTIVITY_CLASSES = {"strong_positive", "weak_positive"}
NEGATIVE_ACTIVITY_CLASSES = {"low_potency_quantitative"}
GRAY_ACTIVITY_CLASSES = {"gray_unmeasured", "unmeasured"}


class InputAdapterError(ValueError):
    """Raised when production input preparation must fail closed."""


def _pd():
    import pandas as pd

    return pd


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_record(path: Path, rows: int, *, final_path: Path | None = None) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise InputAdapterError(f"artifact is missing or empty: {path}")
    return {
        "path": str((final_path or path).resolve()),
        "sha256": _sha256_file(path),
        "bytes": int(path.stat().st_size),
        "rows": int(rows),
    }


def _read_parquet(path: Path, required: set[str], label: str):
    if not path.is_file() or path.stat().st_size == 0:
        raise InputAdapterError(f"{label} parquet is missing or empty: {path}")
    try:
        import pyarrow.parquet as pq

        columns = set(pq.ParquetFile(path).schema.names)
    except Exception as exc:  # pragma: no cover - exact engine errors vary.
        raise InputAdapterError(f"failed to inspect {label} parquet: {path}") from exc
    missing = required - columns
    if missing:
        raise InputAdapterError(f"{label} parquet missing required columns: {sorted(missing)}")
    try:
        frame = _pd().read_parquet(path)
    except Exception as exc:  # pragma: no cover - exact engine errors vary.
        raise InputAdapterError(f"failed to read {label} parquet: {path}") from exc
    return frame


def _clean_text_series(series):
    return series.fillna("").astype(str).str.strip()


def _read_retrieval_ligands(path: Path):
    pd = _pd()
    frame = _read_parquet(path, RETRIEVAL_LIGAND_REQUIRED_COLUMNS, "retrieval ligands")
    frame = frame[list(RETRIEVAL_LIGAND_REQUIRED_COLUMNS)].copy()
    for column in ("ligand_key", "standard_inchikey", "canonical_smiles"):
        frame[column] = _clean_text_series(frame[column])
    if frame[["ligand_key", "standard_inchikey", "canonical_smiles"]].eq("").any().any():
        raise InputAdapterError("retrieval ligands require nonblank canonical identifiers")
    ligand_index = pd.to_numeric(frame["ligand_index"], errors="coerce")
    if ligand_index.isna().any():
        raise InputAdapterError("retrieval ligands contain nonnumeric ligand_index values")
    ligand_index_values = ligand_index.to_numpy(dtype=np.float64)
    if (
        not np.isfinite(ligand_index_values).all()
        or not np.equal(ligand_index_values, np.trunc(ligand_index_values)).all()
        or (ligand_index_values < 0).any()
        or (ligand_index_values > 999_999_999_999).any()
    ):
        raise InputAdapterError(
            "retrieval ligands require finite integer ligand_index values in [0, 999999999999]"
        )
    frame["ligand_index"] = ligand_index.astype("int64")
    if frame["ligand_index"].duplicated().any():
        raise InputAdapterError("retrieval ligands require unique ligand_index values")
    if frame["ligand_key"].duplicated().any():
        raise InputAdapterError("retrieval ligands require unique ligand_key values")
    if frame["canonical_smiles"].duplicated().any():
        raise InputAdapterError("retrieval ligands require unique canonical_smiles values")
    frame["ligand_id"] = frame["ligand_index"].map(lambda value: f"ligand_{int(value):012d}")
    return frame.sort_values("ligand_index", kind="mergesort").reset_index(drop=True)


def _normalize_endpoint(value: object) -> str:
    return str(value).strip().upper()


def _standardized_ligand_key(raw_smiles: str) -> tuple[str, str]:
    try:
        from rdkit import Chem
        from rdkit.Chem.MolStandardize import rdMolStandardize
    except ImportError as exc:  # pragma: no cover - depends on production env.
        raise InputAdapterError("RDKit is required to standardize benchmark ligand_smiles fallback joins") from exc

    mol = Chem.MolFromSmiles(raw_smiles)
    if mol is None:
        params = Chem.SmilesParserParams()
        params.strictCXSMILES = False
        mol = Chem.MolFromSmiles(raw_smiles, params)
    if mol is None:
        raise InputAdapterError(f"invalid benchmark ligand_smiles: {raw_smiles!r}")
    parent = rdMolStandardize.FragmentParent(mol)
    canonical = Chem.MolToSmiles(parent, canonical=True, isomericSmiles=True)
    reparsed = Chem.MolFromSmiles(canonical)
    if reparsed is None:
        try:
            canonical = Chem.MolToSmiles(parent, canonical=True, isomericSmiles=True, kekuleSmiles=True)
        except (RuntimeError, ValueError) as exc:
            raise InputAdapterError(f"standardized benchmark ligand_smiles is invalid: {raw_smiles!r}") from exc
        reparsed = Chem.MolFromSmiles(canonical)
        if reparsed is None:
            raise InputAdapterError(f"standardized benchmark ligand_smiles is invalid: {raw_smiles!r}")
    standard_inchikey = Chem.MolToInchiKey(reparsed)
    if not standard_inchikey:
        raise InputAdapterError(f"unable to derive standardized InChIKey for benchmark ligand_smiles: {raw_smiles!r}")
    suffix = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    return f"{standard_inchikey}#SMILES-{suffix}", canonical


def _standardized_ligand_keys(raw_smiles_values: list[str], workers: int) -> dict[str, tuple[str, str]]:
    if workers < 1:
        raise InputAdapterError("--standardization-workers must be >= 1")
    if workers == 1 or len(raw_smiles_values) < 2:
        pairs = [(raw_smiles, _standardized_ligand_key(raw_smiles)) for raw_smiles in raw_smiles_values]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            values = list(pool.map(_standardized_ligand_key, raw_smiles_values, chunksize=256))
        pairs = list(zip(raw_smiles_values, values, strict=True))
    return dict(pairs)


def _state_and_label(activity_class: object, pactivity: object) -> tuple[str, str]:
    activity_class_value = str(activity_class).strip()
    supported = POSITIVE_ACTIVITY_CLASSES | NEGATIVE_ACTIVITY_CLASSES | GRAY_ACTIVITY_CLASSES
    if activity_class_value not in supported:
        raise InputAdapterError(
            "benchmark train contains unsupported activity_class labels: "
            f"{activity_class_value!r}"
        )
    try:
        value = float(pactivity)
    except (TypeError, ValueError) as exc:
        raise InputAdapterError("benchmark train contains nonnumeric pactivity values") from exc
    if not math.isfinite(value):
        raise InputAdapterError("benchmark train contains nonfinite pactivity values")
    if value >= POSITIVE_THRESHOLD:
        return "measured_positive", "1"
    if value <= NEGATIVE_THRESHOLD:
        return "measured_negative", "0"
    return "gray_unmeasured", ""


def _prepare_benchmark(path: Path):
    frame = _read_parquet(path, BENCHMARK_REQUIRED_COLUMNS, "benchmark train")
    frame = frame[
        [
            "source_db",
            "uniprot",
            "ligand_id",
            "ligand_smiles",
            "ligand_inchikey",
            "endpoint",
            "activity_class",
            "pactivity",
            "evidence_date",
            "split",
        ]
    ].copy()
    for column in (
        "source_db",
        "uniprot",
        "ligand_id",
        "ligand_smiles",
        "ligand_inchikey",
        "endpoint",
        "activity_class",
        "evidence_date",
        "split",
    ):
        frame[column] = _clean_text_series(frame[column])
    if frame[["uniprot", "ligand_smiles", "endpoint", "activity_class"]].eq("").any().any():
        raise InputAdapterError("benchmark train contains blank ligand, target, or endpoint identifiers")
    if not frame["split"].eq("train").all():
        observed = sorted(set(frame["split"]))
        raise InputAdapterError(
            f"benchmark training adapter accepts split=train only; observed {observed}"
        )
    try:
        evidence_dates = frame["evidence_date"].map(date.fromisoformat)
    except (TypeError, ValueError) as exc:
        raise InputAdapterError("benchmark train contains invalid ISO evidence_date values") from exc
    if (evidence_dates > TRAIN_CUTOFF).any():
        latest = max(evidence_dates)
        raise InputAdapterError(
            f"benchmark train contains evidence after preregistered cutoff {TRAIN_CUTOFF}: {latest}"
        )
    states_labels = [
        _state_and_label(activity_class, pactivity)
        for activity_class, pactivity in zip(
            frame["activity_class"], frame["pactivity"], strict=True
        )
    ]
    frame["evidence_state"] = [state for state, _ in states_labels]
    frame["measured_label"] = [label for _, label in states_labels]
    frame["endpoint_normalized"] = [_normalize_endpoint(value) for value in frame["endpoint"]]
    unsupported = sorted(set(frame["endpoint_normalized"]) - {"EC50", "IC50", "KD", "KI"})
    if unsupported:
        raise InputAdapterError(f"benchmark train contains unsupported endpoints: {unsupported}")
    return frame


def _join_ligands(benchmark, retrieval_ligands, *, standardization_workers: int):
    joined = benchmark.merge(
        retrieval_ligands[["canonical_smiles", "ligand_key", "ligand_id", "ligand_index"]],
        how="left",
        left_on="ligand_smiles",
        right_on="canonical_smiles",
        validate="many_to_one",
    )
    joined["ligand_join_method"] = np.where(
        joined["ligand_id_y"].notna(),
        "exact_canonical_smiles",
        "",
    )
    missing = joined["ligand_id_y"].isna()
    if missing.any():
        counts = retrieval_ligands["standard_inchikey"].value_counts()
        unique_inchikeys = set(counts[counts == 1].index.astype(str))
        inchikey_fallback = (
            retrieval_ligands[retrieval_ligands["standard_inchikey"].isin(unique_inchikeys)]
            .set_index("standard_inchikey")[["ligand_id", "ligand_index", "canonical_smiles"]]
            .to_dict("index")
        )
        benchmark_missing = benchmark.loc[missing].copy()
        safe_inchikey = benchmark_missing["ligand_inchikey"].astype(str).isin(inchikey_fallback)
        if safe_inchikey.any():
            safe_rows = benchmark_missing.loc[safe_inchikey]
            safe_records = safe_rows["ligand_inchikey"].astype(str).map(inchikey_fallback)
            joined.loc[safe_rows.index, "ligand_id_y"] = [
                record["ligand_id"] for record in safe_records
            ]
            joined.loc[safe_rows.index, "ligand_index"] = [
                record["ligand_index"] for record in safe_records
            ]
            joined.loc[safe_rows.index, "canonical_smiles"] = [
                record["canonical_smiles"] for record in safe_records
            ]
            joined.loc[safe_rows.index, "ligand_join_method"] = (
                "unique_standard_inchikey"
            )

        missing = joined["ligand_id_y"].isna()
    if missing.any():
        raw_smiles_values = sorted(set(benchmark.loc[missing, "ligand_smiles"].astype(str)))
        raw_to_key = _standardized_ligand_keys(raw_smiles_values, standardization_workers)
        fallback_input = benchmark.loc[missing].copy()
        fallback_input["standardized_ligand_key"] = [
            raw_to_key[str(smiles)][0] for smiles in fallback_input["ligand_smiles"]
        ]
        fallback = fallback_input.merge(
            retrieval_ligands[["ligand_key", "ligand_id", "ligand_index", "canonical_smiles"]],
            how="left",
            left_on="standardized_ligand_key",
            right_on="ligand_key",
            validate="many_to_one",
        )
        still_missing = fallback["ligand_id_y"].isna()
        if still_missing.any():
            examples = sorted(set(fallback.loc[still_missing, "standardized_ligand_key"].astype(str)))[:5]
            raise InputAdapterError(f"benchmark ligands missing from retrieval ligands: {examples}")
        joined.loc[missing, "ligand_id_y"] = fallback["ligand_id_y"].to_numpy()
        joined.loc[missing, "ligand_index"] = fallback["ligand_index"].to_numpy()
        joined.loc[missing, "canonical_smiles"] = fallback["canonical_smiles"].to_numpy()
        joined.loc[missing, "ligand_join_method"] = "standardized_ligand_key"
    joined["ligand_id"] = joined["ligand_id_y"].astype(str)
    joined["target_id"] = joined["uniprot"].astype(str)
    methods = joined["ligand_join_method"].astype(str)
    allowed_methods = {
        "exact_canonical_smiles",
        "unique_standard_inchikey",
        "standardized_ligand_key",
    }
    if methods.eq("").any() or not set(methods).issubset(allowed_methods):
        raise InputAdapterError("ligand join method provenance is incomplete")
    audit_columns = [
        "source_db",
        "uniprot",
        "ligand_smiles",
        "ligand_inchikey",
        "endpoint",
        "pactivity",
        "ligand_id",
        "ligand_join_method",
    ]
    audit = joined[audit_columns].copy()
    audit = audit.sort_values(audit_columns, kind="mergesort").reset_index(drop=True)
    audit_sha256 = hashlib.sha256(
        json.dumps(
            audit.to_dict("records"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    method_counts = {
        method: int(methods.eq(method).sum()) for method in sorted(allowed_methods)
    }
    return joined, {
        "methods": {
            "exact_canonical_smiles": "benchmark ligand_smiles equals retrieval canonical_smiles",
            "unique_standard_inchikey": "exact SMILES unavailable; unique retrieval standard_inchikey match",
            "standardized_ligand_key": "exact SMILES/InChIKey unavailable; RDKit-standardized ligand_key match",
        },
        "counts": method_counts,
        "fallback_rows": int(len(joined) - method_counts["exact_canonical_smiles"]),
        "row_binding_sha256": audit_sha256,
    }


def _collapse_train_rows(joined):
    rows: list[dict[str, str]] = []
    conflict_pairs = 0
    conflict_rows = 0
    classified = joined.copy()
    classified["ontology_group"] = classified["endpoint_normalized"].map(
        lambda endpoint: (
            "direct_binding_reversible"
            if endpoint in DIRECT_BINDING_ENDPOINTS
            else "functional_modulation"
        )
    )
    for (ligand_id, target_id, ontology_group), group in classified.groupby(
        ["ligand_id", "target_id", "ontology_group"], sort=True, dropna=False
    ):
        labels = sorted({str(value) for value in group["measured_label"] if str(value) != ""})
        if len(labels) > 1:
            conflict_pairs += 1
            conflict_rows += int(len(group))
            evidence_state = "gray_unmeasured"
            measured_label = ""
            ontology = str(ontology_group)
        elif labels:
            measured_label = labels[0]
            evidence_state = "measured_positive" if measured_label == "1" else "measured_negative"
            ontology = str(ontology_group)
        else:
            evidence_state = "gray_unmeasured"
            measured_label = ""
            ontology = str(ontology_group)
        rows.append(
            {
                "ligand_id": str(ligand_id),
                "target_id": str(target_id),
                "evidence_state": evidence_state,
                "measured_label": measured_label,
                "ontology": ontology,
            }
        )
    frame = _pd().DataFrame(rows, columns=TRAIN_COLUMNS)
    frame = frame.drop_duplicates(TRAIN_COLUMNS).sort_values(
        ["ligand_id", "target_id", "ontology"], kind="mergesort"
    )
    if frame.duplicated(["ligand_id", "target_id", "ontology"]).any():
        raise InputAdapterError("train CSV would contain duplicate ligand-target-ontology rows")
    return frame.reset_index(drop=True), {
        "conflicting_measured_label_pairs_collapsed": conflict_pairs,
        "conflicting_measured_label_rows_collapsed": conflict_rows,
    }


def _parse_fasta(path: Path) -> dict[str, str]:
    if not path.is_file() or path.stat().st_size == 0:
        raise InputAdapterError(f"target FASTA is missing or empty: {path}")
    sequences: dict[str, str] = {}
    current_id: str | None = None
    chunks: list[str] = []
    valid = re.compile(r"^[A-Z*]+$")

    def flush() -> None:
        nonlocal chunks, current_id
        if current_id is None:
            return
        sequence = "".join(chunks).replace(" ", "").upper()
        if not sequence:
            chunks = []
            return
        if not valid.fullmatch(sequence):
            raise InputAdapterError(f"target FASTA contains invalid sequence for {current_id}")
        prior = sequences.get(current_id)
        if prior is not None and prior != sequence:
            raise InputAdapterError(f"target FASTA contains conflicting sequences for {current_id}")
        sequences[current_id] = sequence
        chunks = []

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                current_id = line[1:].strip().split()[0]
                if not current_id:
                    raise InputAdapterError("target FASTA contains a blank header")
                chunks = []
            else:
                if current_id is None:
                    raise InputAdapterError("target FASTA sequence appeared before any header")
                chunks.append(line)
    flush()
    if not sequences:
        raise InputAdapterError(f"target FASTA contains no sequences: {path}")
    return sequences


def _read_target_clusters(path: Path):
    if not path.is_file() or path.stat().st_size == 0:
        raise InputAdapterError(f"target clusters CSV is missing or empty: {path}")
    try:
        frame = _pd().read_csv(path)
    except Exception as exc:
        raise InputAdapterError(f"failed to read target clusters CSV: {path}") from exc
    required = {"uniprot", "target_cluster_30", "target_cluster_50"}
    missing = required - set(frame.columns)
    if missing:
        raise InputAdapterError(f"target clusters CSV missing required columns: {sorted(missing)}")
    frame = frame[["uniprot", "target_cluster_30", "target_cluster_50"]].copy()
    for column in frame.columns:
        frame[column] = _clean_text_series(frame[column])
    if frame.eq("").any().any():
        raise InputAdapterError("target clusters CSV requires nonblank uniprot,target_cluster_30,target_cluster_50")
    if frame["uniprot"].duplicated().any():
        raise InputAdapterError("target clusters CSV requires unique uniprot values")
    return frame.set_index("uniprot")


def _prepare_ligands(train, retrieval_ligands):
    ids = sorted(set(train["ligand_id"].astype(str)))
    frame = retrieval_ligands[retrieval_ligands["ligand_id"].isin(ids)].copy()
    missing = sorted(set(ids) - set(frame["ligand_id"].astype(str)))
    if missing:
        raise InputAdapterError(f"missing ligand structures for output ligand_ids: {missing[:5]}")
    frame = frame[["ligand_id", "canonical_smiles"]].rename(columns={"canonical_smiles": "smiles"})
    frame = frame.sort_values("ligand_id", kind="mergesort").reset_index(drop=True)
    if frame["ligand_id"].duplicated().any() or frame[["ligand_id", "smiles"]].eq("").any().any():
        raise InputAdapterError("ligand output requires unique nonblank ligand_id and smiles")
    return frame[LIGAND_COLUMNS]


def _prepare_targets(train, fasta_path: Path, clusters_path: Path):
    sequences = _parse_fasta(fasta_path)
    clusters = _read_target_clusters(clusters_path)
    training_targets = sorted(set(train["target_id"].astype(str)))
    missing_sequences = [target_id for target_id in training_targets if target_id not in sequences]
    if missing_sequences:
        raise InputAdapterError(f"benchmark targets missing from target FASTA: {missing_sequences[:5]}")
    missing_clusters = [target_id for target_id in sorted(sequences) if target_id not in clusters.index]
    if missing_clusters:
        raise InputAdapterError(f"target FASTA targets missing from target clusters CSV: {missing_clusters[:5]}")
    frame = _pd().DataFrame(
        [
            {
                "target_id": target_id,
                "sequence": sequences[target_id],
                "target_cluster_30": str(clusters.loc[target_id, "target_cluster_30"]),
                "target_cluster_50": str(clusters.loc[target_id, "target_cluster_50"]),
            }
            for target_id in sorted(sequences)
        ],
        columns=TARGET_COLUMNS,
    )
    if frame["target_id"].duplicated().any() or frame.eq("").any().any():
        raise InputAdapterError("target output requires unique nonblank target_id, sequence, and cluster values")
    return frame


def _write_csv_temp(frame, final_path: Path) -> Path:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{final_path.name}.", suffix=".tmp", dir=final_path.parent)
    os.close(fd)
    tmp = Path(raw_tmp)
    frame.to_csv(tmp, index=False)
    return tmp


def _write_json_temp(payload: Mapping[str, Any], final_path: Path) -> Path:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{final_path.name}.", suffix=".tmp", dir=final_path.parent)
    os.close(fd)
    tmp = Path(raw_tmp)
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return tmp


def _ensure_new_outputs(paths: list[Path]) -> None:
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve(strict=False)
        if resolved in seen:
            raise InputAdapterError(f"output paths must be distinct: {path}")
        seen.add(resolved)
        if path.exists() or path.is_symlink():
            raise InputAdapterError(f"output already exists; refusing to overwrite append-only artifact: {path}")


def build_inputs(args: argparse.Namespace) -> dict[str, Any]:
    outputs = [args.out_train, args.out_ligands, args.out_targets, args.out_manifest]
    _ensure_new_outputs(outputs)
    benchmark = _prepare_benchmark(args.benchmark_train)
    retrieval_ligands = _read_retrieval_ligands(args.retrieval_ligands)
    joined, ligand_join_provenance = _join_ligands(
        benchmark,
        retrieval_ligands,
        standardization_workers=int(getattr(args, "standardization_workers", 1)),
    )
    train, conflict_meta = _collapse_train_rows(joined)
    ligands = _prepare_ligands(train, retrieval_ligands)
    targets = _prepare_targets(train, args.target_fasta, args.target_clusters)
    counts = {
        "train_rows": int(len(train)),
        "ligands": int(len(ligands)),
        "targets": int(len(targets)),
        "measured_positive": int(train["evidence_state"].eq("measured_positive").sum()),
        "measured_negative": int(train["evidence_state"].eq("measured_negative").sum()),
        "gray_unmeasured": int(train["evidence_state"].eq("gray_unmeasured").sum()),
        "direct_binding_reversible": int(
            train["ontology"].eq("direct_binding_reversible").sum()
        ),
        "functional_modulation": int(
            train["ontology"].eq("functional_modulation").sum()
        ),
        "unknown_mixed": int(train["ontology"].eq("unknown_mixed").sum()),
        "ligand_join_exact_canonical_smiles": ligand_join_provenance["counts"][
            "exact_canonical_smiles"
        ],
        "ligand_join_unique_standard_inchikey": ligand_join_provenance["counts"][
            "unique_standard_inchikey"
        ],
        "ligand_join_standardized_ligand_key": ligand_join_provenance["counts"][
            "standardized_ligand_key"
        ],
        **conflict_meta,
    }
    tmp_paths: list[Path] = []
    committed_paths: list[Path] = []
    try:
        train_tmp = _write_csv_temp(train, args.out_train)
        ligands_tmp = _write_csv_temp(ligands, args.out_ligands)
        targets_tmp = _write_csv_temp(targets, args.out_targets)
        tmp_paths.extend([train_tmp, ligands_tmp, targets_tmp])
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at": _utc_now(),
            "inputs": {
                "benchmark_train": {
                    "path": str(args.benchmark_train.resolve()),
                    "sha256": _sha256_file(args.benchmark_train),
                    "rows": int(len(benchmark)),
                },
                "retrieval_ligands": {
                    "path": str(args.retrieval_ligands.resolve()),
                    "sha256": _sha256_file(args.retrieval_ligands),
                    "rows": int(len(retrieval_ligands)),
                },
                "target_fasta": {
                    "path": str(args.target_fasta.resolve()),
                    "sha256": _sha256_file(args.target_fasta),
                    "sequences": int(len(_parse_fasta(args.target_fasta))),
                },
                "target_clusters": {
                    "path": str(args.target_clusters.resolve()),
                    "sha256": _sha256_file(args.target_clusters),
                    "rows": int(len(_read_target_clusters(args.target_clusters))),
                },
            },
            "artifacts": {
                "train": _artifact_record(train_tmp, len(train), final_path=args.out_train),
                "ligands": _artifact_record(ligands_tmp, len(ligands), final_path=args.out_ligands),
                "targets": _artifact_record(targets_tmp, len(targets), final_path=args.out_targets),
            },
            "counts": counts,
            "schema": {
                "train_csv": TRAIN_COLUMNS,
                "ligands_csv": LIGAND_COLUMNS,
                "targets_csv": TARGET_COLUMNS,
                "ligand_id": "deterministic string ligand_<12-digit ligand_index> mapped from retrieval canonical_smiles/ligand_key",
                "measured_label": (
                    f"1 for explicit measured pactivity >= {POSITIVE_THRESHOLD}, "
                    f"0 for explicit measured pactivity <= {NEGATIVE_THRESHOLD}, "
                    "blank for the open interval or collapsed conflicting "
                    "measured pair labels; activity_class is validated but never "
                    "used to override numeric thresholds"
                ),
                "ontology": (
                    "direct_binding_reversible for measured KD/KI-only pairs, "
                    "functional_modulation for measured EC50/IC50-only pairs, "
                    "otherwise unknown_mixed"
                ),
            },
            "provenance": {
                "ligand_join": ligand_join_provenance,
                "pu_policy": "explicit positives and explicit negatives only; gray/unmeasured pairs are ranking-only and never negative",
                "dedupe_policy": (
                    "exact output rows deduped; direct-binding and functional endpoints are preserved "
                    "as separate ontology rows; conflicting labels collapse only within a "
                    "ligand-target-ontology stratum"
                ),
                "sequence_policy": (
                    "out-targets is the full supplied FASTA sequence universe; "
                    "FASTA headers use first token as target_id; empty duplicate "
                    "records are ignored; sequences are whitespace-stripped uppercase strings"
                ),
                "target_cluster_policy": (
                    "every FASTA target requires uniprot,target_cluster_30,target_cluster_50 mapping; "
                    "every training target requires a FASTA sequence"
                ),
                "activity_class_mapping": {
                    "measured_positive": sorted(POSITIVE_ACTIVITY_CLASSES),
                    "measured_negative": sorted(NEGATIVE_ACTIVITY_CLASSES),
                    "gray_unmeasured": sorted(GRAY_ACTIVITY_CLASSES),
                },
            },
        }
        manifest_tmp = _write_json_temp(manifest, args.out_manifest)
        tmp_paths.append(manifest_tmp)
        replacements = [
            (train_tmp, args.out_train),
            (ligands_tmp, args.out_ligands),
            (targets_tmp, args.out_targets),
            (manifest_tmp, args.out_manifest),
        ]
        for tmp, final in replacements:
            tmp.replace(final)
            committed_paths.append(final)
        return manifest
    except BaseException:
        for final in reversed(committed_paths):
            try:
                if final.is_file() or final.is_symlink():
                    final.unlink()
            except OSError:
                pass
        for tmp in tmp_paths:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-train", required=True, type=Path)
    parser.add_argument("--retrieval-ligands", required=True, type=Path)
    parser.add_argument("--target-fasta", required=True, type=Path)
    parser.add_argument("--target-clusters", required=True, type=Path)
    parser.add_argument("--out-train", required=True, type=Path)
    parser.add_argument("--out-ligands", required=True, type=Path)
    parser.add_argument("--out-targets", required=True, type=Path)
    parser.add_argument("--out-manifest", required=True, type=Path)
    parser.add_argument("--standardization-workers", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        build_inputs(args)
    except InputAdapterError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
