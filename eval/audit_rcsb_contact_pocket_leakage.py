#!/usr/bin/env python3
"""Coordinate-level RCSB contact-pocket leakage audit.

This sidecar audit compares RCSB ligand-contact query fragments against only
train/dev prior P2Rank pocket fragments. It validates all bound manifests and
hashes before invoking Foldseek and is evaluation-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import pyarrow.parquet as pq


SCHEMA_VERSION = "skinscout.rcsb-contact-pocket-leakage-audit.v1"
RCSB_FRAGMENT_SCHEMA_VERSION = "skinscout.rcsb-contact-pocket-fragments.v1"
PRIOR_FRAGMENT_SCHEMA_VERSION = "skinscout.pocket-fragment-universe.v1"
BENCHMARK_SCHEMA_VERSION = "activity_benchmark.v1"
FOLDSEEK_FORMAT_FIELDS = [
    "query",
    "target",
    "alntmscore",
    "qtmscore",
    "ttmscore",
    "evalue",
    "bits",
    "alnlen",
    "qlen",
    "tlen",
]
TM_THRESHOLDS = (0.4, 0.5, 0.6)
REQUIRED_RCSB_INDEX_COLUMNS = {
    "pair_id",
    "entry_id",
    "component_id",
    "instance_id",
    "uniprot",
    "ligand_key",
    "fragment_path",
    "fragment_sha256",
}
REQUIRED_RCSB_EXCLUSION_COLUMNS = {
    "pair_id",
    "entry_id",
    "component_id",
    "instance_id",
    "uniprot",
    "ligand_key",
    "reason",
    "detail",
}
REQUIRED_PRIOR_INDEX_COLUMNS = {"uniprot", "fragment_path", "fragment_sha256"}
REQUIRED_ACTIVITY_COLUMNS = {"uniprot"}
ROW_FIELDS = [
    "pair_id",
    "entry_id",
    "component_id",
    "instance_id",
    "query_uniprot",
    "ligand_key",
    "query_fragment_path",
    "query_fragment_sha256",
    "fragment_extracted",
    "foldseek_covered",
    "uncovered_reason",
    "best_prior_uniprot",
    "best_target_id",
    "best_prior_fragment_path",
    "best_alntmscore",
    "best_qtmscore",
    "best_ttmscore",
    "best_max_directional_tm",
    "best_min_directional_tm",
    "best_evalue",
    "best_bits",
    "best_alnlen",
    "best_qlen",
    "best_tlen",
    "leakage_tm40",
    "leakage_tm50",
    "leakage_tm60",
]


class AuditError(RuntimeError):
    """Raised when leakage auditing cannot proceed safely."""


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


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    if root.exists():
        for path in sorted(
            p
            for p in root.rglob("*")
            if p.is_file() and p.name != ".snakemake_timestamp"
        ):
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(_sha256(path).encode("ascii"))
            digest.update(b"\0")
    return digest.hexdigest()


def _require_file(path: Path, label: str) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise AuditError(f"{label} is required and must be non-empty: {path}")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _require_file(path, label)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AuditError(f"{label} is malformed JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise AuditError(f"{label} must contain a JSON object: {path}")
    return payload


def _read_csv(path: Path, label: str, *, allow_empty: bool = False) -> pd.DataFrame:
    _require_file(path, label)
    try:
        frame = pd.read_csv(path, dtype="string").fillna("")
    except Exception as exc:
        raise AuditError(f"{label} failed to parse: {path}: {exc}") from exc
    if frame.empty and not allow_empty:
        raise AuditError(f"{label} contains no rows: {path}")
    return frame


def _require_columns(columns: Iterable[str], required: set[str], label: str) -> None:
    missing = sorted(required - set(columns))
    if missing:
        raise AuditError(f"{label} missing required column(s): {', '.join(missing)}")


def _resolve_bound_path(value: object, manifest_path: Path, label: str) -> Path:
    text = _clean(value)
    if not text:
        raise AuditError(f"{label} path is blank")
    path = Path(text)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _parquet_meta(path: Path, label: str) -> dict[str, Any]:
    _require_file(path, label)
    try:
        parquet = pq.ParquetFile(path)
    except Exception as exc:
        raise AuditError(f"{label} failed to open as parquet: {path}: {exc}") from exc
    columns = list(parquet.schema_arrow.names)
    _require_columns(columns, REQUIRED_ACTIVITY_COLUMNS, label)
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "rows": int(parquet.metadata.num_rows),
        "columns": columns,
    }


def _validate_activity_manifest(
    manifest_path: Path,
    train_parquet: Path,
    dev_parquet: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, pd.DataFrame]]:
    manifest = _read_json(manifest_path, "activity benchmark manifest")
    if manifest.get("schema_version") != BENCHMARK_SCHEMA_VERSION:
        raise AuditError(f"activity benchmark manifest schema_version must be {BENCHMARK_SCHEMA_VERSION}")
    metas = {
        "train": _parquet_meta(train_parquet, "train.parquet"),
        "dev": _parquet_meta(dev_parquet, "dev.parquet"),
    }
    output_sha = manifest.get("output_sha256")
    if not isinstance(output_sha, Mapping):
        raise AuditError("activity benchmark manifest missing output_sha256 object")
    for split, meta in metas.items():
        name = f"{split}.parquet"
        if _clean(output_sha.get(name)) != meta["sha256"]:
            raise AuditError(f"activity benchmark {name} sha256 does not match manifest")
    split_counts = manifest.get("splits", {}).get("counts") if isinstance(manifest.get("splits"), Mapping) else None
    if not isinstance(split_counts, Mapping):
        raise AuditError("activity benchmark manifest missing splits.counts object")
    for split, meta in metas.items():
        if split_counts.get(split) != meta["rows"]:
            raise AuditError(f"activity benchmark {split}.parquet row count does not match manifest")
    frames = {}
    for split, path in (("train", train_parquet), ("dev", dev_parquet)):
        columns = [
            column
            for column in ("uniprot", "split", "source_db")
            if column in metas[split]["columns"]
        ]
        frames[split] = pd.read_parquet(path, columns=columns).fillna("")
    for split, frame in frames.items():
        if "split" in frame.columns:
            bad = sorted(set(_clean(value) for value in frame["split"]) - {split})
            if bad:
                raise AuditError(f"{split}.parquet contains unexpected split values: {', '.join(bad)}")
        if any(not _clean(value) for value in frame["uniprot"]):
            raise AuditError(f"{split}.parquet contains blank uniprot values")
    return manifest, metas, frames


def _validate_fragment_file(root: Path, relative: object, expected_sha: object, label: str) -> Path:
    rel_text = _clean(relative)
    if not rel_text:
        raise AuditError(f"{label} fragment_path is blank")
    rel_path = Path(rel_text)
    if rel_path.is_absolute() or ".." in rel_path.parts:
        raise AuditError(f"{label} fragment_path must be relative and confined: {rel_text}")
    path = (root / rel_path).resolve()
    if root.resolve() not in path.parents:
        raise AuditError(f"{label} fragment_path escapes fragment directory: {rel_text}")
    _require_file(path, f"{label} fragment")
    expected = _clean(expected_sha)
    if expected != _sha256(path):
        raise AuditError(f"{label} fragment sha256 does not match index: {rel_text}")
    return path


def _validate_rcsb_fragments(
    index_path: Path,
    manifest_path: Path,
    fragment_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    frame = _read_csv(index_path, "RCSB contact-pocket index")
    _require_columns(frame.columns, REQUIRED_RCSB_INDEX_COLUMNS, "RCSB contact-pocket index")
    if frame["pair_id"].map(_clean).duplicated().any():
        raise AuditError("RCSB contact-pocket index contains duplicate pair_id values")
    manifest = _read_json(manifest_path, "RCSB contact-pocket manifest")
    if manifest.get("schema_version") != RCSB_FRAGMENT_SCHEMA_VERSION:
        raise AuditError(f"RCSB contact-pocket manifest schema_version must be {RCSB_FRAGMENT_SCHEMA_VERSION}")
    contract = manifest.get("contract")
    if not isinstance(contract, Mapping):
        raise AuditError("RCSB contact-pocket manifest missing contract object")
    for key in ("never_training", "never_calibration", "never_model_selection", "strict_pairs_only"):
        if contract.get(key) is not True:
            raise AuditError(f"RCSB contact-pocket manifest contract.{key} must be true")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise AuditError("RCSB contact-pocket manifest missing artifacts object")
    index_meta = artifacts.get("index_csv")
    exclusion_meta = artifacts.get("exclusions_csv")
    fragment_meta = artifacts.get("fragment_dir")
    if (
        not isinstance(index_meta, Mapping)
        or not isinstance(exclusion_meta, Mapping)
        or not isinstance(fragment_meta, Mapping)
    ):
        raise AuditError(
            "RCSB contact-pocket manifest missing index_csv, exclusions_csv, or fragment_dir artifact"
        )
    if _resolve_bound_path(index_meta.get("path"), manifest_path, "RCSB index") != index_path.resolve():
        raise AuditError("RCSB contact-pocket manifest index path does not match")
    if index_meta.get("sha256") != _sha256(index_path) or int(index_meta.get("rows", -1)) != len(frame):
        raise AuditError("RCSB contact-pocket manifest index hash or row count does not match")
    if _resolve_bound_path(fragment_meta.get("path"), manifest_path, "RCSB fragment_dir") != fragment_dir.resolve():
        raise AuditError("RCSB contact-pocket manifest fragment_dir path does not match")
    if fragment_meta.get("tree_sha256") != _tree_digest(fragment_dir):
        raise AuditError("RCSB contact-pocket manifest fragment tree hash does not match")
    if int(fragment_meta.get("files", -1)) != len(frame):
        raise AuditError("RCSB contact-pocket manifest fragment file count does not match index rows")
    exclusion_path = _resolve_bound_path(
        exclusion_meta.get("path"), manifest_path, "RCSB extraction exclusions"
    )
    exclusions = _read_csv(
        exclusion_path,
        "RCSB contact-pocket extraction exclusions",
        allow_empty=True,
    )
    _require_columns(
        exclusions.columns,
        REQUIRED_RCSB_EXCLUSION_COLUMNS,
        "RCSB contact-pocket extraction exclusions",
    )
    if (
        exclusion_meta.get("sha256") != _sha256(exclusion_path)
        or int(exclusion_meta.get("rows", -1)) != len(exclusions)
    ):
        raise AuditError(
            "RCSB contact-pocket extraction exclusion hash or row count does not match"
        )
    counts = manifest.get("counts")
    if not isinstance(counts, Mapping):
        raise AuditError("RCSB contact-pocket manifest missing counts object")
    try:
        strict_count = int(counts.get("strict_dual_cold_pairs", -1))
        indexed_count = int(counts.get("indexed", -1))
        excluded_count = int(counts.get("excluded", -1))
    except (TypeError, ValueError) as exc:
        raise AuditError("RCSB contact-pocket manifest counts are invalid") from exc
    if (
        indexed_count != len(frame)
        or excluded_count != len(exclusions)
        or strict_count != indexed_count + excluded_count
    ):
        raise AuditError("RCSB contact-pocket strict pair accounting is stale")
    excluded_pair_ids = exclusions["pair_id"].map(_clean)
    if excluded_pair_ids.eq("").any() or excluded_pair_ids.duplicated().any():
        raise AuditError("RCSB contact-pocket extraction exclusions require unique pair_id values")
    if set(frame["pair_id"].map(_clean)) & set(excluded_pair_ids):
        raise AuditError("RCSB indexed and extraction-excluded pair sets overlap")
    for idx, row in frame.iterrows():
        for column in ("pair_id", "entry_id", "component_id", "instance_id", "uniprot", "ligand_key"):
            if not _clean(row[column]):
                raise AuditError(f"RCSB contact-pocket index row {idx} has blank {column}")
        _validate_fragment_file(fragment_dir, row["fragment_path"], row["fragment_sha256"], f"RCSB row {idx}")
    for idx, row in exclusions.iterrows():
        for column in (
            "pair_id",
            "entry_id",
            "component_id",
            "instance_id",
            "uniprot",
            "ligand_key",
            "reason",
        ):
            if not _clean(row[column]):
                raise AuditError(
                    f"RCSB contact-pocket extraction exclusion row {idx} has blank {column}"
                )
    return frame, exclusions, manifest


def _validate_prior_fragments(
    index_path: Path,
    manifest_path: Path,
    fragment_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = _read_csv(index_path, "prior P2Rank pocket index")
    _require_columns(frame.columns, REQUIRED_PRIOR_INDEX_COLUMNS, "prior P2Rank pocket index")
    if frame["uniprot"].map(_clean).duplicated().any():
        raise AuditError("prior P2Rank pocket index contains duplicate uniprot values")
    manifest = _read_json(manifest_path, "prior P2Rank pocket manifest")
    if manifest.get("schema_version") != PRIOR_FRAGMENT_SCHEMA_VERSION:
        raise AuditError(f"prior P2Rank pocket manifest schema_version must be {PRIOR_FRAGMENT_SCHEMA_VERSION}")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise AuditError("prior P2Rank pocket manifest missing artifacts object")
    index_meta = artifacts.get("index")
    if not isinstance(index_meta, Mapping):
        raise AuditError("prior P2Rank pocket manifest missing artifacts.index")
    if _resolve_bound_path(index_meta.get("path"), manifest_path, "prior index") != index_path.resolve():
        raise AuditError("prior P2Rank pocket manifest index path does not match")
    if index_meta.get("sha256") != _sha256(index_path) or int(index_meta.get("rows", -1)) != len(frame):
        raise AuditError("prior P2Rank pocket manifest index hash or row count does not match")
    out_dir = _resolve_bound_path(artifacts.get("out_dir"), manifest_path, "prior fragment tree")
    if out_dir != fragment_dir.resolve():
        raise AuditError("prior P2Rank pocket manifest fragment tree path does not match")
    if artifacts.get("out_dir_tree_sha256") != _tree_digest(fragment_dir):
        raise AuditError("prior P2Rank pocket manifest fragment tree hash does not match")
    if int(manifest.get("counts", {}).get("accepted", -1)) != len(frame):
        raise AuditError("prior P2Rank pocket manifest accepted count does not match index rows")
    for idx, row in frame.iterrows():
        if not _clean(row["uniprot"]):
            raise AuditError(f"prior P2Rank pocket index row {idx} has blank uniprot")
        _validate_fragment_file(fragment_dir, row["fragment_path"], row["fragment_sha256"], f"prior row {idx}")
    return frame, manifest


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


def _slug(value: object, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", _clean(value))
    return text.strip("._")[:80] or fallback


def _link_or_copy(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(source.resolve(), dest)
    except OSError:
        shutil.copy2(source, dest)


def _stage_foldseek_inputs(
    root: Path,
    rcsb_index: pd.DataFrame,
    rcsb_fragment_dir: Path,
    prior_index: pd.DataFrame,
    prior_fragment_dir: Path,
    prior_uniprots: set[str],
) -> tuple[Path, Path, dict[str, dict[str, str]], dict[str, dict[str, str]], pd.DataFrame]:
    query_dir = root / "queries"
    target_dir = root / "prior"
    query_map: dict[str, dict[str, str]] = {}
    target_map: dict[str, dict[str, str]] = {}

    for row_number, row in enumerate(rcsb_index.itertuples(index=False), start=1):
        pair_id = _clean(row.pair_id)
        stem = f"q__{row_number:06d}__{_slug(pair_id, 'pair')}"
        source = _validate_fragment_file(
            rcsb_fragment_dir,
            getattr(row, "fragment_path"),
            getattr(row, "fragment_sha256"),
            f"RCSB row {row_number - 1}",
        )
        _link_or_copy(source, query_dir / f"{stem}.pdb")
        query_map[stem] = {"pair_id": pair_id}

    filtered = prior_index[prior_index["uniprot"].map(lambda value: _clean(value) in prior_uniprots)].copy()
    filtered = filtered.sort_values(["uniprot", "fragment_path"], kind="mergesort").reset_index(drop=True)
    if filtered.empty:
        raise AuditError("filtered prior P2Rank fragment universe is empty for train+dev targets")
    for row_number, row in enumerate(filtered.itertuples(index=False), start=1):
        uniprot = _clean(row.uniprot)
        stem = f"t__{row_number:06d}__{_slug(uniprot, 'uniprot')}"
        source = _validate_fragment_file(
            prior_fragment_dir,
            getattr(row, "fragment_path"),
            getattr(row, "fragment_sha256"),
            f"prior filtered row {row_number - 1}",
        )
        _link_or_copy(source, target_dir / f"{stem}.pdb")
        target_map[stem] = {
            "uniprot": uniprot,
            "fragment_path": _clean(getattr(row, "fragment_path")),
            "fragment_sha256": _clean(getattr(row, "fragment_sha256")),
        }
    return query_dir, target_dir, query_map, target_map, filtered


def _foldseek_version(binary: Path) -> dict[str, Any]:
    command = [str(binary), "version"]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise AuditError(
            "Foldseek version command failed "
            f"(exit={result.returncode}): {(result.stderr or result.stdout).strip()}"
        )
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _run_foldseek(binary: Path, query_dir: Path, target_dir: Path, output_tsv: Path, tmp_dir: Path, threads: int) -> dict[str, Any]:
    command = [
        str(binary),
        "easy-search",
        str(query_dir),
        str(target_dir),
        str(output_tsv),
        str(tmp_dir),
        "--threads",
        str(threads),
        "--alignment-type",
        "1",
        "--exhaustive-search",
        "1",
        "--exact-tmscore",
        "1",
        "--tmalign-hit-order",
        "4",
        "--format-output",
        ",".join(FOLDSEEK_FORMAT_FIELDS),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise AuditError(
            "Foldseek easy-search failed "
            f"(exit={result.returncode}): {(result.stderr or result.stdout).strip()}"
        )
    if not output_tsv.exists():
        raise AuditError(f"Foldseek output was not created: {output_tsv}")
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _resolve_foldseek_id(raw: str, mapping: Mapping[str, dict[str, str]], label: str) -> tuple[str, dict[str, str]]:
    text = _clean(raw)
    if not text:
        raise AuditError(f"Foldseek {label} identifier is blank")
    candidates = [text]
    if "." in text:
        candidates.append(text.rsplit(".", 1)[0])
    base = candidates[-1]
    parts = base.split("_")
    for cut in range(len(parts) - 1, 0, -1):
        suffix = "_".join(parts[cut:])
        if re.fullmatch(r"[A-Za-z0-9](?:_[A-Za-z0-9])*", suffix):
            candidates.append("_".join(parts[:cut]))
    hits = []
    for candidate in candidates:
        if candidate in mapping and candidate not in hits:
            hits.append(candidate)
    if not hits:
        raise AuditError(f"Foldseek {label} identifier is unknown: {text}")
    if len(hits) > 1:
        raise AuditError(f"Foldseek {label} identifier is ambiguous: {text}")
    key = hits[0]
    return key, mapping[key]


def _float(value: str, column: str, line_number: int) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise AuditError(f"Foldseek output line {line_number} has invalid {column}: {value}") from exc
    if not math.isfinite(parsed):
        raise AuditError(
            f"Foldseek output line {line_number} has non-finite {column}: {value}"
        )
    return parsed


def _int(value: str, column: str, line_number: int) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise AuditError(f"Foldseek output line {line_number} has invalid {column}: {value}") from exc
    if parsed <= 0:
        raise AuditError(
            f"Foldseek output line {line_number} has non-positive {column}: {value}"
        )
    return parsed


def _parse_foldseek(
    output_tsv: Path,
    query_map: Mapping[str, dict[str, str]],
    target_map: Mapping[str, dict[str, str]],
) -> dict[str, list[dict[str, Any]]]:
    hits: dict[str, list[dict[str, Any]]] = {}
    with output_tsv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for line_number, fields in enumerate(reader, start=1):
            if not fields or all(not field.strip() for field in fields):
                continue
            if len(fields) != len(FOLDSEEK_FORMAT_FIELDS):
                raise AuditError(
                    f"Foldseek output line {line_number} has {len(fields)} fields; expected {len(FOLDSEEK_FORMAT_FIELDS)}"
                )
            query_key, query_meta = _resolve_foldseek_id(fields[0], query_map, "query")
            target_key, target_meta = _resolve_foldseek_id(fields[1], target_map, "target")
            qtmscore = _float(fields[3], "qtmscore", line_number)
            ttmscore = _float(fields[4], "ttmscore", line_number)
            alntmscore = _float(fields[2], "alntmscore", line_number)
            if alntmscore < 0.0:
                raise AuditError(
                    f"Foldseek output line {line_number} has negative alntmscore: "
                    f"{alntmscore}"
                )
            if any(score < 0.0 or score > 1.0 for score in (qtmscore, ttmscore)):
                raise AuditError(
                    f"Foldseek output line {line_number} has directional TM-score "
                    f"outside [0, 1]: qtmscore={qtmscore}, ttmscore={ttmscore}"
                )
            evalue = _float(fields[5], "evalue", line_number)
            if evalue < 0.0:
                raise AuditError(
                    f"Foldseek output line {line_number} has negative evalue"
                )
            hit = {
                "query_id": query_key,
                "pair_id": query_meta["pair_id"],
                "target_id": target_key,
                "prior_uniprot": target_meta["uniprot"],
                "prior_fragment_path": target_meta["fragment_path"],
                "alntmscore": alntmscore,
                "qtmscore": qtmscore,
                "ttmscore": ttmscore,
                "max_directional_tm": max(qtmscore, ttmscore),
                "min_directional_tm": min(qtmscore, ttmscore),
                "evalue": evalue,
                "bits": _float(fields[6], "bits", line_number),
                "alnlen": _int(fields[7], "alnlen", line_number),
                "qlen": _int(fields[8], "qlen", line_number),
                "tlen": _int(fields[9], "tlen", line_number),
            }
            hits.setdefault(query_meta["pair_id"], []).append(hit)
    return hits


def _best_hit(hits: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not hits:
        return None
    return sorted(
        hits,
        key=lambda hit: (
            hit["max_directional_tm"],
            hit["alntmscore"],
            hit["bits"],
            -hit["evalue"],
            hit["prior_uniprot"],
        ),
        reverse=True,
    )[0]


def _rate(count: int, denominator: int) -> float | None:
    return None if denominator == 0 else count / denominator


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _source_counts(frames: Mapping[str, pd.DataFrame]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for split, frame in frames.items():
        if "source_db" not in frame.columns:
            counts[split] = {"unknown": len(frame)}
            continue
        split_counts = frame["source_db"].map(lambda value: _clean(value) or "unknown").value_counts()
        counts[split] = {str(key): int(value) for key, value in split_counts.sort_index().items()}
    return counts


def _aggregate(rows: list[dict[str, object]]) -> dict[str, Any]:
    total = len(rows)
    covered = sum(1 for row in rows if row["foldseek_covered"] == "true")
    leakage = {}
    for threshold, key in ((0.4, "leakage_tm40"), (0.5, "leakage_tm50"), (0.6, "leakage_tm60")):
        count = sum(1 for row in rows if row[key] == "true")
        leakage[f"{threshold:.1f}"] = {"count": count, "rate_all_pairs": _rate(count, total), "rate_covered_pairs": _rate(count, covered)}
    by_ligand: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_ligand.setdefault(str(row["ligand_key"]), []).append(row)
    ligand_summary = {}
    for threshold, key in ((0.4, "leakage_tm40"), (0.5, "leakage_tm50"), (0.6, "leakage_tm60")):
        leaked = sum(1 for grouped in by_ligand.values() if any(row[key] == "true" for row in grouped))
        ligand_summary[f"{threshold:.1f}"] = {
            "leaked_ligand_queries": leaked,
            "leakage_rate": _rate(leaked, len(by_ligand)),
        }
    return {
        "pairs": {
            "total": total,
            "fragment_extracted": sum(
                1 for row in rows if row["fragment_extracted"] == "true"
            ),
            "fragment_extraction_excluded": sum(
                1 for row in rows if row["fragment_extracted"] == "false"
            ),
            "covered": covered,
            "uncovered": total - covered,
            "coverage_rate": _rate(covered, total),
            "leakage_by_max_directional_tm_threshold": leakage,
        },
        "ligand_queries": {
            "total": len(by_ligand),
            "covered": sum(1 for grouped in by_ligand.values() if any(row["foldseek_covered"] == "true" for row in grouped)),
            "leakage_by_max_directional_tm_threshold": ligand_summary,
        },
    }


def _audit_rows(
    rcsb_index: pd.DataFrame,
    extraction_exclusions: pd.DataFrame,
    hits_by_pair: Mapping[str, list[dict[str, Any]]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in rcsb_index.sort_values(["entry_id", "component_id", "instance_id", "pair_id"], kind="mergesort").itertuples(index=False):
        pair_id = _clean(row.pair_id)
        best = _best_hit(hits_by_pair.get(pair_id, []))
        max_tm = "" if best is None else best["max_directional_tm"]
        out: dict[str, object] = {
            "pair_id": pair_id,
            "entry_id": _clean(row.entry_id),
            "component_id": _clean(row.component_id),
            "instance_id": _clean(row.instance_id),
            "query_uniprot": _clean(row.uniprot),
            "ligand_key": _clean(row.ligand_key),
            "query_fragment_path": _clean(row.fragment_path),
            "query_fragment_sha256": _clean(row.fragment_sha256),
            "fragment_extracted": "true",
            "foldseek_covered": _bool_text(best is not None),
            "uncovered_reason": "" if best is not None else "no_foldseek_hit",
            "best_prior_uniprot": "" if best is None else best["prior_uniprot"],
            "best_target_id": "" if best is None else best["target_id"],
            "best_prior_fragment_path": "" if best is None else best["prior_fragment_path"],
            "best_alntmscore": "" if best is None else best["alntmscore"],
            "best_qtmscore": "" if best is None else best["qtmscore"],
            "best_ttmscore": "" if best is None else best["ttmscore"],
            "best_max_directional_tm": max_tm,
            "best_min_directional_tm": "" if best is None else best["min_directional_tm"],
            "best_evalue": "" if best is None else best["evalue"],
            "best_bits": "" if best is None else best["bits"],
            "best_alnlen": "" if best is None else best["alnlen"],
            "best_qlen": "" if best is None else best["qlen"],
            "best_tlen": "" if best is None else best["tlen"],
        }
        for threshold, key in ((0.4, "leakage_tm40"), (0.5, "leakage_tm50"), (0.6, "leakage_tm60")):
            out[key] = _bool_text(best is not None and float(max_tm) >= threshold)
        rows.append(out)
    for row in extraction_exclusions.itertuples(index=False):
        out = {
            "pair_id": _clean(row.pair_id),
            "entry_id": _clean(row.entry_id),
            "component_id": _clean(row.component_id),
            "instance_id": _clean(row.instance_id),
            "query_uniprot": _clean(row.uniprot),
            "ligand_key": _clean(row.ligand_key),
            "query_fragment_path": "",
            "query_fragment_sha256": "",
            "fragment_extracted": "false",
            "foldseek_covered": "false",
            "uncovered_reason": f"fragment_extraction:{_clean(row.reason)}",
            "best_prior_uniprot": "",
            "best_target_id": "",
            "best_prior_fragment_path": "",
            "best_alntmscore": "",
            "best_qtmscore": "",
            "best_ttmscore": "",
            "best_max_directional_tm": "",
            "best_min_directional_tm": "",
            "best_evalue": "",
            "best_bits": "",
            "best_alnlen": "",
            "best_qlen": "",
            "best_tlen": "",
            "leakage_tm40": "false",
            "leakage_tm50": "false",
            "leakage_tm60": "false",
        }
        rows.append(out)
    return sorted(
        rows,
        key=lambda row: (
            str(row["entry_id"]),
            str(row["component_id"]),
            str(row["instance_id"]),
            str(row["pair_id"]),
        ),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.threads < 1:
        raise AuditError("--threads must be >= 1")
    _clean_outputs(args.out_csv, args.out_manifest)
    try:
        _require_file(args.foldseek, "Foldseek binary")
        if not os.access(args.foldseek, os.X_OK):
            raise AuditError(f"Foldseek binary is not executable: {args.foldseek}")
        activity_manifest, activity_meta, activity_frames = _validate_activity_manifest(
            args.activity_benchmark_manifest,
            args.train_parquet,
            args.dev_parquet,
        )
        rcsb_index, rcsb_exclusions, rcsb_manifest = _validate_rcsb_fragments(
            args.rcsb_index,
            args.rcsb_manifest,
            args.rcsb_fragment_dir,
        )
        prior_index, prior_manifest = _validate_prior_fragments(
            args.prior_index,
            args.prior_manifest,
            args.prior_fragment_dir,
        )
        prior_uniprots = set(
            _clean(value)
            for frame in activity_frames.values()
            for value in frame["uniprot"]
            if _clean(value)
        )
        with tempfile.TemporaryDirectory(prefix="skinscout-rcsb-pocket-audit-") as tmp:
            tmp_root = Path(tmp)
            query_dir, target_dir, query_map, target_map, filtered_prior = _stage_foldseek_inputs(
                tmp_root,
                rcsb_index,
                args.rcsb_fragment_dir,
                prior_index,
                args.prior_fragment_dir,
                prior_uniprots,
            )
            version = _foldseek_version(args.foldseek)
            search = _run_foldseek(
                args.foldseek,
                query_dir,
                target_dir,
                tmp_root / "foldseek.tsv",
                tmp_root / "foldseek_tmp",
                args.threads,
            )
            hits = _parse_foldseek(tmp_root / "foldseek.tsv", query_map, target_map)

        unknown_hit_pairs = sorted(set(hits) - set(rcsb_index["pair_id"].map(_clean)))
        if unknown_hit_pairs:
            raise AuditError(f"Foldseek hits mapped to unknown pair_id values: {unknown_hit_pairs[:5]}")
        rows = _audit_rows(rcsb_index, rcsb_exclusions, hits)
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "contract": {
                "purpose": "coordinate-level RCSB contact-pocket leakage audit",
                "evaluation_only": True,
                "never_training": True,
                "never_calibration": True,
                "never_model_selection": True,
                "missing_foldseek_hits_are_uncovered_not_cold": True,
                "fragment_extraction_failures_are_uncovered_not_cold": True,
                "prior_universe_policy": "prior P2Rank fragments are filtered to unique UniProt targets actually present in current train+dev parquets",
                "leakage_metric": "max(qtmscore, ttmscore) from Foldseek alignment-type 1 exact TM-score output",
                "tm_score_validation": "qtmscore and ttmscore must be finite in [0,1]; alntmscore is retained as a finite non-negative alignment-normalized diagnostic and is not used for leakage thresholds",
            },
            "thresholds": {
                "max_directional_tm": list(TM_THRESHOLDS),
                "columns": {"0.4": "leakage_tm40", "0.5": "leakage_tm50", "0.6": "leakage_tm60"},
            },
            "foldseek": {"version": version, "easy_search": search},
            "inputs": {
                "activity_benchmark_manifest": {
                    "path": str(args.activity_benchmark_manifest.resolve()),
                    "sha256": _sha256(args.activity_benchmark_manifest),
                    "schema_version": activity_manifest["schema_version"],
                },
                "activity_parquets": {
                    split: {key: meta[key] for key in ("path", "sha256", "rows")}
                    for split, meta in activity_meta.items()
                },
                "rcsb_contact_fragments": {
                    "index": {
                        "path": str(args.rcsb_index.resolve()),
                        "sha256": _sha256(args.rcsb_index),
                        "rows": len(rcsb_index),
                    },
                    "exclusions": {
                        "path": str(
                            _resolve_bound_path(
                                rcsb_manifest["artifacts"]["exclusions_csv"]["path"],
                                args.rcsb_manifest,
                                "RCSB extraction exclusions",
                            )
                        ),
                        "sha256": rcsb_manifest["artifacts"]["exclusions_csv"][
                            "sha256"
                        ],
                        "rows": len(rcsb_exclusions),
                    },
                    "manifest": {
                        "path": str(args.rcsb_manifest.resolve()),
                        "sha256": _sha256(args.rcsb_manifest),
                        "schema_version": rcsb_manifest["schema_version"],
                    },
                    "fragment_dir": {
                        "path": str(args.rcsb_fragment_dir.resolve()),
                        "tree_sha256": _tree_digest(args.rcsb_fragment_dir),
                        "files": len(rcsb_index),
                    },
                },
                "prior_p2rank_fragments": {
                    "index": {
                        "path": str(args.prior_index.resolve()),
                        "sha256": _sha256(args.prior_index),
                        "rows": len(prior_index),
                    },
                    "manifest": {
                        "path": str(args.prior_manifest.resolve()),
                        "sha256": _sha256(args.prior_manifest),
                        "schema_version": prior_manifest["schema_version"],
                    },
                    "fragment_dir": {
                        "path": str(args.prior_fragment_dir.resolve()),
                        "tree_sha256": _tree_digest(args.prior_fragment_dir),
                        "files": len(prior_index),
                    },
                },
            },
            "prior_universe": {
                "train_dev_unique_uniprots": len(prior_uniprots),
                "prior_index_rows_total": len(prior_index),
                "filtered_prior_index_rows": len(filtered_prior),
                "filtered_prior_uniprots": int(filtered_prior["uniprot"].map(_clean).nunique()),
                "missing_train_dev_uniprots_from_prior_index": sorted(prior_uniprots - set(filtered_prior["uniprot"].map(_clean))),
            },
            "source_counts": {
                "activity_train_dev": _source_counts(activity_frames),
                "rcsb_query_entries": {
                    str(key): int(value)
                    for key, value in pd.Series(
                        [row["entry_id"] for row in rows], dtype="string"
                    ).value_counts().sort_index().items()
                },
            },
            "summary": _aggregate(rows),
            "outputs": {
                "detailed_csv": {"path": str(args.out_csv.resolve()), "rows": len(rows)},
                "manifest": {"path": str(args.out_manifest.resolve())},
            },
        }
        _write_csv_atomic(args.out_csv, rows)
        payload["outputs"]["detailed_csv"]["sha256"] = _sha256(args.out_csv)
        _write_json_atomic(args.out_manifest, payload)
        print(
            "[rcsb-contact-pocket-leakage-audit] wrote "
            f"rows={len(rows)} manifest={args.out_manifest} csv={args.out_csv}"
        )
        return payload
    except BaseException:
        _clean_outputs(args.out_csv, args.out_manifest)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foldseek", required=True, type=Path)
    parser.add_argument("--threads", required=True, type=int)
    parser.add_argument("--activity-benchmark-manifest", required=True, type=Path)
    parser.add_argument("--train-parquet", required=True, type=Path)
    parser.add_argument("--dev-parquet", required=True, type=Path)
    parser.add_argument("--rcsb-index", required=True, type=Path)
    parser.add_argument("--rcsb-manifest", required=True, type=Path)
    parser.add_argument("--rcsb-fragment-dir", required=True, type=Path)
    parser.add_argument("--prior-index", default=Path("data/pocket_fragments_202608_index.csv"), type=Path)
    parser.add_argument("--prior-manifest", default=Path("data/pocket_fragments_202608_manifest.json"), type=Path)
    parser.add_argument("--prior-fragment-dir", default=Path("data/pocket_fragments_202608"), type=Path)
    parser.add_argument("--out-csv", default=Path("results/audits/rcsb_contact_pocket_leakage_rows.csv"), type=Path)
    parser.add_argument("--out-manifest", default=Path("results/audits/rcsb_contact_pocket_leakage.manifest.json"), type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        run(parse_args(argv))
    except AuditError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
