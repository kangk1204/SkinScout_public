#!/usr/bin/env python3
"""Build a fail-closed Discovery exact-alias map from redistributable sources."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import re
import sqlite3
import sys
import tempfile
import unicodedata
from collections.abc import Iterator
from datetime import datetime
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.discovery_canonical import (  # noqa: E402
    CANONICAL_PIPELINE,
    RDKIT_VERSION_CONTRACT,
    current_rdkit_version,
    discovery_key,
    validate_rdkit_version_contract,
)


SCHEMA_VERSION = "discovery_alias_map.v1"
REGISTRY_SCHEMA_VERSION = "discovery_alias_source_registry.v1"
SOURCE_MANIFEST_SCHEMA_VERSION = "skinscout.discovery-alias-sources.v1"
SOURCE_BUILDER = ROOT / "scripts" / "build_discovery_alias_sources.py"
REQUIRED_SOURCES = ("ChEMBL", "PubChem", "BindingDB", "GtoPdb")
ALLOWED_FORMATS = {"csv", "tsv", "parquet"}

ALIAS_COLUMNS = [
    "alias",
    "raw_alias",
    "discovery_key_sha256",
    "parent_canonical_smiles",
    "parent_inchikey",
    "parent_connectivity_inchikey",
    "input_smiles",
    "source",
    "release",
    "source_record_id",
    "source_inchikey",
]
ALIAS_SCHEMA = pa.schema([(name, pa.string()) for name in ALIAS_COLUMNS])
STREAM_BATCH_SIZE = 10_000
CANONICALIZATION_BATCH_SIZE = 4_096
CANONICALIZATION_POOL_CHUNK_SIZE = 16
MAX_CANONICALIZATION_WORKERS = 64
AMBIGUOUS_ALIAS_SAMPLE_LIMIT = 100
CANONICALIZATION_AUDIT_FILENAME = "canonicalization_exclusions.jsonl"
MAX_CANONICALIZATION_EXCLUSION_PPM = 10_000


class ContractError(ValueError):
    """Raised when an input or output violates the alias-map contract."""


CanonicalValues = tuple[str, str, str, str, str]


def _available_cpu_count() -> int:
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    return max(1, os.cpu_count() or 1)


def _default_canonicalization_workers() -> int:
    return min(16, _available_cpu_count())


def _validated_canonicalization_workers(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError("canonicalization workers must be a positive integer")
    if value > MAX_CANONICALIZATION_WORKERS:
        raise ContractError(
            "canonicalization workers exceed the supported limit "
            f"{MAX_CANONICALIZATION_WORKERS}: {value}"
        )
    return value


def _initialize_canonicalization_worker() -> None:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _relative_path(path: Path, base: Path) -> str:
    return os.path.relpath(path.resolve(), start=base.resolve())


def _require_package_path(path: Path, package_root: Path, *, label: str) -> Path:
    resolved_root = package_root.resolve()
    resolved = path.resolve()
    if resolved == resolved_root or resolved_root not in resolved.parents:
        raise ContractError(f"{label} must remain within the alias package root")
    return resolved


def _resolve_manifest_path(manifest_path: Path, value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{label} must be a nonblank relative path")
    path = Path(value)
    if path.is_absolute():
        raise ContractError(f"{label} must be relative for relocatable data packs")
    return _require_package_path(
        manifest_path.parent / path,
        manifest_path.parent,
        label=label,
    )


def _source_provenance_root(alias_package_root: Path) -> Path:
    root = alias_package_root.resolve()
    if root.name == "discovery_aliases":
        return root.parent
    return root


def _require_provenance_path(path: Path, root: Path, *, label: str) -> Path:
    resolved_root = root.resolve()
    resolved = path.resolve()
    if resolved == resolved_root or resolved_root not in resolved.parents:
        raise ContractError(f"{label} must remain within the Stage 0 provenance root")
    return resolved


def _portable_sources(sources: list[dict[str, Any]], base: Path) -> list[dict[str, Any]]:
    return [
        {
            **source,
            "artifact": {
                **source["artifact"],
                "path": _relative_path(Path(source["artifact"]["path"]), base),
            },
        }
        for source in sources
    ]


def normalize_alias(value: object) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.casefold()


def _require_nonblank_string(mapping: dict[str, Any], key: str, *, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{label}.{key} must be a nonblank string")
    return value.strip()


def _validate_iso_date(value: str, *, label: str) -> str:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise ContractError(f"{label} must be an ISO date YYYY-MM-DD") from exc
    return parsed.strftime("%Y-%m-%d")


def _validate_https_url(value: str, *, label: str) -> str:
    if not value.startswith("https://"):
        raise ContractError(f"{label} must be an https URL")
    return value


def load_registry(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ContractError(f"source registry is missing, empty, or unsafe: {path}")
    try:
        registry = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContractError(f"source registry is not valid JSON: {path}") from exc
    if not isinstance(registry, dict):
        raise ContractError("source registry must be a JSON object")
    if registry.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise ContractError(
            f"source registry schema_version must be {REGISTRY_SCHEMA_VERSION!r}"
        )
    sources = registry.get("sources")
    if not isinstance(sources, list):
        raise ContractError("source registry must contain a sources list")
    return registry


def validate_source_registry(registry: dict[str, Any], registry_path: Path) -> list[dict[str, Any]]:
    registry_root_value = registry.get("registry_root", ".")
    if not isinstance(registry_root_value, str) or not registry_root_value.strip():
        raise ContractError("registry_root must be a nonblank relative path")
    registry_root_path = Path(registry_root_value)
    if registry_root_path.is_absolute():
        raise ContractError("registry_root must be relative")
    registry_root = (registry_path.parent / registry_root_path).resolve()
    if registry_root != registry_path.parent.resolve() and registry_path.parent.resolve() not in registry_root.parents:
        raise ContractError("registry_root must remain within the registry directory")

    seen: set[str] = set()
    normalized_sources: list[dict[str, Any]] = []
    for i, source in enumerate(registry["sources"]):
        label = f"sources[{i}]"
        if not isinstance(source, dict):
            raise ContractError(f"{label} must be an object")
        name = _require_nonblank_string(source, "canonical_name", label=label)
        if name not in REQUIRED_SOURCES:
            raise ContractError(f"{label}.canonical_name is not one of {REQUIRED_SOURCES}")
        if name in seen:
            raise ContractError(f"duplicate source in registry: {name}")
        seen.add(name)
        release = _require_nonblank_string(source, "release", label=label)
        release_date = _validate_iso_date(
            _require_nonblank_string(source, "release_date", label=label),
            label=f"{label}.release_date",
        )
        spdx_license = _require_nonblank_string(source, "spdx_license", label=label)
        if not re.fullmatch(r"[A-Za-z0-9.+-]+", spdx_license):
            raise ContractError(f"{label}.spdx_license is not an SPDX identifier")
        license_url = _validate_https_url(
            _require_nonblank_string(source, "license_url", label=label),
            label=f"{label}.license_url",
        )
        if source.get("redistribution") != "allowed":
            raise ContractError(f"{label}.redistribution must be 'allowed'")
        artifact = source.get("artifact")
        if not isinstance(artifact, dict):
            raise ContractError(f"{label}.artifact must be an object")
        artifact_path_text = _require_nonblank_string(artifact, "path", label=f"{label}.artifact")
        artifact_relative = Path(artifact_path_text)
        if artifact_relative.is_absolute():
            raise ContractError(f"{label}.artifact.path must be relative")
        artifact_path = (registry_root / artifact_relative).resolve()
        if registry_root not in artifact_path.parents:
            raise ContractError(f"{label}.artifact.path escapes registry_root")
        artifact_format = _require_nonblank_string(artifact, "format", label=f"{label}.artifact")
        if artifact_format not in ALLOWED_FORMATS:
            raise ContractError(f"{label}.artifact.format must be one of {sorted(ALLOWED_FORMATS)}")
        expected_hash = _require_nonblank_string(artifact, "sha256", label=f"{label}.artifact")
        if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise ContractError(f"{label}.artifact.sha256 must be lowercase SHA-256 hex")
        expected_bytes = artifact.get("bytes")
        if not isinstance(expected_bytes, int) or expected_bytes < 0:
            raise ContractError(f"{label}.artifact.bytes must be a nonnegative integer")
        columns = source.get("columns")
        if not isinstance(columns, dict):
            raise ContractError(f"{label}.columns must be an object")
        smiles_col = _require_nonblank_string(columns, "smiles", label=f"{label}.columns")
        alias_cols = columns.get("aliases")
        if (
            not isinstance(alias_cols, list)
            or not alias_cols
            or any(not isinstance(col, str) or not col.strip() for col in alias_cols)
        ):
            raise ContractError(f"{label}.columns.aliases must contain at least one column")
        inchikey_col = columns.get("inchikey")
        source_record_id_col = columns.get("source_record_id")
        for optional_name, optional_value in {
            "inchikey": inchikey_col,
            "source_record_id": source_record_id_col,
        }.items():
            if optional_value is not None and (
                not isinstance(optional_value, str) or not optional_value.strip()
            ):
                raise ContractError(f"{label}.columns.{optional_name} must be a nonblank string")
        if artifact_path.is_symlink() or not artifact_path.is_file():
            raise ContractError(f"{label}.artifact.path is missing or unsafe: {artifact_path}")
        observed_bytes = artifact_path.stat().st_size
        if observed_bytes != expected_bytes:
            raise ContractError(
                f"{label}.artifact.bytes drift for {artifact_path}: "
                f"{observed_bytes} != {expected_bytes}"
            )
        observed_hash = sha256_file(artifact_path)
        if observed_hash != expected_hash:
            raise ContractError(
                f"{label}.artifact.sha256 drift for {artifact_path}: "
                f"{observed_hash} != {expected_hash}"
            )
        normalized_alias_cols = [col.strip() for col in alias_cols]
        if len(set(normalized_alias_cols)) != len(normalized_alias_cols):
            raise ContractError(f"{label}.columns.aliases must be unique")
        normalized_sources.append(
            {
                "canonical_name": name,
                "release": release,
                "release_date": release_date,
                "spdx_license": spdx_license,
                "license_url": license_url,
                "redistribution": "allowed",
                "artifact": {
                    "path": str(artifact_path),
                    "format": artifact_format,
                    "sha256": observed_hash,
                    "bytes": observed_bytes,
                },
                "columns": {
                    "smiles": smiles_col,
                    "aliases": normalized_alias_cols,
                    "inchikey": inchikey_col.strip() if isinstance(inchikey_col, str) else None,
                    "source_record_id": (
                        source_record_id_col.strip()
                        if isinstance(source_record_id_col, str)
                        else None
                    ),
                },
            }
        )
    missing = sorted(set(REQUIRED_SOURCES) - seen)
    extra = sorted(seen - set(REQUIRED_SOURCES))
    if missing or extra or len(seen) != len(REQUIRED_SOURCES):
        raise ContractError(
            f"registry must contain exactly {REQUIRED_SOURCES}; missing={missing}, extra={extra}"
        )
    return sorted(normalized_sources, key=lambda item: item["canonical_name"])


def validate_source_manifest_binding(
    source_manifest_path: Path,
    source_registry_path: Path,
    sources: list[dict[str, Any]],
    *,
    provenance_root: Path | None = None,
) -> dict[str, Any]:
    if (
        source_manifest_path.is_symlink()
        or not source_manifest_path.is_file()
        or source_manifest_path.stat().st_size <= 0
    ):
        raise ContractError(
            f"source manifest is missing, empty, or unsafe: {source_manifest_path}"
        )
    try:
        manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContractError("source manifest is not valid JSON") from exc
    if not isinstance(manifest, dict):
        raise ContractError("source manifest must be a JSON object")
    if manifest.get("schema_version") != SOURCE_MANIFEST_SCHEMA_VERSION:
        raise ContractError("source manifest schema_version mismatch")
    unsigned = dict(manifest)
    unsigned.pop("self_binding_sha256", None)
    if manifest.get("self_binding_sha256") != sha256_text(stable_json(unsigned)):
        raise ContractError("source manifest self binding mismatch")

    registry_path = _resolve_manifest_path(
        source_manifest_path,
        manifest.get("source_registry"),
        label="source manifest source_registry",
    )
    if registry_path.resolve() != source_registry_path.resolve():
        raise ContractError("source manifest does not bind the active source registry")
    if manifest.get("source_registry_sha256") != sha256_file(source_registry_path):
        raise ContractError("source manifest registry hash drift")

    builder = manifest.get("builder")
    if not isinstance(builder, dict):
        raise ContractError("source manifest builder record is missing")
    builder_path_value = _require_nonblank_string(
        builder,
        "path",
        label="source manifest builder",
    )
    builder_path = Path(builder_path_value)
    if builder_path.is_absolute() or (ROOT / builder_path).resolve() != SOURCE_BUILDER.resolve():
        raise ContractError("source manifest builder path mismatch")
    if builder.get("sha256") != sha256_file(SOURCE_BUILDER):
        raise ContractError("source manifest builder script hash drift")

    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != set(REQUIRED_SOURCES):
        raise ContractError("source manifest inputs must bind exactly four sources")
    for source_name in REQUIRED_SOURCES:
        record = inputs.get(source_name)
        if not isinstance(record, dict):
            raise ContractError(f"source manifest {source_name} input record is missing")
        for item, label in (
            (record, f"source manifest {source_name} input"),
            (record.get("manifest"), f"source manifest {source_name} upstream manifest"),
        ):
            if not isinstance(item, dict):
                raise ContractError(f"{label} record is missing")
            path_value = _require_nonblank_string(item, "path", label=label)
            relative_path = Path(path_value)
            if relative_path.is_absolute():
                raise ContractError(f"{label}.path must be relative")
            candidate_path = source_manifest_path.parent / relative_path
            if candidate_path.is_symlink():
                raise ContractError(f"{label} is missing or unsafe: {candidate_path}")
            active_path = _require_provenance_path(
                candidate_path,
                provenance_root or source_manifest_path.parent,
                label=label,
            )
            if not active_path.is_file():
                raise ContractError(f"{label} is missing or unsafe: {active_path}")
            expected_hash = _require_nonblank_string(item, "sha256", label=label)
            if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
                raise ContractError(f"{label}.sha256 must be lowercase SHA-256 hex")
            expected_bytes = item.get("bytes")
            if (
                isinstance(expected_bytes, bool)
                or not isinstance(expected_bytes, int)
                or expected_bytes <= 0
            ):
                raise ContractError(f"{label}.bytes must be a positive integer")
            if active_path.stat().st_size != expected_bytes:
                raise ContractError(f"{label} byte-count drift")
            if sha256_file(active_path) != expected_hash:
                raise ContractError(f"{label} SHA-256 drift")

    expected_hashes = {
        Path(source["artifact"]["path"]).name: source["artifact"]["sha256"]
        for source in sources
    }
    expected_bytes = {
        Path(source["artifact"]["path"]).name: source["artifact"]["bytes"]
        for source in sources
    }
    if manifest.get("output_sha256") != expected_hashes:
        raise ContractError("source manifest output SHA-256 bindings mismatch")
    if manifest.get("output_bytes") != expected_bytes:
        raise ContractError("source manifest output byte-count bindings mismatch")
    return manifest


def _validate_source_manifest_row_counts(
    source_manifest: dict[str, Any],
    sources: list[dict[str, Any]],
    counts: dict[str, Any],
) -> None:
    output_rows = source_manifest.get("output_rows")
    if not isinstance(output_rows, dict):
        raise ContractError("source manifest output_rows record is missing")
    expected = {
        Path(source["artifact"]["path"]).name: counts["source_counts"][
            source["canonical_name"]
        ]["input_rows"]
        for source in sources
    }
    if output_rows != expected:
        raise ContractError("source manifest output row-count bindings mismatch")


def _declared_columns(source: dict[str, Any]) -> list[str]:
    columns = source["columns"]
    declared = [columns["smiles"], *columns["aliases"]]
    for optional in (columns.get("inchikey"), columns.get("source_record_id")):
        if optional:
            declared.append(optional)
    return sorted(set(declared))


def iter_source_rows(source: dict[str, Any]) -> Iterator[tuple[int, dict[str, str]]]:
    """Yield one normalized source row at a time without materializing the artifact."""

    path = Path(source["artifact"]["path"])
    columns = _declared_columns(source)
    fmt = source["artifact"]["format"]
    row_index = 0
    try:
        if fmt == "parquet":
            parquet = pq.ParquetFile(path)
            missing = sorted(set(columns) - set(parquet.schema_arrow.names))
            if missing:
                raise ContractError(
                    f"{source['canonical_name']} artifact missing columns: {missing}"
                )
            if parquet.metadata.num_rows <= 0:
                raise ContractError(
                    f"{source['canonical_name']} artifact contains no rows"
                )
            for batch in parquet.iter_batches(
                batch_size=STREAM_BATCH_SIZE,
                columns=columns,
                use_threads=False,
            ):
                values = batch.to_pydict()
                for offset in range(batch.num_rows):
                    yield row_index, {
                        column: ""
                        if values[column][offset] is None
                        else str(values[column][offset])
                        for column in columns
                    }
                    row_index += 1
            return

        delimiter = "\t" if fmt == "tsv" else ","
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
            missing = sorted(set(columns) - set(reader.fieldnames or []))
            if missing:
                raise ContractError(
                    f"{source['canonical_name']} artifact missing columns: {missing}"
                )
            for row in reader:
                yield row_index, {
                    column: "" if row.get(column) is None else str(row[column])
                    for column in columns
                }
                row_index += 1
        if row_index <= 0:
            raise ContractError(f"{source['canonical_name']} artifact contains no rows")
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError(
            f"failed to read {source['canonical_name']} artifact: {path}"
        ) from exc


def _new_stage_path(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        prefix=".discovery-alias-map-",
        suffix=".sqlite",
        dir=directory,
    )
    os.close(descriptor)
    path = Path(raw_path)
    path.unlink()
    return path


def _open_stage(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    con.execute("PRAGMA temp_store=FILE")
    con.execute("PRAGMA cache_size=-65536")
    con.executescript(
        """
        CREATE TABLE smiles_cache (
            raw_smiles TEXT PRIMARY KEY,
            discovery_key_sha256 TEXT NOT NULL,
            parent_canonical_smiles TEXT NOT NULL,
            parent_inchikey TEXT NOT NULL,
            parent_connectivity_inchikey TEXT NOT NULL,
            input_smiles TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE smiles_failures (
            raw_smiles TEXT PRIMARY KEY,
            reason_code TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE direct_keys (
            discovery_key_sha256 TEXT PRIMARY KEY,
            parent_canonical_smiles TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE candidates (
            alias TEXT NOT NULL,
            raw_alias TEXT NOT NULL,
            discovery_key_sha256 TEXT NOT NULL,
            parent_canonical_smiles TEXT NOT NULL,
            parent_inchikey TEXT NOT NULL,
            parent_connectivity_inchikey TEXT NOT NULL,
            input_smiles TEXT NOT NULL,
            source TEXT NOT NULL,
            release TEXT NOT NULL,
            source_record_id TEXT NOT NULL,
            source_inchikey TEXT NOT NULL,
            UNIQUE(source, release, alias, discovery_key_sha256)
        );
        CREATE INDEX candidates_alias_key_idx
            ON candidates(alias, discovery_key_sha256);
        CREATE TABLE canonicalization_exclusions (
            source TEXT NOT NULL,
            row_index INTEGER NOT NULL,
            source_record_id TEXT NOT NULL,
            input_smiles_sha256 TEXT NOT NULL,
            reason_code TEXT NOT NULL,
            PRIMARY KEY(source, row_index)
        ) WITHOUT ROWID;
        """
    )
    return con


def _canonicalize_smiles_worker(
    smiles: str,
) -> tuple[str, CanonicalValues | None]:
    try:
        key = discovery_key(smiles, label="Discovery alias SMILES")
    except ValueError:
        return smiles, None
    return smiles, (
        key.discovery_key_sha256,
        key.parent_canonical_smiles,
        key.parent_inchikey,
        key.parent_connectivity_inchikey,
        key.input_smiles,
    )


def _row_batches(
    rows: Iterator[tuple[int, dict[str, str]]],
) -> Iterator[list[tuple[int, dict[str, str]]]]:
    batch: list[tuple[int, dict[str, str]]] = []
    for item in rows:
        batch.append(item)
        if len(batch) == CANONICALIZATION_BATCH_SIZE:
            yield batch
            batch = []
    if batch:
        yield batch


def _cached_canonicalizations(
    con: sqlite3.Connection,
    smiles_values: list[str],
) -> dict[str, CanonicalValues | None]:
    resolved: dict[str, CanonicalValues | None] = {}
    for offset in range(0, len(smiles_values), 500):
        chunk = smiles_values[offset:offset + 500]
        placeholders = ", ".join("?" for _ in chunk)
        for row in con.execute(
            "SELECT raw_smiles, discovery_key_sha256, parent_canonical_smiles, "
            "parent_inchikey, parent_connectivity_inchikey, input_smiles "
            f"FROM smiles_cache WHERE raw_smiles IN ({placeholders})",
            chunk,
        ):
            resolved[str(row[0])] = tuple(
                str(value) for value in row[1:]
            )  # type: ignore[assignment]
        for raw_smiles, _reason_code in con.execute(
            f"SELECT raw_smiles, reason_code FROM smiles_failures "
            f"WHERE raw_smiles IN ({placeholders})",
            chunk,
        ):
            resolved[str(raw_smiles)] = None
    return resolved


def _canonicalize_batch(
    con: sqlite3.Connection,
    smiles_values: list[str],
    pool: Pool | None,
) -> dict[str, CanonicalValues | None]:
    unique_smiles = list(dict.fromkeys(smiles_values))
    resolved = _cached_canonicalizations(con, unique_smiles)
    uncached = [smiles for smiles in unique_smiles if smiles not in resolved]
    if not uncached:
        return resolved

    if pool is None:
        results = map(_canonicalize_smiles_worker, uncached)
    else:
        results = pool.imap(
            _canonicalize_smiles_worker,
            uncached,
            chunksize=CANONICALIZATION_POOL_CHUNK_SIZE,
        )
    successful: list[tuple[str, ...]] = []
    failed: list[tuple[str, str]] = []
    for raw_smiles, canonical in results:
        resolved[raw_smiles] = canonical
        if canonical is None:
            failed.append((raw_smiles, "rdkit_canonicalization_failed"))
        else:
            successful.append((raw_smiles, *canonical))
    con.executemany(
        "INSERT INTO smiles_cache VALUES (?, ?, ?, ?, ?, ?)",
        successful,
    )
    con.executemany(
        "INSERT INTO smiles_failures VALUES (?, ?)",
        failed,
    )
    return resolved


def _checkpoint_source_progress(
    con: sqlite3.Connection,
    source_name: str,
    input_rows: int,
) -> None:
    if input_rows % STREAM_BATCH_SIZE == 0:
        con.commit()
    if input_rows % 100_000 == 0:
        print(
            f"[alias-map] {source_name}: {input_rows:,} rows staged",
            file=sys.stderr,
            flush=True,
        )


def _stage_sources(
    con: sqlite3.Connection,
    sources: list[dict[str, Any]],
    *,
    pool: Pool | None = None,
) -> dict[str, dict[str, int]]:
    source_counts: dict[str, dict[str, int]] = {}
    for source in sources:
        counts = {
            "input_rows": 0,
            "canonicalized_rows": 0,
            "canonicalization_exclusions": 0,
            "canonicalization_exclusion_fraction_ppm": 0,
            "candidate_aliases": 0,
            "blank_aliases": 0,
            "duplicate_candidate_aliases": 0,
        }
        columns = source["columns"]
        for batch in _row_batches(iter_source_rows(source)):
            canonical_by_smiles = _canonicalize_batch(
                con,
                [row[columns["smiles"]].strip() for _, row in batch],
                pool,
            )
            for row_index, row in batch:
                counts["input_rows"] += 1
                smiles = row[columns["smiles"]].strip()
                canonical = canonical_by_smiles[smiles]
                if canonical is None:
                    source_record_id = (
                        row[columns["source_record_id"]].strip()
                        if columns.get("source_record_id")
                        else ""
                    )
                    con.execute(
                        "INSERT INTO canonicalization_exclusions VALUES (?, ?, ?, ?, ?)",
                        (
                            source["canonical_name"],
                            row_index,
                            source_record_id,
                            hashlib.sha256(smiles.encode("utf-8")).hexdigest(),
                            "rdkit_canonicalization_failed",
                        ),
                    )
                    counts["canonicalization_exclusions"] += 1
                    _checkpoint_source_progress(
                        con,
                        source["canonical_name"],
                        counts["input_rows"],
                    )
                    continue
                counts["canonicalized_rows"] += 1
                (
                    key_sha,
                    parent_smiles,
                    parent_inchikey,
                    connectivity_inchikey,
                    input_smiles,
                ) = canonical
                con.execute(
                    "INSERT OR IGNORE INTO direct_keys VALUES (?, ?)",
                    (key_sha, parent_smiles),
                )
                existing_parent = con.execute(
                    "SELECT parent_canonical_smiles FROM direct_keys "
                    "WHERE discovery_key_sha256 = ?",
                    (key_sha,),
                ).fetchone()
                if existing_parent != (parent_smiles,):
                    raise ContractError(
                        "canonical Discovery key maps to conflicting parent SMILES"
                    )
                source_record_id = (
                    row[columns["source_record_id"]].strip()
                    if columns.get("source_record_id")
                    else ""
                )
                source_inchikey = (
                    row[columns["inchikey"]].strip()
                    if columns.get("inchikey")
                    else ""
                )
                for alias_column in columns["aliases"]:
                    raw_alias = row[alias_column]
                    alias = normalize_alias(raw_alias)
                    if not alias:
                        counts["blank_aliases"] += 1
                        continue
                    counts["candidate_aliases"] += 1
                    cursor = con.execute(
                        f"INSERT OR IGNORE INTO candidates ({', '.join(ALIAS_COLUMNS)}) "
                        f"VALUES ({', '.join('?' for _ in ALIAS_COLUMNS)})",
                        (
                            alias,
                            raw_alias.strip(),
                            key_sha,
                            parent_smiles,
                            parent_inchikey,
                            connectivity_inchikey,
                            input_smiles,
                            source["canonical_name"],
                            source["release"],
                            source_record_id,
                            source_inchikey,
                        ),
                    )
                    if cursor.rowcount == 0:
                        counts["duplicate_candidate_aliases"] += 1
                _checkpoint_source_progress(
                    con,
                    source["canonical_name"],
                    counts["input_rows"],
                )
        counts["canonicalization_exclusion_fraction_ppm"] = (
            (
                counts["canonicalization_exclusions"] * 1_000_000
                + counts["input_rows"]
                - 1
            )
            // counts["input_rows"]
        )
        if (
            counts["canonicalization_exclusion_fraction_ppm"]
            > MAX_CANONICALIZATION_EXCLUSION_PPM
        ):
            raise ContractError(
                f"{source['canonical_name']} canonicalization exclusion fraction "
                f"{counts['canonicalization_exclusion_fraction_ppm']} ppm exceeds "
                f"policy limit {MAX_CANONICALIZATION_EXCLUSION_PPM} ppm"
            )
        if counts["canonicalized_rows"] + counts["canonicalization_exclusions"] != counts["input_rows"]:
            raise ContractError(
                f"{source['canonical_name']} canonicalization row accounting mismatch"
            )
        source_counts[source["canonical_name"]] = counts
        con.commit()
    return source_counts


def _collect_stats(
    con: sqlite3.Connection,
    source_counts: dict[str, dict[str, int]],
) -> dict[str, Any]:
    con.execute("DROP TABLE IF EXISTS ambiguous_aliases")
    con.execute(
        """
        CREATE TABLE ambiguous_aliases AS
        SELECT alias, COUNT(DISTINCT discovery_key_sha256) AS key_count
        FROM candidates
        GROUP BY alias
        HAVING COUNT(DISTINCT discovery_key_sha256) > 1
        """
    )
    con.execute(
        "CREATE UNIQUE INDEX ambiguous_aliases_alias_idx ON ambiguous_aliases(alias)"
    )
    con.commit()

    ambiguous_count = 0
    ambiguous_sample: dict[str, int] = {}
    ambiguous_digest = hashlib.sha256()
    for alias, key_count in con.execute(
        "SELECT alias, key_count FROM ambiguous_aliases ORDER BY alias"
    ):
        ambiguous_count += 1
        if len(ambiguous_sample) < AMBIGUOUS_ALIAS_SAMPLE_LIMIT:
            ambiguous_sample[str(alias)] = int(key_count)
        ambiguous_digest.update(f"{alias}\t{key_count}\n".encode("utf-8"))

    candidate_rows = int(con.execute("SELECT COUNT(*) FROM candidates").fetchone()[0])
    alias_rows = int(
        con.execute(
            """
            SELECT COUNT(*) FROM candidates c
            LEFT JOIN ambiguous_aliases a ON a.alias = c.alias
            WHERE a.alias IS NULL
            """
        ).fetchone()[0]
    )
    unique_keys = int(
        con.execute(
            """
            SELECT COUNT(DISTINCT c.discovery_key_sha256) FROM candidates c
            LEFT JOIN ambiguous_aliases a ON a.alias = c.alias
            WHERE a.alias IS NULL
            """
        ).fetchone()[0]
    )
    direct_rows = int(con.execute("SELECT COUNT(*) FROM direct_keys").fetchone()[0])
    omitted_rows = int(
        con.execute(
            """
            SELECT COUNT(*) FROM candidates c
            JOIN ambiguous_aliases a ON a.alias = c.alias
            """
        ).fetchone()[0]
    )
    if alias_rows <= 0:
        raise ContractError("Discovery alias map contains no unambiguous aliases")
    if direct_rows <= 0:
        raise ContractError("Discovery direct exact reference contains no compounds")
    return {
        "source_counts": source_counts,
        "candidate_alias_rows": candidate_rows,
        "alias_rows": alias_rows,
        "unique_discovery_keys": unique_keys,
        "direct_reference_rows": direct_rows,
        "ambiguous_alias_count": ambiguous_count,
        "ambiguous_alias_key_counts": ambiguous_sample,
        "ambiguous_alias_key_counts_truncated": ambiguous_count > len(ambiguous_sample),
        "ambiguous_alias_key_counts_sha256": ambiguous_digest.hexdigest(),
        "omitted_ambiguous_candidate_rows": omitted_rows,
    }


def _build_stage(
    sources: list[dict[str, Any]],
    stage_path: Path,
    *,
    canonicalization_workers: int = 1,
) -> tuple[sqlite3.Connection, dict[str, Any]]:
    workers = _validated_canonicalization_workers(canonicalization_workers)

    def stage(pool: Pool | None) -> tuple[sqlite3.Connection, dict[str, Any]]:
        con = _open_stage(stage_path)
        try:
            source_counts = _stage_sources(con, sources, pool=pool)
            return con, _collect_stats(con, source_counts)
        except Exception:
            con.close()
            stage_path.unlink(missing_ok=True)
            raise

    if workers == 1:
        return stage(None)
    context = mp.get_context("fork")
    with context.Pool(
        processes=workers,
        initializer=_initialize_canonicalization_worker,
    ) as pool:
        return stage(pool)


def _tmp_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def _remove_outputs(paths: list[Path]) -> None:
    for path in paths:
        for target in (path, _tmp_path(path)):
            try:
                target.unlink()
            except FileNotFoundError:
                pass


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    _atomic_write_bytes(path, (stable_json(data) + "\n").encode("utf-8"))


def _alias_select_sql() -> str:
    selected = ", ".join(f"c.{column}" for column in ALIAS_COLUMNS)
    ordering = ", ".join(f"c.{column}" for column in ALIAS_COLUMNS)
    return (
        f"SELECT {selected} FROM candidates c "
        "LEFT JOIN ambiguous_aliases a ON a.alias = c.alias "
        f"WHERE a.alias IS NULL ORDER BY {ordering}"
    )


def _atomic_write_aliases(path: Path, con: sqlite3.Connection) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    tmp.unlink(missing_ok=True)
    writer = pq.ParquetWriter(
        tmp,
        ALIAS_SCHEMA,
        compression="zstd",
        use_dictionary=False,
    )
    try:
        cursor = con.execute(_alias_select_sql())
        while batch := cursor.fetchmany(STREAM_BATCH_SIZE):
            writer.write_table(
                pa.Table.from_pylist(
                    [dict(zip(ALIAS_COLUMNS, row, strict=True)) for row in batch],
                    schema=ALIAS_SCHEMA,
                )
            )
    finally:
        writer.close()
    os.replace(tmp, path)


def _atomic_write_direct_reference(path: Path, con: sqlite3.Connection) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    tmp.unlink(missing_ok=True)
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for key, smiles in con.execute(
            """
            SELECT discovery_key_sha256, parent_canonical_smiles
            FROM direct_keys
            ORDER BY parent_canonical_smiles, discovery_key_sha256
            """
        ):
            handle.write(f"{smiles} {key}\n")
    os.replace(tmp, path)


def _canonicalization_audit_lines(con: sqlite3.Connection) -> Iterator[str]:
    for source, row_index, source_record_id, smiles_hash, reason_code in con.execute(
        """
        SELECT source, row_index, source_record_id, input_smiles_sha256, reason_code
        FROM canonicalization_exclusions
        ORDER BY source, row_index
        """
    ):
        yield stable_json({
            "input_smiles_sha256": smiles_hash,
            "reason_code": reason_code,
            "row_index": int(row_index),
            "source": source,
            "source_record_id": source_record_id,
        }) + "\n"


def _atomic_write_canonicalization_audit(
    path: Path,
    con: sqlite3.Connection,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    tmp.unlink(missing_ok=True)
    rows = 0
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for line in _canonicalization_audit_lines(con):
            handle.write(line)
            rows += 1
    os.replace(tmp, path)
    return rows


def _validate_canonicalization_audit(
    path: Path,
    con: sqlite3.Connection,
) -> int:
    rows = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        for expected in _canonicalization_audit_lines(con):
            rows += 1
            if handle.readline() != expected:
                raise ContractError(
                    "canonicalization exclusion audit does not match active source rebuild"
                )
        if handle.readline() != "":
            raise ContractError(
                "canonicalization exclusion audit does not match active source rebuild"
            )
    return rows


def _validate_alias_output(
    path: Path,
    con: sqlite3.Connection,
    *,
    expected_rows: int,
) -> None:
    try:
        parquet = pq.ParquetFile(path)
    except Exception as exc:
        raise ContractError(f"aliases parquet failed to parse: {path}") from exc
    if parquet.schema_arrow != ALIAS_SCHEMA:
        raise ContractError("aliases parquet schema mismatch")
    if parquet.metadata.num_rows != expected_rows:
        raise ContractError("aliases parquet row count does not match active source rebuild")

    expected = con.execute(_alias_select_sql())
    observed_rows = 0
    for batch in parquet.iter_batches(
        batch_size=STREAM_BATCH_SIZE,
        columns=ALIAS_COLUMNS,
        use_threads=False,
    ):
        values = batch.to_pydict()
        for offset in range(batch.num_rows):
            expected_row = expected.fetchone()
            observed_row = tuple(
                "" if values[column][offset] is None else str(values[column][offset])
                for column in ALIAS_COLUMNS
            )
            if expected_row is None or observed_row != expected_row:
                raise ContractError("aliases parquet does not match active source rebuild")
            observed_rows += 1
    if observed_rows != expected_rows or expected.fetchone() is not None:
        raise ContractError("aliases parquet does not match active source rebuild")


def _validate_direct_output(path: Path, con: sqlite3.Connection) -> None:
    expected = con.execute(
        """
        SELECT discovery_key_sha256, parent_canonical_smiles
        FROM direct_keys
        ORDER BY parent_canonical_smiles, discovery_key_sha256
        """
    )
    with path.open("r", encoding="utf-8", newline="") as handle:
        for key, smiles in expected:
            if handle.readline() != f"{smiles} {key}\n":
                raise ContractError("direct reference does not match active source rebuild")
        if handle.readline() != "":
            raise ContractError("direct reference does not match active source rebuild")


def _canonical_payload(
    *,
    registry_hash: str,
    source_manifest_hash: str,
    source_hashes: dict[str, str],
    alias_hash: str,
    direct_reference_hash: str,
    builder_hash: str,
    counts: dict[str, Any],
    canonicalization_audit: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "registry_sha256": registry_hash,
        "source_manifest_sha256": source_manifest_hash,
        "source_artifact_sha256": source_hashes,
        "outputs": {
            "aliases_parquet_sha256": alias_hash,
            "direct_reference_sha256": direct_reference_hash,
        },
        "counts": counts,
        "canonicalization_audit": canonicalization_audit,
        "policy": policy,
        "canonical_contract": {
            "rdkit_version": current_rdkit_version(),
            "rdkit_version_contract": RDKIT_VERSION_CONTRACT,
            "pipeline": list(CANONICAL_PIPELINE),
        },
        "builder_script_sha256": builder_hash,
    }


def build(
    *,
    source_registry: Path,
    source_manifest: Path,
    out_aliases_parquet: Path,
    out_direct_reference: Path,
    out_manifest: Path,
    canonicalization_workers: int = 1,
) -> dict[str, Any]:
    workers = _validated_canonicalization_workers(canonicalization_workers)
    canonicalization_audit_path = (
        out_manifest.parent / CANONICALIZATION_AUDIT_FILENAME
    )
    outputs = [
        out_aliases_parquet,
        out_direct_reference,
        canonicalization_audit_path,
        out_manifest,
    ]
    package_root = out_manifest.parent.resolve()
    _require_package_path(source_registry, package_root, label="source registry")
    _require_package_path(source_manifest, package_root, label="source manifest")
    for label, path in (
        ("aliases parquet output", out_aliases_parquet),
        ("direct-reference output", out_direct_reference),
        ("canonicalization audit output", canonicalization_audit_path),
        ("manifest output", out_manifest),
    ):
        _require_package_path(path, package_root, label=label)
    stage_path: Path | None = None
    stage_con: sqlite3.Connection | None = None
    try:
        if len({path.resolve() for path in outputs}) != len(outputs):
            raise ContractError("alias, direct-reference, and manifest outputs must be distinct")
        registry = load_registry(source_registry)
        sources = validate_source_registry(registry, source_registry.resolve())
        source_manifest_payload = validate_source_manifest_binding(
            source_manifest,
            source_registry,
            sources,
            provenance_root=_source_provenance_root(package_root),
        )
        stage_path = _new_stage_path(out_manifest.parent)
        stage_con, stats = _build_stage(
            sources,
            stage_path,
            canonicalization_workers=workers,
        )
        _validate_source_manifest_row_counts(
            source_manifest_payload,
            sources,
            stats,
        )
        _atomic_write_aliases(out_aliases_parquet, stage_con)
        _atomic_write_direct_reference(out_direct_reference, stage_con)
        canonicalization_audit_rows = _atomic_write_canonicalization_audit(
            canonicalization_audit_path,
            stage_con,
        )
        alias_hash = sha256_file(out_aliases_parquet)
        direct_reference_hash = sha256_file(out_direct_reference)
        canonicalization_audit = {
            "path": _relative_path(
                canonicalization_audit_path,
                out_manifest.parent,
            ),
            "sha256": sha256_file(canonicalization_audit_path),
            "bytes": canonicalization_audit_path.stat().st_size,
            "rows": canonicalization_audit_rows,
        }
        expected_audit_rows = sum(
            item["canonicalization_exclusions"]
            for item in stats["source_counts"].values()
        )
        if canonicalization_audit_rows != expected_audit_rows:
            raise ContractError("canonicalization exclusion audit row-count mismatch")
        policy = {
            "fail_closed": True,
            "max_canonicalization_exclusion_fraction_ppm_per_source": (
                MAX_CANONICALIZATION_EXCLUSION_PPM
            ),
        }
        source_hashes = {
            source["canonical_name"]: source["artifact"]["sha256"] for source in sources
        }
        counts = stats
        payload = _canonical_payload(
            registry_hash=sha256_file(source_registry),
            source_manifest_hash=sha256_file(source_manifest),
            source_hashes=source_hashes,
            alias_hash=alias_hash,
            direct_reference_hash=direct_reference_hash,
            builder_hash=sha256_file(Path(__file__).resolve()),
            counts=counts,
            canonicalization_audit=canonicalization_audit,
            policy=policy,
        )
        manifest = {
            **payload,
            "source_registry": _relative_path(source_registry, out_manifest.parent),
            "source_manifest": _relative_path(source_manifest, out_manifest.parent),
            "source_registry_contract": REGISTRY_SCHEMA_VERSION,
            "sources": _portable_sources(sources, out_manifest.parent),
            "output_paths": {
                "aliases_parquet": _relative_path(
                    out_aliases_parquet, out_manifest.parent
                ),
                "direct_reference": _relative_path(
                    out_direct_reference, out_manifest.parent
                ),
            },
            "output_bytes": {
                "aliases_parquet": out_aliases_parquet.stat().st_size,
                "direct_reference": out_direct_reference.stat().st_size,
            },
            "created_at_utc": "1970-01-01T00:00:00Z",
            "canonical_payload_sha256": sha256_text(stable_json(payload)),
        }
        manifest["binding_sha256"] = sha256_text(stable_json(manifest))
        _atomic_write_json(out_manifest, manifest)
        validate_manifest(
            out_manifest,
            expected_registry=source_registry,
            expected_source_manifest=source_manifest,
            canonicalization_workers=workers,
            _active_stage=stage_con,
            _active_counts=stats,
        )
        stage_con.close()
        stage_con = None
        stage_path.unlink(missing_ok=True)
        stage_path = None
        return manifest
    except Exception:
        _remove_outputs(outputs)
        raise
    finally:
        if stage_con is not None:
            stage_con.close()
        if stage_path is not None:
            stage_path.unlink(missing_ok=True)


def validate_manifest(
    manifest_path: str | Path,
    expected_registry: str | Path | None = None,
    expected_source_manifest: str | Path | None = None,
    canonicalization_workers: int = 1,
    *,
    _active_stage: sqlite3.Connection | None = None,
    _active_counts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a Discovery alias-map manifest against active files and binding."""

    workers = _validated_canonicalization_workers(canonicalization_workers)
    if (_active_stage is None) != (_active_counts is None):
        raise ContractError("active stage validation requires stage and counts together")
    manifest_path = Path(manifest_path)
    if manifest_path.is_symlink() or not manifest_path.is_file() or manifest_path.stat().st_size <= 0:
        raise ContractError(f"manifest is missing, empty, or unsafe: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ContractError(f"manifest schema_version must be {SCHEMA_VERSION!r}")
    validate_rdkit_version_contract(
        manifest.get("canonical_contract", {}).get("rdkit_version"),
        label="manifest canonical_contract.rdkit_version",
    )
    if manifest.get("canonical_contract", {}).get("rdkit_version_contract") != RDKIT_VERSION_CONTRACT:
        raise ContractError("manifest canonical_contract.rdkit_version_contract mismatch")
    if manifest.get("canonical_contract", {}).get("pipeline") != list(CANONICAL_PIPELINE):
        raise ContractError("manifest canonical_contract.pipeline mismatch")
    if sha256_file(Path(__file__).resolve()) != manifest.get("builder_script_sha256"):
        raise ContractError("builder script hash drift")

    registry_path = _resolve_manifest_path(
        manifest_path,
        manifest.get("source_registry"),
        label="manifest source_registry",
    )
    if expected_registry is not None and registry_path.resolve() != Path(expected_registry).resolve():
        raise ContractError("manifest source_registry does not match expected_registry")
    if sha256_file(registry_path) != manifest["registry_sha256"]:
        raise ContractError("source registry hash drift")
    registry = load_registry(registry_path)
    sources = validate_source_registry(registry, registry_path.resolve())
    source_manifest_path = _resolve_manifest_path(
        manifest_path,
        manifest.get("source_manifest"),
        label="manifest source_manifest",
    )
    if (
        expected_source_manifest is not None
        and source_manifest_path.resolve() != Path(expected_source_manifest).resolve()
    ):
        raise ContractError("manifest source_manifest does not match expected_source_manifest")
    if sha256_file(source_manifest_path) != manifest.get("source_manifest_sha256"):
        raise ContractError("source manifest hash drift")
    source_manifest_payload = validate_source_manifest_binding(
        source_manifest_path,
        registry_path,
        sources,
        provenance_root=_source_provenance_root(manifest_path.parent),
    )
    if manifest.get("sources") != _portable_sources(sources, manifest_path.parent):
        raise ContractError("manifest normalized source registry metadata mismatch")
    expected_source_hashes = {
        source["canonical_name"]: source["artifact"]["sha256"] for source in sources
    }
    if expected_source_hashes != manifest.get("source_artifact_sha256"):
        raise ContractError("manifest source artifact hash binding mismatch")

    policy = manifest.get("policy")
    expected_policy = {
        "fail_closed": True,
        "max_canonicalization_exclusion_fraction_ppm_per_source": (
            MAX_CANONICALIZATION_EXCLUSION_PPM
        ),
    }
    if policy != expected_policy:
        raise ContractError("manifest canonicalization exclusion policy mismatch")
    audit_record = manifest.get("canonicalization_audit")
    if not isinstance(audit_record, dict):
        raise ContractError("manifest canonicalization_audit record is missing")
    audit_path = _resolve_manifest_path(
        manifest_path,
        audit_record.get("path"),
        label="manifest canonicalization_audit",
    )
    if audit_path != (manifest_path.parent / CANONICALIZATION_AUDIT_FILENAME).resolve():
        raise ContractError("manifest canonicalization audit path mismatch")
    if audit_path.is_symlink() or not audit_path.is_file():
        raise ContractError("canonicalization exclusion audit is missing or unsafe")
    if audit_record.get("sha256") != sha256_file(audit_path):
        raise ContractError("canonicalization exclusion audit hash drift")
    if audit_record.get("bytes") != audit_path.stat().st_size:
        raise ContractError("canonicalization exclusion audit byte-count drift")
    if (
        isinstance(audit_record.get("rows"), bool)
        or not isinstance(audit_record.get("rows"), int)
        or audit_record["rows"] < 0
    ):
        raise ContractError("canonicalization exclusion audit rows must be nonnegative")

    output_paths = manifest.get("output_paths")
    if not isinstance(output_paths, dict):
        raise ContractError("manifest output_paths must be an object")
    aliases_path = _resolve_manifest_path(
        manifest_path,
        output_paths.get("aliases_parquet"),
        label="manifest aliases_parquet",
    )
    direct_path = _resolve_manifest_path(
        manifest_path,
        output_paths.get("direct_reference"),
        label="manifest direct_reference",
    )
    for label, path in (("aliases parquet", aliases_path), ("direct reference", direct_path)):
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise ContractError(f"{label} is missing, empty, or unsafe: {path}")
    alias_hash = sha256_file(aliases_path)
    direct_hash = sha256_file(direct_path)
    if alias_hash != manifest.get("outputs", {}).get("aliases_parquet_sha256"):
        raise ContractError("aliases parquet hash drift")
    if direct_hash != manifest.get("outputs", {}).get("direct_reference_sha256"):
        raise ContractError("direct reference hash drift")
    if aliases_path.stat().st_size != manifest.get("output_bytes", {}).get("aliases_parquet"):
        raise ContractError("aliases parquet byte count drift")
    if direct_path.stat().st_size != manifest.get("output_bytes", {}).get("direct_reference"):
        raise ContractError("direct reference byte count drift")

    stage_path: Path | None = None
    stage_con = _active_stage
    try:
        if stage_con is None:
            stage_path = _new_stage_path(manifest_path.parent)
            stage_con, expected_counts = _build_stage(
                sources,
                stage_path,
                canonicalization_workers=workers,
            )
        else:
            expected_counts = _active_counts
        if not isinstance(expected_counts, dict):
            raise ContractError("active stage counts are missing")
        if manifest.get("counts") != expected_counts:
            raise ContractError("manifest counts do not match active source rebuild")
        _validate_source_manifest_row_counts(
            source_manifest_payload,
            sources,
            expected_counts,
        )
        _validate_alias_output(
            aliases_path,
            stage_con,
            expected_rows=expected_counts["alias_rows"],
        )
        _validate_direct_output(direct_path, stage_con)
        observed_audit_rows = _validate_canonicalization_audit(
            audit_path,
            stage_con,
        )
        if observed_audit_rows != audit_record["rows"]:
            raise ContractError("canonicalization exclusion audit row-count drift")
    finally:
        if _active_stage is None and stage_con is not None:
            stage_con.close()
        if stage_path is not None:
            stage_path.unlink(missing_ok=True)

    payload = _canonical_payload(
        registry_hash=manifest["registry_sha256"],
        source_manifest_hash=manifest["source_manifest_sha256"],
        source_hashes=manifest["source_artifact_sha256"],
        alias_hash=alias_hash,
        direct_reference_hash=direct_hash,
        builder_hash=manifest["builder_script_sha256"],
        counts=manifest["counts"],
        canonicalization_audit=audit_record,
        policy=policy,
    )
    if sha256_text(stable_json(payload)) != manifest.get("canonical_payload_sha256"):
        raise ContractError("canonical payload hash mismatch")
    unsigned = dict(manifest)
    unsigned.pop("binding_sha256", None)
    binding = sha256_text(stable_json(unsigned))
    if binding != manifest.get("binding_sha256"):
        raise ContractError("manifest binding_sha256 mismatch")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-registry", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--out-aliases-parquet", required=True, type=Path)
    parser.add_argument("--out-direct-reference", required=True, type=Path)
    parser.add_argument("--out-manifest", required=True, type=Path)
    parser.add_argument(
        "--canonicalization-workers",
        type=int,
        default=_default_canonicalization_workers(),
        help=(
            "RDKit canonicalization worker processes; SQLite writes remain "
            "single-owner (default: up to 16 affinity-visible CPUs)"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        build(
            source_registry=args.source_registry,
            source_manifest=args.source_manifest,
            out_aliases_parquet=args.out_aliases_parquet,
            out_direct_reference=args.out_direct_reference,
            out_manifest=args.out_manifest,
            canonicalization_workers=args.canonicalization_workers,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
