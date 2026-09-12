#!/usr/bin/env python3
"""eval/leakage_check.py — Three-axis data-leakage audit.

Axes (INSTRUCTIONS.md §14.3):
    sequence: MMseqs2 max identity vs training-cutoff FASTA  → seq_id ≥ 0.30
    ligand:   Morgan-FP Tanimoto vs training ligand DB        → t ≥ 0.50
    pocket:   PLINDER SuCOS vs training holo pockets         → s ≥ 0.50

For each evaluation case (eval_target_uniprot, eval_ligand_smiles), produce a
row with the three measures + a boolean `leak_flag` (OR of the three).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import secrets
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

LOG = logging.getLogger("eval.leakage")
DISCOVERY_AUDIT_SCHEMA = "skinscout.discovery-leakage-audit.v2"
DISCOVERY_SOURCE_KEYS = (
    "input_csv",
    "training_seq_db",
    "training_ligands",
    "training_holo",
    "direct_exact_reference",
    "leakage_audit_csv",
)
CANONICALIZATION_AUDIT_FILENAME = "canonicalization_exclusions.jsonl"
CANONICALIZATION_POLICY = {
    "fail_closed": True,
    "max_canonicalization_exclusion_fraction_ppm_per_source": 10_000,
}

try:
    from discovery_canonical import (
        CANONICAL_PIPELINE,
        RDKIT_VERSION_CONTRACT,
        current_rdkit_version,
        discovery_key,
        validate_rdkit_version_contract,
    )
except ImportError:  # pragma: no cover - script import fallback
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from discovery_canonical import (
        CANONICAL_PIPELINE,
        RDKIT_VERSION_CONTRACT,
        current_rdkit_version,
        discovery_key,
        validate_rdkit_version_contract,
    )


def _write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _write_json_atomic(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_binding_sha256(payload: dict[str, object]) -> str:
    bound = dict(payload)
    bound.pop("binding_sha256", None)
    canonical = json.dumps(
        bound,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _resolve_manifest_file(
    manifest_path: Path,
    value: object,
    label: str,
    *,
    allow_empty: bool = False,
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonblank relative path")
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"{label} must be relative")
    bound = manifest_path.parent / path
    if (
        bound.is_symlink()
        or not bound.is_file()
        or (not allow_empty and bound.stat().st_size <= 0)
    ):
        raise ValueError(f"{label} is missing, empty, or unsafe: {bound}")
    resolved = bound.resolve()
    package_root = manifest_path.parent.resolve()
    if package_root not in resolved.parents:
        raise ValueError(f"{label} escapes the alias package root")
    return resolved


def _validate_canonicalization_audit(
    manifest_path: Path,
    payload: dict[str, object],
    expected_sources: tuple[str, ...],
) -> dict[str, dict[str, int]]:
    if payload.get("policy") != CANONICALIZATION_POLICY:
        raise ValueError("Discovery alias canonicalization policy mismatch")
    audit_record = payload.get("canonicalization_audit")
    if not isinstance(audit_record, dict):
        raise ValueError("Discovery alias canonicalization audit record is missing")
    audit_path = _resolve_manifest_file(
        manifest_path,
        audit_record.get("path"),
        "Discovery alias canonicalization audit",
        allow_empty=True,
    )
    expected_path = (manifest_path.parent / CANONICALIZATION_AUDIT_FILENAME).resolve()
    if audit_path != expected_path:
        raise ValueError("Discovery alias canonicalization audit path mismatch")
    if audit_record.get("sha256") != _sha256(audit_path):
        raise ValueError("Discovery alias canonicalization audit SHA-256 drift")
    if audit_record.get("bytes") != audit_path.stat().st_size:
        raise ValueError("Discovery alias canonicalization audit byte-count drift")
    declared_rows = audit_record.get("rows")
    if isinstance(declared_rows, bool) or not isinstance(declared_rows, int) or declared_rows < 0:
        raise ValueError("Discovery alias canonicalization audit rows must be nonnegative")

    counts = payload.get("counts")
    source_counts = counts.get("source_counts") if isinstance(counts, dict) else None
    if not isinstance(source_counts, dict) or set(source_counts) != set(expected_sources):
        raise ValueError("Discovery alias canonicalization source counts are incomplete")
    validated_counts: dict[str, dict[str, int]] = {}
    expected_audit_rows = 0
    for source_name in expected_sources:
        source_count = source_counts.get(source_name)
        if not isinstance(source_count, dict):
            raise ValueError(
                f"Discovery alias {source_name} canonicalization counts are missing"
            )
        values: dict[str, int] = {}
        for field in (
            "input_rows",
            "canonicalized_rows",
            "canonicalization_exclusions",
            "canonicalization_exclusion_fraction_ppm",
        ):
            value = source_count.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"Discovery alias {source_name} {field} must be nonnegative"
                )
            values[field] = value
        input_rows = values["input_rows"]
        if input_rows <= 0:
            raise ValueError(f"Discovery alias {source_name} input_rows must be positive")
        exclusions = values["canonicalization_exclusions"]
        if values["canonicalized_rows"] + exclusions != input_rows:
            raise ValueError(
                f"Discovery alias {source_name} canonicalization row accounting mismatch"
            )
        expected_fraction = (exclusions * 1_000_000 + input_rows - 1) // input_rows
        if values["canonicalization_exclusion_fraction_ppm"] != expected_fraction:
            raise ValueError(
                f"Discovery alias {source_name} canonicalization fraction mismatch"
            )
        if expected_fraction > CANONICALIZATION_POLICY[
            "max_canonicalization_exclusion_fraction_ppm_per_source"
        ]:
            raise ValueError(
                f"Discovery alias {source_name} canonicalization exclusion policy exceeded"
            )
        validated_counts[source_name] = values
        expected_audit_rows += exclusions

    observed_by_source = {source_name: 0 for source_name in expected_sources}
    observed_rows = 0
    previous: tuple[str, int] | None = None
    with audit_path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n"):
                raise ValueError(
                    f"Discovery alias canonicalization audit row {line_number} lacks newline"
                )
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Discovery alias canonicalization audit row {line_number} is invalid JSON"
                ) from exc
            if (
                not isinstance(record, dict)
                or set(record) != {
                    "input_smiles_sha256",
                    "reason_code",
                    "row_index",
                    "source",
                    "source_record_id",
                }
                or record.get("reason_code") != "rdkit_canonicalization_failed"
                or not re.fullmatch(
                    r"[0-9a-f]{64}", str(record.get("input_smiles_sha256", ""))
                )
                or record.get("source") not in observed_by_source
                or isinstance(record.get("row_index"), bool)
                or not isinstance(record.get("row_index"), int)
                or record["row_index"] < 0
                or not isinstance(record.get("source_record_id"), str)
            ):
                raise ValueError(
                    f"Discovery alias canonicalization audit row {line_number} is invalid"
                )
            canonical_line = json.dumps(
                record,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ) + "\n"
            if line != canonical_line:
                raise ValueError(
                    "Discovery alias canonicalization audit is not canonically encoded"
                )
            marker = (str(record["source"]), int(record["row_index"]))
            if previous is not None and marker <= previous:
                raise ValueError(
                    "Discovery alias canonicalization audit rows are not strictly sorted"
                )
            previous = marker
            observed_by_source[str(record["source"])] += 1
            observed_rows += 1

    if observed_rows != declared_rows or observed_rows != expected_audit_rows:
        raise ValueError("Discovery alias canonicalization audit row-count drift")
    for source_name, source_count in validated_counts.items():
        if observed_by_source[source_name] != source_count["canonicalization_exclusions"]:
            raise ValueError(
                f"Discovery alias {source_name} canonicalization audit count mismatch"
            )
    return validated_counts


def _resolve_relative_file(manifest_path: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonblank relative path")
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"{label} must be relative")
    bound = manifest_path.parent / path
    if bound.is_symlink() or not bound.is_file() or bound.stat().st_size <= 0:
        raise ValueError(f"{label} is missing, empty, or unsafe: {bound}")
    return bound.resolve()


def _load_json_object(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is missing, empty, or unsafe: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is invalid: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonblank(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonblank string")
    return value.strip()


def _upstream_source_contract(
    source_name: str,
    payload: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    if source_name == "ChEMBL":
        if payload.get("schema_version") != "chembl_activity_evidence.v1":
            raise ValueError("ChEMBL upstream manifest schema_version mismatch")
        source = payload.get("source")
        artifact = source
        hash_key, bytes_key = "source_db_sha256", "source_db_bytes"
        release, release_date = None, None
    elif source_name == "BindingDB":
        if payload.get("schema_version") != 1:
            raise ValueError("BindingDB upstream manifest schema_version mismatch")
        source = payload.get("source")
        artifact = payload.get("extracted")
        hash_key, bytes_key = "sha256", "bytes"
        release, release_date = payload.get("release"), payload.get("release_date")
    elif source_name == "GtoPdb":
        if payload.get("schema_version") != "skinscout.gtopdb-source.v1":
            raise ValueError("GtoPdb upstream manifest schema_version mismatch")
        source = payload.get("source")
        artifacts = payload.get("artifacts")
        artifact = artifacts.get("ligands.csv") if isinstance(artifacts, dict) else None
        hash_key, bytes_key = "sha256", "bytes"
        release, release_date = None, None
    elif source_name == "PubChem":
        if payload.get("schema_version") != "skinscout.pubchem-alias-source.v1":
            raise ValueError("PubChem upstream manifest schema_version mismatch")
        source = payload.get("source")
        artifact = payload.get("artifact")
        hash_key, bytes_key = "sha256", "bytes"
        release, release_date = None, None
    else:  # pragma: no cover - guarded by the exact source set check
        raise ValueError(f"Unsupported Discovery alias source: {source_name}")
    if not isinstance(source, dict) or not isinstance(artifact, dict):
        raise ValueError(f"{source_name} upstream manifest records are missing")
    normalized_source = dict(source)
    if release is not None:
        normalized_source["release"] = release
    if release_date is not None:
        normalized_source["release_date"] = release_date
    return normalized_source, {
        "sha256": artifact.get(hash_key),
        "bytes": artifact.get(bytes_key),
    }


def inspect_direct_exact_reference(
    path: Path,
    *,
    expected_sha256: object,
    expected_bytes: object,
    expected_rows: object,
    parent_canonical_smiles: str | None = None,
    discovery_key_sha256: str | None = None,
    stored_keys_out: set[str] | None = None,
) -> dict[str, object]:
    """Validate and query a direct-reference file from one stable descriptor."""

    expected_hash = _nonblank(expected_sha256, "Direct exact reference SHA-256")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise ValueError("Direct exact reference SHA-256 must be lowercase 64-hex")
    byte_count = _positive_int(expected_bytes, "Direct exact reference bytes")
    row_count = _positive_int(expected_rows, "Direct exact reference rows")
    query_supplied = parent_canonical_smiles is not None or discovery_key_sha256 is not None
    if query_supplied:
        if not parent_canonical_smiles or not re.fullmatch(
            r"[0-9a-f]{64}", str(discovery_key_sha256 or "")
        ):
            raise ValueError("Direct exact reference query binding is invalid")
        expected_query_hash = hashlib.sha256(
            parent_canonical_smiles.encode("utf-8")
        ).hexdigest()
        if expected_query_hash != discovery_key_sha256:
            raise ValueError("Direct exact reference query key is not bound to its SMILES")

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"Direct exact reference is missing or unsafe: {path}: {exc}") from exc

    digest = hashlib.sha256()
    observed_rows = 0
    previous: tuple[str, str] | None = None
    matched = False
    if stored_keys_out is not None:
        stored_keys_out.clear()
    try:
        with os.fdopen(descriptor, "rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
                raise ValueError(f"Direct exact reference is not a non-empty regular file: {path}")
            for line_number, raw_line in enumerate(handle, start=1):
                digest.update(raw_line)
                if not raw_line.endswith(b"\n"):
                    raise ValueError(
                        f"Direct exact reference row {line_number} lacks newline"
                    )
                try:
                    line = raw_line[:-1].decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ValueError(
                        f"Direct exact reference row {line_number} is not UTF-8"
                    ) from exc
                fields = line.split(" ")
                if (
                    len(fields) != 2
                    or not fields[0]
                    or not re.fullmatch(r"[0-9a-f]{64}", fields[1])
                ):
                    raise ValueError(
                        f"Direct exact reference row {line_number} must contain "
                        "exactly one SMILES and one lowercase 64-hex key"
                    )
                smiles, key = fields
                if hashlib.sha256(smiles.encode("utf-8")).hexdigest() != key:
                    raise ValueError(
                        f"Direct exact reference row {line_number} key does not bind its SMILES"
                    )
                current = (smiles, key)
                if previous is not None and current <= previous:
                    raise ValueError(
                        "Direct exact reference rows must be unique and strictly sorted"
                    )
                previous = current
                observed_rows += 1
                if stored_keys_out is not None:
                    stored_keys_out.add(key)
                if query_supplied and current == (
                    parent_canonical_smiles,
                    discovery_key_sha256,
                ):
                    matched = True
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise ValueError(f"Direct exact reference could not be read: {path}: {exc}") from exc

    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise ValueError("Direct exact reference changed while it was being validated")
    observed_hash = digest.hexdigest()
    if before.st_size != byte_count:
        raise ValueError("Direct exact reference byte-count drift")
    if observed_rows != row_count:
        raise ValueError("Direct exact reference row-count drift")
    if not secrets.compare_digest(observed_hash, expected_hash):
        raise ValueError("Direct exact reference SHA-256 drift")
    return {
        "path": str(path),
        "bytes": int(before.st_size),
        "sha256": observed_hash,
        "rows": observed_rows,
        "matched": matched,
    }


def inspect_manifest_bound_direct_reference(
    direct_exact_reference: Path,
    payload: dict[str, object],
    *,
    parent_canonical_smiles: str | None = None,
    discovery_key_sha256: str | None = None,
    stored_keys_out: set[str] | None = None,
) -> dict[str, object]:
    outputs = payload.get("outputs")
    output_bytes = payload.get("output_bytes")
    counts = payload.get("counts")
    if not all(isinstance(item, dict) for item in (outputs, output_bytes, counts)):
        raise ValueError("Direct exact manifest output/count records are missing")
    return inspect_direct_exact_reference(
        direct_exact_reference,
        expected_sha256=outputs.get("direct_reference_sha256"),
        expected_bytes=output_bytes.get("direct_reference"),
        expected_rows=counts.get("direct_reference_rows"),
        parent_canonical_smiles=parent_canonical_smiles,
        discovery_key_sha256=discovery_key_sha256,
        stored_keys_out=stored_keys_out,
    )


def validate_direct_exact_manifest(
    direct_exact_reference: Path,
    manifest_path: Path,
    *,
    stored_keys_out: set[str] | None = None,
) -> dict[str, object]:
    """Validate the sealed alias-map bindings needed by a leakage audit."""

    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(f"Direct exact manifest is missing or unsafe: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Direct exact manifest is invalid: {manifest_path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != "discovery_alias_map.v1":
        raise ValueError("Direct exact manifest schema_version mismatch")
    if payload.get("binding_sha256") != _payload_binding_sha256(payload):
        raise ValueError("Direct exact manifest binding_sha256 mismatch")
    outputs = payload.get("outputs")
    output_paths = payload.get("output_paths")
    output_bytes = payload.get("output_bytes")
    if not all(isinstance(item, dict) for item in (outputs, output_paths, output_bytes)):
        raise ValueError("Direct exact manifest output records are missing")
    direct = _resolve_manifest_file(
        manifest_path,
        output_paths.get("direct_reference"),
        "Direct exact manifest direct_reference",
    )
    if direct != direct_exact_reference.resolve():
        raise ValueError("Direct exact manifest does not bind the active reference")
    aliases = _resolve_manifest_file(
        manifest_path,
        output_paths.get("aliases_parquet"),
        "Direct exact manifest aliases_parquet",
    )
    registry = _resolve_manifest_file(
        manifest_path,
        payload.get("source_registry"),
        "Direct exact manifest source_registry",
    )
    inspect_manifest_bound_direct_reference(
        direct,
        payload,
        stored_keys_out=stored_keys_out,
    )
    if outputs.get("aliases_parquet_sha256") != _sha256(aliases):
        raise ValueError("Direct exact manifest aliases SHA-256 drift")
    if output_bytes.get("aliases_parquet") != aliases.stat().st_size:
        raise ValueError("Direct exact manifest aliases byte-count drift")
    if payload.get("registry_sha256") != _sha256(registry):
        raise ValueError("Direct exact manifest registry SHA-256 drift")
    builder = Path(__file__).resolve().parents[1] / "scripts/build_discovery_alias_map.py"
    if payload.get("builder_script_sha256") != _sha256(builder):
        raise ValueError("Direct exact manifest builder script SHA-256 drift")
    canonical_payload = {
        "schema_version": payload.get("schema_version"),
        "registry_sha256": payload.get("registry_sha256"),
        "source_manifest_sha256": payload.get("source_manifest_sha256"),
        "source_artifact_sha256": payload.get("source_artifact_sha256"),
        "outputs": outputs,
        "counts": payload.get("counts"),
        "canonicalization_audit": payload.get("canonicalization_audit"),
        "policy": payload.get("policy"),
        "canonical_contract": payload.get("canonical_contract"),
        "builder_script_sha256": payload.get("builder_script_sha256"),
    }
    canonical_hash = hashlib.sha256(
        json.dumps(
            canonical_payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    if payload.get("canonical_payload_sha256") != canonical_hash:
        raise ValueError("Direct exact manifest canonical payload mismatch")
    return payload


def validate_discovery_alias_package(
    direct_exact_reference: Path,
    manifest_path: Path,
) -> dict[str, object]:
    """Validate the final alias map and its complete four-source provenance chain."""

    payload = validate_direct_exact_manifest(direct_exact_reference, manifest_path)
    registry_path = _resolve_manifest_file(
        manifest_path,
        payload.get("source_registry"),
        "Discovery alias source_registry",
    )
    registry = _load_json_object(registry_path, "Discovery alias source registry")
    if registry.get("schema_version") != "discovery_alias_source_registry.v1":
        raise ValueError("Discovery alias source registry schema_version mismatch")
    if registry.get("registry_root", ".") != ".":
        raise ValueError("Discovery alias source registry_root must be '.'")
    sources = registry.get("sources")
    expected_names = ("ChEMBL", "BindingDB", "GtoPdb", "PubChem")
    if not isinstance(sources, list) or tuple(
        source.get("canonical_name")
        for source in sources
        if isinstance(source, dict)
    ) != expected_names:
        raise ValueError(
            "Discovery alias source registry must contain ChEMBL, BindingDB, "
            "GtoPdb, and PubChem in canonical order"
        )
    map_source_counts = _validate_canonicalization_audit(
        manifest_path,
        payload,
        expected_names,
    )

    source_hashes = payload.get("source_artifact_sha256")
    if not isinstance(source_hashes, dict) or set(source_hashes) != set(expected_names):
        raise ValueError("Direct exact manifest source artifact bindings are incomplete")
    registry_records: dict[str, dict[str, object]] = {}
    for source in sources:
        if not isinstance(source, dict):  # pragma: no cover - guarded above
            raise ValueError("Discovery alias source entry must be an object")
        source_name = str(source["canonical_name"])
        for field in ("release", "release_date", "license", "license_url"):
            _nonblank(source.get(field), f"Discovery alias {source_name}.{field}")
        if not str(source.get("license_url", "")).startswith("https://"):
            raise ValueError(f"Discovery alias {source_name}.license_url must use https")
        if source.get("redistribution") != "allowed":
            raise ValueError(f"Discovery alias {source_name} redistribution is not allowed")
        artifact = source.get("artifact")
        if not isinstance(artifact, dict):
            raise ValueError(f"Discovery alias {source_name} artifact record is missing")
        artifact_path = _resolve_manifest_file(
            registry_path,
            artifact.get("path"),
            f"Discovery alias {source_name} artifact",
        )
        observed_hash = _sha256(artifact_path)
        observed_bytes = artifact_path.stat().st_size
        if artifact.get("sha256") != observed_hash:
            raise ValueError(f"Discovery alias {source_name} artifact SHA-256 drift")
        if artifact.get("bytes") != observed_bytes:
            raise ValueError(f"Discovery alias {source_name} artifact byte-count drift")
        _positive_int(artifact.get("rows"), f"Discovery alias {source_name} artifact.rows")
        if source_hashes.get(source_name) != observed_hash:
            raise ValueError(
                f"Direct exact manifest does not bind the active {source_name} artifact"
            )
        registry_records[source_name] = {
            "source": source,
            "artifact": artifact,
            "path": artifact_path,
        }

    source_manifest_path = _resolve_manifest_file(
        manifest_path,
        payload.get("source_manifest"),
        "Discovery alias source manifest",
    )
    if source_manifest_path.parent != registry_path.parent:
        raise ValueError("Discovery alias source manifest/registry directory mismatch")
    if payload.get("source_manifest_sha256") != _sha256(source_manifest_path):
        raise ValueError("Discovery alias source manifest SHA-256 drift")
    source_manifest = _load_json_object(
        source_manifest_path,
        "Discovery alias source manifest",
    )
    if source_manifest.get("schema_version") != "skinscout.discovery-alias-sources.v1":
        raise ValueError("Discovery alias source manifest schema_version mismatch")
    unsigned_source_manifest = dict(source_manifest)
    unsigned_source_manifest.pop("self_binding_sha256", None)
    canonical = json.dumps(
        unsigned_source_manifest,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if source_manifest.get("self_binding_sha256") != hashlib.sha256(canonical).hexdigest():
        raise ValueError("Discovery alias source manifest self binding mismatch")
    if source_manifest.get("source_registry") != "source_registry.json":
        raise ValueError("Discovery alias source manifest registry path mismatch")
    if source_manifest.get("source_registry_sha256") != _sha256(registry_path):
        raise ValueError("Discovery alias source manifest registry SHA-256 drift")
    builder = source_manifest.get("builder")
    source_builder = Path(__file__).resolve().parents[1] / "scripts/build_discovery_alias_sources.py"
    if not isinstance(builder, dict) or builder.get("sha256") != _sha256(source_builder):
        raise ValueError("Discovery alias source builder script SHA-256 drift")

    source_stats = source_manifest.get("source_stats")
    pubchem_stats = source_stats.get("PubChem") if isinstance(source_stats, dict) else None
    if not isinstance(pubchem_stats, dict):
        raise ValueError("Discovery alias source manifest PubChem stats are missing")
    selected_cids = _require_nonnegative_int(
        pubchem_stats.get("selected_cids"),
        "Discovery alias PubChem selected_cids",
    )
    matched_cids = _require_nonnegative_int(
        pubchem_stats.get("matched_cids"),
        "Discovery alias PubChem matched_cids",
    )
    missing_cids = _require_nonnegative_int(
        pubchem_stats.get("missing_cids"),
        "Discovery alias PubChem missing_cids",
    )
    missing_fraction_ppm = _require_nonnegative_int(
        pubchem_stats.get("missing_fraction_ppm"),
        "Discovery alias PubChem missing_fraction_ppm",
    )
    if selected_cids <= 0 or matched_cids <= 0:
        raise ValueError("Discovery alias PubChem selected/matched CIDs must be positive")
    if selected_cids != matched_cids + missing_cids:
        raise ValueError("Discovery alias PubChem CID stats do not conserve")
    expected_missing_fraction_ppm = (
        (missing_cids * 1_000_000 + selected_cids - 1) // selected_cids
    )
    if missing_fraction_ppm != expected_missing_fraction_ppm:
        raise ValueError("Discovery alias PubChem missing fraction mismatch")
    policy = source_manifest.get("policy")
    if (
        not isinstance(policy, dict)
        or policy.get("PubChem_max_missing_fraction_ppm") != 100_000
        or missing_fraction_ppm > 100_000
    ):
        raise ValueError("Discovery alias PubChem missing-CID policy mismatch")

    audit_artifacts = source_manifest.get("audit_artifacts")
    audit_name = "pubchem_missing_cids.txt"
    if not isinstance(audit_artifacts, dict) or set(audit_artifacts) != {audit_name}:
        raise ValueError("Discovery alias PubChem missing-CID audit record is incomplete")
    audit_record = audit_artifacts[audit_name]
    audit_path = source_manifest_path.parent / audit_name
    if (
        not isinstance(audit_record, dict)
        or audit_path.is_symlink()
        or not audit_path.is_file()
    ):
        raise ValueError("Discovery alias PubChem missing-CID audit is missing or unsafe")
    if (
        audit_record.get("sha256") != _sha256(audit_path)
        or audit_record.get("bytes") != audit_path.stat().st_size
    ):
        raise ValueError("Discovery alias PubChem missing-CID audit hash/bytes drift")
    audit_rows = 0
    previous_cid = -1
    with audit_path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            cid = line.rstrip("\n")
            if not re.fullmatch(r"[0-9]+", cid):
                raise ValueError(
                    f"Discovery alias PubChem missing-CID audit line {line_number} is invalid"
                )
            numeric_cid = int(cid)
            if numeric_cid <= previous_cid:
                raise ValueError(
                    "Discovery alias PubChem missing-CID audit must be strictly increasing"
                )
            previous_cid = numeric_cid
            audit_rows += 1
    if audit_record.get("rows") != audit_rows or audit_rows != missing_cids:
        raise ValueError("Discovery alias PubChem missing-CID audit row-count drift")

    inputs = source_manifest.get("inputs")
    output_hashes = source_manifest.get("output_sha256")
    output_bytes = source_manifest.get("output_bytes")
    output_rows = source_manifest.get("output_rows")
    if not all(
        isinstance(item, dict)
        for item in (inputs, output_hashes, output_bytes, output_rows)
    ):
        raise ValueError("Discovery alias source manifest records are incomplete")
    if set(inputs) != set(expected_names):
        raise ValueError("Discovery alias source manifest inputs are incomplete")

    for source_name in expected_names:
        input_record = inputs.get(source_name)
        if not isinstance(input_record, dict):
            raise ValueError(f"Discovery alias {source_name} input record is missing")
        input_path = _resolve_relative_file(
            source_manifest_path,
            input_record.get("path"),
            f"Discovery alias {source_name} input",
        )
        input_hash = _sha256(input_path)
        input_bytes = input_path.stat().st_size
        if input_record.get("sha256") != input_hash:
            raise ValueError(f"Discovery alias {source_name} input SHA-256 drift")
        if input_record.get("bytes") != input_bytes:
            raise ValueError(f"Discovery alias {source_name} input byte-count drift")

        manifest_record = input_record.get("manifest")
        if not isinstance(manifest_record, dict):
            raise ValueError(f"Discovery alias {source_name} upstream manifest record is missing")
        upstream_manifest_path = _resolve_relative_file(
            source_manifest_path,
            manifest_record.get("path"),
            f"Discovery alias {source_name} upstream manifest",
        )
        if manifest_record.get("sha256") != _sha256(upstream_manifest_path):
            raise ValueError(
                f"Discovery alias {source_name} upstream manifest SHA-256 drift"
            )
        if manifest_record.get("bytes") != upstream_manifest_path.stat().st_size:
            raise ValueError(
                f"Discovery alias {source_name} upstream manifest byte-count drift"
            )
        upstream_payload = _load_json_object(
            upstream_manifest_path,
            f"Discovery alias {source_name} upstream manifest",
        )
        upstream_source, upstream_artifact = _upstream_source_contract(
            source_name,
            upstream_payload,
        )
        if (
            upstream_artifact.get("sha256") != input_hash
            or upstream_artifact.get("bytes") != input_bytes
        ):
            raise ValueError(
                f"Discovery alias {source_name} upstream artifact binding mismatch"
            )
        registry_source = registry_records[source_name]["source"]
        for field in ("release", "release_date", "license", "license_url"):
            expected = upstream_source.get(field)
            if manifest_record.get(field) != expected or registry_source.get(field) != expected:
                raise ValueError(
                    f"Discovery alias {source_name} {field} provenance mismatch"
                )
        if upstream_source.get("name") != source_name:
            raise ValueError(f"Discovery alias {source_name} upstream source.name mismatch")
        if upstream_source.get("redistribution") != "allowed":
            raise ValueError(
                f"Discovery alias {source_name} upstream redistribution is not allowed"
            )

        artifact = registry_records[source_name]["artifact"]
        artifact_name = Path(str(artifact["path"])).name
        if output_hashes.get(artifact_name) != artifact.get("sha256"):
            raise ValueError(f"Discovery alias {source_name} output SHA-256 mismatch")
        if output_bytes.get(artifact_name) != artifact.get("bytes"):
            raise ValueError(f"Discovery alias {source_name} output byte-count mismatch")
        if output_rows.get(artifact_name) != artifact.get("rows"):
            raise ValueError(f"Discovery alias {source_name} output row-count mismatch")
        if map_source_counts[source_name]["input_rows"] != artifact.get("rows"):
            raise ValueError(
                f"Discovery alias {source_name} map/source row-count mismatch"
            )
    return payload


def _source_record(path: Path) -> dict[str, object]:
    record: dict[str, object] = {"path": str(path)}
    if not path.exists():
        record["status"] = "missing"
        return record
    if path.is_file():
        record.update({
            "status": "present",
            "kind": "file",
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
        return record
    files = sorted(item for item in path.rglob("*") if item.is_file())
    digest = hashlib.sha256()
    for file_path in files:
        rel = file_path.relative_to(path).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(file_path).encode("ascii"))
        digest.update(b"\0")
    record.update({
        "status": "present",
        "kind": "directory",
        "n_files": len(files),
        "tree_sha256": digest.hexdigest(),
    })
    return record


def _source_count(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return 1 if path.stat().st_size > 0 else 0
    return sum(1 for item in path.rglob("*") if item.is_file() and item.stat().st_size > 0)


def _require_supplied_nonempty_file(path: Path, label: str) -> None:
    if not path.exists():
        raise RuntimeError(f"{label} is required and missing: {path}")
    if not path.is_file():
        raise RuntimeError(f"{label} must be a file: {path}")
    if path.stat().st_size == 0:
        raise RuntimeError(f"{label} is required and must be non-empty: {path}")


def _read_required_csv(path: Path, required_cols: set[str], label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"{label} is required and must be non-empty: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise SystemExit(f"{label} failed to parse: {path}: {exc}") from exc
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise RuntimeError(
            f"{label} missing required column(s): " + ", ".join(missing)
        )
    if df.empty:
        raise RuntimeError(f"{label} contains no rows")
    return df


def _validate_nonempty_strings(df: pd.DataFrame, column: str, label: str) -> None:
    invalid = [
        int(idx) for idx, value in df[column].items()
        if pd.isna(value) or not str(value).strip()
    ]
    if invalid:
        shown = ", ".join(str(idx) for idx in invalid[:10])
        suffix = "..." if len(invalid) > 10 else ""
        raise RuntimeError(
            f"{label} column '{column}' contains blank values at row index(es) "
            f"{shown}{suffix}"
        )


def _validate_unique_eval_pairs(rows: pd.DataFrame) -> None:
    seen: dict[tuple[str, str], int] = {}
    duplicate_pairs: list[str] = []
    for idx, row in rows.iterrows():
        uid = str(row["uniprot"]).strip()
        try:
            canonical = discovery_key(
                row["smiles"],
                label=f"Leakage audit input row index {int(idx)} smiles",
            ).parent_canonical_smiles
        except ValueError as exc:
            if "not parseable" in str(exc):
                raise RuntimeError(
                    f"Leakage audit input contains invalid SMILES: {str(row['smiles']).strip()}"
                ) from exc
            raise RuntimeError(str(exc)) from exc
        pair = (uid, canonical)
        row_idx = int(idx)
        if pair in seen:
            duplicate_pairs.append(
                f"{uid}/{canonical} at row index(es) {seen[pair]}, {row_idx}"
            )
            continue
        seen[pair] = row_idx
    if duplicate_pairs:
        shown = "; ".join(duplicate_pairs[:10])
        suffix = "..." if len(duplicate_pairs) > 10 else ""
        raise RuntimeError(
            "Leakage audit input contains duplicate canonical evaluation pair(s): "
            f"{shown}{suffix}"
        )


def _validate_thresholds(thresholds: "Thresholds") -> None:
    for name, value in (
        ("seq_id", thresholds.seq_id),
        ("ligand_tanimoto", thresholds.ligand_tanimoto),
        ("pocket_sucos", thresholds.pocket_sucos),
    ):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise RuntimeError(
                f"Leakage threshold '{name}' must be a finite value in [0, 1]: {value}"
            )


@dataclass(frozen=True)
class Thresholds:
    seq_id: float
    ligand_tanimoto: float
    pocket_sucos: float


def _threshold_record(thresholds: Thresholds) -> dict[str, float]:
    _validate_thresholds(thresholds)
    return {
        "seq_id": float(thresholds.seq_id),
        "ligand_tanimoto": float(thresholds.ligand_tanimoto),
        "pocket_sucos": float(thresholds.pocket_sucos),
    }


def _resolve_fasta(path: Path) -> Path | None:
    if path.is_file() and path.suffix.lower() in {".fa", ".faa", ".fasta"}:
        return path
    if path.is_dir():
        for name in ("training_cutoff_seqs.fasta", "training_cutoff.fasta", "sequences.fasta"):
            cand = path / name
            if cand.exists():
                return cand
    return None


def _read_fasta(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    current: str | None = None
    chunks: list[str] = []

    def flush_current() -> None:
        if current is None:
            return
        sequence = "".join(chunks).upper()
        if not sequence:
            raise RuntimeError(f"FASTA entry '{current}' has no sequence: {path}")
        if current in records:
            raise RuntimeError(f"FASTA contains duplicate FASTA header '{current}': {path}")
        records[current] = sequence

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            flush_current()
            parts = line[1:].split()
            if not parts or not parts[0].strip():
                raise RuntimeError(f"FASTA contains a blank header: {path}")
            current = parts[0].strip()
            chunks = []
        else:
            if current is None:
                raise RuntimeError(f"FASTA contains sequence before first header: {path}")
            chunks.append(line)
    flush_current()
    if not records:
        raise RuntimeError(f"FASTA contains no sequence records: {path}")
    return records


def _sequence_identity(query: str, reference: str) -> float:
    """Return conservative containment identity normalized by the shorter sequence.

    This fallback is intentionally fragment/domain sensitive: a short evaluation
    sequence fully contained in a longer training reference must score as 1.0
    instead of being diluted by the full reference length.
    """
    q = "".join(query.upper().split())
    r = "".join(reference.upper().split())
    if not q or not r:
        return 0.0
    previous = [0] * (len(r) + 1)
    for i, qc in enumerate(q, start=1):
        current = [0] * (len(r) + 1)
        for j, rc in enumerate(r, start=1):
            current[j] = max(
                previous[j - 1] + int(qc == rc),
                previous[j],
                current[j - 1],
            )
        previous = current
    denominator = min(len(q), len(r))
    return previous[-1] / denominator if denominator else 0.0


def _mmseqs_easy_search(
    *,
    target_uniprot: str,
    query_sequence: str,
    training_db: Path,
) -> float:
    with tempfile.TemporaryDirectory(prefix="skinscout_mmseqs_") as tmp:
        tmpdir = Path(tmp)
        query_fasta = tmpdir / "query.fasta"
        result_tsv = tmpdir / "result.tsv"
        work_dir = tmpdir / "tmp"
        query_fasta.write_text(f">{target_uniprot}\n{query_sequence.strip()}\n")
        cmd = [
            "mmseqs",
            "easy-search",
            str(query_fasta),
            str(training_db),
            str(result_tsv),
            str(work_dir),
            "--format-output",
            "query,target,pident",
            "--threads",
            "1",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            tail = (res.stderr or res.stdout).strip().splitlines()[-1:] or ["unknown error"]
            raise RuntimeError(
                f"MMseqs sequence leakage search failed for {target_uniprot}: {tail[0]}"
            )
        if not result_tsv.exists() or result_tsv.stat().st_size == 0:
            return 0.0
        scores: list[float] = []
        for line_no, line in enumerate(result_tsv.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                raise RuntimeError(
                    f"MMseqs result has fewer than 3 columns at line {line_no}: {result_tsv}"
                )
            try:
                pident = float(parts[2])
            except ValueError as exc:
                raise RuntimeError(
                    f"MMseqs pident is not numeric at line {line_no}: {result_tsv}"
                ) from exc
            score = pident / 100.0 if pident > 1.0 else pident
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise RuntimeError(
                    f"MMseqs pident is outside [0, 100] at line {line_no}: {result_tsv}"
                )
            scores.append(score)
        return max(scores) if scores else 0.0


def mmseqs_max_seq_id(target_uniprot: str,
                      training_db: Path,
                      query_sequence: str | None = None) -> float | None:
    fasta = _resolve_fasta(training_db)
    if query_sequence and fasta is not None:
        refs = _read_fasta(fasta)
        if not refs:
            return None
        fallback_score = max(_sequence_identity(query_sequence, seq) for seq in refs.values())
        if fallback_score == 1.0:
            return fallback_score
        if shutil.which("mmseqs"):
            mmseqs_score = _mmseqs_easy_search(
                target_uniprot=target_uniprot,
                query_sequence=query_sequence,
                training_db=fasta,
            )
            return max(mmseqs_score, fallback_score)
        return fallback_score
    if query_sequence and shutil.which("mmseqs") and training_db.exists():
        return _mmseqs_easy_search(
            target_uniprot=target_uniprot,
            query_sequence=query_sequence,
            training_db=training_db,
        )
    return None


def ligand_max_tanimoto(smiles: str, training_ligands: Path) -> float | None:
    if not training_ligands.exists():
        import logging as _log
        _log.warning(
            "leakage_check: training ligand reference not found (%s) — "
            "ligand axis will be treated as INCOMPLETE, not clean",
            training_ligands,
        )
        return None
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise RuntimeError(f"Leakage audit input contains invalid SMILES: {smiles}")
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    best = 0.0
    n_refs = 0
    seen_refs: dict[str, int] = {}
    for line_no, line in enumerate(training_ligands.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        smi = line.split()[0]
        ref = Chem.MolFromSmiles(smi)
        if ref is None:
            raise RuntimeError(
                f"Training ligand reference contains invalid SMILES at line "
                f"{line_no}: {training_ligands}"
            )
        canonical = Chem.MolToSmiles(ref, canonical=True)
        if canonical in seen_refs:
            raise RuntimeError(
                "Training ligand reference contains duplicate canonical SMILES "
                f"{canonical} at line(s) {seen_refs[canonical]}, {line_no}: "
                f"{training_ligands}"
            )
        seen_refs[canonical] = line_no
        n_refs += 1
        ref_bv = AllChem.GetMorganFingerprintAsBitVect(ref, radius=2, nBits=2048)
        best = max(best, DataStructs.TanimotoSimilarity(bv, ref_bv))
    return best if n_refs else None


def _direct_exact_keys(path: Path) -> set[str]:
    _require_supplied_nonempty_file(path, "Direct exact reference")
    keys: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            smi = line.split()[0]
            try:
                keys.add(
                    discovery_key(
                        smi,
                        label=f"direct exact reference {path} line {line_no}",
                    ).discovery_key_sha256
                )
            except ValueError as exc:
                raise RuntimeError(str(exc)) from exc
    if not keys:
        raise RuntimeError(f"Direct exact reference contains no parseable rows: {path}")
    return keys


def _finite_unit_score(value: object, label: str) -> float:
    if isinstance(value, bool) or type(value).__name__ == "bool_":
        raise RuntimeError(f"{label} must be numeric")
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} must be numeric") from exc
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise RuntimeError(f"{label} must be a finite value in [0, 1]: {value}")
    return score


def _first_present_score(payload: dict[str, object], keys: tuple[str, ...]) -> object | None:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise RuntimeError(f"duplicate target '{key}'")
        payload[key] = value
    return payload


def pocket_sucos(target_uniprot: str, training_holo: Path) -> float | None:
    if not training_holo.exists():
        import logging as _log
        _log.warning(
            "leakage_check: pocket SuCOS reference not found (%s) — "
            "pocket axis will be treated as INCOMPLETE, not clean",
            training_holo,
        )
        return None
    score_cols = ("pocket_sucos", "sucos", "max_sucos")
    suffix = training_holo.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(training_holo, sep=sep)
        if df.empty:
            raise RuntimeError(f"Pocket SuCOS reference contains no rows: {training_holo}")
        uid_col = next((c for c in ("uniprot", "target_id", "eval_target_uniprot")
                        if c in df.columns), None)
        score_col = next((c for c in score_cols if c in df.columns), None)
        if uid_col is None:
            raise RuntimeError(
                f"Pocket SuCOS reference missing target id column: {training_holo}"
            )
        if score_col is None:
            raise RuntimeError(
                f"Pocket SuCOS reference missing score column: {training_holo}"
            )
        blank_targets = [
            int(idx) for idx, value in df[uid_col].items()
            if pd.isna(value) or not str(value).strip()
        ]
        if blank_targets:
            shown = ", ".join(str(idx) for idx in blank_targets[:10])
            suffix = "..." if len(blank_targets) > 10 else ""
            raise RuntimeError(
                f"Pocket SuCOS reference column '{uid_col}' contains blank values "
                f"at row index(es) {shown}{suffix}: {training_holo}"
            )
        df = df.copy()
        df[uid_col] = df[uid_col].astype(str).str.strip()
        df[score_col] = [
            _finite_unit_score(value, f"{training_holo} column '{score_col}'")
            for value in df[score_col].tolist()
        ]
        rows = df[df[uid_col].astype(str) == target_uniprot]
        if rows.empty:
            return None
        return float(rows[score_col].max())
    if suffix == ".json":
        try:
            payload = json.loads(
                training_holo.read_text(),
                object_pairs_hook=_reject_duplicate_json_keys,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"Pocket SuCOS JSON reference contains {exc}: {training_holo}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Pocket SuCOS JSON reference failed to parse: {training_holo}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"Pocket SuCOS JSON reference must be an object: {training_holo}")
        if not payload:
            raise RuntimeError(f"Pocket SuCOS JSON reference contains no targets: {training_holo}")
        if target_uniprot not in payload:
            return None
        val = payload[target_uniprot]
        if isinstance(val, dict):
            val = _first_present_score(val, score_cols)
            if val is None:
                raise RuntimeError(
                    f"Pocket SuCOS JSON reference missing score for target "
                    f"'{target_uniprot}': {training_holo}"
                )
        if val is None:
            raise RuntimeError(
                f"Pocket SuCOS JSON reference has null score for target "
                f"'{target_uniprot}': {training_holo}"
            )
        return _finite_unit_score(
            val,
            f"Pocket SuCOS JSON reference {training_holo} target '{target_uniprot}'",
        )
    raise RuntimeError(
        "Pocket SuCOS reference has unsupported suffix; expected .csv, .tsv, "
        f"or .json: {training_holo}"
    )


def audit(rows: pd.DataFrame,
          training_seq_db: Path,
          training_ligands: Path,
          training_holo: Path,
          thresholds: Thresholds,
          allow_incomplete: bool = False) -> pd.DataFrame:
    _validate_thresholds(thresholds)
    required_cols = {"uniprot", "smiles"}
    missing_cols = sorted(required_cols - set(rows.columns))
    if missing_cols:
        raise RuntimeError(
            "Leakage audit input missing required column(s): " + ", ".join(missing_cols)
        )
    if rows.empty:
        raise RuntimeError("Leakage audit input contains no rows")
    _validate_nonempty_strings(rows, "uniprot", "Leakage audit input")
    _validate_nonempty_strings(rows, "smiles", "Leakage audit input")
    _validate_unique_eval_pairs(rows)
    out: list[dict] = []
    for _, row in rows.iterrows():
        uid = str(row["uniprot"]).strip()
        smi = str(row["smiles"]).strip()
        try:
            key = discovery_key(smi, label=f"Leakage audit input {uid} smiles")
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        query_sequence = None
        if "sequence" in rows.columns and pd.notna(row["sequence"]):
            sequence = str(row["sequence"]).strip()
            query_sequence = sequence or None
        seq = mmseqs_max_seq_id(uid, training_seq_db, query_sequence=query_sequence)
        tan = ligand_max_tanimoto(smi, training_ligands)
        suc = pocket_sucos(uid, training_holo)
        missing_axes = [
            name for name, value in (
                ("sequence", seq),
                ("ligand", tan),
                ("pocket", suc),
            )
            if value is None
        ]
        leak = (
            (seq is not None and seq >= thresholds.seq_id)
            or (tan is not None and tan >= thresholds.ligand_tanimoto)
            or (suc is not None and suc >= thresholds.pocket_sucos)
        )
        out.append({"uniprot": uid, "smiles": smi,
                    "parent_canonical_smiles": key.parent_canonical_smiles,
                    "discovery_key_sha256": key.discovery_key_sha256,
                    "parent_inchikey": key.parent_inchikey,
                    "parent_connectivity_inchikey": key.parent_connectivity_inchikey,
                    "seq_id": seq, "ligand_tanimoto": tan,
                    "pocket_sucos": suc, "leak_flag": leak,
                    "audit_status": "incomplete" if missing_axes else "ok",
                    "missing_axes": ";".join(missing_axes)})
    df = pd.DataFrame(out)
    if not allow_incomplete and (df["audit_status"] == "incomplete").any():
        missing = sorted(
            {
                axis
                for axes in df.loc[df["audit_status"] == "incomplete", "missing_axes"]
                for axis in str(axes).split(";")
                if axis
            }
        )
        raise RuntimeError(
            "Leakage audit incomplete; missing implemented/reference axes: "
            + ", ".join(missing)
        )
    return df


def sealed_discovery_audit(
    rows: pd.DataFrame,
    out: pd.DataFrame,
    *,
    input_csv: Path,
    training_seq_db: Path,
    training_ligands: Path,
    training_holo: Path,
    direct_exact_reference: Path,
    out_csv: Path,
    thresholds: Thresholds,
    execution_challenge: str,
    manifest_validated_direct_keys: set[str] | None = None,
) -> dict[str, object]:
    direct_keys = (
        manifest_validated_direct_keys
        if manifest_validated_direct_keys is not None
        else _direct_exact_keys(direct_exact_reference)
    )
    challenge = _require_sha256(execution_challenge, "execution_challenge")
    excluded: list[dict[str, object]] = []
    survivors: list[dict[str, object]] = []
    direct_exact_count = 0
    incomplete_audit_count = 0
    input_bindings: list[dict[str, object]] = []
    for idx, row in out.iterrows():
        reasons: list[str] = []
        if row["audit_status"] != "ok":
            reasons.append("incomplete_audit")
            incomplete_audit_count += 1
        if bool(row["leak_flag"]):
            reasons.append("leakage_axis_flag")
        if row["discovery_key_sha256"] in direct_keys:
            reasons.append("direct_exact_match")
            direct_exact_count += 1
        record = {
            "row_index": int(idx),
            "uniprot": str(row["uniprot"]),
            "discovery_key_sha256": str(row["discovery_key_sha256"]),
            "parent_canonical_smiles": str(row["parent_canonical_smiles"]),
            "parent_inchikey": str(row["parent_inchikey"]),
            "parent_connectivity_inchikey": str(row["parent_connectivity_inchikey"]),
        }
        input_bindings.append({
            **record,
            "input_type": "smiles",
            "input_smiles": str(row["smiles"]),
        })
        if reasons:
            excluded.append({**record, "reasons": reasons})
        else:
            survivors.append(record)

    payload = {
        "schema_version": DISCOVERY_AUDIT_SCHEMA,
        "status": (
            "ok"
            if direct_exact_count == 0
            and incomplete_audit_count == 0
            and len(survivors) > 0
            else "failed"
        ),
        "rdkit_version": current_rdkit_version(),
        "rdkit_version_contract": RDKIT_VERSION_CONTRACT,
        "canonical_pipeline": list(CANONICAL_PIPELINE),
        "execution_challenge": challenge,
        "thresholds": _threshold_record(thresholds),
        "inputs": {
            "input_csv": _source_record(input_csv),
            "training_seq_db": _source_record(training_seq_db),
            "training_ligands": _source_record(training_ligands),
            "training_holo": _source_record(training_holo),
            "direct_exact_reference": _source_record(direct_exact_reference),
            "leakage_audit_csv": _source_record(out_csv),
        },
        "counts": {
            "input_rows": int(len(rows)),
            "output_rows": int(len(out)),
            "excluded_rows": len(excluded),
            "survivor_rows": len(survivors),
            "direct_exact_count": direct_exact_count,
            "incomplete_audit_count": incomplete_audit_count,
            "input_source_count": _source_count(input_csv),
            "training_seq_source_count": _source_count(training_seq_db),
            "training_ligand_source_count": _source_count(training_ligands),
            "training_holo_source_count": _source_count(training_holo),
            "direct_exact_source_count": _source_count(direct_exact_reference),
            "leakage_audit_source_count": _source_count(out_csv),
        },
        "direct_exact_count": direct_exact_count,
        "incomplete_audit_count": incomplete_audit_count,
        "neutral_exclusion_count": len(excluded),
        "input_bindings": input_bindings,
        "excluded_rows": excluded,
        "survivors": survivors,
    }
    payload["binding_sha256"] = _payload_binding_sha256(payload)
    validate_sealed_discovery_audit(payload)
    return payload


def _require_object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _require_list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _require_nonempty_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _require_sha256(value: object, label: str) -> str:
    text = _require_nonempty_text(value, label).lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return text


def _require_nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _validate_source_record(
    record: object,
    label: str,
    *,
    require_hash: bool,
) -> None:
    payload = _require_object(record, label)
    _require_nonempty_text(payload.get("path"), f"{label}.path")
    status = _require_nonempty_text(payload.get("status"), f"{label}.status")
    if status != "present":
        raise ValueError(f"{label}.status must be present")
    kind = _require_nonempty_text(payload.get("kind"), f"{label}.kind")
    if kind == "file":
        if require_hash:
            _require_sha256(payload.get("sha256"), f"{label}.sha256")
        if _require_nonnegative_int(payload.get("bytes"), f"{label}.bytes") < 1:
            raise ValueError(f"{label}.bytes must be positive")
    elif kind == "directory":
        _require_sha256(payload.get("tree_sha256"), f"{label}.tree_sha256")
        if _require_nonnegative_int(payload.get("n_files"), f"{label}.n_files") < 1:
            raise ValueError(f"{label}.n_files must be positive")
    else:
        raise ValueError(f"{label}.kind has invalid value: {kind}")


def _validate_threshold_record(value: object) -> dict[str, float]:
    payload = _require_object(value, "thresholds")
    expected = {"seq_id", "ligand_tanimoto", "pocket_sucos"}
    if set(payload) != expected:
        raise ValueError(
            "thresholds must contain exactly seq_id, ligand_tanimoto, pocket_sucos"
        )
    parsed: dict[str, float] = {}
    for key in sorted(expected):
        try:
            parsed[key] = _finite_unit_score(payload[key], f"thresholds.{key}")
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc
    return parsed


def _parse_audit_score(value: object, label: str) -> float | None:
    if pd.isna(value) or str(value).strip() == "":
        return None
    try:
        return _finite_unit_score(value, label)
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc


def _expected_missing_axes(
    seq_id: float | None,
    ligand_tanimoto: float | None,
    pocket_sucos: float | None,
) -> list[str]:
    return [
        axis
        for axis, value in (
            ("sequence", seq_id),
            ("ligand", ligand_tanimoto),
            ("pocket", pocket_sucos),
        )
        if value is None
    ]


def _parse_missing_axes(value: object, label: str) -> list[str]:
    if pd.isna(value):
        text = ""
    else:
        text = str(value).strip()
    axes = [axis.strip() for axis in text.split(";") if axis.strip()]
    valid = {"sequence", "ligand", "pocket"}
    invalid = [axis for axis in axes if axis not in valid]
    if invalid:
        raise ValueError(f"{label} contains invalid axis: {', '.join(invalid)}")
    if len(set(axes)) != len(axes):
        raise ValueError(f"{label} contains duplicate axes")
    return axes


def _expected_leak_flag(
    seq_id: float | None,
    ligand_tanimoto: float | None,
    pocket_sucos: float | None,
    thresholds: dict[str, float],
) -> bool:
    return (
        (seq_id is not None and seq_id >= thresholds["seq_id"])
        or (
            ligand_tanimoto is not None
            and ligand_tanimoto >= thresholds["ligand_tanimoto"]
        )
        or (pocket_sucos is not None and pocket_sucos >= thresholds["pocket_sucos"])
    )


def validate_sealed_discovery_audit(
    payload: dict[str, object],
    *,
    expected_input_key: str | None = None,
) -> None:
    if payload.get("schema_version") != DISCOVERY_AUDIT_SCHEMA:
        raise ValueError(
            "Discovery leakage audit schema_version must be "
            f"{DISCOVERY_AUDIT_SCHEMA}"
        )
    status = _require_nonempty_text(payload.get("status"), "status")
    if status not in {"ok", "failed"}:
        raise ValueError(f"status has invalid value: {status}")
    validate_rdkit_version_contract(payload.get("rdkit_version"))
    if payload.get("rdkit_version_contract") != RDKIT_VERSION_CONTRACT:
        raise ValueError("rdkit_version_contract does not match canonical contract")
    if payload.get("canonical_pipeline") != list(CANONICAL_PIPELINE):
        raise ValueError("canonical_pipeline does not match discovery canonical contract")
    binding = _require_sha256(payload.get("binding_sha256"), "binding_sha256")
    if binding != _payload_binding_sha256(payload):
        raise ValueError("binding_sha256 does not match the canonical audit payload")
    _require_sha256(payload.get("execution_challenge"), "execution_challenge")
    _validate_threshold_record(payload.get("thresholds"))
    inputs = _require_object(payload.get("inputs"), "inputs")
    if set(inputs) != set(DISCOVERY_SOURCE_KEYS):
        raise ValueError(
            "inputs must contain exactly " + ", ".join(DISCOVERY_SOURCE_KEYS)
        )
    for key in DISCOVERY_SOURCE_KEYS:
        _validate_source_record(
            inputs.get(key),
            f"inputs.{key}",
            require_hash=True,
        )
    counts = _require_object(payload.get("counts"), "counts")
    input_rows = _require_nonnegative_int(counts.get("input_rows"), "counts.input_rows")
    output_rows = _require_nonnegative_int(counts.get("output_rows"), "counts.output_rows")
    excluded_count = _require_nonnegative_int(
        counts.get("excluded_rows"),
        "counts.excluded_rows",
    )
    survivor_count = _require_nonnegative_int(
        counts.get("survivor_rows"),
        "counts.survivor_rows",
    )
    direct_exact_count = _require_nonnegative_int(
        counts.get("direct_exact_count"),
        "counts.direct_exact_count",
    )
    source_counts: dict[str, int] = {}
    for key in (
        "input_source_count",
        "training_seq_source_count",
        "training_ligand_source_count",
        "training_holo_source_count",
        "direct_exact_source_count",
        "leakage_audit_source_count",
    ):
        source_counts[key] = _require_nonnegative_int(counts.get(key), f"counts.{key}")
    for key in (
        "input_source_count",
        "training_seq_source_count",
        "training_ligand_source_count",
        "training_holo_source_count",
        "direct_exact_source_count",
        "leakage_audit_source_count",
    ):
        if source_counts[key] < 1:
            raise ValueError(f"counts.{key} must be positive")
    if payload.get("direct_exact_count") != direct_exact_count:
        raise ValueError("direct_exact_count must match counts.direct_exact_count")
    incomplete_audit_count = _require_nonnegative_int(
        counts.get("incomplete_audit_count"),
        "counts.incomplete_audit_count",
    )
    if payload.get("incomplete_audit_count") != incomplete_audit_count:
        raise ValueError(
            "incomplete_audit_count must match counts.incomplete_audit_count"
        )
    neutral_exclusion_count = _require_nonnegative_int(
        payload.get("neutral_exclusion_count"),
        "neutral_exclusion_count",
    )
    if neutral_exclusion_count != excluded_count:
        raise ValueError(
            "neutral_exclusion_count must match counts.excluded_rows"
        )
    survivors = _require_list(payload.get("survivors"), "survivors")
    excluded = _require_list(payload.get("excluded_rows"), "excluded_rows")
    bindings = _require_list(payload.get("input_bindings"), "input_bindings")
    if output_rows != len(bindings):
        raise ValueError("counts.output_rows must match input_bindings length")
    if survivor_count != len(survivors):
        raise ValueError("counts.survivor_rows must match survivors length")
    if excluded_count != len(excluded):
        raise ValueError("counts.excluded_rows must match excluded_rows length")
    if input_rows != output_rows:
        raise ValueError("counts.input_rows must match counts.output_rows")
    if status == "ok" and direct_exact_count != 0:
        raise ValueError("status=ok requires direct_exact_count=0")
    observed_incomplete_audit_count = sum(
        "incomplete_audit"
        in [
            str(reason)
            for reason in _require_list(
                _require_object(row, "excluded_rows[]").get("reasons"),
                "excluded_rows[].reasons",
            )
        ]
        for row in excluded
    )
    if incomplete_audit_count != observed_incomplete_audit_count:
        raise ValueError("incomplete_audit_count must match excluded_rows")
    if status == "ok" and incomplete_audit_count != 0:
        raise ValueError("status=ok requires incomplete_audit_count=0")
    if status == "ok" and survivor_count <= 0:
        raise ValueError("status=ok requires counts.survivor_rows > 0")
    if (
        status == "failed"
        and direct_exact_count == 0
        and incomplete_audit_count == 0
        and survivor_count > 0
    ):
        raise ValueError(
            "status=failed requires a direct match, incomplete audit, or zero survivors"
        )
    observed_keys: set[str] = set()
    survivor_keys: set[str] = set()
    for label, rows in (("survivors", survivors), ("input_bindings", bindings)):
        for idx, row in enumerate(rows):
            obj = _require_object(row, f"{label}[{idx}]")
            row_key = _require_sha256(
                obj.get("discovery_key_sha256"),
                f"{label}[{idx}].discovery_key_sha256",
            )
            observed_keys.add(row_key)
            if label == "survivors":
                survivor_keys.add(row_key)
            if label == "input_bindings":
                input_type = _require_nonempty_text(
                    obj.get("input_type"),
                    f"{label}[{idx}].input_type",
                )
                if input_type not in {"smiles", "sdf"}:
                    raise ValueError(f"{label}[{idx}].input_type must be smiles or sdf")
                if input_type == "smiles":
                    _require_nonempty_text(
                        obj.get("input_smiles"),
                        f"{label}[{idx}].input_smiles",
                    )
                if input_type == "sdf":
                    _require_nonempty_text(
                        obj.get("input_sdf"),
                        f"{label}[{idx}].input_sdf",
                    )
            _require_nonempty_text(obj.get("uniprot"), f"{label}[{idx}].uniprot")
            _require_nonempty_text(
                obj.get("parent_canonical_smiles"),
                f"{label}[{idx}].parent_canonical_smiles",
            )
            _require_nonempty_text(
                obj.get("parent_inchikey"),
                f"{label}[{idx}].parent_inchikey",
            )
            _require_nonempty_text(
                obj.get("parent_connectivity_inchikey"),
                f"{label}[{idx}].parent_connectivity_inchikey",
            )
    for idx, row in enumerate(excluded):
        obj = _require_object(row, f"excluded_rows[{idx}]")
        reasons = _require_list(obj.get("reasons"), f"excluded_rows[{idx}].reasons")
        if not reasons or not all(isinstance(reason, str) and reason for reason in reasons):
            raise ValueError(f"excluded_rows[{idx}].reasons must contain strings")
    if expected_input_key is not None and expected_input_key not in observed_keys:
        raise ValueError("Discovery leakage audit does not bind the active input compound")
    if expected_input_key is not None and expected_input_key not in survivor_keys:
        raise ValueError("Discovery leakage audit does not contain a survivor for the active input compound")


def _csv_bool(value: object, label: str) -> bool:
    if pd.isna(value):
        raise ValueError(f"{label} must be boolean")
    if isinstance(value, bool) or type(value).__name__ == "bool_":
        return bool(value)
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"{label} must be boolean")


def _read_bound_csv(path: Path, label: str) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"{label} failed to parse: {path}: {exc}") from exc
    if frame.empty:
        raise ValueError(f"{label} contains no rows: {path}")
    return frame


def validate_sealed_discovery_audit_sources(
    payload: dict[str, object],
    *,
    expected_sources: dict[str, Path] | None = None,
    expected_thresholds: Thresholds | None = None,
    expected_execution_challenge: str | None = None,
    require_direct_exact_reference: bool = True,
    direct_exact_manifest: Path | None = None,
) -> None:
    """Validate a sealed audit against the active files and row-level evidence."""
    validate_sealed_discovery_audit(payload)
    inputs = _require_object(payload.get("inputs"), "inputs")
    if expected_sources is None:
        sources = {
            key: Path(
                _require_nonempty_text(
                    _require_object(inputs[key], f"inputs.{key}").get("path"),
                    f"inputs.{key}.path",
                )
            )
            for key in DISCOVERY_SOURCE_KEYS
        }
    else:
        if set(expected_sources) != set(DISCOVERY_SOURCE_KEYS):
            raise ValueError(
                "expected_sources must contain exactly "
                + ", ".join(DISCOVERY_SOURCE_KEYS)
            )
        sources = {key: Path(expected_sources[key]) for key in DISCOVERY_SOURCE_KEYS}

    if require_direct_exact_reference:
        try:
            _require_supplied_nonempty_file(
                sources["direct_exact_reference"],
                "Direct exact reference",
            )
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc

    for key, path in sources.items():
        recorded = _require_object(inputs.get(key), f"inputs.{key}")
        active = _source_record(path)
        if recorded != active:
            raise ValueError(
                f"active source record mismatch for inputs.{key}: "
                f"recorded={recorded}, active={active}"
            )

    recorded_thresholds = _validate_threshold_record(payload.get("thresholds"))
    if expected_thresholds is not None:
        expected = _threshold_record(expected_thresholds)
        if recorded_thresholds != expected:
            raise ValueError(
                "Discovery leakage audit thresholds do not match the active run"
            )
    if expected_execution_challenge is not None:
        expected_challenge = _require_sha256(
            expected_execution_challenge,
            "expected_execution_challenge",
        )
        if payload.get("execution_challenge") != expected_challenge:
            raise ValueError(
                "Discovery leakage audit execution_challenge does not match the active run"
            )

    input_frame = _read_bound_csv(sources["input_csv"], "Leakage audit input")
    output_frame = _read_bound_csv(
        sources["leakage_audit_csv"],
        "Leakage audit output",
    )
    input_required = {"uniprot", "smiles"}
    output_required = {
        "uniprot",
        "smiles",
        "parent_canonical_smiles",
        "discovery_key_sha256",
        "parent_inchikey",
        "parent_connectivity_inchikey",
        "seq_id",
        "ligand_tanimoto",
        "pocket_sucos",
        "audit_status",
        "leak_flag",
        "missing_axes",
    }
    for frame, required, label in (
        (input_frame, input_required, "Leakage audit input"),
        (output_frame, output_required, "Leakage audit output"),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{label} missing required column(s): {', '.join(missing)}")
    if len(input_frame) != len(output_frame):
        raise ValueError("Leakage audit input and output row counts do not match")

    counts = _require_object(payload.get("counts"), "counts")
    if int(counts["input_rows"]) != len(input_frame):
        raise ValueError("counts.input_rows does not match the active input CSV")
    if int(counts["output_rows"]) != len(output_frame):
        raise ValueError("counts.output_rows does not match the active audit CSV")

    bindings = _require_list(payload.get("input_bindings"), "input_bindings")
    binding_by_row: dict[int, dict[str, object]] = {}
    for item in bindings:
        binding = _require_object(item, "input_bindings[]")
        row_index = _require_nonnegative_int(
            binding.get("row_index"),
            "input_bindings[].row_index",
        )
        if row_index in binding_by_row:
            raise ValueError(f"input_bindings contains duplicate row_index {row_index}")
        binding_by_row[row_index] = binding
    if set(binding_by_row) != set(range(len(output_frame))):
        raise ValueError("input_bindings row_index values do not cover the active audit CSV")

    if direct_exact_manifest is None:
        direct_keys = _direct_exact_keys(sources["direct_exact_reference"])
    else:
        direct_keys = set()
        validate_direct_exact_manifest(
            sources["direct_exact_reference"],
            direct_exact_manifest,
            stored_keys_out=direct_keys,
        )
    expected_reasons: dict[int, list[str]] = {}
    for row_index in range(len(output_frame)):
        input_row = input_frame.iloc[row_index]
        output_row = output_frame.iloc[row_index]
        binding = binding_by_row[row_index]
        input_uniprot = str(input_row["uniprot"]).strip()
        output_uniprot = str(output_row["uniprot"]).strip()
        if input_uniprot != output_uniprot or binding.get("uniprot") != input_uniprot:
            raise ValueError(f"row {row_index} UniProt binding mismatch")
        try:
            key = discovery_key(
                input_row["smiles"],
                label=f"bound leakage input row {row_index} smiles",
            )
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        expected_identity = {
            "discovery_key_sha256": key.discovery_key_sha256,
            "parent_canonical_smiles": key.parent_canonical_smiles,
            "parent_inchikey": key.parent_inchikey,
            "parent_connectivity_inchikey": key.parent_connectivity_inchikey,
        }
        for field, expected_value in expected_identity.items():
            if str(binding.get(field)) != expected_value:
                raise ValueError(f"row {row_index} input binding mismatch for {field}")
            if str(output_row[field]) != expected_value:
                raise ValueError(f"row {row_index} audit output mismatch for {field}")
        output_key = discovery_key(
            output_row["smiles"],
            label=f"bound leakage output row {row_index} smiles",
        )
        if output_key.discovery_key_sha256 != key.discovery_key_sha256:
            raise ValueError(f"row {row_index} input/output compound mismatch")
        seq_id = _parse_audit_score(output_row["seq_id"], f"row {row_index} seq_id")
        ligand_tanimoto = _parse_audit_score(
            output_row["ligand_tanimoto"],
            f"row {row_index} ligand_tanimoto",
        )
        pocket_sucos_value = _parse_audit_score(
            output_row["pocket_sucos"],
            f"row {row_index} pocket_sucos",
        )
        expected_missing_axes = _expected_missing_axes(
            seq_id,
            ligand_tanimoto,
            pocket_sucos_value,
        )
        observed_missing_axes = _parse_missing_axes(
            output_row["missing_axes"],
            f"row {row_index} missing_axes",
        )
        if observed_missing_axes != expected_missing_axes:
            raise ValueError(f"row {row_index} missing_axes mismatch")
        expected_status = "incomplete" if expected_missing_axes else "ok"
        observed_status = str(output_row["audit_status"]).strip().lower()
        if observed_status != expected_status:
            raise ValueError(f"row {row_index} audit_status mismatch")
        expected_leak = _expected_leak_flag(
            seq_id,
            ligand_tanimoto,
            pocket_sucos_value,
            recorded_thresholds,
        )
        observed_leak = _csv_bool(output_row["leak_flag"], f"row {row_index} leak_flag")
        if observed_leak != expected_leak:
            raise ValueError(f"row {row_index} leak_flag mismatch")
        reasons: list[str] = []
        if observed_status != "ok":
            reasons.append("incomplete_audit")
        if observed_leak:
            reasons.append("leakage_axis_flag")
        if key.discovery_key_sha256 in direct_keys:
            reasons.append("direct_exact_match")
        expected_reasons[row_index] = reasons

    survivor_rows = {
        _require_nonnegative_int(
            _require_object(item, "survivors[]").get("row_index"),
            "survivors[].row_index",
        )
        for item in _require_list(payload.get("survivors"), "survivors")
    }
    excluded_rows: dict[int, list[str]] = {}
    for item in _require_list(payload.get("excluded_rows"), "excluded_rows"):
        excluded = _require_object(item, "excluded_rows[]")
        row_index = _require_nonnegative_int(
            excluded.get("row_index"),
            "excluded_rows[].row_index",
        )
        if row_index in excluded_rows:
            raise ValueError(f"excluded_rows contains duplicate row_index {row_index}")
        excluded_rows[row_index] = [
            str(reason)
            for reason in _require_list(excluded.get("reasons"), "excluded_rows[].reasons")
        ]
    expected_survivors = {
        row_index for row_index, reasons in expected_reasons.items() if not reasons
    }
    expected_excluded = {
        row_index: reasons for row_index, reasons in expected_reasons.items() if reasons
    }
    if survivor_rows != expected_survivors or excluded_rows != expected_excluded:
        raise ValueError("survivor/exclusion rows do not match the active audit CSV")
    direct_exact_count = sum(
        "direct_exact_match" in reasons for reasons in expected_reasons.values()
    )
    if payload.get("direct_exact_count") != direct_exact_count:
        raise ValueError("direct_exact_count does not match the active direct reference")
    incomplete_audit_count = sum(
        "incomplete_audit" in reasons for reasons in expected_reasons.values()
    )
    if payload.get("incomplete_audit_count") != incomplete_audit_count:
        raise ValueError("incomplete_audit_count does not match the active audit CSV")
    expected_status = (
        "ok"
        if direct_exact_count == 0
        and incomplete_audit_count == 0
        and len(expected_survivors) > 0
        else "failed"
    )
    if payload.get("status") != expected_status:
        raise ValueError("status does not match the active audit exclusions")

    source_count_fields = {
        "input_source_count": "input_csv",
        "training_seq_source_count": "training_seq_db",
        "training_ligand_source_count": "training_ligands",
        "training_holo_source_count": "training_holo",
        "direct_exact_source_count": "direct_exact_reference",
        "leakage_audit_source_count": "leakage_audit_csv",
    }
    for count_field, source_key in source_count_fields.items():
        if counts.get(count_field) != _source_count(sources[source_key]):
            raise ValueError(
                f"counts.{count_field} does not match active source {source_key}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path,
                        help="CSV with at minimum {uniprot, smiles}")
    parser.add_argument("--training-seq-db",
                        default=Path("data/mmseqs/training_cutoff_db"),
                        type=Path)
    parser.add_argument("--training-ligands",
                        default=Path("data/chembl37/training_ligands.smi"),
                        type=Path)
    parser.add_argument("--training-holo",
                        default=Path("data/plinder/training_holo_pockets.csv"),
                        type=Path)
    parser.add_argument("--seq-id-threshold", type=float, default=0.30)
    parser.add_argument("--ligand-tanimoto-threshold", type=float, default=0.50)
    parser.add_argument("--pocket-sucos-threshold", type=float, default=0.50)
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Write incomplete rows instead of failing closed.")
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--out-discovery-audit-json", type=Path)
    parser.add_argument("--direct-exact-reference", type=Path)
    parser.add_argument("--direct-exact-manifest", type=Path)
    parser.add_argument(
        "--execution-challenge",
        help="Optional 64-hex freshness challenge supplied by the caller.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.out_csv.exists():
        args.out_csv.unlink()
    if args.out_discovery_audit_json and args.out_discovery_audit_json.exists():
        args.out_discovery_audit_json.unlink()
    if args.out_discovery_audit_json is not None and args.direct_exact_reference is None:
        parser.error(
            "--direct-exact-reference is required with "
            "--out-discovery-audit-json"
        )
    manifest_validated_direct_keys: set[str] | None = None
    if args.direct_exact_manifest is not None:
        if args.direct_exact_reference is None:
            parser.error("--direct-exact-manifest requires --direct-exact-reference")
        if args.out_discovery_audit_json is not None:
            manifest_validated_direct_keys = set()
        try:
            validate_direct_exact_manifest(
                args.direct_exact_reference,
                args.direct_exact_manifest,
                stored_keys_out=manifest_validated_direct_keys,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    try:
        rows = _read_required_csv(args.input_csv, {"uniprot", "smiles"}, "Leakage audit input")
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    thresholds = Thresholds(args.seq_id_threshold,
                            args.ligand_tanimoto_threshold,
                            args.pocket_sucos_threshold)
    try:
        out = audit(rows, args.training_seq_db, args.training_ligands,
                    args.training_holo, thresholds,
                    allow_incomplete=args.allow_incomplete)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv_atomic(out, args.out_csv)
    if args.out_discovery_audit_json is not None:
        try:
            sealed = sealed_discovery_audit(
                rows,
                out,
                input_csv=args.input_csv,
                training_seq_db=args.training_seq_db,
                training_ligands=args.training_ligands,
                training_holo=args.training_holo,
                direct_exact_reference=args.direct_exact_reference,
                out_csv=args.out_csv,
                thresholds=thresholds,
                execution_challenge=(
                    args.execution_challenge or secrets.token_hex(32)
                ),
                manifest_validated_direct_keys=manifest_validated_direct_keys,
            )
        except (RuntimeError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        _write_json_atomic(sealed, args.out_discovery_audit_json)
        if sealed["status"] != "ok":
            raise SystemExit(
                "Discovery leakage audit failed; direct_exact_count="
                f"{sealed['direct_exact_count']}"
            )
    LOG.info("Wrote %s (n=%d, leaks=%d)",
             args.out_csv, len(out), int(out["leak_flag"].sum()))


if __name__ == "__main__":
    main()
