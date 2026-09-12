#!/usr/bin/env python3
"""Build deterministic Discovery alias source parquets from mirrored upstream files."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import sqlite3
import sys
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from mirror_gtopdb import EXPECTED_HEADERS as GTOPDB_EXPECTED_HEADERS


SOURCE_SCHEMA_VERSION = "skinscout.discovery-alias-sources.v1"
REGISTRY_SCHEMA_VERSION = "discovery_alias_source_registry.v1"
PUBCHEM_SOURCE_SCHEMA_VERSION = "skinscout.pubchem-alias-source.v1"
CHEMBL_SOURCE_SCHEMA_VERSION = "chembl_activity_evidence.v1"
BINDINGDB_SOURCE_SCHEMA_VERSION = 1
GTOPDB_SOURCE_SCHEMA_VERSION = "skinscout.gtopdb-source.v1"
OUTPUT_COLUMNS = ("smiles", "alias", "inchikey", "source_record_id")
OUTPUT_SCHEMA = pa.schema([(column, pa.string()) for column in OUTPUT_COLUMNS])
OUTPUT_FILES = {
    "ChEMBL": "chembl_aliases.parquet",
    "BindingDB": "bindingdb_aliases.parquet",
    "GtoPdb": "gtopdb_aliases.parquet",
    "PubChem": "pubchem_aliases.parquet",
}
SOURCE_ORDER = ("ChEMBL", "BindingDB", "GtoPdb", "PubChem")
GTOPDB_HEADER = GTOPDB_EXPECTED_HEADERS["ligands.csv"]
STAGING_INSERT_BATCH_SIZE = 10_000
PARQUET_EXPORT_BATCH_SIZE = 50_000
SOURCE_PROGRESS_INTERVAL = 250_000
PUBCHEM_PROGRESS_INTERVAL = 5_000_000
PUBCHEM_MISSING_CIDS_FILE = "pubchem_missing_cids.txt"
MAX_MISSING_PUBCHEM_CID_FRACTION_PPM = 100_000
SPDXISH = re.compile(r"^[A-Za-z0-9.+-]+$")
KNOWN_LICENSE_IDS = {
    "CC BY 3.0": "CC-BY-3.0",
    "CC BY-SA 3.0": "CC-BY-SA-3.0",
}


class AliasSourceError(ValueError):
    """Raised when a mirrored input or generated output violates the contract."""


def _progress(message: str) -> None:
    print(f"[stage0.discovery-aliases] {message}", file=sys.stderr, flush=True)


def _clean(value: object) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return "" if text.casefold() in {"", "na", "n/a", "nan", "none", "null"} else text


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash_payload(payload: object) -> str:
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _relative_path(path: Path, base: Path) -> str:
    return os.path.relpath(path.resolve(), start=base.resolve())


def _require_file(path: Path, label: str) -> None:
    if path.is_symlink():
        raise AliasSourceError(f"{label} must not be a symlink: {path}")
    if not path.is_file():
        raise AliasSourceError(f"{label} is missing or not a file: {path}")
    if path.stat().st_size <= 0:
        raise AliasSourceError(f"{label} is empty: {path}")


def _load_json_file(path: Path, label: str) -> dict[str, Any]:
    _require_file(path, label)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AliasSourceError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise AliasSourceError(f"{label} must be a JSON object: {path}")
    return payload


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AliasSourceError(f"{label} must be an object")
    return value


def _path_binds_artifact(declared: object, manifest_path: Path, artifact_path: Path) -> bool:
    text = _clean(declared)
    if not text:
        return False
    declared_path = Path(text)
    artifact_resolved = artifact_path.resolve()
    candidates = [declared_path.resolve()] if declared_path.is_absolute() else [
        (manifest_path.parent / declared_path).resolve(),
        (Path.cwd() / declared_path).resolve(),
    ]
    return artifact_resolved in candidates


def _require_artifact_binding(
    *,
    source_name: str,
    manifest_path: Path,
    artifact_path: Path,
    record: Mapping[str, Any],
    path_key: str = "path",
    sha_key: str = "sha256",
    bytes_key: str = "bytes",
) -> None:
    if not _path_binds_artifact(record.get(path_key), manifest_path, artifact_path):
        raise AliasSourceError(f"{source_name} manifest does not bind artifact path")
    artifact_sha = _hash_file(artifact_path)
    artifact_bytes = artifact_path.stat().st_size
    if record.get(sha_key) != artifact_sha:
        raise AliasSourceError(f"{source_name} manifest does not bind artifact sha256")
    if record.get(bytes_key) != artifact_bytes:
        raise AliasSourceError(f"{source_name} manifest does not bind artifact bytes")


def _require_source_metadata(
    *,
    source_name: str,
    source: Mapping[str, Any],
    release: object,
    release_date: object,
    expected_release_date: str | None = None,
) -> dict[str, str]:
    if source.get("name") != source_name:
        raise AliasSourceError(f"{source_name} manifest source.name mismatch")
    license_text = _clean(source.get("license"))
    if not license_text:
        raise AliasSourceError(f"{source_name} manifest does not declare a license")
    release_text = _clean(release)
    if not release_text:
        raise AliasSourceError(f"{source_name} manifest does not declare a release")
    release_date_text = _clean(release_date)
    if not release_date_text:
        raise AliasSourceError(f"{source_name} manifest does not declare a release_date")
    if expected_release_date is not None and release_date_text != expected_release_date:
        raise AliasSourceError(
            f"{source_name} manifest release_date must be {expected_release_date}"
        )
    if source.get("redistribution") != "allowed":
        raise AliasSourceError(f"{source_name} manifest does not allow redistribution")
    license_url = _clean(source.get("license_url"))
    if not license_url.startswith("https://"):
        raise AliasSourceError(f"{source_name} manifest license_url must use https")
    return {
        "release": release_text,
        "release_date": release_date_text,
        "license": license_text,
        "license_url": license_url,
    }


def _manifest_meta(
    manifest_path: Path,
    source_meta: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "path": str(manifest_path.resolve()),
        "sha256": _hash_file(manifest_path),
        "bytes": manifest_path.stat().st_size,
        **dict(source_meta),
    }


def _validate_chembl_manifest(
    manifest_path: Path, artifact_path: Path, release_date: str
) -> dict[str, Any]:
    payload = _load_json_file(manifest_path, "ChEMBL manifest")
    if payload.get("schema_version") != CHEMBL_SOURCE_SCHEMA_VERSION:
        raise AliasSourceError(
            f"ChEMBL manifest schema_version must be {CHEMBL_SOURCE_SCHEMA_VERSION}"
        )
    source = _require_mapping(payload.get("source"), "ChEMBL manifest source")
    _require_artifact_binding(
        source_name="ChEMBL",
        manifest_path=manifest_path,
        artifact_path=artifact_path,
        record=source,
        path_key="source_db",
        sha_key="source_db_sha256",
        bytes_key="source_db_bytes",
    )
    meta = _require_source_metadata(
        source_name="ChEMBL",
        source=source,
        release=source.get("release"),
        release_date=source.get("release_date"),
        expected_release_date=release_date,
    )
    return _manifest_meta(manifest_path, meta)


def _validate_bindingdb_manifest(
    manifest_path: Path, artifact_path: Path, release_date: str
) -> dict[str, Any]:
    payload = _load_json_file(manifest_path, "BindingDB manifest")
    if payload.get("schema_version") != BINDINGDB_SOURCE_SCHEMA_VERSION:
        raise AliasSourceError(
            f"BindingDB manifest schema_version must be {BINDINGDB_SOURCE_SCHEMA_VERSION}"
        )
    source = _require_mapping(payload.get("source"), "BindingDB manifest source")
    extracted = _require_mapping(payload.get("extracted"), "BindingDB manifest extracted")
    _require_artifact_binding(
        source_name="BindingDB",
        manifest_path=manifest_path,
        artifact_path=artifact_path,
        record=extracted,
    )
    meta = _require_source_metadata(
        source_name="BindingDB",
        source=source,
        release=payload.get("release"),
        release_date=payload.get("release_date"),
        expected_release_date=release_date,
    )
    return _manifest_meta(manifest_path, meta)


def _validate_gtopdb_manifest(manifest_path: Path, artifact_path: Path) -> dict[str, Any]:
    payload = _load_json_file(manifest_path, "GtoPdb manifest")
    if payload.get("schema_version") != GTOPDB_SOURCE_SCHEMA_VERSION:
        raise AliasSourceError(
            f"GtoPdb manifest schema_version must be {GTOPDB_SOURCE_SCHEMA_VERSION}"
        )
    source = _require_mapping(payload.get("source"), "GtoPdb manifest source")
    artifacts = _require_mapping(payload.get("artifacts"), "GtoPdb manifest artifacts")
    ligand_artifact = _require_mapping(
        artifacts.get("ligands.csv"), "GtoPdb manifest artifacts['ligands.csv']"
    )
    _require_artifact_binding(
        source_name="GtoPdb",
        manifest_path=manifest_path,
        artifact_path=artifact_path,
        record=ligand_artifact,
    )
    meta = _require_source_metadata(
        source_name="GtoPdb",
        source=source,
        release=source.get("release"),
        release_date=source.get("release_date"),
    )
    return _manifest_meta(manifest_path, meta)


def _validate_pubchem_manifest(manifest_path: Path, artifact_path: Path) -> dict[str, Any]:
    payload = _load_json_file(manifest_path, "PubChem manifest")
    if payload.get("schema_version") != PUBCHEM_SOURCE_SCHEMA_VERSION:
        raise AliasSourceError(
            f"PubChem manifest schema_version must be {PUBCHEM_SOURCE_SCHEMA_VERSION}"
        )
    source = _require_mapping(payload.get("source"), "PubChem manifest source")
    artifact = _require_mapping(payload.get("artifact"), "PubChem manifest artifact")
    _require_artifact_binding(
        source_name="PubChem",
        manifest_path=manifest_path,
        artifact_path=artifact_path,
        record=artifact,
    )
    meta = _require_source_metadata(
        source_name="PubChem",
        source=source,
        release=source.get("release"),
        release_date=source.get("release_date"),
    )
    return _manifest_meta(manifest_path, meta)


def _registry_license(source: str, license_text: str) -> str:
    text = license_text.strip()
    if text in KNOWN_LICENSE_IDS:
        return KNOWN_LICENSE_IDS[text]
    if SPDXISH.fullmatch(text):
        return text
    suffix = re.sub(r"[^A-Za-z0-9.+-]+", "-", source).strip("-")
    return f"LicenseRef-{suffix}"


class AliasStaging:
    """Disk-backed dedupe and selected-CID staging for bounded source builds."""

    def __init__(self, db_path: Path) -> None:
        self.con = sqlite3.connect(db_path)
        self.con.executescript(
            """
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            CREATE TABLE aliases (
                source TEXT NOT NULL,
                smiles TEXT NOT NULL,
                alias TEXT NOT NULL,
                inchikey TEXT NOT NULL,
                source_record_id TEXT NOT NULL,
                UNIQUE(source, smiles, alias, inchikey, source_record_id)
            );
            CREATE TABLE pubchem_cids (
                cid TEXT PRIMARY KEY,
                matched INTEGER NOT NULL DEFAULT 0
            );
            """
        )

    def close(self) -> None:
        self.con.close()

    def add_aliases(self, source: str, rows: list[tuple[str, str, str, str]]) -> None:
        if not rows:
            return
        self.con.executemany(
            """
            INSERT OR IGNORE INTO aliases
                (source, smiles, alias, inchikey, source_record_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            [(source, *row) for row in rows],
        )
        self.con.commit()
        rows.clear()

    def add_cids(self, cids: list[str]) -> None:
        if not cids:
            return
        self.con.executemany(
            "INSERT OR IGNORE INTO pubchem_cids (cid) VALUES (?)",
            [(cid,) for cid in cids],
        )
        self.con.commit()
        cids.clear()

    def iter_selected_cids(self) -> Iterable[str]:
        for (cid,) in self.con.execute(
            "SELECT cid FROM pubchem_cids ORDER BY LENGTH(cid), cid"
        ):
            if not re.fullmatch(r"[0-9]+", cid):
                raise AliasSourceError(f"PubChem CID is not an unsigned decimal integer: {cid}")
            yield str(cid)

    def mark_cids_matched(self, cids: list[str]) -> None:
        if not cids:
            return
        self.con.executemany(
            "UPDATE pubchem_cids SET matched = 1 WHERE cid = ?",
            [(cid,) for cid in cids],
        )
        self.con.commit()
        cids.clear()

    def selected_cid_count(self) -> int:
        return int(self.con.execute("SELECT COUNT(*) FROM pubchem_cids").fetchone()[0])

    def matched_cid_count(self) -> int:
        return int(
            self.con.execute("SELECT COUNT(*) FROM pubchem_cids WHERE matched = 1").fetchone()[0]
        )

    def missing_cids(self, limit: int = 10) -> list[str]:
        return [
            row[0]
            for row in self.con.execute(
                "SELECT cid FROM pubchem_cids WHERE matched = 0 ORDER BY cid LIMIT ?",
                (limit,),
            )
        ]

    def missing_cid_count(self) -> int:
        return int(
            self.con.execute(
                "SELECT COUNT(*) FROM pubchem_cids WHERE matched = 0"
            ).fetchone()[0]
        )

    def iter_missing_cids(self) -> Iterable[str]:
        for (cid,) in self.con.execute(
            """
            SELECT cid
            FROM pubchem_cids
            WHERE matched = 0
            ORDER BY LENGTH(cid), cid
            """
        ):
            yield str(cid)

    def row_count(self, source: str) -> int:
        return int(
            self.con.execute(
                "SELECT COUNT(*) FROM aliases WHERE source = ?",
                (source,),
            ).fetchone()[0]
        )


def _add_alias(
    rows: list[tuple[str, str, str, str]],
    *,
    smiles: str,
    alias: object,
    inchikey: object = "",
    source_record_id: object,
) -> None:
    clean_smiles = _clean(smiles)
    clean_alias = _clean(alias)
    clean_record_id = _clean(source_record_id)
    if not clean_smiles:
        raise AliasSourceError("encountered row with blank required SMILES")
    if clean_alias:
        rows.append((clean_smiles, clean_alias, _clean(inchikey), clean_record_id))


def _flush_aliases(staging: AliasStaging, source: str, rows: list[tuple[str, str, str, str]]) -> None:
    if len(rows) >= STAGING_INSERT_BATCH_SIZE:
        staging.add_aliases(source, rows)


def _flush_cids(staging: AliasStaging, cids: list[str]) -> None:
    if len(cids) >= STAGING_INSERT_BATCH_SIZE:
        staging.add_cids(cids)


def _flush_matched_cids(staging: AliasStaging, cids: list[str]) -> None:
    if len(cids) >= STAGING_INSERT_BATCH_SIZE:
        staging.mark_cids_matched(cids)


def _read_chembl(db_path: Path, staging: AliasStaging) -> None:
    _require_file(db_path, "ChEMBL SQLite")
    uri = f"file:{db_path.resolve()}?mode=ro"
    try:
        con = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as exc:
        raise AliasSourceError(f"failed to open ChEMBL SQLite read-only: {db_path}") from exc
    rows: list[tuple[str, str, str, str]] = []
    scanned = 0
    try:
        required = {
            "compound_structures": {"molregno", "canonical_smiles"},
            "molecule_dictionary": {"molregno", "chembl_id"},
        }
        for table, columns in required.items():
            got = {
                row[1]
                for row in con.execute(f"PRAGMA table_info({table})").fetchall()
            }
            missing = sorted(columns - got)
            if missing:
                raise AliasSourceError(f"ChEMBL table {table} missing columns {missing}")
        struct_columns = {
            row[1] for row in con.execute("PRAGMA table_info(compound_structures)").fetchall()
        }
        dict_columns = {
            row[1] for row in con.execute("PRAGMA table_info(molecule_dictionary)").fetchall()
        }
        inchikey_expr = (
            "COALESCE(cs.standard_inchi_key, '')"
            if "standard_inchi_key" in struct_columns
            else "''"
        )
        pref_name_expr = "COALESCE(md.pref_name, '')" if "pref_name" in dict_columns else "''"
        for smiles, inchikey, chembl_id, pref_name in con.execute(
            f"""
            SELECT cs.canonical_smiles, {inchikey_expr}, md.chembl_id,
                   {pref_name_expr}
            FROM compound_structures cs
            JOIN molecule_dictionary md ON md.molregno = cs.molregno
            WHERE TRIM(COALESCE(cs.canonical_smiles, '')) != ''
              AND TRIM(COALESCE(md.chembl_id, '')) != ''
            """
        ):
            scanned += 1
            _add_alias(
                rows,
                smiles=smiles,
                alias=chembl_id,
                inchikey=inchikey,
                source_record_id=chembl_id,
            )
            _flush_aliases(staging, "ChEMBL", rows)
            if scanned % SOURCE_PROGRESS_INTERVAL == 0:
                _progress(f"ChEMBL molecules scanned={scanned:,}")
            _add_alias(
                rows,
                smiles=smiles,
                alias=pref_name,
                inchikey=inchikey,
                source_record_id=chembl_id,
            )
            _flush_aliases(staging, "ChEMBL", rows)
        tables = {
            row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if "molecule_synonyms" in tables:
            syn_columns = {
                row[1] for row in con.execute("PRAGMA table_info(molecule_synonyms)").fetchall()
            }
            if {"molregno", "synonyms"} <= syn_columns:
                for smiles, inchikey, chembl_id, synonym in con.execute(
                    f"""
                    SELECT cs.canonical_smiles, {inchikey_expr}, md.chembl_id, ms.synonyms
                    FROM molecule_synonyms ms
                    JOIN compound_structures cs ON cs.molregno = ms.molregno
                    JOIN molecule_dictionary md ON md.molregno = ms.molregno
                    WHERE TRIM(COALESCE(cs.canonical_smiles, '')) != ''
                      AND TRIM(COALESCE(md.chembl_id, '')) != ''
                    """
                ):
                    scanned += 1
                    _add_alias(
                        rows,
                        smiles=smiles,
                        alias=synonym,
                        inchikey=inchikey,
                        source_record_id=chembl_id,
                    )
                    _flush_aliases(staging, "ChEMBL", rows)
                    if scanned % SOURCE_PROGRESS_INTERVAL == 0:
                        _progress(f"ChEMBL rows scanned={scanned:,}")
    except sqlite3.Error as exc:
        raise AliasSourceError(f"failed to read ChEMBL SQLite: {db_path}") from exc
    finally:
        con.close()
    staging.add_aliases("ChEMBL", rows)
    if staging.row_count("ChEMBL") <= 0:
        raise AliasSourceError("ChEMBL produced no alias rows")
    _progress(
        f"ChEMBL complete scanned={scanned:,} aliases={staging.row_count('ChEMBL'):,}"
    )


def _pick(row: Mapping[str, str], names: Iterable[str]) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and _clean(value):
            return _clean(value)
    return ""


def _split_aliases(value: object) -> list[str]:
    return [_clean(part) for part in str(value or "").split("|") if _clean(part)]


def _read_bindingdb(tsv_path: Path, staging: AliasStaging) -> dict[str, int]:
    _require_file(tsv_path, "BindingDB TSV")
    rows: list[tuple[str, str, str, str]] = []
    pubchem_cids: list[str] = []
    scanned_rows = 0
    missing_smiles_rows = 0
    try:
        with tsv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not reader.fieldnames:
                raise AliasSourceError("BindingDB TSV missing header")
            for line_number, row in enumerate(reader, start=2):
                scanned_rows += 1
                smiles = _pick(row, ("Ligand SMILES", "Ligand SMILES String"))
                if not smiles:
                    missing_smiles_rows += 1
                    continue
                inchikey = _pick(row, ("Ligand InChI Key", "Ligand InChIKey"))
                monomer_id = _pick(row, ("BindingDB MonomerID", "BindingDB Reactant_set_id"))
                record_id = monomer_id or f"BindingDB:{line_number}"
                alias_values = [
                    _pick(row, ("BindingDB Ligand Name", "BindingDB Ligand Title")),
                    monomer_id,
                    _pick(row, ("PubChem CID", "PubChem CID of Ligand")),
                    _pick(row, ("ChEMBL ID", "ChEMBL ID of Ligand")),
                ]
                for cid in _split_aliases(alias_values[2]):
                    pubchem_cids.append(cid)
                    _flush_cids(staging, pubchem_cids)
                    alias_values.append(f"PubChem CID {cid}")
                for alias in alias_values:
                    for value in _split_aliases(alias) or [_clean(alias)]:
                        _add_alias(
                            rows,
                            smiles=smiles,
                            alias=value,
                            inchikey=inchikey,
                            source_record_id=record_id,
                        )
                        _flush_aliases(staging, "BindingDB", rows)
                if (line_number - 1) % SOURCE_PROGRESS_INTERVAL == 0:
                    _progress(f"BindingDB rows scanned={line_number - 1:,}")
    except OSError as exc:
        raise AliasSourceError(f"failed to read BindingDB TSV: {tsv_path}") from exc
    staging.add_cids(pubchem_cids)
    staging.add_aliases("BindingDB", rows)
    if staging.row_count("BindingDB") <= 0:
        raise AliasSourceError("BindingDB produced no alias rows")
    _progress(
        "BindingDB complete "
        f"aliases={staging.row_count('BindingDB'):,} selected_pubchem_cids="
        f"{staging.selected_cid_count():,} missing_smiles={missing_smiles_rows:,}"
    )
    return {
        "input_rows": scanned_rows,
        "missing_smiles_rows": missing_smiles_rows,
        "eligible_rows": scanned_rows - missing_smiles_rows,
    }


def _read_gtopdb(csv_path: Path, staging: AliasStaging) -> None:
    _require_file(csv_path, "GtoPdb ligands CSV")
    rows: list[tuple[str, str, str, str]] = []
    pubchem_cids: list[str] = []
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            first = handle.readline()
            parsed_banner = next(csv.reader([first]), [])
            banner = str(parsed_banner[0]).strip() if parsed_banner else ""
            if not banner.startswith("#"):
                raise AliasSourceError("GtoPdb ligands CSV missing metadata banner row")
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != GTOPDB_HEADER:
                raise AliasSourceError("GtoPdb ligands CSV header does not match contract")
            for line_number, row in enumerate(reader, start=3):
                smiles = _clean(row.get("SMILES"))
                ligand_id = _clean(row.get("Ligand ID"))
                if not ligand_id:
                    raise AliasSourceError(f"GtoPdb line {line_number} missing Ligand ID")
                if not smiles:
                    continue
                inchikey = _clean(row.get("InChIKey"))
                cid = _clean(row.get("PubChem CID"))
                if cid:
                    pubchem_cids.append(cid)
                    _flush_cids(staging, pubchem_cids)
                aliases = [
                    row.get("Name"),
                    ligand_id,
                    cid,
                    row.get("ChEMBL ID"),
                    row.get("IUPAC name"),
                    row.get("INN"),
                ]
                aliases.extend(_split_aliases(row.get("Synonyms")))
                if cid:
                    aliases.append(f"PubChem CID {cid}")
                for alias in aliases:
                    _add_alias(
                        rows,
                        smiles=smiles,
                        alias=alias,
                        inchikey=inchikey,
                        source_record_id=f"GtoPdb:{ligand_id}",
                    )
                    _flush_aliases(staging, "GtoPdb", rows)
    except OSError as exc:
        raise AliasSourceError(f"failed to read GtoPdb ligands CSV: {csv_path}") from exc
    staging.add_cids(pubchem_cids)
    staging.add_aliases("GtoPdb", rows)
    if staging.row_count("GtoPdb") <= 0:
        raise AliasSourceError("GtoPdb produced no alias rows")


def _read_pubchem(gz_path: Path, staging: AliasStaging) -> dict[str, int]:
    _require_file(gz_path, "PubChem SMILES gzip")
    if staging.selected_cid_count() <= 0:
        raise AliasSourceError("PubChem CID selection is empty")
    rows: list[tuple[str, str, str, str]] = []
    matched_cids: list[str] = []
    selected = iter(staging.iter_selected_cids())
    selected_cid = next(selected, None)
    previous_numeric_cid = -1
    try:
        with gzip.open(gz_path, "rt", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 2:
                    raise AliasSourceError(f"PubChem line {line_number} is not CID<TAB>SMILES")
                cid, smiles = (_clean(parts[0]), _clean(parts[1]))
                if not re.fullmatch(r"[0-9]+", cid):
                    raise AliasSourceError(
                        f"PubChem line {line_number} CID is not an unsigned decimal integer"
                    )
                numeric_cid = int(cid)
                if numeric_cid <= previous_numeric_cid:
                    raise AliasSourceError("PubChem CIDs must be strictly increasing")
                previous_numeric_cid = numeric_cid
                if line_number % PUBCHEM_PROGRESS_INTERVAL == 0:
                    _progress(
                        f"PubChem rows scanned={line_number:,} "
                        f"matched_cids={staging.matched_cid_count():,}"
                    )

                while selected_cid is not None and int(selected_cid) < numeric_cid:
                    selected_cid = next(selected, None)
                if selected_cid is None:
                    break
                if int(selected_cid) != numeric_cid:
                    continue
                matched_cids.append(selected_cid)
                _flush_matched_cids(staging, matched_cids)
                _add_alias(
                    rows,
                    smiles=smiles,
                    alias=f"PubChem CID {selected_cid}",
                    inchikey="",
                    source_record_id=f"PubChem:{selected_cid}",
                )
                _flush_aliases(staging, "PubChem", rows)
                selected_cid = next(selected, None)
                if selected_cid is None:
                    break
    except (OSError, gzip.BadGzipFile) as exc:
        raise AliasSourceError(f"failed to read PubChem gzip: {gz_path}") from exc
    staging.mark_cids_matched(matched_cids)
    staging.add_aliases("PubChem", rows)
    selected_count = staging.selected_cid_count()
    matched_count = staging.matched_cid_count()
    missing_count = staging.missing_cid_count()
    if matched_count <= 0:
        raise AliasSourceError("PubChem selected CID set matched zero rows")
    if selected_count != matched_count + missing_count:
        raise AliasSourceError("PubChem selected/matched/missing CID counts do not conserve")
    missing_fraction_ppm = (
        (missing_count * 1_000_000 + selected_count - 1) // selected_count
    )
    if missing_fraction_ppm > MAX_MISSING_PUBCHEM_CID_FRACTION_PPM:
        missing = staging.missing_cids()
        raise AliasSourceError(
            "PubChem missing selected CID(s) exceed 10% policy: "
            f"missing={missing_count:,} selected={selected_count:,} "
            f"sample={', '.join(missing)}"
        )
    if staging.row_count("PubChem") <= 0:
        raise AliasSourceError("PubChem produced no alias rows")
    _progress(
        f"PubChem complete matched_cids={matched_count:,} "
        f"missing_cids={missing_count:,}"
    )
    return {
        "selected_cids": selected_count,
        "matched_cids": matched_count,
        "missing_cids": missing_count,
        "missing_fraction_ppm": missing_fraction_ppm,
    }


def _tmp(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def _cleanup_outputs(out_dir: Path) -> None:
    for name in [
        *OUTPUT_FILES.values(),
        PUBCHEM_MISSING_CIDS_FILE,
        "source_registry.json",
        "source_manifest.json",
    ]:
        path = out_dir / name
        path.unlink(missing_ok=True)
        _tmp(path).unlink(missing_ok=True)


def _write_parquet(path: Path, staging: AliasStaging, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp(path)
    tmp.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    try:
        cursor = staging.con.execute(
            """
            SELECT smiles, alias, inchikey, source_record_id
            FROM aliases
            WHERE source = ?
            ORDER BY smiles, alias, inchikey, source_record_id
            """,
            (source,),
        )
        while True:
            rows = cursor.fetchmany(PARQUET_EXPORT_BATCH_SIZE)
            if not rows:
                break
            columns = {
                column: [row[index] for row in rows]
                for index, column in enumerate(OUTPUT_COLUMNS)
            }
            table = pa.Table.from_pydict(columns, schema=OUTPUT_SCHEMA)
            if writer is None:
                writer = pq.ParquetWriter(
                    tmp,
                    OUTPUT_SCHEMA,
                    compression="zstd",
                    use_dictionary=False,
                )
            writer.write_table(table)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        tmp.unlink(missing_ok=True)
        raise AliasSourceError(f"{source} produced no alias rows")
    os.replace(tmp, path)


def _write_missing_cid_audit(
    path: Path,
    staging: AliasStaging,
) -> tuple[str, int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp(path)
    tmp.unlink(missing_ok=True)
    row_count = 0
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for cid in staging.iter_missing_cids():
            handle.write(f"{cid}\n")
            row_count += 1
    os.replace(tmp, path)
    return _hash_file(path), path.stat().st_size, row_count


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp(path)
    tmp.unlink(missing_ok=True)
    tmp.write_text(_stable_json(payload) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _validate_output_parquet(path: Path) -> tuple[str, int, int]:
    _require_file(path, f"output parquet {path.name}")
    try:
        parquet = pq.ParquetFile(path)
        schema = parquet.schema_arrow
        metadata = parquet.metadata
    except Exception as exc:
        raise AliasSourceError(f"output parquet failed to parse: {path}") from exc
    if tuple(schema.names) != OUTPUT_COLUMNS:
        raise AliasSourceError(f"output parquet schema mismatch: {path}")
    for field in schema:
        if not pa.types.is_string(field.type):
            raise AliasSourceError(f"output parquet column {field.name} is not string")
    if metadata.num_rows <= 0:
        raise AliasSourceError(f"output parquet has no rows: {path}")
    return _hash_file(path), path.stat().st_size, metadata.num_rows


def _make_registry(
    *,
    out_dir: Path,
    source_meta: Mapping[str, Mapping[str, Any]],
    artifact_meta: Mapping[str, tuple[str, int, int]],
) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    for name in SOURCE_ORDER:
        sha256, byte_count, row_count = artifact_meta[name]
        meta = source_meta[name]
        sources.append(
            {
                "canonical_name": name,
                "release": meta["release"],
                "release_date": meta["release_date"],
                "spdx_license": _registry_license(name, meta["license"]),
                "license": meta["license"],
                "license_url": meta["license_url"] or "https://example.invalid/license-required",
                "redistribution": "allowed",
                "artifact": {
                    "path": OUTPUT_FILES[name],
                    "format": "parquet",
                    "sha256": sha256,
                    "bytes": byte_count,
                    "rows": row_count,
                },
                "columns": {
                    "smiles": "smiles",
                    "aliases": ["alias"],
                    "inchikey": "inchikey",
                    "source_record_id": "source_record_id",
                },
            }
        )
    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "sources": sources,
        "registry_root": ".",
    }


def validate_registry(path: Path) -> dict[str, Any]:
    registry = _load_json_file(path, "source registry")
    if registry.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise AliasSourceError("source registry schema_version mismatch")
    sources = registry.get("sources")
    if not isinstance(sources, list) or len(sources) != 4:
        raise AliasSourceError("source registry must contain exactly 4 sources")
    names = [source.get("canonical_name") for source in sources if isinstance(source, Mapping)]
    if tuple(names) != SOURCE_ORDER:
        raise AliasSourceError(f"source registry source order must be {SOURCE_ORDER}")
    for source in sources:
        if not isinstance(source, Mapping):
            raise AliasSourceError("source registry source must be an object")
        artifact = source.get("artifact")
        columns = source.get("columns")
        if not isinstance(artifact, Mapping) or not isinstance(columns, Mapping):
            raise AliasSourceError("source registry missing artifact or columns")
        if artifact.get("format") != "parquet":
            raise AliasSourceError("source registry artifact format must be parquet")
        if columns.get("smiles") != "smiles" or columns.get("aliases") != ["alias"]:
            raise AliasSourceError("source registry columns mapping mismatch")
        if columns.get("inchikey") != "inchikey" or columns.get("source_record_id") != "source_record_id":
            raise AliasSourceError("source registry optional columns mapping mismatch")
        artifact_path = path.parent / str(artifact.get("path"))
        observed_hash, observed_bytes, observed_rows = _validate_output_parquet(artifact_path)
        if artifact.get("sha256") != observed_hash or artifact.get("bytes") != observed_bytes:
            raise AliasSourceError("source registry artifact hash/bytes drift")
        if artifact.get("rows") != observed_rows:
            raise AliasSourceError("source registry artifact row-count drift")
    return registry


def validate_source_manifest(path: Path) -> dict[str, Any]:
    manifest = _load_json_file(path, "source manifest")
    if manifest.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise AliasSourceError("source manifest schema_version mismatch")
    expected_binding = manifest.get("self_binding_sha256")
    if not isinstance(expected_binding, str):
        raise AliasSourceError("source manifest missing self_binding_sha256")
    without_binding = dict(manifest)
    without_binding.pop("self_binding_sha256", None)
    if _hash_payload(without_binding) != expected_binding:
        raise AliasSourceError("source manifest self binding mismatch")
    registry_path = path.parent / "source_registry.json"
    registry = validate_registry(registry_path)
    if manifest.get("source_registry_sha256") != _hash_file(registry_path):
        raise AliasSourceError("source manifest registry hash drift")
    source_stats = manifest.get("source_stats")
    bindingdb_stats = (
        source_stats.get("BindingDB") if isinstance(source_stats, Mapping) else None
    )
    if not isinstance(bindingdb_stats, Mapping):
        raise AliasSourceError("source manifest missing BindingDB source_stats")
    for key in ("input_rows", "missing_smiles_rows", "eligible_rows"):
        value = bindingdb_stats.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise AliasSourceError(
                f"source manifest BindingDB source_stats.{key} must be nonnegative"
            )
    if bindingdb_stats["input_rows"] != (
        bindingdb_stats["eligible_rows"] + bindingdb_stats["missing_smiles_rows"]
    ):
        raise AliasSourceError("source manifest BindingDB source_stats do not conserve rows")
    pubchem_stats = source_stats.get("PubChem") if isinstance(source_stats, Mapping) else None
    if not isinstance(pubchem_stats, Mapping):
        raise AliasSourceError("source manifest missing PubChem source_stats")
    for key in (
        "selected_cids",
        "matched_cids",
        "missing_cids",
        "missing_fraction_ppm",
    ):
        value = pubchem_stats.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise AliasSourceError(
                f"source manifest PubChem source_stats.{key} must be nonnegative"
            )
    selected_cids = pubchem_stats["selected_cids"]
    matched_cids = pubchem_stats["matched_cids"]
    missing_cids = pubchem_stats["missing_cids"]
    if selected_cids <= 0 or matched_cids <= 0:
        raise AliasSourceError("source manifest PubChem selected/matched CIDs must be positive")
    if selected_cids != matched_cids + missing_cids:
        raise AliasSourceError("source manifest PubChem source_stats do not conserve CIDs")
    expected_missing_fraction_ppm = (
        (missing_cids * 1_000_000 + selected_cids - 1) // selected_cids
    )
    if pubchem_stats["missing_fraction_ppm"] != expected_missing_fraction_ppm:
        raise AliasSourceError("source manifest PubChem missing_fraction_ppm mismatch")
    if expected_missing_fraction_ppm > MAX_MISSING_PUBCHEM_CID_FRACTION_PPM:
        raise AliasSourceError("source manifest PubChem missing CID fraction exceeds policy")

    audit_artifacts = manifest.get("audit_artifacts")
    if not isinstance(audit_artifacts, Mapping) or set(audit_artifacts) != {
        PUBCHEM_MISSING_CIDS_FILE
    }:
        raise AliasSourceError("source manifest PubChem missing-CID audit record is incomplete")
    audit_record = audit_artifacts[PUBCHEM_MISSING_CIDS_FILE]
    if not isinstance(audit_record, Mapping):
        raise AliasSourceError("source manifest PubChem missing-CID audit record is invalid")
    audit_path = path.parent / PUBCHEM_MISSING_CIDS_FILE
    if audit_path.is_symlink() or not audit_path.is_file():
        raise AliasSourceError("PubChem missing-CID audit is missing or unsafe")
    if (
        audit_record.get("sha256") != _hash_file(audit_path)
        or audit_record.get("bytes") != audit_path.stat().st_size
    ):
        raise AliasSourceError("PubChem missing-CID audit hash/bytes drift")
    audit_rows = 0
    previous_cid = -1
    with audit_path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            cid = line.rstrip("\n")
            if not re.fullmatch(r"[0-9]+", cid):
                raise AliasSourceError(
                    f"PubChem missing-CID audit line {line_number} is invalid"
                )
            numeric_cid = int(cid)
            if numeric_cid <= previous_cid:
                raise AliasSourceError("PubChem missing-CID audit must be strictly increasing")
            previous_cid = numeric_cid
            audit_rows += 1
    if audit_record.get("rows") != audit_rows or audit_rows != missing_cids:
        raise AliasSourceError("PubChem missing-CID audit row-count drift")
    output_hashes = manifest.get("output_sha256")
    output_bytes = manifest.get("output_bytes")
    if not isinstance(output_hashes, Mapping) or not isinstance(output_bytes, Mapping):
        raise AliasSourceError("source manifest missing output hashes/bytes")
    for source in registry["sources"]:
        artifact = source["artifact"]
        relpath = artifact["path"]
        artifact_path = path.parent / relpath
        observed_hash, observed_bytes, observed_rows = _validate_output_parquet(artifact_path)
        if output_hashes.get(relpath) != observed_hash:
            raise AliasSourceError(f"source manifest output hash drift for {relpath}")
        if output_bytes.get(relpath) != observed_bytes:
            raise AliasSourceError(f"source manifest output byte drift for {relpath}")
        if manifest.get("output_rows", {}).get(relpath) != observed_rows:
            raise AliasSourceError(f"source manifest output row-count drift for {relpath}")
    return manifest


def build(args: argparse.Namespace) -> dict[str, Any]:
    inputs = {
        "ChEMBL": args.chembl_db,
        "BindingDB": args.bindingdb_tsv,
        "GtoPdb": args.gtopdb_ligands,
        "PubChem": args.pubchem_smiles_gz,
    }
    manifests = {
        "ChEMBL": args.chembl_manifest,
        "BindingDB": args.bindingdb_manifest,
        "GtoPdb": args.gtopdb_manifest,
        "PubChem": args.pubchem_manifest,
    }
    for name, path in inputs.items():
        _require_file(path, f"{name} input")
    out_dir = args.out_dir
    _cleanup_outputs(out_dir)
    try:
        source_meta = {
            "ChEMBL": _validate_chembl_manifest(
                manifests["ChEMBL"],
                inputs["ChEMBL"],
                args.chembl_release_date,
            ),
            "BindingDB": _validate_bindingdb_manifest(
                manifests["BindingDB"],
                inputs["BindingDB"],
                args.bindingdb_release_date,
            ),
            "GtoPdb": _validate_gtopdb_manifest(
                manifests["GtoPdb"],
                inputs["GtoPdb"],
            ),
            "PubChem": _validate_pubchem_manifest(
                manifests["PubChem"],
                inputs["PubChem"],
            ),
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        artifact_meta: dict[str, tuple[str, int, int]] = {}
        with tempfile.TemporaryDirectory(prefix=".alias-stage-", dir=out_dir.parent) as tmpdir:
            staging = AliasStaging(Path(tmpdir) / "aliases.sqlite")
            try:
                _read_chembl(inputs["ChEMBL"], staging)
                bindingdb_stats = _read_bindingdb(inputs["BindingDB"], staging)
                _read_gtopdb(inputs["GtoPdb"], staging)
                pubchem_stats = _read_pubchem(inputs["PubChem"], staging)
                for source in SOURCE_ORDER:
                    path = out_dir / OUTPUT_FILES[source]
                    _write_parquet(path, staging, source)
                    artifact_meta[source] = _validate_output_parquet(path)
                missing_cids_meta = _write_missing_cid_audit(
                    out_dir / PUBCHEM_MISSING_CIDS_FILE,
                    staging,
                )
            finally:
                staging.close()
        registry = _make_registry(
            out_dir=out_dir,
            source_meta=source_meta,
            artifact_meta=artifact_meta,
        )
        registry_path = out_dir / "source_registry.json"
        _write_json(registry_path, registry)
        validate_registry(registry_path)
        output_hashes = {OUTPUT_FILES[name]: artifact_meta[name][0] for name in SOURCE_ORDER}
        output_bytes = {OUTPUT_FILES[name]: artifact_meta[name][1] for name in SOURCE_ORDER}
        manifest_without_binding = {
            "schema_version": SOURCE_SCHEMA_VERSION,
            "builder": {
                "path": "scripts/build_discovery_alias_sources.py",
                "sha256": _hash_file(Path(__file__).resolve()),
            },
            "created_at_utc": "1970-01-01T00:00:00Z",
            "inputs": {
                name: {
                    "path": _relative_path(inputs[name], out_dir),
                    "sha256": _hash_file(inputs[name]),
                    "bytes": inputs[name].stat().st_size,
                    "manifest": {
                        **source_meta[name],
                        "path": _relative_path(manifests[name], out_dir),
                    },
                }
                for name in SOURCE_ORDER
            },
            "output_sha256": output_hashes,
            "output_bytes": output_bytes,
            "output_rows": {OUTPUT_FILES[name]: artifact_meta[name][2] for name in SOURCE_ORDER},
            "audit_artifacts": {
                PUBCHEM_MISSING_CIDS_FILE: {
                    "sha256": missing_cids_meta[0],
                    "bytes": missing_cids_meta[1],
                    "rows": missing_cids_meta[2],
                }
            },
            "source_stats": {
                "BindingDB": bindingdb_stats,
                "PubChem": pubchem_stats,
            },
            "source_registry_sha256": _hash_file(registry_path),
            "source_registry": "source_registry.json",
            "policy": {
                "fail_closed": True,
                "timestamp": "fixed deterministic created_at_utc",
                "self_binding": "self_binding_sha256 hashes this manifest excluding itself",
                "validator": "validate_source_manifest independently hashes registry and parquet outputs",
                "BindingDB_missing_smiles": (
                    "rows without Ligand SMILES are non-small-molecule records; "
                    "they are counted in source_stats and excluded from the alias map"
                ),
                "PubChem_missing_cids": (
                    "selected CIDs absent from the pinned PubChem snapshot are audited "
                    "and excluded only from the PubChem source; more than 10% blocks the build"
                ),
                "PubChem_max_missing_fraction_ppm": (
                    MAX_MISSING_PUBCHEM_CID_FRACTION_PPM
                ),
            },
        }
        manifest = {
            **manifest_without_binding,
            "self_binding_sha256": _hash_payload(manifest_without_binding),
        }
        manifest_path = out_dir / "source_manifest.json"
        _write_json(manifest_path, manifest)
        return validate_source_manifest(manifest_path)
    except Exception:
        _cleanup_outputs(out_dir)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chembl-db", required=True, type=Path)
    parser.add_argument("--chembl-manifest", required=True, type=Path)
    parser.add_argument("--chembl-release-date", required=True)
    parser.add_argument("--bindingdb-tsv", required=True, type=Path)
    parser.add_argument("--bindingdb-manifest", required=True, type=Path)
    parser.add_argument("--bindingdb-release-date", required=True)
    parser.add_argument("--gtopdb-ligands", required=True, type=Path)
    parser.add_argument("--gtopdb-manifest", required=True, type=Path)
    parser.add_argument("--pubchem-smiles-gz", required=True, type=Path)
    parser.add_argument("--pubchem-manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        build(_parser().parse_args(argv))
    except AliasSourceError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
