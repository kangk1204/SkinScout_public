#!/usr/bin/env python3
"""Build provenance-rich ChEMBL human activity evidence parquet files."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


LOG = logging.getLogger("stage0.chembl.evidence")
SCHEMA_VERSION = "chembl_activity_evidence.v1"
SOURCE_NAME = "ChEMBL"
SOURCE_LICENSE = "CC BY-SA 3.0"
SOURCE_LICENSE_URL = "https://chembl.gitbook.io/chembl-interface-documentation/about"
POLICY = {
    "slice": "Homo sapiens SINGLE PROTEIN target activities",
    "required_for_compatibility": [
        "activities.pchembl_value IS NOT NULL",
        "compound_structures.canonical_smiles is nonblank",
        "component_sequences.accession is nonblank",
    ],
    "preserved_not_filtered": [
        "activities.standard_relation including censored relations",
        "activities.data_validity_comment",
        "activities.potential_duplicate",
        "activities.action_type",
    ],
    "join_policy": "target_dictionary -> target_components -> component_sequences -> assays -> activities -> molecule_dictionary -> compound_structures, with docs left-joined",
}

LEGACY_COLUMNS = [
    "target_chembl_id",
    "uniprot",
    "molecule_chembl_id",
    "smiles",
    "act_type",
    "act_value",
    "act_units",
    "pchembl",
]

EVIDENCE_COLUMNS = LEGACY_COLUMNS + [
    "source_name",
    "source_db",
    "source_release",
    "source_version",
    "source_license",
    "source_db_sha256",
    "activity_id",
    "assay_id",
    "assay_chembl_id",
    "assay_type",
    "assay_test_type",
    "assay_category",
    "assay_confidence_score",
    "assay_relationship_type",
    "assay_source_id",
    "assay_source_name",
    "assay_variant_id",
    "assay_variant_accession",
    "assay_variant_mutation",
    "target_id",
    "target_organism",
    "target_type",
    "target_pref_name",
    "target_component_id",
    "molecule_molregno",
    "standard_inchi_key",
    "document_id",
    "document_chembl_id",
    "document_year",
    "pubmed_id",
    "doi",
    "patent_id",
    "document_type",
    "document_source_id",
    "document_source_name",
    "document_chembl_release_id",
    "standard_relation",
    "standard_type",
    "standard_value",
    "standard_units",
    "pchembl_value",
    "data_validity_comment",
    "potential_duplicate",
    "activity_comment",
    "action_type",
]

SCHEMA = pa.schema([
    ("target_chembl_id", pa.string()),
    ("uniprot", pa.string()),
    ("molecule_chembl_id", pa.string()),
    ("smiles", pa.string()),
    ("act_type", pa.string()),
    ("act_value", pa.float64()),
    ("act_units", pa.string()),
    ("pchembl", pa.float64()),
    ("source_name", pa.string()),
    ("source_db", pa.string()),
    ("source_release", pa.string()),
    ("source_version", pa.string()),
    ("source_license", pa.string()),
    ("source_db_sha256", pa.string()),
    ("activity_id", pa.int64()),
    ("assay_id", pa.int64()),
    ("assay_chembl_id", pa.string()),
    ("assay_type", pa.string()),
    ("assay_test_type", pa.string()),
    ("assay_category", pa.string()),
    ("assay_confidence_score", pa.int64()),
    ("assay_relationship_type", pa.string()),
    ("assay_source_id", pa.int64()),
    ("assay_source_name", pa.string()),
    ("assay_variant_id", pa.int64()),
    ("assay_variant_accession", pa.string()),
    ("assay_variant_mutation", pa.string()),
    ("target_id", pa.int64()),
    ("target_organism", pa.string()),
    ("target_type", pa.string()),
    ("target_pref_name", pa.string()),
    ("target_component_id", pa.int64()),
    ("molecule_molregno", pa.int64()),
    ("standard_inchi_key", pa.string()),
    ("document_id", pa.int64()),
    ("document_chembl_id", pa.string()),
    ("document_year", pa.int64()),
    ("pubmed_id", pa.string()),
    ("doi", pa.string()),
    ("patent_id", pa.string()),
    ("document_type", pa.string()),
    ("document_source_id", pa.int64()),
    ("document_source_name", pa.string()),
    ("document_chembl_release_id", pa.int64()),
    ("standard_relation", pa.string()),
    ("standard_type", pa.string()),
    ("standard_value", pa.float64()),
    ("standard_units", pa.string()),
    ("pchembl_value", pa.float64()),
    ("data_validity_comment", pa.string()),
    ("potential_duplicate", pa.bool_()),
    ("activity_comment", pa.string()),
    ("action_type", pa.string()),
])

REQUIRED_COLUMNS = {
    "activities": {
        "activity_id",
        "assay_id",
        "doc_id",
        "molregno",
        "standard_relation",
        "standard_type",
        "standard_value",
        "standard_units",
        "pchembl_value",
        "data_validity_comment",
        "potential_duplicate",
        "activity_comment",
        "action_type",
    },
    "assays": {
        "assay_id",
        "tid",
        "chembl_id",
        "assay_type",
        "assay_test_type",
        "assay_category",
        "confidence_score",
        "relationship_type",
        "src_id",
        "variant_id",
    },
    "target_dictionary": {
        "tid",
        "chembl_id",
        "organism",
        "target_type",
        "pref_name",
    },
    "target_components": {"tid", "component_id"},
    "component_sequences": {"component_id", "accession"},
    "molecule_dictionary": {"molregno", "chembl_id"},
    "compound_structures": {"molregno", "canonical_smiles"},
    "docs": {
        "doc_id",
        "chembl_id",
        "year",
        "pubmed_id",
        "doi",
        "patent_id",
        "doc_type",
        "src_id",
        "chembl_release_id",
    },
    "source": {"src_id", "src_short_name"},
    "variant_sequences": {"variant_id", "mutation", "accession"},
}

def _remove_outputs(*paths: Path) -> None:
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        tmp = _tmp_path(path)
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _tmp_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = _tmp_path(path)
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _table_columns(con: sqlite3.Connection, table: str) -> set[str]:
    rows = con.execute(f"PRAGMA table_info({table})").fetchall()
    if not rows:
        return set()
    return {str(row[1]) for row in rows}


def _validate_schema(con: sqlite3.Connection) -> None:
    errors: list[str] = []
    for table, required in REQUIRED_COLUMNS.items():
        columns = _table_columns(con, table)
        if not columns:
            errors.append(f"missing required table {table}")
            continue
        missing = sorted(required - columns)
        if missing:
            errors.append(f"{table} missing required columns: {', '.join(missing)}")
    if errors:
        raise ValueError("Invalid ChEMBL SQLite schema: " + "; ".join(errors))


def _optional_expr(con: sqlite3.Connection, table: str, column: str, alias: str) -> str:
    columns = _table_columns(con, table)
    if column in columns:
        return f"{table}.{column} AS {alias}"
    return f"NULL AS {alias}"


def _coerce_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    for column in EVIDENCE_COLUMNS:
        if column not in chunk.columns:
            chunk[column] = None
    for column in [
        "act_value",
        "pchembl",
        "standard_value",
        "pchembl_value",
    ]:
        chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
    for column in [
        "activity_id",
        "assay_id",
        "assay_confidence_score",
        "assay_source_id",
        "assay_variant_id",
        "target_id",
        "target_component_id",
        "molecule_molregno",
        "document_id",
        "document_year",
        "document_source_id",
        "document_chembl_release_id",
    ]:
        chunk[column] = pd.to_numeric(chunk[column], errors="coerce").astype("Int64")
    chunk["potential_duplicate"] = (
        pd.to_numeric(chunk["potential_duplicate"], errors="coerce").fillna(0).astype(int) != 0
    )
    string_columns = [
        column
        for column in EVIDENCE_COLUMNS
        if column not in {
            "act_value",
            "pchembl",
            "standard_value",
            "pchembl_value",
            "activity_id",
            "assay_id",
            "assay_confidence_score",
            "assay_source_id",
            "assay_variant_id",
            "target_id",
            "target_component_id",
            "molecule_molregno",
            "document_id",
            "document_year",
            "document_source_id",
            "document_chembl_release_id",
            "potential_duplicate",
        }
    ]
    for column in string_columns:
        chunk[column] = chunk[column].astype("string")
    return chunk[EVIDENCE_COLUMNS]


def _write_empty_parquet(path: Path) -> None:
    table = pa.Table.from_arrays(
        [pa.array([], type=field.type) for field in SCHEMA],
        schema=SCHEMA,
    )
    pq.write_table(table, _tmp_path(path))


def build_evidence(
    *,
    db_path: Path,
    out_dir: Path,
    release: str,
    release_date: str | None,
    source_archive: Path | None,
    chunksize: int,
) -> dict[str, Any]:
    if not db_path.exists():
        raise FileNotFoundError(f"ChEMBL SQLite DB not found: {db_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    human_path = out_dir / "human_activities.parquet"
    evidence_path = out_dir / "activity_evidence.parquet"
    manifest_path = out_dir / "source_manifest.json"
    _remove_outputs(human_path, evidence_path, manifest_path)

    db_sha = _sha256(db_path)
    archive_sha = _sha256(source_archive) if source_archive and source_archive.exists() else None
    writers: list[pq.ParquetWriter] = []
    row_count = 0
    rel_assay_count = 0
    try:
        con = sqlite3.connect(str(db_path), timeout=120)
        try:
            for pragma in (
                "PRAGMA temp_store=MEMORY",
                "PRAGMA cache_size=-2000000",
                "PRAGMA mmap_size=8000000000",
            ):
                con.execute(pragma)
            _validate_schema(con)
            con.executescript(
                """
                CREATE TEMP TABLE human_targets AS
                SELECT td.tid AS target_id,
                       td.chembl_id AS target_chembl_id,
                       td.organism AS target_organism,
                       td.target_type AS target_type,
                       td.pref_name AS target_pref_name,
                       tc.component_id AS target_component_id,
                       cs.accession AS uniprot
                FROM target_dictionary td
                JOIN target_components tc ON tc.tid = td.tid
                JOIN component_sequences cs ON cs.component_id = tc.component_id
                WHERE td.organism = 'Homo sapiens'
                  AND td.target_type = 'SINGLE PROTEIN'
                  AND cs.accession IS NOT NULL
                  AND TRIM(CAST(cs.accession AS TEXT)) != '';
                CREATE INDEX tmp_human_targets_tid ON human_targets(target_id);
                CREATE TEMP TABLE rel_assays AS
                SELECT a.assay_id AS assay_id,
                       ht.target_id AS target_id,
                       ht.target_chembl_id AS target_chembl_id,
                       ht.target_organism AS target_organism,
                       ht.target_type AS target_type,
                       ht.target_pref_name AS target_pref_name,
                       ht.target_component_id AS target_component_id,
                       ht.uniprot AS uniprot
                FROM assays a
                JOIN human_targets ht ON ht.target_id = a.tid;
                CREATE INDEX tmp_rel_assays_aid ON rel_assays(assay_id);
                """
            )
            rel_assay_count = int(con.execute("SELECT COUNT(*) FROM rel_assays").fetchone()[0])
            inchikey_expr = _optional_expr(
                con, "compound_structures", "standard_inchi_key", "standard_inchi_key"
            )
            sql = f"""
                SELECT
                  ra.target_chembl_id AS target_chembl_id,
                  ra.uniprot AS uniprot,
                  md.chembl_id AS molecule_chembl_id,
                  compound_structures.canonical_smiles AS smiles,
                  activities.standard_type AS act_type,
                  activities.standard_value AS act_value,
                  activities.standard_units AS act_units,
                  activities.pchembl_value AS pchembl,
                  ? AS source_name,
                  ? AS source_db,
                  ? AS source_release,
                  ? AS source_version,
                  ? AS source_license,
                  ? AS source_db_sha256,
                  activities.activity_id AS activity_id,
                  assays.assay_id AS assay_id,
                  assays.chembl_id AS assay_chembl_id,
                  assays.assay_type AS assay_type,
                  assays.assay_test_type AS assay_test_type,
                  assays.assay_category AS assay_category,
                  assays.confidence_score AS assay_confidence_score,
                  assays.relationship_type AS assay_relationship_type,
                  assays.src_id AS assay_source_id,
                  assay_source.src_short_name AS assay_source_name,
                  assays.variant_id AS assay_variant_id,
                  variant_sequences.accession AS assay_variant_accession,
                  variant_sequences.mutation AS assay_variant_mutation,
                  ra.target_id AS target_id,
                  ra.target_organism AS target_organism,
                  ra.target_type AS target_type,
                  ra.target_pref_name AS target_pref_name,
                  ra.target_component_id AS target_component_id,
                  md.molregno AS molecule_molregno,
                  {inchikey_expr},
                  docs.doc_id AS document_id,
                  docs.chembl_id AS document_chembl_id,
                  docs.year AS document_year,
                  docs.pubmed_id AS pubmed_id,
                  docs.doi AS doi,
                  docs.patent_id AS patent_id,
                  docs.doc_type AS document_type,
                  docs.src_id AS document_source_id,
                  document_source.src_short_name AS document_source_name,
                  docs.chembl_release_id AS document_chembl_release_id,
                  activities.standard_relation AS standard_relation,
                  activities.standard_type AS standard_type,
                  activities.standard_value AS standard_value,
                  activities.standard_units AS standard_units,
                  activities.pchembl_value AS pchembl_value,
                  activities.data_validity_comment AS data_validity_comment,
                  activities.potential_duplicate AS potential_duplicate,
                  activities.activity_comment AS activity_comment,
                  activities.action_type AS action_type
                FROM rel_assays ra
                JOIN assays ON assays.assay_id = ra.assay_id
                JOIN activities ON activities.assay_id = ra.assay_id
                JOIN molecule_dictionary md ON md.molregno = activities.molregno
                JOIN compound_structures ON compound_structures.molregno = activities.molregno
                LEFT JOIN docs ON docs.doc_id = activities.doc_id
                LEFT JOIN source AS assay_source ON assay_source.src_id = assays.src_id
                LEFT JOIN variant_sequences ON variant_sequences.variant_id = assays.variant_id
                LEFT JOIN source AS document_source ON document_source.src_id = docs.src_id
                WHERE activities.pchembl_value IS NOT NULL
                  AND compound_structures.canonical_smiles IS NOT NULL
                  AND TRIM(CAST(compound_structures.canonical_smiles AS TEXT)) != ''
                ORDER BY activities.activity_id, ra.target_component_id
            """
            params = (
                SOURCE_NAME,
                SOURCE_NAME,
                str(release),
                str(release),
                SOURCE_LICENSE,
                db_sha,
            )
            human_tmp = _tmp_path(human_path)
            evidence_tmp = _tmp_path(evidence_path)
            if rel_assay_count == 0:
                _write_empty_parquet(human_path)
                _write_empty_parquet(evidence_path)
            else:
                human_writer = pq.ParquetWriter(human_tmp, SCHEMA)
                evidence_writer = pq.ParquetWriter(evidence_tmp, SCHEMA)
                writers.extend([human_writer, evidence_writer])
                for chunk in pd.read_sql_query(sql, con, params=params, chunksize=chunksize):
                    chunk = _coerce_chunk(chunk)
                    table = pa.Table.from_pandas(chunk, schema=SCHEMA, preserve_index=False)
                    human_writer.write_table(table)
                    evidence_writer.write_table(table)
                    row_count += len(chunk)
                    LOG.info("streamed ChEMBL activity evidence rows=%d", row_count)
                for writer in writers:
                    writer.close()
                writers.clear()
            human_tmp.replace(human_path)
            evidence_tmp.replace(evidence_path)
        finally:
            con.close()
    except Exception:
        for writer in writers:
            writer.close()
        _remove_outputs(human_path, evidence_path, manifest_path)
        raise

    output_sha = {
        "human_activities.parquet": _sha256(human_path),
        "activity_evidence.parquet": _sha256(evidence_path),
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "name": SOURCE_NAME,
            "release": str(release),
            "release_date": release_date,
            "license": SOURCE_LICENSE,
            "license_url": SOURCE_LICENSE_URL,
            "redistribution": "allowed",
            "redistribution_requirements": "attribution and share-alike",
            "source_db": str(db_path),
            "source_db_sha256": db_sha,
            "source_db_bytes": db_path.stat().st_size,
            "source_archive": str(source_archive) if source_archive else None,
            "source_archive_sha256": archive_sha,
            "source_archive_bytes": (
                source_archive.stat().st_size if source_archive else None
            ),
        },
        "input_sha256": {
            "sqlite_db": db_sha,
            "sqlite_tarball": archive_sha,
        },
        "output_sha256": output_sha,
        "row_counts": {
            "human_activities": row_count,
            "activity_evidence": row_count,
            "human_single_protein_assays": rel_assay_count,
        },
        "schema": {
            "columns": EVIDENCE_COLUMNS,
            "legacy_compatibility_columns": LEGACY_COLUMNS,
        },
        "extraction_policy": POLICY,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    manifest["manifest_sha256_policy"] = (
        "source_manifest.json is excluded from output_sha256 because stable "
        "self-hashing is impossible"
    )
    _json_atomic(manifest_path, manifest)
    return manifest


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path, help="ChEMBL SQLite database path")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--release", default="37")
    parser.add_argument("--release-date")
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--chunksize", type=int, default=250_000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.chunksize < 1:
        print("[stage0.chembl][FATAL] --chunksize must be >= 1", file=sys.stderr)
        return 1
    if not str(args.release).isdigit() or int(args.release) < 1:
        print("[stage0.chembl][FATAL] --release must be a positive integer", file=sys.stderr)
        return 1
    try:
        manifest = build_evidence(
            db_path=args.db,
            out_dir=args.out_dir,
            release=str(args.release),
            release_date=args.release_date,
            source_archive=args.source_archive,
            chunksize=args.chunksize,
        )
    except Exception as exc:
        print(f"[stage0.chembl][FATAL] {exc}", file=sys.stderr)
        return 1
    rows = manifest["row_counts"]["human_activities"]
    print(f"[stage0.chembl] wrote activity evidence rows={rows:,} out_dir={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
