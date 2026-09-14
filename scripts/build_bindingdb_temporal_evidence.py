#!/usr/bin/env python3
"""Normalize BindingDB activity evidence into temporal parquet splits.

The output is evidence, not labels: missing/unknown activity rows are skipped and
never converted into negatives. Each quantitative Ki/IC50/Kd/EC50 value is kept
as a separate row so mixed assay types do not collapse into one target label.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import sqlite3
import zipfile
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator

import pyarrow as pa
import pyarrow.parquet as pq


CSV_FIELD_SIZE_LIMIT = 20_000_000
BINDINGDB_DATABASE_LICENSE = "CC BY 3.0"
BINDINGDB_CURATED_LICENSE = "CC BY 3.0 (BindingDB-curated)"
UNIPROT_ID_RE = re.compile(r"^[A-Z][0-9A-Z]{5,10}$")
ACTIVITY_COLUMNS = {
    "Ki": ("Ki (nM)", "Ki", "Ki(nM)", "Standard Value Ki"),
    "IC50": ("IC50 (nM)", "IC50", "IC50(nM)", "Standard Value IC50"),
    "Kd": ("Kd (nM)", "Kd", "Kd(nM)", "Standard Value Kd"),
    "EC50": ("EC50 (nM)", "EC50", "EC50(nM)", "Standard Value EC50"),
}
RELATION_COLUMNS = {
    "Ki": ("Ki Relation", "Ki Operator", "Ki Relation Symbol"),
    "IC50": ("IC50 Relation", "IC50 Operator", "IC50 Relation Symbol"),
    "Kd": ("Kd Relation", "Kd Operator", "Kd Relation Symbol"),
    "EC50": ("EC50 Relation", "EC50 Operator", "EC50 Relation Symbol"),
}
UNIT_COLUMNS = {
    "Ki": ("Ki Unit", "Ki Units"),
    "IC50": ("IC50 Unit", "IC50 Units"),
    "Kd": ("Kd Unit", "Kd Units"),
    "EC50": ("EC50 Unit", "EC50 Units"),
}
SMILES_COLUMNS = ("Ligand SMILES", "Ligand Smiles", "SMILES", "Canonical SMILES")
INCHIKEY_COLUMNS = ("Ligand InChI Key", "Ligand InChIKey", "InChIKey")
LIGAND_ID_COLUMNS = (
    "BindingDB Reactant_set_id",
    "BindingDB MonomerID",
    "Ligand ID",
    "Molecule ChEMBL ID",
    "ChEMBL ID",
)
ORGANISM_COLUMNS = (
    "Target Source Organism According to Curator or DataSource",
    "Target Organism",
    "Organism",
)
PUBLICATION_DATE_COLUMNS = (
    "Date of publication",
    "Publication Date",
    "Article Publication Date",
    "PubMed Publication Date",
    "Publication Year",
    "Year",
)
CURATION_DATE_COLUMNS = (
    "Date in BindingDB",
    "BindingDB Curation Date",
    "Curation Date",
    "BindingDB Last Updated",
    "Last Updated",
    "Record Date",
    "Release Date",
)
PMID_COLUMNS = ("PMID", "PubMed ID", "BindingDB PMID")
DOI_COLUMNS = ("Article DOI", "DOI")
PATENT_COLUMNS = ("Patent Number", "BindingDB Patent", "Patent ID")
ARTICLE_COLUMNS = ("Article ID", "BindingDB Entry DOI")
SOURCE_COLUMNS = (
    "Curation/DataSource",
    "Source",
    "Source Database",
    "Data Source",
    "Database",
)
LICENSE_COLUMNS = ("License", "Source License")
CHAIN_COUNT_COLUMNS = (
    "Number of Protein Chains in Target (>1 implies a multichain complex)",
    "Number of Protein Chains in Target",
    "Target Chain Count",
)
OUTPUT_COLUMNS = [
    "evidence_id",
    "duplicate_group_id",
    "duplicate_evidence",
    "duplicate_ordinal",
    "input_row_number",
    "source_db",
    "source_origin",
    "source_release",
    "source_license",
    "source_license_url",
    "chembl_derived_license_flag",
    "ligand_smiles",
    "ligand_inchikey",
    "ligand_id",
    "uniprot",
    "target_chain_count",
    "single_chain_target",
    "organism",
    "affinity_type",
    "relation",
    "censor",
    "affinity_value",
    "affinity_unit",
    "source_pmid",
    "source_doi",
    "source_patent",
    "source_article_id",
    "publication_date",
    "curation_date",
    "evidence_date",
    "evidence_date_source",
    "temporal_split",
]

OUTPUT_SCHEMA = pa.schema(
    [
        ("evidence_id", pa.string()),
        ("duplicate_group_id", pa.string()),
        ("duplicate_evidence", pa.bool_()),
        ("duplicate_ordinal", pa.int64()),
        ("input_row_number", pa.int64()),
        ("source_db", pa.string()),
        ("source_origin", pa.string()),
        ("source_release", pa.string()),
        ("source_license", pa.string()),
        ("source_license_url", pa.string()),
        ("chembl_derived_license_flag", pa.bool_()),
        ("ligand_smiles", pa.string()),
        ("ligand_inchikey", pa.string()),
        ("ligand_id", pa.string()),
        ("uniprot", pa.string()),
        ("target_chain_count", pa.int64()),
        ("single_chain_target", pa.bool_()),
        ("organism", pa.string()),
        ("affinity_type", pa.string()),
        ("relation", pa.string()),
        ("censor", pa.bool_()),
        ("affinity_value", pa.float64()),
        ("affinity_unit", pa.string()),
        ("source_pmid", pa.string()),
        ("source_doi", pa.string()),
        ("source_patent", pa.string()),
        ("source_article_id", pa.string()),
        ("publication_date", pa.string()),
        ("curation_date", pa.string()),
        ("evidence_date", pa.string()),
        ("evidence_date_source", pa.string()),
        ("temporal_split", pa.string()),
    ]
)


@contextmanager
def iter_bindingdb_rows(path: Path) -> Iterator[tuple[list[str], Iterable[dict[str, str]]]]:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"BindingDB input required and must be non-empty: {path}")

    csv.field_size_limit(CSV_FIELD_SIZE_LIMIT)
    handle: io.TextIOBase
    zip_handle: zipfile.ZipFile | None = None
    if path.suffix.lower() == ".zip":
        zip_handle = zipfile.ZipFile(path)
        members = [
            name
            for name in zip_handle.namelist()
            if not name.endswith("/") and name.lower().endswith((".tsv", ".txt"))
        ]
        if not members:
            raise SystemExit(f"BindingDB zip contains no TSV/TXT member: {path}")
        chosen = next((name for name in members if Path(name).name == "BindingDB_All.tsv"), members[0])
        handle = io.TextIOWrapper(zip_handle.open(chosen), encoding="utf-8", errors="replace")
    else:
        handle = path.open("r", encoding="utf-8", errors="replace")

    try:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise SystemExit(f"BindingDB input missing header: {path}")
        yield list(reader.fieldnames), reader
    finally:
        handle.close()
        if zip_handle is not None:
            zip_handle.close()


def _first_col(fieldnames: list[str], aliases: Iterable[str]) -> str | None:
    by_lower = {name.strip().lower(): name for name in fieldnames}
    for alias in aliases:
        found = by_lower.get(alias.lower())
        if found is not None:
            return found
    return None


def _all_cols(fieldnames: list[str], aliases: Iterable[str]) -> list[str]:
    by_lower = {name.strip().lower(): name for name in fieldnames}
    return [by_lower[alias.lower()] for alias in aliases if alias.lower() in by_lower]


def _split_multi(value: object) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    return [
        token.strip().upper()
        for token in re.split(r"[;,|]", text)
        if token.strip() and token.strip().upper() not in {"NA", "N/A", "NULL"}
    ]


def _is_human(value: object) -> bool:
    text = str(value or "").strip().lower()
    return text == "human" or "homo sapiens" in text or ",human," in f",{text},"


def _clean_text(value: object) -> str:
    text = str(value or "").strip()
    return "" if text.upper() in {"NA", "N/A", "NULL", "NAN"} else text


def _parse_activity(value: object) -> tuple[str, float] | None:
    text = _clean_text(value).replace(",", "")
    if not text:
        return None
    match = re.match(r"^\s*(<=|>=|<|>|=|~)?\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", text)
    if not match:
        return None
    parsed = float(match.group(2))
    if not math.isfinite(parsed) or parsed <= 0:
        return None
    relation = match.group(1) or "="
    if relation == "~":
        relation = "="
    return relation, parsed


def _parse_date(value: object) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    for fmt in (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%m/%d/%Y",
        "%d-%b-%Y",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            pass
    year = re.fullmatch(r"([12][0-9]{3})", text)
    if year:
        return f"{year.group(1)}-12-31"
    prefix = re.match(r"^([12][0-9]{3})[-/](0?[1-9]|1[0-2])$", text)
    if prefix:
        year_value = int(prefix.group(1))
        month_value = int(prefix.group(2))
        next_month = (
            date(year_value + 1, 1, 1)
            if month_value == 12
            else date(year_value, month_value + 1, 1)
        )
        return (next_month - timedelta(days=1)).isoformat()
    return ""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_text_atomic(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _validate_mirror_source_manifest(
    path: Path | None,
    *,
    input_path: Path,
    input_sha256: str,
    required_release: str,
    required_license_url: str,
) -> dict[str, object] | None:
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        source = payload["source"]
        extracted = payload["extracted"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise SystemExit(f"Invalid BindingDB mirror source manifest: {exc}") from exc
    errors: list[str] = []
    if payload.get("schema_version") != 1:
        errors.append(f"schema_version={payload.get('schema_version')!r}")
    if source.get("name") != "BindingDB":
        errors.append(f"source.name={source.get('name')!r}")
    if payload.get("release") != required_release:
        errors.append(f"release={payload.get('release')!r}")
    if source.get("license") != BINDINGDB_DATABASE_LICENSE:
        errors.append(f"source.license={source.get('license')!r}")
    if source.get("license_url") != required_license_url:
        errors.append(f"source.license_url={source.get('license_url')!r}")
    if extracted.get("sha256") != input_sha256:
        errors.append("extracted.sha256_mismatch")
    extracted_name = Path(str(extracted.get("path") or "")).name
    if extracted_name != input_path.name:
        errors.append(f"extracted.path={extracted.get('path')!r}")
    if errors:
        raise SystemExit("BindingDB mirror source manifest validation failed: " + "; ".join(errors))
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "schema_version": 1,
        "release": required_release,
        "source_license": BINDINGDB_DATABASE_LICENSE,
        "source_license_url": required_license_url,
        "extracted_sha256": input_sha256,
    }


def _write_parquet_atomic(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    arrays = {
        col: [row.get(col) for row in rows]
        for col in OUTPUT_COLUMNS
    }
    pq.write_table(pa.Table.from_pydict(arrays, schema=OUTPUT_SCHEMA), tmp)
    tmp.replace(path)


def _chain_uniprot_groups(fieldnames: list[str], max_chains: int) -> list[tuple[str, ...]]:
    groups: list[tuple[str, ...]] = []
    available = set(fieldnames)
    for idx in range(1, max_chains + 1):
        preferred = tuple(
            col
            for col in (
            f"UniProt (SwissProt) Primary ID of Target Chain {idx}",
            f"UniProt (TrEMBL) Primary ID of Target Chain {idx}",
            )
            if col in available
        )
        if preferred:
            groups.append(preferred)
    if groups:
        return groups
    generic = tuple(
        _all_cols(fieldnames, ("UniProt", "UniProt ID", "UniProt Accession", "Target UniProt"))
    )
    return [generic] if generic else []


def _validate_headers(fieldnames: list[str], max_chains: int) -> dict[str, object]:
    smiles = _first_col(fieldnames, SMILES_COLUMNS)
    inchikey = _first_col(fieldnames, INCHIKEY_COLUMNS)
    ligand_id = _first_col(fieldnames, LIGAND_ID_COLUMNS)
    organism = _first_col(fieldnames, ORGANISM_COLUMNS)
    uniprot_groups = _chain_uniprot_groups(fieldnames, max_chains)
    chain_count = _first_col(fieldnames, CHAIN_COUNT_COLUMNS)
    publication_dates = _all_cols(fieldnames, PUBLICATION_DATE_COLUMNS)
    curation_dates = _all_cols(fieldnames, CURATION_DATE_COLUMNS)
    activity_cols = {
        kind: _first_col(fieldnames, aliases)
        for kind, aliases in ACTIVITY_COLUMNS.items()
    }
    present_activity = {kind: col for kind, col in activity_cols.items() if col}

    missing: list[str] = []
    if not (smiles or inchikey or ligand_id):
        missing.append("ligand identity column")
    if organism is None:
        missing.append("organism column")
    if not uniprot_groups:
        missing.append("UniProt identity column")
    if not (publication_dates or curation_dates):
        missing.append("publication/curation date column")
    if not present_activity:
        missing.append("Ki/IC50/Kd/EC50 evidence column")
    if missing:
        raise SystemExit("BindingDB input missing required " + ", ".join(missing))

    return {
        "smiles": smiles,
        "inchikey": inchikey,
        "ligand_id": ligand_id,
        "organism": organism,
        "uniprot_groups": uniprot_groups,
        "chain_count": chain_count,
        "publication_dates": publication_dates,
        "curation_dates": curation_dates,
        "activity_cols": present_activity,
    }


def _first_value(row: dict[str, str], cols: Iterable[str]) -> str:
    for col in cols:
        value = _clean_text(row.get(col))
        if value:
            return value
    return ""


def _first_date(row: dict[str, str], cols: Iterable[str]) -> str:
    for col in cols:
        parsed = _parse_date(row.get(col))
        if parsed:
            return parsed
    return ""


def _row_uniprots(
    row: dict[str, str], groups: Iterable[tuple[str, ...]]
) -> tuple[list[str], int]:
    out: list[str] = []
    seen: set[str] = set()
    populated_chains = 0
    for group in groups:
        chosen = ""
        for col in group:
            for token in _split_multi(row.get(col)):
                canonical = token.split("-", 1)[0]
                if UNIPROT_ID_RE.fullmatch(canonical):
                    chosen = canonical
                    break
            if chosen:
                break
        if not chosen:
            continue
        populated_chains += 1
        if chosen not in seen:
            out.append(chosen)
            seen.add(chosen)
    return out, populated_chains


def _target_chain_count(value: object) -> int | None:
    text = _clean_text(value)
    if not text:
        return None
    try:
        parsed = int(float(text))
    except ValueError:
        return -1
    return parsed if parsed >= 1 else -1


def _relation_for(row: dict[str, str], fieldnames: list[str], kind: str, parsed_relation: str) -> str:
    col = _first_col(fieldnames, RELATION_COLUMNS[kind])
    relation = _clean_text(row.get(col)) if col else ""
    return relation if relation in {"<", "<=", "=", ">", ">="} else parsed_relation


def _unit_for(row: dict[str, str], fieldnames: list[str], kind: str) -> str:
    col = _first_col(fieldnames, UNIT_COLUMNS[kind])
    unit = _clean_text(row.get(col)) if col else ""
    return unit or ("nM" if "(nM)" in str(_first_col(fieldnames, ACTIVITY_COLUMNS[kind]) or "") else "")


def _base_source(
    row: dict[str, str], args: argparse.Namespace
) -> tuple[str, str, str, str, bool]:
    source_origin = _first_value(row, _all_cols(list(row), SOURCE_COLUMNS))
    source_db = args.source_db
    source_license = _first_value(row, _all_cols(list(row), LICENSE_COLUMNS)) or args.source_license
    scan_text = " ".join(
        [
            source_db,
            source_origin,
            source_license,
            _first_value(row, _all_cols(list(row), LIGAND_ID_COLUMNS)),
        ]
    ).lower()
    chembl_derived = "chembl" in scan_text
    if chembl_derived and "by-sa" not in source_license.lower():
        source_license = "CC BY-SA 3.0 (ChEMBL-derived)"
    elif source_license.lower() == "bindingdb terms" or (
        "bindingdb" in source_origin.lower() and "cc by" not in source_license.lower()
    ):
        source_license = args.source_license
    return (
        source_db,
        source_origin,
        source_license,
        args.source_license_url,
        chembl_derived,
    )


def _table_from_rows(rows: list[dict[str, object]]) -> pa.Table:
    arrays = {column: [row.get(column) for row in rows] for column in OUTPUT_COLUMNS}
    return pa.Table.from_pydict(arrays, schema=OUTPUT_SCHEMA)


def _duplicate_key(identity: dict[str, object]) -> str:
    payload = [
        identity[column]
        for column in (
            "ligand_smiles",
            "ligand_inchikey",
            "ligand_id",
            "uniprot",
            "affinity_type",
            "relation",
            "affinity_value",
            "affinity_unit",
            "source_pmid",
            "source_doi",
            "source_article_id",
            "evidence_date",
            "evidence_date_source",
            "target_chain_count",
            "single_chain_target",
        )
    ]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_final_parquets(
    *,
    spool_path: Path,
    duplicate_groups: set[str],
    outputs: dict[str, Path],
    batch_rows: int,
) -> int:
    temp_paths = {
        name: path.with_suffix(path.suffix + ".tmp") for name, path in outputs.items()
    }
    for path in temp_paths.values():
        path.unlink(missing_ok=True)
    writers: dict[str, pq.ParquetWriter] = {}
    duplicate_rows = 0
    try:
        writers = {
            name: pq.ParquetWriter(path, OUTPUT_SCHEMA, compression="snappy")
            for name, path in temp_paths.items()
        }
        parquet = pq.ParquetFile(spool_path)
        duplicate_idx = OUTPUT_SCHEMA.get_field_index("duplicate_evidence")
        for batch in parquet.iter_batches(batch_size=batch_rows):
            table = pa.Table.from_batches([batch], schema=OUTPUT_SCHEMA)
            group_ids = table.column("duplicate_group_id").to_pylist()
            duplicate_flags = [group_id in duplicate_groups for group_id in group_ids]
            duplicate_rows += sum(duplicate_flags)
            table = table.set_column(
                duplicate_idx,
                "duplicate_evidence",
                pa.array(duplicate_flags, type=pa.bool_()),
            )
            writers["activity_evidence"].write_table(table)
            split_values = table.column("temporal_split").to_pylist()
            pre_indexes = [idx for idx, value in enumerate(split_values) if value == "pre_cutoff"]
            post_indexes = [idx for idx, value in enumerate(split_values) if value == "post_cutoff"]
            if pre_indexes:
                writers["pre_cutoff"].write_table(table.take(pre_indexes))
            if post_indexes:
                writers["post_cutoff"].write_table(table.take(post_indexes))
        for writer in writers.values():
            writer.close()
        writers.clear()
        for name, path in outputs.items():
            temp_paths[name].replace(path)
    except BaseException:
        for writer in writers.values():
            writer.close()
        for path in temp_paths.values():
            path.unlink(missing_ok=True)
        raise
    return duplicate_rows


def normalize_bindingdb(args: argparse.Namespace) -> dict[str, object]:
    cutoff = date.fromisoformat(args.cutoff_date)
    input_sha256 = _sha256_file(args.bindingdb_tsv)
    mirror_source_manifest = _validate_mirror_source_manifest(
        args.source_manifest,
        input_path=args.bindingdb_tsv,
        input_sha256=input_sha256,
        required_release=args.source_release,
        required_license_url=args.source_license_url,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "activity_evidence": args.out_dir / "activity_evidence.parquet",
        "pre_cutoff": args.out_dir / "pre_cutoff.parquet",
        "post_cutoff": args.out_dir / "post_cutoff.parquet",
    }
    manifest_path = args.out_manifest or (args.out_dir / "manifest.json")
    spool_path = args.out_dir / ".bindingdb_activity_evidence.spool.parquet"
    duplicate_db_path = args.out_dir / ".bindingdb_duplicate_counts.sqlite"
    duplicate_db_sidecars = [
        duplicate_db_path,
        duplicate_db_path.with_name(duplicate_db_path.name + "-journal"),
        duplicate_db_path.with_name(duplicate_db_path.name + "-shm"),
        duplicate_db_path.with_name(duplicate_db_path.name + "-wal"),
    ]
    for path in [*outputs.values(), manifest_path, spool_path, *duplicate_db_sidecars]:
        path.unlink(missing_ok=True)
        path.with_suffix(path.suffix + ".tmp").unlink(missing_ok=True)
    spool_writer: pq.ParquetWriter | None = None
    buffer: list[dict[str, object]] = []
    duplicate_db = sqlite3.connect(duplicate_db_path)
    duplicate_db.execute("PRAGMA journal_mode=OFF")
    duplicate_db.execute("PRAGMA synchronous=OFF")
    duplicate_db.execute("PRAGMA temp_store=FILE")
    duplicate_db.execute(
        "CREATE TABLE duplicate_counts (group_id TEXT PRIMARY KEY, evidence_count INTEGER NOT NULL)"
    )
    rows_seen = 0
    evidence_rows = 0
    pre_count = 0
    post_count = 0
    skipped_non_human = 0
    skipped_missing_date = 0
    skipped_missing_uniprot = 0
    skipped_inconsistent_chain_metadata = 0
    skipped_unknown_activity = 0

    def flush_buffer() -> None:
        nonlocal spool_writer
        if not buffer:
            return
        if spool_writer is None:
            spool_writer = pq.ParquetWriter(
                spool_path,
                OUTPUT_SCHEMA,
                compression="snappy",
            )
        spool_writer.write_table(_table_from_rows(buffer))
        buffer.clear()

    try:
        with iter_bindingdb_rows(args.bindingdb_tsv) as (fieldnames, reader):
            cols = _validate_headers(fieldnames, args.chain_max)
            pmid_cols = _all_cols(fieldnames, PMID_COLUMNS)
            doi_cols = _all_cols(fieldnames, DOI_COLUMNS)
            patent_cols = _all_cols(fieldnames, PATENT_COLUMNS)
            article_cols = _all_cols(fieldnames, ARTICLE_COLUMNS)
            for row in reader:
                if args.max_rows > 0 and rows_seen >= args.max_rows:
                    break
                rows_seen += 1
                organism = _clean_text(row.get(cols["organism"]))
                if args.require_human and not _is_human(organism):
                    skipped_non_human += 1
                    continue
                publication_date = _first_date(row, cols["publication_dates"])
                curation_date = _first_date(row, cols["curation_dates"])
                evidence_date = publication_date or curation_date
                evidence_date_source = "publication" if publication_date else "curation"
                if not evidence_date:
                    skipped_missing_date += 1
                    continue
                uniprots, populated_chain_count = _row_uniprots(
                    row, cols["uniprot_groups"]
                )
                if not uniprots:
                    skipped_missing_uniprot += 1
                    continue
                explicit_chain_count = (
                    _target_chain_count(row.get(cols["chain_count"]))
                    if cols["chain_count"]
                    else None
                )
                if explicit_chain_count == -1 or (
                    explicit_chain_count is not None
                    and explicit_chain_count < populated_chain_count
                ):
                    skipped_inconsistent_chain_metadata += 1
                    continue
                target_chain_count = explicit_chain_count or populated_chain_count
                single_chain_target = target_chain_count == 1
                ligand_smiles = (
                    _clean_text(row.get(cols["smiles"])) if cols["smiles"] else ""
                )
                ligand_inchikey = (
                    _clean_text(row.get(cols["inchikey"])).upper()
                    if cols["inchikey"]
                    else ""
                )
                ligand_id = (
                    _clean_text(row.get(cols["ligand_id"])) if cols["ligand_id"] else ""
                )
                if not (ligand_smiles or ligand_inchikey or ligand_id):
                    continue
                (
                    source_db,
                    source_origin,
                    source_license,
                    source_license_url,
                    chembl_flag,
                ) = _base_source(row, args)
                row_had_activity = False
                for affinity_type, activity_col in cols["activity_cols"].items():
                    parsed = _parse_activity(row.get(activity_col))
                    if parsed is None:
                        continue
                    row_had_activity = True
                    parsed_relation, affinity_value = parsed
                    relation = _relation_for(
                        row, fieldnames, affinity_type, parsed_relation
                    )
                    unit = _unit_for(row, fieldnames, affinity_type)
                    split = (
                        "pre_cutoff"
                        if date.fromisoformat(evidence_date) <= cutoff
                        else "post_cutoff"
                    )
                    for uniprot in uniprots:
                        identity: dict[str, object] = {
                            # One-based data-row ordinal; the TSV header is not a
                            # data row and therefore does not shift provenance.
                            "input_row_number": rows_seen,
                            "source_db": source_db,
                            "source_origin": source_origin,
                            "source_release": args.source_release,
                            "source_license": source_license,
                            "source_license_url": source_license_url,
                            "chembl_derived_license_flag": chembl_flag,
                            "ligand_smiles": ligand_smiles,
                            "ligand_inchikey": ligand_inchikey,
                            "ligand_id": ligand_id,
                            "uniprot": uniprot,
                            "target_chain_count": target_chain_count,
                            "single_chain_target": single_chain_target,
                            "organism": organism,
                            "affinity_type": affinity_type,
                            "relation": relation,
                            "censor": relation != "=",
                            "affinity_value": affinity_value,
                            "affinity_unit": unit,
                            "source_pmid": _first_value(row, pmid_cols),
                            "source_doi": _first_value(row, doi_cols),
                            "source_patent": _first_value(row, patent_cols),
                            "source_article_id": _first_value(row, article_cols),
                            "publication_date": publication_date,
                            "curation_date": curation_date,
                            "evidence_date": evidence_date,
                            "evidence_date_source": evidence_date_source,
                            "temporal_split": split,
                        }
                        group_id = _duplicate_key(identity)
                        inserted = duplicate_db.execute(
                            "INSERT OR IGNORE INTO duplicate_counts(group_id, evidence_count) VALUES (?, 1)",
                            (group_id,),
                        ).rowcount
                        if inserted:
                            duplicate_ordinal = 1
                        else:
                            duplicate_db.execute(
                                "UPDATE duplicate_counts SET evidence_count = evidence_count + 1 WHERE group_id = ?",
                                (group_id,),
                            )
                            duplicate_ordinal = int(
                                duplicate_db.execute(
                                    "SELECT evidence_count FROM duplicate_counts WHERE group_id = ?",
                                    (group_id,),
                                ).fetchone()[0]
                            )
                        identity["duplicate_group_id"] = group_id
                        identity["duplicate_ordinal"] = duplicate_ordinal
                        identity["duplicate_evidence"] = False
                        evidence_payload = json.dumps(
                            identity,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        identity["evidence_id"] = hashlib.sha256(
                            evidence_payload.encode("utf-8")
                        ).hexdigest()
                        buffer.append(identity)
                        evidence_rows += 1
                        if split == "pre_cutoff":
                            pre_count += 1
                        else:
                            post_count += 1
                        if len(buffer) >= args.write_chunk_rows:
                            flush_buffer()
                if not row_had_activity:
                    skipped_unknown_activity += 1
                if rows_seen % 100_000 == 0:
                    print(
                        f"[bindingdb.evidence] input_rows={rows_seen:,} "
                        f"evidence_rows={evidence_rows:,}",
                        flush=True,
                    )
        flush_buffer()
        if spool_writer is not None:
            spool_writer.close()
            spool_writer = None
        if evidence_rows == 0:
            raise SystemExit("No human quantitative BindingDB evidence rows were emitted")
        duplicate_db.commit()
        duplicate_groups = {
            str(row[0])
            for row in duplicate_db.execute(
                "SELECT group_id FROM duplicate_counts WHERE evidence_count > 1"
            )
        }
        duplicate_group_count = len(duplicate_groups)
        duplicate_db.close()
        duplicate_rows = _write_final_parquets(
            spool_path=spool_path,
            duplicate_groups=duplicate_groups,
            outputs=outputs,
            batch_rows=args.write_chunk_rows,
        )
    except BaseException:
        if spool_writer is not None:
            spool_writer.close()
        duplicate_db.close()
        spool_path.unlink(missing_ok=True)
        raise
    finally:
        spool_path.unlink(missing_ok=True)
        for path in duplicate_db_sidecars:
            path.unlink(missing_ok=True)

    manifest = {
        "schema_version": "bindingdb_activity_evidence.v1",
        "policy": {
            "cutoff_date": args.cutoff_date,
            "date_source": (
                "publication_date preferred, curation_date fallback; "
                "year/month precision uses the conservative period end; "
                "file mtime never used"
            ),
            "unknown_activity_policy": "unknowns skipped, never emitted as negatives",
            "assay_policy": "Ki/IC50/Kd/EC50 emitted as separate evidence rows",
            "duplicate_policy": (
                "exact evidence groups counted in disk-backed SQLite and marked after "
                "a bounded-memory Parquet spool pass"
            ),
            "memory_policy": (
                "input rows are streamed; normalized rows are buffered up to "
                f"{args.write_chunk_rows} and finalized by Parquet row group"
            ),
            "require_human": args.require_human,
        },
        "source": {
            "path": str(args.bindingdb_tsv),
            "sha256": input_sha256,
            "source_db": args.source_db,
            "source_release": args.source_release,
            "source_license": args.source_license,
            "source_license_url": args.source_license_url,
            "license_policy": {
                "bindingdb_curated": args.source_license,
                "chembl_derived": "CC BY-SA 3.0 (ChEMBL-derived)",
            },
        },
        "inputs": {
            "mirror_source_manifest": mirror_source_manifest,
        },
        "schema": OUTPUT_COLUMNS,
        "counts": {
            "input_rows_seen": rows_seen,
            "activity_evidence": evidence_rows,
            "pre_cutoff": pre_count,
            "post_cutoff": post_count,
            "duplicate_evidence_rows": duplicate_rows,
            "duplicate_groups": duplicate_group_count,
            "skipped_non_human": skipped_non_human,
            "skipped_missing_date": skipped_missing_date,
            "skipped_missing_uniprot": skipped_missing_uniprot,
            "skipped_inconsistent_chain_metadata": skipped_inconsistent_chain_metadata,
            "skipped_unknown_activity": skipped_unknown_activity,
        },
        "artifacts": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
                "schema": OUTPUT_COLUMNS,
            }
            for name, path in outputs.items()
        },
        "manifest_sha256_policy": (
            "manifest.json is excluded from artifact hashes because stable "
            "self-hashing is impossible"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_text_atomic(json.dumps(manifest, indent=2, sort_keys=True) + "\n", manifest_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindingdb-tsv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--cutoff-date", required=True, help="YYYY-MM-DD temporal cutoff")
    parser.add_argument("--out-manifest", type=Path, default=None)
    parser.add_argument("--source-db", default="BindingDB")
    parser.add_argument("--source-release", required=True)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=None,
        help="optional mirror manifest; production workflow supplies and validates it",
    )
    parser.add_argument(
        "--source-license",
        default=BINDINGDB_CURATED_LICENSE,
    )
    parser.add_argument(
        "--source-license-url",
        default="https://www.bindingdb.org/rwd/bind/info.jsp",
    )
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--chain-max", type=int, default=50)
    parser.add_argument("--write-chunk-rows", type=int, default=100_000)
    parser.add_argument("--include-non-human", action="store_true")
    args = parser.parse_args()
    try:
        date.fromisoformat(args.cutoff_date)
    except ValueError as exc:
        raise SystemExit("--cutoff-date must be YYYY-MM-DD") from exc
    if args.max_rows < 0:
        raise SystemExit("--max-rows must be >= 0")
    if args.chain_max < 1 or args.chain_max > 100:
        raise SystemExit("--chain-max must be between 1 and 100")
    if args.write_chunk_rows < 1:
        raise SystemExit("--write-chunk-rows must be >= 1")
    args.require_human = not args.include_non_human
    return args


def main() -> None:
    manifest = normalize_bindingdb(parse_args())
    print(
        "wrote {activity_evidence} evidence rows ({pre_cutoff} pre, {post_cutoff} post)".format(
            **manifest["counts"]
        )
    )


if __name__ == "__main__":
    main()
