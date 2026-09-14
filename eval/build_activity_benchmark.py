#!/usr/bin/env python3
"""Build claim-grade multi-source activity benchmark partitions."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import re
import sys
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


SCHEMA_VERSION = "activity_benchmark.v1"
TARGET_CLUSTER_SCHEMA_VERSION = "skinscout.target-cluster-map.v2"
SCREENABLE_TARGET_CLUSTER_SCHEMA_VERSION = "skinscout.screenable-target-cluster-map.v2"
TARGET_CLUSTER_SCHEMA_VERSIONS = {
    TARGET_CLUSTER_SCHEMA_VERSION,
    SCREENABLE_TARGET_CLUSTER_SCHEMA_VERSION,
}
SOURCE_TARGET_EXCLUSION_SCHEMA_VERSION = (
    "skinscout.activity-source-target-exclusions.v1"
)
SOURCE_TARGET_EXCLUSION_COLUMNS = (
    "schema_version",
    "source_db",
    "source_release",
    "source_document_key",
    "uniprot",
    "reason_code",
    "source_document_url",
    "source_document_target",
    "database_target",
    "reviewed_at_utc",
    "evidence_note",
)
ALLOWED_SOURCE_TARGET_EXCLUSION_REASONS = {
    "source_document_target_mismatch",
}
CANONICAL_SOURCE_DATABASES = {
    "bindingdb": "BindingDB",
    "chembl": "ChEMBL",
    "gtopdb": "GtoPdb",
}
BINDINGDB_CURATED_ORIGINS = {
    "bindingdb",
    "bindingdb curated",
    "bindingdb-curated",
    "bindingdb_curated",
    "curated from the literature by bindingdb",
    "us patent",
    "wipo",
}
GTOPDB_CURATED_ORIGINS = {
    "gtopdb",
    "gtopdb expert-curated literature",
}
UNIPROT_ACCESSION_RE = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})(?:-[0-9]+)?$"
)
ALLOWED_TARGET_EXCLUSION_REASONS = {
    "not_found_in_uniprot_release",
    "unresolved_or_obsolete_uniprot_record",
    "taxonomy_mismatch",
    "sequence_below_minimum_length",
}


class InvalidLigandStructure(ValueError):
    """A nonblank ligand representation did not produce a valid RDKit graph."""


ENDPOINTS = {"IC50", "KI", "KD", "EC50"}
CHEMBL_REQUIRED_RELEASE = "37"
BINDINGDB_DEFAULT_REQUIRED_RELEASE = "2026-08"
GTOPDB_DEFAULT_REQUIRED_RELEASE = "2026.2"
TRAIN_END = date(2023, 12, 31)
DEV_START = date(2024, 1, 1)
DEV_END = date(2024, 12, 31)
TEST_START = date(2025, 1, 1)
VIEW_NAMES = (
    "temporal_test.parquet",
    "ligand_scaffold_cold.parquet",
    "target_cold30.parquet",
    "target_cold50.parquet",
    "dual_cold.parquet",
)
OUTPUT_NAMES = ("train.parquet", "dev.parquet", "test.parquet", *VIEW_NAMES, "manifest.json")

OUTPUT_COLUMNS = [
    "benchmark_id",
    "source_db",
    "source_release",
    "source_license",
    "evidence_id",
    "source_document_id",
    "publication_key",
    "source_document_type",
    "evidence_date",
    "split",
    "uniprot",
    "target_cluster_30",
    "target_cluster_50",
    "ligand_id",
    "ligand_smiles",
    "ligand_inchikey",
    "scaffold_smiles",
    "scaffold_id",
    "endpoint",
    "relation",
    "standard_value_nm",
    "pactivity",
    "activity_class",
    "binary_label",
]

OUTPUT_SCHEMA = pa.schema(
    [
        ("benchmark_id", pa.string()),
        ("source_db", pa.string()),
        ("source_release", pa.string()),
        ("source_license", pa.string()),
        ("evidence_id", pa.string()),
        ("source_document_id", pa.string()),
        ("publication_key", pa.string()),
        ("source_document_type", pa.string()),
        ("evidence_date", pa.string()),
        ("split", pa.string()),
        ("uniprot", pa.string()),
        ("target_cluster_30", pa.string()),
        ("target_cluster_50", pa.string()),
        ("ligand_id", pa.string()),
        ("ligand_smiles", pa.string()),
        ("ligand_inchikey", pa.string()),
        ("scaffold_smiles", pa.string()),
        ("scaffold_id", pa.string()),
        ("endpoint", pa.string()),
        ("relation", pa.string()),
        ("standard_value_nm", pa.float64()),
        ("pactivity", pa.float64()),
        ("activity_class", pa.string()),
        ("binary_label", pa.bool_()),
    ]
)

VIEW_COLUMNS = OUTPUT_COLUMNS + [
    "evaluation_view",
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
]

VIEW_SCHEMA = OUTPUT_SCHEMA.append(pa.field("evaluation_view", pa.string()))
for _field_name in VIEW_COLUMNS[len(OUTPUT_COLUMNS) + 1 :]:
    VIEW_SCHEMA = VIEW_SCHEMA.append(pa.field(_field_name, pa.bool_()))


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


def _bool_false(value: object) -> bool:
    text = _clean(value).lower()
    return text in {"", "0", "false", "f", "no", "n"}


def _clean_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").fillna("").str.strip()
    return cleaned.mask(cleaned.str.upper().isin({"NA", "N/A", "NULL", "NAN", "NONE"}), "")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _output_paths(out_dir: Path) -> list[Path]:
    return [out_dir / name for name in OUTPUT_NAMES]


def _remove_outputs(out_dir: Path) -> None:
    for path in _output_paths(out_dir):
        path.unlink(missing_ok=True)
        path.with_suffix(path.suffix + ".tmp").unlink(missing_ok=True)


def _read_table(path: Path, columns: Iterable[str] | None = None) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"Input required and must be non-empty: {path}")
    wanted = set(columns or [])
    if path.suffix.lower() == ".parquet":
        pf = pq.ParquetFile(path)
        available = set(pf.schema_arrow.names)
        read_columns = [column for column in (columns or pf.schema_arrow.names) if column in available]
        if columns is not None and not read_columns:
            raise SystemExit(f"{path} has none of the requested input columns")
        table = pq.read_table(path, columns=read_columns or None, memory_map=True)
        if table.num_rows == 0:
            return table.to_pandas()
        return table.to_pandas(split_blocks=True, self_destruct=True)
    if path.suffix.lower() in {".csv", ".tsv", ".txt"}:
        sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
        usecols = (lambda column: column in wanted) if wanted else None
        return pd.read_csv(path, sep=sep, usecols=usecols)
    raise SystemExit(f"Unsupported input format: {path}")


def _input_row_count(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"Input required and must be non-empty: {path}")
    if path.suffix.lower() == ".parquet":
        return int(pq.ParquetFile(path).metadata.num_rows)
    if path.suffix.lower() in {".csv", ".tsv", ".txt"}:
        with path.open("rb") as handle:
            lines = sum(1 for _ in handle)
        return max(0, lines - 1)
    raise SystemExit(f"Unsupported input format: {path}")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_columns(df: pd.DataFrame, path: Path, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise SystemExit(f"{path} missing required column(s): {', '.join(missing)}")


def _normalize_doi(value: object) -> str:
    text = _clean(value).lower()
    if not text:
        return ""
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi:\s*", "", text)
    return re.sub(r"\s+", "", text).rstrip(".")


def _normalize_patent(value: object) -> str:
    text = _clean(value).upper()
    if not text:
        return ""
    text = re.sub(r"^PATENT:\s*", "", text)
    return re.sub(r"[^A-Z0-9]", "", text)


def _publication_key(row: pd.Series, source_db: str, source_document_id: str) -> str:
    for column in ("source_doi", "doi", "document_doi"):
        doi = _normalize_doi(row.get(column))
        if doi:
            return f"doi:{doi}"
    for column in ("source_pmid", "pmid", "pubmed_id"):
        pmid = re.sub(r"\D+", "", _clean(row.get(column)))
        if pmid:
            return f"pmid:{pmid}"
    for column in ("source_patent", "patent_id"):
        patent = _normalize_patent(row.get(column))
        if patent:
            return f"patent:{patent}"
    return f"source:{source_db.casefold()}:{source_document_id}"


def _normalize_source_document_key(value: object) -> str:
    text = _clean(value)
    match = re.fullmatch(r"(doi|pmid|patent):(.+)", text, flags=re.IGNORECASE)
    if match is None:
        return ""
    kind = match.group(1).lower()
    payload = match.group(2)
    if kind == "doi":
        normalized = _normalize_doi(payload)
    elif kind == "pmid":
        normalized = re.sub(r"\D+", "", payload)
    else:
        normalized = _normalize_patent(payload)
    return f"{kind}:{normalized}" if normalized else ""


def _load_source_target_exclusions(
    path: Path | None,
) -> tuple[dict[tuple[str, str, str, str], dict[str, str]], dict[str, Any]]:
    if path is None:
        return {}, {
            "applied": False,
            "schema_version": SOURCE_TARGET_EXCLUSION_SCHEMA_VERSION,
            "artifact": None,
        }

    df = _read_table(path)
    if tuple(df.columns) != SOURCE_TARGET_EXCLUSION_COLUMNS:
        raise SystemExit(
            f"{path} source-target exclusion columns must exactly be: "
            + ", ".join(SOURCE_TARGET_EXCLUSION_COLUMNS)
        )
    if df.empty:
        raise SystemExit(f"Source-target exclusion contract must contain at least one rule: {path}")

    exclusions: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for index, row in df.iterrows():
        record = {column: _clean(row[column]) for column in SOURCE_TARGET_EXCLUSION_COLUMNS}
        blank = [column for column, value in record.items() if not value]
        if blank:
            raise SystemExit(
                f"Source-target exclusion row {index} has blank field(s): {', '.join(blank)}"
            )
        if record["schema_version"] != SOURCE_TARGET_EXCLUSION_SCHEMA_VERSION:
            raise SystemExit(
                f"Source-target exclusion row {index} schema_version must be "
                f"{SOURCE_TARGET_EXCLUSION_SCHEMA_VERSION}"
            )
        source_key = record["source_db"].casefold()
        canonical_source = CANONICAL_SOURCE_DATABASES.get(source_key)
        if canonical_source is None or record["source_db"] != canonical_source:
            raise SystemExit(
                f"Source-target exclusion row {index} source_db must be one of: "
                + ", ".join(sorted(CANONICAL_SOURCE_DATABASES.values()))
            )
        normalized_document_key = _normalize_source_document_key(
            record["source_document_key"]
        )
        if record["source_document_key"] != normalized_document_key:
            raise SystemExit(
                f"Source-target exclusion row {index} source_document_key is not normalized"
            )
        if not UNIPROT_ACCESSION_RE.fullmatch(record["uniprot"]):
            raise SystemExit(
                f"Source-target exclusion row {index} has invalid UniProt accession"
            )
        if record["reason_code"] not in ALLOWED_SOURCE_TARGET_EXCLUSION_REASONS:
            raise SystemExit(
                f"Source-target exclusion row {index} has unsupported reason_code"
            )
        parsed_url = urlparse(record["source_document_url"])
        if parsed_url.scheme != "https" or not parsed_url.netloc:
            raise SystemExit(
                f"Source-target exclusion row {index} source_document_url must be an HTTPS URL"
            )
        if record["uniprot"] not in record["database_target"]:
            raise SystemExit(
                f"Source-target exclusion row {index} database_target must identify its UniProt accession"
            )
        if (
            record["source_document_target"].casefold()
            == record["database_target"].casefold()
        ):
            raise SystemExit(
                f"Source-target exclusion row {index} does not describe a target mismatch"
            )
        if not re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z",
            record["reviewed_at_utc"],
        ):
            raise SystemExit(
                f"Source-target exclusion row {index} reviewed_at_utc must be RFC3339 UTC"
            )
        try:
            datetime.fromisoformat(record["reviewed_at_utc"].replace("Z", "+00:00"))
        except ValueError as exc:
            raise SystemExit(
                f"Source-target exclusion row {index} reviewed_at_utc is invalid"
            ) from exc

        key = (
            source_key,
            record["source_release"],
            normalized_document_key,
            record["uniprot"],
        )
        if key in exclusions:
            raise SystemExit(f"Duplicate source-target exclusion rule at row {index}")
        exclusions[key] = record

    return exclusions, {
        "applied": True,
        "schema_version": SOURCE_TARGET_EXCLUSION_SCHEMA_VERSION,
        "artifact": {
            "path": str(path.resolve()),
            "sha256": _sha256_file(path),
            "rows": len(df),
        },
    }


def _apply_source_target_exclusions(
    rows: list[dict[str, object]],
    exclusions: dict[tuple[str, str, str, str], dict[str, str]],
    contract: dict[str, Any],
) -> tuple[list[dict[str, object]], dict[str, Any], Counter[str]]:
    match_counts: Counter[tuple[str, str, str, str]] = Counter()
    filtered_by_source: Counter[str] = Counter()
    retained: list[dict[str, object]] = []
    for row in rows:
        key = (
            _clean(row.get("source_db")).casefold(),
            _clean(row.get("source_release")),
            _clean(row.get("publication_key")),
            _clean(row.get("uniprot")),
        )
        if key not in exclusions:
            retained.append(row)
            continue
        match_counts[key] += 1
        filtered_by_source[key[0]] += 1

    unmatched = [key for key in exclusions if match_counts[key] == 0]
    if unmatched:
        source, release, document, uniprot = sorted(unmatched)[0]
        raise SystemExit(
            "Source-target exclusion rule did not match any normalized claim-grade row: "
            f"source_db={source} source_release={release} "
            f"source_document_key={document} uniprot={uniprot}"
        )

    rules = []
    for key in sorted(exclusions):
        rules.append({**exclusions[key], "matched_rows": match_counts[key]})
    audit = {
        **contract,
        "rule_count": len(exclusions),
        "all_rules_matched": bool(exclusions) and len(match_counts) == len(exclusions),
        "total_filtered_rows": sum(match_counts.values()),
        "filtered_rows_by_source": dict(sorted(filtered_by_source.items())),
        "rules": rules,
        "policy": (
            "Rules bind source database, source release, normalized primary-source document "
            "identifier, and UniProt accession. Every declared rule must match at least one "
            "normalized claim-grade row or benchmark construction fails closed."
        ),
    }
    return retained, audit, filtered_by_source


def _parse_date(value: object) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(float(value)) and float(value).is_integer():
            year = int(value)
            if 1000 <= year <= 2999:
                return f"{year}-12-31"
    text = _clean(value)
    if not text:
        return ""
    try:
        numeric = float(text)
        if math.isfinite(numeric) and numeric.is_integer():
            year = int(numeric)
            if 1000 <= year <= 2999:
                return f"{year}-12-31"
    except ValueError:
        pass
    if len(text) >= 10:
        try:
            return date.fromisoformat(text[:10]).isoformat()
        except ValueError:
            pass
    if text.isdigit() and len(text) == 4:
        return f"{text}-12-31"
    return ""


def _split_for(iso_date: str) -> str:
    parsed = date.fromisoformat(iso_date)
    if parsed <= TRAIN_END:
        return "train"
    if DEV_START <= parsed <= DEV_END:
        return "dev"
    if parsed >= TEST_START:
        return "test"
    raise SystemExit(f"Date outside declared split boundaries: {iso_date}")


def _pactivity(value_nm: float) -> float:
    return 9.0 - math.log10(value_nm)


def _activity_class(pactivity: float) -> str:
    if pactivity >= 7.0:
        return "strong_positive"
    if pactivity >= 5.0:
        return "weak_positive"
    return "low_potency_quantitative"


def _canonical_smiles(smiles: str) -> tuple[str, str]:
    try:
        from rdkit import Chem
    except Exception as exc:  # pragma: no cover - environment-specific
        raise SystemExit(f"RDKit is required to canonicalize ligand SMILES: {exc}") from exc
    params = Chem.SmilesParserParams()
    params.allowCXSMILES = True
    params.parseName = False
    params.strictCXSMILES = True
    mol = Chem.MolFromSmiles(smiles, params)
    parse_mode = "strict"
    if mol is None:
        params.strictCXSMILES = False
        mol = Chem.MolFromSmiles(smiles, params)
        parse_mode = "lenient_cx_metadata"
    if mol is None:
        raise InvalidLigandStructure("RDKit failed strict and lenient CXSMILES parsing")
    return Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True), parse_mode


def _scaffold(canonical_smiles: str) -> tuple[str, str]:
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except Exception as exc:  # pragma: no cover - environment-specific
        raise SystemExit(f"RDKit is required to compute Bemis-Murcko scaffolds: {exc}") from exc
    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol is None:
        raise InvalidLigandStructure("RDKit failed to parse its canonical SMILES output")
    mol = Chem.RemoveHs(mol)
    scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_smiles = Chem.MolToSmiles(scaffold_mol, canonical=True)
    if not scaffold_smiles:
        scaffold_id = "bemis_murcko_acyclic:" + _hash_text(canonical_smiles)
    else:
        scaffold_id = "bemis_murcko:" + _hash_text(scaffold_smiles)
    return scaffold_smiles, scaffold_id


def _load_clusters(
    path: Path,
    manifest_path: Path,
) -> tuple[dict[str, dict[str, str]], dict[str, Any], dict[str, dict[str, Any]]]:
    df = _read_table(path)
    aliases = {
        "uniprot": ("uniprot", "target_uniprot", "accession"),
        "target_cluster_30": ("target_cluster_30", "cluster_30", "main_cluster", "mmseqs30"),
        "target_cluster_50": ("target_cluster_50", "cluster_50", "sensitivity_cluster", "mmseqs50"),
    }
    cols: dict[str, str] = {}
    lower = {column.lower(): column for column in df.columns}
    for key, candidates in aliases.items():
        for candidate in candidates:
            if candidate.lower() in lower:
                cols[key] = lower[candidate.lower()]
                break
    missing = [key for key in aliases if key not in cols]
    if missing:
        raise SystemExit(f"{path} missing target cluster mapping column(s): {', '.join(missing)}")
    mapping: dict[str, dict[str, str]] = {}
    for idx, row in df.iterrows():
        uniprot = _clean(row[cols["uniprot"]])
        c30 = _clean(row[cols["target_cluster_30"]])
        c50 = _clean(row[cols["target_cluster_50"]])
        if not uniprot or not c30 or not c50:
            raise SystemExit(f"{path} has blank target cluster assignment at row {idx}")
        if uniprot in mapping:
            raise SystemExit(f"{path} has duplicate target cluster assignment for {uniprot}")
        mapping[uniprot] = {"target_cluster_30": c30, "target_cluster_50": c50}
    if not manifest_path.exists() or manifest_path.stat().st_size == 0:
        raise SystemExit(
            f"Target cluster manifest is required and must be non-empty: {manifest_path}"
        )
    try:
        manifest = _read_json(manifest_path)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid target cluster manifest JSON: {manifest_path}") from exc
    manifest_schema = manifest.get("schema_version")
    if manifest_schema not in TARGET_CLUSTER_SCHEMA_VERSIONS:
        raise SystemExit(
            "Target cluster manifest schema must be one of "
            f"{sorted(TARGET_CLUSTER_SCHEMA_VERSIONS)}: "
            f"{manifest_path}"
        )
    artifact = manifest.get("artifact") if isinstance(manifest.get("artifact"), dict) else {}
    current_sha256 = _sha256_file(path)
    if artifact.get("sha256") != current_sha256:
        raise SystemExit("Target cluster CSV sha256 does not match its manifest")
    if artifact.get("rows") != len(df):
        raise SystemExit("Target cluster CSV row count does not match its manifest")
    artifact_path = str(artifact.get("path") or "").strip()
    if not artifact_path or Path(artifact_path).resolve() != path.resolve():
        raise SystemExit("Target cluster CSV path does not match its manifest")

    if manifest_schema == SCREENABLE_TARGET_CLUSTER_SCHEMA_VERSION:
        production = (
            manifest.get("production_contract")
            if isinstance(manifest.get("production_contract"), dict)
            else {}
        )
        universe = (
            manifest.get("universe_policy")
            if isinstance(manifest.get("universe_policy"), dict)
            else {}
        )
        if production.get("passes") is not True:
            raise SystemExit("Screenable target cluster production contract must pass")
        if universe.get("evaluation_panel_used") is not False:
            raise SystemExit("Screenable target clusters must be evaluation-panel independent")
        if universe.get("known_target_assistance") is not False:
            raise SystemExit("Screenable target clusters must not use known-target assistance")
        if universe.get("union_target_count") != len(df):
            raise SystemExit("Screenable target cluster universe count does not match its CSV")
    excluded_payload = (
        manifest.get("excluded_targets")
        if isinstance(manifest.get("excluded_targets"), dict)
        else {}
    )
    records = excluded_payload.get("records")
    if not isinstance(records, list) or excluded_payload.get("count") != len(records):
        raise SystemExit("Target cluster manifest excluded target count is invalid")
    exclusions: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise SystemExit(f"Target cluster exclusion at index {index} must be an object")
        accession = _clean(record.get("uniprot"))
        reason = _clean(record.get("reason"))
        if not accession or reason not in ALLOWED_TARGET_EXCLUSION_REASONS:
            raise SystemExit(f"Invalid target cluster exclusion at index {index}")
        if (
            accession in mapping
            and manifest_schema != SCREENABLE_TARGET_CLUSTER_SCHEMA_VERSION
        ):
            raise SystemExit(f"Excluded target unexpectedly has a cluster assignment: {accession}")
        if accession in exclusions:
            raise SystemExit(f"Duplicate target cluster exclusion: {accession}")
        exclusions[accession] = dict(record)

    cluster_meta = {
        "path": str(path.resolve()),
        "sha256": current_sha256,
        "rows": len(df),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256_file(manifest_path),
        "manifest_schema_version": manifest["schema_version"],
        "parameters": manifest.get("parameters", {}),
        "production_contract": manifest.get("production_contract"),
        "universe_policy": manifest.get("universe_policy"),
        "sequence_universe_manifest": (
            manifest.get("inputs", {}).get("sequence_universe_manifest")
            if isinstance(manifest.get("inputs"), dict)
            else None
        ),
    }
    return mapping, cluster_meta, exclusions


def _source_meta(row: pd.Series, default_db: str) -> tuple[str, str, str]:
    source_db = _clean(row.get("source_db")) or _clean(row.get("source_name")) or default_db
    release = _clean(row.get("source_release")) or _clean(row.get("source_version"))
    license_text = _clean(row.get("source_license"))
    return source_db, release, license_text


def _bindingdb_origin_allowed(row: pd.Series) -> bool:
    origin = _clean(row.get("source_origin")).casefold()
    license_text = _clean(row.get("source_license"))
    chembl_flag = _clean(row.get("chembl_derived_license_flag")).lower() in {"true", "1", "yes", "y"}
    return bool(origin in BINDINGDB_CURATED_ORIGINS and license_text and not chembl_flag)


def _bindingdb_reason_masks(
    df: pd.DataFrame,
    *,
    allowed_origins: set[str] = BINDINGDB_CURATED_ORIGINS,
    origin_filter_name: str = "bindingdb_curated_origin",
) -> tuple[dict[str, pd.Series], pd.Series]:
    needed = {
        "affinity_type",
        "relation",
        "affinity_unit",
        "publication_date",
        "evidence_date_source",
        "single_chain_target",
        "source_origin",
        "source_license",
        "source_release",
        "chembl_derived_license_flag",
        "ligand_smiles",
        "source_doi",
        "source_pmid",
        "source_patent",
        "source_article_id",
    }
    empty = pd.Series("", index=df.index, dtype="string")
    text = {
        column: _clean_series(df[column]) if column in df.columns else empty
        for column in needed
    }
    numeric_value = pd.to_numeric(df["affinity_value"], errors="coerce")
    positive_finite = numeric_value.map(
        lambda value: bool(pd.notna(value) and math.isfinite(float(value)) and float(value) > 0)
    )
    evidence_dates = text["publication_date"].map(_parse_date)
    chembl_flag = text["chembl_derived_license_flag"].str.casefold().isin(
        {"true", "1", "yes", "y"}
    )
    origin = text["source_origin"].str.casefold()
    license_present = text["source_license"].ne("")
    curated = origin.isin(allowed_origins)
    origin_allowed = curated & license_present & ~chembl_flag
    masks = {
        "endpoint": ~text["affinity_type"].str.upper().isin(ENDPOINTS),
        "exact_relation": text["relation"].ne("="),
        "nm_units": text["affinity_unit"].str.casefold().ne("nm"),
        "positive_value": ~positive_finite,
        "publication_date_evidence": text["evidence_date_source"].ne("publication"),
        "dated": evidence_dates.eq(""),
        "single_chain_target": ~text["single_chain_target"].str.casefold().isin(
            {"true", "1"}
        ),
        origin_filter_name: ~origin_allowed,
        "nonblank_ligand_smiles": text["ligand_smiles"].eq(""),
        "publication_identifier": (
            text["source_doi"].eq("")
            & text["source_pmid"].eq("")
            & text["source_patent"].eq("")
            & text["source_article_id"].eq("")
        ),
    }
    return masks, evidence_dates


def _bindingdb_arrow_filter(available: set[str]) -> ds.Expression:
    field = ds.field
    expression = (
        field("affinity_type").isin(["EC50", "IC50", "Kd", "Ki", "KD", "KI"])
        & (field("relation") == "=")
        & (field("affinity_unit") == "nM")
        & (field("affinity_value") > 0)
        & (field("evidence_date_source") == "publication")
        & (field("single_chain_target") == True)  # noqa: E712 - Arrow expression
        & field("publication_date").is_valid()
        & field("source_origin").is_valid()
        & field("source_license").is_valid()
    )
    if "ligand_smiles" in available:
        expression &= field("ligand_smiles").is_valid() & (field("ligand_smiles") != "")
    identifier_expression = None
    for column in ("source_doi", "source_pmid", "source_patent", "source_article_id"):
        if column not in available:
            continue
        candidate = field(column).is_valid() & (field(column) != "")
        identifier_expression = (
            candidate
            if identifier_expression is None
            else identifier_expression | candidate
        )
    if identifier_expression is not None:
        expression &= identifier_expression
    if "chembl_derived_license_flag" in available:
        flag = field("chembl_derived_license_flag")
        expression &= flag.is_null() | (flag == False)  # noqa: E712 - Arrow expression
    return expression


def _chembl_reason_masks(
    df: pd.DataFrame,
    columns: Iterable[str],
    *,
    exclude_bindingdb_sources: bool,
) -> tuple[dict[str, pd.Series], set[str]]:
    del columns
    needed_text_columns = {
        "assay_type",
        "target_type",
        "assay_confidence_score",
        "assay_relationship_type",
        "standard_relation",
        "standard_type",
        "standard_units",
        "data_validity_comment",
        "potential_duplicate",
        "document_type",
        "document_source_id",
        "document_source_name",
        "assay_source_id",
        "assay_source_name",
        "assay_variant_id",
        "assay_variant_accession",
        "assay_variant_mutation",
        "smiles",
        "document_chembl_id",
        "document_id",
        "doi",
        "pubmed_id",
        "patent_id",
    }
    empty = pd.Series("", index=df.index, dtype="string")
    text = {
        column: _clean_series(df[column]) if column in df.columns else empty
        for column in needed_text_columns
    }
    numeric_value = pd.to_numeric(df["standard_value"], errors="coerce")
    positive_finite = numeric_value.map(
        lambda value: bool(pd.notna(value) and math.isfinite(float(value)) and float(value) > 0)
    )
    document_year = pd.to_numeric(df["document_year"], errors="coerce")
    valid_year = document_year.map(
        lambda value: bool(
            pd.notna(value)
            and math.isfinite(float(value))
            and float(value).is_integer()
            and 1000 <= int(value) <= 2999
        )
    )
    document_source_name = text["document_source_name"].str.casefold()
    assay_source_name = text["assay_source_name"].str.casefold()
    is_bindingdb_source = document_source_name.str.contains(
        "bindingdb", regex=False
    ) | assay_source_name.str.contains("bindingdb", regex=False)
    discovered_bindingdb_source_ids = {
        value
        for column in ("document_source_id", "assay_source_id")
        for value in text[column][is_bindingdb_source]
        if value
    }
    masks = {
        "assay_type": text["assay_type"].ne("B"),
        "single_protein": text["target_type"].ne("SINGLE PROTEIN"),
        "confidence_9": text["assay_confidence_score"].ne("9"),
        "relationship_d": text["assay_relationship_type"].ne("D"),
        "exact_relation": text["standard_relation"].ne("="),
        "nm_units": text["standard_units"].str.casefold().ne("nm"),
        "endpoint": ~text["standard_type"].str.upper().isin(ENDPOINTS),
        "positive_value": ~positive_finite,
        "validity": ~text["data_validity_comment"].str.casefold().isin(
            {"", "manually validated"}
        ),
        "nonduplicate": ~text["potential_duplicate"].str.casefold().isin(
            {"", "0", "false", "f", "no", "n"}
        ),
        "document_publication": text["document_type"].str.casefold().ne("publication"),
        "dated": ~valid_year,
        "assay_variant": (
            text["assay_variant_id"].ne("")
            | text["assay_variant_accession"].ne("")
            | text["assay_variant_mutation"].ne("")
        ),
        "nonblank_ligand_smiles": text["smiles"].eq(""),
        "publication_identifier": (
            text["document_chembl_id"].eq("")
            & text["document_id"].eq("")
            & text["doi"].eq("")
            & text["pubmed_id"].eq("")
            & text["patent_id"].eq("")
        ),
    }
    if exclude_bindingdb_sources:
        masks["exclude_chembl_bindingdb_source"] = is_bindingdb_source
    return masks, discovered_bindingdb_source_ids


def _chembl_arrow_filter(available: set[str]) -> ds.Expression:
    field = ds.field
    expression = (
        (field("assay_type") == "B")
        & (field("target_type") == "SINGLE PROTEIN")
        & (field("assay_confidence_score") == 9)
        & (field("assay_relationship_type") == "D")
        & (field("standard_relation") == "=")
        & field("standard_type").isin(["EC50", "IC50", "Kd", "Ki", "KD", "KI"])
        & (field("standard_value") > 0)
        & (field("standard_units") == "nM")
        & (field("document_type") == "PUBLICATION")
        & (field("document_year") >= 1000)
        & (field("document_year") <= 2999)
    )
    if "data_validity_comment" in available:
        value = field("data_validity_comment")
        expression &= value.is_null() | value.isin(["", "Manually validated", "manually validated"])
    if "potential_duplicate" in available:
        value = field("potential_duplicate")
        expression &= value.is_null() | (value == False)  # noqa: E712 - Arrow expression
    if "assay_variant_id" in available:
        expression &= field("assay_variant_id").is_null()
    for column in ("assay_variant_accession", "assay_variant_mutation"):
        if column in available:
            value = field(column)
            expression &= value.is_null() | (value == "")
    if "smiles" in available:
        expression &= field("smiles").is_valid() & (field("smiles") != "")
    identifier_expression = None
    for column in ("document_chembl_id", "doi", "pubmed_id", "patent_id"):
        if column not in available:
            continue
        candidate = field(column).is_valid() & (field(column) != "")
        identifier_expression = (
            candidate
            if identifier_expression is None
            else identifier_expression | candidate
        )
    if identifier_expression is not None:
        expression &= identifier_expression
    return expression


def _normalize_chembl(
    path: Path,
    clusters: dict[str, dict[str, str]],
    target_exclusions: dict[str, dict[str, Any]],
    scaffold_cache: dict[str, tuple[str, str, str, str]],
    invalid_structure_cache: dict[str, str],
    *,
    exclude_bindingdb_sources: bool,
) -> tuple[list[dict[str, object]], dict[str, Any], dict[str, list[str]]]:
    required = [
        "assay_type",
        "target_type",
        "assay_confidence_score",
        "assay_relationship_type",
        "standard_relation",
        "standard_type",
        "standard_value",
        "standard_units",
        "data_validity_comment",
        "potential_duplicate",
        "document_type",
        "document_year",
        "uniprot",
        "smiles",
    ]
    optional = [
        "source_db",
        "source_name",
        "source_release",
        "source_version",
        "source_license",
        "activity_id",
        "document_chembl_id",
        "document_id",
        "document_source_id",
        "document_source_name",
        "assay_source_id",
        "assay_source_name",
        "assay_variant_id",
        "assay_variant_accession",
        "assay_variant_mutation",
        "molecule_chembl_id",
        "standard_inchi_key",
        "source_doi",
        "document_doi",
        "doi",
        "patent_id",
        "source_pmid",
        "pmid",
        "pubmed_id",
    ]
    all_columns = [*required, *optional]
    filter_columns = [
        "assay_type",
        "target_type",
        "assay_confidence_score",
        "assay_relationship_type",
        "standard_relation",
        "standard_type",
        "standard_value",
        "standard_units",
        "data_validity_comment",
        "potential_duplicate",
        "document_type",
        "document_year",
        "document_source_id",
        "document_source_name",
        "assay_source_id",
        "assay_source_name",
        "assay_variant_id",
        "assay_variant_accession",
        "assay_variant_mutation",
        "smiles",
        "document_chembl_id",
        "document_id",
        "doi",
        "pubmed_id",
        "patent_id",
    ]
    filter_df = _read_table(path, filter_columns)
    _require_columns(filter_df, path, [column for column in required if column in filter_columns])
    counts: dict[str, Any] = {"input": len(filter_df)}
    reason_masks, discovered_bindingdb_source_ids = _chembl_reason_masks(
        filter_df,
        all_columns,
        exclude_bindingdb_sources=exclude_bindingdb_sources,
    )
    for reason, mask in reason_masks.items():
        counts[f"filtered_{reason}"] = int(mask.sum())
    del reason_masks, filter_df
    gc.collect()

    if path.suffix.lower() == ".parquet":
        dataset = ds.dataset(path, format="parquet")
        available = set(dataset.schema.names)
        selected = [column for column in all_columns if column in available]
        table = dataset.to_table(
            columns=selected,
            filter=_chembl_arrow_filter(available),
        )
        df = table.to_pandas(split_blocks=True, self_destruct=True)
    else:
        df = _read_table(path, all_columns)
    _require_columns(df, path, required)
    final_masks, _ = _chembl_reason_masks(
        df,
        all_columns,
        exclude_bindingdb_sources=exclude_bindingdb_sources,
    )
    rejected = pd.Series(False, index=df.index, dtype=bool)
    for mask in final_masks.values():
        rejected |= mask
    accepted = df.loc[~rejected].copy()
    exclusion_mask = _clean_series(accepted["uniprot"]).isin(target_exclusions)
    counts["filtered_target_sequence_exclusion"] = int(exclusion_mask.sum())
    accepted = accepted.loc[~exclusion_mask]
    rows: list[dict[str, object]] = []
    missing_clusters = sorted(set(_clean_series(accepted["uniprot"])) - set(clusters))
    if missing_clusters:
        raise SystemExit(
            f"Missing target cluster assignment for ChEMBL target {missing_clusters[0]} "
            f"(count={len(missing_clusters)} sample={','.join(missing_clusters[:100])})"
        )
    for _, row in accepted.iterrows():
        endpoint = _clean(row.get("standard_type")).upper()
        value = float(row["standard_value"])
        uniprot = _clean(row.get("uniprot"))
        source_db, source_release, source_license = _source_meta(row, "ChEMBL")
        if source_release and source_release != CHEMBL_REQUIRED_RELEASE:
            raise SystemExit(
                f"ChEMBL evidence source_release must be {CHEMBL_REQUIRED_RELEASE}; observed {source_release}"
            )
        raw_smiles = _clean(row.get("smiles"))
        if raw_smiles in invalid_structure_cache:
            _record_invalid_structure(counts, raw_smiles, invalid_structure_cache[raw_smiles])
            continue
        try:
            benchmark_row = _make_row(
                row=row,
                source_db=source_db,
                source_release=source_release,
                source_license=source_license,
                evidence_id=_clean(row.get("activity_id")),
                document_id=_clean(row.get("document_chembl_id")) or _clean(row.get("document_id")),
                document_type=_clean(row.get("document_type")),
                evidence_date=_parse_date(row.get("document_year")),
                uniprot=uniprot,
                clusters=clusters[uniprot],
                ligand_id=_clean(row.get("molecule_chembl_id")),
                smiles=raw_smiles,
                inchikey=_clean(row.get("standard_inchi_key")),
                endpoint=endpoint,
                relation="=",
                value_nm=value,
                scaffold_cache=scaffold_cache,
            )
        except InvalidLigandStructure as exc:
            invalid_structure_cache[raw_smiles] = str(exc)
            _record_invalid_structure(counts, raw_smiles, str(exc))
            continue
        rows.append(benchmark_row)
    counts["accepted"] = len(rows)
    return rows, counts, {"bindingdb_source_ids": sorted(discovered_bindingdb_source_ids)}


def _normalize_bindingdb(
    path: Path,
    clusters: dict[str, dict[str, str]],
    target_exclusions: dict[str, dict[str, Any]],
    scaffold_cache: dict[str, tuple[str, str, str, str]],
    invalid_structure_cache: dict[str, str],
    required_release: str,
    *,
    source_label: str = "BindingDB",
    default_source_db: str = "BindingDB",
    allowed_origins: set[str] = BINDINGDB_CURATED_ORIGINS,
    origin_filter_name: str = "bindingdb_curated_origin",
) -> tuple[list[dict[str, object]], dict[str, Any]]:
    required = [
        "affinity_type",
        "relation",
        "affinity_value",
        "affinity_unit",
        "publication_date",
        "evidence_date_source",
        "single_chain_target",
        "source_origin",
        "source_release",
        "source_license",
        "uniprot",
        "ligand_smiles",
    ]
    optional = [
        "source_db",
        "source_license_url",
        "chembl_derived_license_flag",
        "evidence_id",
        "source_doi",
        "source_pmid",
        "source_article_id",
        "source_patent",
        "ligand_id",
        "ligand_inchikey",
    ]
    all_columns = [*required, *optional]
    filter_columns = [
        "affinity_type",
        "relation",
        "affinity_value",
        "affinity_unit",
        "publication_date",
        "evidence_date_source",
        "single_chain_target",
        "source_origin",
        "source_release",
        "source_license",
        "chembl_derived_license_flag",
        "ligand_smiles",
        "source_doi",
        "source_pmid",
        "source_patent",
        "source_article_id",
    ]
    filter_df = _read_table(path, filter_columns)
    _require_columns(filter_df, path, [
        "affinity_type",
        "relation",
        "affinity_value",
        "affinity_unit",
        "publication_date",
        "evidence_date_source",
        "single_chain_target",
        "source_origin",
        "source_release",
        "source_license",
    ])
    counts: dict[str, Any] = {"input": len(filter_df)}
    reason_masks, _ = _bindingdb_reason_masks(
        filter_df,
        allowed_origins=allowed_origins,
        origin_filter_name=origin_filter_name,
    )
    for reason, mask in reason_masks.items():
        counts[f"filtered_{reason}"] = int(mask.sum())
    del reason_masks, filter_df
    gc.collect()
    if path.suffix.lower() == ".parquet":
        dataset = ds.dataset(path, format="parquet")
        available = set(dataset.schema.names)
        selected = [column for column in all_columns if column in available]
        table = dataset.to_table(
            columns=selected,
            filter=_bindingdb_arrow_filter(available),
        )
        df = table.to_pandas(split_blocks=True, self_destruct=True)
    else:
        df = _read_table(path, all_columns)
    _require_columns(df, path, required)
    final_masks, evidence_dates = _bindingdb_reason_masks(
        df,
        allowed_origins=allowed_origins,
        origin_filter_name=origin_filter_name,
    )
    rejected = pd.Series(False, index=df.index, dtype=bool)
    for mask in final_masks.values():
        rejected |= mask
    accepted = df.loc[~rejected].copy()
    accepted["__evidence_date"] = evidence_dates.loc[accepted.index]
    exclusion_mask = _clean_series(accepted["uniprot"]).isin(target_exclusions)
    counts["filtered_target_sequence_exclusion"] = int(exclusion_mask.sum())
    accepted = accepted.loc[~exclusion_mask]
    observed_releases = set(_clean_series(accepted["source_release"]))
    if observed_releases != {required_release}:
        observed = ",".join(sorted(value or "<blank>" for value in observed_releases))
        raise SystemExit(
            f"{source_label} evidence source_release must be {required_release}; observed {observed}"
        )
    if "source_db" in accepted.columns:
        observed_sources = set(_clean_series(accepted["source_db"])) - {""}
        if observed_sources and observed_sources != {default_source_db}:
            observed = ",".join(sorted(observed_sources))
            raise SystemExit(
                f"{source_label} evidence source_db must be {default_source_db}; observed {observed}"
            )
    rows: list[dict[str, object]] = []
    missing_clusters = sorted(set(_clean_series(accepted["uniprot"])) - set(clusters))
    if missing_clusters:
        raise SystemExit(
            f"Missing target cluster assignment for {source_label} target {missing_clusters[0]} "
            f"(count={len(missing_clusters)} sample={','.join(missing_clusters[:100])})"
        )
    for _, row in accepted.iterrows():
        endpoint = _clean(row.get("affinity_type")).upper()
        value = float(row["affinity_value"])
        evidence_date = str(row["__evidence_date"])
        uniprot = _clean(row.get("uniprot"))
        source_db, source_release, source_license = _source_meta(row, default_source_db)
        if source_release and source_release != required_release:
            raise SystemExit(
                f"{source_label} evidence source_release must be {required_release}; observed {source_release}"
            )
        raw_smiles = _clean(row.get("ligand_smiles"))
        if raw_smiles in invalid_structure_cache:
            _record_invalid_structure(counts, raw_smiles, invalid_structure_cache[raw_smiles])
            continue
        try:
            benchmark_row = _make_row(
                row=row,
                source_db=source_db,
                source_release=source_release,
                source_license=source_license,
                evidence_id=_clean(row.get("evidence_id")),
                document_id=(
                    _clean(row.get("source_doi"))
                    or _clean(row.get("source_pmid"))
                    or _clean(row.get("source_patent"))
                    or _clean(row.get("source_article_id"))
                ),
                document_type="Publication",
                evidence_date=evidence_date,
                uniprot=uniprot,
                clusters=clusters[uniprot],
                ligand_id=_clean(row.get("ligand_id")),
                smiles=raw_smiles,
                inchikey=_clean(row.get("ligand_inchikey")),
                endpoint=endpoint,
                relation="=",
                value_nm=value,
                scaffold_cache=scaffold_cache,
            )
        except InvalidLigandStructure as exc:
            invalid_structure_cache[raw_smiles] = str(exc)
            _record_invalid_structure(counts, raw_smiles, str(exc))
            continue
        rows.append(benchmark_row)
    counts["accepted"] = len(rows)
    return rows, counts


def _record_invalid_structure(
    counts: dict[str, Any],
    smiles: str,
    reason: str,
) -> None:
    counts["filtered_invalid_ligand_structure"] = (
        int(counts.get("filtered_invalid_ligand_structure", 0)) + 1
    )
    examples = counts.setdefault("invalid_ligand_structure_examples", [])
    digest = _hash_text(smiles)
    if len(examples) < 20 and not any(
        item.get("smiles_sha256") == digest for item in examples
    ):
        examples.append({"smiles_sha256": digest, "reason": reason})


def _make_row(
    *,
    row: pd.Series,
    source_db: str,
    source_release: str,
    source_license: str,
    evidence_id: str,
    document_id: str,
    document_type: str,
    evidence_date: str,
    uniprot: str,
    clusters: dict[str, str],
    ligand_id: str,
    smiles: str,
    inchikey: str,
    endpoint: str,
    relation: str,
    value_nm: float,
    scaffold_cache: dict[str, tuple[str, str, str, str]],
) -> dict[str, object]:
    if not evidence_id:
        evidence_id = _hash_text(
            json.dumps(
                [source_db, document_id, uniprot, ligand_id, smiles, inchikey, endpoint, value_nm],
                separators=(",", ":"),
            )
        )
    if not document_id:
        raise SystemExit("Publication/document identifier is required for document leakage audit")
    if not smiles:
        raise SystemExit("Ligand SMILES is required for Bemis-Murcko scaffold computation")
    if smiles not in scaffold_cache:
        canonical_smiles, parse_mode = _canonical_smiles(smiles)
        scaffold_smiles, scaffold_id = _scaffold(canonical_smiles)
        scaffold_cache[smiles] = (
            canonical_smiles,
            scaffold_smiles,
            scaffold_id,
            parse_mode,
        )
    canonical_smiles, scaffold_smiles, scaffold_id, _parse_mode = scaffold_cache[smiles]
    potency = _pactivity(value_nm)
    ligand_key = (inchikey or ligand_id or smiles).upper()
    split = _split_for(evidence_date)
    publication_key = _publication_key(row, source_db, document_id)
    payload = [
        source_db,
        evidence_id,
        publication_key,
        uniprot,
        ligand_key,
        endpoint,
        evidence_date,
    ]
    return {
        "benchmark_id": _hash_text(json.dumps(payload, separators=(",", ":"))),
        "source_db": source_db,
        "source_release": source_release,
        "source_license": source_license,
        "evidence_id": evidence_id,
        "source_document_id": f"{source_db}:{document_id}",
        "publication_key": publication_key,
        "source_document_type": document_type,
        "evidence_date": evidence_date,
        "split": split,
        "uniprot": uniprot,
        "target_cluster_30": clusters["target_cluster_30"],
        "target_cluster_50": clusters["target_cluster_50"],
        "ligand_id": ligand_id,
        "ligand_smiles": smiles,
        "ligand_inchikey": inchikey,
        "scaffold_smiles": scaffold_smiles,
        "scaffold_id": scaffold_id,
        "endpoint": endpoint,
        "relation": relation,
        "standard_value_nm": value_nm,
        "pactivity": potency,
        "activity_class": _activity_class(potency),
        "binary_label": True if potency >= 5.0 else None,
    }


def _dedupe(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], int]:
    seen: set[tuple[object, ...]] = set()
    out: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda item: (item["evidence_date"], item["source_db"], item["evidence_id"])):
        ligand_key = (_clean(row["ligand_inchikey"]) or _clean(row["ligand_id"]) or _clean(row["ligand_smiles"])).upper()
        key = (
            row["publication_key"],
            row["uniprot"],
            ligand_key,
            row["endpoint"],
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out, len(rows) - len(out)


def _ligand_key_series(df: pd.DataFrame) -> pd.Series:
    return (
        df["ligand_inchikey"]
        .fillna("")
        .replace("", pd.NA)
        .fillna(df["ligand_id"])
        .fillna(df["ligand_smiles"])
        .str.upper()
    )


def _first_observation_filter(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Keep evidence only in the first split where its pair and publication occur."""
    split_order = {"train": 0, "dev": 1, "test": 2}
    working = df.copy()
    pair_keys = list(zip(working["uniprot"], _ligand_key_series(working), strict=True))
    orders = working["split"].map(split_order)
    if orders.isna().any():
        raise SystemExit("Unknown split encountered during first-observation filtering")

    pair_first: dict[tuple[object, object], int] = {}
    publication_first: dict[str, int] = {}
    for pair, publication, order in zip(pair_keys, working["publication_key"], orders, strict=True):
        order_int = int(order)
        pair_first[pair] = min(pair_first.get(pair, order_int), order_int)
        publication_first[publication] = min(publication_first.get(publication, order_int), order_int)

    pair_seen_prior = pd.Series(
        [int(order) > pair_first[pair] for pair, order in zip(pair_keys, orders, strict=True)],
        index=working.index,
        dtype=bool,
    )
    publication_seen_prior = pd.Series(
        [
            int(order) > publication_first[publication]
            for publication, order in zip(working["publication_key"], orders, strict=True)
        ],
        index=working.index,
        dtype=bool,
    )
    removed = pair_seen_prior | publication_seen_prior
    removed_by_split = {
        split: int((removed & working["split"].eq(split)).sum())
        for split in ("train", "dev", "test")
    }
    audit = {
        "policy": (
            "retain a row only in the earliest temporal split containing both its canonical "
            "compound-target pair and canonical publication key"
        ),
        "input_after_deduplication": int(len(working)),
        "retained": int((~removed).sum()),
        "removed_total": int(removed.sum()),
        "removed_pair_seen_in_prior_split": int(pair_seen_prior.sum()),
        "removed_publication_seen_in_prior_split": int(publication_seen_prior.sum()),
        "removed_for_both_reasons": int((pair_seen_prior & publication_seen_prior).sum()),
        "removed_by_split": removed_by_split,
    }
    retained = working.loc[~removed, OUTPUT_COLUMNS].reset_index(drop=True)
    return retained, audit


def _overlap_summary(left: set[object], right: set[object], *, sample_size: int = 20) -> dict[str, Any]:
    overlap = sorted(str(value) for value in (left & right) if _clean(value))
    return {"count": len(overlap), "sample": overlap[:sample_size]}


def _leakage_audit(df: pd.DataFrame) -> dict[str, Any]:
    audit: dict[str, Any] = {}
    split_frames = {name: df[df["split"].eq(name)] for name in ("train", "dev", "test")}
    keys = {
        "pair": lambda frame: set(
            zip(
                frame["uniprot"],
                _ligand_key_series(frame),
                strict=True,
            )
        ),
        "publication": lambda frame: set(frame["publication_key"]),
        "scaffold": lambda frame: set(frame["scaffold_id"]),
        "target_cluster_30": lambda frame: set(frame["target_cluster_30"]),
        "target_cluster_50": lambda frame: set(frame["target_cluster_50"]),
    }
    for key, extractor in keys.items():
        values = {split: extractor(frame) for split, frame in split_frames.items()}
        audit[key] = {
            "train_dev": _overlap_summary(values["train"], values["dev"]),
            "train_test": _overlap_summary(values["train"], values["test"]),
            "dev_test": _overlap_summary(values["dev"], values["test"]),
        }
    return audit


def _assert_no_forbidden_overlap(audit: dict[str, Any]) -> None:
    violations = []
    for key in ("pair", "publication"):
        pairs = audit.get(key, {})
        for pair, overlap in pairs.items():
            if overlap.get("count", 0):
                violations.append(f"{key}.{pair}={overlap['count']}")
    if violations:
        raise SystemExit("Forbidden split leakage overlap: " + ", ".join(violations))


def _with_cold_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    train = df[df["split"].eq("train")]
    prior = df[df["split"].isin(["train", "dev"])]

    pair_series = list(
        zip(
            df["uniprot"],
            _ligand_key_series(df),
            strict=True,
        )
    )
    train_pairs = set(
        zip(
            train["uniprot"],
            _ligand_key_series(train),
            strict=True,
        )
    )
    prior_pairs = set(
        zip(
            prior["uniprot"],
            _ligand_key_series(prior),
            strict=True,
        )
    )
    out["absent_pair_from_train"] = [pair not in train_pairs for pair in pair_series]
    out["absent_pair_from_prior_splits"] = [pair not in prior_pairs for pair in pair_series]
    for column, suffix in [
        ("publication_key", "publication"),
        ("scaffold_id", "scaffold"),
        ("target_cluster_30", "target_cluster_30"),
        ("target_cluster_50", "target_cluster_50"),
    ]:
        train_values = set(train[column])
        prior_values = set(prior[column])
        out[f"absent_{suffix}_from_train"] = ~out[column].isin(train_values)
        out[f"absent_{suffix}_from_prior_splits"] = ~out[column].isin(prior_values)
    return out


def _evaluation_views(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    flagged = _with_cold_flags(df)
    test = flagged[flagged["split"].eq("test")].copy()
    views = {
        "temporal_test.parquet": test,
        "ligand_scaffold_cold.parquet": test[
            test["absent_pair_from_prior_splits"] & test["absent_publication_from_prior_splits"] & test["absent_scaffold_from_prior_splits"]
        ],
        "target_cold30.parquet": test[
            test["absent_pair_from_prior_splits"] & test["absent_publication_from_prior_splits"] & test["absent_target_cluster_30_from_prior_splits"]
        ],
        "target_cold50.parquet": test[
            test["absent_pair_from_prior_splits"] & test["absent_publication_from_prior_splits"] & test["absent_target_cluster_50_from_prior_splits"]
        ],
        "dual_cold.parquet": test[
            test["absent_pair_from_prior_splits"]
            & test["absent_publication_from_prior_splits"]
            & test["absent_scaffold_from_prior_splits"]
            & test["absent_target_cluster_30_from_prior_splits"]
            & test["absent_target_cluster_50_from_prior_splits"]
        ],
    }
    out: dict[str, pd.DataFrame] = {}
    for name, frame in views.items():
        view = frame.copy()
        view["evaluation_view"] = name.removesuffix(".parquet")
        view["claimable"] = len(view) > 0
        out[name] = view[VIEW_COLUMNS].reset_index(drop=True)
    return out


def _write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    table = pa.Table.from_pandas(df[OUTPUT_COLUMNS], schema=OUTPUT_SCHEMA, preserve_index=False)
    pq.write_table(table, tmp, compression="snappy")
    tmp.replace(path)


def _write_view_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    table = pa.Table.from_pandas(df[VIEW_COLUMNS], schema=VIEW_SCHEMA, preserve_index=False)
    pq.write_table(table, tmp, compression="snappy")
    tmp.replace(path)


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _source_manifest(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = _read_json(path)
    return {"path": str(path), "sha256": _sha256_file(path), "payload": payload}


def _manifest_release(payload: dict[str, Any]) -> str:
    source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
    return _clean(source.get("release")) or _clean(source.get("source_release")) or _clean(payload.get("source_release"))


def _manifest_license(payload: dict[str, Any]) -> tuple[str, str]:
    source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
    license_text = (
        _clean(source.get("license"))
        or _clean(source.get("source_license"))
        or _clean(payload.get("source_license"))
    )
    license_url = (
        _clean(source.get("license_url"))
        or _clean(source.get("source_license_url"))
        or _clean(payload.get("source_license_url"))
    )
    return license_text, license_url


def _validate_source_manifest(
    label: str,
    manifest_path: Path | None,
    evidence_path: Path,
    *,
    required_release: str,
    current_sha256: str,
    current_rows: int,
    required_schema_version: str | None = None,
) -> dict[str, Any]:
    if manifest_path is None:
        raise SystemExit(f"--{label}-manifest is required when --{label}-evidence is provided")
    manifest = _source_manifest(manifest_path)
    assert manifest is not None
    payload = manifest["payload"]
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} source manifest must be a JSON object: {manifest_path}")
    if (
        required_schema_version is not None
        and payload.get("schema_version") != required_schema_version
    ):
        raise SystemExit(
            f"{label} source manifest schema must be {required_schema_version}; "
            f"observed {payload.get('schema_version') or '<blank>'}"
        )
    release = _manifest_release(payload)
    if release != required_release:
        raise SystemExit(
            f"{label} source manifest release must be {required_release}; observed {release or '<blank>'}"
        )
    license_text, license_url = _manifest_license(payload)
    if not license_text:
        raise SystemExit(f"{label} source manifest must declare a nonblank license")
    sha_values: set[str] = set()
    output_sha = payload.get("output_sha256")
    if isinstance(output_sha, dict):
        sha_values.update(str(value) for value in output_sha.values() if _clean(value))
    artifacts = payload.get("artifacts")
    if isinstance(artifacts, dict):
        for artifact in artifacts.values():
            if isinstance(artifact, dict) and _clean(artifact.get("sha256")):
                sha_values.add(str(artifact["sha256"]))
    if sha_values and current_sha256 not in sha_values:
        raise SystemExit(
            f"{label} source manifest does not match current evidence artifact sha256 for {evidence_path}"
        )
    row_values: set[int] = set()
    row_counts = payload.get("row_counts") or payload.get("counts")
    if isinstance(row_counts, dict):
        for key in ("activity_evidence", "input_rows_seen"):
            value = row_counts.get(key)
            if isinstance(value, int):
                row_values.add(value)
    for key in ("rows", "row_count", "n_rows"):
        value = payload.get(key)
        if isinstance(value, int):
            row_values.add(value)
    if isinstance(artifacts, dict):
        for artifact in artifacts.values():
            if isinstance(artifact, dict):
                for key in ("rows", "row_count", "n_rows"):
                    value = artifact.get(key)
                    if isinstance(value, int):
                        row_values.add(value)
    if row_values and current_rows not in row_values:
        raise SystemExit(
            f"{label} source manifest row count does not match current evidence rows for {evidence_path}"
        )
    manifest["validated"] = {
        "required_release": required_release,
        "source_license": license_text,
        "source_license_url": license_url or None,
        "current_evidence_sha256": current_sha256,
        "current_evidence_rows": current_rows,
        "validated_sha256_when_exposed": bool(sha_values),
        "validated_row_count_when_exposed": bool(row_values),
    }
    return manifest


def build_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_input_rows < 1:
        raise SystemExit("--max-input-rows must be >= 1")
    bindingdb_required_release = _clean(args.bindingdb_required_release)
    if not bindingdb_required_release:
        raise SystemExit("--bindingdb-required-release must be nonblank")
    gtopdb_required_release = _clean(args.gtopdb_required_release)
    if not gtopdb_required_release:
        raise SystemExit("--gtopdb-required-release must be nonblank")
    if (
        args.chembl_evidence is None
        and args.bindingdb_evidence is None
        and args.gtopdb_evidence is None
    ):
        raise SystemExit("At least one evidence input is required")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _remove_outputs(args.out_dir)
    clusters, cluster_manifest, target_exclusions = _load_clusters(
        args.target_clusters,
        args.target_cluster_manifest,
    )
    source_target_exclusions, source_target_contract = _load_source_target_exclusions(
        args.source_target_exclusions
    )
    rows: list[dict[str, object]] = []
    filter_counts: dict[str, dict[str, Any]] = {}
    input_hashes: dict[str, str] = {
        "target_clusters": cluster_manifest["sha256"],
        "target_cluster_manifest": cluster_manifest["manifest_sha256"],
    }
    if args.source_target_exclusions is not None:
        input_hashes["source_target_exclusions"] = source_target_contract["artifact"][
            "sha256"
        ]
    source_manifests: dict[str, Any] = {}
    scaffold_cache: dict[str, tuple[str, str, str, str]] = {}
    invalid_structure_cache: dict[str, str] = {}
    try:
        if args.chembl_evidence is not None:
            chembl_input_rows = _input_row_count(args.chembl_evidence)
            if chembl_input_rows > args.max_input_rows:
                raise SystemExit(
                    f"Input rows {chembl_input_rows} exceed fail-closed --max-input-rows={args.max_input_rows}"
                )
            input_hashes["chembl_evidence"] = _sha256_file(args.chembl_evidence)
            source_manifests["chembl"] = _validate_source_manifest(
                "chembl",
                args.chembl_manifest,
                args.chembl_evidence,
                required_release=CHEMBL_REQUIRED_RELEASE,
                current_sha256=input_hashes["chembl_evidence"],
                current_rows=chembl_input_rows,
            )
            chembl_rows, counts, source_discovery = _normalize_chembl(
                args.chembl_evidence,
                clusters,
                target_exclusions,
                scaffold_cache,
                invalid_structure_cache,
                exclude_bindingdb_sources=args.bindingdb_evidence is not None,
            )
            filter_counts["chembl"] = counts
            rows.extend(chembl_rows)
        else:
            source_discovery = {}
        if args.bindingdb_evidence is not None:
            bindingdb_input_rows = _input_row_count(args.bindingdb_evidence)
            if bindingdb_input_rows > args.max_input_rows:
                raise SystemExit(
                    f"Input rows {bindingdb_input_rows} exceed fail-closed --max-input-rows={args.max_input_rows}"
                )
            input_hashes["bindingdb_evidence"] = _sha256_file(args.bindingdb_evidence)
            source_manifests["bindingdb"] = _validate_source_manifest(
                "bindingdb",
                args.bindingdb_manifest,
                args.bindingdb_evidence,
                required_release=bindingdb_required_release,
                current_sha256=input_hashes["bindingdb_evidence"],
                current_rows=bindingdb_input_rows,
            )
            bindingdb_rows, counts = _normalize_bindingdb(
                args.bindingdb_evidence,
                clusters,
                target_exclusions,
                scaffold_cache,
                invalid_structure_cache,
                bindingdb_required_release,
            )
            filter_counts["bindingdb"] = counts
            rows.extend(bindingdb_rows)
        if args.gtopdb_evidence is not None:
            gtopdb_input_rows = _input_row_count(args.gtopdb_evidence)
            if gtopdb_input_rows > args.max_input_rows:
                raise SystemExit(
                    f"Input rows {gtopdb_input_rows} exceed fail-closed --max-input-rows={args.max_input_rows}"
                )
            input_hashes["gtopdb_evidence"] = _sha256_file(args.gtopdb_evidence)
            source_manifests["gtopdb"] = _validate_source_manifest(
                "gtopdb",
                args.gtopdb_manifest,
                args.gtopdb_evidence,
                required_release=gtopdb_required_release,
                current_sha256=input_hashes["gtopdb_evidence"],
                current_rows=gtopdb_input_rows,
                required_schema_version="gtopdb_activity_evidence.v1",
            )
            gtopdb_rows, counts = _normalize_bindingdb(
                args.gtopdb_evidence,
                clusters,
                target_exclusions,
                scaffold_cache,
                invalid_structure_cache,
                gtopdb_required_release,
                source_label="GtoPdb",
                default_source_db="GtoPdb",
                allowed_origins=GTOPDB_CURATED_ORIGINS,
                origin_filter_name="gtopdb_expert_curated_origin",
            )
            filter_counts["gtopdb"] = counts
            rows.extend(gtopdb_rows)
        rows, source_target_audit, source_target_filtered = (
            _apply_source_target_exclusions(
                rows,
                source_target_exclusions,
                source_target_contract,
            )
        )
        for source, counts in filter_counts.items():
            filtered = int(source_target_filtered.get(source, 0))
            counts["accepted_before_source_target_exclusions"] = counts["accepted"]
            counts["filtered_source_document_target_mismatch"] = filtered
            counts["accepted"] -= filtered
        total_inputs = sum(counts["input"] for counts in filter_counts.values())
        if total_inputs > args.max_input_rows:
            raise SystemExit(
                f"Input rows {total_inputs} exceed fail-closed --max-input-rows={args.max_input_rows}"
            )
        if not rows:
            raise SystemExit("No claim-grade activity rows survived canonical filters")
        rows, duplicate_rows = _dedupe(rows)
        raw_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
        raw_audit = _leakage_audit(raw_df)
        df, first_observation_audit = _first_observation_filter(raw_df)
        if df.empty:
            raise SystemExit("No activity rows survived first-observation leakage filtering")
        audit = _leakage_audit(df)
        _assert_no_forbidden_overlap(audit)
        split_counts = {split: int(df["split"].eq(split).sum()) for split in ("train", "dev", "test")}
        empty_splits = [split for split, count in split_counts.items() if count == 0]
        if empty_splits:
            raise SystemExit(f"Temporal train/dev/test splits must be nonempty; empty: {', '.join(empty_splits)}")
        for split in ("train", "dev", "test"):
            _write_parquet_atomic(df[df["split"].eq(split)].reset_index(drop=True), args.out_dir / f"{split}.parquet")
        views = _evaluation_views(df)
        for name, view in views.items():
            _write_view_parquet_atomic(view, args.out_dir / name)
        output_hashes = {
            name: _sha256_file(args.out_dir / name)
            for name in ("train.parquet", "dev.parquet", "test.parquet", *VIEW_NAMES)
        }
        view_counts = {name.removesuffix(".parquet"): len(view) for name, view in views.items()}
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "input_sha256": input_hashes,
            "output_sha256": output_hashes,
            "source_manifests": source_manifests,
            "versions": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "pandas": pd.__version__,
                "pyarrow": pa.__version__,
            },
            "licenses": {
                source: _manifest_license(manifest["payload"])[0]
                for source, manifest in source_manifests.items()
                if manifest is not None
            },
            "filters": {
                "ligand_structure_policy": {
                    "parser": "RDKit SmilesParserParams",
                    "strict_first": True,
                    "allow_cxsmiles": True,
                    "parse_name": False,
                    "lenient_cx_metadata_fallback": (
                        "strictCXSMILES=False only after strict parsing fails; molecular graph "
                        "parsing and sanitization must still succeed"
                    ),
                    "unique_input_parse_modes": dict(
                        sorted(Counter(value[3] for value in scaffold_cache.values()).items())
                    ),
                    "invalid_nonblank_unique_inputs": len(invalid_structure_cache),
                    "invalid_nonblank_smiles_sha256": sorted(
                        _hash_text(smiles) for smiles in invalid_structure_cache
                    ),
                    "invalid_graph_policy": (
                        "exclude with per-source row counts and hashed examples only after strict "
                        "and lenient-CX graph parsing both fail"
                    ),
                },
                "chembl_claim_grade": {
                    "assay_type": "B",
                    "target_type": "SINGLE PROTEIN",
                    "confidence_score": 9,
                    "relationship_type": "D",
                    "standard_relation": "=",
                    "standard_units": "nM",
                    "standard_types": sorted(ENDPOINTS),
                    "standard_value": ">0",
                    "validity": "NULL or manually validated",
                    "potential_duplicate": False,
                    "document_type": "Publication",
                    "publication_identifier": "ChEMBL document ID, DOI, PMID, or patent required",
                    "dated": True,
                    "exclude_bindingdb_source_name_when_bindingdb_separate": args.bindingdb_evidence is not None,
                    "bindingdb_source_ids_discovered": source_discovery.get("bindingdb_source_ids", []),
                    "exclude_assay_variants": True,
                    "required_release": CHEMBL_REQUIRED_RELEASE,
                    "assay_variant_id_nonvariant_policy": "NULL/blank required; -1 is treated as a variant marker and rejected",
                    "ligand_smiles": "nonblank and RDKit-parseable",
                },
                "bindingdb_claim_grade": {
                    "required_release": bindingdb_required_release,
                    "default_required_release": BINDINGDB_DEFAULT_REQUIRED_RELEASE,
                    "standard_types": sorted(ENDPOINTS),
                    "relation": "=",
                    "affinity_unit": "nM",
                    "affinity_value": ">0",
                    "evidence_date_source": "publication",
                    "publication_identifier": "DOI, PMID, patent, or source article ID required",
                    "single_chain_target": True,
                    "source_origin_policy": (
                        "Only BindingDB-curated literature and patent origins are accepted; "
                        "imported ChEMBL, PubChem, PDSP, D3R, CSAR, and other external origins "
                        "are excluded until independently licensed and deduplicated"
                    ),
                    "ligand_smiles": "nonblank and RDKit-parseable",
                },
                "gtopdb_claim_grade": {
                    "required_release": gtopdb_required_release,
                    "source_manifest_schema": "gtopdb_activity_evidence.v1",
                    "standard_types": sorted(ENDPOINTS),
                    "relation": "=",
                    "affinity_unit": "nM",
                    "affinity_value": ">0",
                    "evidence_date_source": "publication",
                    "publication_identifier": "NCBI-resolved PMID required",
                    "single_chain_target": True,
                    "source_origin_policy": (
                        "GtoPdb expert-curated rows require explicit source_origin, release, "
                        "license, and a false ChEMBL-derived flag"
                    ),
                    "upstream_filters": (
                        "human canonical single-protein mapping, supported small-molecule type, "
                        "RDKit/InChIKey consistency, exact Ki/Kd/IC50/EC50, nM/pActivity "
                        "consistency, and conservative latest-supporting-PMID date"
                    ),
                    "ligand_smiles": "nonblank and RDKit-parseable",
                },
                "source_target_exclusions": source_target_audit,
                "filter_counts": filter_counts,
                "pair_source_document_duplicates_removed": duplicate_rows,
            },
            "splits": {
                "boundaries": {
                    "train": "<=2023-12-31",
                    "dev": "2024-01-01..2024-12-31",
                    "test": ">=2025-01-01",
                },
                "counts": split_counts,
                "primary": "temporal train/dev/test",
                "first_observation_filter": first_observation_audit,
            },
            "evaluation_views": {
                "outputs": list(VIEW_NAMES),
                "counts": view_counts,
                "claimable": {name: count > 0 for name, count in view_counts.items()},
                "policy": {
                    "temporal_test": "test rows after hard pair/publication leakage checks",
                    "ligand_scaffold_cold": "test rows whose compound-target pair, publication, and scaffold are absent from train/dev",
                    "target_cold30": "test rows whose compound-target pair, publication, and 30% target cluster are absent from train/dev",
                    "target_cold50": "test rows whose compound-target pair, publication, and 50% target cluster are absent from train/dev",
                    "dual_cold": "test rows whose compound-target pair, publication, scaffold, and both 30% and 50% target clusters are absent from train/dev",
                    "empty_views": "explicit but non-claimable",
                },
            },
            "label_semantics": {
                "pactivity": "-log10(activity_molar) = 9 - log10(value_nM)",
                "strong_positive": "pActivity >= 7",
                "weak_positive": "5 <= pActivity < 7",
                "low_potency_quantitative": "pActivity < 5; retained as quantitative evidence, not fabricated negative",
                "binary_label": "True only for pActivity >= 5; null below 5 because missing/comment text is never converted to negatives",
            },
            "target_cluster_policy": {
                "required_input": cluster_manifest,
                "main_threshold": "30%",
                "sensitivity_threshold": "50%",
                "fail_closed_on_missing_assignment": True,
                "audited_sequence_exclusions": {
                    "count": len(target_exclusions),
                    "records": [target_exclusions[key] for key in sorted(target_exclusions)],
                    "filtered_evidence_rows_by_source": {
                        source: counts.get("filtered_target_sequence_exclusion", 0)
                        for source, counts in filter_counts.items()
                    },
                    "policy": (
                        "Only exclusions carried by the provenance-bound target cluster manifest "
                        "are removed; every other claim-grade evidence target requires a sequence "
                        "cluster assignment."
                    ),
                },
            },
            "raw_temporal_overlap_audit": raw_audit,
            "leakage_overlap_audit": audit,
            "manifest_sha256_policy": "manifest.json is excluded from output_sha256 because stable self-hashing is impossible",
        }
        _write_json_atomic(manifest, args.out_dir / "manifest.json")
        return manifest
    except BaseException:
        _remove_outputs(args.out_dir)
        raise


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chembl-evidence", type=Path)
    parser.add_argument("--bindingdb-evidence", type=Path)
    parser.add_argument("--gtopdb-evidence", type=Path)
    parser.add_argument("--target-clusters", required=True, type=Path)
    parser.add_argument("--target-cluster-manifest", required=True, type=Path)
    parser.add_argument("--source-target-exclusions", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--chembl-manifest", type=Path)
    parser.add_argument("--bindingdb-manifest", type=Path)
    parser.add_argument("--gtopdb-manifest", type=Path)
    parser.add_argument("--bindingdb-required-release", default=BINDINGDB_DEFAULT_REQUIRED_RELEASE)
    parser.add_argument("--gtopdb-required-release", default=GTOPDB_DEFAULT_REQUIRED_RELEASE)
    parser.add_argument("--max-input-rows", type=int, default=10_000_000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        manifest = build_benchmark(args)
    except Exception as exc:
        print(f"[activity.benchmark][FATAL] {exc}", file=sys.stderr)
        return 1
    counts = manifest["splits"]["counts"]
    print(
        "[activity.benchmark] wrote "
        f"train={counts['train']} dev={counts['dev']} test={counts['test']} "
        f"out_dir={args.out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
