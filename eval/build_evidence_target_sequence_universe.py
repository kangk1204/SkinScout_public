#!/usr/bin/env python3
"""Build a provenance-bound target sequence universe for activity benchmarks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow.dataset as ds

try:
    from build_activity_benchmark import (
        BINDINGDB_DEFAULT_REQUIRED_RELEASE,
        CHEMBL_REQUIRED_RELEASE,
        GTOPDB_CURATED_ORIGINS,
        GTOPDB_DEFAULT_REQUIRED_RELEASE,
        _bindingdb_arrow_filter,
        _bindingdb_reason_masks,
        _chembl_arrow_filter,
        _chembl_reason_masks,
        _clean_series,
        _input_row_count,
        _sha256_file,
        _validate_source_manifest,
    )
    from build_target_cluster_map import _fasta_records
except ImportError:  # pragma: no cover - module execution from repository root
    from eval.build_activity_benchmark import (
        BINDINGDB_DEFAULT_REQUIRED_RELEASE,
        CHEMBL_REQUIRED_RELEASE,
        GTOPDB_CURATED_ORIGINS,
        GTOPDB_DEFAULT_REQUIRED_RELEASE,
        _bindingdb_arrow_filter,
        _bindingdb_reason_masks,
        _chembl_arrow_filter,
        _chembl_reason_masks,
        _clean_series,
        _input_row_count,
        _sha256_file,
        _validate_source_manifest,
    )
    from eval.build_target_cluster_map import _fasta_records


SCHEMA_VERSION = "skinscout.evidence-target-sequence-universe.v2"
UNIPROT_METADATA_SCHEMA_VERSION = "skinscout.uniprot-search-cache.v1"
UNIPROT_ENDPOINT = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_FIELDS = (
    "accession",
    "id",
    "reviewed",
    "organism_id",
    "organism_name",
    "length",
    "sequence",
    "sequence_version",
)
UNIPROT_TSV_COLUMNS = (
    "Entry",
    "Entry Name",
    "Reviewed",
    "Organism (ID)",
    "Organism",
    "Length",
    "Sequence",
    "Sequence version",
)
UNIPROT_LICENSE = "CC BY 4.0"
UNIPROT_LICENSE_URL = "https://www.uniprot.org/help/license"
SEQUENCE_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYBXZJUO]+$")


def _hash_accessions(accessions: Iterable[str]) -> str:
    payload = "\n".join(sorted(set(accessions))) + "\n"
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_text_atomic(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _remove_derived_outputs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.unlink(missing_ok=True)
        path.with_suffix(path.suffix + ".tmp").unlink(missing_ok=True)


def _accepted_targets(
    path: Path,
    *,
    source: str,
    exclude_bindingdb_sources: bool,
) -> tuple[set[str], int]:
    if path.suffix.lower() != ".parquet":
        raise SystemExit(f"{source} evidence must be Parquet for bounded target extraction: {path}")
    dataset = ds.dataset(path, format="parquet")
    available = set(dataset.schema.names)
    if "uniprot" not in available:
        raise SystemExit(f"{path} missing required column: uniprot")

    if source == "chembl":
        columns = [
            "uniprot",
            "smiles",
            "document_chembl_id",
            "document_id",
            "doi",
            "pubmed_id",
            "patent_id",
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
        ]
        selected = [column for column in columns if column in available]
        table = dataset.to_table(columns=selected, filter=_chembl_arrow_filter(available))
        frame = table.to_pandas(split_blocks=True, self_destruct=True)
        masks, _ = _chembl_reason_masks(
            frame,
            selected,
            exclude_bindingdb_sources=exclude_bindingdb_sources,
        )
    elif source in {"bindingdb", "gtopdb"}:
        columns = [
            "uniprot",
            "ligand_smiles",
            "source_doi",
            "source_pmid",
            "source_patent",
            "source_article_id",
            "affinity_type",
            "relation",
            "affinity_value",
            "affinity_unit",
            "publication_date",
            "evidence_date_source",
            "single_chain_target",
            "source_origin",
            "source_license",
            "source_release",
            "chembl_derived_license_flag",
        ]
        selected = [column for column in columns if column in available]
        table = dataset.to_table(columns=selected, filter=_bindingdb_arrow_filter(available))
        frame = table.to_pandas(split_blocks=True, self_destruct=True)
        reason_kwargs: dict[str, Any] = {}
        if source == "gtopdb":
            reason_kwargs = {
                "allowed_origins": GTOPDB_CURATED_ORIGINS,
                "origin_filter_name": "gtopdb_expert_curated_origin",
            }
        masks, _ = _bindingdb_reason_masks(frame, **reason_kwargs)
    else:  # pragma: no cover - internal invariant
        raise AssertionError(source)

    rejected = pd.Series(False, index=frame.index, dtype=bool)
    for mask in masks.values():
        rejected |= mask
    accepted = frame.loc[~rejected, "uniprot"]
    accessions = set(_clean_series(accepted)) - {""}
    return accessions, int((~rejected).sum())


def _validated_evidence_source(
    *,
    label: str,
    evidence_path: Path,
    manifest_path: Path,
    required_release: str,
    required_schema_version: str | None = None,
) -> dict[str, Any]:
    evidence_sha256 = _sha256_file(evidence_path)
    evidence_rows = _input_row_count(evidence_path)
    validated = _validate_source_manifest(
        label,
        manifest_path,
        evidence_path,
        required_release=required_release,
        current_sha256=evidence_sha256,
        current_rows=evidence_rows,
        required_schema_version=required_schema_version,
    )
    return {
        "evidence_path": str(evidence_path.resolve()),
        "evidence_sha256": evidence_sha256,
        "evidence_rows": evidence_rows,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": validated["sha256"],
        "required_release": required_release,
        "validation": validated["validated"],
    }


def _decode_header(response: Any, name: str) -> str:
    value = response.headers.get(name)
    return str(value).strip() if value is not None else ""


def _request_tsv(url: str, *, attempts: int = 3) -> tuple[str, dict[str, str]]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "SkinScout/target-sequence-universe"},
    )
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = response.read().decode("utf-8")
                return body, {
                    "release": _decode_header(response, "X-UniProt-Release"),
                    "release_date": _decode_header(response, "X-UniProt-Release-Date"),
                    "total_results": _decode_header(response, "X-Total-Results"),
                    "response_date": _decode_header(response, "Date"),
                    "api_deployment_date": _decode_header(
                        response, "X-API-Deployment-Date"
                    ),
                }
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(2 ** (attempt - 1))
    raise SystemExit(f"UniProt request failed after {attempts} attempts: {last_error}")


def _fetch_uniprot_cache(
    accessions: list[str],
    *,
    tsv_path: Path,
    metadata_path: Path,
    required_release: str,
    batch_size: int,
) -> dict[str, Any]:
    all_rows: list[list[str]] = []
    observed_release = ""
    observed_release_date = ""
    batches: list[dict[str, Any]] = []
    for start in range(0, len(accessions), batch_size):
        batch = accessions[start : start + batch_size]
        query = " OR ".join(f"accession:{accession}" for accession in batch)
        params = urllib.parse.urlencode(
            {
                "query": query,
                "fields": ",".join(UNIPROT_FIELDS),
                "format": "tsv",
                "size": max(1, batch_size),
            }
        )
        url = f"{UNIPROT_ENDPOINT}?{params}"
        body, response_meta = _request_tsv(url)
        reader = csv.reader(body.splitlines(), delimiter="\t")
        rows = list(reader)
        if not rows or tuple(rows[0]) != UNIPROT_TSV_COLUMNS:
            raise SystemExit("UniProt TSV response schema does not match the pinned field contract")
        result_rows = rows[1:]
        try:
            total_results = int(response_meta["total_results"])
        except ValueError as exc:
            raise SystemExit("UniProt response lacks a valid X-Total-Results header") from exc
        if total_results != len(result_rows):
            raise SystemExit(
                "UniProt response pagination mismatch: "
                f"header={total_results} rows={len(result_rows)}"
            )
        release = response_meta["release"]
        release_date = response_meta["release_date"]
        if not release or not release_date:
            raise SystemExit("UniProt response lacks release provenance headers")
        if release != required_release:
            raise SystemExit(
                f"UniProt release must be {required_release}; observed {release}"
            )
        if observed_release and release != observed_release:
            raise SystemExit("UniProt release changed between query batches")
        if observed_release_date and release_date != observed_release_date:
            raise SystemExit("UniProt release date changed between query batches")
        observed_release = release
        observed_release_date = release_date
        all_rows.extend(result_rows)
        batches.append(
            {
                "requested_count": len(batch),
                "requested_accessions_sha256": _hash_accessions(batch),
                "returned_count": len(result_rows),
                "url": url,
                "response_date": response_meta["response_date"],
                "api_deployment_date": response_meta["api_deployment_date"],
            }
        )

    lines = ["\t".join(UNIPROT_TSV_COLUMNS)]
    lines.extend("\t".join(row) for row in all_rows)
    _write_text_atomic("\n".join(lines) + "\n", tsv_path)
    metadata = {
        "schema_version": UNIPROT_METADATA_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": {
            "name": "UniProtKB",
            "endpoint": UNIPROT_ENDPOINT,
            "release": observed_release,
            "release_date": observed_release_date,
            "license": UNIPROT_LICENSE,
            "license_url": UNIPROT_LICENSE_URL,
        },
        "query": {
            "fields": list(UNIPROT_FIELDS),
            "requested_count": len(accessions),
            "requested_accessions_sha256": _hash_accessions(accessions),
            "batch_size": batch_size,
            "batches": batches,
        },
        "artifact": {
            "path": str(tsv_path.resolve()),
            "sha256": _sha256_file(tsv_path),
            "rows": len(all_rows),
        },
    }
    _write_json_atomic(metadata, metadata_path)
    return metadata


def _load_uniprot_cache(
    *,
    tsv_path: Path,
    metadata_path: Path,
    required_release: str,
    requested_accessions: list[str],
) -> dict[str, Any]:
    if not tsv_path.exists() or tsv_path.stat().st_size == 0:
        raise SystemExit(f"Cached UniProt TSV is required and must be non-empty: {tsv_path}")
    if not metadata_path.exists() or metadata_path.stat().st_size == 0:
        raise SystemExit(
            f"Cached UniProt metadata is required and must be non-empty: {metadata_path}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != UNIPROT_METADATA_SCHEMA_VERSION:
        raise SystemExit(f"Unsupported UniProt cache metadata schema: {metadata_path}")
    source = metadata.get("source") if isinstance(metadata.get("source"), dict) else {}
    if source.get("release") != required_release:
        raise SystemExit(
            f"Cached UniProt release must be {required_release}; "
            f"observed {source.get('release') or '<blank>'}"
        )
    if source.get("license") != UNIPROT_LICENSE:
        raise SystemExit("Cached UniProt metadata must declare the pinned CC BY 4.0 license")
    query = metadata.get("query") if isinstance(metadata.get("query"), dict) else {}
    if query.get("requested_count") != len(requested_accessions):
        raise SystemExit("Cached UniProt query count does not match current evidence targets")
    if query.get("requested_accessions_sha256") != _hash_accessions(requested_accessions):
        raise SystemExit("Cached UniProt query accession hash does not match current evidence targets")
    artifact = metadata.get("artifact") if isinstance(metadata.get("artifact"), dict) else {}
    current_sha256 = _sha256_file(tsv_path)
    if artifact.get("sha256") != current_sha256:
        raise SystemExit("Cached UniProt TSV sha256 does not match its metadata")
    return metadata


def _uniprot_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != UNIPROT_TSV_COLUMNS:
            raise SystemExit(f"Unexpected UniProt TSV columns: {path}")
        rows: dict[str, dict[str, str]] = {}
        for line_number, row in enumerate(reader, start=2):
            accession = str(row.get("Entry") or "").strip()
            if not accession:
                raise SystemExit(f"Blank UniProt accession at line {line_number}: {path}")
            if accession in rows:
                raise SystemExit(f"Duplicate UniProt accession {accession!r}: {path}")
            rows[accession] = {key: str(value or "").strip() for key, value in row.items()}
    return rows


def _classify_sequences(
    requested: list[str],
    rows: dict[str, dict[str, str]],
    *,
    required_taxon_id: str,
    min_sequence_length: int,
    release: str,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    accepted: dict[str, str] = {}
    exclusions: list[dict[str, Any]] = []
    for accession in requested:
        row = rows.get(accession)
        reason = ""
        if row is None:
            row = {}
            reason = "not_found_in_uniprot_release"
        sequence = re.sub(r"\s+", "", row.get("Sequence", "")).upper()
        organism_id = row.get("Organism (ID)", "")
        length_text = row.get("Length", "")
        if not reason and (not sequence or not organism_id or not length_text):
            reason = "unresolved_or_obsolete_uniprot_record"
        if not reason and organism_id != required_taxon_id:
            reason = "taxonomy_mismatch"
        if not reason:
            try:
                declared_length = int(length_text)
            except ValueError as exc:
                raise SystemExit(f"Invalid UniProt sequence length for {accession}: {length_text}") from exc
            if declared_length != len(sequence):
                raise SystemExit(
                    f"UniProt sequence length mismatch for {accession}: "
                    f"declared={declared_length} observed={len(sequence)}"
                )
            if not SEQUENCE_RE.fullmatch(sequence):
                raise SystemExit(f"UniProt sequence contains unsupported residues: {accession}")
            if len(sequence) < min_sequence_length:
                reason = "sequence_below_minimum_length"
        if reason:
            exclusions.append(
                {
                    "uniprot": accession,
                    "reason": reason,
                    "uniprot_entry": row.get("Entry", ""),
                    "reviewed": row.get("Reviewed", ""),
                    "organism_id": organism_id,
                    "organism": row.get("Organism", ""),
                    "length": int(length_text) if length_text.isdigit() else None,
                    "sequence_version": row.get("Sequence version", ""),
                    "uniprot_release": release,
                }
            )
        else:
            accepted[accession] = sequence
    return accepted, exclusions


def _write_fasta(records: dict[str, str], path: Path) -> None:
    lines: list[str] = []
    for accession in sorted(records):
        lines.append(f">{accession}")
        sequence = records[accession]
        lines.extend(sequence[start : start + 60] for start in range(0, len(sequence), 60))
    _write_text_atomic("\n".join(lines) + ("\n" if lines else ""), path)


def _write_exclusions(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "uniprot",
        "reason",
        "uniprot_entry",
        "reviewed",
        "organism_id",
        "organism",
        "length",
        "sequence_version",
        "uniprot_release",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.min_sequence_length < 1:
        raise SystemExit("--min-sequence-length must be >= 1")
    if args.uniprot_batch_size < 1 or args.uniprot_batch_size > 500:
        raise SystemExit("--uniprot-batch-size must be in [1, 500]")
    if not str(args.required_taxon_id).strip():
        raise SystemExit("--required-taxon-id must be nonblank")
    if not args.required_uniprot_release.strip():
        raise SystemExit("--required-uniprot-release must be nonblank")
    if (
        args.chembl_evidence is None
        and args.bindingdb_evidence is None
        and args.gtopdb_evidence is None
    ):
        raise SystemExit("At least one evidence input is required")

    derived_outputs = [args.out_fasta, args.out_exclusions, args.out_manifest]
    _remove_derived_outputs(derived_outputs)
    source_meta: dict[str, Any] = {}
    targets_by_source: dict[str, set[str]] = {}
    accepted_rows_by_source: dict[str, int] = {}
    try:
        if args.chembl_evidence is not None:
            if args.chembl_manifest is None:
                raise SystemExit("--chembl-manifest is required with --chembl-evidence")
            source_meta["chembl"] = _validated_evidence_source(
                label="chembl",
                evidence_path=args.chembl_evidence,
                manifest_path=args.chembl_manifest,
                required_release=CHEMBL_REQUIRED_RELEASE,
            )
            targets, accepted_rows = _accepted_targets(
                args.chembl_evidence,
                source="chembl",
                exclude_bindingdb_sources=args.bindingdb_evidence is not None,
            )
            targets_by_source["chembl"] = targets
            accepted_rows_by_source["chembl"] = accepted_rows
        if args.bindingdb_evidence is not None:
            if args.bindingdb_manifest is None:
                raise SystemExit("--bindingdb-manifest is required with --bindingdb-evidence")
            source_meta["bindingdb"] = _validated_evidence_source(
                label="bindingdb",
                evidence_path=args.bindingdb_evidence,
                manifest_path=args.bindingdb_manifest,
                required_release=args.bindingdb_required_release,
            )
            targets, accepted_rows = _accepted_targets(
                args.bindingdb_evidence,
                source="bindingdb",
                exclude_bindingdb_sources=False,
            )
            targets_by_source["bindingdb"] = targets
            accepted_rows_by_source["bindingdb"] = accepted_rows
        if args.gtopdb_evidence is not None:
            if args.gtopdb_manifest is None:
                raise SystemExit("--gtopdb-manifest is required with --gtopdb-evidence")
            source_meta["gtopdb"] = _validated_evidence_source(
                label="gtopdb",
                evidence_path=args.gtopdb_evidence,
                manifest_path=args.gtopdb_manifest,
                required_release=args.gtopdb_required_release,
                required_schema_version="gtopdb_activity_evidence.v1",
            )
            targets, accepted_rows = _accepted_targets(
                args.gtopdb_evidence,
                source="gtopdb",
                exclude_bindingdb_sources=False,
            )
            targets_by_source["gtopdb"] = targets
            accepted_rows_by_source["gtopdb"] = accepted_rows

        evidence_targets = set().union(*targets_by_source.values())
        if not evidence_targets:
            raise SystemExit("No claim-grade evidence targets survived source filters")
        base_records = _fasta_records(args.base_fasta)
        base_accessions = {record.accession for record in base_records}
        if len(base_accessions) != len(base_records):
            raise SystemExit(f"Base FASTA contains duplicate accessions: {args.base_fasta}")
        requested_accessions = sorted(evidence_targets)

        cache_exists = args.uniprot_tsv.exists() and args.uniprot_metadata.exists()
        if args.offline and not cache_exists:
            raise SystemExit("--offline requires existing UniProt TSV and metadata cache")
        if args.refresh_uniprot and args.offline:
            raise SystemExit("--refresh-uniprot cannot be combined with --offline")
        if args.refresh_uniprot or not cache_exists:
            if args.offline:  # pragma: no cover - guarded above
                raise AssertionError("offline cache invariant")
            uniprot_meta = _fetch_uniprot_cache(
                requested_accessions,
                tsv_path=args.uniprot_tsv,
                metadata_path=args.uniprot_metadata,
                required_release=args.required_uniprot_release,
                batch_size=args.uniprot_batch_size,
            )
        else:
            uniprot_meta = _load_uniprot_cache(
                tsv_path=args.uniprot_tsv,
                metadata_path=args.uniprot_metadata,
                required_release=args.required_uniprot_release,
                requested_accessions=requested_accessions,
            )

        rows = _uniprot_rows(args.uniprot_tsv)
        accepted, exclusions = _classify_sequences(
            requested_accessions,
            rows,
            required_taxon_id=str(args.required_taxon_id),
            min_sequence_length=args.min_sequence_length,
            release=args.required_uniprot_release,
        )
        _write_fasta(accepted, args.out_fasta)
        _write_exclusions(exclusions, args.out_exclusions)

        target_summaries = {
            source: {
                "accepted_evidence_rows": accepted_rows_by_source[source],
                "target_count": len(targets),
                "target_accessions_sha256": _hash_accessions(targets),
                "missing_from_base_count": len(targets - base_accessions),
            }
            for source, targets in sorted(targets_by_source.items())
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "parameters": {
                "required_taxon_id": str(args.required_taxon_id),
                "min_sequence_length": args.min_sequence_length,
                "required_uniprot_release": args.required_uniprot_release,
                "chembl_quality_filter_contract": "build_activity_benchmark._chembl_reason_masks",
                "bindingdb_quality_filter_contract": "build_activity_benchmark._bindingdb_reason_masks",
                "gtopdb_quality_filter_contract": (
                    "build_activity_benchmark._bindingdb_reason_masks"
                    "(allowed_origins=GTOPDB_CURATED_ORIGINS)"
                ),
            },
            "inputs": {
                "comparison_structure_fasta": {
                    "path": str(args.base_fasta.resolve()),
                    "sha256": _sha256_file(args.base_fasta),
                    "accession_count": len(base_accessions),
                },
                "evidence_sources": source_meta,
                "uniprot_tsv": {
                    "path": str(args.uniprot_tsv.resolve()),
                    "sha256": _sha256_file(args.uniprot_tsv),
                },
                "uniprot_metadata": {
                    "path": str(args.uniprot_metadata.resolve()),
                    "sha256": _sha256_file(args.uniprot_metadata),
                    "source": uniprot_meta["source"],
                    "query": uniprot_meta["query"],
                },
            },
            "evidence_targets": {
                "total_count": len(evidence_targets),
                "accessions_sha256": _hash_accessions(evidence_targets),
                "present_in_structure_fasta_count": len(evidence_targets & base_accessions),
                "absent_from_structure_fasta_count": len(evidence_targets - base_accessions),
                "absent_from_structure_fasta_accessions": sorted(
                    evidence_targets - base_accessions
                ),
                "sequence_source": (
                    "UniProt canonical entry sequence for every evidence target; the cleaned "
                    "AlphaFold FASTA is comparison-only and is never used for sequence clusters"
                ),
                "by_source": target_summaries,
            },
            "artifacts": {
                "target_fasta": {
                    "path": str(args.out_fasta.resolve()),
                    "sha256": _sha256_file(args.out_fasta),
                    "accepted_count": len(accepted),
                    "accepted_accessions": sorted(accepted),
                },
                "exclusions_csv": {
                    "path": str(args.out_exclusions.resolve()),
                    "sha256": _sha256_file(args.out_exclusions),
                    "rows": len(exclusions),
                },
            },
            "exclusions": {
                "count": len(exclusions),
                "records": exclusions,
                "policy": (
                    "Only records absent, unresolved/obsolete, non-required taxonomy, or below "
                    "the declared sequence length are excluded; every other evidence target "
                    "requires a canonical-sequence cluster assignment."
                ),
            },
        }
        _write_json_atomic(manifest, args.out_manifest)
        return manifest
    except BaseException:
        _remove_derived_outputs(derived_outputs)
        raise


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-fasta", required=True, type=Path)
    parser.add_argument("--chembl-evidence", type=Path)
    parser.add_argument("--chembl-manifest", type=Path)
    parser.add_argument("--bindingdb-evidence", type=Path)
    parser.add_argument("--bindingdb-manifest", type=Path)
    parser.add_argument(
        "--bindingdb-required-release",
        default=BINDINGDB_DEFAULT_REQUIRED_RELEASE,
    )
    parser.add_argument("--gtopdb-evidence", type=Path)
    parser.add_argument("--gtopdb-manifest", type=Path)
    parser.add_argument(
        "--gtopdb-required-release",
        default=GTOPDB_DEFAULT_REQUIRED_RELEASE,
    )
    parser.add_argument("--required-uniprot-release", required=True)
    parser.add_argument("--required-taxon-id", default="9606")
    parser.add_argument("--min-sequence-length", type=int, default=30)
    parser.add_argument("--uniprot-batch-size", type=int, default=100)
    parser.add_argument("--uniprot-tsv", required=True, type=Path)
    parser.add_argument("--uniprot-metadata", required=True, type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--refresh-uniprot", action="store_true")
    parser.add_argument("--out-fasta", required=True, type=Path)
    parser.add_argument("--out-exclusions", required=True, type=Path)
    parser.add_argument("--out-manifest", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        manifest = build(args)
    except Exception as exc:
        print(f"[target-sequences][FATAL] {exc}", file=sys.stderr)
        return 1
    print(
        "[target-sequences] wrote "
        f"targets={manifest['evidence_targets']['total_count']} "
        f"clusterable={manifest['artifacts']['target_fasta']['accepted_count']} "
        f"excluded={manifest['exclusions']['count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
