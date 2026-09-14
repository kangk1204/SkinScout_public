#!/usr/bin/env python3
"""Build claim-grade human small-molecule activity evidence from GtoPdb."""

from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem
from rdkit.Chem import Descriptors

from mirror_gtopdb import (
    EXPECTED_HEADERS,
    PUBMED_SCHEMA_VERSION,
    SCHEMA_VERSION as SOURCE_MANIFEST_SCHEMA_VERSION,
    SOURCE_LICENSE,
    SOURCE_LICENSE_URL,
    SOURCE_NAME,
)


SCHEMA_VERSION = "gtopdb_activity_evidence.v1"
SOURCE_ORIGIN = "GtoPdb expert-curated literature"
ALLOWED_LIGAND_TYPES = {"Synthetic organic", "Metabolite", "Natural product"}
ENDPOINT_TO_PUNIT = {
    "IC50": "pIC50",
    "EC50": "pEC50",
    "Ki": "pKi",
    "Kd": "pKd",
}
UNIPROT_RE = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$"
)

OUTPUT_COLUMNS = [
    "evidence_id",
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
    "gtopdb_target_id",
    "gtopdb_ligand_id",
    "supporting_pmids",
    "supporting_pmids_sha256",
    "ligand_type",
    "affinity_pvalue",
    "affinity_consistency_abs_error",
]

OUTPUT_SCHEMA = pa.schema(
    [
        ("evidence_id", pa.string()),
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
        ("gtopdb_target_id", pa.string()),
        ("gtopdb_ligand_id", pa.string()),
        ("supporting_pmids", pa.string()),
        ("supporting_pmids_sha256", pa.string()),
        ("ligand_type", pa.string()),
        ("affinity_pvalue", pa.float64()),
        ("affinity_consistency_abs_error", pa.float64()),
    ]
)


def _clean(value: object) -> str:
    text = str(value or "").strip()
    return "" if text.upper() in {"", "NA", "N/A", "NULL", "NAN", "NONE"} else text


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_values(values: Iterable[str]) -> str:
    payload = "\n".join(sorted(set(values))) + "\n"
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _hash_payload(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _tmp_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def _remove_outputs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.unlink(missing_ok=True)
        _tmp_path(path).unlink(missing_ok=True)


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    tmp.unlink(missing_ok=True)
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_parquet_atomic(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    tmp.unlink(missing_ok=True)
    columns = {column: [row.get(column) for row in rows] for column in OUTPUT_COLUMNS}
    pq.write_table(pa.Table.from_pydict(columns, schema=OUTPUT_SCHEMA), tmp)
    tmp.replace(path)


def _read_release_csv(path: Path, expected_header: tuple[str, ...]) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"GtoPdb CSV is missing or empty: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        next(handle, None)
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != expected_header:
            raise SystemExit(f"GtoPdb CSV schema mismatch: {path}")
        return [
            {column: _clean(value) for column, value in row.items()}
            for row in reader
        ]


def _validate_source_artifact(
    manifest: dict[str, Any],
    manifest_path: Path,
    name: str,
    path: Path,
    *,
    rows: int | None = None,
) -> dict[str, Any]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not isinstance(artifacts.get(name), dict):
        raise SystemExit(f"GtoPdb source manifest lacks artifact metadata for {name}")
    artifact = artifacts[name]
    declared_path = artifact.get("path")
    if not isinstance(declared_path, str) or not declared_path.strip():
        raise SystemExit(f"GtoPdb source manifest path is missing for {name}")
    relative_path = Path(declared_path)
    if relative_path.is_absolute():
        raise SystemExit(f"GtoPdb source manifest path must be relative for {name}")
    if (manifest_path.parent / relative_path).resolve() != path.resolve():
        raise SystemExit(f"GtoPdb source manifest path mismatch for {name}")
    current_sha256 = _sha256_file(path)
    if artifact.get("sha256") != current_sha256:
        raise SystemExit(f"GtoPdb source manifest sha256 mismatch for {name}")
    if artifact.get("bytes") != path.stat().st_size:
        raise SystemExit(f"GtoPdb source manifest byte count mismatch for {name}")
    if rows is not None and artifact.get("rows") != rows:
        raise SystemExit(f"GtoPdb source manifest row count mismatch for {name}")
    return artifact


def _load_source_manifest(
    path: Path,
    *,
    release: str,
    interactions_path: Path,
    interactions_rows: int,
    ligands_path: Path,
    ligands_rows: int,
    mapping_path: Path,
    mapping_rows: int,
    pubmed_path: Path,
) -> tuple[dict[str, Any], str]:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"GtoPdb source manifest is missing or empty: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid GtoPdb source manifest JSON: {path}") from exc
    if manifest.get("schema_version") != SOURCE_MANIFEST_SCHEMA_VERSION:
        raise SystemExit(f"Unsupported GtoPdb source manifest schema: {path}")
    source = manifest.get("source") if isinstance(manifest.get("source"), dict) else {}
    if source.get("name") != SOURCE_NAME or source.get("release") != release:
        raise SystemExit("GtoPdb source name/release does not match the required contract")
    if source.get("license") != SOURCE_LICENSE or source.get("license_url") != SOURCE_LICENSE_URL:
        raise SystemExit("GtoPdb source manifest does not declare the pinned license contract")
    try:
        release_date = date.fromisoformat(str(source.get("release_date") or "")).isoformat()
    except ValueError as exc:
        raise SystemExit("GtoPdb source manifest release_date must be YYYY-MM-DD") from exc
    _validate_source_artifact(
        manifest,
        path,
        "interactions.csv",
        interactions_path,
        rows=interactions_rows,
    )
    _validate_source_artifact(
        manifest,
        path,
        "ligands.csv",
        ligands_path,
        rows=ligands_rows,
    )
    _validate_source_artifact(
        manifest,
        path,
        "GtP_to_UniProt_mapping.csv",
        mapping_path,
        rows=mapping_rows,
    )
    _validate_source_artifact(
        manifest,
        path,
        "pubmed_esummary.json",
        pubmed_path,
    )
    return manifest, release_date


def _parse_date_text(value: object) -> str:
    text = _clean(value)
    if not text:
        return ""
    exact_prefix = re.match(r"^([12][0-9]{3})/(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])", text)
    if exact_prefix:
        try:
            return date(
                int(exact_prefix.group(1)),
                int(exact_prefix.group(2)),
                int(exact_prefix.group(3)),
            ).isoformat()
        except ValueError:
            return ""
    for fmt in ("%Y %b %d", "%Y %B %d"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            pass
    for fmt in ("%Y %b", "%Y %B"):
        try:
            parsed = datetime.strptime(text, fmt)
            last_day = calendar.monthrange(parsed.year, parsed.month)[1]
            return date(parsed.year, parsed.month, last_day).isoformat()
        except ValueError:
            pass
    if re.fullmatch(r"[12][0-9]{3}", text):
        return f"{text}-12-31"
    match = re.match(r"^([12][0-9]{3})", text)
    return f"{match.group(1)}-12-31" if match else ""


def _record_doi(record: dict[str, Any]) -> str:
    article_ids = record.get("articleids")
    if not isinstance(article_ids, list):
        return ""
    for item in article_ids:
        if not isinstance(item, dict):
            continue
        if str(item.get("idtype") or "").casefold() == "doi":
            return _clean(item.get("value")).lower()
    return ""


def _load_pubmed(path: Path, expected_pmids: set[str]) -> dict[str, dict[str, str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid PubMed metadata cache: {path}") from exc
    if payload.get("schema_version") != PUBMED_SCHEMA_VERSION:
        raise SystemExit(f"Unsupported PubMed metadata schema: {path}")
    query = payload.get("query") if isinstance(payload.get("query"), dict) else {}
    if query.get("requested_count") != len(expected_pmids):
        raise SystemExit("PubMed metadata requested_count does not match interactions.csv")
    if query.get("requested_pmids_sha256") != _hash_values(expected_pmids):
        raise SystemExit("PubMed metadata identifier hash does not match interactions.csv")
    records = payload.get("records") if isinstance(payload.get("records"), dict) else {}
    missing = payload.get("missing_pmids")
    if not isinstance(missing, list):
        raise SystemExit("PubMed metadata missing_pmids must be a list")
    if set(records) | {str(value) for value in missing} != expected_pmids:
        raise SystemExit("PubMed metadata does not cover the interactions.csv PMID universe")
    normalized: dict[str, dict[str, str]] = {}
    for pmid, raw in records.items():
        if not isinstance(raw, dict) or str(raw.get("uid") or "") != pmid:
            raise SystemExit(f"PubMed metadata record identity mismatch for PMID {pmid}")
        publication_date = ""
        for field in ("sortpubdate", "epubdate", "pubdate"):
            publication_date = _parse_date_text(raw.get(field))
            if publication_date:
                break
        normalized[pmid] = {
            "publication_date": publication_date,
            "doi": _record_doi(raw),
        }
    return normalized


def _interaction_pmids(rows: list[dict[str, str]]) -> set[str]:
    pmids: set[str] = set()
    for row in rows:
        for token in row["PubMed ID"].split("|"):
            pmid = token.strip()
            if pmid:
                pmids.add(pmid)
    return pmids


def _human_target_mapping(rows: list[dict[str, str]]) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = {}
    for line_number, row in enumerate(rows, start=3):
        if row["Species"] != "Human":
            continue
        target_id = row["GtoPdb IUPHAR ID"]
        uniprot = row["UniProtKB ID"].upper()
        if not target_id or not uniprot:
            raise SystemExit(f"Invalid human target mapping at CSV line {line_number}")
        if "-" in uniprot:
            continue
        if not UNIPROT_RE.fullmatch(uniprot):
            raise SystemExit(f"Invalid human target mapping at CSV line {line_number}")
        mapping.setdefault(target_id, set()).add(uniprot)
    if not mapping:
        raise SystemExit("GtoPdb target mapping contains no human targets")
    return mapping


def _ligand_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    by_id: dict[str, dict[str, str]] = {}
    for line_number, row in enumerate(rows, start=3):
        ligand_id = row["Ligand ID"]
        if not ligand_id:
            raise SystemExit(f"Blank GtoPdb ligand ID at CSV line {line_number}")
        if ligand_id in by_id:
            raise SystemExit(f"Duplicate GtoPdb ligand ID {ligand_id}")
        by_id[ligand_id] = row
    return by_id


def _ligand_structure(
    ligand: dict[str, str],
    *,
    min_molecular_weight: float,
    max_molecular_weight: float,
    max_heavy_atoms: int,
) -> tuple[str, str, float, int] | None:
    if ligand["Type"] not in ALLOWED_LIGAND_TYPES:
        return None
    smiles = ligand["SMILES"]
    source_inchikey = ligand["InChIKey"].upper()
    if not smiles or not source_inchikey:
        return None
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    canonical_smiles = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    computed_inchikey = Chem.MolToInchiKey(molecule).upper()
    if not computed_inchikey or computed_inchikey != source_inchikey:
        return None
    molecular_weight = float(Descriptors.MolWt(molecule))
    heavy_atoms = int(molecule.GetNumHeavyAtoms())
    if (
        not math.isfinite(molecular_weight)
        or molecular_weight < min_molecular_weight
        or molecular_weight > max_molecular_weight
        or heavy_atoms < 3
        or heavy_atoms > max_heavy_atoms
    ):
        return None
    return canonical_smiles, computed_inchikey, molecular_weight, heavy_atoms


def _float(value: object) -> float | None:
    try:
        parsed = float(_clean(value))
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _supporting_publication(
    raw_pmids: str,
    pubmed: dict[str, dict[str, str]],
) -> tuple[str, str, str, str] | None:
    pmids = sorted({token.strip() for token in raw_pmids.split("|") if token.strip()}, key=int)
    if not pmids:
        return None
    if any(pmid not in pubmed or not pubmed[pmid]["publication_date"] for pmid in pmids):
        return None
    anchor = max(pmids, key=lambda pmid: (pubmed[pmid]["publication_date"], int(pmid)))
    joined = "|".join(pmids)
    return (
        anchor,
        pubmed[anchor]["doi"],
        pubmed[anchor]["publication_date"],
        joined,
    )


def _skip(counts: Counter[str], reason: str) -> None:
    counts[f"filtered_{reason}"] += 1


def build(args: argparse.Namespace) -> dict[str, Any]:
    try:
        cutoff = date.fromisoformat(args.cutoff_date)
    except ValueError as exc:
        raise SystemExit("--cutoff-date must be YYYY-MM-DD") from exc
    if not re.fullmatch(r"[0-9]{4}\.[1-4]", args.required_release):
        raise SystemExit("--required-release must use GtoPdb YYYY.N format")
    if args.affinity_consistency_tolerance < 0 or args.affinity_consistency_tolerance > 0.25:
        raise SystemExit("--affinity-consistency-tolerance must be in [0, 0.25]")
    if not 0 < args.min_molecular_weight < args.max_molecular_weight:
        raise SystemExit("molecular-weight bounds must satisfy 0 < min < max")
    if args.max_heavy_atoms < 3:
        raise SystemExit("--max-heavy-atoms must be >= 3")

    outputs = {
        "activity_evidence": args.out_dir / "activity_evidence.parquet",
        "pre_cutoff": args.out_dir / "pre_cutoff.parquet",
        "post_cutoff": args.out_dir / "post_cutoff.parquet",
        "manifest": args.out_dir / "manifest.json",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _remove_outputs(outputs.values())
    try:
        interactions = _read_release_csv(
            args.interactions_csv,
            EXPECTED_HEADERS["interactions.csv"],
        )
        ligands_raw = _read_release_csv(args.ligands_csv, EXPECTED_HEADERS["ligands.csv"])
        mapping_raw = _read_release_csv(
            args.target_mapping_csv,
            EXPECTED_HEADERS["GtP_to_UniProt_mapping.csv"],
        )
        source_manifest, release_date = _load_source_manifest(
            args.source_manifest,
            release=args.required_release,
            interactions_path=args.interactions_csv,
            interactions_rows=len(interactions),
            ligands_path=args.ligands_csv,
            ligands_rows=len(ligands_raw),
            mapping_path=args.target_mapping_csv,
            mapping_rows=len(mapping_raw),
            pubmed_path=args.pubmed_json,
        )
        expected_pmids = _interaction_pmids(interactions)
        pubmed = _load_pubmed(args.pubmed_json, expected_pmids)
        target_mapping = _human_target_mapping(mapping_raw)
        ligands = _ligand_rows(ligands_raw)

        rows: list[dict[str, object]] = []
        seen_evidence: set[str] = set()
        counts: Counter[str] = Counter(input_rows_seen=len(interactions))
        structure_cache: dict[str, tuple[str, str, float, int] | None] = {}
        molecular_weight_min = math.inf
        molecular_weight_max = -math.inf
        heavy_atom_min = sys.maxsize
        heavy_atom_max = 0
        for input_row_number, row in enumerate(interactions, start=1):
            if row["Target Species"] != "Human":
                _skip(counts, "non_human_target")
                continue
            target_id = row["Target ID"]
            uniprot = row["Target UniProt ID"].upper()
            if (
                not target_id
                or not uniprot
                or "|" in uniprot
                or not UNIPROT_RE.fullmatch(uniprot)
                or row["Target Subunit IDs"]
                or row["Target Ligand ID"]
            ):
                _skip(counts, "non_single_protein_target")
                continue
            mapped = target_mapping.get(target_id, set())
            if mapped != {uniprot}:
                _skip(counts, "target_mapping_mismatch")
                continue
            ligand_id = row["Ligand ID"]
            ligand = ligands.get(ligand_id)
            if ligand is None:
                _skip(counts, "missing_ligand")
                continue
            if row["Ligand Type"] != ligand["Type"]:
                _skip(counts, "ligand_type_mismatch")
                continue
            if ligand_id not in structure_cache:
                structure_cache[ligand_id] = _ligand_structure(
                    ligand,
                    min_molecular_weight=args.min_molecular_weight,
                    max_molecular_weight=args.max_molecular_weight,
                    max_heavy_atoms=args.max_heavy_atoms,
                )
            structure = structure_cache[ligand_id]
            if structure is None:
                _skip(counts, "non_small_molecule_or_invalid_structure")
                continue
            canonical_smiles, inchikey, molecular_weight, heavy_atoms = structure
            endpoint = row["Original Affinity Units"]
            expected_punit = ENDPOINT_TO_PUNIT.get(endpoint)
            if expected_punit is None or row["Affinity Units"] != expected_punit:
                _skip(counts, "unsupported_or_inconsistent_endpoint")
                continue
            if row["Original Affinity Relation"] != "=":
                _skip(counts, "non_exact_relation")
                continue
            value_nm = _float(row["Original Affinity Median nm"])
            pvalue = _float(row["Affinity Median"])
            if value_nm is None or value_nm <= 0 or pvalue is None:
                _skip(counts, "missing_or_invalid_affinity")
                continue
            consistency_error = abs(pvalue - (9.0 - math.log10(value_nm)))
            if consistency_error > args.affinity_consistency_tolerance:
                _skip(counts, "affinity_unit_consistency")
                continue
            publication = _supporting_publication(row["PubMed ID"], pubmed)
            if publication is None:
                _skip(counts, "unresolved_publication_date")
                continue
            anchor_pmid, anchor_doi, publication_date, supporting_pmids = publication
            supporting_hash = _hash_values(supporting_pmids.split("|"))
            evidence_payload = {
                "release": args.required_release,
                "target_id": target_id,
                "uniprot": uniprot,
                "ligand_id": ligand_id,
                "inchikey": inchikey,
                "endpoint": endpoint,
                "value_nm": value_nm,
                "pvalue": pvalue,
                "supporting_pmids": supporting_pmids,
                "action": row["Action"],
                "assay": row["Assay Description"],
            }
            evidence_id = _hash_payload(evidence_payload)
            if evidence_id in seen_evidence:
                _skip(counts, "exact_duplicate")
                continue
            seen_evidence.add(evidence_id)
            split = "pre_cutoff" if date.fromisoformat(publication_date) <= cutoff else "post_cutoff"
            source_article_id = f"GtoPdb:{target_id}:{ligand_id}:{supporting_hash[:16]}"
            rows.append(
                {
                    "evidence_id": evidence_id,
                    "input_row_number": input_row_number,
                    "source_db": SOURCE_NAME,
                    "source_origin": SOURCE_ORIGIN,
                    "source_release": args.required_release,
                    "source_license": SOURCE_LICENSE,
                    "source_license_url": SOURCE_LICENSE_URL,
                    "chembl_derived_license_flag": False,
                    "ligand_smiles": canonical_smiles,
                    "ligand_inchikey": inchikey,
                    "ligand_id": f"GtoPdb:{ligand_id}",
                    "uniprot": uniprot,
                    "target_chain_count": 1,
                    "single_chain_target": True,
                    "organism": "Homo sapiens",
                    "affinity_type": endpoint,
                    "relation": "=",
                    "censor": False,
                    "affinity_value": value_nm,
                    "affinity_unit": "nM",
                    "source_pmid": anchor_pmid,
                    "source_doi": anchor_doi,
                    "source_patent": "",
                    "source_article_id": source_article_id,
                    "publication_date": publication_date,
                    "curation_date": release_date,
                    "evidence_date": publication_date,
                    "evidence_date_source": "publication",
                    "temporal_split": split,
                    "gtopdb_target_id": target_id,
                    "gtopdb_ligand_id": ligand_id,
                    "supporting_pmids": supporting_pmids,
                    "supporting_pmids_sha256": supporting_hash,
                    "ligand_type": ligand["Type"],
                    "affinity_pvalue": pvalue,
                    "affinity_consistency_abs_error": consistency_error,
                }
            )
            counts["accepted"] += 1
            counts[f"accepted_{split}"] += 1
            counts[f"accepted_endpoint_{endpoint}"] += 1
            molecular_weight_min = min(molecular_weight_min, molecular_weight)
            molecular_weight_max = max(molecular_weight_max, molecular_weight)
            heavy_atom_min = min(heavy_atom_min, heavy_atoms)
            heavy_atom_max = max(heavy_atom_max, heavy_atoms)

        if not rows:
            raise SystemExit("No claim-grade GtoPdb activity rows survived strict filters")
        rows.sort(
            key=lambda item: (
                str(item["publication_date"]),
                str(item["source_pmid"]),
                str(item["uniprot"]),
                str(item["ligand_id"]),
                str(item["affinity_type"]),
                str(item["evidence_id"]),
            )
        )
        pre_rows = [row for row in rows if row["temporal_split"] == "pre_cutoff"]
        post_rows = [row for row in rows if row["temporal_split"] == "post_cutoff"]
        _write_parquet_atomic(rows, outputs["activity_evidence"])
        _write_parquet_atomic(pre_rows, outputs["pre_cutoff"])
        _write_parquet_atomic(post_rows, outputs["post_cutoff"])

        artifacts = {
            name: {
                "path": path.name,
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
                "rows": len(rows if name == "activity_evidence" else pre_rows if name == "pre_cutoff" else post_rows),
            }
            for name, path in outputs.items()
            if name != "manifest"
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "source": {
                "name": SOURCE_NAME,
                "release": args.required_release,
                "release_date": release_date,
                "license": SOURCE_LICENSE,
                "license_url": SOURCE_LICENSE_URL,
            },
            "inputs": {
                "source_manifest": {
                    "path": os.path.relpath(
                        args.source_manifest.resolve(), start=args.out_dir.resolve()
                    ),
                    "sha256": _sha256_file(args.source_manifest),
                    "schema_version": source_manifest["schema_version"],
                },
                "interactions_csv": {
                    "path": os.path.relpath(
                        args.interactions_csv.resolve(), start=args.out_dir.resolve()
                    ),
                    "sha256": _sha256_file(args.interactions_csv),
                    "rows": len(interactions),
                },
                "ligands_csv": {
                    "path": os.path.relpath(
                        args.ligands_csv.resolve(), start=args.out_dir.resolve()
                    ),
                    "sha256": _sha256_file(args.ligands_csv),
                    "rows": len(ligands_raw),
                },
                "target_mapping_csv": {
                    "path": os.path.relpath(
                        args.target_mapping_csv.resolve(), start=args.out_dir.resolve()
                    ),
                    "sha256": _sha256_file(args.target_mapping_csv),
                    "rows": len(mapping_raw),
                },
                "pubmed_json": {
                    "path": os.path.relpath(
                        args.pubmed_json.resolve(), start=args.out_dir.resolve()
                    ),
                    "sha256": _sha256_file(args.pubmed_json),
                    "requested_pmids": len(expected_pmids),
                    "resolved_dated_pmids": sum(
                        bool(record["publication_date"]) for record in pubmed.values()
                    ),
                },
            },
            "schema": OUTPUT_COLUMNS,
            "policy": {
                "target": (
                    "Human, one canonical UniProt accession, no target subunits or target-ligand "
                    "complex, and exact agreement with the GtoPdb-to-UniProt mapping"
                ),
                "ligand_types": sorted(ALLOWED_LIGAND_TYPES),
                "small_molecule": {
                    "rdkit_parse_required": True,
                    "source_and_rdkit_inchikey_must_match": True,
                    "min_molecular_weight": args.min_molecular_weight,
                    "max_molecular_weight": args.max_molecular_weight,
                    "min_heavy_atoms": 3,
                    "max_heavy_atoms": args.max_heavy_atoms,
                    "observed_molecular_weight_range": [
                        molecular_weight_min,
                        molecular_weight_max,
                    ],
                    "observed_heavy_atom_range": [heavy_atom_min, heavy_atom_max],
                },
                "activity": {
                    "endpoints": sorted(ENDPOINT_TO_PUNIT),
                    "exact_relation_required": "=",
                    "original_value_unit": "nM",
                    "pvalue_pairing": ENDPOINT_TO_PUNIT,
                    "pvalue_consistency": (
                        "abs(source_pvalue - (9-log10(source_nM))) <= tolerance"
                    ),
                    "tolerance": args.affinity_consistency_tolerance,
                },
                "publication": (
                    "all pipe-delimited PMIDs on an aggregated GtoPdb row must resolve to a "
                    "publication date; the latest supporting publication is the conservative "
                    "temporal anchor and document key"
                ),
                "negative_policy": "no absent or qualitative rows are converted to negatives",
                "duplicate_policy": "exact normalized evidence payloads are retained once",
                "temporal_boundary": f"publication_date <= {cutoff.isoformat()} is pre_cutoff",
            },
            "counts": dict(sorted(counts.items())),
            "row_counts": {
                "activity_evidence": len(rows),
                "pre_cutoff": len(pre_rows),
                "post_cutoff": len(post_rows),
            },
            "artifacts": artifacts,
            "output_sha256": {
                path.name: metadata["sha256"] for path, metadata in (
                    (outputs["activity_evidence"], artifacts["activity_evidence"]),
                    (outputs["pre_cutoff"], artifacts["pre_cutoff"]),
                    (outputs["post_cutoff"], artifacts["post_cutoff"]),
                )
            },
            "manifest_sha256_policy": (
                "manifest.json is excluded from artifact hashes because stable self-hashing is impossible"
            ),
        }
        _write_json_atomic(manifest, outputs["manifest"])
        return manifest
    except BaseException:
        _remove_outputs(outputs.values())
        raise


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interactions-csv", required=True, type=Path)
    parser.add_argument("--ligands-csv", required=True, type=Path)
    parser.add_argument("--target-mapping-csv", required=True, type=Path)
    parser.add_argument("--pubmed-json", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--required-release", required=True)
    parser.add_argument("--cutoff-date", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--affinity-consistency-tolerance", type=float, default=0.05)
    parser.add_argument("--min-molecular-weight", type=float, default=50.0)
    parser.add_argument("--max-molecular-weight", type=float, default=1000.0)
    parser.add_argument("--max-heavy-atoms", type=int, default=100)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        manifest = build(args)
    except Exception as exc:
        print(f"[stage0.gtopdb.evidence][FATAL] {exc}", file=sys.stderr)
        return 1
    rows = manifest["row_counts"]
    print(
        "[stage0.gtopdb.evidence] wrote "
        f"all={rows['activity_evidence']} pre={rows['pre_cutoff']} "
        f"post={rows['post_cutoff']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
