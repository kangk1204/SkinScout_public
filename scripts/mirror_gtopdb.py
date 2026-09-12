#!/usr/bin/env python3
"""Mirror a pinned GtoPdb release and its PubMed date metadata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests


SCHEMA_VERSION = "skinscout.gtopdb-source.v1"
PUBMED_SCHEMA_VERSION = "skinscout.pubmed-esummary-cache.v1"
SOURCE_NAME = "GtoPdb"
SOURCE_OFFICIAL_URL = "https://www.guidetopharmacology.org/"
SOURCE_DOWNLOADS_URL = "https://www.guidetopharmacology.org/download.jsp"
SOURCE_LICENSE = "ODbL 1.0; contents CC BY-SA 4.0"
SOURCE_LICENSE_URL = "https://www.guidetopharmacology.org/about.jsp#license"
PUBMED_ENDPOINT = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

ARTIFACT_URLS = {
    "interactions.csv": "https://www.guidetopharmacology.org/DATA/interactions.csv",
    "ligands.csv": "https://www.guidetopharmacology.org/DATA/ligands.csv",
    "GtP_to_UniProt_mapping.csv": (
        "https://www.guidetopharmacology.org/DATA/GtP_to_UniProt_mapping.csv"
    ),
    "file_descriptions.txt": (
        "https://www.guidetopharmacology.org/DATA/file_descriptions.txt"
    ),
}

EXPECTED_HEADERS = {
    "interactions.csv": (
        "Target",
        "Target ID",
        "Target Subunit IDs",
        "Target Gene Symbol",
        "Target UniProt ID",
        "Target Ensembl Gene ID",
        "Target Ligand",
        "Target Ligand ID",
        "Target Ligand Subunit IDs",
        "Target Ligand Gene Symbol",
        "Target Ligand UniProt ID",
        "Target Ligand Ensembl Gene ID",
        "Target Ligand PubChem SID",
        "Target Species",
        "Ligand ID",
        "Ligand",
        "Ligand Type",
        "Ligand Subunit IDs",
        "Ligand Gene Symbol",
        "Ligand Species",
        "Ligand PubChem SID",
        "Approved",
        "Type",
        "Action",
        "Action comment",
        "Selectivity",
        "Endogenous",
        "Primary Target",
        "concentration Range",
        "Affinity Units",
        "Affinity High",
        "Affinity Median",
        "Affinity Low",
        "Original Affinity Units",
        "Original Affinity Low nm",
        "Original Affinity Median nm",
        "Original Affinity High nm",
        "Original Affinity Relation",
        "Assay Description",
        "Receptor Site",
        "Ligand Context",
        "PubMed ID",
        "Webpage URLs",
        "Patent Numbers",
    ),
    "ligands.csv": (
        "Ligand ID",
        "Name",
        "Species",
        "Type",
        "Approved",
        "Withdrawn",
        "Labelled",
        "Radioactive",
        "PubChem SID",
        "PubChem CID",
        "UniProt ID",
        "Ensembl ID",
        "ChEMBL ID",
        "Ligand Subunit IDs",
        "Ligand Subunit Name",
        "Ligand Subunit UniProt IDs",
        "Ligand Subunit Ensembl IDs",
        "IUPAC name",
        "INN",
        "Synonyms",
        "SMILES",
        "InChIKey",
        "InChI",
        "GtoImmuPdb",
        "GtoMPdb",
        "Antibacterial",
    ),
    "GtP_to_UniProt_mapping.csv": (
        "UniProtKB ID",
        "Species",
        "GtoPdb IUPHAR Name",
        "GtoPdb IUPHAR ID",
        "GtP URL",
    ),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_values(values: Iterable[str]) -> str:
    payload = "\n".join(sorted(set(values))) + "\n"
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _tmp_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    tmp.unlink(missing_ok=True)
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _release_banner(path: Path) -> str:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        first = handle.readline()
    parsed = next(csv.reader([first]), [])
    return str(parsed[0]).strip() if parsed else ""


def _expected_banner(release: str, release_date: str) -> str:
    return f"# GtoPdb Version: {release} - published: {release_date}"


def _csv_header_and_rows(path: Path) -> tuple[tuple[str, ...], int]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        next(handle, None)
        reader = csv.reader(handle)
        try:
            header = tuple(next(reader))
        except StopIteration as exc:
            raise SystemExit(f"GtoPdb CSV has no header: {path}") from exc
        rows = sum(1 for _ in reader)
    return header, rows


def _validate_download(
    path: Path,
    *,
    expected_sha256: str,
    release: str,
    release_date: str,
) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"GtoPdb artifact is missing or empty: {path}")
    observed_sha256 = _sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise SystemExit(
            f"GtoPdb artifact sha256 mismatch for {path.name}: "
            f"{observed_sha256} != {expected_sha256}"
        )
    result: dict[str, Any] = {
        "path": path.name,
        "url": ARTIFACT_URLS[path.name],
        "sha256": observed_sha256,
        "bytes": path.stat().st_size,
    }
    if path.suffix.lower() == ".csv":
        banner = _release_banner(path)
        expected_banner = _expected_banner(release, release_date)
        if banner != expected_banner:
            raise SystemExit(
                f"GtoPdb release banner mismatch for {path.name}: "
                f"{banner!r} != {expected_banner!r}"
            )
        header, rows = _csv_header_and_rows(path)
        expected_header = EXPECTED_HEADERS[path.name]
        if header != expected_header:
            raise SystemExit(f"GtoPdb CSV schema mismatch for {path.name}")
        result.update({"release_banner": banner, "rows": rows, "columns": list(header)})
    return result


def _download(session: requests.Session, url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    tmp.unlink(missing_ok=True)
    try:
        with session.get(url, stream=True, timeout=(30, 300)) as response:
            response.raise_for_status()
            with tmp.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        if tmp.stat().st_size == 0:
            raise SystemExit(f"Downloaded empty GtoPdb artifact: {url}")
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _interaction_pmids(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        next(handle, None)
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != EXPECTED_HEADERS["interactions.csv"]:
            raise SystemExit("Cannot collect PMIDs from an unvalidated interactions.csv schema")
        pmids: set[str] = set()
        for line_number, row in enumerate(reader, start=3):
            for token in str(row.get("PubMed ID") or "").split("|"):
                pmid = token.strip()
                if not pmid:
                    continue
                if not re.fullmatch(r"[1-9][0-9]{0,8}", pmid):
                    raise SystemExit(
                        f"Invalid GtoPdb PubMed ID {pmid!r} at CSV line {line_number}"
                    )
                pmids.add(pmid)
    if not pmids:
        raise SystemExit("GtoPdb interactions contain no PubMed identifiers")
    return sorted(pmids, key=int)


def _fetch_pubmed_batch(
    session: requests.Session,
    batch: list[str],
    *,
    email: str,
    api_key: str,
    attempts: int,
) -> dict[str, Any]:
    params = {
        "db": "pubmed",
        "id": ",".join(batch),
        "retmode": "json",
        "tool": "SkinScout",
    }
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = session.get(PUBMED_ENDPOINT, params=params, timeout=(30, 180))
            if response.status_code == 429 or response.status_code >= 500:
                response.raise_for_status()
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or not isinstance(payload.get("result"), dict):
                raise ValueError("NCBI ESummary response is not a result object")
            return payload
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(2 ** (attempt - 1))
    raise SystemExit(f"NCBI PubMed ESummary failed after {attempts} attempts: {last_error}")


def _build_pubmed_cache(
    session: requests.Session,
    pmids: list[str],
    *,
    out_path: Path,
    batch_size: int,
    email: str,
    api_key: str,
    attempts: int,
) -> dict[str, Any]:
    records: dict[str, dict[str, Any]] = {}
    batches: list[dict[str, Any]] = []
    delay_seconds = 0.11 if api_key else 0.36
    for start in range(0, len(pmids), batch_size):
        batch = pmids[start : start + batch_size]
        payload = _fetch_pubmed_batch(
            session,
            batch,
            email=email,
            api_key=api_key,
            attempts=attempts,
        )
        result = payload["result"]
        returned = result.get("uids")
        if not isinstance(returned, list):
            raise SystemExit("NCBI PubMed ESummary response lacks a UID list")
        returned_ids = [str(value) for value in returned]
        if len(returned_ids) != len(set(returned_ids)):
            raise SystemExit("NCBI PubMed ESummary returned duplicate UIDs")
        unexpected = sorted(set(returned_ids) - set(batch), key=int)
        if unexpected:
            raise SystemExit(f"NCBI PubMed ESummary returned unexpected PMID {unexpected[0]}")
        for pmid in returned_ids:
            record = result.get(pmid)
            if not isinstance(record, dict) or str(record.get("uid") or "") != pmid:
                continue
            records[pmid] = record
        batches.append(
            {
                "requested_count": len(batch),
                "requested_pmids_sha256": _hash_values(batch),
                "returned_count": len(returned_ids),
            }
        )
        if start + batch_size < len(pmids):
            time.sleep(delay_seconds)

    missing = sorted(set(pmids) - set(records), key=int)
    cache = {
        "schema_version": PUBMED_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": {
            "name": "NCBI PubMed",
            "endpoint": PUBMED_ENDPOINT,
            "api": "E-utilities ESummary",
        },
        "query": {
            "requested_count": len(pmids),
            "requested_pmids_sha256": _hash_values(pmids),
            "batch_size": batch_size,
            "batches": batches,
        },
        "records": {pmid: records[pmid] for pmid in sorted(records, key=int)},
        "missing_pmids": missing,
    }
    _write_json_atomic(cache, out_path)
    return cache


def _load_pubmed_cache(path: Path, pmids: list[str]) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"PubMed cache is missing or empty: {path}")
    try:
        cache = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid PubMed cache JSON: {path}") from exc
    if cache.get("schema_version") != PUBMED_SCHEMA_VERSION:
        raise SystemExit(f"Unsupported PubMed cache schema: {path}")
    query = cache.get("query") if isinstance(cache.get("query"), dict) else {}
    if query.get("requested_count") != len(pmids):
        raise SystemExit("PubMed cache query count does not match current GtoPdb PMIDs")
    if query.get("requested_pmids_sha256") != _hash_values(pmids):
        raise SystemExit("PubMed cache query hash does not match current GtoPdb PMIDs")
    records = cache.get("records") if isinstance(cache.get("records"), dict) else {}
    missing = cache.get("missing_pmids")
    if not isinstance(missing, list):
        raise SystemExit("PubMed cache missing_pmids must be a list")
    declared = set(records) | {str(value) for value in missing}
    if declared != set(pmids):
        raise SystemExit("PubMed cache records and missing IDs do not cover the requested PMID set")
    for pmid, record in records.items():
        if not isinstance(record, dict) or str(record.get("uid") or "") != pmid:
            raise SystemExit(f"PubMed cache record identity mismatch for PMID {pmid}")
    return cache


def mirror(args: argparse.Namespace) -> dict[str, Any]:
    try:
        release_date = date.fromisoformat(args.release_date).isoformat()
    except ValueError as exc:
        raise SystemExit("--release-date must be YYYY-MM-DD") from exc
    if not re.fullmatch(r"[0-9]{4}\.[1-4]", args.release):
        raise SystemExit("--release must use GtoPdb YYYY.N format")
    if args.pubmed_batch_size < 1 or args.pubmed_batch_size > 200:
        raise SystemExit("--pubmed-batch-size must be in [1, 200]")
    if args.http_attempts < 1:
        raise SystemExit("--http-attempts must be >= 1")

    expected_hashes = {
        "interactions.csv": args.interactions_sha256,
        "ligands.csv": args.ligands_sha256,
        "GtP_to_UniProt_mapping.csv": args.target_mapping_sha256,
        "file_descriptions.txt": args.file_descriptions_sha256,
    }
    for name, value in expected_hashes.items():
        if not re.fullmatch(r"[0-9a-f]{64}", str(value)):
            raise SystemExit(f"Expected sha256 for {name} must be 64 lowercase hex characters")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "source_manifest.json"
    manifest_path.unlink(missing_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "SkinScout/GtoPdb-mirror"})
    artifacts: dict[str, Any] = {}
    for name, url in ARTIFACT_URLS.items():
        path = args.out_dir / name
        if args.refresh or not path.exists():
            if args.offline:
                raise SystemExit(f"--offline requires cached GtoPdb artifact: {path}")
            _download(session, url, path)
        artifacts[name] = _validate_download(
            path,
            expected_sha256=expected_hashes[name],
            release=args.release,
            release_date=release_date,
        )

    pmids = _interaction_pmids(args.out_dir / "interactions.csv")
    pubmed_path = args.out_dir / "pubmed_esummary.json"
    if args.refresh_pubmed or not pubmed_path.exists():
        if args.offline:
            raise SystemExit(f"--offline requires cached PubMed metadata: {pubmed_path}")
        pubmed = _build_pubmed_cache(
            session,
            pmids,
            out_path=pubmed_path,
            batch_size=args.pubmed_batch_size,
            email=args.ncbi_email,
            api_key=args.ncbi_api_key,
            attempts=args.http_attempts,
        )
    else:
        pubmed = _load_pubmed_cache(pubmed_path, pmids)
    records = pubmed["records"]
    missing_pmids = pubmed["missing_pmids"]
    artifacts["pubmed_esummary.json"] = {
        "path": pubmed_path.name,
        "url": PUBMED_ENDPOINT,
        "sha256": _sha256_file(pubmed_path),
        "bytes": pubmed_path.stat().st_size,
        "requested_pmids": len(pmids),
        "resolved_pmids": len(records),
        "missing_pmids": len(missing_pmids),
        "requested_pmids_sha256": _hash_values(pmids),
    }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": {
            "name": SOURCE_NAME,
            "release": args.release,
            "release_date": release_date,
            "official_url": SOURCE_OFFICIAL_URL,
            "downloads_url": SOURCE_DOWNLOADS_URL,
            "license": SOURCE_LICENSE,
            "license_url": SOURCE_LICENSE_URL,
            "redistribution": "allowed",
            "redistribution_requirements": (
                "database attribution/share-alike under ODbL 1.0 and "
                "content attribution/share-alike under CC BY-SA 4.0"
            ),
        },
        "artifacts": artifacts,
        "policies": {
            "release_pin": (
                "every unversioned CSV URL is accepted only when its embedded release banner "
                "and configured sha256 match the pinned release"
            ),
            "pubmed_provenance": (
                "all pipe-delimited interaction PMIDs are queried through NCBI ESummary; "
                "the requested identifier-set hash and returned records are persisted"
            ),
            "network_cache": "--offline never downloads; --refresh and --refresh-pubmed are explicit",
        },
        "manifest_sha256_policy": (
            "source_manifest.json is excluded from artifact hashes because stable self-hashing is impossible"
        ),
    }
    _write_json_atomic(manifest, manifest_path)
    return manifest


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--release", required=True)
    parser.add_argument("--release-date", required=True)
    parser.add_argument("--interactions-sha256", required=True)
    parser.add_argument("--ligands-sha256", required=True)
    parser.add_argument("--target-mapping-sha256", required=True)
    parser.add_argument("--file-descriptions-sha256", required=True)
    parser.add_argument("--pubmed-batch-size", type=int, default=200)
    parser.add_argument("--http-attempts", type=int, default=4)
    parser.add_argument("--ncbi-email", default=os.environ.get("NCBI_EMAIL", ""))
    parser.add_argument("--ncbi-api-key", default=os.environ.get("NCBI_API_KEY", ""))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--refresh-pubmed", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        manifest = mirror(args)
    except Exception as exc:
        print(f"[stage0.gtopdb][FATAL] {exc}", file=sys.stderr)
        return 1
    artifacts = manifest["artifacts"]
    print(
        "[stage0.gtopdb] mirrored "
        f"release={manifest['source']['release']} "
        f"interactions={artifacts['interactions.csv']['rows']} "
        f"pubmed={artifacts['pubmed_esummary.json']['resolved_pmids']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
