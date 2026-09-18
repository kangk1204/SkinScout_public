#!/usr/bin/env python3
"""Mirror and seal the pinned PubChem CID-SMILES source used by Discovery."""

from __future__ import annotations

import argparse
import email.utils
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

import requests


SCHEMA_VERSION = "skinscout.pubchem-alias-source.v1"
SOURCE_NAME = "PubChem"
DEFAULT_URL = (
    "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz"
)
DOWNLOADS_URL = "https://pubchem.ncbi.nlm.nih.gov/docs/downloads"
LICENSE_URL = "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/README-Extras"
SPDX_LICENSE = "LicenseRef-PubChem-Data-Usage"
ARTIFACT_NAME = "CID-SMILES.gz"
MD5_RE = re.compile(r"^[0-9a-f]{32}$")


class MirrorError(ValueError):
    """Raised when the pinned PubChem mirror contract is violated."""


def _digest_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    return _digest_file(path, "sha256")


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _binding_sha256(payload: dict[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("binding_sha256", None)
    return hashlib.sha256(_canonical_json(unsigned).encode("utf-8")).hexdigest()


def _validate_iso_date(value: str, label: str) -> str:
    try:
        return datetime.strptime(value, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError as exc:
        raise MirrorError(f"{label} must be an ISO date YYYY-MM-DD") from exc


def _safe_regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise MirrorError(f"{label} is missing, empty, or unsafe: {path}")
    return path


def _tmp_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    tmp.unlink(missing_ok=True)
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == destination.resolve():
        return
    tmp = _tmp_path(destination)
    tmp.unlink(missing_ok=True)
    try:
        with source.open("rb") as src, tmp.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
        os.replace(tmp, destination)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _download(url: str, destination: Path) -> str:
    if not url.startswith("https://"):
        raise MirrorError("PubChem source URL must use https")
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(destination)
    tmp.unlink(missing_ok=True)
    last_modified = ""
    try:
        with requests.get(url, stream=True, timeout=(30, 600)) as response:
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()
            if "html" in content_type:
                raise MirrorError(f"PubChem source returned HTML instead of gzip: {url}")
            last_modified = response.headers.get("Last-Modified", "")
            with tmp.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        if tmp.stat().st_size <= 0:
            raise MirrorError(f"PubChem source download was empty: {url}")
        os.replace(tmp, destination)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return last_modified


def _http_date_to_iso(value: str) -> str:
    if not value.strip():
        raise MirrorError("PubChem source Last-Modified header is missing")
    parsed = email.utils.parsedate_to_datetime(value)
    if parsed is None:
        raise MirrorError(f"PubChem source Last-Modified header is invalid: {value!r}")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%d")


def _validate_cid_smiles_stream(handle: BinaryIO, path: Path) -> dict[str, Any]:
    decompressed_sha = hashlib.sha256()
    rows = 0
    decompressed_bytes = 0
    first_cid: int | None = None
    previous_cid: int | None = None
    try:
        with gzip.GzipFile(fileobj=handle, mode="rb") as stream:
            for line_number, raw in enumerate(stream, start=1):
                decompressed_sha.update(raw)
                decompressed_bytes += len(raw)
                if not raw.endswith(b"\n"):
                    raise MirrorError(
                        f"PubChem CID-SMILES row {line_number} is not newline terminated: {path}"
                    )
                try:
                    text = raw[:-1].decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise MirrorError(
                        f"PubChem CID-SMILES row {line_number} is not UTF-8: {path}"
                    ) from exc
                fields = text.split("\t")
                if len(fields) != 2 or not fields[0].isdigit() or not fields[1].strip():
                    raise MirrorError(
                        f"PubChem CID-SMILES row {line_number} must be CID<TAB>SMILES: {path}"
                    )
                cid = int(fields[0])
                if cid <= 0:
                    raise MirrorError(
                        f"PubChem CID-SMILES row {line_number} has a nonpositive CID: {path}"
                    )
                if previous_cid is not None and cid <= previous_cid:
                    raise MirrorError(
                        "PubChem CID-SMILES CIDs must be unique and strictly increasing: "
                        f"row {line_number}, {cid} <= {previous_cid}"
                    )
                first_cid = cid if first_cid is None else first_cid
                previous_cid = cid
                rows += 1
    except (OSError, EOFError) as exc:
        raise MirrorError(f"PubChem CID-SMILES gzip failed validation: {path}: {exc}") from exc
    if rows == 0 or first_cid is None or previous_cid is None:
        raise MirrorError(f"PubChem CID-SMILES contains no records: {path}")
    return {
        "rows": rows,
        "first_cid": first_cid,
        "last_cid": previous_cid,
        "decompressed_bytes": decompressed_bytes,
        "decompressed_sha256": decompressed_sha.hexdigest(),
    }


def _validate_cid_smiles(path: Path) -> dict[str, Any]:
    _safe_regular_file(path, "PubChem CID-SMILES artifact")
    with path.open("rb") as handle:
        return _validate_cid_smiles_stream(handle, path)


def _manifest_payload(
    *,
    artifact: Path,
    release_date: str,
    expected_md5: str,
    url: str,
    last_modified_date: str,
    content: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "name": SOURCE_NAME,
            "release": release_date,
            "release_date": release_date,
            "official_url": "https://pubchem.ncbi.nlm.nih.gov/",
            "downloads_url": DOWNLOADS_URL,
            "license": SPDX_LICENSE,
            "license_url": LICENSE_URL,
            "redistribution": "allowed",
            "redistribution_scope": (
                "PubChem-generated CID and isomeric SMILES only; contributor "
                "annotations and synonyms are excluded"
            ),
        },
        "artifact": {
            "path": ARTIFACT_NAME,
            "url": url,
            "bytes": artifact.stat().st_size,
            "md5": expected_md5,
            "sha256": _sha256_file(artifact),
            "last_modified_date": last_modified_date,
            **content,
        },
        "policies": {
            "release_pin": "configured MD5 and Last-Modified date must both match",
            "selection_scope": (
                "the downstream alias builder keeps only CIDs referenced by "
                "BindingDB or GtoPdb"
            ),
        },
        "builder_script_sha256": _sha256_file(Path(__file__).resolve()),
    }


def validate_manifest(
    manifest_path: Path,
    *,
    expected_release_date: str | None = None,
    expected_md5: str | None = None,
    expected_url: str | None = DEFAULT_URL,
) -> dict[str, Any]:
    _safe_regular_file(manifest_path, "PubChem source manifest")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MirrorError(f"PubChem source manifest is invalid JSON: {manifest_path}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise MirrorError(f"PubChem source manifest schema must be {SCHEMA_VERSION}")
    if payload.get("binding_sha256") != _binding_sha256(payload):
        raise MirrorError("PubChem source manifest binding mismatch")
    source = payload.get("source")
    artifact_record = payload.get("artifact")
    if not isinstance(source, dict) or not isinstance(artifact_record, dict):
        raise MirrorError("PubChem source manifest source/artifact records are missing")
    if source.get("name") != SOURCE_NAME or source.get("redistribution") != "allowed":
        raise MirrorError("PubChem source manifest source policy mismatch")
    release_date = _validate_iso_date(str(source.get("release_date", "")), "release_date")
    if source.get("release") != release_date:
        raise MirrorError("PubChem source release must equal release_date")
    if expected_release_date is not None and release_date != expected_release_date:
        raise MirrorError("PubChem source manifest release date mismatch")
    declared_url = str(artifact_record.get("url", ""))
    if expected_url is not None and declared_url != expected_url:
        raise MirrorError("PubChem source manifest URL mismatch")
    last_modified_date = _validate_iso_date(
        str(artifact_record.get("last_modified_date", "")),
        "artifact.last_modified_date",
    )
    if last_modified_date != release_date:
        raise MirrorError(
            "PubChem source artifact.last_modified_date must equal release_date"
        )
    artifact = Path(str(artifact_record.get("path", "")))
    if artifact.is_absolute():
        raise MirrorError("PubChem source manifest artifact path must be relative")
    artifact = (manifest_path.parent / artifact).resolve()
    expected_artifact = (manifest_path.parent / ARTIFACT_NAME).resolve()
    if artifact != expected_artifact:
        raise MirrorError("PubChem source manifest artifact path mismatch")
    _safe_regular_file(artifact, "PubChem CID-SMILES artifact")
    observed_md5 = _digest_file(artifact, "md5")
    declared_md5 = str(artifact_record.get("md5", ""))
    if observed_md5 != declared_md5:
        raise MirrorError("PubChem CID-SMILES MD5 drift")
    if expected_md5 is not None and declared_md5 != expected_md5:
        raise MirrorError("PubChem source manifest configured MD5 mismatch")
    if artifact.stat().st_size != artifact_record.get("bytes"):
        raise MirrorError("PubChem CID-SMILES byte count drift")
    if _sha256_file(artifact) != artifact_record.get("sha256"):
        raise MirrorError("PubChem CID-SMILES SHA-256 drift")
    content = _validate_cid_smiles(artifact)
    for key, value in content.items():
        if artifact_record.get(key) != value:
            raise MirrorError(f"PubChem CID-SMILES {key} drift")
    if payload.get("builder_script_sha256") != _sha256_file(Path(__file__).resolve()):
        raise MirrorError("PubChem mirror builder script hash drift")
    return payload


def mirror(
    *,
    out_dir: Path,
    release_date: str,
    expected_md5: str,
    url: str,
    source_file: Path | None = None,
    source_last_modified_date: str | None = None,
    offline: bool = False,
    refresh: bool = False,
) -> dict[str, Any]:
    release_date = _validate_iso_date(release_date, "release_date")
    expected_md5 = expected_md5.strip().lower()
    if not MD5_RE.fullmatch(expected_md5):
        raise MirrorError("expected_md5 must be lowercase 32-hex MD5")
    if not url.startswith("https://"):
        raise MirrorError("PubChem source URL must use https")
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = out_dir / ARTIFACT_NAME
    manifest = out_dir / "source_manifest.json"
    for path in (_tmp_path(artifact), _tmp_path(manifest)):
        path.unlink(missing_ok=True)
    artifact_replaced = False
    try:
        last_modified_date: str
        if source_file is not None:
            source = _safe_regular_file(source_file, "PubChem source file")
            if source_last_modified_date is None:
                raise MirrorError(
                    "source_last_modified_date is required with source_file"
                )
            last_modified_date = _validate_iso_date(
                source_last_modified_date, "source_last_modified_date"
            )
            if source.resolve() != artifact.resolve():
                _copy_atomic(source, artifact)
                artifact_replaced = True
        elif artifact.exists() and not refresh:
            if not manifest.exists():
                raise MirrorError(
                    "existing PubChem artifact lacks source_manifest.json; "
                    "use --refresh to replace it"
                )
            existing = validate_manifest(
                manifest,
                expected_release_date=release_date,
                expected_md5=expected_md5,
                expected_url=url,
            )
            return existing
        elif offline:
            raise MirrorError("offline PubChem mirror requires an existing sealed artifact")
        else:
            header = _download(url, artifact)
            artifact_replaced = True
            last_modified_date = _http_date_to_iso(header)
        if last_modified_date != release_date:
            raise MirrorError(
                "PubChem source Last-Modified date mismatch: "
                f"{last_modified_date} != {release_date}"
            )
        observed_md5 = _digest_file(artifact, "md5")
        if observed_md5 != expected_md5:
            raise MirrorError(
                f"PubChem CID-SMILES MD5 mismatch: {observed_md5} != {expected_md5}"
            )
        content = _validate_cid_smiles(artifact)
        payload = _manifest_payload(
            artifact=artifact,
            release_date=release_date,
            expected_md5=expected_md5,
            url=url,
            last_modified_date=last_modified_date,
            content=content,
        )
        payload["binding_sha256"] = _binding_sha256(payload)
        _write_json_atomic(manifest, payload)
        return validate_manifest(
            manifest,
            expected_release_date=release_date,
            expected_md5=expected_md5,
            expected_url=url,
        )
    except Exception:
        manifest.unlink(missing_ok=True)
        _tmp_path(manifest).unlink(missing_ok=True)
        if artifact_replaced:
            artifact.unlink(missing_ok=True)
        _tmp_path(artifact).unlink(missing_ok=True)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--release-date", required=True)
    parser.add_argument("--expected-md5", required=True)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--source-file", type=Path)
    parser.add_argument("--source-last-modified-date")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        mirror(
            out_dir=args.out_dir,
            release_date=args.release_date,
            expected_md5=args.expected_md5,
            url=args.url,
            source_file=args.source_file,
            source_last_modified_date=args.source_last_modified_date,
            offline=args.offline,
            refresh=args.refresh,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
