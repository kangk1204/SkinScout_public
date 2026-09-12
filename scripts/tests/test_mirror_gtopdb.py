from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from mirror_gtopdb import (  # noqa: E402
    EXPECTED_HEADERS,
    PUBMED_SCHEMA_VERSION,
    _hash_values,
    _interaction_pmids,
    _load_pubmed_cache,
    _validate_download,
)


def _write_interactions(path: Path, pmids: list[str]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["# GtoPdb Version: 2026.2 - published: 2026-06-15"])
        dict_writer = csv.DictWriter(
            handle,
            fieldnames=EXPECTED_HEADERS["interactions.csv"],
        )
        dict_writer.writeheader()
        for pmid in pmids:
            row = {column: "" for column in EXPECTED_HEADERS["interactions.csv"]}
            row["PubMed ID"] = pmid
            dict_writer.writerow(row)
    return path


def test_release_csv_validation_binds_hash_banner_header_and_rows(tmp_path: Path) -> None:
    path = _write_interactions(tmp_path / "interactions.csv", ["100", "200|300"])
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    artifact = _validate_download(
        path,
        expected_sha256=sha256,
        release="2026.2",
        release_date="2026-06-15",
    )

    assert artifact["rows"] == 2
    assert artifact["path"] == "interactions.csv"
    assert artifact["sha256"] == sha256
    assert artifact["release_banner"].startswith("# GtoPdb Version: 2026.2")
    assert _interaction_pmids(path) == ["100", "200", "300"]


def test_interaction_pmid_parser_fails_closed_on_non_numeric_token(tmp_path: Path) -> None:
    path = _write_interactions(tmp_path / "interactions.csv", ["100|not-a-pmid"])

    with pytest.raises(SystemExit, match="Invalid GtoPdb PubMed ID"):
        _interaction_pmids(path)


def test_pubmed_cache_requires_exact_identifier_universe(tmp_path: Path) -> None:
    path = tmp_path / "pubmed_esummary.json"
    payload = {
        "schema_version": PUBMED_SCHEMA_VERSION,
        "query": {
            "requested_count": 2,
            "requested_pmids_sha256": _hash_values(["100", "200"]),
        },
        "records": {"100": {"uid": "100"}, "200": {"uid": "200"}},
        "missing_pmids": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = _load_pubmed_cache(path, ["100", "200"])
    assert sorted(loaded["records"]) == ["100", "200"]

    with pytest.raises(SystemExit, match="query count"):
        _load_pubmed_cache(path, ["100"])
