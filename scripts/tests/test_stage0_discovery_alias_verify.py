from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import build_discovery_alias_map as alias_map  # noqa: E402
import build_discovery_alias_sources as alias_sources  # noqa: E402
import mirror_pubchem_alias_source as pubchem_mirror  # noqa: E402
from stage0_verify import chk_discovery_alias_integrity  # noqa: E402


PUBCHEM_RELEASE = "2026-08-01"
PUBCHEM_URL = (
    "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Monthly/"
    f"{PUBCHEM_RELEASE}/Extras/CID-SMILES.gz"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_pubchem_source(path: Path) -> str:
    with gzip.GzipFile(filename=str(path), mode="wb", mtime=0) as handle:
        handle.write(b"1\tCCO\n")
    return hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()


def _write_alias_package(repo: Path) -> None:
    data = repo / "data"
    workflow = repo / "workflow"
    workflow.mkdir(parents=True)
    source = repo / "pubchem-source.gz"
    pubchem_md5 = _write_pubchem_source(source)
    workflow.joinpath("config.yaml").write_text(
        "stage0:\n"
        f'  pubchem_alias_release_date: "{PUBCHEM_RELEASE}"\n'
        f'  pubchem_alias_smiles_md5: "{pubchem_md5}"\n'
        f'  pubchem_alias_smiles_url: "{PUBCHEM_URL}"\n',
        encoding="utf-8",
    )
    pubchem_mirror.mirror(
        out_dir=data / "pubchem",
        release_date=PUBCHEM_RELEASE,
        expected_md5=pubchem_md5,
        url=PUBCHEM_URL,
        source_file=source,
        source_last_modified_date=PUBCHEM_RELEASE,
    )

    input_paths = {
        "ChEMBL": (
            data / "chembl37" / "chembl_37" / "chembl_37_sqlite" / "chembl_37.db"
        ),
        "BindingDB": data / "bindingdb" / "BindingDB_All.tsv",
        "GtoPdb": data / "gtopdb" / "ligands.csv",
        "PubChem": data / "pubchem" / "CID-SMILES.gz",
    }
    for name, path in input_paths.items():
        if name != "PubChem":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{name} source\n".encode("ascii"))
    manifest_paths = {
        "ChEMBL": data / "chembl37" / "source_manifest.json",
        "BindingDB": data / "bindingdb" / "bindingdb_source_manifest.json",
        "GtoPdb": data / "gtopdb" / "source_manifest.json",
        "PubChem": data / "pubchem" / "source_manifest.json",
    }
    common_license_url = "https://example.org/license"
    source_metadata = {
        "ChEMBL": {
            "name": "ChEMBL",
            "release": "37",
            "release_date": "2026-05-01",
            "license": "CC-BY-SA-3.0",
            "license_url": common_license_url,
        },
        "BindingDB": {
            "name": "BindingDB",
            "release": "2026-08",
            "release_date": "2026-08-01",
            "license": "CC-BY-3.0",
            "license_url": common_license_url,
        },
        "GtoPdb": {
            "name": "GtoPdb",
            "release": "2026.2",
            "release_date": "2026-06-15",
            "license": "LicenseRef-GtoPdb",
            "license_url": common_license_url,
        },
    }
    chembl_input = input_paths["ChEMBL"]
    manifest_paths["ChEMBL"].write_text(json.dumps({
        "schema_version": "chembl_activity_evidence.v1",
        "source": {
            **source_metadata["ChEMBL"],
            "source_db_sha256": _sha256(chembl_input),
            "source_db_bytes": chembl_input.stat().st_size,
        },
    }) + "\n")
    bindingdb_input = input_paths["BindingDB"]
    manifest_paths["BindingDB"].write_text(json.dumps({
        "schema_version": 1,
        "release": source_metadata["BindingDB"]["release"],
        "release_date": source_metadata["BindingDB"]["release_date"],
        "source": source_metadata["BindingDB"],
        "extracted": {
            "sha256": _sha256(bindingdb_input),
            "bytes": bindingdb_input.stat().st_size,
        },
    }) + "\n")
    gtopdb_input = input_paths["GtoPdb"]
    manifest_paths["GtoPdb"].write_text(json.dumps({
        "schema_version": "skinscout.gtopdb-source.v1",
        "source": source_metadata["GtoPdb"],
        "artifacts": {
            "ligands.csv": {
                "sha256": _sha256(gtopdb_input),
                "bytes": gtopdb_input.stat().st_size,
            },
        },
    }) + "\n")

    source_dir = data / "discovery_aliases" / "sources"
    source_dir.mkdir(parents=True)
    rows = {
        "ChEMBL": ("CCO", "CHEMBL1", "CHEMBL:1"),
        "BindingDB": ("CCN", "BDBM1", "BindingDB:1"),
        "GtoPdb": ("CCC", "GtoPdb 1", "GtoPdb:1"),
        "PubChem": ("CCO", "PubChem CID 1", "PubChem:1"),
    }
    releases = {
        "ChEMBL": ("37", "2026-05-01", "CC-BY-SA-3.0"),
        "BindingDB": ("2026-08", "2026-08-01", "CC-BY-3.0"),
        "GtoPdb": ("2026.2", "2026-06-15", "LicenseRef-GtoPdb"),
        "PubChem": (
            PUBCHEM_RELEASE,
            PUBCHEM_RELEASE,
            "LicenseRef-PubChem-Data-Usage",
        ),
    }
    registry_sources = []
    output_hashes: dict[str, str] = {}
    output_bytes: dict[str, int] = {}
    output_rows: dict[str, int] = {}
    for canonical_name in alias_sources.SOURCE_ORDER:
        filename = alias_sources.OUTPUT_FILES[canonical_name]
        artifact = source_dir / filename
        smiles, alias, record_id = rows[canonical_name]
        table = pa.Table.from_pylist(
            [{
                "smiles": smiles,
                "alias": alias,
                "inchikey": "",
                "source_record_id": record_id,
            }],
            schema=alias_sources.OUTPUT_SCHEMA,
        )
        pq.write_table(table, artifact, compression="zstd", use_dictionary=False)
        release, release_date, license_id = releases[canonical_name]
        digest = _sha256(artifact)
        output_hashes[filename] = digest
        output_bytes[filename] = artifact.stat().st_size
        output_rows[filename] = 1
        registry_sources.append({
            "canonical_name": canonical_name,
            "release": release,
            "release_date": release_date,
            "spdx_license": license_id,
            "license": license_id,
            "license_url": json.loads(
                manifest_paths[canonical_name].read_text()
            )["source"]["license_url"],
            "redistribution": "allowed",
            "artifact": {
                "path": filename,
                "format": "parquet",
                "sha256": digest,
                "bytes": artifact.stat().st_size,
                "rows": 1,
            },
            "columns": {
                "smiles": "smiles",
                "aliases": ["alias"],
                "inchikey": "inchikey",
                "source_record_id": "source_record_id",
            },
        })
    registry = {
        "schema_version": alias_sources.REGISTRY_SCHEMA_VERSION,
        "sources": registry_sources,
        "registry_root": ".",
    }
    registry_path = source_dir / "source_registry.json"
    registry_path.write_text(alias_sources._stable_json(registry) + "\n")
    source_inputs = {}
    for canonical_name in alias_sources.SOURCE_ORDER:
        upstream = json.loads(manifest_paths[canonical_name].read_text())
        upstream_source = upstream["source"]
        input_path = input_paths[canonical_name]
        manifest_path = manifest_paths[canonical_name]
        source_inputs[canonical_name] = {
            "path": alias_sources._relative_path(input_path, source_dir),
            "sha256": _sha256(input_path),
            "bytes": input_path.stat().st_size,
            "manifest": {
                "path": alias_sources._relative_path(manifest_path, source_dir),
                "sha256": _sha256(manifest_path),
                "bytes": manifest_path.stat().st_size,
                "release": upstream_source.get(
                    "release", upstream.get("release")
                ),
                "release_date": upstream_source.get(
                    "release_date", upstream.get("release_date")
                ),
                "license": upstream_source["license"],
                "license_url": upstream_source["license_url"],
            },
        }
    missing_cids_audit = source_dir / alias_sources.PUBCHEM_MISSING_CIDS_FILE
    missing_cids_audit.write_text("", encoding="utf-8")
    source_manifest_unsigned = {
        "schema_version": alias_sources.SOURCE_SCHEMA_VERSION,
        "builder": {
            "path": "scripts/build_discovery_alias_sources.py",
            "sha256": _sha256(SCRIPTS / "build_discovery_alias_sources.py"),
        },
        "created_at_utc": "1970-01-01T00:00:00Z",
        "inputs": source_inputs,
        "output_sha256": output_hashes,
        "output_bytes": output_bytes,
        "output_rows": output_rows,
        "audit_artifacts": {
            missing_cids_audit.name: {
                "sha256": _sha256(missing_cids_audit),
                "bytes": missing_cids_audit.stat().st_size,
                "rows": 0,
            }
        },
        "source_stats": {
            "BindingDB": {
                "input_rows": 1,
                "missing_smiles_rows": 0,
                "eligible_rows": 1,
            },
            "PubChem": {
                "selected_cids": 1,
                "matched_cids": 1,
                "missing_cids": 0,
                "missing_fraction_ppm": 0,
            },
        },
        "source_registry_sha256": _sha256(registry_path),
        "source_registry": "source_registry.json",
        "policy": {
            "fail_closed": True,
            "PubChem_max_missing_fraction_ppm": (
                alias_sources.MAX_MISSING_PUBCHEM_CID_FRACTION_PPM
            ),
        },
    }
    source_manifest = {
        **source_manifest_unsigned,
        "self_binding_sha256": alias_sources._hash_payload(source_manifest_unsigned),
    }
    source_dir.joinpath("source_manifest.json").write_text(
        alias_sources._stable_json(source_manifest) + "\n"
    )

    alias_dir = data / "discovery_aliases"
    alias_map.build(
        source_registry=registry_path,
        source_manifest=source_dir / "source_manifest.json",
        out_aliases_parquet=alias_dir / "aliases.parquet",
        out_direct_reference=alias_dir / "direct_exact_reference.smi",
        out_manifest=alias_dir / "manifest.json",
    )


def test_discovery_alias_integrity_accepts_sealed_package(tmp_path: Path) -> None:
    _write_alias_package(tmp_path)

    result = chk_discovery_alias_integrity(tmp_path)

    assert result.ok, result.detail


def test_discovery_alias_integrity_rejects_tampered_direct_reference(
    tmp_path: Path,
) -> None:
    _write_alias_package(tmp_path)
    direct = tmp_path / "data/discovery_aliases/direct_exact_reference.smi"
    direct.write_text("CCC " + "0" * 64 + "\n", encoding="utf-8")

    result = chk_discovery_alias_integrity(tmp_path)

    assert not result.ok
    assert "direct.output_binding" in result.detail


def test_discovery_alias_integrity_rejects_stale_upstream_manifest(
    tmp_path: Path,
) -> None:
    _write_alias_package(tmp_path)
    upstream = tmp_path / "data" / "chembl37" / "source_manifest.json"
    payload = json.loads(upstream.read_text())
    payload["changed_after_alias_build"] = True
    upstream.write_text(json.dumps(payload) + "\n")

    result = chk_discovery_alias_integrity(tmp_path)

    assert not result.ok
    assert "sources.inputs.ChEMBL.manifest.sha256" in result.detail


def test_discovery_alias_integrity_rejects_same_size_source_drift(
    tmp_path: Path,
) -> None:
    _write_alias_package(tmp_path)
    source = (
        tmp_path
        / "data"
        / "chembl37"
        / "chembl_37"
        / "chembl_37_sqlite"
        / "chembl_37.db"
    )
    source.write_bytes(b"X" * source.stat().st_size)

    result = chk_discovery_alias_integrity(tmp_path)

    assert not result.ok
    assert "sources.inputs.ChEMBL.sha256" in result.detail


def test_discovery_alias_integrity_rejects_tampered_missing_cid_audit(
    tmp_path: Path,
) -> None:
    _write_alias_package(tmp_path)
    audit = (
        tmp_path
        / "data"
        / "discovery_aliases"
        / "sources"
        / alias_sources.PUBCHEM_MISSING_CIDS_FILE
    )
    audit.write_text("123\n", encoding="utf-8")

    result = chk_discovery_alias_integrity(tmp_path)

    assert not result.ok
    assert "sources.PubChem.audit_hash_bytes" in result.detail


def test_discovery_alias_integrity_rejects_tampered_canonicalization_audit(
    tmp_path: Path,
) -> None:
    _write_alias_package(tmp_path)
    audit = (
        tmp_path
        / "data"
        / "discovery_aliases"
        / alias_map.CANONICALIZATION_AUDIT_FILENAME
    )
    audit.write_text(
        json.dumps(
            {
                "input_smiles_sha256": "0" * 64,
                "reason_code": "rdkit_canonicalization_failed",
                "row_index": 999,
                "source": "GtoPdb",
                "source_record_id": "forged",
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    result = chk_discovery_alias_integrity(tmp_path)

    assert not result.ok
    assert "aliases.canonicalization_audit.output_binding" in result.detail
