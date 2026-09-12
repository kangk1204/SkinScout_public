from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import build_discovery_alias_sources as alias_sources  # noqa: E402
from build_discovery_alias_sources import (  # noqa: E402
    GTOPDB_HEADER,
    OUTPUT_COLUMNS,
    _hash_payload,
    _parser,
    build,
    validate_source_manifest,
)


SCRIPT = SCRIPTS / "build_discovery_alias_sources.py"
pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(
    path: Path,
    artifact: Path,
    *,
    source: str,
    release: str,
    release_date: str = "2026-01-02",
    license_text: str = "LicenseRef-Test",
    schema_version: str = "test.source.v1",
) -> Path:
    source_meta = {
        "name": source,
        "release": release,
        "release_date": release_date,
        "license": license_text,
        "license_url": f"https://example.invalid/{source}/license",
        "redistribution": "allowed",
    }
    artifact_meta = {
        "path": artifact.name,
        "sha256": _sha256(artifact),
        "bytes": artifact.stat().st_size,
    }
    if source == "ChEMBL":
        payload = {
            "schema_version": "chembl_activity_evidence.v1",
            "source": {
                **source_meta,
                "source_db": artifact.name,
                "source_db_sha256": artifact_meta["sha256"],
                "source_db_bytes": artifact_meta["bytes"],
            },
        }
    elif source == "BindingDB":
        payload = {
            "schema_version": 1,
            "source": source_meta,
            "release": release,
            "release_date": release_date,
            "extracted": artifact_meta,
        }
    elif source == "GtoPdb":
        payload = {
            "schema_version": "skinscout.gtopdb-source.v1",
            "source": source_meta,
            "artifacts": {"ligands.csv": artifact_meta},
        }
    elif source == "PubChem":
        payload = {
            "schema_version": schema_version,
            "source": source_meta,
            "artifact": artifact_meta,
        }
    else:
        raise AssertionError(f"unknown fixture source {source}")
    path.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _write_chembl(path: Path) -> Path:
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE molecule_dictionary (
            molregno INTEGER PRIMARY KEY,
            chembl_id TEXT NOT NULL,
            pref_name TEXT
        );
        CREATE TABLE compound_structures (
            molregno INTEGER PRIMARY KEY,
            canonical_smiles TEXT NOT NULL,
            standard_inchi_key TEXT
        );
        CREATE TABLE molecule_synonyms (
            molregno INTEGER,
            synonyms TEXT
        );
        INSERT INTO molecule_dictionary VALUES (1, 'CHEMBL1', 'ethanol');
        INSERT INTO compound_structures VALUES
            (1, 'CCO', 'LFQSCWFLJHTTHZ-UHFFFAOYSA-N');
        INSERT INTO molecule_synonyms VALUES (1, 'ethyl alcohol');
        INSERT INTO molecule_synonyms VALUES (1, 'ethanol');
        """
    )
    con.commit()
    con.close()
    return path


def _write_bindingdb(path: Path) -> Path:
    path.write_text(
        "\t".join(
            [
                "Ligand SMILES",
                "Ligand InChI Key",
                "BindingDB Ligand Name",
                "BindingDB MonomerID",
                "PubChem CID",
                "ChEMBL ID",
            ]
        )
        + "\n"
        + "CCN\tIKHGUXGNUITLKF-UHFFFAOYSA-N\tethylamine\tBDBM1\t100\tCHEMBL2\n"
        + "CCN\tIKHGUXGNUITLKF-UHFFFAOYSA-N\tethylamine\tBDBM1\t100\tCHEMBL2\n",
        encoding="utf-8",
    )
    return path


def _write_gtopdb(path: Path, *, malformed_header: bool = False) -> Path:
    header = list(GTOPDB_HEADER)
    if malformed_header:
        header[0] = "Bad ID"
    with path.open("w", encoding="utf-8", newline="") as handle:
        banner_writer = csv.writer(handle, quoting=csv.QUOTE_ALL)
        writer = csv.DictWriter(handle, fieldnames=header)
        banner_writer.writerow(["# GtoPdb Version: 2026.1 - published: 2026-01-02"])
        writer.writeheader()
        row = {column: "" for column in header}
        row.update(
            {
                "Ligand ID": "7",
                "Name": "histamine",
                "PubChem CID": "200",
                "IUPAC name": "2-(1H-imidazol-4-yl)ethanamine",
                "INN": "histamine",
                "Synonyms": "2-imidazol-4-ylethanamine|beta-aminoethylimidazole",
                "SMILES": "NCC1=CN=CN1",
                "InChIKey": "NTYJJOPFIAHURM-UHFFFAOYSA-N",
                "ChEMBL ID": "CHEMBL90",
            }
        )
        writer.writerow({column: row.get(column, "") for column in header})
        no_structure = {column: "" for column in header}
        no_structure.update(
            {
                "Ligand ID": "8",
                "Name": "structureless peptide",
                "Type": "Peptide",
            }
        )
        writer.writerow({column: no_structure.get(column, "") for column in header})
    return path


def _write_pubchem(path: Path, *, include_all: bool = True) -> Path:
    lines = ["100\tCCN\n"]
    if include_all:
        lines.append("200\tNCC1=CN=CN1\n")
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        handle.writelines(lines)
    return path


def _fixture(tmp_path: Path, *, pubchem_include_all: bool = True) -> dict[str, Path]:
    paths = {
        "chembl": _write_chembl(tmp_path / "chembl.sqlite"),
        "bindingdb": _write_bindingdb(tmp_path / "BindingDB_All.tsv"),
        "gtopdb": _write_gtopdb(tmp_path / "ligands.csv"),
        "pubchem": _write_pubchem(
            tmp_path / "pubchem_smiles.tsv.gz", include_all=pubchem_include_all
        ),
        "out": tmp_path / "out",
    }
    paths["chembl_manifest"] = _manifest(
        tmp_path / "chembl_manifest.json",
        paths["chembl"],
        source="ChEMBL",
        release="chembl_37",
        release_date="2026-01-02",
        license_text="CC-BY-SA-3.0",
    )
    paths["bindingdb_manifest"] = _manifest(
        tmp_path / "bindingdb_source_manifest.json",
        paths["bindingdb"],
        source="BindingDB",
        release="2026-01",
        release_date="2026-01-03",
        license_text="LicenseRef-BindingDB",
    )
    paths["gtopdb_manifest"] = _manifest(
        tmp_path / "gtopdb_manifest.json",
        paths["gtopdb"],
        source="GtoPdb",
        release="2026.1",
        release_date="2026-01-02",
        license_text="CC-BY-SA-4.0",
    )
    paths["pubchem_manifest"] = _manifest(
        tmp_path / "pubchem_manifest.json",
        paths["pubchem"],
        source="PubChem",
        release="2026-01",
        release_date="2026-01-04",
        license_text="LicenseRef-PubChem",
        schema_version="skinscout.pubchem-alias-source.v1",
    )
    return paths


def _args(paths: dict[str, Path], out_dir: Path | None = None) -> list[str]:
    return [
        "--chembl-db",
        str(paths["chembl"]),
        "--chembl-manifest",
        str(paths["chembl_manifest"]),
        "--chembl-release-date",
        "2026-01-02",
        "--bindingdb-tsv",
        str(paths["bindingdb"]),
        "--bindingdb-manifest",
        str(paths["bindingdb_manifest"]),
        "--bindingdb-release-date",
        "2026-01-03",
        "--gtopdb-ligands",
        str(paths["gtopdb"]),
        "--gtopdb-manifest",
        str(paths["gtopdb_manifest"]),
        "--pubchem-smiles-gz",
        str(paths["pubchem"]),
        "--pubchem-manifest",
        str(paths["pubchem_manifest"]),
        "--out-dir",
        str(out_dir or paths["out"]),
    ]


def _run(paths: dict[str, Path], out_dir: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *_args(paths, out_dir)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_builder_success_emits_four_parquets_registry_and_manifest(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)

    result = _run(paths)

    assert result.returncode == 0, result.stderr
    manifest = json.loads((paths["out"] / "source_manifest.json").read_text())
    registry = json.loads((paths["out"] / "source_registry.json").read_text())
    assert manifest["schema_version"] == "skinscout.discovery-alias-sources.v1"
    assert registry["schema_version"] == "discovery_alias_source_registry.v1"
    assert [source["canonical_name"] for source in registry["sources"]] == [
        "ChEMBL",
        "BindingDB",
        "GtoPdb",
        "PubChem",
    ]
    for artifact_name in manifest["output_sha256"]:
        frame = pd.read_parquet(paths["out"] / artifact_name)
        parquet = pq.ParquetFile(paths["out"] / artifact_name)
        assert tuple(frame.columns) == OUTPUT_COLUMNS
        assert tuple(str(field.type) for field in parquet.schema_arrow) == tuple(
            ["string"] * len(OUTPUT_COLUMNS)
        )
        assert parquet.metadata.num_rows == len(frame)
        assert not frame.empty
    chembl = pd.read_parquet(paths["out"] / "chembl_aliases.parquet")
    assert {"CHEMBL1", "ethanol", "ethyl alcohol"} <= set(chembl["alias"])
    gtopdb = pd.read_parquet(paths["out"] / "gtopdb_aliases.parquet")
    assert "structureless peptide" not in set(gtopdb["alias"])
    pubchem = pd.read_parquet(paths["out"] / "pubchem_aliases.parquet")
    assert set(pubchem["alias"]) == {"PubChem CID 100", "PubChem CID 200"}
    validate_source_manifest(paths["out"] / "source_manifest.json")


def test_bindingdb_rows_without_smiles_are_counted_and_excluded(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    with paths["bindingdb"].open("a", encoding="utf-8") as handle:
        handle.write("\t\tprotein ligand\tBDBM2\t\t\n")
    paths["bindingdb_manifest"] = _manifest(
        paths["bindingdb_manifest"],
        paths["bindingdb"],
        source="BindingDB",
        release="2026-01",
        release_date="2026-01-03",
        license_text="LicenseRef-BindingDB",
    )

    result = _run(paths)

    assert result.returncode == 0, result.stderr
    manifest = json.loads((paths["out"] / "source_manifest.json").read_text())
    assert manifest["source_stats"]["BindingDB"] == {
        "input_rows": 3,
        "missing_smiles_rows": 1,
        "eligible_rows": 2,
    }
    bindingdb = pd.read_parquet(paths["out"] / "bindingdb_aliases.parquet")
    assert "protein ligand" not in set(bindingdb["alias"])


def test_builder_is_deterministic_on_rerun(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    first = _run(paths)
    assert first.returncode == 0, first.stderr
    first_hashes = {
        path.name: _sha256(path)
        for path in sorted(paths["out"].glob("*"))
        if path.is_file()
    }

    second = _run(paths)

    assert second.returncode == 0, second.stderr
    assert {
        path.name: _sha256(path)
        for path in sorted(paths["out"].glob("*"))
        if path.is_file()
    } == first_hashes


def test_builder_uses_bounded_sqlite_batches_and_metadata_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture(tmp_path)
    with paths["bindingdb"].open("w", encoding="utf-8", newline="") as handle:
        handle.write(
            "\t".join(
                [
                    "Ligand SMILES",
                    "Ligand InChI Key",
                    "BindingDB Ligand Name",
                    "BindingDB MonomerID",
                    "PubChem CID",
                    "ChEMBL ID",
                ]
            )
            + "\n"
        )
        for index in range(20):
            handle.write(
                f"CCN\tIKHGUXGNUITLKF-UHFFFAOYSA-N\tethylamine {index}"
                f"\tBDBM{index}\t{100 + index}\tCHEMBL{index}\n"
            )
    with gzip.open(paths["pubchem"], "wt", encoding="utf-8", newline="") as handle:
        for cid in range(100, 120):
            handle.write(f"{cid}\tCCN\n")
        handle.write("200\tNCC1=CN=CN1\n")
    _manifest(
        paths["bindingdb_manifest"],
        paths["bindingdb"],
        source="BindingDB",
        release="2026-01",
        release_date="2026-01-03",
        license_text="LicenseRef-BindingDB",
    )
    _manifest(
        paths["pubchem_manifest"],
        paths["pubchem"],
        source="PubChem",
        release="2026-01",
        release_date="2026-01-04",
        license_text="LicenseRef-PubChem",
        schema_version="skinscout.pubchem-alias-source.v1",
    )
    observed_alias_batch_sizes: list[int] = []
    original_add_aliases = alias_sources.AliasStaging.add_aliases

    def wrapped_add_aliases(
        self: alias_sources.AliasStaging,
        source: str,
        rows: list[tuple[str, str, str, str]],
    ) -> None:
        observed_alias_batch_sizes.append(len(rows))
        assert len(rows) <= 3
        original_add_aliases(self, source, rows)

    def forbidden_read_table(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("builder validation must not call pq.read_table")

    monkeypatch.setattr(alias_sources, "STAGING_INSERT_BATCH_SIZE", 3)
    monkeypatch.setattr(alias_sources, "PARQUET_EXPORT_BATCH_SIZE", 4)
    monkeypatch.setattr(alias_sources.AliasStaging, "add_aliases", wrapped_add_aliases)
    monkeypatch.setattr(alias_sources.pq, "read_table", forbidden_read_table)

    manifest = build(_parser().parse_args(_args(paths)))

    assert manifest["output_rows"]["bindingdb_aliases.parquet"] > 20
    assert max(observed_alias_batch_sizes) <= 3
    assert not list(tmp_path.glob(".alias-stage-*"))


def test_source_hash_drift_fails_closed(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths["bindingdb"].write_text(paths["bindingdb"].read_text() + "CCC\t\tname\tBDBM2\t100\t\n")

    result = _run(paths)

    assert result.returncode != 0
    assert "BindingDB manifest does not bind artifact sha256" in result.stderr
    assert not (paths["out"] / "source_manifest.json").exists()


@pytest.mark.parametrize(
    ("source_key", "manifest_key", "container_keys", "path_key", "sha_key"),
    [
        ("chembl", "chembl_manifest", ("source",), "source_db", "source_db_sha256"),
        ("bindingdb", "bindingdb_manifest", ("extracted",), "path", "sha256"),
        ("gtopdb", "gtopdb_manifest", ("artifacts", "ligands.csv"), "path", "sha256"),
        ("pubchem", "pubchem_manifest", ("artifact",), "path", "sha256"),
    ],
)
def test_manifest_decoy_hash_does_not_satisfy_contractual_binding(
    tmp_path: Path,
    source_key: str,
    manifest_key: str,
    container_keys: tuple[str, ...],
    path_key: str,
    sha_key: str,
) -> None:
    paths = _fixture(tmp_path)
    payload = json.loads(paths[manifest_key].read_text())
    payload["decoy"] = {"sha256": _sha256(paths[source_key])}
    container = payload
    for key in container_keys:
        container = container[key]
    container[sha_key] = "0" * 64
    paths[manifest_key].write_text(json.dumps(payload, sort_keys=True) + "\n")

    result = _run(paths)

    assert result.returncode != 0
    assert "manifest does not bind artifact sha256" in result.stderr
    assert not (paths["out"] / "source_manifest.json").exists()


@pytest.mark.parametrize(
    ("source_key", "manifest_key", "container_keys", "path_key"),
    [
        ("chembl", "chembl_manifest", ("source",), "source_db"),
        ("bindingdb", "bindingdb_manifest", ("extracted",), "path"),
        ("gtopdb", "gtopdb_manifest", ("artifacts", "ligands.csv"), "path"),
        ("pubchem", "pubchem_manifest", ("artifact",), "path"),
    ],
)
def test_manifest_decoy_path_does_not_satisfy_contractual_binding(
    tmp_path: Path,
    source_key: str,
    manifest_key: str,
    container_keys: tuple[str, ...],
    path_key: str,
) -> None:
    paths = _fixture(tmp_path)
    payload = json.loads(paths[manifest_key].read_text())
    payload["decoy"] = {"path": paths[source_key].name}
    container = payload
    for key in container_keys:
        container = container[key]
    container[path_key] = "wrong-artifact.dat"
    paths[manifest_key].write_text(json.dumps(payload, sort_keys=True) + "\n")

    result = _run(paths)

    assert result.returncode != 0
    assert "manifest does not bind artifact path" in result.stderr
    assert not (paths["out"] / "source_manifest.json").exists()


def test_bindingdb_top_level_release_date_is_contractual(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    payload = json.loads(paths["bindingdb_manifest"].read_text())
    payload["source"]["release_date"] = "2026-01-03"
    payload["decoy"] = {"release_date": "2026-01-03"}
    payload["release_date"] = "2026-01-31"
    paths["bindingdb_manifest"].write_text(json.dumps(payload, sort_keys=True) + "\n")

    result = _run(paths)

    assert result.returncode != 0
    assert "BindingDB manifest release_date must be 2026-01-03" in result.stderr


def test_malformed_gtopdb_header_fails_closed(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _write_gtopdb(paths["gtopdb"], malformed_header=True)
    _manifest(
        paths["gtopdb_manifest"],
        paths["gtopdb"],
        source="GtoPdb",
        release="2026.1",
        release_date="2026-01-02",
        license_text="CC-BY-SA-4.0",
    )

    result = _run(paths)

    assert result.returncode != 0
    assert "GtoPdb ligands CSV header does not match contract" in result.stderr


def test_missing_pubchem_selected_cid_fails_closed(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, pubchem_include_all=False)

    result = _run(paths)

    assert result.returncode != 0
    assert "PubChem missing selected CID" in result.stderr


def test_bounded_missing_pubchem_cids_are_audited(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    with paths["bindingdb"].open("a", encoding="utf-8") as handle:
        for cid in range(101, 110):
            handle.write(
                f"CCN\tIKHGUXGNUITLKF-UHFFFAOYSA-N\tethylamine {cid}"
                f"\tBDBM{cid}\t{cid}\tCHEMBL2\n"
            )
    paths["bindingdb_manifest"] = _manifest(
        paths["bindingdb_manifest"],
        paths["bindingdb"],
        source="BindingDB",
        release="2026-01",
        release_date="2026-01-03",
        license_text="LicenseRef-BindingDB",
    )
    with gzip.open(paths["pubchem"], "wt", encoding="utf-8", newline="") as handle:
        for cid in range(100, 109):
            handle.write(f"{cid}\tCCN\n")
        handle.write("200\tNCC1=CN=CN1\n")
    paths["pubchem_manifest"] = _manifest(
        paths["pubchem_manifest"],
        paths["pubchem"],
        source="PubChem",
        release="2026-01",
        release_date="2026-01-04",
        license_text="LicenseRef-PubChem",
        schema_version="skinscout.pubchem-alias-source.v1",
    )

    result = _run(paths)

    assert result.returncode == 0, result.stderr
    manifest_path = paths["out"] / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["source_stats"]["PubChem"] == {
        "selected_cids": 11,
        "matched_cids": 10,
        "missing_cids": 1,
        "missing_fraction_ppm": 90_910,
    }
    audit_path = paths["out"] / "pubchem_missing_cids.txt"
    assert audit_path.read_text(encoding="utf-8") == "109\n"
    assert manifest["audit_artifacts"]["pubchem_missing_cids.txt"] == {
        "sha256": _sha256(audit_path),
        "bytes": audit_path.stat().st_size,
        "rows": 1,
    }
    validate_source_manifest(manifest_path)

    audit_path.write_text("109\n110\n", encoding="utf-8")
    with pytest.raises(
        alias_sources.AliasSourceError,
        match="PubChem missing-CID audit hash/bytes drift",
    ):
        validate_source_manifest(manifest_path)


def test_symlink_input_fails_closed(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    target = paths["bindingdb"]
    link = tmp_path / "bindingdb-link.tsv"
    link.symlink_to(target)
    paths["bindingdb"] = link

    result = _run(paths)

    assert result.returncode != 0
    assert "BindingDB input must not be a symlink" in result.stderr


def test_stale_outputs_are_removed_on_failure(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, pubchem_include_all=False)
    paths["out"].mkdir()
    for name in (
        "chembl_aliases.parquet",
        "bindingdb_aliases.parquet",
        "gtopdb_aliases.parquet",
        "pubchem_aliases.parquet",
        "source_registry.json",
        "source_manifest.json",
    ):
        (paths["out"] / name).write_text("stale", encoding="utf-8")

    result = _run(paths)

    assert result.returncode != 0
    assert not any(paths["out"].iterdir())


def test_joint_output_and_manifest_forgery_is_rejected(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    result = _run(paths)
    assert result.returncode == 0, result.stderr
    pubchem_path = paths["out"] / "pubchem_aliases.parquet"
    pd.DataFrame(
        [
            {
                "smiles": "CCC",
                "alias": "PubChem CID 100",
                "inchikey": "",
                "source_record_id": "PubChem:100",
            }
        ]
    ).to_parquet(pubchem_path, index=False)
    manifest_path = paths["out"] / "source_manifest.json"
    forged = json.loads(manifest_path.read_text())
    forged["output_sha256"]["pubchem_aliases.parquet"] = _sha256(pubchem_path)
    forged["output_bytes"]["pubchem_aliases.parquet"] = pubchem_path.stat().st_size
    without_binding = dict(forged)
    without_binding.pop("self_binding_sha256", None)
    forged["self_binding_sha256"] = _hash_payload(without_binding)
    manifest_path.write_text(json.dumps(forged, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(
        Exception,
        match="registry artifact hash/bytes drift|output parquet column smiles is not string",
    ):
        validate_source_manifest(manifest_path)
