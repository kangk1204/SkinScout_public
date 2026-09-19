from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from build_gtopdb_activity_evidence import _hash_values  # noqa: E402
from mirror_gtopdb import (  # noqa: E402
    EXPECTED_HEADERS,
    PUBMED_SCHEMA_VERSION,
    SCHEMA_VERSION as SOURCE_SCHEMA_VERSION,
    SOURCE_LICENSE,
    SOURCE_LICENSE_URL,
)


SCRIPT = SCRIPTS / "build_gtopdb_activity_evidence.py"
pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_release_csv(
    path: Path,
    header: tuple[str, ...],
    rows: list[dict[str, object]],
) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["# GtoPdb Version: 2026.2 - published: 2026-06-15"])
        dict_writer = csv.DictWriter(handle, fieldnames=header)
        dict_writer.writeheader()
        for row in rows:
            dict_writer.writerow({column: row.get(column, "") for column in header})
    return path


def _ligand(ligand_id: str = "1", smiles: str = "Oc1ccccc1") -> dict[str, object]:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    return {
        "Ligand ID": ligand_id,
        "Name": f"ligand-{ligand_id}",
        "Type": "Synthetic organic",
        "SMILES": smiles,
        "InChIKey": Chem.MolToInchiKey(molecule),
    }


def _interaction(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "Target": "12S-LOX",
        "Target ID": "1387",
        "Target Gene Symbol": "ALOX12",
        "Target UniProt ID": "P18054",
        "Target Species": "Human",
        "Ligand ID": "1",
        "Ligand": "ligand-1",
        "Ligand Type": "Synthetic organic",
        "Type": "Inhibitor",
        "Action": "Inhibition",
        "Affinity Units": "pIC50",
        "Affinity Median": "6.468521",
        "Original Affinity Units": "IC50",
        "Original Affinity Median nm": "340",
        "Original Affinity Relation": "=",
        "PubMed ID": "100",
    }
    row.update(overrides)
    return row


def _pubmed_record(pmid: str, publication_date: str, doi: str = "") -> dict[str, object]:
    articleids = [{"idtype": "pubmed", "value": pmid}]
    if doi:
        articleids.append({"idtype": "doi", "value": doi})
    return {
        "uid": pmid,
        "sortpubdate": f"{publication_date.replace('-', '/')} 00:00",
        "articleids": articleids,
    }


def _fixture(
    tmp_path: Path,
    interactions: list[dict[str, object]],
    *,
    pubmed_records: dict[str, dict[str, object]],
    missing_pmids: list[str] | None = None,
    mapping_uniprot: str = "P18054",
) -> dict[str, Path]:
    paths = {
        "interactions": tmp_path / "interactions.csv",
        "ligands": tmp_path / "ligands.csv",
        "mapping": tmp_path / "GtP_to_UniProt_mapping.csv",
        "pubmed": tmp_path / "pubmed_esummary.json",
        "source_manifest": tmp_path / "source_manifest.json",
        "out": tmp_path / "out",
    }
    _write_release_csv(
        paths["interactions"],
        EXPECTED_HEADERS["interactions.csv"],
        interactions,
    )
    ligand_ids = sorted({str(row.get("Ligand ID") or "") for row in interactions if row.get("Ligand ID")})
    _write_release_csv(
        paths["ligands"],
        EXPECTED_HEADERS["ligands.csv"],
        [_ligand(ligand_id) for ligand_id in ligand_ids],
    )
    _write_release_csv(
        paths["mapping"],
        EXPECTED_HEADERS["GtP_to_UniProt_mapping.csv"],
        [
            {
                "UniProtKB ID": mapping_uniprot,
                "Species": "Human",
                "GtoPdb IUPHAR Name": "12S-LOX",
                "GtoPdb IUPHAR ID": "1387",
                "GtP URL": "https://www.guidetopharmacology.org/GRAC/ObjectDisplayForward?objectId=1387",
            }
        ],
    )
    all_pmids = sorted(
        {
            token
            for row in interactions
            for token in str(row.get("PubMed ID") or "").split("|")
            if token
        },
        key=int,
    )
    missing = missing_pmids or []
    cache = {
        "schema_version": PUBMED_SCHEMA_VERSION,
        "source": {"name": "NCBI PubMed"},
        "query": {
            "requested_count": len(all_pmids),
            "requested_pmids_sha256": _hash_values(all_pmids),
            "batch_size": 200,
            "batches": [],
        },
        "records": pubmed_records,
        "missing_pmids": missing,
    }
    paths["pubmed"].write_text(json.dumps(cache), encoding="utf-8")
    source_manifest = {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "source": {
            "name": "GtoPdb",
            "release": "2026.2",
            "release_date": "2026-06-15",
            "license": SOURCE_LICENSE,
            "license_url": SOURCE_LICENSE_URL,
        },
        "artifacts": {},
    }
    for name, key, rows in (
        ("interactions.csv", "interactions", len(interactions)),
        ("ligands.csv", "ligands", len(ligand_ids)),
        ("GtP_to_UniProt_mapping.csv", "mapping", 1),
    ):
        path = paths[key]
        source_manifest["artifacts"][name] = {
            "path": path.name,
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
            "rows": rows,
        }
    source_manifest["artifacts"]["pubmed_esummary.json"] = {
        "path": paths["pubmed"].name,
        "sha256": _sha256(paths["pubmed"]),
        "bytes": paths["pubmed"].stat().st_size,
    }
    paths["source_manifest"].write_text(json.dumps(source_manifest), encoding="utf-8")
    return paths


def _run(paths: dict[str, Path]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--interactions-csv",
            str(paths["interactions"]),
            "--ligands-csv",
            str(paths["ligands"]),
            "--target-mapping-csv",
            str(paths["mapping"]),
            "--pubmed-json",
            str(paths["pubmed"]),
            "--source-manifest",
            str(paths["source_manifest"]),
            "--required-release",
            "2026.2",
            "--cutoff-date",
            "2023-10-01",
            "--out-dir",
            str(paths["out"]),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_builder_uses_latest_supporting_pmid_as_conservative_temporal_anchor(
    tmp_path: Path,
) -> None:
    paths = _fixture(
        tmp_path,
        [
            _interaction(),
            _interaction(
                **{
                    "Original Affinity Units": "EC50",
                    "Affinity Units": "pEC50",
                    "Original Affinity Median nm": "100",
                    "Affinity Median": "7",
                    "PubMed ID": "100|200",
                }
            ),
        ],
        pubmed_records={
            "100": _pubmed_record("100", "2020-02-03", "10.1000/old"),
            "200": _pubmed_record("200", "2025-04-05", "10.1000/new"),
        },
    )

    result = _run(paths)

    assert result.returncode == 0, result.stderr
    activity = pd.read_parquet(paths["out"] / "activity_evidence.parquet")
    assert activity.shape[0] == 2
    assert activity["source_db"].unique().tolist() == ["GtoPdb"]
    assert activity["source_release"].unique().tolist() == ["2026.2"]
    assert activity["source_license"].unique().tolist() == [SOURCE_LICENSE]
    post = activity.loc[activity["affinity_type"] == "EC50"].iloc[0]
    assert post["source_pmid"] == "200"
    assert post["source_doi"] == "10.1000/new"
    assert post["supporting_pmids"] == "100|200"
    assert post["publication_date"] == "2025-04-05"
    assert post["temporal_split"] == "post_cutoff"
    assert pd.read_parquet(paths["out"] / "pre_cutoff.parquet").shape[0] == 1
    assert pd.read_parquet(paths["out"] / "post_cutoff.parquet").shape[0] == 1
    manifest = json.loads((paths["out"] / "manifest.json").read_text())
    assert manifest["row_counts"] == {
        "activity_evidence": 2,
        "post_cutoff": 1,
        "pre_cutoff": 1,
    }
    assert manifest["artifacts"]["activity_evidence"]["sha256"]
    assert manifest["artifacts"]["activity_evidence"]["path"] == (
        "activity_evidence.parquet"
    )
    assert manifest["inputs"]["source_manifest"]["path"] == "../source_manifest.json"
    assert "latest supporting publication" in manifest["policy"]["publication"]


def test_builder_filters_mapping_unit_relation_and_unresolved_publication(
    tmp_path: Path,
) -> None:
    paths = _fixture(
        tmp_path,
        [
            _interaction(),
            _interaction(**{"Affinity Units": "pKi"}),
            _interaction(**{"Original Affinity Relation": ">"}),
            _interaction(**{"PubMed ID": "300"}),
        ],
        pubmed_records={
            "100": _pubmed_record("100", "2020-02-03"),
            "300": {"uid": "300", "sortpubdate": "", "articleids": []},
        },
    )

    result = _run(paths)

    assert result.returncode == 0, result.stderr
    activity = pd.read_parquet(paths["out"] / "activity_evidence.parquet")
    assert activity.shape[0] == 1
    manifest = json.loads((paths["out"] / "manifest.json").read_text())
    assert manifest["counts"]["filtered_unsupported_or_inconsistent_endpoint"] == 1
    assert manifest["counts"]["filtered_non_exact_relation"] == 1
    assert manifest["counts"]["filtered_unresolved_publication_date"] == 1


def test_builder_fails_closed_on_target_mapping_mismatch_and_removes_stale_outputs(
    tmp_path: Path,
) -> None:
    paths = _fixture(
        tmp_path,
        [_interaction()],
        pubmed_records={"100": _pubmed_record("100", "2020-02-03")},
        mapping_uniprot="P18055",
    )
    paths["out"].mkdir()
    for name in ("activity_evidence.parquet", "pre_cutoff.parquet", "post_cutoff.parquet", "manifest.json"):
        (paths["out"] / name).write_text("stale\n")

    result = _run(paths)

    assert result.returncode != 0
    assert "No claim-grade GtoPdb activity rows" in result.stderr
    assert not any(paths["out"].iterdir())


def test_builder_rejects_source_manifest_hash_mismatch(tmp_path: Path) -> None:
    paths = _fixture(
        tmp_path,
        [_interaction()],
        pubmed_records={"100": _pubmed_record("100", "2020-02-03")},
    )
    paths["ligands"].write_text(paths["ligands"].read_text() + "\n", encoding="utf-8")

    result = _run(paths)

    assert result.returncode != 0
    assert "source manifest sha256 mismatch for ligands.csv" in result.stderr
    assert not paths["out"].exists() or not any(paths["out"].iterdir())
