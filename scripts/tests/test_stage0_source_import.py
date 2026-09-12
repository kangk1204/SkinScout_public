"""Regression tests for Stage 0 source import and manifesting."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import import_stage0_sources as importer  # noqa: E402


COSING_LABEL = "Stage 0 source: CosIng CSV"
DRUGBANK_LABEL = "Stage 0 source: DrugBank full database XML"
PROTEOME_LABEL = "Stage 0 source: skin proteome LFQ TSV"
GTEX_LABEL = "Stage 0 source: GTEx gene TPM GCT"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_valid_sources(source_dir: Path) -> None:
    source_dir.mkdir(parents=True)
    (source_dir / "cosing.csv").write_text(
        "INCI name,CAS,Function\nWater,7732-18-5,Solvent\n"
    )
    (source_dir / "drugbank_full_database.xml").write_text(
        "<?xml version='1.0' encoding='UTF-8'?>\n"
        "<drugbank><drug><drugbank-id>DB00001</drugbank-id></drug></drugbank>\n"
    )
    (source_dir / "raw_lfq.tsv").write_text(
        "uniprot\tLFQ sample\nP12345\t10.0\n"
    )
    (source_dir / "gtex_v10_gene_tpm.gct").write_text(
        "#1.2\n"
        "1\t3\n"
        "Name\tDescription\tGTEX-1 Skin - Sun Exposed Lower leg\n"
        "ENSG000001\tGENE1\t5.0\n"
    )


def test_import_stage0_sources_copies_valid_files_and_writes_manifest(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "incoming"
    write_valid_sources(source_dir)
    target_root = tmp_path / "repo" / "data"
    manifest = target_root / "manifests" / "stage0_source_manifest.json"
    required = [
        (COSING_LABEL, target_root / "cosing" / "cosing.csv"),
        (DRUGBANK_LABEL, target_root / "drugbank" / "drugbank_full_database.xml"),
        (PROTEOME_LABEL, target_root / "skin_proteome" / "raw_lfq.tsv"),
        (GTEX_LABEL, target_root / "gtex_v10" / "gtex_v10_gene_tpm.gct"),
    ]

    payload = importer.import_stage0_sources(
        required_sources=required,
        source_dir=source_dir,
        source_paths={},
        manifest_path=manifest,
        mode="copy",
    )

    assert payload["status"] == "ok"
    assert payload["imported_source_count"] == 4
    assert (target_root / "cosing" / "cosing.csv").exists()
    assert (target_root / "drugbank" / "drugbank_full_database.xml").exists()
    assert json.loads(manifest.read_text()) == payload
    cosing_entry = {
        entry["key"]: entry for entry in payload["sources"]
    }["cosing_csv"]
    assert cosing_entry["target_sha256"] == sha256(
        target_root / "cosing" / "cosing.csv"
    )
    assert cosing_entry["detail"].startswith("CosIng CSV header ok")


def test_import_stage0_sources_extracts_zip_and_gzip_inputs(tmp_path: Path) -> None:
    source_dir = tmp_path / "incoming"
    source_dir.mkdir()
    drugbank_zip = source_dir / "drugbank_full_database.xml.zip"
    with zipfile.ZipFile(drugbank_zip, "w") as archive:
        archive.writestr(
            "drugbank_full_database.xml",
            "<drugbank><drug><drugbank-id>DB00001</drugbank-id></drug></drugbank>\n",
        )
    gtex_text = (
        "#1.2\n"
        "1\t3\n"
        "Name\tDescription\tGTEX-1 Skin - Not Sun Exposed Suprapubic\n"
        "ENSG000001\tGENE1\t7.0\n"
    )
    with gzip.open(source_dir / "gtex_v10_gene_tpm.gct.gz", "wt") as handle:
        handle.write(gtex_text)
    target_root = tmp_path / "repo" / "data"
    required = [
        (DRUGBANK_LABEL, target_root / "drugbank" / "drugbank_full_database.xml"),
        (GTEX_LABEL, target_root / "gtex_v10" / "gtex_v10_gene_tpm.gct"),
    ]

    payload = importer.import_stage0_sources(
        required_sources=required,
        source_dir=source_dir,
        source_paths={},
        manifest_path=target_root / "manifests" / "sources.json",
        mode="copy",
    )

    assert payload["status"] == "ok"
    actions = {entry["key"]: entry["action"] for entry in payload["sources"]}
    assert actions == {"drugbank_xml": "extract", "gtex_gct": "decompress"}
    assert (target_root / "gtex_v10" / "gtex_v10_gene_tpm.gct").read_text() == gtex_text


def test_import_stage0_sources_discovers_common_download_names_recursively(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "incoming"
    source_dir.mkdir()
    (source_dir / "eu").mkdir()
    (source_dir / "eu" / "CosIng - Glossary of Ingredients.csv").write_text(
        "INCI name,CAS,Function\nWater,7732-18-5,Solvent\n"
    )
    (source_dir / "drugbank").mkdir()
    with zipfile.ZipFile(
        source_dir / "drugbank" / "drugbank_all_full_database.xml.zip",
        "w",
    ) as archive:
        archive.writestr(
            "full database.xml",
            "<drugbank><drug><drugbank-id>DB00001</drugbank-id></drug></drugbank>\n",
        )
    (source_dir / "proteome").mkdir()
    with gzip.open(source_dir / "proteome" / "study_skin_proteome_lfq.tsv.gz", "wt") as handle:
        handle.write("uniprot\tLFQ sample\nP12345\t10.0\n")
    (source_dir / "gtex").mkdir()
    gtex_text = (
        "#1.2\n"
        "1\t3\n"
        "Name\tDescription\tGTEX-1 Skin - Sun Exposed Lower leg\n"
        "ENSG000001\tGENE1\t6.0\n"
    )
    with gzip.open(source_dir / "gtex" / "GTEx_v10_skin_gene_tpm.gct.gz", "wt") as handle:
        handle.write(gtex_text)

    target_root = tmp_path / "repo" / "data"
    required = [
        (COSING_LABEL, target_root / "cosing" / "cosing.csv"),
        (DRUGBANK_LABEL, target_root / "drugbank" / "drugbank_full_database.xml"),
        (PROTEOME_LABEL, target_root / "skin_proteome" / "raw_lfq.tsv"),
        (GTEX_LABEL, target_root / "gtex_v10" / "gtex_v10_gene_tpm.gct"),
    ]

    payload = importer.import_stage0_sources(
        required_sources=required,
        source_dir=source_dir,
        source_paths={},
        manifest_path=target_root / "manifests" / "sources.json",
        mode="copy",
    )

    assert payload["status"] == "ok"
    actions = {entry["key"]: entry["action"] for entry in payload["sources"]}
    assert actions == {
        "cosing_csv": "copy",
        "drugbank_xml": "extract",
        "skin_proteome_tsv": "decompress",
        "gtex_gct": "decompress",
    }
    assert (target_root / "skin_proteome" / "raw_lfq.tsv").read_text().startswith(
        "uniprot\tLFQ"
    )
    assert (target_root / "gtex_v10" / "gtex_v10_gene_tpm.gct").read_text() == gtex_text


def test_import_stage0_sources_reports_ambiguous_recursive_candidates(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "incoming"
    source_dir.mkdir()
    for name in ("cosing_export_a.csv", "cosing_export_b.csv"):
        (source_dir / name).write_text(
            "INCI name,CAS,Function\nWater,7732-18-5,Solvent\n"
        )
    target = tmp_path / "repo" / "data" / "cosing" / "cosing.csv"

    payload = importer.import_stage0_sources(
        required_sources=[(COSING_LABEL, target)],
        source_dir=source_dir,
        source_paths={},
        manifest_path=tmp_path / "repo" / "data" / "manifests" / "sources.json",
        mode="copy",
    )

    assert payload["status"] == "failed"
    assert "multiple recursive source candidates found" in payload["errors"][0]
    assert "pass an explicit source path" in payload["errors"][0]
    assert not target.exists()


def test_import_stage0_sources_reports_missing_source_directory(tmp_path: Path) -> None:
    source_dir = tmp_path / "missing"
    target = tmp_path / "repo" / "data" / "cosing" / "cosing.csv"

    payload = importer.import_stage0_sources(
        required_sources=[(COSING_LABEL, target)],
        source_dir=source_dir,
        source_paths={},
        manifest_path=tmp_path / "repo" / "data" / "manifests" / "sources.json",
        mode="copy",
    )

    assert payload["status"] == "failed"
    assert f"--source-dir does not exist: {source_dir}" in payload["errors"][0]
    assert not target.exists()


def test_import_stage0_sources_installs_optional_drugbank_when_present(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_dir = tmp_path / "incoming"
    source_dir.mkdir()
    (source_dir / "cosing.csv").write_text(
        "INCI name,CAS,Function\nWater,7732-18-5,Solvent\n"
    )
    (source_dir / "drugbank_full_database.xml").write_text(
        "<?xml version='1.0' encoding='UTF-8'?>\n"
        "<drugbank><drug><drugbank-id>DB00001</drugbank-id></drug></drugbank>\n"
    )
    target_root = tmp_path / "repo" / "data"
    required = [(COSING_LABEL, target_root / "cosing" / "cosing.csv")]
    optional = [(DRUGBANK_LABEL, target_root / "drugbank" / "drugbank_full_database.xml")]
    monkeypatch.setattr(importer, "required_stage0_source_artifacts", lambda: required)
    monkeypatch.setattr(importer, "optional_stage0_source_artifacts", lambda: optional)

    payload = importer.import_stage0_sources(
        source_dir=source_dir,
        source_paths={},
        manifest_path=target_root / "manifests" / "sources.json",
        mode="copy",
    )

    assert payload["status"] == "ok"
    assert payload["required_source_count"] == 2
    assert {entry["key"] for entry in payload["sources"]} == {
        "cosing_csv",
        "drugbank_xml",
    }
    assert (target_root / "drugbank" / "drugbank_full_database.xml").exists()


def test_import_stage0_sources_rejects_malformed_source_without_target(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "incoming"
    source_dir.mkdir()
    (source_dir / "cosing.csv").write_text("wrong\nvalue\n")
    target = tmp_path / "repo" / "data" / "cosing" / "cosing.csv"

    payload = importer.import_stage0_sources(
        required_sources=[(COSING_LABEL, target)],
        source_dir=source_dir,
        source_paths={},
        manifest_path=tmp_path / "repo" / "data" / "manifests" / "sources.json",
        mode="copy",
    )

    assert payload["status"] == "failed"
    assert "INCI name column" in payload["errors"][0]
    assert not target.exists()


def test_import_stage0_sources_refuses_different_existing_target_without_force(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "incoming"
    source_dir.mkdir()
    (source_dir / "cosing.csv").write_text(
        "INCI name,CAS,Function\nWater,7732-18-5,Solvent\n"
    )
    target = tmp_path / "repo" / "data" / "cosing" / "cosing.csv"
    target.parent.mkdir(parents=True)
    target.write_text("INCI name,CAS,Function\nOld,1,Old\n")

    payload = importer.import_stage0_sources(
        required_sources=[(COSING_LABEL, target)],
        source_dir=source_dir,
        source_paths={},
        manifest_path=tmp_path / "repo" / "data" / "manifests" / "sources.json",
        mode="copy",
    )

    assert payload["status"] == "failed"
    assert "target exists with different content" in payload["errors"][0]
    assert target.read_text() == "INCI name,CAS,Function\nOld,1,Old\n"
