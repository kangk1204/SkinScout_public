from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from eval.build_evidence_target_sequence_universe import _accepted_targets


pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "eval" / "build_evidence_target_sequence_universe.py"
UNIPROT_COLUMNS = (
    "Entry",
    "Entry Name",
    "Reviewed",
    "Organism (ID)",
    "Organism",
    "Length",
    "Sequence",
    "Sequence version",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_accessions(accessions: list[str]) -> str:
    payload = "\n".join(sorted(set(accessions))) + "\n"
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _chembl_row(accession: str, activity_id: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "source_name": "ChEMBL",
        "source_release": "37",
        "source_license": "CC BY-SA 3.0",
        "activity_id": activity_id,
        "assay_type": "B",
        "target_type": "SINGLE PROTEIN",
        "assay_confidence_score": 9,
        "assay_relationship_type": "D",
        "standard_relation": "=",
        "standard_type": "IC50",
        "standard_value": 100.0,
        "standard_units": "nM",
        "data_validity_comment": "",
        "potential_duplicate": False,
        "document_type": "PUBLICATION",
        "document_year": 2023,
        "document_chembl_id": f"DOC_{activity_id}",
        "document_source_id": 1,
        "document_source_name": "ChEMBL",
        "assay_source_id": 1,
        "assay_source_name": "ChEMBL",
        "assay_variant_id": None,
        "assay_variant_accession": "",
        "assay_variant_mutation": "",
        "uniprot": accession,
        "molecule_chembl_id": f"CHEMBL_{activity_id}",
        "smiles": "CCO",
    }
    row.update(overrides)
    return row


def _bindingdb_row(accession: str, evidence_id: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "uniprot": accession,
        "evidence_id": evidence_id,
        "ligand_smiles": "CCO",
        "affinity_type": "Kd",
        "relation": "=",
        "affinity_value": 100.0,
        "affinity_unit": "nM",
        "publication_date": "2025-01-01",
        "evidence_date_source": "publication",
        "single_chain_target": True,
        "source_origin": "BindingDB",
        "source_license": "CC BY 4.0",
        "source_release": "2026-08",
        "chembl_derived_license_flag": False,
        "source_doi": "10.1000/bindingdb-test",
        "source_pmid": "",
        "source_patent": "",
        "source_article_id": "",
    }
    row.update(overrides)
    return row


def _source_manifest(evidence: Path) -> Path:
    path = evidence.with_suffix(".manifest.json")
    payload = {
        "source": {"name": "ChEMBL", "release": "37", "license": "CC BY-SA 3.0"},
        "output_sha256": {"activity_evidence.parquet": _sha256(evidence)},
        "row_counts": {"activity_evidence": len(pd.read_parquet(evidence))},
    }
    path.write_text(json.dumps(payload) + "\n")
    return path


def _uniprot_cache(
    tmp_path: Path,
    requested: list[str],
    rows: list[list[object]],
) -> tuple[Path, Path]:
    tsv = tmp_path / "uniprot.tsv"
    lines = ["\t".join(UNIPROT_COLUMNS)]
    lines.extend("\t".join("" if value is None else str(value) for value in row) for row in rows)
    tsv.write_text("\n".join(lines) + "\n")
    metadata = tmp_path / "uniprot.metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "schema_version": "skinscout.uniprot-search-cache.v1",
                "source": {
                    "name": "UniProtKB",
                    "endpoint": "https://rest.uniprot.org/uniprotkb/search",
                    "release": "2026_02",
                    "release_date": "10-June-2026",
                    "license": "CC BY 4.0",
                    "license_url": "https://www.uniprot.org/help/license",
                },
                "query": {
                    "requested_count": len(requested),
                    "requested_accessions_sha256": _hash_accessions(requested),
                    "fields": [],
                    "batches": [],
                },
                "artifact": {
                    "path": str(tsv.resolve()),
                    "sha256": _sha256(tsv),
                    "rows": len(rows),
                },
            }
        )
        + "\n"
    )
    return tsv, metadata


def _run(
    tmp_path: Path,
    *,
    rows: list[list[object]],
    mutate_metadata=None,
) -> subprocess.CompletedProcess[str]:
    base = tmp_path / "base.fasta"
    base.write_text(">PBASE\n" + "A" * 40 + "\n")
    evidence = tmp_path / "chembl.parquet"
    pd.DataFrame(
        [
            _chembl_row("PBASE", "base"),
            _chembl_row("PSUPP", "supp"),
            _chembl_row("PNONHUMAN", "nonhuman"),
            _chembl_row("POBSOLETE", "obsolete"),
            _chembl_row("PFILTERED", "filtered", assay_type="F"),
        ]
    ).to_parquet(evidence, index=False)
    requested = ["PBASE", "PNONHUMAN", "POBSOLETE", "PSUPP"]
    tsv, metadata = _uniprot_cache(tmp_path, requested, rows)
    if mutate_metadata is not None:
        payload = json.loads(metadata.read_text())
        mutate_metadata(payload)
        metadata.write_text(json.dumps(payload) + "\n")
    command = [
        sys.executable,
        str(SCRIPT),
        "--base-fasta",
        str(base),
        "--chembl-evidence",
        str(evidence),
        "--chembl-manifest",
        str(_source_manifest(evidence)),
        "--required-uniprot-release",
        "2026_02",
        "--uniprot-tsv",
        str(tsv),
        "--uniprot-metadata",
        str(metadata),
        "--offline",
        "--out-fasta",
        str(tmp_path / "supplemental.fasta"),
        "--out-exclusions",
        str(tmp_path / "exclusions.csv"),
        "--out-manifest",
        str(tmp_path / "manifest.json"),
    ]
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)


def _valid_rows() -> list[list[object]]:
    return [
        [
            "PBASE",
            "PBASE_HUMAN",
            "reviewed",
            "9606",
            "Homo sapiens (Human)",
            40,
            "A" * 40,
            1,
        ],
        [
            "PNONHUMAN",
            "PNONHUMAN_PONAB",
            "reviewed",
            "9601",
            "Pongo abelii",
            40,
            "C" * 40,
            1,
        ],
        ["POBSOLETE", "POBSOLETE_HUMAN", "", "", "", "", "", ""],
        [
            "PSUPP",
            "PSUPP_HUMAN",
            "reviewed",
            "9606",
            "Homo sapiens (Human)",
            40,
            "G" * 40,
            2,
        ],
    ]


def test_builds_human_supplement_and_audits_exclusions(tmp_path: Path) -> None:
    result = _run(tmp_path, rows=_valid_rows())

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "supplemental.fasta").read_text() == (
        ">PBASE\n" + "A" * 40 + "\n>PSUPP\n" + "G" * 40 + "\n"
    )
    exclusions = pd.read_csv(tmp_path / "exclusions.csv")
    assert exclusions[["uniprot", "reason"]].to_dict("records") == [
        {"uniprot": "PNONHUMAN", "reason": "taxonomy_mismatch"},
        {
            "uniprot": "POBSOLETE",
            "reason": "unresolved_or_obsolete_uniprot_record",
        },
    ]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["evidence_targets"]["total_count"] == 4
    assert manifest["schema_version"] == "skinscout.evidence-target-sequence-universe.v2"
    assert manifest["evidence_targets"]["present_in_structure_fasta_count"] == 1
    assert manifest["evidence_targets"]["absent_from_structure_fasta_count"] == 3
    assert manifest["artifacts"]["target_fasta"]["accepted_accessions"] == [
        "PBASE",
        "PSUPP",
    ]
    assert manifest["exclusions"]["count"] == 2
    assert manifest["inputs"]["uniprot_metadata"]["source"]["release"] == "2026_02"


def test_target_extraction_keeps_ligand_structure_quality_columns(tmp_path: Path) -> None:
    chembl = tmp_path / "chembl.parquet"
    pd.DataFrame(
        [
            _chembl_row("PCHEMBL", "valid"),
            _chembl_row("PBLANK", "blank", smiles=""),
        ]
    ).to_parquet(chembl, index=False)
    bindingdb = tmp_path / "bindingdb.parquet"
    pd.DataFrame(
        [
            _bindingdb_row("PBIND", "valid"),
            _bindingdb_row("PBLANK", "blank", ligand_smiles=""),
        ]
    ).to_parquet(bindingdb, index=False)
    gtopdb = tmp_path / "gtopdb.parquet"
    pd.DataFrame(
        [
            _bindingdb_row(
                "PGTOP",
                "valid",
                source_origin="GtoPdb expert-curated literature",
                source_license="ODbL 1.0; contents CC BY-SA 4.0",
                source_release="2026.2",
            ),
            _bindingdb_row(
                "PBLANK",
                "blank",
                ligand_smiles="",
                source_origin="GtoPdb expert-curated literature",
                source_license="ODbL 1.0; contents CC BY-SA 4.0",
                source_release="2026.2",
            ),
        ]
    ).to_parquet(gtopdb, index=False)

    chembl_targets, chembl_rows = _accepted_targets(
        chembl,
        source="chembl",
        exclude_bindingdb_sources=False,
    )
    bindingdb_targets, bindingdb_rows = _accepted_targets(
        bindingdb,
        source="bindingdb",
        exclude_bindingdb_sources=False,
    )
    gtopdb_targets, gtopdb_rows = _accepted_targets(
        gtopdb,
        source="gtopdb",
        exclude_bindingdb_sources=False,
    )

    assert chembl_targets == {"PCHEMBL"}
    assert chembl_rows == 1
    assert bindingdb_targets == {"PBIND"}
    assert bindingdb_rows == 1
    assert gtopdb_targets == {"PGTOP"}
    assert gtopdb_rows == 1


def test_cache_hash_mismatch_fails_closed_and_removes_outputs(tmp_path: Path) -> None:
    for name in ("supplemental.fasta", "exclusions.csv", "manifest.json"):
        (tmp_path / name).write_text("stale\n")

    result = _run(
        tmp_path,
        rows=_valid_rows(),
        mutate_metadata=lambda payload: payload["artifact"].update({"sha256": "0" * 64}),
    )

    assert result.returncode != 0
    assert "sha256 does not match" in result.stderr
    assert not (tmp_path / "supplemental.fasta").exists()
    assert not (tmp_path / "exclusions.csv").exists()
    assert not (tmp_path / "manifest.json").exists()


def test_sequence_length_mismatch_is_not_silently_excluded(tmp_path: Path) -> None:
    rows = _valid_rows()
    rows[-1][5] = 41

    result = _run(tmp_path, rows=rows)

    assert result.returncode != 0
    assert "sequence length mismatch for PSUPP" in result.stderr
    assert not (tmp_path / "manifest.json").exists()
