"""Regression tests for checksum-validated AFDB canonical sequence assembly."""

from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from Bio.SeqUtils.CheckSum import crc64


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "build_canonical_sequences.py"
SPEC = importlib.util.spec_from_file_location("canonical_sequences_under_test", SCRIPT)
assert SPEC and SPEC.loader
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def _checksum(sequence: str) -> str:
    return crc64(sequence).removeprefix("CRC-")


def _write_fragment(
    directory: Path,
    accession: str,
    fragment_number: int,
    sequence: str,
    begin: int,
    full_checksum: str,
    *,
    taxonomy: str = "9606",
) -> Path:
    end = begin + len(sequence) - 1
    path = directory / f"AF-{accession}-F{fragment_number}-model_v4.cif.gz"
    text = f"""data_AF-{accession}-F{fragment_number}
#
_entity_poly.pdbx_seq_one_letter_code_can
;{sequence}
;
#
_ma_target_ref_db_details.db_accession {accession}
_ma_target_ref_db_details.ncbi_taxonomy_id {taxonomy}
_ma_target_ref_db_details.seq_db_align_begin {begin}
_ma_target_ref_db_details.seq_db_align_end {end}
_ma_target_ref_db_details.seq_db_sequence_checksum {full_checksum}
#
_atom_site.group_PDB
ATOM coordinate rows must never be parsed
"""
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(text)
    return path


def _build(tmp_path: Path, *, minimum: int = 1, receptor_dir: Path | None = None):
    return builder.build(
        tmp_path / "afdb",
        tmp_path / "canonical.fasta",
        tmp_path / "canonical.manifest.json",
        minimum,
        receptor_dir,
    )


def test_build_assembles_overlapping_fragments_and_records_provenance(tmp_path: Path) -> None:
    source = tmp_path / "afdb"
    source.mkdir()
    full = "ACDEFGHIK"
    first = _write_fragment(source, "P12345", 1, full[:6], 1, _checksum(full))
    second = _write_fragment(source, "P12345", 2, full[4:], 5, _checksum(full))
    receptors = tmp_path / "clean"
    receptors.mkdir()
    (receptors / "P12345_clean.pdb").write_text("MODEL\n")

    manifest = _build(tmp_path, receptor_dir=receptors)

    assert (tmp_path / "canonical.fasta").read_text() == f">P12345\n{full}\n"
    assert manifest["artifact"]["sha256"] == builder._sha256(tmp_path / "canonical.fasta")
    assert manifest["counts"] == {
        "source_files": 2,
        "canonical_sequences": 1,
        "required_receptors": 1,
        "required_receptors_covered": 1,
    }
    record = manifest["sequences"][0]
    assert record["sequence_sha256"] == builder._sha256_bytes(full.encode("ascii"))
    assert [item["compressed_sha256"] for item in record["source_fragments"]] == [
        builder._sha256(first),
        builder._sha256(second),
    ]
    saved = json.loads((tmp_path / "canonical.manifest.json").read_text())
    assert "not an independently downloaded UniProt FASTA" in saved["provenance"]["derivation"]


def test_rejects_omitted_terminal_fragment_using_full_crc64(tmp_path: Path) -> None:
    source = tmp_path / "afdb"
    source.mkdir()
    full = "ACDEFG"
    _write_fragment(source, "P12345", 1, full[:4], 1, _checksum(full))

    with pytest.raises(builder.CanonicalSequenceError, match="may be incomplete or mutated"):
        _build(tmp_path)


def test_rejects_internal_fragment_gap(tmp_path: Path) -> None:
    source = tmp_path / "afdb"
    source.mkdir()
    full = "ACDEFG"
    checksum = _checksum(full)
    _write_fragment(source, "P12345", 1, "AC", 1, checksum)
    _write_fragment(source, "P12345", 2, "FG", 5, checksum)

    with pytest.raises(builder.CanonicalSequenceError, match="internal fragment gap"):
        _build(tmp_path)


def test_rejects_conflicting_overlap_mutation(tmp_path: Path) -> None:
    source = tmp_path / "afdb"
    source.mkdir()
    full = "ACDEFG"
    checksum = _checksum(full)
    _write_fragment(source, "P12345", 1, "ACDE", 1, checksum)
    _write_fragment(source, "P12345", 2, "XEFG".replace("X", "A"), 3, checksum)

    with pytest.raises(builder.CanonicalSequenceError, match="conflicting overlap"):
        _build(tmp_path)


def test_rejects_nonhuman_taxonomy(tmp_path: Path) -> None:
    source = tmp_path / "afdb"
    source.mkdir()
    full = "ACDE"
    _write_fragment(source, "P12345", 1, full, 1, _checksum(full), taxonomy="10090")

    with pytest.raises(builder.CanonicalSequenceError, match="expected taxonomy 9606"):
        _build(tmp_path)


def test_rejects_duplicate_accession_checksum_inconsistency(tmp_path: Path) -> None:
    source = tmp_path / "afdb"
    source.mkdir()
    _write_fragment(source, "P12345", 1, "ACDE", 1, _checksum("ACDEFG"))
    _write_fragment(source, "P12345", 2, "EFG", 4, _checksum("ACDEFA"))

    with pytest.raises(builder.CanonicalSequenceError, match="inconsistent full-sequence CRC64"):
        _build(tmp_path)


def test_rejects_filename_metadata_accession_inconsistency(tmp_path: Path) -> None:
    source = tmp_path / "afdb"
    source.mkdir()
    full = "ACDE"
    path = _write_fragment(source, "P12345", 1, full, 1, _checksum(full))
    mismatched = source / "AF-Q99999-F1-model_v4.cif.gz"
    path.rename(mismatched)

    with pytest.raises(builder.CanonicalSequenceError, match="does not match source filename"):
        _build(tmp_path)


def test_failure_preserves_existing_atomic_outputs(tmp_path: Path) -> None:
    source = tmp_path / "afdb"
    source.mkdir()
    full = "ACDE"
    _write_fragment(source, "P12345", 1, full, 1, _checksum(full))
    fasta = tmp_path / "canonical.fasta"
    manifest = tmp_path / "canonical.manifest.json"
    fasta.write_text("old fasta\n")
    manifest.write_text("old manifest\n")

    with pytest.raises(builder.CanonicalSequenceError, match="below required minimum"):
        _build(tmp_path, minimum=2)

    assert fasta.read_text() == "old fasta\n"
    assert manifest.read_text() == "old manifest\n"


def test_rejects_missing_required_cleaned_receptor(tmp_path: Path) -> None:
    source = tmp_path / "afdb"
    source.mkdir()
    full = "ACDE"
    _write_fragment(source, "P12345", 1, full, 1, _checksum(full))
    receptors = tmp_path / "clean"
    receptors.mkdir()
    (receptors / "Q99999_clean.pdb").write_text("MODEL\n")

    with pytest.raises(builder.CanonicalSequenceError, match="missing required receptors: Q99999"):
        _build(tmp_path, receptor_dir=receptors)
