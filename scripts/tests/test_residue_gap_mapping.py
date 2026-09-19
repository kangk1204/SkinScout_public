"""Residue-gap mapping regressions (audit F18).

Removing unobserved residues from a PDB used to leave a sequence in which
residue 10 was immediately followed by residue 20. These tests pin the explicit
mapping that replaces that join, and the Stage 4 -> Stage 6 hash binding.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import residue_mapping as rm  # noqa: E402
import stage4_prepare_structures as stage4  # noqa: E402
import stage6_bioemu as stage6  # noqa: E402


def atom_line(
    serial: int,
    *,
    chain: str,
    number: int,
    insertion_code: str = "",
    resname: str = "GLY",
    atom: str = "CA",
    x: float = 0.0,
) -> str:
    return (
        f"ATOM  {serial:5d}  {atom:<3s} {resname:>3s} {chain}"
        f"{number:4d}{insertion_code:1s}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 10.00           C"
    )


def write_pdb(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\nEND\n")
    return path


def continuous_pdb(path: Path, *, chain: str = "A", start: int = 1, n: int = 5) -> Path:
    return write_pdb(
        path,
        [
            atom_line(index, chain=chain, number=number)
            for index, number in enumerate(range(start, start + n), start=1)
        ],
    )


def gapped_pdb(path: Path) -> Path:
    return write_pdb(
        path,
        [
            atom_line(1, chain="A", number=10),
            atom_line(2, chain="A", number=20),
        ],
    )


def mapping_record(pdb: Path, *, target_id: str = "P00001") -> dict:
    return rm.build_mapping_record(
        pdb,
        target_id=target_id,
        source="alphafold_cleaned",
        canonical_sequence="GGGGGGGGGGGGGGGGGGGG",
    )


def test_residue_10_then_20_is_a_gap_and_never_a_join(tmp_path: Path) -> None:
    pdb = gapped_pdb(tmp_path / "P00001_input.pdb")

    mapping = rm.pdb_residue_mapping(pdb)

    assert mapping.continuous is False
    assert mapping.sequence == "GG"
    assert len(mapping.gaps) == 1
    gap = mapping.gaps[0]
    assert gap.kind == rm.GAP_RESIDUE_NUMBER
    assert (gap.before.number, gap.after.number) == (10, 20)
    assert gap.missing_residues == 9
    assert "discontinuous" in (mapping.rejection_reason() or "")

    with pytest.raises(rm.ResidueGapError, match="residue 10"):
        rm.pdb_sequence(pdb)

    # The join is only available through the explicit opt-in used by callers
    # that carry their own mapping (for example the pocket crop).
    assert rm.concatenated_pdb_sequence(pdb) == "GG"


def test_chain_boundary_is_not_a_join_either(tmp_path: Path) -> None:
    pdb = write_pdb(
        tmp_path / "two_chains.pdb",
        [
            atom_line(1, chain="A", number=1),
            atom_line(2, chain="B", number=1),
        ],
    )

    mapping = rm.pdb_residue_mapping(pdb)

    assert mapping.continuous is False
    assert mapping.gaps[0].kind == rm.GAP_CHAIN_BOUNDARY
    with pytest.raises(rm.ResidueGapError):
        rm.pdb_sequence(pdb)


def test_clean_continuous_control_is_unchanged(tmp_path: Path) -> None:
    pdb = continuous_pdb(tmp_path / "clean.pdb")

    mapping = rm.pdb_residue_mapping(pdb)

    assert mapping.continuous is True
    assert mapping.gaps == ()
    assert len(mapping.spans) == 1
    assert rm.pdb_sequence(pdb) == "GGGGG"
    assert rm.concatenated_pdb_sequence(pdb) == "GGGGG"
    assert mapping.rejection_reason() is None


def test_insertion_codes_are_adjacency_not_a_gap(tmp_path: Path) -> None:
    pdb = write_pdb(
        tmp_path / "insertions.pdb",
        [
            atom_line(1, chain="A", number=10),
            atom_line(2, chain="A", number=10, insertion_code="A"),
            atom_line(3, chain="A", number=10, insertion_code="B"),
            atom_line(4, chain="A", number=11),
        ],
    )

    mapping = rm.pdb_residue_mapping(pdb)

    assert mapping.continuous is True
    assert rm.pdb_sequence(pdb) == "GGGG"


def test_stage4_records_the_shared_mapping_stage6_consumes(tmp_path: Path) -> None:
    """The mapping record written in Stage 4 is what Stage 6 validates."""
    pdb = continuous_pdb(tmp_path / "P00001_input.pdb", n=5)
    record_path = rm.mapping_path_for_pdb(pdb)
    record = stage4.build_mapping_record(
        pdb,
        target_id="P00001",
        source="alphafold_cleaned",
        canonical_sequence="GGGGG",
    )
    stage4.write_mapping_record(pdb, record, record_path)

    sequence, rejection = stage6.sampling_sequence(pdb)

    assert rejection is None
    assert sequence == "GGGGG"
    assert record["observed_sequence_sha256"] == rm._sha256_text("GGGGG")
    assert record["canonical_sequence_sha256"] == rm._sha256_text("GGGGG")
    assert record["input_pdb_sha256"] == rm._sha256_file(pdb)
    assert rm.validate_mapping_record(pdb, record) == []

    # A changed structure on the same path no longer matches the record's hash.
    write_pdb(
        pdb,
        [
            atom_line(1, chain="A", number=1, resname="ALA"),
            atom_line(2, chain="A", number=2),
            atom_line(3, chain="A", number=3),
            atom_line(4, chain="A", number=4),
            atom_line(5, chain="A", number=5),
        ],
    )
    sequence, rejection = stage6.sampling_sequence(pdb)
    assert sequence is None
    assert "hash" in (rejection or "")


def test_stage4_mapping_of_a_gapped_structure_blocks_bioemu(tmp_path: Path) -> None:
    pdb = gapped_pdb(tmp_path / "P00002_input.pdb")
    record_path = rm.mapping_path_for_pdb(pdb)
    record = mapping_record(pdb, target_id="P00002")
    rm.write_mapping_record(pdb, record, record_path)

    assert record["observed_continuous"] is False
    assert record["sampling_allowed"] is False
    sequence, rejection = stage6.sampling_sequence(pdb)
    assert sequence is None
    assert "discontinuous" in (rejection or "")
