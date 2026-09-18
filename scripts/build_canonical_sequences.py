#!/usr/bin/env python3
"""Build checksum-validated canonical human sequences from AlphaFold DB v4 mmCIFs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from Bio.SeqUtils.CheckSum import crc64


SCHEMA_VERSION = "skinscout.afdb-v4-canonical-sequences.v1"
HUMAN_TAXONOMY_ID = "9606"
KNOWN_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTUVWYO")
ACCESSION_RE = re.compile(r"^[A-Z0-9]+(?:-[0-9]+)?$")
SOURCE_NAME_RE = re.compile(
    r"^AF-(?P<accession>.+)-F(?P<fragment>[1-9][0-9]*)-model_v4\.cif\.gz$"
)
REQUIRED_TAGS = {
    "_entity_poly.pdbx_seq_one_letter_code_can": "sequence",
    "_ma_target_ref_db_details.db_accession": "accession",
    "_ma_target_ref_db_details.ncbi_taxonomy_id": "taxonomy_id",
    "_ma_target_ref_db_details.seq_db_align_begin": "begin",
    "_ma_target_ref_db_details.seq_db_align_end": "end",
    "_ma_target_ref_db_details.seq_db_sequence_checksum": "checksum",
}


class CanonicalSequenceError(ValueError):
    """Raised when AFDB source data cannot prove a canonical sequence."""


@dataclass(frozen=True)
class Fragment:
    accession: str
    taxonomy_id: str
    begin: int
    end: int
    sequence: str
    checksum: str
    source_path: Path
    source_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _clean_scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    return value


def _read_cif_value(initial: str, lines: Iterable[str]) -> str:
    value = initial.strip()
    if not value:
        for line in lines:
            if line.strip():
                value = line.rstrip("\r\n")
                break
        else:
            raise CanonicalSequenceError("missing mmCIF value at end of file")
    if value.startswith(";"):
        chunks = [value[1:]]
        for line in lines:
            if line.startswith(";"):
                return "".join(chunks)
            chunks.append(line.strip())
        raise CanonicalSequenceError("unterminated multiline mmCIF value")
    return _clean_scalar(value)


def parse_fragment(path: Path) -> Fragment:
    """Read only metadata preceding atom coordinates from one AFDB mmCIF gzip."""
    source_name = SOURCE_NAME_RE.fullmatch(path.name)
    if source_name is None:
        raise CanonicalSequenceError(f"not an AFDB v4 raw mmCIF filename: {path}")
    source_sha256 = _sha256(path)
    values: dict[str, str] = {}
    try:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            lines = iter(handle)
            for line in lines:
                if line.startswith("_atom_site."):
                    break
                if not line.startswith("_"):
                    continue
                fields = line.split(maxsplit=1)
                tag = fields[0]
                remainder = fields[1] if len(fields) == 2 else ""
                if tag not in REQUIRED_TAGS:
                    continue
                key = REQUIRED_TAGS[tag]
                if key in values:
                    raise CanonicalSequenceError(f"{path}: duplicate mmCIF tag {tag}")
                values[key] = _read_cif_value(remainder, lines)
    except (OSError, UnicodeError) as exc:
        raise CanonicalSequenceError(f"cannot read gzip mmCIF {path}: {exc}") from exc

    missing = sorted(set(REQUIRED_TAGS.values()) - values.keys())
    if missing:
        raise CanonicalSequenceError(f"{path}: missing required mmCIF fields: {','.join(missing)}")
    accession = values["accession"].strip().upper()
    if not ACCESSION_RE.fullmatch(accession):
        raise CanonicalSequenceError(f"{path}: invalid database accession {accession!r}")
    if accession != source_name.group("accession"):
        raise CanonicalSequenceError(
            f"{path}: metadata accession {accession} does not match source filename"
        )
    sequence = "".join(values["sequence"].split()).upper()
    invalid = sorted(set(sequence) - KNOWN_AMINO_ACIDS)
    if not sequence or invalid:
        shown = "".join(invalid) or "empty sequence"
        raise CanonicalSequenceError(f"{path}: sequence contains unknown residues: {shown}")
    try:
        begin = int(values["begin"])
        end = int(values["end"])
    except ValueError as exc:
        raise CanonicalSequenceError(f"{path}: fragment range is not integral") from exc
    if begin < 1 or end < begin or len(sequence) != end - begin + 1:
        raise CanonicalSequenceError(
            f"{path}: sequence length {len(sequence)} does not match range {begin}-{end}"
        )
    checksum = values["checksum"].strip().upper().removeprefix("CRC-")
    if not re.fullmatch(r"[0-9A-F]{16}", checksum):
        raise CanonicalSequenceError(f"{path}: invalid UniProt CRC64 checksum {checksum!r}")
    return Fragment(
        accession=accession,
        taxonomy_id=values["taxonomy_id"].strip(),
        begin=begin,
        end=end,
        sequence=sequence,
        checksum=checksum,
        source_path=path,
        source_sha256=source_sha256,
    )


def assemble_fragments(fragments: Iterable[Fragment]) -> tuple[str, str]:
    items = sorted(fragments, key=lambda item: (item.begin, item.end, item.source_path.name))
    if not items:
        raise CanonicalSequenceError("cannot assemble an empty fragment group")
    accession = items[0].accession
    checksums = {item.checksum for item in items}
    taxonomies = {item.taxonomy_id for item in items}
    accessions = {item.accession for item in items}
    if accessions != {accession}:
        raise CanonicalSequenceError(f"mixed accessions in fragment group for {accession}")
    if taxonomies != {HUMAN_TAXONOMY_ID}:
        raise CanonicalSequenceError(
            f"{accession}: expected taxonomy {HUMAN_TAXONOMY_ID}, found {sorted(taxonomies)}"
        )
    if len(checksums) != 1:
        raise CanonicalSequenceError(f"{accession}: inconsistent full-sequence CRC64 checksums")

    residues: dict[int, str] = {}
    for item in items:
        for position, residue in enumerate(item.sequence, start=item.begin):
            previous = residues.setdefault(position, residue)
            if previous != residue:
                raise CanonicalSequenceError(
                    f"{accession}: conflicting overlap at residue {position}"
                )
    if min(residues) != 1:
        raise CanonicalSequenceError(f"{accession}: fragment coverage does not begin at residue 1")
    expected_positions = set(range(1, max(residues) + 1))
    missing = expected_positions - residues.keys()
    if missing:
        first = min(missing)
        raise CanonicalSequenceError(f"{accession}: internal fragment gap begins at residue {first}")
    sequence = "".join(residues[position] for position in range(1, max(residues) + 1))
    actual_checksum = crc64(sequence).removeprefix("CRC-")
    expected_checksum = next(iter(checksums))
    if actual_checksum != expected_checksum:
        raise CanonicalSequenceError(
            f"{accession}: assembled CRC64 {actual_checksum} does not match declared "
            f"full-sequence checksum {expected_checksum}; fragments may be incomplete or mutated"
        )
    return sequence, expected_checksum


def _required_receptors(path: Path | None) -> list[str]:
    if path is None:
        return []
    if not path.is_dir():
        raise CanonicalSequenceError(f"required receptor directory does not exist: {path}")
    accessions = sorted(file.name[: -len("_clean.pdb")] for file in path.glob("*_clean.pdb"))
    if not accessions:
        raise CanonicalSequenceError(f"required receptor directory contains no *_clean.pdb files: {path}")
    if len(accessions) != len(set(accessions)):
        raise CanonicalSequenceError(f"duplicate cleaned receptor accessions in {path}")
    return accessions


def _fasta_bytes(sequences: dict[str, str]) -> bytes:
    lines: list[str] = []
    for accession in sorted(sequences):
        lines.append(f">{accession}")
        sequence = sequences[accession]
        lines.extend(sequence[offset : offset + 80] for offset in range(0, len(sequence), 80))
    return ("\n".join(lines) + "\n").encode("ascii")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def build(
    alphafold_dir: Path,
    out_fasta: Path,
    out_manifest: Path,
    min_sequences: int,
    required_receptor_dir: Path | None = None,
) -> dict[str, object]:
    if not alphafold_dir.is_dir():
        raise CanonicalSequenceError(f"AlphaFold directory does not exist: {alphafold_dir}")
    if min_sequences < 1:
        raise CanonicalSequenceError("--min-sequences must be at least 1")
    if out_fasta.resolve() == out_manifest.resolve():
        raise CanonicalSequenceError("FASTA and manifest outputs must be different paths")
    paths = sorted(alphafold_dir.glob("*.cif.gz"))
    if not paths:
        raise CanonicalSequenceError(f"no *.cif.gz sources found in {alphafold_dir}")

    grouped: dict[str, list[Fragment]] = {}
    for path in paths:
        fragment = parse_fragment(path)
        grouped.setdefault(fragment.accession, []).append(fragment)
    sequences: dict[str, str] = {}
    checksums: dict[str, str] = {}
    for accession, fragments in sorted(grouped.items()):
        sequences[accession], checksums[accession] = assemble_fragments(fragments)
    if len(sequences) < min_sequences:
        raise CanonicalSequenceError(
            f"assembled {len(sequences)} sequences, below required minimum {min_sequences}"
        )

    required = _required_receptors(required_receptor_dir)
    missing_required = sorted(set(required) - sequences.keys())
    if missing_required:
        shown = ",".join(missing_required[:20])
        suffix = "..." if len(missing_required) > 20 else ""
        raise CanonicalSequenceError(f"canonical sequences missing required receptors: {shown}{suffix}")

    fasta = _fasta_bytes(sequences)
    records = []
    for accession in sorted(sequences):
        fragments = sorted(
            grouped[accession], key=lambda item: (item.begin, item.end, item.source_path.name)
        )
        records.append(
            {
                "accession": accession,
                "length": len(sequences[accession]),
                "sequence_sha256": _sha256_bytes(sequences[accession].encode("ascii")),
                "uniprot_crc64": checksums[accession],
                "source_fragments": [
                    {
                        "path": item.source_path.name,
                        "compressed_sha256": item.source_sha256,
                        "seq_db_align_begin": item.begin,
                        "seq_db_align_end": item.end,
                    }
                    for item in fragments
                ],
            }
        )
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "provenance": {
            "source": "AlphaFold Protein Structure Database human proteome v4 raw mmCIF files",
            "derivation": (
                "Canonical sequences assembled offline from AFDB fragment metadata and validated "
                "against the full UniProt CRC64 embedded by AFDB; this is not an independently "
                "downloaded UniProt FASTA snapshot."
            ),
            "alphafold_directory": str(alphafold_dir.resolve()),
            "source_pattern": "*.cif.gz",
        },
        "validation": {
            "taxonomy_id": HUMAN_TAXONOMY_ID,
            "checksum_algorithm": "Bio.SeqUtils.CheckSum.crc64",
            "fragment_policy": "all fragments; exact spans; equal overlaps; gapless from residue 1",
            "unknown_residues_allowed": False,
        },
        "counts": {
            "source_files": len(paths),
            "canonical_sequences": len(sequences),
            "required_receptors": len(required),
            "required_receptors_covered": len(required),
        },
        "required_receptor_coverage": {
            "directory": str(required_receptor_dir.resolve()) if required_receptor_dir else None,
            "missing_accessions": [],
        },
        "artifact": {
            "path": str(out_fasta.resolve()),
            "sha256": _sha256_bytes(fasta),
            "bytes": len(fasta),
            "sequence_count": len(sequences),
        },
        "sequences": records,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    # Validation is complete before either destination is changed. Each publication is atomic.
    _atomic_write(out_fasta, fasta)
    _atomic_write(out_manifest, manifest_bytes)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alphafold-dir", required=True, type=Path)
    parser.add_argument("--out-fasta", required=True, type=Path)
    parser.add_argument("--out-manifest", required=True, type=Path)
    parser.add_argument("--min-sequences", required=True, type=int)
    parser.add_argument("--required-receptor-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = build(
            args.alphafold_dir,
            args.out_fasta,
            args.out_manifest,
            args.min_sequences,
            args.required_receptor_dir,
        )
    except CanonicalSequenceError as exc:
        raise SystemExit(f"[canonical-sequences][FATAL] {exc}") from exc
    counts = manifest["counts"]
    assert isinstance(counts, dict)
    print(
        f"[canonical-sequences] wrote {counts['canonical_sequences']} checksum-validated "
        f"sequences from {counts['source_files']} AFDB v4 fragments"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
