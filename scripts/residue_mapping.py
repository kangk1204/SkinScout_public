#!/usr/bin/env python3
"""Observed-residue mapping for PDB inputs shared across pipeline stages.

A PDB file stores only the residues that were resolved. Removing the missing
residues and joining whatever remains produces a sequence in which residue 10
is immediately followed by residue 20 - a molecule that was never observed and
that Stage 6 would then sample with BioEmu. The old ``pdb_sequence`` did
exactly that join and said nothing about it.

This module makes the discontinuity explicit:

* :func:`pdb_residue_mapping` returns the observed residues, the contiguous
  spans they form, and one gap record per jump (residue-number gap or chain
  boundary).
* :func:`pdb_sequence` returns a sequence only for a genuinely continuous
  observed chain; it raises :class:`ResidueGapError` otherwise.
* :func:`concatenated_pdb_sequence` is the explicit opt-in for callers that
  deliberately build a new construct (for example a pocket crop) and carry
  their own residue mapping.
* :func:`build_mapping_record` / :func:`write_mapping_record` persist the
  sequence hash and observed mapping next to a prepared structure so the next
  stage can prove the file it reads is the file that was mapped.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O",
}

MAPPING_SCHEMA = "skinscout.residue_mapping.v1"
GAP_RESIDUE_NUMBER = "residue_number_gap"
GAP_CHAIN_BOUNDARY = "chain_boundary"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ResidueGapError(ValueError):
    """A continuous sequence would join residues that were never adjacent."""


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ObservedResidue:
    chain: str
    number: int
    insertion_code: str
    name: str

    @property
    def one_letter(self) -> str:
        return THREE_TO_ONE.get(self.name.upper(), "X")

    @property
    def label(self) -> str:
        return f"{self.chain}:{self.name}{self.number}{self.insertion_code}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "chain": self.chain,
            "number": self.number,
            "insertion_code": self.insertion_code,
            "name": self.name,
            "one_letter": self.one_letter,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ObservedResidue":
        return cls(
            chain=str(payload.get("chain", "")),
            number=int(payload.get("number", 0)),
            insertion_code=str(payload.get("insertion_code", "")),
            name=str(payload.get("name", "")),
        )


@dataclass(frozen=True)
class ResidueSpan:
    chain: str
    residues: tuple[ObservedResidue, ...]

    @property
    def sequence(self) -> str:
        return "".join(residue.one_letter for residue in self.residues)

    @property
    def first(self) -> ObservedResidue:
        return self.residues[0]

    @property
    def last(self) -> ObservedResidue:
        return self.residues[-1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "chain": self.chain,
            "first_number": self.first.number,
            "first_insertion_code": self.first.insertion_code,
            "last_number": self.last.number,
            "last_insertion_code": self.last.insertion_code,
            "n_residues": len(self.residues),
            "sequence": self.sequence,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ResidueSpan":
        sequence = str(payload.get("sequence", ""))
        first_number = int(payload.get("first_number", 0))
        chain = str(payload.get("chain", ""))
        residues = []
        for offset, letter in enumerate(sequence):
            name = _one_to_three(letter)
            residues.append(
                ObservedResidue(
                    chain=chain,
                    number=first_number + offset,
                    insertion_code="",
                    name=name,
                )
            )
        return cls(chain=chain, residues=tuple(residues))


@dataclass(frozen=True)
class ResidueGap:
    kind: str
    before: ObservedResidue
    after: ObservedResidue
    missing_residues: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "missing_residues": self.missing_residues,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ResidueGap":
        missing = payload.get("missing_residues")
        return cls(
            kind=str(payload.get("kind", "")),
            before=ObservedResidue.from_dict(payload.get("before") or {}),
            after=ObservedResidue.from_dict(payload.get("after") or {}),
            missing_residues=int(missing) if isinstance(missing, int) else None,
        )


_THREE_BY_ONE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
    "U": "SEC", "O": "PYL", "X": "UNK",
}


def _one_to_three(letter: str) -> str:
    return _THREE_BY_ONE.get(letter.upper(), "UNK")


@dataclass(frozen=True)
class SequenceMapping:
    residues: tuple[ObservedResidue, ...]
    spans: tuple[ResidueSpan, ...]
    gaps: tuple[ResidueGap, ...]

    @property
    def sequence(self) -> str:
        return "".join(residue.one_letter for residue in self.residues)

    @property
    def continuous(self) -> bool:
        return not self.gaps

    def rejection_reason(self) -> str | None:
        if self.continuous:
            return None
        gap = self.gaps[0]
        if gap.kind == GAP_CHAIN_BOUNDARY:
            detail = (
                f"chain {gap.before.chain} ends at {gap.before.number} and chain "
                f"{gap.after.chain} starts at {gap.after.number}"
            )
        else:
            missing = (
                f"{gap.missing_residues} unobserved residue(s)"
                if gap.missing_residues
                else "unobserved residues"
            )
            detail = (
                f"residue {gap.before.number}{gap.before.insertion_code} is followed "
                f"by residue {gap.after.number}{gap.after.insertion_code} "
                f"({missing})"
            )
        return (
            f"observed structure is discontinuous: {detail}; a continuous "
            "sequence would join residues that were never adjacent"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "continuous": self.continuous,
            "spans": [span.to_dict() for span in self.spans],
            "gaps": [gap.to_dict() for gap in self.gaps],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SequenceMapping":
        spans = tuple(
            ResidueSpan.from_dict(item)
            for item in payload.get("spans", [])
            if isinstance(item, dict)
        )
        gaps = tuple(
            ResidueGap.from_dict(item)
            for item in payload.get("gaps", [])
            if isinstance(item, dict)
        )
        residues = tuple(
            residue for span in spans for residue in span.residues
        )
        return cls(residues=residues, spans=spans, gaps=gaps)


def _adjacent(previous: ObservedResidue, current: ObservedResidue) -> bool:
    if previous.chain != current.chain:
        return False
    if current.number == previous.number:
        before = previous.insertion_code
        after = current.insertion_code
        if before == "" and after == "A":
            return True
        if len(before) == 1 and len(after) == 1 and ord(after) == ord(before) + 1:
            return True
        return False
    if current.number == previous.number + 1:
        return current.insertion_code == ""
    return False


def _missing_count(previous: ObservedResidue, current: ObservedResidue) -> int | None:
    if previous.chain != current.chain:
        return None
    if previous.insertion_code or current.insertion_code:
        return None
    if current.number <= previous.number + 1:
        return None
    return current.number - previous.number - 1


def pdb_residue_mapping(pdb: Path) -> SequenceMapping:
    """Parse the observed CA residues and their discontinuities.

    Residues are grouped per chain in file order, sorted by
    (residue number, insertion code). A jump in numbering or a change of chain
    becomes a :class:`ResidueGap` instead of a silent concatenation.
    """
    by_chain: dict[str, list[ObservedResidue]] = {}
    seen: set[tuple[str, int, str]] = set()
    for line in pdb.read_text(errors="replace").splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[12:16].strip() != "CA":
            continue
        chain = line[21]
        try:
            number = int(line[22:26])
        except ValueError:
            continue
        insertion_code = line[26].strip()
        key = (chain, number, insertion_code)
        if key in seen:
            continue
        seen.add(key)
        by_chain.setdefault(chain, []).append(
            ObservedResidue(
                chain=chain,
                number=number,
                insertion_code=insertion_code,
                name=line[17:20].strip().upper(),
            )
        )

    ordered_chain_ids = list(by_chain)
    chains: list[tuple[str, tuple[ObservedResidue, ...]]] = []
    for chain_id in ordered_chain_ids:
        residues = tuple(
            sorted(
                by_chain[chain_id],
                key=lambda residue: (
                    residue.number,
                    residue.insertion_code,
                ),
            )
        )
        chains.append((chain_id, residues))

    spans: list[ResidueSpan] = []
    gaps: list[ResidueGap] = []
    flat: list[ObservedResidue] = []
    previous: ObservedResidue | None = None
    for _chain_id, residues in chains:
        current_span: list[ObservedResidue] = []
        for residue in residues:
            if previous is not None and not _adjacent(previous, residue):
                if current_span:
                    spans.append(
                        ResidueSpan(
                            chain=current_span[0].chain,
                            residues=tuple(current_span),
                        )
                    )
                    current_span = []
                gaps.append(
                    ResidueGap(
                        kind=(
                            GAP_CHAIN_BOUNDARY
                            if previous.chain != residue.chain
                            else GAP_RESIDUE_NUMBER
                        ),
                        before=previous,
                        after=residue,
                        missing_residues=_missing_count(previous, residue),
                    )
                )
            current_span.append(residue)
            flat.append(residue)
            previous = residue
        if current_span:
            spans.append(
                ResidueSpan(
                    chain=current_span[0].chain,
                    residues=tuple(current_span),
                )
            )
    return SequenceMapping(residues=tuple(flat), spans=tuple(spans), gaps=tuple(gaps))


def pdb_sequence(pdb: Path) -> str:
    """Continuous observed sequence, or :class:`ResidueGapError`."""
    mapping = pdb_residue_mapping(pdb)
    if not mapping.continuous:
        raise ResidueGapError(
            f"{pdb}: {mapping.rejection_reason()}"
        )
    return mapping.sequence


def concatenated_pdb_sequence(pdb: Path) -> str:
    """Observed sequence with discontinuities joined.

    Only for callers that deliberately build a new construct and carry their
    own explicit residue mapping (for example a pocket crop with a crop map).
    """
    return pdb_residue_mapping(pdb).sequence


def mapping_path_for_pdb(pdb: Path) -> Path:
    return pdb.with_name(f"{pdb.stem}_residue_mapping.json")


def build_mapping_record(
    pdb: Path,
    *,
    target_id: str,
    source: str,
    canonical_sequence: str,
) -> dict[str, Any]:
    """Describe one prepared structure against its canonical target sequence."""
    mapping = pdb_residue_mapping(pdb)
    record: dict[str, Any] = {
        "schema_version": MAPPING_SCHEMA,
        "target_id": target_id,
        "source": source,
        "input_pdb": pdb.name,
        "input_pdb_sha256": _sha256_file(pdb),
        "canonical_sequence": canonical_sequence,
        "canonical_sequence_sha256": _sha256_text(canonical_sequence),
        "observed_sequence": mapping.sequence,
        "observed_sequence_sha256": _sha256_text(mapping.sequence),
        "observed_continuous": mapping.continuous,
        "sampling_allowed": mapping.continuous,
        "sampling_rejection_reason": mapping.rejection_reason(),
        "spans": [span.to_dict() for span in mapping.spans],
        "gaps": [gap.to_dict() for gap in mapping.gaps],
    }
    return record


def write_mapping_record(
    pdb: Path,
    record: dict[str, Any],
    out_path: Path,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    tmp.replace(out_path)
    return out_path


def load_mapping_record(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def validate_mapping_record(pdb: Path, record: dict[str, Any]) -> list[str]:
    """Return every reason the record does not describe the current PDB."""
    errors: list[str] = []
    if record.get("schema_version") != MAPPING_SCHEMA:
        errors.append(
            f"unexpected residue mapping schema {record.get('schema_version')!r}"
        )
    expected_hash = record.get("input_pdb_sha256")
    if not isinstance(expected_hash, str) or not _SHA256_RE.fullmatch(expected_hash):
        errors.append("residue mapping record has no valid input_pdb_sha256")
    elif _sha256_file(pdb) != expected_hash:
        errors.append("input PDB hash does not match the residue mapping record")
    mapping = pdb_residue_mapping(pdb)
    observed = record.get("observed_sequence")
    if not isinstance(observed, str) or observed != mapping.sequence:
        errors.append("observed sequence does not match the residue mapping record")
    recorded_gaps = record.get("gaps")
    if recorded_gaps is not None:
        expected_gaps = [gap.to_dict() for gap in mapping.gaps]
        if recorded_gaps != expected_gaps:
            errors.append("residue gaps do not match the residue mapping record")
    if record.get("observed_continuous") is False or mapping.gaps:
        errors.append(
            mapping.rejection_reason()
            or "observed structure is discontinuous"
        )
    if record.get("sampling_allowed") is False:
        errors.append("residue mapping record does not allow sampling")
    return errors
