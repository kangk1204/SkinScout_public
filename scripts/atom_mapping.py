#!/usr/bin/env python3
"""Map generated ligand atom indices onto bound-complex ligand atom indices."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ALGORITHM_VERSION = "rdkit-graph-isomorphism-v1"


class AtomMappingError(RuntimeError):
    """Raised when atom mapping cannot be completed safely."""


@dataclass(frozen=True)
class MappingResult:
    status: str
    confidence: str
    generated_canonical_smiles: str
    complex_canonical_smiles: str
    generated_inchikey: str
    complex_inchikey: str
    generated_atom_indices: list[int]
    complex_atom_indices: list[int]
    atom_mapping: list[dict[str, int]]
    mapping_count: int
    atom_map_numbers: str
    message: str

    def to_record(self) -> dict[str, Any]:
        return {
            "algorithm_version": ALGORITHM_VERSION,
            "status": self.status,
            "confidence": self.confidence,
            "generated_canonical_smiles": self.generated_canonical_smiles,
            "complex_canonical_smiles": self.complex_canonical_smiles,
            "generated_inchikey": self.generated_inchikey,
            "complex_inchikey": self.complex_inchikey,
            "generated_atom_indices": self.generated_atom_indices,
            "complex_atom_indices": self.complex_atom_indices,
            "atom_mapping": self.atom_mapping,
            "mapping_count": self.mapping_count,
            "atom_map_numbers": self.atom_map_numbers,
            "message": self.message,
        }


def _require_rdkit() -> tuple[Any, Any]:
    try:
        from rdkit import Chem
        from rdkit.Chem import inchi
    except ImportError as exc:
        raise AtomMappingError(
            "RDKit is required for atom mapping but is not installed"
        ) from exc
    return Chem, inchi


def _read_single_sdf(path: Path, label: str, Chem: Any) -> Any:
    if not path.exists() or path.stat().st_size == 0:
        raise AtomMappingError(f"{label} SDF is missing or empty: {path}")
    supplier = Chem.SDMolSupplier(str(path), sanitize=True, removeHs=False)
    mols = [mol for mol in supplier if mol is not None]
    if len(mols) != 1:
        raise AtomMappingError(
            f"{label} SDF must contain exactly one valid molecule: {path}"
        )
    return mols[0]


def _strip_atom_map_numbers(mol: Any, Chem: Any) -> Any:
    stripped = Chem.Mol(mol)
    for atom in stripped.GetAtoms():
        atom.SetAtomMapNum(0)
    return stripped


def _canonical_smiles(mol: Any, Chem: Any) -> str:
    stripped = _strip_atom_map_numbers(mol, Chem)
    return Chem.MolToSmiles(stripped, canonical=True, isomericSmiles=True)


def _inchikey(mol: Any, Chem: Any, inchi: Any) -> str:
    stripped = _strip_atom_map_numbers(mol, Chem)
    try:
        return inchi.MolToInchiKey(stripped)
    except Exception as exc:  # pragma: no cover - depends on RDKit build flags.
        raise AtomMappingError(f"RDKit failed to compute InChIKey: {exc}") from exc


def _atom_map_state(generated: Any, complex_ligand: Any) -> tuple[str, tuple[int, ...] | None]:
    gen_nums = [atom.GetAtomMapNum() for atom in generated.GetAtoms()]
    complex_nums = [atom.GetAtomMapNum() for atom in complex_ligand.GetAtoms()]
    if not any(gen_nums) and not any(complex_nums):
        return "absent", None
    if (
        all(gen_nums)
        and all(complex_nums)
        and len(set(gen_nums)) == len(gen_nums)
        and len(set(complex_nums)) == len(complex_nums)
        and set(gen_nums) == set(complex_nums)
    ):
        complex_by_map = {
            atom.GetAtomMapNum(): atom.GetIdx()
            for atom in complex_ligand.GetAtoms()
        }
        return "complete", tuple(complex_by_map[num] for num in gen_nums)
    return "incomplete_or_nonbijective_ignored", None


def _same_bond(generated_bond: Any, complex_bond: Any) -> bool:
    return (
        generated_bond.GetBondType() == complex_bond.GetBondType()
        and generated_bond.GetIsAromatic() == complex_bond.GetIsAromatic()
        and generated_bond.GetStereo() == complex_bond.GetStereo()
    )


def _valid_mapping_tuple(generated: Any, complex_ligand: Any, mapping: tuple[int, ...]) -> bool:
    if len(mapping) != generated.GetNumAtoms() or len(set(mapping)) != len(mapping):
        return False
    if sorted(mapping) != list(range(complex_ligand.GetNumAtoms())):
        return False
    for gen_atom in generated.GetAtoms():
        complex_atom = complex_ligand.GetAtomWithIdx(mapping[gen_atom.GetIdx()])
        if (
            gen_atom.GetAtomicNum() != complex_atom.GetAtomicNum()
            or gen_atom.GetFormalCharge() != complex_atom.GetFormalCharge()
            or gen_atom.GetIsAromatic() != complex_atom.GetIsAromatic()
            or gen_atom.GetIsotope() != complex_atom.GetIsotope()
            or gen_atom.GetChiralTag() != complex_atom.GetChiralTag()
        ):
            return False
    for gen_bond in generated.GetBonds():
        begin = mapping[gen_bond.GetBeginAtomIdx()]
        end = mapping[gen_bond.GetEndAtomIdx()]
        complex_bond = complex_ligand.GetBondBetweenAtoms(begin, end)
        if complex_bond is None or not _same_bond(gen_bond, complex_bond):
            return False
    return generated.GetNumBonds() == complex_ligand.GetNumBonds()


def _success_result(
    generated: Any,
    complex_ligand: Any,
    mapping: tuple[int, ...],
    atom_map_numbers: str,
    Chem: Any,
    inchi: Any,
) -> MappingResult:
    generated_indices = list(range(generated.GetNumAtoms()))
    complex_indices = list(mapping)
    return MappingResult(
        status="mapped",
        confidence="high",
        generated_canonical_smiles=_canonical_smiles(generated, Chem),
        complex_canonical_smiles=_canonical_smiles(complex_ligand, Chem),
        generated_inchikey=_inchikey(generated, Chem, inchi),
        complex_inchikey=_inchikey(complex_ligand, Chem, inchi),
        generated_atom_indices=generated_indices,
        complex_atom_indices=complex_indices,
        atom_mapping=[
            {"generated_atom_index": gen_idx, "complex_atom_index": complex_idx}
            for gen_idx, complex_idx in zip(generated_indices, complex_indices, strict=True)
        ],
        mapping_count=1,
        atom_map_numbers=atom_map_numbers,
        message="unique graph isomorphism found",
    )


def _failure_result(
    status: str,
    message: str,
    generated: Any | None = None,
    complex_ligand: Any | None = None,
    mapping_count: int = 0,
    atom_map_numbers: str = "not_evaluated",
    Chem: Any | None = None,
    inchi: Any | None = None,
) -> MappingResult:
    return MappingResult(
        status=status,
        confidence="none",
        generated_canonical_smiles=(
            _canonical_smiles(generated, Chem) if generated is not None and Chem is not None else ""
        ),
        complex_canonical_smiles=(
            _canonical_smiles(complex_ligand, Chem)
            if complex_ligand is not None and Chem is not None
            else ""
        ),
        generated_inchikey=(
            _inchikey(generated, Chem, inchi)
            if generated is not None and Chem is not None and inchi is not None
            else ""
        ),
        complex_inchikey=(
            _inchikey(complex_ligand, Chem, inchi)
            if complex_ligand is not None and Chem is not None and inchi is not None
            else ""
        ),
        generated_atom_indices=[],
        complex_atom_indices=[],
        atom_mapping=[],
        mapping_count=mapping_count,
        atom_map_numbers=atom_map_numbers,
        message=message,
    )


def map_molecules(generated: Any, complex_ligand: Any, Chem: Any, inchi: Any) -> MappingResult:
    generated_graph = _strip_atom_map_numbers(generated, Chem)
    complex_graph = _strip_atom_map_numbers(complex_ligand, Chem)
    atom_map_numbers, atom_map_tuple = _atom_map_state(generated, complex_ligand)

    if (
        generated_graph.GetNumAtoms() != complex_graph.GetNumAtoms()
        or generated_graph.GetNumBonds() != complex_graph.GetNumBonds()
    ):
        return _failure_result(
            "no_mapping",
            "molecules have different atom or bond counts",
            generated_graph,
            complex_graph,
            atom_map_numbers=atom_map_numbers,
            Chem=Chem,
            inchi=inchi,
        )

    params = Chem.SubstructMatchParameters()
    params.useChirality = True
    params.uniquify = False
    params.maxMatches = 2
    matches = tuple(complex_graph.GetSubstructMatches(generated_graph, params))

    if len(matches) == 0:
        return _failure_result(
            "no_mapping",
            "no exact graph isomorphism found",
            generated_graph,
            complex_graph,
            atom_map_numbers=atom_map_numbers,
            Chem=Chem,
            inchi=inchi,
        )

    if len(matches) == 1:
        suffix = ""
        if atom_map_tuple is not None:
            suffix = (
                "_validated" if tuple(matches[0]) == atom_map_tuple else "_conflicting_ignored"
            )
        return _success_result(
            generated_graph,
            complex_graph,
            tuple(matches[0]),
            atom_map_numbers + suffix,
            Chem,
            inchi,
        )

    if atom_map_tuple is not None and _valid_mapping_tuple(
        generated_graph,
        complex_graph,
        atom_map_tuple,
    ):
        result = _success_result(
            generated_graph,
            complex_graph,
            atom_map_tuple,
            atom_map_numbers + "_validated_disambiguated",
            Chem,
            inchi,
        )
        return MappingResult(
            status=result.status,
            confidence="high_validated_atom_maps",
            generated_canonical_smiles=result.generated_canonical_smiles,
            complex_canonical_smiles=result.complex_canonical_smiles,
            generated_inchikey=result.generated_inchikey,
            complex_inchikey=result.complex_inchikey,
            generated_atom_indices=result.generated_atom_indices,
            complex_atom_indices=result.complex_atom_indices,
            atom_mapping=result.atom_mapping,
            mapping_count=1,
            atom_map_numbers=result.atom_map_numbers,
            message="multiple graph isomorphisms disambiguated by validated atom-map numbers",
        )

    return _failure_result(
        "ambiguous",
        "multiple graph isomorphisms found; refusing to choose a mapping",
        generated_graph,
        complex_graph,
        mapping_count=len(matches),
        atom_map_numbers=atom_map_numbers,
        Chem=Chem,
        inchi=inchi,
    )


def map_sdf_files(generated_sdf: Path, complex_ligand_sdf: Path) -> MappingResult:
    Chem, inchi = _require_rdkit()
    generated = _read_single_sdf(generated_sdf, "generated", Chem)
    complex_ligand = _read_single_sdf(complex_ligand_sdf, "complex ligand", Chem)
    return map_molecules(generated, complex_ligand, Chem, inchi)


def write_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Map generated ligand atoms to bound-complex ligand atom indices."
    )
    parser.add_argument("generated_sdf", type=Path)
    parser.add_argument("complex_ligand_sdf", type=Path)
    parser.add_argument("output_json", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = map_sdf_files(args.generated_sdf, args.complex_ligand_sdf)
    except AtomMappingError as exc:
        result = _failure_result("invalid_input", str(exc))
        write_record(args.output_json, result.to_record())
        return 2

    write_record(args.output_json, result.to_record())
    return 0 if result.status == "mapped" else 2


if __name__ == "__main__":
    raise SystemExit(main())
