"""Unit tests for RDKit graph-isomorphism ligand atom mapping."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atom_mapping import ALGORITHM_VERSION, map_molecules  # noqa: E402

pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402
from rdkit.Chem import inchi  # noqa: E402


SCRIPT = Path(__file__).resolve().parents[1] / "atom_mapping.py"


def _mol(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return mol


def _write_sdf(path: Path, mol: Chem.Mol) -> None:
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()


def test_reordered_atoms_maps_generated_to_complex_indices() -> None:
    generated = _mol("CCO")
    complex_ligand = Chem.RenumberAtoms(generated, [2, 0, 1])

    result = map_molecules(generated, complex_ligand, Chem, inchi)

    assert result.status == "mapped"
    assert result.generated_atom_indices == [0, 1, 2]
    assert result.complex_atom_indices == [1, 2, 0]
    assert result.atom_mapping == [
        {"generated_atom_index": 0, "complex_atom_index": 1},
        {"generated_atom_index": 1, "complex_atom_index": 2},
        {"generated_atom_index": 2, "complex_atom_index": 0},
    ]


def test_unique_mapping_includes_required_sidecar_fields() -> None:
    generated = _mol("CCO")
    complex_ligand = Chem.RenumberAtoms(generated, [1, 2, 0])

    record = map_molecules(generated, complex_ligand, Chem, inchi).to_record()

    assert record["algorithm_version"] == ALGORITHM_VERSION
    assert record["status"] == "mapped"
    assert record["confidence"] == "high"
    assert record["generated_canonical_smiles"] == "CCO"
    assert record["complex_canonical_smiles"] == "CCO"
    assert record["generated_inchikey"] == record["complex_inchikey"]
    assert record["generated_atom_indices"] == [0, 1, 2]
    assert record["complex_atom_indices"] == [2, 0, 1]


def test_symmetric_ambiguity_fails_closed_without_atom_maps() -> None:
    generated = _mol("c1ccccc1")
    complex_ligand = Chem.RenumberAtoms(generated, [3, 4, 5, 0, 1, 2])

    result = map_molecules(generated, complex_ligand, Chem, inchi)

    assert result.status == "ambiguous"
    assert result.confidence == "none"
    assert result.generated_atom_indices == []
    assert result.complex_atom_indices == []
    assert result.mapping_count == 2


def test_validated_atom_map_numbers_can_disambiguate_symmetric_graph() -> None:
    generated = _mol("[CH3:10][CH3:20]")
    complex_ligand = Chem.RenumberAtoms(generated, [1, 0])

    result = map_molecules(generated, complex_ligand, Chem, inchi)

    assert result.status == "mapped"
    assert result.confidence == "high_validated_atom_maps"
    assert result.complex_atom_indices == [1, 0]
    assert result.atom_map_numbers == "complete_validated_disambiguated"


def test_conflicting_atom_map_numbers_do_not_override_unique_graph_mapping() -> None:
    generated = _mol("CCO")
    for atom, atom_map in zip(generated.GetAtoms(), [1, 2, 3], strict=True):
        atom.SetAtomMapNum(atom_map)
    complex_ligand = Chem.RenumberAtoms(generated, [2, 0, 1])
    for atom, atom_map in zip(complex_ligand.GetAtoms(), [2, 3, 1], strict=True):
        atom.SetAtomMapNum(atom_map)

    result = map_molecules(generated, complex_ligand, Chem, inchi)

    assert result.status == "mapped"
    assert result.complex_atom_indices == [1, 2, 0]
    assert result.atom_map_numbers == "complete_conflicting_ignored"


def test_non_isomorphic_molecules_fail_closed() -> None:
    result = map_molecules(_mol("CCO"), _mol("CCN"), Chem, inchi)

    assert result.status == "no_mapping"
    assert result.confidence == "none"
    assert result.atom_mapping == []


def test_cli_invalid_input_writes_fail_closed_json(tmp_path: Path) -> None:
    generated = tmp_path / "generated.sdf"
    complex_ligand = tmp_path / "complex.sdf"
    output = tmp_path / "mapping.json"
    generated.write_text("not an sdf\n")
    _write_sdf(complex_ligand, _mol("CCO"))

    res = subprocess.run(
        [sys.executable, str(SCRIPT), str(generated), str(complex_ligand), str(output)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert res.returncode == 2
    record = json.loads(output.read_text())
    assert record["status"] == "invalid_input"
    assert record["confidence"] == "none"
    assert record["atom_mapping"] == []


def test_cli_output_is_deterministic(tmp_path: Path) -> None:
    generated = tmp_path / "generated.sdf"
    complex_ligand = tmp_path / "complex.sdf"
    out1 = tmp_path / "mapping1.json"
    out2 = tmp_path / "mapping2.json"
    _write_sdf(generated, _mol("CCO"))
    _write_sdf(complex_ligand, Chem.RenumberAtoms(_mol("CCO"), [2, 0, 1]))

    cmd = [sys.executable, str(SCRIPT), str(generated), str(complex_ligand)]
    res1 = subprocess.run(cmd + [str(out1)], check=False, capture_output=True, text=True)
    res2 = subprocess.run(cmd + [str(out2)], check=False, capture_output=True, text=True)

    assert res1.returncode == 0, res1.stderr
    assert res2.returncode == 0, res2.stderr
    assert out1.read_text() == out2.read_text()
    record = json.loads(out1.read_text())
    assert record["algorithm_version"] == ALGORITHM_VERSION
    assert record["status"] == "mapped"
