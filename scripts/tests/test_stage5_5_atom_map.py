from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, ChemicalFeatures


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from interaction_anchor import (  # noqa: E402
    PRESERVATION_BASIS,
    SCHEMA_VERSION,
    score_anchor_preservation,
    sha256_file,
)


SCRIPT = SCRIPTS / "stage5_5_atom_map.py"


def _embedded(smiles: str, seed: int = 7) -> Chem.Mol:
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(molecule, randomSeed=seed) == 0
    return molecule


def _write_sdf(path: Path, molecule: Chem.Mol) -> None:
    writer = Chem.SDWriter(str(path))
    writer.write(molecule)
    writer.close()


def _write_complex(path: Path, molecule: Chem.Mol) -> None:
    path.write_text(Chem.MolToPDBBlock(molecule).replace("ATOM  ", "HETATM"))


def _write_inputs(
    tmp_path: Path,
    *,
    smiles: str = "CCO",
    atom_order: list[int] | None = None,
    confirmed_atoms: list[int] | None = None,
    kept: str = "yes",
) -> tuple[Path, Path, Path, Path]:
    parent = _embedded(smiles)
    parent_sdf = tmp_path / "parent.sdf"
    complex_pdb = tmp_path / "complex.pdb"
    report = tmp_path / "boltz.tsv"
    consensus = tmp_path / "consensus.json"
    _write_sdf(parent_sdf, parent)
    bound = Chem.RenumberAtoms(parent, atom_order or list(range(parent.GetNumAtoms())))
    _write_complex(complex_pdb, bound)
    pd.DataFrame(
        [
            {
                "target_id": "P12345",
                "complex_pdb": str(complex_pdb),
                "iptm": 0.8,
                "complex_plddt": 85.0,
                "affinity_log_uM": -1.2,
                "kept": kept,
            }
        ]
    ).to_csv(report, sep="\t", index=False)
    atoms = confirmed_atoms if confirmed_atoms is not None else [0]
    consensus.write_text(
        json.dumps(
            {
                "P12345": {
                    "confirmed_atoms": atoms,
                    "coordinate_system": "boltz_complex_ligand_atom_order_0_based",
                    "degraded": False,
                    "claim_eligible": True,
                }
            }
        )
    )
    return parent_sdf, complex_pdb, report, consensus


def _run(
    parent_sdf: Path,
    report: Path,
    consensus: Path,
    output: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--parent-sdf",
            str(parent_sdf),
            "--boltz-report",
            str(report),
            "--consensus-json",
            str(consensus),
            "--out-json",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_atom_map_translates_reordered_bound_ligand_anchor(tmp_path: Path) -> None:
    parent = _embedded("CCO")
    order = [2, 1, 0, *range(3, parent.GetNumAtoms())]
    parent_sdf, _complex, report, consensus = _write_inputs(
        tmp_path,
        atom_order=order,
        confirmed_atoms=[0],
    )
    output = tmp_path / "anchors.json"

    result = _run(parent_sdf, report, consensus, output)

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text())
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["claim_eligible"] is True
    record = payload["targets"]["P12345"]
    assert payload["analog_pose_verified"] is False
    assert payload["boltz_report_sha256"] == sha256_file(report)
    assert payload["sources"] == {
        "parent_sdf": {
            "path": str(parent_sdf.resolve()),
            "bytes": parent_sdf.stat().st_size,
            "sha256": sha256_file(parent_sdf),
        },
        "consensus_json": {
            "path": str(consensus.resolve()),
            "bytes": consensus.stat().st_size,
            "sha256": sha256_file(consensus),
        },
        "boltz_report": {
            "path": str(report.resolve()),
            "bytes": report.stat().st_size,
            "sha256": sha256_file(report),
        },
    }
    assert record["confirmed_complex_atoms"] == [0]
    assert record["confirmed_parent_atoms"] == [2]
    assert record["confirmed_canonical_parent_atoms"] == [2]
    assert record["canonical_coordinate_system"] == (
        "canonical_smiles_atom_order_0_based"
    )
    assert len(record["canonical_atom_mapping"]) == 3
    assert record["mapping_status"] == "mapped"
    assert record["mapping_confidence"] == "high"
    assert record["claim_eligible"] is True
    assert record["complex_pdb"] == str(_complex.resolve())
    assert record["complex_pdb_bytes"] == _complex.stat().st_size
    assert record["complex_pdb_sha256"] == sha256_file(_complex)


def test_atom_map_rejects_ambiguous_symmetric_mapping_and_stale_output(
    tmp_path: Path,
) -> None:
    parent_sdf, _complex, report, consensus = _write_inputs(
        tmp_path,
        smiles="c1ccccc1",
        confirmed_atoms=[0],
    )
    output = tmp_path / "anchors.json"
    output.write_text("stale\n")

    result = _run(parent_sdf, report, consensus, output)

    assert result.returncode != 0
    assert "atom mapping" in result.stderr
    assert "status=ambiguous" in result.stderr
    assert not output.exists()


def test_atom_map_rejects_consensus_without_quality_kept_target(
    tmp_path: Path,
) -> None:
    parent_sdf, _complex, report, consensus = _write_inputs(tmp_path, kept="no")
    output = tmp_path / "anchors.json"

    result = _run(parent_sdf, report, consensus, output)

    assert result.returncode != 0
    assert "no quality-kept complexes" in result.stderr
    assert not output.exists()


def test_anchor_score_tracks_parent_interaction_feature_retention() -> None:
    parent = Chem.MolFromSmiles("CC(=O)O")
    retained = Chem.MolFromSmiles("CCC(=O)O")
    lost = Chem.MolFromSmiles("CCN")
    factory = ChemicalFeatures.BuildFeatureFactory(
        str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef")
    )

    retained_score = score_anchor_preservation(parent, retained, [2, 3], factory)
    lost_score = score_anchor_preservation(parent, lost, [2, 3], factory)

    assert retained_score.score == pytest.approx(1.0)
    assert retained_score.preserved_anchor_count == 2
    assert retained_score.basis == PRESERVATION_BASIS
    assert lost_score.score == pytest.approx(0.0)
    assert lost_score.preserved_anchor_count == 0


def test_anchor_score_ambiguous_mcs_is_conservative_not_optimistic() -> None:
    parent = Chem.MolFromSmiles("CCO")
    analog = Chem.MolFromSmiles("NC(=O)N")
    factory = ChemicalFeatures.BuildFeatureFactory(
        str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef")
    )

    result = score_anchor_preservation(parent, analog, [0], factory)

    assert result.basis == PRESERVATION_BASIS
    assert result.mapping_ambiguous is True
    assert result.mapping_count == 2
    assert result.score == pytest.approx(0.0)
    assert result.preserved_anchor_count == 0
