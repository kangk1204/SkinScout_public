"""The query and the references have to be the same kind of molecule.

Stage 3 reads the run's SDF with `removeHs=False`, because the file comes from
conformer generation and carries explicit hydrogens. It then fingerprinted that
molecule directly. Every reference it compares against - the ChEMBL mirror in
`stage0_chembl_fingerprints.py` and the retrieval index in
`build_activity_retrieval_index.py` - is built from SMILES, where hydrogens are
implicit, and Morgan treats an explicit hydrogen as an atom.

So a compound compared with itself scored 0.19, not 1.0. Measured on real run
artifacts before the fix: niacinamide 0.27, tapinarof 0.24, alpha-arbutin 0.19.
Every `daina_zoete_proteome.tsv` in `results/runs/` shows it - ethanol against
2,943 targets tops out at 0.1905 - and no evaluation could see it, because the
harness scores from SMILES and never touches the SDF path.

This is the test that would have caught it: one molecule, both paths, 1.0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

# Small, aromatic, sugar-bearing, salt-forming: the shapes where the two
# representations diverge differently.
COMPOUNDS = {
    "ethanol": "CCO",
    "niacinamide": "NC(=O)c1cccnc1",
    "alpha-arbutin": "OC[C@H]1O[C@@H](Oc2ccc(O)cc2)[C@H](O)[C@@H](O)[C@@H]1O",
    "kojic-acid": "OCc1occ(O)c(=O)c1",
    "tranexamic-acid": "NC[C@H]1CC[C@@H](CC1)C(O)=O",
}


def _sdf_with_explicit_hydrogens(tmp_path: Path, name: str, smiles: str) -> Path:
    """What Stage 3 actually receives: a 3D conformer with hydrogens on it."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=20260831)
    path = tmp_path / f"{name}.sdf"
    with Chem.SDWriter(str(path)) as writer:
        writer.write(mol)
    return path


@pytest.mark.parametrize("name,smiles", sorted(COMPOUNDS.items()))
def test_the_run_path_fingerprint_matches_the_retrieval_index(
    tmp_path: Path, name: str, smiles: str
) -> None:
    from rdkit import DataStructs

    from activity_retrieval_scoring import query_features as reference_features
    from stage3_daina_zoete import query_features as run_features

    run_fp, _ = run_features(_sdf_with_explicit_hydrogens(tmp_path, name, smiles))
    reference_fp, _, _, _ = reference_features(smiles)

    assert DataStructs.TanimotoSimilarity(run_fp, reference_fp) == pytest.approx(1.0), name


@pytest.mark.parametrize("name,smiles", sorted(COMPOUNDS.items()))
def test_the_run_path_fingerprint_matches_the_chembl_mirror(
    tmp_path: Path, name: str, smiles: str
) -> None:
    """The mirror standardises less than the index does - plain MolFromSmiles -
    so matching one is not evidence of matching the other."""
    from rdkit import Chem, DataStructs

    from stage0_chembl_fingerprints import MORGAN_GENERATOR
    from stage3_daina_zoete import query_features as run_features

    run_fp, _ = run_features(_sdf_with_explicit_hydrogens(tmp_path, name, smiles))
    mirror_fp = MORGAN_GENERATOR.GetFingerprint(Chem.MolFromSmiles(smiles))

    assert DataStructs.TanimotoSimilarity(run_fp, mirror_fp) == pytest.approx(1.0), name


def test_explicit_hydrogens_really_do_change_the_fingerprint(tmp_path: Path) -> None:
    """Pins the mechanism, so the fix cannot be mistaken for a no-op refactor.

    If a future RDKit made Morgan ignore explicit hydrogens this would fail, and
    the right response is to delete this test - not to conclude the bug was
    imaginary.
    """
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    smiles = COMPOUNDS["alpha-arbutin"]
    naive = generator.GetFingerprint(
        next(iter(Chem.SDMolSupplier(str(_sdf_with_explicit_hydrogens(tmp_path, "a", smiles)),
                                    removeHs=False)))
    )
    reference = generator.GetFingerprint(Chem.MolFromSmiles(smiles))

    assert DataStructs.TanimotoSimilarity(naive, reference) < 0.3


def test_the_connectivity_key_still_survives_the_change(tmp_path: Path) -> None:
    """The fix moved the fingerprint after the standardisation that produces the
    InChIKey. The key is what the leakage modes filter on, so it must be
    untouched."""
    from rdkit import Chem
    from rdkit.Chem import inchi
    from rdkit.Chem.MolStandardize import rdMolStandardize

    from stage3_daina_zoete import query_features as run_features

    for name, smiles in COMPOUNDS.items():
        _, key = run_features(_sdf_with_explicit_hydrogens(tmp_path, name, smiles))
        expected = inchi.MolToInchiKey(
            rdMolStandardize.FragmentParent(Chem.MolFromSmiles(smiles))
        ).split("-", 1)[0]
        assert key == expected, name
