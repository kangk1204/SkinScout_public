"""Regression tests for the input applicability gate.

`INSTRUCTIONS.md` §14-15 has always said this pipeline answers for single small
molecules only, and nothing enforced it: a peptide or a formulation blend ran
to completion and produced a ranked target list indistinguishable from a valid
one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from compound_applicability import (  # noqa: E402
    IN_SCOPE,
    OUT_OF_SCOPE,
    REVIEW,
    assess,
    physchem,
)

CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
CAPSAICIN = "COC1=C(C=CC(=C1)CNC(=O)CCCC/C=C/C(C)C)O"
TRIPEPTIDE = "CC(C)C[C@@H](N)C(=O)N[C@@H](C)C(=O)N[C@@H](Cc1ccccc1)C(=O)O"
SODIUM_LAURYL_SULFATE = "CCCCCCCCCCCCOS(=O)(=O)[O-].[Na+]"
LAURYL_SULFATE_ANION = "CCCCCCCCCCCCOS(=O)(=O)[O-]"


def _codes(result: dict, key: str) -> list[str]:
    return [item["code"] for item in result[key]]


@pytest.mark.parametrize("smiles", [CAFFEINE, CAPSAICIN])
def test_panel_compounds_are_in_scope(smiles: str) -> None:
    assert assess(smiles)["verdict"] == IN_SCOPE


def test_a_single_amide_is_not_mistaken_for_a_peptide() -> None:
    """Capsaicin carries one amide bond and is a validated panel compound."""
    result = assess(CAPSAICIN)
    assert "peptide" not in _codes(result, "exclusions")


def test_a_peptide_is_refused() -> None:
    result = assess(TRIPEPTIDE)
    assert result["verdict"] == OUT_OF_SCOPE
    assert "peptide" in _codes(result, "exclusions")


def test_a_multi_component_input_is_refused() -> None:
    result = assess(SODIUM_LAURYL_SULFATE)
    assert result["verdict"] == OUT_OF_SCOPE
    assert "mixture" in _codes(result, "exclusions")


def test_a_surfactant_is_refused_on_its_own() -> None:
    """A long alkyl tail with an ionic head, with no salt to give it away."""
    result = assess(LAURYL_SULFATE_ANION)
    assert result["verdict"] == OUT_OF_SCOPE
    assert "surfactant" in _codes(result, "exclusions")


def test_polymer_notation_is_refused() -> None:
    result = assess("*CC(*)c1ccccc1")
    assert result["verdict"] == OUT_OF_SCOPE
    assert "polymer" in _codes(result, "exclusions")


def test_an_unparseable_input_is_refused_rather_than_run() -> None:
    result = assess("not a molecule")
    assert result["verdict"] == OUT_OF_SCOPE
    assert "unparseable" in _codes(result, "exclusions")


def test_an_empty_input_is_refused() -> None:
    assert assess("")["verdict"] == OUT_OF_SCOPE


def test_a_compound_outside_the_panel_range_is_flagged_but_not_blocked() -> None:
    """Being larger than anything measured is a caution, not a verdict.

    The panel is 15 compounds, so its bounds cannot say what does not work.
    """
    result = assess("C" * 40)
    assert result["verdict"] == REVIEW
    assert result["exclusions"] == []
    assert "heavy_atoms_outside_panel" in _codes(result, "warnings")
    assert "rotatable_bonds_outside_panel" in _codes(result, "warnings")


def test_every_finding_names_where_the_rule_comes_from() -> None:
    for smiles in (TRIPEPTIDE, LAURYL_SULFATE_ANION, "C" * 40):
        result = assess(smiles)
        for item in [*result["exclusions"], *result["warnings"]]:
            assert item["basis"].strip()


def test_the_property_helper_is_the_one_substitute_discovery_uses() -> None:
    """Both paths must read the same numbers; two copies would drift."""
    import discover_substitutes

    assert discover_substitutes._physchem is physchem


def test_properties_are_reported_for_an_accepted_compound() -> None:
    result = assess(CAFFEINE)
    assert result["properties"]["heavy_atoms"] == 14
    assert result["properties"]["molecular_weight"] == pytest.approx(194.2, abs=0.2)


def _sdf(tmp_path: Path, name: str, smiles_list: list[str]) -> Path:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    path = tmp_path / name
    writer = Chem.SDWriter(str(path))
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        AllChem.Compute2DCoords(mol)
        writer.write(mol)
    writer.close()
    return path


def test_an_sdf_holding_one_molecule_is_assessed_like_a_smiles(tmp_path: Path) -> None:
    from compound_applicability import assess_sdf

    assert assess_sdf(_sdf(tmp_path, "one.sdf", [CAFFEINE]))["verdict"] == IN_SCOPE


def test_a_multi_record_sdf_is_refused_rather_than_silently_truncated(
    tmp_path: Path,
) -> None:
    """Stage 1 takes the first parseable record and drops the rest.

    Without this the second molecule onward vanishes with no message, and the
    run reports on whichever one happened to be first.
    """
    from compound_applicability import assess_sdf

    result = assess_sdf(_sdf(tmp_path, "multi.sdf", [CAFFEINE, CAPSAICIN]))
    assert result["verdict"] == OUT_OF_SCOPE
    assert _codes(result, "exclusions") == ["mixture"]


def test_an_out_of_scope_sdf_is_refused(tmp_path: Path) -> None:
    from compound_applicability import assess_sdf

    result = assess_sdf(_sdf(tmp_path, "peptide.sdf", [TRIPEPTIDE]))
    assert result["verdict"] == OUT_OF_SCOPE
    assert "peptide" in _codes(result, "exclusions")


def test_a_missing_sdf_is_refused(tmp_path: Path) -> None:
    from compound_applicability import assess_sdf

    assert assess_sdf(tmp_path / "absent.sdf")["verdict"] == OUT_OF_SCOPE


def test_the_refusal_message_names_the_rule_and_its_source() -> None:
    from compound_applicability import refusal_message

    message = refusal_message(assess(TRIPEPTIDE))
    assert message is not None
    assert "펩타이드" in message
    assert "INSTRUCTIONS.md" in message


def test_an_accepted_compound_produces_no_refusal() -> None:
    from compound_applicability import refusal_message

    assert refusal_message(assess(CAFFEINE)) is None


UV_FILTERS = {
    "oxybenzone": "COc1ccc(C(=O)c2ccccc2)c(O)c1",
    "avobenzone": "COc1ccc(C(=O)CC(=O)c2ccc(C(C)(C)C)cc2)cc1",
    "octinoxate": "CCCCC(CC)COC(=O)/C=C/c1ccc(OC)cc1",
    "octocrylene": "CCCCC(CC)COC(=O)C(=C(c1ccccc1)c1ccccc1)C#N",
}


@pytest.mark.parametrize("name", sorted(UV_FILTERS))
def test_a_curated_uv_filter_is_refused(name: str) -> None:
    """The rule was declared in the module docstring but never implemented.

    Sunscreen actives work by absorbing light, not by binding a protein, so a
    target prediction for one is meaningless - and all of them ran happily.
    """
    result = assess(UV_FILTERS[name])
    assert result["verdict"] == OUT_OF_SCOPE
    assert "uv_filter" in _codes(result, "exclusions")


def test_the_uv_filter_list_is_consistent_with_its_own_structures() -> None:
    from rdkit import Chem

    from compound_applicability import uv_filter_index

    index = uv_filter_index()
    assert len(index) >= 10
    for connectivity, record in index.items():
        mol = Chem.MolFromSmiles(record["smiles"])
        assert mol is not None, record["inci_name"]
        assert Chem.MolToInchiKey(mol).split("-")[0] == connectivity


def test_a_compound_absent_from_the_uv_list_is_not_called_a_filter() -> None:
    """The list is not exhaustive, so absence means unrecognised, not cleared."""
    assert "uv_filter" not in _codes(assess(CAFFEINE), "exclusions")


def _write_sdf_text(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _molblock(smiles: str) -> str:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)
    return Chem.MolToMolBlock(mol) + "$$$$\n"


def test_content_after_a_valid_record_is_not_silently_dropped(
    tmp_path: Path,
) -> None:
    """RDKit never reports trailing content it fails to recognise as a record.

    Counting the terminators is what makes it visible; without that the file
    ran as whatever its first record happened to be.
    """
    from compound_applicability import assess_sdf

    path = _write_sdf_text(
        tmp_path,
        "trailing.sdf",
        _molblock(CAFFEINE) + "garbage record\nnot a molblock\n$$$$\n",
    )
    result = assess_sdf(path)
    assert result["verdict"] == OUT_OF_SCOPE
    assert "unparseable" in _codes(result, "exclusions")


def test_an_unreadable_leading_record_is_refused(tmp_path: Path) -> None:
    from compound_applicability import assess_sdf

    path = _write_sdf_text(
        tmp_path,
        "leading.sdf",
        "garbage record\nnot a molblock\n$$$$\n" + _molblock(CAFFEINE),
    )
    assert assess_sdf(path)["verdict"] == OUT_OF_SCOPE


def test_a_single_record_without_a_terminator_stays_valid(tmp_path: Path) -> None:
    """Some writers omit the final "$$$$"; that is not a malformed file."""
    from compound_applicability import assess_sdf

    path = _write_sdf_text(
        tmp_path, "no_terminator.sdf", _molblock(CAFFEINE).replace("$$$$\n", "")
    )
    assert assess_sdf(path)["verdict"] == IN_SCOPE
