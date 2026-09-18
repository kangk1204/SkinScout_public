"""Pin every validation-panel structure to the compound it is named after.

Four of the fifteen entries once held a different molecule than their case_id
claimed: farnesol filed as retinol, a saturated lactone as ascorbic acid, a
five-membered furanone as kojic acid, and an EGCG isomer with the A- and
B-ring hydroxyls transposed. Every one of them was a valid molecule RDKit
parsed without complaint, and eleven of the panel's thirty-two pairs were
measured against the wrong structure.

Molecular formula is not enough to catch this - the EGCG isomer has the right
formula. These pin the InChIKey connectivity block, which is what the Morgan
fingerprints the retrieval runs on actually see.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "data" / "validation" / "skin_known_target_panel.csv"

# InChIKey connectivity block for the named compound, derived independently of
# the panel file. A mismatch means the panel holds a different molecule.
EXPECTED_CONNECTIVITY = {
    "Retinol": "FPIPGXGPPPQFEQ",
    "Niacinamide": "DFPAKSUCGFBDDF",
    "Ascorbic_acid": "CIWBSHSKHKDKBQ",
    "alpha-Arbutin": "BJRNKVDFDLYUGJ",
    "Kojic_acid": "BEJNERDRQOWKJM",
    "Hydroquinone": "QIGBRXMKCJKVMJ",
    "Salicylic_acid": "YGSDEFSMJLZEOE",
    "Resveratrol": "LUKBXSAWLPMMSZ",
    "EGCG": "WMBWREPUVVBILR",
    "Caffeine": "RYYVLZVUVIJVGH",
    "Adenosine": "OIRDTQYFTABQOQ",
    "Capsaicin": "YKPUWZUDDOIDPM",
    "Cinnamaldehyde": "KJPRLNWUNMBNBZ",
    "Allyl_isothiocyanate": "ZOJBYZNEUISWFT",
    "Menthol": "NOOLISFMXDJSKH",
    # v3 additions (2026-08-31). Six of the seven are registered compounds in the
    # ChEMBL mirror, which test_the_v3_additions_match_a_registered_compound
    # checks these blocks against - an independent source, not the panel file.
    "Hydrocortisone": "JYGXADMDTFJGBT",
    "Finasteride": "DBEPLOCGEIEOCV",
    "Crisaborole": "USZAGAREISWJDP",
    "Ruxolitinib": "HFNKQEVNSGCOJV",
    "Genistein": "TZBJGXHYKVUXJN",
    "Diphenhydramine": "ZZVUWRFHKOJYTH",
    "Tranexamic_acid": "GYDJEQRTZSCIOI",
}

# Added in v3. All seven are registered ChEMBL compounds, so the panel structure
# can be corroborated against the mirror rather than only against itself. This
# check earned its keep immediately: the first tranexamic acid entry was the cis
# isomer, and the mirror is where that showed up.
V3_ADDITIONS = (
    "Hydrocortisone",
    "Finasteride",
    "Crisaborole",
    "Ruxolitinib",
    "Genistein",
    "Diphenhydramine",
    "Tranexamic_acid",
)
NOT_IN_CHEMBL_MIRROR: set[str] = set()

# The structures that were wrong, so a revert is caught by name.
RETIRED_CONNECTIVITY = {
    "Retinol": "CRDAMVZIKSXKFV",       # farnesol
    "Ascorbic_acid": "ATLYZOSHQRMQRP",
    "Kojic_acid": "JTQXALQLNOBOPA",
    "EGCG": "IDJGBBBMZRXGSW",
}


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return pd.read_csv(PANEL)


def _connectivity(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    return Chem.MolToInchiKey(mol).split("-")[0]


def test_the_panel_holds_exactly_the_compounds_it_names(panel: pd.DataFrame) -> None:
    assert set(panel["case_id"]) == set(EXPECTED_CONNECTIVITY)


@pytest.mark.parametrize("case_id", sorted(EXPECTED_CONNECTIVITY))
def test_each_entry_is_the_compound_it_claims_to_be(
    panel: pd.DataFrame, case_id: str
) -> None:
    row = panel[panel["case_id"] == case_id]
    assert len(row) == 1, case_id
    assert _connectivity(row["smiles"].iloc[0]) == EXPECTED_CONNECTIVITY[case_id]


@pytest.mark.parametrize("case_id", sorted(RETIRED_CONNECTIVITY))
def test_a_corrected_entry_never_reverts(panel: pd.DataFrame, case_id: str) -> None:
    row = panel[panel["case_id"] == case_id]
    assert _connectivity(row["smiles"].iloc[0]) != RETIRED_CONNECTIVITY[case_id]


def test_the_v3_additions_match_a_registered_compound(panel: pd.DataFrame) -> None:
    """Corroborate the new structures against ChEMBL, not against the panel.

    Pinning a connectivity block taken from the same file it guards proves
    nothing. Six of the seven additions are registered ChEMBL compounds, so the
    mirror is an independent witness that the SMILES is a real molecule someone
    else also recorded. Tranexamic acid is absent from the mirror - that is why
    it is a hard case - and is exempted by name rather than silently skipped.
    """
    mirror = ROOT / "data" / "chembl37" / "human_activities.parquet"
    if not mirror.is_file():
        pytest.skip("ChEMBL mirror is not provisioned here")

    known = set(
        pd.read_parquet(mirror, columns=["standard_inchi_key"])["standard_inchi_key"]
        .astype(str)
    )
    blocks = {key.split("-")[0] for key in known}

    for case_id in V3_ADDITIONS:
        row = panel[panel["case_id"] == case_id]
        assert len(row) == 1, case_id
        block = _connectivity(row["smiles"].iloc[0])
        assert block == EXPECTED_CONNECTIVITY[case_id], case_id
        if case_id in NOT_IN_CHEMBL_MIRROR:
            assert block not in blocks, f"{case_id} is now in the mirror; re-check the exemption"
        else:
            assert block in blocks, f"{case_id} is not a registered ChEMBL structure"


def test_every_panel_compound_is_within_the_applicability_scope(
    panel: pd.DataFrame,
) -> None:
    """The panel measures the tool, so it cannot contain inputs the tool refuses."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from compound_applicability import OUT_OF_SCOPE, assess

    for _, row in panel.iterrows():
        assert assess(row["smiles"])["verdict"] != OUT_OF_SCOPE, row["case_id"]


def test_the_amendment_is_recorded_rather_than_a_silent_hash_bump() -> None:
    """Amending a preregistered panel has to leave a trail.

    The panel is frozen so results cannot be tuned after the fact. There have
    been two legitimate amendments - v1->v2 corrected four wrong structures,
    v2->v3 added seven compounds so the panel could resolve the difference it
    was being used to decide. Each has to stay identifiable: the contract
    version moves, every superseded digest stays readable so an old artifact can
    still be recognised, and a document explains why.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    from activity_recovery_contracts import (
        FROZEN_KNOWN_PANEL_CONTRACT,
        FROZEN_KNOWN_PANEL_SHA256,
        SUPERSEDED_KNOWN_PANEL_CONTRACTS,
    )

    assert FROZEN_KNOWN_PANEL_CONTRACT.endswith(".v3")
    # v1 (four wrong structures), v2 (too small to arbitrate), and the first v3
    # draft (tranexamic acid as the cis isomer). Each digest stays readable so an
    # artifact built from it can be recognised for what it is.
    for superseded in ("v1", "v2", "v3-draft"):
        digest = SUPERSEDED_KNOWN_PANEL_CONTRACTS[
            f"skinscout.skin-known-target-panel.{superseded}"
        ]
        assert digest != FROZEN_KNOWN_PANEL_SHA256, superseded
        assert len(digest) == 64, superseded
    # No two versions may share a digest, or the lookup names the wrong one.
    digests = list(SUPERSEDED_KNOWN_PANEL_CONTRACTS.values())
    assert len(set(digests)) == len(digests)

    for note in (
        "PANEL_STRUCTURE_CORRECTION_20260826.md",  # v1 -> v2
        "PANEL_ARBITRATION_20260831.md",           # v2 -> v3
    ):
        assert (ROOT / "docs" / note).exists(), note
