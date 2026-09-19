"""The offline ingredient-name index.

A wet-lab reader knows the ingredient name, not the SMILES. The Workbench
accepted SMILES and nothing else, so they had to find a structure string
elsewhere before they could ask this tool anything.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
BUILDER = ROOT / "scripts" / "build_compound_name_index.py"
INDEX = ROOT / "data" / "compound_names" / "name_index.csv"
KOREAN = ROOT / "data" / "compound_names" / "korean_aliases.csv"


def _module():
    spec = importlib.util.spec_from_file_location("name_index_under_test", BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_hangul_survives_normalisation() -> None:
    """An ASCII-only class erased every Korean name, so none were indexed."""
    normalise = _module().normalise
    assert normalise("나이아신아마이드") == "나이아신아마이드"
    assert normalise("알파-알부틴") == "알파 알부틴"
    assert normalise("Kojic  Acid!") == "kojic acid"


def test_concatenated_upstream_aliases_are_split() -> None:
    """Some alias rows glue every synonym together with '::'."""
    clean = _module()._clean_alias
    assert clean("usnic acid::usninsaeure::chembl242022") == "usnic acid"
    assert clean("Niacinamide") == "Niacinamide"


def test_a_registry_accession_is_not_offered_as_a_name() -> None:
    typeable = _module()._is_typeable
    assert typeable("Niacinamide")
    assert typeable("레티놀")
    assert not typeable("CHEMBL521")
    assert not typeable("123456")


@pytest.mark.skipif(not KOREAN.is_file(), reason="curated Korean aliases absent")
def test_every_curated_korean_alias_resolves() -> None:
    """A curated name that resolves to nothing is worse than no entry."""
    if not INDEX.is_file():
        pytest.skip("name index has not been built")
    index = pd.read_csv(INDEX)
    normalise = _module().normalise
    known = set(index["normalised"])
    with KOREAN.open(encoding="utf-8", newline="") as handle:
        curated = list(csv.DictReader(handle))
    missing = [
        row["korean_name"]
        for row in curated
        if normalise(row["korean_name"]) not in known
    ]
    assert missing == []


@pytest.mark.skipif(not INDEX.is_file(), reason="name index has not been built")
def test_the_index_covers_the_validation_panel() -> None:
    """Every compound the tool is evaluated on should be findable by name."""
    index = pd.read_csv(INDEX)
    normalise = _module().normalise
    known = set(index["normalised"])
    panel = pd.read_csv(ROOT / "data" / "validation" / "skin_known_target_panel.csv")
    missing = [
        name for name in panel["inci_name"]
        if normalise(name) not in known
    ]
    assert missing == [], f"not findable by name: {missing}"


@pytest.mark.skipif(not INDEX.is_file(), reason="name index has not been built")
def test_the_index_carries_a_usable_structure_on_every_row() -> None:
    from rdkit import Chem, RDLogger

    index = pd.read_csv(INDEX)
    assert not index["normalised"].duplicated().any()
    RDLogger.DisableLog("rdApp.*")
    sample = index.sample(min(200, len(index)), random_state=0)
    unreadable = [
        row["display_name"] for row in sample.to_dict("records")
        if Chem.MolFromSmiles(str(row["smiles"])) is None
    ]
    RDLogger.EnableLog("rdApp.*")
    assert unreadable == []
