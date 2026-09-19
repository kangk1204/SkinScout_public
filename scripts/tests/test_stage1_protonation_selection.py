"""양성자 변형 선택이 pH 7.2–7.6 지배 미세상태 쪽으로 결정론적으로 고르는지 검증한다."""

from __future__ import annotations

from scripts.stage1_protonate import select_protonation_variant


def test_prefers_zwitterion_over_neutral_amino_acid() -> None:
    choices = ["NCC(=O)[O-]", "[NH3+]CC(=O)[O-]"]
    assert select_protonation_variant(choices) == "[NH3+]CC(=O)[O-]"


def test_prefers_protonated_amine_over_neutral_base() -> None:
    choices = ["CN1CCOCC1", "C[NH+]1CCOCC1"]  # 모르폴린, pKa 8.4
    assert select_protonation_variant(choices) == "C[NH+]1CCOCC1"


def test_prefers_deprotonated_acid_over_neutral() -> None:
    choices = ["O=C(O)c1ccccc1", "O=C([O-])c1ccccc1"]
    assert select_protonation_variant(choices) == "O=C([O-])c1ccccc1"


def test_selection_is_deterministic_regardless_of_input_order() -> None:
    choices = ["C[NH+]1CCOCC1", "CN1CCOCC1", "C[NH2+]1CCOCC1"]
    assert select_protonation_variant(choices) == select_protonation_variant(
        list(reversed(choices))
    )


def test_implausible_amide_anion_is_skipped() -> None:
    implausible = "CC(=O)[N-]C"  # deprotonated carboxamide N
    plausible = "CC(=O)NC"
    assert select_protonation_variant([implausible, plausible]) == plausible


def test_unparseable_variants_are_skipped() -> None:
    assert select_protonation_variant(["not-a-smiles", "CCO"]) == "CCO"
    assert select_protonation_variant(["not-a-smiles"]) is None


def test_all_implausible_variants_return_none() -> None:
    """모두 배제되면 금지 구조를 되돌리지 않고 None을 반환해야 한다(F06)."""
    assert select_protonation_variant(["CC(=O)[N-]C"]) is None
