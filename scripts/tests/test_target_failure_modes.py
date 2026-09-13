"""Regression tests for documented receptor limitations.

`results/RETROSPECTIVE.md` traced every docking miss on the validation panel to
a receptor the preparation cannot model, and no published artifact said which
targets carry one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from target_failure_modes import (  # noqa: E402
    failure_modes_for,
    is_gpcr,
    load_curated,
)

GPCR_FUNCTION = "G-protein coupled receptor, Receptor, Transducer"


def test_the_shipped_list_covers_the_documented_metal_cases() -> None:
    curated = load_curated()
    assert set(curated) == {"P14679", "P08253", "P14780"}
    for record in curated.values():
        assert record["failure_mode"] == "missing_metal_cofactor"
        assert record["evidence_source"] == "results/RETROSPECTIVE.md"


def test_tyrosinase_warns_even_though_it_is_not_annotated_a_metalloprotease() -> None:
    """The best-documented metal case is why curation beats class inference.

    TYR carries a dinuclear copper centre but is not labelled a metalloprotease,
    so matching on the class annotation would miss it entirely.
    """
    modes = failure_modes_for(
        "P14679",
        molecular_function="Enzyme, Oxidoreductase",
        curated=load_curated(),
    )
    assert [mode["failure_mode"] for mode in modes] == ["missing_metal_cofactor"]
    assert modes[0]["basis"] == "curated"


def test_an_adenosine_receptor_warns_from_its_class(monkeypatch) -> None:
    """Caffeine missing ADORA1/2A/2B/3 is the documented GPCR case."""
    modes = failure_modes_for(
        "P29274", molecular_function=GPCR_FUNCTION, curated=load_curated()
    )
    assert [mode["failure_mode"] for mode in modes] == ["gpcr_inactive_state"]
    assert modes[0]["basis"] == "hpa_molecular_function"


def test_a_target_with_neither_limitation_is_not_warned_about() -> None:
    assert (
        failure_modes_for(
            "P00001", molecular_function="Enzyme, Transferase", curated=load_curated()
        )
        == []
    )


def test_a_curated_gpcr_reports_both_limitations() -> None:
    curated = {
        "P29274": {
            "uniprot": "P29274",
            "gene": "ADORA2A",
            "failure_mode": "missing_metal_cofactor",
            "cofactor": "Zn2+",
            "evidence_source": "test",
        }
    }
    modes = failure_modes_for(
        "P29274", molecular_function=GPCR_FUNCTION, curated=curated
    )
    assert [mode["failure_mode"] for mode in modes] == [
        "missing_metal_cofactor",
        "gpcr_inactive_state",
    ]


def test_class_matching_ignores_unrelated_receptor_annotations() -> None:
    assert is_gpcr("Receptor, Transducer") is False
    assert is_gpcr(None) is False
    assert is_gpcr(GPCR_FUNCTION) is True


def test_a_missing_curated_file_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        load_curated(tmp_path / "absent.csv")


def test_a_curated_row_with_no_evidence_source_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "modes.csv"
    path.write_text(
        "uniprot,gene,failure_mode,cofactor,evidence_source\n"
        "P14679,TYR,missing_metal_cofactor,Cu2+,\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as excinfo:
        load_curated(path)
    assert "evidence_source" in str(excinfo.value)


def test_a_repeated_target_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "modes.csv"
    path.write_text(
        "uniprot,gene,failure_mode,cofactor,evidence_source\n"
        "P14679,TYR,missing_metal_cofactor,Cu2+,src\n"
        "P14679,TYR,gpcr_inactive_state,none,src\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as excinfo:
        load_curated(path)
    assert "P14679" in str(excinfo.value)
