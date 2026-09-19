"""F19 regression: MM-GBSA values are results; selection is a separate policy.

Finite 0 and positive ΔTOTAL values are valid computed results. The readers
must accept them, NaN/Infinity must be rejected as calculation errors, and the
decision to advance only negative values to QM must be recorded with an
explicit reason. Wording/manifest fields must match the actual MDP and ion
setup (position restraints on, 0.15 M background salt, entropy not included).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import mmgbsa_policy  # noqa: E402
import stage7_gromacs_prep as prep  # noqa: E402
import stage7_mmgbsa  # noqa: E402
import stage8_crest  # noqa: E402


def _mmgbsa_report(path: Path, values: list[float | str]) -> Path:
    rows = [
        f"P{index + 1}\t{value}\tok" for index, value in enumerate(values)
    ]
    path.write_text("target_id\tmmgbsa_dg_kcal_mol\tstatus\n" + "\n".join(rows) + "\n")
    return path


def test_minus_one_zero_plus_one_are_all_valid_values(tmp_path: Path) -> None:
    report = _mmgbsa_report(tmp_path / "mmgbsa.tsv", [-1.0, 0.0, 1.0])

    parsed = stage8_crest.read_mmgbsa_report(report)

    assert parsed["mmgbsa_dg_kcal_mol"].tolist() == [-1.0, 0.0, 1.0]


def test_nan_and_infinity_are_rejected_as_calculation_errors(tmp_path: Path) -> None:
    infinite = _mmgbsa_report(tmp_path / "inf.tsv", [float("inf")])
    with pytest.raises(SystemExit) as excinfo:
        stage8_crest.read_mmgbsa_report(infinite)
    assert "must be finite" in str(excinfo.value)

    missing = _mmgbsa_report(tmp_path / "nan.tsv", ["nan"])
    with pytest.raises(SystemExit) as excinfo:
        stage8_crest.read_mmgbsa_report(missing)
    assert "mmgbsa_dg_kcal_mol" in str(excinfo.value)


def test_selection_keeps_only_negative_values_and_records_reasons() -> None:
    frame = pd.DataFrame(
        {
            "target_id": ["P1", "P2", "P3"],
            "mmgbsa_dg_kcal_mol": [-1.0, 0.0, 1.0],
            "status": ["ok", "ok", "ok"],
        }
    )

    annotated = mmgbsa_policy.annotate_selection(frame)

    assert annotated["selected_for_qm"].tolist() == [True, False, False]
    assert annotated["mmgbsa_selection_reason"].tolist() == [
        mmgbsa_policy.SELECTED_REASON,
        mmgbsa_policy.NOT_SELECTED_REASON,
        mmgbsa_policy.NOT_SELECTED_REASON,
    ]
    selected = mmgbsa_policy.select_favorable(frame)
    assert selected["target_id"].tolist() == ["P1"]


def test_selection_honors_top_n_and_reports_zero_selection() -> None:
    frame = pd.DataFrame(
        {
            "target_id": ["P1", "P2", "P3"],
            "mmgbsa_dg_kcal_mol": [-3.0, -2.0, -1.0],
            "status": ["ok", "ok", "ok"],
        }
    )
    assert mmgbsa_policy.select_favorable(frame, top_n=2)["target_id"].tolist() == [
        "P1",
        "P2",
    ]
    none_negative = frame.assign(mmgbsa_dg_kcal_mol=[0.0, 1.0, 2.0])
    assert mmgbsa_policy.select_favorable(none_negative).empty


def test_stage7_parses_zero_and_positive_delta_total() -> None:
    parsed = stage7_mmgbsa.parse_delta_total(
        "DELTA TOTAL     1.2345    0.1    0.2    0.3    0.4\n"
    )
    assert parsed is not None
    assert parsed[0] == pytest.approx(1.2345)

    zero = stage7_mmgbsa.parse_delta_total("DELTA TOTAL     0.0000\n")
    assert zero == (0.0, None)

    assert stage7_mmgbsa.parse_delta_total("DELTA TOTAL     nan\n") is None
    assert stage7_mmgbsa.parse_delta_total("DELTA TOTAL     inf\n") is None


def test_producer_names_quantity_and_conditions(tmp_path: Path, monkeypatch) -> None:
    def fake_run(cmd, **_kwargs):
        (tmp_path / "mmpbsa.in").write_text(
            "&general\n  sys_name = X\n  startframe = 1\n  saltcon = 0.0\n  entropy = 0\n/\n"
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(stage7_mmgbsa.subprocess, "run", fake_run)
    path = stage7_mmgbsa.write_mmpbsa_input(tmp_path, 100)

    text = path.read_text()
    assert "saltcon" in text
    assert "0.150" in text
    assert mmgbsa_policy.QUANTITY == "delta_total_no_entropy"
    assert mmgbsa_policy.ENTROPY_INCLUDED is False
    assert mmgbsa_policy.SALT_CONCENTRATION_MOLAR == 0.150


def test_equilibration_mdps_apply_position_restraints() -> None:
    assert "define = -DPOSRES" in prep.NVT_MDP
    assert "define = -DPOSRES" in prep.NPT_MDP
    assert "define = -DPOSRES" not in prep.PROD_MDP_TEMPLATE


def test_ion_concentration_argument_is_validated(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage7_gromacs_prep.py"),
            "--ensemble-consensus", str(tmp_path / "missing.tsv"),
            "--receptor-manifest", str(tmp_path / "missing_receptors.tsv"),
            "--ligand-sdf", str(tmp_path / "missing.sdf"),
            "--out-dir", str(tmp_path / "out"),
            "--out-manifest", str(tmp_path / "md.tsv"),
            "--ion-concentration-molar", "nan",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    assert result.returncode != 0
    assert "ion-concentration-molar must be a finite value >= 0" in result.stderr


def test_manuscript_wording_matches_mdp_and_mmgbsa_conditions() -> None:
    import stage11_manuscript as manuscript

    text = manuscript.METHODS
    assert "position-restrained NVT/NPT equilibration" in text
    assert "define = -DPOSRES" in text
    assert "saltcon = 0.150" in text
    assert "entropy-free" in text
