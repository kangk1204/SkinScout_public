"""Regression tests for the pLDDT scale used by the Stage 5 quality gate.

Boltz-2 reports ``complex_plddt`` as a fraction (0.94). ``--plddt-threshold``
is validated as 0-100 and defaults to 75. The two were compared directly, so
``0.94 >= 75.0`` was false for every target and the stage rejected everything
it had just co-folded. The first real run of this stage scored

    P00918  iptm 0.959  complex_plddt 0.941
    P37231  iptm 0.881  complex_plddt 0.879
    P23219  iptm 0.639  complex_plddt 0.942

and kept none of them, reporting only "Boltz-2 produced no targets passing
quality thresholds" — a message that reads as a quality problem rather than a
unit problem.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import stage5_boltz2 as stage5  # noqa: E402


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (0.941, 94.1),
        (0.879, 87.9),
        (0.639, 63.9),
        (0.0, 0.0),
        (1.0, 100.0),
    ],
)
def test_fraction_is_scaled_to_percent(raw: float, expected: float) -> None:
    assert stage5._plddt_percent(raw, "P00918") == pytest.approx(expected)


@pytest.mark.parametrize("raw", [75.0, 87.9, 94.1, 100.0])
def test_percent_is_left_alone(raw: float) -> None:
    assert stage5._plddt_percent(raw, "P00918") == pytest.approx(raw)


def test_the_three_measured_targets_clear_the_default_gate() -> None:
    """The values that the first real run produced must pass the default 75."""
    measured = {"P00918": 0.941, "P37231": 0.879, "P23219": 0.942}
    default_threshold = 75.0
    for uid, raw in measured.items():
        assert stage5._plddt_percent(raw, uid) >= default_threshold, uid


def test_a_genuinely_poor_fraction_still_fails() -> None:
    """Scaling must not turn a bad prediction into a passing one."""
    assert stage5._plddt_percent(0.42, "P00918") < 75.0
