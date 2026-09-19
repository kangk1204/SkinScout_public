#!/usr/bin/env python3
"""Shared MM-GBSA provenance and selection policy for Stages 7–9.

Two ideas were conflated: whether an MM-GBSA calculation succeeded, and
whether its result is favourable enough to advance. ``gmx_MMPBSA`` produces a
finite ΔTOTAL for any converged trajectory, including 0 and positive values.
Those are valid computed results, not calculation errors, so the readers accept
any finite number and the *selection* step is what requires a negative ΔTOTAL.
The reason is recorded in its own column.

The reported number is the entropy-free ``DELTA TOTAL`` from
``gmx_MMPBSA --create_input gb`` (entropy is off by default), evaluated at the
same 0.15 M salt concentration used for the explicit-solvent preparation.
"""

from __future__ import annotations

import pandas as pd

VALUE_COLUMN = "mmgbsa_dg_kcal_mol"
SELECTED_COLUMN = "selected_for_qm"
REASON_COLUMN = "mmgbsa_selection_reason"

# These name what the number actually is and the conditions it was computed
# under; they are written into the Stage 7 report and the Stage 9 panel.
QUANTITY = "delta_total_no_entropy"
ENTROPY_INCLUDED = False
SALT_CONCENTRATION_MOLAR = 0.150

# Only a strictly negative ΔTOTAL proceeds to QM refinement.
FAVORABLE_DELTA_TOTAL_MAX = 0.0

SELECTED_REASON = "selected_negative_delta_total"
NOT_SELECTED_REASON = "not_selected_nonnegative_delta_total"


def selection_reason(value: float) -> str:
    """Why this ΔTOTAL is (not) advanced to QM. Values are always valid."""
    return SELECTED_REASON if value < FAVORABLE_DELTA_TOTAL_MAX else NOT_SELECTED_REASON


def annotate_selection(
    mmgbsa: pd.DataFrame,
    *,
    value_column: str = VALUE_COLUMN,
) -> pd.DataFrame:
    """Return a copy with the selection decision and its reason recorded."""
    annotated = mmgbsa.copy()
    values = annotated[value_column].astype(float)
    annotated[SELECTED_COLUMN] = values < FAVORABLE_DELTA_TOTAL_MAX
    annotated[REASON_COLUMN] = [
        selection_reason(value) for value in values
    ]
    return annotated


def select_favorable(
    mmgbsa: pd.DataFrame,
    *,
    value_column: str = VALUE_COLUMN,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Rows with a favourable (negative) ΔTOTAL, best first."""
    annotated = annotate_selection(mmgbsa, value_column=value_column)
    selected = annotated[annotated[SELECTED_COLUMN]].sort_values(value_column)
    if top_n is not None:
        selected = selected.head(top_n)
    return selected
