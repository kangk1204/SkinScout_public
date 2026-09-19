"""Regression tests for locating Boltz-2's predicted structure on disk.

Boltz writes its results under ``boltz_results_<input stem>/`` inside the
``--out_dir`` it is given. ``top_ranked_pdb`` used to check only the shallow
``<out_dir>/predictions/...`` path, so a run that produced a perfectly good
model was reported as having produced nothing at all, and the stage exited with
"Boltz-2 produced no successful cofolding reports". The sibling JSON loaders
use rglob and never had the problem, which is why the failure looked like a
scoring issue rather than a path issue.

Nothing covered this function before, which is how it survived to a real run.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from boltz2_runner import (  # noqa: E402
    load_affinity_payload,
    load_confidence_payload,
    top_ranked_pdb,
)


def _write(path: Path, text: str = "ATOM      1  N   ALA A   1\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_finds_model_under_boltz_results_wrapper(tmp_path: Path) -> None:
    """The layout Boltz actually produces."""
    out_dir = tmp_path / "boltz_out"
    expected = _write(
        out_dir / "boltz_results_input" / "predictions" / "input" / "input_model_0.pdb"
    )
    assert top_ranked_pdb(out_dir, "input") == expected


def test_still_finds_model_at_the_shallow_path(tmp_path: Path) -> None:
    """A flat layout keeps working."""
    out_dir = tmp_path / "boltz_out"
    expected = _write(out_dir / "predictions" / "input" / "input_model_0.pdb")
    assert top_ranked_pdb(out_dir, "input") == expected


def test_shallow_path_wins_when_both_exist(tmp_path: Path) -> None:
    out_dir = tmp_path / "boltz_out"
    shallow = _write(out_dir / "predictions" / "input" / "input_model_0.pdb")
    _write(
        out_dir / "boltz_results_input" / "predictions" / "input" / "input_model_0.pdb",
        "ATOM      2  C   ALA A   1\n",
    )
    assert top_ranked_pdb(out_dir, "input") == shallow


def test_empty_model_file_is_not_accepted(tmp_path: Path) -> None:
    """An empty file is not a prediction; it must not be reported as one."""
    out_dir = tmp_path / "boltz_out"
    _write(
        out_dir / "boltz_results_input" / "predictions" / "input" / "input_model_0.pdb",
        "",
    )
    assert top_ranked_pdb(out_dir, "input") is None


def test_ignores_a_file_of_the_right_name_in_the_wrong_place(tmp_path: Path) -> None:
    """Only ``.../predictions/<id>/<id>_model_0.pdb`` counts."""
    out_dir = tmp_path / "boltz_out"
    _write(out_dir / "scratch" / "input_model_0.pdb")
    _write(out_dir / "predictions" / "input_model_0.pdb")
    assert top_ranked_pdb(out_dir, "input") is None


def test_missing_output_returns_none(tmp_path: Path) -> None:
    assert top_ranked_pdb(tmp_path / "boltz_out", "input") is None


def test_json_loaders_reach_the_nested_layout(tmp_path: Path) -> None:
    """These already worked; pin the behaviour the fix relies on."""
    out_dir = tmp_path / "boltz_out"
    preds = out_dir / "boltz_results_input" / "predictions" / "input"
    _write(preds / "confidence_input_model_0.json", '{"iptm": 0.5}')
    _write(preds / "affinity_input.json", '{"affinity_pred_value": 1.25}')
    assert load_confidence_payload(out_dir) == {"iptm": 0.5}
    assert load_affinity_payload(out_dir) == {"affinity_pred_value": 1.25}
