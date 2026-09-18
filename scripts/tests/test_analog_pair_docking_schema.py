"""F21 regression: pair-docking CSV keeps one fixed schema and fresh scores.

The fast-preset failure row used to write 11 values into a 12-column header, so
the failure reason landed in ``review_required`` and ``note`` stayed empty.
Separately, an informational note (``synthesized_rank_outside_daina_top256``)
made the old ``if not note`` guard skip reading the scores of a successful run.

These tests pin the fixed schema plus the failure/informational split:
* a fast failure row puts the reason in ``note`` and keeps review flags,
* a top256-outside target keeps both its fresh scores and its note,
* a failed step keeps scores empty even if score files exist.
"""

from __future__ import annotations

import csv
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analog_funnel"))

OUT_FIELDS = [
    "seed_dir", "candidate_id", "run_id", "uniprot",
    "daina_rank", "autodock_kcal_mol", "gnina_cnn_affinity",
    "skin_sens", "cosmetic", "funnel_score", "review_required",
    "claim_eligible", "safety_reason", "note",
]


def _docking():
    return importlib.import_module("pair_docking")


def _item(*, review_required: bool = False, claim_eligible: bool = True,
          safety_reason: str = "decision:PASS") -> dict:
    return {
        "seed_dir": "seedA",
        "candidate_id": "C1",
        "run_id": "run1",
        "uniprot": "P11111",
        "smiles": "CCO",
        "funnel_score": "0.9",
        "skin_sens": "PASS",
        "cosmetic": "negative",
        "review_required": review_required,
        "claim_eligible": claim_eligible,
        "safety_reason": safety_reason,
    }


@pytest.fixture()
def docking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = _docking()
    logs = tmp_path / "logs"
    logs.mkdir()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "LOGDIR", logs)
    monkeypatch.setattr(module, "docking_box_dir", lambda: tmp_path / "boxes")
    return module


def _write_daina(run_dir: Path, rows: list[dict]) -> None:
    fast_dir = run_dir / "03_targets/mode_fast"
    fast_dir.mkdir(parents=True)
    fields = ["target_id", "daina_rank", "ligand_smiles"]
    with (fast_dir / "daina_top256.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_output_schema_is_fixed_and_row_order_matches() -> None:
    docking = _docking()
    assert list(docking.OUT_FIELDS) == OUT_FIELDS
    row = docking.result_row(_item())
    assert list(row) == OUT_FIELDS


def test_fast_failure_keeps_reason_in_note(
    monkeypatch: pytest.MonkeyPatch, docking
) -> None:
    monkeypatch.setattr(docking, "run", lambda cmd, log: 1)
    item = _item(review_required=True, claim_eligible=False,
                 safety_reason="decision:FLAG_HIGH")
    row = docking.dock_item(item, 1)

    assert row["note"] == "fast_preset_failed"
    assert row["review_required"] is True
    assert row["claim_eligible"] is False
    assert row["safety_reason"] == "decision:FLAG_HIGH"
    assert row["daina_rank"] == ""
    assert row["autodock_kcal_mol"] == "" and row["gnina_cnn_affinity"] == ""


def test_top256_outside_success_keeps_scores_and_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, docking
) -> None:
    run_dir = tmp_path / "results/runs/run1"
    _write_daina(run_dir, [{"target_id": "P99999", "daina_rank": "7",
                            "ligand_smiles": "CCC"}])
    pair_dir = run_dir / "03_targets/funnel_pair"

    def fake_run(cmd: list[str], log: Path) -> int:
        joined = " ".join(cmd)
        if "stage3_autodock_run.py" in joined:
            with (pair_dir / "autodock_pair.tsv").open("w", newline="", encoding="utf-8") as fh:
                fh.write("target_id\tvina_score\nP11111\t-7.3\n")
        if "stage3_gnina_rescore.py" in joined:
            with (pair_dir / "gnina_pair.tsv").open("w", newline="", encoding="utf-8") as fh:
                fh.write("target_id\tcnn_affinity\nP11111\t0.81\n")
        return 0

    monkeypatch.setattr(docking, "run", fake_run)
    row = docking.dock_item(_item(), 1)

    assert row["daina_rank"] == "1"
    assert row["autodock_kcal_mol"] == "-7.3"
    assert row["gnina_cnn_affinity"] == "0.81"
    assert row["note"] == "synthesized_rank_outside_daina_top256"


def test_failed_step_keeps_scores_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, docking
) -> None:
    run_dir = tmp_path / "results/runs/run1"
    _write_daina(run_dir, [{"target_id": "P11111", "daina_rank": "3",
                            "ligand_smiles": "CCC"}])
    pair_dir = run_dir / "03_targets/funnel_pair"

    def fake_run(cmd: list[str], log: Path) -> int:
        if "stage3_autodock_run.py" in " ".join(cmd):
            # 실패한 단계가 점수 파일을 남겨도 이번 행에는 수용하지 않는다.
            with (pair_dir / "autodock_pair.tsv").open("w", newline="", encoding="utf-8") as fh:
                fh.write("target_id\tvina_score\nP11111\t-9.9\n")
            return 3
        return 0

    monkeypatch.setattr(docking, "run", fake_run)
    row = docking.dock_item(_item(), 1)

    assert row["note"].startswith("step_failed:")
    assert "autodock" in row["note"]
    assert row["daina_rank"] == "3"
    assert row["autodock_kcal_mol"] == ""
    assert row["gnina_cnn_affinity"] == ""
