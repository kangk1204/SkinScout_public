"""C01–C04 회귀: 게이트 제약, fail-closed 안전성, 도킹 신선도/경로 계약."""

from __future__ import annotations

import csv
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analog_funnel"))


def test_repair_module_imports() -> None:
    """C04: os import 누락으로 import 즉시 실패하던 회귀."""
    importlib.import_module("pair_docking_repair")


def test_docking_box_dir_matches_workflow_config() -> None:
    import yaml

    module = importlib.import_module("pair_docking")
    config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text(encoding="utf-8"))
    assert module.docking_box_dir() == ROOT / config["paths"]["docking_boxes"]


def test_safe_decisions_exclude_blank_and_unknown() -> None:
    """C02: 빈 값/UNKNOWN은 도킹 생존 조건이 아니다."""
    for name in ("pair_docking", "validate_generated"):
        module = importlib.import_module(name)
        assert module.SAFE_DECISIONS == {"PASS", "FLAG_HIGH"}
        for bad in ("", "UNKNOWN", "HALT", "REVIEW", "MISSING"):
            assert bad not in module.SAFE_DECISIONS


def test_gate_rejects_alerts_and_missing_evidence() -> None:
    selection = importlib.import_module("select_candidates")
    good = {
        "evidence_tier": "direct_retained",
        "binding_retained": "True",
        "structural_alert_count": "0",
        "property_suitability_score": "0.9",
    }
    bad_alert = {**good, "structural_alert_count": "2"}
    bad_evidence = {**good, "binding_retained": "False", "evidence_tier": "proxy_only"}
    assert selection.gate(good) == (True, "")
    assert selection.gate(bad_alert)[0] is False
    assert selection.gate(bad_evidence)[0] is False


def test_selection_excludes_gate_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """C01: 탈락 후보는 고득점이어도 shortlist를 점유하지 못한다."""
    selection = importlib.import_module("select_candidates")
    seed = tmp_path / "gen" / "seedA"
    seed.mkdir(parents=True)
    columns = [
        "candidate_id", "smiles", "inchikey", "evidence_tier", "binding_retained",
        "structural_alert_count", "property_suitability_score",
        "pharmacophore_preservation_score", "binding_support_score", "safety_triage_score",
    ]
    rows = [
        {"candidate_id": "P1", "smiles": "CCO", "inchikey": "K1", "evidence_tier": "direct_retained",
         "binding_retained": "True", "structural_alert_count": "0", "property_suitability_score": "0.9",
         "pharmacophore_preservation_score": "0.4", "binding_support_score": "0.5", "safety_triage_score": "0.9"},
        {"candidate_id": "R1", "smiles": "CCC", "inchikey": "K3", "evidence_tier": "proxy_only",
         "binding_retained": "False", "structural_alert_count": "2", "property_suitability_score": "0.1",
         "pharmacophore_preservation_score": "1.0", "binding_support_score": "1.0", "safety_triage_score": "1.0"},
    ]
    with (seed / "substitute_candidates.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setattr(selection, "GENROOT", tmp_path / "gen")
    monkeypatch.setattr(selection, "OUT", tmp_path / "shortlist.csv")
    monkeypatch.setattr(selection, "PER_SEED", 5)
    selection.main()
    out = list(csv.DictReader((tmp_path / "shortlist.csv").open(encoding="utf-8-sig")))
    assert [row["candidate_id"] for row in out] == ["P1"]
    rejected = list(csv.DictReader((tmp_path / "shortlist_rejected.csv").open(encoding="utf-8-sig")))
    assert [row["candidate_id"] for row in rejected] == ["R1"]


def test_selection_with_zero_eligible_writes_empty_shortlist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = importlib.import_module("select_candidates")
    seed = tmp_path / "gen" / "seedA"
    seed.mkdir(parents=True)
    columns = [
        "candidate_id", "smiles", "inchikey", "evidence_tier", "binding_retained",
        "structural_alert_count", "property_suitability_score",
        "pharmacophore_preservation_score", "binding_support_score", "safety_triage_score",
    ]
    with (seed / "substitute_candidates.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerow({"candidate_id": "R1", "smiles": "CCC", "inchikey": "K3",
                         "evidence_tier": "proxy_only", "binding_retained": "False",
                         "structural_alert_count": "2", "property_suitability_score": "0.1",
                         "pharmacophore_preservation_score": "1.0", "binding_support_score": "1.0",
                         "safety_triage_score": "1.0"})
    monkeypatch.setattr(selection, "GENROOT", tmp_path / "gen")
    monkeypatch.setattr(selection, "OUT", tmp_path / "shortlist.csv")
    selection.main()
    rows = list(csv.DictReader((tmp_path / "shortlist.csv").open(encoding="utf-8-sig")))
    assert rows == []
