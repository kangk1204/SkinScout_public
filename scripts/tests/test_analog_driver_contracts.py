"""C18 regression: analog session drivers must terminate, use repository
entrypoints, and key docking work by ligand+target instead of run_id alone."""

from __future__ import annotations

import csv
import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analog_funnel"))


def _chain():
    return importlib.import_module("run_safety_chain")


def _docking():
    return importlib.import_module("pair_docking")


def test_zero_reports_zero_processes_terminates_with_failure(tmp_path: Path) -> None:
    chain = _chain()
    gen = tmp_path / "gen"
    gen.mkdir()
    with pytest.raises(chain.GenerationFailed, match="0 reports"):
        chain.wait_for_generation(
            gen=gen,
            expected_seeds=13,
            deadline_seconds=3600.0,
            poll_seconds=0.0,
            process_counter=lambda: 0,
            clock=lambda: 0.0,
            sleep=lambda _seconds: None,
        )


def test_generation_deadline_fails_instead_of_waiting_forever(tmp_path: Path) -> None:
    chain = _chain()
    gen = tmp_path / "gen"
    (gen / "seedA").mkdir(parents=True)
    (gen / "seedA" / "substitute_report.json").write_text("{}", encoding="utf-8")
    ticks = iter([0.0, 5.0, 15.0, 25.0])
    with pytest.raises(chain.GenerationFailed, match="deadline"):
        chain.wait_for_generation(
            gen=gen,
            expected_seeds=13,
            deadline_seconds=10.0,
            poll_seconds=0.0,
            process_counter=lambda: 1,
            clock=lambda: next(ticks),
            sleep=lambda _seconds: None,
        )


def test_partial_generation_finishes_when_no_process_remains(tmp_path: Path) -> None:
    chain = _chain()
    gen = tmp_path / "gen"
    (gen / "seedA").mkdir(parents=True)
    (gen / "seedA" / "substitute_report.json").write_text("{}", encoding="utf-8")
    chain.wait_for_generation(
        gen=gen,
        expected_seeds=13,
        deadline_seconds=0.0,
        poll_seconds=0.0,
        process_counter=lambda: 0,
        clock=lambda: 0.0,
        sleep=lambda _seconds: None,
    )


def test_selection_executes_repository_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    chain = _chain()
    assert chain.SELECT_SCRIPT == ROOT / "scripts/analog_funnel/select_candidates.py"
    assert chain.SELECT_SCRIPT.is_file()
    captured: dict = {}

    def fake_run(command: list[str], **kwargs: object) -> object:
        captured["command"] = command
        captured["kwargs"] = kwargs

        class Result:
            returncode = 0

        return Result()

    monkeypatch.setattr(chain.subprocess, "run", fake_run)
    chain.run_selection()
    command = " ".join(captured["command"])
    assert str(chain.SELECT_SCRIPT) in command
    assert "/tmp/opencode/analog_funnel_select.py" not in command
    assert captured["kwargs"]["cwd"] == ROOT


def test_chain_manifest_records_executed_script_and_config_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chain = _chain()
    monkeypatch.setattr(chain, "FUNNEL", tmp_path)
    monkeypatch.setattr(chain, "CHAIN_MANIFEST", tmp_path / "chain_manifest.json")
    chain.write_chain_manifest("complete", selection={"env": "cosmax-base"})
    payload = json.loads((tmp_path / "chain_manifest.json").read_text(encoding="utf-8"))
    hashed = {entry["path"] for entry in payload["code"]}
    assert str(ROOT / "scripts/analog_funnel/select_candidates.py") in hashed
    assert str(ROOT / "workflow" / "Snakefile") in hashed
    assert all(len(entry["sha256"]) == 64 for entry in payload["code"])


def test_same_ligand_under_two_targets_survives_docking_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    docking = _docking()
    smiles = "CCO"
    run_id = docking.rs._auto_run_id_from_smiles(smiles)
    shortlist = tmp_path / "shortlist.csv"
    with shortlist.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["seed_dir", "candidate_id", "mmr_rank", "smiles", "funnel_score"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "seed_dir": "seedA",
                "candidate_id": "candidateA",
                "mmr_rank": "1",
                "smiles": smiles,
                "funnel_score": "1.0",
            }
        )
        writer.writerow(
            {
                "seed_dir": "seedB",
                "candidate_id": "candidateB",
                "mmr_rank": "1",
                "smiles": smiles,
                "funnel_score": "0.9",
            }
        )
    safety = tmp_path / "safety_status.csv"
    with safety.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "seed_dir", "candidate_id", "run_id", "returncode", "skin_sens", "cosmetic",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "seed_dir": "seedA",
                "candidate_id": "candidateA",
                "run_id": run_id,
                "returncode": "0",
                "skin_sens": "PASS",
                "cosmetic": "negative",
            }
        )
    monkeypatch.setattr(docking, "SHORTLIST", shortlist)
    monkeypatch.setattr(docking, "SAFETY", [safety])
    monkeypatch.setattr(docking, "PER_SEED", 5)
    result = docking.survivors({"seedA": "P11111", "seedB": "P22222"})
    assert {(item["run_id"], item["uniprot"]) for item in result} == {
        (run_id, "P11111"),
        (run_id, "P22222"),
    }


def test_docking_manifest_records_work_keys_and_script_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    docking = _docking()
    manifest = tmp_path / "dock_manifest.json"
    monkeypatch.setattr(docking, "FUNNEL", tmp_path)
    monkeypatch.setattr(docking, "DOCK_MANIFEST", manifest)
    items = [
        {
            "seed_dir": "seedA",
            "candidate_id": "candidateA",
            "run_id": "run1",
            "uniprot": "P11111",
        }
    ]
    docking.write_dock_manifest("selected", items, {"work_key": "ligand_identity+target"})
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["work_items"] == [
        {
            "seed_dir": "seedA",
            "candidate_id": "candidateA",
            "run_id": "run1",
            "uniprot": "P11111",
        }
    ]
    assert payload["config"]["work_key"] == "ligand_identity+target"
    assert str(ROOT / "scripts/analog_funnel/pair_docking.py") in {
        entry["path"] for entry in payload["code"]
    }
