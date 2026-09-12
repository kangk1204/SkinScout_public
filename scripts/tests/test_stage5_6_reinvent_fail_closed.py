"""Regression tests for the fail-closed Stage 5.6 REINVENT contract gate."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[2]


def write_input_sdf(path: Path) -> None:
    molecule = Chem.MolFromSmiles("CCO")
    assert molecule is not None
    writer = Chem.SDWriter(str(path))
    writer.write(molecule)
    writer.close()


def run_contract_gate(
    tmp_path: Path,
    *,
    extra_args: list[str] | None = None,
    consensus: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    input_sdf = tmp_path / "ligand.sdf"
    write_input_sdf(input_sdf)
    default_consensus = tmp_path / "consensus.json"
    default_consensus.write_text(
        '{"P12345": {"confirmed_atoms": [0], "claim_eligible": true}}\n'
    )
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage5_6_reinvent_config.py"),
            "--in-sdf",
            str(input_sdf),
            "--consensus",
            str(consensus or default_consensus),
            "--out-status",
            str(tmp_path / "contract_status.json"),
            "--out-smi",
            str(tmp_path / "generated.smi"),
            *(extra_args or []),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_claim_capable_execution_is_blocked_without_outputs(tmp_path: Path) -> None:
    status = tmp_path / "contract_status.json"
    smiles = tmp_path / "generated.smi"
    status.write_text("stale\n")
    smiles.write_text("stale\n")

    result = run_contract_gate(tmp_path)

    assert result.returncode != 0
    assert "Stage 5.6 is unavailable for claim-capable execution" in result.stderr
    assert "custom scoring plugin bundle is absent" in result.stderr
    assert "ligand atom order is not mapped" in result.stderr
    assert not status.exists()
    assert not smiles.exists()


def test_explicit_diagnostic_mode_emits_blocked_status_and_empty_smiles(
    tmp_path: Path,
) -> None:
    result = run_contract_gate(tmp_path, extra_args=["--allow-empty-output"])

    assert result.returncode == 0, result.stderr
    payload = json.loads((tmp_path / "contract_status.json").read_text())
    assert payload["execution_status"] == "blocked"
    assert payload["claim_eligible"] is False
    assert payload["diagnostic_only"] is True
    assert len(payload["blockers"]) == 3
    assert (tmp_path / "generated.smi").read_text() == ""


def test_missing_consensus_fails_before_diagnostic_outputs(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"

    result = run_contract_gate(
        tmp_path,
        extra_args=["--allow-empty-output"],
        consensus=missing,
    )

    assert result.returncode != 0
    assert "Pose-supported interaction consensus is required" in result.stderr
    assert not (tmp_path / "contract_status.json").exists()
    assert not (tmp_path / "generated.smi").exists()


def test_nonpositive_generation_count_is_rejected(tmp_path: Path) -> None:
    result = run_contract_gate(tmp_path, extra_args=["--num-generated", "0"])

    assert result.returncode != 0
    assert "--num-generated must be positive" in result.stderr
    assert not (tmp_path / "contract_status.json").exists()
    assert not (tmp_path / "generated.smi").exists()


def test_source_contains_no_invalid_reinvent_cli_or_component_contract() -> None:
    source = (ROOT / "scripts/stage5_6_reinvent_config.py").read_text()

    assert "--csv-result" not in source
    assert '[stage]' not in source
    assert '[[stage.scoring.component]]' not in source
    assert 'endpoint = "boltz2_affinity_proxy"' not in source
    assert 'endpoint = "admet_ai_composite"' not in source
