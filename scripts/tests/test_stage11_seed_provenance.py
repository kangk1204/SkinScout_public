"""Regression tests for Stage 11 per-stage random-seed provenance (F43).

The repro pack used to publish environment variables (`BOLTZ_SEED`,
`REINVENT_SEED`) that no producer consumes, and omitted the seeds producers
actually wrote into their manifests. These tests pin the replacement: values
come from the run artifacts and producer constants, uncontrollable stochastic
stages say `uncontrolled`, and absent stages say `unavailable`.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def load_repro_pack_module():
    spec = importlib.util.spec_from_file_location(
        "stage11_repro_pack_for_seeds",
        ROOT / "scripts" / "stage11_repro_pack.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _write_seeded_artifacts(run_dir: Path, *, docking_seed: int, md_seeds: list[int]) -> None:
    _write(
        run_dir / "06_bioemu" / "ensemble_manifest.tsv",
        "target_id\tmedoid_pdb\nP1\tmedoid_00.pdb\n",
    )
    _write(
        run_dir / "06_bioemu" / "ensemble_consensus.tsv",
        "target_id\tdocking_seed\tbest_gnina\n"
        f"P1\t{docking_seed}\t1.0\n",
    )
    rows = "\n".join(
        f"P1\t{replica}\t{seed}"
        for replica, seed in enumerate(md_seeds, start=1)
    )
    _write(
        run_dir / "07_md" / "trajectory_index.tsv",
        f"target_id\treplica\tseed\n{rows}\n",
    )
    _write(
        run_dir / "05_6_analogs" / "reinvent4_status.json",
        json.dumps({"seed": 49242, "execution_status": "completed"}) + "\n",
    )
    _write(
        run_dir / "05_6_analogs" / "reinvent4_finetune_manifest.json",
        json.dumps({"seed": 7, "execution_status": "ready"}) + "\n",
    )


def test_seeded_stages_read_the_values_their_producers_recorded(tmp_path: Path) -> None:
    module = load_repro_pack_module()
    run_dir = tmp_path / "run"
    _write_seeded_artifacts(run_dir, docking_seed=77, md_seeds=[100, 101])

    stages = module._stage_seed_payload(run_dir)["stages"]

    assert stages["ensemble_dock"]["control"] == "seeded"
    assert stages["ensemble_dock"]["seed"] == 77
    assert stages["ensemble_dock"]["source"] == "run_artifact_ensemble_consensus_tsv"
    assert stages["gromacs_production"]["control"] == "seeded"
    assert stages["gromacs_production"]["seed"] == 100
    assert stages["gromacs_production"]["replica_seeds"] == [100, 101]
    assert stages["reinvent4_generation"]["seed"] == 49242
    assert stages["reinvent4_finetune"]["seed"] == 7
    assert stages["sidechain_reconstruction"]["control"] == "seeded"
    assert stages["sidechain_reconstruction"]["seed"] == module._producer_literal_int(
        ROOT / "scripts" / "stage6_bioemu.py", "RECONSTRUCTION_SEED"
    )


def test_changing_a_stage_seed_changes_the_provenance(tmp_path: Path) -> None:
    module = load_repro_pack_module()
    run_dir = tmp_path / "run"
    _write_seeded_artifacts(run_dir, docking_seed=5, md_seeds=[11, 12])
    first = module._stage_seed_payload(run_dir)

    _write_seeded_artifacts(run_dir, docking_seed=6, md_seeds=[21, 22])
    second = module._stage_seed_payload(run_dir)

    assert first != second
    assert first["stages"]["ensemble_dock"]["seed"] == 5
    assert second["stages"]["ensemble_dock"]["seed"] == 6
    assert first["stages"]["gromacs_production"]["replica_seeds"] == [11, 12]
    assert second["stages"]["gromacs_production"]["replica_seeds"] == [21, 22]
    assert (
        first["stages"]["ensemble_dock"]["artifact_sha256"]
        != second["stages"]["ensemble_dock"]["artifact_sha256"]
    )


def test_sidechain_provenance_follows_the_producer_constant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_repro_pack_module()
    run_dir = tmp_path / "run"
    _write(
        run_dir / "06_bioemu" / "ensemble_manifest.tsv",
        "target_id\tmedoid_pdb\nP1\tmedoid_00.pdb\n",
    )
    monkeypatch.setattr(
        module, "_producer_literal_int", lambda producer, name: 424242
    )

    stages = module._stage_seed_payload(run_dir)["stages"]

    assert stages["sidechain_reconstruction"]["seed"] == 424242


def test_stages_without_seed_control_are_uncontrolled(tmp_path: Path) -> None:
    module = load_repro_pack_module()

    stages = module._stage_seed_payload(tmp_path)["stages"]

    assert stages["boltz2"]["control"] == "uncontrolled"
    assert stages["boltz2"]["seed"] is None
    assert stages["boltz2"]["source"] == "producer_has_no_seed_option"
    assert stages["bioemu"]["control"] == "uncontrolled"
    assert stages["bioemu"]["seed"] is None


def test_absent_stage_artifacts_are_unavailable_not_uncontrolled(tmp_path: Path) -> None:
    module = load_repro_pack_module()

    stages = module._stage_seed_payload(tmp_path)["stages"]

    for stage in (
        "etkdg",
        "ensemble_dock",
        "gromacs_production",
        "reinvent4_generation",
        "reinvent4_finetune",
        "sidechain_reconstruction",
    ):
        assert stages[stage]["control"] == "unavailable", stage
        assert stages[stage]["seed"] is None, stage


def test_environment_seed_variables_no_longer_change_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_repro_pack_module()
    baseline = module._stage_seed_payload(tmp_path)

    monkeypatch.setenv("BOLTZ_SEED", "12345")
    monkeypatch.setenv("REINVENT_SEED", "54321")

    assert module._stage_seed_payload(tmp_path) == baseline


@pytest.mark.parametrize("value", ["", "x", "-2"])
def test_malformed_seed_columns_fail_closed(tmp_path: Path, value: str) -> None:
    module = load_repro_pack_module()
    run_dir = tmp_path / "run"
    _write(
        run_dir / "07_md" / "trajectory_index.tsv",
        f"target_id\treplica\tseed\nP1\t1\t{value}\n",
    )

    with pytest.raises(SystemExit, match="seed"):
        module._stage_seed_payload(run_dir)


def test_conflicting_docking_seeds_fail_closed(tmp_path: Path) -> None:
    module = load_repro_pack_module()
    run_dir = tmp_path / "run"
    _write(
        run_dir / "06_bioemu" / "ensemble_consensus.tsv",
        "target_id\tdocking_seed\nP1\t1\nP2\t2\n",
    )

    with pytest.raises(SystemExit, match="conflicting docking seeds"):
        module._stage_seed_payload(run_dir)
