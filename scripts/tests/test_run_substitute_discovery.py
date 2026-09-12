from __future__ import annotations

import csv
import hashlib
import json
import shlex
import sys
from pathlib import Path

import pandas as pd
import pytest
from rdkit import Chem

from scripts import run_substitute_discovery as runner


TARGET = "P12345"
OTHER_TARGET = "Q99999"
PARENT = "Oc1ccccc1"


def _inchikey(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    return Chem.MolToInchiKey(molecule)


def _write_anchor_map(path: Path, smiles: str, target_id: str = TARGET) -> None:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    canonical_molecule = Chem.MolFromSmiles(canonical)
    assert canonical_molecule is not None
    atom_count = canonical_molecule.GetNumAtoms()
    path.parent.mkdir(parents=True, exist_ok=True)
    parent_sdf = path.parent / "anchor_parent.sdf"
    consensus = path.parent / "anchor_consensus.json"
    boltz_report = path.parent / "anchor_boltz.tsv"
    complex_pdb = path.parent / "anchor_complex.pdb"
    parent_sdf.write_text("parent sdf fixture\n")
    consensus.write_text("{}\n")
    boltz_report.write_text("target_id\tkept\n")
    complex_pdb.write_text("MODEL\nENDMDL\n")

    def fingerprint(source: Path) -> dict[str, object]:
        return {
            "path": str(source.resolve()),
            "bytes": source.stat().st_size,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }

    parent_fingerprint = fingerprint(parent_sdf)
    consensus_fingerprint = fingerprint(consensus)
    boltz_fingerprint = fingerprint(boltz_report)
    complex_fingerprint = fingerprint(complex_pdb)
    path.write_text(
        json.dumps(
            {
                "schema_version": "skinscout.interaction_anchor_map.v1",
                "consensus_sha256": consensus_fingerprint["sha256"],
                "boltz_report_sha256": boltz_fingerprint["sha256"],
                "sources": {
                    "parent_sdf": parent_fingerprint,
                    "consensus_json": consensus_fingerprint,
                    "boltz_report": boltz_fingerprint,
                },
                "parent": {
                    "sdf_sha256": parent_fingerprint["sha256"],
                    "canonical_smiles": canonical,
                    "inchikey": Chem.MolToInchiKey(canonical_molecule),
                    "atom_count": atom_count,
                    "heavy_atom_count": atom_count,
                    "canonical_atom_count": atom_count,
                },
                "targets": {
                    target_id: {
                        "target_id": target_id,
                        "source_coordinate_system": (
                            "boltz_complex_ligand_atom_order_0_based"
                        ),
                        "mapped_coordinate_system": (
                            "parent_sdf_atom_order_0_based"
                        ),
                        "confirmed_complex_atoms": [0],
                        "confirmed_parent_atoms": [0],
                        "canonical_coordinate_system": (
                            "canonical_smiles_atom_order_0_based"
                        ),
                        "confirmed_canonical_parent_atoms": [0],
                        "atom_mapping": [
                            {
                                "complex_atom_index": index,
                                "parent_atom_index": index,
                            }
                            for index in range(atom_count)
                        ],
                        "canonical_atom_mapping": [
                            {
                                "canonical_atom_index": index,
                                "parent_atom_index": index,
                            }
                            for index in range(atom_count)
                        ],
                        "mapping_status": "mapped",
                        "mapping_confidence": "high",
                        "complex_pdb": complex_fingerprint["path"],
                        "complex_pdb_bytes": complex_fingerprint["bytes"],
                        "complex_pdb_sha256": complex_fingerprint["sha256"],
                        "claim_eligible": True,
                    }
                },
                "claim_eligible": True,
                "analog_pose_verified": False,
            }
        )
    )


def _write_parent_run(run_dir: Path, run_id: str) -> None:
    ranking = run_dir / "03_targets" / "ranked_targets_v3_with_efficacy.csv"
    ranking.parent.mkdir(parents=True)
    with ranking.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["target_id", "final_score"])
        writer.writeheader()
        writer.writerow({"target_id": TARGET, "final_score": 0.9})
        writer.writerow({"target_id": OTHER_TARGET, "final_score": 0.8})
    summary = {
        "schema_version": "skinscout.run_summary.v1",
        "run_id": run_id,
        "preset": "target-id",
        "mode": "fast",
        "compound": {
            "input_type": "smiles",
            "input_smiles": PARENT,
            "input_canonical_smiles": PARENT,
            "canonical_smiles": PARENT,
        },
        "artifacts": {
            "target_ranking": "03_targets/ranked_targets_v3_with_efficacy.csv"
        },
        "target_prediction": {
            "top_targets": [{"target_id": TARGET}, {"target_id": OTHER_TARGET}]
        },
    }
    summary_path = run_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary))
    summary_md = run_dir / "run_summary.md"
    summary_md.write_text("# Parent run fixture\n")
    verification_log = run_dir / "run_verification.log"
    command = [
        sys.executable,
        "scripts/verify_run_outputs.py",
        "--run-dir",
        str(run_dir),
        "--preset",
        "target-id",
        "--mode",
        "fast",
        "--target-metadata",
        "data/target_metadata.csv",
        "--json-out",
        str(run_dir / ".run_verification_payload.json"),
    ]
    check = {
        "name": "parent target ranking",
        "status": "ok",
        "path": str(ranking),
        "detail": "fixture",
    }
    verification_log.write_text(
        "\n".join(
            [
                "$ " + shlex.join(command),
                "SkinScout run output verification: ok",
                f"run_dir={run_dir} preset=target-id mode=fast",
                f"[ok] {check['name']}: {check['path']} - {check['detail']}",
                "",
            ]
        )
    )

    def verified_artifact(path: Path, name: str) -> dict[str, object]:
        data = path.read_bytes()
        return {
            "name": name,
            "path": str(path),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    verifier_payload = {
        "schema_version": "skinscout.run_output_verification.v1",
        "status": "ok",
        "run_dir": str(run_dir),
        "preset": "target-id",
        "mode": "fast",
        "checks": [check],
    }
    (run_dir / "run_verification.json").write_text(
        json.dumps(
            {
                "schema_version": "skinscout.run_verification.v1",
                "status": "ok",
                "verified_at_utc": "2026-09-05T00:00:00Z",
                "command": command,
                "returncode": 0,
                "stdout_log": "run_verification.log",
                "run_dir": str(run_dir),
                "preset": "target-id",
                "mode": "fast",
                "input_provenance": {
                    "input_type": "smiles",
                    "input_smiles": PARENT,
                    "input_canonical_smiles": PARENT,
                    "input_sdf": None,
                },
                "diagnostic_nonclaimable_reasons": [],
                "verifier_status": "ok",
                "checks": [check],
                "verified_artifacts": [
                    verified_artifact(summary_path, "run_summary_json"),
                    verified_artifact(summary_md, "run_summary_md"),
                    verified_artifact(verification_log, "run_verification_log"),
                    verified_artifact(ranking, "target_ranking"),
                ],
                "verifier_payload": verifier_payload,
            }
        )
    )


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    library = tmp_path / "cosing.parquet"
    activity = tmp_path / "activity.parquet"
    pd.DataFrame(
        [
            {"inci_name": "Anisole", "functions": "PERFUMING", "smiles": "COc1ccccc1"},
            {"inci_name": "Ethoxybenzene", "functions": "PERFUMING", "smiles": "CCOc1ccccc1"},
        ]
    ).to_parquet(library)
    pd.DataFrame(
        [
            {
                "uniprot": TARGET,
                "ligand_smiles": PARENT,
                "ligand_inchikey": _inchikey(PARENT),
                "affinity_value": 100.0,
                "affinity_unit": "nM",
                "relation": "=",
                "censor": False,
                "source_db": "FixtureDB",
                "source_release": "1",
                "source_doi": "10.1000/parent",
                "source_pmid": "1",
                "affinity_type": "Ki",
            },
            {
                "uniprot": TARGET,
                "ligand_smiles": "COc1ccccc1",
                "ligand_inchikey": _inchikey("COc1ccccc1"),
                "affinity_value": 200.0,
                "affinity_unit": "nM",
                "relation": "=",
                "censor": False,
                "source_db": "FixtureDB",
                "source_release": "1",
                "source_doi": "10.1000/anisole",
                "source_pmid": "2",
                "affinity_type": "Ki",
            },
        ]
    ).to_parquet(activity)
    return library, activity


def test_parent_command_uses_verified_target_id_contract(tmp_path: Path) -> None:
    args = runner.parse_args(
        [
            "--smiles",
            PARENT,
            "--run-id",
            "parent_case",
            "--mode",
            "fast",
        ]
    )

    command = runner.build_parent_command(args)

    assert command[command.index("--preset") + 1] == "target-id"
    assert command[command.index("--evidence-mode") + 1] == "evidence"
    assert command[command.index("--smiles") + 1] == PARENT
    assert "--online-safety-readiness" in command
    assert any(
        item.startswith("paths.results_root=")
        for item in command[command.index("--extra-config") + 1 :]
    )
    assert "--skip-output-verification" not in command


def test_anchor_command_targets_stage5_5_with_matching_run_config(
    tmp_path: Path,
) -> None:
    args = runner.parse_args(
        [
            "--smiles",
            PARENT,
            "--run-id",
            "anchor_build_case",
            "--mode",
            "fast",
            "--context-profile",
            "barrier",
            "--results-root",
            str(tmp_path / "runs"),
            "--build-interaction-anchors",
        ]
    )

    command = runner.build_anchor_command(args)

    assert "pharmacophore_anchor_map" in command
    assert "--use-conda" in command
    config = command[command.index("--config") + 1 :]
    assert "run_id=anchor_build_case" in config
    assert "mode=fast" in config
    assert f"compound_smiles={PARENT}" in config
    assert any("default_context_profile" in item and "barrier" in item for item in config)
    assert any("paths" in item and str(tmp_path / "runs") in item for item in config)


def test_target_selection_uses_top_or_validated_explicit_target(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_parent_run(run_dir, "run")
    summary = json.loads((run_dir / "run_summary.json").read_text())

    assert runner.select_target(run_dir, summary, None) == (
        TARGET,
        "parent_top_prediction",
    )
    assert runner.select_target(run_dir, summary, OTHER_TARGET) == (
        OTHER_TARGET,
        "user_selected_ranked_target",
    )
    with pytest.raises(SystemExit, match="absent from the parent target ranking"):
        runner.select_target(run_dir, summary, "P00000")


def test_discovery_command_forwards_explicit_alias_evidence(tmp_path: Path) -> None:
    alias_evidence = tmp_path / "aliases.parquet"
    alias_evidence.write_bytes(b"fixture")
    args = runner.parse_args(
        [
            "--smiles",
            PARENT,
            "--run-id",
            "alias_case",
            "--alias-evidence",
            str(alias_evidence),
        ]
    )

    command = runner.build_discovery_command(
        args,
        parent_smiles=PARENT,
        target_id=TARGET,
        target_basis="parent_top_prediction",
        out_dir=tmp_path / "out",
    )

    assert command[command.index("--alias-evidence") + 1] == str(alias_evidence)


def test_discovery_command_forwards_interaction_anchor_map(tmp_path: Path) -> None:
    anchor_map = tmp_path / "anchors.json"
    anchor_map.write_text("fixture")
    args = runner.parse_args(
        ["--smiles", PARENT, "--run-id", "anchor_command_case"]
    )

    command = runner.build_discovery_command(
        args,
        parent_smiles=PARENT,
        target_id=TARGET,
        target_basis="parent_top_prediction",
        out_dir=tmp_path / "out",
        interaction_anchor_map=anchor_map,
    )

    assert command[command.index("--interaction-anchor-map") + 1] == str(anchor_map)


def test_reused_parent_run_produces_hash_bound_substitute_manifest(
    tmp_path: Path,
) -> None:
    run_id = "substitute_case"
    results_root = tmp_path / "runs"
    run_dir = results_root / run_id
    _write_parent_run(run_dir, run_id)
    library, activity = _write_inputs(tmp_path)
    args = runner.parse_args(
        [
            "--smiles",
            PARENT,
            "--run-id",
            run_id,
            "--reuse-existing-parent-run",
            "--results-root",
            str(results_root),
            "--candidate-library",
            str(library),
            "--activity-evidence",
            str(activity),
            "--max-candidates",
            "10",
        ]
    )

    manifest = runner.run(args)

    assert manifest["schema_version"] == runner.SCHEMA_VERSION
    assert manifest["claimable"] is False
    assert manifest["hypothesis_only"] is True
    assert manifest["wet_lab_required"] is True
    assert manifest["commands"]["parent_reused"] is True
    assert manifest["target"] == {
        "target_id": TARGET,
        "selection_basis": "parent_top_prediction",
    }
    assert manifest["candidate_count"] >= 1
    assert manifest["candidate_summary"]["cosing_candidates"] >= 1
    assert manifest["selection_strategy"]["selected_counts"]["all"] == manifest[
        "candidate_count"
    ]
    out_dir = run_dir / "05_6_substitutes"
    stored = json.loads((out_dir / "substitute_run_manifest.json").read_text())
    assert stored == manifest
    for item in manifest["outputs"].values():
        assert len(item["sha256"]) == 64


def test_reused_parent_run_must_match_requested_compound_and_mode(
    tmp_path: Path,
) -> None:
    run_id = "wrong_parent_identity"
    results_root = tmp_path / "runs"
    _write_parent_run(results_root / run_id, run_id)

    wrong_compound = runner.parse_args([
        "--smiles", "CCN", "--run-id", run_id,
        "--reuse-existing-parent-run", "--results-root", str(results_root),
    ])
    with pytest.raises(SystemExit, match="input SMILES does not match this run"):
        runner.run(wrong_compound)

    wrong_mode = runner.parse_args([
        "--smiles", PARENT, "--run-id", run_id, "--mode", "comprehensive",
        "--reuse-existing-parent-run", "--results-root", str(results_root),
    ])
    with pytest.raises(SystemExit, match="mode does not match"):
        runner.run(wrong_mode)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda record: record.pop("schema_version"), "invalid schema_version"),
        (lambda record: record.__setitem__("returncode", False), "returncode is not 0"),
        (lambda record: record.__setitem__("verifier_status", "failed"), "verifier_status"),
        (lambda record: record.__setitem__("checks", []), "checks must be a non-empty list"),
        (
            lambda record: record.__setitem__(
                "diagnostic_nonclaimable_reasons", ["diagnostic fixture"]
            ),
            "diagnostic non-claimable reasons",
        ),
    ],
)
def test_reused_parent_run_requires_complete_claim_quality_verification(
    tmp_path: Path,
    mutate: object,
    message: str,
) -> None:
    run_id = "invalid_parent_verification"
    run_dir = tmp_path / "runs" / run_id
    _write_parent_run(run_dir, run_id)
    verification_path = run_dir / "run_verification.json"
    verification = json.loads(verification_path.read_text())
    mutate(verification)  # type: ignore[operator]
    verification_path.write_text(json.dumps(verification))
    args = runner.parse_args(
        [
            "--smiles",
            PARENT,
            "--run-id",
            run_id,
            "--reuse-existing-parent-run",
            "--results-root",
            str(tmp_path / "runs"),
        ]
    )

    with pytest.raises(SystemExit, match=message):
        runner._validate_parent_run(run_dir, args)


def test_reused_parent_run_rejects_tampered_consumed_ranking(tmp_path: Path) -> None:
    run_id = "tampered_parent_ranking"
    run_dir = tmp_path / "runs" / run_id
    _write_parent_run(run_dir, run_id)
    ranking = run_dir / "03_targets/ranked_targets_v3_with_efficacy.csv"
    ranking.write_text("target_id,final_score\nQ99999,1.0\n")
    args = runner.parse_args(
        [
            "--smiles",
            PARENT,
            "--run-id",
            run_id,
            "--reuse-existing-parent-run",
            "--results-root",
            str(tmp_path / "runs"),
        ]
    )

    with pytest.raises(SystemExit, match="artifact bytes changed: target_ranking"):
        runner._validate_parent_run(run_dir, args)


def test_substitute_run_id_rejects_path_traversal() -> None:
    with pytest.raises(SystemExit, match="run_id must use only"):
        runner.parse_args(["--smiles", PARENT, "--run-id", "../escape"])


def test_relative_sdf_is_normalized_once_for_parent_anchor_and_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caller_dir = tmp_path / "caller"
    caller_dir.mkdir()
    sdf = caller_dir / "compound.sdf"
    sdf.write_text("caller molecule fixture\n")
    execution_dir = tmp_path / "execution"
    execution_dir.mkdir()
    (execution_dir / "compound.sdf").write_text("different molecule fixture\n")
    monkeypatch.chdir(caller_dir)
    args = runner.parse_args(
        ["--sdf", "compound.sdf", "--run-id", "relative_sdf",
         "--results-root", str(tmp_path / "runs"), "--build-interaction-anchors"]
    )

    def run_parent(command: list[str], label: str) -> None:
        assert label == "parent-target-run"
        assert command[command.index("--sdf") + 1] == str(sdf.resolve())
        monkeypatch.chdir(execution_dir)

    def check_provenance(run_dir: Path, requested: object) -> None:
        assert requested is args
        assert args.sdf == sdf.resolve()
        assert args.sdf.read_bytes() == b"caller molecule fixture\n"
        assert f"compound_sdf={sdf.resolve()}" in runner.build_anchor_command(args)
        raise RuntimeError("stop after checking input provenance")

    monkeypatch.setattr(runner, "_run_command", run_parent)
    monkeypatch.setattr(runner, "_validate_parent_run", check_provenance)
    with pytest.raises(RuntimeError, match="stop after checking input provenance"):
        runner.run(args)


def test_reused_parent_run_auto_uses_target_conditioned_anchor_map(
    tmp_path: Path,
) -> None:
    run_id = "anchor_substitute_case"
    results_root = tmp_path / "runs"
    run_dir = results_root / run_id
    _write_parent_run(run_dir, run_id)
    anchor_map = run_dir / "05_pharmacophore" / "interaction_anchor_map.json"
    _write_anchor_map(anchor_map, PARENT)
    library, activity = _write_inputs(tmp_path)
    args = runner.parse_args(
        [
            "--smiles",
            PARENT,
            "--run-id",
            run_id,
            "--reuse-existing-parent-run",
            "--results-root",
            str(results_root),
            "--candidate-library",
            str(library),
            "--activity-evidence",
            str(activity),
            "--max-candidates",
            "10",
        ]
    )

    manifest = runner.run(args)

    assert manifest["interaction_anchor_map"]["path"] == str(anchor_map)
    report = json.loads(
        (run_dir / "05_6_substitutes" / "substitute_report.json").read_text()
    )
    assert report["target"]["interaction_anchor_conditioned"] is True
    assert all(
        candidate["target_conditioned_anchor_score"] is not None
        for candidate in report["candidates"]
    )
    assert all(
        isinstance(candidate["target_conditioned_mapping_ambiguous"], bool)
        for candidate in report["candidates"]
    )


def test_requested_anchor_build_runs_before_target_conditioned_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "built_anchor_substitute_case"
    results_root = tmp_path / "runs"
    run_dir = results_root / run_id
    _write_parent_run(run_dir, run_id)
    library, activity = _write_inputs(tmp_path)
    args = runner.parse_args(
        [
            "--smiles",
            PARENT,
            "--run-id",
            run_id,
            "--reuse-existing-parent-run",
            "--build-interaction-anchors",
            "--results-root",
            str(results_root),
            "--candidate-library",
            str(library),
            "--activity-evidence",
            str(activity),
            "--max-candidates",
            "10",
        ]
    )
    real_run_command = runner._run_command
    labels: list[str] = []

    def fake_run_command(
        command: list[str],
        label: str,
        *,
        env: dict[str, str] | None = None,
    ) -> None:
        labels.append(label)
        if label == "target-conditioned-anchors":
            _write_anchor_map(
                run_dir / "05_pharmacophore" / "interaction_anchor_map.json",
                PARENT,
            )
            return
        real_run_command(command, label, env=env)

    monkeypatch.setattr(runner, "_run_command", fake_run_command)

    manifest = runner.run(args)

    assert labels == ["target-conditioned-anchors", "substitute-discovery"]
    assert manifest["commands"]["anchors_requested"] is True
    assert "pharmacophore_anchor_map" in manifest["commands"]["anchors"]
    assert manifest["interaction_anchor_map"] is not None


def test_discovery_mode_is_rejected_before_parent_execution(tmp_path: Path) -> None:
    args = runner.parse_args(
        [
            "--smiles",
            PARENT,
            "--run-id",
            "bad_mode",
            "--evidence-mode",
            "discovery",
        ]
    )

    with pytest.raises(SystemExit, match="requires evidence mode"):
        runner.run(args)


def test_report_validation_requires_full_claim_boundary(tmp_path: Path) -> None:
    report_path = tmp_path / "substitute_report.json"
    payload = {
        "schema_version": runner.DISCOVERY_SCHEMA,
        "run_id": "claim_case",
        "target": {"target_id": TARGET},
        "thresholds": {"max_direct_pactivity_loss_for_retention": 0.5},
        "claimable": False,
        "hypothesis_only": True,
        "wet_lab_required": True,
        "selection_strategy": {
            "mode": "global_priority",
            "selected_counts": {
                "all": 1,
                "target_activity": 0,
                "cosmetic_material": 0,
                "strict_pharmacophore": 1,
                "strict_3d_pharmacophore": 1,
                "strict_target_conditioned": 0,
                "both": 0,
            },
        },
        "candidates": [
            {
                "rank": 1,
                "global_priority_rank": 1,
                "priority_score": 0.5,
                "evidence_tier": "proxy_only",
                "activity_evidence_count": 0,
                "binding_retained": None,
                "binding_comparison_basis": None,
                "binding_comparison_strata": [],
                "binding_pactivity_delta": None,
                "admission_bases": [
                    "strict_2d_pharmacophore",
                    "strict_3d_pharmacophore",
                ],
                "pharmacophore_gate_passed": True,
                "strict_3d_pharmacophore_gate_passed": True,
                "cosing_reference": False,
                "selection_tracks": ["feature_proxy"],
                "claimable": False,
                "hypothesis_only": True,
                "wet_lab_required": True,
            }
        ],
    }
    report_path.write_text(json.dumps(payload))

    assert runner._validate_discovery_report(
        report_path,
        run_id="claim_case",
        target_id=TARGET,
    ) == payload

    payload["candidates"][0]["hypothesis_only"] = False
    report_path.write_text(json.dumps(payload))
    with pytest.raises(SystemExit, match="candidate 1 violates"):
        runner._validate_discovery_report(
            report_path,
            run_id="claim_case",
            target_id=TARGET,
        )


def test_report_validation_rejects_inconsistent_selection_tracks(tmp_path: Path) -> None:
    report_path = tmp_path / "substitute_report.json"
    payload = {
        "schema_version": runner.DISCOVERY_SCHEMA,
        "run_id": "track_case",
        "target": {"target_id": TARGET},
        "thresholds": {"max_direct_pactivity_loss_for_retention": 0.5},
        "claimable": False,
        "hypothesis_only": True,
        "wet_lab_required": True,
        "selection_strategy": {
            "mode": "balanced_tracks",
            "selected_counts": {
                "all": 1,
                "target_activity": 1,
                "cosmetic_material": 0,
                "strict_pharmacophore": 0,
                "strict_3d_pharmacophore": 0,
                "strict_target_conditioned": 0,
                "both": 0,
            },
        },
        "candidates": [
            {
                "rank": 1,
                "global_priority_rank": 1,
                "priority_score": 0.75,
                "evidence_tier": "direct_activity",
                "activity_evidence_count": 1,
                "binding_retained": None,
                "binding_comparison_basis": None,
                "binding_comparison_strata": [],
                "binding_pactivity_delta": None,
                "admission_bases": ["same_target_activity_feature_family"],
                "pharmacophore_gate_passed": False,
                "cosing_reference": False,
                "selection_tracks": ["cosmetic_material"],
                "claimable": False,
                "hypothesis_only": True,
                "wet_lab_required": True,
            }
        ],
    }
    report_path.write_text(json.dumps(payload))

    with pytest.raises(SystemExit, match="selection tracks are invalid"):
        runner._validate_discovery_report(
            report_path,
            run_id="track_case",
            target_id=TARGET,
        )


def test_report_validation_rejects_false_direct_retention_claim(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "substitute_report.json"
    payload = {
        "schema_version": runner.DISCOVERY_SCHEMA,
        "run_id": "binding_case",
        "target": {"target_id": TARGET},
        "thresholds": {"max_direct_pactivity_loss_for_retention": 0.5},
        "claimable": False,
        "hypothesis_only": True,
        "wet_lab_required": True,
        "selection_strategy": {
            "mode": "global_priority",
            "selected_counts": {
                "all": 1,
                "target_activity": 1,
                "cosmetic_material": 0,
                "strict_pharmacophore": 0,
                "strict_3d_pharmacophore": 0,
                "strict_target_conditioned": 0,
                "both": 0,
            },
        },
        "candidates": [{
            "rank": 1,
            "global_priority_rank": 1,
            "priority_score": 0.8,
            "evidence_tier": "direct_retained",
            "activity_evidence_count": 2,
            "binding_retained": True,
            "binding_comparison_basis": (
                "same_activity_type_and_source_conservative_delta"
            ),
            "binding_comparison_strata": [{
                "activity_type": "Ki",
                "source": "FixtureDB",
                "candidate_median_pactivity": 7.9,
                "parent_median_pactivity": 8.0,
                "pactivity_delta": -0.1,
                "candidate_evidence_count": 2,
                "parent_evidence_count": 3,
            }],
            "binding_pactivity_delta": -0.1,
            "admission_bases": ["same_target_activity_feature_family"],
            "pharmacophore_gate_passed": False,
            "cosing_reference": False,
            "selection_tracks": ["target_activity"],
            "claimable": False,
            "hypothesis_only": True,
            "wet_lab_required": True,
        }],
    }
    report_path.write_text(json.dumps(payload))

    assert runner._validate_discovery_report(
        report_path,
        run_id="binding_case",
        target_id=TARGET,
    ) == payload

    payload["candidates"][0]["binding_retained"] = False
    report_path.write_text(json.dumps(payload))
    with pytest.raises(SystemExit, match="binding-retention flag is invalid"):
        runner._validate_discovery_report(
            report_path,
            run_id="binding_case",
            target_id=TARGET,
        )
