"""Regression tests for fail-closed evaluation quality gates."""

from __future__ import annotations

import json
import os
import hashlib
import subprocess
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem


def run_script(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
    )


def pose_consensus_entry(atoms: list[int]) -> dict[str, object]:
    return {
        "confirmed_atoms": atoms,
        "pose_supported_interaction_atoms": atoms,
        "evidence_label": "pose-supported interaction atoms",
        "coordinate_system": "boltz_complex_ligand_atom_order_0_based",
        "evidence_sources": ["plip", "prolif"],
        "consensus_min_votes": 2,
        "degraded": False,
        "claim_eligible": True,
        "votes": {"plip": atoms, "prolif": atoms},
    }


def write_interaction_anchor_map(
    path: Path,
    *,
    parent_sdf: Path,
    consensus_json: Path,
    target_id: str,
    confirmed_atoms: list[int],
) -> None:
    parent = next(
        molecule
        for molecule in Chem.SDMolSupplier(str(parent_sdf), removeHs=False)
        if molecule is not None
    )
    complex_pdb = path.parent / f"{target_id}_complex.pdb"
    complex_pdb.write_text(
        "HETATM    1  C1  LIG A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "END\n"
    )
    boltz_report = path.parent / f"{target_id}_boltz.tsv"
    boltz_report.write_text(
        "target_id\tcomplex_pdb\tiptm\tcomplex_plddt\taffinity_log_uM\tkept\n"
        f"{target_id}\t{complex_pdb}\t0.8\t90.0\t-1.0\tyes\n"
    )
    source_records = {
        "parent_sdf": parent_sdf,
        "consensus_json": consensus_json,
        "boltz_report": boltz_report,
    }
    path.write_text(
        json.dumps(
            {
                "schema_version": "skinscout.interaction_anchor_map.v1",
                "consensus_sha256": hashlib.sha256(
                    consensus_json.read_bytes()
                ).hexdigest(),
                "boltz_report_sha256": hashlib.sha256(
                    boltz_report.read_bytes()
                ).hexdigest(),
                "sources": {
                    name: {
                        "path": str(source),
                        "bytes": source.stat().st_size,
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    }
                    for name, source in source_records.items()
                },
                "parent": {
                    "sdf_sha256": hashlib.sha256(parent_sdf.read_bytes()).hexdigest(),
                    "inchikey": Chem.MolToInchiKey(Chem.RemoveHs(parent)),
                    "atom_count": parent.GetNumAtoms(),
                },
                "targets": {
                    target_id: {
                        "target_id": target_id,
                        "source_coordinate_system": (
                            "boltz_complex_ligand_atom_order_0_based"
                        ),
                        "mapped_coordinate_system": "parent_sdf_atom_order_0_based",
                        "confirmed_complex_atoms": confirmed_atoms,
                        "confirmed_parent_atoms": confirmed_atoms,
                        "complex_pdb": str(complex_pdb),
                        "complex_pdb_bytes": complex_pdb.stat().st_size,
                        "complex_pdb_sha256": hashlib.sha256(
                            complex_pdb.read_bytes()
                        ).hexdigest(),
                        "atom_mapping": [
                            {
                                "complex_atom_index": atom_index,
                                "parent_atom_index": atom_index,
                            }
                            for atom_index in range(parent.GetNumAtoms())
                        ],
                        "mapping_status": "mapped",
                        "mapping_confidence": "high",
                        "claim_eligible": True,
                    }
                },
                "claim_eligible": True,
                "analog_pose_verified": False,
            }
        )
    )


def run_eval_all(
    tmp_path: Path,
    *,
    allow_partial: bool = False,
    allow_incomplete_leakage: bool = False,
    allow_threshold_failure: bool = False,
    use_repo_root: bool = False,
    leakage_thresholds: dict[str, str] | None = None,
    eval_thresholds: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[2]
    eval_dir = tmp_path / "results" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    direct_reference = eval_dir / "direct_exact_reference.smi"
    if not direct_reference.exists():
        direct_reference.write_text("N#N isolated-test-reference\n")
    env = os.environ.copy()
    env["PYTHON_BIN"] = sys.executable
    env["COSMAX_ROOT"] = str(repo_root if use_repo_root else tmp_path)
    env["OUT"] = str(tmp_path / "results" / "eval") if use_repo_root else "results/eval"
    env["LOG"] = (
        str(tmp_path / "results" / "logs" / "eval")
        if use_repo_root
        else "results/logs/eval"
    )
    env["CHEMBL_DIR"] = (
        str(tmp_path / "data" / "chembl37") if use_repo_root else "data/chembl37"
    )
    env["MMSEQS_TRAINING_DB"] = (
        str(tmp_path / "data" / "mmseqs" / "training_cutoff_db")
        if use_repo_root
        else "data/mmseqs/training_cutoff_db"
    )
    env["TRAINING_HOLO"] = (
        str(tmp_path / "data" / "plinder" / "training_holo_pockets.csv")
        if use_repo_root
        else "data/plinder/training_holo_pockets.csv"
    )
    env["ACTIVITY_BENCHMARK_MANIFEST"] = str(
        tmp_path / "data/activity_benchmark_202608/manifest.json"
    )
    env["ACTIVITY_RETRIEVAL_INDEX_MANIFEST"] = str(
        tmp_path / "data/activity_retrieval_202608/manifest.json"
    )
    env["ACTIVITY_RECOVERY_PANELS_MANIFEST"] = str(
        tmp_path / "data/activity_recovery_panels_202608/manifest.json"
    )
    env["ACTIVITY_RETRIEVAL_DEV_SELECTION_MANIFEST"] = str(
        tmp_path / "results/eval/activity_retrieval_202608/dev_selection/manifest.json"
    )
    env["ACTIVITY_RETRIEVAL_FINAL_EVAL_MANIFEST"] = str(
        tmp_path / "results/eval/activity_retrieval_202608/test_evaluation/manifest.json"
    )
    env["ACTIVITY_RETRIEVAL_GATE"] = str(
        tmp_path / "data/manifests/activity_retrieval_final_gate.flag"
    )
    if allow_incomplete_leakage:
        env["EVAL_ALLOW_INCOMPLETE_LEAKAGE"] = "1"
    else:
        env.pop("EVAL_ALLOW_INCOMPLETE_LEAKAGE", None)
    if allow_partial:
        env["EVAL_ALLOW_PARTIAL"] = "1"
    else:
        env.pop("EVAL_ALLOW_PARTIAL", None)
    if allow_threshold_failure:
        env["EVAL_ALLOW_THRESHOLD_FAILURE"] = "1"
    else:
        env.pop("EVAL_ALLOW_THRESHOLD_FAILURE", None)
    for key, value in (leakage_thresholds or {}).items():
        env[key] = value
    for key, value in (eval_thresholds or {}).items():
        env[key] = value
    return subprocess.run(
        ["bash", str(repo_root / "eval/run_all.sh")],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )


def test_eval_run_all_fails_closed_on_missing_required_inputs(tmp_path: Path) -> None:
    stale_manifest = tmp_path / "results" / "eval" / "iteration_manifest.json"
    stale_manifest.parent.mkdir(parents=True)
    stale_manifest.write_text('{"stale": true}\n')

    res = run_eval_all(tmp_path)

    assert res.returncode != 0
    assert "missing leakage audit targets" in res.stderr
    assert "EVAL_ALLOW_PARTIAL=1" in res.stderr
    assert not stale_manifest.exists()


def test_eval_run_all_partial_mode_only_warns_on_missing_inputs(tmp_path: Path) -> None:
    res = run_eval_all(tmp_path, allow_partial=True)

    assert res.returncode == 0, res.stderr
    assert "missing leakage audit targets" in res.stderr
    assert "skipping diagnostic step" in res.stderr
    assert "manifest in" in res.stdout
    manifest = tmp_path / "results" / "eval" / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    assert payload["n_steps"] == 0
    assert payload["threshold_status"]["n_checked"] == 0
    assert payload["threshold_status"]["n_missing_required"] == 0
    assert payload["claim_ready"] is False
    assert payload["diagnostic_overrides"]["allow_empty_iteration"] is True
    assert payload["diagnostic_overrides"]["allow_incomplete_eval_inputs"] is True
    assert "no_runnable_eval_steps" in {
        blocker["code"] for blocker in payload["claim_blockers"]
    }


def test_eval_run_all_partial_mode_does_not_hide_threshold_failure(
    tmp_path: Path,
) -> None:
    eval_dir = tmp_path / "results" / "eval"
    eval_dir.mkdir(parents=True)
    pd.DataFrame([{
        "mode": "comprehensive",
        "n_cold_targets": 10,
        "n_truth_in_cold": 1,
        "passes_threshold": False,
        "recall@1": 0.0,
        "recall@5": 0.0,
        "recall@10": 0.0,
        "recall@50": 0.0,
    }]).to_csv(eval_dir / "cold_start_comprehensive.csv", index=False)

    res = run_eval_all(tmp_path, allow_partial=True)

    assert res.returncode != 0
    log = tmp_path / "results" / "logs" / "eval" / "iteration_manifest.log"
    assert "Evaluation metric threshold failure" in log.read_text()
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    assert payload["threshold_status"]["status"] == "failed"
    assert payload["threshold_status"]["n_failed"] == 1
    assert payload["claim_ready"] is False
    assert "threshold_failures" in {
        blocker["code"] for blocker in payload["claim_blockers"]
    }


def test_eval_run_all_threshold_failure_override_is_explicit(
    tmp_path: Path,
) -> None:
    eval_dir = tmp_path / "results" / "eval"
    eval_dir.mkdir(parents=True)
    pd.DataFrame([{
        "mode": "comprehensive",
        "n_cold_targets": 10,
        "n_truth_in_cold": 1,
        "passes_threshold": False,
        "recall@1": 0.0,
        "recall@5": 0.0,
        "recall@10": 0.0,
        "recall@50": 0.0,
    }]).to_csv(eval_dir / "cold_start_comprehensive.csv", index=False)

    res = run_eval_all(
        tmp_path,
        allow_partial=True,
        allow_incomplete_leakage=True,
        allow_threshold_failure=True,
    )

    assert res.returncode == 0, res.stderr
    payload = json.loads((eval_dir / "iteration_manifest.json").read_text())
    assert payload["threshold_status"]["status"] == "failed"
    assert payload["threshold_status"]["n_failed"] == 1
    assert payload["claim_ready"] is False
    assert payload["diagnostic_overrides"]["allow_threshold_failure"] is True
    assert "threshold_failures" in {
        blocker["code"] for blocker in payload["claim_blockers"]
    }


def write_minimal_run_all_inputs(tmp_path: Path) -> None:
    eval_dir = tmp_path / "results" / "eval"
    rankings = eval_dir / "rankings"
    chembl = tmp_path / "data" / "chembl37"
    mmseqs = tmp_path / "data" / "mmseqs" / "training_cutoff_db"
    plinder = tmp_path / "data" / "plinder"
    rankings.mkdir(parents=True)
    chembl.mkdir(parents=True)
    mmseqs.mkdir(parents=True)
    plinder.mkdir(parents=True)
    pd.DataFrame([{
        "run_id": "case",
        "uniprot": "P1",
        "smiles": "CCO",
    }]).to_csv(eval_dir / "eval_targets.csv", index=False)
    pd.DataFrame([{"uniprot": "P1"}]).to_csv(
        eval_dir / "cold_start_truth.csv",
        index=False,
    )
    pd.DataFrame([{"uniprot": "P1", "molecule_chembl_id": "CHEMBL1"}]).to_parquet(
        chembl / "human_activities.parquet",
    )
    (chembl / "training_ligands.smi").write_text("CCN CHEMBL_REF\n")
    (mmseqs / "training_cutoff_seqs.fasta").write_text(">train\nMAAA\n")
    pd.DataFrame([{"uniprot": "P1", "pocket_sucos": 0.0}]).to_csv(
        plinder / "training_holo_pockets.csv",
        index=False,
    )
    for mode in ("comprehensive", "fast", "dti_only"):
        pd.DataFrame([{"target_id": "P1", "rank": 1}]).to_csv(
            rankings / f"cold_start__{mode}.csv",
            index=False,
        )
    (eval_dir / "disagreement_analysis.json").write_text(json.dumps({
        "docking_only": ["P1"],
        "dti_only": [],
        "both_top": [],
    }))


def test_eval_run_all_requires_target_classes_for_disagreement(
    tmp_path: Path,
) -> None:
    write_minimal_run_all_inputs(tmp_path)

    res = run_eval_all(
        tmp_path,
        allow_incomplete_leakage=True,
        use_repo_root=True,
    )

    assert res.returncode != 0
    assert "missing target class reference" in res.stderr


def test_eval_run_all_requires_collected_run_provenance_for_claims(
    tmp_path: Path,
) -> None:
    write_minimal_run_all_inputs(tmp_path)
    eval_dir = tmp_path / "results" / "eval"
    rankings = eval_dir / "rankings"
    for mode in ("comprehensive", "fast", "dti_only"):
        pd.DataFrame([{
            "target_id": "P1",
            "final_score": 0.9,
            "source_count": 3,
            "sources": "autodock;gnina;rtmscore",
        }]).to_csv(rankings / f"cold_start__{mode}.csv", index=False)
    pd.DataFrame([{"uniprot": "P1", "class": "kinase"}]).to_parquet(
        tmp_path / "data" / "chembl37" / "target_classes.parquet",
    )

    res = run_eval_all(
        tmp_path,
        allow_incomplete_leakage=True,
        use_repo_root=True,
    )

    assert res.returncode != 0
    assert "missing collected run provenance manifest" in res.stderr
    assert "EVAL_ALLOW_PARTIAL=1" in res.stderr


def test_eval_run_all_requires_activity_retrieval_gate_for_claims(
    tmp_path: Path,
) -> None:
    write_minimal_run_all_inputs(tmp_path)
    eval_dir = tmp_path / "results" / "eval"
    rankings = eval_dir / "rankings"
    for mode in ("comprehensive", "fast", "dti_only"):
        pd.DataFrame([{
            "target_id": "P1",
            "final_score": 0.9,
            "source_count": 3,
            "sources": "autodock;gnina;rtmscore",
        }]).to_csv(rankings / f"cold_start__{mode}.csv", index=False)
    pd.DataFrame([{"uniprot": "P1", "class": "kinase"}]).to_parquet(
        tmp_path / "data" / "chembl37" / "target_classes.parquet",
    )
    (eval_dir / "collected_runs.json").write_text('{"runs": []}\n')

    res = run_eval_all(
        tmp_path,
        allow_incomplete_leakage=True,
        use_repo_root=True,
    )

    assert res.returncode != 0
    assert "missing activity retrieval benchmark manifest" in res.stderr
    assert "EVAL_ALLOW_PARTIAL=1" in res.stderr


def test_eval_run_all_partial_mode_warns_on_missing_activity_gate(
    tmp_path: Path,
) -> None:
    res = run_eval_all(tmp_path, allow_partial=True)

    assert res.returncode == 0, res.stderr
    assert "missing activity retrieval benchmark manifest" in res.stderr
    assert "skipping diagnostic step" in res.stderr


def test_eval_run_all_removes_stale_output_before_failed_step(
    tmp_path: Path,
) -> None:
    write_minimal_run_all_inputs(tmp_path)
    eval_dir = tmp_path / "results" / "eval"
    rankings = eval_dir / "rankings"
    stale = eval_dir / "cold_start_comprehensive.csv"
    stale.write_text("passes_threshold\nTrue\n")
    pd.DataFrame([{"wrong": "P1"}]).to_csv(
        rankings / "cold_start__comprehensive.csv",
        index=False,
    )

    res = run_eval_all(
        tmp_path,
        allow_incomplete_leakage=True,
        use_repo_root=True,
    )

    assert res.returncode != 0
    assert not stale.exists()


def test_eval_run_all_uses_configured_chembl_fingerprint_reference(
    tmp_path: Path,
) -> None:
    eval_dir = tmp_path / "results" / "eval"
    eval_dir.mkdir(parents=True)
    pd.DataFrame([{"smiles": "CCO"}]).to_csv(
        eval_dir / "generated_analogs.csv",
        index=False,
    )
    pd.DataFrame([{"smiles": "CCN"}]).to_csv(
        eval_dir / "other_cosing.csv",
        index=False,
    )

    res = run_eval_all(
        tmp_path,
        allow_partial=True,
        use_repo_root=True,
    )

    expected = str(tmp_path / "data" / "chembl37" / "fp_morgan2_2048.parquet")
    assert res.returncode == 0, res.stderr
    assert expected in res.stderr


def test_eval_run_all_uses_configured_leakage_references(
    tmp_path: Path,
) -> None:
    eval_dir = tmp_path / "results" / "eval"
    eval_dir.mkdir(parents=True)
    pd.DataFrame([{
        "run_id": "case",
        "uniprot": "P1",
        "smiles": "CCO",
    }]).to_csv(eval_dir / "eval_targets.csv", index=False)

    res = run_eval_all(
        tmp_path,
        allow_partial=True,
        use_repo_root=True,
    )

    expected = str(tmp_path / "data" / "mmseqs" / "training_cutoff_db")
    assert res.returncode == 0, res.stderr
    assert "missing leakage sequence reference" in res.stderr
    assert expected in res.stderr


def test_eval_run_all_uses_configured_leakage_thresholds(
    tmp_path: Path,
) -> None:
    eval_dir = tmp_path / "results" / "eval"
    seq_db = tmp_path / "data" / "mmseqs" / "training_cutoff_db"
    chembl = tmp_path / "data" / "chembl37"
    plinder = tmp_path / "data" / "plinder"
    eval_dir.mkdir(parents=True)
    seq_db.mkdir(parents=True)
    chembl.mkdir(parents=True)
    plinder.mkdir(parents=True)
    pd.DataFrame([{
        "run_id": "case",
        "uniprot": "P1",
        "smiles": "CCO",
        "sequence": "AAAA",
    }]).to_csv(eval_dir / "eval_targets.csv", index=False)
    (seq_db / "training_cutoff_seqs.fasta").write_text(">train\nABCD\n")
    (chembl / "training_ligands.smi").write_text("CCN ref\n")
    pd.DataFrame([{"uniprot": "P1", "pocket_sucos": 0.1}]).to_csv(
        plinder / "training_holo_pockets.csv",
        index=False,
    )

    res = run_eval_all(
        tmp_path,
        allow_partial=True,
        allow_threshold_failure=True,
        use_repo_root=True,
        leakage_thresholds={
            "EVAL_SEQ_ID_THRESHOLD": "0.81",
            "EVAL_LIGAND_TANIMOTO_THRESHOLD": "0.82",
            "EVAL_POCKET_SUCOS_THRESHOLD": "0.83",
        },
    )

    manifest = json.loads((eval_dir / "iteration_manifest.json").read_text())
    command = manifest["steps"][0]["command"]
    assert res.returncode == 0, res.stderr
    assert "--seq-id-threshold" in command
    assert "0.81" in command
    assert "--ligand-tanimoto-threshold" in command
    assert "0.82" in command
    assert "--pocket-sucos-threshold" in command
    assert "0.83" in command


def test_eval_run_all_reads_leakage_thresholds_from_workflow_config(
    tmp_path: Path,
) -> None:
    eval_dir = tmp_path / "results" / "eval"
    seq_db = tmp_path / "data" / "mmseqs" / "training_cutoff_db"
    chembl = tmp_path / "data" / "chembl37"
    plinder = tmp_path / "data" / "plinder"
    workflow = tmp_path / "workflow"
    eval_dir.mkdir(parents=True)
    seq_db.mkdir(parents=True)
    chembl.mkdir(parents=True)
    plinder.mkdir(parents=True)
    workflow.mkdir(parents=True)
    (workflow / "config.yaml").write_text(
        "evaluation:\n"
        "  leakage:\n"
        "    seq_id_threshold: 0.41\n"
        "    ligand_tanimoto_threshold: 0.42\n"
        "    pocket_sucos_threshold: 0.43\n"
    )
    pd.DataFrame([{
        "run_id": "case",
        "uniprot": "P1",
        "smiles": "CCO",
        "sequence": "AAAA",
    }]).to_csv(eval_dir / "eval_targets.csv", index=False)
    (seq_db / "training_cutoff_seqs.fasta").write_text(">train\nABCD\n")
    (chembl / "training_ligands.smi").write_text("CCN ref\n")
    pd.DataFrame([{"uniprot": "P1", "pocket_sucos": 0.1}]).to_csv(
        plinder / "training_holo_pockets.csv",
        index=False,
    )

    res = run_eval_all(
        tmp_path,
        allow_partial=True,
        allow_incomplete_leakage=True,
        allow_threshold_failure=True,
    )

    manifest = json.loads((eval_dir / "iteration_manifest.json").read_text())
    command = manifest["steps"][0]["command"]
    assert res.returncode == 0, res.stderr
    assert "--seq-id-threshold" in command
    assert "0.41" in command
    assert "--ligand-tanimoto-threshold" in command
    assert "0.42" in command
    assert "--pocket-sucos-threshold" in command
    assert "0.43" in command


def test_eval_run_all_reads_metric_thresholds_from_workflow_config(
    tmp_path: Path,
) -> None:
    eval_dir = tmp_path / "results" / "eval"
    rankings = eval_dir / "rankings"
    chembl = tmp_path / "data" / "chembl37"
    workflow = tmp_path / "workflow"
    eval_dir.mkdir(parents=True)
    rankings.mkdir(parents=True)
    chembl.mkdir(parents=True)
    workflow.mkdir(parents=True)
    (workflow / "config.yaml").write_text(
        "evaluation:\n"
        "  thresholds:\n"
        "    cold_start_min_recall_at_50: 0.12\n"
        "    cosmetic_retro_min_mean_top10: 0.13\n"
        "    skin_efficacy_min_mean_precision: 0.14\n"
        "    skin_efficacy_min_mean_recall: 0.15\n"
        "    analog_min_recovery: 0.16\n"
        "    analog_min_novelty: 0.17\n"
        "    analog_min_mean_ra_score: 0.18\n"
        "    pharmacophore_min_preserved_fraction: 0.19\n"
    )
    pd.DataFrame([{"uniprot": "P1"}]).to_csv(
        eval_dir / "cold_start_truth.csv",
        index=False,
    )
    pd.DataFrame([{
        "target_id": "P1",
        "final_score": 1.0,
        "source_count": 3,
        "sources": "autodock;gnina;rtmscore",
    }]).to_csv(
        rankings / "cold_start__comprehensive.csv",
        index=False,
    )
    pd.DataFrame([{
        "uniprot": "P1",
        "molecule_chembl_id": "CHEMBL1",
    }]).to_parquet(chembl / "human_activities.parquet")

    res = run_eval_all(tmp_path, allow_partial=True)

    assert res.returncode == 0, res.stderr
    manifest = json.loads((eval_dir / "iteration_manifest.json").read_text())
    command = next(
        step["command"]
        for step in manifest["steps"]
        if step["name"] == "cold_start_comprehensive"
    )
    assert "--chembl-dir" in command
    assert str(chembl.resolve()) in command
    assert "--min-recall-at-50" in command
    assert "0.12" in command


def test_eval_run_all_records_actual_workflow_config_in_manifest(
    tmp_path: Path,
) -> None:
    eval_dir = tmp_path / "results" / "eval"
    workflow = tmp_path / "workflow"
    eval_dir.mkdir(parents=True)
    workflow.mkdir(parents=True)
    workflow_config = workflow / "config.yaml"
    workflow_config.write_text(
        "evaluation:\n"
        "  thresholds:\n"
        "    cold_start_min_recall_at_50: 0.12\n"
    )

    res = run_eval_all(tmp_path, allow_partial=True)

    assert res.returncode == 0, res.stderr
    manifest = json.loads((eval_dir / "iteration_manifest.json").read_text())
    snapshot = manifest["data_snapshot"]["workflow_config"]
    assert snapshot["path"] == str(workflow_config.resolve())
    assert snapshot["sha256"] == hashlib.sha256(workflow_config.read_bytes()).hexdigest()
    assert manifest["provenance"]["config_sha256"] == snapshot["sha256"]


def test_leakage_check_rejects_boolean_pocket_sucos_reference(
    tmp_path: Path,
) -> None:
    input_csv = tmp_path / "eval_targets.csv"
    seq_db = tmp_path / "training_seq_db"
    ligands = tmp_path / "training_ligands.smi"
    holo = tmp_path / "training_holo.json"
    out = tmp_path / "leakage.csv"
    seq_db.mkdir()
    pd.DataFrame([{
        "uniprot": "P1",
        "smiles": "CCO",
        "sequence": "MAAA",
    }]).to_csv(input_csv, index=False)
    (seq_db / "training_cutoff_seqs.fasta").write_text(">train\nMAAA\n")
    ligands.write_text("CCN REF\n")
    holo.write_text('{"P1": true}\n')
    out.write_text("stale\n")

    res = run_script([
        "eval/leakage_check.py",
        "--input-csv", str(input_csv),
        "--training-seq-db", str(seq_db),
        "--training-ligands", str(ligands),
        "--training-holo", str(holo),
        "--out-csv", str(out),
    ])

    assert res.returncode != 0
    assert "Pocket SuCOS" in res.stderr
    assert "must be numeric" in res.stderr
    assert not out.exists()


def run_cold_start(tmp_path: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    ranking = tmp_path / "ranking.csv"
    truth = tmp_path / "truth.csv"
    out = tmp_path / "cold.csv"
    return run_script([
        "eval/cold_start_eval.py",
        "--mode", "comprehensive",
        "--ranking-csv", str(ranking),
        "--ground-truth-csv", str(truth),
        "--chembl-dir", str(tmp_path / "chembl"),
        "--out-csv", str(out),
        *extra,
    ])


def write_cold_start_inputs(tmp_path: Path, *, truth: str = "P1") -> None:
    (tmp_path / "chembl").mkdir()
    pd.DataFrame([
        {"uniprot": "P1", "molecule_chembl_id": "CHEMBL1"},
        {"uniprot": "P2", "molecule_chembl_id": "CHEMBL2"},
        {"uniprot": "P2", "molecule_chembl_id": "CHEMBL3"},
    ]).to_parquet(tmp_path / "chembl" / "human_activities.parquet")
    pd.DataFrame([
        {"target_id": "P1", "rank": 1},
        {"target_id": "P2", "rank": 2},
    ]).to_csv(
        tmp_path / "ranking.csv",
        index=False,
    )
    pd.DataFrame([{"uniprot": truth}]).to_csv(tmp_path / "truth.csv", index=False)


def run_disagreement(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    return run_script([
        "eval/disagreement_eval.py",
        "--disagreement-json", str(tmp_path / "disagreement.json"),
        "--target-classes", str(tmp_path / "target_classes.parquet"),
        "--out-csv", str(tmp_path / "disagreement.csv"),
    ])


def write_target_classes(tmp_path: Path) -> None:
    pd.DataFrame([
        {"uniprot": "P1", "class": "kinase"},
        {"uniprot": "P2", "class": "gpcr"},
        {"uniprot": "P3", "class": "protease"},
    ]).to_parquet(tmp_path / "target_classes.parquet")


def run_cosmetic_retro(tmp_path: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return run_script([
        "eval/cosmetic_retrospective_eval.py",
        "--cases-csv", str(tmp_path / "cases.csv"),
        "--rankings-dir", str(tmp_path / "rankings"),
        "--out-csv", str(tmp_path / "cosmetic.csv"),
        *extra,
    ])


def write_cosmetic_case(tmp_path: Path, inci: str = "Retinol") -> None:
    (tmp_path / "rankings").mkdir()
    pd.DataFrame([{
        "inci_name": inci,
        "smiles": "CCO",
        "known_targets": "P1;P2",
    }]).to_csv(tmp_path / "cases.csv", index=False)


def test_cosmetic_retro_fails_on_missing_ranking_by_default(tmp_path: Path) -> None:
    write_cosmetic_case(tmp_path)
    (tmp_path / "cosmetic.csv").write_text("stale\n")

    res = run_cosmetic_retro(tmp_path)

    assert res.returncode != 0
    assert "Missing 1 cosmetic retrospective ranking" in res.stderr
    assert not (tmp_path / "cosmetic.csv").exists()


def test_cosmetic_retro_allow_missing_is_diagnostic(tmp_path: Path) -> None:
    write_cosmetic_case(tmp_path)

    res = run_cosmetic_retro(
        tmp_path,
        "--allow-missing-rankings",
        "--allow-threshold-failure",
    )

    assert res.returncode == 0, res.stderr
    row = pd.read_csv(tmp_path / "cosmetic.csv").iloc[0]
    assert row["status"] == "no_ranking"
    assert not bool(row["passes_threshold"])


def test_cosmetic_retro_missing_rankings_flag_does_not_hide_threshold_failure(
    tmp_path: Path,
) -> None:
    write_cosmetic_case(tmp_path)
    (tmp_path / "cosmetic.csv").write_text("stale\n")

    res = run_cosmetic_retro(tmp_path, "--allow-missing-rankings")

    assert res.returncode != 0
    assert "Cosmetic retrospective failed claim threshold" in res.stderr
    assert "pass --allow-threshold-failure only for explicit diagnostics" in res.stderr
    assert not (tmp_path / "cosmetic.csv").exists()


def test_cosmetic_retro_rejects_ranking_without_target_id(tmp_path: Path) -> None:
    write_cosmetic_case(tmp_path)
    (tmp_path / "cosmetic.csv").write_text("stale\n")
    pd.DataFrame([{"score": 1.0}]).to_csv(
        tmp_path / "rankings" / "Retinol__ranked_targets_v3.csv",
        index=False,
    )

    res = run_cosmetic_retro(tmp_path)

    assert res.returncode != 0
    assert "missing required column 'target_id'" in res.stderr
    assert not (tmp_path / "cosmetic.csv").exists()


def test_cosmetic_retro_rejects_blank_case_fields(tmp_path: Path) -> None:
    write_cosmetic_case(tmp_path)
    (tmp_path / "cosmetic.csv").write_text("stale\n")
    pd.DataFrame([{"inci_name": "Retinol", "smiles": "CCO", "known_targets": None}]).to_csv(
        tmp_path / "cases.csv",
        index=False,
    )

    res = run_cosmetic_retro(tmp_path)

    assert res.returncode != 0
    assert (
        "Cosmetic retrospective cases CSV column 'known_targets' contains blank values"
        in res.stderr
    )
    assert not (tmp_path / "cosmetic.csv").exists()


def test_cosmetic_retro_rejects_case_without_smiles(tmp_path: Path) -> None:
    write_cosmetic_case(tmp_path)
    (tmp_path / "cosmetic.csv").write_text("stale\n")
    pd.DataFrame([{"inci_name": "Retinol", "known_targets": "P1;P2"}]).to_csv(
        tmp_path / "cases.csv",
        index=False,
    )
    pd.DataFrame([{"target_id": "P1"}]).to_csv(
        tmp_path / "rankings" / "Retinol__ranked_targets_v3.csv",
        index=False,
    )

    res = run_cosmetic_retro(tmp_path)

    assert res.returncode != 0
    assert "missing required columns ['smiles']" in res.stderr
    assert not (tmp_path / "cosmetic.csv").exists()


def test_cosmetic_retro_rejects_invalid_case_smiles(tmp_path: Path) -> None:
    write_cosmetic_case(tmp_path)
    (tmp_path / "cosmetic.csv").write_text("stale\n")
    pd.DataFrame([{
        "inci_name": "Retinol",
        "smiles": "not-a-smiles",
        "known_targets": "P1;P2",
    }]).to_csv(
        tmp_path / "cases.csv",
        index=False,
    )
    pd.DataFrame([{"target_id": "P1"}]).to_csv(
        tmp_path / "rankings" / "Retinol__ranked_targets_v3.csv",
        index=False,
    )

    res = run_cosmetic_retro(tmp_path)

    assert res.returncode != 0
    assert "column 'smiles' contains invalid SMILES" in res.stderr
    assert not (tmp_path / "cosmetic.csv").exists()


def test_cosmetic_retro_rejects_empty_known_target_token(tmp_path: Path) -> None:
    write_cosmetic_case(tmp_path)
    (tmp_path / "cosmetic.csv").write_text("stale\n")
    pd.DataFrame([{
        "inci_name": "Retinol",
        "smiles": "CCO",
        "known_targets": "P1;;P2",
    }]).to_csv(
        tmp_path / "cases.csv",
        index=False,
    )

    res = run_cosmetic_retro(tmp_path)

    assert res.returncode != 0
    assert (
        "Cosmetic retrospective cases CSV column 'known_targets' contains empty "
        "';'-separated value(s): Retinol"
        in res.stderr
    )
    assert not (tmp_path / "cosmetic.csv").exists()


def test_cosmetic_retro_rejects_duplicate_known_target_token(tmp_path: Path) -> None:
    write_cosmetic_case(tmp_path)
    (tmp_path / "cosmetic.csv").write_text("stale\n")
    pd.DataFrame([{
        "inci_name": "Retinol",
        "smiles": "CCO",
        "known_targets": "P1;P1;P2",
    }]).to_csv(
        tmp_path / "cases.csv",
        index=False,
    )

    res = run_cosmetic_retro(tmp_path)

    assert res.returncode != 0
    assert (
        "Cosmetic retrospective cases CSV column 'known_targets' contains "
        "duplicate ';'-separated value(s): Retinol: P1"
        in res.stderr
    )
    assert not (tmp_path / "cosmetic.csv").exists()


def test_cosmetic_retro_rejects_duplicate_case_inci_name(tmp_path: Path) -> None:
    write_cosmetic_case(tmp_path)
    (tmp_path / "cosmetic.csv").write_text("stale\n")
    pd.DataFrame([
        {"inci_name": "Retinol", "smiles": "CCO", "known_targets": "P1"},
        {"inci_name": "Retinol", "smiles": "CCO", "known_targets": "P1"},
    ]).to_csv(
        tmp_path / "cases.csv",
        index=False,
    )

    res = run_cosmetic_retro(tmp_path)

    assert res.returncode != 0
    assert (
        "Cosmetic retrospective cases CSV contains duplicate inci_name values: Retinol"
        in res.stderr
    )
    assert not (tmp_path / "cosmetic.csv").exists()


def test_cosmetic_retro_rejects_blank_ranking_target_id(tmp_path: Path) -> None:
    write_cosmetic_case(tmp_path)
    (tmp_path / "cosmetic.csv").write_text("stale\n")
    pd.DataFrame([
        {"target_id": "P1", "rank": 1},
        {"target_id": None, "rank": 2},
    ]).to_csv(
        tmp_path / "rankings" / "Retinol__ranked_targets_v3.csv",
        index=False,
    )

    res = run_cosmetic_retro(tmp_path)

    assert res.returncode != 0
    assert "Cosmetic retrospective ranking column 'target_id' contains blank values" in res.stderr
    assert not (tmp_path / "cosmetic.csv").exists()


def test_cosmetic_retro_rejects_duplicate_ranking_target_id(
    tmp_path: Path,
) -> None:
    write_cosmetic_case(tmp_path)
    (tmp_path / "cosmetic.csv").write_text("stale\n")
    pd.DataFrame([
        {"target_id": "P1", "rank": 1},
        {"target_id": "P1", "rank": 2},
    ]).to_csv(
        tmp_path / "rankings" / "Retinol__ranked_targets_v3.csv",
        index=False,
    )

    res = run_cosmetic_retro(tmp_path)

    assert res.returncode != 0
    assert (
        "Cosmetic retrospective ranking contains duplicate target_id values: P1"
        in res.stderr
    )
    assert not (tmp_path / "cosmetic.csv").exists()


def test_cosmetic_retro_writes_recovery_for_valid_ranking(tmp_path: Path) -> None:
    write_cosmetic_case(tmp_path)
    pd.DataFrame([
        {"target_id": "P2", "rank": 1},
        {"target_id": "P3", "rank": 2},
    ]).to_csv(
        tmp_path / "rankings" / "Retinol__ranked_targets_v3.csv",
        index=False,
    )

    res = run_cosmetic_retro(tmp_path)

    assert res.returncode == 0, res.stderr
    row = pd.read_csv(tmp_path / "cosmetic.csv").iloc[0]
    assert row["status"] == "evaluated"
    assert row["top1"] == 1
    assert bool(row["passes_threshold"])


def test_cosmetic_retro_sparse_declared_rank_is_not_promoted_by_row_position(
    tmp_path: Path,
) -> None:
    write_cosmetic_case(tmp_path)
    pd.DataFrame([{"target_id": "P2", "final_rank": 100, "rank": 1}]).to_csv(
        tmp_path / "rankings" / "Retinol__ranked_targets_v3.csv",
        index=False,
    )

    res = run_cosmetic_retro(tmp_path, "--allow-threshold-failure")

    assert res.returncode == 0, res.stderr
    row = pd.read_csv(tmp_path / "cosmetic.csv").iloc[0]
    assert row["top1"] == 0
    assert row["top10"] == 0


def test_cosmetic_retro_fails_below_claim_thresholds(tmp_path: Path) -> None:
    write_cosmetic_case(tmp_path)
    (tmp_path / "cosmetic.csv").write_text("stale\n")
    pd.DataFrame([
        {"target_id": "P3", "rank": 1},
        {"target_id": "P4", "rank": 2},
    ]).to_csv(
        tmp_path / "rankings" / "Retinol__ranked_targets_v3.csv",
        index=False,
    )

    res = run_cosmetic_retro(tmp_path)

    assert res.returncode != 0
    assert "Cosmetic retrospective failed claim threshold" in res.stderr
    assert not (tmp_path / "cosmetic.csv").exists()


def test_cosmetic_retro_threshold_failure_flag_is_diagnostic(tmp_path: Path) -> None:
    write_cosmetic_case(tmp_path)
    pd.DataFrame([
        {"target_id": "P3", "rank": 1},
        {"target_id": "P4", "rank": 2},
    ]).to_csv(
        tmp_path / "rankings" / "Retinol__ranked_targets_v3.csv",
        index=False,
    )

    res = run_cosmetic_retro(tmp_path, "--allow-threshold-failure")

    assert res.returncode == 0, res.stderr
    row = pd.read_csv(tmp_path / "cosmetic.csv").iloc[0]
    assert row["top10"] == 0
    assert not bool(row["passes_threshold"])


def test_cosmetic_retro_rejects_invalid_claim_threshold(tmp_path: Path) -> None:
    write_cosmetic_case(tmp_path)
    (tmp_path / "cosmetic.csv").write_text("stale\n")

    res = run_cosmetic_retro(tmp_path, "--min-mean-top10", "-0.1")

    assert res.returncode != 0
    assert "--min-mean-top10 must be a finite value in [0, 1]" in res.stderr
    assert not (tmp_path / "cosmetic.csv").exists()


def test_disagreement_requires_target_class_reference(tmp_path: Path) -> None:
    (tmp_path / "disagreement.csv").write_text("stale\n")
    (tmp_path / "disagreement.json").write_text(json.dumps({
        "docking_only": ["P1"],
        "dti_only": [],
        "both_top": [],
    }))

    res = run_disagreement(tmp_path)

    assert res.returncode != 0
    assert "Target class reference is required" in res.stderr
    assert not (tmp_path / "disagreement.csv").exists()


def test_disagreement_rejects_empty_target_buckets(tmp_path: Path) -> None:
    write_target_classes(tmp_path)
    (tmp_path / "disagreement.csv").write_text("stale\n")
    (tmp_path / "disagreement.json").write_text(json.dumps({
        "docking_only": [],
        "dti_only": [],
        "both_top": [],
    }))

    res = run_disagreement(tmp_path)

    assert res.returncode != 0
    assert "contains no target IDs" in res.stderr
    assert not (tmp_path / "disagreement.csv").exists()


def test_disagreement_rejects_missing_required_bucket(tmp_path: Path) -> None:
    write_target_classes(tmp_path)
    (tmp_path / "disagreement.csv").write_text("stale\n")
    (tmp_path / "disagreement.json").write_text(json.dumps({
        "docking_only": ["P1"],
        "dti_only": [],
    }))

    res = run_disagreement(tmp_path)

    assert res.returncode != 0
    assert "missing required bucket" in res.stderr
    assert not (tmp_path / "disagreement.csv").exists()


def test_disagreement_rejects_invalid_target_ids(tmp_path: Path) -> None:
    write_target_classes(tmp_path)
    (tmp_path / "disagreement.csv").write_text("stale\n")
    (tmp_path / "disagreement.json").write_text(json.dumps({
        "docking_only": ["P1", None],
        "dti_only": ["  "],
        "both_top": [],
    }))

    res = run_disagreement(tmp_path)

    assert res.returncode != 0
    assert "target IDs must be non-empty strings" in res.stderr
    assert not (tmp_path / "disagreement.csv").exists()


def test_disagreement_rejects_duplicate_target_ids_across_buckets(
    tmp_path: Path,
) -> None:
    write_target_classes(tmp_path)
    (tmp_path / "disagreement.csv").write_text("stale\n")
    (tmp_path / "disagreement.json").write_text(json.dumps({
        "docking_only": ["P1"],
        "dti_only": ["P1"],
        "both_top": [],
    }))

    res = run_disagreement(tmp_path)

    assert res.returncode != 0
    assert "target IDs must be unique across buckets: P1" in res.stderr
    assert not (tmp_path / "disagreement.csv").exists()


def test_disagreement_rejects_duplicate_target_class_uniprot(
    tmp_path: Path,
) -> None:
    (tmp_path / "disagreement.csv").write_text("stale\n")
    (tmp_path / "disagreement.json").write_text(json.dumps({
        "docking_only": ["P1"],
        "dti_only": [],
        "both_top": [],
    }))
    pd.DataFrame([
        {"uniprot": "P1", "class": "kinase"},
        {"uniprot": "P1", "class": "gpcr"},
    ]).to_parquet(tmp_path / "target_classes.parquet")

    res = run_disagreement(tmp_path)

    assert res.returncode != 0
    assert "Target class reference contains duplicate uniprot values: P1" in res.stderr
    assert not (tmp_path / "disagreement.csv").exists()


def test_disagreement_writes_class_counts_for_valid_inputs(tmp_path: Path) -> None:
    write_target_classes(tmp_path)
    (tmp_path / "disagreement.json").write_text(json.dumps({
        "docking_only": ["P1"],
        "dti_only": ["P2"],
        "both_top": ["P3"],
    }))

    res = run_disagreement(tmp_path)

    assert res.returncode == 0, res.stderr
    rows = pd.read_csv(tmp_path / "disagreement.csv").to_dict("records")
    assert {"bucket": "docking_only", "class": "kinase", "count": 1} in rows
    assert {"bucket": "dti_only", "class": "gpcr", "count": 1} in rows
    assert {"bucket": "both_top", "class": "protease", "count": 1} in rows


def test_cold_start_requires_chembl_activity_reference(tmp_path: Path) -> None:
    (tmp_path / "cold.csv").write_text("stale\n")
    pd.DataFrame([{"target_id": "P1", "rank": 1}]).to_csv(
        tmp_path / "ranking.csv",
        index=False,
    )
    pd.DataFrame([{"uniprot": "P1"}]).to_csv(tmp_path / "truth.csv", index=False)

    res = run_cold_start(tmp_path)

    assert res.returncode != 0
    assert "ChEMBL human activity reference is required" in res.stderr
    assert not (tmp_path / "cold.csv").exists()


def test_cold_start_rejects_ranking_without_target_id(tmp_path: Path) -> None:
    write_cold_start_inputs(tmp_path)
    (tmp_path / "cold.csv").write_text("stale\n")
    pd.DataFrame([{"score": 1.0}]).to_csv(tmp_path / "ranking.csv", index=False)

    res = run_cold_start(tmp_path)

    assert res.returncode != 0
    assert "Cold-start ranking missing required columns" in res.stderr
    assert not (tmp_path / "cold.csv").exists()


def test_cold_start_rejects_truth_without_cold_overlap(tmp_path: Path) -> None:
    write_cold_start_inputs(tmp_path, truth="NOT_COLD")
    (tmp_path / "cold.csv").write_text("stale\n")

    res = run_cold_start(tmp_path)

    assert res.returncode != 0
    assert "no overlap with ChEMBL cold-start targets" in res.stderr
    assert not (tmp_path / "cold.csv").exists()


def test_cold_start_rejects_blank_ranking_target_id(tmp_path: Path) -> None:
    write_cold_start_inputs(tmp_path)
    (tmp_path / "cold.csv").write_text("stale\n")
    pd.DataFrame([
        {"target_id": "P1", "rank": 1},
        {"target_id": None, "rank": 2},
    ]).to_csv(
        tmp_path / "ranking.csv",
        index=False,
    )

    res = run_cold_start(tmp_path)

    assert res.returncode != 0
    assert "Cold-start ranking column 'target_id' contains blank values" in res.stderr
    assert not (tmp_path / "cold.csv").exists()


def test_cold_start_rejects_duplicate_ranking_target_id(tmp_path: Path) -> None:
    write_cold_start_inputs(tmp_path)
    (tmp_path / "cold.csv").write_text("stale\n")
    pd.DataFrame([
        {"target_id": "P1", "rank": 1},
        {"target_id": "P1", "rank": 2},
    ]).to_csv(
        tmp_path / "ranking.csv",
        index=False,
    )

    res = run_cold_start(tmp_path)

    assert res.returncode != 0
    assert "Cold-start ranking contains duplicate target_id values: P1" in res.stderr
    assert not (tmp_path / "cold.csv").exists()


def test_cold_start_sparse_declared_rank_is_not_promoted_by_row_position(
    tmp_path: Path,
) -> None:
    write_cold_start_inputs(tmp_path)
    pd.DataFrame([{"target_id": "P1", "final_rank": 100, "rank": 1}]).to_csv(
        tmp_path / "ranking.csv",
        index=False,
    )

    res = run_cold_start(tmp_path, "--allow-threshold-failure")

    assert res.returncode == 0, res.stderr
    row = pd.read_csv(tmp_path / "cold.csv").iloc[0]
    assert row["recall@1"] == 0
    assert row["recall@10"] == 0


def test_cold_start_rejects_boolean_declared_rank(tmp_path: Path) -> None:
    write_cold_start_inputs(tmp_path)
    pd.DataFrame([{"target_id": "P1", "final_rank": True}]).to_csv(
        tmp_path / "ranking.csv",
        index=False,
    )

    res = run_cold_start(tmp_path, "--allow-threshold-failure")

    assert res.returncode != 0
    assert "must not contain boolean values" in res.stderr
    assert not (tmp_path / "cold.csv").exists()


def test_cold_start_rejects_blank_truth_uniprot(tmp_path: Path) -> None:
    write_cold_start_inputs(tmp_path)
    (tmp_path / "cold.csv").write_text("stale\n")
    pd.DataFrame([{"uniprot": "P1"}, {"uniprot": None}]).to_csv(
        tmp_path / "truth.csv",
        index=False,
    )

    res = run_cold_start(tmp_path)

    assert res.returncode != 0
    assert "Cold-start ground truth column 'uniprot' contains blank values" in res.stderr
    assert not (tmp_path / "cold.csv").exists()


def test_cold_start_rejects_duplicate_truth_uniprot(tmp_path: Path) -> None:
    write_cold_start_inputs(tmp_path)
    (tmp_path / "cold.csv").write_text("stale\n")
    pd.DataFrame([{"uniprot": "P1"}, {"uniprot": "P1"}]).to_csv(
        tmp_path / "truth.csv",
        index=False,
    )

    res = run_cold_start(tmp_path)

    assert res.returncode != 0
    assert "Cold-start ground truth contains duplicate uniprot values: P1" in res.stderr
    assert not (tmp_path / "cold.csv").exists()


def test_cold_start_rejects_blank_chembl_uniprot(tmp_path: Path) -> None:
    write_cold_start_inputs(tmp_path)
    (tmp_path / "cold.csv").write_text("stale\n")
    pd.DataFrame([
        {"uniprot": " ", "molecule_chembl_id": "CHEMBL1"},
    ]).to_parquet(tmp_path / "chembl" / "human_activities.parquet")

    res = run_cold_start(tmp_path)

    assert res.returncode != 0
    assert "ChEMBL human activity reference column 'uniprot' contains blank values" in res.stderr
    assert not (tmp_path / "cold.csv").exists()


def test_cold_start_rejects_duplicate_chembl_activity_pair(tmp_path: Path) -> None:
    write_cold_start_inputs(tmp_path)
    (tmp_path / "cold.csv").write_text("stale\n")
    pd.DataFrame([
        {"uniprot": "P1", "molecule_chembl_id": "CHEMBL1"},
        {"uniprot": "P1", "molecule_chembl_id": "CHEMBL1"},
        {"uniprot": "P2", "molecule_chembl_id": "CHEMBL2"},
    ]).to_parquet(tmp_path / "chembl" / "human_activities.parquet")

    res = run_cold_start(tmp_path)

    assert res.returncode != 0
    assert (
        "ChEMBL human activity reference contains duplicate "
        "(uniprot, molecule_chembl_id) pair"
    ) in res.stderr
    assert not (tmp_path / "cold.csv").exists()


def test_cold_start_writes_metrics_for_valid_inputs(tmp_path: Path) -> None:
    write_cold_start_inputs(tmp_path)

    res = run_cold_start(tmp_path)

    assert res.returncode == 0, res.stderr
    metrics = pd.read_csv(tmp_path / "cold.csv").iloc[0]
    assert metrics["n_cold_targets"] == 2
    assert metrics["n_truth_in_cold"] == 1
    assert metrics["recall@1"] == 1.0
    assert bool(metrics["passes_threshold"])


def test_cold_start_fails_below_claim_thresholds(tmp_path: Path) -> None:
    write_cold_start_inputs(tmp_path)
    pd.DataFrame([
        {"target_id": "P3", "rank": 1},
        {"target_id": "P4", "rank": 2},
    ]).to_csv(
        tmp_path / "ranking.csv",
        index=False,
    )

    res = run_cold_start(tmp_path)

    assert res.returncode != 0
    assert "Cold-start recovery failed claim threshold" in res.stderr
    assert not (tmp_path / "cold.csv").exists()


def test_cold_start_threshold_failure_flag_is_diagnostic(tmp_path: Path) -> None:
    write_cold_start_inputs(tmp_path)
    pd.DataFrame([
        {"target_id": "P3", "rank": 1},
        {"target_id": "P4", "rank": 2},
    ]).to_csv(
        tmp_path / "ranking.csv",
        index=False,
    )

    res = run_cold_start(tmp_path, "--allow-threshold-failure")

    assert res.returncode == 0, res.stderr
    metrics = pd.read_csv(tmp_path / "cold.csv").iloc[0]
    assert metrics["recall@50"] == 0.0
    assert not bool(metrics["passes_threshold"])


def test_cold_start_rejects_invalid_claim_threshold(tmp_path: Path) -> None:
    write_cold_start_inputs(tmp_path)
    (tmp_path / "cold.csv").write_text("stale\n")

    res = run_cold_start(tmp_path, "--min-recall-at-50", "1.5")

    assert res.returncode != 0
    assert "--min-recall-at-50 must be a finite value in [0, 1]" in res.stderr
    assert not (tmp_path / "cold.csv").exists()


def test_analog_quality_requires_chembl_reference(tmp_path: Path) -> None:
    analogs = tmp_path / "analogs.csv"
    other = tmp_path / "other.csv"
    out = tmp_path / "analog_quality.csv"
    out.write_text("stale\n")
    pd.DataFrame([{"smiles": "CCO", "ra_score": 0.8}]).to_csv(analogs, index=False)
    pd.DataFrame([{"smiles": "CCN"}]).to_csv(other, index=False)

    res = run_script([
        "eval/analog_quality_eval.py",
        "--analogs-csv", str(analogs),
        "--other-cosing-csv", str(other),
        "--chembl-fp-parquet", str(tmp_path / "missing.parquet"),
        "--out-csv", str(out),
    ])

    assert res.returncode != 0
    assert "ChEMBL fingerprint parquet" in res.stderr
    assert not out.exists()


def test_analog_quality_writes_metrics_for_nonempty_references(tmp_path: Path) -> None:
    analogs = tmp_path / "analogs.csv"
    other = tmp_path / "other.csv"
    chembl = tmp_path / "chembl.parquet"
    out = tmp_path / "analog_quality.csv"
    pd.DataFrame([{"smiles": "CCO", "ra_score": 0.8}]).to_csv(analogs, index=False)
    pd.DataFrame([{"smiles": "CCO"}]).to_csv(other, index=False)
    pd.DataFrame([{"smiles": "CCCC"}]).to_parquet(chembl)

    res = run_script([
        "eval/analog_quality_eval.py",
        "--analogs-csv", str(analogs),
        "--other-cosing-csv", str(other),
        "--chembl-fp-parquet", str(chembl),
        "--out-csv", str(out),
    ])

    assert res.returncode == 0, res.stderr
    metrics = pd.read_csv(out).iloc[0]
    assert metrics["n_analogs"] == 1
    assert metrics["recovery_fraction"] == 1.0


def test_analog_quality_rejects_invalid_analog_smiles(tmp_path: Path) -> None:
    analogs = tmp_path / "analogs.csv"
    other = tmp_path / "other.csv"
    chembl = tmp_path / "chembl.parquet"
    out = tmp_path / "analog_quality.csv"
    out.write_text("stale\n")
    pd.DataFrame([{"smiles": "not_a_smiles", "ra_score": 0.8}]).to_csv(analogs, index=False)
    pd.DataFrame([{"smiles": "CCO"}]).to_csv(other, index=False)
    pd.DataFrame([{"smiles": "CCCC"}]).to_parquet(chembl)

    res = run_script([
        "eval/analog_quality_eval.py",
        "--analogs-csv", str(analogs),
        "--other-cosing-csv", str(other),
        "--chembl-fp-parquet", str(chembl),
        "--out-csv", str(out),
    ])

    assert res.returncode != 0
    assert "Analog CSV contains invalid SMILES" in res.stderr
    assert not out.exists()


def test_analog_quality_rejects_duplicate_canonical_analog_smiles(tmp_path: Path) -> None:
    analogs = tmp_path / "analogs.csv"
    other = tmp_path / "other.csv"
    chembl = tmp_path / "chembl.parquet"
    out = tmp_path / "analog_quality.csv"
    out.write_text("stale\n")
    pd.DataFrame(
        [
            {"smiles": "CCO", "ra_score": 0.8},
            {"smiles": "OCC", "ra_score": 0.9},
        ]
    ).to_csv(analogs, index=False)
    pd.DataFrame([{"smiles": "CCO"}]).to_csv(other, index=False)
    pd.DataFrame([{"smiles": "CCCC"}]).to_parquet(chembl)

    res = run_script([
        "eval/analog_quality_eval.py",
        "--analogs-csv", str(analogs),
        "--other-cosing-csv", str(other),
        "--chembl-fp-parquet", str(chembl),
        "--out-csv", str(out),
        "--min-recovery", "0",
        "--min-novelty", "0",
        "--min-mean-ra-score", "0",
    ])

    assert res.returncode != 0
    assert "Analog CSV contains duplicate canonical SMILES" in res.stderr
    assert "CCO" in res.stderr
    assert not out.exists()


def test_analog_quality_rejects_invalid_reference_smiles(tmp_path: Path) -> None:
    analogs = tmp_path / "analogs.csv"
    other = tmp_path / "other.csv"
    chembl = tmp_path / "chembl.parquet"
    out = tmp_path / "analog_quality.csv"
    out.write_text("stale\n")
    pd.DataFrame([{"smiles": "CCO", "ra_score": 0.8}]).to_csv(analogs, index=False)
    pd.DataFrame([{"smiles": "CCN"}]).to_csv(other, index=False)
    pd.DataFrame([{"smiles": "not_a_smiles"}]).to_parquet(chembl)

    res = run_script([
        "eval/analog_quality_eval.py",
        "--analogs-csv", str(analogs),
        "--other-cosing-csv", str(other),
        "--chembl-fp-parquet", str(chembl),
        "--out-csv", str(out),
    ])

    assert res.returncode != 0
    assert "ChEMBL fingerprint parquet contains invalid SMILES" in res.stderr
    assert not out.exists()


def test_analog_quality_rejects_missing_ra_score_column(tmp_path: Path) -> None:
    analogs = tmp_path / "analogs.csv"
    other = tmp_path / "other.csv"
    chembl = tmp_path / "chembl.parquet"
    out = tmp_path / "analog_quality.csv"
    out.write_text("stale\n")
    pd.DataFrame([{"smiles": "CCO"}]).to_csv(analogs, index=False)
    pd.DataFrame([{"smiles": "CCO"}]).to_csv(other, index=False)
    pd.DataFrame([{"smiles": "CCCC"}]).to_parquet(chembl)

    res = run_script([
        "eval/analog_quality_eval.py",
        "--analogs-csv", str(analogs),
        "--other-cosing-csv", str(other),
        "--chembl-fp-parquet", str(chembl),
        "--out-csv", str(out),
    ])

    assert res.returncode != 0
    assert "missing required column 'ra_score'" in res.stderr
    assert "synthesizability quality gate" in res.stderr
    assert not out.exists()


def test_analog_quality_rejects_non_numeric_ra_score(tmp_path: Path) -> None:
    analogs = tmp_path / "analogs.csv"
    other = tmp_path / "other.csv"
    chembl = tmp_path / "chembl.parquet"
    out = tmp_path / "analog_quality.csv"
    out.write_text("stale\n")
    pd.DataFrame([{"smiles": "CCO", "ra_score": "not_numeric"}]).to_csv(
        analogs,
        index=False,
    )
    pd.DataFrame([{"smiles": "CCO"}]).to_csv(other, index=False)
    pd.DataFrame([{"smiles": "CCCC"}]).to_parquet(chembl)

    res = run_script([
        "eval/analog_quality_eval.py",
        "--analogs-csv", str(analogs),
        "--other-cosing-csv", str(other),
        "--chembl-fp-parquet", str(chembl),
        "--out-csv", str(out),
    ])

    assert res.returncode != 0
    assert "Analog CSV column 'ra_score' contains invalid values" in res.stderr
    assert not out.exists()


def test_analog_quality_rejects_out_of_range_ra_score(tmp_path: Path) -> None:
    analogs = tmp_path / "analogs.csv"
    other = tmp_path / "other.csv"
    chembl = tmp_path / "chembl.parquet"
    out = tmp_path / "analog_quality.csv"
    out.write_text("stale\n")
    pd.DataFrame([{"smiles": "CCO", "ra_score": 1.2}]).to_csv(
        analogs,
        index=False,
    )
    pd.DataFrame([{"smiles": "CCO"}]).to_csv(other, index=False)
    pd.DataFrame([{"smiles": "CCCC"}]).to_parquet(chembl)

    res = run_script([
        "eval/analog_quality_eval.py",
        "--analogs-csv", str(analogs),
        "--other-cosing-csv", str(other),
        "--chembl-fp-parquet", str(chembl),
        "--out-csv", str(out),
    ])

    assert res.returncode != 0
    assert "Analog CSV column 'ra_score' contains invalid values" in res.stderr
    assert "expected finite values in [0, 1]" in res.stderr
    assert not out.exists()


def test_analog_quality_fails_below_claim_thresholds(tmp_path: Path) -> None:
    analogs = tmp_path / "analogs.csv"
    other = tmp_path / "other.csv"
    chembl = tmp_path / "chembl.parquet"
    out = tmp_path / "analog_quality.csv"
    out.write_text("stale\n")
    pd.DataFrame([{"smiles": "CCO", "ra_score": 0.8}]).to_csv(analogs, index=False)
    pd.DataFrame([{"smiles": "CCO"}]).to_csv(other, index=False)
    pd.DataFrame([{"smiles": "CCO"}]).to_parquet(chembl)

    res = run_script([
        "eval/analog_quality_eval.py",
        "--analogs-csv", str(analogs),
        "--other-cosing-csv", str(other),
        "--chembl-fp-parquet", str(chembl),
        "--out-csv", str(out),
    ])

    assert res.returncode != 0
    assert "Analog quality failed claim threshold" in res.stderr
    assert "novelty_fraction=0.000" in res.stderr
    assert not out.exists()


def test_analog_quality_threshold_failure_flag_is_diagnostic(tmp_path: Path) -> None:
    analogs = tmp_path / "analogs.csv"
    other = tmp_path / "other.csv"
    chembl = tmp_path / "chembl.parquet"
    out = tmp_path / "analog_quality.csv"
    pd.DataFrame([{"smiles": "CCO", "ra_score": 0.8}]).to_csv(analogs, index=False)
    pd.DataFrame([{"smiles": "CCO"}]).to_csv(other, index=False)
    pd.DataFrame([{"smiles": "CCO"}]).to_parquet(chembl)

    res = run_script([
        "eval/analog_quality_eval.py",
        "--analogs-csv", str(analogs),
        "--other-cosing-csv", str(other),
        "--chembl-fp-parquet", str(chembl),
        "--out-csv", str(out),
        "--allow-threshold-failure",
    ])

    assert res.returncode == 0, res.stderr
    metrics = pd.read_csv(out).iloc[0]
    assert metrics["novelty_fraction"] == 0.0
    assert not bool(metrics["passes_threshold"])


def test_analog_quality_rejects_invalid_claim_threshold(tmp_path: Path) -> None:
    analogs = tmp_path / "analogs.csv"
    other = tmp_path / "other.csv"
    chembl = tmp_path / "chembl.parquet"
    out = tmp_path / "analog_quality.csv"
    out.write_text("stale\n")
    pd.DataFrame([{"smiles": "CCO", "ra_score": 0.8}]).to_csv(analogs, index=False)
    pd.DataFrame([{"smiles": "CCO"}]).to_csv(other, index=False)
    pd.DataFrame([{"smiles": "CCCC"}]).to_parquet(chembl)

    res = run_script([
        "eval/analog_quality_eval.py",
        "--analogs-csv", str(analogs),
        "--other-cosing-csv", str(other),
        "--chembl-fp-parquet", str(chembl),
        "--out-csv", str(out),
        "--min-novelty", "-0.1",
    ])

    assert res.returncode != 0
    assert "--min-novelty must be a finite value in [0, 1]" in res.stderr
    assert not out.exists()


def test_pharmacophore_conservation_requires_confirmed_atoms(tmp_path: Path) -> None:
    parent = tmp_path / "parent.sdf"
    analogs = tmp_path / "analogs.csv"
    consensus = tmp_path / "consensus.json"
    out = tmp_path / "pharmacophore.csv"
    out.write_text("stale\n")
    writer = Chem.SDWriter(str(parent))
    writer.write(Chem.MolFromSmiles("CCO"))
    writer.close()
    pd.DataFrame([{"smiles": "CCO"}]).to_csv(analogs, index=False)
    consensus.write_text(json.dumps({"feature": {"confirmed_atoms": []}}))

    res = run_script([
        "eval/pharmacophore_conservation_eval.py",
        "--parent-sdf", str(parent),
        "--consensus-json", str(consensus),
        "--analogs-csv", str(analogs),
        "--out-csv", str(out),
    ])

    assert res.returncode != 0
    assert "no confirmed atoms" in res.stderr
    assert not out.exists()


def test_pharmacophore_conservation_rejects_invalid_consensus_json(tmp_path: Path) -> None:
    parent = tmp_path / "parent.sdf"
    analogs = tmp_path / "analogs.csv"
    consensus = tmp_path / "consensus.json"
    out = tmp_path / "pharmacophore.csv"
    out.write_text("stale\n")
    writer = Chem.SDWriter(str(parent))
    writer.write(Chem.MolFromSmiles("CCO"))
    writer.close()
    pd.DataFrame([{"smiles": "CCO"}]).to_csv(analogs, index=False)
    consensus.write_text("{not-json")

    res = run_script([
        "eval/pharmacophore_conservation_eval.py",
        "--parent-sdf", str(parent),
        "--consensus-json", str(consensus),
        "--analogs-csv", str(analogs),
        "--out-csv", str(out),
    ])

    assert res.returncode != 0
    assert "Consensus pharmacophore JSON failed to parse" in res.stderr
    assert not out.exists()


def test_pharmacophore_conservation_rejects_non_object_consensus(tmp_path: Path) -> None:
    parent = tmp_path / "parent.sdf"
    analogs = tmp_path / "analogs.csv"
    consensus = tmp_path / "consensus.json"
    out = tmp_path / "pharmacophore.csv"
    out.write_text("stale\n")
    writer = Chem.SDWriter(str(parent))
    writer.write(Chem.MolFromSmiles("CCO"))
    writer.close()
    pd.DataFrame([{"smiles": "CCO"}]).to_csv(analogs, index=False)
    consensus.write_text(json.dumps([{"confirmed_atoms": [0]}]))

    res = run_script([
        "eval/pharmacophore_conservation_eval.py",
        "--parent-sdf", str(parent),
        "--consensus-json", str(consensus),
        "--analogs-csv", str(analogs),
        "--out-csv", str(out),
    ])

    assert res.returncode != 0
    assert "Consensus pharmacophore JSON must be an object" in res.stderr
    assert not out.exists()


def test_pharmacophore_conservation_rejects_missing_confirmed_atoms_field(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent.sdf"
    analogs = tmp_path / "analogs.csv"
    consensus = tmp_path / "consensus.json"
    out = tmp_path / "pharmacophore.csv"
    out.write_text("stale\n")
    writer = Chem.SDWriter(str(parent))
    writer.write(Chem.MolFromSmiles("CCO"))
    writer.close()
    pd.DataFrame([{"smiles": "CCO"}]).to_csv(analogs, index=False)
    consensus.write_text(json.dumps({
        "valid": {"confirmed_atoms": [0]},
        "missing": {"votes": {"plip": [1]}},
    }))

    res = run_script([
        "eval/pharmacophore_conservation_eval.py",
        "--parent-sdf", str(parent),
        "--consensus-json", str(consensus),
        "--analogs-csv", str(analogs),
        "--out-csv", str(out),
    ])

    assert res.returncode != 0
    assert "Consensus pharmacophore entry missing required field 'confirmed_atoms'" in res.stderr
    assert not out.exists()


def test_pharmacophore_conservation_rejects_bad_confirmed_atoms(tmp_path: Path) -> None:
    parent = tmp_path / "parent.sdf"
    analogs = tmp_path / "analogs.csv"
    consensus = tmp_path / "consensus.json"
    out = tmp_path / "pharmacophore.csv"
    out.write_text("stale\n")
    writer = Chem.SDWriter(str(parent))
    writer.write(Chem.MolFromSmiles("CCO"))
    writer.close()
    pd.DataFrame([{"smiles": "CCO"}]).to_csv(analogs, index=False)
    consensus.write_text(json.dumps({"feature": {"confirmed_atoms": "0"}}))

    res = run_script([
        "eval/pharmacophore_conservation_eval.py",
        "--parent-sdf", str(parent),
        "--consensus-json", str(consensus),
        "--analogs-csv", str(analogs),
        "--out-csv", str(out),
    ])

    assert res.returncode != 0
    assert "confirmed_atoms must be a list of non-negative integers" in res.stderr
    assert not out.exists()


def test_pharmacophore_conservation_rejects_duplicate_confirmed_atoms(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent.sdf"
    analogs = tmp_path / "analogs.csv"
    consensus = tmp_path / "consensus.json"
    out = tmp_path / "pharmacophore.csv"
    out.write_text("stale\n")
    writer = Chem.SDWriter(str(parent))
    writer.write(Chem.MolFromSmiles("CCO"))
    writer.close()
    pd.DataFrame([{"smiles": "CCO"}]).to_csv(analogs, index=False)
    consensus.write_text(json.dumps({"target": {"confirmed_atoms": [0, 1, 1]}}))

    res = run_script([
        "eval/pharmacophore_conservation_eval.py",
        "--parent-sdf", str(parent),
        "--consensus-json", str(consensus),
        "--analogs-csv", str(analogs),
        "--out-csv", str(out),
    ])

    assert res.returncode != 0
    assert "confirmed_atoms contains duplicate atom indices" in res.stderr
    assert not out.exists()


def test_pharmacophore_conservation_allows_target_local_index_reuse(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent.sdf"
    analogs = tmp_path / "analogs.csv"
    consensus = tmp_path / "consensus.json"
    out = tmp_path / "pharmacophore.csv"
    writer = Chem.SDWriter(str(parent))
    writer.write(Chem.MolFromSmiles("CCO"))
    writer.close()
    pd.DataFrame([{"smiles": "CCO"}]).to_csv(analogs, index=False)
    consensus.write_text(json.dumps({
        "target_a": pose_consensus_entry([0, 1]),
        "target_b": pose_consensus_entry([1, 2]),
    }))

    res = run_script([
        "eval/pharmacophore_conservation_eval.py",
        "--parent-sdf", str(parent),
        "--consensus-json", str(consensus),
        "--analogs-csv", str(analogs),
        "--out-csv", str(out),
    ])

    assert res.returncode != 0
    assert "Interaction-atom conservation evaluation is unavailable" in res.stderr
    assert "duplicate" not in res.stderr
    assert not out.exists()


def test_pharmacophore_conservation_does_not_assume_parent_sdf_atom_range(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent.sdf"
    analogs = tmp_path / "analogs.csv"
    consensus = tmp_path / "consensus.json"
    out = tmp_path / "pharmacophore.csv"
    out.write_text("stale\n")
    writer = Chem.SDWriter(str(parent))
    writer.write(Chem.MolFromSmiles("CCO"))
    writer.close()
    pd.DataFrame([{"smiles": "CCO"}]).to_csv(analogs, index=False)
    consensus.write_text(json.dumps({"feature": pose_consensus_entry([99])}))

    res = run_script([
        "eval/pharmacophore_conservation_eval.py",
        "--parent-sdf", str(parent),
        "--consensus-json", str(consensus),
        "--analogs-csv", str(analogs),
        "--out-csv", str(out),
    ])

    assert res.returncode != 0
    assert "Interaction-atom conservation evaluation is unavailable" in res.stderr
    assert "target-specific atom-map sidecar is required" in res.stderr
    assert not out.exists()


def test_pharmacophore_conservation_rejects_invalid_analog_smiles(tmp_path: Path) -> None:
    parent = tmp_path / "parent.sdf"
    analogs = tmp_path / "analogs.csv"
    consensus = tmp_path / "consensus.json"
    out = tmp_path / "pharmacophore.csv"
    out.write_text("stale\n")
    writer = Chem.SDWriter(str(parent))
    writer.write(Chem.MolFromSmiles("CCO"))
    writer.close()
    pd.DataFrame([{"smiles": "not_a_smiles"}]).to_csv(analogs, index=False)
    consensus.write_text(json.dumps({"feature": pose_consensus_entry([0])}))

    res = run_script([
        "eval/pharmacophore_conservation_eval.py",
        "--parent-sdf", str(parent),
        "--consensus-json", str(consensus),
        "--analogs-csv", str(analogs),
        "--out-csv", str(out),
    ])

    assert res.returncode != 0
    assert "Analog CSV contains invalid SMILES" in res.stderr
    assert not out.exists()


def test_pharmacophore_conservation_fails_below_claim_threshold(tmp_path: Path) -> None:
    parent = tmp_path / "parent.sdf"
    analogs = tmp_path / "analogs.csv"
    consensus = tmp_path / "consensus.json"
    out = tmp_path / "pharmacophore.csv"
    out.write_text("stale\n")
    writer = Chem.SDWriter(str(parent))
    writer.write(Chem.MolFromSmiles("CCO"))
    writer.close()
    pd.DataFrame([{"smiles": "CCN"}]).to_csv(analogs, index=False)
    consensus.write_text(json.dumps({"feature": pose_consensus_entry([2])}))

    res = run_script([
        "eval/pharmacophore_conservation_eval.py",
        "--parent-sdf", str(parent),
        "--consensus-json", str(consensus),
        "--analogs-csv", str(analogs),
        "--out-csv", str(out),
    ])

    assert res.returncode != 0
    assert "Interaction-atom conservation evaluation is unavailable" in res.stderr
    assert "target-specific atom-map sidecar is required" in res.stderr
    assert not out.exists()


def test_pharmacophore_conservation_threshold_override_cannot_bypass_atom_map(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent.sdf"
    analogs = tmp_path / "analogs.csv"
    consensus = tmp_path / "consensus.json"
    out = tmp_path / "pharmacophore.csv"
    writer = Chem.SDWriter(str(parent))
    writer.write(Chem.MolFromSmiles("CCO"))
    writer.close()
    pd.DataFrame([{"smiles": "CCN"}]).to_csv(analogs, index=False)
    consensus.write_text(json.dumps({"feature": pose_consensus_entry([2])}))

    res = run_script([
        "eval/pharmacophore_conservation_eval.py",
        "--parent-sdf", str(parent),
        "--consensus-json", str(consensus),
        "--analogs-csv", str(analogs),
        "--out-csv", str(out),
        "--allow-threshold-failure",
    ])

    assert res.returncode != 0
    assert "Interaction-atom conservation evaluation is unavailable" in res.stderr
    assert "--allow-threshold-failure cannot override" in res.stderr
    assert not out.exists()


def test_pharmacophore_conservation_scores_target_conditioned_anchor_retention(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent.sdf"
    analogs = tmp_path / "analogs.csv"
    consensus = tmp_path / "consensus.json"
    anchor_map = tmp_path / "interaction_anchor_map.json"
    out = tmp_path / "pharmacophore.csv"
    detail = tmp_path / "pharmacophore_detail.csv"
    writer = Chem.SDWriter(str(parent))
    writer.write(Chem.MolFromSmiles("CC(=O)O"))
    writer.close()
    pd.DataFrame(
        [
            {"analog_id": "retained", "smiles": "CCC(=O)O"},
            {"analog_id": "lost", "smiles": "CCN"},
        ]
    ).to_csv(analogs, index=False)
    consensus.write_text(
        json.dumps({"P12345": pose_consensus_entry([2, 3])})
    )
    write_interaction_anchor_map(
        anchor_map,
        parent_sdf=parent,
        consensus_json=consensus,
        target_id="P12345",
        confirmed_atoms=[2, 3],
    )

    res = run_script(
        [
            "eval/pharmacophore_conservation_eval.py",
            "--parent-sdf",
            str(parent),
            "--consensus-json",
            str(consensus),
            "--interaction-anchor-map",
            str(anchor_map),
            "--analogs-csv",
            str(analogs),
            "--min-preserved-fraction",
            "0.5",
            "--out-csv",
            str(out),
            "--out-detail-csv",
            str(detail),
        ]
    )

    assert res.returncode == 0, res.stderr
    metric = pd.read_csv(out).iloc[0]
    assert metric["preserved_fraction"] == 0.5
    assert bool(metric["passes_threshold"])
    assert not bool(metric["analog_pose_verified"])
    assert not bool(metric["claimable"])
    assert (
        metric["evaluation_basis"]
        == "pose_supported_parent_anchor_conservative_mcs_feature_preservation"
    )
    rows = pd.read_csv(detail).set_index("analog_id")
    assert rows.loc["retained", "anchor_preservation_score"] == 1.0
    assert rows.loc["lost", "anchor_preservation_score"] == 0.0
    assert rows.loc["retained", "evaluation_basis"] == (
        "pose_supported_parent_anchor_conservative_mcs_feature_preservation"
    )
    assert "mapping_count" in rows.columns
    assert "mapping_ambiguous" in rows.columns
    assert "mapping_truncated" in rows.columns


def test_pharmacophore_conservation_exposes_ambiguous_conservative_mcs(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent.sdf"
    analogs = tmp_path / "analogs.csv"
    consensus = tmp_path / "consensus.json"
    anchor_map = tmp_path / "interaction_anchor_map.json"
    out = tmp_path / "pharmacophore.csv"
    detail = tmp_path / "pharmacophore_detail.csv"
    writer = Chem.SDWriter(str(parent))
    writer.write(Chem.MolFromSmiles("CCO"))
    writer.close()
    pd.DataFrame([{"analog_id": "ambiguous", "smiles": "NC(=O)N"}]).to_csv(
        analogs,
        index=False,
    )
    consensus.write_text(json.dumps({"P12345": pose_consensus_entry([0])}))
    write_interaction_anchor_map(
        anchor_map,
        parent_sdf=parent,
        consensus_json=consensus,
        target_id="P12345",
        confirmed_atoms=[0],
    )

    res = run_script(
        [
            "eval/pharmacophore_conservation_eval.py",
            "--parent-sdf",
            str(parent),
            "--consensus-json",
            str(consensus),
            "--interaction-anchor-map",
            str(anchor_map),
            "--analogs-csv",
            str(analogs),
            "--min-preserved-fraction",
            "0.0",
            "--out-csv",
            str(out),
            "--out-detail-csv",
            str(detail),
        ]
    )

    assert res.returncode == 0, res.stderr
    row = pd.read_csv(detail).iloc[0]
    assert row["evaluation_basis"] == (
        "pose_supported_parent_anchor_conservative_mcs_feature_preservation"
    )
    assert bool(row["mapping_ambiguous"])
    assert row["mapping_count"] == 2
    assert row["anchor_preservation_score"] == 0.0
    assert row["preserved_anchor_count"] == 0


def test_pharmacophore_conservation_fails_closed_below_mapped_anchor_threshold(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent.sdf"
    analogs = tmp_path / "analogs.csv"
    consensus = tmp_path / "consensus.json"
    anchor_map = tmp_path / "interaction_anchor_map.json"
    out = tmp_path / "pharmacophore.csv"
    out.write_text("stale\n")
    writer = Chem.SDWriter(str(parent))
    writer.write(Chem.MolFromSmiles("CC(=O)O"))
    writer.close()
    pd.DataFrame([{"smiles": "CCN"}]).to_csv(analogs, index=False)
    consensus.write_text(json.dumps({"P12345": pose_consensus_entry([2, 3])}))
    write_interaction_anchor_map(
        anchor_map,
        parent_sdf=parent,
        consensus_json=consensus,
        target_id="P12345",
        confirmed_atoms=[2, 3],
    )

    res = run_script(
        [
            "eval/pharmacophore_conservation_eval.py",
            "--parent-sdf",
            str(parent),
            "--consensus-json",
            str(consensus),
            "--interaction-anchor-map",
            str(anchor_map),
            "--analogs-csv",
            str(analogs),
            "--out-csv",
            str(out),
        ]
    )

    assert res.returncode != 0
    assert "Pharmacophore anchor preservation failed threshold" in res.stderr
    assert not out.exists()


def test_pharmacophore_conservation_rejects_stale_anchor_map_hash(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent.sdf"
    analogs = tmp_path / "analogs.csv"
    consensus = tmp_path / "consensus.json"
    anchor_map = tmp_path / "interaction_anchor_map.json"
    out = tmp_path / "pharmacophore.csv"
    writer = Chem.SDWriter(str(parent))
    writer.write(Chem.MolFromSmiles("CC(=O)O"))
    writer.close()
    pd.DataFrame([{"smiles": "CCC(=O)O"}]).to_csv(analogs, index=False)
    consensus.write_text(json.dumps({"P12345": pose_consensus_entry([2, 3])}))
    write_interaction_anchor_map(
        anchor_map,
        parent_sdf=parent,
        consensus_json=consensus,
        target_id="P12345",
        confirmed_atoms=[2, 3],
    )
    consensus.write_text(json.dumps({"P12345": pose_consensus_entry([2])}))

    res = run_script(
        [
            "eval/pharmacophore_conservation_eval.py",
            "--parent-sdf",
            str(parent),
            "--consensus-json",
            str(consensus),
            "--interaction-anchor-map",
            str(anchor_map),
            "--analogs-csv",
            str(analogs),
            "--out-csv",
            str(out),
        ]
    )

    assert res.returncode != 0
    assert "Interaction-anchor consensus hash mismatch" in res.stderr
    assert not out.exists()


def test_skin_efficacy_recovery_fails_on_missing_rankings_by_default(tmp_path: Path) -> None:
    gt = tmp_path / "gt.csv"
    out = tmp_path / "skin.csv"
    out.write_text("stale\n")
    ranked = tmp_path / "ranked"
    ranked.mkdir()
    pd.DataFrame([{
        "inci_name": "Retinol",
        "true_efficacies": "anti_aging",
    }]).to_csv(gt, index=False)

    res = run_script([
        "eval/skin_efficacy_recovery_eval.py",
        "--ranked-dir", str(ranked),
        "--ground-truth-csv", str(gt),
        "--out-csv", str(out),
    ])

    assert res.returncode != 0
    assert "Missing 1 ranking" in res.stderr
    assert not out.exists()


def test_skin_efficacy_recovery_allow_missing_is_diagnostic(tmp_path: Path) -> None:
    gt = tmp_path / "gt.csv"
    out = tmp_path / "skin.csv"
    ranked = tmp_path / "ranked"
    ranked.mkdir()
    pd.DataFrame([{
        "inci_name": "Retinol",
        "true_efficacies": "anti_aging",
    }]).to_csv(gt, index=False)

    res = run_script([
        "eval/skin_efficacy_recovery_eval.py",
        "--ranked-dir", str(ranked),
        "--ground-truth-csv", str(gt),
        "--out-csv", str(out),
        "--allow-missing-rankings",
        "--allow-threshold-failure",
    ])

    assert res.returncode == 0, res.stderr
    row = pd.read_csv(out).iloc[0]
    assert row["status"] == "no_ranking"
    assert not bool(row["passes_threshold"])


def test_skin_efficacy_recovery_missing_rankings_flag_does_not_hide_threshold_failure(
    tmp_path: Path,
) -> None:
    gt = tmp_path / "gt.csv"
    out = tmp_path / "skin.csv"
    out.write_text("stale\n")
    ranked = tmp_path / "ranked"
    ranked.mkdir()
    pd.DataFrame([{
        "inci_name": "Retinol",
        "true_efficacies": "anti_aging",
    }]).to_csv(gt, index=False)

    res = run_script([
        "eval/skin_efficacy_recovery_eval.py",
        "--ranked-dir", str(ranked),
        "--ground-truth-csv", str(gt),
        "--out-csv", str(out),
        "--allow-missing-rankings",
    ])

    assert res.returncode != 0
    assert "Skin-efficacy recovery failed claim threshold" in res.stderr
    assert "pass --allow-threshold-failure only for explicit diagnostics" in res.stderr
    assert not out.exists()


def test_skin_efficacy_recovery_rejects_blank_ground_truth_fields(tmp_path: Path) -> None:
    gt = tmp_path / "gt.csv"
    out = tmp_path / "skin.csv"
    out.write_text("stale\n")
    ranked = tmp_path / "ranked"
    ranked.mkdir()
    pd.DataFrame([{
        "inci_name": "Retinol",
        "true_efficacies": None,
    }]).to_csv(gt, index=False)

    res = run_script([
        "eval/skin_efficacy_recovery_eval.py",
        "--ranked-dir", str(ranked),
        "--ground-truth-csv", str(gt),
        "--out-csv", str(out),
    ])

    assert res.returncode != 0
    assert "Ground truth column 'true_efficacies' contains blank values" in res.stderr
    assert not out.exists()


def test_skin_efficacy_recovery_rejects_empty_true_efficacy_token(
    tmp_path: Path,
) -> None:
    gt = tmp_path / "gt.csv"
    out = tmp_path / "skin.csv"
    out.write_text("stale\n")
    ranked = tmp_path / "ranked"
    ranked.mkdir()
    pd.DataFrame([{
        "inci_name": "Retinol",
        "true_efficacies": "anti_aging;;retinoid",
    }]).to_csv(gt, index=False)

    res = run_script([
        "eval/skin_efficacy_recovery_eval.py",
        "--ranked-dir", str(ranked),
        "--ground-truth-csv", str(gt),
        "--out-csv", str(out),
    ])

    assert res.returncode != 0
    assert (
        "Ground truth column 'true_efficacies' contains empty ';'-separated "
        "value(s): Retinol"
        in res.stderr
    )
    assert not out.exists()


def test_skin_efficacy_recovery_rejects_duplicate_true_efficacy_token(
    tmp_path: Path,
) -> None:
    gt = tmp_path / "gt.csv"
    out = tmp_path / "skin.csv"
    out.write_text("stale\n")
    ranked = tmp_path / "ranked"
    ranked.mkdir()
    pd.DataFrame([{
        "inci_name": "Retinol",
        "true_efficacies": "anti_aging;anti_aging;retinoid",
    }]).to_csv(gt, index=False)

    res = run_script([
        "eval/skin_efficacy_recovery_eval.py",
        "--ranked-dir", str(ranked),
        "--ground-truth-csv", str(gt),
        "--out-csv", str(out),
    ])

    assert res.returncode != 0
    assert (
        "Ground truth column 'true_efficacies' contains duplicate "
        "';'-separated value(s): Retinol: anti_aging"
        in res.stderr
    )
    assert not out.exists()


def test_skin_efficacy_recovery_rejects_duplicate_ground_truth_inci_name(
    tmp_path: Path,
) -> None:
    gt = tmp_path / "gt.csv"
    out = tmp_path / "skin.csv"
    out.write_text("stale\n")
    ranked = tmp_path / "ranked"
    ranked.mkdir()
    pd.DataFrame([
        {
            "inci_name": "Retinol",
            "true_efficacies": "anti_aging",
        },
        {
            "inci_name": "Retinol",
            "true_efficacies": "anti_aging",
        },
    ]).to_csv(gt, index=False)

    res = run_script([
        "eval/skin_efficacy_recovery_eval.py",
        "--ranked-dir", str(ranked),
        "--ground-truth-csv", str(gt),
        "--out-csv", str(out),
    ])

    assert res.returncode != 0
    assert "Ground truth contains duplicate inci_name values: Retinol" in res.stderr
    assert not out.exists()


def test_skin_efficacy_recovery_rejects_ranking_without_efficacy_columns(tmp_path: Path) -> None:
    gt = tmp_path / "gt.csv"
    out = tmp_path / "skin.csv"
    out.write_text("stale\n")
    ranked = tmp_path / "ranked"
    ranked.mkdir()
    pd.DataFrame([{
        "inci_name": "Retinol",
        "true_efficacies": "anti_aging",
    }]).to_csv(gt, index=False)
    pd.DataFrame([{"target_id": "P1"}]).to_csv(
        ranked / "Retinol__ranked_targets_v3_with_efficacy.csv",
        index=False,
    )

    res = run_script([
        "eval/skin_efficacy_recovery_eval.py",
        "--ranked-dir", str(ranked),
        "--ground-truth-csv", str(gt),
        "--out-csv", str(out),
    ])

    assert res.returncode != 0
    assert "missing efficacy_top* columns" in res.stderr
    assert not out.exists()


def test_skin_efficacy_recovery_rejects_blank_top_efficacy_row(tmp_path: Path) -> None:
    gt = tmp_path / "gt.csv"
    out = tmp_path / "skin.csv"
    out.write_text("stale\n")
    ranked = tmp_path / "ranked"
    ranked.mkdir()
    pd.DataFrame([{
        "inci_name": "Retinol",
        "true_efficacies": "anti_aging",
    }]).to_csv(gt, index=False)
    pd.DataFrame([{
        "target_id": "P1",
        "efficacy_top1": " ",
        "efficacy_top2": None,
    }]).to_csv(
        ranked / "Retinol__ranked_targets_v3_with_efficacy.csv",
        index=False,
    )

    res = run_script([
        "eval/skin_efficacy_recovery_eval.py",
        "--ranked-dir", str(ranked),
        "--ground-truth-csv", str(gt),
        "--out-csv", str(out),
    ])

    assert res.returncode != 0
    assert "top-10 rows must each contain at least one non-empty efficacy_top*" in res.stderr
    assert "missing row index(es) 0" in res.stderr
    assert not out.exists()


def test_skin_efficacy_recovery_rejects_efficacy_without_paper_count(
    tmp_path: Path,
) -> None:
    gt = tmp_path / "gt.csv"
    out = tmp_path / "skin.csv"
    out.write_text("stale\n")
    ranked = tmp_path / "ranked"
    ranked.mkdir()
    pd.DataFrame([{
        "inci_name": "Retinol",
        "true_efficacies": "anti_aging",
    }]).to_csv(gt, index=False)
    pd.DataFrame([{"target_id": "P1", "efficacy_top1": "anti_aging"}]).to_csv(
        ranked / "Retinol__ranked_targets_v3_with_efficacy.csv",
        index=False,
    )

    res = run_script([
        "eval/skin_efficacy_recovery_eval.py",
        "--ranked-dir", str(ranked),
        "--ground-truth-csv", str(gt),
        "--out-csv", str(out),
    ])

    assert res.returncode != 0
    assert "must include paper-count evidence" in res.stderr
    assert not out.exists()


def test_skin_efficacy_recovery_rejects_partial_blank_top_efficacy_rows(
    tmp_path: Path,
) -> None:
    gt = tmp_path / "gt.csv"
    out = tmp_path / "skin.csv"
    out.write_text("stale\n")
    ranked = tmp_path / "ranked"
    ranked.mkdir()
    pd.DataFrame([{
        "inci_name": "Retinol",
        "true_efficacies": "anti_aging",
    }]).to_csv(gt, index=False)
    pd.DataFrame([
        {
            "target_id": "P1",
            "efficacy_top1": "anti_aging (12 papers)",
            "efficacy_top2": None,
        },
        {
            "target_id": "P2",
            "efficacy_top1": None,
            "efficacy_top2": " ",
        },
    ]).to_csv(
        ranked / "Retinol__ranked_targets_v3_with_efficacy.csv",
        index=False,
    )

    res = run_script([
        "eval/skin_efficacy_recovery_eval.py",
        "--ranked-dir", str(ranked),
        "--ground-truth-csv", str(gt),
        "--out-csv", str(out),
    ])

    assert res.returncode != 0
    assert "top-10 rows must each contain at least one non-empty efficacy_top*" in res.stderr
    assert "missing row index(es) 1" in res.stderr
    assert "Skin-efficacy recovery failed claim threshold" not in res.stderr
    assert not out.exists()


def test_skin_efficacy_recovery_rejects_duplicate_ranking_target_id(
    tmp_path: Path,
) -> None:
    gt = tmp_path / "gt.csv"
    out = tmp_path / "skin.csv"
    out.write_text("stale\n")
    ranked = tmp_path / "ranked"
    ranked.mkdir()
    pd.DataFrame([{
        "inci_name": "Retinol",
        "true_efficacies": "anti_aging",
    }]).to_csv(gt, index=False)
    pd.DataFrame([
        {"target_id": "P1", "efficacy_top1": "anti_aging (12 papers)"},
        {"target_id": "P1", "efficacy_top1": "anti_aging (12 papers)"},
    ]).to_csv(
        ranked / "Retinol__ranked_targets_v3_with_efficacy.csv",
        index=False,
    )

    res = run_script([
        "eval/skin_efficacy_recovery_eval.py",
        "--ranked-dir", str(ranked),
        "--ground-truth-csv", str(gt),
        "--out-csv", str(out),
    ])

    assert res.returncode != 0
    assert "Ranking contains duplicate target_id values: P1" in res.stderr
    assert not out.exists()


def test_skin_efficacy_recovery_rejects_duplicate_top_efficacy_label_in_row(
    tmp_path: Path,
) -> None:
    gt = tmp_path / "gt.csv"
    out = tmp_path / "skin.csv"
    out.write_text("stale\n")
    ranked = tmp_path / "ranked"
    ranked.mkdir()
    pd.DataFrame([{
        "inci_name": "Retinol",
        "true_efficacies": "anti_aging",
    }]).to_csv(gt, index=False)
    pd.DataFrame([{
        "target_id": "P1",
        "efficacy_top1": "anti_aging (12 papers)",
        "efficacy_top2": "anti_aging (8 papers)",
    }]).to_csv(
        ranked / "Retinol__ranked_targets_v3_with_efficacy.csv",
        index=False,
    )

    res = run_script([
        "eval/skin_efficacy_recovery_eval.py",
        "--ranked-dir", str(ranked),
        "--ground-truth-csv", str(gt),
        "--out-csv", str(out),
    ])

    assert res.returncode != 0
    assert "duplicate efficacy_top* label(s) at row index 0: anti_aging" in res.stderr
    assert not out.exists()


def test_skin_efficacy_recovery_allows_empty_lower_rank_efficacy_slots(tmp_path: Path) -> None:
    gt = tmp_path / "gt.csv"
    out = tmp_path / "skin.csv"
    ranked = tmp_path / "ranked"
    ranked.mkdir()
    pd.DataFrame([{
        "inci_name": "Retinol",
        "true_efficacies": "anti_aging",
    }]).to_csv(gt, index=False)
    pd.DataFrame([{
        "target_id": "P1",
        "efficacy_top1": "anti_aging (12 papers)",
        "efficacy_top2": None,
        "efficacy_top3": None,
    }]).to_csv(
        ranked / "Retinol__ranked_targets_v3_with_efficacy.csv",
        index=False,
    )

    res = run_script([
        "eval/skin_efficacy_recovery_eval.py",
        "--ranked-dir", str(ranked),
        "--ground-truth-csv", str(gt),
        "--out-csv", str(out),
    ])

    assert res.returncode == 0, res.stderr
    row = pd.read_csv(out).iloc[0]
    assert row["precision"] == 1.0
    assert row["recall"] == 1.0


def test_skin_efficacy_recovery_uses_declared_score_not_csv_row_order(
    tmp_path: Path,
) -> None:
    gt = tmp_path / "gt.csv"
    out = tmp_path / "skin.csv"
    ranked = tmp_path / "ranked"
    ranked.mkdir()
    pd.DataFrame([{
        "inci_name": "Retinol",
        "true_efficacies": "anti_aging",
    }]).to_csv(gt, index=False)
    rows = [
        {
            "target_id": "LOW",
            "final_score": 0.0,
            "efficacy_top1": "wrong_label (1 paper)",
        }
    ] + [
        {
            "target_id": f"P{index:02d}",
            "final_score": float(20 - index),
            "efficacy_top1": "anti_aging (1 paper)",
        }
        for index in range(10)
    ]
    pd.DataFrame(rows).to_csv(
        ranked / "Retinol__ranked_targets_v3_with_efficacy.csv", index=False
    )

    res = run_script([
        "eval/skin_efficacy_recovery_eval.py",
        "--ranked-dir", str(ranked),
        "--ground-truth-csv", str(gt),
        "--out-csv", str(out),
    ])

    assert res.returncode == 0, res.stderr
    row = pd.read_csv(out).iloc[0]
    assert row["precision"] == 1.0
    assert row["recall"] == 1.0


def test_skin_efficacy_recovery_fails_below_claim_thresholds(tmp_path: Path) -> None:
    gt = tmp_path / "gt.csv"
    out = tmp_path / "skin.csv"
    out.write_text("stale\n")
    ranked = tmp_path / "ranked"
    ranked.mkdir()
    pd.DataFrame([{
        "inci_name": "Retinol",
        "true_efficacies": "anti_aging",
    }]).to_csv(gt, index=False)
    pd.DataFrame([{"target_id": "P1", "efficacy_top1": "whitening (12 papers)"}]).to_csv(
        ranked / "Retinol__ranked_targets_v3_with_efficacy.csv",
        index=False,
    )

    res = run_script([
        "eval/skin_efficacy_recovery_eval.py",
        "--ranked-dir", str(ranked),
        "--ground-truth-csv", str(gt),
        "--out-csv", str(out),
    ])

    assert res.returncode != 0
    assert "Skin-efficacy recovery failed claim threshold" in res.stderr
    assert not out.exists()


def test_skin_efficacy_recovery_threshold_failure_flag_is_diagnostic(tmp_path: Path) -> None:
    gt = tmp_path / "gt.csv"
    out = tmp_path / "skin.csv"
    ranked = tmp_path / "ranked"
    ranked.mkdir()
    pd.DataFrame([{
        "inci_name": "Retinol",
        "true_efficacies": "anti_aging",
    }]).to_csv(gt, index=False)
    pd.DataFrame([{"target_id": "P1", "efficacy_top1": "whitening (12 papers)"}]).to_csv(
        ranked / "Retinol__ranked_targets_v3_with_efficacy.csv",
        index=False,
    )

    res = run_script([
        "eval/skin_efficacy_recovery_eval.py",
        "--ranked-dir", str(ranked),
        "--ground-truth-csv", str(gt),
        "--out-csv", str(out),
        "--allow-threshold-failure",
    ])

    assert res.returncode == 0, res.stderr
    row = pd.read_csv(out).iloc[0]
    assert row["precision"] == 0.0
    assert row["recall"] == 0.0
    assert not bool(row["passes_threshold"])


def test_skin_efficacy_recovery_rejects_invalid_claim_threshold(tmp_path: Path) -> None:
    gt = tmp_path / "gt.csv"
    out = tmp_path / "skin.csv"
    out.write_text("stale\n")
    ranked = tmp_path / "ranked"
    ranked.mkdir()
    pd.DataFrame([{
        "inci_name": "Retinol",
        "true_efficacies": "anti_aging",
    }]).to_csv(gt, index=False)

    res = run_script([
        "eval/skin_efficacy_recovery_eval.py",
        "--ranked-dir", str(ranked),
        "--ground-truth-csv", str(gt),
        "--out-csv", str(out),
        "--min-mean-recall", "nan",
    ])

    assert res.returncode != 0
    assert "--min-mean-recall must be a finite value in [0, 1]" in res.stderr
    assert not out.exists()
