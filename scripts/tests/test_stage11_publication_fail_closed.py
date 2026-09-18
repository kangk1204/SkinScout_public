"""Regression tests for Stage 11 publication placeholder gates."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
VALID_POSE_CONSENSUS = json.dumps({
    "P1": {
        "confirmed_atoms": [1, 2, 3],
        "pose_supported_interaction_atoms": [1, 2, 3],
        "evidence_label": "pose-supported interaction atoms",
        "coordinate_system": "boltz_complex_ligand_atom_order_0_based",
        "evidence_sources": ["plip", "prolif"],
        "consensus_min_votes": 2,
        "degraded": False,
        "claim_eligible": True,
        "votes": {
            "plip": [1, 2, 3],
            "prolif": [1, 2, 3],
        },
    },
}) + "\n"


def load_repro_pack_module():
    spec = importlib.util.spec_from_file_location(
        "stage11_repro_pack",
        ROOT / "scripts" / "stage11_repro_pack.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repro_pack_etkdg_seed_comes_from_run_artifact_not_current_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_repro_pack_module()
    monkeypatch.setenv("ETKDG_SEED", "123")
    run_dir = tmp_path / "old_run"
    conformers = run_dir / "01_structure" / "conformers_etkdg.sdf"
    conformers.parent.mkdir(parents=True)
    conformers.write_text(
        "old run\n  SkinScout\n\n  0  0  0  0  0  0  0  0  0  0999 V2000\n"
        "M  END\n>  <etkdg_random_seed>\n12345\n\n$$$$\n"
    )

    metadata = module._etkdg_seed_metadata(run_dir)

    assert metadata["seed"] == 12345
    assert metadata["observed"] is True
    assert metadata["source"] == "run_artifact_sdf_property"
    assert metadata["artifact_sha256"] == hashlib.sha256(
        conformers.read_bytes()
    ).hexdigest()
    assert metadata["seed"] != module._effective_etkdg_seed()


def test_repro_pack_marks_historical_etkdg_seed_unknown_without_property(
    tmp_path: Path,
) -> None:
    module = load_repro_pack_module()
    run_dir = tmp_path / "old_run"
    conformers = run_dir / "01_structure" / "conformers_etkdg.sdf"
    conformers.parent.mkdir(parents=True)
    conformers.write_text("historical artifact without seed property\n")

    metadata = module._etkdg_seed_metadata(run_dir)

    assert metadata["seed"] is None
    assert metadata["observed"] is False
    assert metadata["source"] == "run_artifact_property_unavailable"
    assert metadata["current_producer_default"] == 0xC05A


def run_script(
    args: list[str],
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    full_env = None if env is None else {**os.environ, **env}
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=full_env,
    )


def current_git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def write_repro_metadata_sidecars(repro_dir: Path) -> list[dict[str, object]]:
    clean_diff_sha = hashlib.sha256(b"").hexdigest()
    payloads = {
        "config_hash.txt": "a" * 64 + "\n",
        "git_commit.txt": current_git_commit() + "\n",
        "code_snapshot.json": json.dumps({
            "schema_version": "skinscout.stage11-code-snapshot.v1",
            "git_commit": current_git_commit(),
            "dirty": False,
            "tracked_diff_sha256": clean_diff_sha,
            "tracked_diff_bytes": 0,
            "untracked_files": [],
            "untracked_tree_sha256": hashlib.sha256(b"[]").hexdigest(),
            "snapshot_sha256": clean_diff_sha,
            "limitation": None,
        }) + "\n",
        "effective_config.json": json.dumps({
            "schema_version": "skinscout.stage11-effective-config.v1",
            "source": "config_yaml_plus_cli_overrides",
            "base_config_sha256": "a" * 64,
            "effective_config_sha256": "b" * 64,
            "overrides": [],
        }) + "\n",
        "random_seeds.json": '{"etkdg": 49242, "boltz": 0, "reinvent": 0}\n',
        "runtime_log.txt": "runtime\n",
        "tool_versions.lock": "pytest==0\n",
    }
    repro_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for rel, text in sorted(payloads.items()):
        path = repro_dir / rel
        path.write_text(text)
        records.append({
            "relative_path": rel,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    return records


def write_claim_ready_eval_manifest(
    path: Path,
    source_run_dir: Path | None = None,
) -> None:
    from scripts.tests.test_validate_activity_retrieval_gate import _contract
    from scripts.validate_activity_retrieval_gate import create_gate

    eval_output = path.parent / "cosmetic_retrospective.csv"
    eval_output.parent.mkdir(parents=True, exist_ok=True)
    if not eval_output.exists():
        eval_output.write_text("known,target\nA,B\n")
    cold_start_truth = path.parent / "cold_start_truth.csv"
    cold_start_truth.write_text("case_id,target_id\ncase,P1\n")
    eval_digest = hashlib.sha256(eval_output.read_bytes()).hexdigest()
    eval_bytes = eval_output.stat().st_size
    copied_ranking = (
        path.parent
        / "rankings"
        / "cosmetic_retro"
        / "case__ranked_targets_v3.csv"
    )
    copied_ranking.parent.mkdir(parents=True, exist_ok=True)
    if not copied_ranking.exists():
        copied_ranking.write_text(
            "target_id,final_score,source_count,sources\n"
            "P1,0.9,2,autodock;gnina\n"
        )
    cold_start_ranking = path.parent / "rankings" / "cold_start__comprehensive.csv"
    cold_start_ranking.parent.mkdir(parents=True, exist_ok=True)
    cold_start_ranking.write_text(
        "target_id,final_score,source_count,sources,skin_score\n"
        "P1,0.95,4,autodock;gnina;rtmscore;boltz,0.8\n"
    )
    source_run_dir = source_run_dir or path.parent / "source_runs" / "case"
    source_ranking = source_run_dir / "03_targets" / "ranked_targets_v3.csv"
    source_ranking.parent.mkdir(parents=True, exist_ok=True)
    source_ranking.write_text(copied_ranking.read_text())
    copied_digest = hashlib.sha256(copied_ranking.read_bytes()).hexdigest()
    copied_bytes = copied_ranking.stat().st_size
    cold_start_digest = hashlib.sha256(cold_start_ranking.read_bytes()).hexdigest()
    cold_start_bytes = cold_start_ranking.stat().st_size
    source_digest = hashlib.sha256(source_ranking.read_bytes()).hexdigest()
    source_bytes = source_ranking.stat().st_size
    snapshot_root = path.parent / "snapshot_data"
    workflow_config = snapshot_root / "workflow" / "config.yaml"
    target_classes = snapshot_root / "target_classes.parquet"
    chembl_fingerprints = snapshot_root / "fingerprints.parquet"
    training_sequence_db = snapshot_root / "seq_db"
    training_sequence_index = training_sequence_db / "index"
    training_ligands = snapshot_root / "ligands.smi"
    training_holo = snapshot_root / "holo.csv"
    for snapshot_path, text in (
        (workflow_config, "run_id: case\n"),
        (target_classes, "target-class-bytes\n"),
        (chembl_fingerprints, "fingerprint-bytes\n"),
        (training_sequence_index, "seq-index\n"),
        (training_ligands, "CCO ligand-1\n"),
        (training_holo, "target_id,pocket_sucos\nP1,0.1\n"),
    ):
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(text)
    activity_contract_id = hashlib.sha256(
        str(source_run_dir.resolve()).encode("utf-8")
    ).hexdigest()[:12]
    activity_contract_root = path.parent / f"activity-gate-contract-{activity_contract_id}"
    activity_contract_root.mkdir(exist_ok=True)
    activity_gate = activity_contract_root / "activity_retrieval_final_gate.flag"
    if not activity_gate.exists():
        create_gate(**_contract(activity_contract_root), out_gate=activity_gate)

    def file_snapshot(snapshot_path: Path, relative_path: str) -> dict[str, object]:
        return {
            "status": "present",
            "kind": "file",
            "path": str(snapshot_path),
            "relative_path": relative_path,
            "sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
            "bytes": snapshot_path.stat().st_size,
        }

    path.write_text(json.dumps({
        "eval_dir": str(path.parent),
        "rankings_dir": str(path.parent / "rankings"),
        "claim_ready": True,
        "claim_blockers": [],
        "activity_retrieval_gate": {
            "status": "pass",
            "schema_version": "skinscout.activity-retrieval-production-gate.v1",
            "path": str(activity_gate.resolve()),
            "bytes": activity_gate.stat().st_size,
            "sha256": hashlib.sha256(activity_gate.read_bytes()).hexdigest(),
        },
        "diagnostic_overrides": {
            "allow_incomplete_eval_inputs": False,
            "allow_empty_iteration": False,
            "allow_incomplete_leakage": False,
            "allow_threshold_failure": False,
            "allow_missing_input_runs": False,
        },
        "n_steps": 1,
        "n_passed": 1,
        "n_failed": 0,
        "steps": [
            {
                "name": "cosmetic_retrospective",
                "status": "passed",
                "command": [
                    sys.executable,
                    "eval/cosmetic_retrospective_eval.py",
                    "--out-csv",
                    "results/eval/cosmetic_retrospective.csv",
                ],
                "outputs": [str(eval_output)],
            },
        ],
        "input_status": {
            "status": "ok",
            "n_checked": 1,
            "n_incomplete": 0,
            "checked": [{
                "name": "cold_start",
                "present": [str(cold_start_truth)],
                "missing": [],
            }],
            "incomplete": [],
        },
        "leakage_status": {
            "status": "ok",
            "n_rows": 1,
            "n_incomplete": 0,
            "n_leak_flags": 0,
            "n_invalid_rows": 0,
        },
        "threshold_status": {
            "status": "ok",
            "n_checked": 1,
            "n_failed": 0,
            "n_missing_required": 0,
            "checked": [{
                "path": str(eval_output),
                "n_rows": 1,
                "n_failed_rows": 0,
                "n_invalid_rows": 0,
                "passed": True,
            }],
            "failures": [],
            "missing_required": [],
        },
        "ranking_metrics": {
            "cosmetic_retrospective": {
                "top1": 1.0,
                "top5": 1.0,
                "top10": 1.0,
                "n_cases": 1,
                "n_evaluated": 1,
                "n_no_ranking": 0,
                "passes_threshold": True,
                "passes_threshold_valid": True,
            },
        },
        "provenance": {
            "git_commit": current_git_commit(),
            "config_sha256": hashlib.sha256(workflow_config.read_bytes()).hexdigest(),
            "tool_versions": {
                "python": "3.11.15",
                "platform": "Linux-test",
                "rdkit": "2025.09.1",
            },
        },
        "artifact_snapshot": [
            {
                "path": str(eval_output),
                "relative_path": "cosmetic_retrospective.csv",
                "sha256": eval_digest,
                "bytes": eval_bytes,
            },
            {
                "path": str(copied_ranking),
                "relative_path": "rankings/cosmetic_retro/case__ranked_targets_v3.csv",
                "sha256": copied_digest,
                "bytes": copied_bytes,
            },
            {
                "path": str(cold_start_ranking),
                "relative_path": "rankings/cold_start__comprehensive.csv",
                "sha256": cold_start_digest,
                "bytes": cold_start_bytes,
            },
        ],
        "input_runs": {
            "status": "present",
            "manifest": "results/eval/collected_runs.json",
            "out_dir": str(path.parent),
            "rankings_dir": str(path.parent / "rankings"),
            "n_runs": 1,
            "runs": [{
                "run_id": "case",
                "canonical_smiles": "CCO",
                "run_dir": str(source_run_dir),
                "ranked_v3": str(source_ranking),
                "n_leakage_rows": 1,
                "copied": [str(copied_ranking)],
                "copied_artifacts": [{
                    "path": str(copied_ranking),
                    "source_path": str(source_ranking),
                    "source_sha256": source_digest,
                    "source_bytes": source_bytes,
                    "sha256": copied_digest,
                    "bytes": copied_bytes,
                }],
            }],
        },
        "data_snapshot": {
            "workflow_config": file_snapshot(workflow_config, "config.yaml"),
            "target_classes": file_snapshot(target_classes, "target_classes.parquet"),
            "chembl_fingerprints": file_snapshot(
                chembl_fingerprints,
                "fingerprints.parquet",
            ),
            "activity_retrieval_gate": file_snapshot(
                activity_gate,
                "activity_retrieval_final_gate.flag",
            ),
            "training_sequence_db": {
                "status": "present",
                "kind": "directory",
                "path": str(training_sequence_db),
                "n_files": 1,
                "entries": [{
                    "relative_path": "index",
                    "sha256": hashlib.sha256(
                        training_sequence_index.read_bytes()
                    ).hexdigest(),
                    "bytes": training_sequence_index.stat().st_size,
                }],
            },
            "training_ligands": file_snapshot(training_ligands, "ligands.smi"),
            "training_holo": file_snapshot(training_holo, "holo.csv"),
        },
        "top_target_rationale": [{
            "ranking": str(cold_start_ranking),
            "target_id": "P1",
            "score": 0.95,
            "rationale": {
                "source_count": 4,
                "sources": "autodock;gnina;rtmscore;boltz",
                "skin_score": 0.8,
            },
        }],
    }) + "\n")


def write_minimal_claim_sources(run_dir: Path, eval_dir: Path) -> None:
    for rel in (
        "09_report/index.html",
        "03_targets/ranked_targets_v3_with_efficacy.csv",
        "05_pharmacophore/consensus_pharmacophore_atoms.json",
        "05_6_analogs/top30_with_properties.csv",
        "07_md/trajectory_index.tsv",
        "07_md/mmgbsa.tsv",
        "07_5_retrosynthesis/synthesis_priority_ranking.csv",
    ):
        path = run_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            path.write_text(VALID_POSE_CONSENSUS)
        elif path.name == "ranked_targets_v3_with_efficacy.csv":
            path.write_text(
                "target_id,final_score,source_count,sources,efficacy_top1\n"
                "P1,1.0,3,autodock;gnina;rtmscore,hydration\n"
            )
        elif path.suffix in {".csv", ".tsv"}:
            sep = "\t" if path.suffix == ".tsv" else ","
            path.write_text(f"id{sep}score\nx{sep}1\n")
        else:
            path.write_text("<html><body><section>report panel target summary</section></body></html>\n")
    eval_dir.mkdir()
    (eval_dir / "cosmetic_retrospective.csv").write_text("known,target\nA,B\n")
    write_claim_ready_eval_manifest(
        eval_dir / "iteration_manifest.json",
        source_run_dir=run_dir,
    )


def test_figures_fail_without_sources_unless_placeholder_mode(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    run_dir.mkdir()
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    (out_dir / "fig01_workflow.svg").write_text("stale\n")
    (out_dir / "fig08_case_study.png").write_text("stale\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(tmp_path / "eval"),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "requires upstream/eval artifacts" in res.stderr
    assert not captions.exists()
    assert not (out_dir / "fig01_workflow.svg").exists()
    assert not (out_dir / "fig08_case_study.png").exists()


def test_figures_allow_placeholder_mode_writes_draft_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    run_dir.mkdir()

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(tmp_path / "eval"),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
        "--allow-placeholders",
    ])

    assert res.returncode == 0, res.stderr
    assert captions.exists()
    assert (out_dir / "fig01_workflow.svg").exists()
    assert (out_dir / "fig02_target_landscape.png").exists()


def test_figures_default_mode_marks_artifact_count_outputs_diagnostic(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    for rel in (
        "09_report/index.html",
        "03_targets/ranked_targets_v3_with_efficacy.csv",
        "05_pharmacophore/consensus_pharmacophore_atoms.json",
        "05_6_analogs/top30_with_properties.csv",
        "07_md/trajectory_index.tsv",
        "07_md/mmgbsa.tsv",
        "07_5_retrosynthesis/synthesis_priority_ranking.csv",
    ):
        path = run_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            path.write_text(VALID_POSE_CONSENSUS)
        elif path.name == "ranked_targets_v3_with_efficacy.csv":
            path.write_text(
                "target_id,final_score,source_count,sources,efficacy_top1\n"
                "P1,1.0,3,autodock;gnina;rtmscore,hydration\n"
            )
        elif path.suffix in {".csv", ".tsv"}:
            sep = "\t" if path.suffix == ".tsv" else ","
            path.write_text(f"id{sep}score\nx{sep}1\n")
        else:
            path.write_text("<html><body><section>report panel target summary</section></body></html>\n")
    eval_dir.mkdir()
    write_claim_ready_eval_manifest(
        eval_dir / "iteration_manifest.json",
        source_run_dir=run_dir,
    )
    (eval_dir / "cosmetic_retrospective.csv").write_text("known,target\nA,B\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode == 0, res.stderr
    manifest = json.loads(captions.read_text())
    assert {entry["status"] for entry in manifest.values()} == {"diagnostic_only"}
    assert {entry["diagnostic_only"] for entry in manifest.values()} == {True}
    assert {entry["claim_eligible"] for entry in manifest.values()} == {False}
    assert {entry["rendered_content"] for entry in manifest.values()} == {
        "artifact_evidence_counts"
    }
    for entry in manifest.values():
        assert entry["sources"]
        for source in entry["sources"]:
            assert set(source) == {"path", "metric", "value", "sha256"}
            assert source["metric"] in {
                "bytes",
                "claim-ready eval steps",
                "confirmed atoms",
                "html bytes",
                "items",
                "keys",
                "rows",
            }
            assert source["value"] > 0
            assert len(source["sha256"]) == 64
    assert "placeholder" not in captions.read_text().lower()
    assert "Diagnostic artifact-count panel" in (out_dir / "fig01_workflow.svg").read_text()
    assert (out_dir / "fig08_case_study.png").stat().st_size > 0


def test_publication_claim_manifest_binds_figures_captions_and_manuscript() -> None:
    rules = (ROOT / "workflow/rules/stage11_publication.smk").read_text()

    for relative_path in (
        "publication/figures/captions.json",
        "publication/figures/fig01_workflow.svg",
        "publication/figures/fig08_case_study.png",
        "publication/manuscript_draft/00_abstract.md",
        "publication/manuscript_draft/02_methods.md",
        "publication/manuscript_draft/05_references.bib",
    ):
        assert relative_path in rules
    assert '"figure_captions": rules.make_figures.output.captions' in rules
    assert (
        'S11_CLAIM_INPUTS["manuscript_methods"] = '
        "rules.write_manuscript_draft.output.methods"
    ) in rules


def test_figures_reject_eval_manifest_from_another_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    other_run_dir = tmp_path / "other_run"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    write_minimal_claim_sources(run_dir, eval_dir)
    write_claim_ready_eval_manifest(
        eval_dir / "iteration_manifest.json",
        source_run_dir=other_run_dir,
    )

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "input_runs must include the active --run-dir" in res.stderr
    assert not captions.exists()


def test_figures_reject_consensus_without_confirmed_atoms(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    write_minimal_claim_sources(run_dir, eval_dir)
    consensus = run_dir / "05_pharmacophore" / "consensus_pharmacophore_atoms.json"
    consensus.write_text('{"P1": {"confirmed_atoms": [], "votes": {}}}\n')

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "contains no confirmed atoms" in res.stderr
    assert not captions.exists()


def test_figures_reject_header_only_source_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    (out_dir / "fig02_target_landscape.png").write_text("stale\n")
    for rel in (
        "09_report/index.html",
        "05_pharmacophore/consensus_pharmacophore_atoms.json",
        "05_6_analogs/top30_with_properties.csv",
        "07_md/trajectory_index.tsv",
        "07_md/mmgbsa.tsv",
        "07_5_retrosynthesis/synthesis_priority_ranking.csv",
    ):
        path = run_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            path.write_text(VALID_POSE_CONSENSUS)
        elif path.suffix in {".csv", ".tsv"}:
            sep = "\t" if path.suffix == ".tsv" else ","
            path.write_text(f"id{sep}score\nx{sep}1\n")
        else:
            path.write_text("<html>report</html>\n")
    header_only = run_dir / "03_targets" / "ranked_targets_v3_with_efficacy.csv"
    header_only.parent.mkdir(parents=True, exist_ok=True)
    header_only.write_text("target_id,score\n")
    eval_dir.mkdir()
    write_claim_ready_eval_manifest(
        eval_dir / "iteration_manifest.json",
        source_run_dir=run_dir,
    )
    (eval_dir / "cosmetic_retrospective.csv").write_text("known,target\nA,B\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "contains no rows" in res.stderr
    assert not captions.exists()
    assert not (out_dir / "fig02_target_landscape.png").exists()


def test_figures_reject_ranked_targets_missing_publication_columns(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    (out_dir / "fig02_target_landscape.png").write_text("stale\n")
    write_minimal_claim_sources(run_dir, eval_dir)
    ranked = run_dir / "03_targets" / "ranked_targets_v3_with_efficacy.csv"
    ranked.write_text("target_id,final_score\nP1,1.0\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "ranked_targets_v3_with_efficacy.csv missing required publication source column" in res.stderr
    assert "source_count" in res.stderr
    assert "sources" in res.stderr
    assert "efficacy_top1" in res.stderr
    assert not captions.exists()
    assert not (out_dir / "fig02_target_landscape.png").exists()


def test_figures_reject_ranked_targets_duplicate_source_labels(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    (out_dir / "fig02_target_landscape.png").write_text("stale\n")
    write_minimal_claim_sources(run_dir, eval_dir)
    ranked = run_dir / "03_targets" / "ranked_targets_v3_with_efficacy.csv"
    ranked.write_text(
        "target_id,final_score,source_count,sources,efficacy_top1\n"
        "P1,1.0,3,autodock;autodock;gnina,hydration\n"
    )

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "sources contains duplicate label" in res.stderr
    assert "autodock" in res.stderr
    assert not captions.exists()
    assert not (out_dir / "fig02_target_landscape.png").exists()


def test_figures_reject_ranked_targets_empty_source_labels(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    (out_dir / "fig02_target_landscape.png").write_text("stale\n")
    write_minimal_claim_sources(run_dir, eval_dir)
    ranked = run_dir / "03_targets" / "ranked_targets_v3_with_efficacy.csv"
    ranked.write_text(
        "target_id,final_score,source_count,sources,efficacy_top1\n"
        "P1,1.0,2,autodock;;gnina,hydration\n"
    )

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "sources contains empty label" in res.stderr
    assert "ranked_targets_v3_with_efficacy.csv" in res.stderr
    assert not captions.exists()
    assert not (out_dir / "fig02_target_landscape.png").exists()


def test_figures_reject_blank_identifier_source_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    (out_dir / "fig02_target_landscape.png").write_text("stale\n")
    for rel in (
        "09_report/index.html",
        "05_pharmacophore/consensus_pharmacophore_atoms.json",
        "05_6_analogs/top30_with_properties.csv",
        "07_md/trajectory_index.tsv",
        "07_md/mmgbsa.tsv",
        "07_5_retrosynthesis/synthesis_priority_ranking.csv",
    ):
        path = run_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            path.write_text(VALID_POSE_CONSENSUS)
        elif path.suffix in {".csv", ".tsv"}:
            sep = "\t" if path.suffix == ".tsv" else ","
            path.write_text(f"id{sep}score\nx{sep}1\n")
        else:
            path.write_text("<html>report</html>\n")
    blank_id = run_dir / "03_targets" / "ranked_targets_v3_with_efficacy.csv"
    blank_id.parent.mkdir(parents=True, exist_ok=True)
    blank_id.write_text(
        "target_id,final_score,source_count,sources,efficacy_top1\n"
        " ,1,3,autodock;gnina;rtmscore,hydration\n"
    )
    eval_dir.mkdir()
    write_claim_ready_eval_manifest(
        eval_dir / "iteration_manifest.json",
        source_run_dir=run_dir,
    )
    (eval_dir / "cosmetic_retrospective.csv").write_text("known,target\nA,B\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "column 'target_id' contains blank values" in res.stderr
    assert not captions.exists()
    assert not (out_dir / "fig02_target_landscape.png").exists()


def test_figures_reject_duplicate_identifier_source_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    (out_dir / "fig02_target_landscape.png").write_text("stale\n")
    for rel in (
        "09_report/index.html",
        "05_pharmacophore/consensus_pharmacophore_atoms.json",
        "05_6_analogs/top30_with_properties.csv",
        "07_md/trajectory_index.tsv",
        "07_md/mmgbsa.tsv",
        "07_5_retrosynthesis/synthesis_priority_ranking.csv",
    ):
        path = run_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            path.write_text(VALID_POSE_CONSENSUS)
        elif path.suffix in {".csv", ".tsv"}:
            sep = "\t" if path.suffix == ".tsv" else ","
            path.write_text(f"id{sep}score\nx{sep}1\n")
        else:
            path.write_text("<html>report</html>\n")
    duplicate_id = run_dir / "03_targets" / "ranked_targets_v3_with_efficacy.csv"
    duplicate_id.parent.mkdir(parents=True, exist_ok=True)
    duplicate_id.write_text(
        "target_id,final_score,source_count,sources,efficacy_top1\n"
        "P1,1,3,autodock;gnina;rtmscore,hydration\n"
        "P1,0.9,3,autodock;gnina;rtmscore,hydration\n"
    )
    eval_dir.mkdir()
    write_claim_ready_eval_manifest(
        eval_dir / "iteration_manifest.json",
        source_run_dir=run_dir,
    )
    (eval_dir / "cosmetic_retrospective.csv").write_text("known,target\nA,B\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "column 'target_id' contains duplicate values" in res.stderr
    assert not captions.exists()
    assert not (out_dir / "fig02_target_landscape.png").exists()


def test_figures_reject_nonfinite_numeric_source_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    (out_dir / "fig02_target_landscape.png").write_text("stale\n")
    for rel in (
        "09_report/index.html",
        "05_pharmacophore/consensus_pharmacophore_atoms.json",
        "05_6_analogs/top30_with_properties.csv",
        "07_md/trajectory_index.tsv",
        "07_md/mmgbsa.tsv",
        "07_5_retrosynthesis/synthesis_priority_ranking.csv",
    ):
        path = run_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            path.write_text(VALID_POSE_CONSENSUS)
        elif path.suffix in {".csv", ".tsv"}:
            sep = "\t" if path.suffix == ".tsv" else ","
            path.write_text(f"id{sep}score\nx{sep}1\n")
        else:
            path.write_text("<html>report</html>\n")
    nonfinite = run_dir / "03_targets" / "ranked_targets_v3_with_efficacy.csv"
    nonfinite.parent.mkdir(parents=True, exist_ok=True)
    nonfinite.write_text(
        "target_id,final_score,source_count,sources,efficacy_top1\n"
        "P1,inf,3,autodock;gnina;rtmscore,hydration\n"
    )
    eval_dir.mkdir()
    write_claim_ready_eval_manifest(
        eval_dir / "iteration_manifest.json",
        source_run_dir=run_dir,
    )
    (eval_dir / "cosmetic_retrospective.csv").write_text("known,target\nA,B\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "non-finite numeric value" in res.stderr
    assert not captions.exists()
    assert not (out_dir / "fig02_target_landscape.png").exists()


def test_figures_reject_nonnumeric_score_source_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    (out_dir / "fig02_target_landscape.png").write_text("stale\n")
    for rel in (
        "09_report/index.html",
        "05_pharmacophore/consensus_pharmacophore_atoms.json",
        "05_6_analogs/top30_with_properties.csv",
        "07_md/trajectory_index.tsv",
        "07_md/mmgbsa.tsv",
        "07_5_retrosynthesis/synthesis_priority_ranking.csv",
    ):
        path = run_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            path.write_text(VALID_POSE_CONSENSUS)
        elif path.suffix in {".csv", ".tsv"}:
            sep = "\t" if path.suffix == ".tsv" else ","
            path.write_text(f"id{sep}score\nx{sep}1\n")
        else:
            path.write_text("<html>report</html>\n")
    nonnumeric = run_dir / "03_targets" / "ranked_targets_v3_with_efficacy.csv"
    nonnumeric.parent.mkdir(parents=True, exist_ok=True)
    nonnumeric.write_text(
        "target_id,final_score,source_count,sources,efficacy_top1\n"
        "P1,not-a-score,3,autodock;gnina;rtmscore,hydration\n"
    )
    eval_dir.mkdir()
    write_claim_ready_eval_manifest(
        eval_dir / "iteration_manifest.json",
        source_run_dir=run_dir,
    )
    (eval_dir / "cosmetic_retrospective.csv").write_text("known,target\nA,B\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "column 'final_score' must be numeric" in res.stderr
    assert not captions.exists()
    assert not (out_dir / "fig02_target_landscape.png").exists()


def test_figures_reject_non_claim_ready_eval_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    (out_dir / "fig07_eval_bars.png").write_text("stale\n")
    for rel in (
        "09_report/index.html",
        "03_targets/ranked_targets_v3_with_efficacy.csv",
        "05_pharmacophore/consensus_pharmacophore_atoms.json",
        "05_6_analogs/top30_with_properties.csv",
        "07_md/trajectory_index.tsv",
        "07_md/mmgbsa.tsv",
        "07_5_retrosynthesis/synthesis_priority_ranking.csv",
    ):
        path = run_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            path.write_text(VALID_POSE_CONSENSUS)
        elif path.suffix in {".csv", ".tsv"}:
            sep = "\t" if path.suffix == ".tsv" else ","
            path.write_text(f"id{sep}score\nx{sep}1\n")
        else:
            path.write_text("<html>report</html>\n")
    eval_dir.mkdir()
    cold_start_truth = eval_dir / "cold_start_truth.csv"
    cold_start_truth.write_text("case_id,target_id\ncase,P1\n")
    (eval_dir / "iteration_manifest.json").write_text(json.dumps({
        "n_steps": 1,
        "n_passed": 1,
        "n_failed": 0,
        "steps": [
            {
                "name": "cosmetic_retrospective",
                "status": "passed",
                "command": [
                    sys.executable,
                    "eval/cosmetic_retrospective_eval.py",
                    "--out-csv",
                    "results/eval/cosmetic_retrospective.csv",
                ],
                "outputs": [str(eval_dir / "cosmetic_retrospective.csv")],
            },
        ],
        "input_status": {
            "status": "ok",
            "n_checked": 1,
            "n_incomplete": 0,
            "checked": [{
                "name": "cold_start",
                "present": [str(cold_start_truth)],
                "missing": [],
            }],
            "incomplete": [],
        },
        "leakage_status": {
            "status": "ok",
            "n_rows": 1,
            "n_incomplete": 0,
            "n_leak_flags": 0,
            "n_invalid_rows": 0,
        },
        "threshold_status": {
            "status": "failed",
            "n_checked": 1,
            "n_failed": 1,
        },
    }) + "\n")
    (eval_dir / "cosmetic_retrospective.csv").write_text("known,target\nA,B\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "threshold_status.status must be 'ok'" in res.stderr
    assert not captions.exists()
    assert not (out_dir / "fig07_eval_bars.png").exists()


def test_figures_reject_eval_manifest_claim_ready_false(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["claim_ready"] = False
    payload["claim_blockers"] = [{"code": "diagnostic"}]
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "claim_ready must be true for publication" in res.stderr
    assert not captions.exists()


def test_figures_reject_eval_manifest_enabled_diagnostic_override(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["claim_ready"] = True
    payload["claim_blockers"] = []
    payload["diagnostic_overrides"]["allow_threshold_failure"] = True
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "diagnostic_overrides must all be false for publication" in res.stderr
    assert not captions.exists()


@pytest.mark.parametrize("mutation", ["missing_required", "unexpected_key"])
def test_figures_reject_eval_manifest_diagnostic_override_key_mismatch(
    tmp_path: Path,
    mutation: str,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    overrides = payload["diagnostic_overrides"]
    if mutation == "missing_required":
        del overrides["allow_missing_input_runs"]
    else:
        overrides["allow_unvalidated_publication"] = False
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "diagnostic_overrides keys must match publication contract" in res.stderr
    assert not captions.exists()


def test_figures_reject_eval_manifest_missing_required_threshold_metrics(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    (out_dir / "fig07_eval_bars.png").write_text("stale\n")
    for rel in (
        "09_report/index.html",
        "03_targets/ranked_targets_v3_with_efficacy.csv",
        "05_pharmacophore/consensus_pharmacophore_atoms.json",
        "05_6_analogs/top30_with_properties.csv",
        "07_md/trajectory_index.tsv",
        "07_md/mmgbsa.tsv",
        "07_5_retrosynthesis/synthesis_priority_ranking.csv",
    ):
        path = run_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            path.write_text(VALID_POSE_CONSENSUS)
        elif path.suffix in {".csv", ".tsv"}:
            sep = "\t" if path.suffix == ".tsv" else ","
            path.write_text(f"id{sep}score\nx{sep}1\n")
        else:
            path.write_text("<html>report</html>\n")
    eval_dir.mkdir()
    cold_start_truth = eval_dir / "cold_start_truth.csv"
    cold_start_truth.write_text("case_id,target_id\ncase,P1\n")
    (eval_dir / "iteration_manifest.json").write_text(json.dumps({
        "n_steps": 1,
        "n_passed": 1,
        "n_failed": 0,
        "steps": [
            {
                "name": "cold_start_comprehensive",
                "status": "passed",
                "command": [
                    sys.executable,
                    "eval/cold_start_eval.py",
                    "--out-csv",
                    "results/eval/cold_start_comprehensive.csv",
                ],
                "outputs": [str(eval_dir / "cold_start_comprehensive.csv")],
            },
        ],
        "input_status": {
            "status": "ok",
            "n_checked": 1,
            "n_incomplete": 0,
            "checked": [{
                "name": "cold_start",
                "present": [str(cold_start_truth)],
                "missing": [],
            }],
            "incomplete": [],
        },
        "leakage_status": {
            "status": "ok",
            "n_rows": 1,
            "n_incomplete": 0,
            "n_leak_flags": 0,
            "n_invalid_rows": 0,
        },
        "threshold_status": {
            "status": "ok",
            "n_checked": 1,
            "n_failed": 0,
            "n_missing_required": 1,
            "checked": [{
                "path": str(eval_dir / "cold_start_comprehensive.csv"),
                "n_rows": 1,
                "n_failed_rows": 0,
                "n_invalid_rows": 0,
                "passed": True,
            }],
            "failures": [],
            "missing_required": [
                {"path": "results/eval/cold_start_comprehensive.csv", "n_rows": 1},
            ],
        },
    }) + "\n")
    (eval_dir / "cosmetic_retrospective.csv").write_text("known,target\nA,B\n")
    (eval_dir / "cold_start_comprehensive.csv").write_text("metric,value\nx,1\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "missing required passes_threshold columns" in res.stderr
    assert not captions.exists()
    assert not (out_dir / "fig07_eval_bars.png").exists()


def test_figures_reject_eval_manifest_threshold_count_mismatch(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["threshold_status"]["n_checked"] = 2
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert (
        "threshold_status.checked length must equal threshold_status.n_checked"
        in res.stderr
    )
    assert not captions.exists()


def test_figures_reject_eval_manifest_duplicate_checked_input_name(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["input_status"]["n_checked"] = 2
    payload["input_status"]["checked"].append(
        dict(payload["input_status"]["checked"][0])
    )
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "input_status.checked contains duplicate name" in res.stderr
    assert not captions.exists()


def test_figures_reject_eval_manifest_missing_present_input_artifact(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["input_status"]["checked"][0]["present"] = [
        str(eval_dir / "missing_input.csv")
    ]
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "input_status.checked present path must exist" in res.stderr
    assert not captions.exists()


def test_figures_reject_eval_manifest_duplicate_present_input_artifact(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    first_present = payload["input_status"]["checked"][0]["present"][0]
    payload["input_status"]["checked"][0]["present"].append(first_present)
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "input_status.checked present contains duplicate path" in res.stderr
    assert not captions.exists()


def test_figures_reject_eval_manifest_duplicate_checked_threshold_path(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["threshold_status"]["n_checked"] = 2
    payload["threshold_status"]["checked"].append(
        dict(payload["threshold_status"]["checked"][0])
    )
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "threshold_status.checked contains duplicate path" in res.stderr
    assert not captions.exists()


def test_figures_reject_eval_manifest_duplicate_checked_threshold_metric_name(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    shadow_metric = eval_dir / "shadow" / "cosmetic_retrospective.csv"
    shadow_metric.parent.mkdir()
    shadow_metric.write_text((eval_dir / "cosmetic_retrospective.csv").read_text())
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["n_steps"] = 2
    payload["n_passed"] = 2
    payload["steps"].append({
        "name": "shadow_cosmetic_retrospective",
        "status": "passed",
        "command": [
            sys.executable,
            "eval/cosmetic_retrospective_eval.py",
            "--out-csv",
            "results/eval/shadow/cosmetic_retrospective.csv",
        ],
        "outputs": [str(shadow_metric)],
    })
    payload["threshold_status"]["n_checked"] = 2
    payload["threshold_status"]["checked"].append({
        "path": str(shadow_metric),
        "n_rows": 1,
        "n_failed_rows": 0,
        "n_invalid_rows": 0,
        "passed": True,
    })
    payload["artifact_snapshot"].append({
        "path": str(shadow_metric),
        "relative_path": "shadow/cosmetic_retrospective.csv",
        "sha256": hashlib.sha256(shadow_metric.read_bytes()).hexdigest(),
        "bytes": shadow_metric.stat().st_size,
    })
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "threshold_status.checked contains duplicate metric name" in res.stderr
    assert not captions.exists()


def test_figures_reject_eval_manifest_unsupported_ranking_metric(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["ranking_metrics"]["unsupported_metric"] = {
        "passes_threshold": True,
        "passes_threshold_valid": True,
    }
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "ranking_metrics contains metrics without threshold evidence" in res.stderr
    assert "unsupported_metric" in res.stderr
    assert not captions.exists()


def test_figures_reject_direct_eval_manifest_missing_status_counts(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["ranking_metrics"]["cosmetic_retrospective"].pop("n_evaluated")
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "field 'n_evaluated' must be a non-negative integer" in res.stderr
    assert not captions.exists()


def test_figures_reject_direct_eval_manifest_with_no_ranking_diagnostics(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["ranking_metrics"]["cosmetic_retrospective"]["n_no_ranking"] = 1
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "direct evaluator contains no_ranking diagnostics" in res.stderr
    assert not captions.exists()


def test_figures_reject_eval_manifest_step_without_outputs(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["steps"][0]["outputs"] = []
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "steps[0].outputs must be a non-empty list" in res.stderr
    assert not captions.exists()


def test_figures_reject_eval_manifest_step_missing_output_file(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["steps"][0]["outputs"] = [str(eval_dir / "missing_metric.csv")]
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "step output must exist and be non-empty" in res.stderr
    assert "missing_metric.csv" in res.stderr
    assert not captions.exists()


def test_figures_reject_eval_manifest_duplicate_step_output(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["n_steps"] = 2
    payload["n_passed"] = 2
    payload["steps"].append({
        "name": "duplicate_cosmetic_retrospective",
        "status": "passed",
        "command": [
            sys.executable,
            "eval/cosmetic_retrospective_eval.py",
            "--out-csv",
            "results/eval/cosmetic_retrospective.csv",
        ],
        "outputs": payload["steps"][0]["outputs"],
    })
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "steps contain duplicate output path" in res.stderr
    assert "cosmetic_retrospective.csv" in res.stderr
    assert not captions.exists()


def test_figures_reject_eval_manifest_threshold_not_step_output(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    decoy = eval_dir / "decoy_threshold.csv"
    decoy.write_text("metric,value\nx,1\n")
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["threshold_status"]["checked"][0]["path"] = str(decoy)
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert (
        "threshold_status.checked path must match a passed eval step output"
        in res.stderr
    )
    assert not captions.exists()


def test_figures_reject_eval_manifest_ranking_metric_threshold_mismatch(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["ranking_metrics"]["cosmetic_retrospective"]["passes_threshold"] = False
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "ranking_metrics threshold metric must pass" in res.stderr
    assert "cosmetic_retrospective" in res.stderr
    assert not captions.exists()


def test_figures_reject_eval_manifest_missing_input_runs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    payload.pop("input_runs")
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "missing object 'input_runs'" in res.stderr
    assert not captions.exists()


def test_figures_reject_eval_manifest_input_run_duplicate_run_dir_alias(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    duplicate = dict(payload["input_runs"]["runs"][0])
    duplicate["run_id"] = "case_alias"
    duplicate["run_dir"] = str(run_dir / ".")
    payload["input_runs"]["n_runs"] = 2
    payload["input_runs"]["runs"].append(duplicate)
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "input_runs contains duplicate run_dir" in res.stderr
    assert not captions.exists()


def test_figures_reject_eval_manifest_input_run_without_copied_artifacts(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["input_runs"]["runs"][0]["copied"] = []
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "input_runs.runs copied must be a non-empty list" in res.stderr
    assert not captions.exists()


def test_figures_reject_eval_manifest_input_run_duplicate_copied_path(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    run = payload["input_runs"]["runs"][0]
    run["copied"].append(run["copied"][0])
    run["copied_artifacts"].append(dict(run["copied_artifacts"][0]))
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "input_runs copied contains duplicate path" in res.stderr
    assert not captions.exists()


def test_figures_reject_eval_manifest_input_run_duplicate_copied_canonical_path(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    run = payload["input_runs"]["runs"][0]
    duplicate = dict(run["copied_artifacts"][0])
    duplicate["path"] = "rankings/cosmetic_retro/case__ranked_targets_v3.csv"
    run["copied"].append(duplicate["path"])
    run["copied_artifacts"].append(duplicate)
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "input_runs copied contains duplicate path" in res.stderr
    assert not captions.exists()


def test_figures_reject_eval_manifest_input_runs_missing_output_dirs(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["input_runs"].pop("rankings_dir")
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "input_runs.rankings_dir must be non-empty" in res.stderr
    assert not captions.exists()


def test_figures_reject_eval_manifest_input_runs_wrong_rankings_dir(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    wrong_rankings = tmp_path / "wrong_rankings"
    wrong_rankings.mkdir()
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["input_runs"]["rankings_dir"] = str(wrong_rankings)
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "input_runs.rankings_dir must match eval rankings directory" in res.stderr
    assert not captions.exists()


def test_figures_reject_eval_manifest_wrong_top_level_eval_dir(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    wrong_eval_dir = tmp_path / "wrong_eval"
    wrong_eval_dir.mkdir()
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["eval_dir"] = str(wrong_eval_dir)
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "eval_dir must match eval manifest directory" in res.stderr
    assert not captions.exists()


def test_figures_reject_eval_manifest_wrong_top_level_rankings_dir(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    wrong_rankings = tmp_path / "wrong_rankings"
    wrong_rankings.mkdir()
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["rankings_dir"] = str(wrong_rankings)
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "rankings_dir must match eval rankings directory" in res.stderr
    assert not captions.exists()


def test_figures_reject_eval_manifest_input_run_without_leakage_rows(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["input_runs"]["runs"][0]["n_leakage_rows"] = 0
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "input_runs.runs n_leakage_rows must be > 0" in res.stderr
    assert not captions.exists()


def test_figures_reject_eval_manifest_input_run_missing_copied_artifact_records(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["input_runs"]["runs"][0].pop("copied_artifacts")
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "copied_artifacts must be a list matching copied length" in res.stderr
    assert not captions.exists()


def test_figures_reject_eval_manifest_input_run_copied_artifact_digest_mismatch(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["input_runs"]["runs"][0]["copied_artifacts"][0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "copied_artifacts bytes/sha256 must match existing file" in res.stderr
    assert not captions.exists()


def test_figures_reject_eval_manifest_input_run_source_artifact_digest_mismatch(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["input_runs"]["runs"][0]["copied_artifacts"][0]["source_sha256"] = (
        "0" * 64
    )
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert (
        "copied_artifacts source_bytes/source_sha256 must match existing source"
        in res.stderr
    )
    assert not captions.exists()


def test_figures_reject_eval_manifest_input_run_source_outside_run_dir(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    source = Path(payload["input_runs"]["runs"][0]["copied_artifacts"][0]["source_path"])
    outside = eval_dir / "outside_source.csv"
    outside.write_text(source.read_text())
    record = payload["input_runs"]["runs"][0]["copied_artifacts"][0]
    record["source_path"] = str(outside)
    record["source_bytes"] = outside.stat().st_size
    record["source_sha256"] = hashlib.sha256(outside.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "copied_artifacts source_path must be inside run_dir" in res.stderr
    assert not captions.exists()


def test_figures_reject_eval_manifest_input_run_copied_not_snapshotted(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    copied_path = payload["input_runs"]["runs"][0]["copied"][0]
    payload["artifact_snapshot"] = [
        item for item in payload["artifact_snapshot"] if item["path"] != copied_path
    ]
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert (
        "input_runs copied artifact must be included in artifact_snapshot"
        in res.stderr
    )
    assert not captions.exists()


def test_figures_reject_eval_manifest_missing_data_snapshot(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    payload.pop("data_snapshot")
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "missing object 'data_snapshot'" in res.stderr
    assert not captions.exists()


def test_figures_reject_eval_manifest_missing_provenance(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    payload.pop("provenance")
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "missing object 'provenance'" in res.stderr
    assert not captions.exists()


def test_figures_reject_eval_manifest_invalid_provenance_commit(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["provenance"]["git_commit"] = "unknown"
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "provenance.git_commit must be a 40-hex SHA" in res.stderr
    assert not captions.exists()


def test_figures_reject_eval_manifest_config_hash_mismatch(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["provenance"]["config_sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert (
        "provenance.config_sha256 must match data_snapshot.workflow_config"
        in res.stderr
    )
    assert not captions.exists()


def test_figures_reject_eval_manifest_missing_artifact_snapshot(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    payload.pop("artifact_snapshot")
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "artifact_snapshot must be a non-empty list" in res.stderr
    assert not captions.exists()


def test_figures_reject_eval_manifest_artifact_snapshot_digest_mismatch(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["artifact_snapshot"][0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "artifact_snapshot[0] bytes/sha256 must match existing file" in res.stderr
    assert not captions.exists()


def test_figures_reject_eval_manifest_artifact_snapshot_duplicate_path(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    duplicate = dict(payload["artifact_snapshot"][1])
    duplicate["relative_path"] = Path(duplicate["path"]).name
    payload["artifact_snapshot"].append(duplicate)
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "artifact_snapshot contains duplicate path" in res.stderr
    assert not captions.exists()


def test_figures_reject_eval_manifest_artifact_snapshot_traversal_relative_path(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["artifact_snapshot"][0]["relative_path"] = "../cosmetic_retrospective.csv"
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert (
        "artifact_snapshot[0].relative_path must be a safe relative path"
        in res.stderr
    )
    assert not captions.exists()


def test_figures_reject_eval_manifest_artifact_snapshot_absolute_relative_path(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["artifact_snapshot"][0]["relative_path"] = str(tmp_path / "artifact.csv")
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert (
        "artifact_snapshot[0].relative_path must be a safe relative path"
        in res.stderr
    )
    assert not captions.exists()


def test_figures_reject_eval_manifest_artifact_snapshot_relative_path_mismatch(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["artifact_snapshot"][0]["relative_path"] = "renamed_metric.csv"
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "artifact_snapshot relative_path must match artifact path" in res.stderr
    assert not captions.exists()


def test_figures_reject_eval_manifest_artifact_snapshot_nested_basename_only(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["artifact_snapshot"][2]["relative_path"] = "cold_start__comprehensive.csv"
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "artifact_snapshot relative_path must match artifact path" in res.stderr
    assert not captions.exists()


def test_figures_reject_eval_manifest_data_snapshot_without_digest(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["data_snapshot"]["target_classes"].pop("sha256")
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "data_snapshot.target_classes.sha256 must be a SHA-256 digest" in res.stderr
    assert not captions.exists()


def test_figures_reject_eval_manifest_data_snapshot_file_traversal_relative_path(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["data_snapshot"]["workflow_config"]["relative_path"] = "../config.yaml"
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert (
        "data_snapshot.workflow_config.relative_path must be a safe relative path"
        in res.stderr
    )
    assert not captions.exists()


def test_figures_reject_eval_manifest_data_snapshot_digest_mismatch(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["data_snapshot"]["target_classes"]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert (
        "data_snapshot.target_classes bytes/sha256 must match existing file"
        in res.stderr
    )
    assert not captions.exists()


def test_figures_reject_eval_manifest_data_snapshot_duplicate_file_path(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    duplicate = dict(payload["data_snapshot"]["chembl_fingerprints"])
    duplicate["relative_path"] = "target_classes.parquet"
    payload["data_snapshot"]["target_classes"] = duplicate
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "data_snapshot contains duplicate file path" in res.stderr
    assert not captions.exists()


def test_figures_reject_eval_manifest_data_snapshot_file_inside_directory(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    sequence_index = (
        Path(payload["data_snapshot"]["training_sequence_db"]["path"]) / "index"
    )
    payload["data_snapshot"]["target_classes"] = {
        "status": "present",
        "kind": "file",
        "path": str(sequence_index),
        "relative_path": "seq_db/index",
        "sha256": hashlib.sha256(sequence_index.read_bytes()).hexdigest(),
        "bytes": sequence_index.stat().st_size,
    }
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "data_snapshot contains duplicate file path" in res.stderr
    assert "training_sequence_db" in res.stderr
    assert not captions.exists()


def test_figures_reject_eval_manifest_data_snapshot_directory_digest_mismatch(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["data_snapshot"]["training_sequence_db"]["entries"][0]["sha256"] = (
        "0" * 64
    )
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert (
        "data_snapshot.training_sequence_db.entries[0] bytes/sha256 must "
        "match existing file"
    ) in res.stderr
    assert not captions.exists()


def test_figures_reject_eval_manifest_data_snapshot_directory_duplicate_entry(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    directory_snapshot = payload["data_snapshot"]["training_sequence_db"]
    directory_snapshot["n_files"] = 2
    directory_snapshot["entries"].append(dict(directory_snapshot["entries"][0]))
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "data_snapshot directory entries contain duplicate relative_path" in res.stderr
    assert not captions.exists()


def test_figures_reject_eval_manifest_data_snapshot_directory_absolute_entry(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    directory_snapshot = payload["data_snapshot"]["training_sequence_db"]
    directory_snapshot["entries"][0]["relative_path"] = str(
        Path(directory_snapshot["path"]) / "index"
    )
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert (
        "data_snapshot.training_sequence_db.entries[0].relative_path must be "
        "a safe relative path"
    ) in res.stderr
    assert not captions.exists()


def test_figures_reject_eval_manifest_data_snapshot_directory_traversal_entry(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["data_snapshot"]["training_sequence_db"]["entries"][0][
        "relative_path"
    ] = "nested/../index"
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert (
        "data_snapshot.training_sequence_db.entries[0].relative_path must be "
        "a safe relative path"
    ) in res.stderr
    assert not captions.exists()


def test_figures_reject_eval_manifest_data_snapshot_without_file_kind(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["data_snapshot"]["target_classes"].pop("kind")
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert (
        "data_snapshot.target_classes.kind must be 'file' or 'directory'"
        in res.stderr
    )
    assert not captions.exists()


def test_figures_reject_eval_manifest_missing_top_target_rationale(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    payload.pop("top_target_rationale")
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "top_target_rationale must be a non-empty list" in res.stderr
    assert not captions.exists()


def test_figures_reject_eval_manifest_duplicate_top_target_ranking(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["top_target_rationale"].append(
        dict(payload["top_target_rationale"][0])
    )
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "top_target_rationale contains duplicate ranking" in res.stderr
    assert not captions.exists()


def test_figures_reject_eval_manifest_low_top_target_source_count(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    (out_dir / "fig07_eval_bars.png").write_text("stale\n")
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["top_target_rationale"][0]["rationale"] = {
        "source_count": 2,
        "sources": "autodock;gnina",
    }
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "source_count must be >= 3" in res.stderr
    assert not captions.exists()
    assert not (out_dir / "fig07_eval_bars.png").exists()


def test_figures_reject_eval_manifest_top_target_source_count_mismatch(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    (out_dir / "fig07_eval_bars.png").write_text("stale\n")
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["top_target_rationale"][0]["rationale"] = {
        "source_count": 4,
        "sources": "autodock",
    }
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "source_count 4 does not match sources list length 1" in res.stderr
    assert not captions.exists()
    assert not (out_dir / "fig07_eval_bars.png").exists()


def test_figures_reject_eval_manifest_blank_top_target_rationale_text(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    (out_dir / "fig07_eval_bars.png").write_text("stale\n")
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["top_target_rationale"][0]["rationale"]["efficacy_category"] = " "
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert (
        "top_target_rationale[0].rationale.efficacy_category must be non-empty"
        in res.stderr
    )
    assert not captions.exists()
    assert not (out_dir / "fig07_eval_bars.png").exists()


def test_figures_reject_eval_manifest_out_of_range_top_target_skin_score(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    (out_dir / "fig07_eval_bars.png").write_text("stale\n")
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["top_target_rationale"][0]["rationale"]["skin_score"] = 1.2
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert (
        "top_target_rationale[0].rationale.skin_score must be in [0, 1]"
        in res.stderr
    )
    assert not captions.exists()
    assert not (out_dir / "fig07_eval_bars.png").exists()


def test_figures_reject_eval_manifest_top_target_skin_score_mismatch(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    (out_dir / "fig07_eval_bars.png").write_text("stale\n")
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    ranking = Path(payload["top_target_rationale"][0]["ranking"])
    ranking.write_text(
        "target_id,final_score,source_count,sources,skin_score\n"
        "P1,0.95,4,autodock;gnina;rtmscore;boltz,0.8\n"
        "P2,0.50,4,autodock;gnina;rtmscore;boltz,0.4\n"
    )
    for artifact in payload["artifact_snapshot"]:
        if artifact["path"] == str(ranking):
            artifact["sha256"] = hashlib.sha256(ranking.read_bytes()).hexdigest()
            artifact["bytes"] = ranking.stat().st_size
    payload["top_target_rationale"][0]["rationale"]["skin_score"] = 0.5
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "rationale.skin_score must match ranking top row" in res.stderr
    assert not captions.exists()
    assert not (out_dir / "fig07_eval_bars.png").exists()


def test_figures_reject_eval_manifest_top_target_skin_score_missing_source(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    (out_dir / "fig07_eval_bars.png").write_text("stale\n")
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    ranking = Path(payload["top_target_rationale"][0]["ranking"])
    ranking.write_text(
        "target_id,final_score,source_count,sources\n"
        "P1,0.95,4,autodock;gnina;rtmscore;boltz\n"
        "P2,0.50,4,autodock;gnina;rtmscore;boltz\n"
    )
    for artifact in payload["artifact_snapshot"]:
        if artifact["path"] == str(ranking):
            artifact["sha256"] = hashlib.sha256(ranking.read_bytes()).hexdigest()
            artifact["bytes"] = ranking.stat().st_size
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "rationale.skin_score requires ranking top row column" in res.stderr
    assert not captions.exists()
    assert not (out_dir / "fig07_eval_bars.png").exists()


def test_figures_reject_eval_manifest_top_target_not_ranking_top_row(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["top_target_rationale"][0]["target_id"] = "P2"
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "target_id must match ranking top row" in res.stderr
    assert not captions.exists()


def test_figures_reject_eval_manifest_top_target_score_not_ranking_top_row(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["top_target_rationale"][0]["score"] = 0.5
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "score must match ranking top row" in res.stderr
    assert not captions.exists()


def test_figures_reject_eval_manifest_top_target_ranking_duplicate_target_id(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    ranking = Path(payload["top_target_rationale"][0]["ranking"])
    ranking.write_text(
        "target_id,final_score,source_count,sources\n"
        "P1,0.95,4,autodock;gnina;rtmscore;boltz\n"
        "P1,0.50,4,autodock;gnina;rtmscore;boltz\n"
    )
    for artifact in payload["artifact_snapshot"]:
        if artifact["path"] == str(ranking):
            artifact["sha256"] = hashlib.sha256(ranking.read_bytes()).hexdigest()
            artifact["bytes"] = ranking.stat().st_size
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "ranking contains duplicate target_id" in res.stderr
    assert not captions.exists()


def test_figures_reject_eval_manifest_top_target_ranking_extra_fields(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    (out_dir / "fig07_eval_bars.png").write_text("stale\n")
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    ranking = Path(payload["top_target_rationale"][0]["ranking"])
    ranking.write_text(
        "target_id,final_score,source_count,sources\n"
        "P1,0.95,4,autodock;gnina;rtmscore;boltz,extra\n"
        "P2,0.50,4,autodock;gnina;rtmscore;boltz\n"
    )
    for artifact in payload["artifact_snapshot"]:
        if artifact["path"] == str(ranking):
            artifact["sha256"] = hashlib.sha256(ranking.read_bytes()).hexdigest()
            artifact["bytes"] = ranking.stat().st_size
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "ranking row has extra field(s)" in res.stderr
    assert "row index 0" in res.stderr
    assert not captions.exists()
    assert not (out_dir / "fig07_eval_bars.png").exists()


def test_figures_reject_eval_manifest_ranking_source_count_mismatch_row(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    (out_dir / "fig07_eval_bars.png").write_text("stale\n")
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    ranking = Path(payload["top_target_rationale"][0]["ranking"])
    ranking.write_text(
        "target_id,final_score,source_count,sources\n"
        "P1,0.95,4,autodock;gnina;rtmscore;boltz\n"
        "P2,0.50,4,autodock\n"
    )
    for artifact in payload["artifact_snapshot"]:
        if artifact["path"] == str(ranking):
            artifact["sha256"] = hashlib.sha256(ranking.read_bytes()).hexdigest()
            artifact["bytes"] = ranking.stat().st_size
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "ranking source_count=4 but sources lists 1 label(s)" in res.stderr
    assert "row index 1" in res.stderr
    assert not captions.exists()
    assert not (out_dir / "fig07_eval_bars.png").exists()


def test_figures_reject_eval_manifest_ranking_out_of_range_skin_score_row(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    (out_dir / "fig07_eval_bars.png").write_text("stale\n")
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    ranking = Path(payload["top_target_rationale"][0]["ranking"])
    ranking.write_text(
        "target_id,final_score,source_count,sources,skin_score\n"
        "P1,0.95,4,autodock;gnina;rtmscore;boltz,0.8\n"
        "P2,0.50,4,autodock;gnina;rtmscore;boltz,1.2\n"
    )
    for artifact in payload["artifact_snapshot"]:
        if artifact["path"] == str(ranking):
            artifact["sha256"] = hashlib.sha256(ranking.read_bytes()).hexdigest()
            artifact["bytes"] = ranking.stat().st_size
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "ranking skin_score" in res.stderr
    assert "must be in [0, 1]" in res.stderr
    assert "row index 1" in res.stderr
    assert not captions.exists()
    assert not (out_dir / "fig07_eval_bars.png").exists()


def test_figures_reject_eval_manifest_ranking_blank_top_skin_score(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    (out_dir / "fig07_eval_bars.png").write_text("stale\n")
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    ranking = Path(payload["top_target_rationale"][0]["ranking"])
    ranking.write_text(
        "target_id,final_score,source_count,sources,skin_score\n"
        "P1,0.95,4,autodock;gnina;rtmscore;boltz,\n"
        "P2,0.50,4,autodock;gnina;rtmscore;boltz,0.7\n"
    )
    for artifact in payload["artifact_snapshot"]:
        if artifact["path"] == str(ranking):
            artifact["sha256"] = hashlib.sha256(ranking.read_bytes()).hexdigest()
            artifact["bytes"] = ranking.stat().st_size
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "ranking skin_score" in res.stderr
    assert "must be non-empty" in res.stderr
    assert "row index 0" in res.stderr
    assert not captions.exists()
    assert not (out_dir / "fig07_eval_bars.png").exists()


def test_figures_reject_eval_manifest_ranking_blank_non_top_skin_score(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    (out_dir / "fig07_eval_bars.png").write_text("stale\n")
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    ranking = Path(payload["top_target_rationale"][0]["ranking"])
    ranking.write_text(
        "target_id,final_score,source_count,sources,skin_score\n"
        "P1,0.95,4,autodock;gnina;rtmscore;boltz,0.8\n"
        "P2,0.50,4,autodock;gnina;rtmscore;boltz,\n"
    )
    for artifact in payload["artifact_snapshot"]:
        if artifact["path"] == str(ranking):
            artifact["sha256"] = hashlib.sha256(ranking.read_bytes()).hexdigest()
            artifact["bytes"] = ranking.stat().st_size
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "ranking skin_score" in res.stderr
    assert "must be non-empty" in res.stderr
    assert "row index 1" in res.stderr
    assert not captions.exists()
    assert not (out_dir / "fig07_eval_bars.png").exists()


def test_figures_reject_eval_manifest_ranking_blank_rationale_text_row(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    (out_dir / "fig07_eval_bars.png").write_text("stale\n")
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    ranking = Path(payload["top_target_rationale"][0]["ranking"])
    ranking.write_text(
        "target_id,final_score,source_count,sources,efficacy_category\n"
        "P1,0.95,4,autodock;gnina;rtmscore;boltz,barrier\n"
        "P2,0.50,4,autodock;gnina;rtmscore;boltz, \n"
    )
    for artifact in payload["artifact_snapshot"]:
        if artifact["path"] == str(ranking):
            artifact["sha256"] = hashlib.sha256(ranking.read_bytes()).hexdigest()
            artifact["bytes"] = ranking.stat().st_size
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "ranking column 'efficacy_category' is blank" in res.stderr
    assert "row index 1" in res.stderr
    assert not captions.exists()
    assert not (out_dir / "fig07_eval_bars.png").exists()


def test_figures_reject_eval_manifest_ranking_missing_top_rationale_text(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    (out_dir / "fig07_eval_bars.png").write_text("stale\n")
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    ranking = Path(payload["top_target_rationale"][0]["ranking"])
    ranking.write_text(
        "target_id,final_score,source_count,sources,skin_score,efficacy_category\n"
        "P1,0.95,4,autodock;gnina;rtmscore;boltz,0.8\n"
        "P2,0.50,4,autodock;gnina;rtmscore;boltz,0.7,barrier\n"
    )
    for artifact in payload["artifact_snapshot"]:
        if artifact["path"] == str(ranking):
            artifact["sha256"] = hashlib.sha256(ranking.read_bytes()).hexdigest()
            artifact["bytes"] = ranking.stat().st_size
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "ranking column 'efficacy_category' is blank" in res.stderr
    assert "row index 0" in res.stderr
    assert not captions.exists()
    assert not (out_dir / "fig07_eval_bars.png").exists()


def test_figures_reject_eval_manifest_ranking_missing_non_top_rationale_text(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    (out_dir / "fig07_eval_bars.png").write_text("stale\n")
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    ranking = Path(payload["top_target_rationale"][0]["ranking"])
    ranking.write_text(
        "target_id,final_score,source_count,sources,skin_score,efficacy_category\n"
        "P1,0.95,4,autodock;gnina;rtmscore;boltz,0.8,barrier\n"
        "P2,0.50,4,autodock;gnina;rtmscore;boltz,0.7\n"
    )
    for artifact in payload["artifact_snapshot"]:
        if artifact["path"] == str(ranking):
            artifact["sha256"] = hashlib.sha256(ranking.read_bytes()).hexdigest()
            artifact["bytes"] = ranking.stat().st_size
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "ranking column 'efficacy_category' is blank" in res.stderr
    assert "row index 1" in res.stderr
    assert not captions.exists()
    assert not (out_dir / "fig07_eval_bars.png").exists()


def test_figures_reject_eval_manifest_top_target_rationale_text_mismatch(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    (out_dir / "fig07_eval_bars.png").write_text("stale\n")
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    ranking = Path(payload["top_target_rationale"][0]["ranking"])
    ranking.write_text(
        "target_id,final_score,source_count,sources,skin_score,efficacy_category\n"
        "P1,0.95,4,autodock;gnina;rtmscore;boltz,0.8,barrier\n"
        "P2,0.50,4,autodock;gnina;rtmscore;boltz,0.7,barrier\n"
    )
    for artifact in payload["artifact_snapshot"]:
        if artifact["path"] == str(ranking):
            artifact["sha256"] = hashlib.sha256(ranking.read_bytes()).hexdigest()
            artifact["bytes"] = ranking.stat().st_size
    payload["top_target_rationale"][0]["rationale"]["efficacy_category"] = "pigment"
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "rationale.efficacy_category must match ranking top row" in res.stderr
    assert not captions.exists()
    assert not (out_dir / "fig07_eval_bars.png").exists()


def test_figures_reject_eval_manifest_top_target_rationale_text_missing_source(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    (out_dir / "fig07_eval_bars.png").write_text("stale\n")
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    ranking = Path(payload["top_target_rationale"][0]["ranking"])
    ranking.write_text(
        "target_id,final_score,source_count,sources,skin_score\n"
        "P1,0.95,4,autodock;gnina;rtmscore;boltz,0.8\n"
        "P2,0.50,4,autodock;gnina;rtmscore;boltz,0.7\n"
    )
    for artifact in payload["artifact_snapshot"]:
        if artifact["path"] == str(ranking):
            artifact["sha256"] = hashlib.sha256(ranking.read_bytes()).hexdigest()
            artifact["bytes"] = ranking.stat().st_size
    payload["top_target_rationale"][0]["rationale"]["efficacy_category"] = "barrier"
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "rationale.efficacy_category requires ranking top row column" in res.stderr
    assert not captions.exists()
    assert not (out_dir / "fig07_eval_bars.png").exists()


def test_figures_reject_eval_manifest_top_target_ranking_nonfinite_score(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    ranking = Path(payload["top_target_rationale"][0]["ranking"])
    ranking.write_text(
        "target_id,final_score,source_count,sources\n"
        "P1,0.95,4,autodock;gnina;rtmscore;boltz\n"
        "P2,inf,4,autodock;gnina;rtmscore;boltz\n"
    )
    for artifact in payload["artifact_snapshot"]:
        if artifact["path"] == str(ranking):
            artifact["sha256"] = hashlib.sha256(ranking.read_bytes()).hexdigest()
            artifact["bytes"] = ranking.stat().st_size
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "ranking score column 'final_score' must be finite" in res.stderr
    assert not captions.exists()


def test_figures_reject_eval_manifest_top_target_ranking_unsorted_score(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    ranking = Path(payload["top_target_rationale"][0]["ranking"])
    ranking.write_text(
        "target_id,final_score,source_count,sources\n"
        "P1,0.80,4,autodock;gnina;rtmscore;boltz\n"
        "P2,0.95,4,autodock;gnina;rtmscore;boltz\n"
    )
    for artifact in payload["artifact_snapshot"]:
        if artifact["path"] == str(ranking):
            artifact["sha256"] = hashlib.sha256(ranking.read_bytes()).hexdigest()
            artifact["bytes"] = ranking.stat().st_size
    payload["top_target_rationale"][0]["score"] = 0.80
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "ranking top row must have the best final_score" in res.stderr
    assert not captions.exists()


def test_figures_reject_eval_manifest_top_target_ranking_tied_top_score(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    ranking = Path(payload["top_target_rationale"][0]["ranking"])
    ranking.write_text(
        "target_id,final_score,source_count,sources\n"
        "P1,0.95,4,autodock;gnina;rtmscore;boltz\n"
        "P2,0.95,4,autodock;gnina;rtmscore;boltz\n"
    )
    for artifact in payload["artifact_snapshot"]:
        if artifact["path"] == str(ranking):
            artifact["sha256"] = hashlib.sha256(ranking.read_bytes()).hexdigest()
            artifact["bytes"] = ranking.stat().st_size
    payload["top_target_rationale"][0]["score"] = 0.95
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "ranking top row final_score is tied" in res.stderr
    assert not captions.exists()


def test_figures_reject_eval_manifest_top_target_boolean_score(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    ranking = Path(payload["top_target_rationale"][0]["ranking"])
    ranking.write_text(
        "target_id,final_score,source_count,sources\n"
        "P1,1.0,4,autodock;gnina;rtmscore;boltz\n"
    )
    for artifact in payload["artifact_snapshot"]:
        if artifact["path"] == str(ranking):
            artifact["sha256"] = hashlib.sha256(ranking.read_bytes()).hexdigest()
            artifact["bytes"] = ranking.stat().st_size
    payload["top_target_rationale"][0]["score"] = True
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "top_target_rationale score must be finite" in res.stderr
    assert not captions.exists()


def test_figures_reject_eval_manifest_top_target_sources_not_ranking_top_row(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["top_target_rationale"][0]["rationale"] = {
        "source_count": 4,
        "sources": "autodock;gnina;rtmscore;psichic",
    }
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert "rationale.sources must match ranking top row" in res.stderr
    assert not captions.exists()


def test_figures_reject_eval_manifest_top_target_ranking_not_snapshotted(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "figures"
    captions = out_dir / "captions.json"
    out_dir.mkdir()
    captions.write_text('{"stale": true}\n')
    write_minimal_claim_sources(run_dir, eval_dir)
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    ranking_path = payload["top_target_rationale"][0]["ranking"]
    payload["artifact_snapshot"] = [
        item for item in payload["artifact_snapshot"]
        if item["path"] != ranking_path
    ]
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_make_figures.py",
        "--run-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--out-dir", str(out_dir),
        "--out-captions", str(captions),
    ])

    assert res.returncode != 0
    assert "invalid source artifact" in res.stderr
    assert (
        "top_target_rationale ranking must be included in artifact_snapshot"
        in res.stderr
    )
    assert not captions.exists()


def test_manuscript_fails_on_placeholder_text_by_default(tmp_path: Path) -> None:
    out_dir = tmp_path / "manuscript"
    out_dir.mkdir()
    (out_dir / "00_abstract.md").write_text("stale\n")
    (out_dir / "05_references.bib").write_text("stale\n")

    res = run_script([
        "scripts/stage11_manuscript.py",
        "--run-dir", str(tmp_path / "run"),
        "--out-dir", str(out_dir),
    ])

    assert res.returncode != 0
    assert "placeholder scaffold text" in res.stderr
    assert not (out_dir / "00_abstract.md").exists()
    assert not (out_dir / "05_references.bib").exists()


def test_manuscript_writes_source_backed_claim_draft(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "manuscript"
    sources = tmp_path / "publication"
    captions = sources / "figures" / "captions.json"
    repro = sources / "reproducibility" / "artifact_manifest.json"
    data = sources / "data_availability.md"
    captions.parent.mkdir(parents=True)
    repro.parent.mkdir(parents=True)
    captions.write_text(json.dumps({
        "fig01_workflow.svg": {
            "caption": "Source-backed workflow evidence.",
            "status": "source_backed",
            "sources": [{
                "path": "run/09_report/index.html",
                "metric": "html bytes",
                "value": 42,
                "sha256": "a" * 64,
            }],
        },
    }) + "\n")
    repro.write_text(json.dumps({
        "run_dir": str(run_dir),
        "n_artifacts": 2,
        "artifacts": [
            {
                "relative_path": "09_report/index.html",
                "bytes": 42,
                "sha256": "a" * 64,
            },
            {
                "relative_path": "EVAL:iteration_manifest.json",
                "bytes": 10,
                "sha256": "b" * 64,
            },
        ],
        "metadata_files": write_repro_metadata_sidecars(repro.parent),
    }) + "\n")
    data.write_text(
        "# Data Availability\n\n"
        "- The derived `skin_efficacy.graphml` is released at\n"
        "  `Zenodo (DOI: 10.5281/zenodo.1234567)` under CC-BY-4.0.\n"
    )

    res = run_script([
        "scripts/stage11_manuscript.py",
        "--run-dir", str(run_dir),
        "--out-dir", str(out_dir),
        "--captions", str(captions),
        "--repro-manifest", str(repro),
        "--data-availability", str(data),
    ])

    assert res.returncode == 0, res.stderr
    assert len(list(out_dir.iterdir())) == 6
    text = "\n".join(path.read_text() for path in out_dir.iterdir())
    assert "Source-backed workflow evidence" in text
    assert "10.5281/zenodo.1234567" in text
    assert "placeholder" not in text.lower()
    assert "stale" not in text.lower()


def test_manuscript_rejects_malformed_data_availability_doi(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "manuscript"
    sources = tmp_path / "publication"
    captions = sources / "figures" / "captions.json"
    repro = sources / "reproducibility" / "artifact_manifest.json"
    data = sources / "data_availability.md"
    out_dir.mkdir()
    (out_dir / "04_discussion.md").write_text("stale\n")
    captions.parent.mkdir(parents=True)
    repro.parent.mkdir(parents=True)
    captions.write_text(json.dumps({
        "fig01_workflow.svg": {
            "caption": "Source-backed workflow evidence.",
            "status": "source_backed",
            "sources": [{
                "path": "run/09_report/index.html",
                "metric": "bytes",
                "value": 42,
                "sha256": "a" * 64,
            }],
        },
    }) + "\n")
    repro.write_text(json.dumps({
        "run_dir": str(run_dir),
        "n_artifacts": 2,
        "artifacts": [
            {
                "relative_path": "09_report/index.html",
                "bytes": 42,
                "sha256": "a" * 64,
            },
            {
                "relative_path": "EVAL:iteration_manifest.json",
                "bytes": 10,
                "sha256": "b" * 64,
            },
        ],
        "metadata_files": write_repro_metadata_sidecars(repro.parent),
    }) + "\n")
    data.write_text(
        "# Data Availability\n\n"
        "DOI: 10.not-a-doi\n"
    )

    res = run_script([
        "scripts/stage11_manuscript.py",
        "--run-dir", str(run_dir),
        "--out-dir", str(out_dir),
        "--captions", str(captions),
        "--repro-manifest", str(repro),
        "--data-availability", str(data),
    ])

    assert res.returncode != 0
    assert "invalid DOI" in res.stderr
    assert not (out_dir / "04_discussion.md").exists()


def test_manuscript_rejects_duplicate_caption_sources(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "manuscript"
    sources = tmp_path / "publication"
    captions = sources / "figures" / "captions.json"
    repro = sources / "reproducibility" / "artifact_manifest.json"
    data = sources / "data_availability.md"
    out_dir.mkdir()
    (out_dir / "03_results.md").write_text("stale\n")
    captions.parent.mkdir(parents=True)
    repro.parent.mkdir(parents=True)
    caption_source = {
        "path": "run/09_report/index.html",
        "metric": "bytes",
        "value": 42,
        "sha256": "a" * 64,
    }
    captions.write_text(json.dumps({
        "fig01_workflow.svg": {
            "caption": "Source-backed workflow evidence.",
            "status": "source_backed",
            "sources": [caption_source, dict(caption_source)],
        },
    }) + "\n")
    repro.write_text(json.dumps({
        "run_dir": str(run_dir),
        "n_artifacts": 2,
        "artifacts": [
            {
                "relative_path": "09_report/index.html",
                "bytes": 42,
                "sha256": "a" * 64,
            },
            {
                "relative_path": "EVAL:iteration_manifest.json",
                "bytes": 10,
                "sha256": "b" * 64,
            },
        ],
    }) + "\n")
    data.write_text(
        "# Data Availability\n\n"
        "DOI: 10.5281/zenodo.1234567\n"
    )

    res = run_script([
        "scripts/stage11_manuscript.py",
        "--run-dir", str(run_dir),
        "--out-dir", str(out_dir),
        "--captions", str(captions),
        "--repro-manifest", str(repro),
        "--data-availability", str(data),
    ])

    assert res.returncode != 0
    assert "duplicate source" in res.stderr
    assert not (out_dir / "03_results.md").exists()


def test_manuscript_rejects_canonical_duplicate_caption_sources(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "manuscript"
    sources = tmp_path / "publication"
    captions = sources / "figures" / "captions.json"
    repro = sources / "reproducibility" / "artifact_manifest.json"
    data = sources / "data_availability.md"
    out_dir.mkdir()
    (out_dir / "03_results.md").write_text("stale\n")
    captions.parent.mkdir(parents=True)
    repro.parent.mkdir(parents=True)
    caption_source = {
        "path": "run/09_report/index.html",
        "metric": "bytes",
        "value": 42,
        "sha256": "a" * 64,
    }
    alias_source = dict(caption_source)
    alias_source["path"] = "run/09_report/./index.html"
    captions.write_text(json.dumps({
        "fig01_workflow.svg": {
            "caption": "Source-backed workflow evidence.",
            "status": "source_backed",
            "sources": [caption_source, alias_source],
        },
    }) + "\n")
    repro.write_text(json.dumps({
        "run_dir": str(run_dir),
        "n_artifacts": 2,
        "artifacts": [
            {
                "relative_path": "09_report/index.html",
                "bytes": 42,
                "sha256": "a" * 64,
            },
            {
                "relative_path": "EVAL:iteration_manifest.json",
                "bytes": 10,
                "sha256": "b" * 64,
            },
        ],
    }) + "\n")
    data.write_text(
        "# Data Availability\n\n"
        "DOI: 10.5281/zenodo.1234567\n"
    )

    res = run_script([
        "scripts/stage11_manuscript.py",
        "--run-dir", str(run_dir),
        "--out-dir", str(out_dir),
        "--captions", str(captions),
        "--repro-manifest", str(repro),
        "--data-availability", str(data),
    ])

    assert res.returncode != 0
    assert "duplicate source" in res.stderr
    assert not (out_dir / "03_results.md").exists()


def test_manuscript_rejects_caption_source_missing_from_repro_manifest(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "manuscript"
    sources = tmp_path / "publication"
    captions = sources / "figures" / "captions.json"
    repro = sources / "reproducibility" / "artifact_manifest.json"
    data = sources / "data_availability.md"
    out_dir.mkdir()
    (out_dir / "03_results.md").write_text("stale\n")
    captions.parent.mkdir(parents=True)
    repro.parent.mkdir(parents=True)
    captions.write_text(json.dumps({
        "fig01_workflow.svg": {
            "caption": "Source-backed workflow evidence.",
            "status": "source_backed",
            "sources": [{
                "path": "run/09_report/index.html",
                "metric": "bytes",
                "value": 42,
                "sha256": "a" * 64,
            }],
        },
    }) + "\n")
    repro.write_text(json.dumps({
        "run_dir": str(run_dir),
        "n_artifacts": 1,
        "artifacts": [{
            "relative_path": "EVAL:iteration_manifest.json",
            "bytes": 10,
            "sha256": "b" * 64,
        }],
        "metadata_files": write_repro_metadata_sidecars(repro.parent),
    }) + "\n")
    data.write_text(
        "# Data Availability\n\n"
        "DOI: 10.5281/zenodo.1234567\n"
    )

    res = run_script([
        "scripts/stage11_manuscript.py",
        "--run-dir", str(run_dir),
        "--out-dir", str(out_dir),
        "--captions", str(captions),
        "--repro-manifest", str(repro),
        "--data-availability", str(data),
    ])

    assert res.returncode != 0
    assert "captions source sha256 is missing" in res.stderr
    assert not (out_dir / "03_results.md").exists()


def test_manuscript_rejects_caption_source_path_mismatch(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "manuscript"
    sources = tmp_path / "publication"
    captions = sources / "figures" / "captions.json"
    repro = sources / "reproducibility" / "artifact_manifest.json"
    data = sources / "data_availability.md"
    out_dir.mkdir()
    (out_dir / "03_results.md").write_text("stale\n")
    captions.parent.mkdir(parents=True)
    repro.parent.mkdir(parents=True)
    captions.write_text(json.dumps({
        "fig01_workflow.svg": {
            "caption": "Source-backed workflow evidence.",
            "status": "source_backed",
            "sources": [{
                "path": "run/forged_report.html",
                "metric": "bytes",
                "value": 42,
                "sha256": "a" * 64,
            }],
        },
    }) + "\n")
    repro.write_text(json.dumps({
        "run_dir": str(run_dir),
        "n_artifacts": 2,
        "artifacts": [
            {
                "relative_path": "09_report/index.html",
                "bytes": 42,
                "sha256": "a" * 64,
            },
            {
                "relative_path": "EVAL:iteration_manifest.json",
                "bytes": 10,
                "sha256": "b" * 64,
            },
        ],
        "metadata_files": write_repro_metadata_sidecars(repro.parent),
    }) + "\n")
    data.write_text(
        "# Data Availability\n\n"
        "DOI: 10.5281/zenodo.1234567\n"
    )

    res = run_script([
        "scripts/stage11_manuscript.py",
        "--run-dir", str(run_dir),
        "--out-dir", str(out_dir),
        "--captions", str(captions),
        "--repro-manifest", str(repro),
        "--data-availability", str(data),
    ])

    assert res.returncode != 0
    assert "captions source path does not match" in res.stderr
    assert not (out_dir / "03_results.md").exists()


def test_manuscript_rejects_caption_source_bytes_mismatch(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "manuscript"
    sources = tmp_path / "publication"
    captions = sources / "figures" / "captions.json"
    repro = sources / "reproducibility" / "artifact_manifest.json"
    data = sources / "data_availability.md"
    out_dir.mkdir()
    (out_dir / "03_results.md").write_text("stale\n")
    captions.parent.mkdir(parents=True)
    repro.parent.mkdir(parents=True)
    captions.write_text(json.dumps({
        "fig01_workflow.svg": {
            "caption": "Source-backed workflow evidence.",
            "status": "source_backed",
            "sources": [{
                "path": "run/09_report/index.html",
                "metric": "bytes",
                "value": 43,
                "sha256": "a" * 64,
            }],
        },
    }) + "\n")
    repro.write_text(json.dumps({
        "run_dir": str(run_dir),
        "n_artifacts": 2,
        "artifacts": [
            {
                "relative_path": "09_report/index.html",
                "bytes": 42,
                "sha256": "a" * 64,
            },
            {
                "relative_path": "EVAL:iteration_manifest.json",
                "bytes": 10,
                "sha256": "b" * 64,
            },
        ],
        "metadata_files": write_repro_metadata_sidecars(repro.parent),
    }) + "\n")
    data.write_text(
        "# Data Availability\n\n"
        "DOI: 10.5281/zenodo.1234567\n"
    )

    res = run_script([
        "scripts/stage11_manuscript.py",
        "--run-dir", str(run_dir),
        "--out-dir", str(out_dir),
        "--captions", str(captions),
        "--repro-manifest", str(repro),
        "--data-availability", str(data),
    ])

    assert res.returncode != 0
    assert "captions source bytes value does not match" in res.stderr
    assert not (out_dir / "03_results.md").exists()


def test_manuscript_rejects_caption_source_forged_prefix(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "manuscript"
    sources = tmp_path / "publication"
    captions = sources / "figures" / "captions.json"
    repro = sources / "reproducibility" / "artifact_manifest.json"
    data = sources / "data_availability.md"
    out_dir.mkdir()
    (out_dir / "03_results.md").write_text("stale\n")
    captions.parent.mkdir(parents=True)
    repro.parent.mkdir(parents=True)
    captions.write_text(json.dumps({
        "fig01_workflow.svg": {
            "caption": "Source-backed workflow evidence.",
            "status": "source_backed",
            "sources": [{
                "path": "shadow/run/09_report/index.html",
                "metric": "bytes",
                "value": 42,
                "sha256": "a" * 64,
            }],
        },
    }) + "\n")
    repro.write_text(json.dumps({
        "run_dir": str(run_dir),
        "n_artifacts": 2,
        "artifacts": [
            {
                "relative_path": "09_report/index.html",
                "bytes": 42,
                "sha256": "a" * 64,
            },
            {
                "relative_path": "EVAL:iteration_manifest.json",
                "bytes": 10,
                "sha256": "b" * 64,
            },
        ],
        "metadata_files": write_repro_metadata_sidecars(repro.parent),
    }) + "\n")
    data.write_text(
        "# Data Availability\n\n"
        "DOI: 10.5281/zenodo.1234567\n"
    )

    res = run_script([
        "scripts/stage11_manuscript.py",
        "--run-dir", str(run_dir),
        "--out-dir", str(out_dir),
        "--captions", str(captions),
        "--repro-manifest", str(repro),
        "--data-availability", str(data),
    ])

    assert res.returncode != 0
    assert "captions source path does not match" in res.stderr
    assert not (out_dir / "03_results.md").exists()


def test_manuscript_rejects_repro_manifest_run_dir_mismatch(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "manuscript"
    sources = tmp_path / "publication"
    captions = sources / "figures" / "captions.json"
    repro = sources / "reproducibility" / "artifact_manifest.json"
    data = sources / "data_availability.md"
    out_dir.mkdir()
    (out_dir / "03_results.md").write_text("stale\n")
    captions.parent.mkdir(parents=True)
    repro.parent.mkdir(parents=True)
    captions.write_text(json.dumps({
        "fig01_workflow.svg": {
            "caption": "Source-backed workflow evidence.",
            "status": "source_backed",
            "sources": [{
                "path": "run/09_report/index.html",
                "metric": "bytes",
                "value": 42,
                "sha256": "a" * 64,
            }],
        },
    }) + "\n")
    repro.write_text(json.dumps({
        "run_dir": str(tmp_path / "other_run"),
        "n_artifacts": 2,
        "artifacts": [
            {
                "relative_path": "09_report/index.html",
                "bytes": 42,
                "sha256": "a" * 64,
            },
            {
                "relative_path": "EVAL:iteration_manifest.json",
                "bytes": 10,
                "sha256": "b" * 64,
            },
        ],
    }) + "\n")
    data.write_text(
        "# Data Availability\n\n"
        "DOI: 10.5281/zenodo.1234567\n"
    )

    res = run_script([
        "scripts/stage11_manuscript.py",
        "--run-dir", str(run_dir),
        "--out-dir", str(out_dir),
        "--captions", str(captions),
        "--repro-manifest", str(repro),
        "--data-availability", str(data),
    ])

    assert res.returncode != 0
    assert "reproducibility manifest run_dir must match" in res.stderr
    assert not (out_dir / "03_results.md").exists()


def test_manuscript_rejects_repro_manifest_artifact_path_mismatch(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "manuscript"
    sources = tmp_path / "publication"
    captions = sources / "figures" / "captions.json"
    repro = sources / "reproducibility" / "artifact_manifest.json"
    data = sources / "data_availability.md"
    out_dir.mkdir()
    (out_dir / "03_results.md").write_text("stale\n")
    captions.parent.mkdir(parents=True)
    repro.parent.mkdir(parents=True)
    captions.write_text(json.dumps({
        "fig01_workflow.svg": {
            "caption": "Source-backed workflow evidence.",
            "status": "source_backed",
            "sources": [{
                "path": "run/09_report/index.html",
                "metric": "bytes",
                "value": 42,
                "sha256": "a" * 64,
            }],
        },
    }) + "\n")
    repro.write_text(json.dumps({
        "run_dir": str(run_dir),
        "n_artifacts": 2,
        "artifacts": [
            {
                "path": str(run_dir / "forged_report.html"),
                "relative_path": "09_report/index.html",
                "bytes": 42,
                "sha256": "a" * 64,
            },
            {
                "relative_path": "EVAL:iteration_manifest.json",
                "bytes": 10,
                "sha256": "b" * 64,
            },
        ],
    }) + "\n")
    data.write_text(
        "# Data Availability\n\n"
        "DOI: 10.5281/zenodo.1234567\n"
    )

    res = run_script([
        "scripts/stage11_manuscript.py",
        "--run-dir", str(run_dir),
        "--out-dir", str(out_dir),
        "--captions", str(captions),
        "--repro-manifest", str(repro),
        "--data-availability", str(data),
    ])

    assert res.returncode != 0
    assert "artifact path must match relative_path" in res.stderr
    assert not (out_dir / "03_results.md").exists()


def test_manuscript_rejects_repro_manifest_artifact_digest_mismatch(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "manuscript"
    sources = tmp_path / "publication"
    captions = sources / "figures" / "captions.json"
    repro = sources / "reproducibility" / "artifact_manifest.json"
    data = sources / "data_availability.md"
    report = run_dir / "09_report" / "index.html"
    eval_manifest = tmp_path / "eval" / "iteration_manifest.json"
    out_dir.mkdir()
    (out_dir / "03_results.md").write_text("stale\n")
    captions.parent.mkdir(parents=True)
    repro.parent.mkdir(parents=True)
    report.parent.mkdir(parents=True)
    eval_manifest.parent.mkdir(parents=True)
    report.write_text("<html>source-backed report</html>\n")
    eval_manifest.write_text('{"claim_ready": true}\n')
    report_sha = hashlib.sha256(report.read_bytes()).hexdigest()
    captions.write_text(json.dumps({
        "fig01_workflow.svg": {
            "caption": "Source-backed workflow evidence.",
            "status": "source_backed",
            "sources": [{
                "path": "run/09_report/index.html",
                "metric": "bytes",
                "value": report.stat().st_size,
                "sha256": report_sha,
            }],
        },
    }) + "\n")
    repro.write_text(json.dumps({
        "run_dir": str(run_dir),
        "n_artifacts": 2,
        "artifacts": [
            {
                "path": str(report),
                "relative_path": "09_report/index.html",
                "bytes": report.stat().st_size,
                "sha256": report_sha,
            },
            {
                "path": str(eval_manifest),
                "relative_path": "EVAL:iteration_manifest.json",
                "bytes": eval_manifest.stat().st_size + 1,
                "sha256": hashlib.sha256(eval_manifest.read_bytes()).hexdigest(),
            },
        ],
    }) + "\n")
    data.write_text(
        "# Data Availability\n\n"
        "DOI: 10.5281/zenodo.1234567\n"
    )

    res = run_script([
        "scripts/stage11_manuscript.py",
        "--run-dir", str(run_dir),
        "--out-dir", str(out_dir),
        "--captions", str(captions),
        "--repro-manifest", str(repro),
        "--data-availability", str(data),
    ])

    assert res.returncode != 0
    assert "artifact bytes/sha256 must match existing file" in res.stderr
    assert not (out_dir / "03_results.md").exists()


def test_manuscript_rejects_non_claim_ready_eval_artifact(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "manuscript"
    sources = tmp_path / "publication"
    captions = sources / "figures" / "captions.json"
    repro = sources / "reproducibility" / "artifact_manifest.json"
    data = sources / "data_availability.md"
    report = run_dir / "09_report" / "index.html"
    eval_manifest = tmp_path / "eval" / "iteration_manifest.json"
    out_dir.mkdir()
    (out_dir / "03_results.md").write_text("stale\n")
    captions.parent.mkdir(parents=True)
    repro.parent.mkdir(parents=True)
    report.parent.mkdir(parents=True)
    eval_manifest.parent.mkdir(parents=True)
    report.write_text("<html>source-backed report</html>\n")
    eval_manifest.write_text('{"claim_ready": true}\n')
    report_sha = hashlib.sha256(report.read_bytes()).hexdigest()
    eval_sha = hashlib.sha256(eval_manifest.read_bytes()).hexdigest()
    captions.write_text(json.dumps({
        "fig01_workflow.svg": {
            "caption": "Source-backed workflow evidence.",
            "status": "source_backed",
            "sources": [{
                "path": "run/09_report/index.html",
                "metric": "bytes",
                "value": report.stat().st_size,
                "sha256": report_sha,
            }],
        },
    }) + "\n")
    repro.write_text(json.dumps({
        "run_dir": str(run_dir),
        "n_artifacts": 2,
        "artifacts": [
            {
                "path": str(report),
                "relative_path": "09_report/index.html",
                "bytes": report.stat().st_size,
                "sha256": report_sha,
            },
            {
                "path": str(eval_manifest),
                "relative_path": "EVAL:iteration_manifest.json",
                "bytes": eval_manifest.stat().st_size,
                "sha256": eval_sha,
            },
        ],
        "metadata_files": write_repro_metadata_sidecars(repro.parent),
    }) + "\n")
    data.write_text(
        "# Data Availability\n\n"
        "DOI: 10.5281/zenodo.1234567\n"
    )

    res = run_script([
        "scripts/stage11_manuscript.py",
        "--run-dir", str(run_dir),
        "--out-dir", str(out_dir),
        "--captions", str(captions),
        "--repro-manifest", str(repro),
        "--data-availability", str(data),
    ])

    assert res.returncode != 0
    assert "eval artifact is not claim-ready" in res.stderr
    assert not (out_dir / "03_results.md").exists()


def test_manuscript_rejects_repro_manifest_metadata_digest_mismatch(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "manuscript"
    sources = tmp_path / "publication"
    captions = sources / "figures" / "captions.json"
    repro_dir = sources / "reproducibility"
    repro = repro_dir / "artifact_manifest.json"
    data = sources / "data_availability.md"
    report = run_dir / "09_report" / "index.html"
    eval_manifest = tmp_path / "eval" / "iteration_manifest.json"
    metadata_payloads = {
        "config_hash.txt": "a" * 64 + "\n",
        "git_commit.txt": current_git_commit() + "\n",
        "random_seeds.json": '{"etkdg": 49242, "boltz": 0, "reinvent": 0}\n',
        "runtime_log.txt": "runtime\n",
        "tool_versions.lock": "pytest==0\n",
    }
    out_dir.mkdir()
    (out_dir / "03_results.md").write_text("stale\n")
    captions.parent.mkdir(parents=True)
    repro_dir.mkdir(parents=True)
    report.parent.mkdir(parents=True)
    report.write_text("<html>source-backed report</html>\n")
    write_claim_ready_eval_manifest(eval_manifest, source_run_dir=run_dir)
    for rel, text in metadata_payloads.items():
        (repro_dir / rel).write_text(text)
    report_sha = hashlib.sha256(report.read_bytes()).hexdigest()
    captions.write_text(json.dumps({
        "fig01_workflow.svg": {
            "caption": "Source-backed workflow evidence.",
            "status": "source_backed",
            "sources": [{
                "path": "run/09_report/index.html",
                "metric": "bytes",
                "value": report.stat().st_size,
                "sha256": report_sha,
            }],
        },
    }) + "\n")
    repro.write_text(json.dumps({
        "run_dir": str(run_dir),
        "n_artifacts": 2,
        "artifacts": [
            {
                "path": str(report),
                "relative_path": "09_report/index.html",
                "bytes": report.stat().st_size,
                "sha256": report_sha,
            },
            {
                "path": str(eval_manifest),
                "relative_path": "EVAL:iteration_manifest.json",
                "bytes": eval_manifest.stat().st_size,
                "sha256": hashlib.sha256(eval_manifest.read_bytes()).hexdigest(),
            },
        ],
        "metadata_files": [
            {
                "relative_path": rel,
                "bytes": (repro_dir / rel).stat().st_size,
                "sha256": (
                    "b" * 64
                    if rel == "config_hash.txt"
                    else hashlib.sha256((repro_dir / rel).read_bytes()).hexdigest()
                ),
            }
            for rel in sorted(metadata_payloads)
        ],
    }) + "\n")
    data.write_text(
        "# Data Availability\n\n"
        "DOI: 10.5281/zenodo.1234567\n"
    )

    res = run_script([
        "scripts/stage11_manuscript.py",
        "--run-dir", str(run_dir),
        "--out-dir", str(out_dir),
        "--captions", str(captions),
        "--repro-manifest", str(repro),
        "--data-availability", str(data),
    ])

    assert res.returncode != 0
    assert "metadata file bytes/sha256 must match existing file" in res.stderr
    assert not (out_dir / "03_results.md").exists()


def test_manuscript_rejects_repro_manifest_missing_metadata_sidecar(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "manuscript"
    sources = tmp_path / "publication"
    captions = sources / "figures" / "captions.json"
    repro_dir = sources / "reproducibility"
    repro = repro_dir / "artifact_manifest.json"
    data = sources / "data_availability.md"
    report = run_dir / "09_report" / "index.html"
    eval_manifest = tmp_path / "eval" / "iteration_manifest.json"
    metadata = repro_dir / "config_hash.txt"
    out_dir.mkdir()
    (out_dir / "03_results.md").write_text("stale\n")
    captions.parent.mkdir(parents=True)
    repro_dir.mkdir(parents=True)
    report.parent.mkdir(parents=True)
    report.write_text("<html>source-backed report</html>\n")
    write_claim_ready_eval_manifest(eval_manifest, source_run_dir=run_dir)
    metadata.write_text("a" * 64 + "\n")
    report_sha = hashlib.sha256(report.read_bytes()).hexdigest()
    captions.write_text(json.dumps({
        "fig01_workflow.svg": {
            "caption": "Source-backed workflow evidence.",
            "status": "source_backed",
            "sources": [{
                "path": "run/09_report/index.html",
                "metric": "bytes",
                "value": report.stat().st_size,
                "sha256": report_sha,
            }],
        },
    }) + "\n")
    repro.write_text(json.dumps({
        "run_dir": str(run_dir),
        "n_artifacts": 2,
        "artifacts": [
            {
                "path": str(report),
                "relative_path": "09_report/index.html",
                "bytes": report.stat().st_size,
                "sha256": report_sha,
            },
            {
                "path": str(eval_manifest),
                "relative_path": "EVAL:iteration_manifest.json",
                "bytes": eval_manifest.stat().st_size,
                "sha256": hashlib.sha256(eval_manifest.read_bytes()).hexdigest(),
            },
        ],
        "metadata_files": [{
            "relative_path": "config_hash.txt",
            "bytes": metadata.stat().st_size,
            "sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
        }],
    }) + "\n")
    data.write_text(
        "# Data Availability\n\n"
        "DOI: 10.5281/zenodo.1234567\n"
    )

    res = run_script([
        "scripts/stage11_manuscript.py",
        "--run-dir", str(run_dir),
        "--out-dir", str(out_dir),
        "--captions", str(captions),
        "--repro-manifest", str(repro),
        "--data-availability", str(data),
    ])

    assert res.returncode != 0
    assert "metadata_files must match the required reproducibility sidecars" in (
        res.stderr
    )
    assert "runtime_log.txt" in res.stderr
    assert not (out_dir / "03_results.md").exists()


def test_manuscript_rejects_repro_manifest_missing_metadata_files(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "manuscript"
    sources = tmp_path / "publication"
    captions = sources / "figures" / "captions.json"
    repro = sources / "reproducibility" / "artifact_manifest.json"
    data = sources / "data_availability.md"
    out_dir.mkdir()
    (out_dir / "03_results.md").write_text("stale\n")
    captions.parent.mkdir(parents=True)
    repro.parent.mkdir(parents=True)
    captions.write_text(json.dumps({
        "fig01_workflow.svg": {
            "caption": "Source-backed workflow evidence.",
            "status": "source_backed",
            "sources": [{
                "path": "run/09_report/index.html",
                "metric": "bytes",
                "value": 42,
                "sha256": "a" * 64,
            }],
        },
    }) + "\n")
    repro.write_text(json.dumps({
        "run_dir": str(run_dir),
        "n_artifacts": 2,
        "artifacts": [
            {
                "relative_path": "09_report/index.html",
                "bytes": 42,
                "sha256": "a" * 64,
            },
            {
                "relative_path": "EVAL:iteration_manifest.json",
                "bytes": 10,
                "sha256": "b" * 64,
            },
        ],
    }) + "\n")
    data.write_text(
        "# Data Availability\n\n"
        "DOI: 10.5281/zenodo.1234567\n"
    )

    res = run_script([
        "scripts/stage11_manuscript.py",
        "--run-dir", str(run_dir),
        "--out-dir", str(out_dir),
        "--captions", str(captions),
        "--repro-manifest", str(repro),
        "--data-availability", str(data),
    ])

    assert res.returncode != 0
    assert "metadata_files is required" in res.stderr
    assert not (out_dir / "03_results.md").exists()


def test_manuscript_rejects_repro_manifest_duplicate_relative_path(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "manuscript"
    sources = tmp_path / "publication"
    captions = sources / "figures" / "captions.json"
    repro = sources / "reproducibility" / "artifact_manifest.json"
    data = sources / "data_availability.md"
    out_dir.mkdir()
    (out_dir / "03_results.md").write_text("stale\n")
    captions.parent.mkdir(parents=True)
    repro.parent.mkdir(parents=True)
    captions.write_text(json.dumps({
        "fig01_workflow.svg": {
            "caption": "Source-backed workflow evidence.",
            "status": "source_backed",
            "sources": [{
                "path": "run/09_report/index.html",
                "metric": "bytes",
                "value": 42,
                "sha256": "a" * 64,
            }],
        },
    }) + "\n")
    repro.write_text(json.dumps({
        "run_dir": str(run_dir),
        "n_artifacts": 3,
        "artifacts": [
            {
                "relative_path": "09_report/index.html",
                "bytes": 42,
                "sha256": "a" * 64,
            },
            {
                "relative_path": "09_report/index.html",
                "bytes": 43,
                "sha256": "c" * 64,
            },
            {
                "relative_path": "EVAL:iteration_manifest.json",
                "bytes": 10,
                "sha256": "b" * 64,
            },
        ],
    }) + "\n")
    data.write_text(
        "# Data Availability\n\n"
        "DOI: 10.5281/zenodo.1234567\n"
    )

    res = run_script([
        "scripts/stage11_manuscript.py",
        "--run-dir", str(run_dir),
        "--out-dir", str(out_dir),
        "--captions", str(captions),
        "--repro-manifest", str(repro),
        "--data-availability", str(data),
    ])

    assert res.returncode != 0
    assert "duplicate relative_path" in res.stderr
    assert not (out_dir / "03_results.md").exists()


def test_manuscript_rejects_caption_source_invalid_metric(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "manuscript"
    sources = tmp_path / "publication"
    captions = sources / "figures" / "captions.json"
    repro = sources / "reproducibility" / "artifact_manifest.json"
    data = sources / "data_availability.md"
    out_dir.mkdir()
    (out_dir / "03_results.md").write_text("stale\n")
    captions.parent.mkdir(parents=True)
    repro.parent.mkdir(parents=True)
    captions.write_text(json.dumps({
        "fig01_workflow.svg": {
            "caption": "Source-backed workflow evidence.",
            "status": "source_backed",
            "sources": [{
                "path": "run/09_report/index.html",
                "metric": "unverified",
                "value": 42,
                "sha256": "a" * 64,
            }],
        },
    }) + "\n")
    repro.write_text(json.dumps({
        "run_dir": str(run_dir),
        "n_artifacts": 2,
        "artifacts": [
            {
                "relative_path": "09_report/index.html",
                "bytes": 42,
                "sha256": "a" * 64,
            },
            {
                "relative_path": "EVAL:iteration_manifest.json",
                "bytes": 10,
                "sha256": "b" * 64,
            },
        ],
    }) + "\n")
    data.write_text(
        "# Data Availability\n\n"
        "DOI: 10.5281/zenodo.1234567\n"
    )

    res = run_script([
        "scripts/stage11_manuscript.py",
        "--run-dir", str(run_dir),
        "--out-dir", str(out_dir),
        "--captions", str(captions),
        "--repro-manifest", str(repro),
        "--data-availability", str(data),
    ])

    assert res.returncode != 0
    assert "captions source metric is not recognized" in res.stderr
    assert not (out_dir / "03_results.md").exists()


def test_manuscript_rejects_caption_source_nonpositive_value(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "manuscript"
    sources = tmp_path / "publication"
    captions = sources / "figures" / "captions.json"
    repro = sources / "reproducibility" / "artifact_manifest.json"
    data = sources / "data_availability.md"
    out_dir.mkdir()
    (out_dir / "03_results.md").write_text("stale\n")
    captions.parent.mkdir(parents=True)
    repro.parent.mkdir(parents=True)
    captions.write_text(json.dumps({
        "fig01_workflow.svg": {
            "caption": "Source-backed workflow evidence.",
            "status": "source_backed",
            "sources": [{
                "path": "run/09_report/index.html",
                "metric": "bytes",
                "value": 0,
                "sha256": "a" * 64,
            }],
        },
    }) + "\n")
    repro.write_text(json.dumps({
        "run_dir": str(run_dir),
        "n_artifacts": 2,
        "artifacts": [
            {
                "relative_path": "09_report/index.html",
                "bytes": 42,
                "sha256": "a" * 64,
            },
            {
                "relative_path": "EVAL:iteration_manifest.json",
                "bytes": 10,
                "sha256": "b" * 64,
            },
        ],
    }) + "\n")
    data.write_text(
        "# Data Availability\n\n"
        "DOI: 10.5281/zenodo.1234567\n"
    )

    res = run_script([
        "scripts/stage11_manuscript.py",
        "--run-dir", str(run_dir),
        "--out-dir", str(out_dir),
        "--captions", str(captions),
        "--repro-manifest", str(repro),
        "--data-availability", str(data),
    ])

    assert res.returncode != 0
    assert "captions source value must be a positive integer" in res.stderr
    assert not (out_dir / "03_results.md").exists()


def test_manuscript_rejects_non_source_backed_captions(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "manuscript"
    sources = tmp_path / "publication"
    captions = sources / "figures" / "captions.json"
    repro = sources / "reproducibility" / "artifact_manifest.json"
    data = sources / "data_availability.md"
    out_dir.mkdir()
    (out_dir / "03_results.md").write_text("stale\n")
    captions.parent.mkdir(parents=True)
    repro.parent.mkdir(parents=True)
    captions.write_text(json.dumps({
        "fig01_workflow.svg": {
            "caption": "Draft workflow evidence.",
            "status": "draft",
            "sources": [{
                "path": "run/09_report/index.html",
                "metric": "bytes",
                "value": 42,
                "sha256": "a" * 64,
            }],
        },
    }) + "\n")
    repro.write_text(json.dumps({
        "run_dir": str(run_dir),
        "n_artifacts": 1,
        "artifacts": [{
            "relative_path": "EVAL:iteration_manifest.json",
            "bytes": 10,
            "sha256": "b" * 64,
        }],
    }) + "\n")
    data.write_text(
        "# Data Availability\n\n"
        "DOI: 10.5281/zenodo.1234567\n"
    )

    res = run_script([
        "scripts/stage11_manuscript.py",
        "--run-dir", str(run_dir),
        "--out-dir", str(out_dir),
        "--captions", str(captions),
        "--repro-manifest", str(repro),
        "--data-availability", str(data),
    ])

    assert res.returncode != 0
    assert "captions must be source-backed" in res.stderr
    assert not (out_dir / "03_results.md").exists()


def test_manuscript_allow_placeholder_mode_writes_draft(tmp_path: Path) -> None:
    out_dir = tmp_path / "manuscript"

    res = run_script([
        "scripts/stage11_manuscript.py",
        "--run-dir", str(tmp_path / "run"),
        "--out-dir", str(out_dir),
        "--allow-placeholders",
    ])

    assert res.returncode == 0, res.stderr
    assert len(list(out_dir.iterdir())) == 6
    assert (out_dir / "00_abstract.md").exists()
    assert (out_dir / "02_methods.md").exists()
    assert (out_dir / "05_references.bib").exists()


def test_data_availability_requires_real_doi_by_default(tmp_path: Path) -> None:
    out_md = tmp_path / "publication" / "data_availability.md"
    out_md.parent.mkdir()
    out_md.write_text("stale\n")

    res = run_script([
        "scripts/stage11_data_availability.py",
        "--out-md", str(out_md),
    ])

    assert res.returncode != 0
    assert "requires a real skin-efficacy KG DOI" in res.stderr
    assert not out_md.exists()


def test_data_availability_rejects_malformed_doi(tmp_path: Path) -> None:
    out_md = tmp_path / "publication" / "data_availability.md"
    out_md.parent.mkdir()
    out_md.write_text("stale\n")

    res = run_script([
        "scripts/stage11_data_availability.py",
        "--out-md", str(out_md),
        "--skin-efficacy-kg-doi", "10.not-a-doi",
    ])

    assert res.returncode != 0
    assert "requires a real skin-efficacy KG DOI" in res.stderr
    assert not out_md.exists()


def test_data_availability_rejects_invalid_pipeline_source_url(
    tmp_path: Path,
) -> None:
    out_md = tmp_path / "publication" / "data_availability.md"
    out_md.parent.mkdir()
    out_md.write_text("stale\n")

    res = run_script([
        "scripts/stage11_data_availability.py",
        "--out-md", str(out_md),
        "--skin-efficacy-kg-doi", "10.5281/zenodo.1234567",
        "--pipeline-source-url", "not-a-url",
    ])

    assert res.returncode != 0
    assert "requires a public GitHub source URL" in res.stderr
    assert not out_md.exists()


def test_data_availability_rejects_non_github_pipeline_source_url(
    tmp_path: Path,
) -> None:
    out_md = tmp_path / "publication" / "data_availability.md"
    out_md.parent.mkdir()
    out_md.write_text("stale\n")

    res = run_script([
        "scripts/stage11_data_availability.py",
        "--out-md", str(out_md),
        "--skin-efficacy-kg-doi", "10.5281/zenodo.1234567",
        "--pipeline-source-url", "http://localhost/SkinScout",
    ])

    assert res.returncode != 0
    assert "requires a public GitHub source URL" in res.stderr
    assert not out_md.exists()


def test_data_availability_overwrites_stale_statement_with_real_doi(tmp_path: Path) -> None:
    out_md = tmp_path / "publication" / "data_availability.md"
    out_md.parent.mkdir()
    out_md.write_text("stale\n")

    res = run_script([
        "scripts/stage11_data_availability.py",
        "--out-md", str(out_md),
        "--skin-efficacy-kg-doi", "10.5281/zenodo.1234567",
    ])

    assert res.returncode == 0, res.stderr
    text = out_md.read_text()
    assert text.startswith("# Data Availability")
    assert "10.5281/zenodo.1234567" in text
    assert "https://github.com/kangk1204/SkinScout_public" in text
    assert "{user}" not in text
    assert "cosmetic-discovery-pipeline" not in text
    assert "XXXX" not in text
    assert "pending" not in text
    assert "stale" not in text
    assert not out_md.with_suffix(".md.tmp").exists()


def test_data_availability_draft_mode_allows_pending_doi(tmp_path: Path) -> None:
    out_md = tmp_path / "publication" / "data_availability.md"

    res = run_script([
        "scripts/stage11_data_availability.py",
        "--out-md", str(out_md),
        "--draft-doi-ok",
    ])

    assert res.returncode == 0, res.stderr
    assert "DOI: pending" in out_md.read_text()


def test_repro_pack_fails_without_run_artifacts(tmp_path: Path) -> None:
    out_dir = tmp_path / "repro"
    out_dir.mkdir()
    stale_files = (
        "tool_versions.lock",
        "config_hash.txt",
        "git_commit.txt",
        "artifact_manifest.json",
        "random_seeds.json",
        "runtime_log.txt",
    )
    for name in stale_files:
        (out_dir / name).write_text("stale\n")

    res = run_script([
        "scripts/stage11_repro_pack.py",
        "--run-dir", str(tmp_path / "missing_run"),
        "--out-dir", str(out_dir),
    ])

    assert res.returncode != 0
    assert "Run directory is required for reproducibility pack" in res.stderr
    for name in stale_files:
        assert not (out_dir / name).exists()


def test_repro_pack_rejects_failed_tool_version_capture(monkeypatch) -> None:
    module = load_repro_pack_module()

    def fail_check_output(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ["pip", "freeze"])

    monkeypatch.setattr(module.subprocess, "check_output", fail_check_output)

    with pytest.raises(SystemExit, match="Tool version capture failed"):
        module._pip_freeze()


def test_repro_pack_rejects_empty_tool_version_capture(monkeypatch) -> None:
    module = load_repro_pack_module()
    monkeypatch.setattr(module.subprocess, "check_output", lambda *args, **kwargs: "\n")

    with pytest.raises(SystemExit, match="produced no package records"):
        module._pip_freeze()


def test_repro_pack_rejects_failed_git_commit_capture(monkeypatch) -> None:
    module = load_repro_pack_module()

    def fail_check_output(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ["git", "rev-parse", "HEAD"])

    monkeypatch.setattr(module.subprocess, "check_output", fail_check_output)

    with pytest.raises(SystemExit, match="Git commit capture failed"):
        module._git_commit()


def test_repro_pack_rejects_invalid_git_commit_capture(monkeypatch) -> None:
    module = load_repro_pack_module()
    monkeypatch.setattr(
        module.subprocess,
        "check_output",
        lambda *args, **kwargs: "no-git-history\n",
    )

    with pytest.raises(SystemExit, match="invalid commit SHA"):
        module._git_commit()


def test_repro_pack_rejects_missing_runtime_metadata(monkeypatch) -> None:
    module = load_repro_pack_module()
    monkeypatch.setattr(module.platform, "platform", lambda: "")

    with pytest.raises(SystemExit, match="missing required fields: platform"):
        module._runtime_log()


def test_repro_pack_records_required_runtime_metadata() -> None:
    module = load_repro_pack_module()

    runtime_log = module._runtime_log()

    assert "platform: " in runtime_log
    assert "python: " in runtime_log
    assert "python_executable: " in runtime_log
    assert "working_directory: " in runtime_log
    assert "cpu: " in runtime_log
    assert "hostname: " in runtime_log


def test_repro_pack_rejects_empty_source_artifact(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out_dir = run_dir / "publication" / "reproducibility"
    source = run_dir / "09_report" / "index.html"
    source.parent.mkdir(parents=True)
    source.write_text("")
    out_dir.mkdir(parents=True)
    stale_files = (
        "tool_versions.lock",
        "config_hash.txt",
        "git_commit.txt",
        "artifact_manifest.json",
        "random_seeds.json",
        "runtime_log.txt",
    )
    for name in stale_files:
        (out_dir / name).write_text("stale\n")

    res = run_script([
        "scripts/stage11_repro_pack.py",
        "--run-dir", str(run_dir),
        "--out-dir", str(out_dir),
    ])

    assert res.returncode != 0
    assert "source artifact is empty" in res.stderr
    assert "09_report/index.html" in res.stderr
    for name in stale_files:
        assert not (out_dir / name).exists()


def test_repro_pack_requires_final_report_artifact(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out_dir = run_dir / "publication" / "reproducibility"
    source = run_dir / "03_targets" / "ranked_targets_v3.csv"
    source.parent.mkdir(parents=True)
    source.write_text("target_id,score\nP1,1.0\n")
    out_dir.mkdir(parents=True)
    stale_files = (
        "tool_versions.lock",
        "config_hash.txt",
        "git_commit.txt",
        "artifact_manifest.json",
        "random_seeds.json",
        "runtime_log.txt",
    )
    for name in stale_files:
        (out_dir / name).write_text("stale\n")

    res = run_script([
        "scripts/stage11_repro_pack.py",
        "--run-dir", str(run_dir),
        "--out-dir", str(out_dir),
    ])

    assert res.returncode != 0
    assert "required source artifact is missing" in res.stderr
    assert "09_report/index.html" in res.stderr
    for name in stale_files:
        assert not (out_dir / name).exists()


def test_repro_pack_rejects_placeholder_html_report(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out_dir = run_dir / "publication" / "reproducibility"
    report = run_dir / "09_report" / "index.html"
    report.parent.mkdir(parents=True)
    report.write_text("<html>placeholder report</html>\n")
    consensus = run_dir / "03_targets" / "mode_comprehensive" / "top50_4way_consensus.csv"
    consensus.parent.mkdir(parents=True)
    consensus.write_text("target_id,score\nP1,1.0\n")
    ranked = run_dir / "03_targets" / "ranked_targets_v3_with_efficacy.csv"
    ranked.write_text(
        "target_id,final_score,source_count,sources,efficacy_top1\n"
        "P2,1.0,3,autodock;gnina;rtmscore,hydration\n"
    )
    out_dir.mkdir(parents=True)
    (out_dir / "artifact_manifest.json").write_text("stale\n")

    res = run_script([
        "scripts/stage11_repro_pack.py",
        "--run-dir", str(run_dir),
        "--out-dir", str(out_dir),
    ])

    assert res.returncode != 0
    assert "contains scaffold marker 'placeholder'" in res.stderr
    assert "09_report/index.html" in res.stderr
    assert not (out_dir / "artifact_manifest.json").exists()


def test_repro_pack_requires_final_target_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out_dir = run_dir / "publication" / "reproducibility"
    consensus_rel = Path("03_targets/mode_comprehensive/top50_4way_consensus.csv")
    ranked_rel = Path("03_targets/ranked_targets_v3_with_efficacy.csv")
    report = run_dir / "09_report" / "index.html"
    report.parent.mkdir(parents=True)
    report.write_text("<html>source-backed report</html>\n")
    consensus = run_dir / consensus_rel
    consensus.parent.mkdir(parents=True)
    consensus.write_text("target_id,score\nP1,1.0\n")
    out_dir.mkdir(parents=True)
    stale_files = (
        "tool_versions.lock",
        "config_hash.txt",
        "git_commit.txt",
        "artifact_manifest.json",
        "random_seeds.json",
        "runtime_log.txt",
    )
    for name in stale_files:
        (out_dir / name).write_text("stale\n")

    res = run_script([
        "scripts/stage11_repro_pack.py",
        "--run-dir", str(run_dir),
        "--out-dir", str(out_dir),
    ])

    assert res.returncode != 0
    assert "required source artifact is missing" in res.stderr
    assert str(ranked_rel) in res.stderr
    for name in stale_files:
        assert not (out_dir / name).exists()


def test_repro_pack_rejects_ranked_targets_missing_publication_columns(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    out_dir = run_dir / "publication" / "reproducibility"
    report = run_dir / "09_report" / "index.html"
    report.parent.mkdir(parents=True)
    report.write_text("<html>source-backed report</html>\n")
    consensus = run_dir / "03_targets" / "mode_comprehensive" / "top50_4way_consensus.csv"
    consensus.parent.mkdir(parents=True)
    consensus.write_text("target_id,score\nP1,1.0\n")
    ranked = run_dir / "03_targets" / "ranked_targets_v3_with_efficacy.csv"
    ranked.write_text("target_id,final_score\nP2,1.0\n")
    out_dir.mkdir(parents=True)
    (out_dir / "artifact_manifest.json").write_text("stale\n")

    res = run_script([
        "scripts/stage11_repro_pack.py",
        "--run-dir", str(run_dir),
        "--out-dir", str(out_dir),
    ])

    assert res.returncode != 0
    assert "missing required publication column" in res.stderr
    assert "source_count" in res.stderr
    assert "sources" in res.stderr
    assert "efficacy_top1" in res.stderr
    assert "ranked_targets_v3_with_efficacy.csv" in res.stderr
    assert not (out_dir / "artifact_manifest.json").exists()


def test_repro_pack_rejects_header_only_required_target_artifact(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    out_dir = run_dir / "publication" / "reproducibility"
    report = run_dir / "09_report" / "index.html"
    report.parent.mkdir(parents=True)
    report.write_text("<html>source-backed report</html>\n")
    consensus = run_dir / "03_targets" / "mode_comprehensive" / "top50_4way_consensus.csv"
    consensus.parent.mkdir(parents=True)
    consensus.write_text("target_id,score\n")
    ranked = run_dir / "03_targets" / "ranked_targets_v3_with_efficacy.csv"
    ranked.write_text(
        "target_id,final_score,source_count,sources,efficacy_top1\n"
        "P1,1.0,3,autodock;gnina;rtmscore,hydration\n"
    )
    out_dir.mkdir(parents=True)
    stale_files = (
        "tool_versions.lock",
        "config_hash.txt",
        "git_commit.txt",
        "artifact_manifest.json",
        "random_seeds.json",
        "runtime_log.txt",
    )
    for name in stale_files:
        (out_dir / name).write_text("stale\n")

    res = run_script([
        "scripts/stage11_repro_pack.py",
        "--run-dir", str(run_dir),
        "--out-dir", str(out_dir),
    ])

    assert res.returncode != 0
    assert "source artifact contains no data rows" in res.stderr
    assert "top50_4way_consensus.csv" in res.stderr
    for name in stale_files:
        assert not (out_dir / name).exists()


def test_repro_pack_rejects_blank_required_target_identifier(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    out_dir = run_dir / "publication" / "reproducibility"
    report = run_dir / "09_report" / "index.html"
    report.parent.mkdir(parents=True)
    report.write_text("<html>source-backed report</html>\n")
    consensus = run_dir / "03_targets" / "mode_comprehensive" / "top50_4way_consensus.csv"
    consensus.parent.mkdir(parents=True)
    consensus.write_text("target_id,score\nP1,1.0\n")
    ranked = run_dir / "03_targets" / "ranked_targets_v3_with_efficacy.csv"
    ranked.write_text(
        "target_id,final_score,source_count,sources,efficacy_top1\n"
        " ,1.0,3,autodock;gnina;rtmscore,hydration\n"
    )
    out_dir.mkdir(parents=True)
    (out_dir / "artifact_manifest.json").write_text("stale\n")

    res = run_script([
        "scripts/stage11_repro_pack.py",
        "--run-dir", str(run_dir),
        "--out-dir", str(out_dir),
    ])

    assert res.returncode != 0
    assert "column 'target_id' contains blank values" in res.stderr
    assert "ranked_targets_v3_with_efficacy.csv" in res.stderr
    assert not (out_dir / "artifact_manifest.json").exists()


def test_repro_pack_rejects_duplicate_required_target_identifier(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    out_dir = run_dir / "publication" / "reproducibility"
    report = run_dir / "09_report" / "index.html"
    report.parent.mkdir(parents=True)
    report.write_text("<html>source-backed report</html>\n")
    consensus = run_dir / "03_targets" / "mode_comprehensive" / "top50_4way_consensus.csv"
    consensus.parent.mkdir(parents=True)
    consensus.write_text("target_id,score\nP1,1.0\nP1,0.9\n")
    ranked = run_dir / "03_targets" / "ranked_targets_v3_with_efficacy.csv"
    ranked.write_text(
        "target_id,final_score,source_count,sources,efficacy_top1\n"
        "P2,1.0,3,autodock;gnina;rtmscore,hydration\n"
    )
    out_dir.mkdir(parents=True)
    (out_dir / "artifact_manifest.json").write_text("stale\n")

    res = run_script([
        "scripts/stage11_repro_pack.py",
        "--run-dir", str(run_dir),
        "--out-dir", str(out_dir),
    ])

    assert res.returncode != 0
    assert "column 'target_id' contains duplicate values P1" in res.stderr
    assert "top50_4way_consensus.csv" in res.stderr
    assert not (out_dir / "artifact_manifest.json").exists()


def test_repro_pack_rejects_ranked_targets_duplicate_source_labels(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    out_dir = run_dir / "publication" / "reproducibility"
    report = run_dir / "09_report" / "index.html"
    report.parent.mkdir(parents=True)
    report.write_text("<html>source-backed report</html>\n")
    consensus = run_dir / "03_targets" / "mode_comprehensive" / "top50_4way_consensus.csv"
    consensus.parent.mkdir(parents=True)
    consensus.write_text("target_id,score\nP1,1.0\n")
    ranked = run_dir / "03_targets" / "ranked_targets_v3_with_efficacy.csv"
    ranked.write_text(
        "target_id,final_score,source_count,sources,efficacy_top1\n"
        "P2,1.0,3,autodock;autodock;gnina,hydration\n"
    )
    out_dir.mkdir(parents=True)
    (out_dir / "artifact_manifest.json").write_text("stale\n")

    res = run_script([
        "scripts/stage11_repro_pack.py",
        "--run-dir", str(run_dir),
        "--out-dir", str(out_dir),
    ])

    assert res.returncode != 0
    assert "source artifact column 'sources' contains duplicate labels" in res.stderr
    assert "autodock" in res.stderr
    assert "ranked_targets_v3_with_efficacy.csv" in res.stderr
    assert not (out_dir / "artifact_manifest.json").exists()


def test_repro_pack_rejects_duplicate_resolved_source_artifacts(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    out_dir = run_dir / "publication" / "reproducibility"
    report = run_dir / "09_report" / "index.html"
    report.parent.mkdir(parents=True)
    report.write_text("<html>source-backed report</html>\n")
    ranked = run_dir / "03_targets" / "ranked_targets_v3_with_efficacy.csv"
    ranked.parent.mkdir(parents=True)
    ranked.write_text(
        "target_id,final_score,source_count,sources,efficacy_top1\n"
        "P1,1.0,3,autodock;gnina;rtmscore,hydration\n"
    )
    consensus = run_dir / "03_targets" / "mode_comprehensive" / (
        "top50_4way_consensus.csv"
    )
    consensus.parent.mkdir(parents=True)
    consensus.symlink_to(ranked)
    out_dir.mkdir(parents=True)
    (out_dir / "artifact_manifest.json").write_text("stale\n")

    res = run_script([
        "scripts/stage11_repro_pack.py",
        "--run-dir", str(run_dir),
        "--out-dir", str(out_dir),
    ])

    assert res.returncode != 0
    assert "source artifacts resolve to the same file" in res.stderr
    assert "ranked_targets_v3_with_efficacy.csv" in res.stderr
    assert "top50_4way_consensus.csv" in res.stderr
    assert not (out_dir / "artifact_manifest.json").exists()


def test_repro_pack_rejects_nonfinite_required_target_score(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    out_dir = run_dir / "publication" / "reproducibility"
    report = run_dir / "09_report" / "index.html"
    report.parent.mkdir(parents=True)
    report.write_text("<html>source-backed report</html>\n")
    consensus = run_dir / "03_targets" / "mode_comprehensive" / "top50_4way_consensus.csv"
    consensus.parent.mkdir(parents=True)
    consensus.write_text("target_id,score\nP1,inf\n")
    ranked = run_dir / "03_targets" / "ranked_targets_v3_with_efficacy.csv"
    ranked.write_text(
        "target_id,final_score,source_count,sources,efficacy_top1\n"
        "P2,1.0,3,autodock;gnina;rtmscore,hydration\n"
    )
    out_dir.mkdir(parents=True)
    (out_dir / "artifact_manifest.json").write_text("stale\n")

    res = run_script([
        "scripts/stage11_repro_pack.py",
        "--run-dir", str(run_dir),
        "--out-dir", str(out_dir),
    ])

    assert res.returncode != 0
    assert "non-finite numeric value" in res.stderr
    assert "top50_4way_consensus.csv" in res.stderr
    assert not (out_dir / "artifact_manifest.json").exists()


def test_repro_pack_rejects_nonnumeric_required_target_score(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    out_dir = run_dir / "publication" / "reproducibility"
    report = run_dir / "09_report" / "index.html"
    report.parent.mkdir(parents=True)
    report.write_text("<html>source-backed report</html>\n")
    consensus = run_dir / "03_targets" / "mode_comprehensive" / "top50_4way_consensus.csv"
    consensus.parent.mkdir(parents=True)
    consensus.write_text("target_id,score\nP1,not-a-score\n")
    ranked = run_dir / "03_targets" / "ranked_targets_v3_with_efficacy.csv"
    ranked.write_text(
        "target_id,final_score,source_count,sources,efficacy_top1\n"
        "P2,1.0,3,autodock;gnina;rtmscore,hydration\n"
    )
    out_dir.mkdir(parents=True)
    (out_dir / "artifact_manifest.json").write_text("stale\n")

    res = run_script([
        "scripts/stage11_repro_pack.py",
        "--run-dir", str(run_dir),
        "--out-dir", str(out_dir),
    ])

    assert res.returncode != 0
    assert "column 'score' must be numeric" in res.stderr
    assert "top50_4way_consensus.csv" in res.stderr
    assert not (out_dir / "artifact_manifest.json").exists()


def test_repro_pack_rejects_invalid_json_artifact(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out_dir = run_dir / "publication" / "reproducibility"
    report = run_dir / "09_report" / "index.html"
    report.parent.mkdir(parents=True)
    report.write_text("<html>source-backed report</html>\n")
    consensus = run_dir / "03_targets" / "mode_comprehensive" / "top50_4way_consensus.csv"
    consensus.parent.mkdir(parents=True)
    consensus.write_text("target_id,score\nP1,1.0\n")
    ranked = run_dir / "03_targets" / "ranked_targets_v3_with_efficacy.csv"
    ranked.write_text(
        "target_id,final_score,source_count,sources,efficacy_top1\n"
        "P2,1.0,3,autodock;gnina;rtmscore,hydration\n"
    )
    payload = run_dir / "02_admet" / "admet_report.json"
    payload.parent.mkdir(parents=True)
    payload.write_text('{"decision": ')
    out_dir.mkdir(parents=True)
    (out_dir / "artifact_manifest.json").write_text("stale\n")

    res = run_script([
        "scripts/stage11_repro_pack.py",
        "--run-dir", str(run_dir),
        "--out-dir", str(out_dir),
    ])

    assert res.returncode != 0
    assert "failed to parse as JSON" in res.stderr
    assert "02_admet/admet_report.json" in res.stderr
    assert not (out_dir / "artifact_manifest.json").exists()


def test_repro_pack_rejects_empty_json_artifact(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out_dir = run_dir / "publication" / "reproducibility"
    report = run_dir / "09_report" / "index.html"
    report.parent.mkdir(parents=True)
    report.write_text("<html>source-backed report</html>\n")
    consensus = run_dir / "03_targets" / "mode_comprehensive" / "top50_4way_consensus.csv"
    consensus.parent.mkdir(parents=True)
    consensus.write_text("target_id,score\nP1,1.0\n")
    ranked = run_dir / "03_targets" / "ranked_targets_v3_with_efficacy.csv"
    ranked.write_text(
        "target_id,final_score,source_count,sources,efficacy_top1\n"
        "P2,1.0,3,autodock;gnina;rtmscore,hydration\n"
    )
    payload = run_dir / "02_admet" / "admet_report.json"
    payload.parent.mkdir(parents=True)
    payload.write_text("{}\n")
    out_dir.mkdir(parents=True)
    (out_dir / "artifact_manifest.json").write_text("stale\n")

    res = run_script([
        "scripts/stage11_repro_pack.py",
        "--run-dir", str(run_dir),
        "--out-dir", str(out_dir),
    ])

    assert res.returncode != 0
    assert "JSON source artifact contains no keys" in res.stderr
    assert "02_admet/admet_report.json" in res.stderr
    assert not (out_dir / "artifact_manifest.json").exists()


def test_repro_pack_rejects_invalid_seed_metadata(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out_dir = run_dir / "publication" / "reproducibility"
    report = run_dir / "09_report" / "index.html"
    report.parent.mkdir(parents=True)
    report.write_text("<html>source-backed report</html>\n")
    consensus = run_dir / "03_targets" / "mode_comprehensive" / "top50_4way_consensus.csv"
    consensus.parent.mkdir(parents=True)
    consensus.write_text("target_id,score\nP1,1.0\n")
    ranked = run_dir / "03_targets" / "ranked_targets_v3_with_efficacy.csv"
    ranked.write_text(
        "target_id,final_score,source_count,sources,efficacy_top1\n"
        "P1,1.0,3,autodock;gnina;rtmscore,hydration\n"
    )
    md_dir = run_dir / "07_md"
    md_dir.mkdir(parents=True)
    (md_dir / "trajectory_index.tsv").write_text(
        "target_id\treplica\tseed\nP1\t1\t-1\n"
    )
    out_dir.mkdir(parents=True)
    (out_dir / "random_seeds.json").write_text("stale\n")

    res = run_script([
        "scripts/stage11_repro_pack.py",
        "--run-dir", str(run_dir),
        "--out-dir", str(out_dir),
    ])

    assert res.returncode != 0
    assert "Stage seed artifact column 'seed'" in res.stderr
    assert "non-negative integer" in res.stderr
    assert not (out_dir / "random_seeds.json").exists()


def test_repro_pack_requires_eval_manifest_for_claim_ready_output(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    out_dir = run_dir / "publication" / "reproducibility"
    report = run_dir / "09_report" / "index.html"
    report.parent.mkdir(parents=True)
    report.write_text("<html>source-backed report</html>\n")
    consensus = run_dir / "03_targets" / "mode_comprehensive" / "top50_4way_consensus.csv"
    consensus.parent.mkdir(parents=True)
    consensus.write_text("target_id,score\nP1,1.0\n")
    ranked = run_dir / "03_targets" / "ranked_targets_v3_with_efficacy.csv"
    ranked.write_text(
        "target_id,final_score,source_count,sources,efficacy_top1\n"
        "P1,1.0,3,autodock;gnina;rtmscore,hydration\n"
    )
    out_dir.mkdir(parents=True)
    (out_dir / "artifact_manifest.json").write_text("stale\n")

    res = run_script([
        "scripts/stage11_repro_pack.py",
        "--run-dir", str(run_dir),
        "--out-dir", str(out_dir),
    ])

    assert res.returncode != 0
    assert "eval manifest is required for claim-ready publication" in res.stderr
    assert not (out_dir / "artifact_manifest.json").exists()


def test_repro_pack_rejects_non_claim_ready_eval_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = run_dir / "publication" / "reproducibility"
    out_dir.mkdir(parents=True)
    (out_dir / "artifact_manifest.json").write_text("stale\n")
    write_minimal_claim_sources(run_dir, eval_dir)
    consensus = run_dir / "03_targets" / "mode_comprehensive" / "top50_4way_consensus.csv"
    consensus.parent.mkdir(parents=True, exist_ok=True)
    consensus.write_text("target_id,score\nP1,1.0\n")
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["threshold_status"]["status"] = "failed"
    payload["threshold_status"]["n_failed"] = 1
    payload["threshold_status"]["failures"] = [{
        "path": str(eval_dir / "cosmetic_retrospective.csv"),
        "n_rows": 1,
        "n_failed_rows": 1,
        "n_invalid_rows": 0,
    }]
    manifest.write_text(json.dumps(payload) + "\n")

    res = run_script([
        "scripts/stage11_repro_pack.py",
        "--run-dir", str(run_dir),
        "--out-dir", str(out_dir),
        "--eval-manifest", str(manifest),
    ])

    assert res.returncode != 0
    assert "eval manifest is not claim-ready" in res.stderr
    assert "threshold_status.status must be 'ok'" in res.stderr
    assert not (out_dir / "artifact_manifest.json").exists()


def test_repro_pack_rejects_eval_manifest_config_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = run_dir / "publication" / "reproducibility"
    out_dir.mkdir(parents=True)
    (out_dir / "artifact_manifest.json").write_text("stale\n")
    write_minimal_claim_sources(run_dir, eval_dir)
    consensus = run_dir / "03_targets" / "mode_comprehensive" / "top50_4way_consensus.csv"
    consensus.parent.mkdir(parents=True, exist_ok=True)
    consensus.write_text("target_id,score\nP1,1.0\n")
    manifest = eval_dir / "iteration_manifest.json"
    mismatched_config = tmp_path / "workflow" / "config.yaml"
    mismatched_config.parent.mkdir(parents=True)
    mismatched_config.write_text("run_id: different\n")

    res = run_script([
        "scripts/stage11_repro_pack.py",
        "--run-dir", str(run_dir),
        "--out-dir", str(out_dir),
        "--config", str(mismatched_config),
        "--eval-manifest", str(manifest),
    ])

    assert res.returncode != 0
    assert "workflow config digest must match eval manifest" in res.stderr
    assert "provenance.config_sha256" in res.stderr
    assert not (out_dir / "artifact_manifest.json").exists()


def test_repro_pack_rejects_eval_manifest_git_commit_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = run_dir / "publication" / "reproducibility"
    out_dir.mkdir(parents=True)
    (out_dir / "artifact_manifest.json").write_text("stale\n")
    write_minimal_claim_sources(run_dir, eval_dir)
    consensus = run_dir / "03_targets" / "mode_comprehensive" / "top50_4way_consensus.csv"
    consensus.parent.mkdir(parents=True, exist_ok=True)
    consensus.write_text("target_id,score\nP1,1.0\n")
    manifest = eval_dir / "iteration_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["provenance"]["git_commit"] = "b" * 40
    manifest.write_text(json.dumps(payload) + "\n")
    workflow_config = payload["data_snapshot"]["workflow_config"]["path"]

    res = run_script([
        "scripts/stage11_repro_pack.py",
        "--run-dir", str(run_dir),
        "--out-dir", str(out_dir),
        "--config", workflow_config,
        "--eval-manifest", str(manifest),
    ])

    assert res.returncode != 0
    assert "git commit must match eval manifest" in res.stderr
    assert "provenance.git_commit" in res.stderr
    assert not (out_dir / "artifact_manifest.json").exists()


def test_repro_pack_rejects_eval_manifest_run_dir_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    other_run_dir = tmp_path / "other_run"
    out_dir = run_dir / "publication" / "reproducibility"
    out_dir.mkdir(parents=True)
    (out_dir / "artifact_manifest.json").write_text("stale\n")
    write_minimal_claim_sources(run_dir, eval_dir)
    consensus = run_dir / "03_targets" / "mode_comprehensive" / "top50_4way_consensus.csv"
    consensus.parent.mkdir(parents=True, exist_ok=True)
    consensus.write_text("target_id,score\nP1,1.0\n")
    manifest = eval_dir / "iteration_manifest.json"
    write_claim_ready_eval_manifest(manifest, source_run_dir=other_run_dir)
    workflow_config = json.loads(
        manifest.read_text()
    )["data_snapshot"]["workflow_config"]["path"]

    res = run_script([
        "scripts/stage11_repro_pack.py",
        "--run-dir", str(run_dir),
        "--out-dir", str(out_dir),
        "--config", workflow_config,
        "--eval-manifest", str(manifest),
    ])

    assert res.returncode != 0
    assert "eval manifest input_runs must include --run-dir" in res.stderr
    assert str(run_dir) in res.stderr
    assert not (out_dir / "artifact_manifest.json").exists()


def test_repro_pack_records_artifact_hashes(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    out_dir = run_dir / "publication" / "reproducibility"
    consensus_rel = Path("03_targets/mode_comprehensive/top50_4way_consensus.csv")
    ranked_rel = Path("03_targets/ranked_targets_v3_with_efficacy.csv")
    source = run_dir / "09_report" / "index.html"
    source.parent.mkdir(parents=True)
    source.write_text("<html>source-backed report</html>\n")
    consensus = run_dir / consensus_rel
    consensus.parent.mkdir(parents=True)
    consensus.write_text("target_id,score\nP1,1.0\n")
    ranked = run_dir / ranked_rel
    ranked.write_text(
        "target_id,final_score,source_count,sources,efficacy_top1\n"
        "P1,1.0,3,autodock;gnina;rtmscore,hydration\n"
    )
    eval_manifest = eval_dir / "iteration_manifest.json"
    write_claim_ready_eval_manifest(eval_manifest, source_run_dir=run_dir)
    workflow_config = json.loads(
        eval_manifest.read_text()
    )["data_snapshot"]["workflow_config"]["path"]

    res = run_script([
        "scripts/stage11_repro_pack.py",
        "--run-dir", str(run_dir),
        "--out-dir", str(out_dir),
        "--config", workflow_config,
        "--eval-manifest", str(eval_manifest),
    ])

    assert res.returncode == 0, res.stderr
    manifest = json.loads((out_dir / "artifact_manifest.json").read_text())
    artifact_map = {
        artifact["relative_path"]: artifact
        for artifact in manifest["artifacts"]
    }
    assert manifest["n_artifacts"] == 5
    assert artifact_map["09_report/index.html"]["bytes"] == source.stat().st_size
    assert artifact_map["09_report/index.html"]["sha256"] == hashlib.sha256(
        source.read_bytes()
    ).hexdigest()
    assert str(consensus_rel) in artifact_map
    assert str(ranked_rel) in artifact_map
    assert artifact_map["EVAL:iteration_manifest.json"]["path"] == str(eval_manifest)
    assert artifact_map["EVAL:iteration_manifest.json"]["bytes"] == (
        eval_manifest.stat().st_size
    )
    assert artifact_map["EVAL:iteration_manifest.json"]["sha256"] == (
        hashlib.sha256(eval_manifest.read_bytes()).hexdigest()
    )
    assert artifact_map["EVAL:iteration_manifest.json"]["source_git_commit"] == (
        current_git_commit()
    )
    assert artifact_map["EVAL:iteration_manifest.json"]["source_config_sha256"] == (
        hashlib.sha256(Path(workflow_config).read_bytes()).hexdigest()
    )
    assert artifact_map["EVAL:iteration_manifest.json"]["source_run_dirs"] == [
        str(run_dir)
    ]
    metadata_map = {
        metadata["relative_path"]: metadata
        for metadata in manifest["metadata_files"]
    }
    assert set(metadata_map) == {
        "config_hash.txt",
        "git_commit.txt",
        "code_snapshot.json",
        "effective_config.json",
        "random_seeds.json",
        "runtime_log.txt",
        "tool_versions.lock",
    }
    for rel, metadata in metadata_map.items():
        metadata_path = out_dir / rel
        assert metadata["bytes"] == metadata_path.stat().st_size
        assert metadata["sha256"] == hashlib.sha256(
            metadata_path.read_bytes()
        ).hexdigest()
    assert (out_dir / "config_hash.txt").read_text().strip() != "missing"
    assert (out_dir / "tool_versions.lock").exists()
    assert (out_dir / "git_commit.txt").exists()
    assert (out_dir / "random_seeds.json").exists()
    assert (out_dir / "runtime_log.txt").exists()
    code_snapshot = json.loads((out_dir / "code_snapshot.json").read_text())
    assert code_snapshot["schema_version"] == "skinscout.stage11-code-snapshot.v1"
    assert code_snapshot["git_commit"] == current_git_commit()
    effective_config = json.loads((out_dir / "effective_config.json").read_text())
    assert effective_config["schema_version"] == (
        "skinscout.stage11-effective-config.v1"
    )
    assert effective_config["base_config_sha256"] == hashlib.sha256(
        Path(workflow_config).read_bytes()
    ).hexdigest()


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def load_make_figures_module():
    spec = importlib.util.spec_from_file_location(
        "stage11_make_figures",
        ROOT / "scripts" / "stage11_make_figures.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manuscript_outputs_are_not_required_when_draft_opt_out() -> None:
    rules = (ROOT / "workflow/rules/stage11_publication.smk").read_text()

    unconditional_inputs = rules.split("S11_CLAIM_INPUTS = {", 1)[1].split("}", 1)[0]
    assert "manuscript_draft" not in unconditional_inputs
    assert "S11_EMIT_MANUSCRIPT_DRAFT" in rules
    assert "if S11_EMIT_MANUSCRIPT_DRAFT:" in rules
    assert "S11_CLAIM_INPUTS.update(S11_MANUSCRIPT_OUTPUTS)" in rules
    assert 'S11_CLAIM_INPUTS["manuscript_methods"]' in rules

    unconditional_artifacts = rules.split("S11_CLAIM_ARTIFACTS = (", 1)[1].split(
        "\n)", 1
    )[0]
    assert "manuscript_draft" not in unconditional_artifacts


def test_diagnostic_bundle_path_is_explicit_and_fail_closed() -> None:
    rules = (ROOT / "workflow/rules/stage11_publication.smk").read_text()
    snakefile = (ROOT / "workflow/Snakefile").read_text()

    assert 'S11_BUNDLE_MODE = config_choice(' in rules
    assert '{"claim_quality", "diagnostic"}' in rules
    assert 'S11_DIAGNOSTIC_BUNDLE = S11_BUNDLE_MODE == "diagnostic"' in rules
    assert (
        '"--allow-diagnostic" if S11_DIAGNOSTIC_BUNDLE else ""' in rules
    )
    assert "if S11_DIAGNOSTIC_BUNDLE and S11_EMIT_MANUSCRIPT_DRAFT:" in rules
    assert 'PUBLICATION["bundle_mode"] = config_choice(' in snakefile
    assert 'PUBLICATION["bundle_mode"] == "diagnostic"' in snakefile
    assert '"publication.bundle_mode"' in snakefile


def test_source_backed_figures_require_value_unit_and_hash() -> None:
    module = load_make_figures_module()
    caption_path = Path("captions.json")
    count_source = {
        "path": "x.csv",
        "metric": "rows",
        "value": 3,
        "sha256": "a" * 64,
    }

    assert module.caption_claim_status(
        "fig02", "artifact_evidence_counts", [count_source], caption_path
    ) == ("diagnostic_only", True, False)

    with pytest.raises(ValueError, match="unit"):
        module.caption_claim_status(
            "fig02", "measured_quantity", [count_source], caption_path
        )
    with pytest.raises(ValueError, match="SHA-256"):
        module.caption_claim_status(
            "fig02",
            "measured_quantity",
            [{**count_source, "unit": "nM", "sha256": "x"}],
            caption_path,
        )
    with pytest.raises(ValueError, match="finite positive"):
        module.caption_claim_status(
            "fig02",
            "measured_quantity",
            [{**count_source, "unit": "nM", "value": 0}],
            caption_path,
        )

    assert module.caption_claim_status(
        "fig02",
        "measured_quantity",
        [{**count_source, "unit": "nM"}],
        caption_path,
    ) == ("source_backed", False, True)


def write_diagnostic_bundle_run(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path]:
    run_dir = tmp_path / "run"
    captions = write_text(
        run_dir / "publication" / "figures" / "captions.json",
        json.dumps({
            "fig02_target_landscape.png": {
                "caption": "Artifact evidence counts for the sources behind: x",
                "status": "diagnostic_only",
                "diagnostic_only": True,
                "claim_eligible": False,
                "sources": [{
                    "path": "run/03_targets/ranked_targets_v3_with_efficacy.csv",
                    "metric": "rows",
                    "value": 3,
                    "sha256": "a" * 64,
                }],
            },
        }) + "\n",
    )
    eval_manifest = write_text(
        tmp_path / "eval" / "iteration_manifest.json",
        json.dumps({
            "claim_ready": False,
            "claim_blockers": [{"code": "threshold_failures"}],
            "n_steps": 1,
            "threshold_status": {"n_checked": 1, "n_failed": 1},
        }) + "\n",
    )
    out = run_dir / "publication" / "claim_manifest.json"
    return run_dir, captions, eval_manifest, out


def test_diagnostic_bundle_completes_with_not_ready_manifest(tmp_path: Path) -> None:
    run_dir, captions, eval_manifest, out = write_diagnostic_bundle_run(tmp_path)

    res = run_script([
        "scripts/stage11_claim_manifest.py",
        "--run-dir", str(run_dir),
        "--out-manifest", str(out),
        "--evaluation-manifest", str(eval_manifest),
        "--artifact", f"figure_captions={captions}",
        "--allow-diagnostic",
    ])

    assert res.returncode == 0, res.stderr
    payload = json.loads(out.read_text())
    assert payload["claim_ready"] is False
    codes = {blocker["code"] for blocker in payload["blockers"]}
    assert "diagnostic_only_status" in codes
    assert "claim_ineligible_status" in codes


def test_claim_quality_rejects_diagnostic_bundle(tmp_path: Path) -> None:
    run_dir, captions, eval_manifest, out = write_diagnostic_bundle_run(tmp_path)

    res = run_script([
        "scripts/stage11_claim_manifest.py",
        "--run-dir", str(run_dir),
        "--out-manifest", str(out),
        "--evaluation-manifest", str(eval_manifest),
        "--artifact", f"figure_captions={captions}",
    ])

    assert res.returncode != 0
    assert "diagnostic_only_status" in res.stderr
    assert not out.exists()


def test_claim_manifest_rejects_source_backed_caption_without_measured_evidence(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    captions = write_text(
        run_dir / "publication" / "figures" / "captions.json",
        json.dumps({
            "fig02_target_landscape.png": {
                "caption": "Counted rows relabelled as a measured figure.",
                "status": "source_backed",
                "diagnostic_only": False,
                "claim_eligible": True,
                "sources": [{
                    "path": "run/03_targets/ranked_targets_v3_with_efficacy.csv",
                    "metric": "rows",
                    "value": 3,
                    "sha256": "a" * 64,
                }],
            },
        }) + "\n",
    )
    eval_manifest = write_text(
        tmp_path / "eval" / "iteration_manifest.json",
        json.dumps({
            "claim_ready": True,
            "claim_blockers": [],
            "n_steps": 1,
            "threshold_status": {"n_checked": 1, "n_failed": 0},
        }) + "\n",
    )
    out = run_dir / "publication" / "claim_manifest.json"

    res = run_script([
        "scripts/stage11_claim_manifest.py",
        "--run-dir", str(run_dir),
        "--out-manifest", str(out),
        "--evaluation-manifest", str(eval_manifest),
        "--artifact", f"figure_captions={captions}",
    ])

    assert res.returncode != 0
    assert "source_backed_caption_missing_measured_evidence" in res.stderr
    assert not out.exists()


def test_publication_eval_dir_override_is_config_driven() -> None:
    rules = (ROOT / "workflow/rules/stage11_publication.smk").read_text()
    snakefile = (ROOT / "workflow/Snakefile").read_text()

    assert 'S11_EVAL_DIR = PROJECT_ROOT / "results" / "eval"' in rules
    assert "S11_EVAL_DIR_OVERRIDE" in rules
    assert 'PUBLICATION.get("evaluation_dir", "")' in rules
    assert '"publication.evaluation_dir"' in snakefile
