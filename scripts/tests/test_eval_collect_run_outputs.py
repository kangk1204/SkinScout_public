"""Unit tests for collecting run outputs into eval harness inputs."""

from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_collect_run_outputs_creates_eval_targets_and_rankings(tmp_path: Path) -> None:
    run = tmp_path / "retinol_case"
    (run / "01_input").mkdir(parents=True)
    (run / "03_targets" / "mode_comprehensive").mkdir(parents=True)
    (run / "03_targets").mkdir(parents=True, exist_ok=True)
    (run / "01_input" / "compound_canonical.json").write_text(
        json.dumps({"canonical_smiles": "CCO"})
    )
    pd.DataFrame([
        {
            "target_id": "P2",
            "final_rank": 2,
            "final_score": 0.8,
            "source_count": 4,
            "sources": "autodock;boltz;gnina;rtmscore",
            "efficacy_top1": "whitening",
        },
        {
            "target_id": "P1",
            "final_rank": 1,
            "final_score": 0.9,
            "source_count": 4,
            "sources": "autodock;boltz;gnina;rtmscore",
            "efficacy_top1": "anti_aging",
        },
    ]).to_csv(run / "03_targets" / "ranked_targets_v3_with_efficacy.csv", index=False)
    pd.DataFrame([
        {
            "target_id": "P1",
            "rrf_score": 1.0,
            "source_count": 4,
            "sources": "autodock;boltz;gnina;rtmscore",
        },
        {
            "target_id": "P2",
            "rrf_score": 0.9,
            "source_count": 4,
            "sources": "autodock;boltz;gnina;rtmscore",
        },
    ]).to_csv(run / "03_targets" / "mode_comprehensive" / "top50_4way_consensus.csv", index=False)
    pd.DataFrame([
        {"target_id": "P1", "psichic_score": 0.71, "score": 0.71},
    ]).to_csv(
        run / "03_targets" / "mode_comprehensive" / "psichic_sanity.tsv",
        sep="\t",
        index=False,
    )
    fasta = tmp_path / "seqs.fasta"
    fasta.write_text(">P1\nMAAA\n")

    out_dir = tmp_path / "eval"
    rankings = tmp_path / "rankings"
    res = subprocess.run(
        [
            sys.executable,
            "eval/collect_run_outputs.py",
            "--run-dirs", str(run),
            "--out-dir", str(out_dir),
            "--rankings-dir", str(rankings),
            "--sequence-fasta", str(fasta),
            "--top-n-leakage", "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode == 0
    targets = pd.read_csv(out_dir / "eval_targets.csv")
    assert targets.to_dict("records") == [{
        "run_id": "retinol_case",
        "uniprot": "P1",
        "smiles": "CCO",
        "sequence": "MAAA",
    }]
    assert (rankings / "cosmetic_retro" / "retinol_case__ranked_targets_v3.csv").exists()
    assert (
        rankings
        / "cosmetic_retro"
        / "retinol_case__ranked_targets_v3_with_efficacy.csv"
    ).exists()
    assert (rankings / "cold_start__comprehensive.csv").exists()
    dti = pd.read_csv(rankings / "cold_start__dti_only.csv")
    assert dti.to_dict("records") == [{
        "target_id": "P1",
        "psichic_score": 0.71,
        "score": 0.71,
    }]
    manifest = json.loads((out_dir / "collected_runs.json").read_text())
    assert manifest["n_runs"] == 1
    assert manifest["n_leakage_rows"] == 1
    copied = manifest["runs"][0]["copied"]
    copied_artifacts = manifest["runs"][0]["copied_artifacts"]
    assert (
        str(rankings / "cosmetic_retro" / "retinol_case__ranked_targets_v3.csv")
        in copied
    )
    assert (
        str(
            rankings
            / "cosmetic_retro"
            / "retinol_case__ranked_targets_v3_with_efficacy.csv"
        )
        in copied
    )
    assert [artifact["path"] for artifact in copied_artifacts] == copied
    for artifact in copied_artifacts:
        path = Path(artifact["path"])
        source = Path(artifact["source_path"])
        assert artifact["source_path"]
        assert artifact["source_bytes"] == source.stat().st_size
        assert artifact["source_sha256"] == hashlib.sha256(
            source.read_bytes()
        ).hexdigest()
        assert artifact["bytes"] == path.stat().st_size
        assert artifact["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_collect_run_outputs_fails_when_compound_metadata_missing(tmp_path: Path) -> None:
    run = tmp_path / "missing_compound_case"
    (run / "01_input").mkdir(parents=True)
    (run / "03_targets").mkdir(parents=True)
    pd.DataFrame([{"target_id": "P1", "final_score": 0.9}]).to_csv(
        run / "03_targets" / "ranked_targets_v3.csv",
        index=False,
    )
    out_dir = tmp_path / "eval"
    out_dir.mkdir()
    (out_dir / "collected_runs.json").write_text('{"stale": true}\n')
    (out_dir / "iteration_manifest.json").write_text('{"stale": true}\n')
    (out_dir / "eval_targets.csv").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            "eval/collect_run_outputs.py",
            "--run-dirs", str(run),
            "--out-dir", str(out_dir),
            "--rankings-dir", str(tmp_path / "rankings"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "Canonical compound metadata is required" in res.stderr
    assert not (out_dir / "collected_runs.json").exists()
    assert not (out_dir / "iteration_manifest.json").exists()
    assert not (out_dir / "eval_targets.csv").exists()


def test_collect_run_outputs_rejects_blank_compound_smiles(tmp_path: Path) -> None:
    run = tmp_path / "blank_smiles_case"
    (run / "01_input").mkdir(parents=True)
    (run / "03_targets").mkdir(parents=True)
    (run / "01_input" / "compound_canonical.json").write_text(
        json.dumps({"canonical_smiles": "   "})
    )
    pd.DataFrame([{"target_id": "P1", "final_score": 0.9}]).to_csv(
        run / "03_targets" / "ranked_targets_v3.csv",
        index=False,
    )
    out_dir = tmp_path / "eval"
    rankings = tmp_path / "rankings"
    retro = rankings / "cosmetic_retro"
    out_dir.mkdir()
    retro.mkdir(parents=True)
    (out_dir / "collected_runs.json").write_text('{"stale": true}\n')
    (out_dir / "eval_targets.csv").write_text("stale\n")
    (retro / "blank_smiles_case__ranked_targets_v3.csv").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            "eval/collect_run_outputs.py",
            "--run-dirs", str(run),
            "--out-dir", str(out_dir),
            "--rankings-dir", str(rankings),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "missing non-empty 'canonical_smiles'" in res.stderr
    assert not (out_dir / "collected_runs.json").exists()
    assert not (out_dir / "eval_targets.csv").exists()
    assert not (retro / "blank_smiles_case__ranked_targets_v3.csv").exists()


def test_collect_run_outputs_rejects_invalid_compound_smiles(tmp_path: Path) -> None:
    run = tmp_path / "invalid_smiles_case"
    (run / "01_input").mkdir(parents=True)
    (run / "03_targets").mkdir(parents=True)
    (run / "01_input" / "compound_canonical.json").write_text(
        json.dumps({"canonical_smiles": "not-a-smiles"})
    )
    pd.DataFrame([{"target_id": "P1", "final_score": 0.9}]).to_csv(
        run / "03_targets" / "ranked_targets_v3.csv",
        index=False,
    )
    out_dir = tmp_path / "eval"
    rankings = tmp_path / "rankings"
    retro = rankings / "cosmetic_retro"
    out_dir.mkdir()
    retro.mkdir(parents=True)
    (out_dir / "collected_runs.json").write_text('{"stale": true}\n')
    (out_dir / "eval_targets.csv").write_text("stale\n")
    (retro / "invalid_smiles_case__ranked_targets_v3.csv").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            "eval/collect_run_outputs.py",
            "--run-dirs", str(run),
            "--out-dir", str(out_dir),
            "--rankings-dir", str(rankings),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "contains invalid canonical_smiles" in res.stderr
    assert not (out_dir / "collected_runs.json").exists()
    assert not (out_dir / "eval_targets.csv").exists()
    assert not (retro / "invalid_smiles_case__ranked_targets_v3.csv").exists()


def test_collect_run_outputs_rejects_empty_fasta_sequence(tmp_path: Path) -> None:
    run = tmp_path / "empty_fasta_case"
    (run / "01_input").mkdir(parents=True)
    (run / "03_targets").mkdir(parents=True)
    (run / "01_input" / "compound_canonical.json").write_text(
        json.dumps({"canonical_smiles": "CCO"})
    )
    pd.DataFrame([{"target_id": "P1", "final_score": 0.9}]).to_csv(
        run / "03_targets" / "ranked_targets_v3.csv",
        index=False,
    )
    fasta = tmp_path / "seqs.fasta"
    fasta.write_text(">P1\n>P2\nMAAA\n")
    out_dir = tmp_path / "eval"
    rankings = tmp_path / "rankings"
    retro = rankings / "cosmetic_retro"
    out_dir.mkdir()
    retro.mkdir(parents=True)
    (out_dir / "collected_runs.json").write_text('{"stale": true}\n')
    (out_dir / "eval_targets.csv").write_text("stale\n")
    (retro / "empty_fasta_case__ranked_targets_v3.csv").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            "eval/collect_run_outputs.py",
            "--run-dirs", str(run),
            "--out-dir", str(out_dir),
            "--rankings-dir", str(rankings),
            "--sequence-fasta", str(fasta),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "Sequence FASTA entry 'P1' has no sequence" in res.stderr
    assert not (out_dir / "collected_runs.json").exists()
    assert not (out_dir / "eval_targets.csv").exists()
    assert not (retro / "empty_fasta_case__ranked_targets_v3.csv").exists()


def test_collect_run_outputs_rejects_explicit_missing_sequence_fasta(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "eval"
    rankings = tmp_path / "rankings"
    result = subprocess.run(
        [
            sys.executable,
            "eval/collect_run_outputs.py",
            "--run-dirs",
            str(tmp_path / "unused-run"),
            "--out-dir",
            str(out_dir),
            "--rankings-dir",
            str(rankings),
            "--sequence-fasta",
            str(tmp_path / "missing.fasta"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Explicit sequence FASTA" in result.stderr
    assert not (out_dir / "collected_runs.json").exists()


def test_collect_run_outputs_rejects_duplicate_fasta_header(tmp_path: Path) -> None:
    run = tmp_path / "duplicate_fasta_case"
    (run / "01_input").mkdir(parents=True)
    (run / "03_targets").mkdir(parents=True)
    (run / "01_input" / "compound_canonical.json").write_text(
        json.dumps({"canonical_smiles": "CCO"})
    )
    pd.DataFrame([{"target_id": "P1", "final_score": 0.9}]).to_csv(
        run / "03_targets" / "ranked_targets_v3.csv",
        index=False,
    )
    fasta = tmp_path / "seqs.fasta"
    fasta.write_text(">P1\nMAAA\n>P1\nMBBB\n")
    out_dir = tmp_path / "eval"
    rankings = tmp_path / "rankings"
    retro = rankings / "cosmetic_retro"
    retro.mkdir(parents=True)
    out_dir.mkdir()
    (out_dir / "collected_runs.json").write_text('{"stale": true}\n')
    (out_dir / "eval_targets.csv").write_text("stale\n")
    (retro / "duplicate_fasta_case__ranked_targets_v3.csv").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            "eval/collect_run_outputs.py",
            "--run-dirs", str(run),
            "--out-dir", str(out_dir),
            "--rankings-dir", str(rankings),
            "--sequence-fasta", str(fasta),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "Sequence FASTA contains duplicate entry 'P1'" in res.stderr
    assert not (out_dir / "collected_runs.json").exists()
    assert not (out_dir / "eval_targets.csv").exists()
    assert not (retro / "duplicate_fasta_case__ranked_targets_v3.csv").exists()


def test_collect_run_outputs_rejects_fasta_sequence_before_header(
    tmp_path: Path,
) -> None:
    run = tmp_path / "orphan_fasta_case"
    (run / "01_input").mkdir(parents=True)
    (run / "03_targets").mkdir(parents=True)
    (run / "01_input" / "compound_canonical.json").write_text(
        json.dumps({"canonical_smiles": "CCO"})
    )
    pd.DataFrame([{"target_id": "P1", "final_score": 0.9}]).to_csv(
        run / "03_targets" / "ranked_targets_v3.csv",
        index=False,
    )
    fasta = tmp_path / "seqs.fasta"
    fasta.write_text("MAAA\n>P1\nMBBB\n")
    out_dir = tmp_path / "eval"
    out_dir.mkdir()
    (out_dir / "collected_runs.json").write_text('{"stale": true}\n')
    (out_dir / "eval_targets.csv").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            "eval/collect_run_outputs.py",
            "--run-dirs", str(run),
            "--out-dir", str(out_dir),
            "--rankings-dir", str(tmp_path / "rankings"),
            "--sequence-fasta", str(fasta),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "Sequence FASTA contains sequence before first header" in res.stderr
    assert not (out_dir / "collected_runs.json").exists()
    assert not (out_dir / "eval_targets.csv").exists()


def test_collect_run_outputs_rejects_nonpositive_top_n_leakage(
    tmp_path: Path,
) -> None:
    run = tmp_path / "zero_leakage_case"
    (run / "01_input").mkdir(parents=True)
    (run / "03_targets").mkdir(parents=True)
    (run / "01_input" / "compound_canonical.json").write_text(
        json.dumps({"canonical_smiles": "CCO"})
    )
    pd.DataFrame([{"target_id": "P1", "final_score": 0.9}]).to_csv(
        run / "03_targets" / "ranked_targets_v3.csv",
        index=False,
    )
    out_dir = tmp_path / "eval"
    rankings = tmp_path / "rankings"
    out_dir.mkdir()
    (out_dir / "collected_runs.json").write_text('{"stale": true}\n')
    (out_dir / "eval_targets.csv").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            "eval/collect_run_outputs.py",
            "--run-dirs", str(run),
            "--out-dir", str(out_dir),
            "--rankings-dir", str(rankings),
            "--top-n-leakage", "0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "--top-n-leakage must be a positive integer" in res.stderr
    assert not (out_dir / "collected_runs.json").exists()
    assert not (out_dir / "eval_targets.csv").exists()


def test_collect_run_outputs_fails_when_no_rankings_exist(tmp_path: Path) -> None:
    run = tmp_path / "no_rankings_case"
    (run / "01_input").mkdir(parents=True)
    (run / "01_input" / "compound_canonical.json").write_text(
        json.dumps({"canonical_smiles": "CCO"})
    )

    res = subprocess.run(
        [
            sys.executable,
            "eval/collect_run_outputs.py",
            "--run-dirs", str(run),
            "--out-dir", str(tmp_path / "eval"),
            "--rankings-dir", str(tmp_path / "rankings"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "No evaluation ranking outputs found" in res.stderr


def test_collect_run_outputs_fails_on_empty_ranked_targets(tmp_path: Path) -> None:
    run = tmp_path / "empty_ranking_case"
    (run / "01_input").mkdir(parents=True)
    (run / "03_targets").mkdir(parents=True)
    (run / "01_input" / "compound_canonical.json").write_text(
        json.dumps({"canonical_smiles": "CCO"})
    )
    (run / "03_targets" / "ranked_targets_v3.csv").write_text("")

    res = subprocess.run(
        [
            sys.executable,
            "eval/collect_run_outputs.py",
            "--run-dirs", str(run),
            "--out-dir", str(tmp_path / "eval"),
            "--rankings-dir", str(tmp_path / "rankings"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "Skin-weighted target ranking is required" in res.stderr


def test_collect_run_outputs_rejects_blank_skin_weighted_target_id(
    tmp_path: Path,
) -> None:
    run = tmp_path / "blank_skin_case"
    (run / "01_input").mkdir(parents=True)
    (run / "03_targets").mkdir(parents=True)
    (run / "01_input" / "compound_canonical.json").write_text(
        json.dumps({"canonical_smiles": "CCO"})
    )
    pd.DataFrame([
        {"target_id": "P1", "final_score": 0.9},
        {"target_id": " ", "final_score": 0.8},
    ]).to_csv(run / "03_targets" / "ranked_targets_v3.csv", index=False)
    out_dir = tmp_path / "eval"
    rankings = tmp_path / "rankings"
    retro = rankings / "cosmetic_retro"
    retro.mkdir(parents=True)
    (out_dir).mkdir()
    (out_dir / "collected_runs.json").write_text('{"stale": true}\n')
    (out_dir / "eval_targets.csv").write_text("stale\n")
    (retro / "blank_skin_case__ranked_targets_v3.csv").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            "eval/collect_run_outputs.py",
            "--run-dirs", str(run),
            "--out-dir", str(out_dir),
            "--rankings-dir", str(rankings),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert (
        "Skin-weighted target ranking column 'target_id' contains blank values"
        in res.stderr
    )
    assert not (out_dir / "collected_runs.json").exists()
    assert not (out_dir / "eval_targets.csv").exists()
    assert not (retro / "blank_skin_case__ranked_targets_v3.csv").exists()


def test_collect_run_outputs_rejects_duplicate_skin_weighted_target_id(
    tmp_path: Path,
) -> None:
    run = tmp_path / "duplicate_skin_case"
    (run / "01_input").mkdir(parents=True)
    (run / "03_targets").mkdir(parents=True)
    (run / "01_input" / "compound_canonical.json").write_text(
        json.dumps({"canonical_smiles": "CCO"})
    )
    pd.DataFrame([
        {"target_id": "P1", "final_score": 0.9},
        {"target_id": "P1", "final_score": 0.8},
    ]).to_csv(run / "03_targets" / "ranked_targets_v3.csv", index=False)
    out_dir = tmp_path / "eval"
    rankings = tmp_path / "rankings"
    retro = rankings / "cosmetic_retro"
    retro.mkdir(parents=True)
    out_dir.mkdir()
    (out_dir / "collected_runs.json").write_text('{"stale": true}\n')
    (out_dir / "eval_targets.csv").write_text("stale\n")
    (retro / "duplicate_skin_case__ranked_targets_v3.csv").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            "eval/collect_run_outputs.py",
            "--run-dirs", str(run),
            "--out-dir", str(out_dir),
            "--rankings-dir", str(rankings),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert (
        "Skin-weighted target ranking contains duplicate target_id values: P1"
        in res.stderr
    )
    assert not (out_dir / "collected_runs.json").exists()
    assert not (out_dir / "eval_targets.csv").exists()
    assert not (retro / "duplicate_skin_case__ranked_targets_v3.csv").exists()


def test_collect_run_outputs_rejects_nonfinite_skin_weighted_score(
    tmp_path: Path,
) -> None:
    run = tmp_path / "nonfinite_skin_case"
    (run / "01_input").mkdir(parents=True)
    (run / "03_targets").mkdir(parents=True)
    (run / "01_input" / "compound_canonical.json").write_text(
        json.dumps({"canonical_smiles": "CCO"})
    )
    (run / "03_targets" / "ranked_targets_v3.csv").write_text(
        "target_id,final_score\nP1,inf\n"
    )
    out_dir = tmp_path / "eval"
    rankings = tmp_path / "rankings"
    retro = rankings / "cosmetic_retro"
    retro.mkdir(parents=True)
    out_dir.mkdir()
    (out_dir / "collected_runs.json").write_text('{"stale": true}\n')
    (out_dir / "eval_targets.csv").write_text("stale\n")
    (retro / "nonfinite_skin_case__ranked_targets_v3.csv").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            "eval/collect_run_outputs.py",
            "--run-dirs", str(run),
            "--out-dir", str(out_dir),
            "--rankings-dir", str(rankings),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert (
        "Skin-weighted target ranking column 'final_score' must be finite"
        in res.stderr
    )
    assert not (out_dir / "collected_runs.json").exists()
    assert not (out_dir / "eval_targets.csv").exists()
    assert not (retro / "nonfinite_skin_case__ranked_targets_v3.csv").exists()


def test_collect_run_outputs_rejects_out_of_range_skin_weighted_score(
    tmp_path: Path,
) -> None:
    run = tmp_path / "out_of_range_skin_case"
    (run / "01_input").mkdir(parents=True)
    (run / "03_targets").mkdir(parents=True)
    (run / "01_input" / "compound_canonical.json").write_text(
        json.dumps({"canonical_smiles": "CCO"})
    )
    pd.DataFrame([{"target_id": "P1", "final_score": 1.2}]).to_csv(
        run / "03_targets" / "ranked_targets_v3.csv",
        index=False,
    )
    out_dir = tmp_path / "eval"
    rankings = tmp_path / "rankings"
    retro = rankings / "cosmetic_retro"
    retro.mkdir(parents=True)
    out_dir.mkdir()
    (out_dir / "collected_runs.json").write_text('{"stale": true}\n')
    (out_dir / "eval_targets.csv").write_text("stale\n")
    (retro / "out_of_range_skin_case__ranked_targets_v3.csv").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            "eval/collect_run_outputs.py",
            "--run-dirs", str(run),
            "--out-dir", str(out_dir),
            "--rankings-dir", str(rankings),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert (
        "Skin-weighted target ranking column 'final_score' must be in [0, 1]"
        in res.stderr
    )
    assert not (out_dir / "collected_runs.json").exists()
    assert not (out_dir / "eval_targets.csv").exists()
    assert not (retro / "out_of_range_skin_case__ranked_targets_v3.csv").exists()


def test_collect_run_outputs_rejects_skin_weighted_without_scorer_rationale(
    tmp_path: Path,
) -> None:
    run = tmp_path / "missing_skin_sources_case"
    (run / "01_input").mkdir(parents=True)
    (run / "03_targets").mkdir(parents=True)
    (run / "01_input" / "compound_canonical.json").write_text(
        json.dumps({"canonical_smiles": "CCO"})
    )
    pd.DataFrame([{"target_id": "P1", "final_score": 0.9}]).to_csv(
        run / "03_targets" / "ranked_targets_v3.csv",
        index=False,
    )
    out_dir = tmp_path / "eval"
    rankings = tmp_path / "rankings"
    retro = rankings / "cosmetic_retro"
    out_dir.mkdir()
    retro.mkdir(parents=True)
    (out_dir / "collected_runs.json").write_text('{"stale": true}\n')
    (out_dir / "eval_targets.csv").write_text("stale\n")
    (retro / "missing_skin_sources_case__ranked_targets_v3.csv").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            "eval/collect_run_outputs.py",
            "--run-dirs", str(run),
            "--out-dir", str(out_dir),
            "--rankings-dir", str(rankings),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "Skin-weighted target ranking missing required scorer-rationale columns" in res.stderr
    assert "source_count" in res.stderr
    assert "sources" in res.stderr
    assert not (out_dir / "collected_runs.json").exists()
    assert not (out_dir / "eval_targets.csv").exists()
    assert not (retro / "missing_skin_sources_case__ranked_targets_v3.csv").exists()


def test_collect_run_outputs_rejects_skin_weighted_source_count_mismatch(
    tmp_path: Path,
) -> None:
    run = tmp_path / "mismatch_skin_sources_case"
    (run / "01_input").mkdir(parents=True)
    (run / "03_targets").mkdir(parents=True)
    (run / "01_input" / "compound_canonical.json").write_text(
        json.dumps({"canonical_smiles": "CCO"})
    )
    pd.DataFrame([{
        "target_id": "P1",
        "final_score": 0.9,
        "source_count": 4,
        "sources": "autodock",
    }]).to_csv(
        run / "03_targets" / "ranked_targets_v3.csv",
        index=False,
    )
    out_dir = tmp_path / "eval"
    rankings = tmp_path / "rankings"
    retro = rankings / "cosmetic_retro"
    out_dir.mkdir()
    retro.mkdir(parents=True)
    (out_dir / "collected_runs.json").write_text('{"stale": true}\n')
    (out_dir / "eval_targets.csv").write_text("stale\n")
    (retro / "mismatch_skin_sources_case__ranked_targets_v3.csv").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            "eval/collect_run_outputs.py",
            "--run-dirs", str(run),
            "--out-dir", str(out_dir),
            "--rankings-dir", str(rankings),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "Skin-weighted target ranking source_count=4 but sources lists 1 label(s)" in res.stderr
    assert not (out_dir / "collected_runs.json").exists()
    assert not (out_dir / "eval_targets.csv").exists()
    assert not (retro / "mismatch_skin_sources_case__ranked_targets_v3.csv").exists()


def test_collect_run_outputs_rejects_skin_weighted_nonnumeric_source_count(
    tmp_path: Path,
) -> None:
    run = tmp_path / "nonnumeric_skin_sources_case"
    (run / "01_input").mkdir(parents=True)
    (run / "03_targets").mkdir(parents=True)
    (run / "01_input" / "compound_canonical.json").write_text(
        json.dumps({"canonical_smiles": "CCO"})
    )
    pd.DataFrame([{
        "target_id": "P1",
        "final_score": 0.9,
        "source_count": "not-a-number",
        "sources": "autodock;gnina;rtmscore",
    }]).to_csv(
        run / "03_targets" / "ranked_targets_v3.csv",
        index=False,
    )
    out_dir = tmp_path / "eval"
    rankings = tmp_path / "rankings"
    retro = rankings / "cosmetic_retro"
    out_dir.mkdir()
    retro.mkdir(parents=True)
    (out_dir / "collected_runs.json").write_text('{"stale": true}\n')
    (out_dir / "eval_targets.csv").write_text("stale\n")
    (retro / "nonnumeric_skin_sources_case__ranked_targets_v3.csv").write_text(
        "stale\n"
    )

    res = subprocess.run(
        [
            sys.executable,
            "eval/collect_run_outputs.py",
            "--run-dirs", str(run),
            "--out-dir", str(out_dir),
            "--rankings-dir", str(rankings),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert (
        "Skin-weighted target ranking column 'source_count' must contain "
        "integer values"
    ) in res.stderr
    assert "Traceback" not in res.stderr
    assert not (out_dir / "collected_runs.json").exists()
    assert not (out_dir / "eval_targets.csv").exists()
    assert not (retro / "nonnumeric_skin_sources_case__ranked_targets_v3.csv").exists()


def test_collect_run_outputs_rejects_skin_weighted_duplicate_source_labels(
    tmp_path: Path,
) -> None:
    run = tmp_path / "duplicate_skin_sources_case"
    (run / "01_input").mkdir(parents=True)
    (run / "03_targets").mkdir(parents=True)
    (run / "01_input" / "compound_canonical.json").write_text(
        json.dumps({"canonical_smiles": "CCO"})
    )
    pd.DataFrame([{
        "target_id": "P1",
        "final_score": 0.9,
        "source_count": 3,
        "sources": "autodock;autodock;gnina",
    }]).to_csv(
        run / "03_targets" / "ranked_targets_v3.csv",
        index=False,
    )
    out_dir = tmp_path / "eval"
    rankings = tmp_path / "rankings"
    retro = rankings / "cosmetic_retro"
    out_dir.mkdir()
    retro.mkdir(parents=True)
    (out_dir / "collected_runs.json").write_text('{"stale": true}\n')
    (out_dir / "eval_targets.csv").write_text("stale\n")
    (retro / "duplicate_skin_sources_case__ranked_targets_v3.csv").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            "eval/collect_run_outputs.py",
            "--run-dirs", str(run),
            "--out-dir", str(out_dir),
            "--rankings-dir", str(rankings),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "Skin-weighted target ranking column 'sources' contains duplicate labels" in res.stderr
    assert "autodock" in res.stderr
    assert not (out_dir / "collected_runs.json").exists()
    assert not (out_dir / "eval_targets.csv").exists()
    assert not (retro / "duplicate_skin_sources_case__ranked_targets_v3.csv").exists()


def test_collect_run_outputs_rejects_skin_weighted_target_absent_from_stage3(
    tmp_path: Path,
) -> None:
    run = tmp_path / "mismatched_skin_stage3_case"
    (run / "01_input").mkdir(parents=True)
    (run / "03_targets" / "mode_comprehensive").mkdir(parents=True)
    (run / "03_targets").mkdir(parents=True, exist_ok=True)
    (run / "01_input" / "compound_canonical.json").write_text(
        json.dumps({"canonical_smiles": "CCO"})
    )
    pd.DataFrame([{
        "target_id": "P2",
        "final_score": 0.9,
        "source_count": 4,
        "sources": "autodock;boltz;gnina;rtmscore",
    }]).to_csv(
        run / "03_targets" / "ranked_targets_v3.csv",
        index=False,
    )
    pd.DataFrame([{
        "target_id": "P1",
        "rrf_score": 1.0,
        "source_count": 4,
        "sources": "autodock;boltz;gnina;rtmscore",
    }]).to_csv(
        run / "03_targets" / "mode_comprehensive" / "top50_4way_consensus.csv",
        index=False,
    )
    out_dir = tmp_path / "eval"
    rankings = tmp_path / "rankings"
    retro = rankings / "cosmetic_retro"
    out_dir.mkdir()
    retro.mkdir(parents=True)
    (out_dir / "collected_runs.json").write_text('{"stale": true}\n')
    (out_dir / "eval_targets.csv").write_text("stale\n")
    (retro / "mismatched_skin_stage3_case__ranked_targets_v3.csv").write_text(
        "stale\n"
    )
    (rankings / "cold_start__comprehensive.csv").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            "eval/collect_run_outputs.py",
            "--run-dirs", str(run),
            "--out-dir", str(out_dir),
            "--rankings-dir", str(rankings),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert (
        "Skin-weighted target ranking contains target_id values absent from "
        "Comprehensive target ranking"
    ) in res.stderr
    assert "P2" in res.stderr
    assert not (out_dir / "collected_runs.json").exists()
    assert not (out_dir / "eval_targets.csv").exists()
    assert not (
        retro / "mismatched_skin_stage3_case__ranked_targets_v3.csv"
    ).exists()
    assert not (rankings / "cold_start__comprehensive.csv").exists()


def test_collect_run_outputs_rejects_with_efficacy_without_efficacy_columns(
    tmp_path: Path,
) -> None:
    run = tmp_path / "missing_efficacy_case"
    (run / "01_input").mkdir(parents=True)
    (run / "03_targets").mkdir(parents=True)
    (run / "01_input" / "compound_canonical.json").write_text(
        json.dumps({"canonical_smiles": "CCO"})
    )
    pd.DataFrame([{"target_id": "P1", "final_score": 0.9}]).to_csv(
        run / "03_targets" / "ranked_targets_v3_with_efficacy.csv",
        index=False,
    )
    out_dir = tmp_path / "eval"
    rankings = tmp_path / "rankings"
    retro = rankings / "cosmetic_retro"
    retro.mkdir(parents=True)
    out_dir.mkdir()
    (out_dir / "collected_runs.json").write_text('{"stale": true}\n')
    (out_dir / "eval_targets.csv").write_text("stale\n")
    (retro / "missing_efficacy_case__ranked_targets_v3.csv").write_text("stale\n")
    (
        retro / "missing_efficacy_case__ranked_targets_v3_with_efficacy.csv"
    ).write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            "eval/collect_run_outputs.py",
            "--run-dirs", str(run),
            "--out-dir", str(out_dir),
            "--rankings-dir", str(rankings),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "missing efficacy_top* columns" in res.stderr
    assert not (out_dir / "collected_runs.json").exists()
    assert not (out_dir / "eval_targets.csv").exists()
    assert not (retro / "missing_efficacy_case__ranked_targets_v3.csv").exists()
    assert not (
        retro / "missing_efficacy_case__ranked_targets_v3_with_efficacy.csv"
    ).exists()


def test_collect_run_outputs_rejects_with_efficacy_without_efficacy_values(
    tmp_path: Path,
) -> None:
    run = tmp_path / "empty_efficacy_case"
    (run / "01_input").mkdir(parents=True)
    (run / "03_targets").mkdir(parents=True)
    (run / "01_input" / "compound_canonical.json").write_text(
        json.dumps({"canonical_smiles": "CCO"})
    )
    pd.DataFrame([{"target_id": "P1", "final_score": 0.9, "efficacy_top1": " "}]).to_csv(
        run / "03_targets" / "ranked_targets_v3_with_efficacy.csv",
        index=False,
    )
    out_dir = tmp_path / "eval"
    rankings = tmp_path / "rankings"
    retro = rankings / "cosmetic_retro"
    retro.mkdir(parents=True)
    out_dir.mkdir()
    (out_dir / "collected_runs.json").write_text('{"stale": true}\n')
    (out_dir / "eval_targets.csv").write_text("stale\n")
    (retro / "empty_efficacy_case__ranked_targets_v3.csv").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            "eval/collect_run_outputs.py",
            "--run-dirs", str(run),
            "--out-dir", str(out_dir),
            "--rankings-dir", str(rankings),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert (
        "top-10 rows must each contain at least one non-empty efficacy_top*"
        in res.stderr
    )
    assert "missing row index(es) 0" in res.stderr
    assert not (out_dir / "collected_runs.json").exists()
    assert not (out_dir / "eval_targets.csv").exists()
    assert not (retro / "empty_efficacy_case__ranked_targets_v3.csv").exists()


def test_collect_run_outputs_rejects_top_evaluated_rows_without_efficacy_values(
    tmp_path: Path,
) -> None:
    run = tmp_path / "partial_efficacy_case"
    (run / "01_input").mkdir(parents=True)
    (run / "03_targets").mkdir(parents=True)
    (run / "01_input" / "compound_canonical.json").write_text(
        json.dumps({"canonical_smiles": "CCO"})
    )
    pd.DataFrame([
        {
            "target_id": "P1",
            "final_score": 0.9,
            "source_count": 4,
            "sources": "autodock;boltz;gnina;rtmscore",
            "efficacy_top1": " ",
        },
        {
            "target_id": "P2",
            "final_score": 0.8,
            "source_count": 4,
            "sources": "autodock;boltz;gnina;rtmscore",
            "efficacy_top1": "whitening",
        },
    ]).to_csv(
        run / "03_targets" / "ranked_targets_v3_with_efficacy.csv",
        index=False,
    )
    out_dir = tmp_path / "eval"
    rankings = tmp_path / "rankings"
    retro = rankings / "cosmetic_retro"
    retro.mkdir(parents=True)
    out_dir.mkdir()
    (out_dir / "collected_runs.json").write_text('{"stale": true}\n')
    (out_dir / "eval_targets.csv").write_text("stale\n")
    (retro / "partial_efficacy_case__ranked_targets_v3.csv").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            "eval/collect_run_outputs.py",
            "--run-dirs", str(run),
            "--out-dir", str(out_dir),
            "--rankings-dir", str(rankings),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert (
        "top-10 rows must each contain at least one non-empty efficacy_top*"
        in res.stderr
    )
    assert "missing row index(es) 0" in res.stderr
    assert not (out_dir / "collected_runs.json").exists()
    assert not (out_dir / "eval_targets.csv").exists()
    assert not (retro / "partial_efficacy_case__ranked_targets_v3.csv").exists()


def test_collect_run_outputs_rejects_blank_comprehensive_target_id(
    tmp_path: Path,
) -> None:
    run = tmp_path / "blank_comprehensive_case"
    (run / "01_input").mkdir(parents=True)
    (run / "03_targets" / "mode_comprehensive").mkdir(parents=True)
    (run / "01_input" / "compound_canonical.json").write_text(
        json.dumps({"canonical_smiles": "CCO"})
    )
    pd.DataFrame([
        {
            "target_id": "P1",
            "rrf_score": 1.0,
            "source_count": 4,
            "sources": "autodock;boltz;gnina;rtmscore",
        },
        {
            "target_id": "",
            "rrf_score": 0.9,
            "source_count": 4,
            "sources": "autodock;boltz;gnina;rtmscore",
        },
    ]).to_csv(
        run / "03_targets" / "mode_comprehensive" / "top50_4way_consensus.csv",
        index=False,
    )
    out_dir = tmp_path / "eval"
    rankings = tmp_path / "rankings"
    out_dir.mkdir()
    rankings.mkdir()
    (out_dir / "collected_runs.json").write_text('{"stale": true}\n')
    (out_dir / "eval_targets.csv").write_text("stale\n")
    (rankings / "cold_start__comprehensive.csv").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            "eval/collect_run_outputs.py",
            "--run-dirs", str(run),
            "--out-dir", str(out_dir),
            "--rankings-dir", str(rankings),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert (
        "Comprehensive target ranking column 'target_id' contains blank values"
        in res.stderr
    )
    assert not (out_dir / "collected_runs.json").exists()
    assert not (out_dir / "eval_targets.csv").exists()
    assert not (rankings / "cold_start__comprehensive.csv").exists()


def test_collect_run_outputs_rejects_comprehensive_without_scorer_rationale(
    tmp_path: Path,
) -> None:
    run = tmp_path / "missing_comprehensive_sources_case"
    (run / "01_input").mkdir(parents=True)
    (run / "03_targets" / "mode_comprehensive").mkdir(parents=True)
    (run / "01_input" / "compound_canonical.json").write_text(
        json.dumps({"canonical_smiles": "CCO"})
    )
    pd.DataFrame([{"target_id": "P1", "rrf_score": 1.0}]).to_csv(
        run / "03_targets" / "mode_comprehensive" / "top50_4way_consensus.csv",
        index=False,
    )
    out_dir = tmp_path / "eval"
    rankings = tmp_path / "rankings"
    out_dir.mkdir()
    rankings.mkdir()
    (out_dir / "collected_runs.json").write_text('{"stale": true}\n')
    (rankings / "cold_start__comprehensive.csv").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            "eval/collect_run_outputs.py",
            "--run-dirs", str(run),
            "--out-dir", str(out_dir),
            "--rankings-dir", str(rankings),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "missing required scorer-rationale columns" in res.stderr
    assert "source_count" in res.stderr
    assert "sources" in res.stderr
    assert not (out_dir / "collected_runs.json").exists()
    assert not (rankings / "cold_start__comprehensive.csv").exists()


def test_collect_run_outputs_rejects_comprehensive_below_min_source_count(
    tmp_path: Path,
) -> None:
    run = tmp_path / "weak_comprehensive_sources_case"
    (run / "01_input").mkdir(parents=True)
    (run / "03_targets" / "mode_comprehensive").mkdir(parents=True)
    (run / "01_input" / "compound_canonical.json").write_text(
        json.dumps({"canonical_smiles": "CCO"})
    )
    pd.DataFrame([{
        "target_id": "P1",
        "rrf_score": 1.0,
        "source_count": 2,
        "sources": "autodock;gnina",
    }]).to_csv(
        run / "03_targets" / "mode_comprehensive" / "top50_4way_consensus.csv",
        index=False,
    )
    out_dir = tmp_path / "eval"
    rankings = tmp_path / "rankings"
    out_dir.mkdir()
    rankings.mkdir()
    (out_dir / "collected_runs.json").write_text('{"stale": true}\n')
    (rankings / "cold_start__comprehensive.csv").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            "eval/collect_run_outputs.py",
            "--run-dirs", str(run),
            "--out-dir", str(out_dir),
            "--rankings-dir", str(rankings),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "source_count' must be >= 3" in res.stderr
    assert not (out_dir / "collected_runs.json").exists()
    assert not (rankings / "cold_start__comprehensive.csv").exists()


def test_collect_run_outputs_rejects_fast_below_min_source_count(
    tmp_path: Path,
) -> None:
    run = tmp_path / "weak_fast_sources_case"
    (run / "01_input").mkdir(parents=True)
    (run / "03_targets" / "mode_fast").mkdir(parents=True)
    (run / "01_input" / "compound_canonical.json").write_text(
        json.dumps({"canonical_smiles": "CCO"})
    )
    pd.DataFrame([{
        "target_id": "P1",
        "final_score": 1.0,
        "source_count": 1,
        "sources": "psichic",
    }]).to_csv(run / "03_targets" / "mode_fast" / "top50.csv", index=False)
    out_dir = tmp_path / "eval"
    rankings = tmp_path / "rankings"
    out_dir.mkdir()
    rankings.mkdir()
    (out_dir / "collected_runs.json").write_text('{"stale": true}\n')
    (rankings / "cold_start__fast.csv").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            "eval/collect_run_outputs.py",
            "--run-dirs", str(run),
            "--out-dir", str(out_dir),
            "--rankings-dir", str(rankings),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "source_count' must be >= 2" in res.stderr
    assert not (out_dir / "collected_runs.json").exists()
    assert not (rankings / "cold_start__fast.csv").exists()


def test_collect_run_outputs_rejects_blank_dti_only_target_id(
    tmp_path: Path,
) -> None:
    run = tmp_path / "blank_dti_case"
    (run / "01_input").mkdir(parents=True)
    (run / "03_targets" / "mode_comprehensive").mkdir(parents=True)
    (run / "01_input" / "compound_canonical.json").write_text(
        json.dumps({"canonical_smiles": "CCO"})
    )
    pd.DataFrame([
        {"target_id": "P1", "psichic_score": 0.8},
        {"target_id": "  ", "psichic_score": 0.7},
    ]).to_csv(
        run / "03_targets" / "mode_comprehensive" / "psichic_sanity.tsv",
        sep="\t",
        index=False,
    )
    out_dir = tmp_path / "eval"
    rankings = tmp_path / "rankings"
    out_dir.mkdir()
    rankings.mkdir()
    (out_dir / "collected_runs.json").write_text('{"stale": true}\n')
    (out_dir / "eval_targets.csv").write_text("stale\n")
    (rankings / "cold_start__dti_only.csv").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            "eval/collect_run_outputs.py",
            "--run-dirs", str(run),
            "--out-dir", str(out_dir),
            "--rankings-dir", str(rankings),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert (
        "DTI-only target ranking column 'target_id' contains blank values"
        in res.stderr
    )
    assert not (out_dir / "collected_runs.json").exists()
    assert not (out_dir / "eval_targets.csv").exists()
    assert not (rankings / "cold_start__dti_only.csv").exists()


def test_collect_run_outputs_rejects_nonfinite_dti_only_score(
    tmp_path: Path,
) -> None:
    run = tmp_path / "nonfinite_dti_case"
    (run / "01_input").mkdir(parents=True)
    (run / "03_targets" / "mode_comprehensive").mkdir(parents=True)
    (run / "01_input" / "compound_canonical.json").write_text(
        json.dumps({"canonical_smiles": "CCO"})
    )
    (run / "03_targets" / "mode_comprehensive" / "psichic_sanity.tsv").write_text(
        "target_id\tpsichic_score\nP1\tinf\n"
    )
    out_dir = tmp_path / "eval"
    rankings = tmp_path / "rankings"
    out_dir.mkdir()
    rankings.mkdir()
    (out_dir / "collected_runs.json").write_text('{"stale": true}\n')
    (out_dir / "eval_targets.csv").write_text("stale\n")
    (rankings / "cold_start__dti_only.csv").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            "eval/collect_run_outputs.py",
            "--run-dirs", str(run),
            "--out-dir", str(out_dir),
            "--rankings-dir", str(rankings),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert (
        "DTI-only target ranking column 'psichic_score' must be finite"
        in res.stderr
    )
    assert not (out_dir / "collected_runs.json").exists()
    assert not (out_dir / "eval_targets.csv").exists()
    assert not (rankings / "cold_start__dti_only.csv").exists()


def test_collect_run_outputs_rejects_boolean_dti_only_score(
    tmp_path: Path,
) -> None:
    run = tmp_path / "boolean_dti_case"
    (run / "01_input").mkdir(parents=True)
    (run / "03_targets" / "mode_comprehensive").mkdir(parents=True)
    (run / "01_input" / "compound_canonical.json").write_text(
        json.dumps({"canonical_smiles": "CCO"})
    )
    (run / "03_targets" / "mode_comprehensive" / "psichic_sanity.tsv").write_text(
        "target_id\tpsichic_score\nP1\tTrue\n"
    )
    out_dir = tmp_path / "eval"
    rankings = tmp_path / "rankings"
    out_dir.mkdir()
    rankings.mkdir()
    (out_dir / "collected_runs.json").write_text('{"stale": true}\n')
    (out_dir / "eval_targets.csv").write_text("stale\n")
    (rankings / "cold_start__dti_only.csv").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            "eval/collect_run_outputs.py",
            "--run-dirs", str(run),
            "--out-dir", str(out_dir),
            "--rankings-dir", str(rankings),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert (
        "DTI-only target ranking column 'psichic_score' must be numeric"
        in res.stderr
    )
    assert not (out_dir / "collected_runs.json").exists()
    assert not (out_dir / "eval_targets.csv").exists()
    assert not (rankings / "cold_start__dti_only.csv").exists()


def test_collect_run_outputs_rejects_out_of_range_dti_only_score(
    tmp_path: Path,
) -> None:
    run = tmp_path / "out_of_range_dti_case"
    (run / "01_input").mkdir(parents=True)
    (run / "03_targets" / "mode_comprehensive").mkdir(parents=True)
    (run / "01_input" / "compound_canonical.json").write_text(
        json.dumps({"canonical_smiles": "CCO"})
    )
    (run / "03_targets" / "mode_comprehensive" / "psichic_sanity.tsv").write_text(
        "target_id\tpsichic_score\nP1\t-0.1\n"
    )
    out_dir = tmp_path / "eval"
    rankings = tmp_path / "rankings"
    out_dir.mkdir()
    rankings.mkdir()
    (out_dir / "collected_runs.json").write_text('{"stale": true}\n')
    (out_dir / "eval_targets.csv").write_text("stale\n")
    (rankings / "cold_start__dti_only.csv").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            "eval/collect_run_outputs.py",
            "--run-dirs", str(run),
            "--out-dir", str(out_dir),
            "--rankings-dir", str(rankings),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert (
        "DTI-only target ranking column 'psichic_score' must be in [0, 1]"
        in res.stderr
    )
    assert not (out_dir / "collected_runs.json").exists()
    assert not (out_dir / "eval_targets.csv").exists()
    assert not (rankings / "cold_start__dti_only.csv").exists()


def test_collect_run_outputs_removes_stale_rankings_on_failure(tmp_path: Path) -> None:
    run = tmp_path / "partial_failure_case"
    (run / "01_input").mkdir(parents=True)
    (run / "03_targets" / "mode_comprehensive").mkdir(parents=True)
    (run / "03_targets").mkdir(parents=True, exist_ok=True)
    (run / "01_input" / "compound_canonical.json").write_text(
        json.dumps({"canonical_smiles": "CCO"})
    )
    pd.DataFrame([{
        "target_id": "P1",
        "final_score": 0.9,
        "source_count": 4,
        "sources": "autodock;boltz;gnina;rtmscore",
    }]).to_csv(
        run / "03_targets" / "ranked_targets_v3.csv",
        index=False,
    )
    (run / "03_targets" / "mode_comprehensive" / "top50_4way_consensus.csv").write_text(
        ""
    )
    out_dir = tmp_path / "eval"
    rankings = tmp_path / "rankings"
    retro = rankings / "cosmetic_retro"
    out_dir.mkdir()
    retro.mkdir(parents=True)
    (rankings / "cold_start__comprehensive.csv").write_text("stale\n")
    (rankings / "cold_start__fast.csv").write_text("stale\n")
    (rankings / "cold_start__dti_only.csv").write_text("stale\n")
    (retro / "partial_failure_case__ranked_targets_v3.csv").write_text("stale\n")
    (out_dir / "iteration_manifest.json").write_text('{"stale": true}\n')

    res = subprocess.run(
        [
            sys.executable,
            "eval/collect_run_outputs.py",
            "--run-dirs", str(run),
            "--out-dir", str(out_dir),
            "--rankings-dir", str(rankings),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "Comprehensive target ranking is required" in res.stderr
    assert not (out_dir / "collected_runs.json").exists()
    assert not (out_dir / "iteration_manifest.json").exists()
    assert not (out_dir / "eval_targets.csv").exists()
    assert not (rankings / "cold_start__comprehensive.csv").exists()
    assert not (rankings / "cold_start__fast.csv").exists()
    assert not (rankings / "cold_start__dti_only.csv").exists()
    assert not (retro / "partial_failure_case__ranked_targets_v3.csv").exists()


def test_collect_run_outputs_rejects_ambiguous_cold_start_overwrite(
    tmp_path: Path,
) -> None:
    run_a = tmp_path / "case_a"
    run_b = tmp_path / "case_b"
    for run in (run_a, run_b):
        (run / "01_input").mkdir(parents=True)
        (run / "03_targets" / "mode_comprehensive").mkdir(parents=True)
        (run / "01_input" / "compound_canonical.json").write_text(
            json.dumps({"canonical_smiles": "CCO"})
        )
        pd.DataFrame([{"target_id": "P1", "rrf_score": 1.0}]).to_csv(
            run / "03_targets" / "mode_comprehensive" / "top50_4way_consensus.csv",
            index=False,
        )

    res = subprocess.run(
        [
            sys.executable,
            "eval/collect_run_outputs.py",
            "--run-dirs", str(run_a), str(run_b),
            "--out-dir", str(tmp_path / "eval"),
            "--rankings-dir", str(tmp_path / "rankings"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "would overwrite the single evaluator input" in res.stderr


def test_collect_run_outputs_rejects_ambiguous_dti_only_overwrite(
    tmp_path: Path,
) -> None:
    run_a = tmp_path / "case_a"
    run_b = tmp_path / "case_b"
    for run in (run_a, run_b):
        (run / "01_input").mkdir(parents=True)
        (run / "03_targets" / "mode_comprehensive").mkdir(parents=True)
        (run / "01_input" / "compound_canonical.json").write_text(
            json.dumps({"canonical_smiles": "CCO"})
        )
        pd.DataFrame([{"target_id": "P1", "psichic_score": 0.8}]).to_csv(
            run / "03_targets" / "mode_comprehensive" / "psichic_sanity.tsv",
            sep="\t",
            index=False,
        )

    res = subprocess.run(
        [
            sys.executable,
            "eval/collect_run_outputs.py",
            "--run-dirs", str(run_a), str(run_b),
            "--out-dir", str(tmp_path / "eval"),
            "--rankings-dir", str(tmp_path / "rankings"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "Multiple run dirs contain dti_only cold-start rankings" in res.stderr


def test_collect_run_outputs_rejects_duplicate_normalized_case_ids(
    tmp_path: Path,
) -> None:
    run_a = tmp_path / "case a"
    run_b = tmp_path / "case_a"
    for run in (run_a, run_b):
        (run / "01_input").mkdir(parents=True)
        (run / "03_targets").mkdir(parents=True)
        (run / "01_input" / "compound_canonical.json").write_text(
            json.dumps({"canonical_smiles": "CCO"})
        )
        pd.DataFrame([{"target_id": "P1", "final_score": 0.9}]).to_csv(
            run / "03_targets" / "ranked_targets_v3.csv",
            index=False,
        )
    out_dir = tmp_path / "eval"
    rankings = tmp_path / "rankings"
    retro = rankings / "cosmetic_retro"
    retro.mkdir(parents=True)
    out_dir.mkdir()
    (out_dir / "collected_runs.json").write_text('{"stale": true}\n')
    (out_dir / "eval_targets.csv").write_text("stale\n")
    (retro / "case_a__ranked_targets_v3.csv").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            "eval/collect_run_outputs.py",
            "--run-dirs", str(run_a), str(run_b),
            "--out-dir", str(out_dir),
            "--rankings-dir", str(rankings),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "normalize to the same evaluation case id" in res.stderr
    assert not (out_dir / "collected_runs.json").exists()
    assert not (out_dir / "eval_targets.csv").exists()
    assert not (retro / "case_a__ranked_targets_v3.csv").exists()


def test_collect_run_outputs_rejects_duplicate_resolved_run_dirs(
    tmp_path: Path,
) -> None:
    run = tmp_path / "case_a"
    run_alias = tmp_path / "case_b"
    (run / "01_input").mkdir(parents=True)
    (run / "03_targets").mkdir(parents=True)
    (run / "01_input" / "compound_canonical.json").write_text(
        json.dumps({"canonical_smiles": "CCO"})
    )
    pd.DataFrame([{
        "target_id": "P1",
        "final_score": 0.9,
        "source_count": 3,
        "sources": "autodock;gnina;rtmscore",
    }]).to_csv(run / "03_targets" / "ranked_targets_v3.csv", index=False)
    run_alias.symlink_to(run, target_is_directory=True)
    out_dir = tmp_path / "eval"
    rankings = tmp_path / "rankings"
    retro = rankings / "cosmetic_retro"
    retro.mkdir(parents=True)
    out_dir.mkdir()
    (out_dir / "collected_runs.json").write_text('{"stale": true}\n')
    (out_dir / "eval_targets.csv").write_text("stale\n")
    (retro / "case_a__ranked_targets_v3.csv").write_text("stale\n")
    (retro / "case_b__ranked_targets_v3.csv").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            "eval/collect_run_outputs.py",
            "--run-dirs", str(run), str(run_alias),
            "--out-dir", str(out_dir),
            "--rankings-dir", str(rankings),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "resolve to the same run directory" in res.stderr
    assert str(run.resolve()) in res.stderr
    assert not (out_dir / "collected_runs.json").exists()
    assert not (out_dir / "eval_targets.csv").exists()
    assert not (retro / "case_a__ranked_targets_v3.csv").exists()
    assert not (retro / "case_b__ranked_targets_v3.csv").exists()
