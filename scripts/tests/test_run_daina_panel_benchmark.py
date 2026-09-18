"""Regression tests for the Daina known-target panel benchmark harness."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "eval" / "run_daina_panel_benchmark.py"
sys.path.insert(0, str(ROOT / "eval"))

from run_daina_panel_benchmark import _normalize_paths_for_publish  # noqa: E402


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint_words(smiles: str) -> list[int]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprint = generator.GetFingerprint(Chem.MolFromSmiles(smiles))
    bits = np.fromiter((int(bit) for bit in fingerprint.ToBitString()), dtype=np.uint8)
    return np.packbits(bits).view(np.uint64).tolist()


def write_reference(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    chembl_fp = tmp_path / "fp_morgan2_2048.parquet"
    pd.DataFrame(
        [
            {"molecule_chembl_id": "CHEMBL1", "bitvec": fingerprint_words("CCO")},
            {"molecule_chembl_id": "CHEMBL2", "bitvec": fingerprint_words("c1ccccc1")},
        ]
    ).to_parquet(chembl_fp)
    human_activities = tmp_path / "human_activities.parquet"
    pd.DataFrame(
        [
            {
                "molecule_chembl_id": "CHEMBL1",
                "uniprot": "P11111",
                "standard_inchi_key": Chem.MolToInchiKey(Chem.MolFromSmiles("CCO")),
            },
            {
                "molecule_chembl_id": "CHEMBL2",
                "uniprot": "P22222",
                "standard_inchi_key": Chem.MolToInchiKey(
                    Chem.MolFromSmiles("c1ccccc1")
                ),
            },
        ]
    ).to_parquet(human_activities)
    source_manifest = {
        "schema_version": "chembl_activity_evidence.v1",
        "source": {
            "name": "ChEMBL",
            "release": "37",
            "license": "CC BY-SA 3.0",
        },
        "output_sha256": {
            "human_activities.parquet": sha256(human_activities),
        },
        "row_counts": {
            "human_activities": 2,
        },
    }
    source_manifest_path = tmp_path / "source_manifest.json"
    source_manifest_path.write_text(json.dumps(source_manifest, indent=2) + "\n")
    fingerprint_manifest = {
        "schema_version": "chembl_fingerprint_snapshot.v1",
        "source_snapshot": {
            "manifest_path": str(source_manifest_path),
            "manifest_sha256": sha256(source_manifest_path),
            "source_name": "ChEMBL",
            "source_release": "37",
            "source_license": "CC BY-SA 3.0",
        },
        "algorithm": {
            "name": "Morgan ECFP4",
            "radius": 2,
            "n_bits": 2048,
            "use_chirality": False,
        },
        "input": {
            "path": str(human_activities),
            "sha256": sha256(human_activities),
            "rows": 2,
        },
        "artifact": {
            "path": str(chembl_fp),
            "sha256": sha256(chembl_fp),
            "rows": 2,
        },
    }
    (tmp_path / "fingerprint_manifest.json").write_text(
        json.dumps(fingerprint_manifest, indent=2) + "\n"
    )
    return chembl_fp


def run_benchmark(
    *,
    panel: Path,
    chembl_fp: Path,
    out_dir: Path,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    args = [
        sys.executable,
        str(SCRIPT),
        "--panel-csv",
        str(panel),
        "--chembl-fp",
        str(chembl_fp),
        "--out-dir",
        str(out_dir),
    ]
    if extra_args:
        args.extend(extra_args)
    return subprocess.run(
        args,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def resolve_benchmark_path(out_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return out_dir / path


def test_publish_path_normalization_handles_relative_staging_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    staging = Path("results/.benchmark.daina-panel-staging")
    nested = staging / "rankings" / "case.csv"

    assert _normalize_paths_for_publish(str(nested), staging) == "rankings/case.csv"
    assert _normalize_paths_for_publish("leave-query-out", staging) == "leave-query-out"


def test_daina_panel_benchmark_runs_batch_converts_and_evaluates(tmp_path: Path) -> None:
    chembl_fp = write_reference(tmp_path)
    panel = tmp_path / "panel.csv"
    pd.DataFrame(
        [
            {
                "case_id": "ethanol",
                "smiles": "CCO",
                "known_targets": "P11111",
            }
        ]
    ).to_csv(panel, index=False)
    out_dir = tmp_path / "benchmark"

    result = run_benchmark(
        panel=panel,
        chembl_fp=chembl_fp,
        out_dir=out_dir,
        extra_args=[
            "--min-case-top10",
            "1.0",
            "--min-target-top10",
            "1.0",
            "--min-target-top30",
            "1.0",
        ],
    )

    assert result.returncode == 0, result.stderr
    ranking = pd.read_csv(out_dir / "rankings" / "ethanol__ranked_targets_v3.csv")
    assert list(ranking.columns) == [
        "rank",
        "target_id",
        "score",
        "known_target_prior",
        "known_target_prior_norm",
    ]
    assert ranking.iloc[0].to_dict() == {
        "rank": 1,
        "target_id": "P11111",
        "score": 1.0,
        "known_target_prior": 0.0,
        "known_target_prior_norm": 0.0,
    }
    summary = json.loads((out_dir / "skin_known_summary.json").read_text())
    assert summary["passes_threshold"] is True
    assert summary["evaluation_mode"] == "retrospective"
    sidecar = json.loads(
        (out_dir / "rankings" / "ethanol__ranked_targets_v3.metadata.json").read_text()
    )
    assert sidecar["case_id"] == "ethanol"
    assert sidecar["ranking_csv"].endswith("ethanol__ranked_targets_v3.csv")
    assert sidecar["ranking_sha256"]
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["parameters"]["evidence_mode"] == "retrieval"
    assert manifest["inputs"]["panel_csv_sha256"]
    assert manifest["inputs"]["chembl_fp_sha256"]
    assert manifest["inputs"]["human_activities_sha256"]
    assert manifest["inputs"]["source_manifest_sha256"]
    assert manifest["inputs"]["fingerprint_manifest_sha256"]
    assert len(manifest["subprocess_commands"]) == 2
    assert manifest["subprocess_commands"][0]["argv"][1].endswith(
        "scripts/stage3_daina_batch.py"
    )
    assert manifest["outputs"]["summary_path"].endswith("skin_known_summary.json")
    assert manifest["outputs"]["summary_sha256"] == sha256(
        out_dir / "skin_known_summary.json"
    )
    assert manifest["outputs"]["case_metrics_sha256"] == sha256(
        out_dir / "skin_known_cases.csv"
    )
    assert manifest["outputs"]["target_metrics_sha256"] == sha256(
        out_dir / "skin_known_targets.csv"
    )
    assert manifest["rankings"][0]["ranking_metadata_json"].endswith(
        "ethanol__ranked_targets_v3.metadata.json"
    )
    assert manifest["rankings"][0]["ranking_sha256"] == sidecar["ranking_sha256"]


def test_daina_panel_benchmark_failure_does_not_publish_partial_output(
    tmp_path: Path,
) -> None:
    chembl_fp = write_reference(tmp_path)
    panel = tmp_path / "panel.csv"
    pd.DataFrame(
        [
            {"case_id": "dup", "smiles": "CCO", "known_targets": "P11111"},
            {"case_id": "dup", "smiles": "CCN", "known_targets": "P22222"},
        ]
    ).to_csv(panel, index=False)
    out_dir = tmp_path / "benchmark"

    result = run_benchmark(
        panel=panel,
        chembl_fp=chembl_fp,
        out_dir=out_dir,
    )

    assert result.returncode != 0
    assert "duplicate case_id" in result.stderr
    assert not out_dir.exists()
    assert not (tmp_path / ".benchmark.daina-panel-staging").exists()


def test_published_manifest_and_sidecar_paths_resolve_after_atomic_rename(
    tmp_path: Path,
) -> None:
    chembl_fp = write_reference(tmp_path)
    panel = tmp_path / "panel.csv"
    pd.DataFrame(
        [{"case_id": "ethanol", "smiles": "CCO", "known_targets": "P11111"}]
    ).to_csv(panel, index=False)
    out_dir = tmp_path / "benchmark"

    result = run_benchmark(
        panel=panel,
        chembl_fp=chembl_fp,
        out_dir=out_dir,
        extra_args=[
            "--evidence-mode",
            "leave-query-out",
            "--evidence-snapshot-id",
            "chembl-fixture",
            "--allow-threshold-failure",
        ],
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads((out_dir / "manifest.json").read_text())
    for key in ("batch_csv", "evaluator_panel_csv", "sdf_dir"):
        assert resolve_benchmark_path(out_dir, manifest["generated_inputs"][key]).exists()
    for key in (
        "daina_batch_summary_json",
        "case_metrics_csv",
        "target_metrics_csv",
        "summary_path",
    ):
        assert resolve_benchmark_path(out_dir, manifest["outputs"][key]).exists()
    ranking_entry = manifest["rankings"][0]
    for key in ("daina_tsv", "daina_metadata_json", "ranking_csv", "ranking_metadata_json"):
        assert resolve_benchmark_path(out_dir, ranking_entry[key]).exists()

    ranking_sidecar = json.loads(
        resolve_benchmark_path(out_dir, ranking_entry["ranking_metadata_json"]).read_text()
    )
    assert not Path(ranking_sidecar["ranking_csv"]).is_absolute()
    assert resolve_benchmark_path(out_dir, ranking_sidecar["ranking_csv"]).exists()
    assert resolve_benchmark_path(out_dir, ranking_sidecar["inputs"]["ligand_sdf"]).exists()
    summary = json.loads(resolve_benchmark_path(out_dir, manifest["outputs"]["summary_path"]).read_text())
    assert resolve_benchmark_path(out_dir, summary["inputs"]["cases_csv"]).exists()
    assert resolve_benchmark_path(out_dir, summary["inputs"]["rankings_dir"]).exists()
    cases = pd.read_csv(resolve_benchmark_path(out_dir, manifest["outputs"]["case_metrics_csv"]))
    assert resolve_benchmark_path(out_dir, cases.loc[0, "ranking_path"]).exists()
    targets = pd.read_csv(
        resolve_benchmark_path(out_dir, manifest["outputs"]["target_metrics_csv"])
    )
    for metrics in (cases, targets):
        assert resolve_benchmark_path(
            out_dir, metrics.loc[0, "ranking_metadata_path"]
        ).exists()


def test_evaluator_panel_preserves_original_columns_and_only_defaults_missing_required(
    tmp_path: Path,
) -> None:
    chembl_fp = write_reference(tmp_path)
    panel = tmp_path / "panel.csv"
    pd.DataFrame(
        [
            {
                "case_id": "ethanol",
                "inci_name": "Ethyl Alcohol",
                "panel": "beneficial",
                "skin_effect": "barrier support",
                "smiles": "CCO",
                "known_targets": "P11111",
                "known_target_labels": "ADH",
                "evidence_note": "canonical positive control",
                "custom_semantics": "preserve-me",
            }
        ]
    ).to_csv(panel, index=False)
    out_dir = tmp_path / "benchmark"

    result = run_benchmark(
        panel=panel,
        chembl_fp=chembl_fp,
        out_dir=out_dir,
        extra_args=[
            "--min-case-top10",
            "1.0",
            "--min-target-top10",
            "1.0",
            "--min-target-top30",
            "1.0",
        ],
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads((out_dir / "manifest.json").read_text())
    evaluator_panel = pd.read_csv(
        resolve_benchmark_path(out_dir, manifest["generated_inputs"]["evaluator_panel_csv"])
    )
    assert list(evaluator_panel.columns) == [
        "case_id",
        "inci_name",
        "panel",
        "skin_effect",
        "smiles",
        "known_targets",
        "known_target_labels",
        "evidence_note",
        "custom_semantics",
    ]
    row = evaluator_panel.iloc[0]
    assert row["inci_name"] == "Ethyl Alcohol"
    assert row["panel"] == "beneficial"
    assert row["skin_effect"] == "barrier support"
    assert row["known_target_labels"] == "ADH"
    assert row["custom_semantics"] == "preserve-me"


def test_evidence_evaluation_mode_mismatch_is_rejected_without_publish(
    tmp_path: Path,
) -> None:
    chembl_fp = write_reference(tmp_path)
    panel = tmp_path / "panel.csv"
    pd.DataFrame(
        [{"case_id": "ethanol", "smiles": "CCO", "known_targets": "P11111"}]
    ).to_csv(panel, index=False)
    out_dir = tmp_path / "benchmark"

    result = run_benchmark(
        panel=panel,
        chembl_fp=chembl_fp,
        out_dir=out_dir,
        extra_args=[
            "--evidence-mode",
            "retrieval",
            "--evaluation-mode",
            "leave-query-out",
        ],
    )

    assert result.returncode != 0
    assert "Evidence/evaluation mode mismatch" in result.stderr
    assert not out_dir.exists()
    assert not (tmp_path / ".benchmark.daina-panel-staging").exists()


def test_benchmark_manifest_records_and_validates_chembl_manifest_provenance(
    tmp_path: Path,
) -> None:
    chembl_fp = write_reference(tmp_path)
    panel = tmp_path / "panel.csv"
    pd.DataFrame(
        [{"case_id": "ethanol", "smiles": "CCO", "known_targets": "P11111"}]
    ).to_csv(panel, index=False)
    out_dir = tmp_path / "benchmark"

    result = run_benchmark(
        panel=panel,
        chembl_fp=chembl_fp,
        out_dir=out_dir,
        extra_args=["--allow-threshold-failure"],
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads((out_dir / "manifest.json").read_text())
    inputs = manifest["inputs"]
    assert inputs["source_manifest_sha256"] == sha256(tmp_path / "source_manifest.json")
    assert inputs["fingerprint_manifest_sha256"] == sha256(
        tmp_path / "fingerprint_manifest.json"
    )
    assert inputs["source_manifest_metadata"]["source"]["name"] == "ChEMBL"
    assert inputs["source_manifest_metadata"]["source"]["release"] == "37"
    assert inputs["fingerprint_manifest_metadata"]["source_snapshot"][
        "manifest_sha256"
    ] == inputs["source_manifest_sha256"]
    assert inputs["fingerprint_manifest_metadata"]["artifact"]["sha256"] == inputs[
        "chembl_fp_sha256"
    ]

    bad_fp = write_reference(tmp_path / "bad")
    bad_panel = tmp_path / "bad_panel.csv"
    pd.DataFrame(
        [{"case_id": "ethanol", "smiles": "CCO", "known_targets": "P11111"}]
    ).to_csv(bad_panel, index=False)
    fingerprint_manifest_path = bad_fp.parent / "fingerprint_manifest.json"
    bad_manifest = json.loads(fingerprint_manifest_path.read_text())
    bad_manifest["artifact"]["sha256"] = "0" * 64
    fingerprint_manifest_path.write_text(json.dumps(bad_manifest, indent=2) + "\n")

    bad_result = run_benchmark(
        panel=bad_panel,
        chembl_fp=bad_fp,
        out_dir=tmp_path / "bad_benchmark",
    )

    assert bad_result.returncode != 0
    assert "artifact.sha256 does not match current artifact" in bad_result.stderr
