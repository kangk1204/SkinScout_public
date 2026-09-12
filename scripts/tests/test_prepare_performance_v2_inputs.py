"""Fixture tests for the performance-v2 production input adapter."""

from __future__ import annotations

import json
import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "prepare_performance_v2_inputs.py"


def load_adapter():
    spec = importlib.util.spec_from_file_location("prepare_performance_v2_inputs_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_adapter(
    tmp_path: Path,
    benchmark: Path,
    ligands: Path,
    fasta: Path,
    clusters: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    if clusters is None:
        clusters = write_target_clusters(tmp_path)
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--benchmark-train",
            str(benchmark),
            "--retrieval-ligands",
            str(ligands),
            "--target-fasta",
            str(fasta),
            "--target-clusters",
            str(clusters),
            "--out-train",
            str(tmp_path / "train.csv"),
            "--out-ligands",
            str(tmp_path / "ligands.csv"),
            "--out-targets",
            str(tmp_path / "targets.csv"),
            "--out-manifest",
            str(tmp_path / "manifest.json"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def write_ligands(tmp_path: Path) -> Path:
    path = tmp_path / "retrieval_ligands.parquet"
    pd.DataFrame(
        [
            {
                "ligand_index": 7,
                "ligand_key": "AAAAAAAABBBBBB-UHFFFAOYSA-N#SMILES-111111111111",
                "standard_inchikey": "AAAAAAAABBBBBB-UHFFFAOYSA-N",
                "connectivity_key": "AAAAAAAABBBBBB",
                "canonical_smiles": "CCO",
                "standardization_route": "fixture",
                "bitvec": ["0"],
            },
            {
                "ligand_index": 3,
                "ligand_key": "CCCCCCCCDDDDDD-UHFFFAOYSA-N#SMILES-222222222222",
                "standard_inchikey": "CCCCCCCCDDDDDD-UHFFFAOYSA-N",
                "connectivity_key": "CCCCCCCCDDDDDD",
                "canonical_smiles": "CCN",
                "standardization_route": "fixture",
                "bitvec": ["1"],
            },
        ]
    ).to_parquet(path, index=False)
    return path


def write_benchmark(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "benchmark_train.parquet"
    defaults = {
        "source_db": "ChEMBL",
        "source_release": "fixture",
        "source_license": "fixture",
        "evidence_id": "E",
        "source_document_id": "D",
        "publication_key": "P",
        "source_document_type": "Publication",
        "evidence_date": "2023-12-31",
        "split": "train",
        "target_cluster_30": "T",
        "target_cluster_50": "T",
        "ligand_id": "raw",
        "scaffold_smiles": "",
        "scaffold_id": "",
        "relation": "=",
        "standard_value_nm": 1.0,
        "binary_label": None,
    }
    normalized_rows = []
    for row in rows:
        merged = {**defaults, **row}
        if "activity_class" not in merged:
            merged["activity_class"] = (
                "weak_positive"
                if float(merged["pactivity"]) >= 5.0
                else "low_potency_quantitative"
            )
        normalized_rows.append(merged)
    pd.DataFrame(normalized_rows).to_parquet(path, index=False)
    return path


def write_fasta(tmp_path: Path, text: str = ">P1 fixture target\n acd\nEFG\n>P2\nMNP\n>P3\nRST\n") -> Path:
    path = tmp_path / "targets.fasta"
    path.write_text(text, encoding="utf-8")
    return path


def write_target_clusters(tmp_path: Path, rows: list[dict[str, str]] | None = None) -> Path:
    path = tmp_path / "target_clusters.csv"
    if rows is None:
        rows = [
            {"uniprot": "P1", "target_cluster_30": "C30A", "target_cluster_50": "C50A"},
            {"uniprot": "P2", "target_cluster_30": "C30B", "target_cluster_50": "C50B"},
            {"uniprot": "P3", "target_cluster_30": "C30C", "target_cluster_50": "C50C"},
        ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def read_outputs(tmp_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    return (
        pd.read_csv(tmp_path / "train.csv", keep_default_na=False),
        pd.read_csv(tmp_path / "ligands.csv", keep_default_na=False),
        pd.read_csv(tmp_path / "targets.csv", keep_default_na=False),
        json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8")),
    )


def test_cli_writes_expected_schema_and_manifest(tmp_path: Path) -> None:
    ligands = write_ligands(tmp_path)
    benchmark = write_benchmark(
        tmp_path,
        [
            {
                "uniprot": "P1",
                "ligand_smiles": "OCC",
                "ligand_inchikey": "AAAAAAAABBBBBB-UHFFFAOYSA-N",
                "endpoint": "KI",
                "pactivity": 6.2,
            },
            {
                "uniprot": "P2",
                "ligand_smiles": "CCN",
                "ligand_inchikey": "CCCCCCCCDDDDDD-UHFFFAOYSA-N",
                "endpoint": "IC50",
                "pactivity": 4.2,
            },
        ],
    )
    result = run_adapter(tmp_path, benchmark, ligands, write_fasta(tmp_path))
    assert result.returncode == 0, result.stderr
    train, ligand_csv, target_csv, manifest = read_outputs(tmp_path)
    assert train.columns.tolist() == ["ligand_id", "target_id", "evidence_state", "measured_label", "ontology"]
    assert ligand_csv.columns.tolist() == ["ligand_id", "smiles"]
    assert target_csv.columns.tolist() == ["target_id", "sequence", "target_cluster_30", "target_cluster_50"]
    assert train.to_dict("records") == [
        {
            "ligand_id": "ligand_000000000003",
            "target_id": "P2",
            "evidence_state": "measured_negative",
            "measured_label": 0,
            "ontology": "functional_modulation",
        },
        {
            "ligand_id": "ligand_000000000007",
            "target_id": "P1",
            "evidence_state": "measured_positive",
            "measured_label": 1,
            "ontology": "direct_binding_reversible",
        },
    ]
    assert target_csv.to_dict("records") == [
        {"target_id": "P1", "sequence": "ACDEFG", "target_cluster_30": "C30A", "target_cluster_50": "C50A"},
        {"target_id": "P2", "sequence": "MNP", "target_cluster_30": "C30B", "target_cluster_50": "C50B"},
        {"target_id": "P3", "sequence": "RST", "target_cluster_30": "C30C", "target_cluster_50": "C50C"},
    ]
    assert manifest["schema_version"] == "skinscout.performance-v2-inputs.v1"
    assert manifest["counts"]["measured_positive"] == 1
    assert manifest["counts"]["measured_negative"] == 1
    assert manifest["counts"]["direct_binding_reversible"] == 1
    assert manifest["counts"]["functional_modulation"] == 1
    assert manifest["counts"]["unknown_mixed"] == 0
    assert manifest["counts"]["targets"] == 3
    assert manifest["provenance"]["ligand_join"]["counts"] == {
        "exact_canonical_smiles": 1,
        "standardized_ligand_key": 0,
        "unique_standard_inchikey": 1,
    }
    assert len(manifest["provenance"]["ligand_join"]["row_binding_sha256"]) == 64
    assert manifest["artifacts"]["train"]["path"] == str((tmp_path / "train.csv").resolve())


def test_standardized_ligand_key_fallback_is_audited(tmp_path: Path) -> None:
    adapter = load_adapter()
    ligands = write_ligands(tmp_path)
    ligand_frame = pd.read_parquet(ligands)
    ligand_frame.loc[
        ligand_frame["canonical_smiles"].eq("CCN"),
        "ligand_key",
    ] = adapter._standardized_ligand_key("NCC")[0]
    ligand_frame.to_parquet(ligands, index=False)
    benchmark = write_benchmark(
        tmp_path,
        [
            {
                "uniprot": "P2",
                "ligand_smiles": "NCC",
                "ligand_inchikey": "",
                "endpoint": "KD",
                "pactivity": 6.2,
            }
        ],
    )

    result = run_adapter(tmp_path, benchmark, ligands, write_fasta(tmp_path))

    assert result.returncode == 0, result.stderr
    _, _, _, manifest = read_outputs(tmp_path)
    join = manifest["provenance"]["ligand_join"]
    assert join["counts"]["standardized_ligand_key"] == 1
    assert join["fallback_rows"] == 1


def test_gray_rows_are_ranking_only_not_negative(tmp_path: Path) -> None:
    ligands = write_ligands(tmp_path)
    benchmark = write_benchmark(
        tmp_path,
        [
            {
                "uniprot": "P1",
                "ligand_smiles": "CCO",
                "ligand_inchikey": "AAAAAAAABBBBBB-UHFFFAOYSA-N",
                "endpoint": "KD",
                "activity_class": "gray_unmeasured",
                "pactivity": 5.5,
            }
        ],
    )
    result = run_adapter(tmp_path, benchmark, ligands, write_fasta(tmp_path))
    assert result.returncode == 0, result.stderr
    train, _, _, manifest = read_outputs(tmp_path)
    assert train.loc[0, "evidence_state"] == "gray_unmeasured"
    assert train.loc[0, "measured_label"] == ""
    assert manifest["counts"]["gray_unmeasured"] == 1
    assert manifest["counts"]["measured_negative"] == 0


def test_numeric_thresholds_override_coarse_activity_class(tmp_path: Path) -> None:
    ligands = write_ligands(tmp_path)
    benchmark = write_benchmark(
        tmp_path,
        [
            {
                "uniprot": "P1",
                "ligand_smiles": "CCO",
                "ligand_inchikey": "AAAAAAAABBBBBB-UHFFFAOYSA-N",
                "endpoint": "KD",
                "activity_class": "weak_positive",
                "pactivity": 5.5,
            },
            {
                "uniprot": "P2",
                "ligand_smiles": "CCN",
                "ligand_inchikey": "CCCCCCCCDDDDDD-UHFFFAOYSA-N",
                "endpoint": "IC50",
                "activity_class": "weak_positive",
                "pactivity": 5.0,
            },
        ],
    )
    result = run_adapter(tmp_path, benchmark, ligands, write_fasta(tmp_path))
    assert result.returncode == 0, result.stderr
    train, _, _, manifest = read_outputs(tmp_path)
    by_target = train.set_index("target_id")
    assert by_target.loc["P1", "evidence_state"] == "gray_unmeasured"
    assert by_target.loc["P1", "measured_label"] == ""
    assert by_target.loc["P2", "evidence_state"] == "measured_negative"
    assert by_target.loc["P2", "measured_label"] == "0"
    assert by_target.loc["P2", "ontology"] == "functional_modulation"
    assert manifest["counts"]["gray_unmeasured"] == 1
    assert manifest["counts"]["measured_negative"] == 1


def test_conflicting_measured_labels_collapse_to_gray_with_manifest_count(tmp_path: Path) -> None:
    ligands = write_ligands(tmp_path)
    benchmark = write_benchmark(
        tmp_path,
        [
            {
                "uniprot": "P1",
                "ligand_smiles": "CCO",
                "ligand_inchikey": "AAAAAAAABBBBBB-UHFFFAOYSA-N",
                "endpoint": "KI",
                "pactivity": 6.2,
            },
            {
                "uniprot": "P1",
                "ligand_smiles": "CCO",
                "ligand_inchikey": "AAAAAAAABBBBBB-UHFFFAOYSA-N",
                "endpoint": "KI",
                "pactivity": 4.9,
            },
        ],
    )
    result = run_adapter(tmp_path, benchmark, ligands, write_fasta(tmp_path))
    assert result.returncode == 0, result.stderr
    train, _, _, manifest = read_outputs(tmp_path)
    assert train.to_dict("records") == [
        {
            "ligand_id": "ligand_000000000007",
            "target_id": "P1",
            "evidence_state": "gray_unmeasured",
            "measured_label": "",
            "ontology": "direct_binding_reversible",
        }
    ]
    assert manifest["counts"]["conflicting_measured_label_pairs_collapsed"] == 1
    assert manifest["counts"]["conflicting_measured_label_rows_collapsed"] == 2
    assert manifest["counts"]["measured_positive"] == 0
    assert manifest["counts"]["measured_negative"] == 0


def test_direct_and_functional_endpoints_remain_separate_supervision_rows(
    tmp_path: Path,
) -> None:
    ligands = write_ligands(tmp_path)
    benchmark = write_benchmark(
        tmp_path,
        [
            {
                "uniprot": "P1",
                "ligand_smiles": "CCO",
                "ligand_inchikey": "AAAAAAAABBBBBB-UHFFFAOYSA-N",
                "endpoint": "KI",
                "pactivity": 6.2,
            },
            {
                "uniprot": "P1",
                "ligand_smiles": "CCO",
                "ligand_inchikey": "AAAAAAAABBBBBB-UHFFFAOYSA-N",
                "endpoint": "IC50",
                "pactivity": 4.9,
            },
        ],
    )

    result = run_adapter(tmp_path, benchmark, ligands, write_fasta(tmp_path))

    assert result.returncode == 0, result.stderr
    train, _, _, _ = read_outputs(tmp_path)
    assert train[["ontology", "measured_label"]].to_dict("records") == [
        {"ontology": "direct_binding_reversible", "measured_label": 1},
        {"ontology": "functional_modulation", "measured_label": 0},
    ]


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"split": "test", "evidence_date": "2025-01-01"}, "split=train only"),
        ({"split": "train", "evidence_date": "2024-01-01"}, "preregistered cutoff"),
    ],
)
def test_training_adapter_rejects_nontrain_or_post_cutoff_rows(
    tmp_path: Path, override: dict[str, object], message: str
) -> None:
    ligands = write_ligands(tmp_path)
    benchmark = write_benchmark(
        tmp_path,
        [{
            "uniprot": "P1",
            "ligand_smiles": "CCO",
            "ligand_inchikey": "AAAAAAAABBBBBB-UHFFFAOYSA-N",
            "endpoint": "KI",
            "pactivity": 6.2,
            **override,
        }],
    )

    result = run_adapter(tmp_path, benchmark, ligands, write_fasta(tmp_path))

    assert result.returncode == 1
    assert message in result.stderr


def test_deterministic_ligand_ids_ignore_input_order(tmp_path: Path) -> None:
    ligands = write_ligands(tmp_path)
    fasta = write_fasta(tmp_path)
    rows = [
        {
            "uniprot": "P1",
            "ligand_smiles": "CCO",
            "ligand_inchikey": "AAAAAAAABBBBBB-UHFFFAOYSA-N",
            "endpoint": "KD",
            "pactivity": 6.3,
        },
        {
            "uniprot": "P2",
            "ligand_smiles": "CCN",
            "ligand_inchikey": "CCCCCCCCDDDDDD-UHFFFAOYSA-N",
            "endpoint": "KD",
            "pactivity": 4.3,
        },
    ]
    benchmark_a = write_benchmark(tmp_path, rows)
    out_a = tmp_path / "a"
    out_a.mkdir()
    result_a = run_adapter(out_a, benchmark_a, ligands, fasta)
    assert result_a.returncode == 0, result_a.stderr
    benchmark_b = write_benchmark(tmp_path, list(reversed(rows)))
    out_b = tmp_path / "b"
    out_b.mkdir()
    result_b = run_adapter(out_b, benchmark_b, ligands, fasta)
    assert result_b.returncode == 0, result_b.stderr
    assert (out_a / "train.csv").read_text(encoding="utf-8") == (out_b / "train.csv").read_text(encoding="utf-8")
    assert (out_a / "ligands.csv").read_text(encoding="utf-8") == (out_b / "ligands.csv").read_text(encoding="utf-8")


def test_missing_target_sequence_fails_before_outputs(tmp_path: Path) -> None:
    ligands = write_ligands(tmp_path)
    benchmark = write_benchmark(
        tmp_path,
        [
            {
                "uniprot": "P404",
                "ligand_smiles": "CCO",
                "ligand_inchikey": "AAAAAAAABBBBBB-UHFFFAOYSA-N",
                "endpoint": "KD",
                "pactivity": 6.2,
            }
        ],
    )
    result = run_adapter(tmp_path, benchmark, ligands, write_fasta(tmp_path))
    assert result.returncode == 1
    assert "missing from target FASTA" in result.stderr
    assert not (tmp_path / "train.csv").exists()


def test_missing_fasta_target_cluster_mapping_fails_closed(tmp_path: Path) -> None:
    ligands = write_ligands(tmp_path)
    benchmark = write_benchmark(
        tmp_path,
        [
            {
                "uniprot": "P1",
                "ligand_smiles": "CCO",
                "ligand_inchikey": "AAAAAAAABBBBBB-UHFFFAOYSA-N",
                "endpoint": "KD",
                "pactivity": 6.2,
            }
        ],
    )
    clusters = write_target_clusters(
        tmp_path,
        [{"uniprot": "P1", "target_cluster_30": "C30A", "target_cluster_50": "C50A"}],
    )
    result = run_adapter(tmp_path, benchmark, ligands, write_fasta(tmp_path), clusters)
    assert result.returncode == 1
    assert "missing from target clusters CSV" in result.stderr
    assert "P2" in result.stderr
    assert not (tmp_path / "targets.csv").exists()


def test_missing_required_schema_column_fails_closed(tmp_path: Path) -> None:
    ligands = write_ligands(tmp_path)
    benchmark = tmp_path / "bad_benchmark.parquet"
    pd.DataFrame(
        [
            {
                "uniprot": "P1",
                "ligand_smiles": "CCO",
                "ligand_inchikey": "AAAAAAAABBBBBB-UHFFFAOYSA-N",
                "endpoint": "KD",
                "pactivity": 6.2,
            }
        ]
    ).to_parquet(benchmark, index=False)
    result = run_adapter(tmp_path, benchmark, ligands, write_fasta(tmp_path))
    assert result.returncode == 1
    assert "missing required columns" in result.stderr
    assert "source_db" in result.stderr
    assert not (tmp_path / "train.csv").exists()


def test_unknown_activity_class_fails_closed(tmp_path: Path) -> None:
    ligands = write_ligands(tmp_path)
    benchmark = write_benchmark(
        tmp_path,
        [
            {
                "uniprot": "P1",
                "ligand_smiles": "CCO",
                "ligand_inchikey": "AAAAAAAABBBBBB-UHFFFAOYSA-N",
                "endpoint": "KD",
                "activity_class": "maybe_active",
                "pactivity": 6.2,
            }
        ],
    )
    result = run_adapter(tmp_path, benchmark, ligands, write_fasta(tmp_path))
    assert result.returncode == 1
    assert "unsupported activity_class" in result.stderr
    assert not (tmp_path / "train.csv").exists()


def test_fractional_ligand_index_fails_closed(tmp_path: Path) -> None:
    ligands = write_ligands(tmp_path)
    ligand_frame = pd.read_parquet(ligands)
    ligand_frame["ligand_index"] = ligand_frame["ligand_index"].astype(float)
    ligand_frame.loc[0, "ligand_index"] = 7.5
    ligand_frame.to_parquet(ligands, index=False)
    benchmark = write_benchmark(
        tmp_path,
        [
            {
                "uniprot": "P1",
                "ligand_smiles": "CCO",
                "ligand_inchikey": "AAAAAAAABBBBBB-UHFFFAOYSA-N",
                "endpoint": "KD",
                "pactivity": 6.2,
            }
        ],
    )

    result = run_adapter(tmp_path, benchmark, ligands, write_fasta(tmp_path))

    # Some pyarrow builds abort during interpreter teardown after the adapter has
    # already rejected the invalid table, so the portable contract is nonzero.
    assert result.returncode != 0
    assert "finite integer ligand_index" in result.stderr
    assert not (tmp_path / "train.csv").exists()


def test_late_atomic_write_failure_removes_temp_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = load_adapter()
    ligands = write_ligands(tmp_path)
    benchmark = write_benchmark(
        tmp_path,
        [
            {
                "uniprot": "P1",
                "ligand_smiles": "CCO",
                "ligand_inchikey": "AAAAAAAABBBBBB-UHFFFAOYSA-N",
                "endpoint": "KD",
                "pactivity": 6.2,
            }
        ],
    )

    def fail_manifest_write(*_args, **_kwargs):
        raise adapter.InputAdapterError("simulated manifest write failure")

    monkeypatch.setattr(adapter, "_write_json_temp", fail_manifest_write)
    with pytest.raises(adapter.InputAdapterError, match="simulated manifest write failure"):
        adapter.build_inputs(
            argparse.Namespace(
                benchmark_train=benchmark,
                retrieval_ligands=ligands,
                target_fasta=write_fasta(tmp_path),
                target_clusters=write_target_clusters(tmp_path),
                out_train=tmp_path / "train.csv",
                out_ligands=tmp_path / "ligands.csv",
                out_targets=tmp_path / "targets.csv",
                out_manifest=tmp_path / "manifest.json",
            )
        )
    for name in ("train.csv", "ligands.csv", "targets.csv", "manifest.json"):
        assert not (tmp_path / name).exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_mid_commit_failure_rolls_back_all_new_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = load_adapter()
    ligands = write_ligands(tmp_path)
    benchmark = write_benchmark(
        tmp_path,
        [
            {
                "uniprot": "P1",
                "ligand_smiles": "CCO",
                "ligand_inchikey": "AAAAAAAABBBBBB-UHFFFAOYSA-N",
                "endpoint": "KD",
                "pactivity": 6.2,
            }
        ],
    )
    outputs = {
        "out_train": tmp_path / "train.csv",
        "out_ligands": tmp_path / "ligands.csv",
        "out_targets": tmp_path / "targets.csv",
        "out_manifest": tmp_path / "manifest.json",
    }
    original_replace = adapter.Path.replace

    def fail_on_second_commit(path, target):
        if adapter.Path(target) == outputs["out_ligands"]:
            raise OSError("simulated second commit failure")
        return original_replace(path, target)

    monkeypatch.setattr(adapter.Path, "replace", fail_on_second_commit)
    with pytest.raises(OSError, match="simulated second commit failure"):
        adapter.build_inputs(
            argparse.Namespace(
                benchmark_train=benchmark,
                retrieval_ligands=ligands,
                target_fasta=write_fasta(tmp_path),
                target_clusters=write_target_clusters(tmp_path),
                **outputs,
            )
        )

    assert all(not path.exists() for path in outputs.values())
    assert not list(tmp_path.glob(".*.tmp"))
