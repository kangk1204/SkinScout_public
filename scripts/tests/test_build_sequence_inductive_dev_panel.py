from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "eval/build_sequence_inductive_dev_panel.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "benchmark_id": "b1",
        "source_db": "chembl",
        "source_release": "37",
        "source_license": "test",
        "evidence_id": "e1",
        "source_document_id": "d1",
        "publication_key": "pmid:1",
        "source_document_type": "Publication",
        "evidence_date": "2024-06-01",
        "split": "dev",
        "uniprot": "P11111",
        "target_cluster_30": "c30-1",
        "target_cluster_50": "c50-1",
        "ligand_id": "l1",
        "ligand_smiles": "CCO",
        "ligand_inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        "scaffold_smiles": "",
        "scaffold_id": "s1",
        "endpoint": "IC50",
        "relation": "=",
        "standard_value_nm": 1000.0,
        "pactivity": 6.0,
        "activity_class": "weak_positive",
        "binary_label": True,
    }
    row.update(overrides)
    return row


def _write_inputs(
    tmp_path: Path,
    *,
    train_rows: list[dict[str, object]] | None = None,
    dev_rows: list[dict[str, object]] | None = None,
) -> tuple[Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    train_rows = train_rows or [
        _row(
            benchmark_id="train-1",
            evidence_id="train-e1",
            evidence_date="2023-07-01",
            split="train",
            uniprot="P90000",
            target_cluster_30="hot30",
            target_cluster_50="hot50",
            ligand_smiles="CCCC",
            ligand_inchikey="IJDNQMDRQITEOD-UHFFFAOYSA-N",
            scaffold_id="train-scaffold",
            publication_key="pmid:train",
        )
    ]
    dev_rows = dev_rows or [
        _row(),
        _row(
            benchmark_id="b2",
            evidence_id="e2",
            uniprot="P22222",
            target_cluster_30="c30-2",
            target_cluster_50="c50-2",
            ligand_smiles="CCN",
            ligand_inchikey="QUSNBJAOOMFDIB-UHFFFAOYSA-N",
            scaffold_id="s2",
            publication_key="pmid:2",
            pactivity=6.5,
        ),
    ]
    dev_rows = list(dev_rows)
    if not any(row.get("benchmark_id") == "negative-1" for row in dev_rows):
        dev_rows.append(
            _row(
                benchmark_id="negative-1",
                evidence_id="negative-e1",
                uniprot="P33333",
                target_cluster_30="c30-3",
                target_cluster_50="c50-3",
                ligand_smiles="CC(=O)O",
                ligand_inchikey="QTBSBXVTEAMEQO-UHFFFAOYSA-N",
                scaffold_id="s3",
                publication_key="pmid:3",
                endpoint="KI",
                pactivity=4.5,
            )
        )
    train = tmp_path / "train.parquet"
    dev = tmp_path / "dev.parquet"
    pd.DataFrame(train_rows).to_parquet(train, index=False)
    pd.DataFrame(dev_rows).to_parquet(dev, index=False)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "activity_benchmark.v1",
                "output_sha256": {
                    "train.parquet": _sha256(train),
                    "dev.parquet": _sha256(dev),
                },
                "splits": {
                    "counts": {
                        "train": len(train_rows),
                        "dev": len(dev_rows),
                    }
                },
            }
        )
        + "\n"
    )
    return train, dev, manifest


def _run_builder(
    tmp_path: Path,
    train: Path,
    dev: Path,
    manifest: Path,
    *extra: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--train-parquet",
            str(train),
            "--dev-parquet",
            str(dev),
            "--benchmark-manifest",
            str(manifest),
            "--output-parquet",
            str(tmp_path / "out/ranking.parquet"),
            "--output-manifest",
            str(tmp_path / "out/manifest.json"),
            "--output-calibration",
            str(tmp_path / "out/calibration.parquet"),
            *extra,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_success_outputs_loadable_dev_cold_panel_and_manifest(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    result = _run_builder(tmp_path, *inputs)
    assert result.returncode == 0, result.stderr

    sys.path.insert(0, str(ROOT / "eval"))
    from activity_retrieval_model import load_calibration_panel, load_ranking_panel

    ranking = load_ranking_panel(
        tmp_path / "out/ranking.parquet",
        "dev",
        require_source_provenance=True,
    )
    calibration = load_calibration_panel(
        tmp_path / "out/calibration.parquet",
        "dev",
    )
    manifest = json.loads((tmp_path / "out/manifest.json").read_text())

    assert manifest["schema_version"] == "skinscout.sequence-inductive-dev-cold-panel.v1"
    assert manifest["contracts"] == {
        "all_claimable_true": True,
        "all_cold_flags_true": True,
        "dev_only_model_selection": True,
        "cold_calibration_full_population": True,
        "cold_calibration_measured_labels_only": True,
        "no_known_skin_input": True,
        "no_rcsb_input": True,
        "no_target_assistance_to_scorer": True,
        "no_test_input": True,
        "no_truth_assistance_artifact_input": True,
        "positive_truth_definition": "measured pActivity >= positive_threshold in dev.parquet",
        "source_provenance_required": True,
        "split": "dev",
        "truth_targets_collected_only_after_ligand_selection": True,
    }
    assert len(ranking) == 2
    assert set(calibration["label"]) == {0, 1}
    assert set(ranking["split"]) == {"dev"}
    cold_columns = [column for column in ranking.columns if column.startswith("absent_")]
    assert set(ranking[cold_columns + ["claimable"]].stack()) == {True}
    assert set(source for values in ranking["panel_sources"] for source in values) == {
        "activity_quantitative"
    }
    assert set(source for values in ranking["source_databases"] for source in values) == {
        "chembl"
    }
    assert manifest["selection"]["cap"] is None
    assert (
        manifest["selection"]["selected_ligands"]
        == manifest["selection"]["candidate_positive_ligands"]
    )


def test_future_dated_train_row_fails_closed(tmp_path: Path) -> None:
    future_train = [
        _row(
            benchmark_id="train-1",
            evidence_id="train-e1",
            evidence_date="2024-01-01",
            split="train",
            uniprot="P90000",
            target_cluster_30="hot30",
            target_cluster_50="hot50",
            ligand_smiles="CCCC",
            ligand_inchikey="IJDNQMDRQITEOD-UHFFFAOYSA-N",
            scaffold_id="train-scaffold",
            publication_key="pmid:train",
        )
    ]
    inputs = _write_inputs(tmp_path, train_rows=future_train)

    result = _run_builder(tmp_path, *inputs)

    assert result.returncode != 0
    assert "train evidence_date on or after 2024-01-01" in result.stderr
    assert not (tmp_path / "out/ranking.parquet").exists()
    assert not (tmp_path / "out/manifest.json").exists()


def test_no_cold_candidates_fails_closed_and_cleans_stale_outputs(tmp_path: Path) -> None:
    train = [
        _row(
            benchmark_id="train-1",
            evidence_id="train-e1",
            evidence_date="2023-07-01",
            split="train",
            uniprot="P11111",
            target_cluster_30="c30-1",
            target_cluster_50="c50-1",
            scaffold_id="s1",
            publication_key="pmid:1",
        )
    ]
    dev = [_row()]
    inputs = _write_inputs(tmp_path, train_rows=train, dev_rows=dev)
    out = tmp_path / "out"
    out.mkdir()
    for name in (
        "ranking.parquet",
        "calibration.parquet",
        "manifest.json",
        "ranking.parquet.tmp",
    ):
        (out / name).write_text("stale\n")

    result = _run_builder(tmp_path, *inputs)

    assert result.returncode != 0
    assert "empty" in result.stderr
    assert not any(out.glob("*"))


def test_stale_manifest_hash_fails_closed(tmp_path: Path) -> None:
    train, dev, manifest = _write_inputs(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["output_sha256"]["dev.parquet"] = "0" * 64
    manifest.write_text(json.dumps(payload) + "\n")

    result = _run_builder(tmp_path, train, dev, manifest)

    assert result.returncode != 0
    assert "sha256" in result.stderr
    assert not (tmp_path / "out/ranking.parquet").exists()
    assert not (tmp_path / "out/calibration.parquet").exists()
    assert not (tmp_path / "out/manifest.json").exists()


def test_changed_train_set_changes_exclusion(tmp_path: Path) -> None:
    dev_rows = [
        _row(ligand_smiles="CCO", scaffold_id="s1", pactivity=6.1),
        _row(
            benchmark_id="b2",
            evidence_id="e2",
            ligand_smiles="CCN",
            ligand_inchikey="QUSNBJAOOMFDIB-UHFFFAOYSA-N",
            uniprot="P22222",
            target_cluster_30="c30-2",
            target_cluster_50="c50-2",
            scaffold_id="s2",
            publication_key="pmid:2",
            pactivity=6.2,
        ),
    ]
    base = _write_inputs(tmp_path / "base", dev_rows=dev_rows)
    result = _run_builder(tmp_path / "base", *base)
    assert result.returncode == 0, result.stderr
    base_panel = pd.read_parquet(tmp_path / "base/out/ranking.parquet")

    excluding_train = [
        _row(
            benchmark_id="train-1",
            evidence_id="train-e1",
            evidence_date="2023-07-01",
            split="train",
            uniprot="P90000",
            target_cluster_30="other30",
            target_cluster_50="other50",
            ligand_smiles="CCCC",
            ligand_inchikey="IJDNQMDRQITEOD-UHFFFAOYSA-N",
            scaffold_id="s1",
            publication_key="pmid:train",
        )
    ]
    changed = _write_inputs(
        tmp_path / "changed", train_rows=excluding_train, dev_rows=dev_rows
    )
    result = _run_builder(tmp_path / "changed", *changed)
    assert result.returncode == 0, result.stderr
    changed_panel = pd.read_parquet(tmp_path / "changed/out/ranking.parquet")

    assert len(base_panel) == 2
    assert len(changed_panel) == 1
    assert set(changed_panel["canonical_smiles"]) == {"CCN"}


def test_deterministic_cap_is_order_independent(tmp_path: Path) -> None:
    rows = [
        _row(benchmark_id=f"b{i}", evidence_id=f"e{i}", ligand_smiles=smiles, pactivity=6.0 + i / 10)
        for i, smiles in enumerate(["CCO", "CCN", "CCC", "CCCl"], start=1)
    ]
    first = _write_inputs(tmp_path / "first", dev_rows=rows)
    result = _run_builder(tmp_path / "first", *first, "--cap", "2")
    assert result.returncode == 0, result.stderr
    first_panel = pd.read_parquet(tmp_path / "first/out/ranking.parquet")
    first_hash = _sha256(tmp_path / "first/out/ranking.parquet")

    second = _write_inputs(tmp_path / "second", dev_rows=list(reversed(rows)))
    result = _run_builder(tmp_path / "second", *second, "--cap", "2")
    assert result.returncode == 0, result.stderr
    second_panel = pd.read_parquet(tmp_path / "second/out/ranking.parquet")

    assert first_panel.to_dict("records") == second_panel.to_dict("records")
    assert first_hash == _sha256(tmp_path / "second/out/ranking.parquet")
    assert len(first_panel) == 2


def test_failure_removes_temp_outputs(tmp_path: Path) -> None:
    train, dev, manifest = _write_inputs(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "ranking.parquet").write_text("stale\n")
    (out / "ranking.parquet.tmp").write_text("stale tmp\n")
    (out / "manifest.json").write_text("stale\n")
    (out / "manifest.json.tmp").write_text("stale tmp\n")

    result = _run_builder(tmp_path, train, dev, manifest, "--cap", "0")

    assert result.returncode != 0
    assert not any(out.glob("*"))


def test_rejects_output_alias_without_modifying_input(tmp_path: Path) -> None:
    train, dev, manifest = _write_inputs(tmp_path)
    before = dev.read_bytes()
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--train-parquet",
            str(train),
            "--dev-parquet",
            str(dev),
            "--benchmark-manifest",
            str(manifest),
            "--output-parquet",
            str(dev),
            "--output-calibration",
            str(tmp_path / "out/calibration.parquet"),
            "--output-manifest",
            str(tmp_path / "out/manifest.json"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "aliases a protected input" in result.stderr
    assert dev.read_bytes() == before
    assert not (tmp_path / "out").exists()
