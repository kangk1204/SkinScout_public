from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "eval/build_activity_recovery_panels.py"


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
        "target_cluster_30": "P11111",
        "target_cluster_50": "P11111",
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


def _dual_flags() -> dict[str, bool | str]:
    return {
        "evaluation_view": "dual_cold",
        "claimable": True,
        "absent_pair_from_train": True,
        "absent_pair_from_prior_splits": True,
        "absent_publication_from_train": True,
        "absent_publication_from_prior_splits": True,
        "absent_scaffold_from_train": True,
        "absent_scaffold_from_prior_splits": True,
        "absent_target_cluster_30_from_train": True,
        "absent_target_cluster_30_from_prior_splits": True,
        "absent_target_cluster_50_from_train": True,
        "absent_target_cluster_50_from_prior_splits": True,
    }


def _write_inputs(
    tmp_path: Path,
    *,
    dev_rows: list[dict[str, object]] | None = None,
    test_rows: list[dict[str, object]] | None = None,
    dual_rows: list[dict[str, object]] | None = None,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    dev_rows = dev_rows or [_row()]
    test_rows = test_rows or [
        _row(
            benchmark_id="b2",
            evidence_id="e2",
            evidence_date="2025-02-01",
            split="test",
            uniprot="P22222",
            ligand_smiles="c1ccccc1O",
            ligand_inchikey="ISWSIDIOOBJBQZ-UHFFFAOYSA-N",
            pactivity=6.5,
        )
    ]
    dual_rows = dual_rows or [
        _row(
            **_dual_flags(),
            benchmark_id="b3",
            evidence_id="e3",
            evidence_date="2025-03-01",
            split="test",
            uniprot="P33333",
            ligand_smiles="CCN",
            ligand_inchikey="QUSNBJAOOMFDIB-UHFFFAOYSA-N",
            pactivity=6.2,
        )
    ]
    dev = tmp_path / "dev.parquet"
    test = tmp_path / "test.parquet"
    dual = tmp_path / "dual_cold.parquet"
    train = tmp_path / "train.parquet"
    pd.DataFrame([_row(split="train", evidence_date="2023-06-01")]).to_parquet(
        train, index=False
    )
    pd.DataFrame(dev_rows).to_parquet(dev, index=False)
    pd.DataFrame(test_rows).to_parquet(test, index=False)
    pd.DataFrame(dual_rows).to_parquet(dual, index=False)
    benchmark_manifest = tmp_path / "benchmark_manifest.json"
    benchmark_manifest.write_text(
        json.dumps(
            {
                "schema_version": "activity_benchmark.v1",
                "output_sha256": {
                    "train.parquet": _sha256(train),
                    "dev.parquet": _sha256(dev),
                    "test.parquet": _sha256(test),
                    "dual_cold.parquet": _sha256(dual),
                },
                "splits": {
                    "counts": {
                        "train": 1,
                        "dev": len(dev_rows),
                        "test": len(test_rows),
                    }
                },
                "evaluation_views": {"counts": {"dual_cold": len(dual_rows)}},
            }
        )
        + "\n"
    )
    ligands = tmp_path / "ligands.parquet"
    edges = tmp_path / "edges.parquet"
    pd.DataFrame({"ligand_index": [0], "ligand_key": ["k"]}).to_parquet(ligands, index=False)
    pd.DataFrame({"ligand_index": [0], "uniprot": ["P11111"]}).to_parquet(edges, index=False)
    retrieval_manifest = tmp_path / "retrieval_manifest.json"
    retrieval_manifest.write_text(
        json.dumps(
            {
                "schema_version": "skinscout.activity-retrieval-index.v4",
                "inputs": {
                    "benchmark_manifest": {
                        "path": str(benchmark_manifest.resolve()),
                        "sha256": _sha256(benchmark_manifest),
                    },
                    "train_parquet": {
                        "path": str(train.resolve()),
                        "sha256": _sha256(train),
                        "rows": 1,
                    },
                },
                "outputs": {
                    "ligands": {
                        "path": str(ligands),
                        "sha256": _sha256(ligands),
                        "rows": 1,
                    },
                    "edges": {
                        "path": str(edges),
                        "sha256": _sha256(edges),
                        "rows": 1,
                    },
                },
            }
        )
        + "\n"
    )
    known_panel = tmp_path / "skin_known_target_panel.csv"
    shutil.copyfile(ROOT / "data/validation/skin_known_target_panel.csv", known_panel)
    return dev, test, dual, benchmark_manifest, retrieval_manifest, known_panel


def _write_rcsb_panel(
    tmp_path: Path,
    benchmark_manifest: Path,
    *,
    smiles: str = "CC(=O)Oc1ccccc1C(=O)O",
    target: str = "P44444",
) -> tuple[Path, Path, str]:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    standard_inchikey = Chem.MolToInchiKey(mol)
    ligand_key = (
        f"{standard_inchikey}#SMILES-"
        f"{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:12]}"
    )
    flags = {key: True for key in _dual_flags() if key != "evaluation_view"}
    ranking = tmp_path / "rcsb_ranking.parquet"
    pd.DataFrame(
        [
            {
                "query_id": ligand_key,
                "ligand_key": ligand_key,
                "standard_inchikey": standard_inchikey,
                "connectivity_key": standard_inchikey[:14],
                "canonical_smiles": canonical,
                "standardization_route": "fragment_parent_canonical_smiles_key",
                "truth_targets": [target],
                "n_truth_targets": 1,
                "split": "test",
                **flags,
                "source_publication_keys": ["doi:10.1000/rcsb-test"],
            }
        ]
    ).to_parquet(ranking, index=False)

    raw_snapshot = tmp_path / "rcsb_raw.jsonl.gz"
    raw_snapshot.write_bytes(b"synthetic-rcsb-snapshot")
    source_manifest = tmp_path / "rcsb_source_manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "schema_version": "skinscout.rcsb-holo-contact-snapshot.v1",
                "source_license": {
                    "name": "CC0",
                    "url": "https://www.rcsb.org/pages/policies",
                },
                "usage_contract": {
                    "positive_only_direct_contact_evaluation_source": True,
                    "never_training_or_calibration": True,
                },
                "query_sha256s": {
                    "search": "a" * 64,
                    "graphql": "b" * 64,
                },
                "raw_jsonl_gz": {
                    "path": str(raw_snapshot.resolve()),
                    "sha256": _sha256(raw_snapshot),
                    "rows": 1,
                    "bytes": raw_snapshot.stat().st_size,
                },
            }
        )
        + "\n"
    )
    target_clusters = tmp_path / "screenable_target_clusters.csv"
    target_clusters.write_text(
        "uniprot,target_cluster_30,target_cluster_50\nP44444,P44444,P44444\n"
    )
    target_cluster_manifest = tmp_path / "screenable_target_clusters.manifest.json"
    target_cluster_manifest.write_text(
        json.dumps({"schema_version": "synthetic-screenable-target-clusters.v1"}) + "\n"
    )
    benchmark = json.loads(benchmark_manifest.read_text())
    train = benchmark_manifest.parent / "train.parquet"
    dev = benchmark_manifest.parent / "dev.parquet"
    manifest = tmp_path / "rcsb_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "skinscout.rcsb-holo-direct-contact-panel.v1",
                "inputs": {
                    "raw_jsonl_gz": {
                        "path": str(raw_snapshot.resolve()),
                        "sha256": _sha256(raw_snapshot),
                        "rows": 1,
                    },
                    "source_manifest": {
                        "path": str(source_manifest.resolve()),
                        "sha256": _sha256(source_manifest),
                        "schema_version": "skinscout.rcsb-holo-contact-snapshot.v1",
                    },
                    "benchmark_manifest": {
                        "path": str(benchmark_manifest.resolve()),
                        "sha256": _sha256(benchmark_manifest),
                    },
                    "train_parquet": {
                        "path": str(train.resolve()),
                        "sha256": benchmark["output_sha256"]["train.parquet"],
                        "rows": 1,
                    },
                    "dev_parquet": {
                        "path": str(dev.resolve()),
                        "sha256": benchmark["output_sha256"]["dev.parquet"],
                        "rows": len(pd.read_parquet(dev)),
                    },
                    "screenable_target_clusters": {
                        "path": str(target_clusters.resolve()),
                        "sha256": _sha256(target_clusters),
                        "manifest_path": str(target_cluster_manifest.resolve()),
                        "manifest_sha256": _sha256(target_cluster_manifest),
                    },
                },
                "outputs": {
                    "ranking_queries": {
                        "path": str(ranking.resolve()),
                        "sha256": _sha256(ranking),
                        "rows": 1,
                    }
                },
                "contract": {
                    "positive_only": True,
                    "no_inferred_negatives": True,
                    "no_affinities_or_calibration": True,
                    "never_training_or_model_selection": True,
                    "ranking_split": "test",
                    "all_ranking_rows_dual_cold": True,
                    "pocket_leakage_audited": False,
                },
                "passes_adequacy": False,
            }
        )
        + "\n"
    )
    return ranking, manifest, ligand_key


def _run_builder(
    tmp_path: Path,
    dev: Path,
    test: Path,
    dual: Path,
    benchmark_manifest: Path,
    retrieval_manifest: Path,
    known_panel: Path,
    *extra: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dev-parquet",
            str(dev),
            "--test-parquet",
            str(test),
            "--dual-cold-parquet",
            str(dual),
            "--benchmark-manifest",
            str(benchmark_manifest),
            "--retrieval-index-manifest",
            str(retrieval_manifest),
            "--known-panel",
            str(known_panel),
            "--out-dir",
            str(tmp_path / "out"),
            "--dev-query-count",
            "2",
            "--test-query-count",
            "2",
            *extra,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_provenance_hash_row_count_manifest_and_stale_cleanup(tmp_path: Path) -> None:
    dev, test, dual, benchmark_manifest, retrieval_manifest, known_panel = _write_inputs(tmp_path)
    payload = json.loads(benchmark_manifest.read_text())
    payload["output_sha256"]["dev.parquet"] = "0" * 64
    benchmark_manifest.write_text(json.dumps(payload) + "\n")
    out = tmp_path / "out"
    out.mkdir()
    for name in ["dev_ranking_queries.parquet", "manifest.json", "known_panel.csv"]:
        (out / name).write_text("stale\n")

    res = _run_builder(tmp_path, dev, test, dual, benchmark_manifest, retrieval_manifest, known_panel)

    assert res.returncode != 0
    assert "sha256" in res.stderr
    assert not any(out.glob("*"))


def test_deterministic_order_independent_ranking_sampling(tmp_path: Path) -> None:
    dev_rows = [
        _row(benchmark_id="b1", evidence_id="e1", ligand_smiles="CCO", pactivity=6.1),
        _row(benchmark_id="b2", evidence_id="e2", ligand_smiles="CCN", uniprot="P22222", pactivity=6.2),
        _row(benchmark_id="b3", evidence_id="e3", ligand_smiles="CCC", uniprot="P33333", pactivity=4.5),
        _row(benchmark_id="b4", evidence_id="e4", ligand_smiles="CCCl", uniprot="P44444", pactivity=6.3),
    ]
    inputs = _write_inputs(tmp_path, dev_rows=dev_rows)
    first = _run_builder(tmp_path, *inputs)
    assert first.returncode == 0, first.stderr
    first_queries = pd.read_parquet(tmp_path / "out/dev_ranking_queries.parquet")
    first_hash = _sha256(tmp_path / "out/dev_ranking_queries.parquet")

    shuffled = list(reversed(dev_rows))
    inputs = _write_inputs(tmp_path / "second", dev_rows=shuffled)
    second = _run_builder(tmp_path / "second", *inputs)
    assert second.returncode == 0, second.stderr
    second_queries = pd.read_parquet(tmp_path / "second/out/dev_ranking_queries.parquet")

    assert first_queries.to_dict("records") == second_queries.to_dict("records")
    assert first_hash == _sha256(tmp_path / "second/out/dev_ranking_queries.parquet")
    assert list(first_queries.columns) == [
        "query_id",
        "ligand_key",
        "standard_inchikey",
        "connectivity_key",
        "canonical_smiles",
        "standardization_route",
        "truth_targets",
        "n_truth_targets",
        "split",
        "absent_pair_from_prior_splits",
        "absent_pair_from_train",
        "absent_publication_from_prior_splits",
        "absent_publication_from_train",
        "absent_scaffold_from_prior_splits",
        "absent_scaffold_from_train",
        "absent_target_cluster_30_from_prior_splits",
        "absent_target_cluster_30_from_train",
        "absent_target_cluster_50_from_prior_splits",
        "absent_target_cluster_50_from_train",
        "claimable",
    ]
    assert all(first_queries["n_truth_targets"] > 0)
    assert "CCC" not in set(first_queries["canonical_smiles"])


def test_dual_cold_adequacy_is_fail_closed_and_can_pass_only_explicit_criteria(
    tmp_path: Path,
) -> None:
    inputs = _write_inputs(tmp_path)
    result = _run_builder(tmp_path, *inputs)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((tmp_path / "out/manifest.json").read_text())
    adequacy = manifest["selection"]["ranking_queries"]["dual_cold"]["adequacy"]
    assert manifest["schema_version"] == "skinscout.activity-recovery-panels.v5"
    assert manifest["passes_panel_adequacy_gate"] is False
    assert adequacy["passes"] is False
    assert adequacy["observed"] == {
        "effective_target_count": 1.0,
        "max_truth_pair_target_fraction": 1.0,
        "queries": 1,
        "queries_by_panel_source": {"activity_quantitative": 1},
        "source_databases": ["chembl"],
        "source_documents": [
            {"publication_key": "pmid:1", "source_db": "chembl"}
        ],
        "truth_pairs": 1,
        "truth_pairs_by_panel_source": {"activity_quantitative": 1},
        "truth_pairs_by_target": {"P33333": 1},
        "unique_source_documents": 1,
        "unique_truth_targets": 1,
    }

    passing_inputs = _write_inputs(tmp_path / "passing")
    result = _run_builder(
        tmp_path / "passing",
        *passing_inputs,
        "--dual-cold-min-queries",
        "1",
        "--dual-cold-min-targets",
        "1",
        "--dual-cold-min-documents",
        "1",
        "--dual-cold-max-target-fraction",
        "1",
        "--dual-cold-min-effective-targets",
        "1",
    )
    assert result.returncode == 0, result.stderr
    passing = json.loads((tmp_path / "passing/out/manifest.json").read_text())
    assert passing["passes_panel_adequacy_gate"] is True
    assert all(
        passing["selection"]["ranking_queries"]["dual_cold"]["adequacy"][
            "checks"
        ].values()
    )


def test_rcsb_positive_only_panel_merges_into_dual_cold_but_not_calibration(
    tmp_path: Path,
) -> None:
    inputs = _write_inputs(tmp_path)
    rcsb_ranking, rcsb_manifest, rcsb_ligand_key = _write_rcsb_panel(
        tmp_path, inputs[3]
    )
    result = _run_builder(
        tmp_path,
        *inputs,
        "--rcsb-ranking-queries",
        str(rcsb_ranking),
        "--rcsb-panel-manifest",
        str(rcsb_manifest),
        "--dual-cold-min-queries",
        "2",
        "--dual-cold-min-targets",
        "2",
        "--dual-cold-min-documents",
        "2",
        "--dual-cold-max-target-fraction",
        "0.5",
        "--dual-cold-min-effective-targets",
        "2",
    )
    assert result.returncode == 0, result.stderr

    dual = pd.read_parquet(tmp_path / "out/dual_cold_ranking_queries.parquet")
    calibration = pd.read_parquet(tmp_path / "out/test_calibration_pairs.parquet")
    manifest = json.loads((tmp_path / "out/manifest.json").read_text())
    adequacy = manifest["selection"]["ranking_queries"]["dual_cold"]["adequacy"]

    assert set(dual["panel_sources"].explode()) == {
        "activity_quantitative",
        "rcsb_holo_direct_contact",
    }
    assert rcsb_ligand_key in set(dual["ligand_key"])
    assert rcsb_ligand_key not in set(calibration["ligand_key"])
    assert manifest["passes_panel_adequacy_gate"] is True
    assert adequacy["observed"]["queries"] == 2
    assert adequacy["observed"]["source_databases"] == ["chembl", "rcsb_pdb"]
    assert adequacy["observed"]["queries_by_panel_source"] == {
        "activity_quantitative": 1,
        "rcsb_holo_direct_contact": 1,
    }
    assert manifest["inputs"]["rcsb_holo_direct_contact_panel"]["sha256"] == _sha256(
        rcsb_ranking
    )


def test_rcsb_panel_contract_and_argument_pairing_fail_closed(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    rcsb_ranking, rcsb_manifest, _ = _write_rcsb_panel(tmp_path, inputs[3])

    missing_manifest = _run_builder(
        tmp_path,
        *inputs,
        "--rcsb-ranking-queries",
        str(rcsb_ranking),
    )
    assert missing_manifest.returncode != 0
    assert "must be supplied together" in missing_manifest.stderr

    payload = json.loads(rcsb_manifest.read_text())
    payload["contract"]["never_training_or_model_selection"] = False
    rcsb_manifest.write_text(json.dumps(payload) + "\n")
    invalid_contract = _run_builder(
        tmp_path,
        *inputs,
        "--rcsb-ranking-queries",
        str(rcsb_ranking),
        "--rcsb-panel-manifest",
        str(rcsb_manifest),
    )
    assert invalid_contract.returncode != 0
    assert "usage contract" in invalid_contract.stderr


def test_same_standard_inchikey_tautomers_keep_stable_distinct_query_ids(
    tmp_path: Path,
) -> None:
    dev_rows = [
        _row(
            benchmark_id="tautomer-a",
            evidence_id="tautomer-a",
            ligand_smiles="Brc1ccc(-c2nnc(-c3ccccc3)[nH]2)cc1",
            pactivity=6.3,
        ),
        _row(
            benchmark_id="tautomer-b",
            evidence_id="tautomer-b",
            ligand_smiles="Brc1ccc(-c2n[nH]c(-c3ccccc3)n2)cc1",
            uniprot="P22222",
            pactivity=6.4,
        ),
    ]
    inputs = _write_inputs(tmp_path, dev_rows=dev_rows)

    result = _run_builder(tmp_path, *inputs)

    assert result.returncode == 0, result.stderr
    ranking = pd.read_parquet(tmp_path / "out/dev_ranking_queries.parquet")
    assert len(ranking) == 2
    assert ranking["query_id"].is_unique
    assert set(ranking["standard_inchikey"]) == {"PLRURNUSAXSWDI-UHFFFAOYSA-N"}
    assert ranking["ligand_key"].str.startswith(
        "PLRURNUSAXSWDI-UHFFFAOYSA-N#SMILES-"
    ).all()
    assert ranking["ligand_key"].nunique() == 2
    assert set(ranking["standardization_route"]) == {
        "fragment_parent_canonical_smiles_key"
    }


def test_nonreproducible_source_structure_is_explicitly_excluded_without_bond_guessing(
    tmp_path: Path,
) -> None:
    malformed = "C:CC(=O)Nc1ccc2c(c1)c(=O)[nH]c1c(C(N)=O)c(-c3ccc(Oc4ccccc4)cc3)nn12"
    test_rows = [
        _row(
            benchmark_id="invalid-identity",
            evidence_id="invalid-identity",
            evidence_date="2025-02-01",
            split="test",
            ligand_smiles=malformed,
            ligand_inchikey="",
            pactivity=7.0,
        ),
        _row(
            benchmark_id="valid-identity",
            evidence_id="valid-identity",
            evidence_date="2025-02-02",
            split="test",
            ligand_smiles="CCO",
            pactivity=6.5,
        ),
    ]
    inputs = _write_inputs(tmp_path, test_rows=test_rows)

    result = _run_builder(tmp_path, *inputs)

    assert result.returncode == 0, result.stderr
    ranking = pd.read_parquet(tmp_path / "out/test_ranking_queries.parquet")
    assert ranking["canonical_smiles"].tolist() == ["CCO"]
    manifest = json.loads((tmp_path / "out/manifest.json").read_text())
    exclusion = manifest["selection"]["identity_exclusions"]["test"]
    assert exclusion["rows"] == 1
    assert exclusion["unique_ligand_smiles"] == 1
    assert exclusion["by_reason"] == {
        "unable_to_standardize_fragment_parent_identity": 1
    }
    assert exclusion["by_source_db"] == {"chembl": 1}
    assert exclusion["examples"] == [
        {
            "ligand_smiles_sha256": hashlib.sha256(malformed.encode()).hexdigest(),
            "source_db": "chembl",
            "reason": "unable_to_standardize_fragment_parent_identity",
        }
    ]


def test_thresholds_gray_rows_and_measured_only_calibration(tmp_path: Path) -> None:
    dev_rows = [
        _row(benchmark_id="pos", evidence_id="pos", ligand_smiles="CCO", pactivity=6.0),
        _row(benchmark_id="gray", evidence_id="gray", ligand_smiles="CCN", pactivity=5.5),
        _row(benchmark_id="neg", evidence_id="neg", ligand_smiles="CCC", pactivity=5.0, endpoint="KI"),
    ]
    inputs = _write_inputs(tmp_path, dev_rows=dev_rows)
    res = _run_builder(tmp_path, *inputs)
    assert res.returncode == 0, res.stderr

    calibration = pd.read_parquet(tmp_path / "out/dev_calibration_pairs.parquet")
    assert {tuple(value) for value in calibration["benchmark_id"]} == {("pos",), ("neg",)}
    assert set(calibration["sample_weight"]) == {1.0}
    assert {
        tuple(row.benchmark_id): int(row.label)
        for row in calibration.itertuples(index=False)
    } == {
        ("pos",): 1,
        ("neg",): 0,
    }
    assert set(calibration["endpoint_family"]) == {"functional", "direct_binding"}
    manifest = json.loads((tmp_path / "out/manifest.json").read_text())
    assert manifest["selection"]["calibration_pairs"]["dev"]["gray_excluded"] == 1
    assert "no unmeasured negatives" in manifest["algorithm"]["calibration_policy"]
    assert manifest["selection"]["calibration_pairs"]["cap"] == 1024


def test_default_calibration_preserves_measured_prevalence_and_optional_cap_weights(
    tmp_path: Path,
) -> None:
    dev_rows = [
        _row(benchmark_id="p1", evidence_id="p1", ligand_smiles="CCO", pactivity=6.0),
        _row(benchmark_id="p2", evidence_id="p2", ligand_smiles="CCN", pactivity=6.1),
        _row(benchmark_id="n1", evidence_id="n1", ligand_smiles="CCC", pactivity=5.0),
        _row(benchmark_id="n2", evidence_id="n2", ligand_smiles="CCCl", pactivity=4.9),
        _row(benchmark_id="n3", evidence_id="n3", ligand_smiles="CCBr", endpoint="KI", pactivity=4.8),
        _row(benchmark_id="n4", evidence_id="n4", ligand_smiles="CCI", endpoint="KD", pactivity=4.7),
    ]
    inputs = _write_inputs(tmp_path, dev_rows=dev_rows)
    res = _run_builder(tmp_path, *inputs)
    assert res.returncode == 0, res.stderr
    calibration = pd.read_parquet(tmp_path / "out/dev_calibration_pairs.parquet")
    assert len(calibration) == 6
    assert int((calibration["label"] == 1).sum()) == 2
    assert int((calibration["label"] == 0).sum()) == 4
    assert set(calibration["sample_weight"]) == {1.0}

    capped_inputs = _write_inputs(tmp_path / "capped", dev_rows=dev_rows)
    res = _run_builder(tmp_path / "capped", *capped_inputs, "--calibration-cap", "3")
    assert res.returncode == 0, res.stderr
    capped = pd.read_parquet(tmp_path / "capped/out/dev_calibration_pairs.parquet")
    assert len(capped) == 3
    assert int((capped["label"] == 1).sum()) == 1
    assert int((capped["label"] == 0).sum()) == 2
    assert set(capped.loc[capped["label"] == 1, "sample_weight"]) == {2.0}
    assert set(capped.loc[capped["label"] == 0, "sample_weight"]) == {2.0}
    manifest = json.loads((tmp_path / "capped/out/manifest.json").read_text())
    inclusion = manifest["selection"]["calibration_pairs"]["dev"]["inclusion_by_stratum"]
    assert inclusion["functional|label=1"] == {
        "endpoint_family": "functional",
        "label": 1,
        "population": 2,
        "sample": 1,
        "inclusion_rate": 0.5,
        "sample_weight": 2.0,
    }
    assert inclusion["functional|label=0"] == {
        "endpoint_family": "functional",
        "label": 0,
        "population": 2,
        "sample": 1,
        "inclusion_rate": 0.5,
        "sample_weight": 2.0,
    }
    assert inclusion["direct_binding|label=0"] == {
        "endpoint_family": "direct_binding",
        "label": 0,
        "population": 2,
        "sample": 1,
        "inclusion_rate": 0.5,
        "sample_weight": 2.0,
    }


def test_conflicting_duplicate_calibration_measurements_are_excluded_and_counted(
    tmp_path: Path,
) -> None:
    dev_rows = [
        _row(benchmark_id="pos", evidence_id="pos", ligand_smiles="CCO", uniprot="P11111", pactivity=6.2),
        _row(benchmark_id="neg", evidence_id="neg", ligand_smiles="CCO", uniprot="P11111", pactivity=4.9),
        _row(benchmark_id="ok", evidence_id="ok", ligand_smiles="CCN", uniprot="P22222", pactivity=6.3),
    ]
    inputs = _write_inputs(tmp_path, dev_rows=dev_rows)
    res = _run_builder(tmp_path, *inputs)
    assert res.returncode == 0, res.stderr

    calibration = pd.read_parquet(tmp_path / "out/dev_calibration_pairs.parquet")
    assert calibration["benchmark_id"].tolist() == [["ok"]]
    manifest = json.loads((tmp_path / "out/manifest.json").read_text())
    assert manifest["selection"]["calibration_pairs"]["dev"]["conflicts"] == {
        "conflicting_measurement_groups_excluded": 1,
        "conflicting_measurement_rows_excluded": 2,
    }


def test_split_date_and_dual_flag_validation(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path, dev_rows=[_row(split="test")])
    res = _run_builder(tmp_path, *inputs)
    assert res.returncode != 0
    assert "required split=dev" in res.stderr

    inputs = _write_inputs(tmp_path / "bad_date", test_rows=[_row(split="test", evidence_date="2024-12-31")])
    res = _run_builder(tmp_path / "bad_date", *inputs)
    assert res.returncode != 0
    assert "before 2025-01-01" in res.stderr

    bad_dual = _row(**_dual_flags(), split="test", evidence_date="2025-01-01")
    bad_dual["absent_scaffold_from_prior_splits"] = False
    inputs = _write_inputs(tmp_path / "bad_dual", dual_rows=[bad_dual])
    res = _run_builder(tmp_path / "bad_dual", *inputs)
    assert res.returncode != 0
    assert "absent_scaffold_from_prior_splits=False" in res.stderr


def test_known_panel_copied_unchanged_and_recorded(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    known_panel = inputs[-1]
    before = known_panel.read_bytes()
    res = _run_builder(tmp_path, *inputs)
    assert res.returncode == 0, res.stderr

    copied = tmp_path / "out/known_panel.csv"
    assert copied.read_bytes() == before
    manifest = json.loads((tmp_path / "out/manifest.json").read_text())
    assert manifest["inputs"]["known_panel.csv"]["sha256"] == _sha256(known_panel)
    assert manifest["inputs"]["known_panel.csv"]["rows"] == 22
    assert manifest["inputs"]["known_panel.csv"]["contract"]["target_pair_count"] == 46
    assert manifest["outputs"]["known_panel.csv"]["sha256"] == _sha256(known_panel)


def test_rejects_retrieval_index_built_from_a_different_benchmark(
    tmp_path: Path,
) -> None:
    inputs = _write_inputs(tmp_path)
    retrieval_manifest = inputs[-2]
    payload = json.loads(retrieval_manifest.read_text())
    payload["inputs"]["benchmark_manifest"]["sha256"] = "f" * 64
    retrieval_manifest.write_text(json.dumps(payload) + "\n")

    result = _run_builder(tmp_path, *inputs)

    assert result.returncode != 0
    assert "benchmark sha256 differs" in result.stderr
    assert not (tmp_path / "out/manifest.json").exists()


def test_rejects_any_changed_frozen_known_panel(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    known_panel = inputs[-1]
    known_panel.write_text(
        known_panel.read_text().replace("P10276;P10826;P13631", "P10276"),
        encoding="utf-8",
    )

    result = _run_builder(tmp_path, *inputs)

    assert result.returncode != 0
    assert "Frozen known-target panel sha256 mismatch" in result.stderr
    assert not (tmp_path / "out/manifest.json").exists()
