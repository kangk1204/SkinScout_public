from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_activity_retrieval_gate import (  # noqa: E402
    check_gate,
    check_operational_gate,
    create_gate,
    create_operational_gate,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")


def _contract(tmp_path: Path) -> dict[str, Path]:
    exclusions = tmp_path / "source_target_exclusions.csv"
    exclusions.write_text(
        "schema_version,source_db,source_release,source_document_key,uniprot,"
        "reason_code,source_document_url,source_document_target,database_target,"
        "reviewed_at_utc,evidence_note\n"
        "skinscout.activity-source-target-exclusions.v1,BindingDB,2026-08,"
        "patent:US20250257059,Q92932,source_document_target_mismatch,"
        "https://patents.google.com/patent/US20250257059A1/en,PTPN2,"
        "PTPRN2 (UniProt Q92932),2026-08-05T17:03:36Z,test evidence\n"
    )
    benchmark = tmp_path / "benchmark.json"
    _write_json(
        benchmark,
        {
            "schema_version": "activity_benchmark.v1",
            "versions": {"rdkit": "2025.03.6"},
            "input_sha256": {"source_target_exclusions": _sha256(exclusions)},
            "filters": {
                "source_target_exclusions": {
                    "applied": True,
                    "schema_version": "skinscout.activity-source-target-exclusions.v1",
                    "artifact": {
                        "path": str(exclusions.resolve()),
                        "sha256": _sha256(exclusions),
                        "rows": 1,
                    },
                    "rule_count": 1,
                    "all_rules_matched": True,
                    "total_filtered_rows": 2,
                    "rules": [{"matched_rows": 2}],
                }
            },
            "splits": {
                "ligand_identity_policy": {
                    "basis": (
                        "RDKit canonical isomeric SMILES with explicit hydrogens removed"
                    ),
                    "uses_source_ligand_ids": False,
                    "uses_optional_source_inchikey": False,
                    "applies_to": [
                        "benchmark_id",
                        "deduplication",
                        "first_observation",
                        "leakage_audit",
                        "cold_flags",
                    ],
                }
            },
        },
    )

    index = tmp_path / "index.json"
    _write_json(
        index,
        {
            "schema_version": "skinscout.activity-retrieval-index.v4",
            "inputs": {"benchmark_manifest": _record(benchmark)},
        },
    )

    rcsb_ranking = tmp_path / "rcsb_ranking.parquet"
    rcsb_ranking.write_text("frozen RCSB ranking fixture\n")
    rcsb_pairs = tmp_path / "rcsb_pairs.parquet"
    rcsb_pairs.write_text("frozen RCSB pair fixture\n")
    rcsb_manifest = tmp_path / "rcsb_panel.json"
    _write_json(
        rcsb_manifest,
        {
            "schema_version": "skinscout.rcsb-holo-direct-contact-panel.v1",
            "outputs": {
                "pairs": {**_record(rcsb_pairs), "rows": 2},
            },
        },
    )
    known = tmp_path / "known.csv"
    known.write_text("case_id,target\nretinol,P10276\n")
    dual = tmp_path / "dual.parquet"
    dual.write_text("frozen dual-cold panel fixture\n")
    dual_record = {**_record(dual), "rows": 100}
    panels = tmp_path / "panels.json"
    _write_json(
        panels,
        {
            "schema_version": "skinscout.activity-recovery-panels.v5",
            "passes_panel_adequacy_gate": True,
            "inputs": {
                "benchmark_manifest": _record(benchmark),
                "retrieval_index_manifest": _record(index),
                "rcsb_holo_direct_contact_panel": {
                    **_record(rcsb_ranking),
                    "rows": 100,
                    "manifest": {
                        **_record(rcsb_manifest),
                        "schema_version": (
                            "skinscout.rcsb-holo-direct-contact-panel.v1"
                        ),
                    },
                    "contract": {
                        "positive_only": True,
                        "no_affinities_or_calibration": True,
                        "never_training_or_model_selection": True,
                    },
                },
            },
            "outputs": {
                "known_panel.csv": _record(known),
                "dual_cold_ranking_queries.parquet": dual_record,
            },
            "selection": {
                "ranking_queries": {
                    "dual_cold": {
                        "adequacy": {
                            "passes": True,
                            "criteria": {
                                "min_queries": 100,
                                "min_unique_truth_targets": 20,
                                "min_unique_source_documents": 20,
                                "max_truth_pair_target_fraction": 0.20,
                                "min_effective_target_count": 10.0,
                            },
                            "observed": {
                                "queries": 100,
                                "truth_pairs": 100,
                                "unique_truth_targets": 20,
                                "unique_source_documents": 20,
                                "source_databases": ["chembl"],
                                "source_documents": [
                                    {
                                        "source_db": "chembl",
                                        "publication_key": f"doi:10.1/{index}",
                                    }
                                    for index in range(20)
                                ],
                                "max_truth_pair_target_fraction": 0.05,
                                "effective_target_count": 20.0,
                                "truth_pairs_by_target": {
                                    f"P{index:05d}": 5 for index in range(20)
                                },
                            },
                            "checks": {
                                "min_queries": True,
                                "min_unique_truth_targets": True,
                                "min_unique_source_documents": True,
                                "max_truth_pair_target_fraction": True,
                                "min_effective_target_count": True,
                            },
                        }
                    }
                }
            },
        },
    )

    target_manifest = tmp_path / "targets.json"
    _write_json(target_manifest, {"schema_version": "test-targets.v1"})
    recipe = tmp_path / "recipe.json"
    # Mirrors asdict(BASELINE); the gate compares it field-for-field, so a new
    # Recipe field has to appear here as well.
    baseline_recipe = {
        "recipe_id": "chembl_p5_max",
        "max_union5": 0.0,
        "max_union6": 0.0,
        "quality_union6": 0.0,
        "source_consensus6": 0.0,
        "support_union6": 0.0,
        "negative_contrast": 0.0,
        "max_union_any": 0.0,
        "max_union_nonneg": 0.0,
    }
    selected_recipe = {
        **baseline_recipe,
        "recipe_id": "union_p6_consensus",
        "max_union6": 0.68,
        "quality_union6": 0.17,
        "source_consensus6": 0.10,
        "support_union6": 0.05,
    }
    _write_json(
        recipe,
        {
            "schema_version": "skinscout.activity-retrieval-recipe.v1",
            "passes_dev_gate": True,
            "selected_recipe": selected_recipe,
            "baseline_recipe": baseline_recipe,
        },
    )
    selection = tmp_path / "selection.json"
    _write_json(
        selection,
        {
            "schema_version": "skinscout.activity-retrieval-selection.v1",
            "passes_dev_gate": True,
            "inputs": {
                "index_manifest": _record(index),
                "panels_manifest": _record(panels),
                "target_cluster_manifest": _record(target_manifest),
            },
            "outputs": {"recipe.json": _record(recipe)},
        },
    )

    summary = tmp_path / "summary.json"
    decision = {
        "evaluation_completed": True,
        "claim_ready": True,
        "promotion_decision": "promote_selected",
        "operational_recipe_id": "union_p6_consensus",
        "selected_recipe_id": "union_p6_consensus",
        "baseline_recipe_id": "chembl_p5_max",
    }
    dual_cold_gate = {
        "claimable": True,
        "passes_panel_adequacy_gate": True,
        "passes_truth_universe_coverage": True,
        "coverage": {
            "measured": True,
            "truth_pairs": 100,
            "truth_pairs_in_scorable_universe": 100,
            "truth_pairs_outside_scorable_universe": 0,
            "truth_pair_coverage": 1.0,
            "all_truth": {"top10": 0.4, "top30": 0.7, "mrr": 0.3},
            "in_universe": {"top10": 0.4, "top30": 0.7, "mrr": 0.3},
        },
        "policy": "test fixture",
    }
    _write_json(
        summary,
        {
            "schema_version": "skinscout.activity-retrieval-evaluation.v1",
            "passes_frozen_test_gate": True,
            **decision,
            "gates": {"dual_cold": dual_cold_gate},
        },
    )
    final = tmp_path / "final.json"
    _write_json(
        final,
        {
            "schema_version": "skinscout.activity-retrieval-evaluation.v1",
            "passes_frozen_test_gate": True,
            **decision,
            "gates": {"dual_cold": dual_cold_gate},
            "inputs": {
                "index_manifest": _record(index),
                "panels_manifest": _record(panels),
                "panels_schema": "skinscout.activity-recovery-panels.v5",
                "recipe": _record(recipe),
                "target_cluster_manifest": _record(target_manifest),
                "known_panel": _record(known),
            },
            "outputs": {"summary.json": _record(summary)},
        },
    )

    contact_fragments = tmp_path / "rcsb_contact_fragments"
    contact_fragments.mkdir()
    (contact_fragments / "pair-a.pdb").write_text("ATOM pair a\n")
    contact_index = tmp_path / "rcsb_contact_index.csv"
    contact_index.write_text(
        "pair_id,entry_id,component_id,instance_id,uniprot,ligand_key\n"
        "pair-a,1ABC,LIG,1ABC.A,P00001,ligand-a\n"
    )
    contact_exclusions = tmp_path / "rcsb_contact_exclusions.csv"
    contact_exclusions.write_text(
        "pair_id,entry_id,component_id,instance_id,uniprot,ligand_key,reason,detail\n"
        "pair-b,2ABC,LIG,2ABC.A,P00002,ligand-b,download_failed,test\n"
    )
    contact_manifest = tmp_path / "rcsb_contact_fragments.json"
    _write_json(
        contact_manifest,
        {
            "schema_version": "skinscout.rcsb-contact-pocket-fragments.v1",
            "contract": {
                "positive_only": True,
                "no_inferred_negatives": True,
                "never_prediction": True,
                "never_training": True,
                "never_calibration": True,
                "never_model_selection": True,
                "strict_pairs_only": True,
            },
            "inputs": {
                "panel_manifest": {
                    **_record(rcsb_manifest),
                    "schema_version": "skinscout.rcsb-holo-direct-contact-panel.v1",
                },
                "pairs_parquet": {
                    **_record(rcsb_pairs),
                    "rows": 2,
                    "strict_dual_cold_rows": 2,
                },
            },
            "counts": {
                "strict_dual_cold_pairs": 2,
                "indexed": 1,
                "excluded": 1,
            },
            "artifacts": {
                "fragment_dir": {
                    "path": str(contact_fragments.resolve()),
                    "tree_sha256": _tree_sha256(contact_fragments),
                    "files": 1,
                },
                "index_csv": {**_record(contact_index), "rows": 1},
                "exclusions_csv": {**_record(contact_exclusions), "rows": 1},
            },
        },
    )

    prior_fragments = tmp_path / "prior_fragments"
    (prior_fragments / "fragments").mkdir(parents=True)
    (prior_fragments / "fragments" / "P00003.pdb").write_text("ATOM prior\n")
    prior_index = tmp_path / "prior_index.csv"
    prior_index.write_text(
        "uniprot,fragment_path,fragment_sha256\n"
        f"P00003,fragments/P00003.pdb,{_sha256(prior_fragments / 'fragments' / 'P00003.pdb')}\n"
    )
    prior_manifest = tmp_path / "prior_fragments.json"
    _write_json(
        prior_manifest,
        {
            "schema_version": "skinscout.pocket-fragment-universe.v1",
            "artifacts": {
                "out_dir": str(prior_fragments.resolve()),
                "out_dir_tree_sha256": _tree_sha256(prior_fragments),
                "index": {**_record(prior_index), "rows": 1},
            },
        },
    )

    leakage_rows = tmp_path / "pocket_leakage_rows.csv"
    leakage_rows.write_text(
        "pair_id,ligand_key,fragment_extracted,foldseek_covered,"
        "leakage_tm40,leakage_tm50,leakage_tm60\n"
        "pair-a,ligand-a,true,true,true,false,false\n"
        "pair-b,ligand-b,false,false,false,false,false\n"
    )
    pocket_leakage = tmp_path / "pocket_leakage.json"
    _write_json(
        pocket_leakage,
        {
            "schema_version": "skinscout.rcsb-contact-pocket-leakage-audit.v1",
            "contract": {
                "evaluation_only": True,
                "never_training": True,
                "never_calibration": True,
                "never_model_selection": True,
                "missing_foldseek_hits_are_uncovered_not_cold": True,
                "fragment_extraction_failures_are_uncovered_not_cold": True,
            },
            "thresholds": {
                "max_directional_tm": [0.4, 0.5, 0.6],
                "columns": {
                    "0.4": "leakage_tm40",
                    "0.5": "leakage_tm50",
                    "0.6": "leakage_tm60",
                },
            },
            "inputs": {
                "activity_benchmark_manifest": {
                    **_record(benchmark),
                    "schema_version": "activity_benchmark.v1",
                },
                "rcsb_contact_fragments": {
                    "index": {**_record(contact_index), "rows": 1},
                    "exclusions": {**_record(contact_exclusions), "rows": 1},
                    "manifest": {
                        **_record(contact_manifest),
                        "schema_version": "skinscout.rcsb-contact-pocket-fragments.v1",
                    },
                    "fragment_dir": {
                        "path": str(contact_fragments.resolve()),
                        "tree_sha256": _tree_sha256(contact_fragments),
                        "files": 1,
                    },
                },
                "prior_p2rank_fragments": {
                    "index": {**_record(prior_index), "rows": 1},
                    "manifest": {
                        **_record(prior_manifest),
                        "schema_version": "skinscout.pocket-fragment-universe.v1",
                    },
                    "fragment_dir": {
                        "path": str(prior_fragments.resolve()),
                        "tree_sha256": _tree_sha256(prior_fragments),
                        "files": 1,
                    },
                },
            },
            "summary": {
                "pairs": {
                    "total": 2,
                    "fragment_extracted": 1,
                    "fragment_extraction_excluded": 1,
                    "covered": 1,
                    "uncovered": 1,
                    "coverage_rate": 0.5,
                    "leakage_by_max_directional_tm_threshold": {
                        "0.4": {
                            "count": 1,
                            "rate_all_pairs": 0.5,
                            "rate_covered_pairs": 1.0,
                        },
                        "0.5": {
                            "count": 0,
                            "rate_all_pairs": 0.0,
                            "rate_covered_pairs": 0.0,
                        },
                        "0.6": {
                            "count": 0,
                            "rate_all_pairs": 0.0,
                            "rate_covered_pairs": 0.0,
                        },
                    },
                },
                "ligand_queries": {
                    "total": 2,
                    "covered": 1,
                    "leakage_by_max_directional_tm_threshold": {
                        "0.4": {"leaked_ligand_queries": 1, "leakage_rate": 0.5},
                        "0.5": {"leaked_ligand_queries": 0, "leakage_rate": 0.0},
                        "0.6": {"leaked_ligand_queries": 0, "leakage_rate": 0.0},
                    },
                },
            },
            "outputs": {
                "detailed_csv": {**_record(leakage_rows), "rows": 2},
                "manifest": {"path": str(pocket_leakage.resolve())},
            },
        },
    )
    return {
        "benchmark_manifest": benchmark,
        "index_manifest": index,
        "panels_manifest": panels,
        "selection_manifest": selection,
        "final_evaluation_manifest": final,
        "pocket_leakage_manifest": pocket_leakage,
    }


def _rewrite_benchmark_contract(
    contract: dict[str, Path], mutation: Callable[[dict[str, object]], None]
) -> None:
    benchmark_path = contract["benchmark_manifest"]
    benchmark = json.loads(benchmark_path.read_text())
    mutation(benchmark)
    _write_json(benchmark_path, benchmark)

    index_path = contract["index_manifest"]
    index = json.loads(index_path.read_text())
    index["inputs"]["benchmark_manifest"] = _record(benchmark_path)
    _write_json(index_path, index)

    panels_path = contract["panels_manifest"]
    panels = json.loads(panels_path.read_text())
    panels["inputs"]["benchmark_manifest"] = _record(benchmark_path)
    panels["inputs"]["retrieval_index_manifest"] = _record(index_path)
    _write_json(panels_path, panels)

    selection_path = contract["selection_manifest"]
    selection = json.loads(selection_path.read_text())
    selection["inputs"]["index_manifest"] = _record(index_path)
    selection["inputs"]["panels_manifest"] = _record(panels_path)
    _write_json(selection_path, selection)

    final_path = contract["final_evaluation_manifest"]
    final = json.loads(final_path.read_text())
    final["inputs"]["index_manifest"] = _record(index_path)
    final["inputs"]["panels_manifest"] = _record(panels_path)
    _write_json(final_path, final)

    leakage_path = contract["pocket_leakage_manifest"]
    leakage = json.loads(leakage_path.read_text())
    leakage["inputs"]["activity_benchmark_manifest"] = {
        **_record(benchmark_path),
        "schema_version": "activity_benchmark.v1",
    }
    _write_json(leakage_path, leakage)


def _refresh_gate_manifest_records(gate: Path, contract: dict[str, Path]) -> None:
    payload = json.loads(gate.read_text())
    payload["manifests"] = {
        key: {**_record(path), "bytes": path.stat().st_size}
        for key, path in contract.items()
    }
    _write_json(gate, payload)


def _set_failed_promotion(contract: dict[str, Path]) -> None:
    summary_path = Path(
        json.loads(contract["final_evaluation_manifest"].read_text())["outputs"]
        ["summary.json"]["path"]
    )
    summary = json.loads(summary_path.read_text())
    summary.update(
        {
            "passes_frozen_test_gate": False,
            "claim_ready": False,
            "promotion_decision": "retain_frozen_baseline",
            "operational_recipe_id": "chembl_p5_max",
        }
    )
    _write_json(summary_path, summary)
    final = json.loads(contract["final_evaluation_manifest"].read_text())
    final.update(
        {
            "passes_frozen_test_gate": False,
            "claim_ready": False,
            "promotion_decision": "retain_frozen_baseline",
            "operational_recipe_id": "chembl_p5_max",
        }
    )
    final["outputs"]["summary.json"] = _record(summary_path)
    _write_json(contract["final_evaluation_manifest"], final)


def _operational_args(contract: dict[str, Path], tmp_path: Path) -> dict[str, Path]:
    recipe = Path(
        json.loads(contract["selection_manifest"].read_text())["outputs"]["recipe.json"]["path"]
    )
    ligands = tmp_path / "runtime_ligands.parquet"
    edges = tmp_path / "runtime_edges.parquet"
    targets = tmp_path / "runtime_targets.csv"
    source_manifest = tmp_path / "runtime_source_manifest.json"
    ligands.write_text("runtime ligand fixture\n")
    edges.write_text("runtime edge fixture\n")
    targets.write_text("uniprot\nP00001\n")
    _write_json(source_manifest, {"sources": ["ChEMBL", "BindingDB"]})
    manifest = tmp_path / "runtime_index.json"
    _write_json(
        manifest,
        {
            "schema_version": "skinscout.activity-retrieval-index.production.v1",
            "index_role": "production",
            "algorithm": {
                "standardization": [
                    "rdMolStandardize.FragmentParent",
                    "rdMolStandardize.Uncharger",
                ]
            },
            "inputs": {"target_clusters": {**_record(targets), "rows": 1}},
            "outputs": {
                "ligands": {**_record(ligands), "rows": 1},
                "edges": {**_record(edges), "rows": 1},
            },
            "source_counts": {"ChEMBL": 2, "BindingDB": 1},
            "evidence_licensing": {
                "sources_tier": "permissive",
                "sources": ["bindingdb_own"],
                "licences": ["CC BY 4.0 (BindingDB-curated)", "CC BY-SA 3.0"],
                "source_metadata": {
                    "ChEMBL": {
                        "rows": 2,
                        "releases": ["37"],
                        "licences": ["CC BY-SA 3.0"],
                        "source_manifest": _record(source_manifest),
                    },
                    "BindingDB": {
                        "rows": 1,
                        "releases": ["2026-08"],
                        "licences": ["CC BY 4.0 (BindingDB-curated)"],
                        "source_manifest": _record(source_manifest),
                    },
                },
            },
        },
    )
    return {"runtime_index_manifest": manifest, "runtime_recipe": recipe}


def _cli_contract_args(contract: dict[str, Path], gate: Path) -> list[str]:
    return [
        "--benchmark-manifest", str(contract["benchmark_manifest"]),
        "--index-manifest", str(contract["index_manifest"]),
        "--panels-manifest", str(contract["panels_manifest"]),
        "--selection-manifest", str(contract["selection_manifest"]),
        "--final-evaluation-manifest", str(contract["final_evaluation_manifest"]),
        "--pocket-leakage-manifest", str(contract["pocket_leakage_manifest"]),
        "--out-gate", str(gate),
    ]


def test_create_cli_dispatches_to_claim_gate(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    gate = tmp_path / "claim_gate.flag"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/validate_activity_retrieval_gate.py"),
            "create",
            *_cli_contract_args(contract, gate),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert check_gate(gate)["status"] == "pass"


def test_create_operational_cli_dispatches_with_runtime_bindings(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    gate = tmp_path / "operational_gate.flag"
    runtime = _operational_args(contract, tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/validate_activity_retrieval_gate.py"),
            "create-operational",
            *_cli_contract_args(contract, gate),
            "--runtime-index-manifest", str(runtime["runtime_index_manifest"]),
            "--runtime-recipe", str(runtime["runtime_recipe"]),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert check_operational_gate(gate)["status"] == "pass"


def test_gate_creation_and_revalidation_bind_every_production_artifact(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    gate = tmp_path / "gate.flag"

    payload = create_gate(**contract, out_gate=gate)

    assert payload["status"] == "pass"
    assert payload["source_target_exclusions"]["sha256"] == _sha256(
        tmp_path / "source_target_exclusions.csv"
    )
    assert payload["pocket_leakage_rows"]["sha256"] == _sha256(
        tmp_path / "pocket_leakage_rows.csv"
    )
    assert gate.exists()
    assert check_gate(gate)["schema_version"] == (
        "skinscout.activity-retrieval-production-gate.v1"
    )


def test_current_ligand_identity_policy_chain_is_claimable(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    gate = tmp_path / "current_policy_gate.flag"

    payload = create_gate(**contract, out_gate=gate)

    assert payload["status"] == "pass"
    assert check_gate(gate)["status"] == "pass"


def test_hash_consistent_legacy_chain_cannot_create_or_check_a_claim_gate(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    gate = tmp_path / "legacy_claim_gate.flag"
    create_gate(**contract, out_gate=gate)

    def remove_policy(benchmark: dict[str, object]) -> None:
        splits = benchmark["splits"]
        assert isinstance(splits, dict)
        splits.pop("ligand_identity_policy")

    _rewrite_benchmark_contract(contract, remove_policy)
    _refresh_gate_manifest_records(gate, contract)

    with pytest.raises(SystemExit, match="ligand identity policy is required"):
        create_gate(**contract, out_gate=tmp_path / "new_claim_gate.flag")
    with pytest.raises(SystemExit, match="ligand identity policy is required"):
        check_gate(gate)


def test_hash_consistent_legacy_chain_is_only_an_unclaimable_operational_diagnostic(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)

    def remove_policy(benchmark: dict[str, object]) -> None:
        splits = benchmark["splits"]
        assert isinstance(splits, dict)
        splits.pop("ligand_identity_policy")

    _rewrite_benchmark_contract(contract, remove_policy)
    gate = tmp_path / "legacy_operational_gate.flag"

    payload = create_operational_gate(
        **contract, **_operational_args(contract, tmp_path), out_gate=gate
    )

    assert payload["status"] == "pass"
    assert payload["claim_ready"] is False
    assert payload["runtime_decision"]["claim_ready"] is False
    assert check_operational_gate(gate)["claim_ready"] is False


@pytest.mark.parametrize(
    "policy",
    [
        None,
        {"basis": "RDKit canonical isomeric SMILES with explicit hydrogens removed"},
        {
            "basis": "source ligand identifiers",
            "uses_source_ligand_ids": True,
            "uses_optional_source_inchikey": False,
            "applies_to": [
                "benchmark_id",
                "deduplication",
                "first_observation",
                "leakage_audit",
                "cold_flags",
            ],
        },
        {
            "basis": "RDKit canonical isomeric SMILES with explicit hydrogens removed",
            "uses_source_ligand_ids": 0,
            "uses_optional_source_inchikey": False,
            "applies_to": [
                "benchmark_id",
                "deduplication",
                "first_observation",
                "leakage_audit",
                "cold_flags",
            ],
        },
    ],
    ids=("present_null", "partial", "wrong", "integer_zero_flag"),
)
def test_operational_gate_rejects_present_invalid_ligand_identity_policy(
    tmp_path: Path, policy: object
) -> None:
    contract = _contract(tmp_path)

    def replace_policy(benchmark: dict[str, object]) -> None:
        splits = benchmark["splits"]
        assert isinstance(splits, dict)
        splits["ligand_identity_policy"] = policy

    _rewrite_benchmark_contract(contract, replace_policy)

    with pytest.raises(SystemExit, match="identity policy is incomplete or invalid"):
        create_operational_gate(
            **contract,
            **_operational_args(contract, tmp_path),
            out_gate=tmp_path / "invalid_policy_operational_gate.flag",
        )


def test_operational_gate_rejects_identity_policy_without_rdkit_version(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)

    def remove_rdkit_version(benchmark: dict[str, object]) -> None:
        versions = benchmark["versions"]
        assert isinstance(versions, dict)
        versions.pop("rdkit")

    _rewrite_benchmark_contract(contract, remove_rdkit_version)

    with pytest.raises(SystemExit, match="requires versions.rdkit"):
        create_operational_gate(
            **contract,
            **_operational_args(contract, tmp_path),
            out_gate=tmp_path / "missing_rdkit_operational_gate.flag",
        )


def test_operational_gate_retains_frozen_baseline_after_rejected_promotion(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    _set_failed_promotion(contract)
    gate = tmp_path / "operational_gate.flag"

    payload = create_operational_gate(
        **contract, **_operational_args(contract, tmp_path), out_gate=gate
    )

    assert payload["status"] == "pass"
    assert payload["claim_ready"] is False
    assert payload["promotion_decision"] == "diagnostic_runtime_binding"
    assert payload["operational_recipe_id"] == "chembl_p5_max"
    assert payload["evaluation_decision"]["promotion_decision"] == (
        "retain_frozen_baseline"
    )
    assert payload["runtime_decision"]["claim_ready"] is False
    assert payload["claim_gate_required_for_performance_claims"] is True
    assert check_operational_gate(gate)["schema_version"] == (
        "skinscout.activity-retrieval-operational-gate.v1"
    )
    with pytest.raises(SystemExit, match="final evaluation gate did not pass"):
        create_gate(**contract, out_gate=tmp_path / "claim_gate.flag")


def test_gate_tree_binding_ignores_only_snakemake_directory_markers(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    (tmp_path / "rcsb_contact_fragments" / ".snakemake_timestamp").touch()
    (tmp_path / "prior_fragments" / ".snakemake_timestamp").touch()

    gate = tmp_path / "claim_gate.flag"
    create_gate(**contract, out_gate=gate)

    assert check_gate(gate)["status"] == "pass"


def test_operational_gate_rejects_selected_candidate_as_failed_promotion_fallback(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    _set_failed_promotion(contract)
    final = json.loads(contract["final_evaluation_manifest"].read_text())
    summary_path = Path(final["outputs"]["summary.json"]["path"])
    summary = json.loads(summary_path.read_text())
    summary["operational_recipe_id"] = "union_p6_consensus"
    _write_json(summary_path, summary)
    final["operational_recipe_id"] = "union_p6_consensus"
    final["outputs"]["summary.json"] = _record(summary_path)
    _write_json(contract["final_evaluation_manifest"], final)

    with pytest.raises(SystemExit, match="operational_recipe_id is inconsistent"):
        create_operational_gate(
            **contract,
            **_operational_args(contract, tmp_path),
            out_gate=tmp_path / "operational_gate.flag",
        )


def _mutate_runtime_index(
    runtime: dict[str, Path],
    mutation: Callable[[dict[str, object]], None],
) -> None:
    path = runtime["runtime_index_manifest"]
    manifest = json.loads(path.read_text())
    mutation(manifest)
    _write_json(path, manifest)


def test_operational_gate_rejects_source_without_metadata(tmp_path: Path) -> None:
    """ChEMBL rows with no ChEMBL release/license/manifest record failed the
    stored artifact: only the added BindingDB tier was written down."""
    contract = _contract(tmp_path)
    runtime = _operational_args(contract, tmp_path)

    def drop_chembl(manifest: dict[str, object]) -> None:
        licensing = manifest["evidence_licensing"]
        assert isinstance(licensing, dict)
        metadata = licensing["source_metadata"]
        assert isinstance(metadata, dict)
        metadata.pop("ChEMBL")

    _mutate_runtime_index(runtime, drop_chembl)

    with pytest.raises(SystemExit, match="missing per-source.*: ChEMBL"):
        create_operational_gate(
            **contract, **runtime, out_gate=tmp_path / "operational_gate.flag"
        )


def test_operational_gate_rejects_source_metadata_not_in_counts(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    runtime = _operational_args(contract, tmp_path)

    def add_gtopdb(manifest: dict[str, object]) -> None:
        licensing = manifest["evidence_licensing"]
        assert isinstance(licensing, dict)
        metadata = licensing["source_metadata"]
        assert isinstance(metadata, dict)
        metadata["GtoPdb"] = {
            "rows": 1,
            "releases": ["2026-08"],
            "licences": ["ODbL 1.0"],
            "source_manifest": metadata["ChEMBL"]["source_manifest"],
        }

    _mutate_runtime_index(runtime, add_gtopdb)

    with pytest.raises(SystemExit, match="does not match source_counts: GtoPdb"):
        create_operational_gate(
            **contract, **runtime, out_gate=tmp_path / "operational_gate.flag"
        )


def test_operational_gate_rejects_source_metadata_row_mismatch(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    runtime = _operational_args(contract, tmp_path)

    def break_rows(manifest: dict[str, object]) -> None:
        counts = manifest["source_counts"]
        assert isinstance(counts, dict)
        counts["ChEMBL"] = 3

    _mutate_runtime_index(runtime, break_rows)

    with pytest.raises(SystemExit, match="source metadata does not match source_counts rows: ChEMBL"):
        create_operational_gate(
            **contract, **runtime, out_gate=tmp_path / "operational_gate.flag"
        )


@pytest.mark.parametrize("field", ["releases", "licences"])
def test_operational_gate_rejects_empty_source_release_or_license(
    tmp_path: Path, field: str
) -> None:
    contract = _contract(tmp_path)
    runtime = _operational_args(contract, tmp_path)

    def empty_field(manifest: dict[str, object]) -> None:
        licensing = manifest["evidence_licensing"]
        assert isinstance(licensing, dict)
        licensing["source_metadata"]["ChEMBL"][field] = []  # type: ignore[index]

    _mutate_runtime_index(runtime, empty_field)

    with pytest.raises(SystemExit, match="must be a non-empty string or list"):
        create_operational_gate(
            **contract, **runtime, out_gate=tmp_path / "operational_gate.flag"
        )


def test_operational_gate_rejects_missing_source_manifest_digest(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    runtime = _operational_args(contract, tmp_path)

    def break_manifest(manifest: dict[str, object]) -> None:
        licensing = manifest["evidence_licensing"]
        assert isinstance(licensing, dict)
        licensing["source_metadata"]["ChEMBL"]["source_manifest"]["sha256"] = "0" * 64  # type: ignore[index]

    _mutate_runtime_index(runtime, break_manifest)

    with pytest.raises(SystemExit, match="runtime source ChEMBL manifest record is stale"):
        create_operational_gate(
            **contract, **runtime, out_gate=tmp_path / "operational_gate.flag"
        )


def test_operational_gate_accepts_complete_per_source_metadata(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    runtime = _operational_args(contract, tmp_path)
    gate = tmp_path / "operational_gate.flag"

    payload = create_operational_gate(**contract, **runtime, out_gate=gate)

    assert payload["status"] == "pass"
    assert check_operational_gate(gate)["status"] == "pass"


def test_operational_gate_rejects_tampered_runtime_index(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    runtime = _operational_args(contract, tmp_path)
    gate = tmp_path / "operational_gate.flag"
    create_operational_gate(**contract, **runtime, out_gate=gate)
    runtime["runtime_index_manifest"].write_text("{}\n")

    with pytest.raises(SystemExit, match="schema_version must be"):
        check_operational_gate(gate)


def test_gate_check_rejects_any_stale_manifest(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    gate = tmp_path / "gate.flag"
    create_gate(**contract, out_gate=gate)
    contract["index_manifest"].write_text("{}\n")

    with pytest.raises(SystemExit, match="index_manifest.*stale"):
        check_gate(gate)


def test_failed_gate_creation_removes_stale_pass_artifact(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    gate = tmp_path / "gate.flag"
    gate.write_text('{"status": "pass"}\n')
    selection = json.loads(contract["selection_manifest"].read_text())
    selection["passes_dev_gate"] = False
    _write_json(contract["selection_manifest"], selection)

    with pytest.raises(SystemExit, match="dev-selection gate did not pass"):
        create_gate(**contract, out_gate=gate)

    assert not gate.exists()


def test_gate_rejects_final_recipe_not_selected_on_dev(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    other_recipe = tmp_path / "other_recipe.json"
    _write_json(other_recipe, {"schema_version": "other-recipe.v1"})
    final = json.loads(contract["final_evaluation_manifest"].read_text())
    final["inputs"]["recipe"] = _record(other_recipe)
    _write_json(contract["final_evaluation_manifest"], final)

    with pytest.raises(SystemExit, match="does not match dev-selected recipe"):
        create_gate(**contract, out_gate=tmp_path / "gate.flag")


def test_gate_rejects_unapplied_source_target_exclusions(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    benchmark = json.loads(contract["benchmark_manifest"].read_text())
    benchmark["filters"]["source_target_exclusions"]["applied"] = False
    _write_json(contract["benchmark_manifest"], benchmark)

    with pytest.raises(SystemExit, match="source-target exclusions were not applied"):
        create_gate(**contract, out_gate=tmp_path / "gate.flag")


def test_gate_rejects_inadequate_dual_cold_panel(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    panels = json.loads(contract["panels_manifest"].read_text())
    panels["passes_panel_adequacy_gate"] = False
    panels["selection"]["ranking_queries"]["dual_cold"]["adequacy"][
        "passes"
    ] = False
    _write_json(contract["panels_manifest"], panels)

    with pytest.raises(SystemExit, match="dual-cold panel adequacy gate did not pass"):
        create_gate(**contract, out_gate=tmp_path / "gate.flag")


def _rewrite_cold_claim_record(
    contract: dict[str, Path],
    mutation: Callable[[dict[str, object]], None],
) -> None:
    final_path = contract["final_evaluation_manifest"]
    final = json.loads(final_path.read_text())
    summary_path = Path(final["outputs"]["summary.json"]["path"])
    summary = json.loads(summary_path.read_text())
    mutation(final)
    mutation(summary)
    _write_json(summary_path, summary)
    final["outputs"]["summary.json"] = _record(summary_path)
    _write_json(final_path, final)


def test_gate_rejects_zero_scorable_truth_cold_claim(tmp_path: Path) -> None:
    contract = _contract(tmp_path)

    def zero_scorable_truth(payload: dict[str, object]) -> None:
        gates = payload["gates"]  # type: ignore[index]
        assert isinstance(gates, dict)
        dual = gates["dual_cold"]
        assert isinstance(dual, dict)
        dual["claimable"] = False
        dual["passes_truth_universe_coverage"] = False
        coverage = dual["coverage"]
        assert isinstance(coverage, dict)
        coverage["truth_pairs_in_scorable_universe"] = 0
        coverage["truth_pair_coverage"] = 0.0
        coverage["in_universe"] = {"top10": None, "top30": None, "mrr": None}

    _rewrite_cold_claim_record(contract, zero_scorable_truth)

    with pytest.raises(SystemExit, match="claim coverage gate did not pass"):
        create_gate(**contract, out_gate=tmp_path / "gate.flag")


def test_gate_rejects_missing_cold_claim_coverage(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    _rewrite_cold_claim_record(
        contract,
        lambda payload: payload.pop("gates", None),
    )

    with pytest.raises(SystemExit, match="missing the dual-cold claim coverage"):
        create_gate(**contract, out_gate=tmp_path / "gate.flag")


def test_gate_rejects_inconsistent_cold_claim_between_manifest_and_summary(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    summary_path = Path(
        json.loads(contract["final_evaluation_manifest"].read_text())["outputs"]
        ["summary.json"]["path"]
    )
    summary = json.loads(summary_path.read_text())
    summary["gates"]["dual_cold"]["claimable"] = False
    _write_json(summary_path, summary)
    final = json.loads(contract["final_evaluation_manifest"].read_text())
    final["outputs"]["summary.json"] = _record(summary_path)
    _write_json(contract["final_evaluation_manifest"], final)

    with pytest.raises(SystemExit, match="dual-cold claim record is inconsistent"):
        create_gate(**contract, out_gate=tmp_path / "gate.flag")


def test_gate_rejects_missing_rcsb_panel_contract(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    panels = json.loads(contract["panels_manifest"].read_text())
    del panels["inputs"]["rcsb_holo_direct_contact_panel"]["contract"][
        "positive_only"
    ]
    _write_json(contract["panels_manifest"], panels)

    with pytest.raises(SystemExit, match="RCSB panel contract"):
        create_gate(**contract, out_gate=tmp_path / "gate.flag")


def test_gate_check_rejects_mutated_source_target_exclusion_contract(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    gate = tmp_path / "gate.flag"
    create_gate(**contract, out_gate=gate)
    (tmp_path / "source_target_exclusions.csv").write_text("mutated\n")

    with pytest.raises(SystemExit, match="source-target exclusion contract is stale"):
        check_gate(gate)


def test_gate_rejects_pocket_leakage_contract_used_for_model_selection(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    leakage = json.loads(contract["pocket_leakage_manifest"].read_text())
    leakage["contract"]["never_model_selection"] = False
    _write_json(contract["pocket_leakage_manifest"], leakage)

    with pytest.raises(SystemExit, match="pocket leakage contract"):
        create_gate(**contract, out_gate=tmp_path / "gate.flag")


def test_gate_rejects_stale_pocket_leakage_summary(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    leakage = json.loads(contract["pocket_leakage_manifest"].read_text())
    leakage["summary"]["pairs"]["covered"] = 2
    _write_json(contract["pocket_leakage_manifest"], leakage)

    with pytest.raises(SystemExit, match=r"pairs\.covered is stale"):
        create_gate(**contract, out_gate=tmp_path / "gate.flag")


def test_gate_check_rejects_mutated_pocket_fragment_tree(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    gate = tmp_path / "gate.flag"
    create_gate(**contract, out_gate=gate)
    (tmp_path / "rcsb_contact_fragments" / "pair-a.pdb").write_text("mutated\n")

    with pytest.raises(SystemExit, match="RCSB contact fragment directory is stale"):
        check_gate(gate)
