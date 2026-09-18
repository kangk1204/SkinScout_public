from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import os

import pytest
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import build_target_lab_package as package
from compound_applicability import assess
from target_intent import build_target_intents


@pytest.fixture(scope="module")
def intents():
    workbook = os.environ.get("SKINSCOUT_BIOMARKER_XLSX", "").strip()
    gene_map = os.environ.get("SKINSCOUT_GENE_MAP_PATH", "").strip()
    if not workbook or not gene_map:
        pytest.skip(
            "private workbook not provided "
            "(SKINSCOUT_BIOMARKER_XLSX / SKINSCOUT_GENE_MAP_PATH)"
        )
    return build_target_intents(Path(workbook), Path(gene_map))


def source(intent_id="TI-020-TYR", smiles="CCCCC1=C(C=C(C=C1)O)O"):
    return {
        **package.identity(smiles),
        "intent_id": intent_id,
        "evidence_count": 1,
        "unique_publication_count": 1,
        "direction_relations_json": '["supports"]',
        "molecule_pref_name": "test source compound",
    }


def safety(row, decision="PASS", **kwargs):
    result = {
        "compound_id": row["compound_id"],
        "canonical_smiles": row["canonical_isomeric_smiles"],
        "applicability": assess(row["canonical_isomeric_smiles"]),
        "decision": decision,
        "run_valid": True,
        "safety_consensus_valid": True,
        "full_analysis_complete": True,
        "applicability_domain": "inside",
        "skin_sens": {"degraded": False},
        "admet_status": "ok",
    }
    result.update(kwargs)
    return {row["compound_id"]: result}


def evidence(row, **kwargs):
    result = {
        **row,
        "compound_name": "reviewed candidate",
        "evidence_id": "review:1",
        "direction_relation": "supports",
        "primary_evidence_verified": True,
        "independent_review_verified": True,
        "human_relevance_verified": True,
        "conditions_complete": True,
        "material_policy": "eligible",
        "role": "material_candidate",
        "pmid": "1",
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/1/",
        "reported_result": "functional effect",
        "reported_conditions": "defined experimental conditions",
        "limitations": "research only",
        "next_experiment": "orthogonal confirmation",
    }
    result.update(kwargs)
    return result


def matrix(intents, rows, literature=None, safety_rows=None, historical=None):
    return package.build_matrix(
        intents,
        pd.DataFrame(rows),
        pd.DataFrame(historical or []),
        literature or [],
        safety_rows or {},
    )


def test_exact_stereoisomers_are_not_deduplicated():
    first = package.identity("CC(O)C(=O)O")
    left = package.identity("C[C@H](O)C(=O)O")
    right = package.identity("C[C@@H](O)C(=O)O")
    assert len({first["compound_id"], left["compound_id"], right["compound_id"]}) == 3


def test_source_direction_and_computational_pass_do_not_supply_primary_evidence(
    intents,
):
    row = source()
    result, _ = matrix(intents, [row], safety_rows=safety(row))
    assert result.iloc[0].decision == "needs_evidence"
    assert not result.iloc[0].minimum_evidence_met


def test_self_asserted_primary_claim_without_independent_review_cannot_promote(intents):
    row = source()
    result, _ = matrix(
        intents, [row], [evidence(row, independent_review_verified=False)], safety(row)
    )
    assert result.iloc[0].decision == "needs_evidence"
    assert not result.iloc[0].minimum_evidence_met


def test_shared_compound_keeps_both_target_relationships_and_one_safety_record(intents):
    one = source("TI-015-MMP1")
    two = source("TI-016-MMP3")
    result, compounds = matrix(intents, [one, two], safety_rows=safety(one))
    assert len(result) == 2
    assert len(compounds) == 1
    assert result.safety_decision.tolist() == ["PASS", "PASS"]
    assert result.structure_status.eq("abstained_unvalidated_metal_support").all()


def test_historical_dedup_does_not_erase_target_pair(intents):
    row = source("TI-015-MMP1")
    historical = [
        {"uniprot": "P03956", "smiles": row["canonical_isomeric_smiles"]},
        {"uniprot": "P08254", "smiles": row["canonical_isomeric_smiles"]},
    ]
    result, compounds = matrix(intents, [row], historical=historical)
    assert len(result) == 2
    assert len(compounds) == 1
    assert (
        sum(len(json.loads(value)) for value in result.historical_relationship_rows)
        == 2
    )


def test_historical_lineage_counts_and_loss_are_checked(intents):
    row = source("TI-015-MMP1")
    historical = pd.DataFrame(
        [
            {"uniprot": "P03956", "smiles": row["canonical_isomeric_smiles"]},
            {"uniprot": "P08254", "smiles": row["canonical_isomeric_smiles"]},
        ]
    )
    result, _ = matrix(intents, [row], historical=historical.to_dict("records"))
    assert package.historical_counts(intents, historical, result) == {
        "historical_raw_rows": 2,
        "historical_unique_exact_compounds": 1,
        "historical_unique_intent_compound_pairs": 2,
        "historical_lineage_mentions": 2,
    }
    result.loc[0, "historical_relationship_rows"] = "[]"
    with pytest.raises(ValueError, match="lineage"):
        package.historical_counts(intents, historical, result)


def test_nonhuman_primary_result_does_not_qualify_human_target(intents):
    row = source()
    result, _ = matrix(
        intents,
        [row],
        literature=[evidence(row, human_relevance_verified=False)],
        safety_rows=safety(row),
    )
    assert result.iloc[0].decision == "needs_evidence"
    assert not result.iloc[0].minimum_evidence_met


def test_current_partial_halt_is_not_downgraded_by_missing_admet(intents):
    row = source()
    result, _ = matrix(
        intents,
        [row],
        literature=[evidence(row)],
        safety_rows=safety(
            row,
            "HALT",
            run_valid=False,
            full_analysis_complete=False,
            admet_status="error",
        ),
    )
    assert result.iloc[0].decision == "exclude"


def test_flagged_candidate_is_not_promoted_even_with_complete_functional_evidence(
    intents,
):
    row = source()
    result, _ = matrix(intents, [row], [evidence(row)], safety(row, "FLAG_HIGH"))
    assert result.iloc[0].decision == "needs_evidence"


def synthetic_intent(intent_id, gene_symbol, uniprot_id):
    return {
        "schema_version": "skinscout.target-intent.v1",
        "intent_id": intent_id,
        "source_row_id": "1",
        "source_sheet": "test",
        "source_row_number": 1,
        "category": "test",
        "biomarker": gene_symbol,
        "full_name": gene_symbol,
        "marker_type": "protein",
        "role_summary": "test",
        "source_direction": "test",
        "route": "direct_target",
        "entity_type": "protein",
        "entity_name": gene_symbol,
        "gene_symbol": gene_symbol,
        "uniprot_id": uniprot_id,
        "protein_form": None,
        "component_of": None,
        "desired_effect": "inhibit_function",
        "target_scope": "skin",
        "skin_compartment": "dermis",
        "readout": "enzymatic activity",
        "source_evidence": {
            "workbook_path": "test.xlsx",
            "workbook_sha256": "a" * 64,
            "sheet": "test",
            "row_number": 1,
            "gene_map_path": "gene_map.csv",
            "gene_map_sha256": "b" * 64,
            "registry_path": "registry.csv",
            "registry_sha256": "c" * 64,
            "curated_on": "2026-01-01",
        },
        "target_taxid": 9606,
        "role": "material_candidate",
        "docking_eligible": True,
    }


def ilomastat_intents():
    return [
        synthetic_intent("TI-015-MMP1", "MMP1", "P03956"),
        synthetic_intent("TI-016-MMP3", "MMP3", "P08254"),
    ]


def test_reference_control_curation_does_not_leak_to_other_intent():
    intents = ilomastat_intents()
    one = source("TI-015-MMP1")
    two = source("TI-016-MMP3")
    result, _ = matrix(
        intents,
        [one, two],
        [evidence(one, role="reference_control", material_policy="excluded")],
    )
    by_intent = result.set_index("intent_id")
    assert by_intent.loc["TI-015-MMP1", "role"] == "reference_control"
    assert by_intent.loc["TI-015-MMP1", "material_policy"] == "excluded"
    assert by_intent.loc["TI-015-MMP1", "decision"] == "exclude"
    assert by_intent.loc["TI-015-MMP1", "material_policy_scope"] == "intent_compound"
    assert by_intent.loc["TI-016-MMP3", "role"] == "material_candidate"
    assert by_intent.loc["TI-016-MMP3", "material_policy"] == "unknown"
    assert by_intent.loc["TI-016-MMP3", "material_policy_scope"] == "intent_compound"
    assert by_intent.loc["TI-016-MMP3", "decision"] == "needs_evidence"


def test_changing_one_intent_policy_does_not_change_another_intent():
    intents = ilomastat_intents()
    one = source("TI-015-MMP1")
    two = source("TI-016-MMP3")
    baseline, _ = matrix(intents, [one, two])
    curated, _ = matrix(
        intents,
        [one, two],
        [evidence(one, role="reference_control", material_policy="excluded")],
    )
    columns = ["role", "material_policy", "material_policy_scope", "decision"]
    baseline_row = baseline.set_index("intent_id").loc["TI-016-MMP3", columns]
    curated_row = curated.set_index("intent_id").loc["TI-016-MMP3", columns]
    assert baseline_row.tolist() == curated_row.tolist()
    assert baseline.set_index("intent_id").loc["TI-015-MMP1", columns].tolist() == [
        "material_candidate",
        "unknown",
        "intent_compound",
        "needs_evidence",
    ]


def test_explicit_compound_global_policy_applies_to_all_intents_with_scope():
    intents = ilomastat_intents()
    one = source("TI-015-MMP1")
    two = source("TI-016-MMP3")
    basis = "Independent review classified the compound as an assay reference control"
    result, _ = matrix(
        intents,
        [one, two],
        [
            evidence(
                one,
                role="reference_control",
                material_policy="excluded",
                policy_scope="compound_global",
                policy_basis=basis,
            )
        ],
    )
    assert result.role.eq("reference_control").all()
    assert result.material_policy.eq("excluded").all()
    assert result.material_policy_scope.eq("compound_global").all()
    assert result.material_policy_basis.eq(basis).all()
    assert result.role_scope.eq("compound_global").all()
    assert result.decision.eq("exclude").all()


def test_compound_global_policy_requires_explicit_basis():
    intents = ilomastat_intents()
    one = source("TI-015-MMP1")
    two = source("TI-016-MMP3")
    with pytest.raises(ValueError, match="explicit basis"):
        matrix(
            intents,
            [one, two],
            [
                evidence(
                    one,
                    role="reference_control",
                    material_policy="excluded",
                    policy_scope="compound_global",
                    policy_basis="",
                )
            ],
        )


def test_invalid_material_policy_scope_is_rejected():
    intents = ilomastat_intents()
    one = source("TI-015-MMP1")
    with pytest.raises(ValueError, match="Invalid material policy scope"):
        matrix(
            intents,
            [one],
            [evidence(one, policy_scope="global")],
        )


@pytest.mark.parametrize(
    "prefix", ["unresolved-structure-sha256:", "unresolved-source-compound-sha256:"]
)
def test_unresolved_structure_remains_in_full_denominator(intents, prefix):
    row = {
        **source(),
        "compound_id": prefix + "0" * 64,
        "canonical_isomeric_smiles": None,
        "computed_inchikey": None,
    }
    result, compounds = matrix(intents, [row])
    assert len(result) == len(compounds) == 1
    assert result.iloc[0].decision == "unsupported"


def test_all_31_rows_33_intents_remain_when_candidate_coverage_is_sparse(intents):
    row = source()
    result, _ = matrix(intents, [row])
    coverage = package.coverage_table(intents, result)
    assert len(coverage) == 33
    assert coverage.source_row_id.nunique() == 31
    endpoint = coverage[coverage.route.eq("endpoint")]
    assert len(endpoint) == 4
    assert endpoint.uniprot_id.isna().all()
    assert coverage.candidate_count.sum() == 1


def test_filtering_happens_before_top_display_and_empty_scaffolds_are_distinct():
    rows = []
    for number in range(11):
        rows.append(
            {
                "intent_id": "example",
                "compound_id": str(number),
                "decision": "exclude" if number < 10 else "experiment_priority",
                "reviewed_primary_publication_count": 1,
                "canonical_isomeric_smiles": "CCCCO",
            }
        )
    rows.append(
        {**rows[-1], "compound_id": "12", "canonical_isomeric_smiles": "CCCCCO"}
    )
    selected = package.select_priority(pd.DataFrame(rows))
    assert set(selected.compound_id) == {"10", "12"}


def test_mixed_assay_directions_are_not_claimed_to_be_a_same_context_conflict():
    decision, reason = package.aggregate_direction(["supports", "opposes", "unknown"])
    assert decision == "unknown"
    assert "context" in reason


def test_changed_safety_file_cannot_be_reused(tmp_path):
    row = source()
    record = next(iter(safety(row).values()))
    path = tmp_path / "candidate_safety.jsonl"
    path.write_text(json.dumps(record) + "\n")
    package.write_json(
        tmp_path / "manifest.json",
        {
            "candidate_count": 1,
            "artifacts": {"candidate_safety_jsonl": {"sha256": package.digest(path)}},
        },
    )
    assert len(package.load_safety([path])[0]) == 1
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="hash mismatch"):
        package.load_safety([path])


def test_duplicate_safety_same_decision_different_validity_requires_reconciliation(
    tmp_path,
):
    row = source()
    record = next(iter(safety(row, "HALT").values()))
    paths = []
    for index, valid in enumerate((True, False)):
        directory = tmp_path / str(index)
        directory.mkdir()
        path = directory / "candidate_safety.jsonl"
        path.write_text(json.dumps({**record, "safety_consensus_valid": valid}) + "\n")
        package.write_json(
            directory / "manifest.json",
            {
                "candidate_count": 1,
                "artifacts": {
                    "candidate_safety_jsonl": {"sha256": package.digest(path)}
                },
            },
        )
        paths.append(path)
    for ordering in (paths, list(reversed(paths))):
        with pytest.raises(ValueError, match="explicit reconciliation"):
            package.load_safety(ordering)


def test_reconciled_safety_count_contract_is_accepted(tmp_path):
    row = source()
    record = next(iter(safety(row).values()))
    path = tmp_path / "candidate_safety.jsonl"
    path.write_text(json.dumps(record) + "\n")
    package.write_json(
        tmp_path / "manifest.json",
        {
            "schema": "skinscout.target-candidate-safety.v1.reconciliation",
            "selected_unique_compound_count": 1,
            "artifacts": {"candidate_safety_jsonl": {"sha256": package.digest(path)}},
        },
    )
    assert len(package.load_safety([path])[0]) == 1


def model_evidence_batch(tmp_path, row, mutate=None):
    record = next(iter(safety(row).values()))
    record["canonical_smiles_sha256"] = hashlib.sha256(
        record["canonical_smiles"].encode("utf-8")
    ).hexdigest()
    envelope = {
        "schema": "skinscout.target-candidate-safety.v1.model-response",
        "model": "husspred",
        "request_smiles": record["canonical_smiles"],
        "request_smiles_sha256": record["canonical_smiles_sha256"],
        "parsed_payload": {"status": "ok"},
    }
    raw = tmp_path / "raw" / "husspred.json"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(json.dumps(envelope) + "\n")
    if mutate is not None:
        mutated = json.loads(raw.read_text())
        mutate(mutated)
        raw.write_text(json.dumps(mutated) + "\n")
    record["model_results"] = {
        "husspred": {
            "artifact_path": str(raw.resolve()),
            "artifact_sha256": package.digest(raw),
            "request_smiles_sha256": record["canonical_smiles_sha256"],
        }
    }
    path = tmp_path / "candidate_safety.jsonl"
    path.write_text(json.dumps(record) + "\n")
    package.write_json(
        tmp_path / "manifest.json",
        {
            "candidate_count": 1,
            "artifacts": {
                "candidate_safety_jsonl": {"sha256": package.digest(path)}
            },
        },
    )
    return path


def test_handoff_verifies_raw_model_evidence(tmp_path):
    path = model_evidence_batch(tmp_path, source())
    assert len(package.load_safety([path])[0]) == 1


def test_handoff_rejects_tampered_model_artifact_bytes(tmp_path):
    path = model_evidence_batch(tmp_path, source())
    raw = tmp_path / "raw" / "husspred.json"
    raw.write_text(raw.read_text() + " ")
    with pytest.raises(ValueError, match="hash mismatch"):
        package.load_safety([path])


def test_handoff_rejects_model_identity_tamper(tmp_path):
    path = model_evidence_batch(
        tmp_path,
        source(),
        mutate=lambda envelope: envelope.update({"model": "stoptox"}),
    )
    with pytest.raises(ValueError, match="model mismatch"):
        package.load_safety([path])


def test_handoff_rejects_request_identity_tamper(tmp_path):
    path = model_evidence_batch(
        tmp_path,
        source(),
        mutate=lambda envelope: envelope.update({"request_smiles": "CCC"}),
    )
    with pytest.raises(ValueError, match="request identity mismatch"):
        package.load_safety([path])


def test_handoff_rejects_schema_identity_tamper(tmp_path):
    path = model_evidence_batch(
        tmp_path,
        source(),
        mutate=lambda envelope: envelope.update({"schema": "other.schema"}),
    )
    with pytest.raises(ValueError, match="schema mismatch"):
        package.load_safety([path])


def test_changed_claim_cannot_reuse_independent_review(intents, tmp_path):
    original = ROOT / "data/curation/target_candidate_literature_20260915.json"
    path = tmp_path / "curation.json"
    payload = json.loads(original.read_text())
    payload["records"][0]["reported_result"] = "Unsupported replacement claim"
    path.write_text(json.dumps(payload))
    attestation = tmp_path / "review.json"
    package.write_json(
        attestation,
        {
            "schema_version": "skinscout.literature-review.v1",
            "curation_sha256": package.digest(original),
            "records": [],
            "supporting_files": [],
        },
    )
    with pytest.raises(ValueError, match="stale"):
        package.load_curated(
            path,
            ROOT / "results/target_first_20260915/literature",
            intents,
            attestation,
        )


def test_spreadsheet_source_text_cannot_be_executed_as_formula(tmp_path):
    path = tmp_path / "review.xlsx"
    package.write_workbook(
        path, {"source": pd.DataFrame({"assay": ["=1+1", "@SUM(A1)", "normal text"]})}
    )
    sheet = load_workbook(path).active
    assert sheet["A2"].data_type == "s"
    assert sheet["A3"].data_type == "s"
    assert sheet.freeze_panes == "A2"


def test_control_metrics_keep_disagreeing_orderings_separate():
    common = {
        "structure_status": "completed",
        "assay_id": "same-assay",
        "reported_type": "Ki",
        "reported_relation": "=",
        "reported_units": "nM",
    }
    frame = pd.DataFrame(
        [
            {
                **common,
                "control_role": "active_control",
                "reported_value": 300,
                "gnina_cnn_affinity": 4.9,
                "gnina_cnn_score": 0.23,
                "gnina_minimized_affinity": -7.1,
            },
            {
                **common,
                "control_role": "weak_binding_control",
                "reported_value": 70700,
                "gnina_cnn_affinity": 5.7,
                "gnina_cnn_score": 0.38,
                "gnina_minimized_affinity": -6.4,
            },
        ]
    )
    result = package.control_ranking_summary(frame)
    assert result["metric_order_matches_measured_affinity"] == {
        "gnina_cnn_affinity": False,
        "gnina_cnn_score": False,
        "gnina_minimized_affinity": True,
    }
    frame.loc[1, "assay_id"] = "different-assay"
    assert package.control_ranking_summary(frame)["status"] == "not_comparable"


def test_portable_copy_preserves_bytes_and_exposes_original_path(tmp_path):
    source_dir = tmp_path / "source"
    source = source_dir / "structure-sha256:abcd" / "raw.json"
    source.parent.mkdir(parents=True)
    source.write_bytes(b'{"response":"unchanged"}')
    out = tmp_path / "package"
    mapping = []
    package.copy_artifact_tree(source_dir, out / "safety", out, mapping)
    target = out / "safety/structure-sha256_abcd/raw.json"
    assert target.read_bytes() == source.read_bytes()
    assert mapping[0]["package_path"] == "safety/structure-sha256_abcd/raw.json"
    assert mapping[0]["sha256"] == package.digest(source)
    assert package.portable_relative(Path("CON.txt")) == Path("_CON.txt")


def test_portable_copy_rejects_filename_collision(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "a:b").write_text("one")
    (source / "a_b").write_text("two")
    with pytest.raises(ValueError, match="collide"):
        package.copy_artifact_tree(source, tmp_path / "out", tmp_path / "out", [])


def test_review_applicability_keeps_raw_state_and_decision_lineage():
    """`review` must reach decide_candidate as review, not as in_scope, and not
    as an unconditional exclusion."""
    intents = ilomastat_intents()
    row = source("TI-015-MMP1", smiles="CCCCCCCCCCCCCCCCCC(=O)O")
    assert assess(row["canonical_isomeric_smiles"])["verdict"] == "review"

    result, _ = matrix(intents, [row], [evidence(row)], safety(row))
    record = result.set_index("intent_id").loc["TI-015-MMP1"]

    assert record["applicability"] == "review"
    assert record["analysis_applicability"] == "review"
    assert bool(record["analysis_scope_supported"]) is True
    assert record["decision"] == "needs_evidence"
    lineage = json.loads(record["decision_inputs"])
    assert lineage["applicability"]["raw_state"] == "review"
    assert lineage["applicability"]["analysis_scope_supported"] is True
    assert lineage["applicability"]["warning_codes"] == [
        "rotatable_bonds_outside_panel"
    ]
    assert lineage["summary"]["analysis_applicability"] == "review"
    assert lineage["summary"]["analysis_scope_supported"] is True
    assert any(
        "requires review" in reason
        for reason in json.loads(record["decision_reasons"])
    )


def test_out_of_scope_applicability_lineage_is_preserved():
    intents = ilomastat_intents()
    row = source("TI-015-MMP1", smiles="CCO.O")
    assert assess(row["canonical_isomeric_smiles"])["verdict"] == "out_of_scope"

    result, _ = matrix(intents, [row], [evidence(row)], safety(row))
    record = result.set_index("intent_id").loc["TI-015-MMP1"]

    assert record["analysis_applicability"] == "out_of_scope"
    assert bool(record["analysis_scope_supported"]) is False
    assert record["decision"] == "unsupported"
    lineage = json.loads(record["decision_inputs"])
    assert lineage["applicability"]["raw_state"] == "out_of_scope"
    assert lineage["applicability"]["analysis_scope_supported"] is False


def test_in_scope_control_still_records_its_lineage():
    intents = ilomastat_intents()
    row = source("TI-015-MMP1")
    assert assess(row["canonical_isomeric_smiles"])["verdict"] == "in_scope"

    result, _ = matrix(intents, [row], [evidence(row)], safety(row))
    record = result.set_index("intent_id").loc["TI-015-MMP1"]

    assert record["analysis_applicability"] == "in_scope"
    assert bool(record["analysis_scope_supported"]) is True
    assert record["decision"] == "experiment_priority"
    lineage = json.loads(record["decision_inputs"])
    assert lineage["summary"]["analysis_applicability"] == "in_scope"
    assert lineage["summary"]["analysis_scope_supported"] is True
