from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from rdkit import Chem

from scripts import discover_substitutes as discovery


TARGET = "P12345"
PARENT = "Oc1ccccc1"


def _inchikey(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    return Chem.MolToInchiKey(molecule)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_fingerprint(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _anchor_payload(
    smiles: str,
    anchor_indices: list[int],
    *,
    parent_sdf: Path | None = None,
    consensus_json: Path | None = None,
    boltz_report: Path | None = None,
    complex_pdb: Path | None = None,
) -> dict[str, object]:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    canonical_molecule = Chem.MolFromSmiles(canonical)
    assert canonical_molecule is not None
    atom_count = canonical_molecule.GetNumAtoms()
    identity_mapping = [
        {"complex_atom_index": index, "parent_atom_index": index}
        for index in range(atom_count)
    ]
    canonical_mapping = [
        {"canonical_atom_index": index, "parent_atom_index": index}
        for index in range(atom_count)
    ]
    placeholder_sha = "0" * 64
    parent_fingerprint = (
        _source_fingerprint(parent_sdf)
        if parent_sdf is not None
        else {"path": "fixture_parent.sdf", "bytes": 1, "sha256": placeholder_sha}
    )
    consensus_fingerprint = (
        _source_fingerprint(consensus_json)
        if consensus_json is not None
        else {"path": "fixture_consensus.json", "bytes": 1, "sha256": "1" * 64}
    )
    boltz_fingerprint = (
        _source_fingerprint(boltz_report)
        if boltz_report is not None
        else {"path": "fixture_boltz_report.json", "bytes": 1, "sha256": "2" * 64}
    )
    complex_fingerprint = (
        _source_fingerprint(complex_pdb)
        if complex_pdb is not None
        else {"path": "fixture_complex.pdb", "bytes": 1, "sha256": "3" * 64}
    )
    return {
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
            TARGET: {
                "target_id": TARGET,
                "complex_pdb": complex_fingerprint["path"],
                "complex_pdb_bytes": complex_fingerprint["bytes"],
                "complex_pdb_sha256": complex_fingerprint["sha256"],
                "source_coordinate_system": (
                    "boltz_complex_ligand_atom_order_0_based"
                ),
                "mapped_coordinate_system": "parent_sdf_atom_order_0_based",
                "confirmed_complex_atoms": anchor_indices,
                "confirmed_parent_atoms": anchor_indices,
                "canonical_coordinate_system": (
                    "canonical_smiles_atom_order_0_based"
                ),
                "confirmed_canonical_parent_atoms": anchor_indices,
                "atom_mapping": identity_mapping,
                "canonical_atom_mapping": canonical_mapping,
                "mapping_status": "mapped",
                "mapping_confidence": "high",
                "claim_eligible": True,
            }
        },
        "claim_eligible": True,
        "analog_pose_verified": False,
    }


def _write_anchor_map(path: Path, smiles: str, anchor_indices: list[int]) -> None:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    parent_sdf = path.with_name("parent.sdf")
    consensus_json = path.with_name("consensus.json")
    boltz_report = path.with_name("boltz_report.json")
    complex_pdb = path.with_name("complex.pdb")
    parent_sdf.write_text(Chem.MolToMolBlock(molecule), encoding="utf-8")
    consensus_json.write_text(
        json.dumps({"target_id": TARGET, "confirmed_parent_atoms": anchor_indices}),
        encoding="utf-8",
    )
    boltz_report.write_text(
        json.dumps({"target_id": TARGET, "status": "fixture"}),
        encoding="utf-8",
    )
    complex_pdb.write_text(
        "HEADER    FIXTURE COMPLEX\nEND\n",
        encoding="utf-8",
    )
    path.write_text(
        json.dumps(
            _anchor_payload(
                smiles,
                anchor_indices,
                parent_sdf=parent_sdf,
                consensus_json=consensus_json,
                boltz_report=boltz_report,
                complex_pdb=complex_pdb,
            )
        ),
        encoding="utf-8",
    )


def _write_library(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "inci_name": "Anisole",
                "functions": "PERFUMING",
                "smiles": "COc1ccccc1",
            },
            {
                "inci_name": "Catechol",
                "functions": "ANTIOXIDANT",
                "smiles": "Oc1ccccc1O",
            },
            {
                "inci_name": "Ethoxybenzene",
                "functions": "PERFUMING",
                "smiles": "CCOc1ccccc1",
            },
            {
                "inci_name": "Invalid",
                "functions": "",
                "smiles": "not-a-smiles",
            },
        ]
    ).to_parquet(path)


def _write_activity(path: Path) -> None:
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
                "source_pmid": "100",
                "affinity_type": "Ki",
            },
            {
                "uniprot": TARGET,
                "ligand_smiles": "COc1ccccc1",
                "ligand_inchikey": _inchikey("COc1ccccc1"),
                "affinity_value": 300.0,
                "affinity_unit": "nM",
                "relation": "=",
                "censor": False,
                "source_db": "FixtureDB",
                "source_release": "1",
                "source_doi": "10.1000/anisole",
                "source_pmid": "101",
                "affinity_type": "Ki",
            },
            {
                "uniprot": TARGET,
                "ligand_smiles": "Oc1ccccc1O",
                "ligand_inchikey": _inchikey("Oc1ccccc1O"),
                "affinity_value": 30_000.0,
                "affinity_unit": "nM",
                "relation": "=",
                "censor": False,
                "source_db": "FixtureDB",
                "source_release": "1",
                "source_doi": "10.1000/catechol",
                "source_pmid": "102",
                "affinity_type": "Ki",
            },
            {
                "uniprot": "Q99999",
                "ligand_smiles": "CCOc1ccccc1",
                "ligand_inchikey": _inchikey("CCOc1ccccc1"),
                "affinity_value": 1.0,
                "affinity_unit": "nM",
                "relation": "=",
                "censor": False,
                "source_db": "FixtureDB",
                "source_release": "1",
                "source_doi": "10.1000/wrong-target",
                "source_pmid": "103",
                "affinity_type": "Ki",
            },
        ]
    ).to_parquet(path)


def _args(tmp_path: Path) -> object:
    library = tmp_path / "cosing.parquet"
    activity = tmp_path / "activity.parquet"
    _write_library(library)
    _write_activity(activity)
    return discovery.parse_args(
        [
            "--parent-smiles",
            PARENT,
            "--target-id",
            TARGET,
            "--run-id",
            "fixture_run",
            "--candidate-library",
            str(library),
            "--activity-evidence",
            str(activity),
            "--out-dir",
            str(tmp_path / "out"),
            "--min-pharmacophore",
            "0.20",
            "--min-feature-recall",
            "0.50",
            "--max-candidates",
            "20",
        ]
    )


def test_discovery_ranks_direct_retention_and_keeps_proxy_claim_limited(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)

    report = discovery.run(args)

    assert report["schema_version"] == discovery.SCHEMA_VERSION
    assert report["claimable"] is False
    assert report["hypothesis_only"] is True
    by_name = {row["name"]: row for row in report["candidates"]}
    assert by_name["Anisole"]["evidence_tier"] == "direct_retained"
    assert by_name["Anisole"]["binding_retained"] is True
    assert by_name["Anisole"]["activity_sources"] == ["FixtureDB"]
    assert by_name["Anisole"]["activity_citations"][0]["doi"] == "10.1000/anisole"
    assert by_name["Catechol"]["evidence_tier"] == "direct_reduced"
    assert by_name["Catechol"]["binding_retained"] is False
    assert by_name["Ethoxybenzene"]["evidence_tier"] == "proxy_only"
    assert by_name["Ethoxybenzene"]["claimable"] is False
    assert by_name["Ethoxybenzene"]["hypothesis_only"] is True
    assert by_name["Ethoxybenzene"]["wet_lab_required"] is True
    assert by_name["Ethoxybenzene"]["analog_pose_verified"] is False
    assert by_name["Ethoxybenzene"]["selection_tracks"] == ["cosmetic_material"]
    assert 0.0 <= by_name["Ethoxybenzene"]["feature_family_precision"] <= 1.0
    assert 0.0 <= by_name["Ethoxybenzene"]["feature_family_f1"] <= 1.0
    assert report["parent"]["activity_median_pactivity"] == pytest.approx(7.0)
    assert report["source_audit"]["candidate_library_counts"]["invalid_rows"] == 1
    assert report["summary"]["direct_retained_candidates"] == 1
    assert "not a candidate-specific skin-safety" in report["score_definition"][
        "safety_triage_semantics"
    ]
    priorities = [row["priority_score"] for row in report["candidates"]]
    assert priorities == sorted(priorities, reverse=True)
    global_ranks = [row["global_priority_rank"] for row in report["candidates"]]
    assert global_ranks == sorted(global_ranks)
    assert report["selection_strategy"]["mode"] == "balanced_tracks"
    assert report["selection_strategy"]["selected_counts"]["all"] == len(
        report["candidates"]
    )
    three_d_evaluation = report["candidate_pool"]["pharmacophore_3d_evaluation"]
    assert three_d_evaluation["budget"] == 128
    assert three_d_evaluation["eligible_candidates"] == report["candidate_pool"][
        "passing_scored_molecules"
    ]
    assert three_d_evaluation["evaluated_candidates"] == three_d_evaluation[
        "eligible_candidates"
    ]
    assert three_d_evaluation["budget_excluded_candidates"] == 0
    assert report["summary"]["pharmacophore_3d_candidate_budget"] == 128
    assert report["summary"]["pharmacophore_3d_evaluated_candidates"] == (
        three_d_evaluation["evaluated_candidates"]
    )


def test_discovery_writes_interactive_and_machine_readable_artifacts(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)

    discovery.run(args)

    out_dir = Path(args.out_dir)
    for filename in discovery.OUTPUT_FILES:
        assert (out_dir / filename).is_file()
        assert (out_dir / filename).stat().st_size > 0
    payload = json.loads((out_dir / "substitute_report.json").read_text())
    assert payload["run_id"] == "fixture_run"
    assert "Ligand-feature pharmacophore similarity" in payload["limitations"][0]
    html = (out_dir / "substitute_report.html").read_text()
    assert "Search candidate name, ID, InChIKey, or SMILES" in html
    assert "data-tier='direct_retained'" in html
    assert "Structure/property triage" in html
    assert "Matched ΔpActivity" in html
    assert "Conservative endpoint/source stratum" in html
    assert 'id="filter-status" aria-live="polite"' in html
    supplier = Chem.SDMolSupplier(str(out_dir / "substitute_candidates_3d.sdf"), removeHs=False)
    molecules = [molecule for molecule in supplier if molecule is not None]
    assert molecules[0].GetProp("role") == "parent"
    substitute = next(
        molecule
        for molecule in molecules[1:]
        if molecule.GetProp("role") == "substitute_hypothesis"
    )
    assert substitute.GetProp("claimable") == "false"
    assert substitute.GetProp("hypothesis_only") == "true"
    assert substitute.GetProp("wet_lab_required") == "true"
    assert substitute.GetProp("pharmacophore_3d_evaluated") == "true"


def test_discovery_uses_validated_target_conditioned_anchor_ranking(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    anchor_map = tmp_path / "interaction_anchor_map.json"
    _write_anchor_map(anchor_map, PARENT, [0])
    args.interaction_anchor_map = anchor_map

    report = discovery.run(args)

    anchor_payload = json.loads(anchor_map.read_text())
    assert anchor_payload["boltz_report_sha256"] == _sha256(tmp_path / "boltz_report.json")
    assert anchor_payload["claim_eligible"] is True
    assert anchor_payload["analog_pose_verified"] is False
    for source_name in ("parent_sdf", "consensus_json", "boltz_report"):
        source = anchor_payload["sources"][source_name]
        source_path = Path(source["path"])
        assert source_path.is_file()
        assert source["bytes"] == source_path.stat().st_size
        assert source["sha256"] == _sha256(source_path)
    target_anchor = anchor_payload["targets"][TARGET]
    complex_pdb = Path(target_anchor["complex_pdb"])
    assert complex_pdb.is_file()
    assert target_anchor["complex_pdb_bytes"] == complex_pdb.stat().st_size
    assert target_anchor["complex_pdb_sha256"] == _sha256(complex_pdb)
    assert report["target"]["interaction_anchor_conditioned"] is True
    assert report["score_definition"]["priority_score"] == {
        "pharmacophore_preservation": 0.20,
        "target_conditioned_interaction_anchor": 0.10,
        "binding_support": 0.25,
        "safety_triage": 0.15,
        "routeability_proxy": 0.10,
        "material_reference": 0.10,
        "corpus_novelty_proxy": 0.10,
    }
    assert report["source_audit"]["interaction_anchor_map"]["sha256"]
    assert report["summary"]["target_conditioned_anchor_candidates"] == len(
        report["candidates"]
    )
    for candidate in report["candidates"]:
        assert 0.0 <= candidate["target_conditioned_anchor_score"] <= 1.0
        assert candidate["target_conditioned_anchor_count"] == 1
        assert candidate["target_conditioned_anchor_basis"] == (
            "pose_supported_parent_anchor_conservative_mcs_feature_preservation"
        )
        assert isinstance(candidate["target_conditioned_mapping_count"], int)
        assert candidate["target_conditioned_mapping_count"] >= 1
        assert isinstance(candidate["target_conditioned_mapping_ambiguous"], bool)
        assert candidate["target_conditioned_mapping_truncated"] is False
        assert candidate["analog_pose_verified"] is False
    assert "analog pose and affinity are not verified" in report["limitations"][0]
    assert "Target anchor" in (args.out_dir / "substitute_report.html").read_text()


def test_discovery_rejects_anchor_map_when_recorded_source_file_is_missing(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    anchor_map = tmp_path / "interaction_anchor_map.json"
    _write_anchor_map(anchor_map, PARENT, [0])
    (tmp_path / "boltz_report.json").unlink()
    args.interaction_anchor_map = anchor_map

    with pytest.raises(SystemExit, match="boltz_report source file is missing"):
        discovery.run(args)


def test_anchor_source_verification_accepts_matching_artifact_relative_path_when_cwd_is_shadowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    anchor_map = artifact_dir / "interaction_anchor_map.json"
    _write_anchor_map(anchor_map, PARENT, [0])
    payload = json.loads(anchor_map.read_text())
    for record in payload["sources"].values():
        record["path"] = Path(record["path"]).name
    payload["targets"][TARGET]["complex_pdb"] = "complex.pdb"
    anchor_map.write_text(json.dumps(payload))

    shadow = tmp_path / "shadow"
    shadow.mkdir()
    for name in ("parent.sdf", "consensus.json", "boltz_report.json", "complex.pdb"):
        (shadow / name).write_text("wrong shadow content\n")
    monkeypatch.chdir(shadow)

    loaded = discovery.load_anchor_map(
        anchor_map,
        expected_parent_inchikey=_inchikey(PARENT),
        required_target=TARGET,
        verify_source_files=True,
    )

    assert loaded["claim_eligible"] is True


def test_safety_triage_uses_same_critical_alert_penalty_for_parent_and_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_alerts = {name: [] for name in discovery.ALERT_CATALOGS}
    critical_alerts = {name: [] for name in discovery.ALERT_CATALOGS}
    critical_alerts["NIH"] = ["fixture"]

    unflagged = discovery._safety_triage_score(1.0, no_alerts)
    flagged = discovery._safety_triage_score(1.0, critical_alerts)

    assert flagged == pytest.approx(0.72)
    assert flagged - unflagged == pytest.approx(-0.28)

    monkeypatch.setattr(discovery, "_alert_catalogs", lambda: {})
    monkeypatch.setattr(
        discovery,
        "_structural_alerts",
        lambda _molecule, _catalogs: critical_alerts,
    )
    fixed_properties = {
        "molecular_weight": 100.0,
        "logp": 1.0,
        "tpsa": 40.0,
        "hbd": 1.0,
        "hba": 1.0,
        "rotatable_bonds": 1.0,
        "heavy_atoms": 5.0,
        "rings": 0.0,
    }
    monkeypatch.setattr(discovery, "_physchem", lambda _molecule: fixed_properties)
    parent = discovery._standardize_smiles("CCO", "parent")
    candidate = discovery._standardize_smiles("CCN", "candidate")
    rows, _ = discovery._score_candidates(
        parent,
        {
            candidate.smiles: discovery.CandidateSeed(
                smiles=candidate.smiles,
                inchikey=candidate.inchikey,
            )
        },
        min_pharmacophore=0.0,
        min_feature_recall=0.0,
        max_structural_similarity=1.0,
        max_potency_loss=1.0,
        max_candidates=1,
        material_track_fraction=0.0,
        max_3d_conformers=0,
    )
    assert rows[0]["safety_score_delta_vs_parent"] == pytest.approx(0.0)


def test_interactive_html_contains_every_selected_candidate(tmp_path: Path) -> None:
    args = _args(tmp_path)
    report = discovery.run(args)
    template = report["candidates"][0]
    report["candidates"] = [
        {
            **template,
            "rank": index,
            "global_priority_rank": index,
            "candidate_id": f"SUB{index:04d}",
            "name": f"Candidate {index}",
            "inchikey": f"INCHIKEY-{index}",
            "alternate_names": [f"Alias {index}"],
        }
        for index in range(1, 36)
    ]
    report["summary"]["direct_activity_candidates"] = 35
    report["summary"]["direct_retained_candidates"] = 35
    report["summary"]["cosing_candidates"] = 35

    output = tmp_path / "all-candidates.html"
    discovery._write_html(output, report)
    document = output.read_text()

    assert document.count("<article class='candidate'") == 35
    assert "Candidate 35" in document
    assert "alias 35" in document
    assert "Showing 35 of 35 candidates" in document


def test_discovery_is_deterministic_except_generation_timestamp(tmp_path: Path) -> None:
    args = _args(tmp_path)

    first = discovery.run(args)
    args.out_dir = tmp_path / "out_second"
    second = discovery.run(args)

    assert first["candidates"] == second["candidates"]
    assert first["source_audit"] == second["source_audit"]
    assert first["thresholds"] == second["thresholds"]


def test_3d_pharmacophore_metrics_are_deterministic(tmp_path: Path) -> None:
    args = _args(tmp_path)

    first = discovery.run(args)
    args.out_dir = tmp_path / "out_second"
    second = discovery.run(args)

    first_by_smiles = {row["smiles"]: row for row in first["candidates"]}
    second_by_smiles = {row["smiles"]: row for row in second["candidates"]}
    for smiles, first_row in first_by_smiles.items():
        second_row = second_by_smiles[smiles]
        for key in (
            "pharmacophore_3d_status",
            "pharmacophore_3d_parent_conformer_count",
            "pharmacophore_3d_candidate_conformer_count",
            "pharmacophore_3d_matched_feature_count",
            "pharmacophore_3d_parent_feature_count",
            "pharmacophore_3d_feature_family_recall",
            "pharmacophore_3d_feature_distance_rmsd",
            "strict_3d_pharmacophore_gate_passed",
            "pharmacophore_3d_evaluated",
            "pharmacophore_3d_evaluation_rank",
            "pareto_front",
        ):
            assert first_row[key] == second_row[key]
        assert first_row["pharmacophore_3d_candidate_conformer_count"] <= 50


def test_3d_feature_recall_rmsd_and_threshold_boundaries() -> None:
    def feature(family: str, xyz: tuple[float, float, float], atom_id: int):
        return {
            "family": family,
            "atom_ids": (atom_id,),
            "position": SimpleNamespace(x=xyz[0], y=xyz[1], z=xyz[2]),
        }

    parent = [
        feature("Donor", (0.0, 0.0, 0.0), 0),
        feature("Acceptor", (0.0, 0.0, 0.0), 1),
        feature("Aromatic", (0.0, 0.0, 0.0), 2),
        feature("Hydrophobe", (0.0, 0.0, 0.0), 3),
    ]
    candidate = [
        feature("Donor", (1.0, 0.0, 0.0), 10),
        feature("Acceptor", (0.0, 2.0, 0.0), 11),
        feature("Aromatic", (0.0, 0.0, 2.0), 12),
    ]

    matched, rmsd = discovery._matched_feature_rmsd(parent, candidate)
    assert matched == 3
    assert matched / len(parent) == pytest.approx(0.75)
    assert rmsd == pytest.approx(3.0**0.5)
    assert discovery._matched_feature_rmsd(parent[:2], candidate[:2]) == (2, None)

    row = {
        "pharmacophore_gate_passed": True,
        "target_conditioned_anchor_score": None,
        "admission_bases": ["strict_2d_pharmacophore"],
    }
    result = {
        "status": "available",
        "basis": "etkdg_v3_mmff_ligand_feature_alignment",
        "parent_conformer_count": 1,
        "candidate_conformer_count": 1,
        "parent_conformer_status": "etkdg_v3_mmff",
        "candidate_conformer_status": "etkdg_v3_mmff",
        "matched_feature_count": 3,
        "parent_feature_count": 4,
        "feature_family_recall": 0.75,
        "feature_distance_rmsd": 0.5,
    }
    discovery._apply_pharmacophore_3d_result(
        row,
        result,
        evaluated=True,
        evaluation_rank=1,
        min_3d_feature_recall=0.75,
        max_feature_distance_rmsd=0.5,
    )
    assert row["strict_3d_pharmacophore_gate_passed"] is True
    assert "strict_3d_pharmacophore" in row["admission_bases"]

    below = {
        "pharmacophore_gate_passed": True,
        "target_conditioned_anchor_score": None,
        "admission_bases": ["strict_2d_pharmacophore"],
    }
    discovery._apply_pharmacophore_3d_result(
        below,
        {**result, "feature_family_recall": 0.749999},
        evaluated=True,
        evaluation_rank=1,
        min_3d_feature_recall=0.75,
        max_feature_distance_rmsd=0.5,
    )
    assert below["strict_3d_pharmacophore_gate_passed"] is False


def test_3d_candidate_budget_bounds_calls_and_is_input_order_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = discovery._standardize_smiles("CCO", "parent")
    identities = [
        discovery._standardize_smiles(smiles, "candidate")
        for smiles in ("CCN", "CCC", "CCCl", "CCF", "CCS")
    ]

    def make_seeds(
        ordered_identities: list[discovery.MoleculeIdentity],
    ) -> dict[str, discovery.CandidateSeed]:
        return {
            identity.smiles: discovery.CandidateSeed(
                smiles=identity.smiles,
                inchikey=identity.inchikey,
                names={identity.smiles},
            )
            for identity in ordered_identities
        }

    calls: list[str] = []

    def fake_parent_ensemble(
        _molecule: Chem.Mol,
        *,
        seed: int,
        max_conformers: int,
    ) -> tuple[None, list[int], str]:
        assert isinstance(seed, int)
        assert max_conformers == 1
        return None, [], "fixture_parent"

    def fake_3d_comparison(
        _parent_mol: Chem.Mol,
        candidate_mol: Chem.Mol,
        _factory: object,
        _parent_ensemble: object,
        *,
        seed: int,
        max_conformers: int,
        parent_features_by_conformer: object,
    ) -> dict[str, object]:
        assert isinstance(seed, int)
        assert max_conformers == 1
        assert parent_features_by_conformer == {}
        calls.append(Chem.MolToSmiles(candidate_mol, canonical=True))
        return {
            "status": "available",
            "basis": "fixture_ligand_feature_alignment",
            "parent_conformer_count": 1,
            "candidate_conformer_count": 1,
            "parent_conformer_status": "fixture_parent",
            "candidate_conformer_status": "fixture_candidate",
            "matched_feature_count": 3,
            "parent_feature_count": 3,
            "feature_family_recall": 1.0,
            "feature_distance_rmsd": 0.5,
        }

    monkeypatch.setattr(discovery, "_conformer_ensemble", fake_parent_ensemble)
    monkeypatch.setattr(
        discovery,
        "_pharmacophore_3d_comparison",
        fake_3d_comparison,
    )

    def score(
        ordered_identities: list[discovery.MoleculeIdentity],
        *,
        budget: int = 2,
    ) -> tuple[list[dict[str, object]], dict[str, object], list[str]]:
        calls.clear()
        rows, scoring = discovery._score_candidates(
            parent,
            make_seeds(ordered_identities),
            min_pharmacophore=0.0,
            min_feature_recall=0.0,
            max_structural_similarity=1.0,
            max_potency_loss=1.0,
            max_candidates=4,
            material_track_fraction=0.40,
            max_3d_conformers=1,
            max_3d_candidates=budget,
        )
        return rows, scoring, list(calls)

    first_rows, first_scoring, first_calls = score(identities)
    second_rows, second_scoring, second_calls = score(list(reversed(identities)))

    assert len(first_calls) == 2
    assert first_calls == second_calls
    assert [row["smiles"] for row in first_rows] == [
        row["smiles"] for row in second_rows
    ]
    assert [row["pharmacophore_3d_status"] for row in first_rows] == [
        row["pharmacophore_3d_status"] for row in second_rows
    ]
    expected_counts = {
        "budget": 2,
        "eligible_candidates": 5,
        "evaluated_candidates": 2,
        "budget_excluded_candidates": 3,
        "available_evidence_candidates": 2,
    }
    for key, expected in expected_counts.items():
        assert first_scoring["pharmacophore_3d_evaluation"][key] == expected
        assert second_scoring["pharmacophore_3d_evaluation"][key] == expected

    evaluated = [row for row in first_rows if row["pharmacophore_3d_evaluated"]]
    budget_excluded = [
        row for row in first_rows if not row["pharmacophore_3d_evaluated"]
    ]
    assert len(evaluated) == 2
    assert all(row["strict_3d_pharmacophore_gate_passed"] for row in evaluated)
    assert budget_excluded
    for row in budget_excluded:
        assert row["pharmacophore_3d_status"] == "not_evaluated_budget"
        assert row["pharmacophore_3d_feature_family_recall"] is None
        assert row["pharmacophore_3d_feature_distance_rmsd"] is None
        assert row["strict_3d_pharmacophore_gate_passed"] is False
        assert row["pharmacophore_gate_passed"] is True
        assert row["hypothesis_only"] is True

    covered_rows, covered_scoring, covered_calls = score(identities, budget=4)
    assert len(covered_calls) == 4
    assert all(row["pharmacophore_3d_evaluated"] for row in covered_rows)
    assert covered_scoring["pharmacophore_3d_evaluation"][
        "budget_excluded_candidates"
    ] == 1


@pytest.mark.parametrize("budget", [0, 501])
def test_3d_candidate_budget_range_is_validated(
    tmp_path: Path,
    budget: int,
) -> None:
    args = _args(tmp_path)
    args.max_3d_candidates = budget

    with pytest.raises(SystemExit, match="--max-3d-candidates must be in \\[1, 500\\]"):
        discovery._validate_args(args)


@pytest.mark.parametrize("conformers", [-1, 51])
def test_3d_conformer_cap_is_validated(
    tmp_path: Path,
    conformers: int,
) -> None:
    args = _args(tmp_path)
    args.max_3d_conformers = conformers

    with pytest.raises(SystemExit, match="--max-3d-conformers must be in \\[0, 50\\]"):
        discovery._validate_args(args)


def test_unavailable_3d_evidence_fails_strict_gate_without_erasing_hypothesis() -> None:
    parent = discovery._standardize_smiles("CCO", "parent")
    candidate = discovery._standardize_smiles("CCN", "candidate")
    seeds = {
        candidate.smiles: discovery.CandidateSeed(
            smiles=candidate.smiles,
            inchikey=candidate.inchikey,
            names={"Ethylamine"},
        ),
    }

    rows, _scoring = discovery._score_candidates(
        parent,
        seeds,
        min_pharmacophore=0.0,
        min_feature_recall=0.0,
        max_structural_similarity=0.999,
        max_potency_loss=1.0,
        max_candidates=20,
        material_track_fraction=0.40,
        max_3d_conformers=0,
    )

    assert len(rows) == 1
    candidate_row = rows[0]
    assert candidate_row["pharmacophore_gate_passed"] is True
    # 컨포머 수 0으로 돌렸으므로 실패는 **입력 분자 쪽**이다. 후보 쪽 실패와
    # 뭉쳐 놓으면 화면이 후보마다 "이 후보를 판정하지 못했다"고 말하는데,
    # 실제로는 후보를 하나도 볼 수 없었던 것이다.
    assert candidate_row["pharmacophore_3d_status"] == "parent_unavailable"
    assert candidate_row["pharmacophore_3d_feature_family_recall"] is None
    assert candidate_row["pharmacophore_3d_feature_distance_rmsd"] is None
    assert candidate_row["strict_3d_pharmacophore_gate_passed"] is False
    assert candidate_row["claimable"] is False
    assert candidate_row["hypothesis_only"] is True


def test_discovery_fails_closed_on_unusable_activity_contract(tmp_path: Path) -> None:
    library = tmp_path / "cosing.parquet"
    activity = tmp_path / "activity.parquet"
    out_dir = tmp_path / "out"
    _write_library(library)
    pd.DataFrame(
        [
            {
                "uniprot": TARGET,
                "ligand_smiles": "COc1ccccc1",
                "affinity_value": 1.0,
                "affinity_unit": "nM",
            }
        ]
    ).to_parquet(activity)
    args = discovery.parse_args(
        [
            "--parent-smiles",
            PARENT,
            "--target-id",
            TARGET,
            "--candidate-library",
            str(library),
            "--activity-evidence",
            str(activity),
            "--out-dir",
            str(out_dir),
        ]
    )

    with pytest.raises(SystemExit, match="required normalized fields"):
        discovery.run(args)

    assert not any((out_dir / filename).exists() for filename in discovery.OUTPUT_FILES)


def test_discovery_refuses_to_overwrite_existing_bundle(tmp_path: Path) -> None:
    args = _args(tmp_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stale = out_dir / discovery.OUTPUT_FILES[0]
    stale.write_text("stale\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="refusing to overwrite"):
        discovery.run(args)

    assert stale.read_text(encoding="utf-8") == "stale\n"


def test_activity_identity_mismatch_is_rejected_instead_of_reassigned(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    activity = Path(args.activity_evidence[0])
    table = pd.read_parquet(activity)
    anisole = table["ligand_smiles"] == "COc1ccccc1"
    table.loc[anisole, "ligand_inchikey"] = _inchikey(PARENT)
    table.to_parquet(activity)

    report = discovery.run(args)

    anisole_row = next(row for row in report["candidates"] if row["name"] == "Anisole")
    assert anisole_row["evidence_tier"] == "proxy_only"
    assert anisole_row["activity_evidence_count"] == 0
    audit = report["source_audit"]["activity_evidence"][0]
    assert audit["identity_mismatch_rows"] == 1
    assert report["source_audit"]["activity_selection"][0][
        "rejected_identity_mismatch"
    ] == 1


def test_retention_requires_matching_activity_type_and_source(tmp_path: Path) -> None:
    args = _args(tmp_path)
    activity = Path(args.activity_evidence[0])
    table = pd.read_parquet(activity)
    anisole = table["ligand_smiles"] == "COc1ccccc1"
    table.loc[anisole, "affinity_type"] = "EC50"
    table.to_parquet(activity)

    report = discovery.run(args)

    anisole_row = next(row for row in report["candidates"] if row["name"] == "Anisole")
    assert anisole_row["evidence_tier"] == "direct_activity"
    assert anisole_row["binding_retained"] is None
    assert anisole_row["binding_pactivity_delta"] is None
    assert anisole_row["binding_comparison_strata"] == []
    assert (
        anisole_row["binding_status"]
        == "same_target_activity_supported_no_comparable_parent_stratum"
    )


def test_activity_identity_rejects_matching_connectivity_with_detail_difference(
    tmp_path: Path,
) -> None:
    activity = tmp_path / "stereo_activity.parquet"
    stereo_smiles = "C[C@H](O)C(=O)O"
    nonstereo_key = _inchikey("CC(O)C(=O)O")
    pd.DataFrame(
        [
            {
                "uniprot": TARGET,
                "ligand_smiles": stereo_smiles,
                "ligand_inchikey": nonstereo_key,
                "affinity_value": 100.0,
                "affinity_unit": "nM",
                "relation": "=",
                "censor": False,
                "source_db": "FixtureDB",
                "affinity_type": "Ki",
            }
        ]
    ).to_parquet(activity)
    parent = discovery._standardize_smiles(PARENT, "parent")

    seeds, audits, summary = discovery._target_activity_candidates(
        [activity],
        TARGET,
        parent,
        max_direct_molecules=10,
    )

    assert seeds == {}
    assert audits[0]["identity_mismatch_rows"] == 0
    assert audits[0]["identity_detail_mismatch_rows"] == 1
    assert summary[0]["rejected_identity_mismatch"] == 1
    assert summary[0]["rejected_identity_detail_mismatch"] == 1


def _activity_record(pactivity: float) -> dict[str, object]:
    return {
        "pactivity": pactivity,
        "source": "FixtureDB",
        "release": "1",
        "activity_type": "KI",
        "doi": "10.1000/fixture",
        "pmid": "1",
    }


def test_same_target_feature_analogue_is_admitted_without_pharmacophore_claim() -> None:
    parent = discovery._standardize_smiles(
        "CC1=C(/C=C/C(C)=C/C=C/C(C)=C/C(=O)O)C(C)(C)CCC1",
        "parent",
    )
    adapalene = discovery._standardize_smiles(
        "COc1ccc(-c2ccc3cc(C(=O)O)ccc3c2)cc1C12CC3CC(CC(C3)C1)C2",
        "candidate",
    )
    seeds = {
        parent.smiles: discovery.CandidateSeed(
            smiles=parent.smiles,
            inchikey=parent.inchikey,
            activities=[_activity_record(8.0)],
        ),
        adapalene.smiles: discovery.CandidateSeed(
            smiles=adapalene.smiles,
            inchikey=adapalene.inchikey,
            names={"Adapalene"},
            sources={"FixtureDB"},
            activities=[_activity_record(7.5)],
        ),
    }

    rows, _scoring = discovery._score_candidates(
        parent,
        seeds,
        min_pharmacophore=0.55,
        min_feature_recall=0.60,
        max_structural_similarity=0.98,
        max_potency_loss=1.0,
        max_candidates=20,
        material_track_fraction=0.40,
    )

    assert len(rows) == 1
    candidate = rows[0]
    assert candidate["name"] == "Adapalene"
    assert candidate["pharmacophore_preservation_score"] < 0.55
    assert candidate["feature_family_recall"] >= 0.60
    assert candidate["pharmacophore_gate_passed"] is False
    assert candidate["admission_bases"] == [
        "same_target_activity_feature_family"
    ]
    assert candidate["evidence_tier"] == "direct_retained"
    assert candidate["claimable"] is False
    assert candidate["wet_lab_required"] is True


def test_target_conditioned_anchor_score_distinguishes_retained_feature() -> None:
    parent = discovery._standardize_smiles("CC(=O)O", "parent")
    retained = discovery._standardize_smiles("CCC(=O)O", "retained")
    lost = discovery._standardize_smiles("CCN", "lost")
    seeds = {
        retained.smiles: discovery.CandidateSeed(
            smiles=retained.smiles,
            inchikey=retained.inchikey,
            names={"Retained"},
        ),
        lost.smiles: discovery.CandidateSeed(
            smiles=lost.smiles,
            inchikey=lost.inchikey,
            names={"Lost"},
        ),
    }

    rows, scoring = discovery._score_candidates(
        parent,
        seeds,
        min_pharmacophore=0.0,
        min_feature_recall=0.0,
        max_structural_similarity=0.999,
        max_potency_loss=1.0,
        max_candidates=20,
        material_track_fraction=0.40,
        interaction_anchor_payload=_anchor_payload(parent.smiles, [2, 3]),
        target_id=TARGET,
    )

    by_name = {row["name"]: row for row in rows}
    assert by_name["Retained"]["target_conditioned_anchor_score"] == 1.0
    assert by_name["Retained"]["target_conditioned_preserved_anchor_count"] == 2
    assert by_name["Retained"]["target_conditioned_anchor_basis"] == (
        "pose_supported_parent_anchor_conservative_mcs_feature_preservation"
    )
    assert isinstance(by_name["Retained"]["target_conditioned_mapping_count"], int)
    assert by_name["Retained"]["target_conditioned_mapping_count"] >= 1
    assert isinstance(by_name["Retained"]["target_conditioned_mapping_ambiguous"], bool)
    assert by_name["Retained"]["target_conditioned_mapping_truncated"] is False
    assert by_name["Lost"]["target_conditioned_anchor_score"] == 0.0
    assert by_name["Lost"]["target_conditioned_anchor_gate_passed"] is False
    assert by_name["Lost"]["target_conditioned_strict_gate_passed"] is False
    assert "target_conditioned" not in by_name["Lost"]["selection_tracks"]
    assert by_name["Lost"]["target_conditioned_preserved_anchor_count"] == 0
    assert by_name["Lost"]["target_conditioned_anchor_basis"] == (
        "pose_supported_parent_anchor_conservative_mcs_feature_preservation"
    )
    assert isinstance(by_name["Lost"]["target_conditioned_mapping_count"], int)
    assert by_name["Lost"]["target_conditioned_mapping_count"] >= 1
    assert isinstance(by_name["Lost"]["target_conditioned_mapping_ambiguous"], bool)
    assert by_name["Lost"]["target_conditioned_mapping_truncated"] is False
    assert scoring["parent"]["target_conditioned_interaction_anchor"]["enabled"] is True


def test_pareto_front_keeps_tradeoff_candidates_non_dominated() -> None:
    rows = [
        {
            "name": "High pharmacophore",
            "pharmacophore_preservation_score": 0.95,
            "binding_support_score": 0.40,
            "safety_triage_score": 0.80,
            "routeability_proxy": 0.80,
            "corpus_novelty_proxy": 0.40,
        },
        {
            "name": "High novelty",
            "pharmacophore_preservation_score": 0.60,
            "binding_support_score": 0.40,
            "safety_triage_score": 0.80,
            "routeability_proxy": 0.80,
            "corpus_novelty_proxy": 0.90,
        },
        {
            "name": "Dominated",
            "pharmacophore_preservation_score": 0.50,
            "binding_support_score": 0.40,
            "safety_triage_score": 0.70,
            "routeability_proxy": 0.70,
            "corpus_novelty_proxy": 0.30,
        },
    ]

    discovery._assign_pareto_fronts(rows)

    by_name = {row["name"]: row for row in rows}
    assert by_name["High pharmacophore"]["pareto_front"] == 1
    assert by_name["High novelty"]["pareto_front"] == 1
    assert by_name["Dominated"]["pareto_front"] == 2
    assert by_name["Dominated"]["pareto_dominated_by_count"] == 2


def test_pareto_front_is_deterministic_for_equal_candidates() -> None:
    baseline = {
        "pharmacophore_preservation_score": 0.80,
        "binding_support_score": 0.50,
        "safety_triage_score": 0.70,
        "routeability_proxy": 0.60,
        "corpus_novelty_proxy": 0.40,
    }
    rows = [
        {"name": "Equal B", **baseline},
        {"name": "Better binding", **baseline, "binding_support_score": 0.60},
        {"name": "Equal A", **baseline},
    ]

    discovery._assign_pareto_fronts(rows)

    by_name = {row["name"]: row for row in rows}
    assert by_name["Better binding"]["pareto_front"] == 1
    assert by_name["Better binding"]["pareto_dominated_by_count"] == 0
    assert by_name["Equal A"]["pareto_front"] == 2
    assert by_name["Equal B"]["pareto_front"] == 2
    assert by_name["Equal A"]["pareto_dominated_by_count"] == 1
    assert by_name["Equal B"]["pareto_dominated_by_count"] == 1


@pytest.mark.parametrize("bad_value", [None, float("nan"), float("inf")])
def test_pareto_front_rejects_missing_or_nonfinite_objectives(
    bad_value: object,
) -> None:
    rows = [
        {
            "name": "Invalid",
            "pharmacophore_preservation_score": bad_value,
            "binding_support_score": 0.50,
            "safety_triage_score": 0.70,
            "routeability_proxy": 0.60,
            "corpus_novelty_proxy": 0.40,
        }
    ]

    with pytest.raises(SystemExit, match="Pareto objective"):
        discovery._assign_pareto_fronts(rows)


def test_stereo_aware_novelty_keeps_defined_isomer_and_rejects_unspecified_record() -> None:
    parent = discovery._standardize_smiles(
        "CC1=C(/C=C/C(C)=C/C=C/C(C)=C/C(=O)O)C(C)(C)CCC1",
        "parent",
    )
    cis_isomer = discovery._standardize_smiles(
        "CC1=C(/C=C/C(C)=C\\C=C\\C(C)=C\\C(=O)O)C(C)(C)CCC1",
        "cis candidate",
    )
    unspecified = discovery._standardize_smiles(
        "CC(C=CC1=C(C)CCCC1(C)C)=CC=CC(C)=CC(=O)O",
        "unspecified candidate",
    )
    seeds = {
        parent.smiles: discovery.CandidateSeed(
            smiles=parent.smiles,
            inchikey=parent.inchikey,
            activities=[_activity_record(8.0)],
        ),
        cis_isomer.smiles: discovery.CandidateSeed(
            smiles=cis_isomer.smiles,
            inchikey=cis_isomer.inchikey,
            names={"Alitretinoin"},
            activities=[_activity_record(7.5)],
        ),
        unspecified.smiles: discovery.CandidateSeed(
            smiles=unspecified.smiles,
            inchikey=unspecified.inchikey,
            names={"Unspecified retinoic acid"},
            activities=[_activity_record(7.5)],
        ),
    }

    rows, scoring = discovery._score_candidates(
        parent,
        seeds,
        min_pharmacophore=0.55,
        min_feature_recall=0.60,
        max_structural_similarity=0.98,
        max_potency_loss=1.0,
        max_candidates=20,
        material_track_fraction=0.40,
    )

    assert [row["name"] for row in rows] == ["Alitretinoin"]
    assert rows[0]["pharmacophore_gate_passed"] is True
    assert rows[0]["structural_similarity_to_parent"] < 0.98
    assert scoring["exclusions"]["stereo_underspecified"] == 1


def test_alias_enrichment_requires_exact_full_inchikey(tmp_path: Path) -> None:
    all_trans = discovery._standardize_smiles(
        "CC1=C(/C=C/C(C)=C/C=C/C(C)=C/C(=O)O)C(C)(C)CCC1",
        "all-trans",
    )
    cis_isomer = discovery._standardize_smiles(
        "CC1=C(/C=C/C(C)=C\\C=C\\C(C)=C\\C(=O)O)C(C)(C)CCC1",
        "cis",
    )
    alias_path = tmp_path / "aliases.parquet"
    pd.DataFrame(
        [
            {"alias": "TRETINOIN", "inchikey": all_trans.inchikey},
            {"alias": "ALITRETINOIN", "inchikey": cis_isomer.inchikey},
            {"alias": "ALITRETINOIN", "inchikey": cis_isomer.inchikey},
            {"alias": "CHEMBL705", "inchikey": cis_isomer.inchikey},
        ]
    ).to_parquet(alias_path)
    seed = discovery.CandidateSeed(
        smiles=cis_isomer.smiles,
        inchikey=cis_isomer.inchikey,
        names={"numeric-id"},
    )

    audits = discovery._enrich_seed_names({cis_isomer.smiles: seed}, [alias_path])

    assert seed.preferred_name == "ALITRETINOIN"
    assert "TRETINOIN" not in seed.names
    assert audits[0]["matched_rows"] == 3
    assert audits[0]["identity_contract"] == "exact_full_inchikey"


def test_artifact_generation_failure_removes_partial_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    out_dir = Path(args.out_dir)

    def fail_html(_path: Path, _report: dict[str, object]) -> None:
        raise OSError("fixture write failure")

    monkeypatch.setattr(discovery, "_write_html", fail_html)

    with pytest.raises(OSError, match="fixture write failure"):
        discovery.run(args)

    assert not any((out_dir / filename).exists() for filename in discovery.OUTPUT_FILES)
    assert not list(out_dir.parent.glob(f".{out_dir.name}.staging-*"))


def test_feature_fallback_penalizes_unmatched_extra_feature_families() -> None:
    parent = Counter({"Donor": 1, "Acceptor": 1, "Hydrophobe": 2})
    exact = Counter(parent)
    extra = Counter({"Donor": 1, "Acceptor": 4, "Hydrophobe": 2})

    exact_recall, exact_precision, exact_f1 = discovery._feature_overlap_scores(
        parent, exact
    )
    extra_recall, extra_precision, extra_f1 = discovery._feature_overlap_scores(
        parent, extra
    )

    assert (exact_recall, exact_precision, exact_f1) == (1.0, 1.0, 1.0)
    assert extra_recall == 1.0
    assert extra_precision < exact_precision
    assert extra_f1 < exact_f1


def test_candidate_basket_reserves_cosmetic_track_without_hiding_global_rank() -> None:
    rows = [
        {
            "smiles": "target-a",
            "priority_score": 0.95,
            "evidence_tier": "direct_activity",
            "cosing_reference": False,
        },
        {
            "smiles": "target-b",
            "priority_score": 0.90,
            "evidence_tier": "direct_activity",
            "cosing_reference": False,
        },
        {
            "smiles": "target-c",
            "priority_score": 0.85,
            "evidence_tier": "direct_activity",
            "cosing_reference": False,
        },
        {
            "smiles": "material-a",
            "priority_score": 0.60,
            "evidence_tier": "proxy_only",
            "cosing_reference": True,
        },
        {
            "smiles": "material-b",
            "priority_score": 0.55,
            "evidence_tier": "proxy_only",
            "cosing_reference": True,
        },
    ]

    selected, strategy = discovery._select_candidate_basket(
        rows,
        max_candidates=4,
        material_track_fraction=0.50,
    )

    assert [row["smiles"] for row in selected] == [
        "target-a",
        "target-b",
        "material-a",
        "material-b",
    ]
    assert [row["global_priority_rank"] for row in selected] == [1, 2, 4, 5]
    assert strategy["mode"] == "balanced_tracks"
    assert strategy["reserved_material_slots"] == 2
    assert strategy["selected_counts"] == {
        "all": 4,
        "target_activity": 2,
        "cosmetic_material": 2,
        "strict_pharmacophore": 0,
        "strict_3d_pharmacophore": 0,
        "strict_target_conditioned": 0,
        "both": 0,
    }


def test_candidate_basket_reserves_strict_pharmacophore_hits() -> None:
    rows = [
        {
            "smiles": f"target-{index}",
            "priority_score": 0.95 - index * 0.05,
            "evidence_tier": "direct_activity",
            "cosing_reference": False,
            "pharmacophore_gate_passed": False,
        }
        for index in range(4)
    ]
    rows.append(
        {
            "smiles": "strict-low-score",
            "priority_score": 0.30,
            "evidence_tier": "direct_reduced",
            "cosing_reference": False,
            "pharmacophore_gate_passed": True,
        }
    )

    selected, strategy = discovery._select_candidate_basket(
        rows,
        max_candidates=3,
        material_track_fraction=0.0,
        pharmacophore_track_fraction=0.34,
    )

    assert [row["smiles"] for row in selected] == [
        "target-0",
        "target-1",
        "strict-low-score",
    ]
    assert strategy["reserved_pharmacophore_slots"] == 1
    assert strategy["selected_counts"]["strict_pharmacophore"] == 1


def test_conformers_degrade_instead_of_crashing_when_mmff_has_no_parameters() -> None:
    """MMFF에 파라미터가 없는 원소를 만나면 에너지 없이 임베딩 순서를 쓴다.

    두 갈래가 `results = []`로 두고 상태 문자열까지 준비해 두었는데, 바로 아래
    strict=True zip 이 그 자리에서 터져서 그 상태가 반환된 적이 없었다. 등재 원료
    507종 중 3종(유기주석)이 실제로 여기서 죽었고, 5.6b도 같은 후보를 만나면
    같은 자리에서 죽는다.
    """
    from rdkit import Chem

    # 주석은 MMFF94s 파라미터가 없다.
    organotin = Chem.MolFromSmiles("CCCC[Sn](CCCC)(CCCC)OC(C)=O")
    assert organotin is not None

    molecule, conformer_ids, status = discovery._conformer_ensemble(
        organotin, seed=7, max_conformers=5
    )

    assert molecule is not None, "임베딩은 됐는데 최적화 실패로 통째로 버리면 안 된다"
    assert conformer_ids, "컨포머가 남아 있어야 한다"
    assert status == "etkdg_v3_mmff_unavailable", (
        "MMFF가 돌지 않았다는 사실이 상태로 나와야 한다"
    )


def test_conformers_still_order_by_energy_when_mmff_runs() -> None:
    """정상 경로가 망가지지 않았는지. 위 수정이 에너지 정렬을 끄면 안 된다."""
    from rdkit import Chem

    flexible = Chem.MolFromSmiles("CCCCCCCCO")
    molecule, conformer_ids, status = discovery._conformer_ensemble(
        flexible, seed=7, max_conformers=5
    )
    assert status == "etkdg_v3_mmff"
    assert len(conformer_ids) >= 1
    assert len(set(conformer_ids)) == len(conformer_ids)


def test_every_3d_status_is_accounted_for_downstream() -> None:
    """3D 상태를 하나 늘릴 때마다 집계가 조용히 작아지지 않게 한다.

    `pharmacophore_3d_unavailable_candidates` 는 예전에 `== "unavailable"` 하나만
    셌다. 그 뒤로 상태가 `parent_unavailable`, `budget_exhausted`, `recall_only`
    까지 늘었는데, 늘어난 상태의 후보는 3D 근거가 없는데도 그 수에 들어가지
    않았다. 새 상태를 만들면 이 테스트가 먼저 걸린다.
    """
    produced = {
        # _pharmacophore_3d_comparison 이 낼 수 있는 값 전부
        "available",
        "recall_only",
        "unavailable",
        "parent_unavailable",
        "budget_exhausted",
        # 후보 예산 밖에서 붙는 값
        "not_evaluated_budget",
    }
    unaccounted = produced - discovery.NO_3D_EVIDENCE_STATUSES - {"available"}
    assert not unaccounted, (
        f"3D 근거 없음으로 세지 않는 상태가 있습니다: {sorted(unaccounted)}"
    )
    # "available" 은 근거가 있는 유일한 상태여야 한다. recall_only 는 회수율만
    # 잰 것이고 RMSD 가 없으므로 엄격 게이트를 통과할 수 없다.
    assert "available" not in discovery.NO_3D_EVIDENCE_STATUSES
    assert "recall_only" in discovery.NO_3D_EVIDENCE_STATUSES


def test_a_recall_only_row_carries_its_measured_recall_but_fails_the_strict_gate() -> None:
    """맞은 특징이 3개 미만이면 RMSD 는 정의되지 않지만 회수율은 잰 값이다.

    예전에는 그때 회수율까지 버려서, "질의의 파마코포어 특징을 하나도 공유하지
    않는다"는 가장 강한 부정 근거가 "아무도 재지 못했다"와 똑같이 표시됐다.
    """
    row: dict[str, object] = {
        "admission_bases": [],
        "target_conditioned_anchor_score": None,
        "pharmacophore_gate_passed": True,
    }
    result = {
        "status": "recall_only",
        "basis": "etkdg_v3_mmff_ligand_feature_alignment",
        "parent_conformer_count": 3,
        "candidate_conformer_count": 3,
        "parent_conformer_status": "etkdg_v3_mmff",
        "candidate_conformer_status": "etkdg_v3_mmff",
        "matched_feature_count": 0,
        "parent_feature_count": 6,
        "feature_family_recall": 0.0,
        "feature_distance_rmsd": None,
    }
    discovery._apply_pharmacophore_3d_result(
        row, result, evaluated=True, evaluation_rank=1,
        min_3d_feature_recall=0.0, max_feature_distance_rmsd=2.0,
    )
    assert row["pharmacophore_3d_feature_family_recall"] == 0.0, (
        "잰 0.0 이 None 으로 지워지면 안 됩니다"
    )
    assert row["pharmacophore_3d_feature_distance_rmsd"] is None
    assert row["strict_3d_pharmacophore_gate_passed"] is False, (
        "RMSD 가 없으면 엄격 게이트는 통과할 수 없습니다"
    )
