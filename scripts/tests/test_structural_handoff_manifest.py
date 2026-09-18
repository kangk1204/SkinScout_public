"""C22 regression: the structural handoff loader must consume the sealed
structural manifest, join pairs 1:1, and verify pose/source/safety hashes."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import build_target_lab_package as package

STATUS_FIELDS = [
    "pair_id",
    "intent_id",
    "target_id",
    "compound_id",
    "control_role",
    "status",
    "reason",
    "safety_status",
    "applicability_status",
    "safety_consensus_valid",
    "full_analysis_complete",
    "structure_protocol_status",
    "structure_protocol_evidence_sha256",
    "safety_evidence_sha256",
    "safety_evidence_record_sha256",
    "ligand_graph_smiles",
    "ligand_inchi_key",
    "gnina_cnn_affinity",
    "gnina_cnn_score",
    "gnina_minimized_affinity",
    "pose_sdf",
    "pose_sha256",
]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _structural_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "structural"
    inputs = directory / "inputs"
    inputs.mkdir(parents=True)
    ligand = inputs / "ligand.sdf"
    ligand.write_bytes(b"CCO ligand placeholder\n")
    receptor = inputs / "receptor.pdb"
    receptor.write_text("ATOM      1  N   HIS A  57\nEND\n", encoding="utf-8")
    pocket = inputs / "pocket.json"
    _write_json(pocket, {"pockets": [{"rank": 1, "center": [1.0, 2.0, 3.0], "radius": 8.0}]})
    protocol = inputs / "redocking.json"
    _write_json(
        protocol,
        {
            "schema_version": "test.redocking.v1",
            "target_id": "P_TEST",
            "receptor_sha256": package.digest(receptor),
            "pocket_sha256": package.digest(pocket),
            "redocking_pass": True,
        },
    )
    pose = directory / "artifacts" / "pair_x" / "pose_test.sdf"
    pose.parent.mkdir(parents=True)
    pose.write_bytes(b"pose bytes\n")
    exact = package.identity("CCO")
    control = {
        "control_role": "active_control",
        "molecule_chembl_id": "CHEMBL_TEST",
        "structure_compound_id": exact["compound_id"],
        "standard_inchi_key": exact["computed_inchikey"],
        "smiles": "CCO",
        "activity_id": "12345",
        "assay_chembl_id": "CHEMBL_ASSY",
        "standard_type": "Ki",
        "standard_relation": "=",
        "standard_value": 300.0,
        "standard_units": "nM",
        "doi": "10.0000/test",
        "ligand_sdf": "inputs/ligand.sdf",
        "ligand_sdf_sha256": package.digest(ligand),
    }
    source_manifest = {
        "schema_version": "skinscout.target-structural-source-records.v1",
        "target": {"gene": "TEST", "uniprot": "P_TEST", "intent_id": "TI-030-KLK5"},
        "controls": [control],
    }
    source_path = directory / "source_record_manifest.json"
    _write_json(source_path, source_manifest)
    pair = {
        "pair_id": "pair_x",
        "intent_id": "TI-030-KLK5",
        "target_id": "P_TEST",
        "compound_id": exact["compound_id"],
        "control_role": "active_control",
        "control_evidence_id": "CHEMBL_ASSY:activity:12345",
        "status": "completed",
        "reason": "test execution",
        "safety_status": "PASS",
        "applicability_status": "inside",
        "safety_consensus_valid": "true",
        "full_analysis_complete": "true",
        "structure_protocol_status": "qualified",
        "structure_protocol_evidence_sha256": package.digest(protocol),
        "safety_evidence_sha256": "a" * 64,
        "safety_evidence_record_sha256": "b" * 64,
        "ligand_graph_smiles": exact["canonical_isomeric_smiles"],
        "ligand_inchi_key": exact["computed_inchikey"],
        "gnina_cnn_affinity": 6.25,
        "gnina_cnn_score": 0.78,
        "gnina_minimized_affinity": -7.1,
        "pose_sdf": str(pose),
        "pose_sha256": package.digest(pose),
        "ligand_sdf": str(ligand),
        "ligand_sha256": package.digest(ligand),
        "receptor_pdb": str(receptor),
        "receptor_sha256": package.digest(receptor),
        "pocket_json": str(pocket),
        "pocket_sha256": package.digest(pocket),
        "source_control_record_sha256": package.canonical_json_sha256(control),
    }
    pairs_csv = directory / "pairs.csv"
    pairs_csv.write_text("pair_id,intent_id,target_id\npair_x,TI-030-KLK5,P_TEST\n", encoding="utf-8")
    _write_json(
        directory / "structural_manifest.json",
        {
            "schema_version": package.STRUCTURAL_SCHEMA,
            "pair_source": {"path": str(pairs_csv), "sha256": package.digest(pairs_csv)},
            "source_record_manifest": {
                "path": str(source_path),
                "sha256": package.digest(source_path),
            },
            "pairs": [pair],
        },
    )
    with (directory / "pair_status.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATUS_FIELDS)
        writer.writeheader()
        writer.writerow({field: pair[field] for field in STATUS_FIELDS})
    return directory


def _add_halted_pair(directory: Path) -> None:
    manifest_path = directory / "structural_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_path = directory / "source_record_manifest.json"
    source_manifest = json.loads(source_path.read_text(encoding="utf-8"))
    exact = package.identity("CCN")
    control = {
        "control_role": "weak_binding_control",
        "molecule_chembl_id": "CHEMBL_HALT",
        "structure_compound_id": exact["compound_id"],
        "standard_inchi_key": exact["computed_inchikey"],
        "smiles": "CCN",
        "activity_id": "999",
        "assay_chembl_id": "CHEMBL_ASSY",
        "standard_type": "Ki",
        "standard_relation": "=",
        "standard_value": 70700.0,
        "standard_units": "nM",
        "doi": "10.0000/test",
    }
    source_manifest["controls"].append(control)
    _write_json(source_path, source_manifest)
    pair = {
        "pair_id": "pair_halt",
        "intent_id": "TI-030-KLK5",
        "target_id": "P_TEST",
        "compound_id": exact["compound_id"],
        "control_role": "weak_binding_control",
        "control_evidence_id": "CHEMBL_ASSY:activity:999",
        "status": "halted_policy",
        "reason": "Protonation state unsupported; no pose executed",
        "safety_status": "FLAG_HIGH",
        "applicability_status": "review",
        "safety_consensus_valid": "false",
        "full_analysis_complete": "false",
        "structure_protocol_status": "not_run",
        "safety_evidence_sha256": "a" * 64,
        "safety_evidence_record_sha256": "b" * 64,
        "source_control_record_sha256": package.canonical_json_sha256(control),
    }
    manifest["pairs"].append(pair)
    manifest["source_record_manifest"]["sha256"] = package.digest(source_path)
    _write_json(manifest_path, manifest)
    with (directory / "pair_status.csv").open(
        "a", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=STATUS_FIELDS)
        writer.writerow({field: pair.get(field, "") for field in STATUS_FIELDS})


def test_completed_and_policy_halted_pairs_load_together(tmp_path: Path) -> None:
    directory = _structural_dir(tmp_path)
    _add_halted_pair(directory)
    controls, frame, artifacts = package.load_structural_controls(directory)
    assert [control["structure_status"] for control in controls] == [
        "completed",
        "halted_policy",
    ]
    assert controls[1]["structure_reason"].startswith("Protonation")
    assert frame.iloc[1]["structure_status"] == "halted_policy"
    assert artifacts["status"] == "verified"
    assert len(artifacts["pairs"]) == 2
    assert artifacts["pairs"][1]["pose_sha256"] is None


def test_halted_pair_missing_reason_is_rejected(tmp_path: Path) -> None:
    directory = _structural_dir(tmp_path)
    _add_halted_pair(directory)
    manifest_path = directory / "structural_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for pair in manifest["pairs"]:
        if pair["pair_id"] == "pair_halt":
            pair.pop("reason")
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="requires a reason"):
        package.load_structural_controls(directory)


def test_halted_pair_must_not_record_structural_bytes(tmp_path: Path) -> None:
    directory = _structural_dir(tmp_path)
    _add_halted_pair(directory)
    manifest_path = directory / "structural_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pose_hash = package.digest(directory / "inputs" / "ligand.sdf")
    for pair in manifest["pairs"]:
        if pair["pair_id"] == "pair_halt":
            pair["pose_sdf"] = "inputs/ligand.sdf"
            pair["pose_sha256"] = pose_hash
    _write_json(manifest_path, manifest)
    status_path = directory / "pair_status.csv"
    status_rows = list(
        csv.DictReader(status_path.open(newline="", encoding="utf-8"))
    )
    for row in status_rows:
        if row["pair_id"] == "pair_halt":
            row["pose_sdf"] = "inputs/ligand.sdf"
            row["pose_sha256"] = pose_hash
    with status_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATUS_FIELDS)
        writer.writeheader()
        writer.writerows(status_rows)
    with pytest.raises(ValueError, match="must not record structural bytes"):
        package.load_structural_controls(directory)


def test_completed_pair_missing_pose_is_rejected(tmp_path: Path) -> None:
    directory = _structural_dir(tmp_path)
    (directory / "artifacts" / "pair_x" / "pose_test.sdf").unlink()
    with pytest.raises(ValueError, match="pose hash mismatch"):
        package.load_structural_controls(directory)


def test_verified_structural_dir_loads_controls_and_records_hashes(tmp_path: Path) -> None:
    directory = _structural_dir(tmp_path)
    controls, frame, artifacts = package.load_structural_controls(directory)
    assert len(controls) == 1
    assert controls[0]["structure_status"] == "completed"
    assert controls[0]["gnina_cnn_affinity"] == pytest.approx(6.25)
    assert frame.iloc[0]["control_role"] == "active_control"
    assert artifacts["status"] == "verified"
    assert artifacts["pairs"][0]["pose_sha256"] == package.digest(
        directory / "artifacts" / "pair_x" / "pose_test.sdf"
    )
    for key in ("manifest", "pair_status", "source_record_manifest", "pair_source"):
        assert artifacts[key]["sha256"] == package.digest(Path(artifacts[key]["path"]))
    pair = artifacts["pairs"][0]
    for key in (
        "pose_sha256",
        "ligand_sha256",
        "receptor_sha256",
        "pocket_sha256",
        "safety_evidence_sha256",
        "source_control_record_sha256",
    ):
        assert isinstance(pair[key], str) and len(pair[key]) == 64


def test_missing_structural_manifest_is_rejected(tmp_path: Path) -> None:
    directory = _structural_dir(tmp_path)
    (directory / "structural_manifest.json").unlink()
    with pytest.raises(ValueError, match="manifest is required"):
        package.load_structural_controls(directory)


def test_pair_status_change_is_rejected(tmp_path: Path) -> None:
    directory = _structural_dir(tmp_path)
    path = directory / "pair_status.csv"
    path.write_text(path.read_text(encoding="utf-8").replace(",completed,", ",failed,"), encoding="utf-8")
    with pytest.raises(ValueError, match="status disagrees"):
        package.load_structural_controls(directory)


def test_pair_score_change_is_rejected(tmp_path: Path) -> None:
    directory = _structural_dir(tmp_path)
    path = directory / "pair_status.csv"
    path.write_text(path.read_text(encoding="utf-8").replace("6.25", "9.99"), encoding="utf-8")
    with pytest.raises(ValueError, match="gnina_cnn_affinity disagrees"):
        package.load_structural_controls(directory)


def test_pair_target_change_is_rejected(tmp_path: Path) -> None:
    directory = _structural_dir(tmp_path)
    path = directory / "pair_status.csv"
    path.write_text(path.read_text(encoding="utf-8").replace("P_TEST", "P_OTHER"), encoding="utf-8")
    with pytest.raises(ValueError, match="target_id disagrees"):
        package.load_structural_controls(directory)


def test_pose_byte_change_is_rejected(tmp_path: Path) -> None:
    directory = _structural_dir(tmp_path)
    pose = directory / "artifacts" / "pair_x" / "pose_test.sdf"
    pose.write_bytes(pose.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="pose hash mismatch"):
        package.load_structural_controls(directory)


def test_source_ligand_swap_is_rejected(tmp_path: Path) -> None:
    directory = _structural_dir(tmp_path)
    ligand = directory / "inputs" / "ligand.sdf"
    ligand.write_bytes(b"different ligand\n")
    with pytest.raises(ValueError, match="ligand_sdf hash mismatch"):
        package.load_structural_controls(directory)


def test_swapped_source_record_manifest_is_rejected(tmp_path: Path) -> None:
    directory = _structural_dir(tmp_path)
    source = directory / "source_record_manifest.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["target"]["uniprot"] = "Q00000"
    _write_json(source, payload)
    with pytest.raises(ValueError, match="source_record_manifest hash mismatch"):
        package.load_structural_controls(directory)


def test_swapped_pair_source_is_rejected(tmp_path: Path) -> None:
    directory = _structural_dir(tmp_path)
    pairs = directory / "pairs.csv"
    pairs.write_text("pair_id\nelsewhere\n", encoding="utf-8")
    with pytest.raises(ValueError, match="pair_source hash mismatch"):
        package.load_structural_controls(directory)


def test_canonical_json_sha256_matches_producer_canonicalization() -> None:
    payload = {"b": 1, "a": [1, 2], "c": "한글"}
    expected = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
    assert package.canonical_json_sha256(payload) == expected
