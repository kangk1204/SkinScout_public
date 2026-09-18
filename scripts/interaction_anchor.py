#!/usr/bin/env python3
"""Contracts and scoring for parent pose-supported interaction anchors."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "skinscout.interaction_anchor_map.v1"
SOURCE_COORDINATE_SYSTEM = "boltz_complex_ligand_atom_order_0_based"
MAPPED_COORDINATE_SYSTEM = "parent_sdf_atom_order_0_based"
CANONICAL_COORDINATE_SYSTEM = "canonical_smiles_atom_order_0_based"
PRESERVATION_BASIS = (
    "pose_supported_parent_anchor_conservative_mcs_feature_preservation"
)
MAX_MCS_MATCHES = 128
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SUPPORTED_FEATURE_FAMILIES = frozenset(
    {
        "Donor",
        "Acceptor",
        "Aromatic",
        "Hydrophobe",
        "PosIonizable",
        "NegIonizable",
    }
)


class InteractionAnchorError(RuntimeError):
    """Raised when an interaction-anchor artifact is not scientifically usable."""


@dataclass(frozen=True)
class AnchorPreservation:
    score: float
    preserved_anchor_count: int
    anchor_count: int
    mcs_atom_count: int
    mapping_count: int = 0
    mapping_ambiguous: bool = False
    mapping_truncated: bool = False
    basis: str = PRESERVATION_BASIS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise InteractionAnchorError(f"{label} is required and must be non-empty: {path}")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise InteractionAnchorError(f"{label} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise InteractionAnchorError(f"{label} must be a JSON object: {path}")
    return payload


def _atom_indices(value: object, *, label: str) -> list[int]:
    if (
        not isinstance(value, list)
        or any(type(index) is not int or index < 0 for index in value)
        or len(value) != len(set(value))
    ):
        raise InteractionAnchorError(
            f"{label} must be a duplicate-free list of non-negative integers"
        )
    return value


def _sha256_value(value: object, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise InteractionAnchorError(f"{label} must be a lowercase SHA256 digest")
    return value


def _source_record(
    value: object,
    *,
    label: str,
    expected_sha256: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InteractionAnchorError(f"{label} must be an object")
    source_path = value.get("path")
    source_bytes = value.get("bytes")
    source_sha256 = _sha256_value(value.get("sha256"), label=f"{label}.sha256")
    if (
        not isinstance(source_path, str)
        or not source_path.strip()
        or source_path != source_path.strip()
        or type(source_bytes) is not int
        or source_bytes < 1
        or source_sha256 != expected_sha256
    ):
        raise InteractionAnchorError(f"{label} metadata is invalid")
    return value


def _recorded_path_candidates(path_text: str, artifact_path: Path) -> tuple[Path, ...]:
    recorded = Path(path_text).expanduser()
    if recorded.is_absolute():
        return (recorded,)
    return (Path.cwd() / recorded, artifact_path.parent / recorded)


def _verify_recorded_file(
    record: dict[str, Any],
    *,
    artifact_path: Path,
    label: str,
) -> None:
    candidates = _recorded_path_candidates(str(record["path"]), artifact_path)
    for source in candidates:
        if (
            source.is_file()
            and not source.is_symlink()
            and source.stat().st_size == record["bytes"]
            and sha256_file(source) == record["sha256"]
        ):
            return
    shown = ", ".join(str(candidate) for candidate in candidates)
    raise InteractionAnchorError(
        f"{label} source file is missing, unsafe, or fingerprint-mismatched: {shown}"
    )


def validate_anchor_map(
    payload: dict[str, Any],
    *,
    path: Path,
    expected_parent_sha256: str | None = None,
    expected_consensus_sha256: str | None = None,
    expected_parent_inchikey: str | None = None,
    required_target: str | None = None,
) -> dict[str, Any]:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise InteractionAnchorError(
            f"Interaction-anchor schema_version must be {SCHEMA_VERSION!r}: {path}"
        )
    if payload.get("claim_eligible") is not True or payload.get("analog_pose_verified") is not False:
        raise InteractionAnchorError(
            f"Interaction-anchor claim boundary is invalid: {path}"
        )
    consensus_sha256 = _sha256_value(
        payload.get("consensus_sha256"),
        label="Interaction-anchor consensus_sha256",
    )
    boltz_report_sha256 = _sha256_value(
        payload.get("boltz_report_sha256"),
        label="Interaction-anchor boltz_report_sha256",
    )
    parent = payload.get("parent")
    if not isinstance(parent, dict):
        raise InteractionAnchorError(f"Interaction-anchor parent must be an object: {path}")
    parent_sha256 = _sha256_value(
        parent.get("sdf_sha256"),
        label="Interaction-anchor parent.sdf_sha256",
    )
    parent_inchikey = parent.get("inchikey")
    parent_atom_count = parent.get("atom_count")
    if (
        not isinstance(parent_inchikey, str)
        or not parent_inchikey
        or type(parent_atom_count) is not int
        or parent_atom_count < 1
    ):
        raise InteractionAnchorError(f"Interaction-anchor parent metadata is invalid: {path}")
    if expected_parent_sha256 is not None and parent_sha256 != expected_parent_sha256:
        raise InteractionAnchorError(f"Interaction-anchor parent SDF hash mismatch: {path}")
    if expected_parent_inchikey is not None and parent_inchikey != expected_parent_inchikey:
        raise InteractionAnchorError(f"Interaction-anchor parent identity mismatch: {path}")
    if (
        expected_consensus_sha256 is not None
        and consensus_sha256 != expected_consensus_sha256
    ):
        raise InteractionAnchorError(f"Interaction-anchor consensus hash mismatch: {path}")

    sources = payload.get("sources")
    if not isinstance(sources, dict) or set(sources) != {
        "parent_sdf",
        "consensus_json",
        "boltz_report",
    }:
        raise InteractionAnchorError(
            f"Interaction-anchor source fingerprints are incomplete: {path}"
        )
    _source_record(
        sources["parent_sdf"],
        label="Interaction-anchor sources.parent_sdf",
        expected_sha256=parent_sha256,
    )
    _source_record(
        sources["consensus_json"],
        label="Interaction-anchor sources.consensus_json",
        expected_sha256=consensus_sha256,
    )
    _source_record(
        sources["boltz_report"],
        label="Interaction-anchor sources.boltz_report",
        expected_sha256=boltz_report_sha256,
    )

    targets = payload.get("targets")
    if not isinstance(targets, dict) or not targets:
        raise InteractionAnchorError(
            f"Interaction-anchor targets must be a non-empty object: {path}"
        )
    if required_target is not None and required_target not in targets:
        raise InteractionAnchorError(
            f"Interaction-anchor map has no entry for target {required_target}: {path}"
        )

    for target_id, record in targets.items():
        if not isinstance(target_id, str) or not target_id.strip() or not isinstance(record, dict):
            raise InteractionAnchorError(f"Interaction-anchor target entry is invalid: {path}")
        if record.get("target_id") != target_id:
            raise InteractionAnchorError(
                f"Interaction-anchor target_id mismatch for {target_id}: {path}"
            )
        complex_path = record.get("complex_pdb")
        complex_bytes = record.get("complex_pdb_bytes")
        _sha256_value(
            record.get("complex_pdb_sha256"),
            label=f"Interaction-anchor complex_pdb_sha256 for {target_id}",
        )
        if (
            not isinstance(complex_path, str)
            or not complex_path.strip()
            or complex_path != complex_path.strip()
            or type(complex_bytes) is not int
            or complex_bytes < 1
        ):
            raise InteractionAnchorError(
                f"Interaction-anchor complex source metadata is invalid for {target_id}: {path}"
            )
        if (
            record.get("source_coordinate_system") != SOURCE_COORDINATE_SYSTEM
            or record.get("mapped_coordinate_system") != MAPPED_COORDINATE_SYSTEM
            or record.get("mapping_status") != "mapped"
            or record.get("mapping_confidence")
            not in {"high", "high_validated_atom_maps"}
            or record.get("claim_eligible") is not True
        ):
            raise InteractionAnchorError(
                f"Interaction-anchor mapping is not high-confidence and claim-eligible for {target_id}: {path}"
            )
        complex_atoms = _atom_indices(
            record.get("confirmed_complex_atoms"),
            label=f"Interaction-anchor confirmed_complex_atoms for {target_id}",
        )
        parent_atoms = _atom_indices(
            record.get("confirmed_parent_atoms"),
            label=f"Interaction-anchor confirmed_parent_atoms for {target_id}",
        )
        if not complex_atoms or len(complex_atoms) != len(parent_atoms):
            raise InteractionAnchorError(
                f"Interaction-anchor confirmed atom mapping is incomplete for {target_id}: {path}"
            )
        if any(index >= parent_atom_count for index in parent_atoms):
            raise InteractionAnchorError(
                f"Interaction-anchor parent atom index is out of range for {target_id}: {path}"
            )
        raw_mapping = record.get("atom_mapping")
        if not isinstance(raw_mapping, list) or not raw_mapping:
            raise InteractionAnchorError(
                f"Interaction-anchor atom_mapping is empty for {target_id}: {path}"
            )
        complex_to_parent: dict[int, int] = {}
        parent_seen: set[int] = set()
        for item in raw_mapping:
            if not isinstance(item, dict):
                raise InteractionAnchorError(
                    f"Interaction-anchor atom_mapping row is invalid for {target_id}: {path}"
                )
            complex_index = item.get("complex_atom_index")
            parent_index = item.get("parent_atom_index")
            if (
                type(complex_index) is not int
                or complex_index < 0
                or type(parent_index) is not int
                or parent_index < 0
                or parent_index >= parent_atom_count
                or complex_index in complex_to_parent
                or parent_index in parent_seen
            ):
                raise InteractionAnchorError(
                    f"Interaction-anchor atom_mapping is not bijective for {target_id}: {path}"
                )
            complex_to_parent[complex_index] = parent_index
            parent_seen.add(parent_index)
        mapped_confirmed = [complex_to_parent.get(index) for index in complex_atoms]
        if mapped_confirmed != parent_atoms:
            raise InteractionAnchorError(
                f"Interaction-anchor confirmed atom mapping is inconsistent for {target_id}: {path}"
            )

        canonical_atoms_raw = record.get("confirmed_canonical_parent_atoms")
        canonical_mapping_raw = record.get("canonical_atom_mapping")
        canonical_coordinate = record.get("canonical_coordinate_system")
        canonical_fields = (
            canonical_atoms_raw,
            canonical_mapping_raw,
            canonical_coordinate,
        )
        if any(value is not None for value in canonical_fields):
            canonical_smiles = parent.get("canonical_smiles")
            canonical_atom_count = parent.get("canonical_atom_count")
            if (
                not isinstance(canonical_smiles, str)
                or not canonical_smiles
                or type(canonical_atom_count) is not int
                or canonical_atom_count < 1
            ):
                raise InteractionAnchorError(
                    f"Interaction-anchor canonical parent metadata is invalid: {path}"
                )
            if canonical_coordinate != CANONICAL_COORDINATE_SYSTEM:
                raise InteractionAnchorError(
                    f"Interaction-anchor canonical coordinate system is invalid for {target_id}: {path}"
                )
            canonical_atoms = _atom_indices(
                canonical_atoms_raw,
                label=(
                    "Interaction-anchor confirmed_canonical_parent_atoms "
                    f"for {target_id}"
                ),
            )
            if (
                len(canonical_atoms) != len(parent_atoms)
                or any(index >= canonical_atom_count for index in canonical_atoms)
            ):
                raise InteractionAnchorError(
                    f"Interaction-anchor canonical confirmed atoms are incomplete for {target_id}: {path}"
                )
            if not isinstance(canonical_mapping_raw, list) or not canonical_mapping_raw:
                raise InteractionAnchorError(
                    f"Interaction-anchor canonical_atom_mapping is empty for {target_id}: {path}"
                )
            parent_to_canonical: dict[int, int] = {}
            canonical_seen: set[int] = set()
            for item in canonical_mapping_raw:
                if not isinstance(item, dict):
                    raise InteractionAnchorError(
                        f"Interaction-anchor canonical mapping row is invalid for {target_id}: {path}"
                    )
                canonical_index = item.get("canonical_atom_index")
                parent_index = item.get("parent_atom_index")
                if (
                    type(canonical_index) is not int
                    or canonical_index < 0
                    or canonical_index >= canonical_atom_count
                    or type(parent_index) is not int
                    or parent_index < 0
                    or parent_index >= parent_atom_count
                    or canonical_index in canonical_seen
                    or parent_index in parent_to_canonical
                ):
                    raise InteractionAnchorError(
                        f"Interaction-anchor canonical mapping is not bijective for {target_id}: {path}"
                    )
                canonical_seen.add(canonical_index)
                parent_to_canonical[parent_index] = canonical_index
            if (
                len(canonical_seen) != canonical_atom_count
                or [parent_to_canonical.get(index) for index in parent_atoms]
                != canonical_atoms
            ):
                raise InteractionAnchorError(
                    f"Interaction-anchor canonical mapping is inconsistent for {target_id}: {path}"
                )
    return payload


def load_anchor_map(
    path: Path,
    *,
    parent_sdf: Path | None = None,
    consensus_json: Path | None = None,
    boltz_report: Path | None = None,
    expected_parent_inchikey: str | None = None,
    required_target: str | None = None,
    verify_source_files: bool = False,
) -> dict[str, Any]:
    payload = _read_json(path, "Interaction-anchor map")
    validated = validate_anchor_map(
        payload,
        path=path,
        expected_parent_sha256=(sha256_file(parent_sdf) if parent_sdf is not None else None),
        expected_consensus_sha256=(
            sha256_file(consensus_json) if consensus_json is not None else None
        ),
        expected_parent_inchikey=expected_parent_inchikey,
        required_target=required_target,
    )
    if boltz_report is not None and sha256_file(boltz_report) != validated["boltz_report_sha256"]:
        raise InteractionAnchorError(f"Interaction-anchor Boltz report hash mismatch: {path}")
    if verify_source_files:
        for source_name, record in validated["sources"].items():
            _verify_recorded_file(
                record,
                artifact_path=path,
                label=f"Interaction-anchor {source_name}",
            )
        source_targets = (
            {required_target: validated["targets"][required_target]}
            if required_target is not None
            else validated["targets"]
        )
        for target_id, record in source_targets.items():
            _verify_recorded_file(
                {
                    "path": record["complex_pdb"],
                    "bytes": record["complex_pdb_bytes"],
                    "sha256": record["complex_pdb_sha256"],
                },
                artifact_path=path,
                label=f"Interaction-anchor complex PDB for {target_id}",
            )
    return validated


def feature_families_by_atom(molecule: Any, feature_factory: Any) -> dict[int, set[str]]:
    families: dict[int, set[str]] = {
        int(atom.GetIdx()): set() for atom in molecule.GetAtoms()
    }
    for feature in feature_factory.GetFeaturesForMol(molecule):
        family = str(feature.GetFamily())
        if family not in SUPPORTED_FEATURE_FAMILIES:
            continue
        for atom_index in feature.GetAtomIds():
            families[int(atom_index)].add(family)
    return families


def score_anchor_preservation(
    parent: Any,
    analog: Any,
    parent_anchor_indices: list[int],
    feature_factory: Any,
) -> AnchorPreservation:
    from rdkit import Chem
    from rdkit.Chem import rdFMCS

    anchors = _atom_indices(parent_anchor_indices, label="Parent interaction anchors")
    if not anchors:
        raise InteractionAnchorError("Parent interaction anchors must not be empty")
    if any(index >= parent.GetNumAtoms() for index in anchors):
        raise InteractionAnchorError("Parent interaction anchor index is out of range")
    result = rdFMCS.FindMCS(
        [parent, analog],
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrderExact,
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
        matchChiralTag=True,
        timeout=5,
    )
    if result.canceled or result.numAtoms < 1 or not result.smartsString:
        return AnchorPreservation(0.0, 0, len(anchors), 0)
    query = Chem.MolFromSmarts(result.smartsString)
    if query is None:
        return AnchorPreservation(0.0, 0, len(anchors), 0)

    match_parameters = Chem.SubstructMatchParameters()
    match_parameters.useChirality = True
    match_parameters.uniquify = True
    match_parameters.maxMatches = MAX_MCS_MATCHES
    parent_matches = parent.GetSubstructMatches(query, match_parameters)
    analog_matches = analog.GetSubstructMatches(query, match_parameters)
    if not parent_matches or not analog_matches:
        return AnchorPreservation(0.0, 0, len(anchors), int(result.numAtoms))

    parent_features = feature_families_by_atom(parent, feature_factory)
    analog_features = feature_families_by_atom(analog, feature_factory)
    anchor_set = set(anchors)
    mapping_count = len(parent_matches) * len(analog_matches)
    mapping_truncated = (
        len(parent_matches) >= MAX_MCS_MATCHES
        or len(analog_matches) >= MAX_MCS_MATCHES
    )
    worst = len(anchors)
    for parent_match in parent_matches:
        mapped_anchors = anchor_set & set(parent_match)
        for analog_match in analog_matches:
            atom_map = dict(zip(parent_match, analog_match, strict=True))
            preserved = 0
            for parent_index in mapped_anchors:
                analog_index = atom_map[parent_index]
                source_families = parent_features[parent_index]
                mapped_families = analog_features[analog_index]
                if not source_families or source_families & mapped_families:
                    preserved += 1
            worst = min(worst, preserved)
            if worst == 0:
                break
        if worst == 0:
            break
    if mapping_truncated:
        worst = 0
    score = worst / len(anchors)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise InteractionAnchorError("Computed interaction-anchor score is invalid")
    return AnchorPreservation(
        score=score,
        preserved_anchor_count=worst,
        anchor_count=len(anchors),
        mcs_atom_count=int(result.numAtoms),
        mapping_count=mapping_count,
        mapping_ambiguous=mapping_count > 1,
        mapping_truncated=mapping_truncated,
    )
