#!/usr/bin/env python3
"""Map Stage 5.5 bound-complex interaction atoms into parent SDF atom order."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, ChemicalFeatures, inchi

from atom_mapping import ALGORITHM_VERSION, map_molecules
from interaction_anchor import (
    MAPPED_COORDINATE_SYSTEM,
    CANONICAL_COORDINATE_SYSTEM,
    SCHEMA_VERSION,
    SOURCE_COORDINATE_SYSTEM,
    feature_families_by_atom,
    sha256_file,
    validate_anchor_map,
)


LOG = logging.getLogger("stage5_5.atom_map")


def _require_nonempty(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"{label} is required and must be non-empty: {path}")


def _read_single_parent(path: Path) -> Chem.Mol:
    _require_nonempty(path, "Parent SDF")
    molecules = [
        molecule
        for molecule in Chem.SDMolSupplier(str(path), sanitize=True, removeHs=False)
        if molecule is not None
    ]
    if len(molecules) != 1:
        raise SystemExit(f"Parent SDF must contain exactly one readable molecule: {path}")
    return molecules[0]


def _read_consensus(path: Path) -> dict[str, dict[str, Any]]:
    _require_nonempty(path, "Consensus pharmacophore JSON")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Consensus pharmacophore JSON failed to parse: {path}: {exc}") from exc
    if not isinstance(payload, dict) or not payload:
        raise SystemExit(f"Consensus pharmacophore JSON must be a non-empty object: {path}")
    normalized: dict[str, dict[str, Any]] = {}
    for target_id, record in payload.items():
        if not isinstance(target_id, str) or not target_id.strip() or not isinstance(record, dict):
            raise SystemExit(f"Consensus pharmacophore target entry is invalid: {path}")
        atoms = record.get("confirmed_atoms")
        if (
            not isinstance(atoms, list)
            or not atoms
            or any(type(index) is not int or index < 0 for index in atoms)
            or len(atoms) != len(set(atoms))
        ):
            raise SystemExit(
                f"Consensus confirmed_atoms must be non-empty and duplicate-free for {target_id}: {path}"
            )
        if (
            record.get("coordinate_system") != SOURCE_COORDINATE_SYSTEM
            or record.get("degraded") is not False
            or record.get("claim_eligible") is not True
        ):
            raise SystemExit(
                f"Consensus entry is not eligible for atom mapping for {target_id}: {path}"
            )
        normalized[target_id] = record
    return normalized


def _resolve_complex_path(value: object, report_path: Path) -> Path:
    raw = str(value).strip()
    if not raw:
        raise SystemExit(f"Boltz-2 report contains a blank complex_pdb path: {report_path}")
    path = Path(raw).expanduser()
    if path.is_file():
        return path
    relative = report_path.parent / path
    return relative if relative.is_file() else path


def _read_kept_complexes(path: Path) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    _require_nonempty(path, "Boltz-2 report")
    try:
        report = pd.read_csv(path, sep="\t", keep_default_na=False)
    except Exception as exc:
        raise SystemExit(f"Boltz-2 report failed to parse: {path}: {exc}") from exc
    required = {
        "target_id",
        "complex_pdb",
        "iptm",
        "complex_plddt",
        "affinity_log_uM",
        "kept",
    }
    missing = sorted(required - set(report.columns))
    if missing:
        raise SystemExit(f"Boltz-2 report missing required columns {missing}: {path}")
    target_ids = report["target_id"].astype(str).str.strip()
    if target_ids.eq("").any() or target_ids.duplicated().any():
        raise SystemExit(f"Boltz-2 report target_id values must be non-empty and unique: {path}")
    complexes: dict[str, Path] = {}
    quality: dict[str, dict[str, Any]] = {}
    for index, row in report.iterrows():
        target_id = target_ids.loc[index]
        kept = str(row["kept"]).strip().lower()
        if kept not in {"yes", "no"}:
            raise SystemExit(f"Boltz-2 report kept must be yes or no for {target_id}: {path}")
        metrics: dict[str, float] = {}
        for field in ("iptm", "complex_plddt", "affinity_log_uM"):
            try:
                value = float(row[field])
            except (TypeError, ValueError) as exc:
                raise SystemExit(
                    f"Boltz-2 report {field} must be numeric for {target_id}: {path}"
                ) from exc
            if not math.isfinite(value):
                raise SystemExit(
                    f"Boltz-2 report {field} must be finite for {target_id}: {path}"
                )
            metrics[field] = value
        quality[target_id] = {**metrics, "kept": kept == "yes"}
        if kept == "yes":
            complex_path = _resolve_complex_path(row["complex_pdb"], path)
            _require_nonempty(complex_path, f"Boltz complex PDB for {target_id}")
            complexes[target_id] = complex_path
    if not complexes:
        raise SystemExit(f"Boltz-2 report has no quality-kept complexes: {path}")
    return complexes, quality


def _ligand_pdb_block(path: Path) -> tuple[str, list[int]]:
    lines = path.read_text(errors="replace").splitlines()
    hetatm = [line for line in lines if line.startswith("HETATM")]
    if not hetatm:
        raise SystemExit(f"Boltz complex has no ligand HETATM records: {path}")
    serials: list[int] = []
    for line in hetatm:
        try:
            serial = int(line[6:11])
        except ValueError as exc:
            raise SystemExit(f"Boltz complex has invalid HETATM serial: {path}") from exc
        if serial in serials:
            raise SystemExit(f"Boltz complex has duplicate HETATM serial {serial}: {path}")
        serials.append(serial)
    serial_set = set(serials)
    conect: list[str] = []
    for line in lines:
        if not line.startswith("CONECT"):
            continue
        try:
            values = [int(value) for value in line.split()[1:]]
        except ValueError:
            continue
        if not values or values[0] not in serial_set:
            continue
        linked = [value for value in values[1:] if value in serial_set]
        if linked:
            conect.append("CONECT" + f"{values[0]:5d}" + "".join(f"{value:5d}" for value in linked))
    return "\n".join([*hetatm, *conect, "END", ""]), serials


def _bound_ligand(path: Path, parent: Chem.Mol) -> tuple[Chem.Mol, list[int]]:
    block, _serials = _ligand_pdb_block(path)
    ligand = Chem.MolFromPDBBlock(
        block,
        sanitize=False,
        removeHs=False,
        proximityBonding=True,
    )
    if ligand is None:
        raise SystemExit(f"RDKit could not parse the Boltz bound ligand: {path}")
    complex_heavy_original = [
        atom.GetIdx() for atom in ligand.GetAtoms() if atom.GetAtomicNum() != 1
    ]
    parent_heavy = Chem.RemoveHs(parent)
    ligand_heavy = Chem.RemoveHs(ligand, sanitize=False)
    if ligand_heavy.GetNumAtoms() != parent_heavy.GetNumAtoms():
        raise SystemExit(
            "Parent and bound ligand have different heavy-atom counts: "
            f"parent={parent_heavy.GetNumAtoms()} bound={ligand_heavy.GetNumAtoms()}: {path}"
        )
    try:
        ligand_heavy = AllChem.AssignBondOrdersFromTemplate(parent_heavy, ligand_heavy)
        Chem.SanitizeMol(ligand_heavy)
    except Exception as exc:
        raise SystemExit(
            f"Could not recover bound-ligand topology from the parent template: {path}: {exc}"
        ) from exc
    return ligand_heavy, complex_heavy_original


def _map_target(
    *,
    target_id: str,
    parent: Chem.Mol,
    confirmed_complex_atoms: list[int],
    complex_pdb: Path,
    quality: dict[str, Any],
    feature_factory: Any,
    parent_to_canonical: dict[int, int],
    canonical_atom_mapping: list[dict[str, int]],
) -> dict[str, Any]:
    parent_heavy_original = [
        atom.GetIdx() for atom in parent.GetAtoms() if atom.GetAtomicNum() != 1
    ]
    parent_heavy = Chem.RemoveHs(parent)
    bound_heavy, complex_heavy_original = _bound_ligand(complex_pdb, parent)
    mapping = map_molecules(parent_heavy, bound_heavy, Chem, inchi)
    if mapping.status != "mapped" or mapping.confidence not in {
        "high",
        "high_validated_atom_maps",
    }:
        raise SystemExit(
            f"High-confidence atom mapping failed for {target_id}: "
            f"status={mapping.status} message={mapping.message}"
        )

    translated: list[dict[str, int]] = []
    for item in mapping.atom_mapping:
        parent_index = parent_heavy_original[item["generated_atom_index"]]
        complex_index = complex_heavy_original[item["complex_atom_index"]]
        translated.append(
            {
                "parent_atom_index": parent_index,
                "complex_atom_index": complex_index,
            }
        )
    translated.sort(key=lambda item: item["complex_atom_index"])
    complex_to_parent = {
        item["complex_atom_index"]: item["parent_atom_index"] for item in translated
    }
    missing = [index for index in confirmed_complex_atoms if index not in complex_to_parent]
    if missing:
        raise SystemExit(
            f"Confirmed interaction atoms are not mapped heavy atoms for {target_id}: {missing}"
        )
    confirmed_parent_atoms = [
        complex_to_parent[index] for index in confirmed_complex_atoms
    ]
    try:
        confirmed_canonical_parent_atoms = [
            parent_to_canonical[index] for index in confirmed_parent_atoms
        ]
    except KeyError as exc:
        raise SystemExit(
            f"Confirmed parent interaction atom is absent from the canonical heavy-atom mapping for {target_id}: {exc.args[0]}"
        ) from exc
    parent_features = feature_families_by_atom(parent, feature_factory)
    return {
        "target_id": target_id,
        "source_coordinate_system": SOURCE_COORDINATE_SYSTEM,
        "mapped_coordinate_system": MAPPED_COORDINATE_SYSTEM,
        "confirmed_complex_atoms": confirmed_complex_atoms,
        "confirmed_parent_atoms": confirmed_parent_atoms,
        "canonical_coordinate_system": CANONICAL_COORDINATE_SYSTEM,
        "confirmed_canonical_parent_atoms": confirmed_canonical_parent_atoms,
        "canonical_atom_mapping": canonical_atom_mapping,
        "confirmed_parent_feature_families": {
            str(index): sorted(parent_features[index]) for index in confirmed_parent_atoms
        },
        "atom_mapping": translated,
        "mapping_algorithm": ALGORITHM_VERSION,
        "mapping_status": mapping.status,
        "mapping_confidence": mapping.confidence,
        "mapping_count": mapping.mapping_count,
        "complex_pdb": str(complex_pdb.resolve()),
        "complex_pdb_bytes": complex_pdb.stat().st_size,
        "complex_pdb_sha256": sha256_file(complex_pdb),
        "boltz_quality": quality,
        "claim_eligible": True,
    }


def _canonical_parent_mapping(
    parent: Chem.Mol,
) -> tuple[str, Chem.Mol, dict[int, int], list[dict[str, int]]]:
    parent_heavy_original = [
        atom.GetIdx() for atom in parent.GetAtoms() if atom.GetAtomicNum() != 1
    ]
    parent_heavy = Chem.RemoveHs(parent)
    canonical_smiles = Chem.MolToSmiles(
        parent_heavy, canonical=True, isomericSmiles=True
    )
    canonical_parent = Chem.MolFromSmiles(canonical_smiles)
    if canonical_parent is None:
        raise SystemExit("Could not reconstruct the canonical parent molecule")
    mapping = map_molecules(canonical_parent, parent_heavy, Chem, inchi)
    if mapping.status != "mapped" or mapping.confidence not in {
        "high",
        "high_validated_atom_maps",
    }:
        raise SystemExit(
            "Canonical parent atom mapping is not unique: "
            f"status={mapping.status} message={mapping.message}"
        )
    canonical_atom_mapping = [
        {
            "canonical_atom_index": item["generated_atom_index"],
            "parent_atom_index": parent_heavy_original[item["complex_atom_index"]],
        }
        for item in mapping.atom_mapping
    ]
    canonical_atom_mapping.sort(key=lambda item: item["canonical_atom_index"])
    parent_to_canonical = {
        item["parent_atom_index"]: item["canonical_atom_index"]
        for item in canonical_atom_mapping
    }
    return (
        canonical_smiles,
        canonical_parent,
        parent_to_canonical,
        canonical_atom_mapping,
    )


def build_payload(parent_sdf: Path, boltz_report: Path, consensus_json: Path) -> dict[str, Any]:
    parent = _read_single_parent(parent_sdf)
    (
        canonical_smiles,
        canonical_parent,
        parent_to_canonical,
        canonical_atom_mapping,
    ) = _canonical_parent_mapping(parent)
    consensus = _read_consensus(consensus_json)
    kept_complexes, quality = _read_kept_complexes(boltz_report)
    feature_factory = ChemicalFeatures.BuildFeatureFactory(
        str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef")
    )
    targets: dict[str, Any] = {}
    excluded: list[dict[str, str]] = []
    for target_id, record in sorted(consensus.items()):
        if target_id not in kept_complexes:
            excluded.append({"target_id": target_id, "reason": "boltz_quality_not_kept"})
            continue
        targets[target_id] = _map_target(
            target_id=target_id,
            parent=parent,
            confirmed_complex_atoms=list(record["confirmed_atoms"]),
            complex_pdb=kept_complexes[target_id],
            quality=quality[target_id],
            feature_factory=feature_factory,
            parent_to_canonical=parent_to_canonical,
            canonical_atom_mapping=canonical_atom_mapping,
        )
    if not targets:
        raise SystemExit("No quality-kept consensus target produced a valid interaction-anchor map")
    parent_no_h = Chem.RemoveHs(parent)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "mapping_algorithm": ALGORITHM_VERSION,
        "consensus_sha256": sha256_file(consensus_json),
        "boltz_report_sha256": sha256_file(boltz_report),
        "sources": {
            "parent_sdf": {
                "path": str(parent_sdf.resolve()),
                "bytes": parent_sdf.stat().st_size,
                "sha256": sha256_file(parent_sdf),
            },
            "consensus_json": {
                "path": str(consensus_json.resolve()),
                "bytes": consensus_json.stat().st_size,
                "sha256": sha256_file(consensus_json),
            },
            "boltz_report": {
                "path": str(boltz_report.resolve()),
                "bytes": boltz_report.stat().st_size,
                "sha256": sha256_file(boltz_report),
            },
        },
        "parent": {
            "sdf_sha256": sha256_file(parent_sdf),
            "canonical_smiles": canonical_smiles,
            "inchikey": inchi.MolToInchiKey(parent_no_h),
            "atom_count": parent.GetNumAtoms(),
            "heavy_atom_count": parent_no_h.GetNumAtoms(),
            "canonical_atom_count": canonical_parent.GetNumAtoms(),
        },
        "targets": targets,
        "excluded_targets": excluded,
        "claim_eligible": True,
        "analog_pose_verified": False,
    }
    validate_anchor_map(
        payload,
        path=Path("<in-memory-interaction-anchor-map>"),
        expected_parent_sha256=sha256_file(parent_sdf),
        expected_consensus_sha256=sha256_file(consensus_json),
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-sdf", required=True, type=Path)
    parser.add_argument("--boltz-report", required=True, type=Path)
    parser.add_argument("--consensus-json", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out_json.unlink(missing_ok=True)
    payload = build_payload(args.parent_sdf, args.boltz_report, args.consensus_json)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out_json.with_suffix(args.out_json.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(args.out_json)
    LOG.info("Wrote %s (n_targets=%d)", args.out_json, len(payload["targets"]))


if __name__ == "__main__":
    main()
