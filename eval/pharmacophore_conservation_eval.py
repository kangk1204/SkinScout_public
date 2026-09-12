#!/usr/bin/env python3
"""Evaluate target-conditioned parent interaction-anchor retention in analogs.

Stage 5.5 interaction atoms are first mapped from each Boltz bound-complex
ligand into the parent SDF atom order by a validated sidecar. Analog scores are
then structural, feature-aware retention proxies; they are not analog bound-pose
or affinity validation.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from json import JSONDecodeError
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from interaction_anchor import (  # noqa: E402
    InteractionAnchorError,
    PRESERVATION_BASIS,
    load_anchor_map,
    score_anchor_preservation,
)

LOG = logging.getLogger("eval.pharmacophore_conservation")
COORDINATE_SYSTEM = "boltz_complex_ligand_atom_order_0_based"
EVIDENCE_LABEL = "pose-supported interaction atoms"
EVIDENCE_SOURCES = ("plip", "prolif")


def _validate_fraction_threshold(value: float, label: str) -> None:
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise SystemExit(
            f"{label} must be a finite value in [0, 1]: {value!r}"
        )


def _validate_analog_smiles(values: pd.Series, path: Path) -> list[str]:
    from rdkit import Chem

    valid: list[str] = []
    invalid_rows: list[int] = []
    for idx, value in values.items():
        smi = "" if pd.isna(value) else str(value).strip()
        if not smi or Chem.MolFromSmiles(smi) is None:
            invalid_rows.append(int(idx))
            continue
        valid.append(smi)
    if invalid_rows:
        shown = ", ".join(str(i) for i in invalid_rows[:10])
        suffix = "..." if len(invalid_rows) > 10 else ""
        raise SystemExit(
            f"Analog CSV contains invalid SMILES at row index(es) {shown}{suffix}: {path}"
        )
    if not valid:
        raise SystemExit(f"Analog CSV has no valid SMILES: {path}")
    return valid


def _require_nonempty_file(path: Path, label: str) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"{label} is required and must be non-empty: {path}")


def _validate_atom_indices(value: object, *, field: str, path: Path, target: str) -> list[int]:
    if not isinstance(value, list) or any(
        type(atom) is not int or atom < 0 for atom in value
    ):
        raise SystemExit(
            f"Consensus pharmacophore {field} must be a list of non-negative "
            f"integers: {path} entry={target!r}"
        )
    if len(value) != len(set(value)):
        raise SystemExit(
            f"Consensus pharmacophore {field} contains duplicate atom indices: "
            f"{path} entry={target!r}"
        )
    return value


def _read_confirmed_atoms(path: Path) -> dict[str, list[int]]:
    _require_nonempty_file(path, "Consensus pharmacophore JSON")
    try:
        consensus = json.loads(path.read_text())
    except JSONDecodeError as exc:
        raise SystemExit(f"Consensus pharmacophore JSON failed to parse: {path}: {exc}") from exc
    if not isinstance(consensus, dict):
        raise SystemExit(f"Consensus pharmacophore JSON must be an object: {path}")

    confirmed_by_target: dict[str, list[int]] = {}
    for key, info in consensus.items():
        if not key.strip():
            raise SystemExit(
                f"Consensus pharmacophore target ids must be non-empty: {path}"
            )
        if not isinstance(info, dict):
            raise SystemExit(
                "Consensus pharmacophore entries must be objects: "
                f"{path} entry={key!r}"
            )
        if "confirmed_atoms" not in info:
            raise SystemExit(
                "Consensus pharmacophore entry missing required field "
                f"'confirmed_atoms': {path} entry={key!r}"
            )
        confirmed_by_target[key] = _validate_atom_indices(
            info["confirmed_atoms"],
            field="confirmed_atoms",
            path=path,
            target=key,
        )

    if not any(confirmed_by_target.values()):
        raise SystemExit(f"Consensus pharmacophore contains no confirmed atoms: {path}")

    for key, info in consensus.items():
        confirmed = confirmed_by_target[key]
        if info.get("pose_supported_interaction_atoms") != confirmed:
            raise SystemExit(
                "Consensus pharmacophore pose_supported_interaction_atoms must "
                f"match confirmed_atoms: {path} entry={key!r}"
            )
        if info.get("evidence_label") != EVIDENCE_LABEL:
            raise SystemExit(
                f"Consensus pharmacophore evidence_label is invalid: {path} entry={key!r}"
            )
        if info.get("coordinate_system") != COORDINATE_SYSTEM:
            raise SystemExit(
                "Consensus pharmacophore coordinate_system must be "
                f"{COORDINATE_SYSTEM!r}: {path} entry={key!r}"
            )
        if info.get("evidence_sources") != list(EVIDENCE_SOURCES):
            raise SystemExit(
                "Consensus pharmacophore evidence_sources must be exactly "
                f"{list(EVIDENCE_SOURCES)!r}: {path} entry={key!r}"
            )
        if type(info.get("consensus_min_votes")) is not int or info["consensus_min_votes"] != 2:
            raise SystemExit(
                "Consensus pharmacophore consensus_min_votes must be integer 2: "
                f"{path} entry={key!r}"
            )
        if info.get("degraded") is not False:
            raise SystemExit(
                f"Consensus pharmacophore entry is degraded: {path} entry={key!r}"
            )
        if info.get("claim_eligible") is not True:
            raise SystemExit(
                f"Consensus pharmacophore entry is not claim eligible: {path} entry={key!r}"
            )

        votes = info.get("votes")
        if not isinstance(votes, dict) or set(votes) != set(EVIDENCE_SOURCES):
            raise SystemExit(
                "Consensus pharmacophore votes must contain exactly PLIP and ProLIF: "
                f"{path} entry={key!r}"
            )
        source_atoms = {
            source: _validate_atom_indices(
                votes[source],
                field=f"votes.{source}",
                path=path,
                target=key,
            )
            for source in EVIDENCE_SOURCES
        }
        expected = sorted(set(source_atoms["plip"]) & set(source_atoms["prolif"]))
        if confirmed != expected:
            raise SystemExit(
                "Consensus pharmacophore confirmed_atoms must equal the PLIP/ProLIF "
                f"intersection: {path} entry={key!r}"
            )
    return confirmed_by_target


def _read_analogs(path: Path) -> pd.DataFrame:
    _require_nonempty_file(path, "Analog CSV")
    try:
        analogs = pd.read_csv(path)
    except Exception as exc:
        raise SystemExit(f"Analog CSV failed to parse: {path}: {exc}") from exc
    if "smiles" not in analogs.columns:
        raise SystemExit(f"Analog CSV missing required column 'smiles': {path}")
    if analogs.empty:
        raise SystemExit(f"Analog CSV contains no rows: {path}")
    return analogs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-sdf", required=True, type=Path)
    parser.add_argument("--consensus-json", required=True, type=Path)
    parser.add_argument("--interaction-anchor-map", type=Path)
    parser.add_argument("--analogs-csv", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--out-detail-csv", type=Path)
    parser.add_argument("--min-preserved-fraction", type=float, default=0.80)
    parser.add_argument(
        "--allow-threshold-failure",
        action="store_true",
        help="write a below-threshold metric only for explicit diagnostics",
    )
    args = parser.parse_args()
    _validate_fraction_threshold(
        args.min_preserved_fraction,
        "--min-preserved-fraction",
    )
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.out_csv.exists():
        args.out_csv.unlink()
    if args.out_detail_csv is not None:
        args.out_detail_csv.unlink(missing_ok=True)
    from rdkit import Chem
    _require_nonempty_file(args.parent_sdf, "Parent SDF")
    sup = Chem.SDMolSupplier(str(args.parent_sdf), removeHs=False)
    parent = next((m for m in sup if m is not None), None)
    if parent is None:
        raise SystemExit(f"Parent SDF contains no readable molecules: {args.parent_sdf}")
    confirmed_by_target = _read_confirmed_atoms(args.consensus_json)
    analogs = _read_analogs(args.analogs_csv)
    valid_smiles = _validate_analog_smiles(analogs["smiles"], args.analogs_csv)
    if args.interaction_anchor_map is None:
        raise SystemExit(
            "Interaction-atom conservation evaluation is unavailable: Stage 5.5 "
            "indices use the Boltz bound-complex ligand atom order, while --parent-sdf "
            "uses the original SDF atom order. A validated target-specific atom-map "
            "sidecar is required; --allow-threshold-failure cannot override this contract."
        )
    try:
        anchor_map = load_anchor_map(
            args.interaction_anchor_map,
            parent_sdf=args.parent_sdf,
            consensus_json=args.consensus_json,
        )
    except InteractionAnchorError as exc:
        raise SystemExit(str(exc)) from exc

    target_records = anchor_map["targets"]
    for target_id, record in target_records.items():
        if target_id not in confirmed_by_target:
            raise SystemExit(
                f"Interaction-anchor target is absent from consensus: {target_id}"
            )
        if record["confirmed_complex_atoms"] != confirmed_by_target[target_id]:
            raise SystemExit(
                f"Interaction-anchor confirmed atoms do not match consensus for {target_id}"
            )

    from rdkit import Chem, RDConfig
    from rdkit.Chem import ChemicalFeatures

    feature_factory = ChemicalFeatures.BuildFeatureFactory(
        str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef")
    )
    detail_rows: list[dict[str, object]] = []
    scores: list[float] = []
    for analog_index, smiles in enumerate(valid_smiles):
        analog = Chem.MolFromSmiles(smiles)
        assert analog is not None
        analog_id = (
            str(analogs.iloc[analog_index]["analog_id"]).strip()
            if "analog_id" in analogs.columns
            and pd.notna(analogs.iloc[analog_index]["analog_id"])
            else f"analog_{analog_index + 1:06d}"
        )
        for target_id, record in sorted(target_records.items()):
            try:
                result = score_anchor_preservation(
                    parent,
                    analog,
                    list(record["confirmed_parent_atoms"]),
                    feature_factory,
                )
            except InteractionAnchorError as exc:
                raise SystemExit(
                    f"Interaction-anchor scoring failed for {analog_id}/{target_id}: {exc}"
                ) from exc
            scores.append(result.score)
            detail_rows.append(
                {
                    "analog_id": analog_id,
                    "smiles": smiles,
                    "target_id": target_id,
                    "anchor_count": result.anchor_count,
                    "preserved_anchor_count": result.preserved_anchor_count,
                    "anchor_preservation_score": result.score,
                    "mcs_atom_count": result.mcs_atom_count,
                    "mapping_count": result.mapping_count,
                    "mapping_ambiguous": result.mapping_ambiguous,
                    "mapping_truncated": result.mapping_truncated,
                    "evaluation_basis": result.basis,
                    "analog_pose_verified": False,
                    "claimable": False,
                }
            )
    if not scores:
        raise SystemExit("Interaction-anchor evaluation produced no analog-target scores")
    fraction = sum(scores) / len(scores)
    fully_preserved = sum(score == 1.0 for score in scores) / len(scores)
    passes = fraction >= args.min_preserved_fraction
    row = {
        "n_analogs": len(valid_smiles),
        "n_targets": len(target_records),
        "n_analog_target_pairs": len(scores),
        "preserved_fraction": fraction,
        "fully_preserved_pair_fraction": fully_preserved,
        "min_preserved_fraction": args.min_preserved_fraction,
        "passes_threshold": bool(passes),
        "evaluation_basis": PRESERVATION_BASIS,
        "target_conditioned": True,
        "analog_pose_verified": False,
        "claimable": False,
    }
    if not passes and not args.allow_threshold_failure:
        raise SystemExit(
            "Pharmacophore anchor preservation failed threshold: "
            f"preserved_fraction={fraction:.3f} < {args.min_preserved_fraction:.3f}; "
            "pass --allow-threshold-failure only for explicit diagnostics"
        )
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = args.out_csv.with_suffix(args.out_csv.suffix + ".tmp")
    pd.DataFrame([row]).to_csv(tmp_csv, index=False)
    tmp_csv.replace(args.out_csv)
    if args.out_detail_csv is not None:
        args.out_detail_csv.parent.mkdir(parents=True, exist_ok=True)
        tmp_detail = args.out_detail_csv.with_suffix(args.out_detail_csv.suffix + ".tmp")
        pd.DataFrame(detail_rows).to_csv(tmp_detail, index=False)
        tmp_detail.replace(args.out_detail_csv)
    LOG.info(
        "Target-conditioned anchor preservation: %.3f (passes >= %.3f? %s)",
        fraction,
        args.min_preserved_fraction,
        passes,
    )


if __name__ == "__main__":
    main()
