#!/usr/bin/env python3
"""stage5_5_consensus.py - PLIP/ProLIF bound-pose interaction atom consensus.

Input signal layout, per target with ligand atom indexing:
  PLIP   XML -> atoms participating in bound-pose interactions
  ProLIF CSV -> atoms participating in bound-pose interactions

Rule:
    An atom is confirmed when both executable evidence sources flag it.

Output:
  consensus_pharmacophore_atoms.json - target map preserving the historical
      `confirmed_atoms` field for compatibility. The atoms are explicitly
      labelled as pose-supported interactions in the 0-based bound-complex
      ligand atom order; they are not a SMARTS pattern or a causal
      pharmacophore claim.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

LOG = logging.getLogger("stage5_5.consensus")
EVIDENCE_LABEL = "pose-supported interaction atoms"
COORDINATE_SYSTEM = "boltz_complex_ligand_atom_order_0_based"
CONSENSUS_SOURCES = ("plip", "prolif")


def confirm_atoms(votes: dict[str, set[int]], min_votes: int = 2) -> set[int]:
    """Return atom indices confirmed by at least `min_votes` sources."""
    tally: dict[int, int] = {}
    for atoms in votes.values():
        for atom_idx in atoms:
            tally[atom_idx] = tally.get(atom_idx, 0) + 1
    return {atom_idx for atom_idx, count in tally.items() if count >= min_votes}


def parse_plip(xml_path: Path) -> dict[str, set[int]]:
    """Return target_id -> ligand atom index set."""
    import xml.etree.ElementTree as ET

    if not xml_path.exists():
        return {}
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as exc:
        raise SystemExit(f"PLIP XML is malformed: {xml_path}") from exc

    out: dict[str, set[int]] = {}
    for target in root.findall(".//target"):
        target_id = (target.get("id") or "").strip()
        if not target_id:
            raise SystemExit(f"PLIP target missing required id: {xml_path}")
        if target_id in out:
            raise SystemExit(
                f"PLIP evidence contains duplicate target id values: {target_id}: {xml_path}"
            )
        atoms: set[int] = set()
        for atom_el in target.findall(".//ligand_atom"):
            raw_idx = atom_el.get("idx")
            if raw_idx is None:
                raise SystemExit(
                    f"PLIP ligand_atom missing required idx for target {target_id}: {xml_path}"
                )
            try:
                atom_idx = int(raw_idx)
            except ValueError as exc:
                raise SystemExit(
                    f"PLIP ligand_atom idx must be an integer: {target_id}:{raw_idx}: {xml_path}"
                ) from exc
            if atom_idx < 0:
                raise SystemExit(
                    "PLIP ligand_atom idx must be a non-negative integer: "
                    f"{target_id}:{raw_idx}: {xml_path}"
                )
            if atom_idx in atoms:
                raise SystemExit(
                    "PLIP evidence contains duplicate ligand_atom idx values: "
                    f"{target_id}:{atom_idx}: {xml_path}"
                )
            atoms.add(atom_idx)
        if atoms:
            out[target_id] = atoms
    return out


def parse_prolif(csv_path: Path) -> dict[str, set[int]]:
    if not csv_path.exists():
        return {}

    import pandas as pd

    df = pd.read_csv(csv_path)
    required_columns = {"target_id", "atom_idx"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise SystemExit(
            "ProLIF CSV missing required column(s): "
            f"{', '.join(missing_columns)}: {csv_path}"
        )

    raw_targets = df["target_id"]
    normalized_targets = raw_targets.fillna("").astype(str).str.strip()
    blank_targets = raw_targets.isna() | normalized_targets.eq("")
    if blank_targets.any():
        raise SystemExit(f"ProLIF target_id must be non-empty: {csv_path}")

    raw_atoms = df["atom_idx"]
    normalized_atom_values: list[int] = []
    bad_values: set[str] = set()
    negative_values: set[str] = set()
    for target_id, atom in zip(normalized_targets, raw_atoms, strict=False):
        text = str(atom).strip()
        try:
            value = float(text)
        except ValueError:
            bad_values.add(f"{target_id}:{atom}")
            continue
        if not math.isfinite(value) or not value.is_integer():
            bad_values.add(f"{target_id}:{atom}")
            continue
        atom_idx = int(value)
        if atom_idx < 0:
            negative_values.add(f"{target_id}:{atom}")
            continue
        normalized_atom_values.append(atom_idx)

    if bad_values:
        shown = ", ".join(sorted(bad_values)[:10])
        suffix = "..." if len(bad_values) > 10 else ""
        raise SystemExit(f"ProLIF atom_idx must be an integer: {shown}{suffix}: {csv_path}")
    if negative_values:
        shown = ", ".join(sorted(negative_values)[:10])
        suffix = "..." if len(negative_values) > 10 else ""
        raise SystemExit(
            "ProLIF atom_idx must be a non-negative integer: "
            f"{shown}{suffix}: {csv_path}"
        )

    normalized_atoms = pd.Series(normalized_atom_values, index=df.index, dtype=int)
    duplicate_mask = pd.DataFrame(
        {"target_id": normalized_targets, "atom_idx": normalized_atoms},
    ).duplicated()
    if duplicate_mask.any():
        duplicate_keys = sorted(
            {
                f"{target_id}:{atom_idx}"
                for target_id, atom_idx in zip(
                    normalized_targets[duplicate_mask],
                    normalized_atoms[duplicate_mask],
                    strict=False,
                )
            },
        )
        shown = ", ".join(duplicate_keys[:10])
        suffix = "..." if len(duplicate_keys) > 10 else ""
        raise SystemExit(
            "ProLIF evidence contains duplicate target_id/atom_idx rows: "
            f"{shown}{suffix}: {csv_path}"
        )

    out: dict[str, set[int]] = {}
    normalized_df = df.assign(target_id=normalized_targets, atom_idx=normalized_atoms)
    for target_id, sub in normalized_df.groupby("target_id"):
        out[str(target_id)] = set(sub["atom_idx"])
    return out


def consensus_for_all(
    plip: dict[str, set[int]],
    prolif: dict[str, set[int]],
    min_votes: int = 2,
    *,
    degraded: bool = False,
) -> dict[str, dict[str, Any]]:
    targets = set(plip) | set(prolif)
    out: dict[str, dict[str, Any]] = {}
    for target_id in sorted(targets):
        votes = {
            "plip": plip.get(target_id, set()),
            "prolif": prolif.get(target_id, set()),
        }
        atoms = sorted(confirm_atoms(votes, min_votes=min_votes))
        out[target_id] = {
            "confirmed_atoms": atoms,
            "pose_supported_interaction_atoms": atoms,
            "evidence_label": EVIDENCE_LABEL,
            "coordinate_system": COORDINATE_SYSTEM,
            "evidence_sources": list(CONSENSUS_SOURCES),
            "consensus_min_votes": min_votes,
            "degraded": degraded,
            "claim_eligible": (
                not degraded
                and min_votes == len(CONSENSUS_SOURCES)
                and bool(atoms)
            ),
            "votes": {source: sorted(values) for source, values in votes.items()},
        }
    return out


def targets_with_all_sources(source_payloads: dict[str, dict[str, set[int]]]) -> set[str]:
    """Return target ids backed by non-empty evidence from every source."""
    target_sets = [
        {target_id for target_id, atoms in payload.items() if atoms}
        for payload in source_payloads.values()
    ]
    if not target_sets:
        return set()
    return set.intersection(*target_sets)


def require_complete_target_coverage(
    source_payloads: dict[str, dict[str, set[int]]],
) -> set[str]:
    """Fail closed unless every source has the same non-empty target set."""
    empty_sources = [name for name, payload in source_payloads.items() if not payload]
    if empty_sources:
        raise SystemExit(
            "Pose-supported interaction atom evidence is empty for: "
            + ", ".join(empty_sources)
        )

    target_sets = {
        name: {target_id for target_id, atoms in payload.items() if atoms}
        for name, payload in source_payloads.items()
    }
    if any(not targets for targets in target_sets.values()):
        empty_after_filter = [name for name, targets in target_sets.items() if not targets]
        raise SystemExit(
            "Pose-supported interaction atom evidence has no non-empty targets for: "
            + ", ".join(empty_after_filter)
        )
    first_targets = next(iter(target_sets.values()))
    if any(targets != first_targets for targets in target_sets.values()):
        details = "; ".join(
            f"{name}={','.join(sorted(targets)) or '<none>'}"
            for name, targets in sorted(target_sets.items())
        )
        raise SystemExit(
            "Stage 5.5 requires complete same-target PLIP and ProLIF coverage: "
            + details
        )
    return set(first_targets)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plip", required=True, type=Path)
    parser.add_argument("--prolif", required=True, type=Path)
    parser.add_argument("--min-votes", type=int, default=2)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument(
        "--allow-empty-sources",
        action="store_true",
        help="Permit empty source evidence only for explicit degraded diagnostics.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args.out_json.unlink(missing_ok=True)

    if args.min_votes <= 0:
        raise SystemExit("--min-votes must be positive")
    if args.min_votes > len(CONSENSUS_SOURCES):
        raise SystemExit("--min-votes cannot exceed the 2 available Stage 5.5 evidence sources")

    missing = [str(path) for path in (args.plip, args.prolif) if not path.exists()]
    if missing:
        raise SystemExit(
            "Required pose-supported interaction atom source file(s) missing: "
            + ", ".join(missing)
        )

    source_payloads = {
        "plip": parse_plip(args.plip),
        "prolif": parse_prolif(args.prolif),
    }
    if not args.allow_empty_sources:
        require_complete_target_coverage(source_payloads)

    payload = consensus_for_all(
        source_payloads["plip"],
        source_payloads["prolif"],
        min_votes=args.min_votes,
        degraded=args.allow_empty_sources,
    )
    if not any(info["confirmed_atoms"] for info in payload.values()) and not args.allow_empty_sources:
        raise SystemExit("No pose-supported interaction atoms reached the consensus vote threshold")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    tmp_json = args.out_json.with_suffix(args.out_json.suffix + ".tmp")
    tmp_json.write_text(json.dumps(payload, indent=2) + "\n")
    tmp_json.replace(args.out_json)
    LOG.info("Wrote %s (n_targets=%d)", args.out_json, len(payload))


if __name__ == "__main__":
    main()
