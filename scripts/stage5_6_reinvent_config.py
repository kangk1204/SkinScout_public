#!/usr/bin/env python3
"""Fail-closed gate for the not-yet-implemented REINVENT 4 integration.

SkinScout does not currently ship the custom REINVENT 4 scoring plugins or a
validated mapping from bound-complex ligand atom order back to the generator's
molecular representation. Generating a plausible-looking but invalid TOML
would silently discard the intended scientific constraints, so claim-capable
execution is blocked until both contracts are implemented and tested.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

LOG = logging.getLogger("stage5_6.reinvent_contract")

BLOCKERS = (
    "validated REINVENT 4 custom scoring plugin bundle is absent",
    "bound-complex ligand atom order is not mapped to the generator representation",
    "trained prior/agent model artifacts and an upstream-compatible staged-learning config are absent",
)


def smiles_from_sdf(sdf: Path) -> str:
    if not sdf.exists() or sdf.stat().st_size == 0:
        raise SystemExit(f"Input SDF is required and must be non-empty: {sdf}")
    from rdkit import Chem

    supplier = Chem.SDMolSupplier(str(sdf), removeHs=False)
    for molecule in supplier:
        if molecule is not None:
            return Chem.MolToSmiles(molecule, canonical=True)
    raise SystemExit(f"Input SDF contains no readable molecule: {sdf}")


def require_nonempty_file(path: Path, label: str) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"{label} is required and must be non-empty: {path}")


def write_diagnostic_outputs(
    *,
    out_status: Path,
    out_smi: Path,
    mode: str,
    num_generated: int,
    input_smiles: str,
) -> None:
    payload = {
        "schema_version": 1,
        "stage": "5.6",
        "component": "reinvent4_contract_gate",
        "execution_status": "blocked",
        "claim_eligible": False,
        "diagnostic_only": True,
        "requested_mode": mode,
        "requested_num_generated": num_generated,
        "input_smiles": input_smiles,
        "blockers": list(BLOCKERS),
    }
    out_status.parent.mkdir(parents=True, exist_ok=True)
    out_smi.parent.mkdir(parents=True, exist_ok=True)
    tmp_status = out_status.with_suffix(out_status.suffix + ".tmp")
    tmp_smi = out_smi.with_suffix(out_smi.suffix + ".tmp")
    tmp_status.write_text(json.dumps(payload, indent=2) + "\n")
    tmp_smi.write_text("")
    tmp_status.replace(out_status)
    tmp_smi.replace(out_smi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-sdf", required=True, type=Path)
    parser.add_argument("--consensus", required=True, type=Path)
    parser.add_argument(
        "--mode",
        default="mol2mol",
        choices=("mol2mol", "libinvent", "linkinvent"),
    )
    parser.add_argument("--num-generated", type=int, default=5000)
    parser.add_argument("--out-status", required=True, type=Path)
    parser.add_argument("--out-smi", required=True, type=Path)
    parser.add_argument(
        "--allow-empty-output",
        action="store_true",
        help="Emit an explicit blocked-status record and empty SMILES file for diagnostics.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args.out_status.unlink(missing_ok=True)
    args.out_smi.unlink(missing_ok=True)
    if args.num_generated <= 0:
        raise SystemExit("--num-generated must be positive")
    input_smiles = smiles_from_sdf(args.in_sdf)
    require_nonempty_file(args.consensus, "Pose-supported interaction consensus")

    blocker_text = "; ".join(BLOCKERS)
    if not args.allow_empty_output:
        raise SystemExit(
            "Stage 5.6 is unavailable for claim-capable execution: "
            f"{blocker_text}. No REINVENT config or analog output was generated."
        )

    write_diagnostic_outputs(
        out_status=args.out_status,
        out_smi=args.out_smi,
        mode=args.mode,
        num_generated=args.num_generated,
        input_smiles=input_smiles,
    )
    LOG.warning("Stage 5.6 emitted diagnostic-only blocked outputs: %s", args.out_status)


if __name__ == "__main__":
    main()
