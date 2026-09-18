#!/usr/bin/env python3
"""Count the ligands the retrieval-index standardizer cannot process, per split.

The production index build stopped on one SMILES in dev/test that RDKit parses
but cannot give an InChIKey - `C:CC(=O)N...`, where `C:C` is an aromatic bond
between two atoms that are not in a ring. The builder is fail-closed on that,
which is right for a train-only index bound to a benchmark hash, but a single
malformed record should not be able to block a production rebuild of 1.37M rows.

Before deciding an allowance, measure how many there actually are.

Run:
    python scripts/audit_unusable_ligands.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

BENCHMARK = ROOT / "data" / "activity_benchmark_202608"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits",
        default="train,dev,test",
        help="comma-separated benchmark splits to audit",
    )
    parser.add_argument("--json", type=Path, help="write the audit as JSON")
    args = parser.parse_args()

    # The whole path the builder runs, not just the parse: the failure that
    # stopped the production build was in InChIKey derivation, one step later.
    from build_activity_retrieval_index import _standardize_ligand

    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    report: dict[str, dict[str, object]] = {}
    for split in [name.strip() for name in args.splits.split(",") if name.strip()]:
        path = BENCHMARK / f"{split}.parquet"
        if not path.exists():
            print(f"{split}: missing")
            continue
        frame = pd.read_parquet(path, columns=["ligand_smiles"])
        unique = sorted(set(frame["ligand_smiles"].astype(str)))
        failures = []
        for smiles in unique:
            try:
                _standardize_ligand(smiles)
            except Exception as error:  # noqa: BLE001 - we are cataloguing them
                failures.append({"smiles": smiles, "error": str(error)[:200]})
        rows_lost = int(
            frame["ligand_smiles"].astype(str).isin({f["smiles"] for f in failures}).sum()
        )
        report[split] = {
            "unique_ligands": len(unique),
            "unusable_ligands": len(failures),
            "rows_affected": rows_lost,
            "examples": failures[:10],
        }
        share = 100.0 * len(failures) / len(unique) if unique else 0.0
        print(
            f"{split:6s} unique={len(unique):>7,}  unusable={len(failures):>4,} "
            f"({share:.4f}%)  rows lost={rows_lost:>5,}"
        )
        for failure in failures[:5]:
            print(f"         {failure['smiles'][:90]}")

    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
