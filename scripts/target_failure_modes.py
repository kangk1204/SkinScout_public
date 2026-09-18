#!/usr/bin/env python3
"""Attach documented docking failure modes to predicted targets.

`results/RETROSPECTIVE.md` traced every docking miss on the validation panel to
one of two causes the receptor preparation cannot model: a metal cofactor that
AlphaFold does not place (TYR-Cu, MMP-Zn), or a GPCR that AlphaFold predicts in
its inactive state. A docking score on such a receptor is not evidence of weak
binding, and nothing in the published artifacts said so.

The two causes are detected differently on purpose. Metal cofactors come from a
curated list with a named source, because the class labels do not identify them
- TYR is not annotated as a metalloprotease, so class inference would miss the
best-documented case while asserting cofactors for proteins nobody checked.
The GPCR inactive-state limitation is a property of the whole class, so it is
read off the HPA `Molecular function` annotation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

CURATED_DEFAULT = Path("data/validation/known_failure_modes.csv")
CURATED_COLUMNS = ("uniprot", "gene", "failure_mode", "cofactor", "evidence_source")
GPCR_ANNOTATION = "G-protein coupled receptor"


def load_curated(path: Path | None = None) -> dict[str, dict[str, str]]:
    """Read the curated failure-mode list, failing closed on a malformed file."""
    source = path or CURATED_DEFAULT
    if not source.exists() or source.stat().st_size == 0:
        raise SystemExit(f"Known failure-mode CSV is required and must be non-empty: {source}")
    with source.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(set(CURATED_COLUMNS) - set(reader.fieldnames or []))
        if missing:
            raise SystemExit(
                f"Known failure-mode CSV {source} is missing column(s): " + ", ".join(missing)
            )
        records: dict[str, dict[str, str]] = {}
        for index, row in enumerate(reader):
            uniprot = (row.get("uniprot") or "").strip()
            if not uniprot:
                raise SystemExit(f"Known failure-mode CSV {source} has a blank uniprot at row {index}")
            if uniprot in records:
                raise SystemExit(f"Known failure-mode CSV {source} repeats uniprot {uniprot}")
            for column in CURATED_COLUMNS:
                if not (row.get(column) or "").strip():
                    raise SystemExit(
                        f"Known failure-mode CSV {source} has a blank {column} for {uniprot}"
                    )
            records[uniprot] = {column: (row[column] or "").strip() for column in CURATED_COLUMNS}
    return records


def is_gpcr(molecular_function: object) -> bool:
    return GPCR_ANNOTATION in str(molecular_function or "")


def failure_modes_for(
    target_id: str,
    *,
    molecular_function: object = "",
    curated: dict[str, dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Return every documented reason this target's docking score is unreliable."""
    modes: list[dict[str, str]] = []
    record = (curated or {}).get(str(target_id).strip())
    if record is not None:
        modes.append(
            {
                "failure_mode": record["failure_mode"],
                "detail": record["cofactor"],
                "basis": "curated",
                "evidence_source": record["evidence_source"],
            }
        )
    if is_gpcr(molecular_function):
        modes.append(
            {
                "failure_mode": "gpcr_inactive_state",
                "detail": "AlphaFold models GPCRs in the inactive state",
                "basis": "hpa_molecular_function",
                "evidence_source": "results/RETROSPECTIVE.md",
            }
        )
    return modes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--molecular-function", default="")
    parser.add_argument("--curated-csv", type=Path, default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            failure_modes_for(
                args.target_id,
                molecular_function=args.molecular_function,
                curated=load_curated(args.curated_csv),
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
