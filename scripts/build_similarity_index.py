#!/usr/bin/env python3
"""Build a compact index for "which measured molecules look like mine".

The retrieval index answers a different question - which *targets* to rank - and
paying its cost to answer this one is not reasonable: it takes 27 s to open and
holds 3.5 GB, and the Workbench would have to carry that whether or not anyone
asks for a similar compound.

So this writes the two things a similarity lookup actually needs, and nothing
else: the fingerprints as packed bits, and one row of context per ligand. 272 MB
and about half a second per query, against 1.06M molecules.

The context matters as much as the similarity. A close analogue whose own
measurement sits under the activity threshold is a different thing from a close
analogue measured potent, and a list sorted by Tanimoto alone hides that - EGCG's
nearest MMP2 analogues are 0.727 similar and were measured at pActivity 4.06.
Every row therefore carries what its evidence actually said.

Run:
    python scripts/build_similarity_index.py \\
        --index-dir data/activity_retrieval_runtime_merged_202608 \\
        --out-dir data/similarity_index_202609
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "skinscout.similarity-index.v2"
POSITIVE_THRESHOLD = 6.0
NEGATIVE_THRESHOLD = 5.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _packed_bits(bitvec_column: pd.Series) -> np.ndarray:
    """uint64 words as stored -> packed uint8, 2048 bits per ligand."""
    words = np.array([[np.uint64(int(w)) for w in row] for row in bitvec_column], dtype=np.uint64)
    if words.shape[1] != 32:
        raise SystemExit(f"expected 32 uint64 words per fingerprint, got {words.shape[1]}")
    return words.view(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    ligands = pd.read_parquet(
        args.index_dir / "ligands.parquet",
        columns=["ligand_index", "standard_inchikey", "canonical_smiles", "bitvec"],
    ).sort_values("ligand_index")
    edges = pd.read_parquet(
        args.index_dir / "edges.parquet",
        columns=["ligand_index", "uniprot", "max_pactivity", "positive_measurement_count",
                 "negative_measurement_count"],
    )

    grouped = edges.groupby("ligand_index", sort=True)
    context = pd.DataFrame(
        {
            "target_count": grouped["uniprot"].nunique(),
            "best_pactivity": grouped["max_pactivity"].max(),
            "positive_edges": grouped["positive_measurement_count"].apply(lambda s: int((s > 0).sum())),
            "negative_edges": grouped["negative_measurement_count"].apply(lambda s: int((s > 0).sum())),
        }
    )
    # The three targets it is best measured against, for a reader deciding
    # whether this molecule is worth looking up at all. Deduplicated first: a
    # ligand often has several measurements against one target, and without the
    # drop the "top three targets" came back as the same UniProt three times.
    top = (
        edges.sort_values("max_pactivity", ascending=False)
        .drop_duplicates(["ligand_index", "uniprot"])
        .groupby("ligand_index", sort=True)["uniprot"]
        .apply(lambda s: ";".join(s.head(3)))
        .rename("top_targets")
    )

    # Every target this ligand was measured against, not only the best three.
    # Without the full set there is no way to answer the question a substitution
    # actually turns on - "was this candidate measured against the same protein
    # as my compound?" - and a top-three overlap check would answer it wrongly
    # for any ligand with more than three targets. 10.5 MB of strings.
    all_targets = (
        edges[["ligand_index", "uniprot"]]
        .drop_duplicates()
        .groupby("ligand_index", sort=True)["uniprot"]
        .apply(";".join)
        .rename("all_targets")
    )

    table = ligands.set_index("ligand_index").join([context, top, all_targets], how="left")
    table["target_count"] = table["target_count"].fillna(0).astype(int)
    table["top_targets"] = table["top_targets"].fillna("")
    table["all_targets"] = table["all_targets"].fillna("")

    def _label(row) -> str:
        if row["target_count"] == 0 or pd.isna(row["best_pactivity"]):
            return "none"
        if row["best_pactivity"] >= POSITIVE_THRESHOLD:
            return "at_or_above_threshold"
        if row["best_pactivity"] <= NEGATIVE_THRESHOLD:
            return "below_threshold"
        return "between_thresholds"

    table["evidence"] = table.apply(_label, axis=1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    packed = _packed_bits(table["bitvec"])
    np.save(args.out_dir / "fingerprints.npy", packed)

    meta = table.drop(columns=["bitvec"]).reset_index()
    meta["best_pactivity"] = meta["best_pactivity"].astype(float)
    meta.to_parquet(args.out_dir / "ligands.parquet", index=False)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "ligands": int(len(meta)),
        "fingerprint_bits": 2048,
        "label_policy": {
            "at_or_above_threshold": f"best measured pActivity >= {POSITIVE_THRESHOLD}",
            "below_threshold": f"best measured pActivity <= {NEGATIVE_THRESHOLD}",
            "between_thresholds": "measured between the two",
            "none": "in the reference set with no usable measurement",
        },
        "source_index": {
            "path": str(args.index_dir.resolve()),
            "manifest_sha256": _sha256(args.index_dir / "manifest.json"),
        },
        "outputs": {
            "fingerprints": {"bytes": int(packed.nbytes)},
            "ligands": {"rows": int(len(meta))},
        },
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"ligands {len(meta):,}  fingerprints {packed.nbytes/1e6:.0f} MB")
    print(f"wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
