#!/usr/bin/env python3
"""Build a production retrieval index from the run-path evidence, not the benchmark.

The benchmark is deliberately narrow: it exists to measure, so it filters hard
and splits by date. Building the production index from it inherits both, and
that costs real evidence. alpha-arbutin's best tyrosinase analogue,
`CHEMBL1760462` - IC50, pChEMBL 6.11, assay type B, confidence 8, human, single
protein, PMID and DOI present - is in the ChEMBL mirror and in none of the three
benchmark splits, which is most of why arbutin fell from rank 2 to 70 when Stage
3 moved onto the index.

So: evaluation reads the benchmark index, narrow and dated. Production reads
this one, built from the same table a run already retrieves against. Both are
marked by `index_role`, and `load_reference_index` refuses a production index
unless the caller opts in.

Run:
    python scripts/build_runtime_retrieval_index.py \\
        --evidence-dir data/chembl37 \\
        --out-dir data/activity_retrieval_runtime_202608
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from build_activity_retrieval_index import (
    ENDPOINT_FAMILIES,
    _build_edges,
    _build_ligands,
    _write_json_atomic,
    _write_parquet_atomic,
)

SCHEMA_VERSION = "skinscout.activity-retrieval-index.production.v1"
LOG = logging.getLogger("runtime-retrieval-index")
ROOT = Path(__file__).resolve().parents[1]

# ChEMBL writes Ki/Kd in mixed case; the index keys on the upper-case family.
ENDPOINT_ALIASES = {"KI": "KI", "KD": "KD", "IC50": "IC50", "EC50": "EC50"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publication_key(frame: pd.DataFrame) -> pd.Series:
    """One stable citation per row, preferring the most specific identifier.

    The index counts distinct publications per edge, so a blank key would make
    unrelated measurements look like one repeated source. Built with explicit
    string coercion at every step: a NaN leaking through here silently drops the
    row later, which cost a rebuild to find.
    """
    blank = {"", "nan", "none", "<na>", "null"}

    def _clean(name: str) -> pd.Series:
        if name not in frame.columns:
            return pd.Series("", index=frame.index, dtype=object)
        text = frame[name].fillna("").map(lambda value: str(value).strip())
        # .0 suffixes come from ints round-tripped through float columns.
        text = text.str.replace(r"\.0$", "", regex=True)
        return text.mask(text.str.lower().isin(blank), "")

    keys = pd.Series("", index=frame.index, dtype=object)
    # A merged table carries its own key: BindingDB cites by patent or article id
    # more often than by PMID, and those identifiers have no ChEMBL column to
    # live in. Rows without one fall through to the ChEMBL fields below.
    prepared = _clean("publication_key")
    keys.loc[prepared != ""] = prepared.loc[prepared != ""]
    for name, prefix in (("pubmed_id", "pmid:"), ("doi", "doi:"), ("document_chembl_id", "chembl:")):
        text = _clean(name)
        fill = (keys == "") & (text != "")
        keys.loc[fill] = prefix + text.loc[fill]
    assert not keys.isna().any(), "publication key must never be NaN"
    return keys



def _evidence_licensing(evidence_dir: Path) -> dict[str, Any]:
    """Carry the evidence table's licence tier onto the index.

    A merged table can pull in share-alike sources. Once the index is built the
    original table is a directory name away, so the obligation has to travel with
    it - otherwise the only record of what a run's rankings are derived from is
    a path someone has to remember to check.
    """
    manifest = evidence_dir / "manifest.json"
    if not manifest.exists():
        # data/chembl37 has no merge manifest: it is the ChEMBL mirror as shipped.
        return {"sources_tier": "chembl-only", "licences": ["CC BY-SA 3.0 (ChEMBL)"]}
    record = json.loads(manifest.read_text(encoding="utf-8"))
    return {
        "sources_tier": record.get("sources_tier"),
        "sources": record.get("sources"),
        "licences": record.get("licences"),
        "evidence_manifest": str(manifest.resolve()),
    }


def project_runtime_evidence(
    path: Path, target_universe: set[str] | None = None
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Project the run-path activity table into the index builder's schema."""
    frame = pd.read_parquet(path)
    stats = {"input_rows": int(len(frame))}

    endpoint = frame["act_type"].astype(str).str.strip().str.upper()
    supported = endpoint.isin(ENDPOINT_ALIASES)
    stats["dropped_endpoint"] = int((~supported).sum())
    frame = frame[supported]
    endpoint = endpoint[supported].map(ENDPOINT_ALIASES)

    pactivity = pd.to_numeric(frame["pchembl"], errors="coerce")
    finite = pactivity.notna() & np.isfinite(pactivity)
    stats["dropped_no_pactivity"] = int((~finite).sum())
    frame = frame[finite]
    endpoint = endpoint[finite]
    pactivity = pactivity[finite]

    projected = pd.DataFrame(
        {
            # The builder validates this column; a single value keeps it honest
            # about what this index is - not a temporal split.
            "split": "runtime",
            "source_db": frame["source_db"].astype(str).str.strip(),
            "uniprot": frame["uniprot"].astype(str).str.strip(),
            "ligand_smiles": frame["smiles"].astype(str).str.strip(),
            "publication_key": _publication_key(frame),
            "endpoint": endpoint.values,
            "pactivity": pactivity.astype(float).values,
        }
    )
    usable = (
        projected["uniprot"].str.len().gt(0)
        & projected["ligand_smiles"].str.len().gt(0)
        & projected["publication_key"].astype(str).str.len().gt(0)
        & projected["source_db"].str.len().gt(0)
    )
    stats["dropped_incomplete"] = int((~usable).sum())
    projected = projected[usable]
    if target_universe is not None:
        # The mirror carries targets the screenable universe excludes; keeping
        # them makes an index the reader rejects rather than a richer one.
        in_universe = projected["uniprot"].isin(target_universe)
        stats["dropped_outside_universe"] = int((~in_universe).sum())
        projected = projected[in_universe]
    projected = projected.reset_index(drop=True)
    stats["projected_rows"] = int(len(projected))
    stats["targets"] = int(projected["uniprot"].nunique())
    unsupported = sorted(set(projected["endpoint"]) - set(ENDPOINT_FAMILIES))
    if unsupported:
        raise SystemExit(f"projection produced unsupported endpoints: {unsupported}")
    return projected, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--negative-threshold", type=float, default=5.0)
    parser.add_argument("--positive-threshold", type=float, default=6.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-unusable-ligands", type=int, default=0)
    parser.add_argument(
        "--target-clusters",
        type=Path,
        default=ROOT / "data" / "evidence_splits" / "screenable_target_clusters_2026_02.csv",
        help="restrict to the screenable target universe the reader validates against",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not np.isfinite(args.negative_threshold):
        raise SystemExit("--negative-threshold must be finite")
    if not np.isfinite(args.positive_threshold):
        raise SystemExit("--positive-threshold must be finite")
    if args.negative_threshold >= args.positive_threshold:
        raise SystemExit("--negative-threshold must be less than --positive-threshold")
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    if args.max_unusable_ligands < 0:
        raise SystemExit("--max-unusable-ligands must be non-negative")

    source = args.evidence_dir / "human_activities.parquet"
    if not source.exists():
        raise SystemExit(f"run-path evidence table is missing: {source}")

    if not args.target_clusters.is_file() or args.target_clusters.stat().st_size == 0:
        raise SystemExit(
            f"--target-clusters is required and must be non-empty: {args.target_clusters}"
        )
    clusters = pd.read_csv(args.target_clusters)
    if "uniprot" not in clusters.columns:
        raise SystemExit("--target-clusters must contain a 'uniprot' column")
    cluster_ids = clusters["uniprot"].fillna("").astype(str).str.strip()
    if cluster_ids.eq("").any() or cluster_ids.duplicated().any():
        raise SystemExit("--target-clusters contains blank or duplicate UniProt IDs")
    universe = set(cluster_ids)
    print(f"  target universe          {len(universe):>10,}")
    projected, stats = project_runtime_evidence(source, universe)
    for key, value in stats.items():
        print(f"  {key:22s} {value:>10,}")
    if args.dry_run:
        print("\n(dry run - nothing written)")
        return 0

    ligands, raw_smiles_to_key, unusable = _build_ligands(
        projected, workers=args.workers, max_unusable=args.max_unusable_ligands
    )
    edges = _build_edges(
        projected,
        ligands,
        raw_smiles_to_key,
        args.negative_threshold,
        args.positive_threshold,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ligands_path = args.out_dir / "ligands.parquet"
    edges_path = args.out_dir / "edges.parquet"
    _write_parquet_atomic(ligands, ligands_path)
    _write_parquet_atomic(edges, edges_path)

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "index_role": "production",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": (
            "built from the run-path activity table, not the benchmark splits; "
            "evaluation must not read this index"
        ),
        "inputs": {
            "runtime_evidence": {
                "path": str(source.resolve()),
                "sha256": _sha256(source),
                "rows": stats["input_rows"],
            },
            "target_clusters": {
                "path": str(args.target_clusters.resolve()),
                "sha256": _sha256(args.target_clusters),
                "rows": int(len(clusters)),
            },
        },
        "outputs": {
            "ligands": {
                "path": str(ligands_path.resolve()),
                "sha256": _sha256(ligands_path),
                "rows": int(len(ligands)),
            },
            "edges": {
                "path": str(edges_path.resolve()),
                "sha256": _sha256(edges_path),
                "rows": int(len(edges)),
            },
        },
        "projection": stats,
        "evidence_licensing": _evidence_licensing(args.evidence_dir),
        "unusable_ligands": {"count": len(unusable), "records": unusable},
        "label_policy": {
            "negative_threshold": args.negative_threshold,
            "positive_threshold": args.positive_threshold,
        },
        "algorithm": {
            "standardization": [
                "rdMolStandardize.FragmentParent",
                "rdMolStandardize.Uncharger",
                "canonical isomeric SMILES",
            ]
        },
        "source_counts": dict(sorted(Counter(projected["source_db"]).items())),
    }
    _write_json_atomic(manifest, args.out_dir / "manifest.json")
    print(f"\nligands {len(ligands):,}  edges {len(edges):,}  targets {stats['targets']:,}")
    print(f"wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
