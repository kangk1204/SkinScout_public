#!/usr/bin/env python3
"""build_transfer_reachable_panel.py — the set pocket transfer can actually reach.

Route B lends a scorable target's retrieval score to its Foldseek cluster
mates. It therefore reaches exactly one population: targets with no evidence of
their own whose cluster contains a target that has some. A pocket-cold target
is by definition in a cluster with no training member, and the retrieval index
is built from train alone, so cold targets have no donor and transfer is
structurally unable to move them. Measuring Route B on a cold panel proves that
by construction rather than testing anything.

This builds the panel that can be measured: evaluation rows whose target has no
direct evidence but does have a donor in its pocket cluster.

    python eval/build_transfer_reachable_panel.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_BENCHMARK = ROOT / "data" / "activity_benchmark_202608"
DEFAULT_INDEX = ROOT / "data" / "activity_retrieval_202608"
DEFAULT_POCKETS = (
    ROOT / "data" / "pocket_clusters_screenable_202608" / "pocket_cluster_map.csv"
)
DEFAULT_OUT = ROOT / "data" / "transfer_reachable_panel_202608"
LEVELS = ("tm40", "tm50", "tm60")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, object]:
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"required provenance artifact is missing or empty: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "bytes": int(path.stat().st_size),
    }


def reachable_targets(
    pockets: pd.DataFrame, donors: set[str], level: str
) -> tuple[set[str], set[str]]:
    """(reachable, unreachable) among targets with a pocket but no evidence."""
    column = f"pocket_cluster_{level}"
    with_pocket = pockets[pockets[column].notna()]
    donor_clusters = set(with_pocket.loc[with_pocket.index.isin(donors), column])
    evidence_free = with_pocket[~with_pocket.index.isin(donors)]
    reachable = set(evidence_free[evidence_free[column].isin(donor_clusters)].index)
    unreachable = set(evidence_free.index) - reachable
    return reachable, unreachable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--pockets", type=Path, default=DEFAULT_POCKETS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    from build_pocket_cold_panel import NEEDED, to_ranking_panel

    benchmark_manifest = args.benchmark / "manifest.json"
    index_manifest = args.index / "manifest.json"
    edges_path = args.index / "edges.parquet"
    dev_path = args.benchmark / "dev.parquet"
    test_path = args.benchmark / "test.parquet"
    upstream = {
        "benchmark_manifest": _artifact(benchmark_manifest),
        "benchmark_dev": _artifact(dev_path),
        "benchmark_test": _artifact(test_path),
        "retrieval_index_manifest": _artifact(index_manifest),
        "retrieval_edges": _artifact(edges_path),
        "pocket_cluster_map": _artifact(args.pockets),
    }
    pockets = pd.read_csv(args.pockets).set_index("uniprot")
    donors = set(
        pd.read_parquet(edges_path, columns=["uniprot"]).uniprot.unique()
    )
    dev = pd.read_parquet(dev_path, columns=NEEDED)
    test = pd.read_parquet(test_path, columns=NEEDED)
    pool = pd.concat([dev, test], ignore_index=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "skinscout.transfer-reachable-panel.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": (
            "Evaluate pocket-cluster transfer on the only population it can move: "
            "targets with no direct evidence whose cluster contains a donor."
        ),
        "why_not_pocket_cold": (
            "A pocket-cold target's cluster has no training member and the "
            "retrieval index is built from train alone, so it has no donor. "
            "Transfer cannot move it, by construction rather than by measurement."
        ),
        "label_semantics": (
            "The benchmark holds positives only; these are recovery panels."
        ),
        "inputs": upstream,
        "populations": {},
        "views": {},
    }

    for level in LEVELS:
        reachable, unreachable = reachable_targets(pockets, donors, level)
        manifest["populations"][level] = {
            "donors": len(donors & set(pockets.index)),
            "reachable": len(reachable),
            "unreachable": len(unreachable),
        }
        panel = pool[pool.uniprot.isin(reachable)].copy()
        panel["pocket_cluster"] = panel.uniprot.map(pockets[f"pocket_cluster_{level}"])
        panel["transfer_level"] = level
        name = f"transfer_reachable_{level}"
        path = args.out_dir / f"{name}.parquet"
        temp = path.with_suffix(".parquet.tmp")
        panel.to_parquet(temp, index=False)
        temp.replace(path)

        ranking = to_ranking_panel(panel, split="test")
        ranking_path = args.out_dir / f"{name}_ranking.parquet"
        temp = ranking_path.with_suffix(".parquet.tmp")
        ranking.to_parquet(temp, index=False)
        temp.replace(ranking_path)

        manifest["views"][name] = {
            "rows": int(len(panel)),
            "targets": int(panel.uniprot.nunique()),
            "compounds": int(panel.ligand_inchikey.nunique()),
            "ranking_queries": int(len(ranking)),
            "sha256": _sha256(path),
            "ranking_sha256": _sha256(ranking_path),
        }
        print(
            f"{name:26s} 도달가능 표적 {len(reachable):5,} / 도달불가 {len(unreachable):5,} "
            f"| 평가 {len(panel):5,}행, 표적 {panel.uniprot.nunique():3d}, 질의 {len(ranking):4,}"
        )

    manifest_path = args.out_dir / "manifest.json"
    temp = manifest_path.with_suffix(".json.tmp")
    temp.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temp.replace(manifest_path)
    print(f"\n매니페스트 → {manifest_path}")


if __name__ == "__main__":
    main()
