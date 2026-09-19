#!/usr/bin/env python3
"""build_pocket_cold_panel.py — an evaluation panel pocket transfer cannot cheat on.

Route B in docs/COVERAGE_ROUTES_20260824.md borrows ligand evidence across
Foldseek pocket clusters, which makes any panel whose clusters overlap training
circular: the method is scored on targets it was handed the answer for. The
existing views hold out sequence clusters and scaffolds, not pocket clusters -
the dual-cold view reaches only 13 targets, and dev is 2.16% pocket-cold - so
there was nothing to measure Route B on.

Two views, following the benchmark's own convention that a cold view is "test
rows whose pair, publication and <thing> are absent from train/dev":

  strict   evaluation = test, reference = train + dev.  Directly comparable to
           target_cold30 / target_cold50 / dual_cold.
  extended evaluation = dev + test, reference = train.  Larger, and the price
           is that dev rows are used for evaluation; marked non-claimable so it
           cannot be quoted as a headline result.

Both are activity-recovery panels: this benchmark carries only positives, so
they answer "is the known target ranked highly", not "is this pair active".

    python eval/build_pocket_cold_panel.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = ROOT / "data" / "activity_benchmark_202608"
DEFAULT_POCKETS = (
    ROOT / "data" / "pocket_clusters_screenable_202608" / "pocket_cluster_map.csv"
)
DEFAULT_OUT = ROOT / "data" / "pocket_cold_panel_202608"
LEVELS = ("tm40", "tm50", "tm60")
NEEDED = [
    "benchmark_id", "evidence_id", "split", "uniprot", "publication_key",
    "ligand_id", "ligand_smiles", "ligand_inchikey", "scaffold_id",
    "target_cluster_30", "target_cluster_50",
    "endpoint", "standard_value_nm", "pactivity", "activity_class", "binary_label",
]


def _ligand_key(frame: pd.DataFrame) -> pd.Series:
    """Identical to the benchmark builder's key, so the two agree on a pair."""
    return (
        frame["ligand_inchikey"].fillna("").replace("", pd.NA)
        .fillna(frame["ligand_id"]).fillna(frame["ligand_smiles"]).str.upper()
    )


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


def _absent_flags(
    evaluation: pd.DataFrame, reference: pd.DataFrame
) -> pd.DataFrame:
    out = evaluation.copy()
    reference_pairs = set(zip(reference["uniprot"], _ligand_key(reference)))
    pairs = list(zip(out["uniprot"], _ligand_key(out)))
    out["absent_pair_from_reference"] = [pair not in reference_pairs for pair in pairs]
    out["absent_publication_from_reference"] = ~out["publication_key"].isin(
        set(reference["publication_key"])
    )
    return out


def build_view(
    evaluation: pd.DataFrame,
    reference: pd.DataFrame,
    pockets: pd.DataFrame,
    level: str,
) -> pd.DataFrame:
    column = f"pocket_cluster_{level}"
    if column not in pockets.columns:
        raise SystemExit(f"포켓 클러스터 맵에 {column} 열이 없습니다")
    frame = _absent_flags(evaluation, reference)
    frame[column] = frame["uniprot"].map(pockets[column])
    reference_clusters = set(
        pockets.reindex(reference["uniprot"].unique())[column].dropna()
    )
    # A target with no pocket at all is not "cold", it is unscoreable by this
    # route; excluding it keeps the panel about transfer rather than coverage.
    frame[f"absent_pocket_{level}_from_reference"] = (
        frame[column].notna() & ~frame[column].isin(reference_clusters)
    )
    selected = frame[
        frame["absent_pair_from_reference"]
        & frame["absent_publication_from_reference"]
        & frame[f"absent_pocket_{level}_from_reference"]
    ].copy()
    selected["pocket_cold_level"] = level
    return selected.reset_index(drop=True)


def to_ranking_panel(panel: pd.DataFrame, split: str) -> pd.DataFrame:
    """Compound-centric form the existing scorer accepts.

    eval/activity_retrieval_model.load_ranking_panel wants one row per query
    with its truth targets, so the panel is emitted in both shapes: the
    edge-centric one for auditing and this one for scoring.
    """
    import hashlib

    from activity_retrieval_model import query_features

    rows = []
    for (inchikey, smiles), group in panel.groupby(
        ["ligand_inchikey", "ligand_smiles"], dropna=False
    ):
        text = str(smiles).strip()
        if not text:
            continue
        try:
            _fp, standard_inchikey, connectivity_key, canonical = query_features(text)
        except Exception:  # noqa: BLE001 - a query RDKit cannot read is dropped
            continue
        if not (standard_inchikey and connectivity_key and canonical):
            continue
        smiles_hash = hashlib.sha256(canonical.encode()).hexdigest()[:12]
        targets = sorted(set(group["uniprot"].astype(str)))
        rows.append({
            "query_id": f"{standard_inchikey}#{smiles_hash}",
            "ligand_key": f"{standard_inchikey}#SMILES-{smiles_hash}",
            "standard_inchikey": standard_inchikey,
            "connectivity_key": connectivity_key,
            "canonical_smiles": canonical,
            "standardization_route": "fragment_parent_canonical_smiles_key",
            "truth_targets": targets,
            "n_truth_targets": len(targets),
            "split": split,
        })
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.drop_duplicates(subset=["query_id"]).reset_index(drop=True)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--pockets", type=Path, default=DEFAULT_POCKETS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    benchmark_manifest = args.benchmark / "manifest.json"
    upstream: dict[str, dict[str, object]] = {
        "benchmark_manifest": _artifact(benchmark_manifest),
        "pocket_cluster_map": _artifact(args.pockets),
    }
    splits = {}
    for name in ("train", "dev", "test"):
        path = args.benchmark / f"{name}.parquet"
        upstream[f"benchmark_{name}"] = _artifact(path)
        splits[name] = pd.read_parquet(path, columns=NEEDED)
    pockets = pd.read_csv(args.pockets).set_index("uniprot")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Route B lends a scorable target's score to its cluster mates, so it can
    # only reach a target whose cluster already contains one. A pocket-cold
    # target's cluster has no training member by definition, and the retrieval
    # index is built from train alone - so cold targets are unreachable by
    # construction, and a cold panel cannot measure transfer quality. It
    # measures the held-out ranking, which is a different question. The
    # reachable view is the one that can answer "does a borrowed score rank
    # the right target".
    donors = set(
        pd.read_parquet(
            ROOT / "data" / "activity_retrieval_202608" / "edges.parquet",
            columns=["uniprot"],
        ).uniprot.unique()
    ) if (ROOT / "data" / "activity_retrieval_202608" / "edges.parquet").is_file() else set()

    views = {
        "strict": {
            "evaluation": splits["test"],
            "reference": pd.concat([splits["train"], splits["dev"]], ignore_index=True),
            "policy": (
                "test rows whose pair, publication and pocket cluster are absent "
                "from train+dev"
            ),
            "claimable": True,
        },
        "extended": {
            "evaluation": pd.concat([splits["dev"], splits["test"]], ignore_index=True),
            "reference": splits["train"],
            "policy": (
                "dev+test rows whose pair, publication and pocket cluster are "
                "absent from train; larger, but evaluates on dev"
            ),
            "claimable": False,
        },
    }

    manifest = {
        "schema_version": "skinscout.pocket-cold-panel.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": (
            "Measure pocket-cluster transfer (coverage route B) without scoring it "
            "on targets whose cluster it was trained on."
        ),
        "label_semantics": (
            "This benchmark holds positives only, so these are activity-recovery "
            "panels: the question is where a known target ranks, not whether a "
            "pair is active."
        ),
        "inputs": upstream,
        "views": {},
    }

    for view_name, spec in views.items():
        for level in LEVELS:
            panel = build_view(spec["evaluation"], spec["reference"], pockets, level)
            name = f"pocket_cold_{level}_{view_name}"
            out_path = args.out_dir / f"{name}.parquet"
            temp = out_path.with_suffix(".parquet.tmp")
            panel.to_parquet(temp, index=False)
            temp.replace(out_path)
            ranking = to_ranking_panel(panel, split="test")
            ranking_path = args.out_dir / f"{name}_ranking.parquet"
            temp = ranking_path.with_suffix(".parquet.tmp")
            ranking.to_parquet(temp, index=False)
            temp.replace(ranking_path)
            manifest["views"][name] = {
                "policy": spec["policy"],
                "claimable": bool(spec["claimable"] and len(panel) > 0),
                "rows": int(len(panel)),
                "targets": int(panel["uniprot"].nunique()),
                "compounds": int(panel["ligand_inchikey"].nunique()),
                "ranking_queries": int(len(ranking)),
                "sha256": _sha256(out_path),
                "ranking_sha256": _sha256(ranking_path),
            }
            print(
                f"{name:28s} {len(panel):6,}행  표적 {panel['uniprot'].nunique():4d}  "
                f"화합물 {panel['ligand_inchikey'].nunique():5d}  "
                f"랭킹 질의 {len(ranking):5,}"
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
