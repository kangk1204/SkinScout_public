#!/usr/bin/env python3
"""Measure what a run can reach, and what more data would actually add.

`docs/DATA_EXPANSION_20260831.md` was wrong twice about this, both times because
it measured the wrong table:

1. It proposed rebuilding the retrieval index on every benchmark split for +764
   targets. Only 133 of those come from the split; the rest were dropped by the
   benchmark's quality filters.
2. Worse, the retrieval index is an evaluation artefact. Stage 3 retrieves
   against the ChEMBL mirror (`data/chembl37/activity_evidence.parquet`), so
   rebuilding the index changes nothing a researcher sees.

So this reports the run path first, and the benchmark decomposition second, for
whoever is looking at the evaluation numbers.

Run:
    python scripts/analyse_index_coverage_gap.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "data" / "activity_benchmark_202608"
EVIDENCE = ROOT / "data" / "evidence_splits"
SKIN_SCORE = ROOT / "data" / "skin_expression" / "skin_score.tsv"
# The table Stage 3 retrieves against. This is what a researcher's run can reach.
RUNTIME_EVIDENCE = ROOT / "data" / "chembl37" / "activity_evidence.parquet"

BENCHMARK_SPLITS = ("train", "dev", "test")
EVIDENCE_POOLS = ("pre_cutoff", "post_cutoff", "undated")

# The bands the researcher-facing documentation reports.
BANDS: tuple[tuple[str, float, float], ...] = (
    (">=0.6  (high/very_high)", 0.6, 1.01),
    ("0.4-0.6 (medium)", 0.4, 0.6),
    ("0.2-0.4 (low)", 0.2, 0.4),
    ("<0.2   (very_low)", -0.01, 0.2),
)


def _targets(path: Path) -> set[str]:
    table = pq.read_table(path, columns=["uniprot"])
    return {value for value in table.column("uniprot").to_pylist() if value}


def _skin_scores() -> pd.Series:
    frame = pd.read_csv(SKIN_SCORE, sep="\t", dtype={"uniprot": str})
    frame = frame.dropna(subset=["uniprot"])
    return frame.set_index("uniprot")["skin_score"].astype(float)


def _band_table(scores: pd.Series, covered: set[str], label: str) -> pd.DataFrame:
    rows = []
    for name, low, high in BANDS:
        in_band = scores[(scores > low) & (scores <= high)]
        hit = sum(1 for uniprot in in_band.index if uniprot in covered)
        rows.append(
            {
                "band": name,
                "targets": len(in_band),
                label: hit,
                f"{label} %": round(100.0 * hit / len(in_band), 1) if len(in_band) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="write the measurement as JSON")
    args = parser.parse_args()

    splits = {name: _targets(BENCHMARK / f"{name}.parquet") for name in BENCHMARK_SPLITS}
    benchmark_all = set().union(*splits.values())

    pool: set[str] = set()
    pool_by_file = {}
    for name in EVIDENCE_POOLS:
        found = _targets(EVIDENCE / f"{name}.parquet")
        pool_by_file[name] = len(found)
        pool |= found

    from_splits = benchmark_all - splits["train"]
    from_filters = pool - benchmark_all

    runtime = _targets(RUNTIME_EVIDENCE) if RUNTIME_EVIDENCE.exists() else set()

    print("=== what a run can reach (the number that matters) ===")
    print(f"  run path: ChEMBL mirror            {len(runtime):>6,}")
    print(f"  + BindingDB and GtoPdb evidence    {len(pool - runtime):>+6,}")
    print(f"  = every source already downloaded  {len(pool | runtime):>6,}")

    print()
    print("=== where the targets are ===")
    for name in BENCHMARK_SPLITS:
        print(f"  benchmark {name:6s} {len(splits[name]):>6,}")
    for name, count in pool_by_file.items():
        print(f"  evidence  {name:11s} {count:>6,}")

    print()
    print("=== the benchmark index gap (evaluation only) ===")
    print(f"  index today (train only)          {len(splits['train']):>6,}")
    print(f"  + only in dev/test                {len(from_splits):>+6,}   (a split decision)")
    print(f"  = every benchmark split           {len(benchmark_all):>6,}")
    print(f"  + removed by benchmark filters    {len(from_filters):>+6,}   (a quality decision)")
    print(f"  = downloaded evidence pool        {len(pool):>6,}")

    scores = _skin_scores()
    print()
    print("=== skin-expression bands, measured on the run path ===")
    today = _band_table(scores, runtime, "run today")
    proposed = _band_table(scores, pool | runtime, "all sources")
    merged = today.merge(proposed.drop(columns=["targets"]), on="band")
    merged["gain"] = merged["all sources"] - merged["run today"]
    print(merged.to_string(index=False))

    named = {
        "Q92482": "AQP3",
        "P26447": "S100A4",
        "P01100": "FOS",
        "P84243": "H3-3B",
    }
    named["P17643"] = "TYRP1"
    named["P40126"] = "DCT"
    named["P14679"] = "TYR"
    print()
    print("=== named skin targets, against the run path ===")
    placement = {}
    for uniprot, gene in named.items():
        if uniprot in runtime:
            where = "a run already reaches it"
        elif uniprot in pool:
            where = "in BindingDB/GtoPdb - merging the source would reach it"
        else:
            where = "no public evidence anywhere - needs curation"
        placement[uniprot] = where
        score = scores.get(uniprot)
        shown = f"{score:.3f}" if score is not None else "n/a"
        print(f"  {uniprot} {gene:8s} skin={shown}  {where}")

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "runtime_reachable": len(runtime),
                    "runtime_gain_from_other_sources": len(pool - runtime),
                    "index_today": len(splits["train"]),
                    "gained_from_dev_test": len(from_splits),
                    "benchmark_all_splits": len(benchmark_all),
                    "removed_by_benchmark_filters": len(from_filters),
                    "evidence_pool": len(pool),
                    "bands": merged.to_dict(orient="records"),
                    "named_targets": placement,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
