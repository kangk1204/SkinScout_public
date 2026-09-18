#!/usr/bin/env python3
"""List the skin-expressed targets a researcher's run cannot say anything about.

Priority 2 of `docs/DATA_EXPANSION_20260831.md`. The proteome-wide coverage gap
is ~15,000 targets and no amount of curation touches that. But the gap that
matters for a skin tool is small, and it has to be measured against the table the
run actually reads - the ChEMBL mirror Stage 3 retrieves against, not the
benchmark index, which only evaluation ever opens.

For each missing target this writes what a curator needs to start: the accession,
the gene symbol and description, the skin score and tier, the cell type, and
whether activity evidence already exists in BindingDB or GtoPdb - that is a
source-merge job, not a literature job.

It also applies a second axis. Skin expression measures abundance, not whether a
small molecule binding the protein means anything; without that filter the list
fills with keratins and ribosomal proteins.

Run:
    python scripts/build_skin_curation_worklist.py \
        --out data/curation/skin_target_worklist.csv
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
PROTEINATLAS = ROOT / "data" / "hpa" / "proteinatlas.tsv"
# What a researcher's run actually reads. Stage 3 retrieves against the ChEMBL
# mirror's fingerprints, not the benchmark index - measuring coverage against the
# benchmark answers a question nobody asked.
RUNTIME_EVIDENCE = ROOT / "data" / "chembl37" / "activity_evidence.parquet"

EVIDENCE_POOLS = ("pre_cutoff", "post_cutoff", "undated")
DEFAULT_MIN_SCORE = 0.6

# The skin score measures abundance in skin, not whether a small molecule binding
# the protein would mean anything. Without a second axis the worklist fills with
# keratins, ribosomal proteins and mitochondrially encoded complex I subunits -
# highly expressed in skin, and not things a cosmetic ingredient is screened
# against. HPA's protein classes carry the distinction, so use them.
DRUGGABLE_CLASSES = (
    "FDA approved drug targets",
    "Potential drug targets",
    "Enzymes",
    "Transporters",
    "G-protein coupled receptors",
    "Voltage-gated ion channels",
    "Ligand-gated ion channels",
    "Nuclear receptors",
)
# Families that score high on skin abundance and are not small-molecule targets.
STRUCTURAL_PREFIXES = ("KRT", "MT-", "RPL", "RPS", "COL", "HIST", "H1-", "H2A", "H2B", "H3-", "H4-")


def _targets(path: Path) -> set[str]:
    table = pq.read_table(path, columns=["uniprot"])
    return {value for value in table.column("uniprot").to_pylist() if value}


def _gene_table() -> pd.DataFrame:
    """HPA maps accessions to gene symbols; one row can carry several accessions."""
    frame = pd.read_csv(
        PROTEINATLAS,
        sep="\t",
        usecols=["Gene", "Gene description", "Uniprot", "Protein class"],
        dtype=str,
    ).fillna("")
    rows = []
    for record in frame.itertuples(index=False):
        for accession in str(record.Uniprot).split(","):
            accession = accession.strip()
            if accession:
                rows.append(
                    {
                        "uniprot": accession,
                        "gene": record.Gene,
                        "description": getattr(record, "_1", ""),
                        "protein_class": getattr(record, "_3", ""),
                    }
                )
    table = pd.DataFrame(rows)
    return table.drop_duplicates(subset=["uniprot"], keep="first").set_index("uniprot")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "curation" / "skin_target_worklist.csv",
    )
    parser.add_argument(
        "--min-skin-score",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help=f"lowest skin score to include (default {DEFAULT_MIN_SCORE})",
    )
    parser.add_argument(
        "--measure-against",
        choices=("runtime", "benchmark"),
        default="runtime",
        help=(
            "runtime (default) is the ChEMBL mirror a researcher's run retrieves "
            "against; benchmark is the train/dev/test index, which only evaluation "
            "reads"
        ),
    )
    parser.add_argument("--json", type=Path, help="also write a summary as JSON")
    args = parser.parse_args()

    if args.measure_against == "runtime":
        reachable = _targets(RUNTIME_EVIDENCE)
    else:
        reachable = set().union(
            *(_targets(BENCHMARK / f"{name}.parquet") for name in ("train", "dev", "test"))
        )

    pool: set[str] = set()
    for name in EVIDENCE_POOLS:
        pool |= _targets(EVIDENCE / f"{name}.parquet")

    scores = pd.read_csv(SKIN_SCORE, sep="\t", dtype={"uniprot": str}).dropna(
        subset=["uniprot"]
    )
    scores["skin_score"] = scores["skin_score"].astype(float)
    band = scores[scores["skin_score"] >= args.min_skin_score]

    missing = band[~band["uniprot"].isin(reachable)].copy()
    genes = _gene_table()
    missing["gene"] = missing["uniprot"].map(genes["gene"]).fillna("")
    missing["description"] = missing["uniprot"].map(genes["description"]).fillna("")
    missing["protein_class"] = missing["uniprot"].map(genes["protein_class"]).fillna("")
    missing["evidence_exists_elsewhere"] = missing["uniprot"].isin(pool)
    classes = missing["protein_class"].fillna("")
    missing["druggable_class"] = [
        any(name in row for name in DRUGGABLE_CLASSES) for row in classes
    ]
    missing["structural_family"] = [
        str(gene).startswith(STRUCTURAL_PREFIXES) for gene in missing["gene"]
    ]
    # Worth a curator's time only if something could plausibly bind it.
    missing["worth_curating"] = missing["druggable_class"] & ~missing["structural_family"]
    work = []
    for worth, structural, filtered in zip(
        missing["worth_curating"],
        missing["structural_family"],
        missing["evidence_exists_elsewhere"],
    ):
        if structural:
            # Keratin, ribosomal, mitochondrial, collagen, histone. Abundant in
            # skin, not something an ingredient is screened against.
            work.append("structural protein - skip")
        elif not worth:
            # Transcription factors and the like: hard, not impossible. Lower
            # priority rather than excluded outright.
            work.append("not in a druggable class - low priority")
        elif filtered:
            work.append("already in BindingDB/GtoPdb - merge the source")
        else:
            work.append("literature or in-house data")
    missing["work_needed"] = work
    missing = missing.sort_values(
        ["worth_curating", "evidence_exists_elsewhere", "skin_score"],
        ascending=[False, False, False],
    )

    columns = [
        "uniprot",
        "gene",
        "skin_score",
        "tier",
        "cell_type_preferred",
        "worth_curating",
        "evidence_exists_elsewhere",
        "work_needed",
        "druggable_class",
        "structural_family",
        "protein_class",
        "description",
    ]
    out = missing[[c for c in columns if c in missing.columns]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    filtered_count = int(missing["evidence_exists_elsewhere"].sum())
    worth = missing[missing["worth_curating"]]
    print(f"measuring against: {args.measure_against}")
    print(f"skin score >= {args.min_skin_score}: {len(band):,} targets")
    print(f"  reachable today   : {len(band) - len(missing):,}")
    print(f"  missing           : {len(missing):,}")
    print(f"    in BindingDB/GtoPdb but not the run path : {filtered_count:,}")
    print(f"    no public evidence anywhere              : {len(missing) - filtered_count:,}")
    print()
    print(f"  actually worth curating : {len(worth):,}")
    print(f"    the rest are structural or non-druggable: {len(missing) - len(worth):,}")
    print(f"      keratins/ribosomal/mito/collagen/histone: "
          f"{int(missing['structural_family'].sum()):,}")
    print(f"\nwrote {args.out}")
    print()
    print(
        out.head(20).drop(columns=["protein_class", "description"]).to_string(index=False)
    )

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "min_skin_score": args.min_skin_score,
                    "band_targets": len(band),
                    "indexed": len(band) - len(missing),
                    "missing": len(missing),
                    "missing_but_in_other_sources": filtered_count,
                    "missing_everywhere": len(missing) - filtered_count,
                    "measured_against": args.measure_against,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
