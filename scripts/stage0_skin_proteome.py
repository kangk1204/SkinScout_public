#!/usr/bin/env python3
"""stage0_skin_proteome.py — Mirror Dyring-Andersen 2020 skin proteome atlas.

The supplementary table from the Nat. Commun. paper lives at:
  https://www.nature.com/articles/s41467-020-19383-8
The operator places the LFQ supplementary TSV at <OUTDIR>/raw_lfq.tsv before
running this script. Output: <OUTDIR>/skin_proteome.tsv keyed by UniProt.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

LOG = logging.getLogger("stage0.skin_proteome")


def _remove_outputs(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def _write_text_atomic(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _write_tsv_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, sep="\t", index=False)
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--allow-empty-placeholder",
        action="store_true",
        help="write an empty skin_proteome.tsv only for explicit degraded diagnostics",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw = args.out_dir / "raw_lfq.tsv"
    out = args.out_dir / "skin_proteome.tsv"
    _remove_outputs(out)
    if not raw.exists():
        if not args.allow_empty_placeholder:
            raise SystemExit(
                f"Dyring-Andersen 2020 supplementary LFQ TSV is required: {raw}. "
                "Use --allow-empty-placeholder only for degraded diagnostics."
            )
        LOG.warning("Missing supplementary LFQ TSV at %s; emitting empty placeholder.", raw)
        _write_text_atomic("uniprot\tlfq\n", out)
        return

    df = pd.read_csv(raw, sep="\t")
    cols = {c.lower(): c for c in df.columns}
    uid = cols.get("uniprot") or cols.get("accession")
    val_cols = [
        c for low, c in cols.items()
        if low.startswith(("lfq", "intensity")) and c != uid
    ]
    if uid is None or not val_cols:
        if not args.allow_empty_placeholder:
            raise SystemExit(
                f"Skin proteome TSV must include UniProt/accession and LFQ/intensity columns: {raw}"
            )
        LOG.warning("LFQ columns not recognised; emit empty file")
        _write_text_atomic("uniprot\tlfq\n", out)
        return
    df["lfq"] = df[val_cols].max(axis=1)
    df = df[[uid, "lfq"]].rename(columns={uid: "uniprot"}).dropna()
    if df.empty and not args.allow_empty_placeholder:
        raise SystemExit(f"Skin proteome ingest produced no UniProt rows: {raw}")
    _write_tsv_atomic(df, out)
    LOG.info("Wrote %s (n=%d)", out, len(df))


if __name__ == "__main__":
    main()
