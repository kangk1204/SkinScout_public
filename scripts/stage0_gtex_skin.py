#!/usr/bin/env python3
"""stage0_gtex_skin.py — Slice GTEx v10 to skin TPM per UniProt.

Operator places `gtex_v10_gene_tpm.gct` at <OUTDIR>/ before running. We extract
two skin columns (Skin - Sun Exposed Lower leg and Skin - Not Sun Exposed
Suprapubic), take max TPM, map Ensembl gene → UniProt via
`<OUTDIR>/ensembl_to_uniprot.tsv` (operator-supplied) when present.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

LOG = logging.getLogger("stage0.gtex")


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


def _load_ensembl_map(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"GTEx Ensembl-to-UniProt map is required: {path}")
    mp = pd.read_csv(path, sep="\t").rename(columns=str.lower)
    required = {"ensembl", "uniprot"}
    missing = sorted(required - set(mp.columns))
    if missing:
        raise SystemExit(f"GTEx Ensembl-to-UniProt map missing columns {missing}: {path}")
    mp = mp[["ensembl", "uniprot"]].dropna().copy()
    mp["ensembl"] = mp["ensembl"].astype(str).str.strip().str.split(".").str[0]
    mp["uniprot"] = mp["uniprot"].astype(str).str.strip()
    mp = mp[(mp["ensembl"] != "") & (mp["uniprot"] != "")]
    if mp.empty:
        raise SystemExit(f"GTEx Ensembl-to-UniProt map contains no usable rows: {path}")
    return mp.drop_duplicates()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--allow-empty-placeholder",
        action="store_true",
        help="write an empty skin_tpm.tsv only for explicit degraded diagnostics",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    gct = args.out_dir / "gtex_v10_gene_tpm.gct"
    map_path = args.out_dir / "ensembl_to_uniprot.tsv"
    out = args.out_dir / "skin_tpm.tsv"
    _remove_outputs(out)
    if not gct.exists():
        if not args.allow_empty_placeholder:
            raise SystemExit(
                f"GTEx GCT is required for Stage 0 skin expression: {gct}. "
                "Use --allow-empty-placeholder only for degraded diagnostics."
            )
        LOG.warning("GTEx GCT %s missing; emit empty placeholder", gct)
        _write_text_atomic("uniprot\ttpm\n", out)
        return

    df = pd.read_csv(gct, sep="\t", skiprows=2)
    skin_cols = [c for c in df.columns if "Skin" in c]
    if not skin_cols:
        if not args.allow_empty_placeholder:
            raise SystemExit(f"GTEx GCT contains no skin columns: {gct}")
        _write_text_atomic("uniprot\ttpm\n", out)
        return
    df["tpm"] = df[skin_cols].max(axis=1)
    df = df[["Name", "tpm"]].rename(columns={"Name": "ensembl"})
    df["ensembl"] = df["ensembl"].astype(str).str.strip().str.split(".").str[0]
    if not map_path.exists():
        if not args.allow_empty_placeholder:
            raise SystemExit(
                f"GTEx Ensembl-to-UniProt map is required for Stage 0 skin expression: {map_path}. "
                "Use --allow-empty-placeholder only for degraded diagnostics."
            )
        _write_text_atomic("uniprot\ttpm\n", out)
        return
    mp = _load_ensembl_map(map_path)
    df = df.merge(mp, on="ensembl", how="inner")
    df = df[["uniprot", "tpm"]].dropna()

    if df.empty and not args.allow_empty_placeholder:
        raise SystemExit(f"GTEx skin slice produced no UniProt rows: {gct}")
    _write_tsv_atomic(df, out)
    LOG.info("Wrote %s (n=%d)", out, len(df))


if __name__ == "__main__":
    main()
