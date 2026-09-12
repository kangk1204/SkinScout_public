#!/usr/bin/env python3
"""stage0_skin_score_v2.py — Composite SkinScore from proteinatlas.tsv only.

Earlier `stage0_skin_score.py` joined per-axis TSVs that key on Ensembl IDs
(``Gene`` column), which did not map to the UniProt-accession receptor names
under ``data/human_pdbqt/``. This v2 uses the consolidated ``proteinatlas.tsv``
(column ``Uniprot``) so the output keys join cleanly with the receptor set.

Axes (each min-max normalised to [0, 1], then weighted-sum):
    tissue      — skin nTPM from "RNA tissue specific nTPM"           α=0.40
    cell_type   — max nCPM in skin cell types from
                  "RNA single cell type specific nCPM"                β=0.40
    specificity — bonus for tissue-enriched / group-enriched in skin  γ=0.20

Skin cell types (HPA labels): Keratinocytes (basal/suprabasal), Melanocytes,
Fibroblasts, Langerhans cells, Endothelial cells, Hair follicle cells.

Output: data/skin_expression/skin_score.tsv with columns
    uniprot · gene · skin_score · cell_type_preferred · tier
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("stage0.skin_score_v2")

WEIGHTS = {"tissue": 0.40, "cell_type": 0.40, "specificity": 0.20}


def _remove_outputs(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def _write_tsv_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, sep="\t", index=False)
    tmp.replace(path)


# HPA cell-type labels considered "skin"
SKIN_CELLS = {
    "keratinocytes", "basal keratinocytes", "suprabasal keratinocytes",
    "melanocytes", "fibroblasts", "langerhans cells",
    "endothelial cells", "hair follicle cells",
}

# Tissue label substring → matches HPA "RNA tissue specific nTPM" entries
SKIN_TISSUE_RE = re.compile(r"\bskin\b", re.IGNORECASE)


def _parse_kv(blob: str, field: str) -> dict[str, float]:
    """Parse HPA-style 'label: value;label: value' strings → dict."""
    if not isinstance(blob, str) or not blob.strip():
        return {}
    out: dict[str, float] = {}
    for chunk in blob.split(";"):
        if ":" not in chunk:
            continue
        k, v = chunk.split(":", 1)
        try:
            out[k.strip()] = float(v.strip())
        except ValueError as exc:
            raise SystemExit(
                f"{field} contains non-numeric value for {k.strip()!r}: {v.strip()!r}"
            ) from exc
    return out


def _skin_tissue_max(blob: str) -> float:
    return max((v for k, v in _parse_kv(blob, "RNA tissue specific nTPM").items()
                if SKIN_TISSUE_RE.search(k)),
               default=0.0)


def _skin_cell_best(blob: str) -> tuple[float, str]:
    best_v, best_k = 0.0, ""
    for k, v in _parse_kv(blob, "RNA single cell type specific nCPM").items():
        if k.strip().lower() in SKIN_CELLS and v > best_v:
            best_v, best_k = v, k
    return best_v, best_k


def _minmax(s: pd.Series) -> pd.Series:
    s = s.astype(float).fillna(0.0)
    lo, hi = float(s.min()), float(s.max())
    return pd.Series(0.0, index=s.index) if hi - lo < 1e-12 else (s - lo) / (hi - lo)


def _tier(x: float) -> str:
    if x >= 0.80: return "very_high"
    if x >= 0.60: return "high"
    if x >= 0.40: return "medium"
    if x >= 0.20: return "low"
    return "very_low"


def compute(proteinatlas_tsv: Path) -> pd.DataFrame:
    LOG.info("Loading %s", proteinatlas_tsv)
    df = pd.read_csv(proteinatlas_tsv, sep="\t", low_memory=False)
    required = [
        "Uniprot",
        "Gene",
        "RNA tissue specific nTPM",
        "RNA single cell type specific nCPM",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise SystemExit(
            f"proteinatlas.tsv is missing required columns {missing}: {proteinatlas_tsv}"
        )
    normalized_uniprot = df["Uniprot"].fillna("").astype(str).str.strip()
    blank_uniprot = normalized_uniprot.index[normalized_uniprot == ""].tolist()
    if blank_uniprot:
        shown = ",".join(str(value) for value in blank_uniprot[:10])
        suffix = "..." if len(blank_uniprot) > 10 else ""
        raise SystemExit(
            "proteinatlas.tsv column 'Uniprot' contains blank values at "
            f"row index {shown}{suffix}: {proteinatlas_tsv}"
        )
    primary_uniprot = normalized_uniprot.str.split(",").str[0].str.strip()
    duplicate_uniprots = primary_uniprot[primary_uniprot.duplicated()].drop_duplicates().tolist()
    if duplicate_uniprots:
        shown = ",".join(str(value) for value in duplicate_uniprots[:10])
        suffix = "..." if len(duplicate_uniprots) > 10 else ""
        raise SystemExit(
            "proteinatlas.tsv contains duplicate primary Uniprot values: "
            f"{shown}{suffix}: {proteinatlas_tsv}"
        )
    df = df.copy()
    df["Uniprot"] = normalized_uniprot
    df["primary_uniprot"] = primary_uniprot
    LOG.info("  rows with UniProt: %d", len(df))

    tissue_raw = df["RNA tissue specific nTPM"].apply(_skin_tissue_max)
    cell_results = df["RNA single cell type specific nCPM"].apply(_skin_cell_best)
    cell_raw = cell_results.apply(lambda x: x[0])
    cell_label = cell_results.apply(lambda x: x[1])

    # Specificity bonus: tissue-enriched or group-enriched in skin
    spec_col = "RNA tissue specificity"
    spec_dist = "RNA tissue distribution"
    spec_raw = pd.Series(0.0, index=df.index)
    if spec_col in df.columns:
        is_enriched = df[spec_col].fillna("").str.contains("enriched|enhanced", case=False)
        any_skin = (df[spec_dist].fillna("") if spec_dist in df.columns else "").apply(
            lambda s: 1.0 if SKIN_TISSUE_RE.search(s or "") else 0.0
        )
        spec_raw = (is_enriched.astype(float) * any_skin).clip(0, 1)

    tissue_n = _minmax(np.log1p(tissue_raw))
    cell_n = _minmax(np.log1p(cell_raw))
    spec_n = _minmax(spec_raw)

    score = (
        WEIGHTS["tissue"] * tissue_n
        + WEIGHTS["cell_type"] * cell_n
        + WEIGHTS["specificity"] * spec_n
    ).clip(0, 1)

    out = pd.DataFrame({
        "uniprot": df["primary_uniprot"],
        "gene": df["Gene"],
        "skin_score": score.round(4),
        "cell_type_preferred": cell_label.where(cell_label.astype(bool), "unknown"),
    })
    out["tier"] = out["skin_score"].apply(_tier)
    return out.sort_values("uniprot").reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--proteinatlas-tsv", default=Path("data/hpa/proteinatlas.tsv"), type=Path)
    p.add_argument("--out-tsv", required=True, type=Path)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    _remove_outputs(args.out_tsv)
    df = compute(args.proteinatlas_tsv)
    _write_tsv_atomic(df, args.out_tsv)
    LOG.info("Wrote %s (n=%d)  median=%.3f  tiers=%s",
             args.out_tsv, len(df), df["skin_score"].median(),
             df["tier"].value_counts().to_dict())


if __name__ == "__main__":
    main()
