#!/usr/bin/env python3
"""stage0_skin_score.py — Composite SkinScore for every human UniProt.

Formula (INSTRUCTIONS.md §3.1 (f)):
    SkinScore = α·HPA_tissue + β·HPA_cell|HPA_tissue>0 + γ·SkinProteome
              + ε·GTEx_skin + ζ·max_cell_type_score
    α=0.30, β=0.25, γ=0.20, ε=0.10, ζ=0.15

Each axis is min-max normalised to [0, 1] across the protein-coding human
proteome BEFORE the weighted sum. Missing axis = 0 contribution, but the
weights are renormalised so that the result still sums to ≤ 1.

The HPA single-cell table is pan-tissue and cannot independently establish
skin relevance. Its selected cell-type axis contributes only for proteins with
positive expression in the HPA skin-tissue table.

Tier mapping:
    >=0.80 very_high
    >=0.60 high
    >=0.40 medium
    >=0.20 low
    < 0.20 very_low
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("stage0.skin_score")

WEIGHTS = {
    "hpa_tissue": 0.30,
    "hpa_cell":   0.25,
    "proteome":   0.20,
    "gtex":       0.10,
    "sc":         0.15,
}
HPA_EXPRESSION_COLUMNS = ("ntpm", "n_tpm", "ncpm", "n_cpm", "value")
SKIN_CELL_TYPE_TOKENS = (
    "keratinocyte",
    "fibroblast",
    "melanocyte",
    "langerhans",
    "endothelial",
)


def _remove_outputs(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def _write_tsv_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, sep="\t", index=False)
    tmp.replace(path)


def _require_any_column(
    df: pd.DataFrame,
    path: Path,
    role: str,
    candidates: tuple[str, ...],
) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise SystemExit(
        f"{path} is missing required {role} column; expected one of {candidates}"
    )


@dataclass(frozen=True)
class SkinSources:
    hpa_dir: Path
    proteome_dir: Path
    gtex_dir: Path
    sc_dir: Path


def _minmax(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    s = s.where(s.notna(), 0.0)
    lo = float(s.min())
    hi = float(s.max())
    if hi - lo < 1e-12:
        return pd.Series(0.0, index=s.index)
    return (s - lo) / (hi - lo)


def _validate_source_uniprots(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if df.empty:
        return df
    normalized = df["uniprot"].fillna("").astype(str).str.strip()
    blank_rows = normalized.index[normalized == ""].tolist()
    if blank_rows:
        shown = ",".join(str(value) for value in blank_rows[:10])
        suffix = "..." if len(blank_rows) > 10 else ""
        raise SystemExit(
            f"{label} source column 'uniprot' contains blank values at "
            f"row index {shown}{suffix}"
        )
    duplicates = normalized[normalized.duplicated()].drop_duplicates().tolist()
    if duplicates:
        shown = ",".join(str(value) for value in duplicates[:10])
        suffix = "..." if len(duplicates) > 10 else ""
        raise SystemExit(
            f"{label} source contains duplicate uniprot values: {shown}{suffix}"
        )
    out = df.copy()
    out["uniprot"] = normalized
    return out


def _required_nonnegative_numeric(
    df: pd.DataFrame,
    column: str,
    label: str,
) -> pd.Series:
    raw = df[column]
    normalized = raw.fillna("").astype(str).str.strip()
    blank_rows = normalized.index[normalized == ""].tolist()
    if blank_rows:
        shown = ",".join(str(value) for value in blank_rows[:10])
        suffix = "..." if len(blank_rows) > 10 else ""
        raise SystemExit(
            f"{label} source column '{column}' contains blank values at "
            f"row index {shown}{suffix}"
        )
    numeric = pd.to_numeric(raw, errors="coerce")
    bad_rows = numeric.index[numeric.isna()].tolist()
    if bad_rows:
        shown = ",".join(str(value) for value in bad_rows[:10])
        suffix = "..." if len(bad_rows) > 10 else ""
        raise SystemExit(
            f"{label} source column '{column}' contains non-numeric values at "
            f"row index {shown}{suffix}"
        )
    values = numeric.astype(float)
    nonfinite_rows = values.index[~np.isfinite(values)].tolist()
    if nonfinite_rows:
        shown = ",".join(str(value) for value in nonfinite_rows[:10])
        suffix = "..." if len(nonfinite_rows) > 10 else ""
        raise SystemExit(
            f"{label} source column '{column}' contains non-finite values at "
            f"row index {shown}{suffix}"
        )
    negative_rows = values.index[values < 0].tolist()
    if negative_rows:
        shown = ",".join(str(value) for value in negative_rows[:10])
        suffix = "..." if len(negative_rows) > 10 else ""
        raise SystemExit(
            f"{label} source column '{column}' contains negative values at "
            f"row index {shown}{suffix}"
        )
    return values


def _is_skin_cell_type(value: object) -> bool:
    normalized = str(value).strip().lower()
    return any(token in normalized for token in SKIN_CELL_TYPE_TOKENS)


def _mapping_paths(source_dir: Path, identifier_col: str) -> tuple[Path, ...]:
    if identifier_col == "ensembl":
        return (
            source_dir / "ensembl_to_uniprot.tsv",
            source_dir / "hpa_ensembl_to_uniprot.tsv",
        )
    return (
        source_dir / "gene_to_uniprot.tsv",
        source_dir / "hpa_gene_to_uniprot.tsv",
        source_dir / "proteinatlas_gene_to_uniprot.tsv",
    )


def _primary_uniprot(value: object) -> str:
    text = str(value).strip()
    if not text:
        return ""
    return text.replace(";", ",").split(",")[0].strip()


def _proteinatlas_mapping(
    source_dir: Path,
    identifier_col: str,
) -> tuple[pd.DataFrame | None, Path]:
    atlas_path = source_dir / "proteinatlas.tsv"
    if not atlas_path.exists():
        return None, atlas_path
    atlas = pd.read_csv(atlas_path, sep="\t", low_memory=False).rename(columns=str.lower)
    if "uniprot" not in atlas.columns:
        raise SystemExit(f"{atlas_path} must include column 'Uniprot' for HPA UniProt mapping")
    uid = atlas["uniprot"].map(_primary_uniprot)
    pieces: list[pd.DataFrame] = []
    key_columns = ("ensembl",) if identifier_col == "ensembl" else ("gene", "ensembl")
    for key_column in key_columns:
        if key_column not in atlas.columns:
            continue
        piece = pd.DataFrame({
            identifier_col: atlas[key_column].astype(str).str.strip(),
            "uniprot": uid,
        })
        pieces.append(piece)
    if not pieces:
        expected = "Ensembl" if identifier_col == "ensembl" else "Gene or Ensembl"
        raise SystemExit(f"{atlas_path} must include column '{expected}' for HPA UniProt mapping")
    mapping = pd.concat(pieces, ignore_index=True)
    mapping = mapping[
        (mapping[identifier_col] != "")
        & (mapping[identifier_col].str.lower() != "nan")
        & (mapping["uniprot"] != "")
    ].drop_duplicates()
    return mapping, atlas_path


def _map_to_uniprot(
    df: pd.DataFrame,
    source_dir: Path,
    identifier_col: str,
    source_path: Path,
    label: str,
) -> pd.DataFrame:
    mapping_path = next((path for path in _mapping_paths(source_dir, identifier_col) if path.exists()), None)
    if mapping_path is not None:
        mapping = pd.read_csv(mapping_path, sep="\t").rename(columns=str.lower)
    else:
        mapping, mapping_path = _proteinatlas_mapping(source_dir, identifier_col)
    if mapping is None:
        expected = ", ".join(str(path) for path in _mapping_paths(source_dir, identifier_col))
        raise SystemExit(
            f"{source_path} uses column '{identifier_col}' but no UniProt mapping was found; "
            f"expected one of: {expected}, or {source_dir / 'proteinatlas.tsv'}"
        )
    uid_col = "uniprot" if "uniprot" in mapping.columns else None
    key_col = identifier_col if identifier_col in mapping.columns else None
    if uid_col is None or key_col is None:
        raise SystemExit(
            f"{mapping_path} must include columns '{identifier_col}' and 'uniprot' for {label}"
        )
    mp = mapping[[key_col, uid_col]].dropna().copy()
    mp[key_col] = mp[key_col].astype(str).str.strip()
    if identifier_col == "ensembl":
        mp[key_col] = mp[key_col].str.split(".").str[0]
    mp[uid_col] = mp[uid_col].astype(str).str.strip()
    mp = mp[(mp[key_col] != "") & (mp[uid_col] != "")].drop_duplicates()
    if mp.empty:
        raise SystemExit(f"{mapping_path} contains no usable {identifier_col}->UniProt rows")
    out = df.copy()
    out[identifier_col] = out[identifier_col].astype(str).str.strip()
    if identifier_col == "ensembl":
        out[identifier_col] = out[identifier_col].str.split(".").str[0]
    out = out.merge(mp, on=identifier_col, how="inner").drop(columns=[identifier_col])
    if out.empty:
        raise SystemExit(f"{label} source produced no UniProt rows after applying {mapping_path}")
    return out


def _ensure_uniprot_column(
    df: pd.DataFrame,
    source_dir: Path,
    source_path: Path,
    *,
    allowed_identifier_columns: tuple[str, ...],
    label: str,
) -> pd.DataFrame:
    if "uniprot" in df.columns:
        return df
    identifier_col = next((column for column in allowed_identifier_columns if column in df.columns), None)
    if identifier_col is None:
        expected = ("uniprot", *allowed_identifier_columns)
        raise SystemExit(
            f"{source_path} is missing required protein identifier column; "
            f"expected one of {expected}"
        )
    return _map_to_uniprot(df, source_dir, identifier_col, source_path, label)


def load_hpa_tissue(hpa_dir: Path) -> pd.DataFrame:
    fpath = hpa_dir / "rna_tissue_consensus.tsv"
    if not fpath.exists():
        return pd.DataFrame(columns=["uniprot", "skin_nTPM_log"])
    df = pd.read_csv(fpath, sep="\t")
    df = df.rename(columns=str.lower)
    if "tissue" not in df.columns:
        raise SystemExit(f"{fpath} is missing required tissue column")
    skin_mask = df["tissue"].str.contains("skin", case=False, na=False)
    skin_df = df[skin_mask].copy()
    col = _require_any_column(skin_df, fpath, "expression", HPA_EXPRESSION_COLUMNS)
    skin_df = _ensure_uniprot_column(
        skin_df,
        hpa_dir,
        fpath,
        allowed_identifier_columns=("ensembl", "gene"),
        label="HPA tissue",
    )
    skin_df[col] = _required_nonnegative_numeric(skin_df, col, "HPA tissue")
    agg = (
        skin_df.groupby("uniprot", as_index=False)[col]
        .max()
        .rename(columns={col: "skin_nTPM"})
    )
    agg["skin_nTPM_log"] = np.log1p(agg["skin_nTPM"])
    return agg[["uniprot", "skin_nTPM_log"]]


def load_hpa_cell(hpa_dir: Path) -> pd.DataFrame:
    fpath = hpa_dir / "rna_single_cell_type.tsv"
    if not fpath.exists():
        return pd.DataFrame(
            columns=["uniprot", "hpa_cell_log", "hpa_cell_type_preferred"]
        )
    df = pd.read_csv(fpath, sep="\t").rename(columns=str.lower)
    if "cell type" in df.columns:
        df = df.rename(columns={"cell type": "cell_type"})
    if "cell_type" not in df.columns:
        raise SystemExit(f"{fpath} is missing required cell_type column")
    df = df[df["cell_type"].map(_is_skin_cell_type)]
    df = _ensure_uniprot_column(
        df,
        hpa_dir,
        fpath,
        allowed_identifier_columns=("gene",),
        label="HPA cell",
    )
    value_col = _require_any_column(df, fpath, "expression", HPA_EXPRESSION_COLUMNS)
    if df.empty:
        return pd.DataFrame(
            columns=["uniprot", "hpa_cell_log", "hpa_cell_type_preferred"]
        )
    df[value_col] = _required_nonnegative_numeric(df, value_col, "HPA cell")
    df["cell_type"] = df["cell_type"].astype(str).str.strip().str.lower()
    preferred = (
        df.sort_values(
            ["uniprot", value_col, "cell_type"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .drop_duplicates("uniprot", keep="first")
        .copy()
    )
    preferred["hpa_cell_log"] = np.log1p(preferred[value_col])
    preferred["hpa_cell_type_preferred"] = preferred["cell_type"].where(
        preferred[value_col] > 0.0,
        "unknown",
    )
    return preferred[
        ["uniprot", "hpa_cell_log", "hpa_cell_type_preferred"]
    ]


def load_proteome(proteome_dir: Path) -> pd.DataFrame:
    fpath = proteome_dir / "skin_proteome.tsv"
    if not fpath.exists():
        return pd.DataFrame(columns=["uniprot", "proteome_log"])
    df = pd.read_csv(fpath, sep="\t")
    cols = {c.lower(): c for c in df.columns}
    uid_col = cols.get("uniprot") or cols.get("uniprot_id") or cols.get("accession")
    val_col = cols.get("lfq") or cols.get("intensity") or cols.get("abundance")
    if uid_col is None or val_col is None:
        raise SystemExit(
            f"{fpath} is missing required proteome columns; expected an identifier "
            "('uniprot', 'uniprot_id', 'accession') and value "
            "('lfq', 'intensity', 'abundance')"
        )
    out = df[[uid_col, val_col]].rename(
        columns={uid_col: "uniprot", val_col: "value"}
    )
    out["proteome_log"] = np.log1p(
        _required_nonnegative_numeric(out, "value", "Skin proteome")
    )
    return out[["uniprot", "proteome_log"]]


def load_gtex(gtex_dir: Path) -> pd.DataFrame:
    fpath = gtex_dir / "skin_tpm.tsv"
    if not fpath.exists():
        return pd.DataFrame(columns=["uniprot", "gtex_log"])
    df = pd.read_csv(fpath, sep="\t")
    if "uniprot" not in df.columns or "tpm" not in df.columns:
        raise SystemExit(f"{fpath} is missing required columns: uniprot, tpm")
    df["gtex_log"] = np.log1p(_required_nonnegative_numeric(df, "tpm", "GTEx skin"))
    return df[["uniprot", "gtex_log"]]


def load_sc(sc_dir: Path) -> pd.DataFrame:
    fpath = sc_dir / "skin_max_celltype.tsv"
    if not fpath.exists():
        return pd.DataFrame(columns=["uniprot", "sc_log"])
    df = pd.read_csv(fpath, sep="\t")
    if "uniprot" not in df.columns:
        raise SystemExit(f"{fpath} is missing required uniprot column")
    val = _require_any_column(
        df,
        fpath,
        "single-cell expression",
        ("max_expression", "score", "ntpm"),
    )
    df["sc_log"] = np.log1p(_required_nonnegative_numeric(df, val, "Single-cell skin"))
    return df[["uniprot", "sc_log"]]


def compute_skin_score(sources: SkinSources) -> pd.DataFrame:
    hpa_tissue = _validate_source_uniprots(
        load_hpa_tissue(sources.hpa_dir), "HPA tissue"
    ).rename(columns={"skin_nTPM_log": "raw"})
    hpa_cell = _validate_source_uniprots(
        load_hpa_cell(sources.hpa_dir), "HPA cell"
    ).rename(columns={"hpa_cell_log": "raw"})
    if not hpa_cell.empty:
        tissue_supported = set(
            hpa_tissue.loc[hpa_tissue["raw"] > 0.0, "uniprot"].astype(str)
        )
        unsupported = ~hpa_cell["uniprot"].astype(str).isin(tissue_supported)
        hpa_cell.loc[unsupported, "raw"] = 0.0
        hpa_cell.loc[unsupported, "hpa_cell_type_preferred"] = "unknown"

    pieces = {
        "hpa_tissue": hpa_tissue,
        "hpa_cell": hpa_cell,
        "proteome":   _validate_source_uniprots(
            load_proteome(sources.proteome_dir), "Skin proteome"
        ).rename(
            columns={"proteome_log": "raw"}
        ),
        "gtex":       _validate_source_uniprots(
            load_gtex(sources.gtex_dir), "GTEx skin"
        ).rename(
            columns={"gtex_log": "raw"}
        ),
        "sc":         _validate_source_uniprots(
            load_sc(sources.sc_dir), "Single-cell skin"
        ).rename(
            columns={"sc_log": "raw"}
        ),
    }
    uniprots: set[str] = set()
    for df in pieces.values():
        uniprots.update(df["uniprot"].dropna().astype(str).tolist())
    merged = pd.DataFrame({"uniprot": sorted(uniprots)})
    for axis, df in pieces.items():
        if df.empty:
            merged[axis] = 0.0
            continue
        col = (
            df.set_index("uniprot")["raw"].astype(float)
            .reindex(merged["uniprot"]).fillna(0.0)
        )
        merged[axis] = _minmax(col).values

    available_axes = [
        axis
        for axis, df in pieces.items()
        if not df.empty
        and (axis != "hpa_cell" or bool((df["raw"] > 0.0).any()))
    ]
    # Which axes actually carried data, and what each ended up weighing after
    # renormalisation. Without this the published score looks like the
    # documented five-axis formula while several axes may have contributed
    # nothing, and a reader applying that formula gets a different number.
    if not available_axes:
        merged["skin_score"] = 0.0
        contribution = {}
    else:
        weight_sum = sum(WEIGHTS[a] for a in available_axes)
        merged["skin_score"] = sum(
            merged[a] * (WEIGHTS[a] / weight_sum) for a in available_axes
        )
        contribution = {
            axis: round(WEIGHTS[axis] / weight_sum, 6) for axis in available_axes
        }
    merged.attrs["axis_contribution"] = contribution
    merged.attrs["axes_absent"] = sorted(set(WEIGHTS) - set(available_axes))

    def tier(x: float) -> str:
        if x >= 0.80:
            return "very_high"
        if x >= 0.60:
            return "high"
        if x >= 0.40:
            return "medium"
        if x >= 0.20:
            return "low"
        return "very_low"

    merged["tier"] = merged["skin_score"].apply(tier)
    hpa_cell_labels = pieces["hpa_cell"]
    if hpa_cell_labels.empty:
        merged["cell_type_preferred"] = "unknown"
    else:
        merged["cell_type_preferred"] = (
            hpa_cell_labels.set_index("uniprot")["hpa_cell_type_preferred"]
            .reindex(merged["uniprot"])
            .fillna("unknown")
            .to_numpy()
        )
    result = merged[["uniprot", "skin_score", "tier", "cell_type_preferred"]]
    result.attrs["axis_contribution"] = merged.attrs["axis_contribution"]
    result.attrs["axes_absent"] = merged.attrs["axes_absent"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hpa-dir", required=True, type=Path)
    parser.add_argument("--proteome-dir", required=True, type=Path)
    parser.add_argument("--gtex-dir", required=True, type=Path)
    parser.add_argument("--sc-dir", required=True, type=Path)
    parser.add_argument("--out-tsv", required=True, type=Path)
    parser.add_argument(
        "--allow-empty-output",
        action="store_true",
        help="write an empty/zero-row skin score only for explicit degraded diagnostics",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    _remove_outputs(args.out_tsv)
    sources = SkinSources(args.hpa_dir, args.proteome_dir, args.gtex_dir, args.sc_dir)
    df = compute_skin_score(sources)
    if df.empty and not args.allow_empty_output:
        raise SystemExit(
            "SkinScore has no evaluable rows; provide HPA, proteome, GTEx, or scRNA-seq sources "
            "or pass --allow-empty-output only for degraded diagnostics."
        )

    # A score with no accession cannot be looked up by anything downstream, and
    # a literal "nan" in this column made stage3_skin_weighting refuse the whole
    # table - which takes the v3 ranking rule, and therefore a fast run, down
    # with it. Refuse to emit one rather than write a row nobody can use.
    blank = df.index[
        df["uniprot"].isna() | (df["uniprot"].astype(str).str.strip() == "")
    ].tolist()
    if blank:
        raise SystemExit(
            "SkinScore output has rows with no UniProt accession at index(es) "
            + ", ".join(str(index) for index in blank[:10])
            + (" ..." if len(blank) > 10 else "")
        )
    if df["uniprot"].duplicated().any():
        duplicated = sorted(set(df.loc[df["uniprot"].duplicated(), "uniprot"]))
        raise SystemExit(
            "SkinScore output has duplicate accessions: "
            + ", ".join(str(value) for value in duplicated[:10])
        )

    _write_tsv_atomic(df, args.out_tsv)
    provenance = args.out_tsv.with_suffix(args.out_tsv.suffix + ".axes.json")
    provenance.write_text(
        json.dumps(
            {
                "schema_version": "skinscout.skin-score-axes.v1",
                "declared_weights": WEIGHTS,
                "effective_weights": df.attrs.get("axis_contribution", {}),
                "axes_absent": df.attrs.get("axes_absent", []),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    LOG.info("Wrote %s (n=%d)", args.out_tsv, len(df))


if __name__ == "__main__":
    main()
