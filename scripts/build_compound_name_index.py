#!/usr/bin/env python3
"""build_compound_name_index.py — a name a researcher would type -> SMILES.

The Workbench accepted SMILES and nothing else, so a wet-lab reader had to
find a structure string before they could ask anything. This builds a small,
offline index by intersecting the cosmetic ingredient list with the discovery
alias tables, then pulling every alias of the compounds that matched - so both
"Niacinamide" (the INCI name) and "nicotinamide" (what a chemist says) resolve.

Only single small molecules end up here: most CosIng entries are extracts,
mixtures and polymers, which are outside the tool's applicability anyway.

    python scripts/build_compound_name_index.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COSING = ROOT / "data" / "cosing" / "cosing.csv"
DEFAULT_ALIASES = ROOT / "data" / "discovery_aliases" / "aliases.parquet"
DEFAULT_PANEL = ROOT / "data" / "validation" / "skin_known_target_panel.csv"
DEFAULT_UV = ROOT / "data" / "validation" / "uv_filter_reference.csv"
DEFAULT_OUT = ROOT / "data" / "compound_names" / "name_index.csv"
# The reader is a Korean wet-lab researcher; the upstream alias tables carry no
# Korean, so the common ingredient names are curated here and resolved through
# the English entry they point at.
DEFAULT_KOREAN = ROOT / "data" / "compound_names" / "korean_aliases.csv"

# An alias that is a database accession helps nobody typing a name.
ACCESSION_RE = re.compile(r"^(chembl|cid|bdbm|sid|zinc|dtxsid|us\d|wo\d)\w*$", re.I)
# Unicode-aware: an ASCII-only class silently erased every Hangul name, which
# is most of what the reader of this tool will type.
NON_WORD = re.compile(r"[^\w]+", re.UNICODE)


def normalise(value: object) -> str:
    text = NON_WORD.sub(" ", str(value).strip().lower())
    return re.sub(r"\s+", " ", text).strip()


def _clean_alias(alias: str) -> str:
    """Some upstream aliases concatenate every synonym with '::'."""
    return str(alias).split("::")[0].strip()


def _is_typeable(alias: str) -> bool:
    normalised = normalise(alias)
    if not normalised or len(normalised) < 2:
        return False
    if ACCESSION_RE.match(normalised.replace(" ", "")):
        return False
    # A name that is mostly digits is a registry number, not a name. Hangul
    # counts: two syllables already carry a full ingredient name.
    letters = sum(character.isalpha() for character in normalised)
    return letters >= 2


def build(
    cosing: Path,
    aliases: Path,
    panel: Path,
    uv_filters: Path,
    out_path: Path,
    korean: Path | None = None,
) -> pd.DataFrame:
    ingredient_names = set()
    if cosing.exists():
        frame = pd.read_csv(cosing, usecols=["INCI name"])
        ingredient_names |= {
            normalise(name) for name in frame["INCI name"].dropna().astype(str)
        }
    ingredient_names.discard("")
    if not ingredient_names:
        raise SystemExit(f"성분 목록을 읽지 못했습니다: {cosing}")

    alias_frame = pd.read_parquet(
        aliases,
        columns=["alias", "parent_canonical_smiles", "parent_inchikey", "source"],
    )
    alias_frame["alias"] = alias_frame["alias"].map(_clean_alias)
    alias_frame["normalised"] = alias_frame["alias"].map(normalise)
    matched = alias_frame[alias_frame["normalised"].isin(ingredient_names)]
    keys = set(matched["parent_inchikey"].dropna())
    print(
        f"성분명 {len(ingredient_names):,}개 중 {matched['normalised'].nunique():,}개가 "
        f"별칭 테이블과 일치, 화합물 {len(keys):,}개",
        file=sys.stderr,
    )

    # Every alias of a matched compound, so a chemical name finds the same
    # molecule as the INCI name.
    expanded = alias_frame[alias_frame["parent_inchikey"].isin(keys)].copy()
    expanded = expanded[expanded["alias"].map(_is_typeable)]

    rows = expanded.rename(
        columns={
            "alias": "display_name",
            "parent_canonical_smiles": "smiles",
            "parent_inchikey": "inchikey",
        }
    )[["normalised", "display_name", "smiles", "inchikey", "source"]]
    rows["is_ingredient_name"] = rows["normalised"].isin(ingredient_names)

    curated = []
    for path, name_column in ((panel, "inci_name"), (uv_filters, "inci_name")):
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if name_column not in frame.columns or "smiles" not in frame.columns:
            continue
        for record in frame.to_dict("records"):
            for candidate in {record.get(name_column), record.get("common_name"),
                              record.get("case_id")}:
                if not candidate or not str(candidate).strip():
                    continue
                curated.append({
                    "normalised": normalise(candidate),
                    "display_name": str(candidate).replace("_", " "),
                    "smiles": record["smiles"],
                    "inchikey": record.get("inchikey_connectivity") or "",
                    "source": f"curated:{path.name}",
                    "is_ingredient_name": True,
                })
    if curated:
        rows = pd.concat([rows, pd.DataFrame(curated)], ignore_index=True)

    rows = rows[rows["normalised"] != ""]
    rows = rows.dropna(subset=["smiles"])
    # Curated entries win, then ingredient names, then the shortest alias.
    rows["priority"] = (
        rows["source"].str.startswith("curated:").astype(int) * 2
        + rows["is_ingredient_name"].astype(int)
    )
    rows["length"] = rows["display_name"].str.len()
    rows = (
        rows.sort_values(["normalised", "priority", "length"], ascending=[True, False, True])
        .drop_duplicates(subset=["normalised"], keep="first")
        .drop(columns=["priority", "length"])
        .sort_values("normalised")
        .reset_index(drop=True)
    )

    if korean is not None and korean.exists():
        by_name = {row["normalised"]: row for row in rows.to_dict("records")}
        korean_rows = []
        unresolved = []
        for record in pd.read_csv(korean).to_dict("records"):
            target = by_name.get(normalise(record["lookup_name"]))
            if target is None:
                unresolved.append(record["lookup_name"])
                continue
            korean_rows.append({
                "normalised": normalise(record["korean_name"]),
                "display_name": str(record["korean_name"]),
                "smiles": target["smiles"],
                "inchikey": target["inchikey"],
                "source": "curated:korean",
                "is_ingredient_name": True,
            })
        if unresolved:
            print(
                f"한국어 별칭 {len(unresolved)}개는 대응 영문명을 찾지 못했습니다: "
                + ", ".join(sorted(set(unresolved))),
                file=sys.stderr,
            )
        if korean_rows:
            rows = pd.concat([rows, pd.DataFrame(korean_rows)], ignore_index=True)
            rows = rows.drop_duplicates(subset=["normalised"], keep="first")
            print(f"한국어 별칭 {len(korean_rows)}개 추가", file=sys.stderr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out_path, index=False, encoding="utf-8")
    print(f"이름 색인 {len(rows):,}개 → {out_path}", file=sys.stderr)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cosing", type=Path, default=DEFAULT_COSING)
    parser.add_argument("--aliases", type=Path, default=DEFAULT_ALIASES)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--uv-filters", type=Path, default=DEFAULT_UV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--korean", type=Path, default=DEFAULT_KOREAN)
    args = parser.parse_args()
    build(args.cosing, args.aliases, args.panel, args.uv_filters, args.out, args.korean)


if __name__ == "__main__":
    main()
