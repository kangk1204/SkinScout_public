#!/usr/bin/env python3
"""stage2_pains_brenk.py — PAINS / Brenk / NIH structural-alert filters."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import FilterCatalog

LOG = logging.getLogger("stage2.alerts")

CATALOGS = {
    "PAINS_A": FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A,
    "PAINS_B": FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B,
    "PAINS_C": FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C,
    "BRENK":   FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK,
    "NIH":     FilterCatalog.FilterCatalogParams.FilterCatalogs.NIH,
}


def _remove_outputs(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def _write_json_atomic(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-sdf", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    _remove_outputs(args.out_json)
    sup = Chem.SDMolSupplier(str(args.in_sdf), removeHs=False)
    mol = next((m for m in sup if m is not None), None)
    if mol is None:
        raise SystemExit(f"No parseable mol in {args.in_sdf}")
    # consensus는 다섯 입력의 SMILES가 정확히 같은지 확인한다. SDF의 명시적 수소를
    # 그대로 쓰면 admet_ai.json·웹 모델 산출물과 문자열이 어긋나 합의 계산이 죽는다.
    stripped = Chem.RemoveHs(mol)
    if stripped is not None and stripped.GetNumAtoms():
        mol = stripped

    matches: dict[str, list[str]] = {k: [] for k in CATALOGS}
    for name, cat_enum in CATALOGS.items():
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(cat_enum)
        cat = FilterCatalog.FilterCatalog(params)
        for entry in cat.GetMatches(mol):
            matches[name].append(entry.GetDescription())

    payload = {
        "smiles": Chem.MolToSmiles(mol, canonical=True),
        "matches": matches,
        "any_pains": any(matches[k] for k in ("PAINS_A", "PAINS_B", "PAINS_C")),
        "any_brenk": bool(matches["BRENK"]),
        "any_nih":   bool(matches["NIH"]),
    }
    _write_json_atomic(payload, args.out_json)
    LOG.info("Structural alerts → %s", args.out_json)


if __name__ == "__main__":
    main()
