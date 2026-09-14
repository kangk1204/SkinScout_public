#!/usr/bin/env python3
"""stage0_drug_avoidance.py — Drug-similarity reference DB.

Merges:
  * ChEMBL37 phase-4 approved subset (mirrored at data/chembl37/)
  • FDA Orange Book (operator-supplied: <OUTDIR>/orange_book.csv)
  • DrugBank approved subset (only if licensed; otherwise empty placeholder)

Output:
  <OUTDIR>/drugs.parquet      cols: drug_id, name, smiles, inchikey,
                              ecfp4 (32 decimal uint64 words)
  <OUTDIR>/scaffolds.parquet  cols: scaffold_smiles, n_drugs
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

LOG = logging.getLogger("stage0.drug_avoidance")
_MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _remove_outputs(*paths: Path) -> None:
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _write_parquet_pair_atomic(
    drugs_path: Path,
    drugs: pd.DataFrame,
    scaffolds_path: Path,
    scaffolds: pd.DataFrame,
) -> None:
    drugs_path.parent.mkdir(parents=True, exist_ok=True)
    scaffolds_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_drugs = drugs_path.with_suffix(drugs_path.suffix + ".tmp")
    tmp_scaffolds = scaffolds_path.with_suffix(scaffolds_path.suffix + ".tmp")
    drugs.to_parquet(tmp_drugs, index=False)
    scaffolds.to_parquet(tmp_scaffolds, index=False)
    tmp_drugs.replace(drugs_path)
    tmp_scaffolds.replace(scaffolds_path)


def _ecfp4(mol: Chem.Mol) -> list[str]:
    bv = _MORGAN_GENERATOR.GetFingerprint(mol)
    arr = np.zeros(2048, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    packed = np.packbits(arr).view(np.uint64).copy()
    if packed.dtype.byteorder == ">":
        packed = packed.byteswap()
    return [str(int(word)) for word in packed]


def load_chembl_approved(chembl_dir: Path) -> pd.DataFrame:
    fp = chembl_dir / "human_activities.parquet"
    if not fp.exists():
        return pd.DataFrame(columns=["drug_id", "name", "smiles"])
    df = pd.read_parquet(fp)
    required = ["molecule_chembl_id", "smiles"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            "ChEMBL approved parquet is missing required columns "
            f"{missing}: {fp}"
        )
    df = df[required].copy()
    _reject_conflicting_chembl_smiles(df, fp)
    df = df.drop_duplicates("molecule_chembl_id")
    df["drug_id"] = df["molecule_chembl_id"]
    df["name"] = df["molecule_chembl_id"]  # placeholder
    return df[["drug_id", "name", "smiles"]]


def load_orange_book(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["drug_id", "name", "smiles"])
    df = pd.read_csv(path)
    lo = {c.lower(): c for c in df.columns}
    nc = lo.get("name") or lo.get("ingredient") or lo.get("drug_name")
    sc = lo.get("smiles")
    if nc is None or sc is None:
        raise ValueError(
            "Orange Book reference must include a drug name column "
            "('name', 'ingredient', or 'drug_name') and a 'smiles' column: "
            f"{path}"
        )
    out = df[[nc, sc]].rename(columns={nc: "name", sc: "smiles"})
    out["drug_id"] = out["name"]
    return out[["drug_id", "name", "smiles"]]


def load_drugbank(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["drug_id", "name", "smiles"])
    df = pd.read_parquet(path)
    required = ["drug_id", "name", "smiles"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            "DrugBank approved parquet is missing required columns "
            f"{missing}: {path}"
        )
    return df[required]


def _required_text(value: object, column: str, row_idx: int) -> str:
    if pd.isna(value):
        raise ValueError(
            f"Drug-avoidance reference column '{column}' contains blank values "
            f"at row index {row_idx}"
        )
    text = str(value).strip()
    if not text:
        raise ValueError(
            f"Drug-avoidance reference column '{column}' contains blank values "
            f"at row index {row_idx}"
        )
    return text


def _canonical_smiles_key(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return text
    return Chem.MolToSmiles(mol, canonical=True)


def _reject_conflicting_chembl_smiles(df: pd.DataFrame, path: Path) -> None:
    normalized = pd.DataFrame({
        "molecule_chembl_id": df["molecule_chembl_id"].fillna("").astype(str).str.strip(),
        "smiles": df["smiles"].map(_canonical_smiles_key),
    })
    counts = normalized.groupby("molecule_chembl_id", dropna=False)["smiles"].nunique()
    conflicts = [idx for idx, count in counts.items() if count > 1]
    if conflicts:
        shown = ",".join(str(value) for value in conflicts[:10])
        suffix = "..." if len(conflicts) > 10 else ""
        raise ValueError(
            "ChEMBL approved parquet contains duplicate molecule_chembl_id "
            "values with conflicting SMILES: "
            f"{shown}{suffix}: {path}"
        )


def to_records(df: pd.DataFrame) -> list[dict]:
    records = []
    for row_idx, row in df.iterrows():
        drug_id = _required_text(row.get("drug_id"), "drug_id", row_idx)
        name = _required_text(row.get("name"), "name", row_idx)
        smi = _required_text(row.get("smiles"), "smiles", row_idx)
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(
                "Drug-avoidance reference column 'smiles' contains invalid "
                f"SMILES at row index {row_idx}: {smi!r}"
            )
        records.append({
            "drug_id": drug_id,
            "name": name,
            "smiles": smi,
            "inchikey": Chem.MolToInchiKey(mol),
            "ecfp4": _ecfp4(mol),
            "scaffold_smiles": Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol)),
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--chembl-dir", type=Path, default=Path("data/chembl37"))
    parser.add_argument("--orange-book", type=Path,
                        default=Path("data/drug_avoidance/orange_book.csv"))
    parser.add_argument("--drugbank-parquet", type=Path,
                        default=Path("data/drugbank/drugbank_approved.parquet"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_drugs = args.out_dir / "drugs.parquet"
    out_scaff = args.out_dir / "scaffolds.parquet"
    _remove_outputs(out_drugs, out_scaff)
    if args.dry_run:
        df = pd.DataFrame([
            {"drug_id": "PLACEHOLDER1", "name": "Aspirin",
             "smiles": "CC(=O)Oc1ccccc1C(=O)O"},
            {"drug_id": "PLACEHOLDER2", "name": "Ibuprofen",
             "smiles": "CC(C)Cc1ccc(C(C)C(=O)O)cc1"},
        ])
    else:
        df = pd.concat([
            load_chembl_approved(args.chembl_dir),
            load_orange_book(args.orange_book),
            load_drugbank(args.drugbank_parquet),
        ], ignore_index=True).drop_duplicates("smiles")
        if df.empty:
            raise SystemExit(
                "Drug-avoidance full ingest found no approved-drug rows from "
                f"ChEMBL={args.chembl_dir}, OrangeBook={args.orange_book}, "
                f"DrugBank={args.drugbank_parquet}. Use --dry-run only for "
                "explicit placeholder diagnostics."
            )

    records = to_records(df)
    if not records:
        raise SystemExit(
            "Drug-avoidance ingest produced no valid molecular records. "
            "Use --dry-run only for explicit placeholder diagnostics."
        )
    drugs = pd.DataFrame(records)
    scaffolds = (
        drugs.groupby("scaffold_smiles")
        .size().reset_index(name="n_drugs")
        .sort_values("n_drugs", ascending=False)
    )
    _write_parquet_pair_atomic(out_drugs, drugs, out_scaff, scaffolds)
    LOG.info("Wrote %s (n=%d)", out_drugs, len(records))
    LOG.info("Wrote %s (n=%d)", out_scaff, len(scaffolds))


if __name__ == "__main__":
    main()
