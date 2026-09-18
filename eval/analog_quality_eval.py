#!/usr/bin/env python3
"""eval/analog_quality_eval.py — Retrospective analog-generation quality
(INSTRUCTIONS §18.4).

For each known cosmetic ingredient, generate 100 analogs and compute:
    recovery   — fraction of other known cosmetic INCI within Tanimoto ≥ 0.5
  novelty    - fraction with Tanimoto < 0.4 vs the configured ChEMBL snapshot
    synthesizability — average RAscore (≥ 0.7 considered routine)
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import pandas as pd

LOG = logging.getLogger("eval.analog_quality")


def _read_required_table(path: Path, required_cols: set[str], label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"{label} is required and must be non-empty: {path}")
    try:
        if path.suffix == ".parquet":
            df = pd.read_parquet(path, columns=sorted(required_cols))
        else:
            df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"{label} failed to parse: {path}: {exc}") from exc
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise SystemExit(f"{label} missing required columns {missing}: {path}")
    if df.empty:
        raise SystemExit(f"{label} contains no rows: {path}")
    return df


def _validate_smiles(values: pd.Series, label: str, path: Path) -> list[str]:
    from rdkit import Chem

    valid: list[str] = []
    invalid_rows: list[int] = []
    for idx, value in values.items():
        smi = "" if pd.isna(value) else str(value).strip()
        if not smi or Chem.MolFromSmiles(smi) is None:
            invalid_rows.append(int(idx))
            continue
        valid.append(smi)
    if invalid_rows:
        shown = ", ".join(str(i) for i in invalid_rows[:10])
        suffix = "..." if len(invalid_rows) > 10 else ""
        raise SystemExit(
            f"{label} contains invalid SMILES at row index(es) {shown}{suffix}: {path}"
        )
    if not valid:
        raise SystemExit(f"{label} has no valid SMILES: {path}")
    return valid


def _validate_ra_scores(values: pd.Series, label: str, path: Path) -> list[float]:
    scores: list[float] = []
    invalid_rows: list[int] = []
    for idx, value in values.items():
        text = "" if pd.isna(value) else str(value).strip()
        try:
            score = float(text) if text else float("nan")
        except ValueError:
            invalid_rows.append(int(idx))
            continue
        if not math.isfinite(score) or score < 0.0 or score > 1.0:
            invalid_rows.append(int(idx))
            continue
        scores.append(score)
    if invalid_rows:
        shown = ", ".join(str(i) for i in invalid_rows[:10])
        suffix = "..." if len(invalid_rows) > 10 else ""
        raise SystemExit(
            f"{label} column 'ra_score' contains invalid values at row "
            f"index(es) {shown}{suffix}; expected finite values in [0, 1]: {path}"
        )
    return scores


def _validate_unique_canonical_smiles(smiles: list[str], label: str, path: Path) -> None:
    from rdkit import Chem

    seen: dict[str, int] = {}
    duplicate_rows: list[str] = []
    for row_idx, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        canonical = Chem.MolToSmiles(mol, canonical=True)
        if canonical in seen:
            duplicate_rows.append(f"{canonical} at row index(es) {seen[canonical]}, {row_idx}")
            continue
        seen[canonical] = row_idx
    if duplicate_rows:
        shown = "; ".join(duplicate_rows[:10])
        suffix = "..." if len(duplicate_rows) > 10 else ""
        raise SystemExit(
            f"{label} contains duplicate canonical SMILES {shown}{suffix}: {path}"
        )


def _validate_fraction_threshold(value: float, label: str) -> None:
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise SystemExit(
            f"{label} must be a finite value in [0, 1]: {value}"
        )


def _max_tanimoto(query_smi: str, refs: list[str]) -> float:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    mol = Chem.MolFromSmiles(query_smi)
    if mol is None:
        raise ValueError(f"Invalid query SMILES: {query_smi}")
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    best = 0.0
    for r in refs:
        ref = Chem.MolFromSmiles(r)
        if ref is None:
            continue
        rbv = AllChem.GetMorganFingerprintAsBitVect(ref, radius=2, nBits=2048)
        best = max(best, DataStructs.TanimotoSimilarity(bv, rbv))
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analogs-csv", required=True, type=Path)
    parser.add_argument("--other-cosing-csv", required=True, type=Path,
                        help="CSV of other known INCI SMILES")
    parser.add_argument("--chembl-fp-parquet", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--min-recovery", type=float, default=0.01)
    parser.add_argument("--min-novelty", type=float, default=0.80)
    parser.add_argument("--min-mean-ra-score", type=float, default=0.70)
    parser.add_argument(
        "--allow-threshold-failure",
        action="store_true",
        help="write failing analog-quality metrics only for explicit diagnostics",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.out_csv.exists():
        args.out_csv.unlink()
    _validate_fraction_threshold(args.min_recovery, "--min-recovery")
    _validate_fraction_threshold(args.min_novelty, "--min-novelty")
    _validate_fraction_threshold(args.min_mean_ra_score, "--min-mean-ra-score")
    analogs = _read_required_table(args.analogs_csv, {"smiles"}, "Analog CSV")
    if "ra_score" not in analogs.columns:
        raise SystemExit(
            "Analog CSV missing required column 'ra_score' for synthesizability "
            f"quality gate: {args.analogs_csv}"
        )
    analog_smiles = _validate_smiles(analogs["smiles"], "Analog CSV", args.analogs_csv)
    _validate_unique_canonical_smiles(
        analog_smiles,
        "Analog CSV",
        args.analogs_csv,
    )
    sa_scores = _validate_ra_scores(analogs["ra_score"], "Analog CSV", args.analogs_csv)
    other_smiles = _validate_smiles(
        _read_required_table(args.other_cosing_csv, {"smiles"}, "Other CosIng CSV")[
            "smiles"
        ],
        "Other CosIng CSV",
        args.other_cosing_csv,
    )
    chembl_smiles = _validate_smiles(
        _read_required_table(args.chembl_fp_parquet, {"smiles"}, "ChEMBL fingerprint parquet")[
            "smiles"
        ],
        "ChEMBL fingerprint parquet",
        args.chembl_fp_parquet,
    )

    recoveries: list[float] = []
    novelties: list[float] = []
    for smi in analog_smiles:
        rec = _max_tanimoto(smi, other_smiles)
        chembl_sim = _max_tanimoto(smi, chembl_smiles)
        recoveries.append(1.0 if rec >= 0.5 else 0.0)
        novelties.append(1.0 if chembl_sim < 0.4 else 0.0)
    if not recoveries:
        raise SystemExit(f"Analog CSV contains no evaluable analogs: {args.analogs_csv}")

    row = {
        "n_analogs": len(analogs),
        "recovery_fraction": sum(recoveries) / len(recoveries),
        "novelty_fraction": sum(novelties) / len(novelties),
        "mean_ra_score": (sum(sa_scores) / len(sa_scores)) if sa_scores else float("nan"),
    }
    failures: list[str] = []
    if row["recovery_fraction"] < args.min_recovery:
        failures.append(
            f"recovery_fraction={row['recovery_fraction']:.3f} < {args.min_recovery:.3f}"
        )
    if row["novelty_fraction"] < args.min_novelty:
        failures.append(
            f"novelty_fraction={row['novelty_fraction']:.3f} < {args.min_novelty:.3f}"
        )
    if pd.isna(row["mean_ra_score"]) or row["mean_ra_score"] < args.min_mean_ra_score:
        value = "nan" if pd.isna(row["mean_ra_score"]) else f"{row['mean_ra_score']:.3f}"
        failures.append(f"mean_ra_score={value} < {args.min_mean_ra_score:.3f}")
    row["passes_threshold"] = not failures
    if failures and not args.allow_threshold_failure:
        raise SystemExit(
            "Analog quality failed claim threshold(s): "
            + "; ".join(failures)
            + "; pass --allow-threshold-failure only for explicit diagnostics"
        )
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = args.out_csv.with_suffix(args.out_csv.suffix + ".tmp")
    pd.DataFrame([row]).to_csv(tmp_csv, index=False)
    tmp_csv.replace(args.out_csv)
    LOG.info("Analog quality → %s", args.out_csv)


if __name__ == "__main__":
    main()
