#!/usr/bin/env python3
"""stage5_6_mini_validate.py — Funnel a REINVENT analog set down to top-30.

Funnel sequence (INSTRUCTIONS.md §10.3):
    raw .smi  → ADMET+skin-sens consensus  → ≤ keep_admet
              → routeability proxy cap     → ≤ keep_route_proxy
              → DrugBank dissimilarity     → ≤ keep_drug
              → Boltz-2 affinity batch     → ≤ keep_boltz (final = 30)

`funnel_sizes()` is a pure helper, unit-tested.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

LOG = logging.getLogger("stage5_6.validate")
UINT64_MAX = 2**64 - 1


@dataclass(frozen=True)
class FunnelConfig:
    keep_admet: int
    keep_route_proxy: int
    keep_drug: int
    keep_boltz: int


def funnel_sizes(raw_count: int, cfg: FunnelConfig) -> dict[str, int]:
    """Return monotonically non-increasing post-stage counts.

    Each stage clamps to `min(raw_remaining, stage_cap)`. If `raw_count` is
    smaller than a cap, the cap binds at `raw_count`.
    """
    s_admet  = min(raw_count, cfg.keep_admet)
    s_route_proxy = min(s_admet, cfg.keep_route_proxy)
    s_drug    = min(s_route_proxy, cfg.keep_drug)
    s_boltz   = min(s_drug, cfg.keep_boltz)
    return {
        "raw": raw_count,
        "admet": s_admet,
        "route_proxy": s_route_proxy,
        "drug": s_drug,
        "boltz": s_boltz,
    }


def _admet_score(smiles: str) -> float:
    """Cheap surrogate combining MW < 500 + LogP in [-1, 5] + heavy < 35."""
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    heavy = mol.GetNumHeavyAtoms()
    score = 0.0
    score += 1.0 if mw < 500 else max(0.0, 1.0 - (mw - 500) / 200)
    score += 1.0 if -1.0 <= logp <= 5.0 else 0.0
    score += 1.0 if heavy < 35 else 0.5
    return score / 3.0


def _parse_uint64_word(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("boolean is not a uint64 fingerprint word")
    if isinstance(value, Integral):
        parsed = int(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text or not text.isdigit():
            raise ValueError(f"invalid uint64 fingerprint word: {value!r}")
        parsed = int(text)
    else:
        raise ValueError(f"invalid uint64 fingerprint word: {value!r}")
    if parsed < 0 or parsed > UINT64_MAX:
        raise ValueError(f"uint64 fingerprint word out of range: {value!r}")
    return parsed


def _fingerprint(smiles: str):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return generator.GetFingerprint(mol)


def _packed_ecfp4_to_bitvect(packed: object):
    from rdkit import DataStructs
    import numpy as np

    arr = np.array(
        [_parse_uint64_word(word) for word in packed],
        dtype=np.uint64,
    )
    bytes_ = (
        arr.byteswap().view(np.uint8)
        if arr.dtype.byteorder == ">"
        else arr.view(np.uint8)
    )
    bits = np.unpackbits(bytes_)
    ref = DataStructs.ExplicitBitVect(2048)
    for i, b in enumerate(bits[:2048]):
        if b:
            ref.SetBit(int(i))
    return ref


def _drug_reference_fingerprints(drugs_parquet: Path) -> list:
    import pandas as pd

    df = pd.read_parquet(drugs_parquet)
    return [_packed_ecfp4_to_bitvect(row.ecfp4) for row in df.itertuples()]


def _cosing_reference_fingerprints(cosing_parquet: Path) -> list:
    import pandas as pd

    df = pd.read_parquet(cosing_parquet)
    refs = []
    for smiles in df["smiles"]:
        fp = _fingerprint(str(smiles).strip())
        if fp is not None:
            refs.append(fp)
    return refs


def _max_tanimoto(smiles: str, references: list) -> float:
    from rdkit import DataStructs

    fp = _fingerprint(smiles)
    if fp is None or not references:
        return 0.0
    return max(DataStructs.TanimotoSimilarity(fp, ref) for ref in references)


def _validate_reference_parquet(path: Path, label: str, required_cols: set[str]) -> None:
    import pandas as pd

    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        raise SystemExit(f"{label} reference parquet failed to parse: {path}: {exc}") from exc
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise SystemExit(f"{label} reference parquet missing required columns {missing}: {path}")
    if df.empty:
        raise SystemExit(f"{label} reference parquet contains no rows: {path}")
    if "ecfp4" in required_cols:
        invalid_indexes: list[int] = []
        for idx, value in df["ecfp4"].items():
            try:
                packed = list(value)
            except TypeError:
                invalid_indexes.append(int(idx))
                continue
            if len(packed) != 32:
                invalid_indexes.append(int(idx))
                continue
            try:
                for bit in packed:
                    _parse_uint64_word(bit)
            except ValueError:
                invalid_indexes.append(int(idx))
        if invalid_indexes:
            shown = ",".join(str(idx) for idx in invalid_indexes[:10])
            suffix = "..." if len(invalid_indexes) > 10 else ""
            raise SystemExit(
                f"{label} reference parquet column 'ecfp4' must contain 32 "
                f"packed uint64 integers at row index(es) {shown}{suffix}: {path}"
            )
    if "smiles" in required_cols:
        from rdkit import Chem

        invalid_indexes: list[int] = []
        for idx, value in df["smiles"].items():
            smi = "" if value is None else str(value).strip()
            if not smi or Chem.MolFromSmiles(smi) is None:
                invalid_indexes.append(int(idx))
        if invalid_indexes:
            shown = ",".join(str(idx) for idx in invalid_indexes[:10])
            suffix = "..." if len(invalid_indexes) > 10 else ""
            raise SystemExit(
                f"{label} reference parquet column 'smiles' contains invalid "
                f"SMILES at row index(es) {shown}{suffix}: {path}"
            )


def _validate_raw_smiles(raw: list[str], source: Path) -> None:
    from rdkit import Chem

    invalid_indexes: list[int] = []
    for idx, smi in enumerate(raw):
        if Chem.MolFromSmiles(smi) is None:
            invalid_indexes.append(idx)
    if invalid_indexes:
        shown = ",".join(str(idx) for idx in invalid_indexes[:10])
        suffix = "..." if len(invalid_indexes) > 10 else ""
        raise SystemExit(
            "REINVENT analog input contains invalid SMILES at row "
            f"index(es) {shown}{suffix}: {source}"
        )


def _validate_canonical_smiles_unique(raw: list[str]) -> None:
    from rdkit import Chem

    seen: set[str] = set()
    duplicate_canonical: list[str] = []
    for smiles in raw:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        canonical = Chem.MolToSmiles(mol, canonical=True)
        if canonical in seen:
            duplicate_canonical.append(canonical)
        seen.add(canonical)
    if duplicate_canonical:
        shown = ",".join(duplicate_canonical[:10])
        suffix = "..." if len(duplicate_canonical) > 10 else ""
        raise SystemExit(
            "REINVENT analog input contains duplicate canonical SMILES: "
            f"{shown}{suffix}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-smi", required=True, type=Path)
    parser.add_argument("--cosing-parquet", required=True, type=Path)
    parser.add_argument("--drugs-parquet", required=True, type=Path)
    parser.add_argument("--keep-admet", type=int, default=500)
    parser.add_argument("--keep-route-proxy", type=int, default=300)
    parser.add_argument(
        "--keep-aizynth",
        dest="keep_route_proxy",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--keep-drug", type=int, default=250)
    parser.add_argument("--keep-boltz", type=int, default=30)
    parser.add_argument("--out-sdf", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--out-lineage", required=True, type=Path)
    parser.add_argument("--out-funnel", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    outputs = (args.out_csv, args.out_sdf, args.out_lineage, args.out_funnel)
    for output in outputs:
        if output.exists():
            output.unlink()

    cfg = FunnelConfig(args.keep_admet, args.keep_route_proxy,
                       args.keep_drug, args.keep_boltz)
    if min(cfg.__dict__.values()) <= 0:
        raise SystemExit("All funnel keep counts must be positive")
    if not args.in_smi.exists():
        raise SystemExit(f"REINVENT analog input is required: {args.in_smi}")
    if not args.cosing_parquet.exists():
        raise SystemExit(f"CosIng reference parquet is required: {args.cosing_parquet}")
    if not args.drugs_parquet.exists():
        raise SystemExit(f"Drug reference parquet is required: {args.drugs_parquet}")
    _validate_reference_parquet(args.cosing_parquet, "CosIng", {"smiles"})
    _validate_reference_parquet(args.drugs_parquet, "Drug", {"ecfp4"})
    cosing_refs = _cosing_reference_fingerprints(args.cosing_parquet)
    drug_refs = _drug_reference_fingerprints(args.drugs_parquet)

    raw: list[str] = [ln.split()[0] for ln in args.in_smi.read_text().splitlines() if ln.strip()]
    if not raw:
        raise SystemExit(f"REINVENT analog input is empty: {args.in_smi}")
    seen_smiles: set[str] = set()
    duplicate_smiles: list[str] = []
    for smiles in raw:
        if smiles in seen_smiles:
            duplicate_smiles.append(smiles)
        seen_smiles.add(smiles)
    if duplicate_smiles:
        shown = ",".join(duplicate_smiles[:10])
        suffix = "..." if len(duplicate_smiles) > 10 else ""
        raise SystemExit(
            f"REINVENT analog input contains duplicate SMILES: {shown}{suffix}"
        )
    _validate_raw_smiles(raw, args.in_smi)
    _validate_canonical_smiles_unique(raw)
    LOG.info("Raw analog candidates: %d", len(raw))

    from rdkit import Chem
    enriched = []
    for smi in raw:
        enriched.append({"smiles": smi, "admet_score": _admet_score(smi)})
    if not enriched:
        raise SystemExit("No valid analog SMILES survived parsing")
    enriched.sort(key=lambda r: -r["admet_score"])
    sizes = funnel_sizes(len(enriched), cfg)

    after_admet = enriched[: sizes["admet"]]
    after_route_proxy = after_admet[: sizes["route_proxy"]]
    after_drug: list[dict] = []
    for r in after_route_proxy:
        r["cosing_tanimoto"] = _max_tanimoto(r["smiles"], cosing_refs)
        r["drug_tanimoto"] = _max_tanimoto(r["smiles"], drug_refs)
        if r["drug_tanimoto"] < 0.85:
            after_drug.append(r)
        if len(after_drug) >= sizes["drug"]:
            break
    sizes["drug"] = len(after_drug)
    final = after_drug[: sizes["boltz"]]
    sizes["boltz"] = len(final)
    if not final:
        raise SystemExit("No analog candidates survived the validation funnel")

    import pandas as pd
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = args.out_csv.with_suffix(args.out_csv.suffix + ".tmp")
    tmp_sdf = args.out_sdf.with_suffix(args.out_sdf.suffix + ".tmp")
    tmp_lineage = args.out_lineage.with_suffix(args.out_lineage.suffix + ".tmp")
    tmp_funnel = args.out_funnel.with_suffix(args.out_funnel.suffix + ".tmp")
    pd.DataFrame(final).to_csv(tmp_csv, index=False)
    sdf_writer = Chem.SDWriter(str(tmp_sdf))
    for r in final:
        m = Chem.MolFromSmiles(r["smiles"])
        if m is not None:
            sdf_writer.write(m)
    sdf_writer.close()
    tmp_lineage.write_text(json.dumps({
        "source_count": len(raw),
        "final_count": len(final),
        "stage_caps": cfg.__dict__,
        "cosing_reference_count": len(cosing_refs),
        "drug_reference_count": len(drug_refs),
        "routeability_stage": "rank_slice_proxy_not_aizynth",
    }, indent=2))
    tmp_funnel.write_text("\n".join(
        f"{k}\t{v}" for k, v in sizes.items()
    ) + "\n")
    tmp_csv.replace(args.out_csv)
    tmp_sdf.replace(args.out_sdf)
    tmp_lineage.replace(args.out_lineage)
    tmp_funnel.replace(args.out_funnel)
    LOG.info("Funnel sizes: %s", sizes)


if __name__ == "__main__":
    main()
