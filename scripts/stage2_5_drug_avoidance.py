#!/usr/bin/env python3
"""stage2_5_drug_avoidance.py — Compare input compound vs approved-drug DB.

Emits three warning tiers (INSTRUCTIONS.md §6.2):
    Tanimoto ≥ 0.85                 → STRICT_WARNING
    Bemis-Murcko scaffold identical → SCAFFOLD_MATCH
    0.65 ≤ Tanimoto < 0.85          → SOFT_WARNING
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

LOG = logging.getLogger("stage2_5.drugs")
UINT64_MAX = 2**64 - 1
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _remove_outputs(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def _write_json_atomic(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


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


def to_bitvect(packed: list[object]) -> DataStructs.ExplicitBitVect:
    arr = np.array([_parse_uint64_word(word) for word in packed], dtype=np.uint64)
    bytes_ = (
        arr.byteswap().view(np.uint8)
        if arr.dtype.byteorder == ">"
        else arr.view(np.uint8)
    )
    bits = np.unpackbits(bytes_)
    bv = DataStructs.ExplicitBitVect(2048)
    for i, b in enumerate(bits[:2048]):
        if b:
            bv.SetBit(int(i))
    return bv


def validate_packed_ecfp4(drugs: pd.DataFrame, path: Path) -> None:
    invalid_indexes: list[int] = []
    for idx, value in drugs["ecfp4"].items():
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
            "Approved-drug parquet column 'ecfp4' must contain 32 packed "
            f"uint64 integers at row index(es) {shown}{suffix}: {path}"
        )


def validate_nonblank_text_column(df: pd.DataFrame, path: Path, column: str) -> None:
    invalid_indexes: list[int] = []
    for idx, value in df[column].items():
        if pd.isna(value) or str(value).strip() == "":
            invalid_indexes.append(int(idx))
    if invalid_indexes:
        shown = ",".join(str(idx) for idx in invalid_indexes[:10])
        suffix = "..." if len(invalid_indexes) > 10 else ""
        raise SystemExit(
            f"Approved-drug parquet column '{column}' contains blank values at "
            f"row index(es) {shown}{suffix}: {path}"
        )


def validate_scaffold_smiles(scaffolds: pd.DataFrame, path: Path) -> set[str]:
    valid_scaffolds: set[str] = set()
    invalid_indexes: list[int] = []
    for idx, value in scaffolds["scaffold_smiles"].items():
        if pd.isna(value):
            invalid_indexes.append(int(idx))
            continue
        smi = str(value).strip()
        if not smi:
            continue
        if Chem.MolFromSmiles(smi) is None:
            invalid_indexes.append(int(idx))
            continue
        valid_scaffolds.add(smi)
    if invalid_indexes:
        shown = ",".join(str(idx) for idx in invalid_indexes[:10])
        suffix = "..." if len(invalid_indexes) > 10 else ""
        raise SystemExit(
            "Scaffold parquet column 'scaffold_smiles' contains invalid "
            f"SMILES at row index(es) {shown}{suffix}: {path}"
        )
    return valid_scaffolds


def validate_thresholds(strict_threshold: float, soft_threshold: float) -> None:
    for name, value in (
        ("--strict-threshold", strict_threshold),
        ("--soft-threshold", soft_threshold),
    ):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise SystemExit(f"{name} must be a finite value in [0, 1]: {value:g}")
    if strict_threshold < soft_threshold:
        raise SystemExit(
            "--strict-threshold must be >= --soft-threshold: "
            f"{strict_threshold:g} < {soft_threshold:g}"
        )


def fingerprint_input_mol(mol: Chem.Mol) -> Chem.Mol:
    from build_activity_retrieval_index import standardize_parent

    stripped = Chem.RemoveHs(mol)
    if stripped is not None and stripped.GetNumAtoms():
        mol = stripped
    # Stage 1은 도킹용으로 pH 7.4 이온 형태를 만든다. drugs/scaffolds 참조는 중성
    # ChEMBL 구조라, 이온화된 채로 지문·스캐폴드를 만들면 같은 분자도 Tanimoto가
    # 임계값 아래로 떨어져 승인 성분 경고를 놓친다(이는 CosIng·Stage 3에서 이미
    # 고친 것과 같은 계열의 문제다). 검색 인덱스와 같은 함수로 중성화한다.
    try:
        return standardize_parent(mol)
    except ValueError as exc:
        raise SystemExit(f"Unable to standardize query: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-sdf", required=True, type=Path)
    parser.add_argument("--drugs-parquet", required=True, type=Path)
    parser.add_argument("--scaffolds-parquet", required=True, type=Path)
    parser.add_argument("--strict-threshold", type=float, default=0.85)
    parser.add_argument("--soft-threshold", type=float, default=0.65)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument(
        "--allow-missing-reference",
        action="store_true",
        help="Emit clean warnings only for explicit degraded diagnostics when references are unavailable.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    _remove_outputs(args.out_json)
    validate_thresholds(args.strict_threshold, args.soft_threshold)
    missing_refs = [
        str(p) for p in (args.drugs_parquet, args.scaffolds_parquet) if not p.exists()
    ]
    if missing_refs and not args.allow_missing_reference:
        raise SystemExit(
            "Approved-drug reference file(s) required for Stage 2.5 are missing: "
            + ", ".join(missing_refs)
            + ". Use --allow-missing-reference only for explicit degraded diagnostics."
        )

    sup = Chem.SDMolSupplier(str(args.in_sdf), removeHs=False)
    mol = next((m for m in sup if m is not None), None)
    if mol is None:
        raise SystemExit("No mol in input SDF")
    mol = fingerprint_input_mol(mol)
    query_bv = MORGAN_GENERATOR.GetFingerprint(mol)
    query_scaffold = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))

    warnings: list[dict] = []
    max_tanimoto = 0.0
    reference_status = "ok"

    if args.drugs_parquet.exists():
        drugs = pd.read_parquet(args.drugs_parquet)
        required = {"ecfp4", "drug_id", "name"}
        missing_cols = required - set(drugs.columns)
        if missing_cols:
            raise SystemExit(
                "Approved-drug parquet is missing required column(s): "
                + ", ".join(sorted(missing_cols))
            )
        if drugs.empty and not args.allow_missing_reference:
            raise SystemExit(f"Approved-drug parquet is empty: {args.drugs_parquet}")
        validate_nonblank_text_column(drugs, args.drugs_parquet, "drug_id")
        validate_nonblank_text_column(drugs, args.drugs_parquet, "name")
        validate_packed_ecfp4(drugs, args.drugs_parquet)
        for _, row in drugs.iterrows():
            bv = to_bitvect(list(row["ecfp4"]))
            sim = DataStructs.TanimotoSimilarity(query_bv, bv)
            max_tanimoto = max(max_tanimoto, sim)
            if sim >= args.strict_threshold:
                warnings.append({"tier": "STRICT_WARNING",
                                 "drug_id": str(row["drug_id"]),
                                 "drug_name": str(row["name"]),
                                 "tanimoto": sim})
            elif sim >= args.soft_threshold:
                warnings.append({"tier": "SOFT_WARNING",
                                 "drug_id": str(row["drug_id"]),
                                 "drug_name": str(row["name"]),
                                 "tanimoto": sim})
    else:
        reference_status = "missing_degraded"

    if args.scaffolds_parquet.exists():
        scaffolds = pd.read_parquet(args.scaffolds_parquet)
        if "scaffold_smiles" not in scaffolds.columns:
            raise SystemExit("Scaffold parquet is missing required column: scaffold_smiles")
        if scaffolds.empty and not args.allow_missing_reference:
            raise SystemExit(f"Scaffold parquet is empty: {args.scaffolds_parquet}")
        valid_scaffolds = validate_scaffold_smiles(scaffolds, args.scaffolds_parquet)
        if query_scaffold and query_scaffold in valid_scaffolds:
            warnings.append({"tier": "SCAFFOLD_MATCH",
                             "scaffold_smiles": query_scaffold})
    else:
        reference_status = "missing_degraded"
    if reference_status == "ok" and args.allow_missing_reference:
        if (
            args.drugs_parquet.exists()
            and pd.read_parquet(args.drugs_parquet).empty
        ) or (
            args.scaffolds_parquet.exists()
            and pd.read_parquet(args.scaffolds_parquet).empty
        ):
            reference_status = "empty_degraded"

    _write_json_atomic({
        "max_tanimoto_to_approved_drug": max_tanimoto,
        "warnings": warnings,
        "n_warnings": len(warnings),
        "reference_status": reference_status,
    }, args.out_json)
    LOG.info("Drug warnings: %d (max-tanimoto=%.3f)", len(warnings), max_tanimoto)


if __name__ == "__main__":
    # RDKit's static teardown intermittently aborts this process *after* main()
    # has finished and the JSON has been replaced atomically: "terminate called
    # without an active exception", SIGABRT, in roughly 1 invocation in 10. The
    # stage then fails with exit -6 while its output sits complete on disk.
    #
    # Exiting before the C++ static destructors run makes it deterministic.
    # Nothing is lost: _write_json_atomic has already renamed the file into
    # place, and both streams are flushed here.
    import os
    import sys as _sys

    try:
        main()
    except SystemExit as requested:
        code = requested.code
        if code is None:
            status = 0
        elif isinstance(code, int):
            status = code
        else:
            # Normal interpreter shutdown prints a non-int SystemExit argument to
            # stderr; exiting early skips that, and the message is the whole
            # point of every fail-closed path in this script.
            print(code, file=_sys.stderr)
            status = 1
    else:
        status = 0
    _sys.stdout.flush()
    _sys.stderr.flush()
    os._exit(status)
