#!/usr/bin/env python3
"""stage2_5_cosing_match.py — Compare input compound vs CosIng INCI database.

Tiers (INSTRUCTIONS.md §6.1):
    Tanimoto = 1.0 or identical InChIKey → EXACT
    Tanimoto ≥ 0.85                       → SIMILAR
    Tanimoto ≥ 0.65                       → ANALOG
    otherwise                              → NEW

Note: rdkit / numpy / pandas imports are deferred inside `main()` and helpers
so that the pure `classify()` function stays unit-testable in lean envs.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

LOG = logging.getLogger("stage2_5.cosing")
UINT64_MAX = 2**64 - 1


def _remove_outputs(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def _write_json_atomic(
    payload: dict[str, Any],
    path: Path,
    *,
    indent: int | None = 2,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=indent))
    tmp.replace(path)


@dataclass(frozen=True)
class MatchResult:
    level: str  # EXACT | SIMILAR | ANALOG | NEW
    inci: str | None
    functions: list[str]
    tanimoto: float


def classify(tanimoto: float, exact_inchi: bool,
             similar_threshold: float, analog_threshold: float) -> str:
    """Pure tier-decision function — covered by test_stage2_5.py."""
    if exact_inchi or tanimoto >= 0.999:
        return "EXACT"
    if tanimoto >= similar_threshold:
        return "SIMILAR"
    if tanimoto >= analog_threshold:
        return "ANALOG"
    return "NEW"


def _validate_thresholds(similar_threshold: float, analog_threshold: float) -> None:
    for name, value in (
        ("--similar-threshold", similar_threshold),
        ("--analog-threshold", analog_threshold),
    ):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise SystemExit(f"{name} must be a finite value in [0, 1]: {value:g}")
    if similar_threshold < analog_threshold:
        raise SystemExit(
            "--similar-threshold must be >= --analog-threshold: "
            f"{similar_threshold:g} < {analog_threshold:g}"
        )


def _parse_uint64_word(value: Any) -> int:
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


def _to_bitvect(packed: list[Any]) -> Any:
    import numpy as np
    from rdkit import DataStructs

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


def _validate_packed_ecfp4(cosing: Any, path: Path) -> None:
    invalid_indexes: list[int] = []
    for idx, value in cosing["ecfp4"].items():
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
            "CosIng parquet column 'ecfp4' must contain 32 packed uint64 "
            f"integers at row index(es) {shown}{suffix}: {path}"
        )


def _validate_nonblank_text_column(cosing: Any, path: Path, column: str) -> None:
    invalid_indexes: list[int] = []
    for idx, value in cosing[column].items():
        if value is None or str(value).strip() == "":
            invalid_indexes.append(int(idx))
    if invalid_indexes:
        shown = ",".join(str(idx) for idx in invalid_indexes[:10])
        suffix = "..." if len(invalid_indexes) > 10 else ""
        raise SystemExit(
            f"CosIng parquet column '{column}' contains blank values at "
            f"row index(es) {shown}{suffix}: {path}"
        )


def _load_query(sdf_path: Path) -> tuple[str, Any]:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    from build_activity_retrieval_index import standardize_parent

    sup = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mol = next((m for m in sup if m is not None), None)
    if mol is None:
        raise SystemExit(f"No mol in {sdf_path}")
    stripped = Chem.RemoveHs(mol)
    if stripped is not None and stripped.GetNumAtoms():
        mol = stripped
    # Stage 1은 도킹용으로 pH 7.4 이온 형태를 만든다. CosIng 표는 중성 구조라,
    # 이온화된 채로 InChIKey를 계산하면 같은 분자도 exact가 아니라 ANALOG/NEW로
    # 떨어진다(실측: 글라브리딘 -M→0.81, 하이드록시레스베라트롤 -J→0.33).
    # 검색 인덱스와 같은 표준화 함수로 중성화한 뒤 질의한다.
    try:
        mol = standardize_parent(mol)
    except ValueError as exc:
        raise SystemExit(f"Unable to standardize query from {sdf_path}: {exc}") from exc
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    bv = generator.GetFingerprint(mol)
    return Chem.MolToInchiKey(mol), bv


def _attach_standard_keys(cosing: Any) -> Any:
    """참조의 저장 `inchikey`는 stage0가 표준화 전에 만든 값이라 낡을 수 있다.

    실행 경로 질의는 중성 표준화 키로 계산되므로, 참조도 같은 함수로 한 번 더
    계산해 `standard_key`로 붙이고 둘 다 본다. `smiles` 열이 없으면(합성 테스트
    등) 저장 키만 쓴다.
    """
    if "smiles" not in cosing.columns:
        return cosing
    from rdkit import Chem

    from build_activity_retrieval_index import standardize_parent

    keys: list[str] = []
    for value in cosing["smiles"].tolist():
        key = ""
        if value is not None and str(value).strip():
            mol = Chem.MolFromSmiles(str(value))
            if mol is not None:
                try:
                    key = Chem.MolToInchiKey(standardize_parent(mol))
                except (ValueError, RuntimeError):
                    key = ""
        keys.append(key)
    out = cosing.copy()
    out["standard_key"] = keys
    return out


def _best_match(query_inchi: str, query_bv: Any, cosing_df: Any,
                similar: float, analog: float) -> MatchResult:
    from rdkit.Chem import DataStructs

    best_sim = 0.0
    best_row = None
    exact = False
    for _, row in cosing_df.iterrows():
        stored = str(row.get("inchikey"))
        standardized = str(row.get("standard_key") or "")
        if stored == query_inchi or standardized == query_inchi:
            exact = True
            best_row = row
            best_sim = 1.0
            break
        bv = _to_bitvect(list(row["ecfp4"]))
        sim = DataStructs.TanimotoSimilarity(query_bv, bv)
        if sim > best_sim:
            best_sim = sim
            best_row = row

    level = classify(best_sim, exact, similar, analog)
    if level == "NEW" or best_row is None:
        return MatchResult(level="NEW", inci=None, functions=[], tanimoto=best_sim)
    funcs = str(best_row.get("functions", ""))
    return MatchResult(
        level=level,
        inci=str(best_row.get("inci_name")),
        functions=[f for f in funcs.split(";") if f.strip()],
        tanimoto=best_sim,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-sdf", required=True, type=Path)
    parser.add_argument("--cosing-parquet", required=True, type=Path)
    parser.add_argument("--similar-threshold", type=float, default=0.85)
    parser.add_argument("--analog-threshold", type=float, default=0.65)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument(
        "--allow-missing-reference",
        action="store_true",
        help="Emit NEW only for explicit degraded diagnostics when CosIng is unavailable.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    _remove_outputs(args.out_json)
    _validate_thresholds(args.similar_threshold, args.analog_threshold)
    if not args.cosing_parquet.exists():
        if not args.allow_missing_reference:
            raise SystemExit(
                f"CosIng parquet is required for Stage 2.5: {args.cosing_parquet}. "
                "Use --allow-missing-reference only for explicit degraded diagnostics."
            )
        LOG.warning("CosIng parquet missing; emitting explicit degraded NEW result")
        _write_json_atomic({
            "level": "NEW",
            "inci": None,
            "functions": [],
            "tanimoto": 0.0,
            "reference_status": "missing_degraded",
        }, args.out_json, indent=None)
        return

    import pandas as pd
    cosing = pd.read_parquet(args.cosing_parquet)
    required = {"inchikey", "ecfp4", "inci_name", "functions"}
    missing_cols = required - set(cosing.columns)
    if missing_cols:
        raise SystemExit(
            "CosIng parquet is missing required column(s): "
            + ", ".join(sorted(missing_cols))
        )
    if cosing.empty:
        if not args.allow_missing_reference:
            raise SystemExit(f"CosIng parquet is empty: {args.cosing_parquet}")
        LOG.warning("CosIng parquet empty; emitting explicit degraded NEW result")
        _write_json_atomic({
            "level": "NEW",
            "inci": None,
            "functions": [],
            "tanimoto": 0.0,
            "reference_status": "empty_degraded",
        }, args.out_json, indent=None)
        return
    _validate_nonblank_text_column(cosing, args.cosing_parquet, "inchikey")
    _validate_nonblank_text_column(cosing, args.cosing_parquet, "inci_name")
    _validate_packed_ecfp4(cosing, args.cosing_parquet)
    cosing = _attach_standard_keys(cosing)
    query_inchi, query_bv = _load_query(args.in_sdf)
    match = _best_match(query_inchi, query_bv, cosing,
                        args.similar_threshold, args.analog_threshold)

    _write_json_atomic({
        "level": match.level,
        "inci": match.inci,
        "functions": match.functions,
        "tanimoto": match.tanimoto,
        "reference_status": "ok",
    }, args.out_json)
    LOG.info("CosIng match: %s (inci=%s, t=%.3f)",
             match.level, match.inci, match.tanimoto)


if __name__ == "__main__":
    main()
