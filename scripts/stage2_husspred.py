#!/usr/bin/env python3
"""stage2_husspred.py — Human Skin Sensitization predictor (HuSSPred 2024-11).

Web-only model: posts SMILES to the public HuSSPred web service endpoint used
by the current browser client.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

from rdkit import Chem

LOG = logging.getLogger("stage2.husspred")
API_URL = "https://husspred.mml.unc.edu/smiles"
MODEL_OPTIONS = {
    "calculate_ad": True,
    "Binary Sensitization": True,
    "Multiclass Sensitization Potency": True,
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


def smiles_from_sdf(sdf: Path) -> str:
    """외부 API로 보낼 정규 SMILES. 명시적 수소는 떼어 길이를 줄인다.

    Stage 1 SDF는 도킹용으로 수소를 명시해 두는데, 그대로 MolToSmiles하면
    `[H]c1c([H])...`처럼 수소가 전부 붙어 문자열이 몇 배로 길어진다.
    Pred-Skin은 500자 제한이라 실측 786자 펩타이드성 화합물이 422로 거부됐다.
    """
    sup = Chem.SDMolSupplier(str(sdf), removeHs=False)
    for mol in sup:
        if mol is None:
            continue
        stripped = Chem.RemoveHs(mol)
        if stripped is not None and stripped.GetNumAtoms():
            mol = stripped
        return Chem.MolToSmiles(mol, canonical=True)
    raise SystemExit(f"No parseable mol in {sdf}")


def _is_bool_like(value: object) -> bool:
    return (
        isinstance(value, bool)
        or type(value).__name__ == "bool_"
        or (isinstance(value, str) and value.strip().lower() in {"true", "false"})
    )


def _probability(value: object, label: str) -> float:
    if value is None:
        raise SystemExit(f"{label} is required")
    if _is_bool_like(value):
        raise SystemExit(f"{label} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{label} must be numeric") from exc
    if not math.isfinite(parsed) or parsed < 0.0 or parsed > 1.0:
        raise SystemExit(f"{label} must be finite and in [0, 1]")
    return parsed


def _skin_sens_call(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise SystemExit(f"{label} must be a string")
    call = value.strip().lower()
    if call not in {"positive", "negative"}:
        raise SystemExit(f"{label} must be 'positive' or 'negative'")
    return call


def _call_from_classification(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"{label} must be a non-empty string")
    text = value.strip().lower()
    if "non-sensitizer" in text or "non sensitizer" in text:
        return "negative"
    if "sensitizer" in text:
        return "positive"
    if text in {"negative", "non-toxic", "non toxic", "nc"}:
        return "negative"
    if text in {"positive", "toxic", "s"}:
        return "positive"
    raise SystemExit(f"{label} has unsupported classification: {value!r}")


def _parse_husspred_response(raw: dict[str, object]) -> tuple[float, str]:
    if "probability_positive" in raw:
        prob = _probability(
            raw.get("probability_positive"),
            "HuSSPred probability_positive",
        )
        call = (
            _skin_sens_call(raw["call"], "HuSSPred call")
            if "call" in raw and raw.get("call") not in (None, "")
            else ("positive" if prob >= 0.5 else "negative")
        )
        return prob, call

    pred_data = raw.get("pred_data")
    if not isinstance(pred_data, list):
        raise SystemExit("HuSSPred response missing list field pred_data")
    binary_row = None
    for row in pred_data:
        if (
            isinstance(row, list)
            and row
            and isinstance(row[0], str)
            and "binary" in row[0].lower()
        ):
            binary_row = row
            break
    if binary_row is None or len(binary_row) < 3:
        raise SystemExit("HuSSPred response missing Binary Sensitization result")

    call = _call_from_classification(
        binary_row[1],
        "HuSSPred Binary Sensitization classification",
    )
    confidence = _probability(
        binary_row[2],
        "HuSSPred Binary Sensitization confidence",
    )
    prob = confidence if call == "positive" else 1.0 - confidence
    return prob, call


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-sdf", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument(
        "--allow-unavailable",
        action="store_true",
        help="Emit an explicit unavailable HuSSPred record instead of failing.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    _remove_outputs(args.out_json)
    smiles = smiles_from_sdf(args.in_sdf)
    try:
        import requests

        r = requests.post(
            API_URL,
            json={"smiles": smiles, "options": MODEL_OPTIONS},
            timeout=120,
        )
        r.raise_for_status()
        raw = r.json()
        if not isinstance(raw, dict):
            raise SystemExit("HuSSPred response must be a JSON object")
        prob, call = _parse_husspred_response(raw)
        status = "ok"
    except Exception as exc:  # noqa: BLE001
        if not args.allow_unavailable:
            raise SystemExit(
                "HuSSPred is required for Stage 2 skin-sens evidence; set "
                "admet.allow_skin_sens_unavailable=true for an explicit degraded run"
            ) from exc
        LOG.warning("HuSSPred unavailable: %s", exc)
        raw, prob, call, status = {}, None, None, "unavailable"

    _write_json_atomic({
        "smiles": smiles,
        "status": status,
        "skin_sens_probability": prob,
        "skin_sens_call": call,
        "degraded": status == "unavailable",
        "degraded_reason": "husspred_unavailable" if status == "unavailable" else "",
        "raw": raw,
    }, args.out_json)


if __name__ == "__main__":
    main()
