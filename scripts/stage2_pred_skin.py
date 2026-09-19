#!/usr/bin/env python3
"""stage2_pred_skin.py — Pred-Skin skin-sens call."""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

from rdkit import Chem

LOG = logging.getLogger("stage2.pred_skin")
PRED_SKIN_URL = "https://predskin.labmol.com.br/api/predskin/predict"
PRED_SKIN_TASK_URL = "https://predskin.labmol.com.br/api/predskin/tasks/{task_id}"
PRED_SKIN_MODE = "computational_only"
POLL_INTERVAL_SECONDS = 1.0
POLL_TIMEOUT_SECONDS = 180.0


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


def fetch(url: str, smiles: str) -> dict:
    import requests

    r = requests.post(
        url,
        json={
            "smiles": smiles,
            "mode": PRED_SKIN_MODE,
            "dpra_value": None,
            "usens_value": None,
            "dpra_type": "cysteine",
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


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


def _call_from_prediction(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"{label} must be a non-empty string")
    text = value.strip().lower()
    if text in {"nc", "non-sensitizer", "non sensitizer", "negative"}:
        return "negative"
    if text in {"s", "sensitizer", "positive"}:
        return "positive"
    raise SystemExit(f"{label} has unsupported prediction: {value!r}")


def _poll_pred_skin_task(task_id: str) -> dict[str, object]:
    import requests

    deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
    last_status = "QUEUED"
    while time.monotonic() < deadline:
        r = requests.get(PRED_SKIN_TASK_URL.format(task_id=task_id), timeout=30)
        r.raise_for_status()
        payload = r.json()
        if not isinstance(payload, dict):
            raise SystemExit("Pred-Skin task response must be a JSON object")
        status = str(payload.get("status", "")).upper()
        last_status = status or last_status
        if status == "SUCCESS":
            result = payload.get("result")
            if not isinstance(result, dict):
                raise SystemExit("Pred-Skin task response missing object field result")
            return result
        if status == "FAILURE":
            message = payload.get("message") or payload.get("error") or "unknown failure"
            raise SystemExit(f"Pred-Skin task failed: {message}")
        time.sleep(POLL_INTERVAL_SECONDS)
    raise SystemExit(f"Pred-Skin task did not finish before timeout; last_status={last_status}")


def _resolve_pred_skin_response(raw: dict[str, object]) -> dict[str, object]:
    task_id = raw.get("task_id")
    if isinstance(task_id, str) and task_id.strip():
        return _poll_pred_skin_task(task_id.strip())
    result = raw.get("result")
    if isinstance(result, dict):
        return result
    return raw


def _parse_pred_skin_response(raw: dict[str, object]) -> tuple[float, str, dict[str, object]]:
    result = _resolve_pred_skin_response(raw)
    if "probability" in result:
        prob = _probability(result.get("probability"), "pred_skin probability")
        call = (
            _skin_sens_call(result["call"], "pred_skin call")
            if "call" in result and result.get("call") not in (None, "")
            else ("positive" if prob >= 0.5 else "negative")
        )
        return prob, call, result

    individual = result.get("individual_predictions")
    if not isinstance(individual, dict):
        raise SystemExit("Pred-Skin response missing object field individual_predictions")
    student = individual.get("student")
    if not isinstance(student, dict):
        raise SystemExit("Pred-Skin response missing object field individual_predictions.student")
    prob = _probability(student.get("probability"), "pred_skin student probability")
    call = (
        _call_from_prediction(student["prediction"], "pred_skin student prediction")
        if "prediction" in student and student.get("prediction") not in (None, "")
        else ("positive" if prob >= 0.5 else "negative")
    )
    return prob, call, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-sdf", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument(
        "--allow-unavailable",
        action="store_true",
        help="Emit explicit unavailable endpoint records instead of failing.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    _remove_outputs(args.out_json)
    smiles = smiles_from_sdf(args.in_sdf)
    out: dict[str, object] = {"smiles": smiles}
    try:
        raw = fetch(PRED_SKIN_URL, smiles)
        if not isinstance(raw, dict):
            raise SystemExit("pred_skin response must be a JSON object")
        prob, call, result = _parse_pred_skin_response(raw)
        out["pred_skin"] = {
            "raw": result,
            "probability": prob,
            "call": call,
        }
        out["consensus_call"] = call
        out["status"] = "ok"
        out["degraded"] = False
        out["degraded_reason"] = ""
    except Exception as exc:  # noqa: BLE001
        if not args.allow_unavailable:
            raise SystemExit(
                "Pred-Skin is required for Stage 2 skin-sens evidence; set "
                "admet.allow_skin_sens_unavailable=true for an explicit degraded run"
            ) from exc
        LOG.warning("Pred-Skin unavailable: %s", exc)
        out["pred_skin"] = {"status": "unavailable"}
        out["consensus_call"] = None
        out["status"] = "unavailable"
        out["degraded"] = True
        out["degraded_reason"] = "pred_skin_unavailable"

    _write_json_atomic(out, args.out_json)


if __name__ == "__main__":
    main()
