#!/usr/bin/env python3
"""stage2_stoptox.py — STopTox 6-endpoint acute toxicity (incl. skin sens).

Wraps the StopTox CLI if installed; otherwise falls back to the public REST API.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

from rdkit import Chem

LOG = logging.getLogger("stage2.stoptox")
API_URL = "https://stoptox.mml.unc.edu/api/predict"
WEB_URL = "https://stoptox.mml.unc.edu/predict"
ENDPOINTS = ("skin_sens", "eye_irritation", "skin_irritation",
             "acute_oral", "acute_dermal", "acute_inhalation")


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


def via_cli(smiles: str) -> dict | None:
    if not shutil.which("stoptox"):
        return None
    res = subprocess.run(
        ["stoptox", "--smiles", smiles, "--json"],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        return None
    return json.loads(res.stdout)


def via_api(smiles: str) -> dict:
    import requests

    payload = {"smiles": smiles, "endpoints": list(ENDPOINTS)}
    try:
        r = requests.post(API_URL, json=payload, timeout=120)
        r.raise_for_status()
        return r.json()
    except Exception:
        return via_web(smiles)


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


def _skin_sens_result(result: object) -> tuple[float | None, str | None]:
    if not isinstance(result, dict):
        raise SystemExit("STopTox response must be a JSON object")
    skin = result.get("skin_sens")
    if not isinstance(skin, dict):
        raise SystemExit("STopTox response missing object field skin_sens")
    return (
        _probability(
            skin.get("probability"),
            "STopTox skin_sens.probability",
        ),
        _skin_sens_call(skin.get("call"), "STopTox skin_sens.call"),
    )


class _TableTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            text = data.strip()
            if text:
                self._cell.append(text)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._row is not None and self._cell is not None:
            self._row.append(" ".join(self._cell))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None


def _confidence_from_percent(text: str) -> float:
    cleaned = text.strip().rstrip("%")
    return _probability(float(cleaned) / 100.0, "STopTox confidence")


def _call_from_prediction(text: str) -> str:
    normalized = text.strip().lower()
    if (
        "non-toxic" in normalized
        or "non toxic" in normalized
        or "non-sensitizer" in normalized
        or "non sensitizer" in normalized
    ):
        return "negative"
    if "sensitizer" in normalized:
        return "positive"
    if "toxic" in normalized:
        return "positive"
    raise SystemExit(f"STopTox skin sensitization prediction is unsupported: {text!r}")


def _parse_stoptox_html(html: str) -> dict[str, object]:
    parser = _TableTextParser()
    parser.feed(html)
    for row in parser.rows:
        if not row or "skin sensitization" not in row[0].lower():
            continue
        if len(row) < 3:
            raise SystemExit("STopTox skin sensitization row is incomplete")
        confidence = _confidence_from_percent(row[2])
        call = _call_from_prediction(row[1])
        positive_probability = confidence if call == "positive" else 1.0 - confidence
        return {
            "skin_sens": {
                "probability": positive_probability,
                "call": call,
                "confidence": confidence,
                "source": "stoptox_web_html",
            }
        }
    raise SystemExit("STopTox HTML response missing Skin Sensitization row")


def via_web(smiles: str) -> dict[str, object]:
    import requests

    r = requests.get(WEB_URL, params={"smiles": smiles}, timeout=120)
    r.raise_for_status()
    return _parse_stoptox_html(r.text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-sdf", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument(
        "--allow-unavailable",
        action="store_true",
        help="Emit an explicit unavailable STopTox record instead of failing.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    _remove_outputs(args.out_json)
    smiles = smiles_from_sdf(args.in_sdf)
    try:
        result = via_cli(smiles) or via_api(smiles)
        status = "ok"
    except Exception as exc:  # noqa: BLE001
        if not args.allow_unavailable:
            raise SystemExit(
                "STopTox is required for Stage 2 skin-sens evidence; set "
                "admet.allow_skin_sens_unavailable=true for an explicit degraded run"
            ) from exc
        LOG.warning("StopTox unavailable: %s", exc)
        result, status = {}, "unavailable"

    if status == "ok":
        skin_probability, skin_call = _skin_sens_result(result)
    else:
        skin_probability, skin_call = None, None
    _write_json_atomic({
        "smiles": smiles,
        "status": status,
        "degraded": status == "unavailable",
        "degraded_reason": "stoptox_unavailable" if status == "unavailable" else "",
        "raw": result,
        "skin_sens_probability": skin_probability,
        "skin_sens_call": skin_call,
    }, args.out_json)
    LOG.info("STopTox → %s status=%s", args.out_json, status)


if __name__ == "__main__":
    main()
