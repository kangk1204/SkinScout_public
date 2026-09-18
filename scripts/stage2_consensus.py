#!/usr/bin/env python3
"""stage2_consensus.py — Combine ADMET / structural-alert results into a single
report and emit the skin-sens HALT/FLAG_HIGH/PASS decision.

Decision rule (INSTRUCTIONS.md §5):
    ≥ 2 of 3 positive  →  HALT
    1   of 3 positive  →  FLAG_HIGH (sign-off needed)
    0   of 3 positive  →  PASS
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

LOG = logging.getLogger("stage2.consensus")
VALID_SKIN_SENS_CALLS = {"positive", "negative"}


def _remove_outputs(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def _nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _read_json_input(path: Path, label: str) -> dict[str, object]:
    if not _nonempty(path):
        raise SystemExit(
            f"Stage 2 consensus input is missing or empty: {label}={path}"
        )
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"Stage 2 consensus input is invalid JSON: {label}={path}"
        ) from exc
    if not isinstance(payload, dict):
        raise SystemExit(
            f"Stage 2 consensus input must be a JSON object: {label}={path}"
        )
    return payload


def _write_json_atomic(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def _write_text_atomic(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _normalize_call(value: object, label: str, field: str) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str) or value not in VALID_SKIN_SENS_CALLS:
        raise SystemExit(
            f"Stage 2 consensus input has invalid skin-sens call: "
            f"{label}.{field}={value!r}; expected positive or negative"
        )
    return value


def _required_smiles(payload: dict[str, object], label: str) -> str:
    value = payload.get("smiles")
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(
            f"Stage 2 consensus input missing non-empty smiles: {label}"
        )
    return value.strip()


def _validated_consensus_smiles(payloads: dict[str, dict[str, object]]) -> str:
    smiles_by_label = {
        label: _required_smiles(payload, label)
        for label, payload in payloads.items()
    }
    unique = sorted(set(smiles_by_label.values()))
    if len(unique) > 1:
        details = ", ".join(
            f"{label}={smiles}" for label, smiles in sorted(smiles_by_label.items())
        )
        raise SystemExit(f"Stage 2 consensus input smiles mismatch: {details}")
    return unique[0]


def _call(
    payload: dict,
    label: str,
    key_call: str = "skin_sens_call",
    key_prob: str = "skin_sens_probability",
    prob_threshold: float = 0.5,
) -> str | None:
    """Extract a positive/negative call from an arbitrary model payload."""
    if not isinstance(payload, dict):
        return None
    # "unavailable"만 제외하면 status="error" 같은 비정상 산출물이 남아 있는
    # call/probability로 투표할 수 있다(fail-open). 정상("ok")일 때만 인정한다.
    if payload.get("status") != "ok":
        return None
    if "consensus_call" in payload:
        return _normalize_call(payload["consensus_call"], label, "consensus_call")
    call = payload.get(key_call)
    normalized_call = _normalize_call(call, label, key_call)
    if normalized_call is not None:
        return normalized_call
    prob = payload.get(key_prob)
    if prob is not None:
        if isinstance(prob, bool) or not isinstance(prob, (int, float)):
            raise SystemExit(
                f"Stage 2 consensus input has invalid skin-sens probability: "
                f"{label}.{key_prob}={prob!r}; expected numeric 0..1"
            )
        if not 0 <= prob <= 1:
            raise SystemExit(
                f"Stage 2 consensus input has out-of-range skin-sens probability: "
                f"{label}.{key_prob}={prob!r}; expected numeric 0..1"
            )
        return "positive" if prob >= prob_threshold else "negative"
    return None


def unavailable_skin_sens_models(payloads: dict[str, dict]) -> list[str]:
    """Return configured skin-sens model payloads that lack usable evidence."""
    missing = []
    for name in ("husspred", "stoptox", "pred_skin"):
        if _call(payloads.get(name, {}), name) is None:
            missing.append(name)
    return missing


def skin_sens_decision(calls: list[str | None], halt_min_votes: int = 2) -> str:
    """Pure function; unit-testable.

    Args:
        calls: ordered list of {'positive','negative',None} from the configured
            skin-sens models.
        halt_min_votes: minimum number of positive calls required to HALT.

    Returns:
        'HALT' | 'FLAG_HIGH' | 'PASS'
    """
    if not 1 <= halt_min_votes <= 3:
        raise ValueError("halt_min_votes must be between 1 and 3")
    pos = sum(1 for c in calls if c == "positive")
    informative = sum(1 for c in calls if c in ("positive", "negative"))
    if informative == 0:
        return "FLAG_HIGH"  # cannot assess → human review
    if pos >= halt_min_votes:
        return "HALT"
    if pos >= 1:
        return "FLAG_HIGH"
    return "PASS"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admet-ai", required=True, type=Path)
    parser.add_argument("--stoptox", required=True, type=Path)
    parser.add_argument("--husspred", required=True, type=Path)
    parser.add_argument("--pred-skin", required=True, type=Path)
    parser.add_argument("--alerts", required=True, type=Path)
    parser.add_argument("--halt-min-votes", type=int, default=2)
    parser.add_argument("--allow-unavailable-models", action="store_true",
                        help="Allow missing skin-sens model evidence as degraded output.")
    parser.add_argument("--out-report", required=True, type=Path)
    parser.add_argument("--out-decision", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    _remove_outputs(args.out_report, args.out_decision)
    if not 1 <= args.halt_min_votes <= 3:
        raise SystemExit("--halt-min-votes must be between 1 and 3")
    payloads = {
        "admet_ai":  _read_json_input(args.admet_ai, "admet_ai"),
        "stoptox":   _read_json_input(args.stoptox, "stoptox"),
        "husspred":  _read_json_input(args.husspred, "husspred"),
        "pred_skin": _read_json_input(args.pred_skin, "pred_skin"),
        "alerts":    _read_json_input(args.alerts, "alerts"),
    }
    smiles = _validated_consensus_smiles(payloads)

    calls = [
        _call(payloads["husspred"], "husspred"),
        _call(payloads["stoptox"], "stoptox"),
        _call(payloads["pred_skin"], "pred_skin"),
    ]
    missing_models = unavailable_skin_sens_models(payloads)
    if missing_models and not args.allow_unavailable_models:
        raise SystemExit(
            "Missing Stage 2 skin-sens model evidence: "
            + ",".join(missing_models)
            + "; set admet.allow_skin_sens_unavailable=true for an explicit degraded run"
        )
    decision = skin_sens_decision(calls, halt_min_votes=args.halt_min_votes)

    report = {
        "smiles": smiles,
        "skin_sens": {
            "husspred":  calls[0],
            "stoptox":   calls[1],
            "pred_skin": calls[2],
            "decision":  decision,
            "degraded": bool(missing_models),
            "missing_models": missing_models,
        },
        "structural_alerts": payloads["alerts"],
        "admet_ai_predictions": payloads["admet_ai"].get("predictions"),
    }

    _write_json_atomic(report, args.out_report)
    _write_text_atomic(decision + "\n", args.out_decision)
    LOG.info("Skin-sens decision: %s  (calls=%s)", decision, calls)


if __name__ == "__main__":
    main()
