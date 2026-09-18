#!/usr/bin/env python3
"""stage2_5_decision.py — Cosmetic / Drug-avoidance gate decision.

Drug policy options (INSTRUCTIONS.md §6.2):
    strict   — STRICT_WARNING or SCAFFOLD_MATCH → HALT
    moderate — STRICT_WARNING → DOWNWEIGHT, else PROCEED (default)
    lenient  — everything PROCEED (warnings labeled only)

The CosIng level (EXACT / SIMILAR / ANALOG / NEW) does NOT halt the run, but it
flips the result framing: EXACT/SIMILAR matches are repositioning candidates
("new mode of action for the existing INCI X").
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from cosmetic_drug_contract import (
    CosmeticDrugContractError,
    expected_decision_from_drug_warnings,
)

LOG = logging.getLogger("stage2_5.decision")


def _remove_outputs(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def _write_text_atomic(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def decide(cosing: dict, drug: dict, policy: str) -> str:
    """Pure function — unit tested.

    Args:
        cosing: payload from stage2_5_cosing_match (with key 'level').
        drug:   payload from stage2_5_drug_avoidance (with key 'warnings').
        policy: 'strict' | 'moderate' | 'lenient'.

    Returns:
        'HALT' | 'DOWNWEIGHT' | 'PROCEED'
    """
    try:
        return expected_decision_from_drug_warnings(drug, policy)
    except CosmeticDrugContractError as exc:
        raise ValueError(str(exc)) from exc


def framing(cosing: dict) -> str:
    level = cosing.get("level", "NEW")
    inci = cosing.get("inci") or ""
    if level == "EXACT":
        return f"existing cosmetic ingredient (INCI={inci}); reposition / new MoA framing"
    if level == "SIMILAR":
        return f"very similar to INCI={inci} (Tanimoto>=0.85)"
    if level == "ANALOG":
        return f"analog of INCI={inci} (Tanimoto>=0.65)"
    return "novel chemical space"


def degraded_references(cosing: dict, drug: dict) -> list[str]:
    degraded: list[str] = []
    for label, payload in (("CosIng", cosing), ("drug", drug)):
        # 필드가 없으면 참조가 정상이었다고 가정하지 않는다(fail-open 방지).
        status = payload.get("reference_status", "missing")
        if status != "ok":
            degraded.append(f"{label} reference_status={status}")
    return degraded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cosing-json", required=True, type=Path)
    parser.add_argument("--drug-json", required=True, type=Path)
    parser.add_argument("--drug-policy", default="moderate",
                        choices=("strict", "moderate", "lenient"))
    parser.add_argument(
        "--allow-degraded-references",
        action="store_true",
        help="Permit degraded reference payloads only for explicit diagnostics.",
    )
    parser.add_argument("--out-decision", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    _remove_outputs(args.out_decision)
    cosing = json.loads(args.cosing_json.read_text())
    drug = json.loads(args.drug_json.read_text())
    degraded = degraded_references(cosing, drug)
    if degraded and not args.allow_degraded_references:
        raise SystemExit(
            "Stage 2.5 decision requires full references; degraded payload(s): "
            + "; ".join(degraded)
            + ". Use --allow-degraded-references only for explicit diagnostics."
        )
    try:
        decision = decide(cosing, drug, args.drug_policy)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    _write_text_atomic(
        f"{decision}\n# framing: {framing(cosing)}\n"
        f"# policy: {args.drug_policy}\n",
        args.out_decision,
    )
    LOG.info("Decision: %s (framing: %s)", decision, framing(cosing))


if __name__ == "__main__":
    main()
