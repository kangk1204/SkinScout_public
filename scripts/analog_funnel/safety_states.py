#!/usr/bin/env python3
"""Separate availability, applicability, and decision states for the funnel.

A single PASS/FLAG_HIGH string conflated three different questions:

* availability — did the required skin-sens models run?
* applicability — was the compound inside each model's domain?
* decision — did the available evidence contain a risk signal?

A degraded run (a model missing, so the decision can only be FLAG_HIGH) is not
the same as a real risk positive, and neither is an applicability-limited
negative. The functions here keep the states distinct, make a degraded result
explicitly non-claimable, and preserve the exact reasons end-to-end. Diagnostic
docking may still proceed; it can never be promoted to a safety PASS.
"""

from __future__ import annotations

from typing import Iterable

SAFE_DECISIONS = {"PASS", "FLAG_HIGH"}
CLAIMABLE_DECISION = "PASS"

AVAILABILITY_AVAILABLE = "available"
AVAILABILITY_DEGRADED = "degraded_missing_models"
AVAILABILITY_UNAVAILABLE = "unavailable"

APPLICABILITY_WITHIN_DOMAIN = "within_domain"
APPLICABILITY_LIMITED = "limited"
APPLICABILITY_UNKNOWN = "unknown"


def _models(values: Iterable[object] | str | None) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        items = values.replace(",", ";").split(";")
    else:
        items = [str(value) for value in values]
    return sorted({item.strip() for item in items if item and item.strip()})


def _flag(value: object) -> bool:
    """CSV writes booleans as the strings "true"/"false"; "false" is truthy."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


def nested_skin_sens(payload: dict | None) -> dict:
    """The ``skin_sens`` object of an ``admet_report.json`` payload."""
    if not isinstance(payload, dict):
        return {}
    skin = payload.get("skin_sens")
    return skin if isinstance(skin, dict) else {}


def missing_models_from_report(payload: dict | None) -> list[str]:
    """Read missing models from the nested ``skin_sens`` object, not the root."""
    return _models(nested_skin_sens(payload).get("missing_models"))


def safety_state(
    decision: object,
    *,
    missing_models: Iterable[object] | str | None = None,
    applicability_limited_models: Iterable[object] | str | None = None,
    degraded: bool | None = None,
) -> dict:
    """Return the availability/applicability/decision state and exact reasons."""
    raw = str(decision).strip().upper() if decision is not None else ""
    missing = _models(missing_models)
    limited = _models(applicability_limited_models)
    if raw in {"", "UNKNOWN", "MISSING"}:
        raw = "UNKNOWN"
        availability = AVAILABILITY_UNAVAILABLE
    elif missing:
        availability = AVAILABILITY_DEGRADED
    else:
        availability = AVAILABILITY_AVAILABLE
    if _flag(degraded) and availability == AVAILABILITY_AVAILABLE:
        availability = AVAILABILITY_DEGRADED
    # A degraded or unavailable result must never be recorded as PASS.
    effective = raw
    if availability != AVAILABILITY_AVAILABLE and effective == CLAIMABLE_DECISION:
        effective = "FLAG_HIGH"
    if limited:
        applicability = APPLICABILITY_LIMITED
    elif availability == AVAILABILITY_UNAVAILABLE:
        applicability = APPLICABILITY_UNKNOWN
    else:
        applicability = APPLICABILITY_WITHIN_DOMAIN
    reasons: list[str] = []
    if effective != raw and raw != "UNKNOWN":
        reasons.append(f"degraded_not_claimable:{availability}")
    if raw == "HALT":
        reasons.append("decision:HALT")
    if raw == "UNKNOWN":
        reasons.append("decision:unknown")
    if raw == "FLAG_HIGH" and not reasons:
        reasons.append("decision:FLAG_HIGH")
    for model in missing:
        reasons.append(f"missing_models:{model}")
    for model in limited:
        reasons.append(f"applicability_limited:{model}")
    if not reasons and effective == CLAIMABLE_DECISION:
        reasons.append("decision:PASS")
    claimable = (
        effective == CLAIMABLE_DECISION
        and availability == AVAILABILITY_AVAILABLE
        and not limited
    )
    return {
        "decision": effective,
        "raw_decision": raw,
        "availability": availability,
        "applicability": applicability,
        "missing_models": missing,
        "applicability_limited_models": limited,
        "degraded": bool(missing) or _flag(degraded),
        "claimable": claimable,
        "docking_eligible": effective in SAFE_DECISIONS,
        "review_required": not claimable,
        "reasons": reasons,
        "reason": ";".join(reasons),
    }
