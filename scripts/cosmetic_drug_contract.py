"""Shared contract for the SkinScout cosmetic/drug decision gate."""

from __future__ import annotations

from typing import Any

COSMETIC_DRUG_DECISIONS = {"HALT", "DOWNWEIGHT", "PROCEED"}
DRUG_POLICIES = {"strict", "moderate", "lenient"}
POLICY_PREFIX = "# policy:"


class CosmeticDrugContractError(ValueError):
    """Raised when cosmetic/drug artifacts disagree with the stage 2.5 contract."""


def parse_decision_and_policy(
    text: str,
    *,
    label: str = "cosmetic/drug decision",
) -> tuple[str, str]:
    """Return the first non-comment decision line and required policy metadata."""
    decision: str | None = None
    policy: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if decision is None and not line.startswith("#"):
            decision = line
        if line.lower().startswith(POLICY_PREFIX):
            policy = line.split(":", 1)[1].strip()

    if decision is None:
        raise CosmeticDrugContractError(f"{label} contains no decision line")
    if decision not in COSMETIC_DRUG_DECISIONS:
        raise CosmeticDrugContractError(
            f"{label} must be one of {sorted(COSMETIC_DRUG_DECISIONS)}: {decision}"
        )
    if policy is None:
        raise CosmeticDrugContractError(f"{label} missing policy line")
    if policy not in DRUG_POLICIES:
        raise CosmeticDrugContractError(
            f"{label} policy must be one of {sorted(DRUG_POLICIES)}: {policy}"
        )
    return decision, policy


def expected_decision_from_drug_warnings(
    drug_warnings: dict[str, Any],
    policy: str,
) -> str:
    """Recompute the stage 2.5 cosmetic/drug decision from warning tiers."""
    if policy not in DRUG_POLICIES:
        raise CosmeticDrugContractError(
            f"drug policy must be one of {sorted(DRUG_POLICIES)}: {policy}"
        )

    warnings = drug_warnings.get("warnings")
    if not isinstance(warnings, list):
        raise CosmeticDrugContractError(
            "drug-avoidance warnings field 'warnings' must be a list"
        )

    tiers: set[str] = set()
    for idx, warning in enumerate(warnings):
        if not isinstance(warning, dict):
            raise CosmeticDrugContractError(
                f"drug-avoidance warnings entry {idx} must be an object"
            )
        tier = warning.get("tier")
        if isinstance(tier, str) and tier.strip():
            tiers.add(tier.strip())

    if policy == "strict":
        if "STRICT_WARNING" in tiers or "SCAFFOLD_MATCH" in tiers:
            return "HALT"
        if "SOFT_WARNING" in tiers:
            return "DOWNWEIGHT"
        return "PROCEED"
    if policy == "moderate":
        if "STRICT_WARNING" in tiers:
            return "DOWNWEIGHT"
        return "PROCEED"
    return "PROCEED"


def validate_decision_matches_drug_warnings(
    *,
    decision: str,
    policy: str,
    drug_warnings: dict[str, Any],
) -> str:
    """Validate a decision artifact against source drug warnings."""
    expected = expected_decision_from_drug_warnings(drug_warnings, policy)
    if decision != expected:
        raise CosmeticDrugContractError(
            f"cosmetic/drug decision {decision} does not match expected "
            f"{expected} for policy {policy}"
        )
    return expected
