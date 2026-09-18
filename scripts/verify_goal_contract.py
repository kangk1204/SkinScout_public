#!/usr/bin/env python3
"""Verify that a completed SkinScout run satisfies the active goal contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from run_artifact_contract import (
    CONTRACT_CURRENT,
    CONTRACT_LEGACY,
    FAST_BAND_VERIFIED_ARTIFACTS,
    REQUIRED_VERIFIED_ARTIFACTS,
    UnsupportedRunManifestError,
    current_fast_screening_keys,
    legacy_fast_screening_keys,
    required_screening_keys,
    required_verified_artifact_paths,
    resolve_artifact_contract,
    screening_funnel_edges,
)


SUMMARY_SCHEMA = "skinscout.run_summary.v1"
RUN_VERIFICATION_SCHEMA = "skinscout.run_verification.v1"
GOAL_CONTRACT_SCHEMA = "skinscout.goal_contract.v1"
GOAL_CONTRACT_VERIFICATION_SCHEMA = "skinscout.goal_contract_verification.v2"
GOAL_CONTRACT_VERIFICATION_FILENAME = "goal_contract_verification.json"
RUN_MANIFEST_SCHEMA_V2 = "skinscout.run_manifest.v2"
RUN_MANIFEST_SCHEMA_V3 = "skinscout.run_manifest.v3"
TARGET_PRESETS = {"target-id", "report"}
OVERALL_DECISIONS = {"HALT", "FLAG_HIGH", "PASS"}
RECOMMENDED_ACTIONS = {
    "HALT": "stop_before_claim",
    "FLAG_HIGH": "review_before_claim",
    "PASS": "proceed",
}
SKIN_TOXICITY_DECISIONS = {"HALT", "PASS", "REVIEW"}
SKIN_TOXICITY_LEVELS = {"high", "low", "moderate"}
SKIN_SENS_MODELS = {"husspred", "stoptox", "pred_skin"}
SKIN_CONTEXT_DECISIONS = {
    "skin_context_supported",
    "skin_expression_only",
    "skin_efficacy_literature_only",
    "insufficient_skin_context",
}
ADMET_METRICS = (
    "AMES",
    "ClinTox",
    "DILI",
    "Skin_Reaction",
    "hERG",
    "LD50_Zhu",
    "Solubility_AqSolDB",
    "logP",
    "QED",
    "tpsa",
)
# Inclusive [min, max] bounds per endpoint. Probability endpoints and QED are
# fractions; the remaining bounds are deliberately wide sanity ranges around the
# observed ADMET-AI outputs, so a NaN/Infinity or a sign flip cannot pass as a
# valid measurement.
ADMET_METRIC_RANGES: dict[str, tuple[float, float]] = {
    "AMES": (0.0, 1.0),
    "ClinTox": (0.0, 1.0),
    "DILI": (0.0, 1.0),
    "Skin_Reaction": (0.0, 1.0),
    "hERG": (0.0, 1.0),
    "LD50_Zhu": (-10.0, 10.0),
    "Solubility_AqSolDB": (-15.0, 2.0),
    "logP": (-10.0, 15.0),
    "QED": (0.0, 1.0),
    "tpsa": (0.0, 1000.0),
}
# Required artifact names and screening keys are derived from the shared
# contract module so the verifier and the summary writer cannot drift apart.
FAST_REQUIRED_SCREENING_KEYS = set(current_fast_screening_keys())
LEGACY_FAST_REQUIRED_SCREENING_KEYS = set(legacy_fast_screening_keys())
COMPREHENSIVE_REQUIRED_SCREENING_KEYS = set(
    required_screening_keys("comprehensive", CONTRACT_CURRENT)
)


@dataclass(frozen=True)
class GoalCheck:
    name: str
    status: str
    detail: str = ""


def _ok(name: str, detail: str = "") -> GoalCheck:
    return GoalCheck(name=name, status="ok", detail=detail)


def _failed(name: str, detail: str) -> GoalCheck:
    return GoalCheck(name=name, status="failed", detail=detail)


def _read_json(path: Path, name: str) -> tuple[dict[str, Any], GoalCheck]:
    if not path.exists():
        return {}, _failed(name, f"missing: {path}")
    if path.is_symlink() or not path.is_file():
        return {}, _failed(name, f"expected regular non-symlink file: {path}")
    if path.stat().st_size == 0:
        return {}, _failed(name, f"empty file: {path}")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return {}, _failed(name, f"invalid JSON: {exc}")
    if not isinstance(payload, dict):
        return {}, _failed(name, "expected JSON object")
    return payload, _ok(name, f"bytes={path.stat().st_size}")


def _is_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        numeric = float(value)
    except OverflowError:
        return False
    return math.isfinite(numeric)


def _is_probability(value: Any) -> bool:
    return _is_number(value) and 0.0 <= float(value) <= 1.0


def _is_metric_in_range(metric: str, value: Any) -> bool:
    if not _is_number(value):
        return False
    low, high = ADMET_METRIC_RANGES[metric]
    return low <= float(value) <= high


def _is_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def _missing_keys(payload: dict[str, Any], keys: list[str] | tuple[str, ...]) -> list[str]:
    return [key for key in keys if key not in payload]


def _check_summary_schema(summary: dict[str, Any]) -> GoalCheck:
    schema = summary.get("schema_version")
    if schema != SUMMARY_SCHEMA:
        return _failed("run summary schema", f"expected {SUMMARY_SCHEMA}, got {schema!r}")
    preset = summary.get("preset")
    if preset not in TARGET_PRESETS:
        return _failed(
            "target prediction preset",
            f"expected one of {sorted(TARGET_PRESETS)}, got {preset!r}",
        )
    return _ok("run summary schema", f"preset={preset} mode={summary.get('mode')}")


def _check_smiles_input(summary: dict[str, Any], expected_smiles: str | None) -> GoalCheck:
    compound = summary.get("compound")
    if not isinstance(compound, dict):
        return _failed("single SMILES input", "missing compound object")
    required = ("input_type", "input_smiles", "input_canonical_smiles", "canonical_smiles", "inchikey")
    missing = _missing_keys(compound, required)
    if missing:
        return _failed("single SMILES input", f"missing compound fields: {', '.join(missing)}")
    if compound.get("input_type") != "smiles":
        return _failed("single SMILES input", f"input_type is {compound.get('input_type')!r}")
    for key in ("input_smiles", "input_canonical_smiles", "canonical_smiles", "inchikey"):
        if not _is_string(compound.get(key)):
            return _failed("single SMILES input", f"compound.{key} must be non-empty")
    if expected_smiles is not None and compound.get("input_smiles") != expected_smiles:
        return _failed(
            "single SMILES input",
            f"expected input_smiles {expected_smiles!r}, got {compound.get('input_smiles')!r}",
        )
    return _ok(
        "single SMILES input",
        f"{compound.get('input_smiles')} -> {compound.get('canonical_smiles')}",
    )


def _check_overall_decision(summary: dict[str, Any]) -> GoalCheck:
    overall = summary.get("overall_decision")
    if not isinstance(overall, dict):
        return _failed("overall decision", "missing overall_decision object")
    decision = overall.get("decision")
    if decision not in OVERALL_DECISIONS:
        return _failed("overall decision", f"invalid decision {decision!r}")
    expected_action = RECOMMENDED_ACTIONS[decision]
    if overall.get("recommended_action") != expected_action:
        return _failed(
            "overall decision",
            f"expected recommended_action {expected_action!r}, got {overall.get('recommended_action')!r}",
        )
    if not _is_bool(overall.get("requires_human_review")):
        return _failed("overall decision", "requires_human_review must be boolean")
    if not _is_bool(overall.get("claimable")):
        return _failed("overall decision", "claimable must be boolean")
    if decision != "PASS":
        reasons = overall.get("reasons")
        if not isinstance(reasons, list) or not all(_is_string(reason) for reason in reasons):
            return _failed("overall decision", "non-PASS decisions require non-empty reasons")
    return _ok(
        "overall decision",
        f"decision={decision} action={expected_action} claimable={overall.get('claimable')}",
    )


def _check_admet_and_skin_sens(summary: dict[str, Any], *, allow_degraded: bool) -> GoalCheck:
    safety = summary.get("safety")
    if not isinstance(safety, dict):
        return _failed("ADMET and skin-sens evidence", "missing safety object")
    metrics = safety.get("admet_metrics")
    if not isinstance(metrics, dict):
        return _failed("ADMET and skin-sens evidence", "missing admet_metrics object")
    missing_metrics = [
        metric
        for metric in ADMET_METRICS
        if not _is_metric_in_range(metric, metrics.get(metric))
    ]
    if missing_metrics:
        return _failed(
            "ADMET and skin-sens evidence",
            f"missing, non-finite, or out-of-range ADMET metrics: {', '.join(missing_metrics)}",
        )
    risk = safety.get("admet_risk_assessment")
    if not isinstance(risk, dict):
        return _failed("ADMET and skin-sens evidence", "missing admet_risk_assessment object")
    for key in ("high_risk_endpoints", "moderate_risk_endpoints"):
        if not isinstance(risk.get(key), list):
            return _failed("ADMET and skin-sens evidence", f"{key} must be a list")
    evidence = safety.get("skin_sens_evidence")
    if not isinstance(evidence, list):
        return _failed("ADMET and skin-sens evidence", "skin_sens_evidence must be a list")
    by_model = {
        item.get("model"): item
        for item in evidence
        if isinstance(item, dict) and _is_string(item.get("model"))
    }
    missing_models = sorted(SKIN_SENS_MODELS - set(by_model))
    if missing_models:
        return _failed(
            "ADMET and skin-sens evidence",
            f"missing skin-sens model evidence: {', '.join(missing_models)}",
        )
    for model in sorted(SKIN_SENS_MODELS):
        item = by_model[model]
        if item.get("status") != "ok":
            return _failed("ADMET and skin-sens evidence", f"{model} status is {item.get('status')!r}")
        if item.get("call") not in {"positive", "negative"}:
            return _failed("ADMET and skin-sens evidence", f"{model} call is invalid")
        if not _is_number(item.get("probability")):
            return _failed("ADMET and skin-sens evidence", f"{model} probability must be finite numeric")
        if not _is_probability(item.get("probability")):
            return _failed("ADMET and skin-sens evidence", f"{model} probability must be within [0, 1]")
    if safety.get("degraded") and not allow_degraded:
        return _failed("ADMET and skin-sens evidence", "degraded safety evidence is not allowed")
    missing = safety.get("missing_models")
    if missing not in (None, []):
        return _failed("ADMET and skin-sens evidence", f"missing skin-sens models: {missing}")
    return _ok("ADMET and skin-sens evidence", f"metrics={len(ADMET_METRICS)} models=3")


def _check_skin_toxicity(summary: dict[str, Any], *, allow_degraded: bool) -> GoalCheck:
    skin_toxicity = summary.get("skin_toxicity")
    if not isinstance(skin_toxicity, dict):
        return _failed("skin toxicity prediction", "missing skin_toxicity object")
    if skin_toxicity.get("decision") not in SKIN_TOXICITY_DECISIONS:
        return _failed("skin toxicity prediction", f"invalid decision {skin_toxicity.get('decision')!r}")
    if skin_toxicity.get("toxicity_level") not in SKIN_TOXICITY_LEVELS:
        return _failed(
            "skin toxicity prediction",
            f"invalid toxicity_level {skin_toxicity.get('toxicity_level')!r}",
        )
    required = (
        "skin_sens_decision",
        "skin_reaction_risk_level",
        "skin_reaction_value",
        "structural_alerts_present",
        "structural_alert_flags",
        "degraded",
        "missing_models",
        "reasons",
    )
    missing = _missing_keys(skin_toxicity, required)
    if missing:
        return _failed("skin toxicity prediction", f"missing fields: {', '.join(missing)}")
    if not _is_number(skin_toxicity.get("skin_reaction_value")):
        return _failed("skin toxicity prediction", "skin_reaction_value must be numeric")
    if not _is_bool(skin_toxicity.get("structural_alerts_present")):
        return _failed("skin toxicity prediction", "structural_alerts_present must be boolean")
    if not isinstance(skin_toxicity.get("structural_alert_flags"), list):
        return _failed("skin toxicity prediction", "structural_alert_flags must be a list")
    if skin_toxicity.get("degraded") and not allow_degraded:
        return _failed("skin toxicity prediction", "degraded skin-toxicity evidence is not allowed")
    if skin_toxicity.get("missing_models"):
        return _failed("skin toxicity prediction", f"missing skin-sens models: {skin_toxicity.get('missing_models')}")
    reasons = skin_toxicity.get("reasons")
    if not isinstance(reasons, list) or not all(_is_string(reason) for reason in reasons):
        return _failed("skin toxicity prediction", "reasons must be non-empty strings")
    return _ok(
        "skin toxicity prediction",
        f"decision={skin_toxicity.get('decision')} level={skin_toxicity.get('toxicity_level')}",
    )


def _check_target_prediction(
    summary: dict[str, Any],
    *,
    min_targets: int,
    min_target_sources: int,
) -> GoalCheck:
    target_prediction = summary.get("target_prediction")
    if not isinstance(target_prediction, dict):
        return _failed("protein target prediction", "missing target_prediction object")
    n_targets = target_prediction.get("n_targets")
    if not isinstance(n_targets, int) or n_targets < min_targets:
        return _failed("protein target prediction", f"n_targets {n_targets!r} is below {min_targets}")
    screened = target_prediction.get("screened_target_count")
    if not isinstance(screened, int) or isinstance(screened, bool) or screened <= n_targets:
        return _failed("protein target prediction", "screened_target_count must be > n_targets")
    top_targets = target_prediction.get("top_targets")
    if not isinstance(top_targets, list) or not top_targets:
        return _failed("protein target prediction", "top_targets must be non-empty")
    top = top_targets[0]
    if not isinstance(top, dict):
        return _failed("protein target prediction", "top target must be an object")
    for key in ("target_id", "gene_symbol", "protein_name"):
        if not _is_string(top.get(key)):
            return _failed("protein target prediction", f"top target {key} must be non-empty")
    for key in ("final_score", "docking_rrf"):
        if not _is_number(top.get(key)):
            return _failed("protein target prediction", f"top target {key} must be numeric")
    sources = top.get("sources")
    if not isinstance(sources, list) or len(sources) < min_target_sources:
        return _failed(
            "protein target prediction",
            f"top target must have >= {min_target_sources} source labels",
        )
    source_count = top.get("source_count")
    if not isinstance(source_count, int) or source_count < min_target_sources:
        return _failed(
            "protein target prediction",
            f"top target source_count must be >= {min_target_sources}",
        )
    if source_count != len(sources):
        return _failed("protein target prediction", "top target source_count/sources mismatch")
    return _ok("protein target prediction", f"top={top.get('gene_symbol')} targets={n_targets} screened={screened}")


def _check_screening_counts(summary: dict[str, Any], *, contract: str) -> GoalCheck:
    target_prediction = summary.get("target_prediction")
    if not isinstance(target_prediction, dict):
        return _failed("target screening evidence", "missing target_prediction object")
    counts = target_prediction.get("screening_counts")
    if not isinstance(counts, dict) or not counts:
        return _failed("target screening evidence", "screening_counts must be a non-empty object")
    mode = summary.get("mode")
    if mode not in {"fast", "comprehensive", "both"}:
        return _failed("target screening evidence", "summary mode must be fast, comprehensive, or both")
    required_keys = required_screening_keys(mode, contract)
    edges = screening_funnel_edges(mode, contract)
    missing_keys = sorted(required_keys - set(counts))
    if missing_keys:
        return _failed(
            "target screening evidence",
            "screening_counts missing required stage(s): " + ", ".join(missing_keys),
        )
    bad = [
        key
        for key, value in counts.items()
        if not isinstance(value, int) or isinstance(value, bool) or value < 1
    ]
    if bad:
        return _failed("target screening evidence", f"invalid positive integer counts: {', '.join(bad)}")
    violations = [
        f"{upstream}={counts[upstream]} < {downstream}={counts[downstream]}"
        for upstream, downstream in edges
        if upstream in counts and downstream in counts and counts[upstream] < counts[downstream]
    ]
    if violations:
        return _failed("target screening evidence", "non-monotonic screening funnel: " + "; ".join(violations))
    screened = target_prediction.get("screened_target_count")
    if isinstance(screened, int) and not isinstance(screened, bool):
        expected_screened = max(counts.values())
        if screened != expected_screened:
            return _failed(
                "target screening evidence",
                f"screened_target_count {screened} must equal max screening count {expected_screened}",
            )
        final_ranked = counts.get("skin_weighted_ranked_targets")
        if isinstance(final_ranked, int) and expected_screened <= final_ranked:
            return _failed(
                "target screening evidence",
                "screening funnel must include at least one upstream count above final ranked targets",
            )
    return _ok("target screening evidence", ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))


def _check_skin_binding(
    summary: dict[str, Any],
    *,
    min_target_sources: int,
    require_skin_context_supported: bool,
) -> GoalCheck:
    binding = summary.get("skin_specialized_binding")
    if not isinstance(binding, dict):
        return _failed("skin-specialized binding context", "missing skin_specialized_binding object")
    decision = binding.get("skin_context_decision")
    if decision not in SKIN_CONTEXT_DECISIONS:
        return _failed("skin-specialized binding context", f"invalid skin_context_decision {decision!r}")
    bool_fields = (
        "skin_context_supported",
        "skin_expression_supported",
        "skin_efficacy_supported",
        "top_target_skin_expression_supported",
        "top_target_skin_efficacy_supported",
        "top_target_skin_context_supported",
    )
    for field in bool_fields:
        if not _is_bool(binding.get(field)):
            return _failed("skin-specialized binding context", f"{field} must be boolean")
    if require_skin_context_supported and not binding.get("skin_context_supported"):
        return _failed("skin-specialized binding context", "skin_context_supported is required")
    if require_skin_context_supported and not binding.get("top_target_skin_context_supported"):
        return _failed("skin-specialized binding context", "top target skin context support is required")
    for key in ("top_target_id", "top_target_gene_symbol", "top_target_protein_name"):
        if key in binding and not _is_string(binding.get(key)):
            return _failed("skin-specialized binding context", f"{key} must be non-empty")
    for key in ("top_target_final_score", "top_target_docking_rrf", "top_target_skin_score"):
        if not _is_number(binding.get(key)):
            return _failed("skin-specialized binding context", f"{key} must be numeric")
    sources = binding.get("top_target_sources")
    if not isinstance(sources, list) or len(sources) < min_target_sources:
        return _failed(
            "skin-specialized binding context",
            f"top_target_sources must include >= {min_target_sources} labels",
        )
    if not isinstance(binding.get("top_target_source_count"), int):
        return _failed("skin-specialized binding context", "top_target_source_count must be integer")
    if binding.get("top_target_source_count") != len(sources):
        return _failed("skin-specialized binding context", "top_target_source_count/sources mismatch")
    most_skin = binding.get("most_skin_relevant_target")
    if not isinstance(most_skin, dict):
        return _failed("skin-specialized binding context", "missing most_skin_relevant_target object")
    for key in ("target_id", "gene_symbol", "protein_name"):
        if not _is_string(most_skin.get(key)):
            return _failed("skin-specialized binding context", f"most_skin_relevant_target.{key} must be non-empty")
    if not _is_number(most_skin.get("final_score")):
        return _failed("skin-specialized binding context", "most_skin_relevant_target.final_score must be numeric")
    efficacy_count = binding.get("top_targets_with_skin_efficacy")
    if not isinstance(efficacy_count, int) or efficacy_count < 0:
        return _failed("skin-specialized binding context", "top_targets_with_skin_efficacy must be non-negative integer")
    return _ok(
        "skin-specialized binding context",
        f"decision={decision} top_context={binding.get('top_target_skin_context_supported')}",
    )


def _read_run_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_manifest.json"
    if not path.exists() or path.is_symlink() or not path.is_file():
        return {}
    if path.stat().st_size == 0:
        return {}
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _check_contract_version(
    run_dir: Path,
    summary: dict[str, Any],
) -> tuple[GoalCheck, str]:
    """Resolve the artifact contract and refuse to call legacy results current.

    A run that predates the Daina structural overlay is reported as
    ``legacy/not_currently_verified`` instead of being silently accepted. A
    current-contract run whose manifest records a different mode or preset
    than the summary is rejected too: the artifact set is profile-specific.
    """
    try:
        contract = resolve_artifact_contract(run_dir)
    except UnsupportedRunManifestError as exc:
        return _failed("artifact contract version", str(exc)), CONTRACT_LEGACY
    if contract == CONTRACT_LEGACY:
        return (
            _failed(
                "artifact contract version",
                f"{CONTRACT_LEGACY}: run predates the current Daina structural "
                "contract (no v3 run manifest and no Daina structural artifact)",
            ),
            contract,
        )
    manifest = _read_run_manifest(run_dir)
    if manifest and "run_profile" in manifest:
        profile = manifest.get("run_profile")
        if not isinstance(profile, dict):
            return (
                _failed(
                    "artifact contract version",
                    "run_manifest run_profile is not an object",
                ),
                contract,
            )
        mismatches: list[str] = []
        if profile.get("execution_mode") != summary.get("mode"):
            mismatches.append("execution_mode")
        if profile.get("normalized_preset") != summary.get("preset"):
            mismatches.append("normalized_preset")
        if mismatches:
            return (
                _failed(
                    "artifact contract version",
                    "run_manifest profile does not match run_summary: "
                    + ", ".join(mismatches),
                ),
                contract,
            )
    return _ok("artifact contract version", f"{contract} artifact contract evidenced"), contract


def _check_ranking_binding(
    summary: dict[str, Any],
    *,
    run_dir: Path,
    contract: str,
) -> GoalCheck:
    """The summary's ranking path must match its artifact binding exactly.

    A summary that reports one target table but fingerprints a different one is
    rejected here; the fingerprint check below then re-hashes the bound file.
    """
    artifacts = summary.get("artifacts")
    ranking_relative = (
        artifacts.get("target_ranking") if isinstance(artifacts, dict) else None
    )
    if (
        not isinstance(ranking_relative, str)
        or not ranking_relative.strip()
        or ranking_relative != ranking_relative.strip()
        or Path(ranking_relative).is_absolute()
    ):
        return _failed(
            "ranking binding",
            "summary artifacts.target_ranking must be a non-empty run-relative path",
        )
    target_prediction = summary.get("target_prediction")
    predicted = (
        target_prediction.get("ranking_path")
        if isinstance(target_prediction, dict)
        else None
    )
    if predicted is not None and predicted != ranking_relative:
        return _failed(
            "ranking binding",
            f"summary ranking_path {predicted!r} differs from "
            f"artifacts.target_ranking {ranking_relative!r}",
        )
    if contract == CONTRACT_CURRENT and not (run_dir / ranking_relative).is_file():
        return _failed(
            "ranking binding",
            f"bound ranked target table is missing: {ranking_relative}",
        )
    return _ok("ranking binding", ranking_relative)


def _check_verification_record(
    verification: dict[str, Any],
    summary: dict[str, Any],
    *,
    run_dir: Path,
    allow_diagnostic: bool,
    contract: str,
) -> list[GoalCheck]:
    checks: list[GoalCheck] = []
    schema = verification.get("schema_version")
    if schema != RUN_VERIFICATION_SCHEMA:
        checks.append(_failed("run verification schema", f"expected {RUN_VERIFICATION_SCHEMA}, got {schema!r}"))
    else:
        checks.append(_ok("run verification schema", f"status={verification.get('status')}"))

    if verification.get("status") != "ok":
        checks.append(_failed("run verification status", f"status={verification.get('status')!r}"))
    else:
        checks.append(_ok("run verification status", "ok"))

    returncode = verification.get("returncode")
    verifier_status = verification.get("verifier_status")
    execution_errors: list[str] = []
    if (
        not isinstance(returncode, int)
        or isinstance(returncode, bool)
        or returncode != 0
    ):
        execution_errors.append(f"returncode={returncode!r}")
    if verifier_status != "ok":
        execution_errors.append(f"verifier_status={verifier_status!r}")
    if execution_errors:
        checks.append(
            _failed("run verifier execution", ", ".join(execution_errors))
        )
    else:
        checks.append(_ok("run verifier execution", "returncode=0 verifier_status=ok"))

    identity_mismatches: list[str] = []
    if summary.get("run_id") != run_dir.name:
        identity_mismatches.append("run_summary.run_id")
    if verification.get("run_dir") != str(run_dir):
        identity_mismatches.append("run_verification.run_dir")
    for key in ("preset", "mode"):
        if verification.get(key) != summary.get(key):
            identity_mismatches.append(key)
    if identity_mismatches:
        checks.append(
            _failed(
                "run identity binding",
                f"mismatched fields: {', '.join(identity_mismatches)}",
            )
        )
    else:
        checks.append(_ok("run identity binding", f"run_id={run_dir.name}"))

    diagnostic_reasons = verification.get("diagnostic_nonclaimable_reasons")
    if diagnostic_reasons and not allow_diagnostic:
        checks.append(_failed("diagnostic run guard", f"diagnostic reasons present: {diagnostic_reasons}"))
    else:
        checks.append(_ok("diagnostic run guard", "no diagnostic reasons"))

    verifier_checks = verification.get("checks")
    if not isinstance(verifier_checks, list) or not verifier_checks:
        checks.append(_failed("underlying verifier checks", "checks must be a non-empty list"))
    else:
        failed: list[str] = []
        for item in verifier_checks:
            if not isinstance(item, dict):
                failed.append("<malformed>")
            elif item.get("status") != "ok":
                failed.append(str(item.get("name", "<unnamed>")))
        if failed:
            checks.append(_failed("underlying verifier checks", f"non-ok checks: {', '.join(failed)}"))
        else:
            checks.append(_ok("underlying verifier checks", f"{len(verifier_checks)} checks ok"))

    artifacts = verification.get("verified_artifacts")
    if not isinstance(artifacts, list):
        checks.append(_failed("verified artifact fingerprints", "verified_artifacts must be a list"))
    else:
        names = [item.get("name") for item in artifacts if isinstance(item, dict)]
        by_name = {item.get("name"): item for item in artifacts if isinstance(item, dict)}
        mode = summary.get("mode")
        band_records_present = bool(FAST_BAND_VERIFIED_ARTIFACTS.keys() & set(by_name))
        band_outputs_present = any(
            (run_dir / relative).exists()
            for relative in FAST_BAND_VERIFIED_ARTIFACTS.values()
        )
        expected_artifacts = set(REQUIRED_VERIFIED_ARTIFACTS)
        if (
            (contract == CONTRACT_CURRENT and mode in {"fast", "both"})
            or band_records_present
            or band_outputs_present
        ):
            expected_artifacts.update(FAST_BAND_VERIFIED_ARTIFACTS)
        missing = sorted(expected_artifacts - set(by_name))
        duplicates = sorted({str(name) for name in names if names.count(name) > 1})
        if missing or duplicates:
            detail = []
            if missing:
                detail.append(f"missing artifacts: {', '.join(missing)}")
            if duplicates:
                detail.append(f"duplicate artifacts: {', '.join(duplicates)}")
            checks.append(_failed("verified artifact fingerprints", "; ".join(detail)))
        else:
            bad: list[str] = []
            summary_artifacts = summary.get("artifacts")
            ranking_relative = (
                summary_artifacts.get("target_ranking")
                if isinstance(summary_artifacts, dict)
                else None
            )
            ranking_valid = (
                isinstance(ranking_relative, str)
                and ranking_relative.strip()
                and ranking_relative == ranking_relative.strip()
                and not Path(ranking_relative).is_absolute()
            )
            expected_paths = {
                name: run_dir / relative
                for name, relative in required_verified_artifact_paths(
                    mode if isinstance(mode, str) else "",
                    contract,
                    target_ranking_relative=(
                        ranking_relative
                        if ranking_valid
                        else ".invalid-target-ranking"
                    ),
                ).items()
            }
            for name, relative in FAST_BAND_VERIFIED_ARTIFACTS.items():
                if name in expected_artifacts:
                    expected_paths.setdefault(name, run_dir / relative)
            if not ranking_valid:
                expected_paths["target_ranking"] = run_dir.parent / ".invalid-target-ranking"
            for name in sorted(expected_artifacts):
                item = by_name[name]
                sha = item.get("sha256")
                path = item.get("path")
                expected_path = expected_paths[name]
                expected = expected_path.resolve()
                actual = Path(path).resolve() if isinstance(path, str) else None
                if (
                    expected_path.is_symlink()
                    or not expected.is_relative_to(run_dir.resolve())
                    or actual != expected
                    or not expected.is_file()
                ):
                    bad.append(f"{name}.path")
                    continue
                data = expected.read_bytes()
                if item.get("bytes") != len(data) or len(data) <= 0:
                    bad.append(f"{name}.bytes")
                if (
                    not isinstance(sha, str)
                    or re.fullmatch(r"[0-9a-f]{64}", sha) is None
                    or hashlib.sha256(data).hexdigest() != sha
                ):
                    bad.append(f"{name}.sha256")
            if bad:
                checks.append(_failed("verified artifact fingerprints", f"invalid fields: {', '.join(bad)}"))
            else:
                checks.append(
                    _ok(
                        "verified artifact fingerprints",
                        f"{len(expected_artifacts)} required artifacts",
                    )
                )

    input_provenance = verification.get("input_provenance")
    compound = summary.get("compound") if isinstance(summary.get("compound"), dict) else {}
    if not isinstance(input_provenance, dict):
        checks.append(_failed("verification input provenance", "missing input_provenance object"))
    else:
        mismatched = [
            key
            for key in ("input_type", "input_smiles", "input_canonical_smiles")
            if input_provenance.get(key) != compound.get(key)
        ]
        if mismatched:
            checks.append(_failed("verification input provenance", f"mismatched fields: {', '.join(mismatched)}"))
        else:
            checks.append(_ok("verification input provenance", "matches run_summary compound input"))
    return checks


def verify_goal_contract(
    run_dir: Path,
    *,
    smiles: str | None = None,
    min_targets: int = 1,
    min_target_sources: int = 2,
    allow_degraded: bool = False,
    allow_diagnostic: bool = False,
    require_skin_context_supported: bool = False,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    summary, summary_check = _read_json(run_dir / "run_summary.json", "completed run summary")
    verification, verification_check = _read_json(
        run_dir / "run_verification.json",
        "completed run verification record",
    )
    checks = [summary_check, verification_check]
    contract_check, contract = _check_contract_version(run_dir, summary)
    checks.append(contract_check)
    if summary_check.status == "ok":
        checks.extend([
            _check_summary_schema(summary),
            _check_smiles_input(summary, smiles),
            _check_overall_decision(summary),
            _check_admet_and_skin_sens(summary, allow_degraded=allow_degraded),
            _check_skin_toxicity(summary, allow_degraded=allow_degraded),
            _check_target_prediction(
                summary,
                min_targets=min_targets,
                min_target_sources=min_target_sources,
            ),
            _check_screening_counts(summary, contract=contract),
            _check_ranking_binding(summary, run_dir=run_dir, contract=contract),
            _check_skin_binding(
                summary,
                min_target_sources=min_target_sources,
                require_skin_context_supported=require_skin_context_supported,
            ),
        ])
    if verification_check.status == "ok":
        checks.extend(
            _check_verification_record(
                verification,
                summary,
                run_dir=run_dir,
                allow_diagnostic=allow_diagnostic,
                contract=contract,
            )
        )
    status = "ok" if all(check.status == "ok" for check in checks) else "failed"
    return {
        "schema_version": GOAL_CONTRACT_SCHEMA,
        "status": status,
        "run_dir": str(run_dir),
        "contract_version": contract,
        "legacy_label": CONTRACT_LEGACY if contract == CONTRACT_LEGACY else None,
        "requirements": [
            "single_smiles_input",
            "admet_prediction",
            "skin_toxicity_prediction",
            "protein_target_prediction",
            "skin_specialized_material_protein_binding_context",
            "completed_run_verification_evidence",
        ],
        "checks": [asdict(check) for check in checks],
    }


def write_goal_contract_verification(
    payload: dict[str, Any],
    out_path: Path,
) -> Path:
    """Write a new goal-contract verification record without overwriting one.

    The stored ``run_verification.json`` is a producer artifact and is never
    mutated by re-verification. The new record is versioned separately and an
    existing file at the target path is left untouched: the caller must pick a
    new path instead of silently replacing evidence.
    """
    out_path = out_path.resolve()
    if out_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing verification record: {out_path}"
        )
    record = {
        "schema_version": GOAL_CONTRACT_VERIFICATION_SCHEMA,
        "status": payload.get("status"),
        "run_dir": payload.get("run_dir"),
        "contract_version": payload.get("contract_version"),
        "legacy_label": payload.get("legacy_label"),
        "requirements": payload.get("requirements", []),
        "checks": payload.get("checks", []),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    tmp.replace(out_path)
    return out_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--smiles", help="Expected raw input SMILES.")
    parser.add_argument("--min-targets", type=int, default=1)
    parser.add_argument("--min-target-sources", type=int, default=2)
    parser.add_argument("--allow-degraded", action="store_true")
    parser.add_argument("--allow-diagnostic", action="store_true")
    parser.add_argument("--require-skin-context-supported", action="store_true")
    parser.add_argument("--json", action="store_true", help="Emit JSON only.")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--write-verification",
        action="store_true",
        help=(
            "Write a new goal-contract verification record. Existing records "
            "are never overwritten."
        ),
    )
    parser.add_argument(
        "--verification-out",
        type=Path,
        help=(
            "Path for the new verification record; implies "
            "--write-verification and defaults to "
            f"<run-dir>/{GOAL_CONTRACT_VERIFICATION_FILENAME}."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.min_targets < 1:
        raise SystemExit("--min-targets must be >= 1")
    if args.min_target_sources < 1:
        raise SystemExit("--min-target-sources must be >= 1")
    payload = verify_goal_contract(
        args.run_dir,
        smiles=args.smiles,
        min_targets=args.min_targets,
        min_target_sources=args.min_target_sources,
        allow_degraded=args.allow_degraded,
        allow_diagnostic=args.allow_diagnostic,
        require_skin_context_supported=args.require_skin_context_supported,
    )
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.json_out.with_suffix(args.json_out.suffix + ".tmp")
        tmp.write_text(text)
        tmp.replace(args.json_out)
    if args.write_verification or args.verification_out is not None:
        out_path = args.verification_out or (
            args.run_dir / GOAL_CONTRACT_VERIFICATION_FILENAME
        )
        try:
            written = write_goal_contract_verification(payload, out_path)
        except FileExistsError as exc:
            raise SystemExit(str(exc)) from exc
        print(f"SkinScout goal contract verification written: {written}")
    if args.json:
        sys.stdout.write(text)
    else:
        print(f"SkinScout goal contract verification: {payload['status']}")
        print(f"run_dir={payload['run_dir']}")
        for check in payload["checks"]:
            suffix = f" - {check['detail']}" if check.get("detail") else ""
            print(f"[{check['status']}] {check['name']}{suffix}")
    return 0 if payload["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
