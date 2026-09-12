#!/usr/bin/env python3
"""Verify that a completed SkinScout run satisfies the user-facing contract."""

from __future__ import annotations

import argparse
import hashlib
from html import unescape
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem

from cosmetic_drug_contract import (
    CosmeticDrugContractError,
    parse_decision_and_policy,
    validate_decision_matches_drug_warnings,
)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skinscout.contracts.run_profile import (  # noqa: E402
    RUN_MANIFEST_SCHEMA_V3,
    validate_run_manifest,
    validate_target_fast_v2_record,
)


PRESETS = {"safety", "target-id", "report"}
MODES = {"comprehensive", "fast", "both"}
PRESET_DEPTH = {"safety": 1, "target-id": 2, "report": 3}
SAFETY_DECISIONS = {"HALT", "FLAG_HIGH", "PASS"}
FRACTION_COLUMNS = {"final_score", "skin_score", "docking_rrf", "efficacy_score"}
SUMMARY_SCHEMA = "skinscout.run_summary.v1"
VERIFY_OUTPUT_SCHEMA = "skinscout.run_output_verification.v1"
OVERALL_DECISIONS = {"HALT", "FLAG_HIGH", "PASS"}
OVERALL_RECOMMENDED_ACTIONS = {
    "HALT": "stop_before_claim",
    "FLAG_HIGH": "review_before_claim",
    "PASS": "proceed",
}
SKIN_TOXICITY_DECISIONS = {"HALT", "PASS", "REVIEW"}
SKIN_TOXICITY_LEVELS = {"high", "low", "moderate"}
SKIN_TOXICITY_DECISION_ORDER = {"PASS": 0, "REVIEW": 1, "HALT": 2}
SKIN_TOXICITY_LEVEL_ORDER = {"low": 0, "moderate": 1, "high": 2}
ADMET_MODERATE_RISK_THRESHOLD = 0.25
ADMET_HIGH_RISK_THRESHOLD = 0.50
SKIN_EXPRESSION_SUPPORTED_TIERS = {"low", "medium", "high", "very_high"}
SKIN_EXPRESSION_SUPPORTED_THRESHOLD = 0.20
SKIN_CONTEXT_DECISIONS = {
    "skin_context_supported",
    "skin_expression_only",
    "skin_efficacy_literature_only",
    "insufficient_skin_context",
}
SKIN_SENS_MODELS = ("husspred", "stoptox", "pred_skin")
SKIN_SENS_CALLS = {"positive", "negative"}
ADMET_CORE_ENDPOINTS = {
    "molecular_weight",
    "logP",
    "QED",
    "tpsa",
    "AMES",
    "ClinTox",
    "DILI",
    "Skin_Reaction",
    "hERG",
    "Caco2_Wang",
    "LD50_Zhu",
    "Solubility_AqSolDB",
}
ADMET_PROBABILITY_ENDPOINTS = {"AMES", "ClinTox", "DILI", "Skin_Reaction", "hERG"}
SUMMARY_ADMET_METRICS = (
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
LEGACY_FAST_TARGET_ARTIFACTS = {
    "target_fast_psichic": "03_targets/mode_fast/psichic_proteome.tsv",
    "target_fast_daina_zoete": "03_targets/mode_fast/daina_zoete_proteome.tsv",
    "target_fast_dti_rrf": "03_targets/mode_fast/dti_rrf_top25pct.csv",
    "target_fast_autodock": "03_targets/mode_fast/autodock_top5k.tsv",
    "target_fast_rerank_consensus": "03_targets/mode_fast/top50.csv",
}
FAST_TARGET_ARTIFACTS = {
    "target_fast_daina_zoete": "03_targets/mode_fast/daina_zoete_proteome.tsv",
    "target_fast_daina_selected": "03_targets/mode_fast/daina_top256.csv",
    "target_fast_dti_rrf": "03_targets/mode_fast/dti_rrf_top25pct.csv",
    "target_fast_autogrid_manifest": "03_targets/mode_fast/autogrid_map_manifest.json",
    "target_fast_autodock": "03_targets/mode_fast/autodock_top5k.tsv",
    "target_fast_gnina_pose": "03_targets/mode_fast/gnina_pose_rescores.tsv",
    "target_fast_daina_structural_targets": "03_targets/mode_fast/daina_structural_targets.csv",
    "target_fast_rerank_consensus": "03_targets/mode_fast/top50.csv",
    "target_fast_band_reranked": "03_targets/mode_fast/top50_band_reranked.csv",
}
COMPREHENSIVE_TARGET_ARTIFACTS = {
    "target_comprehensive_ligand": "03_targets/mode_comprehensive/ligand.pdbqt",
    "target_comprehensive_autodock": "03_targets/mode_comprehensive/autodock_all_targets.tsv",
    "target_comprehensive_pre_rescore": "03_targets/mode_comprehensive/top_pct_pre_rescore.csv",
    "target_comprehensive_gnina": "03_targets/mode_comprehensive/gnina_rescores.tsv",
    "target_comprehensive_rtmscore": "03_targets/mode_comprehensive/rtmscore_rescores.tsv",
    "target_comprehensive_boltz2": "03_targets/mode_comprehensive/boltz2_affinity_top.tsv",
    "target_comprehensive_consensus": "03_targets/mode_comprehensive/top50_4way_consensus.csv",
}
LEGACY_FAST_SCREENING_COUNT_ARTIFACTS = (
    ("psichic_proteome_targets", "03_targets/mode_fast/psichic_proteome.tsv"),
    ("daina_zoete_targets", "03_targets/mode_fast/daina_zoete_proteome.tsv"),
    ("dti_rrf_candidates", "03_targets/mode_fast/dti_rrf_top25pct.csv"),
    ("autodock_rescored_targets", "03_targets/mode_fast/autodock_top5k.tsv"),
    ("rerank_consensus_targets", "03_targets/mode_fast/top50.csv"),
)
FAST_SCREENING_COUNT_ARTIFACTS = (
    ("daina_zoete_targets", "03_targets/mode_fast/daina_zoete_proteome.tsv"),
    ("daina_primary_candidates", "03_targets/mode_fast/daina_top256.csv"),
    ("autodock_rescored_targets", "03_targets/mode_fast/autodock_top5k.tsv"),
    ("gnina_pose_rescored_targets", "03_targets/mode_fast/gnina_pose_rescores.tsv"),
    ("daina_structural_targets", "03_targets/mode_fast/daina_structural_targets.csv"),
    ("rerank_consensus_targets", "03_targets/mode_fast/top50.csv"),
    ("band_reranked_targets", "03_targets/mode_fast/top50_band_reranked.csv"),
)
COMPREHENSIVE_SCREENING_COUNT_ARTIFACTS = (
    (
        "autodock_screened_targets",
        "03_targets/mode_comprehensive/autodock_all_targets.tsv",
    ),
    (
        "pre_rescore_candidates",
        "03_targets/mode_comprehensive/top_pct_pre_rescore.csv",
    ),
    ("gnina_rescored_targets", "03_targets/mode_comprehensive/gnina_rescores.tsv"),
    (
        "rtmscore_rescored_targets",
        "03_targets/mode_comprehensive/rtmscore_rescores.tsv",
    ),
    (
        "boltz2_affinity_targets",
        "03_targets/mode_comprehensive/boltz2_affinity_top.tsv",
    ),
    (
        "four_way_consensus_targets",
        "03_targets/mode_comprehensive/top50_4way_consensus.csv",
    ),
)
LEGACY_FAST_SCREENING_FUNNEL_EDGES = (
    ("psichic_proteome_targets", "dti_rrf_candidates"),
    ("daina_zoete_targets", "dti_rrf_candidates"),
    ("dti_rrf_candidates", "autodock_rescored_targets"),
    ("autodock_rescored_targets", "rerank_consensus_targets"),
)
DAINA_FAST_SCREENING_FUNNEL_EDGES = (
    ("daina_zoete_targets", "daina_primary_candidates"),
    ("daina_primary_candidates", "autodock_rescored_targets"),
    ("autodock_rescored_targets", "gnina_pose_rescored_targets"),
    ("daina_primary_candidates", "daina_structural_targets"),
    ("daina_structural_targets", "rerank_consensus_targets"),
    ("rerank_consensus_targets", "band_reranked_targets"),
)
COMPREHENSIVE_SCREENING_FUNNEL_EDGES = (
    ("autodock_screened_targets", "pre_rescore_candidates"),
    ("pre_rescore_candidates", "gnina_rescored_targets"),
    ("pre_rescore_candidates", "rtmscore_rescored_targets"),
    ("pre_rescore_candidates", "boltz2_affinity_targets"),
    ("gnina_rescored_targets", "four_way_consensus_targets"),
    ("rtmscore_rescored_targets", "four_way_consensus_targets"),
    ("boltz2_affinity_targets", "four_way_consensus_targets"),
)
FAST_FINAL_FUNNEL_EDGE = ("band_reranked_targets", "skin_weighted_ranked_targets")
COMPREHENSIVE_FINAL_FUNNEL_EDGE = (
    "four_way_consensus_targets",
    "skin_weighted_ranked_targets",
)


@dataclass(frozen=True)
class OutputCheck:
    name: str
    status: str
    path: str
    detail: str = ""


def _nonempty(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def _run_manifest_schema(run_dir: Path) -> str | None:
    path = run_dir / "run_manifest.json"
    if not _nonempty(path):
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    return str(payload.get("schema_version", "")) if isinstance(payload, dict) else ""


def _uses_daina_structural_contract(run_dir: Path) -> bool:
    schema = _run_manifest_schema(run_dir)
    if schema == RUN_MANIFEST_SCHEMA_V3:
        return True
    if schema == "skinscout.run_manifest.v2":
        return False
    return _nonempty(
        run_dir / "03_targets" / "mode_fast" / "daina_structural_targets.csv"
    )


def _check_path(path: Path, name: str) -> OutputCheck:
    if not path.exists():
        return OutputCheck(name, "missing", str(path))
    if path.is_dir():
        return OutputCheck(name, "invalid", str(path), "expected a file")
    if path.stat().st_size == 0:
        return OutputCheck(name, "empty", str(path))
    return OutputCheck(name, "ok", str(path), f"bytes={path.stat().st_size}")


def _read_json(path: Path, name: str) -> tuple[dict[str, Any], OutputCheck]:
    check = _check_path(path, name)
    if check.status != "ok":
        return {}, check
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return {}, OutputCheck(name, "invalid", str(path), f"invalid JSON: {exc}")
    if not isinstance(payload, dict):
        return {}, OutputCheck(name, "invalid", str(path), "expected JSON object")
    return payload, check


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_member_path(manifest_path: Path, value: object) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    root = manifest_path.parent.resolve()
    candidate = (root / value.strip()).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def _verify_run_manifest_contract(
    run_dir: Path,
    checks: list[OutputCheck],
    *,
    preset: str,
    mode: str,
) -> None:
    path = run_dir / "run_manifest.json"
    daina_structural = _uses_daina_structural_contract(run_dir)
    if not path.exists():
        if daina_structural:
            checks.append(
                OutputCheck(
                    "run manifest",
                    "missing",
                    str(path),
                    "Daina structural runs require a finalized v3 manifest",
                )
            )
        return
    payload, check = _read_json(path, "run manifest")
    if not _append_if_bad(checks, check):
        return
    try:
        validate_run_manifest(payload)
    except ValueError as exc:
        checks.append(_invalid("run manifest", path, str(exc)))
        return
    profile = payload.get("run_profile")
    if isinstance(profile, dict):
        if profile.get("normalized_preset") != preset:
            checks.append(
                _invalid(
                    "run manifest",
                    path,
                    "run_profile normalized_preset does not match verifier preset",
                )
            )
        if profile.get("execution_mode") != mode:
            checks.append(
                _invalid(
                    "run manifest",
                    path,
                    "run_profile execution_mode does not match verifier mode",
                )
            )
    if payload.get("schema_version") != RUN_MANIFEST_SCHEMA_V3:
        return
    target_fast = payload.get("target_fast")
    if not isinstance(target_fast, dict):
        return
    canonical = run_dir / str(target_fast.get("path", ""))
    expected_hash = target_fast.get("sha256")
    if not isinstance(expected_hash, str):
        checks.append(
            _invalid("run manifest", path, "target_fast sha256 is not finalized")
        )
    elif not _nonempty(canonical):
        checks.append(OutputCheck("canonical target-fast artifact", "missing", str(canonical)))
    elif _sha256(canonical) != expected_hash:
        checks.append(
            _invalid("run manifest", path, "target_fast sha256 does not match artifact")
        )
    projection = target_fast.get("compatibility_projection")
    if not isinstance(projection, dict):
        return
    top50 = run_dir / str(projection.get("path", ""))
    projection_hash = projection.get("sha256")
    if not isinstance(projection_hash, str):
        checks.append(
            _invalid(
                "run manifest",
                path,
                "compatibility projection sha256 is not finalized",
            )
        )
    elif not _nonempty(top50):
        checks.append(OutputCheck("target-fast top50 projection", "missing", str(top50)))
    elif _sha256(top50) != projection_hash:
        checks.append(
            _invalid(
                "run manifest",
                path,
                "compatibility projection sha256 does not match artifact",
            )
        )


def _read_text(path: Path, name: str) -> tuple[str, OutputCheck]:
    check = _check_path(path, name)
    if check.status != "ok":
        return "", check
    return path.read_text().strip(), check


def _read_csv(path: Path, name: str, *, sep: str = ",") -> tuple[pd.DataFrame, OutputCheck]:
    check = _check_path(path, name)
    if check.status != "ok":
        return pd.DataFrame(), check
    try:
        df = pd.read_csv(path, sep=sep)
    except Exception as exc:  # noqa: BLE001
        return pd.DataFrame(), OutputCheck(name, "invalid", str(path), f"parse failed: {exc}")
    if df.empty:
        return df, OutputCheck(name, "invalid", str(path), "contains no rows")
    return df, check


def _target_screening_count_expectations(
    run_dir: Path,
    *,
    mode: str,
    ranked_count: int,
) -> tuple[dict[str, int], list[OutputCheck]]:
    expectations = {"skin_weighted_ranked_targets": ranked_count}
    checks: list[OutputCheck] = []
    artifacts: list[tuple[str, str]] = []
    if mode in {"fast", "both"}:
        artifacts.extend(
            FAST_SCREENING_COUNT_ARTIFACTS
            if _uses_daina_structural_contract(run_dir)
            else LEGACY_FAST_SCREENING_COUNT_ARTIFACTS
        )
    if mode in {"comprehensive", "both"}:
        artifacts.extend(COMPREHENSIVE_SCREENING_COUNT_ARTIFACTS)
    for key, rel_path in artifacts:
        path = run_dir / rel_path
        sep = "\t" if path.suffix == ".tsv" else ","
        df, check = _read_csv(path, f"target screening count {key}", sep=sep)
        if check.status == "ok":
            expectations[key] = int(len(df))
        else:
            checks.append(check)
    return expectations, checks


def _screening_funnel_edges(
    run_dir: Path, mode: str
) -> tuple[tuple[str, str], ...]:
    edges: list[tuple[str, str]] = []
    if mode in {"fast", "both"}:
        edges.extend(
            DAINA_FAST_SCREENING_FUNNEL_EDGES
            if _uses_daina_structural_contract(run_dir)
            else LEGACY_FAST_SCREENING_FUNNEL_EDGES
        )
    if mode in {"comprehensive", "both"}:
        edges.extend(COMPREHENSIVE_SCREENING_FUNNEL_EDGES)
    if mode == "fast":
        edges.append(FAST_FINAL_FUNNEL_EDGE)
    if mode in {"comprehensive", "both"}:
        edges.append(COMPREHENSIVE_FINAL_FUNNEL_EDGE)
    return tuple(edges)


def _screening_funnel_violations(
    run_dir: Path, counts: dict[str, int], *, mode: str
) -> list[str]:
    return [
        f"{upstream}={counts[upstream]} < {downstream}={counts[downstream]}"
        for upstream, downstream in _screening_funnel_edges(run_dir, mode)
        if upstream in counts and downstream in counts and counts[upstream] < counts[downstream]
    ]


def _append_if_bad(checks: list[OutputCheck], check: OutputCheck) -> bool:
    checks.append(check)
    return check.status == "ok"


def _invalid(name: str, path: Path, detail: str) -> OutputCheck:
    return OutputCheck(name, "invalid", str(path), detail)


def _first_nonempty_line(text: str) -> str | None:
    for line in text.splitlines():
        value = line.strip()
        if value:
            return value
    return None


def _required_text_field(payload: dict[str, Any], key: str, name: str, path: Path) -> str | None:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _canonical_smiles_value(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    mol = Chem.MolFromSmiles(value.strip())
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _verify_source_smiles(
    payload: dict[str, Any],
    checks: list[OutputCheck],
    name: str,
    path: Path,
    compound_smiles: str | None,
) -> bool:
    if compound_smiles is None:
        return True
    expected_smiles = _canonical_smiles_value(compound_smiles)
    if expected_smiles is None:
        return True
    source_smiles = _canonical_smiles_value(payload.get("smiles"))
    if source_smiles is None:
        checks.append(_invalid(name, path, "missing or invalid smiles"))
        return False
    if source_smiles != expected_smiles:
        checks.append(_invalid(name, path, "smiles does not match compound canonical_smiles"))
        return False
    return True


def _bool_like(value: object) -> bool:
    return (
        isinstance(value, bool)
        or type(value).__name__ == "bool_"
        or (isinstance(value, str) and value.strip().lower() in {"true", "false"})
    )


def _numeric_value(
    value: object,
    name: str,
    path: Path,
    field: str,
    *,
    fraction: bool = False,
) -> OutputCheck | None:
    if _bool_like(value):
        return _invalid(name, path, f"{field} must be numeric")
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return _invalid(name, path, f"{field} must be numeric")
    if not math.isfinite(numeric):
        return _invalid(name, path, f"{field} must be finite")
    if fraction and (numeric < 0.0 or numeric > 1.0):
        return _invalid(name, path, f"{field} outside [0, 1]")
    return None


def _text_or_none(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _split_uniprots(value: object) -> list[str]:
    text = _text_or_none(value)
    if text is None:
        return []
    normalized = text.replace(",", ";").replace("|", ";")
    return [part.strip() for part in normalized.split(";") if part.strip()]


def _load_target_metadata(path: Path | None) -> dict[str, dict[str, str]]:
    metadata_path = path or ROOT / "data" / "hpa" / "proteinatlas.tsv"
    if not _nonempty(metadata_path):
        raise SystemExit(f"target metadata is required and must be non-empty: {metadata_path}")
    wanted = {
        "uniprot",
        "uniprot_id",
        "target_id",
        "gene",
        "gene_symbol",
        "symbol",
        "gene description",
        "protein_name",
        "protein",
        "description",
    }
    try:
        df = pd.read_csv(
            metadata_path,
            sep="\t",
            dtype=str,
            usecols=lambda col: col.lower() in wanted,
        )
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"target metadata failed to parse: {metadata_path}: {exc}") from exc
    columns = {col.lower(): col for col in df.columns}
    uid_col = columns.get("uniprot") or columns.get("uniprot_id") or columns.get("target_id")
    gene_col = columns.get("gene") or columns.get("gene_symbol") or columns.get("symbol")
    protein_col = (
        columns.get("gene description")
        or columns.get("protein_name")
        or columns.get("protein")
        or columns.get("description")
    )
    if uid_col is None:
        raise SystemExit(
            f"target metadata missing required UniProt identifier column: {metadata_path}"
        )
    if gene_col is None and protein_col is None:
        raise SystemExit(
            f"target metadata must include a gene or protein-name column: {metadata_path}"
        )
    out: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        gene = _text_or_none(row.get(gene_col)) if gene_col else None
        protein = _text_or_none(row.get(protein_col)) if protein_col else None
        if gene is None and protein is None:
            continue
        for uid in _split_uniprots(row.get(uid_col)):
            out.setdefault(uid, {})
            if gene is not None and "gene_symbol" not in out[uid]:
                out[uid]["gene_symbol"] = gene
            if protein is not None and "protein_name" not in out[uid]:
                out[uid]["protein_name"] = protein
    return out


def _admet_risk_level(value: float) -> str:
    if value >= ADMET_HIGH_RISK_THRESHOLD:
        return "high"
    if value >= ADMET_MODERATE_RISK_THRESHOLD:
        return "moderate"
    return "low"


def _admet_risk_assessment(admet_metrics: dict[str, float]) -> dict[str, Any]:
    endpoints: dict[str, dict[str, Any]] = {}
    for endpoint in sorted(ADMET_PROBABILITY_ENDPOINTS):
        if endpoint not in admet_metrics:
            continue
        value = admet_metrics[endpoint]
        endpoints[endpoint] = {
            "value": value,
            "risk_level": _admet_risk_level(value),
            "moderate_threshold": ADMET_MODERATE_RISK_THRESHOLD,
            "high_threshold": ADMET_HIGH_RISK_THRESHOLD,
        }
    return {
        "endpoints": endpoints,
        "high_risk_endpoints": [
            endpoint
            for endpoint, item in endpoints.items()
            if item["risk_level"] == "high"
        ],
        "moderate_risk_endpoints": [
            endpoint
            for endpoint, item in endpoints.items()
            if item["risk_level"] == "moderate"
        ],
    }


def _raise_skin_toxicity_decision(current: str, candidate: str) -> str:
    if SKIN_TOXICITY_DECISION_ORDER[candidate] > SKIN_TOXICITY_DECISION_ORDER[current]:
        return candidate
    return current


def _raise_skin_toxicity_level(current: str, candidate: str) -> str:
    if SKIN_TOXICITY_LEVEL_ORDER[candidate] > SKIN_TOXICITY_LEVEL_ORDER[current]:
        return candidate
    return current


def _skin_toxicity_summary(safety: dict[str, Any]) -> dict[str, Any]:
    decision = "PASS"
    toxicity_level = "low"
    reasons: list[str] = []

    def review(reason: str, level: str = "moderate") -> None:
        nonlocal decision, toxicity_level
        decision = _raise_skin_toxicity_decision(decision, "REVIEW")
        toxicity_level = _raise_skin_toxicity_level(toxicity_level, level)
        reasons.append(reason)

    skin_decision = safety.get("skin_sens_decision")
    if skin_decision == "HALT":
        decision = _raise_skin_toxicity_decision(decision, "HALT")
        toxicity_level = _raise_skin_toxicity_level(toxicity_level, "high")
        reasons.append("skin sensitization consensus is HALT")
    elif skin_decision == "FLAG_HIGH":
        review("skin sensitization consensus is FLAG_HIGH", "high")
    elif skin_decision != "PASS":
        review(f"skin sensitization decision is unrecognized: {skin_decision!r}", "high")

    admet_risk = safety.get("admet_risk_assessment")
    endpoints = admet_risk.get("endpoints") if isinstance(admet_risk, dict) else {}
    skin_reaction = endpoints.get("Skin_Reaction") if isinstance(endpoints, dict) else None
    skin_reaction_value = None
    skin_reaction_risk = None
    if isinstance(skin_reaction, dict):
        skin_reaction_value = skin_reaction.get("value")
        skin_reaction_risk = skin_reaction.get("risk_level")
        if skin_reaction_risk == "high":
            review("Skin_Reaction ADMET risk is high", "high")
        elif skin_reaction_risk == "moderate":
            review("Skin_Reaction ADMET risk is moderate", "moderate")
        elif skin_reaction_risk != "low":
            review(f"Skin_Reaction ADMET risk is unrecognized: {skin_reaction_risk!r}")
    else:
        review("Skin_Reaction ADMET endpoint is missing")

    structural_flags = safety.get("structural_alert_flags")
    flagged_alerts = (
        sorted(str(key) for key, value in structural_flags.items() if value)
        if isinstance(structural_flags, dict)
        else []
    )
    if flagged_alerts:
        review("structural alerts present: " + ",".join(flagged_alerts))

    degraded = bool(safety.get("degraded"))
    if degraded:
        review("skin-sens evidence is degraded")

    missing_models = safety.get("missing_models")
    missing_model_names = (
        [str(model) for model in missing_models if str(model).strip()]
        if isinstance(missing_models, list)
        else []
    )
    if missing_model_names:
        review("missing skin-sens models: " + ",".join(missing_model_names))

    if not reasons:
        reasons.append("skin sensitization and Skin_Reaction ADMET risk are low")

    return {
        "decision": decision,
        "toxicity_level": toxicity_level,
        "skin_sens_decision": skin_decision,
        "skin_reaction_risk_level": skin_reaction_risk,
        "skin_reaction_value": skin_reaction_value,
        "structural_alerts_present": bool(flagged_alerts),
        "structural_alert_flags": flagged_alerts,
        "degraded": degraded,
        "missing_models": missing_model_names,
        "reasons": reasons,
    }


def _summary_admet_metrics(run_dir: Path) -> dict[str, float]:
    payload, check = _read_json(run_dir / "02_admet" / "admet_report.json", "summary ADMET cross-check")
    if check.status != "ok":
        return {}
    predictions = payload.get("admet_ai_predictions")
    if not isinstance(predictions, dict):
        return {}
    metrics: dict[str, float] = {}
    for endpoint in SUMMARY_ADMET_METRICS:
        value = predictions.get(endpoint)
        if _bool_like(value):
            continue
        try:
            numeric = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            metrics[endpoint] = numeric
    return metrics


def _summary_skin_sens_calls(run_dir: Path) -> dict[str, object]:
    payload, check = _read_json(run_dir / "02_admet" / "admet_report.json", "summary skin-sens cross-check")
    if check.status != "ok":
        return {}
    skin_sens = payload.get("skin_sens")
    if not isinstance(skin_sens, dict):
        return {}
    return {model: skin_sens[model] for model in SKIN_SENS_MODELS if model in skin_sens}


def _summary_structural_alert_flags(run_dir: Path) -> dict[str, bool] | None:
    payload, check = _read_json(
        run_dir / "02_admet" / "admet_report.json",
        "summary structural-alert cross-check",
    )
    if check.status != "ok":
        return None
    structural_alerts = payload.get("structural_alerts")
    if not isinstance(structural_alerts, dict):
        return None
    flags: dict[str, bool] = {}
    for key in ("any_pains", "any_brenk", "any_nih"):
        value = structural_alerts.get(key)
        if not isinstance(value, bool):
            return None
        flags[key] = value
    return flags


def _summary_skin_sens_evidence(run_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in SKIN_SENS_MODELS:
        payload, check = _read_json(
            run_dir / "02_admet" / f"{model}.json",
            f"{model} skin-sens source evidence",
        )
        if check.status != "ok":
            continue
        status = _text_or_none(payload.get("status")) or "n/a"
        call, probability = _skin_sens_source_call(payload, model)
        row: dict[str, object] = {
            "model": model,
            "status": status,
            "call": _text_or_none(call),
            "probability": None,
        }
        if status == "ok" and probability is not None and not _bool_like(probability):
            try:
                numeric = float(probability)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                numeric = math.nan
            if math.isfinite(numeric):
                row["probability"] = numeric
        rows.append(row)
    return rows


def _numeric_series(
    df: pd.DataFrame,
    column: str,
    name: str,
    path: Path,
    *,
    fraction: bool = False,
) -> OutputCheck | None:
    bool_rows = [int(idx) for idx, value in df[column].items() if _bool_like(value)]
    if bool_rows:
        return _invalid(name, path, f"column '{column}' must be numeric")
    numeric = pd.to_numeric(df[column], errors="coerce")
    bad_rows = [
        int(idx)
        for idx, value in numeric.items()
        if pd.isna(value) or not math.isfinite(float(value))
    ]
    if bad_rows:
        shown = ", ".join(str(idx) for idx in bad_rows[:10])
        return _invalid(name, path, f"column '{column}' has non-numeric rows: {shown}")
    if fraction:
        out_of_range = [
            int(idx)
            for idx, value in numeric.items()
            if float(value) < 0.0 or float(value) > 1.0
        ]
        if out_of_range:
            shown = ", ".join(str(idx) for idx in out_of_range[:10])
            return _invalid(name, path, f"column '{column}' outside [0, 1]: {shown}")
    df[column] = numeric
    return None


def _source_labels(value: object) -> list[str]:
    if pd.isna(value) or not str(value).strip():
        return []
    labels = [part.strip() for part in str(value).split(";")]
    if any(label == "" for label in labels):
        return []
    if len(set(labels)) != len(labels):
        return []
    return labels


def _efficacy_labels(row: pd.Series) -> list[str]:
    labels: list[str] = []
    for col in sorted(c for c in row.index if c.startswith("efficacy_top")):
        value = _text_or_none(row.get(col))
        if value is not None:
            labels.append(value)
    return labels


def _row_float(row: pd.Series, key: str) -> float:
    try:
        return float(row.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _row_has_skin_expression_support(row: pd.Series) -> bool:
    tier = _text_or_none(row.get("skin_tier"))
    return (
        (tier in SKIN_EXPRESSION_SUPPORTED_TIERS)
        or _row_float(row, "skin_score") >= SKIN_EXPRESSION_SUPPORTED_THRESHOLD
    )


def _expected_skin_context(
    *,
    best_row: pd.Series,
    top_targets_with_skin_efficacy: int,
) -> tuple[str, bool, bool, list[str]]:
    expression_supported = _row_has_skin_expression_support(best_row)
    efficacy_supported = top_targets_with_skin_efficacy > 0
    reasons = [
        "top targets include low-or-higher skin-expression support"
        if expression_supported
        else "top targets have very-low skin-expression support",
        "top targets include KG skin-efficacy labels"
        if efficacy_supported
        else "top targets lack KG skin-efficacy labels",
    ]
    if expression_supported and efficacy_supported:
        decision = "skin_context_supported"
    elif expression_supported:
        decision = "skin_expression_only"
    elif efficacy_supported:
        decision = "skin_efficacy_literature_only"
    else:
        decision = "insufficient_skin_context"
    return decision, expression_supported, efficacy_supported, reasons


def _validate_target_ids(df: pd.DataFrame, name: str, path: Path) -> list[OutputCheck]:
    checks: list[OutputCheck] = []
    target_ids = df["target_id"].fillna("").astype(str).str.strip()
    if (target_ids == "").any():
        checks.append(_invalid(name, path, "blank target_id value"))
    duplicates = sorted(target_ids[target_ids.duplicated()].unique())
    if duplicates:
        checks.append(
            _invalid(
                name,
                path,
                f"duplicate target_id values: {', '.join(duplicates[:10])}",
            )
        )
    df["target_id"] = target_ids
    return checks


def _validate_source_support(
    df: pd.DataFrame,
    name: str,
    path: Path,
    *,
    min_source_count: int,
) -> list[OutputCheck]:
    checks: list[OutputCheck] = []
    bad = _numeric_series(df, "source_count", name, path)
    if bad is not None:
        checks.append(bad)
        return checks
    for idx, row in df.iterrows():
        source_count = float(row["source_count"])
        labels = _source_labels(row["sources"])
        if not source_count.is_integer() or int(source_count) < min_source_count:
            checks.append(
                _invalid(
                    name,
                    path,
                    f"row {int(idx)} source_count must be integer >= {min_source_count}",
                )
            )
            break
        if int(source_count) != len(labels):
            checks.append(
                _invalid(
                    name,
                    path,
                    f"row {int(idx)} source_count does not match sources labels",
                )
            )
            break
    return checks


def _verify_table(
    checks: list[OutputCheck],
    path: Path,
    name: str,
    *,
    required_columns: set[str],
    numeric_columns: set[str],
    sep: str = ",",
    min_rows: int = 1,
    min_source_count: int | None = None,
) -> None:
    df, check = _read_csv(path, name, sep=sep)
    if not _append_if_bad(checks, check):
        return
    missing = sorted(required_columns - set(df.columns))
    if missing:
        checks.append(_invalid(name, path, f"missing columns {missing}"))
        return
    if len(df) < min_rows:
        checks.append(_invalid(name, path, f"expected at least {min_rows} row(s), found {len(df)}"))
    if "target_id" in required_columns:
        checks.extend(_validate_target_ids(df, name, path))
    for column in sorted(numeric_columns):
        bad = _numeric_series(df, column, name, path)
        if bad is not None:
            checks.append(bad)
    if min_source_count is not None:
        checks.extend(
            _validate_source_support(
                df,
                name,
                path,
                min_source_count=min_source_count,
            )
        )


def _verify_autodock_claim_scores(
    checks: list[OutputCheck],
    path: Path,
    name: str,
) -> None:
    df, check = _read_csv(path, name, sep="\t")
    if not _append_if_bad(checks, check):
        return
    required_columns = {
        "target_id",
        "vina_score",
        "neg_vina_score",
        "engine",
        "map_coverage_complete",
        "map_coverage_numerator",
        "map_coverage_denominator",
        "degraded",
    }
    missing = sorted(required_columns - set(df.columns))
    if missing:
        checks.append(_invalid(name, path, f"missing columns {missing}"))
        return
    checks.extend(_validate_target_ids(df, name, path))
    numeric_invalid = False
    for column in (
        "neg_vina_score",
        "vina_score",
        "map_coverage_numerator",
        "map_coverage_denominator",
    ):
        bad = _numeric_series(df, column, name, path)
        if bad is not None:
            checks.append(bad)
            numeric_invalid = True
    if numeric_invalid:
        return
    expected_denominator: int | None = None
    for idx, row in df.iterrows():
        row_number = int(idx) + 2
        engine = str(row.get("engine", "")).strip()
        if engine != "autodock_gpu":
            checks.append(
                _invalid(
                    name,
                    path,
                    f"row {row_number} expected engine autodock_gpu, found {engine!r}",
                )
            )
            break
        coverage = str(row.get("map_coverage_complete", "")).strip().lower()
        if coverage != "true":
            checks.append(
                _invalid(
                    name,
                    path,
                    f"row {row_number} map coverage must be complete",
                )
            )
            break
        degraded = str(row.get("degraded", "")).strip().lower()
        if degraded != "false":
            checks.append(
                _invalid(
                    name,
                    path,
                    f"row {row_number} degraded must be false for claim outputs",
                )
            )
            break
        if not math.isclose(
            float(row["vina_score"]),
            -float(row["neg_vina_score"]),
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            checks.append(
                _invalid(
                    name,
                    path,
                    f"row {row_number} vina_score and neg_vina_score must be opposites",
                )
            )
            break
        numerator = row["map_coverage_numerator"]
        denominator = row["map_coverage_denominator"]
        if (
            not float(numerator).is_integer()
            or not float(denominator).is_integer()
            or int(denominator) < 1
            or int(numerator) != int(denominator)
        ):
            checks.append(
                _invalid(
                    name,
                    path,
                    f"row {row_number} map coverage counts must be complete integers",
                )
            )
            break
        row_denominator = int(denominator)
        if expected_denominator is None:
            expected_denominator = row_denominator
        elif row_denominator != expected_denominator:
            checks.append(
                _invalid(
                    name,
                    path,
                    "map coverage denominator must be identical on every row",
                )
            )
            break
    if expected_denominator is not None and len(df) != expected_denominator:
        checks.append(
            _invalid(
                name,
                path,
                "claim output row count must equal the complete docking coverage "
                f"denominator: rows={len(df)} denominator={expected_denominator}",
            )
        )


def _verify_compound(run_dir: Path, checks: list[OutputCheck]) -> dict[str, Any] | None:
    path = run_dir / "01_input" / "compound_canonical.json"
    payload, check = _read_json(path, "compound metadata")
    if not _append_if_bad(checks, check):
        return None
    input_type = _required_text_field(payload, "input_type", "compound metadata", path)
    smiles = _required_text_field(payload, "canonical_smiles", "compound metadata", path)
    inchikey = _required_text_field(payload, "inchikey", "compound metadata", path)
    if input_type is None:
        checks.append(_invalid("compound metadata", path, "missing non-empty input_type"))
    elif input_type not in {"smiles", "sdf"}:
        checks.append(_invalid("compound metadata", path, "invalid input_type"))
    if smiles is None:
        checks.append(_invalid("compound metadata", path, "missing non-empty canonical_smiles"))
        return None
    canonical_smiles = _canonical_smiles_value(smiles)
    if canonical_smiles is None:
        checks.append(_invalid("compound metadata", path, f"invalid canonical_smiles: {smiles!r}"))
        return None
    if smiles != canonical_smiles:
        checks.append(
            _invalid(
                "compound metadata",
                path,
                f"canonical_smiles is not canonical: {smiles!r} != {canonical_smiles!r}",
            )
        )
        return None
    if inchikey is None:
        checks.append(_invalid("compound metadata", path, "missing non-empty inchikey"))
        return None
    mol = Chem.MolFromSmiles(smiles)
    expected_inchikey = Chem.MolToInchiKey(mol) if mol is not None else None
    if inchikey != expected_inchikey:
        checks.append(
            _invalid(
                "compound metadata",
                path,
                f"inchikey does not match canonical_smiles: {inchikey!r} != {expected_inchikey!r}",
            )
        )
        return None
    result: dict[str, Any] = {
        "input_type": input_type,
        "canonical_smiles": smiles,
        "inchikey": inchikey,
    }
    if input_type == "smiles":
        if _text_or_none(payload.get("input_sdf")) is not None:
            checks.append(
                _invalid(
                    "compound metadata",
                    path,
                    "input_type smiles cannot include input_sdf",
                )
            )
        input_smiles = _required_text_field(
            payload,
            "input_smiles",
            "compound metadata",
            path,
        )
        input_canonical_smiles = _required_text_field(
            payload,
            "input_canonical_smiles",
            "compound metadata",
            path,
        )
        if input_smiles is None:
            checks.append(_invalid("compound metadata", path, "missing non-empty input_smiles"))
        else:
            input_canonical = _canonical_smiles_value(input_smiles)
            if input_canonical is None:
                checks.append(
                    _invalid(
                        "compound metadata",
                        path,
                        f"invalid input_smiles: {input_smiles!r}",
                    )
                )
            elif input_canonical_smiles != input_canonical:
                checks.append(
                    _invalid(
                        "compound metadata",
                        path,
                        "input_canonical_smiles does not match input_smiles",
                    )
                )
        result["input_smiles"] = input_smiles
        result["input_canonical_smiles"] = input_canonical_smiles
    if input_type == "sdf":
        for field in ("input_smiles", "input_canonical_smiles"):
            if _text_or_none(payload.get(field)) is not None:
                checks.append(
                    _invalid(
                        "compound metadata",
                        path,
                        f"input_type sdf cannot include {field}",
                    )
                )
        input_sdf = _required_text_field(payload, "input_sdf", "compound metadata", path)
        if input_sdf is None:
            checks.append(_invalid("compound metadata", path, "missing non-empty input_sdf"))
        result["input_sdf"] = input_sdf
    return result


def _verify_admet_prediction_payload(
    predictions: object,
    checks: list[OutputCheck],
    name: str,
    path: Path,
    *,
    allow_degraded: bool,
) -> None:
    if not isinstance(predictions, dict):
        if allow_degraded and predictions is None:
            return
        checks.append(_invalid(name, path, "missing ADMET-AI predictions object"))
        return
    if not predictions:
        if allow_degraded:
            return
        checks.append(_invalid(name, path, "empty ADMET-AI predictions"))
        return
    missing = sorted(ADMET_CORE_ENDPOINTS - set(predictions))
    if missing and not allow_degraded:
        checks.append(_invalid(name, path, f"missing ADMET-AI endpoints {missing}"))
        return
    fields_to_check = ADMET_CORE_ENDPOINTS & set(predictions)
    for field in sorted(fields_to_check):
        bad = _numeric_value(
            predictions[field],
            name,
            path,
            f"admet_ai_predictions.{field}",
            fraction=field in ADMET_PROBABILITY_ENDPOINTS,
        )
        if bad is not None:
            checks.append(bad)


def _verify_admet_ai_source(
    run_dir: Path,
    checks: list[OutputCheck],
    report_predictions: object,
    *,
    compound_smiles: str | None,
    allow_degraded: bool,
) -> None:
    path = run_dir / "02_admet" / "admet_ai.json"
    payload, check = _read_json(path, "ADMET-AI source predictions")
    if not _append_if_bad(checks, check):
        return
    status = payload.get("status")
    if status != "ok":
        if not allow_degraded:
            checks.append(_invalid("ADMET-AI source predictions", path, f"status must be ok: {status!r}"))
        return
    _verify_source_smiles(
        payload,
        checks,
        "ADMET-AI source predictions",
        path,
        compound_smiles,
    )
    predictions = payload.get("predictions")
    _verify_admet_prediction_payload(
        predictions,
        checks,
        "ADMET-AI source predictions",
        path,
        allow_degraded=allow_degraded,
    )
    if not isinstance(report_predictions, dict) or not isinstance(predictions, dict):
        return
    for field in sorted(ADMET_CORE_ENDPOINTS & set(report_predictions) & set(predictions)):
        try:
            report_value = float(report_predictions[field])
            source_value = float(predictions[field])
        except (TypeError, ValueError):
            continue
        if math.isfinite(report_value) and math.isfinite(source_value) and abs(report_value - source_value) > 1e-9:
            checks.append(
                _invalid(
                    "ADMET-AI source predictions",
                    path,
                    f"source/report mismatch for endpoint {field}",
                )
            )
            break


def _verify_structural_alert_source(
    run_dir: Path,
    checks: list[OutputCheck],
    report_alerts: object,
    *,
    compound_smiles: str | None,
) -> None:
    path = run_dir / "02_admet" / "structural_alerts.json"
    payload, check = _read_json(path, "structural-alert source evidence")
    if not _append_if_bad(checks, check):
        return
    _verify_source_smiles(
        payload,
        checks,
        "structural-alert source evidence",
        path,
        compound_smiles,
    )
    if not isinstance(payload.get("matches"), dict):
        checks.append(_invalid("structural-alert source evidence", path, "missing matches object"))
    for key in ("any_pains", "any_brenk", "any_nih"):
        if not isinstance(payload.get(key), bool):
            checks.append(_invalid("structural-alert source evidence", path, f"missing boolean {key}"))
            break
        if isinstance(report_alerts, dict):
            report_value = report_alerts.get(key)
            if not isinstance(report_value, bool):
                checks.append(
                    _invalid(
                        "ADMET and skin-sens report",
                        run_dir / "02_admet" / "admet_report.json",
                        f"structural_alerts.{key} must be boolean",
                    )
                )
                break
            if payload.get(key) != report_value:
                checks.append(
                    _invalid(
                        "structural-alert source evidence",
                        path,
                        f"source/report mismatch for structural-alert flag {key}",
                    )
                )
                break
    if isinstance(report_alerts, dict) and isinstance(report_alerts.get("matches"), dict):
        if payload.get("matches") != report_alerts.get("matches"):
            checks.append(
                _invalid(
                    "structural-alert source evidence",
                    path,
                    "source/report mismatch for structural-alert matches",
                )
            )


def _skin_sens_source_call(payload: dict[str, Any], model: str) -> tuple[object, object]:
    if model == "pred_skin":
        pred_skin = payload.get("pred_skin")
        probability = pred_skin.get("probability") if isinstance(pred_skin, dict) else None
        return payload.get("consensus_call"), probability
    return payload.get("skin_sens_call"), payload.get("skin_sens_probability")


def _verify_skin_sens_source(
    run_dir: Path,
    checks: list[OutputCheck],
    report_skin_sens: dict[str, Any],
    *,
    compound_smiles: str | None,
    allow_degraded: bool,
) -> None:
    for model in SKIN_SENS_MODELS:
        path = run_dir / "02_admet" / f"{model}.json"
        name = f"{model} skin-sens source evidence"
        payload, check = _read_json(path, name)
        if not _append_if_bad(checks, check):
            continue
        status = payload.get("status")
        if status != "ok":
            if not allow_degraded:
                checks.append(_invalid(name, path, f"status must be ok: {status!r}"))
            continue
        _verify_source_smiles(payload, checks, name, path, compound_smiles)
        call, probability = _skin_sens_source_call(payload, model)
        if not isinstance(call, str) or call.strip() not in SKIN_SENS_CALLS:
            checks.append(_invalid(name, path, f"{model} call must be positive or negative"))
            continue
        bad_probability = _numeric_value(
            probability,
            name,
            path,
            f"{model} skin-sens probability",
            fraction=True,
        )
        if bad_probability is not None:
            checks.append(bad_probability)
        report_call = report_skin_sens.get(model)
        if isinstance(report_call, str) and report_call.strip() and report_call.strip() != call.strip():
            checks.append(_invalid(name, path, f"source/report mismatch for {model} call"))


def _verify_admet(
    run_dir: Path,
    checks: list[OutputCheck],
    *,
    compound_smiles: str | None,
    allow_degraded: bool,
) -> str | None:
    path = run_dir / "02_admet" / "admet_report.json"
    payload, check = _read_json(path, "ADMET and skin-sens report")
    if not _append_if_bad(checks, check):
        return None
    skin_sens = payload.get("skin_sens")
    if not isinstance(skin_sens, dict):
        checks.append(_invalid("ADMET and skin-sens report", path, "missing object skin_sens"))
        return None
    decision = skin_sens.get("decision")
    if not isinstance(decision, str) or decision.strip() not in SAFETY_DECISIONS:
        checks.append(
            _invalid(
                "ADMET and skin-sens report",
                path,
                f"skin_sens.decision must be one of {sorted(SAFETY_DECISIONS)}",
            )
        )
        return None
    if skin_sens.get("degraded") and not allow_degraded:
        checks.append(_invalid("ADMET and skin-sens report", path, "degraded skin-sens evidence"))
    missing_models = skin_sens.get("missing_models")
    if not isinstance(missing_models, list):
        checks.append(_invalid("ADMET and skin-sens report", path, "missing_models must be a list"))
    elif missing_models and not allow_degraded:
        checks.append(_invalid("ADMET and skin-sens report", path, "missing skin-sens model evidence"))
    for model in SKIN_SENS_MODELS:
        call = skin_sens.get(model)
        if not isinstance(call, str) or call.strip() not in SKIN_SENS_CALLS:
            if not (allow_degraded and isinstance(missing_models, list) and model in missing_models):
                checks.append(_invalid("ADMET and skin-sens report", path, f"skin_sens.{model} must be positive or negative"))
                break
    report_predictions = payload.get("admet_ai_predictions")
    _verify_admet_prediction_payload(
        report_predictions,
        checks,
        "ADMET and skin-sens report",
        path,
        allow_degraded=allow_degraded,
    )
    structural_alerts = payload.get("structural_alerts")
    if not isinstance(structural_alerts, dict):
        checks.append(_invalid("ADMET and skin-sens report", path, "missing structural_alerts object"))
    else:
        for key in ("any_pains", "any_brenk", "any_nih"):
            if not isinstance(structural_alerts.get(key), bool):
                checks.append(
                    _invalid(
                        "ADMET and skin-sens report",
                        path,
                        f"structural_alerts.{key} must be boolean",
                    )
                )
                break
        _verify_structural_alert_source(
            run_dir,
            checks,
            structural_alerts,
            compound_smiles=compound_smiles,
        )
    _verify_admet_ai_source(
        run_dir,
        checks,
        report_predictions,
        compound_smiles=compound_smiles,
        allow_degraded=allow_degraded,
    )
    _verify_skin_sens_source(
        run_dir,
        checks,
        skin_sens,
        compound_smiles=compound_smiles,
        allow_degraded=allow_degraded,
    )

    decision_path = run_dir / "02_admet" / "skin_sens_decision.txt"
    text, text_check = _read_text(decision_path, "skin-sens decision")
    if _append_if_bad(checks, text_check):
        decision_text = _first_nonempty_line(text)
        if decision_text is None:
            checks.append(_invalid("skin-sens decision", decision_path, "contains no decision line"))
        elif decision_text != decision.strip():
            checks.append(
                _invalid(
                    "skin-sens decision",
                    decision_path,
                    "does not match admet_report.json skin_sens.decision",
                )
            )
    return decision.strip()


def _verify_reference_json(
    run_dir: Path,
    checks: list[OutputCheck],
    rel_path: str,
    name: str,
    *,
    allow_degraded: bool,
) -> dict[str, Any] | None:
    path = run_dir / rel_path
    payload, check = _read_json(path, name)
    if not _append_if_bad(checks, check):
        return None
    status = payload.get("reference_status")
    if not isinstance(status, str) or not status.strip():
        checks.append(_invalid(name, path, "missing non-empty reference_status"))
    elif status.strip() != "ok" and not allow_degraded:
        checks.append(_invalid(name, path, f"reference_status must be ok: {status.strip()}"))
    return payload


def _integer_value(value: object, name: str, path: Path, field: str) -> OutputCheck | None:
    bad = _numeric_value(value, name, path, field)
    if bad is not None:
        return bad
    numeric = float(value)  # type: ignore[arg-type]
    if not numeric.is_integer() or numeric < 0:
        return _invalid(name, path, f"{field} must be a non-negative integer")
    return None


def _optional_float(value: object) -> float | None:
    if value is None or _bool_like(value):
        return None
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _optional_int(value: object) -> int | None:
    parsed = _optional_float(value)
    if parsed is None or not parsed.is_integer() or parsed < 0:
        return None
    return int(parsed)


def _expected_cosmetic_drug_summary(
    run_dir: Path,
    *,
    decision: str | None,
    policy: str | None,
) -> dict[str, Any] | None:
    cosing, cosing_check = _read_json(
        run_dir / "02b_cosmetic_drug" / "cosing_match.json",
        "CosIng annotation cross-check",
    )
    drug, drug_check = _read_json(
        run_dir / "02b_cosmetic_drug" / "drug_warnings.json",
        "drug-avoidance warnings cross-check",
    )
    if cosing_check.status != "ok" or drug_check.status != "ok":
        return None
    return {
        "decision": decision,
        "drug_policy": policy,
        "cosing_level": cosing.get("level"),
        "inci": cosing.get("inci"),
        "cosing_functions": cosing.get("functions", []),
        "cosing_tanimoto": _optional_float(cosing.get("tanimoto"))
        if "tanimoto" in cosing
        else None,
        "max_tanimoto_to_approved_drug": _optional_float(
            drug.get("max_tanimoto_to_approved_drug")
        )
        if "max_tanimoto_to_approved_drug" in drug
        else None,
        "n_warnings": _optional_int(drug.get("n_warnings"))
        if "n_warnings" in drug
        else None,
    }


def _verify_cosmetic_drug(
    run_dir: Path,
    checks: list[OutputCheck],
    *,
    allow_degraded: bool,
) -> tuple[str | None, str | None]:
    cosing = _verify_reference_json(
        run_dir,
        checks,
        "02b_cosmetic_drug/cosing_match.json",
        "CosIng annotation",
        allow_degraded=allow_degraded,
    )
    if isinstance(cosing, dict):
        if "functions" in cosing and not isinstance(cosing.get("functions"), list):
            checks.append(
                _invalid(
                    "CosIng annotation",
                    run_dir / "02b_cosmetic_drug" / "cosing_match.json",
                    "functions must be a list",
                )
            )
        if "tanimoto" in cosing:
            bad = _numeric_value(
                cosing.get("tanimoto"),
                "CosIng annotation",
                run_dir / "02b_cosmetic_drug" / "cosing_match.json",
                "tanimoto",
                fraction=True,
            )
            if bad is not None:
                checks.append(bad)
    drug = _verify_reference_json(
        run_dir,
        checks,
        "02b_cosmetic_drug/drug_warnings.json",
        "drug-avoidance warnings",
        allow_degraded=allow_degraded,
    )
    if isinstance(drug, dict):
        if "max_tanimoto_to_approved_drug" in drug:
            bad = _numeric_value(
                drug.get("max_tanimoto_to_approved_drug"),
                "drug-avoidance warnings",
                run_dir / "02b_cosmetic_drug" / "drug_warnings.json",
                "max_tanimoto_to_approved_drug",
                fraction=True,
            )
            if bad is not None:
                checks.append(bad)
        if "n_warnings" in drug:
            bad = _integer_value(
                drug.get("n_warnings"),
                "drug-avoidance warnings",
                run_dir / "02b_cosmetic_drug" / "drug_warnings.json",
                "n_warnings",
            )
            if bad is not None:
                checks.append(bad)
        if "warnings" in drug and not isinstance(drug.get("warnings"), list):
            checks.append(
                _invalid(
                    "drug-avoidance warnings",
                    run_dir / "02b_cosmetic_drug" / "drug_warnings.json",
                    "warnings must be a list",
                )
            )
    path = run_dir / "02b_cosmetic_drug" / "cosmetic_drug_decision.txt"
    text, check = _read_text(path, "cosmetic/drug decision")
    if not _append_if_bad(checks, check):
        return None, None
    try:
        decision, policy = parse_decision_and_policy(text)
    except CosmeticDrugContractError as exc:
        checks.append(
            _invalid(
                "cosmetic/drug decision",
                path,
                str(exc),
            )
        )
        return None, None
    if isinstance(drug, dict):
        try:
            validate_decision_matches_drug_warnings(
                decision=decision,
                policy=policy,
                drug_warnings=drug,
            )
        except CosmeticDrugContractError as exc:
            checks.append(_invalid("cosmetic/drug decision", path, str(exc)))
    return decision, policy


def _verify_target_ranking(
    run_dir: Path,
    checks: list[OutputCheck],
    *,
    mode: str,
    min_targets: int,
) -> None:
    path = run_dir / "03_targets" / "ranked_targets_v3_with_efficacy.csv"
    df, check = _read_csv(path, "skin-weighted target ranking with efficacy")
    if not _append_if_bad(checks, check):
        return
    required = {"target_id", "final_score", "skin_score", "docking_rrf", "source_count", "sources"}
    missing = sorted(required - set(df.columns))
    if missing:
        checks.append(_invalid("skin-weighted target ranking with efficacy", path, f"missing columns {missing}"))
        return
    if len(df) < min_targets:
        checks.append(
            _invalid(
                "skin-weighted target ranking with efficacy",
                path,
                f"expected at least {min_targets} target row(s), found {len(df)}",
            )
        )
    checks.extend(_validate_target_ids(df, "skin-weighted target ranking with efficacy", path))
    for column in sorted(FRACTION_COLUMNS & set(df.columns)):
        bad = _numeric_series(
            df,
            column,
            "skin-weighted target ranking with efficacy",
            path,
            fraction=True,
        )
        if bad is not None:
            checks.append(bad)
    daina_structural = mode == "fast" and _uses_daina_structural_contract(run_dir)
    min_source_count = 1 if daina_structural else (2 if mode == "fast" else 3)
    checks.extend(
        _validate_source_support(
            df,
            "skin-weighted target ranking with efficacy",
            path,
            min_source_count=min_source_count,
        )
    )
    if "final_score" in df.columns and pd.api.types.is_numeric_dtype(df["final_score"]):
        if not df["final_score"].is_monotonic_decreasing:
            checks.append(
                _invalid(
                    "skin-weighted target ranking with efficacy",
                    path,
                    "final_score must be sorted descending",
                )
            )
    efficacy_cols = [col for col in df.columns if col.startswith("efficacy_top")]
    if not efficacy_cols:
        checks.append(
            _invalid(
                "skin-weighted target ranking with efficacy",
                path,
                "missing efficacy_top* columns",
            )
        )
        return
    if not daina_structural:
        for idx, row in df.head(min(10, len(df))).iterrows():
            if not any(not pd.isna(row[col]) and str(row[col]).strip() for col in efficacy_cols):
                checks.append(
                    _invalid(
                        "skin-weighted target ranking with efficacy",
                        path,
                        f"top row {int(idx)} lacks efficacy evidence",
                    )
                )
                break


def _markdown_display_value(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, list):
        return ", ".join(_markdown_display_value(item) for item in value) if value else "none"
    return str(value)


def _verify_markdown_summary(
    run_dir: Path,
    checks: list[OutputCheck],
    payload: dict[str, Any],
    expected_admet_risk: dict[str, Any],
) -> None:
    path = run_dir / "run_summary.md"
    text, check = _read_text(path, "user-facing markdown run summary")
    if not _append_if_bad(checks, check):
        return
    text = text.replace(r"\|", "|")
    if not text:
        checks.append(_invalid("user-facing markdown run summary", path, "contains no text"))
        return

    def require_value(label: str, value: object) -> None:
        if value is None:
            return
        token = _markdown_display_value(value).strip()
        if token and token not in text:
            checks.append(
                _invalid(
                    "user-facing markdown run summary",
                    path,
                    f"missing {label}: {token}",
                )
            )

    def require_entry_value(label: str, prefix: str, value: object) -> None:
        if value is None:
            return
        display = _markdown_display_value(value).strip()
        if not display:
            return
        token = f"- {prefix}: {display}"
        if token not in text:
            checks.append(
                _invalid(
                    "user-facing markdown run summary",
                    path,
                    f"missing {label}: {display}",
                )
            )

    def require_section_entry_value(
        section: str,
        label: str,
        prefix: str,
        value: object,
        *,
        code: bool = False,
        require_none: bool = False,
    ) -> None:
        if value is None and not require_none:
            return
        display = _markdown_display_value(value).strip()
        if not display:
            return
        rendered = f"`{display}`" if code else display
        heading = f"## {section}"
        start = text.find(heading)
        if start < 0:
            checks.append(
                _invalid(
                    "user-facing markdown run summary",
                    path,
                    f"missing section {section}",
                )
            )
            return
        next_heading = text.find("\n## ", start + len(heading))
        section_text = text[start:] if next_heading < 0 else text[start:next_heading]
        token = f"- {prefix}: {rendered}"
        if token not in section_text:
            checks.append(
                _invalid(
                    "user-facing markdown run summary",
                    path,
                    f"missing {label}: {display}",
                )
            )

    def require_table_row(label: str, values: list[object]) -> None:
        expected = [_markdown_display_value(value).strip() for value in values]
        for line in text.splitlines():
            candidate = line.strip()
            if not candidate.startswith("|") or not candidate.endswith("|"):
                continue
            cells = [cell.strip() for cell in candidate.strip("|").split("|")]
            if cells == expected:
                return
        checks.append(
            _invalid(
                "user-facing markdown run summary",
                path,
                f"missing {label}: {' | '.join(expected)}",
            )
        )

    require_section_entry_value(
        "Run",
        "run_id",
        "Run",
        payload.get("run_id"),
        require_none=True,
    )
    require_section_entry_value(
        "Run",
        "preset",
        "Preset",
        payload.get("preset"),
        require_none=True,
    )
    require_section_entry_value(
        "Run",
        "mode",
        "Mode",
        payload.get("mode"),
        require_none=True,
    )
    compound = payload.get("compound")
    if isinstance(compound, dict):
        require_section_entry_value(
            "Run",
            "input_type",
            "Input type",
            compound.get("input_type"),
            require_none=True,
        )
        require_section_entry_value(
            "Run",
            "input_smiles",
            "Input SMILES",
            compound.get("input_smiles"),
            code=True,
            require_none=True,
        )
        require_section_entry_value(
            "Run",
            "input_canonical_smiles",
            "Input canonical SMILES",
            compound.get("input_canonical_smiles"),
            code=True,
            require_none=True,
        )
        require_section_entry_value(
            "Run",
            "input_sdf",
            "Input SDF",
            compound.get("input_sdf"),
            code=True,
            require_none=True,
        )
        require_section_entry_value(
            "Run",
            "canonical_smiles",
            "Compound",
            compound.get("canonical_smiles"),
            code=True,
            require_none=True,
        )
        require_section_entry_value(
            "Run",
            "inchikey",
            "InChIKey",
            compound.get("inchikey"),
            require_none=True,
        )

    safety = payload.get("safety")
    if isinstance(safety, dict):
        require_section_entry_value(
            "Safety And ADMET",
            "safety skin sensitization",
            "Skin sensitization",
            safety.get("skin_sens_decision"),
            require_none=True,
        )
        require_section_entry_value(
            "Safety And ADMET",
            "safety degraded evidence",
            "Degraded evidence",
            safety.get("degraded"),
            require_none=True,
        )
        require_section_entry_value(
            "Safety And ADMET",
            "safety missing skin-sens models",
            "Missing skin-sens models",
            safety.get("missing_models"),
            require_none=True,
        )
        skin_sens_evidence = safety.get("skin_sens_evidence")
        if isinstance(skin_sens_evidence, list):
            for idx, row in enumerate(skin_sens_evidence, start=1):
                if not isinstance(row, dict):
                    continue
                model = row.get("model")
                expected_tokens = [
                    _markdown_display_value(row.get(field))
                    for field in ("model", "status", "call", "probability")
                    if row.get(field) is not None
                ]
                candidate_lines = [
                    line.replace(r"\|", "|")
                    for line in text.splitlines()
                    if model is not None and str(model) in line and "|" in line
                ]
                if not any(
                    all(token in line for token in expected_tokens if token)
                    for line in candidate_lines
                ):
                    checks.append(
                        _invalid(
                            "user-facing markdown run summary",
                            path,
                            "missing skin-sens evidence row "
                            f"{idx}: {' | '.join(expected_tokens)}",
                        )
                    )

    skin_toxicity = payload.get("skin_toxicity")
    if isinstance(skin_toxicity, dict):
        require_section_entry_value(
            "Skin Toxicity",
            "skin_toxicity decision",
            "Decision",
            skin_toxicity.get("decision"),
        )
        require_section_entry_value(
            "Skin Toxicity",
            "skin_toxicity level",
            "Toxicity level",
            skin_toxicity.get("toxicity_level"),
        )
        require_section_entry_value(
            "Skin Toxicity",
            "skin sensitization",
            "Skin sensitization",
            skin_toxicity.get("skin_sens_decision"),
        )
        require_section_entry_value(
            "Skin Toxicity",
            "Skin_Reaction risk",
            "Skin_Reaction risk",
            skin_toxicity.get("skin_reaction_risk_level"),
        )
        require_section_entry_value(
            "Skin Toxicity",
            "Skin_Reaction value",
            "Skin_Reaction value",
            skin_toxicity.get("skin_reaction_value"),
        )
        require_section_entry_value(
            "Skin Toxicity",
            "structural alerts present",
            "Structural alerts present",
            skin_toxicity.get("structural_alerts_present"),
        )
        require_section_entry_value(
            "Skin Toxicity",
            "structural alert flags",
            "Structural alert flags",
            skin_toxicity.get("structural_alert_flags"),
        )
        require_section_entry_value(
            "Skin Toxicity",
            "degraded evidence",
            "Degraded evidence",
            skin_toxicity.get("degraded"),
        )
        require_section_entry_value(
            "Skin Toxicity",
            "missing skin-sens models",
            "Missing skin-sens models",
            skin_toxicity.get("missing_models"),
        )
        reasons = skin_toxicity.get("reasons")
        if isinstance(reasons, list):
            for reason in reasons:
                require_value("skin_toxicity reason", reason)

    cosmetic_drug = payload.get("cosmetic_drug")
    if isinstance(cosmetic_drug, dict):
        require_section_entry_value(
            "Cosmetic And Drug",
            "cosmetic/drug decision",
            "Decision",
            cosmetic_drug.get("decision"),
            require_none=True,
        )
        require_section_entry_value(
            "Cosmetic And Drug",
            "drug policy",
            "Drug policy",
            cosmetic_drug.get("drug_policy"),
            require_none=True,
        )
        require_section_entry_value(
            "Cosmetic And Drug",
            "CosIng level",
            "CosIng level",
            cosmetic_drug.get("cosing_level"),
            require_none=True,
        )
        require_section_entry_value(
            "Cosmetic And Drug",
            "INCI",
            "INCI",
            cosmetic_drug.get("inci"),
            require_none=True,
        )
        require_section_entry_value(
            "Cosmetic And Drug",
            "drug warnings",
            "Drug warnings",
            cosmetic_drug.get("n_warnings"),
            require_none=True,
        )

    overall = payload.get("overall_decision")
    if isinstance(overall, dict):
        require_section_entry_value(
            "Overall Decision",
            "overall_decision",
            "Decision",
            overall.get("decision"),
            require_none=True,
        )
        require_section_entry_value(
            "Overall Decision",
            "recommended_action",
            "Recommended action",
            overall.get("recommended_action"),
            require_none=True,
        )
        require_section_entry_value(
            "Overall Decision",
            "claimable",
            "Claimable",
            overall.get("claimable"),
            require_none=True,
        )
        require_section_entry_value(
            "Overall Decision",
            "requires_human_review",
            "Requires human review",
            overall.get("requires_human_review"),
            require_none=True,
        )

    endpoints = expected_admet_risk.get("endpoints")
    if isinstance(endpoints, dict):
        for endpoint, expected in endpoints.items():
            require_value(f"ADMET endpoint {endpoint}", endpoint)
            if isinstance(expected, dict):
                require_value(f"ADMET risk level {endpoint}", expected.get("risk_level"))
                require_table_row(
                    f"ADMET risk row {endpoint}",
                    [endpoint, expected.get("value"), expected.get("risk_level")],
                )
        require_section_entry_value(
            "Safety And ADMET",
            "high ADMET risk endpoints",
            "High risk endpoints",
            expected_admet_risk.get("high_risk_endpoints"),
        )
        require_section_entry_value(
            "Safety And ADMET",
            "moderate ADMET risk endpoints",
            "Moderate risk endpoints",
            expected_admet_risk.get("moderate_risk_endpoints"),
        )

    if isinstance(safety, dict):
        admet_metrics = safety.get("admet_metrics")
        if isinstance(admet_metrics, dict):
            for metric in SUMMARY_ADMET_METRICS:
                if metric in admet_metrics:
                    require_table_row(
                        f"ADMET metric row {metric}",
                        [metric, admet_metrics.get(metric)],
                    )

    target_prediction = payload.get("target_prediction")
    if isinstance(target_prediction, dict):
        require_section_entry_value(
            "Top Targets",
            "target count",
            "Ranked targets",
            target_prediction.get("n_targets"),
            require_none=True,
        )
        require_section_entry_value(
            "Top Targets",
            "screened target count",
            "Screened target candidates",
            target_prediction.get("screened_target_count"),
            require_none=True,
        )
        screening_counts = target_prediction.get("screening_counts")
        if isinstance(screening_counts, dict):
            for key, value in screening_counts.items():
                require_table_row(f"screening stage row {key}", [key, value])
        top_targets = target_prediction.get("top_targets")
        if isinstance(top_targets, list):
            for rank, target in enumerate(top_targets, start=1):
                if not isinstance(target, dict):
                    continue
                require_value(f"top target {rank} target_id", target.get("target_id"))
                require_value(f"top target {rank} gene_symbol", target.get("gene_symbol"))
                require_value(f"top target {rank} protein_name", target.get("protein_name"))
                require_value(f"top target {rank} skin_tier", target.get("skin_tier"))
                target_id = target.get("target_id")
                if target_id is None:
                    continue
                expected_tokens = [str(target_id)]
                for field in (
                    "final_score",
                    "skin_score",
                    "skin_tier",
                    "docking_rrf",
                    # The retrieval evidence behind a hit was computed and then
                    # dropped before publication once already; contract-check it
                    # so it cannot silently disappear again.
                    "daina_max_tanimoto",
                    "daina_supporting_molecule_id",
                    "daina_known_ligand_count",
                    "sources",
                    "efficacy",
                ):
                    if field in target:
                        expected_tokens.append(_markdown_display_value(target.get(field)))
                candidate_lines = [
                    line.replace(r"\|", "|")
                    for line in text.splitlines()
                    if str(target_id) in line and "|" in line
                ]
                if not any(
                    all(token in line for token in expected_tokens if token)
                    for line in candidate_lines
                ):
                    checks.append(
                        _invalid(
                            "user-facing markdown run summary",
                            path,
                            "missing top target "
                            f"{rank} row details: {' | '.join(expected_tokens)}",
                        )
                    )
    skin_binding = payload.get("skin_specialized_binding")
    if isinstance(skin_binding, dict):
        require_section_entry_value(
            "Skin-Specialized Binding",
            "skin binding context",
            "Context",
            skin_binding.get("context"),
            require_none=True,
        )
        require_section_entry_value(
            "Skin-Specialized Binding",
            "skin context decision",
            "Skin context decision",
            skin_binding.get("skin_context_decision"),
            require_none=True,
        )
        require_section_entry_value(
            "Skin-Specialized Binding",
            "skin context supported",
            "Skin context supported",
            skin_binding.get("skin_context_supported"),
            require_none=True,
        )
        require_section_entry_value(
            "Skin-Specialized Binding",
            "skin expression supported",
            "Skin expression supported",
            skin_binding.get("skin_expression_supported"),
            require_none=True,
        )
        require_section_entry_value(
            "Skin-Specialized Binding",
            "skin efficacy supported",
            "Skin efficacy supported",
            skin_binding.get("skin_efficacy_supported"),
            require_none=True,
        )
        require_section_entry_value(
            "Skin-Specialized Binding",
            "skin binding top target",
            "Top predicted binding target",
            skin_binding.get("top_target_id"),
            require_none=True,
        )
        require_section_entry_value(
            "Skin-Specialized Binding",
            "skin binding top target gene",
            "Top target gene",
            skin_binding.get("top_target_gene_symbol"),
            require_none=True,
        )
        require_section_entry_value(
            "Skin-Specialized Binding",
            "skin binding top target protein",
            "Top target protein",
            skin_binding.get("top_target_protein_name"),
            require_none=True,
        )
        require_section_entry_value(
            "Skin-Specialized Binding",
            "skin binding top final score",
            "Top target final score",
            skin_binding.get("top_target_final_score"),
            require_none=True,
        )
        require_section_entry_value(
            "Skin-Specialized Binding",
            "skin binding top docking RRF",
            "Top target docking RRF",
            skin_binding.get("top_target_docking_rrf"),
            require_none=True,
        )
        require_section_entry_value(
            "Skin-Specialized Binding",
            "skin binding top source count",
            "Top target source count",
            skin_binding.get("top_target_source_count"),
            require_none=True,
        )
        require_section_entry_value(
            "Skin-Specialized Binding",
            "skin binding top sources",
            "Top target sources",
            skin_binding.get("top_target_sources"),
            require_none=True,
        )
        require_section_entry_value(
            "Skin-Specialized Binding",
            "skin binding top skin score",
            "Top target skin score",
            skin_binding.get("top_target_skin_score"),
            require_none=True,
        )
        require_section_entry_value(
            "Skin-Specialized Binding",
            "skin binding top target tier",
            "Top target skin tier",
            skin_binding.get("top_target_skin_tier"),
            require_none=True,
        )
        require_section_entry_value(
            "Skin-Specialized Binding",
            "skin binding top efficacy",
            "Top target skin efficacy",
            skin_binding.get("top_target_efficacy"),
            require_none=True,
        )
        require_section_entry_value(
            "Skin-Specialized Binding",
            "skin binding top expression support",
            "Top target skin expression supported",
            skin_binding.get("top_target_skin_expression_supported"),
            require_none=True,
        )
        require_section_entry_value(
            "Skin-Specialized Binding",
            "skin binding top efficacy support",
            "Top target skin efficacy supported",
            skin_binding.get("top_target_skin_efficacy_supported"),
            require_none=True,
        )
        require_section_entry_value(
            "Skin-Specialized Binding",
            "skin binding top context support",
            "Top target skin context supported",
            skin_binding.get("top_target_skin_context_supported"),
            require_none=True,
        )
        require_section_entry_value(
            "Skin-Specialized Binding",
            "skin binding efficacy count",
            "Top targets with skin-efficacy evidence",
            skin_binding.get("top_targets_with_skin_efficacy"),
            require_none=True,
        )
        most_skin_relevant = skin_binding.get("most_skin_relevant_target")
        if isinstance(most_skin_relevant, dict):
            most_label = _markdown_display_value(most_skin_relevant.get("target_id"))
            gene = most_skin_relevant.get("gene_symbol")
            if gene:
                most_label = f"{gene} ({most_label})"
            require_section_entry_value(
                "Skin-Specialized Binding",
                "most skin-relevant target",
                "Most skin-relevant top target",
                most_label,
                require_none=True,
            )
            for field, label, detail in (
                ("gene_symbol", "Most skin-relevant gene", "most skin-relevant gene"),
                (
                    "protein_name",
                    "Most skin-relevant protein",
                    "most skin-relevant protein",
                ),
            ):
                require_section_entry_value(
                    "Skin-Specialized Binding",
                    detail,
                    label,
                    most_skin_relevant.get(field),
                    require_none=True,
                )
            for field, label in (
                ("final_score", "Most skin-relevant final score"),
                ("docking_rrf", "Most skin-relevant docking RRF"),
                ("source_count", "Most skin-relevant source count"),
                ("sources", "Most skin-relevant sources"),
                ("skin_score", "Most skin-relevant skin score"),
                ("skin_tier", "Most skin-relevant skin tier"),
                ("efficacy", "Most skin-relevant efficacy"),
            ):
                require_section_entry_value(
                    "Skin-Specialized Binding",
                    f"most skin-relevant {field}",
                    label,
                    most_skin_relevant.get(field),
                    require_none=True,
                )
        reasons = skin_binding.get("skin_context_reasons")
        if isinstance(reasons, list):
            for reason in reasons:
                require_value("skin context reason", reason)

    artifacts = payload.get("artifacts")
    if isinstance(artifacts, dict):
        require_value("source artifacts section", "Source Artifacts")
        for key, value in sorted(artifacts.items()):
            require_section_entry_value(
                "Source Artifacts",
                f"artifact path {key}",
                key,
                value,
                code=True,
                require_none=True,
            )


def _verify_run_summary(
    run_dir: Path,
    checks: list[OutputCheck],
    *,
    preset: str,
    mode: str,
    compound_metadata: dict[str, Any] | None,
    compound_smiles: str | None,
    compound_inchikey: str | None,
    safety_decision: str | None,
    cosmetic_decision: str | None,
    cosmetic_policy: str | None,
    target_metadata: dict[str, dict[str, str]],
) -> None:
    path = run_dir / "run_summary.json"
    payload, check = _read_json(path, "user-facing run summary")
    if not _append_if_bad(checks, check):
        return
    if payload.get("schema_version") != SUMMARY_SCHEMA:
        checks.append(_invalid("user-facing run summary", path, "invalid schema_version"))
    if payload.get("run_id") != run_dir.name:
        checks.append(_invalid("user-facing run summary", path, "run_id does not match run directory name"))
    summary_preset = payload.get("preset")
    if not isinstance(summary_preset, str) or summary_preset not in PRESET_DEPTH:
        checks.append(_invalid("user-facing run summary", path, "invalid preset"))
    elif PRESET_DEPTH[summary_preset] < PRESET_DEPTH[preset]:
        checks.append(_invalid("user-facing run summary", path, "preset is shallower than verifier preset"))
    summary_mode = payload.get("mode")
    if summary_mode != mode and not (summary_mode == "both" and mode in {"comprehensive", "fast"}):
        checks.append(_invalid("user-facing run summary", path, "mode does not cover verifier mode"))
    summary_contract_preset = preset
    if (
        isinstance(summary_preset, str)
        and summary_preset in PRESET_DEPTH
        and PRESET_DEPTH[summary_preset] >= PRESET_DEPTH[preset]
    ):
        summary_contract_preset = summary_preset
    summary_contract_mode = (
        summary_mode if isinstance(summary_mode, str) and summary_mode in MODES else mode
    )

    compound = payload.get("compound")
    if not isinstance(compound, dict):
        checks.append(_invalid("user-facing run summary", path, "missing compound object"))
    else:
        if compound_metadata is not None:
            for field in (
                "input_type",
                "input_smiles",
                "input_canonical_smiles",
                "input_sdf",
            ):
                expected = compound_metadata.get(field)
                if expected is not None and compound.get(field) != expected:
                    checks.append(
                        _invalid(
                            "user-facing run summary",
                            path,
                            f"compound {field} mismatch",
                        )
                    )
                if expected is None and compound.get(field) is not None:
                    checks.append(
                        _invalid(
                            "user-facing run summary",
                            path,
                            f"unexpected compound {field}",
                        )
                    )
        if compound_smiles is not None and compound.get("canonical_smiles") != compound_smiles:
            checks.append(_invalid("user-facing run summary", path, "compound canonical_smiles mismatch"))
        if compound_inchikey is not None and compound.get("inchikey") != compound_inchikey:
            checks.append(_invalid("user-facing run summary", path, "compound inchikey mismatch"))

    safety = payload.get("safety")
    if not isinstance(safety, dict):
        checks.append(_invalid("user-facing run summary", path, "missing safety object"))
    elif safety_decision is not None and safety.get("skin_sens_decision") != safety_decision:
        checks.append(_invalid("user-facing run summary", path, "skin_sens_decision mismatch"))
    expected_skin_sens_calls = _summary_skin_sens_calls(run_dir)
    expected_skin_sens_evidence = _summary_skin_sens_evidence(run_dir)
    expected_admet_metrics = _summary_admet_metrics(run_dir)
    expected_admet_risk = _admet_risk_assessment(expected_admet_metrics)
    expected_structural_flags = _summary_structural_alert_flags(run_dir)
    if isinstance(safety, dict):
        skin_sens_calls = safety.get("skin_sens_calls")
        if not isinstance(skin_sens_calls, dict):
            checks.append(_invalid("user-facing run summary", path, "missing skin_sens_calls object"))
        elif expected_skin_sens_calls and skin_sens_calls != expected_skin_sens_calls:
            checks.append(_invalid("user-facing run summary", path, "skin_sens_calls mismatch"))

        skin_sens_evidence = safety.get("skin_sens_evidence")
        if not isinstance(skin_sens_evidence, list) or not all(
            isinstance(row, dict) for row in skin_sens_evidence
        ):
            checks.append(_invalid("user-facing run summary", path, "missing skin_sens_evidence rows"))
        elif expected_skin_sens_evidence and skin_sens_evidence != expected_skin_sens_evidence:
            checks.append(_invalid("user-facing run summary", path, "skin_sens_evidence mismatch"))

        admet_metrics = safety.get("admet_metrics")
        if not isinstance(admet_metrics, dict):
            checks.append(_invalid("user-facing run summary", path, "missing admet_metrics object"))
        else:
            unexpected_metrics = sorted(set(admet_metrics) - set(expected_admet_metrics))
            if unexpected_metrics:
                checks.append(
                    _invalid(
                        "user-facing run summary",
                        path,
                        f"unexpected ADMET metrics {unexpected_metrics}",
                    )
                )
            for endpoint, expected in expected_admet_metrics.items():
                if endpoint not in admet_metrics:
                    checks.append(_invalid("user-facing run summary", path, f"missing ADMET metric {endpoint}"))
                    continue
                bad_metric = _numeric_value(
                    admet_metrics[endpoint],
                    "user-facing run summary",
                    path,
                    f"admet_metrics.{endpoint}",
                    fraction=endpoint in ADMET_PROBABILITY_ENDPOINTS,
                )
                if bad_metric is not None:
                    checks.append(bad_metric)
                elif abs(float(admet_metrics[endpoint]) - expected) > 1e-12:
                    checks.append(_invalid("user-facing run summary", path, f"ADMET metric mismatch for {endpoint}"))

        admet_risk = safety.get("admet_risk_assessment")
        if not isinstance(admet_risk, dict):
            checks.append(_invalid("user-facing run summary", path, "missing admet_risk_assessment object"))
        else:
            endpoints = admet_risk.get("endpoints")
            if not isinstance(endpoints, dict):
                checks.append(_invalid("user-facing run summary", path, "missing admet_risk_assessment.endpoints object"))
            else:
                unexpected_endpoints = sorted(
                    set(endpoints) - set(expected_admet_risk["endpoints"])
                )
                if unexpected_endpoints:
                    checks.append(
                        _invalid(
                            "user-facing run summary",
                            path,
                            f"unexpected ADMET risk endpoints {unexpected_endpoints}",
                        )
                    )
                for endpoint, expected in expected_admet_risk["endpoints"].items():
                    observed = endpoints.get(endpoint)
                    if not isinstance(observed, dict):
                        checks.append(_invalid("user-facing run summary", path, f"missing ADMET risk endpoint {endpoint}"))
                        continue
                    bad_value = _numeric_value(
                        observed.get("value"),
                        "user-facing run summary",
                        path,
                        f"admet_risk_assessment.endpoints.{endpoint}.value",
                        fraction=True,
                    )
                    if bad_value is not None:
                        checks.append(bad_value)
                    elif abs(float(observed["value"]) - expected["value"]) > 1e-12:
                        checks.append(_invalid("user-facing run summary", path, f"ADMET risk value mismatch for {endpoint}"))
                    if observed.get("risk_level") != expected["risk_level"]:
                        checks.append(_invalid("user-facing run summary", path, f"ADMET risk level mismatch for {endpoint}"))
            for list_name in ("high_risk_endpoints", "moderate_risk_endpoints"):
                observed_list = admet_risk.get(list_name)
                expected_list = expected_admet_risk[list_name]
                if not isinstance(observed_list, list) or sorted(str(item) for item in observed_list) != expected_list:
                    checks.append(_invalid("user-facing run summary", path, f"ADMET risk {list_name} mismatch"))
        structural_flags = safety.get("structural_alert_flags")
        if not isinstance(structural_flags, dict):
            checks.append(_invalid("user-facing run summary", path, "safety.structural_alert_flags must be an object"))
        elif any(not isinstance(value, bool) for value in structural_flags.values()):
            checks.append(_invalid("user-facing run summary", path, "safety.structural_alert_flags values must be boolean"))
        elif expected_structural_flags is None:
            checks.append(_invalid("user-facing run summary", path, "cannot validate safety.structural_alert_flags because ADMET structural_alerts are invalid"))
        elif expected_structural_flags is not None and structural_flags != expected_structural_flags:
            checks.append(_invalid("user-facing run summary", path, "safety.structural_alert_flags mismatch"))

    skin_toxicity = payload.get("skin_toxicity")
    skin_toxicity_decision = None
    if not isinstance(skin_toxicity, dict):
        checks.append(_invalid("user-facing run summary", path, "missing skin_toxicity object"))
    else:
        skin_toxicity_decision = skin_toxicity.get("decision")
        if skin_toxicity_decision not in SKIN_TOXICITY_DECISIONS:
            checks.append(_invalid("user-facing run summary", path, "invalid skin_toxicity.decision"))
        if skin_toxicity.get("toxicity_level") not in SKIN_TOXICITY_LEVELS:
            checks.append(_invalid("user-facing run summary", path, "invalid skin_toxicity.toxicity_level"))
        reasons = skin_toxicity.get("reasons")
        if not isinstance(reasons, list) or not reasons or not all(isinstance(r, str) and r.strip() for r in reasons):
            checks.append(_invalid("user-facing run summary", path, "skin_toxicity.reasons must be non-empty strings"))
        if isinstance(safety, dict):
            expected_skin_toxicity = _skin_toxicity_summary(safety)
            for field in (
                "decision",
                "toxicity_level",
                "skin_sens_decision",
                "skin_reaction_risk_level",
                "structural_alerts_present",
                "structural_alert_flags",
                "degraded",
                "missing_models",
                "reasons",
            ):
                if skin_toxicity.get(field) != expected_skin_toxicity[field]:
                    checks.append(_invalid("user-facing run summary", path, f"skin_toxicity {field} mismatch"))
            observed_value = skin_toxicity.get("skin_reaction_value")
            expected_value = expected_skin_toxicity["skin_reaction_value"]
            if expected_value is None:
                if observed_value is not None:
                    checks.append(_invalid("user-facing run summary", path, "skin_toxicity skin_reaction_value mismatch"))
            else:
                bad_value = _numeric_value(
                    observed_value,
                    "user-facing run summary",
                    path,
                    "skin_toxicity.skin_reaction_value",
                    fraction=True,
                )
                if bad_value is not None:
                    checks.append(bad_value)
                elif abs(float(observed_value) - float(expected_value)) > 1e-12:
                    checks.append(_invalid("user-facing run summary", path, "skin_toxicity skin_reaction_value mismatch"))

    expected_cosmetic_drug = _expected_cosmetic_drug_summary(
        run_dir,
        decision=cosmetic_decision,
        policy=cosmetic_policy,
    )
    source_warning_count = (
        expected_cosmetic_drug.get("n_warnings")
        if isinstance(expected_cosmetic_drug, dict)
        else None
    )
    cosmetic_drug = payload.get("cosmetic_drug")
    if not isinstance(cosmetic_drug, dict):
        checks.append(_invalid("user-facing run summary", path, "missing cosmetic_drug object"))
    else:
        if cosmetic_decision is not None and cosmetic_drug.get("decision") != cosmetic_decision:
            checks.append(_invalid("user-facing run summary", path, "cosmetic/drug decision mismatch"))
        if isinstance(expected_cosmetic_drug, dict):
            for field in ("drug_policy", "cosing_level", "inci", "cosing_functions"):
                if cosmetic_drug.get(field) != expected_cosmetic_drug[field]:
                    checks.append(
                        _invalid(
                            "user-facing run summary",
                            path,
                            f"cosmetic_drug {field} mismatch",
                        )
                    )
            for field in ("cosing_tanimoto", "max_tanimoto_to_approved_drug"):
                expected = expected_cosmetic_drug[field]
                observed = cosmetic_drug.get(field)
                if expected is None:
                    if observed is not None:
                        checks.append(
                            _invalid(
                                "user-facing run summary",
                                path,
                                f"cosmetic_drug {field} mismatch",
                            )
                        )
                else:
                    bad = _numeric_value(
                        observed,
                        "user-facing run summary",
                        path,
                        f"cosmetic_drug.{field}",
                        fraction=True,
                    )
                    if bad is not None:
                        checks.append(bad)
                    elif abs(float(observed) - float(expected)) > 1e-12:
                        checks.append(
                            _invalid(
                                "user-facing run summary",
                                path,
                                f"cosmetic_drug {field} mismatch",
                            )
                        )
            expected_warnings = expected_cosmetic_drug["n_warnings"]
            observed_warnings = cosmetic_drug.get("n_warnings")
            if expected_warnings is None:
                if observed_warnings is not None:
                    checks.append(
                        _invalid(
                            "user-facing run summary",
                            path,
                            "cosmetic_drug n_warnings mismatch",
                        )
                    )
            else:
                bad = _integer_value(
                    observed_warnings,
                    "user-facing run summary",
                    path,
                    "cosmetic_drug.n_warnings",
                )
                if bad is not None:
                    checks.append(bad)
                elif int(float(observed_warnings)) != expected_warnings:
                    checks.append(
                        _invalid(
                            "user-facing run summary",
                            path,
                            "cosmetic_drug n_warnings mismatch",
                        )
                    )

    overall = payload.get("overall_decision")
    overall_decision = None
    if not isinstance(overall, dict):
        checks.append(_invalid("user-facing run summary", path, "missing overall_decision object"))
    else:
        overall_decision = overall.get("decision")
        if overall_decision not in OVERALL_DECISIONS:
            checks.append(_invalid("user-facing run summary", path, "invalid overall_decision.decision"))
        elif overall.get("recommended_action") != OVERALL_RECOMMENDED_ACTIONS[overall_decision]:
            checks.append(_invalid("user-facing run summary", path, "overall_decision recommended_action mismatch"))
        if overall.get("requires_human_review") != (overall_decision != "PASS"):
            checks.append(_invalid("user-facing run summary", path, "requires_human_review mismatch"))
        expected_claimable = (
            overall_decision == "PASS"
            and overall.get("recommended_action") == "proceed"
            and overall.get("requires_human_review") is False
        )
        if overall.get("claimable") is not expected_claimable:
            checks.append(_invalid("user-facing run summary", path, "overall_decision claimable mismatch"))
        reasons = overall.get("reasons")
        if not isinstance(reasons, list) or not reasons or not all(isinstance(r, str) and r.strip() for r in reasons):
            checks.append(_invalid("user-facing run summary", path, "overall_decision.reasons must be non-empty strings"))
        if safety_decision == "HALT" and overall_decision != "HALT":
            checks.append(_invalid("user-facing run summary", path, "overall_decision must HALT on skin-sens HALT"))
        elif safety_decision == "FLAG_HIGH" and overall_decision == "PASS":
            checks.append(_invalid("user-facing run summary", path, "overall_decision cannot PASS on skin-sens FLAG_HIGH"))
        if cosmetic_decision == "HALT" and overall_decision != "HALT":
            checks.append(_invalid("user-facing run summary", path, "overall_decision must HALT on cosmetic/drug HALT"))
        elif cosmetic_decision == "DOWNWEIGHT" and overall_decision == "PASS":
            checks.append(_invalid("user-facing run summary", path, "overall_decision cannot PASS on cosmetic/drug DOWNWEIGHT"))
        if isinstance(safety, dict):
            structural_flags = safety.get("structural_alert_flags")
            if (
                isinstance(structural_flags, dict)
                and any(bool(value) for value in structural_flags.values())
                and overall_decision == "PASS"
            ):
                checks.append(_invalid("user-facing run summary", path, "overall_decision cannot PASS with structural alerts"))
            admet_risk = safety.get("admet_risk_assessment")
            if (
                isinstance(admet_risk, dict)
                and isinstance(admet_risk.get("high_risk_endpoints"), list)
                and admet_risk["high_risk_endpoints"]
                and overall_decision == "PASS"
            ):
                checks.append(_invalid("user-facing run summary", path, "overall_decision cannot PASS with high ADMET risk"))
            endpoints = admet_risk.get("endpoints") if isinstance(admet_risk, dict) else None
            skin_reaction = endpoints.get("Skin_Reaction") if isinstance(endpoints, dict) else None
            if (
                isinstance(skin_reaction, dict)
                and skin_reaction.get("risk_level") == "moderate"
                and overall_decision == "PASS"
            ):
                checks.append(_invalid("user-facing run summary", path, "overall_decision cannot PASS with moderate Skin_Reaction risk"))
        if skin_toxicity_decision == "HALT" and overall_decision != "HALT":
            checks.append(_invalid("user-facing run summary", path, "overall_decision must HALT when skin_toxicity is HALT"))
        elif skin_toxicity_decision == "REVIEW" and overall_decision == "PASS":
            checks.append(_invalid("user-facing run summary", path, "overall_decision cannot PASS when skin_toxicity requires review"))
        if isinstance(cosmetic_drug, dict):
            warnings = cosmetic_drug.get("n_warnings")
            if isinstance(warnings, int) and warnings > 0 and overall_decision == "PASS":
                checks.append(_invalid("user-facing run summary", path, "overall_decision cannot PASS with drug warnings"))
        if (
            isinstance(source_warning_count, int)
            and source_warning_count > 0
            and overall_decision == "PASS"
        ):
            checks.append(
                _invalid(
                    "user-facing run summary",
                    path,
                    "overall_decision cannot PASS with source drug warnings",
                )
            )

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        checks.append(_invalid("user-facing run summary", path, "missing artifacts object"))
    else:
        required_artifacts = {
            "compound",
            "admet_report",
            "admet_ai",
            "structural_alerts",
            "husspred",
            "stoptox",
            "pred_skin",
            "skin_sens_decision",
            "cosing_match",
            "drug_warnings",
            "cosmetic_drug_decision",
        }
        if summary_contract_preset in {"target-id", "report"}:
            required_artifacts.add("target_ranking")
            if summary_contract_mode in {"fast", "both"}:
                required_artifacts.update(
                    FAST_TARGET_ARTIFACTS
                    if _uses_daina_structural_contract(run_dir)
                    else LEGACY_FAST_TARGET_ARTIFACTS
                )
            if summary_contract_mode in {"comprehensive", "both"}:
                required_artifacts.update(COMPREHENSIVE_TARGET_ARTIFACTS)
        if summary_contract_preset == "report":
            required_artifacts.add("html_report")
        missing = sorted(required_artifacts - set(artifacts))
        if missing:
            checks.append(_invalid("user-facing run summary", path, f"missing artifact keys {missing}"))
        expected_artifacts = {
            "compound": "01_input/compound_canonical.json",
            "admet_report": "02_admet/admet_report.json",
            "admet_ai": "02_admet/admet_ai.json",
            "structural_alerts": "02_admet/structural_alerts.json",
            "husspred": "02_admet/husspred.json",
            "stoptox": "02_admet/stoptox.json",
            "pred_skin": "02_admet/pred_skin.json",
            "skin_sens_decision": "02_admet/skin_sens_decision.txt",
            "cosing_match": "02b_cosmetic_drug/cosing_match.json",
            "drug_warnings": "02b_cosmetic_drug/drug_warnings.json",
            "cosmetic_drug_decision": "02b_cosmetic_drug/cosmetic_drug_decision.txt",
        }
        if summary_contract_preset in {"target-id", "report"}:
            expected_artifacts["target_ranking"] = "03_targets/ranked_targets_v3_with_efficacy.csv"
            if summary_contract_mode in {"fast", "both"}:
                expected_artifacts.update(
                    FAST_TARGET_ARTIFACTS
                    if _uses_daina_structural_contract(run_dir)
                    else LEGACY_FAST_TARGET_ARTIFACTS
                )
            if summary_contract_mode in {"comprehensive", "both"}:
                expected_artifacts.update(COMPREHENSIVE_TARGET_ARTIFACTS)
        if summary_contract_preset == "report":
            expected_artifacts["html_report"] = "09_report/index.html"
        unexpected = sorted(set(artifacts) - set(expected_artifacts))
        if unexpected:
            checks.append(
                _invalid(
                    "user-facing run summary",
                    path,
                    f"unexpected artifact keys {unexpected}",
                )
            )
        for key, expected in sorted(expected_artifacts.items()):
            if key in artifacts and artifacts.get(key) != expected:
                checks.append(
                    _invalid(
                        "user-facing run summary",
                        path,
                        f"artifact {key} must point to {expected}",
                    )
                )
        for key in sorted(artifacts):
            value = artifacts.get(key)
            if not isinstance(value, str) or not value.strip():
                checks.append(_invalid("user-facing run summary", path, f"artifact {key} is blank"))
                break
            artifact_path = Path(value)
            if artifact_path.is_absolute() or ".." in artifact_path.parts:
                checks.append(
                    _invalid(
                        "user-facing run summary",
                        path,
                        f"artifact {key} must be a relative path inside run directory",
                    )
                )
                break
            if not _nonempty(run_dir / artifact_path):
                checks.append(_invalid("user-facing run summary", path, f"artifact {key} is missing: {value}"))
                break

    _verify_markdown_summary(run_dir, checks, payload, expected_admet_risk)

    if preset not in {"target-id", "report"}:
        if summary_contract_preset not in {"target-id", "report"}:
            unexpected_sections = sorted(
                section
                for section in ("skin_specialized_binding", "target_prediction")
                if section in payload
            )
            if unexpected_sections:
                checks.append(
                    _invalid(
                        "user-facing run summary",
                        path,
                        f"unexpected summary sections {unexpected_sections}",
                    )
                )
        return
    target_prediction = payload.get("target_prediction")
    if not isinstance(target_prediction, dict):
        checks.append(_invalid("user-facing run summary", path, "missing target_prediction object"))
        return
    ranking_path = run_dir / "03_targets" / "ranked_targets_v3_with_efficacy.csv"
    ranking, ranking_check = _read_csv(ranking_path, "summary target ranking cross-check")
    if ranking_check.status != "ok":
        return
    top_targets = target_prediction.get("top_targets")
    if not isinstance(top_targets, list) or not top_targets:
        checks.append(_invalid("user-facing run summary", path, "target_prediction.top_targets must be non-empty"))
        return
    if target_prediction.get("ranking_path") != "03_targets/ranked_targets_v3_with_efficacy.csv":
        checks.append(_invalid("user-facing run summary", path, "target_prediction.ranking_path mismatch"))
    if target_prediction.get("n_targets") != len(ranking):
        checks.append(_invalid("user-facing run summary", path, "target_prediction.n_targets mismatch"))
    expected_screening_counts, screening_count_checks = (
        _target_screening_count_expectations(
            run_dir,
            mode=summary_contract_mode,
            ranked_count=len(ranking),
        )
    )
    checks.extend(screening_count_checks)
    screening_counts = target_prediction.get("screening_counts")
    if not isinstance(screening_counts, dict):
        checks.append(
            _invalid(
                "user-facing run summary",
                path,
                "target_prediction.screening_counts must be an object",
            )
        )
    elif screening_counts != expected_screening_counts:
        checks.append(
            _invalid(
                "user-facing run summary",
                path,
                "target_prediction.screening_counts mismatch",
            )
        )
    else:
        violations = _screening_funnel_violations(
            run_dir,
            expected_screening_counts,
            mode=summary_contract_mode,
        )
        if violations:
            checks.append(
                _invalid(
                    "user-facing run summary",
                    path,
                    "target_prediction.screening_counts non-monotonic funnel: "
                    + "; ".join(violations),
                )
            )
        final_ranked = expected_screening_counts.get("skin_weighted_ranked_targets")
        if isinstance(final_ranked, int) and max(expected_screening_counts.values()) <= final_ranked:
            checks.append(
                _invalid(
                    "user-facing run summary",
                    path,
                    "target_prediction.screening_counts must include an upstream count above final ranked targets",
                )
            )
    expected_screened_count = max(expected_screening_counts.values())
    if target_prediction.get("screened_target_count") != expected_screened_count:
        checks.append(
            _invalid(
                "user-facing run summary",
                path,
                "target_prediction.screened_target_count mismatch",
            )
        )
    if target_prediction.get("top_n") != len(top_targets):
        checks.append(_invalid("user-facing run summary", path, "target_prediction.top_n mismatch"))
    if len(top_targets) > len(ranking):
        checks.append(_invalid("user-facing run summary", path, "target_prediction.top_targets exceeds ranking rows"))
    ranking_top = ranking.head(min(len(top_targets), len(ranking)))
    # strict=False is deliberate: a top_targets/ranking length mismatch is
    # already recorded as an invalid check above, and this loop must keep
    # collecting findings rather than raising.
    for rank, (target, (_, row)) in enumerate(
        zip(top_targets, ranking_top.iterrows(), strict=False), start=1
    ):
        label = f"top_targets[{rank - 1}]"
        if not isinstance(target, dict):
            checks.append(_invalid("user-facing run summary", path, f"{label} must be an object"))
            continue
        expected_target = str(row["target_id"]).strip()
        if target.get("target_id") != expected_target:
            checks.append(_invalid("user-facing run summary", path, f"{label}.target_id mismatch"))
        if "skin_tier" in ranking.columns:
            expected_skin_tier = _text_or_none(row.get("skin_tier"))
            if target.get("skin_tier") != expected_skin_tier:
                checks.append(_invalid("user-facing run summary", path, f"{label}.skin_tier mismatch"))
        for field in ("gene_symbol", "protein_name"):
            value = target.get(field)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                checks.append(
                    _invalid(
                        "user-facing run summary",
                        path,
                        f"{label}.{field} must be a non-empty string or null",
                    )
                )
        metadata = target_metadata.get(expected_target, {})
        if metadata.get("gene_symbol") and target.get("gene_symbol") != metadata["gene_symbol"]:
            checks.append(_invalid("user-facing run summary", path, f"{label}.gene_symbol mismatch"))
        if metadata.get("protein_name") and target.get("protein_name") != metadata["protein_name"]:
            checks.append(_invalid("user-facing run summary", path, f"{label}.protein_name mismatch"))
        for field in ("final_score", "skin_score", "docking_rrf"):
            expected_value = row.get(field)
            bad = _numeric_value(
                target.get(field),
                "user-facing run summary",
                path,
                f"{label}.{field}",
                fraction=True,
            )
            if bad is not None:
                checks.append(bad)
                continue
            try:
                observed_numeric = float(target.get(field))
                expected_numeric = float(expected_value)
            except (TypeError, ValueError):
                continue
            if (
                math.isfinite(observed_numeric)
                and math.isfinite(expected_numeric)
                and abs(observed_numeric - expected_numeric) > 1e-12
            ):
                checks.append(_invalid("user-facing run summary", path, f"{label}.{field} mismatch"))
        expected_source_count = row.get("source_count")
        bad_count = _integer_value(
            target.get("source_count"),
            "user-facing run summary",
            path,
            f"{label}.source_count",
        )
        if bad_count is not None:
            checks.append(bad_count)
        else:
            try:
                observed_source_count = int(float(target.get("source_count")))
                expected_source_count_int = int(float(expected_source_count))
            except (TypeError, ValueError):
                observed_source_count = -1
                expected_source_count_int = -2
            if observed_source_count != expected_source_count_int:
                checks.append(_invalid("user-facing run summary", path, f"{label}.source_count mismatch"))
        if target.get("sources") != _source_labels(row.get("sources")):
            checks.append(_invalid("user-facing run summary", path, f"{label}.sources mismatch"))
        if target.get("efficacy") != _efficacy_labels(row):
            checks.append(_invalid("user-facing run summary", path, f"{label}.efficacy mismatch"))
    first = top_targets[0]
    if not isinstance(first, dict):
        checks.append(_invalid("user-facing run summary", path, "top target entry must be an object"))
        return
    first_target = str(ranking.iloc[0]["target_id"]).strip()
    if first.get("target_id") != first_target:
        checks.append(_invalid("user-facing run summary", path, "top target_id mismatch"))
    for field in ("gene_symbol", "protein_name"):
        value = first.get(field)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            checks.append(
                _invalid(
                    "user-facing run summary",
                    path,
                    f"top_targets[0].{field} must be a non-empty string or null",
                )
            )
    if "skin_tier" in ranking.columns:
        expected_skin_tier = _text_or_none(ranking.iloc[0].get("skin_tier"))
        if first.get("skin_tier") != expected_skin_tier:
            checks.append(_invalid("user-facing run summary", path, "top target skin_tier mismatch"))
    metadata = target_metadata.get(first_target, {})
    if metadata.get("gene_symbol") and first.get("gene_symbol") != metadata["gene_symbol"]:
        checks.append(_invalid("user-facing run summary", path, "top target gene_symbol mismatch"))
    if metadata.get("protein_name") and first.get("protein_name") != metadata["protein_name"]:
        checks.append(_invalid("user-facing run summary", path, "top target protein_name mismatch"))
    if isinstance(overall, dict):
        if overall.get("target_count") != len(ranking):
            checks.append(_invalid("user-facing run summary", path, "overall_decision.target_count mismatch"))
        if overall.get("top_target_id") != first_target:
            checks.append(_invalid("user-facing run summary", path, "overall_decision.top_target_id mismatch"))
    for field in ("final_score", "skin_score", "docking_rrf"):
        bad = _numeric_value(first.get(field), "user-facing run summary", path, f"top_targets[0].{field}")
        if bad is not None:
            checks.append(bad)
    ranking_skin_score = _numeric_value(
        ranking.iloc[0].get("skin_score"),
        "summary target ranking cross-check",
        ranking_path,
        "top row skin_score",
        fraction=True,
    )
    if ranking_skin_score is None:
        try:
            expected_skin_score = float(ranking.iloc[0]["skin_score"])
            observed_skin_score = float(first.get("skin_score"))
        except (TypeError, ValueError):
            expected_skin_score = math.nan
            observed_skin_score = math.nan
        if (
            math.isfinite(expected_skin_score)
            and math.isfinite(observed_skin_score)
            and abs(expected_skin_score - observed_skin_score) > 1e-12
        ):
            checks.append(_invalid("user-facing run summary", path, "top target skin_score mismatch"))

    skin_binding = payload.get("skin_specialized_binding")
    if not isinstance(skin_binding, dict):
        checks.append(_invalid("user-facing run summary", path, "missing skin_specialized_binding object"))
        return
    if skin_binding.get("context") != "skin-specialized material-protein binding":
        checks.append(_invalid("user-facing run summary", path, "invalid skin_specialized_binding.context"))
    if skin_binding.get("ranking_path") != target_prediction.get("ranking_path"):
        checks.append(_invalid("user-facing run summary", path, "skin_specialized_binding.ranking_path mismatch"))
    if skin_binding.get("top_target_id") != first_target:
        checks.append(_invalid("user-facing run summary", path, "skin_specialized_binding.top_target_id mismatch"))
    for field in ("gene_symbol", "protein_name"):
        summary_field = f"top_target_{field}"
        expected = first.get(field)
        observed = skin_binding.get(summary_field)
        if expected is None:
            if observed is not None:
                checks.append(
                    _invalid(
                        "user-facing run summary",
                        path,
                        f"unexpected skin_specialized_binding.{summary_field}",
                    )
                )
        elif observed != expected:
            checks.append(
                _invalid(
                    "user-facing run summary",
                    path,
                    f"skin_specialized_binding.{summary_field} mismatch",
                )
            )
    for field, column, detail in (
        ("top_target_final_score", "final_score", "top final_score"),
        ("top_target_docking_rrf", "docking_rrf", "top docking_rrf"),
    ):
        bad_value = _numeric_value(
            skin_binding.get(field),
            "user-facing run summary",
            path,
            f"skin_specialized_binding.{field}",
            fraction=True,
        )
        if bad_value is not None:
            checks.append(bad_value)
        else:
            try:
                observed = float(skin_binding.get(field))
                expected = float(ranking.iloc[0][column])
            except (TypeError, ValueError):
                observed = math.nan
                expected = math.nan
            if math.isfinite(observed) and math.isfinite(expected) and abs(observed - expected) > 1e-12:
                checks.append(_invalid("user-facing run summary", path, f"skin_specialized_binding {detail} mismatch"))
    bad_source_count = _integer_value(
        skin_binding.get("top_target_source_count"),
        "user-facing run summary",
        path,
        "skin_specialized_binding.top_target_source_count",
    )
    if bad_source_count is not None:
        checks.append(bad_source_count)
    else:
        try:
            observed_source_count = int(float(skin_binding.get("top_target_source_count")))
            expected_source_count = int(float(ranking.iloc[0]["source_count"]))
        except (TypeError, ValueError):
            observed_source_count = -1
            expected_source_count = -2
        if observed_source_count != expected_source_count:
            checks.append(_invalid("user-facing run summary", path, "skin_specialized_binding top source_count mismatch"))
    if skin_binding.get("top_target_sources") != _source_labels(ranking.iloc[0].get("sources")):
        checks.append(_invalid("user-facing run summary", path, "skin_specialized_binding top sources mismatch"))
    bad_skin_binding_score = _numeric_value(
        skin_binding.get("top_target_skin_score"),
        "user-facing run summary",
        path,
        "skin_specialized_binding.top_target_skin_score",
        fraction=True,
    )
    if bad_skin_binding_score is not None:
        checks.append(bad_skin_binding_score)
    elif ranking_skin_score is None:
        try:
            observed = float(skin_binding.get("top_target_skin_score"))
            expected = float(ranking.iloc[0]["skin_score"])
        except (TypeError, ValueError):
            observed = math.nan
            expected = math.nan
        if math.isfinite(observed) and math.isfinite(expected) and abs(observed - expected) > 1e-12:
            checks.append(_invalid("user-facing run summary", path, "skin_specialized_binding top skin_score mismatch"))
    if "skin_tier" in ranking.columns and skin_binding.get("top_target_skin_tier") != _text_or_none(ranking.iloc[0].get("skin_tier")):
        checks.append(_invalid("user-facing run summary", path, "skin_specialized_binding top skin_tier mismatch"))
    if "cell_type_preferred" in ranking.columns:
        expected_cell_type = _text_or_none(ranking.iloc[0].get("cell_type_preferred"))
        if first.get("cell_type_preferred") != expected_cell_type:
            checks.append(_invalid("user-facing run summary", path, "top target cell_type_preferred mismatch"))
        if skin_binding.get("top_target_cell_type_preferred") != expected_cell_type:
            checks.append(_invalid("user-facing run summary", path, "skin_specialized_binding top cell_type_preferred mismatch"))
    expected_efficacy = _efficacy_labels(ranking.iloc[0])
    if skin_binding.get("top_target_efficacy") != expected_efficacy:
        checks.append(_invalid("user-facing run summary", path, "skin_specialized_binding top efficacy mismatch"))
    expected_top_expression_supported = _row_has_skin_expression_support(ranking.iloc[0])
    expected_top_efficacy_supported = bool(expected_efficacy)
    if skin_binding.get("top_target_skin_expression_supported") is not expected_top_expression_supported:
        checks.append(
            _invalid(
                "user-facing run summary",
                path,
                "skin_specialized_binding top expression support mismatch",
            )
        )
    if skin_binding.get("top_target_skin_efficacy_supported") is not expected_top_efficacy_supported:
        checks.append(
            _invalid(
                "user-facing run summary",
                path,
                "skin_specialized_binding top efficacy support mismatch",
            )
        )
    if skin_binding.get("top_target_skin_context_supported") is not (
        expected_top_expression_supported and expected_top_efficacy_supported
    ):
        checks.append(
            _invalid(
                "user-facing run summary",
                path,
                "skin_specialized_binding top context support mismatch",
            )
        )
    expected_with_efficacy = sum(1 for _, row in ranking_top.iterrows() if _efficacy_labels(row))
    if skin_binding.get("top_targets_with_skin_efficacy") != expected_with_efficacy:
        checks.append(_invalid("user-facing run summary", path, "skin_specialized_binding efficacy count mismatch"))
    most_skin_relevant = skin_binding.get("most_skin_relevant_target")
    if not isinstance(most_skin_relevant, dict):
        checks.append(_invalid("user-facing run summary", path, "missing skin_specialized_binding.most_skin_relevant_target object"))
        return
    best_row = max(
        (row for _, row in ranking_top.iterrows()),
        key=lambda row: (
            float(row.get("skin_score") or 0.0),
            bool(_efficacy_labels(row)),
            float(row.get("final_score") or 0.0),
        ),
    )
    best_target = str(best_row["target_id"]).strip()
    if most_skin_relevant.get("target_id") != best_target:
        checks.append(_invalid("user-facing run summary", path, "most skin-relevant target_id mismatch"))
    matching_top_target = next(
        (
            target
            for target in top_targets
            if isinstance(target, dict) and target.get("target_id") == best_target
        ),
        None,
    )
    for field in ("gene_symbol", "protein_name"):
        observed = most_skin_relevant.get(field)
        if observed is not None and (not isinstance(observed, str) or not observed.strip()):
            checks.append(
                _invalid(
                    "user-facing run summary",
                    path,
                    f"most_skin_relevant_target.{field} must be a non-empty string or null",
                )
            )
            continue
        expected = (
            matching_top_target.get(field)
            if isinstance(matching_top_target, dict)
            else None
        )
        if expected is None:
            if observed is not None:
                checks.append(
                    _invalid(
                        "user-facing run summary",
                        path,
                        f"unexpected most_skin_relevant_target.{field}",
                    )
                )
        elif observed != expected:
            checks.append(
                _invalid(
                    "user-facing run summary",
                    path,
                    f"most skin-relevant {field} mismatch",
                )
            )
    for field in ("final_score", "docking_rrf"):
        bad_value = _numeric_value(
            most_skin_relevant.get(field),
            "user-facing run summary",
            path,
            f"most_skin_relevant_target.{field}",
            fraction=True,
        )
        if bad_value is not None:
            checks.append(bad_value)
        else:
            try:
                observed = float(most_skin_relevant.get(field))
                expected = float(best_row[field])
            except (TypeError, ValueError):
                observed = math.nan
                expected = math.nan
            if math.isfinite(observed) and math.isfinite(expected) and abs(observed - expected) > 1e-12:
                checks.append(_invalid("user-facing run summary", path, f"most skin-relevant {field} mismatch"))
    bad_most_source_count = _integer_value(
        most_skin_relevant.get("source_count"),
        "user-facing run summary",
        path,
        "most_skin_relevant_target.source_count",
    )
    if bad_most_source_count is not None:
        checks.append(bad_most_source_count)
    else:
        try:
            observed_source_count = int(float(most_skin_relevant.get("source_count")))
            expected_source_count = int(float(best_row["source_count"]))
        except (TypeError, ValueError):
            observed_source_count = -1
            expected_source_count = -2
        if observed_source_count != expected_source_count:
            checks.append(_invalid("user-facing run summary", path, "most skin-relevant source_count mismatch"))
    if most_skin_relevant.get("sources") != _source_labels(best_row.get("sources")):
        checks.append(_invalid("user-facing run summary", path, "most skin-relevant sources mismatch"))
    if "skin_tier" in ranking.columns and most_skin_relevant.get("skin_tier") != _text_or_none(best_row.get("skin_tier")):
        checks.append(_invalid("user-facing run summary", path, "most skin-relevant skin_tier mismatch"))
    if (
        "cell_type_preferred" in ranking.columns
        and most_skin_relevant.get("cell_type_preferred")
        != _text_or_none(best_row.get("cell_type_preferred"))
    ):
        checks.append(_invalid("user-facing run summary", path, "most skin-relevant cell_type_preferred mismatch"))
    bad_most_skin_score = _numeric_value(
        most_skin_relevant.get("skin_score"),
        "user-facing run summary",
        path,
        "most_skin_relevant_target.skin_score",
        fraction=True,
    )
    if bad_most_skin_score is not None:
        checks.append(bad_most_skin_score)
    else:
        try:
            observed = float(most_skin_relevant.get("skin_score"))
            expected = float(best_row["skin_score"])
        except (TypeError, ValueError):
            observed = math.nan
            expected = math.nan
        if math.isfinite(observed) and math.isfinite(expected) and abs(observed - expected) > 1e-12:
            checks.append(_invalid("user-facing run summary", path, "most skin-relevant skin_score mismatch"))
    if most_skin_relevant.get("efficacy") != _efficacy_labels(best_row):
        checks.append(_invalid("user-facing run summary", path, "most skin-relevant efficacy mismatch"))
    (
        expected_skin_context_decision,
        expected_skin_expression_supported,
        expected_skin_efficacy_supported,
        expected_skin_context_reasons,
    ) = _expected_skin_context(
        best_row=best_row,
        top_targets_with_skin_efficacy=expected_with_efficacy,
    )
    observed_skin_context_decision = skin_binding.get("skin_context_decision")
    if observed_skin_context_decision not in SKIN_CONTEXT_DECISIONS:
        checks.append(_invalid("user-facing run summary", path, "invalid skin_context_decision"))
    elif observed_skin_context_decision != expected_skin_context_decision:
        checks.append(_invalid("user-facing run summary", path, "skin_context_decision mismatch"))
    if skin_binding.get("skin_expression_supported") is not expected_skin_expression_supported:
        checks.append(_invalid("user-facing run summary", path, "skin_expression_supported mismatch"))
    if skin_binding.get("skin_efficacy_supported") is not expected_skin_efficacy_supported:
        checks.append(_invalid("user-facing run summary", path, "skin_efficacy_supported mismatch"))
    if skin_binding.get("skin_context_supported") is not (
        expected_skin_context_decision == "skin_context_supported"
    ):
        checks.append(_invalid("user-facing run summary", path, "skin_context_supported mismatch"))
    if skin_binding.get("skin_context_reasons") != expected_skin_context_reasons:
        checks.append(_invalid("user-facing run summary", path, "skin_context_reasons mismatch"))
    expected_top_context_supported = (
        expected_top_expression_supported and expected_top_efficacy_supported
    )
    if not expected_top_context_supported and isinstance(overall, dict):
        if overall_decision == "PASS":
            checks.append(
                _invalid(
                    "user-facing run summary",
                    path,
                    "overall_decision cannot PASS without top binding target skin context",
                )
            )
        expected_reason = "top binding target lacks direct skin context support"
        reasons = overall.get("reasons")
        if isinstance(reasons, list) and not any(
            expected_reason in str(reason) for reason in reasons
        ):
            checks.append(
                _invalid(
                    "user-facing run summary",
                    path,
                    "overall_decision missing top binding target skin context review reason",
                )
            )
    if expected_skin_context_decision != "skin_context_supported" and isinstance(overall, dict):
        if overall_decision == "PASS":
            checks.append(
                _invalid(
                    "user-facing run summary",
                    path,
                    "overall_decision cannot PASS without supported skin context",
                )
            )
        expected_reason = (
            "skin-specialized binding context requires review: "
            + expected_skin_context_decision
        )
        reasons = overall.get("reasons")
        if isinstance(reasons, list) and expected_reason not in reasons:
            checks.append(
                _invalid(
                    "user-facing run summary",
                    path,
                    "overall_decision missing skin context review reason",
                )
            )


def _read_table_allow_empty(
    path: Path,
    name: str,
    *,
    required_columns: set[str],
    sep: str,
) -> tuple[pd.DataFrame, bool]:
    check = _check_path(path, name)
    if check.status != "ok":
        return pd.DataFrame(), False
    try:
        frame = pd.read_csv(path, sep=sep)
    except Exception:
        return pd.DataFrame(), False
    return frame, not (required_columns - set(frame.columns))


def _manifest_targets(
    payload: dict[str, Any],
    *,
    schema: str,
) -> tuple[list[dict[str, Any]], str | None]:
    if payload.get("schema_version") != schema:
        return [], f"schema_version must be {schema}"
    records = payload.get("targets")
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        return [], "targets must be an array of objects"
    if payload.get("target_count") != len(records):
        return [], "target_count does not match targets"
    target_ids = [str(row.get("target_id", "")).strip() for row in records]
    if any(not target_id for target_id in target_ids) or len(set(target_ids)) != len(target_ids):
        return [], "target_id values must be nonblank and unique"
    return records, None


def _boolean_cell(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    return None


def _validate_target_fast_frame(
    frame: pd.DataFrame,
    path: Path,
) -> str | None:
    required = {
        "schema_version",
        "recipe_id",
        "target_id",
        "target_name",
        "daina_rank",
        "daina_score",
        "daina_score_is_probability",
        "daina_primary_json",
        "structural_status",
        "structure_supported",
        "structure_annotation_json",
        "skin_kg_annotation_json",
        "experimental_annotation_json",
        "final_score",
        "source_count",
        "sources",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        return f"missing columns {missing}"
    if len(frame) != 256:
        return f"canonical Daina target set must contain exactly 256 rows, found {len(frame)}"
    for index, row in frame.iterrows():
        try:
            record = {
                "schema_version": str(row["schema_version"]),
                "recipe_id": str(row["recipe_id"]),
                "target_id": str(row["target_id"]).strip(),
                "target_name": None if pd.isna(row["target_name"]) else str(row["target_name"]),
                "daina_rank": int(row["daina_rank"]),
                "daina_score": float(row["daina_score"]),
                "daina_score_is_probability": _boolean_cell(
                    row["daina_score_is_probability"]
                ),
                "daina_primary_json": str(row["daina_primary_json"]),
                "structural_status": str(row["structural_status"]),
                "structure_supported": _boolean_cell(row["structure_supported"]),
                "structure_annotation_json": str(row["structure_annotation_json"]),
                "skin_kg_annotation_json": str(row["skin_kg_annotation_json"]),
                "experimental_annotation_json": str(row["experimental_annotation_json"]),
                "final_score": float(row["final_score"]),
                "source_count": int(row["source_count"]),
                "sources": str(row["sources"]),
            }
            validate_target_fast_v2_record(record)
        except (TypeError, ValueError) as exc:
            return f"invalid row {int(index)}: {exc}"
    expected_ranks = list(range(1, len(frame) + 1))
    if frame["daina_rank"].astype(int).tolist() != expected_ranks:
        return "daina_rank must be contiguous and ordered"
    return None


def _verify_daina_fast_target_intermediates(
    run_dir: Path,
    checks: list[OutputCheck],
) -> None:
    base = run_dir / "03_targets" / "mode_fast"
    selected_path = base / "daina_top256.csv"
    selected, check = _read_csv(selected_path, "fixed Daina primary target set")
    if _append_if_bad(checks, check):
        required = {
            "target_id",
            "daina_rank",
            "daina_score",
            "daina_score_is_probability",
        }
        missing = sorted(required - set(selected.columns))
        if missing:
            checks.append(
                _invalid("fixed Daina primary target set", selected_path, f"missing columns {missing}")
            )
        elif len(selected) != 256:
            checks.append(
                _invalid(
                    "fixed Daina primary target set",
                    selected_path,
                    f"expected exactly 256 target rows, found {len(selected)}",
                )
            )
        else:
            checks.extend(
                _validate_target_ids(selected, "fixed Daina primary target set", selected_path)
            )
            ranks = pd.to_numeric(selected["daina_rank"], errors="coerce")
            scores = pd.to_numeric(selected["daina_score"], errors="coerce")
            if ranks.isna().any() or ranks.astype(int).tolist() != list(range(1, 257)):
                checks.append(
                    _invalid(
                        "fixed Daina primary target set",
                        selected_path,
                        "daina_rank must be contiguous 1..256",
                    )
                )
            if scores.isna().any() or not scores.is_monotonic_decreasing:
                checks.append(
                    _invalid(
                        "fixed Daina primary target set",
                        selected_path,
                        "daina_score must be numeric and sorted descending",
                    )
                )
            probability = selected["daina_score_is_probability"].map(_boolean_cell)
            if probability.isna().any() or probability.any():
                checks.append(
                    _invalid(
                        "fixed Daina primary target set",
                        selected_path,
                        "Daina score must be explicitly non-probabilistic",
                    )
                )

    _verify_table(
        checks,
        base / "daina_zoete_proteome.tsv",
        "fast Daina-Zoete proteome scores",
        required_columns={"target_id", "max_tanimoto", "score"},
        numeric_columns={"max_tanimoto", "score"},
        sep="\t",
        min_rows=256,
    )
    _verify_table(
        checks,
        base / "dti_rrf_top25pct.csv",
        "Daina top-set compatibility projection",
        required_columns={"target_id", "rrf_score", "source_count", "sources"},
        numeric_columns={"rrf_score"},
        min_rows=256,
        min_source_count=1,
    )

    map_path = base / "autogrid_map_manifest.json"
    map_payload, map_check = _read_json(map_path, "per-query AutoGrid map manifest")
    map_records: list[dict[str, Any]] = []
    if _append_if_bad(checks, map_check):
        map_records, detail = _manifest_targets(
            map_payload, schema="skinscout.autogrid-map-manifest.v1"
        )
        if detail:
            checks.append(_invalid("per-query AutoGrid map manifest", map_path, detail))
        elif not selected.empty and [row["target_id"] for row in map_records] != selected["target_id"].astype(str).tolist():
            checks.append(
                _invalid(
                    "per-query AutoGrid map manifest",
                    map_path,
                    "target order does not match the fixed Daina primary set",
                )
            )
        else:
            allowed = {
                "map_ready",
                "structural_unavailable_no_pocket",
                "structural_unavailable_prep",
                "structural_unavailable_invalid",
                "structure_failed_map",
            }
            for row in map_records:
                if row.get("status") not in allowed:
                    checks.append(
                        _invalid(
                            "per-query AutoGrid map manifest",
                            map_path,
                            f"unsupported status for {row.get('target_id')}: {row.get('status')!r}",
                        )
                    )
                    break
                if row.get("status") == "map_ready":
                    map_fld = _manifest_member_path(map_path, row.get("map_fld"))
                    if map_fld is None:
                        checks.append(
                            _invalid(
                                "per-query AutoGrid map manifest",
                                map_path,
                                f"unsafe or blank map path for {row.get('target_id')}",
                            )
                        )
                        break
                    if not _nonempty(map_fld) or row.get("map_fld_sha256") != _sha256(map_fld):
                        checks.append(
                            _invalid(
                                "per-query AutoGrid map manifest",
                                map_path,
                                f"map file/hash mismatch for {row.get('target_id')}",
                            )
                        )
                        break

    docking_path = base / "autodock_top5k.tsv"
    docking, docking_ok = _read_table_allow_empty(
        docking_path,
        "fast AutoDock-GPU scores",
        required_columns={
            "target_id",
            "vina_score",
            "neg_vina_score",
            "engine",
            "map_coverage_complete",
            "map_coverage_numerator",
            "map_coverage_denominator",
            "degraded",
        },
        sep="\t",
    )
    docking_check = _check_path(docking_path, "fast AutoDock-GPU scores")
    checks.append(
        docking_check
        if docking_check.status != "ok" or docking_ok
        else _invalid("fast AutoDock-GPU scores", docking_path, "missing required columns")
    )
    if docking_ok and not docking.empty:
        checks.extend(_validate_target_ids(docking, "fast AutoDock-GPU scores", docking_path))
        for column in ("vina_score", "neg_vina_score"):
            bad = _numeric_series(docking, column, "fast AutoDock-GPU scores", docking_path)
            if bad is not None:
                checks.append(bad)
        if not docking["engine"].astype(str).eq("autodock_gpu").all():
            checks.append(_invalid("fast AutoDock-GPU scores", docking_path, "engine must be autodock_gpu"))
        if not docking["degraded"].map(_boolean_cell).eq(False).all():
            checks.append(_invalid("fast AutoDock-GPU scores", docking_path, "degraded must be false"))

    docking_status_path = base / "autodock_status_manifest.json"
    docking_payload, docking_status_check = _read_json(
        docking_status_path, "fast docking status manifest"
    )
    docking_records: list[dict[str, Any]] = []
    if _append_if_bad(checks, docking_status_check):
        docking_records, detail = _manifest_targets(
            docking_payload, schema="skinscout.docking-status-manifest.v1"
        )
        if detail:
            checks.append(_invalid("fast docking status manifest", docking_status_path, detail))
        elif not selected.empty and [row["target_id"] for row in docking_records] != selected["target_id"].astype(str).tolist():
            checks.append(
                _invalid(
                    "fast docking status manifest",
                    docking_status_path,
                    "target order does not match the fixed Daina primary set",
                )
            )
        elif docking_ok:
            docked_ids = {
                str(row["target_id"])
                for row in docking_records
                if row.get("status") == "docked"
            }
            if docked_ids != set(docking["target_id"].astype(str)):
                checks.append(
                    _invalid(
                        "fast docking status manifest",
                        docking_status_path,
                        "docked statuses do not match score rows",
                    )
                )

    pose_manifest_path = base / "autodock_top5k_poses" / "pose_manifest.json"
    pose_records: list[dict[str, Any]] = []
    pose_payload, pose_check = _read_json(pose_manifest_path, "real docking pose manifest")
    if _append_if_bad(checks, pose_check):
        pose_records, detail = _manifest_targets(
            pose_payload, schema="skinscout.docking_pose_manifest.v1"
        )
        if detail:
            checks.append(_invalid("real docking pose manifest", pose_manifest_path, detail))
        elif docking_ok and {str(row["target_id"]) for row in pose_records} != set(docking["target_id"].astype(str)):
            checks.append(
                _invalid("real docking pose manifest", pose_manifest_path, "pose/score target parity failed")
            )
        else:
            for row in pose_records:
                pose = _manifest_member_path(
                    pose_manifest_path, row.get("pose_file")
                )
                if pose is None:
                    checks.append(
                        _invalid(
                            "real docking pose manifest",
                            pose_manifest_path,
                            f"unsafe or blank pose path for {row.get('target_id')}",
                        )
                    )
                    break
                if not _nonempty(pose) or row.get("pose_sha256") != _sha256(pose):
                    checks.append(
                        _invalid(
                            "real docking pose manifest",
                            pose_manifest_path,
                            f"pose file/hash mismatch for {row.get('target_id')}",
                        )
                    )
                    break

    gnina_path = base / "gnina_pose_rescores.tsv"
    gnina, gnina_ok = _read_table_allow_empty(
        gnina_path,
        "GNINA actual-pose scores",
        required_columns={
            "target_id",
            "cnn_affinity",
            "score",
            "pose_file",
            "pose_sha256",
            "scored_actual_docked_pose",
            "gpu_enabled",
        },
        sep="\t",
    )
    gnina_check = _check_path(gnina_path, "GNINA actual-pose scores")
    checks.append(
        gnina_check
        if gnina_check.status != "ok" or gnina_ok
        else _invalid("GNINA actual-pose scores", gnina_path, "missing required columns")
    )
    if gnina_ok and not gnina.empty:
        checks.extend(_validate_target_ids(gnina, "GNINA actual-pose scores", gnina_path))
        if not gnina["scored_actual_docked_pose"].map(_boolean_cell).eq(True).all():
            checks.append(_invalid("GNINA actual-pose scores", gnina_path, "all scores must use actual docked poses"))
        if not gnina["gpu_enabled"].map(_boolean_cell).eq(True).all():
            checks.append(_invalid("GNINA actual-pose scores", gnina_path, "GPU scoring must be enabled"))
        pose_by_target = {
            str(row.get("target_id")): row for row in pose_records
        }
        for _, row in gnina.iterrows():
            target_id = str(row.get("target_id"))
            expected = pose_by_target.get(target_id)
            if (
                expected is None
                or str(row.get("pose_file", ""))
                != str(expected.get("pose_file", ""))
                or str(row.get("pose_sha256", ""))
                != str(expected.get("pose_sha256", ""))
            ):
                checks.append(
                    _invalid(
                        "GNINA actual-pose scores",
                        gnina_path,
                        f"pose provenance mismatch for {target_id}",
                    )
                )
                break

    gnina_status_path = base / "gnina_pose_status_manifest.json"
    gnina_payload, gnina_status_check = _read_json(
        gnina_status_path, "GNINA actual-pose status manifest"
    )
    if _append_if_bad(checks, gnina_status_check):
        gnina_records, detail = _manifest_targets(
            gnina_payload, schema="skinscout.gnina-pose-status.v1"
        )
        if detail:
            checks.append(_invalid("GNINA actual-pose status manifest", gnina_status_path, detail))
        elif docking_ok and {str(row["target_id"]) for row in gnina_records} != set(docking["target_id"].astype(str)):
            checks.append(
                _invalid(
                    "GNINA actual-pose status manifest",
                    gnina_status_path,
                    "GNINA status targets must match successfully docked targets",
                )
            )
        elif gnina_ok:
            supported = {
                str(row["target_id"])
                for row in gnina_records
                if row.get("status") == "structure_supported"
            }
            if supported != set(gnina["target_id"].astype(str)):
                checks.append(
                    _invalid(
                        "GNINA actual-pose status manifest",
                        gnina_status_path,
                        "supported statuses do not match GNINA score rows",
                    )
                )

    canonical_path = base / "daina_structural_targets.csv"
    canonical, canonical_check = _read_csv(
        canonical_path, "canonical Daina structural target artifact"
    )
    if _append_if_bad(checks, canonical_check):
        detail = _validate_target_fast_frame(canonical, canonical_path)
        if detail:
            checks.append(
                _invalid("canonical Daina structural target artifact", canonical_path, detail)
            )
        elif not selected.empty and canonical["target_id"].astype(str).tolist() != selected["target_id"].astype(str).tolist():
            checks.append(
                _invalid(
                    "canonical Daina structural target artifact",
                    canonical_path,
                    "Daina primary target order changed during annotation",
                )
            )

    top50_path = base / "top50.csv"
    top50, top50_check = _read_csv(top50_path, "Daina top50 compatibility projection")
    if _append_if_bad(checks, top50_check):
        if len(top50) != 50:
            checks.append(
                _invalid(
                    "Daina top50 compatibility projection",
                    top50_path,
                    f"expected exactly 50 rows, found {len(top50)}",
                )
            )
        elif not canonical.empty and top50["target_id"].astype(str).tolist() != canonical.head(50)["target_id"].astype(str).tolist():
            checks.append(
                _invalid(
                    "Daina top50 compatibility projection",
                    top50_path,
                    "projection does not preserve the first 50 Daina ranks",
                )
            )

    delivered_path = base / "top50_band_reranked.csv"
    delivered, delivered_check = _read_csv(
        delivered_path,
        "reader-facing Daina band-reranked targets",
    )
    if _append_if_bad(checks, delivered_check):
        required = {
            "target_id",
            "daina_rank",
            "rerank_band",
            "final_rank",
            "ranking_basis",
        }
        missing = sorted(required - set(delivered.columns))
        if missing:
            checks.append(
                _invalid(
                    "reader-facing Daina band-reranked targets",
                    delivered_path,
                    "missing columns: " + ", ".join(missing),
                )
            )
        elif len(delivered) != 50:
            checks.append(
                _invalid(
                    "reader-facing Daina band-reranked targets",
                    delivered_path,
                    f"expected exactly 50 rows, found {len(delivered)}",
                )
            )
        elif delivered["final_rank"].astype(int).tolist() != list(range(1, 51)):
            checks.append(
                _invalid(
                    "reader-facing Daina band-reranked targets",
                    delivered_path,
                    "final_rank must be contiguous 1..50 in delivered order",
                )
            )
        elif not top50.empty:
            original_ids = top50["target_id"].astype(str).tolist()
            delivered_ids = delivered["target_id"].astype(str).tolist()
            if delivered_ids[:10] != original_ids[:10]:
                checks.append(
                    _invalid(
                        "reader-facing Daina band-reranked targets",
                        delivered_path,
                        "immutable Daina head ranks 1..10 changed",
                    )
                )
            elif set(delivered_ids) != set(original_ids):
                checks.append(
                    _invalid(
                        "reader-facing Daina band-reranked targets",
                        delivered_path,
                        "delivered top50 must contain exactly the canonical Daina top50 targets",
                    )
                )
            else:
                original_rank = {target_id: index for index, target_id in enumerate(original_ids, 1)}
                observed_rank = dict(
                    zip(
                        delivered_ids,
                        delivered["daina_rank"].astype(int).tolist(),
                        strict=True,
                    )
                )
                if observed_rank != original_rank:
                    checks.append(
                        _invalid(
                            "reader-facing Daina band-reranked targets",
                            delivered_path,
                            "daina_rank no longer records each target's canonical evidence rank",
                        )
                    )


def _verify_legacy_fast_target_intermediates(
    run_dir: Path, checks: list[OutputCheck]
) -> None:
    base = run_dir / "03_targets" / "mode_fast"
    _verify_table(
        checks,
        base / "psichic_proteome.tsv",
        "fast PSICHIC proteome scores",
        required_columns={"target_id", "psichic_score", "score"},
        numeric_columns={"psichic_score", "score"},
        sep="\t",
    )
    _verify_table(
        checks,
        base / "daina_zoete_proteome.tsv",
        "fast Daina-Zoete proteome scores",
        required_columns={"target_id", "max_tanimoto", "score"},
        numeric_columns={"max_tanimoto", "score"},
        sep="\t",
    )
    _verify_table(
        checks,
        base / "dti_rrf_top25pct.csv",
        "fast DTI RRF candidate set",
        required_columns={"target_id", "rrf_score", "source_count", "sources"},
        numeric_columns={"rrf_score"},
        min_source_count=2,
    )
    _verify_autodock_claim_scores(
        checks,
        base / "autodock_top5k.tsv",
        "fast AutoDock-GPU provenance",
    )
    _verify_table(
        checks,
        base / "top50.csv",
        "fast rerank consensus",
        required_columns={"target_id", "rrf_score", "source_count", "sources"},
        numeric_columns={"rrf_score"},
        min_source_count=2,
    )


def _verify_fast_target_intermediates(run_dir: Path, checks: list[OutputCheck]) -> None:
    if _uses_daina_structural_contract(run_dir):
        _verify_daina_fast_target_intermediates(run_dir, checks)
    else:
        _verify_legacy_fast_target_intermediates(run_dir, checks)


def _verify_comprehensive_target_intermediates(run_dir: Path, checks: list[OutputCheck]) -> None:
    base = run_dir / "03_targets" / "mode_comprehensive"
    _append_if_bad(checks, _check_path(base / "ligand.pdbqt", "comprehensive docking ligand PDBQT"))
    _verify_autodock_claim_scores(
        checks,
        base / "autodock_all_targets.tsv",
        "comprehensive AutoDock-GPU provenance",
    )
    _verify_table(
        checks,
        base / "top_pct_pre_rescore.csv",
        "comprehensive pre-rescore target set",
        required_columns={"target_id", "score"},
        numeric_columns={"score"},
    )
    _verify_table(
        checks,
        base / "gnina_rescores.tsv",
        "comprehensive GNINA rescores",
        required_columns={"target_id", "cnn_affinity", "score"},
        numeric_columns={"cnn_affinity", "score"},
        sep="\t",
    )
    _verify_table(
        checks,
        base / "rtmscore_rescores.tsv",
        "comprehensive RTMScore rescores",
        required_columns={"target_id", "rtm_score", "score"},
        numeric_columns={"rtm_score", "score"},
        sep="\t",
    )
    _verify_table(
        checks,
        base / "boltz2_affinity_top.tsv",
        "comprehensive Boltz-2 affinity scores",
        required_columns={"target_id", "boltz2_neg_log_uM", "score"},
        numeric_columns={"boltz2_neg_log_uM", "score"},
        sep="\t",
    )
    _verify_table(
        checks,
        base / "top50_4way_consensus.csv",
        "comprehensive four-way consensus",
        required_columns={"target_id", "rrf_score", "source_count", "sources"},
        numeric_columns={"rrf_score"},
        min_source_count=3,
    )


def _verify_target_intermediates(run_dir: Path, checks: list[OutputCheck], *, mode: str) -> None:
    if mode in {"fast", "both"}:
        _verify_fast_target_intermediates(run_dir, checks)
    if mode in {"comprehensive", "both"}:
        _verify_comprehensive_target_intermediates(run_dir, checks)


def _visible_html_text(text: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", text)
    return " ".join(unescape(without_tags).split())


def _visible_html_lines(text: str) -> list[str]:
    normalized = re.sub(r"<tr\b[^>]*>", "\n", text, flags=re.IGNORECASE)
    normalized = re.sub(r"</tr\s*>", "\n", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"</(p|li|div|h[1-6])\s*>", "\n", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"<(td|th)\b[^>]*>", " | ", normalized, flags=re.IGNORECASE)
    without_tags = re.sub(r"<[^>]+>", " ", normalized)
    return [
        " ".join(unescape(line).split())
        for line in without_tags.splitlines()
        if line.strip()
    ]


def _display_report_list(value: object) -> str:
    if isinstance(value, list):
        items = [str(item) for item in value if str(item).strip()]
        return ", ".join(items) if items else "none"
    if value is None:
        return "n/a"
    return str(value)


def _display_report_bool(value: object) -> str:
    return "yes" if value is True else "no" if value is False else str(value)


def _report_claimable(overall: dict[str, Any]) -> bool:
    return overall.get("claimable") is True


def _report_claim_status(overall: dict[str, Any]) -> str:
    action = overall.get("recommended_action")
    if overall.get("claimable") is True:
        return "proceed"
    if action == "review_before_claim":
        return "human review required before claim"
    if action == "stop_before_claim":
        return "not claimable; stop before claim"
    if action == "proceed":
        return "not claimable; summary claimable is not true"
    return "not claimable"


def _require_report_text(
    checks: list[OutputCheck],
    path: Path,
    visible_text: str,
    label: str,
    token: str,
) -> None:
    if token not in visible_text:
        checks.append(_invalid("Mol* HTML report", path, f"missing report value {label}: {token}"))


def _require_report_value_alternative(
    checks: list[OutputCheck],
    path: Path,
    visible_text: str,
    label: str,
    prefix: str,
    value: object,
) -> None:
    tokens = [
        f"{prefix}: {alternative}"
        for alternative in _report_value_alternatives(value)
    ]
    if not any(token in visible_text for token in tokens):
        display = _markdown_display_value(value)
        checks.append(
            _invalid(
                "Mol* HTML report",
                path,
                f"missing report value {label}: {prefix}: {display}",
            )
        )


def _require_report_equal_value_alternative(
    checks: list[OutputCheck],
    path: Path,
    visible_text: str,
    label: str,
    prefix: str,
    value: object,
) -> None:
    tokens = [
        f"{prefix} = {alternative}"
        for alternative in _report_value_alternatives(value)
    ]
    if not any(token in visible_text for token in tokens):
        display = _markdown_display_value(value)
        checks.append(
            _invalid(
                "Mol* HTML report",
                path,
                f"missing report value {label}: {prefix} = {display}",
            )
        )


def _require_report_metric_row(
    checks: list[OutputCheck],
    path: Path,
    visible_lines: list[str],
    metric: str,
    value: object,
) -> None:
    alternatives = _report_value_alternatives(value)
    if not any(
        "|" in line
        and metric in line
        and any(alternative in line for alternative in alternatives)
        for line in visible_lines
    ):
        checks.append(
            _invalid(
                "Mol* HTML report",
                path,
                "missing report value ADMET metric "
                f"{metric}: {metric}: {_markdown_display_value(value)}",
            )
        )


def _require_report_count_row(
    checks: list[OutputCheck],
    path: Path,
    visible_lines: list[str],
    stage: str,
    value: object,
) -> None:
    alternatives = _report_value_alternatives(value)
    if not any(
        "|" in line
        and stage in line
        and any(alternative in line for alternative in alternatives)
        for line in visible_lines
    ):
        checks.append(
            _invalid(
                "Mol* HTML report",
                path,
                "missing report value screening count "
                f"{stage}: {stage}: {_markdown_display_value(value)}",
            )
        )


def _report_target_detail_tokens(target: dict[str, Any]) -> list[str]:
    return [
        display
        for display, alternatives in _report_target_detail_token_groups(target)
        if display and alternatives
    ]


def _report_value_alternatives(value: object) -> tuple[str, ...]:
    alternatives = {_markdown_display_value(value)}
    if not _bool_like(value):
        try:
            numeric = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            numeric = math.nan
        if math.isfinite(numeric):
            alternatives.update(
                {
                    str(value),
                    str(numeric),
                    f"{numeric:.3f}",
                    f"{numeric:.4f}",
                    f"{numeric:.6f}",
                    f"{numeric:.4g}",
                    f"{numeric:.6g}",
                    f"{numeric:.12g}",
                }
            )
    return tuple(sorted(token for token in alternatives if token))


def _report_skin_sens_evidence_token_groups(
    run_dir: Path,
) -> list[tuple[str, list[tuple[str, tuple[str, ...]]]]]:
    rows: list[tuple[str, list[tuple[str, tuple[str, ...]]]]] = []
    for model in SKIN_SENS_MODELS:
        path = run_dir / "02_admet" / f"{model}.json"
        payload, check = _read_json(path, f"{model} skin-sens source evidence")
        if check.status != "ok":
            continue
        status = _text_or_none(payload.get("status"))
        groups: list[tuple[str, tuple[str, ...]]] = [(model, (model,))]
        if status is not None:
            groups.append((status, (status,)))
        if status == "ok":
            call, probability = _skin_sens_source_call(payload, model)
            if isinstance(call, str) and call.strip() in SKIN_SENS_CALLS:
                groups.append((call.strip(), (call.strip(),)))
            if probability is not None:
                groups.append((
                    _markdown_display_value(probability),
                    _report_value_alternatives(probability),
                ))
        rows.append((model, groups))
    return rows


def _report_target_detail_token_groups(
    target: dict[str, Any],
) -> list[tuple[str, tuple[str, ...]]]:
    groups: list[tuple[str, tuple[str, ...]]] = []
    for field in (
        "target_id",
        "gene_symbol",
        "protein_name",
        "final_score",
        "skin_score",
        "skin_tier",
        "docking_rrf",
    ):
        if field in target:
            value = target.get(field)
            if field in {"gene_symbol", "protein_name"} and _text_or_none(value) is None:
                continue
            groups.append((_markdown_display_value(value), _report_value_alternatives(value)))
    sources = target.get("sources")
    if isinstance(sources, list):
        groups.extend(
            (str(source), (str(source),))
            for source in sources
            if str(source).strip()
        )
    elif sources is not None:
        groups.extend(
            (part.strip(), (part.strip(),))
            for part in str(sources).replace(",", ";").split(";")
            if part.strip()
        )
    efficacy = target.get("efficacy")
    if isinstance(efficacy, list):
        groups.extend(
            (str(item), (str(item),))
            for item in efficacy
            if str(item).strip()
        )
    elif efficacy is not None:
        groups.append((str(efficacy), (str(efficacy),)))
    return groups


def _verify_report_summary_values(
    run_dir: Path,
    checks: list[OutputCheck],
    path: Path,
    visible_text: str,
    visible_lines: list[str],
) -> None:
    summary_path = run_dir / "run_summary.json"
    if not _nonempty(summary_path):
        return
    try:
        payload = json.loads(summary_path.read_text())
    except json.JSONDecodeError:
        return
    if not isinstance(payload, dict):
        return

    compound = payload.get("compound")
    if isinstance(compound, dict):
        input_type = compound.get("input_type")
        if input_type is not None:
            _require_report_text(
                checks,
                path,
                visible_text,
                "input type",
                f"Input type: {input_type}",
            )
        input_smiles = compound.get("input_smiles")
        if input_smiles is not None:
            _require_report_text(
                checks,
                path,
                visible_text,
                "input SMILES",
                f"Input SMILES: {input_smiles}",
            )
        input_canonical_smiles = compound.get("input_canonical_smiles")
        if input_canonical_smiles is not None:
            _require_report_text(
                checks,
                path,
                visible_text,
                "input canonical SMILES",
                f"Input canonical SMILES: {input_canonical_smiles}",
            )
        input_sdf = compound.get("input_sdf")
        if input_sdf is not None:
            _require_report_text(
                checks,
                path,
                visible_text,
                "input SDF",
                f"Input SDF: {input_sdf}",
            )
        canonical_smiles = compound.get("canonical_smiles")
        if canonical_smiles is not None:
            _require_report_text(
                checks,
                path,
                visible_text,
                "canonical SMILES",
                f"canonical SMILES: {canonical_smiles}",
            )
        inchikey = compound.get("inchikey")
        if inchikey is not None:
            _require_report_text(
                checks,
                path,
                visible_text,
                "InChIKey",
                f"InChIKey: {inchikey}",
            )

    overall = payload.get("overall_decision")
    if isinstance(overall, dict):
        decision = overall.get("decision")
        if decision is not None:
            _require_report_text(
                checks,
                path,
                visible_text,
                "overall decision",
                f"Overall decision: {decision}",
            )
        action = overall.get("recommended_action")
        if action is not None:
            _require_report_text(
                checks,
                path,
                visible_text,
                "recommended action",
                f"Recommended action: {action}",
            )
        _require_report_text(
            checks,
            path,
            visible_text,
            "claimable",
            f"Claimable: {'yes' if _report_claimable(overall) else 'no'}",
        )
        _require_report_text(
            checks,
            path,
            visible_text,
            "claim status",
            f"Claim status: {_report_claim_status(overall)}",
        )

    skin_toxicity = payload.get("skin_toxicity")
    if isinstance(skin_toxicity, dict) and skin_toxicity.get("decision") is not None:
        _require_report_text(
            checks,
            path,
            visible_text,
            "skin toxicity decision",
            f"Skin toxicity: {skin_toxicity.get('decision')}",
        )
        if skin_toxicity.get("skin_reaction_value") is not None:
            _require_report_value_alternative(
                checks,
                path,
                visible_text,
                "Skin_Reaction value",
                "Skin_Reaction",
                skin_toxicity.get("skin_reaction_value"),
            )
        if skin_toxicity.get("skin_reaction_risk_level") is not None:
            _require_report_text(
                checks,
                path,
                visible_text,
                "Skin_Reaction risk",
                f"({skin_toxicity.get('skin_reaction_risk_level')})",
            )
        structural_flags = skin_toxicity.get("structural_alert_flags")
        if isinstance(structural_flags, list):
            _require_report_text(
                checks,
                path,
                visible_text,
                "structural alert flags",
                "Structural alerts: " + _display_report_list(structural_flags),
            )
        if skin_toxicity.get("degraded") is not None:
            _require_report_text(
                checks,
                path,
                visible_text,
                "degraded skin-sens evidence",
                "degraded skin-sens evidence: "
                + str(skin_toxicity.get("degraded")),
            )
        missing_models = skin_toxicity.get("missing_models")
        if isinstance(missing_models, list):
            _require_report_text(
                checks,
                path,
                visible_text,
                "missing skin-sens models",
                "missing skin-sens models: "
                + _display_report_list(missing_models),
            )

    if isinstance(payload.get("safety"), dict):
        _require_report_text(
            checks,
            path,
            visible_text,
            "skin-sens 3-model evidence",
            "Skin-sens 3-model evidence",
        )
        for model, detail_groups in _report_skin_sens_evidence_token_groups(run_dir):
            if not detail_groups:
                continue
            if not any(
                all(
                    any(alternative in line for alternative in alternatives)
                    for _, alternatives in detail_groups
                    if alternatives
                )
                for line in visible_lines
                if "|" in line
            ):
                checks.append(
                    _invalid(
                        "Mol* HTML report",
                        path,
                        "missing report value skin-sens evidence row "
                        f"{model}: "
                        + " | ".join(display for display, _ in detail_groups),
                    )
                )

    safety = payload.get("safety")
    admet_risk = safety.get("admet_risk_assessment") if isinstance(safety, dict) else None
    if isinstance(admet_risk, dict):
        _require_report_text(
            checks,
            path,
            visible_text,
            "high ADMET risk endpoints",
            "High ADMET risk endpoints: "
            + _display_report_list(admet_risk.get("high_risk_endpoints")),
        )
        _require_report_text(
            checks,
            path,
            visible_text,
            "moderate ADMET risk endpoints",
            "Moderate ADMET risk endpoints: "
            + _display_report_list(admet_risk.get("moderate_risk_endpoints")),
        )
    admet_metrics = safety.get("admet_metrics") if isinstance(safety, dict) else None
    if isinstance(admet_metrics, dict) and admet_metrics:
        _require_report_text(
            checks,
            path,
            visible_text,
            "summary ADMET metrics section",
            "Summary ADMET metrics",
        )
        for metric in SUMMARY_ADMET_METRICS:
            if metric in admet_metrics:
                _require_report_metric_row(
                    checks,
                    path,
                    visible_lines,
                    metric,
                    admet_metrics.get(metric),
                )

    cosmetic_drug = payload.get("cosmetic_drug")
    if isinstance(cosmetic_drug, dict):
        if cosmetic_drug.get("decision") is not None:
            _require_report_text(
                checks,
                path,
                visible_text,
                "cosmetic/drug decision",
                f"Cosmetic/drug decision: {cosmetic_drug.get('decision')}",
            )
        if cosmetic_drug.get("drug_policy") is not None:
            _require_report_text(
                checks,
                path,
                visible_text,
                "drug policy",
                f"Drug policy: {cosmetic_drug.get('drug_policy')}",
            )
        if cosmetic_drug.get("cosing_level") is not None:
            _require_report_text(
                checks,
                path,
                visible_text,
                "CosIng level",
                f"CosIng level: {cosmetic_drug.get('cosing_level')}",
            )
        if "inci" in cosmetic_drug:
            inci = _text_or_none(cosmetic_drug.get("inci")) or "—"
            _require_report_text(
                checks,
                path,
                visible_text,
                "INCI",
                f"INCI = {inci}",
            )
        functions = cosmetic_drug.get("cosing_functions")
        if isinstance(functions, list):
            function_text = (
                ", ".join(str(item) for item in functions if str(item).strip())
                or "—"
            )
            _require_report_text(
                checks,
                path,
                visible_text,
                "CosIng functions",
                f"functions = {function_text}",
            )
        if cosmetic_drug.get("cosing_tanimoto") is not None:
            _require_report_equal_value_alternative(
                checks,
                path,
                visible_text,
                "CosIng tanimoto",
                "Tanimoto",
                cosmetic_drug.get("cosing_tanimoto"),
            )
        if cosmetic_drug.get("max_tanimoto_to_approved_drug") is not None:
            _require_report_equal_value_alternative(
                checks,
                path,
                visible_text,
                "drug max Tanimoto",
                "max Tanimoto vs approved",
                cosmetic_drug.get("max_tanimoto_to_approved_drug"),
            )
        if cosmetic_drug.get("n_warnings") is not None:
            _require_report_text(
                checks,
                path,
                visible_text,
                "drug warning count",
                f"Drug-avoidance warnings: {cosmetic_drug.get('n_warnings')}",
            )

    skin_binding = payload.get("skin_specialized_binding")
    if isinstance(skin_binding, dict):
        for field, label, prefix in (
            ("skin_context_decision", "skin context decision", "Skin context decision"),
            ("skin_context_supported", "skin context supported", "Skin context supported"),
            ("skin_expression_supported", "skin expression supported", "Skin expression supported"),
            ("skin_efficacy_supported", "skin efficacy supported", "Skin efficacy supported"),
        ):
            if field not in skin_binding:
                continue
            value = skin_binding[field]
            rendered = _display_report_bool(value) if isinstance(value, bool) else str(value)
            _require_report_text(
                checks,
                path,
                visible_text,
                label,
                f"{prefix}: {rendered}",
            )
        top_target = skin_binding.get("top_target_id")
        if top_target is not None:
            _require_report_text(
                checks,
                path,
                visible_text,
                "top binding target",
                f"Top binding target: {top_target}",
            )
        if skin_binding.get("top_target_final_score") is not None:
            _require_report_value_alternative(
                checks,
                path,
                visible_text,
                "top target final score",
                "top target final score",
                skin_binding.get("top_target_final_score"),
            )
        if skin_binding.get("top_target_docking_rrf") is not None:
            _require_report_value_alternative(
                checks,
                path,
                visible_text,
                "top target docking RRF",
                "top target docking RRF",
                skin_binding.get("top_target_docking_rrf"),
            )
        if skin_binding.get("top_target_source_count") is not None:
            _require_report_text(
                checks,
                path,
                visible_text,
                "top target source count",
                f"top target source count: {skin_binding.get('top_target_source_count')}",
            )
        top_target_sources = skin_binding.get("top_target_sources")
        if isinstance(top_target_sources, list):
            _require_report_text(
                checks,
                path,
                visible_text,
                "top target sources",
                "top target sources: " + _display_report_list(top_target_sources),
            )
        if skin_binding.get("top_target_skin_score") is not None:
            _require_report_value_alternative(
                checks,
                path,
                visible_text,
                "top target skin score",
                "top target skin score",
                skin_binding.get("top_target_skin_score"),
            )
        if skin_binding.get("top_target_skin_tier") is not None:
            _require_report_text(
                checks,
                path,
                visible_text,
                "top target skin tier",
                f"top target skin tier: {skin_binding.get('top_target_skin_tier')}",
            )
        top_target_efficacy = skin_binding.get("top_target_efficacy")
        if isinstance(top_target_efficacy, list):
            _require_report_text(
                checks,
                path,
                visible_text,
                "top target skin efficacy",
                "top target skin efficacy: "
                + _display_report_list(top_target_efficacy),
            )
        if skin_binding.get("top_target_skin_context_supported") is not None:
            _require_report_text(
                checks,
                path,
                visible_text,
                "top target skin context supported",
                "top target skin context supported: "
                + _display_report_bool(
                    skin_binding.get("top_target_skin_context_supported")
                ),
            )
        most_skin_relevant = skin_binding.get("most_skin_relevant_target")
        if isinstance(most_skin_relevant, dict):
            most_target = most_skin_relevant.get("target_id")
            if most_target is not None:
                _require_report_text(
                    checks,
                    path,
                    visible_text,
                    "most skin-relevant target",
                    f"Most skin-relevant top target: {most_target}",
                )
            for field, label in (
                ("gene_symbol", "most skin-relevant gene"),
                ("protein_name", "most skin-relevant protein"),
            ):
                value = _text_or_none(most_skin_relevant.get(field))
                if value is not None:
                    _require_report_text(
                        checks,
                        path,
                        visible_text,
                        label,
                        f"{label}: {value}",
                    )
            for field, label, prefix in (
                (
                    "final_score",
                    "most skin-relevant final score",
                    "most skin-relevant final score",
                ),
                (
                    "docking_rrf",
                    "most skin-relevant docking RRF",
                    "most skin-relevant docking RRF",
                ),
                (
                    "skin_score",
                    "most skin-relevant skin score",
                    "most skin-relevant skin score",
                ),
            ):
                if most_skin_relevant.get(field) is not None:
                    _require_report_value_alternative(
                        checks,
                        path,
                        visible_text,
                        label,
                        prefix,
                        most_skin_relevant.get(field),
                    )
            if most_skin_relevant.get("source_count") is not None:
                _require_report_text(
                    checks,
                    path,
                    visible_text,
                    "most skin-relevant source count",
                    "most skin-relevant source count: "
                    + str(most_skin_relevant.get("source_count")),
                )
            most_skin_sources = most_skin_relevant.get("sources")
            if isinstance(most_skin_sources, list):
                _require_report_text(
                    checks,
                    path,
                    visible_text,
                    "most skin-relevant sources",
                    "most skin-relevant sources: "
                    + _display_report_list(most_skin_sources),
                )
            if most_skin_relevant.get("skin_tier") is not None:
                _require_report_text(
                    checks,
                    path,
                    visible_text,
                    "most skin-relevant skin tier",
                    "most skin-relevant skin tier: "
                    + str(most_skin_relevant.get("skin_tier")),
                )
            most_skin_efficacy = most_skin_relevant.get("efficacy")
            if isinstance(most_skin_efficacy, list):
                _require_report_text(
                    checks,
                    path,
                    visible_text,
                    "most skin-relevant efficacy",
                    "most skin-relevant efficacy: "
                    + _display_report_list(most_skin_efficacy),
                )

    target_prediction = payload.get("target_prediction")
    if isinstance(target_prediction, dict):
        if target_prediction.get("screened_target_count") is not None:
            _require_report_text(
                checks,
                path,
                visible_text,
                "screened target candidates",
                "Screened target candidates: "
                + str(target_prediction.get("screened_target_count")),
            )
        screening_counts = target_prediction.get("screening_counts")
        if isinstance(screening_counts, dict):
            _require_report_text(
                checks,
                path,
                visible_text,
                "screening stage counts section",
                "Screening stage counts",
            )
            for stage, count in screening_counts.items():
                _require_report_count_row(
                    checks,
                    path,
                    visible_lines,
                    str(stage),
                    count,
                )
        top_targets = target_prediction.get("top_targets")
    else:
        top_targets = None
    if isinstance(top_targets, list):
        for rank, target in enumerate(top_targets, start=1):
            if not isinstance(target, dict):
                continue
            target_id = target.get("target_id")
            if target_id is None:
                continue
            _require_report_text(
                checks,
                path,
                visible_text,
                f"top target {rank} target_id",
                str(target_id),
            )
            detail_groups = _report_target_detail_token_groups(target)
            candidate_lines = [
                line for line in visible_lines if str(target_id) in line and "|" in line
            ]
            if not any(
                all(
                    any(alternative in line for alternative in alternatives)
                    for _, alternatives in detail_groups
                    if alternatives
                )
                for line in candidate_lines
            ):
                detail_tokens = [
                    display
                    for display, alternatives in detail_groups
                    if display and alternatives
                ]
                checks.append(
                    _invalid(
                        "Mol* HTML report",
                        path,
                        "missing report value top target "
                        f"{rank} row details: {' | '.join(detail_tokens)}",
                    )
                )


def _verify_report(run_dir: Path, checks: list[OutputCheck]) -> None:
    path = run_dir / "09_report" / "index.html"
    check = _check_path(path, "Mol* HTML report")
    if not _append_if_bad(checks, check):
        return
    text = path.read_text(errors="replace")
    lowered = text.lower()
    visible_text = _visible_html_text(text)
    visible_lines = _visible_html_lines(text)
    for token in (
        "<html",
        "admet",
        "ranked targets",
        "skin-efficacy",
        "skin toxicity",
        "overall decision",
        "recommended action",
        "claimable",
        "high admet risk endpoints",
        "drug-avoidance warnings",
        "screened target candidates",
        "screening stage counts",
        "skin context",
        "skin context supported",
    ):
        if token not in lowered:
            checks.append(_invalid("Mol* HTML report", path, f"missing text marker {token!r}"))
            break
    if "placeholder" in lowered:
        checks.append(_invalid("Mol* HTML report", path, "contains placeholder marker"))
    _verify_report_summary_values(run_dir, checks, path, visible_text, visible_lines)


def verify_run(
    run_dir: Path,
    *,
    preset: str,
    mode: str,
    min_targets: int,
    allow_degraded: bool,
    target_metadata_path: Path | None = None,
) -> dict[str, Any]:
    checks: list[OutputCheck] = []
    if not run_dir.exists() or not run_dir.is_dir():
        checks.append(OutputCheck("run directory", "missing", str(run_dir)))
        return {
            "schema_version": VERIFY_OUTPUT_SCHEMA,
            "status": "failed",
            "run_dir": str(run_dir),
            "preset": preset,
            "mode": mode,
            "checks": [asdict(c) for c in checks],
        }
    target_metadata = (
        _load_target_metadata(target_metadata_path)
        if preset in {"target-id", "report"}
        else {}
    )
    _verify_run_manifest_contract(
        run_dir,
        checks,
        preset=preset,
        mode=mode,
    )

    compound_metadata = _verify_compound(run_dir, checks)
    safety_decision = _verify_admet(
        run_dir,
        checks,
        compound_smiles=(
            compound_metadata.get("canonical_smiles")
            if compound_metadata
            else None
        ),
        allow_degraded=allow_degraded,
    )
    cosmetic_decision, cosmetic_policy = _verify_cosmetic_drug(
        run_dir,
        checks,
        allow_degraded=allow_degraded,
    )

    if preset in {"target-id", "report"}:
        if safety_decision == "HALT":
            checks.append(
                OutputCheck(
                    "target-id gate",
                    "blocked",
                    str(run_dir / "02_admet" / "skin_sens_decision.txt"),
                    "skin-sens decision HALT prevents target prediction",
                )
            )
        if cosmetic_decision == "HALT":
            checks.append(
                OutputCheck(
                    "target-id gate",
                    "blocked",
                    str(run_dir / "02b_cosmetic_drug" / "cosmetic_drug_decision.txt"),
                    "cosmetic/drug decision HALT prevents target prediction",
                )
            )
        _verify_target_intermediates(run_dir, checks, mode=mode)
        _verify_target_ranking(run_dir, checks, mode=mode, min_targets=min_targets)
    if preset == "report":
        _verify_report(run_dir, checks)
    if preset in {"safety", "target-id", "report"}:
        _verify_run_summary(
            run_dir,
            checks,
            preset=preset,
            mode=mode,
            compound_metadata=compound_metadata,
            compound_smiles=(
                compound_metadata.get("canonical_smiles")
                if compound_metadata
                else None
            ),
            compound_inchikey=(
                compound_metadata.get("inchikey")
                if compound_metadata
                else None
            ),
            safety_decision=safety_decision,
            cosmetic_decision=cosmetic_decision,
            cosmetic_policy=cosmetic_policy,
            target_metadata=target_metadata,
        )

    status = "ok" if all(check.status == "ok" for check in checks) else "failed"
    return {
        "schema_version": VERIFY_OUTPUT_SCHEMA,
        "status": status,
        "run_dir": str(run_dir),
        "preset": preset,
        "mode": mode,
        "checks": [asdict(check) for check in checks],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="target-id")
    parser.add_argument("--mode", choices=sorted(MODES), default="comprehensive")
    parser.add_argument("--min-targets", type=int, default=1)
    parser.add_argument("--allow-degraded", action="store_true")
    parser.add_argument("--target-metadata", type=Path,
                        help="Optional HPA-style TSV mapping UniProt IDs to gene/protein names.")
    parser.add_argument("--json", action="store_true", help="Emit JSON only")
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.min_targets < 1:
        raise SystemExit("--min-targets must be >= 1")
    payload = verify_run(
        args.run_dir,
        preset=args.preset,
        mode=args.mode,
        min_targets=args.min_targets,
        allow_degraded=args.allow_degraded,
        target_metadata_path=args.target_metadata,
    )
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.json_out.with_suffix(args.json_out.suffix + ".tmp")
        tmp.write_text(text)
        tmp.replace(args.json_out)
    if args.json:
        sys.stdout.write(text)
    else:
        print(f"SkinScout run output verification: {payload['status']}")
        print(f"run_dir={payload['run_dir']} preset={payload['preset']} mode={payload['mode']}")
        for check in payload["checks"]:
            suffix = f" - {check['detail']}" if check.get("detail") else ""
            print(f"[{check['status']}] {check['name']}: {check['path']}{suffix}")
    return 0 if payload["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
