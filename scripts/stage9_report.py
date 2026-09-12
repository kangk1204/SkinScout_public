#!/usr/bin/env python3
"""stage9_report.py — Build the Mol* integrated HTML report.

Required panels (INSTRUCTIONS.md §12 v2 / §15 v3):
  Panel A : ranked targets (with skin-expression / efficacy columns when v3)
  Panel A2: AutoDock-GPU 전수 distribution histogram + top-50 highlight
  Panel A3: DTI vs Docking disagreement scatter (only when run_dti_sanity=true)
  Panel B : 3D Mol* complex viewer (lazy-loaded per target)
  Panel C : ADMET radar + skin-sens 3-consensus
  Panel D : INCI annotation + Drug avoidance (v3, optional)
  Panel E : Generated analogs table (v3, optional)
  Panel F : MD stability + MM-GBSA bar
  Panel G : Synthesis tree (v3, optional)
  Panel H : Skin-efficacy KG inference (v3, optional)
  Panel I : Free-ligand QM conformer/DFT summary
"""

from __future__ import annotations

import argparse
import errno
import hashlib
from html import escape
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

import pandas as pd
from rdkit import Chem

from cosmetic_drug_contract import (
    CosmeticDrugContractError,
    parse_decision_and_policy,
    validate_decision_matches_drug_warnings,
)

LOG = logging.getLogger("stage9.report")
ROOT = Path(__file__).resolve().parents[1]
REPORT_ASSET_DIR = ROOT / "scripts" / "report_assets"
DEFAULT_MOLSTAR_DIR = REPORT_ASSET_DIR / "molstar"
DEFAULT_MOLSTAR_BUNDLE = DEFAULT_MOLSTAR_DIR / "molstar.js"
DEFAULT_MOLSTAR_STYLESHEET = DEFAULT_MOLSTAR_DIR / "molstar.css"
DEFAULT_MOLSTAR_LICENSE = DEFAULT_MOLSTAR_DIR / "LICENSE"
DEFAULT_MOLSTAR_ASSET_MANIFEST = DEFAULT_MOLSTAR_DIR / "ASSET_MANIFEST.json"
DEFAULT_FAST_REPORT_SCRIPT = REPORT_ASSET_DIR / "fast_report.js"
DEFAULT_FAST_REPORT_STYLESHEET = REPORT_ASSET_DIR / "fast_report.css"
FAST_REPORT_SCHEMA_VERSION = "skinscout.report_fast.v1"
PHYSICS_REPORT_SCHEMA_VERSION = "skinscout.report_physics.v1"
REPORT_POINTER_SCHEMA_VERSION = "skinscout.report_pointer.v1"
REPORT_IDENTITY_SCHEMA_VERSION = "skinscout.report_identity.v1"
POSE_MANIFEST_SCHEMA_VERSION = "skinscout.docking_pose_manifest.v1"
POSE_MANIFEST_NAME = "pose_manifest.json"
FAST_REPORT_TARGET_LIMIT = 10
SAFE_TARGET_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _preformatted_json(value: Any, *, limit: int | None = None) -> str:
    text = json.dumps(value, indent=2)
    if limit is not None:
        text = text[:limit]
    return f"<pre>{escape(text)}</pre>"

STAGE3_RANKING_REQUIRED_COLUMNS = {
    "target_id",
    "final_score",
    "skin_score",
    "source_count",
    "sources",
}
ADMET_MODERATE_RISK_THRESHOLD = 0.25
ADMET_HIGH_RISK_THRESHOLD = 0.50
ADMET_RISK_ENDPOINTS = ("AMES", "ClinTox", "DILI", "Skin_Reaction", "hERG")
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
FAST_SCREENING_COUNT_ARTIFACTS = (
    ("daina_zoete_targets", "03_targets/mode_fast/daina_zoete_proteome.tsv"),
    ("daina_primary_candidates", "03_targets/mode_fast/daina_top256.csv"),
    ("autodock_rescored_targets", "03_targets/mode_fast/autodock_top5k.tsv"),
    ("gnina_pose_rescored_targets", "03_targets/mode_fast/gnina_pose_rescores.tsv"),
    ("daina_structural_targets", "03_targets/mode_fast/daina_structural_targets.csv"),
    ("rerank_consensus_targets", "03_targets/mode_fast/top50.csv"),
    ("band_reranked_targets", "03_targets/mode_fast/top50_band_reranked.csv"),
)
LEGACY_FAST_SCREENING_COUNT_ARTIFACTS = (
    ("psichic_proteome_targets", "03_targets/mode_fast/psichic_proteome.tsv"),
    ("daina_zoete_targets", "03_targets/mode_fast/daina_zoete_proteome.tsv"),
    ("dti_rrf_candidates", "03_targets/mode_fast/dti_rrf_top25pct.csv"),
    ("autodock_rescored_targets", "03_targets/mode_fast/autodock_top5k.tsv"),
    ("rerank_consensus_targets", "03_targets/mode_fast/top50.csv"),
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
SKIN_SENS_MODELS = ("husspred", "stoptox", "pred_skin")
SKIN_SENS_CALLS = {"positive", "negative"}
SKIN_TOXICITY_DECISION_ORDER = {"PASS": 0, "REVIEW": 1, "HALT": 2}
SKIN_TOXICITY_LEVEL_ORDER = {"low": 0, "moderate": 1, "high": 2}
SKIN_EXPRESSION_SUPPORTED_TIERS = {"low", "medium", "high", "very_high"}
SKIN_EXPRESSION_SUPPORTED_THRESHOLD = 0.20
OVERALL_DECISION_ORDER = {"PASS": 0, "FLAG_HIGH": 1, "HALT": 2}


HTML_HEAD = """\
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <style>
    body{{font-family:-apple-system,Segoe UI,sans-serif;max-width:1200px;margin:auto;padding:24px;}}
    h1{{margin:0 0 8px;font-size:22px;}}
    h2{{margin-top:32px;padding-bottom:4px;border-bottom:1px solid #ddd;}}
    table{{border-collapse:collapse;width:100%;font-size:13px;}}
    th,td{{border:1px solid #e0e0e0;padding:4px 8px;text-align:left;}}
    th{{background:#fafafa;}}
    .badge{{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600;}}
    .pass{{background:#e6f7e6;color:#155724;}}
    .flag{{background:#fff3cd;color:#856404;}}
    .halt{{background:#f8d7da;color:#721c24;}}
    pre{{background:#f6f8fa;padding:12px;border-radius:6px;overflow-x:auto;font-size:12px;}}
    #molstar-viewer{{height:560px;border:1px solid #ddd;background:#111;}}
  </style>
</head>
<body>
"""

FAST_REPORT_HTML_HEAD = """\
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'self' 'unsafe-eval' 'wasm-unsafe-eval'; style-src 'self'; img-src 'self' data: blob:; worker-src 'self' blob:; connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'none'">
  <meta name="referrer" content="no-referrer" />
  <title>{title}</title>
  <link rel="stylesheet" href="assets/molstar.css" />
  <link rel="stylesheet" href="assets/report.css" />
  <script src="assets/molstar.js" defer></script>
  <script src="assets/report.js" defer></script>
</head>
<body>
"""


def _nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _read_json_required(path: Path, label: str) -> dict[str, Any]:
    if not _nonempty(path):
        raise SystemExit(f"{label} is required and must be non-empty: {path}")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} must be a JSON object: {path}")
    return payload


def _uses_daina_structural_contract(run_dir: Path) -> bool:
    manifest_path = run_dir / "run_manifest.json"
    if _nonempty(manifest_path):
        payload = _read_json_required(manifest_path, "Run manifest")
        schema = payload.get("schema_version")
        if schema == "skinscout.run_manifest.v3":
            return True
        if schema == "skinscout.run_manifest.v2":
            return False
        raise SystemExit(f"unsupported run manifest schema: {schema!r}")
    return _nonempty(
        run_dir / "03_targets" / "mode_fast" / "daina_structural_targets.csv"
    )


def _read_first_line_required(path: Path, label: str) -> str:
    if not _nonempty(path):
        raise SystemExit(f"{label} is required and must be non-empty: {path}")
    for line in path.read_text().splitlines():
        text = line.strip()
        if text:
            return text
    raise SystemExit(f"{label} contains no non-empty lines: {path}")


def _read_json_optional(path_value: str, label: str) -> dict[str, Any] | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists():
        raise SystemExit(f"{label} is required when its path is provided: {path}")
    if not _nonempty(path):
        raise SystemExit(f"{label} is empty: {path}")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} must be a JSON object: {path}")
    return payload


def _required_json_string(payload: dict[str, Any], key: str, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"{label} missing non-empty string field '{key}'")
    return value.strip()


def _optional_json_string(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _canonical_smiles(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"{label} missing non-empty smiles")
    smiles = value.strip()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise SystemExit(f"{label} contains invalid SMILES: {smiles!r}")
    return Chem.MolToSmiles(mol, canonical=True)


def _validate_source_smiles(
    payload: dict[str, Any],
    compound_smiles: str,
    label: str,
) -> None:
    source_smiles = _canonical_smiles(payload.get("smiles"), f"{label} smiles")
    expected_smiles = _canonical_smiles(compound_smiles, "Compound metadata canonical_smiles")
    if source_smiles != expected_smiles:
        raise SystemExit(f"{label} smiles does not match compound canonical_smiles")


def _validated_compound_metadata(payload: dict[str, Any]) -> dict[str, str]:
    inchikey = _required_json_string(payload, "inchikey", "Compound metadata")
    canonical_smiles = _required_json_string(
        payload,
        "canonical_smiles",
        "Compound metadata",
    )
    expected_canonical = _canonical_smiles(
        canonical_smiles,
        "Compound metadata field 'canonical_smiles'",
    )
    if canonical_smiles != expected_canonical:
        raise SystemExit(
            "Compound metadata field 'canonical_smiles' is not canonical: "
            f"{canonical_smiles!r} != {expected_canonical!r}"
        )
    mol = Chem.MolFromSmiles(canonical_smiles)
    expected_inchikey = Chem.MolToInchiKey(mol) if mol is not None else None
    if inchikey != expected_inchikey:
        raise SystemExit(
            "Compound metadata field 'inchikey' does not match canonical_smiles: "
            f"{inchikey!r} != {expected_inchikey!r}"
        )

    input_type = _required_json_string(payload, "input_type", "Compound metadata")
    if input_type not in {"smiles", "sdf"}:
        raise SystemExit(
            "Compound metadata field 'input_type' must be one of: sdf, smiles"
        )
    result = {
        "input_type": input_type,
        "canonical_smiles": canonical_smiles,
        "inchikey": inchikey,
    }
    if input_type == "smiles":
        if _optional_json_string(payload, "input_sdf") is not None:
            raise SystemExit(
                "Compound metadata field 'input_sdf' is not allowed when "
                "input_type is 'smiles'"
            )
        input_smiles = _required_json_string(
            payload,
            "input_smiles",
            "Compound metadata",
        )
        input_canonical_smiles = _required_json_string(
            payload,
            "input_canonical_smiles",
            "Compound metadata",
        )
        expected_input_canonical = _canonical_smiles(
            input_smiles,
            "Compound metadata field 'input_smiles'",
        )
        if input_canonical_smiles != expected_input_canonical:
            raise SystemExit(
                "Compound metadata field 'input_canonical_smiles' does not match "
                "input_smiles: "
                f"{input_canonical_smiles!r} != {expected_input_canonical!r}"
            )
        result["input_smiles"] = input_smiles
        result["input_canonical_smiles"] = input_canonical_smiles
    if input_type == "sdf":
        for field in ("input_smiles", "input_canonical_smiles"):
            if _optional_json_string(payload, field) is not None:
                raise SystemExit(
                    f"Compound metadata field '{field}' is not allowed when "
                    "input_type is 'sdf'"
                )
        result["input_sdf"] = _required_json_string(
            payload,
            "input_sdf",
            "Compound metadata",
        )
    return result


def _compound_identity_html(compound: dict[str, str]) -> str:
    parts = [f"Input type: <code>{escape(compound['input_type'])}</code>"]
    if compound["input_type"] == "smiles":
        parts.extend([
            f"Input SMILES: <code>{escape(compound['input_smiles'])}</code>",
            "Input canonical SMILES: "
            f"<code>{escape(compound['input_canonical_smiles'])}</code>",
        ])
    if compound["input_type"] == "sdf":
        parts.append(f"Input SDF: <code>{escape(compound['input_sdf'])}</code>")
    parts.extend([
        f"InChIKey: <code>{escape(compound['inchikey'])}</code>",
        f"canonical SMILES: <code>{escape(compound['canonical_smiles'])}</code>",
    ])
    return " · ".join(parts)


def _required_json_number(
    payload: dict[str, Any],
    key: str,
    label: str,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    if key not in payload:
        raise SystemExit(f"{label} missing required metric '{key}'")
    value = payload[key]
    if _is_bool_like(value):
        raise SystemExit(f"{label} metric '{key}' must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{label} metric '{key}' must be numeric") from exc
    if not math.isfinite(parsed):
        raise SystemExit(f"{label} metric '{key}' must be finite")
    if min_value is not None and parsed < min_value:
        raise SystemExit(f"{label} metric '{key}' must be >= {min_value:g}")
    if max_value is not None and parsed > max_value:
        raise SystemExit(f"{label} metric '{key}' must be <= {max_value:g}")
    return parsed


def _required_json_nonnegative_int(
    payload: dict[str, Any],
    key: str,
    label: str,
) -> int:
    value = _required_json_number(payload, key, label, min_value=0.0)
    if not value.is_integer():
        raise SystemExit(f"{label} metric '{key}' must be a non-negative integer")
    return int(value)


def _required_json_string_list(
    payload: dict[str, Any],
    key: str,
    label: str,
) -> list[str]:
    if key not in payload:
        raise SystemExit(f"{label} missing required list field '{key}'")
    value = payload[key]
    if not isinstance(value, list):
        raise SystemExit(f"{label} field '{key}' must be a list")
    out: list[str] = []
    for idx, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise SystemExit(
                f"{label} field '{key}' contains non-empty string violation "
                f"at index {idx}"
            )
        out.append(item.strip())
    return out


def _validate_reference_status(payload: dict[str, Any], label: str) -> None:
    status = payload.get("reference_status")
    if not isinstance(status, str) or not status.strip():
        raise SystemExit(f"{label} missing non-empty string field 'reference_status'")
    if status.strip() != "ok":
        raise SystemExit(f"{label} reference_status must be ok: {status.strip()}")


def _validate_cosing_annotation(payload: dict[str, Any]) -> dict[str, Any]:
    label = "CosIng annotation"
    level = _required_json_string(payload, "level", label)
    allowed = {"EXACT", "SIMILAR", "ANALOG", "NEW"}
    if level not in allowed:
        raise SystemExit(f"{label} field 'level' must be one of {sorted(allowed)}")
    inci_value = payload.get("inci")
    if level == "NEW":
        inci = None if inci_value is None else str(inci_value).strip()
    elif not isinstance(inci_value, str) or not inci_value.strip():
        raise SystemExit(f"{label} missing non-empty string field 'inci'")
    else:
        inci = inci_value.strip()
    functions = _required_json_string_list(payload, "functions", label)
    tanimoto = _required_json_number(
        payload,
        "tanimoto",
        label,
        min_value=0.0,
        max_value=1.0,
    )
    _validate_reference_status(payload, label)
    return {
        "level": level,
        "inci": inci,
        "functions": functions,
        "tanimoto": tanimoto,
    }


def _validate_drug_warnings(payload: dict[str, Any]) -> dict[str, Any]:
    label = "Drug-avoidance warnings"
    max_tanimoto = _required_json_number(
        payload,
        "max_tanimoto_to_approved_drug",
        label,
        min_value=0.0,
        max_value=1.0,
    )
    n_warnings = _required_json_nonnegative_int(payload, "n_warnings", label)
    warnings = payload.get("warnings")
    if not isinstance(warnings, list):
        raise SystemExit(f"{label} field 'warnings' must be a list")
    if n_warnings != len(warnings):
        raise SystemExit(
            f"{label} metric 'n_warnings' must equal warnings list length"
        )
    _validate_reference_status(payload, label)
    return {
        "max_tanimoto_to_approved_drug": max_tanimoto,
        "n_warnings": n_warnings,
        "warnings": warnings,
    }


def _skin_sens_decision(admet: dict[str, Any], decision_path: Path) -> str:
    skin_sens = admet.get("skin_sens")
    if not isinstance(skin_sens, dict):
        raise SystemExit("ADMET report missing object field 'skin_sens'")
    decision = skin_sens.get("decision")
    if not isinstance(decision, str) or not decision.strip():
        raise SystemExit("ADMET report missing non-empty skin_sens.decision")
    decision = decision.strip()
    allowed = {"HALT", "FLAG_HIGH", "PASS"}
    if decision not in allowed:
        raise SystemExit(
            "ADMET report skin_sens.decision must be one of "
            f"{sorted(allowed)}: {decision}"
        )
    skin_sens["decision"] = decision
    file_decision = _read_first_line_required(decision_path, "Skin-sens decision")
    if file_decision != decision:
        raise SystemExit("Skin-sens decision file does not match ADMET report")
    return decision


def _structural_alerts(admet: dict[str, Any]) -> dict[str, Any]:
    alerts = admet.get("structural_alerts")
    if not isinstance(alerts, dict):
        raise SystemExit("ADMET report missing object field 'structural_alerts'")
    return alerts


def _numeric_value(value: object, label: str, *, fraction: bool = False) -> float:
    if _is_bool_like(value):
        raise SystemExit(f"{label} must be numeric")
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{label} must be numeric") from exc
    if not math.isfinite(parsed):
        raise SystemExit(f"{label} must be finite")
    if fraction and (parsed < 0.0 or parsed > 1.0):
        raise SystemExit(f"{label} outside [0, 1]")
    return parsed


def _validated_admet_predictions(
    predictions: object,
    *,
    label: str,
    allow_degraded: bool,
) -> dict[str, float]:
    if not isinstance(predictions, dict):
        if allow_degraded and predictions is None:
            return {}
        raise SystemExit(f"{label} missing ADMET-AI predictions object")
    missing = sorted(ADMET_CORE_ENDPOINTS - set(predictions))
    if missing and not allow_degraded:
        raise SystemExit(f"{label} missing ADMET-AI endpoints {missing}")
    if not predictions and not allow_degraded:
        raise SystemExit(f"{label} has empty ADMET-AI predictions")
    values: dict[str, float] = {}
    for endpoint in sorted(ADMET_CORE_ENDPOINTS & set(predictions)):
        values[endpoint] = _numeric_value(
            predictions[endpoint],
            f"{label} admet_ai_predictions.{endpoint}",
            fraction=endpoint in ADMET_PROBABILITY_ENDPOINTS,
        )
    return values


def _validate_admet_ai_source(
    admet_source_path: Path,
    report_predictions: object,
    *,
    compound_smiles: str,
    allow_degraded: bool,
) -> None:
    source = _read_json_required(admet_source_path, "ADMET-AI source predictions")
    status = source.get("status")
    if status != "ok":
        if allow_degraded:
            return
        raise SystemExit(f"ADMET-AI source predictions status must be ok: {status!r}")
    _validate_source_smiles(source, compound_smiles, "ADMET-AI source predictions")
    source_values = _validated_admet_predictions(
        source.get("predictions"),
        label="ADMET-AI source predictions",
        allow_degraded=allow_degraded,
    )
    if not isinstance(report_predictions, dict):
        return
    report_values = _validated_admet_predictions(
        report_predictions,
        label="ADMET report",
        allow_degraded=allow_degraded,
    )
    for endpoint in sorted(set(source_values) & set(report_values)):
        if abs(source_values[endpoint] - report_values[endpoint]) > 1e-9:
            raise SystemExit(f"source/report mismatch for endpoint {endpoint}")


def _validate_structural_alert_source(
    structural_alerts_path: Path,
    report_alerts: object,
    *,
    compound_smiles: str,
) -> None:
    if not isinstance(report_alerts, dict):
        raise SystemExit("ADMET report missing object field 'structural_alerts'")
    if not isinstance(report_alerts.get("matches"), dict):
        raise SystemExit("ADMET report structural_alerts missing matches object")
    source = _read_json_required(
        structural_alerts_path,
        "structural-alert source evidence",
    )
    _validate_source_smiles(source, compound_smiles, "structural-alert source evidence")
    if not isinstance(source.get("matches"), dict):
        raise SystemExit("structural-alert source evidence missing matches object")
    for key in ("any_pains", "any_brenk", "any_nih"):
        report_value = report_alerts.get(key)
        if not isinstance(report_value, bool):
            raise SystemExit(f"ADMET report structural_alerts.{key} must be boolean")
        source_value = source.get(key)
        if not isinstance(source_value, bool):
            raise SystemExit(f"structural-alert source evidence missing boolean {key}")
        if report_value != source_value:
            raise SystemExit(f"source/report mismatch for structural-alert flag {key}")
    if source.get("matches") != report_alerts.get("matches"):
        raise SystemExit("source/report mismatch for structural-alert matches")


def _skin_sens_source_call(payload: dict[str, Any], model: str) -> tuple[object, object]:
    if model == "pred_skin":
        pred_skin = payload.get("pred_skin")
        probability = pred_skin.get("probability") if isinstance(pred_skin, dict) else None
        return payload.get("consensus_call"), probability
    return payload.get("skin_sens_call"), payload.get("skin_sens_probability")


def _validate_skin_sens_report_calls(
    skin_sens: dict[str, Any],
    *,
    allow_degraded: bool,
) -> None:
    if bool(skin_sens.get("degraded")) and not allow_degraded:
        raise SystemExit("ADMET report has degraded skin-sens evidence")
    missing_models = skin_sens.get("missing_models")
    if not isinstance(missing_models, list):
        raise SystemExit("ADMET report skin_sens.missing_models must be a list")
    missing = {str(model).strip() for model in missing_models if str(model).strip()}
    if missing and not allow_degraded:
        raise SystemExit("ADMET report has missing skin-sens model evidence")
    for model in SKIN_SENS_MODELS:
        call = skin_sens.get(model)
        if isinstance(call, str) and call.strip() in SKIN_SENS_CALLS:
            continue
        if allow_degraded and model in missing:
            continue
        raise SystemExit(f"ADMET report skin_sens.{model} must be positive or negative")


def _validate_skin_sens_sources(
    *,
    source_paths: dict[str, Path],
    report_skin_sens: dict[str, Any],
    compound_smiles: str,
    allow_degraded: bool,
) -> None:
    _validate_skin_sens_report_calls(report_skin_sens, allow_degraded=allow_degraded)
    for model in SKIN_SENS_MODELS:
        source = _read_json_required(
            source_paths[model],
            f"{model} skin-sens source evidence",
        )
        status = source.get("status")
        if status != "ok":
            if allow_degraded:
                continue
            raise SystemExit(f"{model} skin-sens source evidence status must be ok: {status!r}")
        _validate_source_smiles(
            source,
            compound_smiles,
            f"{model} skin-sens source evidence",
        )
        call, probability = _skin_sens_source_call(source, model)
        if not isinstance(call, str) or call.strip() not in SKIN_SENS_CALLS:
            raise SystemExit(f"{model} call must be positive or negative")
        _numeric_value(
            probability,
            f"{model} skin-sens probability",
            fraction=True,
        )
        report_call = report_skin_sens.get(model)
        if isinstance(report_call, str) and report_call.strip() != call.strip():
            raise SystemExit(f"source/report mismatch for {model} call")


def _validate_safety_sources(
    args: argparse.Namespace,
    admet: dict[str, Any],
    *,
    compound_smiles: str,
) -> None:
    skin_sens = admet.get("skin_sens")
    if not isinstance(skin_sens, dict):
        raise SystemExit("ADMET report missing object field 'skin_sens'")
    allow_degraded = bool(args.allow_degraded_safety)
    _validate_admet_ai_source(
        Path(args.admet_ai_json),
        admet.get("admet_ai_predictions"),
        compound_smiles=compound_smiles,
        allow_degraded=allow_degraded,
    )
    _validate_structural_alert_source(
        Path(args.structural_alerts_json),
        admet.get("structural_alerts"),
        compound_smiles=compound_smiles,
    )
    _validate_skin_sens_sources(
        source_paths={
            "husspred": Path(args.husspred_json),
            "stoptox": Path(args.stoptox_json),
            "pred_skin": Path(args.pred_skin_json),
        },
        report_skin_sens=skin_sens,
        compound_smiles=compound_smiles,
        allow_degraded=allow_degraded,
    )


def _skin_sens_evidence(args: argparse.Namespace) -> list[dict[str, str]]:
    source_paths = {
        "husspred": Path(args.husspred_json),
        "stoptox": Path(args.stoptox_json),
        "pred_skin": Path(args.pred_skin_json),
    }
    rows: list[dict[str, str]] = []
    for model in SKIN_SENS_MODELS:
        source = _read_json_required(
            source_paths[model],
            f"{model} skin-sens source evidence",
        )
        status = _text_or_none(source.get("status")) or "n/a"
        call, probability = _skin_sens_source_call(source, model)
        if status == "ok":
            if not isinstance(call, str) or call.strip() not in SKIN_SENS_CALLS:
                raise SystemExit(f"{model} call must be positive or negative")
            parsed_probability = _numeric_value(
                probability,
                f"{model} skin-sens probability",
                fraction=True,
            )
            call_text = call.strip()
            probability_text = f"{parsed_probability:.3f}"
        else:
            call_text = _text_or_none(call) or "n/a"
            probability_text = "n/a"
        rows.append({
            "model": model,
            "status": status,
            "call": call_text,
            "probability": probability_text,
        })
    return rows


def _admet_probability_metric(admet: dict[str, Any], endpoint: str) -> float:
    predictions = admet.get("admet_ai_predictions")
    if not isinstance(predictions, dict):
        raise SystemExit("ADMET report missing object field 'admet_ai_predictions'")
    if endpoint not in predictions:
        raise SystemExit(f"ADMET report missing required ADMET-AI endpoint '{endpoint}'")
    return _numeric_value(
        predictions[endpoint],
        f"ADMET report endpoint '{endpoint}'",
        fraction=True,
    )


def _summary_admet_metrics(admet: dict[str, Any]) -> dict[str, float]:
    predictions = admet.get("admet_ai_predictions")
    if not isinstance(predictions, dict):
        return {}
    metrics: dict[str, float] = {}
    for endpoint in SUMMARY_ADMET_METRICS:
        if endpoint not in predictions:
            continue
        metrics[endpoint] = _numeric_value(
            predictions[endpoint],
            f"ADMET report endpoint '{endpoint}'",
            fraction=endpoint in ADMET_PROBABILITY_ENDPOINTS,
        )
    return metrics


def _admet_risk_level(value: float) -> str:
    if value >= ADMET_HIGH_RISK_THRESHOLD:
        return "high"
    if value >= ADMET_MODERATE_RISK_THRESHOLD:
        return "moderate"
    return "low"


def _raise_skin_toxicity_decision(current: str, candidate: str) -> str:
    if SKIN_TOXICITY_DECISION_ORDER[candidate] > SKIN_TOXICITY_DECISION_ORDER[current]:
        return candidate
    return current


def _raise_skin_toxicity_level(current: str, candidate: str) -> str:
    if SKIN_TOXICITY_LEVEL_ORDER[candidate] > SKIN_TOXICITY_LEVEL_ORDER[current]:
        return candidate
    return current


def _raise_overall_decision(current: str, candidate: str) -> str:
    if OVERALL_DECISION_ORDER[candidate] > OVERALL_DECISION_ORDER[current]:
        return candidate
    return current


def _skin_toxicity_summary(
    *,
    admet: dict[str, Any],
    skin_sens_decision: str,
    structural_alerts: dict[str, Any],
) -> dict[str, Any]:
    decision = "PASS"
    toxicity_level = "low"
    reasons: list[str] = []

    def review(reason: str, level: str = "moderate") -> None:
        nonlocal decision, toxicity_level
        decision = _raise_skin_toxicity_decision(decision, "REVIEW")
        toxicity_level = _raise_skin_toxicity_level(toxicity_level, level)
        reasons.append(reason)

    if skin_sens_decision == "HALT":
        decision = _raise_skin_toxicity_decision(decision, "HALT")
        toxicity_level = _raise_skin_toxicity_level(toxicity_level, "high")
        reasons.append("skin sensitization consensus is HALT")
    elif skin_sens_decision == "FLAG_HIGH":
        review("skin sensitization consensus is FLAG_HIGH", "high")

    skin_reaction_value = _admet_probability_metric(admet, "Skin_Reaction")
    skin_reaction_risk = _admet_risk_level(skin_reaction_value)
    if skin_reaction_risk == "high":
        review("Skin_Reaction ADMET risk is high", "high")
    elif skin_reaction_risk == "moderate":
        review("Skin_Reaction ADMET risk is moderate", "moderate")

    flagged_alerts = sorted(
        key
        for key in ("any_pains", "any_brenk", "any_nih")
        if bool(structural_alerts.get(key))
    )
    if flagged_alerts:
        review("structural alerts present: " + ",".join(flagged_alerts))

    skin_sens = admet.get("skin_sens") if isinstance(admet.get("skin_sens"), dict) else {}
    degraded = bool(skin_sens.get("degraded"))
    if degraded:
        review("skin-sens evidence is degraded")
    missing_models = skin_sens.get("missing_models")
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
        "skin_reaction_value": skin_reaction_value,
        "skin_reaction_risk_level": skin_reaction_risk,
        "structural_alert_flags": flagged_alerts,
        "degraded": degraded,
        "missing_models": missing_model_names,
        "reasons": reasons,
    }


def _admet_risk_assessment(admet: dict[str, Any]) -> dict[str, Any]:
    predictions = admet.get("admet_ai_predictions")
    if not isinstance(predictions, dict):
        return {
            "high_risk_endpoints": [],
            "moderate_risk_endpoints": [],
        }
    endpoints: dict[str, dict[str, Any]] = {}
    for endpoint in ADMET_RISK_ENDPOINTS:
        if endpoint not in predictions:
            continue
        value = _admet_probability_metric(admet, endpoint)
        risk_level = _admet_risk_level(value)
        endpoints[endpoint] = {
            "value": value,
            "risk_level": risk_level,
        }
    return {
        "endpoints": endpoints,
        "high_risk_endpoints": [
            endpoint for endpoint, item in endpoints.items()
            if item["risk_level"] == "high"
        ],
        "moderate_risk_endpoints": [
            endpoint for endpoint, item in endpoints.items()
            if item["risk_level"] == "moderate"
        ],
    }


def _text_or_none(value: object) -> str | None:
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
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
        return {}
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
        raise SystemExit(f"target metadata missing UniProt identifier column: {metadata_path}")
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


def _enrich_target_metadata(
    df: pd.DataFrame,
    target_metadata: dict[str, dict[str, str]],
) -> pd.DataFrame:
    if not target_metadata:
        return df
    out = df.copy()
    for field in ("gene_symbol", "protein_name"):
        if field not in out.columns:
            out[field] = None
    for idx, row in out.iterrows():
        target_id = _text_or_none(row.get("target_id"))
        if target_id is None:
            continue
        metadata = target_metadata.get(target_id, {})
        for field in ("gene_symbol", "protein_name"):
            current = _text_or_none(row.get(field))
            if current is None and metadata.get(field):
                out.at[idx, field] = metadata[field]
    return out


def _row_float(row: pd.Series, key: str, *, required: bool = False) -> float:
    value = row.get(key)
    if pd.isna(value) or str(value).strip() == "":
        if required:
            raise SystemExit(f"Stage 3 ranked targets missing numeric value '{key}'")
        return 0.0
    if _is_bool_like(value):
        raise SystemExit(f"Stage 3 ranked targets value '{key}' must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            f"Stage 3 ranked targets value '{key}' must be numeric"
        ) from exc
    if not math.isfinite(parsed):
        raise SystemExit(f"Stage 3 ranked targets value '{key}' must be finite")
    return parsed


def _row_float_optional(row: pd.Series, key: str) -> float | None:
    if key not in row.index or _text_or_none(row.get(key)) is None:
        return None
    return _row_float(row, key, required=True)


def _row_float_fallback(row: pd.Series, keys: tuple[str, ...]) -> float:
    for key in keys:
        value = _row_float_optional(row, key)
        if value is not None:
            return value
    if keys:
        return _row_float(row, keys[0], required=True)
    return 0.0


def _row_float_optional_fallback(row: pd.Series, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = _row_float_optional(row, key)
        if value is not None:
            return value
    return None


def _row_int(row: pd.Series, key: str) -> int | None:
    try:
        numeric = float(row.get(key))
    except (TypeError, ValueError):
        return None
    if not numeric.is_integer():
        return None
    return int(numeric)


def _row_source_labels_for_summary(row: pd.Series) -> list[str]:
    value = row.get("sources")
    if pd.isna(value):
        return []
    return [
        part.strip()
        for part in str(value).replace(",", ";").replace("|", ";").split(";")
        if part.strip()
    ]


def _efficacy_labels(row: pd.Series) -> list[str]:
    labels: list[str] = []
    for col in sorted(c for c in row.index if c.startswith("efficacy_top")):
        value = _text_or_none(row.get(col))
        if value is not None:
            labels.append(value)
    return labels


def _row_has_skin_expression_support(row: pd.Series) -> bool:
    tier = _text_or_none(row.get("skin_tier"))
    return (
        (tier in SKIN_EXPRESSION_SUPPORTED_TIERS)
        or _row_float(row, "skin_score", required=True) >= SKIN_EXPRESSION_SUPPORTED_THRESHOLD
    )


def _skin_context_summary(stage3_top: pd.DataFrame) -> dict[str, Any]:
    ranking_top = stage3_top.head(10)
    top_row = ranking_top.iloc[0]
    best_row = max(
        (row for _, row in ranking_top.iterrows()),
        key=lambda row: (
            _row_float(row, "skin_score"),
            bool(_efficacy_labels(row)),
            _row_float_fallback(row, ("final_score", "score")),
        ),
    )
    expression_supported = _row_has_skin_expression_support(best_row)
    top_target_expression_supported = _row_has_skin_expression_support(top_row)
    top_target_efficacy_supported = bool(_efficacy_labels(top_row))
    top_targets_with_efficacy = sum(
        1 for _, row in ranking_top.iterrows() if _efficacy_labels(row)
    )
    efficacy_supported = top_targets_with_efficacy > 0
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
    return {
        "skin_context_decision": decision,
        "skin_context_supported": decision == "skin_context_supported",
        "skin_expression_supported": expression_supported,
        "skin_efficacy_supported": efficacy_supported,
        "skin_context_reasons": reasons,
        "top_target_id": str(top_row["target_id"]).strip(),
        "top_target_gene_symbol": _text_or_none(top_row.get("gene_symbol")),
        "top_target_protein_name": _text_or_none(top_row.get("protein_name")),
        "top_target_final_score": _row_float(top_row, "final_score", required=True),
        "top_target_docking_rrf": _row_float_optional(top_row, "docking_rrf"),
        "top_target_source_count": _row_int(top_row, "source_count"),
        "top_target_sources": _row_source_labels_for_summary(top_row),
        "top_target_skin_score": _row_float(top_row, "skin_score", required=True),
        "top_target_skin_tier": _text_or_none(top_row.get("skin_tier")),
        "top_target_cell_type_preferred": (
            _text_or_none(top_row.get("cell_type_preferred")) or "unknown"
        ),
        "top_target_efficacy": _efficacy_labels(top_row),
        "top_target_skin_expression_supported": top_target_expression_supported,
        "top_target_skin_efficacy_supported": top_target_efficacy_supported,
        "top_target_skin_context_supported": (
            top_target_expression_supported and top_target_efficacy_supported
        ),
        "top_targets_with_skin_efficacy": top_targets_with_efficacy,
        "most_skin_relevant_target_id": str(best_row["target_id"]).strip(),
        "most_skin_relevant_gene_symbol": _text_or_none(best_row.get("gene_symbol")),
        "most_skin_relevant_protein_name": _text_or_none(best_row.get("protein_name")),
        "most_skin_relevant_final_score": _row_float(best_row, "final_score", required=True),
        "most_skin_relevant_docking_rrf": _row_float_optional(best_row, "docking_rrf"),
        "most_skin_relevant_source_count": _row_int(best_row, "source_count"),
        "most_skin_relevant_sources": _row_source_labels_for_summary(best_row),
        "most_skin_relevant_skin_score": _row_float(best_row, "skin_score", required=True),
        "most_skin_relevant_skin_tier": _text_or_none(best_row.get("skin_tier")),
        "most_skin_relevant_cell_type_preferred": (
            _text_or_none(best_row.get("cell_type_preferred")) or "unknown"
        ),
        "most_skin_relevant_efficacy": _efficacy_labels(best_row),
    }


def _overall_decision_summary(
    *,
    skin_sens_decision: str,
    skin_toxicity: dict[str, Any],
    admet_risk: dict[str, Any],
    cosmetic_decision: str,
    drug_warnings: dict[str, Any] | None,
    skin_context: dict[str, Any],
) -> dict[str, Any]:
    decision = "PASS"
    reasons: list[str] = []

    if skin_sens_decision == "HALT":
        decision = _raise_overall_decision(decision, "HALT")
        reasons.append("skin sensitization consensus is HALT")
    elif skin_sens_decision == "FLAG_HIGH":
        decision = _raise_overall_decision(decision, "FLAG_HIGH")
        reasons.append("skin sensitization consensus is FLAG_HIGH")

    skin_toxicity_decision = skin_toxicity.get("decision")
    if skin_toxicity_decision == "HALT":
        decision = _raise_overall_decision(decision, "HALT")
        reasons.append("skin toxicity is HALT")
    elif skin_toxicity_decision == "REVIEW":
        decision = _raise_overall_decision(decision, "FLAG_HIGH")
        reasons.append("skin toxicity requires review")

    high_risk = admet_risk.get("high_risk_endpoints")
    if isinstance(high_risk, list) and high_risk:
        decision = _raise_overall_decision(decision, "FLAG_HIGH")
        reasons.append("high ADMET risk endpoints: " + ",".join(str(item) for item in high_risk))

    if cosmetic_decision == "HALT":
        decision = _raise_overall_decision(decision, "HALT")
        reasons.append("cosmetic/drug gate is HALT")
    elif cosmetic_decision == "DOWNWEIGHT":
        decision = _raise_overall_decision(decision, "FLAG_HIGH")
        reasons.append("cosmetic/drug gate is DOWNWEIGHT")
    elif cosmetic_decision != "PROCEED":
        decision = _raise_overall_decision(decision, "FLAG_HIGH")
        reasons.append(f"cosmetic/drug decision is unrecognized: {cosmetic_decision!r}")

    warnings = drug_warnings.get("n_warnings") if isinstance(drug_warnings, dict) else None
    if isinstance(warnings, int) and warnings > 0:
        decision = _raise_overall_decision(decision, "FLAG_HIGH")
        reasons.append(f"drug-avoidance warnings present: {warnings}")

    if skin_context.get("skin_context_supported") is not True:
        decision = _raise_overall_decision(decision, "FLAG_HIGH")
        reasons.append(
            "skin-specialized binding context requires review: "
            + str(skin_context.get("skin_context_decision") or "unknown")
        )
    if skin_context.get("top_target_skin_context_supported") is not True:
        decision = _raise_overall_decision(decision, "FLAG_HIGH")
        top_context_target = str(skin_context.get("top_target_id") or "unknown")
        reasons.append(
            "top binding target lacks direct skin context support: "
            + top_context_target
        )

    if decision == "PASS":
        reasons.insert(0, "safety, cosmetic/drug, and skin context gates passed")

    requires_human_review = decision != "PASS"
    recommended_action = {
        "HALT": "stop_before_claim",
        "FLAG_HIGH": "review_before_claim",
        "PASS": "proceed",
    }[decision]
    claimable = (
        decision == "PASS"
        and recommended_action == "proceed"
        and requires_human_review is False
    )
    return {
        "decision": decision,
        "requires_human_review": requires_human_review,
        "recommended_action": recommended_action,
        "claimable": claimable,
        "reasons": reasons,
    }


def _cosmetic_drug_decision(path: Path) -> tuple[str, str]:
    if not _nonempty(path):
        raise SystemExit(
            f"Cosmetic/drug decision is required and must be non-empty: {path}"
        )
    try:
        return parse_decision_and_policy(
            path.read_text(),
            label="Cosmetic/drug decision",
        )
    except CosmeticDrugContractError as exc:
        raise SystemExit(str(exc)) from exc


def _report_decision_html(
    *,
    overall: dict[str, Any],
    skin_toxicity: dict[str, Any],
    admet_risk: dict[str, Any],
    cosmetic_decision: str,
    drug_policy: str,
    drug_warnings: dict[str, Any] | None,
    skin_context: dict[str, Any],
) -> str:
    overall_cls = {"HALT": "halt", "FLAG_HIGH": "flag", "PASS": "pass"}.get(
        str(overall.get("decision")),
        "flag",
    )
    action = overall.get("recommended_action")
    if overall.get("claimable") is True:
        claim_status = "proceed"
    elif action == "review_before_claim":
        claim_status = "human review required before claim"
    elif action == "stop_before_claim":
        claim_status = "not claimable; stop before claim"
    elif action == "proceed":
        claim_status = "not claimable; summary claimable is not true"
    else:
        claim_status = "not claimable"
    reasons = overall.get("reasons")
    reason_items = ""
    if isinstance(reasons, list):
        reason_items = "".join(f"<li>{escape(str(reason))}</li>" for reason in reasons)
    skin_reasons = skin_context.get("skin_context_reasons")
    skin_reason_items = ""
    if isinstance(skin_reasons, list):
        skin_reason_items = "".join(
            f"<li>{escape(str(reason))}</li>" for reason in skin_reasons
        )
    efficacy = skin_context.get("top_target_efficacy")
    efficacy_text = (
        ", ".join(str(item) for item in efficacy)
        if isinstance(efficacy, list) and efficacy
        else "none"
    )
    top_target_sources = skin_context.get("top_target_sources")
    top_target_sources_text = (
        ", ".join(str(item) for item in top_target_sources)
        if isinstance(top_target_sources, list) and top_target_sources
        else "none"
    )
    most_skin_efficacy = skin_context.get("most_skin_relevant_efficacy")
    most_skin_efficacy_text = (
        ", ".join(str(item) for item in most_skin_efficacy)
        if isinstance(most_skin_efficacy, list) and most_skin_efficacy
        else "none"
    )
    most_skin_sources = skin_context.get("most_skin_relevant_sources")
    most_skin_sources_text = (
        ", ".join(str(item) for item in most_skin_sources)
        if isinstance(most_skin_sources, list) and most_skin_sources
        else "none"
    )
    most_skin_gene = skin_context.get("most_skin_relevant_gene_symbol")
    most_skin_protein = skin_context.get("most_skin_relevant_protein_name")
    most_skin_annotation = ""
    if isinstance(most_skin_gene, str) and most_skin_gene.strip():
        most_skin_annotation += (
            f" · most skin-relevant gene: <code>{escape(most_skin_gene)}</code>"
        )
    if isinstance(most_skin_protein, str) and most_skin_protein.strip():
        most_skin_annotation += (
            f" · most skin-relevant protein: <code>{escape(most_skin_protein)}</code>"
        )
    most_skin_docking_rrf = skin_context.get("most_skin_relevant_docking_rrf")
    most_skin_docking_rrf_text = (
        f" · most skin-relevant docking RRF: {float(most_skin_docking_rrf):.3f}"
        if most_skin_docking_rrf is not None
        else ""
    )
    docking_rrf = skin_context.get("top_target_docking_rrf")
    docking_rrf_text = (
        f" · top target docking RRF: {float(docking_rrf):.3f}"
        if docking_rrf is not None
        else ""
    )
    high_admet = admet_risk.get("high_risk_endpoints")
    high_admet_text = (
        ", ".join(str(item) for item in high_admet)
        if isinstance(high_admet, list) and high_admet
        else "none"
    )
    moderate_admet = admet_risk.get("moderate_risk_endpoints")
    moderate_admet_text = (
        ", ".join(str(item) for item in moderate_admet)
        if isinstance(moderate_admet, list) and moderate_admet
        else "none"
    )
    structural_flags = skin_toxicity.get("structural_alert_flags")
    structural_flags_text = (
        ", ".join(str(flag) for flag in structural_flags)
        if isinstance(structural_flags, list) and structural_flags
        else "none"
    )
    missing_models = skin_toxicity.get("missing_models")
    missing_models_text = (
        ", ".join(str(model) for model in missing_models)
        if isinstance(missing_models, list) and missing_models
        else "none"
    )
    top_target_gene = skin_context.get("top_target_gene_symbol")
    top_target_protein = skin_context.get("top_target_protein_name")
    top_target_annotation = ""
    if isinstance(top_target_gene, str) and top_target_gene.strip():
        top_target_annotation += (
            f" · top target gene: <code>{escape(top_target_gene)}</code>"
        )
    if isinstance(top_target_protein, str) and top_target_protein.strip():
        top_target_annotation += (
            f" · top target protein: <code>{escape(top_target_protein)}</code>"
        )
    warning_count = (
        drug_warnings.get("n_warnings")
        if isinstance(drug_warnings, dict)
        else "n/a"
    )
    return (
        "<h2>Panel 0 — Report Decision</h2>"
        f"<p>Overall decision: <span class='badge {overall_cls}'>"
        f"{escape(str(overall['decision']))}</span>"
        f" · Recommended action: <strong>{escape(str(overall['recommended_action']))}</strong>"
        f" · Claimable: <strong>{'yes' if overall['claimable'] else 'no'}</strong>"
        f" · Claim status: {escape(claim_status)}</p>"
        f"<p>Skin toxicity: <strong>{escape(str(skin_toxicity['decision']))}</strong>"
        f" · Cosmetic/drug decision: <strong>{escape(cosmetic_decision)}</strong>"
        f" · Drug policy: <code>{escape(drug_policy)}</code></p>"
        f"<p>Skin_Reaction: <code>{float(skin_toxicity['skin_reaction_value']):.3f}"
        f" ({escape(str(skin_toxicity['skin_reaction_risk_level']))})</code>"
        f" · Structural alerts: <code>{escape(structural_flags_text)}</code>"
        f" · degraded skin-sens evidence: <code>{escape(str(skin_toxicity['degraded']))}</code>"
        f" · missing skin-sens models: <code>{escape(missing_models_text)}</code></p>"
        f"<p>High ADMET risk endpoints: <code>{escape(high_admet_text)}</code>"
        f" · Moderate ADMET risk endpoints: <code>{escape(moderate_admet_text)}</code>"
        f" · Drug-avoidance warnings: <code>{escape(str(warning_count))}</code></p>"
        f"<p>Skin context decision: <strong>{escape(str(skin_context['skin_context_decision']))}</strong>"
        f" · Skin context supported: <strong>{'yes' if skin_context['skin_context_supported'] else 'no'}</strong>"
        f" · Skin expression supported: <strong>{'yes' if skin_context['skin_expression_supported'] else 'no'}</strong>"
        f" · Skin efficacy supported: <strong>{'yes' if skin_context['skin_efficacy_supported'] else 'no'}</strong></p>"
        f"<p>Top binding target: <code>{escape(str(skin_context['top_target_id']))}</code>"
        f"{top_target_annotation}"
        f" · top target final score: {float(skin_context['top_target_final_score']):.3f}"
        f"{docking_rrf_text}"
        f" · top target source count: <code>{escape(str(skin_context['top_target_source_count']))}</code>"
        f" · top target sources: <code>{escape(top_target_sources_text)}</code>"
        f" · top target skin score: {float(skin_context['top_target_skin_score']):.3f}"
        f" · top target skin tier: <code>{escape(str(skin_context['top_target_skin_tier'] or 'n/a'))}</code>"
        f" · top target HPA cell context: <code>{escape(str(skin_context['top_target_cell_type_preferred']))}</code>"
        f" · top target skin efficacy: <code>{escape(efficacy_text)}</code>"
        f" · top target skin context supported: "
        f"<strong>{'yes' if skin_context['top_target_skin_context_supported'] else 'no'}</strong></p>"
        f"<p>Most skin-relevant top target: <code>{escape(str(skin_context['most_skin_relevant_target_id']))}</code>"
        f"{most_skin_annotation}"
        f" · most skin-relevant final score: {float(skin_context['most_skin_relevant_final_score']):.3f}"
        f"{most_skin_docking_rrf_text}"
        f" · most skin-relevant source count: <code>{escape(str(skin_context['most_skin_relevant_source_count']))}</code>"
        f" · most skin-relevant sources: <code>{escape(most_skin_sources_text)}</code>"
        f" · most skin-relevant skin score: {float(skin_context['most_skin_relevant_skin_score']):.3f}"
        f" · most skin-relevant skin tier: <code>{escape(str(skin_context['most_skin_relevant_skin_tier'] or 'n/a'))}</code>"
        f" · most skin-relevant HPA cell context: <code>{escape(str(skin_context['most_skin_relevant_cell_type_preferred']))}</code>"
        f" · most skin-relevant efficacy: <code>{escape(most_skin_efficacy_text)}</code></p>"
        "<p><i>HPA cell context is an expression label, not evidence of cell-selective drug action.</i></p>"
        f"<ul>{reason_items}{skin_reason_items}</ul>"
    )


def _read_csv_optional(
    path_value: str,
    label: str,
    *,
    required_cols: set[str] | None = None,
    nonblank_cols: set[str] | None = None,
    smiles_cols: set[str] | None = None,
) -> pd.DataFrame | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists():
        return None
    if not _nonempty(path):
        raise SystemExit(f"{label} is empty: {path}")
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise SystemExit(f"{label} is empty: {path}") from exc
    missing = sorted((required_cols or set()) - set(df.columns))
    if missing:
        raise SystemExit(f"{label} missing required columns {missing}: {path}")
    if df.empty:
        raise SystemExit(f"{label} contains no rows: {path}")
    for col in sorted(nonblank_cols or set()):
        normalized = df[col].fillna("").astype(str).str.strip()
        blank_indexes = normalized[normalized == ""].index.tolist()
        if blank_indexes:
            shown = ",".join(str(idx) for idx in blank_indexes[:10])
            suffix = "..." if len(blank_indexes) > 10 else ""
            raise SystemExit(
                f"{label} column '{col}' contains blank values at row index(es) "
                f"{shown}{suffix}: {path}"
            )
        df[col] = normalized
    if "target_id" in (nonblank_cols or set()):
        duplicate_ids = df["target_id"][df["target_id"].duplicated()].tolist()
        if duplicate_ids:
            shown = ",".join(duplicate_ids[:10])
            suffix = "..." if len(duplicate_ids) > 10 else ""
            raise SystemExit(
                f"{label} contains duplicate target_id values: "
                f"{shown}{suffix}: {path}"
            )
    for col in sorted(smiles_cols or set()):
        from rdkit import Chem

        invalid_indexes = [
            int(idx)
            for idx, value in df[col].items()
            if Chem.MolFromSmiles(str(value).strip()) is None
        ]
        if invalid_indexes:
            shown = ",".join(str(idx) for idx in invalid_indexes[:10])
            suffix = "..." if len(invalid_indexes) > 10 else ""
            raise SystemExit(
                f"{label} column '{col}' contains invalid SMILES at row "
                f"index(es) {shown}{suffix}: {path}"
            )
    return df


def _read_table_required(
    path: Path,
    label: str,
    *,
    sep: str,
    required_cols: set[str],
    nonblank_cols: set[str] | None = None,
    numeric_cols: set[str] | None = None,
    numeric_gt: dict[str, float] | None = None,
    numeric_lt: dict[str, float] | None = None,
    allowed_values: dict[str, set[str]] | None = None,
) -> pd.DataFrame:
    if not _nonempty(path):
        raise SystemExit(f"{label} is required and must be non-empty: {path}")
    try:
        df = pd.read_csv(path, sep=sep, skip_blank_lines=False)
    except pd.errors.EmptyDataError as exc:
        raise SystemExit(f"{label} is required and must be non-empty: {path}") from exc
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise SystemExit(f"{label} missing required columns {missing}: {path}")
    if df.empty:
        raise SystemExit(f"{label} contains no rows: {path}")
    for col in sorted(nonblank_cols or set()):
        normalized = df[col].fillna("").astype(str).str.strip()
        blank_indexes = normalized[normalized == ""].index.tolist()
        if blank_indexes:
            shown = ",".join(str(idx) for idx in blank_indexes[:10])
            suffix = "..." if len(blank_indexes) > 10 else ""
            raise SystemExit(
                f"{label} column '{col}' contains blank values at row index(es) "
                f"{shown}{suffix}: {path}"
            )
        df[col] = normalized
    if "target_id" in (nonblank_cols or set()):
        duplicate_ids = df["target_id"][df["target_id"].duplicated()].tolist()
        if duplicate_ids:
            shown = ",".join(duplicate_ids[:10])
            suffix = "..." if len(duplicate_ids) > 10 else ""
            raise SystemExit(
                f"{label} contains duplicate target_id values: "
                f"{shown}{suffix}: {path}"
            )
    for col in sorted(numeric_cols or set()):
        bool_like_indexes = [
            int(idx)
            for idx, value in df[col].items()
            if _is_bool_like(value)
        ]
        if bool_like_indexes:
            shown = ",".join(str(idx) for idx in bool_like_indexes[:10])
            suffix = "..." if len(bool_like_indexes) > 10 else ""
            raise SystemExit(
                f"{label} column '{col}' must be numeric at row index(es) "
                f"{shown}{suffix}: {path}"
            )
        numeric = pd.to_numeric(df[col], errors="coerce")
        invalid_indexes = numeric[numeric.isna()].index.tolist()
        if invalid_indexes:
            shown = ",".join(str(idx) for idx in invalid_indexes[:10])
            suffix = "..." if len(invalid_indexes) > 10 else ""
            raise SystemExit(
                f"{label} column '{col}' must be numeric at row index(es) "
                f"{shown}{suffix}: {path}"
            )
        nonfinite_indexes = [
            int(idx)
            for idx, value in numeric.items()
            if not math.isfinite(float(value))
        ]
        if nonfinite_indexes:
            shown = ",".join(str(idx) for idx in nonfinite_indexes[:10])
            suffix = "..." if len(nonfinite_indexes) > 10 else ""
            raise SystemExit(
                f"{label} column '{col}' must be finite at row index(es) "
                f"{shown}{suffix}: {path}"
            )
        df[col] = numeric
    for col, threshold in (numeric_gt or {}).items():
        invalid_indexes = df[col][df[col] <= threshold].index.tolist()
        if invalid_indexes:
            shown = ",".join(str(idx) for idx in invalid_indexes[:10])
            suffix = "..." if len(invalid_indexes) > 10 else ""
            raise SystemExit(
                f"{label} column '{col}' must be > {threshold:g} at row "
                f"index(es) {shown}{suffix}: {path}"
            )
    for col, threshold in (numeric_lt or {}).items():
        invalid_indexes = df[col][df[col] >= threshold].index.tolist()
        if invalid_indexes:
            shown = ",".join(str(idx) for idx in invalid_indexes[:10])
            suffix = "..." if len(invalid_indexes) > 10 else ""
            raise SystemExit(
                f"{label} column '{col}' must be < {threshold:g} at row "
                f"index(es) {shown}{suffix}: {path}"
            )
    for col, allowed in (allowed_values or {}).items():
        normalized = df[col].fillna("").astype(str).str.strip()
        invalid = sorted(set(normalized[~normalized.isin(allowed)]))
        if invalid:
            shown = ",".join(invalid[:10])
            suffix = "..." if len(invalid) > 10 else ""
            raise SystemExit(
                f"{label} column '{col}' contains invalid values: "
                f"{shown}{suffix}"
            )
        df[col] = normalized
    return df


def _target_screening_summary(
    run_dir: Path,
    *,
    mode: str,
    ranked_count: int,
) -> dict[str, Any]:
    counts: dict[str, int] = {
        "skin_weighted_ranked_targets": ranked_count,
    }
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
        df = _read_table_required(
            run_dir / rel_path,
            f"Target screening count {key}",
            sep="\t" if rel_path.endswith(".tsv") else ",",
            required_cols=set(),
        )
        counts[key] = int(len(df))
    return {
        "screened_target_count": max(counts.values()),
        "screening_counts": counts,
    }


def _source_labels(value: object, *, label: str, path: Path, row_index: int) -> list[str]:
    if pd.isna(value) or not str(value).strip():
        raise SystemExit(
            f"{label} column 'sources' contains blank values at row index "
            f"{row_index}: {path}"
        )
    normalized = str(value).replace(",", ";").replace("|", ";")
    labels = [part.strip() for part in normalized.split(";") if part.strip()]
    if not labels:
        raise SystemExit(
            f"{label} column 'sources' contains empty labels at row index "
            f"{row_index}: {path}"
        )
    duplicates = sorted({item for item in labels if labels.count(item) > 1})
    if duplicates:
        shown = ",".join(duplicates[:10])
        suffix = "..." if len(duplicates) > 10 else ""
        raise SystemExit(
            f"{label} column 'sources' contains duplicate labels at row index "
            f"{row_index}: {shown}{suffix}: {path}"
        )
    return labels


def _validate_stage3_ranking_contract(
    df: pd.DataFrame,
    *,
    label: str,
    path: Path,
    mode: str,
    allow_missing_efficacy: bool = False,
) -> None:
    min_source_count = 1 if mode == "fast" else 3
    efficacy_cols = sorted(col for col in df.columns if col.startswith("efficacy_top"))
    if not efficacy_cols:
        raise SystemExit(f"{label} missing efficacy_top* columns: {path}")
    previous_final_score: float | None = None
    for idx, row in df.iterrows():
        row_index = int(idx)
        for column in ("final_score", "skin_score"):
            value = float(row[column])
            if value < 0.0 or value > 1.0:
                raise SystemExit(
                    f"{label} column '{column}' must be between 0 and 1 at "
                    f"row index {row_index}: {path}"
                )
        source_count_value = float(row["source_count"])
        if not source_count_value.is_integer() or source_count_value < min_source_count:
            raise SystemExit(
                f"{label} column 'source_count' must be an integer >= "
                f"{min_source_count} at row index {row_index}: {path}"
            )
        source_labels = _source_labels(
            row["sources"],
            label=label,
            path=path,
            row_index=row_index,
        )
        source_count = int(source_count_value)
        if source_count != len(source_labels):
            raise SystemExit(
                f"{label} source_count={source_count} but sources lists "
                f"{len(source_labels)} label(s) at row index {row_index}: {path}"
            )
        if (
            row_index < 10
            and not _efficacy_labels(row)
            and not (mode == "fast" and allow_missing_efficacy)
        ):
            raise SystemExit(
                f"{label} top target row lacks KG skin-efficacy evidence at "
                f"row index {row_index}: {path}"
            )
        final_score = float(row["final_score"])
        if previous_final_score is not None and final_score > previous_final_score + 1e-12:
            raise SystemExit(f"{label} final_score must be sorted descending: {path}")
        previous_final_score = final_score


def _is_bool_like(value: object) -> bool:
    if isinstance(value, bool) or type(value).__name__ == "bool_":
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"true", "false"}
    return False


def _panel_table(name: str, df: pd.DataFrame, limit: int = 50) -> str:
    if df.empty:
        return f"<h2>{name}</h2><p><i>no data</i></p>"
    return f"<h2>{name}</h2>{df.head(limit).to_html(index=False)}"


def _target_screening_summary_html(summary: dict[str, Any]) -> str:
    counts = summary.get("screening_counts")
    rows = [
        {"Screening stage": key, "Target count": count}
        for key, count in counts.items()
    ] if isinstance(counts, dict) else []
    screened_count = summary.get("screened_target_count")
    table = (
        pd.DataFrame(rows).to_html(index=False)
        if rows
        else "<p><i>no screening counts</i></p>"
    )
    return (
        "<h3>Screening stage counts</h3>"
        f"<p>Screened target candidates: <code>{escape(str(screened_count))}</code></p>"
        + table
    )


def _summary_admet_metrics_html(admet: dict[str, Any]) -> str:
    rows = [
        {"ADMET Metric": metric, "Value": f"{value:.4g}"}
        for metric, value in _summary_admet_metrics(admet).items()
    ]
    if not rows:
        return "<h3>Summary ADMET metrics</h3><p><i>no ADMET metrics</i></p>"
    return (
        "<h3>Summary ADMET metrics</h3>"
        + pd.DataFrame(rows).to_html(index=False)
    )


def _skin_sens_evidence_html(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "<h3>Skin-sens 3-model evidence</h3><p><i>no skin-sens evidence</i></p>"
    return (
        "<h3>Skin-sens 3-model evidence</h3>"
        + pd.DataFrame(rows).to_html(index=False)
    )


def _skin_toxicity_html(summary: dict[str, Any]) -> str:
    cls = {"HALT": "halt", "REVIEW": "flag", "PASS": "pass"}.get(
        str(summary.get("decision")),
        "flag",
    )
    reasons = summary.get("reasons")
    reason_items = ""
    if isinstance(reasons, list):
        reason_items = "".join(f"<li>{reason}</li>" for reason in reasons)
    structural_flags = summary.get("structural_alert_flags")
    flags_text = (
        ", ".join(str(flag) for flag in structural_flags)
        if isinstance(structural_flags, list) and structural_flags
        else "none"
    )
    missing_models = summary.get("missing_models")
    missing_text = (
        ", ".join(str(model) for model in missing_models)
        if isinstance(missing_models, list) and missing_models
        else "none"
    )
    return (
        "<h3>Skin Toxicity</h3>"
        f"<p>Decision: <span class='badge {cls}'>{summary['decision']}</span>"
        f" · toxicity level: <strong>{summary['toxicity_level']}</strong>"
        f" · Skin_Reaction: {summary['skin_reaction_value']:.3f}"
        f" ({summary['skin_reaction_risk_level']})</p>"
        f"<p>Structural alerts: <code>{flags_text}</code>"
        f" · degraded skin-sens evidence: <code>{summary['degraded']}</code>"
        f" · missing skin-sens models: <code>{missing_text}</code></p>"
        f"<ul>{reason_items}</ul>"
    )


def _require_targets_within_stage3(
    df: pd.DataFrame,
    label: str,
    stage3_targets: set[str],
) -> None:
    extra = sorted(set(df["target_id"].astype(str)) - stage3_targets)
    if extra:
        shown = ",".join(extra[:10])
        suffix = "..." if len(extra) > 10 else ""
        raise SystemExit(
            f"{label} contains target_id values absent from Stage 3 "
            f"ranked targets: {shown}{suffix}"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_file(path: Path | None, label: str) -> Path:
    if path is None or not _nonempty(path):
        raise SystemExit(f"{label} is required and must be non-empty: {path}")
    return path


def _required_directory(path: Path | None, label: str) -> Path:
    if path is None or not path.is_dir() or path.is_symlink():
        raise SystemExit(f"{label} is required and must be a non-symlink directory: {path}")
    return path


def _read_json_file(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} must be a JSON object: {path}")
    return payload


def _viewer_assets(
    bundle_value: Path | None,
    stylesheet_value: Path | None,
) -> dict[str, Any]:
    bundle = bundle_value or DEFAULT_MOLSTAR_BUNDLE
    stylesheet = stylesheet_value or (
        bundle.with_name("molstar.css") if bundle_value is not None else DEFAULT_MOLSTAR_STYLESHEET
    )
    if not _nonempty(bundle):
        raise SystemExit(
            "Local Mol* browser bundle is required and must be non-empty: "
            f"{bundle}. Install or vendor it under scripts/report_assets/molstar/ "
            "or pass --molstar-bundle; CDN/runtime network loading is forbidden."
        )
    if not _nonempty(stylesheet):
        raise SystemExit(
            "Local Mol* stylesheet is required and must be non-empty: "
            f"{stylesheet}. CDN/runtime network loading is forbidden."
        )

    manifest_path = bundle.parent / "ASSET_MANIFEST.json"
    manifest: dict[str, Any] | None = None
    provenance_verified = False
    if manifest_path.is_file():
        manifest = _read_json_file(manifest_path, "Mol* asset manifest")
        files = manifest.get("files")
        if not isinstance(files, dict):
            raise SystemExit(f"Mol* asset manifest missing object field 'files': {manifest_path}")
        for path in (bundle, stylesheet):
            expected = files.get(path.name)
            if not isinstance(expected, str) or expected != _sha256_file(path):
                raise SystemExit(f"Mol* asset hash mismatch for {path.name}: {manifest_path}")
        provenance_verified = True
    elif bundle == DEFAULT_MOLSTAR_BUNDLE or stylesheet == DEFAULT_MOLSTAR_STYLESHEET:
        raise SystemExit(f"Vendored Mol* asset manifest is required: {manifest_path}")

    license_path = bundle.parent / "LICENSE"
    if license_path.is_file() and manifest is not None:
        expected_license = manifest.get("files", {}).get("LICENSE")
        if not isinstance(expected_license, str) or expected_license != _sha256_file(license_path):
            raise SystemExit(f"Mol* license hash mismatch: {license_path}")
    else:
        license_path = None
    return {
        "bundle": bundle,
        "stylesheet": stylesheet,
        "license": license_path,
        "asset_manifest": manifest_path if manifest is not None else None,
        "package": manifest.get("package") if manifest else "custom",
        "version": manifest.get("version") if manifest else "custom",
        "license_id": manifest.get("license") if manifest else "unverified",
        "provenance_verified": provenance_verified,
    }


def _required_sdf(path: Path | None, label: str) -> Path:
    checked = _required_file(path, label)
    try:
        supplier = Chem.SDMolSupplier(str(checked), removeHs=False, sanitize=False)
        molecule = next((mol for mol in supplier if mol is not None), None)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"{label} is not a parseable SDF: {checked}: {exc}") from exc
    if molecule is None or molecule.GetNumAtoms() < 1 or molecule.GetNumConformers() < 1:
        raise SystemExit(f"{label} must contain a molecule with 3D coordinates: {checked}")
    conformer = molecule.GetConformer()
    for atom_index in range(molecule.GetNumAtoms()):
        point = conformer.GetAtomPosition(atom_index)
        if not all(math.isfinite(value) for value in (point.x, point.y, point.z)):
            raise SystemExit(f"{label} contains non-finite coordinates: {checked}")
    return checked


def _required_pose_sdf(
    path: Path,
    label: str,
    *,
    target_id: str,
    expected_energy: float,
) -> Path:
    checked = _required_sdf(path, label)
    supplier = Chem.SDMolSupplier(str(checked), removeHs=False, sanitize=False)
    molecule = next((mol for mol in supplier if mol is not None), None)
    if molecule is None:
        raise SystemExit(f"{label} contains no readable molecule: {checked}")
    if not molecule.HasProp("target_id"):
        raise SystemExit(f"{label} is missing SDF property target_id: {checked}")
    actual_target_id = molecule.GetProp("target_id").strip()
    if actual_target_id != target_id:
        raise SystemExit(
            f"{label} target_id mismatch: expected {target_id}, got "
            f"{actual_target_id or '<blank>'}: {checked}"
        )
    if not molecule.HasProp("docking_energy_kcal_mol"):
        raise SystemExit(
            f"{label} is missing SDF property docking_energy_kcal_mol: {checked}"
        )
    try:
        actual_energy = float(molecule.GetProp("docking_energy_kcal_mol"))
    except ValueError as exc:
        raise SystemExit(f"{label} docking energy is not numeric: {checked}") from exc
    if not math.isfinite(actual_energy) or not math.isclose(
        actual_energy,
        expected_energy,
        rel_tol=0.0,
        abs_tol=0.001,
    ):
        raise SystemExit(
            f"{label} docking energy mismatch: expected {expected_energy:.3f}, "
            f"got {actual_energy!r}: {checked}"
        )
    return checked


def _validated_pose_manifest(
    pose_dir: Path,
    score_path: Path,
    scores: pd.DataFrame,
) -> Path:
    manifest_path = _required_file(
        pose_dir / POSE_MANIFEST_NAME,
        "Fast-report docking pose manifest",
    )
    manifest = _read_json_file(manifest_path, "Fast-report docking pose manifest")
    if manifest.get("schema_version") != POSE_MANIFEST_SCHEMA_VERSION:
        raise SystemExit(f"Docking pose manifest schema mismatch: {manifest_path}")
    if manifest.get("score_file") != score_path.name:
        raise SystemExit(f"Docking pose manifest score filename mismatch: {manifest_path}")
    if manifest.get("score_sha256") != _sha256_file(score_path):
        raise SystemExit(f"Docking pose manifest score hash mismatch: {manifest_path}")
    if manifest.get("score_bytes") != score_path.stat().st_size:
        raise SystemExit(f"Docking pose manifest score size mismatch: {manifest_path}")
    records = manifest.get("targets")
    if not isinstance(records, list) or not records:
        raise SystemExit(f"Docking pose manifest contains no target records: {manifest_path}")
    if manifest.get("target_count") != len(records):
        raise SystemExit(f"Docking pose manifest target_count mismatch: {manifest_path}")
    score_by_target = scores.set_index("target_id", drop=False)
    expected_ids = set(score_by_target.index.astype(str))
    actual_ids: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise SystemExit(f"Docking pose manifest target record is invalid: {manifest_path}")
        target_id = _safe_target_id(record.get("target_id"))
        if target_id in actual_ids:
            raise SystemExit(
                f"Docking pose manifest contains duplicate target_id {target_id}: "
                f"{manifest_path}"
            )
        actual_ids.add(target_id)
        if target_id not in expected_ids:
            raise SystemExit(
                f"Docking pose manifest target is absent from score table: {target_id}"
            )
        pose_path = pose_dir / f"{target_id}.sdf"
        if record.get("pose_file") != pose_path.name:
            raise SystemExit(
                f"Docking pose manifest filename mismatch for {target_id}: {manifest_path}"
            )
        if not pose_path.is_file() or pose_path.is_symlink():
            raise SystemExit(f"Docking pose manifest file is missing or unsafe: {pose_path}")
        if record.get("pose_sha256") != _sha256_file(pose_path):
            raise SystemExit(f"Docking pose manifest hash mismatch for {target_id}")
        if record.get("pose_bytes") != pose_path.stat().st_size:
            raise SystemExit(f"Docking pose manifest size mismatch for {target_id}")
        try:
            manifest_energy = float(record.get("docking_energy_kcal_mol"))
            score_energy = float(score_by_target.loc[target_id]["vina_score"])
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                f"Docking pose manifest energy is invalid for {target_id}"
            ) from exc
        if not math.isfinite(manifest_energy) or not math.isclose(
            manifest_energy,
            score_energy,
            rel_tol=0.0,
            abs_tol=0.001,
        ):
            raise SystemExit(
                f"Docking pose manifest energy mismatch for {target_id}: "
                f"{manifest_path}"
            )
    if actual_ids != expected_ids:
        missing = ", ".join(sorted(expected_ids - actual_ids)[:10])
        raise SystemExit(
            f"Docking pose manifest is incomplete for score targets: {missing}"
        )
    return manifest_path


def _safe_target_id(value: object) -> str:
    target_id = str(value).strip()
    if not SAFE_TARGET_ID_RE.fullmatch(target_id) or target_id in {".", ".."}:
        raise SystemExit(f"Unsafe target_id cannot be used as a report asset path: {target_id!r}")
    return target_id


def _receptor_asset_for_target(receptor_dir: Path, target_id: str) -> Path:
    candidates = (
        receptor_dir / f"{target_id}_clean.pdb",
        receptor_dir / f"{target_id}.pdb",
        receptor_dir / f"{target_id}.cif",
        receptor_dir / f"{target_id}.mmcif",
    )
    for candidate in candidates:
        if _nonempty(candidate):
            return candidate
    raise SystemExit(
        "Required receptor asset for top fast target is missing or empty: "
        f"{target_id} in {receptor_dir}"
    )


def _canonical_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _copy_report_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def _receptor_format(path: Path) -> str:
    return "mmcif" if path.suffix.lower() in {".cif", ".mmcif"} else "pdb"


def _fast_report_index_html(
    *,
    run_id: str,
    top_target_id: str,
) -> str:
    return (
        FAST_REPORT_HTML_HEAD.format(title=f"SkinScout fast report - {run_id}")
        + "<header><div><h1>SkinScout 3D report</h1>"
        + f"<p class='meta'>Run <code>{escape(run_id)}</code>; top target "
        + f"<code>{escape(top_target_id)}</code></p>"
        + "<p id='viewer-status' role='status'>Preparing local 3D viewer...</p></div>"
        + "<label class='control' for='target-select'>Target"
        + "<select id='target-select'></select></label>"
        + "<div class='control'><span>Ligand view</span><div class='segmented' role='group' "
        + "aria-label='Ligand view'>"
        + "<button id='show-pose' type='button' aria-pressed='true'>Docked pose</button>"
        + "<button id='show-ligand' type='button' aria-pressed='false'>Input ligand</button>"
        + "</div></div>"
        + "<div class='downloads'><a id='download-receptor' class='download' download>"
        + "Download receptor</a><a id='download-pose' class='download' download>"
        + "Download pose SDF</a><span id='interaction-status'></span></div></header>"
        + "<main><div id='molstar-viewer' data-viewer='molstar-local' "
        + "data-viewer-loaded='false' data-canvas-nonblank='false' "
        + "aria-label='Interactive molecular structure'></div>"
        + "<p id='viewer-fallback' role='alert' hidden></p></main>"
        + "</body></html>\n"
    )


def _write_json_deterministic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, separators=(",", ": ")) + "\n",
        encoding="utf-8",
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    _write_json_deterministic(temporary, payload)
    temporary.replace(path)


def _source_record(path: Path) -> dict[str, Any]:
    return {
        "source_name": path.name,
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _payload_files(root: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in {"manifest.json", "checksums.json"}:
            continue
        if path.is_symlink():
            raise SystemExit(f"Report package payload cannot contain symlinks: {path}")
        files.append({
            "path": relative,
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        })
    return files


def _artifact_id(checksums: dict[str, Any]) -> str:
    return _canonical_hash(checksums)


def _verify_package_payload(package_root: Path, checksums: dict[str, Any]) -> None:
    entries = checksums.get("files")
    if not isinstance(entries, list) or not entries:
        raise SystemExit(f"Report checksums contain no files: {package_root}")
    expected_paths: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise SystemExit(f"Report checksum entry must be an object: {package_root}")
        relative = entry.get("path")
        if not isinstance(relative, str) or not relative or relative.startswith("/"):
            raise SystemExit(f"Invalid report checksum path: {relative!r}")
        unresolved = package_root / relative
        if unresolved.is_symlink():
            raise SystemExit(f"Report payload cannot be a symlink: {relative}")
        candidate = unresolved.resolve()
        try:
            candidate.relative_to(package_root.resolve())
        except ValueError as exc:
            raise SystemExit(f"Report checksum path escapes package: {relative}") from exc
        if not candidate.is_file():
            raise SystemExit(f"Report payload is missing or unsafe: {relative}")
        expected_paths.add(relative)
        if entry.get("sha256") != _sha256_file(candidate):
            raise SystemExit(f"Report payload hash mismatch: {relative}")
        if entry.get("bytes") != candidate.stat().st_size:
            raise SystemExit(f"Report payload size mismatch: {relative}")
    actual_paths: set[str] = set()
    for path in package_root.rglob("*"):
        if path.is_symlink():
            raise SystemExit(f"Report package cannot contain symlinks: {path}")
        if path.is_file():
            relative = path.relative_to(package_root).as_posix()
            if relative not in {"manifest.json", "checksums.json"}:
                actual_paths.add(relative)
    if actual_paths != expected_paths:
        raise SystemExit(f"Report package contains undeclared payload files: {package_root}")


def _verify_existing_package(
    package_root: Path,
    *,
    artifact_id: str,
    schema_version: str,
    require_sealed: bool,
) -> dict[str, Any]:
    if package_root.is_symlink() or not package_root.is_dir():
        raise SystemExit(f"Immutable report package path is unsafe: {package_root}")
    checksums = _read_json_file(package_root / "checksums.json", "Report checksums")
    manifest = _read_json_file(package_root / "manifest.json", "Report manifest")
    if checksums.get("schema_version") != schema_version:
        raise SystemExit(f"Report checksum schema mismatch: {package_root}")
    if _artifact_id(checksums) != artifact_id or manifest.get("artifact_id") != artifact_id:
        raise SystemExit(f"Immutable report artifact ID mismatch: {package_root}")
    _verify_package_payload(package_root, checksums)
    identity_path = package_root / "identity.json"
    identity = _read_json_file(identity_path, "Report identity")
    if identity.get("schema_version") != REPORT_IDENTITY_SCHEMA_VERSION:
        raise SystemExit(f"Report identity schema mismatch: {package_root}")
    if manifest.get("identity_path") != "identity.json":
        raise SystemExit(f"Report manifest identity path mismatch: {package_root}")
    if manifest.get("identity_sha256") != _sha256_file(identity_path):
        raise SystemExit(f"Report manifest identity hash mismatch: {package_root}")
    manifest_core = identity.get("manifest_core")
    if not isinstance(manifest_core, dict) or not manifest_core:
        raise SystemExit(f"Report identity contains no manifest core: {package_root}")
    for key, expected in manifest_core.items():
        if manifest.get(key) != expected:
            raise SystemExit(
                f"Report manifest field is not identity-bound: {key}: {package_root}"
            )
    if identity.get("kind") == "fast":
        parent_linkage = manifest.get("parent_linkage")
        if not isinstance(parent_linkage, dict) or parent_linkage.get(
            "fast_parent_hash"
        ) != artifact_id:
            raise SystemExit(f"Fast report self-parent hash mismatch: {package_root}")
    if require_sealed and manifest.get("sealed") is not True:
        raise SystemExit(f"Existing immutable report package is not browser-sealed: {package_root}")
    return manifest


def _publish_immutable_package(
    temporary_root: Path,
    package_root: Path,
    *,
    artifact_id: str,
    checksum_schema: str,
    require_sealed: bool,
) -> dict[str, Any]:
    _verify_existing_package(
        temporary_root,
        artifact_id=artifact_id,
        schema_version=checksum_schema,
        require_sealed=require_sealed,
    )
    if package_root.exists() or package_root.is_symlink():
        manifest = _verify_existing_package(
            package_root,
            artifact_id=artifact_id,
            schema_version=checksum_schema,
            require_sealed=require_sealed,
        )
        shutil.rmtree(temporary_root)
        return manifest
    try:
        os.rename(temporary_root, package_root)
    except OSError as exc:
        if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
            raise
        manifest = _verify_existing_package(
            package_root,
            artifact_id=artifact_id,
            schema_version=checksum_schema,
            require_sealed=require_sealed,
        )
        shutil.rmtree(temporary_root)
        return manifest
    return _read_json_file(package_root / "manifest.json", "Report manifest")


def _report_relative_path(path: Path, run_dir: Path) -> str | None:
    try:
        return path.resolve().relative_to(run_dir.resolve()).as_posix()
    except ValueError:
        return None


def _json_scalar(value: object) -> object:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        value = value.item()  # type: ignore[union-attr]
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _target_label(row: pd.Series, target_id: str) -> str:
    parts = [target_id]
    for key in ("gene_symbol", "protein_name"):
        value = _text_or_none(row.get(key))
        if value and value not in parts:
            parts.append(value)
    return " - ".join(parts)


def _fast_report_targets(
    args: argparse.Namespace,
    *,
    stage3_top: pd.DataFrame,
    autodock_full: pd.DataFrame,
) -> tuple[str, list[dict[str, Any]], dict[str, Path]]:
    if stage3_top.empty:
        raise SystemExit("Stage 3 ranked targets contains no top target for fast report")
    receptor_dir = _required_directory(args.receptor_dir, "Fast-report receptor directory")
    pose_dir = _required_directory(args.pose_dir, "Fast-report docked-pose directory")
    score_table = autodock_full.copy()
    score_table["target_id"] = score_table["target_id"].astype(str).str.strip()
    if score_table["target_id"].duplicated().any():
        raise SystemExit("AutoDock scores contain duplicate target_id values for fast report")
    scores = score_table.set_index("target_id", drop=False)
    pose_manifest = _validated_pose_manifest(
        pose_dir,
        Path(args.autodock_full),
        score_table,
    )

    top_target_id = _safe_target_id(stage3_top.iloc[0]["target_id"])
    if top_target_id not in scores.index:
        raise SystemExit(
            "Fast-report pose score is missing for top target: " + top_target_id
        )

    targets: list[dict[str, Any]] = []
    source_assets: dict[str, Path] = {"pose_manifest": pose_manifest}
    seen: set[str] = set()
    for _, ranked_row in stage3_top.iterrows():
        target_id = _safe_target_id(ranked_row["target_id"])
        if target_id in seen or target_id not in scores.index:
            continue
        receptor = _required_file(
            _receptor_asset_for_target(receptor_dir, target_id),
            f"Fast-report receptor asset for {target_id}",
        )
        score_row = scores.loc[target_id]
        pose = _required_pose_sdf(
            pose_dir / f"{target_id}.sdf",
            f"Fast-report docked pose SDF for {target_id}",
            target_id=target_id,
            expected_energy=float(score_row["vina_score"]),
        )
        target_root = f"molecules/{target_id}"
        receptor_name = "receptor" + receptor.suffix.lower()
        targets.append({
            "target_id": target_id,
            "label": _target_label(ranked_row, target_id),
            "receptor": {
                "path": f"{target_root}/{receptor_name}",
                "format": _receptor_format(receptor),
            },
            "pose": {"path": f"{target_root}/pose.sdf", "format": "sdf"},
            "interaction": {"path": f"interactions/{target_id}.json"},
            "docking": {
                key: _json_scalar(score_row.get(key))
                for key in ("vina_score", "neg_vina_score", "engine", "degraded")
                if key in score_row.index
            },
        })
        source_assets[f"receptor:{target_id}"] = receptor
        source_assets[f"pose:{target_id}"] = pose
        seen.add(target_id)
        if len(targets) >= FAST_REPORT_TARGET_LIMIT:
            break
    if not targets or targets[0]["target_id"] != top_target_id:
        raise SystemExit(f"Fast-report assets could not be resolved for top target: {top_target_id}")
    return top_target_id, targets, source_assets


def _build_fast_report_package(
    args: argparse.Namespace,
    *,
    stage3_top: pd.DataFrame,
    autodock_full: pd.DataFrame,
    compound: dict[str, str],
) -> dict[str, Any]:
    ligand_sdf = _required_sdf(args.ligand_sdf, "Fast-report ligand SDF asset")
    top_target_id, targets, target_sources = _fast_report_targets(
        args,
        stage3_top=stage3_top,
        autodock_full=autodock_full,
    )
    viewer = _viewer_assets(args.molstar_bundle, args.molstar_stylesheet)
    report_script = _required_file(DEFAULT_FAST_REPORT_SCRIPT, "Fast-report browser script")
    report_stylesheet = _required_file(
        DEFAULT_FAST_REPORT_STYLESHEET,
        "Fast-report stylesheet",
    )
    source_assets: dict[str, Path] = {
        "stage3_top": args.stage3_top,
        "autodock_scores": args.autodock_full,
        "ligand_sdf": ligand_sdf,
        "molstar_bundle": viewer["bundle"],
        "molstar_stylesheet": viewer["stylesheet"],
        "report_script": report_script,
        "report_stylesheet": report_stylesheet,
        **target_sources,
    }
    for optional_name in ("license", "asset_manifest"):
        optional_path = viewer[optional_name]
        if optional_path is not None:
            source_assets[f"molstar_{optional_name}"] = optional_path
    source_hashes = {
        name: _source_record(path)
        for name, path in sorted(source_assets.items())
    }
    args.fast_report_root.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tempfile.mkdtemp(prefix=".build-", dir=args.fast_report_root))

    try:
        _copy_report_file(viewer["bundle"], tmp_root / "assets" / "molstar.js")
        _copy_report_file(viewer["stylesheet"], tmp_root / "assets" / "molstar.css")
        _copy_report_file(report_script, tmp_root / "assets" / "report.js")
        _copy_report_file(report_stylesheet, tmp_root / "assets" / "report.css")
        if viewer["license"] is not None:
            _copy_report_file(viewer["license"], tmp_root / "licenses" / "molstar-LICENSE")
        _copy_report_file(ligand_sdf, tmp_root / "molecules" / "ligand.sdf")
        _copy_report_file(
            target_sources["pose_manifest"],
            tmp_root / "provenance" / POSE_MANIFEST_NAME,
        )
        for target in targets:
            target_id = target["target_id"]
            receptor = target_sources[f"receptor:{target_id}"]
            pose = target_sources[f"pose:{target_id}"]
            _copy_report_file(receptor, tmp_root / target["receptor"]["path"])
            _copy_report_file(pose, tmp_root / target["pose"]["path"])
            _write_json_deterministic(
                tmp_root / target["interaction"]["path"],
                {
                    "schema_version": "skinscout.report_interactions.v1",
                    "target_id": target_id,
                    "pose_path": target["pose"]["path"],
                    "interaction_status": "not_computed",
                    "interactions": [],
                    "docking": target["docking"],
                    "note": "Docking evidence is shown; residue interactions were not computed.",
                },
            )
        targets_payload = {
            "schema_version": "skinscout.report_targets.v1",
            "top_target_id": top_target_id,
            "original_ligand": {"path": "molecules/ligand.sdf", "format": "sdf"},
            "targets": targets,
        }
        _write_json_deterministic(tmp_root / "targets.json", targets_payload)
        stage3_top.head(50).to_csv(
            tmp_root / "ranked_targets.csv",
            index=False,
            lineterminator="\n",
        )

        index_html = _fast_report_index_html(
            run_id=args.run_id,
            top_target_id=top_target_id,
        )
        (tmp_root / "index.html").write_text(index_html, encoding="utf-8")

        if args.skip_browser_verification:
            browser_verification: dict[str, Any] = {
                "schema_version": "skinscout.report_browser_verification.v1",
                "status": "skipped_test_only",
                "viewports": {},
            }
        else:
            try:
                from verify_molstar_report import verify_report_package

                browser_verification = verify_report_package(
                    tmp_root,
                    browser_path=args.browser_path,
                )
            except Exception as exc:  # noqa: BLE001
                raise SystemExit(f"Mol* browser verification failed: {exc}") from exc
        sealed = (
            browser_verification.get("status") == "passed"
            and viewer["provenance_verified"] is True
        )
        compound_manifest = {
            "inchikey": compound["inchikey"],
            "canonical_smiles_sha256": hashlib.sha256(
                compound["canonical_smiles"].encode("utf-8")
            ).hexdigest(),
        }
        viewer_manifest = {
            "name": "Mol*",
            "source": "local",
            "package": viewer["package"],
            "version": viewer["version"],
            "license": viewer["license_id"],
            "provenance_verified": viewer["provenance_verified"],
            "bundle_path": "assets/molstar.js",
            "bundle_sha256": _sha256_file(tmp_root / "assets" / "molstar.js"),
            "stylesheet_path": "assets/molstar.css",
            "runtime_network": False,
        }
        manifest_core = {
            "schema_version": FAST_REPORT_SCHEMA_VERSION,
            "sealed": sealed,
            "run_id": args.run_id,
            "mode": args.mode,
            "top_target_id": top_target_id,
            "target_count": len(targets),
            "compound": compound_manifest,
            "viewer": viewer_manifest,
            "browser_verification": browser_verification,
            "source_hashes": source_hashes,
            "checksums_path": "checksums.json",
        }
        _write_json_deterministic(tmp_root / "identity.json", {
            "schema_version": REPORT_IDENTITY_SCHEMA_VERSION,
            "kind": "fast",
            "manifest_core": manifest_core,
        })
        checksum_schema = "skinscout.report_fast.checksums.v1"
        checksums = {
            "schema_version": checksum_schema,
            "files": _payload_files(tmp_root),
        }
        fast_hash = _artifact_id(checksums)
        package_root = args.fast_report_root / fast_hash
        _write_json_deterministic(tmp_root / "checksums.json", checksums)
        manifest = {
            **manifest_core,
            "artifact_id": fast_hash,
            "identity_path": "identity.json",
            "identity_sha256": _sha256_file(tmp_root / "identity.json"),
            "parent_linkage": {
                "fast_parent_hash": fast_hash,
                "physics_parent_hash": None,
                "physics_parent_status": "not_requested",
                "physics_parent_manifest": None,
            },
        }
        _write_json_deterministic(tmp_root / "manifest.json", manifest)
        published_manifest = _publish_immutable_package(
            tmp_root,
            package_root,
            artifact_id=fast_hash,
            checksum_schema=checksum_schema,
            require_sealed=not args.skip_browser_verification,
        )
    except BaseException:
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        raise

    result = {
        "fast_hash": fast_hash,
        "top_target_id": top_target_id,
        "package_root": package_root,
        "manifest": package_root / "manifest.json",
        "index": package_root / "index.html",
        "sealed": published_manifest.get("sealed") is True,
    }
    _write_json_atomic(
        args.fast_manifest_output,
        {
            "schema_version": REPORT_POINTER_SCHEMA_VERSION,
            "kind": "fast",
            "status": "ready" if result["sealed"] else "generated_unverified",
            "artifact_id": fast_hash,
            "sealed": result["sealed"],
            "package_path": _report_relative_path(package_root, args.run_dir),
            "manifest_path": _report_relative_path(result["manifest"], args.run_dir),
            "index_path": _report_relative_path(result["index"], args.run_dir),
        },
    )
    return result


def _write_report_pointer(
    path: Path,
    *,
    kind: str,
    status: str,
    reason: str,
) -> None:
    _write_json_atomic(path, {
        "schema_version": REPORT_POINTER_SCHEMA_VERSION,
        "kind": kind,
        "status": status,
        "reason": reason,
        "artifact_id": None,
        "sealed": False,
    })


def _build_physics_report_package(
    args: argparse.Namespace,
    *,
    fast_package: dict[str, Any],
) -> dict[str, Any]:
    physics_sources = {
        "boltz": _required_file(args.boltz_report, "Physics Boltz report"),
        "ensemble": _required_file(args.ensemble, "Physics ensemble report"),
        "mmgbsa": _required_file(args.mmgbsa, "Physics MM-GBSA report"),
        "dft": _required_file(args.dft, "Physics DFT report"),
    }
    args.physics_report_root.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tempfile.mkdtemp(prefix=".build-", dir=args.physics_report_root))
    try:
        output_names = {
            "boltz": "data/boltz.tsv",
            "ensemble": "data/ensemble.tsv",
            "mmgbsa": "data/mmgbsa.tsv",
            "dft": "data/dft.tsv",
        }
        for name, source in physics_sources.items():
            _copy_report_file(source, tmp_root / output_names[name])
        _write_json_deterministic(tmp_root / "summary.json", {
            "schema_version": "skinscout.report_physics.summary.v1",
            "run_id": args.run_id,
            "mode": args.mode,
            "parent_fast_hash": fast_package["fast_hash"],
            "inputs": output_names,
        })
        sealed = fast_package["sealed"] is True
        manifest_core = {
            "schema_version": PHYSICS_REPORT_SCHEMA_VERSION,
            "sealed": sealed,
            "run_id": args.run_id,
            "mode": args.mode,
            "parent_fast_hash": fast_package["fast_hash"],
            "source_hashes": {
                name: _source_record(path)
                for name, path in sorted(physics_sources.items())
            },
            "checksums_path": "checksums.json",
        }
        _write_json_deterministic(tmp_root / "identity.json", {
            "schema_version": REPORT_IDENTITY_SCHEMA_VERSION,
            "kind": "physics",
            "manifest_core": manifest_core,
        })
        checksum_schema = "skinscout.report_physics.checksums.v1"
        checksums = {
            "schema_version": checksum_schema,
            "files": _payload_files(tmp_root),
        }
        physics_hash = _artifact_id(checksums)
        package_root = args.physics_report_root / physics_hash
        _write_json_deterministic(tmp_root / "checksums.json", checksums)
        manifest = {
            **manifest_core,
            "artifact_id": physics_hash,
            "identity_path": "identity.json",
            "identity_sha256": _sha256_file(tmp_root / "identity.json"),
        }
        _write_json_deterministic(tmp_root / "manifest.json", manifest)
        published_manifest = _publish_immutable_package(
            tmp_root,
            package_root,
            artifact_id=physics_hash,
            checksum_schema=checksum_schema,
            require_sealed=sealed,
        )
    except BaseException:
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        raise
    result = {
        "physics_hash": physics_hash,
        "parent_fast_hash": fast_package["fast_hash"],
        "package_root": package_root,
        "manifest": package_root / "manifest.json",
        "sealed": published_manifest.get("sealed") is True,
    }
    _write_json_atomic(args.physics_manifest_output, {
        "schema_version": REPORT_POINTER_SCHEMA_VERSION,
        "kind": "physics",
        "status": "ready" if result["sealed"] else "generated_unverified",
        "artifact_id": physics_hash,
        "sealed": result["sealed"],
        "parent_fast_hash": result["parent_fast_hash"],
        "parent_manifest_path": _report_relative_path(
            fast_package["manifest"],
            args.run_dir,
        ),
        "package_path": _report_relative_path(package_root, args.run_dir),
        "manifest_path": _report_relative_path(result["manifest"], args.run_dir),
    })
    return result


def _has_physics_args(args: argparse.Namespace) -> bool:
    return all(
        path is not None
        for path in (args.boltz_report, args.ensemble, args.mmgbsa, args.dft)
    )


def _has_fast_report_args(args: argparse.Namespace) -> bool:
    return all(path is not None for path in (args.ligand_sdf, args.pose_dir, args.receptor_dir))


def _validate_fast_report_arg_contract(args: argparse.Namespace) -> None:
    values = {
        "ligand-sdf": args.ligand_sdf,
        "pose-dir": args.pose_dir,
        "receptor-dir": args.receptor_dir,
    }
    supplied = [name for name, value in values.items() if value is not None]
    if supplied and len(supplied) != len(values):
        missing = sorted(name for name, value in values.items() if value is None)
        raise SystemExit(
            "Fast report asset linkage must be all-or-none; missing: " + ", ".join(missing)
        )
    if args.mode in {"fast", "both"} and not supplied:
        raise SystemExit(
            "Fast report mode requires --ligand-sdf, --pose-dir, and --receptor-dir"
        )


def _validate_physics_arg_contract(args: argparse.Namespace) -> None:
    physics_values = {
        "boltz-report": args.boltz_report,
        "ensemble": args.ensemble,
        "mmgbsa": args.mmgbsa,
        "dft": args.dft,
    }
    supplied = [name for name, value in physics_values.items() if value is not None]
    if supplied and len(supplied) != len(physics_values):
        missing = sorted(name for name, value in physics_values.items() if value is None)
        raise SystemExit(
            "Physics report linkage must be all-or-none; missing: "
            + ", ".join(missing)
        )
    if args.mode in {"comprehensive", "both"} and not supplied:
        raise SystemExit(
            "Comprehensive legacy report mode requires physics report inputs: "
            "--boltz-report, --ensemble, --mmgbsa, --dft"
        )


def _compatibility_stub_html(
    *,
    run_id: str,
    mode: str,
    fast_package: dict[str, Any],
    physics_linkage: dict[str, Any],
) -> str:
    rel_index = Path("..") / "reports" / "fast" / str(fast_package["fast_hash"]) / "index.html"
    body = (
        HTML_HEAD.format(title=f"SkinScout report - {run_id}")
        + f"<h1>SkinScout run <code>{escape(run_id)}</code> · mode={escape(mode)}</h1>"
        + "<h2>Immutable Fast Report</h2>"
        + f"<p>Fast report hash: <code>{escape(str(fast_package['fast_hash']))}</code>"
        + f" · top target: <code>{escape(str(fast_package['top_target_id']))}</code></p>"
        + f"<p><a href='{escape(rel_index.as_posix())}'>Open local interactive molecular report</a></p>"
        + "<h2>Parent Linkage</h2>"
        + "<pre>"
        + escape(json.dumps({
            "fast_parent_hash": fast_package["fast_hash"],
            "fast_sealed": fast_package["sealed"],
            **physics_linkage,
            "fast_manifest": (
                Path("reports") / "fast" / str(fast_package["fast_hash"]) / "manifest.json"
            ).as_posix(),
            "fast_index": (
                Path("reports") / "fast" / str(fast_package["fast_hash"]) / "index.html"
            ).as_posix(),
        }, indent=2, sort_keys=True))
        + "</pre>"
        + "</body></html>\n"
    )
    return body


def render(args: argparse.Namespace) -> str:
    _validate_fast_report_arg_contract(args)
    _validate_physics_arg_contract(args)
    blocks: list[str] = [HTML_HEAD.format(title=f"COSMAX v3 — {args.run_id}")]
    blocks.append(f"<h1>COSMAX run <code>{args.run_id}</code> · mode={args.mode}</h1>")

    compound = _validated_compound_metadata(
        _read_json_required(Path(args.compound_meta), "Compound metadata")
    )
    admet = _read_json_required(Path(args.admet_report), "ADMET report")
    canonical_smiles = compound["canonical_smiles"]
    decision = _skin_sens_decision(admet, Path(args.skin_sens_decision))
    structural_alerts = _structural_alerts(admet)
    _validate_safety_sources(args, admet, compound_smiles=canonical_smiles)
    skin_sens_evidence = _skin_sens_evidence(args)
    skin_toxicity = _skin_toxicity_summary(
        admet=admet,
        skin_sens_decision=decision,
        structural_alerts=structural_alerts,
    )
    admet_risk = _admet_risk_assessment(admet)
    cosmetic_drug_decision, drug_policy = _cosmetic_drug_decision(
        Path(args.cosmetic_drug_decision)
    )
    drug_warnings = _read_json_required(
        args.drug_warnings_json,
        "Drug-avoidance warnings",
    )
    drug_warnings = _validate_drug_warnings(drug_warnings)
    try:
        validate_decision_matches_drug_warnings(
            decision=cosmetic_drug_decision,
            policy=drug_policy,
            drug_warnings=drug_warnings,
        )
    except CosmeticDrugContractError as exc:
        raise SystemExit(str(exc)) from exc
    stage3_top_path = Path(args.stage3_top)
    stage3_top = _read_table_required(
        stage3_top_path,
        "Stage 3 ranked targets",
        sep=",",
        required_cols=STAGE3_RANKING_REQUIRED_COLUMNS,
        nonblank_cols={"target_id", "sources"},
        numeric_cols={"final_score", "skin_score", "source_count"},
    )
    stage3_top = _enrich_target_metadata(
        stage3_top,
        _load_target_metadata(args.target_metadata),
    )
    _validate_stage3_ranking_contract(
        stage3_top,
        label="Stage 3 ranked targets",
        path=stage3_top_path,
        mode=args.mode,
        allow_missing_efficacy=(
            args.mode == "fast" and _uses_daina_structural_contract(Path(args.run_dir))
        ),
    )
    skin_context = _skin_context_summary(stage3_top)
    target_screening = _target_screening_summary(
        Path(args.run_dir),
        mode=args.mode,
        ranked_count=int(len(stage3_top)),
    )
    overall = _overall_decision_summary(
        skin_sens_decision=decision,
        skin_toxicity=skin_toxicity,
        admet_risk=admet_risk,
        cosmetic_decision=cosmetic_drug_decision,
        drug_warnings=drug_warnings,
        skin_context=skin_context,
    )
    stage3_targets = set(stage3_top["target_id"].astype(str))
    autodock_full = _read_table_required(
        Path(args.autodock_full),
        "AutoDock-GPU full scores",
        sep="\t",
        required_cols={"target_id", "neg_vina_score", "vina_score"},
        nonblank_cols={"target_id"},
        numeric_cols={"neg_vina_score", "vina_score"},
    )
    physics_linkage: dict[str, Any] = {
        "physics_artifact_hash": None,
        "physics_parent_hash": None,
        "physics_parent_status": "not_requested",
        "physics_parent_manifest": None,
    }
    boltz_report: pd.DataFrame | None = None
    ensemble: pd.DataFrame | None = None
    mmgbsa: pd.DataFrame | None = None
    dft: pd.DataFrame | None = None
    if _has_physics_args(args):
        assert args.boltz_report is not None
        assert args.ensemble is not None
        assert args.mmgbsa is not None
        assert args.dft is not None
        boltz_report = _read_table_required(
            args.boltz_report,
            "Boltz-2 cofold report",
            sep="\t",
            required_cols={"target_id", "kept"},
            nonblank_cols={"target_id", "kept"},
            allowed_values={"kept": {"yes", "no"}},
        )
        _require_targets_within_stage3(boltz_report, "Boltz-2 cofold report", stage3_targets)
        ensemble = _read_table_required(
            args.ensemble,
            "Ensemble dock consensus",
            sep="\t",
            required_cols={"target_id", "consensus_score"},
            nonblank_cols={"target_id"},
            numeric_cols={"consensus_score"},
            numeric_gt={"consensus_score": 0.0},
        )
        _require_targets_within_stage3(ensemble, "Ensemble dock consensus", stage3_targets)
        mmgbsa = _read_table_required(
            args.mmgbsa,
            "MM-GBSA report",
            sep="\t",
            required_cols={"target_id", "mmgbsa_dg_kcal_mol", "status"},
            nonblank_cols={"target_id", "status"},
            numeric_cols={"mmgbsa_dg_kcal_mol"},
            numeric_lt={"mmgbsa_dg_kcal_mol": 0.0},
            allowed_values={"status": {"ok"}},
        )
        _require_targets_within_stage3(mmgbsa, "MM-GBSA report", stage3_targets)
        dft = _read_table_required(
            args.dft,
            "DFT report",
            sep="\t",
            required_cols={"target_id", "dft_energy_hartree", "status"},
            nonblank_cols={"target_id", "status"},
            numeric_cols={"dft_energy_hartree"},
            numeric_lt={"dft_energy_hartree": 0.0},
            allowed_values={"status": {"ok"}},
        )
        _require_targets_within_stage3(dft, "DFT report", stage3_targets)
    fast_package: dict[str, Any] | None = None
    if _has_fast_report_args(args):
        fast_package = _build_fast_report_package(
            args,
            stage3_top=stage3_top,
            autodock_full=autodock_full,
            compound=compound,
        )
    else:
        _write_report_pointer(
            args.fast_manifest_output,
            kind="fast",
            status="not_requested",
            reason="Legacy comprehensive invocation did not supply fast-report assets.",
        )

    if _has_physics_args(args):
        if fast_package is None:
            physics_linkage = {
                "physics_artifact_hash": None,
                "physics_parent_hash": None,
                "physics_parent_status": "legacy_unlinked",
                "physics_parent_manifest": None,
            }
            _write_report_pointer(
                args.physics_manifest_output,
                kind="physics",
                status="legacy_unlinked",
                reason="Legacy comprehensive invocation has no immutable fast parent.",
            )
        else:
            physics_package = _build_physics_report_package(
                args,
                fast_package=fast_package,
            )
            physics_linkage = {
                "physics_artifact_hash": physics_package["physics_hash"],
                "physics_parent_hash": physics_package["parent_fast_hash"],
                "physics_parent_status": "linked",
                "physics_parent_manifest": (
                    Path("reports")
                    / "fast"
                    / str(physics_package["parent_fast_hash"])
                    / "manifest.json"
                ).as_posix(),
            }
    else:
        _write_report_pointer(
            args.physics_manifest_output,
            kind="physics",
            status="not_requested",
            reason="Physics extension was not requested for this report profile.",
        )
    if args.mode == "fast":
        assert fast_package is not None
        return _compatibility_stub_html(
            run_id=args.run_id,
            mode=args.mode,
            fast_package=fast_package,
            physics_linkage=physics_linkage,
        )
    overall_cls = {"HALT": "halt", "FLAG_HIGH": "flag", "PASS": "pass"}.get(
        str(overall["decision"]),
        "flag",
    )
    cls = {"HALT": "halt", "FLAG_HIGH": "flag", "PASS": "pass"}.get(decision, "flag")
    skin_tox_cls = {"HALT": "halt", "REVIEW": "flag", "PASS": "pass"}.get(
        str(skin_toxicity["decision"]),
        "flag",
    )
    blocks.append(
        f"<p>{_compound_identity_html(compound)} · "
        f"Overall decision: <span class='badge {overall_cls}'>"
        f"{escape(str(overall['decision']))}</span> · "
        f"Recommended action: <code>{escape(str(overall['recommended_action']))}</code> · "
        f"Claimable: <code>{'yes' if overall['claimable'] else 'no'}</code> · "
        f"Skin-sens decision: <span class='badge {cls}'>{escape(decision)}</span> · "
        "Skin toxicity: "
        f"<span class='badge {skin_tox_cls}'>"
        f"{escape(str(skin_toxicity['decision']))}</span></p>"
    )

    blocks.append(_report_decision_html(
        overall=overall,
        skin_toxicity=skin_toxicity,
        admet_risk=admet_risk,
        cosmetic_decision=cosmetic_drug_decision,
        drug_policy=drug_policy,
        drug_warnings=drug_warnings,
        skin_context=skin_context,
    ))
    blocks.append(_target_screening_summary_html(target_screening))
    blocks.append(_panel_table("Panel A — ranked targets", stage3_top))
    blocks.append(_panel_table("Panel A2 — AutoDock-GPU full distribution (top 50)",
                               autodock_full.head(50)))
    if fast_package is not None:
        blocks.append("<h2>Immutable Fast Report</h2>")
        fast_link = (
            Path("..")
            / "reports"
            / "fast"
            / str(fast_package["fast_hash"])
            / "index.html"
        ).as_posix()
        blocks.append(
            f"<p>Fast report hash: <code>{escape(str(fast_package['fast_hash']))}</code>"
            f" · top target: <code>{escape(str(fast_package['top_target_id']))}</code>"
            f" · sealed: <code>{str(fast_package['sealed']).lower()}</code>"
            f" · <a href='{escape(fast_link)}'>open interactive 3D report</a></p>"
        )
    if physics_linkage["physics_artifact_hash"] is not None:
        blocks.append(
            "<p>Physics artifact: "
            f"<code>{escape(str(physics_linkage['physics_artifact_hash']))}</code>"
            " · parent fast artifact: "
            f"<code>{escape(str(physics_linkage['physics_parent_hash']))}</code></p>"
        )

    diss = _read_json_optional(args.disagreement, "Disagreement analysis")
    if diss is not None:
        blocks.append("<h2>Panel A3 — DTI vs Docking disagreement</h2>")
        blocks.append(_preformatted_json(diss, limit=6000))

    blocks.append("<h2>Panel C — ADMET + Skin Toxicity</h2>")
    blocks.append(_skin_toxicity_html(skin_toxicity))
    blocks.append(_skin_sens_evidence_html(skin_sens_evidence))
    blocks.append(_summary_admet_metrics_html(admet))
    blocks.append(_preformatted_json(admet["skin_sens"]))
    blocks.append(_preformatted_json(structural_alerts, limit=4000))

    # --- v3 Panel D: INCI + drug avoidance ---
    cosing = _read_json_optional(args.cosing_json, "CosIng annotation")
    if cosing is not None:
        cosing = _validate_cosing_annotation(cosing)
        blocks.append("<h2>Panel D — INCI annotation + Drug avoidance</h2>")
        blocks.append(f"<p>CosIng level: <strong>{cosing['level']}</strong>"
                      f" · INCI = <code>{cosing['inci'] or '—'}</code>"
                      f" · functions = <code>{', '.join(cosing['functions']) or '—'}</code>"
                      f" · Tanimoto = {cosing['tanimoto']:.3f}</p>")
    if drug_warnings is not None:
        blocks.append("<p>Drug-avoidance: max Tanimoto vs approved = "
                      f"{drug_warnings['max_tanimoto_to_approved_drug']:.3f}, "
                      f"{drug_warnings['n_warnings']} warning(s).</p>")

    # --- v3 Panel E: Generated analogs ---
    analogs = _read_csv_optional(
        args.analogs_csv,
        "Generated analogs table",
        required_cols={"smiles"},
        nonblank_cols={"smiles"},
        smiles_cols={"smiles"},
    )
    if analogs is not None:
        blocks.append(_panel_table(
            "Panel E — Generated analogs (top 30)",
            analogs,
        ))

    if boltz_report is not None and ensemble is not None and mmgbsa is not None:
        blocks.append(_panel_table("Panel F — Boltz-2 cofold report", boltz_report))
        blocks.append(_panel_table("Panel F — Ensemble dock consensus", ensemble))
        blocks.append(_panel_table("Panel F — MM-GBSA",                 mmgbsa))

    # --- v3 Panel G: Synthesis tree ---
    synthesis = _read_csv_optional(
        args.synthesis_csv,
        "Retrosynthesis ranking",
        required_cols={"analog_id", "smiles"},
        nonblank_cols={"analog_id", "smiles"},
        smiles_cols={"smiles"},
    )
    if synthesis is not None:
        blocks.append(_panel_table(
            "Panel G — Retrosynthesis priority ranking",
            synthesis,
        ))

    # --- v3 Panel H: Skin-efficacy inference ---
    kg_efficacy = _read_csv_optional(
        args.kg_efficacy_csv,
        "Skin-efficacy table",
        required_cols={"target_id"},
        nonblank_cols={"target_id"},
    )
    if kg_efficacy is not None:
        _require_targets_within_stage3(
            kg_efficacy,
            "Skin-efficacy table",
            stage3_targets,
        )
        blocks.append(_panel_table(
            "Panel H — Skin-efficacy inference (top targets × KG)",
            kg_efficacy,
        ))

    if dft is not None:
        blocks.append(_panel_table("Panel I — Free-ligand DFT",         dft))

    blocks.append("</body></html>\n")
    return "\n".join(blocks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--mode",
        choices=["both", "comprehensive", "fast"],
        default="comprehensive",
    )
    parser.add_argument("--compound-meta", required=True, type=Path)
    parser.add_argument("--admet-report", required=True, type=Path)
    parser.add_argument("--skin-sens-decision", required=True, type=Path)
    parser.add_argument("--admet-ai-json", required=True, type=Path)
    parser.add_argument("--structural-alerts-json", required=True, type=Path)
    parser.add_argument("--husspred-json", required=True, type=Path)
    parser.add_argument("--stoptox-json", required=True, type=Path)
    parser.add_argument("--pred-skin-json", required=True, type=Path)
    parser.add_argument(
        "--allow-degraded-safety",
        action="store_true",
        help="Allow explicit degraded Stage 2 safety sources in diagnostic reports.",
    )
    parser.add_argument("--stage3-top", required=True, type=Path)
    parser.add_argument("--cosmetic-drug-decision", required=True, type=Path)
    parser.add_argument("--autodock-full", required=True, type=Path)
    parser.add_argument("--ligand-sdf", type=Path)
    parser.add_argument("--ligand-pdbqt", type=Path)
    parser.add_argument(
        "--pose-dir",
        type=Path,
        help="Directory containing real Stage 3 docked poses as <target_id>.sdf.",
    )
    parser.add_argument("--receptor-dir", type=Path)
    parser.add_argument(
        "--fast-report-root",
        type=Path,
        default=None,
        help="Root for immutable reports/fast/<hash> packages; defaults under run-dir.",
    )
    parser.add_argument(
        "--physics-report-root",
        type=Path,
        default=None,
        help="Root for immutable reports/physics/<hash> packages; defaults under run-dir.",
    )
    parser.add_argument("--fast-manifest-output", type=Path, default=None)
    parser.add_argument("--physics-manifest-output", type=Path, default=None)
    parser.add_argument(
        "--molstar-bundle",
        type=Path,
        default=None,
        help=(
            "Local Mol* browser bundle to copy into the report. "
            "No CDN or runtime network fallback is allowed."
        ),
    )
    parser.add_argument(
        "--molstar-stylesheet",
        type=Path,
        default=None,
        help="Local Mol* stylesheet paired with --molstar-bundle.",
    )
    parser.add_argument(
        "--browser-path",
        type=Path,
        default=None,
        help="Chrome/Chromium executable used for desktop/mobile WebGL sealing.",
    )
    parser.add_argument(
        "--skip-browser-verification",
        action="store_true",
        help=(
            "Generate an explicitly unsealed report. This is for unit tests and "
            "developer diagnostics only; production workflows must not use it."
        ),
    )
    parser.add_argument("--boltz-report", type=Path)
    parser.add_argument("--ensemble", type=Path)
    parser.add_argument("--mmgbsa", type=Path)
    parser.add_argument("--dft", type=Path)
    parser.add_argument(
        "--target-metadata",
        default=ROOT / "data" / "hpa" / "proteinatlas.tsv",
        type=Path,
        help="Optional target metadata TSV for gene/protein labels.",
    )
    parser.add_argument("--disagreement", default="")
    # v3 optional panels — pass empty string when not produced for this run
    parser.add_argument("--cosing-json", default="")
    parser.add_argument("--drug-warnings-json", required=True, type=Path)
    parser.add_argument("--analogs-csv", default="")
    parser.add_argument("--synthesis-csv", default="")
    parser.add_argument("--kg-efficacy-csv", default="")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out-html", required=True, type=Path)
    args = parser.parse_args()
    if args.fast_report_root is None:
        args.fast_report_root = args.run_dir / "reports" / "fast"
    if args.physics_report_root is None:
        args.physics_report_root = args.run_dir / "reports" / "physics"
    if args.fast_manifest_output is None:
        args.fast_manifest_output = args.out_html.parent / "fast_report_manifest.json"
    if args.physics_manifest_output is None:
        args.physics_manifest_output = args.out_html.parent / "physics_report_manifest.json"
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    tmp_html = args.out_html.with_suffix(args.out_html.suffix + ".tmp")
    output_files = (
        args.out_html,
        tmp_html,
        args.fast_manifest_output,
        args.physics_manifest_output,
    )
    for output in output_files:
        if output.exists() or output.is_symlink():
            output.unlink()
    try:
        html = render(args)
        args.out_html.parent.mkdir(parents=True, exist_ok=True)
        tmp_html.write_text(html, encoding="utf-8")
        tmp_html.replace(args.out_html)
    except BaseException:
        for output in output_files:
            if output.exists() or output.is_symlink():
                output.unlink()
        raise
    LOG.info("Wrote %s (%d bytes)", args.out_html, len(html))


if __name__ == "__main__":
    main()
