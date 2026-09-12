#!/usr/bin/env python3
"""Write compact user-facing SkinScout run summary artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem

from compound_applicability import assess as assess_applicability
from messages_ko import action_label, decision_label, reason_label
from target_failure_modes import failure_modes_for, load_curated
from cosmetic_drug_contract import (
    CosmeticDrugContractError,
    parse_decision_and_policy,
    validate_decision_matches_drug_warnings,
)


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_SCHEMA = "skinscout.run_summary.v1"
PRESETS = {"safety", "target-id", "report"}
MODES = {"comprehensive", "fast", "both"}
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
ADMET_MODERATE_RISK_THRESHOLD = 0.25
ADMET_HIGH_RISK_THRESHOLD = 0.50
SKIN_SENS_MODELS = ("husspred", "stoptox", "pred_skin")
SKIN_SENS_CALLS = {"positive", "negative"}
SKIN_TOXICITY_DECISION_ORDER = {"PASS": 0, "REVIEW": 1, "HALT": 2}
SKIN_TOXICITY_LEVEL_ORDER = {"low": 0, "moderate": 1, "high": 2}
SKIN_EXPRESSION_SUPPORTED_TIERS = {"low", "medium", "high", "very_high"}
SKIN_EXPRESSION_SUPPORTED_THRESHOLD = 0.20
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
TARGET_FRACTION_COLUMNS = {"docking_rrf", "final_score", "skin_score", "efficacy_score"}
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
LEGACY_FAST_TARGET_ARTIFACTS = {
    "target_fast_psichic": "03_targets/mode_fast/psichic_proteome.tsv",
    "target_fast_daina_zoete": "03_targets/mode_fast/daina_zoete_proteome.tsv",
    "target_fast_dti_rrf": "03_targets/mode_fast/dti_rrf_top25pct.csv",
    "target_fast_autodock": "03_targets/mode_fast/autodock_top5k.tsv",
    "target_fast_rerank_consensus": "03_targets/mode_fast/top50.csv",
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
DAINA_FAST_SCREENING_FUNNEL_EDGES = (
    ("daina_zoete_targets", "daina_primary_candidates"),
    ("daina_primary_candidates", "autodock_rescored_targets"),
    ("autodock_rescored_targets", "gnina_pose_rescored_targets"),
    ("daina_primary_candidates", "daina_structural_targets"),
    ("daina_structural_targets", "rerank_consensus_targets"),
    ("rerank_consensus_targets", "band_reranked_targets"),
)
LEGACY_FAST_SCREENING_FUNNEL_EDGES = (
    ("psichic_proteome_targets", "dti_rrf_candidates"),
    ("daina_zoete_targets", "dti_rrf_candidates"),
    ("dti_rrf_candidates", "autodock_rescored_targets"),
    ("autodock_rescored_targets", "rerank_consensus_targets"),
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


def _nonempty(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def _uses_daina_structural_contract(run_dir: Path) -> bool:
    manifest_path = run_dir / "run_manifest.json"
    if _nonempty(manifest_path):
        payload = _read_json(manifest_path, "run manifest")
        schema = payload.get("schema_version")
        if schema == "skinscout.run_manifest.v3":
            return True
        if schema == "skinscout.run_manifest.v2":
            return False
        raise SystemExit(f"unsupported run manifest schema: {schema!r}")
    return _nonempty(
        run_dir / "03_targets" / "mode_fast" / "daina_structural_targets.csv"
    )


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not _nonempty(path):
        raise SystemExit(f"{label} is required and must be non-empty: {path}")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{label} is invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} must be a JSON object: {path}")
    return payload


def _read_first_line(path: Path, label: str) -> str:
    if not _nonempty(path):
        raise SystemExit(f"{label} is required and must be non-empty: {path}")
    for line in path.read_text().splitlines():
        text = line.strip()
        if text:
            return text
    raise SystemExit(f"{label} contains no non-empty lines: {path}")


def _read_text(path: Path, label: str) -> str:
    if not _nonempty(path):
        raise SystemExit(f"{label} is required and must be non-empty: {path}")
    return path.read_text()


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    if not _nonempty(path):
        raise SystemExit(f"{label} is required and must be non-empty: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"{label} failed to parse: {path}: {exc}") from exc
    if df.empty:
        raise SystemExit(f"{label} contains no rows: {path}")
    return df


def _read_table(path: Path, label: str) -> pd.DataFrame:
    if not _nonempty(path):
        raise SystemExit(f"{label} is required and must be non-empty: {path}")
    sep = "\t" if path.suffix == ".tsv" else ","
    try:
        df = pd.read_csv(path, sep=sep)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"{label} failed to parse: {path}: {exc}") from exc
    if df.empty:
        raise SystemExit(f"{label} contains no rows: {path}")
    return df


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or type(value).__name__ == "bool_":
        raise SystemExit(f"{label} must be numeric")
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{label} must be numeric") from exc
    if not math.isfinite(parsed):
        raise SystemExit(f"{label} must be finite")
    return parsed


def _fraction(value: object, label: str) -> float:
    numeric = _number(value, label)
    if numeric < 0.0 or numeric > 1.0:
        raise SystemExit(f"{label} must be between 0 and 1")
    return numeric


def _text_or_none(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _canonical_smiles(value: object, label: str) -> str:
    smiles = _text_or_none(value)
    if smiles is None:
        raise SystemExit(f"{label} missing non-empty smiles")
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
    expected_smiles = _canonical_smiles(compound_smiles, "compound canonical_smiles")
    if source_smiles != expected_smiles:
        raise SystemExit(f"{label} smiles does not match compound canonical_smiles")


def _validated_compound_metadata(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    canonical_smiles = _text_or_none(payload.get("canonical_smiles"))
    if canonical_smiles is None:
        raise SystemExit(f"compound metadata missing non-empty canonical_smiles: {path}")
    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol is None:
        raise SystemExit(
            f"compound metadata contains invalid canonical_smiles "
            f"{canonical_smiles!r}: {path}"
        )
    expected = Chem.MolToSmiles(mol, canonical=True)
    if canonical_smiles != expected:
        raise SystemExit(
            "compound metadata canonical_smiles is not canonical: "
            f"{canonical_smiles!r} != {expected!r}: {path}"
        )
    inchikey = _text_or_none(payload.get("inchikey"))
    if inchikey is None:
        raise SystemExit(f"compound metadata missing non-empty inchikey: {path}")
    expected_inchikey = Chem.MolToInchiKey(mol)
    if inchikey != expected_inchikey:
        raise SystemExit(
            "compound metadata inchikey does not match canonical_smiles: "
            f"{inchikey!r} != {expected_inchikey!r}: {path}"
        )
    input_type = _text_or_none(payload.get("input_type"))
    if input_type is None:
        raise SystemExit(f"compound metadata missing non-empty input_type: {path}")
    if input_type not in {"smiles", "sdf"}:
        raise SystemExit(f"compound metadata invalid input_type {input_type!r}: {path}")
    result: dict[str, Any] = {
        "input_type": input_type,
        "canonical_smiles": canonical_smiles,
        "inchikey": inchikey,
        # Recorded even when the run was accepted, so a reader can see which
        # scope checks the input passed rather than assuming there were none.
        "applicability": assess_applicability(canonical_smiles),
    }
    if input_type == "smiles":
        if _text_or_none(payload.get("input_sdf")) is not None:
            raise SystemExit(
                f"compound metadata input_type 'smiles' cannot include input_sdf: {path}"
            )
        input_smiles = _text_or_none(payload.get("input_smiles"))
        if input_smiles is None:
            raise SystemExit(f"compound metadata missing non-empty input_smiles: {path}")
        input_mol = Chem.MolFromSmiles(input_smiles)
        if input_mol is None:
            raise SystemExit(
                f"compound metadata contains invalid input_smiles "
                f"{input_smiles!r}: {path}"
            )
        input_canonical = _text_or_none(payload.get("input_canonical_smiles"))
        expected_input_canonical = Chem.MolToSmiles(input_mol, canonical=True)
        if input_canonical != expected_input_canonical:
            raise SystemExit(
                "compound metadata input_canonical_smiles does not match "
                f"input_smiles: {input_canonical!r} != "
                f"{expected_input_canonical!r}: {path}"
            )
        result["input_smiles"] = input_smiles
        result["input_canonical_smiles"] = input_canonical
    if input_type == "sdf":
        for field in ("input_smiles", "input_canonical_smiles"):
            if _text_or_none(payload.get(field)) is not None:
                raise SystemExit(
                    f"compound metadata input_type 'sdf' cannot include {field}: {path}"
                )
        input_sdf = _text_or_none(payload.get("input_sdf"))
        if input_sdf is None:
            raise SystemExit(f"compound metadata missing non-empty input_sdf: {path}")
        result["input_sdf"] = input_sdf
    return result


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
        "molecular function",
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
    function_col = columns.get("molecular function")
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
        function = _text_or_none(row.get(function_col)) if function_col else None
        if gene is None and protein is None:
            continue
        for uid in _split_uniprots(row.get(uid_col)):
            out.setdefault(uid, {})
            if gene is not None and "gene_symbol" not in out[uid]:
                out[uid]["gene_symbol"] = gene
            if protein is not None and "protein_name" not in out[uid]:
                out[uid]["protein_name"] = protein
            if function is not None and "molecular_function" not in out[uid]:
                out[uid]["molecular_function"] = function
    return out


def _screening_funnel_edges(
    mode: str, *, daina_structural: bool
) -> tuple[tuple[str, str], ...]:
    edges: list[tuple[str, str]] = []
    if mode in {"fast", "both"}:
        edges.extend(
            DAINA_FAST_SCREENING_FUNNEL_EDGES
            if daina_structural
            else LEGACY_FAST_SCREENING_FUNNEL_EDGES
        )
    if mode in {"comprehensive", "both"}:
        edges.extend(COMPREHENSIVE_SCREENING_FUNNEL_EDGES)
    if mode == "fast":
        edges.append(FAST_FINAL_FUNNEL_EDGE)
    if mode in {"comprehensive", "both"}:
        edges.append(COMPREHENSIVE_FINAL_FUNNEL_EDGE)
    return tuple(edges)


def _validate_screening_funnel(
    counts: dict[str, int], *, mode: str, daina_structural: bool
) -> None:
    violations = [
        f"{upstream}={counts[upstream]} < {downstream}={counts[downstream]}"
        for upstream, downstream in _screening_funnel_edges(
            mode, daina_structural=daina_structural
        )
        if upstream in counts and downstream in counts and counts[upstream] < counts[downstream]
    ]
    if violations:
        raise SystemExit("target screening counts are non-monotonic: " + "; ".join(violations))
    final_ranked = counts.get("skin_weighted_ranked_targets")
    if (
        daina_structural
        and mode == "fast"
        and isinstance(final_ranked, int)
        and counts.get("daina_structural_targets") != final_ranked
    ):
        raise SystemExit(
            "Daina structural annotation must preserve every primary ranked target"
        )
    if (
        not daina_structural
        and isinstance(final_ranked, int)
        and max(counts.values()) <= final_ranked
    ):
        raise SystemExit(
            "target screening counts must include at least one upstream count above final ranked targets"
        )


def _validated_skin_sens(
    skin_sens: dict[str, Any],
    *,
    decision: str,
    allow_degraded: bool,
) -> tuple[dict[str, str], bool, list[str]]:
    if decision not in {"HALT", "FLAG_HIGH", "PASS"}:
        raise SystemExit(f"skin-sens decision must be HALT, FLAG_HIGH, or PASS: {decision!r}")
    degraded = bool(skin_sens.get("degraded"))
    if degraded and not allow_degraded:
        raise SystemExit("skin-sens evidence is degraded")
    raw_missing = skin_sens.get("missing_models")
    if not isinstance(raw_missing, list):
        raise SystemExit("skin-sens missing_models must be a list")
    missing_models = [str(model).strip() for model in raw_missing if str(model).strip()]
    if missing_models and not allow_degraded:
        raise SystemExit("skin-sens evidence is missing model calls: " + ",".join(missing_models))
    calls: dict[str, str] = {}
    for model in SKIN_SENS_MODELS:
        call = _text_or_none(skin_sens.get(model))
        if call not in SKIN_SENS_CALLS:
            if allow_degraded and model in missing_models:
                continue
            raise SystemExit(f"skin-sens {model} call must be positive or negative")
        calls[model] = call
    return calls, degraded, missing_models


def _validated_structural_alert_flags(admet_report: dict[str, Any]) -> dict[str, bool]:
    structural_alerts = admet_report.get("structural_alerts")
    if not isinstance(structural_alerts, dict):
        raise SystemExit("ADMET and skin-sens report missing object structural_alerts")
    flags: dict[str, bool] = {}
    for key in ("any_pains", "any_brenk", "any_nih"):
        value = structural_alerts.get(key)
        if not isinstance(value, bool):
            raise SystemExit(f"structural_alerts.{key} must be boolean")
        flags[key] = value
    return flags


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
    if not predictions:
        return {}
    values: dict[str, float] = {}
    for key in sorted(ADMET_CORE_ENDPOINTS & set(predictions)):
        endpoint_label = f"{label} admet_ai_predictions.{key}"
        values[key] = (
            _fraction(predictions[key], endpoint_label)
            if key in ADMET_PROBABILITY_ENDPOINTS
            else _number(predictions[key], endpoint_label)
        )
    return values


def _admet_metrics(admet_report: dict[str, Any], *, allow_degraded: bool) -> dict[str, float]:
    values = _validated_admet_predictions(
        admet_report.get("admet_ai_predictions"),
        label="ADMET and skin-sens report",
        allow_degraded=allow_degraded,
    )
    metrics: dict[str, float] = {}
    for key in SUMMARY_ADMET_METRICS:
        if key in values:
            metrics[key] = values[key]
    return metrics


def _validate_admet_ai_source(
    run_dir: Path,
    report_predictions: object,
    *,
    compound_smiles: str,
    allow_degraded: bool,
) -> None:
    path = run_dir / "02_admet" / "admet_ai.json"
    payload = _read_json(path, "ADMET-AI source predictions")
    status = payload.get("status")
    if status != "ok":
        if allow_degraded:
            return
        raise SystemExit(f"ADMET-AI source predictions status must be ok: {status!r}")
    _validate_source_smiles(payload, compound_smiles, "ADMET-AI source predictions")
    source_values = _validated_admet_predictions(
        payload.get("predictions"),
        label="ADMET-AI source predictions",
        allow_degraded=allow_degraded,
    )
    if not isinstance(report_predictions, dict):
        return
    report_values = _validated_admet_predictions(
        report_predictions,
        label="ADMET and skin-sens report",
        allow_degraded=allow_degraded,
    )
    for field in sorted(set(report_values) & set(source_values)):
        if abs(report_values[field] - source_values[field]) > 1e-9:
            raise SystemExit(f"source/report mismatch for endpoint {field}")


def _validate_structural_alert_source(
    run_dir: Path,
    report_alerts: object,
    *,
    compound_smiles: str,
) -> None:
    path = run_dir / "02_admet" / "structural_alerts.json"
    payload = _read_json(path, "structural-alert source evidence")
    _validate_source_smiles(payload, compound_smiles, "structural-alert source evidence")
    if not isinstance(payload.get("matches"), dict):
        raise SystemExit("structural-alert source evidence missing matches object")
    for key in ("any_pains", "any_brenk", "any_nih"):
        source_value = payload.get(key)
        if not isinstance(source_value, bool):
            raise SystemExit(f"structural-alert source evidence missing boolean {key}")
        if isinstance(report_alerts, dict):
            report_value = report_alerts.get(key)
            if isinstance(report_value, bool) and report_value != source_value:
                raise SystemExit(f"source/report mismatch for structural-alert flag {key}")
    if isinstance(report_alerts, dict):
        report_matches = report_alerts.get("matches")
        if isinstance(report_matches, dict) and payload.get("matches") != report_matches:
            raise SystemExit("source/report mismatch for structural-alert matches")


def _skin_sens_source_call(payload: dict[str, Any], model: str) -> tuple[object, object]:
    if model == "pred_skin":
        pred_skin = payload.get("pred_skin")
        probability = pred_skin.get("probability") if isinstance(pred_skin, dict) else None
        return payload.get("consensus_call"), probability
    return payload.get("skin_sens_call"), payload.get("skin_sens_probability")


def _validate_skin_sens_sources(
    run_dir: Path,
    report_skin_sens: dict[str, Any],
    *,
    compound_smiles: str,
    allow_degraded: bool,
) -> None:
    for model in SKIN_SENS_MODELS:
        path = run_dir / "02_admet" / f"{model}.json"
        payload = _read_json(path, f"{model} skin-sens source evidence")
        status = payload.get("status")
        if status != "ok":
            if allow_degraded:
                continue
            raise SystemExit(f"{model} skin-sens source evidence status must be ok: {status!r}")
        _validate_source_smiles(
            payload,
            compound_smiles,
            f"{model} skin-sens source evidence",
        )
        call, probability = _skin_sens_source_call(payload, model)
        if not isinstance(call, str) or call.strip() not in SKIN_SENS_CALLS:
            raise SystemExit(f"{model} call must be positive or negative")
        _fraction(probability, f"{model} skin-sens probability")
        report_call = report_skin_sens.get(model)
        if isinstance(report_call, str) and report_call.strip() and report_call.strip() != call.strip():
            raise SystemExit(f"source/report mismatch for {model} call")


def _skin_sens_source_evidence(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in SKIN_SENS_MODELS:
        path = run_dir / "02_admet" / f"{model}.json"
        payload = _read_json(path, f"{model} skin-sens source evidence")
        status = _text_or_none(payload.get("status")) or "n/a"
        call, probability = _skin_sens_source_call(payload, model)
        if status == "ok":
            call_value = _text_or_none(call)
            if call_value not in SKIN_SENS_CALLS:
                raise SystemExit(f"{model} call must be positive or negative")
            probability_value: float | None = _fraction(
                probability,
                f"{model} skin-sens probability",
            )
        else:
            call_value = _text_or_none(call)
            probability_value = None
        rows.append({
            "model": model,
            "status": status,
            "call": call_value,
            "probability": probability_value,
        })
    return rows


def _validate_safety_sources(
    run_dir: Path,
    admet_report: dict[str, Any],
    *,
    compound_smiles: str,
    allow_degraded: bool,
) -> None:
    skin_sens = admet_report.get("skin_sens")
    if not isinstance(skin_sens, dict):
        raise SystemExit("ADMET and skin-sens report missing object skin_sens")
    _validate_admet_ai_source(
        run_dir,
        admet_report.get("admet_ai_predictions"),
        compound_smiles=compound_smiles,
        allow_degraded=allow_degraded,
    )
    _validate_structural_alert_source(
        run_dir,
        admet_report.get("structural_alerts"),
        compound_smiles=compound_smiles,
    )
    _validate_skin_sens_sources(
        run_dir,
        skin_sens,
        compound_smiles=compound_smiles,
        allow_degraded=allow_degraded,
    )


def _admet_risk_level(value: float) -> str:
    if value >= ADMET_HIGH_RISK_THRESHOLD:
        return "high"
    if value >= ADMET_MODERATE_RISK_THRESHOLD:
        return "moderate"
    return "low"


def _admet_risk_assessment(admet_metrics: dict[str, float]) -> dict[str, Any]:
    endpoints: dict[str, dict[str, Any]] = {}
    for endpoint in ADMET_RISK_ENDPOINTS:
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


def _summary_artifacts(run_dir: Path, preset: str, mode: str) -> dict[str, str]:
    artifacts = {
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
    if preset in {"target-id", "report"}:
        artifacts["target_ranking"] = "03_targets/ranked_targets_v3_with_efficacy.csv"
        if mode in {"fast", "both"}:
            artifacts.update(
                FAST_TARGET_ARTIFACTS
                if _uses_daina_structural_contract(run_dir)
                else LEGACY_FAST_TARGET_ARTIFACTS
            )
        if mode in {"comprehensive", "both"}:
            artifacts.update(COMPREHENSIVE_TARGET_ARTIFACTS)
    if preset == "report":
        artifacts["html_report"] = "09_report/index.html"
    return artifacts


def _cosmetic_drug_summary(run_dir: Path) -> dict[str, Any]:
    cosing = _read_json(run_dir / "02b_cosmetic_drug" / "cosing_match.json", "CosIng annotation")
    drug = _read_json(
        run_dir / "02b_cosmetic_drug" / "drug_warnings.json",
        "drug-avoidance warnings",
    )
    decision_path = run_dir / "02b_cosmetic_drug" / "cosmetic_drug_decision.txt"
    try:
        decision, policy = parse_decision_and_policy(
            _read_text(decision_path, "cosmetic/drug decision"),
        )
        validate_decision_matches_drug_warnings(
            decision=decision,
            policy=policy,
            drug_warnings=drug,
        )
    except CosmeticDrugContractError as exc:
        raise SystemExit(str(exc)) from exc
    return {
        "decision": decision,
        "drug_policy": policy,
        "cosing_level": cosing.get("level"),
        "inci": cosing.get("inci"),
        "cosing_functions": cosing.get("functions", []),
        "cosing_tanimoto": _number(cosing.get("tanimoto"), "CosIng tanimoto")
        if "tanimoto" in cosing
        else None,
        "max_tanimoto_to_approved_drug": _number(
            drug.get("max_tanimoto_to_approved_drug"),
            "drug max_tanimoto_to_approved_drug",
        )
        if "max_tanimoto_to_approved_drug" in drug
        else None,
        "n_warnings": int(_number(drug.get("n_warnings"), "drug n_warnings"))
        if "n_warnings" in drug
        else None,
    }


def _efficacy_labels(row: pd.Series) -> list[str]:
    labels: list[str] = []
    for col in sorted(c for c in row.index if c.startswith("efficacy_top")):
        value = row[col]
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            labels.append(text)
    return labels


def _source_labels(value: object, *, row_index: int, path: Path) -> list[str]:
    if pd.isna(value) or not str(value).strip():
        raise SystemExit(
            "target ranking column 'sources' contains blank values "
            f"at row index {row_index}: {path}"
        )
    normalized = str(value).replace(",", ";").replace("|", ";")
    labels = [part.strip() for part in normalized.split(";") if part.strip()]
    if not labels:
        raise SystemExit(
            "target ranking column 'sources' contains empty labels "
            f"at row index {row_index}: {path}"
        )
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        shown = ", ".join(duplicates[:10])
        suffix = "..." if len(duplicates) > 10 else ""
        raise SystemExit(
            "target ranking column 'sources' contains duplicate labels "
            f"at row index {row_index}: {shown}{suffix}: {path}"
        )
    return labels


def _integer_at_least(value: object, label: str, *, minimum: int) -> int:
    numeric = _number(value, label)
    if not numeric.is_integer() or numeric < minimum:
        raise SystemExit(f"{label} must be an integer >= {minimum}")
    return int(numeric)


def _validate_target_ranking(
    df: pd.DataFrame,
    path: Path,
    *,
    min_source_count: int,
) -> None:
    seen_targets: set[str] = set()
    previous_final_score: float | None = None
    for idx, row in df.iterrows():
        row_index = int(idx)
        target_id = str(row["target_id"]).strip() if not pd.isna(row["target_id"]) else ""
        if not target_id:
            raise SystemExit(
                "target ranking column 'target_id' contains blank values "
                f"at row index {row_index}: {path}"
            )
        if target_id in seen_targets:
            raise SystemExit(f"target ranking contains duplicate target_id {target_id!r}: {path}")
        seen_targets.add(target_id)
        for column in sorted(TARGET_FRACTION_COLUMNS & set(df.columns)):
            _fraction(row[column], f"target ranking {column} at row index {row_index}")
        source_count = _integer_at_least(
            row["source_count"],
            f"target ranking source_count at row index {row_index}",
            minimum=min_source_count,
        )
        labels = _source_labels(row["sources"], row_index=row_index, path=path)
        if source_count != len(labels):
            raise SystemExit(
                f"target ranking source_count={source_count} but sources lists "
                f"{len(labels)} label(s) at row index {row_index}: {path}"
            )
        final_score = float(row["final_score"])
        if previous_final_score is not None and final_score > previous_final_score + 1e-12:
            raise SystemExit(f"target ranking final_score must be sorted descending: {path}")
        previous_final_score = final_score


def _optional_number(value: object) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _optional_int(value: object) -> int | None:
    number = _optional_number(value)
    return None if number is None else int(number)


def _optional_bool(value: object) -> bool | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "false"}:
            return token == "true"
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return bool(value)


def _target_brief(target: dict[str, Any]) -> dict[str, Any]:
    return {
        "target_id": target.get("target_id"),
        "gene_symbol": target.get("gene_symbol"),
        "protein_name": target.get("protein_name"),
        "final_score": target.get("final_score"),
        "final_score_semantics": target.get("final_score_semantics"),
        "skin_score": target.get("skin_score"),
        "skin_tier": target.get("skin_tier"),
        "cell_type_preferred": target.get("cell_type_preferred"),
        "docking_rrf": target.get("docking_rrf"),
        "daina_max_tanimoto": target.get("daina_max_tanimoto"),
        "daina_supporting_molecule_id": target.get("daina_supporting_molecule_id"),
        "daina_known_ligand_count": target.get("daina_known_ligand_count"),
        "daina_is_self_match": target.get("daina_is_self_match"),
        "ranking_basis": target.get("ranking_basis"),
        "source_count": target.get("source_count"),
        "sources": target.get("sources", []),
        "efficacy": target.get("efficacy", []),
    }


def _evidence_kind(target: dict[str, Any]) -> str | None:
    """Whether a row reports a measured interaction or extrapolates to a new one.

    A max Tanimoto of 1.0 means a reference ligand with a fingerprint identical
    to the query is already measured on this target, which is the common case
    for well-characterised cosmetic ingredients. Runs that predate the field
    render blank rather than guessing.
    """
    flag = target.get("daina_is_self_match")
    if flag is None or (isinstance(flag, str) and not flag.strip()):
        return None
    if isinstance(flag, str):
        return "retrieved" if flag.strip().lower() == "true" else "predicted"
    return "retrieved" if bool(flag) else "predicted"


def _skin_score_value(target: dict[str, Any]) -> float:
    try:
        return float(target.get("skin_score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _has_skin_expression_support(target: dict[str, Any]) -> bool:
    tier = _text_or_none(target.get("skin_tier"))
    return (
        (tier in SKIN_EXPRESSION_SUPPORTED_TIERS)
        or _skin_score_value(target) >= SKIN_EXPRESSION_SUPPORTED_THRESHOLD
    )


def _skin_context_decision(
    *,
    most_skin_relevant: dict[str, Any],
    top_targets_with_skin_efficacy: int,
) -> tuple[str, bool, bool, list[str]]:
    expression_supported = _has_skin_expression_support(most_skin_relevant)
    efficacy_supported = top_targets_with_skin_efficacy > 0
    reasons: list[str] = []
    if expression_supported:
        reasons.append("top targets include low-or-higher skin-expression support")
    else:
        reasons.append("top targets have very-low skin-expression support")
    if efficacy_supported:
        reasons.append("top targets include KG skin-efficacy labels")
    else:
        reasons.append("top targets lack KG skin-efficacy labels")

    if expression_supported and efficacy_supported:
        decision = "skin_context_supported"
    elif expression_supported:
        decision = "skin_expression_only"
    elif efficacy_supported:
        decision = "skin_efficacy_literature_only"
    else:
        decision = "insufficient_skin_context"
    return decision, expression_supported, efficacy_supported, reasons


def _skin_specialized_binding_summary(target_prediction: dict[str, Any]) -> dict[str, Any]:
    top_targets = target_prediction.get("top_targets")
    if not isinstance(top_targets, list) or not top_targets:
        return {
            "context": "skin-specialized material-protein binding",
            "ranking_path": target_prediction.get("ranking_path"),
            "skin_context_decision": "insufficient_skin_context",
            "skin_context_supported": False,
            "skin_expression_supported": False,
            "skin_efficacy_supported": False,
            "skin_context_reasons": [
                "target prediction produced no top target",
                "top targets lack KG skin-efficacy labels",
            ],
            "top_target_id": None,
            "top_target_gene_symbol": None,
            "top_target_protein_name": None,
            "top_target_final_score": None,
            "top_target_docking_rrf": None,
            "top_target_source_count": None,
            "top_target_sources": [],
            "top_target_skin_score": None,
            "top_target_skin_tier": None,
            "top_target_cell_type_preferred": None,
            "top_target_efficacy": [],
            "top_target_skin_expression_supported": False,
            "top_target_skin_efficacy_supported": False,
            "top_target_skin_context_supported": False,
            "top_targets_with_skin_efficacy": 0,
            "most_skin_relevant_target": None,
        }
    targets = [target for target in top_targets if isinstance(target, dict)]
    if not targets:
        return {
            "context": "skin-specialized material-protein binding",
            "ranking_path": target_prediction.get("ranking_path"),
            "skin_context_decision": "insufficient_skin_context",
            "skin_context_supported": False,
            "skin_expression_supported": False,
            "skin_efficacy_supported": False,
            "skin_context_reasons": [
                "target prediction produced no top target",
                "top targets lack KG skin-efficacy labels",
            ],
            "top_target_id": None,
            "top_target_gene_symbol": None,
            "top_target_protein_name": None,
            "top_target_final_score": None,
            "top_target_docking_rrf": None,
            "top_target_source_count": None,
            "top_target_sources": [],
            "top_target_skin_score": None,
            "top_target_skin_tier": None,
            "top_target_cell_type_preferred": None,
            "top_target_efficacy": [],
            "top_target_skin_expression_supported": False,
            "top_target_skin_efficacy_supported": False,
            "top_target_skin_context_supported": False,
            "top_targets_with_skin_efficacy": 0,
            "most_skin_relevant_target": None,
        }
    top_target = targets[0]
    top_target_expression_supported = _has_skin_expression_support(top_target)
    top_target_efficacy_supported = bool(top_target.get("efficacy"))
    most_skin_relevant = max(
        targets,
        key=lambda target: (
            _skin_score_value(target),
            bool(target.get("efficacy")),
            float(target.get("final_score") or 0.0),
        ),
    )
    top_targets_with_skin_efficacy = sum(1 for target in targets if target.get("efficacy"))
    (
        skin_context_decision,
        skin_expression_supported,
        skin_efficacy_supported,
        skin_context_reasons,
    ) = _skin_context_decision(
        most_skin_relevant=most_skin_relevant,
        top_targets_with_skin_efficacy=top_targets_with_skin_efficacy,
    )
    return {
        "context": "skin-specialized material-protein binding",
        "ranking_path": target_prediction.get("ranking_path"),
        "skin_context_decision": skin_context_decision,
        "skin_context_supported": skin_context_decision == "skin_context_supported",
        "skin_expression_supported": skin_expression_supported,
        "skin_efficacy_supported": skin_efficacy_supported,
        "skin_context_reasons": skin_context_reasons,
        "top_target_id": top_target.get("target_id"),
        "top_target_gene_symbol": top_target.get("gene_symbol"),
        "top_target_protein_name": top_target.get("protein_name"),
        "top_target_final_score": top_target.get("final_score"),
        "top_target_docking_rrf": top_target.get("docking_rrf"),
        "top_target_source_count": top_target.get("source_count"),
        "top_target_sources": top_target.get("sources", []),
        "top_target_skin_score": top_target.get("skin_score"),
        "top_target_skin_tier": top_target.get("skin_tier"),
        "top_target_cell_type_preferred": top_target.get("cell_type_preferred"),
        "top_target_efficacy": top_target.get("efficacy", []),
        "top_target_skin_expression_supported": top_target_expression_supported,
        "top_target_skin_efficacy_supported": top_target_efficacy_supported,
        "top_target_skin_context_supported": (
            top_target_expression_supported and top_target_efficacy_supported
        ),
        "top_targets_with_skin_efficacy": top_targets_with_skin_efficacy,
        "most_skin_relevant_target": _target_brief(most_skin_relevant),
    }


def _target_summary(
    run_dir: Path,
    *,
    mode: str,
    top_n: int,
    target_metadata: dict[str, dict[str, str]],
) -> dict[str, Any]:
    path = run_dir / "03_targets" / "ranked_targets_v3_with_efficacy.csv"
    curated_failure_modes = load_curated()
    df = _read_csv(path, "skin-weighted target ranking with efficacy")
    required = {"target_id", "final_score", "skin_score", "docking_rrf", "source_count", "sources"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"target ranking missing required columns {missing}: {path}")
    # Fast mode has one immutable ranking source (Daina); structure/KG/prior
    # fields are annotations and may be unavailable for an otherwise valid row.
    min_source_count = 1 if mode == "fast" else 3
    _validate_target_ranking(df, path, min_source_count=min_source_count)
    screening_counts: dict[str, int] = {
        "skin_weighted_ranked_targets": int(len(df)),
    }
    daina_structural = _uses_daina_structural_contract(run_dir)
    if mode in {"fast", "both"}:
        fast_artifacts = (
            FAST_SCREENING_COUNT_ARTIFACTS
            if daina_structural
            else LEGACY_FAST_SCREENING_COUNT_ARTIFACTS
        )
        for key, rel_path in fast_artifacts:
            screening_counts[key] = int(
                len(_read_table(run_dir / rel_path, f"target screening count {key}"))
            )
    if mode in {"comprehensive", "both"}:
        for key, rel_path in COMPREHENSIVE_SCREENING_COUNT_ARTIFACTS:
            screening_counts[key] = int(
                len(_read_table(run_dir / rel_path, f"target screening count {key}"))
            )
    _validate_screening_funnel(
        screening_counts,
        mode=mode,
        daina_structural=daina_structural,
    )
    top_targets: list[dict[str, Any]] = []
    for _, row in df.head(top_n).iterrows():
        target_id = str(row["target_id"]).strip()
        if not target_id:
            raise SystemExit(f"target ranking contains blank target_id: {path}")
        sources = _source_labels(row["sources"], row_index=int(row.name), path=path)
        metadata = target_metadata.get(target_id, {})
        top_targets.append(
            {
                "target_id": target_id,
                "gene_symbol": metadata.get("gene_symbol"),
                "protein_name": metadata.get("protein_name"),
                # Every docking miss on the validation panel came from a
                # receptor limitation rather than from weak binding, and the
                # artifacts never said which targets carry one.
                "failure_modes": failure_modes_for(
                    target_id,
                    molecular_function=metadata.get("molecular_function", ""),
                    curated=curated_failure_modes,
                ),
                "final_score": _number(row["final_score"], "target final_score"),
                "final_score_semantics": _text_or_none(row.get("final_score_semantics")),
                "skin_score": _number(row["skin_score"], "target skin_score"),
                "skin_tier": _text_or_none(row.get("skin_tier")),
                "cell_type_preferred": (
                    _text_or_none(row.get("cell_type_preferred")) or "unknown"
                ),
                "docking_rrf": _number(row["docking_rrf"], "target docking_rrf"),
                "daina_max_tanimoto": _optional_number(row.get("daina_max_tanimoto")),
                "daina_supporting_molecule_id": _text_or_none(
                    row.get("daina_supporting_molecule_id")
                ),
                "daina_known_ligand_count": _optional_int(
                    row.get("daina_known_ligand_count")
                ),
                "daina_is_self_match": _optional_bool(row.get("daina_is_self_match")),
                "ranking_basis": _text_or_none(row.get("ranking_basis")),
                "source_count": int(_number(row["source_count"], "target source_count")),
                "sources": sources,
                "efficacy": _efficacy_labels(row),
            }
        )
    return {
        "ranking_path": "03_targets/ranked_targets_v3_with_efficacy.csv",
        "n_targets": int(len(df)),
        "screened_target_count": max(screening_counts.values()),
        "screening_counts": screening_counts,
        "top_n": int(min(top_n, len(df))),
        "top_targets": top_targets,
    }


def _raise_decision(current: str, candidate: str) -> str:
    rank = {"PASS": 0, "FLAG_HIGH": 1, "HALT": 2}
    return candidate if rank[candidate] > rank[current] else current


def _overall_decision(
    *,
    preset: str,
    safety: dict[str, Any],
    skin_toxicity: dict[str, Any],
    cosmetic_drug: dict[str, Any],
    target_prediction: dict[str, Any] | None,
    skin_specialized_binding: dict[str, Any] | None,
) -> dict[str, Any]:
    decision = "PASS"
    reasons: list[str] = []

    skin_decision = safety.get("skin_sens_decision")
    if skin_decision == "HALT":
        decision = _raise_decision(decision, "HALT")
        reasons.append("skin sensitization consensus is HALT")
    elif skin_decision == "FLAG_HIGH":
        decision = _raise_decision(decision, "FLAG_HIGH")
        reasons.append("skin sensitization consensus is FLAG_HIGH")
    elif skin_decision != "PASS":
        decision = _raise_decision(decision, "FLAG_HIGH")
        reasons.append(f"skin sensitization decision is unrecognized: {skin_decision!r}")

    if safety.get("degraded"):
        decision = _raise_decision(decision, "FLAG_HIGH")
        reasons.append("safety evidence is degraded")
    missing_models = safety.get("missing_models")
    if isinstance(missing_models, list) and missing_models:
        decision = _raise_decision(decision, "FLAG_HIGH")
        reasons.append("missing skin-sens models: " + ",".join(str(model) for model in missing_models))

    skin_toxicity_decision = skin_toxicity.get("decision")
    if skin_toxicity_decision == "HALT":
        decision = _raise_decision(decision, "HALT")
        reasons.append("skin toxicity is HALT")
    elif skin_toxicity_decision == "REVIEW":
        decision = _raise_decision(decision, "FLAG_HIGH")
        reasons.append("skin toxicity requires review")
    elif skin_toxicity_decision != "PASS":
        decision = _raise_decision(decision, "FLAG_HIGH")
        reasons.append(f"skin toxicity decision is unrecognized: {skin_toxicity_decision!r}")

    structural_flags = safety.get("structural_alert_flags")
    if isinstance(structural_flags, dict):
        flagged = sorted(key for key, value in structural_flags.items() if value)
        if flagged:
            decision = _raise_decision(decision, "FLAG_HIGH")
            reasons.append("structural alerts present: " + ",".join(flagged))
    admet_risk = safety.get("admet_risk_assessment")
    if isinstance(admet_risk, dict):
        high_risk = admet_risk.get("high_risk_endpoints")
        if isinstance(high_risk, list) and high_risk:
            decision = _raise_decision(decision, "FLAG_HIGH")
            reasons.append("high ADMET risk endpoints: " + ",".join(str(item) for item in high_risk))
        endpoints = admet_risk.get("endpoints")
        skin_reaction = endpoints.get("Skin_Reaction") if isinstance(endpoints, dict) else None
        if isinstance(skin_reaction, dict) and skin_reaction.get("risk_level") == "moderate":
            decision = _raise_decision(decision, "FLAG_HIGH")
            reasons.append("Skin_Reaction ADMET risk requires review: moderate")

    cosmetic_decision = cosmetic_drug.get("decision")
    if cosmetic_decision == "HALT":
        decision = _raise_decision(decision, "HALT")
        reasons.append("cosmetic/drug gate is HALT")
    elif cosmetic_decision == "DOWNWEIGHT":
        decision = _raise_decision(decision, "FLAG_HIGH")
        reasons.append("cosmetic/drug gate is DOWNWEIGHT")
    elif cosmetic_decision != "PROCEED":
        decision = _raise_decision(decision, "FLAG_HIGH")
        reasons.append(f"cosmetic/drug decision is unrecognized: {cosmetic_decision!r}")

    warnings = cosmetic_drug.get("n_warnings")
    if isinstance(warnings, int) and warnings > 0:
        decision = _raise_decision(decision, "FLAG_HIGH")
        reasons.append(f"drug-avoidance warnings present: {warnings}")

    top_target_id = None
    target_count = None
    if preset in {"target-id", "report"}:
        if not isinstance(target_prediction, dict):
            decision = _raise_decision(decision, "FLAG_HIGH")
            reasons.append("target prediction summary is missing")
        else:
            target_count = target_prediction.get("n_targets")
            top_targets = target_prediction.get("top_targets")
            if isinstance(top_targets, list) and top_targets and isinstance(top_targets[0], dict):
                top_target_id = top_targets[0].get("target_id")
                top_gene = top_targets[0].get("gene_symbol")
                top_label = f"{top_gene} ({top_target_id})" if top_gene else str(top_target_id)
                reasons.append(f"top predicted target is {top_label} among {target_count} ranked targets")
            else:
                decision = _raise_decision(decision, "FLAG_HIGH")
                reasons.append("target prediction produced no top target")
            if not isinstance(skin_specialized_binding, dict):
                decision = _raise_decision(decision, "FLAG_HIGH")
                reasons.append("skin-specialized binding summary is missing")
            else:
                skin_context_supported = skin_specialized_binding.get("skin_context_supported")
                skin_context_decision = skin_specialized_binding.get("skin_context_decision")
                if skin_context_supported is not True:
                    decision = _raise_decision(decision, "FLAG_HIGH")
                    if isinstance(skin_context_decision, str) and skin_context_decision.strip():
                        reasons.append(
                            "skin-specialized binding context requires review: "
                            + skin_context_decision
                        )
                    else:
                        reasons.append(
                            "skin-specialized binding context is missing or unrecognized"
                        )
                if skin_specialized_binding.get("top_target_skin_context_supported") is not True:
                    decision = _raise_decision(decision, "FLAG_HIGH")
                    top_context_target = skin_specialized_binding.get("top_target_id")
                    suffix = (
                        f": {top_context_target}"
                        if isinstance(top_context_target, str) and top_context_target.strip()
                        else ""
                    )
                    reasons.append(
                        "top binding target lacks direct skin context support" + suffix
                    )

    if decision == "PASS" and "safety and cosmetic/drug gates passed" not in reasons:
        reasons.insert(0, "safety and cosmetic/drug gates passed")
    elif not reasons:
        reasons.append("safety and cosmetic/drug gates passed")

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
        "top_target_id": top_target_id,
        "target_count": target_count,
    }


def build_summary(
    run_dir: Path,
    *,
    preset: str,
    mode: str,
    top_n: int,
    target_metadata_path: Path | None = None,
    allow_degraded: bool = False,
) -> dict[str, Any]:
    compound_path = run_dir / "01_input" / "compound_canonical.json"
    compound = _validated_compound_metadata(
        _read_json(compound_path, "compound metadata"),
        compound_path,
    )
    admet = _read_json(run_dir / "02_admet" / "admet_report.json", "ADMET and skin-sens report")
    skin_sens = admet.get("skin_sens")
    if not isinstance(skin_sens, dict):
        raise SystemExit("ADMET and skin-sens report missing object skin_sens")
    decision = _read_first_line(run_dir / "02_admet" / "skin_sens_decision.txt", "skin-sens decision")
    report_decision = skin_sens.get("decision")
    if report_decision != decision:
        raise SystemExit("skin-sens decision file does not match ADMET report")

    skin_sens_calls, degraded, missing_models = _validated_skin_sens(
        skin_sens,
        decision=decision,
        allow_degraded=allow_degraded,
    )
    admet_metrics = _admet_metrics(admet, allow_degraded=allow_degraded)
    _validate_safety_sources(
        run_dir,
        admet,
        compound_smiles=compound["canonical_smiles"],
        allow_degraded=allow_degraded,
    )
    skin_sens_evidence = _skin_sens_source_evidence(run_dir)
    safety_summary = {
        "skin_sens_decision": decision,
        "skin_sens_calls": skin_sens_calls,
        "skin_sens_evidence": skin_sens_evidence,
        "degraded": degraded,
        "missing_models": missing_models,
        "admet_metrics": admet_metrics,
        "admet_risk_assessment": _admet_risk_assessment(admet_metrics),
        "structural_alert_flags": _validated_structural_alert_flags(admet),
    }
    skin_toxicity_summary = _skin_toxicity_summary(safety_summary)
    cosmetic_drug_summary = _cosmetic_drug_summary(run_dir)
    target_prediction = (
        _target_summary(
            run_dir,
            mode=mode,
            top_n=top_n,
            target_metadata=_load_target_metadata(target_metadata_path),
        )
        if preset in {"target-id", "report"}
        else None
    )
    skin_specialized_binding = (
        _skin_specialized_binding_summary(target_prediction)
        if target_prediction is not None
        else None
    )

    summary: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA,
        "run_id": run_dir.name,
        "preset": preset,
        "mode": mode,
        "compound": {
            "input_type": compound.get("input_type"),
            **(
                {"input_smiles": compound.get("input_smiles")}
                if compound.get("input_smiles") is not None
                else {}
            ),
            **(
                {"input_canonical_smiles": compound.get("input_canonical_smiles")}
                if compound.get("input_canonical_smiles") is not None
                else {}
            ),
            **(
                {"input_sdf": compound.get("input_sdf")}
                if compound.get("input_sdf") is not None
                else {}
            ),
            "canonical_smiles": compound.get("canonical_smiles"),
            "inchikey": compound.get("inchikey"),
            # Rebuilt field by field, so anything not named here is dropped -
            # which silently removed the applicability record and with it the
            # report section that reads from it.
            "applicability": compound.get("applicability"),
        },
        "safety": safety_summary,
        "skin_toxicity": skin_toxicity_summary,
        "cosmetic_drug": cosmetic_drug_summary,
        "artifacts": _summary_artifacts(run_dir, preset, mode),
        "overall_decision": _overall_decision(
            preset=preset,
            safety=safety_summary,
            skin_toxicity=skin_toxicity_summary,
            cosmetic_drug=cosmetic_drug_summary,
            target_prediction=target_prediction,
            skin_specialized_binding=skin_specialized_binding,
        ),
    }
    if target_prediction is not None:
        summary["target_prediction"] = target_prediction
        summary["skin_specialized_binding"] = skin_specialized_binding
    return summary


def write_summary(summary: dict[str, Any], out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_json.with_suffix(out_json.suffix + ".tmp")
    tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    tmp.replace(out_json)


def _display_value(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, list):
        return ", ".join(_display_value(item) for item in value) if value else "none"
    return str(value)


def _table_cell(value: object) -> str:
    return _display_value(value).replace("\n", " ").replace("|", r"\|")


def _korean_headline(overall: dict[str, Any]) -> list[str]:
    """A Korean block at the top, so the verdict is readable without the Workbench.

    The README sends a wet-lab reader to this file for "판정과 다음에 할 일". The
    body below is English and machine-oriented, and its decision word FLAG_HIGH
    appears nowhere in the Korean documentation - every Korean surface calls that
    REVIEW. This states the verdict, what it means, and the reasons in the same
    words the screen uses.
    """
    badge, meaning = decision_label(overall.get("decision"))
    lines = [
        "> ## 한눈에 보기",
        ">",
        f"> **판정: {badge}** - {meaning}",
    ]
    action = action_label(overall.get("recommended_action"))
    if action:
        lines.append(f"> **다음에 할 일:** {action}")
    if overall.get("claimable") is False:
        lines.append(
            "> 이 결과만으로 효능이나 직접 결합을 주장할 수 없습니다. "
            "표적 순위는 **실험 전 가설**입니다."
        )
    reasons = overall.get("reasons")
    if isinstance(reasons, list) and reasons:
        lines.extend([">", "> **그렇게 판정한 이유**", ">"])
        lines.extend(f"> - {reason_label(reason)}" for reason in reasons)
    lines.extend(
        [
            ">",
            "> 결과를 읽는 법은 README의 `결과 읽기` 절에, 어디까지 믿을 수 있는지는",
            "> `docs/RESEARCHER_GUIDE.md`에 실측 수치와 함께 있습니다.",
            "",
        ]
    )
    return lines


def render_markdown_summary(summary: dict[str, Any]) -> str:
    compound = summary.get("compound") if isinstance(summary.get("compound"), dict) else {}
    safety = summary.get("safety") if isinstance(summary.get("safety"), dict) else {}
    admet_risk = (
        safety.get("admet_risk_assessment")
        if isinstance(safety.get("admet_risk_assessment"), dict)
        else {}
    )
    skin_toxicity = (
        summary.get("skin_toxicity")
        if isinstance(summary.get("skin_toxicity"), dict)
        else {}
    )
    cosmetic_drug = (
        summary.get("cosmetic_drug")
        if isinstance(summary.get("cosmetic_drug"), dict)
        else {}
    )
    overall = (
        summary.get("overall_decision")
        if isinstance(summary.get("overall_decision"), dict)
        else {}
    )
    skin_binding = (
        summary.get("skin_specialized_binding")
        if isinstance(summary.get("skin_specialized_binding"), dict)
        else {}
    )

    lines = [
        "# SkinScout Run Summary",
        "",
        *_korean_headline(overall),
        "## Run",
        f"- Run: {_display_value(summary.get('run_id'))}",
        f"- Preset: {_display_value(summary.get('preset'))}",
        f"- Mode: {_display_value(summary.get('mode'))}",
        f"- Input type: {_display_value(compound.get('input_type'))}",
        f"- Input SMILES: `{_display_value(compound.get('input_smiles'))}`",
        f"- Input canonical SMILES: `{_display_value(compound.get('input_canonical_smiles'))}`",
        f"- Input SDF: `{_display_value(compound.get('input_sdf'))}`",
        f"- Compound: `{_display_value(compound.get('canonical_smiles'))}`",
        f"- InChIKey: {_display_value(compound.get('inchikey'))}",
    ]
    applicability = compound.get("applicability")
    if isinstance(applicability, dict) and applicability.get("warnings"):
        lines.extend(
            [
                "",
                "### Input applicability",
                "",
                "The input was accepted, but it sits outside the range the "
                "validation panel covers. The panel is 15 compounds, so this is "
                "a caution about untested territory rather than a verdict.",
                "",
            ]
        )
        for warning in applicability["warnings"]:
            if isinstance(warning, dict):
                lines.append(
                    f"- {_display_value(warning.get('detail'))} "
                    f"({_display_value(warning.get('basis'))})"
                )
    lines.extend([
        "",
        "## Overall Decision",
        f"- Decision: {_display_value(overall.get('decision'))}",
        f"- Recommended action: {_display_value(overall.get('recommended_action'))}",
        f"- Claimable: {_display_value(overall.get('claimable'))}",
        f"- Requires human review: {_display_value(overall.get('requires_human_review'))}",
    ])
    reasons = overall.get("reasons")
    if isinstance(reasons, list) and reasons:
        lines.extend(["", "### Reasons"])
        lines.extend(f"- {_display_value(reason)}" for reason in reasons)

    lines.extend(
        [
            "",
            "## Skin Toxicity",
            f"- Decision: {_display_value(skin_toxicity.get('decision'))}",
            f"- Toxicity level: {_display_value(skin_toxicity.get('toxicity_level'))}",
            f"- Skin sensitization: {_display_value(skin_toxicity.get('skin_sens_decision'))}",
            f"- Skin_Reaction risk: {_display_value(skin_toxicity.get('skin_reaction_risk_level'))}",
            f"- Skin_Reaction value: {_display_value(skin_toxicity.get('skin_reaction_value'))}",
            f"- Structural alerts present: {_display_value(skin_toxicity.get('structural_alerts_present'))}",
            f"- Structural alert flags: {_display_value(skin_toxicity.get('structural_alert_flags'))}",
            f"- Degraded evidence: {_display_value(skin_toxicity.get('degraded'))}",
            f"- Missing skin-sens models: {_display_value(skin_toxicity.get('missing_models'))}",
        ]
    )
    toxicity_reasons = skin_toxicity.get("reasons")
    if isinstance(toxicity_reasons, list) and toxicity_reasons:
        lines.extend(f"- Reason: {_display_value(reason)}" for reason in toxicity_reasons)

    lines.extend(
        [
            "",
            "## Safety And ADMET",
            f"- Skin sensitization: {_display_value(safety.get('skin_sens_decision'))}",
            f"- Degraded evidence: {_display_value(safety.get('degraded'))}",
            f"- Missing skin-sens models: {_display_value(safety.get('missing_models'))}",
        ]
    )
    skin_sens_evidence = safety.get("skin_sens_evidence")
    if isinstance(skin_sens_evidence, list) and skin_sens_evidence:
        lines.extend([
            "",
            "| Skin-sens model | Status | Call | Probability |",
            "|---|---|---|---:|",
        ])
        for row in skin_sens_evidence:
            if not isinstance(row, dict):
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        _table_cell(row.get("model")),
                        _table_cell(row.get("status")),
                        _table_cell(row.get("call")),
                        _table_cell(row.get("probability")),
                    ]
                )
                + " |"
            )
    endpoints = admet_risk.get("endpoints")
    if isinstance(endpoints, dict) and endpoints:
        lines.extend(["", "| Endpoint | Value | Risk |", "|---|---:|---|"])
        for endpoint in ADMET_RISK_ENDPOINTS:
            item = endpoints.get(endpoint)
            if not isinstance(item, dict):
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        _table_cell(endpoint),
                        _table_cell(item.get("value")),
                        _table_cell(item.get("risk_level")),
                    ]
                )
                + " |"
            )
    admet_metrics = safety.get("admet_metrics")
    if isinstance(admet_metrics, dict) and admet_metrics:
        lines.extend(["", "| ADMET Metric | Value |", "|---|---:|"])
        for metric in SUMMARY_ADMET_METRICS:
            if metric in admet_metrics:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            _table_cell(metric),
                            _table_cell(admet_metrics.get(metric)),
                        ]
                    )
                    + " |"
                )
    lines.extend(
        [
            f"- High risk endpoints: {_display_value(admet_risk.get('high_risk_endpoints'))}",
            f"- Moderate risk endpoints: {_display_value(admet_risk.get('moderate_risk_endpoints'))}",
        ]
    )

    lines.extend(
        [
            "",
            "## Cosmetic And Drug",
            f"- Decision: {_display_value(cosmetic_drug.get('decision'))}",
            f"- Drug policy: {_display_value(cosmetic_drug.get('drug_policy'))}",
            f"- CosIng level: {_display_value(cosmetic_drug.get('cosing_level'))}",
            f"- INCI: {_display_value(cosmetic_drug.get('inci'))}",
            f"- Drug warnings: {_display_value(cosmetic_drug.get('n_warnings'))}",
        ]
    )

    target_prediction = summary.get("target_prediction")
    if isinstance(target_prediction, dict):
        lines.extend(
            [
                "",
                "## Top Targets",
                f"- Ranked targets: {_display_value(target_prediction.get('n_targets'))}",
                f"- Screened target candidates: {_display_value(target_prediction.get('screened_target_count'))}",
                "",
                "| Rank | Target | Gene | Protein | Final | Skin | Skin Tier | Docking RRF | Nearest analog | Reference | Known ligands | Evidence | Sources | Efficacy |",
                "|---:|---|---|---|---:|---:|---|---:|---:|---|---:|---|---|---|",
            ]
        )
        top_targets = target_prediction.get("top_targets")
        if isinstance(top_targets, list):
            for rank, target in enumerate(top_targets, start=1):
                if not isinstance(target, dict):
                    continue
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(rank),
                            _table_cell(target.get("target_id")),
                            _table_cell(target.get("gene_symbol")),
                            _table_cell(target.get("protein_name")),
                            _table_cell(target.get("final_score")),
                            _table_cell(target.get("skin_score")),
                            _table_cell(target.get("skin_tier")),
                            _table_cell(target.get("docking_rrf")),
                            _table_cell(target.get("daina_max_tanimoto")),
                            _table_cell(target.get("daina_supporting_molecule_id")),
                            _table_cell(target.get("daina_known_ligand_count")),
                            _table_cell(_evidence_kind(target)),
                            _table_cell(target.get("sources")),
                            _table_cell(target.get("efficacy")),
                        ]
                    )
                    + " |"
                )
        limited = [
            target
            for target in (top_targets if isinstance(top_targets, list) else [])
            if isinstance(target, dict) and target.get("failure_modes")
        ]
        if limited:
            lines.extend(
                [
                    "",
                    "### Receptor limitations on these targets",
                    "",
                    "A docking score on these receptors is not evidence of weak "
                    "binding. Every docking miss on the validation panel came "
                    "from one of these limitations rather than from the compound "
                    "(`results/RETROSPECTIVE.md`).",
                    "",
                    "| Target | Gene | Limitation | Detail | Basis |",
                    "|---|---|---|---|---|",
                ]
            )
            for target in limited:
                for mode_record in target.get("failure_modes") or []:
                    if not isinstance(mode_record, dict):
                        continue
                    lines.append(
                        "| "
                        + " | ".join(
                            [
                                _table_cell(target.get("target_id")),
                                _table_cell(target.get("gene_symbol")),
                                _table_cell(mode_record.get("failure_mode")),
                                _table_cell(mode_record.get("detail")),
                                _table_cell(mode_record.get("basis")),
                            ]
                        )
                        + " |"
                    )
        screening_counts = target_prediction.get("screening_counts")
        if isinstance(screening_counts, dict) and screening_counts:
            lines.extend(["", "| Screening Stage | Targets |", "|---|---:|"])
            for key in sorted(screening_counts):
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            _table_cell(key),
                            _table_cell(screening_counts.get(key)),
                        ]
                    )
                    + " |"
                )

    if skin_binding:
        most_skin_relevant = skin_binding.get("most_skin_relevant_target")
        most_skin_relevant_entry = (
            most_skin_relevant if isinstance(most_skin_relevant, dict) else {}
        )
        most_label = "n/a"
        if most_skin_relevant_entry:
            most_label = _display_value(most_skin_relevant_entry.get("target_id"))
            gene = most_skin_relevant_entry.get("gene_symbol")
            if gene:
                most_label = f"{gene} ({most_label})"
        lines.extend(
            [
                "",
                "## Skin-Specialized Binding",
                f"- Context: {_display_value(skin_binding.get('context'))}",
                f"- Skin context decision: {_display_value(skin_binding.get('skin_context_decision'))}",
                f"- Skin context supported: {_display_value(skin_binding.get('skin_context_supported'))}",
                f"- Skin expression supported: {_display_value(skin_binding.get('skin_expression_supported'))}",
                f"- Skin efficacy supported: {_display_value(skin_binding.get('skin_efficacy_supported'))}",
                f"- Top predicted binding target: {_display_value(skin_binding.get('top_target_id'))}",
                f"- Top target gene: {_display_value(skin_binding.get('top_target_gene_symbol'))}",
                f"- Top target protein: {_display_value(skin_binding.get('top_target_protein_name'))}",
                f"- Top target final score: {_display_value(skin_binding.get('top_target_final_score'))}",
                f"- Top target docking RRF: {_display_value(skin_binding.get('top_target_docking_rrf'))}",
                f"- Top target source count: {_display_value(skin_binding.get('top_target_source_count'))}",
                f"- Top target sources: {_display_value(skin_binding.get('top_target_sources'))}",
                f"- Top target skin score: {_display_value(skin_binding.get('top_target_skin_score'))}",
                f"- Top target skin tier: {_display_value(skin_binding.get('top_target_skin_tier'))}",
                f"- Top target HPA cell context: {_display_value(skin_binding.get('top_target_cell_type_preferred'))}",
                f"- Top target skin efficacy: {_display_value(skin_binding.get('top_target_efficacy'))}",
                f"- Top target skin expression supported: {_display_value(skin_binding.get('top_target_skin_expression_supported'))}",
                f"- Top target skin efficacy supported: {_display_value(skin_binding.get('top_target_skin_efficacy_supported'))}",
                f"- Top target skin context supported: {_display_value(skin_binding.get('top_target_skin_context_supported'))}",
                f"- Most skin-relevant top target: {most_label}",
                f"- Most skin-relevant gene: {_display_value(most_skin_relevant_entry.get('gene_symbol'))}",
                f"- Most skin-relevant protein: {_display_value(most_skin_relevant_entry.get('protein_name'))}",
                f"- Most skin-relevant final score: {_display_value(most_skin_relevant_entry.get('final_score'))}",
                f"- Most skin-relevant docking RRF: {_display_value(most_skin_relevant_entry.get('docking_rrf'))}",
                f"- Most skin-relevant source count: {_display_value(most_skin_relevant_entry.get('source_count'))}",
                f"- Most skin-relevant sources: {_display_value(most_skin_relevant_entry.get('sources'))}",
                f"- Most skin-relevant skin score: {_display_value(most_skin_relevant_entry.get('skin_score'))}",
                f"- Most skin-relevant skin tier: {_display_value(most_skin_relevant_entry.get('skin_tier'))}",
                f"- Most skin-relevant HPA cell context: {_display_value(most_skin_relevant_entry.get('cell_type_preferred'))}",
                f"- Most skin-relevant efficacy: {_display_value(most_skin_relevant_entry.get('efficacy'))}",
                f"- Top targets with skin-efficacy evidence: {_display_value(skin_binding.get('top_targets_with_skin_efficacy'))}",
                "- Interpretation boundary: HPA cell context is an expression label, not evidence of cell-selective drug action.",
            ]
        )
        reasons = skin_binding.get("skin_context_reasons")
        if isinstance(reasons, list) and reasons:
            lines.extend(f"- Reason: {_display_value(reason)}" for reason in reasons)

    artifacts = summary.get("artifacts")
    if isinstance(artifacts, dict) and artifacts:
        lines.extend(["", "## Source Artifacts"])
        for key in sorted(artifacts):
            lines.append(f"- {key}: `{_display_value(artifacts[key])}`")

    return "\n".join(lines) + "\n"


def write_markdown_summary(summary: dict[str, Any], out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_md.with_suffix(out_md.suffix + ".tmp")
    tmp.write_text(render_markdown_summary(summary))
    tmp.replace(out_md)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="target-id")
    parser.add_argument("--mode", choices=sorted(MODES), default="comprehensive")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--target-metadata", type=Path,
                        help="Optional HPA-style TSV mapping UniProt IDs to gene/protein names.")
    parser.add_argument(
        "--allow-degraded",
        action="store_true",
        help="Allow explicitly degraded ADMET/skin-sens evidence in diagnostic summaries.",
    )
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-md", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.top_n < 1:
        raise SystemExit("--top-n must be >= 1")
    out_json = args.out_json or args.run_dir / "run_summary.json"
    summary = build_summary(
        args.run_dir,
        preset=args.preset,
        mode=args.mode,
        top_n=args.top_n,
        target_metadata_path=args.target_metadata,
        allow_degraded=args.allow_degraded,
    )
    write_summary(summary, out_json)
    out_md = args.out_md or args.run_dir / "run_summary.md"
    write_markdown_summary(summary, out_md)
    print(f"SkinScout run summary written: {out_json}")
    print(f"SkinScout markdown run summary written: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
