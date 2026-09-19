#!/usr/bin/env python3
"""Target-intent and candidate-decision contracts for target-first discovery.

The public functions deliberately exchange plain dictionaries so command-line
builders and report code can share the policy without importing pandas types.
Unknown values fail closed: missing evidence never becomes directional support,
and a computational PASS never becomes a biological safety claim.
"""

from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "skinscout.target-intent.v1"
REGISTRY_PATH = ROOT / "data" / "curation" / "target_intents_20260915.csv"

ROUTES = {"direct_target", "pathway", "endpoint"}
DESIRED_EFFECTS = {
    "inhibit_function",
    "activate_function",
    "decrease_expression",
    "increase_expression",
    "decrease_endpoint",
    "increase_endpoint",
    "context_required",
}
EVIDENCE_AXES = {
    "binding",
    "protein_function",
    "expression",
    "pathway_activity",
    "phenotype",
    "cooccurrence",
}
DIRECTNESS_VALUES = {"direct_supported", "indirect_supported", "unknown"}
MODEL_LEVELS = {
    "purified_protein",
    "engineered_cell",
    "primary_cell",
    "reconstructed_tissue",
    "ex_vivo_tissue",
    "in_vivo_animal",
    "human_study",
    "unknown",
}
ROLES = {"material_candidate", "reference_control", "exploratory_candidate"}
DECISIONS = {"experiment_priority", "needs_evidence", "exclude", "unsupported"}
DECISION_SUMMARY_KEYS = (
    "structure_valid",
    "route_supported",
    "analysis_applicability",
    "analysis_scope_supported",
    "direction_relation",
    "material_policy",
    "required_information_complete",
    "minimum_evidence_met",
    "functional_evidence",
)
DECISION_SAFETY_KEYS = (
    "run_valid",
    "safety_consensus_valid",
    "full_analysis_complete",
    "decision",
    "applicability_domain",
)

INTENT_KEYS = (
    "schema_version",
    "intent_id",
    "source_row_id",
    "source_sheet",
    "source_row_number",
    "category",
    "biomarker",
    "full_name",
    "marker_type",
    "role_summary",
    "source_direction",
    "route",
    "entity_type",
    "entity_name",
    "gene_symbol",
    "uniprot_id",
    "protein_form",
    "component_of",
    "desired_effect",
    "target_scope",
    "skin_compartment",
    "readout",
    "source_evidence",
    "target_taxid",
    "role",
    "docking_eligible",
)

_ALLOWED_AXES = {
    "inhibit_function": {"protein_function"},
    "activate_function": {"protein_function"},
    "decrease_expression": {"expression"},
    "increase_expression": {"expression"},
    "decrease_endpoint": {"pathway_activity", "phenotype"},
    "increase_endpoint": {"pathway_activity", "phenotype"},
    "context_required": set(),
}

_EFFECT_ALIASES = {
    "inhibit": "inhibit_function",
    "inhibition": "inhibit_function",
    "inhibitor": "inhibit_function",
    "antagonist": "inhibit_function",
    "blocker": "inhibit_function",
    "inhibit_function": "inhibit_function",
    "activate": "activate_function",
    "activation": "activate_function",
    "activator": "activate_function",
    "agonist": "activate_function",
    "activate_function": "activate_function",
    "decrease_expression": "decrease_expression",
    "expression_decrease": "decrease_expression",
    "downregulator": "decrease_expression",
    "increase_expression": "increase_expression",
    "expression_increase": "increase_expression",
    "upregulator": "increase_expression",
    "decrease_endpoint": "decrease_endpoint",
    "endpoint_decrease": "decrease_endpoint",
    "increase_endpoint": "increase_endpoint",
    "endpoint_increase": "increase_endpoint",
}

_OPPOSITE_EFFECT = {
    "inhibit_function": "activate_function",
    "activate_function": "inhibit_function",
    "decrease_expression": "increase_expression",
    "increase_expression": "decrease_expression",
    "decrease_endpoint": "increase_endpoint",
    "increase_endpoint": "decrease_endpoint",
}


def _text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _nullable(value: Any) -> str | None:
    text = _text(value)
    return text or None


def _bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = _text(value).lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{field} must be true or false")


def _load_workbook_rows(path: Path) -> list[dict[str, Any]]:
    import pandas as pd

    raw = pd.read_excel(path, sheet_name=0, header=None)
    header_index: int | None = None
    for index, row in raw.iterrows():
        values = {_text(value) for value in row.tolist()}
        if "바이오마커 (약칭)" in values and "카테고리" in values:
            header_index = int(index)
            break
    if header_index is None:
        raise ValueError(f"target workbook header was not found: {path}")

    headers = [_text(value) for value in raw.iloc[header_index].tolist()]
    required = {
        "카테고리",
        "바이오마커 (약칭)",
        "바이오마커 풀네임",
        "마커 유형",
        "역할 요약",
        "효능 지표 방향",
    }
    if not required.issubset(headers):
        raise ValueError(
            f"target workbook is missing required columns: {sorted(required - set(headers))}"
        )

    rows: list[dict[str, Any]] = []
    for index in range(header_index + 1, len(raw)):
        values = {
            _text(headers[column]): raw.iat[index, column]
            for column in range(len(headers))
            if headers[column]
        }
        biomarker = _text(values.get("바이오마커 (약칭)"))
        if not biomarker:
            continue
        rows.append(
            {
                "source_row_number": index + 1,
                "category": _text(values.get("카테고리")),
                "biomarker": biomarker,
                "full_name": _text(values.get("바이오마커 풀네임")),
                "marker_type": _text(values.get("마커 유형")),
                "role_summary": _text(values.get("역할 요약")),
                "source_direction": _text(values.get("효능 지표 방향")),
            }
        )
    return rows


def _load_gene_map(path: Path) -> dict[str, set[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"gene", "uniprot"}.issubset(reader.fieldnames):
            raise ValueError("gene map must contain gene and uniprot columns")
        result: dict[str, set[str]] = {}
        for row in reader:
            label = _text(row.get("gene"))
            if not label:
                continue
            result[label] = {
                part.strip()
                for part in _text(row.get("uniprot")).split(";")
                if part.strip()
            }
    return result


def _load_registry() -> list[dict[str, str]]:
    with REGISTRY_PATH.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 33:
        raise ValueError(
            f"target intent registry must contain exactly 33 rows, found {len(rows)}"
        )
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_target_intents(workbook_path: Path, gene_map_path: Path) -> list[dict]:
    """Build the curated 33-intent registry from the original 31 workbook rows.

    The workbook supplies exact source language and row lineage.  The curated
    registry supplies interpretation that cannot safely be inferred from Korean
    arrows alone.  Any workbook or gene-map drift fails closed.
    """

    workbook_path = Path(workbook_path)
    gene_map_path = Path(gene_map_path)
    workbook_rows = _load_workbook_rows(workbook_path)
    if len(workbook_rows) != 31:
        raise ValueError(
            f"target workbook must contain exactly 31 biomarker rows, found {len(workbook_rows)}"
        )
    by_biomarker = {row["biomarker"]: row for row in workbook_rows}
    if len(by_biomarker) != 31:
        raise ValueError("target workbook biomarker labels must be unique")
    gene_map = _load_gene_map(gene_map_path)
    workbook_sha256 = _sha256(workbook_path)
    gene_map_sha256 = _sha256(gene_map_path)
    registry_sha256 = _sha256(REGISTRY_PATH)

    intents: list[dict] = []
    seen_ids: set[str] = set()
    for curated in _load_registry():
        biomarker = _text(curated.get("biomarker"))
        source = by_biomarker.get(biomarker)
        if source is None:
            raise ValueError(f"curated biomarker is absent from workbook: {biomarker}")
        expected_row = int(_text(curated.get("source_row_number")))
        if source["source_row_number"] != expected_row:
            raise ValueError(
                f"workbook row drift for {biomarker}: expected {expected_row}, found {source['source_row_number']}"
            )

        uniprot_id = _nullable(curated.get("uniprot_id"))
        gene_map_label = _nullable(curated.get("gene_map_label"))
        if uniprot_id and (
            not gene_map_label or uniprot_id not in gene_map.get(gene_map_label, set())
        ):
            raise ValueError(f"gene-map does not support {biomarker} -> {uniprot_id}")

        intent = {
            "schema_version": SCHEMA_VERSION,
            "intent_id": _text(curated.get("intent_id")),
            "source_row_id": _text(curated.get("source_row_id")),
            "source_sheet": "Sheet1",
            "source_row_number": expected_row,
            **{
                key: source[key]
                for key in (
                    "category",
                    "biomarker",
                    "full_name",
                    "marker_type",
                    "role_summary",
                    "source_direction",
                )
            },
            "route": _text(curated.get("route")),
            "entity_type": _text(curated.get("entity_type")),
            "entity_name": _text(curated.get("entity_name")),
            "gene_symbol": _nullable(curated.get("gene_symbol")),
            "uniprot_id": uniprot_id,
            "protein_form": _nullable(curated.get("protein_form")),
            "component_of": _nullable(curated.get("component_of")),
            "desired_effect": _text(curated.get("desired_effect")),
            "target_scope": _text(curated.get("target_scope")),
            "skin_compartment": _text(curated.get("skin_compartment")),
            "readout": _text(curated.get("readout")),
            "source_evidence": {
                "workbook_path": str(workbook_path),
                "workbook_sha256": workbook_sha256,
                "sheet": "Sheet1",
                "row_number": expected_row,
                "gene_map_path": str(gene_map_path),
                "gene_map_sha256": gene_map_sha256,
                "registry_path": str(REGISTRY_PATH),
                "registry_sha256": registry_sha256,
                "curated_on": "2026-09-15",
            },
            "target_taxid": int(curated["target_taxid"])
            if _text(curated.get("target_taxid"))
            else None,
            "role": _text(curated.get("role")),
            "docking_eligible": _bool(
                curated.get("docking_eligible"), field="docking_eligible"
            ),
        }
        validate_intent(intent)
        if intent["intent_id"] in seen_ids:
            raise ValueError(f"duplicate intent_id: {intent['intent_id']}")
        seen_ids.add(intent["intent_id"])
        intents.append(intent)

    source_ids = {intent["source_row_id"] for intent in intents}
    if len(source_ids) != 31:
        raise ValueError(
            f"target intents must retain exactly 31 source rows, found {len(source_ids)}"
        )
    return intents


def validate_intent(row: dict) -> None:
    """Raise ``ValueError`` unless *row* satisfies the v1 intent contract."""

    if not isinstance(row, dict):
        raise ValueError("intent must be a dictionary")  # noqa: TRY004 - public contract
    missing = [key for key in INTENT_KEYS if key not in row]
    extra = sorted(set(row) - set(INTENT_KEYS))
    if missing:
        raise ValueError(f"intent is missing required keys: {missing}")
    if extra:
        raise ValueError(f"intent has unknown keys: {extra}")
    if row["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    for key in (
        "intent_id",
        "source_row_id",
        "source_sheet",
        "category",
        "biomarker",
        "full_name",
        "marker_type",
        "role_summary",
        "source_direction",
        "entity_type",
        "entity_name",
        "target_scope",
        "skin_compartment",
        "readout",
    ):
        if not isinstance(row[key], str) or not row[key].strip():
            raise ValueError(f"{key} must be a non-empty string")
    if (
        not isinstance(row["source_row_number"], int)
        or isinstance(row["source_row_number"], bool)
        or row["source_row_number"] < 1
    ):
        raise ValueError("source_row_number must be a positive integer")
    if row["route"] not in ROUTES:
        raise ValueError(f"invalid route: {row['route']!r}")
    if row["desired_effect"] not in DESIRED_EFFECTS:
        raise ValueError(f"invalid desired_effect: {row['desired_effect']!r}")
    if row["role"] not in ROLES:
        raise ValueError(f"invalid role: {row['role']!r}")
    if not isinstance(row["docking_eligible"], bool):
        raise ValueError("docking_eligible must be a boolean")  # noqa: TRY004 - contract violation
    if not isinstance(row["source_evidence"], dict):
        raise ValueError("source_evidence must be a dictionary")  # noqa: TRY004 - contract violation
    source_required = {
        "workbook_path",
        "workbook_sha256",
        "sheet",
        "row_number",
        "gene_map_path",
        "gene_map_sha256",
        "registry_path",
        "registry_sha256",
        "curated_on",
    }
    if set(row["source_evidence"]) != source_required:
        raise ValueError("source_evidence keys do not match the v1 contract")
    for hash_key in ("workbook_sha256", "gene_map_sha256", "registry_sha256"):
        if not re.fullmatch(
            r"[0-9a-f]{64}", _text(row["source_evidence"].get(hash_key))
        ):
            raise ValueError(f"source_evidence.{hash_key} must be a lowercase SHA-256")
    if row["source_evidence"].get("row_number") != row["source_row_number"]:
        raise ValueError("source_evidence.row_number must match source_row_number")

    nullable_strings = ("gene_symbol", "uniprot_id", "protein_form", "component_of")
    for key in nullable_strings:
        if row[key] is not None and (
            not isinstance(row[key], str) or not row[key].strip()
        ):
            raise ValueError(f"{key} must be null or a non-empty string")
    if row["target_taxid"] is not None and (
        not isinstance(row["target_taxid"], int)
        or isinstance(row["target_taxid"], bool)
        or row["target_taxid"] < 1
    ):
        raise ValueError("target_taxid must be null or a positive integer")

    if row["route"] == "endpoint":
        if (
            row["uniprot_id"] is not None
            or row["target_taxid"] is not None
            or row["docking_eligible"]
        ):
            raise ValueError(
                "endpoint intents cannot have UniProt/taxid or be docking eligible"
            )
    else:
        if row["uniprot_id"] is None or row["target_taxid"] is None:
            raise ValueError("protein/pathway intents require UniProt and target_taxid")
    if row["docking_eligible"] and row["route"] != "direct_target":
        raise ValueError("only direct_target intents may be docking eligible")


def _normalize_effect(evidence: dict) -> tuple[str | None, str | None]:
    raw = (
        _text(evidence.get("observed_effect") or evidence.get("action_type"))
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )
    if not raw:
        return None, "effect direction is not reported"
    if any(
        token in raw for token in ("partial_agonist", "inverse_agonist", "allosteric")
    ):
        return (
            None,
            f"{raw} requires assay context and is not collapsed to a binary direction",
        )
    normalized = _EFFECT_ALIASES.get(raw)
    if normalized is None:
        return None, f"unrecognized effect direction: {raw}"
    return normalized, None


def classify_evidence(intent: dict, evidence: dict) -> dict:
    """Classify one evidence record without inferring facts absent from the source."""

    validate_intent(intent)
    if not isinstance(evidence, dict):
        raise ValueError("evidence must be a dictionary")  # noqa: TRY004 - public contract
    axis = _text(evidence.get("evidence_axis")) or "cooccurrence"
    if axis not in EVIDENCE_AXES:
        raise ValueError(f"invalid evidence_axis: {axis!r}")
    directness = _text(evidence.get("directness")) or "unknown"
    if directness not in DIRECTNESS_VALUES:
        raise ValueError(f"invalid directness: {directness!r}")
    model_level = _text(evidence.get("model_level")) or "unknown"
    if model_level not in MODEL_LEVELS:
        raise ValueError(f"invalid model_level: {model_level!r}")

    result = {
        "evidence_axis": axis,
        "directness": directness,
        "direction_relation": "unknown",
        "reason": "",
    }
    if axis == "cooccurrence":
        result["directness"] = "unknown"
        result["reason"] = "cooccurrence does not establish causality or direction"
        return result
    if axis == "binding":
        result["reason"] = (
            "binding alone does not establish the requested functional direction"
        )
        return result
    if axis not in _ALLOWED_AXES[intent["desired_effect"]]:
        result["reason"] = f"{axis} evidence cannot answer {intent['desired_effect']}"
        return result

    evidence_target = _nullable(evidence.get("target_uniprot_id"))
    evidence_taxid = evidence.get("target_taxid")
    if intent["route"] == "endpoint":
        if evidence_target is not None or evidence_taxid is not None:
            result["reason"] = "endpoint evidence cannot use a protein target identity"
            return result
        if _nullable(evidence.get("target_intent_id")) != intent["intent_id"]:
            result["reason"] = (
                "endpoint evidence intent identity is missing or mismatched"
            )
            return result
    else:
        if evidence_target != intent["uniprot_id"]:
            result["reason"] = (
                "evidence target UniProt identity is missing or mismatched"
            )
            return result
        if evidence_taxid != intent["target_taxid"]:
            result["reason"] = "evidence target taxid is missing or mismatched"
            return result
    effect, uncertainty = _normalize_effect(evidence)
    if uncertainty:
        result["reason"] = uncertainty
        return result
    desired = intent["desired_effect"]
    if effect == desired:
        result["direction_relation"] = "supports"
        result["reason"] = (
            f"reported {axis} effect matches {desired}; model_level={model_level}"
        )
    elif effect == _OPPOSITE_EFFECT.get(desired):
        result["direction_relation"] = "opposes"
        result["reason"] = (
            f"reported {axis} effect opposes {desired}; model_level={model_level}"
        )
    else:
        result["reason"] = f"reported effect {effect} is not comparable with {desired}"
    return result


def decide_candidate(intent: dict, summary: dict, safety: dict) -> dict:
    """Apply the approved fail-closed candidate decision precedence."""

    validate_intent(intent)
    if not isinstance(summary, dict) or not isinstance(safety, dict):
        raise ValueError("summary and safety must be dictionaries")  # noqa: TRY004 - public contract

    direction = summary.get("direction_relation", "unknown")
    material_policy = summary.get("material_policy", "unknown")
    analysis_applicability = summary.get("analysis_applicability", "unknown")
    # The raw applicability state (`in_scope`/`review`/`out_of_scope`) and the
    # computed support flag are separate inputs. Collapsing `review` into
    # `in_scope` let a panel-range caution reach experiment_priority.
    analysis_scope_supported = summary.get("analysis_scope_supported")
    if analysis_scope_supported is None:
        analysis_scope_supported = analysis_applicability == "in_scope"
    safety_decision = safety.get("decision", "UNAVAILABLE")
    safety_domain = safety.get("applicability_domain", "unknown")
    # C20: consensus validity, completion, and the legacy run flag are separate
    # states.  A record never gains a valid consensus by implication from
    # run_valid, while a partial two-positive HALT remains explicitly valid.
    consensus_valid = safety.get("safety_consensus_valid") is True
    full_analysis_complete = (
        safety.get("run_valid") is True
        and safety.get("full_analysis_complete") is True
    )
    if direction not in {"supports", "opposes", "unknown", "conflict"}:
        raise ValueError(f"invalid direction_relation: {direction!r}")
    if material_policy not in {"eligible", "excluded", "unknown"}:
        raise ValueError(f"invalid material_policy: {material_policy!r}")
    if analysis_applicability not in {"in_scope", "review", "out_of_scope", "unknown"}:
        raise ValueError(f"invalid analysis_applicability: {analysis_applicability!r}")
    if not isinstance(analysis_scope_supported, bool):
        raise ValueError(
            f"invalid analysis_scope_supported: {analysis_scope_supported!r}"
        )
    if safety_decision not in {"PASS", "FLAG_HIGH", "HALT", "UNAVAILABLE"}:
        raise ValueError(f"invalid safety decision: {safety_decision!r}")
    if safety_domain not in {"inside", "review", "outside", "unknown"}:
        raise ValueError(f"invalid safety applicability_domain: {safety_domain!r}")

    reasons: list[str] = []
    if summary.get("structure_valid") is not True:
        reasons.append("exact compound structure is not validated")
    if summary.get("route_supported") is not True:
        reasons.append("target or endpoint route is not validated")
    if analysis_applicability == "out_of_scope":
        reasons.append("requested analysis is outside the supported scope")
    if reasons:
        return {"decision": "unsupported", "reasons": reasons}

    if consensus_valid and safety_decision == "HALT":
        reasons.append("current valid safety run is HALT")
    if direction == "opposes":
        reasons.append("aggregate evidence direction opposes the target intent")
    if material_policy == "excluded":
        reasons.append("material policy explicitly excludes this candidate")
    if reasons:
        return {"decision": "exclude", "reasons": reasons}

    if not analysis_scope_supported or analysis_applicability == "review":
        if analysis_applicability == "review":
            reasons.append("analysis applicability requires review")
        elif analysis_applicability == "unknown":
            reasons.append("analysis applicability is unknown")
        else:
            reasons.append("analysis scope support is not established")
    if not full_analysis_complete:
        reasons.append("no valid current safety run")
    if safety_decision == "FLAG_HIGH":
        reasons.append("safety prediction is FLAG_HIGH")
    elif safety_decision != "PASS":
        reasons.append("safety prediction is missing or unknown")
    if safety_domain != "inside":
        reasons.append("safety applicability domain is not confirmed inside")
    if direction == "conflict":
        reasons.append("applicable evidence has conflicting directions")
    elif direction != "supports":
        reasons.append("supporting direction is not established")
    if summary.get("required_information_complete") is not True:
        reasons.append(
            "required concentration, condition, or assay information is incomplete"
        )
    if summary.get("minimum_evidence_met") is not True:
        reasons.append("minimum independent primary evidence is not met")
    if summary.get("functional_evidence") is not True:
        reasons.append("functional or endpoint evidence is missing")
    if material_policy != "eligible":
        reasons.append("material eligibility is not established")
    if reasons:
        return {"decision": "needs_evidence", "reasons": reasons}

    return {
        "decision": "experiment_priority",
        "reasons": [
            "exact structure and intent mapping are validated",
            "directional functional or endpoint evidence meets the minimum evidence policy",
            "current computation is PASS within its stated applicability domain",
            "PASS is a computational screen and does not establish biological safety",
        ],
    }


__all__ = [
    "DECISIONS",
    "DECISION_SAFETY_KEYS",
    "DECISION_SUMMARY_KEYS",
    "DESIRED_EFFECTS",
    "DIRECTNESS_VALUES",
    "EVIDENCE_AXES",
    "INTENT_KEYS",
    "MODEL_LEVELS",
    "ROLES",
    "ROUTES",
    "SCHEMA_VERSION",
    "build_target_intents",
    "classify_evidence",
    "decide_candidate",
    "validate_intent",
]
