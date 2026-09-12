"""Canonical SkinScout run-profile and versioned run-manifest contracts.

All public entry points must resolve presets through :func:`resolve_run_profile`.
Legacy v1 payloads remain supported through ``RunProfile.legacy_fields()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping


RUN_MANIFEST_SCHEMA = "skinscout.run_manifest.v2"
RUN_MANIFEST_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2] / "schemas" / "run_manifest_v2.json"
)
RUN_MANIFEST_SCHEMA_V3 = "skinscout.run_manifest.v3"
RUN_MANIFEST_SCHEMA_V3_PATH = (
    Path(__file__).resolve().parents[2] / "schemas" / "run_manifest_v3.json"
)
TARGET_FAST_SCHEMA_V2 = "skinscout.target_fast.v2"
TARGET_FAST_SCHEMA_V2_PATH = (
    Path(__file__).resolve().parents[2] / "schemas" / "target_fast_v2.json"
)
DAINA_STRUCTURAL_OVERLAY_RECIPE_ID = "daina-structural-overlay-v1"
DAINA_STRUCTURAL_TARGETS_ARTIFACT_ID = "target_fast_daina_structural_targets"
DAINA_STRUCTURAL_TARGETS_PATH = "03_targets/mode_fast/daina_structural_targets.csv"
TOP50_COMPAT_ARTIFACT_ID = "target_fast_top50_compat"
TOP50_COMPAT_PATH = "03_targets/mode_fast/top50.csv"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class NormalizedPreset(str, Enum):
    STAGE0 = "stage0"
    SAFETY = "safety"
    TARGET_ID = "target-id"
    REPORT = "report"


class ExecutionMode(str, Enum):
    FAST = "fast"
    COMPREHENSIVE = "comprehensive"
    BOTH = "both"


class AnalysisProfile(str, Enum):
    STAGE0 = "stage0"
    SAFETY = "safety"
    TARGET_FAST = "target_fast"
    TARGET_COMPREHENSIVE = "target_comprehensive"
    TARGET_COMPARE = "target_compare"
    REPORT_FAST = "report_fast"
    PHYSICS_FULL = "physics_full"


class EvidenceMode(str, Enum):
    EVIDENCE = "evidence"
    DISCOVERY = "discovery"


class ContextProfile(str, Enum):
    AUTO = "auto"
    GENERAL_SKIN = "general_skin"
    PIGMENTATION = "pigmentation"
    ANTI_AGING = "anti_aging"
    BARRIER = "barrier"
    ACNE = "acne"
    INFLAMMATION = "inflammation"
    IRRITATION_SENSITIZATION = "irritation_sensitization"


BASE_PRESETS = frozenset(item.value for item in NormalizedPreset)
SOTA_PRESET_ALIASES = {
    "target-id-sota": NormalizedPreset.TARGET_ID.value,
    "report-sota": NormalizedPreset.REPORT.value,
}
PRESETS = BASE_PRESETS | frozenset(SOTA_PRESET_ALIASES)
MODES = frozenset(item.value for item in ExecutionMode)
ANALYSIS_PROFILES = frozenset(item.value for item in AnalysisProfile)
EVIDENCE_MODES = frozenset(item.value for item in EvidenceMode)
CONTEXT_PROFILES = frozenset(item.value for item in ContextProfile)

FAST_TARGET_ARTIFACT_IDS = (
    "target_fast_psichic",
    "target_fast_daina_zoete",
    "target_fast_dti_rrf",
    "target_fast_autodock",
    "target_fast_rerank_consensus",
)
PHYSICS_TARGET_ARTIFACT_IDS = (
    "target_comprehensive_ligand",
    "target_comprehensive_autodock",
    "target_comprehensive_pre_rescore",
    "target_comprehensive_gnina",
    "target_comprehensive_rtmscore",
    "target_comprehensive_boltz2",
    "target_comprehensive_consensus",
)
TARGET_FAST_STRUCTURAL_STATUSES = frozenset(
    {
        "structure_supported",
        "structural_unavailable_no_pocket",
        "structural_unavailable_prep",
        "structural_unavailable_invalid",
        "structure_failed_map",
        "structure_failed_docking",
        "structure_failed_gnina",
        "structure_not_requested",
    }
)


def _sha256_file(path: Path, label: str) -> str:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"{label} schema could not be read: {path}: {exc}") from exc
    return hashlib.sha256(data).hexdigest()


def _enum_value(enum_type: type[Enum], value: str, label: str) -> Enum:
    try:
        return enum_type(value)
    except ValueError as exc:
        allowed = ", ".join(sorted(item.value for item in enum_type))
        raise ValueError(f"{label} must be one of: {allowed}; got {value!r}") from exc


def _analysis_profile(
    preset: NormalizedPreset,
    mode: ExecutionMode,
) -> AnalysisProfile:
    if preset is NormalizedPreset.STAGE0:
        return AnalysisProfile.STAGE0
    if preset is NormalizedPreset.SAFETY:
        return AnalysisProfile.SAFETY
    if preset is NormalizedPreset.TARGET_ID:
        return {
            ExecutionMode.FAST: AnalysisProfile.TARGET_FAST,
            ExecutionMode.COMPREHENSIVE: AnalysisProfile.TARGET_COMPREHENSIVE,
            ExecutionMode.BOTH: AnalysisProfile.TARGET_COMPARE,
        }[mode]
    if mode is ExecutionMode.FAST:
        return AnalysisProfile.REPORT_FAST
    return AnalysisProfile.PHYSICS_FULL


@dataclass(frozen=True, slots=True)
class RunProfile:
    requested_preset: str
    normalized_preset: NormalizedPreset
    execution_mode: ExecutionMode
    analysis_profile: AnalysisProfile
    evidence_mode: EvidenceMode
    context_profile: ContextProfile
    sota_claim: bool

    def to_dict(self) -> dict[str, str | bool]:
        return {
            "requested_preset": self.requested_preset,
            "normalized_preset": self.normalized_preset.value,
            "execution_mode": self.execution_mode.value,
            "analysis_profile": self.analysis_profile.value,
            "evidence_mode": self.evidence_mode.value,
            "context_profile": self.context_profile.value,
            "sota_claim": self.sota_claim,
        }

    def legacy_fields(self) -> dict[str, str | bool]:
        """Return the unchanged v1 launcher/readiness field projection."""

        return {
            "preset": self.normalized_preset.value,
            "requested_preset": self.requested_preset,
            "mode": self.execution_mode.value,
            "sota_claim": self.sota_claim,
            "context_profile": self.context_profile.value,
        }

    def expected_fast_artifact_ids(self) -> tuple[str, ...]:
        if self.normalized_preset not in {
            NormalizedPreset.TARGET_ID,
            NormalizedPreset.REPORT,
        }:
            return ()
        if self.execution_mode not in {ExecutionMode.FAST, ExecutionMode.BOTH}:
            return ()
        return FAST_TARGET_ARTIFACT_IDS

    def expected_physics_artifact_ids(self) -> tuple[str, ...]:
        if self.normalized_preset not in {
            NormalizedPreset.TARGET_ID,
            NormalizedPreset.REPORT,
        }:
            return ()
        if self.execution_mode not in {
            ExecutionMode.COMPREHENSIVE,
            ExecutionMode.BOTH,
        }:
            return ()
        return PHYSICS_TARGET_ARTIFACT_IDS


def resolve_run_profile(
    requested_preset: str,
    execution_mode: str,
    *,
    evidence_mode: str = EvidenceMode.EVIDENCE.value,
    context_profile: str = ContextProfile.AUTO.value,
    sota_claim: bool = False,
) -> RunProfile:
    """Resolve every public run selector into one immutable profile."""

    if requested_preset not in PRESETS:
        allowed = ", ".join(sorted(PRESETS))
        raise ValueError(
            f"requested_preset must be one of: {allowed}; got {requested_preset!r}"
        )
    normalized = SOTA_PRESET_ALIASES.get(requested_preset, requested_preset)
    preset = _enum_value(NormalizedPreset, normalized, "normalized_preset")
    mode = _enum_value(ExecutionMode, execution_mode, "execution_mode")
    evidence = _enum_value(EvidenceMode, evidence_mode, "evidence_mode")
    context = _enum_value(ContextProfile, context_profile, "context_profile")
    claim = bool(sota_claim or requested_preset in SOTA_PRESET_ALIASES)
    if claim and preset not in {NormalizedPreset.TARGET_ID, NormalizedPreset.REPORT}:
        raise ValueError("sota_claim is only valid with target-id/report runs")
    return RunProfile(
        requested_preset=requested_preset,
        normalized_preset=preset,
        execution_mode=mode,
        analysis_profile=_analysis_profile(preset, mode),
        evidence_mode=evidence,
        context_profile=context,
        sota_claim=claim,
    )


def canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def schema_sha256() -> str:
    try:
        data = RUN_MANIFEST_SCHEMA_PATH.read_bytes()
    except OSError as exc:
        raise ValueError(
            f"run manifest schema could not be read: {RUN_MANIFEST_SCHEMA_PATH}: {exc}"
        ) from exc
    return hashlib.sha256(data).hexdigest()


def schema_sha256_v3() -> str:
    return _sha256_file(RUN_MANIFEST_SCHEMA_V3_PATH, "run manifest v3")


def target_fast_schema_sha256_v2() -> str:
    return _sha256_file(TARGET_FAST_SCHEMA_V2_PATH, "target-fast v2")


def build_run_manifest_v2(
    *,
    run_id: str,
    profile: RunProfile,
    config_payload: object,
    image_sha256: str | None = None,
    data_sha256: str | None = None,
    fast_artifact_id: str | None = None,
    physics_artifact_id: str | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "run_id": run_id,
        "run_profile": profile.to_dict(),
        "hashes": {
            "schema_sha256": schema_sha256(),
            "config_sha256": canonical_json_sha256(config_payload),
            "image_sha256": image_sha256,
            "data_sha256": data_sha256,
        },
        "artifact_ids": {
            "fast_artifact_id": fast_artifact_id,
            "physics_artifact_id": physics_artifact_id,
            "expected_fast": list(profile.expected_fast_artifact_ids()),
            "expected_physics": list(profile.expected_physics_artifact_ids()),
        },
    }
    validate_run_manifest_v2(manifest)
    return manifest


def build_run_manifest_v3(
    *,
    run_id: str,
    profile: RunProfile,
    config_payload: object,
    image_sha256: str | None = None,
    data_sha256: str | None = None,
    target_fast_sha256: str | None = None,
    top50_compat_sha256: str | None = None,
    physics_artifact_id: str | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_version": RUN_MANIFEST_SCHEMA_V3,
        "run_id": run_id,
        "run_profile": profile.to_dict(),
        "recipe_id": DAINA_STRUCTURAL_OVERLAY_RECIPE_ID,
        "hashes": {
            "schema_sha256": schema_sha256_v3(),
            "config_sha256": canonical_json_sha256(config_payload),
            "image_sha256": image_sha256,
            "data_sha256": data_sha256,
            "target_fast_schema_sha256": target_fast_schema_sha256_v2(),
        },
        "artifact_ids": {
            "fast_artifact_id": DAINA_STRUCTURAL_TARGETS_ARTIFACT_ID,
            "physics_artifact_id": physics_artifact_id,
            "expected_fast": [DAINA_STRUCTURAL_TARGETS_ARTIFACT_ID],
            "expected_physics": list(profile.expected_physics_artifact_ids()),
        },
        "target_fast": {
            "schema_version": TARGET_FAST_SCHEMA_V2,
            "artifact_id": DAINA_STRUCTURAL_TARGETS_ARTIFACT_ID,
            "path": DAINA_STRUCTURAL_TARGETS_PATH,
            "sha256": target_fast_sha256,
            "compatibility_projection": {
                "artifact_id": TOP50_COMPAT_ARTIFACT_ID,
                "path": TOP50_COMPAT_PATH,
                "sha256": top50_compat_sha256,
            },
        },
    }
    validate_run_manifest_v3(manifest)
    return manifest


def _require_sha256(value: object, label: str, *, nullable: bool) -> None:
    if nullable and value is None:
        return
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        suffix = " or null" if nullable else ""
        raise ValueError(f"{label} must be a lowercase SHA256{suffix}")


def _require_exact_keys(
    payload: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    observed = set(payload)
    if observed == expected:
        return
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    details: list[str] = []
    if missing:
        details.append("missing=" + ",".join(missing))
    if extra:
        details.append("extra=" + ",".join(extra))
    raise ValueError(f"{label} fields are invalid ({'; '.join(details)})")


def validate_run_manifest_v2(payload: Mapping[str, Any]) -> None:
    """Fail closed on the fields that define the v2 compatibility boundary."""

    _require_exact_keys(
        payload,
        {"schema_version", "run_id", "run_profile", "hashes", "artifact_ids"},
        "run manifest",
    )
    if payload.get("schema_version") != RUN_MANIFEST_SCHEMA:
        raise ValueError(f"schema_version must be {RUN_MANIFEST_SCHEMA}")
    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or not RUN_ID_RE.fullmatch(run_id):
        raise ValueError("run_id must satisfy the SkinScout run identifier contract")
    raw_profile = payload.get("run_profile")
    if not isinstance(raw_profile, Mapping):
        raise ValueError("run_profile must be an object")
    profile = resolve_run_profile(
        str(raw_profile.get("requested_preset", "")),
        str(raw_profile.get("execution_mode", "")),
        evidence_mode=str(raw_profile.get("evidence_mode", "")),
        context_profile=str(raw_profile.get("context_profile", "")),
        sota_claim=raw_profile.get("sota_claim") is True,
    )
    if dict(raw_profile) != profile.to_dict():
        raise ValueError("run_profile fields are not the canonical resolver output")
    hashes = payload.get("hashes")
    if not isinstance(hashes, Mapping):
        raise ValueError("hashes must be an object")
    _require_exact_keys(
        hashes,
        {"schema_sha256", "config_sha256", "image_sha256", "data_sha256"},
        "hashes",
    )
    _require_sha256(hashes.get("schema_sha256"), "schema_sha256", nullable=False)
    if hashes.get("schema_sha256") != schema_sha256():
        raise ValueError("schema_sha256 does not match the installed v2 schema")
    _require_sha256(hashes.get("config_sha256"), "config_sha256", nullable=False)
    _require_sha256(hashes.get("image_sha256"), "image_sha256", nullable=True)
    _require_sha256(hashes.get("data_sha256"), "data_sha256", nullable=True)
    artifact_ids = payload.get("artifact_ids")
    if not isinstance(artifact_ids, Mapping):
        raise ValueError("artifact_ids must be an object")
    _require_exact_keys(
        artifact_ids,
        {
            "fast_artifact_id",
            "physics_artifact_id",
            "expected_fast",
            "expected_physics",
        },
        "artifact_ids",
    )
    for key in ("fast_artifact_id", "physics_artifact_id"):
        value = artifact_ids.get(key)
        if value is not None and (
            not isinstance(value, str)
            or not value.strip()
            or value != value.strip()
        ):
            raise ValueError(f"{key} must be a non-empty string or null")
    if artifact_ids.get("expected_fast") != list(profile.expected_fast_artifact_ids()):
        raise ValueError("expected_fast does not match the canonical profile")
    if artifact_ids.get("expected_physics") != list(
        profile.expected_physics_artifact_ids()
    ):
        raise ValueError("expected_physics does not match the canonical profile")


def _validated_run_profile(raw_profile: object) -> RunProfile:
    if not isinstance(raw_profile, Mapping):
        raise ValueError("run_profile must be an object")
    profile = resolve_run_profile(
        str(raw_profile.get("requested_preset", "")),
        str(raw_profile.get("execution_mode", "")),
        evidence_mode=str(raw_profile.get("evidence_mode", "")),
        context_profile=str(raw_profile.get("context_profile", "")),
        sota_claim=raw_profile.get("sota_claim") is True,
    )
    if dict(raw_profile) != profile.to_dict():
        raise ValueError("run_profile fields are not the canonical resolver output")
    return profile


def validate_run_manifest_v3(payload: Mapping[str, Any]) -> None:
    """Fail closed on the v3 Daina structural overlay run-manifest boundary."""

    _require_exact_keys(
        payload,
        {
            "schema_version",
            "run_id",
            "run_profile",
            "recipe_id",
            "hashes",
            "artifact_ids",
            "target_fast",
        },
        "run manifest",
    )
    if payload.get("schema_version") != RUN_MANIFEST_SCHEMA_V3:
        raise ValueError(f"schema_version must be {RUN_MANIFEST_SCHEMA_V3}")
    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or not RUN_ID_RE.fullmatch(run_id):
        raise ValueError("run_id must satisfy the SkinScout run identifier contract")
    profile = _validated_run_profile(payload.get("run_profile"))
    if not profile.expected_fast_artifact_ids():
        raise ValueError("v3 Daina structural overlay requires a fast target profile")
    if payload.get("recipe_id") != DAINA_STRUCTURAL_OVERLAY_RECIPE_ID:
        raise ValueError(
            f"recipe_id must be {DAINA_STRUCTURAL_OVERLAY_RECIPE_ID}"
        )
    hashes = payload.get("hashes")
    if not isinstance(hashes, Mapping):
        raise ValueError("hashes must be an object")
    _require_exact_keys(
        hashes,
        {
            "schema_sha256",
            "config_sha256",
            "image_sha256",
            "data_sha256",
            "target_fast_schema_sha256",
        },
        "hashes",
    )
    _require_sha256(hashes.get("schema_sha256"), "schema_sha256", nullable=False)
    if hashes.get("schema_sha256") != schema_sha256_v3():
        raise ValueError("schema_sha256 does not match the installed v3 schema")
    _require_sha256(hashes.get("config_sha256"), "config_sha256", nullable=False)
    _require_sha256(hashes.get("image_sha256"), "image_sha256", nullable=True)
    _require_sha256(hashes.get("data_sha256"), "data_sha256", nullable=True)
    _require_sha256(
        hashes.get("target_fast_schema_sha256"),
        "target_fast_schema_sha256",
        nullable=False,
    )
    if hashes.get("target_fast_schema_sha256") != target_fast_schema_sha256_v2():
        raise ValueError(
            "target_fast_schema_sha256 does not match the installed v2 schema"
        )

    artifact_ids = payload.get("artifact_ids")
    if not isinstance(artifact_ids, Mapping):
        raise ValueError("artifact_ids must be an object")
    _require_exact_keys(
        artifact_ids,
        {
            "fast_artifact_id",
            "physics_artifact_id",
            "expected_fast",
            "expected_physics",
        },
        "artifact_ids",
    )
    if artifact_ids.get("fast_artifact_id") != DAINA_STRUCTURAL_TARGETS_ARTIFACT_ID:
        raise ValueError(
            f"fast_artifact_id must be {DAINA_STRUCTURAL_TARGETS_ARTIFACT_ID}"
        )
    physics_artifact_id = artifact_ids.get("physics_artifact_id")
    if physics_artifact_id is not None and (
        not isinstance(physics_artifact_id, str)
        or not physics_artifact_id.strip()
        or physics_artifact_id != physics_artifact_id.strip()
    ):
        raise ValueError("physics_artifact_id must be a non-empty string or null")
    if artifact_ids.get("expected_fast") != [DAINA_STRUCTURAL_TARGETS_ARTIFACT_ID]:
        raise ValueError("expected_fast does not match the canonical v3 profile")
    if artifact_ids.get("expected_physics") != list(
        profile.expected_physics_artifact_ids()
    ):
        raise ValueError("expected_physics does not match the canonical profile")

    target_fast = payload.get("target_fast")
    if not isinstance(target_fast, Mapping):
        raise ValueError("target_fast must be an object")
    _require_exact_keys(
        target_fast,
        {
            "schema_version",
            "artifact_id",
            "path",
            "sha256",
            "compatibility_projection",
        },
        "target_fast",
    )
    if target_fast.get("schema_version") != TARGET_FAST_SCHEMA_V2:
        raise ValueError(f"target_fast schema_version must be {TARGET_FAST_SCHEMA_V2}")
    if target_fast.get("artifact_id") != DAINA_STRUCTURAL_TARGETS_ARTIFACT_ID:
        raise ValueError(
            f"target_fast artifact_id must be {DAINA_STRUCTURAL_TARGETS_ARTIFACT_ID}"
        )
    if target_fast.get("path") != DAINA_STRUCTURAL_TARGETS_PATH:
        raise ValueError(f"target_fast path must be {DAINA_STRUCTURAL_TARGETS_PATH}")
    _require_sha256(target_fast.get("sha256"), "target_fast sha256", nullable=True)
    projection = target_fast.get("compatibility_projection")
    if not isinstance(projection, Mapping):
        raise ValueError("compatibility_projection must be an object")
    _require_exact_keys(
        projection,
        {"artifact_id", "path", "sha256"},
        "compatibility_projection",
    )
    if projection.get("artifact_id") != TOP50_COMPAT_ARTIFACT_ID:
        raise ValueError(
            f"compatibility_projection artifact_id must be {TOP50_COMPAT_ARTIFACT_ID}"
        )
    if projection.get("path") != TOP50_COMPAT_PATH:
        raise ValueError(f"compatibility_projection path must be {TOP50_COMPAT_PATH}")
    _require_sha256(
        projection.get("sha256"),
        "compatibility_projection sha256",
        nullable=True,
    )


def validate_run_manifest(payload: Mapping[str, Any]) -> None:
    """Dispatch only known run-manifest versions and fail closed otherwise."""

    schema = payload.get("schema_version")
    if schema == RUN_MANIFEST_SCHEMA:
        validate_run_manifest_v2(payload)
        return
    if schema == RUN_MANIFEST_SCHEMA_V3:
        validate_run_manifest_v3(payload)
        return
    raise ValueError(f"unsupported run manifest schema: {schema!r}")


def validate_target_fast_v2_record(payload: Mapping[str, Any]) -> None:
    """Validate one deserialized row of the canonical target-fast v2 CSV."""

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
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(
            "target-fast record fields are invalid (missing=" + ",".join(missing) + ")"
        )
    forbidden = sorted(
        key for key in ("probability", "calibrated_probability") if key in payload
    )
    if forbidden:
        raise ValueError(
            "target-fast record must not expose calibrated probability fields: "
            + ",".join(forbidden)
        )
    if payload.get("schema_version") != TARGET_FAST_SCHEMA_V2:
        raise ValueError(f"schema_version must be {TARGET_FAST_SCHEMA_V2}")
    if payload.get("recipe_id") != DAINA_STRUCTURAL_OVERLAY_RECIPE_ID:
        raise ValueError(f"recipe_id must be {DAINA_STRUCTURAL_OVERLAY_RECIPE_ID}")
    target_id = payload.get("target_id")
    if (
        not isinstance(target_id, str)
        or not target_id.strip()
        or target_id != target_id.strip()
    ):
        raise ValueError("target_id must be a non-empty string")
    target_name = payload.get("target_name")
    if target_name is not None and not isinstance(target_name, str):
        raise ValueError("target_name must be a string or null")
    if (
        not isinstance(payload.get("daina_rank"), int)
        or isinstance(payload.get("daina_rank"), bool)
        or payload["daina_rank"] < 1
    ):
        raise ValueError("daina_rank must be an integer >= 1")
    if not isinstance(payload.get("daina_score"), (int, float)) or isinstance(
        payload.get("daina_score"), bool
    ):
        raise ValueError("daina_score must be numeric")
    if not math.isfinite(payload["daina_score"]):
        raise ValueError("daina_score must be finite")
    if payload.get("daina_score_is_probability") is not False:
        raise ValueError("daina_score_is_probability must be false")
    if payload.get("structural_status") not in TARGET_FAST_STRUCTURAL_STATUSES:
        raise ValueError("structural_status is not supported")
    expected_supported = payload.get("structural_status") == "structure_supported"
    if payload.get("structure_supported") is not expected_supported:
        raise ValueError("structure_supported must agree with structural_status")
    for label in (
        "daina_primary_json",
        "structure_annotation_json",
        "skin_kg_annotation_json",
        "experimental_annotation_json",
    ):
        value = payload.get(label)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must be a non-empty JSON string")
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} must contain valid JSON") from exc
        if not isinstance(decoded, Mapping):
            raise ValueError(f"{label} must encode an object")
    primary = json.loads(str(payload["daina_primary_json"]))
    if primary != {
        "rank": payload["daina_rank"],
        "score": payload["daina_score"],
        "score_is_probability": False,
    }:
        raise ValueError("daina_primary_json must agree with Daina scalar fields")
    final_score = payload.get("final_score")
    if not isinstance(final_score, (int, float)) or isinstance(final_score, bool):
        raise ValueError("final_score must be numeric")
    if float(final_score) != float(payload["daina_score"]):
        raise ValueError("final_score must preserve the Daina primary score")
    source_count = payload.get("source_count")
    if (
        not isinstance(source_count, int)
        or isinstance(source_count, bool)
        or source_count < 1
    ):
        raise ValueError("source_count must be an integer >= 1")
    sources = payload.get("sources")
    if not isinstance(sources, str) or not sources.strip():
        raise ValueError("sources must be a non-empty string")


def migrate_run_manifest_v2(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Upgrade a v1 summary or normalize v2 without ever downgrading newer data."""

    schema = payload.get("schema_version")
    if schema == RUN_MANIFEST_SCHEMA:
        migrated = json.loads(json.dumps(payload))
        validate_run_manifest_v2(migrated)
        return migrated
    if isinstance(schema, str) and schema.startswith("skinscout.run_manifest.v"):
        raise ValueError(f"unsupported forward run manifest schema: {schema}")
    if schema not in {1, "skinscout.run_summary.v1"}:
        raise ValueError("only run_summary.v1 payloads can migrate to run_manifest.v2")
    requested = str(payload.get("requested_preset") or payload.get("preset") or "")
    profile = resolve_run_profile(
        requested,
        str(payload.get("mode") or ""),
        evidence_mode=str(payload.get("evidence_mode") or EvidenceMode.EVIDENCE.value),
        context_profile=str(payload.get("context_profile") or ContextProfile.AUTO.value),
        sota_claim=payload.get("sota_claim") is True,
    )
    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("legacy payload run_id must be a non-empty string")
    return build_run_manifest_v2(
        run_id=run_id,
        profile=profile,
        config_payload={"legacy_v1": dict(payload), "run_profile": profile.to_dict()},
    )
