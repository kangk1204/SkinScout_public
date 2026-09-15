from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from skinscout.contracts.run_profile import (
    CONTEXT_PROFILES,
    MODES,
    PRESETS,
    RUN_MANIFEST_SCHEMA,
    AnalysisProfile,
    build_run_manifest_v2,
    migrate_run_manifest_v2,
    resolve_run_profile,
    validate_run_manifest,
    validate_run_manifest_v2,
)


ROOT = Path(__file__).resolve().parents[2]


EXPECTED_ANALYSIS_PROFILES = {
    ("stage0", "fast"): "stage0",
    ("stage0", "comprehensive"): "stage0",
    ("stage0", "both"): "stage0",
    ("safety", "fast"): "safety",
    ("safety", "comprehensive"): "safety",
    ("safety", "both"): "safety",
    ("target-id", "fast"): "target_fast",
    ("target-id", "comprehensive"): "target_comprehensive",
    ("target-id", "both"): "target_compare",
    ("report", "fast"): "report_fast",
    ("report", "comprehensive"): "physics_full",
    ("report", "both"): "physics_full",
}


@pytest.mark.parametrize("requested_preset", sorted(PRESETS))
@pytest.mark.parametrize("execution_mode", sorted(MODES))
def test_run_profile_golden_vectors(
    requested_preset: str,
    execution_mode: str,
) -> None:
    profile = resolve_run_profile(requested_preset, execution_mode)
    normalized = {
        "target-id-sota": "target-id",
        "report-sota": "report",
    }.get(requested_preset, requested_preset)

    assert profile.requested_preset == requested_preset
    assert profile.normalized_preset.value == normalized
    assert profile.execution_mode.value == execution_mode
    assert profile.analysis_profile.value == EXPECTED_ANALYSIS_PROFILES[
        (normalized, execution_mode)
    ]
    assert profile.sota_claim is requested_preset.endswith("-sota")
    assert profile.legacy_fields() == {
        "preset": normalized,
        "requested_preset": requested_preset,
        "mode": execution_mode,
        "sota_claim": requested_preset.endswith("-sota"),
        "context_profile": "auto",
    }


@pytest.mark.parametrize("context_profile", sorted(CONTEXT_PROFILES))
def test_run_profile_preserves_every_context(context_profile: str) -> None:
    profile = resolve_run_profile(
        "target-id-sota",
        "fast",
        evidence_mode="discovery",
        context_profile=context_profile,
    )

    assert profile.context_profile.value == context_profile
    assert profile.evidence_mode.value == "discovery"
    assert profile.analysis_profile is AnalysisProfile.TARGET_FAST


@pytest.mark.parametrize("preset", ["stage0", "safety"])
def test_run_profile_rejects_sota_for_non_target_presets(preset: str) -> None:
    with pytest.raises(ValueError, match="only valid with target-id/report"):
        resolve_run_profile(preset, "fast", sota_claim=True)


def test_manifest_v2_is_canonical_and_hash_bound() -> None:
    profile = resolve_run_profile(
        "report-sota",
        "both",
        evidence_mode="evidence",
        context_profile="barrier",
    )
    manifest = build_run_manifest_v2(
        run_id="retinol_report",
        profile=profile,
        config_payload={"mode": "both", "seed": 42},
        image_sha256="1" * 64,
        data_sha256="2" * 64,
    )

    validate_run_manifest_v2(manifest)
    validate_run_manifest(manifest)
    assert manifest["schema_version"] == RUN_MANIFEST_SCHEMA
    assert manifest["run_profile"]["analysis_profile"] == "physics_full"
    assert manifest["artifact_ids"]["expected_fast"]
    assert manifest["artifact_ids"]["expected_physics"]
    assert len(manifest["hashes"]["config_sha256"]) == 64


def test_manifest_v2_migration_is_forward_only_and_idempotent() -> None:
    legacy = {
        "schema_version": "skinscout.run_summary.v1",
        "run_id": "niacinamide_fast",
        "preset": "target-id",
        "mode": "fast",
    }
    migrated = migrate_run_manifest_v2(legacy)

    assert migrate_run_manifest_v2(migrated) == migrated
    with pytest.raises(ValueError, match="unsupported forward"):
        migrate_run_manifest_v2(
            {"schema_version": "skinscout.run_manifest.v3"}
        )


def test_manifest_dispatch_rejects_unknown_versions() -> None:
    with pytest.raises(ValueError, match="unsupported run manifest schema"):
        validate_run_manifest({"schema_version": "skinscout.run_manifest.v99"})


def test_manifest_json_schema_matches_contract_enums() -> None:
    schema = json.loads((ROOT / "schemas" / "run_manifest_v2.json").read_text())

    Draft202012Validator.check_schema(schema)
    profile_properties = schema["properties"]["run_profile"]["properties"]
    assert set(profile_properties["requested_preset"]["enum"]) == PRESETS
    assert set(profile_properties["execution_mode"]["enum"]) == MODES
    assert set(profile_properties["context_profile"]["enum"]) == CONTEXT_PROFILES

    manifest = build_run_manifest_v2(
        run_id="schema_case",
        profile=resolve_run_profile("report", "fast"),
        config_payload={},
    )
    Draft202012Validator(schema).validate(manifest)


def test_manifest_rejects_noncanonical_alias_projection() -> None:
    profile = resolve_run_profile("target-id-sota", "fast")
    manifest = build_run_manifest_v2(
        run_id="alias_case",
        profile=profile,
        config_payload={},
    )
    manifest["run_profile"]["normalized_preset"] = "target-id-sota"

    with pytest.raises(ValueError, match="canonical resolver output"):
        validate_run_manifest_v2(manifest)


def test_manifest_rejects_unversioned_extra_fields() -> None:
    manifest = build_run_manifest_v2(
        run_id="extra_field_case",
        profile=resolve_run_profile("safety", "fast"),
        config_payload={},
    )
    manifest["unversioned_extension"] = True

    with pytest.raises(ValueError, match="extra=unversioned_extension"):
        validate_run_manifest_v2(manifest)
