from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from skinscout.contracts.run_profile import (
    DAINA_STRUCTURAL_OVERLAY_RECIPE_ID,
    DAINA_STRUCTURAL_TARGETS_ARTIFACT_ID,
    DAINA_STRUCTURAL_TARGETS_PATH,
    RUN_MANIFEST_SCHEMA,
    RUN_MANIFEST_SCHEMA_V3,
    TARGET_FAST_SCHEMA_V2,
    TARGET_FAST_STRUCTURAL_STATUSES,
    TOP50_COMPAT_ARTIFACT_ID,
    TOP50_COMPAT_PATH,
    build_run_manifest_v2,
    build_run_manifest_v3,
    migrate_run_manifest_v2,
    resolve_run_profile,
    validate_run_manifest,
    validate_run_manifest_v2,
    validate_run_manifest_v3,
    validate_target_fast_v2_record,
)


ROOT = Path(__file__).resolve().parents[2]


def test_archived_v2_manifest_contract_remains_unchanged() -> None:
    profile = resolve_run_profile("target-id-sota", "fast", context_profile="acne")
    archived_v2 = build_run_manifest_v2(
        run_id="archived_v2_fast",
        profile=profile,
        config_payload={"archived": "v2"},
        image_sha256="a" * 64,
        data_sha256="b" * 64,
        fast_artifact_id="target_fast_rerank_consensus",
    )

    validate_run_manifest_v2(archived_v2)
    validate_run_manifest(archived_v2)
    assert migrate_run_manifest_v2(archived_v2) == archived_v2
    assert archived_v2["schema_version"] == RUN_MANIFEST_SCHEMA
    assert archived_v2["artifact_ids"]["expected_fast"] == [
        "target_fast_psichic",
        "target_fast_daina_zoete",
        "target_fast_dti_rrf",
        "target_fast_autodock",
        "target_fast_rerank_consensus",
    ]


def test_manifest_v3_declares_daina_structural_overlay_contract() -> None:
    manifest = build_run_manifest_v3(
        run_id="daina_overlay_fast",
        profile=resolve_run_profile("target-id", "fast"),
        config_payload={"recipe": DAINA_STRUCTURAL_OVERLAY_RECIPE_ID},
        target_fast_sha256="1" * 64,
        top50_compat_sha256="2" * 64,
    )

    validate_run_manifest_v3(manifest)
    validate_run_manifest(manifest)
    assert manifest["schema_version"] == RUN_MANIFEST_SCHEMA_V3
    assert manifest["recipe_id"] == DAINA_STRUCTURAL_OVERLAY_RECIPE_ID
    assert manifest["artifact_ids"]["fast_artifact_id"] == (
        DAINA_STRUCTURAL_TARGETS_ARTIFACT_ID
    )
    assert manifest["artifact_ids"]["expected_fast"] == [
        DAINA_STRUCTURAL_TARGETS_ARTIFACT_ID
    ]
    assert manifest["target_fast"] == {
        "schema_version": TARGET_FAST_SCHEMA_V2,
        "artifact_id": DAINA_STRUCTURAL_TARGETS_ARTIFACT_ID,
        "path": DAINA_STRUCTURAL_TARGETS_PATH,
        "sha256": "1" * 64,
        "compatibility_projection": {
            "artifact_id": TOP50_COMPAT_ARTIFACT_ID,
            "path": TOP50_COMPAT_PATH,
            "sha256": "2" * 64,
        },
    }


def test_manifest_v3_json_schema_matches_python_contract() -> None:
    schema = json.loads((ROOT / "schemas" / "run_manifest_v3.json").read_text())
    manifest = build_run_manifest_v3(
        run_id="schema_v3_case",
        profile=resolve_run_profile("report", "both"),
        config_payload={"mode": "both"},
    )

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(manifest)


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("recipe_id",), "other-recipe", "recipe_id must be"),
        (
            ("target_fast", "path"),
            "03_targets/mode_fast/top50.csv",
            "target_fast path must be",
        ),
        (
            ("target_fast", "compatibility_projection", "artifact_id"),
            DAINA_STRUCTURAL_TARGETS_ARTIFACT_ID,
            "compatibility_projection artifact_id must be",
        ),
    ],
)
def test_manifest_v3_fails_closed_on_noncanonical_fields(
    path: tuple[str, ...],
    value: str,
    match: str,
) -> None:
    manifest = build_run_manifest_v3(
        run_id="bad_v3_case",
        profile=resolve_run_profile("target-id", "fast"),
        config_payload={},
    )
    edited = copy.deepcopy(manifest)
    cursor = edited
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value

    with pytest.raises(ValueError, match=match):
        validate_run_manifest_v3(edited)


def test_manifest_v3_requires_fast_target_capable_profile() -> None:
    with pytest.raises(ValueError, match="requires a fast target profile"):
        build_run_manifest_v3(
            run_id="safety_v3_case",
            profile=resolve_run_profile("safety", "fast"),
            config_payload={},
        )


def _target_fast_record(
    *, status: str = "structure_supported", rank: int = 1, score: float = 0.0
) -> dict[str, object]:
    return {
        "schema_version": TARGET_FAST_SCHEMA_V2,
        "recipe_id": DAINA_STRUCTURAL_OVERLAY_RECIPE_ID,
        "target_id": "P12345",
        "target_name": "Example target",
        "daina_rank": rank,
        "daina_score": score,
        "daina_score_is_probability": False,
        "daina_primary_json": json.dumps(
            {"rank": rank, "score": score, "score_is_probability": False},
            sort_keys=True,
            separators=(",", ":"),
        ),
        "structural_status": status,
        "structure_supported": status == "structure_supported",
        "structure_annotation_json": json.dumps(
            {
                "status": status,
                "pocket_id": "AF-P12345-F1:pocket-1",
                "structure_source": "alphafold",
                "docking_score": -7.4,
                "gnina_score": -6.1,
            }
        ),
        "skin_kg_annotation_json": json.dumps(
            {
                "evidence_score": 3.5,
                "terms": ["barrier"],
                "source_ids": ["skin-kg:barrier"],
            }
        ),
        "experimental_annotation_json": json.dumps(
            {
                "assay_count": 2,
                "best_activity_value": 120.0,
                "best_activity_unit": "nM",
            }
        ),
        "final_score": score,
        "source_count": 4,
        "sources": "daina;autodock_gpu;gnina;skin_kg",
    }


def test_target_fast_v2_record_preserves_daina_primary_and_annotations() -> None:
    record = _target_fast_record(rank=7, score=-8.25)
    schema = json.loads((ROOT / "schemas" / "target_fast_v2.json").read_text())

    validate_target_fast_v2_record(record)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(record)


@pytest.mark.parametrize("status", sorted(TARGET_FAST_STRUCTURAL_STATUSES))
def test_target_fast_v2_accepts_every_structural_status(status: str) -> None:
    validate_target_fast_v2_record(_target_fast_record(status=status))


def test_target_fast_statuses_match_published_schema() -> None:
    schema = json.loads((ROOT / "schemas" / "target_fast_v2.json").read_text())
    assert TARGET_FAST_STRUCTURAL_STATUSES == frozenset(schema["properties"]["structural_status"]["enum"])


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_target_fast_rejects_nonfinite_scores(score: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        validate_target_fast_v2_record(_target_fast_record(score=score))


def test_target_fast_v2_rejects_probability_like_extensions() -> None:
    record = _target_fast_record(status="structural_unavailable_no_pocket")
    record["probability"] = 0.9

    with pytest.raises(ValueError, match="must not expose calibrated probability"):
        validate_target_fast_v2_record(record)
