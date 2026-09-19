"""Shared artifact-contract equality between the writer and the verifier.

The F14 regression was that ``summarize_run_outputs.py`` and
``verify_goal_contract.py`` carried separate copies of the fast screening-key
table. This test pins both to the single shared contract module, so the writer
cannot publish a Daina artifact set that the verifier still reads as
PSICHIC/DTI (or the reverse).
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import run_artifact_contract as contract  # noqa: E402
import summarize_run_outputs as summary  # noqa: E402
import verify_goal_contract as verifier  # noqa: E402


def test_fast_screening_keys_are_identical_between_verifier_and_summarizer() -> None:
    shared = contract.required_screening_keys("fast", contract.CONTRACT_CURRENT)
    assert verifier.FAST_REQUIRED_SCREENING_KEYS == set(shared)
    assert {
        name for name, _ in summary.FAST_SCREENING_COUNT_ARTIFACTS
    } | {contract.FINAL_SCREENING_KEY} == set(shared)


def test_screening_key_tables_are_the_same_objects() -> None:
    assert summary.FAST_SCREENING_COUNT_ARTIFACTS is (
        contract.FAST_SCREENING_ARTIFACTS_CURRENT
    )
    assert summary.LEGACY_FAST_SCREENING_COUNT_ARTIFACTS is (
        contract.FAST_SCREENING_ARTIFACTS_LEGACY
    )
    assert summary.COMPREHENSIVE_SCREENING_COUNT_ARTIFACTS is (
        contract.COMPREHENSIVE_SCREENING_ARTIFACTS
    )


def test_comprehensive_screening_keys_are_identical() -> None:
    shared = contract.required_screening_keys(
        "comprehensive", contract.CONTRACT_CURRENT
    )
    assert verifier.COMPREHENSIVE_REQUIRED_SCREENING_KEYS == set(shared)
    assert {
        name for name, _ in summary.COMPREHENSIVE_SCREENING_COUNT_ARTIFACTS
    } | {contract.FINAL_SCREENING_KEY} == set(shared)


def test_legacy_fast_keys_are_not_the_current_contract() -> None:
    legacy = contract.required_screening_keys("fast", contract.CONTRACT_LEGACY)
    current = contract.required_screening_keys("fast", contract.CONTRACT_CURRENT)
    assert verifier.LEGACY_FAST_REQUIRED_SCREENING_KEYS == set(legacy)
    assert legacy != current
    assert "psichic_proteome_targets" in legacy
    assert "psichic_proteome_targets" not in current


def test_writer_published_paths_cover_every_required_verified_artifact() -> None:
    for mode in ("fast", "both"):
        required = contract.required_verified_artifact_paths(
            mode,
            contract.CONTRACT_CURRENT,
            target_ranking_relative="03_targets/ranked_targets_v3_with_efficacy.csv",
        )
        assert contract.FAST_BAND_VERIFIED_ARTIFACTS.keys() <= required.keys()
    published = summary.target_artifact_map("fast", contract.CONTRACT_CURRENT)
    for name, path in contract.FAST_BAND_VERIFIED_ARTIFACTS.items():
        assert path in published.values(), name


def test_contract_module_uses_the_canonical_manifest_schema_versions() -> None:
    from skinscout.contracts.run_profile import (
        RUN_MANIFEST_SCHEMA,
        RUN_MANIFEST_SCHEMA_V3,
    )

    assert contract.RUN_MANIFEST_SCHEMA_V2 == RUN_MANIFEST_SCHEMA
    assert contract.RUN_MANIFEST_SCHEMA_V3 == RUN_MANIFEST_SCHEMA_V3
    assert verifier.RUN_MANIFEST_SCHEMA_V2 == RUN_MANIFEST_SCHEMA
    assert verifier.RUN_MANIFEST_SCHEMA_V3 == RUN_MANIFEST_SCHEMA_V3
