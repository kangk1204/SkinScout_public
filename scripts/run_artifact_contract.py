#!/usr/bin/env python3
"""Single source of truth for SkinScout run artifact contracts.

The summary writer (``summarize_run_outputs.py``) and the goal-contract
verifier (``verify_goal_contract.py``) used to carry independent copies of the
fast-mode artifact and screening-key tables. The copies drifted: the verifier
still required the old PSICHIC/DTI funnel while the writer had moved to the
Daina structural overlay, so every current fast run failed re-verification
with "missing artifacts" even when its own summary was written by the current
writer.

This module owns the tables once. The writer publishes the paths, the verifier
requires the same paths, and a shared-key equality test pins them together.
Runs that predate the current Daina structural contract are not silently
accepted; they resolve to :data:`CONTRACT_LEGACY` and are labelled
``legacy/not_currently_verified``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


RUN_MANIFEST_SCHEMA_V2 = "skinscout.run_manifest.v2"
RUN_MANIFEST_SCHEMA_V3 = "skinscout.run_manifest.v3"

CONTRACT_CURRENT = "current"
CONTRACT_LEGACY = "legacy/not_currently_verified"

DAINA_STRUCTURAL_TARGETS_PATH = (
    "03_targets/mode_fast/daina_structural_targets.csv"
)
DAINA_BAND_RERANKED_TARGETS_PATH = (
    "03_targets/mode_fast/daina_band_reranked_targets.csv"
)
FAST_TOP50_PATH = "03_targets/mode_fast/top50.csv"
FAST_TOP50_BAND_PATH = "03_targets/mode_fast/top50_band_reranked.csv"

# Artifact path maps used by the summary writer. The keys are stable names the
# summaries and downstream readers already know.
TARGET_FAST_ARTIFACTS_CURRENT: dict[str, str] = {
    "target_fast_daina_zoete": "03_targets/mode_fast/daina_zoete_proteome.tsv",
    "target_fast_daina_selected": "03_targets/mode_fast/daina_top256.csv",
    "target_fast_dti_rrf": "03_targets/mode_fast/dti_rrf_top25pct.csv",
    "target_fast_autogrid_manifest": "03_targets/mode_fast/autogrid_map_manifest.json",
    "target_fast_autodock": "03_targets/mode_fast/autodock_top5k.tsv",
    "target_fast_gnina_pose": "03_targets/mode_fast/gnina_pose_rescores.tsv",
    "target_fast_daina_structural_targets": DAINA_STRUCTURAL_TARGETS_PATH,
    "target_fast_rerank_consensus": FAST_TOP50_PATH,
    "target_fast_band_reranked": FAST_TOP50_BAND_PATH,
    "target_fast_band_reranked_full": DAINA_BAND_RERANKED_TARGETS_PATH,
}

TARGET_FAST_ARTIFACTS_LEGACY: dict[str, str] = {
    "target_fast_psichic": "03_targets/mode_fast/psichic_proteome.tsv",
    "target_fast_daina_zoete": "03_targets/mode_fast/daina_zoete_proteome.tsv",
    "target_fast_dti_rrf": "03_targets/mode_fast/dti_rrf_top25pct.csv",
    "target_fast_autodock": "03_targets/mode_fast/autodock_top5k.tsv",
    "target_fast_rerank_consensus": FAST_TOP50_PATH,
}

TARGET_COMPREHENSIVE_ARTIFACTS: dict[str, str] = {
    "target_comprehensive_ligand": "03_targets/mode_comprehensive/ligand.pdbqt",
    "target_comprehensive_autodock": "03_targets/mode_comprehensive/autodock_all_targets.tsv",
    "target_comprehensive_pre_rescore": "03_targets/mode_comprehensive/top_pct_pre_rescore.csv",
    "target_comprehensive_gnina": "03_targets/mode_comprehensive/gnina_rescores.tsv",
    "target_comprehensive_rtmscore": "03_targets/mode_comprehensive/rtmscore_rescores.tsv",
    "target_comprehensive_boltz2": "03_targets/mode_comprehensive/boltz2_affinity_top.tsv",
    "target_comprehensive_consensus": "03_targets/mode_comprehensive/top50_4way_consensus.csv",
}

# Screening-count sources. The current Daina contract replaced the PSICHIC/DTI
# contract; the legacy tuple is kept only so legacy runs can be labelled
# honestly instead of being measured against the wrong funnel.
FAST_SCREENING_ARTIFACTS_CURRENT: tuple[tuple[str, str], ...] = (
    ("daina_zoete_targets", "03_targets/mode_fast/daina_zoete_proteome.tsv"),
    ("daina_primary_candidates", "03_targets/mode_fast/daina_top256.csv"),
    ("autodock_rescored_targets", "03_targets/mode_fast/autodock_top5k.tsv"),
    ("gnina_pose_rescored_targets", "03_targets/mode_fast/gnina_pose_rescores.tsv"),
    ("daina_structural_targets", DAINA_STRUCTURAL_TARGETS_PATH),
    ("rerank_consensus_targets", FAST_TOP50_PATH),
    ("band_reranked_targets", FAST_TOP50_BAND_PATH),
)

FAST_SCREENING_ARTIFACTS_LEGACY: tuple[tuple[str, str], ...] = (
    ("psichic_proteome_targets", "03_targets/mode_fast/psichic_proteome.tsv"),
    ("daina_zoete_targets", "03_targets/mode_fast/daina_zoete_proteome.tsv"),
    ("dti_rrf_candidates", "03_targets/mode_fast/dti_rrf_top25pct.csv"),
    ("autodock_rescored_targets", "03_targets/mode_fast/autodock_top5k.tsv"),
    ("rerank_consensus_targets", FAST_TOP50_PATH),
)

COMPREHENSIVE_SCREENING_ARTIFACTS: tuple[tuple[str, str], ...] = (
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

FINAL_SCREENING_KEY = "skin_weighted_ranked_targets"

# Verified-artifact names required by the goal contract. The base set is
# written by the run verifier for every target-id/report preset. Fast runs
# additionally need the Daina band artifacts so the verified ranking and the
# ranking a summary reports cannot silently diverge.
REQUIRED_VERIFIED_ARTIFACTS: tuple[str, ...] = (
    "run_summary_json",
    "run_summary_md",
    "run_verification_log",
    "target_ranking",
)

FAST_BAND_VERIFIED_ARTIFACTS: dict[str, str] = {
    "target_fast_original_ranking": FAST_TOP50_PATH,
    "target_fast_band_ranking": FAST_TOP50_BAND_PATH,
    "target_fast_band_targets": DAINA_BAND_RERANKED_TARGETS_PATH,
}

DAINA_FAST_SCREENING_FUNNEL_EDGES: tuple[tuple[str, str], ...] = (
    ("daina_zoete_targets", "daina_primary_candidates"),
    ("daina_primary_candidates", "autodock_rescored_targets"),
    ("autodock_rescored_targets", "gnina_pose_rescored_targets"),
    ("daina_primary_candidates", "daina_structural_targets"),
    ("daina_structural_targets", "rerank_consensus_targets"),
    ("rerank_consensus_targets", "band_reranked_targets"),
)

LEGACY_FAST_SCREENING_FUNNEL_EDGES: tuple[tuple[str, str], ...] = (
    ("psichic_proteome_targets", "dti_rrf_candidates"),
    ("daina_zoete_targets", "dti_rrf_candidates"),
    ("dti_rrf_candidates", "autodock_rescored_targets"),
    ("autodock_rescored_targets", "rerank_consensus_targets"),
)

COMPREHENSIVE_SCREENING_FUNNEL_EDGES: tuple[tuple[str, str], ...] = (
    ("autodock_screened_targets", "pre_rescore_candidates"),
    ("pre_rescore_candidates", "gnina_rescored_targets"),
    ("pre_rescore_candidates", "rtmscore_rescored_targets"),
    ("pre_rescore_candidates", "boltz2_affinity_targets"),
    ("gnina_rescored_targets", "four_way_consensus_targets"),
    ("rtmscore_rescored_targets", "four_way_consensus_targets"),
    ("boltz2_affinity_targets", "four_way_consensus_targets"),
)

FAST_FINAL_FUNNEL_EDGE_CURRENT = ("band_reranked_targets", FINAL_SCREENING_KEY)
FAST_FINAL_FUNNEL_EDGE_LEGACY = ("rerank_consensus_targets", FINAL_SCREENING_KEY)
COMPREHENSIVE_FINAL_FUNNEL_EDGE = ("four_way_consensus_targets", FINAL_SCREENING_KEY)


class UnsupportedRunManifestError(ValueError):
    """The run declares a manifest version this contract does not know."""


def _nonempty(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def _read_manifest_schema(run_dir: Path) -> str | None:
    path = run_dir / "run_manifest.json"
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise UnsupportedRunManifestError(
            f"run manifest is empty or not a regular file: {path}"
        )
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UnsupportedRunManifestError(
            f"run manifest is unreadable: {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise UnsupportedRunManifestError(
            f"run manifest must be a JSON object: {path}"
        )
    schema = payload.get("schema_version")
    if not isinstance(schema, str) or not schema.strip():
        raise UnsupportedRunManifestError(
            f"run manifest has no schema_version: {path}"
        )
    return schema


def resolve_artifact_contract(run_dir: Path) -> str:
    """Classify a run directory as current or legacy/not-currently-verified.

    The manifest schema is authoritative when present: v3 is the Daina
    structural overlay, v2 is the replaced PSICHIC/DTI contract. Runs without a
    manifest are current only when the Daina structural artifact they claim is
    actually present; otherwise they are legacy and must be reported with the
    ``legacy/not_currently_verified`` label instead of being accepted.
    """
    schema = _read_manifest_schema(run_dir)
    if schema == RUN_MANIFEST_SCHEMA_V3:
        return CONTRACT_CURRENT
    if schema == RUN_MANIFEST_SCHEMA_V2:
        return CONTRACT_LEGACY
    if schema:
        raise UnsupportedRunManifestError(
            f"unsupported run manifest schema: {schema!r}"
        )
    if _nonempty(run_dir / DAINA_STRUCTURAL_TARGETS_PATH):
        return CONTRACT_CURRENT
    return CONTRACT_LEGACY


def daina_structural_contract_used(run_dir: Path) -> bool:
    return resolve_artifact_contract(run_dir) == CONTRACT_CURRENT


def target_artifact_map(mode: str, contract: str) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    if mode in {"fast", "both"}:
        artifacts.update(
            TARGET_FAST_ARTIFACTS_CURRENT
            if contract == CONTRACT_CURRENT
            else TARGET_FAST_ARTIFACTS_LEGACY
        )
    if mode in {"comprehensive", "both"}:
        artifacts.update(TARGET_COMPREHENSIVE_ARTIFACTS)
    return artifacts


def screening_count_artifacts(
    mode: str, contract: str
) -> tuple[tuple[str, str], ...]:
    artifacts: list[tuple[str, str]] = []
    if mode in {"fast", "both"}:
        artifacts.extend(
            FAST_SCREENING_ARTIFACTS_CURRENT
            if contract == CONTRACT_CURRENT
            else FAST_SCREENING_ARTIFACTS_LEGACY
        )
    if mode in {"comprehensive", "both"}:
        artifacts.extend(COMPREHENSIVE_SCREENING_ARTIFACTS)
    return tuple(artifacts)


def required_screening_keys(mode: str, contract: str) -> frozenset[str]:
    keys = {
        name
        for name, _ in screening_count_artifacts(mode, contract)
    }
    keys.add(FINAL_SCREENING_KEY)
    return frozenset(keys)


def screening_funnel_edges(
    mode: str, contract: str
) -> tuple[tuple[str, str], ...]:
    edges: list[tuple[str, str]] = []
    if mode in {"fast", "both"}:
        edges.extend(
            DAINA_FAST_SCREENING_FUNNEL_EDGES
            if contract == CONTRACT_CURRENT
            else LEGACY_FAST_SCREENING_FUNNEL_EDGES
        )
    if mode in {"comprehensive", "both"}:
        edges.extend(COMPREHENSIVE_SCREENING_FUNNEL_EDGES)
    if mode == "fast":
        edges.append(
            FAST_FINAL_FUNNEL_EDGE_CURRENT
            if contract == CONTRACT_CURRENT
            else FAST_FINAL_FUNNEL_EDGE_LEGACY
        )
    if mode in {"comprehensive", "both"}:
        edges.append(COMPREHENSIVE_FINAL_FUNNEL_EDGE)
    return tuple(edges)


def required_verified_artifact_paths(
    mode: str,
    contract: str,
    *,
    target_ranking_relative: str,
) -> dict[str, str]:
    """Name → run-relative path for every artifact the contract requires.

    ``target_ranking`` follows the summary's own binding so a summary that
    reports one ranking but fingerprints another is caught rather than merged.
    """
    required = {
        "run_summary_json": "run_summary.json",
        "run_summary_md": "run_summary.md",
        "run_verification_log": "run_verification.log",
        "target_ranking": target_ranking_relative,
    }
    if mode in {"fast", "both"} and contract == CONTRACT_CURRENT:
        required.update(FAST_BAND_VERIFIED_ARTIFACTS)
    return required


def current_fast_screening_keys() -> frozenset[str]:
    return required_screening_keys("fast", CONTRACT_CURRENT)


def legacy_fast_screening_keys() -> frozenset[str]:
    return required_screening_keys("fast", CONTRACT_LEGACY)
