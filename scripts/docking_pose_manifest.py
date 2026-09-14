#!/usr/bin/env python3
"""Shared reader for the docking pose manifest AutoDock writes.

Both rescorers must agree on what a pose is and on when a pose set is valid.
Keeping one copy is what stops them drifting into scoring different geometries
than the docking they are presented as rescoring.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

SCHEMA_VERSION = "skinscout.docking_pose_manifest.v1"
MANIFEST_NAME = "pose_manifest.json"

LOG = logging.getLogger("stage3.poses")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nonempty(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def read_pose_manifest(path: Path, pose_dir: Path) -> dict[str, dict[str, object]]:
    """Load and verify every pose the manifest declares.

    Each record is checked for a contained path, the expected filename, and a
    matching SHA-256, so a rescorer can never read a pose the docking run did
    not produce.
    """
    if not _nonempty(path):
        raise SystemExit(f"Docking pose manifest is missing or empty: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Docking pose manifest is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Docking pose manifest must contain an object: {path}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise SystemExit(
            "Docking pose manifest has unsupported schema_version: "
            f"{payload.get('schema_version')!r}"
        )
    records = payload.get("targets")
    if not isinstance(records, list):
        raise SystemExit("Docking pose manifest targets must be an array")
    if payload.get("target_count") != len(records):
        raise SystemExit("Docking pose manifest target_count does not match targets")
    output: dict[str, dict[str, object]] = {}
    pose_root = pose_dir.resolve()
    for index, value in enumerate(records):
        if not isinstance(value, dict):
            raise SystemExit(f"Docking pose manifest record {index} must be an object")
        target_id = str(value.get("target_id", "")).strip()
        if not target_id or target_id in output:
            raise SystemExit(
                f"Docking pose manifest has blank/duplicate target_id: {target_id!r}"
            )
        pose_name = str(value.get("pose_file", "")).strip()
        pose_path = (pose_root / pose_name).resolve()
        try:
            pose_path.relative_to(pose_root)
        except ValueError as exc:
            raise SystemExit(
                f"Docking pose manifest path escapes pose directory: {pose_name}"
            ) from exc
        if pose_path.name != f"{target_id}.sdf" or not _nonempty(pose_path):
            raise SystemExit(
                f"Docking pose manifest references invalid pose for {target_id}: {pose_path}"
            )
        expected_sha = str(value.get("pose_sha256", "")).strip()
        if not expected_sha or sha256_file(pose_path) != expected_sha:
            raise SystemExit(
                f"Docking pose SHA-256 mismatch for {target_id}: {pose_path}"
            )
        record = dict(value)
        record["pose_path"] = pose_path
        output[target_id] = record
    return output


def check_pose_parity(
    scorer: str,
    top_ids: list[str],
    poses: dict[str, dict[str, object]],
    *,
    allow_unscored: bool,
    allow_unposed: bool = False,
) -> list[str]:
    """Decide which scored targets have a verified pose, and refuse the rest.

    Scoring a target with someone else's geometry is never acceptable, so a
    target without a pose is fatal by default. Two callers legitimately differ
    from exact parity and each has to say so:

    * ``allow_unscored`` - the caller docked a superset and rescores a slice,
      as the comprehensive path does over a top percentage.
    * ``allow_unposed`` - the score table mixes docked targets with ones that
      were never docked at all. The comprehensive top list merges AutoDock
      results with DiffDock blind hits on no-pocket receptors, and those have
      no pose to rescore. They are returned so the caller can record them as
      skipped rather than score them wrongly.

    Returns the target ids that have no pose.
    """
    missing = sorted(set(top_ids) - set(poses))
    extra = sorted(set(poses) - set(top_ids))
    if missing and not allow_unposed:
        raise SystemExit(
            f"{scorer} score/pose manifest target parity failed: "
            f"missing_pose={missing[:10]}"
        )
    if missing:
        LOG.info(
            "%s skipping %d of %d scored targets with no docked pose",
            scorer,
            len(missing),
            len(top_ids),
        )
    if extra and not allow_unscored:
        raise SystemExit(
            f"{scorer} pose manifest contains targets absent from the score "
            f"table: extra_pose={extra[:10]} "
            "(pass --allow-unscored-poses when rescoring a subset)"
        )
    if extra:
        LOG.info(
            "%s rescoring %d of %d docked targets; %d poses left unscored",
            scorer,
            len(top_ids),
            len(poses),
            len(extra),
        )
    return missing
