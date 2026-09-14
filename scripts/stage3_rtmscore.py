#!/usr/bin/env python3
"""stage3_rtmscore.py — Residue-atom interaction-aware DL scoring (RTMScore)."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from pathlib import Path

import pandas as pd

try:
    from .docking_pose_manifest import check_pose_parity, read_pose_manifest
except ImportError:  # Executed as a standalone script.
    from docking_pose_manifest import (  # type: ignore[no-redef]
        check_pose_parity,
        read_pose_manifest,
    )

LOG = logging.getLogger("stage3.rtmscore")
RTM_STATUS_SCHEMA_VERSION = "skinscout.rtmscore-pose-status.v1"


def _add_repo_root_to_path() -> None:
    root = Path(__file__).resolve().parents[1]
    root_text = str(root)
    if root_text in sys.path:
        return
    pythonpath_entries = [
        path for path in os.environ.get("PYTHONPATH", "").split(os.pathsep) if path
    ]
    sys.path.insert(min(len(sys.path), 1 + len(pythonpath_entries)), root_text)


_add_repo_root_to_path()


def _remove_outputs(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def _nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _tmp_output(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_suffix(path.suffix + ".tmp")


def _invalid_row_message(column: str, rows: list[int], detail: str) -> str:
    shown = ", ".join(str(idx) for idx in rows[:10])
    suffix = "..." if len(rows) > 10 else ""
    return (
        f"RTMScore top-target CSV column '{column}' contains {detail} at row "
        f"index(es) {shown}{suffix}"
    )


def _read_top_csv(path: Path) -> pd.DataFrame:
    if not _nonempty(path):
        raise SystemExit(
            f"RTMScore top-target CSV is required and must be non-empty: {path}"
        )
    top = pd.read_csv(path)
    if "target_id" not in top.columns:
        raise SystemExit(
            f"RTMScore top-target CSV missing required column 'target_id': {path}"
        )
    if top.empty:
        raise SystemExit(f"RTMScore top-target CSV contains no rows: {path}")
    blank_targets = [
        int(idx)
        for idx, value in top["target_id"].items()
        if pd.isna(value) or not str(value).strip()
    ]
    if blank_targets:
        raise SystemExit(
            _invalid_row_message("target_id", blank_targets, "blank values")
        )
    top["target_id"] = top["target_id"].astype(str).str.strip()
    duplicate_targets = top["target_id"][top["target_id"].duplicated()].tolist()
    if duplicate_targets:
        shown = ", ".join(duplicate_targets[:10])
        suffix = "..." if len(duplicate_targets) > 10 else ""
        raise SystemExit(
            "RTMScore top-target CSV contains duplicate target_id values: "
            f"{shown}{suffix}"
        )
    return top


def rtmscore(
    receptor_pdb: Path, ligand_sdf: Path, *, device: str | None = None
) -> float | None:
    """Wrap the RTMScore python API. Returns score (higher=better) or None."""
    if not _nonempty(receptor_pdb) or not _nonempty(ligand_sdf):
        return None
    try:
        from rtmscore import predict_affinity   # type: ignore[import-untyped]
    except ImportError:
        LOG.warning("rtmscore not installed; returning None")
        return None
    try:
        score = predict_affinity(
            str(receptor_pdb), str(ligand_sdf), **({"device": device} if device else {})
        )
        if (
            isinstance(score, bool)
            or type(score).__name__ == "bool_"
            or (
                isinstance(score, str)
                and score.strip().lower() in {"true", "false"}
            )
        ):
            raise SystemExit(
                f"RTMScore output for {receptor_pdb.name} must be numeric"
            )
        parsed = float(score)
        if not math.isfinite(parsed):
            raise SystemExit(
                f"RTMScore output for {receptor_pdb.name} must be finite"
            )
        return parsed
    except Exception as exc:  # noqa: BLE001
        # A missing checkout, an unusable device and a genuine per-target
        # failure all looked identical at debug level, so an environment fault
        # surfaced only as "produced no usable rescores".
        LOG.warning("RTMScore failed for %s: %s", receptor_pdb.name, exc)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-csv", required=True, type=Path)
    parser.add_argument("--ligand-sdf", type=Path)
    parser.add_argument("--pose-manifest", type=Path)
    parser.add_argument("--pose-dir", type=Path)
    parser.add_argument(
        "--allow-unposed-targets",
        action="store_true",
        help=(
            "Record score-table targets that were never docked as skipped "
            "instead of failing, for callers whose top list also carries blind "
            "hits on receptors with no pocket."
        ),
    )
    parser.add_argument(
        "--allow-unscored-poses",
        action="store_true",
        help=(
            "Permit pose-manifest targets absent from the score table, for "
            "callers that dock a superset and rescore a subset."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        help=(
            "Force the RTMScore device. Without it a visible GPU is preferred "
            "and CPU is used when DGL cannot actually run on it."
        ),
    )
    parser.add_argument("--clean-dir", required=True, type=Path)
    parser.add_argument("--out-scores", required=True, type=Path)
    parser.add_argument("--out-status-manifest", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    pose_mode = args.pose_manifest is not None or args.pose_dir is not None
    if pose_mode:
        if args.pose_manifest is None or args.pose_dir is None:
            raise SystemExit("--pose-manifest and --pose-dir must be supplied together")
        if args.ligand_sdf is not None:
            raise SystemExit("--ligand-sdf cannot be combined with pose-manifest mode")
    elif args.ligand_sdf is None:
        raise SystemExit("Either --ligand-sdf or pose-manifest mode is required")
    if args.out_status_manifest is not None and not pose_mode:
        raise SystemExit("--out-status-manifest requires pose-manifest mode")
    if args.allow_unscored_poses and not pose_mode:
        raise SystemExit("--allow-unscored-poses requires pose-manifest mode")
    if args.allow_unposed_targets and not pose_mode:
        raise SystemExit("--allow-unposed-targets requires pose-manifest mode")

    _remove_outputs(args.out_scores, *( [args.out_status_manifest] if args.out_status_manifest else [] ))
    top = _read_top_csv(args.top_csv)
    if not top.empty:
        try:
            __import__("rtmscore")
        except ImportError as exc:
            raise SystemExit("rtmscore is not installed") from exc
    poses = (
        read_pose_manifest(args.pose_manifest, args.pose_dir)
        if pose_mode and args.pose_manifest is not None and args.pose_dir is not None
        else {}
    )
    if pose_mode:
        unposed = set(
            check_pose_parity(
                "RTMScore",
                top["target_id"].astype(str).tolist(),
                poses,
                allow_unscored=args.allow_unscored_poses,
                allow_unposed=args.allow_unposed_targets,
            )
        )
    if not pose_mode and not top.empty and not _nonempty(args.ligand_sdf):
        raise SystemExit(f"Ligand SDF is missing or empty for RTMScore: {args.ligand_sdf}")

    n_written = 0
    status_records: list[dict[str, object]] = []
    tmp_scores = _tmp_output(args.out_scores)
    try:
        with tmp_scores.open("w", newline="") as fh:
            w = csv.writer(fh, delimiter="\t")
            if pose_mode:
                w.writerow([
                    "target_id",
                    "rtm_score",
                    "score",
                    "pose_file",
                    "pose_sha256",
                    "scored_actual_docked_pose",
                ])
            else:
                w.writerow(["target_id", "rtm_score", "score"])
            for uid in top["target_id"].astype(str):
                receptor = args.clean_dir / f"{uid}_clean.pdb"
                if not _nonempty(receptor):
                    status_records.append({
                        "target_id": uid,
                        "status": "structure_failed_rtmscore",
                        "detail": "clean receptor structure is missing or empty",
                    })
                    continue
                if pose_mode and uid in unposed:
                    status_records.append({
                        "target_id": uid,
                        "status": "structure_failed_rtmscore",
                        "detail": "target was never docked, so it has no pose to rescore",
                    })
                    continue
                if pose_mode:
                    record = poses[uid]
                    ligand = Path(str(record["pose_path"]))
                    pose_file = str(record["pose_file"])
                    pose_sha = str(record["pose_sha256"])
                else:
                    assert args.ligand_sdf is not None
                    ligand = args.ligand_sdf
                    pose_file = ""
                    pose_sha = ""
                s = rtmscore(receptor, ligand, device=args.device)
                if s is None:
                    status_records.append({
                        "target_id": uid,
                        "status": "structure_failed_rtmscore",
                        "detail": "RTMScore returned no usable score",
                        "pose_file": pose_file,
                        "pose_sha256": pose_sha,
                    })
                    continue
                if pose_mode:
                    w.writerow([uid, f"{s:.4f}", f"{s:.4f}", pose_file, pose_sha, "true"])
                else:
                    w.writerow([uid, f"{s:.4f}", f"{s:.4f}"])
                status_records.append({
                    "target_id": uid,
                    "status": "structure_supported",
                    "pose_file": pose_file,
                    "pose_sha256": pose_sha,
                })
                n_written += 1
    except BaseException:
        tmp_scores.unlink(missing_ok=True)
        raise
    if not top.empty and n_written == 0:
        tmp_scores.unlink(missing_ok=True)
        raise SystemExit("RTMScore produced no usable rescores")
    tmp_scores.replace(args.out_scores)
    if args.out_status_manifest is not None:
        args.out_status_manifest.write_text(
            json.dumps(
                {
                    "schema_version": RTM_STATUS_SCHEMA_VERSION,
                    "scored_actual_docked_pose": pose_mode,
                    "device_requested": args.device,
                    "target_count": len(status_records),
                    "targets": status_records,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    LOG.info("RTMScores → %s (%d scored)", args.out_scores, n_written)


if __name__ == "__main__":
    main()
