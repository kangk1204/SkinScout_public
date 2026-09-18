#!/usr/bin/env python3
"""stage3_gnina_rescore.py — GNINA CNN rescoring for the top-X% targets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd

try:
    from .docking_pose_manifest import check_pose_parity, read_pose_manifest
except ImportError:  # Executed as a standalone script.
    from docking_pose_manifest import (  # type: ignore[no-redef]
        check_pose_parity,
        read_pose_manifest,
    )

LOG = logging.getLogger("stage3.gnina")
CNN_RE = re.compile(r"CNNaffinity:\s*(\S+)")

GNINA_STATUS_SCHEMA_VERSION = "skinscout.gnina-pose-status.v1"


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
        f"GNINA top-target CSV column '{column}' contains {detail} at row "
        f"index(es) {shown}{suffix}"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _read_top_csv(path: Path, *, allow_empty: bool = False) -> pd.DataFrame:
    if not _nonempty(path):
        raise SystemExit(
            f"GNINA top-target CSV is required and must be non-empty: {path}"
        )
    top = pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",")
    if "target_id" not in top.columns:
        raise SystemExit(
            f"GNINA top-target CSV missing required column 'target_id': {path}"
        )
    if top.empty and not allow_empty:
        raise SystemExit(f"GNINA top-target CSV contains no rows: {path}")
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
            "GNINA top-target CSV contains duplicate target_id values: "
            f"{shown}{suffix}"
        )
    return top


def gnina_score(
    receptor: Path,
    ligand_sdf: Path,
    *,
    use_gpu: bool = False,
) -> float | None:
    if not _nonempty(receptor) or not _nonempty(ligand_sdf):
        return None
    if not shutil.which("gnina"):
        return None
    cmd = ["gnina", "--score_only", "--cnn_scoring", "rescore"]
    if not use_gpu:
        cmd.append("--no_gpu")
    cmd.extend(["-r", str(receptor), "-l", str(ligand_sdf)])
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        LOG.warning(
            "GNINA failed for %s: %s",
            receptor.name,
            (res.stderr or res.stdout).strip()[-500:],
        )
        return None
    m = CNN_RE.search(res.stdout)
    if not m:
        return None
    value = m.group(1)
    if value.strip().lower() in {"true", "false"}:
        raise SystemExit(f"GNINA output for {receptor.name} must be numeric")
    try:
        score = float(value)
    except ValueError as exc:
        raise SystemExit(
            f"GNINA output for {receptor.name} must be numeric"
        ) from exc
    if not math.isfinite(score):
        raise SystemExit(f"GNINA output for {receptor.name} must be finite")
    return score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-csv", required=True, type=Path)
    parser.add_argument("--ligand-sdf", type=Path)
    parser.add_argument("--pose-manifest", type=Path)
    parser.add_argument("--pose-dir", type=Path)
    parser.add_argument("--clean-dir", required=True, type=Path)
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GNINA CUDA scoring. Legacy callers retain the CPU-only default.",
    )
    parser.add_argument(
        "--allow-partial-structure",
        action="store_true",
        help="Keep per-target GNINA failures as status records.",
    )
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
    parser.add_argument("--out-scores", required=True, type=Path)
    parser.add_argument("--out-status-manifest", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    outputs = [args.out_scores]
    if args.out_status_manifest is not None:
        outputs.append(args.out_status_manifest)
    _remove_outputs(*outputs)
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
    if args.allow_partial_structure and not pose_mode:
        raise SystemExit("--allow-partial-structure requires pose-manifest mode")
    if args.allow_unscored_poses and not pose_mode:
        raise SystemExit("--allow-unscored-poses requires pose-manifest mode")
    if args.allow_unposed_targets and not pose_mode:
        raise SystemExit("--allow-unposed-targets requires pose-manifest mode")

    top = _read_top_csv(args.top_csv, allow_empty=args.allow_partial_structure)
    poses = (
        read_pose_manifest(args.pose_manifest, args.pose_dir)
        if pose_mode and args.pose_manifest is not None and args.pose_dir is not None
        else {}
    )
    top_ids = top["target_id"].astype(str).tolist()
    if pose_mode:
        unposed = set(
            check_pose_parity(
                "GNINA",
                top_ids,
                poses,
                allow_unscored=args.allow_unscored_poses,
                allow_unposed=args.allow_unposed_targets,
            )
        )
    if not top.empty and not shutil.which("gnina"):
        raise SystemExit("gnina is not available on PATH")
    if not pose_mode and not top.empty and args.ligand_sdf is not None and not _nonempty(args.ligand_sdf):
        raise SystemExit(f"Ligand SDF is missing or empty for GNINA: {args.ligand_sdf}")
    n_written = 0
    status_records: list[dict[str, object]] = []
    tmp_scores = _tmp_output(args.out_scores)
    try:
        with tmp_scores.open("w", newline="") as fh:
            w = csv.writer(fh, delimiter="\t")
            if pose_mode:
                w.writerow(
                    [
                        "target_id",
                        "cnn_affinity",
                        "score",
                        "pose_file",
                        "pose_sha256",
                        "scored_actual_docked_pose",
                        "gpu_enabled",
                    ]
                )
            else:
                w.writerow(["target_id", "cnn_affinity", "score"])
            for uid in top["target_id"].astype(str):
                receptor = args.clean_dir / f"{uid}_clean.pdb"
                if not _nonempty(receptor):
                    status_records.append(
                        {
                            "target_id": uid,
                            "status": "structure_failed_gnina",
                            "detail": "clean receptor structure is missing or empty",
                        }
                    )
                    continue
                if pose_mode and uid in unposed:
                    status_records.append({
                        "target_id": uid,
                        "status": "structure_failed_gnina",
                        "detail": "target was never docked, so it has no pose to rescore",
                    })
                    continue
                if pose_mode:
                    pose_record = poses[uid]
                    ligand = Path(str(pose_record["pose_path"]))
                    pose_file = str(pose_record["pose_file"])
                    pose_sha = str(pose_record["pose_sha256"])
                else:
                    assert args.ligand_sdf is not None
                    ligand = args.ligand_sdf
                    pose_file = ""
                    pose_sha = ""
                cnn = gnina_score(receptor, ligand, use_gpu=args.use_gpu)
                if cnn is None:
                    status_records.append(
                        {
                            "target_id": uid,
                            "status": "structure_failed_gnina",
                            "detail": "GNINA returned no usable CNN affinity",
                            "pose_file": pose_file,
                            "pose_sha256": pose_sha,
                        }
                    )
                    continue
                if pose_mode:
                    w.writerow(
                        [
                            uid,
                            f"{cnn:.4f}",
                            f"{cnn:.4f}",
                            pose_file,
                            pose_sha,
                            "true",
                            str(args.use_gpu).lower(),
                        ]
                    )
                else:
                    w.writerow([uid, f"{cnn:.4f}", f"{cnn:.4f}"])
                status_records.append(
                    {
                        "target_id": uid,
                        "status": "structure_supported",
                        "detail": "GNINA scored the exported AutoDock pose",
                        "cnn_affinity": round(cnn, 4),
                        "pose_file": pose_file,
                        "pose_sha256": pose_sha,
                    }
                )
                n_written += 1
    except BaseException:
        tmp_scores.unlink(missing_ok=True)
        raise
    if not top.empty and n_written == 0 and not args.allow_partial_structure:
        tmp_scores.unlink(missing_ok=True)
        raise SystemExit("GNINA produced no usable rescores")
    tmp_scores.replace(args.out_scores)
    if args.out_status_manifest is not None:
        _write_json_atomic(
            args.out_status_manifest,
            {
                "schema_version": GNINA_STATUS_SCHEMA_VERSION,
                "pose_manifest": str(args.pose_manifest),
                "target_count": len(status_records),
                "supported_count": n_written,
                "gpu_enabled": args.use_gpu,
                "targets": status_records,
            },
        )
    LOG.info("GNINA rescores → %s", args.out_scores)


if __name__ == "__main__":
    main()
