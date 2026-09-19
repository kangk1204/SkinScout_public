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


class RTMScoreStructureError(RuntimeError):
    """A single structure could not be scored; the rest of the batch continues.

    Only expected input problems belong here: a missing/empty/malformed
    receptor or ligand. `_is_structure_input_error` also recognises the same
    class name from the installed rtmscore adapter, so the adapter can raise
    its own subclass without importing this module.
    """

    rtmscore_structure_error = True


class RTMScoreBatchError(RuntimeError):
    """Model init, device, shape or programming failure: stop the whole batch.

    Returning None per target for these hid an unusable environment behind a
    "partial coverage" output. The batch must fail instead.
    """


STRUCTURE_INPUT_EXCEPTION_TYPES = (
    FileNotFoundError,
    IsADirectoryError,
    NotADirectoryError,
    PermissionError,
)


def _is_structure_input_error(exc: BaseException) -> bool:
    if isinstance(exc, STRUCTURE_INPUT_EXCEPTION_TYPES):
        return True
    if getattr(exc, "rtmscore_structure_error", False):
        return True
    return type(exc).__name__ == "RTMScoreStructureError"


def _rtmscore_backend_provenance(*, probe: bool) -> dict[str, object]:
    """Record which torch_scatter backend RTMScore uses and its evidence.

    The adapter owns the fallback; this asks it. An installed module without
    the provenance API (an upstream checkout or a test double) is recorded as
    missing rather than assumed. Provenance never fails scoring itself.
    """
    try:
        import rtmscore as rtmscore_module  # type: ignore[import-untyped]
    except ImportError as exc:
        return {"available": False, "reason": f"rtmscore not importable: {exc}"}
    payload: dict[str, object] = {"available": True}
    provenance = getattr(rtmscore_module, "backend_provenance", None)
    if callable(provenance):
        try:
            payload["provenance"] = {"recorded": True, **provenance()}
        except Exception as exc:  # noqa: BLE001 - diagnostic only
            payload["provenance"] = {
                "recorded": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    else:
        payload["provenance"] = {
            "recorded": False,
            "reason": "installed rtmscore module has no backend_provenance()",
        }
    if probe:
        probe_fn = getattr(rtmscore_module, "probe_scatter_parity", None)
        if callable(probe_fn):
            try:
                payload["parity"] = probe_fn()
            except Exception as exc:  # noqa: BLE001 - diagnostic only
                payload["parity"] = {
                    "status": "error",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
        else:
            payload["parity"] = {
                "status": "unavailable",
                "reason": "installed rtmscore module has no probe_scatter_parity()",
            }
    return payload


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


def _predict_score(
    receptor_pdb: Path, ligand_sdf: Path, *, device: str | None = None
) -> float:
    """Call the RTMScore API, separating structure failures from batch failures.

    Raises RTMScoreStructureError for an expected bad structure input and
    RTMScoreBatchError for anything that affects every target: a missing
    package, model init, device/shape, or programming errors.
    """
    if not _nonempty(receptor_pdb):
        raise RTMScoreStructureError(
            f"receptor structure is missing or empty: {receptor_pdb}"
        )
    if not _nonempty(ligand_sdf):
        raise RTMScoreStructureError(
            f"ligand/pose structure is missing or empty: {ligand_sdf}"
        )
    try:
        from rtmscore import predict_affinity   # type: ignore[import-untyped]
    except ImportError as exc:
        raise RTMScoreBatchError(
            "rtmscore package is not installed; the whole batch cannot be scored"
        ) from exc
    try:
        score = predict_affinity(
            str(receptor_pdb), str(ligand_sdf), **({"device": device} if device else {})
        )
    except Exception as exc:  # noqa: BLE001 - classified immediately
        if _is_structure_input_error(exc):
            raise RTMScoreStructureError(
                f"RTMScore rejected the structure for {receptor_pdb.name}: {exc}"
            ) from exc
        raise RTMScoreBatchError(
            f"RTMScore model/runtime failure for {receptor_pdb.name}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
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
    try:
        parsed = float(score)
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            f"RTMScore output for {receptor_pdb.name} must be numeric"
        ) from exc
    if not math.isfinite(parsed):
        raise SystemExit(
            f"RTMScore output for {receptor_pdb.name} must be finite"
        )
    return parsed


def rtmscore_with_reason(
    receptor_pdb: Path, ligand_sdf: Path, *, device: str | None = None
) -> tuple[float | None, str]:
    """Score one pose, returning (None, reason) for a per-structure failure.

    RTMScoreBatchError propagates: callers must not treat a broken environment
    as a target that simply failed to score.
    """
    try:
        return _predict_score(receptor_pdb, ligand_sdf, device=device), ""
    except RTMScoreStructureError as exc:
        LOG.warning("RTMScore skipped %s: %s", receptor_pdb.name, exc)
        return None, str(exc)


def rtmscore(
    receptor_pdb: Path, ligand_sdf: Path, *, device: str | None = None
) -> float | None:
    """Wrap the RTMScore python API. Returns score (higher=better) or None.

    None means this one structure could not be scored. Model/environment
    failures raise RTMScoreBatchError rather than masquerading as a miss.
    """
    score, _reason = rtmscore_with_reason(receptor_pdb, ligand_sdf, device=device)
    return score


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
                s, reason = rtmscore_with_reason(receptor, ligand, device=args.device)
                if s is None:
                    status_records.append({
                        "target_id": uid,
                        "status": "structure_failed_rtmscore",
                        "detail": reason or "RTMScore returned no usable score",
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
    except RTMScoreBatchError as exc:
        # A broken environment/model must not be recorded as targets that
        # simply failed to score; the whole batch stops and the tmp file goes.
        tmp_scores.unlink(missing_ok=True)
        raise SystemExit(f"RTMScore batch failed: {exc}") from exc
    except BaseException:
        tmp_scores.unlink(missing_ok=True)
        raise
    if not top.empty and n_written == 0:
        tmp_scores.unlink(missing_ok=True)
        raise SystemExit("RTMScore produced no usable rescores")
    tmp_scores.replace(args.out_scores)
    if args.out_status_manifest is not None:
        status_counts: dict[str, int] = {}
        for record in status_records:
            status = str(record["status"])
            status_counts[status] = status_counts.get(status, 0) + 1
        failure_reasons = [
            {
                "target_id": str(record["target_id"]),
                "status": str(record["status"]),
                "detail": str(record.get("detail") or ""),
            }
            for record in status_records
            if record["status"] != "structure_supported"
        ]
        partial_coverage = bool(failure_reasons) and n_written > 0
        args.out_status_manifest.write_text(
            json.dumps(
                {
                    "schema_version": RTM_STATUS_SCHEMA_VERSION,
                    "scored_actual_docked_pose": pose_mode,
                    "device_requested": args.device,
                    "target_count": len(status_records),
                    "input_target_count": len(top),
                    "scored_count": n_written,
                    "status_counts": status_counts,
                    "partial_coverage": partial_coverage,
                    "failure_reasons": failure_reasons,
                    "rtmscore_backend": _rtmscore_backend_provenance(probe=True),
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
