#!/usr/bin/env python3
"""Score multiple compounds against one loaded ChEMBL Daina reference snapshot."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from stage3_daina_zoete import (
    EVIDENCE_MODES,
    QUALITY_POLICIES,
    SCORING_METHODS,
    _parse_cutoff,
    _sha256,
    _write_json_atomic,
    apply_quality_policy,
    apply_temporal_cutoff,
    load_activities,
    load_fingerprints,
    query_features,
    score_query_against_reference,
    validate_reference_overlap,
)


REQUIRED_BATCH_COLUMNS = {
    "case_id",
    "ligand_sdf",
    "out_scores",
    "out_metadata_json",
}


def _read_batch(path: Path) -> pd.DataFrame:
    try:
        batch = pd.read_csv(path)
    except Exception as exc:
        raise SystemExit(f"Unable to read Daina batch CSV: {path}") from exc
    missing = sorted(REQUIRED_BATCH_COLUMNS - set(batch.columns))
    if missing:
        raise SystemExit(f"Daina batch CSV missing required columns {missing}: {path}")
    if batch.empty:
        raise SystemExit(f"Daina batch CSV contains no cases: {path}")
    for column in sorted(REQUIRED_BATCH_COLUMNS):
        blanks = batch[column].isna() | batch[column].astype(str).str.strip().eq("")
        if blanks.any():
            raise SystemExit(
                f"Daina batch CSV column {column!r} contains blank row(s): "
                f"{batch.index[blanks].tolist()[:10]}"
            )
        batch[column] = batch[column].astype(str).str.strip()
    if batch["case_id"].duplicated().any():
        raise SystemExit("Daina batch CSV contains duplicate case_id values")
    output_paths = batch["out_scores"].tolist() + batch["out_metadata_json"].tolist()
    if len(output_paths) != len(set(output_paths)):
        raise SystemExit("Daina batch CSV contains duplicate output paths")
    return batch


def _staging_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.daina-batch-staging")


def _cleanup(paths: list[Path]) -> None:
    for path in paths:
        path.unlink(missing_ok=True)
        _staging_path(path).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-csv", required=True, type=Path)
    parser.add_argument("--chembl-fp", required=True, type=Path)
    parser.add_argument("--out-summary-json", required=True, type=Path)
    parser.add_argument("--evidence-mode", choices=EVIDENCE_MODES, default="retrieval")
    parser.add_argument("--evidence-snapshot-id")
    parser.add_argument("--cutoff-date")
    parser.add_argument("--exclude-reference-similarity", type=float, default=0.85)
    parser.add_argument("--quality-policy", choices=QUALITY_POLICIES, default="legacy")
    parser.add_argument("--scoring-method", choices=SCORING_METHODS, default="max-similarity")
    args = parser.parse_args()

    batch = _read_batch(args.batch_csv)
    final_paths = [
        *(Path(value) for value in batch["out_scores"]),
        *(Path(value) for value in batch["out_metadata_json"]),
        args.out_summary_json,
    ]
    _cleanup(final_paths)
    if not math.isfinite(args.exclude_reference_similarity) or not (
        0.0 < args.exclude_reference_similarity <= 1.0
    ):
        raise SystemExit("--exclude-reference-similarity must be finite and in (0, 1]")
    cutoff = _parse_cutoff(args.cutoff_date)
    if args.evidence_mode == "temporal" and cutoff is None:
        raise SystemExit("--evidence-mode=temporal requires --cutoff-date")
    if args.evidence_mode != "temporal" and cutoff is not None:
        raise SystemExit("--cutoff-date is only valid with --evidence-mode=temporal")
    if args.evidence_mode != "retrieval" and not str(
        args.evidence_snapshot_id or ""
    ).strip():
        raise SystemExit(
            f"--evidence-mode={args.evidence_mode} requires --evidence-snapshot-id"
        )
    if not args.chembl_fp.exists():
        raise SystemExit(f"ChEMBL fingerprint parquet is required: {args.chembl_fp}")
    activities_path = args.chembl_fp.parent / "human_activities.parquet"
    if not activities_path.exists():
        raise SystemExit(f"Missing {activities_path}")

    fp_df = load_fingerprints(args.chembl_fp)
    act = load_activities(
        activities_path,
        quality_policy=args.quality_policy,
        evidence_mode=args.evidence_mode,
    )
    initial_activity_rows = len(act)
    act, quality_stats = apply_quality_policy(act, args.quality_policy)
    temporal_stats = None
    if cutoff is not None:
        act, temporal_stats = apply_temporal_cutoff(act, cutoff)
    validate_reference_overlap(act, fp_df, activities_path)

    reference = {
        "fingerprints": str(args.chembl_fp),
        "fingerprints_sha256": _sha256(args.chembl_fp),
        "activities": str(activities_path),
        "activities_sha256": _sha256(activities_path),
    }
    staged_pairs: list[tuple[Path, Path]] = []
    case_summaries: list[dict[str, object]] = []
    try:
        for row in batch.to_dict("records"):
            case_id = str(row["case_id"])
            ligand_sdf = Path(str(row["ligand_sdf"]))
            out_scores = Path(str(row["out_scores"]))
            out_metadata = Path(str(row["out_metadata_json"]))
            if not ligand_sdf.exists():
                raise SystemExit(f"Daina batch ligand SDF missing for {case_id}: {ligand_sdf}")
            qfp, query_connectivity_key = query_features(ligand_sdf)
            scores, scoring_stats = score_query_against_reference(
                qfp=qfp,
                query_connectivity_key=query_connectivity_key,
                fp_df=fp_df,
                act=act,
                evidence_mode=args.evidence_mode,
                exclude_reference_similarity=args.exclude_reference_similarity,
                scoring_method=args.scoring_method,
            )
            scores["quality_policy"] = args.quality_policy
            score_stage = _staging_path(out_scores)
            metadata_stage = _staging_path(out_metadata)
            score_stage.parent.mkdir(parents=True, exist_ok=True)
            metadata_stage.parent.mkdir(parents=True, exist_ok=True)
            scores.to_csv(score_stage, sep="\t", index=False)
            metadata = {
                "schema_version": "skinscout.daina-run.v1",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "case_id": case_id,
                "evidence_mode": args.evidence_mode,
                "evidence_snapshot_id": args.evidence_snapshot_id,
                "cutoff_date": cutoff.isoformat() if cutoff is not None else None,
                "exclude_reference_similarity": (
                    args.exclude_reference_similarity
                    if args.evidence_mode != "retrieval"
                    else None
                ),
                "quality_policy": args.quality_policy,
                "scoring_method": args.scoring_method,
                "score_is_calibrated_probability": False,
                "inputs": {
                    "ligand_sdf": str(ligand_sdf),
                    "ligand_sdf_sha256": _sha256(ligand_sdf),
                    **reference,
                },
                "counts": {
                    "initial_activity_rows": initial_activity_rows,
                    **scoring_stats,
                    "ranked_targets": len(scores),
                },
                "quality_filter": quality_stats,
                "temporal_filter": temporal_stats,
            }
            metadata_stage.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n"
            )
            staged_pairs.extend([(score_stage, out_scores), (metadata_stage, out_metadata)])
            case_summaries.append(
                {
                    "case_id": case_id,
                    "ranked_targets": len(scores),
                    **scoring_stats,
                }
            )
        for staged, final in staged_pairs:
            staged.replace(final)
        _write_json_atomic(
            {
                "schema_version": "skinscout.daina-batch.v1",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "n_cases": len(case_summaries),
                "reference": reference,
                "evidence_mode": args.evidence_mode,
                "evidence_snapshot_id": args.evidence_snapshot_id,
                "cutoff_date": cutoff.isoformat() if cutoff is not None else None,
                "quality_policy": args.quality_policy,
                "scoring_method": args.scoring_method,
                "cases": case_summaries,
            },
            args.out_summary_json,
        )
    except BaseException:
        _cleanup(final_paths)
        raise


if __name__ == "__main__":
    main()
