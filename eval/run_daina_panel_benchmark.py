#!/usr/bin/env python3
"""Run a reproducible Daina known-target panel benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


ROOT = Path(__file__).resolve().parents[1]
DAINA_BATCH = ROOT / "scripts" / "stage3_daina_batch.py"
KNOWN_TARGET_EVAL = ROOT / "eval" / "skin_known_target_recovery_eval.py"
REQUIRED_PANEL_COLUMNS = {"case_id", "smiles", "known_targets"}
EVALUATOR_REQUIRED_COLUMNS = {
    "case_id",
    "inci_name",
    "panel",
    "skin_effect",
    "smiles",
    "known_targets",
}
CASE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
EVIDENCE_MODES = ("retrieval", "leave-query-out", "temporal")
EVALUATION_MODES = ("retrospective", "leave-query-out", "temporal")
QUALITY_POLICIES = ("legacy", "high-confidence", "claim-grade")
SCORING_METHODS = ("max-similarity", "quality-hybrid")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"{label} is required and must be non-empty: {path}")
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        raise SystemExit(f"Unable to read {label}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} must contain a JSON object: {path}")
    return payload


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _clean_staging(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _staging_dir(out_dir: Path) -> Path:
    return out_dir.parent / f".{out_dir.name}.daina-panel-staging"


def _validate_fraction(value: float, label: str) -> None:
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise SystemExit(f"{label} must be a finite value in [0, 1]: {value}")


def _read_panel(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"Panel CSV is required and must be non-empty: {path}")
    try:
        panel = pd.read_csv(path)
    except Exception as exc:
        raise SystemExit(f"Unable to read panel CSV: {path}: {exc}") from exc
    missing = sorted(REQUIRED_PANEL_COLUMNS - set(panel.columns))
    if missing:
        raise SystemExit(f"Panel CSV missing required columns {missing}: {path}")
    if panel.empty:
        raise SystemExit(f"Panel CSV contains no cases: {path}")

    panel = panel.copy()
    seen: set[str] = set()
    for idx, row in panel.iterrows():
        case_id = str(row["case_id"]).strip()
        if not case_id:
            raise SystemExit(f"Panel case_id is blank at row index {idx}")
        if not CASE_ID_RE.fullmatch(case_id) or case_id in {".", ".."}:
            raise SystemExit(
                "Panel case_id must be a safe file stem containing only "
                f"letters, digits, '.', '_' or '-': {case_id!r}"
            )
        if case_id in seen:
            raise SystemExit(f"Panel CSV contains duplicate case_id values: {case_id}")
        seen.add(case_id)

        smiles = str(row["smiles"]).strip()
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise SystemExit(
                f"Panel smiles contains invalid SMILES at row index {idx}: {smiles!r}"
            )
        targets = [token.strip() for token in str(row["known_targets"]).split(";")]
        if not targets or any(not token for token in targets):
            raise SystemExit(f"Panel known_targets contains empty value(s): {case_id}")
        duplicates = sorted({target for target in targets if targets.count(target) > 1})
        if duplicates:
            raise SystemExit(
                f"Panel known_targets contains duplicate value(s) for {case_id}: "
                + ", ".join(duplicates[:10])
            )
        panel.at[idx, "case_id"] = case_id
        panel.at[idx, "smiles"] = smiles
        panel.at[idx, "known_targets"] = ";".join(targets)
    return panel


def _relative_to_benchmark(path: Path, benchmark_root: Path) -> str:
    try:
        return path.resolve().relative_to(benchmark_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _resolve_benchmark_path(value: str, benchmark_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return benchmark_root / path


def _normalize_paths_for_publish(value: Any, staging: Path) -> Any:
    if isinstance(value, dict):
        return {
            key: _normalize_paths_for_publish(item, staging)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_normalize_paths_for_publish(item, staging) for item in value]
    if isinstance(value, str):
        path = Path(value)
        try:
            return path.resolve().relative_to(staging.resolve()).as_posix()
        except (OSError, ValueError):
            return value
    return value


def _require_resolvable_paths(
    payload: Any,
    benchmark_root: Path,
    *,
    context: str,
    key_hint: str = "",
) -> None:
    path_keys = {
        "path",
        "panel_csv",
        "chembl_fp",
        "human_activities",
        "source_manifest",
        "fingerprint_manifest",
        "batch_csv",
        "evaluator_panel_csv",
        "sdf_dir",
        "daina_batch_summary_json",
        "case_metrics_csv",
        "target_metrics_csv",
        "summary_path",
        "daina_tsv",
        "daina_metadata_json",
        "ranking_csv",
        "ranking_metadata_json",
        "ranking_path",
        "rankings_dir",
        "cases_csv",
        "ligand_sdf",
        "fingerprints",
        "activities",
        "manifest_path",
    }
    if isinstance(payload, dict):
        for key, value in payload.items():
            hint = key if not key_hint else f"{key_hint}.{key}"
            _require_resolvable_paths(value, benchmark_root, context=context, key_hint=hint)
    elif isinstance(payload, list):
        for idx, value in enumerate(payload):
            _require_resolvable_paths(
                value,
                benchmark_root,
                context=context,
                key_hint=f"{key_hint}[{idx}]",
            )
    elif isinstance(payload, str) and key_hint.rsplit(".", 1)[-1] in path_keys:
        path = _resolve_benchmark_path(payload, benchmark_root)
        if not path.exists():
            raise SystemExit(f"{context} path does not resolve after publish: {key_hint}={payload}")


def _read_and_validate_chembl_provenance(chembl_fp: Path) -> dict[str, Any]:
    chembl_dir = chembl_fp.parent
    activities = chembl_dir / "human_activities.parquet"
    source_manifest_path = chembl_dir / "source_manifest.json"
    fingerprint_manifest_path = chembl_dir / "fingerprint_manifest.json"
    if not activities.exists() or activities.stat().st_size == 0:
        raise SystemExit(f"ChEMBL human activities parquet is required: {activities}")

    source_manifest = _read_json(source_manifest_path, "ChEMBL source manifest")
    source = source_manifest.get("source")
    if not isinstance(source, dict):
        raise SystemExit(f"ChEMBL source manifest missing source object: {source_manifest_path}")
    release = str(source.get("release", "")).strip()
    license_name = str(source.get("license", "")).strip()
    if not release or not license_name:
        raise SystemExit(
            "ChEMBL source manifest requires nonblank source.release and "
            f"source.license: {source_manifest_path}"
        )

    fingerprint_manifest = _read_json(
        fingerprint_manifest_path,
        "ChEMBL fingerprint manifest",
    )
    source_snapshot = fingerprint_manifest.get("source_snapshot")
    input_meta = fingerprint_manifest.get("input")
    artifact_meta = fingerprint_manifest.get("artifact")
    algorithm_meta = fingerprint_manifest.get("algorithm")
    if not isinstance(source_snapshot, dict):
        raise SystemExit(
            f"ChEMBL fingerprint manifest missing source_snapshot object: {fingerprint_manifest_path}"
        )
    if not isinstance(input_meta, dict) or not isinstance(artifact_meta, dict):
        raise SystemExit(
            f"ChEMBL fingerprint manifest missing input/artifact object: {fingerprint_manifest_path}"
        )
    if not isinstance(algorithm_meta, dict):
        raise SystemExit(
            f"ChEMBL fingerprint manifest missing algorithm object: {fingerprint_manifest_path}"
        )

    source_manifest_sha = _sha256(source_manifest_path)
    activities_sha = _sha256(activities)
    chembl_fp_sha = _sha256(chembl_fp)
    checks = (
        (
            source_snapshot.get("manifest_sha256"),
            source_manifest_sha,
            "source_snapshot.manifest_sha256",
        ),
        (input_meta.get("sha256"), activities_sha, "input.sha256"),
        (artifact_meta.get("sha256"), chembl_fp_sha, "artifact.sha256"),
    )
    for observed, expected, label in checks:
        if str(observed or "").strip() != expected:
            raise SystemExit(
                f"ChEMBL fingerprint manifest {label} does not match current artifact: "
                f"{fingerprint_manifest_path}"
            )

    normalized_source_snapshot = dict(source_snapshot)
    normalized_source_snapshot["manifest_path"] = str(source_manifest_path.resolve())
    normalized_input_meta = dict(input_meta)
    normalized_input_meta["path"] = str(activities.resolve())
    normalized_artifact_meta = dict(artifact_meta)
    normalized_artifact_meta["path"] = str(chembl_fp.resolve())

    return {
        "source_manifest": str(source_manifest_path.resolve()),
        "source_manifest_sha256": source_manifest_sha,
        "source_manifest_metadata": {
            "schema_version": source_manifest.get("schema_version", ""),
            "source": source,
        },
        "fingerprint_manifest": str(fingerprint_manifest_path.resolve()),
        "fingerprint_manifest_sha256": _sha256(fingerprint_manifest_path),
        "fingerprint_manifest_metadata": {
            "schema_version": fingerprint_manifest.get("schema_version", ""),
            "source_snapshot": normalized_source_snapshot,
            "algorithm": algorithm_meta,
            "input": normalized_input_meta,
            "artifact": normalized_artifact_meta,
        },
        "chembl_fp": str(chembl_fp.resolve()),
        "chembl_fp_sha256": chembl_fp_sha,
        "human_activities": str(activities.resolve()),
        "human_activities_sha256": activities_sha,
    }


def _write_sdf(path: Path, case_id: str, smiles: str) -> None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise SystemExit(f"Invalid SMILES for {case_id}: {smiles}")
    AllChem.Compute2DCoords(mol)
    mol.SetProp("_Name", case_id)
    writer = Chem.SDWriter(str(path))
    try:
        writer.write(mol)
    finally:
        writer.close()


def _write_inputs(panel: pd.DataFrame, staging: Path) -> tuple[Path, Path]:
    sdf_dir = staging / "sdf"
    daina_dir = staging / "daina"
    sdf_dir.mkdir(parents=True)
    daina_dir.mkdir(parents=True)
    batch_rows: list[dict[str, str]] = []
    eval_rows: list[dict[str, Any]] = []
    for row in panel.to_dict("records"):
        case_id = str(row["case_id"])
        sdf_path = sdf_dir / f"{case_id}.sdf"
        _write_sdf(sdf_path, case_id, str(row["smiles"]))
        batch_rows.append(
            {
                "case_id": case_id,
                "ligand_sdf": str(sdf_path),
                "out_scores": str(daina_dir / f"{case_id}.tsv"),
                "out_metadata_json": str(daina_dir / f"{case_id}.metadata.json"),
            }
        )
        eval_row = dict(row)
        defaults = {
            "inci_name": case_id,
            "panel": "control",
            "skin_effect": "Daina known-target benchmark",
        }
        for column, default in defaults.items():
            if (
                column not in eval_row
                or pd.isna(eval_row[column])
                or not str(eval_row[column]).strip()
            ):
                eval_row[column] = default
        eval_rows.append(eval_row)
    batch_csv = staging / "daina_batch.csv"
    eval_panel_csv = staging / "known_target_panel_for_eval.csv"
    pd.DataFrame(batch_rows).to_csv(batch_csv, index=False)
    evaluator_panel = pd.DataFrame(eval_rows)
    missing_eval_columns = sorted(EVALUATOR_REQUIRED_COLUMNS - set(evaluator_panel.columns))
    if missing_eval_columns:
        raise SystemExit(
            f"Evaluator panel construction missing required columns {missing_eval_columns}"
        )
    evaluator_panel.to_csv(eval_panel_csv, index=False)
    return batch_csv, eval_panel_csv


def _run_command(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    record = {
        "argv": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    if result.returncode != 0:
        raise RuntimeError(json.dumps(record, indent=2))
    return record


def _convert_rankings(panel: pd.DataFrame, staging: Path) -> list[dict[str, Any]]:
    daina_dir = staging / "daina"
    rankings_dir = staging / "rankings"
    rankings_dir.mkdir()
    converted: list[dict[str, Any]] = []
    for row in panel.to_dict("records"):
        case_id = str(row["case_id"])
        tsv_path = daina_dir / f"{case_id}.tsv"
        if not tsv_path.exists() or tsv_path.stat().st_size == 0:
            raise SystemExit(f"Daina TSV is required and must be non-empty: {tsv_path}")
        try:
            scores = pd.read_csv(tsv_path, sep="\t")
        except Exception as exc:
            raise SystemExit(f"Unable to read Daina TSV: {tsv_path}: {exc}") from exc
        required = {"target_id", "score"}
        missing = sorted(required - set(scores.columns))
        if missing:
            raise SystemExit(f"Daina TSV missing required columns {missing}: {tsv_path}")
        if scores.empty:
            raise SystemExit(f"Daina TSV contains no target scores: {tsv_path}")
        ranking = scores.loc[:, ["target_id", "score"]].copy()
        ranking["target_id"] = ranking["target_id"].astype(str).str.strip()
        if ranking["target_id"].eq("").any() or ranking["target_id"].duplicated().any():
            raise SystemExit(f"Daina TSV contains blank or duplicate target_id values: {tsv_path}")
        ranking["score"] = pd.to_numeric(ranking["score"], errors="coerce")
        if ranking["score"].isna().any() or ~ranking["score"].map(math.isfinite).all():
            raise SystemExit(f"Daina TSV score column contains non-finite values: {tsv_path}")
        ranking = ranking.sort_values(
            ["score", "target_id"], ascending=[False, True], kind="mergesort"
        ).reset_index(drop=True)
        ranking.insert(0, "rank", range(1, len(ranking) + 1))
        ranking["known_target_prior"] = 0.0
        ranking["known_target_prior_norm"] = 0.0
        out_csv = rankings_dir / f"{case_id}__ranked_targets_v3.csv"
        ranking.to_csv(out_csv, index=False)
        source_metadata = daina_dir / f"{case_id}.metadata.json"
        if not source_metadata.exists() or source_metadata.stat().st_size == 0:
            raise SystemExit(
                f"Daina metadata sidecar is required and must be non-empty: {source_metadata}"
            )
        try:
            metadata = json.loads(source_metadata.read_text())
        except Exception as exc:
            raise SystemExit(f"Unable to read Daina metadata sidecar: {source_metadata}") from exc
        sidecar = rankings_dir / f"{case_id}__ranked_targets_v3.metadata.json"
        metadata = _normalize_paths_for_publish(metadata, staging)
        metadata["ranking_csv"] = _relative_to_benchmark(out_csv, staging)
        metadata["ranking_sha256"] = _sha256(out_csv)
        _write_json_atomic(metadata, source_metadata)
        _write_json_atomic(metadata, sidecar)
        converted.append(
            {
                "case_id": case_id,
                "daina_tsv": _relative_to_benchmark(tsv_path, staging),
                "daina_metadata_json": _relative_to_benchmark(source_metadata, staging),
                "ranking_csv": _relative_to_benchmark(out_csv, staging),
                "ranking_metadata_json": _relative_to_benchmark(sidecar, staging),
                "ranking_sha256": metadata["ranking_sha256"],
                "n_ranked_targets": int(len(ranking)),
            }
        )
    return converted


def _derived_evaluation_mode(evidence_mode: str, requested: str | None) -> str:
    expected = "retrospective" if evidence_mode == "retrieval" else evidence_mode
    if requested and requested != expected:
        raise SystemExit(
            "Evidence/evaluation mode mismatch; required mapping is "
            "retrieval->retrospective, leave-query-out->leave-query-out, "
            f"temporal->temporal, got {evidence_mode}->{requested}"
        )
    return expected


def _normalize_evaluation_outputs(
    *,
    case_metrics: Path,
    target_metrics: Path,
    eval_summary: Path,
    staging: Path,
) -> None:
    for metrics_path in (case_metrics, target_metrics):
        metrics = pd.read_csv(metrics_path)
        changed = False
        for column in ("ranking_path", "ranking_metadata_path"):
            if column not in metrics.columns:
                continue
            metrics[column] = metrics[column].map(
                lambda value: _normalize_paths_for_publish(value, staging)
                if isinstance(value, str)
                else value
            )
            changed = True
        if changed:
            metrics.to_csv(metrics_path, index=False)
    summary = _read_json(eval_summary, "Skin known-target evaluation summary")
    summary = _normalize_paths_for_publish(summary, staging)
    _write_json_atomic(summary, eval_summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-csv", required=True, type=Path)
    parser.add_argument("--chembl-fp", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--evidence-mode", choices=EVIDENCE_MODES, default="retrieval")
    parser.add_argument("--evaluation-mode", choices=EVALUATION_MODES)
    parser.add_argument("--evidence-snapshot-id", default="")
    parser.add_argument("--cutoff-date", default="")
    parser.add_argument("--exclude-reference-similarity", type=float, default=0.85)
    parser.add_argument("--quality-policy", choices=QUALITY_POLICIES, default="legacy")
    parser.add_argument("--scoring-method", choices=SCORING_METHODS, default="max-similarity")
    parser.add_argument("--min-case-top10", type=float, default=0.80)
    parser.add_argument("--min-target-top10", type=float, default=0.50)
    parser.add_argument("--min-target-top30", type=float, default=0.60)
    parser.add_argument(
        "--allow-threshold-failure",
        action="store_true",
        help="write failing recovery metrics only for explicit diagnostics",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    staging = _staging_dir(out_dir)
    if out_dir.exists():
        raise SystemExit(f"Output directory already exists; refusing to overwrite: {out_dir}")
    if not args.chembl_fp.exists():
        raise SystemExit(f"ChEMBL fingerprint parquet is required: {args.chembl_fp}")
    chembl_provenance = _read_and_validate_chembl_provenance(args.chembl_fp)
    for value, label in (
        (args.exclude_reference_similarity, "--exclude-reference-similarity"),
        (args.min_case_top10, "--min-case-top10"),
        (args.min_target_top10, "--min-target-top10"),
        (args.min_target_top30, "--min-target-top30"),
    ):
        _validate_fraction(value, label)
    if args.evidence_mode == "temporal" and not args.cutoff_date:
        raise SystemExit("--evidence-mode=temporal requires --cutoff-date")
    if args.evidence_mode != "temporal" and args.cutoff_date:
        raise SystemExit("--cutoff-date is only valid with --evidence-mode=temporal")
    if args.evidence_mode != "retrieval" and not args.evidence_snapshot_id.strip():
        raise SystemExit(f"--evidence-mode={args.evidence_mode} requires --evidence-snapshot-id")

    evaluation_mode = _derived_evaluation_mode(args.evidence_mode, args.evaluation_mode)
    if evaluation_mode == "temporal" and (
        not args.evidence_snapshot_id.strip() or not args.cutoff_date.strip()
    ):
        raise SystemExit(
            "--evaluation-mode=temporal requires --evidence-snapshot-id and --cutoff-date"
        )

    panel = _read_panel(args.panel_csv)
    commands: list[dict[str, Any]] = []
    _clean_staging(staging)
    try:
        staging.mkdir(parents=True)
        batch_csv, eval_panel_csv = _write_inputs(panel, staging)
        batch_summary = staging / "daina_batch_summary.json"
        daina_command = [
            sys.executable,
            str(DAINA_BATCH),
            "--batch-csv",
            str(batch_csv),
            "--chembl-fp",
            str(args.chembl_fp),
            "--out-summary-json",
            str(batch_summary),
            "--evidence-mode",
            args.evidence_mode,
            "--exclude-reference-similarity",
            f"{args.exclude_reference_similarity:g}",
            "--quality-policy",
            args.quality_policy,
            "--scoring-method",
            args.scoring_method,
        ]
        if args.evidence_snapshot_id.strip():
            daina_command.extend(["--evidence-snapshot-id", args.evidence_snapshot_id.strip()])
        if args.cutoff_date.strip():
            daina_command.extend(["--cutoff-date", args.cutoff_date.strip()])
        commands.append(_run_command(daina_command))

        converted = _convert_rankings(panel, staging)
        case_metrics = staging / "skin_known_cases.csv"
        target_metrics = staging / "skin_known_targets.csv"
        eval_summary = staging / "skin_known_summary.json"
        eval_command = [
            sys.executable,
            str(KNOWN_TARGET_EVAL),
            "--cases-csv",
            str(eval_panel_csv),
            "--rankings-dir",
            str(staging / "rankings"),
            "--out-csv",
            str(case_metrics),
            "--out-target-csv",
            str(target_metrics),
            "--out-summary-json",
            str(eval_summary),
            "--evaluation-mode",
            evaluation_mode,
            "--evidence-snapshot",
            args.evidence_snapshot_id.strip(),
            "--cutoff-date",
            args.cutoff_date.strip(),
            "--min-case-top10",
            f"{args.min_case_top10:g}",
            "--min-target-top10",
            f"{args.min_target_top10:g}",
            "--min-target-top30",
            f"{args.min_target_top30:g}",
        ]
        if args.allow_threshold_failure:
            eval_command.append("--allow-threshold-failure")
        commands.append(_run_command(eval_command))
        _normalize_evaluation_outputs(
            case_metrics=case_metrics,
            target_metrics=target_metrics,
            eval_summary=eval_summary,
            staging=staging,
        )

        manifest = {
            "schema_version": "skinscout.daina-panel-benchmark.v1",
            "created_at_utc": _utc_now(),
            "parameters": {
                "evidence_mode": args.evidence_mode,
                "evaluation_mode": evaluation_mode,
                "evidence_snapshot_id": args.evidence_snapshot_id.strip(),
                "cutoff_date": args.cutoff_date.strip(),
                "exclude_reference_similarity": args.exclude_reference_similarity,
                "quality_policy": args.quality_policy,
                "scoring_method": args.scoring_method,
                "thresholds": {
                    "min_case_top10": args.min_case_top10,
                    "min_target_top10": args.min_target_top10,
                    "min_target_top30": args.min_target_top30,
                    "allow_threshold_failure": args.allow_threshold_failure,
                },
            },
            "inputs": {
                "panel_csv": str(args.panel_csv.resolve()),
                "panel_csv_sha256": _sha256(args.panel_csv),
                **chembl_provenance,
            },
            "generated_inputs": {
                "batch_csv": _relative_to_benchmark(batch_csv, staging),
                "evaluator_panel_csv": _relative_to_benchmark(eval_panel_csv, staging),
                "sdf_dir": _relative_to_benchmark(staging / "sdf", staging),
            },
            "subprocess_commands": commands,
            "rankings": converted,
            "outputs": {
                "daina_batch_summary_json": _relative_to_benchmark(batch_summary, staging),
                "case_metrics_csv": _relative_to_benchmark(case_metrics, staging),
                "target_metrics_csv": _relative_to_benchmark(target_metrics, staging),
                "summary_path": _relative_to_benchmark(eval_summary, staging),
                "daina_batch_summary_sha256": _sha256(batch_summary),
                "case_metrics_sha256": _sha256(case_metrics),
                "target_metrics_sha256": _sha256(target_metrics),
                "summary_sha256": _sha256(eval_summary),
            },
        }
        manifest = _normalize_paths_for_publish(manifest, staging)
        _require_resolvable_paths(manifest, staging, context="Benchmark manifest")
        _write_json_atomic(manifest, staging / "manifest.json")
        staging.replace(out_dir)
    except BaseException:
        _clean_staging(staging)
        raise


if __name__ == "__main__":
    main()
