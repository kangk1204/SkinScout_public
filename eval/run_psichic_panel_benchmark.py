#!/usr/bin/env python3
"""Audit existing per-case PSICHIC proteome rankings against a known-target panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_PANEL_COLUMNS = {"case_id", "known_targets"}
CASE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SOURCE_TSV_NAME = "psichic_proteome.tsv"
DEFAULT_SOURCE_TEMPLATE = f"{{case_id}}/{SOURCE_TSV_NAME}"
SCORE_SEMANTICS = "raw_psichic_model_score_higher_is_better_not_calibrated_probability"
METADATA_CANDIDATES = (
    "psichic_proteome.metadata.json",
    "psichic_metadata.json",
    "metadata.json",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _clean_staging(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _staging_dir(out_dir: Path) -> Path:
    return out_dir.parent / f".{out_dir.name}.psichic-panel-staging"


def _relative_to_benchmark(path: Path, benchmark_root: Path) -> str:
    try:
        return path.resolve().relative_to(benchmark_root.resolve()).as_posix()
    except ValueError:
        return str(path)


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
        if pd.isna(row["case_id"]):
            raise SystemExit(f"Panel case_id is blank at row index {idx}")
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

        if pd.isna(row["known_targets"]):
            raise SystemExit(f"Panel known_targets is blank for {case_id}")
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
        panel.at[idx, "known_targets"] = ";".join(targets)
    return panel


def _validate_source_template(source_template: str) -> str:
    template = source_template.strip()
    if template.count("{case_id}") != 1:
        raise SystemExit("--source-template must contain exactly one {case_id} placeholder")
    remainder = template.replace("{case_id}", "")
    if "{" in remainder or "}" in remainder:
        raise SystemExit("--source-template contains an unsupported placeholder")
    template_path = Path(template)
    if template_path.is_absolute() or ".." in template_path.parts:
        raise SystemExit("--source-template must be a relative path without '..'")
    return template


def _source_tsv_path(psichic_root: Path, source_template: str, case_id: str) -> Path:
    return psichic_root / source_template.format(case_id=case_id)


def _bool_like(value: object) -> bool:
    return (
        isinstance(value, bool)
        or type(value).__name__ == "bool_"
        or (isinstance(value, str) and value.strip().lower() in {"true", "false"})
    )


def _read_ranking(tsv_path: Path) -> pd.DataFrame:
    try:
        scores = pd.read_csv(tsv_path, sep="\t")
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=["rank", "target_id", "psichic_score"])
    except Exception as exc:
        raise SystemExit(f"Unable to read PSICHIC TSV: {tsv_path}: {exc}") from exc
    required = {"target_id", "psichic_score"}
    missing = sorted(required - set(scores.columns))
    if missing:
        raise SystemExit(f"PSICHIC TSV missing required columns {missing}: {tsv_path}")
    if scores.empty:
        return pd.DataFrame(columns=["rank", "target_id", "psichic_score"])

    ranking = scores.loc[:, ["target_id", "psichic_score"]].copy()
    ranking["target_id"] = ranking["target_id"].astype(str).str.strip()
    if ranking["target_id"].eq("").any() or ranking["target_id"].duplicated().any():
        raise SystemExit(f"PSICHIC TSV contains blank or duplicate target_id values: {tsv_path}")
    if ranking["psichic_score"].map(_bool_like).any():
        raise SystemExit(f"PSICHIC TSV psichic_score column must be numeric: {tsv_path}")
    ranking["psichic_score"] = pd.to_numeric(ranking["psichic_score"], errors="coerce")
    finite = ranking["psichic_score"].map(lambda value: math.isfinite(float(value)))
    if ranking["psichic_score"].isna().any() or not finite.all():
        raise SystemExit(f"PSICHIC TSV psichic_score column contains non-finite values: {tsv_path}")

    ranking = ranking.sort_values(
        ["psichic_score", "target_id"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    ranking.insert(0, "rank", range(1, len(ranking) + 1))
    return ranking


def _read_json_if_object(path: Path) -> dict[str, Any] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _discover_model_metadata(tsv_path: Path) -> dict[str, Any]:
    for name in METADATA_CANDIDATES:
        metadata_path = tsv_path.with_name(name)
        metadata = _read_json_if_object(metadata_path)
        if metadata is None:
            continue
        discovered: dict[str, Any] = {
            "metadata_json": str(metadata_path.resolve()),
            "metadata_json_sha256": _sha256(metadata_path),
        }
        for key in ("model", "model_name", "checkpoint", "checkpoint_path", "weights"):
            value = metadata.get(key)
            if value not in (None, ""):
                discovered[key] = value
        return discovered
    return {}


def _audit_panel(
    panel: pd.DataFrame,
    psichic_root: Path,
    source_template: str,
    staging: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    rankings_dir = staging / "rankings"
    rankings_dir.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []

    for panel_row in panel.to_dict("records"):
        case_id = str(panel_row["case_id"])
        known_targets = str(panel_row["known_targets"]).split(";")
        tsv_path = _source_tsv_path(psichic_root, source_template, case_id)
        base_row: dict[str, Any] = {
            "case_id": case_id,
            "known_targets": ";".join(known_targets),
            "source_status": "",
            "source_tsv": "",
            "source_tsv_sha256": "",
            "ranking_csv": "",
            "covered_by_source": False,
            "ranked_target_count": 0,
            "known_target_ranked": False,
            "best_known_target": "",
            "best_known_target_rank": "",
            "best_known_target_score": "",
            "score_semantics": SCORE_SEMANTICS,
        }
        if not tsv_path.exists():
            base_row["source_status"] = "missing_source_tsv"
            rows.append(base_row)
            target_rows.extend(
                _target_audit_rows(case_id, known_targets, base_row["source_status"])
            )
            sources.append(
                {
                    "case_id": case_id,
                    "status": "missing_source_tsv",
                    "path": str(tsv_path.resolve()),
                }
            )
            continue
        if tsv_path.stat().st_size == 0:
            base_row["source_status"] = "no_case_coverage_empty_tsv"
            rows.append(base_row)
            target_rows.extend(
                _target_audit_rows(case_id, known_targets, base_row["source_status"])
            )
            sources.append(
                {
                    "case_id": case_id,
                    "status": "no_case_coverage_empty_tsv",
                    "path": str(tsv_path.resolve()),
                    "sha256": _sha256(tsv_path),
                    "model_metadata": _discover_model_metadata(tsv_path),
                }
            )
            continue

        ranking = _read_ranking(tsv_path)
        tsv_sha = _sha256(tsv_path)
        base_row["source_tsv"] = str(tsv_path.resolve())
        base_row["source_tsv_sha256"] = tsv_sha
        if ranking.empty:
            base_row["source_status"] = "no_case_coverage_no_ranked_targets"
            rows.append(base_row)
            target_rows.extend(
                _target_audit_rows(case_id, known_targets, base_row["source_status"])
            )
            sources.append(
                {
                    "case_id": case_id,
                    "status": "no_case_coverage_no_ranked_targets",
                    "path": str(tsv_path.resolve()),
                    "sha256": tsv_sha,
                    "model_metadata": _discover_model_metadata(tsv_path),
                }
            )
            continue

        out_csv = rankings_dir / f"{case_id}__psichic_direct_ranked_targets.csv"
        ranking.to_csv(out_csv, index=False)
        target_hits = ranking[ranking["target_id"].isin(known_targets)].copy()
        base_row["source_status"] = (
            "known_target_ranked" if not target_hits.empty else "known_target_missing_from_ranking"
        )
        base_row["ranking_csv"] = _relative_to_benchmark(out_csv, staging)
        base_row["covered_by_source"] = True
        base_row["ranked_target_count"] = int(len(ranking))
        if not target_hits.empty:
            best = target_hits.sort_values(["rank"], ascending=True).iloc[0]
            base_row["known_target_ranked"] = True
            base_row["best_known_target"] = str(best["target_id"])
            base_row["best_known_target_rank"] = int(best["rank"])
            base_row["best_known_target_score"] = float(best["psichic_score"])
        rows.append(base_row)
        target_rows.extend(
            _target_audit_rows(
                case_id,
                known_targets,
                base_row["source_status"],
                ranking=ranking,
            )
        )
        sources.append(
            {
                "case_id": case_id,
                "status": base_row["source_status"],
                "path": str(tsv_path.resolve()),
                "sha256": tsv_sha,
                "ranking_csv": _relative_to_benchmark(out_csv, staging),
                "ranking_sha256": _sha256(out_csv),
                "ranked_target_count": int(len(ranking)),
                "model_metadata": _discover_model_metadata(tsv_path),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(target_rows), sources


def _target_audit_rows(
    case_id: str,
    known_targets: list[str],
    source_status: str,
    *,
    ranking: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    indexed = (
        ranking.set_index("target_id", drop=False)
        if ranking is not None and not ranking.empty
        else None
    )
    rows: list[dict[str, Any]] = []
    for target_id in known_targets:
        target_rank: int | str = ""
        target_score: float | str = ""
        target_status = source_status
        if indexed is not None:
            if target_id in indexed.index:
                hit = indexed.loc[target_id]
                target_rank = int(hit["rank"])
                target_score = float(hit["psichic_score"])
                target_status = "known_target_ranked"
            else:
                target_status = "known_target_missing_from_ranking"
        rows.append(
            {
                "case_id": case_id,
                "known_target": target_id,
                "source_status": source_status,
                "target_status": target_status,
                "target_rank": target_rank,
                "target_score": target_score,
                "top10": bool(target_rank != "" and int(target_rank) <= 10),
                "top30": bool(target_rank != "" and int(target_rank) <= 30),
                "reciprocal_rank": 1.0 / int(target_rank) if target_rank != "" else 0.0,
                "score_semantics": SCORE_SEMANTICS,
            }
        )
    return rows


def _build_summary(cases: pd.DataFrame, targets: pd.DataFrame) -> dict[str, Any]:
    total = int(len(cases))
    present = int(cases["source_tsv_sha256"].astype(str).ne("").sum())
    covered = int(cases["covered_by_source"].astype(bool).sum())
    ranked = int(cases["known_target_ranked"].astype(bool).sum())
    ranks = pd.to_numeric(cases["best_known_target_rank"], errors="coerce")
    covered_ranks = ranks.dropna()
    case_top10 = int((ranks <= 10).fillna(False).sum())
    case_top30 = int((ranks <= 30).fillna(False).sum())
    target_total = int(len(targets))
    target_top10 = int(targets["top10"].astype(bool).sum())
    target_top30 = int(targets["top30"].astype(bool).sum())
    target_ranks = pd.to_numeric(targets["target_rank"], errors="coerce").dropna()
    return {
        "schema_version": "skinscout.psichic-direct-panel-audit-summary.v1",
        "score_semantics": SCORE_SEMANTICS,
        "panel_denominator_cases": total,
        "source_tsv_present_cases": present,
        "source_covered_cases": covered,
        "known_target_ranked_cases": ranked,
        "missing_source_cases": int((cases["source_status"] == "missing_source_tsv").sum()),
        "no_case_coverage_cases": int(
            cases["source_status"].isin(
                ["no_case_coverage_empty_tsv", "no_case_coverage_no_ranked_targets"]
            ).sum()
        ),
        "known_target_missing_from_covered_ranking_cases": int(
            (cases["source_status"] == "known_target_missing_from_ranking").sum()
        ),
        "coverage_fraction_of_full_panel": covered / total,
        "known_target_ranked_fraction_of_full_panel": ranked / total,
        "known_target_ranked_fraction_of_covered_cases": ranked / covered if covered else None,
        "case_top10_count": case_top10,
        "case_top30_count": case_top30,
        "case_top10_fraction_of_full_panel": case_top10 / total,
        "case_top30_fraction_of_full_panel": case_top30 / total,
        "case_top10_fraction_of_covered_cases": case_top10 / covered if covered else None,
        "case_top30_fraction_of_covered_cases": case_top30 / covered if covered else None,
        "target_pair_denominator": target_total,
        "target_pair_top10_count": target_top10,
        "target_pair_top30_count": target_top30,
        "target_pair_top10_fraction": target_top10 / target_total,
        "target_pair_top30_fraction": target_top30 / target_total,
        "target_pair_mrr": float(targets["reciprocal_rank"].mean()),
        "best_known_target_rank_min": int(covered_ranks.min()) if not covered_ranks.empty else None,
        "best_known_target_rank_median": float(covered_ranks.median()) if not covered_ranks.empty else None,
        "best_known_target_rank_max": int(covered_ranks.max()) if not covered_ranks.empty else None,
        "known_target_pair_rank_min": int(target_ranks.min()) if not target_ranks.empty else None,
        "known_target_pair_rank_median": float(target_ranks.median()) if not target_ranks.empty else None,
        "known_target_pair_rank_max": int(target_ranks.max()) if not target_ranks.empty else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-csv", required=True, type=Path)
    parser.add_argument(
        "--psichic-root",
        required=True,
        type=Path,
        help=f"Directory containing per-case <case_id>/{SOURCE_TSV_NAME} files.",
    )
    parser.add_argument(
        "--source-template",
        default=DEFAULT_SOURCE_TEMPLATE,
        help=(
            "Relative path below --psichic-root containing exactly one {case_id} "
            f"placeholder (default: {DEFAULT_SOURCE_TEMPLATE})."
        ),
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    out_dir = args.out_dir
    staging = _staging_dir(out_dir)
    if out_dir.exists():
        raise SystemExit(f"Output directory already exists; refusing to overwrite: {out_dir}")
    if not args.psichic_root.exists() or not args.psichic_root.is_dir():
        raise SystemExit(f"PSICHIC root directory is required: {args.psichic_root}")

    panel = _read_panel(args.panel_csv)
    source_template = _validate_source_template(args.source_template)
    _clean_staging(staging)
    try:
        staging.mkdir(parents=True)
        cases, targets, source_records = _audit_panel(
            panel, args.psichic_root, source_template, staging
        )
        cases_csv = staging / "psichic_direct_panel_cases.csv"
        targets_csv = staging / "psichic_direct_panel_targets.csv"
        summary_json = staging / "psichic_direct_panel_summary.json"
        cases.to_csv(cases_csv, index=False)
        targets.to_csv(targets_csv, index=False)
        summary = _build_summary(cases, targets)
        _write_json_atomic(summary, summary_json)
        manifest = {
            "schema_version": "skinscout.psichic-direct-panel-audit.v1",
            "created_at_utc": _utc_now(),
            "inputs": {
                "panel_csv": str(args.panel_csv.resolve()),
                "panel_csv_sha256": _sha256(args.panel_csv),
                "psichic_root": str(args.psichic_root.resolve()),
                "source_template": source_template,
            },
            "score_semantics": SCORE_SEMANTICS,
            "sources": source_records,
            "outputs": {
                "case_audit_csv": _relative_to_benchmark(cases_csv, staging),
                "case_audit_csv_sha256": _sha256(cases_csv),
                "target_audit_csv": _relative_to_benchmark(targets_csv, staging),
                "target_audit_csv_sha256": _sha256(targets_csv),
                "summary_json": _relative_to_benchmark(summary_json, staging),
                "summary_json_sha256": _sha256(summary_json),
            },
            "summary": summary,
        }
        _write_json_atomic(manifest, staging / "manifest.json")
        staging.replace(out_dir)
    except BaseException:
        _clean_staging(staging)
        raise


if __name__ == "__main__":
    main()
