#!/usr/bin/env python3
"""Evaluate known skin compound -> protein target recovery.

The panel is intentionally explicit: each case names one skin-relevant
small molecule, its beneficial/adverse skin context, and one or more known
protein targets. The evaluator scores both case-level recovery (at least one
known target recovered) and target-pair recovery (each known target recovered).
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path

import pandas as pd
from rdkit import Chem


LOG = logging.getLogger("eval.skin_known_target")
SUMMARY_SCHEMA = "skinscout.sota_known_target_recovery.v1"
DEFAULT_CASES = Path("data/validation/skin_known_target_panel.csv")
TARGET_KS = (1, 5, 10, 30)
EVALUATION_MODES = ("retrospective", "leave-query-out", "temporal")
DAINA_RUN_SCHEMA = "skinscout.daina-run.v1"
RANK_COLUMNS = ("final_rank", "rank", "rank_skin", "target_rank", "global_rank")
CANONICAL_RANK_COLUMN = "_skinscout_rank"
SCORE_COLUMNS = (
    "final_skin_weighted",
    "final_score",
    "skin_weighted_score",
    "rrf_score",
    "score",
    "docking_rrf",
    "psichic_score",
    "pred",
    "predicted_affinity",
)
KNOWN_TARGET_PRIOR_COLUMNS = ("known_target_prior", "known_target_prior_norm")
ALLOWED_PANELS = {
    "beneficial",
    "adverse_or_irritant",
    "adverse_or_restricted",
    "beneficial_or_irritant",
    "control",
}
CONTEXT_PROFILES = {
    "auto",
    "general_skin",
    "pigmentation",
    "anti_aging",
    "barrier",
    "acne",
    "inflammation",
    "irritation_sensitization",
}
CONTEXT_KEYWORDS = (
    ("pigmentation", ("whitening", "brightening", "depigment", "tyrosinase")),
    ("anti_aging", ("anti-aging", "collagen", "mmp", "retinoid", "antioxidant")),
    ("barrier", ("barrier", "wound repair", "repair")),
    ("acne", ("acne", "keratolytic", "comedolytic")),
    ("inflammation", ("anti-inflammatory", "inflammatory", "inflammation")),
    ("irritation_sensitization", ("irritant", "sensory", "nociception", "fragrance")),
)


def _write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _write_json_atomic(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _validate_fraction_threshold(value: float, label: str) -> None:
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise SystemExit(f"{label} must be a finite value in [0, 1]: {value}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_evaluation_metadata(
    evaluation_mode: str,
    evidence_snapshot: str,
    cutoff_date: str,
) -> None:
    if cutoff_date:
        try:
            date.fromisoformat(cutoff_date)
        except ValueError as exc:
            raise SystemExit(
                f"--cutoff-date must use YYYY-MM-DD format: {cutoff_date}"
            ) from exc
    if evaluation_mode == "temporal" and (not evidence_snapshot or not cutoff_date):
        raise SystemExit(
            "--evaluation-mode temporal requires both --evidence-snapshot and "
            "--cutoff-date; temporal evaluation fails closed without explicit evidence "
            "snapshot metadata"
        )
    if evaluation_mode == "leave-query-out" and not evidence_snapshot:
        raise SystemExit(
            "--evaluation-mode leave-query-out requires --evidence-snapshot; "
            "leakage-controlled evaluation fails closed without an auditable snapshot"
        )
    if evaluation_mode != "temporal" and cutoff_date:
        raise SystemExit(
            "--cutoff-date is only valid with --evaluation-mode temporal"
        )


def _effect_direction(panel: object) -> str:
    text = str(panel).strip()
    if text == "beneficial":
        return "beneficial"
    if text in {"adverse_or_irritant", "adverse_or_restricted"}:
        return "adverse"
    if text == "beneficial_or_irritant":
        return "mixed"
    if text == "control":
        return "control"
    return "unknown"


def _infer_context_profile(skin_effect: object) -> str:
    text = str(skin_effect).strip().lower()
    for profile, keywords in CONTEXT_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return profile
    return "general_skin"


def _case_context_profile(row: pd.Series, requested_profile: str) -> str:
    if requested_profile != "auto":
        return requested_profile
    return _infer_context_profile(row["skin_effect"])


def _evidence_grade(row: pd.Series) -> str:
    note = str(row.get("evidence_note", "")).strip().lower()
    if "positive control" in note or "canonical" in note:
        return "A_known_target_control"
    if note:
        return "B_literature_panel"
    return "C_panel_assertion"


def _validate_nonempty_strings(df: pd.DataFrame, column: str, label: str) -> None:
    invalid = [
        int(idx)
        for idx, value in df[column].items()
        if pd.isna(value) or not str(value).strip()
    ]
    if invalid:
        shown = ", ".join(str(idx) for idx in invalid[:10])
        suffix = "..." if len(invalid) > 10 else ""
        raise SystemExit(
            f"{label} column '{column}' contains blank values at row index(es) "
            f"{shown}{suffix}"
        )
    df[column] = df[column].astype(str).str.strip()


def _validate_unique_strings(df: pd.DataFrame, column: str, label: str) -> None:
    normalized = df[column].astype(str).str.strip()
    duplicate_ids = normalized[normalized.duplicated()].tolist()
    if duplicate_ids:
        shown = ", ".join(str(value) for value in duplicate_ids[:10])
        suffix = "..." if len(duplicate_ids) > 10 else ""
        raise SystemExit(f"{label} contains duplicate {column} values: {shown}{suffix}")
    df[column] = normalized


def _validate_smiles(df: pd.DataFrame, column: str, label: str) -> None:
    invalid: list[int] = []
    normalized: list[str] = []
    for idx, value in df[column].items():
        text = str(value).strip()
        if Chem.MolFromSmiles(text) is None:
            invalid.append(int(idx))
        normalized.append(text)
    if invalid:
        shown = ", ".join(str(idx) for idx in invalid[:10])
        suffix = "..." if len(invalid) > 10 else ""
        raise SystemExit(
            f"{label} column '{column}' contains invalid SMILES at row index(es) "
            f"{shown}{suffix}"
        )
    df[column] = normalized


def _parse_semicolon_list(value: object, label: str, case_id: str) -> list[str]:
    tokens = [token.strip() for token in str(value).split(";")]
    if any(not token for token in tokens):
        raise SystemExit(f"{label} contains empty ';'-separated value(s): {case_id}")
    duplicates = sorted({token for token in tokens if tokens.count(token) > 1})
    if duplicates:
        shown = ", ".join(duplicates[:10])
        suffix = "..." if len(duplicates) > 10 else ""
        raise SystemExit(
            f"{label} contains duplicate ';'-separated value(s): "
            f"{case_id}: {shown}{suffix}"
        )
    return tokens


def _read_cases(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"Skin known-target panel is required and must be non-empty: {path}")
    try:
        cases = pd.read_csv(path)
    except Exception as exc:
        raise SystemExit(f"Skin known-target panel failed to parse: {path}: {exc}") from exc
    required = {"case_id", "inci_name", "panel", "skin_effect", "smiles", "known_targets"}
    missing = sorted(required - set(cases.columns))
    if missing:
        raise SystemExit(f"Skin known-target panel missing required columns {missing}: {path}")
    if cases.empty:
        raise SystemExit(f"Skin known-target panel contains no rows: {path}")
    for column in sorted(required):
        _validate_nonempty_strings(cases, column, "Skin known-target panel")
    _validate_unique_strings(cases, "case_id", "Skin known-target panel")
    _validate_smiles(cases, "smiles", "Skin known-target panel")
    invalid_panels = sorted(set(cases["panel"].astype(str)) - ALLOWED_PANELS)
    if invalid_panels:
        raise SystemExit(
            "Skin known-target panel column 'panel' contains unsupported value(s): "
            + ", ".join(invalid_panels)
        )
    if "known_target_labels" in cases.columns:
        _validate_nonempty_strings(cases, "known_target_labels", "Skin known-target panel")
        for _, row in cases.iterrows():
            case_id = str(row["case_id"])
            targets = _parse_semicolon_list(
                row["known_targets"],
                "Skin known-target panel column 'known_targets'",
                case_id,
            )
            labels = _parse_semicolon_list(
                row["known_target_labels"],
                "Skin known-target panel column 'known_target_labels'",
                case_id,
            )
            if len(labels) != len(targets):
                raise SystemExit(
                    "Skin known-target panel known_target_labels count must match "
                    f"known_targets count: {case_id}"
                )
    return cases


def _read_ranking(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"Known-target ranking is required and must be non-empty: {path}")
    try:
        ranking = pd.read_csv(path)
    except Exception as exc:
        raise SystemExit(f"Known-target ranking failed to parse: {path}: {exc}") from exc
    if "target_id" not in ranking.columns:
        raise SystemExit(f"Known-target ranking missing required column 'target_id': {path}")
    if ranking.empty:
        raise SystemExit(f"Known-target ranking contains no rows: {path}")
    _validate_nonempty_strings(ranking, "target_id", "Known-target ranking")
    duplicate_ids = ranking["target_id"][ranking["target_id"].duplicated()].tolist()
    if duplicate_ids:
        shown = ", ".join(str(value) for value in duplicate_ids[:10])
        suffix = "..." if len(duplicate_ids) > 10 else ""
        raise SystemExit(
            f"Known-target ranking contains duplicate target_id values: "
            f"{shown}{suffix}: {path}"
        )
    return _ordered_ranking(ranking, path, "Known-target ranking")


def _has_nonzero_known_target_prior(ranking: pd.DataFrame, path: Path) -> bool:
    missing_columns = [
        column for column in KNOWN_TARGET_PRIOR_COLUMNS if column not in ranking.columns
    ]
    if missing_columns:
        raise SystemExit(
            "Known-target ranking is missing required prior-audit column(s) "
            f"{missing_columns}; unbiased recovery claims require explicit zero-valued "
            f"audit columns: {path}"
        )
    assisted = False
    for column in KNOWN_TARGET_PRIOR_COLUMNS:
        raw = ranking[column]
        nonblank = ~raw.isna() & raw.astype(str).str.strip().ne("")
        if not nonblank.all():
            first = int(nonblank[~nonblank].index[0])
            raise SystemExit(
                f"Known-target ranking column '{column}' contains a blank prior-audit "
                f"value at row index {first}: {path}"
            )
        values = pd.to_numeric(raw, errors="coerce")
        invalid = values.isna() | ~values.fillna(0.0).map(math.isfinite)
        if invalid.any():
            first = int(invalid[invalid].index[0])
            raise SystemExit(
                f"Known-target ranking column '{column}' contains non-numeric "
                f"assisted prior value at row index {first}: {path}"
            )
        out_of_range = nonblank & ((values < 0.0) | (values > 1.0))
        if out_of_range.any():
            first = int(out_of_range[out_of_range].index[0])
            raise SystemExit(
                f"Known-target ranking column '{column}' must be in [0, 1] "
                f"at row index {first}: {path}"
            )
        assisted = assisted or bool(values.ne(0.0).any())
    return assisted


def _numeric_series(df: pd.DataFrame, column: str, label: str, path: Path) -> pd.Series:
    if pd.api.types.is_bool_dtype(df[column].dtype) or df[column].map(
        lambda value: isinstance(value, bool)
    ).any():
        raise SystemExit(
            f"{label} column '{column}' must not contain boolean values: {path}"
        )
    values = pd.to_numeric(df[column], errors="coerce")
    invalid = values.isna() | ~values.map(math.isfinite)
    if invalid.any():
        first = int(invalid[invalid].index[0])
        raise SystemExit(
            f"{label} column '{column}' contains non-finite numeric value "
            f"at row index {first}: {path}"
        )
    return values


def _ordered_ranking(ranking: pd.DataFrame, path: Path, label: str) -> pd.DataFrame:
    for column in RANK_COLUMNS:
        if column not in ranking.columns:
            continue
        values = _numeric_series(ranking, column, label, path)
        if ((values % 1) != 0).any() or (values < 1).any():
            raise SystemExit(f"{label} column '{column}' must contain positive integer ranks: {path}")
        if values.duplicated().any():
            duplicate = values[values.duplicated()].iloc[0]
            raise SystemExit(f"{label} column '{column}' contains duplicate rank {int(duplicate)}: {path}")
        ranking = ranking.copy()
        ranking[column] = values.astype(int)
        ordered = ranking.sort_values(
            [column, "target_id"], ascending=[True, True]
        ).reset_index(drop=True)
        # Honour the declared rank. Renumbering by row position would silently
        # compress a subset export (for example a global_rank slice) into a
        # dense 1..N sequence and report better ranks than were produced.
        ordered[CANONICAL_RANK_COLUMN] = ordered[column].astype(int)
        return ordered
    for column in SCORE_COLUMNS:
        if column not in ranking.columns:
            continue
        values = _numeric_series(ranking, column, label, path)
        ranking = ranking.copy()
        ranking[column] = values
        ordered = ranking.sort_values(
            [column, "target_id"], ascending=[False, True]
        ).reset_index(drop=True)
        ordered[CANONICAL_RANK_COLUMN] = range(1, len(ordered) + 1)
        return ordered
    raise SystemExit(
        f"{label} missing ranking order column; expected one of "
        f"{list(RANK_COLUMNS)} or score column {list(SCORE_COLUMNS)}: {path}"
    )


def _ranking_candidates(rankings_dir: Path, case_id: str, inci_name: str) -> list[Path]:
    stems = []
    for value in (case_id, inci_name):
        normalized = str(value).strip().replace(" ", "_")
        if normalized and normalized not in stems:
            stems.append(normalized)
    suffixes = (
        "__ranked_targets_v3_with_efficacy.csv",
        "__ranked_targets_v3.csv",
    )
    return [rankings_dir / f"{stem}{suffix}" for stem in stems for suffix in suffixes]


def _find_ranking(rankings_dir: Path, case_id: str, inci_name: str) -> Path | None:
    for path in _ranking_candidates(rankings_dir, case_id, inci_name):
        if path.exists():
            return path
    return None


def _ranking_metadata_path(ranking_path: Path) -> Path:
    return ranking_path.with_suffix(".metadata.json")


def _validate_ranking_metadata(
    ranking_path: Path,
    *,
    case_id: str,
    evaluation_mode: str,
    evidence_snapshot: str,
    cutoff_date: str,
) -> dict[str, str]:
    metadata_path = _ranking_metadata_path(ranking_path)
    if not metadata_path.exists() or metadata_path.stat().st_size == 0:
        raise SystemExit(
            "Leakage-controlled known-target evaluation requires a non-empty "
            f"Daina ranking metadata sidecar: {metadata_path}"
        )
    try:
        payload = json.loads(metadata_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise SystemExit(
            f"Unable to parse Daina ranking metadata sidecar: {metadata_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise SystemExit(
            f"Daina ranking metadata sidecar must contain a JSON object: {metadata_path}"
        )
    if payload.get("schema_version") != DAINA_RUN_SCHEMA:
        raise SystemExit(
            "Daina ranking metadata schema mismatch; expected "
            f"{DAINA_RUN_SCHEMA}: {metadata_path}"
        )

    expected_evidence_mode = (
        "retrieval" if evaluation_mode == "retrospective" else evaluation_mode
    )
    actual_evidence_mode = str(payload.get("evidence_mode", "")).strip()
    if actual_evidence_mode != expected_evidence_mode:
        raise SystemExit(
            "Daina ranking evidence_mode does not match evaluator mode: "
            f"expected={expected_evidence_mode!r} actual={actual_evidence_mode!r}: "
            f"{metadata_path}"
        )
    actual_snapshot = str(payload.get("evidence_snapshot_id") or "").strip()
    if actual_snapshot != evidence_snapshot:
        raise SystemExit(
            "Daina ranking evidence snapshot does not match evaluator snapshot: "
            f"expected={evidence_snapshot!r} actual={actual_snapshot!r}: {metadata_path}"
        )
    actual_cutoff = str(payload.get("cutoff_date") or "").strip()
    if actual_cutoff != cutoff_date:
        raise SystemExit(
            "Daina ranking cutoff date does not match evaluator cutoff date: "
            f"expected={cutoff_date!r} actual={actual_cutoff!r}: {metadata_path}"
        )
    if payload.get("score_is_calibrated_probability") is not False:
        raise SystemExit(
            "Daina similarity score must be explicitly labeled as not a calibrated "
            f"probability: {metadata_path}"
        )
    metadata_case_id = str(payload.get("case_id") or "").strip()
    if metadata_case_id and metadata_case_id != case_id:
        raise SystemExit(
            "Daina ranking metadata case_id does not match the evaluated case: "
            f"expected={case_id!r} actual={metadata_case_id!r}: {metadata_path}"
        )

    expected_ranking_sha = str(payload.get("ranking_sha256") or "").strip().lower()
    actual_ranking_sha = _sha256(ranking_path)
    if expected_ranking_sha != actual_ranking_sha:
        raise SystemExit(
            "Daina ranking SHA-256 does not match its metadata sidecar: "
            f"expected={expected_ranking_sha or '<missing>'} "
            f"actual={actual_ranking_sha}: {metadata_path}"
        )
    return {
        "ranking_metadata_path": str(metadata_path),
        "ranking_metadata_sha256": _sha256(metadata_path),
        "ranking_sha256": actual_ranking_sha,
        "ranking_evidence_mode": actual_evidence_mode,
        "ranking_evidence_snapshot": actual_snapshot,
        "ranking_cutoff_date": actual_cutoff,
    }


def _target_rank_map(ranking: pd.DataFrame) -> dict[str, int]:
    return {
        str(target_id): int(rank)
        for target_id, rank in zip(
            ranking["target_id"].astype(str),
            ranking[CANONICAL_RANK_COLUMN],
            strict=True,
        )
    }


def _target_metadata(ranking: pd.DataFrame, target_id: str) -> dict[str, object]:
    row = ranking[ranking["target_id"].astype(str) == target_id]
    if row.empty:
        return {}
    record = row.iloc[0]
    out: dict[str, object] = {}
    for column in (
        "final_score",
        "docking_rrf",
        "skin_score",
        "skin_tier",
        "source_count",
        "sources",
        "gene_symbol",
        "protein_name",
    ):
        if column in ranking.columns and not pd.isna(record[column]):
            out[column] = record[column]
    return out


def _labels_for(row: pd.Series, targets: list[str]) -> list[str]:
    if "known_target_labels" not in row.index or pd.isna(row["known_target_labels"]):
        return targets
    labels = _parse_semicolon_list(
        row["known_target_labels"],
        "Skin known-target panel column 'known_target_labels'",
        str(row["case_id"]),
    )
    if len(labels) != len(targets):
        raise SystemExit(
            "Skin known-target panel known_target_labels count must match "
            f"known_targets count: {row['case_id']}"
        )
    return labels


def _case_and_target_rows(
    cases: pd.DataFrame,
    rankings_dir: Path,
    *,
    allow_missing_rankings: bool,
    allow_assisted_known_target_prior: bool,
    context_profile: str,
    evaluation_mode: str,
    evidence_snapshot: str,
    cutoff_date: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[Path]]:
    case_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []
    missing_rankings: list[Path] = []
    for _, case in cases.iterrows():
        case_id = str(case["case_id"]).strip()
        inci = str(case["inci_name"]).strip()
        case_context = _case_context_profile(case, context_profile)
        effect_direction = _effect_direction(case["panel"])
        evidence_grade = _evidence_grade(case)
        evidence_note = str(case.get("evidence_note", "")).strip()
        targets = _parse_semicolon_list(
            case["known_targets"],
            "Skin known-target panel column 'known_targets'",
            case_id,
        )
        labels = _labels_for(case, targets)
        ranking_path = _find_ranking(rankings_dir, case_id, inci)
        if ranking_path is None:
            expected = _ranking_candidates(rankings_dir, case_id, inci)[0]
            missing_rankings.append(expected)
            status = "no_ranking"
            ranks: dict[str, int] = {}
            ranking = pd.DataFrame()
            ranking_audit = {
                "ranking_metadata_path": "",
                "ranking_metadata_sha256": "",
                "ranking_sha256": "",
                "ranking_evidence_mode": "",
                "ranking_evidence_snapshot": "",
                "ranking_cutoff_date": "",
            }
        else:
            status = "evaluated"
            ranking = _read_ranking(ranking_path)
            if evaluation_mode == "retrospective":
                ranking_audit = {
                    "ranking_metadata_path": "",
                    "ranking_metadata_sha256": "",
                    "ranking_sha256": _sha256(ranking_path),
                    "ranking_evidence_mode": "retrospective-unverified",
                    "ranking_evidence_snapshot": "",
                    "ranking_cutoff_date": "",
                }
            else:
                ranking_audit = _validate_ranking_metadata(
                    ranking_path,
                    case_id=case_id,
                    evaluation_mode=evaluation_mode,
                    evidence_snapshot=evidence_snapshot,
                    cutoff_date=cutoff_date,
                )
            assisted_by_prior = _has_nonzero_known_target_prior(ranking, ranking_path)
            if assisted_by_prior and not allow_assisted_known_target_prior:
                raise SystemExit(
                    "Known-target ranking contains nonzero assisted known-target prior "
                    f"column(s); unbiased recovery claims require rerunning without "
                    f"known_target_prior/known_target_prior_norm or passing "
                    f"--allow-assisted-known-target-prior for diagnostics only: {ranking_path}"
                )
            ranks = _target_rank_map(ranking)
        if ranking_path is None:
            assisted_by_prior = False

        target_hit_flags = {k: [] for k in TARGET_KS}
        known_rank_pairs: list[str] = []
        recovered_top10: list[str] = []
        finite_ranks: list[int] = []
        reciprocal_ranks: list[float] = []
        best_target = ""
        best_rank: int | None = None
        for target_id, label in zip(targets, labels, strict=True):
            rank = ranks.get(target_id)
            known_rank_pairs.append(f"{target_id}:{rank if rank is not None else 'NA'}")
            has_finite_rank = rank is not None
            reciprocal_rank = 1.0 / rank if rank is not None else 0.0
            reciprocal_ranks.append(reciprocal_rank)
            if rank is not None:
                finite_ranks.append(rank)
            if rank is not None and (best_rank is None or rank < best_rank):
                best_rank = rank
                best_target = target_id
            if rank is not None and rank <= 10:
                recovered_top10.append(target_id)
            metadata = _target_metadata(ranking, target_id) if status == "evaluated" else {}
            target_row: dict[str, object] = {
                "case_id": case_id,
                "inci_name": inci,
                "panel": case["panel"],
                "skin_effect": case["skin_effect"],
                "context_profile": case_context,
                "effect_direction": effect_direction,
                "evidence_grade": evidence_grade,
                "evidence_note": evidence_note,
                "assisted_by_known_target_prior": assisted_by_prior,
                "target_id": target_id,
                "target_label": label,
                "status": status,
                "rank": rank if rank is not None else "",
                "has_finite_rank": int(has_finite_rank),
                "reciprocal_rank": reciprocal_rank,
                **ranking_audit,
            }
            for k in TARGET_KS:
                hit = int(rank is not None and rank <= k)
                target_hit_flags[k].append(hit)
                target_row[f"target_top{k}"] = hit
            target_row.update(metadata)
            target_rows.append(target_row)

        case_row: dict[str, object] = {
            "case_id": case_id,
            "inci_name": inci,
            "panel": case["panel"],
            "skin_effect": case["skin_effect"],
            "context_profile": case_context,
            "effect_direction": effect_direction,
            "evidence_grade": evidence_grade,
            "evidence_note": evidence_note,
            "assisted_by_known_target_prior": assisted_by_prior,
            "status": status,
            "ranking_path": str(ranking_path) if ranking_path is not None else "",
            **ranking_audit,
            "n_known_targets": len(targets),
            "n_ranked_known_targets": len(finite_ranks),
            "target_pair_mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
            "mean_finite_rank": float(pd.Series(finite_ranks).mean()) if finite_ranks else "",
            "median_finite_rank": float(pd.Series(finite_ranks).median()) if finite_ranks else "",
            "ranked_target_coverage": len(finite_ranks) / len(targets),
            "case_coverage": int(status == "evaluated"),
            "best_known_target_rank": best_rank if best_rank is not None else "",
            "best_known_target_id": best_target,
            "known_target_ranks": ";".join(known_rank_pairs),
            "recovered_top10_targets": ";".join(recovered_top10),
        }
        for k in TARGET_KS:
            hits = target_hit_flags[k]
            case_row[f"case_top{k}"] = int(any(hits))
            case_row[f"target_top{k}_fraction"] = sum(hits) / len(hits)
        case_rows.append(case_row)

    if missing_rankings and not allow_missing_rankings:
        preview = ", ".join(str(path) for path in missing_rankings[:5])
        raise SystemExit(
            f"Missing {len(missing_rankings)} skin known-target ranking file(s); "
            f"pass --allow-missing-rankings for diagnostics only: {preview}"
        )
    return case_rows, target_rows, missing_rankings


def _apply_thresholds(
    case_df: pd.DataFrame,
    target_df: pd.DataFrame,
    *,
    min_case_top10: float,
    min_target_top10: float,
    min_target_top30: float,
    allow_threshold_failure: bool,
) -> tuple[bool, list[str]]:
    evaluated_cases = case_df[case_df["status"] == "evaluated"]
    evaluated_targets = target_df[target_df["status"] == "evaluated"]
    failures: list[str] = []
    if evaluated_cases.empty:
        failures.append("n_evaluated_cases=0")
    if evaluated_targets.empty:
        failures.append("n_evaluated_targets=0")
    case_top10 = float(evaluated_cases["case_top10"].mean()) if not evaluated_cases.empty else 0.0
    target_top10 = float(evaluated_targets["target_top10"].mean()) if not evaluated_targets.empty else 0.0
    target_top30 = float(evaluated_targets["target_top30"].mean()) if not evaluated_targets.empty else 0.0
    if case_top10 < min_case_top10:
        failures.append(f"case_top10={case_top10:.3f} < {min_case_top10:.3f}")
    if target_top10 < min_target_top10:
        failures.append(f"target_top10={target_top10:.3f} < {min_target_top10:.3f}")
    if target_top30 < min_target_top30:
        failures.append(f"target_top30={target_top30:.3f} < {min_target_top30:.3f}")
    passes = not failures
    if failures and not allow_threshold_failure:
        raise SystemExit(
            "Skin known-target recovery failed claim threshold(s): "
            + "; ".join(failures)
            + "; pass --allow-threshold-failure only for explicit diagnostics"
        )
    return passes, failures


def _mean_fraction(df: pd.DataFrame, column: str) -> float:
    if df.empty:
        return 0.0
    values = pd.to_numeric(df[column], errors="coerce")
    if values.isna().any():
        raise SystemExit(f"Skin known-target recovery column '{column}' must be numeric")
    return float(values.mean())


def _finite_rank_metrics(target_df: pd.DataFrame) -> dict[str, object]:
    if target_df.empty:
        return {
            "target_pair_mrr": 0.0,
            "mean_finite_rank": 0.0,
            "median_finite_rank": 0.0,
            "ranked_target_coverage": 0.0,
            "ranked_target_coverage_numerator": 0,
            "ranked_target_coverage_denominator": 0,
            "target_pair_mrr_denominator": 0,
        }
    rank_values = pd.to_numeric(target_df["rank"], errors="coerce")
    finite = rank_values.dropna()
    reciprocal = rank_values.map(lambda value: 0.0 if pd.isna(value) else 1.0 / float(value))
    return {
        "target_pair_mrr": float(reciprocal.mean()),
        "mean_finite_rank": float(finite.mean()) if not finite.empty else 0.0,
        "median_finite_rank": float(finite.median()) if not finite.empty else 0.0,
        "ranked_target_coverage": float(len(finite) / len(target_df)),
        "ranked_target_coverage_numerator": int(len(finite)),
        "ranked_target_coverage_denominator": int(len(target_df)),
        "target_pair_mrr_denominator": int(len(target_df)),
    }


def _evaluated_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "status" not in frame.columns:
        return frame
    return frame[frame["status"] == "evaluated"]


def _coverage_metrics(case_df: pd.DataFrame, target_df: pd.DataFrame) -> dict[str, object]:
    # target_top{k} averages over evaluated rows only, so the rank metrics must
    # use the same population; otherwise one summary block mixes denominators
    # whenever a case has no ranking file.
    ranked = _finite_rank_metrics(_evaluated_rows(target_df))
    evaluated_cases = case_df[case_df["status"] == "evaluated"]
    case_denominator = len(case_df)
    ranked.update(
        {
            "case_coverage": float(len(evaluated_cases) / case_denominator)
            if case_denominator
            else 0.0,
            "case_coverage_numerator": int(len(evaluated_cases)),
            "case_coverage_denominator": int(case_denominator),
        }
    )
    return ranked


def _group_metrics(
    case_df: pd.DataFrame,
    target_df: pd.DataFrame,
    group_column: str,
) -> dict[str, dict[str, object]]:
    groups: dict[str, dict[str, object]] = {}
    for value in sorted(case_df[group_column].astype(str).unique()):
        case_subset = case_df[case_df[group_column].astype(str) == value]
        target_subset = target_df[target_df[group_column].astype(str) == value]
        evaluated_cases = case_subset[case_subset["status"] == "evaluated"]
        evaluated_targets = target_subset[target_subset["status"] == "evaluated"]
        coverage = _coverage_metrics(case_subset, target_subset)
        groups[value] = {
            "n_cases": int(len(case_subset)),
            "n_targets": int(len(target_subset)),
            "n_evaluated_cases": int(len(evaluated_cases)),
            "n_evaluated_targets": int(len(evaluated_targets)),
            "case_top10": _mean_fraction(evaluated_cases, "case_top10"),
            "target_top10": _mean_fraction(evaluated_targets, "target_top10"),
            "target_top30": _mean_fraction(evaluated_targets, "target_top30"),
            "target_pair_mrr": coverage["target_pair_mrr"],
            "mean_finite_rank": coverage["mean_finite_rank"],
            "median_finite_rank": coverage["median_finite_rank"],
            "ranked_target_coverage": coverage["ranked_target_coverage"],
            "case_coverage": coverage["case_coverage"],
        }
    return groups


def _summary_payload(
    *,
    cases_path: Path,
    rankings_dir: Path,
    out_csv: Path,
    out_target_csv: Path | None,
    case_df: pd.DataFrame,
    target_df: pd.DataFrame,
    context_profile: str,
    min_case_top10: float,
    min_target_top10: float,
    min_target_top30: float,
    evaluation_mode: str,
    evidence_snapshot: str,
    cutoff_date: str,
    passes: bool,
    failures: list[str],
    assisted_by_known_target_prior: bool,
) -> dict[str, object]:
    evaluated_cases = case_df[case_df["status"] == "evaluated"]
    evaluated_targets = target_df[target_df["status"] == "evaluated"]
    coverage = _coverage_metrics(case_df, target_df)
    ranking_audit_columns = [
        "case_id",
        "ranking_path",
        "ranking_sha256",
        "ranking_metadata_path",
        "ranking_metadata_sha256",
        "ranking_evidence_mode",
        "ranking_evidence_snapshot",
        "ranking_cutoff_date",
    ]
    ranking_evidence_audit = evaluated_cases.loc[
        :, ranking_audit_columns
    ].to_dict("records")
    return {
        "schema_version": SUMMARY_SCHEMA,
        "created_at_utc": _utc_now(),
        "context_profile": context_profile,
        "evaluation_mode": evaluation_mode,
        "evaluation_metadata": {
            "mode": evaluation_mode,
            "evidence_snapshot": evidence_snapshot,
            "cutoff_date": cutoff_date,
        },
        "inputs": {
            "cases_csv": str(cases_path),
            "rankings_dir": str(rankings_dir),
        },
        "ranking_evidence_audit": ranking_evidence_audit,
        "outputs": {
            "case_metrics_csv": str(out_csv),
            "target_metrics_csv": str(out_target_csv) if out_target_csv else "",
        },
        "assisted_by_known_target_prior": assisted_by_known_target_prior,
        "thresholds": {
            "min_case_top10": min_case_top10,
            "min_target_top10": min_target_top10,
            "min_target_top30": min_target_top30,
        },
        "metrics": {
            "n_cases": int(len(case_df)),
            "n_targets": int(len(target_df)),
            "n_evaluated_cases": int(len(evaluated_cases)),
            "n_evaluated_targets": int(len(evaluated_targets)),
            "case_top10": _mean_fraction(evaluated_cases, "case_top10"),
            "target_top10": _mean_fraction(evaluated_targets, "target_top10"),
            "target_top30": _mean_fraction(evaluated_targets, "target_top30"),
            "target_pair_mrr": coverage["target_pair_mrr"],
            "mean_finite_rank": coverage["mean_finite_rank"],
            "median_finite_rank": coverage["median_finite_rank"],
            "ranked_target_coverage": coverage["ranked_target_coverage"],
            "case_coverage": coverage["case_coverage"],
        },
        "denominators": {
            "target_pair_mrr": coverage["target_pair_mrr_denominator"],
            "mean_finite_rank": coverage["ranked_target_coverage_numerator"],
            "median_finite_rank": coverage["ranked_target_coverage_numerator"],
            "ranked_target_coverage_numerator": coverage["ranked_target_coverage_numerator"],
            "ranked_target_coverage_denominator": coverage["ranked_target_coverage_denominator"],
            "case_coverage_numerator": coverage["case_coverage_numerator"],
            "case_coverage_denominator": coverage["case_coverage_denominator"],
        },
        "by_panel": _group_metrics(case_df, target_df, "panel"),
        "by_context_profile": _group_metrics(case_df, target_df, "context_profile"),
        "passes_threshold": passes,
        "threshold_failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-csv", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--rankings-dir", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--out-target-csv", type=Path)
    parser.add_argument("--out-summary-json", type=Path)
    parser.add_argument(
        "--evaluation-mode",
        choices=EVALUATION_MODES,
        default="retrospective",
        help="Evaluation design metadata; temporal mode requires snapshot and cutoff metadata.",
    )
    parser.add_argument(
        "--evidence-snapshot",
        default="",
        help="Evidence snapshot path or identifier used for leave-query-out/temporal diagnostics.",
    )
    parser.add_argument(
        "--cutoff-date",
        default="",
        help="YYYY-MM-DD evidence cutoff date; required for temporal evaluation.",
    )
    parser.add_argument(
        "--context-profile",
        choices=sorted(CONTEXT_PROFILES),
        default="auto",
        help="Requested skin context for SOTA evidence reporting; auto infers per case.",
    )
    parser.add_argument("--min-case-top10", type=float, default=0.01)
    parser.add_argument("--min-target-top10", type=float, default=0.01)
    parser.add_argument("--min-target-top30", type=float, default=0.01)
    parser.add_argument(
        "--allow-missing-rankings",
        action="store_true",
        help="write diagnostic no_ranking rows instead of failing on missing rankings",
    )
    parser.add_argument(
        "--allow-threshold-failure",
        action="store_true",
        help="write failing recovery metrics only for explicit diagnostics",
    )
    parser.add_argument(
        "--allow-assisted-known-target-prior",
        action="store_true",
        help=(
            "allow ranking files with nonzero known_target_prior columns; "
            "diagnostic only, not an unbiased recovery claim"
        ),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.out_csv.exists():
        args.out_csv.unlink()
    if args.out_target_csv and args.out_target_csv.exists():
        args.out_target_csv.unlink()
    if args.out_summary_json and args.out_summary_json.exists():
        args.out_summary_json.unlink()
    _validate_evaluation_metadata(
        args.evaluation_mode,
        str(args.evidence_snapshot).strip(),
        str(args.cutoff_date).strip(),
    )
    for value, label in (
        (args.min_case_top10, "--min-case-top10"),
        (args.min_target_top10, "--min-target-top10"),
        (args.min_target_top30, "--min-target-top30"),
    ):
        _validate_fraction_threshold(value, label)
    cases = _read_cases(args.cases_csv)
    case_rows, target_rows, _ = _case_and_target_rows(
        cases,
        args.rankings_dir,
        allow_missing_rankings=args.allow_missing_rankings,
        allow_assisted_known_target_prior=args.allow_assisted_known_target_prior,
        context_profile=args.context_profile,
        evaluation_mode=args.evaluation_mode,
        evidence_snapshot=str(args.evidence_snapshot).strip(),
        cutoff_date=str(args.cutoff_date).strip(),
    )
    case_df = pd.DataFrame(case_rows)
    target_df = pd.DataFrame(target_rows)
    passes, failures = _apply_thresholds(
        case_df,
        target_df,
        min_case_top10=args.min_case_top10,
        min_target_top10=args.min_target_top10,
        min_target_top30=args.min_target_top30,
        allow_threshold_failure=args.allow_threshold_failure,
    )
    assisted_by_known_target_prior = bool(
        case_df.get("assisted_by_known_target_prior", pd.Series(dtype=bool)).astype(bool).any()
    )
    case_df["passes_threshold"] = passes
    case_df["threshold_failures"] = ";".join(failures)
    case_df["evaluation_mode"] = args.evaluation_mode
    case_df["evidence_snapshot"] = str(args.evidence_snapshot).strip()
    case_df["cutoff_date"] = str(args.cutoff_date).strip()
    case_df["assisted_prior_diagnostic"] = assisted_by_known_target_prior
    target_df["passes_threshold"] = passes
    target_df["threshold_failures"] = ";".join(failures)
    target_df["evaluation_mode"] = args.evaluation_mode
    target_df["evidence_snapshot"] = str(args.evidence_snapshot).strip()
    target_df["cutoff_date"] = str(args.cutoff_date).strip()
    target_df["assisted_prior_diagnostic"] = assisted_by_known_target_prior
    _write_csv_atomic(case_df, args.out_csv)
    if args.out_target_csv is not None:
        _write_csv_atomic(target_df, args.out_target_csv)
    if args.out_summary_json is not None:
        _write_json_atomic(
            _summary_payload(
                cases_path=args.cases_csv,
                rankings_dir=args.rankings_dir,
                out_csv=args.out_csv,
                out_target_csv=args.out_target_csv,
                case_df=case_df,
                target_df=target_df,
                context_profile=args.context_profile,
                min_case_top10=args.min_case_top10,
                min_target_top10=args.min_target_top10,
                min_target_top30=args.min_target_top30,
                evaluation_mode=args.evaluation_mode,
                evidence_snapshot=str(args.evidence_snapshot).strip(),
                cutoff_date=str(args.cutoff_date).strip(),
                passes=passes,
                failures=failures,
                assisted_by_known_target_prior=assisted_by_known_target_prior,
            ),
            args.out_summary_json,
        )
    evaluated_cases = case_df[case_df["status"] == "evaluated"]
    evaluated_targets = target_df[target_df["status"] == "evaluated"]
    LOG.info(
        "Skin known-target recovery: cases top10=%.2f targets top10=%.2f top30=%.2f",
        float(evaluated_cases["case_top10"].mean()) if not evaluated_cases.empty else 0.0,
        float(evaluated_targets["target_top10"].mean()) if not evaluated_targets.empty else 0.0,
        float(evaluated_targets["target_top30"].mean()) if not evaluated_targets.empty else 0.0,
    )


if __name__ == "__main__":
    main()
