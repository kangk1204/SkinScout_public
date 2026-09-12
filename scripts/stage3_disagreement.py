#!/usr/bin/env python3
"""stage3_disagreement.py — Docking-vs-DTI disagreement & sanity report
(INSTRUCTIONS.md §6.2 (d)).

Outputs:
  • overlap@50 between docking top50 and DTI top50
  • docking-only set : docking top50 ∩ DTI bottom 50%  (bias-free signal)
  • dti-only set     : DTI top50      ∩ docking bottom 50% (DTI overfit cue)
  • per-class breakdown when receptor → class mapping is available in the
    ChEMBL mirror (kinase / GPCR / NR / ion channel / protease / orphan).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import pandas as pd

LOG = logging.getLogger("stage3.disagreement")


def _remove_outputs(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def _write_json_atomic(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def read_required_csv(
    path: Path,
    sep: str,
    required_cols: set[str],
    label: str,
) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"{label} is required and must be non-empty: {path}")
    try:
        df = pd.read_csv(path, sep=sep)
    except Exception as exc:
        raise SystemExit(f"{label} failed to parse: {path}: {exc}") from exc
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise SystemExit(f"{label} missing required columns {missing}: {path}")
    if df.empty:
        raise SystemExit(f"{label} contains no rows: {path}")
    return df


def _invalid_row_message(label: str, column: str, rows: list[int], detail: str) -> str:
    shown = ", ".join(str(idx) for idx in rows[:10])
    suffix = "..." if len(rows) > 10 else ""
    return (
        f"{label} column '{column}' contains {detail} at row "
        f"index(es) {shown}{suffix}"
    )


def _bool_like_indexes(series: pd.Series) -> list[int]:
    return [
        int(idx)
        for idx, value in series.items()
        if (
            isinstance(value, bool)
            or type(value).__name__ == "bool_"
            or (
                isinstance(value, str)
                and value.strip().lower() in {"true", "false"}
            )
        )
    ]


def rank_targets(df: pd.DataFrame, score_col: str, label: str) -> list[str]:
    df = _normalize_target_ids(df, label)
    bool_scores = _bool_like_indexes(df[score_col])
    if bool_scores:
        raise SystemExit(
            _invalid_row_message(label, score_col, bool_scores, "invalid values")
        )
    scores = pd.to_numeric(df[score_col], errors="coerce")
    invalid_scores = [
        int(idx)
        for idx, value in scores.items()
        if pd.isna(value) or not math.isfinite(float(value))
    ]
    if invalid_scores:
        raise SystemExit(
            _invalid_row_message(label, score_col, invalid_scores, "invalid values")
        )
    df = df.copy()
    df[score_col] = scores.astype(float)
    df = df.sort_values(score_col, ascending=False)
    return df["target_id"].tolist()


def ordered_targets(df: pd.DataFrame, label: str) -> list[str]:
    df = _normalize_target_ids(df, label)
    return df["target_id"].tolist()


def require_targets_present(
    source_targets: list[str],
    reference_targets: list[str],
    source_label: str,
    reference_label: str,
) -> None:
    reference_set = set(reference_targets)
    missing = [target for target in source_targets if target not in reference_set]
    if missing:
        shown = ", ".join(missing[:10])
        suffix = "..." if len(missing) > 10 else ""
        raise SystemExit(
            f"{source_label} target_id value(s) missing from "
            f"{reference_label}: {shown}{suffix}"
        )


def _normalize_target_ids(df: pd.DataFrame, label: str) -> pd.DataFrame:
    blank_targets = [
        int(idx)
        for idx, value in df["target_id"].items()
        if pd.isna(value) or not str(value).strip()
    ]
    if blank_targets:
        raise SystemExit(
            _invalid_row_message(label, "target_id", blank_targets, "blank values")
        )
    df = df.copy()
    df["target_id"] = df["target_id"].astype(str).str.strip()
    duplicate_targets = df["target_id"][df["target_id"].duplicated()].tolist()
    if duplicate_targets:
        shown = ", ".join(duplicate_targets[:10])
        suffix = "..." if len(duplicate_targets) > 10 else ""
        raise SystemExit(
            f"{label} contains duplicate target_id values: {shown}{suffix}"
        )
    return df


def protein_class_map(chembl_dir: Path) -> dict[str, str]:
    path = chembl_dir / "target_classes.parquet"
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(
            f"Target class reference is required and must be non-empty: {path}"
        )
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        raise SystemExit(f"Target class reference failed to parse: {path}: {exc}") from exc
    required = {"uniprot", "class"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(
            f"Target class reference missing required columns {missing}: {path}"
        )
    if df.empty:
        raise SystemExit(f"Target class reference contains no rows: {path}")
    df = df.copy()
    for col in ("uniprot", "class"):
        blank_rows = [
            int(idx)
            for idx, value in df[col].items()
            if pd.isna(value) or not str(value).strip()
        ]
        if blank_rows:
            raise SystemExit(
                _invalid_row_message(
                    "Target class reference",
                    col,
                    blank_rows,
                    "blank values",
                )
            )
        df[col] = df[col].astype(str).str.strip()
    duplicate_uniprots = df["uniprot"][df["uniprot"].duplicated()].tolist()
    if duplicate_uniprots:
        shown = ", ".join(duplicate_uniprots[:10])
        suffix = "..." if len(duplicate_uniprots) > 10 else ""
        raise SystemExit(
            f"Target class reference contains duplicate uniprot values: {shown}{suffix}"
        )
    return dict(zip(df["uniprot"], df["class"], strict=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docking-top", required=True, type=Path)
    parser.add_argument("--autodock-full", required=True, type=Path)
    parser.add_argument("--psichic", required=True, type=Path)
    parser.add_argument("--chembl-dir", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    _remove_outputs(args.out_json)
    docking_top = read_required_csv(
        args.docking_top,
        ",",
        {"target_id"},
        "Docking top consensus",
    )
    dock_full = read_required_csv(
        args.autodock_full,
        "\t",
        {"target_id", "neg_vina_score"},
        "AutoDock full score table",
    )
    dti_full = read_required_csv(
        args.psichic,
        "\t",
        {"target_id", "psichic_score"},
        "PSICHIC score table",
    )

    docking_ranked = rank_targets(
        dock_full,
        "neg_vina_score",
        "AutoDock full score table",
    )
    dti_ranked = rank_targets(
        dti_full,
        "psichic_score",
        "PSICHIC score table",
    )
    docking_consensus_ranked = ordered_targets(
        docking_top,
        "Docking top consensus",
    )
    require_targets_present(
        docking_consensus_ranked,
        docking_ranked,
        "Docking top consensus",
        "AutoDock full score table",
    )
    require_targets_present(
        docking_consensus_ranked,
        dti_ranked,
        "Docking top consensus",
        "PSICHIC score table",
    )
    if not docking_ranked:
        raise SystemExit("AutoDock full score table produced no ranked targets")
    if not dti_ranked:
        raise SystemExit("PSICHIC score table produced no ranked targets")
    if not docking_consensus_ranked:
        raise SystemExit("Docking top consensus produced no ranked targets")

    top_n = min(50, len(docking_consensus_ranked), len(dti_ranked))
    if top_n == 0:
        raise SystemExit("Disagreement top-N window is empty")
    docking_top_set = set(docking_consensus_ranked[:top_n])
    dti_top_set = set(dti_ranked[:top_n])

    half_doc = set(docking_ranked[: len(docking_ranked) // 2 or 1])
    half_dti = set(dti_ranked[: len(dti_ranked) // 2 or 1])
    docking_bottom = set(docking_ranked) - half_doc
    dti_bottom = set(dti_ranked) - half_dti

    docking_only = docking_top_set & dti_bottom
    dti_only = dti_top_set & docking_bottom
    both = docking_top_set & dti_top_set

    cls_map = protein_class_map(args.chembl_dir)

    def by_class(s: set[str]) -> dict[str, int]:
        out: dict[str, int] = {}
        for u in s:
            cls = cls_map.get(u, "orphan")
            out[cls] = out.get(cls, 0) + 1
        return out

    payload = {
        "n_docking_full": len(docking_ranked),
        "n_dti_full": len(dti_ranked),
        "top_n_window": top_n,
        "overlap_at_top_n": len(both),
        "docking_only_count": len(docking_only),
        "dti_only_count": len(dti_only),
        "docking_only": sorted(docking_only),
        "dti_only": sorted(dti_only),
        "both_top": sorted(both),
        "per_class": {
            "docking_only": by_class(docking_only),
            "dti_only": by_class(dti_only),
            "both": by_class(both),
        },
    }
    _write_json_atomic(payload, args.out_json)
    LOG.info("Disagreement → %s  overlap=%d  docking_only=%d  dti_only=%d",
             args.out_json, len(both), len(docking_only), len(dti_only))


if __name__ == "__main__":
    main()
