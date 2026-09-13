#!/usr/bin/env python3
"""Select the fixed Daina primary target set without changing its ranking."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def _nonempty(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def _remove_outputs(*paths: Path | None) -> None:
    for path in paths:
        if path is not None and path.exists():
            path.unlink()


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _metadata(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    if not _nonempty(path):
        raise SystemExit(f"Daina metadata JSON is required and must be non-empty: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Daina metadata JSON is invalid: {path}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Daina metadata JSON must contain an object: {path}")
    if payload.get("schema_version") != "skinscout.daina-run.v1":
        raise SystemExit(
            "Daina metadata JSON has unsupported schema_version: "
            f"{payload.get('schema_version')!r}"
        )
    if payload.get("score_is_calibrated_probability") is not False:
        raise SystemExit(
            "Daina metadata must state score_is_calibrated_probability=false"
        )
    return payload


def _read_daina(path: Path) -> pd.DataFrame:
    if not _nonempty(path):
        raise SystemExit(f"Daina score table is required and must be non-empty: {path}")
    try:
        frame = pd.read_csv(path, sep="\t")
    except Exception as exc:
        raise SystemExit(f"Daina score table failed to parse: {path}: {exc}") from exc
    required = {
        "target_id",
        "score",
        "max_tanimoto",
        "evidence_count",
        "supporting_molecule_id",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(
            "Daina score table is missing required column(s): " + ", ".join(missing)
        )
    if frame.empty:
        raise SystemExit(f"Daina score table contains no target rows: {path}")
    targets = frame["target_id"].astype("string").str.strip()
    blank = targets.isna() | targets.eq("")
    if blank.any():
        raise SystemExit(
            "Daina score table contains blank target_id at row index(es): "
            + ", ".join(str(int(index)) for index in frame.index[blank][:10])
        )
    duplicate = targets[targets.duplicated()].tolist()
    if duplicate:
        raise SystemExit(
            "Daina score table contains duplicate target_id values: "
            + ", ".join(str(value) for value in duplicate[:10])
        )
    boolean_scores = frame["score"].map(
        lambda value: isinstance(value, bool)
        or (isinstance(value, str) and value.strip().lower() in {"true", "false"})
    )
    scores = pd.to_numeric(frame["score"], errors="coerce")
    invalid = boolean_scores | scores.isna() | ~scores.map(
        lambda value: math.isfinite(float(value)) if not pd.isna(value) else False
    )
    if invalid.any():
        raise SystemExit(
            "Daina score table contains non-finite/non-numeric score at row index(es): "
            + ", ".join(str(int(index)) for index in frame.index[invalid][:10])
        )
    tanimoto = pd.to_numeric(frame["max_tanimoto"], errors="coerce")
    invalid_tanimoto = tanimoto.isna() | ~tanimoto.map(
        lambda value: (
            math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0
            if not pd.isna(value)
            else False
        )
    )
    if invalid_tanimoto.any():
        raise SystemExit(
            "Daina score table contains max_tanimoto outside [0, 1] at row index(es): "
            + ", ".join(str(int(index)) for index in frame.index[invalid_tanimoto][:10])
        )
    evidence = pd.to_numeric(frame["evidence_count"], errors="coerce")
    invalid_evidence = evidence.isna() | ~evidence.map(
        lambda value: (
            math.isfinite(float(value))
            and float(value) >= 0
            and float(value).is_integer()
            if not pd.isna(value)
            else False
        )
    )
    if invalid_evidence.any():
        raise SystemExit(
            "Daina score table contains invalid evidence_count at row index(es): "
            + ", ".join(str(int(index)) for index in frame.index[invalid_evidence][:10])
        )
    supporting = frame["supporting_molecule_id"].astype("string").str.strip()
    blank_support = supporting.isna() | supporting.eq("")
    if blank_support.any():
        raise SystemExit(
            "Daina score table contains blank supporting_molecule_id at row "
            "index(es): "
            + ", ".join(str(int(index)) for index in frame.index[blank_support][:10])
        )
    frame = frame.copy()
    frame["supporting_molecule_id"] = supporting.astype(str)
    frame["target_id"] = targets.astype(str)
    frame["score"] = scores.astype(float)
    frame["max_tanimoto"] = tanimoto.astype(float)
    frame["evidence_count"] = evidence.astype(int)
    return frame.sort_values(
        ["score", "target_id"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def select_daina_targets(
    scores: pd.DataFrame,
    *,
    top_n: int,
    metadata: dict[str, object],
) -> pd.DataFrame:
    selected = scores.head(top_n).copy()
    output = pd.DataFrame(
        {
            "target_id": selected["target_id"],
            "daina_rank": range(1, len(selected) + 1),
            "daina_score": selected["score"],
            "daina_score_is_probability": "false",
            "daina_evidence_mode": str(metadata.get("evidence_mode", "unknown")),
            "daina_quality_policy": selected.get(
                "quality_policy",
                pd.Series(
                    [str(metadata.get("quality_policy", "unknown"))] * len(selected)
                ),
            ).astype(str).tolist(),
            "daina_scoring_method": str(metadata.get("scoring_method", "unknown")),
            "daina_max_tanimoto": pd.to_numeric(
                selected["max_tanimoto"], errors="coerce"
            ),
            # stage3_daina_zoete emits `evidence_count`. The previous
            # `n_known_ligands` lookup matched no column the scorer produces and
            # fell through to the zero default, so this was 0 on every run.
            "daina_known_ligand_count": pd.to_numeric(
                selected["evidence_count"], errors="coerce"
            ).astype(int),
            # The reference molecule behind max_tanimoto. Without it a reader
            # cannot tell a retrieved known interaction from a prediction.
            "daina_supporting_molecule_id": selected[
                "supporting_molecule_id"
            ].astype(str),
            # active / weak / inactive for the nearest analogue's measurement.
            # The mirror path has no labels, so it reports "unknown" rather than
            # implying the evidence was checked and found positive.
            "daina_supporting_evidence": selected.get(
                "supporting_evidence", pd.Series(["unknown"] * len(selected))
            ).astype(str).tolist(),
        }
    )
    if output.empty:
        raise SystemExit("Daina target selection produced no rows")
    return output


def compatibility_projection(selected: pd.DataFrame) -> pd.DataFrame:
    projected = selected.copy()
    projected["score"] = projected["daina_score"]
    projected["rrf_score"] = projected["daina_score"]
    projected["source_count"] = 1
    projected["sources"] = "daina"
    projected["recipe_id"] = "daina-primary-v1"
    return projected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daina-scores", required=True, type=Path)
    parser.add_argument("--daina-metadata", type=Path)
    parser.add_argument("--top-n", type=int, default=256)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--out-compat-csv", type=Path)
    args = parser.parse_args()

    _remove_outputs(args.out_csv, args.out_compat_csv)
    if args.top_n != 256:
        raise SystemExit(
            "--top-n is a frozen public contract and must equal 256: "
            f"{args.top_n}"
        )
    selected = select_daina_targets(
        _read_daina(args.daina_scores),
        top_n=args.top_n,
        metadata=_metadata(args.daina_metadata),
    )
    _write_csv_atomic(selected, args.out_csv)
    if args.out_compat_csv is not None:
        _write_csv_atomic(compatibility_projection(selected), args.out_compat_csv)


if __name__ == "__main__":
    main()
