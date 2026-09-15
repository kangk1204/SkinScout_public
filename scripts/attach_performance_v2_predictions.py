#!/usr/bin/env python3
"""Append validated performance-v2 evidence without changing legacy target ranks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from performance_v2_model import (  # noqa: E402
    ContractError,
    RANKING_COLUMNS,
    append_only_outputs,
    sha256_file,
    validate_ranking_manifest,
)


SCHEMA_VERSION = "skinscout.performance-v2-stage3-attachment.v1"
APPENDED_COLUMNS = (
    "performance_v2_model_rank",
    "performance_v2_ranking_score",
    "performance_v2_score_semantics",
    "performance_v2_direct_binding_probability",
    "performance_v2_functional_modulation_probability",
    "performance_v2_ood_route",
    "performance_v2_abstained",
    "performance_v2_coverage_status",
    "performance_v2_candidate_id",
    "performance_v2_promotion_status",
    "performance_v2_ranking_manifest_sha256",
    "performance_v2_model_manifest_sha256",
)


class AttachmentError(ValueError):
    """Raised when additive attachment cannot be proven safe."""


def _read_csv(path: Path, label: str) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
        raise AttachmentError(f"{label} is missing, empty, or unsafe: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        if len(columns) != len(set(columns)):
            raise AttachmentError(f"{label} contains duplicate columns")
        rows = list(reader)
    if not columns or not rows:
        raise AttachmentError(f"{label} contains no data rows")
    return columns, rows


def _target_index(rows: list[dict[str, str]], label: str) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for row in rows:
        target_id = str(row.get("target_id") or "").strip()
        if not target_id:
            raise AttachmentError(f"{label}.target_id contains blanks")
        if target_id in output:
            raise AttachmentError(f"{label}.target_id contains duplicate value: {target_id}")
        output[target_id] = row
    return output


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
        raise AttachmentError(f"{label} is missing, empty, or unsafe: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AttachmentError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise AttachmentError(f"{label} must be a JSON object")
    return payload


def _canonical_sha256(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_csv_atomic(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _attach(
    *,
    ranked_csv: Path,
    ranking_csv: Path,
    ranking_manifest_path: Path,
    out_csv: Path,
    out_manifest: Path,
    candidate_id: str,
) -> None:
    if candidate_id not in {"P1", "P2", "P3"}:
        raise AttachmentError("candidate_id must be P1, P2, or P3")
    ranking_manifest = validate_ranking_manifest(ranking_manifest_path)
    outputs = ranking_manifest.get("outputs")
    inputs = ranking_manifest.get("inputs")
    ranking_record = outputs.get("ranking_csv") if isinstance(outputs, dict) else None
    if not isinstance(ranking_record, dict) or ranking_record.get("sha256") != sha256_file(ranking_csv):
        raise AttachmentError("ranking manifest does not bind the active ranking CSV")
    model_record = inputs.get("model_manifest") if isinstance(inputs, dict) else None
    if not isinstance(model_record, dict):
        raise AttachmentError("ranking manifest is missing model manifest provenance")
    model_manifest_sha256 = str(model_record.get("sha256") or "")
    if len(model_manifest_sha256) != 64:
        raise AttachmentError("ranking manifest model SHA-256 is invalid")

    legacy_columns, legacy_rows = _read_csv(ranked_csv, "legacy ranked targets")
    if "target_id" not in legacy_columns:
        raise AttachmentError("legacy ranked targets missing target_id")
    collisions = sorted(set(legacy_columns) & set(APPENDED_COLUMNS))
    if collisions:
        raise AttachmentError(f"legacy ranked targets already contain v2 columns: {collisions}")
    legacy_index = _target_index(legacy_rows, "legacy ranked targets")

    ranking_columns, ranking_rows = _read_csv(ranking_csv, "performance-v2 ranking")
    if tuple(ranking_columns) != tuple(RANKING_COLUMNS):
        raise AttachmentError("performance-v2 ranking columns changed")
    ranking_index = _target_index(ranking_rows, "performance-v2 ranking")
    missing = sorted(set(legacy_index) - set(ranking_index))
    if missing:
        raise AttachmentError(f"performance-v2 ranking missing legacy targets: {missing[:5]}")

    ranking_manifest_sha256 = sha256_file(ranking_manifest_path)
    model_ranks = {
        str(row["target_id"]).strip(): str(row["performance_v2_rank"])
        for row in ranking_rows
    }
    output_rows: list[dict[str, str]] = []
    for legacy in legacy_rows:
        target_id = str(legacy["target_id"]).strip()
        v2 = ranking_index[target_id]
        abstained = str(v2["performance_v2_abstained"]).strip().lower()
        if abstained not in {"true", "false"}:
            raise AttachmentError(f"invalid performance_v2_abstained value for {target_id}")
        output_rows.append(
            {
                **legacy,
                "performance_v2_model_rank": model_ranks[target_id],
                "performance_v2_ranking_score": v2["performance_v2_ranking_score"],
                "performance_v2_score_semantics": v2["performance_v2_score_semantics"],
                "performance_v2_direct_binding_probability": v2[
                    "performance_v2_direct_binding_probability"
                ],
                "performance_v2_functional_modulation_probability": v2[
                    "performance_v2_functional_modulation_probability"
                ],
                "performance_v2_ood_route": v2["performance_v2_ood_route"],
                "performance_v2_abstained": abstained,
                "performance_v2_coverage_status": "abstained_ood" if abstained == "true" else "scored",
                "performance_v2_candidate_id": candidate_id,
                "performance_v2_promotion_status": "research_candidate_not_promoted",
                "performance_v2_ranking_manifest_sha256": ranking_manifest_sha256,
                "performance_v2_model_manifest_sha256": model_manifest_sha256,
            }
        )

    output_columns = [*legacy_columns, *APPENDED_COLUMNS]
    _write_csv_atomic(out_csv, output_columns, output_rows)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "append_only": True,
        "candidate_id": candidate_id,
        "promotion_status": "research_candidate_not_promoted",
        "legacy_order_preserved": True,
        "legacy_columns_preserved": legacy_columns,
        "appended_columns": list(APPENDED_COLUMNS),
        "counts": {
            "legacy_targets": len(legacy_rows),
            "v2_universe_targets": len(ranking_rows),
            "attached_targets": len(output_rows),
            "abstained_targets": sum(
                row["performance_v2_abstained"] == "true" for row in output_rows
            ),
        },
        "inputs": {
            "legacy_ranked_targets_sha256": sha256_file(ranked_csv),
            "performance_v2_ranking_sha256": sha256_file(ranking_csv),
            "performance_v2_ranking_manifest_sha256": ranking_manifest_sha256,
            "performance_v2_model_manifest_sha256": model_manifest_sha256,
        },
        "output": {"path": str(out_csv.resolve()), "sha256": sha256_file(out_csv)},
    }
    payload["binding_sha256"] = _canonical_sha256(payload)
    _write_json_atomic(out_manifest, payload)


def attach(
    *,
    ranked_csv: Path,
    ranking_csv: Path,
    ranking_manifest_path: Path,
    out_csv: Path,
    out_manifest: Path,
    candidate_id: str,
) -> None:
    try:
        with append_only_outputs((out_csv, out_manifest)):
            _attach(
                ranked_csv=ranked_csv,
                ranking_csv=ranking_csv,
                ranking_manifest_path=ranking_manifest_path,
                out_csv=out_csv,
                out_manifest=out_manifest,
                candidate_id=candidate_id,
            )
    except ContractError as exc:
        raise AttachmentError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranked-csv", required=True, type=Path)
    parser.add_argument("--ranking-csv", required=True, type=Path)
    parser.add_argument("--ranking-manifest", required=True, type=Path)
    parser.add_argument("--candidate-id", default="P1")
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--out-manifest", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    attach(
        ranked_csv=args.ranked_csv,
        ranking_csv=args.ranking_csv,
        ranking_manifest_path=args.ranking_manifest,
        out_csv=args.out_csv,
        out_manifest=args.out_manifest,
        candidate_id=args.candidate_id,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AttachmentError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
