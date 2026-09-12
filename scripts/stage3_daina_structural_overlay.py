#!/usr/bin/env python3
"""Attach structural, skin-KG, and experimental evidence to Daina ranks.

The Daina order and score are immutable primary evidence. All later evidence is
annotation-only and cannot promote or demote a target.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from stage3_kg_efficacy_label import load_graph, top_efficacy_records


TARGET_SCHEMA = "skinscout.target_fast.v2"
RECIPE_ID = "daina-structural-overlay-v1"
# This overlay annotates; it never reorders. `score`, `rrf_score`, `docking_rrf`
# and `final_score` are all copies of the Daina score, so the column names alone
# would suggest a fusion that does not happen. Every row states the quantity its
# order actually came from - and that quantity is not always the similarity.


def _ranking_basis(scoring_method: str) -> str:
    """Name the column a reader can sort by to reproduce `daina_rank`.

    Under `max-similarity` the score is the nearest-analogue Tanimoto, so the
    two are one number. Under the promoted recipe they are not: on alpha-arbutin
    the two orders disagree at 253 of the top 256 positions. Stamping
    `daina_max_tanimoto` on a recipe-scored row sends a reader to sort by a
    column that will not reproduce the ranking, and makes the file look corrupt.
    """
    method = str(scoring_method or "").strip() or "unknown"
    if method == "max-similarity":
        return "daina_max_tanimoto"
    return f"daina_score ({method})"


RANKING_BASIS = "daina_max_tanimoto"
FINAL_STRUCTURAL_STATUSES = {
    "structure_supported",
    "structural_unavailable_no_pocket",
    "structural_unavailable_prep",
    "structural_unavailable_invalid",
    "structure_failed_map",
    "structure_failed_docking",
    "structure_failed_gnina",
    # Docking only the band that re-ranking consults leaves the rest unscored
    # by choice. Saying "prep failed" for a target nobody asked about would be
    # the same class of untruth this pipeline has been removing.
    "structure_not_requested",
}


def _nonempty(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def _remove_outputs(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def _read_json(path: Path, label: str, schema: str) -> dict[str, Any]:
    if not _nonempty(path):
        raise SystemExit(f"{label} is missing or empty: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} must contain an object: {path}")
    if payload.get("schema_version") != schema:
        raise SystemExit(
            f"{label} has unsupported schema_version: "
            f"{payload.get('schema_version')!r}"
        )
    return payload


def _records_by_target(payload: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    records = payload.get("targets")
    if not isinstance(records, list):
        raise SystemExit(f"{label} targets must be an array")
    if payload.get("target_count") != len(records):
        raise SystemExit(f"{label} target_count does not match targets")
    output: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(records):
        if not isinstance(value, dict):
            raise SystemExit(f"{label} record {index} must be an object")
        target_id = str(value.get("target_id", "")).strip()
        if not target_id or target_id in output:
            raise SystemExit(f"{label} has blank/duplicate target_id: {target_id!r}")
        output[target_id] = value
    return output


def _read_selected(path: Path) -> pd.DataFrame:
    if not _nonempty(path):
        raise SystemExit(f"Daina selected-target CSV is missing or empty: {path}")
    frame = pd.read_csv(path)
    required = {
        "target_id",
        "daina_rank",
        "daina_score",
        "daina_score_is_probability",
        "daina_max_tanimoto",
        "daina_known_ligand_count",
        "daina_supporting_molecule_id",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(
            "Daina selected-target CSV missing required column(s): "
            + ", ".join(missing)
        )
    if frame.empty:
        raise SystemExit("Daina selected-target CSV contains no rows")
    frame = frame.copy()
    frame["target_id"] = frame["target_id"].astype(str).str.strip()
    if frame["target_id"].eq("").any() or frame["target_id"].duplicated().any():
        raise SystemExit("Daina selected-target target_id values must be nonblank and unique")
    ranks = pd.to_numeric(frame["daina_rank"], errors="coerce")
    scores = pd.to_numeric(frame["daina_score"], errors="coerce")
    if ranks.isna().any() or scores.isna().any():
        raise SystemExit("Daina selected-target rank/score values must be numeric")
    expected = list(range(1, len(frame) + 1))
    if ranks.astype(int).tolist() != expected:
        raise SystemExit("Daina selected-target ranks must be contiguous and ordered")
    if not all(math.isfinite(float(value)) for value in scores):
        raise SystemExit("Daina selected-target scores must be finite")
    probability_flags = frame["daina_score_is_probability"].astype(str).str.lower()
    if not probability_flags.eq("false").all():
        raise SystemExit("Daina scores must not be represented as probabilities")
    frame["daina_rank"] = ranks.astype(int)
    frame["daina_score"] = scores.astype(float)
    return frame


def _read_score_table(path: Path, score_column: str, label: str) -> dict[str, float]:
    if not _nonempty(path):
        raise SystemExit(f"{label} table is missing or empty: {path}")
    frame = pd.read_csv(path, sep="\t")
    required = {"target_id", score_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"{label} table missing column(s): {', '.join(missing)}")
    if frame.empty:
        return {}
    targets = frame["target_id"].astype(str).str.strip()
    if targets.eq("").any() or targets.duplicated().any():
        raise SystemExit(f"{label} target_id values must be nonblank and unique")
    scores = pd.to_numeric(frame[score_column], errors="coerce")
    if scores.isna().any() or not all(math.isfinite(float(value)) for value in scores):
        raise SystemExit(f"{label} {score_column} values must be finite numeric values")
    return dict(zip(targets, scores.astype(float), strict=True))


def _skin_annotations(path: Path) -> dict[str, dict[str, object]]:
    if not _nonempty(path):
        raise SystemExit(f"Skin-expression table is missing or empty: {path}")
    frame = pd.read_csv(path, sep="\t")
    required = {"uniprot", "skin_score"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(
            "Skin-expression table missing required column(s): " + ", ".join(missing)
        )
    output: dict[str, dict[str, object]] = {}
    for _, row in frame.iterrows():
        target_id = str(row["uniprot"]).strip()
        if not target_id:
            continue
        score = _required_unit_float(
            row["skin_score"], f"Skin-expression score for {target_id}"
        )
        output[target_id] = {
            "score": score,
            "tier": _optional_text(row.get("tier")) or "unknown",
            "cell_type_preferred": (
                _optional_text(row.get("cell_type_preferred")) or "unknown"
            ),
        }
    return output


def _experimental_annotations(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        raise SystemExit(f"Known-target prior CSV does not exist: {path}")
    frame = pd.read_csv(path)
    required = {"target_id", "prior_score"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(
            "Known-target prior CSV missing required column(s): " + ", ".join(missing)
        )
    output: dict[str, dict[str, object]] = {}
    for _, row in frame.iterrows():
        target_id = str(row["target_id"]).strip()
        if not target_id:
            raise SystemExit("Known-target prior rows require a nonblank target_id")
        score = _required_unit_float(
            row["prior_score"], f"Known-target prior score for {target_id}"
        )
        output[target_id] = {
            "supported": score > 0.0,
            "prior_score": score,
            "evidence_count": int(row.get("evidence_count", 0)),
            "source_case_ids": _optional_text(row.get("evidence_case_ids")) or "",
            "source_panels": _optional_text(row.get("evidence_panels")) or "",
        }
    return output


def _kg_annotation(graph: Any, target_id: str) -> dict[str, object]:
    records = top_efficacy_records(graph, target_id, 3)
    categories = [
        {"category": str(record["category"]), "n_papers": int(record["n_papers"])}
        for record in records
    ]
    directions: set[str] = set()
    for record in records:
        for attrs in (record.get("edge_data", {}), record.get("node_attrs", {})):
            value = str(
                attrs.get("skin_effect_direction", attrs.get("effect_direction", ""))
            ).strip().lower()
            if value:
                directions.add(value)
    direction = next(iter(directions)) if len(directions) == 1 else (
        "context_dependent" if directions else "unknown"
    )
    return {
        "supported": bool(records),
        "categories": categories,
        "skin_effect_direction": direction,
    }


def _optional_text(value: object) -> str | None:
    """Return a stripped string, or None for JSON null/absent/blank values.

    ``str(None)`` yields the literal ``"None"``; routing every optional
    manifest field through this helper keeps that string out of the published
    artifact and out of the SHA-256 audit fields.
    """
    if value is None:
        return None
    return str(value).strip() or None


def _required_unit_float(value: object, label: str) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{label} must be a number: {value!r}") from exc
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise SystemExit(f"{label} must be a finite value in [0, 1]: {score!r}")
    return score


def _require_score_matches_status(
    target_id: str,
    *,
    status: str,
    docking_status: str,
    docking_score: float | None,
    gnina_score: float | None,
) -> None:
    """Reject rows whose structural status contradicts the emitted scores.

    Without this a ``structure_failed_docking`` row could still publish an
    AutoDock energy, and a ``docked`` target whose score row went missing was
    silently relabelled ``structure_failed_gnina``.
    """
    if docking_status == "docked" and docking_score is None:
        raise SystemExit(
            f"{target_id} is marked docked but has no AutoDock score row"
        )
    if status == "structure_supported":
        if docking_score is None or gnina_score is None:
            raise SystemExit(
                f"{target_id} is structure_supported but is missing an "
                "AutoDock or GNINA score"
            )
        return
    if status == "structure_failed_gnina":
        if docking_score is None:
            raise SystemExit(
                f"{target_id} failed GNINA after docking but has no AutoDock score"
            )
        if gnina_score is not None:
            raise SystemExit(
                f"{target_id} failed GNINA but still carries a GNINA score"
            )
        return
    if docking_score is not None or gnina_score is not None:
        raise SystemExit(
            f"{target_id} has structural status {status!r} but still carries a "
            "docking or GNINA score"
        )


def _json_cell(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-csv", required=True, type=Path)
    parser.add_argument("--map-manifest", required=True, type=Path)
    parser.add_argument("--docking-scores", required=True, type=Path)
    parser.add_argument("--docking-status-manifest", required=True, type=Path)
    parser.add_argument("--gnina-scores", required=True, type=Path)
    parser.add_argument("--gnina-status-manifest", required=True, type=Path)
    parser.add_argument("--skin-tsv", required=True, type=Path)
    parser.add_argument("--skin-kg", required=True, type=Path)
    parser.add_argument("--known-target-priors", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--out-top50", required=True, type=Path)
    args = parser.parse_args()

    _remove_outputs(args.out_csv, args.out_top50)
    selected = _read_selected(args.selected_csv)
    map_records = _records_by_target(
        _read_json(
            args.map_manifest,
            "AutoGrid map manifest",
            "skinscout.autogrid-map-manifest.v1",
        ),
        "AutoGrid map manifest",
    )
    docking_records = _records_by_target(
        _read_json(
            args.docking_status_manifest,
            "Docking status manifest",
            "skinscout.docking-status-manifest.v1",
        ),
        "Docking status manifest",
    )
    gnina_records = _records_by_target(
        _read_json(
            args.gnina_status_manifest,
            "GNINA status manifest",
            "skinscout.gnina-pose-status.v1",
        ),
        "GNINA status manifest",
    )
    selected_ids = selected["target_id"].tolist()
    if set(map_records) != set(selected_ids) or set(docking_records) != set(selected_ids):
        raise SystemExit("Selected/map/docking manifests must have exact target parity")
    docking_scores = _read_score_table(
        args.docking_scores, "vina_score", "AutoDock score"
    )
    gnina_scores = _read_score_table(
        args.gnina_scores, "cnn_affinity", "GNINA score"
    )
    if set(gnina_records) != set(docking_scores):
        raise SystemExit("GNINA status records must match successfully docked targets")
    if not set(gnina_scores).issubset(gnina_records):
        raise SystemExit("GNINA score targets must be present in its status manifest")

    skin = _skin_annotations(args.skin_tsv)
    experimental = _experimental_annotations(args.known_target_priors)
    graph = load_graph(args.skin_kg)
    rows: list[dict[str, object]] = []
    for _, primary in selected.iterrows():
        target_id = str(primary["target_id"])
        map_record = map_records[target_id]
        docking_record = docking_records[target_id]
        map_status = str(map_record.get("status", ""))
        docking_status = str(docking_record.get("status", ""))
        if map_status != "map_ready":
            status = map_status
        elif docking_status != "docked":
            status = "structure_failed_docking"
        else:
            status = str(
                gnina_records.get(target_id, {}).get(
                    "status", "structure_failed_gnina"
                )
            )
        if status not in FINAL_STRUCTURAL_STATUSES:
            raise SystemExit(
                f"Unsupported final structural status for {target_id}: {status!r}"
            )
        docking_score = docking_scores.get(target_id)
        gnina_score = gnina_scores.get(target_id)
        gnina_record = gnina_records.get(target_id, {})
        _require_score_matches_status(
            target_id,
            status=status,
            docking_status=docking_status,
            docking_score=docking_score,
            gnina_score=gnina_score,
        )
        structure = {
            "status": status,
            "pocket_id": (
                _optional_text(map_record.get("pocket_id"))
                if map_status == "map_ready"
                else None
            ),
            "pocket_source": (
                _optional_text(map_record.get("pocket_source"))
                if map_status == "map_ready"
                else None
            ),
            "structure_source": (
                _optional_text(map_record.get("structure_source"))
                if map_status == "map_ready"
                else None
            ),
            "docking_score": docking_score,
            "gnina_score": gnina_score,
            "map_cache_key": _optional_text(map_record.get("map_cache_key")),
            "pose_file": _optional_text(gnina_record.get("pose_file")),
            "pose_sha256": _optional_text(gnina_record.get("pose_sha256")),
        }
        skin_row = skin.get(
            target_id,
            {"score": 0.0, "tier": "very_low", "cell_type_preferred": "unknown"},
        )
        kg = _kg_annotation(graph, target_id)
        exp = experimental.get(
            target_id,
            {
                "supported": False,
                "prior_score": 0.0,
                "evidence_count": 0,
                "source_case_ids": "",
                "source_panels": "",
            },
        )
        sources = ["daina"]
        if status == "structure_supported":
            sources.extend(["autodock_gpu", "gnina"])
        if target_id in skin:
            sources.append("skin_expression")
        if kg["supported"]:
            sources.append("skin_kg")
        if exp["supported"]:
            sources.append("experimental_prior")
        max_tanimoto = _required_unit_float(
            primary["daina_max_tanimoto"],
            f"Daina max Tanimoto for {target_id}",
        )
        known_ligand_count = int(primary["daina_known_ligand_count"])
        supporting_molecule_id = str(primary["daina_supporting_molecule_id"]).strip()
        if not supporting_molecule_id:
            raise SystemExit(
                f"{target_id} has a blank daina_supporting_molecule_id"
            )
        daina_primary = {
            "rank": int(primary["daina_rank"]),
            "score": float(primary["daina_score"]),
            "score_is_probability": False,
        }
        rows.append(
            {
                "schema_version": TARGET_SCHEMA,
                "recipe_id": RECIPE_ID,
                "target_id": target_id,
                "target_name": "",
                "ranking_basis": _ranking_basis(
                    str(primary.get("daina_scoring_method", "unknown"))
                ),
                "daina_rank": daina_primary["rank"],
                "daina_score": daina_primary["score"],
                "daina_score_is_probability": False,
                "daina_evidence_mode": str(primary.get("daina_evidence_mode", "unknown")),
                "daina_quality_policy": str(primary.get("daina_quality_policy", "unknown")),
                "daina_scoring_method": str(primary.get("daina_scoring_method", "unknown")),
                "daina_primary_json": _json_cell(daina_primary),
                "daina_max_tanimoto": max_tanimoto,
                "daina_known_ligand_count": known_ligand_count,
                "daina_supporting_molecule_id": supporting_molecule_id,
                # What the nearest analogue's own measurement said, under the
                # index's threshold policy. The ranking does not use it - see
                # docs/RECIPE_RUNPATH_MEASURED_20260831.md section 7 - so the
                # reader needs it to tell a potent analogue from a weak one.
                "daina_supporting_evidence": str(
                    primary.get("daina_supporting_evidence", "unknown")
                ),
                # A Tanimoto of 1.0 means a reference ligand with a fingerprint
                # identical to the query is already measured on this target, so
                # the row reports a known interaction rather than predicting a
                # new one. It is a fact about the reference set, not a
                # confidence score, and the supporting molecule id above lets a
                # reader confirm it.
                "daina_is_self_match": bool(max_tanimoto >= 1.0),
                "structural_status": status,
                "structure_supported": status == "structure_supported",
                "structure_source": structure["structure_source"] or "",
                "pocket_id": structure["pocket_id"] or "",
                "autodock_energy_kcal_mol": docking_score,
                "gnina_cnn_affinity": gnina_score,
                "map_cache_key": structure["map_cache_key"] or "",
                "pose_file": structure["pose_file"] or "",
                "pose_sha256": structure["pose_sha256"] or "",
                "structure_annotation_json": _json_cell(structure),
                "skin_score": skin_row["score"],
                "skin_tier": skin_row["tier"],
                "cell_type_preferred": skin_row["cell_type_preferred"],
                "skin_kg_supported": bool(kg["supported"]),
                "skin_kg_categories": ";".join(
                    str(item["category"]) for item in kg["categories"]
                ),
                "skin_effect_direction": kg["skin_effect_direction"],
                "skin_kg_annotation_json": _json_cell(kg),
                "experimental_supported": bool(exp["supported"]),
                "experimental_prior_score": exp["prior_score"],
                "experimental_evidence_count": exp["evidence_count"],
                "experimental_source_case_ids": exp["source_case_ids"],
                "experimental_annotation_json": _json_cell(exp),
                # Compatibility fields retain the Daina primary score/order.
                "score": daina_primary["score"],
                "rrf_score": daina_primary["score"],
                "docking_rrf": daina_primary["score"],
                "final_score": daina_primary["score"],
                "source_count": len(sources),
                "sources": ";".join(sources),
            }
        )

    output = pd.DataFrame(rows)
    if output["daina_rank"].tolist() != list(range(1, len(output) + 1)):
        raise SystemExit("Internal error: Daina primary order changed during overlay")
    if output["daina_score"].tolist() != selected["daina_score"].tolist():
        raise SystemExit("Internal error: Daina primary scores changed during overlay")
    _write_csv_atomic(output, args.out_csv)
    _write_csv_atomic(output.head(50), args.out_top50)


if __name__ == "__main__":
    main()
