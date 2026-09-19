#!/usr/bin/env python3
"""stage3_kg_efficacy_label.py — Attach top-3 efficacy categories to each
ranked target by querying the skin-efficacy KG (NetworkX GraphML).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

LOG = logging.getLogger("stage3.kg_label")

MECHANISM_CLASSES = {"pathway_effect", "direct_binding"}
MECHANISM_ACTIONS = {
    "activation",
    "agonist",
    "antagonist",
    "downregulation",
    "inhibition",
    "modulation",
    "unknown",
    "upregulation",
}
SKIN_EFFECT_DIRECTIONS = {"beneficial", "adverse", "context_dependent", "unknown"}
SKIN_EFFECT_CONFIDENCES = {"high", "medium", "low"}
PREDICTION_COVERAGES = {
    "kg_efficacy_only",
    "kg_efficacy_with_partial_mechanism_metadata",
    "kg_efficacy_with_structured_mechanism_metadata",
}
OOD_ROUTES = {
    "none",
    "missing_structured_mechanism_metadata",
    "ambiguous_structured_mechanism_metadata",
}
SEMANTIC_COLUMNS = [
    "biochemical_rank_score",
    "biochemical_score_is_probability",
    "mechanism_class",
    "mechanism_action",
    "skin_effect_direction",
    "skin_effect_confidence",
    "prediction_coverage",
    "uncertainty_score",
    "ood_route",
    "evidence_provenance_json",
]


def _remove_outputs(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def _nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _write_csv_atomic(df: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def load_graph(graphml: Path) -> Any:
    import networkx as nx
    if not _nonempty(graphml):
        raise SystemExit(f"Skin-efficacy KG GraphML is required and must be non-empty: {graphml}")
    graph = nx.read_graphml(graphml)
    if graph.number_of_nodes() == 0:
        raise SystemExit(f"Skin-efficacy KG contains no nodes: {graphml}")
    return graph


def _nonnegative_paper_count(value: object, uniprot: str, category: str) -> int:
    try:
        n_papers = int(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            "Skin-efficacy KG edge must provide non-negative counted paper evidence "
            f"for {uniprot} -> {category}: {value!r}"
        ) from exc
    if n_papers < 0:
        raise SystemExit(
            "Skin-efficacy KG edge must provide non-negative counted paper evidence "
            f"for {uniprot} -> {category}: {n_papers}"
        )
    return n_papers


def _paper_evidence(data: dict[str, Any], uniprot: str, category: str) -> tuple[int, int, str]:
    basis = str(data.get("n_papers_basis", "legacy_n_papers")).strip()
    counted_value = data.get("n_papers_counted", data.get("n_papers"))
    counted = _nonnegative_paper_count(counted_value, uniprot, category)
    seed = _nonnegative_paper_count(
        data.get("n_papers_seed", 0), uniprot, category
    )
    if "curated_seed" in basis and "n_papers_counted" not in data:
        raise SystemExit(
            "Skin-efficacy KG curated seed edge must provide n_papers_counted "
            f"for {uniprot} -> {category}"
        )
    if counted == 0 and "curated_seed" not in basis:
        raise SystemExit(
            "Skin-efficacy KG edge must provide positive counted paper evidence "
            f"for {uniprot} -> {category}"
        )
    return counted, seed, basis


def top_efficacy(graph: Any, uniprot: str, top_n: int = 3) -> list[tuple[str, int]]:
    """Return [(efficacy_category, n_papers), ...] sorted descending by papers."""
    gene_node = f"gene:{uniprot}"
    if gene_node not in graph:
        return []
    rows: list[tuple[str, int]] = []
    seen_names: set[str] = set()
    for _, dst, data in graph.out_edges(gene_node, data=True):
        node_attrs = graph.nodes[dst]
        if node_attrs.get("type") != "EfficacyCategory":
            continue
        name = str(node_attrs.get("name", dst.replace("category:", ""))).strip()
        if not name:
            raise SystemExit(
                f"Skin-efficacy KG category name must be non-empty for target {uniprot}: {dst}"
            )
        normalized_name = name.casefold()
        if normalized_name in seen_names:
            raise SystemExit(
                "Skin-efficacy KG contains duplicate efficacy category label "
                f"for target {uniprot}: {name}"
            )
        seen_names.add(normalized_name)
        n_papers, _seed, _basis = _paper_evidence(data, uniprot, name)
        rows.append((name, n_papers))
    rows.sort(key=lambda x: (-x[1], x[0].casefold(), x[0]))
    return rows[:top_n]


def _clean_metadata_value(value: object) -> str:
    import pandas as pd

    if pd.isna(value):
        return ""
    return str(value).strip()


def _metadata_values(
    *,
    edge_data: dict[str, Any],
    node_attrs: dict[str, Any],
    keys: tuple[str, ...],
) -> set[str]:
    values: set[str] = set()
    for key in keys:
        for attrs in (edge_data, node_attrs):
            value = _clean_metadata_value(attrs.get(key))
            if value:
                values.add(value)
    return values


def _single_metadata_value(
    values: set[str],
    *,
    allowed: set[str],
    field: str,
    uniprot: str,
    category: str,
) -> tuple[str | None, bool]:
    if not values:
        return None, False
    normalized = {value.lower() for value in values}
    invalid = sorted(normalized - allowed)
    if invalid:
        raise SystemExit(
            f"Skin-efficacy KG {field} has unsupported value(s) for "
            f"{uniprot} -> {category}: {', '.join(invalid)}"
        )
    if len(normalized) > 1:
        return None, True
    return next(iter(normalized)), False


def top_efficacy_records(
    graph: Any,
    uniprot: str,
    top_n: int = 3,
) -> list[dict[str, Any]]:
    """Return top efficacy records with edge/node metadata for semantic labels."""
    gene_node = f"gene:{uniprot}"
    if gene_node not in graph:
        return []
    rows: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for _, dst, data in graph.out_edges(gene_node, data=True):
        node_attrs = dict(graph.nodes[dst])
        if node_attrs.get("type") != "EfficacyCategory":
            continue
        name = str(node_attrs.get("name", dst.replace("category:", ""))).strip()
        if not name:
            raise SystemExit(
                f"Skin-efficacy KG category name must be non-empty for target {uniprot}: {dst}"
            )
        normalized_name = name.casefold()
        if normalized_name in seen_names:
            raise SystemExit(
                "Skin-efficacy KG contains duplicate efficacy category label "
                f"for target {uniprot}: {name}"
            )
        seen_names.add(normalized_name)
        edge_data = dict(data)
        n_papers, n_papers_seed, n_papers_basis = _paper_evidence(
            edge_data, uniprot, name
        )
        rows.append(
            {
                "category": name,
                "n_papers": n_papers,
                "n_papers_seed": n_papers_seed,
                "n_papers_basis": n_papers_basis,
                "node_id": dst,
                "edge_data": edge_data,
                "node_attrs": node_attrs,
            }
        )
    rows.sort(
        key=lambda x: (
            -x["n_papers"],
            -x["n_papers_seed"],
            str(x["category"]).casefold(),
            str(x["node_id"]),
        )
    )
    return rows[:top_n]


def _rank_score_and_basis(row: Any) -> tuple[str, str]:
    import pandas as pd

    if "docking_rrf" not in row.index:
        return "", "docking_rrf_absent"
    value = row["docking_rrf"]
    if pd.isna(value) or (isinstance(value, str) and not value.strip()):
        return "", "docking_rrf_blank"
    if _is_bool_like(value):
        raise SystemExit("Ranked target CSV column 'docking_rrf' must be numeric")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            "Ranked target CSV column 'docking_rrf' must be numeric"
        ) from exc
    if not math.isfinite(numeric):
        raise SystemExit("Ranked target CSV column 'docking_rrf' must be finite")
    return str(value), "copied_from_docking_rrf"


def _semantic_row(
    uniprot: str,
    ranked_row: Any,
    records: list[dict[str, Any]],
) -> dict[str, str]:
    class_values: set[str] = set()
    action_values: set[str] = set()
    direction_values: set[str] = set()
    ambiguous_fields: set[str] = set()
    explicit_fields: set[str] = set()

    categories: list[dict[str, Any]] = []
    paper_counts: dict[str, int] = {}
    for record in records:
        category = str(record["category"])
        edge_data = record["edge_data"]
        node_attrs = record["node_attrs"]
        paper_counts[category] = int(record["n_papers"])
        categories.append({"category": category, "n_papers": int(record["n_papers"])})

        class_value, class_ambiguous = _single_metadata_value(
            _metadata_values(
                edge_data=edge_data,
                node_attrs=node_attrs,
                keys=("mechanism_class",),
            ),
            allowed=MECHANISM_CLASSES,
            field="mechanism_class",
            uniprot=uniprot,
            category=category,
        )
        action_value, action_ambiguous = _single_metadata_value(
            _metadata_values(
                edge_data=edge_data,
                node_attrs=node_attrs,
                keys=("mechanism_action", "action"),
            ),
            allowed=MECHANISM_ACTIONS,
            field="mechanism_action",
            uniprot=uniprot,
            category=category,
        )
        direction_value, direction_ambiguous = _single_metadata_value(
            _metadata_values(
                edge_data=edge_data,
                node_attrs=node_attrs,
                keys=("skin_effect_direction", "effect_direction"),
            ),
            allowed=SKIN_EFFECT_DIRECTIONS,
            field="skin_effect_direction",
            uniprot=uniprot,
            category=category,
        )

        if class_value is not None:
            class_values.add(class_value)
            explicit_fields.add("mechanism_class")
        if action_value is not None:
            action_values.add(action_value)
            explicit_fields.add("mechanism_action")
        if direction_value is not None:
            direction_values.add(direction_value)
            explicit_fields.add("skin_effect_direction")
        if class_ambiguous:
            ambiguous_fields.add("mechanism_class")
        if action_ambiguous:
            ambiguous_fields.add("mechanism_action")
        if direction_ambiguous:
            ambiguous_fields.add("skin_effect_direction")

    mechanism_class = "pathway_effect"
    if len(class_values) == 1:
        mechanism_class = next(iter(class_values))
    elif len(class_values) > 1:
        ambiguous_fields.add("mechanism_class")

    mechanism_action = "unknown"
    if len(action_values) == 1:
        mechanism_action = next(iter(action_values))
    elif len(action_values) > 1:
        ambiguous_fields.add("mechanism_action")

    skin_effect_direction = "context_dependent"
    if len(direction_values) == 1:
        skin_effect_direction = next(iter(direction_values))
    elif len(direction_values) > 1:
        ambiguous_fields.add("skin_effect_direction")

    if ambiguous_fields:
        confidence = "low"
        uncertainty = "0.900"
        ood_route = "ambiguous_structured_mechanism_metadata"
    elif {"mechanism_class", "mechanism_action", "skin_effect_direction"} <= explicit_fields:
        confidence = "high"
        uncertainty = "0.100"
        ood_route = "none"
    elif explicit_fields:
        confidence = "medium"
        uncertainty = "0.500"
        ood_route = "missing_structured_mechanism_metadata"
    else:
        confidence = "low"
        uncertainty = "0.800"
        ood_route = "missing_structured_mechanism_metadata"

    if {"mechanism_class", "mechanism_action", "skin_effect_direction"} <= explicit_fields:
        coverage = "kg_efficacy_with_structured_mechanism_metadata"
    elif explicit_fields:
        coverage = "kg_efficacy_with_partial_mechanism_metadata"
    else:
        coverage = "kg_efficacy_only"

    biochemical_rank_score, biochemical_basis = _rank_score_and_basis(ranked_row)
    basis_parts = [
        "kg_associated_with_efficacy_categories",
        "mechanism_class:"
        + (
            "explicit_structured_metadata"
            if "mechanism_class" in explicit_fields
            else "default_pathway_effect"
        ),
        "mechanism_action:"
        + (
            "explicit_structured_metadata"
            if "mechanism_action" in explicit_fields
            else "unknown_no_structured_metadata"
        ),
        "skin_effect_direction:"
        + (
            "explicit_structured_metadata"
            if "skin_effect_direction" in explicit_fields
            else "context_dependent_no_structured_metadata"
        ),
        f"biochemical_rank_score:{biochemical_basis}",
    ]
    if ambiguous_fields:
        basis_parts.append("ambiguous_fields:" + ",".join(sorted(ambiguous_fields)))

    provenance = {
        "categories": categories,
        "derivation_basis": basis_parts,
        "paper_counts": paper_counts,
        "target": uniprot,
    }

    return {
        "biochemical_rank_score": biochemical_rank_score,
        "biochemical_score_is_probability": "false",
        "mechanism_class": mechanism_class,
        "mechanism_action": mechanism_action,
        "skin_effect_direction": skin_effect_direction,
        "skin_effect_confidence": confidence,
        "prediction_coverage": coverage,
        "uncertainty_score": uncertainty,
        "ood_route": ood_route,
        "evidence_provenance_json": json.dumps(
            provenance,
            sort_keys=True,
            separators=(",", ":"),
        ),
    }


def _validate_semantic_rows(rows: list[dict[str, str]]) -> None:
    for idx, row in enumerate(rows):
        if row["biochemical_score_is_probability"] != "false":
            raise SystemExit(
                "Internal Stage 3 semantic label error: "
                f"biochemical_score_is_probability must be false at row {idx}"
            )
        if row["mechanism_class"] not in MECHANISM_CLASSES:
            raise SystemExit(
                f"Internal Stage 3 semantic label error: unsupported mechanism_class "
                f"{row['mechanism_class']!r} at row {idx}"
            )
        if row["mechanism_action"] not in MECHANISM_ACTIONS:
            raise SystemExit(
                f"Internal Stage 3 semantic label error: unsupported mechanism_action "
                f"{row['mechanism_action']!r} at row {idx}"
            )
        if row["skin_effect_direction"] not in SKIN_EFFECT_DIRECTIONS:
            raise SystemExit(
                "Internal Stage 3 semantic label error: unsupported "
                f"skin_effect_direction {row['skin_effect_direction']!r} at row {idx}"
            )
        if row["skin_effect_confidence"] not in SKIN_EFFECT_CONFIDENCES:
            raise SystemExit(
                "Internal Stage 3 semantic label error: unsupported "
                f"skin_effect_confidence {row['skin_effect_confidence']!r} at row {idx}"
            )
        if row["prediction_coverage"] not in PREDICTION_COVERAGES:
            raise SystemExit(
                "Internal Stage 3 semantic label error: unsupported "
                f"prediction_coverage {row['prediction_coverage']!r} at row {idx}"
            )
        if row["ood_route"] not in OOD_ROUTES:
            raise SystemExit(
                f"Internal Stage 3 semantic label error: unsupported ood_route "
                f"{row['ood_route']!r} at row {idx}"
            )
        try:
            uncertainty = float(row["uncertainty_score"])
        except ValueError as exc:
            raise SystemExit(
                f"Internal Stage 3 semantic label error: uncertainty_score must be numeric at row {idx}"
            ) from exc
        if not 0.0 <= uncertainty <= 1.0:
            raise SystemExit(
                "Internal Stage 3 semantic label error: uncertainty_score must be "
                f"in [0, 1] at row {idx}"
            )
        try:
            provenance = json.loads(row["evidence_provenance_json"])
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"Internal Stage 3 semantic label error: provenance JSON is malformed at row {idx}"
            ) from exc
        if sorted(provenance) != ["categories", "derivation_basis", "paper_counts", "target"]:
            raise SystemExit(
                "Internal Stage 3 semantic label error: provenance JSON has "
                f"unexpected keys at row {idx}"
            )


def _source_labels(value: object, row_idx: int) -> list[str]:
    import pandas as pd

    if pd.isna(value):
        raise SystemExit(
            "Ranked target CSV column 'sources' contains blank values at "
            f"row index {row_idx}"
        )
    text = str(value).strip()
    if not text:
        raise SystemExit(
            "Ranked target CSV column 'sources' contains blank values at "
            f"row index {row_idx}"
        )
    labels = [part.strip() for part in text.split(";")]
    if any(label == "" for label in labels):
        raise SystemExit(
            "Ranked target CSV column 'sources' contains empty labels at "
            f"row index {row_idx}"
        )
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        shown = ", ".join(duplicates[:10])
        suffix = "..." if len(duplicates) > 10 else ""
        raise SystemExit(
            "Ranked target CSV column 'sources' contains duplicate labels at "
            f"row index {row_idx}: {shown}{suffix}"
        )
    return labels


def _is_bool_like(value: object) -> bool:
    return (
        isinstance(value, bool)
        or type(value).__name__ == "bool_"
        or (
            isinstance(value, str)
            and value.strip().lower() in {"true", "false"}
        )
    )


def _validate_publication_evidence_columns(df: Any, path: Path) -> None:
    import pandas as pd

    required = {"final_score", "source_count", "sources"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(
            "Ranked target CSV missing required publication column(s) "
            f"{missing}: {path}"
        )
    for idx, value in df["final_score"].items():
        if pd.isna(value) or (isinstance(value, str) and not value.strip()):
            raise SystemExit(
                "Ranked target CSV column 'final_score' contains blank values "
                f"at row index {int(idx)}: {path}"
            )
        if _is_bool_like(value):
            raise SystemExit(
                "Ranked target CSV column 'final_score' must be numeric at "
                f"row index {int(idx)}: {path}"
            )
        try:
            score = float(value)
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                "Ranked target CSV column 'final_score' must be numeric at "
                f"row index {int(idx)}: {path}"
            ) from exc
        if not math.isfinite(score):
            raise SystemExit(
                "Ranked target CSV column 'final_score' must be finite at "
                f"row index {int(idx)}: {path}"
            )
    for idx, row in df.iterrows():
        if _is_bool_like(row["source_count"]):
            raise SystemExit(
                "Ranked target CSV column 'source_count' must be an integer at "
                f"row index {int(idx)}: {path}"
            )
        try:
            source_count_raw = float(row["source_count"])
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                "Ranked target CSV column 'source_count' must be an integer at "
                f"row index {int(idx)}: {path}"
            ) from exc
        if not math.isfinite(source_count_raw) or not source_count_raw.is_integer():
            raise SystemExit(
                "Ranked target CSV column 'source_count' must be an integer at "
                f"row index {int(idx)}: {path}"
            )
        source_count = int(source_count_raw)
        if source_count < 1:
            raise SystemExit(
                "Ranked target CSV column 'source_count' must be >= 1 at "
                f"row index {int(idx)}: {path}"
            )
        labels = _source_labels(row["sources"], int(idx))
        if source_count != len(labels):
            raise SystemExit(
                f"Ranked target CSV source_count={source_count} but sources "
                f"lists {len(labels)} label(s) at row index {int(idx)}: {path}"
            )


def read_ranked_targets(path: Path) -> Any:
    if not _nonempty(path):
        raise SystemExit(f"Ranked target CSV is required and must be non-empty: {path}")
    import pandas as pd
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise SystemExit(f"Ranked target CSV failed to parse: {path}: {exc}") from exc
    if "target_id" not in df.columns:
        raise SystemExit(f"Ranked target CSV missing required column 'target_id': {path}")
    if df.empty:
        raise SystemExit(f"Ranked target CSV contains no rows: {path}")
    invalid = [
        int(idx) for idx, value in df["target_id"].items()
        if pd.isna(value) or not str(value).strip()
    ]
    if invalid:
        shown = ", ".join(str(idx) for idx in invalid[:10])
        suffix = "..." if len(invalid) > 10 else ""
        raise SystemExit(
            "Ranked target CSV column 'target_id' contains blank values at "
            f"row index(es) {shown}{suffix}: {path}"
        )
    normalized_target_ids = df["target_id"].astype(str).str.strip()
    duplicate_target_ids = sorted(normalized_target_ids[normalized_target_ids.duplicated()].unique())
    if duplicate_target_ids:
        shown = ", ".join(duplicate_target_ids[:10])
        suffix = "..." if len(duplicate_target_ids) > 10 else ""
        raise SystemExit(
            "Ranked target CSV column 'target_id' contains duplicate values after "
            f"trimming whitespace: {shown}{suffix}: {path}"
        )
    df["target_id"] = normalized_target_ids
    _validate_publication_evidence_columns(df, path)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranked-csv", required=True, type=Path)
    parser.add_argument("--kg-graphml", required=True, type=Path)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument(
        "--allow-missing-efficacy",
        action="store_true",
        help=(
            "Keep targets without KG edges as explicit unknown annotations. "
            "This is used by Daina-primary fast runs and never filters targets."
        ),
    )
    parser.add_argument("--out-csv", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    _remove_outputs(args.out_csv)
    if args.top_n < 1:
        raise SystemExit(f"--top-n must be >= 1: {args.top_n}")
    import pandas as pd
    df = read_ranked_targets(args.ranked_csv)
    graph = load_graph(args.kg_graphml)
    LOG.info("KG loaded — nodes=%d edges=%d",
             graph.number_of_nodes(), graph.number_of_edges())

    eff_rows: list[dict[str, str]] = []
    semantic_rows: list[dict[str, str]] = []
    missing_efficacy: list[str] = []
    for idx, uid in df["target_id"].astype(str).str.strip().items():
        records = top_efficacy_records(graph, uid, args.top_n)
        if not records:
            missing_efficacy.append(uid)
        row = {f"efficacy_top{i + 1}": "" for i in range(args.top_n)}
        for i, record in enumerate(records[:args.top_n]):
            name = record["category"]
            n_papers = record["n_papers"]
            if record["n_papers_basis"].startswith("curated_seed"):
                row[f"efficacy_top{i + 1}"] = (
                    f"{name} (curated prior; {n_papers} counted papers)"
                )
            else:
                row[f"efficacy_top{i + 1}"] = f"{name} ({n_papers} papers)"
        eff_rows.append(row)
        semantic_rows.append(_semantic_row(uid, df.loc[idx], records))

    if missing_efficacy and not args.allow_missing_efficacy:
        shown = ", ".join(missing_efficacy[:10])
        suffix = "..." if len(missing_efficacy) > 10 else ""
        raise SystemExit(
            "Skin-efficacy KG missing efficacy evidence for ranked target(s): "
            f"{shown}{suffix}"
        )
    if missing_efficacy:
        LOG.warning(
            "Skin-efficacy KG has no efficacy edge for %d ranked target(s); "
            "preserving them with unknown/context-dependent annotations",
            len(missing_efficacy),
        )

    eff_df = pd.DataFrame(eff_rows)
    _validate_semantic_rows(semantic_rows)
    semantic_df = pd.DataFrame(semantic_rows, columns=SEMANTIC_COLUMNS)
    out_df = pd.concat([df.reset_index(drop=True), eff_df, semantic_df], axis=1)
    _write_csv_atomic(out_df, args.out_csv)
    LOG.info("Wrote %s", args.out_csv)


if __name__ == "__main__":
    main()
