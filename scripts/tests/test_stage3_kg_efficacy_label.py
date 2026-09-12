"""Unit tests for Stage 3 KG efficacy label shaping."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage3_kg_efficacy_label import (  # noqa: E402
    load_graph,
    top_efficacy,
    top_efficacy_records,
)

ROOT = Path(__file__).resolve().parents[2]


def test_top_efficacy_returns_empty_when_gene_absent() -> None:
    graph = nx.MultiDiGraph()
    assert top_efficacy(graph, "P12345", top_n=3) == []


def test_top_efficacy_sorts_by_papers_and_limits() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node("gene:P12345", type="Gene")
    graph.add_node("category:barrier", type="EfficacyCategory", name="Barrier")
    graph.add_node("category:anti", type="EfficacyCategory", name="Anti-aging")
    graph.add_edge("gene:P12345", "category:barrier", n_papers=2)
    graph.add_edge("gene:P12345", "category:anti", n_papers=8)

    assert top_efficacy(graph, "P12345", top_n=1) == [("Anti-aging", 8)]


def test_top_efficacy_ties_are_insertion_order_independent() -> None:
    def graph_with_order(categories: list[tuple[str, str]]) -> nx.MultiDiGraph:
        graph = nx.MultiDiGraph()
        graph.add_node("gene:P12345", type="Gene")
        for node_id, name in categories:
            graph.add_node(node_id, type="EfficacyCategory", name=name)
            graph.add_edge("gene:P12345", node_id, n_papers=4)
        return graph

    forward = graph_with_order(
        [("category:barrier", "Barrier"), ("category:anti", "Anti-aging")]
    )
    reverse = graph_with_order(
        [("category:anti", "Anti-aging"), ("category:barrier", "Barrier")]
    )

    expected = [("Anti-aging", 4), ("Barrier", 4)]
    assert top_efficacy(forward, "P12345") == expected
    assert top_efficacy(reverse, "P12345") == expected
    assert [row["category"] for row in top_efficacy_records(forward, "P12345")] == [
        "Anti-aging",
        "Barrier",
    ]
    assert [row["category"] for row in top_efficacy_records(reverse, "P12345")] == [
        "Anti-aging",
        "Barrier",
    ]


def test_load_graph_fails_when_graphml_missing(tmp_path: Path) -> None:
    try:
        load_graph(tmp_path / "missing.graphml")
    except SystemExit as exc:
        assert "Skin-efficacy KG GraphML is required" in str(exc)
    else:
        raise AssertionError("missing KG GraphML should fail closed")


def run_labeler(
    tmp_path: Path,
    *,
    ranked_csv: Path,
    kg_graphml: Path,
    top_n: int = 3,
) -> subprocess.CompletedProcess[str]:
    (tmp_path / "ranked_with_efficacy.csv").write_text("stale\n")
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage3_kg_efficacy_label.py"),
            "--ranked-csv",
            str(ranked_csv),
            "--kg-graphml",
            str(kg_graphml),
            "--top-n",
            str(top_n),
            "--out-csv",
            str(tmp_path / "ranked_with_efficacy.csv"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def write_kg(path: Path) -> None:
    graph = nx.MultiDiGraph()
    graph.add_node("gene:P12345", type="Gene")
    graph.add_node("category:anti", type="EfficacyCategory", name="Anti-aging")
    graph.add_edge("gene:P12345", "category:anti", n_papers=8)
    nx.write_graphml(graph, path)


def ranked_row(target_id: str = "P12345") -> dict[str, object]:
    return {
        "target_id": target_id,
        "final_score": 0.9,
        "source_count": 3,
        "sources": "autodock;gnina;rtmscore",
    }


def test_cli_fails_when_ranked_csv_missing_target_id(tmp_path: Path) -> None:
    ranked = tmp_path / "ranked.csv"
    kg = tmp_path / "kg.graphml"
    pd.DataFrame([{"not_target_id": "P12345"}]).to_csv(ranked, index=False)
    write_kg(kg)

    res = run_labeler(tmp_path, ranked_csv=ranked, kg_graphml=kg)

    assert res.returncode != 0
    assert "missing required column 'target_id'" in res.stderr
    assert not (tmp_path / "ranked_with_efficacy.csv").exists()


def test_cli_fails_when_ranked_csv_empty(tmp_path: Path) -> None:
    ranked = tmp_path / "ranked.csv"
    kg = tmp_path / "kg.graphml"
    ranked.write_text("")
    write_kg(kg)

    res = run_labeler(tmp_path, ranked_csv=ranked, kg_graphml=kg)

    assert res.returncode != 0
    assert "Ranked target CSV is required and must be non-empty" in res.stderr
    assert not (tmp_path / "ranked_with_efficacy.csv").exists()


def test_cli_fails_when_ranked_csv_has_no_rows(tmp_path: Path) -> None:
    ranked = tmp_path / "ranked.csv"
    kg = tmp_path / "kg.graphml"
    pd.DataFrame(columns=["target_id"]).to_csv(ranked, index=False)
    write_kg(kg)

    res = run_labeler(tmp_path, ranked_csv=ranked, kg_graphml=kg)

    assert res.returncode != 0
    assert "Ranked target CSV contains no rows" in res.stderr
    assert not (tmp_path / "ranked_with_efficacy.csv").exists()


def test_cli_fails_when_ranked_csv_has_blank_target_id(tmp_path: Path) -> None:
    ranked = tmp_path / "ranked.csv"
    kg = tmp_path / "kg.graphml"
    pd.DataFrame([{"target_id": "P12345"}, {"target_id": None}]).to_csv(
        ranked,
        index=False,
    )
    write_kg(kg)

    res = run_labeler(tmp_path, ranked_csv=ranked, kg_graphml=kg)

    assert res.returncode != 0
    assert "column 'target_id' contains blank values" in res.stderr
    assert not (tmp_path / "ranked_with_efficacy.csv").exists()


def test_cli_fails_when_ranked_csv_has_duplicate_target_id(tmp_path: Path) -> None:
    ranked = tmp_path / "ranked.csv"
    kg = tmp_path / "kg.graphml"
    pd.DataFrame(
        [{"target_id": "P12345"}, {"target_id": " P12345 "}],
    ).to_csv(ranked, index=False)
    write_kg(kg)

    res = run_labeler(tmp_path, ranked_csv=ranked, kg_graphml=kg)

    assert res.returncode != 0
    assert "column 'target_id' contains duplicate values" in res.stderr
    assert "P12345" in res.stderr
    assert not (tmp_path / "ranked_with_efficacy.csv").exists()


def test_cli_fails_when_top_n_is_invalid(tmp_path: Path) -> None:
    ranked = tmp_path / "ranked.csv"
    kg = tmp_path / "kg.graphml"
    pd.DataFrame([{"target_id": "P12345"}]).to_csv(ranked, index=False)
    write_kg(kg)

    res = run_labeler(tmp_path, ranked_csv=ranked, kg_graphml=kg, top_n=0)

    assert res.returncode != 0
    assert "--top-n must be >= 1" in res.stderr
    assert not (tmp_path / "ranked_with_efficacy.csv").exists()


def test_cli_fails_when_kg_graphml_missing(tmp_path: Path) -> None:
    ranked = tmp_path / "ranked.csv"
    pd.DataFrame([ranked_row()]).to_csv(ranked, index=False)

    res = run_labeler(tmp_path, ranked_csv=ranked, kg_graphml=tmp_path / "missing.graphml")

    assert res.returncode != 0
    assert "Skin-efficacy KG GraphML is required" in res.stderr
    assert not (tmp_path / "ranked_with_efficacy.csv").exists()


def test_cli_fails_when_ranked_target_missing_kg_efficacy(tmp_path: Path) -> None:
    ranked = tmp_path / "ranked.csv"
    kg = tmp_path / "kg.graphml"
    pd.DataFrame(
        [ranked_row("P12345"), ranked_row("P99999")],
    ).to_csv(ranked, index=False)
    write_kg(kg)

    res = run_labeler(tmp_path, ranked_csv=ranked, kg_graphml=kg)

    assert res.returncode != 0
    assert (
        "Skin-efficacy KG missing efficacy evidence for ranked target(s): P99999"
    ) in res.stderr
    assert not (tmp_path / "ranked_with_efficacy.csv").exists()


def test_cli_rejects_kg_edge_without_positive_paper_evidence(tmp_path: Path) -> None:
    ranked = tmp_path / "ranked.csv"
    kg = tmp_path / "kg.graphml"
    pd.DataFrame([ranked_row()]).to_csv(ranked, index=False)
    graph = nx.MultiDiGraph()
    graph.add_node("gene:P12345", type="Gene")
    graph.add_node("category:anti", type="EfficacyCategory", name="Anti-aging")
    graph.add_edge("gene:P12345", "category:anti", n_papers=0)
    nx.write_graphml(graph, kg)

    res = run_labeler(tmp_path, ranked_csv=ranked, kg_graphml=kg)

    assert res.returncode != 0
    assert "positive counted paper evidence" in res.stderr
    assert not (tmp_path / "ranked_with_efficacy.csv").exists()


def test_curated_seed_weight_is_not_presented_as_counted_papers(tmp_path: Path) -> None:
    ranked = tmp_path / "ranked.csv"
    kg = tmp_path / "kg.graphml"
    pd.DataFrame([ranked_row()]).to_csv(ranked, index=False)
    graph = nx.MultiDiGraph()
    graph.add_node("gene:P12345", type="Gene")
    graph.add_node("category:anti", type="EfficacyCategory", name="Anti-aging")
    graph.add_edge(
        "gene:P12345",
        "category:anti",
        n_papers=0,
        n_papers_counted=0,
        n_papers_seed=562,
        n_papers_basis="curated_seed",
    )
    nx.write_graphml(graph, kg)

    res = run_labeler(tmp_path, ranked_csv=ranked, kg_graphml=kg)

    assert res.returncode == 0, res.stderr
    out = pd.read_csv(tmp_path / "ranked_with_efficacy.csv", dtype=str)
    assert out.loc[0, "efficacy_top1"] == (
        "Anti-aging (curated prior; 0 counted papers)"
    )
    assert "562 papers" not in out.loc[0, "efficacy_top1"]
    provenance = json.loads(out.loc[0, "evidence_provenance_json"])
    assert provenance["paper_counts"] == {"Anti-aging": 0}


def test_cli_rejects_duplicate_kg_category_labels_for_target(tmp_path: Path) -> None:
    ranked = tmp_path / "ranked.csv"
    kg = tmp_path / "kg.graphml"
    pd.DataFrame([ranked_row()]).to_csv(ranked, index=False)
    graph = nx.MultiDiGraph()
    graph.add_node("gene:P12345", type="Gene")
    graph.add_node("category:anti_primary", type="EfficacyCategory", name="Anti-aging")
    graph.add_node("category:anti_alias", type="EfficacyCategory", name=" Anti-aging ")
    graph.add_edge("gene:P12345", "category:anti_primary", n_papers=8)
    graph.add_edge("gene:P12345", "category:anti_alias", n_papers=5)
    nx.write_graphml(graph, kg)

    res = run_labeler(tmp_path, ranked_csv=ranked, kg_graphml=kg)

    assert res.returncode != 0
    assert "duplicate efficacy category label" in res.stderr
    assert "Anti-aging" in res.stderr
    assert not (tmp_path / "ranked_with_efficacy.csv").exists()


def test_cli_trims_ranked_target_ids_in_output(tmp_path: Path) -> None:
    ranked = tmp_path / "ranked.csv"
    kg = tmp_path / "kg.graphml"
    pd.DataFrame([ranked_row(" P12345 ")]).to_csv(
        ranked,
        index=False,
    )
    write_kg(kg)

    res = run_labeler(tmp_path, ranked_csv=ranked, kg_graphml=kg)

    assert res.returncode == 0, res.stderr
    out = pd.read_csv(tmp_path / "ranked_with_efficacy.csv")
    assert out.loc[0, "target_id"] == "P12345"
    assert out.loc[0, "efficacy_top1"] == "Anti-aging (8 papers)"


def test_cli_writes_efficacy_labels_with_valid_inputs(tmp_path: Path) -> None:
    ranked = tmp_path / "ranked.csv"
    kg = tmp_path / "kg.graphml"
    pd.DataFrame([ranked_row()]).to_csv(ranked, index=False)
    write_kg(kg)

    res = run_labeler(tmp_path, ranked_csv=ranked, kg_graphml=kg)

    assert res.returncode == 0, res.stderr
    out = pd.read_csv(tmp_path / "ranked_with_efficacy.csv")
    assert out.loc[0, "efficacy_top1"] == "Anti-aging (8 papers)"


def test_cli_appends_semantics_without_changing_legacy_scores_or_labels(
    tmp_path: Path,
) -> None:
    ranked = tmp_path / "ranked.csv"
    kg = tmp_path / "kg.graphml"
    ranked_df = pd.DataFrame(
        [{
            **ranked_row(),
            "docking_rrf": 0.42,
            "skin_score": 0.73,
        }],
    )
    ranked_df.to_csv(ranked, index=False)
    write_kg(kg)

    res = run_labeler(tmp_path, ranked_csv=ranked, kg_graphml=kg)

    assert res.returncode == 0, res.stderr
    out = pd.read_csv(tmp_path / "ranked_with_efficacy.csv", dtype=str)
    assert out.loc[0, "final_score"] == "0.9"
    assert out.loc[0, "docking_rrf"] == "0.42"
    assert out.loc[0, "skin_score"] == "0.73"
    assert out.loc[0, "efficacy_top1"] == "Anti-aging (8 papers)"
    assert out.loc[0, "biochemical_rank_score"] == "0.42"
    assert out.loc[0, "biochemical_score_is_probability"] == "false"
    assert out.loc[0, "mechanism_class"] == "pathway_effect"
    assert out.loc[0, "mechanism_action"] == "unknown"
    assert out.loc[0, "skin_effect_direction"] == "context_dependent"
    assert out.loc[0, "skin_effect_confidence"] == "low"
    assert out.loc[0, "prediction_coverage"] == "kg_efficacy_only"
    assert out.loc[0, "ood_route"] == "missing_structured_mechanism_metadata"


def test_cli_semantics_leave_biochemical_rank_blank_without_docking_rrf(
    tmp_path: Path,
) -> None:
    ranked = tmp_path / "ranked.csv"
    kg = tmp_path / "kg.graphml"
    pd.DataFrame([ranked_row()]).to_csv(ranked, index=False)
    write_kg(kg)

    res = run_labeler(tmp_path, ranked_csv=ranked, kg_graphml=kg)

    assert res.returncode == 0, res.stderr
    out = pd.read_csv(tmp_path / "ranked_with_efficacy.csv", dtype=str).fillna("")
    assert out.loc[0, "biochemical_rank_score"] == ""
    provenance = json.loads(out.loc[0, "evidence_provenance_json"])
    assert (
        "biochemical_rank_score:docking_rrf_absent"
        in provenance["derivation_basis"]
    )


@pytest.mark.parametrize(
    ("docking_rrf", "message"),
    [
        ("not-a-score", "must be numeric"),
        (True, "must be numeric"),
        (float("inf"), "must be finite"),
        (float("-inf"), "must be finite"),
    ],
)
def test_cli_rejects_invalid_docking_rrf(
    tmp_path: Path, docking_rrf: object, message: str
) -> None:
    ranked = tmp_path / "ranked.csv"
    kg = tmp_path / "kg.graphml"
    pd.DataFrame([{**ranked_row(), "docking_rrf": docking_rrf}]).to_csv(
        ranked, index=False
    )
    write_kg(kg)

    result = run_labeler(tmp_path, ranked_csv=ranked, kg_graphml=kg)

    assert result.returncode != 0
    assert message in result.stderr


def test_cli_ambiguous_structured_kg_yields_unknown_context_dependent_low_confidence(
    tmp_path: Path,
) -> None:
    ranked = tmp_path / "ranked.csv"
    kg = tmp_path / "kg.graphml"
    pd.DataFrame([ranked_row()]).to_csv(ranked, index=False)
    graph = nx.MultiDiGraph()
    graph.add_node("gene:P12345", type="Gene")
    graph.add_node(
        "category:anti",
        type="EfficacyCategory",
        name="Anti-aging",
        mechanism_action="activation",
        skin_effect_direction="beneficial",
    )
    graph.add_node(
        "category:barrier",
        type="EfficacyCategory",
        name="Barrier",
        mechanism_action="inhibition",
        skin_effect_direction="adverse",
    )
    graph.add_edge("gene:P12345", "category:anti", n_papers=8)
    graph.add_edge("gene:P12345", "category:barrier", n_papers=7)
    nx.write_graphml(graph, kg)

    res = run_labeler(tmp_path, ranked_csv=ranked, kg_graphml=kg)

    assert res.returncode == 0, res.stderr
    out = pd.read_csv(tmp_path / "ranked_with_efficacy.csv", dtype=str)
    assert out.loc[0, "mechanism_class"] == "pathway_effect"
    assert out.loc[0, "mechanism_action"] == "unknown"
    assert out.loc[0, "skin_effect_direction"] == "context_dependent"
    assert out.loc[0, "skin_effect_confidence"] == "low"
    assert out.loc[0, "ood_route"] == "ambiguous_structured_mechanism_metadata"
    provenance = json.loads(out.loc[0, "evidence_provenance_json"])
    assert (
        "ambiguous_fields:mechanism_action,skin_effect_direction"
        in provenance["derivation_basis"]
    )


def test_cli_honors_explicit_direct_binding_structured_metadata(
    tmp_path: Path,
) -> None:
    ranked = tmp_path / "ranked.csv"
    kg = tmp_path / "kg.graphml"
    pd.DataFrame([{**ranked_row(), "docking_rrf": 0.51}]).to_csv(
        ranked,
        index=False,
    )
    graph = nx.MultiDiGraph()
    graph.add_node("gene:P12345", type="Gene")
    graph.add_node("category:anti", type="EfficacyCategory", name="Anti-aging")
    graph.add_edge(
        "gene:P12345",
        "category:anti",
        n_papers=8,
        mechanism_class="direct_binding",
        mechanism_action="inhibition",
        skin_effect_direction="beneficial",
    )
    nx.write_graphml(graph, kg)

    res = run_labeler(tmp_path, ranked_csv=ranked, kg_graphml=kg)

    assert res.returncode == 0, res.stderr
    out = pd.read_csv(tmp_path / "ranked_with_efficacy.csv", dtype=str)
    assert out.loc[0, "biochemical_rank_score"] == "0.51"
    assert out.loc[0, "biochemical_score_is_probability"] == "false"
    assert out.loc[0, "mechanism_class"] == "direct_binding"
    assert out.loc[0, "mechanism_action"] == "inhibition"
    assert out.loc[0, "skin_effect_direction"] == "beneficial"
    assert out.loc[0, "skin_effect_confidence"] == "high"
    assert (
        out.loc[0, "prediction_coverage"]
        == "kg_efficacy_with_structured_mechanism_metadata"
    )
    assert out.loc[0, "uncertainty_score"] == "0.100"
    assert out.loc[0, "ood_route"] == "none"


def test_cli_writes_deterministic_canonical_provenance(tmp_path: Path) -> None:
    ranked = tmp_path / "ranked.csv"
    kg = tmp_path / "kg.graphml"
    pd.DataFrame([{**ranked_row(), "docking_rrf": 0.51}]).to_csv(
        ranked,
        index=False,
    )
    write_kg(kg)

    first = run_labeler(tmp_path, ranked_csv=ranked, kg_graphml=kg)
    assert first.returncode == 0, first.stderr
    out1 = pd.read_csv(tmp_path / "ranked_with_efficacy.csv", dtype=str)
    provenance1 = out1.loc[0, "evidence_provenance_json"]

    second = run_labeler(tmp_path, ranked_csv=ranked, kg_graphml=kg)
    assert second.returncode == 0, second.stderr
    out2 = pd.read_csv(tmp_path / "ranked_with_efficacy.csv", dtype=str)
    provenance2 = out2.loc[0, "evidence_provenance_json"]

    assert provenance1 == provenance2
    assert " " not in provenance1
    payload = json.loads(provenance1)
    assert list(payload.keys()) == [
        "categories",
        "derivation_basis",
        "paper_counts",
        "target",
    ]
    assert payload["target"] == "P12345"
    assert payload["categories"] == [{"category": "Anti-aging", "n_papers": 8}]
    assert payload["paper_counts"] == {"Anti-aging": 8}
    assert "timestamp" not in provenance1.lower()


def test_cli_rejects_unsupported_structured_semantic_metadata(tmp_path: Path) -> None:
    ranked = tmp_path / "ranked.csv"
    kg = tmp_path / "kg.graphml"
    pd.DataFrame([ranked_row()]).to_csv(ranked, index=False)
    graph = nx.MultiDiGraph()
    graph.add_node("gene:P12345", type="Gene")
    graph.add_node("category:anti", type="EfficacyCategory", name="Anti-aging")
    graph.add_edge(
        "gene:P12345",
        "category:anti",
        n_papers=8,
        mechanism_class="category_inferred_binding",
    )
    nx.write_graphml(graph, kg)

    res = run_labeler(tmp_path, ranked_csv=ranked, kg_graphml=kg)

    assert res.returncode != 0
    assert "mechanism_class has unsupported value" in res.stderr
    assert "category_inferred_binding" in res.stderr
    assert not (tmp_path / "ranked_with_efficacy.csv").exists()


def test_cli_fails_when_ranked_csv_missing_publication_evidence_columns(
    tmp_path: Path,
) -> None:
    ranked = tmp_path / "ranked.csv"
    kg = tmp_path / "kg.graphml"
    pd.DataFrame([{
        "target_id": "P12345",
        "final_score": 0.9,
        "sources": "autodock;gnina;rtmscore",
    }]).to_csv(ranked, index=False)
    write_kg(kg)

    res = run_labeler(tmp_path, ranked_csv=ranked, kg_graphml=kg)

    assert res.returncode != 0
    assert "Ranked target CSV missing required publication column" in res.stderr
    assert "source_count" in res.stderr
    assert not (tmp_path / "ranked_with_efficacy.csv").exists()


def test_cli_fails_when_ranked_source_count_is_noninteger(
    tmp_path: Path,
) -> None:
    ranked = tmp_path / "ranked.csv"
    kg = tmp_path / "kg.graphml"
    row = ranked_row()
    row["source_count"] = 2.5
    pd.DataFrame([row]).to_csv(ranked, index=False)
    write_kg(kg)

    res = run_labeler(tmp_path, ranked_csv=ranked, kg_graphml=kg)

    assert res.returncode != 0
    assert "column 'source_count' must be an integer" in res.stderr
    assert not (tmp_path / "ranked_with_efficacy.csv").exists()


def test_cli_fails_when_ranked_sources_are_missing(
    tmp_path: Path,
) -> None:
    ranked = tmp_path / "ranked.csv"
    kg = tmp_path / "kg.graphml"
    row = ranked_row()
    row["source_count"] = 1
    row["sources"] = None
    pd.DataFrame([row]).to_csv(ranked, index=False)
    write_kg(kg)

    res = run_labeler(tmp_path, ranked_csv=ranked, kg_graphml=kg)

    assert res.returncode != 0
    assert "column 'sources' contains blank values" in res.stderr
    assert not (tmp_path / "ranked_with_efficacy.csv").exists()
