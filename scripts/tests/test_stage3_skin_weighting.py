"""Unit tests for Stage 3 v3 skin-expression weighting (§7.2)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage3_skin_weighting import apply_skin_weight, read_decision  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def write_drug_warnings(path: Path) -> Path:
    path.write_text(
        '{"warnings": [], "n_warnings": 0, "reference_status": "ok"}\n'
    )
    return path


def test_skin_weight_combines_axes() -> None:
    finals = apply_skin_weight(
        docking_scores={"A": 1.0, "B": 0.5, "C": 0.1},
        skin_scores={"A": 0.9, "B": 0.6, "C": 0.3},
        skin_weight=0.30,
    )
    # A: 0.70 × 1.0 + 0.30 × 0.9 = 0.97
    assert abs(finals["A"] - 0.97) < 1e-9


def test_below_threshold_triggers_penalty() -> None:
    finals = apply_skin_weight(
        docking_scores={"A": 1.0, "Orphan": 0.5},
        skin_scores={"A": 0.5, "Orphan": 0.0},
        skin_weight=0.30,
        min_threshold=0.05,
        penalty_factor=0.30,
    )
    # Orphan (skin=0) normalised dock = 0 (because it's the min in the pair),
    # final = (0.70 × 0 + 0.30 × 0) × 0.30 = 0.0
    # but A is the max → final = 0.70 × 1 + 0.30 × 0.5 = 0.85
    assert finals["A"] > finals["Orphan"]
    assert finals["Orphan"] == 0.0


def test_orphan_target_penalised_to_30pct() -> None:
    # Construct a case where Orphan has a competitive docking score so we can
    # see the 30 % hard-cap kick in.
    finals = apply_skin_weight(
        docking_scores={"Skin": 0.5, "Orphan": 1.0},
        skin_scores={"Skin": 0.9, "Orphan": 0.0},
        skin_weight=0.30,
    )
    # Pre-penalty Orphan: 0.70 × 1 + 0.30 × 0 = 0.70
    # After 0.30 penalty: 0.21
    assert abs(finals["Orphan"] - 0.21) < 1e-9
    # Skin: 0.70 × 0 + 0.30 × 0.9 = 0.27 (no penalty)
    assert finals["Skin"] > finals["Orphan"]


def test_skin_weight_zero_recovers_docking_only() -> None:
    finals = apply_skin_weight(
        docking_scores={"A": 1.0, "B": 0.0},
        skin_scores={"A": 0.0, "B": 0.0},
        skin_weight=0.0,
        min_threshold=0.05,
        penalty_factor=0.30,
    )
    # Both targets have skin<0.05 → both penalised to 0.30 × normalised docking
    assert abs(finals["A"] - 0.30 * 1.0) < 1e-9
    assert finals["B"] == 0.0


def test_skin_weight_constant_nonzero_known_prior_scores_still_contribute() -> None:
    finals = apply_skin_weight(
        docking_scores={"A": 1.0, "B": 0.0},
        skin_scores={"A": 0.2, "B": 0.8},
        skin_weight=0.0,
        known_target_prior_weight=0.5,
        known_target_prior_scores={"A": 0.72, "B": 0.72},
    )
    assert abs(finals["A"] - 1.0) < 1e-9
    assert abs(finals["B"] - 0.5) < 1e-9


def test_known_target_prior_min_score_and_power_shapes_prior_influence() -> None:
    finals = apply_skin_weight(
        docking_scores={"A": 1.0, "B": 0.5},
        skin_scores={"A": 0.2, "B": 0.8},
        skin_weight=0.0,
        known_target_prior_weight=1.0,
        known_target_prior_scores={"A": 0.5, "B": 0.9},
        known_target_prior_min_score=0.6,
        known_target_prior_power=2.0,
    )
    # min_score=0.6 filters out A; only B remains and is reshaped as 0.9² = 0.81.
    # then normalize over docking candidates: only B stays non-zero, so it becomes 1.0
    assert abs(finals["A"] - 0.0) < 1e-9
    assert abs(finals["B"] - 1.0) < 1e-9


def test_empty_inputs() -> None:
    assert apply_skin_weight({}, {}) == {}


def test_skin_weight_full_recovers_skin_only() -> None:
    finals = apply_skin_weight(
        docking_scores={"A": 1.0, "B": 0.0},
        skin_scores={"A": 0.5, "B": 0.9},
        skin_weight=1.0,
    )
    # Both have skin ≥ threshold → final = 1.0 × skin
    assert abs(finals["A"] - 0.5) < 1e-9
    assert abs(finals["B"] - 0.9) < 1e-9


def test_read_decision_requires_configured_file() -> None:
    with pytest.raises(SystemExit, match="Cosmetic/drug decision file is required"):
        read_decision(None)


def test_read_decision_fails_for_configured_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.txt"
    drug = write_drug_warnings(tmp_path / "drug_warnings.json")

    with pytest.raises(SystemExit, match="Cosmetic/drug decision file is required"):
        read_decision(missing, drug)


def test_read_decision_requires_drug_warning_source(tmp_path: Path) -> None:
    path = tmp_path / "decision.txt"
    path.write_text("PROCEED\n# policy: moderate\n")

    with pytest.raises(SystemExit, match="Drug-avoidance warnings JSON is required"):
        read_decision(path)


def test_read_decision_uses_first_line(tmp_path: Path) -> None:
    path = tmp_path / "decision.txt"
    path.write_text("DOWNWEIGHT\n# policy: strict\n")
    drug = tmp_path / "drug_warnings.json"
    drug.write_text(
        '{"warnings": [{"tier": "SOFT_WARNING"}], '
        '"n_warnings": 1, "reference_status": "ok"}\n'
    )
    assert read_decision(path, drug) == "DOWNWEIGHT"


def test_read_decision_requires_policy_line(tmp_path: Path) -> None:
    path = tmp_path / "decision.txt"
    path.write_text("PROCEED\n")
    drug = write_drug_warnings(tmp_path / "drug_warnings.json")

    with pytest.raises(SystemExit, match="missing policy line"):
        read_decision(path, drug)


def run_skin_weight(
    tmp_path: Path,
    skin_tsv: Path,
    extra_args: list[str] | None = None,
    top_csv: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    top = top_csv or tmp_path / "top.csv"
    if top_csv is None:
        pd.DataFrame([{"target_id": "P1", "rrf_score": 1.0}]).to_csv(top, index=False)
    decision = tmp_path / "cosmetic_drug_decision.txt"
    decision.write_text("PROCEED\n# policy: moderate\n")
    args = list(extra_args or [])
    if "--cosmetic-decision" not in args:
        args.extend(["--cosmetic-decision", str(decision)])
    if "--drug-json" not in args:
        drug = write_drug_warnings(tmp_path / "drug_warnings.json")
        args.extend(["--drug-json", str(drug)])
    (tmp_path / "ranked.csv").write_text("stale\n")
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage3_skin_weighting.py"),
            "--top-csv",
            str(top),
            "--skin-tsv",
            str(skin_tsv),
            "--out-csv",
            str(tmp_path / "ranked.csv"),
            *args,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_fails_when_top_csv_empty(tmp_path: Path) -> None:
    top = tmp_path / "top.csv"
    top.write_text("")
    skin = tmp_path / "skin.tsv"
    pd.DataFrame([{"uniprot": "P1", "skin_score": 0.8}]).to_csv(
        skin, sep="\t", index=False
    )

    res = run_skin_weight(tmp_path, skin, top_csv=top)

    assert res.returncode != 0
    assert "Top-target CSV is required and must be non-empty" in res.stderr
    assert not (tmp_path / "ranked.csv").exists()


def test_cli_fails_when_top_csv_missing_target_id(tmp_path: Path) -> None:
    top = tmp_path / "top.csv"
    pd.DataFrame([{"rrf_score": 1.0}]).to_csv(top, index=False)
    skin = tmp_path / "skin.tsv"
    pd.DataFrame([{"uniprot": "P1", "skin_score": 0.8}]).to_csv(
        skin, sep="\t", index=False
    )

    res = run_skin_weight(tmp_path, skin, top_csv=top)

    assert res.returncode != 0
    assert "Top-target CSV missing required column 'target_id'" in res.stderr
    assert not (tmp_path / "ranked.csv").exists()


def test_cli_fails_when_top_csv_missing_known_docking_score_column(
    tmp_path: Path,
) -> None:
    top = tmp_path / "top.csv"
    top.write_text(
        "target_id,sources,source_count\n"
        "P1,autodock;gnina;rtmscore,3\n"
    )
    skin = tmp_path / "skin.tsv"
    pd.DataFrame([{"uniprot": "P1", "skin_score": 0.8}]).to_csv(
        skin,
        sep="\t",
        index=False,
    )

    res = run_skin_weight(tmp_path, skin, top_csv=top)

    assert res.returncode != 0
    assert "Top-target CSV missing docking score column" in res.stderr
    assert not (tmp_path / "ranked.csv").exists()


def test_cli_outputs_known_target_prior_columns(tmp_path: Path) -> None:
    top = tmp_path / "top.csv"
    top.write_text("target_id,rrf_score\nA,1.0\nB,0.5\n")
    skin = tmp_path / "skin.tsv"
    skin.write_text(
        "uniprot\tskin_score\nA\t0.5\nB\t0.9\n",
        encoding="utf-8",
    )
    prior = tmp_path / "known_target.csv"
    prior.write_text("target_id,prior_score\nA,0.2\nB,0.9\n", encoding="utf-8")

    res = run_skin_weight(
        tmp_path,
        skin,
        extra_args=[
            "--known-target-prior-csv",
            str(prior),
            "--skin-weight",
            "0.0",
            "--known-target-prior-weight",
            "0.5",
        ],
        top_csv=top,
    )

    assert res.returncode == 0, res.stderr
    ranked = pd.read_csv(tmp_path / "ranked.csv")
    assert "known_target_prior" in ranked.columns
    assert "known_target_prior_norm" in ranked.columns
    assert abs(ranked.loc[ranked["target_id"] == "A", "known_target_prior"].iloc[0] - 0.2) < 1e-9
    assert abs(ranked.loc[ranked["target_id"] == "A", "known_target_prior_norm"].iloc[0] - 0.0) < 1e-9
    assert (
        abs(ranked.loc[ranked["target_id"] == "B", "known_target_prior"].iloc[0] - 0.9) < 1e-9
    )
    assert ranked.loc[ranked["target_id"] == "B", "known_target_prior_norm"].iloc[0] == 1.0


def test_preserve_primary_ranking_uses_upstream_final_rank_not_daina_score(
    tmp_path: Path,
) -> None:
    top = tmp_path / "top.csv"
    pd.DataFrame(
        [
            {
                "target_id": "P1",
                "daina_rank": 1,
                "final_rank": 1,
                "rrf_score": 0.90,
                "band_rrf_score": None,
                "source_count": 1,
                "sources": "daina",
            },
            {
                "target_id": "P3",
                "daina_rank": 3,
                "final_rank": 2,
                "rrf_score": 0.70,
                "band_rrf_score": 0.04,
                "source_count": 1,
                "sources": "daina",
            },
            {
                "target_id": "P2",
                "daina_rank": 2,
                "final_rank": 3,
                "rrf_score": 0.80,
                "band_rrf_score": 0.03,
                "source_count": 1,
                "sources": "daina",
            },
        ]
    ).to_csv(top, index=False)
    skin = tmp_path / "skin.tsv"
    skin.write_text(
        "uniprot\tskin_score\nP1\t0.8\nP2\t0.8\nP3\t0.8\n",
        encoding="utf-8",
    )

    result = run_skin_weight(
        tmp_path,
        skin,
        extra_args=["--preserve-primary-ranking"],
        top_csv=top,
    )

    assert result.returncode == 0, result.stderr
    ranked = pd.read_csv(tmp_path / "ranked.csv")
    assert ranked["target_id"].tolist() == ["P1", "P3", "P2"]
    assert ranked["final_rank"].tolist() == [1, 2, 3]
    assert ranked["final_score"].tolist() == pytest.approx([1.0, 2 / 3, 1 / 3])
    assert set(ranked["final_score_semantics"]) == {"ordinal_rank"}
    assert ranked["daina_rank"].tolist() == [1, 3, 2]
    assert ranked["rrf_score"].tolist() == pytest.approx([0.9, 0.7, 0.8])
    assert ranked["band_rrf_score"].iloc[1:].tolist() == pytest.approx([0.04, 0.03])

    graph = nx.MultiDiGraph()
    graph.add_node("category:test", type="EfficacyCategory", name="Test efficacy")
    for target_id in ranked["target_id"]:
        graph.add_node(f"gene:{target_id}", type="Gene")
        graph.add_edge(f"gene:{target_id}", "category:test", n_papers=1)
    kg = tmp_path / "kg.graphml"
    nx.write_graphml(graph, kg)
    labeled = tmp_path / "ranked_with_efficacy.csv"
    kg_result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage3_kg_efficacy_label.py"),
            "--ranked-csv", str(tmp_path / "ranked.csv"),
            "--kg-graphml", str(kg),
            "--out-csv", str(labeled),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert kg_result.returncode == 0, kg_result.stderr
    final = pd.read_csv(labeled)
    assert final["target_id"].tolist() == ["P1", "P3", "P2"]
    assert final["final_score"].is_monotonic_decreasing


def test_cli_propagates_preferred_hpa_cell_type(tmp_path: Path) -> None:
    skin = tmp_path / "skin.tsv"
    skin.write_text(
        "uniprot\tskin_score\tcell_type_preferred\n"
        "P1\t0.8\tmelanocytes\n",
        encoding="utf-8",
    )

    res = run_skin_weight(tmp_path, skin)

    assert res.returncode == 0, res.stderr
    ranked = pd.read_csv(tmp_path / "ranked.csv")
    assert ranked.loc[0, "cell_type_preferred"] == "melanocytes"


def test_cli_fails_when_preferred_hpa_cell_type_is_blank(tmp_path: Path) -> None:
    skin = tmp_path / "skin.tsv"
    skin.write_text(
        "uniprot\tskin_score\tcell_type_preferred\n"
        "P1\t0.8\t \n",
        encoding="utf-8",
    )

    res = run_skin_weight(tmp_path, skin)

    assert res.returncode != 0
    assert (
        "Skin-expression score table column 'cell_type_preferred' contains blank values"
        in res.stderr
    )
    assert not (tmp_path / "ranked.csv").exists()


def test_cli_fails_when_top_csv_scores_are_not_numeric(tmp_path: Path) -> None:
    top = tmp_path / "top.csv"
    pd.DataFrame([{"target_id": "P1", "rrf_score": "not-a-number"}]).to_csv(
        top, index=False
    )
    skin = tmp_path / "skin.tsv"
    pd.DataFrame([{"uniprot": "P1", "skin_score": 0.8}]).to_csv(
        skin, sep="\t", index=False
    )

    res = run_skin_weight(tmp_path, skin, top_csv=top)

    assert res.returncode != 0
    assert "Top-target CSV column 'rrf_score' contains invalid values" in res.stderr
    assert not (tmp_path / "ranked.csv").exists()


def test_cli_fails_when_top_csv_score_is_boolean(tmp_path: Path) -> None:
    top = tmp_path / "top.csv"
    top.write_text("target_id,rrf_score\nP1,True\n")
    skin = tmp_path / "skin.tsv"
    pd.DataFrame([{"uniprot": "P1", "skin_score": 0.8}]).to_csv(
        skin, sep="\t", index=False
    )

    res = run_skin_weight(tmp_path, skin, top_csv=top)

    assert res.returncode != 0
    assert "Top-target CSV column 'rrf_score' contains invalid values" in res.stderr
    assert not (tmp_path / "ranked.csv").exists()


def test_cli_fails_when_top_csv_has_partially_invalid_scores(tmp_path: Path) -> None:
    top = tmp_path / "top.csv"
    top.write_text("target_id,rrf_score\nP1,1.0\nP2,not-a-number\n")
    skin = tmp_path / "skin.tsv"
    pd.DataFrame(
        [
            {"uniprot": "P1", "skin_score": 0.8},
            {"uniprot": "P2", "skin_score": 0.7},
        ]
    ).to_csv(skin, sep="\t", index=False)

    res = run_skin_weight(tmp_path, skin, top_csv=top)

    assert res.returncode != 0
    assert "Top-target CSV column 'rrf_score' contains invalid values" in res.stderr
    assert not (tmp_path / "ranked.csv").exists()


def test_cli_fails_when_top_csv_has_blank_target_ids(tmp_path: Path) -> None:
    top = tmp_path / "top.csv"
    top.write_text("target_id,rrf_score\nP1,1.0\n ,0.9\n")
    skin = tmp_path / "skin.tsv"
    pd.DataFrame([{"uniprot": "P1", "skin_score": 0.8}]).to_csv(
        skin, sep="\t", index=False
    )

    res = run_skin_weight(tmp_path, skin, top_csv=top)

    assert res.returncode != 0
    assert "Top-target CSV column 'target_id' contains blank values" in res.stderr
    assert not (tmp_path / "ranked.csv").exists()


def test_cli_fails_when_top_csv_has_duplicate_target_ids(tmp_path: Path) -> None:
    top = tmp_path / "top.csv"
    top.write_text("target_id,rrf_score\nP1,1.0\nP1,0.9\n")
    skin = tmp_path / "skin.tsv"
    pd.DataFrame([{"uniprot": "P1", "skin_score": 0.8}]).to_csv(
        skin, sep="\t", index=False
    )

    res = run_skin_weight(tmp_path, skin, top_csv=top)

    assert res.returncode != 0
    assert (
        "Top-target CSV column 'target_id' contains duplicate values: P1"
        in res.stderr
    )
    assert not (tmp_path / "ranked.csv").exists()


def test_cli_fails_when_skin_score_table_missing(tmp_path: Path) -> None:
    res = run_skin_weight(tmp_path, tmp_path / "missing.tsv")

    assert res.returncode != 0
    assert "Skin-expression score table is required" in res.stderr
    assert not (tmp_path / "ranked.csv").exists()


def test_cli_fails_when_skin_score_table_missing_required_columns(tmp_path: Path) -> None:
    skin = tmp_path / "skin.tsv"
    pd.DataFrame([{"uniprot": "P1"}]).to_csv(skin, sep="\t", index=False)

    res = run_skin_weight(tmp_path, skin)

    assert res.returncode != 0
    assert "missing required column" in res.stderr
    assert not (tmp_path / "ranked.csv").exists()


def test_cli_fails_when_skin_score_table_has_invalid_scores(tmp_path: Path) -> None:
    skin = tmp_path / "skin.tsv"
    skin.write_text("uniprot\tskin_score\nP1\t0.8\nP2\tnot-a-number\n")

    res = run_skin_weight(tmp_path, skin)

    assert res.returncode != 0
    assert (
        "Skin-expression score table column 'skin_score' contains invalid values"
        in res.stderr
    )
    assert not (tmp_path / "ranked.csv").exists()


def test_cli_fails_when_skin_score_table_has_boolean_score(tmp_path: Path) -> None:
    skin = tmp_path / "skin.tsv"
    skin.write_text("uniprot\tskin_score\nP1\tTrue\n")

    res = run_skin_weight(tmp_path, skin)

    assert res.returncode != 0
    assert (
        "Skin-expression score table column 'skin_score' contains invalid values"
        in res.stderr
    )
    assert not (tmp_path / "ranked.csv").exists()


def test_cli_fails_when_skin_score_table_has_out_of_range_scores(
    tmp_path: Path,
) -> None:
    skin = tmp_path / "skin.tsv"
    skin.write_text("uniprot\tskin_score\nP1\t1.2\n")

    res = run_skin_weight(tmp_path, skin)

    assert res.returncode != 0
    assert (
        "Skin-expression score table column 'skin_score' contains values outside [0, 1]"
        in res.stderr
    )
    assert not (tmp_path / "ranked.csv").exists()


def test_cli_fails_when_skin_score_table_has_blank_uniprots(tmp_path: Path) -> None:
    skin = tmp_path / "skin.tsv"
    skin.write_text("uniprot\tskin_score\nP1\t0.8\n \t0.7\n")

    res = run_skin_weight(tmp_path, skin)

    assert res.returncode != 0
    assert (
        "Skin-expression score table column 'uniprot' contains blank values"
        in res.stderr
    )
    assert not (tmp_path / "ranked.csv").exists()


def test_cli_fails_when_skin_score_table_has_duplicate_uniprots(tmp_path: Path) -> None:
    skin = tmp_path / "skin.tsv"
    skin.write_text("uniprot\tskin_score\nP1\t0.8\nP1\t0.7\n")

    res = run_skin_weight(tmp_path, skin)

    assert res.returncode != 0
    assert (
        "Skin-expression score table column 'uniprot' contains duplicate values: P1"
        in res.stderr
    )
    assert not (tmp_path / "ranked.csv").exists()


def test_cli_missing_top_target_skin_score_uses_zero_penalty(tmp_path: Path) -> None:
    top = tmp_path / "top.csv"
    top.write_text("target_id,rrf_score\nP1,1.0\nP2,0.9\n")
    skin = tmp_path / "skin.tsv"
    skin.write_text("uniprot\tskin_score\nP1\t0.8\n")

    res = run_skin_weight(tmp_path, skin, top_csv=top)

    assert res.returncode == 0, res.stderr
    assert (
        "Skin-expression score table is missing 1 top-target id(s); "
        "using skin_score=0.0 and skin_tier=very_low for: P2"
    ) in res.stderr
    ranked = pd.read_csv(tmp_path / "ranked.csv")
    p2 = ranked[ranked["target_id"] == "P2"].iloc[0]
    assert p2["skin_score"] == 0.0
    assert p2["skin_tier"] == "very_low"
    assert p2["cell_type_preferred"] == "unknown"


def test_cli_fails_when_configured_cosmetic_decision_missing(tmp_path: Path) -> None:
    skin = tmp_path / "skin.tsv"
    pd.DataFrame([{"uniprot": "P1", "skin_score": 0.8}]).to_csv(
        skin, sep="\t", index=False
    )

    res = run_skin_weight(
        tmp_path,
        skin,
        ["--cosmetic-decision", str(tmp_path / "missing_decision.txt")],
    )

    assert res.returncode != 0
    assert "Cosmetic/drug decision file is required" in res.stderr
    assert not (tmp_path / "ranked.csv").exists()


def test_cli_fails_when_configured_drug_warning_source_missing(
    tmp_path: Path,
) -> None:
    skin = tmp_path / "skin.tsv"
    pd.DataFrame([{"uniprot": "P1", "skin_score": 0.8}]).to_csv(
        skin, sep="\t", index=False
    )

    res = run_skin_weight(
        tmp_path,
        skin,
        ["--drug-json", str(tmp_path / "missing_drug_warnings.json")],
    )

    assert res.returncode != 0
    assert "Drug-avoidance warnings is required" in res.stderr
    assert not (tmp_path / "ranked.csv").exists()


def test_cli_fails_when_cosmetic_decision_disagrees_with_drug_source(
    tmp_path: Path,
) -> None:
    skin = tmp_path / "skin.tsv"
    pd.DataFrame([{"uniprot": "P1", "skin_score": 0.8}]).to_csv(
        skin, sep="\t", index=False
    )
    decision = tmp_path / "decision.txt"
    decision.write_text("PROCEED\n# policy: moderate\n")
    drug = tmp_path / "drug_warnings.json"
    drug.write_text(
        '{"warnings": [{"tier": "STRICT_WARNING"}], '
        '"n_warnings": 1, "reference_status": "ok"}\n'
    )

    res = run_skin_weight(
        tmp_path,
        skin,
        [
            "--cosmetic-decision",
            str(decision),
            "--drug-json",
            str(drug),
        ],
    )

    assert res.returncode != 0
    assert (
        "cosmetic/drug decision PROCEED does not match expected DOWNWEIGHT "
        "for policy moderate"
    ) in res.stderr
    assert not (tmp_path / "ranked.csv").exists()


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (
            ["--skin-weight", "-0.1"],
            "--skin-weight must be a finite value in [0, 1]: -0.1",
        ),
        (
            ["--skin-weight", "nan"],
            "--skin-weight must be a finite value in [0, 1]: nan",
        ),
        (
            ["--min-threshold", "1.1"],
            "--min-threshold must be a finite value in [0, 1]: 1.1",
        ),
        (
            ["--cosmetic-downweight-factor", "0"],
            "--cosmetic-downweight-factor must be a finite value in (0, 1]: 0",
        ),
        (
            ["--cosmetic-downweight-factor", "1.1"],
            "--cosmetic-downweight-factor must be a finite value in (0, 1]: 1.1",
        ),
        (
            ["--min-source-count", "0"],
            "--min-source-count must be >= 1: 0",
        ),
    ],
)
def test_cli_fails_when_weighting_parameters_are_invalid(
    tmp_path: Path,
    args: list[str],
    message: str,
) -> None:
    skin = tmp_path / "skin.tsv"
    pd.DataFrame([{"uniprot": "P1", "skin_score": 0.8}]).to_csv(
        skin, sep="\t", index=False
    )

    res = run_skin_weight(tmp_path, skin, args)

    assert res.returncode != 0
    assert message in res.stderr
    assert not (tmp_path / "ranked.csv").exists()


def test_cli_fails_when_required_source_count_is_missing(tmp_path: Path) -> None:
    top = tmp_path / "top.csv"
    pd.DataFrame([{"target_id": "P1", "rrf_score": 1.0}]).to_csv(top, index=False)
    skin = tmp_path / "skin.tsv"
    pd.DataFrame([{"uniprot": "P1", "skin_score": 0.8}]).to_csv(
        skin, sep="\t", index=False
    )

    res = run_skin_weight(
        tmp_path,
        skin,
        ["--min-source-count", "3"],
        top_csv=top,
    )

    assert res.returncode != 0
    assert "missing required column 'source_count'" in res.stderr
    assert not (tmp_path / "ranked.csv").exists()


def test_cli_fails_when_source_count_is_below_required_minimum(
    tmp_path: Path,
) -> None:
    top = tmp_path / "top.csv"
    pd.DataFrame([{
        "target_id": "P1",
        "rrf_score": 1.0,
        "source_count": 2,
        "sources": "autodock;gnina",
    }]).to_csv(top, index=False)
    skin = tmp_path / "skin.tsv"
    pd.DataFrame([{"uniprot": "P1", "skin_score": 0.8}]).to_csv(
        skin, sep="\t", index=False
    )

    res = run_skin_weight(
        tmp_path,
        skin,
        ["--min-source-count", "3"],
        top_csv=top,
    )

    assert res.returncode != 0
    assert "values below --min-source-count 3" in res.stderr
    assert not (tmp_path / "ranked.csv").exists()


def test_cli_fails_when_sources_are_blank(tmp_path: Path) -> None:
    top = tmp_path / "top.csv"
    top.write_text("target_id,rrf_score,source_count,sources\nP1,1.0,3, \n")
    skin = tmp_path / "skin.tsv"
    pd.DataFrame([{"uniprot": "P1", "skin_score": 0.8}]).to_csv(
        skin, sep="\t", index=False
    )

    res = run_skin_weight(
        tmp_path,
        skin,
        ["--min-source-count", "3"],
        top_csv=top,
    )

    assert res.returncode != 0
    assert "Top-target CSV column 'sources' contains blank values" in res.stderr
    assert not (tmp_path / "ranked.csv").exists()


def test_cli_fails_when_sources_have_duplicate_labels(tmp_path: Path) -> None:
    top = tmp_path / "top.csv"
    pd.DataFrame([{
        "target_id": "P1",
        "rrf_score": 1.0,
        "source_count": 3,
        "sources": "autodock;autodock;gnina",
    }]).to_csv(top, index=False)
    skin = tmp_path / "skin.tsv"
    pd.DataFrame([{"uniprot": "P1", "skin_score": 0.8}]).to_csv(
        skin, sep="\t", index=False
    )

    res = run_skin_weight(
        tmp_path,
        skin,
        ["--min-source-count", "3"],
        top_csv=top,
    )

    assert res.returncode != 0
    assert (
        "Top-target CSV column 'sources' contains duplicate labels at row "
        "index 0: autodock"
    ) in res.stderr
    assert not (tmp_path / "ranked.csv").exists()


def test_cli_fails_when_source_count_does_not_match_sources(
    tmp_path: Path,
) -> None:
    top = tmp_path / "top.csv"
    pd.DataFrame([{
        "target_id": "P1",
        "rrf_score": 1.0,
        "source_count": 4,
        "sources": "autodock",
    }]).to_csv(top, index=False)
    skin = tmp_path / "skin.tsv"
    pd.DataFrame([{"uniprot": "P1", "skin_score": 0.8}]).to_csv(
        skin, sep="\t", index=False
    )

    res = run_skin_weight(
        tmp_path,
        skin,
        ["--min-source-count", "3"],
        top_csv=top,
    )

    assert res.returncode != 0
    assert "Top-target CSV source_count=4 but sources lists 1 label(s)" in res.stderr
    assert not (tmp_path / "ranked.csv").exists()


def test_cli_writes_ranked_targets_with_valid_skin_scores(tmp_path: Path) -> None:
    skin = tmp_path / "skin.tsv"
    pd.DataFrame([{"uniprot": "P1", "skin_score": 0.8, "tier": "high"}]).to_csv(
        skin, sep="\t", index=False
    )

    res = run_skin_weight(tmp_path, skin)

    assert res.returncode == 0, res.stderr
    out = pd.read_csv(tmp_path / "ranked.csv")
    assert out["target_id"].tolist() == ["P1"]
    assert out["skin_score"].tolist() == [0.8]


def test_cli_preserves_source_rationale_columns(tmp_path: Path) -> None:
    top = tmp_path / "top.csv"
    pd.DataFrame([{
        "target_id": "P1",
        "rrf_score": 1.0,
        "source_count": 4,
        "sources": "autodock;gnina;rtmscore;boltz",
    }]).to_csv(top, index=False)
    skin = tmp_path / "skin.tsv"
    pd.DataFrame([{"uniprot": "P1", "skin_score": 0.8, "tier": "high"}]).to_csv(
        skin, sep="\t", index=False
    )

    res = run_skin_weight(
        tmp_path,
        skin,
        ["--min-source-count", "3"],
        top_csv=top,
    )

    assert res.returncode == 0, res.stderr
    out = pd.read_csv(tmp_path / "ranked.csv")
    assert out["source_count"].tolist() == [4]
    assert out["sources"].tolist() == ["autodock;gnina;rtmscore;boltz"]


def test_prior_power_clamps_out_of_range_scores_instead_of_going_complex() -> None:
    """A negative base with a fractional power yields a complex number.

    apply_skin_weight is a documented pure helper, so it must clamp rather than
    fail with an unrelated TypeError from the comparison that follows.
    """
    finals = apply_skin_weight(
        docking_scores={"A": 1.0, "B": 0.5},
        skin_scores={"A": 0.9, "B": 0.9},
        skin_weight=0.30,
        known_target_prior_scores={"A": -0.5, "B": 1.5},
        known_target_prior_weight=0.20,
        known_target_prior_power=0.5,
    )

    assert set(finals) == {"A", "B"}
    assert all(isinstance(value, float) for value in finals.values())
    assert finals["A"] > finals["B"]


def test_prior_power_rejects_non_finite_prior_scores() -> None:
    with pytest.raises(SystemExit):
        apply_skin_weight(
            docking_scores={"A": 1.0},
            skin_scores={"A": 0.9},
            known_target_prior_scores={"A": float("nan")},
            known_target_prior_weight=0.20,
            known_target_prior_power=0.5,
        )
