"""Unit tests for reciprocal rank fusion (INSTRUCTIONS.md §6.2 (c))."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage3_rrf import (  # noqa: E402
    _rank_from_table,
    reciprocal_rank_fusion,
    source_contributions,
    source_support,
)

ROOT = Path(__file__).resolve().parents[2]


def test_rrf_top_of_list_beats_bottom() -> None:
    rrf = reciprocal_rank_fusion(
        {"src": ["A", "B", "C"]}, k=60,
    )
    assert rrf["A"] > rrf["B"] > rrf["C"]


def test_rrf_consensus_beats_single_source() -> None:
    rrf = reciprocal_rank_fusion(
        {
            "x": ["A", "B", "C"],
            "y": ["A", "C", "B"],
            "z": ["B", "A", "C"],
        },
        k=60,
    )
    # A is rank 1, 1, 2 → best summed score
    assert rrf["A"] > rrf["B"]
    assert rrf["A"] > rrf["C"]


def test_rrf_value_matches_formula() -> None:
    rrf = reciprocal_rank_fusion({"s": ["A", "B"]}, k=60)
    assert abs(rrf["A"] - 1 / (60 + 1)) < 1e-12
    assert abs(rrf["B"] - 1 / (60 + 2)) < 1e-12


def test_rrf_missing_target_in_one_list() -> None:
    rrf = reciprocal_rank_fusion(
        {"x": ["A", "B"], "y": ["B"]}, k=60,
    )
    # A appears only in x at rank 1
    # B appears in x rank 2 and y rank 1
    expected_a = 1 / 61
    expected_b = 1 / 62 + 1 / 61
    assert abs(rrf["A"] - expected_a) < 1e-12
    assert abs(rrf["B"] - expected_b) < 1e-12
    assert rrf["B"] > rrf["A"]


def test_rrf_with_four_score_sources() -> None:
    """The pipeline's 4-way RRF (AutoDock, GNINA, RTM, Boltz-2)."""
    rrf = reciprocal_rank_fusion(
        {
            "autodock": ["T1", "T2", "T3", "T4"],
            "gnina":    ["T2", "T1", "T4", "T3"],
            "rtm":      ["T1", "T3", "T2", "T4"],
            "boltz":    ["T1", "T2", "T3", "T4"],
        },
        k=60,
    )
    ranked = sorted(rrf.items(), key=lambda kv: -kv[1])
    assert ranked[0][0] == "T1"   # rank 1,2,1,1 dominates


def test_source_support_counts_independent_scorers() -> None:
    support = source_support(
        {
            "autodock": ["T1", "T2"],
            "gnina": ["T2", "T3"],
            "boltz": [],
        }
    )
    assert support["T1"] == {"autodock"}
    assert support["T2"] == {"autodock", "gnina"}
    assert support["T3"] == {"gnina"}


def test_rrf_counts_duplicate_targets_once_per_source() -> None:
    rrf = reciprocal_rank_fusion({"src": ["T1", "T1", "T2"]}, k=60)

    assert abs(rrf["T1"] - 1 / 61) < 1e-12
    assert abs(rrf["T2"] - 1 / 62) < 1e-12


def test_weighted_rrf_value_matches_formula() -> None:
    rrf = reciprocal_rank_fusion(
        {"dock": ["A", "B"], "ml": ["B", "A"]},
        k=60,
        source_weights={"dock": 2.0, "ml": 0.5},
    )

    assert abs(rrf["A"] - (2.0 / 61 + 0.5 / 62)) < 1e-12
    assert abs(rrf["B"] - (2.0 / 62 + 0.5 / 61)) < 1e-12


def test_equal_rrf_default_matches_explicit_unit_weights() -> None:
    ranked = {"dock": ["A", "B"], "ml": ["B", "A"]}

    default_rrf = reciprocal_rank_fusion(ranked, k=60)
    weighted_rrf = reciprocal_rank_fusion(
        ranked,
        k=60,
        source_weights={"dock": 1.0, "ml": 1.0},
    )

    assert weighted_rrf == default_rrf


def test_source_contributions_sum_to_weighted_rrf_score() -> None:
    ranked = {"dock": ["A", "B"], "ml": ["B", "A"]}
    weights = {"dock": 3.0, "ml": 0.25}

    rrf = reciprocal_rank_fusion(ranked, k=60, source_weights=weights)
    contributions = source_contributions(ranked, k=60, source_weights=weights)

    for target, score in rrf.items():
        assert abs(sum(contributions[target].values()) - score) < 1e-12


def test_weighted_rrf_rejects_mismatched_source_labels() -> None:
    with pytest.raises(ValueError, match="missing labels.*ml"):
        reciprocal_rank_fusion(
            {"dock": ["A"], "ml": ["A"]},
            source_weights={"dock": 1.0},
        )

    with pytest.raises(ValueError, match="extra labels.*extra"):
        reciprocal_rank_fusion(
            {"dock": ["A"]},
            source_weights={"dock": 1.0, "extra": 1.0},
        )


def test_weighted_rrf_rejects_bool_and_nonpositive_weights() -> None:
    with pytest.raises(ValueError, match="must not be boolean"):
        reciprocal_rank_fusion({"dock": ["A"]}, source_weights={"dock": True})

    with pytest.raises(ValueError, match="positive finite"):
        reciprocal_rank_fusion({"dock": ["A"]}, source_weights={"dock": 0.0})


def test_source_support_counts_duplicate_targets_once_per_source() -> None:
    support = source_support({"src": ["T1", "T1"], "other": ["T1"]})

    assert support["T1"] == {"src", "other"}


def test_rank_from_table_rejects_duplicate_targets(tmp_path: Path) -> None:
    scores = tmp_path / "scores.tsv"
    pd.DataFrame(
        [
            {"target_id": "T1", "score": 10.0},
            {"target_id": "T2", "score": 9.0},
            {"target_id": "T1", "score": 1.0},
        ]
    ).to_csv(scores, sep="\t", index=False)

    with pytest.raises(SystemExit, match="duplicate target_id values: T1"):
        _rank_from_table(scores)


def test_rank_from_table_fails_on_missing_required_columns(tmp_path: Path) -> None:
    scores = tmp_path / "scores.tsv"
    pd.DataFrame([{"target_id": "T1", "confidence": 0.5}]).to_csv(
        scores, sep="\t", index=False
    )

    with pytest.raises(SystemExit, match="missing required RRF column"):
        _rank_from_table(scores)


def test_rank_from_table_fails_on_partially_invalid_scores(tmp_path: Path) -> None:
    scores = tmp_path / "scores.tsv"
    scores.write_text("target_id\tscore\nT1\t10.0\nT2\tnot-a-number\n")

    with pytest.raises(SystemExit, match="RRF column 'score' contains invalid values"):
        _rank_from_table(scores)


def test_rank_from_table_fails_on_boolean_scores(tmp_path: Path) -> None:
    scores = tmp_path / "scores.tsv"
    scores.write_text("target_id\tscore\nT1\tTrue\n")

    with pytest.raises(SystemExit, match="RRF column 'score' contains invalid values"):
        _rank_from_table(scores)


def test_rank_from_table_fails_on_blank_target_ids(tmp_path: Path) -> None:
    scores = tmp_path / "scores.tsv"
    scores.write_text("target_id\tscore\nT1\t10.0\n \t9.0\n")

    with pytest.raises(SystemExit, match="RRF column 'target_id' contains blank values"):
        _rank_from_table(scores)


def test_rank_from_table_sorts_numeric_scores_not_lexicographic(tmp_path: Path) -> None:
    scores = tmp_path / "scores.tsv"
    scores.write_text("target_id\tscore\nT1\t9\nT2\t10\n")

    assert _rank_from_table(scores) == ["T2", "T1"]


def test_rrf_cli_removes_stale_output_on_invalid_input(tmp_path: Path) -> None:
    scores = tmp_path / "scores.tsv"
    out = tmp_path / "rrf.csv"
    pd.DataFrame([{"target_id": "T1", "confidence": 0.5}]).to_csv(
        scores, sep="\t", index=False
    )
    out.write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage3_rrf.py"),
            "--inputs",
            f"{scores}=bad",
            "--out-csv",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "missing required RRF column" in res.stderr
    assert not out.exists()


def test_rrf_cli_rejects_duplicate_target_ids_without_output(tmp_path: Path) -> None:
    scores = tmp_path / "scores.tsv"
    out = tmp_path / "rrf.csv"
    pd.DataFrame(
        [
            {"target_id": "T1", "score": 10.0},
            {"target_id": "T1", "score": 9.0},
        ]
    ).to_csv(scores, sep="\t", index=False)
    out.write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage3_rrf.py"),
            "--inputs",
            f"{scores}=src",
            "--out-csv",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "contains duplicate target_id values: T1" in res.stderr
    assert not out.exists()


def test_rrf_cli_rejects_duplicate_input_labels_without_output(tmp_path: Path) -> None:
    scores_a = tmp_path / "scores_a.tsv"
    scores_b = tmp_path / "scores_b.tsv"
    out = tmp_path / "rrf.csv"
    pd.DataFrame([{"target_id": "T1", "score": 10.0}]).to_csv(
        scores_a, sep="\t", index=False
    )
    pd.DataFrame([{"target_id": "T2", "score": 9.0}]).to_csv(
        scores_b, sep="\t", index=False
    )
    out.write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage3_rrf.py"),
            "--inputs",
            f"{scores_a}=dock,{scores_b}=dock",
            "--out-csv",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "duplicate RRF input label" in res.stderr
    assert "dock" in res.stderr
    assert not out.exists()


def test_rrf_cli_writes_weighted_audit_columns_and_recipe_id(tmp_path: Path) -> None:
    scores_a = tmp_path / "scores_a.tsv"
    scores_b = tmp_path / "scores_b.tsv"
    out = tmp_path / "rrf.csv"
    pd.DataFrame(
        [
            {"target_id": "T1", "score": 10.0},
            {"target_id": "T2", "score": 9.0},
        ]
    ).to_csv(scores_a, sep="\t", index=False)
    pd.DataFrame(
        [
            {"target_id": "T2", "score": 10.0},
            {"target_id": "T1", "score": 9.0},
        ]
    ).to_csv(scores_b, sep="\t", index=False)

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage3_rrf.py"),
            "--inputs",
            f"{scores_a}=dock,{scores_b}=ml",
            "--source-weights",
            "dock=2.0,ml=0.5",
            "--recipe-id",
            "weighted-v1",
            "--out-csv",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode == 0, res.stderr
    df = pd.read_csv(out)
    row_t1 = df[df["target_id"] == "T1"].iloc[0]
    assert row_t1["recipe_id"] == "weighted-v1"
    assert row_t1["dock_rank"] == 1
    assert row_t1["ml_rank"] == 2
    assert abs(row_t1["dock_rrf_contribution"] - 2.0 / 61) < 1e-12
    assert abs(row_t1["ml_rrf_contribution"] - 0.5 / 62) < 1e-12
    assert (
        abs(
            row_t1["rrf_score"]
            - row_t1["dock_rrf_contribution"]
            - row_t1["ml_rrf_contribution"]
        )
        < 1e-12
    )


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ("dock=1.0,dock=2.0", "duplicate label: dock"),
        (
            "dock=1.0",
            "labels must exactly match --inputs labels; missing labels: ['ml']",
        ),
        (
            "dock=1.0,ml=1.0,extra=1.0",
            "labels must exactly match --inputs labels; extra labels: ['extra']",
        ),
        ("dock=true,ml=1.0", "must be a positive finite weight"),
        ("dock=nan,ml=1.0", "must be a positive finite weight"),
        ("dock=0,ml=1.0", "must be a positive finite weight"),
        ("dock=-1,ml=1.0", "must be a positive finite weight"),
    ],
)
def test_rrf_cli_rejects_invalid_source_weights_without_output(
    tmp_path: Path,
    weights: str,
    message: str,
) -> None:
    scores_a = tmp_path / "scores_a.tsv"
    scores_b = tmp_path / "scores_b.tsv"
    out = tmp_path / "rrf.csv"
    pd.DataFrame([{"target_id": "T1", "score": 10.0}]).to_csv(
        scores_a, sep="\t", index=False
    )
    pd.DataFrame([{"target_id": "T1", "score": 9.0}]).to_csv(
        scores_b, sep="\t", index=False
    )
    out.write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage3_rrf.py"),
            "--inputs",
            f"{scores_a}=dock,{scores_b}=ml",
            "--source-weights",
            weights,
            "--out-csv",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert message in res.stderr
    assert not out.exists()


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--rrf-k", "0"], "--rrf-k must be >= 1: 0"),
        (
            ["--min-sources-per-target", "0"],
            "--min-sources-per-target must be >= 1: 0",
        ),
        (["--top-n", "0"], "--top-n must be >= 1: 0"),
        (["--top-pct", "0"], "--top-pct must be a finite value in (0, 1]: 0"),
        (["--top-pct", "nan"], "--top-pct must be a finite value in (0, 1]: nan"),
        (["--top-pct", "1.1"], "--top-pct must be a finite value in (0, 1]: 1.1"),
    ],
)
def test_rrf_cli_rejects_invalid_numeric_args_without_output(
    tmp_path: Path,
    args: list[str],
    message: str,
) -> None:
    scores = tmp_path / "scores.tsv"
    out = tmp_path / "rrf.csv"
    pd.DataFrame([{"target_id": "T1", "score": 1.0}]).to_csv(
        scores, sep="\t", index=False
    )
    out.write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage3_rrf.py"),
            "--inputs",
            f"{scores}=src",
            "--out-csv",
            str(out),
            *args,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert message in res.stderr
    assert not out.exists()


def test_equal_scores_rank_the_same_way_whatever_order_they_arrive_in(tmp_path):
    """Ties are the normal case here, not an edge case.

    `stage3_pick_top` normalises equal raw scores to equal normalised scores on
    purpose, so whole blocks of targets reach RRF tied. Sorting those with an
    unstable quicksort and no tiebreak made the fused rank depend on the input
    file's row order: the same 25 tied targets came back in different orders,
    spreading their RRF contribution by ~39%.
    """
    import sys

    import pandas as pd

    sys.path.insert(0, str(ROOT / "scripts"))
    from stage3_rrf import _rank_from_table

    identifiers = [f"T{index:02d}" for index in range(25)]
    path = tmp_path / "tied.csv"

    pd.DataFrame({"target_id": identifiers, "score": [0.5] * 25}).to_csv(path, index=False)
    forward = _rank_from_table(path)
    pd.DataFrame({"target_id": identifiers[::-1], "score": [0.5] * 25}).to_csv(path, index=False)
    reversed_input = _rank_from_table(path)

    assert forward == reversed_input
    assert forward == sorted(identifiers), "ties must fall back to target_id"


def test_a_real_score_still_outranks_a_tie(tmp_path):
    """The tiebreak must not become the sort."""
    import sys

    import pandas as pd

    sys.path.insert(0, str(ROOT / "scripts"))
    from stage3_rrf import _rank_from_table

    path = tmp_path / "mixed.csv"
    pd.DataFrame(
        {"target_id": ["Z_best", "A_tied", "B_tied"], "score": [0.9, 0.5, 0.5]}
    ).to_csv(path, index=False)

    assert _rank_from_table(path) == ["Z_best", "A_tied", "B_tied"]
