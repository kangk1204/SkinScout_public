"""Band re-ranking: reorder only where docking was measured to help.

The measurement behind this is docs/RERANK_EXPERIMENT_20260828.md - re-ranking
the whole list is worse than not re-ranking at all, while re-ranking ranks
11-50 moves top-30 recovery from 14/28 to 20/28.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "stage3_band_rerank.py"


def _module():
    spec = importlib.util.spec_from_file_location("band_rerank_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fast_reader_facing_v3_path_consumes_band_rerank() -> None:
    rules = (ROOT / "workflow/rules/stage3_v3_skin_weight.smk").read_text()
    fast_branch = rules.split('if MODE == "fast":', 1)[1].split("return", 2)[1]
    assert "top50_band_reranked.csv" in fast_branch


def _frame(count: int = 60) -> pd.DataFrame:
    """Docking deliberately disagrees with similarity, so movement is visible."""
    return pd.DataFrame({
        "target_id": [f"P{index:05d}" for index in range(1, count + 1)],
        "daina_rank": range(1, count + 1),
        "ranking_basis": ["daina_max_tanimoto"] * count,
        # Deliberately opposed to similarity: the worst similarity rank carries
        # the best docking score, so any reordering is unmistakable.
        "autodock_energy_kcal_mol": [-3.0 - 0.05 * index for index in range(count)],
        "gnina_cnn_affinity": [1.0 + 0.05 * index for index in range(count)],
    })


def test_the_head_is_never_reordered() -> None:
    """Measured: re-ranking a top-10 target went the wrong way in 8 of 8 pairs."""
    ordered = _module().band_rerank(_frame(), keep=10, band=50)

    head = ordered.head(10)
    assert head["daina_rank"].tolist() == list(range(1, 11))
    assert not head["rerank_band"].any()


def test_the_tail_keeps_its_similarity_order() -> None:
    ordered = _module().band_rerank(_frame(), keep=10, band=50)

    tail = ordered[ordered["daina_rank"] > 50]
    assert tail["daina_rank"].tolist() == sorted(tail["daina_rank"].tolist())
    assert not tail["rerank_band"].any()


def test_only_the_band_moves() -> None:
    ordered = _module().band_rerank(_frame(), keep=10, band=50)

    band = ordered[ordered["rerank_band"]]
    assert len(band) == 40
    assert set(band["daina_rank"]) == set(range(11, 51))
    # Docking was set to disagree with similarity, so the band must be reversed.
    assert band["daina_rank"].tolist() == sorted(band["daina_rank"].tolist(), reverse=True)
    # And it must still occupy exactly positions 11-50.
    assert band["final_rank"].tolist() == list(range(11, 51))


def test_every_row_keeps_the_position_similarity_gave_it() -> None:
    """Without this a reader cannot tell what the re-ranking moved."""
    ordered = _module().band_rerank(_frame(), keep=10, band=50)

    assert set(ordered["daina_rank"]) == set(range(1, 61))
    moved = ordered[ordered["final_rank"] != ordered["daina_rank"]]
    assert len(moved) > 0
    assert moved["rerank_band"].all()


def test_the_basis_says_what_the_order_came_from() -> None:
    module = _module()

    reordered = module.band_rerank(_frame(), keep=10, band=50)
    assert set(reordered["ranking_basis"]) == {"daina_head_then_band_rrf"}

    # Disabled: the order is similarity's, and the basis must not claim otherwise.
    untouched = module.band_rerank(_frame(), keep=50, band=50)
    assert set(untouched["ranking_basis"]) == {"daina_max_tanimoto"}
    assert untouched["daina_rank"].tolist() == list(range(1, 61))


def test_a_run_with_no_docking_scores_is_left_alone() -> None:
    frame = _frame().drop(columns=["autodock_energy_kcal_mol", "gnina_cnn_affinity"])

    ordered = _module().band_rerank(frame, keep=10, band=50)

    # Daina is the only opinion, so the fusion cannot change its order.
    assert ordered["daina_rank"].tolist() == list(range(1, 61))


def test_a_scorer_that_reached_only_some_targets_still_contributes() -> None:
    frame = _frame()
    frame.loc[frame.daina_rank.between(11, 30), "gnina_cnn_affinity"] = None

    ordered = _module().band_rerank(frame, keep=10, band=50)

    assert "gnina_cnn_affinity" in set(ordered["band_rrf_sources"].iloc[10].split(";"))
    assert ordered["final_rank"].tolist() == list(range(1, 61))


def test_a_missing_required_column_is_refused() -> None:
    with pytest.raises(SystemExit, match="daina_rank"):
        _module().band_rerank(_frame().drop(columns=["daina_rank"]))


def test_fractional_ranks_are_refused_before_any_int_cast() -> None:
    """1.1/2.9 used to survive `astype(int)` and pass as 1/2."""
    frame = _frame(count=5)
    frame["daina_rank"] = frame["daina_rank"].astype(float)
    frame.loc[frame.index[1], "daina_rank"] = 2.9

    with pytest.raises(SystemExit, match="정수"):
        _module().band_rerank(frame, keep=2, band=4)


def test_non_finite_ranks_are_refused() -> None:
    module = _module()
    for bad in (float("inf"), float("-inf"), float("nan")):
        frame = _frame(count=5)
        frame["daina_rank"] = frame["daina_rank"].astype(float)
        frame.loc[frame.index[0], "daina_rank"] = bad
        with pytest.raises(SystemExit):
            module.band_rerank(frame, keep=2, band=4)


def test_blank_and_whitespace_ids_are_refused_after_trim() -> None:
    module = _module()
    for bad in ("", "   ", "\t"):
        frame = _frame(count=5)
        frame.loc[frame.index[0], "target_id"] = bad
        with pytest.raises(SystemExit, match="target_id"):
            module.band_rerank(frame, keep=2, band=4)


def test_ids_that_collide_after_trim_are_refused_and_valid_ids_are_normalized() -> None:
    module = _module()
    collision = _frame(count=5)
    collision.loc[collision.index[1], "target_id"] = f"  {collision.loc[0, 'target_id']}  "
    with pytest.raises(SystemExit, match="중복"):
        module.band_rerank(collision, keep=2, band=4)

    padded = _frame(count=5)
    padded.loc[padded.index[0], "target_id"] = f"  {padded.loc[0, 'target_id']}  "
    ordered = module.band_rerank(padded, keep=2, band=4)
    assert ordered.loc[0, "target_id"] == "P00001"


def test_valid_input_is_unchanged_by_the_stricter_rank_contract() -> None:
    frame = _frame(count=12)

    ordered = _module().band_rerank(frame, keep=2, band=6)

    assert sorted(ordered["daina_rank"].tolist()) == list(range(1, 13))
    assert sorted(ordered["target_id"].tolist()) == sorted(frame["target_id"].tolist())
    assert ordered["final_rank"].tolist() == list(range(1, 13))

    untouched = _module().band_rerank(frame, keep=12, band=12)
    assert untouched["target_id"].tolist() == frame["target_id"].tolist()
    assert untouched["daina_rank"].tolist() == frame["daina_rank"].tolist()


def test_the_cli_writes_both_outputs(tmp_path: Path) -> None:
    source = tmp_path / "in.csv"
    _frame().to_csv(source, index=False)
    out = tmp_path / "out.csv"
    top50 = tmp_path / "top50.csv"

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--in-csv", str(source),
         "--out-csv", str(out), "--out-top50", str(top50)],
        cwd=ROOT, capture_output=True, text=True,
    )

    assert result.returncode == 0, result.stderr
    assert len(pd.read_csv(out)) == 60
    assert len(pd.read_csv(top50)) == 50
    assert "재정렬 대상" in result.stdout


def test_the_defaults_match_the_measured_band() -> None:
    module = _module()

    assert module.DEFAULT_KEEP == 10
    assert module.DEFAULT_BAND == 50
    assert module.RRF_K == 60


# --- The shipped stage against the panel it was measured on ---

SCORES = ROOT / "data" / "validation" / "rerank_band_20260828"


def _panel_ranks(keep: int, band: int) -> pd.DataFrame:
    """Run the shipped stage over the committed panel score tables."""
    truth = pd.read_csv(SCORES / "known_targets.csv")
    known: dict[str, list[str]] = {}
    for record in truth.to_dict("records"):
        known.setdefault(record["case_id"], []).append(record["target_id"])

    module = _module()
    rows = []
    for directory in sorted(SCORES.iterdir()):
        if not directory.is_dir() or directory.name not in known:
            continue
        frame = (
            pd.read_csv(directory / "selected.csv")
            .merge(
                pd.read_csv(directory / "autodock.tsv", sep="\t")[["target_id", "vina_score"]],
                on="target_id", how="left",
            )
            .merge(
                pd.read_csv(directory / "gnina.tsv", sep="\t")[["target_id", "cnn_affinity"]],
                on="target_id", how="left",
            )
            .rename(columns={
                "vina_score": "autodock_energy_kcal_mol",
                "cnn_affinity": "gnina_cnn_affinity",
            })
        )
        frame["ranking_basis"] = "daina_max_tanimoto"
        ordered = module.band_rerank(frame, keep=keep, band=band).set_index("target_id")
        for target_id in known[directory.name]:
            if target_id in ordered.index:
                rows.append({
                    "daina": float(ordered.loc[target_id, "daina_rank"]),
                    "band": float(ordered.loc[target_id, "final_rank"]),
                })
    return pd.DataFrame(rows)


@pytest.mark.skipif(not SCORES.is_dir(), reason="panel score tables absent")
def test_the_stage_reproduces_the_measurement_it_cites() -> None:
    """The docstring's numbers have to be what this code actually produces.

    It scores the band against itself, which is what a run that only docks the
    band actually has - the cheaper arm of the experiment, and the one shipped.
    """
    pytest.importorskip("scipy")
    from scipy.stats import wilcoxon

    ranks = _panel_ranks(keep=10, band=50)

    assert len(ranks) == 28
    assert ranks.daina.mean() == pytest.approx(43.2, abs=0.1)
    assert int((ranks.daina <= 30).sum()) == 14
    assert ranks.band.mean() == pytest.approx(38.5, abs=0.1)
    assert int((ranks.band <= 30).sum()) == 20
    assert wilcoxon(ranks.daina, ranks.band)[1] < 0.01


@pytest.mark.skipif(not SCORES.is_dir(), reason="panel score tables absent")
def test_reranking_the_whole_list_would_have_been_worse() -> None:
    """Why the head is held: the alternative was measured and is worse."""
    everything = _panel_ranks(keep=0, band=10**6)
    banded = _panel_ranks(keep=10, band=50)

    assert everything.band.mean() > everything.daina.mean()
    assert banded.band.mean() < everything.band.mean()


# --- Only the band is docked, and the rest says so ---


def test_a_target_outside_the_band_is_not_called_a_failure() -> None:
    """Grids outside the re-ranking band are never read, so they are not built.

    The overlay demands exact parity between the selection and the manifests,
    so the skipped targets keep a row - with a status of their own. Reusing
    "structural_unavailable_prep" would say preparation failed for a target
    nobody asked about.
    """
    import json

    schema = json.loads(
        (ROOT / "schemas" / "target_fast_v2.json").read_text(encoding="utf-8")
    )
    assert "structure_not_requested" in schema["properties"]["structural_status"]["enum"]

    overlay = (ROOT / "scripts" / "stage3_daina_structural_overlay.py").read_text(
        encoding="utf-8"
    )
    final = overlay.split("FINAL_STRUCTURAL_STATUSES = {", 1)[1].split("}", 1)[0]
    assert "structure_not_requested" in final


def test_the_band_bounds_default_to_no_restriction() -> None:
    """Unrestricted has to be the same code path, not a separate one."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "autogrid_under_test", ROOT / "scripts" / "stage3_autogrid_maps.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    assert module._rank_in_band(1, 0, 0) is True
    assert module._rank_in_band(9999, 0, 0) is True
    # keep=10, band=50 -> ranks 11..50
    assert module._rank_in_band(10, 11, 50) is False
    assert module._rank_in_band(11, 11, 50) is True
    assert module._rank_in_band(50, 11, 50) is True
    assert module._rank_in_band(51, 11, 50) is False
    # A one-sided bound restricts only that side.
    assert module._rank_in_band(5, 0, 50) is True
    assert module._rank_in_band(80, 0, 50) is False


def test_the_workflow_derives_the_docking_band_from_the_rerank_band() -> None:
    """Docking a range the re-ranking does not read would be waste; docking
    less than it reads would silently weaken the ranking."""
    rules = (ROOT / "workflow" / "rules" / "stage3b_fast.smk").read_text(encoding="utf-8")

    assert 'DOCKING["fast_mode_rerank_keep"] + 1' in rules
    assert 'DOCKING["fast_mode_rerank_band"]' in rules
    assert "--dock-rank-from" in rules and "--dock-rank-to" in rules
