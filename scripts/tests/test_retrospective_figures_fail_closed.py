"""Regression tests for retrospective figure input gates."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ORDER = [
    "retinol",
    "niacinamide",
    "ascorbic_acid",
    "aspirin",
    "resveratrol",
    "egcg",
    "caffeine",
    "kojic_acid",
    "alpha_arbutin",
]


def run_script(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def write_rankings(root: Path, *, missing_skin_weighted: str | None = None) -> None:
    for name in ORDER:
        target_dir = root / f"results/runs/{name}_demo/03_targets"
        target_dir.mkdir(parents=True)
        pd.DataFrame([
            {"target_id": "P1", "rank": 1, "vina_kcal_mol": -7.0},
            {"target_id": "P2", "rank": 2, "vina_kcal_mol": -6.0},
        ]).to_csv(target_dir / "demo_ranked_targets.csv", index=False)
        if name != missing_skin_weighted:
            pd.DataFrame([
                {"target_id": "P1", "rank_skin": 1},
                {"target_id": "P2", "rank_skin": 2},
            ]).to_csv(target_dir / "demo_skin_weighted.csv", index=False)


def write_skin_score(root: Path) -> None:
    out_dir = root / "data/skin_expression"
    out_dir.mkdir(parents=True)
    pd.DataFrame([
        {"uniprot": "P02533", "skin_score": 0.9},
        {"uniprot": "P14679", "skin_score": 0.8},
        {"uniprot": "P20930", "skin_score": 0.7},
        {"uniprot": "P03956", "skin_score": 0.6},
        {"uniprot": "P40261", "skin_score": 0.5},
        {"uniprot": "P00000", "skin_score": 0.1},
    ]).to_csv(out_dir / "skin_score.tsv", sep="\t", index=False)


def test_retrospective_figures_require_skin_score_by_default(tmp_path: Path) -> None:
    write_rankings(tmp_path)
    out_dir = tmp_path / "figures"

    res = run_script([
        "scripts/make_retrospective_figures.py",
        "--root", str(tmp_path),
        "--out-dir", str(out_dir),
    ])

    assert res.returncode != 0
    assert "SkinScore TSV is required for fig3" in res.stderr
    assert not out_dir.exists()


def test_retrospective_figures_require_skin_weighted_rankings_by_default(tmp_path: Path) -> None:
    write_rankings(tmp_path, missing_skin_weighted="retinol")
    write_skin_score(tmp_path)
    out_dir = tmp_path / "figures"

    res = run_script([
        "scripts/make_retrospective_figures.py",
        "--root", str(tmp_path),
        "--out-dir", str(out_dir),
    ])

    assert res.returncode != 0
    assert "Skin-weighted ranking is required for fig1" in res.stderr
    assert not out_dir.exists()


def test_retrospective_figures_reject_blank_vina_energy_without_partial_outputs(
    tmp_path: Path,
) -> None:
    write_rankings(tmp_path)
    write_skin_score(tmp_path)
    retinol = tmp_path / "results/runs/retinol_demo/03_targets/demo_ranked_targets.csv"
    pd.DataFrame([
        {"target_id": "P1", "rank": 1, "vina_kcal_mol": ""},
        {"target_id": "P2", "rank": 2, "vina_kcal_mol": -6.0},
    ]).to_csv(retinol, index=False)
    out_dir = tmp_path / "figures"

    res = run_script([
        "scripts/make_retrospective_figures.py",
        "--root", str(tmp_path),
        "--out-dir", str(out_dir),
    ])

    assert res.returncode != 0
    assert "Vina ranking for retinol column 'vina_kcal_mol' contains blank values" in res.stderr
    assert not out_dir.exists()


def test_retrospective_figures_require_explicit_skin_rank(tmp_path: Path) -> None:
    write_rankings(tmp_path)
    write_skin_score(tmp_path)
    retinol = tmp_path / "results/runs/retinol_demo/03_targets/demo_skin_weighted.csv"
    pd.DataFrame([
        {"target_id": "P1"},
        {"target_id": "P2"},
    ]).to_csv(retinol, index=False)
    out_dir = tmp_path / "figures"

    res = run_script([
        "scripts/make_retrospective_figures.py",
        "--root", str(tmp_path),
        "--out-dir", str(out_dir),
    ])

    assert res.returncode != 0
    assert "Skin-weighted ranking for retinol is missing required columns ['rank_skin']" in res.stderr
    assert not out_dir.exists()


def test_retrospective_figures_write_all_figures_for_valid_tsv_inputs(
    tmp_path: Path,
) -> None:
    write_rankings(tmp_path)
    write_skin_score(tmp_path)
    out_dir = tmp_path / "figures"

    res = run_script([
        "scripts/make_retrospective_figures.py",
        "--root", str(tmp_path),
        "--out-dir", str(out_dir),
    ])

    assert res.returncode == 0, res.stderr
    assert (out_dir / "fig1_recovery_slope.png").exists()
    assert (out_dir / "fig2_deltaG_strip.png").exists()
    assert (out_dir / "fig3_skinscore.png").exists()


def test_retrospective_figures_diagnostic_flags_allow_partial_outputs(tmp_path: Path) -> None:
    write_rankings(tmp_path, missing_skin_weighted="retinol")
    out_dir = tmp_path / "figures"

    res = run_script([
        "scripts/make_retrospective_figures.py",
        "--root", str(tmp_path),
        "--out-dir", str(out_dir),
        "--allow-missing-skin-weighted",
        "--allow-missing-skin-score",
    ])

    assert res.returncode == 0, res.stderr
    assert (out_dir / "fig1_recovery_slope.png").exists()
    assert (out_dir / "fig2_deltaG_strip.png").exists()
    assert not (out_dir / "fig3_skinscore.png").exists()
