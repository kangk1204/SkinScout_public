#!/usr/bin/env python3
"""make_retrospective_figures.py — publication figures from the real demo CSVs.

Reads the committed demo rankings (Vina + skin-weighted) and the SkinScore TSV,
and renders three figures into results/figures/:

  fig1_recovery_slope.png  — per (compound, known-target) rank: Vina → skin-weighted
  fig2_deltaG_strip.png    — per-compound ΔG distribution, known targets highlighted
  fig3_skinscore.png       — SkinScore proteome distribution + known skin proteins

Every number is read from disk; nothing is hard-coded except the known-target map
(literature ground truth) and figure styling.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOG = logging.getLogger("figs")
ROOT = Path(__file__).resolve().parents[1]

# (compound, {uniprot: label})  — literature ground truth
KNOWN = {
    "retinol":       {"P10276": "RAR-α", "P10826": "RAR-β", "P13631": "RAR-γ"},
    "niacinamide":   {"P40261": "NNMT", "Q96EB6": "SIRT1", "P28907": "CD38"},
    "ascorbic_acid": {"P13674": "P4HA1", "O15460": "P4HA2", "P07237": "P4HB"},
    "aspirin":       {"P23219": "COX-1", "P35354": "COX-2"},
    "resveratrol":   {"Q96EB6": "SIRT1", "P35869": "AHR", "Q16236": "NRF2"},
    "kojic_acid":    {"P14679": "TYR"},
    "alpha_arbutin": {"P14679": "TYR"},
    "egcg":          {"P14780": "MMP9", "P08253": "MMP2", "P26358": "DNMT1", "P14679": "TYR"},
    "caffeine":      {"P30542": "ADORA1", "P29274": "ADORA2A", "P29275": "ADORA2B",
                      "P0DMS8": "ADORA3", "Q14432": "PDE3A"},
}
PRETTY = {
    "retinol": "Retinol", "niacinamide": "Niacinamide", "ascorbic_acid": "Ascorbic acid",
    "aspirin": "Aspirin", "resveratrol": "Resveratrol", "kojic_acid": "Kojic acid",
    "alpha_arbutin": "α-Arbutin", "egcg": "EGCG", "caffeine": "Caffeine",
}
ORDER = ["retinol", "niacinamide", "ascorbic_acid", "aspirin", "resveratrol",
         "egcg", "caffeine", "kojic_acid", "alpha_arbutin"]


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"{label} is required: {path}")


def _validate_columns(
    path: Path,
    required: set[str],
    label: str,
    *,
    sep: str = ",",
) -> pd.DataFrame:
    df = pd.read_csv(path, sep=sep)
    cols = set(df.columns)
    missing = sorted(required - cols)
    if missing:
        raise SystemExit(f"{label} is missing required columns {missing}: {path}")
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


def _validate_nonblank(df: pd.DataFrame, column: str, label: str) -> None:
    invalid = [
        int(idx)
        for idx, value in df[column].items()
        if pd.isna(value) or not str(value).strip()
    ]
    if invalid:
        raise SystemExit(_invalid_row_message(label, column, invalid, "blank values"))


def _validate_numeric(
    df: pd.DataFrame,
    column: str,
    label: str,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
    integer: bool = False,
) -> None:
    _validate_nonblank(df, column, label)
    values = pd.to_numeric(df[column], errors="coerce")
    invalid = [
        int(idx)
        for idx, value in values.items()
        if pd.isna(value) or not np.isfinite(float(value))
    ]
    if invalid:
        raise SystemExit(_invalid_row_message(label, column, invalid, "invalid values"))
    if min_value is not None:
        below = [int(idx) for idx, value in values.items() if float(value) < min_value]
        if below:
            raise SystemExit(
                _invalid_row_message(label, column, below, f"values below {min_value:g}")
            )
    if max_value is not None:
        above = [int(idx) for idx, value in values.items() if float(value) > max_value]
        if above:
            raise SystemExit(
                _invalid_row_message(label, column, above, f"values above {max_value:g}")
            )
    if integer:
        noninteger = [
            int(idx)
            for idx, value in values.items()
            if not float(value).is_integer()
        ]
        if noninteger:
            raise SystemExit(
                _invalid_row_message(label, column, noninteger, "non-integer values")
            )


def _vina(root: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(root / f"results/runs/{name}_demo/03_targets/demo_ranked_targets.csv")


def _skin(root: Path, name: str, *, allow_missing: bool) -> pd.DataFrame | None:
    p = root / f"results/runs/{name}_demo/03_targets/demo_skin_weighted.csv"
    if not p.exists():
        if allow_missing:
            LOG.warning("skin-weighted ranking missing; skip %s in fig1: %s", name, p)
            return None
        raise SystemExit(f"Skin-weighted ranking is required for fig1: {p}")
    return pd.read_csv(p)


def validate_inputs(
    root: Path,
    *,
    allow_missing_skin_weighted: bool,
    allow_missing_skin_score: bool,
) -> None:
    for name in ORDER:
        vina_path = root / f"results/runs/{name}_demo/03_targets/demo_ranked_targets.csv"
        _require_file(vina_path, f"Vina ranking for {name}")
        vina = _validate_columns(
            vina_path,
            {"target_id", "rank", "vina_kcal_mol"},
            f"Vina ranking for {name}",
        )
        _validate_nonblank(vina, "target_id", f"Vina ranking for {name}")
        _validate_numeric(vina, "rank", f"Vina ranking for {name}", min_value=1, integer=True)
        _validate_numeric(vina, "vina_kcal_mol", f"Vina ranking for {name}")
        skin_path = root / f"results/runs/{name}_demo/03_targets/demo_skin_weighted.csv"
        if skin_path.exists():
            skin = _validate_columns(
                skin_path,
                {"target_id", "rank_skin"},
                f"Skin-weighted ranking for {name}",
            )
            _validate_nonblank(skin, "target_id", f"Skin-weighted ranking for {name}")
            _validate_numeric(
                skin,
                "rank_skin",
                f"Skin-weighted ranking for {name}",
                min_value=1,
                integer=True,
            )
        elif not allow_missing_skin_weighted:
            raise SystemExit(f"Skin-weighted ranking is required for fig1: {skin_path}")

    skin_score = root / "data/skin_expression/skin_score.tsv"
    if skin_score.exists():
        skin = _validate_columns(
            skin_score,
            {"uniprot", "skin_score"},
            "SkinScore TSV",
            sep="\t",
        )
        _validate_nonblank(skin, "uniprot", "SkinScore TSV")
        _validate_numeric(skin, "skin_score", "SkinScore TSV", min_value=0, max_value=1)
    elif not allow_missing_skin_score:
        raise SystemExit(f"SkinScore TSV is required for fig3: {skin_score}")


def fig1_recovery_slope(root: Path, out_dir: Path, *, allow_missing_skin_weighted: bool) -> None:
    fig, ax = plt.subplots(figsize=(10, 7.5))
    x_vina, x_skin = 0.0, 1.0
    n = 0
    label_pts: list[tuple[float, str, str]] = []  # (skin_rank, text, color)
    for name in ORDER:
        vina = _vina(root, name)
        skin = _skin(root, name, allow_missing=allow_missing_skin_weighted)
        if skin is None:
            continue
        vrank = dict(zip(vina["target_id"].astype(str), vina["rank"], strict=True))
        if "rank_skin" not in skin.columns:
            raise SystemExit(
                f"Skin-weighted ranking for {name} is missing required columns "
                "['rank_skin']"
            )
        srank_values = pd.to_numeric(skin["rank_skin"], errors="coerce")
        invalid_rank = [
            int(idx)
            for idx, value in srank_values.items()
            if pd.isna(value) or not np.isfinite(float(value)) or float(value) < 1
            or not float(value).is_integer()
        ]
        if invalid_rank:
            raise SystemExit(
                _invalid_row_message(
                    f"Skin-weighted ranking for {name}",
                    "rank_skin",
                    invalid_rank,
                    "invalid values",
                )
            )
        srank = dict(zip(skin["target_id"].astype(str), srank_values.astype(int), strict=True))
        for uid, lbl in KNOWN[name].items():
            if uid not in vrank or uid not in srank:
                continue
            v, s = vrank[uid], srank[uid]
            improved = s < v
            color = "#1a7f37" if improved else ("#9aa0a6" if s == v else "#c1121f")
            ax.plot([x_vina, x_skin], [v, s], "-", color=color, lw=1.7, alpha=0.85,
                    marker="o", markersize=4.5, zorder=2)
            # collect labels for any endpoint that lands in Top-15 either side
            if s <= 15 or v <= 15:
                label_pts.append((s, f"{PRETTY[name]}·{lbl}  (#{v}→#{s})", color))
            n += 1

    # de-overlap labels on the right column: sort by skin rank, enforce a minimum
    # vertical gap in log space by nudging text y-positions.
    label_pts.sort(key=lambda t: t[0])
    log_min_gap = 0.085
    placed_logy: list[float] = []
    for i, (s, text, color) in enumerate(label_pts):
        ly = float(np.log10(max(s, 0.9)))
        if i > 0 and (ly - placed_logy[-1]) < log_min_gap:
            ly = placed_logy[-1] + log_min_gap   # monotonic; no while-loop
        placed_logy.append(ly)
        ty = 10 ** ly
        ax.plot([x_skin, x_skin + 0.09], [s, ty], "-", color=color, lw=0.5, alpha=0.45, zorder=1)
        ax.text(x_skin + 0.10, ty, text, fontsize=7.2, va="center", color=color)

    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_ylim(320, 0.8)
    ax.set_xlim(-0.15, 2.05)
    ax.set_xticks([x_vina, x_skin])
    ax.set_xticklabels(["Vina\n(docking only)", "+ Skin-weighting\n(0.7·dock + 0.3·skin)"], fontsize=10)
    ax.set_ylabel("Rank of known target among 300 receptors (log scale, lower = better)")
    ax.axhline(10, color="#888", ls=":", lw=0.8)
    ax.axhline(30, color="#bbb", ls=":", lw=0.8)
    ax.text(-0.13, 10, "Top-10", fontsize=7, color="#666", va="center")
    ax.text(-0.13, 30, "Top-30", fontsize=7, color="#999", va="center")
    ax.set_title(f"Known-target recovery: docking vs. skin-weighted ({n} compound–target pairs)",
                 fontsize=11.5)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], color="#1a7f37", lw=2, label="improved by skin-weighting"),
        Line2D([0], [0], color="#c1121f", lw=2, label="demoted (systemic target)"),
        Line2D([0], [0], color="#9aa0a6", lw=2, label="unchanged"),
    ], loc="lower left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_recovery_slope.png", dpi=160)
    plt.close(fig)
    LOG.info("fig1 written (%d pairs)", n)


def fig2_deltaG_strip(root: Path, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    rng = np.random.default_rng(0)
    for i, name in enumerate(ORDER):
        df = _vina(root, name)
        e = df["vina_kcal_mol"].astype(float).values
        jitter = rng.uniform(-0.18, 0.18, size=len(e))
        ax.scatter(np.full(len(e), i) + jitter, e, s=6, color="#c9d6e5", alpha=0.6, zorder=1)
        known_uids = set(KNOWN[name])
        kdf = df[df["target_id"].astype(str).isin(known_uids)]
        for _, row in kdf.iterrows():
            lbl = KNOWN[name][str(row["target_id"])]
            ax.scatter([i], [row["vina_kcal_mol"]], s=42, color="#c1121f", zorder=3, edgecolor="k", lw=0.4)
            ax.annotate(f"{lbl} #{int(row['rank'])}", (i, row["vina_kcal_mol"]),
                        xytext=(i + 0.22, row["vina_kcal_mol"]), fontsize=6.5, va="center")
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([PRETTY[n] for n in ORDER], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("AutoDock Vina ΔG (kcal/mol)")
    ax.set_title("Per-compound binding-energy distribution (300 receptors); red = known targets",
                 fontsize=11)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_deltaG_strip.png", dpi=160)
    plt.close(fig)
    LOG.info("fig2 written")


def fig3_skinscore(root: Path, out_dir: Path, *, allow_missing_skin_score: bool) -> None:
    tsv = root / "data/skin_expression/skin_score.tsv"
    if not tsv.exists():
        if allow_missing_skin_score:
            LOG.warning("skin_score.tsv missing; skip fig3: %s", tsv)
            return
        raise SystemExit(f"SkinScore TSV is required for fig3: {tsv}")
    df = pd.read_csv(tsv, sep="\t")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(df["skin_score"], bins=60, color="#c9d6e5", edgecolor="#7892b0", lw=0.3)
    ax.set_yscale("log")
    ax.set_xlabel("SkinScore (0–1)")
    ax.set_ylabel("Number of proteins (log scale)")
    known = {"P02533": "KRT14", "P14679": "TYR", "P20930": "FLG",
             "P03956": "MMP1", "P40261": "NNMT"}
    # collect (score, label), sort by score, stagger label heights to avoid overlap
    items = []
    for uid, lbl in known.items():
        r = df[df["uniprot"] == uid]
        if len(r):
            items.append((float(r["skin_score"].iloc[0]), lbl))
    items.sort()
    ymax = ax.get_ylim()[1]
    levels = [0.40, 0.13, 0.40, 0.13, 0.40]  # alternate high/low
    for i, (sc, lbl) in enumerate(items):
        ax.axvline(sc, color="#c1121f", ls="--", lw=1.0, alpha=0.8)
        ax.annotate(f"{lbl} {sc:.2f}", (sc, ymax * levels[i % len(levels)]),
                    fontsize=8, ha="center", color="#c1121f",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    ax.set_title(f"SkinScore distribution across {len(df):,} human proteins "
                 f"(median {df['skin_score'].median():.3f})", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_skinscore.png", dpi=160)
    plt.close(fig)
    LOG.info("fig3 written")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Project root containing results/runs and data/skin_expression.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Figure output directory. Defaults to ROOT/results/figures.",
    )
    parser.add_argument(
        "--allow-missing-skin-weighted",
        action="store_true",
        help="Diagnostic mode: skip compounds missing demo_skin_weighted.csv in fig1.",
    )
    parser.add_argument(
        "--allow-missing-skin-score",
        action="store_true",
        help="Diagnostic mode: skip fig3 when data/skin_expression/skin_score.tsv is missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out_dir = (args.out_dir if args.out_dir is not None else root / "results" / "figures").resolve()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    validate_inputs(
        root,
        allow_missing_skin_weighted=args.allow_missing_skin_weighted,
        allow_missing_skin_score=args.allow_missing_skin_score,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fig1_recovery_slope(root, out_dir, allow_missing_skin_weighted=args.allow_missing_skin_weighted)
    fig2_deltaG_strip(root, out_dir)
    fig3_skinscore(root, out_dir, allow_missing_skin_score=args.allow_missing_skin_score)
    LOG.info("All figures → %s", out_dir)


if __name__ == "__main__":
    main()
