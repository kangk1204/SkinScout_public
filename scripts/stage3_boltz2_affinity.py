#!/usr/bin/env python3
"""stage3_boltz2_affinity.py — Boltz-2 co-folding affinity head for top-X%.

For each candidate receptor:
  • When the sequence exceeds `--max-residues` (12 GB GPU host constraint),
    crop to residues within `--crop-radius` Å of the pocket centre from
    `--box-dir` and record the original → cropped residue mapping. Without a
    box the target is rejected with an explanatory error, never truncated.
  • Invoke the Boltz-2 YAML CLI (`boltz predict input.yaml --out_dir ...`).
  • Capture the affinity head and emit a higher=better score.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from boltz2_runner import (
    ReceptorLengthError,
    affinity_score,
    load_affinity_payload,
    nonempty,
    prepare_receptor_for_boltz,
    run_boltz_predict,
    write_affinity_yaml,
)

LOG = logging.getLogger("stage3.boltz2_affinity")


def _remove_outputs(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def _tmp_output(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_suffix(path.suffix + ".tmp")


def _invalid_row_message(column: str, rows: list[int], detail: str) -> str:
    shown = ", ".join(str(idx) for idx in rows[:10])
    suffix = "..." if len(rows) > 10 else ""
    return (
        f"Boltz-2 top-target CSV column '{column}' contains {detail} at row "
        f"index(es) {shown}{suffix}"
    )


def _validate_cli_args(max_residues: int, crop_radius: float) -> None:
    if max_residues < 1:
        raise SystemExit(f"--max-residues must be >= 1: {max_residues}")
    if not math.isfinite(crop_radius) or crop_radius <= 0:
        raise SystemExit(
            f"--crop-radius must be a finite value > 0: {crop_radius:g}"
        )


def _read_top_csv(path: Path) -> pd.DataFrame:
    if not nonempty(path):
        raise SystemExit(
            f"Boltz-2 top-target CSV is required and must be non-empty: {path}"
        )
    top = pd.read_csv(path)
    if "target_id" not in top.columns:
        raise SystemExit(
            f"Boltz-2 top-target CSV missing required column 'target_id': {path}"
        )
    if top.empty:
        raise SystemExit(f"Boltz-2 top-target CSV contains no rows: {path}")
    blank_targets = [
        int(idx)
        for idx, value in top["target_id"].items()
        if pd.isna(value) or not str(value).strip()
    ]
    if blank_targets:
        raise SystemExit(
            _invalid_row_message("target_id", blank_targets, "blank values")
        )
    top["target_id"] = top["target_id"].astype(str).str.strip()
    duplicate_targets = top["target_id"][top["target_id"].duplicated()].tolist()
    if duplicate_targets:
        shown = ", ".join(duplicate_targets[:10])
        suffix = "..." if len(duplicate_targets) > 10 else ""
        raise SystemExit(
            "Boltz-2 top-target CSV contains duplicate target_id values: "
            f"{shown}{suffix}"
        )
    return top


def _affinity_score_or_fail(payload: dict, receptor_pdb: Path) -> float:
    score = affinity_score(payload)
    if score is None:
        raise SystemExit(
            "Boltz-2 affinity payload for "
            f"{receptor_pdb.name} produced no valid affinity score"
        )
    return score


def call_boltz(receptor_pdb: Path, ligand_sdf: Path, workdir: Path,
               max_residues: int, crop_radius: float,
               *, pocket_path: Path | None = None) -> float | None:
    """Score one target, saying why when it cannot.

    Every failure path used to return None in silence, so a run that scored 44
    of 50 targets left no record of what happened to the other six - and those
    six can be the ones the other scorers ranked highest.
    """
    label = receptor_pdb.name
    if not nonempty(receptor_pdb) or not nonempty(ligand_sdf):
        LOG.warning("Boltz-2 skipped %s: receptor or ligand is missing/empty", label)
        return None
    if not shutil.which("boltz"):
        LOG.warning("Boltz-2 skipped %s: boltz is not on PATH", label)
        return None
    try:
        model_receptor, crop_meta = prepare_receptor_for_boltz(
            receptor_pdb,
            max_residues=max_residues,
            crop_radius=crop_radius,
            out_dir=workdir,
            label=label,
            pocket_path=pocket_path,
        )
    except ReceptorLengthError as exc:
        raise SystemExit(f"Boltz-2 receptor length cap exceeded: {exc}") from exc
    if crop_meta["crop_mode"] == "pocket_crop":
        LOG.info(
            "Boltz-2 cropped %s from %d to %d residues; mapping: %s",
            label,
            crop_meta["input_residues"],
            crop_meta["model_residues"],
            crop_meta["crop_map"],
        )
    input_yaml = workdir / "input.yaml"
    out_dir = workdir / "boltz_out"
    try:
        write_affinity_yaml(
            model_receptor,
            ligand_sdf,
            input_yaml,
            crop_map=(
                Path(crop_meta["crop_map"]) if crop_meta.get("crop_map") else None
            ),
        )
    except ValueError as exc:
        LOG.warning("Boltz-2 skipped %s: could not build input YAML: %s", label, exc)
        return None
    res = run_boltz_predict(
        input_yaml,
        out_dir,
        diffusion_samples=1,
        recycling_steps=3,
    )
    if res is None:
        LOG.warning("Boltz-2 failed for %s: prediction did not run", label)
        return None
    if res.returncode != 0:
        detail = (res.stderr or res.stdout or "").strip()[-500:]
        LOG.warning("Boltz-2 failed for %s: exit %d: %s", label, res.returncode, detail)
        return None
    payload = load_affinity_payload(out_dir)
    if payload is None:
        LOG.warning(
            "Boltz-2 reported success for %s but wrote no affinity payload", label
        )
        return None
    return _affinity_score_or_fail(payload, receptor_pdb)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-csv", required=True, type=Path)
    parser.add_argument("--ligand-sdf", required=True, type=Path)
    parser.add_argument("--clean-dir", required=True, type=Path)
    parser.add_argument(
        "--box-dir",
        type=Path,
        default=None,
        help="Docking box directory used to centre the pocket crop for long receptors.",
    )
    parser.add_argument("--max-residues", type=int, default=700)
    parser.add_argument("--crop-radius", type=float, default=20.0)
    parser.add_argument("--out-scores", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    _remove_outputs(args.out_scores)
    _validate_cli_args(args.max_residues, args.crop_radius)
    top = _read_top_csv(args.top_csv)
    if not top.empty and not shutil.which("boltz"):
        raise SystemExit("boltz is not available on PATH")
    if not top.empty and not nonempty(args.ligand_sdf):
        raise SystemExit(f"Ligand SDF is missing or empty for Boltz-2 affinity: {args.ligand_sdf}")
    n_written = 0
    tmp_scores = _tmp_output(args.out_scores)
    try:
        with (
            tempfile.TemporaryDirectory() as tmp,
            tmp_scores.open("w", newline="") as fh,
        ):
            w = csv.writer(fh, delimiter="\t")
            w.writerow(["target_id", "boltz2_neg_log_uM", "score"])
            for uid in top["target_id"].astype(str):
                receptor = args.clean_dir / f"{uid}_clean.pdb"
                if not nonempty(receptor):
                    continue
                pocket_path = (
                    args.box_dir / f"{uid}.box.txt"
                    if args.box_dir is not None
                    else None
                )
                score = call_boltz(receptor, args.ligand_sdf,
                                   Path(tmp) / uid,
                                   args.max_residues, args.crop_radius,
                                   pocket_path=pocket_path)
                if score is None:
                    continue
                w.writerow([uid, f"{score:.4f}", f"{score:.4f}"])
                n_written += 1
    except BaseException:
        tmp_scores.unlink(missing_ok=True)
        raise
    if not top.empty and n_written == 0:
        tmp_scores.unlink(missing_ok=True)
        raise SystemExit("Boltz-2 produced no usable affinity scores")
    tmp_scores.replace(args.out_scores)
    LOG.info("Boltz-2 affinities → %s", args.out_scores)


if __name__ == "__main__":
    main()
