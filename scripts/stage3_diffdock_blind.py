#!/usr/bin/env python3
"""stage3_diffdock_blind.py — DiffDock-L blind docking for no-pocket receptors."""

from __future__ import annotations

import argparse
import csv
import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

LOG = logging.getLogger("stage3.diffdock")


def _remove_outputs(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def _tmp_output(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_suffix(path.suffix + ".tmp")


def _write_text_atomic(text: str, path: Path) -> None:
    tmp = _tmp_output(path)
    tmp.write_text(text)
    tmp.replace(path)


def diffdock(receptor_pdb: Path, ligand_sdf_or_pdbqt: Path,
             workdir: Path) -> float | None:
    if not shutil.which("diffdock"):
        return None
    cmd = [
        "diffdock",
        "--protein_path", str(receptor_pdb),
        "--ligand", str(ligand_sdf_or_pdbqt),
        "--out_dir", str(workdir),
        "--samples_per_complex", "5",
        "--inference_steps", "20",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        return None
    # DiffDock-L emits rank1_confidence-<score>.sdf — parse <score>.
    for sdf in workdir.glob("rank1_confidence*.sdf"):
        match = re.search(r"rank1_confidence_?(-?\d+(?:\.\d+)?)", sdf.stem)
        if match is None:
            continue
        try:
            return float(match.group(1))
        except ValueError:
            continue
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ligand", required=True, type=Path,
                        help="SDF or PDBQT ligand input.")
    parser.add_argument("--no-pocket-list", required=True, type=Path)
    parser.add_argument("--clean-dir", required=True, type=Path)
    parser.add_argument("--out-scores", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    _remove_outputs(args.out_scores)
    if not args.no_pocket_list.exists():
        raise SystemExit(f"No-pocket target list is required: {args.no_pocket_list}")

    uids = [u.strip() for u in args.no_pocket_list.read_text().splitlines() if u.strip()]
    LOG.info("DiffDock-L blind docking %d receptors", len(uids))
    if not uids:
        _write_text_atomic(
            "target_id\tdiffdock_confidence\tscore\tneg_vina_score\n",
            args.out_scores,
        )
        return

    rows: list[list[str]] = []
    with tempfile.TemporaryDirectory() as tmp:
        for uid in uids:
            receptor = args.clean_dir / f"{uid}_clean.pdb"
            if not receptor.exists():
                continue
            score = diffdock(receptor, args.ligand, Path(tmp) / uid)
            if score is None:
                continue
            # DiffDock confidence is higher = better; treat as a pseudo-energy
            # for the RRF/pick_top pipeline.
            rows.append([uid, f"{score:.4f}", f"{score:.4f}", f"{score:.4f}"])
    if not rows:
        raise SystemExit("DiffDock-L produced no blind docking scores for no-pocket targets")

    tmp_scores = _tmp_output(args.out_scores)
    with tmp_scores.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["target_id", "diffdock_confidence", "score", "neg_vina_score"])
        w.writerows(rows)
    tmp_scores.replace(args.out_scores)
    LOG.info("DiffDock-L blind scores → %s", args.out_scores)


if __name__ == "__main__":
    main()
