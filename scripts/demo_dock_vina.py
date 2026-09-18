#!/usr/bin/env python3
"""demo_dock_vina.py — DEMO Stage 3 target identification with AutoDock Vina.

A CPU-based stand-in for the full AutoDock-GPU 전수 docking, used to demonstrate
the Stage 3 flow end-to-end on the real AlphaFold receptor set built in Stage 0:

    ligand SDF → Meeko ligand PDBQT
    → for each of the first N with-pocket receptors:
         Vina dock (receptor PDBQT + P2Rank box) → best binding energy
    → rank by energy → demo ranked_targets.csv

This is NOT the production engine (AutoDock-GPU over all 20k receptors); it
proves the pipeline produces a sensible ranked target list from real inputs.
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

LOG = logging.getLogger("demo.dock_vina")


def prepare_ligand_pdbqt(sdf: Path, out_pdbqt: Path) -> bool:
    if not shutil.which("mk_prepare_ligand.py"):
        return False
    out_pdbqt.parent.mkdir(parents=True, exist_ok=True)

    # meeko 0.7.x requires explicit Hs + 3D coords on the input. Stage 1's
    # standardize/protonate outputs hold implicit Hs and only 2D coords for
    # several inputs, which makes mk_prepare_ligand fail with
    # "RDKit molecule has implicit Hs. Need explicit Hs.". Rewrite the SDF
    # with AddHs + an ETKDGv3 embed so any Stage 1 intermediate works.
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        rdkit_ready = False
    else:
        rdkit_ready = True

    src_sdf = sdf
    if rdkit_ready:
        sup = Chem.SDMolSupplier(str(sdf), removeHs=False)
        mol = next((m for m in sup if m is not None), None)
        if mol is None:
            return False
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) == -1:
            # fallback: random coords if ETKDG fails (e.g. very small / strained)
            AllChem.EmbedMolecule(mol, useRandomCoords=True)
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        except ValueError:
            pass
        fixed = out_pdbqt.with_suffix(".prepped.sdf")
        writer = Chem.SDWriter(str(fixed))
        writer.write(mol)
        writer.close()
        src_sdf = fixed

    res = subprocess.run(
        ["mk_prepare_ligand.py", "-i", str(src_sdf), "-o", str(out_pdbqt)],
        capture_output=True, text=True,
    )
    return res.returncode == 0 and out_pdbqt.exists()


def parse_box(box_path: Path) -> tuple[list[float], list[float]] | None:
    if not box_path.exists():
        return None
    f: dict[str, float] = {}
    for line in box_path.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            try:
                f[k.strip()] = float(v.strip())
            except ValueError:
                pass
    try:
        return (
            [f["center_x"], f["center_y"], f["center_z"]],
            [f["size_x"], f["size_y"], f["size_z"]],
        )
    except KeyError:
        return None


def dock_one(vina_cls, receptor: Path, ligand_pdbqt: Path,
             center: list[float], size: list[float],
             exhaustiveness: int, sf_name: str = "vina") -> float | None:
    try:
        v = vina_cls(sf_name=sf_name, verbosity=0)
        v.set_receptor(str(receptor))
        v.set_ligand_from_file(str(ligand_pdbqt))
        v.compute_vina_maps(center=center, box_size=size)
        v.dock(exhaustiveness=exhaustiveness, n_poses=3)
        energies = v.energies(n_poses=1)
        return float(energies[0][0])  # best total affinity (kcal/mol)
    except Exception as exc:  # noqa: BLE001 — per-receptor robustness
        LOG.debug("dock fail %s: %s", receptor.name, exc)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ligand-sdf", required=True, type=Path)
    parser.add_argument("--pdbqt-dir", required=True, type=Path)
    parser.add_argument("--box-dir", required=True, type=Path)
    parser.add_argument("--n-receptors", type=int, default=300,
                        help="DEMO subset size (random sample of with-pocket receptors).")
    parser.add_argument("--include", default="",
                        help="Comma-separated UniProt IDs to force-include "
                             "(e.g. known targets for a recovery check).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--score-func", default="vina", choices=("vina", "vinardo", "ad4"),
                        help="Vina scoring function. 'vinardo' is more accurate "
                             "on metal/halogen-binding pockets (recovers e.g. TYR-Cu²⁺ "
                             "ligands better than default 'vina').")
    parser.add_argument("--out-csv", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    try:
        from vina import Vina
    except ImportError:
        raise SystemExit("AutoDock Vina python API not installed (pip install vina)")

    import random

    with tempfile.TemporaryDirectory() as tmp:
        ligand_pdbqt = Path(tmp) / "ligand.pdbqt"
        if not prepare_ligand_pdbqt(args.ligand_sdf, ligand_pdbqt):
            raise SystemExit("Failed to prepare ligand PDBQT via Meeko")
        LOG.info("Ligand PDBQT ready: %s", ligand_pdbqt)

        # Force-include named targets (e.g. known binders) + a random sample of
        # the rest, so the demo ranking is interpretable.
        forced = [u.strip() for u in args.include.split(",") if u.strip()]
        all_recs = sorted(args.pdbqt_dir.glob("*.pdbqt"))
        by_uid = {p.stem: p for p in all_recs}
        chosen: list[Path] = [by_uid[u] for u in forced if u in by_uid]
        rest = [p for p in all_recs if p.stem not in set(forced)]
        random.Random(args.seed).shuffle(rest)
        chosen += rest[: max(0, args.n_receptors - len(chosen))]
        receptors = chosen
        LOG.info("Docking ligand against %d receptors (%d forced + random), exhaustiveness=%d",
                 len(receptors), len([u for u in forced if u in by_uid]), args.exhaustiveness)

        rows: list[tuple[str, float]] = []
        for i, rec in enumerate(receptors, 1):
            uid = rec.stem
            box = parse_box(args.box_dir / f"{uid}.box.txt")
            if box is None:
                continue
            energy = dock_one(Vina, rec, ligand_pdbqt, box[0], box[1],
                              args.exhaustiveness, sf_name=args.score_func)
            if energy is not None:
                rows.append((uid, energy))
            if i % 25 == 0:
                LOG.info("  … %d/%d docked (%d scored)", i, len(receptors), len(rows))

    rows.sort(key=lambda r: r[1])  # most negative (strongest) first
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "target_id", "vina_kcal_mol"])
        for rank, (uid, e) in enumerate(rows, 1):
            w.writerow([rank, uid, f"{e:.2f}"])
    LOG.info("Wrote %s — top hit: %s",
             args.out_csv, rows[0] if rows else "none")


if __name__ == "__main__":
    main()
