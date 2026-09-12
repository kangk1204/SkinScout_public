#!/usr/bin/env python3
"""stage1_etkdg.py — ETKDGv3 conformer generation + MMFF94 ranking + RMSD prune."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

LOG = logging.getLogger("stage1.etkdg")


def _remove_outputs(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def _write_sdf_atomic(
    mol: Chem.Mol,
    conformers: list[tuple[float, int, int]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    writer = Chem.SDWriter(str(tmp))
    for energy, status, cid in conformers:
        mol.SetProp("mmff_energy", f"{energy:.3f}")
        mol.SetProp("mmff_status", "converged" if status == 0 else "not_converged")
        writer.write(mol, confId=cid)
    writer.close()
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-sdf", required=True, type=Path)
    parser.add_argument("--out-sdf", required=True, type=Path)
    parser.add_argument("--n-conf", type=int, default=50)
    parser.add_argument("--rmsd-prune", type=float, default=0.5)
    parser.add_argument("--max-iters", type=int, default=200)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    _remove_outputs(args.out_sdf)
    supplier = Chem.SDMolSupplier(str(args.in_sdf), removeHs=False)
    mol = next((m for m in supplier if m is not None), None)
    if mol is None:
        raise SystemExit(f"No parseable mol in {args.in_sdf}")
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 0xC05A
    params.pruneRmsThresh = args.rmsd_prune
    params.numThreads = 0
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=args.n_conf, params=params)
    if not cids:
        raise SystemExit("ETKDG failed to embed any conformer")

    energies: list[tuple[float, int, int]] = []
    n_nonconverged = 0
    for cid in cids:
        res = AllChem.MMFFOptimizeMolecule(mol, maxIters=args.max_iters, confId=cid)
        props = AllChem.MMFFGetMoleculeProperties(mol)
        if props is None:
            continue
        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
        if ff is None:
            continue
        if res != 0:
            n_nonconverged += 1
        energies.append((ff.CalcEnergy(), res, cid))

    energies.sort()
    if not energies:
        raise SystemExit("MMFF failed to optimize every embedded conformer")
    _write_sdf_atomic(mol, energies, args.out_sdf)
    LOG.info("Wrote %d conformers (%d not converged; best E=%.3f) → %s",
             len(energies), n_nonconverged,
             energies[0][0] if energies else float("nan"), args.out_sdf)


if __name__ == "__main__":
    main()
