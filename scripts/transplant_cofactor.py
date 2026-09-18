#!/usr/bin/env python3
"""transplant_cofactor.py — Splice a metal cofactor into an AlphaFold receptor.

AlphaFold does not model metal cofactors. For metalloproteins like TYR (Cu²⁺ ×2),
MMP-family (Zn²⁺), CA (Zn²⁺) etc., blind docking against bare AF structures
fails because the ligand cannot chelate a metal that isn't there.

This script reads the cleaned AF PDB, identifies the metal-binding residues
(usually His sidechain Nδ1 / Nε2), computes the centroid per metal site, and
appends HETATM lines with the chosen element at those centroids. The resulting
PDB feeds back into Meeko → docking with a tighter box around the metal site.

Built-in metal sites are listed for the four cosmetic-relevant targets that
failed in the v1 retrospective:
    P14679 TYR  → 2× Cu²⁺ (H180/202/211 and H363/367/390)
    P14780 MMP9 → 1× catalytic Zn²⁺  (H401/405/411)
    P08253 MMP2 → 1× catalytic Zn²⁺  (H403/407/413)
    P00918 CA2  → 1× Zn²⁺            (H94/96/119)

Custom mappings can be added by editing METAL_SITES below.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

LOG = logging.getLogger("transplant_cofactor")


@dataclass(frozen=True)
class MetalSite:
    name: str        # e.g. "CuA"
    element: str     # PDB element (right-justified 2 char): "CU", "ZN"
    residues: tuple[int, ...]   # residue numbers (chain A assumed)


METAL_SITES: dict[str, list[MetalSite]] = {
    "P14679": [
        MetalSite("CuA", "CU", (180, 202, 211)),
        MetalSite("CuB", "CU", (363, 367, 390)),
    ],
    "P14780": [
        MetalSite("ZnCat", "ZN", (401, 405, 411)),
    ],
    "P08253": [
        MetalSite("ZnCat", "ZN", (403, 407, 413)),
    ],
    "P00918": [
        MetalSite("Zn", "ZN", (94, 96, 119)),
    ],
}


def parse_residue_atoms(pdb_path: Path,
                        target_residues: set[int]) -> dict[int, dict[str, tuple[float, float, float]]]:
    """Return {resnum: {atomname: (x,y,z)}} for ATOM lines matching target residues."""
    out: dict[int, dict[str, tuple[float, float, float]]] = {}
    for line in pdb_path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        try:
            resnum = int(line[22:26])
        except ValueError:
            continue
        if resnum not in target_residues:
            continue
        name = line[12:16].strip()
        try:
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
        except ValueError:
            continue
        out.setdefault(resnum, {})[name] = (x, y, z)
    return out


def site_centroid(atoms: dict[int, dict[str, tuple[float, float, float]]],
                  residues: tuple[int, ...]) -> tuple[float, float, float] | None:
    """His ND1/NE2 평균 좌표 (둘 다 없으면 CA 사용). 모자라는 잔기는 skip."""
    coords = []
    for r in residues:
        ra = atoms.get(r, {})
        for nm in ("ND1", "NE2", "CA"):
            if nm in ra:
                coords.append(ra[nm])
                break
    if len(coords) < max(1, len(residues) - 1):
        # 너무 많이 누락 → centroid 의미 없음
        return None
    n = len(coords)
    return (sum(c[0] for c in coords) / n,
            sum(c[1] for c in coords) / n,
            sum(c[2] for c in coords) / n)


def append_metal_hetatm(lines: list[str], element: str,
                        center: tuple[float, float, float],
                        resnum: int) -> list[str]:
    """Insert a HETATM right before the END marker (or at the bottom)."""
    serial = 99000 + (resnum % 1000)
    elem = element.rjust(2)
    rec = (
        f"HETATM{serial:5d} {elem:>4} {element[:3]:<3} A{resnum:4d}    "
        f"{center[0]:8.3f}{center[1]:8.3f}{center[2]:8.3f}"
        f"  1.00  0.00          {elem}\n"
    )
    out = []
    inserted = False
    for ln in lines:
        if not inserted and ln.startswith("END"):
            out.append(rec)
            inserted = True
        out.append(ln + ("\n" if not ln.endswith("\n") else ""))
    if not inserted:
        out.append(rec)
    return out


def build_box(centroids: list[tuple[float, float, float]],
              edge: float = 18.0) -> dict[str, float]:
    """Bounding box of all metal centroids, expanded by `edge/2` Å each axis."""
    xs = [c[0] for c in centroids]; ys = [c[1] for c in centroids]; zs = [c[2] for c in centroids]
    cx, cy, cz = sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs)
    return {"center_x": cx, "center_y": cy, "center_z": cz,
            "size_x": edge, "size_y": edge, "size_z": edge}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uniprot", required=True,
                        help="UniProt accession (must have an entry in METAL_SITES)")
    parser.add_argument("--in-pdb", required=True, type=Path,
                        help="AF cleaned PDB (data/human_clean/<uid>_clean.pdb)")
    parser.add_argument("--out-pdb", required=True, type=Path)
    parser.add_argument("--out-box", type=Path, default=None)
    parser.add_argument("--box-edge", type=float, default=18.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    sites = METAL_SITES.get(args.uniprot)
    if not sites:
        raise SystemExit(f"No METAL_SITES entry for {args.uniprot}. Edit transplant_cofactor.py.")

    LOG.info("UniProt %s: %d metal sites", args.uniprot, len(sites))

    target_res = set()
    for s in sites:
        target_res.update(s.residues)
    atoms = parse_residue_atoms(args.in_pdb, target_res)

    lines = args.in_pdb.read_text().splitlines()
    centroids: list[tuple[float, float, float]] = []
    for s in sites:
        c = site_centroid(atoms, s.residues)
        if c is None:
            LOG.warning("Skipping %s — too many missing residues among %s",
                        s.name, s.residues)
            continue
        LOG.info("  %s (%s × %d His) centroid = (%.2f, %.2f, %.2f)",
                 s.name, s.element, len(s.residues), *c)
        centroids.append(c)
        lines = append_metal_hetatm(lines, s.element, c, max(s.residues))

    if not centroids:
        raise SystemExit("No metal sites could be transplanted — check residue numbering.")

    args.out_pdb.parent.mkdir(parents=True, exist_ok=True)
    args.out_pdb.write_text("".join(
        ln if ln.endswith("\n") else ln + "\n" for ln in lines
    ))
    LOG.info("Wrote %s (+%d HETATM)", args.out_pdb, len(centroids))

    if args.out_box:
        box = build_box(centroids, args.box_edge)
        args.out_box.parent.mkdir(parents=True, exist_ok=True)
        if args.out_box.suffix == ".json":
            args.out_box.write_text(json.dumps(box, indent=2))
        else:
            args.out_box.write_text("\n".join(f"{k} = {v:.3f}" for k, v in box.items()) + "\n")
        LOG.info("Wrote box → %s  (edge %.1f Å, center %.2f/%.2f/%.2f)",
                 args.out_box, args.box_edge,
                 box["center_x"], box["center_y"], box["center_z"])


if __name__ == "__main__":
    main()
