#!/usr/bin/env python3
"""Derive a docking box from the real extent of a P2Rank pocket.

`stage0_meeko_prep.write_box` sized every box as
``min(2 * (max(radius, 8) + 4), 30)``, and every ``*.pockets.json`` carries the
constant fallback ``radius = 12.0`` — the value is 12.0 in all 2,985 non-empty
pocket files. The expression therefore collapsed to the 30 A cap for all 15,038
receptors and box size never depended on pocket geometry at all.

P2Rank does publish the geometry: ``*_predictions.csv`` lists the surface atom
serials of each pocket, and those resolve against the cleaned PDB. Measured over
a 400-target sample the derived box is smaller than the 30 A cube for 88% of
targets (median volume 0.40x) and *larger* for 12% — so the constant was
simultaneously wasting search volume on most receptors and truncating the pocket
on the rest.

The box stays centred on P2Rank's predicted pocket centre, which is where a
ligand is expected to sit; only the extent is derived.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np


SCHEMA_VERSION = "skinscout.pocket-derived-box.v1"
# Headroom past the outermost pocket surface atom, so a ligand has room to sit
# against the wall rather than being pinned to the atom centres.
DEFAULT_HEADROOM = 4.0
DEFAULT_MIN_EDGE = 16.0
DEFAULT_MAX_EDGE = 40.0
# Three atoms can be collinear; four is the smallest set that can bound a volume.
MIN_SURFACE_ATOMS = 4
AXES = ("x", "y", "z")


class PocketBoxError(ValueError):
    """Raised when a pocket cannot yield a trustworthy box."""


@dataclass(frozen=True, slots=True)
class PocketBox:
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    half_width: tuple[float, float, float]
    surface_atoms: int
    clamped_low: tuple[str, ...]
    clamped_high: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "center": list(self.center),
            "size": list(self.size),
            "half_width": list(self.half_width),
            "surface_atoms": self.surface_atoms,
            "clamped_low": list(self.clamped_low),
            "clamped_high": list(self.clamped_high),
        }


def read_top_pocket(prediction_csv: Path) -> dict[str, Any]:
    """Return the rank-1 pocket's centre and surface atom serials."""
    try:
        text = prediction_csv.read_text(encoding="utf-8")
    except OSError as exc:
        raise PocketBoxError(f"pocket prediction unreadable: {prediction_csv}: {exc}") from exc
    rows = [
        {(key or "").strip(): (value or "").strip() for key, value in row.items()}
        for row in csv.DictReader(text.splitlines())
    ]
    ranked = [row for row in rows if row.get("rank") == "1"]
    if not ranked:
        raise PocketBoxError(f"no rank-1 pocket in {prediction_csv}")
    row = ranked[0]
    try:
        center = tuple(float(row[f"center_{axis}"]) for axis in AXES)
    except (KeyError, ValueError) as exc:
        raise PocketBoxError(f"rank-1 pocket has no usable centre: {prediction_csv}") from exc
    if not all(math.isfinite(value) for value in center):
        raise PocketBoxError(f"rank-1 pocket centre is not finite: {prediction_csv}")
    serials = [token for token in row.get("surf_atom_ids", "").split() if token]
    if not serials:
        raise PocketBoxError(f"rank-1 pocket lists no surface atoms: {prediction_csv}")
    try:
        atom_ids = {int(token) for token in serials}
    except ValueError as exc:
        raise PocketBoxError(
            f"rank-1 pocket surface atom serial is not an integer: {prediction_csv}"
        ) from exc
    return {"center": center, "atom_ids": atom_ids}


def read_atom_coordinates(pdb_path: Path, atom_ids: set[int]) -> np.ndarray:
    """Resolve atom serials against a PDB, refusing a partial match.

    A silently dropped atom would shrink the derived box, so a missing serial is
    an error rather than a smaller pocket.
    """
    if not atom_ids:
        raise PocketBoxError("no atom serials requested")
    try:
        lines = pdb_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PocketBoxError(f"structure unreadable: {pdb_path}: {exc}") from exc
    found: dict[int, tuple[float, float, float]] = {}
    for line in lines:
        if not line.startswith(("ATOM", "HETATM")):
            continue
        try:
            serial = int(line[6:11])
        except (ValueError, IndexError):
            continue
        if serial not in atom_ids or serial in found:
            continue
        try:
            found[serial] = (
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            )
        except (ValueError, IndexError) as exc:
            raise PocketBoxError(
                f"atom {serial} has unparsable coordinates in {pdb_path}"
            ) from exc
    missing = sorted(atom_ids - set(found))
    if missing:
        shown = ", ".join(str(value) for value in missing[:10])
        raise PocketBoxError(
            f"{len(missing)} pocket surface atom(s) absent from {pdb_path}: {shown}"
        )
    coordinates = np.array([found[serial] for serial in sorted(found)], dtype=float)
    if not np.isfinite(coordinates).all():
        raise PocketBoxError(f"pocket surface atom coordinates are not finite: {pdb_path}")
    return coordinates


def derive_box(
    center: Sequence[float],
    surface_coordinates: np.ndarray,
    *,
    headroom: float = DEFAULT_HEADROOM,
    min_edge: float = DEFAULT_MIN_EDGE,
    max_edge: float = DEFAULT_MAX_EDGE,
    min_surface_atoms: int = MIN_SURFACE_ATOMS,
) -> PocketBox:
    """Size a box around ``center`` so it contains the pocket's surface atoms."""
    if not math.isfinite(headroom) or headroom < 0.0:
        raise PocketBoxError(f"headroom must be finite and >= 0: {headroom!r}")
    for name, value in (("min_edge", min_edge), ("max_edge", max_edge)):
        if not math.isfinite(value) or value <= 0.0:
            raise PocketBoxError(f"{name} must be finite and > 0: {value!r}")
    if min_edge > max_edge:
        raise PocketBoxError(f"min_edge {min_edge:g} exceeds max_edge {max_edge:g}")
    origin = np.asarray(center, dtype=float)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise PocketBoxError("pocket centre must be three finite coordinates")
    coordinates = np.asarray(surface_coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise PocketBoxError("surface coordinates must be an (n, 3) array")
    if len(coordinates) < min_surface_atoms:
        raise PocketBoxError(
            f"pocket has {len(coordinates)} surface atom(s); "
            f"at least {min_surface_atoms} are needed to bound a volume"
        )

    # Chebyshev half-width per axis: the box must reach the outermost atom.
    half = np.abs(coordinates - origin).max(axis=0)
    raw = 2.0 * (half + headroom)
    clamped_low = tuple(AXES[i] for i in range(3) if raw[i] < min_edge)
    clamped_high = tuple(AXES[i] for i in range(3) if raw[i] > max_edge)
    size = np.clip(raw, min_edge, max_edge)
    if not np.isfinite(size).all() or (size <= 0).any():
        raise PocketBoxError("derived box size is not positive and finite")
    return PocketBox(
        center=(float(origin[0]), float(origin[1]), float(origin[2])),
        size=(float(size[0]), float(size[1]), float(size[2])),
        half_width=(float(half[0]), float(half[1]), float(half[2])),
        surface_atoms=int(len(coordinates)),
        clamped_low=clamped_low,
        clamped_high=clamped_high,
    )


def format_box_text(box: PocketBox) -> str:
    lines = [f"center_{axis} = {value:.3f}" for axis, value in zip(AXES, box.center, strict=True)]
    lines += [f"size_{axis} = {value:.3f}" for axis, value in zip(AXES, box.size, strict=True)]
    return "\n".join([*lines, ""])


def box_for_target(
    prediction_csv: Path,
    pdb_path: Path,
    **kwargs: Any,
) -> PocketBox:
    """Read one target's pocket and structure and derive its box."""
    pocket = read_top_pocket(prediction_csv)
    coordinates = read_atom_coordinates(pdb_path, pocket["atom_ids"])
    return derive_box(pocket["center"], coordinates, **kwargs)
