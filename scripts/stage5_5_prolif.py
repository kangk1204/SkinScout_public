#!/usr/bin/env python3
"""stage5_5_prolif.py — Compute ProLIF interaction fingerprint per target."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path

import pandas as pd

LOG = logging.getLogger("stage5_5.prolif")


def kept_complexes(report_path: Path) -> list[tuple[str, Path]]:
    if not report_path.exists() or report_path.stat().st_size == 0:
        raise SystemExit(f"Boltz report is required and must be non-empty: {report_path}")
    report = pd.read_csv(report_path, sep="\t", skip_blank_lines=False)
    required = {"target_id", "complex_pdb", "kept"}
    missing = sorted(required - set(report.columns))
    if missing:
        raise SystemExit(f"Boltz report missing required columns: {missing}")
    if report.empty:
        raise SystemExit("Boltz report contains no rows")
    if not set(report["kept"].astype(str)).issubset({"yes", "no"}):
        raise SystemExit("Boltz report kept column must contain only yes/no")
    rows: list[tuple[str, Path]] = []
    for _, row in report[report["kept"] == "yes"].iterrows():
        target_id = str(row["target_id"]).strip()
        complex_pdb = Path(str(row["complex_pdb"]).strip())
        if not target_id or not complex_pdb.exists() or complex_pdb.stat().st_size == 0:
            raise SystemExit(f"Kept Boltz complex is missing or empty for {target_id}: {complex_pdb}")
        rows.append((target_id, complex_pdb))
    if not rows:
        raise SystemExit("Boltz report contains no kept complex poses")
    return rows


def _coerce_atom_index(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            return None
    if isinstance(value, bool):
        return None
    try:
        atom_idx = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return atom_idx if atom_idx >= 0 else None


def _as_iterable_indices(value: object) -> Iterable[object]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (value,)
    if isinstance(value, Iterable):
        return value
    return (value,)


def _ligand_indices_from_metadata(metadata: Mapping[object, object]) -> set[int]:
    indices: set[int] = set()
    # ProLIF's residue-local ``indices`` can differ from the ligand molecule
    # order. ``parent_indices`` maps the interaction back to the parent
    # Molecule.from_mda(ligand_atoms), whose order matches the AtomGroup.
    for key in ("parent_indices", "indices"):
        payload = metadata.get(key)
        if not isinstance(payload, Mapping):
            continue
        ligand_values = payload.get("ligand")
        for value in _as_iterable_indices(ligand_values):
            atom_idx = _coerce_atom_index(value)
            if atom_idx is not None:
                indices.add(atom_idx)
        if indices:
            return indices
    return indices


def _iter_interaction_metadata(value: object) -> Iterable[Mapping[object, object]]:
    if isinstance(value, Mapping):
        if isinstance(value.get("parent_indices"), Mapping) or isinstance(value.get("indices"), Mapping):
            yield value
        for child in value.values():
            yield from _iter_interaction_metadata(child)
        return
    metadata = getattr(value, "metadata", None)
    if isinstance(metadata, Mapping):
        yield metadata
    interactions = getattr(value, "interactions", None)
    if callable(interactions):
        for interaction in interactions():
            yield from _iter_interaction_metadata(interaction)
        return
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        for child in value:
            yield from _iter_interaction_metadata(child)


def ligand_atom_indices_from_fingerprint(
    fp: object,
    *,
    ligand_atom_count: int | None = None,
) -> set[int]:
    indices: set[int] = set()
    interaction_payload = getattr(fp, "ifp", fp)
    for metadata in _iter_interaction_metadata(interaction_payload):
        indices.update(_ligand_indices_from_metadata(metadata))
    if ligand_atom_count is not None:
        out_of_range = sorted(index for index in indices if index >= ligand_atom_count)
        if out_of_range:
            raise ValueError(
                "ProLIF parent ligand atom indices exceed the bound ligand atom count "
                f"{ligand_atom_count}: {out_of_range}"
            )
    return indices


def fingerprint_complex(plf: object, mda: object, pdb: Path) -> set[int]:
    universe = mda.Universe(str(pdb))
    protein_atoms = universe.select_atoms("protein")
    ligand_atoms = universe.select_atoms("not protein")
    if protein_atoms.n_atoms == 0:
        raise ValueError("complex contains no protein atoms")
    if ligand_atoms.n_atoms == 0:
        raise ValueError("complex contains no non-protein ligand atoms")
    if len(ligand_atoms.residues) != 1:
        raise ValueError(
            "complex must contain exactly one non-protein ligand residue; "
            f"found {len(ligand_atoms.residues)}"
        )
    ligand = plf.Molecule.from_mda(ligand_atoms)
    protein = plf.Molecule.from_mda(protein_atoms)
    interactions = plf.Fingerprint().generate(ligand, protein, metadata=True)
    return ligand_atom_indices_from_fingerprint(
        interactions,
        ligand_atom_count=ligand_atoms.n_atoms,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boltz-dir", required=True, type=Path)
    parser.add_argument("--boltz-report", type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Emit an explicit empty degraded CSV when ProLIF evidence is unavailable.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.out_csv.exists():
        args.out_csv.unlink()
    if not args.boltz_dir.exists():
        raise SystemExit(f"Boltz output directory does not exist: {args.boltz_dir}")
    try:
        import prolif as plf  # noqa: F401  (heavy import)
        import MDAnalysis as mda
        has_prolif = True
    except ImportError:
        has_prolif = False
        if not args.allow_empty:
            raise SystemExit(
                "ProLIF and MDAnalysis are required for Stage 5.5 pharmacophore evidence; "
                "use --allow-empty only for explicit degraded diagnostics"
            )
        LOG.warning("prolif/MDAnalysis missing; emitting explicit empty degraded CSV")

    rows: list[dict] = []
    n_complexes = 0
    failed_targets: list[str] = []
    if has_prolif:
        if args.boltz_report is None:
            raise SystemExit("--boltz-report is required to select quality-approved complexes")
        import prolif as plf
        import MDAnalysis as mda
        complexes = kept_complexes(args.boltz_report)
        for target_id, pdb in complexes:
            n_complexes += 1
            try:
                atom_indices = fingerprint_complex(plf, mda, pdb)
                if not atom_indices:
                    raise ValueError("cannot extract ligand atom_idx from ProLIF interaction metadata")
                rows.extend(
                    {"target_id": target_id, "atom_idx": atom_idx}
                    for atom_idx in sorted(atom_indices)
                )
            except Exception as exc:  # noqa: BLE001
                failed_targets.append(target_id)
                LOG.warning("ProLIF failed for %s: %s", target_id, exc)

    if has_prolif and n_complexes == 0:
        raise SystemExit(f"No Boltz complex PDBs found under {args.boltz_dir}")
    if failed_targets and not args.allow_empty:
        raise SystemExit(
            "ProLIF failed for target(s) "
            + ",".join(failed_targets)
            + "; use --allow-empty only for explicit degraded diagnostics"
        )
    if has_prolif and not rows and not args.allow_empty:
        raise SystemExit("No ProLIF interaction fingerprints were produced")

    df_out = pd.DataFrame(rows, columns=["target_id", "atom_idx"])
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = args.out_csv.with_suffix(args.out_csv.suffix + ".tmp")
    df_out.to_csv(tmp_csv, index=False)
    tmp_csv.replace(args.out_csv)
    LOG.info("ProLIF table → %s (n=%d)", args.out_csv, len(df_out))


if __name__ == "__main__":
    main()
