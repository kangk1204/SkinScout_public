#!/usr/bin/env python3
"""stage1_xtb_opt.py — xTB GFN2 single-point + opt on the lowest MMFF conformer."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from rdkit import Chem

LOG = logging.getLogger("stage1.xtb")


def _remove_outputs(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def _nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _write_sdf_atomic(mol: Chem.Mol, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    writer = Chem.SDWriter(str(tmp))
    writer.write(mol)
    writer.close()
    tmp.replace(path)


def _write_text_atomic(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _write_json_atomic(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def _load_json_object(path: Path) -> dict[str, Any]:
    if not _nonempty(path):
        raise SystemExit(f"Input metadata missing or empty: {path}")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Input metadata is not valid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Input metadata must be a JSON object: {path}")
    return payload


def _text_or_none(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _required_text_field(payload: dict[str, Any], field: str, path: Path) -> str:
    value = _text_or_none(payload.get(field))
    if value is None:
        raise SystemExit(f"Input metadata missing non-empty {field}: {path}")
    return value


def _canonical_smiles(value: str, field: str, path: Path) -> str:
    mol = Chem.MolFromSmiles(value)
    if mol is None:
        raise SystemExit(f"Input metadata contains invalid {field}: {value!r}: {path}")
    return Chem.MolToSmiles(mol, canonical=True)


def _load_input_provenance(path: Path) -> dict[str, str]:
    payload = _load_json_object(path)
    input_type = _required_text_field(payload, "input_type", path)
    if input_type not in {"smiles", "sdf"}:
        raise SystemExit(f"Input metadata invalid input_type {input_type!r}: {path}")

    if input_type == "smiles":
        if _text_or_none(payload.get("input_sdf")) is not None:
            raise SystemExit(
                f"Input metadata input_type 'smiles' cannot include input_sdf: {path}"
            )
        input_smiles = _required_text_field(payload, "input_smiles", path)
        input_canonical = _required_text_field(
            payload,
            "input_canonical_smiles",
            path,
        )
        expected_input_canonical = _canonical_smiles(input_smiles, "input_smiles", path)
        if input_canonical != expected_input_canonical:
            raise SystemExit(
                "Input metadata input_canonical_smiles does not match input_smiles: "
                f"{input_canonical!r} != {expected_input_canonical!r}: {path}"
            )
        return {
            "input_type": input_type,
            "input_smiles": input_smiles,
            "input_canonical_smiles": input_canonical,
        }

    for field in ("input_smiles", "input_canonical_smiles"):
        if _text_or_none(payload.get(field)) is not None:
            raise SystemExit(
                f"Input metadata input_type 'sdf' cannot include {field}: {path}"
            )
    input_sdf = _required_text_field(payload, "input_sdf", path)
    return {
        "input_type": input_type,
        "input_sdf": input_sdf,
    }


def _identity_mol(mol: Chem.Mol) -> Chem.Mol:
    identity = Chem.RemoveHs(Chem.Mol(mol), sanitize=True)
    if identity is None or identity.GetNumAtoms() == 0:
        raise SystemExit("Could not derive heavy-atom identity molecule for metadata")
    return identity


def write_xyz(mol: Chem.Mol, conf_id: int, path: Path) -> None:
    conf = mol.GetConformer(conf_id)
    lines = [str(mol.GetNumAtoms()), "stage1 xtb input"]
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        lines.append(f"{atom.GetSymbol():<2} {pos.x:>12.6f} {pos.y:>12.6f} {pos.z:>12.6f}")
    path.write_text("\n".join(lines) + "\n")


def read_xyz_back(mol: Chem.Mol, xyz_path: Path) -> bool:
    if not _nonempty(xyz_path):
        return False
    lines = xyz_path.read_text().splitlines()
    if len(lines) < 2 + mol.GetNumAtoms():
        return False
    coords = []
    try:
        n_atoms = int(lines[0].strip())
        if n_atoms != mol.GetNumAtoms():
            return False
        for line in lines[2:2 + mol.GetNumAtoms()]:
            parts = line.split()
            if len(parts) < 4:
                return False
            coords.append((float(parts[1]), float(parts[2]), float(parts[3])))
    except ValueError:
        return False
    conf = mol.GetConformer(0)
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, (x, y, z))
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-sdf", required=True, type=Path)
    parser.add_argument(
        "--in-meta",
        required=True,
        type=Path,
        help="Metadata produced by stage1_standardize.py; input provenance is preserved.",
    )
    parser.add_argument("--out-sdf", required=True, type=Path)
    parser.add_argument("--out-inchikey", required=True, type=Path)
    parser.add_argument("--out-meta", required=True, type=Path)
    parser.add_argument("--gfn", type=int, default=2)
    parser.add_argument(
        "--allow-mmff-fallback",
        action="store_true",
        help="Allow MMFF geometry passthrough when xTB is unavailable or fails.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    _remove_outputs(args.out_sdf, args.out_inchikey, args.out_meta)
    input_provenance = _load_input_provenance(args.in_meta)
    supplier = Chem.SDMolSupplier(str(args.in_sdf), removeHs=False)
    mols = [m for m in supplier if m is not None]
    if not mols:
        raise SystemExit(f"No conformer in {args.in_sdf}")
    base_mol = mols[0]  # lowest MMFF energy after sort in stage1_etkdg.py
    base_mol = Chem.Mol(base_mol)

    xtb_status = "optimized"
    xtb_returncode: int | None = None
    xtb_error_tail = ""

    if not shutil.which("xtb"):
        if not args.allow_mmff_fallback:
            raise SystemExit(
                "xtb is required for Stage 1 QM refinement; install it or set "
                "stage1.allow_xtb_fallback=true for an explicit MMFF degraded run"
            )
        LOG.warning("xtb not found on PATH — explicit MMFF fallback enabled.")
        out_mol = base_mol
        xtb_status = "fallback_mmff"
        xtb_error_tail = "xtb_not_found"
    else:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            xyz = tmp_path / "mol.xyz"
            write_xyz(base_mol, 0, xyz)
            cmd = ["xtb", str(xyz), f"--gfn{args.gfn}", "--opt", "tight",
                   "--alpb", "water"]
            res = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True)
            xtb_returncode = res.returncode
            opt_xyz = tmp_path / "xtbopt.xyz"
            if res.returncode == 0 and read_xyz_back(base_mol, opt_xyz):
                pass
            else:
                xtb_error_tail = res.stderr[-500:] or res.stdout[-500:] or "invalid_xtbopt_xyz"
                if not args.allow_mmff_fallback:
                    raise SystemExit(
                        "xtb optimization failed or produced no valid xtbopt.xyz; set "
                        "stage1.allow_xtb_fallback=true for an explicit MMFF degraded run"
                    )
                LOG.warning(
                    "xtb failed: %s — explicit MMFF fallback enabled",
                    xtb_error_tail[-200:],
                )
                xtb_status = "fallback_mmff"
        out_mol = base_mol

    _write_sdf_atomic(out_mol, args.out_sdf)

    identity = _identity_mol(out_mol)
    canonical = Chem.MolToSmiles(identity, canonical=True)
    inchikey = Chem.MolToInchiKey(identity)
    _write_text_atomic(inchikey + "\n", args.out_inchikey)
    _write_json_atomic({
        **input_provenance,
        "canonical_smiles": canonical,
        "inchikey": inchikey,
        "num_heavy_atoms": out_mol.GetNumHeavyAtoms(),
        "xtb_gfn": args.gfn,
        "xtb_status": xtb_status,
        "xtb_returncode": xtb_returncode,
        "xtb_error_tail": xtb_error_tail,
    }, args.out_meta)
    LOG.info("Done → %s  inchikey=%s", args.out_sdf, inchikey)


if __name__ == "__main__":
    main()
