#!/usr/bin/env python3
"""Shared Boltz-2 CLI helpers for pipeline wrappers."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

from rdkit import Chem


THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O",
}


def nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def pdb_sequence(pdb: Path) -> str:
    seq: list[str] = []
    last_key = None
    for line in pdb.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[12:16].strip() != "CA":
            continue
        key = (line[21], line[22:26].strip(), line[26].strip())
        if key == last_key:
            continue
        last_key = key
        seq.append(THREE_TO_ONE.get(line[17:20].strip(), "X"))
    return "".join(seq)


def ligand_smiles(ligand_sdf: Path) -> str:
    supplier = Chem.SDMolSupplier(str(ligand_sdf), removeHs=False)
    for mol in supplier:
        if mol is not None:
            return Chem.MolToSmiles(mol, canonical=True)
    text = ligand_sdf.read_text(errors="replace").strip()
    if text:
        return text.splitlines()[0].strip()
    raise ValueError(f"No ligand molecule or SMILES-like text found in {ligand_sdf}")


def _yaml_quote(value: str) -> str:
    return json.dumps(value)


def write_affinity_yaml(receptor_pdb: Path, ligand_sdf: Path, yaml_path: Path) -> None:
    seq = pdb_sequence(receptor_pdb)
    if not seq:
        raise ValueError(f"No CA sequence could be parsed from {receptor_pdb}")
    smiles = ligand_smiles(ligand_sdf)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(
        "version: 1\n"
        "sequences:\n"
        "  - protein:\n"
        "      id: A\n"
        f"      sequence: {_yaml_quote(seq)}\n"
        "  - ligand:\n"
        "      id: B\n"
        f"      smiles: {_yaml_quote(smiles)}\n"
        "properties:\n"
        "  - affinity:\n"
        "      binder: B\n"
    )


def run_boltz_predict(
    input_yaml: Path,
    out_dir: Path,
    *,
    diffusion_samples: int = 1,
    recycling_steps: int = 3,
    use_msa_server: bool = True,
    no_kernels: bool = True,
    accelerator: str = "gpu",
    output_format: str = "pdb",
) -> subprocess.CompletedProcess[str] | None:
    if not shutil.which("boltz"):
        return None
    if output_format not in {"pdb", "mmcif"}:
        raise ValueError(f"Unsupported Boltz output format: {output_format}")
    cmd = [
        "boltz",
        "predict",
        str(input_yaml),
        "--out_dir",
        str(out_dir),
        "--accelerator",
        accelerator,
        "--diffusion_samples",
        str(diffusion_samples),
        "--recycling_steps",
        str(recycling_steps),
        "--output_format",
        output_format,
        "--override",
    ]
    if use_msa_server:
        cmd.append("--use_msa_server")
    if no_kernels:
        cmd.append("--no_kernels")
    return subprocess.run(cmd, capture_output=True, text=True)


def top_ranked_pdb(out_dir: Path, input_id: str) -> Path | None:
    """Return Boltz's highest-confidence PDB for one prediction input.

    Boltz nests its results under ``boltz_results_<input stem>/`` inside the
    ``--out_dir`` it was given, so the predictions are one level deeper than the
    path this used to check. Looking only at the shallow path meant every real
    run produced a model file and was then reported as having produced nothing.
    The sibling loaders already use rglob, which is why they kept working.
    """
    expected = out_dir / "predictions" / input_id / f"{input_id}_model_0.pdb"
    if nonempty(expected):
        return expected
    matches = sorted(
        path
        for path in out_dir.rglob(f"{input_id}_model_0.pdb")
        if path.parent.name == input_id and path.parent.parent.name == "predictions"
    )
    for path in matches:
        if nonempty(path):
            return path
    return None


def _load_first_json(paths: list[Path]) -> dict | None:
    for path in paths:
        if not nonempty(path):
            continue
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def load_affinity_payload(out_dir: Path) -> dict | None:
    return _load_first_json(
        [
            *sorted(out_dir.rglob("affinity_*.json")),
            out_dir / "affinity.json",
        ]
    )


def load_confidence_payload(out_dir: Path) -> dict | None:
    return _load_first_json(
        [
            *sorted(out_dir.rglob("confidence_*.json")),
            out_dir / "report.json",
        ]
    )


def finite_float(value: object) -> float | None:
    if (
        isinstance(value, bool)
        or type(value).__name__ == "bool_"
        or (isinstance(value, str) and value.strip().lower() in {"true", "false"})
    ):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def affinity_score(payload: dict) -> float | None:
    probability = finite_float(payload.get("affinity_probability_binary"))
    if probability is not None:
        if 0.0 <= probability <= 1.0:
            return probability
        return None
    pred_value = finite_float(payload.get("affinity_pred_value"))
    if pred_value is not None:
        return -pred_value
    legacy = finite_float(payload.get("log_uM_affinity"))
    if legacy is not None:
        return -legacy
    return None
