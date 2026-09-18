#!/usr/bin/env python3
"""Shared Boltz-2 CLI helpers for pipeline wrappers."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

from rdkit import Chem

from residue_mapping import (
    THREE_TO_ONE,
    ResidueGapError,
    concatenated_pdb_sequence,
    pdb_residue_mapping,
    pdb_sequence,
)


__all__ = [
    "THREE_TO_ONE",
    "ReceptorLengthError",
    "ResidueGapError",
    "affinity_score",
    "concatenated_pdb_sequence",
    "crop_pdb_to_pocket",
    "finite_float",
    "ligand_smiles",
    "load_affinity_payload",
    "load_confidence_payload",
    "load_pocket_center",
    "nonempty",
    "pdb_residue_mapping",
    "pdb_sequence",
    "prepare_receptor_for_boltz",
    "run_boltz_predict",
    "top_ranked_pdb",
    "write_affinity_yaml",
]


def nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


class ReceptorLengthError(ValueError):
    """A receptor cannot be brought under the Boltz-2 length cap.

    Raised instead of silently truncating the sequence: dropping residues at
    the wrong end can remove the pocket, and the resulting prediction is then
    a different protein reported under the original target identity.
    """


def _finite_center(values) -> tuple[float, float, float] | None:
    try:
        center = tuple(float(value) for value in values)
    except (TypeError, ValueError):
        return None
    if len(center) != 3 or not all(math.isfinite(value) for value in center):
        return None
    return center


def load_pocket_center(path: Path | None) -> tuple[float, float, float] | None:
    """Read a pocket centre from a stage-4 box JSON or a docking-box text file.

    Stage 4 writes ``{"pockets": [{"center": [x, y, z], ...}]}``; the derived
    docking boxes write ``center_x = ...`` lines. Both are in the coordinate
    frame of the receptor handed to Boltz-2, which is what a crop needs.
    """
    if path is None or not nonempty(path):
        return None
    text = path.read_text(errors="replace")
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        pockets = payload.get("pockets")
        if isinstance(pockets, list):
            for pocket in pockets:
                if isinstance(pocket, dict):
                    center = _finite_center(pocket.get("center") or ())
                    if center is not None:
                        return center
        center = _finite_center(payload.get("center") or ())
        if center is not None:
            return center
        center = _finite_center(
            [payload.get("center_x"), payload.get("center_y"), payload.get("center_z")]
        )
        return center
    values: dict[str, str] = {}
    for line in stripped.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return _finite_center(
        [values.get("center_x"), values.get("center_y"), values.get("center_z")]
    )


def crop_pdb_to_pocket(
    receptor_pdb: Path,
    center: tuple[float, float, float],
    crop_radius: float,
    out_pdb: Path,
    mapping_tsv: Path,
) -> dict:
    """Keep residues with any atom within ``crop_radius`` of ``center``.

    The output PDB keeps the original chain/residue numbering, and
    ``mapping_tsv`` records which original residue each 1-based position of the
    Boltz input sequence corresponds to. Without that mapping a cropped model
    cannot be read back against the original target.
    """
    if crop_radius <= 0 or not math.isfinite(crop_radius):
        raise ReceptorLengthError(f"crop radius must be finite and > 0: {crop_radius}")
    order: list[tuple[str, str, str]] = []
    residues: dict[tuple[str, str, str], dict] = {}
    for line in receptor_pdb.read_text(errors="replace").splitlines():
        if not line.startswith("ATOM"):
            continue
        key = (line[21], line[22:26].strip(), line[26].strip())
        entry = residues.get(key)
        if entry is None:
            entry = {
                "lines": [],
                "coords": [],
                "ca": None,
                "resname": line[17:20].strip(),
            }
            residues[key] = entry
            order.append(key)
        entry["lines"].append(line)
        try:
            coord = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            continue
        entry["coords"].append(coord)
        if line[12:16].strip() == "CA":
            entry["ca"] = coord
    kept: list[tuple[tuple[str, str, str], dict, float]] = []
    for key in order:
        entry = residues[key]
        if not entry["coords"]:
            continue
        distance = min(math.dist(coord, center) for coord in entry["coords"])
        if distance <= crop_radius:
            kept.append((key, entry, distance))
    if not kept:
        raise ReceptorLengthError(
            "no residue lies within "
            f"{crop_radius:g} A of the pocket centre {center}; the pocket box "
            f"and receptor may be in different coordinate frames ({receptor_pdb})"
        )
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    mapping_tsv.parent.mkdir(parents=True, exist_ok=True)
    remark = (
        "REMARK 950 SKINSCOUT POCKET CROP radius="
        f"{crop_radius:g} center={center[0]:.3f},{center[1]:.3f},{center[2]:.3f}"
    )
    out_lines = [remark]
    mapping_rows: list[list[str]] = []
    crop_index = 0
    for key, entry, _distance in kept:
        out_lines.extend(entry["lines"])
        if entry["ca"] is not None:
            crop_index += 1
            mapping_rows.append(
                [str(crop_index), key[0], key[1], key[2], entry["resname"]]
            )
    out_lines.append("END")
    out_pdb.write_text("\n".join(out_lines) + "\n")
    header = [
        "crop_index",
        "original_chain",
        "original_resseq",
        "original_icode",
        "original_resname",
    ]
    mapping_tsv.write_text(
        "\n".join(
            ["\t".join(header)]
            + ["\t".join(row) for row in mapping_rows]
        )
        + "\n"
    )
    return {
        "input_residues": len(order),
        "kept_residues": len(kept),
        "sequence_positions": crop_index,
        "crop_map": str(mapping_tsv),
    }


def prepare_receptor_for_boltz(
    receptor_pdb: Path,
    *,
    max_residues: int,
    crop_radius: float,
    out_dir: Path,
    label: str,
    pocket_path: Path | None = None,
) -> tuple[Path, dict]:
    """Return the receptor Boltz-2 should see, cropping around a known pocket.

    A receptor at or under the cap is returned unchanged. A longer receptor is
    cropped only when a pocket box is available; otherwise this raises
    :class:`ReceptorLengthError` so the stage stops with an explanatory error
    instead of silently running past the memory cap or dropping the pocket.
    """
    # The crop below deliberately keeps only pocket residues, so the result is
    # discontinuous by construction and carries `receptor_crop_map.tsv` as its
    # explicit residue mapping. Concatenation is the point here; the strict
    # `pdb_sequence` is reserved for structures that claim to be continuous.
    sequence = concatenated_pdb_sequence(receptor_pdb)
    if not sequence:
        return receptor_pdb, {
            "crop_mode": "sequence_unavailable",
            "input_residues": 0,
            "model_residues": 0,
            "crop_map": "",
        }
    if len(sequence) <= max_residues:
        return receptor_pdb, {
            "crop_mode": "not_needed",
            "input_residues": len(sequence),
            "model_residues": len(sequence),
            "crop_map": "",
        }
    center = load_pocket_center(pocket_path)
    if center is None:
        raise ReceptorLengthError(
            f"{label}: receptor {receptor_pdb.name} has {len(sequence)} residues "
            f"> --max-residues {max_residues} and no pocket box is available for "
            "a pocket-centred crop; provide a pre-cropped receptor or raise "
            "--max-residues"
        )
    cropped = out_dir / "receptor_cropped.pdb"
    mapping = out_dir / "receptor_crop_map.tsv"
    info = crop_pdb_to_pocket(receptor_pdb, center, crop_radius, cropped, mapping)
    model_residues = len(concatenated_pdb_sequence(cropped))
    if model_residues < 1:
        raise ReceptorLengthError(
            f"{label}: pocket-centred crop produced no parseable sequence from "
            f"{receptor_pdb}"
        )
    if model_residues > max_residues:
        raise ReceptorLengthError(
            f"{label}: pocket-centred crop at {crop_radius:g} A still leaves "
            f"{model_residues} residues > --max-residues {max_residues}; lower "
            "--crop-radius or provide a pre-cropped receptor"
        )
    return cropped, {
        "crop_mode": "pocket_crop",
        "input_residues": len(sequence),
        "model_residues": model_residues,
        "crop_map": str(mapping),
        "crop_radius": crop_radius,
        "pocket_center": list(center),
        "kept_residues": info["kept_residues"],
    }


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


def write_affinity_yaml(
    receptor_pdb: Path,
    ligand_sdf: Path,
    yaml_path: Path,
    *,
    crop_map: Path | None = None,
) -> None:
    """Write the Boltz-2 input YAML for one receptor.

    A continuous receptor sequence is required. ``crop_map`` is the explicit
    opt-in for a pocket crop: the crop deliberately keeps a non-contiguous
    residue subset and records each kept residue in the crop map, so the
    concatenation is mapped rather than invented.
    """
    if crop_map is not None and not nonempty(crop_map):
        raise ValueError(f"crop map is missing or empty: {crop_map}")
    seq = (
        concatenated_pdb_sequence(receptor_pdb)
        if crop_map is not None
        else pdb_sequence(receptor_pdb)
    )
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
