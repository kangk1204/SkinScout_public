#!/usr/bin/env python3
"""Run one real Meeko -> AutoGrid -> AutoDock-GPU -> GNINA qualification."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, TextIO

from rdkit import Chem
from rdkit.Chem import AllChem


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_CONFIG = ROOT / "workflow" / "config.yaml"
SCHEMA_VERSION = "skinscout.structural-qualification.v1"
AUTOGRID_VERSION = "4.2.8+6d2847b"


class SmokeError(RuntimeError):
    """A structural qualification step failed."""


def docking_box_dir() -> Path:
    """Resolve the docking box directory from the workflow config.

    Hardcoding it here let the qualification run against a different box set
    than the workflow, which is exactly how a stale uniform-cube directory
    would survive a config change.
    """
    import yaml

    try:
        config = yaml.safe_load(WORKFLOW_CONFIG.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SmokeError(f"workflow config could not be read: {WORKFLOW_CONFIG}: {exc}") from exc
    try:
        configured = config["paths"]["docking_boxes"]
    except (KeyError, TypeError) as exc:
        raise SmokeError(
            f"workflow config has no paths.docking_boxes: {WORKFLOW_CONFIG}"
        ) from exc
    if not str(configured).strip():
        raise SmokeError(f"paths.docking_boxes is blank: {WORKFLOW_CONFIG}")
    path = Path(str(configured).strip())
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _run(
    command: list[str],
    *,
    log_handle: TextIO,
    env: dict[str, str],
) -> None:
    log_handle.write("$ " + shlex.join(command) + "\n")
    log_handle.flush()
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.stdout:
        log_handle.write(result.stdout)
        if not result.stdout.endswith("\n"):
            log_handle.write("\n")
    if result.stderr:
        log_handle.write(result.stderr)
        if not result.stderr.endswith("\n"):
            log_handle.write("\n")
    log_handle.write(f"[exit={result.returncode}]\n")
    log_handle.flush()
    if result.returncode != 0:
        raise SmokeError(
            f"qualification command exited {result.returncode}: {shlex.join(command)}"
        )


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SmokeError(f"{label} is unreadable: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SmokeError(f"{label} must contain a JSON object: {path}")
    return payload


def _box_center(path: Path) -> tuple[float, float, float]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    try:
        return tuple(float(values[f"center_{axis}"]) for axis in "xyz")  # type: ignore[return-value]
    except (KeyError, ValueError) as exc:
        raise SmokeError(f"invalid docking-box center: {path}") from exc


def _selected_target_ids(path: Path) -> list[str]:
    if not _nonempty(path):
        raise SmokeError(f"Daina qualification selection is unavailable: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "target_id" not in reader.fieldnames:
            raise SmokeError("Daina qualification selection lacks target_id")
        target_ids = [
            str(row.get("target_id") or "").strip()
            for row in reader
            if str(row.get("target_id") or "").strip()
        ]
    if len(target_ids) != 256 or len(set(target_ids)) != 256:
        raise SmokeError(
            "Daina qualification must produce exactly 256 unique ranked targets"
        )
    return target_ids


def _select_target(selected_path: Path) -> str:
    receptor_dir = ROOT / "data/human_pdbqt"
    box_dir = docking_box_dir()
    clean_dir = ROOT / "data/human_clean"
    no_pocket_path = ROOT / "data/no_pocket_targets.list"
    no_pocket = (
        {line.strip() for line in no_pocket_path.read_text().splitlines() if line.strip()}
        if no_pocket_path.exists()
        else set()
    )
    for target_id in _selected_target_ids(selected_path):
        receptor = receptor_dir / f"{target_id}.pdbqt"
        if target_id in no_pocket or not _nonempty(receptor):
            continue
        if _nonempty(box_dir / f"{target_id}.box.txt") and _nonempty(
            clean_dir / f"{target_id}_clean.pdb"
        ):
            return target_id
    raise SmokeError(
        "Daina top 256 has no target with receptor PDBQT, docking box, and clean structure"
    )


def _write_ligand(path: Path, center: tuple[float, float, float]) -> None:
    molecule = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    if AllChem.EmbedMolecule(molecule, randomSeed=0x5C017) != 0:
        raise SmokeError("RDKit could not embed the qualification ligand")
    AllChem.UFFOptimizeMolecule(molecule, maxIters=200)
    conformer = molecule.GetConformer()
    centroid = [
        sum(conformer.GetAtomPosition(index)[axis] for index in range(molecule.GetNumAtoms()))
        / molecule.GetNumAtoms()
        for axis in range(3)
    ]
    for index in range(molecule.GetNumAtoms()):
        position = conformer.GetAtomPosition(index)
        conformer.SetAtomPosition(
            index,
            tuple(position[axis] - centroid[axis] + center[axis] for axis in range(3)),
        )
    molecule.SetProp("_Name", "SkinScout qualification ethanol")
    writer = Chem.SDWriter(str(path))
    try:
        writer.write(molecule)
    finally:
        writer.close()
    if not _nonempty(path):
        raise SmokeError(f"qualification ligand was not written: {path}")


def _single_target_record(payload: dict[str, Any], label: str) -> dict[str, Any]:
    records = payload.get("targets")
    if not isinstance(records, list) or len(records) != 1 or not isinstance(records[0], dict):
        raise SmokeError(f"{label} must contain exactly one target record")
    return records[0]


def _write_report_inputs(run_dir: Path, target_id: str, gnina_score: float) -> None:
    canonical = run_dir / "03_targets/mode_fast/daina_structural_targets.csv"
    with canonical.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "target_id",
                "final_score",
                "daina_rank",
                "daina_score",
                "daina_score_is_probability",
                "skin_score",
                "skin_tier",
                "source_count",
                "sources",
                "structural_status",
                "gnina_score",
                "efficacy_top1",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "target_id": target_id,
                "final_score": "1.0",
                "daina_rank": "1",
                "daina_score": "1.0",
                "daina_score_is_probability": "false",
                "skin_score": "0.0",
                "skin_tier": "not_evaluated",
                "source_count": "3",
                "sources": "daina;autodock_gpu;gnina",
                "structural_status": "structure_supported",
                "gnina_score": f"{gnina_score:.4f}",
                "efficacy_top1": "qualification fixture only",
            }
        )
    shutil.copy2(canonical, canonical.parent / "top50.csv")


def run_smoke(out_dir: Path, daina_python: Path, chembl_fp: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / "run"
    fast_dir = run_dir / "03_targets/mode_fast"
    fast_dir.mkdir(parents=True, exist_ok=True)
    if not _nonempty(daina_python) or not os.access(daina_python, os.X_OK):
        raise SmokeError(f"Daina qualification Python is unavailable: {daina_python}")
    if not _nonempty(chembl_fp):
        raise SmokeError(f"Stage 0 ChEMBL fingerprints are unavailable: {chembl_fp}")
    ligand_sdf = run_dir / "01_input/qualification_ligand.sdf"
    ligand_sdf.parent.mkdir(parents=True, exist_ok=True)
    _write_ligand(ligand_sdf, (0.0, 0.0, 0.0))

    daina_scores = fast_dir / "daina_zoete_proteome.tsv"
    daina_metadata = fast_dir / "daina_zoete_proteome.metadata.json"
    selected = fast_dir / "daina_top256.csv"
    daina_compat = fast_dir / "dti_rrf_top25pct.csv"
    no_pocket = out_dir / "no_pocket.list"
    no_pocket.write_text("", encoding="utf-8")
    ligand_pdbqt = run_dir / "03_targets/mode_comprehensive/ligand.pdbqt"
    ligand_pdbqt.parent.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    smoke_bin = out_dir / "bin"
    smoke_bin.mkdir(exist_ok=True)
    gnina = ROOT / "tools/gnina"
    if not _nonempty(gnina) or not os.access(gnina, os.X_OK):
        raise SmokeError(f"pinned GNINA executable is unavailable: {gnina}")
    gnina_link = smoke_bin / "gnina"
    if gnina_link.exists() or gnina_link.is_symlink():
        gnina_link.unlink()
    gnina_link.symlink_to(gnina)
    environment["PATH"] = f"{smoke_bin}:{Path.home() / '.local/bin'}:{environment.get('PATH', '')}"
    environment["AUTODOCK_GPU_BIN"] = str(
        ROOT / "tools/AutoDock-GPU/bin/autodock_gpu_128wi"
    )

    prepare_ligand = shutil.which("mk_prepare_ligand.py")
    if not prepare_ligand:
        raise SmokeError("Meeko mk_prepare_ligand.py is unavailable in the qualification environment")

    command_log = out_dir / "commands.log"
    with command_log.open("w", encoding="utf-8") as log_handle:
        _run(
            [
                str(daina_python),
                str(ROOT / "scripts/stage3_daina_zoete.py"),
                "--ligand-sdf",
                str(ligand_sdf),
                "--chembl-fp",
                str(chembl_fp),
                "--evidence-mode",
                "retrieval",
                "--quality-policy",
                "legacy",
                "--scoring-method",
                "max-similarity",
                "--out-scores",
                str(daina_scores),
                "--out-metadata-json",
                str(daina_metadata),
            ],
            log_handle=log_handle,
            env=environment,
        )
        _run(
            [
                str(daina_python),
                str(ROOT / "scripts/stage3_select_daina.py"),
                "--daina-scores",
                str(daina_scores),
                "--daina-metadata",
                str(daina_metadata),
                "--top-n",
                "256",
                "--out-csv",
                str(selected),
                "--out-compat-csv",
                str(daina_compat),
            ],
            log_handle=log_handle,
            env=environment,
        )
        target_id = _select_target(selected)
        box_path = docking_box_dir() / f"{target_id}.box.txt"
        _write_ligand(ligand_sdf, _box_center(box_path))
        _run(
            [prepare_ligand, "-i", str(ligand_sdf), "-o", str(ligand_pdbqt)],
            log_handle=log_handle,
            env=environment,
        )
        map_manifest = fast_dir / "autogrid_map_manifest.json"
        _run(
            [
                sys.executable,
                str(ROOT / "scripts/stage3_autogrid_maps.py"),
                "--selected-csv",
                str(selected),
                "--ligand-pdbqt",
                str(ligand_pdbqt),
                "--receptor-dir",
                str(ROOT / "data/human_pdbqt"),
                "--box-dir",
                str(docking_box_dir()),
                "--no-pocket-list",
                str(no_pocket),
                "--autogrid-bin",
                str(ROOT / "tools/AutoGrid/autogrid4"),
                "--parameter-file",
                str(ROOT / "tools/AutoDock-GPU/AD4_parameters.dat"),
                "--autogrid-version",
                AUTOGRID_VERSION,
                "--cache-dir",
                str(out_dir / "autogrid_cache"),
                "--out-map-dir",
                str(fast_dir / "autogrid_maps"),
                "--out-manifest",
                str(map_manifest),
            ],
            log_handle=log_handle,
            env=environment,
        )
        map_record = _single_target_record(_load_json(map_manifest, "AutoGrid manifest"), "AutoGrid manifest")
        if map_record.get("status") != "map_ready":
            raise SmokeError(f"AutoGrid smoke did not produce maps: {map_record}")

        docking_scores = fast_dir / "autodock_top5k.tsv"
        docking_status = fast_dir / "autodock_status_manifest.json"
        pose_dir = fast_dir / "docked_poses"
        _run(
            [
                sys.executable,
                str(ROOT / "scripts/stage3_autodock_run.py"),
                "--ligand-pdbqt",
                str(ligand_pdbqt),
                "--receptor-dir",
                str(ROOT / "data/human_pdbqt"),
                "--box-dir",
                str(docking_box_dir()),
                "--map-manifest",
                str(map_manifest),
                "--engine",
                "autodock_gpu",
                "--nrun",
                "1",
                "--progress-every",
                "1",
                "--out-scores",
                str(docking_scores),
                "--out-status-manifest",
                str(docking_status),
                "--out-pose-dir",
                str(pose_dir),
            ],
            log_handle=log_handle,
            env=environment,
        )
        docking_record = _single_target_record(
            _load_json(docking_status, "AutoDock status manifest"),
            "AutoDock status manifest",
        )
        if docking_record.get("status") != "docked":
            raise SmokeError(f"AutoDock-GPU smoke did not dock a pose: {docking_record}")

        gnina_scores = fast_dir / "gnina_pose_rescores.tsv"
        gnina_status = fast_dir / "gnina_status_manifest.json"
        _run(
            [
                sys.executable,
                str(ROOT / "scripts/stage3_gnina_rescore.py"),
                "--top-csv",
                str(docking_scores),
                "--pose-manifest",
                str(pose_dir / "pose_manifest.json"),
                "--pose-dir",
                str(pose_dir),
                "--clean-dir",
                str(ROOT / "data/human_clean"),
                "--use-gpu",
                "--out-scores",
                str(gnina_scores),
                "--out-status-manifest",
                str(gnina_status),
            ],
            log_handle=log_handle,
            env=environment,
        )

    gnina_payload = _load_json(gnina_status, "GNINA status manifest")
    gnina_record = _single_target_record(gnina_payload, "GNINA status manifest")
    if gnina_payload.get("gpu_enabled") is not True or gnina_record.get("status") != "structure_supported":
        raise SmokeError(f"GNINA CUDA pose smoke did not pass: {gnina_payload}")
    with gnina_scores.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 1 or rows[0].get("scored_actual_docked_pose") != "true" or rows[0].get("gpu_enabled") != "true":
        raise SmokeError("GNINA score output does not prove GPU scoring of the exported pose")
    try:
        gnina_value = float(rows[0]["cnn_affinity"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SmokeError("GNINA score output lacks a numeric CNN affinity") from exc
    _write_report_inputs(run_dir, target_id, gnina_value)

    outputs = {
        "daina_scores": daina_scores,
        "daina_metadata": daina_metadata,
        "daina_selected": selected,
        "daina_compatibility": daina_compat,
        "map_manifest": fast_dir / "autogrid_map_manifest.json",
        "docking_scores": docking_scores,
        "docking_status": docking_status,
        "pose_manifest": pose_dir / "pose_manifest.json",
        "gnina_scores": gnina_scores,
        "gnina_status": gnina_status,
        "canonical_targets": fast_dir / "daina_structural_targets.csv",
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "completed_at_epoch": int(time.time()),
        "target_id": target_id,
        "ligand": "ethanol",
        "daina_target_count": 256,
        "daina_evidence_mode": "retrieval",
        "autogrid_version": AUTOGRID_VERSION,
        "gpu_checks": {
            "autodock_gpu_opencl_pose": "passed",
            "gnina_cuda_actual_pose": "passed",
        },
        "outputs": {
            name: {
                "path": str(path.relative_to(out_dir)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for name, path in outputs.items()
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--daina-python", required=True, type=Path)
    parser.add_argument("--chembl-fp", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / "qualification_status.json"
    _write_json(status_path, {"schema_version": SCHEMA_VERSION, "status": "running"})
    try:
        payload = run_smoke(
            out_dir,
            args.daina_python.expanduser().resolve(),
            args.chembl_fp.expanduser().resolve(),
        )
    except Exception as exc:
        _write_json(
            status_path,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "failed",
                "detail": str(exc),
            },
        )
        raise
    _write_json(status_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
