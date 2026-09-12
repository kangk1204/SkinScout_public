#!/usr/bin/env python3
"""stage3_autodock_run.py — Run AutoDock-GPU over the receptor proteome.

By default docks every PDBQT in --receptor-dir minus the no-pocket set.
With --restrict-list, only dock receptors named in that file (used by MODE-FAST
to dock just the top-25 % from the DTI RRF stage).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger("stage3.autodock")
LOCAL_AUTODOCK_CANDIDATES = (
    ROOT / "tools/autodock_gpu/bin/autodock_gpu_128wi",
    ROOT / "tools/autodock_gpu/autodock_gpu_128wi",
    ROOT / "tools/AutoDock-GPU/bin/autodock_gpu_128wi",
)
BOX_CENTER_FIELDS = ("center_x", "center_y", "center_z")
BOX_SIZE_FIELDS = ("size_x", "size_y", "size_z")
BOX_REQUIRED_FIELDS = (*BOX_CENTER_FIELDS, *BOX_SIZE_FIELDS)
SAFE_TARGET_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
POSE_MANIFEST_NAME = "pose_manifest.json"
POSE_MANIFEST_SCHEMA_VERSION = "skinscout.docking_pose_manifest.v1"
MAP_MANIFEST_SCHEMA_VERSION = "skinscout.autogrid-map-manifest.v1"
DOCKING_STATUS_SCHEMA_VERSION = "skinscout.docking-status-manifest.v1"
POSE_PDBQT_RECORDS = (
    "MODEL",
    "REMARK",
    "ROOT",
    "ENDROOT",
    "BRANCH",
    "ENDBRANCH",
    "ATOM",
    "HETATM",
    "TORSDOF",
    "ENDMDL",
)


def _remove_outputs(*paths: Path) -> None:
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def _nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _tmp_output(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_suffix(path.suffix + ".tmp")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    directories = [root]
    for path in sorted(root.rglob("*")):
        if path.is_file():
            _fsync_file(path)
        elif path.is_dir():
            directories.append(path)
    for directory in reversed(directories):
        _fsync_directory(directory)


def _write_json_atomic_durable(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _safe_target_id(target_id: str) -> str:
    if (
        not target_id
        or not SAFE_TARGET_ID_RE.fullmatch(target_id)
        or target_id in {".", ".."}
        or ".." in target_id.split(".")
    ):
        raise SystemExit(
            "Unsafe docking target_id cannot be used as pose filename: "
            f"{target_id!r}"
        )
    return target_id


def _pose_path(pose_dir: Path, target_id: str) -> Path:
    return pose_dir / f"{_safe_target_id(target_id)}.sdf"


def _import_rdkit_chem():
    try:
        from rdkit import Chem  # type: ignore[import-untyped]
    except ImportError as exc:
        raise SystemExit(
            "RDKit is required to export and validate docking poses"
        ) from exc
    return Chem


def _validate_sdf_pose(path: Path, target_id: str) -> None:
    if not _nonempty(path):
        raise SystemExit(
            f"Docking pose SDF is missing or empty for {target_id}: {path}"
        )
    Chem = _import_rdkit_chem()
    try:
        supplier = Chem.SDMolSupplier(str(path), sanitize=True, removeHs=False)
        mols = [mol for mol in supplier if mol is not None]
    except Exception as exc:
        raise SystemExit(
            f"Docking pose SDF is not parseable for {target_id}: {path}"
        ) from exc
    if not mols:
        raise SystemExit(
            f"Docking pose SDF contains no parseable molecule for {target_id}: "
            f"{path}"
        )


def _write_sdf_from_pdbqt_pose(
    pdbqt_pose: Path,
    sdf_pose: Path,
    target_id: str,
    energy: float,
) -> None:
    try:
        from meeko import PDBQTMolecule, RDKitMolCreate  # type: ignore[import-untyped]
    except ImportError as exc:
        raise SystemExit(
            "Meeko is required to convert docked PDBQT poses to SDF"
        ) from exc
    Chem = _import_rdkit_chem()
    try:
        try:
            pdbqt_mol = PDBQTMolecule.from_file(str(pdbqt_pose), skip_typing=True)
        except TypeError:
            pdbqt_mol = PDBQTMolecule.from_file(str(pdbqt_pose))
        rdkit_mols = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
        if not rdkit_mols:
            raise ValueError("Meeko returned no RDKit molecules")
        mol = next((candidate for candidate in rdkit_mols if candidate is not None), None)
        if mol is None:
            raise ValueError("Meeko returned no reconstructable RDKit molecule")
        if hasattr(mol, "SetProp"):
            mol.SetProp("_Name", target_id)
            mol.SetProp("target_id", target_id)
            mol.SetProp("docking_energy_kcal_mol", f"{energy:.3f}")
        sdf_pose.parent.mkdir(parents=True, exist_ok=True)
        writer = Chem.SDWriter(str(sdf_pose))
        try:
            writer.write(mol)
        finally:
            writer.close()
    except Exception as exc:
        raise SystemExit(
            f"Failed to export docked pose SDF for {target_id}: {sdf_pose}"
        ) from exc
    _validate_sdf_pose(sdf_pose, target_id)


def autodock_gpu_binary() -> str | None:
    env_path = os.environ.get("AUTODOCK_GPU_BIN")
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path).expanduser())
    if os.environ.get("SKINSCOUT_DISABLE_LOCAL_TOOL_PATHS") != "1":
        candidates.extend(LOCAL_AUTODOCK_CANDIDATES)
    path = shutil.which("autodock_gpu_128wi")
    if path:
        return path
    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def autodock_map_for_receptor(receptor: Path) -> Path | None:
    candidates = (
        receptor.with_suffix(".maps.fld"),
        receptor.with_suffix(".fld"),
        receptor.parent / f"{receptor.stem}.maps.fld",
    )
    for candidate in candidates:
        if _nonempty(candidate):
            return candidate
    return None


def _read_map_manifest(
    path: Path,
) -> tuple[list[dict[str, object]], dict[str, Path]]:
    if not _nonempty(path):
        raise SystemExit(f"AutoGrid map manifest is missing or empty: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"AutoGrid map manifest is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"AutoGrid map manifest must contain an object: {path}")
    if payload.get("schema_version") != MAP_MANIFEST_SCHEMA_VERSION:
        raise SystemExit(
            "AutoGrid map manifest has unsupported schema_version: "
            f"{payload.get('schema_version')!r}"
        )
    records = payload.get("targets")
    if not isinstance(records, list) or not records:
        raise SystemExit("AutoGrid map manifest must contain non-empty targets")
    if payload.get("target_count") != len(records):
        raise SystemExit("AutoGrid map manifest target_count does not match targets")
    normalized: list[dict[str, object]] = []
    ready: dict[str, Path] = {}
    seen: set[str] = set()
    manifest_root = path.parent.resolve()
    for index, value in enumerate(records):
        if not isinstance(value, dict):
            raise SystemExit(
                f"AutoGrid map manifest target record {index} must be an object"
            )
        target_id = _safe_target_id(str(value.get("target_id", "")))
        if target_id in seen:
            raise SystemExit(
                f"AutoGrid map manifest contains duplicate target_id: {target_id}"
            )
        seen.add(target_id)
        status = str(value.get("status", "")).strip()
        if not status:
            raise SystemExit(
                f"AutoGrid map manifest status is blank for {target_id}"
            )
        record = dict(value)
        record["target_id"] = target_id
        normalized.append(record)
        if status != "map_ready":
            continue
        raw_map = str(value.get("map_fld", "")).strip()
        if not raw_map:
            raise SystemExit(
                f"AutoGrid map manifest map_ready record lacks map_fld: {target_id}"
            )
        candidate = Path(raw_map)
        map_path = (manifest_root / candidate).resolve()
        try:
            map_path.relative_to(manifest_root)
        except ValueError as exc:
            raise SystemExit(
                f"AutoGrid map path escapes the run directory for {target_id}: {raw_map}"
            ) from exc
        if not _nonempty(map_path):
            raise SystemExit(
                f"AutoGrid map manifest references missing/empty map for {target_id}: "
                f"{map_path}"
            )
        expected_sha = str(value.get("map_fld_sha256", "")).strip()
        if expected_sha and _sha256_file(map_path) != expected_sha:
            raise SystemExit(
                f"AutoGrid map manifest SHA-256 mismatch for {target_id}: {map_path}"
            )
        ready[target_id] = map_path
    return normalized, ready


def receptor_list(
    receptor_dir: Path,
    no_pocket: set[str],
    restrict: list[str] | None,
    max_receptors: int = 0,
) -> list[Path]:
    receptor_by_uid = {
        pdbqt.stem: pdbqt
        for pdbqt in sorted(receptor_dir.glob("*.pdbqt"))
        if _nonempty(pdbqt)
    }
    target_ids = list(receptor_by_uid) if restrict is None else restrict
    eligible_ids = [uid for uid in target_ids if uid not in no_pocket]
    if max_receptors > 0:
        eligible_ids = eligible_ids[:max_receptors]
    missing = [uid for uid in eligible_ids if uid not in receptor_by_uid]
    if missing:
        shown = ", ".join(missing[:10])
        suffix = "..." if len(missing) > 10 else ""
        raise SystemExit(
            "Selected docking target(s) are missing non-empty receptor PDBQT files: "
            f"{shown}{suffix}"
        )
    return [receptor_by_uid[uid] for uid in eligible_ids]


def _unique_ordered(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        uid = value.strip()
        if not uid or uid in seen:
            continue
        out.append(uid)
        seen.add(uid)
    return out


def read_uniprots(path: Path) -> list[str]:
    if not path.exists():
        return []
    if path.suffix.lower() == ".csv":
        with path.open() as fh:
            reader = csv.DictReader(fh)
            return _unique_ordered([r.get("target_id", "") for r in reader])
    return _unique_ordered(path.read_text().splitlines())


def parse_box(
    box_path: Path,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if not _nonempty(box_path):
        raise SystemExit(f"Docking box file is missing or empty: {box_path}")
    fields: dict[str, str] = {}
    for line in box_path.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            fields[k.strip()] = v.strip()
    missing = [key for key in BOX_REQUIRED_FIELDS if key not in fields]
    if missing:
        shown = ", ".join(missing)
        raise SystemExit(
            f"Docking box file missing required field(s) {shown}: {box_path}"
        )
    center = (
        _checked_box_field(fields, "center_x", box_path),
        _checked_box_field(fields, "center_y", box_path),
        _checked_box_field(fields, "center_z", box_path),
    )
    size = (
        _checked_box_field(fields, "size_x", box_path),
        _checked_box_field(fields, "size_y", box_path),
        _checked_box_field(fields, "size_z", box_path),
    )
    for key, value in zip(BOX_SIZE_FIELDS, size, strict=True):
        if value <= 0.0:
            raise SystemExit(
                f"Docking box field '{key}' must be > 0: {box_path}"
            )
    return (center, size)


def _checked_box_field(fields: dict[str, str], key: str, box_path: Path) -> float:
    raw = fields[key]
    try:
        value = float(raw)
    except ValueError as exc:
        raise SystemExit(
            f"Docking box field '{key}' must be numeric: {box_path}"
        ) from exc
    if not math.isfinite(value):
        raise SystemExit(
            f"Docking box field '{key}' must be finite: {box_path}"
        )
    return value


# AutoDock Vina warns above this search volume that the default exhaustiveness
# is insufficient (27000 A^3 is a 30 A cube).
DEFAULT_SEARCH_VOLUME_ADVISORY = 27000.0
# Measured on four derived boxes of 1.5-2.1x the advisory, three ligands and six
# seeds each: effort 4, 8 and 16 all left a clashing pose and a seed-to-seed
# spread near 0.61 kcal/mol, while 32 removed the clash and cut the spread to
# 0.030. The convergence is a cliff at 32, not a gradual slope, so a plain
# multiple of the configured effort is not enough for a low fast-mode base.
DEFAULT_HIGH_VOLUME_MIN_EFFORT = 32


def box_volume(size: tuple[float, float, float]) -> float:
    return float(size[0]) * float(size[1]) * float(size[2])


def sampling_effort(
    base: int,
    volume: float,
    *,
    advisory: float = DEFAULT_SEARCH_VOLUME_ADVISORY,
    high_volume_floor: int = DEFAULT_HIGH_VOLUME_MIN_EFFORT,
) -> int:
    """Search effort for one receptor, scaled by its box volume.

    Boxes are no longer a uniform cube, so a single configured effort now means
    very different sampling densities across receptors. Under-sampling a large
    box does not merely add noise: it returns clashing positive-energy poses.
    """
    if base < 1:
        raise SystemExit(f"docking search effort must be >= 1: {base}")
    if not math.isfinite(advisory) or advisory <= 0.0:
        raise SystemExit(f"search volume advisory must be positive: {advisory!r}")
    if high_volume_floor < 1:
        raise SystemExit(
            f"high-volume search effort floor must be >= 1: {high_volume_floor}"
        )
    if not math.isfinite(volume) or volume <= 0.0:
        raise SystemExit(f"docking box volume must be positive and finite: {volume!r}")
    if volume <= advisory:
        return base
    scaled = int(math.ceil(base * volume / advisory))
    return max(scaled, high_volume_floor)


def parse_dlg_best(dlg_path: Path) -> float | None:
    """Pull the lowest cluster binding-energy from AutoDock-GPU's .dlg output."""
    parsed = parse_dlg_best_pose(dlg_path)
    if parsed is not None:
        return parsed[0]
    best = None
    if not _nonempty(dlg_path):
        return None
    for line in dlg_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("DOCKED: USER    Estimated Free Energy of Binding"):
            try:
                val = float(line.split("=")[1].split()[0])
                if not math.isfinite(val):
                    raise SystemExit(_nonfinite_dlg_message(dlg_path))
                best = val if best is None else min(best, val)
            except (IndexError, ValueError):
                continue
        elif line.startswith("Run") and "kcal/mol" in line:
            try:
                val = float(line.split()[-2])
                if not math.isfinite(val):
                    raise SystemExit(_nonfinite_dlg_message(dlg_path))
                best = val if best is None else min(best, val)
            except (IndexError, ValueError):
                continue
    return best


def parse_dlg_best_pose(dlg_path: Path) -> tuple[float, str] | None:
    """Return the lowest-energy docked PDBQT pose embedded in an AutoDock DLG."""
    if not _nonempty(dlg_path):
        return None

    best: tuple[float, list[str]] | None = None
    current_energy: float | None = None
    current_lines: list[str] = []
    in_model = False

    def finish_pose() -> None:
        nonlocal best, current_energy, current_lines
        pose_records = [
            line for line in current_lines
            if line.startswith(("ATOM", "HETATM", "ROOT", "BRANCH"))
        ]
        if current_energy is None or not pose_records:
            return
        if best is None or current_energy < best[0]:
            best = (current_energy, list(current_lines))

    for raw_line in dlg_path.read_text().splitlines():
        line = raw_line.strip()
        if line.startswith("DOCKED: USER    Estimated Free Energy of Binding"):
            try:
                val = float(line.split("=")[1].split()[0])
            except (IndexError, ValueError):
                continue
            if not math.isfinite(val):
                raise SystemExit(_nonfinite_dlg_message(dlg_path))
            current_energy = val
            continue
        if not line.startswith("DOCKED:"):
            continue
        docked = line.removeprefix("DOCKED:").strip()
        if docked.startswith("MODEL"):
            finish_pose()
            current_energy = None
            current_lines = [docked]
            in_model = True
            continue
        if not docked:
            continue
        if docked.startswith(POSE_PDBQT_RECORDS):
            if not in_model and not current_lines:
                in_model = True
            if in_model:
                current_lines.append(docked)
            if docked.startswith("ENDMDL"):
                finish_pose()
                current_energy = None
                current_lines = []
                in_model = False

    finish_pose()
    if best is None:
        return None
    energy, pose_lines = best
    header = [
        f"REMARK SKINSCOUT_TARGET_DOCKING_ENERGY {energy:.3f}",
        f"REMARK SKINSCOUT_SOURCE_DLG {dlg_path.name}",
    ]
    return energy, "\n".join([*header, *pose_lines, ""])


# AutoDock-GPU v1.6 accepts only these -lsmet tokens. The human-readable
# method names used in configuration are aliased to them here; anything else is
# rejected loudly. Passing an unaccepted token makes the binary abort during job
# setup while still exiting 0, so an unvalidated value fails silently.
AUTODOCK_LS_METHODS = ("sw", "sd", "fire", "ad", "adam")
AUTODOCK_LS_ALIASES = {
    "adadelta": "ad",
    "solis-wets": "sw",
    "solis_wets": "sw",
    "steepest-descent": "sd",
    "steepest_descent": "sd",
}


def canonical_ls_method(value: str) -> str:
    token = str(value).strip().lower()
    token = AUTODOCK_LS_ALIASES.get(token, token)
    if token not in AUTODOCK_LS_METHODS:
        raise SystemExit(
            f"Unsupported AutoDock-GPU local search method: {value!r}. "
            "Expected one of " + ", ".join(AUTODOCK_LS_METHODS)
        )
    return token


def run_autodock_gpu_one(
    ligand: Path,
    map_file: Path,
    nrun: int,
    ls_method: str,
    workdir: Path,
    autodock_bin: str,
    pose_sdf: Path | None = None,
    target_id: str | None = None,
) -> float | None:
    if not _nonempty(ligand) or not _nonempty(map_file):
        return None
    workdir.mkdir(parents=True, exist_ok=True)
    # The subprocess runs from the map bundle so the field file can reach its
    # sibling maps, which leaves any relative caller path unresolvable:
    # AutoDock-GPU reported "Can't open ligand data file" for every receptor.
    ligand = ligand.resolve()
    res_prefix = (workdir / map_file.stem.replace(".maps", "")).resolve()
    cmd = [
        autodock_bin,
        "-L", str(ligand),
        "-M", str(map_file),
        "--xmloutput", "0",
        "-nrun", str(nrun),
        "-lsmet", canonical_ls_method(ls_method),
        "--resnam", str(res_prefix),
    ]
    # AutoGrid field files commonly reference sibling maps by relative path.
    # Running from the map bundle keeps that contract intact while all result
    # outputs remain absolute under the temporary work directory.
    res = subprocess.run(
        cmd,
        cwd=map_file.parent,
        capture_output=True,
        text=True,
    )
    # AutoDock-GPU exits 0 even when a job aborts during setup, so the exit
    # status alone cannot distinguish success from failure.
    combined = f"{res.stdout or ''}\n{res.stderr or ''}"
    if res.returncode != 0 or "was not successful" in combined:
        LOG.warning(
            "AutoDock-GPU failed for %s: %s",
            map_file.stem,
            combined.strip()[-500:],
        )
        return None
    dlg_path = Path(f"{res_prefix}.dlg")
    if pose_sdf is None:
        energy = parse_dlg_best(dlg_path)
        if energy is None:
            LOG.warning(
                "AutoDock-GPU reported success but no binding energy could be "
                "read from %s",
                dlg_path,
            )
        return energy
    parsed = parse_dlg_best_pose(dlg_path)
    if parsed is None:
        LOG.warning(
            "AutoDock-GPU reported success but no scored pose could be read "
            "from %s",
            dlg_path,
        )
        return None
    energy, pose_text = parsed
    pose_pdbqt = workdir / f"{_safe_target_id(target_id or map_file.stem)}.pdbqt"
    pose_pdbqt.write_text(pose_text)
    _write_sdf_from_pdbqt_pose(
        pose_pdbqt,
        pose_sdf,
        _safe_target_id(target_id or map_file.stem),
        energy,
    )
    return energy


def run_vina_one(
    ligand: Path,
    receptor: Path,
    box_geometry: tuple[tuple[float, float, float], tuple[float, float, float]],
    exhaustiveness: int,
    cpu: int,
    pose_sdf: Path | None = None,
    workdir: Path | None = None,
) -> float | None:
    if not _nonempty(ligand) or not _nonempty(receptor):
        return None
    center, size = box_geometry
    try:
        from vina import Vina  # type: ignore[import-untyped]
    except ImportError:
        LOG.warning("Vina Python package is unavailable")
        return None
    try:
        vina = Vina(sf_name="vina", cpu=cpu, seed=1, verbosity=0)
        vina.set_receptor(str(receptor))
        vina.set_ligand_from_file(str(ligand))
        vina.compute_vina_maps(center=list(center), box_size=list(size))
        vina.dock(exhaustiveness=max(1, exhaustiveness), n_poses=1)
        energies = vina.energies(n_poses=1)
        if len(energies) == 0:
            return None
        energy = float(energies[0][0])
        if pose_sdf is not None:
            if workdir is None:
                raise ValueError("workdir is required when exporting Vina poses")
            pose_pdbqt = workdir / f"{_safe_target_id(receptor.stem)}.pdbqt"
            vina.write_poses(str(pose_pdbqt), n_poses=1, overwrite=True)
            if not _nonempty(pose_pdbqt):
                raise SystemExit(
                    f"Vina produced no docked pose for {receptor.stem}: "
                    f"{pose_pdbqt}"
                )
            _write_sdf_from_pdbqt_pose(
                pose_pdbqt,
                pose_sdf,
                _safe_target_id(receptor.stem),
                energy,
            )
        return energy
    except Exception as exc:
        LOG.warning("Vina failed for %s: %s", receptor.stem, exc)
        return None


def _checked_docking_energy(value: float, target_id: str) -> float:
    if not math.isfinite(value):
        raise SystemExit(f"Docking energy for {target_id} must be finite: {value!r}")
    return value


def _publish_pose_dir(staged_pose_dir: Path, out_pose_dir: Path) -> None:
    _fsync_tree(staged_pose_dir)
    out_pose_dir.parent.mkdir(parents=True, exist_ok=True)
    backup_dir: Path | None = None
    if out_pose_dir.exists():
        backup_dir = out_pose_dir.with_name(
            f".{out_pose_dir.name}.old.{os.getpid()}"
        )
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        out_pose_dir.replace(backup_dir)
    try:
        staged_pose_dir.replace(out_pose_dir)
        _fsync_directory(out_pose_dir.parent)
    except BaseException:
        if (
            backup_dir is not None
            and backup_dir.exists()
            and not out_pose_dir.exists()
        ):
            backup_dir.replace(out_pose_dir)
        raise
    if backup_dir is not None and backup_dir.exists():
        shutil.rmtree(backup_dir)
        _fsync_directory(out_pose_dir.parent)


def _validate_score_pose_parity(target_ids: list[str], pose_dir: Path) -> None:
    expected = {_safe_target_id(target_id) for target_id in target_ids}
    actual = {path.stem for path in pose_dir.glob("*.sdf") if path.is_file()}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        detail = []
        if missing:
            detail.append(f"missing pose(s): {', '.join(missing[:10])}")
        if extra:
            detail.append(f"extra pose(s): {', '.join(extra[:10])}")
        raise SystemExit("Docking score/pose parity failed: " + "; ".join(detail))
    for target_id in sorted(expected):
        _validate_sdf_pose(pose_dir / f"{target_id}.sdf", target_id)


def _write_pose_manifest(
    pose_dir: Path,
    score_path: Path,
    target_energies: dict[str, float],
    *,
    engine: str,
) -> None:
    records = []
    for target_id, energy in target_energies.items():
        pose_path = pose_dir / f"{_safe_target_id(target_id)}.sdf"
        records.append({
            "target_id": target_id,
            "docking_energy_kcal_mol": round(energy, 3),
            "pose_file": pose_path.name,
            "pose_sha256": _sha256_file(pose_path),
            "pose_bytes": pose_path.stat().st_size,
        })
    _write_json_atomic_durable(
        pose_dir / POSE_MANIFEST_NAME,
        {
            "schema_version": POSE_MANIFEST_SCHEMA_VERSION,
            "engine": engine,
            "score_file": score_path.name,
            "score_sha256": _sha256_file(score_path),
            "score_bytes": score_path.stat().st_size,
            "target_count": len(records),
            "targets": records,
        },
    )


def _write_docking_status_manifest(
    path: Path,
    *,
    map_manifest: Path | None,
    target_records: list[dict[str, object]],
    target_energies: dict[str, float],
    failed_targets: set[str],
    engine: str,
) -> None:
    records: list[dict[str, object]] = []
    for source in target_records:
        target_id = str(source["target_id"])
        map_status = str(source.get("status", ""))
        record: dict[str, object] = {
            "target_id": target_id,
            "map_status": map_status,
            "status": map_status,
            "docking_energy_kcal_mol": None,
        }
        if map_status == "map_ready":
            if target_id in target_energies:
                record["status"] = "docked"
                record["docking_energy_kcal_mol"] = round(
                    target_energies[target_id], 3
                )
            elif target_id in failed_targets:
                record["status"] = "structure_failed_docking"
            else:
                record["status"] = "structure_failed_docking"
        records.append(record)
    _write_json_atomic_durable(
        path,
        {
            "schema_version": DOCKING_STATUS_SCHEMA_VERSION,
            "engine": engine,
            "map_manifest": str(map_manifest) if map_manifest is not None else None,
            "target_count": len(records),
            "docked_count": len(target_energies),
            "targets": records,
        },
    )


def _nonfinite_dlg_message(dlg_path: Path) -> str:
    return f"AutoDock-GPU DLG contains non-finite binding energy: {dlg_path}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ligand-pdbqt", required=True, type=Path)
    parser.add_argument("--receptor-dir", required=True, type=Path)
    parser.add_argument("--box-dir", required=True, type=Path)
    parser.add_argument("--no-pocket-list", type=Path, default=None)
    parser.add_argument("--restrict-list", type=Path, default=None)
    parser.add_argument(
        "--map-manifest",
        type=Path,
        default=None,
        help=(
            "Explicit per-query AutoGrid manifest. When supplied, AutoDock uses "
            "only the exact map_fld paths recorded there and never scans receptor "
            "siblings for stale maps."
        ),
    )
    parser.add_argument(
        "--allow-partial-structure",
        action="store_true",
        help=(
            "Publish successful poses and explicit failure statuses even when "
            "some selected targets cannot be docked."
        ),
    )
    parser.add_argument(
        "--engine",
        choices=("autodock_gpu", "vina"),
        default="autodock_gpu",
        help=(
            "Docking engine. autodock_gpu is the claim path and requires complete "
            "grid-map coverage; vina is an explicit degraded diagnostic mode."
        ),
    )
    parser.add_argument("--nrun", type=int, default=20)
    parser.add_argument(
        "--search-volume-advisory",
        type=float,
        default=DEFAULT_SEARCH_VOLUME_ADVISORY,
        help=(
            "box volume in A^3 above which the search effort is raised "
            "(27000 is a 30 A cube, the volume Vina warns past)"
        ),
    )
    parser.add_argument(
        "--high-volume-min-nrun",
        type=int,
        default=DEFAULT_HIGH_VOLUME_MIN_EFFORT,
        help="minimum search effort for a box above the advisory volume",
    )
    parser.add_argument("--ls-method", default="ad")
    parser.add_argument(
        "--max-receptors",
        type=int,
        default=0,
        help="Maximum receptors to dock after filtering; 0 means no cap.",
    )
    parser.add_argument(
        "--vina-cpu",
        type=int,
        default=0,
        help="CPU threads passed to Vina fallback; 0 uses os.cpu_count().",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Log receptor progress every N attempted receptors; 0 disables.",
    )
    parser.add_argument("--out-scores", required=True, type=Path)
    parser.add_argument(
        "--out-status-manifest",
        type=Path,
        default=None,
        help="Optional per-selected-target docking status manifest.",
    )
    parser.add_argument(
        "--out-pose-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for one real docked pose SDF per scored target. "
            "Legacy direct callers may omit this; Snakemake claim rules supply it."
        ),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    pose_outputs = [args.out_pose_dir] if args.out_pose_dir else []
    manifest_outputs = [args.out_status_manifest] if args.out_status_manifest else []
    _remove_outputs(args.out_scores, *pose_outputs, *manifest_outputs)
    if args.nrun < 1:
        raise SystemExit("--nrun must be >= 1")
    if args.high_volume_min_nrun < 1:
        raise SystemExit("--high-volume-min-nrun must be >= 1")
    if (
        not math.isfinite(args.search_volume_advisory)
        or args.search_volume_advisory <= 0.0
    ):
        raise SystemExit("--search-volume-advisory must be positive and finite")
    if args.max_receptors < 0:
        raise SystemExit("--max-receptors must be >= 0")
    if args.vina_cpu < 0:
        raise SystemExit("--vina-cpu must be >= 0")
    if args.progress_every < 0:
        raise SystemExit("--progress-every must be >= 0")
    if args.map_manifest is not None and args.engine != "autodock_gpu":
        raise SystemExit("--map-manifest is only valid with --engine autodock_gpu")
    if args.allow_partial_structure and args.map_manifest is None:
        raise SystemExit("--allow-partial-structure requires --map-manifest")
    if args.out_status_manifest is not None and args.map_manifest is None:
        raise SystemExit("--out-status-manifest requires --map-manifest")

    if args.map_manifest is not None and args.restrict_list is not None:
        raise SystemExit("--map-manifest and --restrict-list cannot be combined")
    if args.no_pocket_list is not None and not args.no_pocket_list.exists():
        raise SystemExit(f"No-pocket target list does not exist: {args.no_pocket_list}")
    if args.restrict_list is not None and not args.restrict_list.exists():
        raise SystemExit(f"Docking restrict list does not exist: {args.restrict_list}")
    map_records: list[dict[str, object]] = []
    explicit_maps: dict[str, Path] = {}
    if args.map_manifest is not None:
        map_records, explicit_maps = _read_map_manifest(args.map_manifest)
        no_pocket: set[str] = set()
        restrict = [
            str(record["target_id"])
            for record in map_records
            if str(record.get("status", "")) == "map_ready"
        ]
    else:
        no_pocket = (
            set(read_uniprots(args.no_pocket_list)) if args.no_pocket_list else set()
        )
        restrict = read_uniprots(args.restrict_list) if args.restrict_list else None
    receptors = receptor_list(
        args.receptor_dir,
        no_pocket,
        restrict,
        max_receptors=args.max_receptors,
    )
    cap_detail = f" (cap={args.max_receptors})" if args.max_receptors > 0 else ""
    LOG.info("Docking %d receptors%s", len(receptors), cap_detail)
    if not receptors and not args.allow_partial_structure:
        raise SystemExit("No receptors selected for AutoDock-GPU")
    autodock_bin = autodock_gpu_binary()
    if args.engine == "autodock_gpu" and receptors and autodock_bin is None:
        raise SystemExit(
            "autodock_gpu_128wi is not available on PATH, AUTODOCK_GPU_BIN, "
            "or tools/autodock_gpu/bin"
        )
    if not _nonempty(args.ligand_pdbqt):
        raise SystemExit(f"Ligand PDBQT is missing or empty for AutoDock-GPU: {args.ligand_pdbqt}")
    if args.map_manifest is not None:
        map_files = {receptor: explicit_maps.get(receptor.stem) for receptor in receptors}
        map_count = len(explicit_maps)
        map_denominator = len(map_records)
    else:
        map_files = {
            receptor: autodock_map_for_receptor(receptor) for receptor in receptors
        }
        map_count = sum(1 for map_file in map_files.values() if map_file is not None)
        map_denominator = len(receptors)
    map_coverage_complete = map_count == map_denominator
    LOG.info(
        "AutoDock-GPU map coverage: %d/%d selected targets",
        map_count,
        map_denominator,
    )
    if (
        args.engine == "autodock_gpu"
        and not map_coverage_complete
        and not args.allow_partial_structure
    ):
        raise SystemExit(
            "AutoDock-GPU map coverage incomplete: "
            f"{map_count}/{map_denominator} receptors have .maps.fld inputs"
        )
    if args.engine == "vina":
        vina_cpu = args.vina_cpu if args.vina_cpu > 0 else max(1, os.cpu_count() or 1)
        LOG.warning("Using explicit degraded Vina diagnostic mode; outputs are not claimable")
        LOG.info("Vina CPU threads: %d", vina_cpu)
    else:
        vina_cpu = 1

    n_written = 0
    raised_effort_targets = 0
    unscaled_effort_targets = 0
    written_target_ids: list[str] = []
    written_target_energies: dict[str, float] = {}
    failed_targets: list[str] = []
    tmp_scores = _tmp_output(args.out_scores)
    # Stage beside the destination. The default temp root is often a separate
    # filesystem - /tmp is tmpfs on this host - and os.replace cannot move a
    # directory across devices, so publishing died after every pose was written.
    staging_parent = (
        args.out_pose_dir.parent if args.out_pose_dir is not None
        else args.out_scores.parent
    )
    staging_parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(dir=staging_parent) as tmp:
            tmp_root = Path(tmp)
            staged_pose_dir = tmp_root / "poses"
            if args.out_pose_dir is not None:
                staged_pose_dir.mkdir()
            with tmp_scores.open("w", newline="") as fh:
                writer = csv.writer(fh, delimiter="\t")
                writer.writerow([
                    "target_id",
                    "vina_score",
                    "neg_vina_score",
                    "engine",
                    "map_coverage_complete",
                    "map_coverage_numerator",
                    "map_coverage_denominator",
                    "degraded",
                ])
                for idx, r in enumerate(receptors, start=1):
                    uid = r.stem
                    _safe_target_id(uid)
                    box = args.box_dir / f"{uid}.box.txt"
                    log_progress = args.progress_every and (
                        idx % args.progress_every == 0 or idx == len(receptors)
                    )
                    box_geometry = parse_box(box) if args.engine == "vina" else None
                    # The box volume sets the search effort. AutoDock-GPU encodes
                    # the box in its precomputed maps and does not otherwise need
                    # the file, so a missing one leaves the effort unscaled rather
                    # than failing a run that could proceed.
                    scaling_geometry = box_geometry
                    if scaling_geometry is None:
                        try:
                            scaling_geometry = parse_box(box)
                        except SystemExit:
                            scaling_geometry = None
                            unscaled_effort_targets += 1
                    effort = args.nrun
                    if scaling_geometry is not None:
                        effort = sampling_effort(
                            args.nrun,
                            box_volume(scaling_geometry[1]),
                            advisory=args.search_volume_advisory,
                            high_volume_floor=args.high_volume_min_nrun,
                        )
                        if effort > args.nrun:
                            raised_effort_targets += 1
                    energy = None
                    if args.engine == "autodock_gpu":
                        map_file = map_files[r]
                        if map_file is not None and autodock_bin is not None:
                            energy = run_autodock_gpu_one(
                                args.ligand_pdbqt,
                                map_file,
                                effort,
                                args.ls_method,
                                tmp_root / uid,
                                autodock_bin,
                                (
                                    _pose_path(staged_pose_dir, uid)
                                    if args.out_pose_dir is not None
                                    else None
                                ),
                                uid,
                            )
                    if energy is None and args.engine == "vina":
                        assert box_geometry is not None
                        energy = run_vina_one(
                            args.ligand_pdbqt,
                            r,
                            box_geometry,
                            effort,
                            vina_cpu,
                            (
                                _pose_path(staged_pose_dir, uid)
                                if args.out_pose_dir is not None
                                else None
                            ),
                            tmp_root / uid,
                        )
                    if energy is None:
                        failed_targets.append(uid)
                        if log_progress:
                            LOG.info(
                                "Docking progress %d/%d receptors (written=%d)",
                                idx,
                                len(receptors),
                                n_written,
                            )
                        continue
                    energy = _checked_docking_energy(energy, uid)
                    writer.writerow([
                        uid,
                        f"{energy:.3f}",
                        f"{-energy:.3f}",
                        args.engine,
                        str(map_coverage_complete).lower(),
                        str(map_count),
                        str(map_denominator),
                        str(args.engine == "vina").lower(),
                    ])
                    n_written += 1
                    written_target_ids.append(uid)
                    written_target_energies[uid] = energy
                    if log_progress:
                        LOG.info(
                            "Docking progress %d/%d receptors (written=%d)",
                            idx,
                            len(receptors),
                            n_written,
                        )
            if n_written == 0 and not args.allow_partial_structure:
                raise SystemExit("AutoDock-GPU produced no usable receptor scores")
            if (
                args.engine == "autodock_gpu"
                and failed_targets
                and not args.allow_partial_structure
            ):
                shown = ", ".join(failed_targets[:10])
                suffix = "..." if len(failed_targets) > 10 else ""
                raise SystemExit(
                    "AutoDock-GPU failed to produce scores for all selected "
                    "receptors: "
                    f"{len(failed_targets)}/{len(receptors)} failed "
                    f"[{shown}{suffix}]"
                )
            if args.out_pose_dir is not None:
                _validate_score_pose_parity(written_target_ids, staged_pose_dir)
            _fsync_file(tmp_scores)
            if args.out_pose_dir is not None:
                _publish_pose_dir(staged_pose_dir, args.out_pose_dir)
            tmp_scores.replace(args.out_scores)
            _fsync_directory(args.out_scores.parent)
            if args.out_pose_dir is not None:
                _write_pose_manifest(
                    args.out_pose_dir,
                    args.out_scores,
                    written_target_energies,
                    engine=args.engine,
                )
            if args.out_status_manifest is not None:
                _write_docking_status_manifest(
                    args.out_status_manifest,
                    map_manifest=args.map_manifest,
                    target_records=map_records,
                    target_energies=written_target_energies,
                    failed_targets=set(failed_targets),
                    engine=args.engine,
                )
    except BaseException:
        tmp_scores.unlink(missing_ok=True)
        args.out_scores.unlink(missing_ok=True)
        if args.out_pose_dir is not None:
            _remove_outputs(args.out_pose_dir)
        if args.out_status_manifest is not None:
            _remove_outputs(args.out_status_manifest)
        raise
    if raised_effort_targets:
        LOG.info(
            "Raised search effort above --nrun %d for %d/%d receptor(s) whose box "
            "exceeds %.0f A^3 (floor %d)",
            args.nrun,
            raised_effort_targets,
            len(receptors),
            args.search_volume_advisory,
            args.high_volume_min_nrun,
        )
    if unscaled_effort_targets:
        LOG.warning(
            "No docking box for %d/%d receptor(s); their search effort stayed at "
            "--nrun %d and was not scaled to the box volume",
            unscaled_effort_targets,
            len(receptors),
            args.nrun,
        )
    LOG.info("Wrote %s", args.out_scores)


if __name__ == "__main__":
    main()
