#!/usr/bin/env python3
"""Build ligand-specific AutoGrid4 maps for the fixed Daina target set.

Every selected target receives a manifest record. Missing pockets/prepared
receptors and per-target map failures remain explicit instead of disappearing.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = "skinscout.autogrid-map-manifest.v1"
AUTOGRID_VERSION = "4.2.8+6d2847b"
STRUCTURE_SOURCE = "AlphaFold-human-v4"
POCKET_SOURCE = "P2Rank-top1"
SAFE_TARGET = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
ATOM_TYPE = re.compile(r"^[A-Za-z][A-Za-z0-9]*$")
BOX_FIELDS = (
    "center_x",
    "center_y",
    "center_z",
    "size_x",
    "size_y",
    "size_z",
)
LOCAL_AUTOGRID_CANDIDATES = (
    Path(__file__).resolve().parents[1] / "tools/autogrid/bin/autogrid4",
    Path(__file__).resolve().parents[1] / "tools/AutoGrid/autogrid4",
    Path(__file__).resolve().parents[1] / "tools/AutoGrid/bin/autogrid4",
)
LOCAL_PARAMETER_CANDIDATES = (
    Path(__file__).resolve().parents[1] / "tools/AutoDock-GPU/AD4_parameters.dat",
    Path(__file__).resolve().parents[1] / "tools/autogrid/share/AD4.1_bound.dat",
    Path(__file__).resolve().parents[1] / "tools/autogrid/AD4.1_bound.dat",
    Path(__file__).resolve().parents[1] / "tools/AutoGrid/AD4.1_bound.dat",
    Path(__file__).resolve().parents[1] / "tools/AutoGrid/parameter_library/AD4.1_bound.dat",
)


class InvalidTargetInput(ValueError):
    """A target cannot be structurally evaluated because its input is invalid."""


@dataclass(frozen=True)
class Box:
    center: tuple[float, float, float]
    size: tuple[float, float, float]


def _nonempty(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _safe_target_id(value: object) -> str:
    target_id = str(value).strip()
    if (
        not target_id
        or target_id in {".", ".."}
        or not SAFE_TARGET.fullmatch(target_id)
    ):
        raise InvalidTargetInput(f"unsafe target_id: {value!r}")
    return target_id


def _read_selected(path: Path) -> list[dict[str, str]]:
    if not _nonempty(path):
        raise SystemExit(f"Daina selected-target CSV is missing or empty: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"target_id", "daina_rank", "daina_score"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise SystemExit(
                "Daina selected-target CSV missing required column(s): "
                + ", ".join(missing)
            )
        rows = list(reader)
    if not rows:
        raise SystemExit(f"Daina selected-target CSV contains no rows: {path}")
    seen: set[str] = set()
    expected_rank = 1
    for row in rows:
        try:
            target_id = _safe_target_id(row.get("target_id", ""))
        except InvalidTargetInput as exc:
            raise SystemExit(str(exc)) from exc
        if target_id in seen:
            raise SystemExit(f"Daina selected-target CSV has duplicate target_id: {target_id}")
        seen.add(target_id)
        try:
            rank = int(str(row.get("daina_rank", "")))
            score = float(str(row.get("daina_score", "")))
        except ValueError as exc:
            raise SystemExit(
                f"Daina selected-target row has invalid rank/score for {target_id}"
            ) from exc
        if rank != expected_rank:
            raise SystemExit(
                "Daina selected-target ranks must be contiguous and ordered; "
                f"expected {expected_rank}, found {rank} for {target_id}"
            )
        if not math.isfinite(score):
            raise SystemExit(f"Daina selected-target score must be finite: {target_id}")
        expected_rank += 1
    return rows


def _read_receptor_list(path: Path) -> list[dict[str, str]]:
    """Read an unranked receptor list, one target id per line.

    The comprehensive path docks every receptor rather than a ranked selection,
    so it has no Daina rank or score to supply. Inventing them to satisfy the
    selected-CSV contract would put a fabricated ranking into the manifest, so
    the two inputs stay separate and only one may be given.
    """
    if not _nonempty(path):
        raise SystemExit(f"Receptor list is missing or empty: {path}")
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        try:
            target_id = _safe_target_id(text)
        except InvalidTargetInput as exc:
            raise SystemExit(f"{path}: {exc}") from exc
        if target_id in seen:
            raise SystemExit(f"Receptor list has duplicate target_id: {target_id}")
        seen.add(target_id)
        rows.append(
            {
                "target_id": target_id,
                "daina_rank": str(len(rows) + 1),
                "daina_score": "",
            }
        )
    if not rows:
        raise SystemExit(f"Receptor list contains no target ids: {path}")
    return rows


def _read_no_pocket(path: Path) -> set[str]:
    if not path.exists():
        raise SystemExit(f"No-pocket target list does not exist: {path}")
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def _parse_box(path: Path) -> Box:
    if not _nonempty(path):
        raise FileNotFoundError(path)
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    missing = sorted(set(BOX_FIELDS) - set(values))
    if missing:
        raise InvalidTargetInput(
            f"box file is missing field(s) {', '.join(missing)}: {path}"
        )
    try:
        parsed = tuple(float(values[field]) for field in BOX_FIELDS)
    except ValueError as exc:
        raise InvalidTargetInput(f"box file contains non-numeric geometry: {path}") from exc
    if not all(math.isfinite(value) for value in parsed):
        raise InvalidTargetInput(f"box file contains non-finite geometry: {path}")
    if any(value <= 0.0 for value in parsed[3:]):
        raise InvalidTargetInput(f"box sizes must be positive: {path}")
    return Box(center=parsed[:3], size=parsed[3:])


def _pdbqt_atom_types(path: Path) -> tuple[str, ...]:
    if not _nonempty(path):
        raise FileNotFoundError(path)
    types: set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        fields = line.split()
        if not fields:
            continue
        atom_type = fields[-1]
        if not ATOM_TYPE.fullmatch(atom_type):
            raise InvalidTargetInput(
                f"invalid AutoDock atom type {atom_type!r} in {path}"
            )
        types.add(atom_type)
    if not types:
        raise InvalidTargetInput(f"PDBQT contains no typed ATOM/HETATM records: {path}")
    return tuple(sorted(types))


def _grid_points(size: tuple[float, float, float], spacing: float) -> tuple[int, int, int]:
    points: list[int] = []
    for dimension in size:
        count = int(math.ceil(dimension / spacing))
        if count % 2:
            count += 1
        if count < 2 or count > 126:
            raise InvalidTargetInput(
                "AutoGrid box requires an even npts value in [2, 126]; "
                f"size={dimension:g}, spacing={spacing:g}, npts={count}"
            )
        points.append(count)
    return tuple(points)  # type: ignore[return-value]


def _resolve_executable(explicit: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit.expanduser())
    env_value = os.environ.get("AUTOGRID4_BIN")
    if env_value:
        candidates.append(Path(env_value).expanduser())
    found = shutil.which("autogrid4")
    if found:
        candidates.append(Path(found))
    candidates.extend(LOCAL_AUTOGRID_CANDIDATES)
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise SystemExit("autogrid4 executable is not available")


def _rank_in_band(rank: int, first: int, last: int) -> bool:
    """Whether this Daina rank is one the caller asked to dock.

    Both bounds default to 0, meaning no restriction, so the unrestricted path
    is the same code path rather than a separate one.
    """
    if first <= 0 and last <= 0:
        return True
    if first > 0 and rank < first:
        return False
    if last > 0 and rank > last:
        return False
    return True


def _resolve_parameter_file(explicit: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit.expanduser())
    env_value = os.environ.get("AUTODOCK_PARAMETER_FILE")
    if env_value:
        candidates.append(Path(env_value).expanduser())
    candidates.extend(LOCAL_PARAMETER_CANDIDATES)
    for candidate in candidates:
        if _nonempty(candidate):
            return candidate.resolve()
    raise SystemExit("AutoDock4 parameter file is not available")


def _gpf_text(
    *,
    target_id: str,
    receptor: Path,
    receptor_types: tuple[str, ...],
    ligand_types: tuple[str, ...],
    box: Box,
    spacing: float,
    parameter_file: Path,
) -> str:
    npts = _grid_points(box.size, spacing)
    lines = [
        f"parameter_file {parameter_file}",
        f"npts {' '.join(str(value) for value in npts)}",
        f"gridfld {target_id}.maps.fld",
        f"spacing {spacing:.6f}",
        f"receptor_types {' '.join(receptor_types)}",
        f"ligand_types {' '.join(ligand_types)}",
        f"receptor {receptor}",
        "gridcenter " + " ".join(f"{value:.6f}" for value in box.center),
        "smooth 0.500000",
    ]
    lines.extend(f"map {target_id}.{atom_type}.map" for atom_type in ligand_types)
    lines.extend(
        [
            f"elecmap {target_id}.e.map",
            f"dsolvmap {target_id}.d.map",
            "dielectric -0.1465",
        ]
    )
    return "\n".join(lines) + "\n"


def _expected_map_files(target_id: str, ligand_types: tuple[str, ...]) -> list[str]:
    return [
        f"{target_id}.maps.fld",
        *(f"{target_id}.{atom_type}.map" for atom_type in ligand_types),
        f"{target_id}.e.map",
        f"{target_id}.d.map",
    ]


def _cache_entry_valid(
    entry: Path,
    *,
    cache_key: str,
    target_id: str,
    ligand_types: tuple[str, ...],
) -> bool:
    manifest_path = entry / "cache_manifest.json"
    if not _nonempty(manifest_path):
        return False
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "skinscout.autogrid-cache-entry.v1"
        or payload.get("cache_key") != cache_key
        or payload.get("target_id") != target_id
        or not isinstance(payload.get("files"), dict)
    ):
        return False
    files = payload["files"]
    expected = _expected_map_files(target_id, ligand_types)
    if set(files) != set(expected):
        return False
    for name in expected:
        metadata = files.get(name)
        path = entry / name
        if (
            not isinstance(metadata, dict)
            or not _nonempty(path)
            or metadata.get("bytes") != path.stat().st_size
            or metadata.get("sha256") != _sha256(path)
        ):
            return False
    return True


def _build_cache_entry(
    *,
    entry: Path,
    cache_key: str,
    target_id: str,
    receptor: Path,
    receptor_types: tuple[str, ...],
    ligand_types: tuple[str, ...],
    box: Box,
    spacing: float,
    parameter_file: Path,
    autogrid: Path,
    autogrid_version: str,
) -> None:
    if _cache_entry_valid(
        entry,
        cache_key=cache_key,
        target_id=target_id,
        ligand_types=ligand_types,
    ):
        return
    entry.parent.mkdir(parents=True, exist_ok=True)
    lock_path = entry.parent / f".{cache_key}.lock"
    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if _cache_entry_valid(
            entry,
            cache_key=cache_key,
            target_id=target_id,
            ligand_types=ligand_types,
        ):
            return
        staging = Path(tempfile.mkdtemp(prefix=f".{cache_key}.staging.", dir=entry.parent))
        try:
            gpf = staging / f"{target_id}.gpf"
            gpf.write_text(
                _gpf_text(
                    target_id=target_id,
                    receptor=receptor,
                    receptor_types=receptor_types,
                    ligand_types=ligand_types,
                    box=box,
                    spacing=spacing,
                    parameter_file=parameter_file,
                ),
                encoding="utf-8",
            )
            glg = staging / f"{target_id}.glg"
            result = subprocess.run(
                [str(autogrid), "-p", gpf.name, "-l", glg.name],
                cwd=staging,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                detail = (result.stderr or result.stdout).strip()[-1000:]
                raise RuntimeError(f"autogrid4 exited {result.returncode}: {detail}")
            missing = [
                name
                for name in _expected_map_files(target_id, ligand_types)
                if not _nonempty(staging / name)
            ]
            if missing:
                raise RuntimeError(
                    "autogrid4 omitted required map file(s): " + ", ".join(missing)
                )
            _write_json_atomic(
                staging / "cache_manifest.json",
                {
                    "schema_version": "skinscout.autogrid-cache-entry.v1",
                    "cache_key": cache_key,
                    "target_id": target_id,
                    "autogrid_version": autogrid_version,
                    "files": {
                        name: {"sha256": _sha256(staging / name), "bytes": (staging / name).stat().st_size}
                        for name in _expected_map_files(target_id, ligand_types)
                    },
                },
            )
            if entry.exists():
                shutil.rmtree(entry)
            staging.replace(entry)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise


def _copy_cache_entry(entry: Path, destination: Path) -> None:
    def link_or_copy(source: str, target: str) -> str:
        try:
            os.link(source, target)
            return target
        except OSError:
            return shutil.copy2(source, target)

    shutil.copytree(entry, destination, copy_function=link_or_copy)


def build_maps(args: argparse.Namespace) -> dict[str, object]:
    if (args.selected_csv is None) == (args.receptor_list is None):
        raise SystemExit(
            "Provide exactly one of --selected-csv or --receptor-list"
        )
    selected = (
        _read_selected(args.selected_csv)
        if args.selected_csv is not None
        else _read_receptor_list(args.receptor_list)
    )
    no_pocket = _read_no_pocket(args.no_pocket_list)
    if not _nonempty(args.ligand_pdbqt):
        raise SystemExit(f"Ligand PDBQT is missing or empty: {args.ligand_pdbqt}")
    try:
        ligand_types = _pdbqt_atom_types(args.ligand_pdbqt)
    except (FileNotFoundError, InvalidTargetInput) as exc:
        raise SystemExit(f"Invalid ligand PDBQT: {exc}") from exc
    autogrid = _resolve_executable(args.autogrid_bin)
    parameter_file = _resolve_parameter_file(args.parameter_file)
    cache_root = args.cache_dir.expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    args.out_map_dir.parent.mkdir(parents=True, exist_ok=True)
    staged_map_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{args.out_map_dir.name}.staging.",
            dir=args.out_map_dir.parent,
        )
    )
    records: list[dict[str, object]] = []
    try:
        for row in selected:
            target_id = _safe_target_id(row["target_id"])
            base: dict[str, object] = {
                "target_id": target_id,
                "daina_rank": int(row["daina_rank"]),
                # Null rather than a stand-in number: the unranked receptor-list
                # mode has no Daina score, and writing one would put a
                # fabricated ranking into a hash-bound manifest.
                "daina_score": (
                    float(row["daina_score"]) if str(row["daina_score"]).strip() else None
                ),
                "status": "",
                "detail": "",
                "map_fld": "",
                "map_cache_key": "",
                "pocket_id": "",
                "pocket_source": "",
                "structure_source": "",
            }
            # Outside the band the re-ranking consults, so it was never asked
            # for. Recorded with a status of its own so the manifest keeps exact
            # parity with the selection without claiming a failure.
            if not _rank_in_band(base["daina_rank"], args.dock_rank_from, args.dock_rank_to):
                base.update(
                    status="structure_not_requested",
                    detail=(
                        "Daina 순위 "
                        f"{args.dock_rank_from}-{args.dock_rank_to} 밖이라 도킹 대상이 아닙니다"
                    ),
                )
                records.append(base)
                continue
            if target_id in no_pocket:
                base.update(
                    status="structural_unavailable_no_pocket",
                    detail="Stage 0 recorded no usable pocket",
                )
                records.append(base)
                continue
            receptor = args.receptor_dir / f"{target_id}.pdbqt"
            box_path = args.box_dir / f"{target_id}.box.txt"
            if not _nonempty(receptor) or not _nonempty(box_path):
                missing = []
                if not _nonempty(receptor):
                    missing.append("receptor PDBQT")
                if not _nonempty(box_path):
                    missing.append("docking box")
                base.update(
                    status="structural_unavailable_prep",
                    detail="missing " + " and ".join(missing),
                )
                records.append(base)
                continue
            try:
                box = _parse_box(box_path)
                receptor_types = _pdbqt_atom_types(receptor)
                _grid_points(box.size, args.spacing)
            except (InvalidTargetInput, FileNotFoundError) as exc:
                base.update(status="structural_unavailable_invalid", detail=str(exc))
                records.append(base)
                continue
            recipe = {
                "schema_version": "skinscout.autogrid-cache-key.v1",
                "target_id": target_id,
                "receptor_sha256": _sha256(receptor),
                "box": {"center": box.center, "size": box.size},
                "spacing": args.spacing,
                "receptor_types": receptor_types,
                "ligand_types": ligand_types,
                "autogrid_version": args.autogrid_version,
                "autogrid_sha256": _sha256(autogrid),
                "parameter_sha256": _sha256(parameter_file),
            }
            cache_key = _json_sha256(recipe)
            entry = cache_root / cache_key
            try:
                _build_cache_entry(
                    entry=entry,
                    cache_key=cache_key,
                    target_id=target_id,
                    receptor=receptor.resolve(),
                    receptor_types=receptor_types,
                    ligand_types=ligand_types,
                    box=box,
                    spacing=args.spacing,
                    parameter_file=parameter_file,
                    autogrid=autogrid,
                    autogrid_version=args.autogrid_version,
                )
                destination = staged_map_dir / target_id
                _copy_cache_entry(entry, destination)
                map_path = destination / f"{target_id}.maps.fld"
                base.update(
                    status="map_ready",
                    detail="ligand-specific AutoGrid maps ready",
                    pocket_id=f"{target_id}:p2rank_top1",
                    pocket_source=POCKET_SOURCE,
                    structure_source=STRUCTURE_SOURCE,
                    map_fld=str(
                        Path(
                            os.path.relpath(
                                args.out_map_dir / target_id / map_path.name,
                                args.out_manifest.parent,
                            )
                        )
                    ),
                    map_cache_key=cache_key,
                    map_fld_sha256=_sha256(map_path),
                )
            except Exception as exc:
                base.update(
                    status="structure_failed_map",
                    detail=str(exc)[:1000],
                    map_cache_key=cache_key,
                )
            records.append(base)

        if args.out_map_dir.exists():
            shutil.rmtree(args.out_map_dir)
        staged_map_dir.replace(args.out_map_dir)
    except BaseException:
        shutil.rmtree(staged_map_dir, ignore_errors=True)
        raise

    counts: dict[str, int] = {}
    for record in records:
        status = str(record["status"])
        counts[status] = counts.get(status, 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        # Whichever input named the receptors is recorded with its digest; the
        # other stays null so a reader can tell a ranked selection from an
        # unranked receptor list.
        "selected_csv": (
            str(args.selected_csv) if args.selected_csv is not None else None
        ),
        "selected_csv_sha256": (
            _sha256(args.selected_csv) if args.selected_csv is not None else None
        ),
        "receptor_list": (
            str(args.receptor_list) if args.receptor_list is not None else None
        ),
        "receptor_list_sha256": (
            _sha256(args.receptor_list) if args.receptor_list is not None else None
        ),
        "ligand_pdbqt": str(args.ligand_pdbqt),
        "ligand_pdbqt_sha256": _sha256(args.ligand_pdbqt),
        "ligand_atom_types": list(ligand_types),
        "autogrid_version": args.autogrid_version,
        "autogrid_sha256": _sha256(autogrid),
        "parameter_file_sha256": _sha256(parameter_file),
        "cache_root": str(cache_root),
        "target_count": len(records),
        "status_counts": counts,
        "targets": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-csv", type=Path)
    parser.add_argument(
        "--receptor-list",
        type=Path,
        help=(
            "Unranked newline-delimited target ids, for callers that build maps "
            "for every receptor instead of a ranked selection."
        ),
    )
    parser.add_argument("--ligand-pdbqt", required=True, type=Path)
    parser.add_argument("--receptor-dir", required=True, type=Path)
    parser.add_argument("--box-dir", required=True, type=Path)
    parser.add_argument("--no-pocket-list", required=True, type=Path)
    parser.add_argument("--autogrid-bin", type=Path)
    parser.add_argument("--parameter-file", type=Path)
    parser.add_argument("--autogrid-version", default=AUTOGRID_VERSION)
    parser.add_argument("--spacing", type=float, default=0.375)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("~/.cache/skinscout/autogrid"),
    )
    parser.add_argument(
        "--dock-rank-from",
        type=int,
        default=0,
        help=(
            "이 Daina 순위부터만 격자를 만듭니다(포함). 0이면 제한 없음. "
            "재정렬이 참조하지 않는 순위는 계산할 이유가 없습니다."
        ),
    )
    parser.add_argument(
        "--dock-rank-to",
        type=int,
        default=0,
        help="이 Daina 순위까지만 격자를 만듭니다(포함). 0이면 제한 없음.",
    )
    parser.add_argument("--out-map-dir", required=True, type=Path)
    parser.add_argument("--out-manifest", required=True, type=Path)
    args = parser.parse_args()

    if not math.isfinite(args.spacing) or args.spacing <= 0.0:
        raise SystemExit(f"--spacing must be finite and > 0: {args.spacing!r}")
    if args.out_manifest.exists():
        args.out_manifest.unlink()
    try:
        payload = build_maps(args)
        _write_json_atomic(args.out_manifest, payload)
    except BaseException:
        args.out_manifest.unlink(missing_ok=True)
        if args.out_map_dir.exists():
            shutil.rmtree(args.out_map_dir)
        raise


if __name__ == "__main__":
    main()
