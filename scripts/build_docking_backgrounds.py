#!/usr/bin/env python3
"""Dock a fixed background ligand panel to give each receptor its own scale.

`eval/docking_normalization` scores a query as its displacement below the
receptor's own background rather than as a raw energy, which removes the
pocket-size bias that otherwise makes a proteome-wide screen rank receptors
instead of fit. That needs one empirical score distribution per receptor, and
this builds it.

The background is query-independent: dock the panel once per receptor and every
future compound reuses the result. Runs are resumable at receptor granularity,
so a long screen can be interrupted without losing completed work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path
from typing import Any

BACKGROUND_SCHEMA = "skinscout.docking_background_run.v1"
BOX_FIELD_RE = re.compile(r"(\w+)\s*=\s*(-?[0-9.]+)")


class BackgroundBuildError(ValueError):
    """Raised when an input or a produced background violates its contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_box(path: Path) -> tuple[list[float], list[float]]:
    fields = {k: float(v) for k, v in BOX_FIELD_RE.findall(path.read_text())}
    missing = sorted(
        {"center_x", "center_y", "center_z", "size_x", "size_y", "size_z"} - set(fields)
    )
    if missing:
        raise BackgroundBuildError(f"box missing {', '.join(missing)}: {path}")
    return (
        [fields["center_x"], fields["center_y"], fields["center_z"]],
        [fields["size_x"], fields["size_y"], fields["size_z"]],
    )


def load_ligand_panel(path: Path) -> dict[str, str]:
    """Read {ligand_id: pdbqt} and refuse a panel too thin to define a scale."""
    if not path.exists() or path.stat().st_size == 0:
        raise BackgroundBuildError(f"ligand panel is required: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or not payload:
        raise BackgroundBuildError(f"ligand panel must be a non-empty object: {path}")
    for ligand_id, pdbqt in payload.items():
        if not isinstance(pdbqt, str) or "ATOM" not in pdbqt:
            raise BackgroundBuildError(f"ligand {ligand_id} has no PDBQT atoms")
    return payload


_PANEL: dict[str, str] = {}
_ARGS: argparse.Namespace | None = None


def _init(panel_path: str, args: argparse.Namespace) -> None:
    global _PANEL, _ARGS
    _PANEL = load_ligand_panel(Path(panel_path))
    _ARGS = args


def dock_receptor(target_id: str) -> tuple[str, int]:
    assert _ARGS is not None
    shard = _ARGS.out_dir / f"{target_id}.json"
    if shard.exists():
        return target_id, -1
    from vina import Vina

    try:
        center, size = read_box(_ARGS.box_dir / f"{target_id}.box.txt")
    except (BackgroundBuildError, OSError):
        return target_id, 0
    receptor = _ARGS.receptor_dir / f"{target_id}.pdbqt"
    scores: dict[str, float] = {}
    for ligand_id, pdbqt in _PANEL.items():
        try:
            engine = Vina(sf_name=_ARGS.scoring, cpu=1, seed=_ARGS.seed, verbosity=0)
            engine.set_receptor(str(receptor))
            engine.set_ligand_from_string(pdbqt)
            engine.compute_vina_maps(center=center, box_size=size)
            engine.dock(exhaustiveness=_ARGS.exhaustiveness, n_poses=1)
            scores[ligand_id] = float(engine.energies(n_poses=1)[0][0])
        except Exception:
            # One failed pose must not lose the rest of the receptor's panel.
            continue
    shard.parent.mkdir(parents=True, exist_ok=True)
    tmp = shard.with_suffix(".tmp")
    tmp.write_text(json.dumps(scores, sort_keys=True))
    tmp.replace(shard)
    return target_id, len(scores)


def summarise(args: argparse.Namespace, panel: dict[str, str]) -> dict[str, Any]:
    """Turn completed shards into ReceptorBackground records plus a manifest."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
    from docking_normalization import (  # noqa: PLC0415
        DockingNormalizationError,
        background_manifest,
        build_receptor_background,
    )

    records = []
    rejected: dict[str, str] = {}
    for shard in sorted(args.out_dir.glob("*.json")):
        scores = json.loads(shard.read_text())
        try:
            records.append(
                build_receptor_background(
                    shard.stem,
                    list(scores.values()),
                    method=args.method,
                    min_ligands=args.min_ligands,
                )
            )
        except DockingNormalizationError as exc:
            rejected[shard.stem] = str(exc)
    if not records:
        raise BackgroundBuildError(
            "no receptor produced a usable background; "
            f"{len(rejected)} rejected, first: {next(iter(rejected.items()), None)}"
        )
    manifest = background_manifest(
        records,
        panel_id=args.panel_id,
        panel_sha256=_sha256_text(json.dumps(panel, sort_keys=True)),
        engine="AutoDock Vina (python bindings)",
        engine_version=args.engine_version,
        method=args.method,
    )
    manifest.update(
        {
            "run_schema_version": BACKGROUND_SCHEMA,
            "created_at_utc": _utc_now(),
            "box_dir": str(args.box_dir),
            "receptor_dir": str(args.receptor_dir),
            "exhaustiveness": int(args.exhaustiveness),
            "scoring_function": args.scoring,
            "seed": int(args.seed),
            "panel_ligands": int(len(panel)),
            "receptors_rejected": len(rejected),
            "rejection_reasons": dict(sorted(rejected.items())[:20]),
        }
    )
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.out_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    backgrounds = {record.target_id: record.to_dict() for record in records}
    args.out_backgrounds.parent.mkdir(parents=True, exist_ok=True)
    args.out_backgrounds.write_text(json.dumps(backgrounds, indent=2, sort_keys=True) + "\n")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receptor-list", required=True, type=Path)
    parser.add_argument("--ligand-panel", required=True, type=Path)
    parser.add_argument("--receptor-dir", required=True, type=Path)
    parser.add_argument("--box-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--out-backgrounds", required=True, type=Path)
    parser.add_argument("--out-manifest", required=True, type=Path)
    parser.add_argument("--panel-id", default="background-v1")
    parser.add_argument("--engine-version", default="unknown")
    parser.add_argument("--method", default="robust", choices=("robust", "gaussian"))
    parser.add_argument("--min-ligands", type=int, default=30)
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--scoring", default="vina")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument(
        "--summarise-only",
        action="store_true",
        help="skip docking and summarise the shards already on disk",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        panel = load_ligand_panel(args.ligand_panel)
        targets = [
            line.strip()
            for line in args.receptor_list.read_text().splitlines()
            if line.strip()
        ]
        if not targets:
            raise BackgroundBuildError(f"receptor list is empty: {args.receptor_list}")
        if not args.summarise_only:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            todo = [t for t in targets if not (args.out_dir / f"{t}.json").exists()]
            print(f"[backgrounds] {len(todo)}/{len(targets)} receptors to dock", flush=True)
            if todo:
                with Pool(
                    processes=max(1, args.processes),
                    initializer=_init,
                    initargs=(str(args.ligand_panel), args),
                ) as pool:
                    for done, (target_id, n) in enumerate(
                        pool.imap_unordered(dock_receptor, todo), start=1
                    ):
                        if done % 25 == 0:
                            print(f"  {done}/{len(todo)} ({target_id}: {n})", flush=True)
        manifest = summarise(args, panel)
    except BackgroundBuildError as exc:
        print(str(exc))
        return 1
    print(
        f"[backgrounds] {manifest['receptors']} receptors, "
        f"{manifest['receptors_rejected']} rejected, "
        f"scale median {manifest['scale_median']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
