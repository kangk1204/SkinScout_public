#!/usr/bin/env python3
"""stage7_5_aizynth.py — Run AiZynthFinder MCTS on each analog SDF entry."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import shutil
import subprocess
from pathlib import Path
from collections.abc import Mapping

LOG = logging.getLogger("stage7_5.aizynth")


def nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_asset_manifest(path: Path, label: str) -> dict:
    if not nonempty(path):
        raise SystemExit(f"{label} is required and must be a non-empty JSON file: {path}")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{label} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(payload, Mapping) or not isinstance(payload.get("artifacts"), list):
        raise SystemExit(f"{label} must contain an artifacts list: {path}")
    if not payload["artifacts"]:
        raise SystemExit(f"{label} contains no artifacts: {path}")
    checked = []
    for index, item in enumerate(payload["artifacts"]):
        if isinstance(item, str):
            raw_path = item
            expected_hash = None
            name = item
        elif isinstance(item, Mapping):
            raw_path = item.get("path")
            expected_hash = item.get("sha256")
            name = str(item.get("name") or raw_path or index)
        else:
            raise SystemExit(f"{label} artifact {index} must be a path or object: {path}")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise SystemExit(f"{label} artifact {index} has no path: {path}")
        artifact = Path(raw_path)
        if not artifact.is_absolute():
            artifact = path.parent / artifact
        artifact = artifact.resolve(strict=False)
        if not nonempty(artifact):
            raise SystemExit(f"{label} artifact {name} is missing or empty: {artifact}")
        observed = sha256_file(artifact)
        if expected_hash is not None and str(expected_hash).lower() != observed:
            raise SystemExit(
                f"{label} artifact hash mismatch for {name}: "
                f"expected {expected_hash}, observed {observed}"
            )
        checked.append({"name": name, "path": str(artifact), "sha256": observed})
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "artifacts": checked}


def run_aizynth(
    smiles: str,
    out_dir: Path,
    iters: int,
    time_limit: int,
    max_xform: int,
    *,
    executable: str = "aizynthcli",
    config_path: Path | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = out_dir / "aizynth.yml"
    tmp_cfg = cfg.with_suffix(cfg.suffix + ".tmp")
    if config_path is None:
        tmp_cfg.write_text(_render_config(iters, time_limit, max_xform))
    else:
        tmp_cfg.write_bytes(config_path.read_bytes())
    tmp_cfg.replace(cfg)
    out_json = out_dir / "routes.json"
    if out_json.exists():
        out_json.unlink()
    cmd = [executable, "--config", str(cfg),
           "--smiles", smiles, "--output", str(out_json)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"aizynthcli failed for SMILES {smiles!r} with exit code "
            f"{res.returncode}: {res.stderr.strip()}"
        )
    if not out_json.exists():
        raise RuntimeError(
            f"aizynthcli did not produce routes JSON for SMILES {smiles!r}: "
            f"{out_json}"
        )
    if out_json.stat().st_size == 0:
        raise RuntimeError(
            f"aizynthcli produced an empty routes JSON for SMILES {smiles!r}: "
            f"{out_json}"
        )
    try:
        payload = json.loads(out_json.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"aizynthcli produced invalid routes JSON for SMILES {smiles!r}: {out_json}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"aizynthcli routes JSON is not an object for SMILES {smiles!r}: "
            f"{out_json}"
        )
    if not isinstance(payload.get("routes"), list):
        raise RuntimeError(
            f"aizynthcli routes JSON missing list field 'routes' for SMILES {smiles!r}: {out_json}"
        )
    return payload


def _render_config(iters: int, time_limit: int, max_xform: int) -> str:
    return f"""\
policy:
  files:
    uspto: data/aizynth/uspto_model.onnx
filter:
  files:
    uspto_filter: data/aizynth/uspto_filter_model.onnx
stock:
  files:
    zinc: data/aizynth/zinc_stock.hdf5
properties:
  iteration_limit: {iters}
  return_first: false
  time_limit: {time_limit}
  max_transforms: {max_xform}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-sdf", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--iteration-limit", type=int, default=100)
    parser.add_argument("--time-limit", type=int, default=120)
    parser.add_argument("--max-transforms", type=int, default=6)
    parser.add_argument("--executable", default="aizynthcli")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model-manifest", type=Path)
    parser.add_argument("--stock-manifest", type=Path)
    parser.add_argument("--out-manifest", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    from rdkit import Chem
    if args.out_manifest.exists():
        args.out_manifest.unlink()
    provenance_path = args.out_dir / "run_provenance.json"
    provenance_path.unlink(missing_ok=True)
    if not nonempty(args.in_sdf):
        raise SystemExit(
            f"AiZynth input SDF is required and must be non-empty: {args.in_sdf}"
        )
    if bool(args.model_manifest) != bool(args.stock_manifest):
        raise SystemExit("--model-manifest and --stock-manifest must be provided together")
    if args.config is not None and not nonempty(args.config):
        raise SystemExit(f"AiZynthFinder config is required and must be non-empty: {args.config}")
    if not shutil.which(args.executable):
        raise SystemExit(f"{args.executable} is not available on PATH")

    asset_provenance = {
        "config": (
            {"path": str(args.config.resolve()), "sha256": sha256_file(args.config)}
            if args.config is not None
            else None
        ),
        "model_manifest": (
            validate_asset_manifest(args.model_manifest, "AiZynthFinder model manifest")
            if args.model_manifest is not None
            else None
        ),
        "stock_manifest": (
            validate_asset_manifest(args.stock_manifest, "AiZynthFinder stock manifest")
            if args.stock_manifest is not None
            else None
        ),
        "executable": args.executable,
        "iteration_limit": args.iteration_limit,
        "time_limit": args.time_limit,
        "max_transforms": args.max_transforms,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sup = Chem.SDMolSupplier(str(args.in_sdf), removeHs=False)

    rows: list[list[object]] = []
    n_valid = 0
    for i, mol in enumerate(sup, start=1):
        if mol is None:
            continue
        n_valid += 1
        smi = Chem.MolToSmiles(mol, canonical=True)
        target_dir = args.out_dir / f"analog_{i:04d}"
        try:
            payload = run_aizynth(smi, target_dir,
                                  args.iteration_limit,
                                  args.time_limit,
                                  args.max_transforms,
                                  executable=args.executable,
                                  config_path=args.config)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        n_routes = len(payload["routes"])
        rows.append([
            f"analog_{i:04d}",
            smi,
            str(target_dir / "routes.json"),
            n_routes,
        ])

    if n_valid == 0:
        raise SystemExit(f"AiZynth input SDF contains no valid molecules: {args.in_sdf}")
    if not rows:
        raise SystemExit(f"AiZynth produced no route payloads: {args.in_sdf}")

    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(asset_provenance, indent=2, sort_keys=True) + "\n")

    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    tmp_manifest = args.out_manifest.with_suffix(args.out_manifest.suffix + ".tmp")
    with tmp_manifest.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["analog_id", "smiles", "routes_json", "n_routes"])
        w.writerows(rows)
    tmp_manifest.replace(args.out_manifest)
    LOG.info("Wrote manifest → %s", args.out_manifest)


if __name__ == "__main__":
    main()
