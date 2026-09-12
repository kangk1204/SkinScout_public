#!/usr/bin/env python3
"""Build provenance-bound top-P2Rank pocket fragments for Foldseek."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from Bio.PDB import PDBIO, PDBParser, Select
from Bio.PDB.Polypeptide import is_aa


SCHEMA_VERSION = "skinscout.pocket-fragment-universe.v1"
TARGET_CLUSTER_SCHEMA_VERSION = "skinscout.target-cluster-map.v2"
# The screenable proteome map carries the same artifact contract under its own
# schema name. Accepting only the evidence map confined the pocket-fragment
# universe to targets that already had ligand evidence, which is exactly the
# population cross-cluster transfer cannot help.
SCREENABLE_TARGET_CLUSTER_SCHEMA_VERSION = "skinscout.screenable-target-cluster-map.v2"
SUPPORTED_TARGET_CLUSTER_SCHEMA_VERSIONS = (
    TARGET_CLUSTER_SCHEMA_VERSION,
    SCREENABLE_TARGET_CLUSTER_SCHEMA_VERSION,
)
INDEX_FIELDS = [
    "uniprot",
    "pocket_rank",
    "pocket_score",
    "seed_residue_count",
    "fragment_residue_count",
    "source_pdb_sha256",
    "source_prediction_sha256",
    "fragment_path",
    "fragment_sha256",
]
EXCLUSION_FIELDS = ["uniprot", "reason"]
RESIDUE_ID_RE = re.compile(r"([A-Za-z0-9])_(-?\d+[A-Za-z]?)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    if root.exists():
        for path in sorted(
            p
            for p in root.rglob("*")
            if p.is_file() and p.name != ".snakemake_timestamp"
        ):
            rel = path.relative_to(root).as_posix()
            digest.update(rel.encode("utf-8"))
            digest.update(b"\0")
            digest.update(_sha256(path).encode("ascii"))
            digest.update(b"\0")
    return digest.hexdigest()


def _require_file(path: Path, label: str) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"{label} is required and must be non-empty: {path}")


def _read_targets(path: Path) -> list[str]:
    _require_file(path, "targets CSV")
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise SystemExit(f"Unable to read targets CSV: {path}") from exc
    frame.columns = [str(col).strip() for col in frame.columns]
    if "uniprot" not in frame.columns:
        raise SystemExit(f"targets CSV missing required column 'uniprot': {path}")
    values = frame["uniprot"].fillna("").astype(str).str.strip()
    if values.eq("").any():
        raise SystemExit("targets CSV column 'uniprot' contains blank values")
    if values.duplicated().any():
        raise SystemExit("targets CSV contains duplicate uniprot values")
    return sorted(values.tolist())


def _read_manifest(path: Path, targets_csv: Path, target_count: int) -> dict[str, Any]:
    _require_file(path, "target cluster manifest")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid target cluster manifest JSON: {path}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version")
        not in SUPPORTED_TARGET_CLUSTER_SCHEMA_VERSIONS
    ):
        allowed = " or ".join(SUPPORTED_TARGET_CLUSTER_SCHEMA_VERSIONS)
        raise SystemExit(
            f"target cluster manifest schema must be {allowed}: {path}"
        )
    artifact = payload.get("artifact")
    if not isinstance(artifact, dict):
        raise SystemExit("target cluster manifest missing artifact object")
    expected_sha = str(artifact.get("sha256") or "").strip()
    if expected_sha != _sha256(targets_csv):
        raise SystemExit("target cluster manifest artifact sha256 does not match --targets-csv")
    if artifact.get("rows") != target_count:
        raise SystemExit("target cluster manifest artifact rows does not match --targets-csv")
    return payload


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or type(value).__name__ == "bool_":
        raise ValueError(f"{label} must be numeric")
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    return parsed


def _positive_int(value: object, label: str) -> int:
    parsed = _finite_float(value, label)
    if parsed < 1 or not parsed.is_integer():
        raise ValueError(f"{label} must be a positive integer")
    return int(parsed)


def _parse_seed_ids(value: object) -> set[tuple[str, str]]:
    text = "" if value is None else str(value).strip()
    if not text:
        raise ValueError("residue_ids must be nonblank")
    seeds = {(chain, residue) for chain, residue in RESIDUE_ID_RE.findall(text)}
    if not seeds:
        raise ValueError("residue_ids contains no parseable chain_residue identifiers")
    return seeds


def _read_top_pocket(
    path: Path,
) -> tuple[int, float, set[tuple[str, str]]] | None:
    _require_file(path, "P2Rank prediction CSV")
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise SystemExit(f"Unable to read P2Rank prediction CSV: {path}") from exc
    frame.columns = [str(col).strip() for col in frame.columns]
    required = {"rank", "score", "residue_ids"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"P2Rank prediction CSV missing required columns {missing}: {path}")
    if frame.empty:
        return None
    ranks: list[int] = []
    for value in frame["rank"]:
        try:
            ranks.append(_positive_int(value, "rank"))
        except ValueError as exc:
            raise SystemExit(f"Invalid P2Rank rank in {path.name}: {exc}") from exc
    rank_one_rows = [idx for idx, rank in enumerate(ranks) if rank == 1]
    if len(rank_one_rows) != 1:
        raise SystemExit(
            f"P2Rank prediction CSV must contain exactly one rank=1 row: {path}"
        )
    row = frame.iloc[rank_one_rows[0]]
    try:
        score = _finite_float(row["score"], "score")
        seeds = _parse_seed_ids(row["residue_ids"])
    except ValueError as exc:
        raise SystemExit(f"Invalid rank=1 P2Rank row in {path.name}: {exc}") from exc
    return 1, score, seeds


def _residue_token(residue: Any) -> str:
    seq_id = residue.id[1]
    insertion = str(residue.id[2]).strip()
    return f"{seq_id}{insertion}"


def _eligible_residues(structure: Any) -> dict[str, list[Any]]:
    by_chain: dict[str, list[Any]] = {}
    model = next(structure.get_models(), None)
    if model is None:
        raise SystemExit("PDB contains no models")
    for chain in model:
        residues = [
            residue
            for residue in chain
            if residue.id[0] == " " and is_aa(residue, standard=True) and "CA" in residue
        ]
        if residues:
            by_chain[chain.id] = residues
    return by_chain


def _selected_residues(
    pdb_path: Path,
    seeds: set[tuple[str, str]],
    context_residues: int,
) -> tuple[Any, set[tuple[str, tuple[Any, ...]]]]:
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    except Exception as exc:
        raise SystemExit(f"Unable to parse PDB: {pdb_path}") from exc
    by_chain = _eligible_residues(structure)
    selected: set[tuple[str, tuple[Any, ...]]] = set()
    for chain_id, residue_text in sorted(seeds):
        residues = by_chain.get(chain_id)
        if not residues:
            raise SystemExit(
                f"Declared seed residue {chain_id}_{residue_text} is missing from {pdb_path.name}"
            )
        positions = {
            _residue_token(residue): index for index, residue in enumerate(residues)
        }
        if residue_text not in positions:
            raise SystemExit(
                f"Declared seed residue {chain_id}_{residue_text} is missing from {pdb_path.name}"
            )
        center = positions[residue_text]
        start = max(0, center - context_residues)
        stop = min(len(residues), center + context_residues + 1)
        for residue in residues[start:stop]:
            selected.add((chain_id, residue.id))
    return structure, selected


class FragmentSelect(Select):
    def __init__(self, selected: set[tuple[str, tuple[Any, ...]]]) -> None:
        self.selected = selected

    def accept_model(self, model: Any) -> bool:
        return model.id == 0

    def accept_residue(self, residue: Any) -> bool:
        return (residue.get_parent().id, residue.id) in self.selected


def _write_fragment(structure: Any, selected: set[tuple[str, tuple[Any, ...]]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(path), FragmentSelect(selected))


def _validate_fragment(path: Path, expected: set[tuple[str, tuple[Any, ...]]]) -> None:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(path.stem, str(path))
    observed: set[tuple[str, tuple[Any, ...]]] = set()
    for residues in _eligible_residues(structure).values():
        for residue in residues:
            observed.add((residue.get_parent().id, residue.id))
            ca_atoms = [atom for atom in residue if atom.id == "CA"]
            if len(ca_atoms) != 1:
                raise SystemExit(f"Fragment residue does not contain exactly one CA atom: {path}")
    if observed != expected:
        raise SystemExit(f"Fragment residue set mismatch after writing: {path}")


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _clean_outputs(
    out_dir: Path,
    out_index: Path,
    out_exclusions: Path,
    out_manifest: Path,
) -> None:
    for path in (out_index, out_exclusions, out_manifest):
        path.unlink(missing_ok=True)
        path.with_name(f".{path.name}.tmp").unlink(missing_ok=True)
    if out_dir.exists():
        if not out_dir.is_dir():
            out_dir.unlink()
        else:
            shutil.rmtree(out_dir)


def _publish_file(staged: Path, final: Path) -> None:
    final.parent.mkdir(parents=True, exist_ok=True)
    staged.replace(final)


def _target_inputs(uid: str, pdb_dir: Path, pocket_dir: Path) -> tuple[Path, Path]:
    return pdb_dir / f"{uid}_clean.pdb", pocket_dir / f"{uid}_clean.pdb_predictions.csv"


def build_fragments(args: argparse.Namespace) -> None:
    if args.context_residues < 0:
        raise SystemExit("--context-residues must be >= 0")
    if args.min_fragment_residues < 1:
        raise SystemExit("--min-fragment-residues must be >= 1")
    if not str(args.p2rank_version).strip():
        raise SystemExit("--p2rank-version must be nonblank")
    _require_file(args.p2rank_params, "P2Rank params")

    _clean_outputs(args.out_dir, args.out_index, args.out_exclusions, args.out_manifest)
    staging_dir = args.out_dir.with_name(f".{args.out_dir.name}.staging")
    staging_index = args.out_index.with_name(f".{args.out_index.name}.tmp")
    staging_exclusions = args.out_exclusions.with_name(f".{args.out_exclusions.name}.tmp")
    staging_manifest = args.out_manifest.with_name(f".{args.out_manifest.name}.tmp")
    shutil.rmtree(staging_dir, ignore_errors=True)
    for path in (staging_index, staging_exclusions, staging_manifest):
        path.unlink(missing_ok=True)

    try:
        targets = _read_targets(args.targets_csv)
        target_manifest = _read_manifest(
            args.target_cluster_manifest,
            args.targets_csv,
            len(targets),
        )

        index_rows: list[dict[str, object]] = []
        exclusions: list[dict[str, object]] = []
        fragment_dir = staging_dir / "fragments"
        fragment_dir.mkdir(parents=True, exist_ok=True)
        for uid in targets:
            pdb_path, prediction_path = _target_inputs(uid, args.pdb_dir, args.pocket_dir)
            pdb_exists = pdb_path.exists()
            prediction_exists = prediction_path.exists()
            if not pdb_exists and not prediction_exists:
                exclusions.append({"uniprot": uid, "reason": "missing_structure_and_pocket"})
                continue
            if pdb_exists != prediction_exists:
                missing = "PDB" if not pdb_exists else "P2Rank prediction CSV"
                raise SystemExit(f"{missing} missing for {uid}; refusing partial input")
            _require_file(pdb_path, "PDB")
            _require_file(prediction_path, "P2Rank prediction CSV")

            top_pocket = _read_top_pocket(prediction_path)
            if top_pocket is None:
                exclusions.append({"uniprot": uid, "reason": "no_ranked_pocket"})
                continue
            rank, score, seeds = top_pocket
            structure, selected = _selected_residues(pdb_path, seeds, args.context_residues)
            if len(selected) < args.min_fragment_residues:
                exclusions.append({"uniprot": uid, "reason": "fragment_below_minimum"})
                continue
            fragment_path = fragment_dir / f"{uid}_p2rank_top1_context{args.context_residues}.pdb"
            _write_fragment(structure, selected, fragment_path)
            _validate_fragment(fragment_path, selected)
            index_rows.append(
                {
                    "uniprot": uid,
                    "pocket_rank": rank,
                    "pocket_score": f"{score:.12g}",
                    "seed_residue_count": len(seeds),
                    "fragment_residue_count": len(selected),
                    "source_pdb_sha256": _sha256(pdb_path),
                    "source_prediction_sha256": _sha256(prediction_path),
                    "fragment_path": str(fragment_path.relative_to(staging_dir)),
                    "fragment_sha256": _sha256(fragment_path),
                }
            )

        _write_csv(staging_index, INDEX_FIELDS, index_rows)
        _write_csv(staging_exclusions, EXCLUSION_FIELDS, exclusions)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "definition": (
                "Local top-P2Rank pocket-fragment universe for downstream Foldseek; "
                "fragments are pocket-context residues, not whole-protein similarity inputs."
            ),
            "target_manifest_provenance": {
                "path": str(args.target_cluster_manifest.resolve()),
                "sha256": _sha256(args.target_cluster_manifest),
                "schema_version": target_manifest["schema_version"],
                "artifact": {
                    "path": str(args.targets_csv.resolve()),
                    "sha256": _sha256(args.targets_csv),
                    "rows": len(targets),
                },
            },
            "p2rank": {
                "params_path": str(args.p2rank_params.resolve()),
                "params_sha256": _sha256(args.p2rank_params),
                "version": str(args.p2rank_version).strip(),
            },
            "extraction_contract": {
                "source_pdb_name": "<uniprot>_clean.pdb",
                "source_prediction_name": "<uniprot>_clean.pdb_predictions.csv",
                "pocket_rank": 1,
                "context_residues": args.context_residues,
                "min_fragment_residues": args.min_fragment_residues,
                "residue_policy": "standard amino-acid residues with CA atoms only",
                "missing_both_policy": "audited exclusion missing_structure_and_pocket",
                "missing_one_policy": "fail closed",
            },
            "counts": {
                "targets": len(targets),
                "accepted": len(index_rows),
                "excluded": len(exclusions),
            },
            "artifacts": {
                "out_dir": str(args.out_dir.resolve()),
                "out_dir_tree_sha256": _tree_digest(staging_dir),
                "index": {
                    "path": str(args.out_index.resolve()),
                    "sha256": _sha256(staging_index),
                    "rows": len(index_rows),
                },
                "exclusions": {
                    "path": str(args.out_exclusions.resolve()),
                    "sha256": _sha256(staging_exclusions),
                    "rows": len(exclusions),
                },
            },
        }
        _write_json(staging_manifest, manifest)

        staging_dir.replace(args.out_dir)
        _publish_file(staging_index, args.out_index)
        _publish_file(staging_exclusions, args.out_exclusions)
        _publish_file(staging_manifest, args.out_manifest)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        _clean_outputs(args.out_dir, args.out_index, args.out_exclusions, args.out_manifest)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets-csv", required=True, type=Path)
    parser.add_argument("--target-cluster-manifest", required=True, type=Path)
    parser.add_argument("--pdb-dir", required=True, type=Path)
    parser.add_argument("--pocket-dir", required=True, type=Path)
    parser.add_argument("--p2rank-params", required=True, type=Path)
    parser.add_argument("--p2rank-version", required=True)
    parser.add_argument("--context-residues", type=int, default=2)
    parser.add_argument("--min-fragment-residues", type=int, default=12)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--out-index", required=True, type=Path)
    parser.add_argument("--out-exclusions", required=True, type=Path)
    parser.add_argument("--out-manifest", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        build_fragments(parse_args(argv))
    except SystemExit as exc:
        if isinstance(exc.code, str):
            print(exc.code, file=sys.stderr)
            return 1
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
