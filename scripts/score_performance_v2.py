#!/usr/bin/env python3
"""Score all target embeddings with an additive performance-v2 model."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from performance_v2_model import (  # noqa: E402
    append_only_outputs,
    artifact_record,
    compute_query_ligand_features,
    fixture_query_ligand_features,
    ranking_manifest,
    score_fixture_model,
    score_fixture_query_vector,
    score_torch_pair_model,
    score_torch_query_vector,
    validate_artifact,
    validate_model_manifest,
    validate_ranking_manifest,
    write_csv_atomic,
    write_json_atomic,
)


def _smiles_from_compound_json(path: Path) -> tuple[str, dict[str, object]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"--compound-json is missing or empty: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--compound-json is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"--compound-json root must be an object: {path}")
    for field in ("canonical_smiles", "input_canonical_smiles", "input_smiles"):
        value = str(payload.get(field) or "").strip()
        if value:
            provenance = {
                "compound_json": artifact_record(path),
                "compound_json_smiles_field": field,
            }
            return value, provenance
    raise SystemExit(
        "--compound-json must contain canonical_smiles, input_canonical_smiles, or input_smiles"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-manifest", required=True, type=Path)
    query = parser.add_mutually_exclusive_group(required=True)
    query.add_argument("--ligand-id")
    query.add_argument("--smiles")
    query.add_argument("--compound-json", type=Path)
    parser.add_argument("--out-ranking", required=True, type=Path)
    parser.add_argument("--out-manifest", required=True, type=Path)
    parser.add_argument("--min-score", type=float)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        help=(
            "Force the scoring device. Without it a visible GPU is preferred "
            "and CPU is used if that GPU turns out to be unusable."
        ),
    )
    parser.add_argument(
        "--query-max-length",
        type=int,
        help="Must equal the max token length recorded by the embedding manifest.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.query_max_length is not None and args.query_max_length < 1:
        raise SystemExit("--query-max-length must be >= 1")
    if args.min_score is not None and not math.isfinite(args.min_score):
        raise SystemExit("--min-score must be finite")
    with append_only_outputs((args.out_ranking, args.out_manifest)):
        return _run(args)


def _run(args: argparse.Namespace) -> int:
    model_manifest = validate_model_manifest(args.model_manifest)
    model_path = Path(model_manifest["model_artifact"]["path"])
    embedding_manifest = Path(model_manifest["inputs"]["embedding_manifest"]["path"])
    query_provenance: dict[str, object] = {}
    if args.compound_json is not None:
        query_smiles, compound_provenance = _smiles_from_compound_json(args.compound_json)
        query_provenance.update(compound_provenance)
    else:
        query_smiles = args.smiles

    if args.ligand_id:
        query_id = args.ligand_id
        query_provenance.update({
            "input_type": "ligand_id",
            "embedding": {"source": "bound_cached_ligand_embedding"},
        })
    elif model_path.suffix == ".npz":
        ligand_artifact = model_manifest["inputs"]["embedding_manifest"]
        embedding_manifest_payload = json.loads(Path(ligand_artifact["path"]).read_text(encoding="utf-8"))
        ligand_record = embedding_manifest_payload["artifacts"]["ligands"]
        ligand_path = validate_artifact(
            ligand_record,
            base_dir=Path(ligand_artifact["path"]).parent,
            label="fixture ligand embeddings",
        )
        query_id, query_vector, smiles_provenance = fixture_query_ligand_features(
            smiles=query_smiles,
            ligand_embedding_csv=ligand_path,
        )
        query_provenance.update(smiles_provenance)
    else:
        query_id, query_vector, smiles_provenance = compute_query_ligand_features(
            smiles=query_smiles,
            embedding_manifest_path=embedding_manifest,
            max_length=args.query_max_length,
        )
        query_provenance.update(smiles_provenance)
    query_provenance["scoring_policy"] = {"min_score": args.min_score}

    if model_path.suffix == ".npz" and args.ligand_id:
        ranking = score_fixture_model(
            ligand_id=query_id,
            model_npz=model_path,
            embedding_manifest_path=embedding_manifest,
            train_ligands=model_manifest.get("train_ligands", []),
            train_targets=model_manifest.get("train_targets", []),
            calibration=model_manifest.get("calibration", {}),
            min_score=args.min_score,
        )
    elif model_path.suffix == ".npz":
        ranking = score_fixture_query_vector(
            query_ligand_id=query_id,
            ligand_vector=query_vector,
            model_npz=model_path,
            embedding_manifest_path=embedding_manifest,
            train_ligands=model_manifest.get("train_ligands", []),
            train_targets=model_manifest.get("train_targets", []),
            calibration=model_manifest.get("calibration", {}),
            min_score=args.min_score,
        )
    elif args.ligand_id:
        ranking = score_torch_pair_model(
            ligand_id=query_id,
            model_pt=model_path,
            embedding_manifest_path=embedding_manifest,
            train_ligands=model_manifest.get("train_ligands", []),
            train_targets=model_manifest.get("train_targets", []),
            calibration=model_manifest.get("calibration", {}),
            min_score=args.min_score,
            device_mode=args.device,
        )
    else:
        ranking = score_torch_query_vector(
            query_ligand_id=query_id,
            ligand_vector=query_vector,
            model_pt=model_path,
            embedding_manifest_path=embedding_manifest,
            train_ligands=model_manifest.get("train_ligands", []),
            train_targets=model_manifest.get("train_targets", []),
            calibration=model_manifest.get("calibration", {}),
            min_score=args.min_score,
            device_mode=args.device,
        )
    write_csv_atomic(ranking, args.out_ranking)
    payload = ranking_manifest(
        ranking_csv=args.out_ranking,
        model_manifest=args.model_manifest,
        query_ligand_id=query_id,
        rows=len(ranking),
        query_provenance=query_provenance,
    )
    write_json_atomic(payload, args.out_manifest)
    validate_ranking_manifest(args.out_manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
