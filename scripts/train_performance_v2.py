#!/usr/bin/env python3
"""Train the additive performance-v2 pair model."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from performance_v2_model import (  # noqa: E402
    BUDGET_SCHEMA,
    ESM2_CHECKPOINT,
    ESM2_LICENSE,
    MAX_PEAK_VRAM_GIB,
    MAX_TRAINABLE_PARAMS,
    MODEL_SCHEMA,
    MOLFORMER_CHECKPOINT,
    MOLFORMER_LICENSE,
    PRODUCTION_SEEDS,
    append_only_outputs,
    artifact_record,
    sha256_file,
    sha256_payload,
    train_fixture_model,
    train_torch_pair_model,
    utc_now,
    validate_budget_manifest,
    validate_model_manifest,
    write_json_atomic,
)


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if len(seeds) != 3:
        raise argparse.ArgumentTypeError("expected exactly 3 comma-separated seeds")
    return seeds


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--embedding-manifest", required=True, type=Path)
    parser.add_argument("--out-model", required=True, type=Path)
    parser.add_argument("--out-manifest", required=True, type=Path)
    parser.add_argument("--out-budget", required=True, type=Path)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seeds", type=parse_seeds, default=PRODUCTION_SEEDS)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        raise SystemExit("--learning-rate must be > 0")
    if not args.fixture:
        if tuple(args.seeds) != PRODUCTION_SEEDS:
            raise SystemExit(f"production --seeds must be exactly {PRODUCTION_SEEDS}")
        if args.out_model.suffix != ".pt":
            raise SystemExit("non-fixture performance-v2 training writes a .pt model artifact")
        if args.epochs < 1:
            raise SystemExit("--epochs must be >= 1")
        if args.projection_dim < 1:
            raise SystemExit("--projection-dim must be >= 1")
    with append_only_outputs((args.out_model, args.out_manifest, args.out_budget)):
        return _run(args)


def _run(args: argparse.Namespace) -> int:
    if args.fixture:
        fit = train_fixture_model(
            args.train,
            args.embedding_manifest,
            args.out_model,
            seeds=args.seeds,
        )
        model_kind = "performance_v2_projection_gated_dual_encoder_fixture"
    else:
        fit = train_torch_pair_model(
            args.train,
            args.embedding_manifest,
            args.out_model,
            seeds=args.seeds,
            epochs=args.epochs,
            projection_dim=args.projection_dim,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            device_mode="cpu" if args.cpu else "cuda",
        )
        model_kind = "performance_v2_projection_gated_dual_encoder_torch"
    budget = {
        "schema_version": BUDGET_SCHEMA,
        "created_at": utc_now(),
        "execution_mode": fit["execution_mode"],
        "peak_vram_gib": float(fit["peak_vram_gib"]),
        "peak_vram_source": fit["peak_vram_source"],
        "peak_vram_measured": bool(fit["peak_vram_measured"]),
        "max_peak_vram_gib": MAX_PEAK_VRAM_GIB,
        "trainable_params": int(fit["trainable_params"]),
        "max_trainable_params": MAX_TRAINABLE_PARAMS,
        "fallback_model_ids": [
            MOLFORMER_CHECKPOINT,
            ESM2_CHECKPOINT,
        ],
        "license_profile": {
            "redistribution": "model head and derived embeddings only",
            "model_licenses": {
                MOLFORMER_CHECKPOINT: MOLFORMER_LICENSE,
                f"facebook/{ESM2_CHECKPOINT}": ESM2_LICENSE,
            },
        },
    }
    write_json_atomic(budget, args.out_budget)
    validate_budget_manifest(args.out_budget)
    manifest = {
        "schema_version": MODEL_SCHEMA,
        "created_at": utc_now(),
        "model_kind": model_kind,
        "training_policy": {
            "pu_policy": "BCE/calibration use explicit measured positives/negatives only; gray/unmeasured are ranking-only",
            "contrastive": "multi-positive with unlabeled negatives marked ranking-only",
            "calibration": "train-only SHA-256 pair-hash/target-stratified holdout excluded from head fit",
            "seeds": list(args.seeds),
            "prior_debias": (
                "target-balanced measured-event fitting followed by unweighted train-only "
                "measured holdout calibration; no deployment-prior correction applied"
            ),
            "frozen_cached_embeddings": True,
            "encoder_requires_grad": False,
            "encoder_no_grad": True,
            "head": "separate ligand/target projections to common latent retrieval space plus interaction event head",
            "seed_execution": "sequential independent heads; scoring and calibration use mean predictions",
            "minibatching": (
                "deterministic ligand-shard-local DataLoader indexes with globally normalized "
                "epoch objective and bounded sampled contrastive targets"
            ),
            "target_embedding_cache": "all target shards cached after first access",
        },
        "training_hyperparameters": {
            "epochs": int(args.epochs),
            "projection_dim": int(args.projection_dim),
            "learning_rate": float(args.learning_rate),
            "requested_batch_size": int(args.batch_size),
            "actual_batch_size": fit["actual_batch_size"],
            "mixed_precision": bool(fit.get("mixed_precision", False)),
            "objective_normalization": "global_weight_mass_one_optimizer_step_per_epoch",
            "target_cache_shards": fit.get("target_cache_shards"),
        },
        "inputs": {
            "train_csv": {
                "path": str(args.train.resolve()),
                "sha256": sha256_file(args.train),
            },
            "embedding_manifest": artifact_record(args.embedding_manifest),
            "budget_manifest": artifact_record(args.out_budget),
        },
        "model_artifact": artifact_record(args.out_model),
        "model_state_sha256": fit["model_state_sha256"],
        "trainable_params": int(fit["trainable_params"]),
        "effective_target_count": float(fit["effective_target_count"]),
        "ranking_only_pairs": int(fit["ranking_only_pairs"]),
        "ranking_only_pairs_used": int(fit.get("ranking_only_pairs_used", 0)),
        "train_ligands": fit["train_ligands"],
        "train_targets": fit["train_targets"],
        "train_domain_sha256": sha256_payload(
            {
                "train_ligands": fit["train_ligands"],
                "train_targets": fit["train_targets"],
            }
        ),
        "calibration": fit["calibration"],
        "calibration_split": fit["calibration_split"],
    }
    write_json_atomic(manifest, args.out_manifest)
    validate_model_manifest(args.out_manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
