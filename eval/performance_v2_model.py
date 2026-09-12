#!/usr/bin/env python3
"""Additive performance-v2 scientific model contracts and fixture model.

This module is intentionally import-light: it uses only the cosmax-base stack.
Torch, transformers, and model checkpoints are imported by CLI execution paths
only. The production path trains a small projection head over frozen cached
embeddings; the deterministic fixture path below exercises the same contracts
without downloads or GPU requirements.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import re
import shutil
import warnings
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence, Set
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

INPUTS_SCHEMA = "skinscout.performance-v2-inputs.v1"
EMBEDDING_SCHEMA = "skinscout.performance-v2-embeddings.v1"
MODEL_SCHEMA = "skinscout.performance-v2-model.v1"
BUDGET_SCHEMA = "skinscout.performance-v2-budget.v1"
RANKING_SCHEMA = "skinscout.performance-v2-ranking.v1"
CALIBRATION_SCHEMA = "skinscout.performance-v2-calibration.v1"

MOLFORMER_CHECKPOINT = "ibm-research/MoLFormer-XL-both-10pct"
MOLFORMER_REVISION = "compat-v4"
MOLFORMER_LICENSE = "apache-2.0"
ESM2_CHECKPOINT = "esm2_t30_150M_UR50D"
ESM2_REVISION = "main"
ESM2_LICENSE = "mit"

MAX_TRAINABLE_PARAMS = 10_000_000
MAX_PEAK_VRAM_GIB = 11.5
PRODUCTION_SEEDS = (17, 42, 73)
ECFP_BITS = 2048
ECFP_RADIUS = 2
PROTEIN_WINDOW_OVERLAP = 128
MOLFORMER_POOLING = "upstream_pooler_else_attention_masked_mean"
MOLFORMER_PRECISION = "fp32_model_weights_and_forward"
PROTEIN_POOLING = "last_hidden_state_residue_mean_excluding_bos_eos_pad"
PROTEIN_WINDOW_STRATEGY = "deterministic_overlapping_residue_windows_length_weighted_mean"
HF_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
INPUT_ARTIFACT_COLUMNS = {
    "train": ["ligand_id", "target_id", "evidence_state", "measured_label", "ontology"],
    "ligands": ["ligand_id", "smiles"],
    "targets": ["target_id", "sequence", "target_cluster_30", "target_cluster_50"],
}

ONTOLOGY = (
    "direct_binding_reversible",
    "functional_modulation",
    "cofactor_substrate",
    "reactive_covalent_sensor",
    "metabolite_prodrug",
    "pathway_effect",
    "unknown_mixed",
)
OOD_ROUTES = (
    "in_domain",
    "ligand_cold",
    "target_cold",
    "dual_cold",
    "structure_abstained",
)


class ContractError(ValueError):
    """Raised when a performance-v2 fail-closed contract is violated."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ContractError(f"{label} is required and must be non-empty: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContractError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ContractError(f"{label} root must be an object: {path}")
    return payload


def write_json_atomic(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def _output_candidates(path: Path) -> tuple[Path, ...]:
    return path, path.with_name(path.name + ".tmp")


def require_new_outputs(paths: Iterable[Path]) -> tuple[Path, ...]:
    outputs = tuple(Path(path) for path in paths)
    normalized = tuple(path.resolve(strict=False) for path in outputs)
    if len(set(normalized)) != len(normalized):
        raise ContractError("output paths must be distinct")
    for index, path in enumerate(normalized):
        for other in normalized[index + 1 :]:
            if path in other.parents or other in path.parents:
                raise ContractError("output paths must not contain one another")
    for output in outputs:
        for candidate in _output_candidates(output):
            if candidate.exists() or candidate.is_symlink():
                raise ContractError(f"refusing to overwrite pre-existing output: {candidate}")
    return outputs


def cleanup_new_outputs(paths: Iterable[Path]) -> None:
    for output in paths:
        for candidate in reversed(_output_candidates(Path(output))):
            if candidate.is_symlink() or candidate.is_file():
                candidate.unlink(missing_ok=True)
            elif candidate.is_dir():
                shutil.rmtree(candidate)


@contextmanager
def append_only_outputs(paths: Iterable[Path]):
    outputs = require_new_outputs(paths)
    try:
        yield outputs
    except BaseException:
        cleanup_new_outputs(outputs)
        raise


def load_transformer_fp32_then_place(
    auto_model: Any,
    *,
    model_id: str,
    revision: str,
    trust_remote_code: bool,
    device: Any,
    model_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    """Instantiate checkpoint code in fp32, then move it without a model-wide cast."""
    model = auto_model.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=trust_remote_code,
        **dict(model_kwargs or {}),
    )
    return model.to(device=device)


def artifact_record(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ContractError(f"artifact is missing or empty: {path}")
    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
    }
    if rows is not None:
        record["rows"] = int(rows)
    return record


def validate_artifact(record: Mapping[str, Any], *, base_dir: Path, label: str) -> Path:
    if not isinstance(record, Mapping):
        raise ContractError(f"{label} artifact record is missing")
    raw_path = str(record.get("path") or "").strip()
    if not raw_path:
        raise ContractError(f"{label} artifact path is missing")
    path = Path(raw_path)
    if not path.is_absolute():
        path = base_dir / path
    if not path.is_file() or path.stat().st_size == 0:
        raise ContractError(f"{label} artifact is missing or empty: {path}")
    if record.get("sha256") != sha256_file(path):
        raise ContractError(f"{label} artifact SHA-256 mismatch: {path}")
    if "bytes" in record and int(record["bytes"]) != path.stat().st_size:
        raise ContractError(f"{label} artifact byte-count mismatch: {path}")
    return path


def validate_embedding_artifact(
    record: Mapping[str, Any], *, base_dir: Path, label: str
) -> None:
    fmt = str(record.get("format") or "csv")
    if fmt == "csv":
        validate_artifact(record, base_dir=base_dir, label=label)
        return
    if fmt != "npz_shards":
        raise ContractError(f"{label} embedding format is unsupported: {fmt}")
    if record.get("dtype") != "float32":
        raise ContractError(f"{label} npz_shards dtype must be float32")
    if int(record.get("dim", 0)) < 1:
        raise ContractError(f"{label} npz_shards dim must be positive")
    shards = record.get("shards")
    if not isinstance(shards, Sequence) or isinstance(shards, (str, bytes)) or not shards:
        raise ContractError(f"{label} npz_shards requires non-empty shards")
    total_rows = 0
    for idx, shard in enumerate(shards):
        shard_path = validate_artifact(shard, base_dir=base_dir, label=f"{label} shard {idx}")
        try:
            arrays = np.load(shard_path, allow_pickle=False)
        except Exception as exc:
            raise ContractError(f"{label} shard failed to load: {shard_path}") from exc
        with arrays:
            if "ids" not in arrays or "vectors" not in arrays:
                raise ContractError(f"{label} shard requires ids and vectors arrays")
            vectors = arrays["vectors"]
            ids = arrays["ids"]
            if vectors.dtype != np.float32:
                raise ContractError(f"{label} shard vectors must be float32")
            if vectors.ndim != 2 or vectors.shape[1] != int(record["dim"]):
                raise ContractError(f"{label} shard vector dimensions do not match manifest")
            if len(ids) != vectors.shape[0] or vectors.shape[0] < 1:
                raise ContractError(f"{label} shard ids/vectors row mismatch")
            if not np.isfinite(vectors).all():
                raise ContractError(f"{label} shard vectors must all be finite")
            ecfp_bits = int(record.get("ecfp_bits", 0))
            if ecfp_bits:
                ecfp_offset = int(record.get("ecfp_offset", -1))
                if ecfp_offset < 1 or ecfp_offset + ecfp_bits != vectors.shape[1]:
                    raise ContractError(f"{label} ECFP layout does not match vector dimensions")
                bit_values = vectors[:, ecfp_offset:]
                if not np.logical_or(bit_values == 0.0, bit_values == 1.0).all():
                    raise ContractError(f"{label} ECFP features must preserve independent binary bits")
            total_rows += int(vectors.shape[0])
    if int(record.get("rows", -1)) != total_rows:
        raise ContractError(f"{label} npz_shards row count mismatch")


def require_schema(payload: Mapping[str, Any], expected: str, label: str) -> None:
    if payload.get("schema_version") != expected:
        raise ContractError(f"{label} schema_version must be {expected}")


def _validate_provenance_record(record: object, label: str) -> None:
    if not isinstance(record, Mapping):
        raise ContractError(f"{label} artifact record is missing")
    if not str(record.get("path") or "").strip():
        raise ContractError(f"{label} artifact path is missing")
    if not SHA256_PATTERN.fullmatch(str(record.get("sha256") or "")):
        raise ContractError(f"{label} artifact SHA-256 is invalid")


def _validate_prepared_csv(
    path: Path,
    *,
    expected_columns: Sequence[str],
    expected_rows: int,
    label: str,
) -> None:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            columns = next(reader, None)
            if columns != list(expected_columns):
                raise ContractError(f"{label} columns do not match prepared schema")
            rows = sum(1 for _ in reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise ContractError(f"{label} could not be read as CSV: {path}") from exc
    if rows != expected_rows:
        raise ContractError(f"{label} row count does not match prepared manifest")


def validate_input_manifest(
    path: Path,
    *,
    verify_files: bool = True,
) -> dict[str, Any]:
    payload = read_json_object(path, "performance-v2 input manifest")
    require_schema(payload, INPUTS_SCHEMA, "performance-v2 input manifest")
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ContractError("performance-v2 input manifest missing raw inputs")
    for key in ("benchmark_train", "retrieval_ligands", "target_fasta", "target_clusters"):
        label = f"performance-v2 raw input {key}"
        if verify_files:
            validate_artifact(inputs.get(key), base_dir=path.parent, label=label)
        else:
            _validate_provenance_record(inputs.get(key), label)
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ContractError("performance-v2 input manifest missing artifacts")
    artifact_paths: dict[str, Path] = {}
    for key in INPUT_ARTIFACT_COLUMNS:
        record = artifacts.get(key)
        label = f"performance-v2 prepared {key}"
        if verify_files:
            artifact_paths[key] = validate_artifact(
                record,
                base_dir=path.parent,
                label=label,
            )
        else:
            _validate_provenance_record(record, label)
        if not isinstance(record, Mapping) or int(record.get("rows", 0)) < 1:
            raise ContractError(f"performance-v2 prepared {key} row count is invalid")
    schema = payload.get("schema")
    if not isinstance(schema, Mapping):
        raise ContractError("performance-v2 input manifest missing schema contract")
    for key, expected in INPUT_ARTIFACT_COLUMNS.items():
        if schema.get(f"{key}_csv") != expected:
            raise ContractError(f"performance-v2 prepared {key} columns changed")
        if verify_files:
            _validate_prepared_csv(
                artifact_paths[key],
                expected_columns=expected,
                expected_rows=int(artifacts[key]["rows"]),
                label=f"performance-v2 prepared {key}",
            )
    counts = payload.get("counts")
    if not isinstance(counts, Mapping):
        raise ContractError("performance-v2 input manifest missing counts")
    expected_counts = {
        "train": "train_rows",
        "ligands": "ligands",
        "targets": "targets",
    }
    for artifact_key, count_key in expected_counts.items():
        if int(counts.get(count_key, -1)) != int(artifacts[artifact_key]["rows"]):
            raise ContractError(
                f"performance-v2 input manifest {count_key} does not match artifact rows"
            )
    return payload


def validate_embedding_manifest(path: Path) -> dict[str, Any]:
    payload = read_json_object(path, "embedding manifest")
    require_schema(payload, EMBEDDING_SCHEMA, "embedding manifest")
    mode = str(payload.get("mode") or "")
    if mode not in {"fixture_precomputed", "frozen_checkpoint"}:
        raise ContractError("embedding manifest mode is invalid")
    checkpoints = payload.get("checkpoints")
    if not isinstance(checkpoints, Mapping):
        raise ContractError("embedding manifest missing checkpoints")
    ligand = checkpoints.get("ligand")
    protein = checkpoints.get("protein")
    if not isinstance(ligand, Mapping) or not isinstance(protein, Mapping):
        raise ContractError("embedding manifest requires ligand and protein checkpoints")
    if ligand.get("model_id") != MOLFORMER_CHECKPOINT or ligand.get("revision") != MOLFORMER_REVISION:
        raise ContractError("embedding manifest ligand checkpoint changed")
    if protein.get("model_id") != ESM2_CHECKPOINT or protein.get("revision") != ESM2_REVISION:
        raise ContractError("embedding manifest protein checkpoint changed")
    if mode == "frozen_checkpoint":
        for label, checkpoint in (("ligand", ligand), ("protein", protein)):
            resolved = str(checkpoint.get("resolved_revision") or "")
            if not HF_COMMIT_PATTERN.fullmatch(resolved):
                raise ContractError(
                    f"embedding manifest {label} resolved_revision must be an immutable 40-hex commit"
                )
        if protein.get("hf_model_id") != f"facebook/{ESM2_CHECKPOINT}":
            raise ContractError("embedding manifest protein Hugging Face model ID changed")
        if ligand.get("license_id") != MOLFORMER_LICENSE or protein.get("license_id") != ESM2_LICENSE:
            raise ContractError("embedding manifest checkpoint license IDs changed")
    license_profile = payload.get("license_profile")
    if not isinstance(license_profile, Mapping) or not license_profile.get("redistribution"):
        raise ContractError("embedding manifest license_profile.redistribution is required")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ContractError("embedding manifest missing artifacts")
    for key in ("ligands", "targets"):
        validate_embedding_artifact(artifacts.get(key), base_dir=path.parent, label=f"embedding {key}")
    if mode == "frozen_checkpoint":
        ligand_artifact = artifacts.get("ligands")
        target_artifact = artifacts.get("targets")
        if not isinstance(ligand_artifact, Mapping) or not isinstance(target_artifact, Mapping):
            raise ContractError("production embedding artifacts are missing")
        if ligand_artifact.get("format") != "npz_shards" or target_artifact.get("format") != "npz_shards":
            raise ContractError("production embeddings must use sharded NPZ artifacts")
        molformer_dim = int(ligand_artifact.get("molformer_dim", 0))
        if int(ligand_artifact.get("ecfp_bits", 0)) != ECFP_BITS:
            raise ContractError(f"production ligand embeddings require {ECFP_BITS} independent ECFP bits")
        if int(ligand_artifact.get("ecfp_radius", -1)) != ECFP_RADIUS:
            raise ContractError(f"production ligand embeddings require Morgan radius {ECFP_RADIUS}")
        if ligand_artifact.get("ecfp_encoding") != "independent_binary_float32":
            raise ContractError("production ligand ECFP encoding is invalid")
        if int(ligand_artifact.get("ecfp_offset", -1)) != molformer_dim:
            raise ContractError("production ligand ECFP offset must equal molformer_dim")
        if molformer_dim < 1 or int(ligand_artifact.get("dim", 0)) != molformer_dim + ECFP_BITS:
            raise ContractError("production ligand dimensions do not match MoLFormer plus ECFP bits")
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ContractError("embedding manifest missing inputs")
    for key in ("ligands", "targets"):
        validate_artifact(inputs.get(key), base_dir=path.parent, label=f"embedding input {key}")
    input_manifest_record = inputs.get("input_manifest")
    if input_manifest_record is None:
        if mode == "frozen_checkpoint":
            raise ContractError("production embedding manifest requires prepared input lineage")
    else:
        input_manifest_path = validate_artifact(
            input_manifest_record,
            base_dir=path.parent,
            label="embedding prepared input manifest",
        )
        input_manifest = validate_input_manifest(input_manifest_path, verify_files=False)
        for key in ("ligands", "targets"):
            if input_manifest["artifacts"][key].get("sha256") != inputs[key].get("sha256"):
                raise ContractError(
                    f"embedding input {key} does not match prepared input manifest"
                )
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping) or not provenance.get("deterministic_canonical_ids"):
        raise ContractError("embedding manifest requires deterministic canonical ID provenance")
    if mode == "frozen_checkpoint":
        pooling = provenance.get("pooling")
        if not isinstance(pooling, Mapping) or pooling.get("ligand") != MOLFORMER_POOLING:
            raise ContractError("production MoLFormer pooling contract changed")
        if pooling.get("protein") != PROTEIN_POOLING:
            raise ContractError("production ESM2 pooling must use residue-only last-hidden-state mean")
        if pooling.get("protein_add_pooling_layer") is not False:
            raise ContractError("production ESM2 must disable the untrained pooler")
        precision = provenance.get("precision")
        if not isinstance(precision, Mapping) or precision.get("ligand") != MOLFORMER_PRECISION:
            raise ContractError("production MoLFormer must use fp32 weights and forward")
        if precision.get("protein") not in {
            MOLFORMER_PRECISION,
            "fp32_model_weights_with_forward_autocast_fp16",
        }:
            raise ContractError("production ESM2 precision contract is invalid")
        ligand_tokenization = provenance.get("ligand_tokenization")
        if not isinstance(ligand_tokenization, Mapping):
            raise ContractError("production embedding manifest requires ligand tokenization evidence")
        if ligand_tokenization.get("strategy") != "tokenizer_truncation_fixed_max_tokens":
            raise ContractError("production ligand tokenization strategy changed")
        if int(ligand_tokenization.get("max_tokens", 0)) < 1:
            raise ContractError("production ligand tokenization max_tokens must be positive")
        if ligand_tokenization.get("truncation") is not True:
            raise ContractError("production ligand tokenization must record truncation=True")
        windowing = provenance.get("protein_windowing")
        if not isinstance(windowing, Mapping):
            raise ContractError("production embedding manifest requires protein windowing evidence")
        if windowing.get("strategy") != PROTEIN_WINDOW_STRATEGY:
            raise ContractError("production protein windowing strategy changed")
        max_tokens = int(windowing.get("max_tokens", 0))
        special_tokens = int(windowing.get("special_tokens_per_window", -1))
        residue_window = int(windowing.get("residue_window", 0))
        overlap = int(windowing.get("overlap_residues", -1))
        if max_tokens < 3 or special_tokens < 1 or residue_window != max_tokens - special_tokens:
            raise ContractError("production protein residue-window dimensions are invalid")
        if overlap < 0 or overlap >= residue_window:
            raise ContractError("production protein window overlap is invalid")
        for key in ("sequence_count", "long_sequence_count", "total_window_count"):
            if int(windowing.get(key, -1)) < 0:
                raise ContractError(f"production protein windowing {key} is invalid")
        if int(windowing["sequence_count"]) != int(artifacts["targets"].get("rows", -1)):
            raise ContractError("production protein windowing sequence count mismatch")
        if int(windowing["total_window_count"]) < int(windowing["sequence_count"]):
            raise ContractError("production protein window count cannot omit sequences")
        if windowing.get("full_sequence_coverage") is not True or windowing.get("silent_truncation") is not False:
            raise ContractError("production protein windowing must record full coverage without truncation")
    freeze = payload.get("freeze_contract")
    if not isinstance(freeze, Mapping) or freeze.get("encoders_frozen") is not True:
        raise ContractError("embedding manifest requires frozen encoder contract")
    if freeze.get("requires_grad") is not False or freeze.get("no_grad") is not True:
        raise ContractError("embedding manifest must record requires_grad=False and no_grad=True")
    return payload


def embedding_feature_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the immutable feature semantics a trained head consumes."""
    artifacts = payload.get("artifacts") or {}
    checkpoints = payload.get("checkpoints") or {}
    provenance = payload.get("provenance") or {}
    return {
        "schema_version": payload.get("schema_version"),
        "mode": payload.get("mode"),
        "checkpoints": checkpoints,
        "ligand_artifact": {
            key: (artifacts.get("ligands") or {}).get(key)
            for key in (
                "sha256",
                "dim",
                "molformer_dim",
                "ecfp_bits",
                "ecfp_radius",
                "ecfp_encoding",
            )
        },
        "ligand_shards": [
            {key: shard.get(key) for key in ("sha256", "rows", "dim")}
            for shard in (artifacts.get("ligands") or {}).get("shards", ())
        ],
        "target_artifact": {
            key: (artifacts.get("targets") or {}).get(key)
            for key in ("sha256", "dim")
        },
        "target_shards": [
            {key: shard.get(key) for key in ("sha256", "rows", "dim")}
            for shard in (artifacts.get("targets") or {}).get("shards", ())
        ],
        "pooling": provenance.get("pooling"),
        "precision": provenance.get("precision"),
        "ligand_tokenization": provenance.get("ligand_tokenization"),
        "protein_windowing": provenance.get("protein_windowing"),
    }


def validate_budget_manifest(path: Path) -> dict[str, Any]:
    payload = read_json_object(path, "budget manifest")
    require_schema(payload, BUDGET_SCHEMA, "budget manifest")
    peak = float(payload.get("peak_vram_gib", math.inf))
    params = int(payload.get("trainable_params", MAX_TRAINABLE_PARAMS + 1))
    if peak > MAX_PEAK_VRAM_GIB:
        raise ContractError(f"peak_vram_gib exceeds {MAX_PEAK_VRAM_GIB:g}: {peak:g}")
    if params > MAX_TRAINABLE_PARAMS:
        raise ContractError(f"trainable_params exceeds {MAX_TRAINABLE_PARAMS}: {params}")
    execution_mode = str(payload.get("execution_mode") or "")
    peak_source = str(payload.get("peak_vram_source") or "")
    if execution_mode == "cuda":
        if peak_source != "torch.cuda.max_memory_allocated" or payload.get("peak_vram_measured") is not True:
            raise ContractError("CUDA budget requires measured torch.cuda.max_memory_allocated evidence")
        if not math.isfinite(peak) or peak <= 0.0:
            raise ContractError("CUDA budget peak_vram_gib measurement is unavailable")
    elif execution_mode in {"cpu", "fixture"}:
        if peak_source != "not_applicable_cpu" or payload.get("peak_vram_measured") is not False:
            raise ContractError("CPU/fixture budget must explicitly mark CUDA VRAM as not applicable")
        if peak != 0.0:
            raise ContractError("CPU/fixture budget peak_vram_gib must be zero")
    else:
        raise ContractError("budget manifest execution_mode must be cuda, cpu, or fixture")
    fallbacks = payload.get("fallback_model_ids")
    if not isinstance(fallbacks, Sequence) or isinstance(fallbacks, (str, bytes)) or not fallbacks:
        raise ContractError("budget manifest requires fallback_model_ids")
    license_profile = payload.get("license_profile")
    if not isinstance(license_profile, Mapping) or not license_profile.get("redistribution"):
        raise ContractError("budget manifest requires license_profile.redistribution")
    expected_licenses = {
        MOLFORMER_CHECKPOINT: MOLFORMER_LICENSE,
        f"facebook/{ESM2_CHECKPOINT}": ESM2_LICENSE,
    }
    if license_profile.get("model_licenses") != expected_licenses:
        raise ContractError("budget manifest model license profile changed")
    return payload


def validate_calibration_manifest(payload: Mapping[str, Any]) -> None:
    require_schema(payload, CALIBRATION_SCHEMA, "calibration manifest")
    if payload.get("logit_source") not in {
        "saved_parameter_average",
        "saved_prediction_ensemble",
    }:
        raise ContractError("calibration logits must come from the saved scoring model")
    state_hash = str(payload.get("model_state_sha256") or "")
    if not SHA256_PATTERN.fullmatch(state_hash):
        raise ContractError("calibration manifest model_state_sha256 is invalid")
    split = payload.get("split_evidence")
    if not isinstance(split, Mapping):
        raise ContractError("calibration manifest missing split_evidence")
    if split.get("strategy") != "sha256_pair_hash_target_ontology_label_stratified":
        raise ContractError("calibration split strategy is invalid")
    if split.get("source") != "train_only" or split.get("disjoint_pair_hashes") is not True:
        raise ContractError("calibration split must be train-only and pair-disjoint")
    for key in ("fit_pair_hashes_sha256", "holdout_pair_hashes_sha256"):
        if not SHA256_PATTERN.fullmatch(str(split.get(key) or "")):
            raise ContractError(f"calibration split {key} is invalid")
    for key in ("fit_pairs", "holdout_pairs", "fit_measured_rows", "holdout_measured_rows"):
        if int(split.get(key, -1)) < 0:
            raise ContractError(f"calibration split {key} must be nonnegative")
    event_counts = split.get("event_counts")
    if not isinstance(event_counts, Mapping):
        raise ContractError("calibration split event_counts is missing")
    for event in ("direct_binding_reversible", "functional_modulation"):
        record = payload.get(event)
        if not isinstance(record, Mapping):
            raise ContractError(f"calibration manifest missing {event}")
        counts = event_counts.get(event)
        if not isinstance(counts, Mapping):
            raise ContractError(f"calibration split missing counts for {event}")
        holdout_pos = int(counts.get("holdout_pos", -1))
        holdout_neg = int(counts.get("holdout_neg", -1))
        if holdout_pos < 0 or holdout_neg < 0:
            raise ContractError(f"calibration split counts are invalid for {event}")
        if record.get("status") == "measured_event_calibrated":
            if int(record.get("n_pos", 0)) < 1 or int(record.get("n_neg", 0)) < 1:
                raise ContractError(f"{event} calibrator lacks measured positive/negative coverage")
            if record.get("coverage_source") != "train_only_calibration_holdout":
                raise ContractError(f"{event} calibrator coverage source is invalid")
            if int(record["n_pos"]) != holdout_pos or int(record["n_neg"]) != holdout_neg:
                raise ContractError(f"{event} calibrator counts do not match holdout evidence")
            if not all(math.isfinite(float(record.get(key, math.nan))) for key in ("a", "b")):
                raise ContractError(f"{event} calibrator coefficients are nonfinite")
        elif record.get("status") != "fallback_unavailable":
            raise ContractError(f"{event} calibrator status is invalid")


def validate_model_manifest(path: Path) -> dict[str, Any]:
    payload = read_json_object(path, "model manifest")
    require_schema(payload, MODEL_SCHEMA, "model manifest")
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ContractError("model manifest missing inputs")
    train_csv = validate_artifact(
        inputs.get("train_csv"), base_dir=path.parent, label="model train_csv"
    )
    embedding_manifest = validate_artifact(
        inputs.get("embedding_manifest"), base_dir=path.parent, label="model embedding_manifest"
    )
    embedding_payload = validate_embedding_manifest(embedding_manifest)
    prepared_input_record = embedding_payload["inputs"].get("input_manifest")
    if prepared_input_record is not None:
        prepared_input_path = validate_artifact(
            prepared_input_record,
            base_dir=embedding_manifest.parent,
            label="model prepared input manifest",
        )
        prepared_inputs = validate_input_manifest(prepared_input_path, verify_files=False)
        train_record = inputs.get("train_csv")
        if (
            not isinstance(train_record, Mapping)
            or prepared_inputs["artifacts"]["train"].get("sha256")
            != train_record.get("sha256")
        ):
            raise ContractError("model train_csv does not match prepared input manifest")
    budget_manifest = validate_artifact(
        inputs.get("budget_manifest"), base_dir=path.parent, label="model budget_manifest"
    )
    budget_payload = validate_budget_manifest(budget_manifest)
    model_artifact = validate_artifact(
        payload.get("model_artifact"), base_dir=path.parent, label="model artifact"
    )
    validate_calibration_manifest(payload.get("calibration") or {})
    state_hash = str(payload.get("model_state_sha256") or "")
    calibration_hash = str((payload.get("calibration") or {}).get("model_state_sha256") or "")
    if not SHA256_PATTERN.fullmatch(state_hash) or state_hash != calibration_hash:
        raise ContractError("model manifest state hash does not bind calibration logits")
    if int(payload.get("trainable_params", MAX_TRAINABLE_PARAMS + 1)) > MAX_TRAINABLE_PARAMS:
        raise ContractError("model manifest trainable_params exceeds budget")
    if int(payload.get("trainable_params", -1)) != int(budget_payload.get("trainable_params", -2)):
        raise ContractError("model and budget trainable_params do not match")
    policy = payload.get("training_policy")
    if not isinstance(policy, Mapping):
        raise ContractError("model manifest missing training_policy")
    if policy.get("frozen_cached_embeddings") is not True:
        raise ContractError("model manifest must train on frozen cached embeddings")
    if policy.get("encoder_requires_grad") is not False or policy.get("encoder_no_grad") is not True:
        raise ContractError("model manifest must prove encoders require_grad=False/no_grad")
    seeds = policy.get("seeds")
    if (
        not isinstance(seeds, Sequence)
        or isinstance(seeds, (str, bytes))
        or len(seeds) != 3
        or len(set(map(int, seeds))) != 3
    ):
        raise ContractError("model manifest requires exactly three distinct sequential seeds")
    if model_artifact.suffix == ".pt" and tuple(map(int, seeds)) != PRODUCTION_SEEDS:
        raise ContractError(f"production model seeds must be exactly {PRODUCTION_SEEDS}")
    if payload.get("calibration_split") != (payload.get("calibration") or {}).get("split_evidence"):
        raise ContractError("model calibration split evidence is inconsistent")
    hyperparameters = payload.get("training_hyperparameters")
    if not isinstance(hyperparameters, Mapping):
        raise ContractError("model manifest missing training_hyperparameters")
    if int(hyperparameters.get("requested_batch_size", 0)) < 1:
        raise ContractError("model requested_batch_size must be positive")
    if not math.isfinite(float(hyperparameters.get("learning_rate", math.nan))) or float(
        hyperparameters.get("learning_rate", 0.0)
    ) <= 0.0:
        raise ContractError("model learning_rate must be finite and positive")
    if model_artifact.suffix not in {".npz", ".pt"}:
        raise ContractError("performance-v2 model artifact must be .npz or .pt")
    if (
        model_artifact.suffix == ".pt"
        and (payload.get("calibration") or {}).get("logit_source")
        != "saved_prediction_ensemble"
    ):
        raise ContractError("production torch calibration must use saved prediction ensemble logits")
    try:
        train_frame = pd.read_csv(train_csv)
    except Exception as exc:
        raise ContractError(f"model train_csv could not be read: {train_csv}") from exc
    split = payload.get("calibration_split")
    if not isinstance(split, Mapping):
        raise ContractError("model calibration_split is missing")
    domain = training_domain_evidence(
        train_frame,
        calibration_fraction=float(split.get("fraction", math.nan)),
        include_ranking_only_contrastive=model_artifact.suffix == ".pt",
    )
    for field in ("train_ligands", "train_targets"):
        values = payload.get(field)
        if (
            not isinstance(values, Sequence)
            or isinstance(values, (str, bytes))
            or list(values) != domain[field]
        ):
            raise ContractError(f"model {field} does not match bound train_csv")
    if payload.get("train_domain_sha256") != domain["train_domain_sha256"]:
        raise ContractError("model train_domain_sha256 does not match bound train_csv")
    for field in ("ranking_only_pairs", "ranking_only_pairs_used"):
        if int(payload.get(field, -1)) != int(domain[field]):
            raise ContractError(f"model {field} does not match bound train_csv")
    if split != domain["calibration_split"]:
        raise ContractError("model calibration split cannot be reproduced from train_csv")
    return payload


def validate_ontology(value: str) -> str:
    value = str(value).strip()
    if value not in ONTOLOGY:
        raise ContractError(f"ontology value is invalid: {value}")
    return value


def measured_mask(frame: pd.DataFrame) -> pd.Series:
    required = {"evidence_state", "measured_label"}
    missing = required - set(frame.columns)
    if missing:
        raise ContractError(f"PU frame missing columns: {sorted(missing)}")
    state = frame["evidence_state"].fillna("").astype(str).str.strip()
    label = pd.to_numeric(frame["measured_label"], errors="coerce")
    valid = state.isin({"measured_positive", "measured_negative"})
    bad = frame.loc[valid & ~label.isin([0, 1])]
    if not bad.empty:
        raise ContractError("measured positives/negatives require measured_label 0 or 1")
    gray = state.isin({"gray_unmeasured", "unmeasured", "unknown"})
    if (gray & label.eq(0)).any():
        raise ContractError("gray/unmeasured evidence must never be labelled negative")
    unknown = ~(valid | gray)
    if unknown.any():
        raise ContractError("evidence_state contains unsupported values")
    return valid


def split_pu_training(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    measured = frame.loc[measured_mask(frame)].copy()
    ranking_only = frame.loc[~measured_mask(frame)].copy()
    if "ranking_only" not in ranking_only.columns:
        ranking_only["ranking_only"] = True
    ranking_flags = ranking_only["ranking_only"].astype("string").fillna("").str.strip().str.lower()
    if not ranking_flags.eq("true").all():
        raise ContractError("unlabeled contrastive negatives must be marked ranking-only")
    return measured, ranking_only


def training_domain_evidence(
    train: pd.DataFrame,
    *,
    calibration_fraction: float,
    include_ranking_only_contrastive: bool,
) -> dict[str, Any]:
    required = {"ligand_id", "target_id", "evidence_state", "measured_label", "ontology"}
    missing = required - set(train.columns)
    if missing:
        raise ContractError(f"train-domain evidence missing columns: {sorted(missing)}")
    measured, ranking_only = split_pu_training(train)
    fit_measured, calibration_holdout, split_evidence = calibration_holdout_split(
        measured,
        fraction=calibration_fraction,
    )
    fit_ligands = fit_measured["ligand_id"].astype(str).str.strip()
    fit_targets = fit_measured["target_id"].astype(str).str.strip()
    if fit_ligands.eq("").any() or fit_targets.eq("").any():
        raise ContractError("train-domain evidence requires nonblank ligand_id and target_id")
    train_ligands = set(fit_ligands)
    train_targets = set(fit_targets)
    ranking_only_used = 0
    if include_ranking_only_contrastive:
        positive_by_ligand: dict[str, set[str]] = {}
        candidate_by_ligand: dict[str, set[str]] = {}
        for ligand_id, target_id, label in zip(
            fit_ligands,
            fit_targets,
            fit_measured["measured_label"].astype(int),
            strict=True,
        ):
            candidate_by_ligand.setdefault(ligand_id, set()).add(target_id)
            if int(label) == 1:
                positive_by_ligand.setdefault(ligand_id, set()).add(target_id)
        holdout_pairs = set(
            zip(
                calibration_holdout["ligand_id"].astype(str).str.strip(),
                calibration_holdout["target_id"].astype(str).str.strip(),
                strict=True,
            )
        )
        for ligand_id, target_id in zip(
            ranking_only["ligand_id"].astype(str).str.strip(),
            ranking_only["target_id"].astype(str).str.strip(),
            strict=True,
        ):
            if not ligand_id or not target_id:
                raise ContractError(
                    "ranking-only contrastive pair requires ligand_id and target_id"
                )
            if (ligand_id, target_id) in holdout_pairs:
                continue
            if ligand_id not in positive_by_ligand:
                continue
            candidate_by_ligand.setdefault(ligand_id, set()).add(target_id)
            ranking_only_used += 1
        for ligand_id, positives in positive_by_ligand.items():
            candidates = candidate_by_ligand.get(ligand_id, set())
            if candidates - positives:
                train_ligands.add(ligand_id)
                train_targets.update(candidates)
    ordered_ligands = sorted(train_ligands)
    ordered_targets = sorted(train_targets)
    return {
        "train_ligands": ordered_ligands,
        "train_targets": ordered_targets,
        "train_domain_sha256": sha256_payload(
            {
                "train_ligands": ordered_ligands,
                "train_targets": ordered_targets,
            }
        ),
        "ranking_only_pairs": int(len(ranking_only)),
        "ranking_only_pairs_used": int(ranking_only_used),
        "calibration_split": split_evidence,
    }


def target_balanced_weights(frame: pd.DataFrame) -> np.ndarray:
    if "target_id" not in frame.columns or "measured_label" not in frame.columns:
        raise ContractError("target-balanced weights require target_id and measured_label")
    if frame.empty:
        return np.zeros(0, dtype=float)
    balanced = pd.DataFrame({
        "target": frame["target_id"].astype(str).to_numpy(),
        "label": pd.to_numeric(frame["measured_label"], errors="raise").astype(int).to_numpy(),
    })
    if balanced["target"].eq("").any():
        raise ContractError("target-balanced weights require nonblank target_id values")
    target_count = int(balanced["target"].nunique())
    labels_per_target = balanced.groupby("target", sort=False)["label"].transform("nunique")
    rows_per_target_label = balanced.groupby(["target", "label"], sort=False)["label"].transform("size")
    weights = (
        1.0 / target_count / labels_per_target.to_numpy(dtype=float) / rows_per_target_label.to_numpy(dtype=float)
    )
    weights *= len(weights) / weights.sum()
    return weights


def effective_target_count(frame: pd.DataFrame, weights: Sequence[float]) -> float:
    if frame.empty:
        return 0.0
    by_target: dict[str, float] = {}
    for target, weight in zip(frame["target_id"].astype(str), weights, strict=True):
        by_target[target] = by_target.get(target, 0.0) + float(weight)
    total = sum(by_target.values())
    if total <= 0:
        return 0.0
    # A target whose weights sum to zero contributes 0*log(0) == 0 to the
    # entropy; leaving it in the vector makes numpy return NaN instead.
    probs = np.array(
        [value / total for value in by_target.values() if value > 0.0],
        dtype=float,
    )
    if probs.size == 0:
        return 0.0
    entropy = -float(np.sum(probs * np.log(probs)))
    if not math.isfinite(entropy):
        raise ContractError("effective target count entropy is not finite")
    return float(math.exp(entropy))


def smoothed_prior_logit(pos: int, neg: int, *, alpha: float = 1.0) -> float:
    if pos < 0 or neg < 0 or alpha <= 0:
        raise ContractError(
            "smoothed prior counts must be non-negative and alpha must be positive"
        )
    return math.log((pos + alpha) / (neg + alpha))


def prior_correction_beta(
    *,
    train_pos: int,
    train_neg: int,
    deploy_pos: int,
    deploy_neg: int,
    beta: float,
    alpha: float = 1.0,
    source: str = "train",
) -> float:
    if source != "train":
        raise ContractError("prior correction beta must be selected on train-only evidence")
    if not math.isfinite(beta) or beta < 0.0 or beta > 1.0:
        raise ContractError("prior correction beta must be finite and within [0, 1]")
    train = smoothed_prior_logit(train_pos, train_neg, alpha=alpha)
    deploy = smoothed_prior_logit(deploy_pos, deploy_neg, alpha=alpha)
    return beta * (deploy - train)


def sha256_ndarrays(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def calibration_holdout_split(
    measured: pd.DataFrame,
    *,
    fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    required = {"ligand_id", "target_id", "ontology", "measured_label", "evidence_state"}
    missing = required - set(measured.columns)
    if missing:
        raise ContractError(f"calibration split missing columns: {sorted(missing)}")
    if not math.isfinite(fraction) or fraction <= 0.0 or fraction >= 0.5:
        raise ContractError("calibration holdout fraction must be within (0, 0.5)")
    if not measured_mask(measured).all():
        raise ContractError("calibration split accepts explicit measured events only")

    frame = measured.copy().reset_index(drop=True)
    frame["ligand_id"] = frame["ligand_id"].astype(str).str.strip()
    frame["target_id"] = frame["target_id"].astype(str).str.strip()
    frame["ontology"] = frame["ontology"].astype(str).map(validate_ontology)
    frame["measured_label"] = pd.to_numeric(frame["measured_label"], errors="raise").astype(int)
    if frame["ligand_id"].eq("").any() or frame["target_id"].eq("").any():
        raise ContractError("calibration split requires nonblank ligand_id and target_id")
    frame["_pair_key"] = (
        frame["ligand_id"] + "\0" + frame["target_id"] + "\0" + frame["ontology"]
    )
    conflicts = frame.groupby("_pair_key", sort=False)["measured_label"].nunique()
    if (conflicts > 1).any():
        raise ContractError("the same ligand/target/ontology pair has conflicting measured labels")
    frame["_pair_hash"] = frame["_pair_key"].map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )

    holdout_hashes: set[str] = set()
    strata = frame.groupby(["target_id", "ontology", "measured_label"], sort=True)
    for _, group in strata:
        unique_pairs = sorted(set(group["_pair_hash"].tolist()))
        if len(unique_pairs) < 2:
            continue
        holdout_count = max(1, int(math.floor(len(unique_pairs) * fraction)))
        holdout_count = min(holdout_count, len(unique_pairs) - 1)
        holdout_hashes.update(unique_pairs[:holdout_count])

    holdout_mask = frame["_pair_hash"].isin(holdout_hashes)
    fit = frame.loc[~holdout_mask].copy()
    holdout = frame.loc[holdout_mask].copy()
    fit_hashes = sorted(set(fit["_pair_hash"].tolist()))
    held_hashes = sorted(set(holdout["_pair_hash"].tolist()))
    if set(fit_hashes) & set(held_hashes):
        raise ContractError("calibration fit and holdout pair hashes overlap")

    def digest_hashes(values: Sequence[str]) -> str:
        return hashlib.sha256(("\n".join(values) + "\n").encode("ascii")).hexdigest()

    event_counts: dict[str, dict[str, int]] = {}
    for event in ONTOLOGY:
        fit_event = fit.loc[fit["ontology"].eq(event), "measured_label"]
        holdout_event = holdout.loc[holdout["ontology"].eq(event), "measured_label"]
        event_counts[event] = {
            "fit_pos": int(fit_event.eq(1).sum()),
            "fit_neg": int(fit_event.eq(0).sum()),
            "holdout_pos": int(holdout_event.eq(1).sum()),
            "holdout_neg": int(holdout_event.eq(0).sum()),
        }
    evidence = {
        "strategy": "sha256_pair_hash_target_ontology_label_stratified",
        "source": "train_only",
        "fraction": float(fraction),
        "fit_pair_hashes_sha256": digest_hashes(fit_hashes),
        "holdout_pair_hashes_sha256": digest_hashes(held_hashes),
        "fit_pairs": int(len(fit_hashes)),
        "holdout_pairs": int(len(held_hashes)),
        "fit_measured_rows": int(len(fit)),
        "holdout_measured_rows": int(len(holdout)),
        "disjoint_pair_hashes": True,
        "event_counts": event_counts,
    }
    columns = [column for column in frame.columns if not column.startswith("_pair_")]
    return fit[columns].reset_index(drop=True), holdout[columns].reset_index(drop=True), evidence


class PlattCalibrator:
    def __init__(
        self,
        a: float,
        b: float,
        n_pos: int,
        n_neg: int,
        status: str = "measured_event_calibrated",
    ) -> None:
        self.a = float(a)
        self.b = float(b)
        self.n_pos = int(n_pos)
        self.n_neg = int(n_neg)
        self.status = str(status)

    def predict(self, scores: Sequence[float]) -> np.ndarray:
        values = np.asarray(scores, dtype=float)
        logits = self.a * values + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))

    def to_manifest(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "a": float(self.a),
            "b": float(self.b),
            "n_pos": int(self.n_pos),
            "n_neg": int(self.n_neg),
            "semantics": "measured_event_probability",
            "coverage_source": "train_only_calibration_holdout",
        }


def fit_platt(scores: Sequence[float], labels: Sequence[int]) -> PlattCalibrator | None:
    x = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    valid = np.isfinite(x) & np.isin(y, [0.0, 1.0])
    x = x[valid]
    y = y[valid]
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos < 1 or n_neg < 1:
        return None

    def loss(theta: np.ndarray) -> float:
        logits = np.clip(theta[0] * x + theta[1], -60, 60)
        return float(np.sum(np.logaddexp(0.0, logits) - y * logits) + 1e-4 * np.sum(theta * theta))

    res = minimize(loss, np.zeros(2, dtype=float), method="BFGS")
    if not res.success:
        raise ContractError(f"Platt calibration failed: {res.message}")
    return PlattCalibrator(float(res.x[0]), float(res.x[1]), n_pos=n_pos, n_neg=n_neg)


def calibration_manifest(
    direct: PlattCalibrator | None,
    functional: PlattCalibrator | None,
    *,
    split_evidence: Mapping[str, Any],
    model_state_sha256: str,
    logit_source: str = "saved_parameter_average",
) -> dict[str, Any]:
    def record(cal: PlattCalibrator | None) -> dict[str, Any]:
        if cal is None:
            return {
                "status": "fallback_unavailable",
                "semantics": "not_a_probability",
                "reason": "missing measured positive/negative calibration coverage",
            }
        return cal.to_manifest()

    payload = {
        "schema_version": CALIBRATION_SCHEMA,
        "logit_source": logit_source,
        "model_state_sha256": str(model_state_sha256),
        "split_evidence": dict(split_evidence),
        "direct_binding_reversible": record(direct),
        "functional_modulation": record(functional),
    }
    validate_calibration_manifest(payload)
    return payload


def _string_domain(values: Iterable[str]) -> Set[str]:
    if isinstance(values, Set):
        return values
    return frozenset(map(str, values))


def ood_route(ligand_id: str, target_id: str, train_ligands: Iterable[str], train_targets: Iterable[str], *, structure_abstained: bool = False) -> str:
    if structure_abstained:
        return "structure_abstained"
    ligand_seen = str(ligand_id) in _string_domain(train_ligands)
    target_seen = str(target_id) in _string_domain(train_targets)
    if ligand_seen and target_seen:
        return "in_domain"
    if ligand_seen:
        return "target_cold"
    if target_seen:
        return "ligand_cold"
    return "dual_cold"


def should_abstain(route: str, *, score: float | None = None, min_score: float | None = None) -> bool:
    if route not in OOD_ROUTES:
        raise ContractError(f"OOD route is invalid: {route}")
    if route in {"dual_cold", "structure_abstained"}:
        return True
    if min_score is not None and (score is None or not math.isfinite(score) or score < min_score):
        return True
    return False


def deterministic_vector(identifier: str, dim: int) -> np.ndarray:
    values: list[float] = []
    counter = 0
    while len(values) < dim:
        digest = hashlib.sha256(f"{identifier}:{counter}".encode("utf-8")).digest()
        values.extend((byte / 127.5) - 1.0 for byte in digest)
        counter += 1
    vec = np.array(values[:dim], dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm else vec


def fixture_embedding_frame(ids: Sequence[str], *, id_column: str, dim: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for identifier in sorted({str(value).strip() for value in ids if str(value).strip()}):
        vec = deterministic_vector(identifier, dim)
        row: dict[str, Any] = {id_column: identifier}
        row.update({f"e{i}": float(vec[i]) for i in range(dim)})
        rows.append(row)
    if not rows:
        raise ContractError(f"no fixture IDs supplied for {id_column}")
    return pd.DataFrame(rows)


def canonicalize_smiles(smiles: str) -> str:
    from rdkit import Chem

    raw = str(smiles or "").strip()
    if not raw:
        raise ContractError("query SMILES must be nonblank")
    mol = Chem.MolFromSmiles(raw)
    if mol is None:
        raise ContractError("query SMILES failed RDKit parsing")
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    if not canonical:
        raise ContractError("query SMILES canonicalization produced a blank value")
    return canonical


def canonical_smiles_record(smiles: str) -> dict[str, Any]:
    canonical = canonicalize_smiles(smiles)
    return {
        "input_type": "smiles",
        "canonical_smiles_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "canonicalization": "RDKit MolToSmiles canonical=True isomericSmiles=True",
        "raw_smiles_recorded": False,
    }


def load_embedding_table(path: Path, id_column: str) -> tuple[list[str], np.ndarray]:
    frame = pd.read_csv(path)
    if id_column not in frame.columns:
        raise ContractError(f"embedding table missing {id_column}: {path}")
    feature_cols = [
        col
        for col in frame.columns
        if col.startswith("e")
    ]
    if not feature_cols:
        raise ContractError(f"embedding table has no e* feature columns: {path}")
    ids = frame[id_column].astype(str).str.strip().tolist()
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ContractError(f"embedding table requires unique nonblank {id_column}")
    matrix = frame[feature_cols].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ContractError(f"embedding table contains nonfinite values: {path}")
    return ids, matrix


def load_embedding_artifact(
    record: Mapping[str, Any], *, base_dir: Path, id_column: str, label: str
) -> tuple[list[str], np.ndarray]:
    fmt = str(record.get("format") or "csv")
    if fmt == "csv":
        path = validate_artifact(record, base_dir=base_dir, label=label)
        return load_embedding_table(path, id_column)
    validate_embedding_artifact(record, base_dir=base_dir, label=label)
    ids: list[str] = []
    matrices: list[np.ndarray] = []
    for shard in record["shards"]:
        shard_path = validate_artifact(shard, base_dir=base_dir, label=f"{label} shard")
        with np.load(shard_path, allow_pickle=False) as arrays:
            ids.extend(str(value) for value in arrays["ids"].tolist())
            matrices.append(arrays["vectors"].astype(np.float32, copy=True))
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ContractError(f"{label} requires unique nonblank IDs across shards")
    return ids, np.vstack(matrices)


class EmbeddingStore:
    """Indexable embedding access without materializing all production shards."""

    def __init__(
        self,
        record: Mapping[str, Any],
        *,
        base_dir: Path,
        id_column: str,
        label: str,
        cache_shards: int = 2,
        validated: bool = False,
    ) -> None:
        self.label = label
        self.dim = int(record.get("dim", 0))
        self._matrix: np.ndarray | None = None
        self._shard_paths: list[Path] = []
        self._shard_for_index: list[int] = []
        self._row_for_index: list[int] = []
        self._cache_limit = max(1, int(cache_shards))
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.shard_cache_misses = 0
        fmt = str(record.get("format") or "csv")
        if fmt == "csv":
            path = validate_artifact(record, base_dir=base_dir, label=label)
            self.ids, self._matrix = load_embedding_table(path, id_column)
            self.dim = int(self._matrix.shape[1])
        else:
            if not validated:
                validate_embedding_artifact(record, base_dir=base_dir, label=label)
            self.ids = []
            for shard_index, shard in enumerate(record["shards"]):
                shard_path = validate_artifact(
                    shard, base_dir=base_dir, label=f"{label} shard {shard_index}"
                )
                self._shard_paths.append(shard_path)
                with np.load(shard_path, allow_pickle=False) as arrays:
                    shard_ids = [str(value) for value in arrays["ids"].tolist()]
                self.ids.extend(shard_ids)
                self._shard_for_index.extend([shard_index] * len(shard_ids))
                self._row_for_index.extend(range(len(shard_ids)))
        if any(not value for value in self.ids) or len(set(self.ids)) != len(self.ids):
            raise ContractError(f"{label} requires unique nonblank IDs")
        self.id_to_index = {identifier: index for index, identifier in enumerate(self.ids)}

    def __len__(self) -> int:
        return len(self.ids)

    def _shard_matrix(self, shard_index: int) -> np.ndarray:
        cached = self._cache.pop(shard_index, None)
        if cached is not None:
            self._cache[shard_index] = cached
            return cached
        with np.load(self._shard_paths[shard_index], allow_pickle=False) as arrays:
            matrix = arrays["vectors"].astype(np.float32, copy=True)
        self.shard_cache_misses += 1
        self._cache[shard_index] = matrix
        while len(self._cache) > self._cache_limit:
            self._cache.popitem(last=False)
        return matrix

    def fetch(self, indices: Sequence[int] | np.ndarray) -> np.ndarray:
        requested = np.asarray(indices, dtype=np.int64).reshape(-1)
        if requested.size == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        if requested.min() < 0 or requested.max() >= len(self):
            raise ContractError(f"{self.label} embedding index is out of range")
        if self._matrix is not None:
            return self._matrix[requested].astype(np.float32, copy=False)
        output = np.empty((len(requested), self.dim), dtype=np.float32)
        shard_indexes = np.asarray([self._shard_for_index[index] for index in requested], dtype=np.int64)
        row_indexes = np.asarray([self._row_for_index[index] for index in requested], dtype=np.int64)
        for shard_index in np.unique(shard_indexes):
            positions = np.flatnonzero(shard_indexes == shard_index)
            output[positions] = self._shard_matrix(int(shard_index))[row_indexes[positions]]
        return output

    def vector_for_id(self, identifier: str) -> np.ndarray:
        if identifier not in self.id_to_index:
            raise ContractError(f"{self.label} ID is missing from embeddings: {identifier}")
        return self.fetch([self.id_to_index[identifier]])[0]

    def locality_key(self, index: int) -> int:
        if index < 0 or index >= len(self):
            raise ContractError(f"{self.label} embedding index is out of range")
        return 0 if self._matrix is not None else self._shard_for_index[index]

    def iter_batches(self, batch_size: int) -> Iterable[tuple[list[str], np.ndarray]]:
        if batch_size < 1:
            raise ContractError("embedding iteration batch_size must be positive")
        for start in range(0, len(self), batch_size):
            indexes = np.arange(start, min(start + batch_size, len(self)), dtype=np.int64)
            yield self.ids[start : start + len(indexes)], self.fetch(indexes)


class DeterministicShardBatchSampler:
    """Shuffle shard groups and rows deterministically while keeping batches shard-local."""

    def __init__(self, locality_keys: Sequence[int] | np.ndarray, batch_size: int, seed: int) -> None:
        keys = np.asarray(locality_keys, dtype=np.int64).reshape(-1)
        if batch_size < 1:
            raise ContractError("shard-local batch_size must be positive")
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        if keys.size == 0:
            self._groups: tuple[np.ndarray, ...] = ()
            return
        sorted_rows = np.argsort(keys, kind="stable")
        sorted_keys = keys[sorted_rows]
        boundaries = np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1
        self._groups = tuple(np.split(sorted_rows, boundaries))

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        for group_index in rng.permutation(len(self._groups)):
            rows = self._groups[int(group_index)].copy()
            rng.shuffle(rows)
            for start in range(0, len(rows), self.batch_size):
                yield rows[start : start + self.batch_size].tolist()

    def __len__(self) -> int:
        return sum(math.ceil(len(group) / self.batch_size) for group in self._groups)


def pair_features(ligand: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.concatenate([ligand, target, ligand * target, np.abs(ligand - target)], axis=0)


def resolve_hf_commit_hash(
    *,
    model: Any,
    tokenizer: Any,
    model_id: str,
    revision: str,
) -> str:
    candidates: set[str] = set()
    config = getattr(model, "config", None)
    for value in (
        getattr(config, "_commit_hash", None),
        getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
        if isinstance(getattr(tokenizer, "init_kwargs", None), Mapping)
        else None,
    ):
        normalized = str(value or "").lower()
        if HF_COMMIT_PATTERN.fullmatch(normalized):
            candidates.add(normalized)
    try:
        from transformers.utils.hub import cached_file

        config_path = cached_file(model_id, "config.json", revision=revision)
    except Exception:
        config_path = None
    if config_path:
        match = re.search(r"[/\\]snapshots[/\\]([0-9a-fA-F]{40})[/\\]", str(config_path))
        if match:
            candidates.add(match.group(1).lower())
    if not candidates:
        raise ContractError(f"could not resolve immutable Hugging Face commit for {model_id}@{revision}")
    if len(candidates) != 1:
        raise ContractError(f"conflicting Hugging Face commits resolved for {model_id}@{revision}")
    return next(iter(candidates))


def _pooled_transformer_embeddings(
    *,
    texts: Sequence[str],
    model_id: str,
    revision: str,
    max_length: int,
    trust_remote_code: bool,
    expected_resolved_revision: str,
) -> tuple[np.ndarray, str]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    model = load_transformer_fp32_then_place(
        AutoModel,
        model_id=model_id,
        revision=revision,
        trust_remote_code=trust_remote_code,
        device=device,
    )
    model.eval()
    resolved_revision = resolve_hf_commit_hash(
        model=model,
        tokenizer=tokenizer,
        model_id=model_id,
        revision=revision,
    )
    if resolved_revision != expected_resolved_revision:
        raise ContractError("loaded checkpoint commit does not match embedding manifest")
    with torch.no_grad():
        encoded = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        # MoLFormer compat-v4 is numerically unstable under fp16 autocast.
        outputs = model(**encoded)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    pooled_array = pooled.detach().float().cpu().numpy().astype(np.float32)
    if not np.isfinite(pooled_array).all():
        raise ContractError("query transformer produced nonfinite embeddings")
    return pooled_array, resolved_revision


def ecfp_bit_vector(
    canonical_smiles: str,
    *,
    n_bits: int = ECFP_BITS,
    radius: int = ECFP_RADIUS,
) -> np.ndarray:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol is None:
        raise ContractError("canonical query SMILES failed RDKit parsing")
    if n_bits != ECFP_BITS or radius != ECFP_RADIUS:
        raise ContractError(f"performance-v2 ECFP must use {ECFP_BITS} bits at radius {ECFP_RADIUS}")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    bitvect = generator.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bitvect, arr)
    return arr.astype(np.float32)


def compute_query_ligand_features(
    *,
    smiles: str,
    embedding_manifest_path: Path,
    max_length: int | None = None,
) -> tuple[str, np.ndarray, dict[str, Any]]:
    embedding_manifest = validate_embedding_manifest(embedding_manifest_path)
    ligand_record = embedding_manifest["artifacts"]["ligands"]
    ligand_checkpoint = embedding_manifest["checkpoints"]["ligand"]
    trained_max_length = int(
        embedding_manifest["provenance"]["ligand_tokenization"]["max_tokens"]
    )
    if max_length is not None and int(max_length) != trained_max_length:
        raise ContractError(
            "query max length must exactly match the bound training embedding tokenizer policy"
        )
    canonical = canonicalize_smiles(smiles)
    molformer_batch, loaded_revision = _pooled_transformer_embeddings(
        texts=[canonical],
        model_id=MOLFORMER_CHECKPOINT,
        revision=str(ligand_checkpoint["resolved_revision"]),
        max_length=trained_max_length,
        trust_remote_code=True,
        expected_resolved_revision=str(ligand_checkpoint["resolved_revision"]),
    )
    molformer = molformer_batch[0]
    molformer_dim = int(ligand_record["molformer_dim"])
    if len(molformer) != molformer_dim:
        raise ContractError("query MoLFormer dimension does not match embedding manifest")
    ecfp = ecfp_bit_vector(
        canonical,
        n_bits=int(ligand_record["ecfp_bits"]),
        radius=int(ligand_record["ecfp_radius"]),
    )
    vector = np.concatenate([molformer, ecfp]).astype(np.float32)
    if len(vector) != int(ligand_record["dim"]):
        raise ContractError("query ligand dimension does not match embedding manifest")
    if not np.isfinite(vector).all():
        raise ContractError("query ligand features must all be finite")
    provenance = canonical_smiles_record(canonical)
    provenance.update({
        "embedding": {
                "ligand_checkpoint": {
                    "model_id": MOLFORMER_CHECKPOINT,
                    "revision": MOLFORMER_REVISION,
                    "resolved_revision": loaded_revision,
                },
                "tokenization": {
                    "strategy": "tokenizer_truncation_fixed_max_tokens",
                    "max_tokens": trained_max_length,
                    "truncation": True,
                },
                "ecfp": {
                    "radius": int(ligand_record["ecfp_radius"]),
                    "bits": int(ligand_record["ecfp_bits"]),
                    "encoding": "independent_binary_float32",
                },
            "source": "on_demand_query_embedding",
        }
    })
    query_id = "query_" + provenance["canonical_smiles_sha256"][:16]
    return query_id, vector, provenance


def fixture_query_ligand_features(
    *,
    smiles: str,
    ligand_embedding_csv: Path,
) -> tuple[str, np.ndarray, dict[str, Any]]:
    canonical = canonicalize_smiles(smiles)
    _, ligand_matrix = load_embedding_table(ligand_embedding_csv, "ligand_id")
    provenance = canonical_smiles_record(canonical)
    provenance.update({
        "embedding": {
            "source": "deterministic_fixture_query_embedding",
            "checkpoint_compatible_with": {
                "model_id": MOLFORMER_CHECKPOINT,
                "revision": MOLFORMER_REVISION,
            },
        }
    })
    query_id = "query_" + provenance["canonical_smiles_sha256"][:16]
    return query_id, deterministic_vector(provenance["canonical_smiles_sha256"], ligand_matrix.shape[1]), provenance


def train_fixture_model(
    train_csv: Path,
    embedding_manifest_path: Path,
    out_model_npz: Path,
    *,
    seeds: Sequence[int] = (11, 17, 23),
    steps: int = 240,
    learning_rate: float = 0.35,
) -> dict[str, Any]:
    embedding_manifest = validate_embedding_manifest(embedding_manifest_path)
    artifacts = embedding_manifest["artifacts"]
    ligand_ids, ligand_matrix = load_embedding_artifact(
        artifacts["ligands"], base_dir=embedding_manifest_path.parent, id_column="ligand_id", label="ligand embeddings"
    )
    target_ids, target_matrix = load_embedding_artifact(
        artifacts["targets"], base_dir=embedding_manifest_path.parent, id_column="target_id", label="target embeddings"
    )
    ligand_index = {value: i for i, value in enumerate(ligand_ids)}
    target_index = {value: i for i, value in enumerate(target_ids)}

    train = pd.read_csv(train_csv)
    measured, ranking_only = split_pu_training(train)
    fit_measured, calibration_holdout, split_evidence = calibration_holdout_split(measured)
    if fit_measured.empty or fit_measured["measured_label"].nunique() != 2:
        raise ContractError("fixture training requires measured positives and negatives")

    def feature_matrix(rows: pd.DataFrame) -> np.ndarray:
        features: list[np.ndarray] = []
        for _, row in rows.iterrows():
            ligand_id = str(row["ligand_id"]).strip()
            target_id = str(row["target_id"]).strip()
            if ligand_id not in ligand_index or target_id not in target_index:
                raise ContractError("training pair is missing a bound embedding")
            features.append(
                pair_features(
                    ligand_matrix[ligand_index[ligand_id]],
                    target_matrix[target_index[target_id]],
                )
            )
        if not features:
            return np.zeros((0, ligand_matrix.shape[1] * 4 + 1), dtype=np.float32)
        matrix = np.vstack(features).astype(np.float32)
        return np.hstack([matrix, np.ones((len(matrix), 1), dtype=np.float32)])

    weights = target_balanced_weights(fit_measured)
    y = fit_measured["measured_label"].astype(int).to_numpy(dtype=np.float32)
    x = feature_matrix(fit_measured)

    seed_weights = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        w = rng.normal(0.0, 0.02, size=x.shape[1]).astype(np.float64)
        for _ in range(steps):
            logits = np.clip(x @ w, -60, 60)
            pred = 1.0 / (1.0 + np.exp(-logits))
            grad = (x.T @ ((pred - y) * weights)) / len(y) + 1e-3 * w
            w -= learning_rate * grad
        seed_weights.append(w.astype(np.float32))
    coef = np.mean(np.vstack(seed_weights), axis=0)
    model_state_sha256 = sha256_ndarrays({"coef": coef})
    out_model_npz.parent.mkdir(parents=True, exist_ok=True)
    model_tmp = out_model_npz.with_name(out_model_npz.name + ".tmp")
    with model_tmp.open("wb") as handle:
        np.savez(
            handle,
            coef=coef,
            seeds=np.array(list(seeds), dtype=np.int64),
            model_state_sha256=np.array(model_state_sha256),
        )
    model_tmp.replace(out_model_npz)

    holdout_x = feature_matrix(calibration_holdout)
    holdout_scores = holdout_x @ coef if len(holdout_x) else np.zeros(0, dtype=np.float32)
    holdout_labels = calibration_holdout["measured_label"].astype(int).to_numpy()
    direct_mask = calibration_holdout["ontology"].astype(str).eq("direct_binding_reversible").to_numpy()
    func_mask = calibration_holdout["ontology"].astype(str).eq("functional_modulation").to_numpy()
    direct = fit_platt(holdout_scores[direct_mask], holdout_labels[direct_mask]) if direct_mask.any() else None
    functional = fit_platt(holdout_scores[func_mask], holdout_labels[func_mask]) if func_mask.any() else None
    return {
        "trainable_params": int(x.shape[1]),
        "effective_target_count": effective_target_count(fit_measured, weights),
        "ranking_only_pairs": int(len(ranking_only)),
        "calibration": calibration_manifest(
            direct,
            functional,
            split_evidence=split_evidence,
            model_state_sha256=model_state_sha256,
        ),
        "calibration_split": split_evidence,
        "model_state_sha256": model_state_sha256,
        "train_ligands": sorted(fit_measured["ligand_id"].astype(str).unique().tolist()),
        "train_targets": sorted(fit_measured["target_id"].astype(str).unique().tolist()),
        "execution_mode": "fixture",
        "peak_vram_gib": 0.0,
        "peak_vram_source": "not_applicable_cpu",
        "peak_vram_measured": False,
        "actual_batch_size": None,
    }


def _make_torch_pair_head(
    ligand_dim: int,
    target_dim: int,
    projection_dim: int,
):
    import torch
    import torch.nn.functional as functional

    class PairHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ligand_projection = torch.nn.Linear(ligand_dim, projection_dim, bias=False)
            self.target_projection = torch.nn.Linear(target_dim, projection_dim, bias=False)
            self.gate = torch.nn.Linear(2 * projection_dim, projection_dim)
            self.interaction = torch.nn.Linear(3 * projection_dim, projection_dim)
            self.ontology_head = torch.nn.Linear(projection_dim, len(ONTOLOGY))
            self.log_temperature = torch.nn.Parameter(torch.zeros(()))

        def project_ligands(self, ligands):
            return functional.normalize(self.ligand_projection(ligands), dim=-1)

        def project_targets(self, targets):
            return functional.normalize(self.target_projection(targets), dim=-1)

        def pair_logits(self, ligands, targets):
            ligand_z = self.project_ligands(ligands)
            target_z = self.project_targets(targets)
            gate = torch.sigmoid(self.gate(torch.cat([ligand_z, target_z], dim=-1)))
            fused = gate * ligand_z + (1.0 - gate) * target_z
            interaction = torch.cat(
                [fused, ligand_z * target_z, torch.abs(ligand_z - target_z)], dim=-1
            )
            ontology_logits = self.ontology_head(functional.silu(self.interaction(interaction)))
            temperature = torch.exp(self.log_temperature).clamp(0.05, 100.0)
            ranking = torch.sum(ligand_z * target_z, dim=-1) / temperature
            return ontology_logits, ranking

    return PairHead()


def train_torch_pair_model(
    train_csv: Path,
    embedding_manifest_path: Path,
    out_model_pt: Path,
    *,
    seeds: Sequence[int] = PRODUCTION_SEEDS,
    epochs: int = 80,
    projection_dim: int = 128,
    learning_rate: float = 3e-3,
    batch_size: int = 256,
    device_mode: str = "cuda",
    calibration_fraction: float = 0.2,
    max_contrastive_targets: int = 64,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    if tuple(map(int, seeds)) != PRODUCTION_SEEDS:
        raise ContractError(f"production training seeds must be exactly {PRODUCTION_SEEDS}")
    if epochs < 1 or projection_dim < 1 or batch_size < 1:
        raise ContractError("epochs, projection_dim, and batch_size must be positive")
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ContractError("learning_rate must be finite and positive")
    if max_contrastive_targets < 2:
        raise ContractError("max_contrastive_targets must be at least two")
    if device_mode not in {"cpu", "cuda"}:
        raise ContractError("device_mode must be cpu or cuda")
    if device_mode == "cuda" and not torch.cuda.is_available():
        raise ContractError("CUDA training requested but unavailable; use explicit CPU mode")
    device = torch.device(device_mode)
    use_amp = device.type == "cuda"
    if use_amp:
        torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    embedding_manifest = validate_embedding_manifest(embedding_manifest_path)
    artifacts = embedding_manifest["artifacts"]
    ligand_store = EmbeddingStore(
        artifacts["ligands"],
        base_dir=embedding_manifest_path.parent,
        id_column="ligand_id",
        label="ligand embeddings",
        validated=True,
    )
    target_store = EmbeddingStore(
        artifacts["targets"],
        base_dir=embedding_manifest_path.parent,
        id_column="target_id",
        label="target embeddings",
        cache_shards=max(8, len(artifacts["targets"].get("shards", ()))),
        validated=True,
    )
    ligand_dim = ligand_store.dim
    target_dim = target_store.dim

    train = pd.read_csv(train_csv)
    measured, ranking_only = split_pu_training(train)
    fit_measured, calibration_holdout, split_evidence = calibration_holdout_split(
        measured, fraction=calibration_fraction
    )
    if fit_measured.empty or fit_measured["measured_label"].nunique() != 2:
        raise ContractError("performance-v2 fit split requires measured positives and negatives")

    ontology_index = {event: index for index, event in enumerate(ONTOLOGY)}

    def pair_indexes(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        ligand_ids = frame["ligand_id"].astype("string").fillna("").str.strip()
        target_ids = frame["target_id"].astype("string").fillna("").str.strip()
        ligand_indexes = ligand_ids.map(ligand_store.id_to_index)
        target_indexes = target_ids.map(target_store.id_to_index)
        if ligand_indexes.isna().any() or target_indexes.isna().any():
            raise ContractError("training pair is missing a bound embedding")
        return (
            ligand_indexes.to_numpy(dtype=np.int64),
            target_indexes.to_numpy(dtype=np.int64),
        )

    fit_ligand_indexes, fit_target_indexes = pair_indexes(fit_measured)
    holdout_ligand_indexes, holdout_target_indexes = pair_indexes(calibration_holdout)
    labels = fit_measured["measured_label"].astype(int).to_numpy(dtype=np.float32)
    ontology_indexes = fit_measured["ontology"].astype(str).map(ontology_index).to_numpy(dtype=np.int64)
    weights = target_balanced_weights(fit_measured).astype(np.float32)

    positive_by_ligand: dict[int, set[int]] = {}
    candidate_by_ligand: dict[int, set[int]] = {}
    for ligand_idx, target_idx, label in zip(fit_ligand_indexes, fit_target_indexes, labels, strict=True):
        ligand_idx = int(ligand_idx)
        target_idx = int(target_idx)
        candidate_by_ligand.setdefault(ligand_idx, set()).add(target_idx)
        if int(label) == 1:
            positive_by_ligand.setdefault(ligand_idx, set()).add(target_idx)
    holdout_ligand_targets = {
        (str(ligand_id).strip(), str(target_id).strip())
        for ligand_id, target_id in zip(
            calibration_holdout["ligand_id"],
            calibration_holdout["target_id"],
            strict=True,
        )
    }
    ranking_only_used = 0
    for ligand_value, target_value in zip(ranking_only["ligand_id"], ranking_only["target_id"], strict=True):
        ligand_id = str(ligand_value).strip()
        target_id = str(target_value).strip()
        if not ligand_id or not target_id:
            raise ContractError("ranking-only contrastive pair requires ligand_id and target_id")
        if (ligand_id, target_id) in holdout_ligand_targets:
            continue
        if ligand_id not in ligand_store.id_to_index or target_id not in target_store.id_to_index:
            raise ContractError("ranking-only pair is missing a bound embedding")
        ligand_idx = ligand_store.id_to_index[ligand_id]
        if ligand_idx not in positive_by_ligand:
            continue
        candidate_by_ligand.setdefault(ligand_idx, set()).add(target_store.id_to_index[target_id])
        ranking_only_used += 1
    anchors = sorted(
        ligand_idx
        for ligand_idx, positives in positive_by_ligand.items()
        if candidate_by_ligand.get(ligand_idx, set()) - positives
    )
    pair_locality = np.fromiter(
        (ligand_store.locality_key(int(index)) for index in fit_ligand_indexes),
        dtype=np.int64,
        count=len(fit_ligand_indexes),
    )
    anchor_locality = np.fromiter(
        (ligand_store.locality_key(index) for index in anchors),
        dtype=np.int64,
        count=len(anchors),
    )

    probe = _make_torch_pair_head(ligand_dim, target_dim, projection_dim)
    trainable_params = int(sum(parameter.numel() for parameter in probe.parameters()))
    if trainable_params > MAX_TRAINABLE_PARAMS:
        raise ContractError(f"trainable_params exceeds {MAX_TRAINABLE_PARAMS}: {trainable_params}")
    del probe

    class IndexDataset(torch.utils.data.Dataset):
        def __init__(self, length: int) -> None:
            self.length = int(length)

        def __len__(self) -> int:
            return self.length

        def __getitem__(self, index: int) -> int:
            return int(index)

    def to_device(matrix: np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(matrix)).to(device, non_blocking=use_amp)

    def contrastive_loss(model, anchor_offsets: np.ndarray, rng: np.random.Generator):
        selected_ligands: list[int] = []
        selected_targets: list[list[int]] = []
        positive_counts: list[int] = []
        for anchor_offset in anchor_offsets:
            ligand_idx = anchors[int(anchor_offset)]
            positives = sorted(positive_by_ligand[ligand_idx])
            negatives = sorted(candidate_by_ligand[ligand_idx] - positive_by_ligand[ligand_idx])
            positive_limit = min(len(positives), max_contrastive_targets - 1)
            if len(positives) > positive_limit:
                chosen = rng.choice(positives, size=positive_limit, replace=False).tolist()
                chosen_positives = sorted(map(int, chosen))
            else:
                chosen_positives = positives
            negative_limit = min(len(negatives), max_contrastive_targets - len(chosen_positives))
            if negative_limit < 1:
                continue
            if len(negatives) > negative_limit:
                chosen = rng.choice(negatives, size=negative_limit, replace=False).tolist()
                chosen_negatives = sorted(map(int, chosen))
            else:
                chosen_negatives = negatives
            selected_ligands.append(ligand_idx)
            selected_targets.append(chosen_positives + chosen_negatives)
            positive_counts.append(len(chosen_positives))
        if not selected_ligands:
            return torch.zeros((), device=device)
        flat_targets = [target for row in selected_targets for target in row]
        ligand_z = model.project_ligands(to_device(ligand_store.fetch(selected_ligands)))
        target_z = model.project_targets(to_device(target_store.fetch(flat_targets)))
        temperature = torch.exp(model.log_temperature).clamp(0.05, 100.0)
        terms = []
        cursor = 0
        for row_index, (targets_for_ligand, positive_count) in enumerate(
            zip(selected_targets, positive_counts, strict=True)
        ):
            width = len(targets_for_ligand)
            logits = (target_z[cursor : cursor + width] * ligand_z[row_index]).sum(dim=-1) / temperature
            numerator = torch.logsumexp(logits[:positive_count], dim=0)
            denominator = torch.logsumexp(logits, dim=0)
            terms.append(-(numerator - denominator))
            cursor += width
        return torch.stack(terms).mean()

    def train_one_seed(seed: int, active_batch_size: int) -> dict[str, Any]:
        torch.manual_seed(seed)
        if use_amp:
            torch.cuda.manual_seed_all(seed)
        model = _make_torch_pair_head(ligand_dim, target_dim, projection_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        pair_loader = torch.utils.data.DataLoader(
            IndexDataset(len(fit_measured)),
            batch_sampler=DeterministicShardBatchSampler(
                pair_locality, active_batch_size, seed
            ),
            num_workers=0,
        )
        anchor_loader = None
        if anchors:
            anchor_loader = torch.utils.data.DataLoader(
                IndexDataset(len(anchors)),
                batch_sampler=DeterministicShardBatchSampler(
                    anchor_locality, active_batch_size, seed + 1_000_003
                ),
                num_workers=0,
            )
        rng = np.random.default_rng(seed)
        model.train()
        global_weight_sum = float(weights.sum())
        if not math.isfinite(global_weight_sum) or global_weight_sum <= 0.0:
            raise ContractError("target-balanced training weights must have positive finite mass")
        for _ in range(epochs):
            anchor_iterator = iter(anchor_loader) if anchor_loader is not None else None
            optimizer.zero_grad(set_to_none=True)
            for row_offsets_tensor in pair_loader:
                row_offsets = row_offsets_tensor.numpy().astype(np.int64, copy=False)
                ligand_batch = to_device(ligand_store.fetch(fit_ligand_indexes[row_offsets]))
                target_batch = to_device(target_store.fetch(fit_target_indexes[row_offsets]))
                label_batch = torch.from_numpy(labels[row_offsets]).to(device)
                ontology_batch = torch.from_numpy(ontology_indexes[row_offsets]).to(device)
                weight_batch = torch.from_numpy(weights[row_offsets]).to(device)
                if anchor_iterator is None:
                    anchor_offsets = np.zeros(0, dtype=np.int64)
                else:
                    try:
                        anchor_offsets = next(anchor_iterator).numpy().astype(np.int64, copy=False)
                    except StopIteration:
                        anchor_iterator = iter(anchor_loader)
                        anchor_offsets = next(anchor_iterator).numpy().astype(np.int64, copy=False)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    event_logits, _ = model.pair_logits(ligand_batch, target_batch)
                    selected_logits = event_logits.gather(1, ontology_batch[:, None]).squeeze(1)
                    bce_rows = functional.binary_cross_entropy_with_logits(
                        selected_logits, label_batch, reduction="none"
                    )
                    bce_loss = (bce_rows * weight_batch).sum() / global_weight_sum
                    ranking_loss = contrastive_loss(model, anchor_offsets, rng)
                    loss = bce_loss + 0.25 * ranking_loss / max(1, len(pair_loader))
                scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        return {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

    states: list[dict[str, Any]] = []
    actual_batch_sizes: list[int] = []
    for seed_value in map(int, seeds):
        active_batch_size = int(batch_size)
        while True:
            try:
                states.append(train_one_seed(seed_value, active_batch_size))
                actual_batch_sizes.append(active_batch_size)
                break
            except RuntimeError as exc:
                is_oom = use_amp and "out of memory" in str(exc).lower()
                if not is_oom or active_batch_size <= 1:
                    raise
                active_batch_size = max(1, active_batch_size // 2)
                gc.collect()
                torch.cuda.empty_cache()

    for state_index, state in enumerate(states):
        for key, value in state.items():
            if not bool(torch.isfinite(value).all()):
                raise ContractError(
                    f"ensemble model state contains nonfinite values: seed={state_index} key={key}"
                )
    model_state_sha256 = sha256_ndarrays(
        {
            f"seed_{state_index}.{key}": value.detach().cpu().numpy()
            for state_index, state in enumerate(states)
            for key, value in state.items()
        }
    )
    ensemble_models = []
    for state in states:
        model = _make_torch_pair_head(ligand_dim, target_dim, projection_dim).to(device)
        model.load_state_dict(state)
        model.eval()
        ensemble_models.append(model)

    holdout_event_logits: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(calibration_holdout), min(actual_batch_sizes)):
            stop = min(start + min(actual_batch_sizes), len(calibration_holdout))
            ligand_batch = to_device(ligand_store.fetch(holdout_ligand_indexes[start:stop]))
            target_batch = to_device(target_store.fetch(holdout_target_indexes[start:stop]))
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                seed_logits = [model.pair_logits(ligand_batch, target_batch)[0] for model in ensemble_models]
                logits = torch.stack(seed_logits, dim=0).mean(dim=0)
            holdout_event_logits.append(logits.detach().float().cpu().numpy())
    if holdout_event_logits:
        calibration_logits = np.vstack(holdout_event_logits)
    else:
        calibration_logits = np.zeros((0, len(ONTOLOGY)), dtype=np.float32)
    holdout_labels = calibration_holdout["measured_label"].astype(int).to_numpy()
    direct_mask = calibration_holdout["ontology"].astype(str).eq("direct_binding_reversible").to_numpy()
    functional_mask = calibration_holdout["ontology"].astype(str).eq("functional_modulation").to_numpy()
    direct = fit_platt(
        calibration_logits[direct_mask, ontology_index["direct_binding_reversible"]],
        holdout_labels[direct_mask],
    ) if direct_mask.any() else None
    functional_calibrator = fit_platt(
        calibration_logits[functional_mask, ontology_index["functional_modulation"]],
        holdout_labels[functional_mask],
    ) if functional_mask.any() else None

    out_model_pt.parent.mkdir(parents=True, exist_ok=True)
    model_tmp = out_model_pt.with_name(out_model_pt.name + ".tmp")
    torch.save(
        {
            "architecture": "separate_projection_gated_ontology_prediction_ensemble_v3",
            "state_dicts": states,
            "model_state_sha256": model_state_sha256,
            "ligand_dim": ligand_dim,
            "target_dim": target_dim,
            "projection_dim": int(projection_dim),
            "ontology": list(ONTOLOGY),
            "seeds": list(map(int, seeds)),
            "seed_aggregation": "prediction_mean",
            "embedding_feature_contract_sha256": sha256_payload(
                embedding_feature_contract(embedding_manifest)
            ),
        },
        model_tmp,
    )
    model_tmp.replace(out_model_pt)
    if use_amp:
        torch.cuda.synchronize(device)
        peak_bytes = int(torch.cuda.max_memory_allocated(device))
        if peak_bytes <= 0:
            raise ContractError("CUDA peak VRAM measurement is unavailable")
        peak_vram_gib = peak_bytes / float(1024**3)
        peak_source = "torch.cuda.max_memory_allocated"
        peak_measured = True
    else:
        peak_vram_gib = 0.0
        peak_source = "not_applicable_cpu"
        peak_measured = False

    used_ligand_indexes = set(map(int, fit_ligand_indexes.tolist()))
    used_target_indexes = set(map(int, fit_target_indexes.tolist()))
    for ligand_idx in anchors:
        used_ligand_indexes.add(ligand_idx)
        used_target_indexes.update(candidate_by_ligand[ligand_idx])
    return {
        "trainable_params": trainable_params,
        "effective_target_count": effective_target_count(
            fit_measured, target_balanced_weights(fit_measured)
        ),
        "ranking_only_pairs": int(len(ranking_only)),
        "ranking_only_pairs_used": int(ranking_only_used),
        "calibration": calibration_manifest(
            direct,
            functional_calibrator,
            split_evidence=split_evidence,
            model_state_sha256=model_state_sha256,
            logit_source="saved_prediction_ensemble",
        ),
        "calibration_split": split_evidence,
        "model_state_sha256": model_state_sha256,
        "train_ligands": sorted(ligand_store.ids[index] for index in used_ligand_indexes),
        "train_targets": sorted(target_store.ids[index] for index in used_target_indexes),
        "execution_mode": device.type,
        "peak_vram_gib": float(peak_vram_gib),
        "peak_vram_source": peak_source,
        "peak_vram_measured": peak_measured,
        "requested_batch_size": int(batch_size),
        "actual_batch_size": int(min(actual_batch_sizes)),
        "per_seed_batch_sizes": actual_batch_sizes,
        "mixed_precision": bool(use_amp),
        "contrastive_max_targets": int(max_contrastive_targets),
        "batch_locality": "ligand_embedding_shard_with_global_epoch_objective_normalization",
        "target_cache_shards": max(8, len(artifacts["targets"].get("shards", ()))),
    }


def score_fixture_model(
    *,
    ligand_id: str,
    model_npz: Path,
    embedding_manifest_path: Path,
    train_ligands: Sequence[str],
    train_targets: Sequence[str],
    calibration: Mapping[str, Any],
    min_score: float | None = None,
) -> pd.DataFrame:
    embedding_manifest = validate_embedding_manifest(embedding_manifest_path)
    artifacts = embedding_manifest["artifacts"]
    ligand_ids, ligand_matrix = load_embedding_artifact(
        artifacts["ligands"], base_dir=embedding_manifest_path.parent, id_column="ligand_id", label="ligand embeddings"
    )
    target_ids, target_matrix = load_embedding_artifact(
        artifacts["targets"], base_dir=embedding_manifest_path.parent, id_column="target_id", label="target embeddings"
    )
    if ligand_id not in ligand_ids:
        raise ContractError(f"ligand_id is missing from embeddings: {ligand_id}")
    coef = np.load(model_npz)["coef"].astype(np.float32)
    ligand_vec = ligand_matrix[ligand_ids.index(ligand_id)]
    train_ligand_domain = frozenset(map(str, train_ligands))
    train_target_domain = frozenset(map(str, train_targets))
    rows: list[dict[str, Any]] = []
    for target_id, target_vec in zip(target_ids, target_matrix, strict=True):
        features = np.concatenate([pair_features(ligand_vec, target_vec), np.ones(1, dtype=np.float32)])
        score = float(features @ coef)
        route = ood_route(ligand_id, target_id, train_ligand_domain, train_target_domain)
        abstained = should_abstain(route, score=score, min_score=min_score)
        direct_record = calibration.get("direct_binding_reversible") if isinstance(calibration, Mapping) else None
        func_record = calibration.get("functional_modulation") if isinstance(calibration, Mapping) else None
        direct_probability: float | None = None
        functional_probability: float | None = None
        if not abstained and isinstance(direct_record, Mapping) and direct_record.get("status") == "measured_event_calibrated":
            direct_probability = float(PlattCalibrator(float(direct_record["a"]), float(direct_record["b"]), int(direct_record["n_pos"]), int(direct_record["n_neg"])).predict([score])[0])
        if not abstained and isinstance(func_record, Mapping) and func_record.get("status") == "measured_event_calibrated":
            functional_probability = float(PlattCalibrator(float(func_record["a"]), float(func_record["b"]), int(func_record["n_pos"]), int(func_record["n_neg"])).predict([score])[0])
        rows.append({
            "ligand_id": ligand_id,
            "target_id": target_id,
            "performance_v2_ranking_score": score,
            "performance_v2_score_semantics": "ranking_score_not_probability",
            "performance_v2_ood_route": route,
            "performance_v2_abstained": bool(abstained),
            "performance_v2_direct_binding_probability": direct_probability,
            "performance_v2_functional_modulation_probability": functional_probability,
        })
    ranked = pd.DataFrame(rows).sort_values(
        ["performance_v2_abstained", "performance_v2_ranking_score", "target_id"],
        ascending=[True, False, True],
    )
    ranked["performance_v2_rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def score_fixture_query_vector(
    *,
    query_ligand_id: str,
    ligand_vector: np.ndarray,
    model_npz: Path,
    embedding_manifest_path: Path,
    train_ligands: Sequence[str],
    train_targets: Sequence[str],
    calibration: Mapping[str, Any],
    min_score: float | None = None,
) -> pd.DataFrame:
    embedding_manifest = validate_embedding_manifest(embedding_manifest_path)
    artifacts = embedding_manifest["artifacts"]
    target_ids, target_matrix = load_embedding_artifact(
        artifacts["targets"], base_dir=embedding_manifest_path.parent, id_column="target_id", label="target embeddings"
    )
    coef = np.load(model_npz)["coef"].astype(np.float32)
    if len(ligand_vector) != target_matrix.shape[1]:
        # Fixture embeddings are symmetric; production .pt scoring handles
        # distinct ligand and target dimensions.
        raise ContractError("fixture query embedding dimension does not match target fixture embeddings")
    train_ligand_domain = frozenset(map(str, train_ligands))
    train_target_domain = frozenset(map(str, train_targets))
    rows: list[dict[str, Any]] = []
    for target_id, target_vec in zip(target_ids, target_matrix, strict=True):
        features = np.concatenate([pair_features(ligand_vector, target_vec), np.ones(1, dtype=np.float32)])
        score = float(features @ coef)
        route = ood_route(
            query_ligand_id, target_id, train_ligand_domain, train_target_domain
        )
        abstained = should_abstain(route, score=score, min_score=min_score)
        direct_record = calibration.get("direct_binding_reversible") if isinstance(calibration, Mapping) else None
        func_record = calibration.get("functional_modulation") if isinstance(calibration, Mapping) else None
        direct_probability: float | None = None
        functional_probability: float | None = None
        if not abstained and isinstance(direct_record, Mapping) and direct_record.get("status") == "measured_event_calibrated":
            direct_probability = float(PlattCalibrator(float(direct_record["a"]), float(direct_record["b"]), int(direct_record["n_pos"]), int(direct_record["n_neg"])).predict([score])[0])
        if not abstained and isinstance(func_record, Mapping) and func_record.get("status") == "measured_event_calibrated":
            functional_probability = float(PlattCalibrator(float(func_record["a"]), float(func_record["b"]), int(func_record["n_pos"]), int(func_record["n_neg"])).predict([score])[0])
        rows.append({
            "ligand_id": query_ligand_id,
            "target_id": target_id,
            "performance_v2_ranking_score": score,
            "performance_v2_score_semantics": "ranking_score_not_probability",
            "performance_v2_ood_route": route,
            "performance_v2_abstained": bool(abstained),
            "performance_v2_direct_binding_probability": direct_probability,
            "performance_v2_functional_modulation_probability": functional_probability,
        })
    ranked = pd.DataFrame(rows).sort_values(
        ["performance_v2_abstained", "performance_v2_ranking_score", "target_id"],
        ascending=[True, False, True],
    )
    ranked["performance_v2_rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def _placed_scoring_model(
    torch: Any,
    ligand_dim: int,
    target_dim: int,
    projection_dim: int,
    *,
    device_mode: str | None,
) -> tuple[Any, Any]:
    """Put the scoring head on a device, falling back to CPU when CUDA cannot.

    Scoring used to hardcode CUDA whenever a GPU was merely visible, with no
    way to ask for CPU. A model trained on CPU then died on a busy or
    memory-constrained GPU before a single batch ran - the existing retry only
    covers out-of-memory raised during batching, not failure to place the
    model. Scoring is a forward pass, so CPU is always a valid answer.
    """
    if device_mode is not None and device_mode not in {"cpu", "cuda"}:
        raise ContractError("device_mode must be cpu or cuda")
    if device_mode == "cuda" and not torch.cuda.is_available():
        raise ContractError("device_mode is cuda but no CUDA device is available")
    requested = device_mode or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(requested)
        return device, _make_torch_pair_head(ligand_dim, target_dim, projection_dim).to(device)
    except RuntimeError:
        if device_mode is not None or requested != "cuda":
            raise
        warnings.warn(
            "CUDA is visible but unusable for scoring; falling back to CPU",
            RuntimeWarning,
            stacklevel=2,
        )
        device = torch.device("cpu")
        return device, _make_torch_pair_head(ligand_dim, target_dim, projection_dim).to(device)


def score_torch_pair_model(
    *,
    ligand_id: str,
    model_pt: Path,
    embedding_manifest_path: Path,
    train_ligands: Sequence[str],
    train_targets: Sequence[str],
    calibration: Mapping[str, Any],
    min_score: float | None = None,
    device_mode: str | None = None,
) -> pd.DataFrame:
    embedding_manifest = validate_embedding_manifest(embedding_manifest_path)
    ligand_store = EmbeddingStore(
        embedding_manifest["artifacts"]["ligands"],
        base_dir=embedding_manifest_path.parent,
        id_column="ligand_id",
        label="ligand embeddings",
        validated=True,
    )
    ligand_vector = ligand_store.vector_for_id(ligand_id)
    return _score_torch_vector(
        query_ligand_id=ligand_id,
        ligand_vector=ligand_vector,
        model_pt=model_pt,
        embedding_manifest_path=embedding_manifest_path,
        train_ligands=train_ligands,
        train_targets=train_targets,
        calibration=calibration,
        min_score=min_score,
        device_mode=device_mode,
    )


def score_torch_query_vector(
    *,
    query_ligand_id: str,
    ligand_vector: np.ndarray,
    model_pt: Path,
    embedding_manifest_path: Path,
    train_ligands: Sequence[str],
    train_targets: Sequence[str],
    calibration: Mapping[str, Any],
    min_score: float | None = None,
    device_mode: str | None = None,
) -> pd.DataFrame:
    return _score_torch_vector(
        query_ligand_id=query_ligand_id,
        ligand_vector=ligand_vector,
        model_pt=model_pt,
        embedding_manifest_path=embedding_manifest_path,
        train_ligands=train_ligands,
        train_targets=train_targets,
        calibration=calibration,
        min_score=min_score,
        device_mode=device_mode,
    )


def _score_torch_vector(
    *,
    query_ligand_id: str,
    ligand_vector: np.ndarray,
    model_pt: Path,
    embedding_manifest_path: Path,
    train_ligands: Sequence[str],
    train_targets: Sequence[str],
    calibration: Mapping[str, Any],
    min_score: float | None,
    device_mode: str | None = None,
    score_batch_size: int = 2048,
) -> pd.DataFrame:
    import torch

    embedding_manifest = validate_embedding_manifest(embedding_manifest_path)
    target_store = EmbeddingStore(
        embedding_manifest["artifacts"]["targets"],
        base_dir=embedding_manifest_path.parent,
        id_column="target_id",
        label="target embeddings",
        validated=True,
    )
    try:
        checkpoint = torch.load(model_pt, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise ContractError(
            "installed Torch lacks safe weights_only checkpoint loading; upgrade Torch"
        ) from exc
    if checkpoint.get("architecture") != "separate_projection_gated_ontology_prediction_ensemble_v3":
        raise ContractError("torch model architecture is unsupported")
    if tuple(checkpoint.get("ontology") or ()) != ONTOLOGY:
        raise ContractError("torch model ontology order does not match contract")
    expected_feature_hash = sha256_payload(embedding_feature_contract(embedding_manifest))
    if checkpoint.get("embedding_feature_contract_sha256") != expected_feature_hash:
        raise ContractError("torch model checkpoint feature contract does not match embeddings")
    state_dicts = checkpoint.get("state_dicts")
    if (
        not isinstance(state_dicts, Sequence)
        or isinstance(state_dicts, (str, bytes))
        or len(state_dicts) != len(PRODUCTION_SEEDS)
    ):
        raise ContractError("torch model requires one state_dict per preregistered seed")
    for state_index, state_dict in enumerate(state_dicts):
        if not isinstance(state_dict, Mapping) or not state_dict:
            raise ContractError(f"torch model state_dict is missing for seed {state_index}")
        for key, value in state_dict.items():
            if not hasattr(value, "detach") or not bool(torch.isfinite(value).all()):
                raise ContractError(f"torch model state contains nonfinite values: {key}")
    ligand_dim = int(checkpoint["ligand_dim"])
    target_dim = int(checkpoint["target_dim"])
    projection_dim = int(checkpoint["projection_dim"])
    if len(ligand_vector) != ligand_dim or target_store.dim != target_dim:
        raise ContractError("query or target embedding dimensions do not match model artifact")
    state_hash = sha256_ndarrays(
        {
            f"seed_{state_index}.{key}": value.detach().cpu().numpy()
            for state_index, state_dict in enumerate(state_dicts)
            for key, value in state_dict.items()
        }
    )
    if state_hash != checkpoint.get("model_state_sha256"):
        raise ContractError("torch model state hash mismatch")
    if state_hash != calibration.get("model_state_sha256"):
        raise ContractError("calibration is not bound to the scored torch model state")
    device, first_model = _placed_scoring_model(
        torch, ligand_dim, target_dim, projection_dim, device_mode=device_mode
    )
    models = [first_model]
    models.extend(
        _make_torch_pair_head(ligand_dim, target_dim, projection_dim).to(device)
        for _ in state_dicts[1:]
    )
    for model, state_dict in zip(models, state_dicts, strict=True):
        model.load_state_dict(state_dict)
        model.eval()
    ligand_array = np.asarray(ligand_vector, dtype=np.float32)
    direct_index = ONTOLOGY.index("direct_binding_reversible")
    functional_index = ONTOLOGY.index("functional_modulation")
    direct_record = calibration.get("direct_binding_reversible") if isinstance(calibration, Mapping) else None
    func_record = calibration.get("functional_modulation") if isinstance(calibration, Mapping) else None
    direct_calibrator = None
    functional_calibrator = None
    if isinstance(direct_record, Mapping) and direct_record.get("status") == "measured_event_calibrated":
        direct_calibrator = PlattCalibrator(
            float(direct_record["a"]),
            float(direct_record["b"]),
            int(direct_record["n_pos"]),
            int(direct_record["n_neg"]),
        )
    if isinstance(func_record, Mapping) and func_record.get("status") == "measured_event_calibrated":
        functional_calibrator = PlattCalibrator(
            float(func_record["a"]),
            float(func_record["b"]),
            int(func_record["n_pos"]),
            int(func_record["n_neg"]),
        )

    train_ligand_domain = frozenset(map(str, train_ligands))
    train_target_domain = frozenset(map(str, train_targets))
    rows: list[dict[str, Any]] = []
    active_batch_size = int(score_batch_size)
    start = 0
    while start < len(target_store):
        stop = min(start + active_batch_size, len(target_store))
        target_indexes = np.arange(start, stop, dtype=np.int64)
        try:
            target_matrix = target_store.fetch(target_indexes)
            ligand_rows = np.repeat(ligand_array[None, :], len(target_indexes), axis=0)
            with torch.no_grad():
                ligand_tensor = torch.from_numpy(ligand_rows).to(device)
                target_tensor = torch.from_numpy(target_matrix).to(device)
                seed_outputs = [model.pair_logits(ligand_tensor, target_tensor) for model in models]
                event_logits = torch.stack([output[0] for output in seed_outputs], dim=0).mean(dim=0)
                ranking_scores = torch.stack([output[1] for output in seed_outputs], dim=0).mean(dim=0)
            scores = ranking_scores.detach().float().cpu().numpy()
            event_scores = event_logits.detach().float().cpu().numpy()
        except RuntimeError as exc:
            if device.type != "cuda" or "out of memory" not in str(exc).lower() or active_batch_size <= 1:
                raise
            torch.cuda.empty_cache()
            active_batch_size = max(1, active_batch_size // 2)
            continue
        target_ids = target_store.ids[start:stop]
        for row_offset, (target_id, score) in enumerate(zip(target_ids, scores, strict=True)):
            route = ood_route(
                query_ligand_id, target_id, train_ligand_domain, train_target_domain
            )
            abstained = should_abstain(route, score=float(score), min_score=min_score)
            direct_probability = (
                float(direct_calibrator.predict([event_scores[row_offset, direct_index]])[0])
                if direct_calibrator is not None and not abstained
                else None
            )
            functional_probability = (
                float(functional_calibrator.predict([event_scores[row_offset, functional_index]])[0])
                if functional_calibrator is not None and not abstained
                else None
            )
            rows.append({
                "ligand_id": query_ligand_id,
                "target_id": target_id,
                "performance_v2_ranking_score": float(score),
                "performance_v2_score_semantics": "ranking_score_not_probability",
                "performance_v2_ood_route": route,
                "performance_v2_abstained": bool(abstained),
                "performance_v2_direct_binding_probability": direct_probability,
                "performance_v2_functional_modulation_probability": functional_probability,
            })
        start = stop
    ranked = pd.DataFrame(rows).sort_values(
        ["performance_v2_abstained", "performance_v2_ranking_score", "target_id"],
        ascending=[True, False, True],
    )
    ranked["performance_v2_rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def ranking_manifest(
    *,
    ranking_csv: Path,
    model_manifest: Path,
    query_ligand_id: str,
    rows: int,
    query_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": RANKING_SCHEMA,
        "created_at": utc_now(),
        "query": {"ligand_id": str(query_ligand_id), **dict(query_provenance or {})},
        "inputs": {
            "model_manifest": artifact_record(model_manifest),
        },
        "outputs": {
            "ranking_csv": artifact_record(ranking_csv, rows=rows),
        },
        "semantics": {
            "ranking_score": "not_probability",
            "probability_columns": "present only with measured-event Platt calibrator coverage",
        },
    }
    return payload


RANKING_COLUMNS = (
    "ligand_id",
    "target_id",
    "performance_v2_ranking_score",
    "performance_v2_score_semantics",
    "performance_v2_ood_route",
    "performance_v2_abstained",
    "performance_v2_direct_binding_probability",
    "performance_v2_functional_modulation_probability",
    "performance_v2_rank",
)
if len(RANKING_COLUMNS) != len(set(RANKING_COLUMNS)):
    raise RuntimeError("performance-v2 ranking columns must be unique")


def validate_ranking_manifest(path: Path) -> dict[str, Any]:
    payload = read_json_object(path, "ranking manifest")
    require_schema(payload, RANKING_SCHEMA, "ranking manifest")
    inputs = payload.get("inputs")
    outputs = payload.get("outputs")
    if not isinstance(inputs, Mapping) or not isinstance(outputs, Mapping):
        raise ContractError("ranking manifest requires inputs and outputs")
    model_manifest_path = validate_artifact(
        inputs.get("model_manifest"), base_dir=path.parent, label="ranking model_manifest"
    )
    model_manifest = validate_model_manifest(model_manifest_path)
    ranking_csv = validate_artifact(
        outputs.get("ranking_csv"), base_dir=path.parent, label="ranking CSV"
    )
    frame = pd.read_csv(ranking_csv)
    if tuple(frame.columns) != RANKING_COLUMNS:
        raise ContractError("ranking CSV columns do not match performance-v2 contract")
    expected_rows = int(outputs["ranking_csv"].get("rows", -1))
    if expected_rows != len(frame):
        raise ContractError("ranking CSV row count does not match ranking manifest")
    if frame.empty:
        raise ContractError("ranking CSV must contain at least one target")
    ligand_ids = frame["ligand_id"].fillna("").astype(str).str.strip()
    targets = frame["target_id"].fillna("").astype(str).str.strip()
    if ligand_ids.eq("").any() or ligand_ids.nunique() != 1:
        raise ContractError("ranking CSV requires one nonblank ligand_id/query_id")
    if targets.eq("").any() or targets.duplicated().any():
        raise ContractError("ranking CSV requires unique nonblank target_id values")
    scores = pd.to_numeric(frame["performance_v2_ranking_score"], errors="coerce")
    if scores.isna().any() or not np.isfinite(scores.to_numpy(dtype=float)).all():
        raise ContractError("ranking scores must be finite numeric values")
    semantics = frame["performance_v2_score_semantics"].fillna("").astype(str)
    if not semantics.eq("ranking_score_not_probability").all():
        raise ContractError("ranking scores must be labelled ranking_score_not_probability")
    routes = frame["performance_v2_ood_route"].fillna("").astype(str)
    if not routes.isin(OOD_ROUTES).all():
        raise ContractError("ranking CSV contains invalid OOD route")
    abstained = frame["performance_v2_abstained"]
    normalized_abstained = abstained.map(
        lambda value: value if isinstance(value, bool) else str(value).strip().lower()
    )
    if not normalized_abstained.isin([True, False, "true", "false"]).all():
        raise ContractError("ranking CSV abstention values must be boolean")
    abstained_bool = normalized_abstained.map(lambda value: value is True or value == "true")
    query = payload.get("query")
    if not isinstance(query, Mapping):
        raise ContractError("ranking manifest query provenance is missing")
    if str(query.get("ligand_id") or "") != ligand_ids.iloc[0]:
        raise ContractError("ranking manifest query ligand_id does not match ranking CSV")
    if query.get("input_type") not in {"ligand_id", "smiles"}:
        raise ContractError("ranking manifest query input_type is invalid")
    scoring_policy = query.get("scoring_policy")
    if not isinstance(scoring_policy, Mapping) or "min_score" not in scoring_policy:
        raise ContractError("ranking manifest query scoring_policy is missing")
    raw_min_score = scoring_policy.get("min_score")
    try:
        min_score = None if raw_min_score is None else float(raw_min_score)
    except (TypeError, ValueError) as exc:
        raise ContractError("ranking manifest min_score must be numeric or null") from exc
    if min_score is not None and not math.isfinite(min_score):
        raise ContractError("ranking manifest min_score must be finite")
    train_ligand_domain = frozenset(map(str, model_manifest.get("train_ligands", [])))
    train_target_domain = frozenset(map(str, model_manifest.get("train_targets", [])))
    for row_index, (target_id, route, score) in enumerate(zip(targets, routes, scores, strict=True)):
        expected_route = ood_route(
            ligand_ids.iloc[0],
            target_id,
            train_ligand_domain,
            train_target_domain,
        )
        structure_override = route == "structure_abstained" and query.get("structure_abstained") is True
        if route != expected_route and not structure_override:
            raise ContractError("ranking CSV OOD route does not match model train domains")
        expected_abstention = should_abstain(route, score=float(score), min_score=min_score)
        if bool(abstained_bool.iloc[row_index]) != expected_abstention:
            raise ContractError("ranking CSV abstention does not match OOD/scoring policy")
    ranks = pd.to_numeric(frame["performance_v2_rank"], errors="coerce")
    expected_ranks = list(range(1, len(frame) + 1))
    if ranks.isna().any() or ranks.astype(int).tolist() != expected_ranks:
        raise ContractError("ranking CSV ranks must be contiguous 1..N")
    expected_order = (
        pd.DataFrame({
            "abstained": abstained_bool.to_numpy(dtype=bool),
            "score": scores.to_numpy(dtype=float),
            "target": targets.to_numpy(dtype=str),
            "position": np.arange(len(frame)),
        })
        .sort_values(["abstained", "score", "target"], ascending=[True, False, True])
        ["position"]
        .astype(int)
        .tolist()
    )
    if expected_order != list(range(len(frame))):
        raise ContractError("ranking CSV rows/ranks do not follow score ordering semantics")
    calibration = model_manifest.get("calibration") or {}
    for column, event in (
        ("performance_v2_direct_binding_probability", "direct_binding_reversible"),
        ("performance_v2_functional_modulation_probability", "functional_modulation"),
    ):
        values = pd.to_numeric(frame[column], errors="coerce")
        nonblank = ~frame[column].isna() & frame[column].astype(str).str.strip().ne("")
        if nonblank.any():
            if (nonblank & abstained_bool).any():
                raise ContractError(f"{column} must be blank for abstained rows")
            if values[nonblank].isna().any():
                raise ContractError(f"{column} must be numeric when present")
            if ((values[nonblank] < 0.0) | (values[nonblank] > 1.0)).any():
                raise ContractError(f"{column} values must be within [0, 1]")
            event_calibration = calibration.get(event) if isinstance(calibration, Mapping) else None
            if not isinstance(event_calibration, Mapping) or event_calibration.get("status") != "measured_event_calibrated":
                raise ContractError(f"{column} requires measured-event calibrator coverage")
    if query.get("input_type") == "smiles":
        canonical_hash = str(query.get("canonical_smiles_sha256") or "")
        if len(canonical_hash) != 64 or any(ch not in "0123456789abcdef" for ch in canonical_hash):
            raise ContractError("ranking manifest query canonical_smiles_sha256 is invalid")
        if query.get("raw_smiles_recorded") is not False:
            raise ContractError("ranking manifest must not record raw query SMILES")
    semantics_payload = payload.get("semantics")
    if not isinstance(semantics_payload, Mapping) or semantics_payload.get("ranking_score") != "not_probability":
        raise ContractError("ranking manifest must declare ranking_score not_probability semantics")
    return payload


def write_tsv_rows(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ContractError(f"no rows to write: {path}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)
