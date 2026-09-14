#!/usr/bin/env python3
"""Build performance-v2 frozen embeddings or deterministic fixtures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from performance_v2_model import (  # noqa: E402
    ContractError,
    ECFP_BITS,
    ECFP_RADIUS,
    EMBEDDING_SCHEMA,
    ESM2_CHECKPOINT,
    ESM2_LICENSE,
    ESM2_REVISION,
    MOLFORMER_CHECKPOINT,
    MOLFORMER_LICENSE,
    MOLFORMER_POOLING,
    MOLFORMER_PRECISION,
    MOLFORMER_REVISION,
    PROTEIN_POOLING,
    PROTEIN_WINDOW_OVERLAP,
    PROTEIN_WINDOW_STRATEGY,
    append_only_outputs,
    artifact_record,
    canonicalize_smiles,
    ecfp_bit_vector,
    fixture_embedding_frame,
    load_transformer_fp32_then_place,
    resolve_hf_commit_hash,
    sha256_file,
    utc_now,
    validate_embedding_manifest,
    validate_input_manifest,
    write_csv_atomic,
    write_json_atomic,
)

OUTPUT_SHARD_ROWS = 4096


def _embedding_inputs(args: argparse.Namespace) -> dict[str, object]:
    inputs: dict[str, object] = {
        "ligands": {
            "path": str(args.ligands.resolve()),
            "sha256": sha256_file(args.ligands),
        },
        "targets": {
            "path": str(args.targets.resolve()),
            "sha256": sha256_file(args.targets),
        },
    }
    input_manifest_record = getattr(args, "input_manifest_record", None)
    if input_manifest_record is not None:
        inputs["input_manifest"] = input_manifest_record
    return inputs


def _validate_input_lineage(args: argparse.Namespace) -> dict[str, object] | None:
    if args.input_manifest is None:
        if not args.fixture:
            raise SystemExit("production embedding build requires --input-manifest")
        return None
    try:
        payload = validate_input_manifest(args.input_manifest)
    except ContractError as exc:
        raise SystemExit(str(exc)) from exc
    for key, active_path in (("ligands", args.ligands), ("targets", args.targets)):
        if payload["artifacts"][key].get("sha256") != sha256_file(active_path):
            raise SystemExit(f"--{key} does not match --input-manifest artifact")
    return artifact_record(args.input_manifest)


def _read_ids(path: Path, column: str) -> list[str]:
    frame = pd.read_csv(path)
    if column not in frame.columns:
        raise SystemExit(f"{path} missing required column {column}")
    ids = frame[column].fillna("").astype(str).str.strip().tolist()
    ids = [value for value in ids if value]
    if not ids:
        raise SystemExit(f"{path} contains no {column} values")
    return sorted(set(ids))


def _read_ligands(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = {"ligand_id", "smiles"} - set(frame.columns)
    if missing:
        raise SystemExit(f"{path} missing required ligand columns: {sorted(missing)}")
    frame = frame[["ligand_id", "smiles"]].copy()
    frame["ligand_id"] = frame["ligand_id"].fillna("").astype(str).str.strip()
    frame["smiles"] = frame["smiles"].fillna("").astype(str).str.strip()
    if frame["ligand_id"].eq("").any() or frame["smiles"].eq("").any():
        raise SystemExit("ligand input requires nonblank ligand_id and smiles")
    if frame["ligand_id"].duplicated().any():
        raise SystemExit("ligand input requires unique ligand_id values")
    return frame.sort_values("ligand_id").reset_index(drop=True)


def _read_targets(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = {"target_id", "sequence"} - set(frame.columns)
    if missing:
        raise SystemExit(f"{path} missing required target columns: {sorted(missing)}")
    frame = frame[["target_id", "sequence"]].copy()
    frame["target_id"] = frame["target_id"].fillna("").astype(str).str.strip()
    frame["sequence"] = frame["sequence"].fillna("").astype(str).str.strip()
    if frame["target_id"].eq("").any() or frame["sequence"].eq("").any():
        raise SystemExit("target input requires nonblank target_id and sequence")
    if frame["target_id"].duplicated().any():
        raise SystemExit("target input requires unique target_id values")
    return frame.sort_values("target_id").reset_index(drop=True)


def _mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom


def _pooled_output(outputs, attention_mask):
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output
    if hasattr(outputs, "last_hidden_state"):
        return _mean_pool(outputs.last_hidden_state, attention_mask)
    if isinstance(outputs, tuple) and len(outputs) > 0:
        return _mean_pool(outputs[0], attention_mask)
    raise SystemExit("checkpoint output did not expose pooled or last-hidden embeddings")


def _residue_pooled_output(outputs, attention_mask, special_tokens_mask):
    if hasattr(outputs, "last_hidden_state"):
        hidden = outputs.last_hidden_state
    elif isinstance(outputs, tuple) and len(outputs) > 0:
        hidden = outputs[0]
    else:
        raise SystemExit("ESM2 checkpoint output did not expose last_hidden_state")
    residue_mask = attention_mask.bool() & ~special_tokens_mask.bool()
    counts = residue_mask.sum(dim=1)
    if bool((counts < 1).any()):
        raise SystemExit("ESM2 window contains no residue tokens")
    mask = residue_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def _embedding_device(torch_module, *, cpu: bool):
    if cpu:
        return torch_module.device("cpu")
    if not torch_module.cuda.is_available():
        raise SystemExit("CUDA checkpoint embedding is required unless --cpu is supplied")
    return torch_module.device("cuda")


def _protein_windows(
    sequence: str, *, residue_window: int, overlap: int
) -> list[tuple[int, int, str]]:
    if residue_window < 1 or overlap < 0 or overlap >= residue_window:
        raise SystemExit("protein residue-window/overlap policy is invalid")
    if not sequence:
        raise SystemExit("protein sequence must be nonblank")
    if len(sequence) <= residue_window:
        return [(0, len(sequence), sequence)]
    step = residue_window - overlap
    starts = []
    start = 0
    while True:
        starts.append(start)
        if start + residue_window >= len(sequence):
            break
        start += step
    return [
        (start, min(start + residue_window, len(sequence)), sequence[start : start + residue_window])
        for start in starts
    ]


def _tokenize_protein_windows(tokenizer, windows: list[str], *, max_length: int):
    encoded = tokenizer(
        windows,
        padding=True,
        truncation=False,
        return_special_tokens_mask=True,
        return_tensors="pt",
    )
    if "special_tokens_mask" not in encoded:
        raise SystemExit("ESM2 tokenizer did not return special_tokens_mask")
    if int(encoded["input_ids"].shape[1]) > max_length:
        raise SystemExit("ESM2 residue window exceeded the configured token limit")
    return encoded


def _iter_embed_texts(
    *,
    texts: list[str],
    model_id: str,
    revision: str,
    batch_size: int,
    max_length: int,
    trust_remote_code: bool,
    cpu: bool,
):
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = _embedding_device(torch, cpu=cpu)
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
    cursor = 0
    active_batch = int(batch_size)
    try:
        with torch.no_grad():
            while cursor < len(texts):
                chunk = texts[cursor : cursor + active_batch]
                try:
                    encoded = tokenizer(
                        chunk,
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt",
                    )
                    encoded = {key: value.to(device) for key, value in encoded.items()}
                    # MoLFormer compat-v4's random feature map overflows under
                    # CUDA fp16 autocast. Keep both weights and forward in fp32.
                    pooled = _pooled_output(model(**encoded), encoded["attention_mask"])
                    if not bool(torch.isfinite(pooled).all()):
                        raise SystemExit("MoLFormer produced nonfinite embeddings")
                    yield (
                        pooled.detach().float().cpu().numpy().astype(np.float32),
                        cursor,
                        len(chunk),
                        resolved_revision,
                    )
                    cursor += len(chunk)
                    if active_batch < batch_size:
                        active_batch = min(batch_size, active_batch * 2)
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower() or active_batch <= 1:
                        raise
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    active_batch = max(1, active_batch // 2)
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _iter_embed_proteins(
    *,
    sequences: list[str],
    model_id: str,
    revision: str,
    batch_size: int,
    max_length: int,
    overlap: int,
    cpu: bool,
):
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = _embedding_device(torch, cpu=cpu)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    special_tokens = int(tokenizer.num_special_tokens_to_add(pair=False))
    residue_window = max_length - special_tokens
    if residue_window < 1 or overlap < 0 or overlap >= residue_window:
        raise SystemExit("--protein-max-length is too small for the window/overlap policy")
    model = load_transformer_fp32_then_place(
        AutoModel,
        model_id=model_id,
        revision=revision,
        trust_remote_code=False,
        device=device,
        model_kwargs={"add_pooling_layer": False},
    )
    model.eval()
    resolved_revision = resolve_hf_commit_hash(
        model=model,
        tokenizer=tokenizer,
        model_id=model_id,
        revision=revision,
    )
    cursor = 0
    active_window_batch = int(batch_size)
    try:
        with torch.no_grad():
            while cursor < len(sequences):
                chunk = sequences[cursor : cursor + batch_size]
                records: list[tuple[int, int, int, str]] = []
                long_count = 0
                for owner, sequence in enumerate(chunk):
                    windows = _protein_windows(
                        sequence, residue_window=residue_window, overlap=overlap
                    )
                    long_count += int(len(windows) > 1)
                    records.extend(
                        (owner, start, stop, window) for start, stop, window in windows
                    )
                weighted_sums: np.ndarray | None = None
                total_weights = np.zeros(len(chunk), dtype=np.float64)
                window_cursor = 0
                while window_cursor < len(records):
                    current = records[window_cursor : window_cursor + active_window_batch]
                    try:
                        encoded = _tokenize_protein_windows(
                            tokenizer,
                            [record[3] for record in current],
                            max_length=max_length,
                        )
                        special_mask = encoded.pop("special_tokens_mask")
                        encoded = {key: value.to(device) for key, value in encoded.items()}
                        special_mask = special_mask.to(device)
                        with torch.autocast(
                            device_type=device.type,
                            dtype=torch.float16,
                            enabled=device.type == "cuda",
                        ):
                            pooled = _residue_pooled_output(
                                model(**encoded), encoded["attention_mask"], special_mask
                            )
                        matrix = pooled.detach().float().cpu().numpy().astype(np.float32)
                    except RuntimeError as exc:
                        if (
                            device.type != "cuda"
                            or "out of memory" not in str(exc).lower()
                            or active_window_batch <= 1
                        ):
                            raise
                        torch.cuda.empty_cache()
                        active_window_batch = max(1, active_window_batch // 2)
                        continue
                    if weighted_sums is None:
                        weighted_sums = np.zeros((len(chunk), matrix.shape[1]), dtype=np.float64)
                    for vector, (owner, start, stop, _) in zip(matrix, current, strict=True):
                        weight = stop - start
                        weighted_sums[owner] += vector.astype(np.float64) * weight
                        total_weights[owner] += weight
                    window_cursor += len(current)
                    if active_window_batch < batch_size:
                        active_window_batch = min(batch_size, active_window_batch * 2)
                if weighted_sums is None or (total_weights <= 0).any():
                    raise SystemExit("ESM2 window pooling produced no embeddings")
                yield (
                    (weighted_sums / total_weights[:, None]).astype(np.float32),
                    cursor,
                    len(chunk),
                    resolved_revision,
                    {
                        "special_tokens_per_window": special_tokens,
                        "residue_window": residue_window,
                        "long_sequence_count": long_count,
                        "total_window_count": len(records),
                    },
                )
                cursor += len(chunk)
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _ecfp_bits(smiles_values: list[str], *, n_bits: int, radius: int) -> np.ndarray:
    return np.vstack(
        [ecfp_bit_vector(smiles, n_bits=n_bits, radius=radius) for smiles in smiles_values]
    ).astype(np.float32, copy=False)


def _embedding_frame(ids: list[str], matrix: np.ndarray, id_column: str, *, prefix: str = "e") -> pd.DataFrame:
    frame = pd.DataFrame({id_column: ids})
    for idx in range(matrix.shape[1]):
        frame[f"{prefix}{idx}"] = matrix[:, idx].astype(float)
    return frame


def _write_npz_shard(path: Path, ids: list[str], vectors: np.ndarray) -> dict[str, object]:
    if not np.isfinite(vectors).all():
        raise SystemExit(f"refusing to write nonfinite embedding shard: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp.npz")
    if tmp.exists():
        tmp.unlink()
    np.savez_compressed(
        tmp,
        ids=np.array(ids, dtype="U"),
        vectors=vectors.astype(np.float32, copy=False),
    )
    tmp.replace(path)
    return artifact_record(path, rows=len(ids))


class _NpzShardWriter:
    def __init__(self, root: Path, prefix: str, *, max_rows: int = OUTPUT_SHARD_ROWS) -> None:
        self.root = root
        self.prefix = prefix
        self.max_rows = int(max_rows)
        self.shards: list[dict[str, object]] = []
        self.rows = 0
        self.dim: int | None = None
        self._ids: list[str] = []
        self._vectors: list[np.ndarray] = []
        self._buffered_rows = 0

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or len(ids) != len(matrix):
            raise SystemExit("embedding shard writer received invalid ids/vectors")
        if not np.isfinite(matrix).all():
            raise SystemExit("embedding shard writer received nonfinite vectors")
        if self.dim is None:
            self.dim = int(matrix.shape[1])
        elif matrix.shape[1] != self.dim:
            raise SystemExit("embedding dimension changed between inference batches")
        cursor = 0
        while cursor < len(ids):
            take = min(self.max_rows - self._buffered_rows, len(ids) - cursor)
            self._ids.extend(ids[cursor : cursor + take])
            self._vectors.append(matrix[cursor : cursor + take].copy())
            self._buffered_rows += take
            self.rows += take
            cursor += take
            if self._buffered_rows == self.max_rows:
                self._flush()

    def finish(self) -> None:
        if self._buffered_rows:
            self._flush()

    def _flush(self) -> None:
        matrix = np.vstack(self._vectors).astype(np.float32, copy=False)
        path = self.root / f"{self.prefix}_{len(self.shards):06d}.npz"
        self.shards.append(_write_npz_shard(path, self._ids, matrix))
        self._ids = []
        self._vectors = []
        self._buffered_rows = 0


def _npz_sharded_record(
    root: Path,
    *,
    shards: list[dict[str, object]],
    rows: int,
    dim: int,
    molformer_dim: int | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "format": "npz_shards",
        "path": str(root.resolve()),
        "dtype": "float32",
        "dim": int(dim),
        "rows": int(rows),
        "shards": shards,
    }
    if molformer_dim is not None:
        record.update({
            "molformer_dim": int(molformer_dim),
            "ecfp_offset": int(molformer_dim),
            "ecfp_bits": ECFP_BITS,
            "ecfp_radius": ECFP_RADIUS,
            "ecfp_encoding": "independent_binary_float32",
        })
    return record


def build_fixture(args: argparse.Namespace) -> dict[str, object]:
    ligand_ids = _read_ids(args.ligands, "ligand_id")
    target_ids = _read_ids(args.targets, "target_id")
    ligand_frame = fixture_embedding_frame(ligand_ids, id_column="ligand_id", dim=args.fixture_dim)
    target_frame = fixture_embedding_frame(target_ids, id_column="target_id", dim=args.fixture_dim)
    write_csv_atomic(ligand_frame, args.out_ligands)
    write_csv_atomic(target_frame, args.out_targets)
    manifest = {
        "schema_version": EMBEDDING_SCHEMA,
        "created_at": utc_now(),
        "mode": "fixture_precomputed" if args.fixture else "frozen_checkpoint",
        "checkpoints": {
            "ligand": {
                "model_id": MOLFORMER_CHECKPOINT,
                "revision": MOLFORMER_REVISION,
                "default": True,
                "license_id": MOLFORMER_LICENSE,
            },
            "protein": {
                "model_id": ESM2_CHECKPOINT,
                "revision": ESM2_REVISION,
                "default": True,
                "license_id": ESM2_LICENSE,
            },
        },
        "artifacts": {
            "ligands": artifact_record(args.out_ligands, rows=len(ligand_frame)),
            "targets": artifact_record(args.out_targets, rows=len(target_frame)),
        },
        "inputs": _embedding_inputs(args),
        "provenance": {
            "deterministic_canonical_ids": True,
            "batching": {
                "requested_ligand_batch_size": int(args.batch_size),
                "requested_protein_batch_size": int(args.protein_batch_size),
                "oom_backoff": "halve-batch-until-singleton",
            },
            "execution": "no_grad_fp16_cuda_when_available",
        },
        "freeze_contract": {
            "encoders_frozen": True,
            "requires_grad": False,
            "no_grad": True,
            "fixture": True,
        },
        "license_profile": {
            "redistribution": "derived embeddings only; checkpoint weights not redistributed",
            "fallback_model_ids": [MOLFORMER_CHECKPOINT, ESM2_CHECKPOINT],
        },
    }
    write_json_atomic(manifest, args.out_manifest)
    return validate_embedding_manifest(args.out_manifest)


def build_checkpoint_embeddings(args: argparse.Namespace) -> dict[str, object]:
    ligands = _read_ligands(args.ligands)
    targets = _read_targets(args.targets)
    ligands["canonical_smiles"] = [canonicalize_smiles(value) for value in ligands["smiles"]]
    ligand_model_id = MOLFORMER_CHECKPOINT
    target_model_id = f"facebook/{ESM2_CHECKPOINT}"
    args.out_ligands.mkdir(parents=True)
    args.out_targets.mkdir(parents=True)
    ligand_writer = _NpzShardWriter(args.out_ligands, "ligands")
    ligand_resolved_revision: str | None = None
    molformer_dim: int | None = None
    for embeddings, cursor, size, resolved_revision in _iter_embed_texts(
        texts=ligands["canonical_smiles"].tolist(),
        model_id=ligand_model_id,
        revision=MOLFORMER_REVISION,
        batch_size=args.batch_size,
        max_length=args.ligand_max_length,
        trust_remote_code=True,
        cpu=args.cpu,
    ):
        if ligand_resolved_revision not in {None, resolved_revision}:
            raise SystemExit("MoLFormer resolved revision changed during embedding")
        ligand_resolved_revision = resolved_revision
        if molformer_dim not in {None, int(embeddings.shape[1])}:
            raise SystemExit("MoLFormer embedding dimension changed between batches")
        molformer_dim = int(embeddings.shape[1])
        chunk = ligands.iloc[cursor : cursor + size]
        ecfp = _ecfp_bits(
            chunk["canonical_smiles"].tolist(),
            n_bits=args.ecfp_bits,
            radius=args.ecfp_radius,
        )
        vectors = np.hstack([embeddings, ecfp]).astype(np.float32)
        ligand_writer.add(chunk["ligand_id"].tolist(), vectors)
    ligand_writer.finish()

    target_writer = _NpzShardWriter(args.out_targets, "targets")
    protein_resolved_revision: str | None = None
    protein_windowing: dict[str, int] | None = None
    long_sequence_count = 0
    total_window_count = 0
    for embeddings, cursor, size, resolved_revision, windowing in _iter_embed_proteins(
        sequences=targets["sequence"].tolist(),
        model_id=target_model_id,
        revision=ESM2_REVISION,
        batch_size=args.protein_batch_size,
        max_length=args.protein_max_length,
        overlap=args.protein_window_overlap,
        cpu=args.cpu,
    ):
        if protein_resolved_revision not in {None, resolved_revision}:
            raise SystemExit("ESM2 resolved revision changed during embedding")
        protein_resolved_revision = resolved_revision
        if protein_windowing is None:
            protein_windowing = {
                "special_tokens_per_window": int(windowing["special_tokens_per_window"]),
                "residue_window": int(windowing["residue_window"]),
            }
        elif protein_windowing != {
            "special_tokens_per_window": int(windowing["special_tokens_per_window"]),
            "residue_window": int(windowing["residue_window"]),
        }:
            raise SystemExit("ESM2 tokenizer window dimensions changed during embedding")
        long_sequence_count += int(windowing["long_sequence_count"])
        total_window_count += int(windowing["total_window_count"])
        chunk = targets.iloc[cursor : cursor + size]
        target_writer.add(chunk["target_id"].tolist(), embeddings)
    target_writer.finish()
    if (
        ligand_writer.dim is None
        or target_writer.dim is None
        or molformer_dim is None
        or ligand_resolved_revision is None
        or protein_resolved_revision is None
        or protein_windowing is None
    ):
        raise SystemExit("embedding build produced no shards")
    manifest = {
        "schema_version": EMBEDDING_SCHEMA,
        "created_at": utc_now(),
        "mode": "frozen_checkpoint",
        "checkpoints": {
            "ligand": {
                "model_id": MOLFORMER_CHECKPOINT,
                "revision": MOLFORMER_REVISION,
                "resolved_revision": ligand_resolved_revision,
                "default": True,
                "license_id": MOLFORMER_LICENSE,
            },
            "protein": {
                "model_id": ESM2_CHECKPOINT,
                "hf_model_id": target_model_id,
                "revision": ESM2_REVISION,
                "resolved_revision": protein_resolved_revision,
                "default": True,
                "license_id": ESM2_LICENSE,
            },
        },
        "artifacts": {
            "ligands": _npz_sharded_record(
                args.out_ligands,
                shards=ligand_writer.shards,
                rows=ligand_writer.rows,
                dim=ligand_writer.dim,
                molformer_dim=molformer_dim,
            ),
            "targets": _npz_sharded_record(
                args.out_targets,
                shards=target_writer.shards,
                rows=target_writer.rows,
                dim=target_writer.dim,
            ),
        },
        "inputs": _embedding_inputs(args),
        "provenance": {
            "deterministic_canonical_ids": True,
            "batching": {
                "requested_ligand_batch_size": int(args.batch_size),
                "requested_protein_batch_size": int(args.protein_batch_size),
                "oom_backoff": "halve-batch-until-singleton",
                "artifact_shard_rows": OUTPUT_SHARD_ROWS,
            },
            "execution": (
                "torch.no_grad; MoLFormer fp32 weights/forward; ESM2 fp32 weights "
                "with forward autocast(cuda,float16)"
                if not args.cpu
                else "torch.no_grad; fp32 model weights/forward on CPU; autocast disabled"
            ),
            "precision": {
                "ligand": MOLFORMER_PRECISION,
                "protein": (
                    "fp32_model_weights_with_forward_autocast_fp16"
                    if not args.cpu
                    else "fp32_model_weights_and_forward"
                ),
            },
            "ligand_tokenization": {
                "strategy": "tokenizer_truncation_fixed_max_tokens",
                "max_tokens": int(args.ligand_max_length),
                "truncation": True,
            },
            "streaming": "chunked npz shard writes; full corpus is not held in RAM",
            "ligand_features": ["molformer_pooled", "2048_independent_binary_morgan_bits"],
            "target_features": ["esm2_150m_pooled"],
            "pooling": {
                "ligand": MOLFORMER_POOLING,
                "protein": PROTEIN_POOLING,
                "protein_add_pooling_layer": False,
            },
            "protein_windowing": {
                "strategy": PROTEIN_WINDOW_STRATEGY,
                "max_tokens": int(args.protein_max_length),
                "special_tokens_per_window": protein_windowing["special_tokens_per_window"],
                "residue_window": protein_windowing["residue_window"],
                "overlap_residues": int(args.protein_window_overlap),
                "sequence_count": int(len(targets)),
                "long_sequence_count": int(long_sequence_count),
                "total_window_count": int(total_window_count),
                "full_sequence_coverage": True,
                "silent_truncation": False,
                "window_pooling": "residue_count_weighted_mean",
            },
        },
        "freeze_contract": {
            "encoders_frozen": True,
            "requires_grad": False,
            "no_grad": True,
            "fixture": False,
        },
        "license_profile": {
            "redistribution": "derived embeddings only; checkpoint weights not redistributed",
            "fallback_model_ids": [MOLFORMER_CHECKPOINT, ESM2_CHECKPOINT],
        },
    }
    write_json_atomic(manifest, args.out_manifest)
    return validate_embedding_manifest(args.out_manifest)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ligands", required=True, type=Path)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--out-ligands", required=True, type=Path)
    parser.add_argument("--out-targets", required=True, type=Path)
    parser.add_argument("--out-manifest", required=True, type=Path)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--fixture-dim", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--protein-batch-size", type=int, default=8)
    parser.add_argument("--ligand-max-length", type=int, default=256)
    parser.add_argument("--protein-max-length", type=int, default=1024)
    parser.add_argument("--protein-window-overlap", type=int, default=PROTEIN_WINDOW_OVERLAP)
    parser.add_argument("--ecfp-bits", type=int, default=2048)
    parser.add_argument("--ecfp-radius", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.fixture_dim < 2:
        raise SystemExit("--fixture-dim must be >= 2")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.protein_batch_size < 1:
        raise SystemExit("--protein-batch-size must be >= 1")
    if args.ligand_max_length < 1:
        raise SystemExit("--ligand-max-length must be >= 1")
    if args.protein_max_length < 3:
        raise SystemExit("--protein-max-length must be >= 3")
    if args.protein_window_overlap < 0 or args.protein_window_overlap >= args.protein_max_length - 1:
        raise SystemExit("--protein-window-overlap must fit within the residue window")
    if args.ecfp_bits != ECFP_BITS:
        raise SystemExit(f"--ecfp-bits must be exactly {ECFP_BITS}")
    if args.ecfp_radius != ECFP_RADIUS:
        raise SystemExit(f"--ecfp-radius must be exactly {ECFP_RADIUS}")
    with append_only_outputs((args.out_ligands, args.out_targets, args.out_manifest)):
        args.input_manifest_record = _validate_input_lineage(args)
        if args.fixture:
            build_fixture(args)
        else:
            build_checkpoint_embeddings(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
