#!/usr/bin/env python3
"""Build provenance-bound ESM-2 embeddings for screenable targets."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol

import numpy as np
import pandas as pd


SCREENABLE_SCHEMA_VERSION = "skinscout.screenable-target-cluster-map.v2"
SCHEMA_VERSION = "skinscout.target-sequence-embeddings.v1"
DEFAULT_MODEL_NAME = "esm2_t6_8M_UR50D"
DEFAULT_MAX_CHUNK_RESIDUES = 1022
DEFAULT_TOKENS_PER_BATCH = 8192
SEQUENCE_RE = re.compile(r"^[A-Z*.-]+$")


class SequenceEmbedder(Protocol):
    model_name: str
    dimension: int

    def embed_chunks(self, chunks: list[tuple[str, str]]) -> dict[str, np.ndarray]:
        """Return one float vector for each chunk id."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_file(path: Path, label: str) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"{label} is required and must be non-empty: {path}")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _require_file(path, label)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Unable to parse {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} root must be an object: {path}")
    return payload


def _resolve_manifest_path(raw_path: object, *, manifest_path: Path, label: str) -> Path:
    text = str(raw_path or "").strip()
    if not text:
        raise SystemExit(f"screenable manifest missing {label} path")
    path = Path(text)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _artifact_record(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(path.resolve()), "sha256": _sha256(path)}
    if rows is not None:
        record["rows"] = int(rows)
    return record


def _assert_safe_output_files(
    *, input_paths: Iterable[Path], output_paths: Iterable[Path]
) -> None:
    protected = {path.resolve() for path in input_paths}
    outputs = tuple(path.resolve() for path in output_paths)
    destructive = outputs + tuple(
        path.with_suffix(path.suffix + ".tmp").resolve() for path in outputs
    )
    if len(set(destructive)) != len(destructive):
        raise SystemExit("target embedding output paths must be distinct")
    aliases = sorted(str(path) for path in set(destructive) & protected)
    if aliases:
        raise SystemExit(
            "target embedding output path aliases a protected input: "
            + ", ".join(aliases)
        )


def _declared_sequence_inputs(
    target_csv: Path, target_manifest: Path
) -> list[Path]:
    manifest = _read_json(target_manifest, "screenable target manifest")
    inputs = manifest.get("inputs")
    base_record = inputs.get("base_target_csv") if isinstance(inputs, dict) else None
    supplemental_record = (
        inputs.get("supplemental_fasta") if isinstance(inputs, dict) else None
    )
    if not isinstance(base_record, dict) or not isinstance(supplemental_record, dict):
        raise SystemExit("screenable target manifest missing FASTA provenance")
    return [
        target_csv,
        target_manifest,
        _resolve_manifest_path(
            base_record.get("fasta_path"),
            manifest_path=target_manifest,
            label="inputs.base_target_csv.fasta_path",
        ),
        _resolve_manifest_path(
            supplemental_record.get("path"),
            manifest_path=target_manifest,
            label="inputs.supplemental_fasta.path",
        ),
    ]


def _accession_from_header(header: str, path: Path) -> str:
    token = header.strip().split()[0] if header.strip() else ""
    if "|" in token:
        parts = token.split("|")
        if len(parts) >= 2 and parts[1].strip():
            token = parts[1]
    accession = token.strip()
    if not accession:
        raise SystemExit(f"FASTA header lacks accession: {path}")
    return accession


def parse_fasta(path: Path) -> dict[str, str]:
    """Parse a FASTA file into accession -> sequence with duplicate checks."""
    _require_file(path, "FASTA")
    records: dict[str, str] = {}
    current: str | None = None
    parts: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise SystemExit(f"Unable to decode FASTA as UTF-8: {path}") from exc
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current is not None:
                _store_fasta_record(records, current, parts, path)
            current = _accession_from_header(line[1:], path)
            if current in records:
                raise SystemExit(f"Duplicate FASTA accession {current!r}: {path}")
            parts = []
            continue
        if current is None:
            raise SystemExit(
                f"FASTA sequence appears before first header at line {line_number}: {path}"
            )
        sequence_line = "".join(line.split()).upper()
        if not SEQUENCE_RE.fullmatch(sequence_line):
            raise SystemExit(
                f"FASTA sequence contains unsupported characters at line {line_number}: {path}"
            )
        parts.append(sequence_line)
    if current is None:
        raise SystemExit(f"FASTA contains no records: {path}")
    _store_fasta_record(records, current, parts, path)
    return records


def _store_fasta_record(
    records: dict[str, str], accession: str, parts: list[str], path: Path
) -> None:
    sequence = "".join(parts).replace("*", "").replace(".", "").replace("-", "")
    if not sequence:
        raise SystemExit(f"FASTA accession {accession!r} has an empty sequence: {path}")
    records[accession] = sequence


def load_screenable_targets(
    target_csv: Path, target_manifest: Path
) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    manifest = _read_json(target_manifest, "screenable target manifest")
    if manifest.get("schema_version") != SCREENABLE_SCHEMA_VERSION:
        raise SystemExit(
            "screenable target manifest schema_version must be "
            f"{SCREENABLE_SCHEMA_VERSION}: {target_manifest}"
        )
    policy = manifest.get("universe_policy")
    if not isinstance(policy, dict):
        raise SystemExit("screenable target manifest missing universe_policy")
    if policy.get("evaluation_panel_used") is not False:
        raise SystemExit("screenable target manifest must set evaluation_panel_used=false")
    if policy.get("known_target_assistance") is not False:
        raise SystemExit("screenable target manifest must set known_target_assistance=false")
    production = manifest.get("production_contract")
    if not isinstance(production, dict) or production.get("passes") is not True:
        raise SystemExit("screenable target manifest production_contract.passes must be true")

    _require_file(target_csv, "screenable target CSV")
    frame = pd.read_csv(target_csv)
    if "uniprot" not in frame.columns:
        raise SystemExit(f"screenable target CSV missing required column 'uniprot': {target_csv}")
    accessions = frame["uniprot"].fillna("").astype(str).str.strip()
    if accessions.eq("").any():
        raise SystemExit(f"screenable target CSV contains blank uniprot: {target_csv}")
    if accessions.duplicated().any():
        duplicates = sorted(accessions[accessions.duplicated()].unique())
        raise SystemExit(
            f"screenable target CSV contains duplicate UniProt accessions: {duplicates[:10]}"
        )
    target_ids = sorted(accessions.tolist())

    artifact = manifest.get("artifact")
    if not isinstance(artifact, dict):
        raise SystemExit("screenable target manifest missing artifact")
    if artifact.get("sha256") != _sha256(target_csv) or int(
        artifact.get("rows", -1)
    ) != len(target_ids):
        raise SystemExit("screenable target CSV sha256/rows do not match its manifest")

    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise SystemExit("screenable target manifest missing inputs")
    base_record = inputs.get("base_target_csv")
    supplemental_record = inputs.get("supplemental_fasta")
    if not isinstance(base_record, dict) or not isinstance(supplemental_record, dict):
        raise SystemExit("screenable target manifest missing FASTA provenance")
    source_manifest = base_record.get("source_manifest")
    if (
        not isinstance(source_manifest, dict)
        or source_manifest.get("sequence_role")
        != "canonical_uniprot_reference_sequence"
    ):
        raise SystemExit(
            "screenable target manifest base FASTA must be canonical UniProt sequence data"
        )

    base_fasta = _resolve_manifest_path(
        base_record.get("fasta_path"),
        manifest_path=target_manifest,
        label="inputs.base_target_csv.fasta_path",
    )
    supplemental_fasta = _resolve_manifest_path(
        supplemental_record.get("path"),
        manifest_path=target_manifest,
        label="inputs.supplemental_fasta.path",
    )
    _require_file(base_fasta, "base target FASTA")
    _require_file(supplemental_fasta, "supplemental FASTA")
    if base_record.get("fasta_sha256") != _sha256(base_fasta):
        raise SystemExit("base_target_csv.fasta_path sha256 is stale")
    if supplemental_record.get("sha256") != _sha256(supplemental_fasta):
        raise SystemExit("supplemental_fasta.path sha256 is stale")

    sequences = parse_fasta(base_fasta)
    supplemental = parse_fasta(supplemental_fasta)
    overlap = sorted(set(sequences) & set(supplemental))
    if overlap:
        raise SystemExit(f"base and supplemental FASTA accessions overlap: {overlap[:10]}")
    sequences.update(supplemental)

    target_set = set(target_ids)
    fasta_set = set(sequences)
    if fasta_set != target_set:
        missing = sorted(target_set - fasta_set)
        extra = sorted(fasta_set - target_set)
        raise SystemExit(
            "screenable target FASTA coverage must exactly match target CSV accessions; "
            f"missing={missing[:10]} extra={extra[:10]}"
        )
    provenance = {
        "target_csv": _artifact_record(target_csv, rows=len(target_ids)),
        "target_manifest": _artifact_record(target_manifest),
        "base_target_fasta": _artifact_record(
            base_fasta, rows=len(sequences) - len(supplemental)
        ),
        "supplemental_fasta": _artifact_record(supplemental_fasta, rows=len(supplemental)),
    }
    return target_ids, sequences, provenance


def chunk_sequence(sequence: str, max_chunk_residues: int) -> list[str]:
    if max_chunk_residues < 1 or max_chunk_residues > DEFAULT_MAX_CHUNK_RESIDUES:
        raise SystemExit("--max-chunk-residues must be between 1 and 1022")
    return [
        sequence[start : start + max_chunk_residues]
        for start in range(0, len(sequence), max_chunk_residues)
    ]


def _validate_embedding(vector: np.ndarray, *, accession: str, dimension: int) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32)
    if arr.shape != (dimension,):
        raise SystemExit(
            f"embedding for {accession} has dimension {arr.shape}; expected ({dimension},)"
        )
    if not np.isfinite(arr).all():
        raise SystemExit(f"embedding for {accession} contains non-finite values")
    norm = float(np.linalg.norm(arr.astype(np.float64)))
    if not math.isfinite(norm) or norm <= 0.0:
        raise SystemExit(f"embedding for {accession} has zero or non-finite norm")
    return arr


def embed_sequences(
    accessions: Iterable[str],
    sequences: dict[str, str],
    embedder: SequenceEmbedder,
    *,
    max_chunk_residues: int = DEFAULT_MAX_CHUNK_RESIDUES,
    tokens_per_batch: int = DEFAULT_TOKENS_PER_BATCH,
) -> list[tuple[str, np.ndarray]]:
    if tokens_per_batch < 1:
        raise SystemExit("--tokens-per-batch must be positive")
    pending: list[tuple[str, str, str]] = []
    weights: dict[str, list[tuple[str, int]]] = {}
    for accession in accessions:
        chunks = chunk_sequence(sequences[accession], max_chunk_residues)
        weights[accession] = []
        for index, chunk in enumerate(chunks):
            chunk_id = f"{accession}:{index}"
            pending.append((accession, chunk_id, chunk))
            weights[accession].append((chunk_id, len(chunk)))

    # ESM pads every sequence in a batch to the longest member. Sorting by
    # length keeps that padded footprint close to the configured token budget.
    pending.sort(key=lambda item: (len(item[2]), item[1]))
    chunk_vectors: dict[str, np.ndarray] = {}
    batch: list[tuple[str, str]] = []
    for _, chunk_id, chunk in pending:
        chunk_tokens = len(chunk) + 2  # BOS and EOS added by the batch converter.
        if chunk_tokens > tokens_per_batch:
            raise SystemExit(
                f"--tokens-per-batch={tokens_per_batch} cannot fit chunk {chunk_id} "
                f"with {chunk_tokens} model tokens"
            )
        padded_tokens = (len(batch) + 1) * chunk_tokens
        if batch and padded_tokens > tokens_per_batch:
            chunk_vectors.update(embedder.embed_chunks(batch))
            batch = []
        batch.append((chunk_id, chunk))
    if batch:
        chunk_vectors.update(embedder.embed_chunks(batch))

    output: list[tuple[str, np.ndarray]] = []
    dimension = int(embedder.dimension)
    for accession in accessions:
        total = sum(length for _, length in weights[accession])
        pooled = np.zeros(dimension, dtype=np.float64)
        for chunk_id, length in weights[accession]:
            if chunk_id not in chunk_vectors:
                raise SystemExit(f"embedder did not return chunk embedding: {chunk_id}")
            vector = _validate_embedding(
                chunk_vectors[chunk_id], accession=accession, dimension=dimension
            )
            pooled += vector.astype(np.float64) * (length / total)
        output.append(
            (
                accession,
                _validate_embedding(
                    pooled.astype(np.float32), accession=accession, dimension=dimension
                ),
            )
        )
    return output


def _write_embeddings_parquet_atomic(rows: list[tuple[str, np.ndarray]], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not rows:
        raise SystemExit("no embeddings to write")
    dim = int(rows[0][1].shape[0])
    for accession, vector in rows:
        _validate_embedding(vector, accession=accession, dimension=dim)
    flat = np.concatenate([vector.astype(np.float32, copy=False) for _, vector in rows])
    table = pa.table(
        {
            "uniprot": pa.array([accession for accession, _ in rows], type=pa.string()),
            "embedding": pa.FixedSizeListArray.from_arrays(
                pa.array(flat, type=pa.float32()), dim
            ),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    pq.write_table(table, tmp)
    tmp.replace(path)


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _remove_outputs(*paths: Path) -> None:
    for path in paths:
        path.unlink(missing_ok=True)
        path.with_suffix(path.suffix + ".tmp").unlink(missing_ok=True)


class FairEsmEmbedder:
    def __init__(self, *, model_name: str, device: str) -> None:
        if model_name != DEFAULT_MODEL_NAME:
            raise SystemExit(
                f"only the preregistered embedding model {DEFAULT_MODEL_NAME!r} is supported"
            )
        self.model_name = model_name
        try:
            import esm
            import torch
        except ImportError as exc:
            raise SystemExit(
                "fair-esm and torch are required for real ESM-2 embedding builds"
            ) from exc
        try:
            loader = getattr(esm.pretrained, model_name)
        except AttributeError as exc:
            raise SystemExit(f"fair-esm does not provide model loader {model_name!r}") from exc
        self._torch = torch
        self._esm = esm
        self.model, alphabet = loader()
        self.model.eval()
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(device)
        self.batch_converter = alphabet.get_batch_converter()
        self.dimension = int(getattr(self.model, "embed_dim", 0) or self.model.args.embed_dim)
        self.representation_layer = int(
            getattr(self.model, "num_layers", 0) or self.model.args.layers
        )
        self.checkpoint_sha256 = self._checkpoint_sha256()

    def _checkpoint_sha256(self) -> str:
        hub_dir = Path(self._torch.hub.get_dir()) / "checkpoints"
        checkpoint = hub_dir / f"{self.model_name}.pt"
        if checkpoint.is_file() and checkpoint.stat().st_size > 0:
            return _sha256(checkpoint)
        raise SystemExit(f"unable to locate downloaded fair-esm checkpoint: {checkpoint}")

    def embed_chunks(self, chunks: list[tuple[str, str]]) -> dict[str, np.ndarray]:
        _, _, tokens = self.batch_converter(chunks)
        tokens = tokens.to(self.device)
        with self._torch.no_grad():
            result = self.model(
                tokens, repr_layers=[self.representation_layer], return_contacts=False
            )
        representations = result["representations"][self.representation_layer]
        output: dict[str, np.ndarray] = {}
        for row_index, (chunk_id, sequence) in enumerate(chunks):
            residue_repr = representations[row_index, 1 : len(sequence) + 1]
            vector = residue_repr.mean(0).detach().cpu().numpy().astype(np.float32)
            output[chunk_id] = vector
        return output


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _model_metadata(embedder: SequenceEmbedder) -> dict[str, Any]:
    return {
        "name": embedder.model_name,
        "representation_layer": int(getattr(embedder, "representation_layer", 0)),
        "checkpoint_sha256": getattr(embedder, "checkpoint_sha256", None),
        "fair_esm_version": _package_version("fair-esm") or _package_version("esm"),
        "torch_version": _package_version("torch"),
    }


def build_target_sequence_embeddings(
    *,
    target_csv: Path,
    target_manifest: Path,
    out_parquet: Path,
    out_manifest: Path,
    embedder: SequenceEmbedder | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "auto",
    tokens_per_batch: int = DEFAULT_TOKENS_PER_BATCH,
    max_chunk_residues: int = DEFAULT_MAX_CHUNK_RESIDUES,
) -> dict[str, Any]:
    _assert_safe_output_files(
        input_paths=_declared_sequence_inputs(target_csv, target_manifest),
        output_paths=(out_parquet, out_manifest),
    )
    _remove_outputs(out_parquet, out_manifest)
    try:
        target_ids, sequences, source_artifacts = load_screenable_targets(
            target_csv, target_manifest
        )
        if embedder is None:
            embedder = FairEsmEmbedder(model_name=model_name, device=device)
        rows = embed_sequences(
            target_ids,
            sequences,
            embedder,
            max_chunk_residues=max_chunk_residues,
            tokens_per_batch=tokens_per_batch,
        )
        _write_embeddings_parquet_atomic(rows, out_parquet)
        output_record = {
            "path": str(out_parquet.resolve()),
            "sha256": _sha256(out_parquet),
            "rows": len(rows),
            "dimension": int(embedder.dimension),
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "source_schema_version": SCREENABLE_SCHEMA_VERSION,
            "inputs": {
                "target_clusters": source_artifacts["target_csv"],
                "target_cluster_manifest": source_artifacts["target_manifest"],
            },
            "source_artifacts": source_artifacts,
            "model": _model_metadata(embedder),
            "embedding": {
                "dimension": int(embedder.dimension),
                "dtype": "float32",
                "pooling": "mean residue pooling weighted by chunk residue counts",
                "chunk_policy": {
                    "max_chunk_residues": int(max_chunk_residues),
                    "max_model_residues_per_chunk": DEFAULT_MAX_CHUNK_RESIDUES,
                    "overlap": 0,
                    "deterministic_non_overlapping": True,
                    "tokens_per_batch": int(tokens_per_batch),
                },
            },
            "universe_policy": {
                "evaluation_panel_used": False,
                "known_target_assistance": False,
                "production_contract_passes": True,
            },
            "contract": {
                "evaluation_panel_used": False,
                "known_target_assistance": False,
                "training_labels_used": False,
            },
            "outputs": {
                "embeddings": output_record,
            },
            "artifact": {
                "path": output_record["path"],
                "sha256": output_record["sha256"],
                "rows": output_record["rows"],
                "columns": ["uniprot", "embedding"],
            },
        }
        _write_json_atomic(manifest, out_manifest)
        return manifest
    except BaseException:
        _remove_outputs(out_parquet, out_manifest)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-csv", required=True, type=Path)
    parser.add_argument("--target-manifest", required=True, type=Path)
    parser.add_argument("--out-parquet", required=True, type=Path)
    parser.add_argument("--out-manifest", required=True, type=Path)
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        choices=(DEFAULT_MODEL_NAME,),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tokens-per-batch", type=int, default=DEFAULT_TOKENS_PER_BATCH)
    parser.add_argument("--max-chunk-residues", type=int, default=DEFAULT_MAX_CHUNK_RESIDUES)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_target_sequence_embeddings(
        target_csv=args.target_csv,
        target_manifest=args.target_manifest,
        out_parquet=args.out_parquet,
        out_manifest=args.out_manifest,
        model_name=args.model_name,
        device=args.device,
        tokens_per_batch=args.tokens_per_batch,
        max_chunk_residues=args.max_chunk_residues,
    )
    print(
        "[target-sequence-embeddings] wrote "
        f"rows={manifest['artifact']['rows']} dim={manifest['embedding']['dimension']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
