#!/usr/bin/env python3
"""stage3_psichic.py — PSICHIC sequence-only DTI over the human proteome."""

from __future__ import annotations

import argparse
import csv
import gc
import logging
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Callable

from rdkit import Chem

LOG = logging.getLogger("stage3.psichic")


def _add_repo_root_to_path() -> None:
    root = Path(__file__).resolve().parents[1]
    root_text = str(root)
    if root_text in sys.path:
        return
    pythonpath_entries = [
        path for path in os.environ.get("PYTHONPATH", "").split(os.pathsep) if path
    ]
    sys.path.insert(min(len(sys.path), 1 + len(pythonpath_entries)), root_text)


_add_repo_root_to_path()


def _remove_outputs(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def _tmp_output(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_suffix(path.suffix + ".tmp")


def _is_cuda_oom(exc: BaseException) -> bool:
    try:
        import torch
    except Exception:
        torch = None  # type: ignore[assignment]
    if torch is not None:
        oom_classes = [
            getattr(torch, "OutOfMemoryError", None),
            getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None),
        ]
        if any(cls is not None and isinstance(exc, cls) for cls in oom_classes):
            return True
    return "OutOfMemoryError" in type(exc).__name__ and "CUDA out of memory" in str(exc)


def _clear_cuda_cache(exc: BaseException | None = None) -> None:
    if exc is not None and exc.__traceback__ is not None:
        traceback.clear_frames(exc.__traceback__)
    gc.collect()
    try:
        import torch
    except Exception:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _predict_batch_with_cuda_oom_backoff(
    model: object,
    smiles: str,
    batch: list[tuple[str, str]],
    score_batch_size: int,
    cpu_fallback_model: Callable[[], object] | None = None,
) -> list[object]:
    seqs = [item[1] for item in batch]
    try:
        return list(
            model.predict_batch(  # type: ignore[attr-defined]
                smiles,
                seqs,
                score_batch_size=score_batch_size,
            )
        )
    except Exception as exc:
        if not _is_cuda_oom(exc):
            raise
        _clear_cuda_cache(exc)
        if len(batch) <= 1:
            if cpu_fallback_model is not None:
                LOG.warning(
                    "PSICHIC CUDA OOM at batch size 1; retrying one window on CPU"
                )
                cpu_model = cpu_fallback_model()
                return list(
                    cpu_model.predict_batch(  # type: ignore[attr-defined]
                        smiles,
                        seqs,
                        score_batch_size=score_batch_size,
                    )
                )
            raise SystemExit(
                "PSICHIC CUDA out-of-memory even at batch size 1; "
                "use a larger GPU or reduce the receptor set."
            ) from None
        next_size = max(1, len(batch) // 2)
        LOG.warning(
            "PSICHIC CUDA OOM at batch size %d; retrying with batch size %d",
            len(batch),
            next_size,
        )
        scores: list[object] = []
        for start in range(0, len(batch), next_size):
            scores.extend(
                _predict_batch_with_cuda_oom_backoff(
                    model,
                    smiles,
                    batch[start : start + next_size],
                    score_batch_size,
                    cpu_fallback_model,
                )
            )
        return scores


def ligand_smiles(sdf: Path) -> str:
    sup = Chem.SDMolSupplier(str(sdf), removeHs=False)
    for m in sup:
        if m is not None:
            return Chem.MolToSmiles(m, canonical=True)
    raise SystemExit(f"No mol in {sdf}")


def read_canonical_fasta(path: Path) -> dict[str, str]:
    """Read canonical sequences without deriving peptide adjacency from a PDB."""
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"Canonical protein FASTA is required and must be non-empty: {path}")
    records: dict[str, str] = {}
    accession: str | None = None
    chunks: list[str] = []

    def store() -> None:
        if accession is None:
            return
        sequence = "".join(chunks).upper().rstrip("*")
        if not sequence:
            raise SystemExit(f"Canonical FASTA has empty sequence for {accession}: {path}")
        invalid = sorted(set(sequence) - set("ABCDEFGHIKLMNPQRSTVWXYZUO"))
        if invalid:
            raise SystemExit(
                f"Canonical FASTA has unsupported residues {invalid} for {accession}: {path}"
            )
        records[accession] = sequence

    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            store()
            token = line[1:].split()[0]
            parts = token.split("|")
            accession = (parts[1] if len(parts) >= 2 else token).strip()
            if not accession or accession in records:
                raise SystemExit(
                    f"Canonical FASTA has blank or duplicate accession at line {line_number}: {path}"
                )
            chunks = []
        else:
            if accession is None:
                raise SystemExit(
                    f"Canonical FASTA sequence precedes first header at line {line_number}: {path}"
                )
            chunks.append("".join(line.split()))
    store()
    if not records:
        raise SystemExit(f"Canonical protein FASTA has no records: {path}")
    return records


def sequence_windows(seq: str, max_length: int, stride: int) -> list[str]:
    if max_length < 1:
        raise ValueError(f"max_length must be >= 1: {max_length}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1: {stride}")
    if len(seq) <= max_length:
        return [seq]
    end_start = len(seq) - max_length
    starts = list(range(0, end_start + 1, stride))
    if starts[-1] != end_start:
        starts.append(end_start)
    return [seq[start : start + max_length] for start in starts]


def _is_bool_like(value: object) -> bool:
    return (
        isinstance(value, bool)
        or type(value).__name__ == "bool_"
        or (
            isinstance(value, str)
            and value.strip().lower() in {"true", "false"}
        )
    )


def _checked_psichic_score(value: object, target_id: str) -> float:
    if _is_bool_like(value):
        raise SystemExit(
            f"PSICHIC score for {target_id} must be numeric: {value!r}"
        )
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            f"PSICHIC score for {target_id} must be numeric: {value!r}"
        ) from exc
    if not math.isfinite(score):
        raise SystemExit(
            f"PSICHIC score for {target_id} must be finite: {value!r}"
        )
    return score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ligand-sdf", required=True, type=Path)
    parser.add_argument("--clean-dir", required=True, type=Path)
    parser.add_argument("--sequence-fasta", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--score-batch-size", type=int, default=8)
    parser.add_argument("--max-sequence-length", type=int, default=700)
    parser.add_argument("--sequence-window-stride", type=int, default=None)
    parser.add_argument("--out-scores", required=True, type=Path)
    parser.add_argument(
        "--allow-unavailable",
        action="store_true",
        help="Emit an explicit header-only degraded table when PSICHIC is unavailable.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.batch_size < 1:
        raise SystemExit(f"--batch-size must be >= 1: {args.batch_size}")
    if args.score_batch_size < 1:
        raise SystemExit(f"--score-batch-size must be >= 1: {args.score_batch_size}")
    if args.max_sequence_length < 1:
        raise SystemExit(
            f"--max-sequence-length must be >= 1: {args.max_sequence_length}"
        )
    stride = args.sequence_window_stride or args.max_sequence_length
    if stride < 1:
        raise SystemExit(f"--sequence-window-stride must be >= 1: {stride}")

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    _remove_outputs(args.out_scores)
    smi = ligand_smiles(args.ligand_sdf)
    pdb_files = sorted(args.clean_dir.glob("*_clean.pdb"))
    if not pdb_files:
        raise SystemExit(f"No cleaned receptor PDBs found in {args.clean_dir}")
    LOG.info("PSICHIC over %d receptors", len(pdb_files))
    sequences = read_canonical_fasta(args.sequence_fasta)
    receptor_ids = [pdb.stem.removesuffix("_clean") for pdb in pdb_files]
    missing_sequences = sorted(set(receptor_ids) - set(sequences))
    if missing_sequences:
        shown = ", ".join(missing_sequences[:10])
        suffix = "..." if len(missing_sequences) > 10 else ""
        raise SystemExit(
            "Canonical FASTA lacks sequence(s) for cleaned receptor(s): "
            f"{shown}{suffix}"
        )

    try:
        from psichic import PSICHIC   # type: ignore[import-untyped]
        model = PSICHIC.load_pretrained()
    except ImportError:
        if not args.allow_unavailable:
            raise SystemExit(
                "psichic is required for Stage 3 DTI sanity evidence; install it or set "
                "docking.allow_psichic_unavailable=true for an explicit degraded run"
            )
        LOG.warning("psichic not installed; emitting explicit header-only degraded result")
        model = None

    cpu_model: object | None = None

    def cpu_fallback_model() -> object:
        nonlocal cpu_model
        if cpu_model is None:
            LOG.warning("Loading PSICHIC CPU fallback model after CUDA OOM")
            cpu_model = PSICHIC.load_pretrained(device="cpu")  # type: ignore[name-defined]
        return cpu_model

    target_order: list[str] = []
    target_scores: dict[str, float] = {}
    long_targets = 0
    total_windows = 0
    tmp_scores = _tmp_output(args.out_scores)
    with tmp_scores.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["target_id", "psichic_score", "score"])
        if model is None:
            tmp_scores.replace(args.out_scores)
            return
        batch: list[tuple[str, str]] = []
        def flush() -> None:
            if not batch:
                return
            scores = _predict_batch_with_cuda_oom_backoff(
                model,
                smi,
                batch,
                args.score_batch_size,
                cpu_fallback_model,
            )
            if len(scores) != len(batch):
                raise SystemExit(
                    "PSICHIC returned "
                    f"{len(scores)} score(s) for {len(batch)} receptor window(s)"
                )
            for (uid, _seq), score in zip(batch, scores, strict=True):
                score_float = _checked_psichic_score(score, uid)
                target_scores[uid] = max(target_scores.get(uid, score_float), score_float)
            batch.clear()
        for pdb in pdb_files:
            uid = pdb.stem.removesuffix("_clean")
            seq = sequences[uid]
            if uid not in target_scores:
                target_order.append(uid)
            windows = sequence_windows(seq, args.max_sequence_length, stride)
            total_windows += len(windows)
            if len(windows) > 1:
                long_targets += 1
            for window in windows:
                batch.append((uid, window))
                if len(batch) >= args.batch_size:
                    flush()
        flush()
        for uid in target_order:
            score = target_scores.get(uid)
            if score is None:
                continue
            w.writerow([uid, f"{score:.4f}", f"{score:.4f}"])
    if not target_scores:
        tmp_scores.unlink(missing_ok=True)
        raise SystemExit("PSICHIC produced no ranked receptor scores")
    tmp_scores.replace(args.out_scores)
    if long_targets:
        LOG.info(
            "PSICHIC windowed %d long receptor sequence(s) into %d total windows "
            "(max_length=%d, stride=%d)",
            long_targets,
            total_windows,
            args.max_sequence_length,
            stride,
        )
    LOG.info("PSICHIC scores → %s", args.out_scores)


if __name__ == "__main__":
    main()
