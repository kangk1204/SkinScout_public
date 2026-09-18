"""Compatibility adapter for the upstream PSICHIC inference package.

The upstream repository does not expose the small ``PSICHIC.load_pretrained()``
surface used by this pipeline, so this module wraps its production inference
helpers while keeping the pipeline-facing API stable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable


DEFAULT_ROOT = Path(
    os.environ.get("PSICHIC_ROOT", str(Path.home() / ".local" / "opt" / "PSICHIC"))
)
DEFAULT_MODEL = os.environ.get("PSICHIC_MODEL", "PDBv2020_PSICHIC")


class PSICHICPrerequisiteError(RuntimeError):
    """PSICHIC cannot start because a prerequisite is missing.

    Covers a missing upstream checkout, a missing runtime dependency, and a
    missing or unreadable checkpoint. Inference failures are never wrapped in
    this type, so callers can offer an explicit diagnostic degraded path
    without hiding genuine model bugs.
    """

    def __init__(self, message: str, *, missing: str) -> None:
        super().__init__(message)
        self.missing = missing


def _readable_checkpoint(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            handle.read(1)
    except OSError:
        return False
    return True


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer >= 1: {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{name} must be an integer >= 1: {raw!r}")
    return value


def _load_upstream():
    prod_dir = DEFAULT_ROOT / "PSICHIC-prod"
    if not (prod_dir / "inference.py").exists():
        raise PSICHICPrerequisiteError(
            f"PSICHIC upstream checkout not found at {DEFAULT_ROOT}. "
            "Clone https://github.com/huankoh/PSICHIC or set PSICHIC_ROOT.",
            missing="upstream",
        )
    for path in (str(DEFAULT_ROOT), str(prod_dir)):
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        import inference  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise PSICHICPrerequisiteError(
            f"PSICHIC runtime dependency is missing: {exc.name}",
            missing=f"dependency:{exc.name}",
        ) from exc

    return inference


def _configure_esm_batch_sizes() -> None:
    import esm_batched.esm_batched as esm_batched  # type: ignore[import-not-found]

    esm_batched.LENGTH_BIN_BATCH_SIZES = [
        (300, _env_int("PSICHIC_ESM_SHORT_BATCH", 4)),
        (500, _env_int("PSICHIC_ESM_MEDIUM_BATCH", 2)),
        (700, _env_int("PSICHIC_ESM_LONG_BATCH", 1)),
    ]


class PSICHIC:
    """Small pipeline-facing wrapper around upstream PSICHIC-prod inference."""

    def __init__(self, model, inference_module, device, model_name: str) -> None:
        self.model = model
        self._inference = inference_module
        self.device = device
        self.model_name = model_name
        self.has_cls, self.has_mcls = inference_module._model_has_classification(model_name)

    @classmethod
    def load_pretrained(
        cls,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
    ) -> "PSICHIC":
        inference = _load_upstream()
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise PSICHICPrerequisiteError(
                f"PSICHIC runtime dependency is missing: {exc.name}",
                missing=f"dependency:{exc.name}",
            ) from exc

        # Upstream load_model() reads these three files unconditionally; probe
        # readability here so a missing checkpoint is reported as a
        # prerequisite rather than as an arbitrary inference crash.
        weights_dir = DEFAULT_ROOT / "trained_weights" / model_name
        for filename in ("config.json", "degree.pt", "model.pt"):
            checkpoint = weights_dir / filename
            if not _readable_checkpoint(checkpoint):
                raise PSICHICPrerequisiteError(
                    f"PSICHIC checkpoint is missing or unreadable: {checkpoint}",
                    missing=f"checkpoint:{filename}",
                )

        torch_device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        model = inference.load_model(model_name, torch_device)
        return cls(model, inference, torch_device, model_name)

    def _ligand_features(self, smiles: str) -> tuple[str, dict[str, dict]]:
        canonical_map, failed = self._inference._build_canonical_map([smiles])
        if failed or smiles not in canonical_map:
            raise ValueError(f"PSICHIC could not canonicalize ligand SMILES: {smiles!r}")
        canonical = canonical_map[smiles]

        # Avoid multiprocessing after CUDA init: featurize the one ligand inline.
        from utils.ligand_init import smiles2graph  # type: ignore[import-not-found]

        ligand = self._inference._post_transform_ligand(smiles2graph(canonical))
        return canonical, {canonical: ligand}

    def predict_batch(
        self,
        smiles: str,
        sequences: Iterable[str],
        *,
        score_batch_size: int | None = None,
    ) -> list[float]:
        seqs = [str(seq).strip().upper() for seq in sequences if str(seq).strip()]
        if not seqs:
            return []

        canonical, chunk_ligands = self._ligand_features(smiles)
        proteins = {f"seq{i}": seq for i, seq in enumerate(seqs)}
        _configure_esm_batch_sizes()
        protein_features = self._inference.precompute_proteins(proteins, self.device)
        batch_pairs = [(pid, seq, canonical, smiles) for pid, seq in proteins.items()]
        score_batch_size = score_batch_size or len(batch_pairs)
        if score_batch_size < 1:
            raise ValueError(f"score_batch_size must be >= 1: {score_batch_size}")

        scores: list[float] = []
        for start in range(0, len(batch_pairs), score_batch_size):
            chunk = batch_pairs[start : start + score_batch_size]
            reg_np, _cls_np, _mcls_np = self._inference._score_batch(
                chunk,
                chunk_ligands,
                protein_features,
                self.model,
                self.device.type == "cuda",
                self.has_cls,
                self.has_mcls,
                self.device,
            )
            scores.extend(float(value) for value in reg_np[: len(chunk)])
        return scores
