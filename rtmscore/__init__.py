"""Compatibility adapter for the upstream RTMScore repository.

RTMScore is distributed as a GitHub repository with an example script rather
than an installable Python package. This adapter provides the
``predict_affinity(protein_pdb, ligand_sdf)`` API used by the SkinScout wrappers
and delegates to the upstream implementation lazily at call time.

``scatter_add`` backend provenance
----------------------------------

torch_scatter is a compiled extension and is frequently absent from the
environment RTMScore runs in. This adapter installs a pure-torch fallback, but
a fallback is not the same thing as the real extension: their numeric equality
is unverified until ``probe_scatter_parity()`` compares the active backend
against an independent reference on this machine. ``backend_provenance()``
records which backend is in use, and ``probe_scatter_parity()`` reports the
parity evidence - or says plainly that the real backend is absent and no
parity is claimed.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from typing import Iterable


DEFAULT_ROOT = Path(
    os.environ.get("RTMSCORE_ROOT", str(Path.home() / ".local" / "opt" / "RTMScore"))
)
DEFAULT_MODEL = DEFAULT_ROOT / "trained_models" / "rtmscore_model1.pth"

SHIM_FLAG = "__skinscout_torch_scatter_shim__"

# Upstream VSDataset wraps its structure parsers and raises these messages (or
# collapses to an empty zip) when the receptor/ligand cannot be turned into a
# graph. They are the expected per-structure failures; anything else out of
# ``module.scoring`` is treated as a batch-wide model/programming failure.
STRUCTURE_ERROR_MESSAGES = (
    "graph of pocket cannot be generated",
    "ligands should be",
    "will be supported",
    "should be a list of rdkit",
    "not enough values to unpack",
)


class RTMScoreStructureError(RuntimeError):
    """Expected bad structure input; only the current target is affected."""

    rtmscore_structure_error = True


def _module_available(name: str) -> bool:
    if name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _real_torch_scatter_available() -> bool:
    module = sys.modules.get("torch_scatter")
    if module is not None:
        return not getattr(module, SHIM_FLAG, False)
    try:
        return importlib.util.find_spec("torch_scatter") is not None
    except (ImportError, ValueError):
        return False


def _install_torch_scatter_fallback() -> None:
    try:
        import torch_scatter  # type: ignore[import-not-found]  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    import torch

    module = types.ModuleType("torch_scatter")
    setattr(module, SHIM_FLAG, True)
    module.__skinscout_shim_reason__ = (  # type: ignore[attr-defined]
        "real torch_scatter extension is not installed"
    )

    def scatter_add(src, index, dim=0, out=None, dim_size=None):
        """Sum ``src`` into destination bins given by a 1-D ``index``.

        Semantics match the subset of ``torch_scatter.scatter_add`` upstream
        uses: a 1-D index is broadcast along ``dim``; ``dim_size`` sizes the
        output; dtype and device follow ``src``. Not parity-proven against the
        real extension - see ``probe_scatter_parity``.
        """
        if dim < 0:
            dim += src.dim()
        if dim_size is None:
            dim_size = int(index.max().item()) + 1 if index.numel() else 0
        if out is None:
            size = list(src.shape)
            size[dim] = dim_size
            out = src.new_zeros(size)
        if index.dim() == 1:
            view = [1] * src.dim()
            view[dim] = index.shape[0]
            index = index.view(view).expand_as(src)
        else:
            while index.dim() < src.dim():
                index = index.unsqueeze(-1)
            index = index.expand_as(src)
        index = index.to(device=src.device, dtype=torch.long)
        return out.scatter_add_(dim, index, src)

    module.scatter_add = scatter_add
    sys.modules["torch_scatter"] = module


def backend_provenance() -> dict[str, object]:
    """Which ``scatter_add`` backend is active, without importing torch.

    ``parity_claim`` stays "unverified" here: provenance describes where the
    code came from, not whether its numbers match the real extension. Only
    ``probe_scatter_parity()`` can upgrade that claim, and only when the real
    backend is present to compare against.
    """
    shim = sys.modules.get("torch_scatter")
    shim_installed = shim is not None and bool(getattr(shim, SHIM_FLAG, False))
    real_available = _real_torch_scatter_available()
    if shim_installed:
        backend = "shim"
    elif real_available:
        backend = "real"
    else:
        backend = "unavailable"
    return {
        "backend": backend,
        "real_backend_available": real_available,
        "shim_installed": shim_installed,
        "shim_reason": (
            getattr(shim, "__skinscout_shim_reason__", None)
            if shim_installed
            else None
        ),
        "torch_available": _module_available("torch"),
        "aggregations_used": ["scatter_add"],
        "parity_claim": "unverified",
    }


def _reference_scatter_add(torch, src, index, dim=0, dim_size=None):
    """Independent per-element reference, no ``scatter_add_`` involved.

    A python accumulation is slow but is the point: comparing an implementation
    against the same operator it delegates to proves nothing.
    """
    if dim < 0:
        dim += src.dim()
    flat_index = [int(value) for value in index.reshape(-1).tolist()]
    if dim_size is None:
        dim_size = (max(flat_index) + 1) if flat_index else 0
    out_shape = [
        dim_size if axis == dim else int(src.shape[axis])
        for axis in range(src.dim())
    ]
    out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
    for position in range(int(src.shape[dim])):
        src_slices = [slice(None)] * src.dim()
        src_slices[dim] = position
        out_slices = [slice(None)] * out.dim()
        out_slices[dim] = flat_index[position]
        out[tuple(out_slices)] = out[tuple(out_slices)] + src[tuple(src_slices)]
    return out


def _parity_inputs(torch, *, dtype, device) -> Iterable[tuple]:
    index = torch.tensor([0, 2, 0, 1], dtype=torch.long, device=device)
    return (
        (
            "1d_dim0_repeated_indices",
            torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=dtype, device=device),
            index,
            0,
            None,
        ),
        (
            "2d_dim0_dim_size_larger_than_max",
            torch.arange(12, dtype=dtype, device=device).reshape(4, 3),
            index,
            0,
            5,
        ),
        (
            "2d_dim1_broadcast_index",
            torch.arange(12, dtype=dtype, device=device).reshape(2, 6),
            torch.tensor([0, 2, 0, 1, 2, 0], dtype=torch.long, device=device),
            1,
            3,
        ),
    )


def probe_scatter_parity(
    *,
    dtypes: tuple[str, ...] = ("float32", "float64"),
    device: str = "cpu",
) -> dict[str, object]:
    """Compare the active backend against an independent reference.

    The shim only installs when the real extension is missing, so this can
    never compare shim against real in one process. With the real backend
    absent it skips and records that no parity is claimed; it does not
    substitute a self-check for parity evidence.
    """
    provenance = backend_provenance()
    if not provenance["real_backend_available"]:
        return {
            "status": "skipped",
            "reason": (
                "real torch_scatter backend is not installed; shim/real parity "
                "cannot be measured, so no parity is claimed"
            ),
            "backend": provenance["backend"],
            "parity_claim": "unverified",
        }
    try:
        import torch
        import torch_scatter
    except ImportError as exc:
        return {
            "status": "skipped",
            "reason": f"parity probe import failed: {exc.name}",
            "backend": provenance["backend"],
            "parity_claim": "unverified",
        }
    if device == "cuda" and not torch.cuda.is_available():
        return {
            "status": "skipped",
            "reason": "cuda was requested for the parity probe but is not available",
            "backend": provenance["backend"],
            "parity_claim": "unverified",
        }
    cases: list[dict[str, object]] = []
    for dtype_name in dtypes:
        dtype = getattr(torch, dtype_name)
        for name, src, index, dim, dim_size in _parity_inputs(
            torch, dtype=dtype, device=device
        ):
            actual = torch_scatter.scatter_add(
                src, index, dim=dim, dim_size=dim_size
            )
            reference = _reference_scatter_add(
                torch, src, index, dim=dim, dim_size=dim_size
            )
            match = actual.shape == reference.shape and bool(
                torch.allclose(actual, reference)
            )
            cases.append(
                {
                    "case": name,
                    "dtype": dtype_name,
                    "device": device,
                    "match": match,
                }
            )
    passed = all(bool(case["match"]) for case in cases)
    return {
        "status": "passed" if passed else "failed",
        "backend": provenance["backend"],
        "aggregation": "scatter_add",
        "dtypes": list(dtypes),
        "device": device,
        "parity_claim": "verified" if passed else "failed",
        "cases": cases,
    }


def _is_structure_error(exc: BaseException) -> bool:
    if isinstance(
        exc, (FileNotFoundError, IsADirectoryError, NotADirectoryError, PermissionError)
    ):
        return True
    text = str(exc).lower()
    return any(marker in text for marker in STRUCTURE_ERROR_MESSAGES)


def _load_upstream():
    script = DEFAULT_ROOT / "example" / "rtmscore.py"
    if not script.exists():
        raise RuntimeError(
            f"RTMScore upstream checkout not found at {DEFAULT_ROOT}. "
            "Clone https://github.com/sc8668/RTMScore or set RTMSCORE_ROOT."
        )
    root = str(DEFAULT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    spec = importlib.util.spec_from_file_location("_rtmscore_upstream", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load RTMScore script: {script}")
    module = importlib.util.module_from_spec(spec)
    try:
        _install_torch_scatter_fallback()
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "RTMScore upstream dependency is missing. Use the dedicated RTMScore "
            "environment from the upstream README or install the missing module: "
            f"{exc.name}"
        ) from exc
    return module


def _usable_device(requested: str | None) -> str:
    """Pick a device DGL can actually run on.

    RTMScore runs on DGL, and a CPU-only DGL build raises "Device API cuda is
    not enabled" the moment a graph is moved. Selecting CUDA from
    torch.cuda.is_available() alone therefore failed every call on a machine
    with a visible GPU and a CPU DGL - which is why this scorer had never
    produced a usable score here. An explicit request is honoured as given.
    """
    import warnings

    import torch

    if requested is not None:
        return requested
    if not torch.cuda.is_available():
        return "cpu"
    try:
        import dgl

        dgl.graph(([0], [0])).to("cuda")
    except Exception:
        warnings.warn(
            "DGL cannot use CUDA; running RTMScore on CPU",
            RuntimeWarning,
            stacklevel=2,
        )
        return "cpu"
    return "cuda"


def predict_affinity(
    protein_pdb: str | os.PathLike[str],
    ligand_sdf: str | os.PathLike[str],
    *,
    model_path: str | os.PathLike[str] | None = None,
    cutoff: float = 10.0,
    device: str | None = None,
) -> float:
    module = _load_upstream()
    model = Path(model_path) if model_path is not None else DEFAULT_MODEL
    if not model.exists():
        raise RuntimeError(f"RTMScore model checkpoint not found: {model}")

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        babel_libdir = Path(conda_prefix) / "lib" / "openbabel" / "3.1.0"
        if babel_libdir.exists():
            os.environ.setdefault("BABEL_LIBDIR", str(babel_libdir))

    args = {
        "batch_size": 128,
        "dist_threhold": 5,
        "device": _usable_device(device),
        "num_workers": 2,
        "num_node_featsp": 41,
        "num_node_featsl": 41,
        "num_edge_featsp": 5,
        "num_edge_featsl": 10,
        "hidden_dim0": 128,
        "hidden_dim": 128,
        "n_gaussians": 10,
        "dropout_rate": 0.10,
    }
    try:
        _ids, scores = module.scoring(
            prot=str(protein_pdb),
            lig=str(ligand_sdf),
            modpath=str(model),
            cut=cutoff,
            gen_pocket=False,
            explicit_H=False,
            use_chirality=True,
            parallel=False,
            **args,
        )
    except Exception as exc:
        # The upstream dataset is constructed inside this one call, so the
        # only way to separate a bad structure from a broken model is by the
        # error it produces. Known structure-parse failures become per-target
        # errors; everything else keeps propagating and fails the batch.
        if _is_structure_error(exc):
            raise RTMScoreStructureError(str(exc)) from exc
        raise
    if len(scores) == 0:
        raise RTMScoreStructureError("RTMScore returned no scores")
    return float(scores[0])
