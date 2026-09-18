"""Compatibility adapter for the upstream RTMScore repository.

RTMScore is distributed as a GitHub repository with an example script rather
than an installable Python package. This adapter provides the
``predict_affinity(protein_pdb, ligand_sdf)`` API used by the SkinScout wrappers
and delegates to the upstream implementation lazily at call time.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path


DEFAULT_ROOT = Path(
    os.environ.get("RTMSCORE_ROOT", str(Path.home() / ".local" / "opt" / "RTMScore"))
)
DEFAULT_MODEL = DEFAULT_ROOT / "trained_models" / "rtmscore_model1.pth"


def _install_torch_scatter_fallback() -> None:
    try:
        import torch_scatter  # type: ignore[import-not-found]  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    import torch

    module = types.ModuleType("torch_scatter")

    def scatter_add(src, index, dim=0, out=None, dim_size=None):
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
    if len(scores) == 0:
        raise RuntimeError("RTMScore returned no scores")
    return float(scores[0])
