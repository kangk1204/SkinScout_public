"""Regression tests for the device the performance-v2 scorer runs on.

Scoring hardcoded CUDA whenever a GPU was merely visible, with no way to ask
for CPU. A model trained on CPU then died on a busy GPU before a single batch
ran: the existing retry only halves the batch on out-of-memory raised during
batching, and never sees a failure to place the model at all.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))

import performance_v2_model as model_module  # noqa: E402
from performance_v2_model import ContractError, _placed_scoring_model  # noqa: E402


class _FakeTorch:
    """Minimal stand-in: CUDA is visible but placing anything on it fails."""

    def __init__(self, *, cuda_available: bool, cuda_placement_fails: bool) -> None:
        self.cuda = SimpleNamespace(is_available=lambda: cuda_available)
        self._cuda_placement_fails = cuda_placement_fails
        self.placed_on: list[str] = []

    def device(self, name: str) -> str:
        return name


class _FakeHead:
    def __init__(self, fake_torch: _FakeTorch) -> None:
        self._torch = fake_torch

    def to(self, device: str):
        if device == "cuda" and self._torch._cuda_placement_fails:
            raise RuntimeError("CUDA error: no kernel image is available")
        self._torch.placed_on.append(device)
        return self


@pytest.fixture
def stub_head(monkeypatch):
    def _install(fake_torch: _FakeTorch):
        monkeypatch.setattr(
            model_module,
            "_make_torch_pair_head",
            lambda *args, **kwargs: _FakeHead(fake_torch),
        )
        return fake_torch

    return _install


def test_an_unusable_gpu_falls_back_to_cpu_instead_of_dying(stub_head) -> None:
    torch = stub_head(_FakeTorch(cuda_available=True, cuda_placement_fails=True))

    with pytest.warns(RuntimeWarning, match="falling back to CPU"):
        device, _ = _placed_scoring_model(torch, 4, 4, 2, device_mode=None)

    assert device == "cpu"
    assert torch.placed_on == ["cpu"]


def test_a_working_gpu_is_still_used_when_nothing_is_requested(stub_head) -> None:
    torch = stub_head(_FakeTorch(cuda_available=True, cuda_placement_fails=False))

    device, _ = _placed_scoring_model(torch, 4, 4, 2, device_mode=None)

    assert device == "cuda"


def test_cpu_can_be_requested_even_with_a_gpu_present(stub_head) -> None:
    """A CPU-trained model has to be scoreable on CPU on purpose."""
    torch = stub_head(_FakeTorch(cuda_available=True, cuda_placement_fails=False))

    device, _ = _placed_scoring_model(torch, 4, 4, 2, device_mode="cpu")

    assert device == "cpu"
    assert torch.placed_on == ["cpu"]


def test_an_explicit_cuda_request_is_not_silently_downgraded(stub_head) -> None:
    """Asking for CUDA and getting CPU would misreport what produced a score."""
    torch = stub_head(_FakeTorch(cuda_available=True, cuda_placement_fails=True))

    with pytest.raises(RuntimeError):
        _placed_scoring_model(torch, 4, 4, 2, device_mode="cuda")


def test_requesting_cuda_without_a_gpu_is_refused(stub_head) -> None:
    torch = stub_head(_FakeTorch(cuda_available=False, cuda_placement_fails=False))

    with pytest.raises(ContractError, match="no CUDA device"):
        _placed_scoring_model(torch, 4, 4, 2, device_mode="cuda")


def test_an_unknown_device_is_refused(stub_head) -> None:
    torch = stub_head(_FakeTorch(cuda_available=True, cuda_placement_fails=False))

    with pytest.raises(ContractError, match="cpu or cuda"):
        _placed_scoring_model(torch, 4, 4, 2, device_mode="mps")


def test_only_the_torch_scorers_take_a_device(stub_head) -> None:
    """Fixture scorers never touch torch; accepting a device would mislead."""
    import inspect

    for name in ("score_torch_pair_model", "score_torch_query_vector"):
        assert "device_mode" in inspect.signature(getattr(model_module, name)).parameters
    for name in ("score_fixture_model", "score_fixture_query_vector"):
        assert "device_mode" not in inspect.signature(
            getattr(model_module, name)
        ).parameters
