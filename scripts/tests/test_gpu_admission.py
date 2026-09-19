"""Tests for transient host GPU admission checks."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "scripts" / "gpu_admission.py"


def load_gpu_admission_module():
    spec = importlib.util.spec_from_file_location("gpu_admission_under_test", MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Result:
    def __init__(self, stdout: str, returncode: int = 0) -> None:
        self.stdout = stdout
        self.returncode = returncode


def fake_nvidia_smi(
    gpu_output: str,
    compute_output: str = "",
    *,
    compute_returncode: int = 0,
):
    def run(command: list[str], *args, **kwargs) -> Result:
        if any("--query-compute-apps" in item for item in command):
            return Result(compute_output, compute_returncode)
        return Result(gpu_output)

    return run


@pytest.mark.parametrize(
    ("free_mib", "utilization", "expected"),
    (
        (6144, 90, True),
        (6143, 10, False),
        (10000, 91, False),
    ),
)
def test_probe_enforces_memory_and_utilization_thresholds(
    free_mib: int,
    utilization: int,
    expected: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission = load_gpu_admission_module()
    used_mib = 12288 - free_mib
    output = (
        f"0, GPU-test, NVIDIA GeForce RTX 3080 Ti, 12288, {used_mib}, "
        f"{free_mib}, {utilization}\n"
    )
    monkeypatch.setattr(
        admission.subprocess,
        "run",
        fake_nvidia_smi(output),
    )

    payload = admission.probe(command="/usr/bin/nvidia-smi", cwd=ROOT)

    assert payload["available_for_analysis"] is expected
    assert payload["status"] == ("ready" if expected else "warning")


def test_probe_fails_closed_on_unparseable_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission = load_gpu_admission_module()
    monkeypatch.setattr(
        admission.subprocess,
        "run",
        fake_nvidia_smi("not,a,valid,row\n"),
    )

    payload = admission.probe(command="/usr/bin/nvidia-smi", cwd=ROOT)

    assert payload["detected"] is False
    assert payload["available_for_analysis"] is False
    assert "해석하지 못했습니다" in payload["detail"]


def test_probe_blocks_external_compute_process_even_with_free_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission = load_gpu_admission_module()
    gpu = "0, GPU-test, NVIDIA RTX, 12288, 1024, 11264, 5\n"
    compute = "4321, GPU-test, /opt/external/python, 1024\n"
    monkeypatch.setattr(
        admission.subprocess,
        "run",
        fake_nvidia_smi(gpu, compute),
    )
    monkeypatch.setattr(admission.os, "getpid", lambda: 9999)

    payload = admission.probe(command="/usr/bin/nvidia-smi", cwd=ROOT)

    assert payload["available_for_analysis"] is False
    assert payload["external_compute_process_count"] == 1
    assert payload["active_compute_processes"][0]["pid"] == 4321
    assert "다른 GPU 계산 1개" in payload["detail"]


def test_probe_ignores_its_own_cuda_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission = load_gpu_admission_module()
    gpu = "0, GPU-test, NVIDIA RTX, 12288, 1024, 11264, 5\n"
    compute = "4321, GPU-test, /opt/skinscout/python, 128\n"
    monkeypatch.setattr(
        admission.subprocess,
        "run",
        fake_nvidia_smi(gpu, compute),
    )
    monkeypatch.setattr(admission.os, "getpid", lambda: 4321)

    payload = admission.probe(command="/usr/bin/nvidia-smi", cwd=ROOT)

    assert payload["available_for_analysis"] is True
    assert payload["external_compute_process_count"] == 0


def test_probe_ignores_small_ptyxis_desktop_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission = load_gpu_admission_module()
    gpu = "0, GPU-test, NVIDIA RTX, 12288, 354, 11556, 0\n"
    compute = "4321, GPU-test, /usr/bin/ptyxis, 72\n"
    monkeypatch.setattr(
        admission.subprocess,
        "run",
        fake_nvidia_smi(gpu, compute),
    )
    monkeypatch.setattr(admission.os, "getpid", lambda: 9999)

    payload = admission.probe(command="/usr/bin/nvidia-smi", cwd=ROOT)

    assert payload["available_for_analysis"] is True
    assert payload["external_compute_process_count"] == 0
    assert payload["ignored_desktop_context_count"] == 1
    assert payload["ignored_desktop_contexts"][0]["pid"] == 4321
    assert "소형 터미널 GPU 컨텍스트 1개" in payload["detail"]


@pytest.mark.parametrize(
    "process_name",
    (
        "/usr/bin/gnome-control-center",
        "/usr/bin/gnome-shell",
        "/usr/bin/google-chrome",
        "/usr/bin/firefox",
    ),
)
def test_probe_ignores_the_desktop_apps_a_researcher_actually_has_open(
    process_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """설정 창이나 브라우저를 열어 둔 것만으로 표적 예측이 막히면 안 된다.

    실제로 그랬다. gnome-control-center 가 32 MiB를 잡고 있다는 이유로
    target_fast, target_comprehensive, report, substitute 가 전부 blocked 였다.
    GPU는 사용률 0%에 여유 11.7GB였다.
    """
    admission = load_gpu_admission_module()
    gpu = "0, GPU-test, NVIDIA RTX, 12288, 354, 11556, 0\n"
    compute = f"4321, GPU-test, {process_name}, 32\n"
    monkeypatch.setattr(admission.subprocess, "run", fake_nvidia_smi(gpu, compute))
    monkeypatch.setattr(admission.os, "getpid", lambda: 9999)

    payload = admission.probe(command="/usr/bin/nvidia-smi", cwd=ROOT)

    assert payload["available_for_analysis"] is True
    assert payload["external_compute_process_count"] == 0
    assert payload["ignored_desktop_context_count"] == 1


@pytest.mark.parametrize(
    ("process_name", "used_memory_mib"),
    (
        ("/usr/bin/ptyxis", 257),
        # 이름이 목록에 있어도 실제로 모델을 올리면 걸러야 한다.
        ("/usr/bin/google-chrome", 4096),
        ("/opt/external/python", 72),
    ),
)
def test_probe_does_not_ignore_large_or_non_terminal_compute_contexts(
    process_name: str,
    used_memory_mib: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission = load_gpu_admission_module()
    gpu = "0, GPU-test, NVIDIA RTX, 12288, 1024, 11264, 0\n"
    compute = f"4321, GPU-test, {process_name}, {used_memory_mib}\n"
    monkeypatch.setattr(
        admission.subprocess,
        "run",
        fake_nvidia_smi(gpu, compute),
    )
    monkeypatch.setattr(admission.os, "getpid", lambda: 9999)

    payload = admission.probe(command="/usr/bin/nvidia-smi", cwd=ROOT)

    assert payload["available_for_analysis"] is False
    assert payload["external_compute_process_count"] == 1
    assert payload["ignored_desktop_context_count"] == 0
