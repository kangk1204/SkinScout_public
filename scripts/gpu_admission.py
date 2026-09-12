#!/usr/bin/env python3
"""Measure whether the primary NVIDIA GPU has room for a SkinScout run."""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


MIN_FREE_MIB = 6144
MAX_UTILIZATION_PERCENT = 90
MAX_BENIGN_DESKTOP_CONTEXT_MIB = 256
# GPU 메모리를 조금 잡지만 모델을 돌릴 수 없는 데스크톱 프로세스들. 이것을 "외부
# 계산"으로 세면 표적 예측이 통째로 막힌다. 실제로 설정 창을 열어 둔 것만으로
# 전부 blocked 가 됐다 - GPU는 사용률 0%에 여유 11.7GB였는데도.
#
# 이 목록은 이름만으로 통과시키지 않는다. 256 MiB 상한이 함께 걸려 있어서, 같은
# 이름의 프로세스가 실제로 모델을 올리면 걸러진다.
BENIGN_DESKTOP_PROCESS_NAMES = frozenset(
    {
        "gnome-terminal-server",
        "ptyxis",
        # GNOME 셸 계열. 창을 열어 두는 것만으로 32 MiB 정도를 잡는다.
        "gnome-control-center",
        "gnome-shell",
        "gnome-software",
        "nautilus",
        # 브라우저도 합성에 GPU를 쓴다. 문서를 읽는 동안 분석이 막히면 안 된다.
        "chrome",
        "google-chrome",
        "firefox",
        "Xwayland",
    }
)


def _is_benign_desktop_context(
    process_name: str,
    used_memory_mib: int | None,
) -> bool:
    """Recognize small terminal-rendering contexts that cannot run a model."""
    basename = Path(process_name).name.lower()
    return (
        basename in BENIGN_DESKTOP_PROCESS_NAMES
        and used_memory_mib is not None
        and 0 <= used_memory_mib <= MAX_BENIGN_DESKTOP_CONTEXT_MIB
    )


def _unavailable(detail: str) -> dict[str, Any]:
    return {
        "status": "warning",
        "label": "GPU",
        "detail": detail,
        "detected": False,
        "available_for_analysis": False,
        "minimum_free_mib": MIN_FREE_MIB,
        "maximum_utilization_percent": MAX_UTILIZATION_PERCENT,
        "active_compute_processes": [],
        "external_compute_process_count": 0,
        "ignored_desktop_contexts": [],
        "ignored_desktop_context_count": 0,
        "devices": [],
        "device_details": [],
    }


def probe(
    *,
    command: str | None = None,
    cwd: Path | None = None,
) -> dict[str, Any]:
    command = command or shutil.which("nvidia-smi")
    if command is None:
        return _unavailable("nvidia-smi를 찾지 못했습니다.")
    try:
        result = subprocess.run(
            [
                command,
                "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=4,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return _unavailable("GPU 상태를 확인하지 못했습니다.")
    if result.returncode != 0 or not result.stdout.strip():
        return _unavailable(
            "NVIDIA GPU가 없거나 드라이버가 준비되지 않았습니다."
        )

    device_details: list[dict[str, Any]] = []
    for row in csv.reader(result.stdout.splitlines()):
        if len(row) != 7:
            continue
        index, uuid, name, total_raw, used_raw, free_raw, utilization_raw = (
            value.strip() for value in row
        )
        try:
            total_mib = int(float(total_raw))
            used_mib = int(float(used_raw))
            free_mib = int(float(free_raw))
            utilization_percent = int(float(utilization_raw))
        except ValueError:
            continue
        available_for_analysis = (
            free_mib >= MIN_FREE_MIB
            and utilization_percent <= MAX_UTILIZATION_PERCENT
        )
        device_details.append(
            {
                "index": index,
                "uuid": uuid,
                "name": name,
                "total_mib": total_mib,
                "used_mib": used_mib,
                "free_mib": free_mib,
                "utilization_percent": utilization_percent,
                "available_for_analysis": available_for_analysis,
            }
        )
    if not device_details:
        return _unavailable("GPU 상태 값을 해석하지 못했습니다.")

    primary = device_details[0]
    process_query_error: str | None = None
    active_compute_processes: list[dict[str, Any]] = []
    ignored_desktop_contexts: list[dict[str, Any]] = []
    try:
        process_result = subprocess.run(
            [
                command,
                "--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=4,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        process_query_error = "GPU compute process 상태를 확인하지 못했습니다."
    else:
        if process_result.returncode != 0:
            process_query_error = "GPU compute process 상태를 확인하지 못했습니다."
        else:
            for row in csv.reader(process_result.stdout.splitlines()):
                if len(row) != 4:
                    process_query_error = "GPU compute process 상태 값을 해석하지 못했습니다."
                    active_compute_processes = []
                    break
                pid_raw, gpu_uuid, process_name, memory_raw = (
                    value.strip() for value in row
                )
                try:
                    pid = int(pid_raw)
                except ValueError:
                    process_query_error = "GPU compute process 상태 값을 해석하지 못했습니다."
                    active_compute_processes = []
                    break
                if gpu_uuid != primary["uuid"] or pid == os.getpid():
                    continue
                try:
                    used_memory_mib: int | None = int(float(memory_raw))
                except ValueError:
                    used_memory_mib = None
                process = {
                    "pid": pid,
                    "gpu_uuid": gpu_uuid,
                    "process_name": process_name,
                    "used_memory_mib": used_memory_mib,
                }
                if _is_benign_desktop_context(process_name, used_memory_mib):
                    ignored_desktop_contexts.append(process)
                else:
                    active_compute_processes.append(process)

    available_for_analysis = (
        bool(primary["available_for_analysis"])
        and process_query_error is None
        and not active_compute_processes
    )
    detail = (
        f"{primary['name']} · 여유 {primary['free_mib']:,}/{primary['total_mib']:,} MiB"
        f" · 사용률 {primary['utilization_percent']}%"
    )
    if process_query_error is not None:
        detail += f" · {process_query_error}"
    elif active_compute_processes:
        detail += f" · 다른 GPU 계산 {len(active_compute_processes)}개가 실행 중입니다."
    elif ignored_desktop_contexts:
        detail += (
            " · 소형 터미널 GPU 컨텍스트 "
            f"{len(ignored_desktop_contexts)}개는 계산 작업에서 제외했습니다."
        )
    elif not available_for_analysis:
        detail += " · 다른 GPU 작업이 실행 중이거나 실행 여유가 부족합니다."
    return {
        "status": "ready" if available_for_analysis else "warning",
        "label": "GPU",
        "detail": detail,
        "detected": True,
        "available_for_analysis": available_for_analysis,
        "minimum_free_mib": MIN_FREE_MIB,
        "maximum_utilization_percent": MAX_UTILIZATION_PERCENT,
        "selected_index": primary["index"],
        "selected_uuid": primary["uuid"],
        "total_mib": primary["total_mib"],
        "used_mib": primary["used_mib"],
        "free_mib": primary["free_mib"],
        "utilization_percent": primary["utilization_percent"],
        "active_compute_processes": active_compute_processes,
        "external_compute_process_count": len(active_compute_processes),
        "ignored_desktop_contexts": ignored_desktop_contexts,
        "ignored_desktop_context_count": len(ignored_desktop_contexts),
        "compute_process_query_ok": process_query_error is None,
        "devices": [
            (
                f"{device['index']}: {device['name']}"
                f" ({device['free_mib']:,}/{device['total_mib']:,} MiB free,"
                f" {device['utilization_percent']}% utilization)"
            )
            for device in device_details
        ],
        "device_details": device_details,
    }
