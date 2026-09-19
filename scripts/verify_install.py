#!/usr/bin/env python3
"""설치가 실제로 쓸 수 있는 상태인지 한 번에 점검한다.

`install_skinscout.sh`가 끝난 뒤 이 스크립트를 돌리면, 빠진 것이 무엇이고 그것이
분석을 막는지(FAIL) 아니면 일부 기능만 막는지(WARN)를 한 화면에 보여준다. 표준
라이브러리만 쓰므로 시스템 python3로 돌아가고, 패키지 확인은 cosmax-base env의
python을 빌려서 한다.

예:
  python3 scripts/verify_install.py                    # 기본 demo 프로파일
  python3 scripts/verify_install.py --profile analog   # 대체소재 전용 설치
  python3 scripts/verify_install.py --json             # 기계 판독용
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PROFILE_MIN_DISK_GB = {"analog": 5, "demo": 300, "full": 500}
PROFILE_MIN_MEM_GB = {"analog": 4, "demo": 32, "full": 32}

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"


def _result(checks: list[dict], check_id: str, label: str, status: str, detail: str) -> None:
    checks.append({"id": check_id, "label": label, "status": status, "detail": detail})


def find_env_python() -> Path | None:
    """cosmax-base env의 python. 두 가지 루트 관례를 모두 본다.

    이 호스트처럼 `~/.local/share/micromamba`와 `~/.local/share/mamba`가 둘 다
    있는 경우, 실제 패키지가 든 쪽을 골라야 점검이 거짓 실패하지 않는다.
    """
    roots = [Path(os.environ.get("MAMBA_ROOT_PREFIX", "~/.local/share/micromamba")).expanduser()]
    roots.append(Path("~/.local/share/mamba").expanduser())
    candidates = [
        root / "envs" / "cosmax-base" / "bin" / "python"
        for root in roots
    ]
    existing = [path for path in candidates if path.is_file() and os.access(path, os.X_OK)]
    for path in existing:
        probe = subprocess.run(
            [str(path), "-c", "import rdkit"],
            capture_output=True, text=True, timeout=120, check=False,
        )
        if probe.returncode == 0:
            return path
    return existing[0] if existing else None


def check_os(checks: list[dict]) -> None:
    system = platform.system()
    machine = platform.machine()
    if system != "Linux" or machine not in {"x86_64", "AMD64"}:
        _result(checks, "os", "운영체제", FAIL, f"{system} {machine} (Ubuntu 24.04 x86_64 필요)")
        return
    version = "unknown"
    release = Path("/etc/os-release")
    if release.is_file():
        match = re.search(r'^VERSION_ID="?([^"\n]+)', release.read_text(encoding="utf-8"), re.M)
        if match:
            version = match.group(1)
    if version == "24.04":
        _result(checks, "os", "운영체제", PASS, f"Ubuntu {version} {machine}")
    else:
        _result(checks, "os", "운영체제", WARN, f"Ubuntu {version} {machine} (24.04에서 검증됨)")


def check_resources(checks: list[dict], profile: str) -> None:
    min_disk = PROFILE_MIN_DISK_GB[profile]
    try:
        free_gb = shutil.disk_usage(ROOT).free / (1024**3)
        if free_gb >= min_disk:
            _result(checks, "disk", "여유 디스크", PASS, f"{free_gb:,.0f} GB (필요 {min_disk} GB)")
        else:
            _result(checks, "disk", "여유 디스크", FAIL, f"{free_gb:,.0f} GB (필요 {min_disk} GB)")
    except OSError as exc:
        _result(checks, "disk", "여유 디스크", WARN, str(exc))

    if profile == "analog":
        return
    min_mem = PROFILE_MIN_MEM_GB[profile]
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        _result(checks, "memory", "메모리", WARN, "/proc/meminfo 없음")
        return
    match = re.search(r"^MemTotal:\s+(\d+) kB", meminfo.read_text(encoding="utf-8"), re.M)
    if not match:
        _result(checks, "memory", "메모리", WARN, "MemTotal을 읽지 못함")
        return
    total_gb = int(match.group(1)) / (1024**2)
    status = PASS if total_gb >= min_mem else WARN
    _result(checks, "memory", "메모리", status, f"{total_gb:,.0f} GB (권장 {min_mem} GB)")


def check_gpu(checks: list[dict], profile: str) -> None:
    if profile == "analog":
        _result(checks, "gpu", "GPU", SKIP, "analog 프로파일은 GPU가 필요 없습니다")
        return
    binary = shutil.which("nvidia-smi")
    if binary is None:
        _result(checks, "gpu", "GPU", FAIL, "nvidia-smi가 없습니다 (드라이버 설치 필요)")
        return
    try:
        completed = subprocess.run(
            [binary, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _result(checks, "gpu", "GPU", FAIL, str(exc))
        return
    if completed.returncode != 0:
        _result(checks, "gpu", "GPU", FAIL, (completed.stderr or "nvidia-smi 실패").strip())
        return
    first = (completed.stdout or "").strip().splitlines()[0] if completed.stdout.strip() else ""
    _result(checks, "gpu", "GPU", PASS, first or "nvidia-smi 정상")
def check_runtime(checks: list[dict], env_python: Path | None) -> None:
    if env_python is None:
        _result(checks, "runtime", "SkinScout 실행 환경", FAIL, "cosmax-base env의 python을 찾지 못했습니다")
        return
    modules = "rdkit, pandas, numpy"
    completed = subprocess.run(
        [str(env_python), "-c", f"import {modules.replace(', ', ', ')}"],
        capture_output=True, text=True, timeout=120, check=False,
    )
    if completed.returncode == 0:
        _result(checks, "runtime", "SkinScout 실행 환경", PASS, f"cosmax-base · {modules}")
    else:
        _result(checks, "runtime", "SkinScout 실행 환경", FAIL, (completed.stderr or "").strip()[-200:])
        return
    # xlsx 표적 리스트 입력에만 쓰인다 - 없어도 나머지는 돈다.
    completed = subprocess.run(
        [str(env_python), "-c", "import openpyxl"],
        capture_output=True, text=True, timeout=60, check=False,
    )
    if completed.returncode == 0:
        _result(checks, "openpyxl", "엑셀 입력 지원", PASS, "openpyxl 있음")
    else:
        _result(checks, "openpyxl", "엑셀 입력 지원", WARN, "openpyxl 없음 (CSV는 됩니다)")


def check_tools(checks: list[dict], profile: str) -> None:
    if profile == "analog":
        _result(checks, "autogrid", "AutoGrid", SKIP, "analog 프로파일은 도킹을 쓰지 않습니다")
        return
    binary = ROOT / "tools" / "AutoGrid" / "autogrid4"
    manifest = ROOT / "tools" / "AutoGrid" / "source_manifest.json"
    if binary.is_file() and os.access(binary, os.X_OK) and manifest.is_file():
        _result(checks, "autogrid", "AutoGrid", PASS, str(binary))
    else:
        _result(checks, "autogrid", "AutoGrid", FAIL, f"설치 필요: {binary}")


def check_data(checks: list[dict], profile: str) -> None:
    if profile == "analog":
        items = {
            "cosing": ROOT / "data" / "cosing" / "cosing.parquet",
            "similarity": ROOT / "data" / "similarity_index_202609" / "manifest.json",
        }
        missing = [name for name, path in items.items() if not path.is_file()]
        if missing:
            _result(checks, "analog_data", "대체소재 데이터", FAIL, "없음: " + ", ".join(missing))
        else:
            _result(checks, "analog_data", "대체소재 데이터", PASS, "CosIng 표 + 유사도 인덱스")
        return

    stage0 = ROOT / "data" / "manifests" / "stage0_complete.flag"
    if stage0.is_file():
        _result(checks, "stage0", "표적 데이터(Stage 0)", PASS, "완료 표식 있음")
    else:
        _result(checks, "stage0", "표적 데이터(Stage 0)", WARN, "아직 없음 - 표적 예측·도킹 불가")

    gate = ROOT / "data" / "manifests" / "activity_retrieval_operational_gate.flag"
    if gate.is_file():
        _result(checks, "gate", "검색 게이트", PASS, "운영 게이트 있음")
    else:
        _result(checks, "gate", "검색 게이트", FAIL, "운영 게이트 없음 - 표적 검색 불가")

    config = ROOT / "workflow" / "config.yaml"
    index_dir = None
    if config.is_file():
        match = re.search(
            r'^\s*daina_recipe_index_dir:\s*["\']?([^"\'\s]+)',
            config.read_text(encoding="utf-8"), re.M,
        )
        if match:
            index_dir = ROOT / match.group(1)
    if index_dir is None:
        _result(checks, "index", "검색 인덱스", WARN, "config에서 인덱스 경로를 읽지 못함")
    elif (index_dir / "manifest.json").is_file():
        _result(checks, "index", "검색 인덱스", PASS, str(index_dir))
    else:
        _result(checks, "index", "검색 인덱스", WARN, f"아직 없음: {index_dir}")


def check_smoke(checks: list[dict], env_python: Path | None, skip: bool) -> None:
    if skip:
        _result(checks, "smoke", "동작 확인", SKIP, "--skip-smoke")
        return
    if env_python is None:
        _result(checks, "smoke", "동작 확인", SKIP, "실행 환경 없음")
        return
    script = ROOT / "scripts" / "explore_target.py"
    if not script.is_file():
        _result(checks, "smoke", "동작 확인", FAIL, f"없음: {script}")
        return
    started = time.monotonic()
    completed = subprocess.run(
        [str(env_python), str(script), "--search", "tyr"],
        capture_output=True, text=True, timeout=300, check=False,
    )
    elapsed = time.monotonic() - started
    if completed.returncode == 0 and "TYR" in completed.stdout:
        _result(checks, "smoke", "동작 확인", PASS, f"표적 이름 검색 {elapsed:.1f}s")
    else:
        _result(checks, "smoke", "동작 확인", FAIL, (completed.stderr or "").strip()[-200:])


def run_checks(profile: str, skip_smoke: bool) -> list[dict]:
    checks: list[dict] = []
    env_python = find_env_python()
    check_os(checks)
    check_resources(checks, profile)
    check_gpu(checks, profile)
    check_runtime(checks, env_python)
    check_tools(checks, profile)
    check_data(checks, profile)
    check_smoke(checks, env_python, skip_smoke)
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", choices=sorted(PROFILE_MIN_DISK_GB), default="demo")
    parser.add_argument("--json", action="store_true", help="기계 판독용 JSON 출력")
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()

    checks = run_checks(args.profile, args.skip_smoke)
    failures = [check for check in checks if check["status"] == FAIL]

    if args.json:
        print(json.dumps({"profile": args.profile, "checks": checks}, ensure_ascii=False, indent=1))
    else:
        icons = {PASS: "OK  ", WARN: "주의", FAIL: "실패", SKIP: "건너뜀"}
        print(f"설치 점검 (프로파일: {args.profile})")
        for check in checks:
            print(f"  [{icons[check['status']]}] {check['label']}: {check['detail']}")
        if failures:
            print(f"\n막는 문제 {len(failures)}개. 위 '실패' 항목을 해결한 뒤 다시 실행하세요.")
        else:
            print("\n막는 문제 없음.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
