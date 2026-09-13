#!/usr/bin/env python3
"""Start SkinScout Workbench in the background and open it in a browser."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "results" / "logs"


def _open_private_log(path: Path):
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    os.fchmod(descriptor, 0o600)
    return os.fdopen(descriptor, "a", encoding="utf-8")


def _write_private_text(path: Path, text: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        os.write(descriptor, text.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _is_workbench(url: str, timeout: float = 1.0) -> bool:
    try:
        with urlopen(f"{url}/api/identity", timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return (
        isinstance(payload, dict)
        and payload.get("application") == "SkinScout Workbench"
    )


def _port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _choose_port(host: str, preferred: int) -> tuple[int, bool]:
    if preferred < 1 or preferred > 65535:
        raise ValueError("포트는 1에서 65535 사이여야 합니다.")
    preferred_url = f"http://{host}:{preferred}"
    if _is_workbench(preferred_url):
        return preferred, True
    if _port_available(host, preferred):
        return preferred, False
    for port in range(preferred + 1, min(preferred + 21, 65536)):
        if _port_available(host, port):
            return port, False
    raise RuntimeError("사용 가능한 로컬 포트를 찾지 못했습니다.")


def _open_browser(url: str) -> None:
    if webbrowser.open(url, new=1):
        return
    opener = next(
        (path for name in ("xdg-open", "wslview") if (path := _which(name))),
        None,
    )
    if opener:
        subprocess.Popen(
            [opener, url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )


def _which(name: str) -> str | None:
    from shutil import which

    return which(name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--require-auth",
        action="store_true",
        help="로그인을 요구합니다. 로컬 주소가 아닌 곳에 열 때는 자동으로 켜집니다.",
    )
    args = parser.parse_args(argv)

    port, already_running = _choose_port(args.host, args.port)
    url = f"http://{args.host}:{port}"
    if args.dry_run:
        print(f"SkinScout Workbench 준비: {url}")
        return 0

    if not already_running:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOG_DIR / "workbench.log"
        pid_path = LOG_DIR / "workbench.pid"
        with _open_private_log(log_path) as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_workbench.py"),
                    "--host",
                    args.host,
                    "--port",
                    str(port),
                    *(["--require-auth"] if args.require_auth else []),
                ],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
            )
        _write_private_text(pid_path, f"{process.pid}\n")
        if args.host not in {"127.0.0.1", "localhost", "::1"} or args.require_auth:
            print(
                "로그인이 필요한 모드로 시작했습니다. 접속 토큰은 Workbench가 표시한 "
                "소유자 전용 token 파일에서 확인하세요. 서버 로그:\n"
                f"  {log_path}",
                flush=True,
            )
        for _ in range(60):
            if _is_workbench(url):
                break
            if process.poll() is not None:
                raise RuntimeError(f"Workbench 시작에 실패했습니다. 로그: {log_path}")
            time.sleep(0.25)
        else:
            process.terminate()
            raise RuntimeError(f"Workbench가 15초 안에 시작되지 않았습니다. 로그: {log_path}")

    print(f"SkinScout Workbench: {url}")
    if not args.no_browser:
        _open_browser(url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
