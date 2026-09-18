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


LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
WILDCARD_BIND_HOSTS = {"0.0.0.0", "::"}


def _workbench_identity(url: str, timeout: float = 1.0) -> dict | None:
    try:
        with urlopen(f"{url}/api/identity", timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("application") != "SkinScout Workbench"
    ):
        return None
    return payload


def _is_workbench(url: str, timeout: float = 1.0) -> bool:
    return _workbench_identity(url, timeout) is not None


def _requested_policy(
    host: str,
    *,
    require_auth: bool,
    behind_https_proxy: bool,
    allowed_hosts: object = (),
) -> dict[str, object]:
    remote = host.lower() not in LOOPBACK_HOSTS
    requested_hosts = (
        allowed_hosts
        if isinstance(allowed_hosts, (list, tuple, set, frozenset))
        else ()
    )
    return {
        "require_auth": bool(require_auth or remote or behind_https_proxy),
        "behind_https_proxy": bool(behind_https_proxy),
        "bind_host": host,
        "allowed_hosts": frozenset(
            str(item).strip().lower() for item in requested_hosts if str(item).strip()
        ),
    }


def _bind_host_serves(existing: object, requested: object) -> bool:
    if not isinstance(existing, str) or not existing:
        return False
    if not isinstance(requested, str) or not requested:
        return False
    existing = existing.lower()
    requested = requested.lower()
    if existing == requested:
        return True
    if existing in WILDCARD_BIND_HOSTS:
        return True
    return existing in LOOPBACK_HOSTS and requested in LOOPBACK_HOSTS


def _identity_matches_policy(identity: dict, policy: dict[str, object]) -> bool:
    """Reuse only an instance whose published security mode is the requested one.

    A server built before the mode was exposed cannot prove its mode, so it is
    never reused; the launcher starts a separate instance instead.
    """
    require_auth = identity.get("require_auth")
    behind_https_proxy = identity.get("behind_https_proxy")
    if not isinstance(require_auth, bool) or not isinstance(behind_https_proxy, bool):
        return False
    existing_allowed = identity.get("allowed_hosts")
    if not isinstance(existing_allowed, list):
        return False
    requested_allowed = policy.get("allowed_hosts") or frozenset()
    if not isinstance(requested_allowed, frozenset):
        requested_allowed = frozenset(requested_allowed)
    return (
        require_auth == policy["require_auth"]
        and behind_https_proxy == policy["behind_https_proxy"]
        and _bind_host_serves(identity.get("bind_host"), policy["bind_host"])
        and requested_allowed <= {str(item).strip().lower() for item in existing_allowed}
    )


IPV6_UNSUPPORTED_MESSAGE = (
    "IPv6 주소(예: ::1)는 아직 지원하지 않습니다. "
    "--host 127.0.0.1 또는 --host localhost를 사용하세요. "
    "(Workbench 서버가 AF_INET 소켓에 바인딩하고 URL도 대괄호 표기가 필요합니다.)"
)


def _reject_ipv6(host: str) -> None:
    if ":" in host:
        raise SystemExit(IPV6_UNSUPPORTED_MESSAGE)


def _port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _choose_port(
    host: str,
    preferred: int,
    *,
    require_auth: bool = False,
    behind_https_proxy: bool = False,
    allowed_hosts: object = (),
) -> tuple[int, bool]:
    """Return ``(port, already_running)`` for the requested security mode.

    An instance already running with a different mode is never reused and is
    never killed; a separate instance is started on the next free port.
    """
    _reject_ipv6(host)
    if preferred < 1 or preferred > 65535:
        raise ValueError("포트는 1에서 65535 사이여야 합니다.")
    policy = _requested_policy(
        host,
        require_auth=require_auth,
        behind_https_proxy=behind_https_proxy,
        allowed_hosts=allowed_hosts,
    )
    candidates = (preferred, *range(preferred + 1, min(preferred + 21, 65536)))
    for port in candidates:
        identity = _workbench_identity(f"http://{host}:{port}")
        if identity is not None:
            if _identity_matches_policy(identity, policy):
                return port, True
            # Wrong security mode: leave it running and keep looking.
            continue
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
    parser.add_argument(
        "--behind-https-proxy",
        action="store_true",
        help="HTTPS reverse proxy가 외부 TLS를 종료할 때만 지정합니다.",
    )
    parser.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        metavar="HOST",
        help="reverse proxy가 전달하는 공개 Host 이름. 서버 정책에 그대로 전달됩니다.",
    )
    args = parser.parse_args(argv)
    _reject_ipv6(args.host)

    policy = _requested_policy(
        args.host,
        require_auth=args.require_auth,
        behind_https_proxy=args.behind_https_proxy,
        allowed_hosts=args.allowed_host,
    )
    port, already_running = _choose_port(
        args.host,
        args.port,
        require_auth=args.require_auth,
        behind_https_proxy=args.behind_https_proxy,
        allowed_hosts=args.allowed_host,
    )
    url = f"http://{args.host}:{port}"
    if args.dry_run:
        print(f"SkinScout Workbench 준비: {url}")
        return 0

    if not already_running:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOG_DIR / "workbench.log"
        pid_path = LOG_DIR / "workbench.pid"
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_workbench.py"),
            "--host",
            args.host,
            "--port",
            str(port),
        ]
        if args.require_auth:
            command.append("--require-auth")
        if args.behind_https_proxy:
            command.append("--behind-https-proxy")
        for allowed_host in args.allowed_host:
            command.extend(["--allowed-host", allowed_host])
        with _open_private_log(log_path) as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
            )
        _write_private_text(pid_path, f"{process.pid}\n")
        if policy["require_auth"]:
            print(
                "로그인이 필요한 모드로 시작했습니다. 접속 토큰은 Workbench가 표시한 "
                "소유자 전용 token 파일에서 확인하세요. 서버 로그:\n"
                f"  {log_path}",
                flush=True,
            )
        for _ in range(60):
            identity = _workbench_identity(url)
            if identity is not None and _identity_matches_policy(identity, policy):
                break
            if identity is not None and not _identity_matches_policy(identity, policy):
                process.terminate()
                raise RuntimeError(
                    "Workbench가 요청한 보안 모드와 다르게 시작되었습니다. "
                    f"로그: {log_path}"
                )
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
