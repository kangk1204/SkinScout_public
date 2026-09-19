#!/usr/bin/env python3
"""Serve the local SkinScout Workbench with Python's standard library only."""

from __future__ import annotations

import argparse
import atexit
import csv
import hashlib
import io
import json
import mimetypes
import os
import platform
import re
import secrets
import signal
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlsplit

from skinscout.contracts.run_profile import resolve_run_profile
from workbench.coordinator import (
    DEFAULT_GPU_RESOURCE,
    HEARTBEAT_SECONDS,
    AuthError,
    CoordinatorError,
    DurableCoordinator,
    InvalidTransition,
    LeaseUnavailable,
)


ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "eval"
SCRIPTS_DIR = ROOT / "scripts"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import stage0_verify  # noqa: E402
import gpu_admission  # noqa: E402
from target_failure_modes import failure_modes_for, load_curated  # noqa: E402
from compound_applicability import (  # noqa: E402
    assess as assess_applicability,
    assess_sdf as assess_sdf_applicability,
    refusal_message as applicability_refusal,
)

_FAILURE_MODE_CACHE: dict[str, Any] | None = None
_FAILURE_MODE_CACHE_LOCK = threading.Lock()


def _failure_mode_state() -> dict[str, Any]:
    """Curated receptor limitations and whether that warning source loaded.

    A missing or malformed CSV used to be cached as an empty mapping, so a
    deployment that never had the file looked exactly like a run with zero
    documented limitations. An unavailable source is deliberately not reused:
    the next request retries it, so the file can appear after startup.
    """
    global _FAILURE_MODE_CACHE
    path = ROOT / "data" / "validation" / "known_failure_modes.csv"
    try:
        mtime = path.stat().st_mtime if path.exists() else None
    except OSError:
        mtime = None
    with _FAILURE_MODE_CACHE_LOCK:
        cached = _FAILURE_MODE_CACHE
        if (
            cached is not None
            and cached.get("status") == "ok"
            and cached.get("mtime") == mtime
        ):
            return cached
        try:
            records = load_curated(path)
        except (SystemExit, OSError) as exc:
            state = {
                "status": "unavailable",
                "records": {},
                "error": str(exc) or "failure-mode CSV를 읽지 못했습니다.",
                "path": path,
                "mtime": mtime,
            }
        else:
            state = {
                "status": "ok",
                "records": records,
                "error": None,
                "path": path,
                "mtime": mtime,
            }
        _FAILURE_MODE_CACHE = state
        return state


def _failure_mode_records() -> dict[str, dict[str, str]]:
    """Backwards-compatible accessor for the curated records only."""
    return _failure_mode_state()["records"]


def _warnings_payload(state: dict[str, Any] | None = None) -> dict[str, Any]:
    """Availability of the curated warning list, separate from its contents."""
    state = state if state is not None else _failure_mode_state()
    return {
        "status": state["status"],
        "available": state["status"] == "ok",
        "error": state["error"],
        "path": _relative_path(state["path"]),
    }

STATIC_DIR = ROOT / "workbench" / "static"
RUNS_DIR = ROOT / "results" / "runs"
LOG_DIR = RUNS_DIR / ".workbench_logs"
UPLOAD_DIR = RUNS_DIR / ".workbench_inputs"
DEFAULT_STATE_DIR = Path(
    os.environ.get(
        "SKINSCOUT_STATE_DIR",
        Path.home() / ".local" / "share" / "skinscout" / "state",
    )
)
DEFAULT_SECRETS_DIR = Path(
    os.environ.get(
        "SKINSCOUT_SECRETS_DIR",
        DEFAULT_STATE_DIR.parent / "secrets",
    )
)
WORKER_API_TOKEN_PATH = DEFAULT_SECRETS_DIR / "worker-api.token"
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
UNIPROT_ACCESSION_RE = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|"
    r"[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})(?:-[0-9]+)?$"
)
MAX_BODY_BYTES = 12 * 1024 * 1024
MAX_SDF_BYTES = 10 * 1024 * 1024
# Longer than any cosmetic ingredient; a paste this size is a mistake, not input.
MAX_SMILES_LENGTH = 4096
MAX_SUBSTITUTE_REPORT_BYTES = 24 * 1024 * 1024
MAX_SUBSTITUTE_MANIFEST_BYTES = 2 * 1024 * 1024
BASE_RUNTIME_MIN_FREE_GB = 80
STAGE0_MIN_FREE_GB = 200
GPU_ANALYSIS_MIN_FREE_MIB = gpu_admission.MIN_FREE_MIB
GPU_ANALYSIS_MAX_UTILIZATION_PERCENT = gpu_admission.MAX_UTILIZATION_PERCENT
STATUS_CACHE_SECONDS = 15.0
_CSRF_TOKEN = secrets.token_urlsafe(32)
_CSRF_TOKEN_GENERATED_AT = time.monotonic()
_CSRF_TOKEN_LIFETIME = 3600.0  # rotate every hour


def _csrf_token() -> str:
    """Return CSRF token, rotating periodically for long-lived servers."""
    global _CSRF_TOKEN, _CSRF_TOKEN_GENERATED_AT
    now = time.monotonic()
    if now - _CSRF_TOKEN_GENERATED_AT > _CSRF_TOKEN_LIFETIME:
        _CSRF_TOKEN = secrets.token_urlsafe(32)
        _CSRF_TOKEN_GENERATED_AT = now
    return _CSRF_TOKEN
SETUP_PROFILES = {
    "demo": {
        "label": "Demo",
        "description": "기본 Workbench, 안전성 분석, 빠른 Target 데모 실행 환경",
        "default": True,
    },
    "full": {
        "label": "Full",
        "description": "Demo 환경에 고급 Target/보고서 모델 준비를 추가",
        "default": False,
    },
}
TARGET_DEMO_REQUIREMENTS = (
    "gpu",
    "gnina",
    "autodock_gpu",
    "autogrid",
    "autodock_gpu_env",
    "meeko_env",
)
TARGET_ADVANCED_REQUIREMENTS = (
    *TARGET_DEMO_REQUIREMENTS,
    "rtmscore",
    "psichic",
)
TARGET_FAST_REQUIREMENTS = TARGET_DEMO_REQUIREMENTS
# DiffDock is a hard input of the comprehensive DAG (autodock_pick_top_pct
# consumes diffdock_blind_no_pocket's scores), so leaving it out of readiness
# let the Workbench show green and then fail mid-run.
TARGET_COMPREHENSIVE_REQUIREMENTS = (*TARGET_ADVANCED_REQUIREMENTS, "boltz", "diffdock")
REPORT_REQUIREMENTS = (
    *TARGET_COMPREHENSIVE_REQUIREMENTS,
    "bioemu_env",
    "md_env",
    "qm_env",
)
MODEL_REQUIREMENT_LABELS = {
    "gpu": "CUDA GPU",
    "gnina": "GNINA",
    "rtmscore": "RTMScore",
    "psichic": "PSICHIC",
    "autodock_gpu": "AutoDock-GPU",
    "autogrid": "AutoGrid4 + AD4 파라미터",
    "autodock_gpu_env": "AutoGrid/AutoDock 지원 환경",
    "meeko_env": "Meeko 환경",
    "boltz": "Boltz-2",
    "bioemu_env": "BioEmu 환경",
    "md_env": "MD 환경",
    "qm_env": "QM 환경",
}
SUBSTITUTE_SCHEMA = "skinscout.substitute_discovery.v2"
SUBSTITUTE_RUN_SCHEMA = "skinscout.substitute_run.v1"
SUBSTITUTE_SELECTION_TRACKS = {
    "target_activity",
    "cosmetic_material",
    "feature_proxy",
}
SUBSTITUTE_RELATIVE_DIR = Path("05_6_substitutes")
SUBSTITUTE_ARTIFACTS = (
    ("substitute_report", "substitute_report.json"),
    ("substitute_candidates", "substitute_candidates.csv"),
    ("substitute_candidates_3d", "substitute_candidates_3d.sdf"),
    ("substitute_interactive_report", "substitute_report.html"),
    ("substitute_report_markdown", "substitute_report.md"),
    ("substitute_run_manifest", "substitute_run_manifest.json"),
)
SUBSTITUTE_OUTPUT_FILES = tuple(
    filename
    for label, filename in SUBSTITUTE_ARTIFACTS
    if label != "substitute_run_manifest"
)
# 대체소재 검색(Workbench "대체소재 검색" 화면)이 실제로 여는 두 가지.
ALTERNATIVES_INPUTS = (
    ("CosIng 원료 표", ROOT / "data" / "cosing" / "cosing.parquet"),
    ("활성 측정 라이브러리", ROOT / "data" / "similarity_index_202609" / "manifest.json"),
)

SUBSTITUTE_ACTIVITY_EVIDENCE = (
    ("ChEMBL activity evidence", ROOT / "data" / "chembl37" / "activity_evidence.parquet"),
    (
        "BindingDB activity evidence",
        ROOT / "data" / "bindingdb" / "evidence_v1" / "activity_evidence.parquet",
    ),
    (
        "GtoPdb activity evidence",
        ROOT / "data" / "gtopdb" / "evidence_v1" / "activity_evidence.parquet",
    ),
)


@dataclass
class Job:
    job_id: str
    kind: str
    run_id: str | None
    status: str = "queued"
    started_at: str | None = None
    ended_at: str | None = None
    returncode: int | None = None
    detail: str | None = None
    command: list[str] = field(default_factory=list)
    log_path: Path | None = None
    process: subprocess.Popen[str] | None = field(default=None, repr=False)
    attempt_id: int | None = field(default=None, repr=False)
    attempt_token: str | None = field(default=None, repr=False)
    lease_resource: str | None = field(default=None, repr=False)
    cleanup_paths: tuple[Path, ...] = field(default_factory=tuple, repr=False)

    def public(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "run_id": self.run_id,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "returncode": self.returncode,
            "detail": self.detail,
            "command": self.command,
            "log_path": _relative_path(self.log_path) if self.log_path else None,
            "log_tail": _tail(self.log_path),
        }


_JOBS: dict[str, Job] = {}
_JOBS_LOCK = threading.Lock()
_COORDINATOR: DurableCoordinator | None = None
_COORDINATOR_LOCK = threading.Lock()
_WORKER_API_TOKEN: str | None = None
_WORKER_API_TOKEN_FINGERPRINT: tuple[int, int, int, int] | None = None
_WORKER_API_TOKEN_LOCK = threading.Lock()
_STATUS_CACHE: tuple[float, dict[str, Any]] | None = None
_METADATA_CACHE: tuple[float, dict[str, dict[str, str]]] | None = None
_STATUS_CACHE_LOCK = threading.Lock()
_METADATA_CACHE_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _coordinator() -> DurableCoordinator:
    global _COORDINATOR
    with _COORDINATOR_LOCK:
        if _COORDINATOR is None:
            coordinator = DurableCoordinator(
                DEFAULT_STATE_DIR / "coordinator.sqlite3",
                artifact_root=DEFAULT_STATE_DIR.parent / "results",
            )
            try:
                _reconcile_interrupted_processes(coordinator)
                coordinator.recover()
            except BaseException:
                coordinator.close()
                raise
            _COORDINATOR = coordinator
        return _COORDINATOR


def _close_coordinator() -> None:
    global _COORDINATOR
    with _COORDINATOR_LOCK:
        if _COORDINATOR is not None:
            _COORDINATOR.close()
            _COORDINATOR = None


atexit.register(_close_coordinator)


def _process_stat(pid: int) -> tuple[str, str] | None:
    if os.name != "posix":
        return None
    try:
        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        fields = stat_text[stat_text.rfind(")") + 2 :].split()
        return fields[0], fields[19]
    except (OSError, IndexError):
        return None


def _process_birth_id(pid: int) -> str | None:
    process_stat = _process_stat(pid)
    return process_stat[1] if process_stat is not None else None


def _process_identity(process: subprocess.Popen[str]) -> dict[str, Any]:
    pid = int(process.pid)
    return {
        "pid": pid,
        # Every managed child is created with start_new_session=True, so its
        # process-group identifier is its PID even if it exits immediately.
        "pgid": pid,
        "birth_id": _process_birth_id(pid),
        "owner_pid": os.getpid(),
    }


def _identity_is_live(identity: dict[str, Any]) -> bool:
    try:
        pid = int(identity["pid"])
    except (KeyError, TypeError, ValueError):
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    process_stat = _process_stat(pid)
    if process_stat is not None and process_stat[0] == "Z":
        return False
    expected_birth = identity.get("birth_id")
    return expected_birth is None or (
        process_stat is not None and process_stat[1] == str(expected_birth)
    )


def _signal_process_identity(identity: dict[str, Any], sig: signal.Signals) -> None:
    if not _identity_is_live(identity):
        return
    pid = int(identity["pid"])
    try:
        if os.name == "posix":
            os.killpg(int(identity.get("pgid", pid)), sig)
        else:
            os.kill(pid, sig)
    except OSError:
        pass


def _stop_process_identity(identity: dict[str, Any], timeout: float = 8.0) -> bool:
    _signal_process_identity(identity, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while _identity_is_live(identity) and time.monotonic() < deadline:
        time.sleep(0.05)
    if _identity_is_live(identity):
        _signal_process_identity(identity, signal.SIGKILL)
        deadline = time.monotonic() + 2.0
        while _identity_is_live(identity) and time.monotonic() < deadline:
            time.sleep(0.05)
    return not _identity_is_live(identity)


def _reconcile_interrupted_processes(coordinator: DurableCoordinator) -> None:
    for record in coordinator.list_jobs(
        statuses={"running", "cancel_requested"},
        limit=1_000,
    ):
        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue
        local_managed = payload.get("managed_by") == "workbench-local-process" or (
            isinstance(payload.get("command"), list)
            and isinstance(payload.get("log_path"), str)
        )
        if not local_managed:
            continue
        identity = payload.get("process_identity")
        if not isinstance(identity, dict):
            coordinator.recover_interrupted_job(
                str(record["job_id"]),
                reason="missing_process_identity_after_restart",
                requeue=False,
            )
            continue
        owner_pid = identity.get("owner_pid")
        if (
            isinstance(owner_pid, int)
            and owner_pid != os.getpid()
            and _process_stat(owner_pid) is not None
        ):
            # 다른 워크벤치 프로세스가 아직 살아서 소유한 작업이다. 재시작 복구
            # 대상이 아니므로 건드리지 않는다(같은 state dir를 쓰는 두 서버가
            # 서로의 장시간 작업을 죽이던 문제).
            continue
        stopped = _stop_process_identity(identity)
        coordinator.recover_interrupted_job(
            str(record["job_id"]),
            reason="previous_workbench_process_stopped" if stopped else "orphan_process_could_not_stop",
            requeue=stopped and record["status"] != "cancel_requested",
        )


def _worker_api_token() -> str:
    global _WORKER_API_TOKEN, _WORKER_API_TOKEN_FINGERPRINT
    rotated = False
    with _WORKER_API_TOKEN_LOCK:
        path = WORKER_API_TOKEN_PATH
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            file_info = path.lstat()
        except FileNotFoundError:
            token = secrets.token_urlsafe(32)
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(path, flags, 0o600)
            try:
                os.write(descriptor, (token + "\n").encode("ascii"))
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            file_info = path.lstat()
        else:
            if path.is_symlink() or not path.is_file():
                raise RuntimeError("Worker API token path must be a regular file")
            fingerprint = (
                file_info.st_dev,
                file_info.st_ino,
                file_info.st_mtime_ns,
                file_info.st_size,
            )
            if (
                _WORKER_API_TOKEN is not None
                and _WORKER_API_TOKEN_FINGERPRINT == fingerprint
            ):
                return _WORKER_API_TOKEN
            flags = os.O_RDONLY
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(path, flags)
            try:
                opened_info = os.fstat(descriptor)
                if not stat.S_ISREG(opened_info.st_mode):
                    raise RuntimeError("Worker API token path must be a regular file")
                token = os.read(descriptor, 4096).decode("ascii").strip()
                file_info = opened_info
            finally:
                os.close(descriptor)
        if file_info.st_mode & 0o777 != 0o600:
            raise RuntimeError("Worker API token file permissions must be 0600")
        if len(token.encode("ascii")) < 32:
            raise RuntimeError("Worker API token is invalid")
        previous = _WORKER_API_TOKEN
        _WORKER_API_TOKEN = token
        _WORKER_API_TOKEN_FINGERPRINT = (
            file_info.st_dev,
            file_info.st_ino,
            file_info.st_mtime_ns,
            file_info.st_size,
        )
        rotated = previous is not None and not secrets.compare_digest(previous, token)
    if rotated and _COORDINATOR is not None:
        _COORDINATOR.invalidate_active_attempts("worker_secret_rotated")
    return token


ACCESS_TOKEN_PATH = DEFAULT_SECRETS_DIR / "workbench-access.token"
SESSION_COOKIE = "skinscout_session"
SESSION_TTL_SECONDS = 12 * 60 * 60
_ACCESS_TOKEN_LOCK = threading.Lock()
_SESSIONS: dict[str, float] = {}
_SESSIONS_LOCK = threading.Lock()
# Off by default so the local single-user flow is unchanged; turned on
# automatically for any non-loopback bind, which is the only way to reach the
# Workbench from another machine without an SSH tunnel.
REQUIRE_AUTH = False
SECURE_SESSION_COOKIE = False


def _read_or_create_secret(path: Path) -> str:
    """A 0600, symlink-refusing secret file, created on first use."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        file_info = path.lstat()
    except FileNotFoundError:
        token = secrets.token_urlsafe(32)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o600)
        try:
            os.write(descriptor, (token + "\n").encode("ascii"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return token
    if path.is_symlink() or not stat.S_ISREG(file_info.st_mode):
        raise RuntimeError(f"{path} must be a regular file")
    if file_info.st_mode & 0o777 != 0o600:
        raise RuntimeError(f"{path} permissions must be 0600")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        token = os.read(descriptor, 4096).decode("ascii").strip()
    finally:
        os.close(descriptor)
    if len(token.encode("ascii")) < 32:
        raise RuntimeError(f"{path} does not hold a usable token")
    return token


def access_token() -> str:
    with _ACCESS_TOKEN_LOCK:
        return _read_or_create_secret(ACCESS_TOKEN_PATH)


def _prune_sessions(now: float) -> None:
    for session_id, expiry in list(_SESSIONS.items()):
        if expiry <= now:
            _SESSIONS.pop(session_id, None)


def open_session(token: str) -> str:
    """Exchange the access token for a session id, or raise."""
    expected = access_token()
    if not isinstance(token, str) or not secrets.compare_digest(token.strip(), expected):
        raise PermissionError("접속 토큰이 올바르지 않습니다.")
    session_id = secrets.token_urlsafe(32)
    now = time.time()
    with _SESSIONS_LOCK:
        _prune_sessions(now)
        _SESSIONS[session_id] = now + SESSION_TTL_SECONDS
    return session_id


def close_session(session_id: str | None) -> None:
    if not session_id:
        return
    with _SESSIONS_LOCK:
        _SESSIONS.pop(session_id, None)


def session_is_valid(session_id: str | None) -> bool:
    if not session_id:
        return False
    now = time.time()
    with _SESSIONS_LOCK:
        _prune_sessions(now)
        return session_id in _SESSIONS


def _worker_authorized(authorization: str | None) -> bool:
    prefix = "Bearer "
    candidate = (
        authorization.removeprefix(prefix)
        if isinstance(authorization, str) and authorization.startswith(prefix)
        else ""
    )
    return secrets.compare_digest(candidate, _worker_api_token())


def _relative_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return None


def _json_value(value: Any) -> Any:
    if isinstance(value, float) and (value != value or value in {float("inf"), float("-inf")}):
        return None
    if isinstance(value, Path):
        return _relative_path(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return value


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _fingerprint_matches(
    record: object,
    path: Path,
    *,
    root: Path,
) -> bool:
    if not isinstance(record, dict) or path.is_symlink():
        return False
    try:
        path.resolve().relative_to(root.resolve())
        file_stat = path.stat()
    except (OSError, ValueError):
        return False
    expected_bytes = record.get("bytes")
    expected_sha256 = record.get("sha256")
    if (
        not path.is_file()
        or file_stat.st_size <= 0
        or isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes != file_stat.st_size
        or not isinstance(expected_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
    ):
        return False
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return False
    return secrets.compare_digest(digest.hexdigest(), expected_sha256)


def _substitute_report(run_id: str) -> dict[str, Any] | None:
    run_dir = RUNS_DIR / run_id
    substitute_dir = run_dir / SUBSTITUTE_RELATIVE_DIR
    path = substitute_dir / "substitute_report.json"
    if not path.exists():
        return None
    manifest_path = substitute_dir / "substitute_run_manifest.json"
    try:
        if (
            not manifest_path.is_file()
            or manifest_path.is_symlink()
            or manifest_path.stat().st_size <= 0
            or manifest_path.stat().st_size > MAX_SUBSTITUTE_MANIFEST_BYTES
        ):
            raise ValueError("hash-bound run manifest is missing or invalid")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("run manifest must be a JSON object")
        if manifest.get("schema_version") != SUBSTITUTE_RUN_SCHEMA:
            raise ValueError("run manifest schema_version is invalid")
        if manifest.get("run_id") != run_id:
            raise ValueError("run manifest run_id does not match")
        if (
            manifest.get("claimable") is not False
            or manifest.get("hypothesis_only") is not True
            or manifest.get("wet_lab_required") is not True
        ):
            raise ValueError("run manifest claim boundary is invalid")
        outputs = manifest.get("outputs")
        if not isinstance(outputs, dict) or set(outputs) != set(SUBSTITUTE_OUTPUT_FILES):
            raise ValueError("run manifest output set is invalid")
        for filename in SUBSTITUTE_OUTPUT_FILES:
            if not _fingerprint_matches(
                outputs.get(filename),
                substitute_dir / filename,
                root=run_dir,
            ):
                raise ValueError(f"artifact fingerprint mismatch: {filename}")
        parent_run = manifest.get("parent_run")
        if (
            not isinstance(parent_run, dict)
            or parent_run.get("preset") != "target-id"
            or not _fingerprint_matches(
                parent_run.get("summary"),
                run_dir / "run_summary.json",
                root=run_dir,
            )
            or not _fingerprint_matches(
                parent_run.get("verification"),
                run_dir / "run_verification.json",
                root=run_dir,
            )
        ):
            raise ValueError("parent run fingerprints are invalid")
        parent_summary = _read_json(run_dir / "run_summary.json")
        parent_verification = _read_json(run_dir / "run_verification.json")
        if (
            not isinstance(parent_summary, dict)
            or parent_summary.get("run_id") != run_id
            or parent_summary.get("preset") != "target-id"
            or not isinstance(parent_verification, dict)
            or parent_verification.get("status") != "ok"
            or parent_verification.get("preset") != "target-id"
        ):
            raise ValueError("parent target run is not verified")
        report_size = path.stat().st_size
        if not path.is_file() or report_size <= 0:
            raise ValueError("report is empty or not a file")
        if report_size > MAX_SUBSTITUTE_REPORT_BYTES:
            raise ValueError("report exceeds the Workbench size limit")
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return {
            "status": "blocked",
            "error": f"대체소재 보고서를 검증하지 못했습니다: {exc}",
            "candidates": [],
        }
    if not isinstance(payload, dict):
        return {
            "status": "blocked",
            "error": "대체소재 보고서는 JSON object여야 합니다.",
            "candidates": [],
        }
    errors: list[str] = []
    if payload.get("schema_version") != SUBSTITUTE_SCHEMA:
        errors.append("schema_version")
    if payload.get("run_id") != run_id:
        errors.append("run_id")
    if payload.get("status") not in {"completed", "completed_no_candidates"}:
        errors.append("status")
    if (
        payload.get("claimable") is not False
        or payload.get("hypothesis_only") is not True
        or payload.get("wet_lab_required") is not True
    ):
        errors.append("claim boundary")
    target = payload.get("target")
    target_id = target.get("target_id") if isinstance(target, dict) else None
    if not isinstance(target_id, str) or not UNIPROT_ACCESSION_RE.fullmatch(target_id):
        errors.append("target_id")
    anchor_conditioned = (
        target.get("interaction_anchor_conditioned", False)
        if isinstance(target, dict)
        else False
    )
    if not isinstance(anchor_conditioned, bool):
        errors.append("target.interaction_anchor_conditioned")
        anchor_conditioned = False
    manifest_target = manifest.get("target")
    if (
        not isinstance(manifest_target, dict)
        or manifest_target.get("target_id") != target_id
    ):
        errors.append("manifest target")
    parent = payload.get("parent")
    if (
        not isinstance(parent, dict)
        or not isinstance(parent.get("smiles"), str)
        or not parent.get("smiles")
    ):
        errors.append("parent")
    parent_compound = parent_summary.get("compound")
    parent_summary_smiles = (
        parent_compound.get("canonical_smiles")
        if isinstance(parent_compound, dict)
        else None
    )
    parent_report_smiles = parent.get("smiles") if isinstance(parent, dict) else None
    if (
        not isinstance(parent_summary_smiles, str)
        or not parent_summary_smiles
        or parent_run.get("canonical_smiles") != parent_summary_smiles
        or parent_report_smiles != parent_summary_smiles
    ):
        errors.append("parent identity")
    anchor_fingerprint = manifest.get("interaction_anchor_map")
    if anchor_conditioned:
        if not _fingerprint_matches(
            anchor_fingerprint,
            run_dir / "05_pharmacophore" / "interaction_anchor_map.json",
            root=run_dir,
        ):
            errors.append("interaction anchor fingerprint")
    elif anchor_fingerprint is not None:
        errors.append("unexpected interaction anchor fingerprint")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or len(candidates) > 500:
        errors.append("candidates")
        candidates = []
    thresholds = payload.get("thresholds")
    raw_max_potency_loss = (
        thresholds.get("max_direct_pactivity_loss_for_retention")
        if isinstance(thresholds, dict)
        else None
    )
    max_potency_loss = _as_float(raw_max_potency_loss)
    if (
        isinstance(raw_max_potency_loss, bool)
        or max_potency_loss is None
        or max_potency_loss < 0.0
    ):
        errors.append("thresholds.max_direct_pactivity_loss_for_retention")
    raw_min_pharmacophore = (
        thresholds.get("min_pharmacophore_preservation_score")
        if isinstance(thresholds, dict)
        else None
    )
    raw_min_feature_recall = (
        thresholds.get("min_feature_family_recall")
        if isinstance(thresholds, dict)
        else None
    )
    min_pharmacophore = _as_float(raw_min_pharmacophore)
    min_feature_recall = _as_float(raw_min_feature_recall)
    if (
        isinstance(raw_min_pharmacophore, bool)
        or min_pharmacophore is None
        or not 0.0 <= min_pharmacophore <= 1.0
    ):
        errors.append("thresholds.min_pharmacophore_preservation_score")
    if (
        isinstance(raw_min_feature_recall, bool)
        or min_feature_recall is None
        or not 0.0 <= min_feature_recall <= 1.0
    ):
        errors.append("thresholds.min_feature_family_recall")
    seen_ids: set[str] = set()
    seen_smiles: set[str] = set()
    seen_global_ranks: set[int] = set()
    previous_priority: float | None = None
    previous_global_rank = 0
    allowed_tiers = {"direct_retained", "direct_activity", "direct_reduced", "proxy_only"}
    allowed_admission_bases = {
        "strict_2d_pharmacophore",
        "same_target_activity_feature_family",
    }
    for expected_rank, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict):
            errors.append(f"candidate[{expected_rank}]")
            break
        candidate_id = candidate.get("candidate_id")
        smiles = candidate.get("smiles")
        if candidate.get("rank") != expected_rank:
            errors.append(f"candidate[{expected_rank}].rank")
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in seen_ids:
            errors.append(f"candidate[{expected_rank}].candidate_id")
        if not isinstance(smiles, str) or not smiles or smiles in seen_smiles:
            errors.append(f"candidate[{expected_rank}].smiles")
        if candidate.get("evidence_tier") not in allowed_tiers:
            errors.append(f"candidate[{expected_rank}].evidence_tier")
        admission_bases = candidate.get("admission_bases")
        pharmacophore_gate_passed = candidate.get("pharmacophore_gate_passed")
        if (
            not isinstance(admission_bases, list)
            or not admission_bases
            or len(admission_bases) != len(set(admission_bases))
            or any(value not in allowed_admission_bases for value in admission_bases)
            or not isinstance(pharmacophore_gate_passed, bool)
            or pharmacophore_gate_passed
            != ("strict_2d_pharmacophore" in admission_bases)
            or (
                "same_target_activity_feature_family" in admission_bases
                and candidate.get("evidence_tier") == "proxy_only"
            )
            or (
                not pharmacophore_gate_passed
                and "same_target_activity_feature_family" not in admission_bases
            )
        ):
            errors.append(f"candidate[{expected_rank}].admission_bases")
        if (
            candidate.get("claimable") is not False
            or candidate.get("hypothesis_only") is not True
            or candidate.get("wet_lab_required") is not True
        ):
            errors.append(f"candidate[{expected_rank}].claim boundary")
        if not isinstance(candidate.get("cosing_reference"), bool):
            errors.append(f"candidate[{expected_rank}].cosing_reference")
        global_rank = candidate.get("global_priority_rank")
        if (
            isinstance(global_rank, bool)
            or not isinstance(global_rank, int)
            or global_rank < 1
            or global_rank in seen_global_ranks
            or global_rank <= previous_global_rank
        ):
            errors.append(f"candidate[{expected_rank}].global_priority_rank")
        tracks = candidate.get("selection_tracks")
        expected_tracks: list[str] = []
        if candidate.get("evidence_tier") != "proxy_only":
            expected_tracks.append("target_activity")
        if candidate.get("cosing_reference") is True:
            expected_tracks.append("cosmetic_material")
        if not expected_tracks:
            expected_tracks.append("feature_proxy")
        if (
            not isinstance(tracks, list)
            or tracks != expected_tracks
            or any(track not in SUBSTITUTE_SELECTION_TRACKS for track in tracks)
        ):
            errors.append(f"candidate[{expected_rank}].selection_tracks")
        sources = candidate.get("candidate_sources")
        if (
            not isinstance(sources, list)
            or any(not isinstance(source, str) or not source for source in sources)
        ):
            errors.append(f"candidate[{expected_rank}].candidate_sources")
        activity_count = candidate.get("activity_evidence_count")
        if (
            isinstance(activity_count, bool)
            or not isinstance(activity_count, int)
            or activity_count < 0
        ):
            errors.append(f"candidate[{expected_rank}].activity_evidence_count")
        tier = candidate.get("evidence_tier")
        retained = candidate.get("binding_retained")
        if tier == "direct_retained" and retained is not True:
            errors.append(f"candidate[{expected_rank}].binding_retained")
        elif tier == "direct_reduced" and retained is not False:
            errors.append(f"candidate[{expected_rank}].binding_retained")
        elif tier in {"direct_activity", "proxy_only"} and retained is not None:
            errors.append(f"candidate[{expected_rank}].binding_retained")
        if tier == "proxy_only" and activity_count != 0:
            errors.append(f"candidate[{expected_rank}].activity_evidence_count")
        if tier in {"direct_retained", "direct_activity", "direct_reduced"} and (
            not isinstance(activity_count, int) or activity_count < 1
        ):
            errors.append(f"candidate[{expected_rank}].activity_evidence_count")
        comparison_basis = candidate.get("binding_comparison_basis")
        comparison_strata = candidate.get("binding_comparison_strata")
        raw_binding_delta = candidate.get("binding_pactivity_delta")
        binding_delta = _as_float(raw_binding_delta)
        if tier in {"direct_retained", "direct_reduced"}:
            valid_stratum_deltas: list[float] = []
            if (
                comparison_basis
                != "same_activity_type_and_source_conservative_delta"
                or not isinstance(comparison_strata, list)
                or not comparison_strata
                or isinstance(raw_binding_delta, bool)
                or binding_delta is None
            ):
                errors.append(f"candidate[{expected_rank}].binding_comparison")
            else:
                for stratum in comparison_strata:
                    if not isinstance(stratum, dict):
                        valid_stratum_deltas = []
                        break
                    candidate_value_raw = stratum.get("candidate_median_pactivity")
                    parent_value_raw = stratum.get("parent_median_pactivity")
                    delta_raw = stratum.get("pactivity_delta")
                    candidate_value = _as_float(candidate_value_raw)
                    parent_value = _as_float(parent_value_raw)
                    stratum_delta = _as_float(delta_raw)
                    candidate_count = stratum.get("candidate_evidence_count")
                    parent_count = stratum.get("parent_evidence_count")
                    if (
                        not isinstance(stratum.get("activity_type"), str)
                        or not stratum.get("activity_type")
                        or not isinstance(stratum.get("source"), str)
                        or not stratum.get("source")
                        or any(
                            isinstance(value, bool)
                            for value in (
                                candidate_value_raw,
                                parent_value_raw,
                                delta_raw,
                                candidate_count,
                                parent_count,
                            )
                        )
                        or candidate_value is None
                        or parent_value is None
                        or stratum_delta is None
                        or not isinstance(candidate_count, int)
                        or candidate_count < 1
                        or not isinstance(parent_count, int)
                        or parent_count < 1
                        or abs((candidate_value - parent_value) - stratum_delta) > 1e-5
                    ):
                        valid_stratum_deltas = []
                        break
                    valid_stratum_deltas.append(stratum_delta)
                if (
                    not valid_stratum_deltas
                    or binding_delta is None
                    or abs(binding_delta - min(valid_stratum_deltas)) > 1e-5
                    or max_potency_loss is None
                    or (tier == "direct_retained")
                    != (binding_delta >= -max_potency_loss)
                ):
                    errors.append(f"candidate[{expected_rank}].binding_comparison")
        elif (
            comparison_basis is not None
            or comparison_strata != []
            or raw_binding_delta is not None
        ):
            errors.append(f"candidate[{expected_rank}].binding_comparison")
        for score_name in (
            "pharmacophore_preservation_score",
            "feature_family_recall",
            "feature_family_precision",
            "feature_family_f1",
            "binding_support_score",
            "safety_triage_score",
            "routeability_proxy",
            "priority_score",
        ):
            score = _as_float(candidate.get(score_name))
            if score is None or not 0.0 <= score <= 1.0:
                errors.append(f"candidate[{expected_rank}].{score_name}")
            if score_name == "priority_score" and score is not None:
                if previous_priority is not None and score > previous_priority:
                    errors.append(f"candidate[{expected_rank}].priority_score_order")
                previous_priority = score
        raw_anchor_score = candidate.get("target_conditioned_anchor_score")
        anchor_score = _as_float(raw_anchor_score)
        if anchor_conditioned:
            anchor_count = candidate.get("target_conditioned_anchor_count")
            preserved_anchor_count = candidate.get(
                "target_conditioned_preserved_anchor_count"
            )
            mapping_count = candidate.get("target_conditioned_mapping_count")
            mapping_ambiguous = candidate.get(
                "target_conditioned_mapping_ambiguous"
            )
            mapping_truncated = candidate.get(
                "target_conditioned_mapping_truncated"
            )
            if (
                isinstance(raw_anchor_score, bool)
                or anchor_score is None
                or not 0.0 <= anchor_score <= 1.0
                or isinstance(anchor_count, bool)
                or not isinstance(anchor_count, int)
                or anchor_count < 1
                or isinstance(preserved_anchor_count, bool)
                or not isinstance(preserved_anchor_count, int)
                or not 0 <= preserved_anchor_count <= anchor_count
                or candidate.get("target_conditioned_anchor_basis")
                != "pose_supported_parent_anchor_conservative_mcs_feature_preservation"
                or isinstance(mapping_count, bool)
                or not isinstance(mapping_count, int)
                or mapping_count < 0
                or not isinstance(mapping_ambiguous, bool)
                or mapping_ambiguous != (mapping_count > 1)
                or not isinstance(mapping_truncated, bool)
                or candidate.get("analog_pose_verified") is not False
            ):
                errors.append(f"candidate[{expected_rank}].target_conditioned_anchor")
        elif raw_anchor_score is not None:
            errors.append(f"candidate[{expected_rank}].target_conditioned_anchor")
        pharmacophore_score = _as_float(
            candidate.get("pharmacophore_preservation_score")
        )
        feature_recall = _as_float(candidate.get("feature_family_recall"))
        if (
            isinstance(pharmacophore_gate_passed, bool)
            and pharmacophore_score is not None
            and feature_recall is not None
            and min_pharmacophore is not None
            and min_feature_recall is not None
            and pharmacophore_gate_passed
            != (
                pharmacophore_score >= min_pharmacophore
                and feature_recall >= min_feature_recall
            )
        ):
            errors.append(f"candidate[{expected_rank}].pharmacophore_gate_passed")
        if isinstance(candidate_id, str):
            seen_ids.add(candidate_id)
        if isinstance(smiles, str):
            seen_smiles.add(smiles)
        if isinstance(global_rank, int) and not isinstance(global_rank, bool):
            seen_global_ranks.add(global_rank)
            previous_global_rank = global_rank
    summary = payload.get("summary")
    expected_summary = {
        "direct_activity_candidates": sum(
            candidate.get("evidence_tier") != "proxy_only"
            for candidate in candidates
            if isinstance(candidate, dict)
        ),
        "direct_retained_candidates": sum(
            candidate.get("evidence_tier") == "direct_retained"
            for candidate in candidates
            if isinstance(candidate, dict)
        ),
        "cosing_candidates": sum(
            candidate.get("cosing_reference") is True
            for candidate in candidates
            if isinstance(candidate, dict)
        ),
        "proxy_only_candidates": sum(
            candidate.get("evidence_tier") == "proxy_only"
            for candidate in candidates
            if isinstance(candidate, dict)
        ),
        "strict_pharmacophore_candidates": sum(
            candidate.get("pharmacophore_gate_passed") is True
            for candidate in candidates
            if isinstance(candidate, dict)
        ),
        "target_supported_feature_analogue_candidates": sum(
            "same_target_activity_feature_family"
            in candidate.get("admission_bases", [])
            for candidate in candidates
            if isinstance(candidate, dict)
        ),
        "target_supported_nonpharmacophore_candidates": sum(
            candidate.get("pharmacophore_gate_passed") is False
            and "same_target_activity_feature_family"
            in candidate.get("admission_bases", [])
            for candidate in candidates
            if isinstance(candidate, dict)
        ),
    }
    if not isinstance(summary, dict) or any(
        summary.get(name) != expected
        for name, expected in expected_summary.items()
    ):
        errors.append("summary")
    expected_anchor_candidates = sum(
        candidate.get("target_conditioned_anchor_score") is not None
        for candidate in candidates
        if isinstance(candidate, dict)
    )
    if isinstance(summary, dict) and (
        (
            anchor_conditioned
            and summary.get("target_conditioned_anchor_candidates")
            != expected_anchor_candidates
        )
        or (
            "target_conditioned_anchor_candidates" in summary
            and summary.get("target_conditioned_anchor_candidates")
            != expected_anchor_candidates
        )
    ):
        errors.append("summary.target_conditioned_anchor_candidates")
    if (
        manifest.get("candidate_count") != len(candidates)
        or manifest.get("candidate_summary") != summary
    ):
        errors.append("manifest candidate summary")
    candidate_pool = payload.get("candidate_pool")
    passing_count = (
        candidate_pool.get("passing_scored_molecules")
        if isinstance(candidate_pool, dict)
        else None
    )
    if (
        isinstance(passing_count, bool)
        or not isinstance(passing_count, int)
        or passing_count < len(candidates)
    ):
        errors.append("candidate_pool.passing_scored_molecules")
    strategy = payload.get("selection_strategy")
    selected_counts = strategy.get("selected_counts") if isinstance(strategy, dict) else None
    eligible_counts = strategy.get("eligible_counts") if isinstance(strategy, dict) else None
    material_fraction = (
        _as_float(strategy.get("material_track_fraction"))
        if isinstance(strategy, dict)
        else None
    )
    pharmacophore_fraction = (
        _as_float(strategy.get("pharmacophore_track_fraction"))
        if isinstance(strategy, dict)
        else None
    )
    reserved_slots = strategy.get("reserved_material_slots") if isinstance(strategy, dict) else None
    reserved_pharmacophore_slots = (
        strategy.get("reserved_pharmacophore_slots")
        if isinstance(strategy, dict)
        else None
    )
    expected_selected_counts = {
        "all": len(candidates),
        "target_activity": sum(
            "target_activity" in candidate.get("selection_tracks", [])
            for candidate in candidates
            if isinstance(candidate, dict)
        ),
        "cosmetic_material": sum(
            "cosmetic_material" in candidate.get("selection_tracks", [])
            for candidate in candidates
            if isinstance(candidate, dict)
        ),
        "strict_pharmacophore": sum(
            candidate.get("pharmacophore_gate_passed") is True
            for candidate in candidates
            if isinstance(candidate, dict)
        ),
        "both": sum(
            len(candidate.get("selection_tracks", [])) > 1
            for candidate in candidates
            if isinstance(candidate, dict)
        ),
    }
    if (
        not isinstance(strategy, dict)
        or strategy.get("mode") not in {"balanced_tracks", "global_priority"}
        or material_fraction is None
        or not 0.0 <= material_fraction <= 1.0
        or pharmacophore_fraction is None
        or not 0.0 <= pharmacophore_fraction <= 1.0
        or isinstance(reserved_slots, bool)
        or not isinstance(reserved_slots, int)
        or reserved_slots < 0
        or isinstance(reserved_pharmacophore_slots, bool)
        or not isinstance(reserved_pharmacophore_slots, int)
        or reserved_pharmacophore_slots < 0
        or selected_counts != expected_selected_counts
        or not isinstance(eligible_counts, dict)
        or eligible_counts.get("all") != passing_count
        or any(
            isinstance(eligible_counts.get(name), bool)
            or not isinstance(eligible_counts.get(name), int)
            or eligible_counts.get(name) < expected_selected_counts[name]
            for name in (
                "target_activity",
                "cosmetic_material",
                "strict_pharmacophore",
            )
        )
        or reserved_slots > expected_selected_counts["cosmetic_material"]
        or reserved_pharmacophore_slots
        > expected_selected_counts["strict_pharmacophore"]
        or (
            strategy.get("mode") == "global_priority"
            and (reserved_slots != 0 or reserved_pharmacophore_slots != 0)
        )
    ):
        errors.append("selection_strategy")
    if manifest.get("selection_strategy") != strategy:
        errors.append("manifest selection_strategy")
    if errors:
        return {
            "status": "blocked",
            "error": "대체소재 보고서 계약 위반: " + ", ".join(errors[:12]),
            "candidates": [],
        }
    return payload


def _safe_run_id(value: str) -> str:
    run_id = value.strip()
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError(
            "run ID는 영문/숫자로 시작하고 영문, 숫자, '.', '_' 또는 '-'만 사용할 수 있습니다."
        )
    return run_id


def _new_run_id() -> str:
    return "workbench_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(3)


def _find_command(name: str) -> str | None:
    path = shutil.which(name)
    if path:
        return path
    candidates = []
    if name == "micromamba":
        candidates.extend([Path.home() / ".local" / "bin" / "micromamba"])
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _mamba_root_candidates() -> list[Path]:
    candidates: list[Path] = []
    configured = os.environ.get("MAMBA_ROOT_PREFIX")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        [
            Path.home() / ".local" / "share" / "micromamba",
            Path.home() / ".local" / "share" / "mamba",
        ]
    )
    return list(dict.fromkeys(candidates))


def _mamba_root_for_env(environment: str) -> Path | None:
    for root in _mamba_root_candidates():
        if (root / "envs" / environment).is_dir():
            return root
    return None


def _subprocess_env(*, mamba_environment: str | None = None) -> dict[str, str]:
    environment = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if mamba_environment:
        root = _mamba_root_for_env(mamba_environment)
        if root is not None:
            environment["MAMBA_ROOT_PREFIX"] = str(root)
    return environment


def _os_release() -> dict[str, str]:
    values: dict[str, str] = {}
    path = Path("/etc/os-release")
    if not path.exists():
        return values
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key] = value.strip().strip('"')
    except OSError:
        return values
    return values


def _is_wsl() -> bool:
    release = platform.release().lower()
    try:
        version = Path("/proc/version").read_text(encoding="utf-8").lower()
    except OSError:
        version = ""
    return "microsoft" in release or "microsoft" in version or bool(os.environ.get("WSL_INTEROP"))


def _gpu_info() -> dict[str, Any]:
    return gpu_admission.probe(command=_find_command("nvidia-smi"), cwd=ROOT)


def _count_files(directory: Path, pattern: str) -> int:
    if not directory.exists():
        return 0
    try:
        return sum(1 for _ in directory.glob(pattern))
    except OSError:
        return 0


def _micromamba_env_exists(environment: str) -> bool:
    command = _find_command("micromamba")
    if command is None:
        return False
    if _mamba_root_for_env(environment) is not None:
        return True
    try:
        result = subprocess.run(
            [command, "env", "list", "--json"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
            env=_subprocess_env(),
        )
        payload = json.loads(result.stdout) if result.returncode == 0 else {}
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return False
    environments = payload.get("envs", []) if isinstance(payload, dict) else []
    return any(str(path).rstrip("/").endswith(f"/envs/{environment}") for path in environments)


def _model_requirement_ready(payload: dict[str, Any], requirement: str) -> bool:
    if requirement == "gpu":
        gpu = payload.get("gpu")
        return isinstance(gpu, dict) and gpu.get("cuda_available") is True
    mappings = {
        "gnina": ("tools", "gnina", "available"),
        "autodock_gpu": ("tools", "autodock_gpu", "available"),
        "autogrid": ("tools", "autogrid", "available"),
        "boltz": ("tools", "boltz", "available"),
        "diffdock": ("tools", "diffdock", "available"),
        "rtmscore": ("imports", "rtmscore", "available"),
        "psichic": ("imports", "psichic", "available"),
        "autodock_gpu_env": ("envs", "autodock_gpu", "declared"),
        "meeko_env": ("envs", "meeko", "declared"),
        "bioemu_env": ("envs", "bioemu", "declared"),
        "md_env": ("envs", "md", "declared"),
        "qm_env": ("envs", "qm", "declared"),
    }
    mapping = mappings.get(requirement)
    if mapping is None:
        return False
    section, name, expected = mapping
    section_payload = payload.get(section)
    item = section_payload.get(name) if isinstance(section_payload, dict) else None
    return isinstance(item, dict) and item.get("status") == expected


def _model_capabilities() -> dict[str, Any]:
    micromamba = _find_command("micromamba")
    if micromamba is not None and _micromamba_env_exists("cosmax-boltz2"):
        command = [
            micromamba,
            "run",
            "-n",
            "cosmax-boltz2",
            "python",
            str(ROOT / "scripts" / "model_readiness.py"),
        ]
        environment = _subprocess_env(mamba_environment="cosmax-boltz2")
    else:
        command = [sys.executable, str(ROOT / "scripts" / "model_readiness.py")]
        environment = _subprocess_env()
    if not Path(command[-1]).exists():
        missing = ["모델 상태 확인 스크립트"]
        return {
            "target_fast": {"ready": False, "missing": missing},
            "target_comprehensive": {"ready": False, "missing": missing},
            "report": {"ready": False, "missing": missing},
        }
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env=environment,
        )
        payload = json.loads(result.stdout) if result.stdout.strip() else {}
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        detail = [f"모델 상태 확인 실패: {exc}"]
        return {
            "target_fast": {"ready": False, "missing": detail},
            "target_comprehensive": {"ready": False, "missing": detail},
            "report": {"ready": False, "missing": detail},
        }

    def capability(requirements: tuple[str, ...]) -> dict[str, Any]:
        missing = [
            MODEL_REQUIREMENT_LABELS.get(requirement, requirement)
            for requirement in requirements
            if not _model_requirement_ready(payload, requirement)
        ]
        return {"ready": not missing, "missing": missing}

    return {
        "target_fast": capability(TARGET_FAST_REQUIREMENTS),
        "target_comprehensive": capability(TARGET_COMPREHENSIVE_REQUIREMENTS),
        "report": capability(REPORT_REQUIREMENTS),
    }


def _safety_capability() -> dict[str, Any]:
    micromamba = _find_command("micromamba")
    if micromamba is None or not _micromamba_env_exists("cosmax-base"):
        return {"ready": False, "missing": ["SkinScout 기본 환경"]}
    try:
        result = subprocess.run(
            [
                micromamba,
                "run",
                "-n",
                "cosmax-base",
                "python",
                str(ROOT / "scripts" / "safety_readiness.py"),
                "--runtime-mode",
                "conda",
                "--json",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env=_subprocess_env(mamba_environment="cosmax-base"),
        )
        payload = json.loads(result.stdout) if result.stdout.strip() else {}
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        return {"ready": False, "missing": [f"안전성 사전검사 실패: {exc}"]}
    checks = payload.get("checks") if isinstance(payload, dict) else None
    failed = [
        str(check.get("name") or "알 수 없는 항목")
        for check in checks or []
        if isinstance(check, dict) and check.get("ok") is not True
    ]
    if result.returncode != 0 and not failed:
        failed = ["안전성 모델·연결 설정"]
    return {"ready": not failed, "missing": failed}


def _activity_retrieval_capability() -> dict[str, Any]:
    gate = ROOT / "data" / "manifests" / "activity_retrieval_operational_gate.flag"
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "validate_activity_retrieval_gate.py"),
                "check-operational",
                "--gate",
                str(gate),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env=_subprocess_env(mamba_environment="cosmax-base"),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ready": False, "missing": [f"활성 검색 운영 gate 확인 실패: {exc}"]}
    if result.returncode == 0:
        return {"ready": True, "missing": []}
    detail = (result.stderr or result.stdout).strip().splitlines()
    reason = detail[-1] if detail else "운영 gate가 없거나 현재 데이터와 맞지 않음"
    return {"ready": False, "missing": [f"활성 검색 운영 gate({reason})"]}


def _target_search_capability() -> dict[str, Any]:
    """표적 검색이 읽는 생산 인덱스가 있는지만 본다.

    무거운 프레임은 여기서 열지 않는다 - 상태 화면이 매번 1.7M행을 읽으면 안 된다.
    실제 로드는 첫 조회 요청에서 한 번만 일어난다.
    """
    from explore_target import _discover_index_dir

    index_dir = _discover_index_dir()
    if index_dir is None:
        return {"ready": False, "missing": ["활성 검색 인덱스(Stage 0 표적 데이터)"]}
    missing = [
        name
        for name in ("edges.parquet", "ligands.parquet")
        if not (index_dir / name).is_file()
    ]
    if missing:
        return {
            "ready": False,
            "missing": [f"활성 검색 인덱스 파일({', '.join(missing)})"],
        }
    manifest_path = index_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        from activity_retrieval_scoring import INDEX_SCHEMA, PRODUCTION_INDEX_SCHEMA
    except (ImportError, OSError, json.JSONDecodeError) as exc:
        return {
            "ready": False,
            "missing": [f"활성 검색 인덱스 manifest({exc})"],
        }
    if not isinstance(manifest, dict):
        return {
            "ready": False,
            "missing": ["활성 검색 인덱스 manifest(JSON object 필요)"],
        }
    permitted = {INDEX_SCHEMA, PRODUCTION_INDEX_SCHEMA}
    if manifest.get("index_role") != "production":
        return {
            "ready": False,
            "missing": ["활성 검색 인덱스 role(production 필요)"],
        }
    if manifest.get("schema_version") not in permitted:
        return {
            "ready": False,
            "missing": [
                f"활성 검색 인덱스 schema_version({manifest.get('schema_version')!r}; "
                f"허용: {sorted(permitted)})"
            ],
        }
    return {"ready": True, "missing": [], "index_dir": str(index_dir)}


def _stage0_state() -> dict[str, Any]:
    manifest = ROOT / "data" / "manifests" / "stage0_complete.flag"
    clean_count = _count_files(ROOT / "data" / "human_clean", "*_clean.pdb")
    pocket_count = _count_files(ROOT / "data" / "human_pockets", "*.pockets.json")
    pdbqt_count = _count_files(ROOT / "data" / "human_pdbqt", "*.pdbqt")
    if manifest.exists():
        return {
            "status": "ready",
            "label": "Stage 0 데이터",
            "detail": f"인간 단백질 {clean_count:,}개 · pocket {pocket_count:,}개 · PDBQT {pdbqt_count:,}개",
            "counts": {"clean": clean_count, "pockets": pocket_count, "pdbqt": pdbqt_count},
            "manifest": _relative_path(manifest),
        }
    if clean_count or pocket_count or pdbqt_count:
        return {
            "status": "partial",
            "label": "Stage 0 데이터",
            "detail": f"부분 준비됨 · 단백질 {clean_count:,}개 · target 분석 전 추가 준비 필요",
            "counts": {"clean": clean_count, "pockets": pocket_count, "pdbqt": pdbqt_count},
            "manifest": None,
        }
    return {
        "status": "blocked",
        "label": "Stage 0 데이터",
        "detail": "target 분석용 대규모 데이터가 아직 준비되지 않았습니다.",
        "counts": {"clean": 0, "pockets": 0, "pdbqt": 0},
        "manifest": None,
    }


def _build_status() -> dict[str, Any]:
    release = _os_release()
    distro_id = release.get("ID", "").lower()
    distro_name = release.get("PRETTY_NAME", platform.system())
    is_ubuntu = distro_id == "ubuntu" or "ubuntu" in distro_name.lower()
    wsl = _is_wsl()
    ubuntu_ready = is_ubuntu and (platform.system() == "Linux")
    micromamba = _find_command("micromamba")
    runtime_ready = _micromamba_env_exists("cosmax-base")
    p2rank_ready = any(
        candidate.exists()
        for candidate in (
            ROOT / "tools" / "p2rank" / "prank",
            ROOT / "tools" / "p2rank_2.5" / "prank",
            ROOT / "tools" / "p2rank",
        )
    )
    disk = shutil.disk_usage(ROOT)
    free_gb = round(disk.free / (1024**3), 1)
    base_disk_ready = free_gb >= BASE_RUNTIME_MIN_FREE_GB
    stage0_disk_ready = free_gb >= STAGE0_MIN_FREE_GB
    disk_status = (
        "ready" if stage0_disk_ready else "warning" if base_disk_ready else "blocked"
    )
    stage0 = _stage0_state()
    gpu_info = _gpu_info()
    safety_capability = _safety_capability() if runtime_ready else {
        "ready": False,
        "missing": ["SkinScout 기본 환경"],
    }
    retrieval_capability = _activity_retrieval_capability() if runtime_ready else {
        "ready": False,
        "missing": ["SkinScout 기본 환경"],
    }
    model_capabilities = _model_capabilities() if runtime_ready else {
        "target_fast": {"ready": False, "missing": ["SkinScout 기본 환경"]},
        "target_comprehensive": {"ready": False, "missing": ["SkinScout 기본 환경"]},
        "report": {"ready": False, "missing": ["SkinScout 기본 환경"]},
    }

    def analysis_state(key: str, *, needs_stage0: bool, readiness_label: str) -> dict[str, Any]:
        missing = list(model_capabilities[key]["missing"])
        if not runtime_ready:
            missing = ["SkinScout 기본 환경"]
        elif not safety_capability["ready"]:
            missing = [
                f"안전성 사전검사({item})"
                for item in safety_capability["missing"]
            ] + missing
        if needs_stage0 and stage0["status"] != "ready":
            missing.insert(0, "Stage 0/report-fast 표적 데이터")
        if not retrieval_capability["ready"]:
            missing = list(retrieval_capability["missing"]) + missing
        if gpu_info.get("available_for_analysis") is not True:
            free_mib = gpu_info.get("free_mib")
            utilization = gpu_info.get("utilization_percent")
            process_count = gpu_info.get("external_compute_process_count")
            current_parts = []
            if isinstance(process_count, int):
                current_parts.append(f"외부 계산 {process_count}개")
            if isinstance(free_mib, int):
                current_parts.append(f"여유 {free_mib:,} MiB")
            if isinstance(utilization, int):
                current_parts.append(f"사용률 {utilization}%")
            current = ", ".join(current_parts) or "상태 확인 불가"
            missing.insert(
                0,
                (
                    f"GPU 실행 가능 상태(현재: {current}; 필요: 외부 계산 0개, "
                    f"여유 {GPU_ANALYSIS_MIN_FREE_MIB:,} MiB 이상, 사용률 "
                    f"{GPU_ANALYSIS_MAX_UTILIZATION_PERCENT}% 이하)"
                ),
            )
        missing = list(dict.fromkeys(missing))
        return {
            "ready": not missing,
            "status": "ready" if not missing else "blocked",
            "missing": missing,
            "detail": f"{readiness_label} 가능" if not missing else "필요: " + ", ".join(missing),
        }

    analysis_readiness = {
        "safety": {
            "ready": safety_capability["ready"],
            "status": "ready" if safety_capability["ready"] else "blocked",
            "missing": safety_capability["missing"],
            "detail": (
                "로컬 사전조건 준비됨 · 외부 감작성 모델 연결은 실행 시작 시 확인"
                if safety_capability["ready"]
                else "필요: " + ", ".join(safety_capability["missing"])
            ),
        },
        "target_fast": analysis_state(
            "target_fast",
            needs_stage0=True,
            readiness_label="Demo 빠른 Target 분석",
        ),
        "target_comprehensive": analysis_state(
            "target_comprehensive",
            needs_stage0=True,
            readiness_label="Full/Advanced Target 분석",
        ),
        "report": analysis_state(
            "report",
            needs_stage0=True,
            readiness_label="Full/Advanced 보고서",
        ),
    }
    substitute_missing = list(analysis_readiness["target_fast"]["missing"])
    substitute_inputs = (
        ("CosIng 후보 라이브러리", ROOT / "data" / "cosing" / "cosing.parquet"),
        *SUBSTITUTE_ACTIVITY_EVIDENCE,
    )
    for label, path in substitute_inputs:
        if not path.exists() or not path.is_file() or path.stat().st_size == 0:
            substitute_missing.append(label)
    substitute_missing = list(dict.fromkeys(substitute_missing))
    analysis_readiness["substitute"] = {
        "ready": not substitute_missing,
        "status": "ready" if not substitute_missing else "blocked",
        "missing": substitute_missing,
        "detail": (
            "분석 가능"
            if not substitute_missing
            else "필요: " + ", ".join(substitute_missing)
        ),
    }
    substitute_target_conditioned_missing = list(
        analysis_readiness["target_comprehensive"]["missing"]
    )
    for label, path in substitute_inputs:
        if not path.exists() or not path.is_file() or path.stat().st_size == 0:
            substitute_target_conditioned_missing.append(label)
    substitute_target_conditioned_missing = list(
        dict.fromkeys(substitute_target_conditioned_missing)
    )
    analysis_readiness["substitute_target_conditioned"] = {
        "ready": not substitute_target_conditioned_missing,
        "status": "ready" if not substitute_target_conditioned_missing else "blocked",
        "missing": substitute_target_conditioned_missing,
        "detail": (
            "3D 표적 anchor 분석 가능"
            if not substitute_target_conditioned_missing
            else "필요: " + ", ".join(substitute_target_conditioned_missing)
        ),
    }
    # 대체소재 검색은 CosIng 표와 압축 지문 인덱스 두 개만 있으면 돈다. 표적
    # 예측 스테이지와 무관하므로 준비 상태도 따로 센다 - 나머지가 하나도 없어도
    # 이 화면은 쓸 수 있고, 그것을 시작 화면에서 알 수 있어야 한다.
    alternatives_missing = []
    for label, path in ALTERNATIVES_INPUTS:
        if not path.exists() or (path.is_file() and path.stat().st_size == 0):
            alternatives_missing.append(label)
    analysis_readiness["alternatives"] = {
        "ready": not alternatives_missing,
        "status": "ready" if not alternatives_missing else "blocked",
        "missing": alternatives_missing,
        "detail": (
            "등재 원료에서 핵심구조 유지 후보를 찾을 수 있습니다"
            if not alternatives_missing
            else "필요: " + ", ".join(alternatives_missing)
        ),
    }
    # 표적 검색은 생산 인덱스 하나만 있으면 돈다. GPU·안전성 모델과 무관하므로
    # 따로 센다 - 표적 예측이 준비되지 않은 설치에서도 이 화면은 쓸 수 있다.
    target_search_capability = _target_search_capability()
    analysis_readiness["target_search"] = {
        "ready": target_search_capability["ready"],
        "status": "ready" if target_search_capability["ready"] else "blocked",
        "missing": target_search_capability["missing"],
        "detail": (
            "단백질 이름으로 알려진 결합 화합물을 찾을 수 있습니다"
            if target_search_capability["ready"]
            else "필요: " + ", ".join(target_search_capability["missing"])
        ),
    }

    checks = [
        {
            "id": "os",
            "label": "Ubuntu 실행 환경",
            "status": "ready" if ubuntu_ready else "blocked",
            "detail": f"{distro_name}{' · WSL2' if wsl else ''}",
            "action": None if ubuntu_ready else "OS별 안전 설치 안내 열기",
        },
        {
            "id": "repository",
            "label": "SkinScout 소스",
            "status": "ready" if (ROOT / "scripts" / "run_skinscout.py").exists() else "blocked",
            "detail": str(ROOT),
            "action": None,
        },
        {
            "id": "micromamba",
            "label": "패키지 관리자",
            "status": "ready" if micromamba else "blocked",
            "detail": micromamba or "micromamba가 필요합니다.",
            "action": "설치 가이드 열기" if not micromamba else None,
        },
        {
            "id": "runtime",
            "label": "기본 실행 환경",
            "status": "ready" if runtime_ready else "action",
            "detail": "cosmax-base 준비됨" if runtime_ready else "SkinScout 환경 준비가 필요합니다.",
            "action": None if runtime_ready else "환경 준비 시작",
        },
        {
            "id": "p2rank",
            "label": "Pocket 분석 도구",
            "status": "ready" if p2rank_ready else "action",
            "detail": "P2Rank 준비됨" if p2rank_ready else "기본 설치 단계에서 준비됩니다.",
            "action": None if p2rank_ready else "환경 준비 시작",
        },
        {
            "id": "gpu-analysis",
            "label": "GPU 실행 상태",
            "status": "ready" if gpu_info["available_for_analysis"] else "warning",
            "detail": gpu_info["detail"],
            "action": (
                None
                if gpu_info["available_for_analysis"]
                else "GPU 작업 종료 후 새로고침"
            ),
        },
        {
            "id": "disk",
            "label": "여유 디스크",
            "status": disk_status,
            "detail": f"{free_gb:,.1f} GB 사용 가능",
            "action": (
                f"Stage 0를 시작하려면 최소 {STAGE0_MIN_FREE_GB} GB가 필요합니다."
                if not stage0_disk_ready
                else None
            ),
        },
        {
            "id": "stage0",
            **stage0,
            "action": "Stage 0 데이터 준비" if stage0["status"] != "ready" else None,
        },
        {
            "id": "safety-analysis",
            "label": "안전성 분석",
            "status": analysis_readiness["safety"]["status"],
            "detail": analysis_readiness["safety"]["detail"],
            "action": None if analysis_readiness["safety"]["ready"] else "기본 환경 준비",
        },
        {
            "id": "target-analysis",
            "label": "Demo 빠른 Target 분석",
            "status": analysis_readiness["target_fast"]["status"],
            "detail": analysis_readiness["target_fast"]["detail"],
            "action": None if analysis_readiness["target_fast"]["ready"] else "누락 항목 확인",
        },
        {
            "id": "advanced-analysis",
            "label": "Full/Advanced Target·보고서",
            "status": analysis_readiness["report"]["status"],
            "detail": analysis_readiness["report"]["detail"],
            "action": None if analysis_readiness["report"]["ready"] else "누락 항목 확인",
        },
        {
            "id": "alternatives-search",
            "label": "대체소재 검색(핵심구조 유지)",
            "status": analysis_readiness["alternatives"]["status"],
            "detail": analysis_readiness["alternatives"]["detail"],
            "action": None if analysis_readiness["alternatives"]["ready"] else "누락 항목 확인",
        },
        {
            "id": "target-search",
            "label": "표적 검색(알려진 결합 화합물)",
            "status": analysis_readiness["target_search"]["status"],
            "detail": analysis_readiness["target_search"]["detail"],
            "action": None if analysis_readiness["target_search"]["ready"] else "누락 항목 확인",
        },
        {
            "id": "substitute-analysis",
            "label": "Pharmacophore 대체소재 발굴",
            "status": analysis_readiness["substitute"]["status"],
            "detail": analysis_readiness["substitute"]["detail"],
            "action": None if analysis_readiness["substitute"]["ready"] else "누락 항목 확인",
        },
    ]
    if ubuntu_ready and analysis_readiness["target_fast"]["ready"]:
        next_action = "안전성 또는 Target 분석을 시작할 수 있습니다."
    elif ubuntu_ready and analysis_readiness["safety"]["ready"]:
        if gpu_info["detected"] and not gpu_info["available_for_analysis"]:
            next_action = (
                "안전성 분석은 지금 시작할 수 있습니다. Target 분석은 현재 GPU 작업이 "
                "끝난 뒤 상태를 다시 확인하세요."
            )
        else:
            next_action = "안전성 분석부터 시작할 수 있습니다. Target 분석은 누락 항목을 먼저 준비하세요."
    elif not ubuntu_ready:
        next_action = "먼저 현재 운영체제에 맞는 Ubuntu 실행 환경을 준비하세요."
    elif not runtime_ready:
        next_action = "SkinScout 기본 실행 환경을 준비하세요."
    else:
        next_action = "안전성 분석은 가능할 수 있습니다. target 분석은 Stage 0 준비 후 시작하세요."
    return {
        "application": "SkinScout Workbench",
        "csrf_token": _csrf_token(),
        "checked_at": _now(),
        "system": {
            "platform": platform.system(),
            "release": platform.release(),
            "distro": distro_name,
            "distro_id": distro_id,
            "is_ubuntu": is_ubuntu,
            "is_wsl": wsl,
            "ubuntu_ready": ubuntu_ready,
        },
        "runtime": {
            "python": sys.executable,
            "python_version": platform.python_version(),
            "micromamba": micromamba,
            "base_env": runtime_ready,
            "p2rank": p2rank_ready,
        },
        "gpu": gpu_info,
        "disk": {
            "free_gb": free_gb,
            "status": disk_status,
            "base_ready": base_disk_ready,
            "stage0_ready": stage0_disk_ready,
            "base_min_free_gb": BASE_RUNTIME_MIN_FREE_GB,
            "stage0_min_free_gb": STAGE0_MIN_FREE_GB,
        },
        "stage0": stage0,
        "analysis_readiness": analysis_readiness,
        "checks": checks,
        "next_action": next_action,
        "safe_boundary": (
            "Workbench는 OS 디스크를 포맷하거나 부트 장치를 수정하지 않습니다. "
            "NVIDIA 드라이버 설치는 자동으로 수행하지 않으며, Target 분석 전 운영체제에 "
            "정상 설치되어 있어야 합니다."
        ),
        "install_plan": [
            {
                "id": "ubuntu",
                "title": "Ubuntu 실행 환경",
                "status": "ready" if ubuntu_ready else "manual",
                "detail": "Ubuntu Desktop 또는 Ubuntu on WSL2를 안전하게 준비합니다. NVIDIA 드라이버는 사전 조건입니다.",
            },
            {
                "id": "runtime",
                "title": "Demo 프로필",
                "status": "ready" if runtime_ready else "available",
                "detail": "micromamba, cosmax-base, P2Rank을 자동으로 준비합니다.",
            },
            {
                "id": "stage0",
                "title": "Stage 0 데이터",
                "status": stage0["status"],
                "detail": "target 분석에 필요한 인간 proteome과 pocket 데이터를 준비합니다.",
            },
            {
                "id": "advanced-models",
                "title": "Full 프로필",
                "status": analysis_readiness["report"]["status"],
                "detail": "PSICHIC, RTMScore, Boltz, BioEmu, MD, QM 등 고급 보고서 도구는 Demo 준비와 별도로 확인합니다.",
            },
        ],
        "setup_profiles": SETUP_PROFILES,
    }


def get_status(force: bool = False) -> dict[str, Any]:
    global _STATUS_CACHE
    now = time.monotonic()
    with _STATUS_CACHE_LOCK:
        if not force and _STATUS_CACHE and now - _STATUS_CACHE[0] < STATUS_CACHE_SECONDS:
            return _STATUS_CACHE[1]
    payload = _build_status()
    with _STATUS_CACHE_LOCK:
        _STATUS_CACHE = (time.monotonic(), payload)
    return payload


def _refresh_jobs() -> None:
    coordinator = _coordinator()
    coordinator.recover()
    with _JOBS_LOCK:
        for job in _JOBS.values():
            if job.process is None or job.status not in {"queued", "running", "cancel_requested"}:
                continue
            returncode = job.process.poll()
            if returncode is None:
                try:
                    record = coordinator.get_job(job.job_id)
                except KeyError:
                    continue
                job.status = str(record["status"])
                continue
            job.returncode = returncode
            job.ended_at = _now()
            try:
                record = coordinator.get_job(job.job_id)
            except KeyError:
                cancelled = job.status == "cancel_requested"
                job.status = (
                    "cancelled"
                    if cancelled
                    else ("completed" if returncode == 0 else "failed")
                )
            else:
                job.status = str(record["status"])
            if returncode != 0 and not job.detail:
                job.detail = "실행 로그에서 실패 원인을 확인하세요."


def _job_payload(job: Job) -> dict[str, Any]:
    _refresh_jobs()
    return job.public()


def _durable_job_adapter(record: dict[str, Any]) -> Job:
    payload = record.get("payload")
    payload = payload if isinstance(payload, dict) else {}
    log_path_text = payload.get("log_path")
    log_path = (
        ROOT / log_path_text
        if isinstance(log_path_text, str) and log_path_text
        else None
    )
    created_at = record.get("created_at")
    started_at = None
    if isinstance(created_at, (int, float)):
        started_at = datetime.fromtimestamp(
            created_at,
            tz=timezone.utc,
        ).isoformat(timespec="seconds")
    return Job(
        job_id=str(record["job_id"]),
        kind=str(record["kind"]),
        run_id=(
            str(payload["run_id"])
            if isinstance(payload.get("run_id"), str)
            else None
        ),
        status=str(record["status"]),
        started_at=started_at,
        command=[
            str(item)
            for item in payload.get("command", [])
            if isinstance(item, str)
        ],
        log_path=log_path,
    )


def _monitor_process_job(job: Job) -> None:
    process = job.process
    attempt_id = job.attempt_id
    attempt_token = job.attempt_token
    if process is None or attempt_id is None or attempt_token is None:
        return
    coordinator = _coordinator()
    while True:
        try:
            returncode = process.wait(timeout=HEARTBEAT_SECONDS)
            break
        except subprocess.TimeoutExpired:
            try:
                heartbeat = coordinator.heartbeat(attempt_id, attempt_token)
            except CoordinatorError:
                try:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGTERM)
                    else:
                        process.terminate()
                except OSError:
                    pass
                _cleanup_private_inputs(job.cleanup_paths)
                return
            with _JOBS_LOCK:
                job.status = (
                    "cancel_requested"
                    if heartbeat["cancel_requested"]
                    else "running"
                )
    try:
        record = coordinator.get_job(job.job_id)
        if record["status"] == "cancel_requested":
            final = coordinator.acknowledge_cancel(attempt_id, attempt_token)
        else:
            final = coordinator.complete_attempt(
                attempt_id,
                attempt_token,
                status="completed" if returncode == 0 else "failed",
                result={"returncode": returncode},
            )
    except (CoordinatorError, KeyError):
        final = coordinator.get_job(job.job_id)
    with _JOBS_LOCK:
        job.returncode = returncode
        job.ended_at = _now()
        job.status = str(final["status"])
        if returncode != 0 and job.status != "cancelled" and not job.detail:
            job.detail = "실행 로그에서 실패 원인을 확인하세요."
    _cleanup_private_inputs(job.cleanup_paths)


def _cleanup_private_inputs(paths: tuple[Path, ...]) -> None:
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            # A terminal analysis result must not be hidden merely because a
            # best-effort cleanup failed. The private directory remains 0700.
            pass


def _python_command() -> list[str]:
    micromamba = _find_command("micromamba")
    if micromamba and _micromamba_env_exists("cosmax-base"):
        return [micromamba, "run", "-n", "cosmax-base", "python"]
    return [sys.executable]


def _start_process_job(
    *,
    kind: str,
    run_id: str | None,
    command: list[str],
    log_path: Path,
    resource: str = "gpu:0",
    payload_extra: dict[str, Any] | None = None,
    require_gpu_admission: bool = False,
    cleanup_paths: tuple[Path, ...] = (),
) -> Job:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    job_id = f"{kind}_{run_id or int(time.time())}_{secrets.token_hex(4)}"
    coordinator = _coordinator()
    durable_payload = {
        "run_id": run_id,
        "command": command,
        "log_path": _relative_path(log_path),
        "managed_by": "workbench-local-process",
    }
    if payload_extra:
        durable_payload.update(payload_extra)
    coordinator.create_job(
        kind,
        durable_payload,
        job_id=job_id,
        priority=100 if kind == "run" else 10,
    )
    try:
        attempt = coordinator.claim_job(
            job_id,
            worker_id=f"workbench:{os.getpid()}",
            resource=resource,
        )
    except LeaseUnavailable as exc:
        coordinator.cancel_job(job_id)
        _cleanup_private_inputs(cleanup_paths)
        raise RuntimeError("GPU가 다른 SkinScout 작업에서 사용 중입니다.") from exc
    if require_gpu_admission:
        gpu_info = _gpu_info()
        if gpu_info.get("available_for_analysis") is not True:
            coordinator.complete_attempt(
                attempt.attempt_id,
                attempt.token,
                status="failed",
                result={"reason": "gpu_admission_failed", "gpu": gpu_info},
            )
            _cleanup_private_inputs(cleanup_paths)
            detail = str(gpu_info.get("detail") or "GPU 상태를 확인하지 못했습니다.")
            raise RuntimeError(
                f"GPU 실행 상태를 다시 확인했지만 분석을 시작할 수 없습니다: {detail}"
            )
    try:
        with log_path.open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=_subprocess_env(mamba_environment="cosmax-base"),
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
            )
        identity = _process_identity(process)
        coordinator.record_process_identity(
            attempt.attempt_id,
            attempt.token,
            identity,
        )
    except OSError as exc:
        if "process" in locals():
            try:
                _stop_process_identity(_process_identity(process), timeout=2.0)
            except OSError:
                process.terminate()
        coordinator.complete_attempt(
            attempt.attempt_id,
            attempt.token,
            status="failed",
            result={"reason": "process_start_failed"},
        )
        _cleanup_private_inputs(cleanup_paths)
        raise RuntimeError(f"프로세스를 시작하지 못했습니다: {exc}") from exc
    except CoordinatorError:
        _stop_process_identity(_process_identity(process), timeout=2.0)
        _cleanup_private_inputs(cleanup_paths)
        raise
    job = Job(
        job_id=job_id,
        kind=kind,
        run_id=run_id,
        status="running",
        started_at=_now(),
        command=command,
        log_path=log_path,
        process=process,
        attempt_id=attempt.attempt_id,
        attempt_token=attempt.token,
        lease_resource=resource,
        cleanup_paths=cleanup_paths,
    )
    with _JOBS_LOCK:
        _JOBS[job_id] = job
    threading.Thread(
        target=_monitor_process_job,
        args=(job,),
        name=f"monitor_{job_id}",
        daemon=True,
    ).start()
    return job


def _active_job_for_run(run_id: str) -> Job | None:
    _refresh_jobs()
    with _JOBS_LOCK:
        for job in _JOBS.values():
            if job.run_id == run_id and job.status in {"queued", "running", "cancel_requested"}:
                return job
    for record in _coordinator().list_jobs(
        statuses={"queued", "running", "cancel_requested"},
        limit=1_000,
    ):
        payload = record.get("payload")
        if isinstance(payload, dict) and payload.get("run_id") == run_id:
            return _durable_job_adapter(record)
    return None


def _durable_job_for_run(run_id: str) -> dict[str, Any] | None:
    for record in _coordinator().list_jobs(limit=1_000):
        payload = record.get("payload")
        if isinstance(payload, dict) and payload.get("run_id") == run_id:
            return record
    return None


def _lifecycle_status(
    active: Job | None,
    durable_record: dict[str, Any] | None,
) -> str:
    """The run's actual lifecycle state, durable record first.

    A run summary is written before the verifier runs, so a failed or cancelled
    coordinator job can still have one on disk. Treating the summary as proof
    of completion lost that terminal state; only a run with no durable record
    at all falls back to the filesystem's verdict.
    """
    if active is not None:
        return active.status
    if isinstance(durable_record, dict):
        status = durable_record.get("status")
        if isinstance(status, str) and status:
            return status
    return "completed"


def _tail(path: Path | None, max_bytes: int = 48_000) -> str:
    if path is None or not path.exists():
        return ""
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - max_bytes), os.SEEK_SET)
            data = stream.read(max_bytes)
    except OSError:
        return ""
    return data.decode("utf-8", errors="replace")


def _load_metadata() -> dict[str, dict[str, str]]:
    global _METADATA_CACHE
    path = ROOT / "data" / "hpa" / "proteinatlas.tsv"
    if not path.exists():
        return {}
    mtime = path.stat().st_mtime
    if _METADATA_CACHE and _METADATA_CACHE[0] == mtime:
        return _METADATA_CACHE[1]
    metadata: dict[str, dict[str, str]] = {}
    try:
        with path.open(encoding="utf-8", newline="", errors="replace") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            for row in reader:
                target_id = (row.get("Uniprot") or row.get("UniProt") or "").strip()
                if not target_id:
                    continue
                metadata[target_id] = {
                    "gene_symbol": (row.get("Gene") or "").strip(),
                    "protein_name": (row.get("Gene description") or "").strip(),
                    "molecular_function": (row.get("Molecular function") or "").strip(),
                }
    except (OSError, csv.Error):
        return {}
    _METADATA_CACHE = (mtime, metadata)
    return metadata


def _as_float(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _analysis_kind(record: dict[str, Any] | None) -> str | None:
    payload = record.get("payload") if isinstance(record, dict) else None
    value = payload.get("analysis_kind") if isinstance(payload, dict) else None
    return value if isinstance(value, str) and value else None


def _summary_card(
    run_id: str,
    summary: dict[str, Any] | None,
    status: str,
    *,
    analysis_kind: str | None = None,
) -> dict[str, Any]:
    summary = summary or {}
    compound = summary.get("compound") if isinstance(summary.get("compound"), dict) else {}
    overall = summary.get("overall_decision") if isinstance(summary.get("overall_decision"), dict) else {}
    binding = summary.get("skin_specialized_binding") if isinstance(summary.get("skin_specialized_binding"), dict) else {}
    target = summary.get("target_prediction") if isinstance(summary.get("target_prediction"), dict) else {}
    top_targets = target.get("top_targets") if isinstance(target.get("top_targets"), list) else []
    top = top_targets[0] if top_targets and isinstance(top_targets[0], dict) else {}
    run_dir = RUNS_DIR / run_id
    substitute = _substitute_report(run_id)
    has_substitute = analysis_kind == "substitute" or substitute is not None or (
        run_dir / SUBSTITUTE_RELATIVE_DIR / "substitute_run_manifest.json"
    ).is_file()
    try:
        updated = datetime.fromtimestamp(run_dir.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
    except OSError:
        updated = None
    verification = _verification_state(run_id)
    return {
        "run_id": run_id,
        "status": _settled_status(status, verification),
        "verification": verification,
        # Whether the pipeline left a readable summary is a different fact from
        # how the job ended: a failed or cancelled job can still have partial
        # outputs, and a completed job can have none. Keep them separate.
        "outputs_available": bool(summary),
        "progress": _run_progress(run_id),
        "viewer": _viewer_state(run_id),
        "preset": "substitute" if has_substitute else summary.get("preset"),
        "analysis_kind": "substitute" if has_substitute else summary.get("preset"),
        "mode": summary.get("mode"),
        "input_smiles": compound.get("input_smiles") or compound.get("canonical_smiles"),
        "canonical_smiles": compound.get("canonical_smiles"),
        "decision": overall.get("decision"),
        "recommended_action": overall.get("recommended_action"),
        # A claim needs the pipeline's own verdict, a passing verifier, and no
        # diagnostic-only condition recorded by the CLI.
        "claimable": (
            overall.get("claimable") is True
            and verification["ok"]
            and not verification.get("diagnostic_nonclaimable_reasons")
            and not overall.get("diagnostic_nonclaimable_reasons")
        ),
        "requires_human_review": overall.get("requires_human_review"),
        "reason_count": len(overall.get("reasons", [])) if isinstance(overall.get("reasons"), list) else 0,
        "top_target": {
            "target_id": top.get("target_id") or binding.get("top_target_id"),
            "gene_symbol": top.get("gene_symbol") or binding.get("top_target_gene_symbol"),
            "protein_name": top.get("protein_name") or binding.get("top_target_protein_name"),
            "final_score": top.get("final_score") or binding.get("top_target_final_score"),
        },
        "target_count": target.get("n_targets"),
        "substitute_count": (
            len(substitute.get("candidates", []))
            if isinstance(substitute, dict) and substitute.get("status") != "blocked"
            else None
        ),
        "skin_context_supported": binding.get("skin_context_supported"),
        "updated_at": updated,
    }


def _run_summary(run_id: str) -> dict[str, Any] | None:
    return _read_json(RUNS_DIR / run_id / "run_summary.json")


# Stage progress is read from artifacts the pipeline actually wrote. The UI
# previously pinned every running job to the second of six chips regardless of
# what the run was doing, which is worse than showing nothing: on a 42-hour run
# it tells the reader the job is stuck.
RUN_STAGE_EVIDENCE: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("input", "입력 확인", ("01_input",)),
    ("safety", "안전성", ("02_admet", "02b_cosmetic_drug")),
    ("targets", "Target", ("03_targets",)),
    ("physics", "구조/물리", ("04_target", "05_boltz2", "05_pharmacophore",
                              "06_bioemu", "07_md", "08_qm")),
    ("report", "보고서", ("09_report", "run_summary.json")),
    ("verification", "검증", ("run_verification.json",)),
)


def _stage_has_output(run_dir: Path, name: str) -> bool:
    path = run_dir / name
    try:
        if path.is_dir():
            return any(path.iterdir())
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _run_progress(run_id: str) -> dict[str, Any]:
    """Which stages have produced output, and whether that is knowable."""
    run_dir = RUNS_DIR / run_id
    stages: list[dict[str, Any]] = []
    for key, label, evidence in RUN_STAGE_EVIDENCE:
        done = any(_stage_has_output(run_dir, name) for name in evidence)
        stages.append({"key": key, "label": label, "done": done})
    completed = sum(1 for stage in stages if stage["done"])
    return {
        "stages": stages,
        "completed": completed,
        # Nothing on disk yet means the run has not reached a stage we can name.
        # Say so rather than inventing a position.
        "determinate": completed > 0,
    }


def _viewer_state(run_id: str) -> dict[str, Any]:
    """Whether the offline 3D viewer exists, and why not when it does not."""
    run_dir = RUNS_DIR / run_id
    index = run_dir / "viewer" / "index.html"
    status = _read_json(run_dir / "run_viewer_status.json") or {}
    if index.is_file():
        if _run_verification(run_id) is not None and _run_file_for_download(
            run_id, "viewer/index.html"
        ) is None:
            return {
                "available": False,
                "path": None,
                "reason": "3D viewer가 검증된 산출물 seal에 포함되지 않았습니다.",
            }
        return {"available": True, "path": "viewer/index.html", "reason": None}
    reason = status.get("reason") if isinstance(status.get("reason"), str) else None
    return {"available": False, "path": None, "reason": reason}


def _run_verification(run_id: str) -> dict[str, Any] | None:
    return _read_json(RUNS_DIR / run_id / "run_verification.json")


def _verification_state(run_id: str) -> dict[str, Any]:
    """What the verifier concluded about this run, as a first-class fact.

    `run_summary.json` is written before the verifier runs and says nothing
    about whether the run passed it, so treating the summary's existence as
    success let a failed run present itself as completed and claimable. Of the
    44 committed runs, 6 fail verification.
    """
    payload = _run_verification(run_id)
    if payload is None:
        return {
            "checked": False,
            "ok": False,
            "status": None,
            "verifier_status": None,
            "error": None,
            "diagnostic_nonclaimable_reasons": [],
        }
    status = payload.get("status")
    verifier_status = payload.get("verifier_status")
    error = _verification_contract_error(run_id, payload)
    raw_reasons = payload.get("diagnostic_nonclaimable_reasons")
    reasons = (
        [str(reason).strip() for reason in raw_reasons if str(reason).strip()]
        if isinstance(raw_reasons, list)
        else []
    )
    return {
        "checked": True,
        "ok": error is None,
        "status": status if isinstance(status, str) else None,
        "verifier_status": verifier_status if isinstance(verifier_status, str) else None,
        "error": error,
        # CLI가 run_verification.json 최상위에 기록한 진단 비주장 사유(F03).
        "diagnostic_nonclaimable_reasons": reasons,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_artifact_map(
    run_id: str,
    payload: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], str | None]:
    run_dir = (RUNS_DIR / run_id).resolve()
    artifacts = payload.get("verified_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return {}, "verified_artifacts가 없거나 비어 있습니다."
    verified: dict[str, dict[str, Any]] = {}
    for position, artifact in enumerate(artifacts, start=1):
        if not isinstance(artifact, dict):
            return {}, f"verified_artifacts[{position}] 형식이 올바르지 않습니다."
        raw_path = artifact.get("path")
        size = artifact.get("bytes")
        digest = artifact.get("sha256")
        if (
            not isinstance(raw_path, str)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            return {}, f"verified_artifacts[{position}] fingerprint가 올바르지 않습니다."
        path = Path(raw_path)
        try:
            relative = path.resolve().relative_to(run_dir).as_posix()
        except (ValueError, OSError):
            return {}, f"verified_artifacts[{position}] 경로가 실행 디렉터리와 맞지 않습니다."
        if relative in verified:
            return {}, f"검증 산출물 경로가 중복되었습니다: {relative}"
        try:
            if path.is_symlink() or not path.is_file() or path.stat().st_size != size:
                return {}, f"검증 산출물의 크기 또는 형식이 바뀌었습니다: {relative}"
            if not secrets.compare_digest(_sha256_file(path), digest):
                return {}, f"검증 산출물의 SHA-256이 바뀌었습니다: {relative}"
        except OSError as exc:
            return {}, f"검증 산출물을 읽을 수 없습니다: {relative}: {exc}"
        verified[relative] = dict(artifact)
    return verified, None


def _verification_contract_error(run_id: str, payload: dict[str, Any]) -> str | None:
    explicit = payload.get("verifier_contract_error")
    if isinstance(explicit, str) and explicit and payload.get("status") != "ok":
        return explicit
    run_dir = (RUNS_DIR / run_id).resolve()
    summary = _run_summary(run_id)
    if not isinstance(summary, dict):
        return "run_summary.json을 읽을 수 없습니다."
    if summary.get("schema_version") != "skinscout.run_summary.v1":
        return "run summary schema_version이 올바르지 않습니다."
    if summary.get("run_id") != run_id:
        return "run summary run_id가 현재 실행과 맞지 않습니다."
    input_provenance = payload.get("input_provenance")
    compound = summary.get("compound")
    if not isinstance(input_provenance, dict) or not isinstance(compound, dict):
        return "검증 기록과 run summary의 입력 provenance가 없습니다."
    for provenance_field in (
        "input_type",
        "input_smiles",
        "input_canonical_smiles",
        "input_sdf",
    ):
        if input_provenance.get(provenance_field) != compound.get(provenance_field):
            return f"검증 기록 {provenance_field}가 run summary와 맞지 않습니다."
    try:
        from run_skinscout import _validate_stored_completed_verification_record

        _validate_stored_completed_verification_record(
            payload,
            run_dir,
            preset=str(summary.get("preset") or ""),
            mode=str(summary.get("mode") or ""),
        )
    except (ImportError, SystemExit) as exc:
        return str(exc) or "검증 기록 계약을 확인하지 못했습니다."
    return None


class _RunSealSnapshot:
    """One request's immutable view of a run's verified artifact seal.

    Verifying ``verified_artifacts`` hashes every sealed file. Repeating that
    inside each member read made a bundle request cost (members x artifacts)
    hashes, which the audit called out as O(N^2) I/O for larger bundles. A
    request builds this snapshot once and reuses it.

    TOCTOU protection is kept: every member read still compares its bytes to
    the snapshot fingerprint, and ``unchanged()`` re-stats every sealed
    artifact at the end of the request so a file that changes after
    verification fails the request instead of being served stale.
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.run_dir = (RUNS_DIR / run_id).resolve()
        self.payload = _run_verification(run_id)
        self.claims_success = _verification_claims_success(self.payload)
        self.contract_error = (
            _verification_contract_error(run_id, self.payload)
            if self.claims_success and isinstance(self.payload, dict)
            else None
        )
        self.artifacts: dict[str, dict[str, Any]] | None = None
        self.artifact_error: str | None = None
        self.fingerprints: dict[str, tuple[int, int]] = {}
        if self.claims_success and self.contract_error is None and isinstance(self.payload, dict):
            self.artifacts, self.artifact_error = _verified_artifact_map(run_id, self.payload)
            if self.artifact_error is None:
                self._capture_fingerprints()

    def _capture_fingerprints(self) -> None:
        fingerprints: dict[str, tuple[int, int]] = {}
        for relative, artifact in (self.artifacts or {}).items():
            try:
                info = (self.run_dir / relative).stat()
            except OSError:
                self.artifact_error = "검증 산출물을 읽을 수 없습니다."
                self.artifacts = None
                self.fingerprints = {}
                return
            if info.st_size != artifact.get("bytes"):
                self.artifact_error = "검증 산출물의 크기가 바뀌었습니다."
                self.artifacts = None
                self.fingerprints = {}
                return
            fingerprints[relative] = (info.st_size, info.st_mtime_ns)
        self.fingerprints = fingerprints

    def seal_for_relative(self, relative: str) -> dict[str, Any] | None:
        """The seal entry for one run-relative path, or the failure marker."""
        if not self.claims_success:
            return None
        if self.contract_error is not None:
            return {"invalid": True}
        if relative == "run_verification.json":
            # This is the attestation itself and cannot recursively seal its bytes.
            return {"attestation": True}
        if self.artifact_error is not None or self.artifacts is None:
            return {"invalid": True}
        return self.artifacts.get(relative, {"invalid": True})

    def unchanged(self) -> bool:
        """Whether every sealed artifact still matches the snapshot metadata.

        Unsealed runs have nothing to keep stable, and a failed contract never
        yields a bundle, so both count as unchanged here.
        """
        if not self.claims_success or self.contract_error is not None:
            return True
        if self.artifact_error is not None or self.artifacts is None:
            return False
        for relative, expected in self.fingerprints.items():
            try:
                info = (self.run_dir / relative).stat()
            except OSError:
                return False
            if (info.st_size, info.st_mtime_ns) != expected:
                return False
        return True


def _verified_run_file(
    run_id: str,
    relative: str,
    *,
    snapshot: _RunSealSnapshot | None = None,
) -> Path | None:
    if snapshot is not None and snapshot.run_id == run_id:
        if (
            snapshot.claims_success
            and snapshot.contract_error is None
            and snapshot.artifact_error is None
            and snapshot.artifacts is not None
            and relative in snapshot.artifacts
        ):
            return RUNS_DIR / run_id / relative
        return None
    payload = _run_verification(run_id)
    if payload is None or _verification_contract_error(run_id, payload) is not None:
        return None
    artifacts, error = _verified_artifact_map(run_id, payload)
    if error is not None or relative not in artifacts:
        return None
    return RUNS_DIR / run_id / relative


def _successful_seal_for_path(
    path: Path,
    *,
    snapshot: _RunSealSnapshot | None = None,
) -> dict[str, Any] | None:
    """Return the successful attestation entry for an already-resolved path."""
    try:
        parts = path.resolve().relative_to(RUNS_DIR.resolve()).parts
    except (ValueError, OSError):
        return None
    if len(parts) < 2:
        return None
    run_id = parts[0]
    relative = Path(*parts[1:]).as_posix()
    if snapshot is not None and snapshot.run_id == run_id:
        return snapshot.seal_for_relative(relative)
    payload = _run_verification(run_id)
    if not isinstance(payload, dict):
        return None
    if payload.get("status") != "ok" or payload.get("verifier_status") != "ok":
        return None
    if _verification_contract_error(run_id, payload) is not None:
        return {"invalid": True}
    if relative == "run_verification.json":
        # This is the attestation itself and cannot recursively seal its bytes.
        return {"attestation": True}
    artifacts, error = _verified_artifact_map(run_id, payload)
    if error is not None:
        return {"invalid": True}
    return artifacts.get(relative, {"invalid": True})


def _read_integrity_bound_run_bytes(
    path: Path,
    *,
    snapshot: _RunSealSnapshot | None = None,
) -> bytes | None:
    """Read one stable file snapshot and enforce a successful seal if present."""
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return None
    with os.fdopen(descriptor, "rb") as stream:
        file_stat = os.fstat(stream.fileno())
        if not stat.S_ISREG(file_stat.st_mode):
            return None
        data = stream.read()
    # `snapshot=None` keeps the older monkeypatch-friendly positional call.
    seal = (
        _successful_seal_for_path(path, snapshot=snapshot)
        if snapshot is not None
        else _successful_seal_for_path(path)
    )
    if isinstance(seal, dict) and seal.get("invalid") is True:
        return None
    if (
        isinstance(seal, dict)
        and seal.get("attestation") is not True
        and (
            len(data) != seal.get("bytes")
            or not secrets.compare_digest(
                hashlib.sha256(data).hexdigest(), str(seal.get("sha256"))
            )
        )
    ):
        return None
    return data


def _settled_status(status: str, verification: dict[str, Any]) -> str:
    """Refine a finished run's status with the verifier's verdict."""
    if status != "completed":
        return status
    if not verification["checked"]:
        return "unverified"
    return "completed" if verification["ok"] else "verification_failed"


def list_runs() -> list[dict[str, Any]]:
    _refresh_jobs()
    rows: list[dict[str, Any]] = []
    durable_records = _coordinator().list_jobs(limit=1_000)
    durable_by_run = {
        str(payload["run_id"]): record
        for record in durable_records
        if isinstance((payload := record.get("payload")), dict)
        and isinstance(payload.get("run_id"), str)
        and payload.get("run_id")
    }
    directories: list[Path] = []
    if RUNS_DIR.exists():
        try:
            directories = sorted(
                (
                    path
                    for path in RUNS_DIR.iterdir()
                    if path.is_dir() and not path.name.startswith(".")
                ),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            directories = []
    for directory in directories[:100]:
        run_id = directory.name
        summary = _run_summary(run_id)
        active = _active_job_for_run(run_id)
        durable_record = durable_by_run.get(run_id)
        if summary is None and active is None and durable_record is None:
            continue
        rows.append(
            _summary_card(
                run_id,
                summary,
                _lifecycle_status(active, durable_record),
                analysis_kind=_analysis_kind(durable_record),
            )
        )
    known_ids = {row["run_id"] for row in rows}
    for record in durable_records:
        payload = record.get("payload")
        run_id = payload.get("run_id") if isinstance(payload, dict) else None
        if (
            record["kind"] != "run"
            or not isinstance(run_id, str)
            or not run_id
            or run_id in known_ids
        ):
            continue
        rows.insert(
            0,
            _summary_card(
                run_id,
                None,
                str(record["status"]),
                analysis_kind=_analysis_kind(record),
            ),
        )
        known_ids.add(run_id)
    return rows[:100]


# Extensions the run directory may serve directly. Everything a wet-lab reader
# needs to take away is here; anything else stays behind the registered-artifact
# path so this endpoint cannot become a general file server.
RUN_FILE_CONTENT_TYPES: dict[str, str] = {
    ".json": "application/json; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
    ".csv": "text/csv; charset=utf-8",
    ".tsv": "text/tab-separated-values; charset=utf-8",
    ".log": "text/plain; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".pdb": "chemical/x-pdb",
    ".sdf": "chemical/x-mdl-sdfile",
    ".pdbqt": "text/plain; charset=utf-8",
}


DIAGNOSTIC_RUN_FILES = frozenset({"run_verification.json", "run_verification.log"})


def _verification_claims_success(verification: dict[str, Any] | None) -> bool:
    return (
        isinstance(verification, dict)
        and verification.get("status") == "ok"
        and verification.get("verifier_status") == "ok"
    )


def _untrusted_diagnostic(run_id: str, relative: str) -> bool:
    """Only the untrusted diagnostic channel may serve this file.

    A run that claims success but fails its contract must not hand out
    scientific artifacts. Its attestation and verification log are exactly what
    a reader needs to understand that failure, so they stay reachable - clearly
    marked as untrusted diagnostics instead of sealed results.
    """
    if relative not in DIAGNOSTIC_RUN_FILES:
        return False
    verification = _run_verification(run_id)
    if not _verification_claims_success(verification):
        return False
    return _verification_contract_error(run_id, verification) is not None


def _path_is_untrusted_diagnostic(path: Path) -> bool:
    try:
        parts = path.resolve().relative_to(RUNS_DIR.resolve()).parts
    except (ValueError, OSError):
        return False
    if len(parts) < 2:
        return False
    return _untrusted_diagnostic(parts[0], Path(*parts[1:]).as_posix())


def _file_is_withheld(run_id: str, relative: str) -> bool:
    """True when a scientific artifact exists but the broken seal blocks it."""
    if relative in DIAGNOSTIC_RUN_FILES:
        return False
    verification = _run_verification(run_id)
    if not _verification_claims_success(verification):
        return False
    return _verification_contract_error(run_id, verification) is not None


def _run_file_for_download(
    run_id: str,
    relative: str,
    *,
    snapshot: _RunSealSnapshot | None = None,
) -> Path | None:
    """Resolve a run-relative path, or None if it escapes or is not servable.

    Nothing in production calls the coordinator's promote API, so every artifact
    row rendered as "등록 필요" with no link and a reader could not get a single
    file out of the browser. Serving straight from the run directory needs the
    same containment the registered path has: no absolute paths, no traversal,
    no symlink escape, and a closed extension set.

    A caller that touches several files in one request can pass a
    `_RunSealSnapshot` so the seal is verified once instead of per file.
    """
    try:
        safe_run_id = _safe_run_id(run_id)
    except ValueError:
        return None
    decoded = unquote(relative or "")
    candidate = Path(decoded)
    if (
        not decoded
        or "\\" in decoded
        or candidate.is_absolute()
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        return None
    run_dir = (RUNS_DIR / safe_run_id).resolve()
    try:
        resolved = (run_dir / candidate).resolve()
        resolved.relative_to(run_dir)
    except (ValueError, OSError):
        return None
    # Reject symlinks pointing outside the run directory.
    if resolved.is_symlink():
        real_target = resolved.resolve()
        try:
            real_target.relative_to(run_dir)
        except (ValueError, OSError):
            return None
    if not resolved.is_file():
        return None
    if resolved.suffix.lower() not in RUN_FILE_CONTENT_TYPES:
        return None
    if snapshot is not None and snapshot.run_id == safe_run_id:
        claims_success = snapshot.claims_success
        contract_error = snapshot.contract_error
    else:
        verification = _run_verification(safe_run_id)
        claims_success = _verification_claims_success(verification)
        contract_error = (
            _verification_contract_error(safe_run_id, verification)
            if claims_success
            else None
        )
    if claims_success:
        # Once a record asserts success, any seal or identity drift fails
        # closed. Runs with no successful attestation remain available for
        # explicitly unverified diagnostic browsing.
        if contract_error is not None:
            # The scientific outputs are not trustworthy, but the attestation
            # and its log are the failure evidence itself; keep those on an
            # explicitly untrusted diagnostic path.
            return resolved if decoded in DIAGNOSTIC_RUN_FILES else None
        if decoded != "run_verification.json" and _verified_run_file(
            safe_run_id, decoded, snapshot=snapshot
        ) is None:
            return None
    return resolved


# What a reader would want to take away or send to a colleague. The viewer is
# included whole because it is self-contained and opens from a file:// URL.
BUNDLE_ROOT_FILES = (
    "run_summary.json",
    "run_summary.md",
    "run_verification.json",
    "run_verification.log",
)
BUNDLE_MAX_BYTES = 256 * 1024 * 1024


def _run_bundle_zip(run_id: str) -> bytes | None:
    """Zip a run's readable outputs, or None when there is nothing to send."""
    try:
        safe_run_id = _safe_run_id(run_id)
    except ValueError:
        return None
    run_dir = (RUNS_DIR / safe_run_id).resolve()
    if not run_dir.is_dir():
        return None
    # One request, one seal verification: member reads below reuse this
    # snapshot instead of re-hashing every verified artifact per member.
    seal_snapshot = _RunSealSnapshot(safe_run_id)

    members: list[tuple[Path, str]] = []
    total = 0
    for name in BUNDLE_ROOT_FILES:
        resolved = _run_file_for_download(safe_run_id, name, snapshot=seal_snapshot)
        if resolved is not None:
            members.append((resolved, name))
            total += resolved.stat().st_size
    viewer_dir = run_dir / "viewer"
    if viewer_dir.is_dir():
        for path in sorted(viewer_dir.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            try:
                relative = path.resolve().relative_to(run_dir).as_posix()
            except (ValueError, OSError):
                continue
            size = path.stat().st_size
            if total + size > BUNDLE_MAX_BYTES:
                break
            members.append((path, relative))
            total += size
    if not members:
        return None

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, relative in members:
            data = _read_integrity_bound_run_bytes(path, snapshot=seal_snapshot)
            if data is None:
                return None
            archive.writestr(f"{safe_run_id}/{relative}", data)
    # Every member matched the snapshot when read. If any sealed artifact
    # changed while the request was in flight, fail instead of returning a
    # bundle that mixes verified and post-verification state.
    if not seal_snapshot.unchanged():
        return None
    return buffer.getvalue()


def _run_file_row(run_id: str, label: str, relative: str) -> dict[str, Any]:
    """One artifact row, with a working link when the file is really there.

    The trust label is part of the row: a sealed scientific artifact, an
    ordinary unverified output, and a diagnostic file from a run whose success
    attestation is broken must not look alike in the UI or its API.
    """
    resolved = _run_file_for_download(run_id, relative)
    if resolved is None:
        withheld = _file_is_withheld(run_id, relative)
        return {
            "label": label,
            "path": relative,
            "status": "blocked" if withheld else "missing",
            "size_bytes": None,
            "extension": Path(relative).suffix.lower(),
            "detail": (
                "성공 검증 계약이 깨져 차단된 과학 산출물입니다."
                if withheld
                else None
            ),
            "download_url": None,
        }
    verification = _verification_state(run_id)
    untrusted = _untrusted_diagnostic(run_id, relative)
    trust_status = (
        "untrusted_diagnostic"
        if untrusted
        else "verified"
        if verification["ok"]
        else "unverified"
    )
    return {
        "label": label,
        "path": relative,
        "status": "available",
        "size_bytes": resolved.stat().st_size,
        "extension": resolved.suffix.lower(),
        "verification_status": trust_status,
        "trust": trust_status,
        "untrusted": untrusted,
        "detail": (
            "검증 계약이 깨진 실행의 진단 파일입니다. 과학적 결과로 인용하지 마세요."
            if untrusted
            else None
        ),
        "download_url": (
            f"/api/runs/{quote(run_id)}/file?path={quote(relative)}"
        ),
    }


def _artifact_rows(run_id: str, summary: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
    run_dir = RUNS_DIR / run_id
    rows: list[dict[str, Any]] = []
    for label, relative in sorted(artifacts.items()):
        if not isinstance(relative, str):
            continue
        candidate = (run_dir / relative).resolve()
        try:
            candidate.relative_to(run_dir.resolve())
        except ValueError:
            rows.append({"label": label, "path": relative, "status": "blocked", "detail": "허용되지 않은 경로"})
            continue
        rows.append(_run_file_row(run_id, label, relative))
    for label, filename in (
        ("run_summary", "run_summary.json"),
        ("run_summary_markdown", "run_summary.md"),
        ("run_verification", "run_verification.json"),
        ("run_verification_log", "run_verification.log"),
    ):
        if not any(row["path"] == filename for row in rows):
            rows.append(_run_file_row(run_id, label, filename))
    substitute_dir = run_dir / SUBSTITUTE_RELATIVE_DIR
    if substitute_dir.exists():
        for label, filename in SUBSTITUTE_ARTIFACTS:
            candidate = substitute_dir / filename
            if not candidate.is_file():
                continue
            relative = (SUBSTITUTE_RELATIVE_DIR / filename).as_posix()
            if any(row["path"] == relative for row in rows):
                continue
            rows.append(_run_file_row(run_id, label, relative))
    return rows


def _registered_artifact_rows(run_id: str) -> list[dict[str, Any]]:
    record = _durable_job_for_run(run_id)
    if record is None:
        return []
    rows: list[dict[str, Any]] = []
    for artifact in _coordinator().list_artifacts(str(record["job_id"])):
        payload = _registered_artifact_payload(artifact)
        namespace = str(payload.get("namespace") or "artifact")
        for file_record in payload["files"]:
            relative = str(file_record["path"])
            rows.append({
                "label": f"{namespace}:{relative}",
                "path": relative,
                "status": "available",
                "size_bytes": file_record.get("bytes"),
                "extension": Path(relative).suffix.lower(),
                "sha256": file_record.get("sha256"),
                "artifact_id": payload["artifact_id"],
                "download_url": file_record["download_url"],
            })
    return rows


def _target_ranking_path(run_id: str, summary: dict[str, Any]) -> Path | None:
    artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
    relative = artifacts.get("target_ranking") or "03_targets/ranked_targets_v3_with_efficacy.csv"
    if not isinstance(relative, str):
        return None
    run_dir = RUNS_DIR / run_id
    candidate = (run_dir / relative).resolve()
    try:
        candidate.relative_to(run_dir.resolve())
    except ValueError:
        return None
    if not candidate.is_file():
        return None
    verification = _verification_state(run_id)
    if verification["ok"]:
        return _verified_run_file(run_id, relative)
    return candidate


def _target_rows(
    run_id: str,
    summary: dict[str, Any],
    *,
    query: str = "",
    skin_tier: str = "all",
    source: str = "all",
    sort: str = "final_score",
    limit: int = 50,
) -> dict[str, Any]:
    failure_state = _failure_mode_state()
    warnings = _warnings_payload(failure_state)
    curated_failure_modes = failure_state["records"]
    path = _target_ranking_path(run_id, summary)
    if path is None:
        top = summary.get("target_prediction", {}).get("top_targets", []) if isinstance(summary.get("target_prediction"), dict) else []
        rows = [row for row in top if isinstance(row, dict)]
        return {"rows": rows[:limit], "total": len(rows), "source_path": None, "exploratory": True, "warnings": warnings, "verification": _verification_state(run_id)}
    metadata = _load_metadata()
    query_lower = query.strip().lower()
    filtered: list[dict[str, Any]] = []
    data = _read_integrity_bound_run_bytes(path)
    if data is None:
        return {"rows": [], "total": 0, "source_path": _relative_path(path), "exploratory": True, "warnings": warnings, "verification": _verification_state(run_id), "error": "ranking 파일의 무결성을 확인하지 못했습니다."}
    try:
        with io.StringIO(data.decode("utf-8", errors="replace"), newline="") as stream:
            reader = csv.DictReader(stream)
            for rank, raw in enumerate(reader, start=1):
                target_id = (raw.get("target_id") or "").strip()
                if not target_id:
                    continue
                labels = metadata.get(target_id, {})
                sources = [item for item in (raw.get("sources") or "").split(";") if item]
                efficacy = [
                    value.strip()
                    for key, value in raw.items()
                    if key.startswith("efficacy_") and value and value.strip()
                ]
                row = {
                    "rank": rank,
                    "target_id": target_id,
                    "gene_symbol": labels.get("gene_symbol") or raw.get("gene_symbol") or target_id,
                    "protein_name": labels.get("protein_name") or raw.get("protein_name") or "",
                    "final_score": _as_float(raw.get("final_score")),
                    "final_score_semantics": (raw.get("final_score_semantics") or "unknown").strip(),
                    "ranking_basis": (raw.get("ranking_basis") or "unknown").strip(),
                    "docking_rrf": _as_float(raw.get("docking_rrf")),
                    "skin_score": _as_float(raw.get("skin_score")),
                    "skin_tier": (raw.get("skin_tier") or "unknown").strip(),
                    "cell_type_preferred": (
                        raw.get("cell_type_preferred") or "unknown"
                    ).strip(),
                    "source_count": int(float(raw.get("source_count") or 0)),
                    # Retrieval evidence. Absent on runs that predate the
                    # fields, so the drawer must render without them.
                    "daina_max_tanimoto": _as_float(raw.get("daina_max_tanimoto")),
                    "daina_supporting_molecule_id": (
                        raw.get("daina_supporting_molecule_id") or ""
                    ).strip(),
                    # What the nearest analogue's own measurement said under the
                    # index's threshold policy. The rank does not use it, so the
                    # drawer shows it; runs predating the column send "unknown".
                    "daina_supporting_evidence": (
                        raw.get("daina_supporting_evidence") or "unknown"
                    ).strip(),
                    "daina_known_ligand_count": (
                        int(float(raw["daina_known_ligand_count"]))
                        if str(raw.get("daina_known_ligand_count") or "").strip()
                        else None
                    ),
                    "daina_is_self_match": (
                        str(raw["daina_is_self_match"]).strip().lower() == "true"
                        if str(raw.get("daina_is_self_match") or "").strip()
                        else None
                    ),
                    "failure_modes": failure_modes_for(
                        target_id,
                        molecular_function=labels.get("molecular_function", ""),
                        curated=curated_failure_modes,
                    ),
                    "sources": sources,
                    "efficacy": efficacy,
                }
                haystack = " ".join(
                    str(row.get(key, "")) for key in ("target_id", "gene_symbol", "protein_name")
                ).lower()
                if query_lower and query_lower not in haystack:
                    continue
                if skin_tier != "all" and row["skin_tier"] != skin_tier:
                    continue
                if source != "all" and source not in sources:
                    continue
                filtered.append(row)
    except (OSError, csv.Error, ValueError):
        return {"rows": [], "total": 0, "source_path": _relative_path(path), "exploratory": True, "warnings": warnings, "error": "ranking 파일을 읽지 못했습니다."}
    if sort == "skin_score":
        filtered.sort(key=lambda row: row["skin_score"] if row["skin_score"] is not None else -1, reverse=True)
    elif sort == "docking_rrf":
        filtered.sort(key=lambda row: row["docking_rrf"] if row["docking_rrf"] is not None else -1, reverse=True)
    elif sort == "source_count":
        filtered.sort(key=lambda row: row["source_count"], reverse=True)
    else:
        filtered.sort(key=lambda row: row["final_score"] if row["final_score"] is not None else -1, reverse=True)
    # `rank` stays the position in the full ranking file. Renumbering it from 1
    # after a filter made a globally 5th target read as the top hit.
    for position, row in enumerate(filtered, start=1):
        row["filtered_position"] = position
        row["original_rank"] = row["rank"]
    return {
        "rows": filtered[: max(1, min(limit, 200))],
        "total": len(filtered),
        "source_path": _relative_path(path),
        "exploratory": True,
        "warnings": warnings,
        "verification": _verification_state(run_id),
    }


def get_run(run_id: str) -> dict[str, Any] | None:
    try:
        run_id = _safe_run_id(run_id)
    except ValueError:
        return None
    run_dir = RUNS_DIR / run_id
    summary = _run_summary(run_id)
    active = _active_job_for_run(run_id)
    durable_record = _durable_job_for_run(run_id)
    if not run_dir.exists() and active is None and durable_record is None:
        return None
    if summary is None:
        durable_job = (
            _durable_job_adapter(durable_record)
            if durable_record is not None
            else None
        )
        status = (
            active.status
            if active is not None
            else str(durable_record["status"])
            if durable_record is not None
            else "incomplete"
        )
        return {
            "run": _summary_card(
                run_id,
                None,
                status,
                analysis_kind=_analysis_kind(durable_record),
            ),
            "summary": None,
            "verification": None,
            "artifacts": _registered_artifact_rows(run_id),
            "logs": _tail(active.log_path if active else LOG_DIR / f"{run_id}.log"),
            "targets": {"rows": [], "total": 0, "source_path": None, "exploratory": True},
            "substitutes": _json_value(_substitute_report(run_id)),
            "job": (
                _job_payload(active)
                if active is not None
                else durable_job.public()
                if durable_job is not None
                else None
            ),
        }
    verification = _read_json(run_dir / "run_verification.json")
    substitutes = _substitute_report(run_id)
    return {
        "run": _summary_card(
            run_id,
            summary,
            _lifecycle_status(active, durable_record),
            analysis_kind=_analysis_kind(durable_record),
        ),
        "summary": _json_value(summary),
        "verification": _json_value(verification),
        "artifacts": _registered_artifact_rows(run_id) + _artifact_rows(run_id, summary),
        "logs": _tail(active.log_path if active else LOG_DIR / f"{run_id}.log"),
        "targets": _target_rows(run_id, summary, limit=50),
        "substitutes": _json_value(substitutes),
        "job": _job_payload(active) if active else None,
    }


def _installation_payload() -> dict[str, Any]:
    status = get_status()
    system = status["system"]
    if system["ubuntu_ready"]:
        os_guidance = {
            "title": "Ubuntu 환경이 감지되었습니다.",
            "body": "현재 컴퓨터의 OS 디스크는 건드리지 않고 SkinScout 실행 환경만 준비합니다.",
            "steps": ["환경 준비 버튼을 누릅니다.", "설치 로그를 확인합니다.", "완료 후 안전성 분석 또는 target 분석을 시작합니다."],
            "command": None,
        }
    elif system["platform"] == "Windows":
        os_guidance = {
            "title": "Windows에서는 Ubuntu on WSL2가 필요합니다.",
            "body": "WSL2 설치는 관리자 권한과 재부팅이 필요할 수 있습니다. Workbench는 이 작업을 몰래 실행하지 않습니다.",
            "steps": ["PowerShell을 관리자 권한으로 엽니다.", "Microsoft의 WSL2/Ubuntu 안내를 따라 설치합니다.", "Ubuntu 터미널을 한 번 실행한 뒤 Workbench를 다시 엽니다."],
            "command": "wsl --install -d Ubuntu",
        }
    else:
        os_guidance = {
            "title": "지원되는 Ubuntu 실행 환경이 아직 감지되지 않았습니다.",
            "body": "Ubuntu Desktop, Ubuntu on WSL2, 또는 연구실 표준 VM을 준비한 뒤 다시 확인하세요.",
            "steps": ["기관의 Ubuntu 설치 정책을 확인합니다.", "VM 또는 부팅 USB를 사용합니다.", "Ubuntu에서 Workbench를 다시 실행합니다."],
            "command": None,
        }
    return {"status": status, "os_guidance": os_guidance}


def _start_setup_install(profile: str = "demo") -> Job:
    profile = str(profile or "demo").strip().lower()
    if profile not in SETUP_PROFILES:
        raise ValueError("설치 프로필은 demo 또는 full만 선택할 수 있습니다.")
    status = get_status(force=True)
    if not status["system"]["ubuntu_ready"]:
        raise PermissionError("Ubuntu 환경에서만 SkinScout 기본 환경 자동 준비를 시작할 수 있습니다.")
    for record in _coordinator().list_jobs(
        statuses={"queued", "running", "cancel_requested"},
        limit=1_000,
    ):
        if record["kind"] == "setup":
            raise RuntimeError("이미 실행 중인 환경 준비 작업이 있습니다.")
    command = [
        shutil.which("bash") or "/bin/bash",
        str(ROOT / "scripts" / "bootstrap_runtime.sh"),
        "--runtime",
        "conda",
    ]
    if profile == "full":
        command.extend(["--profile", "full"])
    return _start_process_job(
        kind="setup",
        run_id=None,
        command=command,
        log_path=LOG_DIR / f"setup_{int(time.time())}.log",
        resource="host:setup",
    )


def _start_stage0(cores: int) -> Job:
    status = get_status(force=True)
    if status["stage0"]["status"] == "ready":
        raise RuntimeError("Stage 0 데이터가 이미 준비되어 있습니다.")
    if not status.get("disk", {}).get("stage0_ready", False):
        free_gb = status.get("disk", {}).get("free_gb", "알 수 없음")
        raise RuntimeError(
            f"Stage 0에는 최소 {STAGE0_MIN_FREE_GB} GB의 여유 공간이 필요합니다 "
            f"(현재 {free_gb} GB)."
        )
    if cores < 1 or cores > 256:
        raise ValueError("cores는 1에서 256 사이여야 합니다.")
    run_id = "stage0_bootstrap"
    active = _active_job_for_run(run_id)
    if active:
        raise RuntimeError("Stage 0 준비가 이미 실행 중입니다.")
    command = [
        *_python_command(),
        str(ROOT / "scripts" / "run_skinscout.py"),
        "--preset",
        "stage0",
        "--run-id",
        run_id,
        "--cores",
        str(cores),
        "--allow-stage0-build",
    ]
    return _start_process_job(
        kind="stage0",
        run_id=run_id,
        command=command,
        log_path=LOG_DIR / f"{run_id}.log",
    )


def _require_discovery_alias_package() -> None:
    try:
        check = stage0_verify.chk_discovery_alias_integrity(ROOT)
        if not check.ok:
            raise ValueError(check.detail)
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            "Discovery 분석에는 검증된 Stage 0 화합물 별칭 데이터가 필요합니다. "
            f"Workbench에서 Stage 0 데이터를 먼저 준비하세요: {exc}"
        ) from exc


# What a reader needs to see before committing hours of compute: the molecule
# the parser actually read, and whether the tool can say anything about it.
PREVIEW_PROPERTY_LABELS: dict[str, str] = {
    "molecular_weight": "분자량",
    "logp": "logP",
    "tpsa": "TPSA",
    "hbd": "수소결합 주개",
    "hba": "수소결합 받개",
    "rotatable_bonds": "회전 가능 결합",
    "heavy_atoms": "무거운 원자",
    "rings": "고리",
    "fraction_csp3": "sp3 탄소 비율",
    "qed": "QED",
}
PREVIEW_VERDICT_LABELS = {
    "in_scope": "분석 가능",
    "review": "분석 가능 (확인 필요)",
    "out_of_scope": "분석할 수 없음",
    "invalid": "구조를 읽지 못했습니다",
}


NAME_INDEX_PATH = ROOT / "data" / "compound_names" / "name_index.csv"
NAME_SEARCH_LIMIT = 12
_NAME_INDEX_CACHE: list[dict[str, str]] | None = None
_NAME_INDEX_LOCK = threading.Lock()
# Matches the builder: an ASCII-only class erases Hangul, which is most of what
# this tool's reader types.
_NAME_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)


def _normalise_name(value: str) -> str:
    return re.sub(r"\s+", " ", _NAME_NON_WORD.sub(" ", value.strip().lower())).strip()


def _name_index() -> list[dict[str, str]]:
    """Offline name -> SMILES index, loaded once.

    The Workbench took SMILES and nothing else, so a reader had to find a
    structure string before they could ask the tool anything at all.
    """
    global _NAME_INDEX_CACHE
    if _NAME_INDEX_CACHE is not None:
        return _NAME_INDEX_CACHE
    with _NAME_INDEX_LOCK:
        if _NAME_INDEX_CACHE is not None:
            return _NAME_INDEX_CACHE
        rows: list[dict[str, str]] = []
        if NAME_INDEX_PATH.is_file():
            try:
                with NAME_INDEX_PATH.open(encoding="utf-8", newline="") as handle:
                    for record in csv.DictReader(handle):
                        if record.get("normalised") and record.get("smiles"):
                            rows.append({
                                "normalised": record["normalised"],
                                "display_name": record.get("display_name", ""),
                                "smiles": record["smiles"],
                                "inchikey": record.get("inchikey", ""),
                                "source": record.get("source", ""),
                            })
            except (OSError, csv.Error):
                rows = []
        _NAME_INDEX_CACHE = rows
        return rows


def compound_search(query: str) -> dict[str, Any]:
    """Resolve an ingredient name to a structure, exact matches first."""
    normalised = _normalise_name(query)
    if len(normalised) < 2:
        return {"query": query, "matches": [], "available": bool(_name_index())}
    index = _name_index()
    exact: list[dict[str, str]] = []
    prefix: list[dict[str, str]] = []
    contains: list[dict[str, str]] = []
    for row in index:
        name = row["normalised"]
        if name == normalised:
            exact.append(row)
        elif name.startswith(normalised):
            prefix.append(row)
        elif normalised in name:
            contains.append(row)
    ordered = (
        exact
        + sorted(prefix, key=lambda row: len(row["display_name"]))
        + sorted(contains, key=lambda row: len(row["display_name"]))
    )
    seen: set[str] = set()
    matches: list[dict[str, Any]] = []
    for row in ordered:
        key = row["inchikey"] or row["smiles"]
        if key in seen:
            continue
        seen.add(key)
        matches.append({
            "name": row["display_name"],
            "smiles": row["smiles"],
            "inchikey": row["inchikey"],
            "source": row["source"],
            "exact": row["normalised"] == normalised,
        })
        if len(matches) >= NAME_SEARCH_LIMIT:
            break
    return {"query": query, "matches": matches, "available": bool(index)}


def _depiction_svg(molecule: Any) -> str | None:
    """A 2D drawing of what the parser read, so a wrong molecule is visible."""
    try:
        from rdkit.Chem import rdDepictor
        from rdkit.Chem.Draw import rdMolDraw2D
    except ImportError:  # pragma: no cover - RDKit is a hard dependency
        return None
    try:
        rdDepictor.Compute2DCoords(molecule)
        drawer = rdMolDraw2D.MolDraw2DSVG(360, 260)
        drawer.drawOptions().clearBackground = False
        drawer.DrawMolecule(molecule)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception:  # noqa: BLE001 - a drawing failure must not block a run
        return None


_SIMILARITY_INDEX: Any = None
_SIMILARITY_LOCK = threading.Lock()
_INGREDIENT_LIBRARY: Any = None
_INGREDIENT_LOCK = threading.Lock()

# 두 목록 모두 이 개수까지만. MCS 판정이 후보당 한 번씩 돌기 때문에, 상한 없이
# 열어 두면 한 요청이 워커를 몇 초씩 붙잡는다.
MAX_ALTERNATIVE_ROWS = 50


def _similarity_index() -> Any:
    """Load once, keep it. 272 MB and ~3 s, so it is paid on first use only.

    Loading eagerly at start-up would make every launch slower for a feature
    most sessions never touch, and loading per request would make each lookup
    take seconds instead of half a second.

    성공만 기억한다. 실패까지 기억하면, 인덱스를 만든 뒤에도 서버를 다시 띄우기
    전까지 화면이 계속 "준비되지 않았습니다"라고 말한다 - 사용자가 시킨 대로
    했는데도 아무것도 달라지지 않는 상태가 된다. 실패 경로는 파일이 없는지 보는
    것뿐이라 매번 다시 시도해도 싸다.
    """
    global _SIMILARITY_INDEX
    if _SIMILARITY_INDEX is not None:
        return _SIMILARITY_INDEX
    with _SIMILARITY_LOCK:
        if _SIMILARITY_INDEX is not None:
            return _SIMILARITY_INDEX
        sys.path.insert(0, str(ROOT / "scripts"))
        try:
            from similar_compounds import load_index

            _SIMILARITY_INDEX = load_index()
        except (SystemExit, OSError, ValueError) as exc:
            raise ValueError(
                "활성 측정 라이브러리가 준비되지 않았습니다. "
                "scripts/build_similarity_index.py 를 먼저 실행하세요."
            ) from exc
    return _SIMILARITY_INDEX


def _ingredient_library() -> Any:
    """CosIng 원료 라이브러리. 507종이라 3초가 아니라 0.3초에 열린다.

    인덱스와 마찬가지로 성공만 기억한다.
    """
    global _INGREDIENT_LIBRARY
    if _INGREDIENT_LIBRARY is not None:
        return _INGREDIENT_LIBRARY
    with _INGREDIENT_LOCK:
        if _INGREDIENT_LIBRARY is not None:
            return _INGREDIENT_LIBRARY
        sys.path.insert(0, str(ROOT / "scripts"))
        try:
            from alternative_ingredients import load_ingredient_library

            _INGREDIENT_LIBRARY = load_ingredient_library()
        except (FileNotFoundError, SystemExit, OSError, ValueError) as exc:
            raise ValueError(
                "화장품 원료 라이브러리가 준비되지 않았습니다. "
                "scripts/stage0_cosing.py 로 data/cosing/cosing.parquet 을 먼저 만드세요."
            ) from exc
    return _INGREDIENT_LIBRARY


def _bounded_int(value: Any, default: int, low: int, high: int) -> int:
    """JSON은 Infinity와 NaN을 실어 보낼 수 있고, int()는 둘 다에서 터진다.

    OverflowError를 잡지 않으면 `{"limit": Infinity}` 하나로 요청 처리 자체가
    끊겨 응답이 나가지 않는다.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:  # NaN
        return default
    if number == float("inf"):
        return high
    if number == float("-inf"):
        return low
    try:
        return max(low, min(high, int(number)))
    except (TypeError, ValueError, OverflowError):
        return default


def _bounded_float(value: Any, default: float, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:
        return default
    return max(low, min(high, number))


def _flag(value: Any) -> bool:
    """`"false"`도 문자열이라 bool()에서는 참이다. 화면이 보내지 않는 값이라도,
    스크립트에서 그렇게 부르면 필터가 조용히 켜진다."""
    if isinstance(value, str):
        return value.strip().lower() not in ("", "false", "0", "no", "off")
    return bool(value)


# 라이브러리를 훑은 **뒤에야** 존재하는 정렬 기준. `scan_alternatives`의
# SORT_MODES(core·merged·polarity)는 구조 신호만으로 줄을 세우지만, 아래
# 기준들은 ADMET 예측과 종합 점수가 붙은 다음에야 값이 생긴다. 여기서 서버가
# 전체 후보를 정렬한 뒤에 자르므로, 화면이 요청한 순서의 진짜 상위가 나간다.
# (열 이름, 클수록 좋은가) - `workbench/static/app.js`의 CLIENT_SORTS 와 같은 짝.
POST_SCAN_SORTS: dict[str, tuple[str, bool]] = {
    "score": ("score_total", True),
    "safety": ("score_safety", True),
    "evidence": ("score_evidence", True),
    "similarity": ("similarity", True),
    "skin": ("Skin_Reaction", False),
    "logp": ("logP", False),
}

ADMET_CACHE_PATH = Path("data/cosing/admet_cache.parquet")
_ADMET_CACHE: "Any | None" = None
_ADMET_CACHE_LOADED = False


def _admet_cache() -> "Any | None":
    """미리 계산해 둔 원료별 ADMET. 없으면 None 이고, 그때는 열이 붙지 않는다.

    요청마다 예측하지 않는 이유는 시간이 아니라 일관성이다 - 같은 원료가 요청
    때마다 다른 값을 받으면 정렬이 흔들린다. 캐시는 scripts/build_admet_cache.py
    가 만든다(실측 8,716종에 약 3분).
    """
    global _ADMET_CACHE, _ADMET_CACHE_LOADED
    if _ADMET_CACHE_LOADED:
        return _ADMET_CACHE
    _ADMET_CACHE_LOADED = True
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        import pandas as pd

        if ADMET_CACHE_PATH.exists():
            _ADMET_CACHE = pd.read_parquet(ADMET_CACHE_PATH)
    except Exception:                             # noqa: BLE001 - 캐시가 없어도 돌아야 한다
        # 캐시를 못 읽으면 ADMET 열 없이 검색만 돌린다. 화면에는 안전 축이
        # 중앙값으로 채워졌다는 것이 `admet_available: false` 로 나간다.
        _ADMET_CACHE = None
    return _ADMET_CACHE


def attach_admet(frame: "Any") -> "Any":
    """후보 표에 ADMET 열을 붙인다. 캐시가 없으면 원본을 그대로 돌려준다."""
    cache = _admet_cache()
    if cache is None or "inchikey" not in getattr(frame, "columns", []):
        return frame
    overlap = [c for c in cache.columns if c != "inchikey" and c in frame.columns]
    joinable = cache.drop(columns=overlap) if overlap else cache
    return frame.merge(joinable, on="inchikey", how="left")


def score_rows(frame: "Any") -> "Any":
    """축별 점수와 종합 순위를 붙인다. 실패해도 검색 결과 자체는 살린다."""
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from candidate_score import score_candidates

        return score_candidates(frame)
    except Exception:                             # noqa: BLE001
        # 점수를 못 내도 검색 결과 자체는 살린다. 점수 열이 없으면 화면은
        # 기존 정렬(유사도)로 떨어진다.
        return frame


# 컨포머 생성은 원자 수에 따라 수십 ms~수 초다. 후보를 열 때마다 만들되,
# 같은 분자를 다시 열면 다시 만들지 않는다. 화면에서 이리저리 눌러 보는 것이
# 이 기능의 쓰임이라 재계산이 곧 체감 지연이 된다.
_STRUCTURE_3D_CACHE: dict[str, dict[str, Any]] = {}
STRUCTURE_3D_CACHE_LIMIT = 256
_STRUCTURE_3D_CACHE_LOCK = threading.Lock()


def compound_structure_3d(payload: dict[str, Any]) -> dict[str, Any]:
    """SMILES 하나를 3D 좌표(MOL 블록)로. 화면의 Mol* 가 그대로 읽는다.

    2D 그림과 달리 3D 는 **생성한 좌표**다. 실험 구조도, 도킹 포즈도 아니다.
    같은 분자라도 컨포머는 여럿이고 여기서는 힘장으로 하나만 고른다. 그 사실이
    화면에서 사라지면 사용자는 이것을 결합 자세로 읽는다.
    """
    smiles = str(payload.get("smiles") or "").strip()
    if not smiles:
        return {"ok": False, "reason": "SMILES 가 비어 있습니다."}
    cached = _STRUCTURE_3D_CACHE.get(smiles)
    if cached is not None:
        return cached

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:                    # noqa: BLE001
        return {"ok": False, "reason": f"RDKit 를 불러오지 못했습니다: {exc}"}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"ok": False, "reason": "SMILES 를 읽지 못했습니다."}
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    # 씨앗을 고정한다. 같은 분자를 다시 열었을 때 다른 배좌가 뜨면 사용자는
    # 무엇이 바뀐 것인지 알 수 없다.
    params.randomSeed = 0xC0FFEE
    if AllChem.EmbedMolecule(mol, params) != 0:
        # 고리가 많거나 입체가 빡빡하면 첫 시도가 실패한다. 무작위 좌표에서
        # 다시 시도하되, 그래도 안 되면 만들지 못했다고 말한다.
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            return {"ok": False, "reason": "3차원 좌표를 만들지 못했습니다."}
    # 반환값 0 은 수렴, 1 은 "최대 반복 안에 수렴하지 않음"이고 -1 이 진짜
    # 실패다. 1 을 실패로 보면 회전 결합이 많은 분자(레티놀 등)가 전부 최적화
    # 안 된 것으로 기록된다 - 실제로는 기본 200회가 모자란 것이다.
    optimised = "none"
    converged = False
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            code = AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
            if code >= 0:
                optimised, converged = "MMFF94", code == 0
        if optimised == "none":
            code = AllChem.UFFOptimizeMolecule(mol, maxIters=2000)
            if code >= 0:
                optimised, converged = "UFF", code == 0
    except Exception:                             # noqa: BLE001 - 최적화 실패는 치명적이지 않다
        optimised, converged = "none", False

    result = {
        "ok": True,
        "format": "mol",
        "data": Chem.MolToMolBlock(mol),
        "atoms": mol.GetNumAtoms(),
        "force_field": optimised,
        # 수렴하지 않아도 좌표는 쓸 만하다. 다만 "완전히 이완된 배좌"는 아니다.
        "converged": converged,
        # 화면에 그대로 띄울 한 줄. 이 좌표가 무엇인지 오해하지 않도록.
        "note": ("계산으로 만든 컨포머 하나입니다. 실험 구조나 결합 자세가 "
                 "아니며, 같은 분자도 다른 배좌를 가질 수 있습니다."),
    }
    with _STRUCTURE_3D_CACHE_LOCK:
        if len(_STRUCTURE_3D_CACHE) >= STRUCTURE_3D_CACHE_LIMIT:
            # Evict oldest entries instead of clearing the entire cache.
            evict_count = max(1, STRUCTURE_3D_CACHE_LIMIT // 4)
            for _ in range(evict_count):
                if _STRUCTURE_3D_CACHE:
                    _STRUCTURE_3D_CACHE.pop(next(iter(_STRUCTURE_3D_CACHE)))
        _STRUCTURE_3D_CACHE[smiles] = result
    return result


def compound_alternatives(payload: dict[str, Any]) -> dict[str, Any]:
    """활성 핵심구조를 유지한 대체소재 후보와, 그 구조가 이미 측정된 기록.

    두 목록을 한 번에 돌려준다. 앞의 것(`ingredients`)은 화장품 원료로 등재된
    것들이라 바로 대체소재 후보가 되고, 뒤의 것(`measured`)은 등재 여부와 무관하게
    무엇에 붙는지 측정된 것들이라 근거가 된다. 어느 쪽도 혼자서는 답이 아니다 -
    등재됐지만 아무도 재 본 적 없는 후보와, 잘 측정됐지만 원료가 아닌 분자는 서로
    다른 결정을 부른다.

    두 목록 모두 유사도가 아니라 핵심구조 유지 판정을 함께 들고 나간다. 유사도만
    보이면 곁사슬만 닮은 분자가 대체소재로 읽힌다.
    """
    from alternative_ingredients import (
        MAX_QUERY_HEAVY_ATOMS,
        annotate_core_retention,
        attach_measured_evidence,
        murcko_core,
        MAX_3D_CANDIDATES,
        SORT_MODES,
        annotate_three_d,
        scan_alternatives,
        signal_disagreement,
    )
    from build_activity_retrieval_index import _standardize_mol
    from rdkit import Chem
    from similar_compounds import find_similar, unique_targets
    import pandas as pd

    smiles = str(payload.get("smiles", "")).strip()
    if not smiles:
        raise ValueError("SMILES를 입력하세요.")
    if len(smiles) > MAX_SMILES_LENGTH:
        raise ValueError(f"SMILES가 너무 깁니다(최대 {MAX_SMILES_LENGTH}자).")

    limit = _bounded_int(payload.get("limit", 20), 20, 1, MAX_ALTERNATIVE_ROWS)
    # 기본값 0. 유사도로 미리 거르면 핵심구조를 유지한 후보가 먼저 버려진다 -
    # 코직산의 파이라논 유도체는 핵심을 80% 유지하는데 유사도가 0.209라, 예전
    # 기본값 0.25에서는 목록이 통째로 비어 있었다.
    floor = _bounded_float(payload.get("min_similarity", 0.0), 0.0, 0.0, 1.0)
    require_core = _flag(payload.get("require_core"))
    exclude_self = _flag(payload.get("exclude_self"))
    include_measured = payload.get("include_measured", True) is not False
    # 파마코포어는 구조 판정과 다른 것을 재는 두 번째 신호다. 첫 요청에 3.7초가
    # 더 들어(507종 지문 생성) 기본은 꺼져 있고, 화면에서 켠다.
    with_pharmacophore = _flag(payload.get("with_pharmacophore"))
    # 3D는 상위 몇 건만 다시 채점한다. 실측으로 유연한 질의는 25건에 28초까지 걸려
    # 전수는 물론이고 기본값으로도 둘 수 없다.
    with_three_d = _flag(payload.get("with_three_d"))
    # CosIng 은 화장품 원료만 담은 목록이 아니다. 금지·제한 물질도 규제 목적으로
    # 등재돼 있고, 그런 항목은 **신고 배합목적이 비어 있다**. 실측으로 7,484종 중
    # 1,215종(16.2%)이 그렇고, 그 안에는 메타조신(마약성 진통제), 헥사클로로
    # 살충제, 아세틸콜린 같은 것이 들어 있다. 그것들을 "대체소재 후보"로 점수와
    # 함께 내놓으면 읽는 쪽은 쓸 수 있는 원료로 읽는다.
    #
    # 그래서 기본으로 숨기되, **몇 건을 숨겼는지 반드시 말한다.** 구조가 금지
    # 물질과 닮았다는 사실 자체는 알 가치가 있으므로 켜서 볼 수 있게 남긴다.
    hide_undeclared = payload.get("include_undeclared", False) is not True
    sort_by = str(payload.get("sort_by") or "core")
    # 점수·ADMET 기준은 라이브러리를 훑은 **뒤에야** 존재하므로 `scan_alternatives`
    # 가 정렬할 수 없다. 예전에는 이런 요청이 조용히 "core"로 떨어지고 화면이
    # 받아 온 20행만 다시 줄 세웠다 - 그래서 "종합 점수 순"이 실제로는
    # "핵심구조 상위 20을 점수로 재배열"이었고, 21위였던 더 좋은 후보는 영영
    # 올라오지 못했다. 이제 전체 후보에 점수를 붙이므로 여기서 진짜로 정렬한다.
    post_sort = POST_SCAN_SORTS.get(sort_by)
    scan_sort = "core" if post_sort else sort_by
    if scan_sort not in SORT_MODES:
        scan_sort = "core"
        if not post_sort:
            sort_by = "core"
    # 합친 순위는 세 기준이 다 있어야 낼 수 있다. 파마코포어를 끈 채로 요청하면
    # 조용히 핵심구조 순으로 돌려주지 않고, 필요한 것을 켠다.
    if scan_sort == "merged":
        with_pharmacophore = True

    try:
        query_mol = _standardize_mol(smiles)
    except (ValueError, RuntimeError) as exc:
        raise ValueError("RDKit이 이 SMILES를 읽지 못했습니다. 오타를 확인하세요.") from exc

    # 크기 검사가 가장 먼저다. Murcko 골격도 InChIKey도 원자 수에 따라 초 단위로
    # 늘어나므로, 뒤에 두면 거절할 입력에 몇 분을 쓰고 나서 거절하게 된다.
    if query_mol.GetNumHeavyAtoms() > MAX_QUERY_HEAVY_ATOMS:
        raise ValueError(
            f"입력 분자가 너무 큽니다(중원자 {query_mol.GetNumHeavyAtoms()}개). "
            f"이 화면은 중원자 {MAX_QUERY_HEAVY_ATOMS}개 이하의 단일 저분자를 다룹니다."
        )

    scaffold, scaffold_basis = murcko_core(query_mol)
    query_info = {
        "canonical_smiles": Chem.MolToSmiles(query_mol),
        "inchikey": Chem.MolToInchiKey(query_mol),
        "heavy_atoms": int(query_mol.GetNumHeavyAtoms()),
        "scaffold_smiles": Chem.MolToSmiles(scaffold) if scaffold is not None else "",
        "acyclic": scaffold is None and scaffold_basis == "acyclic_query",
        "heavy_atom_note": (
            "중원자가 적은 분자일수록 같은 유지율이 더 쉽게 나옵니다. "
            "옆의 비중 값을 함께 보세요."
            if query_mol.GetNumHeavyAtoms() <= 12 else ""
        ),
    }

    result: dict[str, Any] = {"query": query_info}

    # 두 라이브러리는 따로 실패한다. 하나가 없다고 다른 하나의 결과까지 지우면,
    # 준비된 절반조차 못 보게 된다.
    index: Any = None
    index_error: str | None = None
    try:
        index = _similarity_index()
    except ValueError as exc:
        index_error = str(exc)

    library: Any = None
    library_error: str | None = None
    try:
        library = _ingredient_library()
    except ValueError as exc:
        library_error = str(exc)

    if library is None and index is None:
        raise ValueError(library_error or index_error or "라이브러리를 열 수 없습니다.")

    if library is None:
        result["ingredients"] = {
            "rows": [], "unavailable": library_error, "evidence_available": index is not None,
            "scanned": 0, "core_kept": 0, "unjudged": 0, "grade_counts": {}, "matches": 0,
            "summary": "", "library_size": 0, "registered_entries": 0, "resolved_entries": 0,
            "note": "", "min_similarity": floor, "require_core": require_core,
            "query_measured": False,
        }
    else:
        # 파마코포어 백엔드는 RDKit 의 BaseFeatures.fdef 파일과 5.6b 모듈에 기댄다.
        # 그것이 열리지 않는다고 요청 전체를 죽이면, 파마코포어가 전혀 필요 없는
        # 측정 근거 표까지 함께 사라진다 - 위에서 index 와 library 를 따로 실패
        # 시키는 것과 같은 이유로, 세 번째 신호도 따로 실패해야 한다.
        # 정렬을 '합친 순위'로 바꾼 것만으로 화면 전체가 비는 경로였다.
        pharmacophore_error: str | None = None
        try:
            scan = scan_alternatives(
                smiles,
                library,
                min_similarity=floor,
                require_core=require_core,
                exclude_self=exclude_self,
                with_pharmacophore=with_pharmacophore,
                sort_by=scan_sort,
            )
        except (ValueError, ImportError, OSError, AttributeError, KeyError) as exc:
            if not with_pharmacophore:
                raise
            pharmacophore_error = str(exc) or exc.__class__.__name__
            with_pharmacophore = False
            if scan_sort == "merged":
                scan_sort = "core"
                sort_by = "core"
            scan = scan_alternatives(
                smiles,
                library,
                min_similarity=floor,
                require_core=require_core,
                exclude_self=exclude_self,
                with_pharmacophore=False,
                sort_by=scan_sort,
            )
        # 점수는 **대조한 후보 전체**를 기준으로 매긴다. 예전에는 head(limit) 로
        # 자른 뒤에 매겼는데, 종합 점수가 백분위 순위의 가중평균이라 기준 집합이
        # 바뀌면 값도 순서도 바뀐다: 같은 후보가 limit=20 과 limit=100 에서 다른
        # 점수를 받았고(실측 최대 0.046), 20행 안에서의 순서가 전체 기준 순서와
        # 달랐다. 화면에 뜬 0.82 가 무엇에 대한 0.82 인지 말할 수 없는 값이었다.
        #
        # 대가는 `attach_measured_evidence`를 20행이 아니라 전체(7,484행)에
        # 돌리는 것으로 요청당 약 4초다. 3D 좌표 생성만 비싸므로 그것은 자른
        # 뒤에 남겨 둔다.
        scored = scan.frame
        # `query_measured`는 `attach_measured_evidence`가 frame.attrs 에 남긴다.
        # 그런데 pandas 3 의 merge 는 attrs 를 버린다 - 아래 `attach_admet`이
        # merge 라서, 그 뒤에 읽으면 언제나 False 다. 그러면 화면은 측정 기록이
        # 있는 화합물에도 "입력한 화합물 자체에 활성 측정 기록이 없어…"라고 적는다
        # (나이아신아마이드로 재현: 1행 measured_target_count=8 인데 문구는 없다고 함).
        # 그래서 attrs 는 붙인 **직후에** 꺼내 값으로 들고 간다.
        query_measured = False
        if index is not None and not scored.empty:
            scored = attach_measured_evidence(scored, index, smiles)
            query_measured = bool(scored.attrs.get("query_measured"))
        if not scored.empty:
            scored = attach_admet(scored)
            scored = score_rows(scored)
        if post_sort is not None and not scored.empty:
            column, descending = post_sort
            if column in scored.columns:
                # 결측은 언제나 맨 뒤. 화면의 클라이언트 정렬과 같은 규칙이고,
                # 재지 못한 후보를 "가장 좋음"으로 올리지 않는다. 동점은
                # InChIKey 로 갈라 실행마다 순서가 달라지지 않게 한다.
                key = pd.to_numeric(scored[column], errors="coerce")
                scored = (
                    scored.assign(_sort_key=key, _sort_missing=key.isna())
                    .sort_values(
                        ["_sort_missing", "_sort_key", "inchikey"],
                        ascending=[True, not descending, True],
                        kind="mergesort",
                    )
                    .drop(columns=["_sort_key", "_sort_missing"])
                )
            else:
                # 그 기준으로 줄 세울 값이 아예 없다. 조용히 다른 순서를 내지 않고
                # 화면에 무엇으로 정렬했는지 사실대로 알린다.
                sort_by = scan_sort
        # 점수를 매긴 **뒤에** 거른다. 앞에서 거르면 백분위의 기준 집합이 바뀌어
        # 같은 후보가 이 체크박스 하나로 다른 점수를 받는다.
        undeclared_hidden = 0
        if hide_undeclared and not scored.empty and "functions" in scored.columns:
            declared = scored["functions"].map(
                lambda value: bool(str(value or "").strip())
            )
            # **이 화면에서** 몇 건이 빠졌는지를 센다. 라이브러리 전체의 미신고
            # 수(실측 1,215종)를 내놓으면 20행짜리 표 옆에서 뜻이 통하지 않는다.
            undeclared_hidden = int((~declared).head(limit).sum())
            scored = scored[declared]
        frame = scored.head(limit).reset_index(drop=True)
        if with_three_d and not frame.empty:
            frame = annotate_three_d(frame, smiles, limit=min(limit, MAX_3D_CANDIDATES))
        result["ingredients"] = _ingredient_payload(
            scan, frame, library, index, query_measured, floor, require_core, index_error
        )
        # 두 신호가 얼마나 어긋나는지. 등재 원료 507종에서 Spearman 0.09 - 사실상
        # 무상관이다. 이것을 화면에 내지 않으면 두 열이 나란히 있다는 사실만으로
        # 서로를 뒷받침하는 것처럼 읽힌다.
        result["ingredients"]["signal_agreement"] = (
            signal_disagreement(scan.frame) if with_pharmacophore else {"available": False}
        )
        result["ingredients"]["pharmacophore"] = with_pharmacophore
        result["ingredients"]["pharmacophore_unavailable"] = pharmacophore_error
        result["ingredients"]["sort_by"] = sort_by
        result["ingredients"]["undeclared_hidden"] = undeclared_hidden
        result["ingredients"]["include_undeclared"] = not hide_undeclared
        result["ingredients"]["three_d"] = with_three_d
        result["ingredients"]["three_d_limit"] = min(limit, MAX_3D_CANDIDATES)
        # ADMET 과 종합 점수가 실제로 붙었는지 화면에 알린다. 붙지 않았는데
        # 정렬 선택지에 그 항목이 보이면 읽는 쪽이 없는 값으로 줄을 세운다.
        #
        # **키가 있는지가 아니라 값이 있는지**로 센다. `_alternative_row` 는
        # ADMET 열을 언제나 내보내고 없으면 None 을 넣으므로, 키로 판단하면
        # 캐시가 아예 없는 설치에서도 참이 된다 - 그러면 화면은 안전 축이 통째로
        # 추정값인 표를 "예측이 붙었다"로 띄운다. 번들 없이 깐 설치가 정확히
        # 그 상태다.
        rows_out = result["ingredients"].get("rows") or []

        def _measured_rows(key: str) -> int:
            return sum(1 for row in rows_out if row.get(key) is not None)

        admet_rows = _measured_rows("Skin_Reaction")
        evidence_rows = sum(
            1 for row in rows_out
            if row.get("measured_evidence") not in (None, "", "not_measured",
                                                    "evidence_unavailable")
        )
        result["ingredients"]["admet_available"] = admet_rows > 0
        result["ingredients"]["score_available"] = _measured_rows("score_total") > 0
        # 몇 행이 실측이고 몇 행이 채워진 값인지. 화면이 "이 표의 절반은 추정"을
        # 말할 수 있어야 한다 - 행마다 붙는 `추정` 배지만으로는 전체 그림이 안 보인다.
        result["ingredients"]["axis_measured_rows"] = {
            "safety": admet_rows,
            "evidence": evidence_rows,
            "total": len(rows_out),
        }
        try:
            sys.path.insert(0, str(ROOT / "scripts"))
            from candidate_score import DEFAULT_WEIGHTS, score_explanation

            result["ingredients"]["score_note"] = score_explanation()
            result["ingredients"]["score_weights"] = dict(DEFAULT_WEIGHTS)
        except Exception:                         # noqa: BLE001
            result["ingredients"]["score_note"] = ""

    if include_measured and index is not None:
        measured = find_similar(smiles, index, limit=limit, min_similarity=max(floor, 0.35),
                                exclude_self=exclude_self)
        if not measured.empty:
            measured = annotate_core_retention(measured, smiles, "canonical_smiles")
        result["measured"] = {
            "rows": [
                {
                    "rank": _int_or_zero(row["rank"]),
                    "similarity": _float_or_zero(row["similarity"]),
                    "inchikey": _text(row.get("standard_inchikey")),
                    "smiles": _text(row.get("canonical_smiles")),
                    "target_count": _int_or_zero(row.get("target_count")),
                    "best_pactivity": _round_or_none(row.get("best_pactivity")),
                    "evidence": _text(row.get("evidence"), "none"),
                    "top_targets": unique_targets(row.get("top_targets")),
                    "core_grade": _text(row.get("core_grade"), "unknown"),
                    "core_label_ko": _text(row.get("core_label_ko")),
                    "core_coverage": _float_or_none(row.get("core_coverage")),
                    "core_share": _float_or_zero(row.get("core_share")),
                    # 상한에 걸려 끊긴 MCS를 확정 판정처럼 보이지 않게 하려면 이
                    # 값이 화면까지 가야 한다. 등재 원료 표에는 가는데 여기서만
                    # 빠져 있었다.
                    "core_basis": _text(row.get("core_basis")),
                    "scaffold_match": bool(row.get("scaffold_match")),
                }
                for row in measured.to_dict("records")
            ],
            "library_size": int(index.manifest.get("ligands", 0)),
            "min_similarity": max(floor, 0.35),
        }
    elif include_measured:
        result["measured"] = {"rows": [], "library_size": 0, "unavailable": index_error}

    return result


def _ingredient_payload(scan, frame, library, index, query_measured, floor, require_core, index_error):
    from alternative_ingredients import library_note, scan_summary

    return {
        # 측정 라이브러리가 없으면 모든 행이 "측정된 적 없음"으로 보이는데, 그것은
        # 사실이 아니라 대조를 못 한 것이다. 행마다 표시를 바꿀 수 있도록 알린다.
        "evidence_available": index is not None,
        "evidence_unavailable_reason": index_error,
        # 입력 화합물에 측정 기록이 없으면 "같은 표적" 칸은 언제나 빈다. 그것을
        # "겹치는 표적이 없다"로 읽지 않도록 화면에 사실을 넘긴다.
        "query_measured": query_measured,
        # 화면의 번호는 **이 표에서 몇 번째인가**이고, 스캔이 매긴 순위는 따로
        # 남긴다. 둘을 하나로 합치면 둘 다 틀린 값이 된다:
        #  - 미신고 물질을 걸러내면 번호에 구멍이 난다(1, 3, 4, ...)
        #  - 종합 점수 순으로 정렬하면 핵심구조 순위가 그대로 찍혀 "1977위"가
        #    첫 줄에 온다. 실측으로 안전 순 정렬은 1977·5773·5663... 이었다.
        # 표적 표가 쓰는 것과 같은 짝(`original_rank` + `filtered_position`)이다.
        "rows": [
            _alternative_row(row, evidence_available=index is not None, position=position)
            for position, row in enumerate(frame.to_dict("records"), start=1)
        ],
        "scanned": int(scan.scanned),
        "core_kept": int(scan.kept),
        "unjudged": int(scan.unjudged),
        "grade_counts": {str(k): int(v) for k, v in scan.grade_counts.items()},
        "matches": int(len(scan.frame)),
        "summary": scan_summary(scan),
        "library_size": int(scan.library_size),
        "registered_entries": int(library.registered_entries),
        "resolved_entries": int(library.resolved_entries),
        "note": library_note(library),
        "min_similarity": floor,
        "require_core": require_core,
    }


def _round_or_none(value: Any) -> float | None:
    if value is None or value != value:
        return None
    return round(float(value), 2)


def _text(value: Any, default: str = "") -> str:
    """숫자 칸에서 잡은 NaN 함정이 문자열 칸에도 있다.

    `str(value or default)`는 NaN에서 `"nan"`을 내놓는다 - NaN이 참이라
    `or`를 그냥 통과하기 때문이다. 그래서 측정 기록이 없는 후보의
    `measured_evidence`가 `"not_measured"`가 아니라 `"nan"`이 되고, 그 값은
    화면의 라벨 표에도 CSV의 어느 범주에도 속하지 않는다.
    """
    if value is None or value != value:
        return default
    text = str(value)
    return text if text else default


# 표적 검색은 같은 인덱스를 요청마다 다시 열 수 없다 - edges 가 1.7M행이라
# 첫 로드가 수 초 걸린다. manifest 의 mtime 을 열쇠로 프레임을 한 번만 읽고,
# Stage 0 이 인덱스를 다시 만들면 mtime 이 바뀌어 자동으로 다시 읽는다.
_TARGET_INDEX_CACHE: dict[str, tuple[float, Any, Any]] = {}
_TARGET_INDEX_CACHE_LOCK = threading.Lock()


def _target_index_frames(index_dir: Path) -> tuple[Any, Any]:
    from explore_target import load_index

    stamp = (index_dir / "manifest.json").stat().st_mtime
    key = str(index_dir)
    with _TARGET_INDEX_CACHE_LOCK:
        cached = _TARGET_INDEX_CACHE.get(key)
        if cached is not None and cached[0] == stamp:
            return cached[1], cached[2]
        edges, ligands = load_index(index_dir)
        _TARGET_INDEX_CACHE[key] = (stamp, edges, ligands)
        return edges, ligands


def _target_count(value: Any) -> int:
    rounded = _round_or_none(value)
    return int(rounded) if rounded is not None else 0


def _target_threshold_label(max_pactivity: Any) -> str:
    value = _round_or_none(max_pactivity)
    if value is None:
        return "unknown"
    if value >= 6.0:
        return "positive"
    if value >= 5.0:
        return "borderline"
    return "below"


def _target_summary(entry: Any) -> dict[str, Any]:
    return {
        "uniprot": entry.uniprot,
        "gene": entry.gene,
        "description": entry.description,
    }


def target_binders(payload: dict[str, Any]) -> dict[str, Any]:
    """표적 단백질 → 그 표적에 측정 기록이 있는 화합물.

    CLI `explore_target` 과 같은 해석·정렬을 쓴다. 인덱스가 없으면 실행하지
    않고 그 사실과 사유를 돌려준다 - 화면은 그 문장을 그대로 보여 준다.
    """
    from explore_target import _csv_entries, _discover_index_dir, rank_target, resolve_target

    query = _text(payload.get("target") or payload.get("search"))
    if not query:
        return {
            "available": True,
            "resolved": False,
            "error": "표적 이름이나 UniProt 계정번호를 입력하세요.",
        }

    mode = _text(payload.get("mode"), "balanced")
    if mode not in ("balanced", "potency", "evidence"):
        mode = "balanced"
    try:
        limit = int(payload.get("limit") or 20)
    except (TypeError, ValueError):
        limit = 20
    limit = max(1, min(50, limit))

    entry, candidate_ids = resolve_target(query)
    if entry is None:
        entries = {item.uniprot: item for item in _csv_entries()}
        candidates = []
        for uniprot in candidate_ids[:20]:
            item = entries.get(uniprot)
            candidates.append(
                {
                    "uniprot": uniprot,
                    "gene": item.gene if item else "",
                    "description": item.description if item else "",
                }
            )
        return {"available": True, "resolved": False, "query": query, "candidates": candidates}

    index_dir = _discover_index_dir()
    if index_dir is None:
        return {
            "available": False,
            "resolved": True,
            "target": _target_summary(entry),
            "reason": (
                "이 컴퓨터에 검색 인덱스가 없습니다. Stage 0 표적 데이터를 먼저 준비하세요."
            ),
        }
    try:
        edges, ligands = _target_index_frames(index_dir)
        frame = rank_target(edges, ligands, entry.uniprot, limit, mode)
    except SystemExit as exc:
        return {
            "available": False,
            "resolved": True,
            "target": _target_summary(entry),
            "reason": str(exc),
        }

    rows = []
    for position, (_, row) in enumerate(frame.iterrows(), 1):
        rows.append(
            {
                "rank": position,
                "smiles": _text(row.get("canonical_smiles")),
                "inchikey": _text(row.get("standard_inchikey")),
                "max_pactivity": _round_or_none(row.get("max_pactivity")),
                "median_pactivity": _round_or_none(row.get("median_pactivity")),
                "positive_measurement_count": _target_count(row.get("positive_measurement_count")),
                "publication_count": _target_count(row.get("publication_count")),
                "measurement_count": _target_count(row.get("measurement_count")),
                "source_db": _text(row.get("source_db")),
                "threshold": _target_threshold_label(row.get("max_pactivity")),
            }
        )
    return {
        "available": True,
        "resolved": True,
        "target": _target_summary(entry),
        "mode": mode,
        "rows": rows,
        "index_role": "production",
        "measurement_note": (
            "측정 기록은 ChEMBL·BindingDB 유래입니다. '측정됨'은 '세다'와 다르고, "
            "양성 문턱(6.0) 아래 줄은 근거로 쓰지 마세요."
        ),
    }


_INGREDIENT_FRAME_LOCK = threading.Lock()
_INGREDIENT_FRAME: Any = None


def _ingredient_frame() -> Any:
    """CosIng 등재 원료 표를 한 번만 읽는다(첫 요청 수 초). 없으면 None."""
    global _INGREDIENT_FRAME
    if _INGREDIENT_FRAME is not None:
        return _INGREDIENT_FRAME
    with _INGREDIENT_FRAME_LOCK:
        if _INGREDIENT_FRAME is None:
            try:
                from alternative_ingredients import load_ingredient_library

                _INGREDIENT_FRAME = load_ingredient_library().frame
            except (FileNotFoundError, ImportError, OSError):
                return None
    return _INGREDIENT_FRAME


def target_list_discovery(payload: dict[str, Any]) -> dict[str, Any]:
    """표적 리스트를 한 번에 발굴한다 (CLI discover_from_targets의 서버 판).

    표적 리스트 내용은 응답 밖으로 나가지 않는다 - 계산에만 쓰고 저장하지 않는다.
    """
    from discover_from_targets import TargetRow, annotate_cosing, discover, resolve_targets
    from explore_target import _discover_index_dir

    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        return {"available": True, "error": "표적 행이 없습니다."}
    if len(raw_rows) > 200:
        return {"available": True, "error": "한 번에 200개 표적까지 됩니다."}

    rows: list[Any] = []
    for item in raw_rows:
        if not isinstance(item, dict):
            continue
        label = _text(item.get("target") or item.get("uniprot") or item.get("gene"))
        if label:
            rows.append(
                TargetRow(
                    label=label,
                    category=_text(item.get("category")),
                    direction=_text(item.get("direction")),
                )
            )
    if not rows:
        return {"available": True, "error": "쓸 수 있는 표적 행이 없습니다."}

    try:
        top = int(payload.get("top") or 10)
    except (TypeError, ValueError):
        top = 10
    top = max(1, min(20, top))
    mode = _text(payload.get("mode"), "balanced")
    if mode not in ("balanced", "potency", "evidence"):
        mode = "balanced"

    index_dir = _discover_index_dir()
    if index_dir is None:
        return {
            "available": False,
            "reason": "검색 인덱스가 없습니다. Stage 0 표적 데이터를 먼저 준비하세요.",
        }

    resolved_rows = resolve_targets(rows)
    try:
        edges, ligands = _target_index_frames(index_dir)
        candidates, coverage = discover(
            index_dir, resolved_rows, top, mode, edges=edges, ligands=ligands
        )
    except SystemExit as exc:
        return {"available": False, "reason": str(exc)}

    frame = _ingredient_frame()
    if frame is not None:
        annotate_cosing(candidates, frame)
    else:
        for candidate in candidates:
            candidate["cosing_match"] = "unavailable"
    for record in coverage:
        record["n_registered"] = sum(
            1
            for candidate in candidates
            if candidate["uniprot"] == record["uniprot"] and candidate["cosing_match"] == "exact"
        )

    summary = {
        "targets": len(resolved_rows),
        "resolved": sum(1 for row in resolved_rows if row.resolved),
        "covered": sum(1 for record in coverage if record["in_index"]),
        "candidates": len(candidates),
        "registered": sum(1 for candidate in candidates if candidate["cosing_match"] == "exact"),
    }
    return {
        "available": True,
        "summary": summary,
        "coverage": coverage,
        "candidates": candidates[:300],
        "truncated": len(candidates) > 300,
        "mode": mode,
        "top": top,
        "measurement_note": (
            "측정 기록은 ChEMBL·BindingDB 유래입니다. '측정됨'은 '세다'와 다르고, "
            "양성 문턱(6.0) 아래 줄은 근거로 쓰지 마세요."
        ),
    }


def _int_or_zero(value: Any) -> int:
    """pandas가 결측을 NaN으로 돌려주는데, NaN은 참이라 `value or 0`을 통과한다.

    그대로 int()에 넣으면 요청 전체가 "cannot convert float NaN to integer"로
    죽는다 - 측정 기록이 없는 후보가 하나만 섞여도 그렇게 된다.
    """
    if value is None or value != value:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float_or_zero(value: Any) -> float:
    if value is None or value != value:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _float_or_none(value: Any) -> float | None:
    """잰 값만 숫자로 내보낸다. 재지 못했으면 null.

    `_float_or_zero`를 유지율에 쓰면 판정하지 못한 후보가 "0% 남았다"로 나간다.
    화면은 그것을 잰 값으로 읽고(`pct()`가 0%로 찍는다), 정렬에서도 최하위로
    민다. 없는 것과 0인 것은 다른 사실이다.
    """
    if value is None or value != value:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# 화면에 내보내는 ADMET 항목. scripts/build_admet_cache.py 가 만드는 캐시의
# 부분집합이고, 화장품 원료 판단에 쓰이는 것만 고른다.
ADMET_ROW_KEYS = (
    "logP", "tpsa", "QED",
    "Skin_Reaction", "AMES", "hERG", "DILI", "Carcinogens_Lagunin",
    "Solubility_AqSolDB", "Caco2_Wang",
)


def _alternative_row(row: dict[str, Any], *, evidence_available: bool = True,
                     position: int | None = None) -> dict[str, Any]:
    targets = row.get("measured_top_targets")
    shared = row.get("shared_targets")
    evidence = _text(row.get("measured_evidence"), "not_measured")
    if not evidence_available:
        # "아무도 재 본 적 없다"와 "대조할 라이브러리가 없다"는 다른 말이다.
        evidence = "evidence_unavailable"
    return {
        # 스캔이 매긴 순위(핵심구조 유지 순). 정렬을 바꾸거나 걸러내면 화면의
        # 번호와 달라지므로 둘을 따로 낸다.
        "original_rank": _int_or_zero(row["rank"]),
        "rank": position if position is not None else _int_or_zero(row["rank"]),
        "inci_name": _text(row.get("inci_name")),
        "synonyms": [_text(n) for n in (row.get("synonyms") or []) if _text(n)][:4],
        "cas": _text(row.get("cas")),
        "functions": [f.strip() for f in _text(row.get("functions")).split(";") if f.strip()],
        # 배합목적이 비어 있으면 CosIng 에 이름은 있어도 화장품 용도로 신고된
        # 것이 아니다. 빈 칸으로 두면 "아직 안 적혔나 보다"로 읽히므로 값으로 낸다.
        "cosmetic_use_declared": bool(_text(row.get("functions")).strip()),
        "smiles": _text(row.get("canonical_smiles")),
        "inchikey": _text(row.get("inchikey")),
        "similarity": _float_or_zero(row.get("similarity")),
        "core_grade": _text(row.get("core_grade"), "unknown"),
        "core_label_ko": _text(row.get("core_label_ko")),
        "core_coverage": _float_or_none(row.get("core_coverage")),
        "core_share": _float_or_none(row.get("core_share")),
        "core_basis": _text(row.get("core_basis")),
        "scaffold_match": bool(row.get("scaffold_match")),
        "is_query": bool(row.get("is_query")),
        # TPSA 근접도. 음수이고 0에 가까울수록 극성이 비슷하다.
        "polarity_closeness": _round_or_none(row.get("polarity_closeness")),
        # 파마코포어를 켜지 않았으면 없는 열이다. 0.0으로 채우면 "특징이 하나도
        # 안 겹친다"로 읽히므로 None으로 둔다.
        "pharm_similarity": _round_or_none(row.get("pharm_similarity")),
        # 지문이 비어 잴 수 없었던 후보. 0.00 으로 두면 "특징이 하나도 안 겹친다"로
        # 읽히는데, 실제로는 재지 못한 것이다(큰 유연 지질·펩타이드 12.8%).
        "pharm_usable": bool(row.get("pharm_usable", True)),
        "pharm_recall": _round_or_none(row.get("pharm_recall")),
        "pharm_precision": _round_or_none(row.get("pharm_precision")),
        # 세 기준의 백분위 순위 평균. 켜지 않았으면 없는 값이다.
        "merged_score": _round_or_none(row.get("merged_score")),
        # 3D는 상위 몇 건만 본다. `not_rescored`는 "3D에서 나빴다"가 아니라
        # "재채점하지 않았다"이고, 그 둘을 같은 칸에 같은 모양으로 찍으면 안 된다.
        "three_d_status": _text(row.get("three_d_status")),
        "three_d_recall": _round_or_none(row.get("three_d_recall")),
        "three_d_rmsd": _round_or_none(row.get("three_d_rmsd")),
        "three_d_evaluated_pairs": _int_or_zero(
            row.get("three_d_evaluated_pairs")
        ),
        "three_d_total_pairs": _int_or_zero(row.get("three_d_total_pairs")),
        "measured_evidence": evidence,
        "measured_target_count": _int_or_zero(row.get("measured_target_count")),
        "measured_best_pactivity": _round_or_none(row.get("measured_best_pactivity")),
        "measured_top_targets": list(targets) if isinstance(targets, (list, tuple)) else [],
        # 후보가 입력 화합물과 같은 단백질에서 실제로 측정된 적이 있는가. 핵심구조를
        # 유지하는 목적이 활성 유지이므로, 이 화면이 낼 수 있는 가장 결정에 가까운
        # 근거다. 비어 있다고 활성이 없다는 뜻은 아니다.
        "shared_targets": list(shared) if isinstance(shared, (list, tuple)) else [],
        # 연결성(InChIKey 앞 14자)만 맞은 근거는 입체가 다른 분자의 것일 수 있다.
        "measured_match": _text(row.get("measured_match")),
        # 미리 계산해 둔 ADMET. 캐시가 없으면 전부 None 이고, 그 사실은
        # `admet_available: false` 로 따로 나간다. 0.0 으로 채우면 "위험 없음"
        # 으로 읽히므로 절대 채우지 않는다.
        **{key: _round_or_none(row.get(key)) for key in ADMET_ROW_KEYS},
        # 축별 점수와 종합 순위. 점수를 못 낸 경우에도 None 으로만 남는다.
        "score_structure": _round_or_none(row.get("score_structure")),
        "score_evidence": _round_or_none(row.get("score_evidence")),
        "score_safety": _round_or_none(row.get("score_safety")),
        "score_total": _round_or_none(row.get("score_total")),
        "score_rank": _int_or_zero(row.get("score_rank")) or None,
        # "measured" 는 그 축을 실제 값으로 계산했다는 뜻이고, "imputed" 는
        # 쓸 값이 없어 중앙(0.5)으로 채웠다는 뜻이다. 둘을 같은 숫자로만
        # 보여 주면 채운 값이 잰 값처럼 읽힌다.
        "score_structure_basis": _text(row.get("score_structure_basis"), "imputed"),
        "score_safety_basis": _text(row.get("score_safety_basis"), "imputed"),
        "score_evidence_basis": _text(row.get("score_evidence_basis"), "imputed"),
    }


def compound_preview(payload: dict[str, Any]) -> dict[str, Any]:
    """Canonical structure, 2D drawing and applicability, before any run starts.

    The applicability gate ran only inside the submit handler, and its property
    warnings were dropped entirely, so a reader learned nothing about their
    input until a multi-hour run had already been launched - or, for a typo
    that still parses, until it finished.
    """
    from rdkit import Chem
    from rdkit import RDLogger

    smiles = str(payload.get("smiles", "")).strip()
    if not smiles:
        raise ValueError("SMILES를 입력하세요.")
    if len(smiles) > MAX_SMILES_LENGTH:
        raise ValueError(f"SMILES가 너무 깁니다(최대 {MAX_SMILES_LENGTH}자).")

    RDLogger.DisableLog("rdApp.*")
    molecule = Chem.MolFromSmiles(smiles)
    RDLogger.EnableLog("rdApp.*")
    if molecule is None:
        return {
            "valid": False,
            "verdict": "invalid",
            "verdict_label": PREVIEW_VERDICT_LABELS["invalid"],
            "can_start": False,
            "message": "RDKit이 이 SMILES를 읽지 못했습니다. 오타를 확인하세요.",
            "input_smiles": smiles,
        }

    assessment = assess_applicability(smiles)
    verdict = str(assessment.get("verdict", "review"))
    properties = assessment.get("properties") or {}
    return {
        "valid": True,
        "input_smiles": smiles,
        "canonical_smiles": Chem.MolToSmiles(molecule),
        "inchikey": Chem.MolToInchiKey(molecule),
        "formula": rdMolDescriptors_formula(molecule),
        "svg": _depiction_svg(molecule),
        "verdict": verdict,
        "verdict_label": PREVIEW_VERDICT_LABELS.get(verdict, verdict),
        "can_start": verdict != "out_of_scope",
        "exclusions": assessment.get("exclusions") or [],
        # Computed and stored on every run but never shown; a reader could not
        # see why their compound sits outside the validated panel.
        "warnings": assessment.get("warnings") or [],
        "properties": [
            {
                "key": key,
                "label": PREVIEW_PROPERTY_LABELS.get(key, key),
                "value": value,
            }
            for key, value in properties.items()
        ],
        "message": applicability_refusal(assessment),
    }


def rdMolDescriptors_formula(molecule: Any) -> str | None:
    try:
        from rdkit.Chem import rdMolDescriptors

        return rdMolDescriptors.CalcMolFormula(molecule)
    except Exception:  # noqa: BLE001
        return None


@dataclass
class BatchItem:
    label: str
    smiles: str
    status: str = "queued"
    run_id: str | None = None
    error: str | None = None

    def public(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "smiles": self.smiles,
            "status": self.status,
            "run_id": self.run_id,
            "error": self.error,
        }


@dataclass
class Batch:
    batch_id: str
    items: list[BatchItem]
    preset: str
    mode: str
    evidence_mode: str
    status: str = "running"
    created_at: str = ""
    cancel_requested: bool = False

    def public(self) -> dict[str, Any]:
        done = sum(1 for item in self.items if item.status in TERMINAL_ITEM_STATUS)
        return {
            "batch_id": self.batch_id,
            "status": self.status,
            "preset": self.preset,
            "mode": self.mode,
            "evidence_mode": self.evidence_mode,
            "created_at": self.created_at,
            "total": len(self.items),
            "finished": done,
            "items": [item.public() for item in self.items],
        }


TERMINAL_ITEM_STATUS = {"completed", "failed", "skipped", "cancelled"}
MAX_BATCH_ITEMS = 50
# Matches what the single-compound form sends; the runs are sequential anyway.
BATCH_CORES = 4
_BATCHES: dict[str, Batch] = {}
_BATCHES_LOCK = threading.Lock()
BATCH_POLL_SECONDS = 5.0


def _parse_batch_items(payload: dict[str, Any]) -> list[BatchItem]:
    """Accept a list of {name|smiles} or a pasted CSV/one-per-line block."""
    raw = payload.get("compounds")
    entries: list[tuple[str, str]] = []
    if isinstance(raw, list):
        for record in raw:
            # Either {"label": ..., "smiles": ...} or a bare name/SMILES string.
            if isinstance(record, str):
                value = record.strip()
                if value:
                    entries.append((value, value))
                continue
            if not isinstance(record, dict):
                continue
            smiles = str(record.get("smiles", "")).strip()
            label = str(record.get("label", "") or smiles).strip()
            if smiles:
                entries.append((label, smiles))
    elif isinstance(raw, str):
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # "name,SMILES" or a bare SMILES; a header row is skipped by parsing.
            parts = [part.strip() for part in line.split(",", 1)]
            if len(parts) == 2 and parts[1]:
                entries.append((parts[0] or parts[1], parts[1]))
            else:
                entries.append((parts[0], parts[0]))
    if not entries:
        raise ValueError("실행할 화합물이 없습니다.")
    if len(entries) > MAX_BATCH_ITEMS:
        raise ValueError(f"한 번에 최대 {MAX_BATCH_ITEMS}개까지 실행할 수 있습니다.")

    items: list[BatchItem] = []
    seen: set[str] = set()
    for label, smiles in entries:
        resolved = smiles
        # A name is as good as a structure here; the lookup is the same index
        # the single-compound form uses.
        if not _looks_like_smiles(smiles):
            matches = compound_search(smiles).get("matches") or []
            exact = next((match for match in matches if match["exact"]), None)
            if exact is None:
                items.append(BatchItem(label=label, smiles=smiles, status="skipped",
                                       error="이름으로 찾지 못했습니다."))
                continue
            resolved = exact["smiles"]
            label = label or exact["name"]
        if resolved in seen:
            items.append(BatchItem(label=label, smiles=resolved, status="skipped",
                                   error="같은 구조가 이미 목록에 있습니다."))
            continue
        seen.add(resolved)
        items.append(BatchItem(label=label or resolved, smiles=resolved))
    return items


_RDKIT_LOG_LOCK = threading.Lock()


def _looks_like_smiles(value: str) -> bool:
    from rdkit import Chem
    from rdkit import RDLogger

    with _RDKIT_LOG_LOCK:
        RDLogger.DisableLog("rdApp.*")
        try:
            molecule = Chem.MolFromSmiles(value)
        finally:
            RDLogger.EnableLog("rdApp.*")
    return molecule is not None


def start_batch(payload: dict[str, Any]) -> Batch:
    """Queue several compounds and run them one after another.

    One run per request meant a screening question could not be asked at all.
    Runs stay sequential: they contend for the same GPU, and the existing
    admission gate refuses a second one anyway.
    """
    items = _parse_batch_items(payload)
    batch = Batch(
        batch_id=f"batch_{secrets.token_hex(6)}",
        items=items,
        preset=str(payload.get("preset", "safety")).strip(),
        mode=str(payload.get("mode", "fast")).strip(),
        evidence_mode=str(payload.get("evidence_mode", "evidence")).strip(),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    with _BATCHES_LOCK:
        _BATCHES[batch.batch_id] = batch
    threading.Thread(target=_drive_batch, args=(batch,), daemon=True).start()
    return batch


def _drive_batch(batch: Batch) -> None:
    for item in batch.items:
        if batch.cancel_requested:
            item.status = "cancelled"
            continue
        if item.status in TERMINAL_ITEM_STATUS:
            continue
        try:
            job = _start_run({
                "input_type": "smiles",
                "smiles": item.smiles,
                "preset": batch.preset,
                "mode": batch.mode,
                "evidence_mode": batch.evidence_mode,
                "cores": BATCH_CORES,
            })
        except (ValueError, RuntimeError, OSError) as exc:
            item.status = "failed"
            item.error = str(exc)
            continue
        item.run_id = job.run_id
        item.status = "running"
        while True:
            time.sleep(BATCH_POLL_SECONDS)
            active = _active_job_for_run(job.run_id) if job.run_id else None
            if active is None:
                break
            if batch.cancel_requested and active.status != "cancel_requested":
                try:
                    cancel_job(job.run_id)
                except (ValueError, RuntimeError, OSError):
                    pass
        verification = _verification_state(job.run_id or "")
        summary = _run_summary(job.run_id or "")
        if summary is None:
            item.status = "failed"
            item.error = "결과 요약이 만들어지지 않았습니다."
        elif batch.cancel_requested:
            item.status = "cancelled"
        elif not verification["ok"]:
            item.status = "failed"
            item.error = verification.get("error") or "검증을 통과하지 못했습니다."
        else:
            item.status = "completed"
    if batch.cancel_requested:
        batch.status = "cancelled"
    else:
        successful = sum(item.status == "completed" for item in batch.items)
        failed = sum(item.status == "failed" for item in batch.items)
        batch.status = "partial" if successful and failed else ("failed" if failed else "completed")


def get_batch(batch_id: str) -> Batch | None:
    with _BATCHES_LOCK:
        return _BATCHES.get(batch_id)


def list_batches() -> list[dict[str, Any]]:
    with _BATCHES_LOCK:
        return [batch.public() for batch in
                sorted(_BATCHES.values(), key=lambda b: b.created_at, reverse=True)]


def cancel_batch(batch_id: str) -> Batch | None:
    batch = get_batch(batch_id)
    if batch is None:
        return None
    batch.cancel_requested = True
    for item in batch.items:
        if item.status == "queued":
            item.status = "cancelled"
        elif item.status == "running" and item.run_id:
            try:
                cancel_job(item.run_id)
            except (ValueError, RuntimeError, OSError):
                pass
    return batch


def _start_run(payload: dict[str, Any]) -> Job:
    input_type = str(payload.get("input_type", "smiles")).strip().lower()
    if input_type not in {"smiles", "sdf"}:
        raise ValueError("입력 형식은 SMILES 또는 SDF여야 합니다.")
    requested_preset = str(payload.get("preset", "safety")).strip()
    mode = str(payload.get("mode", "fast")).strip()
    substitute_run = requested_preset == "substitute"
    target_conditioned = payload.get("target_conditioned", True)
    if substitute_run and not isinstance(target_conditioned, bool):
        raise ValueError("3D 표적 anchor 옵션은 참/거짓 값이어야 합니다.")
    if substitute_run and mode != "fast":
        raise ValueError("대체소재 발굴의 Workbench 기본 경로는 빠른 분석 모드만 지원합니다.")
    evidence_mode = str(payload.get("evidence_mode", "evidence")).strip()
    if substitute_run and evidence_mode != "evidence":
        raise ValueError("대체소재 발굴은 동일 타겟 활성 근거를 사용하는 Evidence 모드만 지원합니다.")
    profile_preset = "target-id" if substitute_run else requested_preset
    try:
        profile = resolve_run_profile(
            profile_preset,
            mode,
            evidence_mode=evidence_mode,
            context_profile=str(payload.get("context_profile", "auto")).strip(),
            sota_claim=(payload.get("sota_claim") is True and not substitute_run),
        )
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    if profile.normalized_preset.value == "stage0":
        raise ValueError("Stage 0 is not exposed through the beginner run API.")
    preset = profile.normalized_preset.value
    if substitute_run:
        readiness_key = (
            "substitute_target_conditioned" if target_conditioned else "substitute"
        )
    elif preset == "safety":
        readiness_key = "safety"
    elif preset == "report":
        readiness_key = "report"
    else:
        readiness_key = "target_fast" if mode == "fast" else "target_comprehensive"
    readiness = get_status(force=True).get("analysis_readiness", {}).get(readiness_key, {})
    if readiness.get("ready") is not True:
        missing = readiness.get("missing") if isinstance(readiness.get("missing"), list) else []
        detail = ", ".join(str(item) for item in missing) or "필수 실행 환경"
        raise RuntimeError(f"이 분석을 시작하기 전에 준비가 필요합니다: {detail}")
    if profile.evidence_mode.value == "discovery":
        _require_discovery_alias_package()
    run_id = _safe_run_id(str(payload.get("run_id") or _new_run_id()))
    run_dir = RUNS_DIR / run_id
    if _active_job_for_run(run_id):
        raise RuntimeError("같은 이름의 분석이 이미 실행 중입니다.")
    if (run_dir / "run_summary.json").exists():
        raise RuntimeError("같은 run ID의 결과가 이미 있습니다. 다른 이름을 사용하세요.")
    command = [*_python_command()]
    if substitute_run:
        command.extend([
            str(ROOT / "scripts" / "run_substitute_discovery.py"),
            "--mode",
            mode,
            "--evidence-mode",
            "evidence",
            "--context-profile",
            profile.context_profile.value,
            "--run-id",
            run_id,
        ])
        target_id = str(payload.get("target_id", "")).strip().upper()
        if target_id:
            if not UNIPROT_ACCESSION_RE.fullmatch(target_id):
                raise ValueError("선택 타겟은 유효한 UniProt accession 형식이어야 합니다.")
            command.extend(["--target-id", target_id])
        raw_max_candidates = payload.get("max_candidates", 50)
        if isinstance(raw_max_candidates, bool):
            raise ValueError("대체 후보 수는 정수여야 합니다.")
        try:
            max_candidates = int(raw_max_candidates)
        except (TypeError, ValueError) as exc:
            raise ValueError("대체 후보 수는 정수여야 합니다.") from exc
        if max_candidates < 1 or max_candidates > 200:
            raise ValueError("대체 후보 수는 1에서 200 사이여야 합니다.")
        command.extend(["--max-candidates", str(max_candidates)])
        if target_conditioned:
            command.append("--build-interaction-anchors")
    else:
        command.extend([
            str(ROOT / "scripts" / "run_skinscout.py"),
            "--preset",
            requested_preset,
            "--mode",
            mode,
            "--evidence-mode",
            profile.evidence_mode.value,
            "--context-profile",
            profile.context_profile.value,
            "--run-id",
            run_id,
        ])
        command.append("--online-safety-readiness")
        if profile.sota_claim and requested_preset not in {"target-id-sota", "report-sota"}:
            command.append("--sota-claim")
    cores = int(payload.get("cores", 4))
    if cores < 1 or cores > 256:
        raise ValueError("cores는 1에서 256 사이여야 합니다.")
    command.extend(["--cores", str(cores)])
    if input_type == "smiles":
        smiles = str(payload.get("smiles", "")).strip()
        if not smiles:
            raise ValueError("SMILES를 입력하세요.")
        # A documented scope boundary blocks the run; a property warning does
        # not, because it only says the input sits outside a 15-compound panel.
        refusal = applicability_refusal(assess_applicability(smiles))
        if refusal is not None:
            raise ValueError(refusal)
        command.extend(["--smiles", smiles])
    else:
        sdf_content = str(payload.get("sdf_content", ""))
        if not sdf_content.strip():
            raise ValueError("SDF 파일을 선택하세요.")
        if len(sdf_content.encode("utf-8")) > MAX_SDF_BYTES:
            raise ValueError("SDF 파일은 10 MB 이하만 지원합니다.")
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
        UPLOAD_DIR.chmod(0o700)
        sdf_path = UPLOAD_DIR / f"{run_id}.sdf"
        try:
            descriptor = os.open(
                sdf_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            try:
                stream = os.fdopen(descriptor, "wb")
            except BaseException:
                os.close(descriptor)
                raise
            with stream:
                stream.write(sdf_content.encode("utf-8"))
                stream.flush()
                os.fsync(stream.fileno())
            refusal = applicability_refusal(assess_sdf_applicability(sdf_path))
            if refusal is not None:
                raise ValueError(refusal)
        except BaseException:
            sdf_path.unlink(missing_ok=True)
            raise
        command.extend(["--sdf", str(sdf_path)])
    log_path = LOG_DIR / f"{run_id}.log"
    cleanup_paths = (sdf_path,) if input_type == "sdf" else ()
    try:
        return _start_process_job(
            kind="run",
            run_id=run_id,
            command=command,
            log_path=log_path,
            payload_extra={
                "run_profile": profile.to_dict(),
                "analysis_kind": "substitute" if substitute_run else profile_preset,
                "requested_preset": requested_preset,
                "target_conditioned": target_conditioned if substitute_run else None,
            },
            resource="cpu:safety" if preset == "safety" else DEFAULT_GPU_RESOURCE,
            require_gpu_admission=preset != "safety",
            cleanup_paths=cleanup_paths,
        )
    except Exception:
        _cleanup_private_inputs(cleanup_paths)
        raise


def cancel_job(run_id: str) -> Job | None:
    job = _active_job_for_run(run_id)
    if job is None:
        return job
    record = _coordinator().cancel_job(job.job_id)
    job.status = str(record["status"])
    job.detail = "취소 요청을 보냈습니다. 실행 중인 하위 계산도 종료합니다."
    if job.process is None:
        return job
    process = job.process
    try:
        if os.name == "posix":
            if process.pid is not None:
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        else:
            process.terminate()
    except (OSError, TypeError):
        pass

    def force_stop() -> None:
        try:
            process.wait(timeout=8)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except OSError:
            pass

    threading.Thread(target=force_stop, name=f"cancel_{run_id}", daemon=True).start()
    return job


def _shutdown_process_jobs() -> None:
    with _JOBS_LOCK:
        jobs = [
            job
            for job in _JOBS.values()
            if job.process is not None and job.process.poll() is None
        ]
    coordinator = _COORDINATOR
    for job in jobs:
        if coordinator is not None:
            try:
                coordinator.cancel_job(job.job_id)
            except (CoordinatorError, KeyError):
                pass
        try:
            identity = _process_identity(job.process)
        except OSError:
            job.process.terminate()
        else:
            _signal_process_identity(identity, signal.SIGTERM)
    deadline = time.monotonic() + 8.0
    for job in jobs:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            job.process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                identity = _process_identity(job.process)
            except OSError:
                job.process.kill()
            else:
                _signal_process_identity(identity, signal.SIGKILL)
        finally:
            _cleanup_private_inputs(job.cleanup_paths)


def _registered_artifact_file_for_run(
    run_id: str,
    relative: str,
) -> tuple[str, str] | None:
    try:
        run_id = _safe_run_id(run_id)
    except ValueError:
        return None
    decoded = unquote(relative)
    candidate = Path(decoded)
    if (
        not decoded
        or "\\" in decoded
        or candidate.is_absolute()
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        return None
    normalized = candidate.as_posix()
    record = _durable_job_for_run(run_id)
    if record is None:
        return None
    matches: list[tuple[str, str]] = []
    for artifact in _coordinator().list_artifacts(str(record["job_id"])):
        manifest = artifact.get("manifest")
        files = manifest.get("files") if isinstance(manifest, dict) else None
        if not isinstance(files, list):
            continue
        if any(
            isinstance(entry, dict) and entry.get("path") == normalized
            for entry in files
        ):
            matches.append((str(artifact["artifact_id"]), normalized))
    if len(matches) != 1:
        return None
    return matches[0]


def _content_type(path: str) -> str:
    suffix = Path(path).suffix.lower()
    explicit = {
        ".css": "text/css; charset=utf-8",
        ".js": "text/javascript; charset=utf-8",
        ".html": "text/html; charset=utf-8",
        ".json": "application/json; charset=utf-8",
        ".csv": "text/csv; charset=utf-8",
        ".tsv": "text/tab-separated-values; charset=utf-8",
        ".sdf": "chemical/x-mdl-sdfile",
        ".pdb": "chemical/x-pdb",
        ".cif": "chemical/x-cif",
        ".mmcif": "chemical/x-cif",
        ".wasm": "application/wasm",
    }
    if suffix in explicit:
        return explicit[suffix]
    guessed, _encoding = mimetypes.guess_type(path)
    return guessed or "application/octet-stream"


def _registered_artifact_payload(artifact: dict[str, Any]) -> dict[str, Any]:
    artifact_id = str(artifact["artifact_id"])
    files = []
    manifest = artifact.get("manifest", {})
    manifest_files = manifest.get("files", []) if isinstance(manifest, dict) else []
    for entry in manifest_files:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            continue
        relative = str(entry["path"])
        files.append({
            "path": relative,
            "bytes": entry.get("bytes"),
            "sha256": entry.get("sha256"),
            "download_url": (
                f"/api/v2/artifacts/{artifact_id}/files/{quote(relative, safe='/')}"
            ),
        })
    return {
        "artifact_id": artifact_id,
        "content_sha256": artifact.get("content_sha256"),
        "job_id": artifact.get("job_id"),
        "attempt_id": artifact.get("attempt_id"),
        "namespace": artifact.get("namespace"),
        "created_at": artifact.get("created_at"),
        "schema_version": manifest.get("schema_version") if isinstance(manifest, dict) else None,
        "files": files,
    }


class WorkbenchHandler(BaseHTTPRequestHandler):
    server_version = "SkinScoutWorkbench/0.1"

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write("[workbench] " + format % args + "\n")

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(_json_value(payload), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self._send_security_headers()
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_text(self, body: str, content_type: str, status: int = 200) -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self._send_security_headers()
        self.end_headers()
        try:
            self.wfile.write(encoded)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _handle_session_request(self) -> None:
        """Exchange the access token for a session cookie, or end a session."""
        if not self._request_is_same_origin():
            self._error("다른 웹사이트에서 보낸 요청은 허용하지 않습니다.", 403)
            return
        try:
            payload = self._read_body()
        except (TypeError, ValueError) as exc:
            self._error(str(exc), 400)
            return
        if payload.get("action") == "logout":
            close_session(self._session_id())
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", "16")
            self.send_header(
                "Set-Cookie",
                f"{SESSION_COOKIE}=; Path=/; Max-Age=0; HttpOnly; SameSite=Strict"
                + ("; Secure" if SECURE_SESSION_COOKIE else ""),
            )
            self._send_security_headers()
            self.end_headers()
            self.wfile.write(b'{"ok": true}    ')
            return
        try:
            session_id = open_session(str(payload.get("token", "")))
        except PermissionError as exc:
            # Deliberately the same message and status for a wrong token as for
            # a missing one, so this cannot be used to probe.
            self._error(str(exc), 401)
            return
        body = json.dumps({"authenticated": True}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header(
            "Set-Cookie",
            f"{SESSION_COOKIE}={session_id}; Path=/; Max-Age={SESSION_TTL_SECONDS}; "
            "HttpOnly; SameSite=Strict"
            + ("; Secure" if SECURE_SESSION_COOKIE else ""),
        )
        self._send_security_headers()
        self.end_headers()
        self.wfile.write(body)

    def _send_run_file(self, resolved: Path, *, embeddable: bool = False) -> None:
        """Serve one file out of a run directory.

        `embeddable` relaxes framing for the offline viewer only: the page is
        same-origin and needs to render inside the results screen, and Mol*
        needs eval. Every other response keeps DENY / no-eval.
        """
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(resolved, flags)
        except OSError:
            self._error("파일을 읽지 못했습니다.", 404)
            return
        stream = os.fdopen(descriptor, "rb")
        with stream, ExitStack() as resources:
            file_stat = os.fstat(stream.fileno())
            size = file_stat.st_size
            seal = _successful_seal_for_path(resolved)
            untrusted_diagnostic = _path_is_untrusted_diagnostic(resolved)
            if not stat.S_ISREG(file_stat.st_mode) or (
                isinstance(seal, dict)
                and seal.get("invalid") is True
                and not untrusted_diagnostic
            ):
                self._error("검증된 파일의 무결성이 바뀌었습니다.", 409)
                return
            verification_status = "untrusted_diagnostic" if untrusted_diagnostic else "unverified"
            if not untrusted_diagnostic and isinstance(seal, dict):
                verification_status = (
                    "attestation" if seal.get("attestation") is True else "verified"
                )
            snapshot = None
            if verification_status == "verified":
                # Hash and serve the same snapshot. Re-reading the source inode
                # after verification allowed an in-place writer to change the
                # bytes before they reached the client.
                snapshot = resources.enter_context(
                    tempfile.SpooledTemporaryFile(max_size=1024 * 1024, mode="w+b")
                )
                digest = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
                    snapshot.write(chunk)
                if (
                    size != seal.get("bytes")
                    or not secrets.compare_digest(
                        digest.hexdigest(), str(seal.get("sha256"))
                    )
                ):
                    self._error("검증된 파일의 무결성이 바뀌었습니다.", 409)
                    return
                snapshot.seek(0)
            content_type = RUN_FILE_CONTENT_TYPES.get(
                resolved.suffix.lower(), "application/octet-stream"
            )
            response_stream = snapshot if snapshot is not None else stream
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(size))
            self.send_header("X-SkinScout-Verification", verification_status)
            if untrusted_diagnostic:
                # A shorter, stable marker for clients that only care whether
                # the bytes are trusted at all.
                self.send_header("X-SkinScout-Trust", "untrusted")
            if not embeddable:
                # Sanitize filename for Content-Disposition header to prevent
                # header injection via control characters or quotes.
                safe_name = re.sub(r'[\x00-\x1f\x7f"\\]', "_", resolved.name)
                self.send_header(
                    "Content-Disposition",
                    f'attachment; filename="{safe_name}"',
                )
            self._send_security_headers(
                allow_molstar_eval=embeddable, embeddable=embeddable
            )
            self.end_headers()
            shutil.copyfileobj(response_stream, self.wfile, length=1024 * 1024)

    def _send_bytes(
        self, body: bytes, content_type: str, *, filename: str | None = None
    ) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if filename:
            self.send_header(
                "Content-Disposition", f'attachment; filename="{filename}"'
            )
        self._send_security_headers()
        self.end_headers()
        self.wfile.write(body)

    def _send_vendor_headers(self) -> None:
        """CSP for the vendored structure editor frame only.

        JSME is a GWT bundle: its bootstrap writes inline scripts and uses
        javascript: URLs, so it cannot run under the Workbench's policy. It is
        isolated in its own frame instead of relaxing the app page - the frame
        holds no run data and its only channel out is a postMessage carrying a
        SMILES string.
        """
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
            "connect-src 'self'; font-src 'self'; frame-ancestors 'self'; "
            "base-uri 'none'; form-action 'none'",
        )
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "SAMEORIGIN")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")

    def _send_security_headers(
        self, *, allow_molstar_eval: bool = False, embeddable: bool = False
    ) -> None:
        script_src = "'self' 'wasm-unsafe-eval'"
        if allow_molstar_eval:
            script_src += " 'unsafe-eval'"
        frame_ancestors = "'self'" if embeddable else "'none'"
        # Mol* instantiates its WebAssembly from a data: URI, which counts as a
        # connect-src fetch. The viewer's own scripts are bundled assets, so
        # 'unsafe-inline' is still never granted.
        connect_src = "'self' data: blob:" if allow_molstar_eval else "'self'"
        self.send_header(
            "Content-Security-Policy",
            f"default-src 'none'; script-src {script_src}; style-src 'self' 'unsafe-inline'; "
            f"img-src 'self' data: blob:; connect-src {connect_src}; font-src 'self'; "
            # The results screen embeds the run's own offline viewer. Without an
            # explicit frame-src this falls back to default-src 'none' and the
            # browser blocks the frame even though the child allows the parent.
            "worker-src 'self' blob:; frame-src 'self'; "
            f"frame-ancestors {frame_ancestors}; "
            "base-uri 'none'; form-action 'self'",
        )
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "X-Frame-Options", "SAMEORIGIN" if embeddable else "DENY"
        )
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")

    def _error(self, message: str, status: int = 400) -> None:
        self._send_json({"error": message}, status=status)

    def _read_body(self) -> dict[str, Any]:
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            raise TypeError("application/json 요청만 지원합니다.")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("요청 본문 크기를 읽지 못했습니다.") from exc
        if length < 0 or length > MAX_BODY_BYTES:
            raise ValueError("요청이 너무 큽니다.")
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("JSON 요청만 지원합니다.") from exc
        if not isinstance(payload, dict):
            raise ValueError("JSON 객체가 필요합니다.")
        return payload

    def _request_is_same_origin(self) -> bool:
        origin = self.headers.get("Origin")
        if not origin:
            return True
        host = self.headers.get("Host", "")
        parsed = urlsplit(origin)
        return parsed.scheme in {"http", "https"} and parsed.netloc == host

    def _request_host_is_allowed(self) -> bool:
        hostname = _host_header_name(self.headers.get("Host"))
        return hostname is not None and hostname in _allowed_request_hosts(self.server)

    def _reject_disallowed_request(self) -> bool:
        """One Host/Origin gate for every method, reads included.

        The old code only checked the Host on POST, so a DNS-rebound GET or
        HEAD reached run data. The bind address alone was never the boundary.
        """
        if not self._request_host_is_allowed():
            self._error("허용되지 않은 Host 헤더입니다.", 403)
            return True
        if not self._request_is_same_origin():
            self._error("다른 웹사이트에서 보낸 요청은 허용하지 않습니다.", 403)
            return True
        return False

    @staticmethod
    def _byte_range(value: str | None, size: int) -> tuple[int, int, int]:
        if not value:
            return 200, 0, max(0, size - 1)
        if size <= 0 or not value.startswith("bytes=") or "," in value:
            raise ValueError("invalid byte range")
        spec = value.removeprefix("bytes=").strip()
        if "-" not in spec:
            raise ValueError("invalid byte range")
        start_text, end_text = spec.split("-", 1)
        try:
            if not start_text:
                suffix = int(end_text)
                if suffix <= 0:
                    raise ValueError
                start = max(0, size - suffix)
                end = size - 1
            else:
                start = int(start_text)
                end = int(end_text) if end_text else size - 1
                if start < 0 or start >= size or end < start:
                    raise ValueError
                end = min(end, size - 1)
        except ValueError as exc:
            raise ValueError("invalid byte range") from exc
        return 206, start, end

    def _send_registered_artifact_file(
        self,
        artifact_id: str,
        relative_path: str,
        *,
        head_only: bool,
    ) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", artifact_id):
            self._error("등록된 artifact를 찾지 못했습니다.", 404)
            return
        try:
            artifact, entry, path = _coordinator().resolve_artifact_file(
                artifact_id,
                relative_path,
            )
        except (KeyError, ValueError):
            self._error("등록된 artifact 파일을 찾지 못했습니다.", 404)
            return
        except CoordinatorError as exc:
            self._error(str(exc), 409)
            return
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags)
        except OSError:
            self._error("등록된 artifact 파일을 열지 못했습니다.", 409)
            return
        with os.fdopen(descriptor, "rb") as stream:
            namespace_parts = Path(str(artifact.get("namespace", ""))).parts
            allow_molstar_eval = (
                namespace_parts[-2:] == ("reports", "fast")
                and relative_path.lower().endswith((".html", ".htm"))
            )
            file_stat = os.fstat(stream.fileno())
            expected_size = int(entry["bytes"])
            if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size != expected_size:
                self._error("Artifact 무결성 검증에 실패했습니다.", 409)
                return
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            if not secrets.compare_digest(digest.hexdigest(), str(entry["sha256"])):
                self._error("Artifact 무결성 검증에 실패했습니다.", 409)
                return
            etag = f'"{entry["sha256"]}"'
            if self.headers.get("If-None-Match") == etag:
                self.send_response(304)
                self.send_header("ETag", etag)
                self.send_header("Cache-Control", "public, max-age=31536000, immutable")
                self._send_security_headers(allow_molstar_eval=allow_molstar_eval)
                self.end_headers()
                return
            try:
                status, start, end = self._byte_range(
                    self.headers.get("Range"),
                    expected_size,
                )
            except ValueError:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{expected_size}")
                self.send_header("Content-Length", "0")
                self._send_security_headers(allow_molstar_eval=allow_molstar_eval)
                self.end_headers()
                return
            length = 0 if expected_size == 0 else end - start + 1
            self.send_response(status)
            self.send_header("Content-Type", _content_type(relative_path))
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("ETag", etag)
            self.send_header("Cache-Control", "public, max-age=31536000, immutable")
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{expected_size}")
            self._send_security_headers(allow_molstar_eval=allow_molstar_eval)
            self.end_headers()
            if head_only or length == 0:
                return
            stream.seek(start)
            remaining = length
            try:
                while remaining:
                    chunk = stream.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                return

    def _handle_worker_post(self, path: str, payload: dict[str, Any]) -> bool:
        if path == "/api/v2/worker/claim":
            worker_id = str(payload.get("worker_id", "")).strip()
            if not worker_id:
                raise ValueError("worker_id is required")
            resource = str(payload.get("resource", DEFAULT_GPU_RESOURCE)).strip()
            if resource != DEFAULT_GPU_RESOURCE:
                raise ValueError(
                    f"worker claims are restricted to {DEFAULT_GPU_RESOURCE}"
                )
            attempt = _coordinator().claim_next(
                worker_id=worker_id,
                resource=resource,
            )
            if attempt is None:
                self._send_json({"schema_version": 2, "attempt": None})
            else:
                workspace = _coordinator().attempt_workspace(
                    attempt.attempt_id,
                    attempt.token,
                )
                self._send_json({
                    "schema_version": 2,
                    "attempt": attempt.public(),
                    "attempt_token": attempt.token,
                    "workspace": str(workspace),
                }, 201)
            return True
        parts = [unquote(item) for item in path.split("/") if item]
        if len(parts) != 6 or parts[:4] != ["api", "v2", "worker", "attempts"]:
            return False
        try:
            attempt_id = int(parts[4])
        except ValueError as exc:
            raise ValueError("attempt id must be an integer") from exc
        token = str(payload.get("attempt_token", ""))
        action = parts[5]
        if action == "heartbeat":
            result = _coordinator().heartbeat(attempt_id, token)
        elif action == "events":
            event_type = str(payload.get("type", "")).strip()
            if not event_type:
                raise ValueError("event type is required")
            event_payload = payload.get("payload", {})
            if not isinstance(event_payload, dict):
                raise ValueError("event payload must be an object")
            result = _coordinator().publish_event(
                attempt_id,
                token,
                event_type,
                event_payload,
            )
        elif action == "promote":
            namespace = str(payload.get("namespace", "")).strip()
            if not namespace:
                raise ValueError("artifact namespace is required")
            manifest = payload.get("manifest")
            if not isinstance(manifest, dict):
                raise ValueError("artifact manifest must be an object")
            result = _coordinator().promote_attempt_artifacts(
                attempt_id,
                token,
                namespace=namespace,
                manifest=manifest,
            )
        elif action == "complete":
            result_payload = payload.get("result", {})
            if not isinstance(result_payload, dict):
                raise ValueError("result must be an object")
            result = _coordinator().complete_attempt(
                attempt_id,
                token,
                status=str(payload.get("status", "completed")),
                result=result_payload,
            )
        elif action == "cancelled":
            result = _coordinator().acknowledge_cancel(attempt_id, token)
        else:
            return False
        self._send_json({"schema_version": 2, "result": result})
        return True

    def _session_id(self) -> str | None:
        raw = self.headers.get("Cookie") or ""
        for part in raw.split(";"):
            name, _, value = part.strip().partition("=")
            if name == SESSION_COOKIE and value:
                return value
        return None

    def _authorised_for_read(self, path: str) -> bool:
        """Every read used to be open, including run results and rankings.

        With auth off this is unchanged. With auth on - which any non-loopback
        bind requires - the shell and the login route stay reachable so a
        browser can render the sign-in screen.
        """
        if not REQUIRE_AUTH:
            return True
        if path in {"/", "/index.html", "/api/identity", "/api/session"} or path.startswith("/assets/"):
            return True
        return session_is_valid(self._session_id())

    def do_GET(self) -> None:  # noqa: N802
        if self._reject_disallowed_request():
            return
        parsed = urlsplit(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        if not self._authorised_for_read(path):
            self._error("로그인이 필요합니다.", 401)
            return
        if path == "/" or path == "/index.html":
            self._serve_static("index.html")
            return
        if path == "/api/session":
            self._send_json({
                "authenticated": not REQUIRE_AUTH or session_is_valid(self._session_id()),
                "required": REQUIRE_AUTH,
            })
            return
        if path.startswith("/assets/"):
            self._serve_static(path.removeprefix("/assets/"))
            return
        if path == "/api/identity":
            self._send_json(identity_payload(self.server))
            return
        if path == "/api/status":
            self._send_json(get_status(force=query.get("force", ["0"])[0] == "1"))
            return
        if path == "/api/setup":
            self._send_json(_installation_payload())
            return
        if path == "/api/compound/search":
            self._send_json(compound_search(query.get("q", [""])[0]))
            return
        if path == "/api/batches":
            self._send_json({"batches": list_batches()})
            return
        if path.startswith("/api/batches/"):
            batch = get_batch(unquote(path.removeprefix("/api/batches/")))
            if batch is None:
                self._error("배치를 찾지 못했습니다.", 404)
            else:
                self._send_json({"batch": batch.public()})
            return
        if path == "/api/runs":
            self._send_json({"runs": list_runs()})
            return
        if path == "/api/v2/runs":
            self._send_json({"schema_version": 2, "runs": list_runs()})
            return
        v2_parts = [unquote(item) for item in path.split("/") if item]
        if len(v2_parts) >= 4 and v2_parts[:3] == ["api", "v2", "artifacts"]:
            artifact_id = v2_parts[3]
            if len(v2_parts) == 4:
                try:
                    artifact, _root = _coordinator().resolve_artifact(artifact_id)
                except (KeyError, ValueError):
                    self._error("등록된 artifact를 찾지 못했습니다.", 404)
                except CoordinatorError as exc:
                    self._error(str(exc), 409)
                else:
                    self._send_json({
                        "schema_version": 2,
                        "artifact": _registered_artifact_payload(artifact),
                    })
                return
            if len(v2_parts) >= 6 and v2_parts[4] == "files":
                relative = "/".join(v2_parts[5:])
                self._send_registered_artifact_file(
                    artifact_id,
                    relative,
                    head_only=False,
                )
                return
            self._error("등록된 artifact 경로를 찾지 못했습니다.", 404)
            return
        if len(v2_parts) == 5 and v2_parts[:3] == ["api", "v2", "jobs"] and v2_parts[4] == "events":
            try:
                events = _coordinator().list_events(v2_parts[3])
                _coordinator().get_job(v2_parts[3])
            except KeyError:
                self._error("작업을 찾지 못했습니다.", 404)
            else:
                self._send_json({"schema_version": 2, "events": events})
            return
        if len(v2_parts) >= 4 and v2_parts[:3] == ["api", "v2", "runs"]:
            run_id = v2_parts[3]
            record = _durable_job_for_run(run_id)
            detail = get_run(run_id)
            if detail is None and record is None:
                self._error("분석 결과를 찾지 못했습니다.", 404)
                return
            if len(v2_parts) == 5 and v2_parts[4] == "artifacts":
                registered = (
                    _coordinator().list_artifacts(str(record["job_id"]))
                    if record is not None
                    else []
                )
                self._send_json({
                    "schema_version": 2,
                    "run_id": run_id,
                    "artifacts": [
                        _registered_artifact_payload(artifact)
                        for artifact in registered
                    ],
                    "legacy_artifacts": (
                        detail.get("artifacts", []) if detail is not None else []
                    ),
                })
            else:
                if detail is None:
                    self._error("분석 결과를 찾지 못했습니다.", 404)
                    return
                self._send_json({
                    "schema_version": 2,
                    "run": detail,
                    "events_url": (
                        f"/api/v2/jobs/{record['job_id']}/events"
                        if record is not None
                        else None
                    ),
                })
            return
        parts = [unquote(item) for item in path.split("/") if item]
        if len(parts) >= 3 and parts[:2] == ["api", "runs"]:
            run_id = parts[2]
            if len(parts) == 4 and parts[3] == "artifact":
                relative = query.get("path", [""])[0]
                registered = _registered_artifact_file_for_run(run_id, relative)
                if registered is not None:
                    artifact_id, registered_path = registered
                    self._send_registered_artifact_file(
                        artifact_id,
                        registered_path,
                        head_only=False,
                    )
                    return
                # Nothing in production promotes artifacts, so fall back to the
                # run directory rather than telling a reader their own results
                # are unavailable.
                resolved = _run_file_for_download(run_id, relative)
                if resolved is None:
                    self._error("내려받을 수 있는 파일이 아닙니다.", 404)
                else:
                    self._send_run_file(resolved)
                return
            if len(parts) == 4 and parts[3] == "file":
                resolved = _run_file_for_download(run_id, query.get("path", [""])[0])
                if resolved is None:
                    self._error("내려받을 수 있는 파일이 아닙니다.", 404)
                else:
                    self._send_run_file(resolved)
                return
            if len(parts) == 4 and parts[3] == "bundle.zip":
                bundle = _run_bundle_zip(run_id)
                if bundle is None:
                    self._error("묶어서 내려받을 결과가 없습니다.", 404)
                else:
                    self._send_bytes(
                        bundle,
                        "application/zip",
                        filename=f"{_safe_run_id(run_id)}_results.zip",
                    )
                return
            if len(parts) >= 4 and parts[3] == "viewer":
                relative = "/".join(["viewer", *parts[4:]]) if len(parts) > 4 else "viewer/index.html"
                resolved = _run_file_for_download(run_id, relative)
                if resolved is None:
                    self._error("3D 뷰어 파일을 찾지 못했습니다.", 404)
                else:
                    self._send_run_file(resolved, embeddable=True)
                return
            if len(parts) == 4 and parts[3] == "targets":
                detail = get_run(run_id)
                if detail is None or detail.get("summary") is None:
                    self._error("분석 결과를 찾지 못했습니다.", 404)
                    return
                try:
                    limit = int(query.get("limit", ["50"])[0])
                except ValueError:
                    limit = 50
                target_data = _target_rows(
                    run_id,
                    detail["summary"],
                    query=query.get("q", [""])[0],
                    skin_tier=query.get("skin_tier", ["all"])[0],
                    source=query.get("source", ["all"])[0],
                    sort=query.get("sort", ["final_score"])[0],
                    limit=limit,
                )
                self._send_json(target_data)
                return
            detail = get_run(run_id)
            if detail is None:
                self._error("분석 결과를 찾지 못했습니다.", 404)
            else:
                self._send_json(detail)
            return
        if path.startswith("/api/jobs/"):
            job_id = unquote(path.removeprefix("/api/jobs/"))
            _refresh_jobs()
            with _JOBS_LOCK:
                job = _JOBS.get(job_id)
            if job is None:
                try:
                    job = _durable_job_adapter(_coordinator().get_job(job_id))
                except KeyError:
                    self._error("작업을 찾지 못했습니다.", 404)
                    return
                self._send_json(job.public())
            else:
                self._send_json(job.public())
            return
        self._error("요청한 경로를 찾지 못했습니다.", 404)

    def do_HEAD(self) -> None:  # noqa: N802
        if self._reject_disallowed_request():
            return
        path = urlsplit(self.path).path
        if not self._authorised_for_read(path):
            self._error("로그인이 필요합니다.", 401)
            return
        parts = [unquote(item) for item in path.split("/") if item]
        if (
            len(parts) >= 6
            and parts[:3] == ["api", "v2", "artifacts"]
            and parts[4] == "files"
        ):
            self._send_registered_artifact_file(
                parts[3],
                "/".join(parts[5:]),
                head_only=True,
            )
            return
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self._send_security_headers()
        self.end_headers()

    def _serve_static(self, filename: str) -> None:
        # Mol* 번들은 scripts/report_assets/molstar 에 이미 있다(4.8 MB). 정적
        # 디렉터리로 복사하면 같은 파일이 저장소에 두 벌 들어가므로, 그 경로만
        # 읽기 전용으로 열어 준다. 이름을 정확히 아는 파일만 통과시킨다.
        if filename.startswith("molstar/"):
            name = filename.split("/", 1)[1]
            if name not in {"molstar.js", "molstar.css", "LICENSE"}:
                self._error("정적 파일을 찾지 못했습니다.", 404)
                return
            candidate = (ROOT / "scripts" / "report_assets" / "molstar" / name).resolve()
        else:
            candidate = (STATIC_DIR / filename).resolve()
        try:
            if not filename.startswith("molstar/"):
                candidate.relative_to(STATIC_DIR.resolve())
        except ValueError:
            self._error("허용되지 않은 정적 파일 경로입니다.", 404)
            return
        if not candidate.is_file():
            self._error("정적 파일을 찾지 못했습니다.", 404)
            return
        try:
            stream = candidate.open("rb")
            size = os.fstat(stream.fileno()).st_size
        except OSError:
            self._error("정적 파일을 읽지 못했습니다.", 500)
            return
        # The vendored structure editor ships images and a GWT bundle, so this
        # cannot assume UTF-8 text any more. It also compiles its drawing code
        # at load time, which needs eval.
        # The vendored structure editor runs in its own frame precisely so the
        # Workbench page never needs 'unsafe-eval'.
        with stream:
            vendored = candidate.is_relative_to((STATIC_DIR / "vendor").resolve())
            self.send_response(200)
            self.send_header("Content-Type", _content_type(candidate.name))
            self.send_header("Content-Length", str(size))
            if vendored:
                self._send_vendor_headers()
            else:
                self._send_security_headers()
            self.end_headers()
            shutil.copyfileobj(stream, self.wfile, length=1024 * 1024)

    def do_POST(self) -> None:  # noqa: N802
        if self._reject_disallowed_request():
            return
        path = urlsplit(self.path).path
        if path == "/api/session":
            self._handle_session_request()
            return
        if REQUIRE_AUTH and not path.startswith("/api/v2/worker/"):
            if not session_is_valid(self._session_id()):
                self._error("로그인이 필요합니다.", 401)
                return
        worker_request = path.startswith("/api/v2/worker/")
        if worker_request:
            if not _worker_authorized(self.headers.get("Authorization")):
                self._error("Worker API authentication failed.", 401)
                return
        else:
            if not secrets.compare_digest(self.headers.get("X-SkinScout-Token", ""), _csrf_token()):
                self._error("Workbench 보안 토큰이 없거나 올바르지 않습니다.", 403)
                return
        try:
            payload = self._read_body()
        except TypeError as exc:
            self._error(str(exc), 415)
            return
        except ValueError as exc:
            self._error(str(exc), 400)
            return
        try:
            if worker_request and self._handle_worker_post(path, payload):
                return
            if path == "/api/setup/install":
                job = _start_setup_install(str(payload.get("profile", "demo")))
                self._send_json({"job": job.public()}, 202)
                return
            if path == "/api/setup/stage0":
                job = _start_stage0(int(payload.get("cores", 16)))
                self._send_json({"job": job.public()}, 202)
                return
            if path == "/api/compound/preview":
                self._send_json(compound_preview(payload))
                return
            if path == "/api/compound/alternatives":
                self._send_json(compound_alternatives(payload))
                return
            if path == "/api/target/binders":
                self._send_json(target_binders(payload))
                return
            if path == "/api/target/discover":
                self._send_json(target_list_discovery(payload))
                return
            if path == "/api/compound/structure3d":
                self._send_json(compound_structure_3d(payload))
                return
            if path == "/api/batches":
                batch = start_batch(payload)
                self._send_json({"batch": batch.public()}, 202)
                return
            if path == "/api/runs":
                job = _start_run(payload)
                self._send_json({"job": job.public()}, 202)
                return
            if path == "/api/v2/runs":
                job = _start_run(payload)
                self._send_json({
                    "schema_version": 2,
                    "job": job.public(),
                    "events_url": f"/api/v2/jobs/{job.job_id}/events",
                }, 202)
                return
            parts = [unquote(item) for item in path.split("/") if item]
            if len(parts) == 4 and parts[:2] == ["api", "batches"] and parts[3] == "cancel":
                batch = cancel_batch(parts[2])
                if batch is None:
                    self._error("배치를 찾지 못했습니다.", 404)
                else:
                    self._send_json({"batch": batch.public()})
                return
            if len(parts) == 4 and parts[:2] == ["api", "runs"] and parts[3] == "cancel":
                job = cancel_job(parts[2])
                if job is None:
                    self._error("실행 중인 작업을 찾지 못했습니다.", 404)
                else:
                    self._send_json({"job": job.public()})
                return
            if (
                len(parts) == 5
                and parts[:3] == ["api", "v2", "runs"]
                and parts[4] == "cancel"
            ):
                job = cancel_job(parts[3])
                if job is None:
                    self._error("실행 중인 작업을 찾지 못했습니다.", 404)
                else:
                    self._send_json({"schema_version": 2, "job": job.public()})
                return
            if worker_request:
                self._error("Worker API endpoint not found.", 404)
                return
        except AuthError:
            self._error("Worker API authentication failed.", 401)
            return
        except LeaseUnavailable as exc:
            self._error(str(exc), 409)
            return
        except InvalidTransition as exc:
            self._error(str(exc), 409)
            return
        except PermissionError as exc:
            self._error(str(exc), 409)
            return
        except (RuntimeError, ValueError, OSError, OverflowError) as exc:
            self._error(str(exc), 400)
            return
        self._error("요청한 경로를 찾지 못했습니다.", 404)


LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
WILDCARD_BIND_HOSTS = {"0.0.0.0", "::"}


def _host_header_name(value: object) -> str | None:
    """Parse the hostname out of a Host header without trusting its port."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = urlsplit("//" + value.strip())
        hostname = parsed.hostname
    except ValueError:
        return None
    return hostname.lower() if hostname else None


def _configured_proxy_hosts() -> tuple[str, ...]:
    raw = os.environ.get("SKINSCOUT_ALLOWED_HOSTS", "")
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _normalise_allowed_hosts(values: object) -> frozenset[str]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple, set, frozenset)):
        return frozenset()
    return frozenset(
        name for value in values if (name := _host_header_name(str(value)))
    )


def _allowed_request_hosts(server: object = None) -> frozenset[str]:
    """Hosts this server accepts, including the documented proxy exception.

    Loopback names and the bind address keep the local UI working. A reverse
    proxy deployment must name its public host(s) explicitly through
    ``--allowed-host`` or ``SKINSCOUT_ALLOWED_HOSTS``; an arbitrary Host is
    never accepted, whichever HTTP method carried it.
    """
    hosts = set(LOOPBACK_HOSTS)
    bind_host = getattr(server, "workbench_bind_host", None)
    if isinstance(bind_host, str) and bind_host and bind_host.lower() not in WILDCARD_BIND_HOSTS:
        hosts.add(bind_host.lower())
    allowed = getattr(server, "workbench_allowed_hosts", ())
    if isinstance(allowed, (list, tuple, set, frozenset)):
        hosts.update(str(item).lower() for item in allowed if item)
    return frozenset(hosts)


def identity_payload(server: object = None) -> dict[str, Any]:
    """Public identity plus the security mode a launcher must verify.

    `/api/identity` stays reachable before sign-in so the UI can render the
    login screen. Exposing the mode here is what lets `start_workbench.py`
    refuse to attach an auth-required launch to an auth-off instance.
    """
    return {
        "application": "SkinScout Workbench",
        "bind_host": getattr(server, "workbench_bind_host", None),
        "require_auth": bool(REQUIRE_AUTH),
        "behind_https_proxy": bool(
            getattr(server, "workbench_proxy_mode", False)
        ),
        "allowed_hosts": sorted(
            getattr(server, "workbench_allowed_hosts", ()) or ()
        ),
    }


def create_server(
    host: str = "127.0.0.1",
    port: int = 8080,
    *,
    require_auth: bool | None = None,
    behind_https_proxy: bool = False,
    allowed_hosts: object = (),
) -> ThreadingHTTPServer:
    """Bind the Workbench.

    A non-loopback bind is accepted only when an HTTPS reverse proxy is the
    public transport. This keeps authenticated sessions in Secure cookies and
    prevents the documented remote path from silently sending tokens over
    plaintext HTTP.

    IPv6 is rejected explicitly: the stdlib ``ThreadingHTTPServer`` binds an
    AF_INET socket, so ``--host ::1`` produced an opaque ``gaierror`` instead of
    a message a reader can act on. Use ``127.0.0.1`` or ``localhost``.
    """
    if ":" in host:
        raise ValueError(
            "IPv6 주소(예: ::1)는 지원하지 않습니다. "
            "127.0.0.1 또는 localhost를 사용하세요."
        )
    global REQUIRE_AUTH, SECURE_SESSION_COOKIE
    remote = host not in LOOPBACK_HOSTS
    REQUIRE_AUTH = (
        remote or behind_https_proxy
        if require_auth is None
        else bool(require_auth)
    )
    if remote and not REQUIRE_AUTH:
        raise ValueError(
            "로컬 주소가 아닌 곳에 열려면 인증이 필요합니다(--require-auth)."
        )
    if behind_https_proxy and not REQUIRE_AUTH:
        raise ValueError("HTTPS reverse proxy 구성에는 인증이 필요합니다.")
    if remote and not behind_https_proxy:
        raise ValueError(
            "로컬 주소가 아닌 곳에는 평문 HTTP로 열 수 없습니다. "
            "SSH 터널을 사용하거나 HTTPS reverse proxy 뒤에서 "
            "--behind-https-proxy를 지정하세요."
        )
    SECURE_SESSION_COOKIE = bool(behind_https_proxy)
    _worker_api_token()
    if REQUIRE_AUTH:
        access_token()
    _coordinator()
    requested_hosts: list[object] = list(_configured_proxy_hosts())
    if isinstance(allowed_hosts, str):
        requested_hosts.append(allowed_hosts)
    elif isinstance(allowed_hosts, (list, tuple, set, frozenset)):
        requested_hosts.extend(allowed_hosts)
    httpd = ThreadingHTTPServer((host, port), WorkbenchHandler)
    httpd.workbench_bind_host = host
    httpd.workbench_proxy_mode = bool(behind_https_proxy)
    httpd.workbench_allowed_hosts = _normalise_allowed_hosts(requested_hosts)
    return httpd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="기본값은 loopback입니다.")
    parser.add_argument("--port", type=int, default=8080)
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
        help=(
            "reverse proxy가 전달하는 공개 Host 이름. 지정한 이름 외의 Host로 "
            "오는 요청은 거부됩니다. SKINSCOUT_ALLOWED_HOSTS 환경변수로도 "
            "지정할 수 있습니다."
        ),
    )
    args = parser.parse_args(argv)
    server = create_server(
        args.host,
        args.port,
        require_auth=True if args.require_auth else None,
        behind_https_proxy=args.behind_https_proxy,
        allowed_hosts=args.allowed_host,
    )
    print(f"SkinScout Workbench: http://{args.host}:{args.port}", flush=True)
    if REQUIRE_AUTH:
        print(
            "로그인이 필요합니다. 접속 토큰은 소유자 전용 파일에 저장되어 있습니다:\n"
            f"  {ACCESS_TOKEN_PATH}",
            flush=True,
        )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nSkinScout Workbench stopped.")
    finally:
        _shutdown_process_jobs()
        server.server_close()
    return 0


atexit.register(_shutdown_process_jobs)


if __name__ == "__main__":
    raise SystemExit(main())
