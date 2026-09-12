"""Durable SQLite coordinator for local SkinScout workbench jobs.

The coordinator owns the only SQLite connection in the process. Worker code is
expected to use the methods here instead of opening the database directly.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import secrets
import shutil
import sqlite3
import stat
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


HEARTBEAT_SECONDS = 30.0
ATTEMPT_EXPIRY_SECONDS = 120.0
DEFAULT_GPU_RESOURCE = "gpu:0"
TERMINAL_STATES = frozenset({"completed", "failed", "cancelled"})
SENSITIVE_FIELD_NAMES = frozenset({
    "access_key",
    "api_key",
    "authorization",
    "bearer",
    "client_secret",
    "credential",
    "credentials",
    "password",
    "passwd",
    "private_key",
    "secret",
    "secret_key",
    "token",
    "worker_api_token",
})
SENSITIVE_FIELD_SUFFIXES = (
    "_access_key",
    "_api_key",
    "_credential",
    "_credentials",
    "_password",
    "_private_key",
    "_secret",
    "_secret_key",
    "_token",
)


class CoordinatorError(RuntimeError):
    """Base class for coordinator contract failures."""


class AuthError(CoordinatorError):
    """Raised when an attempt token is missing or invalid."""

    status_code = 401


class LeaseUnavailable(CoordinatorError):
    """Raised when the requested exclusive resource is already leased."""


class InvalidTransition(CoordinatorError):
    """Raised when a requested state transition is not allowed."""


@dataclass(frozen=True)
class ClaimedAttempt:
    """A newly claimed attempt.

    ``token`` is returned exactly once to the worker and is never persisted in
    SQLite. Public payload helpers deliberately omit it.
    """

    job_id: str
    attempt_id: int
    token: str = field(repr=False)
    lease_resource: str
    heartbeat_deadline_seconds: float = HEARTBEAT_SECONDS
    expiry_seconds: float = ATTEMPT_EXPIRY_SECONDS

    def public(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "attempt_id": self.attempt_id,
            "lease_resource": self.lease_resource,
            "heartbeat_deadline_seconds": self.heartbeat_deadline_seconds,
            "expiry_seconds": self.expiry_seconds,
        }


@dataclass
class _QueueItem:
    callback: Any
    completed: threading.Event
    deadline: float
    state_lock: threading.Lock = field(default_factory=threading.Lock)
    cancelled: bool = False
    committed: bool = False
    result: Any = None
    error: BaseException | None = None


class DurableCoordinator:
    """SQLite/WAL coordinator with a single protected writer connection."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        clock: Any = time.time,
        artifact_root: str | Path | None = None,
        queue_timeout_seconds: float = 30.0,
    ) -> None:
        if queue_timeout_seconds <= 0:
            raise ValueError("queue_timeout_seconds must be positive")
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_root = (
            Path(artifact_root)
            if artifact_root is not None
            else self.db_path.parent / "artifacts"
        )
        self.attempt_root = self.artifact_root / "attempts"
        self.committed_root = self.artifact_root / "committed"
        self.staging_root = self.artifact_root / ".staging"
        self.quarantine_root = self.artifact_root / ".quarantine"
        self.artifact_root.parent.mkdir(parents=True, exist_ok=True)
        for directory in (
            self.artifact_root,
            self.attempt_root,
            self.committed_root,
            self.staging_root,
            self.quarantine_root,
        ):
            self._ensure_private_directory(directory)
        self._clock = clock
        self._queue_timeout_seconds = float(queue_timeout_seconds)
        self._writer_lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.db_path,
            isolation_level=None,
            check_same_thread=False,
            timeout=30.0,
        )
        self._connection.row_factory = sqlite3.Row
        self._initialize()
        self._reconcile_artifact_storage()
        self._work_queue: queue.Queue[_QueueItem | None] = queue.Queue()
        self._writer_thread_id: int | None = None
        self._active_queue_item: _QueueItem | None = None
        self._closed = False
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="skinscout-coordinator-writer",
            daemon=True,
        )
        self._writer_thread.start()

    @property
    def connection(self) -> sqlite3.Connection:
        """Expose the owned connection for diagnostics and tests only."""

        return self._connection

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._work_queue.put(None)
        self._writer_thread.join(timeout=30)
        if self._writer_thread.is_alive():
            raise CoordinatorError("coordinator writer queue did not stop")
        self._connection.close()

    def create_job(
        self,
        kind: str,
        payload: dict[str, Any] | None = None,
        *,
        job_id: str | None = None,
        priority: int = 0,
        max_attempts: int = 3,
    ) -> dict[str, Any]:
        self._reject_sensitive(payload or {})
        job_id = job_id or secrets.token_hex(16)
        payload_text = self._json(payload or {})

        def body() -> dict[str, Any]:
            now = self._now()
            self._connection.execute(
                """
                INSERT INTO jobs (
                    job_id, kind, status, payload_json, priority, max_attempts,
                    created_at, updated_at
                )
                VALUES (?, ?, 'queued', ?, ?, ?, ?, ?)
                """,
                (job_id, kind, payload_text, priority, max_attempts, now, now),
            )
            self._append_event(job_id, None, "job.created", {"status": "queued"}, now)
            return self.get_job(job_id)

        return self._mutate(body)

    def claim_next(
        self,
        *,
        worker_id: str,
        resource: str = DEFAULT_GPU_RESOURCE,
    ) -> ClaimedAttempt | None:
        def body() -> ClaimedAttempt | None:
            now = self._now()
            self._expire_locked(now)
            if self._active_lease(resource) is not None:
                raise LeaseUnavailable(f"{resource} is already leased")
            job = self._connection.execute(
                """
                SELECT * FROM jobs
                WHERE status = 'queued'
                ORDER BY priority DESC, created_at ASC, job_id ASC
                LIMIT 1
                """
            ).fetchone()
            if job is None:
                return None
            return self._claim_locked(job, worker_id, resource, now)

        return self._mutate(body)

    def claim_job(
        self,
        job_id: str,
        *,
        worker_id: str,
        resource: str = DEFAULT_GPU_RESOURCE,
    ) -> ClaimedAttempt:
        def body() -> ClaimedAttempt:
            now = self._now()
            self._expire_locked(now)
            if self._active_lease(resource) is not None:
                raise LeaseUnavailable(f"{resource} is already leased")
            job = self._job_row(job_id)
            if job["status"] != "queued":
                raise InvalidTransition(
                    f"cannot claim job {job_id} while it is {job['status']}"
                )
            return self._claim_locked(job, worker_id, resource, now)

        return self._mutate(body)

    def heartbeat(self, attempt_id: int, token: str) -> dict[str, Any]:
        def body() -> dict[str, Any]:
            now = self._now()
            attempt = self._active_attempt_or_none(attempt_id, token, now)
            if attempt is None:
                return {"error": "attempt is expired or no longer active"}
            expires_at = now + ATTEMPT_EXPIRY_SECONDS
            self._connection.execute(
                """
                UPDATE attempts
                SET last_heartbeat_at = ?, expires_at = ?
                WHERE attempt_id = ?
                """,
                (now, expires_at, attempt_id),
            )
            self._connection.execute(
                "UPDATE resource_leases SET expires_at = ? WHERE attempt_id = ?",
                (expires_at, attempt_id),
            )
            self._append_event(attempt["job_id"], attempt_id, "attempt.heartbeat", {}, now)
            job = self._job_row(attempt["job_id"])
            return {
                "job_id": attempt["job_id"],
                "attempt_id": attempt_id,
                "status": job["status"],
                "cancel_requested": job["status"] == "cancel_requested",
                "next_heartbeat_seconds": HEARTBEAT_SECONDS,
            }

        result = self._mutate(body)
        if "error" in result:
            raise InvalidTransition(result["error"])
        return result

    def publish_event(
        self,
        attempt_id: int,
        token: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._reject_sensitive(payload or {})
        def body() -> dict[str, Any]:
            now = self._now()
            attempt = self._active_attempt_or_none(attempt_id, token, now)
            if attempt is None:
                return {"error": "attempt is expired or no longer active"}
            job = self._job_row(attempt["job_id"])
            if job["status"] != "running":
                raise InvalidTransition(f"cannot publish while job is {job['status']}")
            return self._append_event(attempt["job_id"], attempt_id, event_type, payload or {}, now)

        result = self._mutate(body)
        if "error" in result:
            raise InvalidTransition(result["error"])
        return result

    def complete_attempt(
        self,
        attempt_id: int,
        token: str,
        *,
        status: str = "completed",
        result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._reject_sensitive(result or {})
        if status not in {"completed", "failed"}:
            raise InvalidTransition(f"unsupported terminal attempt status: {status}")
        def body() -> dict[str, Any]:
            now = self._now()
            attempt = self._active_attempt_or_none(attempt_id, token, now)
            if attempt is None:
                return {"error": "attempt is expired or no longer active"}
            job = self._job_row(attempt["job_id"])
            if job["status"] != "running":
                raise InvalidTransition(f"cannot complete while job is {job['status']}")
            self._connection.execute(
                "UPDATE attempts SET status = ?, ended_at = ? WHERE attempt_id = ?",
                (status, now, attempt_id),
            )
            self._release_lease(attempt_id)
            self._set_job_status(attempt["job_id"], status, now)
            self._append_event(attempt["job_id"], attempt_id, f"job.{status}", result or {}, now)
            return self.get_job(attempt["job_id"])

        result = self._mutate(body)
        if "error" in result:
            raise InvalidTransition(result["error"])
        return result

    def acknowledge_cancel(self, attempt_id: int, token: str) -> dict[str, Any]:
        def body() -> dict[str, Any]:
            now = self._now()
            attempt = self._active_attempt_or_none(attempt_id, token, now)
            if attempt is None:
                return {"error": "attempt is expired or no longer active"}
            job = self._job_row(attempt["job_id"])
            if job["status"] != "cancel_requested":
                raise InvalidTransition(
                    f"cannot acknowledge cancellation while job is {job['status']}"
                )
            self._connection.execute(
                "UPDATE attempts SET status = 'cancelled', ended_at = ? "
                "WHERE attempt_id = ?",
                (now, attempt_id),
            )
            self._release_lease(attempt_id)
            self._set_job_status(attempt["job_id"], "cancelled", now)
            self._append_event(
                attempt["job_id"],
                attempt_id,
                "job.cancelled",
                {"reason": "worker_acknowledged"},
                now,
            )
            return self.get_job(attempt["job_id"])

        result = self._mutate(body)
        if "error" in result:
            raise InvalidTransition(result["error"])
        return result

    def attempt_workspace(self, attempt_id: int, token: str) -> Path:
        def body() -> Path:
            now = self._now()
            attempt = self._active_attempt_or_none(attempt_id, token, now)
            if attempt is None:
                raise InvalidTransition("attempt is expired or no longer active")
            workspace = self._attempt_workspace_path(
                attempt_id,
                str(attempt["token_hash"]),
            )
            try:
                workspace.mkdir(mode=0o700, parents=False, exist_ok=False)
            except FileExistsError as exc:
                raise CoordinatorError("attempt workspace already exists") from exc
            self._fsync_directory(self.attempt_root)
            return workspace

        return self._mutate(body)

    def promote_attempt_artifacts(
        self,
        attempt_id: int,
        token: str,
        *,
        namespace: str,
        manifest: dict[str, Any],
    ) -> dict[str, Any]:
        self._reject_sensitive(manifest)
        canonical_manifest = self._validated_artifact_manifest(manifest)
        manifest_text = json.dumps(
            canonical_manifest,
            sort_keys=True,
            separators=(",", ":"),
        )
        artifact_id = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
        namespace_path = self._safe_relative_path(namespace, "artifact namespace")

        staging = self.staging_root / f"{artifact_id}.{attempt_id}"
        destination = self.committed_root / namespace_path / artifact_id
        workspace = self._attempt_workspace_path(
            attempt_id,
            self._hash_token(token),
        )
        staging_relative = staging.relative_to(self.artifact_root).as_posix()
        destination_relative = destination.relative_to(self.artifact_root).as_posix()

        def prepare() -> dict[str, Any] | None:
            now = self._now()
            attempt = self._active_attempt_or_none(attempt_id, token, now)
            if attempt is None:
                raise InvalidTransition("attempt is expired or no longer active")
            job = self._job_row(attempt["job_id"])
            if job["status"] != "running":
                raise InvalidTransition(
                    f"cannot promote artifacts while job is {job['status']}"
                )
            existing = self._connection.execute(
                "SELECT * FROM artifact_registry WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchone()
            if existing is not None:
                if (
                    existing["job_id"] == attempt["job_id"]
                    and int(existing["attempt_id"]) == attempt_id
                ):
                    return self._public_artifact(existing)
                raise InvalidTransition("artifact hash is already registered")
            pending = self._connection.execute(
                "SELECT artifact_id FROM artifact_promotions WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchone()
            if pending is not None:
                raise InvalidTransition("artifact promotion is already in progress")
            self._verify_workspace(workspace, canonical_manifest)
            if staging.exists():
                raise CoordinatorError("artifact staging path already exists")
            destination.parent.mkdir(parents=True, exist_ok=True)
            self._fsync_directory_chain(destination.parent, self.committed_root)
            if destination.exists():
                raise InvalidTransition("artifact destination already exists")
            self._connection.execute(
                """
                INSERT INTO artifact_promotions (
                    artifact_id, job_id, attempt_id, namespace, status,
                    staging_path, destination_path, manifest_json, created_at
                ) VALUES (?, ?, ?, ?, 'prepared', ?, ?, ?, ?)
                """,
                (
                    artifact_id,
                    attempt["job_id"],
                    attempt_id,
                    namespace_path.as_posix(),
                    staging_relative,
                    destination_relative,
                    manifest_text,
                    now,
                ),
            )
            return None

        existing = self._mutate(prepare)
        if existing is not None:
            return existing

        published = False
        try:
            workspace.rename(staging)
            self._fsync_directory(self.attempt_root)
            self._fsync_directory(self.staging_root)
            self._verify_workspace(staging, canonical_manifest)
            self._write_text_atomic(
                staging / "artifact_manifest.json",
                manifest_text + "\n",
            )
            self._seal_tree(staging, seal_root=False)
            self._fsync_tree(staging)

            def authorize() -> None:
                now = self._now()
                attempt = self._active_attempt_or_none(attempt_id, token, now)
                if attempt is None:
                    raise InvalidTransition("attempt is expired or no longer active")
                job = self._job_row(attempt["job_id"])
                if job["status"] != "running":
                    raise InvalidTransition(
                        f"cannot promote artifacts while job is {job['status']}"
                    )
                cursor = self._connection.execute(
                    """
                    UPDATE artifact_promotions
                    SET status = 'authorized', authorized_at = ?
                    WHERE artifact_id = ? AND status = 'prepared'
                    """,
                    (now, artifact_id),
                )
                if cursor.rowcount != 1:
                    raise CoordinatorError("artifact promotion authorization was lost")

            self._mutate(authorize)
            try:
                staging.rename(destination)
            except OSError as exc:
                raise CoordinatorError(
                    "artifact promotion must be an atomic same-filesystem rename"
                ) from exc
            published = True
            destination.chmod(0o555)
            self._fsync_directory(destination)
            self._fsync_directory(self.staging_root)
            self._fsync_directory_chain(destination.parent, self.committed_root)

            def finalize() -> dict[str, Any]:
                row = self._connection.execute(
                    "SELECT * FROM artifact_promotions WHERE artifact_id = ?",
                    (artifact_id,),
                ).fetchone()
                if row is None or row["status"] != "authorized":
                    raise CoordinatorError("authorized artifact promotion is missing")
                return self._finalize_promotion_locked(row, self._now())

            artifact = self._mutate(finalize)
        except Exception:
            if not published:
                self._restore_failed_promotion(
                    artifact_id=artifact_id,
                    workspace=workspace,
                    staging=staging,
                )
            raise
        if workspace.exists():
            self._remove_tree(workspace)
        return artifact

    def list_artifacts(self, job_id: str) -> list[dict[str, Any]]:
        def body() -> list[dict[str, Any]]:
            rows = self._connection.execute(
                "SELECT * FROM artifact_registry WHERE job_id = ? "
                "ORDER BY created_at ASC, artifact_id ASC",
                (job_id,),
            ).fetchall()
            return [self._public_artifact(row) for row in rows]

        return self._read(body)

    def resolve_artifact(self, artifact_id: str) -> tuple[dict[str, Any], Path]:
        def body() -> tuple[dict[str, Any], Path]:
            row = self._connection.execute(
                "SELECT * FROM artifact_registry WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchone()
            if row is None:
                raise KeyError(artifact_id)
            path = self.artifact_root / str(row["relative_path"])
            if not path.is_dir():
                raise CoordinatorError("registered artifact directory is missing")
            return self._public_artifact(row), path

        return self._read(body)

    def resolve_artifact_file(
        self,
        artifact_id: str,
        relative_path: str,
    ) -> tuple[dict[str, Any], dict[str, Any], Path]:
        artifact, root = self.resolve_artifact(artifact_id)
        relative = self._safe_relative_path(relative_path, "artifact file path").as_posix()
        entries = {
            str(entry["path"]): dict(entry)
            for entry in artifact["manifest"]["files"]
        }
        entry = entries.get(relative)
        if entry is None:
            raise KeyError(relative_path)
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise CoordinatorError("registered artifact file is missing or invalid")
        return artifact, entry, path

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        def body() -> dict[str, Any]:
            now = self._now()
            job = self._job_row(job_id)
            if job["status"] in TERMINAL_STATES:
                return self.get_job(job_id)
            if job["status"] == "queued":
                self._set_job_status(job_id, "cancelled", now)
                self._append_event(job_id, None, "job.cancelled", {"idempotent": False}, now)
                return self.get_job(job_id)
            if job["status"] != "cancel_requested":
                self._set_job_status(job_id, "cancel_requested", now)
                self._append_event(job_id, None, "job.cancel_requested", {"idempotent": False}, now)
            return self.get_job(job_id)

        return self._mutate(body)

    def recover(self) -> dict[str, int]:
        def body() -> dict[str, int]:
            now = self._now()
            return self._expire_locked(now)

        return self._mutate(body)

    def record_process_identity(
        self,
        attempt_id: int,
        token: str,
        identity: dict[str, Any],
    ) -> dict[str, Any]:
        """Bind a locally spawned process identity to its durable attempt."""

        self._reject_sensitive(identity)

        def body() -> dict[str, Any]:
            now = self._now()
            attempt = self._active_attempt_or_none(attempt_id, token, now)
            if attempt is None:
                raise InvalidTransition("attempt is expired or no longer active")
            job = self._job_row(attempt["job_id"])
            payload = self._loads(job["payload_json"])
            payload["process_identity"] = dict(identity)
            self._connection.execute(
                "UPDATE jobs SET payload_json = ?, updated_at = ? WHERE job_id = ?",
                (self._json(payload), now, attempt["job_id"]),
            )
            self._append_event(
                attempt["job_id"],
                attempt_id,
                "attempt.process_started",
                dict(identity),
                now,
            )
            return self.get_job(attempt["job_id"])

        return self._mutate(body)

    def recover_interrupted_job(
        self,
        job_id: str,
        *,
        reason: str,
        requeue: bool,
    ) -> dict[str, Any]:
        """Resolve a local attempt after its owning server process disappeared."""

        def body() -> dict[str, Any]:
            now = self._now()
            job = self._job_row(job_id)
            attempts = self._connection.execute(
                "SELECT * FROM attempts WHERE job_id = ? AND status = 'active'",
                (job_id,),
            ).fetchall()
            for attempt in attempts:
                self._connection.execute(
                    "UPDATE attempts SET status = 'expired', ended_at = ? WHERE attempt_id = ?",
                    (now, attempt["attempt_id"]),
                )
                self._release_lease(int(attempt["attempt_id"]))
                self._append_event(
                    job_id,
                    int(attempt["attempt_id"]),
                    "attempt.interrupted",
                    {"reason": reason},
                    now,
                )
            can_retry = self._attempt_count(job_id) < int(job["max_attempts"])
            next_status = "queued" if requeue and can_retry else "failed"
            self._set_job_status(job_id, next_status, now)
            self._append_event(
                job_id,
                None,
                f"job.{next_status}",
                {"reason": reason},
                now,
            )
            return self.get_job(job_id)

        return self._mutate(body)

    def invalidate_active_attempts(self, reason: str) -> dict[str, int]:
        """Invalidate all active leases after a trusted worker-secret rotation."""

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("lease invalidation reason is required")

        def body() -> dict[str, int]:
            now = self._now()
            attempts = self._connection.execute(
                "SELECT * FROM attempts WHERE status = 'active' ORDER BY attempt_id"
            ).fetchall()
            requeued = 0
            for attempt in attempts:
                attempt_id = int(attempt["attempt_id"])
                job = self._job_row(attempt["job_id"])
                self._connection.execute(
                    "UPDATE attempts SET status = 'expired', ended_at = ? "
                    "WHERE attempt_id = ?",
                    (now, attempt_id),
                )
                self._release_lease(attempt_id)
                self._append_event(
                    attempt["job_id"],
                    attempt_id,
                    "attempt.invalidated",
                    {"reason": reason.strip()},
                    now,
                )
                if job["status"] == "cancel_requested":
                    self._set_job_status(attempt["job_id"], "cancelled", now)
                    self._append_event(
                        attempt["job_id"],
                        attempt_id,
                        "job.cancelled",
                        {"reason": reason.strip()},
                        now,
                    )
                elif self._attempt_count(attempt["job_id"]) < int(job["max_attempts"]):
                    self._set_job_status(attempt["job_id"], "queued", now)
                    self._append_event(
                        attempt["job_id"],
                        attempt_id,
                        "job.requeued",
                        {"reason": reason.strip()},
                        now,
                    )
                    requeued += 1
                else:
                    self._set_job_status(attempt["job_id"], "failed", now)
                    self._append_event(
                        attempt["job_id"],
                        attempt_id,
                        "job.failed",
                        {"reason": reason.strip(), "attempts_exhausted": True},
                        now,
                    )
            return {"invalidated_attempts": len(attempts), "requeued_jobs": requeued}

        return self._mutate(body)

    def get_job(self, job_id: str) -> dict[str, Any]:
        def body() -> dict[str, Any]:
            row = self._connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if row is None:
                raise KeyError(job_id)
            return self._public_job(row)

        return self._read(body)

    def list_jobs(
        self,
        *,
        statuses: set[str] | frozenset[str] | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if limit < 1 or limit > 1_000:
            raise ValueError("limit must be between 1 and 1000")
        requested_statuses = frozenset(statuses) if statuses is not None else None
        allowed_statuses = TERMINAL_STATES | {"queued", "running", "cancel_requested"}
        if requested_statuses is not None and not requested_statuses <= allowed_statuses:
            raise ValueError("statuses contains an unsupported job status")

        def body() -> list[dict[str, Any]]:
            if requested_statuses is None:
                rows = self._connection.execute(
                    "SELECT * FROM jobs ORDER BY created_at DESC, job_id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            elif not requested_statuses:
                rows = []
            else:
                placeholders = ",".join("?" for _ in requested_statuses)
                rows = self._connection.execute(
                    f"SELECT * FROM jobs WHERE status IN ({placeholders}) "
                    "ORDER BY created_at DESC, job_id DESC LIMIT ?",
                    (*sorted(requested_statuses), limit),
                ).fetchall()
            return [self._public_job(row) for row in rows]

        return self._read(body)

    def list_events(self, job_id: str) -> list[dict[str, Any]]:
        def body() -> list[dict[str, Any]]:
            rows = self._connection.execute(
                """
                SELECT event_id, job_id, attempt_id, seq, type, payload_json, created_at
                FROM events
                WHERE job_id = ?
                ORDER BY seq ASC
                """,
                (job_id,),
            ).fetchall()
            return [
                {
                    "event_id": row["event_id"],
                    "job_id": row["job_id"],
                    "attempt_id": row["attempt_id"],
                    "seq": row["seq"],
                    "type": row["type"],
                    "payload": self._loads(row["payload_json"]),
                    "created_at": row["created_at"],
                }
                for row in rows
            ]

        return self._read(body)

    def _initialize(self) -> None:
        with self._writer_lock:
            mode = self._connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            if str(mode).lower() != "wal":
                raise CoordinatorError(f"SQLite did not enable WAL mode: {mode}")
            self._connection.execute("PRAGMA synchronous=FULL")
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.execute("PRAGMA busy_timeout=30000")
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                for statement in (
                    """
                    CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN (
                            'queued', 'running', 'cancel_requested',
                            'completed', 'failed', 'cancelled'
                        )
                    ),
                    payload_json TEXT NOT NULL,
                    priority INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 3,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                    )
                    """,
                    """
                    CREATE TABLE IF NOT EXISTS attempts (
                    attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL REFERENCES jobs(job_id),
                    worker_id TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('active', 'expired', 'completed', 'failed', 'cancelled')
                    ),
                    token_hash TEXT NOT NULL,
                    lease_resource TEXT NOT NULL,
                    started_at REAL NOT NULL,
                    last_heartbeat_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    ended_at REAL
                    )
                    """,
                    """
                    CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL REFERENCES jobs(job_id),
                    attempt_id INTEGER REFERENCES attempts(attempt_id),
                    seq INTEGER NOT NULL,
                    type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    UNIQUE(job_id, seq)
                    )
                    """,
                    """
                    CREATE TRIGGER IF NOT EXISTS events_no_update
                    BEFORE UPDATE ON events
                    BEGIN
                        SELECT RAISE(ABORT, 'events are append-only');
                    END
                    """,
                    """
                    CREATE TRIGGER IF NOT EXISTS events_no_delete
                    BEFORE DELETE ON events
                    BEGIN
                        SELECT RAISE(ABORT, 'events are append-only');
                    END
                    """,
                    """
                    CREATE TABLE IF NOT EXISTS resource_leases (
                    resource TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL REFERENCES jobs(job_id),
                    attempt_id INTEGER NOT NULL REFERENCES attempts(attempt_id),
                    worker_id TEXT NOT NULL,
                    acquired_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                    )
                    """,
                    """
                    CREATE TABLE IF NOT EXISTS artifact_registry (
                    artifact_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL REFERENCES jobs(job_id),
                    attempt_id INTEGER NOT NULL REFERENCES attempts(attempt_id),
                    namespace TEXT NOT NULL,
                    relative_path TEXT NOT NULL UNIQUE,
                    manifest_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                    )
                    """,
                    """
                    CREATE TABLE IF NOT EXISTS artifact_promotions (
                    artifact_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL REFERENCES jobs(job_id),
                    attempt_id INTEGER NOT NULL REFERENCES attempts(attempt_id),
                    namespace TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('prepared', 'authorized')),
                    staging_path TEXT NOT NULL UNIQUE,
                    destination_path TEXT NOT NULL UNIQUE,
                    manifest_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    authorized_at REAL
                    )
                    """,
                ):
                    self._connection.execute(statement)
            except Exception:
                self._connection.execute("ROLLBACK")
                raise
            self._connection.execute("COMMIT")

    def _reconcile_artifact_storage(self) -> None:
        """Recover authorized publications and quarantine untrusted remnants."""

        self._connection.execute("BEGIN IMMEDIATE")
        try:
            promotions = self._connection.execute(
                "SELECT * FROM artifact_promotions ORDER BY created_at, artifact_id"
            ).fetchall()
            for promotion in promotions:
                staging = self._promotion_path(
                    promotion["staging_path"],
                    self.staging_root,
                    "artifact staging path",
                )
                destination = self._promotion_path(
                    promotion["destination_path"],
                    self.committed_root,
                    "artifact destination path",
                )
                if (
                    promotion["status"] == "authorized"
                    and destination.is_dir()
                    and not destination.is_symlink()
                    and not staging.exists()
                ):
                    manifest = self._validated_artifact_manifest(
                        self._loads(promotion["manifest_json"])
                    )
                    self._verify_workspace(
                        destination,
                        manifest,
                        allow_root_manifest=True,
                    )
                    self._seal_tree(destination)
                    self._fsync_tree(destination)
                    self._fsync_directory_chain(
                        destination.parent,
                        self.committed_root,
                    )
                    self._finalize_promotion_locked(promotion, self._now())
                    continue
                if destination.exists() or destination.is_symlink():
                    self._quarantine_path(destination, "incomplete-promotion")
                if staging.exists() or staging.is_symlink():
                    attempt = self._connection.execute(
                        "SELECT token_hash FROM attempts WHERE attempt_id = ?",
                        (promotion["attempt_id"],),
                    ).fetchone()
                    if attempt is None:
                        self._quarantine_path(staging, "orphaned-promotion")
                    else:
                        self._restore_staging_path(
                            staging,
                            self._attempt_workspace_path(
                                int(promotion["attempt_id"]),
                                str(attempt["token_hash"]),
                            ),
                        )
                self._connection.execute(
                    "DELETE FROM artifact_promotions WHERE artifact_id = ?",
                    (promotion["artifact_id"],),
                )
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        self._connection.execute("COMMIT")

        registered_paths: set[Path] = set()
        rows = self._connection.execute(
            "SELECT * FROM artifact_registry ORDER BY artifact_id"
        ).fetchall()
        for row in rows:
            path = self._promotion_path(
                row["relative_path"],
                self.committed_root,
                "registered artifact path",
            )
            manifest = self._validated_artifact_manifest(
                self._loads(row["manifest_json"])
            )
            if not path.is_dir() or path.is_symlink():
                raise CoordinatorError(
                    f"registered artifact directory is missing: {row['artifact_id']}"
                )
            self._verify_workspace(path, manifest, allow_root_manifest=True)
            registered_paths.add(path.resolve())

        for staged in sorted(self.staging_root.iterdir()):
            self._quarantine_path(staged, "staging")

        for manifest_path in sorted(self.committed_root.rglob("artifact_manifest.json")):
            artifact_path = manifest_path.parent
            if artifact_path.resolve() not in registered_paths:
                self._quarantine_path(artifact_path, "unregistered")

    def _promotion_path(
        self,
        relative_value: Any,
        expected_root: Path,
        label: str,
    ) -> Path:
        relative = self._safe_relative_path(relative_value, label)
        path = self.artifact_root / relative
        try:
            path.resolve(strict=False).relative_to(expected_root.resolve())
        except ValueError as exc:
            raise CoordinatorError(f"{label} is outside its storage root") from exc
        return path

    def _finalize_promotion_locked(
        self,
        promotion: sqlite3.Row,
        now: float,
    ) -> dict[str, Any]:
        artifact_id = str(promotion["artifact_id"])
        destination = self._promotion_path(
            promotion["destination_path"],
            self.committed_root,
            "artifact destination path",
        )
        manifest = self._validated_artifact_manifest(
            self._loads(promotion["manifest_json"])
        )
        self._verify_workspace(destination, manifest, allow_root_manifest=True)
        registered = self._connection.execute(
            "SELECT * FROM artifact_registry WHERE artifact_id = ?",
            (artifact_id,),
        ).fetchone()
        if registered is None:
            self._connection.execute(
                """
                INSERT INTO artifact_registry (
                    artifact_id, job_id, attempt_id, namespace,
                    relative_path, manifest_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact_id,
                    promotion["job_id"],
                    promotion["attempt_id"],
                    promotion["namespace"],
                    promotion["destination_path"],
                    promotion["manifest_json"],
                    promotion["authorized_at"] or promotion["created_at"],
                ),
            )
            self._append_event(
                promotion["job_id"],
                int(promotion["attempt_id"]),
                "artifact.promoted",
                {
                    "artifact_id": artifact_id,
                    "namespace": promotion["namespace"],
                    "relative_path": promotion["destination_path"],
                },
                now,
            )
        self._connection.execute(
            "DELETE FROM artifact_promotions WHERE artifact_id = ?",
            (artifact_id,),
        )
        registered = self._connection.execute(
            "SELECT * FROM artifact_registry WHERE artifact_id = ?",
            (artifact_id,),
        ).fetchone()
        if registered is None:
            raise CoordinatorError("artifact registry insert failed")
        return self._public_artifact(registered)

    def _restore_failed_promotion(
        self,
        *,
        artifact_id: str,
        workspace: Path,
        staging: Path,
    ) -> None:
        if staging.exists() or staging.is_symlink():
            self._restore_staging_path(staging, workspace)

        def cleanup() -> None:
            row = self._connection.execute(
                "SELECT destination_path FROM artifact_promotions WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchone()
            if row is None:
                return
            destination = self._promotion_path(
                row["destination_path"],
                self.committed_root,
                "artifact destination path",
            )
            if destination.exists() or destination.is_symlink():
                return
            self._connection.execute(
                "DELETE FROM artifact_promotions WHERE artifact_id = ?",
                (artifact_id,),
            )

        self._mutate(cleanup)

    def _restore_staging_path(self, staging: Path, workspace: Path) -> None:
        if staging.is_symlink() or not staging.is_dir():
            self._quarantine_path(staging, "invalid-staging")
            return
        staging.chmod(0o755)
        for path in staging.rglob("*"):
            if path.is_symlink():
                self._quarantine_path(staging, "invalid-staging")
                return
            path.chmod(0o755 if path.is_dir() else 0o644)
        manifest_path = staging / "artifact_manifest.json"
        if manifest_path.exists() or manifest_path.is_symlink():
            try:
                manifest_path.chmod(0o600)
            except OSError:
                pass
            manifest_path.unlink()
        if workspace.exists() or workspace.is_symlink():
            self._quarantine_path(staging, "duplicate-workspace")
            return
        try:
            staging.rename(workspace)
        except OSError as exc:
            raise CoordinatorError("could not restore failed attempt workspace") from exc
        self._fsync_directory(self.staging_root)

    @staticmethod
    def _ensure_private_directory(path: Path) -> None:
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError:
            path.mkdir(mode=0o700, parents=False, exist_ok=False)
        else:
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise CoordinatorError(f"unsafe artifact directory: {path}")
        path.chmod(0o700)

    def _attempt_workspace_path(self, attempt_id: int, token_hash: str) -> Path:
        if attempt_id < 1:
            raise CoordinatorError("invalid attempt identifier")
        if (
            len(token_hash) != 64
            or token_hash.lower() != token_hash
            or any(character not in "0123456789abcdef" for character in token_hash)
        ):
            raise CoordinatorError("invalid attempt token hash")
        return self.attempt_root / f"{attempt_id}-{token_hash[:32]}"

    def _mutate(self, body: Any) -> Any:
        def transaction() -> Any:
            with self._writer_lock:
                self._connection.execute("BEGIN IMMEDIATE")
                try:
                    result = body()
                except Exception:
                    self._connection.execute("ROLLBACK")
                    raise
                item = getattr(self, "_active_queue_item", None)
                if item is None:
                    self._connection.execute("COMMIT")
                    return result
                with item.state_lock:
                    if item.cancelled or time.monotonic() >= item.deadline:
                        self._connection.execute("ROLLBACK")
                        raise CoordinatorError("coordinator writer queue timed out")
                    self._connection.execute("COMMIT")
                    item.committed = True
                return result

        return self._submit(transaction)

    def _read(self, body: Any) -> Any:
        if threading.get_ident() == self._writer_thread_id:
            return body()
        return self._submit(body)

    def _submit(self, callback: Any) -> Any:
        if threading.get_ident() == self._writer_thread_id:
            return callback()
        if self._closed:
            raise CoordinatorError("coordinator is closed")
        item = _QueueItem(
            callback=callback,
            completed=threading.Event(),
            deadline=time.monotonic() + self._queue_timeout_seconds,
        )
        self._work_queue.put(item)
        if not item.completed.wait(timeout=self._queue_timeout_seconds):
            with item.state_lock:
                if not item.completed.is_set() and not item.committed:
                    item.cancelled = True
                    raise CoordinatorError("coordinator writer queue timed out")
            item.completed.wait()
        if item.error is not None:
            raise item.error
        return item.result

    def _writer_loop(self) -> None:
        self._writer_thread_id = threading.get_ident()
        while True:
            item = self._work_queue.get()
            if item is None:
                return
            with item.state_lock:
                if item.cancelled or time.monotonic() >= item.deadline:
                    item.error = CoordinatorError("coordinator writer queue timed out")
                    item.completed.set()
                    continue
            try:
                self._active_queue_item = item
                item.result = item.callback()
            except BaseException as exc:
                item.error = exc
            finally:
                self._active_queue_item = None
                item.completed.set()

    def _claim_locked(
        self,
        job: sqlite3.Row,
        worker_id: str,
        resource: str,
        now: float,
    ) -> ClaimedAttempt:
        token = secrets.token_hex(32)
        token_hash = self._hash_token(token)
        cursor = self._connection.execute(
            """
            INSERT INTO attempts (
                job_id, worker_id, status, token_hash, lease_resource,
                started_at, last_heartbeat_at, expires_at
            )
            VALUES (?, ?, 'active', ?, ?, ?, ?, ?)
            """,
            (
                job["job_id"],
                worker_id,
                token_hash,
                resource,
                now,
                now,
                now + ATTEMPT_EXPIRY_SECONDS,
            ),
        )
        attempt_id = int(cursor.lastrowid)
        self._connection.execute(
            """
            INSERT INTO resource_leases (
                resource, job_id, attempt_id, worker_id, acquired_at, expires_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                resource,
                job["job_id"],
                attempt_id,
                worker_id,
                now,
                now + ATTEMPT_EXPIRY_SECONDS,
            ),
        )
        self._set_job_status(job["job_id"], "running", now)
        self._append_event(
            job["job_id"],
            attempt_id,
            "attempt.claimed",
            {"worker_id": worker_id, "lease_resource": resource},
            now,
        )
        return ClaimedAttempt(job["job_id"], attempt_id, token, resource)

    def _expire_locked(self, now: float) -> dict[str, int]:
        expired_attempts = self._connection.execute(
            """
            SELECT * FROM attempts
            WHERE status = 'active' AND expires_at <= ?
            ORDER BY attempt_id ASC
            """,
            (now,),
        ).fetchall()
        recovered = 0
        for attempt in expired_attempts:
            job = self._job_row(attempt["job_id"])
            self._connection.execute(
                "UPDATE attempts SET status = 'expired', ended_at = ? WHERE attempt_id = ?",
                (now, attempt["attempt_id"]),
            )
            self._release_lease(attempt["attempt_id"])
            self._append_event(
                attempt["job_id"],
                attempt["attempt_id"],
                "attempt.expired",
                {"lease_resource": attempt["lease_resource"]},
                now,
            )
            if job["status"] == "running":
                if self._attempt_count(attempt["job_id"]) < int(job["max_attempts"]):
                    self._set_job_status(attempt["job_id"], "queued", now)
                    self._append_event(
                        attempt["job_id"],
                        None,
                        "job.requeued",
                        {"reason": "attempt_expired"},
                        now,
                    )
                    recovered += 1
                else:
                    self._set_job_status(attempt["job_id"], "failed", now)
                    self._append_event(
                        attempt["job_id"],
                        None,
                        "job.failed",
                        {"reason": "attempts_exhausted"},
                        now,
                    )
            elif job["status"] == "cancel_requested":
                self._connection.execute(
                    "UPDATE attempts SET status = 'cancelled' WHERE attempt_id = ?",
                    (attempt["attempt_id"],),
                )
                self._set_job_status(attempt["job_id"], "cancelled", now)
                self._append_event(
                    attempt["job_id"],
                    None,
                    "job.cancelled",
                    {"reason": "attempt_expired_after_cancel"},
                    now,
                )
        stale_leases = self._connection.execute(
            "SELECT attempt_id FROM resource_leases WHERE expires_at <= ?",
            (now,),
        ).fetchall()
        for lease in stale_leases:
            self._release_lease(int(lease["attempt_id"]))
        return {"expired_attempts": len(expired_attempts), "requeued_jobs": recovered}

    def _active_attempt_or_none(
        self,
        attempt_id: int,
        token: str,
        now: float,
    ) -> sqlite3.Row | None:
        attempt = self._connection.execute(
            "SELECT * FROM attempts WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone()
        if attempt is None or not secrets.compare_digest(
            attempt["token_hash"],
            self._hash_token(token),
        ):
            raise AuthError("invalid attempt token")
        if attempt["status"] != "active" or float(attempt["expires_at"]) <= now:
            if attempt["status"] == "active":
                self._expire_locked(now)
            return None
        lease = self._connection.execute(
            "SELECT * FROM resource_leases WHERE attempt_id = ? AND resource = ?",
            (attempt_id, attempt["lease_resource"]),
        ).fetchone()
        if lease is None:
            raise InvalidTransition("attempt does not hold its lease")
        return attempt

    def _append_event(
        self,
        job_id: str,
        attempt_id: int | None,
        event_type: str,
        payload: dict[str, Any],
        now: float,
    ) -> dict[str, Any]:
        row = self._connection.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 AS next_seq FROM events WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        seq = int(row["next_seq"])
        cursor = self._connection.execute(
            """
            INSERT INTO events (job_id, attempt_id, seq, type, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (job_id, attempt_id, seq, event_type, self._json(payload), now),
        )
        return {
            "event_id": int(cursor.lastrowid),
            "job_id": job_id,
            "attempt_id": attempt_id,
            "seq": seq,
            "type": event_type,
            "payload": payload,
            "created_at": now,
        }

    @classmethod
    def _validated_artifact_manifest(
        cls,
        manifest: dict[str, Any],
    ) -> dict[str, Any]:
        if set(manifest) != {"schema_version", "files"}:
            raise ValueError("artifact manifest fields are invalid")
        if manifest.get("schema_version") != "skinscout.attempt_artifacts.v1":
            raise ValueError("artifact manifest schema_version is invalid")
        files = manifest.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError("artifact manifest files must be a non-empty list")
        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        for index, entry in enumerate(files, start=1):
            if not isinstance(entry, dict) or set(entry) != {"path", "bytes", "sha256"}:
                raise ValueError(f"artifact manifest file {index} fields are invalid")
            relative = cls._safe_relative_path(
                entry.get("path"),
                f"artifact manifest file {index} path",
            ).as_posix()
            if Path(relative).name == "artifact_manifest.json":
                raise ValueError("artifact manifest filename is reserved")
            if relative in seen:
                raise ValueError(f"artifact manifest path is duplicated: {relative}")
            seen.add(relative)
            size = entry.get("bytes")
            if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                raise ValueError(f"artifact manifest file {index} bytes are invalid")
            sha256 = entry.get("sha256")
            if (
                not isinstance(sha256, str)
                or len(sha256) != 64
                or any(character not in "0123456789abcdef" for character in sha256)
            ):
                raise ValueError(f"artifact manifest file {index} sha256 is invalid")
            normalized.append({"path": relative, "bytes": size, "sha256": sha256})
        normalized.sort(key=lambda entry: entry["path"])
        return {
            "schema_version": "skinscout.attempt_artifacts.v1",
            "files": normalized,
        }

    @staticmethod
    def _safe_relative_path(value: Any, label: str) -> Path:
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise ValueError(f"{label} must be a non-empty relative path")
        path = Path(value)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError(f"{label} must not escape its root")
        return path

    def _verify_workspace(
        self,
        workspace: Path,
        manifest: dict[str, Any],
        *,
        allow_root_manifest: bool = False,
    ) -> None:
        if not workspace.is_dir() or workspace.is_symlink():
            raise InvalidTransition("attempt workspace is missing or invalid")
        expected = {str(entry["path"]): entry for entry in manifest["files"]}
        observed: set[str] = set()
        for path in workspace.rglob("*"):
            if path.is_symlink():
                raise InvalidTransition("attempt workspace cannot contain symlinks")
            if not path.is_file():
                continue
            relative = path.relative_to(workspace).as_posix()
            if relative == "artifact_manifest.json" and allow_root_manifest:
                expected_manifest = json.dumps(
                    manifest,
                    sort_keys=True,
                    separators=(",", ":"),
                ) + "\n"
                try:
                    observed_manifest = path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError) as exc:
                    raise InvalidTransition("artifact manifest is unreadable") from exc
                if observed_manifest != expected_manifest:
                    raise InvalidTransition("artifact manifest does not match the registry")
                continue
            observed.add(relative)
            entry = expected.get(relative)
            if entry is None:
                raise InvalidTransition(f"attempt workspace file is unmanifested: {relative}")
            data = path.read_bytes()
            if len(data) != entry["bytes"]:
                raise InvalidTransition(f"artifact bytes mismatch: {relative}")
            if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise InvalidTransition(f"artifact sha256 mismatch: {relative}")
        missing = sorted(set(expected) - observed)
        if missing:
            raise InvalidTransition(
                "attempt workspace is missing manifest files: " + ", ".join(missing)
            )

    @staticmethod
    def _fsync_file(path: Path) -> None:
        flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise CoordinatorError(
                f"could not open artifact file for durable publish: {path}"
            ) from exc
        try:
            if not path.is_file() or path.is_symlink():
                raise InvalidTransition("artifact publish tree contains unsafe file")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise CoordinatorError(
                f"could not open artifact directory for durable publish: {path}"
            ) from exc
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _fsync_tree(self, root: Path) -> None:
        directories = [root]
        for path in root.rglob("*"):
            if path.is_symlink():
                raise InvalidTransition("artifact publish tree cannot contain symlinks")
            if path.is_file():
                self._fsync_file(path)
            elif path.is_dir():
                directories.append(path)
            else:
                raise InvalidTransition("artifact publish tree contains unsupported entry")
        for directory in sorted(
            directories,
            key=lambda candidate: len(candidate.parts),
            reverse=True,
        ):
            self._fsync_directory(directory)

    def _fsync_directory_chain(self, path: Path, stop: Path) -> None:
        current = path.resolve(strict=True)
        stop_resolved = stop.resolve(strict=True)
        try:
            current.relative_to(stop_resolved)
        except ValueError as exc:
            raise CoordinatorError("artifact publish directory escapes storage root") from exc
        while True:
            self._fsync_directory(current)
            if current == stop_resolved:
                return
            current = current.parent

    @staticmethod
    def _seal_tree(root: Path, *, seal_root: bool = True) -> None:
        paths = sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True)
        for path in paths:
            if path.is_symlink():
                raise InvalidTransition("artifact tree cannot contain symlinks")
            if path.is_file():
                path.chmod(0o444)
            elif path.is_dir():
                path.chmod(0o555)
        if seal_root:
            root.chmod(0o555)

    @staticmethod
    def _remove_tree(path: Path) -> None:
        if not path.exists() and not path.is_symlink():
            return
        if path.is_symlink() or path.is_file():
            path.unlink()
            return
        for child in path.rglob("*"):
            if not child.is_symlink():
                try:
                    child.chmod(0o755 if child.is_dir() else 0o644)
                except OSError:
                    pass
        path.chmod(0o755)
        shutil.rmtree(path)

    def _quarantine_path(self, path: Path, reason: str) -> None:
        destination = self.quarantine_root / (
            f"{reason}-{path.name}-{secrets.token_hex(8)}"
        )
        try:
            path.rename(destination)
        except OSError as exc:
            raise CoordinatorError(
                f"could not quarantine invalid artifact path: {path}"
            ) from exc

    def _write_text_atomic(self, path: Path, text: str) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            os.write(descriptor, text.encode("utf-8"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        temporary.replace(path)
        self._fsync_directory(path.parent)

    def _public_artifact(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "artifact_id": row["artifact_id"],
            "job_id": row["job_id"],
            "attempt_id": row["attempt_id"],
            "namespace": row["namespace"],
            "relative_path": row["relative_path"],
            "manifest": self._loads(row["manifest_json"]),
            "created_at": row["created_at"],
        }

    def _set_job_status(self, job_id: str, status: str, now: float) -> None:
        self._connection.execute(
            "UPDATE jobs SET status = ?, updated_at = ? WHERE job_id = ?",
            (status, now, job_id),
        )

    def _release_lease(self, attempt_id: int) -> None:
        self._connection.execute("DELETE FROM resource_leases WHERE attempt_id = ?", (attempt_id,))

    def _active_lease(self, resource: str) -> sqlite3.Row | None:
        return self._connection.execute(
            "SELECT * FROM resource_leases WHERE resource = ?",
            (resource,),
        ).fetchone()

    def _job_row(self, job_id: str) -> sqlite3.Row:
        row = self._connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return row

    def _attempt_count(self, job_id: str) -> int:
        row = self._connection.execute(
            "SELECT COUNT(*) AS count FROM attempts WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        return int(row["count"])

    def _public_job(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "job_id": row["job_id"],
            "kind": row["kind"],
            "status": row["status"],
            "payload": self._loads(row["payload_json"]),
            "priority": row["priority"],
            "max_attempts": row["max_attempts"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _now(self) -> float:
        return float(self._clock())

    @staticmethod
    def _hash_token(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    @classmethod
    def _reject_sensitive(cls, payload: Any, path: str = "payload") -> None:
        if isinstance(payload, dict):
            for key, value in payload.items():
                normalized = str(key).strip().lower()
                if normalized in SENSITIVE_FIELD_NAMES or normalized.endswith(
                    SENSITIVE_FIELD_SUFFIXES
                ):
                    raise ValueError(f"sensitive field is not serializable: {path}.{key}")
                cls._reject_sensitive(value, f"{path}.{key}")
        elif isinstance(payload, list):
            for index, value in enumerate(payload):
                cls._reject_sensitive(value, f"{path}[{index}]")

    @staticmethod
    def _json(payload: dict[str, Any]) -> str:
        import json

        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _loads(payload: str) -> dict[str, Any]:
        import json

        value = json.loads(payload)
        return value if isinstance(value, dict) else {}
