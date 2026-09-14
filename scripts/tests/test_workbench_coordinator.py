"""Contract tests for the durable workbench coordinator."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from workbench.coordinator import (
    ATTEMPT_EXPIRY_SECONDS,
    AuthError,
    CoordinatorError,
    DurableCoordinator,
    InvalidTransition,
    LeaseUnavailable,
)


class ManualClock:
    def __init__(self, value: float = 1_000.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _artifact_manifest(files: dict[str, bytes]) -> dict[str, object]:
    return {
        "schema_version": "skinscout.attempt_artifacts.v1",
        "files": [
            {
                "path": path,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            for path, content in sorted(files.items())
        ],
    }


@pytest.fixture
def coordinator(tmp_path: Path) -> DurableCoordinator:
    db = tmp_path / "coordinator.sqlite3"
    coord = DurableCoordinator(db, clock=ManualClock())
    try:
        yield coord
    finally:
        coord.close()


def test_coordinator_uses_full_sqlite_synchronous_mode(
    coordinator: DurableCoordinator,
) -> None:
    synchronous = coordinator.connection.execute("PRAGMA synchronous").fetchone()[0]
    assert synchronous == 2


def test_auth_stores_only_hash_and_public_payloads_redact_token(tmp_path: Path) -> None:
    clock = ManualClock()
    coord = DurableCoordinator(tmp_path / "state.sqlite3", clock=clock)
    try:
        coord.create_job("target_fast", {"compound": "case"}, job_id="job-auth")
        attempt = coord.claim_next(worker_id="worker-1")
        assert attempt is not None
        assert len(attempt.token) == 64
        assert attempt.token not in repr(attempt)
        assert attempt.token not in str(attempt.public())
        assert "token" not in attempt.public()

        row = coord.connection.execute("SELECT token_hash FROM attempts").fetchone()
        assert row["token_hash"] != attempt.token
        assert len(row["token_hash"]) == 64
        assert attempt.token not in coord.db_path.read_bytes().decode("latin1", errors="ignore")
        assert attempt.token not in str(coord.get_job("job-auth"))

        with pytest.raises(AuthError):
            coord.heartbeat(attempt.attempt_id, "wrong-token")
        assert coord.get_job("job-auth")["status"] == "running"
    finally:
        coord.close()


def test_attempt_workspace_is_token_bound_private_and_exclusive(
    tmp_path: Path,
) -> None:
    coord = DurableCoordinator(tmp_path / "state.sqlite3")
    try:
        coord.create_job("report_fast", job_id="workspace-path")
        attempt = coord.claim_next(worker_id="worker-1")
        assert attempt is not None
        predictable_path = coord.attempt_root / str(attempt.attempt_id)
        attacker_target = tmp_path / "attacker-target"
        attacker_target.mkdir()
        predictable_path.symlink_to(attacker_target, target_is_directory=True)

        workspace = coord.attempt_workspace(attempt.attempt_id, attempt.token)

        assert workspace.parent == coord.attempt_root
        assert workspace != predictable_path
        assert workspace.name.startswith(f"{attempt.attempt_id}-")
        assert not workspace.is_symlink()
        assert workspace.stat().st_mode & 0o777 == 0o700
        with pytest.raises(CoordinatorError, match="already exists"):
            coord.attempt_workspace(attempt.attempt_id, attempt.token)
    finally:
        coord.close()


def test_coordinator_rejects_symlinked_artifact_root(tmp_path: Path) -> None:
    real_root = tmp_path / "real-artifacts"
    real_root.mkdir()
    linked_root = tmp_path / "linked-artifacts"
    linked_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(CoordinatorError, match="unsafe artifact directory"):
        DurableCoordinator(
            tmp_path / "state" / "coordinator.sqlite3",
            artifact_root=linked_root,
        )


def test_append_only_event_sequence_is_monotonic(coordinator: DurableCoordinator) -> None:
    coord = coordinator
    coord.create_job("target_fast", job_id="job-events")
    attempt = coord.claim_next(worker_id="worker-1")
    assert attempt is not None
    coord.publish_event(attempt.attempt_id, attempt.token, "stage.started", {"stage": "dock"})
    coord.heartbeat(attempt.attempt_id, attempt.token)
    coord.complete_attempt(attempt.attempt_id, attempt.token, result={"artifact": "fast"})

    events = coord.list_events("job-events")
    assert [event["seq"] for event in events] == [1, 2, 3, 4, 5]
    assert [event["type"] for event in events] == [
        "job.created",
        "attempt.claimed",
        "stage.started",
        "attempt.heartbeat",
        "job.completed",
    ]
    with pytest.raises(sqlite3.IntegrityError):
        coord.connection.execute(
            """
            INSERT INTO events (job_id, seq, type, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("job-events", 5, "duplicate", "{}", 1.0),
        )
    with pytest.raises(sqlite3.IntegrityError):
        coord.connection.execute(
            "UPDATE events SET type = 'mutated' WHERE job_id = ?",
            ("job-events",),
        )
    with pytest.raises(sqlite3.IntegrityError):
        coord.connection.execute("DELETE FROM events WHERE job_id = ?", ("job-events",))


def test_gpu_lease_is_exclusive_until_release(coordinator: DurableCoordinator) -> None:
    coord = coordinator
    coord.create_job("target_fast", job_id="first")
    coord.create_job("target_fast", job_id="second")
    first = coord.claim_next(worker_id="worker-1", resource="gpu:0")
    assert first is not None

    with pytest.raises(LeaseUnavailable):
        coord.claim_next(worker_id="worker-2", resource="gpu:0")

    coord.complete_attempt(first.attempt_id, first.token)
    second = coord.claim_next(worker_id="worker-2", resource="gpu:0")
    assert second is not None
    assert second.job_id == "second"


def test_expiry_and_restart_recovery_releases_lease_and_requeues(tmp_path: Path) -> None:
    clock = ManualClock()
    db = tmp_path / "restart.sqlite3"
    coord = DurableCoordinator(db, clock=clock)
    try:
        coord.create_job("target_fast", job_id="job-recover")
        attempt = coord.claim_next(worker_id="worker-1")
        assert attempt is not None
    finally:
        coord.close()

    clock.advance(ATTEMPT_EXPIRY_SECONDS + 1)
    restarted = DurableCoordinator(db, clock=clock)
    try:
        result = restarted.recover()
        assert result == {"expired_attempts": 1, "requeued_jobs": 1}
        assert restarted.get_job("job-recover")["status"] == "queued"
        lease_count = restarted.connection.execute(
            "SELECT COUNT(*) AS n FROM resource_leases"
        ).fetchone()["n"]
        assert lease_count == 0
        retry = restarted.claim_next(worker_id="worker-2")
        assert retry is not None
        assert retry.job_id == "job-recover"
        assert retry.attempt_id != attempt.attempt_id
    finally:
        restarted.close()


def test_expired_attempt_cannot_publish_or_complete(tmp_path: Path) -> None:
    clock = ManualClock()
    coord = DurableCoordinator(tmp_path / "expiry.sqlite3", clock=clock)
    try:
        coord.create_job("target_fast", job_id="job-expired")
        attempt = coord.claim_next(worker_id="worker-1")
        assert attempt is not None
        clock.advance(ATTEMPT_EXPIRY_SECONDS + 1)

        with pytest.raises(InvalidTransition):
            coord.publish_event(attempt.attempt_id, attempt.token, "stage.done")
        with pytest.raises(InvalidTransition):
            coord.complete_attempt(attempt.attempt_id, attempt.token)

        assert coord.get_job("job-expired")["status"] == "queued"
        assert [event["type"] for event in coord.list_events("job-expired")] == [
            "job.created",
            "attempt.claimed",
            "attempt.expired",
            "job.requeued",
        ]
    finally:
        coord.close()


def test_cancellation_is_idempotent_and_blocks_worker_completion(
    coordinator: DurableCoordinator,
) -> None:
    coord = coordinator
    coord.create_job("target_fast", job_id="job-cancel")
    attempt = coord.claim_next(worker_id="worker-1")
    assert attempt is not None

    first = coord.cancel_job("job-cancel")
    second = coord.cancel_job("job-cancel")
    assert first == second
    assert first["status"] == "cancel_requested"
    assert coord.heartbeat(attempt.attempt_id, attempt.token)["cancel_requested"] is True
    with pytest.raises(InvalidTransition):
        coord.complete_attempt(attempt.attempt_id, attempt.token)
    cancel_events = [event["type"] for event in coord.list_events("job-cancel")]
    assert cancel_events.count("job.cancel_requested") == 1

    coord.create_job("target_fast", job_id="queued-cancel")
    assert coord.cancel_job("queued-cancel")["status"] == "cancelled"
    assert coord.cancel_job("queued-cancel")["status"] == "cancelled"
    queued_cancel_events = [event["type"] for event in coord.list_events("queued-cancel")]
    assert queued_cancel_events.count("job.cancelled") == 1


def test_sqlite_uses_wal_and_every_mutation_begins_immediate(tmp_path: Path) -> None:
    traces: list[str] = []
    coord = DurableCoordinator(tmp_path / "wal.sqlite3")
    coord.connection.set_trace_callback(traces.append)
    try:
        assert coord.connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        coord.create_job("target_fast", job_id="job-wal")
        attempt = coord.claim_next(worker_id="worker-1")
        assert attempt is not None
        coord.publish_event(attempt.attempt_id, attempt.token, "stage")
        coord.complete_attempt(attempt.attempt_id, attempt.token)
    finally:
        coord.close()

    begin_statements = [trace for trace in traces if trace.upper().startswith("BEGIN")]
    assert begin_statements
    assert all(trace.upper() == "BEGIN IMMEDIATE" for trace in begin_statements)


def test_writer_queue_serializes_concurrent_mutations(tmp_path: Path) -> None:
    coord = DurableCoordinator(tmp_path / "queue.sqlite3")
    errors: list[BaseException] = []

    def create(index: int) -> None:
        try:
            coord.create_job("target_fast", job_id=f"job-{index:03d}")
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=create, args=(index,)) for index in range(20)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        assert not errors
        assert len(coord.list_jobs(limit=100)) == 20
    finally:
        coord.close()


def test_timed_out_queued_mutation_never_commits(tmp_path: Path) -> None:
    coord = DurableCoordinator(
        tmp_path / "queue-timeout.sqlite3",
        queue_timeout_seconds=0.1,
    )
    blocker_started = threading.Event()
    release_blocker = threading.Event()
    blocker_errors: list[BaseException] = []

    def block_writer() -> None:
        try:
            coord._submit(  # noqa: SLF001
                lambda: (blocker_started.set(), release_blocker.wait(timeout=2))
            )
        except BaseException as exc:
            blocker_errors.append(exc)

    try:
        blocker = threading.Thread(target=block_writer)
        blocker.start()
        assert blocker_started.wait(timeout=1)

        with pytest.raises(CoordinatorError, match="timed out"):
            coord.create_job("target_fast", job_id="must-not-appear")

        release_blocker.set()
        blocker.join(timeout=1)
        deadline = time.monotonic() + 1
        while coord._work_queue.qsize() and time.monotonic() < deadline:  # noqa: SLF001
            time.sleep(0.01)
        assert coord.connection.execute(
            "SELECT COUNT(*) AS n FROM jobs WHERE job_id = ?",
            ("must-not-appear",),
        ).fetchone()["n"] == 0
        assert all(isinstance(error, CoordinatorError) for error in blocker_errors)
    finally:
        release_blocker.set()
        coord.close()


def test_list_jobs_applies_status_filter_before_limit(tmp_path: Path) -> None:
    coord = DurableCoordinator(tmp_path / "filtered-list.sqlite3")
    try:
        coord.create_job("target_fast", job_id="running-oldest")
        attempt = coord.claim_job(
            "running-oldest",
            worker_id="worker-1",
            resource="cpu:test",
        )
        for index in range(12):
            coord.create_job("target_fast", job_id=f"newer-{index:02d}")

        rows = coord.list_jobs(statuses={"running"}, limit=5)

        assert [row["job_id"] for row in rows] == ["running-oldest"]
        coord.complete_attempt(attempt.attempt_id, attempt.token)
    finally:
        coord.close()


def test_cancel_ack_releases_lease_and_is_terminal(coordinator: DurableCoordinator) -> None:
    coord = coordinator
    coord.create_job("target_fast", job_id="cancel-ack")
    attempt = coord.claim_job("cancel-ack", worker_id="worker-1")
    coord.cancel_job("cancel-ack")

    cancelled = coord.acknowledge_cancel(attempt.attempt_id, attempt.token)

    assert cancelled["status"] == "cancelled"
    with pytest.raises(InvalidTransition):
        coord.complete_attempt(attempt.attempt_id, attempt.token)
    assert coord.connection.execute(
        "SELECT COUNT(*) AS n FROM resource_leases"
    ).fetchone()["n"] == 0


def test_sensitive_payload_fields_are_never_serialized(
    coordinator: DurableCoordinator,
) -> None:
    coord = coordinator
    with pytest.raises(ValueError, match="sensitive field"):
        coord.create_job("target_fast", {"worker_api_token": "never-store"})
    with pytest.raises(ValueError, match="sensitive field"):
        coord.create_job("target_fast", {"api_key": "never-store"})
    with pytest.raises(ValueError, match="sensitive field"):
        coord.create_job("target_fast", {"oauth_client_secret": "never-store"})
    coord.create_job("target_fast", job_id="safe")
    attempt = coord.claim_next(worker_id="worker-1")
    assert attempt is not None
    with pytest.raises(ValueError, match="sensitive field"):
        coord.publish_event(
            attempt.attempt_id,
            attempt.token,
            "unsafe",
            {"nested": {"authorization": "Bearer secret"}},
        )


def test_expiry_is_evaluated_when_queued_mutation_executes(tmp_path: Path) -> None:
    clock = ManualClock()
    coord = DurableCoordinator(tmp_path / "queued-expiry.sqlite3", clock=clock)
    blocker_started = threading.Event()
    release_blocker = threading.Event()
    publish_errors: list[BaseException] = []
    try:
        coord.create_job("target_fast", job_id="queued-expiry")
        attempt = coord.claim_next(worker_id="worker-1")
        assert attempt is not None

        blocker = threading.Thread(
            target=lambda: coord._submit(  # noqa: SLF001
                lambda: (blocker_started.set(), release_blocker.wait(timeout=5))
            )
        )
        blocker.start()
        assert blocker_started.wait(timeout=5)

        def publish() -> None:
            try:
                coord.publish_event(
                    attempt.attempt_id,
                    attempt.token,
                    "late",
                )
            except BaseException as exc:
                publish_errors.append(exc)

        publisher = threading.Thread(target=publish)
        publisher.start()
        deadline = time.monotonic() + 5
        while coord._work_queue.qsize() < 1 and time.monotonic() < deadline:  # noqa: SLF001
            time.sleep(0.01)
        clock.advance(ATTEMPT_EXPIRY_SECONDS + 1)
        release_blocker.set()
        blocker.join(timeout=5)
        publisher.join(timeout=5)

        assert len(publish_errors) == 1
        assert isinstance(publish_errors[0], InvalidTransition)
        assert "late" not in [
            event["type"] for event in coord.list_events("queued-expiry")
        ]
        assert coord.get_job("queued-expiry")["status"] == "queued"
    finally:
        release_blocker.set()
        coord.close()


def test_artifact_promotion_snapshots_and_registers_immutable_files(
    tmp_path: Path,
) -> None:
    db = tmp_path / "state" / "coordinator.sqlite3"
    artifact_root = tmp_path / "results"
    coord = DurableCoordinator(db, artifact_root=artifact_root)
    files = {
        "index.html": b"<html>sealed</html>\n",
        "assets/pose.sdf": b"pose\n",
    }
    try:
        coord.create_job("report_fast", job_id="artifact-job")
        attempt = coord.claim_next(worker_id="worker-1")
        assert attempt is not None
        workspace = coord.attempt_workspace(attempt.attempt_id, attempt.token)
        for relative, content in files.items():
            path = workspace / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)

        artifact = coord.promote_attempt_artifacts(
            attempt.attempt_id,
            attempt.token,
            namespace="runs/artifact-job/reports/fast",
            manifest=_artifact_manifest(files),
        )

        assert not workspace.exists()
        assert artifact["namespace"] == "runs/artifact-job/reports/fast"
        assert coord.list_artifacts("artifact-job") == [artifact]
        resolved, entry, path = coord.resolve_artifact_file(
            artifact["artifact_id"],
            "assets/pose.sdf",
        )
        assert resolved == artifact
        assert entry["sha256"] == hashlib.sha256(files["assets/pose.sdf"]).hexdigest()
        assert path.read_bytes() == files["assets/pose.sdf"]
        assert path.stat().st_mode & 0o222 == 0
        manifest_path = path.parents[1] / "artifact_manifest.json"
        assert json.loads(manifest_path.read_text()) == artifact["manifest"]
        assert [event["type"] for event in coord.list_events("artifact-job")][-1] == (
            "artifact.promoted"
        )
    finally:
        coord.close()

    restarted = DurableCoordinator(db, artifact_root=artifact_root)
    try:
        assert restarted.list_artifacts("artifact-job")[0]["artifact_id"] == artifact[
            "artifact_id"
        ]
    finally:
        restarted.close()


def test_artifact_promotion_fsyncs_tree_and_publish_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coord = DurableCoordinator(tmp_path / "state.sqlite3")
    tree_calls: list[Path] = []
    directory_calls: list[Path] = []
    real_fsync_tree = coord._fsync_tree
    real_fsync_directory = coord._fsync_directory

    def track_tree(path: Path) -> None:
        tree_calls.append(path)
        real_fsync_tree(path)

    def track_directory(path: Path) -> None:
        directory_calls.append(path)
        real_fsync_directory(path)

    monkeypatch.setattr(coord, "_fsync_tree", track_tree)
    monkeypatch.setattr(coord, "_fsync_directory", track_directory)
    try:
        coord.create_job("report_fast", job_id="artifact-fsync")
        attempt = coord.claim_next(worker_id="worker-1")
        assert attempt is not None
        workspace = coord.attempt_workspace(attempt.attempt_id, attempt.token)
        content = b"durable"
        workspace.joinpath("index.html").write_bytes(content)

        artifact = coord.promote_attempt_artifacts(
            attempt.attempt_id,
            attempt.token,
            namespace="runs/artifact-fsync/reports/fast",
            manifest=_artifact_manifest({"index.html": content}),
        )

        assert len(tree_calls) == 1
        assert tree_calls[0].parent == coord.staging_root
        destination = coord.artifact_root / artifact["relative_path"]
        assert coord.staging_root in directory_calls
        assert destination.parent in directory_calls
        assert coord.committed_root in directory_calls
    finally:
        coord.close()


def test_artifact_promotion_failure_keeps_workspace_and_registry_clean(
    tmp_path: Path,
) -> None:
    coord = DurableCoordinator(tmp_path / "state.sqlite3")
    try:
        coord.create_job("report_fast", job_id="artifact-fail")
        attempt = coord.claim_next(worker_id="worker-1")
        assert attempt is not None
        workspace = coord.attempt_workspace(attempt.attempt_id, attempt.token)
        workspace.joinpath("index.html").write_bytes(b"actual")

        with pytest.raises(InvalidTransition, match="sha256 mismatch"):
            coord.promote_attempt_artifacts(
                attempt.attempt_id,
                attempt.token,
                namespace="runs/artifact-fail/reports/fast",
                manifest=_artifact_manifest({"index.html": b"expect"}),
            )

        assert workspace.joinpath("index.html").read_bytes() == b"actual"
        assert coord.list_artifacts("artifact-fail") == []
        assert not list(coord.committed_root.rglob("artifact_manifest.json"))
    finally:
        coord.close()


def test_authorized_artifact_publication_is_finalized_after_restart(
    tmp_path: Path,
) -> None:
    db = tmp_path / "state.sqlite3"
    coord = DurableCoordinator(db)
    artifact_id = ""
    try:
        coord.create_job("report_fast", job_id="artifact-rollback")
        attempt = coord.claim_next(worker_id="worker-1")
        assert attempt is not None
        workspace = coord.attempt_workspace(attempt.attempt_id, attempt.token)
        content = b"sealed"
        workspace.joinpath("index.html").write_bytes(content)
        coord.connection.execute(
            """
            CREATE TRIGGER block_artifact_insert
            BEFORE INSERT ON artifact_registry
            BEGIN
                SELECT RAISE(ABORT, 'blocked for test');
            END
            """
        )

        with pytest.raises(sqlite3.IntegrityError, match="blocked for test"):
            coord.promote_attempt_artifacts(
                attempt.attempt_id,
                attempt.token,
                namespace="runs/artifact-rollback/reports/fast",
                manifest=_artifact_manifest({"index.html": content}),
            )
        row = coord.connection.execute(
            "SELECT * FROM artifact_promotions"
        ).fetchone()
        assert row["status"] == "authorized"
        artifact_id = str(row["artifact_id"])
        destination = coord.artifact_root / str(row["destination_path"])
        assert destination.joinpath("index.html").read_bytes() == content
        assert coord.list_artifacts("artifact-rollback") == []
        coord.connection.execute("DROP TRIGGER block_artifact_insert")
    finally:
        coord.close()

    restarted = DurableCoordinator(db)
    try:
        artifacts = restarted.list_artifacts("artifact-rollback")
        assert [item["artifact_id"] for item in artifacts] == [artifact_id]
        assert restarted.connection.execute(
            "SELECT COUNT(*) AS n FROM artifact_promotions"
        ).fetchone()["n"] == 0
        assert [
            event["type"]
            for event in restarted.list_events("artifact-rollback")
        ][-1] == "artifact.promoted"
    finally:
        restarted.close()


def test_expired_attempt_cannot_promote_artifacts(tmp_path: Path) -> None:
    clock = ManualClock()
    coord = DurableCoordinator(tmp_path / "state.sqlite3", clock=clock)
    try:
        coord.create_job("report_fast", job_id="artifact-expired")
        attempt = coord.claim_next(worker_id="worker-1")
        assert attempt is not None
        workspace = coord.attempt_workspace(attempt.attempt_id, attempt.token)
        content = b"late"
        workspace.joinpath("index.html").write_bytes(content)
        clock.advance(ATTEMPT_EXPIRY_SECONDS + 1)

        with pytest.raises(InvalidTransition, match="expired"):
            coord.promote_attempt_artifacts(
                attempt.attempt_id,
                attempt.token,
                namespace="runs/artifact-expired/reports/fast",
                manifest=_artifact_manifest({"index.html": content}),
            )

        assert coord.list_artifacts("artifact-expired") == []
        assert workspace.exists()
    finally:
        coord.close()


def test_artifact_manifest_rejects_traversal_unmanifested_files_and_symlinks(
    tmp_path: Path,
) -> None:
    coord = DurableCoordinator(tmp_path / "state.sqlite3")
    try:
        coord.create_job("report_fast", job_id="artifact-invalid")
        attempt = coord.claim_next(worker_id="worker-1")
        assert attempt is not None
        workspace = coord.attempt_workspace(attempt.attempt_id, attempt.token)
        content = b"valid"
        workspace.joinpath("index.html").write_bytes(content)

        with pytest.raises(ValueError, match="escape"):
            coord.promote_attempt_artifacts(
                attempt.attempt_id,
                attempt.token,
                namespace="../outside",
                manifest=_artifact_manifest({"index.html": content}),
            )

        workspace.joinpath("extra.txt").write_text("not listed")
        with pytest.raises(InvalidTransition, match="unmanifested"):
            coord.promote_attempt_artifacts(
                attempt.attempt_id,
                attempt.token,
                namespace="runs/artifact-invalid/reports/fast",
                manifest=_artifact_manifest({"index.html": content}),
            )
        workspace.joinpath("extra.txt").unlink()
        workspace.joinpath("link.txt").symlink_to("index.html")
        with pytest.raises(InvalidTransition, match="symlinks"):
            coord.promote_attempt_artifacts(
                attempt.attempt_id,
                attempt.token,
                namespace="runs/artifact-invalid/reports/fast",
                manifest=_artifact_manifest(
                    {"index.html": content, "link.txt": content}
                ),
            )
    finally:
        coord.close()


def test_secret_rotation_invalidation_requeues_active_job(tmp_path: Path) -> None:
    coord = DurableCoordinator(tmp_path / "state.sqlite3")
    try:
        coord.create_job("target_fast", job_id="rotate-secret")
        attempt = coord.claim_next(worker_id="worker-1")
        assert attempt is not None

        result = coord.invalidate_active_attempts("worker_secret_rotated")

        assert result == {"invalidated_attempts": 1, "requeued_jobs": 1}
        assert coord.get_job("rotate-secret")["status"] == "queued"
        with pytest.raises(InvalidTransition):
            coord.heartbeat(attempt.attempt_id, attempt.token)
        assert coord.connection.execute(
            "SELECT COUNT(*) AS n FROM resource_leases"
        ).fetchone()["n"] == 0
    finally:
        coord.close()
