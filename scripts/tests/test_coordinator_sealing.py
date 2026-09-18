"""Audit F39: coordinator state permissions and retained-fd sealing resistance."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
from pathlib import Path

import pytest

from workbench.coordinator import CoordinatorError, DurableCoordinator


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


def test_state_directory_and_database_are_private_under_permissive_umask(
    tmp_path: Path,
) -> None:
    state = tmp_path / "state"
    db = state / "coordinator.sqlite3"
    previous = os.umask(0o000)
    try:
        coord = DurableCoordinator(db)
    finally:
        os.umask(previous)
    try:
        assert stat.S_IMODE(state.stat().st_mode) == 0o700
        assert stat.S_IMODE(db.stat().st_mode) == 0o600
        for suffix in ("-wal", "-shm"):
            auxiliary = Path(str(db) + suffix)
            if auxiliary.exists():
                assert stat.S_IMODE(auxiliary.stat().st_mode) == 0o600
        for directory in (
            coord.artifact_root,
            coord.attempt_root,
            coord.committed_root,
            coord.staging_root,
            coord.quarantine_root,
        ):
            assert stat.S_IMODE(directory.stat().st_mode) == 0o700

        coord.create_job("report_fast", job_id="umask-job")
        attempt = coord.claim_next(worker_id="worker-1")
        assert attempt is not None
        workspace = coord.attempt_workspace(attempt.attempt_id, attempt.token)
        assert stat.S_IMODE(workspace.stat().st_mode) == 0o700
    finally:
        coord.close()


def test_reopened_database_permissions_are_restored_to_private(tmp_path: Path) -> None:
    db = tmp_path / "state" / "coordinator.sqlite3"
    DurableCoordinator(db).close()
    db.chmod(0o644)

    coord = DurableCoordinator(db)
    try:
        assert stat.S_IMODE(db.stat().st_mode) == 0o600
    finally:
        coord.close()


def test_state_ancestor_writable_by_others_is_rejected() -> None:
    base = Path(tempfile.mkdtemp(dir="/tmp", prefix="skinscout-ancestor-"))
    try:
        shared = base / "shared"
        shared.mkdir()
        base.chmod(0o755)
        shared.chmod(0o777)

        with pytest.raises(CoordinatorError, match="writable by others"):
            DurableCoordinator(shared / "state" / "coordinator.sqlite3")
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_writable_ancestor_behind_a_private_parent_is_allowed(tmp_path: Path) -> None:
    private = tmp_path / "private"
    private.mkdir()
    private.chmod(0o700)
    writable = private / "group-writable"
    writable.mkdir()
    writable.chmod(0o775)

    coord = DurableCoordinator(writable / "state" / "coordinator.sqlite3")
    try:
        assert stat.S_IMODE((writable / "state").stat().st_mode) == 0o700
    finally:
        coord.close()


def test_retained_writable_fd_cannot_mutate_sealed_artifact(tmp_path: Path) -> None:
    coord = DurableCoordinator(tmp_path / "state.sqlite3")
    try:
        coord.create_job("report_fast", job_id="seal-fd")
        attempt = coord.claim_next(worker_id="worker-1")
        assert attempt is not None
        workspace = coord.attempt_workspace(attempt.attempt_id, attempt.token)
        content = b"sealed-before-promotion\n"
        source = workspace / "result.bin"
        source.write_bytes(content)
        descriptor = os.open(source, os.O_WRONLY)
        try:
            artifact = coord.promote_attempt_artifacts(
                attempt.attempt_id,
                attempt.token,
                namespace="runs/seal-fd/reports/fast",
                manifest=_artifact_manifest({"result.bin": content}),
            )
            os.pwrite(descriptor, b"tampered", 0)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

        _, entry, sealed = coord.resolve_artifact_file(
            artifact["artifact_id"],
            "result.bin",
        )
        sealed_bytes = sealed.read_bytes()
        assert sealed_bytes == content
        assert hashlib.sha256(sealed_bytes).hexdigest() == entry["sha256"]
        assert stat.S_IMODE(sealed.stat().st_mode) & 0o222 == 0
    finally:
        coord.close()
