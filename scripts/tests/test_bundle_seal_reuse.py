"""F42: a bundle request verifies the run seal once, not once per member.

The audit found that each member read re-validated every entry in
``verified_artifacts``, so the hash work grew with (members x artifacts). These
tests instrument ``_sha256_file`` (the seal's hash-read primitive) and assert
one verification pass per request, while a member or sealed artifact that
changes mid-request still fails closed.

This counts hash reads, not wall-clock time or bytes: no real I/O delay was
measured, so the test proves the call-count shape, not a latency number.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from workbench import server


def _write_sealed_run(
    runs_dir: Path,
    run_id: str,
    viewer_count: int,
    *,
    extra_sealed: str | None = None,
) -> Path:
    run = runs_dir / run_id
    viewer = run / "viewer"
    viewer.mkdir(parents=True)
    files: list[Path] = []
    summary = run / "run_summary.json"
    summary.write_text(
        json.dumps({"schema_version": "skinscout.run_summary.v1", "run_id": run_id}),
        encoding="utf-8",
    )
    files.append(summary)
    for index in range(viewer_count):
        page = viewer / f"page{index:02d}.html"
        page.write_text(f"<html>{index}</html>", encoding="utf-8")
        files.append(page)
    if extra_sealed is not None:
        extra = run / extra_sealed
        extra.parent.mkdir(parents=True, exist_ok=True)
        extra.write_text("sealed but not bundled\n", encoding="utf-8")
        files.append(extra)
    verification = {
        "schema_version": "skinscout.run_verification.v1",
        "status": "ok",
        "verifier_status": "ok",
        "verified_artifacts": [
            {
                "name": path.name,
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in files
        ],
    }
    (run / "run_verification.json").write_text(
        json.dumps(verification), encoding="utf-8"
    )
    return run


@pytest.mark.parametrize("viewer_count", [6, 12, 24])
def test_each_sealed_artifact_is_hashed_once_per_bundle_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    viewer_count: int,
) -> None:
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path)
    run_id = f"sealed_{viewer_count}"
    _write_sealed_run(tmp_path, run_id, viewer_count)
    # The attestation contract itself is covered elsewhere; this test isolates
    # the snapshot reuse inside the bundle path.
    monkeypatch.setattr(
        server, "_verification_contract_error", lambda *_args, **_kwargs: None
    )

    real_sha = server._sha256_file
    digests: list[str] = []

    def counting_sha(path: Path) -> str:
        digests.append(Path(path).name)
        return real_sha(path)

    monkeypatch.setattr(server, "_sha256_file", counting_sha)

    bundle = server._run_bundle_zip(run_id)

    assert bundle is not None
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        names = set(archive.namelist())
    assert f"{run_id}/run_summary.json" in names
    assert len([name for name in names if name.startswith(f"{run_id}/viewer/")]) == viewer_count
    # One verification pass over `verified_artifacts` (summary + pages). The
    # previous per-member re-validation made this (members x artifacts).
    assert len(digests) == viewer_count + 1


def test_a_member_changed_mid_request_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path)
    run_id = "tampered_member"
    run = _write_sealed_run(tmp_path, run_id, 5)
    monkeypatch.setattr(
        server, "_verification_contract_error", lambda *_args, **_kwargs: None
    )
    victim = run / "viewer" / "page04.html"

    real_read = server._read_integrity_bound_run_bytes
    tampered = {"done": False}

    def tampering_read(path: Path, *, snapshot: server._RunSealSnapshot | None = None):
        if not tampered["done"]:
            tampered["done"] = True
            victim.write_text("<html>forged and longer than the seal</html>", encoding="utf-8")
        return real_read(path, snapshot=snapshot)

    monkeypatch.setattr(server, "_read_integrity_bound_run_bytes", tampering_read)

    assert server._run_bundle_zip(run_id) is None


def test_a_sealed_non_member_changed_mid_request_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sealed artifact the bundle does not carry still gates the request."""
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path)
    run_id = "tampered_non_member"
    run = _write_sealed_run(tmp_path, run_id, 4, extra_sealed="03_targets/notes.json")
    monkeypatch.setattr(
        server, "_verification_contract_error", lambda *_args, **_kwargs: None
    )
    victim = run / "03_targets" / "notes.json"

    real_read = server._read_integrity_bound_run_bytes
    tampered = {"done": False}

    def tampering_read(path: Path, *, snapshot: server._RunSealSnapshot | None = None):
        if not tampered["done"]:
            tampered["done"] = True
            victim.write_text("forged after the seal snapshot\n", encoding="utf-8")
        return real_read(path, snapshot=snapshot)

    monkeypatch.setattr(server, "_read_integrity_bound_run_bytes", tampering_read)

    assert server._run_bundle_zip(run_id) is None


def test_a_stable_sealed_bundle_is_returned_and_unchanged_detects_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path)
    run_id = "stable_snapshot"
    _write_sealed_run(tmp_path, run_id, 3)
    monkeypatch.setattr(
        server, "_verification_contract_error", lambda *_args, **_kwargs: None
    )

    snapshot = server._RunSealSnapshot(run_id)

    assert snapshot.claims_success is True
    assert snapshot.contract_error is None
    assert snapshot.artifact_error is None
    assert snapshot.artifacts is not None
    assert len(snapshot.artifacts) == 4
    assert snapshot.unchanged() is True
    assert server._run_bundle_zip(run_id) is not None

    (tmp_path / run_id / "viewer" / "page00.html").write_text(
        "changed after the snapshot\n", encoding="utf-8"
    )
    assert snapshot.unchanged() is False
