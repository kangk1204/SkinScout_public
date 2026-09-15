#!/usr/bin/env python3
"""Validate immutable Stage 9 fast and physics report packages."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any


REPORT_POINTER_SCHEMA_VERSION = "skinscout.report_pointer.v1"
REPORT_IDENTITY_SCHEMA_VERSION = "skinscout.report_identity.v1"
PACKAGE_SCHEMAS = {
    "fast": {
        "manifest": "skinscout.report_fast.v1",
        "checksums": "skinscout.report_fast.checksums.v1",
    },
    "physics": {
        "manifest": "skinscout.report_physics.v1",
        "checksums": "skinscout.report_physics.checksums.v1",
    },
}
POINTER_RELATIVE_PATHS = {
    "fast": "09_report/fast_report_manifest.json",
    "physics": "09_report/physics_report_manifest.json",
}


class ReportContractError(ValueError):
    """Raised when a report pointer or immutable package fails validation."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ReportContractError(f"{label} is missing or unsafe: {path}")
    if path.stat().st_size == 0:
        raise ReportContractError(f"{label} is empty: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        raise ReportContractError(f"{label} is invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict) or not payload:
        raise ReportContractError(f"{label} must be a non-empty JSON object: {path}")
    return payload


def _reject_symlink_components(path: Path, root: Path, label: str) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ReportContractError(f"{label} escapes the run directory: {path}") from exc
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ReportContractError(f"{label} contains a symlink component: {current}")


def _require_exact_pointer_path(path: Path, run_dir: Path, kind: str) -> Path:
    expected = run_dir / POINTER_RELATIVE_PATHS[kind]
    try:
        resolved = path.resolve(strict=True)
        expected_resolved = expected.resolve(strict=True)
    except OSError as exc:
        raise ReportContractError(
            f"{kind} report pointer is missing: {expected}"
        ) from exc
    _reject_symlink_components(expected, run_dir, f"{kind} report pointer")
    if path.is_symlink() or expected.is_symlink() or resolved != expected_resolved:
        raise ReportContractError(
            f"{kind} report pointer must be {POINTER_RELATIVE_PATHS[kind]}"
        )
    return expected


def _declared_run_path(
    value: object,
    *,
    run_dir: Path,
    field: str,
    expected: str,
) -> Path:
    if not isinstance(value, str) or not value:
        raise ReportContractError(f"Report pointer field {field} must be a path string")
    if "\\" in value:
        raise ReportContractError(f"Report pointer field {field} is not canonical: {value!r}")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ReportContractError(f"Report pointer field {field} is unsafe: {value!r}")
    if value != expected:
        raise ReportContractError(
            f"Report pointer field {field} must be {expected!r}, got {value!r}"
        )
    candidate = run_dir / Path(*relative.parts)
    _reject_symlink_components(candidate, run_dir, f"Report pointer field {field}")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(run_dir.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ReportContractError(
            f"Report pointer field {field} is missing or escapes the run: {value!r}"
        ) from exc
    return candidate


def _validate_checksum_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ReportContractError(f"Invalid report checksum path: {value!r}")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ReportContractError(f"Unsafe report checksum path: {value!r}")
    canonical = relative.as_posix()
    if canonical != value or canonical in {"manifest.json", "checksums.json"}:
        raise ReportContractError(f"Non-canonical report checksum path: {value!r}")
    return canonical


def _validate_package(
    *,
    run_dir: Path,
    kind: str,
    artifact_id: str,
    pointer: dict[str, Any],
) -> dict[str, Any]:
    expected_package = f"reports/{kind}/{artifact_id}"
    package_root = _declared_run_path(
        pointer.get("package_path"),
        run_dir=run_dir,
        field=f"{kind}.package_path",
        expected=expected_package,
    )
    if package_root.is_symlink() or not package_root.is_dir():
        raise ReportContractError(f"Immutable {kind} report package is unsafe: {package_root}")

    expected_manifest = f"{expected_package}/manifest.json"
    manifest_path = _declared_run_path(
        pointer.get("manifest_path"),
        run_dir=run_dir,
        field=f"{kind}.manifest_path",
        expected=expected_manifest,
    )
    checksums_path = package_root / "checksums.json"
    identity_path = package_root / "identity.json"
    checksums = _read_json_object(checksums_path, f"{kind} report checksums")
    manifest = _read_json_object(manifest_path, f"{kind} report manifest")
    identity = _read_json_object(identity_path, f"{kind} report identity")

    schemas = PACKAGE_SCHEMAS[kind]
    if checksums.get("schema_version") != schemas["checksums"]:
        raise ReportContractError(f"{kind} report checksum schema mismatch")
    if manifest.get("schema_version") != schemas["manifest"]:
        raise ReportContractError(f"{kind} report manifest schema mismatch")
    if identity.get("schema_version") != REPORT_IDENTITY_SCHEMA_VERSION:
        raise ReportContractError(f"{kind} report identity schema mismatch")
    if identity.get("kind") != kind:
        raise ReportContractError(f"{kind} report identity kind mismatch")
    if _canonical_hash(checksums) != artifact_id:
        raise ReportContractError(f"{kind} report artifact ID does not match checksums")
    if manifest.get("artifact_id") != artifact_id:
        raise ReportContractError(f"{kind} report manifest artifact ID mismatch")
    if manifest.get("sealed") is not True or pointer.get("sealed") is not True:
        raise ReportContractError(f"{kind} report package is not sealed")
    if manifest.get("checksums_path") != "checksums.json":
        raise ReportContractError(f"{kind} report checksums path mismatch")
    if manifest.get("identity_path") != "identity.json":
        raise ReportContractError(f"{kind} report identity path mismatch")
    if manifest.get("identity_sha256") != _sha256(identity_path):
        raise ReportContractError(f"{kind} report identity hash mismatch")

    manifest_core = identity.get("manifest_core")
    if not isinstance(manifest_core, dict) or not manifest_core:
        raise ReportContractError(f"{kind} report identity has no manifest core")
    for key, expected in manifest_core.items():
        if manifest.get(key) != expected:
            raise ReportContractError(
                f"{kind} report manifest field is not identity-bound: {key}"
            )

    entries = checksums.get("files")
    if not isinstance(entries, list) or not entries:
        raise ReportContractError(f"{kind} report checksums contain no files")
    expected_files: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ReportContractError(f"{kind} report checksum entry must be an object")
        relative = _validate_checksum_path(entry.get("path"))
        if relative in expected_files:
            raise ReportContractError(f"Duplicate {kind} report checksum path: {relative}")
        digest = entry.get("sha256")
        size = entry.get("bytes")
        if not _is_sha256(digest):
            raise ReportContractError(f"Invalid {kind} report SHA256 for {relative}")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ReportContractError(f"Invalid {kind} report size for {relative}")
        candidate = package_root / Path(*PurePosixPath(relative).parts)
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(package_root.resolve(strict=True))
        except (OSError, ValueError) as exc:
            raise ReportContractError(
                f"{kind} report payload is missing or escapes package: {relative}"
            ) from exc
        if candidate.is_symlink() or not candidate.is_file():
            raise ReportContractError(f"{kind} report payload is unsafe: {relative}")
        if candidate.stat().st_size != size:
            raise ReportContractError(f"{kind} report payload size mismatch: {relative}")
        if _sha256(candidate) != digest:
            raise ReportContractError(f"{kind} report payload hash mismatch: {relative}")
        expected_files[relative] = entry

    actual_files: set[str] = set()
    for candidate in package_root.rglob("*"):
        if candidate.is_symlink():
            raise ReportContractError(
                f"{kind} report package cannot contain symlinks: {candidate}"
            )
        if candidate.is_file():
            relative = candidate.relative_to(package_root).as_posix()
            if relative not in {"manifest.json", "checksums.json"}:
                actual_files.add(relative)
    if actual_files != set(expected_files):
        raise ReportContractError(f"{kind} report package contains undeclared payload files")
    if "identity.json" not in expected_files:
        raise ReportContractError(f"{kind} report identity is not checksum-bound")
    if kind == "fast" and "index.html" not in expected_files:
        raise ReportContractError("Fast report package has no checksum-bound index.html")

    result: dict[str, Any] = {
        "kind": kind,
        "artifact_id": artifact_id,
        "sealed": True,
        "run_id": manifest.get("run_id"),
        "package_path": expected_package,
        "manifest_path": expected_manifest,
        "checksums_path": f"{expected_package}/checksums.json",
        "identity_path": f"{expected_package}/identity.json",
        "pointer_path": POINTER_RELATIVE_PATHS[kind],
        "pointer_sha256": _sha256(run_dir / POINTER_RELATIVE_PATHS[kind]),
        "manifest_sha256": _sha256(manifest_path),
        "checksums_sha256": _sha256(checksums_path),
        "identity_sha256": _sha256(identity_path),
        "payload_count": len(expected_files),
        "payload_bytes": sum(int(entry["bytes"]) for entry in expected_files.values()),
        "payloads": [
            {
                "relative_path": f"{expected_package}/{relative}",
                "bytes": entry["bytes"],
                "sha256": entry["sha256"],
            }
            for relative, entry in sorted(expected_files.items())
        ],
    }
    if kind == "fast":
        expected_index = f"{expected_package}/index.html"
        _declared_run_path(
            pointer.get("index_path"),
            run_dir=run_dir,
            field="fast.index_path",
            expected=expected_index,
        )
        parent_linkage = manifest.get("parent_linkage")
        if not isinstance(parent_linkage, dict):
            raise ReportContractError("Fast report manifest parent_linkage is missing")
        if parent_linkage.get("fast_parent_hash") != artifact_id:
            raise ReportContractError("Fast report self-parent hash mismatch")
        result["index_path"] = expected_index
    else:
        result["parent_fast_hash"] = manifest.get("parent_fast_hash")
    return result


def _validate_ready_pointer(
    *,
    path: Path,
    run_dir: Path,
    kind: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    pointer_path = _require_exact_pointer_path(path, run_dir, kind)
    pointer = _read_json_object(pointer_path, f"{kind} report pointer")
    if pointer.get("schema_version") != REPORT_POINTER_SCHEMA_VERSION:
        raise ReportContractError(f"{kind} report pointer schema mismatch")
    if pointer.get("kind") != kind:
        raise ReportContractError(f"{kind} report pointer kind mismatch")
    if pointer.get("status") != "ready":
        raise ReportContractError(
            f"{kind} report pointer is not ready: {pointer.get('status')!r}"
        )
    artifact_id = pointer.get("artifact_id")
    if not _is_sha256(artifact_id):
        raise ReportContractError(f"{kind} report pointer artifact ID is invalid")
    package = _validate_package(
        run_dir=run_dir,
        kind=kind,
        artifact_id=artifact_id,
        pointer=pointer,
    )
    return pointer, package


def validate_report_packages(
    *,
    run_dir: Path,
    fast_pointer: Path,
    physics_pointer: Path,
    require_physics: bool = False,
) -> dict[str, Any]:
    """Validate sealed immutable report packages and their parent linkage."""

    try:
        if run_dir.is_symlink():
            raise ReportContractError(f"Report run directory is unsafe: {run_dir}")
        run_root = run_dir.resolve(strict=True)
    except OSError as exc:
        raise ReportContractError(f"Report run directory is missing: {run_dir}") from exc
    if run_root.is_symlink() or not run_root.is_dir():
        raise ReportContractError(f"Report run directory is unsafe: {run_dir}")

    fast_doc, fast_package = _validate_ready_pointer(
        path=fast_pointer,
        run_dir=run_root,
        kind="fast",
    )
    physics_path = _require_exact_pointer_path(physics_pointer, run_root, "physics")
    physics_doc = _read_json_object(physics_path, "physics report pointer")
    if physics_doc.get("schema_version") != REPORT_POINTER_SCHEMA_VERSION:
        raise ReportContractError("physics report pointer schema mismatch")
    if physics_doc.get("kind") != "physics":
        raise ReportContractError("physics report pointer kind mismatch")

    physics_package: dict[str, Any] | None = None
    physics_status = physics_doc.get("status")
    if physics_status == "not_requested":
        if require_physics:
            raise ReportContractError("Physics report is required but was not requested")
        if physics_doc.get("artifact_id") is not None or physics_doc.get("sealed") is not False:
            raise ReportContractError("Not-requested physics pointer contains artifact state")
        if not isinstance(physics_doc.get("reason"), str) or not physics_doc["reason"].strip():
            raise ReportContractError("Not-requested physics pointer has no reason")
        forbidden_fields = {
            "package_path",
            "manifest_path",
            "parent_fast_hash",
            "parent_manifest_path",
        }
        populated = sorted(
            field for field in forbidden_fields if physics_doc.get(field) is not None
        )
        if populated:
            raise ReportContractError(
                "Not-requested physics pointer contains package fields: "
                + ", ".join(populated)
            )
    elif physics_status == "ready":
        physics_artifact_id = physics_doc.get("artifact_id")
        if not _is_sha256(physics_artifact_id):
            raise ReportContractError("physics report pointer artifact ID is invalid")
        physics_package = _validate_package(
            run_dir=run_root,
            kind="physics",
            artifact_id=physics_artifact_id,
            pointer=physics_doc,
        )
        fast_artifact_id = fast_package["artifact_id"]
        expected_fast_manifest = fast_package["manifest_path"]
        if physics_doc.get("parent_fast_hash") != fast_artifact_id:
            raise ReportContractError("Physics pointer parent fast hash mismatch")
        if physics_doc.get("parent_manifest_path") != expected_fast_manifest:
            raise ReportContractError("Physics pointer parent manifest path mismatch")
        if physics_package.get("parent_fast_hash") != fast_artifact_id:
            raise ReportContractError("Physics package parent fast hash mismatch")
    else:
        raise ReportContractError(
            f"physics report pointer has non-publishable status: {physics_status!r}"
        )

    manifest_run_ids = {fast_package.get("run_id")}
    if physics_package is not None:
        manifest_run_ids.add(physics_package.get("run_id"))
    if manifest_run_ids != {run_root.name}:
        raise ReportContractError(
            "Report package run_id does not match the active run directory: "
            f"expected {run_root.name!r}, got {sorted(str(value) for value in manifest_run_ids)!r}"
        )

    return {
        "schema_version": "skinscout.report_publication_contract.v1",
        "status": "passed",
        "run_id": run_root.name,
        "fast": fast_package,
        "physics": {
            "status": physics_status,
            "pointer_path": POINTER_RELATIVE_PATHS["physics"],
            "pointer_sha256": _sha256(physics_path),
            "package": physics_package,
        },
        "physics_required": require_physics,
        "fast_pointer_status": fast_doc["status"],
    }
