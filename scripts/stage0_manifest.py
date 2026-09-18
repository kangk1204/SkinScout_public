#!/usr/bin/env python3
"""Stage 0 receptor PDBQT preparation manifest.

One contract shared by the producer (``stage0_meeko_prep.py``), readiness
(``data_readiness.py``) and the final verifier (``stage0_verify.py``). The
producer records the expected / succeeded / failed receptor set together with
the policy it enforced and the input hashes it used; readiness and the verifier
judge that record instead of a touch file, so a completion marker can no longer
stand in for a set the producer did not actually produce.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "skinscout.stage0-receptor-pdbqt.v1"
MANIFEST_FILENAME = "receptor_prep_manifest.json"
TARGET_STATUSES = {"ok", "failed"}


def manifest_path(pdbqt_dir: Path) -> Path:
    return Path(pdbqt_dir) / MANIFEST_FILENAME


def sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_fraction(value: object) -> bool:
    if isinstance(value, bool):
        return False
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return math.isfinite(parsed) and 0.0 <= parsed <= 1.0


def build_manifest(
    *,
    scope: str,
    policy: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if scope not in {"full", "restricted"}:
        raise ValueError(f"unknown Stage 0 receptor manifest scope: {scope!r}")
    failed = [record for record in records if record.get("status") == "failed"]
    success = [record for record in records if record.get("status") == "ok"]
    expected = len(records)
    min_fraction = float(policy["min_success_fraction"])
    min_count = int(policy["min_success_count"])
    policy_met = (
        len(success) >= min_count
        and expected > 0
        and len(success) / expected >= min_fraction
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": scope,
        "status": "ok" if policy_met else "failed",
        "policy": {
            "min_success_fraction": min_fraction,
            "min_success_count": min_count,
        },
        "counts": {
            "expected": expected,
            "success": len(success),
            "failed": len(failed),
        },
        "targets": [dict(record) for record in records],
    }


def write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def load_manifest(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    path = Path(path)
    if not path.is_file():
        return None, f"receptor prep manifest is missing: {path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"receptor prep manifest is unreadable: {path}: {exc}"
    if not isinstance(payload, dict):
        return None, f"receptor prep manifest must be a JSON object: {path}"
    return payload, None


def _validate_policy(payload: Mapping[str, Any], errors: list[str]) -> dict[str, Any]:
    policy = payload.get("policy")
    if not isinstance(policy, Mapping):
        errors.append("policy")
        return {}
    if not _is_finite_fraction(policy.get("min_success_fraction")):
        errors.append("policy.min_success_fraction")
    if not _is_int(policy.get("min_success_count")) or int(
        policy.get("min_success_count", -1)
    ) < 0:
        errors.append("policy.min_success_count")
    return dict(policy)


def _validate_counts(payload: Mapping[str, Any], errors: list[str]) -> dict[str, int]:
    counts = payload.get("counts")
    if not isinstance(counts, Mapping):
        errors.append("counts")
        return {}
    parsed: dict[str, int] = {}
    for key in ("expected", "success", "failed"):
        value = counts.get(key)
        if not _is_int(value) or int(value) < 0:
            errors.append(f"counts.{key}")
        else:
            parsed[key] = int(value)
    if len(parsed) == 3 and parsed["expected"] != parsed["success"] + parsed["failed"]:
        errors.append("counts.expected != success + failed")
    return parsed


def _validate_target(
    record: object,
    index: int,
    errors: list[str],
) -> tuple[str | None, str, str | None]:
    if not isinstance(record, Mapping):
        errors.append(f"targets[{index}]")
        return None, "", None
    uniprot = record.get("uniprot")
    if not isinstance(uniprot, str) or not uniprot.strip():
        errors.append(f"targets[{index}].uniprot")
        return None, "", None
    status = record.get("status")
    if status not in TARGET_STATUSES:
        errors.append(f"targets[{index}].status")
        return str(uniprot), "", None
    reason = record.get("reason")
    if status == "failed" and (not isinstance(reason, str) or not reason.strip()):
        errors.append(f"targets[{index}].reason")
    for field in ("clean_pdb_sha256", "pocket_json_sha256"):
        value = record.get(field)
        if not isinstance(value, str) or len(value) != 64:
            errors.append(f"targets[{index}].{field}")
    return str(uniprot), str(status), reason if isinstance(reason, str) else None


def validate_manifest(
    payload: object,
    *,
    pdbqt_dir: Path,
    expected_policy: Mapping[str, Any] | None = None,
    require_full_scope: bool = True,
    check_success_files: bool = True,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate a producer manifest against the policy it claims to enforce.

    Returns ``(index, errors)`` where ``index`` holds the success/failed sets
    plus the failed reasons once validation has enough structure to build it.
    Every error is fatal for the caller; this function never repairs a
    manifest.
    """
    errors: list[str] = []
    if not isinstance(payload, Mapping):
        return None, ["manifest root must be an object"]
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version")

    scope = payload.get("scope")
    if scope not in {"full", "restricted"}:
        errors.append("scope")
    if require_full_scope and scope == "restricted":
        errors.append("scope is 'restricted'; a full receptor set is required")

    policy = _validate_policy(payload, errors)
    counts = _validate_counts(payload, errors)

    if expected_policy is not None:
        expected_fraction = expected_policy.get("min_success_fraction")
        expected_count = expected_policy.get("min_success_count")
        if (
            _is_finite_fraction(expected_fraction)
            and _is_finite_fraction(policy.get("min_success_fraction"))
            and float(policy["min_success_fraction"]) < float(expected_fraction)
        ):
            errors.append(
                "policy.min_success_fraction is weaker than the configured gate "
                f"({policy['min_success_fraction']} < {expected_fraction})"
            )
        if (
            _is_int(expected_count)
            and _is_int(policy.get("min_success_count"))
            and int(policy["min_success_count"]) < int(expected_count)
        ):
            errors.append(
                "policy.min_success_count is weaker than the configured gate "
                f"({policy['min_success_count']} < {expected_count})"
            )

    targets = payload.get("targets")
    if not isinstance(targets, list):
        errors.append("targets")
        return None, errors

    success: set[str] = set()
    failed: dict[str, str] = {}
    seen: set[str] = set()
    for index, record in enumerate(targets):
        uniprot, status, reason = _validate_target(record, index, errors)
        if uniprot is None:
            continue
        if uniprot in seen:
            errors.append(f"targets[{index}].uniprot duplicated")
            continue
        seen.add(uniprot)
        if status == "ok":
            success.add(uniprot)
        elif status == "failed":
            failed[uniprot] = reason or ""

    if "expected" in counts and len(targets) != counts["expected"]:
        errors.append("targets length does not match counts.expected")
    if "success" in counts and len(success) != counts["success"]:
        errors.append("ok target count does not match counts.success")
    if "failed" in counts and len(failed) != counts["failed"]:
        errors.append("failed target count does not match counts.failed")

    fraction_ok = bool(policy) and _is_finite_fraction(
        policy.get("min_success_fraction")
    )
    count_ok = bool(policy) and _is_int(policy.get("min_success_count"))
    if (
        fraction_ok
        and count_ok
        and counts
        and len(targets) == counts.get("expected")
    ):
        expected = int(counts["expected"])
        computed_met = (
            len(success) >= int(policy["min_success_count"])
            and expected > 0
            and len(success) / expected >= float(policy["min_success_fraction"])
        )
        status = payload.get("status")
        if status not in {"ok", "failed"}:
            errors.append("status")
        elif computed_met != (status == "ok"):
            errors.append("status does not match the recorded policy and counts")
        if not computed_met:
            errors.append("recorded policy gate is not met")

    if check_success_files:
        missing_files = sorted(
            target
            for target in success
            if not (Path(pdbqt_dir) / f"{target}.pdbqt").is_file()
            or (Path(pdbqt_dir) / f"{target}.pdbqt").stat().st_size == 0
        )
        if missing_files:
            errors.append(
                "success targets lack a non-empty PDBQT file: "
                f"{', '.join(missing_files[:5])}"
                + ("..." if len(missing_files) > 5 else "")
            )

    index = {
        "success": success,
        "failed": failed,
        "policy": policy,
        "counts": counts,
    }
    return index, errors


def failed_preview(failed: Mapping[str, str], limit: int = 3) -> str:
    if not failed:
        return "none"
    items = [
        f"{target}: {reason}" if reason else target
        for target, reason in sorted(failed.items())
    ]
    preview = "; ".join(items[:limit])
    if len(items) > limit:
        preview += f"; ... (+{len(items) - limit} more)"
    return preview
