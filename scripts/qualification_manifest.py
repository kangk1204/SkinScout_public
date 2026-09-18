#!/usr/bin/env python3
"""Build and verify SkinScout Release 1 qualification evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRUST_POLICY = ROOT / "compose/qualification-trust-policy.json"
SCHEMA_VERSION = "skinscout.release-qualification.v1"
VALIDATION_SCHEMA_VERSION = "skinscout.release-qualification-validation.v1"
ENVIRONMENT_SCHEMA_VERSION = "skinscout.qualification-environment.v1"
ARTIFACT_LOCK_SCHEMA_VERSION = "skinscout.qualification-artifact-lock.v1"
LEDGER_SCHEMA_VERSION = "skinscout.qualification-sla-ledger.v1"
TRUST_POLICY_SCHEMA_VERSION = "skinscout.qualification-trust-policy.v1"

PANELS = ("public", "sealed")
PANEL_SIZE = 20
REPLICATES = (1, 2, 3)
OBSERVATIONS_PER_PANEL = PANEL_SIZE * len(REPLICATES)
P95_RANK = math.ceil(0.95 * OBSERVATIONS_PER_PANEL)
SLA_SECONDS = 60 * 60
MAX_SUCCESS_SECONDS = 75 * 60
MIN_MEMORY_BYTES = 128 * 1024**3
MIN_GPU_MEMORY_BYTES = 24_000_000_000
MIN_STORAGE_BYTES_PER_SECOND = 7_000_000_000
REQUIRED_IMAGE_NAMES = {"skinscout-app", "skinscout-target", "skinscout-physics"}
REQUIRED_COMMAND_NAMES = {
    "host_nvidia_smi",
    "container_nvidia_smi",
    "cuda_tensor",
    "storage_benchmark",
    "qualification_runner",
}
EXPECTED_TOP_LEVEL_KEYS = {
    "schema_version",
    "qualification_evidence_status",
    "reference_host_status",
    "release_qualified",
    "release_qualification_reason",
    "thresholds",
    "inputs",
    "environment",
    "artifact_lock",
    "ledger",
    "panel_summaries",
    "gates",
    "failed_gates",
    "binding_sha256",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CPU_RE = re.compile(r"\bAMD Ryzen 9 7950X\b", re.IGNORECASE)


class QualificationError(ValueError):
    """Raised when qualification evidence is malformed or stale."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise QualificationError(f"{label} is missing, empty, or unsafe: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QualificationError(f"{label} is invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise QualificationError(f"{label} must be a JSON object")
    return payload


def _source_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file() or resolved.stat().st_size <= 0:
        raise QualificationError(f"qualification input is missing, empty, or unsafe: {path}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _active_file_record(value: Any, label: str) -> dict[str, Any]:
    record = _object(value, label)
    _require_exact_keys(record, {"path", "bytes", "sha256"}, label)
    path = Path(_text(record.get("path"), f"{label}.path"))
    active = _source_record(path)
    if record != active:
        raise QualificationError(f"{label} does not match its active immutable file")
    return active


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if extra:
            details.append(f"unexpected {', '.join(extra)}")
        raise QualificationError(f"{label} fields are invalid: {'; '.join(details)}")


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise QualificationError(f"{label} must be an object")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise QualificationError(f"{label} must be a list")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QualificationError(f"{label} must be a non-empty string")
    return value.strip()


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise QualificationError(f"{label} must be an integer >= {minimum}")
    return value


def _finite_number(value: Any, label: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QualificationError(f"{label} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < minimum:
        raise QualificationError(f"{label} must be finite and >= {minimum}")
    return parsed


def _sha256_text(value: Any, label: str) -> str:
    text = _text(value, label).lower()
    if not SHA256_RE.fullmatch(text) or len(set(text)) <= 1:
        raise QualificationError(f"{label} must be a non-placeholder SHA-256 digest")
    return text


def _timestamp(value: Any, label: str) -> tuple[str, datetime]:
    text = _text(value, label)
    candidate = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise QualificationError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise QualificationError(f"{label} must be UTC")
    canonical = parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return canonical, parsed.astimezone(timezone.utc)


def _normalize_environment(raw: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema_version",
        "captured_at_utc",
        "os",
        "kernel",
        "cpu",
        "memory_bytes",
        "gpu",
        "nvidia",
        "docker",
        "nvidia_container_toolkit",
        "storage",
        "cpu_governor",
        "commands",
    }
    _require_exact_keys(raw, expected, "environment")
    if raw.get("schema_version") != ENVIRONMENT_SCHEMA_VERSION:
        raise QualificationError(f"environment schema_version must be {ENVIRONMENT_SCHEMA_VERSION}")

    os_info = _object(raw.get("os"), "environment.os")
    _require_exact_keys(os_info, {"id", "version_id", "pretty_name"}, "environment.os")
    cpu = _object(raw.get("cpu"), "environment.cpu")
    _require_exact_keys(cpu, {"model", "logical_cores"}, "environment.cpu")
    gpu = _object(raw.get("gpu"), "environment.gpu")
    _require_exact_keys(gpu, {"name", "memory_bytes", "count"}, "environment.gpu")
    nvidia = _object(raw.get("nvidia"), "environment.nvidia")
    _require_exact_keys(nvidia, {"package", "driver_version"}, "environment.nvidia")
    docker = _object(raw.get("docker"), "environment.docker")
    _require_exact_keys(docker, {"version", "compose_version"}, "environment.docker")
    toolkit = _object(
        raw.get("nvidia_container_toolkit"),
        "environment.nvidia_container_toolkit",
    )
    _require_exact_keys(toolkit, {"version"}, "environment.nvidia_container_toolkit")
    storage = _object(raw.get("storage"), "environment.storage")
    _require_exact_keys(
        storage,
        {"device", "technology", "sequential_read_bytes_per_second"},
        "environment.storage",
    )

    commands: list[dict[str, Any]] = []
    command_names: set[str] = set()
    for index, entry_raw in enumerate(_list(raw.get("commands"), "environment.commands")):
        entry = _object(entry_raw, f"environment.commands[{index}]")
        _require_exact_keys(
            entry,
            {"name", "argv", "exit_code", "stdout_sha256"},
            f"environment.commands[{index}]",
        )
        name = _text(entry.get("name"), f"environment.commands[{index}].name")
        if name in command_names:
            raise QualificationError(f"duplicate environment command: {name}")
        command_names.add(name)
        argv = [
            _text(argument, f"environment.commands[{index}].argv")
            for argument in _list(entry.get("argv"), f"environment.commands[{index}].argv")
        ]
        if not argv:
            raise QualificationError(f"environment.commands[{index}].argv cannot be empty")
        exit_code = _integer(
            entry.get("exit_code"),
            f"environment.commands[{index}].exit_code",
        )
        if exit_code != 0:
            raise QualificationError(f"qualification command failed: {name}")
        commands.append(
            {
                "name": name,
                "argv": argv,
                "exit_code": exit_code,
                "stdout_sha256": _sha256_text(
                    entry.get("stdout_sha256"),
                    f"environment.commands[{index}].stdout_sha256",
                ),
            }
        )
    missing_commands = sorted(REQUIRED_COMMAND_NAMES - command_names)
    if missing_commands:
        raise QualificationError(
            "environment commands missing required checks: " + ", ".join(missing_commands)
        )

    captured_at, _ = _timestamp(raw.get("captured_at_utc"), "environment.captured_at_utc")
    return {
        "schema_version": ENVIRONMENT_SCHEMA_VERSION,
        "captured_at_utc": captured_at,
        "os": {
            "id": _text(os_info.get("id"), "environment.os.id").lower(),
            "version_id": _text(os_info.get("version_id"), "environment.os.version_id"),
            "pretty_name": _text(os_info.get("pretty_name"), "environment.os.pretty_name"),
        },
        "kernel": _text(raw.get("kernel"), "environment.kernel"),
        "cpu": {
            "model": _text(cpu.get("model"), "environment.cpu.model"),
            "logical_cores": _integer(
                cpu.get("logical_cores"), "environment.cpu.logical_cores", minimum=1
            ),
        },
        "memory_bytes": _integer(
            raw.get("memory_bytes"), "environment.memory_bytes", minimum=1
        ),
        "gpu": {
            "name": _text(gpu.get("name"), "environment.gpu.name"),
            "memory_bytes": _integer(
                gpu.get("memory_bytes"), "environment.gpu.memory_bytes", minimum=1
            ),
            "count": _integer(gpu.get("count"), "environment.gpu.count", minimum=1),
        },
        "nvidia": {
            "package": _text(nvidia.get("package"), "environment.nvidia.package"),
            "driver_version": _text(
                nvidia.get("driver_version"), "environment.nvidia.driver_version"
            ),
        },
        "docker": {
            "version": _text(docker.get("version"), "environment.docker.version"),
            "compose_version": _text(
                docker.get("compose_version"), "environment.docker.compose_version"
            ),
        },
        "nvidia_container_toolkit": {
            "version": _text(
                toolkit.get("version"), "environment.nvidia_container_toolkit.version"
            )
        },
        "storage": {
            "device": _text(storage.get("device"), "environment.storage.device"),
            "technology": _text(
                storage.get("technology"), "environment.storage.technology"
            ).lower(),
            "sequential_read_bytes_per_second": _integer(
                storage.get("sequential_read_bytes_per_second"),
                "environment.storage.sequential_read_bytes_per_second",
                minimum=1,
            ),
        },
        "cpu_governor": _text(raw.get("cpu_governor"), "environment.cpu_governor").lower(),
        "commands": sorted(commands, key=lambda item: item["name"]),
    }


def _normalize_artifact_lock(raw: Mapping[str, Any]) -> dict[str, Any]:
    _require_exact_keys(
        raw,
        {"schema_version", "created_at_utc", "images", "data", "models"},
        "artifact_lock",
    )
    if raw.get("schema_version") != ARTIFACT_LOCK_SCHEMA_VERSION:
        raise QualificationError(
            f"artifact_lock schema_version must be {ARTIFACT_LOCK_SCHEMA_VERSION}"
        )
    created_at, _ = _timestamp(raw.get("created_at_utc"), "artifact_lock.created_at_utc")

    images: list[dict[str, Any]] = []
    image_names: set[str] = set()
    for index, entry_raw in enumerate(_list(raw.get("images"), "artifact_lock.images")):
        entry = _object(entry_raw, f"artifact_lock.images[{index}]")
        _require_exact_keys(
            entry,
            {"name", "reference", "sbom_sha256", "provenance_sha256"},
            f"artifact_lock.images[{index}]",
        )
        name = _text(entry.get("name"), f"artifact_lock.images[{index}].name")
        if name in image_names:
            raise QualificationError(f"duplicate artifact image: {name}")
        image_names.add(name)
        reference = _text(entry.get("reference"), f"artifact_lock.images[{index}].reference")
        if "@sha256:" not in reference:
            raise QualificationError(f"artifact image is not digest-pinned: {name}")
        digest = _sha256_text(
            reference.rsplit("@sha256:", 1)[1],
            f"artifact_lock.images[{index}].reference digest",
        )
        images.append(
            {
                "name": name,
                "reference": reference.rsplit("@sha256:", 1)[0] + "@sha256:" + digest,
                "sbom_sha256": _sha256_text(
                    entry.get("sbom_sha256"),
                    f"artifact_lock.images[{index}].sbom_sha256",
                ),
                "provenance_sha256": _sha256_text(
                    entry.get("provenance_sha256"),
                    f"artifact_lock.images[{index}].provenance_sha256",
                ),
            }
        )
    if image_names != REQUIRED_IMAGE_NAMES:
        raise QualificationError(
            "artifact_lock images must be exactly " + ", ".join(sorted(REQUIRED_IMAGE_NAMES))
        )

    def normalize_named_hashes(field: str) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        names: set[str] = set()
        for index, entry_raw in enumerate(_list(raw.get(field), f"artifact_lock.{field}")):
            entry = _object(entry_raw, f"artifact_lock.{field}[{index}]")
            _require_exact_keys(
                entry,
                {"name", "version", "sha256"},
                f"artifact_lock.{field}[{index}]",
            )
            name = _text(entry.get("name"), f"artifact_lock.{field}[{index}].name")
            if name in names:
                raise QualificationError(f"duplicate artifact_lock.{field} name: {name}")
            names.add(name)
            normalized.append(
                {
                    "name": name,
                    "version": _text(
                        entry.get("version"), f"artifact_lock.{field}[{index}].version"
                    ),
                    "sha256": _sha256_text(
                        entry.get("sha256"), f"artifact_lock.{field}[{index}].sha256"
                    ),
                }
            )
        if not normalized:
            raise QualificationError(f"artifact_lock.{field} cannot be empty")
        return sorted(normalized, key=lambda item: item["name"])

    return {
        "schema_version": ARTIFACT_LOCK_SCHEMA_VERSION,
        "created_at_utc": created_at,
        "images": sorted(images, key=lambda item: item["name"]),
        "data": normalize_named_hashes("data"),
        "models": normalize_named_hashes("models"),
    }


def _normalize_panel(
    panel_id: str,
    raw: Mapping[str, Any],
) -> dict[str, Any]:
    _require_exact_keys(
        raw,
        {"selection_method", "panel_sha256", "compound_ids_sha256", "compound_ids"},
        f"ledger.panels.{panel_id}",
    )
    expected_method = "fixed_public" if panel_id == "public" else "sealed_hash_sampled"
    method = _text(raw.get("selection_method"), f"ledger.panels.{panel_id}.selection_method")
    if method != expected_method:
        raise QualificationError(
            f"ledger.panels.{panel_id}.selection_method must be {expected_method}"
        )
    compound_ids = [
        _text(value, f"ledger.panels.{panel_id}.compound_ids")
        for value in _list(raw.get("compound_ids"), f"ledger.panels.{panel_id}.compound_ids")
    ]
    if len(compound_ids) != PANEL_SIZE or len(set(compound_ids)) != PANEL_SIZE:
        raise QualificationError(
            f"ledger.panels.{panel_id} must contain {PANEL_SIZE} unique compounds"
        )
    compound_ids = sorted(compound_ids)
    compound_ids_sha256 = _sha256_text(
        raw.get("compound_ids_sha256"),
        f"ledger.panels.{panel_id}.compound_ids_sha256",
    )
    expected_ids_sha256 = _canonical_sha256({"compound_ids": compound_ids})
    if compound_ids_sha256 != expected_ids_sha256:
        raise QualificationError(
            f"ledger.panels.{panel_id}.compound_ids_sha256 does not bind compound_ids"
        )
    return {
        "selection_method": method,
        "panel_sha256": _sha256_text(
            raw.get("panel_sha256"), f"ledger.panels.{panel_id}.panel_sha256"
        ),
        "compound_ids_sha256": compound_ids_sha256,
        "compound_ids": compound_ids,
    }


def _normalize_observation(
    raw: Mapping[str, Any],
    *,
    index: int,
    panels: Mapping[str, Mapping[str, Any]],
    environment_sha256: str,
    artifact_lock_sha256: str,
) -> dict[str, Any]:
    label = f"ledger.observations[{index}]"
    expected_keys = {
        "panel_id",
        "compound_id",
        "replicate",
        "outcome",
        "accepted_at",
        "ready_at",
        "failure_reason",
        "stage_elapsed_seconds",
        "cache_counts",
        "environment_sha256",
        "artifact_lock_sha256",
        "run_cache_cleared",
            "sealed_report",
    }
    if "duration_seconds" in raw:
        expected_keys.add("duration_seconds")
    _require_exact_keys(
        raw,
        expected_keys,
        label,
    )
    panel_id = _text(raw.get("panel_id"), f"{label}.panel_id")
    if panel_id not in PANELS:
        raise QualificationError(f"{label}.panel_id must be public or sealed")
    compound_id = _text(raw.get("compound_id"), f"{label}.compound_id")
    if compound_id not in panels[panel_id]["compound_ids"]:
        raise QualificationError(f"{label}.compound_id is not in its panel manifest")
    replicate = _integer(raw.get("replicate"), f"{label}.replicate", minimum=1)
    if replicate not in REPLICATES:
        raise QualificationError(f"{label}.replicate must be one of {REPLICATES}")
    outcome = _text(raw.get("outcome"), f"{label}.outcome").lower()
    if outcome not in {"ready", "failure", "timeout"}:
        raise QualificationError(f"{label}.outcome is unsupported: {outcome}")
    accepted_at, accepted = _timestamp(raw.get("accepted_at"), f"{label}.accepted_at")

    ready_at_raw = raw.get("ready_at")
    failure_reason_raw = raw.get("failure_reason")
    sealed_report_raw = raw.get("sealed_report")
    duration_seconds: float | None
    if outcome == "ready":
        ready_at, ready = _timestamp(ready_at_raw, f"{label}.ready_at")
        duration_seconds = (ready - accepted).total_seconds()
        if duration_seconds < 0:
            raise QualificationError(f"{label}.ready_at precedes accepted_at")
        if failure_reason_raw is not None:
            raise QualificationError(f"{label}.failure_reason must be null for ready runs")
        sealed_report: dict[str, Any] | None = _active_file_record(
            sealed_report_raw, f"{label}.sealed_report"
        )
        failure_reason: str | None = None
    else:
        if ready_at_raw is not None:
            raise QualificationError(f"{label}.ready_at must be null for {outcome}")
        ready_at = None
        duration_seconds = None
        failure_reason = _text(failure_reason_raw, f"{label}.failure_reason")
        if sealed_report_raw is not None:
            raise QualificationError(
                f"{label}.sealed_report must be null for {outcome}"
            )
        sealed_report = None

    if "duration_seconds" in raw:
        supplied_duration = raw.get("duration_seconds")
        if duration_seconds is None:
            if supplied_duration is not None:
                raise QualificationError(f"{label}.duration_seconds must be null for {outcome}")
        else:
            parsed_duration = _finite_number(
                supplied_duration,
                f"{label}.duration_seconds",
            )
            if parsed_duration != duration_seconds:
                raise QualificationError(f"{label}.duration_seconds does not match timestamps")

    stages_raw = _object(raw.get("stage_elapsed_seconds"), f"{label}.stage_elapsed_seconds")
    stages: dict[str, float] = {}
    for stage_name, value in stages_raw.items():
        name = _text(stage_name, f"{label}.stage_elapsed_seconds key")
        stages[name] = _finite_number(value, f"{label}.stage_elapsed_seconds.{name}")
    if duration_seconds is not None and sum(stages.values()) > duration_seconds + 1e-6:
        raise QualificationError(f"{label}.stage_elapsed_seconds exceed wall-clock duration")

    cache_raw = _object(raw.get("cache_counts"), f"{label}.cache_counts")
    _require_exact_keys(cache_raw, {"hits", "misses"}, f"{label}.cache_counts")
    cache_counts = {
        "hits": _integer(cache_raw.get("hits"), f"{label}.cache_counts.hits"),
        "misses": _integer(cache_raw.get("misses"), f"{label}.cache_counts.misses"),
    }
    if raw.get("run_cache_cleared") is not True:
        raise QualificationError(f"{label}.run_cache_cleared must be true")
    if _sha256_text(raw.get("environment_sha256"), f"{label}.environment_sha256") != environment_sha256:
        raise QualificationError(f"{label}.environment_sha256 does not match active input")
    if _sha256_text(raw.get("artifact_lock_sha256"), f"{label}.artifact_lock_sha256") != artifact_lock_sha256:
        raise QualificationError(f"{label}.artifact_lock_sha256 does not match active input")

    return {
        "panel_id": panel_id,
        "compound_id": compound_id,
        "replicate": replicate,
        "outcome": outcome,
        "accepted_at": accepted_at,
        "ready_at": ready_at,
        "failure_reason": failure_reason,
        "duration_seconds": duration_seconds,
        "stage_elapsed_seconds": dict(sorted(stages.items())),
        "cache_counts": cache_counts,
        "environment_sha256": environment_sha256,
        "artifact_lock_sha256": artifact_lock_sha256,
        "run_cache_cleared": True,
        "sealed_report": sealed_report,
    }


def _normalize_ledger(
    raw: Mapping[str, Any],
    *,
    environment_sha256: str,
    artifact_lock_sha256: str,
) -> dict[str, Any]:
    _require_exact_keys(
        raw,
        {
            "schema_version",
            "created_at_utc",
            "panels",
            "shared_precompute_sha256",
            "observations",
        },
        "ledger",
    )
    if raw.get("schema_version") != LEDGER_SCHEMA_VERSION:
        raise QualificationError(f"ledger schema_version must be {LEDGER_SCHEMA_VERSION}")
    created_at, _ = _timestamp(raw.get("created_at_utc"), "ledger.created_at_utc")
    panels_raw = _object(raw.get("panels"), "ledger.panels")
    _require_exact_keys(panels_raw, set(PANELS), "ledger.panels")
    panels = {
        panel_id: _normalize_panel(
            panel_id,
            _object(panels_raw.get(panel_id), f"ledger.panels.{panel_id}"),
        )
        for panel_id in PANELS
    }
    overlap = set(panels["public"]["compound_ids"]) & set(
        panels["sealed"]["compound_ids"]
    )
    if overlap:
        raise QualificationError("public and sealed panels must be disjoint")
    shared_precompute_sha256 = _sha256_text(
        raw.get("shared_precompute_sha256"), "ledger.shared_precompute_sha256"
    )
    observations = [
        _normalize_observation(
            _object(entry, f"ledger.observations[{index}]"),
            index=index,
            panels=panels,
            environment_sha256=environment_sha256,
            artifact_lock_sha256=artifact_lock_sha256,
        )
        for index, entry in enumerate(_list(raw.get("observations"), "ledger.observations"))
    ]

    expected_keys = {
        (panel_id, compound_id, replicate)
        for panel_id in PANELS
        for compound_id in panels[panel_id]["compound_ids"]
        for replicate in REPLICATES
    }
    observed_keys = {
        (entry["panel_id"], entry["compound_id"], entry["replicate"])
        for entry in observations
    }
    if len(observations) != len(observed_keys):
        raise QualificationError("ledger observations contain duplicate panel/compound/replicate rows")
    missing = expected_keys - observed_keys
    extra = observed_keys - expected_keys
    if missing or extra:
        raise QualificationError(
            "ledger observations must contain exactly three runs for every panel compound"
        )
    report_paths = [
        entry["sealed_report"]["path"]
        for entry in observations
        if entry["sealed_report"] is not None
    ]
    if len(report_paths) != len(set(report_paths)):
        raise QualificationError("every ready observation must bind a distinct sealed report file")
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "created_at_utc": created_at,
        "panels": panels,
        "shared_precompute_sha256": shared_precompute_sha256,
        "observations": sorted(
            observations,
            key=lambda item: (item["panel_id"], item["compound_id"], item["replicate"]),
        ),
    }


def _duration_sort_value(observation: Mapping[str, Any]) -> float:
    duration = observation.get("duration_seconds")
    return math.inf if duration is None else float(duration)


def _finite_or_null(value: float) -> tuple[float | None, bool]:
    if math.isinf(value):
        return None, True
    return value, False


def _panel_summary(panel_id: str, ledger: Mapping[str, Any]) -> dict[str, Any]:
    rows = [row for row in ledger["observations"] if row["panel_id"] == panel_id]
    durations = sorted(_duration_sort_value(row) for row in rows)
    if len(durations) != OBSERVATIONS_PER_PANEL:
        raise QualificationError(
            f"{panel_id} panel must contain exactly {OBSERVATIONS_PER_PANEL} observations"
        )
    p95 = durations[P95_RANK - 1]
    p95_seconds, p95_is_infinite = _finite_or_null(p95)
    successful = [float(row["duration_seconds"]) for row in rows if row["duration_seconds"] is not None]

    compound_medians: list[dict[str, Any]] = []
    for compound_id in ledger["panels"][panel_id]["compound_ids"]:
        values = sorted(
            _duration_sort_value(row) for row in rows if row["compound_id"] == compound_id
        )
        if len(values) != len(REPLICATES):
            raise QualificationError(f"{panel_id}/{compound_id} does not have three replicates")
        median_seconds, median_is_infinite = _finite_or_null(values[1])
        compound_medians.append(
            {
                "compound_id": compound_id,
                "median_seconds": median_seconds,
                "median_is_infinite": median_is_infinite,
                "passed": not median_is_infinite and float(median_seconds) <= SLA_SECONDS,
            }
        )

    return {
        "panel_id": panel_id,
        "compound_count": PANEL_SIZE,
        "observation_count": OBSERVATIONS_PER_PANEL,
        "p95_method": "nearest_rank",
        "p95_rank": P95_RANK,
        "p95_seconds": p95_seconds,
        "p95_is_infinite": p95_is_infinite,
        "successful_count": len(successful),
        "failure_count": sum(row["outcome"] == "failure" for row in rows),
        "timeout_count": sum(row["outcome"] == "timeout" for row in rows),
        "max_successful_seconds": max(successful) if successful else None,
        "compound_medians": compound_medians,
    }


def _gate(name: str, observed: Any, threshold: Any, passed: bool) -> dict[str, Any]:
    return {
        "name": name,
        "observed": observed,
        "threshold": threshold,
        "passed": bool(passed),
        "status": "passed" if passed else "failed",
    }


def _environment_gates(environment: Mapping[str, Any]) -> list[dict[str, Any]]:
    os_info = environment["os"]
    cpu = environment["cpu"]
    gpu = environment["gpu"]
    storage = environment["storage"]
    os_passed = os_info["id"] == "ubuntu" and (
        os_info["version_id"] == "24.04" or os_info["version_id"].startswith("24.04.")
    )
    return [
        _gate(
            "reference_os",
            f"{os_info['id']} {os_info['version_id']}",
            "Ubuntu 24.04.x",
            os_passed,
        ),
        _gate(
            "reference_cpu",
            cpu["model"],
            "AMD Ryzen 9 7950X",
            bool(CPU_RE.search(cpu["model"])),
        ),
        _gate(
            "reference_memory",
            environment["memory_bytes"],
            f">={MIN_MEMORY_BYTES}",
            environment["memory_bytes"] >= MIN_MEMORY_BYTES,
        ),
        _gate(
            "reference_gpu",
            gpu["name"],
            "NVIDIA GeForce RTX 4090",
            gpu["name"].casefold() == "nvidia geforce rtx 4090".casefold(),
        ),
        _gate(
            "reference_gpu_memory",
            gpu["memory_bytes"],
            f">={MIN_GPU_MEMORY_BYTES}",
            gpu["memory_bytes"] >= MIN_GPU_MEMORY_BYTES,
        ),
        _gate(
            "reference_storage",
            {
                "device": storage["device"],
                "technology": storage["technology"],
                "sequential_read_bytes_per_second": storage[
                    "sequential_read_bytes_per_second"
                ],
            },
            f"NVMe and >={MIN_STORAGE_BYTES_PER_SECOND}",
            storage["technology"] == "nvme"
            and Path(storage["device"]).name.startswith("nvme")
            and storage["sequential_read_bytes_per_second"]
            >= MIN_STORAGE_BYTES_PER_SECOND,
        ),
        _gate(
            "reference_cpu_governor",
            environment["cpu_governor"],
            "performance",
            environment["cpu_governor"] == "performance",
        ),
    ]


def _panel_gates(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    panel_id = str(summary["panel_id"])
    p95_observed: Any = "infinity" if summary["p95_is_infinite"] else summary["p95_seconds"]
    medians_passed = all(row["passed"] for row in summary["compound_medians"])
    maximum = summary["max_successful_seconds"]
    return [
        _gate(
            f"{panel_id}_nearest_rank_p95",
            p95_observed,
            f"<={SLA_SECONDS}",
            not summary["p95_is_infinite"] and float(summary["p95_seconds"]) <= SLA_SECONDS,
        ),
        _gate(
            f"{panel_id}_all_compound_medians",
            sum(row["passed"] for row in summary["compound_medians"]),
            PANEL_SIZE,
            medians_passed,
        ),
        _gate(
            f"{panel_id}_max_successful_duration",
            maximum,
            f"<={MAX_SUCCESS_SECONDS}",
            maximum is not None and float(maximum) <= MAX_SUCCESS_SECONDS,
        ),
    ]


def _reference_host_status(
    environment: Mapping[str, Any],
    environment_gates: Sequence[Mapping[str, Any]],
) -> str:
    if all(gate["passed"] for gate in environment_gates):
        return "reference_candidate"
    version = environment["os"]["version_id"]
    non_os_pass = all(gate["passed"] for gate in environment_gates if gate["name"] != "reference_os")
    if environment["os"]["id"] == "ubuntu" and version.startswith("26.04") and non_os_pass:
        return "ubuntu_26_04_compatibility_only"
    return "not_reference_host"


def _assemble_manifest(
    *,
    environment: Mapping[str, Any],
    artifact_lock: Mapping[str, Any],
    ledger: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    environment_gates = _environment_gates(environment)
    panel_summaries = [_panel_summary(panel_id, ledger) for panel_id in PANELS]
    gates = environment_gates + [
        gate for summary in panel_summaries for gate in _panel_gates(summary)
    ]
    evidence_passed = all(gate["passed"] for gate in gates)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "qualification_evidence_status": "passed" if evidence_passed else "failed",
        "reference_host_status": _reference_host_status(environment, environment_gates),
        "release_qualified": False,
        "release_qualification_reason": "detached_signature_verification_required",
        "thresholds": {
            "panels": list(PANELS),
            "compounds_per_panel": PANEL_SIZE,
            "replicates_per_compound": len(REPLICATES),
            "observations_per_panel": OBSERVATIONS_PER_PANEL,
            "p95_method": "nearest_rank",
            "p95_rank": P95_RANK,
            "p95_seconds_max": SLA_SECONDS,
            "compound_median_seconds_max": SLA_SECONDS,
            "successful_observation_seconds_max": MAX_SUCCESS_SECONDS,
            "failure_timeout_duration": "infinity",
        },
        "inputs": dict(inputs),
        "environment": dict(environment),
        "artifact_lock": dict(artifact_lock),
        "ledger": dict(ledger),
        "panel_summaries": panel_summaries,
        "gates": gates,
        "failed_gates": [gate for gate in gates if not gate["passed"]],
    }
    payload["binding_sha256"] = _canonical_sha256(payload)
    return payload


def evaluate(
    *,
    environment_json: Path,
    sla_ledger_json: Path,
    artifact_lock_json: Path,
) -> dict[str, Any]:
    environment_raw = _read_json_object(environment_json, "qualification environment")
    artifact_raw = _read_json_object(artifact_lock_json, "qualification artifact lock")
    environment_record = _source_record(environment_json)
    artifact_record = _source_record(artifact_lock_json)
    ledger_record = _source_record(sla_ledger_json)
    environment = _normalize_environment(environment_raw)
    artifact_lock = _normalize_artifact_lock(artifact_raw)
    ledger = _normalize_ledger(
        _read_json_object(sla_ledger_json, "qualification SLA ledger"),
        environment_sha256=environment_record["sha256"],
        artifact_lock_sha256=artifact_record["sha256"],
    )
    payload = _assemble_manifest(
        environment=environment,
        artifact_lock=artifact_lock,
        ledger=ledger,
        inputs={
            "environment": environment_record,
            "artifact_lock": artifact_record,
            "sla_ledger": ledger_record,
        },
    )
    validate_manifest(payload)
    return payload


def validate_manifest(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping):
        raise QualificationError("qualification manifest must be an object")
    _require_exact_keys(payload, EXPECTED_TOP_LEVEL_KEYS, "qualification manifest")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise QualificationError(f"schema_version must be {SCHEMA_VERSION}")
    binding = _sha256_text(payload.get("binding_sha256"), "binding_sha256")
    unsigned = dict(payload)
    unsigned.pop("binding_sha256")
    if _canonical_sha256(unsigned) != binding:
        raise QualificationError("qualification manifest binding_sha256 mismatch")
    if payload.get("release_qualified") is not False:
        raise QualificationError("unsigned qualification manifest cannot set release_qualified")
    if payload.get("release_qualification_reason") != "detached_signature_verification_required":
        raise QualificationError("qualification release reason is invalid")

    inputs = _object(payload.get("inputs"), "qualification manifest inputs")
    _require_exact_keys(inputs, {"environment", "artifact_lock", "sla_ledger"}, "inputs")
    for name, record_raw in inputs.items():
        record = _object(record_raw, f"inputs.{name}")
        _require_exact_keys(record, {"path", "bytes", "sha256"}, f"inputs.{name}")
        _text(record.get("path"), f"inputs.{name}.path")
        _integer(record.get("bytes"), f"inputs.{name}.bytes", minimum=1)
        _sha256_text(record.get("sha256"), f"inputs.{name}.sha256")

    environment = _normalize_environment(
        _object(payload.get("environment"), "qualification manifest environment")
    )
    artifact_lock = _normalize_artifact_lock(
        _object(payload.get("artifact_lock"), "qualification manifest artifact_lock")
    )
    ledger = _normalize_ledger(
        _object(payload.get("ledger"), "qualification manifest ledger"),
        environment_sha256=inputs["environment"]["sha256"],
        artifact_lock_sha256=inputs["artifact_lock"]["sha256"],
    )
    fresh = _assemble_manifest(
        environment=environment,
        artifact_lock=artifact_lock,
        ledger=ledger,
        inputs=inputs,
    )
    if fresh != dict(payload):
        raise QualificationError("qualification manifest does not match fresh internal evaluation")


def validate_manifest_against_inputs(
    payload: Mapping[str, Any],
    *,
    environment_json: Path,
    sla_ledger_json: Path,
    artifact_lock_json: Path,
) -> None:
    validate_manifest(payload)
    fresh = evaluate(
        environment_json=environment_json,
        sla_ledger_json=sla_ledger_json,
        artifact_lock_json=artifact_lock_json,
    )
    if fresh != dict(payload):
        raise QualificationError("qualification manifest does not match active input files")


def load_trust_policy(path: Path = DEFAULT_TRUST_POLICY) -> dict[str, str]:
    payload = _read_json_object(path, "qualification trust policy")
    _require_exact_keys(
        payload,
        {"schema_version", "fail_closed", "verification", "trusted_signer"},
        "qualification trust policy",
    )
    if payload.get("schema_version") != TRUST_POLICY_SCHEMA_VERSION:
        raise QualificationError(
            f"qualification trust policy schema_version must be {TRUST_POLICY_SCHEMA_VERSION}"
        )
    if payload.get("fail_closed") is not True or payload.get("verification") != "cosign":
        raise QualificationError("qualification trust policy must be fail-closed Cosign")
    signer = _object(payload.get("trusted_signer"), "qualification trust policy signer")
    _require_exact_keys(
        signer,
        {"type", "certificate_identity_regexp", "issuer"},
        "qualification trust policy signer",
    )
    if signer.get("type") != "keyless":
        raise QualificationError("qualification trust policy signer must be keyless")
    identity_regexp = _text(
        signer.get("certificate_identity_regexp"),
        "qualification trust policy signer certificate_identity_regexp",
    )
    if not identity_regexp.startswith("^") or not identity_regexp.endswith("$"):
        raise QualificationError("qualification signer identity regexp must be fully anchored")
    try:
        re.compile(identity_regexp)
    except re.error as exc:
        raise QualificationError("qualification signer identity regexp is invalid") from exc
    issuer = _text(signer.get("issuer"), "qualification trust policy signer issuer")
    if not issuer.startswith("https://"):
        raise QualificationError("qualification signer issuer must use HTTPS")
    return {
        "certificate_identity_regexp": identity_regexp,
        "issuer": issuer,
        "policy_path": str(path.resolve(strict=True)),
        "policy_sha256": _sha256(path.resolve(strict=True)),
    }


def verify_cosign_signature(
    manifest_path: Path,
    *,
    bundle_path: Path,
    trust_policy: Mapping[str, str],
) -> None:
    if bundle_path.is_symlink() or not bundle_path.is_file() or bundle_path.stat().st_size <= 0:
        raise QualificationError(f"Cosign bundle is missing, empty, or unsafe: {bundle_path}")
    command = [
        "cosign",
        "verify-blob",
        "--bundle",
        str(bundle_path),
        "--certificate-identity-regexp",
        trust_policy["certificate_identity_regexp"],
        "--certificate-oidc-issuer",
        trust_policy["issuer"],
    ]
    command.append(str(manifest_path))
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False)
    except OSError as exc:
        raise QualificationError(f"cannot execute Cosign verification: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        raise QualificationError(f"Cosign qualification signature verification failed: {detail}")


def validate_release(
    *,
    manifest_path: Path,
    environment_json: Path,
    sla_ledger_json: Path,
    artifact_lock_json: Path,
    bundle_path: Path | None = None,
    require_signed: bool = True,
) -> dict[str, Any]:
    payload = _read_json_object(manifest_path, "qualification manifest")
    validate_manifest_against_inputs(
        payload,
        environment_json=environment_json,
        sla_ledger_json=sla_ledger_json,
        artifact_lock_json=artifact_lock_json,
    )
    signature_verified = False
    trust_policy_record: dict[str, str] | None = None
    if bundle_path is not None:
        trust_policy_record = load_trust_policy()
        verify_cosign_signature(
            manifest_path,
            bundle_path=bundle_path,
            trust_policy=trust_policy_record,
        )
        signature_verified = True
    elif require_signed:
        raise QualificationError("signed qualification requires a detached Cosign bundle")

    evidence_passed = payload["qualification_evidence_status"] == "passed"
    release_qualified = evidence_passed and signature_verified
    if require_signed and not release_qualified:
        raise QualificationError("signed qualification evidence did not pass all release gates")
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "manifest_path": str(manifest_path.resolve(strict=True)),
        "manifest_sha256": _sha256(manifest_path.resolve(strict=True)),
        "binding_sha256": payload["binding_sha256"],
        "qualification_evidence_status": payload["qualification_evidence_status"],
        "reference_host_status": payload["reference_host_status"],
        "validated": True,
        "diagnostic": not require_signed,
        "qualified": release_qualified,
        "signature_verified": signature_verified,
        "signature_status": "verified" if signature_verified else "unverified",
        "trust_policy": trust_policy_record,
        "release_qualified": release_qualified,
        "release_qualification_reason": (
            "signed_evidence_passed"
            if release_qualified
            else "evidence_failed"
            if not evidence_passed
            else "detached_signature_verification_required"
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = subparsers.add_parser("evaluate", help="build qualification evidence")
    evaluate_parser.add_argument("--environment-json", type=Path, required=True)
    evaluate_parser.add_argument("--sla-ledger-json", type=Path, required=True)
    evaluate_parser.add_argument("--artifact-lock-json", type=Path, required=True)
    evaluate_parser.add_argument("--out-json", type=Path, required=True)
    evaluate_parser.add_argument("--allow-failed-evidence", action="store_true")

    validate_parser = subparsers.add_parser("validate", help="validate active evidence")
    validate_parser.add_argument("--manifest", type=Path, required=True)
    validate_parser.add_argument("--environment-json", type=Path, required=True)
    validate_parser.add_argument("--sla-ledger-json", type=Path, required=True)
    validate_parser.add_argument("--artifact-lock-json", type=Path, required=True)
    validate_parser.add_argument("--signature-bundle", type=Path)
    validate_parser.add_argument(
        "--allow-unsigned-evidence",
        action="store_true",
        help=(
            "Diagnostic only: validate evidence without qualifying a release. "
            "Exit 3 when evidence passes but is not signed; exit 2 on failed evidence."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "evaluate":
            args.out_json.unlink(missing_ok=True)
            payload = evaluate(
                environment_json=args.environment_json,
                sla_ledger_json=args.sla_ledger_json,
                artifact_lock_json=args.artifact_lock_json,
            )
            _write_json_atomic(args.out_json, payload)
            print(json.dumps(payload, sort_keys=True))
            if payload["qualification_evidence_status"] == "passed":
                return 0
            if args.allow_failed_evidence:
                return 3
            return 2

        result = validate_release(
            manifest_path=args.manifest,
            environment_json=args.environment_json,
            sla_ledger_json=args.sla_ledger_json,
            artifact_lock_json=args.artifact_lock_json,
            bundle_path=args.signature_bundle,
            require_signed=not args.allow_unsigned_evidence,
        )
        print(json.dumps(result, sort_keys=True))
        if result["release_qualified"]:
            return 0
        if result["diagnostic"] and result["qualification_evidence_status"] == "passed":
            return 3
        return 2
    except (QualificationError, OSError) as exc:
        print(f"[SkinScout][ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
