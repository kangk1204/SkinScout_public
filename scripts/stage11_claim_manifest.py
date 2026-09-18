#!/usr/bin/env python3
"""Emit a fail-closed publication claim manifest for a completed run."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from report_package_contract import (
        ReportContractError,
        validate_report_packages,
    )
except ModuleNotFoundError:  # pragma: no cover - importlib-based unit tests
    from scripts.report_package_contract import (
        ReportContractError,
        validate_report_packages,
    )

try:
    from validate_activity_retrieval_gate import (
        GATE_SCHEMA as ACTIVITY_RETRIEVAL_GATE_SCHEMA,
        check_gate as check_activity_retrieval_gate,
    )
except ModuleNotFoundError:  # pragma: no cover - importlib-based unit tests
    from scripts.validate_activity_retrieval_gate import (
        GATE_SCHEMA as ACTIVITY_RETRIEVAL_GATE_SCHEMA,
        check_gate as check_activity_retrieval_gate,
    )


SCIENTIFIC_CLAIM_SCOPE = "computational_and_literature_candidate_validation_only"
GIT_COMMIT_LENGTH = 40
SHA256_LENGTH = 64


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _safe_resolve_artifact(path_text: str, run_dir: Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = run_dir / path
    return path.resolve()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _json_or_none(path: Path) -> Any | None:
    if path.suffix.lower() != ".json":
        return None
    return json.loads(path.read_text())


def _status_blockers(payload: Any, source: str) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []

    def walk(value: Any, location: str) -> None:
        if isinstance(value, dict):
            if value.get("execution_status") == "blocked":
                blockers.append({
                    "code": "blocked_execution_status",
                    "source": source,
                    "field": f"{location}.execution_status",
                })
            if value.get("diagnostic_only") is True:
                blockers.append({
                    "code": "diagnostic_only_status",
                    "source": source,
                    "field": f"{location}.diagnostic_only",
                })
            if value.get("claim_eligible") is False:
                blockers.append({
                    "code": "claim_ineligible_status",
                    "source": source,
                    "field": f"{location}.claim_eligible",
                })
            for key, child in value.items():
                walk(child, f"{location}.{key}")
        elif isinstance(value, list):
            for idx, child in enumerate(value):
                walk(child, f"{location}[{idx}]")

    walk(payload, "$")
    return blockers


def _artifact_record(label: str, path: Path, run_dir: Path) -> dict[str, Any]:
    return {
        "label": label,
        "path": str(path),
        "relative_path": str(path.relative_to(run_dir)),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _parse_artifact_spec(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"artifact must use LABEL=PATH syntax: {spec!r}")
    label, path = spec.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label:
        raise ValueError(f"artifact label must be non-empty: {spec!r}")
    if not path:
        raise ValueError(f"artifact path must be non-empty: {spec!r}")
    return label, path


def _git_commit_from_repo() -> str | None:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    if len(commit) == 40 and all(ch in "0123456789abcdefABCDEF" for ch in commit):
        return commit
    return None


def _canonical_sidecar_text(run_dir: Path, name: str) -> str | None:
    path = run_dir / "publication" / "reproducibility" / name
    if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
        return None
    return path.read_text().strip() or None


def _valid_hex(value: str | None, length: int) -> bool:
    return bool(
        value
        and len(value) == length
        and all(ch in "0123456789abcdefABCDEF" for ch in value)
    )


def _repro_provenance(
    manifest_path: Path,
) -> tuple[str | None, str | None, list[dict[str, Any]]]:
    blockers: list[dict[str, Any]] = []
    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        return None, None, [{
            "code": "invalid_reproducibility_manifest",
            "path": str(manifest_path),
            "detail": str(exc),
        }]
    metadata = manifest.get("metadata_files") if isinstance(manifest, dict) else None
    if not isinstance(metadata, list):
        return None, None, [{
            "code": "invalid_reproducibility_metadata",
            "path": str(manifest_path),
        }]
    values: dict[str, str] = {}
    for name, length in (
        ("git_commit.txt", GIT_COMMIT_LENGTH),
        ("config_hash.txt", SHA256_LENGTH),
    ):
        records = [
            record for record in metadata
            if isinstance(record, dict) and record.get("relative_path") == name
        ]
        sidecar = manifest_path.parent / name
        if len(records) != 1 or not sidecar.is_file() or sidecar.is_symlink():
            blockers.append({
                "code": "missing_reproducibility_provenance",
                "field": name,
                "path": str(sidecar),
            })
            continue
        record = records[0]
        digest = _sha256(sidecar)
        if record.get("bytes") != sidecar.stat().st_size or record.get("sha256") != digest:
            blockers.append({
                "code": "reproducibility_provenance_binding_mismatch",
                "field": name,
                "path": str(sidecar),
            })
            continue
        value = sidecar.read_text().strip()
        if not _valid_hex(value, length):
            blockers.append({
                "code": "invalid_reproducibility_provenance_format",
                "field": name,
                "path": str(sidecar),
            })
            continue
        values[name] = value.lower()
    return values.get("git_commit.txt"), values.get("config_hash.txt"), blockers


def _collect_manifest(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    blockers: list[dict[str, Any]] = []
    run_dir = args.run_dir.resolve()
    artifact_records: list[dict[str, Any]] = []
    artifact_paths_by_label: dict[str, Path] = {}
    seen_labels: set[str] = set()
    seen_paths: set[Path] = set()

    if not run_dir.exists() or not run_dir.is_dir():
        blockers.append({
            "code": "missing_run_dir",
            "path": str(run_dir),
        })

    report_packages: dict[str, Any] | None = None
    report_pointer_values = (
        args.fast_report_pointer,
        args.physics_report_pointer,
    )
    if any(value is not None for value in report_pointer_values):
        if not all(value is not None for value in report_pointer_values):
            blockers.append({
                "code": "incomplete_report_pointer_contract",
                "detail": (
                    "both --fast-report-pointer and --physics-report-pointer "
                    "are required"
                ),
            })
        elif run_dir.exists() and run_dir.is_dir():
            try:
                report_packages = validate_report_packages(
                    run_dir=run_dir,
                    fast_pointer=args.fast_report_pointer,
                    physics_pointer=args.physics_report_pointer,
                    require_physics=args.require_physics_report,
                )
            except ReportContractError as exc:
                blockers.append({
                    "code": "invalid_report_package_contract",
                    "detail": str(exc),
                })

    for spec in args.artifact:
        try:
            label, path_text = _parse_artifact_spec(spec)
        except ValueError as exc:
            blockers.append({"code": "invalid_artifact_spec", "detail": str(exc)})
            continue

        path = _safe_resolve_artifact(path_text, run_dir)
        if label in seen_labels:
            blockers.append({"code": "duplicate_artifact_label", "label": label})
            continue
        seen_labels.add(label)
        if path in seen_paths:
            blockers.append({"code": "duplicate_artifact_path", "path": str(path)})
            continue
        seen_paths.add(path)

        if not _is_relative_to(path, run_dir):
            blockers.append({
                "code": "artifact_outside_run_dir",
                "label": label,
                "path": str(path),
            })
            continue
        if not path.exists() or not path.is_file():
            blockers.append({
                "code": "missing_artifact",
                "label": label,
                "path": str(path),
            })
            continue
        if path.stat().st_size == 0:
            blockers.append({
                "code": "empty_artifact",
                "label": label,
                "path": str(path),
            })
            continue

        artifact_records.append(_artifact_record(label, path, run_dir))
        artifact_paths_by_label[label] = path
        try:
            status_payload = _json_or_none(path)
        except json.JSONDecodeError as exc:
            blockers.append({
                "code": "invalid_json_artifact",
                "label": label,
                "path": str(path),
                "detail": str(exc),
            })
            continue
        if status_payload is not None:
            blockers.extend(_status_blockers(status_payload, label))

    evaluation_manifest: dict[str, Any] | None = None
    activity_retrieval_gate: dict[str, Any] | None = None
    eval_payload: dict[str, Any] | None = None
    if args.evaluation_manifest is None:
        blockers.append({"code": "missing_evaluation_manifest"})
    else:
        eval_path = args.evaluation_manifest.resolve()
        if not eval_path.exists() or not eval_path.is_file():
            blockers.append({
                "code": "missing_evaluation_manifest",
                "path": str(eval_path),
            })
        elif eval_path.stat().st_size == 0:
            blockers.append({
                "code": "empty_evaluation_manifest",
                "path": str(eval_path),
            })
        else:
            try:
                loaded_eval_payload = json.loads(eval_path.read_text())
            except json.JSONDecodeError as exc:
                blockers.append({
                    "code": "invalid_evaluation_manifest_json",
                    "path": str(eval_path),
                    "detail": str(exc),
                })
            else:
                if not isinstance(loaded_eval_payload, dict):
                    blockers.append({
                        "code": "invalid_evaluation_manifest_shape",
                        "path": str(eval_path),
                    })
                    loaded_eval_payload = {}
                eval_payload = loaded_eval_payload
                blockers.extend(_status_blockers(eval_payload, "evaluation_manifest"))
                if eval_payload.get("claim_ready") is not True:
                    blockers.append({
                        "code": "evaluation_not_claim_ready",
                        "path": str(eval_path),
                    })
                if eval_payload.get("claim_blockers"):
                    blockers.append({
                        "code": "evaluation_claim_blockers",
                        "path": str(eval_path),
                        "detail": eval_payload.get("claim_blockers"),
                    })
                gate_record = eval_payload.get("activity_retrieval_gate")
                if (
                    not isinstance(gate_record, dict)
                    or gate_record.get("status") != "pass"
                    or gate_record.get("schema_version")
                    != ACTIVITY_RETRIEVAL_GATE_SCHEMA
                ):
                    blockers.append({
                        "code": "activity_retrieval_gate_not_ready",
                        "path": str(eval_path),
                    })
                else:
                    gate_path_value = gate_record.get("path")
                    gate_path = (
                        Path(gate_path_value).resolve()
                        if isinstance(gate_path_value, str) and gate_path_value
                        else None
                    )
                    expected_gate = (
                        args.activity_retrieval_gate.resolve()
                        if args.activity_retrieval_gate is not None
                        else gate_path
                    )
                    if gate_path is None or expected_gate != gate_path:
                        blockers.append({
                            "code": "activity_retrieval_gate_path_mismatch",
                            "path": str(gate_path) if gate_path is not None else None,
                        })
                    elif (
                        gate_path.is_symlink()
                        or not gate_path.is_file()
                        or gate_path.stat().st_size <= 0
                    ):
                        blockers.append({
                            "code": "missing_activity_retrieval_gate",
                            "path": str(gate_path),
                        })
                    elif (
                        gate_record.get("bytes") != gate_path.stat().st_size
                        or gate_record.get("sha256") != _sha256(gate_path)
                    ):
                        blockers.append({
                            "code": "activity_retrieval_gate_binding_mismatch",
                            "path": str(gate_path),
                        })
                    else:
                        try:
                            check_activity_retrieval_gate(gate_path)
                        except (OSError, ValueError, SystemExit) as exc:
                            blockers.append({
                                "code": "invalid_activity_retrieval_gate",
                                "path": str(gate_path),
                                "detail": str(exc),
                            })
                        else:
                            activity_retrieval_gate = {
                                "status": "pass",
                                "schema_version": ACTIVITY_RETRIEVAL_GATE_SCHEMA,
                                "path": str(gate_path),
                                "bytes": gate_path.stat().st_size,
                                "sha256": _sha256(gate_path),
                            }
                evaluation_manifest = {
                    "path": str(eval_path),
                    "bytes": eval_path.stat().st_size,
                    "sha256": _sha256(eval_path),
                    "claim_ready": eval_payload.get("claim_ready"),
                }

    sidecar_git_commit = None
    sidecar_config_hash = None
    repro_path = artifact_paths_by_label.get("reproducibility")
    if repro_path is not None:
        sidecar_git_commit, sidecar_config_hash, repro_blockers = _repro_provenance(
            repro_path
        )
        blockers.extend(repro_blockers)
        eval_provenance = eval_payload.get("provenance") if eval_payload else None
        if not isinstance(eval_provenance, dict):
            blockers.append({"code": "missing_evaluation_provenance"})
        else:
            eval_git = str(eval_provenance.get("git_commit", "")).strip().lower()
            eval_config = str(eval_provenance.get("config_sha256", "")).strip().lower()
            if not _valid_hex(eval_git, GIT_COMMIT_LENGTH) or not _valid_hex(
                eval_config, SHA256_LENGTH
            ):
                blockers.append({"code": "invalid_evaluation_provenance_format"})
            elif (
                sidecar_git_commit != eval_git
                or sidecar_config_hash != eval_config
            ):
                blockers.append({"code": "evaluation_reproducibility_provenance_mismatch"})
    elif run_dir.exists():
        sidecar_git_commit = _canonical_sidecar_text(run_dir, "git_commit.txt")
        sidecar_config_hash = _canonical_sidecar_text(run_dir, "config_hash.txt")
        if sidecar_git_commit is not None and not _valid_hex(
            sidecar_git_commit, GIT_COMMIT_LENGTH
        ):
            blockers.append({"code": "invalid_git_commit_format"})
        if sidecar_config_hash is not None and not _valid_hex(
            sidecar_config_hash, SHA256_LENGTH
        ):
            blockers.append({"code": "invalid_config_hash_format"})
    payload: dict[str, Any] = {
        "run_id": run_dir.name,
        "git_commit": sidecar_git_commit or _git_commit_from_repo(),
        "config_hash": sidecar_config_hash,
        "scientific_claim_scope": SCIENTIFIC_CLAIM_SCOPE,
        "artifacts": artifact_records,
        "report_packages": report_packages,
        "evaluation_manifest": evaluation_manifest,
        "activity_retrieval_gate": activity_retrieval_gate,
        "claim_ready": not blockers,
        "blockers": blockers,
    }
    return payload, blockers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out-manifest", required=True, type=Path)
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="required publication artifact label and path; repeat as needed",
    )
    parser.add_argument("--evaluation-manifest", type=Path)
    parser.add_argument("--activity-retrieval-gate", type=Path)
    parser.add_argument("--fast-report-pointer", type=Path)
    parser.add_argument("--physics-report-pointer", type=Path)
    parser.add_argument("--require-physics-report", action="store_true")
    parser.add_argument(
        "--allow-diagnostic",
        action="store_true",
        help="write a claim_ready=false manifest instead of failing on blockers",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.artifact:
        args.out_manifest.unlink(missing_ok=True)
        parser.error("at least one --artifact LABEL=PATH is required")

    payload, blockers = _collect_manifest(args)
    if blockers and not args.allow_diagnostic:
        args.out_manifest.unlink(missing_ok=True)
        print(
            "claim manifest is not claim-ready: "
            + ", ".join(str(item["code"]) for item in blockers),
            file=sys.stderr,
        )
        return 1

    if blockers:
        payload["claim_ready"] = False
    _write_json_atomic(args.out_manifest, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
