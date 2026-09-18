#!/usr/bin/env python3
"""Run available SkinScout evaluation checks and emit an iteration manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import secrets
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

try:
    from discovery_canonical import current_rdkit_version, validate_rdkit_version_contract
    from leakage_check import (
        Thresholds as LeakageThresholds,
        validate_sealed_discovery_audit_sources,
    )
    from prospective_promotion_eval import (
        validate_manifest_against_metrics as validate_prospective_manifest,
    )
    from preregistered_candidate_matrix import (
        CandidateMatrixError,
        evaluate_results as evaluate_candidate_results,
        validate_preregistration_contract,
        validate_selection_manifest,
    )
except ImportError:  # pragma: no cover - importlib/spec fallback
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from discovery_canonical import current_rdkit_version, validate_rdkit_version_contract
    from leakage_check import (
        Thresholds as LeakageThresholds,
        validate_sealed_discovery_audit_sources,
    )
    from prospective_promotion_eval import (
        validate_manifest_against_metrics as validate_prospective_manifest,
    )
    from preregistered_candidate_matrix import (
        CandidateMatrixError,
        evaluate_results as evaluate_candidate_results,
        validate_preregistration_contract,
        validate_selection_manifest,
    )

PROSPECTIVE_PROMOTION_SCHEMA = "skinscout.prospective-discovery-promotion.v1"
DIRECT_DISCOVERY_SOURCE_LABELS = {
    "direct",
    "direct_exact",
    "exact_direct",
    "discovery_direct",
    "discovery_exact",
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from validate_activity_retrieval_gate import (  # noqa: E402
    GATE_SCHEMA as ACTIVITY_RETRIEVAL_GATE_SCHEMA,
    check_gate as check_activity_retrieval_gate,
)


def _resolve_direct_exact_inputs(
    *,
    eval_dir: Path,
    explicit_reference: Path | None,
    explicit_manifest: Path | None,
    project_root: Path = PROJECT_ROOT,
) -> tuple[Path, Path | None]:
    if explicit_reference is not None:
        return explicit_reference, explicit_manifest
    legacy_reference = eval_dir / "direct_exact_reference.smi"
    if legacy_reference.exists():
        return legacy_reference, explicit_manifest
    canonical_dir = project_root / "data" / "discovery_aliases"
    return (
        canonical_dir / "direct_exact_reference.smi",
        explicit_manifest or canonical_dir / "manifest.json",
    )


@dataclass
class EvalStep:
    name: str
    status: str
    command: list[str]
    stdout: str
    stderr: str
    outputs: list[str]


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def run_step(name: str, command: list[str], outputs: list[Path]) -> EvalStep:
    seen: set[Path] = set()
    for path in outputs:
        if path in seen:
            continue
        seen.add(path)
        if path.exists() and path.is_file():
            path.unlink()
    res = subprocess.run(command, capture_output=True, text=True, check=False)
    status = (
        "passed"
        if (
            res.returncode == 0
            and all(p.exists() and p.stat().st_size > 0 for p in outputs)
        )
        else "failed"
    )
    return EvalStep(
        name=name,
        status=status,
        command=command,
        stdout=res.stdout[-4000:],
        stderr=res.stderr[-4000:],
        outputs=[str(p) for p in outputs],
    )


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _activity_retrieval_gate_status(path: Path) -> dict[str, object]:
    status: dict[str, object] = {
        "status": "missing",
        "path": str(path.resolve()),
        "schema_version": ACTIVITY_RETRIEVAL_GATE_SCHEMA,
    }
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        return status
    status.update({
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    })
    try:
        payload = check_activity_retrieval_gate(path)
    except (OSError, ValueError, SystemExit) as exc:
        status["status"] = "failed"
        status["error"] = str(exc)
        return status
    if (
        payload.get("schema_version") != ACTIVITY_RETRIEVAL_GATE_SCHEMA
        or payload.get("status") != "pass"
    ):
        status["status"] = "failed"
        status["error"] = "activity retrieval gate schema/status mismatch"
        return status
    status["status"] = "pass"
    return status


def _file_record(
    path: Path,
    root: Path,
    snapshot_label: str = "Evaluation artifact snapshot",
) -> dict[str, object]:
    rel_path = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
    size = path.stat().st_size
    if size == 0:
        raise SystemExit(f"{snapshot_label} contains empty file: {rel_path}")
    return {
        "path": str(path),
        "relative_path": rel_path,
        "bytes": size,
        "sha256": _sha256(path),
    }


def _artifact_snapshot(paths: list[Path], out_manifest: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    seen: set[Path] = set()
    seen_relative_paths: dict[str, Path] = {}
    for root in paths:
        if not root.exists():
            continue
        files = (
            [root]
            if root.is_file()
            else sorted(p for p in root.rglob("*") if p.is_file())
        )
        for path in files:
            resolved = path.resolve()
            if resolved == out_manifest.resolve() or resolved in seen:
                continue
            seen.add(resolved)
            record = _file_record(path, root if root.is_dir() else root.parent)
            relative_path = str(record["relative_path"])
            previous = seen_relative_paths.get(relative_path)
            if previous is not None:
                raise SystemExit(
                    "Evaluation artifact snapshot contains duplicate "
                    f"relative_path {relative_path!r}: {previous}, {resolved}"
                )
            seen_relative_paths[relative_path] = resolved
            records.append(record)
    return records


def _data_snapshot(paths: dict[str, Path | None]) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    seen_file_paths: dict[Path, str] = {}
    for name, path in paths.items():
        if path is None:
            records[name] = {"path": None, "status": "not_applicable"}
            continue
        if not path.exists():
            records[name] = {
                "path": str(path),
                "status": "missing",
            }
            continue
        if path.is_file():
            resolved = path.resolve()
            previous = seen_file_paths.get(resolved)
            if previous is not None:
                raise SystemExit(
                    "Data snapshot contains duplicate file path: "
                    f"{name} and {previous}: {resolved}"
                )
            seen_file_paths[resolved] = name
            records[name] = {
                "path": str(path),
                "status": "present",
                "kind": "file",
                **_file_record(path, path.parent, "Data snapshot"),
            }
            continue
        files = sorted(p for p in path.rglob("*") if p.is_file())
        if not files:
            raise SystemExit(f"Data snapshot contains empty directory: {path}")
        entries = []
        for file_path in files:
            resolved = file_path.resolve()
            previous = seen_file_paths.get(resolved)
            if previous is not None:
                raise SystemExit(
                    "Data snapshot contains duplicate file path: "
                    f"{name} and {previous}: {resolved}"
                )
            seen_file_paths[resolved] = name
            entries.append(_file_record(file_path, path, "Data snapshot"))
        records[name] = {
            "path": str(path),
            "status": "present",
            "kind": "directory",
            "n_files": len(entries),
            "entries": entries,
        }
    return records


def _tool_versions() -> dict[str, str]:
    tools = ["git", "snakemake", "python", "rdkit", "mmseqs", "gnina", "xtb", "gmx"]
    versions: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for tool in tools:
        if tool in {"python", "rdkit"}:
            continue
        exe = shutil.which(tool)
        if exe is None:
            versions[tool] = "missing"
            continue
        cmd = [exe, "--version"]
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            text = (res.stdout or res.stderr).strip().splitlines()
            versions[tool] = text[0] if text else "available"
        except (OSError, subprocess.TimeoutExpired):
            versions[tool] = "available"
    try:
        from rdkit import rdBase

        versions["rdkit"] = rdBase.rdkitVersion
    except ImportError:
        versions["rdkit"] = "missing"
    return versions


def _require_40_hex_sha(value: str, label: str) -> str:
    value = value.strip()
    if len(value) != 40 or any(ch not in "0123456789abcdefABCDEF" for ch in value):
        raise SystemExit(f"{label} must be a 40-hex SHA, got: {value or '<blank>'}")
    return value


def _require_valid_smiles(value: str, label: str) -> str:
    smiles = value.strip()
    if not smiles:
        raise SystemExit(f"{label} must be non-empty")
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise SystemExit(
            f"{label} validation requires RDKit to be installed"
        ) from exc
    if Chem.MolFromSmiles(smiles) is None:
        raise SystemExit(f"{label} must be a parseable SMILES: {smiles}")
    return smiles


def _validate_fraction_threshold(value: float, label: str) -> None:
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise SystemExit(
            f"{label} must be a finite value in [0, 1]: {value!r}"
        )


def _provenance(config_path: Path = Path("workflow/config.yaml")) -> dict[str, object]:
    if not config_path.exists():
        raise SystemExit(
            f"Workflow config is required for iteration provenance: {config_path}"
        )
    git_commit = _require_40_hex_sha(_git_commit(), "Git commit provenance")
    tool_versions = _tool_versions()
    for tool in ("python", "platform", "rdkit"):
        value = str(tool_versions.get(tool, "")).strip()
        if not value:
            raise SystemExit(f"Tool provenance is missing required entry: {tool}")
        if tool in {"python", "rdkit"} and value.lower() == "missing":
            raise SystemExit(
                f"Tool provenance requires an available {tool} version"
            )
    try:
        validate_rdkit_version_contract(tool_versions["rdkit"])
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return {
        "git_commit": git_commit,
        "config_sha256": _sha256(config_path),
        "tool_versions": tool_versions,
        "rdkit_version_contract": {
            "mode": "exact-runtime",
            "required_version": current_rdkit_version(),
        },
    }


def _read_csv_if_present(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise SystemExit(f"{path.name} failed to parse: {exc}") from exc


def _read_json_object_if_present(path: Path, label: str) -> dict[str, object] | None:
    if not path.exists():
        return None
    if path.stat().st_size == 0:
        raise SystemExit(f"{label} is empty: {path}")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{label} failed to parse: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} must be a JSON object: {path}")
    return payload


def _prospective_promotion_status(
    path: Path,
    metrics_csv: Path,
) -> dict[str, object]:
    payload = _read_json_object_if_present(path, "Prospective promotion manifest")
    if payload is None:
        return {
            "status": "not_run",
            "path": str(path),
            "metrics_path": str(metrics_csv),
            "model_promotion_ready": False,
            "software_release_independent": True,
            "sota_claim_scope": "separate",
        }
    try:
        validate_prospective_manifest(payload, metrics_csv=metrics_csv)
    except (RuntimeError, SystemExit, ValueError) as exc:
        raise SystemExit(f"Prospective promotion manifest is invalid: {path}: {exc}") from exc
    failed = payload["failed_gates"]
    status = str(payload["status"]).strip()
    return {
        "status": status,
        "path": str(path),
        "metrics_path": str(metrics_csv),
        "metrics_sha256": _sha256(metrics_csv),
        "model_promotion_ready": status == "promote" and not failed,
        "software_release_independent": True,
        "sota_claim_scope": "separate",
        "failed_gates": failed,
    }


def _read_required_json_object(path: Path, label: str) -> dict[str, object]:
    if not path.exists():
        raise SystemExit(f"{label} is missing: {path}")
    if path.stat().st_size == 0:
        raise SystemExit(f"{label} is empty: {path}")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{label} failed to parse: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} must be a JSON object: {path}")
    return payload


def _candidate_experiment_status(
    prereg_json: Path,
    results_csv: Path,
    selection_json: Path,
    *,
    prereg_arg: Path | None,
    results_arg: Path | None,
    selection_arg: Path | None,
) -> dict[str, object]:
    paths = {
        "preregistration": prereg_json,
        "results": results_csv,
        "selection": selection_json,
    }
    explicit = {
        "preregistration": prereg_arg is not None,
        "results": results_arg is not None,
        "selection": selection_arg is not None,
    }
    present = {name: path.exists() for name, path in paths.items()}
    if not any(explicit.values()) and not any(present.values()):
        return {
            "status": "not_run",
            "eligible": False,
            "paths": {name: str(path) for name, path in paths.items()},
            "guardrail_status": "not_run",
        }
    missing = [
        name
        for name, path in paths.items()
        if explicit[name] or not path.exists() or path.stat().st_size == 0
        if not path.exists() or path.stat().st_size == 0
    ]
    if missing:
        detail = ", ".join(f"{name}={paths[name]}" for name in missing)
        raise SystemExit(
            "Candidate experiment artifacts are all-or-none; missing required "
            f"artifact(s): {detail}"
        )
    try:
        prereg_payload = validate_preregistration_contract(
            _read_required_json_object(prereg_json, "Candidate preregistration")
        )
        selection_payload = validate_selection_manifest(
            _read_required_json_object(selection_json, "Candidate selection"),
            prereg_json=prereg_json,
            results_csv=results_csv,
        )
        fresh_selection = evaluate_candidate_results(prereg_json, results_csv)
    except CandidateMatrixError as exc:
        raise SystemExit(
            "Candidate experiment artifact validation failed: "
            f"{exc}"
        ) from exc
    if selection_payload != fresh_selection:
        raise SystemExit(
            "Candidate experiment artifact validation failed: selection JSON "
            "does not match fresh evaluation"
        )
    selected = selection_payload.get("selected")
    selected_candidate = None
    selected_config = None
    if isinstance(selected, dict):
        selected_candidate = selected.get("candidate_id")
        selected_config = selected.get("config_id")
    eligible = isinstance(selected, dict) and bool(selected.get("eligible"))
    rejected = selection_payload.get("rejected", [])
    eligible_rows = selection_payload.get("eligible", [])
    return {
        "status": "validated",
        "eligible": eligible,
        "paths": {
            name: {
                "path": str(path),
                "sha256": _sha256(path),
            }
            for name, path in paths.items()
        },
        "preregistration_sha256": prereg_payload["preregistration_sha256"],
        "selection_binding_sha256": selection_payload["binding_sha256"],
        "selected_candidate": selected_candidate,
        "selected_config": selected_config,
        "selected": selected,
        "selection_status": selection_payload["status"],
        "guardrail_status": {
            "status": "passed" if eligible else "not_eligible",
            "n_eligible": len(eligible_rows) if isinstance(eligible_rows, list) else 0,
            "n_rejected": len(rejected) if isinstance(rejected, list) else 0,
        },
    }


def _require_copied_artifact_record(
    record: object,
    copied_path: str,
    run_dir: Path,
    row_idx: int,
    copied_idx: int,
) -> dict[str, object]:
    if not isinstance(record, dict):
        raise SystemExit(
            "Collected-runs manifest run field 'copied_artifacts' must contain "
            f"objects: row index {row_idx}, copied index {copied_idx}"
        )
    record_path = str(record.get("path", "")).strip()
    if record_path != copied_path:
        raise SystemExit(
            "Collected-runs manifest copied_artifacts path must match copied "
            f"path at row index {row_idx}, copied index {copied_idx}"
        )
    source_path = str(record.get("source_path", "")).strip()
    if not source_path:
        raise SystemExit(
            "Collected-runs manifest copied_artifacts source_path must be "
            f"non-empty at row index {row_idx}, copied index {copied_idx}"
        )
    source = Path(source_path)
    if not source.exists() or not source.is_file() or source.stat().st_size == 0:
        raise SystemExit(
            "Collected-runs manifest copied_artifacts source_path must exist "
            f"and be non-empty at row index {row_idx}, copied index {copied_idx}: "
            f"{source_path}"
        )
    try:
        source.resolve().relative_to(run_dir.resolve())
    except ValueError as exc:
        raise SystemExit(
            "Collected-runs manifest copied_artifacts source_path must be "
            f"inside run_dir at row index {row_idx}, copied index {copied_idx}: "
            f"{source_path}"
        ) from exc
    try:
        expected_source_bytes = _positive_int(
            record.get("source_bytes"),
            "Collected-runs manifest copied_artifacts source_bytes",
        )
    except SystemExit as exc:
        raise SystemExit(
            f"{exc}: row index {row_idx}, copied index {copied_idx}"
        ) from exc
    if expected_source_bytes != source.stat().st_size:
        raise SystemExit(
            "Collected-runs manifest copied_artifacts source_bytes do not "
            f"match existing source at row index {row_idx}, copied index "
            f"{copied_idx}: {expected_source_bytes} != {source.stat().st_size}"
        )
    expected_source_sha = str(record.get("source_sha256", "")).strip().lower()
    if len(expected_source_sha) != 64 or any(
        ch not in "0123456789abcdef" for ch in expected_source_sha
    ):
        raise SystemExit(
            "Collected-runs manifest copied_artifacts source_sha256 must be a "
            f"SHA-256 digest: row index {row_idx}, copied index {copied_idx}"
        )
    actual_source_sha = _sha256(source)
    if expected_source_sha != actual_source_sha:
        raise SystemExit(
            "Collected-runs manifest copied_artifacts source_sha256 does not "
            f"match existing source at row index {row_idx}, copied index "
            f"{copied_idx}"
        )
    path = Path(copied_path)
    size = path.stat().st_size
    try:
        expected_bytes = _positive_int(
            record.get("bytes"),
            "Collected-runs manifest copied_artifacts bytes",
        )
    except SystemExit as exc:
        raise SystemExit(
            f"{exc}: row index {row_idx}, copied index {copied_idx}"
        ) from exc
    if expected_bytes != size:
        raise SystemExit(
            "Collected-runs manifest copied_artifacts bytes do not match "
            f"existing file at row index {row_idx}, copied index {copied_idx}: "
            f"{expected_bytes} != {size}"
        )
    expected_sha = str(record.get("sha256", "")).strip().lower()
    if len(expected_sha) != 64 or any(
        ch not in "0123456789abcdef" for ch in expected_sha
    ):
        raise SystemExit(
            "Collected-runs manifest copied_artifacts sha256 must be a "
            f"SHA-256 digest: row index {row_idx}, copied index {copied_idx}"
        )
    actual_sha = _sha256(path)
    if expected_sha != actual_sha:
        raise SystemExit(
            "Collected-runs manifest copied_artifacts sha256 does not match "
            f"existing file at row index {row_idx}, copied index {copied_idx}"
        )
    return {
        "path": copied_path,
        "source_path": source_path,
        "source_bytes": source.stat().st_size,
        "source_sha256": actual_source_sha,
        "bytes": size,
        "sha256": actual_sha,
    }


def _input_runs(collected_runs_manifest: Path) -> dict[str, object]:
    payload = _read_json_object_if_present(
        collected_runs_manifest,
        "Collected-runs manifest",
    )
    if payload is None:
        return {
            "status": "missing",
            "manifest": str(collected_runs_manifest),
            "n_runs": 0,
            "runs": [],
        }
    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        raise SystemExit(
            "Collected-runs manifest field 'runs' must be a non-empty list: "
            f"{collected_runs_manifest}"
        )
    n_runs = payload.get("n_runs")
    if n_runs is not None:
        parsed_n_runs = _positive_int(n_runs, "Collected-runs manifest field 'n_runs'")
        if parsed_n_runs != len(runs):
            raise SystemExit(
                "Collected-runs manifest field 'n_runs' does not match runs "
                f"length: {parsed_n_runs} != {len(runs)}"
            )
    seen_run_ids: set[str] = set()
    seen_run_dirs: set[Path] = set()
    normalized_runs: list[dict[str, object]] = []
    for idx, run in enumerate(runs):
        if not isinstance(run, dict):
            raise SystemExit(
                "Collected-runs manifest field 'runs' must contain objects: "
                f"row index {idx}"
            )
        run_id = str(run.get("run_id", "")).strip()
        canonical_smiles = str(run.get("canonical_smiles", "")).strip()
        if not run_id:
            raise SystemExit(
                "Collected-runs manifest run is missing non-empty 'run_id': "
                f"row index {idx}"
            )
        if run_id in seen_run_ids:
            raise SystemExit(
                "Collected-runs manifest contains duplicate run_id values: "
                f"{run_id}"
            )
        seen_run_ids.add(run_id)
        if not canonical_smiles:
            raise SystemExit(
                "Collected-runs manifest run is missing non-empty "
                f"'canonical_smiles': row index {idx}"
            )
        canonical_smiles = _require_valid_smiles(
            canonical_smiles,
            f"Collected-runs manifest run canonical_smiles at row index {idx}",
        )
        run_dir_text = str(run.get("run_dir", "")).strip()
        if not run_dir_text:
            raise SystemExit(
                "Collected-runs manifest run is missing non-empty 'run_dir': "
                f"row index {idx}"
            )
        run_dir = Path(run_dir_text)
        if not run_dir.exists() or not run_dir.is_dir():
            raise SystemExit(
                "Collected-runs manifest run_dir must exist and be a directory: "
                f"{run_dir_text}"
            )
        resolved_run_dir = run_dir.expanduser().resolve(strict=False)
        if resolved_run_dir in seen_run_dirs:
            raise SystemExit(
                "Collected-runs manifest contains duplicate run_dir values: "
                f"{resolved_run_dir}"
            )
        seen_run_dirs.add(resolved_run_dir)
        copied = run.get("copied", [])
        if not isinstance(copied, list):
            raise SystemExit(
                "Collected-runs manifest run field 'copied' must be a list: "
                f"row index {idx}"
            )
        copied_artifacts = run.get("copied_artifacts")
        if not isinstance(copied_artifacts, list) or len(copied_artifacts) != len(copied):
            raise SystemExit(
                "Collected-runs manifest run field 'copied_artifacts' must be "
                "a list matching copied length: row index "
                f"{idx}"
            )
        copied_paths: list[str] = []
        copied_records: list[dict[str, object]] = []
        seen_copied_paths: set[Path] = set()
        for copied_idx, copied_value in enumerate(copied):
            copied_path = str(copied_value).strip()
            if not copied_path:
                raise SystemExit(
                    "Collected-runs manifest run field 'copied' contains a "
                    f"blank path at row index {idx}, copied index {copied_idx}"
                )
            resolved_copied_path = Path(copied_path).expanduser().resolve(strict=False)
            if resolved_copied_path in seen_copied_paths:
                raise SystemExit(
                    "Collected-runs manifest run field 'copied' contains "
                    "duplicate paths: row index "
                    f"{idx}, copied index {copied_idx}: {resolved_copied_path}"
                )
            seen_copied_paths.add(resolved_copied_path)
            if not Path(copied_path).exists():
                raise SystemExit(
                    "Collected-runs manifest copied artifact is missing: "
                    f"{copied_path}"
                )
            copied_paths.append(copied_path)
            copied_records.append(
                _require_copied_artifact_record(
                    copied_artifacts[copied_idx],
                    copied_path,
                    run_dir,
                    idx,
                    copied_idx,
                )
            )
        normalized_runs.append({
            "run_id": run_id,
            "run_dir": run_dir_text,
            "canonical_smiles": canonical_smiles,
            "ranked_v3": str(run.get("ranked_v3", "")).strip(),
            "n_leakage_rows": len(run.get("leakage_rows", []))
            if isinstance(run.get("leakage_rows"), list)
            else 0,
            "copied": copied_paths,
            "copied_artifacts": copied_records,
        })
    return {
        "status": "present",
        "manifest": str(collected_runs_manifest),
        "out_dir": str(payload.get("out_dir", "")).strip(),
        "rankings_dir": str(payload.get("rankings_dir", "")).strip(),
        "n_runs": len(normalized_runs),
        "runs": normalized_runs,
    }


def _input_runs_claim_blocker(
    input_runs: dict[str, object],
    eval_dir: Path,
    active_rankings_dir: Path,
) -> str | None:
    if input_runs.get("status") != "present":
        return (
            f"status={input_runs.get('status')}, "
            f"n_runs={input_runs.get('n_runs')}"
        )
    if int(input_runs.get("n_runs", 0)) <= 0:
        return "n_runs must be > 0"
    runs = input_runs.get("runs")
    if not isinstance(runs, list) or not runs:
        return "runs must be a non-empty list"
    for idx, run in enumerate(runs):
        if not isinstance(run, dict):
            return f"runs[{idx}] must be an object"
        copied = run.get("copied")
        copied_artifacts = run.get("copied_artifacts")
        if not isinstance(copied, list) or not copied:
            return f"runs[{idx}].copied must be a non-empty list"
        if not isinstance(copied_artifacts, list) or len(copied_artifacts) != len(copied):
            return (
                f"runs[{idx}].copied_artifacts must match copied length "
                "and be non-empty"
            )
        if int(run.get("n_leakage_rows", 0)) <= 0:
            return f"runs[{idx}].n_leakage_rows must be > 0"
    out_dir_text = str(input_runs.get("out_dir", "")).strip()
    if not out_dir_text:
        return "out_dir must be recorded in collected run provenance"
    out_dir = Path(out_dir_text)
    if not out_dir.exists() or not out_dir.is_dir():
        return f"out_dir must exist and be a directory: {out_dir_text}"
    if out_dir.resolve() != eval_dir.resolve():
        return (
            "out_dir must match active --eval-dir: "
            f"{out_dir_text} != {eval_dir}"
        )
    rankings_dir_text = str(input_runs.get("rankings_dir", "")).strip()
    if not rankings_dir_text:
        return "rankings_dir must be recorded in collected run provenance"
    manifest_rankings_dir = Path(rankings_dir_text)
    if not manifest_rankings_dir.exists() or not manifest_rankings_dir.is_dir():
        return f"rankings_dir must exist and be a directory: {rankings_dir_text}"
    if manifest_rankings_dir.resolve() != active_rankings_dir.resolve():
        return (
            "rankings_dir must match active --rankings-dir: "
            f"{rankings_dir_text} != {active_rankings_dir}"
        )
    rankings_root = manifest_rankings_dir.resolve()
    for idx, run in enumerate(runs):
        copied = run.get("copied")
        if not isinstance(copied, list):
            return f"runs[{idx}].copied must be a non-empty list"
        for copied_idx, copied_path in enumerate(copied):
            try:
                Path(str(copied_path)).resolve().relative_to(rankings_root)
            except ValueError:
                return (
                    f"runs[{idx}].copied[{copied_idx}] must be inside "
                    f"rankings_dir: {copied_path}"
                )
    return None


def _sha256_text_is_valid(value: object, label: str) -> str:
    text = str(value).strip().lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise SystemExit(f"{label} must be a SHA-256 digest")
    return text


def _canonical_smiles(value: object, label: str) -> str:
    smiles = _require_valid_smiles(str(value), label)
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise SystemExit(f"{label} must be a parseable SMILES: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True)


def _read_skin_known_cases_for_ledger(path: Path) -> pd.DataFrame:
    df = _read_csv_if_present(path)
    if df is None or df.empty:
        raise SystemExit(f"Skin known-target panel is required for SOTA ledger: {path}")
    required = ("case_id", "smiles")
    _require_metric_columns(df, required, "skin known-target panel")
    case_ids = _required_text_column(df, "case_id", "skin known-target panel")
    if case_ids.duplicated().any():
        duplicate = case_ids[case_ids.duplicated()].iloc[0]
        raise SystemExit(
            f"skin known-target panel contains duplicate case_id values: {duplicate}"
        )
    _required_text_column(df, "smiles", "skin known-target panel")
    df = df.copy()
    df["case_id"] = case_ids
    return df


def _skin_known_run_ledger(
    ledger_path: Path,
    cases_csv: Path,
    rankings_dir: Path,
    input_runs: dict[str, object],
) -> dict[str, object]:
    if not ledger_path.exists():
        return {
            "status": "missing",
            "path": str(ledger_path),
            "n_rows": 0,
            "n_cases": 0,
        }
    if ledger_path.stat().st_size == 0:
        raise SystemExit(f"Skin known-target run ledger is empty: {ledger_path}")
    try:
        ledger = pd.read_csv(ledger_path)
    except Exception as exc:
        raise SystemExit(
            f"Skin known-target run ledger failed to parse: {ledger_path}: {exc}"
        ) from exc
    required = (
        "case_id",
        "smiles",
        "run_id",
        "command",
        "source_run_dir",
        "copied_ranking_path",
        "config_sha256",
        "run_manifest_sha256",
    )
    _require_metric_columns(ledger, required, "skin known-target run ledger")
    if ledger.empty:
        raise SystemExit(f"Skin known-target run ledger contains no rows: {ledger_path}")
    for column in required:
        _required_text_column(ledger, column, "skin known-target run ledger")

    cases = _read_skin_known_cases_for_ledger(cases_csv)
    case_smiles = {
        str(row["case_id"]): _canonical_smiles(
            row["smiles"],
            f"skin known-target panel smiles for {row['case_id']}",
        )
        for _, row in cases.iterrows()
    }
    ledger_case_ids = ledger["case_id"].astype(str).str.strip()
    duplicate_cases = ledger_case_ids[ledger_case_ids.duplicated()].tolist()
    if duplicate_cases:
        shown = ", ".join(duplicate_cases[:10])
        raise SystemExit(
            f"Skin known-target run ledger contains duplicate case_id values: {shown}"
        )
    expected_cases = set(case_smiles)
    observed_cases = set(ledger_case_ids)
    missing_cases = sorted(expected_cases - observed_cases)
    extra_cases = sorted(observed_cases - expected_cases)
    if missing_cases or extra_cases:
        detail = []
        if missing_cases:
            detail.append(f"missing={missing_cases[:10]}")
        if extra_cases:
            detail.append(f"extra={extra_cases[:10]}")
        raise SystemExit(
            "Skin known-target run ledger must cover exactly the panel case_id "
            f"set: {'; '.join(detail)}"
        )

    runs = input_runs.get("runs")
    if input_runs.get("status") != "present" or not isinstance(runs, list):
        return {
            "status": "input_runs_not_present",
            "path": str(ledger_path),
            "n_rows": int(len(ledger)),
            "n_cases": len(expected_cases),
        }
    runs_by_id = {
        str(run.get("run_id", "")).strip(): run
        for run in runs
        if isinstance(run, dict)
    }
    copied_by_path: dict[Path, dict[str, object]] = {}
    for run in runs:
        if not isinstance(run, dict):
            continue
        for record in run.get("copied_artifacts", []):
            if not isinstance(record, dict):
                continue
            copied_by_path[Path(str(record.get("path", ""))).resolve()] = record

    rankings_root = rankings_dir.resolve()
    normalized_rows: list[dict[str, object]] = []
    for idx, row in ledger.iterrows():
        case_id = str(row["case_id"]).strip()
        expected_smiles = case_smiles[case_id]
        observed_smiles = _canonical_smiles(
            row["smiles"],
            f"skin known-target run ledger smiles for {case_id}",
        )
        if observed_smiles != expected_smiles:
            raise SystemExit(
                "Skin known-target run ledger smiles must match panel "
                f"canonical SMILES for {case_id}: {observed_smiles} != {expected_smiles}"
            )
        for col in ("config_sha256", "run_manifest_sha256"):
            _sha256_text_is_valid(
                row[col],
                f"Skin known-target run ledger column '{col}' row index {idx}",
            )
        run_id = str(row["run_id"]).strip()
        if run_id not in runs_by_id:
            raise SystemExit(
                "Skin known-target run ledger run_id must exist in collected "
                f"run provenance: {run_id}"
            )
        run_record = runs_by_id[run_id]
        source_run_dir = Path(str(row["source_run_dir"]).strip())
        if not source_run_dir.exists() or not source_run_dir.is_dir():
            raise SystemExit(
                "Skin known-target run ledger source_run_dir must exist and "
                f"be a directory: {source_run_dir}"
            )
        if source_run_dir.resolve() != Path(str(run_record.get("run_dir"))).resolve():
            raise SystemExit(
                "Skin known-target run ledger source_run_dir must match "
                f"collected run provenance for {run_id}"
            )
        copied_path = Path(str(row["copied_ranking_path"]).strip())
        if not copied_path.exists() or not copied_path.is_file() or copied_path.stat().st_size == 0:
            raise SystemExit(
                "Skin known-target run ledger copied_ranking_path must exist "
                f"and be non-empty: {copied_path}"
            )
        try:
            copied_path.resolve().relative_to(rankings_root)
        except ValueError as exc:
            raise SystemExit(
                "Skin known-target run ledger copied_ranking_path must be "
                f"inside skin-known rankings dir: {copied_path}"
            ) from exc
        expected_names = {
            f"{case_id}__ranked_targets_v3.csv",
            f"{case_id}__ranked_targets_v3_with_efficacy.csv",
        }
        if copied_path.name not in expected_names:
            raise SystemExit(
                "Skin known-target run ledger copied_ranking_path filename "
                f"must match case_id {case_id}: {copied_path.name}"
            )
        try:
            copied_ranking = pd.read_csv(copied_path)
        except Exception as exc:
            raise SystemExit(
                f"Skin known-target copied ranking failed to parse: {copied_path}: {exc}"
            ) from exc
        prior_columns = ("known_target_prior", "known_target_prior_norm")
        missing_prior_columns = [
            column for column in prior_columns if column not in copied_ranking.columns
        ]
        if missing_prior_columns:
            raise SystemExit(
                "Skin known-target copied ranking must retain prior audit "
                f"columns {list(prior_columns)}: missing={missing_prior_columns}: "
                f"{copied_path}"
            )
        for prior_column in prior_columns:
            prior_values = pd.to_numeric(
                copied_ranking[prior_column],
                errors="coerce",
            )
            invalid = prior_values.isna() | ~prior_values.map(math.isfinite)
            if invalid.any():
                first = int(invalid[invalid].index[0])
                raise SystemExit(
                    "Skin known-target copied ranking prior audit column "
                    f"'{prior_column}' must be finite at row index {first}: "
                    f"{copied_path}"
                )
            if prior_values.ne(0.0).any():
                raise SystemExit(
                    "Skin known-target copied ranking must have zero known-target "
                    f"prior contribution for an unbiased SOTA claim: "
                    f"column={prior_column}: {copied_path}"
                )
        copied_record = copied_by_path.get(copied_path.resolve())
        if copied_record is None:
            raise SystemExit(
                "Skin known-target run ledger copied_ranking_path must be "
                f"in collected run copied_artifacts: {copied_path}"
            )
        expected_sha = str(copied_record.get("sha256", "")).strip().lower()
        actual_sha = _sha256(copied_path)
        if expected_sha != actual_sha:
            raise SystemExit(
                "Skin known-target run ledger copied_ranking_path sha256 "
                f"does not match collected run artifact: {copied_path}"
            )
        manifest_path = str(row.get("run_manifest_path", "")).strip()
        if manifest_path:
            manifest = Path(manifest_path)
            if not manifest.exists() or not manifest.is_file() or manifest.stat().st_size == 0:
                raise SystemExit(
                    "Skin known-target run ledger run_manifest_path must "
                    f"exist and be non-empty: {manifest}"
                )
            expected_manifest_sha = _sha256_text_is_valid(
                row["run_manifest_sha256"],
                f"Skin known-target run ledger run_manifest_sha256 row index {idx}",
            )
            if _sha256(manifest) != expected_manifest_sha:
                raise SystemExit(
                    "Skin known-target run ledger run_manifest_sha256 does "
                    f"not match run_manifest_path for {case_id}"
                )
        normalized_rows.append({
            "case_id": case_id,
            "run_id": run_id,
            "copied_ranking_path": str(copied_path),
            "source_run_dir": str(source_run_dir),
            "known_target_prior_audit": "zero",
        })
    return {
        "status": "ok",
        "path": str(ledger_path),
        "n_rows": int(len(ledger)),
        "n_cases": len(expected_cases),
        "cases": normalized_rows,
    }


REQUIRED_BASELINE_TYPES = {
    "nearest_neighbor",
    "dti_only",
    "docking_structure",
    "full_skinscout",
}
REQUIRED_ABLATION_FAMILIES = {
    "dti_only",
    "docking_only",
    "skin_expression_only",
    "structure_model_only",
    "full_ensemble",
}


def _float_equals(value: object, expected: float, label: str) -> None:
    parsed = _finite_float(value, label)
    if not math.isclose(parsed, expected, rel_tol=1e-9, abs_tol=1e-12):
        raise SystemExit(f"{label} must equal active evaluation value {expected}: {parsed}")


def _sota_baseline_fairness(
    path: Path,
    cases_csv: Path,
    seq_id_threshold: float,
    ligand_tanimoto_threshold: float,
    pocket_sucos_threshold: float,
) -> dict[str, object]:
    if not path.exists():
        return {"status": "missing", "path": str(path), "n_rows": 0}
    df = _read_csv_if_present(path)
    if df is None or df.empty:
        raise SystemExit(f"SOTA baseline fairness table is required and non-empty: {path}")
    required = (
        "baseline_id",
        "baseline_type",
        "status",
        "panel_sha256",
        "target_universe",
        "uniprot_mapping",
        "seq_id_threshold",
        "ligand_tanimoto_threshold",
        "pocket_sucos_threshold",
        "input_evidence_status",
        "training_data_status",
        "license_status",
    )
    _require_metric_columns(df, required, "SOTA baseline fairness table")
    for col in required:
        _required_text_column(df, col, "SOTA baseline fairness table")
    if df["baseline_id"].astype(str).str.strip().duplicated().any():
        raise SystemExit("SOTA baseline fairness table contains duplicate baseline_id values")
    statuses = df["status"].astype(str).str.strip()
    invalid_status = sorted(set(statuses) - {"included", "excluded", "blocked"})
    if invalid_status:
        raise SystemExit(
            f"SOTA baseline fairness table contains invalid status values {invalid_status}"
        )
    included = df[statuses == "included"].copy()
    included_types = set(included["baseline_type"].astype(str).str.strip())
    missing_required = sorted(REQUIRED_BASELINE_TYPES - included_types)
    if missing_required:
        raise SystemExit(
            "SOTA baseline fairness table missing included required baseline_type "
            f"value(s): {missing_required}"
        )
    panel_sha = _sha256(cases_csv)
    target_universes = set(included["target_universe"].astype(str).str.strip())
    mappings = set(included["uniprot_mapping"].astype(str).str.strip())
    if len(target_universes) != 1:
        raise SystemExit("SOTA baseline fairness included rows must use one target_universe")
    if len(mappings) != 1:
        raise SystemExit("SOTA baseline fairness included rows must use one uniprot_mapping")
    for idx, row in included.iterrows():
        if str(row["panel_sha256"]).strip().lower() != panel_sha:
            raise SystemExit(
                "SOTA baseline fairness panel_sha256 must match active "
                f"skin-known panel at row index {idx}"
            )
        _float_equals(
            row["seq_id_threshold"],
            seq_id_threshold,
            f"SOTA baseline fairness seq_id_threshold row index {idx}",
        )
        _float_equals(
            row["ligand_tanimoto_threshold"],
            ligand_tanimoto_threshold,
            f"SOTA baseline fairness ligand_tanimoto_threshold row index {idx}",
        )
        _float_equals(
            row["pocket_sucos_threshold"],
            pocket_sucos_threshold,
            f"SOTA baseline fairness pocket_sucos_threshold row index {idx}",
        )
        if str(row["input_evidence_status"]).strip() not in {"same", "equivalent"}:
            raise SystemExit(
                "SOTA baseline fairness included rows require same or "
                f"equivalent input_evidence_status at row index {idx}"
            )
        if str(row["training_data_status"]).strip() in {
            "trained_on_panel",
            "undisclosed",
            "unknown",
        }:
            raise SystemExit(
                "SOTA baseline fairness included rows must not be trained on "
                f"the panel or undisclosed at row index {idx}"
            )
        if str(row["license_status"]).strip() != "compatible":
            raise SystemExit(
                "SOTA baseline fairness included rows require compatible "
                f"license_status at row index {idx}"
            )
    return {
        "status": "ok",
        "path": str(path),
        "n_rows": int(len(df)),
        "included_baseline_types": sorted(included_types),
        "target_universe": next(iter(target_universes)),
        "uniprot_mapping": next(iter(mappings)),
    }


def _sota_ablation_freeze(path: Path, cases_csv: Path) -> dict[str, object]:
    if not path.exists():
        return {"status": "missing", "path": str(path), "n_rows": 0}
    df = _read_csv_if_present(path)
    if df is None or df.empty:
        raise SystemExit(f"SOTA ablation freeze table is required and non-empty: {path}")
    required = (
        "ablation_id",
        "ablation_family",
        "status",
        "panel_sha256",
        "target_universe",
        "case_top10",
        "target_top10",
        "target_top30",
    )
    _require_metric_columns(df, required, "SOTA ablation freeze table")
    for col in ("ablation_id", "ablation_family", "status", "panel_sha256", "target_universe"):
        _required_text_column(df, col, "SOTA ablation freeze table")
    if df["ablation_id"].astype(str).str.strip().duplicated().any():
        raise SystemExit("SOTA ablation freeze table contains duplicate ablation_id values")
    statuses = df["status"].astype(str).str.strip()
    invalid_status = sorted(set(statuses) - {"included", "excluded", "blocked"})
    if invalid_status:
        raise SystemExit(
            f"SOTA ablation freeze table contains invalid status values {invalid_status}"
        )
    included = df[statuses == "included"].copy()
    included_families = set(included["ablation_family"].astype(str).str.strip())
    missing_required = sorted(REQUIRED_ABLATION_FAMILIES - included_families)
    if missing_required:
        raise SystemExit(
            "SOTA ablation freeze table missing included required ablation_family "
            f"value(s): {missing_required}"
        )
    panel_sha = _sha256(cases_csv)
    target_universes = set(included["target_universe"].astype(str).str.strip())
    if len(target_universes) != 1:
        raise SystemExit("SOTA ablation freeze included rows must use one target_universe")
    for idx, row in included.iterrows():
        if str(row["panel_sha256"]).strip().lower() != panel_sha:
            raise SystemExit(
                "SOTA ablation freeze panel_sha256 must match active "
                f"skin-known panel at row index {idx}"
            )
        for col in ("case_top10", "target_top10", "target_top30"):
            _fraction_float(row[col], f"SOTA ablation freeze {col} row index {idx}")
    return {
        "status": "ok",
        "path": str(path),
        "n_rows": int(len(df)),
        "included_ablation_families": sorted(included_families),
        "target_universe": next(iter(target_universes)),
    }


def _sota_freeze_claim_blocker(
    sota_active: bool,
    sota_freeze: dict[str, object],
) -> str | None:
    if not sota_active:
        return None
    required = (
        ("skin_known_run_ledger", "skin known-target run ledger"),
        ("baseline_fairness", "SOTA baseline fairness table"),
        ("ablation_freeze", "SOTA ablation freeze table"),
    )
    blockers = []
    for key, label in required:
        record = sota_freeze.get(key)
        if not isinstance(record, dict) or record.get("status") != "ok":
            status = record.get("status") if isinstance(record, dict) else "missing"
            blockers.append(f"{label} status={status}")
    return "; ".join(blockers) if blockers else None


def _diagnostic_overrides(args: argparse.Namespace) -> dict[str, bool]:
    return {
        "allow_incomplete_eval_inputs": bool(args.allow_incomplete_eval_inputs),
        "allow_empty_iteration": bool(args.allow_empty_iteration),
        "allow_incomplete_leakage": bool(args.allow_incomplete_leakage),
        "allow_threshold_failure": bool(args.allow_threshold_failure),
        "allow_missing_input_runs": bool(args.allow_missing_input_runs),
    }


def _claim_blockers(
    manifest: dict[str, object],
    input_status: dict[str, object],
    leakage_status: dict[str, object],
    threshold_status: dict[str, object],
    activity_retrieval_gate: dict[str, object],
    input_runs_claim_blocker: str | None,
    sota_freeze_claim_blocker: str | None,
) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    n_steps = int(manifest.get("n_steps", 0))
    if int(input_status.get("n_incomplete", 0)):
        blockers.append({
            "code": "incomplete_eval_inputs",
            "detail": input_status.get("incomplete", []),
        })
    if n_steps == 0:
        blockers.append({"code": "no_runnable_eval_steps"})
    if int(manifest.get("n_failed", 0)):
        blockers.append({
            "code": "failed_eval_steps",
            "detail": [
                step
                for step in manifest.get("steps", [])
                if isinstance(step, dict) and step.get("status") == "failed"
            ],
        })
    if int(threshold_status.get("n_failed", 0)):
        blockers.append({
            "code": "threshold_failures",
            "detail": [
                *threshold_status.get("failures", []),
                *threshold_status.get("missing_required", []),
            ],
        })
    if n_steps > 0 and int(leakage_status.get("n_leak_flags", 0)):
        blockers.append({
            "code": "leakage_flags",
            "detail": {
                "status": leakage_status.get("status"),
                "n_leak_flags": leakage_status.get("n_leak_flags"),
            },
        })
    if n_steps > 0 and not _leakage_claim_ready(leakage_status):
        blockers.append({
            "code": "leakage_not_claim_ready",
            "detail": {
                "status": leakage_status.get("status"),
                "n_incomplete": leakage_status.get("n_incomplete"),
                "n_invalid_rows": leakage_status.get("n_invalid_rows"),
                "sealed_audit": leakage_status.get("sealed_audit"),
            },
        })
    if n_steps > 0 and int(threshold_status.get("n_checked", 0)) == 0:
        blockers.append({"code": "no_thresholds_checked"})
    if n_steps > 0 and input_runs_claim_blocker is not None:
        blockers.append({
            "code": "input_runs_not_claim_ready",
            "detail": input_runs_claim_blocker,
        })
    if n_steps > 0 and sota_freeze_claim_blocker is not None:
        blockers.append({
            "code": "sota_freeze_not_claim_ready",
            "detail": sota_freeze_claim_blocker,
        })
    if (
        n_steps > 0
        and not blockers
        and activity_retrieval_gate.get("status") != "pass"
    ):
        blockers.append({
            "code": "activity_retrieval_gate_not_ready",
            "detail": activity_retrieval_gate,
        })
    return blockers


def _sealed_leakage_status(
    path: Path,
    *,
    expected_sources: dict[str, Path],
    expected_thresholds: LeakageThresholds,
    expected_execution_challenge: str,
    direct_exact_manifest: Path | None = None,
) -> dict[str, object]:
    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        return {
            "status": "missing",
            "path": str(path),
            "direct_exact_count": 0,
            "incomplete_audit_count": 0,
            "neutral_exclusion_count": 0,
            "n_survivors": 0,
        }
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "invalid",
            "path": str(path),
            "error": f"failed to parse: {exc}",
            "direct_exact_count": 0,
            "incomplete_audit_count": 0,
            "neutral_exclusion_count": 0,
            "n_survivors": 0,
        }
    if not isinstance(payload, dict):
        return {
            "status": "invalid",
            "path": str(path),
            "error": "payload must be a JSON object",
            "direct_exact_count": 0,
            "incomplete_audit_count": 0,
            "neutral_exclusion_count": 0,
            "n_survivors": 0,
        }
    try:
        validate_sealed_discovery_audit_sources(
            payload,
            expected_sources=expected_sources,
            expected_thresholds=expected_thresholds,
            expected_execution_challenge=expected_execution_challenge,
            require_direct_exact_reference=True,
            direct_exact_manifest=direct_exact_manifest,
        )
    except (RuntimeError, ValueError) as exc:
        return {
            "status": "invalid",
            "path": str(path),
            "error": str(exc),
            "direct_exact_count": 0,
            "incomplete_audit_count": 0,
            "neutral_exclusion_count": 0,
            "n_survivors": 0,
        }
    counts = payload["counts"]
    assert isinstance(counts, dict)
    return {
        "status": str(payload["status"]),
        "path": str(path),
        "schema_version": str(payload["schema_version"]),
        "binding_sha256": str(payload["binding_sha256"]),
        "execution_challenge": str(payload["execution_challenge"]),
        "direct_exact_count": int(payload["direct_exact_count"]),
        "incomplete_audit_count": int(payload["incomplete_audit_count"]),
        "neutral_exclusion_count": int(payload["neutral_exclusion_count"]),
        "n_survivors": int(counts["survivor_rows"]),
    }


def _leakage_status(
    eval_dir: Path,
    sealed_audit: dict[str, object] | None = None,
    thresholds: LeakageThresholds | None = None,
) -> dict[str, object]:
    thresholds = thresholds or LeakageThresholds(0.30, 0.50, 0.50)
    sealed = sealed_audit or {
        "status": "not_run",
        "path": str(eval_dir / "discovery_leakage_audit.json"),
        "direct_exact_count": 0,
        "incomplete_audit_count": 0,
        "neutral_exclusion_count": 0,
        "n_survivors": 0,
    }
    df = _read_csv_if_present(eval_dir / "leakage_audit.csv")
    if df is None:
        return {
            "status": "not_run",
            "n_rows": 0,
            "n_incomplete": 0,
            "n_leak_flags": 0,
            "n_invalid_rows": 0,
            "sealed_audit": sealed,
        }
    required = {
        "audit_status",
        "leak_flag",
        "missing_axes",
        "uniprot",
        "smiles",
        "seq_id",
        "ligand_tanimoto",
        "pocket_sucos",
    }
    missing = sorted(required - set(df.columns))
    if df.empty or missing:
        return {
            "status": "invalid",
            "n_rows": int(len(df)),
            "n_incomplete": 0,
            "n_leak_flags": 0,
            "n_invalid_rows": int(len(df)) if df.empty else 0,
            "missing_columns": missing,
            "sealed_audit": sealed,
        }

    statuses = df["audit_status"].astype(str).str.strip().str.lower()
    valid_status = statuses.isin({"ok", "incomplete"})
    leak_values = [_as_bool(value) for value in df["leak_flag"].tolist()]
    invalid_leak = [value is None for value in leak_values]
    score_mismatch: list[bool] = []
    missing_axes_mismatch: list[bool] = []
    status_mismatch: list[bool] = []
    leak_mismatch: list[bool] = []
    for pos, (_, row) in enumerate(df.iterrows()):
        parsed_scores = {
            "seq_id": _as_unit_score_or_none(row["seq_id"]),
            "ligand_tanimoto": _as_unit_score_or_none(row["ligand_tanimoto"]),
            "pocket_sucos": _as_unit_score_or_none(row["pocket_sucos"]),
        }
        score_mismatch.append(any(value == "invalid" for value in parsed_scores.values()))
        if score_mismatch[-1]:
            missing_axes_mismatch.append(True)
            status_mismatch.append(True)
            leak_mismatch.append(True)
            continue
        expected_axes = [
            axis
            for axis, score_name in (
                ("sequence", "seq_id"),
                ("ligand", "ligand_tanimoto"),
                ("pocket", "pocket_sucos"),
            )
            if parsed_scores[score_name] is None
        ]
        observed_axes = _parse_missing_axes(row["missing_axes"])
        missing_axes_mismatch.append(observed_axes != expected_axes)
        expected_status = "incomplete" if expected_axes else "ok"
        status_mismatch.append(statuses.iloc[pos] != expected_status)
        expected_leak = (
            (
                parsed_scores["seq_id"] is not None
                and parsed_scores["seq_id"] >= thresholds.seq_id
            )
            or (
                parsed_scores["ligand_tanimoto"] is not None
                and parsed_scores["ligand_tanimoto"] >= thresholds.ligand_tanimoto
            )
            or (
                parsed_scores["pocket_sucos"] is not None
                and parsed_scores["pocket_sucos"] >= thresholds.pocket_sucos
            )
        )
        leak_mismatch.append(leak_values[pos] != expected_leak)
    blank_required = (
        df[["uniprot", "smiles"]]
        .isna()
        .any(axis=1)
        | (df["uniprot"].astype(str).str.strip() == "")
        | (df["smiles"].astype(str).str.strip() == "")
    )
    invalid_rows = int(
        (
            ~valid_status
            | pd.Series(invalid_leak, index=df.index)
            | pd.Series(score_mismatch, index=df.index)
            | pd.Series(missing_axes_mismatch, index=df.index)
            | pd.Series(status_mismatch, index=df.index)
            | pd.Series(leak_mismatch, index=df.index)
            | blank_required
        ).sum()
    )
    incomplete = int((statuses == "incomplete").sum())
    leak_flags = int(sum(value is True for value in leak_values))
    if invalid_rows:
        status = "invalid"
    elif leak_flags:
        status = "leak_detected"
    elif incomplete:
        status = "incomplete"
    else:
        status = "ok"
    return {
        "status": status,
        "n_rows": int(len(df)),
        "n_incomplete": incomplete,
        "n_leak_flags": leak_flags,
        "n_invalid_rows": invalid_rows,
        "sealed_audit": sealed,
    }


def _leakage_claim_ready(leakage_status: dict[str, object]) -> bool:
    sealed = leakage_status.get("sealed_audit")
    return (
        leakage_status.get("status") == "ok"
        and leakage_status.get("n_leak_flags") == 0
        and isinstance(sealed, dict)
        and sealed.get("status") == "ok"
        and sealed.get("direct_exact_count") == 0
        and sealed.get("incomplete_audit_count") == 0
        and int(sealed.get("n_survivors", 0)) > 0
    )


def _as_bool(value: object) -> bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def _as_unit_score_or_none(value: object) -> float | str | None:
    if pd.isna(value) or str(value).strip() == "":
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return "invalid"
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        return "invalid"
    return score


def _parse_missing_axes(value: object) -> list[str] | str:
    if pd.isna(value):
        text = ""
    else:
        text = str(value).strip()
    axes = [axis.strip() for axis in text.split(";") if axis.strip()]
    if any(axis not in {"sequence", "ligand", "pocket"} for axis in axes):
        return "invalid"
    if len(set(axes)) != len(axes):
        return "invalid"
    return axes


def _threshold_values(values: list[object]) -> tuple[list[bool], int]:
    parsed = [_as_bool(value) for value in values]
    invalid = sum(value is None for value in parsed)
    return [bool(value) if value is not None else False for value in parsed], invalid


def _threshold_metric(value: object) -> dict[str, bool]:
    parsed = _as_bool(value)
    return {
        "passes_threshold": parsed is True,
        "passes_threshold_valid": parsed is not None,
    }


THRESHOLDED_METRIC_FILES = {
    "cold_start_comprehensive.csv",
    "cold_start_fast.csv",
    "cold_start_dti_only.csv",
    "cosmetic_retrospective.csv",
    "skin_known_target_recovery.csv",
    "skin_efficacy_recovery.csv",
    "analog_quality.csv",
    "pharmacophore_conservation.csv",
}
SKIN_KNOWN_CONTEXT_PROFILES = {
    "auto",
    "general_skin",
    "pigmentation",
    "anti_aging",
    "barrier",
    "acne",
    "inflammation",
    "irritation_sensitization",
}


def _require_metric_columns(df: pd.DataFrame,
                            required: tuple[str, ...],
                            label: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise SystemExit(f"{label} missing required metric columns {missing}")


def _require_single_metric_row(df: pd.DataFrame, label: str) -> None:
    if len(df) != 1:
        raise SystemExit(f"{label} must contain exactly one metric row")


def _metric_paths(metric_source: Path | set[Path]) -> set[Path]:
    if isinstance(metric_source, Path):
        return {
            path.resolve()
            for path in metric_source.glob("*.csv")
            if path.is_file() and path.stat().st_size > 0
        }
    return {
        path.resolve()
        for path in metric_source
        if path.suffix == ".csv" and path.is_file() and path.stat().st_size > 0
    }


def _threshold_status(metric_source: Path | set[Path]) -> dict[str, object]:
    metric_paths = _metric_paths(metric_source)
    checked: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    missing_required: list[dict[str, object]] = []
    for path in sorted(metric_paths):
        df = _read_csv_if_present(path)
        if df is None or df.empty:
            continue
        if "passes_threshold" not in df.columns:
            if path.name in THRESHOLDED_METRIC_FILES:
                missing_required.append({
                    "path": str(path),
                    "n_rows": int(len(df)),
                })
            continue
        values, n_invalid = _threshold_values(df["passes_threshold"].tolist())
        passed = all(values)
        record = {
            "path": str(path),
            "n_rows": int(len(df)),
            "n_failed_rows": int(sum(1 for v in values if not v)),
            "n_invalid_rows": int(n_invalid),
        }
        checked.append({**record, "passed": passed})
        if not passed:
            failures.append(record)
    failed = bool(failures or missing_required)
    return {
        "status": "failed" if failed else "ok",
        "n_checked": len(checked),
        "n_failed": len(failures) + len(missing_required),
        "n_missing_required": len(missing_required),
        "checked": checked,
        "failures": failures,
        "missing_required": missing_required,
    }


def _finite_float(value: object, label: str) -> float:
    if pd.isna(value):
        raise SystemExit(f"{label} is missing")
    if (
        isinstance(value, bool)
        or type(value).__name__ == "bool_"
        or (isinstance(value, str) and value.strip().lower() in {"true", "false"})
    ):
        raise SystemExit(f"{label} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{label} must be numeric") from exc
    if not math.isfinite(parsed):
        raise SystemExit(f"{label} must be finite")
    return parsed


def _fraction_float(value: object, label: str) -> float:
    parsed = _finite_float(value, label)
    if parsed < 0 or parsed > 1:
        raise SystemExit(f"{label} must be in [0, 1]")
    return parsed


def _fraction_column_mean(df: pd.DataFrame, col: str, label: str) -> float:
    bool_indexes = [
        int(idx)
        for idx, value in df[col].items()
        if (
            isinstance(value, bool)
            or type(value).__name__ == "bool_"
            or (isinstance(value, str) and value.strip().lower() in {"true", "false"})
        )
    ]
    if bool_indexes:
        raise SystemExit(
            f"{label} must be numeric at row index(es) {bool_indexes[:5]}"
        )
    values = pd.to_numeric(df[col], errors="coerce")
    missing = values[values.isna()].index.tolist()
    if missing:
        raise SystemExit(
            f"{label} must be numeric at row index(es) {missing[:5]}"
        )
    out_of_range = [
        idx
        for idx, value in values.items()
        if (
            not math.isfinite(float(value))
            or float(value) < 0
            or float(value) > 1
        )
    ]
    if out_of_range:
        raise SystemExit(
            f"{label} must be in [0, 1] at row index(es) {out_of_range[:5]}"
        )
    return float(values.mean())


def _positive_int(value: object, label: str) -> int:
    parsed = _finite_float(value, label)
    if parsed <= 0 or not parsed.is_integer():
        raise SystemExit(f"{label} must be a positive integer")
    return int(parsed)


def _required_text_column(df: pd.DataFrame, col: str, label: str) -> pd.Series:
    values = df[col].fillna("").astype(str).str.strip()
    blank_indexes = values[values == ""].index.tolist()
    if blank_indexes:
        raise SystemExit(
            f"{label} column '{col}' contains blank values "
            f"at row index(es) {blank_indexes[:5]}"
        )
    return values


def _direct_eval_status_counts(
    df: pd.DataFrame,
    label: str,
    passes_threshold: list[bool | None],
) -> dict[str, int]:
    _require_metric_columns(df, ("status",), label)
    statuses = _required_text_column(df, "status", label)
    allowed_statuses = {"evaluated", "no_ranking"}
    invalid = sorted(set(statuses) - allowed_statuses)
    if invalid:
        raise SystemExit(f"{label} contains invalid status values {invalid}")
    passing_non_evaluated = [
        idx
        for idx, status, passing in zip(
            statuses.index.tolist(),
            statuses.tolist(),
            passes_threshold,
            strict=True,
        )
        if passing is True and status != "evaluated"
    ]
    if passing_non_evaluated:
        raise SystemExit(
            f"{label} contains passing threshold rows with non-evaluated "
            f"status at row index(es) {passing_non_evaluated[:5]}"
        )
    return {
        "n_evaluated": int((statuses == "evaluated").sum()),
        "n_no_ranking": int((statuses == "no_ranking").sum()),
    }


def _source_labels(value: object, label: str, row_idx: int) -> list[str]:
    if pd.isna(value) or not str(value).strip():
        raise SystemExit(f"{label} column 'sources' is blank in top row")
    labels = [part.strip() for part in str(value).split(";")]
    if any(source == "" for source in labels):
        raise SystemExit(
            f"{label} column 'sources' contains empty source labels at row "
            f"index {row_idx}"
        )
    duplicates = sorted({source for source in labels if labels.count(source) > 1})
    if duplicates:
        raise SystemExit(
            f"{label} column 'sources' contains duplicate labels at row "
            f"index {row_idx}: {', '.join(duplicates[:10])}"
        )
    return labels


def _reject_direct_discovery_source_labels(
    labels: list[str],
    *,
    path: Path,
    row_idx: int,
) -> None:
    direct = sorted(
        label for label in labels if label.strip().lower() in DIRECT_DISCOVERY_SOURCE_LABELS
    )
    if direct:
        raise SystemExit(
            "Discovery direct records cannot be support or ranking features: "
            f"{path.name} row index {row_idx} has source label(s) {direct}"
        )


def _min_source_count_for_ranking(path: Path) -> int | None:
    if path.name == "cold_start__dti_only.csv":
        return None
    if path.name == "cold_start__fast.csv":
        return 2
    if path.name == "cold_start__comprehensive.csv":
        return 3
    if path.parent.name == "cosmetic_retro":
        return 2
    return None


def _disagreement_metrics(df: pd.DataFrame, label: str) -> dict[str, object]:
    required = ("bucket", "class", "count")
    _require_metric_columns(df, required, label)
    buckets = _required_text_column(df, "bucket", label)
    classes = _required_text_column(df, "class", label)
    allowed_buckets = {"docking_only", "dti_only", "both_top"}
    invalid_buckets = sorted(set(buckets) - allowed_buckets)
    if invalid_buckets:
        raise SystemExit(f"{label} contains invalid bucket values {invalid_buckets}")
    duplicate_pairs = df.loc[
        pd.DataFrame({"bucket": buckets, "class": classes}).duplicated(),
        ["bucket", "class"],
    ]
    if not duplicate_pairs.empty:
        shown = [
            f"{str(row['bucket']).strip()}:{str(row['class']).strip()}"
            for _, row in duplicate_pairs.head(5).iterrows()
        ]
        raise SystemExit(
            f"{label} contains duplicate bucket/class rows: {shown}"
        )
    counts = [
        _positive_int(value, f"{label} column 'count' row index {idx}")
        for idx, value in df["count"].items()
    ]
    return {
        "n_rows": int(len(df)),
        "n_classes": int(classes.nunique()),
        "n_targets": int(sum(counts)),
        "buckets": sorted(buckets.unique().tolist()),
    }


def _input_status(eval_dir: Path,
                  rankings_dir: Path,
                  chembl_fp_parquet: Path,
                  training_seq_db: Path,
                  training_ligands: Path,
                  training_holo: Path,
                  skin_known_cases_csv: Path = Path(
                      "data/validation/skin_known_target_panel.csv",
                  ),
                  skin_known_rankings_dir: Path | None = None) -> dict[str, object]:
    if skin_known_rankings_dir is None:
        skin_known_rankings_dir = rankings_dir / "skin_known_target"

    def is_available(path: Path) -> bool:
        if path.is_file():
            return path.stat().st_size > 0
        if path.is_dir():
            return any(
                child.is_file() and child.stat().st_size > 0
                for child in path.rglob("*")
            )
        return False

    groups = {
        "cold_start": {
            "triggers": [
                eval_dir / "cold_start_truth.csv",
                rankings_dir / "cold_start__comprehensive.csv",
                rankings_dir / "cold_start__fast.csv",
                rankings_dir / "cold_start__dti_only.csv",
            ],
            "required": [
                eval_dir / "cold_start_truth.csv",
                rankings_dir / "cold_start__comprehensive.csv",
                rankings_dir / "cold_start__fast.csv",
                rankings_dir / "cold_start__dti_only.csv",
            ],
        },
        "analog_quality": {
            "triggers": [
                eval_dir / "generated_analogs.csv",
                eval_dir / "other_cosing.csv",
            ],
            "required": [
                eval_dir / "generated_analogs.csv",
                eval_dir / "other_cosing.csv",
                chembl_fp_parquet,
            ],
        },
        "pharmacophore_conservation": {
            "triggers": [
                eval_dir / "parent.sdf",
                eval_dir / "consensus.json",
                eval_dir / "interaction_anchor_map.json",
                eval_dir / "generated_analogs.csv",
            ],
            "required": [
                eval_dir / "parent.sdf",
                eval_dir / "consensus.json",
                eval_dir / "interaction_anchor_map.json",
                eval_dir / "generated_analogs.csv",
            ],
        },
        "skin_known_target_recovery": {
            "triggers": [
                skin_known_rankings_dir,
                eval_dir / "skin_known_target_recovery.csv",
                eval_dir / "skin_known_target_recovery_targets.csv",
            ],
            "required": [
                skin_known_cases_csv,
                skin_known_rankings_dir,
            ],
        },
    }
    incomplete: list[dict[str, object]] = []
    checked: list[dict[str, object]] = []
    eval_targets = eval_dir / "eval_targets.csv"
    if eval_targets.exists():
        paths = [eval_targets, training_seq_db, training_ligands, training_holo]
        missing = [path for path in paths if not is_available(path)]
        record = {
            "name": "leakage",
            "present": [str(path) for path in paths if is_available(path)],
            "missing": [str(path) for path in missing],
        }
        checked.append(record)
        if missing:
            incomplete.append(record)
    for name, spec in groups.items():
        triggers = spec["triggers"]
        paths = spec["required"]
        if not any(path.exists() for path in triggers):
            continue
        present = [path for path in paths if is_available(path)]
        missing = [path for path in paths if not is_available(path)]
        record = {
            "name": name,
            "present": [str(path) for path in present],
            "missing": [str(path) for path in missing],
        }
        checked.append(record)
        if missing:
            incomplete.append(record)
    return {
        "status": "ok" if not incomplete else "incomplete",
        "n_checked": len(checked),
        "n_incomplete": len(incomplete),
        "checked": checked,
        "incomplete": incomplete,
    }


def _ranking_metrics(metric_source: Path | set[Path]) -> dict[str, object]:
    metrics: dict[str, object] = {}
    metric_paths = _metric_paths(metric_source)
    paths_by_name = {path.name: path for path in metric_paths}
    for path in (
        paths_by_name.get("cold_start_comprehensive.csv"),
        paths_by_name.get("cold_start_fast.csv"),
        paths_by_name.get("cold_start_dti_only.csv"),
    ):
        if path is None:
            continue
        df = _read_csv_if_present(path)
        if df is None or df.empty:
            continue
        cold_metric_cols = (
            "n_cold_targets",
            "n_truth_in_cold",
            "recall@1",
            "recall@5",
            "recall@10",
            "recall@50",
        )
        _require_metric_columns(df, cold_metric_cols, path.name)
        _require_single_metric_row(df, path.name)
        row = df.iloc[0]
        metrics[path.stem] = {
            col: (
                _finite_float(row[col], f"{path.name} column '{col}'")
                if col.startswith("n_")
                else _fraction_float(row[col], f"{path.name} column '{col}'")
            )
            for col in cold_metric_cols
        }
        if "passes_threshold" in df.columns:
            metrics[path.stem].update(_threshold_metric(row["passes_threshold"]))
    retro = _read_csv_if_present(paths_by_name["cosmetic_retrospective.csv"]) if "cosmetic_retrospective.csv" in paths_by_name else None
    if retro is not None and not retro.empty:
        retro_metric_cols = ("top1", "top5", "top10")
        _require_metric_columns(
            retro,
            retro_metric_cols,
            "cosmetic_retrospective.csv",
        )
        metrics["cosmetic_retrospective"] = {
            col: _fraction_column_mean(
                retro,
                col,
                f"cosmetic_retrospective.csv column '{col}'",
            )
            for col in retro_metric_cols
        }
        metrics["cosmetic_retrospective"]["n_cases"] = int(len(retro))
        if "passes_threshold" in retro.columns:
            values = [_as_bool(v) for v in retro["passes_threshold"].tolist()]
            metrics["cosmetic_retrospective"].update(
                _direct_eval_status_counts(
                    retro,
                    "cosmetic_retrospective.csv",
                    values,
                )
            )
            metrics["cosmetic_retrospective"]["passes_threshold"] = all(
                value is True for value in values
            )
            metrics["cosmetic_retrospective"]["passes_threshold_valid"] = all(
                value is not None for value in values
            )
    skin = _read_csv_if_present(paths_by_name["skin_efficacy_recovery.csv"]) if "skin_efficacy_recovery.csv" in paths_by_name else None
    if skin is not None and not skin.empty:
        skin_metric_cols = ("precision", "recall")
        _require_metric_columns(
            skin,
            skin_metric_cols,
            "skin_efficacy_recovery.csv",
        )
        metrics["skin_efficacy_recovery"] = {
            col: _fraction_column_mean(
                skin,
                col,
                f"skin_efficacy_recovery.csv column '{col}'",
            )
            for col in skin_metric_cols
        }
        metrics["skin_efficacy_recovery"]["n_cases"] = int(len(skin))
        if "passes_threshold" in skin.columns:
            values = [_as_bool(v) for v in skin["passes_threshold"].tolist()]
            metrics["skin_efficacy_recovery"].update(
                _direct_eval_status_counts(
                    skin,
                    "skin_efficacy_recovery.csv",
                    values,
                )
            )
            metrics["skin_efficacy_recovery"]["passes_threshold"] = all(
                value is True for value in values
            )
            metrics["skin_efficacy_recovery"]["passes_threshold_valid"] = all(
                value is not None for value in values
            )
    skin_known = (
        _read_csv_if_present(paths_by_name["skin_known_target_recovery.csv"])
        if "skin_known_target_recovery.csv" in paths_by_name
        else None
    )
    if skin_known is not None and not skin_known.empty:
        skin_known_cols = (
            "case_top10",
            "target_top10_fraction",
            "target_top30_fraction",
            "status",
            "panel",
            "context_profile",
        )
        _require_metric_columns(
            skin_known,
            skin_known_cols,
            "skin_known_target_recovery.csv",
        )
        values = (
            [_as_bool(v) for v in skin_known["passes_threshold"].tolist()]
            if "passes_threshold" in skin_known.columns
            else []
        )
        evaluated = skin_known[
            skin_known["status"].fillna("").astype(str).str.strip() == "evaluated"
        ]
        target_pair_metrics: dict[str, float] = {}
        skin_known_targets = (
            _read_csv_if_present(
                paths_by_name["skin_known_target_recovery_targets.csv"],
            )
            if "skin_known_target_recovery_targets.csv" in paths_by_name
            else None
        )
        if skin_known_targets is not None and not skin_known_targets.empty:
            target_cols = ("target_top10", "target_top30", "status")
            _require_metric_columns(
                skin_known_targets,
                target_cols,
                "skin_known_target_recovery_targets.csv",
            )
            evaluated_targets = skin_known_targets[
                skin_known_targets["status"].fillna("").astype(str).str.strip()
                == "evaluated"
            ]
            target_pair_metrics = {
                "target_top10": _fraction_column_mean(
                    evaluated_targets,
                    "target_top10",
                    "skin_known_target_recovery_targets.csv column 'target_top10'",
                ) if not evaluated_targets.empty else 0.0,
                "target_top30": _fraction_column_mean(
                    evaluated_targets,
                    "target_top30",
                    "skin_known_target_recovery_targets.csv column 'target_top30'",
                ) if not evaluated_targets.empty else 0.0,
                "n_evaluated_targets": int(len(evaluated_targets)),
            }
        metrics["skin_known_target_recovery"] = {
            "n_cases": int(len(skin_known)),
            "n_evaluated": int(len(evaluated)),
            "n_no_ranking": int(
                (skin_known["status"].astype(str) == "no_ranking").sum(),
            ),
            "n_panels": int(skin_known["panel"].astype(str).nunique()),
            "n_context_profiles": int(
                skin_known["context_profile"].astype(str).nunique(),
            ),
            "case_top10": _fraction_column_mean(
                evaluated,
                "case_top10",
                "skin_known_target_recovery.csv column 'case_top10'",
            ) if not evaluated.empty else 0.0,
            "target_top10": target_pair_metrics.get(
                "target_top10",
                _fraction_column_mean(
                    evaluated,
                    "target_top10_fraction",
                    "skin_known_target_recovery.csv column 'target_top10_fraction'",
                ) if not evaluated.empty else 0.0,
            ),
            "target_top30": target_pair_metrics.get(
                "target_top30",
                _fraction_column_mean(
                    evaluated,
                    "target_top30_fraction",
                    "skin_known_target_recovery.csv column 'target_top30_fraction'",
                ) if not evaluated.empty else 0.0,
            ),
            "n_evaluated_targets": int(
                target_pair_metrics.get("n_evaluated_targets", 0),
            ),
        }
        if values:
            metrics["skin_known_target_recovery"].update(
                _direct_eval_status_counts(
                    skin_known,
                    "skin_known_target_recovery.csv",
                    values,
                )
            )
            metrics["skin_known_target_recovery"]["passes_threshold"] = all(
                value is True for value in values
            )
            metrics["skin_known_target_recovery"]["passes_threshold_valid"] = all(
                value is not None for value in values
            )
    analog = _read_csv_if_present(paths_by_name["analog_quality.csv"]) if "analog_quality.csv" in paths_by_name else None
    if analog is not None and not analog.empty:
        analog_metric_cols = (
            "recovery_fraction",
            "novelty_fraction",
            "mean_ra_score",
        )
        _require_metric_columns(
            analog,
            analog_metric_cols,
            "analog_quality.csv",
        )
        _require_single_metric_row(analog, "analog_quality.csv")
        row = analog.iloc[0]
        metrics["analog_quality"] = {
            col: _fraction_float(row[col], f"analog_quality.csv column '{col}'")
            for col in analog_metric_cols
            if col in analog.columns and pd.notna(row[col])
        }
        if "passes_threshold" in analog.columns:
            metrics["analog_quality"].update(_threshold_metric(row["passes_threshold"]))
    pharm = _read_csv_if_present(paths_by_name["pharmacophore_conservation.csv"]) if "pharmacophore_conservation.csv" in paths_by_name else None
    if pharm is not None and not pharm.empty:
        pharm_metric_cols = ("preserved_fraction",)
        _require_metric_columns(
            pharm,
            pharm_metric_cols,
            "pharmacophore_conservation.csv",
        )
        _require_single_metric_row(pharm, "pharmacophore_conservation.csv")
        row = pharm.iloc[0]
        metrics["pharmacophore_conservation"] = {
            col: _fraction_float(
                row[col],
                f"pharmacophore_conservation.csv column '{col}'",
            )
            for col in pharm_metric_cols
            if col in pharm.columns and pd.notna(row[col])
        }
        if "passes_threshold" in pharm.columns:
            metrics["pharmacophore_conservation"].update(
                _threshold_metric(row["passes_threshold"])
            )
    disagreement = _read_csv_if_present(paths_by_name["disagreement_per_class.csv"]) if "disagreement_per_class.csv" in paths_by_name else None
    if disagreement is not None and not disagreement.empty:
        metrics["disagreement"] = _disagreement_metrics(
            disagreement,
            "disagreement_per_class.csv",
        )
    return metrics


def _top_target_rationale(rankings_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    candidates = sorted(rankings_dir.glob("cold_start__*.csv"))
    candidates.extend(sorted((rankings_dir / "cosmetic_retro").glob("*.csv")))
    score_cols = (
        "final_score",
        "rrf_score",
        "score",
        "skin_weighted_score",
        "dock_score",
        "psichic_score",
    )
    rationale_cols = (
        "source_count",
        "sources",
        "skin_score",
        "efficacy_category",
        "cosmetic_decision",
        "drug_decision",
    )
    for path in candidates:
        df = _read_csv_if_present(path)
        if df is None:
            continue
        if df.empty:
            raise SystemExit(f"{path.name} contains no rows")
        if "target_id" not in df.columns:
            raise SystemExit(f"{path.name} missing required column 'target_id'")
        target_ids = df["target_id"].astype(str).str.strip()
        blank_ids = target_ids == ""
        if blank_ids.any():
            first_blank = int(blank_ids[blank_ids].index[0])
            raise SystemExit(
                f"{path.name} column 'target_id' is blank at row index {first_blank}"
            )
        duplicate_ids = target_ids[target_ids.duplicated()].tolist()
        if duplicate_ids:
            shown = ", ".join(duplicate_ids[:10])
            suffix = "..." if len(duplicate_ids) > 10 else ""
            raise SystemExit(
                f"{path.name} contains duplicate target_id values: {shown}{suffix}"
            )
        row = df.iloc[0]
        target_id = target_ids.iloc[0]
        score_col = next(
            (col for col in score_cols if col in df.columns and pd.notna(row[col])),
            None,
        )
        if score_col is None:
            raise SystemExit(
                f"{path.name} top row must include at least one score column "
                f"from {list(score_cols)}"
            )
        scores = [
            _finite_float(value, f"{path.name} column '{score_col}'")
            for value in df[score_col]
        ]
        score = scores[0]
        best_score = max(scores)
        if best_score > score and not math.isclose(
            best_score,
            score,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise SystemExit(
                f"{path.name} top row must have the best {score_col}"
            )
        tied_top_indexes = [
            idx
            for idx, value in enumerate(scores)
            if math.isclose(value, score, rel_tol=1e-9, abs_tol=1e-12)
        ]
        if len(tied_top_indexes) > 1:
            shown = ", ".join(str(idx) for idx in tied_top_indexes[:10])
            suffix = "..." if len(tied_top_indexes) > 10 else ""
            raise SystemExit(
                f"{path.name} top row {score_col} is tied at row index(es) "
                f"{shown}{suffix}"
            )
        min_source_count = _min_source_count_for_ranking(path)
        requires_scorer_coverage = min_source_count is not None
        if requires_scorer_coverage:
            missing_rationale = [
                col for col in ("source_count", "sources") if col not in df.columns
            ]
            if missing_rationale:
                raise SystemExit(
                    f"{path.name} missing required top-target rationale "
                    f"columns {missing_rationale}"
                )
            for row_idx, rationale_row in df.iterrows():
                row_source_count = _positive_int(
                    rationale_row["source_count"],
                    f"{path.name} column 'source_count'",
                )
                row_labels = _source_labels(
                    rationale_row["sources"],
                    path.name,
                    int(row_idx),
                )
                _reject_direct_discovery_source_labels(
                    row_labels,
                    path=path,
                    row_idx=int(row_idx),
                )
                if row_source_count != len(row_labels):
                    if int(row_idx) == 0:
                        raise SystemExit(
                            f"{path.name} source_count={row_source_count} but "
                            f"sources lists {len(row_labels)} label(s) in top row"
                        )
                    raise SystemExit(
                        f"{path.name} source_count={row_source_count} but "
                        f"sources lists {len(row_labels)} label(s) at row "
                        f"index {int(row_idx)}"
                    )
                if row_source_count < min_source_count:
                    if int(row_idx) == 0:
                        raise SystemExit(
                            f"{path.name} top row requires source_count >= "
                            f"{min_source_count} for claim-quality rationale"
                        )
                    raise SystemExit(
                        f"{path.name} row index {int(row_idx)} requires "
                        f"source_count >= {min_source_count} for "
                        "claim-quality rationale"
                    )
        if "skin_score" in df.columns:
            for row_idx, value in df["skin_score"].items():
                if pd.isna(value) or (isinstance(value, str) and not value.strip()):
                    if int(row_idx) == 0:
                        raise SystemExit(
                            f"{path.name} column 'skin_score' is blank in top row"
                        )
                    raise SystemExit(
                        f"{path.name} column 'skin_score' is blank "
                        f"at row index {int(row_idx)}"
                    )
                _fraction_float(
                    value,
                    (
                        f"{path.name} column 'skin_score'"
                        if int(row_idx) == 0
                        else (
                            f"{path.name} column 'skin_score' "
                            f"at row index {int(row_idx)}"
                        )
                    ),
                )
        for text_col in (
            "efficacy_category",
            "cosmetic_decision",
            "drug_decision",
        ):
            if text_col not in df.columns:
                continue
            for row_idx, value in df[text_col].items():
                if pd.isna(value) or (isinstance(value, str) and not value.strip()):
                    if int(row_idx) == 0:
                        raise SystemExit(
                            f"{path.name} column '{text_col}' is blank in top row"
                        )
                    raise SystemExit(
                        f"{path.name} column '{text_col}' is blank "
                        f"at row index {int(row_idx)}"
                    )
        rationale: dict[str, object] = {}
        for col in rationale_cols:
            if col not in df.columns or pd.isna(row[col]):
                continue
            if col == "source_count":
                rationale[col] = _positive_int(
                    row[col],
                    f"{path.name} column '{col}'",
                )
                continue
            if col == "sources" and "source_count" in rationale:
                labels = _source_labels(row[col], path.name, int(row.name))
                source_count = int(rationale["source_count"])
                if source_count != len(labels):
                    raise SystemExit(
                        f"{path.name} source_count={source_count} but sources "
                        f"lists {len(labels)} label(s) in top row"
                    )
            if col == "skin_score":
                rationale[col] = _fraction_float(
                    row[col],
                    f"{path.name} column '{col}'",
                )
                continue
            value = row[col].item() if hasattr(row[col], "item") else row[col]
            if isinstance(value, str):
                value = value.strip()
                if not value:
                    raise SystemExit(
                        f"{path.name} column '{col}' is blank in top row"
                    )
            rationale[col] = value
        if min_source_count is not None:
            source_count = int(rationale.get("source_count", 0))
            if source_count < min_source_count:
                raise SystemExit(
                    f"{path.name} top row requires source_count >= "
                    f"{min_source_count} for claim-quality rationale"
                )
        rows.append({
            "ranking": str(path),
            "target_id": target_id,
            "score": score,
            "rationale": rationale,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", default=Path("results/eval"), type=Path)
    parser.add_argument("--rankings-dir",
                        default=Path("results/eval/rankings"),
                        type=Path)
    parser.add_argument("--out-manifest",
                        default=Path("results/eval/iteration_manifest.json"),
                        type=Path)
    parser.add_argument("--workflow-config",
                        default=Path("workflow/config.yaml"),
                        type=Path)
    parser.add_argument(
        "--activity-retrieval-gate",
        default=Path("data/manifests/activity_retrieval_final_gate.flag"),
        type=Path,
        help="Sealed dual-cold/calibration activity-retrieval production gate.",
    )
    parser.add_argument("--collected-runs-manifest",
                        default=None,
                        type=Path,
                        help="Manifest emitted by eval/collect_run_outputs.py.")
    parser.add_argument("--skin-known-run-ledger-csv",
                        default=None,
                        type=Path,
                        help="Per-case same-run ledger for skin known-target "
                             "SOTA recovery claims.")
    parser.add_argument("--sota-baselines-csv",
                        default=None,
                        type=Path,
                        help="Baseline/comparator fairness freeze table for "
                             "SOTA claims.")
    parser.add_argument("--sota-ablations-csv",
                        default=None,
                        type=Path,
                        help="Ablation freeze table for SOTA claims.")
    parser.add_argument("--prospective-promotion-manifest",
                        default=None,
                        type=Path,
                        help="Manifest emitted by eval/prospective_promotion_eval.py.")
    parser.add_argument("--prospective-metrics-csv",
                        default=None,
                        type=Path,
                        help="Active metrics CSV that the prospective promotion "
                             "manifest must exactly reproduce.")
    parser.add_argument("--candidate-prereg-json",
                        default=None,
                        type=Path,
                        help="Sealed preregistered E0-E3 candidate matrix JSON.")
    parser.add_argument("--candidate-results-csv",
                        default=None,
                        type=Path,
                        help="Exact 37-row E0-E3 candidate results CSV.")
    parser.add_argument("--candidate-selection-json",
                        default=None,
                        type=Path,
                        help="Selection JSON emitted from the active candidate "
                             "results.")
    parser.add_argument("--target-classes",
                        default=Path("data/chembl37/target_classes.parquet"),
                        type=Path)
    parser.add_argument("--chembl-fp-parquet",
                        default=Path("data/chembl37/fp_morgan2_2048.parquet"),
                        type=Path)
    parser.add_argument("--chembl-dir", default=Path("data/chembl37"), type=Path)
    parser.add_argument("--training-seq-db",
                        default=Path("data/mmseqs/training_cutoff_db"),
                        type=Path)
    parser.add_argument("--training-ligands",
                        default=Path("data/chembl37/training_ligands.smi"),
                        type=Path)
    parser.add_argument("--training-holo",
                        default=Path("data/plinder/training_holo_pockets.csv"),
                        type=Path)
    parser.add_argument("--direct-exact-reference",
                        default=None,
                        type=Path,
                        help="Required exact direct-evidence SMILES reference for a "
                             "claim-quality Discovery audit; defaults to "
                             "EVAL_DIR/direct_exact_reference.smi when present, "
                             "otherwise the canonical Stage 0 reference.")
    parser.add_argument("--direct-exact-manifest",
                        default=None,
                        type=Path,
                        help="Optional sealed alias-map manifest. It is required "
                             "automatically for the canonical Stage 0 reference.")
    parser.add_argument("--skin-known-cases-csv",
                        default=Path("data/validation/skin_known_target_panel.csv"),
                        type=Path)
    parser.add_argument("--skin-known-rankings-dir",
                        default=Path("results/eval/rankings/skin_known_target"),
                        type=Path)
    parser.add_argument("--seq-id-threshold", type=float, default=0.30)
    parser.add_argument("--ligand-tanimoto-threshold", type=float, default=0.50)
    parser.add_argument("--pocket-sucos-threshold", type=float, default=0.50)
    parser.add_argument("--cold-start-min-recall-at-50", type=float, default=0.01)
    parser.add_argument("--cosmetic-retro-min-mean-top10", type=float, default=0.01)
    parser.add_argument("--skin-known-min-case-top10", type=float, default=0.80)
    parser.add_argument("--skin-known-min-target-top10", type=float, default=0.50)
    parser.add_argument("--skin-known-min-target-top30", type=float, default=0.60)
    parser.add_argument("--skin-known-context-profile", default="auto")
    parser.add_argument("--skin-efficacy-min-mean-precision", type=float, default=0.01)
    parser.add_argument("--skin-efficacy-min-mean-recall", type=float, default=0.01)
    parser.add_argument("--analog-min-recovery", type=float, default=0.01)
    parser.add_argument("--analog-min-novelty", type=float, default=0.80)
    parser.add_argument("--analog-min-mean-ra-score", type=float, default=0.70)
    parser.add_argument("--pharmacophore-min-preserved-fraction", type=float, default=0.80)
    parser.add_argument("--allow-incomplete-leakage", action="store_true")
    parser.add_argument("--allow-threshold-failure", action="store_true",
                        help="Write a manifest with failing metric thresholds "
                             "only for explicit diagnostics.")
    parser.add_argument("--allow-incomplete-eval-inputs", action="store_true",
                        help="Write a manifest with partial benchmark inputs "
                             "only for explicit diagnostics.")
    parser.add_argument("--allow-empty-iteration", action="store_true",
                        help="Write a zero-step manifest only for explicit diagnostics.")
    parser.add_argument("--allow-missing-input-runs", action="store_true",
                        help="Allow missing collected-run provenance only "
                             "for explicit diagnostics.")
    args = parser.parse_args()
    for value, label in (
        (args.seq_id_threshold, "--seq-id-threshold"),
        (args.ligand_tanimoto_threshold, "--ligand-tanimoto-threshold"),
        (args.pocket_sucos_threshold, "--pocket-sucos-threshold"),
        (args.cold_start_min_recall_at_50, "--cold-start-min-recall-at-50"),
        (args.cosmetic_retro_min_mean_top10, "--cosmetic-retro-min-mean-top10"),
        (args.skin_known_min_case_top10, "--skin-known-min-case-top10"),
        (args.skin_known_min_target_top10, "--skin-known-min-target-top10"),
        (args.skin_known_min_target_top30, "--skin-known-min-target-top30"),
        (
            args.skin_efficacy_min_mean_precision,
            "--skin-efficacy-min-mean-precision",
        ),
        (args.skin_efficacy_min_mean_recall, "--skin-efficacy-min-mean-recall"),
        (args.analog_min_recovery, "--analog-min-recovery"),
        (args.analog_min_novelty, "--analog-min-novelty"),
        (args.analog_min_mean_ra_score, "--analog-min-mean-ra-score"),
        (
            args.pharmacophore_min_preserved_fraction,
            "--pharmacophore-min-preserved-fraction",
        ),
    ):
        _validate_fraction_threshold(value, label)
    if args.skin_known_context_profile not in SKIN_KNOWN_CONTEXT_PROFILES:
        raise SystemExit(
            "--skin-known-context-profile must be one of "
            + ", ".join(sorted(SKIN_KNOWN_CONTEXT_PROFILES))
            + f": {args.skin_known_context_profile!r}"
        )

    args.eval_dir.mkdir(parents=True, exist_ok=True)
    collected_runs_manifest = (
        args.collected_runs_manifest
        if args.collected_runs_manifest is not None
        else args.eval_dir / "collected_runs.json"
    )
    skin_known_run_ledger_csv = (
        args.skin_known_run_ledger_csv
        if args.skin_known_run_ledger_csv is not None
        else args.eval_dir / "skin_known_target_run_ledger.csv"
    )
    sota_baselines_csv = (
        args.sota_baselines_csv
        if args.sota_baselines_csv is not None
        else args.eval_dir / "sota_baselines.csv"
    )
    sota_ablations_csv = (
        args.sota_ablations_csv
        if args.sota_ablations_csv is not None
        else args.eval_dir / "sota_ablations.csv"
    )
    prospective_promotion_manifest = (
        args.prospective_promotion_manifest
        if args.prospective_promotion_manifest is not None
        else args.eval_dir / "prospective_promotion_manifest.json"
    )
    prospective_metrics_csv = (
        args.prospective_metrics_csv
        if args.prospective_metrics_csv is not None
        else args.eval_dir / "prospective_metrics.csv"
    )
    candidate_prereg_json = (
        args.candidate_prereg_json
        if args.candidate_prereg_json is not None
        else args.eval_dir / "candidate_preregistration.json"
    )
    candidate_results_csv = (
        args.candidate_results_csv
        if args.candidate_results_csv is not None
        else args.eval_dir / "candidate_results.csv"
    )
    candidate_selection_json = (
        args.candidate_selection_json
        if args.candidate_selection_json is not None
        else args.eval_dir / "candidate_selection.json"
    )
    steps: list[EvalStep] = []
    py = sys.executable
    threshold_failure_args = (
        ["--allow-threshold-failure"] if args.allow_threshold_failure else []
    )

    eval_targets = args.eval_dir / "eval_targets.csv"
    leakage_csv = args.eval_dir / "leakage_audit.csv"
    discovery_audit_path = args.eval_dir / "discovery_leakage_audit.json"
    leakage_ran = False
    leakage_execution_challenge: str | None = None
    leakage_refs = [
        args.training_seq_db,
        args.training_ligands,
        args.training_holo,
    ]
    (
        effective_direct_exact_reference,
        effective_direct_exact_manifest,
    ) = _resolve_direct_exact_inputs(
        eval_dir=args.eval_dir,
        explicit_reference=args.direct_exact_reference,
        explicit_manifest=args.direct_exact_manifest,
    )
    if (
        eval_targets.exists()
        and (
            all(path.exists() for path in leakage_refs)
            or not args.allow_incomplete_eval_inputs
        )
    ):
        out = leakage_csv
        out_discovery = discovery_audit_path
        discovery_inputs_ready = effective_direct_exact_reference.is_file() and (
            effective_direct_exact_manifest is None
            or effective_direct_exact_manifest.is_file()
        )
        discovery_inputs_explicit = (
            args.direct_exact_reference is not None
            or args.direct_exact_manifest is not None
        )
        cmd = [
            py, "eval/leakage_check.py",
            "--input-csv", str(eval_targets),
            "--training-seq-db", str(args.training_seq_db),
            "--training-ligands", str(args.training_ligands),
            "--training-holo", str(args.training_holo),
            "--seq-id-threshold", str(args.seq_id_threshold),
            "--ligand-tanimoto-threshold", str(args.ligand_tanimoto_threshold),
            "--pocket-sucos-threshold", str(args.pocket_sucos_threshold),
            "--out-csv", str(out),
        ]
        outputs = [out]
        if (
            discovery_inputs_ready
            or discovery_inputs_explicit
            or not args.allow_incomplete_eval_inputs
        ):
            leakage_execution_challenge = secrets.token_hex(32)
            cmd.extend([
                "--out-discovery-audit-json", str(out_discovery),
                "--direct-exact-reference", str(effective_direct_exact_reference),
                "--execution-challenge", leakage_execution_challenge,
            ])
            if effective_direct_exact_manifest is not None:
                cmd.extend([
                    "--direct-exact-manifest",
                    str(effective_direct_exact_manifest),
                ])
            outputs.append(out_discovery)
            leakage_ran = True
        else:
            out_discovery.unlink(missing_ok=True)
        if args.allow_incomplete_leakage:
            cmd.append("--allow-incomplete")
        steps.append(run_step("leakage", cmd, outputs))

    truth = args.eval_dir / "cold_start_truth.csv"
    for mode in ("comprehensive", "fast", "dti_only"):
        ranking = args.rankings_dir / f"cold_start__{mode}.csv"
        if ranking.exists() and truth.exists():
            out = args.eval_dir / f"cold_start_{mode}.csv"
            steps.append(run_step(
                f"cold_start_{mode}",
                [
                    py, "eval/cold_start_eval.py",
                    "--mode", mode,
                    "--ranking-csv", str(ranking),
                    "--ground-truth-csv", str(truth),
                    "--chembl-dir", str(args.chembl_dir),
                    "--min-recall-at-50", str(args.cold_start_min_recall_at_50),
                    *threshold_failure_args,
                    "--out-csv", str(out),
                ],
                [out],
            ))

    disagreement = args.eval_dir / "disagreement_analysis.json"
    if disagreement.exists():
        out = args.eval_dir / "disagreement_per_class.csv"
        steps.append(run_step(
            "disagreement",
            [
                py, "eval/disagreement_eval.py",
                "--disagreement-json", str(disagreement),
                "--target-classes", str(args.target_classes),
                "--out-csv", str(out),
            ],
            [out],
        ))

    retro_dir = args.rankings_dir / "cosmetic_retro"
    if retro_dir.exists():
        out = args.eval_dir / "cosmetic_retrospective.csv"
        steps.append(run_step(
            "cosmetic_retrospective",
            [
                py, "eval/cosmetic_retrospective_eval.py",
                "--rankings-dir", str(retro_dir),
                "--min-mean-top10", str(args.cosmetic_retro_min_mean_top10),
                *threshold_failure_args,
                "--out-csv", str(out),
            ],
            [out],
        ))
        out = args.eval_dir / "skin_efficacy_recovery.csv"
        steps.append(run_step(
            "skin_efficacy_recovery",
            [
                py, "eval/skin_efficacy_recovery_eval.py",
                "--ranked-dir", str(retro_dir),
                "--min-mean-precision",
                str(args.skin_efficacy_min_mean_precision),
                "--min-mean-recall",
                str(args.skin_efficacy_min_mean_recall),
                *threshold_failure_args,
                "--out-csv", str(out),
            ],
            [out],
        ))

    if args.skin_known_rankings_dir.exists():
        out = args.eval_dir / "skin_known_target_recovery.csv"
        out_targets = args.eval_dir / "skin_known_target_recovery_targets.csv"
        out_summary = args.eval_dir / "skin_known_target_recovery_summary.json"
        steps.append(run_step(
            "skin_known_target_recovery",
            [
                py, "eval/skin_known_target_recovery_eval.py",
                "--cases-csv", str(args.skin_known_cases_csv),
                "--rankings-dir", str(args.skin_known_rankings_dir),
                "--out-csv", str(out),
                "--out-target-csv", str(out_targets),
                "--out-summary-json", str(out_summary),
                "--context-profile", args.skin_known_context_profile,
                "--min-case-top10", str(args.skin_known_min_case_top10),
                "--min-target-top10", str(args.skin_known_min_target_top10),
                "--min-target-top30", str(args.skin_known_min_target_top30),
                *threshold_failure_args,
            ],
            [out, out_targets, out_summary],
        ))

    analogs = args.eval_dir / "generated_analogs.csv"
    other_cosing = args.eval_dir / "other_cosing.csv"
    if analogs.exists() and other_cosing.exists() and args.chembl_fp_parquet.exists():
        out = args.eval_dir / "analog_quality.csv"
        steps.append(run_step(
            "analog_quality",
            [
                py, "eval/analog_quality_eval.py",
                "--analogs-csv", str(analogs),
                "--other-cosing-csv", str(other_cosing),
                "--chembl-fp-parquet", str(args.chembl_fp_parquet),
                "--min-recovery", str(args.analog_min_recovery),
                "--min-novelty", str(args.analog_min_novelty),
                "--min-mean-ra-score", str(args.analog_min_mean_ra_score),
                *threshold_failure_args,
                "--out-csv", str(out),
            ],
            [out],
        ))

    parent_sdf = args.eval_dir / "parent.sdf"
    consensus = args.eval_dir / "consensus.json"
    interaction_anchors = args.eval_dir / "interaction_anchor_map.json"
    if (
        parent_sdf.exists()
        and consensus.exists()
        and interaction_anchors.exists()
        and analogs.exists()
    ):
        out = args.eval_dir / "pharmacophore_conservation.csv"
        detail = args.eval_dir / "pharmacophore_conservation_detail.csv"
        steps.append(run_step(
            "pharmacophore_conservation",
            [
                py, "eval/pharmacophore_conservation_eval.py",
                "--parent-sdf", str(parent_sdf),
                "--consensus-json", str(consensus),
                "--interaction-anchor-map", str(interaction_anchors),
                "--analogs-csv", str(analogs),
                "--min-preserved-fraction",
                str(args.pharmacophore_min_preserved_fraction),
                *threshold_failure_args,
                "--out-csv", str(out),
                "--out-detail-csv", str(detail),
            ],
            [out, detail],
        ))

    current_metric_paths = {
        Path(output).resolve()
        for step in steps
        for output in step.outputs
        if Path(output).exists() and Path(output).is_file()
    }
    metric_source: Path | set[Path] = current_metric_paths if steps else args.eval_dir
    if leakage_ran:
        assert leakage_execution_challenge is not None
        sealed_audit_status = _sealed_leakage_status(
            discovery_audit_path,
            expected_sources={
                "input_csv": eval_targets,
                "training_seq_db": args.training_seq_db,
                "training_ligands": args.training_ligands,
                "training_holo": args.training_holo,
                "direct_exact_reference": effective_direct_exact_reference,
                "leakage_audit_csv": leakage_csv,
            },
            expected_thresholds=LeakageThresholds(
                args.seq_id_threshold,
                args.ligand_tanimoto_threshold,
                args.pocket_sucos_threshold,
            ),
            expected_execution_challenge=leakage_execution_challenge,
            direct_exact_manifest=effective_direct_exact_manifest,
        )
        leakage_status = _leakage_status(
            args.eval_dir,
            sealed_audit_status,
            LeakageThresholds(
                args.seq_id_threshold,
                args.ligand_tanimoto_threshold,
                args.pocket_sucos_threshold,
            ),
        )
    else:
        leakage_status = {
            "status": "not_run",
            "n_rows": 0,
            "n_incomplete": 0,
            "n_leak_flags": 0,
            "n_invalid_rows": 0,
            "sealed_audit": {
                "status": "not_run",
                "path": str(discovery_audit_path),
                "direct_exact_count": 0,
                "incomplete_audit_count": 0,
                "neutral_exclusion_count": 0,
                "n_survivors": 0,
            },
        }
    threshold_status = _threshold_status(metric_source)
    input_status = _input_status(
        args.eval_dir,
        args.rankings_dir,
        args.chembl_fp_parquet,
        args.training_seq_db,
        args.training_ligands,
        args.training_holo,
        skin_known_cases_csv=args.skin_known_cases_csv,
        skin_known_rankings_dir=args.skin_known_rankings_dir,
    )
    input_runs = _input_runs(collected_runs_manifest)
    input_runs_claim_blocker = _input_runs_claim_blocker(
        input_runs,
        args.eval_dir,
        args.rankings_dir,
    )
    sota_active = (
        args.skin_known_rankings_dir.exists()
        or (args.eval_dir / "skin_known_target_recovery.csv").exists()
    )
    if sota_active:
        sota_freeze: dict[str, object] = {
            "status": "checked",
            "skin_known_run_ledger": _skin_known_run_ledger(
                skin_known_run_ledger_csv,
                args.skin_known_cases_csv,
                args.skin_known_rankings_dir,
                input_runs,
            ),
            "baseline_fairness": _sota_baseline_fairness(
                sota_baselines_csv,
                args.skin_known_cases_csv,
                args.seq_id_threshold,
                args.ligand_tanimoto_threshold,
                args.pocket_sucos_threshold,
            ),
            "ablation_freeze": _sota_ablation_freeze(
                sota_ablations_csv,
                args.skin_known_cases_csv,
            ),
        }
    else:
        sota_freeze = {
            "status": "not_applicable",
            "reason": "skin known-target recovery was not present in this iteration",
            "skin_known_run_ledger": {
                "status": "not_applicable",
                "path": str(skin_known_run_ledger_csv),
            },
            "baseline_fairness": {
                "status": "not_applicable",
                "path": str(sota_baselines_csv),
            },
            "ablation_freeze": {
                "status": "not_applicable",
                "path": str(sota_ablations_csv),
            },
        }
    sota_freeze_claim_blocker = _sota_freeze_claim_blocker(
        sota_active,
        sota_freeze,
    )
    candidate_experiment = _candidate_experiment_status(
        candidate_prereg_json,
        candidate_results_csv,
        candidate_selection_json,
        prereg_arg=args.candidate_prereg_json,
        results_arg=args.candidate_results_csv,
        selection_arg=args.candidate_selection_json,
    )
    prospective_promotion = _prospective_promotion_status(
        prospective_promotion_manifest,
        prospective_metrics_csv,
    )
    activity_retrieval_gate = _activity_retrieval_gate_status(
        args.activity_retrieval_gate
    )
    manifest = {
        "eval_dir": str(args.eval_dir),
        "rankings_dir": str(args.rankings_dir),
        "provenance": _provenance(args.workflow_config),
        "n_steps": len(steps),
        "n_passed": sum(1 for s in steps if s.status == "passed"),
        "n_failed": sum(1 for s in steps if s.status == "failed"),
        "steps": [asdict(step) for step in steps],
        "input_runs": input_runs,
        "sota_freeze": sota_freeze,
        "prospective_promotion": prospective_promotion,
        "candidate_experiment": candidate_experiment,
        "activity_retrieval_gate": activity_retrieval_gate,
        "input_status": input_status,
        "leakage_status": leakage_status,
        "threshold_status": threshold_status,
        "ranking_metrics": _ranking_metrics(metric_source),
        "top_target_rationale": _top_target_rationale(args.rankings_dir),
        "data_snapshot": _data_snapshot({
            "workflow_config": args.workflow_config,
            "target_classes": args.target_classes,
            "chembl_fingerprints": args.chembl_fp_parquet,
            "skin_known_target_panel": args.skin_known_cases_csv,
            "skin_known_run_ledger": skin_known_run_ledger_csv,
            "sota_baselines": sota_baselines_csv,
            "sota_ablations": sota_ablations_csv,
            "prospective_promotion": prospective_promotion_manifest,
            "prospective_metrics": prospective_metrics_csv,
            "candidate_preregistration": candidate_prereg_json,
            "candidate_results": candidate_results_csv,
            "candidate_selection": candidate_selection_json,
            "activity_retrieval_gate": args.activity_retrieval_gate,
            "direct_exact_reference": effective_direct_exact_reference,
            "direct_exact_manifest": effective_direct_exact_manifest,
            "training_sequence_db": args.training_seq_db,
            "training_ligands": args.training_ligands,
            "training_holo": args.training_holo,
        }),
        "artifact_snapshot": _artifact_snapshot(
            [args.eval_dir, args.rankings_dir, args.target_classes],
            args.out_manifest,
        ),
    }
    diagnostic_overrides = _diagnostic_overrides(args)
    claim_blockers = _claim_blockers(
        manifest,
        input_status,
        leakage_status,
        threshold_status,
        activity_retrieval_gate,
        input_runs_claim_blocker,
        sota_freeze_claim_blocker,
    )
    manifest["diagnostic_overrides"] = diagnostic_overrides
    manifest["claim_blockers"] = claim_blockers
    manifest["claim_ready"] = (
        not claim_blockers and not any(diagnostic_overrides.values())
    )
    manifest["release_governance"] = {
        "software_release_ready": manifest["n_steps"] > 0
        and manifest["n_failed"] == 0
        and input_status["n_incomplete"] == 0
        and not any(diagnostic_overrides.values()),
        "model_promotion_ready": bool(prospective_promotion["model_promotion_ready"])
        and activity_retrieval_gate["status"] == "pass"
        and (
            candidate_experiment["status"] == "not_run"
            or (
                candidate_experiment["status"] == "validated"
                and bool(candidate_experiment["eligible"])
            )
        ),
        "sota_claim_ready": manifest["claim_ready"],
    }
    _write_json_atomic(args.out_manifest, manifest)
    if input_status["n_incomplete"] and not args.allow_incomplete_eval_inputs:
        failed = ", ".join(
            item["name"] for item in input_status["incomplete"][:5]
        )
        raise SystemExit(
            "Incomplete evaluation input set(s) block claim-quality iteration; "
            f"pass --allow-incomplete-eval-inputs only for explicit diagnostics: {failed}"
        )
    if manifest["n_steps"] == 0 and not args.allow_empty_iteration:
        raise SystemExit(
            "No evaluation steps were runnable; provide eval inputs or pass "
            "--allow-empty-iteration only for explicit diagnostics"
        )
    if manifest["n_steps"] > 0 and leakage_status.get("n_leak_flags"):
        raise SystemExit(
            "Detected leakage flags block claim-quality iterations; "
            f"status={leakage_status.get('status')}, "
            f"n_leak_flags={leakage_status.get('n_leak_flags')}"
        )
    if manifest["n_failed"]:
        raise SystemExit(f"{manifest['n_failed']} evaluation step(s) failed")
    if threshold_status["n_failed"] and not args.allow_threshold_failure:
        failed_items = [
            *threshold_status["failures"],
            *threshold_status["missing_required"],
        ]
        failed = ", ".join(item["path"] for item in failed_items[:5])
        raise SystemExit(
            "Evaluation metric threshold failure(s) block claim-quality iteration; "
            f"pass --allow-threshold-failure only for explicit diagnostics: {failed}"
        )
    if (
        manifest["n_steps"] > 0
        and not args.allow_incomplete_leakage
        and not _leakage_claim_ready(leakage_status)
    ):
        raise SystemExit(
            "Leakage audit is required for claim-quality iterations; "
            f"status={leakage_status.get('status')}, "
            "sealed_status="
            f"{leakage_status.get('sealed_audit', {}).get('status')}. "
            "Provide eval_targets.csv with complete leakage references or pass "
            "--allow-incomplete-leakage only for explicit diagnostics"
        )
    if (
        manifest["n_steps"] > 0
        and threshold_status["n_checked"] == 0
        and not args.allow_threshold_failure
    ):
        raise SystemExit(
            "No evaluation metric thresholds were checked; claim-quality "
            "iterations require at least one thresholded metric or "
            "--allow-threshold-failure only for explicit diagnostics"
        )
    if (
        manifest["n_steps"] > 0
        and not args.allow_missing_input_runs
        and input_runs_claim_blocker is not None
    ):
        raise SystemExit(
            "Collected run provenance is required for claim-quality iterations; "
            f"{input_runs_claim_blocker}. "
            "Run eval/collect_run_outputs.py first or pass "
            "--allow-missing-input-runs only for explicit diagnostics"
        )
    if (
        manifest["n_steps"] > 0
        and not any(diagnostic_overrides.values())
        and sota_freeze_claim_blocker is not None
    ):
        raise SystemExit(
            "SOTA freeze artifacts are required for claim-quality iterations; "
            f"{sota_freeze_claim_blocker}. "
            "Provide skin-known run ledger, baseline fairness, and ablation "
            "freeze tables or pass diagnostic overrides only for explicit "
            "diagnostics"
        )
    print(json.dumps({"manifest": str(args.out_manifest), **manifest}, indent=2))


if __name__ == "__main__":
    main()
