#!/usr/bin/env python3
"""stage11_repro_pack.py — Emit a reproducibility pack for the run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import logging
import math
import os
import platform
import subprocess
import sys
import tempfile
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

LOG = logging.getLogger("stage11.repro")

OUTPUT_FILES = (
    "tool_versions.lock",
    "config_hash.txt",
    "git_commit.txt",
    "artifact_manifest.json",
    "random_seeds.json",
    "runtime_log.txt",
)
REQUIRED_SOURCE_ARTIFACTS = (
    "09_report/index.html",
    "03_targets/mode_comprehensive/top50_4way_consensus.csv",
    "03_targets/ranked_targets_v3_with_efficacy.csv",
)
REQUIRED_NUMERIC_COLUMNS = {"final_score", "score"}


def _remove_outputs(out_dir: Path) -> None:
    for name in OUTPUT_FILES:
        try:
            (out_dir / name).unlink()
        except FileNotFoundError:
            pass


def _git_commit() -> str:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise SystemExit(
            "Git commit capture failed; reproducibility pack requires "
            "a recorded source commit"
        ) from exc
    if len(commit) != 40 or any(ch not in "0123456789abcdefABCDEF" for ch in commit):
        raise SystemExit(
            "Git commit capture produced an invalid commit SHA; "
            f"expected 40 hex characters, got {commit!r}"
        )
    return commit


def _pip_freeze() -> str:
    try:
        versions = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise SystemExit(
            "Tool version capture failed; reproducibility pack requires "
            "non-empty pip freeze output"
        ) from exc
    if not versions.strip():
        raise SystemExit(
            "Tool version capture produced no package records; "
            "reproducibility pack requires non-empty pip freeze output"
        )
    return versions


def _config_hash(config_path: Path) -> str:
    if not config_path.exists():
        raise SystemExit(f"Workflow config is required for reproducibility pack: {config_path}")
    return hashlib.sha256(config_path.read_bytes()).hexdigest()


def _env_seed(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        seed = int(raw)
    except ValueError as exc:
        raise SystemExit(
            f"Reproducibility seed {name} must be a non-negative integer: {raw!r}"
        ) from exc
    if seed < 0:
        raise SystemExit(
            f"Reproducibility seed {name} must be a non-negative integer: {raw!r}"
        )
    return seed


def _runtime_log() -> str:
    required = {
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "working_directory": str(Path.cwd()),
    }
    missing = [name for name, value in required.items() if not str(value).strip()]
    if missing:
        raise SystemExit(
            "Runtime metadata capture failed; missing required fields: "
            + ", ".join(missing)
        )
    fields = {
        **required,
        "cpu": platform.processor() or "unknown",
        "hostname": platform.node() or "unknown",
    }
    return "\n".join(f"{name}: {value}" for name, value in fields.items()) + "\n"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_table_source(path: Path, run_dir: Path) -> None:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        with path.open(newline="") as fh:
            reader = csv.DictReader(fh, delimiter=delimiter)
            rows = list(reader)
    except (csv.Error, UnicodeDecodeError, OSError) as exc:
        raise SystemExit(
            "Reproducibility pack source artifact failed to parse as a table: "
            f"{path.relative_to(run_dir)}: {exc}"
        ) from exc
    if not rows:
        raise SystemExit(
            "Reproducibility pack source artifact contains no data rows: "
            f"{path.relative_to(run_dir)}"
        )
    _validate_table_schema(path, run_dir, reader.fieldnames or [], rows)
    _validate_required_numeric_columns(path, run_dir, reader.fieldnames or [], rows)

    id_cols = [
        col
        for col in ("target_id", "id", "known", "target")
        if col in (reader.fieldnames or [])
    ]
    for col in id_cols:
        seen: set[str] = set()
        duplicates: list[str] = []
        blank_indexes: list[int] = []
        for idx, row in enumerate(rows):
            value = str(row.get(col, "")).strip()
            if not value:
                blank_indexes.append(idx)
                continue
            if value in seen and value not in duplicates:
                duplicates.append(value)
            seen.add(value)
        if blank_indexes:
            shown = ", ".join(str(idx) for idx in blank_indexes[:10])
            suffix = "..." if len(blank_indexes) > 10 else ""
            raise SystemExit(
                "Reproducibility pack source artifact column "
                f"'{col}' contains blank values at row index(es) "
                f"{shown}{suffix}: {path.relative_to(run_dir)}"
            )
        if duplicates:
            shown = ", ".join(duplicates[:10])
            suffix = "..." if len(duplicates) > 10 else ""
            raise SystemExit(
                "Reproducibility pack source artifact column "
                f"'{col}' contains duplicate values {shown}{suffix}: "
                f"{path.relative_to(run_dir)}"
            )

    for idx, row in enumerate(rows):
        for col, raw in row.items():
            value = str(raw or "").strip()
            if not value:
                continue
            try:
                numeric = float(value)
            except ValueError:
                continue
            if not math.isfinite(numeric):
                raise SystemExit(
                    "Reproducibility pack source artifact column "
                    f"'{col}' contains non-finite numeric value at row index "
                    f"{idx}: {path.relative_to(run_dir)}"
                )


def _validate_required_numeric_columns(
    path: Path,
    run_dir: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    rel = path.relative_to(run_dir)
    for col in sorted(REQUIRED_NUMERIC_COLUMNS & set(fieldnames)):
        for idx, row in enumerate(rows):
            value = str(row.get(col, "")).strip()
            if not value:
                raise SystemExit(
                    "Reproducibility pack source artifact column "
                    f"'{col}' must be numeric at row index {idx}: {rel}"
                )
            try:
                numeric = float(value)
            except ValueError as exc:
                raise SystemExit(
                    "Reproducibility pack source artifact column "
                    f"'{col}' must be numeric at row index {idx}: {rel}"
                ) from exc
            if not math.isfinite(numeric):
                raise SystemExit(
                    "Reproducibility pack source artifact column "
                    f"'{col}' contains non-finite numeric value at row index "
                    f"{idx}: {rel}"
                )


def _validate_table_schema(
    path: Path,
    run_dir: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    if path.name != "ranked_targets_v3_with_efficacy.csv":
        return
    rel = path.relative_to(run_dir)
    required = {"target_id", "final_score", "source_count", "sources", "efficacy_top1"}
    missing = sorted(required - set(fieldnames))
    if missing:
        raise SystemExit(
            "Reproducibility pack source artifact missing required "
            f"publication column(s) {missing}: {rel}"
        )
    for idx, row in enumerate(rows):
        source_count_raw = str(row.get("source_count", "")).strip()
        try:
            source_count = int(source_count_raw)
        except ValueError as exc:
            raise SystemExit(
                "Reproducibility pack source artifact column 'source_count' "
                f"must be an integer at row index {idx}: {rel}"
            ) from exc
        if source_count < 2:
            raise SystemExit(
                "Reproducibility pack source artifact column 'source_count' "
                f"must be >= 2 at row index {idx}: {rel}"
            )
        sources = str(row.get("sources", "")).strip()
        labels = [part.strip() for part in sources.split(";")]
        if any(label == "" for label in labels):
            raise SystemExit(
                "Reproducibility pack source artifact column 'sources' "
                f"contains empty labels at row index {idx}: {rel}"
            )
        duplicates = sorted({label for label in labels if labels.count(label) > 1})
        if duplicates:
            shown = ", ".join(duplicates[:10])
            suffix = "..." if len(duplicates) > 10 else ""
            raise SystemExit(
                "Reproducibility pack source artifact column 'sources' "
                f"contains duplicate labels {shown}{suffix} "
                f"at row index {idx}: {rel}"
            )
        if len(labels) != source_count:
            raise SystemExit(
                "Reproducibility pack source artifact source_count="
                f"{source_count} but sources lists {len(labels)} label(s) "
                f"at row index {idx}: {rel}"
            )
        efficacy = str(row.get("efficacy_top1", "")).strip()
        if not efficacy:
            raise SystemExit(
                "Reproducibility pack source artifact column 'efficacy_top1' "
                f"is blank at row index {idx}: {rel}"
            )


def _validate_json_source(path: Path, run_dir: Path) -> None:
    try:
        doc = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        raise SystemExit(
            "Reproducibility pack source artifact failed to parse as JSON: "
            f"{path.relative_to(run_dir)}: {exc}"
        ) from exc

    if isinstance(doc, list):
        if not doc:
            raise SystemExit(
                "Reproducibility pack JSON source artifact contains no items: "
                f"{path.relative_to(run_dir)}"
            )
        return
    if isinstance(doc, dict):
        if not doc:
            raise SystemExit(
                "Reproducibility pack JSON source artifact contains no keys: "
                f"{path.relative_to(run_dir)}"
            )
        return
    raise SystemExit(
        "Reproducibility pack JSON source artifact must be an object or list: "
        f"{path.relative_to(run_dir)}"
    )


def _validate_html_source(path: Path, run_dir: Path) -> None:
    try:
        text = path.read_text()
    except (UnicodeDecodeError, OSError) as exc:
        raise SystemExit(
            "Reproducibility pack HTML source artifact failed to read: "
            f"{path.relative_to(run_dir)}: {exc}"
        ) from exc
    lowered = text.lower()
    if "<html" not in lowered or "</html>" not in lowered:
        raise SystemExit(
            "Reproducibility pack HTML source artifact lacks complete HTML "
            f"structure: {path.relative_to(run_dir)}"
        )
    for marker in ("placeholder", "stale"):
        if marker in lowered:
            raise SystemExit(
                "Reproducibility pack HTML source artifact contains scaffold "
                f"marker '{marker}': {path.relative_to(run_dir)}"
            )


def _validate_required_source_artifact(path: Path, run_dir: Path) -> None:
    if path.suffix.lower() in {".csv", ".tsv"}:
        _validate_table_source(path, run_dir)
    elif path.suffix.lower() == ".json":
        _validate_json_source(path, run_dir)
    elif path.suffix.lower() in {".html", ".htm"}:
        _validate_html_source(path, run_dir)


def _validate_eval_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(
            "Reproducibility pack eval manifest is required: "
            f"{path}"
        )
    if path.stat().st_size == 0:
        raise SystemExit(
            "Reproducibility pack eval manifest is empty: "
            f"{path}"
        )
    try:
        doc = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        raise SystemExit(
            "Reproducibility pack eval manifest failed to parse as JSON: "
            f"{path}: {exc}"
        ) from exc

    validator_path = Path(__file__).with_name("stage11_make_figures.py")
    spec = importlib.util.spec_from_file_location(
        "stage11_make_figures_for_repro",
        validator_path,
    )
    if spec is None or spec.loader is None:
        raise SystemExit(
            "Reproducibility pack could not load eval manifest validator: "
            f"{validator_path}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        module._validate_iteration_manifest(doc, path)
    except ValueError as exc:
        raise SystemExit(
            "Reproducibility pack eval manifest is not claim-ready: "
            f"{exc}"
        ) from exc
    return {
        "relative_path": "EVAL:iteration_manifest.json",
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "source_git_commit": str(doc["provenance"]["git_commit"]).lower(),
        "source_config_sha256": str(doc["provenance"]["config_sha256"]).lower(),
        "source_run_dirs": [
            str(run["run_dir"])
            for run in doc["input_runs"]["runs"]
        ],
    }


def _eval_manifest_covers_run_dir(
    artifacts: list[dict[str, Any]],
    run_dir: Path,
) -> bool:
    expected = run_dir.expanduser().resolve(strict=False)
    for artifact in artifacts:
        if artifact.get("relative_path") != "EVAL:iteration_manifest.json":
            continue
        source_run_dirs = artifact.get("source_run_dirs")
        if not isinstance(source_run_dirs, list):
            return False
        for source_run_dir in source_run_dirs:
            if not isinstance(source_run_dir, str) or not source_run_dir.strip():
                continue
            candidate = Path(source_run_dir).expanduser().resolve(strict=False)
            if candidate == expected:
                return True
        return False
    return True


def _validate_report_packages_for_publication(
    *,
    run_dir: Path,
    fast_pointer: Path | None,
    physics_pointer: Path | None,
    require_physics: bool,
) -> dict[str, Any] | None:
    if fast_pointer is None and physics_pointer is None:
        return None
    if fast_pointer is None or physics_pointer is None:
        raise SystemExit(
            "Reproducibility pack requires both --fast-report-pointer and "
            "--physics-report-pointer"
        )
    try:
        return validate_report_packages(
            run_dir=run_dir,
            fast_pointer=fast_pointer,
            physics_pointer=physics_pointer,
            require_physics=require_physics,
        )
    except ReportContractError as exc:
        raise SystemExit(
            f"Reproducibility pack immutable report validation failed: {exc}"
        ) from exc


def _artifact_manifest(
    run_dir: Path,
    out_dir: Path,
    eval_manifest: Path | None = None,
) -> list[dict[str, Any]]:
    if not run_dir.exists() or not run_dir.is_dir():
        raise SystemExit(f"Run directory is required for reproducibility pack: {run_dir}")

    for rel in REQUIRED_SOURCE_ARTIFACTS:
        required = run_dir / rel
        if not required.exists():
            raise SystemExit(
                "Reproducibility pack required source artifact is missing: "
                f"{rel}"
            )
        if required.stat().st_size == 0:
            raise SystemExit(
                "Reproducibility pack source artifact is empty: "
                f"{rel}"
            )
        _validate_required_source_artifact(required, run_dir)

    excluded = out_dir.resolve()
    records: list[dict[str, Any]] = []
    seen_resolved_paths: dict[Path, Path] = {}
    for path in sorted(p for p in run_dir.rglob("*") if p.is_file()):
        resolved = path.resolve()
        if resolved == excluded or resolved.is_relative_to(excluded):
            continue
        previous = seen_resolved_paths.get(resolved)
        if previous is not None:
            raise SystemExit(
                "Reproducibility pack source artifacts resolve to the same "
                f"file: {previous.relative_to(run_dir)} and "
                f"{path.relative_to(run_dir)}"
            )
        seen_resolved_paths[resolved] = path
        size = path.stat().st_size
        if size == 0:
            raise SystemExit(
                "Reproducibility pack source artifact is empty: "
                f"{path.relative_to(run_dir)}"
            )
        if path.suffix.lower() == ".json":
            _validate_json_source(path, run_dir)
        elif path.suffix.lower() in {".html", ".htm"}:
            _validate_html_source(path, run_dir)
        records.append({
            "relative_path": str(path.relative_to(run_dir)),
            "path": str(path),
            "bytes": size,
            "sha256": _sha256(path),
        })
    if eval_manifest is not None:
        records.append(_validate_eval_manifest(eval_manifest))
    if not records:
        raise SystemExit(
            "Reproducibility pack requires at least one source artifact in "
            f"run directory: {run_dir}"
        )
    return records


def _metadata_record(name: str, text: str) -> dict[str, Any]:
    data = text.encode()
    return {
        "relative_path": name,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--config", default=Path("workflow/config.yaml"), type=Path)
    parser.add_argument("--eval-manifest", default=None, type=Path)
    parser.add_argument("--fast-report-pointer", default=None, type=Path)
    parser.add_argument("--physics-report-pointer", default=None, type=Path)
    parser.add_argument("--require-physics-report", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    _remove_outputs(args.out_dir)
    report_packages = _validate_report_packages_for_publication(
        run_dir=args.run_dir,
        fast_pointer=args.fast_report_pointer,
        physics_pointer=args.physics_report_pointer,
        require_physics=args.require_physics_report,
    )
    artifacts = _artifact_manifest(args.run_dir, args.out_dir, args.eval_manifest)
    if not _eval_manifest_covers_run_dir(artifacts, args.run_dir):
        raise SystemExit(
            "Reproducibility pack eval manifest input_runs must include "
            f"--run-dir: {args.run_dir}"
        )
    config_sha256 = _config_hash(args.config)
    git_commit = _git_commit()
    eval_git_commit = next(
        (
            str(artifact.get("source_git_commit", "")).strip().lower()
            for artifact in artifacts
            if artifact.get("relative_path") == "EVAL:iteration_manifest.json"
        ),
        "",
    )
    if eval_git_commit and git_commit.lower() != eval_git_commit:
        raise SystemExit(
            "Reproducibility pack git commit must match eval manifest "
            "provenance.git_commit; got "
            f"{git_commit} from git rev-parse HEAD but eval manifest records "
            f"{eval_git_commit}"
        )
    eval_config_sha256 = next(
        (
            str(artifact.get("source_config_sha256", "")).strip().lower()
            for artifact in artifacts
            if artifact.get("relative_path") == "EVAL:iteration_manifest.json"
        ),
        "",
    )
    if eval_config_sha256 and config_sha256.lower() != eval_config_sha256:
        raise SystemExit(
            "Reproducibility pack workflow config digest must match eval "
            "manifest provenance.config_sha256; got "
            f"{config_sha256} from {args.config} but eval manifest records "
            f"{eval_config_sha256}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    metadata_payloads = {
        "tool_versions.lock": _pip_freeze(),
        "config_hash.txt": config_sha256 + "\n",
        "git_commit.txt": git_commit + "\n",
        "random_seeds.json": json.dumps({
            "etkdg": _env_seed("ETKDG_SEED", 49242),  # 0xC05A
            "boltz": _env_seed("BOLTZ_SEED", 0),
            "reinvent": _env_seed("REINVENT_SEED", 0),
        }, indent=2) + "\n",
        "runtime_log.txt": _runtime_log(),
    }
    if args.eval_manifest is None:
        raise SystemExit(
            "Reproducibility pack eval manifest is required for "
            "claim-ready publication output; pass --eval-manifest"
        )
    payloads = {
        **metadata_payloads,
        "artifact_manifest.json": json.dumps({
            **({"report_packages": report_packages} if report_packages else {}),
            "run_dir": str(args.run_dir),
            "n_artifacts": len(artifacts),
            "artifacts": artifacts,
            "metadata_files": [
                _metadata_record(name, text)
                for name, text in sorted(metadata_payloads.items())
            ],
        }, indent=2) + "\n",
    }
    with tempfile.TemporaryDirectory(prefix=".stage11_repro_", dir=args.out_dir.parent) as tmp:
        stage_dir = Path(tmp)
        for name, text in payloads.items():
            (stage_dir / name).write_text(text)
        for name in OUTPUT_FILES:
            (stage_dir / name).replace(args.out_dir / name)
    LOG.info("Repro pack written to %s", args.out_dir)


if __name__ == "__main__":
    main()
